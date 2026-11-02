"""The runtime's durability contract: budget, interruption, resume, clear refusals.

These are the properties that separate a harness from a script, and they belong to
the composition root rather than the CLI: the CLI only forwards flags, while the
runtime owns the ticket, the round directories and the report. Testing at this layer
also lets the tests assert on the store's state between two episodes — an interrupted
ticket must still be open, a resumed report must say where it resumed from.

All doubles come from `tests/_design_fakes.py`; the real stack gets one guarded
integration test at the end so the doubles cannot drift away from the contract
without a visible failure.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from _design_fakes import ClearancePolicy, design_spec, install
from pah.app import runtime as runtime_module
from pah.app.runtime import RuntimeConfig
from pah.app.store import Store
from pah.core.engine import Engine
from pah.core.model import Cost, RepairPlan

REPO = Path(__file__).resolve().parents[1]


def run_episode(
    out: Path,
    spec,
    *,
    rounds: int = 3,
    stop_after_round: int | None = None,
    resume: bool = False,
    policy: str = "rule",
) -> dict:
    """One composed episode in `out`, with the runtime closed afterwards."""
    config = RuntimeConfig(
        out=out,
        policy=policy,
        rounds=rounds,
        stop_after_round=stop_after_round,
        resume=resume,
    )
    with runtime_module.build(config) as composed:
        return composed.run_episode(
            spec, policy_name=policy, rounds=rounds, resume=resume
        )


# --------------------------------------------------------------------------- #
# Budget
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rounds", [1, 2, 3])
def test_the_budget_stops_at_exactly_the_requested_rounds(monkeypatch, tmp_path, rounds):
    install(monkeypatch, mode="always_fail")
    report = run_episode(tmp_path / "ep", design_spec(clearance_mm=0.0), rounds=rounds)

    assert report["episode"]["verdict"] == "exhausted"
    assert report["episode"]["reason"] == "budget spent"
    assert len(report["rounds"]) == rounds
    assert report["budget"]["rounds"] == {"budget": rounds, "used": rounds, "left": 0, "overrun": 0}
    assert report["ticket"]["outcome"] == "exhausted"


def test_a_zero_round_budget_is_refused_before_anything_runs(monkeypatch, tmp_path):
    install(monkeypatch)
    with pytest.raises(ValueError, match="rounds must be >= 1"):
        runtime_module.budget_for(0)


# --------------------------------------------------------------------------- #
# Interruption and resume
# --------------------------------------------------------------------------- #
def test_stop_then_resume_reproduces_the_uninterrupted_hash_sequence(monkeypatch, tmp_path):
    install(monkeypatch)
    spec = design_spec(clearance_mm=0.0)

    baseline = run_episode(tmp_path / "baseline", spec)
    assert baseline["episode"]["verdict"] == "repaired"
    hashes = baseline["trace"]["replay_hashes"]
    assert len(hashes) == 2  # 0.0 -> 0.1 (fail) -> 0.2 (pass)

    interrupted = run_episode(tmp_path / "interrupted", spec, stop_after_round=1)
    assert interrupted["run"]["status"] == "stopped"
    assert interrupted["ticket"]["outcome"] is None  # still open, waiting to resume
    assert interrupted["trace"]["replay_hashes"] == hashes[:1]

    resumed = run_episode(tmp_path / "interrupted", spec, resume=True)
    assert resumed["run"]["status"] == "completed"
    assert resumed["run"]["resume"] is True
    assert resumed["episode"]["verdict"] == "repaired"
    assert resumed["trace"]["replay_hashes"] == hashes
    assert resumed["ticket"]["outcome"] == "repaired"
    assert any(note.startswith("resumed from round") for note in resumed["notes"])


def test_a_resume_keeps_the_budget_the_ticket_was_opened_with(monkeypatch, tmp_path):
    """`--rounds` is a mandate recorded at open time; a resume may not widen it
    silently, or two halves of one episode would answer to different ceilings."""
    install(monkeypatch, mode="always_fail")
    spec = design_spec(clearance_mm=0.0)

    first = run_episode(tmp_path / "ep", spec, rounds=3, stop_after_round=1)
    assert first["budget"]["rounds"]["budget"] == 3

    resumed = run_episode(tmp_path / "ep", spec, rounds=99, resume=True)
    assert resumed["budget"]["rounds"] == {"budget": 3, "used": 3, "left": 0, "overrun": 0}
    assert any("keeps the recorded budget" in note for note in resumed["notes"])


def test_a_round_interrupted_mid_flight_is_closed_as_aborted_on_resume(monkeypatch, tmp_path):
    """A killed process leaves a round with no verdict. The resume must not pretend it
    never happened (it spent budget) and must not rerun it (it may have written state)."""
    install(monkeypatch)
    spec = design_spec(clearance_mm=0.0)
    out = tmp_path / "ep"

    partial = run_episode(out, spec, stop_after_round=1)
    assert len(partial["rounds"]) == 1

    # Simulate the kill: a round that was begun and then lost with the process.
    store = Store(out / "db.sqlite")
    try:
        engine = Engine(store)
        ticket_id = store.kv.get(runtime_module.RUN_MARKER)
        parts = tuple(part.id for part in spec.parts)
        engine.begin_round(
            ticket_id,
            RepairPlan(
                id="killed-mid-round",
                operator="set_clearance",
                targets=parts,
                params={"clearance_mm": 0.15, "parts": parts},
                cost_est=Cost(geom_calls=1),
                proposed_by="rule",
                reason="the process died here",
            ),
        )
    finally:
        store.close()

    resumed = run_episode(out, spec, resume=True)

    aborted = [row for row in resumed["rounds"] if row["verdict"] == "aborted"]
    assert len(aborted) == 1
    assert any("as 'aborted'" in note for note in resumed["notes"])
    assert resumed["counts"]["unknown"] >= 1
    assert resumed["episode"]["verdict"] == "repaired"


# --------------------------------------------------------------------------- #
# Refusals
# --------------------------------------------------------------------------- #
def test_resuming_a_finished_episode_is_refused(monkeypatch, tmp_path):
    install(monkeypatch)
    spec = design_spec(clearance_mm=0.0)
    run_episode(tmp_path / "ep", spec)

    with pytest.raises(RuntimeError, match="already ended"):
        run_episode(tmp_path / "ep", spec, resume=True)


def test_resuming_an_unstarted_directory_is_refused(monkeypatch, tmp_path):
    install(monkeypatch)
    # A resume builds the runtime (which opens the store) and then refuses; the
    # refusal has to name the missing run, not crash on a missing table.
    with pytest.raises(RuntimeError, match="nothing to resume"):
        run_episode(tmp_path / "never-ran", design_spec(), resume=True)


def test_the_runtime_refuses_to_write_into_another_directory(monkeypatch, tmp_path):
    install(monkeypatch)
    config = RuntimeConfig(out=tmp_path / "bound")
    with runtime_module.build(config) as composed:
        with pytest.raises(RuntimeError, match="bound to"):
            composed.run_episode(design_spec(), out_dir=tmp_path / "elsewhere")


def test_a_missing_module_is_a_runtime_error_not_an_import_error():
    with pytest.raises(RuntimeError) as error:
        runtime_module._require("pah.a_module_that_does_not_exist", "the thing (TX)")
    message = str(error.value)
    assert "has not landed yet" in message
    assert "pah.a_module_that_does_not_exist" in message
    assert "TX" in message


# --------------------------------------------------------------------------- #
# The report is the deliverable, including for a stopped episode
# --------------------------------------------------------------------------- #
def test_a_stopped_episode_still_leaves_a_complete_report(monkeypatch, tmp_path):
    install(monkeypatch)
    out = tmp_path / "ep"
    report = run_episode(out, design_spec(clearance_mm=0.0), stop_after_round=1)

    assert (out / "spec.json").exists()
    assert (out / "spec-initial.json").exists()
    assert (out / "trace.jsonl").exists()
    assert (out / "report.json").read_text("utf-8") == json.dumps(
        json.loads((out / "report.json").read_text("utf-8")), ensure_ascii=False, indent=2
    ) + "\n"
    assert report["geometry"]["complete"] is True
    assert report["run"]["status"] == "stopped"
    assert report["episode"]["verdict"] == "stopped"
    assert report["counts"]["repaired_by_policy"] == 0  # not repaired, not a failure
    assert report["dataset_runs_touched"] == []


# --------------------------------------------------------------------------- #
# Integration: the same properties against T1/T2/T3
# --------------------------------------------------------------------------- #
def test_resume_against_the_real_stack(tmp_path):
    """Budget + resume with the real materialiser and cascade.

    The case is built here rather than taken from `fixtures/design`: it asks for a
    `min_wall_mm` no shipped operator can reach, so the episode is guaranteed to spend
    its whole round budget instead of repairing in one round — which is what a resume
    test needs. Guarded rather than mocked: if a teammate's module is missing the test
    skips, and if the real cascade's `spent` meter, materialisation contract or the
    loop's `seen` parameter changes, this is where it shows up.
    """
    pytest.importorskip("pah.design.materialize")
    pytest.importorskip("pah.tools.verifiers.design_cascade")
    pytest.importorskip("pah.policy.design_policies")
    from pah.cli.design import default_spec
    from pah.design.spec import FitConstraints

    spec = default_spec(kind="peg_hole", spec_id="resume-probe").edit(
        constraints=FitConstraints(
            min_clearance_mm=0.10, max_clearance_mm=0.60, min_wall_mm=29.0
        )
    )

    baseline = run_episode(tmp_path / "baseline", spec, rounds=2)
    assert baseline["episode"]["verdict"] == "exhausted"
    assert len(baseline["trace"]["replay_hashes"]) == 2

    interrupted = run_episode(
        tmp_path / "interrupted", spec, rounds=2, stop_after_round=1
    )
    assert interrupted["run"]["status"] == "stopped"
    assert len(interrupted["trace"]["replay_hashes"]) == 1
    opened = Store(tmp_path / "interrupted" / "db.sqlite")
    try:
        assert opened.kv.get(runtime_module.RUN_MARKER) is not None
    finally:
        opened.close()

    resumed = run_episode(tmp_path / "interrupted", spec, rounds=2, resume=True)
    assert resumed["trace"]["replay_hashes"] == baseline["trace"]["replay_hashes"]
    assert resumed["ticket"]["outcome"] == baseline["ticket"]["outcome"] == "exhausted"
