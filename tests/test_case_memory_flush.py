"""The case-memory flush: a *successful* episode has to reach the ledger.

The bug this file pins (measured 2026-11-02, 48 eval cells): the ledger held 17
rows and every one said `fail`. `CaseMemoryPolicy.flush()` only ever ran from
`propose()`, and the loop stops the moment a repair passes — so the one round
that succeeded was never handed to a `propose` call, `candidates()` requires
`passes > 0`, and the memory could never replay. The mechanism was dead while
looking like a policy that simply never hit.

What is asserted, in the order the mechanism can fail:

1. one passing episode writes an `outcome="pass"` row (without the end-of-episode
   flush, zero rows are written);
2. the second episode on the same failure class **replays from the ledger**
   (`proposed_by == "case"`, `memory_hits == 1`) and closes in one round;
3. a failing episode still records only `fail`, and the ledger still offers
   nothing — a success is the only thing that becomes a candidate;
4. `flush` is idempotent and callable on its own;
5. a broken bookkeeping hook costs a report note, not the episode.

Two pieces of shared state this file is careful about. The ledger path is
redirected with `$PAH_CASE_MEMORY` in an autouse fixture: the process default is
`runs/case_memory.sqlite`, which is *the* ledger the next eval reads, and a test
that appends to it changes a number somebody else reports. And the policy under
test is the one the plugin system builds (`case_memory` is a builtin plugin), so
episode construction goes through the real registration path rather than injecting
a double that the runtime would never choose.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from _design_fakes import design_spec, install  # noqa: E402  (pytest puts tests/ on sys.path)
from pah.app import runtime as runtime_module
from pah.app.runtime import RuntimeConfig
from pah.core.model import Budget, FailureCertificate, RepairPlan, Round
from pah.design.api import assembly_from_spec
from pah.policy.case_memory import CaseMemory, CaseMemoryPolicy
from pah.policy.design_ops import register_builtin
from pah.policy.design_policies import RulePolicy
from pah.policy.registry import OperatorRegistry

POLICY = "case_memory"
BUDGET = Budget(rounds=3, geom_calls=100, sim_calls=20)


# --------------------------------------------------------------------------- #
# Harness
# --------------------------------------------------------------------------- #
def registry() -> OperatorRegistry:
    ops = OperatorRegistry()
    register_builtin(ops)
    return ops


@pytest.fixture(autouse=True)
def isolated_ledger(tmp_path, monkeypatch):
    """Point the plugin's ledger at `tmp_path`; never at `runs/case_memory.sqlite`."""
    path = tmp_path / "case_memory.sqlite"
    monkeypatch.setenv("PAH_CASE_MEMORY", str(path))
    return path


def ledger(path: Path) -> CaseMemory:
    return CaseMemory(path)


def certificate(clearance_mm: float = 0.0) -> FailureCertificate:
    return FailureCertificate(
        code="insufficient_clearance",
        family="geometry",
        locus={"interface": "iface_0"},
        magnitude={"clearance_mm": clearance_mm},
        observed_by="fake_fit",
    )


def run_episode(
    monkeypatch,
    tmp_path: Path,
    *,
    name: str = "ep",
    rounds: int = 3,
    mode: str = "band",
    policy_name: str = POLICY,
    policy: object | None = None,
) -> dict:
    """One composed episode: real operators, fake cascade, plugin-built policy.

    `policy=None` is the faithful path — `policy_for("case_memory")` finds the
    builtin plugin's factory and builds the real policy around a real rule inner
    policy. A test that injects its own instance has to use a name no plugin claims,
    because `Runtime.policy_for` asks the plugin host *before* it asks T3's builder.
    """
    ops = registry()
    install(monkeypatch, mode=mode, policies={policy_name: policy} if policy else None)
    # After `install`, so the real action space replaces the single-operator fake: the
    # ledger validates against real parameter models and the bridge applies for real.
    monkeypatch.setattr(runtime_module, "resolve_operators", lambda: ops)
    cfg = RuntimeConfig(out=tmp_path / name, policy=policy_name, rounds=rounds)
    with runtime_module.build(cfg) as composed:
        return composed.run_episode(
            design_spec(clearance_mm=0.0), policy_name=policy_name, rounds=rounds
        )


def outcomes(memory: CaseMemory) -> list[str]:
    return [str(row["outcome"]) for row in memory.rows()]


def memory_hits(report: dict) -> int:
    """The eval's `memory_hits` column: rounds a case-memory replay produced."""
    return sum(1 for row in report["rounds"] if row.get("proposed_by") == "case")


# --------------------------------------------------------------------------- #
# 1. A success reaches the ledger
# --------------------------------------------------------------------------- #
def test_a_passing_episode_writes_a_pass_row(monkeypatch, tmp_path, isolated_ledger):
    """The regression this file exists for.

    The loop stops after the round that passed, so no later `propose` can flush it;
    before the end-of-episode flush this test would find an empty ledger.
    """
    report = run_episode(monkeypatch, tmp_path, name="pass")
    memory = ledger(isolated_ledger)

    assert report["episode"]["verdict"] == "repaired"
    assert report["rounds"][0]["proposed_by"] == "rule"  # the inner policy decided
    assert outcomes(memory) == ["pass"]
    row = memory.rows()[0]
    assert row["operator"] == "set_clearance"
    assert row["cert_codes"] == "insufficient_clearance"
    assert row["kind"] == "peg_hole"
    # The ledger must offer it back, which is the whole point of recording it.
    assert [
        record.operator
        for record in memory.candidates(
            kind="peg_hole", cert_codes="insufficient_clearance"
        )
    ] == ["set_clearance"]


def test_the_flush_is_reported_as_bookkeeping_not_as_the_outcome(monkeypatch, tmp_path):
    """The report says the ledger caught up; the verdict still comes from the verifiers."""
    report = run_episode(monkeypatch, tmp_path, name="note")

    assert report["episode"]["verdict"] == "repaired"
    assert any("bookkeeping" in note for note in report["notes"]), report["notes"]


# --------------------------------------------------------------------------- #
# 2. The next episode replays from the ledger
# --------------------------------------------------------------------------- #
def test_the_second_episode_replays_the_remembered_success(monkeypatch, tmp_path, isolated_ledger):
    """Same failure class, same build: round one comes from memory, not from the rule."""
    first = run_episode(monkeypatch, tmp_path, name="first")
    second = run_episode(monkeypatch, tmp_path, name="second")
    memory = ledger(isolated_ledger)

    assert first["rounds"][0]["proposed_by"] == "rule"
    assert memory_hits(first) == 0
    assert second["episode"]["verdict"] == "repaired"
    assert second["episode"]["rounds"] == 1
    assert second["rounds"][0]["proposed_by"] == "case"
    assert memory_hits(second) == 1
    # And the replay is credited in the ledger too: two passes, still no failures.
    assert outcomes(memory) == ["pass", "pass"]


def test_the_replayed_plan_is_the_remembered_number(monkeypatch, tmp_path):
    run_episode(monkeypatch, tmp_path, name="learn")

    report = run_episode(monkeypatch, tmp_path, name="replay")

    params = report["rounds"][0]["params"]
    assert report["rounds"][0]["operator"] == "set_clearance"
    assert params["clearance_mm"] == pytest.approx(0.40)  # the band middle the rule chose


# --------------------------------------------------------------------------- #
# 3. A failure is recorded as a failure, and buys nothing
# --------------------------------------------------------------------------- #
def test_a_failing_episode_records_only_failures(monkeypatch, tmp_path, isolated_ledger):
    """`always_fail` exhausts the budget; every round it produced is a failed attempt,
    and none of them may become a candidate."""
    report = run_episode(
        monkeypatch, tmp_path, name="fail", rounds=2, mode="always_fail"
    )
    memory = ledger(isolated_ledger)

    assert report["episode"]["verdict"] == "exhausted"
    assert len(report["rounds"]) == 2
    assert outcomes(memory) == ["fail", "fail"]
    assert memory.candidates(kind="peg_hole", cert_codes="insufficient_clearance") == []


# --------------------------------------------------------------------------- #
# 4. The flush is idempotent and callable on its own
# --------------------------------------------------------------------------- #
def _closed_round(plan: RepairPlan, verdict: str = "pass", index: int = 0) -> Round:
    return Round(id=f"r{index}", index=index, plan=plan, verdict=verdict)


def test_flush_is_idempotent_and_standalone():
    """One propose-time memory, then the same round flushed twice: one row."""
    ops = registry()
    memory = CaseMemory(":memory:")
    policy = CaseMemoryPolicy(inner=RulePolicy(registry=ops), memory=memory, registry=ops)
    assembly = assembly_from_spec(spec=design_spec(clearance_mm=0.0), version=1)

    plan = policy.propose(assembly, [certificate()], [], BUDGET)
    assert plan is not None
    closed = _closed_round(plan)

    assert policy.flush([closed]) == 1
    assert policy.flush([closed]) == 0  # the same round, already written
    assert policy.flush() == 0  # and nothing pending is left
    assert outcomes(memory) == ["pass"]


def test_flush_accepts_a_ticket_or_a_round_sequence():
    ops = registry()
    memory = CaseMemory(":memory:")
    policy = CaseMemoryPolicy(inner=RulePolicy(registry=ops), memory=memory, registry=ops)
    assembly = assembly_from_spec(spec=design_spec(clearance_mm=0.0), version=1)
    plan = policy.propose(assembly, [certificate()], [], BUDGET)
    assert plan is not None

    class TicketLike:
        rounds = (_closed_round(plan),)

    assert policy.flush(TicketLike()) == 1
    with pytest.raises(TypeError):
        policy.flush("not a ticket")


def test_an_open_round_stays_pending():
    """A round with no verdict yet must not be written as a failure."""
    ops = registry()
    memory = CaseMemory(":memory:")
    policy = CaseMemoryPolicy(inner=RulePolicy(registry=ops), memory=memory, registry=ops)
    assembly = assembly_from_spec(spec=design_spec(clearance_mm=0.0), version=1)
    plan = policy.propose(assembly, [certificate()], [], BUDGET)
    assert plan is not None
    open_round = Round(id="r0", index=0, plan=plan, verdict=None)

    assert policy.flush([open_round]) == 0
    assert memory.rows() == []
    assert policy.flush([_closed_round(plan)]) == 1  # the verdict arrived later


# --------------------------------------------------------------------------- #
# 5. A broken hook cannot break the episode
# --------------------------------------------------------------------------- #
class BrokenFlush(CaseMemoryPolicy):
    """A ledger whose write fails: the episode must still complete and report.

    It fails on its *second* call, which is the runtime's end-of-episode one. Failing
    on the first would crash `propose` — a policy bug, and not what this test is
    about: the claim is that the runtime's bookkeeping call is guarded.
    """

    def flush(self, ticket_or_rounds: object = None) -> int:
        self.flush_calls = getattr(self, "flush_calls", 0) + 1
        if self.flush_calls > 1:
            raise RuntimeError("disk on fire")
        return super().flush(ticket_or_rounds)


def test_a_failing_flush_costs_a_note_not_the_episode(monkeypatch, tmp_path, isolated_ledger):
    policy = BrokenFlush(
        inner=RulePolicy(registry=registry()),
        memory=CaseMemory(":memory:"),
        registry=registry(),
    )

    report = run_episode(
        monkeypatch, tmp_path, name="broken", policy_name="broken", policy=policy
    )

    assert report["episode"]["verdict"] == "repaired"
    assert report["counts"]["repaired_by_policy"] == 1
    assert any("bookkeeping failed" in note for note in report["notes"]), report["notes"]
    assert "disk on fire" in " ".join(report["notes"])
    assert ledger(isolated_ledger).rows() == []  # nothing was recorded, and nothing broke


def test_a_policy_without_flush_is_untouched(monkeypatch, tmp_path):
    """`rule`/`random`/`llm` keep no ledger; the hook is looked up, not required.

    `RulePolicy` has no `flush` at all, so this is the "attribute absent" path —
    distinct from `BrokenFlush` above, where it exists and fails.
    """
    report = run_episode(
        monkeypatch,
        tmp_path,
        name="plain",
        policy_name="plain",
        policy=RulePolicy(registry=registry()),
    )

    assert not hasattr(RulePolicy(registry=registry()), "flush")
    assert report["episode"]["verdict"] == "repaired"
    assert not any("bookkeeping" in note for note in report["notes"])

