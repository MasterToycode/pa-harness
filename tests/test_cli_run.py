"""`pah run` / `pah design new` / `pah design sanity` — the CLI's own acceptance.

Two layers, on purpose:

* **contract tests with the doubles from `tests/_design_fakes.py`** — they pin the
  behaviour the CLI owns: exit codes, the artefact set in the run directory, exactly
  N rounds for `--rounds N`, a resumed episode whose `replay_hashes` equal an
  uninterrupted one's, and an injected verifier fault that costs one round instead of
  the episode. They keep passing while T1–T3 are still being written.
* **integration tests against the real modules**, guarded by `importorskip` — one
  episode on a shipped fixture, one `design sanity`, and the two acceptance commands
  from the task verbatim. They are skipped, not faked, when a teammate's module is
  absent: a skipped real test says "not yet", a mocked one says "fine" when it is not.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from _design_fakes import design_spec, install
from pah.cli.main import main
from pah.design.spec import DesignSpec

REPO = Path(__file__).resolve().parents[1]


def write_spec(
    path: Path, *, clearance_mm: float = 0.0, band: tuple[float, float] = (0.20, 0.60)
) -> Path:
    """Writes a failing-by-default spec where the CLI will read it."""
    spec = design_spec(clearance_mm=clearance_mm, band=band, spec_id=f"cli-{path.stem}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec.model_dump()), "utf-8")
    return path


def read_report(out: Path) -> dict:
    return json.loads((out / "report.json").read_text("utf-8"))


# --------------------------------------------------------------------------- #
# pah design new
# --------------------------------------------------------------------------- #
def test_design_new_writes_a_provenanced_spec(tmp_path, capsys):
    out = tmp_path / "design"
    rc = main(
        [
            "design",
            "new",
            "--requirement",
            "a peg that seats a bracket",
            "--kind",
            "peg_hole",
            "--out",
            str(out),
            "--no-llm",
        ]
    )
    assert rc == 0
    spec = DesignSpec.model_validate_json((out / "spec.json").read_text("utf-8"))
    assert spec.interface.kind == "peg_hole"
    assert spec.clearance_mm == pytest.approx(0.10)
    # Who picked the numbers must be recoverable: that is the only provenance a
    # report cannot reconstruct later.
    assert spec.provenance["params"].startswith("deterministic-defaults")
    assert spec.provenance["parts"] == "deterministic-defaults"
    assert "wrote" in capsys.readouterr().out


@pytest.mark.parametrize("kind,clearance", [("snap_fit", 0.10), ("dovetail", 0.10)])
def test_design_new_defaults_are_inside_the_fit_band(tmp_path, kind, clearance):
    """A generator that hands the loop a broken design would make every demo look
    like a policy failure."""
    out = tmp_path / kind
    assert (
        main(["design", "new", "--kind", kind, "--out", str(out), "--no-llm"]) == 0
    )
    spec = DesignSpec.model_validate_json((out / "spec.json").read_text("utf-8"))
    assert spec.interface.kind == kind
    assert spec.clearance_mm == pytest.approx(clearance)
    assert spec.constraints.in_band(spec.clearance_mm)


def test_design_new_with_an_unknown_kind_is_a_usage_error(tmp_path):
    with pytest.raises(SystemExit) as exit_info:
        main(["design", "new", "--kind", "weld", "--out", str(tmp_path / "x")])
    assert exit_info.value.code == 2  # argparse's own usage code


# --------------------------------------------------------------------------- #
# pah run — artefact set and the repair story
# --------------------------------------------------------------------------- #
def test_run_repairs_and_writes_every_artifact(monkeypatch, tmp_path):
    install(monkeypatch)
    spec_path = write_spec(tmp_path / "spec.json")
    out = tmp_path / "ep0"

    rc = main(
        ["run", "--spec", str(spec_path), "--policy", "rule", "--rounds", "3", "--out", str(out)]
    )

    assert rc == 0
    report = read_report(out)
    assert report["episode"]["verdict"] == "repaired"
    assert report["rounds"], "a repair must leave at least one round"
    assert report["rounds"][-1]["verdict"] == "pass"
    assert report["rounds"][-1]["operator"] == "set_clearance"
    # 口径隔离: a design-plane report never carries a dataset number.
    assert report["plane"] == "design"
    assert report["dataset_runs_touched"] == []
    # The outcome column and the health columns are separate keys, never a sum.
    assert report["counts"]["repaired_by_policy"] == 1
    assert report["counts"]["environment_errors"] == 0
    assert report["counts"]["unknown"] == 0
    assert report["geometry"]["complete"] is True
    assert sorted(path.name for path in (out / "geometry").glob("*.stl")) == [
        "body.stl",
        "insert.stl",
    ]
    for artifact in (
        "spec.json",
        "spec-initial.json",
        "trace.jsonl",
        "report.json",
        "round-0/spec.json",
    ):
        assert (out / artifact).exists(), artifact
    assert len(report["trace"]["replay_hashes"]) >= 1


def test_the_round_budget_stops_at_exactly_three(monkeypatch, tmp_path):
    """`--rounds 3` spends three rounds and stops, whatever else is left over."""
    install(monkeypatch, mode="always_fail")
    spec_path = write_spec(tmp_path / "spec.json")
    out = tmp_path / "ep0"

    rc = main(
        ["run", "--spec", str(spec_path), "--policy", "rule", "--rounds", "3", "--out", str(out)]
    )

    assert rc == 0
    report = read_report(out)
    assert report["episode"]["verdict"] == "exhausted"
    assert report["episode"]["reason"] == "budget spent"
    assert len(report["rounds"]) == 3
    assert [row["index"] for row in report["rounds"]] == [0, 1, 2]
    assert report["budget"]["rounds"] == {"budget": 3, "used": 3, "left": 0, "overrun": 0}
    hashes = report["trace"]["replay_hashes"]
    assert len(hashes) == 3 and len(set(hashes)) == 3


def test_an_injected_verifier_fault_costs_one_round_not_the_episode(monkeypatch, tmp_path):
    install(monkeypatch)
    spec_path = write_spec(tmp_path / "spec.json")
    out = tmp_path / "ep0"

    rc = main(
        [
            "run",
            "--spec",
            str(spec_path),
            "--policy",
            "rule",
            "--rounds",
            "3",
            "--out",
            str(out),
            "--inject-fault",
            "verifier",
        ]
    )

    assert rc == 0
    report = read_report(out)
    assert report["rounds"][0]["verdict"] == "error"
    assert report["counts"]["error_rounds"] == [0]
    assert report["counts"]["environment_errors"] == 1
    # The episode continued after the fault: a broken tool costs a round, not the run.
    assert report["episode"]["verdict"] == "repaired"
    kinds = [
        json.loads(line)["kind"]
        for line in (out / "trace.jsonl").read_text("utf-8").splitlines()
        if line.strip()
    ]
    assert "verifier.error" in kinds
    assert kinds[-1] == "ticket.end"


# --------------------------------------------------------------------------- #
# pah run — reproducibility, interruption, resume
# --------------------------------------------------------------------------- #
def test_two_runs_of_the_same_spec_produce_the_same_hash_sequence(monkeypatch, tmp_path):
    install(monkeypatch)
    spec_path = write_spec(tmp_path / "spec.json")
    first, second = tmp_path / "a", tmp_path / "b"

    for out in (first, second):
        assert main(
            ["run", "--spec", str(spec_path), "--policy", "rule", "--rounds", "3", "--out", str(out)]
        ) == 0

    assert read_report(first)["trace"]["replay_hashes"] == read_report(second)["trace"][
        "replay_hashes"
    ]


def test_stop_then_resume_reproduces_the_hash_sequence(monkeypatch, tmp_path):
    """The acceptance test for a harness: the store, not the process, holds progress."""
    install(monkeypatch)
    spec_path = write_spec(tmp_path / "spec.json")
    baseline_out, interrupted = tmp_path / "baseline", tmp_path / "interrupted"

    assert main(
        [
            "run",
            "--spec",
            str(spec_path),
            "--policy",
            "rule",
            "--rounds",
            "3",
            "--out",
            str(baseline_out),
        ]
    ) == 0
    baseline = read_report(baseline_out)["trace"]["replay_hashes"]

    assert main(
        [
            "run",
            "--spec",
            str(spec_path),
            "--policy",
            "rule",
            "--rounds",
            "3",
            "--out",
            str(interrupted),
            "--stop-after-round",
            "1",
        ]
    ) == 0
    stopped = read_report(interrupted)
    assert stopped["run"]["status"] == "stopped"

    assert main(
        [
            "run",
            "--spec",
            str(spec_path),
            "--policy",
            "rule",
            "--rounds",
            "3",
            "--out",
            str(interrupted),
            "--resume",
        ]
    ) == 0
    resumed = read_report(interrupted)

    assert resumed["run"]["resume"] is True
    assert resumed["run"]["status"] == "completed"
    assert resumed["episode"]["verdict"] == "repaired"
    assert resumed["trace"]["replay_hashes"] == baseline
    assert any(note.startswith("resumed from round") for note in resumed["notes"])


def test_resume_without_a_recorded_run_is_a_clean_refusal(monkeypatch, tmp_path):
    install(monkeypatch)
    spec_path = write_spec(tmp_path / "spec.json")
    out = tmp_path / "empty"
    out.mkdir()

    rc = main(
        ["run", "--spec", str(spec_path), "--policy", "rule", "--out", str(out), "--resume"]
    )

    assert rc == 3
    assert not (out / "report.json").exists()


def test_a_second_run_into_the_same_directory_is_refused(monkeypatch, tmp_path, capsys):
    install(monkeypatch)
    spec_path = write_spec(tmp_path / "spec.json")
    out = tmp_path / "ep0"
    args = ["run", "--spec", str(spec_path), "--policy", "rule", "--rounds", "1", "--out", str(out)]

    assert main(args) == 0
    capsys.readouterr()
    assert main(args) == 3
    assert "already holds run" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# pah run — the three refusals that must be one sentence, never a traceback
# --------------------------------------------------------------------------- #
def test_a_missing_spec_is_exit_3_and_writes_nothing(tmp_path, capsys):
    out = tmp_path / "ep0"
    rc = main(["run", "--spec", str(tmp_path / "nope.json"), "--out", str(out)])
    assert rc == 3
    assert "no spec" in capsys.readouterr().out
    assert not out.exists()


def test_llm_without_a_provider_is_exit_3(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PAH_CONFIG", str(tmp_path / "absent.toml"))
    spec_path = write_spec(tmp_path / "spec.json")
    out = tmp_path / "ep0"

    rc = main(
        ["run", "--spec", str(spec_path), "--policy", "llm", "--rounds", "1", "--out", str(out)]
    )

    assert rc == 3
    assert "llm provider" in capsys.readouterr().out
    assert not (out / "report.json").exists()


def test_a_missing_teammate_module_is_exit_3_with_a_sentence(monkeypatch, tmp_path, capsys):
    """The runtime must fail as one sentence, not as an `ImportError` traceback."""
    from pah.app import runtime as runtime_module

    def missing() -> None:
        raise RuntimeError(
            "the geometry materialiser (T1) has not landed yet: module "
            "'pah.design.materialize' is absent."
        )

    monkeypatch.setattr(runtime_module, "resolve_materializer", missing)
    spec_path = write_spec(tmp_path / "spec.json")
    out = tmp_path / "ep0"

    rc = main(["run", "--spec", str(spec_path), "--out", str(out)])

    assert rc == 3
    assert "has not landed yet" in capsys.readouterr().out
    assert not (out / "report.json").exists()


# --------------------------------------------------------------------------- #
# pah design sanity — the real gate (skipped while T2 is absent)
# --------------------------------------------------------------------------- #
def test_design_sanity_runs_the_real_registry(capsys):
    pytest.importorskip("pah.tools.verifiers.design_cascade")
    rc = main(["design", "sanity"])
    output = capsys.readouterr().out
    assert rc == 0
    assert "design verifier(s)" in output
    assert "PASS" in output


# --------------------------------------------------------------------------- #
# Integration: the acceptance commands, against the real modules
# --------------------------------------------------------------------------- #
def test_the_acceptance_demo_commands_run_end_to_end(tmp_path):
    """`pah design new` then `pah run`, exactly as the task states them."""
    pytest.importorskip("pah.design.materialize")
    pytest.importorskip("pah.tools.verifiers.design_cascade")
    pytest.importorskip("pah.policy.design_policies")
    demo = tmp_path / "demo"

    assert main(["design", "new", "--kind", "peg_hole", "--out", str(demo), "--no-llm"]) == 0
    assert (
        main(
            [
                "run",
                "--spec",
                str(demo / "spec.json"),
                "--policy",
                "rule",
                "--rounds",
                "3",
                "--out",
                str(demo / "ep0"),
            ]
        )
        == 0
    )

    report = read_report(demo / "ep0")
    assert report["plane"] == "design"
    assert (demo / "ep0" / "trace.jsonl").exists()
    assert report["trace"]["rows"] > 0
    assert report["geometry"]["complete"] is True


def test_a_shipped_fixture_runs_against_the_real_cascade(tmp_path):
    """The 口径 check on real output: design numbers only, never a dataset figure."""
    pytest.importorskip("pah.design.materialize")
    pytest.importorskip("pah.tools.verifiers.design_cascade")
    pytest.importorskip("pah.policy.design_policies")
    fixture = REPO / "fixtures" / "design" / "demo-peg-hole-interfere.json"
    if not fixture.exists():
        pytest.skip("T1/T2 have not shipped fixtures/design yet")

    out = tmp_path / "ep0"
    assert main(
        ["run", "--spec", str(fixture), "--policy", "rule", "--rounds", "1", "--out", str(out)]
    ) == 0

    report = read_report(out)
    assert report["episode"]["verdict"] in ("repaired", "exhausted", "infeasible", "failed")
    assert report["dataset_runs_touched"] == []
    assert report["design"]["spec_id"] == "demo-peg-hole-interfere"
    assert report["trace"]["replay_hashes"], "a real round must leave a geometry hash"

def test_an_unknown_policy_name_is_exit_3(monkeypatch, tmp_path, capsys):
    """A typo must name what is available, including plugin policies — otherwise the
    extension point is invisible exactly when someone needs it."""
    install(monkeypatch)
    from pah.app import runtime as runtime_module

    def unknown(name: str, **_kwargs) -> None:
        raise ValueError(f"unknown policy {name!r}; the design plane ships ('rule',)")

    monkeypatch.setattr(runtime_module, "resolve_policy", unknown)
    spec_path = write_spec(tmp_path / "spec.json")

    rc = main(
        ["run", "--spec", str(spec_path), "--policy", "typo", "--rounds", "1", "--out", str(tmp_path / "ep")]
    )

    assert rc == 3
    output = capsys.readouterr().out
    assert "unknown policy" in output
    assert "Policy plugins loaded" in output


def test_a_policy_plugin_can_be_selected_by_name(tmp_path, monkeypatch):
    """`--policy case_memory` works because the runtime resolves plugin names before
    asking T3: the extension point has to be reachable from the command line."""
    pytest.importorskip("pah.policy.case_memory")
    pytest.importorskip("pah.design.materialize")
    pytest.importorskip("pah.tools.verifiers.design_cascade")
    pytest.importorskip("pah.policy.design_policies")
    monkeypatch.setenv("PAH_CASE_MEMORY", str(tmp_path / "memory.sqlite"))
    demo = tmp_path / "demo"
    assert main(["design", "new", "--kind", "peg_hole", "--out", str(demo), "--no-llm"]) == 0

    out = demo / "ep0"
    rc = main(
        [
            "run",
            "--spec",
            str(demo / "spec.json"),
            "--policy",
            "case_memory",
            "--rounds",
            "1",
            "--out",
            str(out),
        ]
    )

    assert rc == 0
    report = read_report(out)
    assert report["run"]["policy"] == "case_memory"
    assert report["geometry"]["complete"] is True
