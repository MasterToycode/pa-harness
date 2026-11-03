"""P3-3: the independent holdout, the separated report, and memory-vs-model columns.

What is pinned here
-------------------
* **The suite is real**: at least twenty parseable `DesignSpec`s in
  `fixtures/design/eval-holdout/`, with no `spec_id` shared with the development
  fixtures, and a mix of clean designs and deliberate defects (the README's claim).
* **The development suite does not absorb it**: `load_suite(fixtures/design)` reads
  one directory level only, so the four demos stay four and the holdout stays its own
  suite. This is the audit's central merge risk.
* **Memory hits and model decisions are distinct**: `RepairPlan.proposed_by` is
  counted into `memory_hits`/`model_decisions`; the summary has both columns and
  never adds them together.
* **The report separates suites**: two suites in one run produce two table blocks and
  one honesty line each; a small suite says 案例观察，不构成策略优劣的统计主张.
* **The manufacturing line reads the orientation source**: a `recommended` pose no
  longer prints "no print orientation declared".

Offline and fast: the episodes are injected (`run=` / patched `run_suite`), so no
verifier, materialiser or slicer is invoked.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pah.cli.eval import resolve_suites
from pah.cli.main import main
from pah.design.spec import DesignSpec
from pah.eval import report as eval_report
from pah.eval import runner as eval_runner

REPO = Path(__file__).resolve().parents[1]
HOLDOUT = REPO / "fixtures" / "design" / "eval-holdout"
DEV = REPO / "fixtures" / "design"


# --------------------------------------------------------------------------- #
# The suite itself
# --------------------------------------------------------------------------- #
def test_the_holdout_has_twenty_parseable_cases_unique_against_the_demos():
    files = sorted(HOLDOUT.glob("*.json"))
    assert len(files) >= 20, f"holdout needs >= 20 cases, found {len(files)}"

    holdout_ids: set[str] = set()
    for path in files:
        spec = DesignSpec.model_validate_json(path.read_text("utf-8"))
        holdout_ids.add(spec.spec_id)
    assert len(holdout_ids) == len(files), "duplicate spec_id inside the holdout"

    # Every development fixture, one directory level: no id may be reused. A holdout
    # case that also lives in the tuning set is not an independent case.
    dev_ids = {
        DesignSpec.model_validate_json(path.read_text("utf-8")).spec_id
        for path in sorted(DEV.glob("*.json"))
    }
    assert dev_ids, "the development fixtures are missing"
    assert holdout_ids.isdisjoint(dev_ids), holdout_ids & dev_ids


def test_the_holdout_is_a_mix_of_clean_and_defective_designs():
    cases = [
        DesignSpec.model_validate_json(path.read_text("utf-8"))
        for path in sorted(HOLDOUT.glob("*.json"))
    ]
    intents = [str(spec.provenance.get("intent", "")) for spec in cases]
    clean = sum(1 for intent in intents if intent == "clean")
    defective = len(intents) - clean
    # The README claims a mix; a suite of only clean (or only broken) designs would
    # measure one half of a policy's behaviour.
    assert clean >= 6, intents
    assert defective >= 6, intents
    assert {spec.interface.kind for spec in cases} == {"peg_hole", "snap_fit", "dovetail"}
    # Every case records how it was generated; an unexplained case cannot be audited.
    for spec in cases:
        assert spec.provenance.get("generator") == "scripts/make_holdout.py"
        assert spec.provenance.get("suite", "").startswith("eval-holdout")
    # Both orientation states appear, so the declared-vs-recommended columns are
    # exercised by the suite rather than by one hand-made fixture.
    declared = [spec for spec in cases if spec.constraints.print_orientation_deg is not None]
    assert 0 < len(declared) < len(cases)


def test_one_directory_is_one_suite():
    """The dev suite must not silently absorb the holdout or the stress fixtures."""
    dev_cases, dev_problems = eval_runner.load_suite(DEV)
    assert dev_problems == []
    assert {case.name for case in dev_cases} == {
        "demo-dovetail",
        "demo-peg-hole-clear",
        "demo-peg-hole-interfere",
        "demo-snap-fit",
    }
    assert all(case.suite == "design" for case in dev_cases)

    holdout_cases, _ = eval_runner.load_suite(HOLDOUT, label="holdout")
    assert len(holdout_cases) >= 20
    assert all(case.suite == "holdout" for case in holdout_cases)

    stress_cases, _ = eval_runner.load_suite(DEV / "stress", label="stress")
    assert stress_cases
    assert all(case.suite == "stress" for case in stress_cases)


def test_the_readme_states_the_suite_is_evaluation_only():
    readme = (HOLDOUT / "README.md").read_text("utf-8")
    assert "not for tuning" in readme.lower()
    assert str(eval_runner.__name__) or True  # keep the import meaningful
    assert "seed" in readme.lower()
    assert "pah eval --suite holdout" in readme


def test_a_single_file_suite_loads_one_case(tmp_path):
    """`--suite path/to/case.json` is how a quick check is run; the case id is the stem.

    Regression: `Path.relative_to(itself)` is `Path('.')`, whose name is empty, so a
    naive `with_suffix('')` raised before the suite ever ran.
    """
    path = sorted(HOLDOUT.glob("*.json"))[0]
    cases, problems = eval_runner.load_suite(path)

    assert problems == []
    assert len(cases) == 1
    assert cases[0].name == path.stem
    assert cases[0].suite == HOLDOUT.name


# --------------------------------------------------------------------------- #
# The success denominator: tool health is not a failed repair
# --------------------------------------------------------------------------- #
def summary_row(
    *,
    case: str,
    policy: str = "rule",
    suite: str = "holdout",
    status: str = "completed",
    verdict: str = "repaired",
    repaired: int = 1,
    env: int = 0,
    unknown: int = 0,
    memory: int = 0,
    model: int = 0,
    orientation_source: str = "recommended",
) -> dict:
    """One row shaped like `runner._row_from_report`'s output, for report-only tests."""
    return {
        "case": case,
        "suite": suite,
        "policy": policy,
        "status": status,
        "verdict": verdict,
        "repaired_by_policy": repaired,
        "rounds": 1,
        "rounds_budget": 3,
        "geom_calls_charged": 1,
        "sim_calls_charged": 0,
        "verifier_geom_calls": 1,
        "verifier_sim_calls": 0,
        "deterministic_generator": True,
        "budget_overrun_total": 0,
        "environment_errors": env,
        "unknown": unknown,
        "physical_failures_final": 0 if verdict == "repaired" else 1,
        "memory_hits": memory,
        "model_decisions": model,
        "proposed_by_counts": {"case": memory, "rule": 0, "llm": model, "human": 0, "other": 0},
        "kind": "peg_hole",
        "action_space": 7,
        "manufacturing_evaluated": orientation_source in {"declared", "recommended"},
        "orientation_source": orientation_source,
        "print_orientation_deg": [0.0, 0.0, 0.0] if orientation_source != "none" else None,
        "final_clearance_mm": 0.2,
        "trace": "t",
        "report": "r",
        "error": "",
    }


def test_environment_error_cells_leave_the_success_denominator():
    """A cell whose verifier could not run is unknown, not a failed repair."""
    rows = [
        summary_row(case="holdout-00", repaired=1, env=0),
        summary_row(case="holdout-01", verdict="exhausted", repaired=0, env=1, unknown=1),
    ]

    summary = eval_report.summary_markdown(
        rows, suite="holdout", policies=["rule"], rounds=3
    )
    block = summary.split("## Per policy")[1].split("## Per case")[0]

    # One repaired out of one *attributable* cell, not out of two.
    assert "1/1 (100.0%)" in block
    assert "1/2" not in block
    # The exclusion is named under the table; a silent denominator is its own bug.
    assert "denominator excludes 1 environment-error cell(s)" in block
    assert "`rule`: 1" in block
    # The health evidence is still visible on the cell and in the health column.
    assert "env=1" in summary


def test_a_suite_with_only_environment_errors_reports_na():
    """`0/0` must never render as `0%`: no attributable cell, no ratio."""
    rows = [
        summary_row(
            case="holdout-02", verdict="failed", repaired=0, env=2, unknown=1
        )
    ]

    summary = eval_report.summary_markdown(
        rows, suite="holdout", policies=["rule"], rounds=3
    )
    block = summary.split("## Per policy")[1].split("## Per case")[0]

    assert "n/a" in block
    assert "0/0" not in block
    assert "(0.0%)" not in block
    assert "denominator excludes 1 environment-error cell(s)" in block


def test_no_denominator_note_when_nothing_was_excluded():
    rows = [summary_row(case="holdout-03", repaired=1, env=0)]

    summary = eval_report.summary_markdown(
        rows, suite="holdout", policies=["rule"], rounds=3
    )

    assert "1/1 (100.0%)" in summary
    assert "denominator excludes" not in summary


# --------------------------------------------------------------------------- #
# Runner: suite labels and memory/model columns
# --------------------------------------------------------------------------- #
def fake_report(
    *,
    verdict: str = "repaired",
    rounds: int = 1,
    proposed_by: list[str] | None = None,
    orientation_source: str = "recommended",
    deterministic: bool = True,
) -> dict:
    """The slice of a run report the runner reads, with explicit decision sources."""
    sources = proposed_by if proposed_by is not None else ["rule"] * rounds
    return {
        "budget": {
            "rounds": {"budget": 3, "used": rounds, "left": 0},
            "geom_calls": {"charged": 4, "overrun": 0},
            "sim_calls": {"charged": 1, "overrun": 0},
            "verifier_spend": {"geom_calls": 3, "sim_calls": 1},
            "overrun_total": 0,
        },
        "counts": {
            "repaired_by_policy": 1 if verdict == "repaired" else 0,
            "environment_errors": 0,
            "unknown": 0,
            "physical_failures_final": 0,
        },
        "run": {"deterministic_generator": deterministic},
        "verdicts": {
            "printable": {
                "orientation": (
                    {"source": orientation_source, "angles_deg": [0.0, 0.0, 0.0]}
                    if orientation_source in {"declared", "recommended"}
                    else {"source": orientation_source}
                )
            }
        },
        "episode": {"verdict": verdict, "rounds": rounds},
        "rounds": [{"proposed_by": source} for source in sources],
        "design": {"final_clearance_mm": 0.25},
        "trace": {"path": "trace.jsonl"},
    }


def test_a_two_case_subset_records_the_suite_and_the_decision_sources(tmp_path):
    cases, problems = eval_runner.load_suite(HOLDOUT, label="holdout")
    assert problems == []
    subset = cases[:2]

    rows = eval_runner.run_suite(
        subset,
        policies=["rule"],
        rounds=2,
        out=tmp_path / "eval",
        run=lambda case, policy, out_dir, rounds: fake_report(
            rounds=2, proposed_by=["case", "rule"]
        ),
    )

    assert len(rows) == 2
    assert {row["suite"] for row in rows} == {"holdout"}
    for row in rows:
        assert row["memory_hits"] == 1  # one replay from the ledger
        assert row["model_decisions"] == 0  # no model decision in these cells
        assert row["proposed_by_counts"] == {"case": 1, "rule": 1, "llm": 0, "human": 0, "other": 0}
        assert row["status"] == "completed"
    assert rows[0]["report"].endswith("report.json")


def test_memory_and_model_decisions_are_counted_separately(tmp_path):
    cases, _ = eval_runner.load_suite(HOLDOUT, label="holdout")
    rows = eval_runner.run_suite(
        cases[:1],
        policies=["case_memory"],
        rounds=3,
        out=tmp_path / "eval",
        run=lambda case, policy, out_dir, rounds: fake_report(
            rounds=3, proposed_by=["case", "case", "llm"]
        ),
    )
    row = rows[0]
    assert row["memory_hits"] == 2
    assert row["model_decisions"] == 1


# --------------------------------------------------------------------------- #
# Report: separated suites, memory columns, honest statistics
# --------------------------------------------------------------------------- #
def test_the_summary_lists_memory_hits_and_keeps_suites_apart():
    rows = [
        {
            "case": "demo-peg-hole-clear",
            "suite": "dev",
            "policy": "case_memory",
            "status": "completed",
            "verdict": "repaired",
            "repaired_by_policy": 1,
            "rounds": 1,
            "rounds_budget": 3,
            "geom_calls_charged": 4,
            "sim_calls_charged": 1,
            "verifier_geom_calls": 3,
            "verifier_sim_calls": 1,
            "deterministic_generator": True,
            "budget_overrun_total": 0,
            "environment_errors": 0,
            "unknown": 0,
            "physical_failures_final": 0,
            "memory_hits": 1,
            "model_decisions": 0,
            "proposed_by_counts": {"case": 1, "rule": 0, "llm": 0, "human": 0, "other": 0},
            "kind": "peg_hole",
            "action_space": 7,
            "manufacturing_evaluated": True,
            "orientation_source": "recommended",
            "print_orientation_deg": [0.0, 0.0, 0.0],
            "final_clearance_mm": 0.25,
            "trace": "t",
            "report": "r",
            "error": "",
        },
        {
            "case": "holdout-peg-hole-clean-00",
            "suite": "holdout",
            "policy": "case_memory",
            "status": "completed",
            "verdict": "exhausted",
            "repaired_by_policy": 0,
            "rounds": 3,
            "rounds_budget": 3,
            "geom_calls_charged": 7,
            "sim_calls_charged": 2,
            "verifier_geom_calls": 5,
            "verifier_sim_calls": 2,
            "deterministic_generator": True,
            "budget_overrun_total": 0,
            "environment_errors": 0,
            "unknown": 0,
            "physical_failures_final": 1,
            "memory_hits": 0,
            "model_decisions": 2,
            "proposed_by_counts": {"case": 0, "rule": 1, "llm": 2, "human": 0, "other": 0},
            "kind": "peg_hole",
            "action_space": 7,
            "manufacturing_evaluated": True,
            "orientation_source": "recommended",
            "print_orientation_deg": [0.0, 0.0, 0.0],
            "final_clearance_mm": 0.1,
            "trace": "t",
            "report": "r",
            "error": "",
        },
    ]

    summary = eval_report.summary_markdown(
        rows, suite="dev,holdout", policies=["case_memory"], rounds=3
    )

    assert "- suites:" in summary
    assert "## Per policy — suite `dev`" in summary
    assert "## Per policy — suite `holdout`" in summary
    # The memory/model columns exist and are not collapsed into one.
    assert "| memory_hits | model_decisions |" in summary
    # Each suite's table only counts its own cells: 1/1 dev, 0/1 holdout.
    dev_block = summary.split("## Per policy — suite `dev`")[1].split("## Per")[0]
    holdout_block = summary.split("## Per policy — suite `holdout`")[1].split("## Per")[0]
    assert "1/1 (100.0%)" in dev_block
    assert "0/1 (0.0%)" in holdout_block
    # Decision-source flags are visible per cell (memory hit here, model there).
    assert "mem=1" in summary
    assert "llm=2" in summary
    # Two small suites: each says it is a case observation, with its own n.
    assert summary.count("案例观察，不构成策略优劣的统计主张") == 2
    assert "n=1 cases" in summary


def test_a_twenty_case_holdout_is_no_longer_called_a_case_observation():
    rows = [
        {
            "case": f"holdout-{index:02d}",
            "suite": "holdout",
            "policy": "rule",
            "status": "completed",
            "verdict": "repaired",
            "repaired_by_policy": 1,
            "rounds": 1,
            "rounds_budget": 3,
            "geom_calls_charged": 1,
            "sim_calls_charged": 0,
            "verifier_geom_calls": 1,
            "verifier_sim_calls": 0,
            "deterministic_generator": True,
            "budget_overrun_total": 0,
            "environment_errors": 0,
            "unknown": 0,
            "physical_failures_final": 0,
            "memory_hits": 0,
            "model_decisions": 0,
            "kind": "peg_hole",
            "action_space": 7,
            "manufacturing_evaluated": True,
            "orientation_source": "declared",
            "print_orientation_deg": [0.0, 0.0, 0.0],
            "final_clearance_mm": 0.2,
            "trace": "t",
            "report": "r",
            "error": "",
        }
        for index in range(20)
    ]
    summary = eval_report.summary_markdown(
        rows, suite="holdout", policies=["rule"], rounds=3
    )
    assert "案例观察，不构成策略优劣的统计主张" not in summary
    assert "n=20" in summary
    assert "confidence interval" in summary


def test_the_manufacturing_line_reads_the_orientation_source():
    """A recommended pose is evaluated; the old "no print orientation declared"
    sentence is a lie for it (audit P2-2 changed what an undeclared spec means)."""
    rows = [
        {"status": "completed", "manufacturing_evaluated": True, "orientation_source": "recommended"},
        {"status": "completed", "manufacturing_evaluated": True, "orientation_source": "declared"},
    ]
    line = eval_report._manufacturing_line(rows)
    assert "no print orientation declared" not in line
    assert "evaluated for 2/2 cells" in line
    assert "recommended: 1" in line
    assert "declared: 1" in line

    none_rows = [
        {"status": "completed", "manufacturing_evaluated": False, "orientation_source": "none"}
    ]
    line = eval_report._manufacturing_line(none_rows)
    assert "not evaluated" in line
    assert "no print orientation declared" not in line

    # Legacy rows (no source column) keep the old sentence: an archived summary and a
    # fresh one must not describe the same run in two different ways.
    legacy = [{"status": "completed", "manufacturing_evaluated": False}]
    assert "no print orientation declared" in eval_report._manufacturing_line(legacy)


# --------------------------------------------------------------------------- #
# The CLI
# --------------------------------------------------------------------------- #
def test_suite_aliases_resolve_to_the_repository_directories():
    resolved = dict(resolve_suites("dev,holdout"))
    assert resolved["dev"] == DEV
    assert resolved["holdout"] == HOLDOUT
    assert list(resolved) == ["dev", "holdout"]


def test_cli_runs_a_two_case_holdout_subset(monkeypatch, tmp_path, capsys):
    """`--suite holdout --cases 2` reaches the runner with two labelled cases and the
    summary names the suite."""
    captured: dict = {}

    def fake_run_suite(cases, *, policies, rounds, out, config_path=None, seed=0):
        captured["suites"] = sorted({case.suite for case in cases})
        captured["names"] = [case.name for case in cases]
        captured["rounds"] = rounds
        return [
            {
                "case": case.name,
                "suite": case.suite,
                "policy": policy,
                "status": "completed",
                "verdict": "repaired",
                "repaired_by_policy": 1,
                "rounds": 1,
                "rounds_budget": rounds,
                "geom_calls_charged": 1,
                "sim_calls_charged": 0,
                "verifier_geom_calls": 1,
                "verifier_sim_calls": 0,
                "deterministic_generator": True,
                "budget_overrun_total": 0,
                "environment_errors": 0,
                "unknown": 0,
                "physical_failures_final": 0,
                "memory_hits": 1,
                "model_decisions": 0,
                "kind": "peg_hole",
                "action_space": 7,
                "manufacturing_evaluated": True,
                "orientation_source": "recommended",
                "print_orientation_deg": [0.0, 0.0, 0.0],
                "final_clearance_mm": 0.2,
                "trace": "t",
                "report": "r",
                "error": "",
            }
            for case in cases
            for policy in policies
        ]

    monkeypatch.setattr(eval_runner, "run_suite", fake_run_suite)
    out = tmp_path / "eval"

    code = main(
        [
            "eval",
            "--suite",
            "holdout",
            "--cases",
            "2",
            "--policies",
            "rule",
            "--rounds",
            "2",
            "--out",
            str(out),
        ]
    )

    assert code == 0
    assert captured["suites"] == ["holdout"]
    assert len(captured["names"]) == 2
    assert captured["rounds"] == 2
    printed = capsys.readouterr().out
    assert "suite `holdout` — 2 case(s)" in printed
    summary = (out / "summary.md").read_text("utf-8")
    assert "- suite: `holdout` (2 cases)" in summary
    assert "memory_hits" in summary
    assert "案例观察" in summary  # two cases cannot support a ranking
    rows = [
        json.loads(line)
        for line in (out / "results.jsonl").read_text("utf-8").splitlines()
        if line.strip()
    ]
    assert rows[0]["schema"] == eval_report.RESULTS_SCHEMA
    assert {row["suite"] for row in rows[1:]} == {"holdout"}


def test_cli_rejects_zero_cases(tmp_path, capsys):
    code = main(["eval", "--suite", "holdout", "--cases", "0", "--out", str(tmp_path / "e")])
    assert code == 3
    assert "--cases" in capsys.readouterr().out
