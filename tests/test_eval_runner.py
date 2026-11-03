"""The eval plane: equal budgets, raw rows, and columns that stay apart.

Three things are under test, in the order they can lie:

1. **Loading** — a suite file that is not a `DesignSpec` is a reported problem, never a
   silent skip, and both accepted layouts (bare spec, `{"spec": ...}` wrapper) load.
2. **Running** — every (case, policy) cell gets the same `--rounds`, its own directory,
   and a broken cell becomes a row with null outcome columns instead of aborting the
   suite. A suite that stops on case three is a comparison of one policy over two cases.
3. **Reporting** — `repaired_by_policy`, `environment_errors`, `unknown` and
   `harness_errors` are separate columns, and `manufacturing_validated: not evaluated`
   appears when no case declares a build orientation. The summary may never add tool
   health to a repair outcome.

The episode layer is injected (`run=`), so this file tests the accounting, not T1–T3.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from _design_fakes import design_spec
from pah.cli.main import main
from pah.eval import report as eval_report
from pah.eval import runner as eval_runner


# --------------------------------------------------------------------------- #
# Doubles
# --------------------------------------------------------------------------- #
def fake_report(
    *,
    verdict: str = "repaired",
    rounds: int = 2,
    repaired: int = 1,
    env: int = 0,
    unknown: int = 0,
    budget_rounds: int = 3,
    geom: int = 7,
    sim: int = 0,
    clearance: float = 0.30,
) -> dict:
    """The slice of a run report the eval runner reads, with explicit numbers."""
    return {
        "budget": {
            "rounds": {"budget": budget_rounds, "used": rounds, "left": 0},
            "geom_calls": {"charged": geom},
            "sim_calls": {"charged": sim},
            "verifier_spend": {"geom_calls": 2, "sim_calls": 1},
        },
        "counts": {
            "repaired_by_policy": repaired,
            "environment_errors": env,
            "unknown": unknown,
            "physical_failures_final": 0,
        },
        "episode": {"verdict": verdict, "rounds": rounds},
        "design": {"final_clearance_mm": clearance},
        "trace": {"path": "trace.jsonl"},
        "run": {"wall_s": 0.01},
    }


def suite_dir(tmp_path: Path, *, count: int = 2) -> Path:
    directory = tmp_path / "suite"
    directory.mkdir()
    for index in range(count):
        spec = design_spec(
            clearance_mm=0.0, spec_id=f"case-{index}"
        )
        (directory / f"case-{index}.json").write_text(json.dumps(spec.model_dump()), "utf-8")
    return directory


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def test_suite_loading_accepts_both_layouts_and_reports_bad_files(tmp_path):
    directory = tmp_path / "suite"
    directory.mkdir()
    spec = design_spec(spec_id="wrapped")
    (directory / "bare.json").write_text(json.dumps(spec.model_dump()), "utf-8")
    (directory / "wrapped.json").write_text(
        json.dumps({"spec": spec.model_dump(), "expect": "repaired"}), "utf-8"
    )
    (directory / "broken.json").write_text("{not json", "utf-8")

    cases, problems = eval_runner.load_suite(directory)

    assert sorted(case.name for case in cases) == ["bare", "wrapped"]
    assert [case.name for case in cases] == ["bare", "wrapped"]  # stable file order
    assert len(problems) == 1
    assert problems[0].name == "broken"
    assert "JSONDecodeError" in problems[0].error or "Expecting" in problems[0].error


def test_a_missing_suite_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="no such suite"):
        eval_runner.load_suite(tmp_path / "nope")


def test_action_space_is_reported_per_kind(tmp_path):
    cases, _ = eval_runner.load_suite(suite_dir(tmp_path, count=1))
    # T3's operator table is imported lazily; when present it must answer for the kind.
    if eval_runner.action_space("peg_hole") is None:
        assert cases[0].action_space is None
    else:
        assert cases[0].action_space == eval_runner.action_space("peg_hole")
        assert 0 < cases[0].action_space <= len(eval_runner.DESIGN_OPERATOR_IDS)


# --------------------------------------------------------------------------- #
# Running
# --------------------------------------------------------------------------- #
def test_every_cell_runs_with_the_same_budget_in_its_own_directory(tmp_path):
    cases, _ = eval_runner.load_suite(suite_dir(tmp_path))
    calls: list[tuple[str, str, Path, int]] = []

    def run(case, policy, out_dir, rounds):
        calls.append((case.name, policy, out_dir, rounds))
        (out_dir / "report.json").write_text("{}", "utf-8")
        return fake_report(rounds=rounds, budget_rounds=rounds)

    rows = eval_runner.run_suite(
        cases, policies=["rule", "random"], rounds=2, out=tmp_path / "eval", run=run
    )

    assert len(rows) == len(cases) * 2
    assert all(rounds == 2 for _, _, _, rounds in calls)
    assert len({out_dir for _, _, out_dir, _ in calls}) == len(calls)
    assert all(out_dir.is_dir() for _, _, out_dir, _ in calls)
    assert all(row["status"] == "completed" for row in rows)
    assert [row["policy"] for row in rows] == ["rule", "random", "rule", "random"]
    assert all(row["rounds_budget"] == 2 for row in rows)


def test_a_broken_cell_is_a_row_and_the_suite_continues(tmp_path):
    cases, _ = eval_runner.load_suite(suite_dir(tmp_path))

    def run(case, policy, out_dir, rounds):
        if policy == "random":
            raise RuntimeError("module has not landed")
        return fake_report(rounds=rounds)

    rows = eval_runner.run_suite(
        cases, policies=["rule", "random"], rounds=3, out=tmp_path / "eval", run=run
    )

    failed = [row for row in rows if row["status"] == "error"]
    assert len(failed) == len(cases)
    assert {row["policy"] for row in failed} == {"random"}
    assert all(row["repaired_by_policy"] is None for row in failed)
    assert all(row["rounds"] is None for row in failed)
    assert "RuntimeError: module has not landed" in failed[0]["error"]
    # The other policy's cells still ran: a harness failure must not erase them.
    assert sum(1 for row in rows if row["status"] == "completed") == len(cases)


def test_rerunning_a_cell_starts_from_a_clean_directory(tmp_path):
    cases, _ = eval_runner.load_suite(suite_dir(tmp_path, count=1))
    stale = tmp_path / "eval" / "runs" / f"{cases[0].name}__rule"
    stale.mkdir(parents=True)
    (stale / "leftover.txt").write_text("old run", "utf-8")

    eval_runner.run_suite(
        cases,
        policies=["rule"],
        rounds=1,
        out=tmp_path / "eval",
        run=lambda case, policy, out_dir, rounds: fake_report(rounds=1),
    )

    assert not (stale / "leftover.txt").exists()


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def test_outcome_and_tool_health_are_separate_columns():
    rows = [
        {
            "case": "a",
            "policy": "rule",
            "status": "completed",
            "verdict": "repaired",
            "repaired_by_policy": 1,
            "rounds": 2,
            "rounds_budget": 3,
            "geom_calls_charged": 6,
            "sim_calls_charged": 0,
            "verifier_geom_calls": 2,
            "verifier_sim_calls": 0,
            "environment_errors": 0,
            "unknown": 0,
            "physical_failures_final": 0,
            "kind": "peg_hole",
            "action_space": 7,
            "manufacturing_evaluated": False,
            "print_orientation_deg": None,
            "final_clearance_mm": 0.3,
            "trace": "a/rule/trace.jsonl",
            "report": "a/rule/report.json",
            "error": "",
        },
        {
            "case": "a",
            "policy": "random",
            "status": "completed",
            "verdict": "exhausted",
            "repaired_by_policy": 0,
            "rounds": 3,
            "rounds_budget": 3,
            "geom_calls_charged": 9,
            "sim_calls_charged": 1,
            "verifier_geom_calls": 3,
            "verifier_sim_calls": 1,
            "environment_errors": 2,
            "unknown": 1,
            "physical_failures_final": 1,
            "kind": "peg_hole",
            "action_space": 7,
            "manufacturing_evaluated": False,
            "print_orientation_deg": None,
            "final_clearance_mm": 0.1,
            "trace": "a/random/trace.jsonl",
            "report": "a/random/report.json",
            "error": "",
        },
        {
            "case": "b",
            "policy": "random",
            "status": "error",
            "verdict": None,
            "repaired_by_policy": None,
            "rounds": None,
            "rounds_budget": None,
            "geom_calls_charged": None,
            "sim_calls_charged": None,
            "verifier_geom_calls": None,
            "verifier_sim_calls": None,
            "environment_errors": None,
            "unknown": None,
            "physical_failures_final": None,
            "kind": "snap_fit",
            "action_space": 7,
            "manufacturing_evaluated": False,
            "print_orientation_deg": None,
            "final_clearance_mm": None,
            "trace": "b/random/trace.jsonl",
            "report": "b/random/report.json",
            "error": "RuntimeError: boom",
        },
    ]

    summary = eval_report.summary_markdown(
        rows, suite="fixtures/design", policies=["rule", "random"], rounds=3
    )

    # The four groups are named and never summed into one "success rate".
    for column in (
        "repaired_by_policy",
        "environment_errors",
        "unknown",
        "harness_errors",
    ):
        assert column in summary
    assert "success_rate" not in summary
    assert "100.0%" in summary  # rule: 1/1
    # random's only completed cell carries an environment error, so it leaves the
    # `repaired_by_policy` denominator: `n/a`, not `0/1 (0.0%)`. The exclusion is
    # named under the table rather than applied silently.
    assert "n/a" in summary
    assert "denominator excludes 1 environment-error cell(s)" in summary
    # The health flags sit on the cell, not in the outcome column.
    assert "exhausted (env=2, unknown=1)" in summary
    assert "harness_error" in summary
    # "Not evaluated" is stated, and is not spelled as "printable".
    assert "manufacturing_validated: not evaluated (no print orientation declared)" in summary


def test_a_declared_print_orientation_is_reported_as_evaluated():
    rows = [
        {
            "case": "a",
            "policy": "rule",
            "status": "completed",
            "verdict": "repaired",
            "repaired_by_policy": 1,
            "rounds": 1,
            "rounds_budget": 1,
            "geom_calls_charged": 1,
            "sim_calls_charged": 0,
            "verifier_geom_calls": 1,
            "verifier_sim_calls": 0,
            "environment_errors": 0,
            "unknown": 0,
            "physical_failures_final": 0,
            "kind": "peg_hole",
            "action_space": 7,
            "manufacturing_evaluated": True,
            "print_orientation_deg": [0.0, 0.0, 0.0],
            "final_clearance_mm": 0.3,
            "trace": "trace.jsonl",
            "report": "report.json",
            "error": "",
        }
    ]

    summary = eval_report.summary_markdown(
        rows, suite="s", policies=["rule"], rounds=1
    )
    assert "evaluated for 1/1 cells" in summary


def test_results_jsonl_round_trips(tmp_path):
    rows = [{"case": "a", "policy": "rule", "status": "completed"}, {"case": "a", "policy": "random"}]
    path = eval_report.write_results(tmp_path / "results.jsonl", rows)

    lines = [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]
    assert lines[0] == {"schema": eval_report.RESULTS_SCHEMA}
    assert lines[1:] == rows


# --------------------------------------------------------------------------- #
# pah eval (CLI)
# --------------------------------------------------------------------------- #
def test_cli_eval_writes_both_artefacts(monkeypatch, tmp_path):
    suite = suite_dir(tmp_path, count=2)

    def fake_run_suite(cases, *, policies, rounds, out, config_path=None, seed=0):
        out = Path(out)
        return [
            {
                "case": case.name,
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
                "environment_errors": 0,
                "unknown": 0,
                "physical_failures_final": 0,
                "kind": "peg_hole",
                "action_space": 7,
                "manufacturing_evaluated": False,
                "print_orientation_deg": None,
                "final_clearance_mm": 0.3,
                "trace": str(out / "trace.jsonl"),
                "report": str(out / "report.json"),
                "error": "",
            }
            for case in cases
            for policy in policies
        ]

    monkeypatch.setattr(eval_runner, "run_suite", fake_run_suite)
    out = tmp_path / "eval"

    rc = main(
        [
            "eval",
            "--suite",
            str(suite),
            "--policies",
            "rule,random",
            "--rounds",
            "2",
            "--out",
            str(out),
        ]
    )

    assert rc == 0
    assert (out / "results.jsonl").exists()
    assert (out / "summary.md").exists()
    summary = (out / "summary.md").read_text("utf-8")
    assert "`rule`" in summary and "`random`" in summary
    assert "manufacturing_validated: not evaluated" in summary


def test_cli_eval_exits_1_when_a_cell_failed(monkeypatch, tmp_path):
    suite = suite_dir(tmp_path, count=1)
    monkeypatch.setattr(
        eval_runner,
        "run_suite",
        lambda cases, **kwargs: [
            {
                "case": cases[0].name,
                "policy": "rule",
                "status": "error",
                "verdict": None,
                "repaired_by_policy": None,
                "rounds": None,
                "rounds_budget": None,
                "geom_calls_charged": None,
                "sim_calls_charged": None,
                "verifier_geom_calls": None,
                "verifier_sim_calls": None,
                "environment_errors": None,
                "unknown": None,
                "physical_failures_final": None,
                "kind": "peg_hole",
                "action_space": 7,
                "manufacturing_evaluated": False,
                "print_orientation_deg": None,
                "final_clearance_mm": None,
                "trace": "t",
                "report": "r",
                "error": "RuntimeError: boom",
            }
        ],
    )

    rc = main(["eval", "--suite", str(suite), "--policies", "rule", "--out", str(tmp_path / "e")])
    assert rc == 1


def test_cli_eval_missing_suite_is_exit_3(tmp_path, capsys):
    rc = main(["eval", "--suite", str(tmp_path / "nope"), "--out", str(tmp_path / "e")])
    assert rc == 3
    assert "no such suite" in capsys.readouterr().out


def test_cli_eval_llm_without_a_provider_is_exit_3(tmp_path, monkeypatch, capsys):
    """Asking a model with no provider is a broken request, not eight error cells:
    refusing up front keeps the comparison table free of infrastructure noise."""
    monkeypatch.setenv("PAH_CONFIG", str(tmp_path / "absent.toml"))
    suite = suite_dir(tmp_path, count=1)

    rc = main(
        ["eval", "--suite", str(suite), "--policies", "rule,llm", "--out", str(tmp_path / "e")]
    )

    assert rc == 3
    assert "llm provider" in capsys.readouterr().out
    assert not (tmp_path / "e" / "summary.md").exists()


def test_parse_policies_drops_blanks_and_duplicates_but_keeps_order():
    assert main.__module__  # keep the import used; the parser itself is trivial
    from pah.cli.eval import parse_policies

    assert parse_policies("random, rule ,, rule") == ["random", "rule"]
    assert parse_policies("   ") == []
