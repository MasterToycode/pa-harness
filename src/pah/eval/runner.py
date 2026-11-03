"""Equal-budget evaluation: every (case, policy) pair gets the same episode budget.

Why a runner instead of a loop in the CLI
-----------------------------------------
The comparison the project needs is "does policy A repair more designs than policy B
*without spending more*". That is only a fair question if every cell of the table is
run under identical conditions: same case, same `--rounds`, a fresh run directory,
and the same accounting. Those are four invariants, so they live in one function that
a test can pin — not in an argparse handler that grows a flag per experiment.

What is measured, and what is deliberately not
-----------------------------------------------
* `repaired_by_policy` — did the policy's episode reach a verified pass. This is a
  **design-plane outcome**, never a dataset yield; `runs/m0-connector` and
  `runs/m1-min` are frozen evidence with their own numbers and different directories
  (`docs/internal/repair-loop-spec.md` §6). No row here carries a dataset figure.
* `environment_errors`, `unknown` — tool health and unresolvable state, counted in
  their own columns. They are *not* subtracted from a "success rate", because a
  broken verifier is not a policy failure and a silent merge of the two is exactly
  the reporting bug the contract's §7.6 forbids.
* `harness_errors` — a cell that could not run at all (a missing module, a bad spec).
  Kept apart again: those rows have no outcome, so averaging them as zero would
  understate every policy equally and hide the breakage.

A failed cell does not abort the suite. A ten-case comparison that stops on case
three is a comparison of one policy over two cases; the row records the exception
text and the next cell runs.

Result of a legal no-op
-----------------------
`run_suite` returns plain dicts (one per cell) and writes nothing. Persisting is
`eval/report.py`'s job, so the same rows can be printed, diffed or handed to a test
without touching a file.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from pah.app.runtime import RuntimeConfig, build
from pah.design.api import DESIGN_OPERATOR_IDS
from pah.design.spec import DesignSpec

__all__ = [
    "CaseProblem",
    "RunFn",
    "SuiteCase",
    "action_space",
    "default_run",
    "load_suite",
    "run_suite",
]


@dataclass(frozen=True)
class SuiteCase:
    """One design under test: its file name, the spec, and the file it came from."""

    name: str
    spec: DesignSpec
    path: Path
    #: The suite this case belongs to (`dev`, `holdout`, a stress directory). Carried
    #: on the case because the report must never merge a tuning set and an evaluation
    #: set into one success rate; a row that lost its suite could only be merged.
    suite: str = ""
    #: How many of the seven operators this design's connector kind can express.
    #: `design_ops.expressible` owns the answer; `None` means T3 is not importable
    #: yet and the report says "unknown" instead of inventing a number.
    action_space: int | None = None


@dataclass(frozen=True)
class CaseProblem:
    """A suite file that is not a usable design. Reported, never silently skipped."""

    name: str
    path: str
    error: str


#: `(case, policy, out_dir, rounds) -> report dict`. Injected so the runner and the
#: report format can be tested without T1–T3 or a real episode.
RunFn = Callable[[SuiteCase, str, Path, int], dict[str, Any]]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_suite(
    suite: str | Path, *, label: str | None = None
) -> tuple[list[SuiteCase], list[CaseProblem]]:
    """Every `*.json` **in one directory level** of `suite`, plus the files that failed.

    Why not recursive any more: a holdout suite lives at
    `fixtures/design/eval-holdout/`, and a recursive load of the development suite
    would have silently folded the evaluation cases into the tuning set — the exact
    merge the audit (P3-3) is about. One directory is one suite; a nested suite is
    named explicitly (`--suite holdout`, or its path).

    Two layouts are accepted on purpose. A file may be the spec itself (what
    `pah design new` writes) or wrap it as `{"spec": {...}}` (what a fixture that also
    carries an expected verdict looks like). Guessing further — inferring a kind from
    a mesh, defaulting a missing constraint — would mean the suite silently changes
    what a run was given, which is the failure mode this function exists to avoid.
    """
    path = Path(suite)
    if not path.exists():
        raise FileNotFoundError(f"no such suite: {path}")
    files = [path] if path.is_file() else sorted(path.glob("*.json"))
    suite_label = label or (path.name if path.is_dir() else path.parent.name)
    cases: list[SuiteCase] = []
    problems: list[CaseProblem] = []
    for file in files:
        name = _case_name(path, file)
        try:
            payload = json.loads(file.read_text("utf-8"))
            if isinstance(payload, dict) and "spec" in payload:
                payload = payload["spec"]
            spec = DesignSpec.model_validate(payload)
        except Exception as error:  # noqa: BLE001 - a bad file is a row, not a crash
            problems.append(
                CaseProblem(name=name, path=str(file), error=f"{type(error).__name__}: {error}")
            )
            continue
        cases.append(
            SuiteCase(
                name=name,
                spec=spec,
                path=file,
                suite=suite_label,
                action_space=action_space(spec.interface.kind),
            )
        )
    return cases, problems


def _case_name(root: Path, file: Path) -> str:
    """A stable, filesystem-safe case id: the path under the suite without `.json`.

    A suite may be a directory or one file (`--suite path/to/case.json`). For a single
    file `relative_to` yields `Path('.')`, whose name is empty, so the file's stem is
    the only sensible id and is handled before the relative-path branch.
    """
    if file.is_file() and file == root:
        return file.stem
    try:
        relative = file.relative_to(root)
    except ValueError:  # `suite` was a file and `file` is not resolvable under it
        return file.stem
    if not relative.name:
        return file.stem
    parts = list(relative.with_suffix("").parts)
    return ".".join(parts)


def action_space(kind: str) -> int | None:
    """How many operators `kind` can express, or `None` when T3 is not importable.

    Reported next to the outcome because an equal-budget comparison is unreadable
    without it: a connector kind that only three of seven operators can touch gives
    its policy fewer chances, and that is an explanation for a lower repair rate —
    not a worse policy.
    """
    try:
        from pah.policy import design_ops  # noqa: PLC0415
    except Exception:  # noqa: BLE001 - a missing T3 must not break evaluation loading
        return None
    expressible = getattr(design_ops, "expressible", None)
    if not callable(expressible):
        return None
    try:
        return sum(1 for operator_id in DESIGN_OPERATOR_IDS if expressible(kind, operator_id))
    except Exception:  # noqa: BLE001 - a kind T3 does not know is "unknown", not zero
        return None


# --------------------------------------------------------------------------- #
# Running
# --------------------------------------------------------------------------- #
def default_run(
    case: SuiteCase,
    policy: str,
    out_dir: Path,
    rounds: int,
    *,
    config_path: str | Path | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """One real episode in `out_dir`: a fresh runtime per cell, so nothing leaks.

    A runtime holds a store and a trace bound to one directory, so a shared instance
    across cells would append two cases to one trace. One runtime per cell is also
    what makes a failed cell's directory readable on its own.
    """
    composed = build(
        RuntimeConfig(
            out=Path(out_dir),
            config_path=Path(config_path) if config_path else None,
            seed=seed,
            policy=policy,
            rounds=rounds,
        )
    )
    try:
        return composed.run_episode(
            case.spec, policy_name=policy, rounds=rounds, out_dir=Path(out_dir), resume=False
        )
    finally:
        composed.close()


def run_suite(
    cases: Sequence[SuiteCase],
    *,
    policies: Iterable[str],
    rounds: int,
    out: str | Path,
    run: RunFn | None = None,
    config_path: str | Path | None = None,
    seed: int = 0,
    clean: bool = True,
) -> list[dict[str, Any]]:
    """Run every (case, policy) cell and return one row per cell, in a fixed order.

    Cells are keyed `<case>__<policy>` under `<out>/runs/`, which is what the report
    links to. `clean=True` wipes a cell's directory before running it — the
    alternative is a rerun that silently resumes an old episode and reports its
    rounds as this run's.
    """
    out = Path(out)
    runner = run or partial(default_run, config_path=config_path, seed=seed)
    names = [case.name for case in cases]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        # Cell directories are keyed by case name, and two suites may legitimately have
        # a `case-01`. Running both into one output directory would let the second cell
        # overwrite the first's trace, so the collision is refused by name.
        raise ValueError(
            f"case names appear more than once across the suites being run: {duplicates}"
        )
    rows: list[dict[str, Any]] = []
    for case in cases:
        for policy in policies:
            run_dir = out / "runs" / f"{case.name}__{policy}"
            if clean and run_dir.exists():
                shutil.rmtree(run_dir)
            run_dir.mkdir(parents=True, exist_ok=True)
            started = time.monotonic()
            try:
                report = runner(case, policy, run_dir, rounds)
                row = _row_from_report(case, policy, report, run_dir)
            except Exception as error:  # noqa: BLE001 - a broken cell is a row, not a crash
                row = _error_row(case, policy, run_dir, error)
            row["wall_s"] = round(time.monotonic() - started, 3)
            rows.append(row)
    return rows


def _manufacturing(report: dict[str, Any] | None) -> tuple[bool, list[float] | None]:
    """Did a manufacturing check run, and in which orientation? `None` = not evaluated.

    Read off the run's own report (`verdicts.printable.orientation`) rather than off
    the suite case's spec. Since the orientation search (audit P2-2) an undeclared
    spec is still evaluated — in a *recommended* pose — so the old "was a pose
    declared?" test would print `manufacturing_validated: not evaluated` for runs that
    did evaluate, which is the same overclaim in the opposite direction.

    The three states stay apart: `declared` and `recommended` both mean an overhang
    rule ran, and the angles come back with them so a reader can see which pose;
    `none` (a process with no overhang rule, or `--no-verify`), `unknown` (a cell
    whose report could not be read) and a missing report are all "not evaluated".
    """
    block = ((report or {}).get("verdicts") or {}).get("printable") or {}
    orientation = block.get("orientation") or {}
    if orientation.get("source") not in {"declared", "recommended"}:
        return False, None
    angles = orientation.get("angles_deg")
    if angles is None:
        return False, None
    return True, [float(value) for value in angles]


def _orientation_source(report: dict[str, Any] | None) -> str | None:
    """`declared` / `recommended` / `none` / `unknown`, or `None` for an old report.

    Recorded separately from the boolean so the summary can say *who* chose the build
    pose: a person who declared one and a search that supplied one are different
    evidence (audit P2-2), and a memory-vs-model comparison must not blur them.
    """
    block = ((report or {}).get("verdicts") or {}).get("printable") or {}
    source = (block.get("orientation") or {}).get("source")
    return source if isinstance(source, str) and source else None


def _decision_sources(report: dict[str, Any] | None) -> dict[str, int]:
    """How many proposals came from memory, the rule layer, or a model.

    `RepairPlan.proposed_by` is the only place a plan records where it came from
    (`case` for a case-memory replay, `rule` for the deterministic ladder, `llm` for a
    model decision). Counting it per cell is what lets the summary separate *memory
    hits* from *model decisions* instead of crediting both to "the policy".
    """
    counts = {"case": 0, "rule": 0, "llm": 0, "human": 0, "other": 0}
    for rnd in (report or {}).get("rounds") or ():
        source = str(rnd.get("proposed_by") or "other")
        counts[source if source in counts else "other"] += 1
    return counts


def _row_from_report(
    case: SuiteCase, policy: str, report: dict[str, Any], run_dir: Path
) -> dict[str, Any]:
    """The comparison columns, read off one episode report."""
    budget = report["budget"]
    counts = report["counts"]
    manufacturing, orientation = _manufacturing(report)
    decisions = _decision_sources(report)
    return {
        "case": case.name,
        "suite": case.suite,
        "policy": policy,
        "status": "completed",
        "verdict": report["episode"]["verdict"],
        "repaired_by_policy": counts["repaired_by_policy"],
        "rounds": report["episode"]["rounds"],
        "rounds_budget": budget["rounds"]["budget"],
        "geom_calls_charged": budget["geom_calls"]["charged"],
        "sim_calls_charged": budget["sim_calls"]["charged"],
        # Overrun is a separate column, not left gone negative: the ledger permits
        # one overspending round and refuses the next, so equal-budget cells are only
        # comparable when you can filter out the ones that spent more than they had.
        "deterministic_generator": (report.get("run") or {}).get("deterministic_generator"),
        "budget_overrun_total": budget.get("overrun_total", 0),
        "geom_overrun": budget["geom_calls"].get("overrun", 0),
        "sim_overrun": budget["sim_calls"].get("overrun", 0),
        "verifier_geom_calls": budget["verifier_spend"]["geom_calls"],
        "verifier_sim_calls": budget["verifier_spend"]["sim_calls"],
        "environment_errors": counts["environment_errors"],
        "unknown": counts["unknown"],
        "physical_failures_final": counts["physical_failures_final"],
        # Memory and model decisions are separate columns: a policy that replayed a
        # remembered action must not read as "the model chose this", nor the reverse.
        "memory_hits": decisions["case"],
        "model_decisions": decisions["llm"],
        "proposed_by_counts": decisions,
        "kind": case.spec.interface.kind,
        "action_space": case.action_space,
        # 口径: "no orientation declared" is its own state, never a pass.
        "manufacturing_evaluated": manufacturing,
        "orientation_source": _orientation_source(report),
        "print_orientation_deg": orientation,
        "final_clearance_mm": report["design"]["final_clearance_mm"],
        "trace": report["trace"]["path"],
        "report": str(run_dir / "report.json"),
        "error": "",
    }


def _error_row(
    case: SuiteCase, policy: str, run_dir: Path, error: BaseException
) -> dict[str, Any]:
    """A cell that never produced an episode. Its outcome columns stay null.

    Null rather than zero: a missing module is not a failed repair, and a report that
    writes 0 into `repaired_by_policy` has quietly counted the harness's own breakage
    against the policy.
    """
    return {
        "case": case.name,
        "suite": case.suite,
        "policy": policy,
        "status": "error",
        "verdict": None,
        "repaired_by_policy": None,
        "rounds": None,
        "rounds_budget": None,
        "geom_calls_charged": None,
        "sim_calls_charged": None,
        "deterministic_generator": None,
        "budget_overrun_total": None,
        "geom_overrun": None,
        "sim_overrun": None,
        "verifier_geom_calls": None,
        "verifier_sim_calls": None,
        "environment_errors": None,
        "unknown": None,
        "physical_failures_final": None,
        "memory_hits": None,
        "model_decisions": None,
        "proposed_by_counts": None,
        "kind": case.spec.interface.kind,
        "action_space": case.action_space,
        # A cell that produced no episode has no report to read, so nothing was
        # evaluated — the same answer as a run whose process has no overhang rule.
        # Unknown is never written as a pass (audit P0-1).
        "manufacturing_evaluated": False,
        "orientation_source": None,
        "print_orientation_deg": None,
        "final_clearance_mm": None,
        "trace": str(run_dir / "trace.jsonl"),
        "report": str(run_dir / "report.json"),
        "error": f"{type(error).__name__}: {error}",
    }

