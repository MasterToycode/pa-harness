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
def load_suite(suite: str | Path) -> tuple[list[SuiteCase], list[CaseProblem]]:
    """Every `*.json` under `suite` as a `DesignSpec`, plus the files that were not.

    Two layouts are accepted on purpose. A file may be the spec itself (what
    `pah design new` writes) or wrap it as `{"spec": {...}}` (what a fixture that also
    carries an expected verdict looks like). Guessing further — inferring a kind from
    a mesh, defaulting a missing constraint — would mean the suite silently changes
    what a run was given, which is the failure mode this function exists to avoid.
    """
    path = Path(suite)
    if not path.exists():
        raise FileNotFoundError(f"no such suite: {path}")
    files = [path] if path.is_file() else sorted(path.rglob("*.json"))
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
            SuiteCase(name=name, spec=spec, path=file, action_space=action_space(spec.interface.kind))
        )
    return cases, problems


def _case_name(root: Path, file: Path) -> str:
    """A stable, filesystem-safe case id: the path under the suite without `.json`."""
    try:
        relative = file.relative_to(root)
    except ValueError:  # `suite` was a single file
        relative = Path(file.name)
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


def _manufacturing(case: SuiteCase) -> tuple[bool, list[float] | None]:
    """Was a build orientation declared? `None` means "not evaluated", not "printable".

    The distinction is a contract one (`FitConstraints.print_orientation_deg` and
    `docs/pah-dev-doc-v1.md` §6): an overhang verdict needs an orientation, and a run
    that declared none must not be counted as having passed a manufacturing check.
    Read defensively so an older spec model without the field reads as "not declared".
    """
    orientation = getattr(case.spec.constraints, "print_orientation_deg", None)
    if orientation is None:
        return False, None
    return True, [float(value) for value in orientation]


def _row_from_report(
    case: SuiteCase, policy: str, report: dict[str, Any], run_dir: Path
) -> dict[str, Any]:
    """The comparison columns, read off one episode report."""
    budget = report["budget"]
    counts = report["counts"]
    manufacturing, orientation = _manufacturing(case)
    return {
        "case": case.name,
        "policy": policy,
        "status": "completed",
        "verdict": report["episode"]["verdict"],
        "repaired_by_policy": counts["repaired_by_policy"],
        "rounds": report["episode"]["rounds"],
        "rounds_budget": budget["rounds"]["budget"],
        "geom_calls_charged": budget["geom_calls"]["charged"],
        "sim_calls_charged": budget["sim_calls"]["charged"],
        "verifier_geom_calls": budget["verifier_spend"]["geom_calls"],
        "verifier_sim_calls": budget["verifier_spend"]["sim_calls"],
        "environment_errors": counts["environment_errors"],
        "unknown": counts["unknown"],
        "physical_failures_final": counts["physical_failures_final"],
        "kind": case.spec.interface.kind,
        "action_space": case.action_space,
        # 口径: "no orientation declared" is its own state, never a pass.
        "manufacturing_evaluated": manufacturing,
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
        "policy": policy,
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
        "kind": case.spec.interface.kind,
        "action_space": case.action_space,
        "manufacturing_evaluated": _manufacturing(case)[0],
        "print_orientation_deg": _manufacturing(case)[1],
        "final_clearance_mm": None,
        "trace": str(run_dir / "trace.jsonl"),
        "report": str(run_dir / "report.json"),
        "error": f"{type(error).__name__}: {error}",
    }
