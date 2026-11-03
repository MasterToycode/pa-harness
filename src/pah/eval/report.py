"""The evaluation report: `results.jsonl` for machines, `summary.md` for humans.

The two files carry the same rows; only the shape differs. `results.jsonl` is one
raw cell per line (append-friendly, diff-friendly, no aggregation decisions baked
in), and `summary.md` is the aggregated view a reader actually opens.

Three rules this module exists to enforce
-----------------------------------------
1. **Outcome and tool health are never merged into one number.** A policy's
   `repaired_by_policy` count answers "did the loop close on this design"; the
   `environment_errors` and `unknown` columns answer "was the evidence trustworthy";
   the `harness_errors` column answers "did the cell run at all". A table that added
   them together would produce a "success rate" that moves when a verifier crashes,
   and `docs/contracts/v2-interfaces.md` §7.6 forbids exactly that (`docs/pah-dev-doc-v1.md`
   §4.5 is where the rule comes from).
2. **A tuning suite and an evaluation suite are never one row.** The audit (P3-3)
   found `case_memory` looking good on the very demos it was built against. The
   summary therefore renders one block per suite: a holdout result that a reader
   could add to a demo result is not a holdout result.
3. **Memory hits and model decisions are different columns.** `RepairPlan.proposed_by`
   already says where a proposal came from (`case` = a replay from the ledger,
   `llm` = a model decision); crediting both to "the policy" would make the memory
   look like reasoning and the model look like memory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

__all__ = ["RESULTS_SCHEMA", "summary_markdown", "write_results"]

#: Bumped when the row shape changes, so an old `results.jsonl` cannot be read as new.
#: `/2` adds `suite`, `memory_hits`, `model_decisions`, `orientation_source`.
RESULTS_SCHEMA = "pah-eval-rows/2"

#: The outcome column, then the budget columns, then the health columns, then the two
#: decision-source columns. Order is the reading order of a comparison table; the
#: grouping is what keeps them apart. `charged` is what the engine's ledger recorded
#: on rounds; `verified` is the cascade's own meter, which also counts the
#: verification an episode does before its first round.
_HEADER = (
    "| policy | cells | repaired_by_policy | mean_rounds | mean_geom_charged "
    "| mean_geom_verified | mean_sim_charged | mean_sim_verified "
    "| environment_errors | unknown | harness_errors | memory_hits | model_decisions |"
)
_RULE = "|---" * 13 + "|"


def write_results(path: str | Path, rows: Sequence[dict[str, Any]]) -> Path:
    """Writes one JSON object per row, plus a schema line at the top of the file.

    The header row is a normal JSON object with `"schema"` and no `"case"`, so a
    reader can tell the file's version without a sidecar and a JSONL consumer can
    skip it by that key.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"schema": RESULTS_SCHEMA}, ensure_ascii=False) + "\n")
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return target


def summary_markdown(
    rows: Sequence[dict[str, Any]],
    *,
    suite: str,
    policies: Sequence[str],
    rounds: int,
    problems: Iterable[Any] = (),
) -> str:
    """The human view: per-suite policy comparison, per-suite cases, then 口径.

    `suite` is the fallback label for rows that predate the `suite` column; when rows
    carry their own label (they do from `runner.load_suite`), each label gets its own
    block and no table spans two suites.
    """
    problems = list(problems)
    groups = _by_suite(rows, fallback=suite)
    completed = [row for row in rows if row["status"] == "completed"]
    errors = [row for row in rows if row["status"] == "error"]
    lines: list[str] = []
    lines.append("# PAH v2 evaluation — design plane")
    lines.append("")
    if len(groups) == 1:
        lines.append(f"- suite: {_suite_bullet(groups)}")
    else:
        lines.append(f"- suites: {_suite_bullet(groups)}")
    lines.append(f"- policies: {', '.join(f'`{name}`' for name in policies)}")
    lines.append(f"- rounds per episode: **{rounds}** (equal budget for every cell)")
    lines.append(f"- cells: {len(rows)} ({len(completed)} completed, {len(errors)} harness errors)")
    if problems:
        lines.append(f"- suite files with problems: {len(problems)}")
    lines.append("")
    lines.append("## Statistical status")
    lines.append("")
    for label, group in groups.items():
        lines.append(_statistical_line(label, group))
    lines.append("")
    lines.append("## 口径（读数字之前先读这一段）")
    lines.append("")
    lines.append(
        "- This table is the **design plane only** (`DesignSpec → geometry → "
        "certificates → policy → re-materialise → re-verify`). No number here comes "
        "from `runs/m0-connector` or `runs/m1-min`; those are frozen dataset evidence "
        "with their own reports and are never merged into these columns."
    )
    lines.append(
        "- `repaired_by_policy` is an outcome; `environment_errors` and `unknown` are "
        "tool health; `harness_errors` is a cell that never ran. **The columns are "
        "never summed** — a crashed verifier must not look like a worse policy."
    )
    lines.append(
        "- `memory_hits` counts proposals served from the case-memory ledger; "
        "`model_decisions` counts proposals a model made. They are separate evidence "
        "and are never added into one 'the policy decided' number."
    )
    lines.append(
        "- `mean_*` columns are over completed cells only. A mean that included the "
        "error cells would understate every policy by the same invented amount."
    )
    lines.append(_manufacturing_line(rows))
    lines.append("")
    if len(groups) == 1:
        lines.extend(_policies_table(completed, errors, policies))
        lines.append("")
        lines.extend(_cases_table(rows, policies))
    else:
        for label, group in groups.items():
            lines.append(f"## Per policy — suite `{label}`")
            lines.append("")
            lines.extend(
                _policies_table(
                    [row for row in completed if _suite_of(row) == label],
                    [row for row in errors if _suite_of(row) == label],
                    policies,
                )
            )
            lines.append("")
            lines.append(f"## Per case — suite `{label}`")
            lines.append("")
            lines.extend(_cases_table(group, policies))
            lines.append("")
    lines.append("## How to read one cell")
    lines.append("")
    lines.append(
        "`repaired`/`exhausted`/`infeasible`/`error`/`failed` are the loop's verdicts "
        "(`policy/loop.py`); `harness_error` marks a cell that never produced an "
        "episode, with the exception in `results.jsonl`. `action_space` counts the "
        "operators that can express the case's connector kind — a smaller number is an "
        "explanation for a lower repair rate, not a worse policy. `mem=`/`llm=` on a "
        "cell count proposals that came from the memory ledger or from a model."
    )
    if problems:
        lines.append("")
        lines.append("## Suite files that were not usable")
        lines.append("")
        for problem in problems:
            name = getattr(problem, "name", "?")
            path = getattr(problem, "path", "?")
            error = getattr(problem, "error", str(problem))
            lines.append(f"- `{name}` ({path}): {error}")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def _policies_table(
    completed: Sequence[dict[str, Any]],
    errors: Sequence[dict[str, Any]],
    policies: Sequence[str],
) -> list[str]:
    lines = [_HEADER, _RULE]
    for policy in policies:
        cells = [row for row in completed if row["policy"] == policy]
        failures = [row for row in errors if row["policy"] == policy]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{policy}`",
                    str(len(cells) + len(failures)),
                    _ratio(sum(1 for row in cells if row["repaired_by_policy"]), len(cells)),
                    _mean(row["rounds"] for row in cells),
                    _mean(row["geom_calls_charged"] for row in cells),
                    _mean(row["verifier_geom_calls"] for row in cells),
                    _mean(row["sim_calls_charged"] for row in cells),
                    _mean(row["verifier_sim_calls"] for row in cells),
                    str(sum(row["environment_errors"] or 0 for row in cells)),
                    str(sum(row["unknown"] or 0 for row in cells)),
                    str(len(failures)),
                    str(sum(row.get("memory_hits") or 0 for row in cells)),
                    str(sum(row.get("model_decisions") or 0 for row in cells)),
                ]
            )
            + " |"
        )
    return lines


def _cases_table(rows: Sequence[dict[str, Any]], policies: Sequence[str]) -> list[str]:
    lines = ["| case | kind | action_space | " + " | ".join(f"`{name}`" for name in policies) + " |"]
    lines.append("|---" * (3 + len(policies)) + "|")
    for case in _cases_in_order(rows):
        example = next((row for row in rows if row["case"] == case), None)
        cells = []
        for policy in policies:
            row = next(
                (item for item in rows if item["case"] == case and item["policy"] == policy),
                None,
            )
            cells.append(_cell(row))
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{case}`",
                    (example or {}).get("kind") or "-",
                    str(
                        (example or {}).get("action_space")
                        if (example or {}).get("action_space") is not None
                        else "?"
                    ),
                    *cells,
                ]
            )
            + " |"
        )
    return lines


# --------------------------------------------------------------------------- #
# The lines that keep a reader honest
# --------------------------------------------------------------------------- #
def _statistical_line(label: str, rows: Sequence[dict[str, Any]]) -> str:
    """What the suite's size can and cannot support (audit P3-3 / report §5).

    A ranking needs at least twenty independent cases, a fixed generator and seed,
    equal budgets, and a confidence interval. Below that the numbers are a **case
    observation**: useful for debugging, not a claim about which policy is better.
    The line is emitted per suite, so a 24-case holdout and a 4-case demo cannot
    borrow each other's credibility.
    """
    cases = {row["case"] for row in rows}
    n = len(cases)
    completed = [row for row in rows if row["status"] == "completed"]
    known = [
        row.get("deterministic_generator")
        for row in completed
        if row.get("deterministic_generator") is not None
    ]
    deterministic = bool(known) and all(known) and len(known) == len(completed)
    if n < 20 or not deterministic:
        state = "true" if deterministic else ("unknown" if not known else "not always true")
        return (
            f"- `{label}`: **案例观察，不构成策略优劣的统计主张** — n={n} cases, "
            f"deterministic_generator={state}. A policy ranking needs ≥20 independent "
            "cases, a fixed generator and seed, equal budgets and a confidence "
            "interval; none of that is claimed here."
        )
    return (
        f"- `{label}`: n={n}, deterministic_generator=true, equal budget. "
        "Effect sizes only — no confidence interval has been computed, so this is "
        "still not a claim that one policy is better."
    )


def _manufacturing_line(rows: Sequence[dict[str, Any]]) -> str:
    """The build-orientation 口径 line: "not evaluated" is never written as "printable".

    Since the orientation search (audit P2-2) an undeclared spec can still be
    evaluated — in a *recommended* pose — so the old wording ("no print orientation
    declared") is only true for rows that carry no orientation source at all. The
    line now reads the report's `orientation.source` (declared / recommended /
    none / unknown) and says which of them ran, never merging "not evaluated" into
    "printable" (`docs/pah-dev-doc-v1.md` §6).
    """
    declared = recommended = legacy = 0
    missing = 0
    for row in rows:
        source = row.get("orientation_source")
        if source == "declared":
            declared += 1
        elif source == "recommended":
            recommended += 1
        elif source in {"none", "unknown"}:
            missing += 1
        elif row.get("manufacturing_evaluated"):
            legacy += 1
        else:
            missing += 1
    evaluated = declared + recommended + legacy
    total = len(rows)
    if evaluated == 0:
        if missing and not any(row.get("orientation_source") for row in rows):
            # Legacy rows (no source column): the original sentence, kept verbatim so
            # an old summary and a new one do not disagree about the same run.
            return (
                "- manufacturing_validated: not evaluated (no print orientation declared) "
                "— that is not a claim that the designs are printable."
            )
        return (
            "- manufacturing_validated: not evaluated (orientation source: "
            f"{_sources(rows)}; no overhang rule ran) — that is not a claim that the "
            "designs are printable."
        )
    parts = [f"evaluated for {evaluated}/{total} cells"]
    detail = []
    if declared:
        detail.append(f"declared: {declared}")
    if recommended:
        detail.append(f"recommended: {recommended}")
    if legacy:
        detail.append(f"source not recorded: {legacy}")
    if missing:
        detail.append(f"not evaluated: {missing}")
    return "- manufacturing_validated: " + parts[0] + " (" + ", ".join(detail) + ")."


def _sources(rows: Sequence[dict[str, Any]]) -> str:
    seen = sorted({str(row.get("orientation_source")) for row in rows if row.get("orientation_source")})
    return ", ".join(seen) if seen else "none"


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _suite_of(row: dict[str, Any]) -> str:
    return str(row.get("suite") or "")


def _by_suite(
    rows: Sequence[dict[str, Any]], *, fallback: str
) -> dict[str, list[dict[str, Any]]]:
    """Rows grouped by suite label, first appearance order preserved."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        label = _suite_of(row) or fallback
        groups.setdefault(label, []).append(row)
    return groups


def _suite_bullet(groups: dict[str, list[dict[str, Any]]]) -> str:
    parts = []
    for label, group in groups.items():
        n = len({row["case"] for row in group})
        parts.append(f"`{label}` ({n} case{'s' if n != 1 else ''})")
    return ", ".join(parts)


def _cases_in_order(rows: Sequence[dict[str, Any]]) -> list[str]:
    seen: list[str] = []
    for row in rows:
        if row["case"] not in seen:
            seen.append(row["case"])
    return seen


def _cell(row: dict[str, Any] | None) -> str:
    """One case×policy cell: the verdict, plus health and decision-source flags."""
    if row is None:
        return "—"
    if row["status"] == "error":
        return "harness_error"
    verdict = str(row["verdict"])
    flags = []
    if row["environment_errors"]:
        flags.append(f"env={row['environment_errors']}")
    if row["unknown"]:
        flags.append(f"unknown={row['unknown']}")
    memory = row.get("memory_hits") or 0
    model = row.get("model_decisions") or 0
    if memory:
        flags.append(f"mem={memory}")
    if model:
        flags.append(f"llm={model}")
    return verdict + (f" ({', '.join(flags)})" if flags else "")


def _ratio(count: int, total: int) -> str:
    if total == 0:
        return "-"
    return f"{count}/{total} ({100.0 * count / total:.1f}%)"


def _mean(values: Iterable[Any]) -> str:
    numbers = [float(value) for value in values if isinstance(value, (int, float))]
    if not numbers:
        return "-"
    return f"{sum(numbers) / len(numbers):.2f}"
