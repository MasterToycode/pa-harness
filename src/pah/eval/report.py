"""The evaluation report: `results.jsonl` for machines, `summary.md` for humans.

The two files carry the same rows; only the shape differs. `results.jsonl` is one
raw cell per line (append-friendly, diff-friendly, no aggregation decisions baked
in), and `summary.md` is the aggregated view a reader actually opens.

The one rule this module exists to enforce
------------------------------------------
**Outcome and tool health are never merged into one number.** A policy's
`repaired_by_policy` count answers "did the loop close on this design"; the
`environment_errors` and `unknown` columns answer "was the evidence trustworthy"; the
`harness_errors` column answers "did the cell run at all". A table that added them
together would produce a "success rate" that moves when a verifier crashes, and
`docs/contracts/v2-interfaces.md` §7.6 forbids exactly that (`docs/pah-dev-doc-v1.md`
§4.5 is where the rule comes from). The 口径 block at the top of every summary says
so out loud, because a pasted table loses its footnotes first.

It also says, out loud, that no dataset number appears: the design plane's numbers
and the frozen M0/M1 evidence live in different directories and different reports,
and a summary that mixed them would make a change in the dataset look like a change
in the agent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

__all__ = ["RESULTS_SCHEMA", "summary_markdown", "write_results"]

#: Bumped if the row shape changes, so an old `results.jsonl` cannot be read as new.
RESULTS_SCHEMA = "pah-eval-rows/1"

#: The outcome column, then the budget columns, then the health columns. Order is
#: the reading order of a comparison table; the grouping is what keeps them apart.
#: `charged` is what the engine's ledger recorded on rounds; `verified` is the
#: cascade's own meter, which also counts the verification an episode does before its
#: first round. Both are reported because a 0-round repair charged nothing and still
#: built triangles — printing only one of them lets a reader conclude either way.
_HEADER = (
    "| policy | cells | repaired_by_policy | mean_rounds | mean_geom_charged "
    "| mean_geom_verified | mean_sim_charged | mean_sim_verified "
    "| environment_errors | unknown | harness_errors |"
)
_RULE = "|---|---|---|---|---|---|---|---|---|---|---|"


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
    """The human view: per-policy comparison, then one row per case, then 口径."""
    problems = list(problems)
    completed = [row for row in rows if row["status"] == "completed"]
    errors = [row for row in rows if row["status"] == "error"]
    lines: list[str] = []
    lines.append("# PAH v2 evaluation — design plane")
    lines.append("")
    lines.append(f"- suite: `{suite}`")
    lines.append(f"- policies: {', '.join(f'`{name}`' for name in policies)}")
    lines.append(f"- rounds per episode: **{rounds}** (equal budget for every cell)")
    lines.append(f"- cells: {len(rows)} ({len(completed)} completed, {len(errors)} harness errors)")
    if problems:
        lines.append(f"- suite files with problems: {len(problems)}")
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
        "- `mean_*` columns are over completed cells only. A mean that included the "
        "error cells would understate every policy by the same invented amount."
    )
    lines.append(_manufacturing_line(rows))
    lines.append("")
    lines.append("## Per policy")
    lines.append("")
    lines.append(_HEADER)
    lines.append(_RULE)
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
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Per case")
    lines.append("")
    header = "| case | kind | action_space | " + " | ".join(f"`{name}`" for name in policies) + " |"
    lines.append(header)
    lines.append("|---" * (3 + len(policies)) + "|")
    for case in _cases_in_order(rows):
        example = next((row for row in rows if row["case"] == case), None)
        cells = []
        for policy in policies:
            row = next((item for item in rows if item["case"] == case and item["policy"] == policy), None)
            cells.append(_cell(row))
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{case}`",
                    (example or {}).get("kind") or "-",
                    str((example or {}).get("action_space") if (example or {}).get("action_space") is not None else "?"),
                    *cells,
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## How to read one cell")
    lines.append("")
    lines.append(
        "`repaired`/`exhausted`/`infeasible`/`error`/`failed` are the loop's verdicts "
        "(`policy/loop.py`); `harness_error` marks a cell that never produced an "
        "episode, with the exception in `results.jsonl`. `action_space` counts the "
        "operators that can express the case's connector kind — a smaller number is an "
        "explanation for a lower repair rate, not a worse policy."
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


def _manufacturing_line(rows: Sequence[dict[str, Any]]) -> str:
    """The build-orientation 口径 line: "not evaluated" is never written as "printable".

    `summary.md` is the artefact people quote, so the absence of a manufacturing
    check has to be visible in it. A table of green outcomes with no such line is how
    "we did not look" turns into "we looked and it passed" — the one merge the
    contract forbids (`docs/pah-dev-doc-v1.md` §6).
    """
    evaluated = [row for row in rows if row.get("manufacturing_evaluated")]
    if not evaluated:
        return (
            "- manufacturing_validated: not evaluated (no print orientation declared) "
            "— that is not a claim that the designs are printable."
        )
    return (
        f"- manufacturing_validated: evaluated for {len(evaluated)}/{len(rows)} "
        "cells (those cases declare a print orientation); the rest are not evaluated."
    )


def _cases_in_order(rows: Sequence[dict[str, Any]]) -> list[str]:
    seen: list[str] = []
    for row in rows:
        if row["case"] not in seen:
            seen.append(row["case"])
    return seen


def _cell(row: dict[str, Any] | None) -> str:
    """One case×policy cell: the verdict, plus the two health flags when they fired."""
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
