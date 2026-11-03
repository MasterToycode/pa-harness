"""`pah eval` — equal-budget comparison of decision policies on a design suite.

    pah eval --suite fixtures/design --policies rule,random --rounds 3 --out runs/eval
    pah eval --suite holdout --policies rule,random --rounds 3 --out runs/eval-holdout
    pah eval --suite dev,holdout --policies rule,case_memory --rounds 3 --out runs/eval

Writes two files and prints the second:

    <out>/results.jsonl   one raw cell per line (see `eval/runner.py`)
    <out>/summary.md      the per-suite and per-policy tables (see `eval/report.py`)

A suite is one directory of `DesignSpec` JSON files. `dev` and `holdout` are named
aliases (`fixtures/design` and its `eval-holdout/` subdirectory); naming two suites
runs both in one command and the summary prints them as **separate blocks**, because
a holdout result folded into a demo result is no longer a holdout result.

Everything about *what* is compared and *how the columns are kept apart* lives in
`pah.eval.runner` and `pah.eval.report`; this module owns the argument surface, the
writing of the two files and the exit code. That split is deliberate: the accounting
rules are the part a test must pin, and the CLI is the part a person can change
without re-running the experiment.

Exit codes: `0` every cell ran and every suite file loaded; `1` a cell or a suite file
failed (the rows and the summary say which); `3` the request is broken (no such
suite, an empty `--policies`, an unwritable output).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from pah.cli.main import EXIT_CRASH, EXIT_NO, EXIT_OK

__all__ = ["add_parser", "parse_policies", "resolve_suites"]

#: Named suites. `dev` is the four demos + stress fixtures people tune against;
#: `holdout` is the evaluation set the audit (P3-3) requires. Resolved against the
#: repository root, so `pah eval --suite holdout` works from any directory.
SUITE_ALIASES: dict[str, str] = {
    "dev": "fixtures/design",
    "holdout": "fixtures/design/eval-holdout",
    "stress": "fixtures/design/stress",
}


def add_parser(groups: argparse._SubParsersAction) -> None:
    """Mounts the `eval` command under the root parser (called from `cli/main.py`)."""
    evaluate = groups.add_parser("eval", help="equal-budget policy comparison on a suite")
    evaluate.add_argument(
        "--suite",
        required=True,
        help=(
            "one or more comma-separated suites: a directory of DesignSpec JSON files, "
            "or a name (`dev`, `holdout`, `stress`)"
        ),
    )
    evaluate.add_argument(
        "--policies",
        default="rule,random",
        help="comma-separated policy names (default: rule,random)",
    )
    evaluate.add_argument(
        "--rounds", type=int, default=3, help="episode budget given to every cell"
    )
    evaluate.add_argument(
        "--cases",
        type=int,
        default=None,
        help="run only the first N cases of each suite (a cheap smoke subset)",
    )
    evaluate.add_argument("--out", required=True, help="directory for runs/, results.jsonl, summary.md")
    evaluate.add_argument("--config", default=None, help="pah.toml (plugins and LLM providers)")
    evaluate.add_argument(
        "--allow-paid-gen",
        dest="allow_paid_gen",
        action="store_true",
        help="authorise a generator that costs money for every cell (refused by default)",
    )
    evaluate.add_argument("--seed", type=int, default=0, help="seed handed to every policy")
    evaluate.add_argument("--json", action="store_true", help="print the rows as JSON")
    evaluate.set_defaults(handler=_eval)


def parse_policies(text: str) -> list[str]:
    """`"rule, random"` -> `["rule", "random"]`, order preserved, duplicates dropped.

    Order is preserved because a table whose policy order changes between runs is
    harder to diff than one whose order is the one the caller typed.
    """
    seen: list[str] = []
    for item in text.split(","):
        name = item.strip()
        if name and name not in seen:
            seen.append(name)
    return seen


def resolve_suites(text: str) -> list[tuple[str, Path]]:
    """`"dev,holdout"` (or paths) -> `[(label, path), ...]`, order preserved.

    The label is what the summary prints. An alias keeps its alias (`holdout`), while
    an explicit path gets the directory's name — a report that said `suite:
    eval-holdout` for one invocation and `suite: holdout` for another would make two
    runs look like two different experiments.
    """
    root = Path(__file__).resolve().parents[3]
    resolved: list[tuple[str, Path]] = []
    for entry in (item.strip() for item in text.split(",")):
        if not entry:
            continue
        alias = SUITE_ALIASES.get(entry)
        if alias is not None:
            resolved.append((entry, root / alias))
            continue
        path = Path(entry)
        label = path.name if path.is_dir() else path.parent.name
        resolved.append((label, path))
    return resolved


def _eval(args: argparse.Namespace) -> int:
    """Loads the suite, runs every cell, writes the two artefacts, prints the summary."""
    from pah.app.providers import (  # noqa: PLC0415 - only paid configs need it
        PAID_GENERATION_VAR,
        PaidGenerationRefused,
        ProviderError,
        paid_generation_record,
    )
    from pah.eval import report as eval_report  # noqa: PLC0415 - keeps `pah --help` light
    from pah.eval import runner as eval_runner  # noqa: PLC0415

    policies = parse_policies(args.policies)
    if not policies:
        print("pah eval: --policies is empty; name at least one policy")
        return EXIT_CRASH
    if args.cases is not None and args.cases < 1:
        print("pah eval: --cases must be at least 1")
        return EXIT_CRASH
    if args.allow_paid_gen:
        # One authorisation for the whole comparison: every cell builds its own
        # runtime, and a flag that only opened the gate for the first cell would make
        # the table mix paid and refused cells.
        os.environ[PAID_GENERATION_VAR] = "1"
    try:
        # Refuse before the first cell writes anything: an eval that silently spends
        # a generation per cell is the most expensive possible accident.
        paid = paid_generation_record(args.config, action="pah eval")
    except (PaidGenerationRefused, ProviderError) as error:
        print(f"pah eval: {error}")
        return EXIT_CRASH
    if "llm" in policies:
        # Same discipline as `pah run`: a comparison that asks a model with no provider
        # (or no key) is a broken request, not eight harness-error rows. Refusing here
        # keeps the table honest instead of filling it with infrastructure noise.
        from pah.app import runtime as runtime_module  # noqa: PLC0415

        try:
            runtime_module.require_provider(args.config)
        except RuntimeError as error:
            print(f"pah eval: {error}")
            return EXIT_CRASH

    suites = resolve_suites(args.suite)
    if not suites:
        print(f"pah eval: --suite {args.suite!r} names no suite")
        return EXIT_CRASH
    cases: list[Any] = []
    problems: list[Any] = []
    for label, path in suites:
        try:
            suite_cases, suite_problems = eval_runner.load_suite(path, label=label)
        except FileNotFoundError as error:
            print(f"pah eval: {error}")
            return EXIT_CRASH
        except OSError as error:
            print(f"pah eval: cannot read {path}: {error}")
            return EXIT_CRASH
        if args.cases is not None:
            suite_cases = suite_cases[: args.cases]
        print(f"pah eval: suite `{label}` — {len(suite_cases)} case(s) from {path}")
        cases.extend(suite_cases)
        problems.extend(suite_problems)

    for problem in problems:
        print(f"pah eval: skipping {problem.path}: {problem.error}")
    if not cases:
        print(f"pah eval: no usable DesignSpec under {args.suite}")
        return EXIT_NO

    out = Path(args.out)
    try:
        rows = eval_runner.run_suite(
            cases,
            policies=policies,
            rounds=args.rounds,
            out=out,
            config_path=args.config,
            seed=args.seed,
        )
    except OSError as error:
        print(f"pah eval: cannot write under {out}: {error}")
        return EXIT_CRASH

    if paid is not None:
        # The cells are the rows here; `results.jsonl` is this command's report, so the
        # authorisation is recorded on every row rather than in a separate file nobody
        # reads. Every cell used the same provider — the gate is process-wide.
        for row in rows:
            row["paid_generation"] = True
            row["paid_provider"] = paid["provider"]

    results_path = eval_report.write_results(out / "results.jsonl", rows)
    summary = eval_report.summary_markdown(
        rows,
        suite=str(args.suite),
        policies=policies,
        rounds=args.rounds,
        problems=problems,
    )
    summary_path = out / "summary.md"
    summary_path.write_text(summary, "utf-8")

    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=1))
    else:
        print(summary)
        print(f"wrote {results_path}")
        print(f"wrote {summary_path}")

    broken = [row for row in rows if row["status"] == "error"]
    return EXIT_OK if not broken and not problems else EXIT_NO
