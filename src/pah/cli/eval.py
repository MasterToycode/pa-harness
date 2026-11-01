"""`pah eval` — equal-budget comparison of decision policies on a design suite.

    pah eval --suite fixtures/design --policies rule,random --rounds 3 --out runs/eval

Writes two files and prints the second:

    <out>/results.jsonl   one raw cell per line (see `eval/runner.py`)
    <out>/summary.md      the per-policy and per-case tables (see `eval/report.py`)

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
from pathlib import Path

from pah.cli.main import EXIT_CRASH, EXIT_NO, EXIT_OK

__all__ = ["add_parser", "parse_policies"]


def add_parser(groups: argparse._SubParsersAction) -> None:
    """Mounts the `eval` command under the root parser (called from `cli/main.py`)."""
    evaluate = groups.add_parser("eval", help="equal-budget policy comparison on a suite")
    evaluate.add_argument(
        "--suite",
        required=True,
        help="a directory of DesignSpec JSON files, or one such file",
    )
    evaluate.add_argument(
        "--policies",
        default="rule,random",
        help="comma-separated policy names (default: rule,random)",
    )
    evaluate.add_argument(
        "--rounds", type=int, default=3, help="episode budget given to every cell"
    )
    evaluate.add_argument("--out", required=True, help="directory for runs/, results.jsonl, summary.md")
    evaluate.add_argument("--config", default=None, help="pah.toml (plugins and LLM providers)")
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


def _eval(args: argparse.Namespace) -> int:
    """Loads the suite, runs every cell, writes the two artefacts, prints the summary."""
    from pah.eval import report as eval_report  # noqa: PLC0415 - keeps `pah --help` light
    from pah.eval import runner as eval_runner  # noqa: PLC0415

    policies = parse_policies(args.policies)
    if not policies:
        print("pah eval: --policies is empty; name at least one policy")
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

    try:
        cases, problems = eval_runner.load_suite(args.suite)
    except FileNotFoundError as error:
        print(f"pah eval: {error}")
        return EXIT_CRASH
    except OSError as error:
        print(f"pah eval: cannot read {args.suite}: {error}")
        return EXIT_CRASH

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
