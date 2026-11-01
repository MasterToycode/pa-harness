"""`pah run` — one repair episode, composed from the design plane, start to finish.

    pah run --spec runs/demo/spec.json --policy rule --rounds 3 --out runs/demo/ep0 \
            [--resume] [--inject-fault verifier] [--stop-after-round N]

The command is assembly and I/O only. It loads a `DesignSpec`, asks
`pah.app.runtime.build` for a composed runtime, runs the episode and prints what was
written. Every decision — what failed, what to repair, whether the result is good —
belongs to a verifier or a policy, never to this file; the three rules in
`docs/contracts/v2-interfaces.md` §9 are what a reader should check it against.

Exit codes are `cli/main.py`'s:

    0  the episode ran to a stop condition and left its geometry behind
    1  a business answer of "no": the loop could not establish a result, or the run
       finished without the geometry it is supposed to deliver
    2  unused here (it means "not reproduced" for `probe connector`)
    3  the request or the environment is broken: missing spec, missing LLM provider,
       a neighbour module that has not landed, a run directory that already exists
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pah.cli.main import EXIT_CRASH, EXIT_NO, EXIT_OK
from pah.design.spec import DesignSpec

__all__ = ["add_parser"]

#: The three builtin policies. `--policy` accepts any other name too: a policy plugin
#: (contract §6) publishes one into `host.policies`, and the runtime resolves plugin
#: names before asking T3 — a closed argparse list would make the extension point
#: unreachable from the command line, which is the only thing a plugin needs it for.
POLICIES = ("rule", "random", "llm")

#: Mirrors `pah.app.runtime.FAULTS`. Kept local so `pah --help` does not import the
#: geometry stack; `runtime.build` re-validates the value, so a drift fails loudly
#: instead of silently accepting a fault nobody injects.
FAULTS = ("verifier",)


def add_parser(groups: argparse._SubParsersAction) -> None:
    """Mounts the `run` command under the root parser (called from `cli/main.py`)."""
    run = groups.add_parser("run", help="run one repair episode on a design spec")
    run.add_argument("--spec", required=True, help="DesignSpec JSON to repair")
    run.add_argument(
        "--policy",
        default="rule",
        help="decision policy: " + "|".join(POLICIES) + " or a policy plugin's name",
    )
    run.add_argument(
        "--rounds", type=int, default=3, help="the round ceiling this episode may spend"
    )
    run.add_argument("--out", required=True, help="run directory for trace/report/geometry")
    run.add_argument(
        "--resume",
        action="store_true",
        help="continue the interrupted episode already recorded in --out",
    )
    run.add_argument(
        "--inject-fault",
        dest="inject_fault",
        choices=FAULTS,
        default=None,
        help="make the cascade fail once on purpose (acceptance: the loop recovers)",
    )
    run.add_argument(
        "--stop-after-round",
        dest="stop_after_round",
        type=int,
        default=None,
        help="stop cleanly after N closed rounds, leaving the run resumable",
    )
    run.add_argument("--seed", type=int, default=0, help="seed handed to the policy")
    run.add_argument("--geom-calls", dest="geom_calls", type=int, default=None)
    run.add_argument("--sim-calls", dest="sim_calls", type=int, default=None)
    run.add_argument(
        "--config", default=None, help="pah.toml (plugins and LLM providers)"
    )
    run.add_argument("--json", action="store_true", help="print the report as JSON")
    run.set_defaults(handler=_run)


def _run(args: argparse.Namespace) -> int:
    """Loads the spec, composes the runtime, runs the episode, reports the outcome."""
    from pah.app import runtime  # noqa: PLC0415 - keeps `pah --help` off the geometry stack

    spec = _load_spec(args.spec)
    if spec is None:
        return EXIT_CRASH
    if args.resume and args.inject_fault:
        print(
            "pah run: --inject-fault is not allowed with --resume; the fault would be "
            "injected into a different process than the round it is meant to break"
        )
        return EXIT_CRASH
    if args.policy == "llm":
        # Preflight before composing: with no provider configured the honest answer is
        # "there is no model to ask", not a stack trace from inside T3's import.
        try:
            runtime.require_provider(args.config)
        except RuntimeError as error:
            print(f"pah run: {error}")
            return EXIT_CRASH

    cfg = runtime.RuntimeConfig(
        out=Path(args.out),
        config_path=Path(args.config) if args.config else None,
        seed=args.seed,
        geom_calls=args.geom_calls,
        sim_calls=args.sim_calls,
        inject_fault=args.inject_fault,
        stop_after_round=args.stop_after_round,
        policy=args.policy,
        rounds=args.rounds,
        resume=args.resume,
    )
    try:
        composed = runtime.build(cfg)
    except (RuntimeError, ValueError) as error:
        print(f"pah run: {error}")
        return EXIT_CRASH
    except Exception as error:  # noqa: BLE001 - a broken config is a one-line refusal
        print(f"pah run: could not compose the harness: {type(error).__name__}: {error}")
        return EXIT_CRASH

    try:
        report = composed.run_episode(
            spec,
            policy_name=args.policy,
            rounds=args.rounds,
            out_dir=Path(args.out),
            resume=args.resume,
        )
    except (RuntimeError, ValueError) as error:
        print(f"pah run: {error}")
        return EXIT_CRASH
    finally:
        composed.close()

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=1))
    else:
        _print_report(report)
    return _exit_code(report)


def _load_spec(path: str) -> DesignSpec | None:
    """The spec at `path`, or `None` after saying why it cannot be used."""
    target = Path(path)
    if not target.exists():
        print(f"pah run: no spec at {target}")
        return None
    try:
        return DesignSpec.model_validate_json(target.read_text("utf-8"))
    except Exception as error:  # noqa: BLE001 - a bad spec is a client error, not a crash
        print(f"pah run: {target} is not a DesignSpec: {type(error).__name__}: {error}")
        return None


def _exit_code(report: dict) -> int:
    """`0` when the episode delivered, `1` when the answer is "no".

    A `stopped` episode exits `0`: the caller asked it to stop and it did, and the
    run directory is resumable. An incomplete geometry set exits `1` even when the
    loop says `repaired`, because the deliverable of a design run is a usable STL
    set, not a verdict on its own.
    """
    if report["episode"]["verdict"] == "failed":
        return EXIT_NO
    if not report["geometry"]["complete"]:
        return EXIT_NO
    return EXIT_OK


def _print_report(report: dict) -> None:
    run = report["run"]
    design = report["design"]
    episode = report["episode"]
    budget = report["budget"]
    counts = report["counts"]

    print()
    print("=" * 66)
    print(f"pah run — {design['spec_id']}  policy={run['policy']}  "
          f"rounds≤{budget['rounds']['budget']}")
    print("=" * 66)
    print(f"  requirement   {design['requirement'][:60] or '-'}")
    print(f"  connector     {design['kind']}  "
          f"clearance {design['initial_clearance_mm']:+.3f} -> "
          f"{design['final_clearance_mm']:+.3f} mm  "
          f"(band {design['fit_band_mm'][0]}..{design['fit_band_mm'][1]})")
    print(f"  spec          revision {design['final_revision']}  "
          f"{design['initial_content_key'][:18]} -> {design['final_content_key'][:18]}")
    print()
    for row in report["rounds"]:
        codes = ",".join(row["codes"]) or "-"
        print(f"  round {row['index']}  {row['verdict']:<5s} {row['operator']:<20s} "
              f"{codes:<40s} geom={row['geom_calls']} sim={row['sim_calls']}")
    print()
    print(f"  verdict       {episode['verdict']}  ({episode['reason']})")
    print(f"  budget        rounds {budget['rounds']['used']}/{budget['rounds']['budget']}  "
          f"geom {budget['geom_calls']['charged']}/{budget['geom_calls']['budget']}  "
          f"sim {budget['sim_calls']['charged']}/{budget['sim_calls']['budget']}")
    print(f"  counts        repaired_by_policy={counts['repaired_by_policy']}  "
          f"environment_errors={counts['environment_errors']}  "
          f"unknown={counts['unknown']}  not_evaluated={counts['not_evaluated']}")
    for verifier_id, reason in report["verifiers"].get("notes", {}).items():
        print(f"  not evaluated {verifier_id}: {reason}")
    print(f"  geometry      {report['geometry']['dir']}"
          f"  ({len(report['geometry']['files'])} file(s), "
          f"complete={report['geometry']['complete']})")
    print(f"  trace         {report['trace']['path']}  "
          f"({report['trace']['rows']} rows, "
          f"{len(report['trace']['replay_hashes'])} geometry hashes)")
    print(f"  report        {Path(run['out']) / 'report.json'}")
    for note in report.get("notes", ()):
        print(f"  note          {note}")
    if run["status"] == "stopped":
        print()
        print("  stopped cleanly. Resume with: "
              f"pah run --spec {Path(run['out']) / 'spec-initial.json'} "
              f"--policy {run['policy']} --rounds {budget['rounds']['budget']} "
              f"--out {run['out']} --resume")
