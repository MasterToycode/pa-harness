"""`pah` command line — the first slice of the contract in `pah-dev-doc-v1.md` §7.4.

`verify sanity` (§4.8's run-before-use gate) and `probe connector` (the 705/388
reproduction) are the M0 commands; v2 adds `run`, `design`, `eval` and `plugins`,
each mounted from its own `cli/<name>.py` so this file stays the dispatcher and the
existing two commands keep their behaviour byte for byte.

Exit codes are §7.4's and nothing else reads them:

    0  the thing asked for was true
    1  a business answer of "no"  (a verifier disagrees with its fixtures)
    2  not reproducible           (the probe ran and missed the baseline)
    3  the tool itself crashed
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback

from pah.app.env import load_env_file
from pah.tools.verifiers import default_registry

EXIT_OK = 0
EXIT_NO = 1
EXIT_NOT_REPRODUCED = 2
EXIT_CRASH = 3


def main(argv: list[str] | None = None) -> int:
    """Parses `argv` (default `sys.argv[1:]`) and dispatches; returns the exit code."""
    # A repo-local `.env` fills in missing credentials before any handler looks for
    # them, so a key is pasted once instead of exported in every shell. Values never
    # reach stdout (JSON consumers) and never get printed: only the names do, on
    # stderr, because "did my key load?" should not require reading source.
    loaded = load_env_file()
    if loaded:
        names = ", ".join(sorted(loaded))
        print(f"pah: loaded {len(loaded)} variable(s) from .env: {names}", file=sys.stderr)
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except KeyboardInterrupt:
        return EXIT_NO
    except Exception:  # noqa: BLE001  (a crash must be exit 3, never a traceback-only)
        traceback.print_exc()
        return EXIT_CRASH


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pah", description=__doc__.splitlines()[0])
    groups = parser.add_subparsers(dest="group", required=True)

    verify = groups.add_parser("verify", help="verifier self-checks")
    verify_commands = verify.add_subparsers(dest="command", required=True)
    sanity = verify_commands.add_parser(
        "sanity", help="run every verifier's known_good/known_bad fixtures (§4.8)"
    )
    sanity.add_argument("--json", action="store_true", help="machine-readable output")
    sanity.set_defaults(handler=_verify_sanity)

    probe = groups.add_parser("probe", help="run one verifier across the benchmark")
    probe_commands = probe.add_subparsers(dest="command", required=True)
    connector = probe_commands.add_parser("connector", help="connector_feasible probe")
    connector.add_argument("--split", default="feasible", choices=["feasible", "all"])
    connector.add_argument(
        "--cases", default="all", help="'all' or the first N assemblies"
    )
    connector.add_argument("--out", default="runs/m0-connector", help="run directory")
    connector.add_argument("--jobs", type=int, default=8, help="worker processes")
    connector.add_argument(
        "--unit-mm",
        default="scale",
        choices=["scale", "1000", "meta_scale"],
        help="the contested scale protocol; 'scale' is the gold one",
    )
    connector.add_argument(
        "--gold",
        default=None,
        help="frozen per-edge rows for the edge-by-edge diff ('' to skip)",
    )
    connector.add_argument("--json", action="store_true", help="machine-readable output")
    connector.set_defaults(handler=_probe_connector)

    # v2 groups. Imported here, not at module import time, so `pah --help` and the
    # two M0 commands do not pay for the geometry stack (the modules keep their own
    # heavy imports inside their handlers for the same reason).
    from pah.cli import build as build_cli  # noqa: PLC0415
    from pah.cli import design as design_cli  # noqa: PLC0415
    from pah.cli import eval as eval_cli  # noqa: PLC0415
    from pah.cli import export as export_cli  # noqa: PLC0415
    from pah.cli import plugins as plugins_cli  # noqa: PLC0415
    from pah.cli import preview as preview_cli  # noqa: PLC0415
    from pah.cli import providers as providers_cli  # noqa: PLC0415
    from pah.cli import run as run_cli  # noqa: PLC0415
    from pah.cli import serve as serve_cli  # noqa: PLC0415

    build_cli.add_parser(groups)
    run_cli.add_parser(groups)
    preview_cli.add_parser(groups)
    serve_cli.add_parser(groups)
    design_cli.add_parser(groups)
    eval_cli.add_parser(groups)
    export_cli.add_parser(groups)
    plugins_cli.add_parser(groups)
    providers_cli.add_parser(groups)

    return parser


# --------------------------------------------------------------------------- #
# pah verify sanity
# --------------------------------------------------------------------------- #
def _verify_sanity(args: argparse.Namespace) -> int:
    """§4.8: a verifier that fails its own fixtures is disabled, not warned about."""
    registry = default_registry()
    results = registry.sanity()
    payload = {
        "verifiers": {
            verifier_id: {
                "ok": ok,
                "cost_class": registry.get(verifier_id).spec.cost_class,
                "codes": list(registry.get(verifier_id).spec.codes),
                "why": registry.why_disabled(verifier_id),
            }
            for verifier_id, ok in results.items()
        },
        "disabled": sorted(registry.disabled()),
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=1))
    else:
        for verifier_id, detail in payload["verifiers"].items():
            mark = "PASS" if detail["ok"] else "FAIL"
            print(f"  {mark}  {verifier_id:20s} {detail['cost_class']:9s} "
                  f"{len(detail['codes'])} codes")
            if detail["why"]:
                print(f"        disabled: {detail['why']}")
        print(f"\n{len(results)} verifier(s), "
              f"{len(registry.disabled())} disabled")
    return EXIT_OK if not registry.disabled() else EXIT_NO


# --------------------------------------------------------------------------- #
# pah probe connector
# --------------------------------------------------------------------------- #
def _probe_connector(args: argparse.Namespace) -> int:
    from pah.cli import probe  # noqa: PLC0415  (imports trimesh lazily)

    cases = None if str(args.cases).lower() == "all" else int(args.cases)
    gold = None if args.gold == "" else (args.gold or probe.DEFAULT_GOLD)
    report = probe.run(
        out=args.out,
        split=args.split,
        cases=cases,
        jobs=args.jobs,
        unit_mm_rule=args.unit_mm,
        gold=gold,
        quiet=args.json,
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=1))
    else:
        _print_report(report)
    if not report["complete"]:
        return EXIT_NO
    if report["scope"] != "gold" or report["reproduced"]:
        # A `--cases N` run only had to finish; its rate is not the baseline's.
        return EXIT_OK
    return EXIT_NOT_REPRODUCED


def _print_report(report: dict) -> None:
    gold = report["gold"]
    delta = report["delta"]
    print()
    print("=" * 62)
    print("M0 connector probe — gold standard 705 / 388 / 55.0%")
    print("=" * 62)
    print(f"  split={report['split']}  scope={report['scope']}  "
          f"assemblies={report['assemblies']}  edges_total={report['edges_total']}  "
          f"unit_mm rule={report['unit_mm_rule']}")
    rows = [
        ("edges_total", report["edges_total"], None),
        ("feasible", report["edges_in_split"], gold["feasible"]),
        ("success", report["success"], gold["success"]),
    ]
    for name, value, base in rows:
        tail = "" if base is None else f"   gold {base:<6d} delta {value - base:+d}"
        print(f"  {name:14s} {value:6d}{tail}")
    print(f"  {'rate':14s} {report['rate']:6.1%}   gold {gold['rate']:.1%}"
          f"   delta {delta['rate']:+.1%}")
    print(f"  {'ran':14s} {report['ran']:6d} / {report['edges_in_split']}")
    print()
    print(f"  unreatable_too_small {report['unreatable_too_small']:4d} (gold 29)"
          f" | adapter-feasible {report['adapter_feasible']}")
    for code, count in list(report["codes"].items())[:8]:
        print(f"    {code:26s} {count:4d}")
    diff = report.get("edge_diff") or {}
    if diff.get("available"):
        print(f"\n  edge-by-edge vs frozen gold rows: "
              f"{diff['compared']} compared, {diff['mismatches']} mismatch")
        for example in diff["examples"]:
            print(f"    {example['case_id']}:{example['edge_id']} "
                  f"adapter={example['adapter']} gold={example['gold']}")
    print()
    if report["scope"] != "gold":
        print(f"  SUBSET OK   ({report['seconds']} s) — gold columns compare to the "
              f"full split, not to these {report['assemblies']} assemblies")
    else:
        extra = ""
        if report.get("environment_edges"):
            extra = (f"\n  {report['environment_edges']} edge(s) gold lost to "
                     f"`MemoryError` pass here -> adjusted success "
                     f"{report['adjusted_success']} / {report['edges_in_split']}")
        print(f"  {report['verdict'].upper()}"
              f"   ({report['seconds']} s){extra}")


if __name__ == "__main__":
    sys.exit(main())
