"""`pah plugins list` — what this build has mounted, where it came from, what it declares.

The command exists so that "the harness is extensible" is a table you can read in
a terminal, not a claim. It builds an **empty** host (no geometry, no verifier
sanity, no episode) and lets every declared plugin register into it: discovery and
registration are the same call, so a plugin that claims to add an operator but
only prints a banner shows up as a failed row.

Exit codes follow `cli/main.py`: `0` every declared plugin loaded, `1` at least one
did not (the rows say which and why), `3` the request itself was broken (a
`pah.toml` that is not TOML, or a `--config` pointing at nothing).
"""

from __future__ import annotations

import argparse
import json

from pah import plugins as plugin_api
from pah.cli.main import EXIT_CRASH, EXIT_NO, EXIT_OK

__all__ = ["add_parser"]


def add_parser(groups: argparse._SubParsersAction) -> None:
    """Mounts the `plugins` group under the root parser (called from `cli/main.py`)."""
    plugins = groups.add_parser("plugins", help="plugin catalogue and loader")
    commands = plugins.add_subparsers(dest="command", required=True)
    listing = commands.add_parser(
        "list", help="print every plugin: kind, source, api_version, declarations"
    )
    listing.add_argument(
        "--config",
        default=None,
        help="pah.toml to read [plugins] modules from (default: ./pah.toml or $PAH_CONFIG)",
    )
    listing.add_argument(
        "--no-entry-points",
        action="store_true",
        help="skip the importlib.metadata scan (a hermetic listing)",
    )
    listing.add_argument("--json", action="store_true", help="machine-readable output")
    listing.set_defaults(handler=_list)


def _list(args: argparse.Namespace) -> int:
    """Loads the catalogue into a throwaway host and prints it.

    The host is empty on purpose: `pah plugins list` must not require the design
    plane, a database or a run directory. A plugin that only works after an episode
    has already run is a plugin whose `register` is wrong, and the listing is where
    that shows up.
    """
    host = plugin_api.PluginHost()
    try:
        report = plugin_api.load(
            host,
            config_path=args.config,
            include_entry_points=not args.no_entry_points,
        )
    except plugin_api.PluginError as error:
        # A broken config is a broken request, not a broken plugin: say it in one
        # line instead of a traceback, and stop before pretending a run happened.
        print(f"pah plugins list: {error}")
        return EXIT_CRASH

    if args.json:
        payload = {
            "api_version": plugin_api.API_VERSION,
            "kinds": list(plugin_api.KINDS),
            "plugins": [
                {
                    "name": record.name,
                    "kind": record.kind,
                    "source": record.source,
                    "api_version": record.api_version,
                    "module": record.module,
                    "declarations": list(record.declarations),
                    "ok": record.ok,
                    "error": record.error,
                    "optional": record.optional,
                }
                for record in report.records
            ],
            # `failed` counts only failures a run must refuse for; a shipped module
            # that is mid-flight is a `warning`, kept apart so CI can tell them apart.
            "failed": len(report.blocking),
            "warnings": len(report.warnings),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=1))
    else:
        _print_table(report)
    return EXIT_OK if not report.blocking else EXIT_NO


def _print_table(report: plugin_api.PluginLoadReport) -> None:
    header = f"{'KIND':10s} {'SOURCE':10s} {'API':4s} {'STATUS':6s} NAME / DECLARES"
    print(header)
    print("-" * len(header))
    for record in report.records:
        if record.ok:
            status = "OK"
        elif record.optional:
            status = "WARN"
        else:
            status = "FAIL"
        print(
            f"{record.kind or '-':10s} {record.source:10s} "
            f"{record.api_version or '-':4s} {status:6s} {record.name}"
        )
        for line in record.declarations:
            print(f"{'':32s}   {line}")
        if record.error:
            print(f"{'':32s}   ! {record.error}")
    counts = report.by_kind()
    summary = ", ".join(f"{kind}={len(rows)}" for kind, rows in sorted(counts.items()))
    print()
    print(f"{len(report.records)} plugin row(s) [api_version {plugin_api.API_VERSION}]: {summary}")
    if report.blocking:
        print(
            f"{len(report.blocking)} failed: "
            + ", ".join(record.name for record in report.blocking)
        )
    if report.warnings:
        print(
            f"{len(report.warnings)} warning(s), not fatal: "
            + ", ".join(record.name for record in report.warnings)
        )
