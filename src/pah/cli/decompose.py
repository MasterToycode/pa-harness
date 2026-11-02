"""`pah decompose` — one solid in, printable pieces that bolt back together out.

Why this command is the point of the whole N-part thread
--------------------------------------------------------
Up to now PAH could only design the shape it could parametrise: a blank, a socket,
a peg. A decomposition flips that — the *object* decides its own shape, and the
harness contributes only what a generative model cannot: the interfaces, the
evidence, and the printable files.

So the run is:

    whole mesh ──▶ decomposer ──▶ N blanks ──▶ pairing ──▶ N-part spec
               ──▶ materialise (machine every interface) ──▶ verify each pair
               ──▶ 3MF + per-part STL + report

Every arrow after the decomposer is exercised by the two-part path already; what
this command adds is the loop over interfaces and the honesty about what a
decomposed part is not.

What the report refuses to claim
--------------------------------
Two checks do not apply to a decomposed part and are reported as `not_evaluated`
rather than passed: `insert_material` (it compares a *generated* insert against a
parametric reference) and `physics_settle` (its model is a peg in a bore, not a
plate on a plate). `verdicts.assemblable.validated` therefore reads `partial`, and
that word is the deliverable: it says which claims have evidence behind them.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

EXIT_OK = 0
EXIT_NO = 1
EXIT_CRASH = 3

_AXES = {"auto": None, "x": 0, "y": 1, "z": 2}


def add_parser(groups: argparse._SubParsersAction) -> None:
    decompose = groups.add_parser(
        "decompose", help="split one mesh into pieces, machine connectors, verify, export"
    )
    decompose.add_argument("mesh", help="the whole object (.stl/.glb/.obj/.ply)")
    decompose.add_argument(
        "--parts", type=int, default=2, help="how many pieces to cut (default 2)"
    )
    decompose.add_argument(
        "--axis",
        default="auto",
        choices=sorted(_AXES),
        help="cut axis; 'auto' takes the longest extent",
    )
    decompose.add_argument(
        "--clearance",
        type=float,
        default=0.20,
        help="radial fit clearance in mm (default 0.20, inside the 0.10-0.60 band)",
    )
    decompose.add_argument("--out", default=None, help="run directory (default runs/decompose-<ts>)")
    decompose.add_argument(
        "--decomposer",
        default="local",
        help=(
            "'local' (slab splitter, free), 'dir:<path>' (replay a downloaded X-Part "
            "job, free), or 'xpart' (call 腾讯云 组件生成 — costs credits)"
        ),
    )
    decompose.add_argument(
        "--max-interfaces",
        type=int,
        default=12,
        help=(
            "how many joints to machine and verify (default 12). Every joint costs a "
            "socket, a peg and a full verifier sweep, so a 19-part decomposition's 30 "
            "contacts are capped rather than silently taking an hour"
        ),
    )
    decompose.add_argument(
        "--size-mm",
        type=float,
        default=None,
        help="longest extent of the finished object in mm; X-Part answers in normalised units",
    )
    decompose.add_argument(
        "--orientation",
        default=None,
        help=(
            "build orientation as 'x,y,z' degrees, if you know the pose each piece is "
            "printed in. Omitted by default: with N pieces the pair has no single build "
            "pose, and declaring one that is not the printer's would turn an untested "
            "claim into a green tick"
        ),
    )
    decompose.add_argument(
        "--no-verify", action="store_true", help="build only; skip the verifier cascade"
    )
    decompose.add_argument(
        "--allow-paid-gen",
        dest="allow_paid_gen",
        action="store_true",
        help="authorise --decomposer xpart, which spends 腾讯云 credits",
    )
    decompose.add_argument("--json", action="store_true", help="print the report as JSON")
    decompose.set_defaults(handler=_decompose)


def _orientation(text: str | None):
    """`'0,90,0'` as the spec's `Vec3`, or `None` for "not declared"."""
    if not text:
        return None
    pieces = [piece.strip() for piece in str(text).replace(" ", ",").split(",") if piece.strip()]
    if len(pieces) != 3:
        raise SystemExit(f"--orientation wants three angles, got {text!r}")
    return tuple(float(piece) for piece in pieces)  # type: ignore[return-value]


def _example(name: str):
    """Import `examples/<name>.py` by path: the paid adapters live outside the package.

    They are not part of `pah` because they are *choices* — which vendor, which
    model, whose credits — and a package that imports them at startup would make the
    free path depend on the paid one's dependencies.
    """
    import importlib.util  # noqa: PLC0415
    import sys  # noqa: PLC0415

    path = Path(__file__).resolve().parents[3] / "examples" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_pah_example_{name}", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load example adapter at {path}")
    module = importlib.util.module_from_spec(spec)
    # Register before executing: `dataclasses.dataclass` looks the module up in
    # `sys.modules` to resolve string annotations, and an exec without the
    # registration dies with `'NoneType' object has no attribute '__dict__'`.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _run_decomposer(kind: str, whole, args):
    """Pick the decomposer the caller named, and say what it will cost."""
    from pah.design.decompose import split_slabs  # noqa: PLC0415

    if kind == "local":
        return split_slabs(whole, count=args.parts, axis=_AXES[args.axis])

    example = _example("xpart_decomposer")
    if kind.startswith("dir:"):
        directory = kind.split(":", 1)[1]
        return example.load_decomposition_dir(directory, target_size_mm=args.size_mm)
    if kind == "xpart":
        print(
            "  组件生成会消耗腾讯云积分：1 次组件任务"
            + ("（+20 积分后处理）" if False else "")
            + "。这一单提交后即计费，取消不会退款。"
        )
        return example.XPartDecomposer(target_size_mm=args.size_mm).decompose(whole)
    raise SystemExit(
        f"unknown --decomposer {kind!r}; use 'local', 'dir:<path>' or 'xpart'"
    )


def _paid_decomposer(kind: str) -> dict | None:
    """The report fields for a paid decomposer, or `None` for the free ones.

    `--decomposer xpart` is the one value that calls 腾讯云's 组件生成 API, so it is
    declared to the same gate the generator providers use: a billing unit plus the
    vendor adapter's `examples/` path. `--allow-paid-gen` is the only thing that opens
    it; the cost warning in `_run_decomposer` stays as context, but this command now
    refuses instead of leaving the caller to read a warning and press on.
    """
    if kind != "xpart":
        return None
    from pah.app.providers import ProviderConfig, require_paid_generation  # noqa: PLC0415

    config = ProviderConfig(
        id="xpart",
        kind="python",
        billing="per_call",
        compat={"target": "examples/xpart_decomposer.py:XPartDecomposer"},
    )
    require_paid_generation(config, action="pah decompose --decomposer xpart")
    return {
        "paid_generation": True,
        "provider": f"{config.id}:{config.kind}",
        "billing": config.billing,
    }


def _largest_assembly(parts, interfaces):
    """The biggest connected group of pieces, plus what it left behind.

    A decomposition is allowed to return more pieces than we manage to connect — the
    real X-Part run returned 19 and a capped pairing connected only a few of them. The
    spec's own invariant refuses a part that no interface reaches (a part nothing
    holds is a part nothing holds), and that invariant is right: the answer is not to
    weaken it but to build the assembly that *is* connected and say which pieces were
    left out.
    """
    adjacency: dict[str, set[str]] = {part.id: set() for part in parts}
    for interface in interfaces:
        adjacency[interface.a].add(interface.b)
        adjacency[interface.b].add(interface.a)

    seen: set[str] = set()
    groups: list[set[str]] = []
    for part in parts:
        if part.id in seen:
            continue
        group: set[str] = set()
        stack = [part.id]
        while stack:
            node = stack.pop()
            if node in group:
                continue
            group.add(node)
            stack.extend(adjacency[node] - group)
        seen |= group
        groups.append(group)

    def score(group: set[str]) -> tuple[int, int]:
        joints = sum(
            1 for interface in interfaces if interface.a in group and interface.b in group
        )
        return joints, len(group)

    best = max(groups, key=score) if groups else set()
    kept = tuple(part for part in parts if part.id in best)
    kept_interfaces = tuple(
        interface
        for interface in interfaces
        if interface.a in best and interface.b in best
    )
    left = sorted(part.id for part in parts if part.id not in best)
    return kept, kept_interfaces, left


def _decompose(args: argparse.Namespace) -> int:
    from pah.app.providers import PAID_GENERATION_VAR, PaidGenerationRefused  # noqa: PLC0415

    if args.allow_paid_gen:
        # Set before the decomposer is chosen, so the same process-wide authorisation
        # covers `--decomposer xpart` and anything it resolves downstream.
        os.environ[PAID_GENERATION_VAR] = "1"
    try:
        paid = _paid_decomposer(args.decomposer)
    except PaidGenerationRefused as error:
        print(f"pah decompose: {error}")
        return EXIT_CRASH

    import trimesh  # noqa: PLC0415

    from pah.design.decompose import (  # noqa: PLC0415
        file_hash,
        parts_from_decomposition,
        store_parts,
    )
    from pah.design.materialize import materialize
    from pah.design.pairing import propose_interfaces, spec_from_parts
    from pah.design.spec import FitConstraints

    started = time.time()
    source = Path(args.mesh)
    if not source.exists():
        print(f"  no such mesh: {source}")
        return EXIT_NO

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = Path(args.out) if args.out else Path("runs") / f"decompose-{stamp}"
    out.mkdir(parents=True, exist_ok=True)

    whole = trimesh.load(source, force="mesh")
    try:
        decomposition = _run_decomposer(args.decomposer, whole, args)
    except ValueError as error:
        print(f"  cannot decompose {source.name}: {error}")
        return EXIT_NO
    # Store the blanks before the spec exists: `PartSpec.source` names the *file*, so
    # the file has to be written first. The alternative — hashing the mesh — looked
    # equal and was not: STL's float32 coordinates move a 120 mm part by 4e-6 mm, and
    # on a real X-Part job every one of the 19 parts then "failed" its own integrity
    # check.
    stored = store_parts(decomposition.parts, out)
    sources = {part_id: file_hash(path) for part_id, path in stored.items()}

    parts = parts_from_decomposition(decomposition, source_hashes=sources)
    pairing = propose_interfaces(
        parts, clearance_mm=args.clearance, limit=args.max_interfaces
    )
    if not pairing.interfaces:
        print("  the decomposition produced no pair that can carry a connector:")
        print(f"    {pairing.reasons() or 'no pair of pieces is adjacent'}")
        return EXIT_NO

    parts, interfaces, left_out = _largest_assembly(parts, pairing.interfaces)
    if len(interfaces) == 0:
        print("  no connected assembly could be built from this decomposition:")
        print(f"    {len(parts)} piece(s) in the largest group, none joined")
        return EXIT_NO

    spec = spec_from_parts(
        spec_id=f"decompose-{stamp}",
        parts=parts,
        interfaces=interfaces,
        requirement=f"{source.name} decomposed into {len(parts)} printable pieces",
        # The build orientation stays undeclared unless the caller states it. Wall
        # thickness is orientation independent and is still measured; overhang is not
        # evaluated, and the report says `printable: partial` rather than claiming a
        # check that did not happen.
        constraints=FitConstraints(print_orientation_deg=_orientation(args.orientation)),
        provenance={
            "decomposition": decomposition.provenance,
            "source_mesh": str(source),
            **(
                {
                    "unconnected_parts": left_out,
                    "note": (
                        "these pieces came back from the decomposition but no interface "
                        "reaches them, so they are not part of the reported assembly"
                    ),
                }
                if left_out
                else {}
            ),
        },
    )

    design = materialize(spec, out)

    certificates: list = []
    notes: dict[str, str] = {}
    cost = None
    sequence = None
    if not args.no_verify:
        from pah.tools.verifiers.design_cascade import (  # noqa: PLC0415
            DesignVerifierCascade,
            default_registry,
        )

        cascade = DesignVerifierCascade(registry=default_registry(), root=out)
        certificates = cascade.check(design.assembly)
        notes = cascade.notes
        cost = cascade.spent
        sequence = cascade.sequence
    else:
        # `--no-verify` must not read as "everything passed". Every registered check
        # is recorded as not run, so the verdicts say `partial` with the full list.
        from pah.tools.verifiers.design_cascade import default_registry  # noqa: PLC0415

        notes = {
            verifier_id: "verification skipped: this run was built with --no-verify"
            for verifier_id in default_registry().ids()
        }

    from pah.app.runtime import _assembly_block, _verdicts, verdict_text  # noqa: PLC0415

    verdicts = _verdicts(certificates, notes, out / "geometry", sequence=sequence)

    part_rows = {}
    for part in spec.parts:
        mesh = design.meshes[part.id]
        part_rows[part.id] = {
            "role": part.role,
            "size_mm": [round(float(value), 3) for value in part.size_mm],
            "pose_mm": [round(float(value), 3) for value in part.pose_mm],
            "volume_mm3": round(float(mesh.volume), 3),
            "watertight": bool(mesh.is_watertight),
            "faces": int(len(mesh.faces)),
            "source": part.source,
        }

    report = {
        "schema": "pah-decompose-report/1",
        "run": {
            "spec_id": spec.spec_id,
            "source_mesh": str(source),
            "out": str(out),
            "seconds": round(time.time() - started, 2),
            "decomposer": decomposition.provenance,
            "parts": len(spec.parts),
            "parts_returned": len(decomposition.parts),
            "left_out": len(left_out),
            "interfaces": len(spec.interfaces),
            "clearance_mm": args.clearance,
            "verified": not args.no_verify,
            # Present only when a paid decomposer was authorised: the report is where
            # a later reader finds out whether this job spent 腾讯云 credits.
            **(paid or {}),
        },
        "parts": part_rows,
        "interfaces": [
            {
                "id": interface.id,
                "a": interface.a,
                "b": interface.b,
                "kind": interface.kind,
                "params": interface.params,
                "mate_axis": design.interfaces.get(interface.id, {}).get("mate_axis"),
            }
            for interface in spec.interfaces
        ],
        "skipped_pairs": [list(item) for item in pairing.skipped],
        "certificates": [
            {
                "code": certificate.code,
                "family": certificate.family,
                "locus": certificate.locus,
                "magnitude": certificate.magnitude,
                "observed_by": certificate.observed_by,
            }
            for certificate in certificates
        ],
        "notes": notes,
        "assembly": _assembly_block(certificates, sequence),
        "verdicts": verdicts,
        "cost": None
        if cost is None
        else {
            "geom_calls": cost.geom_calls,
            "sim_calls": cost.sim_calls,
            "wall_s": cost.wall_s,
        },
    }
    report_path = out / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )

    from pah.cli.export import export_run  # noqa: PLC0415

    manifest = export_run(out)

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=1))
    else:
        _print(report, manifest, out)
    unknown = any(verdicts[name]["ok"] is None for name in ("assemblable", "printable"))
    if unknown:
        # A check could not run, so the answer is not "no" — it is "we do not know".
        # Exit 3 is the tool-could-not-answer code, which is exactly this case.
        return EXIT_CRASH
    return EXIT_OK if verdicts["assemblable"]["ok"] and verdicts["printable"]["ok"] else EXIT_NO


def _print(report: dict, manifest: dict, out: Path) -> None:
    from pah.app.runtime import verdict_text  # noqa: PLC0415

    run = report["run"]
    print(f"pah decompose · {run['parts']} parts / {run['interfaces']} interfaces · {run['seconds']}s")
    print(f"  source {run['source_mesh']}  ({run['decomposer'].get('method')}, axis {run['decomposer'].get('axis')})")
    if run.get("paid_generation"):
        print(f"  paid gen {run['paid_provider']}  billing={run.get('paid_billing')}"
              f"  (authorised with --allow-paid-gen)")
    for part_id, row in report["parts"].items():
        print(
            f"    {part_id:10s} {row['volume_mm3']:9.2f} mm3  "
            f"{row['size_mm']}  watertight={row['watertight']}"
        )
    for row in report["interfaces"]:
        params = row["params"]
        print(
            f"  joint   {row['id']:28s} {row['a']}->{row['b']}  "
            f"peg {params.get('d_peg_mm')} / hole {params.get('d_hole_mm')} / "
            f"engage {params.get('engage_mm')}  axis {row['mate_axis']}"
        )
    for row in report["skipped_pairs"]:
        print(f"  skipped {row[0]}/{row[1]}: {row[2]}")

    verdicts = report["verdicts"]
    for name in ("assemblable", "printable"):
        verdict = verdicts[name]
        mark = verdict_text(verdict)
        extra = f"  未评估: {', '.join(verdict['not_evaluated'])}" if verdict["not_evaluated"] else ""
        print(f"  {name:10s} {mark} ({verdict['validated']}){extra}")
        if verdict["blocking"]:
            print(f"             blocking: {', '.join(verdict['blocking'])}")
    for code in report["certificates"]:
        if code["family"] != "environment":
            print(f"  evidence  {code['code']:24s} {code['locus'].get('interface', '')}")
    print(f"  artifacts {len(manifest['artifacts'])} file(s) under {out}/print")
    print(f"  report    {out / 'report.json'}")
