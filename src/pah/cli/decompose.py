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


def _decompose(args: argparse.Namespace) -> int:
    import trimesh  # noqa: PLC0415

    from pah.design.decompose import parts_from_decomposition, split_slabs, store_parts
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
        decomposition = split_slabs(whole, count=args.parts, axis=_AXES[args.axis])
    except ValueError as error:
        print(f"  cannot decompose {source.name}: {error}")
        return EXIT_NO
    parts = parts_from_decomposition(decomposition)
    pairing = propose_interfaces(parts, clearance_mm=args.clearance)
    if not pairing.interfaces:
        print("  the decomposition produced no pair that can carry a connector:")
        print(f"    {pairing.reasons() or 'no pair of pieces is adjacent'}")
        return EXIT_NO

    spec = spec_from_parts(
        spec_id=f"decompose-{stamp}",
        parts=parts,
        interfaces=pairing.interfaces,
        requirement=f"{source.name} decomposed into {args.parts} printable pieces",
        # The build orientation stays undeclared unless the caller states it. Wall
        # thickness is orientation independent and is still measured; overhang is not
        # evaluated, and the report says `printable: partial` rather than claiming a
        # check that did not happen.
        constraints=FitConstraints(print_orientation_deg=_orientation(args.orientation)),
        provenance={
            "decomposition": decomposition.provenance,
            "source_mesh": str(source),
        },
    )

    store_parts(decomposition.parts, out)
    design = materialize(spec, out)

    certificates: list = []
    notes: dict[str, str] = {}
    cost = None
    if not args.no_verify:
        from pah.tools.verifiers.design_cascade import (  # noqa: PLC0415
            DesignVerifierCascade,
            default_registry,
        )

        cascade = DesignVerifierCascade(registry=default_registry(), root=out)
        certificates = cascade.check(design.assembly)
        notes = cascade.notes
        cost = cascade.spent

    from pah.app.runtime import _verdicts  # noqa: PLC0415

    verdicts = _verdicts(certificates, notes, out / "geometry")

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
            "interfaces": len(spec.interfaces),
            "clearance_mm": args.clearance,
            "verified": not args.no_verify,
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
    return EXIT_OK if verdicts["assemblable"]["ok"] and verdicts["printable"]["ok"] else EXIT_NO


def _print(report: dict, manifest: dict, out: Path) -> None:
    run = report["run"]
    print(f"pah decompose · {run['parts']} parts / {run['interfaces']} interfaces · {run['seconds']}s")
    print(f"  source {run['source_mesh']}  ({run['decomposer'].get('method')}, axis {run['decomposer'].get('axis')})")
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
        mark = "是" if verdict["ok"] else "否"
        extra = f"  未评估: {', '.join(verdict['not_evaluated'])}" if verdict["not_evaluated"] else ""
        print(f"  {name:10s} {mark} ({verdict['validated']}){extra}")
        if verdict["blocking"]:
            print(f"             blocking: {', '.join(verdict['blocking'])}")
    for code in report["certificates"]:
        if code["family"] != "environment":
            print(f"  evidence  {code['code']:24s} {code['locus'].get('interface', '')}")
    print(f"  artifacts {len(manifest['artifacts'])} file(s) under {out}/print")
    print(f"  report    {out / 'report.json'}")
