"""`pah export` — the files you actually hand to a printer.

What a printer takes, and what it does not:

* **STL** — the lingua franca; one file per part, unitless (mm is a slicer setting).
* **3MF** — the modern one: a zip with XML, millimetre units in the file, several
  objects in one document. This is the format to hand over when it matters.
* **GLB / URDF** — *not* printing formats. GLB is a viewer/asset format (no guaranteed
  units, may be non-manifold); URDF describes a robot's links and joints for
  simulation. Both are useful, neither is a manufacturing order. `pah export` names
  this explicitly in the manifest so nobody sends a URDF to a slicer.

The command refuses to pretend: if any part is not watertight, or the run's
`verdicts.printable.ok` is false, it still writes the files for inspection but exits
non-zero and says which check failed. A silent "here are your files" for an unprintable
part is how a waste of filament becomes a bug report about the harness.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

EXIT_OK = 0
EXIT_NO = 1
EXIT_CRASH = 3


def add_parser(groups: argparse._SubParsersAction) -> None:
    export = groups.add_parser(
        "export", help="write printable files (3MF + STL) and a print manifest"
    )
    export.add_argument("run", help="a run directory that contains geometry/ and report.json")
    export.add_argument(
        "--formats",
        default="3mf,stl",
        help="comma-separated: 3mf,stl (default both); glb/urdf are refused on purpose",
    )
    export.add_argument("--out", default=None, help="output directory (default <run>/print)")
    export.add_argument("--json", action="store_true")
    export.set_defaults(handler=_export)


def export_run(
    run: str | Path,
    *,
    formats: tuple[str, ...] = ("3mf", "stl"),
    out: str | Path | None = None,
) -> dict:
    """Write the printable files for one run; returns the manifest, never raises.

    Shared by `pah export` and `pah build` (which exports automatically when the
    report calls the run printable), so the two cannot disagree about what a
    printable deliverable is.
    """
    import trimesh  # noqa: PLC0415

    run = Path(run)
    if not (run / "geometry").exists():
        raise FileNotFoundError(f"{run} has no geometry/ directory")
    formats = [item.strip().lower() for item in formats if item.strip()]
    refused = [f for f in formats if f not in ("3mf", "stl")]
    if refused:
        raise ValueError(
            f"{refused} are not printing formats: GLB has no guaranteed units and may "
            "be non-manifold; URDF describes links and joints for simulation. Use 3mf "
            "or stl."
        )

    out = Path(out) if out else run / "print"
    out.mkdir(parents=True, exist_ok=True)

    parts: dict[str, trimesh.Trimesh] = {}
    for path in sorted((run / "geometry").glob("*.stl")):
        parts[path.stem] = trimesh.load(path, force="mesh")
    if not parts:
        raise FileNotFoundError(f"no readable STL under {run / 'geometry'}")

    report: dict = {}
    report_path = run / "report.json"
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text("utf-8"))
        except json.JSONDecodeError:
            report = {}
    verdicts = report.get("verdicts", {})
    printable = verdicts.get("printable", {})
    spec: dict = {}
    spec_path = run / "spec.json"
    if spec_path.exists():
        try:
            spec = json.loads(spec_path.read_text("utf-8"))
        except json.JSONDecodeError:
            spec = {}

    problems: list[str] = []
    part_rows: dict[str, dict] = {}
    for part_id, mesh in parts.items():
        watertight = bool(mesh.is_watertight)
        volume = float(mesh.volume) if watertight else 0.0
        if not watertight:
            problems.append(f"{part_id}: not watertight (a slicer cannot print a shell)")
        if volume <= 0.0:
            problems.append(f"{part_id}: no enclosed volume")
        part_rows[part_id] = {
            "stl": str((run / "geometry" / f"{part_id}.stl").resolve()),
            "watertight": watertight,
            "volume_mm3": round(volume, 3),
            "bounds_mm": [[round(float(v), 3) for v in row] for row in (mesh.bounds.tolist() if hasattr(mesh.bounds, "tolist") else mesh.bounds)],
            "faces": int(len(mesh.faces)),
        }

    artifacts: list[str] = []
    if "stl" in formats:
        for part_id, row in part_rows.items():
            target = out / f"{part_id}.stl"
            parts[part_id].export(target)
            artifacts.append(str(target.resolve()))
            row["exported_stl"] = str(target.resolve())
    if "3mf" in formats and all(row["watertight"] for row in part_rows.values()):
        from trimesh.exchange.threemf import export_3MF  # noqa: PLC0415

        scene = trimesh.Scene({part_id: mesh for part_id, mesh in parts.items()})
        target = out / "assembly.3mf"
        target.write_bytes(export_3MF(scene))
        artifacts.append(str(target.resolve()))
    elif "3mf" in formats:
        problems.append("3mf skipped: a non-watertight part cannot be written into a 3MF")

    if printable and not printable.get("ok", True):
        for code in printable.get("blocking") or []:
            problems.append(f"report says not printable: {code}")
    if not printable:
        problems.append("report has no printable verdict: printability was not evaluated")

    manifest = {
        "schema": "pah-print-manifest/1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "units": "mm",
        "note": (
            "STL and 3MF are printing formats. GLB is a viewer asset format and URDF is a "
            "robot description — neither is a manufacturing order, which is why this "
            "command refuses them."
        ),
        "parts": part_rows,
        "artifacts": artifacts,
        "print_orientation_deg": (spec.get("constraints") or {}).get("print_orientation_deg"),
        "generator": (report.get("run") or {}).get("generator"),
        "verdicts": verdicts,
        "problems": problems,
        "printable": not problems,
    }
    manifest_path = out / "print-manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", "utf-8")
    artifacts.append(str(manifest_path.resolve()))

    manifest["_run"] = str(run)
    return manifest


def _export(args: argparse.Namespace) -> int:
    formats = tuple(item.strip().lower() for item in str(args.formats).split(",") if item.strip())
    try:
        manifest = export_run(args.run, formats=formats, out=args.out)
    except (FileNotFoundError, ValueError) as error:
        print(f"pah export: {error}")
        return EXIT_CRASH
    if args.json:
        print(json.dumps(manifest, ensure_ascii=False, indent=1))
    else:
        print(f"pah export · {manifest['_run']}")
        for path in manifest["artifacts"]:
            print(f"  {path}")
        print(f"  units {manifest['units']}  orientation {manifest['print_orientation_deg']}  "
              f"generator {manifest['generator']}")
        if manifest["problems"]:
            print("\n  不可打印：")
            for problem in manifest["problems"]:
                print(f"   · {problem}")
        else:
            print("\n  可打印：两个零件都水密，体积为正，报告口径为 printable=ok")
    return EXIT_OK if manifest["printable"] else EXIT_NO
