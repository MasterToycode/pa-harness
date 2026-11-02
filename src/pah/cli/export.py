"""`pah export` — the files you actually hand to a printer.

What a printer takes, and what it does not:

* **STL** — the lingua franca; one file per part, unitless (mm is a slicer setting).
* **3MF** — the modern one: a zip with XML, millimetre units in the file, several
  objects in one document. This is the format to hand over when it matters.
* **GLB / URDF** — *not* printing formats. GLB is a viewer/asset format (no guaranteed
  units, may be non-manifold); URDF describes a robot's links and joints for
  simulation. Both are useful, neither is a manufacturing order. `pah export` names
  this explicitly in the manifest so nobody sends a URDF to a slicer.

Why the gate reads `validated` and not just `ok`
-----------------------------------------------
The audit (`docs/remediation-plan.md` P0-2) found the previous rule —

    printable = not problems,  problems = {not watertight, no volume, 3mf skipped,
                                           printable.ok is false}

— so a run whose report said `validated: partial` (no build orientation declared, so
overhang was never evaluated) exported `printable: true`. That is the most direct
overclaim the harness had: the file that says "this goes to a printer" was written by
a check that had not run.

So a claim is exportable only when it is **`ok is true`, `validated == "full"`, and
`not_evaluated` is empty**. Unknown (`ok: null`) is refused the same way a defect is,
because "a check crashed" is not a licence to print. `--for-print` turns the rule into
an exit code and, crucially, decides *before* writing anything: a refused export
leaves no stray STL that someone could mistake for a go-ahead.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

EXIT_OK = 0
EXIT_NO = 1
EXIT_CRASH = 3

_CLAIMS = ("printable", "assemblable")


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
    export.add_argument(
        "--for-print",
        action="store_true",
        help=(
            "strict: exit 3 and write nothing unless the report fully validates both "
            "claims (printable needs ok && validated=='full' && no not-evaluated checks)"
        ),
    )
    export.add_argument("--json", action="store_true", help="machine-readable output")
    export.set_defaults(handler=_export)


def _read_report(run: Path) -> tuple[dict, str | None, dict]:
    """`(report, sha256-of-its-bytes, verdicts)`; a missing report is a not-a-pass."""
    path = run / "report.json"
    if not path.exists():
        return {}, None, {}
    raw = path.read_bytes()
    try:
        report = json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}, hashlib.sha256(raw).hexdigest(), {}
    return report, hashlib.sha256(raw).hexdigest(), report.get("verdicts") or {}


def _claim(claim: dict, *, meshes_closed: bool, name: str) -> dict:
    """Whether one verdict is strong enough to put a product claim on a manifest."""
    missing = sorted(claim.get("not_evaluated") or [])
    blocking = list(claim.get("blocking") or [])
    validated = claim.get("validated")
    ok = claim.get("ok")
    reasons: list[str] = []
    if ok is None:
        reasons.append("the report says unknown: a check could not run")
    elif ok is False:
        reasons.append(f"the report says not {name}")
    for code in blocking:
        # The code, not just a sentence: it is the thing a repair operator and a
        # person both act on, and dropping it made the old test's expectation
        # (`insufficient_material` in problems) silently disappear behind prose.
        reasons.append(f"report says not {name}: {code}")
    if validated != "full":
        reasons.append(f"verified only '{validated}', not 'full'")
    if missing:
        reasons.append("not evaluated: " + ", ".join(missing))
    if not meshes_closed:
        reasons.append("the exported meshes are not all closed solids")
    return {
        "ok": bool(ok is True and validated == "full" and not missing and meshes_closed),
        "validated": validated,
        "not_evaluated": missing,
        "blocking": blocking,
        "reasons": reasons,
    }


def export_run(
    run: str | Path,
    *,
    formats: tuple[str, ...] = ("3mf", "stl"),
    out: str | Path | None = None,
    strict: bool = False,
) -> dict:
    """Write the printable files for one run; returns the manifest, never raises.

    Shared by `pah export` and `pah build`, so the two cannot disagree about what a
    printable deliverable is — the disagreement the audit found was inside this
    function, between the file checks and the report's own claim.
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

    parts: dict[str, trimesh.Trimesh] = {}
    for path in sorted((run / "geometry").glob("*.stl")):
        parts[path.stem] = trimesh.load(path, force="mesh")
    if not parts:
        raise FileNotFoundError(f"no readable STL under {run / 'geometry'}")

    report, report_sha, verdicts = _read_report(run)
    spec_path = run / "spec.json"
    spec: dict = {}
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
        bounds = mesh.bounds.tolist() if hasattr(mesh.bounds, "tolist") else mesh.bounds
        part_rows[part_id] = {
            "stl": str((run / "geometry" / f"{part_id}.stl").resolve()),
            "watertight": watertight,
            "volume_mm3": round(volume, 3),
            "bounds_mm": [[round(float(v), 3) for v in row] for row in bounds],
            "faces": int(len(mesh.faces)),
        }
    meshes_closed = not problems

    readiness = {
        "printable": _claim(
            verdicts.get("printable") or {}, meshes_closed=meshes_closed, name="printable"
        ),
        "assembly_ready": _claim(
            verdicts.get("assemblable") or {}, meshes_closed=meshes_closed, name="assemblable"
        ),
    }
    # `problems` is what a person reads. It has to contain both the file-level trouble
    # and the report's own blockers: a masked reason is how "not printable" turns into
    # "no idea why".
    seen = set(problems)
    for reason in readiness["printable"]["reasons"]:
        if reason not in seen:
            seen.add(reason)
            problems.append(reason)
    # The file checks are part of the claim, not a separate list somebody has to
    # remember to read: a non-watertight STL makes both claims false and says why.
    for row in readiness.values():
        if problems:
            row["ok"] = False

    artifacts: list[str] = []
    wrote = not (strict and not readiness["printable"]["ok"])
    if wrote:
        out.mkdir(parents=True, exist_ok=True)
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

    manifest = {
        "schema": "pah-print-manifest/2",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "units": "mm",
        "note": (
            "STL and 3MF are printing formats. GLB is a viewer asset format and URDF is a "
            "robot description — neither is a manufacturing order, which is why this "
            "command refuses them. `printable` and `assembly_ready` are true only when "
            "the report fully validated that claim: ok, validated=='full', and no "
            "not-evaluated checks."
        ),
        "parts": part_rows,
        "artifacts": artifacts,
        "wrote_files": wrote,
        "print_orientation_deg": (spec.get("constraints") or {}).get("print_orientation_deg"),
        "generator": (report.get("run") or {}).get("generator"),
        "report_sha256": report_sha,
        "verdicts": verdicts,
        "readiness": readiness,
        "problems": problems,
        "printable": readiness["printable"]["ok"],
        "assembly_ready": readiness["assembly_ready"]["ok"],
    }
    if wrote:
        manifest_path = out / "print-manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", "utf-8"
        )
        artifacts.append(str(manifest_path.resolve()))

    manifest["_run"] = str(run)
    return manifest


def _export(args: argparse.Namespace) -> int:
    formats = tuple(item.strip().lower() for item in str(args.formats).split(",") if item.strip())
    try:
        manifest = export_run(
            args.run, formats=formats, out=args.out, strict=bool(args.for_print)
        )
    except (FileNotFoundError, ValueError) as error:
        print(f"pah export: {error}")
        return EXIT_CRASH
    if args.json:
        print(json.dumps(manifest, ensure_ascii=False, indent=1))
    else:
        print(f"pah export · {manifest['_run']}")
        for path in manifest["artifacts"]:
            print(f"  {path}")
        print(
            f"  units {manifest['units']}  orientation {manifest['print_orientation_deg']}  "
            f"generator {manifest['generator']}"
        )
        for name in ("printable", "assembly_ready"):
            row = manifest["readiness"][name]
            mark = "是" if row["ok"] else "否"
            if row["reasons"]:
                print(f"  {name:14s} {mark}")
                for reason in row["reasons"]:
                    print(f"     · {reason}")
            else:
                print(f"  {name:14s} {mark} ({row['validated']})")
        if manifest["problems"]:
            print("\n  不满足的条件：")
            for problem in manifest["problems"]:
                print(f"   · {problem}")
        if not manifest["wrote_files"]:
            print("\n  严格模式：未写出任何文件（先判后写）。")
    if args.for_print and not manifest["readiness"]["printable"]["ok"]:
        return EXIT_CRASH
    return EXIT_OK if manifest["printable"] else EXIT_NO
