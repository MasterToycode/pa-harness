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

`--slice`: the manufacturing layer
----------------------------------
Mesh checks cannot answer "can this be manufactured": a slicer can. `--slice` runs a
dry slice of every part (`pah.tools.slicer`) and appends the result to the manifest
under `slicing`, then ANDs it into `printable`. Two consequences worth stating:

* **A missing slicer is not a pass.** The whole point of the audit slice is evidence,
  so "no PrusaSlicer/OrcaSlicer/CuraEngine on this machine" becomes
  `printable: false` with the reason in `problems` — never a silent skip.
* **`--for-print` decides after slicing too**, because a failed dry run is as much a
  refusal as a partial report. The dry-run G-code lands in a scratch directory first
  and is only kept under `<out>/gcode/` when the export actually writes, so a refused
  strict export still leaves the output directory untouched.

A passed dry run is evidence about *geometry under a profile*, not a promise that the
part prints well; the manifest says which backend produced the numbers and which
fields it did not report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
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
    export.add_argument(
        "--slice",
        dest="slice",
        action="store_true",
        help=(
            "dry-run a slicer on every part and append the evidence to the manifest; "
            "a missing slicer or a failed slice makes printable false"
        ),
    )
    export.add_argument(
        "--slice-profile",
        dest="slice_profile",
        default=None,
        help="slicer settings file for --slice (PrusaSlicer/OrcaSlicer --load, CuraEngine -j)",
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
    slice: bool = False,
    slice_profile: str | Path | None = None,
) -> dict:
    """Write the printable files for one run; returns the manifest, never raises.

    Shared by `pah export` and `pah build`, so the two cannot disagree about what a
    printable deliverable is — the disagreement the audit found was inside this
    function, between the file checks and the report's own claim.

    `slice=True` appends a slicer dry run to the gate. It is opt-in because it needs an
    external program and can take minutes; when it is requested, a missing slicer is a
    refusal (`printable: false`), not a skip.
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
    slicing: dict | None = None
    scratch: tempfile.TemporaryDirectory | None = None
    try:
        if slice:
            # The dry run happens *before* the write decision so `--for-print` can
            # refuse on it, and it slices the run's own STL files — the same solids
            # that would be exported. Nothing is placed in `out` until the gate passes.
            scratch = tempfile.TemporaryDirectory(prefix="pah-slice-")
            slicing = _slice_parts(part_rows, Path(scratch.name), profile=slice_profile)
            for problem in slicing["problems"]:
                if problem not in problems:
                    problems.append(problem)
        # The manufacturing layer sits *on top of* the existing gate: mesh checks and
        # the report's own claim must still hold, and now a slicer must also agree.
        printable_ok = readiness["printable"]["ok"] and (slicing is None or bool(slicing["ok"]))
        wrote = not (strict and not printable_ok)
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
            if slicing is not None:
                _persist_gcode(slicing, out / "gcode")

        manifest = {
            "schema": "pah-print-manifest/3",
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "units": "mm",
            "note": (
                "STL and 3MF are printing formats. GLB is a viewer asset format and URDF is "
                "a robot description — neither is a manufacturing order, which is why this "
                "command refuses them. `printable` and `assembly_ready` are true only when "
                "the report fully validated that claim: ok, validated=='full', and no "
                "not-evaluated checks. With --slice, `printable` also requires a slicer "
                "dry run to succeed; `slicing` records which backend produced the "
                "numbers, and a failed or missing slicer keeps `printable` false."
            ),
            "parts": part_rows,
            "artifacts": artifacts,
            "wrote_files": wrote,
            "print_orientation_deg": (spec.get("constraints") or {}).get("print_orientation_deg"),
            "generator": (report.get("run") or {}).get("generator"),
            "report_sha256": report_sha,
            "verdicts": verdicts,
            "readiness": readiness,
            "slicing": slicing,
            "problems": problems,
            "printable": printable_ok,
            "assembly_ready": readiness["assembly_ready"]["ok"],
        }
        if wrote:
            manifest_path = out / "print-manifest.json"
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", "utf-8"
            )
            artifacts.append(str(manifest_path.resolve()))
    finally:
        if scratch is not None:
            scratch.cleanup()

    manifest["_run"] = str(run)
    return manifest


def _slice_parts(
    part_rows: dict[str, dict],
    scratch: Path,
    *,
    profile: str | Path | None = None,
) -> dict:
    """Dry-slice every part into `scratch`; returns the manifest `slicing` block.

    The shape is the manifest's, not `SliceResult`'s, because a manifest has to say
    *why* it is false in one place (`problems`) and still keep the per-part evidence
    that succeeded. A missing slicer produces `backend: None` and one problem — the
    audit's rule is that "not installed" and "rejected" both make `printable` false,
    and the sentence explains which one happened.

    `profile` is recorded, not hashed: a dry run with defaults and one with a project
    profile are different experiments, and a manifest that did not say which would let
    two incomparable numbers sit side by side.
    """
    from pah.tools.slicer import SlicerMissing, find_slicer, slice_mesh  # noqa: PLC0415

    try:
        slicer = find_slicer()
    except SlicerMissing as error:
        return {
            "backend": None,
            "profile": str(profile) if profile is not None else None,
            "per_part": {},
            "ok": False,
            "problems": [str(error)],
            "requested": True,
        }

    per_part: dict[str, dict] = {}
    problems: list[str] = []
    for part_id, row in part_rows.items():
        try:
            result = slice_mesh(row["stl"], scratch, profile=profile, slicer=slicer)
        except SlicerMissing as error:  # the executable vanished between discovery and use
            # Same row shape as a real result, with the numbers explicitly unknown: a
            # consumer must not have to guess whether a missing key means "not
            # reported" or "the schema changed".
            per_part[part_id] = {
                "ok": False,
                "backend": slicer.backend,
                "gcode_path": None,
                "seconds_estimate": None,
                "filament_mm": None,
                "filament_g": None,
                "support_used": None,
                "layers": None,
                "empty_layers": 0,
                "problems": [str(error)],
            }
            problems.append(f"{part_id}: {error}")
            continue
        per_part[part_id] = result.to_dict()
        problems.extend(f"{part_id}: {problem}" for problem in result.problems)
    return {
        "backend": slicer.backend,
        "profile": str(profile) if profile is not None else None,
        "per_part": per_part,
        "ok": bool(per_part) and all(item["ok"] for item in per_part.values()),
        "problems": problems,
        "requested": True,
    }


def _persist_gcode(slicing: dict, target_dir: Path) -> None:
    """Move the dry run's G-code under `<out>/gcode/` and repoint the manifest rows.

    The scratch directory dies with this function, so a path left pointing at it would
    be a broken promise in a manifest. Copying is cheap and keeps the evidence next to
    the files it belongs to.
    """
    per_part = slicing.get("per_part") or {}
    if not per_part:
        return
    target_dir.mkdir(parents=True, exist_ok=True)
    for part_id, row in per_part.items():
        source = row.get("gcode_path")
        if not source or not Path(source).exists():
            continue
        target = target_dir / f"{part_id}.gcode"
        shutil.copyfile(source, target)
        row["gcode_path"] = str(target.resolve())


def _export(args: argparse.Namespace) -> int:
    formats = tuple(item.strip().lower() for item in str(args.formats).split(",") if item.strip())
    try:
        manifest = export_run(
            args.run,
            formats=formats,
            out=args.out,
            strict=bool(args.for_print),
            slice=bool(args.slice),
            slice_profile=args.slice_profile,
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
        slicing = manifest.get("slicing")
        if slicing is not None:
            if slicing["ok"]:
                print(f"  slicing       {slicing['backend']}")
                for part_id, row in slicing["per_part"].items():
                    print(
                        f"     · {part_id}: {row['seconds_estimate']}s, "
                        f"{row['filament_mm']} mm filament, {row['layers']} layers, "
                        f"support={row['support_used']}"
                    )
            else:
                print(f"  slicing       否 ({slicing['backend'] or '没有切片器'})")
        if manifest["problems"]:
            print("\n  不满足的条件：")
            for problem in manifest["problems"]:
                print(f"   · {problem}")
        if not manifest["wrote_files"]:
            print("\n  严格模式：未写出任何文件（先判后写）。")
    if args.for_print and not manifest["printable"]:
        return EXIT_CRASH
    return EXIT_OK if manifest["printable"] else EXIT_NO
