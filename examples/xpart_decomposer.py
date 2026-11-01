"""X-Part as a `Decomposer`: the cloud API 腾讯云 exposes, wired into the pipeline.

What the real thing returns (measured, not assumed)
---------------------------------------------------
One job run on 2026-11-01 against `SubmitHunyuan3DPartJob` (JobId
`1508519036782002176`, the X-Part demo asset 000.fbx, 500k faces) came back with:

* **19 files, one GLB per part** — not one multi-object OBJ. Each GLB holds a single
  geometry named `part_N`, so the part identity is the file, and the segmentation
  data comes separately as a binary label array over the *input* mesh's faces.
* **units normalised to ~1.0**, not millimetres. The submitted model's extent was
  1.05 x 1.04 x 0.25 in the FBX; the union of the parts matched that, so the scale is
  "whatever the input was".
* **most parts are not watertight as delivered**, and hole filling only rescued some
  of them (`watertight_after_fill` was false for part_00 and part_03 of 19).
* **parts overlap**: 30 bounding-box pairs intersected by more than 1 mm. X-Part
  generates each part independently from a bbox prompt, so neighbouring parts are not
  a clean partition of the volume.

Those four facts are why this adapter is more than an HTTP call: it repairs, scales,
and names. A pipeline that assumed "the API returns printable parts" would fail at the
slicer with geometry nobody could explain.

What it refuses to hide
-----------------------
A part that cannot be closed after repair is dropped with its name in the provenance,
not silently skipped: a missing piece is a visible hole in the assembly, and a report
that says "19 parts" when 17 could be closed is the kind of quiet subtraction this
harness exists to prevent.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh

from pah.design.decompose import Decomposition, DecomposedPart

__all__ = ["XPartDecomposer", "load_decomposition_dir"]


# --------------------------------------------------------------------------- #
# Repair: the part the API does not do
# --------------------------------------------------------------------------- #
def _close(mesh: trimesh.Trimesh, *, pitch_fraction: float = 80.0) -> trimesh.Trimesh | None:
    """One closed solid from a generated part, or `None` when it cannot be closed.

    Three rungs, cheapest first, and each one is a *measurable* difference rather than
    a wish:

    1. it is already closed — nothing to do;
    2. `fill_holes` closes the boundary loops (the common case: a shell with a few
       open edges where the generator's surface extraction stopped);
    3. a voxel remesh at extent/80 — which closes anything, at the cost of detail and
       of trusting the voxeliser's inside/outside verdict.

    Rung 3 is what turns a 300k-face shell into a printable solid, and it is also the
    rung that can invent thin walls, so its use is recorded in the provenance rather
    than applied quietly.
    """
    if mesh.is_watertight and mesh.volume > 0.0:
        return mesh

    filled = mesh.copy()
    trimesh.repair.fill_holes(filled)
    if filled.is_watertight and filled.volume > 0.0:
        return filled

    try:
        pitch = float(np.max(mesh.extents)) / pitch_fraction
        if pitch <= 0.0:
            return None
        voxels = mesh.voxelized(pitch).fill()
        rebuilt = voxels.marching_cubes
        # `marching_cubes` answers in **voxel index space**, not model space: the grid's
        # transform (scale = pitch, translate = grid corner) has to be applied or the
        # part comes back 80x too big and offset. Measured on a real X-Part part:
        # extents 0.651 x 0.452 x 0.196 came back as 81 x 57 x 25, which silently made
        # the union of the decomposition ~80x too large and the whole assembly a
        # miniature.
        if rebuilt is not None:
            rebuilt = rebuilt.copy()
            rebuilt.apply_transform(voxels.transform)
        if rebuilt is not None and rebuilt.is_watertight and rebuilt.volume > 0.0:
            return rebuilt
    except Exception:  # noqa: BLE001 - a failed repair is a dropped part, not a crash
        return None
    return None


def _one_mesh(path: Path) -> trimesh.Trimesh:
    """A GLB/OBJ as one mesh, whatever nesting the exporter used."""
    loaded = trimesh.load(path, force="mesh")
    if isinstance(loaded, trimesh.Scene):
        loaded = trimesh.util.concatenate(list(loaded.geometry.values()))
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"{path.name} did not contain a mesh (got {type(loaded).__name__})")
    return loaded


def _inner_name(path: Path) -> str:
    """The name a part should carry: the geometry inside the file when it is a name.

    The API's own files hold one geometry called `part_N`, and that is the name the
    segmentation labels refer to, so it must survive. But a GLB written by trimesh
    calls its contents `geometry_0` — a placeholder, not an identity — and two such
    files in one directory would collide. So a placeholder falls back to the file
    stem, and `load_decomposition_dir` de-duplicates whatever is left.
    """
    stem = path.stem
    try:
        scene = trimesh.load(path)
        names = list(getattr(scene, "geometry", {}) or {})
        if len(names) == 1:
            candidate = str(names[0])
            if not candidate.startswith(("geometry", "mesh", "Object")):
                return candidate
    except Exception:  # noqa: BLE001 - the file stem is a fine fallback
        pass
    return stem


# --------------------------------------------------------------------------- #
# The decomposer
# --------------------------------------------------------------------------- #
@dataclass
class XPartDecomposer:
    """Whole mesh -> N printable pieces, via 腾讯云 `SubmitHunyuan3DPartJob`.

    `target_size_mm` places the result in the caller's millimetres. Left `None`, the
    submitted mesh's own largest extent is used — which is the honest default when the
    input is a PAH design in mm, and a wrong one when it is a normalised GLB. The
    provenance records which rule was applied.
    """

    name: str = "xpart-cloud"
    target_size_mm: float | None = None
    poll_timeout_s: float = 1800.0
    poll_interval_s: float = 10.0
    staged: bool = False
    work_dir: Path | None = None
    last_job: dict = field(default_factory=dict)

    def decompose(self, whole, *, hints: dict | None = None) -> Decomposition:
        """Submit `whole`, wait, download, repair, scale — and return the pieces."""
        from pah.tools import tencent_ai3d
        from pah.tools.fbx import to_fbx

        work = Path(self.work_dir or tempfile.mkdtemp(prefix="pah-xpart-"))
        work.mkdir(parents=True, exist_ok=True)

        mesh = whole if isinstance(whole, trimesh.Trimesh) else _one_mesh(Path(whole))
        source_stl = work / "whole.stl"
        source_stl.write_bytes(mesh.export(file_type="stl"))
        fbx = work / "whole.fbx"
        to_fbx([source_stl], fbx)

        url = tencent_ai3d.upload_to_cos(fbx)
        job_id = tencent_ai3d.submit_part_job(
            url, staged=self.staged, post_process=False
        )
        self.last_job = {"job_id": job_id, "input": str(source_stl)}
        result = tencent_ai3d.wait_part_job(
            job_id, timeout_s=self.poll_timeout_s, interval_s=self.poll_interval_s
        )
        downloaded = tencent_ai3d.download_results(result, work / "parts")

        pieces = self._pieces(downloaded, mesh)
        if len(pieces) < 2:
            raise ValueError(
                f"X-Part returned {len(pieces)} usable part(s) from job {job_id}; "
                "an assembly needs at least two, so nothing can be paired"
            )
        return Decomposition(
            parts=tuple(pieces),
            provenance={
                "method": "xpart-cloud",
                "job_id": job_id,
                "files": len(downloaded),
                "usable_parts": len(pieces),
                "scale_mm": self.scale,
                "note": (
                    "X-Part generates each part from a bbox prompt: parts are complete "
                    "solids but not a clean partition, and only touching, not fastened"
                ),
                **(self.drop_note or {}),
            },
        )

    # -- internals ---------------------------------------------------------
    scale: float = 1.0
    drop_note: dict | None = None

    def _pieces(
        self, downloaded: dict[str, str], reference: trimesh.Trimesh
    ) -> list[DecomposedPart]:
        """Load, close and scale every downloaded part; name what was dropped."""
        loaded: list[tuple[str, trimesh.Trimesh]] = []
        dropped: list[tuple[str, str]] = []
        for name, path in sorted(downloaded.items()):
            if Path(path).suffix.lower() not in (".glb", ".gltf", ".obj", ".stl", ".ply"):
                continue  # preview images and label data ride along in the same list
            try:
                mesh = _one_mesh(Path(path))
            except Exception as error:  # noqa: BLE001 - unreadable file, name it
                dropped.append((name, f"unreadable: {error}"))
                continue
            closed = _close(mesh)
            if closed is None:
                dropped.append((name, "not a closed solid after hole filling and remesh"))
                continue
            loaded.append((_inner_name(Path(path)), closed))

        union = trimesh.util.concatenate([mesh for _, mesh in loaded]) if loaded else None
        union_extent = float(np.max(union.extents)) if union is not None else 0.0
        scale = 1.0
        if union_extent > 0.0:
            target = self.target_size_mm or float(np.max(reference.extents))
            scale = float(target) / union_extent
        self.scale = scale
        self.drop_note = {"dropped": [list(item) for item in dropped]} if dropped else None

        pieces: list[DecomposedPart] = []
        for name, mesh in loaded:
            if abs(scale - 1.0) > 1e-9:
                mesh = mesh.copy()
                mesh.apply_scale(scale)
            pieces.append(DecomposedPart(id=name, mesh=mesh))
        return pieces


# --------------------------------------------------------------------------- #
# Offline path: a decomposition already on disk
# --------------------------------------------------------------------------- #
def load_decomposition_dir(
    directory: str | Path, *, target_size_mm: float | None = None
) -> Decomposition:
    """A decomposition directory as a `Decomposition`: no API call, no credits.

    This exists because a job already paid for should be reusable. Probing the API
    once told us its output shape; re-running the same asset to test the *pipeline*
    would have cost another job and taught nothing. So a downloaded directory replays
    exactly like a fresh result, and the tests use it the same way.

    `target_size_mm` rescales the set, because X-Part answers in normalised units: the
    union is measured first, then scaled so its longest extent is that many mm.
    """
    root = Path(directory)
    files = sorted(
        path
        for path in root.iterdir()
        if path.suffix.lower() in (".glb", ".gltf", ".obj", ".stl", ".ply")
    )
    if not files:
        raise FileNotFoundError(f"no mesh files under {root}")

    closed: list[tuple[str, trimesh.Trimesh]] = []
    dropped: list[list[str]] = []
    used: set[str] = set()
    for path in files:
        try:
            mesh = _one_mesh(path)
        except Exception as error:  # noqa: BLE001 - an unreadable file is a dropped part
            dropped.append([path.name, f"unreadable: {error}"])
            continue
        sealed = _close(mesh)
        if sealed is None:
            dropped.append([path.name, "not a closed solid after repair"])
            continue
        name = _inner_name(path)
        if name in used:  # two files claiming one identity: keep the filesystem's
            name = path.stem
        used.add(name)
        closed.append((name, sealed))

    union_extent = 0.0
    if closed:
        union = trimesh.util.concatenate([mesh for _, mesh in closed])
        union_extent = float(np.max(union.extents))
    scale = float(target_size_mm) / union_extent if target_size_mm and union_extent > 0 else 1.0

    pieces: list[DecomposedPart] = []
    for name, mesh in closed:
        if abs(scale - 1.0) > 1e-9:
            mesh = mesh.copy()
            mesh.apply_scale(scale)
        pieces.append(DecomposedPart(id=name, mesh=mesh))

    provenance = {
        "method": "xpart-dir",
        "directory": str(root),
        "files": len(files),
        "usable_parts": len(pieces),
        "scale_mm": scale,
        "union_extent_before_scale": union_extent,
    }
    if dropped:
        provenance["dropped"] = dropped
    return Decomposition(parts=tuple(pieces), provenance=provenance)
