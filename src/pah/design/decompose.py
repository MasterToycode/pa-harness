"""Turning one whole object into the pieces you would actually print.

The port
--------
`Decomposer` is the seam the X-Part pipeline plugs into: a whole mesh in, N
watertight pieces out. Cloud X-Part, a local diffusion model, or the slab splitter
below are interchangeable behind it, because everything after this point reads a
`DesignSpec` and never asks where the geometry came from.

Why a local splitter ships with a cloud decomposer
--------------------------------------------------
Two reasons, both practical:

* **The pipeline has to be runnable without spending money.** Every step after the
  decomposition — pairing, machining, the six verifiers, export — is what this
  evening's work is about. If the only way to exercise it were a paid API call,
  the integration would be untested until the credits arrived, and a bug in the
  pairing code would look like a bug in the API client.
* **It is the honest control.** The slab splitter cuts a solid into pieces whose
  faces are *exact* complements — strictly better mating geometry than X-Part
  produces. So if a pipe of pieces still cannot be assembled, the cause is the
  connector layer, not the decomposition. That is a useful thing to be able to say.

What the splitter refuses
-------------------------
It does not pretend to be a semantic decomposition. It cuts along the longest axis
into equal slabs, and a gearbox comes out as slices rather than as a shaft and a
case. `Decomposition.provenance["method"]` says so, and the parts are named
`part_00`… because calling a slab "housing" would be a lie a report would repeat.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import trimesh

from pah.design.spec import PartSpec

__all__ = [
    "DecomposedPart",
    "Decomposer",
    "Decomposition",
    "check_store",
    "file_hash",
    "mesh_hash",
    "parts_from_decomposition",
    "split_slabs",
    "store_parts",
]

#: Parts are written here under a run directory, and read back by `materialize`.
PARTS_DIR = "parts"


@dataclass(frozen=True)
class DecomposedPart:
    """One piece a decomposer returned: its name and its solid, in world mm."""

    id: str
    mesh: trimesh.Trimesh


@dataclass(frozen=True)
class Decomposition:
    """Every piece plus what produced it.

    `provenance` is copied into the spec, so a report can say "these shapes came
    from job 1384…" without the reader having to trust a filename. It is not part of
    the content key — the geometry itself is, through each `PartSpec.source` — and
    that split is deliberate: who produced a mesh is not the same claim as what the
    mesh is.
    """

    parts: tuple[DecomposedPart, ...]
    provenance: dict = field(default_factory=dict)

    def part_ids(self) -> tuple[str, ...]:
        return tuple(part.id for part in self.parts)


@runtime_checkable
class Decomposer(Protocol):
    """A whole mesh in, the printable pieces out."""

    name: str

    def decompose(
        self, whole: trimesh.Trimesh, *, hints: dict | None = None
    ) -> Decomposition: ...


# --------------------------------------------------------------------------- #
# Identity and storage
# --------------------------------------------------------------------------- #
def file_hash(path: str | Path) -> str:
    """The content address of a blank **file** — what `PartSpec.source` records.

    Two rounds of testing said the mesh could not be the identity across a round
    trip. Triangles survive; the numbers in them do not: STL stores float32, and at
    120 mm that is a 4e-6 mm wobble, so a canonical hash at 1e-6 disagreed for all 19
    parts of a real X-Part job while the triangles were geometrically identical.

    Hashing the file is exact by construction and states the right thing: the spec
    names *this artefact*, and `check_store` re-reads the same bytes. A coarser
    coordinate hash would only make the mismatch rarer, not impossible.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()[:32]


def mesh_hash(mesh: trimesh.Trimesh) -> str:
    """A content address for a mesh *object*: its triangle soup, canonically ordered.

    Used for comparison and reporting (are these two solids the same), not for
    `PartSpec.source` — see `file_hash` for why the spec names a file instead.
    Rounds to 1e-6 mm, orders each triangle's vertices, orders the triangles, and
    includes the face count. Volume is deliberately absent: it is derived from the
    same triangles and STL's float32 coordinates move it in the sixth decimal.
    """
    triangles = np.round(np.asarray(mesh.triangles, dtype=float), 6)
    within = np.lexsort(
        (triangles[:, :, 2], triangles[:, :, 1], triangles[:, :, 0]), axis=1
    )
    triangles = np.take_along_axis(triangles, within[:, :, None], axis=1)
    flat = triangles.reshape(len(triangles), 9)
    order = np.lexsort(flat.T[::-1])
    digest = hashlib.sha256()
    digest.update(flat[order].tobytes())
    digest.update(f"|faces={len(mesh.faces)}".encode())
    return "sha256:" + digest.hexdigest()[:32]


def store_parts(parts: tuple[DecomposedPart, ...], out_dir: Path) -> dict[str, Path]:
    """Write each piece's blank to `<out_dir>/parts/<id>.stl`, return the paths.

    The blanks are kept because the spec records only their *hash*: a materialiser
    asked to rebuild an episode from `spec.json` needs the meshes somewhere, and
    re-deriving them from a paid API call is not a rebuild.
    """
    target = Path(out_dir) / PARTS_DIR
    target.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for part in parts:
        path = target / f"{part.id}.stl"
        path.write_bytes(part.mesh.export(file_type="stl"))
        paths[part.id] = path
    return paths


def check_store(spec, out_dir: Path) -> dict[str, Path]:
    """The blanks a spec names, loaded from `<out_dir>/parts/`, hash-checked.

    The hash check is the point: an edited STL that no longer matches
    `PartSpec.source` would be silently printed, and every verifier would then be
    judging triangles the design never described.
    """
    root = Path(out_dir) / PARTS_DIR
    found: dict[str, Path] = {}
    problems: list[str] = []
    for part in spec.parts:
        if not part.source:
            continue
        path = root / f"{part.id}.stl"
        if not path.exists():
            problems.append(f"{part.id}: no blank at {path}")
            continue
        actual = file_hash(path)
        if actual != part.source:
            problems.append(
                f"{part.id}: blank at {path} hashes {actual}, the spec says {part.source}"
            )
            continue
        found[part.id] = path
    if problems:
        raise FileNotFoundError(
            "the decomposed blanks this spec names are missing or changed (an "
            "episode cannot be rebuilt from geometry it does not have): "
            + "; ".join(problems)
        )
    return found


# --------------------------------------------------------------------------- #
# The local splitter
# --------------------------------------------------------------------------- #
def _watertight_copy(mesh: trimesh.Trimesh, what: str) -> trimesh.Trimesh:
    """`mesh` if it is closed, a hole-filled attempt if not, or a refusal."""
    if mesh.is_watertight and mesh.volume > 0.0:
        return mesh
    repaired = mesh.copy()
    trimesh.repair.fill_holes(repaired)
    if repaired.is_watertight and repaired.volume > 0.0:
        return repaired
    raise ValueError(
        f"{what} is not a closed solid (watertight={mesh.is_watertight}, "
        f"volume={mesh.volume!r}); a decomposition of an open shell cannot be "
        "printed and cannot be verified, so it is refused here rather than three "
        "stages later"
    )


def split_slabs(
    whole: trimesh.Trimesh,
    *,
    count: int = 2,
    axis: int | None = None,
) -> Decomposition:
    """Cut `whole` into `count` equal slabs along its longest axis.

    Each slab is the *intersection* of the whole with a slab box, not the box
    itself: the pieces' outer surfaces stay the object's real surfaces, and the two
    cut faces of a pair are exact complements. That is the property that makes the
    pieces assemble geometrically before any connector exists.
    """
    if count < 2:
        raise ValueError(f"a decomposition needs at least 2 parts, got {count}")
    solid = _watertight_copy(whole, "the whole mesh")

    extents = np.asarray(solid.extents, dtype=float)
    cut_axis = int(np.argmax(extents)) if axis is None else int(axis)
    low, high = solid.bounds[0][cut_axis], solid.bounds[1][cut_axis]
    edges = np.linspace(float(low), float(high), count + 1)

    pieces: list[DecomposedPart] = []
    for index in range(count):
        start, end = float(edges[index]), float(edges[index + 1])
        size = np.asarray(extents, dtype=float).copy()
        size[cut_axis] = max(end - start, 1e-6)
        centre = np.asarray(solid.bounds, dtype=float).mean(axis=0)
        centre[cut_axis] = (start + end) / 2.0
        slab = trimesh.creation.box(extents=tuple(size))
        slab.apply_translation(centre)
        piece = trimesh.boolean.intersection([solid, slab], engine="manifold")
        if isinstance(piece, trimesh.Scene):
            geometries = list(piece.geometry.values())
            if not geometries:
                continue
            piece = trimesh.util.concatenate(geometries) if len(geometries) > 1 else geometries[0]
        if not isinstance(piece, trimesh.Trimesh) or piece.volume <= 1e-6:
            continue
        pieces.append(DecomposedPart(id=f"part_{index:02d}", mesh=piece))

    if len(pieces) < 2:
        raise ValueError(
            f"splitting along axis {cut_axis} produced {len(pieces)} solid piece(s); "
            "try another axis or fewer parts"
        )
    return Decomposition(
        parts=tuple(pieces),
        provenance={
            "method": "local-slabs",
            "axis": "xyz"[cut_axis],
            "count": count,
            "note": "equal slabs along the longest axis; not a semantic decomposition",
        },
    )


def parts_from_decomposition(
    decomposition: Decomposition,
    *,
    role: str = "part",
    source_hashes: dict[str, str] | None = None,
) -> tuple[PartSpec, ...]:
    """A `PartSpec` per piece: bbox envelope, bbox centre, and the blank's hash.

    The envelope is the piece's bounding box, which is what every verifier and
    report means by a part's size; the *shape* is the mesh, and `source` is how the
    materialiser finds it. Keeping both is what lets a decomposed part still answer
    "how thin is it" without loading a mesh.
    """
    specs: list[PartSpec] = []
    for index, part in enumerate(decomposition.parts):
        bounds = np.asarray(part.mesh.bounds, dtype=float)
        size = bounds[1] - bounds[0]
        centre = bounds.mean(axis=0)
        specs.append(
            PartSpec(
                id=part.id,
                # The first piece of a chained assembly is the frame; giving it the
                # `body` role keeps the two-part reports and the CLI reading the same
                # way. It is a hint, never topology — interfaces decide that.
                role="body" if index == 0 and role == "part" else role,
                size_mm=tuple(float(value) for value in size),  # type: ignore[arg-type]
                pose_mm=tuple(float(value) for value in centre),  # type: ignore[arg-type]
                source=(source_hashes or {}).get(part.id) or mesh_hash(part.mesh),
            )
        )
    return tuple(specs)
