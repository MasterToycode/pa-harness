"""Cutting and growing connector features on *arbitrary* solids.

Why this exists next to `connectors/`
-------------------------------------
The connector builders in `pah/design/connectors/` build a part from its
`size_mm` blank: they are the parametric side, where the design decides the whole
shape. A decomposition (X-Part, or the local splitter) hands back meshes instead —
geometry the design did **not** choose — and the interface still has to be built
into it. That is this module: subtract a socket from a mesh, union a peg onto a
mesh, at a frame somebody else computed.

The split is deliberate rather than tidy. A parametric builder may exploit
anything it knows about its blank (peg_hole's blind-socket floor is `size_mm[2]/2 -
engage`); machining may assume nothing about the mesh except that it is watertight
and positive-volume. Keeping them apart is what stops "the body is a box" from
leaking into the code that handles a scanned gearbox housing.

Frame contract
--------------
Every function takes an **axis** and a **point**, both world frame:

* `axis` is the unit normal from the *host* part toward the *mating* part, i.e. the
  direction the mating part retreats when it is pulled out;
* `point` is on the mating plane, where the two solids meet when seated.

The socket goes into the host along `-axis` and the peg grows out of the mating
part along `-axis` too, so both live in the same cylinder and the clearance is the
only difference between them. That is what makes the pair verify: at the seated
pose the two solids share a face and no volume.

Determinism
-----------
Same inputs, same bytes: the side count is fixed (64, the same as `peg_hole`), no
operation reads a random seed, and the CSG engine is named explicitly. A cut that
came out differently on a second run would make the artifact hash a lie.
"""

from __future__ import annotations

import numpy as np
import trimesh

__all__ = [
    "MachiningError",
    "add_peg",
    "cut_socket",
    "cylinder_between",
    "material_at",
]

#: Sides of every cutter and peg. 64 is `peg_hole`'s number: ~6 um inside nominal on
#: a 10 mm diameter, an order of magnitude below any fit band the constraints express.
SECTIONS = 64

#: How far the cutter starts outside the mating plane. A cylinder whose cap is
#: coplanar with the face it enters leaves slivers in every CSG kernel; starting
#: 1 mm clear costs nothing and removes the class.
_MOUTH_OVERSHOOT_MM = 1.0

#: How deep a peg's root is buried in its own part. The peg has to *join* the
#: material it grows from, and a root that stops exactly at the surface is a
#: tangency, which is not a union.
_ROOT_PAD_MM = 2.0


class MachiningError(ValueError):
    """A boolean produced something that is not one closed solid.

    Named, because the alternative is a verifier reporting a wall-thickness failure
    on a mesh that is actually a `trimesh.Scene` with two shells in it.
    """


def _unit(vector) -> np.ndarray:
    array = np.asarray(vector, dtype=float).reshape(3)
    length = float(np.linalg.norm(array))
    if length <= 0.0:
        raise MachiningError(f"{tuple(vector)!r} is not a direction")
    return array / length


def cylinder_between(start, end, radius: float) -> trimesh.Trimesh:
    """A closed cylinder spanning two points, with `radius` about the segment.

    Built centred on the origin and then rotated, because `trimesh.creation.cylinder`
    has no "from here to there" form; the rotation comes from `align_vectors` so no
    hand-rolled basis can get the handedness wrong.
    """
    if radius <= 0.0:
        raise MachiningError(f"cylinder radius must be positive, got {radius!r}")
    first = np.asarray(start, dtype=float).reshape(3)
    second = np.asarray(end, dtype=float).reshape(3)
    vector = second - first
    length = float(np.linalg.norm(vector))
    if length <= 0.0:
        raise MachiningError("cylinder between identical points has no volume")

    mesh = trimesh.creation.cylinder(radius=float(radius), height=length, sections=SECTIONS)
    rotation = trimesh.geometry.align_vectors((0.0, 0.0, 1.0), vector / length)
    transform = np.eye(4)
    transform[:3, :3] = rotation[:3, :3]
    transform[:3, 3] = (first + second) / 2.0
    mesh.apply_transform(transform)
    return mesh


def _one_solid(result, what: str) -> trimesh.Trimesh:
    """The boolean's result, or a `MachiningError` naming what went wrong."""
    if isinstance(result, trimesh.Scene):
        geometries = list(result.geometry.values())
        if len(geometries) != 1:
            raise MachiningError(
                f"{what} produced {len(geometries)} shells, not one solid; the "
                "feature probably cut the part in two"
            )
        result = geometries[0]
    if not isinstance(result, trimesh.Trimesh):
        raise MachiningError(f"{what} produced {type(result).__name__}, not a mesh")
    volume = float(result.volume) if result.volume is not None else float("nan")
    if not np.isfinite(volume) or volume <= 0.0 or not result.is_watertight:
        raise MachiningError(
            f"{what} produced a non-closed solid (watertight={result.is_watertight}, "
            f"volume={result.volume!r})"
        )
    return result


def cut_socket(
    mesh: trimesh.Trimesh,
    *,
    axis,
    point,
    d_hole_mm: float,
    depth_mm: float,
) -> trimesh.Trimesh:
    """Subtract a blind socket from `mesh`, opening at `point`, going along `-axis`.

    `depth_mm` is measured from the mating plane, so the socket's floor is exactly
    where a peg of `engage_mm` depth lands. The cutter overshoots the plane by 1 mm,
    which removes nothing (there is no material outside the face) and keeps the
    boolean away from a coplanar cap.
    """
    direction = _unit(axis)
    mouth = np.asarray(point, dtype=float).reshape(3)
    cutter = cylinder_between(
        mouth + direction * _MOUTH_OVERSHOOT_MM,
        mouth - direction * float(depth_mm),
        float(d_hole_mm) / 2.0,
    )
    return _one_solid(
        trimesh.boolean.difference([mesh, cutter], engine="manifold"), "socket"
    )


def add_peg(
    mesh: trimesh.Trimesh,
    *,
    axis,
    point,
    d_peg_mm: float,
    engage_mm: float,
    root_pad_mm: float = _ROOT_PAD_MM,
) -> trimesh.Trimesh:
    """Union a peg onto `mesh`, standing out along `-axis` from `point`.

    `engage_mm` is how far the peg protrudes past the mating plane (into the other
    part's socket); `root_pad_mm` buries the root in the part's own material so the
    union closes instead of meeting at a tangency.
    """
    direction = _unit(axis)
    mouth = np.asarray(point, dtype=float).reshape(3)
    peg = cylinder_between(
        mouth - direction * float(engage_mm),
        mouth + direction * float(root_pad_mm),
        float(d_peg_mm) / 2.0,
    )
    return _one_solid(trimesh.boolean.union([mesh, peg], engine="manifold"), "peg")


def material_at(mesh: trimesh.Trimesh, point, axis) -> bool:
    """Is there material at `point`, just inside `mesh` along `-axis`?

    Used to check that a computed mating frame actually lands on the solid. A pair
    whose chosen frame sits in empty space is a pairing bug, and finding it with one
    ray test here is cheaper than reading a wall-thickness failure three verifiers
    later. The probe is 0.1 mm inside the surface, not on it: a point exactly on a
    face belongs to neither side reliably.
    """
    direction = _unit(axis)
    origin = np.asarray(point, dtype=float).reshape(3) + direction * 0.1
    ray = origin - direction * 1.0
    locations, _, _ = mesh.ray.intersects_location(
        ray_origins=[ray], ray_directions=[direction], multiple_hits=False
    )
    return len(locations) > 0
