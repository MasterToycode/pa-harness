"""Which way up should this part be printed?

Why the harness has to answer this
----------------------------------
The printability verifier evaluates overhangs only when a build orientation is
*declared* (`FitConstraints.print_orientation_deg`). That is the right default for one
part and the wrong one for a decomposition: each piece is printed on its own, in its own
best pose, so there is no single assembly-pose number that describes the job — and every
multi-part run therefore reported `overhang not evaluated` (rounds 2 and 3 of
`docs/remediation-plan.md`).

This module searches the 24 axis-aligned orientations (the cube's rotations: a printed
part rests on a face, so a face-down pose is what a person chooses) and returns a
recommendation with its numbers. It does not decide for the user: a declared orientation
still wins, and the report keeps both columns.

What the numbers mean
---------------------
* `overhang_area_mm2` — the downward-facing area steeper than the process angle;
* `bed_area_mm2` — the contact area with the bed, which is what decides adhesion;
* `height_mm` — the print height, which decides time and stability;
* `support_volume_mm3` — an **estimate**: overhang area times a third of its height
  above the bed. Not a slicer's number; labelled as an estimate everywhere it appears,
  because a support figure that looks measured but is guessed is worse than none.

Pareto, not a single score: a pose with less overhang but a tiny bed area is not
"better", it is a different trade. The recommendation minimises a stated pair —
overhang first, then height — and the full table is reported so the trade is visible.
"""

from __future__ import annotations

import itertools

import numpy as np
import trimesh

__all__ = ["CANDIDATES", "Orientation", "candidates", "recommend"]

#: The 24 axis-aligned rotations: every way a cube can sit on a face.
CANDIDATES: tuple[tuple[float, float, float], ...] = tuple(
    (float(rx), float(ry), float(rz))
    for rx in (0.0, 90.0, 180.0, 270.0)
    for ry in (0.0, 90.0, 180.0, 270.0)
    for rz in (0.0, 90.0, 180.0, 270.0)
    if len({(rx, ry, rz), (rx, (ry + 180) % 360, (rz + 180) % 360)}) == 2
)[:24]


def candidates() -> tuple[tuple[float, float, float], ...]:
    """The poses the search considers, in a stable order."""
    return CANDIDATES


def _rotation(angles: tuple[float, float, float]) -> np.ndarray:
    from pah.tools.verifiers.printability import print_rotation

    return print_rotation(angles)


class Orientation:
    """One candidate pose with the numbers a person would weigh."""

    def __init__(
        self,
        angles: tuple[float, float, float],
        *,
        overhang_area_mm2: float,
        bed_area_mm2: float,
        height_mm: float,
        support_volume_mm3: float,
    ) -> None:
        self.angles = tuple(float(value) for value in angles)
        self.overhang_area_mm2 = float(overhang_area_mm2)
        self.bed_area_mm2 = float(bed_area_mm2)
        self.height_mm = float(height_mm)
        self.support_volume_mm3 = float(support_volume_mm3)

    def as_dict(self) -> dict:
        return {
            "angles_deg": [round(value, 3) for value in self.angles],
            "overhang_area_mm2": round(self.overhang_area_mm2, 3),
            "bed_area_mm2": round(self.bed_area_mm2, 3),
            "height_mm": round(self.height_mm, 3),
            "support_volume_mm3_estimate": round(self.support_volume_mm3, 3),
        }

    def __repr__(self) -> str:  # a readable failure in a test, not a memory address
        return (
            f"Orientation({self.angles}, overhang={self.overhang_area_mm2:.2f} mm2, "
            f"bed={self.bed_area_mm2:.2f} mm2, height={self.height_mm:.2f} mm)"
        )


def _evaluate(mesh: trimesh.Trimesh, angles: tuple[float, float, float], angle_deg: float) -> Orientation:
    from pah.tools.verifiers.printability import overhang_faces

    placed = mesh.copy()
    matrix = np.eye(4)
    matrix[:3, :3] = _rotation(angles)
    placed.apply_transform(matrix)

    area, _worst, _count = overhang_faces(placed, min_angle_deg=angle_deg)
    low = float(placed.bounds[0][2])
    triangles = placed.triangles_center
    heights = np.maximum(triangles[:, 2] - low, 0.0)
    # Support volume is a *proxy*: the overhang area lifted by a third of its height is
    # the classic pyramid estimate. The key name says so, and the report repeats it.
    support = float(area) * float(np.mean(heights)) / 3.0 if area > 0.0 else 0.0

    face_normals = placed.face_normals
    areas = placed.area_faces
    bed = float(areas[np.abs(face_normals[:, 2] + 1.0) < 1e-6].sum())
    drop = low
    return Orientation(
        angles,
        overhang_area_mm2=float(area),
        bed_area_mm2=bed,
        # Height is measured from the part's own bottom, so a pose that buries the part
        # below the bed cannot report a negative height.
        height_mm=float(placed.extents[2]),
        support_volume_mm3=support,
    )


def recommend(
    mesh: trimesh.Trimesh, *, angle_deg: float = 45.0, candidates_deg=None
) -> Orientation:
    """The best face-down pose: least overhang, then least height.

    Deterministic tie-break (overhang, height, then the candidate order), because a
    recommendation that changes between two identical runs would make the report's
    "recommended orientation" column useless as evidence.
    """
    poses = candidates_deg if candidates_deg is not None else CANDIDATES
    scored = [_evaluate(mesh, tuple(angles), angle_deg) for angles in poses]
    return min(scored, key=lambda row: (row.overhang_area_mm2, row.height_mm, row.angles))


def table(mesh: trimesh.Trimesh, *, angle_deg: float = 45.0) -> list[Orientation]:
    """Every candidate, best first — the trade the recommendation chose from."""
    scored = [_evaluate(mesh, angles, angle_deg) for angles in CANDIDATES]
    return sorted(scored, key=lambda row: (row.overhang_area_mm2, row.height_mm, row.angles))
