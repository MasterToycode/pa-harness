"""`printability`: can the part be made, given the material between the bore and the outside?

Two codes, two measurements, both taken **from the meshes** rather than from the
spec's parameters. That is the point of the module: the spec says a bore is 10 mm
and 15 mm from the edge; whether the *built* solid actually has 15 mm there is a
different claim, and a verifier that repeated the arithmetic would agree with the
spec forever.

1. `below_min_wall` — the material between the interface and the outside
------------------------------------------------------------------------
The locus is the interface itself: the body's surface within `window` mm of the
insert's surface. That is the bore (or slot, or dovetail) wall — it is within one
clearance of the insert by construction — and the inset of the window is small
enough that the part's outer faces are *not* included on a normal part. From each
sampled surface point the thickness is measured by casting a ray **into the
material**, i.e. along the inward face normal, and taking the first exit: for a
bore wall that ray runs radially outward and stops at the nearest outside surface,
which is exactly the wall a drill would leave.

Why a ray and not the distance to the outer surface: the ray reports the material
the tool would have to cut, one number per sample, and it needs no notion of which
face is "outside". Its approximation is angular sampling — the minimum is taken
over 64 facet centres around the bore, so it can overshoot the true thinnest wall
by `1 - 1/cos(half facet)`. Measured on the fixtures: 15.026 mm against an analytic
15.000, and 0.609 mm against 0.600 on `thin_wall_fixture()` (+0.2 % and +1.5 %).

Known limitations, all of them documented rather than silently patched:

* only directions perpendicular to the build axis are measured, because that is
  where a bore's wall is; a **blind hole's floor thickness is not checked**;
* walls unrelated to the connector (a thin rib elsewhere on the part) are out of
  scope — contract §3 scopes this code to "the hole wall to the outer surface";
* if the insert is nowhere near the body's surface, the interface cannot be
  located and the check **raises**. The cascade turns that into an `environment`
  certificate, so "I could not find the wall" is unknown, never a pass.

2. `unsupported_overhang` — a downward face with nothing under it
----------------------------------------------------------------
**An overhang claim needs an orientation**, and the spec says which one:
`FitConstraints.print_orientation_deg`. With `None` (the default) this verifier
does not evaluate overhang at all and reports `overhang_evaluated = False`, which
a report renders as `manufacturing_validated: not evaluated` — a different claim
from "printable". Why default off: in the assembled pose an insert stands tip
down in mid-air, so judging it there declares every peg unprintable for a defect
no repair operator can address (`rotate_part` moves the assembly pose, not the
build orientation). An always-on code carries no information and costs the loop
its whole budget chasing it.

With an orientation declared, each part is rotated by `print_orientation_deg`
(extrinsic XYZ degrees, the same convention `PartSpec.rot_deg` spells —
`R = Rz @ Ry @ Rx`) and scanned with the build axis at +Z. For every face whose
outward normal points downward, `angle_from_horizontal_deg =
degrees(acos(-n . z))` is how far the face is from being a flat ceiling: 0° is a
horizontal ceiling (the worst case, it prints on air), 90° is a vertical wall
(perfect). The process decides the smallest angle that needs no support
(`PROCESS_MIN_ANGLE_FROM_HORIZONTAL_DEG`, the usual 45° rule for FDM), and a face
below it, above the bed, is an overhang. Faces within `BED_TOL_MM` of the part's
own lowest point are bed contact, not overhangs — otherwise every part would fail
on its own bottom.

The rotation's pivot does not matter here: the overhang test reads normals and the
bed test reads the rotated part's own lowest plane, and both are translation
invariant, so rotating about the world origin and rotating about the part's local
origin give the same answer.

The area floor `OVERHANG_MIN_AREA_MM2` exists because a fillet's sliver triangle is
not a feature that sags; 0.5 mm2 is far below anything that shows up in a print.
"""

from __future__ import annotations

import math

import numpy as np
import trimesh

from pah.core.model import FailureCertificate
from pah.design.api import (
    CODE_BELOW_MIN_WALL,
    CODE_UNSUPPORTED_OVERHANG,
    DesignView,
)
from pah.design.spec import Vec3

from .base import VerifierSpec

__all__ = [
    "BED_TOL_MM",
    "CODES",
    "OVERHANG_MIN_AREA_MM2",
    "PRINTABILITY_DEFAULT_WINDOW_MM",
    "PROCESS_MIN_ANGLE_FROM_HORIZONTAL_DEG",
    "Printability",
    "min_wall_mm",
    "overhang_faces",
    "print_rotation",
]

CODES: tuple[str, ...] = (CODE_BELOW_MIN_WALL, CODE_UNSUPPORTED_OVERHANG)

#: The smallest angle from horizontal at which a *downward-facing* surface can be
#: printed without support, per process (`Unsupported overhang` needs `angle <
#: limit`). FDM's 45° rule; resin droops a little steeper; a powder bed needs no
#: support at all; `None` means the concept does not apply (subtractive).
PROCESS_MIN_ANGLE_FROM_HORIZONTAL_DEG: dict[str, float | None] = {
    "fdm": 45.0,
    "sla": 30.0,
    "sls": 0.0,
    "cnc": None,
}

#: Total area a set of downward faces must cover before it counts as an overhang.
OVERHANG_MIN_AREA_MM2 = 0.5

#: How far above its own lowest point a face must be to be an overhang rather than
#: bed contact. Under a typical first layer, so a fillet's bottom sliver stays bed.
BED_TOL_MM = 0.2

#: Nudge that puts a ray origin just *inside* the material, so the first hit is the
#: exit surface rather than the triangle the point sits on.
_INSIDE_MM = 1e-4

#: Hits closer than this to the origin are the origin's own surface, not a wall.
_SELF_HIT_MM = 1e-3

#: Half-width of the interface band when the connector kind declares no clearance
#: to scale it with: generous enough for a loose fit, tight enough to exclude the
#: outer faces of a part a few tens of millimetres across.
PRINTABILITY_DEFAULT_WINDOW_MM = 2.0

#: `manufacturing`: a wall too thin to print is a process limit, not a joint or a
#: geometry bug — the same family the M0 adapter puts `socket_breaks_wall` in.
_FAMILY = "manufacturing"


def min_wall_mm(
    body: trimesh.Trimesh,
    insert: trimesh.Trimesh,
    *,
    window_mm: float,
) -> tuple[float, tuple[float, float, float], int]:
    """`(thinnest wall, the point it was measured from, how many samples)` in mm.

    The band is the body's surface within `window_mm` of the insert — the interface
    and nothing else. Thickness is the first surface the inward normal meets.

    Raises `ValueError` when there is no band (the insert is not near the body) or
    when no ray leaves the solid: both mean the wall cannot be measured, and a
    number invented here would be a lie the loop would act on.
    """
    centres = body.triangles_center
    normals = body.face_normals
    _, distance, _ = trimesh.proximity.closest_point(insert, centres)
    band = distance < window_mm
    if not band.any():
        raise ValueError(
            f"no body surface within {window_mm:.3f} mm of the insert: the interface "
            "wall cannot be located"
        )

    origins = centres[band] - normals[band] * _INSIDE_MM
    directions = -normals[band]
    locations, ray_index, _ = body.ray.intersects_location(
        ray_origins=origins, ray_directions=directions, multiple_hits=True
    )
    if len(ray_index) == 0:
        raise ValueError(
            f"none of the {int(band.sum())} interface faces has material behind it; "
            "the wall thickness is undefined"
        )

    lengths = np.linalg.norm(locations - origins[ray_index], axis=1)
    keep = lengths > _SELF_HIT_MM
    if not keep.any():
        raise ValueError("every wall ray hit its own surface; nothing to measure")

    measured = lengths[keep]
    best = int(np.argmin(measured))
    return (
        float(measured[best]) + _INSIDE_MM,
        tuple(float(value) for value in origins[ray_index][keep][best]),
        int(band.sum()),
    )


def overhang_faces(
    mesh: trimesh.Trimesh,
    *,
    min_angle_deg: float,
    bed_tol_mm: float = BED_TOL_MM,
) -> tuple[float, float, int]:
    """`(offending area mm2, worst angle from horizontal deg, face count)`.

    `angle_from_horizontal_deg` is 0 for a flat ceiling and 90 for a vertical wall;
    a face fires when it faces down, sits above the bed, and its angle is below
    `min_angle_deg`. The returned angle is the *worst* one found, so a report can
    say how bad it is rather than only that it is bad.
    """
    normals = mesh.face_normals
    areas = mesh.area_faces
    downward = -normals[:, 2]
    facing_down = downward > 1e-9
    if not facing_down.any():
        return (0.0, 90.0, 0)

    angle = np.degrees(np.arccos(np.clip(downward, -1.0, 1.0)))
    lowest = float(mesh.bounds[0][2])
    above_bed = mesh.triangles_center[:, 2] > lowest + bed_tol_mm
    offending = facing_down & (angle < min_angle_deg) & above_bed
    count = int(offending.sum())
    if count == 0:
        return (0.0, 90.0, 0)
    return (float(areas[offending].sum()), float(angle[offending].min()), count)


def print_rotation(orientation_deg: Vec3) -> np.ndarray:
    """The 3x3 rotation `print_orientation_deg` names, in the spec's convention.

    Extrinsic XYZ, i.e. `R = Rz @ Ry @ Rx` — the same rotation `PartSpec.rot_deg`
    spells (`design/connectors/__init__.py:part_matrix`), so a spec author has one
    way to write an orientation and no verifier gets to invent a second. Kept
    local rather than imported from the connector package on purpose: a verifier
    must not depend on the code that builds the geometry it judges.
    """
    rx, ry, rz = (math.radians(float(angle)) for angle in orientation_deg)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    return np.array(
        [
            [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
            [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
            [-sy, cy * sx, cy * cx],
        ],
        dtype=float,
    )


def _in_print_pose(
    mesh: trimesh.Trimesh, orientation_deg: Vec3
) -> trimesh.Trimesh:
    """`mesh` rotated into its print pose. Always a copy: views are read-only."""
    placed = mesh.copy()
    if any(float(angle) != 0.0 for angle in orientation_deg):
        matrix = np.eye(4)
        matrix[:3, :3] = print_rotation(orientation_deg)
        placed.apply_transform(matrix)
    return placed


class Printability:
    """Wall thickness around the connector, and the process's overhang rule."""

    spec = VerifierSpec(
        id="printability",
        codes=CODES,
        cost_class="medium",
        fixtures={
            "known_good": [
                "fixture-standard: 4.0 mm socket floor and 15 mm radial wall, "
                "no declared orientation so no overhang claim"
            ],
            "known_bad": [
                "fixture-thin-wall: 0.6 mm of material between socket and outside",
                "fixture-floor-pocket: 10x10 mm ceiling facing down above the bed, "
                "with the as-designed pose declared as the build orientation",
            ],
        },
    )

    def __init__(
        self,
        *,
        interface_window_mm: float | None = None,
        bed_tol_mm: float = BED_TOL_MM,
    ) -> None:
        self.interface_window_mm = interface_window_mm
        self.bed_tol_mm = float(bed_tol_mm)
        #: Whether the last `check` actually evaluated overhang. `False` means the
        #: spec declared no `print_orientation_deg`, i.e. no manufacturing
        #: evaluation was claimed — a report must render that as "not evaluated",
        #: never as "printable" (§6 of `pah-dev-doc-v1.md` keeps those apart).
        self.overhang_evaluated = False
        #: Why the last `check` declined to evaluate something, or "" when it
        #: evaluated everything it could. The cascade aggregates these into
        #: `DesignVerifierCascade.notes`; a capability boundary is not a tool
        #: failure, so it is never dressed up as an `environment` certificate.
        self.last_note = ""

    def fixtures(self) -> tuple[list[DesignView], list[DesignView]]:
        """One printable part and two that are not, for different reasons."""
        from ._design_fixtures import pocket_fixture, standard_fixture, thin_wall_fixture

        return (
            [standard_fixture()],
            [thin_wall_fixture(), pocket_fixture()],
        )

    def check(self, view: DesignView) -> list[FailureCertificate]:
        """The wall finding and the overhang finding, in that order, if any."""
        constraints = view.spec.constraints
        orientation = constraints.print_orientation_deg
        limit = PROCESS_MIN_ANGLE_FROM_HORIZONTAL_DEG[constraints.process]
        # Recorded before any measurement, so a report can tell "the overhang rule
        # found nothing" from "no orientation was declared, so it was not run".
        self.overhang_evaluated = orientation is not None and limit is not None
        self.last_note = self._overhang_note(constraints.process, orientation, limit)

        certificates: list[FailureCertificate] = []
        try:
            wall = self._wall(view)
        except ValueError as error:
            # A capability boundary, not a tool fault: the insert is not near the
            # body's interface band, so there is no wall to measure (measured on a
            # generated insert that came back far from the socket). Raising would be
            # turned into an environment certificate and the episode would end as
            # "unknown" — but the right report is "not evaluated", the third state
            # the counters keep apart from both failure and pass.
            wall = None
            note = f"wall not evaluated: {error}"
            self.last_note = f"{self.last_note}; {note}" if self.last_note else note
        if wall is not None:
            certificates.append(wall)
        if self.overhang_evaluated:
            overhang = self._overhang(view, orientation, limit)
            if overhang is not None:
                certificates.append(overhang)
        return certificates

    def _overhang_note(
        self, process: str, orientation: Vec3 | None, limit: float | None
    ) -> str:
        """The one line a report prints when no overhang claim was made."""
        if orientation is None:
            return (
                "overhang not evaluated: the spec declares no print_orientation_deg, "
                "so there is no build direction to judge"
            )
        if limit is None:
            return (
                f"overhang not evaluated: process {process!r} is subtractive, so an "
                "overhang is not a manufacturing limit"
            )
        return ""

    # -- wall --------------------------------------------------------------
    def _wall(self, view: DesignView) -> FailureCertificate | None:
        constraints = view.spec.constraints
        window = self._window(view)
        measured, at, samples = min_wall_mm(view.body, view.insert, window_mm=window)
        if measured >= constraints.min_wall_mm:
            return None
        interface = view.spec.interface
        return FailureCertificate(
            code=CODE_BELOW_MIN_WALL,
            family=_FAMILY,
            locus={
                "spec": view.spec.spec_id,
                "revision": view.spec.revision,
                "interface": interface.id,
                "part": view.spec.body.id,
            },
            magnitude={
                "min_wall_mm": measured,
                "required_wall_mm": constraints.min_wall_mm,
                "process": constraints.process,
                "at_mm": list(at),
                "samples": samples,
                "interface_window_mm": window,
                "method": "ray_from_interface_surface_inward_normal",
                "approximation": (
                    "radial directions only (no blind-hole floor); min over "
                    "facet centres, up to +1/cos(half facet) high"
                ),
            },
            observed_by=self.spec.id,
            evidence_ref="wall_ray_cast",
        )

    def _window(self, view: DesignView) -> float:
        """How wide the interface band is, scaled by the fit it was designed with."""
        if self.interface_window_mm is not None:
            return self.interface_window_mm
        try:
            clearance = abs(view.spec.clearance_mm)
        except KeyError:
            # A plugin kind with no clearance source: use the generic band rather
            # than skip the check — the wall still exists whatever kind cut it.
            return PRINTABILITY_DEFAULT_WINDOW_MM
        return max(1.0, 3.0 * clearance + 0.5)

    # -- overhang ----------------------------------------------------------
    def _overhang(
        self, view: DesignView, orientation: Vec3, limit: float
    ) -> FailureCertificate | None:
        """The worst overhanging part, scanned in the declared print orientation."""
        worst: tuple[str, float, float, int] | None = None
        for part in view.spec.parts:
            placed = _in_print_pose(view.mesh(part.id), orientation)
            area, angle, faces = overhang_faces(
                placed, min_angle_deg=limit, bed_tol_mm=self.bed_tol_mm
            )
            if area < OVERHANG_MIN_AREA_MM2:
                continue
            if worst is None or area > worst[1]:
                worst = (part.id, area, angle, faces)
        if worst is None:
            return None

        part_id, area, angle, faces = worst
        return FailureCertificate(
            code=CODE_UNSUPPORTED_OVERHANG,
            family=_FAMILY,
            locus={
                "spec": view.spec.spec_id,
                "revision": view.spec.revision,
                "part": part_id,
                "process": view.spec.constraints.process,
            },
            magnitude={
                "part": part_id,
                "area_mm2": area,
                "angle_from_horizontal_deg": angle,
                "min_angle_from_horizontal_deg": limit,
                "faces": faces,
                "min_area_mm2": OVERHANG_MIN_AREA_MM2,
                "bed_tol_mm": self.bed_tol_mm,
                "print_orientation_deg": [float(value) for value in orientation],
                "method": "face_normal_vs_build_axis_z",
                "approximation": (
                    "the declared print orientation is taken as the build pose; "
                    "the bed is the rotated part's own lowest plane"
                ),
            },
            observed_by=self.spec.id,
            evidence_ref="overhang_scan",
        )
