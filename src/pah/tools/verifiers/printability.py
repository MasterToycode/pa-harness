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
**An overhang claim needs an orientation**, and the spec may declare one:
`FitConstraints.print_orientation_deg`. When it does, that pose wins — a person who
chose a build direction is not overruled by a search. When it does not, this
verifier *searches* the 24 face-down poses (`design/orientation.py:recommend`) and
judges the least-overhanging one, recording the result as `orientation_source =
"recommended"` so the report can keep the "declared" and "recommended" columns
apart (audit P2-2 / C2).

Why search instead of declining: a decomposition prints each piece on its own, so
the assembly pose is not a build pose for any of them, and "overhang not
evaluated" on every multi-part run was a hole in the evidence rather than a
cautious answer. Why the source is recorded: a recommendation is a *suggestion*
about a decision nobody made, and a report that prints one "orientation" number
for both cases is how a guess starts reading like a choice. The declared pose is
still the only one the loop may act on (`rotate_part` moves the assembly pose, not
the build pose).

The only orientation-free case left is a process with no overhang rule
(`PROCESS_MIN_ANGLE_FROM_HORIZONTAL_DEG[process] is None`, i.e. subtractive):
there `overhang_evaluated = False` and `last_note` says why, because an
orientation would have nothing to judge.

With an orientation resolved, each part is rotated by it (extrinsic XYZ degrees,
the same convention `PartSpec.rot_deg` spells — `R = Rz @ Ry @ Rx`) and scanned
with the build axis at +Z. For every face whose outward normal points downward,
`angle_from_horizontal_deg = degrees(acos(-n . z))` is how far the face is from
being a flat ceiling: 0° is a horizontal ceiling (the worst case, it prints on
air), 90° is a vertical wall (perfect). The process decides the smallest angle
that needs no support (`PROCESS_MIN_ANGLE_FROM_HORIZONTAL_DEG`, the usual 45° rule
for FDM), and a face below it, above the bed, is an overhang. Faces within
`BED_TOL_MM` of the part's own lowest point are bed contact, not overhangs —
otherwise every part would fail on its own bottom.

The rotation's pivot does not matter here: the overhang test reads normals and the
bed test reads the rotated part's own lowest plane, and both are translation
invariant, so rotating about the world origin and rotating about the part's local
origin give the same answer.

Known limitation of the search: it recommends **one** pose, taken from the insert,
and applies it to every part of the view. Two parts of one joint are printed
separately and may want different poses, so a body with an overhang that the
insert's pose does not clear is reported as `unsupported_overhang` when a
different pose would have cleared it. That is the conservative direction (it names
a real overhang in the pose that was actually evaluated) and it is visible in the
certificate: `orientation_source` says the orientation was recommended, and
`recommended_overhang_area_mm2` gives the insert's own number, so a reader can see
the two do not describe the same part. A per-part search is a contract change
(`FitConstraints.print_orientation_deg` is one pose), not a local one.

The area floor `OVERHANG_MIN_AREA_MM2` exists because a fillet's sliver triangle is
not a feature that sags; 0.5 mm2 is far below anything that shows up in a print.

The bridge rule: an angle is half the question
----------------------------------------------
A ceiling needs support only if the opening under it is **wider than the process can
cross**. `ProcessProfile.bridge_mm` is that width for FDM (8 mm); resin is 0, because
a vat is not bridgeable at all. So a connected ceiling whose narrow dimension is at
most `bridge_mm` is a bridge, not an unsupported overhang, and its area no longer
counts. This is what stops a 0.3 mm step across a 6.4 mm bore from making every
decomposed piece unprintable — measured 2026-11-02, 6.148 mm2 per part, a false
positive no repair operator can act on.

It is a *span* rule, not a smaller area floor: a 40 x 6 mm ceiling (240 mm2) is
bridged while a 10 x 10 mm one (100 mm2) is reported, which no single area number can
reproduce. The forgiveness is never silent — `Printability.bridged_overhangs` and the
report's `verdicts.printable.orientation.bridged_ceilings` name the span that was
accepted, and a certificate that still fires carries them in its magnitude, so "we
bridged this" and "we never looked" cannot be read the same way.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import trimesh

from pah.core.model import FailureCertificate
from pah.design.orientation import Orientation, table
from pah.design.process import PROCESS_PROFILES, effective_min_wall_mm, profile_for
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
    "Ceiling",
    "OVERHANG_MIN_AREA_MM2",
    "PRINTABILITY_DEFAULT_WINDOW_MM",
    "PROCESS_MIN_ANGLE_FROM_HORIZONTAL_DEG",
    "Printability",
    "ceiling_regions",
    "min_wall_mm",
    "overhang_faces",
    "print_rotation",
]

CODES: tuple[str, ...] = (CODE_BELOW_MIN_WALL, CODE_UNSUPPORTED_OVERHANG)

#: The smallest angle from horizontal at which a *downward-facing* surface can be
#: printed without support, per process (`Unsupported overhang` needs `angle <
#: limit`). FDM's 45° rule; resin droops a little steeper; a powder bed needs no
#: support at all; `None` means the concept does not apply (subtractive).
#: Kept as a derived view of pah.design.process so the two cannot drift: the table
#: lives with the process, not with the verifier (audit P2-1).
PROCESS_MIN_ANGLE_FROM_HORIZONTAL_DEG: dict[str, float | None] = {
    name: profile.overhang_deg for name, profile in PROCESS_PROFILES.items()
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
    bridge_mm: float = 0.0,
) -> tuple[float, float, int]:
    """`(offending area mm2, worst angle from horizontal deg, face count)`.

    `angle_from_horizontal_deg` is 0 for a flat ceiling and 90 for a vertical wall;
    a face fires when it faces down, sits above the bed, and its angle is below
    `min_angle_deg`. The returned angle is the *worst* one found, so a report can
    say how bad it is rather than only that it is bad.

    `bridge_mm` > 0 turns on the bridged-span rule: a connected ceiling whose
    horizontal *span* is at most `bridge_mm` is printed by the first layer across
    the gap, so it is not an unsupported overhang and its area is not counted. The
    default 0 keeps the plain angle rule, which is what the orientation search ranks
    poses by (`design/orientation.py`): the search is about the trade between poses,
    and a bridge is bridgeable in every one of them.
    """
    area, worst, count = _facing_down_area(
        mesh, min_angle_deg=min_angle_deg, bed_tol_mm=bed_tol_mm
    )
    if not bridge_mm or bridge_mm <= 0.0 or count == 0:
        return area, worst, count

    kept = [
        ceiling
        for ceiling in ceiling_regions(
            mesh, min_angle_deg=min_angle_deg, bed_tol_mm=bed_tol_mm
        )
        if ceiling.span_mm > bridge_mm
    ]
    if not kept:
        return (0.0, 90.0, 0)
    return (
        float(sum(ceiling.area_mm2 for ceiling in kept)),
        float(min(ceiling.worst_angle_deg for ceiling in kept)),
        int(sum(ceiling.faces for ceiling in kept)),
    )


@dataclass(frozen=True)
class Ceiling:
    """One connected group of downward faces, with the span it has to cross.

    A *ceiling* is a set of edge-adjacent faces that all face down and sit above the
    part's own bed plane. Its `span_mm` is the narrow dimension of the thing it
    covers — the distance the first layer has to bridge:

    * the shorter side of its horizontal footprint, because the longer side runs
      along the material that anchors it and bridging happens across the short way;
    * but never more than the diameter of a circle with the same area, so a **thin**
      ceiling is judged by the gap it crosses rather than by the bore it sits on. An
      annular groove does not get to inherit the diameter of the hole in the middle
      of it.

    Measured: 6.40 mm for a 6.4 mm bore's disc ceiling, 2.70 mm for a 0.3 mm-wide
    groove around that bore, 4.86 mm for the same groove around a 20 mm bore (where a
    footprint-only rule would have said 20 mm), and 10.00 mm for a 10 x 10 mm recess —
    the last one is what keeps the rule from forgiving real ceilings.
    """

    area_mm2: float
    worst_angle_deg: float
    faces: int
    span_mm: float
    centre_mm: tuple[float, float, float]


def ceiling_regions(
    mesh: trimesh.Trimesh,
    *,
    min_angle_deg: float,
    bed_tol_mm: float = BED_TOL_MM,
) -> list[Ceiling]:
    """Every connected ceiling on `mesh`, with its area, angle and bridging span.

    Connectedness matters: a finely tessellated 20 mm ceiling must be one region, not
    two hundred slivers, or a per-triangle size test would call every real ceiling a
    bridge. The span is therefore measured per region, never per face.

    The footprint measure is axis-aligned, so an opening rotated in the build plane
    reads up to sqrt(2) wider than it is and is judged more strictly — conservative
    in the direction that reports rather than forgives.
    """
    mask, _area, _worst, _count = _facing_down_mask(
        mesh, min_angle_deg=min_angle_deg, bed_tol_mm=bed_tol_mm
    )
    if not mask.any():
        return []

    angles = _angles_from_horizontal(mesh, mask)
    centres = mesh.triangles_center
    ceilings: list[Ceiling] = []
    for faces in _face_components(mesh, mask):
        corners = np.unique(mesh.faces[faces])
        extent = mesh.vertices[corners].max(axis=0) - mesh.vertices[corners].min(axis=0)
        area = float(mesh.area_faces[faces].sum())
        ceilings.append(
            Ceiling(
                area_mm2=area,
                worst_angle_deg=float(angles[faces].min()),
                faces=int(faces.size),
                span_mm=_span_mm(extent, area),
                centre_mm=tuple(float(value) for value in centres[faces].mean(axis=0)),
            )
        )
    return ceilings


def _span_mm(extent: np.ndarray, area_mm2: float) -> float:
    """The narrow dimension of a ceiling: see `Ceiling.span_mm` for why both bounds."""
    footprint = float(min(extent[0], extent[1]))
    equal_area_diameter = 2.0 * math.sqrt(max(area_mm2, 0.0) / math.pi)
    return min(footprint, equal_area_diameter)


def _facing_down_mask(
    mesh: trimesh.Trimesh, *, min_angle_deg: float, bed_tol_mm: float
) -> tuple[np.ndarray, float, float, int]:
    """The down-facing, above-the-bed, too-shallow mask plus its area/angle/count."""
    normals = mesh.face_normals
    areas = mesh.area_faces
    downward = -normals[:, 2]
    facing_down = downward > 1e-9
    if not facing_down.any():
        return np.zeros(len(areas), dtype=bool), 0.0, 90.0, 0

    angle = np.degrees(np.arccos(np.clip(downward, -1.0, 1.0)))
    lowest = float(mesh.bounds[0][2])
    above_bed = mesh.triangles_center[:, 2] > lowest + bed_tol_mm
    mask = facing_down & (angle < min_angle_deg) & above_bed
    count = int(mask.sum())
    if count == 0:
        return mask, 0.0, 90.0, 0
    return mask, float(areas[mask].sum()), float(angle[mask].min()), count


def _facing_down_area(
    mesh: trimesh.Trimesh, *, min_angle_deg: float, bed_tol_mm: float
) -> tuple[float, float, int]:
    _mask, area, worst, count = _facing_down_mask(
        mesh, min_angle_deg=min_angle_deg, bed_tol_mm=bed_tol_mm
    )
    return area, worst, count


def _angles_from_horizontal(mesh: trimesh.Trimesh, mask: np.ndarray) -> np.ndarray:
    """`degrees(acos(-n . z))` for every face; 0 is a flat ceiling, 90 a wall."""
    downward = -mesh.face_normals[:, 2]
    return np.degrees(np.arccos(np.clip(downward, -1.0, 1.0)))


def _face_components(mesh: trimesh.Trimesh, mask: np.ndarray) -> list[np.ndarray]:
    """The connected groups of edge-adjacent faces inside `mask`."""
    indices = np.nonzero(mask)[0]
    if indices.size <= 1:
        return [indices] if indices.size else []

    # Adjacency needs shared vertices, and geometry that came from an STL may carry
    # one copy of each vertex per triangle. Merge on a copy — the view's meshes are
    # read-only — and note that merging preserves the face order, so the original
    # indices stay valid.
    merged = mesh.copy()
    merged.merge_vertices()
    pairs = merged.face_adjacency
    if len(pairs) == 0:
        return [np.array([face], dtype=np.int64) for face in indices]
    inside = mask[pairs].all(axis=1)
    if not inside.any():
        return [np.array([face], dtype=np.int64) for face in indices]

    position = {int(face): slot for slot, face in enumerate(indices)}
    local = np.array(
        [[position[int(a)], position[int(b)]] for a, b in pairs[inside]], dtype=np.int64
    )
    groups = trimesh.graph.connected_components(local, nodes=np.arange(indices.size))
    return [indices[np.asarray(group, dtype=np.int64)] for group in groups]


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
        #: process has no overhang rule at all (subtractive), so nothing was
        #: claimed — a report must render that as "not evaluated", never as
        #: "printable" (§6 of `pah-dev-doc-v1.md` keeps those apart).
        self.overhang_evaluated = False
        #: Where the orientation the last `check` used came from: `"declared"` (the
        #: spec named one), `"recommended"` (the search filled the gap) or `"none"`
        #: (no rule to apply it to). Kept on the instance because the report's
        #: declared-versus-recommended columns cannot be reconstructed from the
        #: certificates — a clean design produces none.
        self.orientation_source = "none"
        #: The angles actually used, or `None` when nothing was evaluated.
        self.orientation_deg: Vec3 | None = None
        #: The winning candidate when `orientation_source == "recommended"`, else None.
        self.recommendation: Orientation | None = None
        #: The best few candidates, best first, so a report can show the trade the
        #: recommendation chose from without re-running the search.
        self.orientation_top: tuple[Orientation, ...] = ()
        #: Ceilings the bridge rule let pass in the last `check`, one entry per
        #: connected ceiling (`part`, `span_mm`, `area_mm2`, `faces`). Empty means
        #: nothing was forgiven. The runtime prints it beside the orientation column,
        #: so "no overhang finding" is legible as "evaluated, and these spans were
        #: bridged" rather than as silence.
        self.bridged_overhangs: tuple[dict, ...] = ()
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
        limit = PROCESS_MIN_ANGLE_FROM_HORIZONTAL_DEG[constraints.process]
        angles = self._resolve_orientation(view, limit)

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
        if self.overhang_evaluated and angles is not None:
            overhang = self._overhang(view, angles, limit)
            if overhang is not None:
                certificates.append(overhang)
        return certificates

    def _resolve_orientation(
        self, view: DesignView, limit: float | None
    ) -> Vec3 | None:
        """Which build orientation to judge overhangs in, and where it came from.

        Precedence, and the reason for it (audit P2-2 / C2):

        * **declared** — `FitConstraints.print_orientation_deg` wins. A person who
          chose a pose is not overruled by a search, and a spec that declares one is
          usually encoding a fixture or a customer requirement.
        * **recommended** — no declaration, so the 24 face-down poses are searched
          (`design/orientation.py`, one 24-pose sweep) and the least-overhanging one
          is used. A decomposition prints each piece in its own best pose, so
          declining here left every multi-part run reporting "not evaluated" — a hole
          in the evidence, not a cautious answer.
        * **none** — the process has no overhang rule (subtractive), so there is no
          claim an orientation could support; `last_note` says so.

        The chosen facts are recorded on the instance, because a *clean* design
        produces no certificate and the report's two columns still have to be filled.
        """
        process = view.spec.constraints.process
        declared = view.spec.constraints.print_orientation_deg
        self.recommendation = None
        self.orientation_top = ()
        self.orientation_deg = None
        self.last_note = ""

        if limit is None:
            self.orientation_source = "none"
            self.overhang_evaluated = False
            self.last_note = self._overhang_note(process)
            return None
        if declared is not None:
            self.orientation_source = "declared"
            self.overhang_evaluated = True
            self.orientation_deg = tuple(float(value) for value in declared)
            return self.orientation_deg

        rows = table(view.insert, angle_deg=limit)  # one sweep: the winner and its trade
        # `rows[0]` is `recommend()` by construction — same candidates, same
        # (overhang, height, angles) key, and `sorted` is stable — so the search is
        # paid for once instead of twice.
        self.orientation_top = tuple(rows[:3])
        self.recommendation = rows[0]
        self.orientation_source = "recommended"
        self.overhang_evaluated = True
        self.orientation_deg = self.recommendation.angles
        return self.orientation_deg

    def _overhang_note(self, process: str) -> str:
        """The one line a report prints when the overhang rule did not apply."""
        return (
            f"overhang not evaluated: process {process!r} is subtractive, so an "
            "overhang is not a manufacturing limit"
        )

    # -- wall --------------------------------------------------------------
    def _wall(self, view: DesignView) -> FailureCertificate | None:
        constraints = view.spec.constraints
        window = self._window(view)
        profile = profile_for(getattr(constraints, "process", ""))
        if profile is None:
            # No profile means no thresholds that apply: say so instead of borrowing
            # FDM's numbers for a process nobody described (audit P2-1).
            self.last_note = (
                "wall thickness not evaluated: no process profile for "
                f"{getattr(constraints, 'process', '')!r}"
            )
            return []
        required_wall = effective_min_wall_mm(constraints, profile)

        measured, at, samples = min_wall_mm(view.body, view.insert, window_mm=window)
        if measured >= required_wall:
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
        """The worst overhanging part, in the resolved pose, after the bridge rule.

        The bridge rule is a *span* rule, not a smaller area threshold (audit P2-2,
        ruling of 2026-11-02): a ceiling that only has to cross `bridge_mm` is
        printed by the first layer, and the finding is dropped with its span
        recorded — on the instance (`bridged_overhangs`) and, when something else in
        the same part does fire, in the certificate's magnitude. Dropping it silently
        would trade a false positive for an invisible assumption.
        """
        bridge_mm = self._bridge_mm(view)
        worst: tuple[str, float, float, int] | None = None
        bridged: list[dict] = []
        for part in view.spec.parts:
            placed = _in_print_pose(view.mesh(part.id), orientation)
            for ceiling in ceiling_regions(
                placed, min_angle_deg=limit, bed_tol_mm=self.bed_tol_mm
            ):
                if ceiling.area_mm2 < OVERHANG_MIN_AREA_MM2:
                    continue
                if bridge_mm > 0.0 and ceiling.span_mm <= bridge_mm:
                    bridged.append(
                        {
                            "part": part.id,
                            "span_mm": round(ceiling.span_mm, 3),
                            "area_mm2": round(ceiling.area_mm2, 3),
                            "faces": ceiling.faces,
                        }
                    )
                    continue
                if worst is None or ceiling.area_mm2 > worst[1]:
                    worst = (
                        part.id,
                        ceiling.area_mm2,
                        ceiling.worst_angle_deg,
                        ceiling.faces,
                    )
        self.bridged_overhangs = tuple(bridged)
        if worst is None:
            return None

        part_id, area, angle, faces = worst
        magnitude: dict = {
            "part": part_id,
            "area_mm2": area,
            "angle_from_horizontal_deg": angle,
            "min_angle_from_horizontal_deg": limit,
            "faces": faces,
            "min_area_mm2": OVERHANG_MIN_AREA_MM2,
            "bed_tol_mm": self.bed_tol_mm,
            "bridge_mm": bridge_mm,
            # Both names for the pose that was judged: `orientation_deg` is the
            # canonical one, `print_orientation_deg` stays because reports and
            # policies written before P2-2 key on it.
            "orientation_deg": [float(value) for value in orientation],
            "print_orientation_deg": [float(value) for value in orientation],
            "orientation_source": self.orientation_source,
            "method": "face_normal_vs_build_axis_z",
            "approximation": (
                "the resolved print orientation is taken as the build pose; the bed "
                "is the rotated part's own lowest plane; a ceiling whose horizontal "
                "span is at most bridge_mm is treated as bridged, not unsupported"
            ),
        }
        if bridged:
            # What the span rule let pass in the *same* part: the finding must not
            # hide the fact that a decision was taken about other ceilings.
            magnitude["bridged_ceilings"] = list(bridged)
        if self.recommendation is not None:
            # The recommendation is about the *insert*; `area_mm2` above is about the
            # worst part in that pose. Keeping both is what lets a reader see the two
            # numbers describe different solids (see the module docstring).
            magnitude["recommended_overhang_area_mm2"] = round(
                self.recommendation.overhang_area_mm2, 3
            )
        return FailureCertificate(
            code=CODE_UNSUPPORTED_OVERHANG,
            family=_FAMILY,
            locus={
                "spec": view.spec.spec_id,
                "revision": view.spec.revision,
                "part": part_id,
                "process": view.spec.constraints.process,
            },
            magnitude=magnitude,
            observed_by=self.spec.id,
            evidence_ref="overhang_scan",
        )

    def _bridge_mm(self, view: DesignView) -> float:
        """The span this process bridges: 0 for resin, `None` for powder bed/CNC.

        `bridge_mm=None` means the question does not arise (the process does not need
        support at all), and 0 means it never forgives a span (resin). Both end up as
        "no bridging" here; the difference is already carried by `overhang_deg`.
        """
        profile = profile_for(view.spec.constraints.process)
        return float(profile.bridge_mm) if profile and profile.bridge_mm else 0.0
