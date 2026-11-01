"""A trapezoidal tongue that can only enter along X: the "cannot pull straight out" kind.

The other two kinds are retained by friction or by a spring; this one is retained
by having flanks. That is why it is the kind a `switch_connector` repair reaches
for when the failure is "it pulled apart", and why its mate axis is +X: the part
slides in, it does not press in.

Local frames (the base class puts both in world coordinates)
-----------------------------------------------------------
* body: blank centred on the origin, minus a trapezoidal groove pocket that is
  open at the body's **+X face** and at its **top face**. The cross-section lives
  in Y/Z: half-width `width_mm / 2` at the opening, widening by
  `depth_mm * tan(angle_deg)` per side at the groove floor. The widening is the
  whole point: a tongue that cannot be lifted out of a channel that narrows
  upward.
* insert: origin at the tongue's **leading face**, on the plane of its top face
  -- which is the body's top face when seated. So `insert.pose_mm`'s X is where
  the slide begins and its Z is the body's top surface; `approach_mm` then steps
  the insert back out of the pocket along +X.

Seated, the tongue's leading face is one engagement in from the body's +X face and
the pocket floor is `clearance` further in, so the origin lands at
`(size_x / 2 - engage, 0, +size_z / 2)` in the body's local frame -- the datum
`seated_pose_mm` reports, and where a `switch_connector` has to move an insert it
just re-kinded.

Clearance
---------
`clearance_mm` is per side, and it is applied both ways: the tongue is
`clearance` narrower than the groove at the opening and `clearance` shallower
than the groove is deep, so the seated pose cannot interfere with the pocket
floor or either flank. The flanks stay parallel to the groove's because both use
the same `angle_deg`; the gap is `clearance` measured along Y, which is what the
fit verifier compares against its band.

The handle
----------
A block above the opening, `max(3 * width, 20) mm` wide and `max(0.8 * depth, 6)
mm` tall, exactly as long as the engagement. It is what makes the retention
geometric rather than notional: it is wider than the groove, so the tongue cannot
be lifted out, and it is a fixed proportion because the spec has no field for it
and a second way to say "how big is the handle" would be a second truth.

Its underside rests on the plate, because that is what a slider does. A coplanar
face contact is a **zero-volume** intersection -- a zero-thickness sheet if it is
computed at all -- not interference: the design-plane rule is "intersection
volume > 1e-6 mm3", and two solids that touch over an area have no volume in
common. Nothing here should add a token gap to make that look tidier, since the
gap would be a number the spec does not contain.
"""

from __future__ import annotations

import math

import numpy as np
import trimesh

from pah.design.connectors import ConnectorBuilderBase, blank_box
from pah.design.spec import DesignSpec, PartSpec, Vec3

__all__ = ["DovetailBuilder"]

#: `extrude_polygon` builds in a (Y, Z) cross-section extruded along +Z; this
#: permutation reads it back as "extruded along +X", which is the slide axis. It
#: is a rotation (determinant +1), so face winding -- and the volume's sign --
#: survives the move.
_EXTRUDE_ALONG_X = np.array(
    [
        [0.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def _trapezoid_prism(
    *, half_top: float, half_bottom: float, depth: float, length: float
) -> trimesh.Trimesh:
    """A trapezoidal prism in the local frame: X in `[0, length]`, Z in `[-depth, 0]`.

    `half_top` is the half-width at the opening (Z = 0), `half_bottom` at the
    floor (Z = -depth). Both the groove cutter and the tongue are this shape, so
    the clearance between them is visible in one place instead of being the
    difference between two differently-written constructions.
    """
    section = trimesh.path.polygons.Polygon(
        [
            (-half_top, 0.0),
            (-half_bottom, -depth),
            (half_bottom, -depth),
            (half_top, 0.0),
        ]
    )
    solid = trimesh.creation.extrude_polygon(section, height=length)
    solid.apply_transform(_EXTRUDE_ALONG_X)
    return solid


class DovetailBuilder(ConnectorBuilderBase):
    """A trapezoidal groove pocket and the matching tongue plus a handle."""

    kind = "dovetail"
    local_mate_axis: Vec3 = (1.0, 0.0, 0.0)

    def build_local(
        self, body: PartSpec, insert: PartSpec, params
    ) -> tuple[trimesh.Trimesh, trimesh.Trimesh]:
        """Body with a trapezoidal pocket, and a tongue plus handle."""
        size_x, _, size_z = (float(value) for value in body.size_mm)
        tan_angle = math.tan(math.radians(params.angle_deg))
        half_open = params.width_mm / 2.0
        half_floor = half_open + params.depth_mm * tan_angle
        top_local = size_z / 2.0
        pocket_len = params.engage_mm + params.clearance_mm

        # The pocket starts one engagement in from the +X face and runs one
        # millimetre past it: the extra keeps the cutter off the face the blank
        # and the cutter would otherwise share, while removing nothing the pocket
        # did not already remove.
        groove = _trapezoid_prism(
            half_top=half_open,
            half_bottom=half_floor,
            depth=params.depth_mm,
            length=pocket_len + 1.0,
        )
        groove.apply_translation((size_x / 2.0 - pocket_len, 0.0, top_local))
        body_mesh = trimesh.boolean.difference(
            [blank_box(body.size_mm), groove], engine="manifold"
        )

        tongue_depth = params.depth_mm - params.clearance_mm
        tongue_half_top = half_open - params.clearance_mm
        tongue = _trapezoid_prism(
            half_top=tongue_half_top,
            half_bottom=tongue_half_top + tongue_depth * tan_angle,
            depth=tongue_depth,
            length=params.engage_mm,
        )
        handle_w = max(3.0 * params.width_mm, 20.0)
        handle_h = max(0.8 * params.depth_mm, 6.0)
        handle = trimesh.creation.box(
            extents=(params.engage_mm, handle_w, handle_h),
            transform=trimesh.transformations.translation_matrix(
                (params.engage_mm / 2.0, 0.0, handle_h / 2.0)
            ),
        )
        insert_mesh = trimesh.boolean.union([tongue, handle], engine="manifold")
        return body_mesh, insert_mesh

    def _seat_point_mm(self, spec: DesignSpec) -> Vec3:
        """The tongue's leading face on the top plane, in the body's local frame.

        The pocket opens at the body's +X face and is `engage + clearance` long,
        so the tongue's leading face -- held off the pocket floor by `clearance`
        -- ends up exactly `engage` in from that face.
        """
        params = spec.interface.validated_params()
        size_x, _, size_z = (float(value) for value in spec.body.size_mm)
        return (size_x / 2.0 - params.engage_mm, 0.0, size_z / 2.0)
