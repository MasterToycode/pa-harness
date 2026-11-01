"""A round peg in a blind socket: the demo case, built rather than found.

The proposal's headline example is "peg 10.2 mm in a 10.0 mm hole", which v1 had
to look for in a dataset and v2 gets to *decide*. This builder is where that
decision becomes triangles.

Why a blind socket rather than a through hole
---------------------------------------------
A through hole is the obvious model and it is the wrong one for this harness: a
peg with no head, passing through a plate with nothing under it, has nothing
holding it against gravity, so the physics verifier finds it on the floor after
the settle and every peg_hole case fails for a reason that is not a design
defect. Measured, not argued. A blind socket is a real fit -- a dowel located by
a socket and its fit -- it gives the peg a floor to sit on, and it keeps the
insertion path a straight line along +Z. The through variant is a *different
part*, so it belongs in a future `peg_hole_through` kind, not in a flag here.

The seated pose is therefore "peg bottom on the socket floor":
`insert.pose_mm.z = body_top_z - engage_mm`. The peg then stands
`peg_length_mm - engage_mm` proud of the top face, which is the length the
physics body and the gripper get to hold. The peg bottom and the socket floor are
**coplanar, not overlapping**: at the seated pose the two solids touch over a
face, so their intersection has zero volume. That is the design-plane rule
("intersection volume > 1e-6 mm3" counts as interference), and a fixture must not
shave a micron off the pose to make the arithmetic look tidier -- a normal design
must never be reported as interference.

Local frames (the base class puts both in world coordinates)
-----------------------------------------------------------
* body: blank centred on the origin; the socket axis crosses the body at
  `feature_xy_mm` in X/Y, its opening is the top face and its floor is
  `engage_mm` below that.
* insert: origin at the peg's **leading tip**; the peg runs from z = 0 to
  `peg_length_mm`, a square handle sits on top of it, and `approach_mm` steps the
  whole insert back up the axis from the seated pose.

The handle is a fixed proportion, not a parameter
-------------------------------------------------
`2 * d_peg` across (floored at 12 mm) and `d_peg / 2` tall (floored at 4 mm), so
even a 2 mm peg has a body a gripper and a printer can see. The spec has no field
for "how big is the handle", and inventing one would be a second truth about a
part the interface does not own; the contract says insert = public feature plus a
piece of gripping body, so that is what this is.

Resolution
----------
Peg and socket are 64-gon cylinders inscribed in their nominal circles: 6 um
tighter than the nominal surface on a 10 mm diameter, an order of magnitude below
any fit band `FitConstraints` can express, so the nominal diameters remain the
contract. Fixing the side count is part of the determinism claim -- a tessellation
left to a library default would make the STL bytes depend on trimesh's version.
"""

from __future__ import annotations

import trimesh

from pah.design.connectors import ConnectorBuilderBase, blank_box
from pah.design.spec import PartSpec, Vec3

__all__ = ["PegHoleBuilder"]

#: Sides of the peg and socket cylinders; see the module docstring for why 64.
SECTIONS = 64


class PegHoleBuilder(ConnectorBuilderBase):
    """A cylindrical peg that seats on the floor of a blind socket along +Z."""

    kind = "peg_hole"
    local_mate_axis: Vec3 = (0.0, 0.0, 1.0)

    def build_local(
        self, body: PartSpec, insert: PartSpec, params
    ) -> tuple[trimesh.Trimesh, trimesh.Trimesh]:
        """Body with a blind socket, and a peg with a handle above it."""
        top_local = float(body.size_mm[2]) / 2.0
        # One millimetre above the top face keeps the cutter's opening off the
        # blank's own face, which is where CSG kernels produce slivers. The floor
        # stays exactly `engage` below the top, because that is where the insert's
        # tip lands when seated.
        socket = trimesh.creation.cylinder(
            radius=params.d_hole_mm / 2.0,
            height=params.engage_mm + 1.0,
            sections=SECTIONS,
            transform=trimesh.transformations.translation_matrix(
                (
                    float(params.feature_xy_mm[0]),
                    float(params.feature_xy_mm[1]),
                    top_local + 1.0 - (params.engage_mm + 1.0) / 2.0,
                )
            ),
        )
        body_mesh = trimesh.boolean.difference(
            [blank_box(body.size_mm), socket], engine="manifold"
        )

        peg_length = params.peg_length_mm
        peg = trimesh.creation.cylinder(
            radius=params.d_peg_mm / 2.0,
            height=peg_length,
            sections=SECTIONS,
            transform=trimesh.transformations.translation_matrix((0.0, 0.0, peg_length / 2.0)),
        )
        handle_xy = max(2.0 * params.d_peg_mm, 12.0)
        handle_h = max(params.d_peg_mm / 2.0, 4.0)
        handle = trimesh.creation.box(
            extents=(handle_xy, handle_xy, handle_h),
            transform=trimesh.transformations.translation_matrix(
                (0.0, 0.0, peg_length + handle_h / 2.0)
            ),
        )
        # The peg's top cap is coplanar with the handle's bottom face. A union is
        # still the right construct: two touching shells would each be closed but
        # the part would be two bodies, and physics would drop one of them.
        insert_mesh = trimesh.boolean.union([peg, handle], engine="manifold")
        return body_mesh, insert_mesh
