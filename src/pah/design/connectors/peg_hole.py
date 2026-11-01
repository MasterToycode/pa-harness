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
`peg_length_mm - engage_mm` proud of the top face. The peg bottom and the socket
floor are **coplanar, not overlapping**: at the seated pose the two solids touch
over a face, so their intersection has zero volume. That is the design-plane rule
("intersection volume > 1e-6 mm3" counts as interference), and a fixture must not
shave a micron off the pose to make the arithmetic look tidier -- a normal design
must never be reported as interference.

Why the insert is just the peg
------------------------------
A peg with a gripping plate on top was the first model, and the physics gate
rejected it for a real reason: MuJoCo collides convex hulls, and a handle wider
than the peg is not convex (the insert's hull/volume ratio measured 1.681), so
the hull erases the step the design depends on -- the peg would behave like a
short fat block and the settle would be meaningless. Two honest answers exist:
decompose the insert into convex pieces, or let the peg be a peg. The second is
cheaper and removes a number the spec never had: the `2 * d_peg` handle was this
file's invention, not the design's. So the insert is one cylinder of diameter
`d_peg_mm` and length `peg_length_mm` -- convex, watertight, volume `pi r^2 L`.

`PartSpec.size_mm` is therefore an **envelope hint** for a peg_hole insert, not a
shape input: the builder reads the interface parameters and the body's blank, and
reads nothing else off the insert. The envelope is still recorded in the
`Assembly`, because that is what a report shows about the part.

Local frames (the base class puts both in world coordinates)
-----------------------------------------------------------
* body: blank centred on the origin; the socket axis crosses the body at
  `feature_xy_mm` in X/Y, its opening is the top face and its floor is
  `engage_mm` below that.
* insert: origin at the peg's **leading tip**; the cylinder runs from z = 0 to
  `peg_length_mm`, and `approach_mm` steps the whole insert back up the axis from
  the seated pose.

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
from pah.design.spec import DesignSpec, PartSpec, Vec3

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
        """Body with a blind socket, and the peg that goes in it."""
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
        insert_mesh = trimesh.creation.cylinder(
            radius=params.d_peg_mm / 2.0,
            height=peg_length,
            sections=SECTIONS,
            transform=trimesh.transformations.translation_matrix((0.0, 0.0, peg_length / 2.0)),
        )
        return body_mesh, insert_mesh

    def _seat_point_mm(self, spec: DesignSpec) -> Vec3:
        """The socket floor's centre, in the body's local frame."""
        params = spec.interface.validated_params()
        return (
            float(params.feature_xy_mm[0]),
            float(params.feature_xy_mm[1]),
            float(spec.body.size_mm[2]) / 2.0 - params.engage_mm,
        )
