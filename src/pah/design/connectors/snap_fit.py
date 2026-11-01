"""A cantilever hook that snaps into an undercut: the "no fastener" alternative.

Local frames (the base class puts both in world coordinates)
-----------------------------------------------------------
* body: blank centred on the origin, minus (a) a square **through slot** of side
  `slot_w_mm` along Z and (b) a wider **undercut chamber** whose ceiling sits
  `engage_mm` below the top face. The chamber opens into the slot, so the solid
  keeps one closed shell and the slot stays a valid insertion route.
* insert: origin at the hook's **leading tip**. The hook is a square flange
  `hook_w_mm` across and `hook_h_mm` tall; the cantilever beam (`beam_thk_mm` in
  X, `hook_w_mm` in Y, `beam_len_mm` long) rises from the hook's top face.

Approximation, stated plainly
-----------------------------
The insert is a **rigid** body, drawn in the shape it has *after* it has sprung
into the undercut. A real snap hook is wider than its slot when free and narrows
elastically while it passes; a rigid mesh of that free shape could not be
inserted at all, and the insertion-path verifier would be right to reject it. So
this model keeps `hook_w <= slot_w` -- the lateral play
`(slot_w - hook_w) / 2` the parameter model reports as this kind's clearance --
makes the chamber `hook_h` deeper than the slot on each side, and leaves the
spring action unmodelled.

What the model therefore *does* prove: both solids are watertight, positive
volume, non-interfering when seated, with a clear straight-line insertion path.
What it does *not* prove: retention force. That is a physics question (stiffness,
deflection, pull-out), not a triangles question, and no design-plane verifier
claims it -- `docs/internal/geometry-notes.md` repeats this where the verifiers
will read it.

The chamber is `hook_h + 2 * play` tall and `play` clear of the seated hook on
every side, which is why the seated pose never interferes even though the hook
fills the undercut it is meant to catch on.
"""

from __future__ import annotations

import trimesh

from pah.design.connectors import ConnectorBuilderBase, blank_box
from pah.design.spec import PartSpec, Vec3

__all__ = ["SnapFitBuilder"]


class SnapFitBuilder(ConnectorBuilderBase):
    """A through slot plus an undercut chamber, and a hook on a cantilever beam."""

    kind = "snap_fit"
    local_mate_axis: Vec3 = (0.0, 0.0, 1.0)

    def build_local(
        self, body: PartSpec, insert: PartSpec, params
    ) -> tuple[trimesh.Trimesh, trimesh.Trimesh]:
        """Body with slot and chamber, and a beam with a hook at its tip."""
        play = params.radial_clearance_mm
        top_local = float(body.size_mm[2]) / 2.0

        slot = blank_box((params.slot_w_mm, params.slot_w_mm, float(body.size_mm[2]) + 2.0))
        # The chamber ceiling is `engage` below the top face; the hook's top face
        # seats against it (with `play` of clearance) and the chamber's floor is
        # that much lower again.
        chamber_side = params.slot_w_mm + 2.0 * (params.hook_h_mm + play)
        chamber_height = params.hook_h_mm + 2.0 * play
        chamber = trimesh.creation.box(
            extents=(chamber_side, chamber_side, chamber_height),
            transform=trimesh.transformations.translation_matrix(
                (0.0, 0.0, top_local - params.engage_mm - params.hook_h_mm / 2.0)
            ),
        )
        slotted = trimesh.boolean.difference(
            [blank_box(body.size_mm), slot], engine="manifold"
        )
        body_mesh = trimesh.boolean.difference([slotted, chamber], engine="manifold")

        hook = trimesh.creation.box(
            extents=(params.hook_w_mm, params.hook_w_mm, params.hook_h_mm),
            transform=trimesh.transformations.translation_matrix(
                (0.0, 0.0, params.hook_h_mm / 2.0)
            ),
        )
        beam = trimesh.creation.box(
            extents=(params.beam_thk_mm, params.hook_w_mm, params.beam_len_mm),
            transform=trimesh.transformations.translation_matrix(
                (0.0, 0.0, params.hook_h_mm + params.beam_len_mm / 2.0)
            ),
        )
        insert_mesh = trimesh.boolean.union([hook, beam], engine="manifold")
        return body_mesh, insert_mesh
