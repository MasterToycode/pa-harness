"""Does the joint hold, or is it only friction?

Why this check exists at all
----------------------------
`physics_settle` models "a peg-shaped insert seated in a bore": it measures the bore
wall around the axis and refuses a view where the wall is thin compared to the insert.
A decomposed part is a plate or a block 60 mm wide around a 4 mm socket, so that guard
fires on every pair and the whole physics tier said `not evaluated` for multi-part
assemblies (audit P1-2). That is honest, but it left the one question a person actually
asks unanswered: **will it fall apart?**

What this verifier can and cannot prove
---------------------------------------
It is a *geometric* retention test, not a friction model. Two ways a joint stays put:

1. **form closure** — something blocks the pull-out: a snap hook's undercut, a dovetail
   flank, a lip. Measurable exactly, by translating the mating part outward along the
   interface axis and boolean-testing the overlap with its host;
2. **the seat carries it** — gravity pushes the mating part *into* the seat (a peg
   standing in an upward-opening socket), so even a plain clearance fit does not fall.

What is deliberately not claimed: a horizontal peg in a horizontal hole is held only by
friction, and friction depends on material, surface finish and print orientation. The
verifier therefore says nothing about it — `dot(gravity, pull-out) == 0` is neither
"holds" nor "falls", and inventing a friction coefficient would turn a guess into a
certificate. Only a joint whose pull-out direction points **downward** is reported: that
is the case where geometry provably cannot hold it, and that is a real defect.

The three states the report renders
-----------------------------------
* no certificate, no note — retained (form closure, or the seat carries it);
* `joint_not_retained` — nothing blocks the pull-out *and* gravity points that way;
* a note — the meshes could not be tested (a tool boundary, not a verdict).
"""

from __future__ import annotations

import numpy as np
import trimesh

from pah.core.model import FailureCertificate
from pah.design.api import CODE_JOINT_NOT_RETAINED, DesignView

from .base import VerifierSpec
from .interface_interference import INTERFERENCE_MIN_VOLUME_MM3, unit_axis

__all__ = ["CODES", "PULL_MM", "JointRetention"]

CODES: tuple[str, ...] = (CODE_JOINT_NOT_RETAINED,)

#: How far the mating part is pulled before the overlap test. Half a millimetre is
#: enough to engage any undercut a hobby print can produce (a 0.2 mm lip is already a
#: lip) and small enough that a clearance of 0.2 mm is not crossed by the motion alone.
PULL_MM = 0.5

#: Which way the world pulls.
GRAVITY: tuple[float, float, float] = (0.0, 0.0, -9.81)

_FAMILY = "assembly"


class JointRetention:
    """Form closure or a gravity-carried seat; nothing else is claimed."""

    spec = VerifierSpec(
        id="joint_retention",
        codes=CODES,
        cost_class="medium",
        fixtures={
            "known_good": [
                "fixture-seated-peg: an upward-opening socket, gravity pushes the peg in",
            ],
            "known_bad": [
                "fixture-hanging-peg: the same pair flipped, so pulling out is downhill",
            ],
        },
    )

    def __init__(self, *, pull_mm: float = PULL_MM) -> None:
        self.pull_mm = float(pull_mm)
        self.last_note = ""

    # -- fixtures ----------------------------------------------------------
    def fixtures(self) -> tuple[list[DesignView], list[DesignView]]:
        """One retained pair and one pair that gravity takes apart.

        Both are built from `trimesh` booleans rather than by the materialiser, for the
        reason `_design_fixtures.py` gives: a verifier that could not be judged against
        geometry the pipeline did not produce would be marking its own homework.
        """
        from pah.design.spec import DesignSpec, InterfaceSpec, PartSpec

        def build(*, flipped: bool) -> DesignView:
            plate = trimesh.creation.box(extents=(30.0, 30.0, 8.0))
            plate.apply_translation((0.0, 0.0, -4.0))  # top face at z = 0
            bore = trimesh.creation.cylinder(radius=2.2, height=8.0, sections=32)
            bore.apply_translation((0.0, 0.0, -4.0))  # blind socket: mouth z=0, floor z=-8
            host = trimesh.boolean.difference([plate, bore], engine="manifold")

            peg = trimesh.creation.cylinder(radius=2.0, height=10.0, sections=32)
            peg.apply_translation((0.0, 0.0, 1.0))  # seated: tip at the socket floor
            if flipped:
                host.apply_transform(
                    trimesh.transformations.rotation_matrix(np.pi, (1.0, 0.0, 0.0))
                )
                peg.apply_transform(
                    trimesh.transformations.rotation_matrix(np.pi, (1.0, 0.0, 0.0))
                )

            spec = DesignSpec(
                spec_id="retention-fixture",
                parts=(
                    PartSpec(id="body", role="body", size_mm=(30.0, 30.0, 8.0)),
                    PartSpec(
                        id="peg",
                        role="insert",
                        size_mm=(4.0, 4.0, 10.0),
                        pose_mm=(0.0, 0.0, 1.0),
                    ),
                ),
                interfaces=(
                    InterfaceSpec(
                        id="iface",
                        kind="peg_hole",
                        a="body",
                        b="peg",
                        params={"d_peg_mm": 4.0, "d_hole_mm": 4.4, "engage_mm": 4.0},
                    ),
                ),
            )
            return DesignView(
                spec=spec,
                meshes={"body": host, "peg": peg},
                # The flipped fixture turns the pair over, so the direction from host to
                # mating part turns with it: pulling out is now downward, which is what
                # makes gravity the thing that takes it apart.
                mate_axis=(0.0, 0.0, -1.0) if flipped else (0.0, 0.0, 1.0),
                approach_mm=6.0,
            )

        return [build(flipped=False)], [build(flipped=True)]

    # -- the check ---------------------------------------------------------
    def check(self, view: DesignView) -> list[FailureCertificate]:
        self.last_note = ""
        try:
            host = view.body
            guest = view.insert
        except Exception as error:  # noqa: BLE001 - a missing mesh is not a verdict
            self.last_note = f"joint retention not evaluated: {error}"
            return []

        axis = np.asarray(unit_axis(view.mate_axis), dtype=float)
        # mate_axis points from the host to the mating part, so pulling it *out* moves
        # it along +axis. (The first version moved it along -axis, which pushed the peg
        # deeper into the socket and reported nothing — the sign is the whole test.)
        moved = guest.copy()
        moved.apply_translation(axis * self.pull_mm)
        blocked = _overlap_volume(host, moved)
        if blocked > INTERFERENCE_MIN_VOLUME_MM3:
            # Form closure: something catches it. This is the good case and it says
            # nothing, because a verifier may only report what is wrong.
            return []

        gravity = np.asarray(GRAVITY, dtype=float)
        downhill = float(np.dot(gravity, axis))
        if downhill <= 1e-9:
            # Gravity does not pull it out: either the seat carries it or the joint is
            # friction-only in a direction gravity is silent about. Neither is a proven
            # defect, and the second is not this verifier's to call.
            return []

        return [
            FailureCertificate(
                code=CODE_JOINT_NOT_RETAINED,
                family=_FAMILY,
                locus={
                    "spec": view.spec.spec_id,
                    "revision": view.spec.revision,
                    "interface": view.spec.interface.id,
                    "parts": [view.spec.body.id, view.spec.insert.id],
                },
                magnitude={
                    "pulled_mm": round(self.pull_mm, 4),
                    "overlap_volume_mm3": round(blocked, 6),
                    "gravity_along_pull_mps2": round(downhill, 4),
                    "min_volume_mm3": INTERFERENCE_MIN_VOLUME_MM3,
                },
                observed_by="joint_retention",
                evidence_ref="pull_out_boolean@manifold",
            )
        ]


def _overlap_volume(first: trimesh.Trimesh, second: trimesh.Trimesh) -> float:
    """Volume shared by two solids; 0.0 when the boolean cannot produce one."""
    shared = trimesh.boolean.intersection([first, second], engine="manifold")
    if isinstance(shared, trimesh.Scene):
        geometries = list(shared.geometry.values())
        if not geometries:
            return 0.0
        shared = trimesh.util.concatenate(geometries) if len(geometries) > 1 else geometries[0]
    if not isinstance(shared, trimesh.Trimesh):
        return 0.0
    volume = float(shared.volume)
    return volume if np.isfinite(volume) and volume > 0.0 else 0.0
