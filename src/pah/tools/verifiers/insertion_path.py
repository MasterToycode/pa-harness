"""`insertion_path`: can the insert reach the seat along its mating direction?

The sweep
---------
The insert is translated along `mate_axis` from `+approach_mm` above the seat down
to one step short of it, and at every sample the overlap with the body is
measured. Any sample with more shared volume than the interference noise floor
means the insert cannot pass: `insertion_path_blocked`.

The seated pose is deliberately **not** sampled. Whether the two solids overlap
where they finally rest is `interface_interference`'s finding, with its own code;
repeating it here would make one defect look like two and teach a policy to apply
two repairs for it.

Where the numbers come from
---------------------------
`approach_mm` and `mate_axis` are the *built* geometry's insertion facts, and the
contract puts them on the view (`DesignView.mate_axis`, `DesignView.approach_mm`,
`design/api.py` §2) rather than making each verifier re-derive them. That matters
beyond tidiness: a dovetail slides in along +X, so a sweep along a hard-coded +Z
would report every slide fit as blocked — a false finding no repair operator can
address, on a design that is perfectly insertable. The verifier therefore carries
no axis and no approach of its own; the materialiser tells it what was built.

Why this verifier is worth its medium tier
------------------------------------------
For a straight bore and a rigid body, a seat that fits implies a path that fits:
translating the body along the bore's own axis cannot create a collision it did
not have. So every certificate this verifier can produce is a **structural**
obstruction — a stepped or undercut bore, a bore that is not straight, an insert
whose collar is wider than the entry — i.e. an assembly that no sequence of moves
could have produced. That is exactly the class of error a seat-only check calls
healthy, and `stepped_fixture()` is the proof: it is clean at the seat, blocked on
the way in.

Resolution and cost
-------------------
14 samples over the approach, all of them booleans on the same two solids
(~4 ms each at 64 facets). One step at the default 6 mm approach is 0.43 mm, so
`blocked_at_mm` is accurate to about that; a finer sweep would cost proportionally
more for a number a repair policy does not need to that precision — it needs to
know *that* the path is blocked and roughly how far in it gets.
"""

from __future__ import annotations

import numpy as np
import trimesh

from pah.core.model import FailureCertificate
from pah.design.api import (
    CODE_APPROACH_NOT_CLEAR,
    CODE_INSERTION_PATH_BLOCKED,
    DesignView,
)

from .base import VerifierSpec
from .interface_interference import (
    INTERFERENCE_MIN_VOLUME_MM3,
    intersection_volume_mm3,
    unit_axis,
)

__all__ = ["CODES", "SAMPLES", "InsertionPath"]

CODES: tuple[str, ...] = (CODE_INSERTION_PATH_BLOCKED, CODE_APPROACH_NOT_CLEAR)

#: Samples between `+approach` and the seat. 14 keeps the step at 0.43 mm for the
#: default 6 mm approach while the whole sweep stays around 60 ms.
SAMPLES = 14

#: Refinement (audit P1-3). A fixed 14 samples over a 6 mm approach steps 0.43 mm, so a
#: wall thinner than that sits between two samples and the sweep calls the path clear.
#: Any interval whose ends come within REFINE_BELOW_MM is resampled to MIN_STEP_MM.
MIN_STEP_MM = 0.1
REFINE_BELOW_MM = 0.5
#: Ceiling on refinement per interval and on booleans per check. A pathological pair
#: must not hang a run; hitting the ceiling means "not shown", never "clear".
MAX_REFINEMENT_STEPS = 64
MAX_BOOLEANS = 200

#: `assembly`: an insert that cannot reach its seat is a property of the joint.
_FAMILY = "assembly"

#: A position in the sweep is blocked by the same volume that makes a seated pair
#: interfere — one noise floor for both booleans, so the two verifiers can never
#: disagree about whether 1e-4 mm3 of sliver is a collision.
_MIN_VOLUME_MM3 = INTERFERENCE_MIN_VOLUME_MM3


class InsertionPath:
    """The insert translated down its mating axis, one boolean intersect per step."""

    spec = VerifierSpec(
        id="insertion_path",
        codes=CODES,
        cost_class="medium",
        fixtures={
            "known_good": ["fixture-standard: straight bore, clean at every sample"],
            "known_bad": [
                "fixture-oversized: too wide for the bore, blocked at the first sample",
                "fixture-stepped-bore: clean at the seat, blocked by the narrow entry",
            ],
        },
    )

    def fixtures(self) -> tuple[list[DesignView], list[DesignView]]:
        """One clean sweep and two that cannot reach the seat for different reasons."""
        from ._design_fixtures import oversized_fixture, standard_fixture, stepped_fixture

        return (
            [standard_fixture()],
            [oversized_fixture(), stepped_fixture()],
        )

    def check(self, view: DesignView) -> list[FailureCertificate]:
        """The first blocked position along the sweep, or none.

        Two changes for audit P1-3, both about not claiming more than a sampler can:

        * **the start must be reachable.** Before, the sweep began at `+approach` and
          never asked whether *that* pose was clear. An insert starting inside the body
          reported "no obstruction" — the strongest possible false positive. Now the
          first pose is tested, and a start that collides is `approach_not_clear`.
        * **the sweep refines where it matters.** A fixed 14 samples across a 6 mm
          approach steps 0.43 mm, so a 0.2 mm wall between two samples is stepped over
          and reported clear. Any interval whose end poses come within
          `REFINE_BELOW_MM` of the body is resampled down to `MIN_STEP_MM`, bounded by
          `MAX_BOOLEANS` so a pathological pair cannot hang a run.

        Straight-line only: this is one translation along `mate_axis`, and a doorknob
        path (rotate then slide) is out of scope. The certificate says so.
        """
        approach = float(view.approach_mm)
        if approach <= 0.0:
            # No approach distance means no path: the insert is already seated.
            return []

        body, insert = view.body, view.insert
        direction = unit_axis(view.mate_axis)
        interface = view.spec.interface

        def pose(height: float) -> trimesh.Trimesh:
            moved = insert.copy()
            moved.apply_translation(direction * height)
            return moved

        def cert(code: str, height: float, volume: float, extra: dict) -> FailureCertificate:
            return FailureCertificate(
                code=code,
                family=_FAMILY,
                locus={
                    "spec": view.spec.spec_id,
                    "revision": view.spec.revision,
                    "interface": interface.id,
                    "parts": [interface.a, interface.b],
                    "kind": interface.kind,
                },
                magnitude={
                    "blocked_at_mm": round(height, 6),
                    "volume_mm3": round(volume, 6),
                    "approach_mm": approach,
                    "mate_axis": [float(value) for value in direction],
                    "path": "single straight pull along mate_axis",
                    **extra,
                },
                observed_by=self.spec.id,
                evidence_ref="mate_axis_sweep@adaptive",
            )

        budget = {"booleans": 0}

        def overlap(height: float) -> float:
            if budget["booleans"] >= MAX_BOOLEANS:
                return 0.0  # budget exhausted: stop claiming, do not invent a block
            budget["booleans"] += 1
            return intersection_volume_mm3(body, pose(height))

        # 1) the start pose has to be reachable from outside
        start_pose = pose(approach)
        start_volume = overlap(approach)
        if start_volume > _MIN_VOLUME_MM3:
            # A start pose that already overlaps the body is *both* an obstruction at
            # the first sample and, when the approach did not get the insert clear of
            # the body's extent, a too-short approach. Two certificates, because they
            # are two different repairs: back the insert out further, or clear what it
            # is standing in. Reporting only the first (the old behaviour) is what let
            # an insert that began inside the body pass as "no obstruction".
            certificates = [
                cert(
                    CODE_INSERTION_PATH_BLOCKED,
                    approach,
                    start_volume,
                    {
                        "step_mm": approach / SAMPLES,
                        "samples": SAMPLES,
                        "booleans": budget["booleans"],
                        "entry_blocked": True,
                    },
                )
            ]
            if self._inside_bounds(start_pose, body):
                certificates.append(
                    cert(
                        CODE_APPROACH_NOT_CLEAR,
                        approach,
                        start_volume,
                        {
                            "reason": (
                                "the insert is still inside the body's extent at the "
                                "start of a %.3f mm approach, so it cannot be brought "
                                "in from outside along mate_axis" % approach
                            ),
                            "booleans": budget["booleans"],
                        },
                    )
                )
            return certificates

        # 2) coarse ladder, descending: the first block found is the shallowest
        # The ladder runs from the start down to MIN_STEP_MM above the seat, not to
        # one coarse step above it. The old bottom sample was pproach / SAMPLES
        # (0.43 mm), so the last 0.43 mm was never swept — and a thin feature living
        # there (a membrane across the bore, the flash a first print leaves) was
        # reported clear by construction. The seat itself stays excluded: at height 0
        # the peg rests on the socket floor, which is a contact, not an obstruction.
        span = max(approach - MIN_STEP_MM, 0.0)
        coarse = [MIN_STEP_MM + span * (k / SAMPLES) for k in range(SAMPLES, -1, -1)]
        clearances = [self._clearance(body, pose(height)) for height in coarse]
        for index, height in enumerate(coarse):
            volume = overlap(height)
            if volume > _MIN_VOLUME_MM3:
                return [
                    cert(
                        CODE_INSERTION_PATH_BLOCKED,
                        height,
                        volume,
                        {
                            "step_mm": approach / SAMPLES,
                            "samples": SAMPLES,
                            "booleans": budget["booleans"],
                        },
                    )
                ]
            # Refine the gap to the previous (higher) sample when either end is close
            # to the body: that is where a thin wall can hide.
            if index == 0:
                continue
            upper = coarse[index - 1]
            near = min(clearances[index], clearances[index - 1])
            if near > REFINE_BELOW_MM:
                continue
            span = upper - height
            if span <= MIN_STEP_MM:
                continue
            steps = min(int(span / MIN_STEP_MM) + 1, MAX_REFINEMENT_STEPS)
            for fine in range(1, steps + 1):
                mid = upper - span * (fine / (steps + 1))
                fine_volume = overlap(mid)
                if fine_volume > _MIN_VOLUME_MM3:
                    return [
                        cert(
                            CODE_INSERTION_PATH_BLOCKED,
                            mid,
                            fine_volume,
                            {
                                "step_mm": round(span / (steps + 1), 6),
                                "samples": budget["booleans"],
                                "refined": True,
                                "booleans": budget["booleans"],
                            },
                        )
                    ]
        return []

    @staticmethod
    def _inside_bounds(moved: trimesh.Trimesh, body: trimesh.Trimesh) -> bool:
        """Is any part of moved inside the body's axis-aligned box?

        The coarse test for "the stated approach does not clear the part". It is
        deliberately an AABB question: making it a distance question would need the
        very boolean we are trying to avoid, and "still within the body's extent" is
        the fact that decides which of the two certificates is true.
        """
        low, high = body.bounds
        vertices = np.asarray(moved.vertices, dtype=float)
        inside = np.all((vertices >= low) & (vertices <= high), axis=1)
        return bool(inside.any())

    @staticmethod
    def _clearance(body: trimesh.Trimesh, moved: trimesh.Trimesh) -> float:
        """Rough separation between the moved insert and the body, in mm.

        Vertex-to-surface distance is enough for the one decision it drives — *should
        this gap be refined* — and costs one nearest-point query instead of a boolean.
        Being approximate here is safe in the only direction that matters: it can
        refine a gap that did not need it, never skip one that did, because `2 *
        noise` is far below any real clearance.
        """
        try:
            _, distance, _ = trimesh.proximity.closest_point(body, moved.vertices)
        except Exception:  # noqa: BLE001 - an unmeasurable gap means "refine it"
            return 0.0
        finite = distance[np.isfinite(distance)]
        return float(finite.min()) if len(finite) else 0.0
