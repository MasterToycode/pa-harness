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

from pah.core.model import FailureCertificate
from pah.design.api import CODE_INSERTION_PATH_BLOCKED, DesignView

from .base import VerifierSpec
from .interface_interference import (
    INTERFERENCE_MIN_VOLUME_MM3,
    intersection_volume_mm3,
    unit_axis,
)

__all__ = ["CODES", "SAMPLES", "InsertionPath"]

CODES: tuple[str, ...] = (CODE_INSERTION_PATH_BLOCKED,)

#: Samples between `+approach` and the seat. 14 keeps the step at 0.43 mm for the
#: default 6 mm approach while the whole sweep stays around 60 ms.
SAMPLES = 14

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
        """One certificate for the first blocked position, else none."""
        approach = float(view.approach_mm)
        if approach <= 0.0:
            # No approach distance means no path: the insert is already seated.
            return []

        body, insert = view.body, view.insert
        direction = unit_axis(view.mate_axis)
        step = approach / SAMPLES
        # Descending from `+approach` to one step above the seat: the first
        # collision found is the shallowest obstruction, which is what a repair
        # has to clear. `k = 0` (the seat) is excluded on purpose — the peg rests
        # on the socket floor there, which is a contact, not an obstruction.
        for k in range(SAMPLES, 0, -1):
            height = step * k
            moved = insert.copy()
            moved.apply_translation(direction * height)
            volume = intersection_volume_mm3(body, moved)
            if volume <= _MIN_VOLUME_MM3:
                continue
            interface = view.spec.interface
            return [
                FailureCertificate(
                    code=CODE_INSERTION_PATH_BLOCKED,
                    family=_FAMILY,
                    locus={
                        "spec": view.spec.spec_id,
                        "revision": view.spec.revision,
                        "interface": interface.id,
                        "parts": [interface.a, interface.b],
                        "kind": interface.kind,
                    },
                    magnitude={
                        "blocked_at_mm": height,
                        "volume_mm3": volume,
                        "approach_mm": approach,
                        "step_mm": step,
                        "samples": SAMPLES,
                        "mate_axis": [float(value) for value in direction],
                    },
                    observed_by=self.spec.id,
                    evidence_ref=f"mate_axis_sweep@{SAMPLES}",
                )
            ]
        return []
