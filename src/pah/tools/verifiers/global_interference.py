"""`global_interference`: do *any* two parts of an assembly share space?

The blind spot this exists for
------------------------------
The N-part path judges one interface at a time (`pah/design/pairs.py`): each joint is
handed a two-part *shadow* and the six verifiers run on it. That trick is why six
pieces of physics did not have to be rewritten for decompositions, and it comes with
a hole no amount of per-joint checking can close:

* a shadow contains exactly the two parts its interface joins, so a pair that **no
  interface joins** is never looked at by anybody. Two arms of an L-shaped assembly
  can interpenetrate by millimetres while every joint verifies clean;
* and "every joint is clear" is not "the assembly is clear". A part can be
  individually well fitted to each of its neighbours and still sit where a third,
  non-adjacent part already is.

This verifier asks the missing question: for **every** unordered pair of parts in the
spec, in the seated pose, how much volume do the two solids share?

Adjacent pairs are checked too, and that is a decision
-----------------------------------------------------
It is tempting to skip pairs that have an interface — `interface_interference`
already measures those, so the numbers would be duplicated. Three reasons not to:

1. **The two verifiers do not read the same view.** `interface_interference` is given
   a pair shadow with that interface's own axis and exactly two meshes; this one gets
   the global view and has to find the pairs itself. If the shadow builder ever paired
   the wrong ids, or loaded a mesh from the wrong part, the per-pair check would clear
   a seat forever. An independent second reading of the same solids is what notices.
2. **The seat's requirement is the same as any other pair's: zero volume.** There is
   no overlap a joint is allowed to have. Fit lives in the clearance, not in
   interpenetration, so "they are joined" is not a reason to tolerate material
   occupying the same space.
3. **The claim has to be true as stated.** A report that says "no global interference"
   while an adjacent pair overlaps would be false, and leaving the pair out to dodge a
   duplicate would make the truth depend on which verifier ran first.

The cost is that a seated interference on an *adjacent* pair is reported twice: once
as `geometric_interference` by the interface verifier and once as
`global_interference` here. `locus["interface"]` is set for a pair an interface joins
and absent for a pair none joins, so a report can dedupe or label the two without
re-deriving the topology — which is exactly the information `pairs.py` throws away.

Threshold, cost, and why the numbers are shared
-----------------------------------------------
The noise floor is `interface_interference.INTERFERENCE_MIN_VOLUME_MM3`, imported
rather than restated: two verifiers measuring the same pair with different ideas of
what counts as "no overlap" would eventually disagree about the same 1e-4 mm3 sliver,
and the loop would see one defect flicker on and off between checks.

The whole cost model is one exact bounding-box filter. Two axis-aligned boxes that do
not overlap on at least one axis bound two disjoint point sets, so their shared volume
is exactly zero and no kernel is called. A decomposition stacks its pieces, so most of
the N*(N-1)/2 pairs never touch a boolean — which is what makes an all-pairs check
affordable on a 19-part X-Part job, where the number of pairs is 171.

Fail closed, never silent
-------------------------
A mesh that cannot be read, is empty, or is not watertight makes the shared volume
undefined. That raises here, and the cascade files the round's evidence as
`environment` — unknown, never healthy. Returning an empty list would be the worst
answer available: it looks exactly like "I measured every pair and they are all
clear".

Fixtures
--------
`fixtures()` builds its own two views with `trimesh` and a `DesignSpec`, never through
`pah.design.materialize`: a gate whose proof geometry came from the builder under
test would be marking its own homework (`_design_fixtures.py`, module header). The
pair differs in exactly one number, the distance between the two plates:

* `known_good` — two 40 x 40 x 10 plates 5 mm apart, disjoint;
* `known_bad`  — the same two plates 3 mm into each other: a 40 x 40 x 3 = 4800 mm3
  overlap, four orders of magnitude above the noise floor.
"""

from __future__ import annotations

import itertools

import numpy as np
import trimesh

from pah.core.model import FailureCertificate
from pah.design.api import DesignVerifierRegistry, DesignView
from pah.design.spec import DesignSpec, FitConstraints, InterfaceSpec, PartSpec

from .base import VerifierSpec
from .interface_interference import INTERFERENCE_MIN_VOLUME_MM3, intersection_volume_mm3

__all__ = ["CODE_GLOBAL_INTERFERENCE", "CODES", "GlobalInterference", "register"]

#: The code this verifier owns. Declared here as a constant so the contract file
#: (`pah.design.api.DESIGN_CODES`) and the report layer can cite the same string
#: rather than a literal in three places. Note the direction of the dependency: the
#: contract may *name* this code, but it must not import this module — this module
#: imports `pah.design.api`, and doing it both ways is a cycle.
CODE_GLOBAL_INTERFERENCE = "global_interference"

CODES: tuple[str, ...] = (CODE_GLOBAL_INTERFERENCE,)

#: `assembly`: two parts of one assembly occupying one space is a property of the
#: assembly, not of either solid, and it is not a manufacturing limit.
_FAMILY = "assembly"

#: The fixture blank: 40 x 40 mm face, 10 mm thick. Big enough that the boolean
#: result is not a sliver and small enough that the whole sanity pass is milliseconds.
_PLATE_MM = (40.0, 40.0, 10.0)
#: `known_good`: a 5 mm air gap between the two plates.
_FIXTURE_GAP_MM = 5.0
#: `known_bad`: 3 mm of shared material, the number the audit report asks for.
_FIXTURE_OVERLAP_MM = 3.0


def _interface_between(spec: DesignSpec, first: str, second: str) -> str | None:
    """The id of the interface joining `first` and `second`, or `None`.

    Only used to label a certificate: a pair without an interface is the pair this
    verifier exists for, and a report has to be able to tell the two apart.
    """
    pair = {first, second}
    for interface in spec.interfaces:
        if {interface.a, interface.b} == pair:
            return interface.id
    return None


def _boxes_overlap(a: trimesh.Trimesh, b: trimesh.Trimesh) -> bool:
    """Could `a` and `b` share any volume? An exact, cheap pre-filter.

    Boxes that do not overlap on at least one axis bound two disjoint point sets, so
    the shared volume is exactly zero — the boolean is provably unnecessary, not just
    unlikely to find something. The comparison is strict: boxes that only *touch* on a
    plane share a face, not a volume, and the seated poses of the shipped fixtures sit
    exactly on that boundary.
    """
    low = np.maximum(a.bounds[0], b.bounds[0])
    high = np.minimum(a.bounds[1], b.bounds[1])
    return bool(np.all(high > low))


def _solid(view: DesignView, part_id: str) -> trimesh.Trimesh:
    """The part's world-frame mesh, with a refusal that names the part.

    `DesignView.mesh` already raises for a missing path or file; this adds the one
    case it cannot see from a file name — a mesh with no triangles has no volume to
    compare, and reporting it as "not overlapping" would be a silent pass.
    """
    mesh = view.mesh(part_id)
    if mesh is None or len(mesh.faces) == 0:
        raise ValueError(
            f"part {part_id!r} has an empty mesh; a shared volume against it cannot "
            "be measured, so this design's interference is unknown rather than clear"
        )
    return mesh


class GlobalInterference:
    """Every pair of parts, one boolean intersect each, in the seated pose."""

    spec = VerifierSpec(
        id="global_interference",
        codes=CODES,
        cost_class="medium",
        fixtures={
            "known_good": ["fixture-split-plates: two 40x40x10 plates 5 mm apart"],
            "known_bad": [
                "fixture-overlap-plates: the same pair overlapping 3 mm, 4800 mm3"
            ],
        },
    )

    def __init__(self, *, min_volume_mm3: float = INTERFERENCE_MIN_VOLUME_MM3) -> None:
        # Exposed for the same reason it is on `InterfaceInterference`: a different
        # materialiser may have a different noise floor, and a caller that lowers the
        # floor owes its own `known_bad` evidence.
        self.min_volume_mm3 = float(min_volume_mm3)

    def fixtures(self) -> tuple[list[DesignView], list[DesignView]]:
        """One disjoint pair and one overlapping pair, differing only in the gap."""
        return (
            [_plate_view("fixture-plates-apart", _FIXTURE_GAP_MM)],
            [_plate_view("fixture-plates-overlap", -_FIXTURE_OVERLAP_MM)],
        )

    def check(self, view: DesignView) -> list[FailureCertificate]:
        """One certificate per pair whose shared volume exceeds the noise floor.

        Pairs are visited in sorted id order so the certificate list is stable: a
        report that reorders its evidence between two runs of the same design is a
        diff nobody can read.
        """
        part_ids = sorted(view.part_ids())
        certificates: list[FailureCertificate] = []
        for first, second in itertools.combinations(part_ids, 2):
            a = _solid(view, first)
            b = _solid(view, second)
            if not _boxes_overlap(a, b):
                continue
            volume = self._shared_volume(first, a, second, b)
            if volume <= self.min_volume_mm3:
                continue
            certificates.append(self._certificate(view, first, second, volume))
        return certificates

    # -- internals ---------------------------------------------------------
    def _shared_volume(
        self,
        first: str,
        a: trimesh.Trimesh,
        second: str,
        b: trimesh.Trimesh,
    ) -> float:
        """The volume both parts occupy, or a refusal that names the solid to blame.

        The watertightness check is repeated here (the boolean would raise on its own)
        only to put the *part id* in the message: `intersection_solid` labels its two
        inputs "body" and "insert", which is the two-part vocabulary and would send a
        reader looking for the wrong thing in an N-part spec.
        """
        for mesh, part_id in ((a, first), (b, second)):
            if not mesh.is_watertight:
                raise ValueError(
                    f"part {part_id!r} is not watertight ({len(mesh.faces)} faces), so "
                    "the volume it shares with another part is undefined; this design's "
                    "interference is unknown rather than clear"
                )
        return intersection_volume_mm3(a, b)

    def _certificate(
        self,
        view: DesignView,
        first: str,
        second: str,
        volume: float,
    ) -> FailureCertificate:
        locus: dict = {
            "spec": view.spec.spec_id,
            "revision": view.spec.revision,
            "parts": [first, second],
        }
        interface_id = _interface_between(view.spec, first, second)
        if interface_id is not None:
            # Present only for the pairs a joint already covers. A report that wants
            # one finding per defect dedupes on this key; a pair without it is a
            # collision no interface-level check could have seen.
            locus["interface"] = interface_id
        return FailureCertificate(
            code=CODE_GLOBAL_INTERFERENCE,
            family=_FAMILY,
            locus=locus,
            magnitude={
                "volume_mm3": volume,
                "min_volume_mm3": self.min_volume_mm3,
            },
            observed_by=self.spec.id,
            evidence_ref="all_pairs_boolean_intersection@manifold",
        )


# --------------------------------------------------------------------------- #
# Fixtures — built from trimesh primitives, one change per fixture
# --------------------------------------------------------------------------- #
#: A spec needs *some* interface: `DesignSpec` refuses a multi-part spec that nothing
#: connects, because a part nothing holds is a part that cannot ship. This one is
#: never read by the verifier — the pairs it judges are precisely the ones no
#: interface joins — and its numbers are a nominal 6 mm peg, so a reader is not
#: invited to believe they describe the plates.
_FIXTURE_INTERFACE = (
    InterfaceSpec(
        id="iface_plates",
        kind="peg_hole",
        a="plate_lower",
        b="plate_upper",
        params={
            "d_peg_mm": 5.8,
            "d_hole_mm": 6.0,
            "engage_mm": 4.0,
            "approach_mm": 6.0,
        },
    ),
)


def _plate_view(spec_id: str, gap_mm: float) -> DesignView:
    """Two identical plates with `gap_mm` of air between them.

    `gap_mm` is the **distance between the faces**, which is the number the two
    fixtures are meant to differ by: positive is air, negative is shared material. It
    is converted to a centre offset through the plate thickness here, because the first
    version of this helper took the centre offset and called it a gap, so `known_good`
    overlapped by 5 mm — and the fixture gate caught it on its first run, which is the
    reason a verifier ships fixtures at all.
    """
    offset_z_mm = _PLATE_MM[2] + gap_mm
    relation = "air between" if gap_mm >= 0.0 else "overlap between"
    lower = trimesh.creation.box(extents=_PLATE_MM)
    upper = trimesh.creation.box(extents=_PLATE_MM)
    upper.apply_translation((0.0, 0.0, offset_z_mm))
    spec = DesignSpec(
        spec_id=spec_id,
        requirement=(
            f"two {_PLATE_MM[0]:g} x {_PLATE_MM[1]:g} x {_PLATE_MM[2]:g} mm plates, "
            f"{abs(gap_mm):g} mm of {relation} them"
        ),
        parts=(
            PartSpec(
                id="plate_lower",
                role="body",
                size_mm=_PLATE_MM,
                pose_mm=(0.0, 0.0, 0.0),
            ),
            PartSpec(
                id="plate_upper",
                role="part",
                size_mm=_PLATE_MM,
                pose_mm=(0.0, 0.0, offset_z_mm),
            ),
        ),
        interfaces=_FIXTURE_INTERFACE,
        constraints=FitConstraints(),
    )
    return DesignView(
        spec=spec,
        root=None,
        meshes={"plate_lower": lower, "plate_upper": upper},
    )


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #
def register(registry: DesignVerifierRegistry) -> None:
    """Add this verifier to `registry`; declaring the code is the caller's job.

    `DesignVerifierRegistry.register` refuses an id that is not in
    `pah.design.api.DESIGN_CODES`, and that guard is the reason two verifiers cannot
    quietly own one code. So the wiring is two steps and the order matters: declare
    `CODE_GLOBAL_INTERFERENCE` in `DESIGN_CODES` first, then call this from
    `default_registry()`.

    This module deliberately does not import `design_cascade`: the cascade imports the
    verifiers, so a verifier importing the cascade back would be a cycle — and every
    later verifier would have to be written around it.
    """
    registry.register(GlobalInterference())
