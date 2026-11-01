"""Looking at an N-part assembly one joint at a time.

Why this exists
---------------
Everything the harness already knows how to judge — clearance in band, interference
at the seat, a clear insertion path, wall thickness, overhang, a physics settle — is
a statement about **one pair of solids moving along one axis**. Six verifiers encode
that, and they encode it well enough that rewriting them for N parts would mean
re-deriving six pieces of physics with the same chance of getting each one subtly
wrong.

So this module does the cheap thing instead: for each interface, build the two-part
**shadow** — a `DesignSpec` holding just those two parts in the roles the two-part
contract expects (`body`/`insert`), and a `DesignView` holding just their two meshes
and that interface's own frame. A verifier cannot tell the difference, and nothing
in the verifier layer has to change.

What this deliberately does not do
----------------------------------
It does not merge the pairs' evidence into one verdict. A pair can be perfect and
its neighbour broken; the report has to keep them apart, which is why the return
type is a mapping keyed by interface id rather than a single certificate list.
"""

from __future__ import annotations

from dataclasses import dataclass

from pah.design.api import DesignView, MaterializedDesign
from pah.design.spec import DesignSpec, InterfaceSpec, Vec3

__all__ = ["PairView", "pair_spec", "pair_view"]

#: The frame a pair falls back to when nothing recorded one: a plain axial peg,
#: which is what the two-part default is too. Named so a missing fact is visible in
#: a stack trace instead of silently becoming +Z.
_DEFAULT_AXIS: Vec3 = (0.0, 0.0, 1.0)
_DEFAULT_APPROACH_MM = 6.0


@dataclass(frozen=True)
class PairView:
    """One interface as the two-part contract sees it."""

    interface: InterfaceSpec
    spec: DesignSpec
    view: DesignView


def pair_spec(spec: DesignSpec, interface_id: str) -> DesignSpec:
    """A two-part `DesignSpec` holding just `interface_id`'s pair.

    The two parts keep their envelopes, poses and `source` hashes, so the shadow is
    still a faithful description of *those* solids; only the roles are rewritten,
    because `body`/`insert` is how the two-part contract spells "host" and "mating
    part" and the verifiers read it.
    """
    interface = spec.interface_by_id(interface_id)
    host = spec.part(interface.a)
    guest = spec.part(interface.b)
    return DesignSpec(
        spec_id=f"{spec.spec_id}::{interface.id}",
        requirement=spec.requirement,
        parts=(
            host.model_copy(update={"role": "body"}),
            guest.model_copy(update={"role": "insert"}),
        ),
        interfaces=(interface,),
        constraints=spec.constraints,
        provenance=dict(spec.provenance),
    )


def pair_view(design: MaterializedDesign, interface_id: str) -> PairView:
    """The `DesignView` one interface's verifiers should judge.

    The frame comes from `design.interfaces` when the materialiser recorded one, and
    falls back to the design-level facts otherwise — never to a hard-coded axis for
    a kind. Dovetail is the reason that rule exists: it slides along +X, and a +Z
    sweep calls a perfect slide-joint blocked.
    """
    interface = design.spec.interface_by_id(interface_id)
    facts = (design.interfaces or {}).get(interface_id) or {}
    axis = tuple(float(value) for value in facts.get("mate_axis", design.mate_axis))
    approach = float(facts.get("approach_mm", design.approach_mm))
    shadow = pair_spec(design.spec, interface_id)
    view = DesignView(
        spec=shadow,
        root=design.root,
        meshes={
            part_id: design.meshes[part_id]
            for part_id in (interface.a, interface.b)
            if part_id in design.meshes
        },
        paths={
            part_id: design.paths[part_id]
            for part_id in (interface.a, interface.b)
            if part_id in design.paths
        },
        mate_axis=axis,
        approach_mm=approach,
    )
    return PairView(interface=interface, spec=shadow, view=view)
