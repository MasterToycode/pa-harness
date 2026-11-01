"""`interface_interference`: do the two solids occupy the same space when seated?

The measurement is one boolean intersection of the body and the insert, both taken
in their assembled pose. `ConnectorGeometry.meshes` are world-frame solids
(`design/api.py`), so "assembled pose" is simply "the meshes as given": nothing
here re-applies `PartSpec.pose_mm`. Transforming them again is the classic double
transform bug — a part that sits 20 mm off would look clean.

The threshold, and why it is 1e-3 mm3 and not the contract's 1e-6
---------------------------------------------------------------
`docs/contracts/v2-interfaces.md` §3 states the criterion as "boolean intersection
volume > 1e-6 mm3". Measured on the design plane's own fixtures
(`_design_fixtures`), the intersection of a correctly-fitted pair is *exactly*
empty — before and after an STL round trip — because a 0.05 mm clearance is two
orders of magnitude above the faceting sag of a 64-facet cylinder (0.0053 mm).
The 1e-3 mm3 floor is not there for this perfect case, it is there for the ones
that are not perfect:

* a coarser or decimated mesh, a boolean that leaves a sliver triangle along a
  coincident face, or a solver that returns a 1e-9 mm3 lens where two surfaces
  were meant to touch — all of these produce a *positive* volume that is not an
  interference, and `1e-6` would call every one of them a failure;
* the smallest interference this domain can express is still four orders above
  it: a pin 0.1 mm over-size over an 8 mm engagement overlaps by
  pi * (5.1^2 - 5.0^2) * 8 = 25.4 mm3, which is what `oversized_fixture()` measures
  (25.343 mm3 against a 25.35 mm3 analytic value).

So 1e-3 mm3 trades a blind spot smaller than any manufacturable interference for
immunity to numeric noise. The constant is exposed on the constructor because a
different materialiser may have a different noise floor; the default is the value
the fixtures were tuned against, and a caller that lowers it must bring its own
known_bad evidence.

`max_depth_mm` is an *upper bound*, not a penetration depth
-----------------------------------------------------------
The field is the intersection solid's extent along the view's `mate_axis` — the
bounding box, as the contract asks for a "cheap" estimate. For a peg too fat for
its hole it is the engaged length (8 mm here), i.e. how deep the overlap reaches
along the insertion direction; it is not the radial depth of the overlap, and for
a sliver crossing the axis at an angle it can overstate both. A policy sizing a
`resize_insert` move should use `volume_mm3` and the fit verifier's clearance,
not this number.

The axis comes from the view, never from a constant
---------------------------------------------------
`DesignView.mate_axis` (contract §2) is the insertion direction of the *built*
geometry: +Z for a peg and a snap hook, +X for a dovetail slide. Measuring the
depth along a hard-coded +Z would make the number meaningless for every slide fit,
which is why this module carries no axis of its own.
"""

from __future__ import annotations

import numpy as np
import trimesh
from trimesh.boolean import intersection

from pah.core.model import FailureCertificate
from pah.design.api import CODE_GEOMETRIC_INTERFERENCE, DesignView

from .base import VerifierSpec

__all__ = [
    "CODES",
    "INTERFERENCE_MIN_VOLUME_MM3",
    "InterfaceInterference",
    "extent_along",
    "intersection_solid",
    "intersection_volume_mm3",
    "unit_axis",
]

CODES: tuple[str, ...] = (CODE_GEOMETRIC_INTERFERENCE,)

#: Anything below this volume is noise, not a collision. See the module docstring.
INTERFERENCE_MIN_VOLUME_MM3 = 1e-3

#: `manifold3d`, the engine the project pins. Explicit so a trimesh default
#: change cannot silently swap the measurement for a different one.
_ENGINE = "manifold"

#: `assembly`: two parts of a joint occupying one space is a property of the
#: joint, not of either solid, and it is certainly not a manufacturing limit.
_FAMILY = "assembly"


def intersection_solid(a: trimesh.Trimesh, b: trimesh.Trimesh) -> trimesh.Trimesh | None:
    """The solid both meshes share, or `None` when they are disjoint.

    Both inputs must be watertight: a boolean on an open shell has no defined
    volume, and reporting 0.0 for it would be a silent pass. A non-watertight
    input therefore raises, and the cascade files the round's evidence as
    `environment` — unknown, never healthy.

    An empty result is normal here (a correctly fitted pair does not intersect),
    and trimesh returns an empty `Trimesh` rather than `None` for it, so the
    caller must treat `is_empty` as "no overlap" too.
    """
    for mesh, role in ((a, "body"), (b, "insert")):
        if mesh is None or len(mesh.faces) == 0:
            raise ValueError(f"the {role} mesh is empty; nothing to intersect")
        if not mesh.is_watertight:
            raise ValueError(
                f"the {role} mesh is not watertight ({len(mesh.faces)} faces); a "
                "boolean intersection on an open shell has no defined volume"
            )
    result = intersection([a, b], engine=_ENGINE)
    if result is None:
        return None
    if isinstance(result, (list, tuple)):  # one mesh per input pair, by contract
        raise TypeError(
            "trimesh returned %d solids for one pair; the engine's contract changed"
            % len(result)
        )
    return result


def intersection_volume_mm3(a: trimesh.Trimesh, b: trimesh.Trimesh) -> float:
    """The volume both solids share, in mm3; 0.0 when they are disjoint."""
    return _mesh_volume(intersection_solid(a, b))


def _mesh_volume(mesh: trimesh.Trimesh | None) -> float:
    """`mesh.volume`, with the zero-volume degeneracies mapped to 0.0."""
    if mesh is None or len(mesh.faces) == 0 or mesh.is_empty:
        return 0.0
    # A zero-volume result makes trimesh's mass properties divide 0/0 and warn;
    # the answer is still 0.0, so the warning is noise about a handled case.
    with np.errstate(invalid="ignore", divide="ignore"):
        return abs(float(mesh.volume))


def unit_axis(axis: tuple[float, float, float]) -> np.ndarray:
    """`axis` normalised, with a zero vector refused.

    A `DesignView` documents `mate_axis` as a unit vector, but the depth estimate
    and the path sweep both scale with it and a plugin kind is not checked
    anywhere: normalising here means a non-unit axis is harmless, and a zero one
    raises (unknown) instead of silently reporting a zero depth.
    """
    vector = np.asarray(axis, dtype=float)
    length = float(np.linalg.norm(vector))
    if length == 0.0:
        raise ValueError(f"mate_axis {tuple(axis)!r} has no direction")
    return vector / length


def extent_along(mesh: trimesh.Trimesh | None, axis: tuple[float, float, float]) -> float:
    """The mesh's bounding-box thickness measured along `axis`, in mm."""
    if mesh is None or mesh.is_empty:
        return 0.0
    low, high = np.asarray(mesh.bounds, dtype=float)
    return float((high - low) @ np.abs(unit_axis(axis)))


class InterfaceInterference:
    """The part of the two solids that cannot both be material at the seat."""

    spec = VerifierSpec(
        id="interface_interference",
        codes=CODES,
        cost_class="medium",
        fixtures={
            "known_good": ["fixture-standard: 0.05 mm radial clearance, disjoint"],
            "known_bad": ["fixture-oversized: 0.10 mm interference, 25.3 mm3"],
        },
    )

    def __init__(self, *, min_volume_mm3: float = INTERFERENCE_MIN_VOLUME_MM3) -> None:
        self.min_volume_mm3 = float(min_volume_mm3)

    def fixtures(self) -> tuple[list[DesignView], list[DesignView]]:
        """The pair the §4.8 gate runs: one disjoint seat, one overlapping seat."""
        from ._design_fixtures import oversized_fixture, standard_fixture

        return ([standard_fixture()], [oversized_fixture()])

    def check(self, view: DesignView) -> list[FailureCertificate]:
        """One certificate when the shared volume exceeds the noise floor."""
        overlap = intersection_solid(view.body, view.insert)
        volume = _mesh_volume(overlap)
        if volume <= self.min_volume_mm3:
            return []

        interface = view.spec.interface
        return [
            FailureCertificate(
                code=CODE_GEOMETRIC_INTERFERENCE,
                family=_FAMILY,
                locus={
                    "spec": view.spec.spec_id,
                    "revision": view.spec.revision,
                    "interface": interface.id,
                    "parts": [interface.a, interface.b],
                    "kind": interface.kind,
                },
                magnitude={
                    "volume_mm3": volume,
                    "max_depth_mm": extent_along(overlap, view.mate_axis),
                    "min_volume_mm3": self.min_volume_mm3,
                    "mate_axis": list(unit_axis(view.mate_axis)),
                },
                observed_by=self.spec.id,
                evidence_ref=f"boolean_intersection@{_ENGINE}",
            )
        ]
