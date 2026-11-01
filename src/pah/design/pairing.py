"""Deciding which decomposed pieces touch, and what connector goes between them.

Why this is a separate step from the decomposition
--------------------------------------------------
A decomposition answers "where does this object split". It does not answer "how do
the pieces hold together": X-Part's own paper says its parts are only *touching*,
not fastened, and the local splitter is even more honest — its pieces are cut from
one solid, so their faces are exact complements and still fall apart.

So the pipeline is: `Decomposer` -> `propose_interfaces` -> `DesignSpec` ->
materialise. This module is the middle arrow, and it is the only place that decides
*where* a connector goes. Everything downstream reads the spec and never looks for
a mating face again.

Two rules that keep it honest
-----------------------------
1. **Adjacency is a bounding-box fact, and that is a design decision.** Two pieces
   whose boxes touch within a tolerance are treated as a joint. That is wrong for a
   C-shaped piece wrapped around another, and right for everything a slab
   decomposition produces. The alternative — exact contact-area analysis on 100k-face
   meshes — costs minutes per pair to answer a question a repair loop can revisit;
   so the cheap test ships, its limitation is written down, and `contact_of` returns
   the numbers it used rather than a bare boolean.
2. **A pair that cannot carry a connector is skipped by name, not silently.** Too
   thin to engage, no overlap area, clearance that would break through the wall:
   each has a reason string. A pipeline that quietly connected 3 of 4 pairs and
   reported "assembled" is the failure mode this whole harness exists to prevent.
"""

from __future__ import annotations

from dataclasses import dataclass

from pah.design.spec import (
    DesignSpec,
    FitConstraints,
    InterfaceSpec,
    PartSpec,
    Vec3,
)

__all__ = [
    "PairingReport",
    "PartContact",
    "contact_of",
    "face_frame",
    "propose_interfaces",
    "spec_from_parts",
]

#: Axis names, in the order a deterministic tie-break resolves them.
_AXES = ("x", "y", "z")


@dataclass(frozen=True)
class PartContact:
    """One candidate joint: which parts, along which axis, and the numbers behind it.

    `axis` points from `a` to `b`. `point_mm` is the centre of the shared face, in
    world coordinates. `gap_mm` is negative when the boxes overlap (a decomposition
    that returns interpenetrating pieces), zero when they meet, positive when there
    is a real gap.
    """

    a: str
    b: str
    axis: Vec3
    point_mm: Vec3
    gap_mm: float
    overlap_mm: tuple[float, float]
    #: Thickness of each part along the interface axis — the depth budget.
    thickness_mm: tuple[float, float]

    @property
    def thinnest_mm(self) -> float:
        return min(self.thickness_mm)


@dataclass(frozen=True)
class PairingReport:
    """What got connected and what did not, with a reason per skipped pair."""

    interfaces: tuple[InterfaceSpec, ...]
    skipped: tuple[tuple[str, str, str], ...] = ()  # (a, b, reason)

    def reasons(self) -> str:
        return "; ".join(f"{a}/{b}: {why}" for a, b, why in self.skipped)


def _centre(part: PartSpec) -> tuple[float, float, float]:
    return tuple(float(value) for value in part.pose_mm)


def _extent(part: PartSpec, axis: int) -> float:
    return float(part.size_mm[axis])


def contact_of(
    first: PartSpec, second: PartSpec, *, tolerance_mm: float = 0.5
) -> PartContact | None:
    """The single axis along which two parts are side by side, or `None`.

    "Side by side" means: their boxes overlap on two axes, and on the third they are
    within `tolerance_mm` of touching. Exactly one axis must qualify. Two would mean
    the pieces share a corner — a joint there has no face to machine, and picking one
    of the two axes at random is how a design gets a peg through a corner.
    """
    gaps: list[tuple[float, int]] = []
    for index in range(3):
        low = max(
            _centre(first)[index] - _extent(first, index) / 2.0,
            _centre(second)[index] - _extent(second, index) / 2.0,
        )
        high = min(
            _centre(first)[index] + _extent(first, index) / 2.0,
            _centre(second)[index] + _extent(second, index) / 2.0,
        )
        gaps.append((low - high, index))  # >0 = separated, <0 = overlapping

    # The separating axis is the one with a non-negative gap; the other two must
    # overlap (negative gap) or the boxes are diagonal neighbours, not side by side.
    candidates = [
        (gap, index) for gap, index in gaps if gap >= -1e-9
    ]
    touching = [(gap, index) for gap, index in candidates if gap <= tolerance_mm]
    if len(touching) != 1:
        return None
    gap, axis_index = touching[0]
    for other_gap, other_index in gaps:
        if other_index != axis_index and other_gap > -1e-6:
            return None  # diagonal: they meet at an edge or corner, not a face

    direction = 1.0 if _centre(second)[axis_index] >= _centre(first)[axis_index] else -1.0
    axis = tuple(1.0 if index == axis_index and direction > 0 else
                 (-1.0 if index == axis_index else 0.0) for index in range(3))

    # Centre of the shared face: the plane halfway across the gap, centred on the
    # mutual overlap of the other two axes.
    point = [0.0, 0.0, 0.0]
    for index in range(3):
        if index == axis_index:
            face_first = _centre(first)[index] + direction * _extent(first, index) / 2.0
            face_second = _centre(second)[index] - direction * _extent(second, index) / 2.0
            point[index] = (face_first + face_second) / 2.0
        else:
            low = max(
                _centre(first)[index] - _extent(first, index) / 2.0,
                _centre(second)[index] - _extent(second, index) / 2.0,
            )
            high = min(
                _centre(first)[index] + _extent(first, index) / 2.0,
                _centre(second)[index] + _extent(second, index) / 2.0,
            )
            point[index] = (low + high) / 2.0

    others = [index for index in range(3) if index != axis_index]
    low = max(
        _centre(first)[others[0]] - _extent(first, others[0]) / 2.0,
        _centre(second)[others[0]] - _extent(second, others[0]) / 2.0,
    )
    high = min(
        _centre(first)[others[0]] + _extent(first, others[0]) / 2.0,
        _centre(second)[others[0]] + _extent(second, others[0]) / 2.0,
    )
    overlap_a = high - low
    low = max(
        _centre(first)[others[1]] - _extent(first, others[1]) / 2.0,
        _centre(second)[others[1]] - _extent(second, others[1]) / 2.0,
    )
    high = min(
        _centre(first)[others[1]] + _extent(first, others[1]) / 2.0,
        _centre(second)[others[1]] + _extent(second, others[1]) / 2.0,
    )
    overlap_b = high - low

    return PartContact(
        a=first.id,
        b=second.id,
        axis=axis,  # type: ignore[arg-type]
        point_mm=tuple(point),  # type: ignore[arg-type]
        gap_mm=float(gap),
        overlap_mm=(float(overlap_a), float(overlap_b)),
        thickness_mm=(float(_extent(first, axis_index)), float(_extent(second, axis_index))),
    )


def face_frame(contact: PartContact) -> tuple[Vec3, Vec3]:
    """`(axis, point)` — the frame `pah.design.machining` takes."""
    return contact.axis, contact.point_mm


#: A peg narrower than this cannot survive an FDM nozzle (0.4 mm line, three
#: perimeters), and a socket narrower than it cannot be printed at all.
#:
#: Calibrated against the shipped case rather than guessed: a 12 mm plate cut three
#: ways gives 4 mm slabs, and 60% of that is 2.4 mm — under the first value tried
#: (3.0 mm), which silently refused to connect a perfectly ordinary three-piece
#: print. The guard that matters is the wall budget below, not this constant.
_MIN_PEG_MM = 2.0
#: Wider than this and the peg is a structural feature nobody asked for; it also
#: stops a large flat face from getting a peg the size of the part.
_MAX_PEG_MM = 20.0
#: A peg may use at most this fraction of the face it sits on, and of the thinnest
#: part's thickness. Both are wall-thickness guards, not fit numbers.
_FACE_FRACTION = 0.5
_THICKNESS_FRACTION = 0.6


def _peg_for(contact: PartContact, clearance_mm: float) -> tuple[float, float]:
    """`(d_peg_mm, engage_mm)` for one contact, or raise with the reason it has none.

    The peg has to fit three budgets at once: the smaller side of the shared face,
    the thinner part's thickness, and the absolute size band. Taking the minimum of
    the three, rather than a constant, is what lets one rule serve a 60 mm plate and
    a 12 mm cube.
    """
    face_budget = _FACE_FRACTION * min(contact.overlap_mm)
    depth_budget = _THICKNESS_FRACTION * contact.thinnest_mm
    diameter = min(face_budget, depth_budget, _MAX_PEG_MM)
    if diameter < _MIN_PEG_MM:
        raise ValueError(
            f"face {min(contact.overlap_mm):.1f} mm and thickness "
            f"{contact.thinnest_mm:.1f} mm leave a peg of only {diameter:.1f} mm"
        )
    # Engage as deep as the thinner part safely allows, but never past 8 mm: past
    # that a blind socket is a hole a hobby printer fills with its own sag.
    engage = min(depth_budget, 8.0)
    if engage < 2.0 * clearance_mm + 1.0:
        raise ValueError(f"thickness {contact.thinnest_mm:.1f} mm leaves no socket depth")
    return float(diameter), float(engage)


def propose_interfaces(
    parts: tuple[PartSpec, ...],
    *,
    tolerance_mm: float = 0.5,
    clearance_mm: float = 0.20,
) -> PairingReport:
    """One `peg_hole` per adjacent pair, oriented `a` = host (socket), `b` = insert.

    The host is the part on the negative side of the axis, so "a grows a socket and b
    grows a peg" is a statement about the axis rather than about which piece the
    decomposition happened to name first. A chain therefore lands the way a person
    would draw it: every middle piece carries a peg on one face and a socket on the
    other.
    """
    interfaces: list[InterfaceSpec] = []
    skipped: list[tuple[str, str, str]] = []
    ordered = sorted(parts, key=lambda part: part.id)

    for index, first in enumerate(ordered):
        for second in ordered[index + 1 :]:
            contact = contact_of(first, second, tolerance_mm=tolerance_mm)
            if contact is None:
                continue
            try:
                diameter, engage = _peg_for(contact, clearance_mm)
            except ValueError as error:
                skipped.append((first.id, second.id, str(error)))
                continue
            # `contact.axis` points from `first` to `second`, so `first` is the part
            # on the negative side: the host, whose face gets the socket.
            host, guest = first, second
            interfaces.append(
                InterfaceSpec(
                    id=f"iface_{host.id}_{guest.id}",
                    kind="peg_hole",
                    a=host.id,
                    b=guest.id,
                    params={
                        "d_peg_mm": round(diameter, 3),
                        "d_hole_mm": round(diameter + 2.0 * clearance_mm, 3),
                        "engage_mm": round(engage, 3),
                        "approach_mm": 6.0,
                    },
                )
            )
    return PairingReport(interfaces=tuple(interfaces), skipped=tuple(skipped))


def spec_from_parts(
    *,
    spec_id: str,
    parts: tuple[PartSpec, ...],
    interfaces: tuple[InterfaceSpec, ...],
    requirement: str = "",
    constraints: FitConstraints | None = None,
    provenance: dict | None = None,
) -> DesignSpec:
    """The `DesignSpec` a decomposition plus its pairing describes."""
    return DesignSpec(
        spec_id=spec_id,
        requirement=requirement,
        parts=parts,
        interfaces=interfaces,
        constraints=constraints or FitConstraints(),
        provenance=provenance or {},
    )
