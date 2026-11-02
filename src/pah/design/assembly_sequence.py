"""Searching for an assembly order that a straight-line disassembly would allow.

Why an order at all
-------------------
Every verifier in the design plane answers a question about **one moment**: is this
joint clear when it is seated, does this clearance sit in the band, is this wall thick
enough. Assembly is a process, not a moment, and the process can be blocked even when
every moment is fine: a part surrounded on all sides, or a chain whose middle piece can
only be slid in from a direction another piece already occupies.

So this module searches for a **disassembly sequence**: repeatedly pick a part that can
be pulled out of the assembly without passing through any part still in place, until
nothing is left. Reversing that sequence gives an assembly order, and every reversed
step is a legal insertion — the parts are rigid and the swept volumes are the same in
both directions.

What `found=True` proves, and what "not found" does not
------------------------------------------------------
* `found=True` is a **constructive** result. Every step of `order` was swept against
  the solids that remain in that step, so the reverse order can be built by a machine
  with no reorientation and no compliant parts. It is a proof about *these* triangles.
* `found=False` is **not** a proof of impossibility, and no caller may present it as
  "cannot be assembled". The search is a heuristic over a deliberately small move set:

  1. only **straight-line pulls** are tried, in both directions of an interface normal
     (`pah.design.pairing.contact_of` supplies the axis). Rotations, tilting a part
     out of a socket, and multi-segment paths are not modelled at all, so a part that
     can only come out at an angle is reported as not found;
  2. it is **depth-first with backtracking under a move budget** (`max_attempts`). A
     cut-short search proves nothing except that the budget was too small, and the
     `reason` string says so instead of letting a reader infer impossibility;
  3. the swept volume is **sampled**, and the step is capped at `_MAX_STEP_MM`
     (0.5 mm). A wall thinner than the step could in principle be tunnelled through
     without any sample landing inside it — four times thinner than the thinnest wall
     the printability tier will pass, so it is stated as a boundary rather than
     engineered away. The step only coarsens for very long pulls (`_MAX_SAMPLES`).

  A caller that quotes this module in a report should quote `reason`, which names
  which of the three applies; `found` alone loses the distinction.

The seated pose is sampled, and an interfering assembly has no sequence
----------------------------------------------------------------------
The sweep starts at distance 0, so parts that already share volume cannot be
"removed": the search's first sample already collides. That is deliberate. A design in
which two solids occupy the same space has no physical assembly to describe, and
`global_interference` is the verifier that reports it; quietly finding an order anyway
would give the repair loop a reason to believe an impossible design is fine.

Pure geometry
-------------
No LLM, no network, no MuJoCo, no file I/O. The caller already has the triangles (a
materialiser built them), and this is a question about those triangles: can they be
moved into place. Whether the joint then *holds* under gravity is the physics settle's
question, a different model on a different tier.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import trimesh

from pah.design.pairing import contact_of
from pah.design.spec import DesignSpec
from pah.tools.verifiers.interface_interference import (
    INTERFERENCE_MIN_VOLUME_MM3,
    intersection_volume_mm3,
    unit_axis,
)

__all__ = ["SequenceResult", "find_sequence"]

#: Never sample the pull more coarsely than this. See the module docstring: it is the
#: wall thickness below which a solid could be tunnelled through without a collision
#: being sampled.
_MAX_STEP_MM = 0.5

#: Upper bound on the samples per pull. A 40 mm pull at the step above would be 80
#: booleans per pair, and a 19-part job would spend its whole budget on one move; the
#: cap coarsens the step for long pulls, where the distances involved are large
#: compared with the geometry being tunnelled through.
_MAX_SAMPLES = 32

#: How far past the last other solid a part is swept before the sweep is declared
#: safe. 1 mm of margin keeps the final sample off a shared plane, where a boolean may
#: return a sliver that is not a collision.
_EXIT_MARGIN_MM = 1.0


@dataclass(frozen=True)
class SequenceResult:
    """What the search concluded, and in how many moves.

    `order` is the **disassembly** order — the order the parts were taken out — because
    that is the order the search proved; `assembly_order` is the reverse and is what a
    report should print to a person. Keeping only one of them in the dataclass and
    deriving the other would leave every caller to decide which is which, which is how
    a printer eventually gets handed the sequence backwards.
    """

    found: bool
    order: tuple[str, ...]
    #: The part the search could not free, when it failed; `None` when it succeeded.
    blocked_at: str | None
    attempts: int
    #: A sentence for a report, not a log line: it says what was searched and, on
    #: failure, that the search failing is not a proof.
    reason: str
    #: The smallest set of parts the search could not get through, when it failed.
    #: Narrower than the whole assembly, which is what makes it worth printing.
    remaining: tuple[str, ...] = ()

    @property
    def assembly_order(self) -> tuple[str, ...]:
        """The order a person builds it in: the disassembly sequence, reversed."""
        return tuple(reversed(self.order))


def find_sequence(
    spec: DesignSpec,
    meshes: Mapping[str, trimesh.Trimesh],
    *,
    max_attempts: int = 64,
) -> SequenceResult:
    """Search for a straight-line disassembly order of `spec` built as `meshes`.

    `meshes` are world-frame solids, seated pose, one per part of `spec` — the same
    objects `pah.design.materialize` returns in `MaterializedDesign.meshes`. Extra or
    missing ids raise: a sweep over the wrong solid would answer a question nobody
    asked, and a silently ignored extra would make the answer look complete.

    `max_attempts` caps how many candidate moves (one part, one direction) are tried.
    It is a budget, not a resolution: exhausting it produces `found=False` with a
    `reason` that says the search was cut short.
    """
    part_ids = tuple(part.id for part in spec.parts)
    missing = sorted(set(part_ids) - set(meshes))
    if missing:
        raise KeyError(
            f"no mesh for part(s) {missing!r}; a disassembly claim has to be about the "
            "solids it names"
        )
    extra = sorted(set(meshes) - set(part_ids))
    if extra:
        raise KeyError(
            f"meshes for {extra!r}, which spec {spec.spec_id!r} does not name; a solid "
            "the design does not describe cannot be swept as if it had been designed"
        )
    solids = {part_id: _solid(meshes[part_id], part_id) for part_id in part_ids}
    return _Search(
        solids=solids,
        directions=_pull_directions(spec),
        max_attempts=max_attempts,
        min_volume_mm3=INTERFERENCE_MIN_VOLUME_MM3,
    ).run()


# --------------------------------------------------------------------------- #
# Solids and directions
# --------------------------------------------------------------------------- #
def _solid(mesh: trimesh.Trimesh, part_id: str) -> trimesh.Trimesh:
    """The part as one closed solid, or a refusal naming it.

    Fail closed: every measurement below is a boolean volume, and a boolean on an open
    shell has no defined volume. Treating the parts as clear because the kernel
    returned nothing would turn an unreadable mesh into an assembly claim.
    """
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError(
            f"part {part_id!r} has no triangles; a sweep against an empty mesh is not "
            "a measurement"
        )
    if not mesh.is_watertight:
        raise ValueError(
            f"part {part_id!r} is not watertight ({len(mesh.faces)} faces), so the "
            "volume it sweeps through another part is undefined"
        )
    return mesh


def _direction_key(vector: np.ndarray) -> tuple[float, float, float]:
    """`vector` as a hashable, deduplicated direction (rounded to a micron)."""
    x, y, z = (round(float(component), 6) for component in np.asarray(vector)[:3])
    return x, y, z


def _pull_directions(
    spec: DesignSpec,
) -> dict[str, tuple[tuple[float, float, float], ...]]:
    """Both signs of every interface normal each part takes part in.

    The axis comes from `contact_of`, the same call `pah.design.materialize` used to
    machine the joint, so the search pulls along the normal a socket was actually cut
    on instead of re-deriving one from the connector kind. Dovetail is why that
    matters: it slides along +X, and a hard-coded +Z would call every slide joint
    stuck.

    The separating direction is listed first — `contact_of` points from `a` to `b`, so
    `a` retreats along `-axis` and `b` along `+axis` — because that is the direction
    that usually works and the search tries candidates in order. The opposite sign is
    still tried: a part can be the host of one joint and the guest of another, and
    which way it leaves is a property of the whole assembly, not of one interface.

    An interface whose envelopes are not adjacent raises. The design's poses and its
    interface graph disagree at that point (which is also what the materialiser
    refuses), and searching over a silently shrunk direction set would report "not
    found" for a reason the reader could not tell apart from real geometry.
    """
    found: dict[str, list[tuple[float, float, float]]] = {
        part.id: [] for part in spec.parts
    }
    for interface in spec.interfaces:
        contact = contact_of(spec.part(interface.a), spec.part(interface.b))
        if contact is None:
            raise ValueError(
                f"interface {interface.id!r} joins {interface.a!r} and {interface.b!r}, "
                "but their envelopes are not adjacent in the spec's poses, so the joint "
                "has no normal to pull along"
            )
        axis = unit_axis(contact.axis)
        for part_id, sign in (
            (interface.a, -1.0),
            (interface.a, 1.0),
            (interface.b, 1.0),
            (interface.b, -1.0),
        ):
            key = _direction_key(axis * sign)
            if key not in found[part_id]:
                found[part_id].append(key)
    return {part_id: tuple(values) for part_id, values in found.items()}


# --------------------------------------------------------------------------- #
# The sweep
# --------------------------------------------------------------------------- #
def _projection_range(
    bounds: np.ndarray, direction: np.ndarray
) -> tuple[float, float]:
    """`(min, max)` of `direction . point` over an axis-aligned box.

    Exact, and it touches no vertices: the extreme projection of a box is the sum over
    the three axes of whichever bound the direction's sign selects. A crate of 100k
    vertices would otherwise be projected on every sample.
    """
    low = np.asarray(bounds[0], dtype=float)
    high = np.asarray(bounds[1], dtype=float)
    pick_low = np.where(direction >= 0.0, low, high)
    pick_high = np.where(direction >= 0.0, high, low)
    return float(pick_low @ direction), float(pick_high @ direction)


def _boxes_overlap(a: trimesh.Trimesh, b: trimesh.Trimesh) -> bool:
    """Could `a` and `b` share volume? Exact, and cheaper than a boolean by orders.

    Boxes that do not overlap on one axis bound disjoint point sets, so no sample can
    collide there. On a decomposition most samples are like that, which is what keeps
    a 32-sample sweep of a 19-part assembly affordable. Strict on purpose: boxes that
    only touch on a plane share a face, not a volume, and a seated chain sits exactly
    there.
    """
    low = np.maximum(a.bounds[0], b.bounds[0])
    high = np.minimum(a.bounds[1], b.bounds[1])
    return bool(np.all(high > low))


def _exit_distance(
    moving: trimesh.Trimesh,
    direction: np.ndarray,
    others: list[trimesh.Trimesh],
) -> float:
    """How far `moving` must travel before every other solid is behind it.

    Once the moving part's near face along `direction` is past every other solid's far
    face, the boxes are disjoint and no later sample can collide. That makes the sweep
    finite without guessing a distance per assembly, and it cannot under-shoot: the
    sample at that distance is already clear of everything.
    """
    if not others:
        return 0.0
    far = max(_projection_range(other.bounds, direction)[1] for other in others)
    near = _projection_range(moving.bounds, direction)[0]
    return max(0.0, far - near + _EXIT_MARGIN_MM)


class _Search:
    """Depth-first disassembly over one assembly, with a move budget.

    Mutable state on purpose: `attempts`, the abort flag and the stuck set describe one
    call, not the question, and threading them through every recursion would obscure
    the two lines that actually decide a step.
    """

    def __init__(
        self,
        *,
        solids: dict[str, trimesh.Trimesh],
        directions: dict[str, tuple[tuple[float, float, float], ...]],
        max_attempts: int,
        min_volume_mm3: float,
    ) -> None:
        self.solids = solids
        self.directions = directions
        self.budget = max(1, int(max_attempts))
        self.min_volume_mm3 = float(min_volume_mm3)
        self.attempts = 0
        self.aborted = False
        self.cut_at: str | None = None
        self.stuck: tuple[str, ...] | None = None

    # -- entry point -------------------------------------------------------
    def run(self) -> SequenceResult:
        order: list[str] = []
        complete = self._search(frozenset(self.solids), order)
        if complete:
            return SequenceResult(
                found=True,
                order=tuple(order),
                blocked_at=None,
                attempts=self.attempts,
                reason=(
                    f"straight-line disassembly found in {self.attempts} candidate "
                    f"moves: {' -> '.join(order)}; assemble in the reverse order: "
                    f"{' -> '.join(reversed(order))}. Every step was swept against the "
                    "parts still in place, so this order is constructively verified; "
                    "only pulls along an interface normal are modelled"
                ),
            )
        remaining = self.stuck or tuple(sorted(self.solids))
        if self.aborted:
            reason = (
                "no straight-line disassembly order found within the "
                f"{self.budget} candidate-move budget (cut short while trying "
                f"{self.cut_at!r}); a cut-short search is not a proof of impossibility, "
                "and more moves may still be found if max_attempts is raised"
            )
        else:
            reason = (
                f"no straight-line disassembly order found after {self.attempts} "
                f"candidate moves: part(s) {', '.join(remaining)} never came free. This "
                "is an exhausted heuristic search, not a proof of impossibility — "
                "rotations, tilted pulls and multi-segment paths are not modelled"
            )
        return SequenceResult(
            found=False,
            order=(),
            blocked_at=self.cut_at or (remaining[0] if remaining else None),
            attempts=self.attempts,
            reason=reason,
            remaining=remaining,
        )

    # -- search ------------------------------------------------------------
    def _search(self, remaining: frozenset[str], order: list[str]) -> bool:
        if not remaining:
            return True
        if len(remaining) == 1:
            # Nothing else is in the way, so the last part is removable whatever its
            # interfaces say — and a part with none at all (a one-part spec) would
            # otherwise have no direction to try and could never be taken out.
            order.append(next(iter(remaining)))
            return True
        for part_id in sorted(remaining):
            for direction in self.directions.get(part_id, ()):
                self.attempts += 1
                if self.attempts > self.budget:
                    self.aborted = True
                    self.cut_at = part_id
                    return False
                if not self._clear(part_id, direction, remaining):
                    continue
                order.append(part_id)
                if self._search(remaining - {part_id}, order):
                    return True
                order.pop()
                if self.aborted:
                    return False
        self._record_stuck(remaining)
        return False

    def _record_stuck(self, remaining: frozenset[str]) -> None:
        """Remember the *smallest* set the search could not get through.

        The deepest stuck state is the narrowest description of the problem ("these
        two parts trap each other"), which is what a report should print. Keeping the
        root state instead would say "the whole assembly": true, and useless.
        """
        candidate = tuple(sorted(remaining))
        if self.stuck is None or len(candidate) < len(self.stuck):
            self.stuck = candidate

    def _clear(
        self,
        part_id: str,
        direction: tuple[float, float, float],
        remaining: frozenset[str],
    ) -> bool:
        """Can `part_id` be pulled the whole way along `direction` without touching?

        Every sample has to be clear, not just the end pose: a path *through* material
        is not a path, even when the part ends up outside. Distance 0 is sampled too,
        so an assembly that already interpenetrates has no removable part — the honest
        answer for a design whose solids share space (see the module docstring).
        """
        moving = self.solids[part_id]
        axis = np.asarray(direction, dtype=float)
        others = [self.solids[other] for other in sorted(remaining) if other != part_id]
        distance = _exit_distance(moving, axis, others)
        if distance <= 0.0:
            return True
        steps = min(_MAX_SAMPLES, max(1, int(math.ceil(distance / _MAX_STEP_MM))))
        step = axis * (distance / steps)
        probe = moving.copy()
        for index in range(0, steps + 1):
            if index:
                probe.apply_translation(step)
            for other in others:
                if not _boxes_overlap(probe, other):
                    continue
                if intersection_volume_mm3(probe, other) > self.min_volume_mm3:
                    return False
        return True
