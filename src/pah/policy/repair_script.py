"""M1 minimum repair script: the deterministic repair ladder for the two anchor-gap codes.

M1 asks one question: **how much of the 388/705 yield is lost to a repairable
anchor rather than to unprintable geometry?** The 238 edges behind that question
carry two codes (`gap_too_wide` 166, `no_material_hit` 72), and both are diagnosed
before any boolean runs — they are properties of the ray the connector is asked to
sit on, not of the meshes.

Scope (M1 = "repair without touching geometry")
------------------------------------------------
Allowed: connector radius / depth, connector family, in-plane anchor position.
Forbidden here: split / merge / thicken / scale, re-modelling a family, writing
`edges_v2.json`. `repair_edge` therefore never touches a part mesh — it only ever
produces a new `(edge dict, theta, section)` for `connector.build` to consume.

Why the radius ladder is enough for about a third of `gap_too_wide`
-------------------------------------------------------------------
`connector._build_peg_socket` clamps `r` to the local wall cap
`r_wall_max = t_b / 2.25` (`connector.py:759`), then requires `eng_max > need`
where `eng_max = min(d_cap, t_b + gap - 1.25 r)` and `need = gap + r`
(`:807-820`). At `r == r_wall_max` the wall term equals `need` **identically**:

    t_b - 1.25 r = t_b - 1.25 * t_b/2.25 = 0.4444 t_b = r
    =>  eng_max - need = t_b - 2.25 r = 0

so `eng_max <= need` holds and the edge fails `gap_too_wide` however small the gap
is. Asking for any `r` strictly below the wall cap turns that equality into slack.
The cheapest repair in this file is therefore a parameter change, not an algorithm
change.

What the other two operators are for
------------------------------------
A `gap_too_wide` edge whose gap alone exceeds `d_cap` cannot be saved by any
radius (`need >= gap > d_cap >= eng_max`). The only lever left inside M1 is the
anchor's in-plane position: the ray at a different `(u, v)` measures a different
local gap and wall. `relocate_connector` searches that grid and hands
`connector.build` a **new `edge` dict** whose `anchor` is the chosen point
(`connector._resolve_frame` prefers `anchor` over `center`, `:562-568`) — exactly
the "re-pick the anchor inside the patch" route.

`no_material_hit` needs the same lever and no radius alternative: the verdict is
"this ray hits no material on one side", and `line_intervals` never looks at `r`,
so `resize_connector` cannot change it. The rule table (`pah-dev-doc-v1.md` §5.3)
agrees — `relocate_connector` comes first for that code.

`swap_family` is the third rung. It cannot change either target code (both fire
before the section is built), so it exists for the case where a radius fix does
get past the gate and then trips a *volume* check (`peg_detached` /
`socket_breaks_wall`): a D-shaped and a round pin remove different volumes.

Determinism
-----------
Every decision is a pure function of the measured numbers and of a fixed
enumeration order (radius rungs, grid positions, operator ladder). There is no
randomness, no dict-order dependence and — unless the caller sets a wall-clock or
absolute deadline — no dependence on time at all. Given the same `edge` and the
same budget, `repair_edge` emits the same attempt sequence.

Optional residual-wall floor (default off)
------------------------------------------
`wall_need_abs` / `wall_need_frac` are an **opt-in** extra bound on the pin radius:
the socket must leave `max(wall_need_abs, wall_need_frac * t_b)` of B's first layer
behind it, so `closed_form_radius` also requires `r <= (t_b - wall_need) / 2.25`.
Both default to `0.0`, which adds no bound and leaves every decision bit-identical
to the M1 run (`runs/m1-min/rows.jsonl`); when they are on, the radius ladder is
searched *under* the floor, so a pin that would eat the wall is never planned
instead of being built and rejected. That floor is a caller's standard, not the
source's: `connector.build` enforces only `r <= t_b / 2.25` at `clamp=True`.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from pah.tools.verifiers.connector_feasible import (
    DEPTH_MIN_MM,
    R_MIN_MM,
    SECTION,
    classify,
)

__all__ = [
    "MAX_ATTEMPTS",
    "AssemblyParts",
    "AttemptBudget",
    "RepairAttempt",
    "RepairReport",
    "closed_form_radius",
    "gate_margin",
    "next_operator",
    "repair_edge",
    "resize_rungs",
]

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
#: Hard cap from the M1 brief: "每个 edge 最多 3 次尝试". A caller may ask for
#: fewer attempts, never for more.
MAX_ATTEMPTS = 3

#: `connector.BOTTOM_GAP_R + connector.MIN_WALL_R` (`connector.py:66-67`). The
#: gate subtracts `1.25 * r` of socket budget from the wall, so the closed-form
#: radius policy has to use the same number. It is the only constant this module
#: mirrors from the source, and `scripts/m1_repair.py` reports
#: observed-vs-predicted agreement so the mirror stays checkable.
SOCKET_BUDGET_R = 1.25
#: Wall constraint `r <= t_b / 2.25` (`connector.py:759`, `:774`).
WALL_RATIO = 2.25
#: Headroom under the wall cap. At `r == t_b/2.25` the gate equality bites and
#: `wall_remaining == 0.25 r` exactly (the print floor, not above it); 2 % below
#: the cap buys gate slack and wall margin for +2 % pin radius.
WALL_HEADROOM = 0.98
#: Five rungs in `[r_min, r_hi]` — the brief's "在 [r_min, r_cap] 内扫 5 个半径档".
RESIZE_RUNGS = 5
#: In-plane search fractions of `0.9 * half_w` / `0.9 * half_h`, the same box
#: `connector._resolve_frame` clips `(u, v)` to (`:556-561`), so the grid never
#: asks for a position the source itself would refuse.
GRID_FRACS = (0.0, -0.45, 0.45, -0.8, 0.8)
#: Codes that mean "no radius and no anchor will help this edge".
_TERMINAL_CODES = frozenset(
    {"unreatable_too_small", "host_geometry_unreliable", "no_manifold3d"}
)
#: Codes that fire *after* the pre-boolean gate — i.e. the peg and socket were
#: built, so the position and radius were good enough. Everything else is a gate
#: failure and is fixed by moving/shrinking, not by changing the family.
_POST_GATE_CODES = frozenset(
    {"peg_detached", "socket_breaks_wall", "open_after_bool", "bool_failed"}
)

LADDER_GAP = ("resize_connector", "relocate_connector", "swap_family")
LADDER_MATERIAL = ("relocate_connector", "resize_connector", "swap_family")
_LADDER_BY_CODE = {
    "gap_too_wide": LADDER_GAP,
    "no_material_hit": LADDER_MATERIAL,
}


def next_operator(
    source: str, used: Sequence[str], *, gate_failed: bool
) -> str | None:
    """The next rung of the §5.3 ladder, given what the previous attempts found.

    Two rules, both from the rule table's ordering plus the code semantics:

    * still failing at the gate → walk the table left to right (cheapest first);
    * already past the gate and failing on a volume check → the in-plane search
      cannot help (the gate already passes here), so the next cheap lever is the
      family swap, which changes what the boolean removes without moving anything.

    Pure function of `(source, used, gate_failed)`, so the sequence is reproducible.
    """
    remaining = [op for op in _LADDER_BY_CODE.get(source, LADDER_GAP) if op not in used]
    if not remaining:
        return None
    if not gate_failed and "swap_family" in remaining:
        return "swap_family"
    return remaining[0]


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RepairAttempt:
    """One `connector.build` call, recorded even when it succeeded.

    `reason` is the promoted code vocabulary of `connector_feasible.classify`
    (`gap_too_wide`, `peg_detached`, ...), never a merged "failed": the point of
    the ledger is that the residual codes say where to look next. `raw_reason` /
    `detail` keep what the source actually said, so nothing is lost between the
    two vocabularies.
    """

    edge_id: str
    operator: str
    params: dict
    success: bool
    reason: str | None
    elapsed_s: float
    raw_reason: str = ""
    detail: str = ""


@dataclass(frozen=True)
class RepairReport:
    """Everything one edge cost, including the attempts that failed.

    `exhausted` says the budget ran out while a rung was still unspent (a
    truncated search, to be re-run), while `stopped == "terminal"` says the
    remaining rungs were skipped *because they provably cannot help* — a claim
    the caller can check against `reason`.
    """

    case_id: str
    edge_id: str
    success: bool
    reason: str | None
    attempts: tuple[RepairAttempt, ...]
    wall_s: float
    build_calls: int
    measure_calls: int
    exhausted: bool
    stopped: str
    final: dict = field(default_factory=dict)
    result: dict = field(default_factory=dict)

    @property
    def n_attempts(self) -> int:
        return len(self.attempts)

    @property
    def operators(self) -> tuple[str, ...]:
        return tuple(a.operator for a in self.attempts)


@dataclass(frozen=True)
class AttemptBudget:
    """What one edge may spend.

    `max_attempts` caps repair attempts (build calls that changed the plan) and is
    clamped to `MAX_ATTEMPTS`. `deadline` is an absolute `time.monotonic()` value
    so a whole run can stop at 90 minutes; it is the only field that can make two
    identical calls diverge, which is why it defaults to `None` and why the
    determinism test leaves it unset.
    """

    max_attempts: int = MAX_ATTEMPTS
    max_build_calls: int | None = None
    max_measure_calls: int | None = None
    max_wall_s: float | None = None
    deadline: float | None = None

    def attempts_cap(self) -> int:
        return max(1, min(int(self.max_attempts), MAX_ATTEMPTS))

    def limit_hit(
        self,
        *,
        attempts: int,
        build_calls: int,
        measure_calls: int,
        elapsed_s: float,
        now: float,
    ) -> str | None:
        """The first limit already spent, or `None` to keep going."""
        if attempts >= self.attempts_cap():
            return "attempts"
        if self.max_build_calls is not None and build_calls >= self.max_build_calls:
            return "build_calls"
        if self.max_measure_calls is not None and measure_calls >= self.max_measure_calls:
            return "measure_calls"
        if self.max_wall_s is not None and elapsed_s >= self.max_wall_s:
            return "wall_s"
        if self.deadline is not None and now >= self.deadline:
            return "deadline"
        return None


@dataclass(frozen=True)
class AssemblyParts:
    """The one assembly a repair may look at: meshes, indices, scale.

    `meshes` must be the undecimated, trimmed parts in metres — the geometry the
    thickness was measured on (`adapter-contract.md` §E2). Nothing here is mutated
    by a repair; `scripts/m1_repair.py` hashes the tuples before and after the pass
    to prove it.
    """

    case_id: str
    unit_mm: float
    meshes: tuple[Any, ...]
    part_meta: tuple[dict | None, ...] = ()
    n_parts: int = 0

    def part(self, index: int) -> Any:
        return self.meshes[int(index)]

    def quality(self, index: int) -> dict | None:
        try:
            return self.part_meta[int(index)]
        except (IndexError, TypeError):
            return None


# --------------------------------------------------------------------------- #
# Ports — the two calls into the source, injectable so tests need no geometry
# --------------------------------------------------------------------------- #
BuildPort = Callable[..., dict]
MeasurePort = Callable[..., dict]


def _connector_build(
    assembly: AssemblyParts, edge: dict, theta: Sequence[float], *, section: str
) -> dict:
    """`connector.build` with the parameter set the frozen M0 rows were measured
    under (`connector_feasible._build_once`): `clamp=True`, `verify_stl=False`,
    `overlap_check=True`, `depth_min_mm=0`, `snap=None`.

    `verify_stl=False` is not a relaxed printability check here: `bnd1 == 0` is
    enforced on both outputs at `connector.py:906-910` independently of that flag,
    so a success still means "closed". What is skipped is the STL round-trip, to
    keep the repair cost comparable with the M0 baseline it is lifting.
    """
    from pah.tools.verifiers.connector_feasible import _connector  # noqa: PLC0415

    a_i, b_i = int(edge["a_index"]), int(edge["b_index"])
    return _connector().build(
        assembly.part(a_i),
        assembly.part(b_i),
        edge,
        theta,
        clamp=True,
        verify_stl=False,
        overlap_check=True,
        unit_mm=float(assembly.unit_mm),
        r_min_mm=R_MIN_MM,
        depth_min_mm=DEPTH_MIN_MM,
        host_quality={"a": assembly.quality(a_i), "b": assembly.quality(b_i)},
        section=section,
        snap=None,
    )


def _connector_measure(
    assembly: AssemblyParts, edge: dict, theta: Sequence[float]
) -> dict:
    """`connector.measure_axis` — ray intervals only, no boolean (`:498`).

    This is the cheap half of the call: it returns exactly the `gap` / `t_a` /
    `t_b` / `penetration` a build's gate will measure, because both go through
    `_resolve_frame` + `line_intervals` with the same `(edge, theta)`. That is
    what makes a closed-form radius policy possible.
    """
    from pah.tools.verifiers.connector_feasible import _connector  # noqa: PLC0415

    a_i, b_i = int(edge["a_index"]), int(edge["b_index"])
    return _connector().measure_axis(assembly.part(a_i), assembly.part(b_i), edge, theta)


# --------------------------------------------------------------------------- #
# Closed-form gate model — a rearrangement of `connector._build_peg_socket` §5,
# not a new criterion (see the module docstring)
# --------------------------------------------------------------------------- #
def gate_margin(gap: float, t_b: float, d_cap: float, r: float) -> tuple[bool, float]:
    """`(passes_the_pre-boolean_gate, slack_in_data_units)`.

    `slack = eng_max - need`. Positive means the build reaches the boolean stage;
    negative means it returns `gap_too_wide` without spending one.
    """
    wall = t_b + gap - SOCKET_BUDGET_R * r
    if wall <= 0:
        return False, -math.inf
    eng_max = min(d_cap, wall) if d_cap > 0 else wall
    need = gap + r if gap > 0 else 0.0
    return bool(eng_max > need), float(eng_max - need)


def resize_rungs(r_min: float, r_hi: float, n: int = RESIZE_RUNGS) -> tuple[float, ...]:
    """`n` radii in `[r_min, r_hi]`, largest first — smallest edit, strongest pin.

    Geometric spacing (the quantity that matters is a ratio, not a difference),
    and `r_hi` first because the largest admissible radius is both the least
    edited design and the one with the most engagement length. A `r_hi` at or
    below the print floor collapses to `(r_min,)`: one honest attempt, no fake
    ladder.
    """
    lo, hi = float(r_min), float(r_hi)
    if not hi > lo or n < 2:
        return (lo,)
    step = (hi / lo) ** (1.0 / (n - 1))
    return tuple(hi / (step**i) for i in range(n))


def closed_form_radius(
    gap: float, t_b: float, d_cap: float, r_min: float, r_cap: float,
    *, wall_need: float = 0.0,
) -> float:
    """The largest `r` that clears the wall cap strictly and leaves gate slack.

    Three upper bounds, smallest wins:

    * `r_cap` — in-plane room at the patch (`connector.radius_cap`);
    * `WALL_HEADROOM * t_b / 2.25` — under the wall clamp, so `t_b - 2.25r > 0`;
    * `(d_cap - gap) * 0.98` — the depth cap must exceed `gap + r`.

    `wall_need` (opt-in, default `0.0` = off) adds a fourth: the socket has to
    leave that much of `t_b` behind it, i.e. `r <= (t_b - wall_need) / 2.25`. It is
    off by default so the M1 decision sequence is unchanged; see the module
    docstring.

    Returns `r_min` when nothing is admissible, so the caller still makes one
    attempt and records the source's own reason instead of guessing.
    """
    bounds: list[float] = []
    if r_cap > 0:
        bounds.append(float(r_cap))
    if t_b > 0:
        bounds.append(WALL_HEADROOM * float(t_b) / WALL_RATIO)
        if wall_need > 0.0:
            bounds.append((float(t_b) - float(wall_need)) / WALL_RATIO)
    if d_cap > 0:
        bounds.append((float(d_cap) - float(gap)) * 0.98)
    admissible = [b for b in bounds if b > 0]
    if not admissible:
        return float(r_min)
    return max(min(admissible), float(r_min))


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _norm(vector: Sequence[float]) -> tuple[float, float, float]:
    n = [float(x) for x in vector]
    length = math.sqrt(sum(x * x for x in n)) or 1.0
    return (n[0] / length, n[1] / length, n[2] / length)


def _plane_basis(n: Sequence[float]) -> tuple[list[float], list[float]]:
    """`connector.plane_basis` (`:138-145`), copied exactly.

    The copy has to be *exact*, not merely orthogonal: `(u, v)` is turned into a
    world point by this basis in the relocation operator and by the source in
    `_resolve_frame`, and if the two bases differ by a rotation then the position
    measured is not the position built — which reads as "the ray had material
    where the peg did not". The first M1 pass made exactly that mistake; the
    `test_plane_basis_matches_the_source_construction` case pins it.
    """
    nn = _norm(n)
    ref = [0.0, 0.0, 1.0] if abs(nn[2]) < 0.9 else [1.0, 0.0, 0.0]
    e1 = [
        ref[1] * nn[2] - ref[2] * nn[1],
        ref[2] * nn[0] - ref[0] * nn[2],
        ref[0] * nn[1] - ref[1] * nn[0],
    ]
    length = math.sqrt(sum(x * x for x in e1)) or 1.0
    e1 = [x / length for x in e1]
    e2 = [
        nn[1] * e1[2] - nn[2] * e1[1],
        nn[2] * e1[0] - nn[0] * e1[2],
        nn[0] * e1[1] - nn[1] * e1[0],
    ]
    return e1, e2


def _theta(
    u: float, v: float, n: Sequence[float], r: float, depth: float
) -> tuple[float, ...]:
    return (float(u), float(v), n[0], n[1], n[2], float(r), float(depth))


def _grid(half_w: float, half_h: float) -> tuple[tuple[float, float], ...]:
    """Centre-first 5×5 in-plane offsets, in data units.

    Ordered by how far the anchor moves (`max`, then `sum` of the fractions), so
    the first position that works is also the smallest edit — the "最小修改成本"
    ordering of `pah-dev-doc-v1.md` §6.3 rather than an arbitrary sweep.
    """
    w, h = 0.9 * float(half_w), 0.9 * float(half_h)
    if w <= 0 or h <= 0:
        return ((0.0, 0.0),)
    pairs = sorted(
        ((fu, fv) for fu in GRID_FRACS for fv in GRID_FRACS),
        key=lambda p: (max(abs(p[0]), abs(p[1])), abs(p[0]) + abs(p[1]), p[0], p[1]),
    )
    return tuple((fu * w, fv * h) for fu, fv in pairs)


def _scalars(result: dict) -> dict:
    """The build result without its two output meshes — a report must not pin GBs."""
    keep = (
        "ok", "reason", "detail", "degen_code", "u", "v", "r", "r_cap", "depth",
        "t_a", "t_b", "gap", "penetration", "a_front", "b_near", "wall_remaining",
        "embed", "base_t", "tip_t", "sock_bot_t", "protrusion", "peg_volume",
        "sock_volume", "dv_a", "dv_b", "r_clamped", "depth_clamped",
        "peg_occupancy_in_a", "socket_occupancy_in_b", "unit_mm", "anchor_shift",
        "section",
    )
    return {k: result[k] for k in keep if k in result}


def _verdict(result: dict) -> tuple[str, str, str]:
    """`(code, raw_reason, detail)` — the promoted code, never a merged word."""
    code, _mechanism = classify(result)
    return code, str(result.get("reason") or ""), str(result.get("detail") or "")


def _terminal(code: str, result: dict) -> bool:
    """Whether the rungs left are provably useless for this edge."""
    if code in _TERMINAL_CODES:
        return True
    return code == "bool_failed" and "MemoryError" in str(result.get("detail") or "")


# --------------------------------------------------------------------------- #
# State carried between rungs
# --------------------------------------------------------------------------- #
@dataclass
class _State:
    """What the ladder knows after each attempt; the only mutable thing here."""

    edge: dict
    unit_mm: float
    r_min: float
    r: float
    depth: float
    section: str = SECTION
    u: float = 0.0
    v: float = 0.0
    resize_tries: int = 0
    reading: dict | None = None
    rungs: tuple[float, ...] = ()
    gate_failed: bool = True
    #: Opt-in residual-wall floor (see the module docstring). Both zero — the
    #: default — means the floor is off and every plan is the M1 plan.
    wall_need_abs: float = 0.0
    wall_need_frac: float = 0.0
    #: How far the repair moved the anchor from the one on the record, in data
    #: units. Only `relocate_connector` changes it — the operators that do not
    #: move the anchor must report a zero shift, or "did we move the joint?" would
    #: be unanswerable.
    shift_units: float = 0.0

    @property
    def wall_floor_on(self) -> bool:
        return self.wall_need_abs > 0.0 or self.wall_need_frac > 0.0

    def wall_need(self, t_b: float) -> float:
        """The floor at this `t_b`, in data units: `max(abs, frac * t_b)`."""
        return max(float(self.wall_need_abs), float(self.wall_need_frac) * float(t_b))

    def depth_for(self, r: float, gap: float, t_b: float, d_cap: float) -> float:
        """The requested `eng`, shortened when the residual-wall floor demands it.

        With the floor off this is `min(2r, d_cap)`, unchanged. With it on, the
        build's own arithmetic decides how deep the socket ends up:
        `wall_remaining = t_b + gap - eng - r`, so keeping `floor` behind it means
        `eng <= t_b + gap - 1.25 r - floor` (the socket also spends `0.25 r` of
        bottom gap). Shortening the request is the least invasive way to honour the
        floor: the radius bound already guarantees that window still contains
        `need = gap + r`, otherwise the caller would be asking for the impossible.
        """
        depth = min(2.0 * float(r), d_cap) if d_cap > 0 else 2.0 * float(r)
        if not self.wall_floor_on:
            return depth
        floor = self.wall_need(t_b)
        keep = float(t_b) + float(gap) - SOCKET_BUDGET_R * float(r) - floor
        need = float(gap) + float(r) if gap > 0 else 0.0
        if keep > 0.0 and depth > keep:
            return max(keep, need)
        return depth

    def mm(self, value: float) -> float:
        return float(value) * self.unit_mm

    def theta(self, n: Sequence[float]) -> tuple[float, ...]:
        return _theta(self.u, self.v, n, self.r, self.depth)

    def params(self, operator: str, **extra: Any) -> dict:
        """Millimetre parameters at the 225 mm scale — the unit every report uses."""
        out = {
            "operator": operator,
            "u_mm": round(self.mm(self.u), 6),
            "v_mm": round(self.mm(self.v), 6),
            "r_mm": round(self.mm(self.r), 6),
            "depth_mm": round(self.mm(self.depth), 6),
            "section": self.section,
            "shift_mm": round(self.mm(self.shift_units), 6),
        }
        out.update(extra)
        return out

    def take(self, reading: dict) -> None:
        """Adopt measured gate inputs (from a ray pass or a failed build)."""
        self.reading = {
            "ok": bool(reading.get("ok", True)),
            "gap": float(reading.get("gap") or 0.0),
            "t_b": float(reading.get("t_b") or 0.0),
            "t_a": float(reading.get("t_a") or 0.0),
        }

    def forget(self) -> None:
        self.reading = None


# --------------------------------------------------------------------------- #
# The repair itself
# --------------------------------------------------------------------------- #
def repair_edge(
    assembly: AssemblyParts,
    edge: dict,
    *,
    budget: AttemptBudget,
    build: BuildPort | None = None,
    measure: MeasurePort | None = None,
    wall_need_abs: float = 0.0,
    wall_need_frac: float = 0.0,
) -> RepairReport:
    """Try the cheapest operator that can still work, at most `MAX_ATTEMPTS` times.

    The ladder is §5.3's rule table restricted to M1's three parameter operators,
    and every rung is a **real** `connector.build` call — nothing is skipped on a
    prediction alone, so a "not repaired" row carries the source's own reason
    instead of this module's opinion. The prediction only orders the work: it
    picks which of the five radius rungs to try first and ends the in-plane search
    as soon as a position clears the gate.

    `wall_need_abs` / `wall_need_frac` are the opt-in residual-wall floor (module
    docstring). At their defaults nothing is added, so the plan — and the row it
    produces — is exactly the M1 plan.

    `edge` is read, never written: the relocated variant is a copy.
    """
    started = time.perf_counter()
    build = build or _connector_build
    measure = measure or _connector_measure
    cap = budget.attempts_cap()

    unit_mm = float(assembly.unit_mm)
    r_min = R_MIN_MM / unit_mm
    r_cap = float(edge.get("r_cap") or edge.get("r_inplane_max") or 0.0)
    d_cap = float(edge.get("d_cap") or edge.get("depth_max") or 0.0)
    n = _norm(edge["n_ab"])
    source = str(edge.get("code") or "")
    edge_id = str(edge.get("edge_id", ""))
    base = [float(x) for x in (edge.get("anchor") or edge.get("center") or (0.0, 0.0, 0.0))]
    depth0 = min(2.0 * r_min, d_cap) if d_cap > 0 else 2.0 * r_min

    state = _State(
        edge=dict(edge),
        unit_mm=unit_mm,
        r_min=r_min,
        r=r_min,
        depth=depth0,
        wall_need_abs=float(wall_need_abs),
        wall_need_frac=float(wall_need_frac),
    )
    attempts: list[RepairAttempt] = []
    build_calls = measure_calls = 0
    stopped = "attempts"
    reason: str | None = None
    last: dict = {}

    def measure_at(u: float, v: float, r: float, depth: float, edge_now: dict) -> dict:
        nonlocal measure_calls
        measure_calls += 1
        return measure(assembly, edge_now, _theta(u, v, n, r, depth))

    used: list[str] = []
    for _rung in range(cap):
        limit = budget.limit_hit(
            attempts=len(attempts),
            build_calls=build_calls,
            measure_calls=measure_calls,
            elapsed_s=time.perf_counter() - started,
            now=time.monotonic(),
        )
        if limit is not None:
            stopped = limit
            break
        operator = next_operator(source, used, gate_failed=state.gate_failed)
        if operator is None:
            stopped = "attempts" if attempts else "no_plan"
            break
        used.append(operator)

        plan = _plan(operator, state=state, n=n, base=base, r_cap=r_cap, d_cap=d_cap,
                     measure_at=measure_at)
        if plan is None:  # this rung has nothing to offer for this edge
            continue
        theta, section, params = plan

        t0 = time.perf_counter()
        build_calls += 1
        result = build(assembly, state.edge, theta, section=section)
        elapsed = time.perf_counter() - t0

        if result.get("ok"):
            attempts.append(
                RepairAttempt(edge_id=edge_id, operator=operator, params=params,
                              success=True, reason=None, elapsed_s=round(elapsed, 4))
            )
            last = result
            stopped, reason = "success", None
            break

        code, raw_reason, detail = _verdict(result)
        attempts.append(
            RepairAttempt(
                edge_id=edge_id,
                operator=operator,
                params=params,
                success=False,
                reason=code,
                elapsed_s=round(elapsed, 4),
                raw_reason=raw_reason,
                detail=detail,
            )
        )
        last = result
        reason = code
        state.section = section
        state.gate_failed = code not in _POST_GATE_CODES
        if "gap" in result and "t_b" in result:
            # The build already measured this position; reuse it instead of paying
            # a second ray pass for the next rung.
            state.take({"ok": True, **{k: result[k] for k in ("gap", "t_b", "t_a")
                                      if k in result}})
        if _terminal(code, result):
            stopped = "terminal"
            break
    else:
        # The loop ran out of rungs *or* of budget — say which. A complete search
        # that still failed is not a truncated one, and a report that cannot tell
        # them apart would hide the work left undone.
        if next_operator(source, used, gate_failed=state.gate_failed) is None:
            stopped = "rungs" if attempts else "no_plan"
        else:
            stopped = "attempts"

    success = bool(attempts and attempts[-1].success)
    if not success and reason is None:
        # Nothing was built because every rung was ruled out by a measurement
        # (no material anywhere in the patch, say). The edge keeps the code it
        # came in with — that is the honest verdict, and `stopped` says this
        # module added nothing to it.
        reason = source or None
    final = {
        "operator": attempts[-1].operator if attempts else "",
        "params": attempts[-1].params if attempts else {},
        "reason": reason,
        "section": state.section,
        "r_mm": round(state.mm(state.r), 6),
        "r_min_mm": R_MIN_MM,
        "r_cap_mm": round(state.mm(r_cap), 6),
        "d_cap_mm": round(state.mm(d_cap), 6),
        "anchor_shift_mm": round(state.mm(state.shift_units), 6),
    }
    return RepairReport(
        case_id=str(assembly.case_id),
        edge_id=edge_id,
        success=success,
        reason=reason,
        attempts=tuple(attempts),
        wall_s=round(time.perf_counter() - started, 4),
        build_calls=build_calls,
        measure_calls=measure_calls,
        exhausted=stopped
        in ("attempts", "deadline", "wall_s", "build_calls", "measure_calls"),
        stopped=stopped,
        final=final,
        result=_scalars(last),
    )


# --------------------------------------------------------------------------- #
# Rungs
# --------------------------------------------------------------------------- #
def _plan(
    operator: str,
    *,
    state: _State,
    n: Sequence[float],
    base: Sequence[float],
    r_cap: float,
    d_cap: float,
    measure_at: Callable[..., dict],
) -> tuple[tuple[float, ...], str, dict] | None:
    """Materialise one rung into `(theta, section, params)`, or `None` to skip."""
    if operator == "resize_connector":
        return _plan_resize(state, n, d_cap, measure_at, r_cap)
    if operator == "relocate_connector":
        return _plan_relocate(state, n, base, r_cap, d_cap, measure_at)
    if operator == "swap_family":
        return _plan_swap(state, n, d_cap)
    raise ValueError(f"unknown operator {operator!r}")


def _plan_resize(
    state: _State,
    n: Sequence[float],
    d_cap: float,
    measure_at: Callable[..., dict],
    r_cap: float,
) -> tuple[tuple[float, ...], str, dict] | None:
    """Closed-form radius over five rungs, largest admissible first.

    The first call measures one ray at the current position; a later call reuses
    that reading, so repeating this operator after a post-gate failure costs no
    measurement at all.
    """
    if state.reading is None:
        reading = measure_at(state.u, state.v, state.r, state.depth, state.edge)
        if not reading.get("ok"):
            return None  # no material on one side: no radius can change that
        state.take(reading)
        assert state.reading is not None
        r_hi = closed_form_radius(
            state.reading["gap"], state.reading["t_b"], d_cap, state.r_min, r_cap,
            wall_need=state.wall_need(state.reading["t_b"]),
        )
        state.rungs = resize_rungs(state.r_min, r_hi)
    assert state.reading is not None
    gap, t_b = state.reading["gap"], state.reading["t_b"]

    if state.resize_tries == 0:
        r = state.rungs[-1]
        for candidate in state.rungs:
            if gate_margin(gap, t_b, d_cap, candidate)[0]:
                r = candidate
                break
    else:  # a previous rung failed past the gate → next shorter pin
        r = state.rungs[min(state.resize_tries, len(state.rungs) - 1)]
    state.resize_tries += 1
    state.r = r
    state.depth = state.depth_for(r, gap, t_b, d_cap)
    passes, slack = gate_margin(gap, t_b, d_cap, r)
    floor = {}
    if state.wall_floor_on:
        # Recorded only when the floor is in play, so a default run's `params` are
        # byte-identical to the M1 rows.
        floor["wall_need_mm"] = round(state.mm(state.wall_need(t_b)), 6)
    params = state.params(
        "resize_connector",
        gap_mm=round(state.mm(gap), 6),
        t_b_mm=round(state.mm(t_b), 6),
        gate_predicted=bool(passes),
        gate_slack_mm=round(state.mm(slack), 6) if math.isfinite(slack) else None,
        rungs_mm=[round(state.mm(x), 6) for x in state.rungs],
        **floor,
    )
    return state.theta(n), state.section, params


def _plan_relocate(
    state: _State,
    n: Sequence[float],
    base: Sequence[float],
    r_cap: float,
    d_cap: float,
    measure_at: Callable[..., dict],
) -> tuple[tuple[float, ...], str, dict] | None:
    """Search the patch for an in-plane position whose ray clears the gate.

    The first position (centre-first order) where both parts have material *and*
    the closed-form radius clears the gate wins; a new `edge` dict carries that
    point as `anchor`, so the source re-derives `(u, v) = (0, 0)` from it
    (`connector.py:562-568`). If no position clears the gate, the best measured
    margin still gets one real attempt instead of a skip.
    """
    half_w = float(state.edge.get("half_w") or 0.0)
    half_h = float(state.edge.get("half_h") or 0.0)
    depth = min(2.0 * state.r_min, d_cap) if d_cap > 0 else 2.0 * state.r_min

    best: tuple[float, float, float, dict, float] | None = None
    chosen: tuple[float, float, float, dict, float] | None = None
    tried = 0
    for du, dv in _grid(half_w, half_h):
        reading = measure_at(du, dv, state.r_min, depth, state.edge)
        tried += 1
        if not reading.get("ok"):
            continue
        gap = float(reading.get("gap") or 0.0)
        t_b = float(reading.get("t_b") or 0.0)
        r = closed_form_radius(
            gap, t_b, d_cap, state.r_min, r_cap, wall_need=state.wall_need(t_b)
        )
        passes, slack = gate_margin(gap, t_b, d_cap, r)
        candidate = (
            du, dv, r,
            {"ok": True, "gap": gap, "t_b": t_b, "t_a": float(reading.get("t_a") or 0.0)},
            slack,
        )
        if passes:
            chosen = candidate
            break
        if best is None or slack > best[4]:
            best = candidate
    if chosen is None:
        if best is None:
            return None
        chosen = best
    du, dv, r, reading, _slack = chosen

    e1, e2 = _plane_basis(n)
    point = [float(base[i]) + du * e1[i] + dv * e2[i] for i in range(3)]
    state.edge = dict(state.edge)
    state.edge["anchor"] = point
    state.u, state.v = 0.0, 0.0
    state.shift_units = math.hypot(du, dv)
    state.take(reading)
    assert state.reading is not None
    state.r = r
    state.depth = state.depth_for(r, state.reading["gap"], state.reading["t_b"], d_cap)
    state.rungs = resize_rungs(state.r_min, r)
    state.resize_tries = 0
    passes, slack = gate_margin(state.reading["gap"], state.reading["t_b"], d_cap, r)
    floor = {}
    if state.wall_floor_on:
        floor["wall_need_mm"] = round(state.mm(state.wall_need(state.reading["t_b"])), 6)
    params = state.params(
        "relocate_connector",
        grid_positions=tried,
        du_mm=round(state.mm(du), 6),
        dv_mm=round(state.mm(dv), 6),
        gap_mm=round(state.mm(state.reading["gap"]), 6),
        t_b_mm=round(state.mm(state.reading["t_b"]), 6),
        gate_predicted=bool(passes),
        gate_slack_mm=round(state.mm(slack), 6) if math.isfinite(slack) else None,
        **floor,
    )
    return state.theta(n), state.section, params


def _plan_swap(
    state: _State, n: Sequence[float], d_cap: float
) -> tuple[tuple[float, ...], str, dict] | None:
    """Round pin ↔ D-shaped pin at the current radius and position.

    Precondition (`pah-dev-doc-v1.md` §5.6 criterion 3): an earlier rung must have
    reached the boolean stage. Before that the section is never built, so swapping
    it cannot change the verdict and the attempt would be a guaranteed repeat of
    the failure it is meant to fix.
    """
    if state.gate_failed:
        return None
    section = "circle" if state.section != "circle" else "D"
    state.section = section
    r = state.r if state.r > 0 else state.r_min
    if state.reading is not None:
        state.depth = state.depth_for(r, state.reading["gap"], state.reading["t_b"], d_cap)
    else:
        state.depth = min(2.0 * r, d_cap) if d_cap > 0 else 2.0 * r
    params = state.params("swap_family", section=section, gate_predicted=True)
    return state.theta(n), section, params
