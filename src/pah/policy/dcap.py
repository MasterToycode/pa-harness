"""The socket side's own wall — a local single-layer measurement, and the depth cap built on it.

Why this module exists
----------------------
`runs/m1-min/report.md` §4.1 leaves 73 `gap_too_wide` edges that no radius and no
in-plane position can rescue, because the pre-boolean gate is

    eng_max = min(d_cap, t_b + gap - 1.25 r)        # connector.py:813
    need    = gap + r                               # connector.py:803

and `d_cap` is an **edge-level constant read from `edges_v2.json`**
(`connector.depth_cap`, `:394-411`): `d_cap = 0.8 * min(t_a, t_b)` with both walls
taken from the contact graph. Two things are wrong with using it as the socket's
depth budget:

* `min(t_a, t_b)` mixes the two sides of the joint. `t_a` is the **peg** side; the
  source only ever uses it to limit the peg's *embed* (`embed = min(0.6 r, 0.5 t_a)`,
  `:790`). It says nothing about how deep a blind hole may be cut into B.
* the file's `t_b` is a per-patch *thickness* number, not the socket-side wall at
  the anchor. On this corpus it runs up to 205 mm on a 150 mm part — that is the
  whole part span along the ray (a solid region), i.e. the "含空腔总跨度" defect
  `pah-dev-doc-v1.md` §9.1 records.

So the cap this module builds is `0.8 *` the thickness of **the first material
layer of the B side along the insertion normal, at the anchor**, measured on the
undecimated meshes. B is the socket side because `_build_peg_socket` reports
`wall_remaining = near_b[1] - sock_bot_t` (`:830`) — the wall the socket can eat
is B's, measured on B's ray. A is in the signature because the same ray must show
that the plane is on B's *free* side: a ray whose A material already crosses into
B's first layer (penetration) is not measuring a free wall, so it is rejected.

How the measurement avoids the "total span" mistake
---------------------------------------------------
`connector.line_intervals` returns **every** material interval along the ±normal
line, so a hollow shell gives two (front wall, back wall) and a solid region gives
one long interval. This module takes **only the first interval ahead of the
interface plane** and never adds a second one: a cavity is a gap between
intervals, so the only way to cross one is to sum intervals, which is exactly the
mistake being corrected here. `WallProbe.span_mm` keeps the total span for the
record — the number that must *not* be used, reported so a reader can see the
difference on real rows.

What this module deliberately does not do
-----------------------------------------
It never thins a wall to buy a pass. `residual_wall_need_mm` states the floor
(`max(0.25 mm, 20 % of the local wall)`) and `thin_wall_veto` turns a build that
lands under it into an honest `degenerate_thin_wall` failure, so the ladder moves
on instead of taking the win. Nothing here mutates an `edge` dict in place:
`override_depth_cap` returns a copy, which is what keeps the intervention at the
data layer and off `edges_v2.json`.

Units
-----
Meshes and the anchor/normal are in **data units** (metres on this corpus, per
`docs/internal/adapter-contract.md` §A); every returned length is in
**millimetres** at the `unit_mm` the caller passes, the scale every report uses
(225 mm frame, `unit_mm ≈ 1500`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

from pah.tools.verifiers.connector_feasible import R_MIN_MM

__all__ = [
    "DEPTH_MAX_RATIO",
    "SLIVER_RATIO",
    "WALL_FRAC",
    "WALL_FRAC_FLOOR_MM",
    "WallProbe",
    "depth_cap_mm",
    "local_single_wall_mm",
    "override_depth_cap",
    "probe_local_wall",
    "residual_wall_need_mm",
    "thin_wall_veto",
]

#: `connector.DEPTH_MAX_RATIO` (`:81`). Mirrored, the same way
#: `pah.policy.repair_script` mirrors `1.25` and `2.25`: the data-layer cap has to
#: be built with the same ratio the gate would have applied, or the override is a
#: different experiment.
DEPTH_MAX_RATIO = 0.8
#: `connector.SLIVER_RATIO` (`:274`). A ray whose origin lies on a surface cuts a
#: 1e-4–1e-3 mm sliver that is nearest to `t = 0` and is not a wall; the source
#: drops those in `_nearest`, and a measurement that kept one would report a
#: 10^-3 mm wall and refuse every repair for the wrong reason.
SLIVER_RATIO = 1e-3
#: "该边原 `t_b` 的 20 %" — the relative half of the residual-wall floor.
WALL_FRAC = 0.2
#: `connector.R_MIN_MM` — the absolute half of the floor (0.25 mm, SLA print floor).
WALL_FRAC_FLOOR_MM = R_MIN_MM
#: What a layer must be worth keeping as "the" wall: below this the measurement is
#: noise rather than geometry, and the honest answer is `None` (测不准).
_MIN_LAYER_MM = 1e-3


# --------------------------------------------------------------------------- #
# The measurement
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class WallProbe:
    """One local-wall reading, with the evidence it was built from.

    `wall_mm` is the median single-layer thickness; `None` means the anchor could
    not be measured (no material on the socket side, a ray-parity degeneracy, or
    no ray clean enough to trust). `span_mm` is the total material span along the
    same ray — the quantity that must **not** be turned into a depth cap, kept so
    the two can be compared on a real row.
    """

    wall_mm: float | None
    rays: int
    n_valid: int
    per_ray_mm: tuple[float, ...]
    span_mm: float | None
    cavity_after_mm: float | None
    n_layers: int
    centre_ok: bool
    detail: str = ""

    @property
    def measured(self) -> bool:
        return self.wall_mm is not None


def _longest_side(mesh: Any) -> float:
    """The mesh's longest bounding-box side, in data units."""
    lo, hi = mesh.bounds
    return max(float(hi[i] - lo[i]) for i in range(3))


def _auto_sample_radius(mesh_b: Any) -> float:
    """In-plane ring radius for the multi-point median: 0.5 % of the part's length.

    Small on purpose. The ring is there to survive tessellation noise and a
    grazing facet, not to average over the patch: the number has to describe the
    spot where the pin will actually be drilled, so the spread is a *quality*
    signal (it is reported per ray) rather than something to be smoothed away. At
    this corpus's 150 mm parts the ring is ≈0.75 mm across, well inside the
    smallest patch (`half_h` can be 1.8 mm).
    """
    return 0.005 * _longest_side(mesh_b)


def _ring(rays: int, radius: float) -> list[tuple[float, float]]:
    """Centre first, then `rays - 1` points on a circle — deterministic order."""
    n = max(1, int(rays))
    if n == 1 or radius <= 0.0:
        return [(0.0, 0.0)]
    out = [(0.0, 0.0)]
    for k in range(n - 1):
        angle = 2.0 * math.pi * k / (n - 1)
        out.append((radius * math.cos(angle), radius * math.sin(angle)))
    return out


def _single_layer(intervals: Sequence[tuple[float, float]], *, plane: float = 0.0):
    """The first material layer ahead of `plane`: `(layer, cavity_after, span, n)`.

    `intervals` are `connector.line_intervals`' `[(t0, t1), ...]`, ascending.
    Intervals entirely behind the plane are ignored (they are B's material on the
    far side of the joint, not material a socket can be cut into). Among the ones
    ahead, the nearest is the layer; its thickness is `t1 - t0`, i.e. the same
    quantity the source calls `t_b` on that ray.

    Returns `(None, ...)` when nothing is ahead. `cavity_after` is the distance
    from the layer's back face to the next material, so a reader can tell "single
    wall with a cavity behind it" from "the part stays solid".
    """
    ahead = [(float(t0), float(t1)) for t0, t1 in intervals if float(t1) > plane]
    if not ahead:
        return None, None, None, 0
    thick = [t1 - t0 for t0, t1 in ahead]
    cut = max(thick) * SLIVER_RATIO
    kept = [iv for iv, th in zip(ahead, thick) if th >= cut]
    if not kept:
        kept = ahead
    kept.sort()
    layer = kept[0]
    cavity = None
    if len(kept) > 1:
        cavity = max(0.0, kept[1][0] - layer[1])
    span = kept[-1][1] - max(kept[0][0], plane)
    return layer, cavity, span, len(kept)


def probe_local_wall(
    part_a_mesh: Any,
    part_b_mesh: Any,
    anchor_world: Sequence[float],
    normal_world: Sequence[float],
    *,
    rays: int = 5,
    unit_mm: float = 1.0,
    sample_radius: float | None = None,
) -> WallProbe:
    """Measure B's first material layer at the anchor; see the module docstring.

    The median over the ring is the reading. Two conditions have to hold or the
    answer is `None` rather than a number:

    * the **centre** ray (the drill point itself) must be valid — a median of four
      neighbour rays says nothing about the spot the socket goes into;
    * at least half the rays must be valid — otherwise the patch is a patch edge,
      not a wall.

    `part_a_mesh` may be `None`: A is used only to reject rays where the two parts
    interpenetrate (B's first interval is then not a free wall), so a caller that
    only has the socket side loses that check and nothing else.

    `unit_mm` converts data units to millimetres; the default 1.0 means "the
    meshes are already millimetres".
    """
    from pah.tools.verifiers.connector_feasible import _connector  # noqa: PLC0415

    connector = _connector()
    n = [float(x) for x in normal_world]
    norm = math.sqrt(sum(x * x for x in n))
    if not norm > 1e-12:
        return WallProbe(None, max(1, int(rays)), 0, (), None, None, 0, False, "zero_normal")
    n = [x / norm for x in n]
    anchor = [float(x) for x in anchor_world]
    e1, e2 = connector.plane_basis(n)
    radius = _auto_sample_radius(part_b_mesh) if sample_radius is None else float(sample_radius)
    scale = float(unit_mm) if unit_mm else 1.0

    walls: list[float] = []
    spans: list[float] = []
    cavities: list[float] = []
    layers: list[int] = []
    centre_ok = False
    reasons: list[str] = []
    offsets = _ring(rays, radius)
    for index, (du, dv) in enumerate(offsets):
        origin = [
            anchor[i] + du * e1[i] + dv * e2[i] for i in range(3)
        ]
        try:
            iv_b, odd_b = connector.line_intervals(part_b_mesh, origin, n)
        except Exception as exc:  # noqa: BLE001 - a bad ray is a failed measurement
            reasons.append(f"ray_raised:{type(exc).__name__}")
            continue
        if odd_b:
            reasons.append(f"ray_parity_odd(b={int(odd_b)})")
            continue
        layer, cavity, span, n_layers = _single_layer(iv_b)
        if layer is None:
            reasons.append("no_material_hit_b")
            continue
        # The joint has to be a joint: if A's material reaches past B's near face
        # the two parts interpenetrate here and B's first interval is not a free
        # wall (`connector.py:795-799` treats that case as degenerate). `part_a`
        # may be `None`, which simply skips this check.
        if part_a_mesh is not None:
            try:
                iv_a, odd_a = connector.line_intervals(part_a_mesh, origin, n)
            except Exception as exc:  # noqa: BLE001
                reasons.append(f"ray_raised_a:{type(exc).__name__}")
                continue
            if odd_a:
                reasons.append(f"ray_parity_odd(a={int(odd_a)})")
                continue
            near_a = _nearest_ahead(iv_a)
            if near_a is not None and near_a[1] > layer[0] + _pen_tol(part_b_mesh):
                reasons.append("penetrated")
                continue
        wall_mm = (layer[1] - layer[0]) * scale
        if wall_mm < _MIN_LAYER_MM:
            reasons.append("degenerate_thin_wall")
            continue
        walls.append(wall_mm)
        spans.append(span * scale if span is not None else wall_mm)
        layers.append(n_layers)
        if cavity is not None:
            cavities.append(cavity * scale)
        if index == 0:
            centre_ok = True

    n_valid = len(walls)
    need = max(1, (max(1, int(rays)) + 1) // 2)
    if not centre_ok or n_valid < need:
        headline = (
            "centre_ray_invalid" if not centre_ok else f"too_few_rays({n_valid}<{need})"
        )
        reasons = [headline] + list(dict.fromkeys(reasons))
        return WallProbe(
            None,
            max(1, int(rays)),
            n_valid,
            tuple(round(w, 6) for w in walls),
            round(_median(spans), 6) if spans else None,
            round(_median(cavities), 6) if cavities else None,
            max(layers) if layers else 0,
            centre_ok,
            ";".join(reasons),
        )
    walls_sorted = sorted(walls)
    span_sorted = sorted(spans)
    return WallProbe(
        wall_mm=round(_median(walls_sorted), 6),
        rays=max(1, int(rays)),
        n_valid=n_valid,
        per_ray_mm=tuple(round(w, 6) for w in walls),
        span_mm=round(_median(span_sorted), 6),
        cavity_after_mm=round(_median(sorted(cavities)), 6) if cavities else None,
        n_layers=max(layers),
        centre_ok=True,
        detail=";".join(dict.fromkeys(reasons)) if reasons else "",
    )


def _nearest_ahead(intervals: Sequence[tuple[float, float]]):
    """The interval of A nearest the plane, if any (used only for the penetration test)."""
    if not intervals:
        return None
    return min(
        intervals,
        key=lambda iv: 0.0 if iv[0] <= 0.0 <= iv[1] else min(abs(iv[0]), abs(iv[1])),
    )


def _pen_tol(mesh: Any) -> float:
    """Tolerance for "A crossed into B", relative to the part size (`line_intervals`' own)."""
    return 1e-6 * max(_longest_side(mesh), 1e-12)


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return 0.5 * (float(ordered[mid - 1]) + float(ordered[mid]))


def local_single_wall_mm(
    part_a_mesh: Any,
    part_b_mesh: Any,
    anchor_world: Sequence[float],
    normal_world: Sequence[float],
    *,
    rays: int = 5,
    unit_mm: float = 1.0,
    sample_radius: float | None = None,
) -> float | None:
    """`probe_local_wall(...).wall_mm` — the socket side's single wall in millimetres.

    B is the inserted side: `_build_peg_socket` cuts the socket into B and reports
    `wall_remaining` against B's own ray (`connector.py:830`), so B's first layer
    is the wall a blind socket sits in. A is used only to reject rays where the
    two parts interpenetrate.

    Returns `None` when the wall cannot be measured — the anchor misses B, the ray
    is degenerate, or too few of the `rays` sample points agree. A caller must
    treat `None` as "this lever does not apply here", never as a wall of zero.
    """
    return probe_local_wall(
        part_a_mesh,
        part_b_mesh,
        anchor_world,
        normal_world,
        rays=rays,
        unit_mm=unit_mm,
        sample_radius=sample_radius,
    ).wall_mm


# --------------------------------------------------------------------------- #
# The cap and the floor built on that measurement
# --------------------------------------------------------------------------- #
def depth_cap_mm(wall_mm: float) -> float:
    """`0.8 * wall` — the same ratio `connector.depth_cap` applies to the file's wall.

    Only the *wall* changes provenance (local single layer instead of the file's
    `min(t_a, t_b)`); the ratio stays the source's, so the two caps differ by
    exactly the quantity under test.
    """
    return DEPTH_MAX_RATIO * float(wall_mm)


def residual_wall_need_mm(wall_mm: float, *, r_min_mm: float = WALL_FRAC_FLOOR_MM,
                          frac: float = WALL_FRAC) -> float:
    """The floor a socket must leave behind: `max(r_min, frac * wall)`, the stricter.

    `wall_mm` is the wall **at the position being cut**, in millimetres. Passing
    the file's `t_b` instead would compare two different rulers — on this corpus
    that number reaches 205 mm (a whole part span) while `wall_remaining` in the
    build is measured from B's `near_b[1]`, so a 20 % floor of it would reject
    every repair for a reason that has nothing to do with wall thinning.
    """
    return max(float(r_min_mm), float(frac) * max(0.0, float(wall_mm)))


def override_depth_cap(
    edge: dict,
    *,
    wall_mm: float,
    unit_mm: float,
    source: str = "local_single_wall",
) -> dict:
    """A **copy** of `edge` whose depth cap comes from the measured wall.

    Both keys the two readers use are set, because they read different ones:
    `pah.policy.repair_script` prefers `edge["d_cap"]` for its closed-form gate,
    and `connector.depth_cap` prefers `edge["depth_max"]` inside the build. Setting
    only one would leave the planner and the builder on different caps — the same
    class of bug as measuring one position and drilling another.

    The input dict is never written to: `edges_v2.json` and the M0/M1 ledgers stay
    exactly as they are, which is what makes this a re-runnable experiment rather
    than a data edit.
    """
    cap_units = depth_cap_mm(wall_mm) / float(unit_mm)
    out = dict(edge)
    out["depth_max"] = cap_units
    out["d_cap"] = cap_units
    out["dcap_source"] = source
    out["dcap_wall_mm"] = round(float(wall_mm), 6)
    return out


def thin_wall_veto(
    result: dict,
    *,
    unit_mm: float,
    wall_mm: float | None,
    r_min_mm: float = WALL_FRAC_FLOOR_MM,
    frac: float = WALL_FRAC,
) -> dict | None:
    """Turn a build that ate the wall into an honest failure, or `None` to accept it.

    `connector.build` with `clamp=True` does **not** test the residual wall
    (`:855` is guarded by `and not clamp`), so a successful build can sit on — or
    under — the floor. This is the check the experiment owes: `wall_remaining` is
    compared against `residual_wall_need_mm`, taking the **thinner** of the wall
    measured at the anchor and the wall the build actually found at the drilled
    position (a relocation may move onto a thinner spot; the stricter number is
    the one that matters).

    The returned dict is shaped like a `connector.build` failure so the ladder can
    consume it unchanged: `classify` turns it into `degenerate_thin_wall`, which is
    not a terminal code, so the next rung still gets its chance.
    """
    if not result.get("ok"):
        return None
    remain = result.get("wall_remaining")
    if remain is None:
        return None
    build_wall = result.get("t_b")
    reference = None
    if wall_mm is not None:
        reference = float(wall_mm)
    if build_wall is not None:
        build_wall_mm = float(build_wall) * float(unit_mm)
        reference = build_wall_mm if reference is None else min(reference, build_wall_mm)
    need = residual_wall_need_mm(
        reference if reference is not None else 0.0, r_min_mm=r_min_mm, frac=frac
    )
    remain_mm = float(remain) * float(unit_mm)
    if remain_mm >= need - 1e-9:
        return None
    veto = {
        "ok": False,
        "reason": "degenerate_geometry",
        "degen_code": "degenerate_thin_wall",
        "detail": (
            "degenerate_thin_wall(wall_guard("
            f"wall_remaining={remain_mm:.4f}mm<need={need:.4f}mm,"
            f"t_b_local={(build_wall or 0.0) * float(unit_mm):.4f}mm"
            f",t_wall_anchor={(wall_mm if wall_mm is not None else float('nan')):.4f}mm))"
        ),
        "wall_remaining": float(remain),
        "wall_need_mm": float(need),
    }
    for key in ("u", "v", "r", "gap", "t_a", "t_b", "penetration"):
        if key in result:
            veto[key] = result[key]
    return veto
