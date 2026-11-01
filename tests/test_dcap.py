"""Tests for the local single-wall measurement and the depth cap built on it.

No benchmark geometry: the wall is measured on small synthetic solids built with
`trimesh.create_box`, so a hollow shell, a solid block, a three-layer stack and a
patch that the ray misses are all one constructor away. The only external
dependency is `assembly_rl.connector` (`line_intervals` / `plane_basis`), which is
the code under test's own ray engine — the module is a *reader* of the source, and
faking its ray casts would test the fake.

What is pinned here, in order of how expensive the bug would be:

* a hollow shell measures its **first layer**, not the span across the cavity;
* a solid block measures its whole thickness, and a three-layer stack still
  reports one layer (`n_layers` says how many were there);
* an unmeasurable anchor returns `None`, never a wall of zero;
* the override touches only the dict it is given, and sets both keys the two
  readers (`repair_script` and `connector`) actually look at;
* the residual-wall floor is the stricter of the absolute and the relative rule,
  and a build under it is turned into `degenerate_thin_wall`;
* with the floor off, `repair_script` plans exactly what M1 planned (that is the
  `runs/m1-min/rows.jsonl` regression, at unit scale).
"""

from __future__ import annotations

import copy

import pytest
import trimesh

from pah.policy.dcap import (
    DEPTH_MAX_RATIO,
    WALL_FRAC,
    depth_cap_mm,
    local_single_wall_mm,
    override_depth_cap,
    probe_local_wall,
    residual_wall_need_mm,
    thin_wall_veto,
)

#: `unit_mm = 1500` is the corpus scale (225 mm frame on 150 mm parts). The
#: synthetic meshes below are built in *millimetres* and measured with
#: `unit_mm=1.0`, so the numbers in the assertions are the millimetres they look
#: like; the scale only has to cancel, which is what `test_wall_scales_with_unit_mm`
#: checks.
UNIT_MM = 1500.0


# --------------------------------------------------------------------------- #
# synthetic solids
# --------------------------------------------------------------------------- #
def box(lo, hi) -> trimesh.Trimesh:
    """An axis-aligned box from two corners, in millimetres."""
    extents = [hi[i] - lo[i] for i in range(3)]
    mesh = trimesh.creation.box(extents=extents)
    mesh.apply_translation([(lo[i] + hi[i]) / 2.0 for i in range(3)])
    return mesh


def shell_plate(thickness: float = 1.0, cavity: float = 2.0, size: float = 20.0,
                layers: int = 2) -> trimesh.Trimesh:
    """A stack of `layers` plates along z, each `thickness` thick, `cavity` apart.

    Built by unioning disjoint slabs (a cavity is empty space between them, so the
    union is a valid, closed, multi-shell solid — no boolean required).
    """
    parts = []
    z = 0.0
    for _ in range(layers):
        parts.append(box((-size / 2, -size / 2, z), (size / 2, size / 2, z + thickness)))
        z += thickness + cavity
    return trimesh.util.concatenate(parts)


def empty_part() -> trimesh.Trimesh:
    """A part that is nowhere near the anchor: every ray misses it."""
    return box((500.0, 500.0, 500.0), (510.0, 510.0, 510.0))


#: The normal points A -> B, i.e. +z; the anchor sits on the interface at z = 0.
ANCHOR = (0.0, 0.0, 0.0)
NORMAL = (0.0, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# 1. single layer, not the span
# --------------------------------------------------------------------------- #
def test_hollow_shell_measures_the_first_layer_and_not_the_span():
    """The mistake this module exists to correct: a socket is a blind hole.

    Two 1 mm plates 2 mm apart: the first layer is 1 mm, the span across the
    cavity is 4 mm. A cap taken from the span would drill straight through.
    """
    part_b = shell_plate(thickness=1.0, cavity=2.0, layers=2)
    probe = probe_local_wall(None, part_b, ANCHOR, NORMAL, unit_mm=1.0, sample_radius=0.5)
    assert probe.wall_mm == pytest.approx(1.0, abs=1e-6)
    assert probe.span_mm == pytest.approx(4.0, abs=1e-6)
    assert probe.cavity_after_mm == pytest.approx(2.0, abs=1e-6)
    assert probe.n_layers == 2
    assert local_single_wall_mm(None, part_b, ANCHOR, NORMAL, unit_mm=1.0,
                                sample_radius=0.5) == pytest.approx(1.0, abs=1e-6)


def test_three_layers_are_never_summed():
    part_b = shell_plate(thickness=1.0, cavity=1.0, layers=3)
    probe = probe_local_wall(None, part_b, ANCHOR, NORMAL, unit_mm=1.0, sample_radius=0.5)
    assert probe.n_layers == 3
    assert probe.wall_mm == pytest.approx(1.0, abs=1e-6)  # not 3.0
    assert probe.span_mm == pytest.approx(5.0, abs=1e-6)


def test_solid_region_reports_its_full_thickness():
    """No cavity here, so the layer *is* the part: the cap must not be clipped."""
    part_b = box((-10.0, -10.0, 0.0), (10.0, 10.0, 4.0))
    probe = probe_local_wall(None, part_b, ANCHOR, NORMAL, unit_mm=1.0, sample_radius=0.5)
    assert probe.n_layers == 1
    assert probe.wall_mm == pytest.approx(4.0, abs=1e-6)
    assert probe.cavity_after_mm is None


def test_material_ahead_only_counts_from_the_interface():
    """A slab that starts *behind* the anchor is B's far side, not the socket wall."""
    behind = box((-10.0, -10.0, -6.0), (10.0, 10.0, -2.0))
    ahead = box((-10.0, -10.0, 0.5), (10.0, 10.0, 1.5))
    part_b = trimesh.util.concatenate([behind, ahead])
    probe = probe_local_wall(None, part_b, ANCHOR, NORMAL, unit_mm=1.0, sample_radius=0.5)
    assert probe.wall_mm == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# 2. unmeasurable is None, never zero
# --------------------------------------------------------------------------- #
def test_missing_material_returns_none():
    probe = probe_local_wall(None, empty_part(), ANCHOR, NORMAL, unit_mm=1.0)
    assert probe.wall_mm is None
    assert probe.measured is False
    assert "no_material_hit_b" in probe.detail
    assert local_single_wall_mm(None, empty_part(), ANCHOR, NORMAL) is None


def test_a_patch_edge_ray_is_not_a_wall(monkeypatch):
    """Only the centre ray hits the part → the median would describe one point.

    The brief's "多点采样取中位数" is a robustness measure, not a way to average
    over a patch that has no material around the drill point, so the answer is
    `None` and the reason says why.
    """
    part_b = box((-1.0, -1.0, 0.0), (1.0, 1.0, 1.0))  # 2 mm across
    probe = probe_local_wall(
        None, part_b, ANCHOR, NORMAL, unit_mm=1.0, sample_radius=3.0, rays=5
    )
    assert probe.centre_ok is True
    assert probe.n_valid == 1
    assert probe.wall_mm is None
    assert "too_few_rays" in probe.detail


def test_a_zero_normal_is_refused():
    probe = probe_local_wall(None, box((-1, -1, 0), (1, 1, 1)), ANCHOR, (0.0, 0.0, 0.0))
    assert probe.wall_mm is None
    assert probe.detail == "zero_normal"


def test_penetrating_host_is_refused():
    """A material already past B's near face: B's first interval is not a free wall."""
    part_a = box((-10.0, -10.0, 0.5), (10.0, 10.0, 3.0))     # A overlaps B's slab
    part_b = box((-10.0, -10.0, 0.0), (10.0, 10.0, 4.0))
    probe = probe_local_wall(part_a, part_b, ANCHOR, NORMAL, unit_mm=1.0, sample_radius=0.5)
    assert probe.wall_mm is None
    assert "penetrated" in probe.detail


def test_wall_scales_with_unit_mm():
    """Meshes in data units: the same solid at `unit_mm=1500` is 1500× the millimetres."""
    part_b = box((-10.0, -10.0, 0.0), (10.0, 10.0, 4.0))
    assert local_single_wall_mm(None, part_b, ANCHOR, NORMAL, unit_mm=1500.0,
                                sample_radius=0.5) == pytest.approx(4.0 * 1500.0, rel=1e-9)


# --------------------------------------------------------------------------- #
# 3. the cap and the floor
# --------------------------------------------------------------------------- #
def test_depth_cap_keeps_the_sources_ratio():
    assert DEPTH_MAX_RATIO == 0.8
    assert depth_cap_mm(2.0) == pytest.approx(1.6)


def test_residual_wall_floor_takes_the_stricter_rule():
    # thin wall → the absolute 0.25 mm floor wins
    assert residual_wall_need_mm(1.0) == pytest.approx(0.25)
    # thick wall → the 20 % rule wins
    assert residual_wall_need_mm(4.0) == pytest.approx(WALL_FRAC * 4.0)
    # the switch-over is at 0.25 / 0.2 = 1.25 mm
    assert residual_wall_need_mm(1.25) == pytest.approx(0.25)
    assert residual_wall_need_mm(1.26) == pytest.approx(0.252)


# --------------------------------------------------------------------------- #
# 4. the override is local to the dict it is given
# --------------------------------------------------------------------------- #
def edge_dict(**overrides) -> dict:
    edge = {
        "edge_id": "asm_000000:7",
        "code": "gap_too_wide",
        "center": [0.0, 0.0, 0.0],
        "anchor": [0.0, 0.0, 0.0],
        "n_ab": [0.0, 0.0, 1.0],
        "half_w": 6.0 / UNIT_MM,
        "half_h": 2.0 / UNIT_MM,
        "r_inplane_max": 1.2 / UNIT_MM,
        "depth_max": 0.6 / UNIT_MM,
        "t_a": 0.7 / UNIT_MM,
        "t_b": 3.0 / UNIT_MM,
        "r_cap": 1.2 / UNIT_MM,
        "d_cap": 0.6 / UNIT_MM,
    }
    edge.update(overrides)
    return edge


def test_override_returns_a_copy_and_sets_both_keys():
    edge = edge_dict()
    before = copy.deepcopy(edge)
    out = override_depth_cap(edge, wall_mm=2.0, unit_mm=UNIT_MM)
    assert edge == before                      # the caller's dict is untouched
    assert out is not edge
    # both readers agree: `repair_script` reads d_cap, `connector.depth_cap` depth_max
    assert out["d_cap"] == out["depth_max"]
    assert out["d_cap"] == pytest.approx(depth_cap_mm(2.0) / UNIT_MM)
    assert out["dcap_source"] == "local_single_wall"
    assert out["dcap_wall_mm"] == pytest.approx(2.0)


def test_override_does_not_leak_into_another_edge():
    first = edge_dict(edge_id="a:1")
    second = edge_dict(edge_id="a:2")
    override_depth_cap(first, wall_mm=1.0, unit_mm=UNIT_MM)
    assert second["d_cap"] == pytest.approx(0.6 / UNIT_MM)
    assert second["depth_max"] == pytest.approx(0.6 / UNIT_MM)
    assert "dcap_source" not in second


def test_the_build_will_actually_read_the_override():
    """`connector.depth_cap` prefers `edge["depth_max"]` — the key has to be there."""
    connector = pytest.importorskip("pah.tools.verifiers.connector_feasible")._connector()
    edge = edge_dict()
    out = override_depth_cap(edge, wall_mm=2.0, unit_mm=UNIT_MM)
    assert connector.depth_cap(edge) == pytest.approx(0.6 / UNIT_MM)
    assert connector.depth_cap(out) == pytest.approx(depth_cap_mm(2.0) / UNIT_MM)


# --------------------------------------------------------------------------- #
# 5. the residual-wall veto
# --------------------------------------------------------------------------- #
def build_ok(wall_remaining_mm: float, t_b_mm: float = 1.8) -> dict:
    return {
        "ok": True,
        "reason": "",
        "detail": "",
        "wall_remaining": wall_remaining_mm / UNIT_MM,
        "t_b": t_b_mm / UNIT_MM,
        "r": 0.6 / UNIT_MM,
        "gap": 0.3 / UNIT_MM,
    }


def test_a_build_that_leaves_the_floor_alone_is_accepted():
    # floor = max(0.25, 0.2*1.8) = 0.36 mm; 0.40 mm is above it
    assert thin_wall_veto(build_ok(0.40), unit_mm=UNIT_MM, wall_mm=1.8) is None


def test_a_build_that_eats_the_wall_becomes_degenerate_thin_wall():
    classify = pytest.importorskip(
        "pah.tools.verifiers.connector_feasible"
    ).classify
    veto = thin_wall_veto(build_ok(0.16), unit_mm=UNIT_MM, wall_mm=1.8)
    assert veto is not None and veto["ok"] is False
    assert veto["degen_code"] == "degenerate_thin_wall"
    assert veto["wall_need_mm"] == pytest.approx(0.36)
    # the ladder consumes this dict like any other failure, and the code is not
    # terminal, so the next rung still gets its chance
    assert classify(veto)[0] == "degenerate_thin_wall"


def test_the_veto_uses_the_thinner_of_the_two_walls():
    """A relocation can move onto a thinner wall than the anchor's; the stricter wins."""
    result = build_ok(0.30, t_b_mm=3.0)          # floor at t_b=3.0 is 0.60 mm
    assert thin_wall_veto(result, unit_mm=UNIT_MM, wall_mm=4.0) is not None
    assert thin_wall_veto(result, unit_mm=UNIT_MM, wall_mm=None) is not None
    # the anchor's 1.0 mm wall is thinner than the build's 3.0 mm one, so the floor
    # drops to 0.25 mm and the same 0.30 mm becomes acceptable
    assert thin_wall_veto(result, unit_mm=UNIT_MM, wall_mm=1.0) is None


def test_a_failed_build_is_not_vetoed_twice():
    failed = {"ok": False, "reason": "degenerate_geometry",
              "detail": "degenerate_other(gap_too_wide_for_wall(gap=1mm))"}
    assert thin_wall_veto(failed, unit_mm=UNIT_MM, wall_mm=1.8) is None


def test_a_build_without_a_wall_reading_is_left_alone():
    assert thin_wall_veto({"ok": True}, unit_mm=UNIT_MM, wall_mm=1.8) is None


# --------------------------------------------------------------------------- #
# 6. the floor-off path is the M1 path
# --------------------------------------------------------------------------- #
def test_floor_off_plans_the_same_radius_as_the_three_bound_policy():
    from pah.policy.repair_script import closed_form_radius

    gap, t_b, d_cap, r_min, r_cap = 0.3, 1.8, 1.6, 0.25, 0.8
    off = closed_form_radius(gap, t_b, d_cap, r_min, r_cap)
    expected = min(r_cap, 0.98 * t_b / 2.25, (d_cap - gap) * 0.98)
    assert off == pytest.approx(expected)
    # an explicit zero is the same as the default: the bound is only added when on
    assert closed_form_radius(gap, t_b, d_cap, r_min, r_cap, wall_need=0.0) == off


def test_floor_on_shrinks_the_radius_to_keep_the_wall():
    from pah.policy.repair_script import closed_form_radius

    gap, t_b, d_cap, r_min, r_cap = 0.3, 1.8, 1.6, 0.25, 0.8
    need = residual_wall_need_mm(1.8)           # 0.36 mm
    on = closed_form_radius(gap, t_b, d_cap, r_min, r_cap, wall_need=need)
    assert on == pytest.approx((t_b - need) / 2.25)
    assert on < closed_form_radius(gap, t_b, d_cap, r_min, r_cap)


def test_floor_off_depth_is_the_two_to_one_pin():
    from pah.policy.repair_script import _State

    state = _State(edge={}, unit_mm=UNIT_MM, r_min=0.25, r=0.6, depth=0.0)
    assert state.depth_for(0.6, gap=0.3, t_b=1.8, d_cap=1.6) == pytest.approx(1.2)
    assert state.depth_for(0.6, gap=0.3, t_b=1.8, d_cap=0.9) == pytest.approx(0.9)


def test_floor_on_shortens_the_requested_engagement():
    from pah.policy.repair_script import _State

    state = _State(edge={}, unit_mm=UNIT_MM, r_min=0.25, r=0.6, depth=0.0,
                   wall_need_abs=0.25, wall_need_frac=WALL_FRAC)
    depth = state.depth_for(0.6, gap=0.3, t_b=1.8, d_cap=1.6)
    floor = residual_wall_need_mm(1.8)
    # t_b + gap - 1.25 r - floor, and never below the gate's `need = gap + r`
    assert depth == pytest.approx(min(1.2, 1.8 + 0.3 - 0.75 - floor))
    assert depth >= 0.3 + 0.6 - 1e-12


def test_floor_off_params_carry_no_floor_keys():
    """The regression hinge: an M1-shaped `params` dict must not grow a new key."""
    from pah.policy.repair_script import _State

    state = _State(edge={}, unit_mm=UNIT_MM, r_min=0.25, r=0.6, depth=0.0)
    assert "wall_need_mm" not in state.params("resize_connector")
    assert state.wall_floor_on is False
    assert _State(edge={}, unit_mm=UNIT_MM, r_min=0.25, r=0.6, depth=0.0,
                  wall_need_frac=0.2).wall_floor_on is True


# --------------------------------------------------------------------------- #
# 7. end to end on the real connector: the measurement equals the build's t_b
# --------------------------------------------------------------------------- #
def test_measured_wall_matches_the_builds_local_t_b():
    """Self-calibration: the source measures `t_b` on the same ray, so they agree.

    This is the check that keeps the measurement from drifting away from the
    number the gate will use: `probe_local_wall` reads B's first layer, and
    `connector.measure_axis` reports exactly that as `t_b`.
    """
    connector = pytest.importorskip("pah.tools.verifiers.connector_feasible")._connector()
    part_b = shell_plate(thickness=1.0, cavity=2.0, layers=2)
    part_a = box((-10.0, -10.0, -2.0), (10.0, 10.0, 0.0))
    edge = {"center": [0.0, 0.0, 0.0], "anchor": [0.0, 0.0, 0.0], "n_ab": [0.0, 0.0, 1.0]}
    reading = connector.measure_axis(part_a, part_b, edge, None)
    probe = probe_local_wall(part_a, part_b, ANCHOR, NORMAL, unit_mm=1.0, sample_radius=0.5)
    assert reading["ok"] is True
    assert probe.wall_mm == pytest.approx(reading["t_b"], abs=1e-6)
