"""Tests for the M1 repair script — no geometry, no GPU, no source repo.

The two ports (`build`, `measure`) are the only way this module touches the
world, so the tests drive them with small deterministic stubs. That keeps the
four properties the M1 brief demands testable in milliseconds:

* the same input twice gives the same attempt sequence;
* `AttemptBudget` stops the ladder;
* one success stops it too, before any more expensive rung;
* a failure keeps the source's own reason code rather than a merged "failed".
"""

from __future__ import annotations

import copy

import pytest

from pah.policy.repair_script import (
    MAX_ATTEMPTS,
    AssemblyParts,
    AttemptBudget,
    RepairAttempt,
    closed_form_radius,
    gate_margin,
    next_operator,
    repair_edge,
    resize_rungs,
)

#: 0.25 mm at unit_mm = 1500 — the print floor in data units, as the pipeline
#: computes it (`connector_feasible.R_MIN_MM / unit_mm`).
UNIT_MM = 1500.0
R_MIN = 0.25 / UNIT_MM

#: `gap_too_wide_for_wall` as the source spells it, with numbers a real row
#: carries (`asm_000000:7`).
GAP_DETAIL = (
    "degenerate_other(gap_too_wide_for_wall(gap=0.8963mm,need=1.2032mm"
    ">eng_max=0.6139mm))"
)
NO_MATERIAL_DETAIL = "degenerate_other(no_material_hit(a=0,b=1))"


def edge_of(code: str, **overrides) -> dict:
    """One `edges_v2` row as the repair stage receives it, in data units."""
    edge = {
        "edge_id": "asm_000000:7",
        "a_index": 1,
        "b_index": 6,
        "code": code,
        "center": [0.0, 0.0, 0.0],
        "anchor": [0.001, 0.002, 0.003],
        "n_ab": [0.0, 0.0, 1.0],
        "half_w": 2.0 / UNIT_MM,
        "half_h": 1.5 / UNIT_MM,
        "r_inplane_max": 1.2 / UNIT_MM,
        "depth_max": 1.6 / UNIT_MM,
        "r_cap": 1.2 / UNIT_MM,
        "d_cap": 1.6 / UNIT_MM,
        "t_a": 30.0 / UNIT_MM,
        "t_b": 1.8 / UNIT_MM,
    }
    edge.update(overrides)
    return edge


class Ports:
    """A `build` / `measure` pair that replays a fixed script of verdicts.

    `reading` may be a dict (the same ray result everywhere) or a callable
    `(edge, theta) -> dict` when a test needs the measurement to depend on where
    the anchor is — which is exactly what the relocation search reacts to.
    """

    def __init__(self, results: list[dict], reading: dict | Callable | None = None) -> None:
        self.results = list(results)
        self.reading = reading or {"ok": True, "gap": 0.5 / UNIT_MM,
                                   "t_b": 1.8 / UNIT_MM, "t_a": 30.0 / UNIT_MM}
        self.builds: list[tuple[dict, tuple, str]] = []
        self.measures: list[tuple] = []

    def measure(self, assembly, edge, theta):
        self.measures.append((dict(edge), tuple(theta)))
        if callable(self.reading):
            return self.reading(edge, theta)
        return dict(self.reading)

    def build(self, assembly, edge, theta, *, section):
        self.builds.append((dict(edge), tuple(theta), section))
        if not self.results:
            raise AssertionError("build called more often than the script allows")
        return dict(self.results.pop(0))


@pytest.fixture()
def assembly() -> AssemblyParts:
    return AssemblyParts(case_id="asm_000000", unit_mm=UNIT_MM, meshes=(None,) * 8, n_parts=8)


def fail_gap() -> dict:
    return {"ok": False, "reason": "degenerate_geometry", "detail": GAP_DETAIL,
            "gap": 0.8963 / UNIT_MM, "t_b": 1.0 / UNIT_MM}


def fail_material() -> dict:
    return {"ok": False, "reason": "degenerate_geometry", "detail": NO_MATERIAL_DETAIL}


def ok_build(r_mm: float = 0.4) -> dict:
    return {"ok": True, "reason": "", "detail": "", "r": r_mm / UNIT_MM,
            "depth": 2 * r_mm / UNIT_MM, "wall_remaining": 0.4 / UNIT_MM,
            "dv_a": 1e-11, "dv_b": -1e-11}


# --------------------------------------------------------------------------- #
# The closed-form gate, which is the whole reason the radius ladder works
# --------------------------------------------------------------------------- #
def test_gate_is_closed_exactly_at_the_wall_cap():
    """`r == t_b / 2.25` makes `eng_max == need`, and `<=` then rejects it.

    This is the mechanism behind most of the 166 `gap_too_wide` rows: the clamp
    lands the radius on the equality, so the edge fails however small the gap.
    """
    t_b, gap = 1.8, 0.3
    r_cap = t_b / 2.25
    passes, slack = gate_margin(gap, t_b, d_cap=1.6, r=r_cap)
    assert passes is False and slack == pytest.approx(0.0, abs=1e-12)
    # strictly below the cap there is slack again
    assert gate_margin(gap, t_b, d_cap=1.6, r=r_cap * 0.98)[0] is True


def test_gap_wider_than_the_depth_cap_is_not_fixable_by_radius():
    t_b, gap, d_cap = 1.8, 0.9, 0.61
    for r in resize_rungs(R_MIN, 1.2 / UNIT_MM):
        assert gate_margin(gap, t_b, d_cap, r)[0] is False


def test_closed_form_radius_respects_all_three_caps():
    r_min = 0.1
    r = closed_form_radius(gap=0.2, t_b=1.8, d_cap=1.6, r_min=r_min, r_cap=0.9)
    assert r <= 0.9
    assert r <= 0.98 * 1.8 / 2.25 + 1e-12
    assert r <= (1.6 - 0.2) * 0.98 + 1e-12
    assert r >= r_min


def test_closed_form_radius_never_returns_less_than_the_print_floor():
    assert closed_form_radius(2.0, 0.1, 0.1, r_min=0.25, r_cap=0.3) == 0.25


def test_resize_rungs_are_five_descending_geometric_steps():
    rungs = resize_rungs(0.2, 1.0)
    assert len(rungs) == 5
    assert rungs[0] == pytest.approx(1.0)
    assert rungs[-1] == pytest.approx(0.2)
    assert all(a > b for a, b in zip(rungs, rungs[1:]))


def test_resize_rungs_collapse_when_there_is_no_room():
    assert resize_rungs(0.4, 0.4) == (0.4,)
    assert resize_rungs(0.4, 0.1) == (0.4,)


# --------------------------------------------------------------------------- #
# Ladder order (pah-dev-doc-v1.md §5.3)
# --------------------------------------------------------------------------- #
def test_ladder_follows_the_rule_table_for_each_code():
    assert next_operator("gap_too_wide", (), gate_failed=True) == "resize_connector"
    assert next_operator("no_material_hit", (), gate_failed=True) == "relocate_connector"
    # still stuck at the gate -> walk the table left to right
    assert next_operator("gap_too_wide", ("resize_connector",), gate_failed=True) == (
        "relocate_connector"
    )
    # past the gate and failing on a volume check -> the in-plane search cannot
    # help, so the family swap is next
    assert next_operator("gap_too_wide", ("resize_connector",), gate_failed=False) == (
        "swap_family"
    )
    assert next_operator("gap_too_wide", ("resize_connector", "swap_family"),
                         gate_failed=True) == "relocate_connector"
    assert next_operator("gap_too_wide", ("resize_connector", "relocate_connector",
                                          "swap_family"), gate_failed=True) is None


# --------------------------------------------------------------------------- #
# 1. determinism
# --------------------------------------------------------------------------- #
def _sequence(report) -> list[tuple]:
    return [(a.operator, tuple(sorted(a.params.items())), a.success, a.reason)
            for a in report.attempts]


def test_same_input_twice_gives_the_same_attempt_sequence(assembly):
    def run() -> list[tuple]:
        ports = Ports([fail_gap(), fail_material(), fail_gap()])
        report = repair_edge(
            assembly, edge_of("gap_too_wide"),
            budget=AttemptBudget(max_attempts=MAX_ATTEMPTS),
            build=ports.build, measure=ports.measure,
        )
        return _sequence(report)

    first, second = run(), run()
    assert first == second
    # resize then relocate; the family swap is ruled out because neither attempt
    # ever reached the boolean stage
    assert [row[0] for row in first] == ["resize_connector", "relocate_connector"]


def test_keys_the_ladder_does_not_read_cannot_change_the_sequence(assembly):
    """Determinism is about the decision inputs, not about dict identity."""
    ports = Ports([fail_gap(), fail_material(), fail_gap()])
    report = repair_edge(assembly, edge_of("gap_too_wide"), budget=AttemptBudget(),
                         build=ports.build, measure=ports.measure)
    again = Ports([fail_gap(), fail_material(), fail_gap()])
    repeat = repair_edge(assembly, copy.deepcopy(edge_of("gap_too_wide")),
                         budget=AttemptBudget(), build=again.build, measure=again.measure)
    assert _sequence(report) == _sequence(repeat)


# --------------------------------------------------------------------------- #
# 2. budget
# --------------------------------------------------------------------------- #
def test_budget_stops_the_ladder(assembly):
    ports = Ports([fail_gap()] * 5)
    report = repair_edge(assembly, edge_of("gap_too_wide"),
                         budget=AttemptBudget(max_attempts=2),
                         build=ports.build, measure=ports.measure)
    assert report.n_attempts == 2
    assert report.build_calls == 2
    assert report.stopped == "attempts"
    assert report.exhausted is True
    assert len(ports.results) == 3  # three verdicts were never spent


def test_max_attempts_cannot_be_raised_above_the_brief_cap(assembly):
    assert AttemptBudget(max_attempts=99).attempts_cap() == MAX_ATTEMPTS
    post_gate = {"ok": False, "reason": "socket_breaks_wall",
                 "detail": "socket_no_effect(dv_b=0.0,expected=1.0)"}
    ports = Ports([dict(post_gate)] * 9)
    report = repair_edge(assembly, edge_of("gap_too_wide"),
                         budget=AttemptBudget(max_attempts=99),
                         build=ports.build, measure=ports.measure)
    # past the gate → resize, swap, relocate; a fourth rung does not exist, and a
    # caller asking for 99 attempts still gets no more than three
    assert report.n_attempts == MAX_ATTEMPTS
    assert report.build_calls == MAX_ATTEMPTS
    assert report.stopped == "rungs"


def test_deadline_is_reported_as_exhausted(assembly):
    ports = Ports([fail_gap()] * 3)
    report = repair_edge(assembly, edge_of("gap_too_wide"),
                         budget=AttemptBudget(deadline=0.0),
                         build=ports.build, measure=ports.measure)
    assert report.n_attempts == 0
    assert report.stopped == "deadline"
    assert report.exhausted is True


# --------------------------------------------------------------------------- #
# 3. one success stops the ladder
# --------------------------------------------------------------------------- #
def test_success_on_the_first_rung_stops_before_the_expensive_ones(assembly):
    ports = Ports([ok_build()])
    report = repair_edge(assembly, edge_of("gap_too_wide"), budget=AttemptBudget(),
                         build=ports.build, measure=ports.measure)
    assert report.success is True
    assert report.n_attempts == 1
    assert report.operators == ("resize_connector",)
    assert report.build_calls == 1
    assert report.reason is None
    assert report.stopped == "success"
    assert len(ports.results) == 0


def test_second_rung_success_keeps_the_first_failure_on_record(assembly):
    ports = Ports([fail_gap(), ok_build()])
    report = repair_edge(assembly, edge_of("gap_too_wide"), budget=AttemptBudget(),
                         build=ports.build, measure=ports.measure)
    assert report.success is True
    assert report.n_attempts == 2
    assert [a.success for a in report.attempts] == [False, True]
    assert report.attempts[0].reason == "gap_too_wide"


def test_no_material_hit_starts_with_the_anchor_search(assembly):
    ports = Ports([{"ok": False, "reason": "degenerate_geometry",
                    "detail": NO_MATERIAL_DETAIL}] * 3)
    report = repair_edge(assembly, edge_of("no_material_hit"), budget=AttemptBudget(),
                         build=ports.build, measure=ports.measure)
    assert report.operators[0] == "relocate_connector"
    assert report.reason == "no_material_hit"
    # the search starts at the recorded anchor and moves only if it has to
    assert ports.measures[0][1][0] == 0.0 and ports.measures[0][1][1] == 0.0


def test_impossible_rungs_are_skipped_rather_than_spending_attempts(assembly):
    """No material anywhere in the patch → neither a radius nor a position helps.

    The grid is still searched (that is the evidence), but the two rungs it
    rules out are not spent as build attempts; only the family swap is left.
    """

    def reading(edge, theta):
        return {"ok": False, "reason": "degenerate_geometry",
                "detail": NO_MATERIAL_DETAIL}

    ports = Ports([{"ok": False, "reason": "degenerate_geometry",
                    "detail": NO_MATERIAL_DETAIL}], reading=reading)
    report = repair_edge(assembly, edge_of("no_material_hit"), budget=AttemptBudget(),
                         build=ports.build, measure=ports.measure)
    assert report.operators == ()  # nothing was worth a build call
    assert len(ports.measures) == 26  # 25 grid positions + 1 resize probe
    assert report.stopped == "no_plan"
    assert report.reason == "no_material_hit"  # the code it came in with


# --------------------------------------------------------------------------- #
# 4. failures keep the source's reason code
# --------------------------------------------------------------------------- #
def test_failure_keeps_the_promoted_code_and_the_raw_reason(assembly):
    ports = Ports([fail_gap()] * 3)
    report = repair_edge(assembly, edge_of("gap_too_wide"),
                         budget=AttemptBudget(max_attempts=1),
                         build=ports.build, measure=ports.measure)
    attempt = report.attempts[0]
    assert attempt.reason == "gap_too_wide"          # the 13-code vocabulary
    assert attempt.raw_reason == "degenerate_geometry"  # what the source said
    assert "gap_too_wide_for_wall" in attempt.detail
    assert report.reason == "gap_too_wide"
    assert report.exhausted is True


def test_terminal_codes_stop_the_ladder_without_burning_attempts(assembly):
    ports = Ports([{"ok": False, "reason": "unreatable_too_small",
                    "detail": "r_max=0.1mm(vs r_min 0.25)"}])
    report = repair_edge(assembly, edge_of("gap_too_wide"), budget=AttemptBudget(),
                         build=ports.build, measure=ports.measure)
    assert report.n_attempts == 1
    assert report.stopped == "terminal"
    assert report.exhausted is False
    assert report.reason == "unreatable_too_small"


# --------------------------------------------------------------------------- #
# Side conditions the brief asks for
# --------------------------------------------------------------------------- #
def test_input_edge_is_never_mutated(assembly):
    edge = edge_of("gap_too_wide")
    before = copy.deepcopy(edge)
    ports = Ports([fail_gap(), ok_build()])
    repair_edge(assembly, edge, budget=AttemptBudget(),
                build=ports.build, measure=ports.measure)
    assert edge == before


def test_plane_basis_matches_the_source_construction():
    """`connector.plane_basis` (`connector.py:138-145`) reproduced exactly.

    The relocation operator converts `(u, v)` to a world point with this basis, so
    a rotation between this copy and the source's would measure one position and
    build another. The first M1 pass shipped that bug and lost every
    `no_material_hit` repair to it, which is why the numbers are pinned here.
    """
    from pah.policy.repair_script import _plane_basis  # noqa: PLC0415

    # n = +Z: ref is (1,0,0) because |n_z| >= 0.9 -> e1 = -Y, e2 = +X
    e1, e2 = _plane_basis([0.0, 0.0, 1.0])
    assert e1 == pytest.approx([0.0, -1.0, 0.0], abs=1e-12)
    assert e2 == pytest.approx([1.0, 0.0, 0.0], abs=1e-12)
    # n = +X: ref is (0,0,1) -> e1 = +Y, e2 = +Z
    e1, e2 = _plane_basis([1.0, 0.0, 0.0])
    assert e1 == pytest.approx([0.0, 1.0, 0.0], abs=1e-12)
    assert e2 == pytest.approx([0.0, 0.0, 1.0], abs=1e-12)
    # unnormalised input must not change the frame
    assert _plane_basis([0.0, 0.0, 7.0])[0] == pytest.approx([0.0, -1.0, 0.0], abs=1e-12)


def test_relocation_moves_the_anchor_when_the_centre_has_no_material(assembly):
    """The centre ray misses one part; the search must move and say so."""

    def reading(edge, theta):
        u, v = theta[0], theta[1]
        if u == 0.0 and v == 0.0:
            return {"ok": False, "reason": "degenerate_geometry",
                    "detail": NO_MATERIAL_DETAIL}
        return {"ok": True, "gap": 0.3 / UNIT_MM, "t_b": 1.8 / UNIT_MM,
                "t_a": 30.0 / UNIT_MM}

    edge = edge_of("no_material_hit")
    before = copy.deepcopy(edge)
    ports = Ports([ok_build()], reading=reading)
    report = repair_edge(assembly, edge, budget=AttemptBudget(),
                         build=ports.build, measure=ports.measure)
    assert report.success is True
    built_edge, theta, _section = ports.builds[0]
    assert built_edge["anchor"] != edge["anchor"]
    assert theta[0] == 0.0 and theta[1] == 0.0  # the anchor now carries the offset
    assert edge == before  # the caller's dict is untouched
    assert report.final["anchor_shift_mm"] > 0.0


def test_attempt_records_are_frozen_and_carry_the_operator():
    attempt = RepairAttempt(edge_id="e", operator="resize_connector", params={},
                            success=False, reason="gap_too_wide", elapsed_s=0.1)
    with pytest.raises(Exception):
        attempt.success = True  # type: ignore[misc]
    assert attempt.reason == "gap_too_wide"
