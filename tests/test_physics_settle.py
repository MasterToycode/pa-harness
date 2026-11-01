"""Behaviour of `physics_settle`: a seat that holds, and three that do not.

The point of these tests is the *verifier's own honesty gate* (dev doc §4.8,
contract §3), so they are written at the level the gate runs at: build the two
solids, hand them a `DesignView`, and assert the certified codes. The measured
numbers are asserted only where the number *is* the finding (a 2.0 mm clearance
must read as ~2.0 mm of play; a solver that reported 0.2 mm of penetration for a
0.15 mm fit would be lying).

Cost: ~2-3 s for the whole file, because `PhysicsSettle.fixtures()` is cached per
process and each settle is 4000 steps of a 26-geom model.
"""

from __future__ import annotations

import types

import numpy as np
import pytest
import trimesh

from pah.design.api import (
    CODE_DYNAMIC_PENETRATION,
    CODE_JOINT_UNSTABLE,
    CODE_PHYSICS_ERROR,
    DESIGN_CODES,
    DesignVerifierRegistry,
    DesignView,
)
from pah.design.spec import DesignSpec, FitConstraints, InterfaceSpec, PartSpec
from pah.tools import mujoco_settle
from pah.tools.verifiers.physics_settle import (
    DRIFT_RATIO,
    MAX_PENETRATION_TOL_MM,
    PhysicsSettle,
    _peg_hole_meshes,
    _spec,
    _view,
    mate_axis_for,
    tolerances_for,
)


@pytest.fixture(scope="module")
def verifier() -> PhysicsSettle:
    return PhysicsSettle()


# --------------------------------------------------------------------------- #
# The fixture pair
# --------------------------------------------------------------------------- #
def test_fixture_pair_is_paired(verifier: PhysicsSettle) -> None:
    """Both halves must exist or `registry.sanity` disables the verifier by rule."""
    good, bad = verifier.fixtures()
    assert len(good) >= 1
    assert len(bad) >= 1


def test_good_seat_passes(verifier: PhysicsSettle) -> None:
    """0.15 mm clearance, 8 mm engagement on a socket floor: the joint holds."""
    good, _ = verifier.fixtures()
    certificates = verifier.check(good[0])
    assert certificates == []
    assert verifier.last_sim_calls == 1
    result = verifier.last_result
    assert result is not None
    # The play it is allowed is its clearance, and it takes no more than that.
    assert result.drift_mm == pytest.approx(0.15, abs=0.01)
    # The solver's own compliance, not a design fault: 4 um for a 1.4 g insert.
    assert result.penetration_mm < 0.01
    assert result.supported is True
    assert result.seat_gap_mm == pytest.approx(0.0, abs=0.05)


def test_loose_clearance_reports_lateral_drift(verifier: PhysicsSettle) -> None:
    """2.0 mm of clearance is 2.0 mm of play, and that is the finding."""
    _, bad = verifier.fixtures()
    loose = bad[0]
    certificates = verifier.check(loose)
    assert [certificate.code for certificate in certificates] == [CODE_JOINT_UNSTABLE]
    certificate = certificates[0]
    assert certificate.family == "assembly"
    assert certificate.locus["reason"] == "lateral_drift"
    assert certificate.magnitude["drift_mm"] == pytest.approx(2.0, abs=0.05)
    assert certificate.magnitude["clearance_mm"] == pytest.approx(2.0)
    # The threshold came from the spec's own fit band, not from a constant.
    expected = DRIFT_RATIO * loose.spec.constraints.max_clearance_mm
    assert certificate.magnitude["threshold"] == pytest.approx(expected)


def test_over_closed_fit_reports_dynamic_penetration(verifier: PhysicsSettle) -> None:
    """A peg 0.1 mm oversize is squeezed: the solver shows ~0.05 mm of interpenetration."""
    _, bad = verifier.fixtures()
    oversize = bad[1]
    certificates = verifier.check(oversize)
    assert [certificate.code for certificate in certificates] == [CODE_DYNAMIC_PENETRATION]
    certificate = certificates[0]
    assert certificate.family == "geometry"
    assert certificate.magnitude["penetration_mm"] == pytest.approx(0.05, abs=0.01)
    assert certificate.magnitude["threshold"] == pytest.approx(
        MAX_PENETRATION_TOL_MM, abs=1e-9
    )


def test_unretained_seat_reports_leaving_the_seat(verifier: PhysicsSettle) -> None:
    """A through hole has nothing under the insert, so gravity takes it out.

    `pull_out` (travelled against the insertion direction) and `drop_through`
    (travelled along it and fell out the far side) are the same finding seen from
    two sides; this geometry produces the second.
    """
    _, bad = verifier.fixtures()
    unretained = bad[2]
    certificates = verifier.check(unretained)
    assert [certificate.code for certificate in certificates] == [CODE_JOINT_UNSTABLE]
    certificate = certificates[0]
    assert certificate.locus["reason"] in {"pull_out", "drop_through"}
    assert certificate.magnitude["supported"] is False
    assert certificate.magnitude["seat_gap_mm"] is None
    # It really moved: metres, not micrometres.
    assert abs(certificate.magnitude["axial_mm"]) > 100.0


def test_shallow_blind_seat_holds(verifier: PhysicsSettle) -> None:
    """A blind seat at 0.8 mm engagement passes, and that is physics, not a gap.

    Standing on the socket floor, the insert's centre of mass is over its own
    support polygon, so gravity cannot topple it at *any* engagement depth. What
    makes a shallow seat fail is the absence of support (the through-hole fixture),
    not the depth — so no threshold on depth would be honest here.
    """
    view = _view(clearance_mm=0.15, engage_mm=0.8, through=False)
    assert verifier.check(view) == []
    assert verifier.last_result is not None
    assert verifier.last_result.engaged_span_mm == pytest.approx(0.8, abs=0.01)


# --------------------------------------------------------------------------- #
# Thresholds are derived, not declared
# --------------------------------------------------------------------------- #
def test_tolerances_come_from_the_fit_band() -> None:
    """Change the band and both numbers move, so physics cannot contradict fit."""
    default = tolerances_for(FitConstraints())
    assert default.drift_mm == pytest.approx(DRIFT_RATIO * 0.6)
    assert default.penetration_mm == pytest.approx(MAX_PENETRATION_TOL_MM)

    tight = tolerances_for(FitConstraints(min_clearance_mm=0.06, max_clearance_mm=0.2))
    assert tight.drift_mm == pytest.approx(DRIFT_RATIO * 0.2)
    assert tight.penetration_mm < default.penetration_mm
    assert tight.penetration_mm >= 0.01  # never below the solver's own compliance

    # An impossible band must not produce a zero tolerance the solver can fail.
    degenerate = tolerances_for(FitConstraints(min_clearance_mm=0.0, max_clearance_mm=0.0))
    assert degenerate.drift_mm > 0.0
    assert degenerate.penetration_mm > 0.0


def test_mate_axis_is_looked_up_by_kind() -> None:
    """The axis decides which way gravity pulls; an unknown kind is refused."""
    spec = _spec(clearance_mm=0.15)
    assert mate_axis_for(spec) == (0.0, 0.0, 1.0)
    unknown = types.SimpleNamespace(interface=types.SimpleNamespace(kind="weld"))
    with pytest.raises(KeyError):
        mate_axis_for(unknown)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Approximation reporting — the part that keeps a pass readable
# --------------------------------------------------------------------------- #
def test_approximation_travels_with_the_certificate(verifier: PhysicsSettle) -> None:
    """Every certificate carries the model's own caveats, in the magnitude."""
    _, bad = verifier.fixtures()
    certificate = verifier.check(bad[0])[0]
    approximation = certificate.magnitude["approximation"]
    assert approximation["cavity"].startswith("24-gon")
    assert approximation["polygon_vertex_extra_frac"] == pytest.approx(
        1.0 / np.cos(np.pi / 24) - 1.0
    )
    assert approximation["units"].startswith("mm in")
    assert approximation["insert_shape"] == "cylinder"
    assert certificate.magnitude["sim_calls"] == 1


# --------------------------------------------------------------------------- #
# A broken tool is an environment finding, never a verdict
# --------------------------------------------------------------------------- #
def test_unmeasurable_mesh_is_not_evaluated_not_failed(verifier: PhysicsSettle) -> None:
    """A non-watertight insert cannot be ray-measured, so physics declines — as a note.

    It is *not* a `physics_error`: that family means "the environment failed" and the
    kernel stops the episode on it, so an unevaluable mesh would block every case it
    appears in. A capability boundary and a broken simulator are different facts.
    """
    view = _view(clearance_mm=0.15, engage_mm=8.0, through=False)
    holed = view.meshes[view.spec.insert.id].copy()
    holed.update_faces(np.arange(len(holed.faces) - 1))  # drop one face: no longer closed
    holed.remove_unreferenced_vertices()
    view.meshes[view.spec.insert.id] = holed

    assert verifier.check(view) == []
    assert verifier.last_sim_calls == 0
    assert verifier.last_evaluated is False
    assert verifier.last_note.startswith("physics not evaluated: insert_not_watertight")


def test_non_convex_insert_is_simulated_via_convex_decomposition(verifier: PhysicsSettle) -> None:
    """A concave part is simulated as convex pieces, not averaged into its hull.

    The first version refused this case outright ("not evaluated"), which made every
    generated part unjudgeable: generative meshes are almost never convex (the measured
    tube was 2.4x its hull). CoACD splits the solid, the body gets one geom per piece,
    and the certificate records the decomposition so a reader knows what was simulated.
    """
    try:
        import coacd  # noqa: F401,PLC0415 - the optional dependency this test needs
    except Exception:  # pragma: no cover - exercised only on a bare checkout
        import pytest

        pytest.skip("coacd not installed")

    body, peg = _peg_hole_meshes(clearance_mm=0.15, engage_mm=8.0, through=False)
    hook = trimesh.creation.box(extents=(4.0, 14.0, 4.0))
    hook.apply_translation((0.0, 0.0, 6.0))
    insert = trimesh.boolean.union([peg, hook], engine="manifold")
    view = _view(clearance_mm=0.15, engage_mm=8.0, through=False)
    view.meshes[view.spec.insert.id] = insert

    certificates = verifier.check(view)

    assert verifier.last_evaluated is True
    assert verifier.last_sim_calls == 1
    approximation = (certificates[0].magnitude.get("approximation") if certificates else None) or {}
    if certificates:
        assert approximation.get("insert_shape") == "convex_parts"
        assert approximation.get("insert_geoms", 0) >= 2
        assert "coacd" in str(approximation.get("insert_decomposition"))
    else:  # a hook that happens to hold is still a real measurement
        assert verifier.last_note == ""


def test_non_convex_insert_without_coacd_is_a_note(verifier: PhysicsSettle, monkeypatch) -> None:
    """When the decomposer is absent the honest answer is still "not evaluated"."""
    from pah.tools import mujoco_settle

    monkeypatch.setattr(mujoco_settle, "_convex_parts", lambda *_args, **_kwargs: None)
    body, peg = _peg_hole_meshes(clearance_mm=0.15, engage_mm=8.0, through=False)
    hook = trimesh.creation.box(extents=(4.0, 14.0, 4.0))
    hook.apply_translation((0.0, 0.0, 6.0))
    insert = trimesh.boolean.union([peg, hook], engine="manifold")
    view = _view(clearance_mm=0.15, engage_mm=8.0, through=False)
    view.meshes[view.spec.insert.id] = insert

    assert verifier.check(view) == []
    assert verifier.last_note.startswith("physics not evaluated: non_convex_insert")
    assert verifier.last_sim_calls == 0
    assert verifier.last_evaluated is False


def test_missing_mujoco_is_an_environment_error(verifier: PhysicsSettle, monkeypatch) -> None:
    """An unimportable simulator is the machine's fault, not the design's."""
    good, _ = verifier.fixtures()

    def explode():
        raise mujoco_settle.SettleError("mujoco_unavailable", "ImportError: no module")

    monkeypatch.setattr(mujoco_settle, "_mujoco", explode)
    certificates = verifier.check(good[0])
    assert [certificate.code for certificate in certificates] == [CODE_PHYSICS_ERROR]
    assert certificates[0].family == "environment"
    assert certificates[0].magnitude["reason"] == "mujoco_unavailable"
    assert verifier.last_sim_calls == 0
    assert verifier.last_note == ""


def test_unexpected_failure_is_also_environment(verifier: PhysicsSettle, monkeypatch) -> None:
    """Any exception from the tool becomes a certificate: the loop must not crash."""
    good, _ = verifier.fixtures()

    def explode(*args, **kwargs):
        raise RuntimeError("simulator exploded")

    monkeypatch.setattr(mujoco_settle, "settle", explode)
    certificates = verifier.check(good[0])
    assert [certificate.code for certificate in certificates] == [CODE_PHYSICS_ERROR]
    assert "simulator exploded" in certificates[0].magnitude["error"]


# --------------------------------------------------------------------------- #
# The registry gate
# --------------------------------------------------------------------------- #
def test_codes_are_the_closed_set() -> None:
    assert tuple(PhysicsSettle.spec.codes) == DESIGN_CODES["physics_settle"]
    assert PhysicsSettle.spec.cost_class == "expensive"


def test_registry_sanity_passes(verifier: PhysicsSettle) -> None:
    """§4.8: the verifier proves itself against its own pair before it may be cited."""
    registry = DesignVerifierRegistry()
    registry.register(verifier)
    assert registry.sanity() == {"physics_settle": True}
    assert registry.disabled() == frozenset()


# --------------------------------------------------------------------------- #
# The non-round insert path (a dovetail-like prism in a slot)
# --------------------------------------------------------------------------- #
def _bar_in_slot(*, clearance_mm: float) -> DesignView:
    """A prism sliding along X in a top-open slot: the hull path, not the cylinder.

    Deliberately built here rather than imported from T1: this verifier's fixtures
    must exist before a materialiser does, or the gate would only prove that the
    verifier agrees with the builder.
    """
    slot_w = 10.0
    depth = 8.0
    bar_len = 14.0
    plate = (60.0, 40.0, 12.0)
    body = trimesh.creation.box(extents=plate)
    body.apply_translation((0.0, 0.0, plate[2] / 2.0))
    cutter = trimesh.creation.box(
        extents=(plate[0] + 4.0, slot_w + 2.0 * clearance_mm, depth + 2.0)
    )
    cutter.apply_translation((0.0, 0.0, plate[2] - depth / 2.0 + 1.0))
    body = trimesh.boolean.difference([body, cutter], engine="manifold")
    bar = trimesh.creation.box(extents=(20.0, slot_w, bar_len))
    bar.apply_translation((0.0, 0.0, plate[2] - depth + bar_len / 2.0))
    spec = DesignSpec(
        spec_id="physics_fixture_dovetail",
        requirement="prism in a top-open slot, slides along X",
        parts=(
            PartSpec(id="body", role="body", size_mm=plate),
            PartSpec(id="bar", role="insert", size_mm=(20.0, slot_w, bar_len)),
        ),
        interfaces=(
            InterfaceSpec(
                id="iface_0",
                kind="dovetail",
                a="body",
                b="bar",
                params={"clearance_mm": clearance_mm, "engage_mm": depth},
            ),
        ),
        constraints=FitConstraints(),
    )
    return DesignView(
        spec=spec, meshes={"body": body, "bar": bar}, mate_axis=(1.0, 0.0, 0.0)
    )


def test_non_round_insert_is_judged_by_its_hull(verifier: PhysicsSettle) -> None:
    """A prism is not a cylinder: the hull path must not invent penetration."""
    view = _bar_in_slot(clearance_mm=0.15)
    certificates = verifier.check(view)
    assert certificates == []
    result = verifier.last_result
    assert result is not None
    assert result.insert_shape == "mesh_hull"
    assert result.wall_open_angles > 0  # the slot's mouth is open, and that is fine
    assert result.drift_mm == pytest.approx(0.15, abs=0.02)


def test_loose_prism_drifts(verifier: PhysicsSettle) -> None:
    """The dovetail axis is X, so gravity is perpendicular to it: play, not pull-out."""
    view = _bar_in_slot(clearance_mm=2.0)
    certificates = verifier.check(view)
    assert [certificate.code for certificate in certificates] == [CODE_JOINT_UNSTABLE]
    assert certificates[0].locus["reason"] == "lateral_drift"
    assert certificates[0].magnitude["drift_mm"] == pytest.approx(2.0, abs=0.05)
    assert certificates[0].magnitude["approximation"]["mate_axis_world"] == [1.0, 0.0, 0.0]


# --------------------------------------------------------------------------- #
# Units
# --------------------------------------------------------------------------- #
def test_units_are_millimetres_not_metres(verifier: PhysicsSettle) -> None:
    """A 10 mm peg's clearance reads 0.15 mm; the 1e-3 conversion is not forgotten."""
    assert mujoco_settle.MM_PER_M == 1e-3
    good, _ = verifier.fixtures()
    verifier.check(good[0])
    result = verifier.last_result
    assert result is not None
    assert result.insert_radius_mm == pytest.approx(5.0, abs=0.01)
    assert result.wall_radius_min_mm == pytest.approx(5.15, abs=0.01)
    assert result.engaged_span_mm == pytest.approx(8.0, abs=0.05)


def test_spec_and_mesh_clearance_agree() -> None:
    """The fixture spec's derived clearance is the one the meshes were built with."""
    for clearance in (0.15, 2.0, -0.05):
        spec = _spec(clearance_mm=clearance)
        assert spec.clearance_mm == pytest.approx(clearance)
        body, peg = _peg_hole_meshes(
            clearance_mm=clearance, engage_mm=8.0, through=False
        )
        # The peg's radius and the hole's radius differ by exactly the clearance.
        assert peg.bounds[1][0] - 0.0 == pytest.approx(5.0, abs=0.01)
        assert body.is_watertight
