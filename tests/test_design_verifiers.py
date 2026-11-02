"""The design plane's verifiers, their self-check gate, and the cascade adapter.

What these tests are for
------------------------
The loop optimises whatever the verifiers say, so the interesting properties are
not "does the boolean work" but:

* **each verifier can tell a good design from a bad one** — its own
  `known_good`/`known_bad` pair, run through the same gate `pah design sanity`
  runs, and a proof that the gate really does disable a verifier that lies;
* **the codes partition the phenomena** — a zero-clearance fit is a band
  violation, not a collision; a seated face contact is neither; a pin that cannot
  reach its seat fails the sweep while the seat itself stays clean;
* **a broken tool is unknown, never healthy** — a raising verifier and an
  unreadable solid both come back as `environment`-family certificates through
  the cascade, and the other verifiers still report;
* **the cascade pays for its work** — `spent` grows per verifier call and per
  materialisation, because `RepairLoop` charges the delta to the round
  (`tests/test_loop_budget.py` pins the loop's side of that contract).

The fixtures are built by `_design_fixtures` with trimesh booleans, not by the
materialiser: a gate whose fixtures came from the code under test would be
marking its own homework, and T1's builder is developed in parallel.
"""

from __future__ import annotations

import inspect
import warnings

import numpy as np
import pytest
import trimesh
from trimesh.creation import box

from pah.core.model import Cost
from pah.design.api import (
    DESIGN_CODES,
    DesignVerifierRegistry,
    DesignView,
    MaterializedDesign,
    assembly_from_spec,
)
from pah.tools.verifiers.base import VerifierSpec
from pah.tools.verifiers import _design_fixtures as fixtures
from pah.tools.verifiers.design_cascade import DesignVerifierCascade, default_registry
from pah.tools.verifiers.fit_clearance import FitClearance
from pah.tools.verifiers.insertion_path import InsertionPath
from pah.tools.verifiers.interface_interference import (
    INTERFERENCE_MIN_VOLUME_MM3,
    InterfaceInterference,
    intersection_solid,
    intersection_volume_mm3,
)
from pah.tools.verifiers.printability import Printability

VERIFIERS = (FitClearance(), InterfaceInterference(), InsertionPath(), Printability())


def own_registry() -> DesignVerifierRegistry:
    """The four verifiers this module owns, without the physics tier.

    The cascade tests below are about *this* adapter's bookkeeping — view,
    materialise, fail closed, charge — so they pin the set they charge for. The
    physics tier runs MuJoCo (seconds per check) and is paid in `sim_calls`; that
    is T5's contract, and `default_registry()` is where it is covered.
    """
    registry = DesignVerifierRegistry()
    for verifier in VERIFIERS:
        registry.register(verifier)
    return registry


def codes_of(certificates) -> list[str]:
    return [cert.code for cert in certificates]


def by_code(certificates, code: str):
    matches = [cert for cert in certificates if cert.code == code]
    assert len(matches) == 1, f"expected exactly one {code!r}, got {codes_of(certificates)}"
    return matches[0]


# --------------------------------------------------------------------------- #
# The §4.8 gate: a verifier proves itself, or it does not run
# --------------------------------------------------------------------------- #
def test_every_design_verifier_passes_its_own_fixtures():
    registry = default_registry()
    status = registry.sanity()

    assert set(status) == set(registry.ids())
    assert all(status.values()), {k: registry.why_disabled(k) for k, v in status.items() if not v}
    assert registry.disabled() == frozenset()
    for verifier_id in ("fit_clearance", "interface_interference", "insertion_path", "printability"):
        assert verifier_id in registry.ids()


def test_each_verifier_declares_exactly_the_codes_the_contract_assigns_it():
    for verifier in VERIFIERS:
        assert tuple(verifier.spec.codes) == DESIGN_CODES[verifier.spec.id]
    # The cascade is free to ship fewer verifiers than the contract names (the
    # physics tier is a separate delivery) but never an unknown id.
    assert set(default_registry().ids()) <= set(DESIGN_CODES)


def test_a_verifier_that_always_says_good_is_disabled():
    """The expensive failure: a verifier that cannot fail makes every design pass."""

    class AlwaysHealthy(FitClearance):
        def check(self, view):
            return []

    registry = DesignVerifierRegistry()
    registry.register(AlwaysHealthy())
    status = registry.sanity()

    assert status == {"fit_clearance": False}
    assert registry.disabled() == frozenset({"fit_clearance"})
    assert "known_bad[0] judged positive" in registry.why_disabled("fit_clearance")
    # And a disabled verifier cannot write evidence any more: the frozen
    # `DesignVerifierRegistry.check` skips it, and so does the cascade.
    assert registry.check(fixtures.loose_fixture()) == []


def test_a_verifier_that_always_fails_is_disabled():
    class AlwaysUnhappy(FitClearance):
        def check(self, view):
            return super().check(fixtures.oversized_fixture())

    registry = DesignVerifierRegistry()
    registry.register(AlwaysUnhappy())
    status = registry.sanity()

    assert status == {"fit_clearance": False}
    assert "known_good[0] judged negative" in registry.why_disabled("fit_clearance")


# --------------------------------------------------------------------------- #
# The caliber: a seated blind-socket joint is a face contact, not a collision
# --------------------------------------------------------------------------- #
def test_the_seated_design_earns_no_certificate_at_all():
    """The peg rests on the socket floor: zero shared volume, five clean checks."""
    view = fixtures.standard_fixture()

    assert intersection_volume_mm3(view.body, view.insert) <= INTERFERENCE_MIN_VOLUME_MM3
    for verifier in VERIFIERS:
        assert verifier.check(view) == [], verifier.spec.id
    assert own_registry().check(view) == []


def test_the_default_registry_judges_a_good_design_clean():
    """The whole mounted cascade, including any tier T5 has added.

    A verifier that fires on a design nobody can fault is worse than useless: the
    loop would spend its budget repairing a healthy part. So the *full* default
    registry — not just this module's four — has to stay silent on the canonical
    good pair. A red here is cross-tier news, not a fixture problem.
    """
    assert default_registry().check(fixtures.standard_fixture()) == []


def test_a_zero_clearance_fit_is_a_band_violation_not_a_collision():
    """Contact and overlap are different findings, and different repairs."""
    view = fixtures.clearance_fixture(10.0, 10.0, tag="press")

    assert codes_of(InterfaceInterference().check(view)) == []
    certificates = FitClearance().check(view)
    assert codes_of(certificates) == ["insufficient_clearance"]
    assert by_code(certificates, "insufficient_clearance").magnitude["clearance_mm"] == 0.0


# --------------------------------------------------------------------------- #
# fit_clearance
# --------------------------------------------------------------------------- #
def test_fit_clearance_reads_the_band_from_the_constraints():
    good = FitClearance().check(fixtures.standard_fixture())
    assert good == []

    loose = by_code(FitClearance().check(fixtures.loose_fixture()), "excessive_clearance")
    assert loose.magnitude["clearance_mm"] == pytest.approx(1.0)
    assert loose.magnitude["max_mm"] == pytest.approx(0.60)
    assert loose.magnitude["deviation_mm"] == pytest.approx(0.40)
    assert loose.magnitude["target_fit"] == "slip"

    tight = by_code(FitClearance().check(fixtures.oversized_fixture()), "insufficient_clearance")
    assert tight.magnitude["clearance_mm"] == pytest.approx(-0.10)
    assert tight.magnitude["min_mm"] == pytest.approx(0.03)
    assert tight.magnitude["deviation_mm"] == pytest.approx(-0.13)
    assert tight.locus["kind"] == "peg_hole"
    assert tight.observed_by == "fit_clearance"


def test_fit_clearance_follows_the_spec_not_the_mesh():
    """An edit to the diameters is judged even though no geometry was rebuilt."""
    view = fixtures.standard_fixture()
    edited = view.spec.with_interface(params={"d_peg_mm": 10.4})
    judged = FitClearance().check(DesignView(spec=edited, meshes=dict(view.meshes)))

    assert codes_of(judged) == ["insufficient_clearance"]
    assert judged[0].magnitude["clearance_mm"] == pytest.approx(-0.20)


# --------------------------------------------------------------------------- #
# interface_interference
# --------------------------------------------------------------------------- #
def test_interference_fires_only_when_the_solids_really_overlap():
    assert InterfaceInterference().check(fixtures.standard_fixture()) == []

    certificate = by_code(
        InterfaceInterference().check(fixtures.oversized_fixture()), "geometric_interference"
    )
    magnitude = certificate.magnitude
    # pi * (5.1^2 - 5.0^2) * 8 mm of engagement = 25.35 mm3 analytic.
    assert magnitude["volume_mm3"] == pytest.approx(25.35, abs=0.05)
    # The bbox extent along the mate axis: how far the overlap reaches in Z.
    assert magnitude["max_depth_mm"] == pytest.approx(8.0, abs=1e-6)
    assert magnitude["min_volume_mm3"] == INTERFERENCE_MIN_VOLUME_MM3
    assert certificate.family == "assembly"


def test_interference_treats_a_boolean_sliver_as_noise():
    """The threshold's whole job: 1e-4 mm3 of sliver is not a collision."""
    view = fixtures.standard_fixture()
    strict = InterfaceInterference(min_volume_mm3=0.0)
    loose = InterfaceInterference()

    assert intersection_volume_mm3(view.body, view.insert) == 0.0
    assert strict.check(view) == [] == loose.check(view)


def test_a_touching_pair_is_a_zero_volume_sheet_not_an_overlap():
    """Exact coplanar contact: manifold returns a surface, with zero volume.

    This is the shape a seated blind socket produces (T1 measured the same on the
    materialiser), and the reason the interference verifier judges by **volume**:
    `len(faces) > 0` or `is_watertight` as the criterion would flag every
    correctly seated joint. Only `volume > 1e-3 mm3` is a collision.
    """
    left = box(extents=(10.0, 10.0, 10.0))
    right = box(extents=(10.0, 10.0, 10.0))
    right.apply_translation((10.0, 0.0, 0.0))  # shares the face at x = 5

    sheet = intersection_solid(left, right)
    assert sheet is not None
    assert len(sheet.faces) > 0 and sheet.is_watertight  # a surface, not nothing
    assert intersection_volume_mm3(left, right) == 0.0


def test_the_seated_check_needs_no_warning_suppression_to_be_correct():
    """A zero-volume blend must not turn into a warning-as-evidence trap."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert InterfaceInterference().check(fixtures.standard_fixture()) == []


def test_interference_measures_depth_along_the_view_axis():
    view = fixtures.oversized_fixture()
    along_z = by_code(InterfaceInterference().check(view), "geometric_interference")
    assert along_z.magnitude["max_depth_mm"] == pytest.approx(8.0, abs=1e-6)

    sideways = DesignView(
        spec=view.spec, meshes=dict(view.meshes), mate_axis=(1.0, 0.0, 0.0)
    )
    along_x = by_code(InterfaceInterference().check(sideways), "geometric_interference")
    # The overlap is an annulus of radius 5.0-5.1, so its extent along X is a
    # diameter; measuring it along a hard-coded Z would report the 8 mm engagement.
    assert along_x.magnitude["max_depth_mm"] == pytest.approx(10.2, abs=1e-6)


def test_an_open_shell_is_unknown_rather_than_healthy():
    """A boolean on an open shell has no defined volume: raise, never report 0."""
    open_body = box(extents=fixtures.BODY_SIZE_MM)
    open_body.update_faces(np.arange(len(open_body.faces)) != 0)
    open_body.remove_unreferenced_vertices()
    assert not open_body.is_watertight

    view = DesignView(
        spec=fixtures.standard_fixture().spec,
        meshes={"body": open_body, "insert": fixtures.standard_fixture().insert},
    )
    with pytest.raises(ValueError, match="not watertight"):
        InterfaceInterference().check(view)


# --------------------------------------------------------------------------- #
# insertion_path
# --------------------------------------------------------------------------- #
def test_insertion_path_finds_what_the_seat_cannot_see():
    """A clean seat plus a narrow entry: only the sweep can call that blocked."""
    view = fixtures.stepped_fixture()

    assert InterfaceInterference().check(view) == []
    assert FitClearance().check(view) == []

    certificate = by_code(InsertionPath().check(view), "insertion_path_blocked")
    magnitude = certificate.magnitude
    assert magnitude["blocked_at_mm"] == pytest.approx(magnitude["approach_mm"])
    assert magnitude["volume_mm3"] > INTERFERENCE_MIN_VOLUME_MM3
    assert magnitude["samples"] == 14
    assert magnitude["step_mm"] == pytest.approx(6.0 / 14)


def test_insertion_path_is_clean_when_the_pin_fits():
    assert InsertionPath().check(fixtures.standard_fixture()) == []


def test_insertion_path_reports_a_pin_that_cannot_enter():
    certificate = by_code(
        InsertionPath().check(fixtures.oversized_fixture()), "insertion_path_blocked"
    )
    assert certificate.magnitude["blocked_at_mm"] == pytest.approx(6.0)
    # Blocked at the very first sample: the pin is wider than the bore, so it is
    # stopped before it starts. The seated interference is a separate finding.
    assert certificate.magnitude["blocked_at_mm"] == certificate.magnitude["approach_mm"]


def test_insertion_path_does_not_exist_without_an_approach():
    view = fixtures.standard_fixture()
    seated = DesignView(spec=view.spec, meshes=dict(view.meshes), approach_mm=0.0)
    assert InsertionPath().check(seated) == []


def test_insertion_path_sweeps_along_the_axis_the_view_reports():
    """A slide fit inserts along +X; a hard-coded +Z sweep would slander every dovetail."""
    view = fixtures.standard_fixture()
    assert InsertionPath().check(view) == []  # axial peg: +Z is right

    sideways = DesignView(
        spec=view.spec,
        meshes=dict(view.meshes),
        approach_mm=6.0,
        mate_axis=(1.0, 0.0, 0.0),
    )
    certificate = by_code(InsertionPath().check(sideways), "insertion_path_blocked")
    assert certificate.magnitude["mate_axis"] == [1.0, 0.0, 0.0]
    # Dragged 6 mm sideways, the peg is 6 mm into the socket wall at the first sample.
    assert certificate.magnitude["blocked_at_mm"] == pytest.approx(6.0)


# --------------------------------------------------------------------------- #
# printability
# --------------------------------------------------------------------------- #
def test_a_thin_wall_between_the_socket_and_the_outside_is_measured():
    certificate = by_code(
        Printability().check(fixtures.thin_wall_fixture()), "below_min_wall"
    )
    magnitude = certificate.magnitude
    # The fixture asks for 0.6 mm; sampling facet centres overshoots by ~1.5 %.
    assert magnitude["min_wall_mm"] == pytest.approx(0.6, abs=0.02)
    assert magnitude["required_wall_mm"] == pytest.approx(1.2)
    assert magnitude["min_wall_mm"] < magnitude["required_wall_mm"]
    assert magnitude["method"] == "ray_from_interface_surface_inward_normal"
    assert certificate.family == "manufacturing"


def test_a_wall_that_holds_is_not_reported():
    assert Printability().check(fixtures.standard_fixture()) == []


def test_a_downward_ceiling_above_the_bed_is_an_overhang():
    certificate = by_code(
        Printability().check(fixtures.pocket_fixture()), "unsupported_overhang"
    )
    magnitude = certificate.magnitude
    assert magnitude["part"] == "body"
    assert magnitude["area_mm2"] == pytest.approx(100.0, abs=1.0)
    assert magnitude["angle_from_horizontal_deg"] == pytest.approx(0.0)
    assert magnitude["min_angle_from_horizontal_deg"] == pytest.approx(45.0)
    assert certificate.locus["process"] == "fdm"


def test_a_bed_face_is_not_an_overhang():
    """Every part sits on its own bottom face; calling that an overhang fails all."""
    view = fixtures.standard_fixture()
    assert Printability().check(view) == []
    for mesh in (view.body, view.insert):
        lowest = float(mesh.bounds[0][2])
        bed_faces = np.isclose(mesh.triangles_center[:, 2], lowest, atol=1e-9)
        assert bed_faces.any(), "the fixture must actually have a bed face"


def test_no_declared_orientation_is_searched_not_skipped():
    """No declaration now means "recommend one", not "no manufacturing claim" (P2-2).

    The old contract made an undeclared orientation *unevaluated*; a decomposition
    prints every piece in its own best pose, so that left every multi-part run with
    a hole in its evidence. The verifier searches the 24 face-down poses instead and
    records that the pose was recommended rather than chosen by a person.
    """
    view = fixtures.standard_fixture()
    verifier = Printability()

    assert view.spec.constraints.print_orientation_deg is None
    certificates = verifier.check(view)

    assert verifier.overhang_evaluated is True
    assert verifier.orientation_source == "recommended"
    assert verifier.recommendation is not None
    assert verifier.orientation_deg == verifier.recommendation.angles
    assert verifier.orientation_top[0] is verifier.recommendation
    assert verifier.last_note == ""  # a recommendation is an answer, not a boundary
    assert certificates == []


def test_a_declared_orientation_is_evaluated():
    verifier = Printability()
    certificates = verifier.check(fixtures.pocket_fixture())

    assert verifier.overhang_evaluated is True
    assert verifier.orientation_source == "declared"
    assert codes_of(certificates) == ["unsupported_overhang"]


def test_the_print_orientation_decides_whether_a_face_overhangs():
    """Same solid, two orientations: the socket floor becomes a ceiling upside down."""
    view = fixtures.standard_fixture()
    as_drawn = view.spec.edit(constraints=fixtures.AS_DESIGNED_PRINT_BAND)
    assert Printability().check(DesignView(spec=as_drawn, meshes=dict(view.meshes))) == []

    upside_down = fixtures.AS_DESIGNED_PRINT_BAND.model_copy(
        update={"print_orientation_deg": (180.0, 0.0, 0.0)}
    )
    flipped = view.spec.edit(constraints=upside_down)
    certificate = by_code(
        Printability().check(DesignView(spec=flipped, meshes=dict(view.meshes))),
        "unsupported_overhang",
    )
    assert certificate.magnitude["print_orientation_deg"] == [180.0, 0.0, 0.0]
    assert certificate.magnitude["part"] == "body"
    # The socket floor: a 10 mm disc, 78.5 mm2, now facing down 2 mm above the bed.
    assert certificate.magnitude["area_mm2"] == pytest.approx(78.5, abs=2.0)


def test_the_overhang_note_explains_why_nothing_was_evaluated():
    """`not evaluated` survives only where there is no rule to apply a pose to."""
    verifier = Printability()

    verifier.check(fixtures.standard_fixture())
    assert verifier.overhang_evaluated is True
    assert verifier.last_note == ""  # a recommended pose is still an evaluation

    # Subtractive: an orientation would have nothing to judge, so this is a boundary.
    view = fixtures.pocket_fixture()
    cnc = view.spec.edit(
        constraints=fixtures.AS_DESIGNED_PRINT_BAND.model_copy(update={"process": "cnc"})
    )
    verifier.check(DesignView(spec=cnc, meshes=dict(view.meshes)))
    assert verifier.overhang_evaluated is False
    assert verifier.orientation_source == "none"
    assert "overhang not evaluated" in verifier.last_note
    assert "subtractive" in verifier.last_note


def test_the_cascade_aggregates_the_verifiers_notes(tmp_path):
    """The report's `verifiers.notes` channel: a boundary, not a failure."""
    # A recommended orientation is an answer, so printability leaves no note.
    searched, searched_assembly, _ = _cascade_over(fixtures.standard_fixture(), tmp_path)
    certificates = searched.check(searched_assembly)
    assert certificates == []
    assert "printability" not in searched.notes

    # A process with no overhang rule does leave one, and it reaches the report.
    view = fixtures.pocket_fixture()
    cnc = view.spec.edit(
        constraints=fixtures.AS_DESIGNED_PRINT_BAND.model_copy(update={"process": "cnc"})
    )
    bounded, bounded_assembly, _ = _cascade_over(
        DesignView(spec=cnc, meshes=dict(view.meshes)), tmp_path
    )
    bounded.check(bounded_assembly)
    assert "printability" in bounded.notes
    assert "not evaluated" in bounded.notes["printability"]


def test_the_process_decides_whether_overhangs_are_a_limit():
    view = fixtures.pocket_fixture()
    cnc = view.spec.edit(
        constraints=fixtures.AS_DESIGNED_PRINT_BAND.model_copy(update={"process": "cnc"})
    )
    verifier = Printability()
    assert verifier.check(DesignView(spec=cnc, meshes=dict(view.meshes))) == []
    assert verifier.overhang_evaluated is False


# --------------------------------------------------------------------------- #
# The cascade: fail closed, charge honestly, materialise when it must
# --------------------------------------------------------------------------- #
def _assembly_of(spec):
    """A design-plane assembly with no geometry paths: the state an operator leaves."""
    return assembly_from_spec(spec=spec)


def _no_geometry():
    """A materialiser that builds nothing, for tests about metering rather than solids."""

    def materialize(spec, out_dir):
        return MaterializedDesign(
            spec=spec,
            assembly=assembly_from_spec(spec=spec),
            meshes={},
            paths={},
            root=out_dir,
        )

    return materialize


def _cascade_over(view: DesignView, root, *, stop_on_hard: bool = False):
    """A cascade over `view`'s spec, with a recording materialiser standing in for T1.

    The assembly deliberately carries no geometry paths (`assembly_from_spec`
    leaves `geom.path` empty), which is the state a v0 operator leaves behind and
    the exact case the adapter has to materialise for.
    """
    assembly = assembly_from_spec(spec=view.spec)
    calls: list[str] = []

    def materialize(spec, out_dir):
        calls.append(spec.content_key())
        return MaterializedDesign(
            spec=spec,
            assembly=assembly,
            meshes=dict(view.meshes),
            paths={},
            root=root,
        )

    cascade = DesignVerifierCascade(
        own_registry(),
        root,
        stop_on_hard=stop_on_hard,
        materializer=materialize,
    )
    return cascade, assembly, calls


def test_the_cascade_satisfies_the_loop_port():
    """`LoopVerifiers` is a structural protocol: `check(assembly) -> list[Cert]`.

    The loop also reads an optional `spent` meter and charges its delta to the
    round (`tests/test_loop_budget.py`), so that attribute is part of the contract
    this adapter has to meet.
    """
    cascade = DesignVerifierCascade(default_registry())

    assert list(inspect.signature(cascade.check).parameters) == ["assembly"]
    assert isinstance(cascade.spent, Cost)


def test_a_raising_verifier_becomes_an_environment_certificate(tmp_path):
    """A broken tool must never look like a pass, and never stop the other checks."""

    class Exploding(InsertionPath):
        def check(self, view):
            raise MemoryError("simulated allocation failure")

    registry = DesignVerifierRegistry()
    registry.register(FitClearance())
    registry.register(InterfaceInterference())
    registry.register(Exploding())
    registry.register(Printability())

    view = fixtures.oversized_fixture()
    assembly = assembly_from_spec(spec=view.spec)
    cascade = DesignVerifierCascade(
        registry,
        tmp_path,
        materializer=lambda spec, out: MaterializedDesign(
            spec=spec, assembly=assembly, meshes=dict(view.meshes), paths={}, root=out
        ),
    )
    certificates = cascade.check(assembly)

    environment = [cert for cert in certificates if cert.family == "environment"]
    assert len(environment) == 1
    assert environment[0].code == "verifier_error"
    assert environment[0].observed_by == "design_cascade"
    assert "MemoryError" in environment[0].magnitude["error"]
    # The verifiers that do work still reported: a crash does not erase evidence.
    assert "geometric_interference" in codes_of(certificates)
    assert "insufficient_clearance" in codes_of(certificates)


def test_the_cascade_stops_when_the_caller_asked_it_to(tmp_path):
    cascade, assembly, _ = _cascade_over(fixtures.oversized_fixture(), tmp_path, stop_on_hard=True)
    certificates = cascade.check(assembly)

    # fit_clearance is cheapest and fires, so the medium tiers never run.
    assert codes_of(certificates) == ["insufficient_clearance"]
    # One materialisation + one verifier call.
    assert cascade.spent.geom_calls == 2


def test_the_cascade_meters_its_work(tmp_path):
    cascade, assembly, calls = _cascade_over(fixtures.standard_fixture(), tmp_path)
    enabled = len(own_registry().ids())

    assert cascade.spent.geom_calls == 0
    first = cascade.check(assembly)
    # One geometry call for the materialisation, one per cheap/medium verifier that
    # ran. An expensive tier would be billed in `sim_calls` instead.
    assert cascade.spent.geom_calls == 1 + enabled
    assert cascade.spent.sim_calls == 0
    assert calls == [fixtures.standard_fixture().spec.content_key()]

    cascade.check(assembly)
    # The second pass reuses the built solids, so only verification is charged.
    assert cascade.spent.geom_calls == 1 + 2 * enabled
    assert len(calls) == 1
    assert first == []


def test_an_expensive_verifier_is_billed_in_simulation_calls(tmp_path):
    """The physics tier must not look free, and must not be charged for a refusal."""

    class PretendPhysics(InsertionPath):
        spec = VerifierSpec(
            id="physics_settle",
            codes=DESIGN_CODES["physics_settle"],
            cost_class="expensive",
            fixtures={},
        )

        def __init__(self, calls: int) -> None:
            self.last_sim_calls = calls

        def check(self, view):
            return []

    for reported, expected in ((1, 1), (0, 0)):  # a refusal costs no simulation
        registry = DesignVerifierRegistry()
        registry.register(PretendPhysics(reported))
        cascade = DesignVerifierCascade(registry, tmp_path, materializer=_no_geometry())
        cascade.check(_assembly_of(fixtures.standard_fixture().spec))
        assert cascade.spent.sim_calls == expected, reported
        assert cascade.spent.geom_calls == 1  # the materialisation, charged once

    # Without the attribute a raise is still an attempt: one call, never zero.
    class Raising(InsertionPath):
        spec = VerifierSpec(
            id="physics_settle",
            codes=DESIGN_CODES["physics_settle"],
            cost_class="expensive",
            fixtures={},
        )

        def check(self, view):
            raise RuntimeError("no mujoco")

    registry = DesignVerifierRegistry()
    registry.register(Raising())
    cascade = DesignVerifierCascade(registry, tmp_path, materializer=_no_geometry())
    cascade.check(_assembly_of(fixtures.standard_fixture().spec))
    assert cascade.spent.sim_calls == 1


def test_the_cascade_materialises_once_per_spec(tmp_path):
    view = fixtures.oversized_fixture()
    cascade, assembly, calls = _cascade_over(view, tmp_path)

    certificates = cascade.check(assembly)
    # The materialised solids are what the verifiers judged, not an empty view.
    assert "geometric_interference" in codes_of(certificates)
    assert len(calls) == 1

    cascade.check(assembly)
    assert len(calls) == 1


def test_the_cascade_prefers_the_builder_over_recorded_paths(tmp_path):
    """Only the builder knows the insertion axis, so recorded STLs are a fallback.

    An assembly records paths but no `mate_axis`; a view rebuilt from those paths
    would default to +Z and report a slide fit as blocked. The cascade therefore
    materialises whenever it has somewhere to write, even when the STLs exist.
    """
    view = fixtures.standard_fixture()
    assembly = assembly_from_spec(
        spec=view.spec,
        geometry_paths={"body": "gone/body.stl", "insert": "gone/insert.stl"},
    )
    calls: list[str] = []

    def materialize(spec, out_dir):
        calls.append(spec.content_key())
        return MaterializedDesign(
            spec=spec, assembly=assembly, meshes=dict(view.meshes), paths={}, root=out_dir
        )

    cascade = DesignVerifierCascade(own_registry(), tmp_path, materializer=materialize)
    certificates = cascade.check(assembly)

    assert certificates == []
    assert len(calls) == 1


def test_a_non_design_assembly_is_unknown_not_healthy(tmp_path):
    """An M0/M1 case has no spec: the cascade must say unknown, not 'no failures'."""
    from pah.core.model import Assembly

    dataset_assembly = Assembly(
        id="asm_000028",
        version=1,
        parts=(),
        joints=(),
        connectors=(),
        constraints={},
    )
    cascade = DesignVerifierCascade(default_registry(), tmp_path)
    certificates = cascade.check(dataset_assembly)

    assert len(certificates) == 1
    assert certificates[0].family == "environment"
    assert certificates[0].code == "verifier_error"
    assert certificates[0].locus["assembly"] == "asm_000028"


def test_the_cascade_reports_a_missing_root_instead_of_crashing():
    cascade = DesignVerifierCascade(default_registry())
    certificates = cascade.check(assembly_from_spec(spec=fixtures.standard_fixture().spec))

    assert len(certificates) == 1
    assert certificates[0].family == "environment"
    assert "root" in certificates[0].magnitude["error"]


def test_an_empty_mesh_is_unknown_rather_than_healthy():
    """A view with no triangles cannot be materialised into a judgement."""
    empty = DesignView(spec=fixtures.standard_fixture().spec, meshes={})
    with pytest.raises(KeyError):
        InterfaceInterference().check(empty)


def test_fixture_meshes_are_closed_solids():
    """A fixture that is not a closed solid would make every boolean meaningless."""
    for view in (fixtures.standard_fixture(), fixtures.oversized_fixture()):
        for part_id in view.part_ids():
            mesh = view.mesh(part_id)
            assert isinstance(mesh, trimesh.Trimesh)
            assert mesh.is_watertight and mesh.volume > 0.0



