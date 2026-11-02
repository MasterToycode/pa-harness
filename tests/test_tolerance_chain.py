"""Tolerance chain: a fit band and a process capability are different claims (P2-3).

The audit's C3 asks for two things that the design plane did not have before:

* given a process profile and a feature size, the report must say **where** the
  tolerance band came from — not just assert a range;
* a fit the process cannot make must be `not_manufacturable`, not a band
  violation. "You asked for the wrong gap" and "this process cannot hold that
  gap" are different diagnoses with different repairs.

The tests are split the way the code is: the pure function
(`design/process.py:achievable_clearance_mm`) is checked against its documented
arithmetic and its monotonicity, and the verifier is checked against the four
cases the audit names plus the boundaries that keep the two codes apart.

Numbers used below, from `PROCESS_PROFILES` and `CLEARANCE_ERROR_FRACTION = 0.25`:

| process | tolerance (diametral) | clearance floor at a ~10 mm feature |
|---|---|---|
| fdm | 0.20 mm | 0.05 mm |
| sla | 0.10 mm | 0.025 mm |
| cnc | 0.05 mm | 0.0125 mm |
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from pah.app.runtime import _fit_capability_block, _verdicts
from pah.core.model import Budget, FailureCertificate
from pah.design import spec_of
from pah.design.api import CODE_NOT_MANUFACTURABLE, DESIGN_CODES, DesignView
from pah.design.process import (
    CLEARANCE_ERROR_FRACTION,
    PROCESS_PROFILES,
    achievable_clearance_mm,
    profile_for,
)
from pah.design.spec import (
    DesignSpec,
    FitConstraints,
    InterfaceSpec,
    PartSpec,
    PegHoleParams,
)
from pah.policy.design_policies import DEFAULT_LADDER, RulePolicy
from pah.tools.verifiers.fit_clearance import CODES, FitClearance
from test_design_ops import (  # noqa: E402  (pytest puts tests/ on sys.path)
    base_assembly,
    build_registry,
    cert,
)


def fit_spec(
    *,
    d_peg_mm: float = 9.9,
    d_hole_mm: float = 10.0,
    process: str = "fdm",
    min_clearance_mm: float = 0.10,
    max_clearance_mm: float = 0.60,
    target_fit: str = "slip",
) -> DesignSpec:
    """One peg/socket spec carrying only the numbers the fit question needs."""
    return DesignSpec(
        spec_id="tolerance-chain",
        parts=(
            PartSpec(id="body", role="body", size_mm=(60.0, 40.0, 12.0)),
            PartSpec(
                id="insert",
                role="insert",
                size_mm=(22.0, 22.0, 20.0),
                pose_mm=(0.0, 0.0, -2.0),
            ),
        ),
        interfaces=(
            InterfaceSpec(
                id="iface_0",
                kind="peg_hole",
                a="body",
                b="insert",
                params={"d_peg_mm": d_peg_mm, "d_hole_mm": d_hole_mm, "engage_mm": 8.0},
            ),
        ),
        constraints=FitConstraints(
            min_clearance_mm=min_clearance_mm,
            max_clearance_mm=max_clearance_mm,
            process=process,
            target_fit=target_fit,
        ),
    )


def codes_of(view) -> list[str]:
    return [certificate.code for certificate in FitClearance().check(view)]


# --------------------------------------------------------------------------- #
# The pure function: where the number comes from
# --------------------------------------------------------------------------- #
def test_the_capability_code_is_registered_under_its_owner():
    """The §4.8 gate compares `spec.codes` to `DESIGN_CODES` exactly, order included."""
    assert CODE_NOT_MANUFACTURABLE == "not_manufacturable"
    assert CODES == DESIGN_CODES["fit_clearance"]


@pytest.mark.parametrize("process", sorted(PROCESS_PROFILES))
def test_the_floor_never_worsens_as_the_feature_grows(process: str):
    """A larger feature is never harder: the profile quotes tolerance absolutely."""
    profile = PROCESS_PROFILES[process]
    floors = [achievable_clearance_mm(profile, size)[0] for size in (1.0, 2.0, 5.0, 10.0, 20.0)]

    assert floors == sorted(floors, reverse=True)
    assert floors[-1] == pytest.approx(profile.tolerance_mm * CLEARANCE_ERROR_FRACTION)


def test_a_feature_below_the_process_floor_is_proportionally_harder():
    profile = PROCESS_PROFILES["fdm"]
    at_floor = achievable_clearance_mm(profile, profile.min_feature_mm)[0]
    below = achievable_clearance_mm(profile, profile.min_feature_mm / 2.0)[0]

    # The same absolute error on a feature half the size is twice the fit problem.
    assert below == pytest.approx(2.0 * at_floor)


def test_the_floor_is_the_documented_fraction_of_the_tolerance():
    """The chain in `achievable_clearance_mm`: diameter band -> radius -> random part."""
    profile = PROCESS_PROFILES["fdm"]
    best, worst, note = achievable_clearance_mm(profile, 9.7)

    assert best == pytest.approx(profile.tolerance_mm * CLEARANCE_ERROR_FRACTION)
    # Asking for the floor and getting the other end of the same band: [0, 2 * best].
    assert worst == pytest.approx(2.0 * best)


def test_the_floor_names_the_inputs_it_came_from():
    """C3: a report has to print the source, so the note carries the numbers."""
    profile = PROCESS_PROFILES["fdm"]
    _best, _worst, note = achievable_clearance_mm(profile, 9.7)

    for token in ("fdm", f"{profile.tolerance_mm:.3f}", "9.70", "floor"):
        assert token in note


def test_a_non_positive_feature_is_refused():
    with pytest.raises(ValueError, match="positive"):
        achievable_clearance_mm(PROCESS_PROFILES["fdm"], 0.0)


# --------------------------------------------------------------------------- #
# The verifier: the audit's four cases
# --------------------------------------------------------------------------- #
def test_fdm_cannot_hold_a_two_hundredths_clearance():
    """The headline case: 0.02 mm is not a fit, and the certificate proves why."""
    view = DesignView(spec=fit_spec(d_peg_mm=9.96, min_clearance_mm=0.0))
    assert view.spec.clearance_mm == pytest.approx(0.02)

    certificates = FitClearance().check(view)

    assert [certificate.code for certificate in certificates] == [CODE_NOT_MANUFACTURABLE]
    certificate = certificates[0]
    assert certificate.family == "manufacturing"
    magnitude = certificate.magnitude
    assert magnitude["designed_clearance_mm"] == pytest.approx(0.02)
    assert magnitude["process_best_clearance_mm"] == pytest.approx(0.05)
    assert magnitude["process_worst_clearance_mm"] == pytest.approx(0.10)
    assert magnitude["process"] == "fdm"
    assert magnitude["feature_mm"] == pytest.approx(9.96)
    assert magnitude["band_mm"] == pytest.approx([0.0, 0.60])
    assert magnitude["expected_tolerance_mm"] == pytest.approx(0.05)
    assert "ream" in magnitude["post_process"]
    assert "fdm" in magnitude["source"]
    assert certificate.observed_by == "fit_clearance"


def test_fdm_holds_a_two_tenths_clearance_and_reports_its_source():
    """A clean fit emits no evidence, so C3's source has to live on the verifier."""
    verifier = FitClearance()
    view = DesignView(spec=fit_spec(d_peg_mm=9.6, min_clearance_mm=0.10))

    assert view.spec.clearance_mm == pytest.approx(0.2)
    assert verifier.check(view) == []
    assert verifier.last_note == ""
    assert verifier.last_capability is not None
    assert verifier.last_capability["manufacturable"] is True
    assert verifier.last_capability["process_best_clearance_mm"] == pytest.approx(0.05)
    assert verifier.last_capability["designed_clearance_mm"] == pytest.approx(0.2)
    assert "fdm" in verifier.last_capability["source"]


def test_a_gap_at_the_process_floor_is_makeable():
    """0.05 mm is FDM's floor (0.20 / 4); asking for it is tight, not impossible.

    The clearance arrives as `(10.0 - 9.9) / 2 = 0.04999999999999982`, so this also
    pins the comparison's float slack: a design at the stated capability must not
    be refused by binary arithmetic.
    """
    verifier = FitClearance()
    view = DesignView(spec=fit_spec(d_peg_mm=9.9, min_clearance_mm=0.03))

    # The trace the rule leaves: the float is *below* the floor, and the verdict is
    # still "makeable". A strict `<` would flip this design to not_manufacturable.
    assert view.spec.clearance_mm < 0.05
    assert verifier.check(view) == []
    assert verifier.last_capability["manufacturable"] is True


def test_sla_holds_the_gap_fdm_cannot_and_the_band_still_speaks():
    """The same 0.05 mm is inside resin's capability, so only the band can fire."""
    inside_everything = DesignView(
        spec=fit_spec(process="sla", min_clearance_mm=0.03, max_clearance_mm=0.60)
    )
    assert codes_of(inside_everything) == []

    off_band = DesignView(
        spec=fit_spec(process="sla", min_clearance_mm=0.10, max_clearance_mm=0.60)
    )
    assert codes_of(off_band) == ["insufficient_clearance"]


def test_the_capability_floor_outranks_the_band_code():
    """Below the floor *and* off-band is one diagnosis, not two: the process."""
    view = DesignView(spec=fit_spec(d_peg_mm=9.96, min_clearance_mm=0.10))

    assert codes_of(view) == [CODE_NOT_MANUFACTURABLE]


def test_zero_and_negative_clearances_are_intents_not_capability_failures():
    """A press fit is a design choice; the capability code is about gaps that open."""
    assert codes_of(DesignView(spec=fit_spec(d_peg_mm=10.0, min_clearance_mm=0.03))) == [
        "insufficient_clearance"
    ]
    assert codes_of(DesignView(spec=fit_spec(d_peg_mm=10.2, min_clearance_mm=0.03))) == [
        "insufficient_clearance"
    ]


def test_a_process_without_a_profile_is_not_evaluated():
    """No profile means no capability claim — the P2-1 rule, borrowed.

    `FitConstraints.process` is a closed literal, so a validated spec cannot reach
    this branch; the guard exists for a schema that grows a process the table has
    not described yet. A duck-typed view is the only way to exercise it, and it is
    worth exercising: the alternative would be borrowing FDM's numbers and
    reporting them as if they applied.
    """
    assert profile_for("origami") is None

    verifier = FitClearance()
    stub = _stub_view(process="origami")
    certificates = verifier.check(stub)

    assert [certificate.code for certificate in certificates] == ["insufficient_clearance"]
    assert verifier.last_capability is None
    assert "not evaluated" in verifier.last_note
    assert "origami" in verifier.last_note

    # The same numbers *with* a profile do produce the capability code, so the
    # assertion above is about the missing profile and not about a silent path.
    assert codes_of(DesignView(spec=fit_spec(d_peg_mm=9.96, min_clearance_mm=0.0))) == [
        CODE_NOT_MANUFACTURABLE
    ]


def _stub_view(*, process: str):
    """A view with the fields `fit_clearance` reads, and a process no profile has."""
    constraints = SimpleNamespace(
        process=process,
        min_clearance_mm=0.03,
        max_clearance_mm=0.60,
        target_fit="slip",
        in_band=lambda clearance: 0.03 <= clearance <= 0.60,
    )
    interface = SimpleNamespace(
        id="iface_0",
        kind="peg_hole",
        a="body",
        b="insert",
        validated_params=lambda: PegHoleParams(d_peg_mm=9.96, d_hole_mm=10.0, engage_mm=8.0),
    )
    spec = SimpleNamespace(
        spec_id="stub",
        revision=1,
        constraints=constraints,
        interface=interface,
        clearance_mm=0.02,
    )
    return SimpleNamespace(spec=spec)


# --------------------------------------------------------------------------- #
# The repair side: a capability failure must be fixable by changing the gap
# --------------------------------------------------------------------------- #
def test_a_capability_failure_is_repairable_by_the_gap_operators():
    """The code is only useful if the action space accepts it (P2-3 follow-up).

    Before this, `applicable()` on the new code returned exactly the fallback
    operators -- `['regenerate_interface', 'switch_connector']` -- so "0.02 mm is
    too tight for FDM" was answered by changing the connector paradigm.
    """
    registry = build_registry()
    applicable = registry.applicable(base_assembly(), [cert(CODE_NOT_MANUFACTURABLE)])

    assert "set_clearance" in applicable
    assert "resize_feature" in applicable


def test_the_rule_policy_repairs_a_capability_failure_with_set_clearance_first():
    """First action, not a fallback: the tight gap is a number problem."""
    registry = build_registry()
    assembly = base_assembly()  # peg_hole, clearance 0.0, band [0.10, 0.60]
    policy = RulePolicy(registry=registry)

    plan = policy.propose(
        assembly,
        [cert(CODE_NOT_MANUFACTURABLE)],
        [],
        Budget(rounds=3, geom_calls=100, sim_calls=20),
    )

    assert plan is not None
    assert plan.operator == "set_clearance"
    # `mid` of the band, and above any profile floor in the table: the repair is a
    # gap the process can actually hold.
    assert plan.params["clearance_mm"] == pytest.approx(0.35)
    repaired = registry.apply(assembly, plan)
    assert spec_of(repaired).constraints.in_band(spec_of(repaired).clearance_mm)

    # The fallback ladder is still there for other codes; it is not what this one gets.
    assert DEFAULT_LADDER[CODE_NOT_MANUFACTURABLE][0] == "set_clearance"
    assert "switch_connector" not in DEFAULT_LADDER[CODE_NOT_MANUFACTURABLE]


# --------------------------------------------------------------------------- #
# The report side: blocking, and the tolerance chain behind a clean fit (C3)
# --------------------------------------------------------------------------- #
def test_a_capability_failure_blocks_the_assembly_claim(tmp_path):
    """A gap the process cannot hold means the joint cannot be made, not "maybe"."""
    certificate = FailureCertificate(
        code=CODE_NOT_MANUFACTURABLE,
        family="manufacturing",
        locus={"interface": "iface_0"},
        magnitude={"designed_clearance_mm": 0.02},
        observed_by="fit_clearance",
    )

    verdicts = _verdicts([certificate], {}, tmp_path)

    assert "not_manufacturable" in verdicts["assemblable"]["blocking"]
    assert verdicts["assemblable"]["ok"] is False


def test_the_report_carries_the_tolerance_chain_after_a_clean_fit():
    """A clean fit emits no certificate, so the source can only come from the
    verifier instance — the same trick `_orientation_block` plays for orientation."""
    verifier = FitClearance()
    view = DesignView(spec=fit_spec(d_peg_mm=9.6, min_clearance_mm=0.10))
    assert verifier.check(view) == []  # clean: nothing to cite in the evidence list

    block = _fit_capability_block(_fake_cascade(verifier))

    assert block["available"] is True
    assert block["process"] == "fdm"
    assert block["process_best_clearance_mm"] == pytest.approx(0.05)
    assert block["process_worst_clearance_mm"] == pytest.approx(0.10)
    assert block["designed_clearance_mm"] == pytest.approx(0.2)
    assert block["expected_tolerance_mm"] == pytest.approx(0.05)
    assert block["band_mm"] == pytest.approx([0.10, 0.60])
    assert block["manufacturable"] is True
    assert "fdm" in block["source"]
    assert block["post_process"]


def test_the_report_admits_when_no_tolerance_chain_was_recorded():
    """Not reachable / not evaluated is `available: false`, never a borrowed band."""
    assert _fit_capability_block(SimpleNamespace()) == {"available": False}
    # A verifier that never ran this design has no capability to report.
    assert _fit_capability_block(_fake_cascade(FitClearance())) == {"available": False}


class _VerifierRegistry:
    """The one method `_fit_capability_block` uses on a cascade's registry."""

    def __init__(self, verifier):
        self._verifier = verifier

    def get(self, verifier_id: str):
        if verifier_id != "fit_clearance":
            raise KeyError(verifier_id)
        return self._verifier


def _fake_cascade(verifier):
    """The shape the block unwraps: `verifiers._inner.registry.get(...)`."""
    return SimpleNamespace(_inner=SimpleNamespace(registry=_VerifierRegistry(verifier)))
