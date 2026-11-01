"""`physics_settle`: the expensive rung — gravity on the seated joint, really run.

What it decides
---------------
`docs/pah-dev-doc-v1.md` §6: *静态无干涉不等于动力学稳定*. The other four design-plane
verifiers answer questions about the geometry as drawn; this one answers the only
question that needs a clock: **does the insert stay where the design put it once
gravity is on?** Its three codes are the design plane's closed set for that:

* `joint_unstable` — the insert moved. `locus["reason"]` says how: `lateral_drift`
  (the joint's play under gravity), `pull_out` (travelled against the insertion
  direction), `drop_through` (travelled along it and left the cavity entirely), or
  `tilt` (ended wedged off the mate axis without the centre of mass moving).
  `magnitude` carries the measured millimetres and the threshold they were judged
  against.
* `dynamic_penetration` — the solver had to overlap the insert and the wall to keep
  the joint together: the fit is over-closed in the built geometry. This is what a
  *negative* designed clearance looks like dynamically, and it is the reason a
  press fit cannot be certified by this model (see `mujoco_settle`).
* `physics_error` — `family="environment"`, and **only for a tool that failed**: MuJoCo
  missing, a model that will not compile, an integrator that diverged, an unexpected
  exception. It is **not** in any failure rate and it is **not** a pass
  (`docs/contracts/v2-interfaces.md` §3).

**A geometry outside the model's representation is not a tool failure.** MuJoCo collides
a mesh as its convex hull, so a snap hook's undercut (the feature that retains it) cannot
be represented; likewise a non-watertight solid cannot be ray-measured. Those cases return
**no certificate** and set `last_note` instead: capability boundary, tool failure and
physical defect are three different facts and a report has to keep them apart. Emitting
`physics_error` for them would be worse than useless — `family="environment"` makes the
loop treat the state as *unknown*, which stops the episode, so an unevaluable insert would
block every case it appears in. `last_note` is how a report says **"physics: not
evaluated"** without pretending it passed; `last_sim_calls` stays 0 in that case.

Where the thresholds come from
------------------------------
`drift` and `penetration` tolerances are **derived from the spec's own
`FitConstraints`**, not hard-coded:

* `drift <= 1.7 x constraints.max_clearance_mm` (floored at 0.2 mm). A slip fit is
  allowed to move by its clearance, so the band's own ceiling is the natural scale;
  1.7x means physics never contradicts a fit the constraint band already accepts.
  At the default 0.6 mm band this is 1.02 mm, which is also 10% of the 10 mm peg the
  demo case uses — where a peg stops being *located* in its hole.
* `penetration <= min(0.03, constraints.min_clearance_mm / 3)` mm (floored at
  0.01 mm). The contact solver's own compliance is measured at 0.004 mm for the
  fixture insert (1.4 g under 9.81 m/s^2), so 0.01 mm is 2.5x the noise floor, and
  a third of the minimum legal clearance means a design that keeps any legal
  clearance cannot reach it.
* `pull_out <= 0.5` mm and `tilt <= 15 deg` are absolute: they describe having left
  the seat at all, not a fit tolerance.

The mate axis
-------------
The axis comes from `DesignView.mate_axis` — the insertion facts of the *built*
geometry, as the contract now carries them. This module does **not** assume +Z:
gravity stays world -Z, but "which way is out of the socket" is read off the view,
because for a dovetail (a slide fit along +X) the two are perpendicular and a
hard-coded +Z would turn "the insert slid along its groove" into "the insert fell
out of a hole it was never in". `MATE_AXIS_BY_KIND` survives only as
(a) the axis a fixture declares for itself, and (b) the fallback for a view that
does not carry one (a zero/absent vector), with `mate_axis_for` still refusing an
unknown kind rather than guessing.

Fixtures
--------
`fixtures()` builds four `DesignView`s out of trimesh solids, not out of a
materialiser, so the gate runs before T1's builder exists and cannot be "fixed" by
changing it. They are the four cases measured in `docs/internal/physics-notes.md`:
a 0.15 mm blind seat (must pass), a 2.0 mm clearance (play), a 0.05 mm oversize peg
(penetration), and a through hole with nothing under the insert (falls out).

Cost
----
One `check` runs one MuJoCo settle: 4000 steps at 2e-4 s, measured at 0.04-0.41 s
of CPU plus 0.1-0.3 s of mesh/raycast setup. The verifier publishes
`last_sim_calls` (0 or 1) after every call, and every certificate carries
`magnitude["sim_calls"]`, so a cascade can bill the round without reaching into
this module.
"""

from __future__ import annotations

import functools

import trimesh

from pah.core.model import FailureCertificate
from pah.design.api import (
    CODE_DYNAMIC_PENETRATION,
    CODE_JOINT_UNSTABLE,
    CODE_PHYSICS_ERROR,
    DesignView,
)
from pah.design.spec import DesignSpec, FitConstraints, InterfaceSpec, PartSpec
from pah.tools import mujoco_settle
from pah.tools.mujoco_settle import SettleResult, SettleTolerances
from pah.tools.verifiers.base import VerifierSpec

__all__ = [
    "DRIFT_RATIO",
    "MATE_AXIS_BY_KIND",
    "PENETRATION_FRACTION",
    "PhysicsSettle",
    "mate_axis_for",
    "tolerances_for",
]

#: The mate axis (insertion direction) each builtin connector kind seats along.
#: `dovetail` slides along the body's X axis; the other two seat along +Z.
#: A plugin kind must add itself here — the alternative is guessing, and a wrong
#: axis turns gravity from "pulls the insert out" into "pushes it in".
MATE_AXIS_BY_KIND: dict[str, tuple[float, float, float]] = {
    "peg_hole": (0.0, 0.0, 1.0),
    "snap_fit": (0.0, 0.0, 1.0),
    "dovetail": (1.0, 0.0, 0.0),
}

#: `drift_tol = DRIFT_RATIO * constraints.max_clearance_mm`. See the module docstring.
DRIFT_RATIO = 1.7
#: `penetration_tol = min(0.03, PENETRATION_FRACTION * constraints.min_clearance_mm)`.
PENETRATION_FRACTION = 1.0 / 3.0
#: Never let a degenerate band (0 mm clearance) make a tolerance smaller than the
#: contact solver's own compliance, or the verifier would report software softness.
MIN_DRIFT_TOL_MM = 0.2
MIN_PENETRATION_TOL_MM = 0.01
MAX_PENETRATION_TOL_MM = 0.03
#: Absolute: having left the seat, and being wedged, are not fit tolerances.
PULL_OUT_TOL_MM = 0.5
TILT_TOL_DEG = 15.0

#: `SettleError` reasons that mean **the geometry is outside this model's
#: representation**, not "the simulator is broken". A decline is a *coverage* fact:
#: `check` returns no certificate and sets `last_note`. Emitting `physics_error`
#: (`family="environment"`) for one of these would make the loop read "unknown" and stop
#: the episode — so an insert whose undercut MuJoCo cannot represent would block every
#: case it appears in. That is the difference between "I cannot judge this" and "the tool
#: broke", and only the second one is an environment failure.
#:
#: Everything *not* listed here — `mujoco_unavailable`, `model_compile_failed`,
#: `step_failed`, `diverged`, `wall_budget_exceeded`, `raycast_failed`, an unexpected
#: exception — is a tool failure and does emit `physics_error`.
MODEL_BOUNDARY_REASONS = frozenset(
    {
        # MuJoCo collides a mesh as its convex hull; a concave insert's retention
        # feature would be averaged away, so the simulation would describe another part.
        "non_convex_insert",
        # Solids the ray engine cannot trust, and views that are not a seated pair.
        "body_not_watertight",
        "insert_not_watertight",
        "body_not_a_mesh",
        "insert_not_a_mesh",
        "body_empty",
        "insert_empty",
        "insert_degenerate",
        "insert_no_volume",
        "not_engaged",
        "no_cavity_measured",
        "cavity_implausible",
        "no_walls",
        "empty_cavity_span",
        # A view that declares no usable axis is a view problem, not a broken engine.
        "bad_axis",
        "bad_limits",
    }
)

#: Fixture dimensions, one place so a report can cite them.
FIXTURE_PLATE_MM = (60.0, 40.0, 12.0)
FIXTURE_PEG_MM = 10.0
FIXTURE_PEG_LEN_MM = 14.0
FIXTURE_ENGAGE_MM = 8.0


# --------------------------------------------------------------------------- #
# The tolerance derivation, in one place
# --------------------------------------------------------------------------- #
def tolerances_for(constraints: FitConstraints) -> SettleTolerances:
    """Turn the task's fit band into the settle's pass/fail thresholds.

    This is the only place the two planes' numbers touch: the fit verifier judges a
    clearance against the band, and physics judges the *motion* against the same
    band's scale, so changing the band moves both instead of letting one contradict
    the other. See the module docstring for why each multiple is what it is.
    """
    drift = max(DRIFT_RATIO * float(constraints.max_clearance_mm), MIN_DRIFT_TOL_MM)
    penetration = min(
        MAX_PENETRATION_TOL_MM, PENETRATION_FRACTION * float(constraints.min_clearance_mm)
    )
    penetration = max(penetration, MIN_PENETRATION_TOL_MM)
    return SettleTolerances(
        drift_mm=drift,
        penetration_mm=penetration,
        pull_out_mm=PULL_OUT_TOL_MM,
        tilt_deg=TILT_TOL_DEG,
    )


def mate_axis_for(spec: DesignSpec) -> tuple[float, float, float]:
    """The insertion direction for `spec`'s connector kind, or `KeyError`.

    Only a fallback: a `DesignView` carries `mate_axis` from the builder, and that
    is what `check` uses. This table is for fixtures and for a view that forgot to
    declare one, and it still refuses a kind it does not know rather than assuming
    +Z — a wrong axis turns "pulled out of the socket" into "pushed further in".
    """
    kind = spec.interface.kind
    try:
        return MATE_AXIS_BY_KIND[kind]
    except KeyError:
        raise KeyError(
            f"connector kind {kind!r} has no mate axis in MATE_AXIS_BY_KIND and the "
            "view declares none; gravity is applied relative to that axis, so the "
            "check refuses rather than assume +Z"
        ) from None


def _axis_of(view: DesignView) -> tuple[float, float, float]:
    """The view's declared mate axis, falling back to its connector kind."""
    declared = getattr(view, "mate_axis", None)
    if declared is not None:
        values = tuple(float(value) for value in declared)
        if any(abs(value) > 1e-12 for value in values):
            return values  # type: ignore[return-value]
    return mate_axis_for(view.spec)


def _observed_by() -> str:
    return f"physics_settle@{mujoco_settle.engine_version()}"


# --------------------------------------------------------------------------- #
# The verifier
# --------------------------------------------------------------------------- #
class PhysicsSettle:
    """Settle the seated pair under gravity and report what moved.

    `check` never raises: a simulation that could not run becomes a
    `physics_error` certificate (`family="environment"`), because the loop must keep
    going and an unknown state must never be read as a pass.
    """

    spec = VerifierSpec(
        id="physics_settle",
        codes=(CODE_JOINT_UNSTABLE, CODE_DYNAMIC_PENETRATION, CODE_PHYSICS_ERROR),
        cost_class="expensive",
        fixtures={
            "known_good": [
                {
                    "label": "blind peg-hole, 0.15 mm clearance, 8 mm engagement",
                    "clearance_mm": 0.15,
                    "engage_mm": FIXTURE_ENGAGE_MM,
                    "through": False,
                }
            ],
            "known_bad": [
                {
                    "label": "2.0 mm clearance: the joint's play exceeds the fit band",
                    "clearance_mm": 2.0,
                    "engage_mm": FIXTURE_ENGAGE_MM,
                    "through": False,
                },
                {
                    "label": "0.05 mm oversize peg: the fit is over-closed",
                    "clearance_mm": -0.05,
                    "engage_mm": FIXTURE_ENGAGE_MM,
                    "through": False,
                },
                {
                    "label": "through hole: nothing under the insert, gravity takes it",
                    "clearance_mm": 0.15,
                    "engage_mm": FIXTURE_ENGAGE_MM,
                    "through": True,
                },
            ],
        },
    )

    def __init__(self) -> None:
        #: `1` when the last `check` ran one settle, `0` when it could not. A cascade
        #: bills `sim_calls` off this because the verifier, not the cascade, knows
        #: whether a simulation actually happened.
        self.last_sim_calls: int = 0
        #: The last result, for a trace that wants the raw measurements.
        self.last_result: SettleResult | None = None
        self.last_error: str = ""
        #: Set when the check **declined** because the geometry is outside the model's
        #: representation: `"physics not evaluated: <reason>: <detail>"`. Empty means the
        #: check either evaluated the joint (a pass or a finding) or hit a tool failure
        #: (`last_error`). A report must read this before counting a `[]` as "physics
        #: passed" — `[]` here can also mean "not looked at".
        self.last_note: str = ""

    @property
    def last_evaluated(self) -> bool:
        """Whether the last `check` actually simulated the joint."""
        return self.last_sim_calls == 1 and not self.last_note

    # -- the check ---------------------------------------------------------
    def check(self, view: DesignView) -> list[FailureCertificate]:
        """Every way this seat failed to hold — `[]` when it held, never a raise."""
        self.last_sim_calls = 0
        self.last_result = None
        self.last_error = ""
        self.last_note = ""
        spec = view.spec
        try:
            axis = _axis_of(view)
            tolerances = tolerances_for(spec.constraints)
            result = mujoco_settle.settle(
                view.body,
                view.insert,
                axis=axis,
                clearance_mm=spec.clearance_mm,
                tolerances=tolerances,
            )
        except KeyError as error:  # an unknown connector kind, or a plugin's clearance
            return self._declined(
                "unsupported_connector_kind", str(error.args[0]) if error.args else str(error)
            )
        except mujoco_settle.SettleError as error:
            if error.reason in MODEL_BOUNDARY_REASONS:
                return self._declined(error.reason, error.detail)
            self.last_error = error.detail or error.reason
            return [self._failed(view, error.reason, self.last_error)]
        except Exception as error:  # noqa: BLE001 - a broken tool is never a verdict
            self.last_error = f"{type(error).__name__}: {error}"
            return [self._failed(view, "unexpected_failure", self.last_error)]
        self.last_sim_calls = 1
        self.last_result = result
        return self._certificates(view, result)

    # -- certificates ------------------------------------------------------
    def _locus(self, view: DesignView, reason: str = "") -> dict:
        spec = view.spec
        return {
            "spec": spec.spec_id,
            "revision": spec.revision,
            "kind": spec.interface.kind,
            "interface": spec.interface.id,
            "parts": [spec.body.id, spec.insert.id],
            "reason": reason,
        }

    def _measurements(self, result: SettleResult) -> dict:
        """The numbers a report or a policy needs, in mm, plus the model's caveats."""
        return {
            "drift_mm": round(result.drift_mm, 4),
            "clearance_mm": round(result.clearance_mm, 4),
            "wall_radius_min_mm": round(result.wall_radius_min_mm, 4),
            "wall_radius_max_mm": round(result.wall_radius_max_mm, 4),
            "engaged_span_mm": round(result.engaged_span_mm, 4),
            "penetration_mm": round(result.penetration_mm, 4),
            "axial_mm": round(result.axial_mm, 4),
            "pull_out_mm": round(result.pull_out_mm, 4),
            "slip_in_mm": round(result.slip_in_mm, 4),
            "tilt_deg": round(result.tilt_deg, 3),
            "supported": result.supported,
            "seat_gap_mm": None if result.seat_gap_mm is None else round(result.seat_gap_mm, 4),
            "insert_shape": result.insert_shape,
            "insert_radius_mm": round(result.insert_radius_mm, 4),
            "steps": result.steps,
            "simulated_s": result.simulated_s,
            "wall_s": round(result.wall_s, 4),
            "sim_calls": 1,
            "approximation": result.model,
        }

    def _certificates(self, view: DesignView, result: SettleResult) -> list[FailureCertificate]:
        measurements = self._measurements(result)
        tolerances = result.model["tolerances"]
        certificates: list[FailureCertificate] = []
        for finding in result.findings:
            if finding == "penetration":
                certificates.append(
                    FailureCertificate(
                        code=CODE_DYNAMIC_PENETRATION,
                        family="geometry",
                        locus=self._locus(view, "contact_penetration"),
                        magnitude={
                            **measurements,
                            "value": measurements["penetration_mm"],
                            "unit": "mm",
                            "threshold": tolerances["penetration_mm"],
                            "direction": "gt",
                        },
                        observed_by=_observed_by(),
                        evidence_ref="mujoco contact.dist, deepest contact over the run",
                    )
                )
                continue
            reason = {
                "drift": "lateral_drift",
                "pull_out": "pull_out",
                "drop_through": "drop_through",
                "tilt": "tilt",
            }.get(finding, finding)
            value = {
                "drift": measurements["drift_mm"],
                "pull_out": measurements["pull_out_mm"],
                "drop_through": measurements["axial_mm"],
                "tilt": measurements["tilt_deg"],
            }.get(finding)
            threshold = {
                "drift": tolerances["drift_mm"],
                "pull_out": tolerances["pull_out_mm"],
                "drop_through": tolerances["pull_out_mm"],
                "tilt": tolerances["tilt_deg"],
            }.get(finding)
            certificates.append(
                FailureCertificate(
                    code=CODE_JOINT_UNSTABLE,
                    family="assembly",
                    locus=self._locus(view, reason),
                    magnitude={
                        **measurements,
                        "value": value,
                        "unit": "deg" if finding == "tilt" else "mm",
                        "threshold": threshold,
                        "direction": "gt",
                    },
                    observed_by=_observed_by(),
                    evidence_ref="mujoco 0.8 s settle, gravity -Z, local frame of the mate axis",
                )
            )
        return certificates

    def _declined(self, reason: str, detail: str) -> list[FailureCertificate]:
        """Decline to judge: a coverage note, no certificate (see `MODEL_BOUNDARY_REASONS`).

        The geometry is outside what a convex-primitive simulator can represent, so the
        honest answer is "not evaluated". It is deliberately *not* `physics_error`: that
        family means "the environment failed", the loop stops the episode on it, and an
        unevaluable insert would then block every case it appears in.
        """
        self.last_sim_calls = 0
        self.last_result = None
        self.last_error = ""
        self.last_note = (
            f"physics not evaluated: {reason}: {detail}"
            if detail
            else f"physics not evaluated: {reason}"
        )
        return []

    def _failed(self, view: DesignView, reason: str, detail: str) -> FailureCertificate:
        return FailureCertificate(
            code=CODE_PHYSICS_ERROR,
            family="environment",
            locus=self._locus(view, reason),
            magnitude={
                "reason": reason,
                "error": detail,
                "sim_calls": self.last_sim_calls,
                "unit": "mm",
                "approximation": {
                    "cavity": "N-gon of convex boxes",
                    "units": "mm in, m inside (x1e-3), mm out",
                    "engine": mujoco_settle.engine_version(),
                },
            },
            observed_by=_observed_by(),
            evidence_ref="physics_settle could not run; not a failure and not a pass",
        )

    # -- fixtures ----------------------------------------------------------
    def fixtures(self) -> tuple[list[DesignView], list[DesignView]]:
        """`(good, bad)` views this verifier builds itself; cached, they cost ~1 s."""
        return _fixture_pair()


# --------------------------------------------------------------------------- #
# Fixtures: hand-built solids, so the gate exists before any materialiser does
# --------------------------------------------------------------------------- #
def _spec(*, clearance_mm: float) -> DesignSpec:
    """A `peg_hole` spec whose numbers match the meshes `_peg_hole_meshes` builds."""
    d_peg = FIXTURE_PEG_MM
    d_hole = d_peg + 2.0 * clearance_mm
    return DesignSpec(
        spec_id=f"physics_fixture_c{clearance_mm:+.2f}".replace(".", "p"),
        requirement="physics settle fixture: a peg seated in a blind socket",
        parts=(
            PartSpec(id="body", role="body", size_mm=FIXTURE_PLATE_MM, pose_mm=(0.0, 0.0, 0.0)),
            PartSpec(
                id="peg",
                role="insert",
                size_mm=(d_peg, d_peg, FIXTURE_PEG_LEN_MM),
                pose_mm=(0.0, 0.0, FIXTURE_PLATE_MM[2] - FIXTURE_ENGAGE_MM),
            ),
        ),
        interfaces=(
            InterfaceSpec(
                id="iface_0",
                kind="peg_hole",
                a="body",
                b="peg",
                params={
                    "d_peg_mm": d_peg,
                    "d_hole_mm": d_hole,
                    "engage_mm": FIXTURE_ENGAGE_MM,
                    "peg_len_mm": FIXTURE_PEG_LEN_MM,
                },
            ),
        ),
        constraints=FitConstraints(),
    )


def _peg_hole_meshes(
    *, clearance_mm: float, engage_mm: float, through: bool
) -> tuple[trimesh.Trimesh, trimesh.Trimesh]:
    """A plate with a socket plus the peg that seats in it, in world millimetres.

    Seated pose: the hole opens on the plate's top face and the peg's bottom rests
    on the socket floor (blind), or 2 mm inside the top face (through). The blind
    socket is the build `peg_hole` uses: hole depth = engage, peg length =
    engage + 6, so the 6 mm handle sticks out.
    """
    bx, by, bz = FIXTURE_PLATE_MM
    body = trimesh.creation.box(extents=(bx, by, bz))
    body.apply_translation((0.0, 0.0, bz / 2.0))
    d_hole = FIXTURE_PEG_MM + 2.0 * clearance_mm
    depth = bz + 2.0 if through else engage_mm
    # +2 mm of cutter above the top face so the subtraction leaves a clean rim.
    hole = trimesh.creation.cylinder(radius=d_hole / 2.0, height=depth + 2.0, sections=96)
    hole.apply_translation((0.0, 0.0, bz - depth / 2.0 + 1.0))
    body = trimesh.boolean.difference([body, hole], engine="manifold")
    peg = trimesh.creation.cylinder(
        radius=FIXTURE_PEG_MM / 2.0, height=FIXTURE_PEG_LEN_MM, sections=96
    )
    seat_z = (bz - 2.0 if through else bz - engage_mm) + FIXTURE_PEG_LEN_MM / 2.0
    peg.apply_translation((0.0, 0.0, seat_z))
    return body, peg


def _view(*, clearance_mm: float, engage_mm: float, through: bool) -> DesignView:
    body, peg = _peg_hole_meshes(
        clearance_mm=clearance_mm, engage_mm=engage_mm, through=through
    )
    spec = _spec(clearance_mm=clearance_mm)
    return DesignView(
        spec=spec,
        meshes={spec.body.id: body, spec.insert.id: peg},
        mate_axis=mate_axis_for(spec),
        approach_mm=float(spec.interface.params.get("approach_mm", 6.0)),
    )


@functools.lru_cache(maxsize=1)
def _fixture_pair() -> tuple[list[DesignView], list[DesignView]]:
    """The fixture pair, built once: three boolean ops and three settles per process."""
    good = [_view(clearance_mm=0.15, engage_mm=FIXTURE_ENGAGE_MM, through=False)]
    bad = [
        _view(clearance_mm=2.0, engage_mm=FIXTURE_ENGAGE_MM, through=False),
        _view(clearance_mm=-0.05, engage_mm=FIXTURE_ENGAGE_MM, through=False),
        _view(clearance_mm=0.15, engage_mm=FIXTURE_ENGAGE_MM, through=True),
    ]
    return good, bad
