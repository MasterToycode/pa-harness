"""`fit_clearance`: is the gap the task asked for, and can the process hold it?

What this verifier is allowed to look at
---------------------------------------
Only the `DesignSpec`: the connector kind, its parameters, and `FitConstraints`.
It reads the clearance through `DesignSpec.clearance_mm`, which delegates to
`design/spec.py:radial_clearance_mm` — the one place that knows how each kind
spells clearance (`(d_hole - d_peg) / 2` for a peg, `clearance_mm` per side for a
dovetail, and so on). Re-deriving that arithmetic here is how two modules start
judging two different numbers for one design.

That is also why this verifier is `cheap` and never touches a triangle. There are
two questions a fit can raise, and they belong to two different verifiers:

* **did we ask for a sane gap?** — intent, judged here from the numbers;
* **do the built solids actually overlap?** — geometry, judged by
  `interface_interference` from the meshes.

So a 10.0 mm pin in a 10.0 mm hole is `insufficient_clearance` (a press fit where
the task asked for a slip fit) and *not* `geometric_interference` (zero clearance
is contact, not overlap). A 10.2 mm pin is both. Keeping the two codes apart is
what lets a report say whether the agent fixed the intent or the geometry.

The verifier says nothing about which number the band "should" be. `FitConstraints`
is a design input (`design/spec.py`), and a verifier that second-guessed it would
be inventing requirements — the loop optimises whatever the verifiers say.

The capability floor: `not_manufacturable` (audit P2-3)
-------------------------------------------------------
A band violation is a *design* statement. It says nothing about whether the
process can hold that gap at all, and a report that only ever says "the gap is
outside [0.10, 0.60]" cannot explain why a 0.02 mm slip fit was never going to
work on an FDM machine. So the verifier asks the second question and uses a
second code:

* `insufficient_clearance` / `excessive_clearance` — **design choice**: the gap is
  not what the task asked for;
* `not_manufacturable` — **process capability**: no gap this tight survives the
  process's own error band, whatever the task asked for. It fires *instead of*
  `insufficient_clearance`, because "make it tighter" is not a repair when the
  printer cannot resolve the gap.

Both families are physical (`assembly` for the band, `manufacturing` for the
capability), so a policy sees both as hard evidence and neither falls into the
`environment` bucket.

Why only a *positive* clearance can be not manufacturable: a zero or negative
clearance is a press fit — a deliberate intent (a dowel is *made* to interfere),
and the band is the right question to ask about it. The code is about a gap that
is meant to open but is too small to guarantee; firing it on a press fit would
relabel an intent as a capability failure.

The floor comes from `design/process.py:achievable_clearance_mm`, which derives
it from the process's own tolerance and feature floor and returns the sentence a
report prints as the band's source (C3). Everything the report needs is also
recorded on `last_capability` after every `check` — including for a *clean*
design, which produces no certificate at all, so a report that wants to print
"designed 0.20, process floor 0.05 (fdm)" on a passing fit has to read that
attribute rather than the evidence list.

An unknown process is *not evaluated*, the rule P2-1 set for a missing profile:
`profile_for` returns `None`, `last_note` says why, and only the band codes can
fire. The capability question is skipped, never answered with borrowed FDM
numbers.

`post_process` is the advice half of the audit: a tight fit on a layered process
is usually made by finishing the bore after printing, and a certificate that only
says "too tight" leaves a policy with nothing to propose besides editing the
spec.

Known limitation
----------------
`DesignSpec.clearance_mm` resolves clearance through `builtin_params_model`, so a
plugin kind that is not one of the three builtins raises `KeyError` here. That is
deliberate: the cascade turns the exception into an `environment`-family
certificate, i.e. "unknown", never a pass. A plugin connector needs a clearance
source on the design plane before its fits can be judged.
"""

from __future__ import annotations

import math

from pah.core.model import FailureCertificate
from pah.design.api import (
    CODE_EXCESSIVE_CLEARANCE,
    CODE_INSUFFICIENT_CLEARANCE,
    CODE_NOT_MANUFACTURABLE,
    DesignView,
)
from pah.design.process import (
    CLEARANCE_ERROR_FRACTION,
    ProcessProfile,
    achievable_clearance_mm,
    profile_for,
)

from .base import VerifierSpec

__all__ = ["CODES", "FitClearance"]

CODES: tuple[str, ...] = (
    CODE_INSUFFICIENT_CLEARANCE,
    CODE_EXCESSIVE_CLEARANCE,
    CODE_NOT_MANUFACTURABLE,
)

#: Which family a band violation belongs to. `assembly`: the two parts do not fit
#: together as asked — it is a property of the joint, not of either solid, and it
#: is not a manufacturing limit (that is `printability`'s family).
_FAMILY = "assembly"

#: A capability floor is a *manufacturing* limit, the same family `below_min_wall`
#: uses: the design is fine, the process is not. The two families must stay
#: different, or a report cannot separate "we asked for the wrong fit" from "we
#: asked for a fit this process cannot make".
_MANUFACTURING_FAMILY = "manufacturing"

#: Where each builtin kind keeps the size of the insert's public feature — the
#: dimension the process has to hold a tolerance on. It is the smaller side of the
#: pair for all three builtins (a peg is smaller than its socket, a hook than its
#: slot, a tongue no wider than its groove), so it is also the conservative pick.
#: The names live here rather than being imported from the connector package on
#: purpose: a verifier must be able to judge geometry it did not build (the same
#: reason `printability` keeps its own copy of the process table).
_FEATURE_FIELD: dict[str, str] = {
    "peg_hole": "d_peg_mm",
    "snap_fit": "hook_w_mm",
    "dovetail": "width_mm",
}

#: What to do about a fit that is at the edge of the process. Kept next to the
#: check because the advice depends on the same profile the floor came from.
_POST_PROCESS: dict[str, str] = {
    "fdm": (
        "ream or drill the bore after printing: FDM bores print undersized, and a "
        "reamed bore holds a fit the layers cannot"
    ),
    "sla": (
        "print the bore oversize and finish with a reamer; cured resin shrinks and "
        "creeps, so an as-printed resin fit is not a fit"
    ),
    "sls": "ream or hone the bore; powder-bed bores print tight and rough",
    "mjf": "ream or hone the bore; powder-bed bores print tight and rough",
    "cnc": (
        "bore the hole with a tool that holds the tolerance; the limit here is "
        "tooling, not the machine"
    ),
}

_DEFAULT_POST_PROCESS = (
    "finish the mating surfaces after the primary process; the printed dimension "
    "is not the fit dimension"
)


class FitClearance:
    """The gap the spec asks for, against the band and against the process."""

    spec = VerifierSpec(
        id="fit_clearance",
        codes=CODES,
        cost_class="cheap",
        fixtures={
            "known_good": ["fixture-standard: 9.9 mm pin / 10.0 mm bore, 0.05 mm clearance"],
            "known_bad": [
                "fixture-oversized: 10.2 mm pin / 10.0 mm bore, -0.10 mm clearance",
                "fixture-loose: 8.0 mm pin / 10.0 mm bore, 1.00 mm clearance",
            ],
        },
    )

    def __init__(self) -> None:
        #: Why the last `check` declined to evaluate something, or "" when it
        #: evaluated everything it could. The cascade aggregates these into
        #: `DesignVerifierCascade.notes`; a capability boundary is not a tool
        #: failure, so it is never dressed up as an `environment` certificate.
        self.last_note = ""
        #: The tolerance-chain facts of the last `check`, or `None` when no
        #: process profile (or no feature size) let the capability be judged. It
        #: is recorded on the *instance* because a clean design emits no
        #: certificate, and a report still has to print the band's source (C3).
        self.last_capability: dict | None = None

    def fixtures(self) -> tuple[list[DesignView], list[DesignView]]:
        """The pair the §4.8 gate runs: one in-band design, one out on each side."""
        from ._design_fixtures import loose_fixture, oversized_fixture, standard_fixture

        return (
            [standard_fixture()],
            [oversized_fixture(), loose_fixture()],
        )

    # -- check -------------------------------------------------------------
    def check(self, view: DesignView) -> list[FailureCertificate]:
        """Capability first: a gap the process cannot hold is not a band question."""
        self.last_note = ""
        self.last_capability = None
        constraints = view.spec.constraints
        clearance = view.spec.clearance_mm

        capability = self._capability(view, clearance)
        if capability is not None and self.last_capability is not None:
            best, worst, note = capability
            if _below_capability(clearance, best):
                return [
                    self._not_manufacturable(
                        view, clearance, best, worst, note, facts=self.last_capability
                    )
                ]

        return self._band(view, clearance)

    # -- capability --------------------------------------------------------
    def _capability(
        self, view: DesignView, clearance: float
    ) -> tuple[float, float, str] | None:
        """`(best, worst, source)` for this process and feature, or `None`.

        `None` means "not evaluated", and `last_note` says which half was missing
        — an unknown process name or a connector kind with no public feature size.
        The band check still runs in that case; the capability claim is simply not
        made.
        """
        constraints = view.spec.constraints
        process = constraints.process
        profile = profile_for(process)
        if profile is None:
            self.last_note = (
                "fit manufacturability not evaluated: no process profile for "
                f"{process!r}"
            )
            return None

        feature = self._feature_mm(view)
        if feature is None:
            return None

        best, worst, note = achievable_clearance_mm(profile, float(feature))
        self.last_capability = {
            "designed_clearance_mm": clearance,
            "process_best_clearance_mm": best,
            "process_worst_clearance_mm": worst,
            # The per-side error this design is expected to see; a report prints it
            # beside the designed gap so the two numbers cannot be confused.
            "expected_tolerance_mm": best,
            "realized_clearance_mm": [clearance - best, clearance + best],
            "band_mm": [constraints.min_clearance_mm, constraints.max_clearance_mm],
            "process": process,
            "feature_mm": feature,
            "tolerance_mm": profile.tolerance_mm,
            "min_feature_mm": profile.min_feature_mm,
            "clearance_error_fraction": CLEARANCE_ERROR_FRACTION,
            "manufacturable": not _below_capability(clearance, best),
            "post_process": _post_process(profile),
            "source": note,
        }
        return best, worst, note

    def _feature_mm(self, view: DesignView) -> float | None:
        """The insert's public feature size, or `None` (with a note) if unknown."""
        kind = view.spec.interface.kind
        field = _FEATURE_FIELD.get(kind)
        if field is None:
            self.last_note = (
                "fit manufacturability not evaluated: connector kind "
                f"{kind!r} declares no public feature size to hold a tolerance on"
            )
            return None
        value = getattr(view.spec.interface.validated_params(), field, None)
        if value is None:
            self.last_note = (
                "fit manufacturability not evaluated: connector kind "
                f"{kind!r} has no {field!r} in its parameters"
            )
            return None
        return float(value)

    def _not_manufacturable(
        self,
        view: DesignView,
        clearance: float,
        best: float,
        worst: float,
        note: str,
        *,
        facts: dict,
    ) -> FailureCertificate:
        """The certificate that says the process cannot hold this gap.

        `facts` is the recorded capability of this check (`last_capability`); the
        tolerance provenance in it is what C3 asks the evidence to carry, so the
        certificate repeats it rather than making a reader re-run the chain.
        """
        constraints = view.spec.constraints
        interface = view.spec.interface
        return FailureCertificate(
            code=CODE_NOT_MANUFACTURABLE,
            family=_MANUFACTURING_FAMILY,
            locus={
                "spec": view.spec.spec_id,
                "revision": view.spec.revision,
                "interface": interface.id,
                "parts": [interface.a, interface.b],
                "kind": interface.kind,
                "process": constraints.process,
            },
            magnitude={
                "designed_clearance_mm": clearance,
                "process_best_clearance_mm": best,
                "process_worst_clearance_mm": worst,
                "expected_tolerance_mm": best,
                "realized_clearance_mm": [clearance - best, clearance + best],
                "band_mm": [constraints.min_clearance_mm, constraints.max_clearance_mm],
                "process": constraints.process,
                "feature_mm": float(facts["feature_mm"]),
                "tolerance_mm": float(facts["tolerance_mm"]),
                "min_feature_mm": float(facts["min_feature_mm"]),
                "clearance_error_fraction": CLEARANCE_ERROR_FRACTION,
                "target_fit": constraints.target_fit,
                "post_process": facts["post_process"],
                "source": note,
                "method": "tolerance_chain_from_process_profile",
            },
            observed_by=self.spec.id,
            evidence_ref="tolerance_chain",
        )

    # -- band --------------------------------------------------------------
    def _band(self, view: DesignView, clearance: float) -> list[FailureCertificate]:
        """The design-intent check: is the gap inside the band the task asked for?"""
        constraints = view.spec.constraints
        if constraints.in_band(clearance):
            return []

        low = clearance < constraints.min_clearance_mm
        code = CODE_INSUFFICIENT_CLEARANCE if low else CODE_EXCESSIVE_CLEARANCE
        # Signed distance to the band edge it missed, so a policy can size its
        # `set_clearance`/`resize_feature` move without recomputing the band.
        deviation = (
            clearance - constraints.min_clearance_mm
            if low
            else clearance - constraints.max_clearance_mm
        )
        interface = view.spec.interface
        magnitude: dict = {
            "clearance_mm": clearance,
            "min_mm": constraints.min_clearance_mm,
            "max_mm": constraints.max_clearance_mm,
            "target_fit": constraints.target_fit,
            "deviation_mm": deviation,
            "process": constraints.process,
            "post_process": _post_process(profile_for(constraints.process)),
        }
        if self.last_capability is not None:
            # The capability facts travel with the band finding too: a reader must
            # not have to guess whether the gap was merely off-band or unmakeable.
            for key in (
                "process_best_clearance_mm",
                "process_worst_clearance_mm",
                "expected_tolerance_mm",
                "realized_clearance_mm",
                "feature_mm",
                "source",
            ):
                magnitude[key] = self.last_capability[key]
        return [
            FailureCertificate(
                code=code,
                family=_FAMILY,
                locus={
                    "spec": view.spec.spec_id,
                    "revision": view.spec.revision,
                    "interface": interface.id,
                    "parts": [interface.a, interface.b],
                    "kind": interface.kind,
                },
                magnitude=magnitude,
                observed_by=self.spec.id,
            )
        ]


def _below_capability(clearance: float, best_mm: float) -> bool:
    """Whether a designed clearance is positive but under the process's floor.

    The positive half is the press-fit exclusion: a zero or negative clearance is
    an intent, and the band is the right question to ask about it. The comparison
    carries a relative slack because the clearance is computed as
    `(d_hole - d_peg) / 2`, so a design sitting exactly on the floor arrives a few
    1e-16 short of it (`(10.0 - 9.9) / 2 == 0.04999999999999982`); without the
    slack, binary arithmetic would call a fit at the process's stated capability
    unmakeable.
    """
    if clearance <= 0.0:
        return False
    if math.isclose(clearance, best_mm, rel_tol=1e-9, abs_tol=1e-12):
        return False
    return clearance < best_mm


def _post_process(profile: ProcessProfile | None) -> str:
    """The finishing advice for a tight fit on this process."""
    if profile is None:
        return _DEFAULT_POST_PROCESS
    return _POST_PROCESS.get(profile.name, _DEFAULT_POST_PROCESS)
