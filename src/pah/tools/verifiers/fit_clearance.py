"""`fit_clearance`: is the designed gap inside the band the task asked for?

What this verifier is allowed to look at
----------------------------------------
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

Known limitation
----------------
`DesignSpec.clearance_mm` resolves clearance through `builtin_params_model`, so a
plugin kind that is not one of the three builtins raises `KeyError` here. That is
deliberate: the cascade turns the exception into an `environment`-family
certificate, i.e. "unknown", never a pass. A plugin connector needs a clearance
source on the design plane before its fits can be judged.
"""

from __future__ import annotations

from pah.core.model import FailureCertificate
from pah.design.api import (
    CODE_EXCESSIVE_CLEARANCE,
    CODE_INSUFFICIENT_CLEARANCE,
    DesignView,
)

from .base import VerifierSpec

__all__ = ["CODES", "FitClearance"]

CODES: tuple[str, ...] = (CODE_INSUFFICIENT_CLEARANCE, CODE_EXCESSIVE_CLEARANCE)

#: Which family a band violation belongs to. `assembly`: the two parts do not fit
#: together as asked — it is a property of the joint, not of either solid, and it
#: is not a manufacturing limit (that is `printability`'s family).
_FAMILY = "assembly"


class FitClearance:
    """The gap the spec asks for, against the gap the constraints allow."""

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

    def fixtures(self) -> tuple[list[DesignView], list[DesignView]]:
        """The pair the §4.8 gate runs: one in-band design, one out on each side."""
        from ._design_fixtures import loose_fixture, oversized_fixture, standard_fixture

        return (
            [standard_fixture()],
            [oversized_fixture(), loose_fixture()],
        )

    def check(self, view: DesignView) -> list[FailureCertificate]:
        """One certificate when the clearance is outside the band, else none."""
        constraints = view.spec.constraints
        clearance = view.spec.clearance_mm
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
                magnitude={
                    "clearance_mm": clearance,
                    "min_mm": constraints.min_clearance_mm,
                    "max_mm": constraints.max_clearance_mm,
                    "target_fit": constraints.target_fit,
                    "deviation_mm": deviation,
                },
                observed_by=self.spec.id,
            )
        ]
