"""The deterministic policy: read the evidence, pick the cheapest legal action.

This is the `RulePolicy` of `docs/internal/repair-loop-spec.md` §2.3 — the P0
policy, and the baseline every later policy is measured against. It is
deliberately dumb: no model, no search, no randomness. If a rule-based policy
cannot close the loop, a learned one has nothing to close.

Two properties earn their keep:

- **It never invents an action.** Every candidate comes from the operator
  registry, and an operator whose preconditions fail is skipped.
- **It does not repeat itself.** An operator already tried on the same evidence
  for the same locus is skipped, so a failing plan cannot be resubmitted until
  the budget runs out — that is what the cycle guard would otherwise have to
  catch after the fact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pah.core.ids import key
from pah.core.model import Assembly, Budget, Cost, FailureCertificate, RepairPlan, Round

__all__ = ["DEFAULT_RULES", "RulePolicy", "OperatorCatalogue"]

#: Failure code -> operator ids, cheapest useful first. Only operators that are
#: actually registered survive `applicable`; an absent one is skipped, not an error.
DEFAULT_RULES: dict[str, tuple[str, ...]] = {
    "gap_too_wide": ("resize_connector", "relocate_connector", "swap_family"),
    "no_material_hit": ("relocate_connector", "resize_connector", "swap_family"),
    "unreatable_too_small": ("swap_family", "relocate_connector"),
    "wall_too_thin": ("resize_connector", "swap_family"),
    "interpenetration": ("relocate_connector", "resize_connector"),
    "insertion_blocked": ("relocate_connector", "swap_family"),
    "socket_breaks_wall": ("resize_connector", "relocate_connector"),
    "peg_detached": ("relocate_connector", "swap_family"),
    "torsion_free": ("swap_family", "relocate_connector"),
    "degenerate_other": ("relocate_connector", "swap_family"),
}

#: A deterministic in-plane spiral, in mm, for the relocating operator. Handing
#: the registry a fixed ladder rather than a random offset is what makes two runs
#: of the same case identical.
_OFFSETS: tuple[tuple[float, float], ...] = (
    (0.0, 0.0),
    (0.5, 0.0),
    (0.0, 0.5),
    (-0.5, 0.0),
    (0.0, -0.5),
    (0.5, 0.5),
    (-0.5, -0.5),
)


class OperatorCatalogue(Protocol):
    """The slice of the registry this policy uses."""

    def applicable(self, asm: Assembly, certs: list[FailureCertificate]) -> list[str]: ...

    def get(self, operator_id: str): ...


@dataclass
class RulePolicy:
    """Deterministic repair decisions from structured evidence."""

    registry: OperatorCatalogue
    name: str = "rule"
    rules: dict[str, tuple[str, ...]] = None  # type: ignore[assignment]
    #: Declared cost of one rule-based round, used for the loop's budget preview.
    geom_calls_per_round: int = 1

    def __post_init__(self) -> None:
        if self.rules is None:
            self.rules = dict(DEFAULT_RULES)

    def propose(
        self,
        assembly: Assembly,
        certs: list[FailureCertificate],
        history: list[Round],
        budget_left: Budget,
    ) -> RepairPlan | None:
        """The first legal, untried action for the strongest piece of evidence.

        Evidence order is the verifier's: the cascade already ran cheapest-first,
        so the first certificate names the failure that is cheapest to attack.
        """
        tried = _tried(history)
        applicable = set(self.registry.applicable(assembly, certs))

        for cert in certs:
            for operator_id in self.rules.get(cert.code, ()):
                if operator_id not in applicable:
                    continue
                params = self._params(operator_id, cert, assembly, tried)
                if params is None:
                    continue
                signature = (operator_id, cert.code, _locus_key(cert), _params_key(params))
                if signature in tried:
                    continue
                return RepairPlan(
                    id=key("plan", operator_id, cert.code, _locus_key(cert), _params_key(params)),
                    operator=operator_id,
                    targets=tuple(params.get("parts", ())) or tuple(cert.locus.get("parts", ())),
                    params=params,
                    cost_est=Cost(geom_calls=self.geom_calls_per_round),
                    proposed_by="rule",
                    reason=f"{cert.code} -> {operator_id}",
                )
        return None

    # -- parameter construction -------------------------------------------

    def _params(
        self,
        operator_id: str,
        cert: FailureCertificate,
        assembly: Assembly,
        tried: set[tuple],
    ) -> dict | None:
        locus = cert.locus if isinstance(cert.locus, dict) else {}
        joint = locus.get("joint")
        parts = tuple(locus.get("parts", ()))
        if not joint or len(parts) != 2:
            # Without a joint and its two parts the registry cannot address the
            # change, so this certificate yields no action rather than a guess.
            return None
        base = {"joint_id": str(joint), "parts": parts}

        if operator_id == "resize_connector":
            gap = _number(cert.magnitude, "value")
            if gap is None:
                return None
            return {**base, "gap_mm": float(gap)}

        if operator_id == "relocate_connector":
            return {**base, **self._next_offset(base, cert, tried)}

        if operator_id == "swap_family":
            return {**base, "section": "D"}

        return None

    def _next_offset(
        self, base: dict, cert: FailureCertificate, tried: set[tuple]
    ) -> dict:
        """The first offset in the ladder not already attempted at this locus.

        Detecting the repeat here (rather than after the geometry came back the
        same) saves a round of budget for every retry.
        """
        for du, dv in _OFFSETS:
            params = {**base, "du_mm": du, "dv_mm": dv}
            if ("relocate_connector", cert.code, _locus_key(cert), _params_key(params)) not in tried:
                return {"du_mm": du, "dv_mm": dv}
        du, dv = _OFFSETS[-1]
        return {"du_mm": du, "dv_mm": dv}


# -------------------------------------------------------------------------- #


def _tried(history: list[Round]) -> set[tuple]:
    return {
        (r.plan.operator, r.plan.reason.split(" -> ")[0], *_locus_and_params(r.plan))
        for r in history
    }


def _locus_and_params(plan: RepairPlan) -> tuple[str, str]:
    return _locus_key_from_plan(plan), _params_key(dict(plan.params))


def _locus_key(cert: FailureCertificate) -> str:
    locus = cert.locus if isinstance(cert.locus, dict) else {}
    return key("locus", locus.get("joint"), locus.get("parts"))


def _locus_key_from_plan(plan: RepairPlan) -> str:
    return key("locus", plan.params.get("joint_id"), tuple(plan.targets))


def _params_key(params: dict) -> str:
    return key("params", sorted((str(k), str(v)) for k, v in params.items()))


def _number(mapping: object, field: str) -> float | None:
    if isinstance(mapping, dict):
        value = mapping.get(field)
        if isinstance(value, (int, float)):
            return float(value)
    return None
