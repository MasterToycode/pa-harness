"""The L0 verifier contract: a registry whose entries must prove themselves.

`docs/pah-dev-doc-v1.md` §3.4 gives verifiers the only write access to a
`FailureCertificate`'s observed fields. §4.8 then draws the consequence: a
verifier that lies is the most expensive failure in the loop, because the loop
optimises its output. So the fixture check is a *runtime* gate here, not a
test-time habit — `Registry.sanity` runs every verifier's `known_good` /
`known_bad` fixtures and disables the ones that disagree with themselves, the
same way §4.8 requires before the loop may trust a number.

`Case` mirrors `case.json` (§8.1). It lives here rather than in `pah.core`
because M0 needs one field set wide — the fields a verifier reads — and widening
`pah.core` before the loop exists would fix the shape too early.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

from pah.core.model import Assembly, FailureCertificate

__all__ = [
    "Case",
    "CostClass",
    "Registry",
    "Verifier",
    "VerifierDisabled",
    "VerifierSpec",
]

CostClass = Literal["cheap", "medium", "expensive"]


@dataclass(frozen=True)
class Case:
    """One assembly plus the joints a verifier may judge (`case.json`, §8.1).

    `parts[i]["path"]` is decimated geometry — good for mass and for looking at,
    **never** for booleans or thickness (`docs/internal/adapter-contract.md`
    §E2). `parts[i]["pre_decimate"]` is the only legal boolean input.
    """

    case_id: str
    parts: tuple[dict, ...] = ()
    joints: tuple[dict, ...] = ()
    constraints: dict = field(default_factory=dict)
    scale_mm: float = 225.0
    scale_k: float = 0.7
    #: The frozen numbers the harness must reproduce, or its adapter is wrong.
    baseline: dict = field(default_factory=dict)


@dataclass(frozen=True)
class VerifierSpec:
    """What a verifier claims, so the kernel can route and gate it.

    `codes` is the closed set the verifier may emit (§3.4 rejects an unregistered
    code). `fixtures` carries at least one `known_good` and one `known_bad` item;
    `sanity` refuses a verifier that ships none, because an unproven verifier is
    indistinguishable from a lying one.
    """

    id: str
    codes: tuple[str, ...]
    cost_class: CostClass
    fixtures: dict


@runtime_checkable
class Verifier(Protocol):
    """A tool that may observe whether a case is physically acceptable.

    `check` returns the failures it observed and an empty list when it observed
    none. `assembly` lets a caller pass the design snapshot under test; M0
    verifiers judge the geometry on disk, so they accept and ignore it.
    """

    spec: VerifierSpec

    def case_of(self, item: dict) -> Case:
        """Materialise one fixture item into the single-edge case it names."""
        ...

    def check(
        self, case: Case, assembly: Assembly | None = None
    ) -> list[FailureCertificate]:
        """Every failure this verifier can prove about `case` — possibly none."""
        ...


class VerifierDisabled(RuntimeError):
    """The verifier failed its own fixtures and must not be cited."""


class Registry:
    """The verifier catalogue, plus the §4.8 gate that keeps it honest."""

    def __init__(self) -> None:
        self._verifiers: dict[str, Verifier] = {}
        self._disabled: set[str] = set()
        self._reasons: dict[str, str] = {}

    # -- registration ------------------------------------------------------

    def register(self, verifier: Verifier) -> None:
        """Adds `verifier`; a repeated id is a programming error, not a merge."""
        spec = verifier.spec
        if spec.id in self._verifiers:
            raise ValueError(f"verifier {spec.id!r} is already registered")
        self._verifiers[spec.id] = verifier

    def get(self, verifier_id: str) -> Verifier:
        """The verifier at `verifier_id`."""
        try:
            return self._verifiers[verifier_id]
        except KeyError:
            known = ", ".join(sorted(self._verifiers)) or "none"
            raise KeyError(f"no verifier {verifier_id!r}; registered: {known}") from None

    def ids(self) -> tuple[str, ...]:
        """Every registered id, sorted."""
        return tuple(sorted(self._verifiers))

    def disabled(self) -> frozenset[str]:
        """Ids whose fixtures disagreed with them on the last `sanity` run."""
        return frozenset(self._disabled)

    def why_disabled(self, verifier_id: str) -> str:
        """The fixture that failed, for the health report."""
        return self._reasons.get(verifier_id, "")

    # -- the §4.8 gate -----------------------------------------------------

    def sanity(self) -> dict[str, bool]:
        """Runs every verifier's fixtures; `False` means freshly disabled.

        A `known_good` item must produce **no** certificate and a `known_bad`
        item must produce **at least one**. Both halves matter: a verifier that
        never fires and a verifier that always fires are equally useless, and
        only a paired fixture catches either.
        """
        result: dict[str, bool] = {}
        for verifier_id, verifier in self._verifiers.items():
            reason = _first_disagreement(verifier)
            result[verifier_id] = reason == ""
            if reason:
                self._disabled.add(verifier_id)
                self._reasons[verifier_id] = reason
            else:
                self._disabled.discard(verifier_id)
                self._reasons.pop(verifier_id, None)
        return result


def _first_disagreement(verifier: Verifier) -> str:
    """The first fixture `verifier` gets wrong, as a sentence; "" when it agrees."""
    fixtures = verifier.spec.fixtures or {}
    good = fixtures.get("known_good") or ()
    bad = fixtures.get("known_bad") or ()
    if not good or not bad:
        return "ships no known_good/known_bad fixture pair"
    for item in good:
        found = verifier.check(verifier.case_of(item))
        if found:
            return (
                "known_good judged negative: "
                f"{_label(item)} -> {found[0].code}"
            )
    for item in bad:
        if not verifier.check(verifier.case_of(item)):
            return f"known_bad judged positive: {_label(item)}"
    return ""


def _label(item: dict) -> str:
    """A fixture item as one readable token."""
    return str(item.get("label") or item.get("case_id") or item)
