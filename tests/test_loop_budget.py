"""The loop's budget arithmetic: verification costs money, and a snapshot is judged once.

Two behaviours are pinned here, both of which decide whether an "equal budget"
comparison of two policies is fair (`docs/pah-dev-doc-v1.md` §9):

* a verifier cascade that reports its spend (`spent`) has that spend charged to the
  round it made possible, so an expensive cascade cannot look free;
* the same assembly snapshot is verified once, not twice — the loop used to judge a
  round's output and then judge that same snapshot again at the top of the next
  iteration, doubling the expensive tier for one answer.

The fakes below satisfy the `LoopEngine` / `LoopVerifiers` / `LoopRegistry` /
`LoopPolicy` protocols in `pah.policy.loop`. They are deliberately tiny: this file
tests the loop's arithmetic, not any geometry.
"""

from __future__ import annotations

from pah.core.model import (
    Assembly,
    Budget,
    Cost,
    FailureCertificate,
    Mandate,
    RepairPlan,
    Round,
    Ticket,
)
from pah.policy.loop import RepairLoop

# --------------------------------------------------------------------------- #
# Doubles
# --------------------------------------------------------------------------- #


class FakeEngine:
    """The four commands the loop uses, recorded so a test can read them back."""

    def __init__(self) -> None:
        self.rounds: dict[str, Round] = {}
        self.ended: tuple[str, str] | None = None
        self.refused: str | None = None
        self._n = 0

    def begin_round(self, ticket_id: str, plan: RepairPlan) -> Round:
        if self.refused:
            raise RuntimeError(self.refused)
        self._n += 1
        rnd = Round(id=f"r{self._n}", index=self._n - 1, plan=plan)
        self.rounds[rnd.id] = rnd
        return rnd

    def stage_round(self, round_id: str, stage: str) -> None:
        self.rounds[round_id].stage = stage

    def close_round(
        self,
        round_id: str,
        *,
        verdict: str,
        evidence: list[FailureCertificate],
        cost: Cost,
        artifacts: dict[str, str],
        edit_cost: float = 0.0,
        preserve_ratio: float = 1.0,
    ) -> Round:
        rnd = self.rounds[round_id]
        rnd.verdict = verdict
        rnd.evidence = list(evidence)
        rnd.cost = cost
        rnd.artifacts = dict(artifacts)
        rnd.edit_cost = edit_cost
        rnd.preserve_ratio = preserve_ratio
        return rnd

    def end_ticket(self, ticket_id: str, outcome: str, reason: str = "") -> None:
        self.ended = (outcome, reason)


def _assembly(version: int = 0) -> Assembly:
    return Assembly(
        id=f"a{version}",
        version=version,
        parts=(
            {
                "id": "body",
                "geom": {"hash": f"h{version}"},
            },
            {
                "id": "peg",
                "geom": {"hash": f"h{version}"},
            },
        ),
        joints=(),
        connectors=(),
        constraints={},
        parent=None if version == 0 else f"a{version - 1}",
    )


def _certificate(code: str = "fake_failure") -> FailureCertificate:
    return FailureCertificate(
        code=code,
        family="geometry",
        locus={"part": "peg"},
        magnitude={"mm": 0.5},
        observed_by="fake_verifier",
    )


class BumpRegistry:
    """One operator: it bumps the geometry hash, which is all the loop observes."""

    def __init__(self) -> None:
        self.applied = 0

    def applicable(self, asm: Assembly, certs: list[FailureCertificate]) -> list[str]:
        return ["bump"]

    def apply(self, asm: Assembly, plan: RepairPlan) -> Assembly:
        self.applied += 1
        return _assembly(asm.version + 1)


class AlwaysPropose:
    name = "always"

    def __init__(self, cost_est: Cost | None = None) -> None:
        self.cost_est = cost_est or Cost(geom_calls=1)

    def propose(self, asm, certs, history, budget_left):
        return RepairPlan(
            id=f"p{len(history)}",
            operator="bump",
            targets=("body",),
            params={},
            cost_est=self.cost_est,
            proposed_by="rule",
        )


class MeteredCascade:
    """A cascade that fails every snapshot below `healthy_at` and reports its spend."""

    def __init__(
        self,
        healthy_at: int = 99,
        *,
        per_check: Cost = Cost(geom_calls=2, sim_calls=1),
        as_method: bool = False,
    ) -> None:
        self.healthy_at = healthy_at
        self.per_check = per_check
        self.calls: list[str] = []
        self._spent = Cost()
        self._as_method = as_method

    def check(self, asm: Assembly) -> list[FailureCertificate]:
        self.calls.append(asm.id)
        self._spent = Cost(
            geom_calls=self._spent.geom_calls + self.per_check.geom_calls,
            sim_calls=self._spent.sim_calls + self.per_check.sim_calls,
        )
        if asm.version >= self.healthy_at:
            return []
        return [_certificate()]

    # The loop reads `spent` either as an attribute or as a method; both are valid
    # ports, and both must charge the same way.
    @property
    def spent(self) -> Cost:
        return self._spent


class MeteredCascadeAsMethod(MeteredCascade):
    """Same as above, but reports through a callable, which the loop also accepts."""

    def spent(self) -> Cost:  # type: ignore[override]
        return self._spent


class UnmeteredCascade(MeteredCascade):
    """A cascade with nothing to say about cost — the backwards-compatible case."""

    @property
    def spent(self):  # type: ignore[override]
        return None


def _ticket(rounds: int = 3, geom: int = 100, sim: int = 20) -> Ticket:
    return Ticket(
        id="t1",
        lineage="lineage-1",
        mandate=Mandate(
            strength="explicit",
            basis={},
            scope_parts=("body", "peg"),
            scope_operations=("bump",),
            frozen_parts=(),
            budget=Budget(rounds=rounds, geom_calls=geom, sim_calls=sim),
            policy_version="test@1",
        ),
    )


def _run(cascade, *, policy=None, ticket=None) -> tuple:
    engine = FakeEngine()
    registry = BumpRegistry()
    loop = RepairLoop(
        engine=engine,
        verifiers=cascade,
        registry=registry,
        policy=policy or AlwaysPropose(),
    )
    result = loop.run(ticket or _ticket(), _assembly())
    return result, engine, registry


# --------------------------------------------------------------------------- #
# Verification is charged to the round it made possible
# --------------------------------------------------------------------------- #


def test_verification_spend_is_charged_to_the_round():
    # a0 fails, a1 fails, a2 passes. Iteration 1 verifies a0 and a1 (4 geom, 2 sim),
    # iteration 2 verifies a2 (2 geom, 1 sim). Each round also declares 1 geom call.
    cascade = MeteredCascade(healthy_at=2)
    result, engine, _ = _run(cascade)

    assert result.verdict == "repaired"
    assert result.rounds == 2
    assert result.cost == Cost(geom_calls=8, sim_calls=3)

    charged = [rnd.cost for rnd in engine.rounds.values()]
    assert charged == [Cost(geom_calls=5, sim_calls=2), Cost(geom_calls=3, sim_calls=1)]


def test_a_cascade_without_a_meter_charges_only_the_estimate():
    cascade = UnmeteredCascade(healthy_at=1)
    result, engine, _ = _run(cascade)

    assert result.verdict == "repaired"
    assert result.cost == Cost(geom_calls=1, sim_calls=0)
    assert [rnd.cost for rnd in engine.rounds.values()] == [Cost(geom_calls=1)]


def test_a_meter_exposed_as_a_method_is_read_the_same_way():
    cascade = MeteredCascadeAsMethod(healthy_at=1)
    result, _, _ = _run(cascade)

    assert result.verdict == "repaired"
    assert result.cost == Cost(geom_calls=5, sim_calls=2)


def test_verification_spend_can_exhaust_the_budget():
    # a0 and a1 cost 2 geom calls each; two rounds of 1 + 4 geom calls exceed 5.
    cascade = MeteredCascade(healthy_at=99)
    result, engine, registry = _run(cascade, ticket=_ticket(rounds=5, geom=5))

    assert result.verdict == "exhausted"
    assert result.reason == "budget spent"
    assert engine.ended == ("exhausted", "budget spent")
    assert result.rounds == 1  # the second round could not be paid for
    assert registry.applied == 1


# --------------------------------------------------------------------------- #
# A snapshot is verified once, not twice
# --------------------------------------------------------------------------- #


def test_each_snapshot_is_verified_once():
    cascade = MeteredCascade(healthy_at=2)
    _run(cascade)

    # a0, a1, a2 — not a0, a1, a1, a2.
    assert cascade.calls == ["a0", "a1", "a2"]


def test_cache_reads_appear_in_the_trace_as_cached():
    events: list[tuple[str, dict]] = []
    cascade = MeteredCascade(healthy_at=99)
    engine = FakeEngine()
    loop = RepairLoop(
        engine=engine,
        verifiers=cascade,
        registry=BumpRegistry(),
        policy=AlwaysPropose(),
        observer=lambda kind, fields: events.append((kind, dict(fields))),
    )
    loop.run(_ticket(rounds=2), _assembly())

    verify_rows = [fields for kind, fields in events if kind == "verify"]
    # before/after of round 1, then round 2 reuses a1's verdict, verifies a2, and
    # the closing iteration reuses a2's verdict before seeing the budget is gone.
    assert [row["phase"] for row in verify_rows] == [
        "before",
        "after",
        "before",
        "after",
        "before",
    ]
    assert [bool(row.get("cached")) for row in verify_rows] == [
        False,
        False,
        True,
        False,
        True,
    ]


# --------------------------------------------------------------------------- #
# The charge is a floor, never a credit
# --------------------------------------------------------------------------- #


def test_a_meter_that_goes_backwards_cannot_refund_a_round():
    class Forgetful(MeteredCascade):
        def check(self, asm):  # noqa: D401 - a deliberately broken meter
            found = super().check(asm)
            self._spent = Cost()  # the meter resets itself: a broken tool
            return found

    result, engine, _ = _run(Forgetful(healthy_at=1))
    assert all(rnd.cost.geom_calls >= 1 for rnd in engine.rounds.values())
    assert result.cost.geom_calls >= 1
    # And the loop still finishes: a bad meter is not a reason to hang.
    assert result.verdict == "repaired"
