"""Resume fidelity for the loop's cycle detection.

Cycle detection is "have I been here before". A process that restarts only knows
the snapshot in front of it, so it walks back into geometry the first run already
ruled a cycle — and the two runs then disagree on how many rounds the episode
took, which is exactly the acceptance check that a kill-and-resume must pass.

`RepairLoop.run(..., seen=...)` is the port for that memory. These tests pin the
mechanism: the same restart converges with the history and diverges without it.
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
from pah.policy.loop import RepairLoop, assembly_fingerprint


class FakeEngine:
    def __init__(self) -> None:
        self.rounds: list[Round] = []
        self.ended: tuple[str, str] | None = None

    def begin_round(self, ticket_id: str, plan: RepairPlan) -> Round:
        rnd = Round(id=f"r{len(self.rounds)}", index=len(self.rounds), plan=plan)
        self.rounds.append(rnd)
        return rnd

    def stage_round(self, round_id: str, stage: str) -> None:
        return None

    def close_round(self, round_id, *, verdict, evidence, cost, artifacts, edit_cost=0.0, preserve_ratio=1.0):
        rnd = next(r for r in self.rounds if r.id == round_id)
        rnd.verdict = verdict
        rnd.evidence = list(evidence)
        rnd.cost = cost
        return rnd

    def end_ticket(self, ticket_id: str, outcome: str, reason: str = "") -> None:
        self.ended = (outcome, reason)


def _assembly(version: int) -> Assembly:
    """Two snapshots that the registry ping-pongs between, both always failing."""
    return Assembly(
        id=f"a{version}",
        version=version,
        parts=({"id": "body", "geom": {"hash": f"h{version}"}},),
        joints=(),
        connectors=(),
        constraints={},
    )


def _cert() -> FailureCertificate:
    return FailureCertificate(
        code="fake_failure",
        family="geometry",
        locus={},
        magnitude={},
        observed_by="fake",
    )


class PingPongRegistry:
    """Applies to the *other* snapshot, so the pair is a guaranteed cycle."""

    def applicable(self, asm, certs):
        return ["swap"]

    def apply(self, asm, plan):
        return _assembly(1 if asm.version == 0 else 0)


class EndlessPolicy:
    name = "endless"

    def propose(self, asm, certs, history, budget_left):
        return RepairPlan(
            id=f"p{len(history)}",
            operator="swap",
            targets=("body",),
            params={},
            cost_est=Cost(geom_calls=1),
            proposed_by="rule",
        )


class FailingVerifiers:
    def check(self, asm):
        return [_cert()]


def _ticket(rounds: int = 9) -> Ticket:
    return Ticket(
        id="t1",
        lineage="lineage-1",
        mandate=Mandate(
            strength="explicit",
            basis={},
            scope_parts=("body",),
            scope_operations=("swap",),
            frozen_parts=(),
            budget=Budget(rounds=rounds, geom_calls=100, sim_calls=10),
            policy_version="test@1",
        ),
    )


def _run(assembly: Assembly, *, seen=()) -> tuple:
    engine = FakeEngine()
    loop = RepairLoop(
        engine=engine,
        verifiers=FailingVerifiers(),
        registry=PingPongRegistry(),
        policy=EndlessPolicy(),
    )
    result = loop.run(_ticket(), assembly, seen=seen)
    return result, engine


def test_an_uninterrupted_run_stops_at_the_cycle_after_two_rounds():
    result, engine = _run(_assembly(0))

    assert result.verdict == "infeasible"
    assert result.reason == "geometry cycle"
    assert len(engine.rounds) == 2


def test_a_restart_with_the_seen_history_reproduces_the_remaining_round_only():
    """The first run's trace said a0 was seen; the restart must not walk back into it."""
    result, engine = _run(_assembly(1), seen=[assembly_fingerprint(_assembly(0))])

    assert result.verdict == "infeasible"
    assert result.reason == "geometry cycle"
    assert len(engine.rounds) == 1  # exactly the round the uninterrupted run still owed


def test_a_restart_without_the_history_diverges():
    """Without `seen` the same restart spends an extra round rediscovering the cycle.
    This is the bug the port exists to prevent, kept as an executable reminder."""
    result, engine = _run(_assembly(1))

    assert result.verdict == "infeasible"
    assert len(engine.rounds) == 2  # one more than the uninterrupted run owed
