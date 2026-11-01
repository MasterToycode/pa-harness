"""What "unknown" means at the loop's boundary, and what a zero-round episode costs.

Both behaviours were forced by the first real episode that ran on a generated mesh:

* the printability verifier could not find the interface band on the generated
  insert, filled in an `environment` certificate, and the loop reported the design
  **repaired** — an unknown read as a pass, which is the one thing the whole evidence
  contract forbids;
* a clean design is verified once and ends with **zero rounds**, so the verification
  work was never charged and the report said the episode cost nothing.
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


class Engine:
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
        rnd.cost = cost
        return rnd

    def end_ticket(self, ticket_id: str, outcome: str, reason: str = "") -> None:
        self.ended = (outcome, reason)


class Nothing:
    def applicable(self, asm, certs):
        return []

    def apply(self, asm, plan):
        return asm


class NeverProposes:
    name = "never"

    def propose(self, asm, certs, history, budget_left):
        return None


def _assembly() -> Assembly:
    return Assembly(
        id="a0",
        version=0,
        parts=({"id": "body", "geom": {"hash": "h"}},),
        joints=(),
        connectors=(),
        constraints={},
    )


def _ticket(rounds: int = 3) -> Ticket:
    return Ticket(
        id="t",
        lineage="l",
        mandate=Mandate(
            strength="explicit",
            basis={},
            scope_parts=("body",),
            scope_operations=(),
            frozen_parts=(),
            budget=Budget(rounds=rounds, geom_calls=20, sim_calls=5),
            policy_version="test@1",
        ),
    )


def _run(cascade) -> tuple:
    engine = Engine()
    loop = RepairLoop(
        engine=engine, verifiers=cascade, registry=Nothing(), policy=NeverProposes()
    )
    return loop.run(_ticket(), _assembly()), engine


def test_an_environment_certificate_without_an_exception_is_not_a_pass():
    """Measured on the first generated-mesh episode: the cascade caught a verifier
    failure itself, returned an environment certificate, and the episode was reported
    `repaired`. Unknown must end the episode, not be counted as clean."""

    class EnvCascade:
        spent = Cost()

        def check(self, assembly):
            return [
                FailureCertificate(
                    code="verifier_error",
                    family="environment",
                    locus={"verifier": "printability"},
                    magnitude={"error": "no interface band"},
                    observed_by="cascade",
                )
            ]

    result, engine = _run(EnvCascade())

    assert result.verdict == "failed"
    assert result.reason == "verifier could not run"
    assert engine.ended == ("stopped", "verifier could not run")
    assert engine.rounds == []


def test_a_clean_design_still_reports_the_verification_it_spent():
    """Zero rounds is the most common success; reporting zero geometric work for it
    would make an expensive cascade look free."""

    class SpendyCascade:
        def __init__(self) -> None:
            self.spent = Cost()

        def check(self, assembly):
            self.spent = Cost(
                geom_calls=self.spent.geom_calls + 3, sim_calls=self.spent.sim_calls + 1
            )
            return []

    result, engine = _run(SpendyCascade())

    assert result.verdict == "repaired"
    assert engine.rounds == []
    assert result.cost == Cost(geom_calls=3, sim_calls=1)
