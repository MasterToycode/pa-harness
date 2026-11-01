"""Stop a build you started by mistake: the loop must have a brake.

The web page submits a generation to a paid API; without a stop that is a
commitment. The loop therefore asks a `cancel` callable once per round and ends with
`verdict="cancelled"` / ticket outcome `stopped`. What it can and cannot do is part
of the contract:

* it **stops further rounds** — the next generation is never submitted;
* it **cannot abort a call already in flight** — a submitted job is billed whether or
  not we wait for it, and Python cannot kill a thread that is blocked in HTTP.

`pah serve` exposes it as a button; the CLI gets it from Ctrl+C, which the runtime
turns into the same cancellation path.
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


def _assembly(version: int = 0) -> Assembly:
    return Assembly(
        id=f"a{version}",
        version=version,
        parts=({"id": "body", "geom": {"hash": f"h{version}"}},),
        joints=(),
        connectors=(),
        constraints={},
    )


class FailingVerifiers:
    spent = Cost()

    def check(self, asm):
        return [
            FailureCertificate(
                code="insufficient_material",
                family="geometry",
                locus={},
                magnitude={"volume_ratio": 0.29},
                observed_by="fake",
            )
        ]


class BumpRegistry:
    def __init__(self) -> None:
        self.applied = 0

    def applicable(self, asm, certs):
        return ["bump"]

    def apply(self, asm, plan):
        self.applied += 1
        return _assembly(asm.version + 1)


class AlwaysPropose:
    name = "always"

    def propose(self, asm, certs, history, budget_left):
        return RepairPlan(
            id=f"p{len(history)}",
            operator="bump",
            targets=("body",),
            params={},
            cost_est=Cost(geom_calls=1),
            proposed_by="rule",
        )


def _ticket(rounds: int = 5) -> Ticket:
    return Ticket(
        id="t",
        lineage="l",
        mandate=Mandate(
            strength="explicit",
            basis={},
            scope_parts=("body",),
            scope_operations=("bump",),
            frozen_parts=(),
            budget=Budget(rounds=rounds, geom_calls=50, sim_calls=10),
            policy_version="test@1",
        ),
    )


def _run(cancel) -> tuple:
    engine = Engine()
    loop = RepairLoop(
        engine=engine,
        verifiers=FailingVerifiers(),
        registry=BumpRegistry(),
        policy=AlwaysPropose(),
        cancel=cancel,
    )
    return loop.run(_ticket(), _assembly()), engine


# --------------------------------------------------------------------------- #
# the brake
# --------------------------------------------------------------------------- #
def test_cancelling_before_the_first_round_spends_nothing():
    result, engine = _run(cancel=lambda: True)

    assert result.verdict == "cancelled"
    assert result.reason == "cancelled by request"
    assert result.rounds == 0
    assert engine.rounds == []
    # The engine's vocabulary for "a person ended this" is `stopped`, not `dropped`.
    assert engine.ended == ("stopped", "cancelled by request")


def test_cancelling_after_one_round_keeps_that_round_and_stops():
    calls = {"count": 0}

    def cancel() -> bool:
        # False for the first check, True afterwards: the episode gets one round.
        calls["count"] += 1
        return calls["count"] > 1

    result, engine = _run(cancel=cancel)

    assert result.verdict == "cancelled"
    assert result.rounds == 1
    assert len(engine.rounds) == 1
    assert engine.rounds[0].verdict == "fail"


def test_without_a_cancel_callable_nothing_changes():
    result, engine = _run(cancel=None)

    assert result.verdict == "exhausted"
    assert len(engine.rounds) == 5


# --------------------------------------------------------------------------- #
# the page's side of it
# --------------------------------------------------------------------------- #
def test_the_page_can_ask_a_running_job_to_stop():
    from pah.cli.serve import JobState

    job = JobState()
    job.begin()
    job.cancel()

    snapshot = job.snapshot()
    assert snapshot["cancelling"] is True
    assert job.cancel_event.is_set()
    assert any("cancel requested" in line for line in snapshot["log"])

    job.finish(result={"verdict": "cancelled"})
    assert job.snapshot()["state"] == "done"
    assert job.snapshot()["cancelling"] is False


def test_cancelling_an_idle_job_does_nothing():
    from pah.cli.serve import JobState

    job = JobState()
    job.cancel()

    assert job.snapshot()["state"] == "idle"
    assert job.cancel_event.is_set() is False
