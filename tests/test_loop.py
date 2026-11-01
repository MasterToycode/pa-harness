"""Contract tests for the repair loop.

What is locked here is the loop's *shape*, not any dataset's success rate: the
four stop conditions, that a broken tool cannot end an episode, that a budget is
a ceiling rather than a target, and that every step is observed.
"""

from __future__ import annotations

import pytest

from pah.core.model import (
    Assembly,
    Attention,
    Budget,
    Cost,
    FailureCertificate,
    Mandate,
    RepairPlan,
    Round,
    Ticket,
)
from pah.policy.loop import RepairLoop, assembly_fingerprint, hard_failures

MANDATE = Mandate(
    strength="explicit",
    basis={"via": "probe"},
    scope_parts=("p1",),
    scope_operations=("resize_interface", "relocate_anchor"),
    frozen_parts=("p0",),
    budget=Budget(rounds=5, geom_calls=100, sim_calls=20),
    policy_version="pv-test",
)


def make_ticket(rounds: int = 5) -> Ticket:
    mandate = Mandate(
        strength=MANDATE.strength,
        basis=MANDATE.basis,
        scope_parts=MANDATE.scope_parts,
        scope_operations=MANDATE.scope_operations,
        frozen_parts=MANDATE.frozen_parts,
        budget=Budget(rounds=rounds, geom_calls=100, sim_calls=20),
        policy_version=MANDATE.policy_version,
    )
    return Ticket(id="t1", lineage="asm:1", mandate=mandate, created_at="2026-01-01T00:00:00")


def make_assembly(connector_radius: float = 0.5) -> Assembly:
    return Assembly(
        id="asm_1",
        version=0,
        parts=({"id": "p0", "geom": {"hash": "h0"}}, {"id": "p1", "geom": {"hash": "h1"}}),
        joints=({"id": "j0", "a": "p0", "b": "p1"},),
        connectors=({"id": "c0", "family": "pin", "params": {"r": connector_radius}},),
        constraints={"process": "SLA"},
    )


def cert(code: str = "gap_too_wide", family: str = "assembly") -> FailureCertificate:
    return FailureCertificate(
        code=code,
        family=family,  # type: ignore[arg-type]
        locus={"parts": ["p0", "p1"]},
        magnitude={"value": 0.6, "unit": "mm"},
        observed_by="fake_verifier",
    )


def plan(operator: str = "resize_interface", radius: float = 0.7, **cost) -> RepairPlan:
    return RepairPlan(
        id=f"plan-{operator}-{radius}",
        operator=operator,
        targets=("p1",),
        params={"r": radius},
        cost_est=Cost(**cost) if cost else Cost(geom_calls=1),
    )


# -- test doubles ----------------------------------------------------------


class FakeEngine:
    """Records commands and enforces only the budget — scope rules are the real
    engine's job and are tested there."""

    def __init__(self, budget: Budget | None = None) -> None:
        self.budget = budget
        self.rounds: list[Round] = []
        self.ended: list[tuple[str, str]] = []
        self.staged: list[tuple[str, str]] = []
        self.calls: list[str] = []

    def begin_round(self, ticket_id: str, plan_: RepairPlan) -> Round:
        self.calls.append("begin_round")
        round_ = Round(id=f"r{len(self.rounds)}", index=len(self.rounds), plan=plan_)
        self.rounds.append(round_)
        return round_

    def stage_round(self, round_id: str, stage: str) -> None:
        self.calls.append(f"stage:{stage}")
        self.staged.append((round_id, stage))

    def close_round(self, round_id, *, verdict, evidence, cost, artifacts, edit_cost=0.0, preserve_ratio=1.0):
        self.calls.append("close_round")
        for round_ in self.rounds:
            if round_.id == round_id:
                round_.verdict = verdict  # type: ignore[assignment]
                round_.evidence = list(evidence)
                round_.cost = cost
                round_.artifacts = dict(artifacts)
                return round_
        raise AssertionError(f"unknown round {round_id}")

    def end_ticket(self, ticket_id: str, outcome: str, reason: str = "") -> Ticket:
        self.calls.append("end_ticket")
        self.ended.append((outcome, reason))
        return make_ticket()


class FakeVerifiers:
    """Returns the queued certificate lists in order; the last one repeats."""

    def __init__(self, *batches: list[FailureCertificate], raises_at: int | None = None) -> None:
        self.batches = list(batches) or [[]]
        self.calls = 0
        self.raises_at = raises_at

    def check(self, assembly: Assembly) -> list[FailureCertificate]:
        index = min(self.calls, len(self.batches) - 1)
        self.calls += 1
        if self.raises_at is not None and self.calls - 1 == self.raises_at:
            raise MemoryError("Unable to allocate 56 MiB")
        return list(self.batches[index])


class FakeRegistry:
    def __init__(self, *, raise_on: int | None = None) -> None:
        self.applied: list[str] = []
        self.raise_on = raise_on

    def applicable(self, assembly: Assembly, certs: list[FailureCertificate]) -> list[str]:
        return ["resize_interface"]

    def apply(self, assembly: Assembly, plan_: RepairPlan) -> Assembly:
        if self.raise_on is not None and len(self.applied) == self.raise_on:
            raise RuntimeError("operator exploded")
        self.applied.append(plan_.operator)
        radius = float(plan_.params.get("r", 0.5))
        return Assembly(
            id=assembly.id,
            version=assembly.version + 1,
            parts=assembly.parts,
            joints=assembly.joints,
            connectors=({"id": "c0", "family": "pin", "params": {"r": radius}},),
            constraints=assembly.constraints,
            parent=f"{assembly.id}@v{assembly.version}",
        )


class FakePolicy:
    name = "fake"

    def __init__(self, *plans: RepairPlan | None) -> None:
        self.plans = list(plans)
        self.seen_budgets: list[Budget] = []

    def propose(self, assembly, certs, history, budget_left):
        self.seen_budgets.append(budget_left)
        if not self.plans:
            return None
        return self.plans.pop(0)


def build_loop(*, verifiers, policy, registry=None, budget=None, max_rounds=32, observer=None):
    engine = FakeEngine(budget)
    loop = RepairLoop(
        engine=engine,
        verifiers=verifiers,
        registry=registry or FakeRegistry(),
        policy=policy,
        observer=observer,
        max_rounds=max_rounds,
    )
    return loop, engine


# -- stop conditions -------------------------------------------------------


def test_a_clean_assembly_never_opens_a_round():
    loop, engine = build_loop(verifiers=FakeVerifiers([]), policy=FakePolicy())
    result = loop.run(make_ticket(), make_assembly())

    assert result.verdict == "repaired"
    assert result.rounds == 0
    assert engine.rounds == []
    assert engine.ended == [("repaired", "no physical failure")]


def test_one_failed_round_then_pass():
    verifiers = FakeVerifiers([cert()], [])
    loop, engine = build_loop(verifiers=verifiers, policy=FakePolicy(plan()))
    result = loop.run(make_ticket(), make_assembly())

    assert result.verdict == "repaired"
    assert result.rounds == 1
    assert engine.rounds[0].verdict == "pass"
    assert verifiers.calls == 2  # verified before the round and after it


def test_budget_is_a_ceiling_not_a_target():
    loop, engine = build_loop(
        verifiers=FakeVerifiers([cert()]),
        policy=FakePolicy(*[plan(radius=0.6 + i * 0.01) for i in range(10)]),
    )
    result = loop.run(make_ticket(rounds=3), make_assembly())

    assert result.verdict == "exhausted"
    assert len(engine.rounds) == 3
    assert result.rounds == 3
    assert engine.ended == [("exhausted", "budget spent")]


def test_a_geometry_budget_stops_the_loop_too():
    loop, engine = build_loop(
        verifiers=FakeVerifiers([cert()]),
        policy=FakePolicy(*[plan(radius=0.6 + i * 0.01, geom_calls=40) for i in range(5)]),
    )
    result = loop.run(make_ticket(rounds=5), make_assembly())

    # 40 + 40 + 40 = 120 > 100, so the third round never starts.
    assert result.verdict == "exhausted"
    assert len(engine.rounds) == 2


def test_no_action_is_infeasible_not_a_failure():
    loop, engine = build_loop(verifiers=FakeVerifiers([cert()]), policy=FakePolicy(None))
    result = loop.run(make_ticket(), make_assembly())

    assert result.verdict == "infeasible"
    assert result.reason == "no applicable action"
    assert engine.ended == [("infeasible", "no applicable action")]


def test_max_rounds_is_a_backstop():
    loop, engine = build_loop(
        verifiers=FakeVerifiers([cert()]),
        policy=FakePolicy(*[plan(radius=0.6 + i * 0.01) for i in range(10)]),
        max_rounds=2,
    )
    result = loop.run(make_ticket(rounds=99), make_assembly())

    assert result.verdict == "exhausted"
    assert result.reason == "max_rounds reached"
    assert len(engine.rounds) == 2


def test_a_cycle_stops_the_loop():
    """A policy that keeps proposing geometry already seen cannot make progress;
    the loop says so instead of spending the budget."""

    class SameGeometry(FakeRegistry):
        def apply(self, assembly, plan_):
            self.applied.append(plan_.operator)
            return Assembly(
                id=assembly.id,
                version=assembly.version + 1,
                parts=assembly.parts,
                joints=assembly.joints,
                connectors=assembly.connectors,  # unchanged on purpose
                constraints=assembly.constraints,
            )

    loop, engine = build_loop(
        verifiers=FakeVerifiers([cert()]),
        policy=FakePolicy(*[plan() for _ in range(5)]),
        registry=SameGeometry(),
    )
    result = loop.run(make_ticket(), make_assembly())

    assert result.verdict == "infeasible"
    assert result.reason == "geometry cycle"
    assert result.rounds == 1


# -- broken tools must not end the episode ---------------------------------


def test_an_operator_error_closes_the_round_and_the_loop_goes_on():
    registry = FakeRegistry(raise_on=0)
    loop, engine = build_loop(
        verifiers=FakeVerifiers([cert()]),
        policy=FakePolicy(plan(radius=0.6), plan(radius=0.8)),
        registry=registry,
    )
    result = loop.run(make_ticket(), make_assembly())

    assert engine.rounds[0].verdict == "error"
    assert result.rounds == 2  # the failed round still cost an attempt
    assert result.verdict in ("repaired", "exhausted", "infeasible")


def test_a_verifier_that_crashes_after_a_repair_is_not_a_pass():
    """The whole point of the environment family: a tool that cannot run leaves
    the state unknown, and unknown must never be read as success."""
    verifiers = FakeVerifiers([cert()], [], raises_at=1)
    loop, engine = build_loop(verifiers=verifiers, policy=FakePolicy(plan(radius=0.8)))

    result = loop.run(make_ticket(), make_assembly())

    # The after-repair verification raised, so that round is an error round.
    assert engine.rounds[0].verdict == "error"
    assert result.rounds == 1
    # The next verification runs cleanly and finds nothing, so the episode ends
    # repaired — but the trace still shows the crashed round.
    assert result.verdict == "repaired"


def test_a_verifier_that_crashes_before_any_round_ends_the_episode():
    verifiers = FakeVerifiers([], raises_at=0)
    loop, engine = build_loop(verifiers=verifiers, policy=FakePolicy(plan()))

    result = loop.run(make_ticket(), make_assembly())

    assert result.verdict == "failed"
    assert result.reason == "verifier could not run"
    assert engine.rounds == []
    assert result.certificates[0].family == "environment"


def test_environment_certificates_never_count_as_physical_failure():
    only_environment = [cert(code="verifier_error", family="environment")]
    assert hard_failures(only_environment) == []
    assert hard_failures([cert(), *only_environment]) == [cert()]


# -- observation -----------------------------------------------------------


def test_the_event_sequence_is_complete():
    events: list[str] = []
    loop, _engine = build_loop(
        verifiers=FakeVerifiers([cert()], []),
        policy=FakePolicy(plan()),
        observer=lambda kind, fields: events.append(kind),
    )
    loop.run(make_ticket(), make_assembly())

    assert events == [
        "episode.begin",
        "verify",
        "round.begin",
        "verify",
        "round.close",
        "ticket.end",
    ]


def test_the_policy_sees_the_remaining_budget_shrink():
    policy = FakePolicy(*[plan(radius=0.6 + i * 0.01) for i in range(4)])
    loop, _engine = build_loop(verifiers=FakeVerifiers([cert()]), policy=policy)
    loop.run(make_ticket(rounds=4), make_assembly())

    rounds_left = [b.rounds for b in policy.seen_budgets]
    assert rounds_left == [4, 3, 2, 1]


def test_stage_is_recorded_after_materialisation():
    loop, engine = build_loop(verifiers=FakeVerifiers([cert()], []), policy=FakePolicy(plan()))
    loop.run(make_ticket(), make_assembly())

    assert ("r0", "materialized") in engine.staged


# -- fingerprinting --------------------------------------------------------


def test_fingerprint_changes_when_the_connector_changes():
    assert assembly_fingerprint(make_assembly(0.5)) != assembly_fingerprint(make_assembly(0.6))


def test_fingerprint_ignores_part_order():
    forward = make_assembly()
    backward = Assembly(
        id=forward.id,
        version=forward.version,
        parts=tuple(reversed(forward.parts)),
        joints=forward.joints,
        connectors=forward.connectors,
        constraints=forward.constraints,
    )
    assert assembly_fingerprint(forward) == assembly_fingerprint(backward)


def test_fingerprint_stable_across_calls():
    assert assembly_fingerprint(make_assembly()) == assembly_fingerprint(make_assembly())


# -- ticket bookkeeping ----------------------------------------------------


def test_attention_is_not_consulted_by_the_loop():
    """The loop drives geometry; a question for a person is the engine's
    business and must not silently pause an episode."""
    ticket = make_ticket()
    ticket.attention = Attention(id="q1", question="which part may I move?")
    loop, engine = build_loop(verifiers=FakeVerifiers([cert()], []), policy=FakePolicy(plan()))
    result = loop.run(ticket, make_assembly())

    assert result.verdict == "repaired"
    assert len(engine.rounds) == 1
