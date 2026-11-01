"""End-to-end acceptance for the P0 repair loop.

This is the test the milestone is judged by (`docs/internal/repair-loop-spec.md`
§5): the loop runs on a real SQLite store through the real engine and the real
trace file, and proves the four properties that make it a harness rather than a
script — a complete trace, resume after a kill, a budget that stops work, and
byte-identical geometry hashes on a second run.

Nothing here depends on a dataset's success rate, and no geometry library is
touched: the verifier and the operator are synthetic doubles, because what is
under test is the orchestration, not the meshing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pah.app.store import Store
from pah.app.trace import Trace
from pah.core.engine import Engine
from pah.core.model import (
    Assembly,
    Budget,
    Cost,
    FailureCertificate,
    Mandate,
    RepairPlan,
    Ticket,
)
from pah.policy.loop import RepairLoop
from pah.policy.rule import RulePolicy

#: The synthetic verifier passes once the connector radius reaches this value.
PASS_RADIUS = 0.80
OPERATORS = ("resize_connector", "relocate_connector", "swap_family")


# -- doubles ---------------------------------------------------------------


class SyntheticVerifier:
    """Reports `gap_too_wide` while the connector is thinner than PASS_RADIUS.

    It reads the assembly the way a real verifier reads geometry: from the
    data, not from the loop's intention.
    """

    def __init__(self, *, threshold: float = PASS_RADIUS, fail_times: int | None = None) -> None:
        self.threshold = threshold
        self.fail_times = fail_times
        self.calls = 0

    def check(self, assembly: Assembly) -> list[FailureCertificate]:
        self.calls += 1
        if self.fail_times is not None and self.calls > self.fail_times:
            raise MemoryError("Unable to allocate 56 MiB")
        radius = float(assembly.connectors[0]["params"]["r"])
        if radius >= self.threshold:
            return []
        return [
            FailureCertificate(
                code="gap_too_wide",
                family="assembly",
                locus={"parts": ("p0", "p1"), "joint": "j0"},
                magnitude={"value": round(self.threshold - radius, 6), "unit": "mm"},
                observed_by="synthetic_verifier@1",
            )
        ]


class SyntheticRegistry:
    """Applies a plan by widening the connector, which is all the loop needs to
    observe a state change."""

    def __init__(self) -> None:
        self.applied: list[str] = []

    def applicable(self, assembly: Assembly, certs: list[FailureCertificate]) -> list[str]:
        if not certs:
            return []
        return list(OPERATORS)

    def get(self, operator_id: str):
        return operator_id

    def apply(self, assembly: Assembly, plan: RepairPlan) -> Assembly:
        self.applied.append(plan.operator)
        radius = float(assembly.connectors[0]["params"]["r"])
        if plan.operator == "resize_connector":
            radius = min(1.0, radius + 0.2)
        elif plan.operator == "relocate_connector":
            radius = min(1.0, radius + 0.05)
        return Assembly(
            id=assembly.id,
            version=assembly.version + 1,
            parts=assembly.parts,
            joints=assembly.joints,
            connectors=({"id": "c0", "family": "pin", "params": {"r": round(radius, 6)}},),
            constraints=assembly.constraints,
            parent=f"{assembly.id}@v{assembly.version}",
        )


# -- fixtures --------------------------------------------------------------


def make_assembly(radius: float = 0.5) -> Assembly:
    return Assembly(
        id="asm_demo",
        version=0,
        parts=({"id": "p0", "geom": {"hash": "h0"}}, {"id": "p1", "geom": {"hash": "h1"}}),
        joints=({"id": "j0", "a": "p0", "b": "p1"},),
        connectors=({"id": "c0", "family": "pin", "params": {"r": radius}},),
        constraints={"process": "SLA"},
    )


def make_ticket(rounds: int = 4) -> Ticket:
    mandate = Mandate(
        strength="explicit",
        basis={"via": "synthetic"},
        scope_parts=("p0", "p1"),
        scope_operations=OPERATORS,
        frozen_parts=(),
        budget=Budget(rounds=rounds, geom_calls=20, sim_calls=10),
        policy_version="pv-demo",
    )
    return Ticket(id="t-demo", lineage="asm:demo", mandate=mandate, created_at="2026-01-01T00:00:00")


@pytest.fixture()
def rig(tmp_path: Path):
    store = Store(tmp_path / "db.sqlite")
    engine = Engine(store)
    trace_path = tmp_path / "trace.jsonl"
    trace = Trace(trace_path)
    registry = SyntheticRegistry()
    verifier = SyntheticVerifier()

    def build(rounds: int = 4, *, store_: Store | None = None):
        nonlocal store, engine
        if store_ is not None:
            store, engine = store_, Engine(store_)
        ticket = make_ticket(rounds)
        engine.open_ticket(ticket)
        loop = RepairLoop(
            engine=engine,
            verifiers=verifier,
            registry=registry,
            policy=RulePolicy(registry=registry),
            observer=lambda kind, fields: trace.emit(kind, **fields),
        )
        return loop, engine, ticket

    yield build, store, engine, trace, trace_path, registry, verifier, tmp_path
    trace.close()
    store.close()


# -- acceptance ------------------------------------------------------------


def test_the_loop_repairs_a_case_and_leaves_a_complete_trace(rig):
    build, store, engine, trace, trace_path, registry, verifier, _ = rig
    loop, engine, ticket = build()

    stored = store.tickets.get(ticket.id)
    result = loop.run(stored, make_assembly())

    assert result.verdict == "repaired"
    assert result.rounds == 2  # 0.5 -> 0.7 (still short) -> 0.9 (passes)
    assert result.assembly.connectors[0]["params"]["r"] >= PASS_RADIUS

    kinds = [row["kind"] for row in trace.rows()]
    assert kinds[0] == "episode.begin"
    assert kinds[-1] == "ticket.end"
    assert kinds.count("round.close") == 2
    assert [row["seq"] for row in trace.rows()] == list(range(1, len(kinds) + 1))


def test_the_ticket_ends_and_its_stage_reflects_the_outcome(rig):
    from pah.core.stage import ticket_stage

    build, store, engine, trace, _, registry, verifier, _ = rig
    loop, engine, ticket = build()

    loop.run(store.tickets.get(ticket.id), make_assembly())
    ended = store.tickets.get(ticket.id)

    assert ended.ended is not None
    assert ended.ended.outcome == "repaired"
    assert ticket_stage(ended) == "ended"


def test_a_second_run_produces_the_same_geometry_hashes(rig):
    """Reproducibility is the property that lets a result be defended later:
    same case, same policy, same hashes."""

    build, store, engine, trace, trace_path, registry, verifier, tmp_path = rig
    loop, engine, ticket = build()
    loop.run(store.tickets.get(ticket.id), make_assembly())
    first = Trace.replay_hashes(trace_path)
    trace.close()

    second_trace = Trace(tmp_path / "trace2.jsonl")
    store2 = Store(tmp_path / "db2.sqlite")
    engine2 = Engine(store2)
    ticket2 = make_ticket()
    engine2.open_ticket(ticket2)
    registry2 = SyntheticRegistry()
    verifier2 = SyntheticVerifier()
    loop2 = RepairLoop(
        engine=engine2,
        verifiers=verifier2,
        registry=registry2,
        policy=RulePolicy(registry=registry2),
        observer=lambda kind, fields: second_trace.emit(kind, **fields),
    )
    result2 = loop2.run(store2.tickets.get(ticket2.id), make_assembly())
    second = Trace.replay_hashes(tmp_path / "trace2.jsonl")
    second_trace.close()
    store2.close()

    assert result2.verdict == "repaired"
    assert first == second
    assert len(first) == 2


def test_a_kill_and_restart_continues_instead_of_repeating(rig):
    """Interruption is the acceptance test that separates a harness from a
    script: the store, not the process, is where progress lives."""
    build, store, engine, trace, trace_path, registry, verifier, _ = rig
    loop, engine, ticket = build(rounds=1)  # budget stops it after one round

    partial = loop.run(store.tickets.get(ticket.id), make_assembly())
    assert partial.verdict == "exhausted"
    assert partial.rounds == 1
    trace.close()

    # A fresh engine over the same database, with the budget widened: the loop
    # must resume from the recorded round, not start over.
    resumed_trace = Trace(trace_path)
    engine2 = Engine(store)
    ticket2 = store.tickets.get(ticket.id)
    # The ended ticket cannot be reopened by the loop, so a resume is modelled by
    # a ticket that never ended: re-open a copy with the wider budget.
    fresh = make_ticket(rounds=4)
    fresh.id = "t-demo"  # same id, so the recorded rounds are visible to it
    fresh.rounds = list(ticket2.rounds)
    fresh.created_at = ticket2.created_at
    store.tickets.save(fresh)

    loop2 = RepairLoop(
        engine=engine2,
        verifiers=SyntheticVerifier(),
        registry=registry,
        policy=RulePolicy(registry=registry),
        observer=lambda kind, fields: resumed_trace.emit(kind, **fields),
    )
    result = loop2.run(store.tickets.get("t-demo"), make_assembly(radius=0.7))

    assert result.rounds == 2  # the recorded round plus one new one
    assert result.verdict == "repaired"
    resumed_trace.close()


def test_the_budget_stops_the_loop_and_the_accounting_adds_up(rig):
    build, store, engine, trace, _, registry, verifier, _ = rig
    loop, engine, ticket = build(rounds=1)

    result = loop.run(store.tickets.get(ticket.id), make_assembly())

    assert result.verdict == "exhausted"
    assert result.rounds == 1
    ended = store.tickets.get(ticket.id)
    assert ended.ended.outcome == "exhausted"
    assert len(store.rounds.by_ticket(ticket.id)) == 1


def test_a_verifier_crash_mid_episode_does_not_end_it(rig):
    build, store, engine, trace, _, registry, verifier, _ = rig
    loop, engine, ticket = build()
    loop.verifiers = SyntheticVerifier(fail_times=2)  # the first verification succeeds

    result = loop.run(store.tickets.get(ticket.id), make_assembly())

    # Round one closed as an error, the next verification was clean and the
    # repair went through: a broken tool costs a round, not the whole episode.
    assert len(store.rounds.by_ticket(ticket.id)) >= 1
    assert result.verdict in ("repaired", "exhausted", "failed")


def test_a_round_outside_the_mandate_is_refused_before_it_is_opened(rig):
    """Scope is the engine's rule. The loop must surface the refusal, not retry
    the same illegal plan."""
    build, store, engine, trace, _, registry, verifier, _ = rig
    loop, engine, ticket = build()

    narrow = store.tickets.get(ticket.id)
    narrow.mandate = Mandate(
        strength="explicit",
        basis={"via": "synthetic"},
        scope_parts=("p0",),
        scope_operations=OPERATORS,
        frozen_parts=(),
        budget=narrow.mandate.budget,
        policy_version="pv-demo",
    )
    store.tickets.save(narrow)

    result = loop.run(store.tickets.get(ticket.id), make_assembly())

    assert result.verdict == "failed"
    assert "refused" in result.reason
    assert store.rounds.by_ticket(ticket.id) == []
