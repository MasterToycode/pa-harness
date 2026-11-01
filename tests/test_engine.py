"""The engine's four invariants, its round lifecycle, and its one transaction per command.

Two properties are worth more than the rest. First, a refused plan must leave no
Round and no spent budget, because a policy is allowed to retry after a
`UserError` and would otherwise corrupt the ledger by trying again. Second, a
command that dies halfway must leave the ticket exactly as it was: the engine is
the only writer, so nothing downstream can repair a half-written aggregate.

The store is an in-memory double rather than SQLite on purpose — it can roll back
and it can be told to fail on the nth write, which is what makes the transaction
tests possible at all.
"""

from __future__ import annotations

import copy
from collections.abc import Iterator
from contextlib import contextmanager

import pytest

from pah.core import ids
from pah.core.engine import Engine
from pah.core.errors import USER_ERRORS, UserError
from pah.core.model import Budget, BudgetUsed, Cost, FailureCertificate, Mandate, RepairPlan, Round, Ticket
from pah.core.stage import ticket_stage

# --------------------------------------------------------------------------
# an in-memory store that can roll back and can be made to fail mid-write
# --------------------------------------------------------------------------


class FakeRepositories:
    """A `Repositories` holding aggregates in memory, with one rollback point per command."""

    def __init__(self) -> None:
        self._tickets: dict[str, Ticket] = {}
        self._rounds: dict[str, dict[str, Round]] = {}
        self.commits = 0
        self.rollbacks = 0
        self.writes = 0
        self.fail_on_write: int | None = None
        self._depth = 0
        self._snapshot: tuple | None = None
        self.tickets = _Tickets(self)
        self.rounds = _Rounds(self)

    @contextmanager
    def atomic(self) -> Iterator[None]:
        """The outermost block snapshots the store, so any escaping exception restores it whole."""
        if self._depth == 0:
            self._snapshot = copy.deepcopy((self._tickets, self._rounds))
            self.writes = 0
        self._depth += 1
        try:
            yield
        except BaseException:
            self._depth -= 1
            if self._depth == 0:
                self._tickets, self._rounds = self._snapshot
                self._snapshot = None
                self.rollbacks += 1
            raise
        self._depth -= 1
        if self._depth == 0:
            self._snapshot = None
            self.commits += 1

    def write(self) -> None:
        """Counts a write, raising once the armed write number is reached."""
        self.writes += 1
        if self.fail_on_write is not None and self.writes >= self.fail_on_write:
            raise RuntimeError(f"write {self.writes} failed")

    def state(self) -> tuple:
        """A comparable copy of everything stored, for assertions about rollback."""
        return copy.deepcopy((self._tickets, self._rounds))


class _Tickets:
    """The ticket repository: aggregates in and out, never a live reference."""

    def __init__(self, store: FakeRepositories) -> None:
        self._store = store

    def get(self, ticket_id: str) -> Ticket | None:
        found = self._store._tickets.get(ticket_id)
        return None if found is None else copy.deepcopy(found)

    def save(self, ticket: Ticket) -> None:
        self._store.write()
        self._store._tickets[ticket.id] = copy.deepcopy(ticket)

    def open(self) -> list[Ticket]:
        return [copy.deepcopy(t) for t in self._store._tickets.values() if t.ended is None]

    def active(self) -> list[Ticket]:
        """Open tickets that want work: queued inputs or a live assignment."""
        return [
            t
            for t in self.open()
            if t.undelivered
            or (t.assignments and t.assignments[-1].state in ("pending", "running"))
        ]


class _Rounds:
    """The round repository, keyed by the ticket that owns each round."""

    def __init__(self, store: FakeRepositories) -> None:
        self._store = store

    def save(self, ticket_id: str, rnd: Round) -> None:
        self._store.write()
        self._store._rounds.setdefault(ticket_id, {})[rnd.id] = copy.deepcopy(rnd)

    def by_ticket(self, ticket_id: str) -> list[Round]:
        rows = self._store._rounds.get(ticket_id, {})
        return [copy.deepcopy(r) for r in sorted(rows.values(), key=lambda r: r.index)]


class Counter:
    """A clock that returns a new instant per call, so the order of stamps is visible."""

    def __init__(self) -> None:
        self.ticks = 0

    def __call__(self) -> str:
        self.ticks += 1
        return f"2026-01-01T00:{self.ticks // 60:02d}:{self.ticks % 60:02d}+00:00"


# --------------------------------------------------------------------------
# fixtures and builders
# --------------------------------------------------------------------------


def make_mandate(**overrides) -> Mandate:
    fields = {
        "strength": "explicit",
        "basis": {"via": "human_request"},
        "scope_parts": ("p1", "p3"),
        "scope_operations": ("resize_interface", "add_connector"),
        "frozen_parts": ("p0",),
        "budget": Budget(rounds=3, geom_calls=20, sim_calls=4),
        "policy_version": "pv-test",
    }
    fields.update(overrides)
    return Mandate(**fields)


def make_plan(**overrides) -> RepairPlan:
    fields = {
        "id": "plan-1",
        "operator": "resize_interface",
        "targets": ("p1",),
        "params": {"dr": 0.3},
        "cost_est": Cost(geom_calls=2, sim_calls=1),
    }
    fields.update(overrides)
    return RepairPlan(**fields)


def make_ticket(**overrides) -> Ticket:
    fields = {"id": "t1", "lineage": "asm:1", "mandate": make_mandate()}
    fields.update(overrides)
    return Ticket(**fields)


def begin(engine: Engine, ticket: Ticket, **plan_overrides) -> Round:
    return engine.begin_round(ticket.id, make_plan(**plan_overrides))


def close(engine: Engine, round_id: str, verdict: str = "fail", **overrides) -> Round:
    fields = {"verdict": verdict, "evidence": [], "cost": Cost(), "artifacts": {}}
    fields.update(overrides)
    return engine.close_round(round_id, **fields)


@pytest.fixture
def store() -> FakeRepositories:
    return FakeRepositories()


@pytest.fixture
def clock() -> Counter:
    return Counter()


@pytest.fixture
def engine(store: FakeRepositories, clock: Counter) -> Engine:
    return Engine(store, clock=clock)


@pytest.fixture
def ticket(engine: Engine) -> Ticket:
    return engine.open_ticket(make_ticket())


# --------------------------------------------------------------------------
# invariant 1 — the operator is in the mandate's operations
# --------------------------------------------------------------------------


def test_operator_outside_the_mandate_is_refused(engine: Engine, ticket: Ticket):
    with pytest.raises(UserError) as err:
        begin(engine, ticket, operator="redraw_from_scratch")
    assert err.value.code == "operator_not_in_scope"
    assert err.value.values["operator"] == "redraw_from_scratch"


def test_operator_inside_the_mandate_is_accepted(engine: Engine, ticket: Ticket):
    assert begin(engine, ticket, operator="add_connector").plan.operator == "add_connector"


# --------------------------------------------------------------------------
# invariant 2 — targets are in scope and not frozen
# --------------------------------------------------------------------------


def test_target_outside_the_mandate_is_refused(engine: Engine, ticket: Ticket):
    with pytest.raises(UserError) as err:
        begin(engine, ticket, targets=("p9",))
    assert err.value.code == "target_not_in_scope"
    assert err.value.values["outside"] == ["p9"]


def test_target_inside_the_mandate_is_accepted(engine: Engine, ticket: Ticket):
    assert begin(engine, ticket, targets=("p1", "p3")).plan.targets == ("p1", "p3")


def test_frozen_target_is_refused_even_while_in_scope(engine: Engine):
    """A frozen part outranks the scope: listing it as workable does not unfreeze it."""
    ticket = engine.open_ticket(
        make_ticket(mandate=make_mandate(scope_parts=("p0", "p1"), frozen_parts=("p0",)))
    )
    with pytest.raises(UserError) as err:
        begin(engine, ticket, targets=("p0",))
    assert err.value.code == "target_frozen"
    assert err.value.values["frozen"] == ["p0"]


def test_a_frozen_part_the_plan_leaves_alone_is_no_obstacle(engine: Engine):
    ticket = engine.open_ticket(
        make_ticket(mandate=make_mandate(scope_parts=("p0", "p1"), frozen_parts=("p0",)))
    )
    assert begin(engine, ticket, targets=("p1",)).plan.targets == ("p1",)


# --------------------------------------------------------------------------
# invariant 3 — all three budget dimensions still cover the plan
# --------------------------------------------------------------------------


def test_a_plan_estimated_beyond_the_budget_is_refused_before_any_round(engine: Engine, ticket: Ticket):
    with pytest.raises(UserError) as err:
        begin(engine, ticket, cost_est=Cost(geom_calls=200))
    assert err.value.code == "budget_exhausted"
    assert err.value.values["dimension"] == "geom_calls"


def test_the_rounds_dimension_stops_exactly_at_the_ceiling(engine: Engine, store: FakeRepositories):
    """Off by one in either direction would cut a run short or overspend it."""
    ticket = engine.open_ticket(make_ticket(mandate=make_mandate(budget=Budget(3, 90, 90))))
    for _ in range(3):
        begin(engine, ticket)
    assert store.tickets.get(ticket.id).used.rounds == 3

    with pytest.raises(UserError) as err:
        begin(engine, ticket)
    assert err.value.code == "budget_exhausted"
    assert err.value.values == {"dimension": "rounds", "needed": 4, "budget": 3, "used": 3}
    assert store.tickets.get(ticket.id).used.rounds == 3


def test_the_geom_dimension_allows_an_exact_fit_and_refuses_the_next_call(engine: Engine):
    ticket = engine.open_ticket(make_ticket(mandate=make_mandate(budget=Budget(9, 4, 90))))
    close(engine, begin(engine, ticket).id, cost=Cost(geom_calls=2))
    second = begin(engine, ticket)  # 2 spent + 2 estimated == the ceiling
    close(engine, second.id, cost=Cost(geom_calls=2))

    with pytest.raises(UserError) as err:
        begin(engine, ticket)
    assert err.value.values["dimension"] == "geom_calls"
    assert err.value.values["used"] == 4


def test_the_sim_dimension_is_checked_too(engine: Engine):
    ticket = engine.open_ticket(make_ticket(mandate=make_mandate(budget=Budget(9, 90, 1))))
    close(engine, begin(engine, ticket).id, cost=Cost(sim_calls=1))

    with pytest.raises(UserError) as err:
        begin(engine, ticket)
    assert err.value.values["dimension"] == "sim_calls"


# --------------------------------------------------------------------------
# invariant 4 — the ticket has not ended
# --------------------------------------------------------------------------


def test_a_live_ticket_accepts_a_round(engine: Engine, ticket: Ticket):
    assert ticket.ended is None
    assert begin(engine, ticket).index == 0


def test_an_ended_ticket_refuses_a_round(engine: Engine, ticket: Ticket):
    engine.end_ticket(ticket.id, "infeasible", "the interface cannot be widened")

    with pytest.raises(UserError) as err:
        begin(engine, ticket)
    assert err.value.code == "ticket_ended"
    assert err.value.values["outcome"] == "infeasible"


def test_an_unknown_ticket_is_refused(engine: Engine):
    with pytest.raises(UserError) as err:
        engine.begin_round("nope", make_plan())
    assert err.value.code == "ticket_missing"


# --------------------------------------------------------------------------
# a refusal records nothing and spends nothing
# --------------------------------------------------------------------------


def test_a_refused_plan_adds_no_round_and_spends_no_budget(
    engine: Engine, store: FakeRepositories, ticket: Ticket
):
    commits = store.commits
    with pytest.raises(UserError):
        begin(engine, ticket, targets=("p9",))

    stored = store.tickets.get(ticket.id)
    assert stored.rounds == []
    assert stored.used == BudgetUsed()
    assert store.rounds.by_ticket(ticket.id) == []
    assert store.commits == commits
    assert store.rollbacks == 1


def test_a_refused_over_budget_plan_keeps_the_round_it_could_not_afford(engine: Engine, store: FakeRepositories):
    ticket = engine.open_ticket(make_ticket(mandate=make_mandate(budget=Budget(1, 90, 90))))
    begin(engine, ticket)

    with pytest.raises(UserError):
        begin(engine, ticket)
    assert store.tickets.get(ticket.id).used.rounds == 1


# --------------------------------------------------------------------------
# begin_round writes both repositories as one thing
# --------------------------------------------------------------------------


def test_begin_round_opens_an_indexed_round_in_both_repositories(engine: Engine, store: FakeRepositories, ticket: Ticket):
    commits = store.commits
    rnd = begin(engine, ticket)

    assert (rnd.index, rnd.stage) == (0, "proposed")
    assert rnd.id == ids.key("round", ticket.id, 0)
    assert store.rounds.by_ticket(ticket.id) == [rnd]
    assert store.tickets.get(ticket.id).rounds == [rnd]
    # the round's own save nests inside the command's transaction: one commit, not two
    assert store.commits == commits + 1


def test_the_index_counts_the_rounds_already_there(engine: Engine, ticket: Ticket):
    begin(engine, ticket)
    second = begin(engine, ticket)
    assert (second.index, second.id) == (1, ids.key("round", ticket.id, 1))


def test_begin_round_restamps_updated_at(engine: Engine, store: FakeRepositories, ticket: Ticket):
    begin(engine, ticket)
    stored = store.tickets.get(ticket.id)
    assert stored.created_at < stored.updated_at


# --------------------------------------------------------------------------
# stage_round moves forward only
# --------------------------------------------------------------------------


def test_a_round_stage_advances_forward(engine: Engine, store: FakeRepositories, ticket: Ticket):
    rnd = begin(engine, ticket)
    for stage in ("materialized", "checked", "simulated"):
        engine.stage_round(rnd.id, stage)
        assert store.rounds.by_ticket(ticket.id)[0].stage == stage


def test_going_back_to_an_earlier_stage_is_refused(engine: Engine, store: FakeRepositories, ticket: Ticket):
    rnd = begin(engine, ticket)
    engine.stage_round(rnd.id, "checked")

    with pytest.raises(UserError) as err:
        engine.stage_round(rnd.id, "materialized")
    assert err.value.code == "stage_regression"
    assert store.rounds.by_ticket(ticket.id)[0].stage == "checked"


def test_restaging_the_stage_it_already_holds_writes_nothing(engine: Engine, store: FakeRepositories, ticket: Ticket):
    """A resumed executor may re-issue a stage; that must not look like progress or like an error."""
    rnd = begin(engine, ticket)
    engine.stage_round(rnd.id, "materialized")
    before = store.state()

    engine.stage_round(rnd.id, "materialized")
    assert store.writes == 0  # the no-op opens a transaction and writes nothing inside it
    assert store.state() == before


def test_the_judged_stage_is_not_a_caller_stage(engine: Engine, ticket: Ticket):
    """`judged` means a verdict exists; reaching it without one would make the stage lie."""
    rnd = begin(engine, ticket)
    for stage in ("judged", "proposed", "reviewed"):
        with pytest.raises(UserError) as err:
            engine.stage_round(rnd.id, stage)
        assert err.value.code == "unknown_stage"


# --------------------------------------------------------------------------
# close_round writes the one verdict
# --------------------------------------------------------------------------


def test_closing_records_the_verdict_evidence_cost_and_artifacts(engine: Engine, store: FakeRepositories, ticket: Ticket):
    rnd = begin(engine, ticket)
    closed = close(
        engine,
        rnd.id,
        verdict="fail",
        evidence=[
            FailureCertificate(
                code="gap_too_wide",
                family="assembly",
                locus={"edge": "e1"},
                magnitude={"gap_mm": 2.4},
                observed_by="connector_feasible@v1",
            )
        ],
        cost=Cost(geom_calls=2, sim_calls=1, wall_s=0.4),
        artifacts={"assembly": "sha256:abc"},
        edit_cost=0.25,
        preserve_ratio=0.75,
    )

    assert (closed.verdict, closed.stage) == ("fail", "judged")
    assert closed.artifacts == {"assembly": "sha256:abc"}
    assert (closed.edit_cost, closed.preserve_ratio) == (0.25, 0.75)
    assert closed.cost.geom_calls == 2
    assert closed.evidence[0].code == "gap_too_wide"
    assert store.rounds.by_ticket(ticket.id)[0] == closed
    assert store.tickets.get(ticket.id).rounds[0] == closed


def test_a_closed_round_cannot_be_closed_again(engine: Engine, store: FakeRepositories, ticket: Ticket):
    rnd = begin(engine, ticket)
    close(engine, rnd.id, verdict="fail")

    with pytest.raises(UserError) as err:
        close(engine, rnd.id, verdict="pass")
    assert err.value.code == "round_already_closed"
    assert store.rounds.by_ticket(ticket.id)[0].verdict == "fail"


def test_a_closed_round_cannot_be_restaged(engine: Engine, ticket: Ticket):
    rnd = begin(engine, ticket)
    close(engine, rnd.id, verdict="pass")

    with pytest.raises(UserError) as err:
        engine.stage_round(rnd.id, "checked")
    assert err.value.code == "round_already_closed"


def test_an_unknown_verdict_is_refused(engine: Engine, store: FakeRepositories, ticket: Ticket):
    rnd = begin(engine, ticket)

    with pytest.raises(UserError) as err:
        close(engine, rnd.id, verdict="probably_fine")
    assert err.value.code == "unknown_verdict"
    assert store.rounds.by_ticket(ticket.id)[0].verdict is None


def test_closing_an_unknown_round_is_refused(engine: Engine):
    with pytest.raises(UserError) as err:
        close(engine, "nope", verdict="fail")
    assert err.value.code == "round_missing"


# --------------------------------------------------------------------------
# close_round is what the derived stage reads
# --------------------------------------------------------------------------


def test_closing_a_pass_at_the_rounds_ceiling_clears_person_budget(engine: Engine, store: FakeRepositories):
    ticket = engine.open_ticket(make_ticket(mandate=make_mandate(budget=Budget(1, 90, 90))))
    rnd = begin(engine, ticket)
    assert ticket_stage(store.tickets.get(ticket.id)) == "person/budget"

    close(engine, rnd.id, verdict="pass")
    assert ticket_stage(store.tickets.get(ticket.id)) == "working/queued"


def test_closing_a_fail_at_the_rounds_ceiling_leaves_person_budget(engine: Engine, store: FakeRepositories):
    ticket = engine.open_ticket(make_ticket(mandate=make_mandate(budget=Budget(1, 90, 90))))
    close(engine, begin(engine, ticket).id, verdict="fail")
    assert ticket_stage(store.tickets.get(ticket.id)) == "person/budget"


def test_closing_a_pass_puts_a_signoff_ticket_in_review(engine: Engine, store: FakeRepositories):
    ticket = engine.open_ticket(make_ticket(signoff_required=True))
    rnd = begin(engine, ticket)
    assert ticket_stage(store.tickets.get(ticket.id)) == "working/queued"

    close(engine, rnd.id, verdict="pass")
    assert ticket_stage(store.tickets.get(ticket.id)) == "person/review"


# --------------------------------------------------------------------------
# end_ticket happens once
# --------------------------------------------------------------------------


def test_ending_records_the_outcome_and_the_reason_once(engine: Engine, store: FakeRepositories, ticket: Ticket):
    ended = engine.end_ticket(ticket.id, "exhausted", "budget.rounds")

    assert ended.ended.outcome == "exhausted"
    assert ended.ended.reason == "budget.rounds"
    assert ended.ended.by == "executor"
    assert store.tickets.get(ticket.id).ended == ended.ended
    assert ticket_stage(store.tickets.get(ticket.id)) == "ended"


def test_ending_twice_returns_the_first_record(engine: Engine, store: FakeRepositories, ticket: Ticket):
    first = engine.end_ticket(ticket.id, "exhausted", "budget.rounds")
    before = store.state()

    again = engine.end_ticket(ticket.id, "repaired", "a second opinion")

    assert again.ended == first.ended
    assert store.writes == 0
    assert store.state() == before
    assert ticket_stage(store.tickets.get(ticket.id)) == "ended"


def test_an_unknown_outcome_is_refused(engine: Engine, store: FakeRepositories, ticket: Ticket):
    with pytest.raises(UserError) as err:
        engine.end_ticket(ticket.id, "victory")
    assert err.value.code == "unknown_outcome"
    assert store.tickets.get(ticket.id).ended is None


def test_ending_an_unknown_ticket_is_refused(engine: Engine):
    with pytest.raises(UserError) as err:
        engine.end_ticket("nope", "dropped")
    assert err.value.code == "ticket_missing"


def test_a_round_of_an_ended_ticket_cannot_be_closed(engine: Engine, store: FakeRepositories, ticket: Ticket):
    """Ending freezes the round history; a late verdict would rewrite a closed record."""
    rnd = begin(engine, ticket)
    engine.end_ticket(ticket.id, "stopped", "a person stopped it")

    with pytest.raises(UserError) as err:
        close(engine, rnd.id, verdict="fail")
    assert err.value.code == "round_missing"
    assert store.rounds.by_ticket(ticket.id)[0].verdict is None


# --------------------------------------------------------------------------
# note_ticket records without granting or queueing
# --------------------------------------------------------------------------


def test_a_note_is_recorded_without_queueing_a_delivery(engine: Engine, store: FakeRepositories, ticket: Ticket):
    before = store.tickets.get(ticket.id).used
    engine.note_ticket(ticket.id, "the p1 rib looks thin in the render")

    stored = store.tickets.get(ticket.id)
    note = stored.inputs[-1]
    assert (note["kind"], note["by"], note["text"]) == ("note", "human", "the p1 rib looks thin in the render")
    assert note["at"] == stored.updated_at
    assert stored.undelivered == []
    assert store.tickets.active() == []
    assert stored.used == before


def test_a_note_on_an_ended_ticket_is_refused(engine: Engine, ticket: Ticket):
    engine.end_ticket(ticket.id, "dropped")

    with pytest.raises(UserError) as err:
        engine.note_ticket(ticket.id, "for the record")
    assert err.value.code == "ticket_ended"


def test_a_note_on_an_unknown_ticket_is_refused(engine: Engine):
    with pytest.raises(UserError) as err:
        engine.note_ticket("nope", "text")
    assert err.value.code == "ticket_missing"


# --------------------------------------------------------------------------
# open_ticket is idempotent and stamps its own clock
# --------------------------------------------------------------------------


def test_open_ticket_stamps_both_times_from_the_clock(engine: Engine):
    ticket = engine.open_ticket(make_ticket())
    assert ticket.created_at == "2026-01-01T00:00:01+00:00"
    assert ticket.updated_at == ticket.created_at


def test_opening_the_same_id_twice_keeps_the_history(engine: Engine, store: FakeRepositories, ticket: Ticket):
    begin(engine, ticket)
    before = store.state()

    again = engine.open_ticket(make_ticket())

    assert again.id == ticket.id
    assert len(again.rounds) == 1
    assert again.created_at == ticket.created_at
    assert store.writes == 0
    assert store.state() == before


# --------------------------------------------------------------------------
# one transaction per command, whatever happens in the middle
# --------------------------------------------------------------------------


def _nothing(engine: Engine, ticket: Ticket) -> dict:
    return {}


def _one_round(engine: Engine, ticket: Ticket) -> dict:
    return {"round_id": begin(engine, ticket).id}


_COMMANDS = [
    ("open_ticket", _nothing, lambda e, t, ctx: e.open_ticket(make_ticket(id="t2")), 1),
    ("begin_round", _nothing, lambda e, t, ctx: begin(e, t), 2),
    ("stage_round", _one_round, lambda e, t, ctx: e.stage_round(ctx["round_id"], "materialized"), 2),
    ("close_round", _one_round, lambda e, t, ctx: close(e, ctx["round_id"]), 2),
    ("end_ticket", _nothing, lambda e, t, ctx: e.end_ticket(t.id, "stopped"), 1),
    ("note_ticket", _nothing, lambda e, t, ctx: e.note_ticket(t.id, "note"), 1),
]


@pytest.mark.parametrize(
    "name,prepare,command,writes", _COMMANDS, ids=[case[0] for case in _COMMANDS]
)
def test_a_command_that_dies_mid_write_leaves_the_store_untouched(
    engine: Engine,
    store: FakeRepositories,
    name: str,
    prepare,
    command,
    writes: int,
):
    """The failure lands on the command's last write, so an earlier write of the same command must be undone."""
    ticket = engine.open_ticket(make_ticket())
    context = prepare(engine, ticket)
    before = store.state()
    commits = store.commits

    store.fail_on_write = writes
    try:
        with pytest.raises(RuntimeError):
            command(engine, ticket, context)
    finally:
        store.fail_on_write = None

    assert store.state() == before
    assert store.commits == commits
    assert store.rollbacks == 1
    # and the store is usable again: the same command now succeeds
    command(engine, ticket, context)
    assert store.commits == commits + 1


# --------------------------------------------------------------------------
# the refusal table itself
# --------------------------------------------------------------------------


REQUIRED_CODES = {
    "ticket_ended",
    "ticket_missing",
    "round_missing",
    "operator_not_in_scope",
    "target_not_in_scope",
    "target_frozen",
    "budget_exhausted",
    "unknown_operator",
    "round_already_closed",
    "stage_regression",
}


def test_the_refusal_table_documents_every_required_code():
    assert REQUIRED_CODES <= set(USER_ERRORS)
    assert all(isinstance(text, str) and text.strip() for text in USER_ERRORS.values())


def test_a_refusal_carries_its_code_message_and_values():
    err = UserError("budget_exhausted", "no room for one more round", {"dimension": "rounds"})
    assert err.code == "budget_exhausted"
    assert err.message == "no room for one more round"
    assert err.values == {"dimension": "rounds"}
    assert "budget_exhausted" in str(err) and "no room for one more round" in str(err)
    assert UserError("ticket_missing", "gone").values == {}
