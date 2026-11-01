"""One test per row of the stage table, plus the ordering rules between rows.

The point of these tests is the property `tests/stage.spec.ts` protects in the
prototype: the table is read top to bottom and the first matching row wins, so
an earlier row must beat a later one even when both would match.
"""

from __future__ import annotations

from pah.core.model import (
    Assignment,
    Attention,
    Budget,
    Cost,
    Mandate,
    RepairPlan,
    Report,
    Round,
    Ticket,
    TicketEnd,
)
from pah.core.stage import can_retry, ticket_stage, waits_for_person

MANDATE = Mandate(
    strength="explicit",
    basis={"via": "human_request"},
    scope_parts=("p1",),
    scope_operations=("resize_interface",),
    frozen_parts=("p0",),
    budget=Budget(rounds=5, geom_calls=200, sim_calls=40),
    policy_version="pv-test",
)


def make_ticket(**overrides) -> Ticket:
    ticket = Ticket(id="t1", lineage="asm:1", mandate=MANDATE, created_at="2026-01-01T00:00:00")
    for name, value in overrides.items():
        setattr(ticket, name, value)
    return ticket


def round_(index: int, verdict=None) -> Round:
    plan = RepairPlan(id=f"plan{index}", operator="resize_interface", targets=("p1",), params={})
    return Round(id=f"r{index}", index=index, plan=plan, verdict=verdict, cost=Cost())


def assignment(state="running", report=None, delivered_at=None) -> Assignment:
    return Assignment(id="a1", state=state, report=report, delivered_at=delivered_at)


# -- rows ------------------------------------------------------------------


def test_no_assignment_is_queued():
    assert ticket_stage(make_ticket()) == "working/queued"


def test_pending_assignment_is_queued():
    assert ticket_stage(make_ticket(assignments=[assignment("pending")])) == "working/queued"


def test_running_assignment_is_working():
    assert ticket_stage(make_ticket(assignments=[assignment("running")])) == "working"


def test_waiting_report_is_idle():
    ticket = make_ticket(assignments=[assignment("running", Report("waiting", "ok", "2026-01-01T01:00:00"))])
    assert ticket_stage(ticket) == "idle"


def test_ended_assignment_without_report_is_idle():
    assert ticket_stage(make_ticket(assignments=[assignment("ended")])) == "idle"


def test_blocked_report_is_person_blocked():
    ticket = make_ticket(assignments=[assignment("running", Report("blocked", "stuck", "2026-01-01T01:00:00"))])
    assert ticket_stage(ticket) == "person/blocked"
    assert waits_for_person(ticket)


def test_failed_report_is_person_failed():
    ticket = make_ticket(assignments=[assignment("running", Report("failed", "boom", "2026-01-01T01:00:00"))])
    assert ticket_stage(ticket) == "person/failed"


def test_input_after_report_releases_the_hold():
    """A person's message that reached the worker ends the hold, so the ticket
    returns to work instead of staying blocked."""
    ticket = make_ticket(
        assignments=[
            assignment(
                "running",
                Report("blocked", "stuck", "2026-01-01T01:00:00"),
                delivered_at="2026-01-01T02:00:00",
            )
        ]
    )
    assert ticket_stage(ticket) == "working"
    assert not waits_for_person(ticket)


def test_attention_is_person_answer():
    ticket = make_ticket(
        assignments=[assignment("running", Report("blocked", "s", "2026-01-01T01:00:00"))],
        attention=Attention(id="q1", question="which part may I move?"),
    )
    # person/answer outranks person/blocked: the ticket is asking a question.
    assert ticket_stage(ticket) == "person/answer"


def test_failed_delivery_is_person_delivery():
    ticket = make_ticket(assignments=[assignment("running")], delivery_failed=True)
    assert ticket_stage(ticket) == "person/delivery"


def test_budget_exhausted_is_person_budget_when_not_passed():
    ticket = make_ticket(rounds=[round_(0), round_(1), round_(2), round_(3), round_(4)])
    # rounds 5/5 used, no pass
    assert ticket_stage(ticket) == "person/budget"


def test_budget_exhausted_but_passed_is_not_person_budget():
    ticket = make_ticket(rounds=[round_(0, "pass")])
    assert ticket_stage(ticket) == "working/queued"


def test_signoff_required_is_person_review():
    ticket = make_ticket(rounds=[round_(0, "pass")], signoff_required=True)
    assert ticket_stage(ticket) == "person/review"


def test_signoff_done_leaves_review():
    ticket = make_ticket(
        rounds=[round_(0, "pass")], signoff_required=True, signoff_at="2026-01-01T03:00:00"
    )
    assert ticket_stage(ticket) != "person/review"


def test_ended_wins_over_everything():
    ticket = make_ticket(ended=TicketEnd(at="x", by="human", outcome="stopped"))
    assert ticket_stage(ticket) == "ended"


# -- ordering --------------------------------------------------------------


def test_delivery_failure_outranks_attention_and_budget():
    ticket = make_ticket(
        delivery_failed=True,
        attention=Attention(id="q", question="q"),
        rounds=[round_(i) for i in range(5)],
    )
    assert ticket_stage(ticket) == "person/delivery"


def test_attention_outranks_budget():
    ticket = make_ticket(attention=Attention(id="q", question="q"), rounds=[round_(i) for i in range(5)])
    assert ticket_stage(ticket) == "person/answer"


# -- retry -----------------------------------------------------------------


def test_can_retry_on_a_stalled_ticket():
    ticket = make_ticket(assignments=[assignment("running", Report("failed", "x", "2026-01-01T01:00:00"))])
    assert can_retry(ticket)


def test_cannot_retry_while_working():
    assert not can_retry(make_ticket(assignments=[assignment("running")]))


def test_cannot_retry_a_stopped_ticket():
    ticket = make_ticket(
        ended=TicketEnd(at="x", by="human", outcome="stopped"),
        assignments=[assignment("running", Report("failed", "x", "2026-01-01T01:00:00"))],
    )
    assert not can_retry(ticket)
