"""Derive the one stage a Ticket is in.

The table is `docs/pah-dev-doc-v1.md` §3.6, read top to bottom; the first
matching row wins. Nothing here writes state — that is the engine's job — so a
stage can never disagree with the facts it is computed from, including after a
crash and a restart.
"""

from __future__ import annotations

from pah.core.model import (
    Ticket,
    latest_assignment,
    latest_report,
    latest_round,
    standing_report,
)

__all__ = ["STAGES", "ticket_stage", "can_retry", "waits_for_person", "is_person_stage"]

#: `person/*` means the ticket cannot continue without a person.
STAGES = (
    "person/delivery",
    "person/answer",
    "person/review",
    "person/budget",
    "person/blocked",
    "person/failed",
    "working/queued",
    "idle",
    "working",
    "ended",
)

_PERSON_PREFIX = "person/"


def waits_for_person(ticket: Ticket) -> bool:
    """The latest standing report is `blocked` or `failed`, and nothing a person
    said has reached the ticket since. Automatic inputs stay held while true —
    otherwise a blocked ticket would keep waking itself."""
    report = standing_report(ticket)
    return report is not None and report.state in ("blocked", "failed")


def can_retry(ticket: Ticket) -> bool:
    """A new assignment is the remedy for a stalled one, but only while the
    ticket still waits on a person or its last delivery failed."""
    if ticket.ended is not None:
        return False
    if ticket.delivery_failed:
        return True
    stage = ticket_stage(ticket)
    return stage in ("person/delivery", "person/blocked", "person/failed")


def is_person_stage(stage: str) -> bool:
    return stage.startswith(_PERSON_PREFIX)


def _awaiting_signoff(ticket: Ticket) -> bool:
    """A round passed, but the mandate says a person must confirm before the
    ticket counts as done (physical assembly, a printed part)."""
    if not ticket.signoff_required or ticket.signoff_at is not None:
        return False
    last = latest_round(ticket)
    return last is not None and last.verdict == "pass"


def ticket_stage(ticket: Ticket) -> str:
    if ticket.ended is not None:
        return "ended"

    if ticket.delivery_failed:
        return "person/delivery"

    if ticket.attention is not None:
        return "person/answer"

    if _awaiting_signoff(ticket):
        return "person/review"

    if ticket.used.exhausted(ticket.mandate.budget) and not ticket.passed:
        return "person/budget"

    if waits_for_person(ticket):
        report = standing_report(ticket)
        # `waits_for_person` guarantees a report, and only these two states match.
        return f"person/{report.state}"  # type: ignore[union-attr]

    assignment = latest_assignment(ticket)
    if assignment is None or assignment.state == "pending":
        return "working/queued"

    report = latest_report(ticket)
    if (report is not None and report.state == "waiting") or assignment.state == "ended":
        return "idle"

    return "working"
