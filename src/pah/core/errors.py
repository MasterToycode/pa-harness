"""Refusals a caller can act on.

A `UserError` is the engine saying no to a request that arrived through the
command face: a plan outside the mandate, a frozen part, a spent budget, a
ticket that already ended. It is not a fault report. The request was understood
and refused, nothing was written, and no budget moved — which is why a policy or
an LLM may catch it and propose something else, while everything else that
escapes a command is a real bug and must not be swallowed.

`USER_ERRORS` is the one table the CLI, the panel and the trace read, so the code
a person is shown and the code written to a log cannot drift apart.
"""

from __future__ import annotations

__all__ = ["USER_ERRORS", "UserError"]


#: code -> one sentence of English. Every code the engine can raise is here.
USER_ERRORS: dict[str, str] = {
    "ticket_ended": "The ticket has ended; only end_ticket may be called on it.",
    "ticket_missing": "No ticket carries that id.",
    "round_missing": "No open ticket holds a round with that id; a round id cannot be reversed.",
    "operator_not_in_scope": "The mandate does not list this operator among the operations it allows.",
    "target_not_in_scope": "The plan would change a part the mandate does not cover.",
    "target_frozen": "The plan would change a part the mandate froze.",
    "budget_exhausted": "The ticket has no budget left for this plan in at least one dimension.",
    "unknown_operator": "No such operator is registered; a plan may not invent an action.",
    "round_already_closed": "The round already carries a verdict; a closed round is history.",
    "stage_regression": "A round's stage may only move forward.",
    "unknown_stage": "Only materialized, checked and simulated advance a round.",
    "unknown_verdict": "The verdict is not one the domain model allows.",
    "unknown_outcome": "The outcome is not one a ticket may end with.",
}


class UserError(Exception):
    """A refusal: `code` names the rule that stopped the command, `values` the specifics."""

    def __init__(self, code: str, message: str, values: dict | None = None) -> None:
        self.code = code
        self.message = message
        self.values = dict(values) if values else {}
        super().__init__(f"{code}: {message}")
