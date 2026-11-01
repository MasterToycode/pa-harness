"""The engine: the only writer of domain state.

Every change to a Ticket or a Round goes through one command here, one
transaction per command (`docs/internal/repair-loop-spec.md` §2.1). Nothing in
this module judges geometry, calls a verifier or decides what to repair: it
decides only whether a request is allowed, and records it if it is. A command
that refuses leaves no Round and spends no budget, so a policy may retry freely
after a `UserError` instead of unwinding a half-written aggregate.

Four invariants are checked on every `begin_round` (§2.1): the operator is in
the mandate's operations, the targets are in scope and not frozen, the budget
still covers the plan in all three dimensions, and the ticket has not ended.
Checks run existence, ended, scope, frozen, budget — the cheapest refusal that
is true, and `ended` outranks scope because a ticket that cannot work at all
should not send its caller to fix a scope that no longer matters.

Three rules the spec leaves open, fixed here because two implementations must
not differ on them:

* A Round costs one `rounds` unit the moment it exists, while `geom_calls` and
  `sim_calls` enter the ledger when the Round is closed with its measured cost.
  `plan.cost_est` is therefore a reservation on the call dimensions only, and
  `close_round` never refuses for an overrun: the calls already happened, and
  dropping the round would delete the evidence. The overspend shows up in the
  next `begin_round`, which then has less room.
* Round ids are content hashes and cannot be reversed to a ticket, and the
  round repository offers no lookup by id. A round is found by scanning the open
  tickets, which also states the rule that an ended ticket's rounds are frozen
  history.
* `TicketEnd.by` is fixed to `"executor"`: the P0 command face carries no
  operator identity, and inventing one per caller would make the trace lie.
"""

from __future__ import annotations

from typing import Sequence

from pah.core import ids
from pah.core.errors import UserError
from pah.core.model import (
    Cost,
    FailureCertificate,
    RepairPlan,
    Round,
    RoundVerdict,
    Ticket,
    TicketEnd,
)
from pah.core.ports import Clock, Repositories, utc_now

__all__ = ["Engine"]

#: The one order a round's stage may follow; `proposed` is the start, `judged` is
#: written by `close_round` because a verdict is what makes a round judged.
_STAGE_ORDER = ("proposed", "materialized", "checked", "simulated", "judged")

#: The only stages a caller may advance to. `judged` without a verdict would be
#: a lie, so it is reachable through `close_round` alone.
_ADVANCE = ("materialized", "checked", "simulated")

_VERDICTS = ("pass", "fail", "infeasible", "error", "aborted")

_OUTCOMES = ("repaired", "exhausted", "infeasible", "dropped", "stopped")

#: Budget dimensions in the order they are checked, so a refusal names one cause.
_DIMENSIONS = ("rounds", "geom_calls", "sim_calls")


class Engine:
    """Commands that read, check and record; each one is a single transaction."""

    def __init__(self, repos: Repositories, *, clock: Clock | None = None) -> None:
        """Binds the engine to one store; `clock` is the only source of the timestamps it writes."""
        self._repos = repos
        self._clock = clock if clock is not None else utc_now

    # -- tickets -----------------------------------------------------------

    def open_ticket(self, ticket: Ticket) -> Ticket:
        """Opening an id that already exists returns the stored ticket untouched, so a replayed trigger can never reset a ticket's history."""
        with self._repos.atomic():
            existing = self._repos.tickets.get(ticket.id)
            if existing is not None:
                return existing
            at = self._clock()
            ticket.created_at = at
            ticket.updated_at = at
            self._repos.tickets.save(ticket)
        return ticket

    def end_ticket(self, ticket_id: str, outcome: str, reason: str = "") -> Ticket:
        """Ending happens once: an ended ticket is returned as it stands, so the first outcome, time and reason stay the record."""
        with self._repos.atomic():
            ticket = self._ticket(ticket_id)
            if ticket.ended is not None:
                return ticket
            if outcome not in _OUTCOMES:
                raise UserError(
                    "unknown_outcome",
                    f"{outcome!r} is not an outcome a ticket may end with",
                    {"outcome": outcome, "outcomes": list(_OUTCOMES)},
                )
            at = self._clock()
            ticket.ended = TicketEnd(at=at, by="executor", outcome=outcome, reason=reason)
            ticket.updated_at = at
            self._repos.tickets.save(ticket)
        return ticket

    def note_ticket(self, ticket_id: str, text: str) -> None:
        """A note is recorded in the ticket's input log and grants nothing: no scope widens, no delivery is queued, no budget moves."""
        with self._repos.atomic():
            ticket = self._ticket(ticket_id)
            self._refuse_if_ended(ticket)
            at = self._clock()
            ticket.inputs.append({"kind": "note", "by": "human", "text": text, "at": at})
            ticket.updated_at = at
            self._repos.tickets.save(ticket)

    # -- rounds ------------------------------------------------------------

    def begin_round(self, ticket_id: str, plan: RepairPlan) -> Round:
        """A round is recorded only when a live ticket and its mandate allow that exact plan, in one transaction with the round's row."""
        with self._repos.atomic():
            ticket = self._ticket(ticket_id)
            self._refuse_if_ended(ticket)
            self._check_scope(ticket, plan)
            self._check_budget(ticket, plan)
            index = len(ticket.rounds)
            rnd = Round(id=ids.key("round", ticket_id, index), index=index, plan=plan)
            ticket.rounds.append(rnd)
            ticket.updated_at = self._clock()
            self._repos.tickets.save(ticket)
            self._repos.rounds.save(ticket_id, rnd)
        return rnd

    def stage_round(self, round_id: str, stage: str) -> None:
        """A round's stage only moves forward, and re-staging the stage it already holds writes nothing, so a resumed executor may re-issue it."""
        with self._repos.atomic():
            ticket, rnd = self._locate(round_id)
            if rnd.verdict is not None:
                raise UserError(
                    "round_already_closed",
                    f"round {round_id!r} was closed with verdict {rnd.verdict!r}",
                    {"round_id": round_id, "verdict": rnd.verdict},
                )
            if stage not in _ADVANCE:
                raise UserError(
                    "unknown_stage",
                    f"{stage!r} is not a stage a caller may advance to",
                    {"stage": stage, "advance": list(_ADVANCE)},
                )
            if _STAGE_ORDER.index(stage) < _STAGE_ORDER.index(rnd.stage):
                raise UserError(
                    "stage_regression",
                    f"round {round_id!r} is at {rnd.stage!r} and cannot return to {stage!r}",
                    {"round_id": round_id, "stage": rnd.stage, "requested": stage},
                )
            if stage == rnd.stage:
                return
            rnd.stage = stage
            ticket.updated_at = self._clock()
            self._repos.tickets.save(ticket)
            self._repos.rounds.save(ticket.id, rnd)

    def close_round(
        self,
        round_id: str,
        *,
        verdict: RoundVerdict,
        evidence: Sequence[FailureCertificate],
        cost: Cost,
        artifacts: dict[str, str],
        edit_cost: float = 0.0,
        preserve_ratio: float = 1.0,
    ) -> Round:
        """Closing records the round's one verdict and its measured cost, and a closed round may never be closed again."""
        with self._repos.atomic():
            ticket, rnd = self._locate(round_id)
            if rnd.verdict is not None:
                raise UserError(
                    "round_already_closed",
                    f"round {round_id!r} was already closed with verdict {rnd.verdict!r}",
                    {"round_id": round_id, "verdict": rnd.verdict},
                )
            if verdict not in _VERDICTS:
                raise UserError(
                    "unknown_verdict",
                    f"{verdict!r} is not a verdict the domain model allows",
                    {"verdict": verdict, "verdicts": list(_VERDICTS)},
                )
            rnd.verdict = verdict
            rnd.evidence = list(evidence)
            rnd.cost = cost
            rnd.artifacts = dict(artifacts)
            rnd.edit_cost = edit_cost
            rnd.preserve_ratio = preserve_ratio
            rnd.stage = "judged"
            ticket.updated_at = self._clock()
            self._repos.tickets.save(ticket)
            self._repos.rounds.save(ticket.id, rnd)
        return rnd

    # -- lookups -----------------------------------------------------------

    def _ticket(self, ticket_id: str) -> Ticket:
        """The ticket at `ticket_id`, or a refusal: an unknown id is never invented."""
        ticket = self._repos.tickets.get(ticket_id)
        if ticket is None:
            raise UserError(
                "ticket_missing", f"no ticket with id {ticket_id!r}", {"ticket_id": ticket_id}
            )
        return ticket

    def _refuse_if_ended(self, ticket: Ticket) -> None:
        """Only `end_ticket` may touch an ended ticket; everything else is refused."""
        if ticket.ended is not None:
            raise UserError(
                "ticket_ended",
                f"ticket {ticket.id!r} ended with outcome {ticket.ended.outcome!r}",
                {"ticket_id": ticket.id, "outcome": ticket.ended.outcome},
            )

    def _locate(self, round_id: str) -> tuple[Ticket, Round]:
        """The open ticket holding `round_id` and that round; the rounds of an ended ticket are frozen."""
        for ticket in self._repos.tickets.open():
            for rnd in ticket.rounds:
                if rnd.id == round_id:
                    return ticket, rnd
        raise UserError(
            "round_missing",
            f"no open ticket holds round {round_id!r}",
            {"round_id": round_id},
        )

    # -- the four invariants -----------------------------------------------

    def _check_scope(self, ticket: Ticket, plan: RepairPlan) -> None:
        """The operator and every target must be inside the mandate, and a frozen part stays frozen even when it is in scope."""
        mandate = ticket.mandate
        if plan.operator not in mandate.scope_operations:
            raise UserError(
                "operator_not_in_scope",
                f"operator {plan.operator!r} is not among the mandate's operations",
                {"operator": plan.operator, "scope_operations": list(mandate.scope_operations)},
            )
        outside = [part for part in plan.targets if part not in mandate.scope_parts]
        if outside:
            raise UserError(
                "target_not_in_scope",
                f"the plan would change parts outside the mandate: {outside}",
                {
                    "targets": list(plan.targets),
                    "outside": outside,
                    "scope_parts": list(mandate.scope_parts),
                },
            )
        frozen = [part for part in plan.targets if part in mandate.frozen_parts]
        if frozen:
            raise UserError(
                "target_frozen",
                f"the mandate froze parts the plan would change: {frozen}",
                {"targets": list(plan.targets), "frozen": frozen},
            )

    def _check_budget(self, ticket: Ticket, plan: RepairPlan) -> None:
        """All three dimensions must still fit the plan, counting the round this call is about to add and its estimated calls."""
        budget = ticket.mandate.budget
        used = ticket.used
        needed = {
            "rounds": used.rounds + 1,
            "geom_calls": used.geom_calls + plan.cost_est.geom_calls,
            "sim_calls": used.sim_calls + plan.cost_est.sim_calls,
        }
        ceiling = {
            "rounds": budget.rounds,
            "geom_calls": budget.geom_calls,
            "sim_calls": budget.sim_calls,
        }
        spent = {"rounds": used.rounds, "geom_calls": used.geom_calls, "sim_calls": used.sim_calls}
        for dimension in _DIMENSIONS:
            if needed[dimension] > ceiling[dimension]:
                raise UserError(
                    "budget_exhausted",
                    f"the plan needs {needed[dimension]} {dimension}, the mandate allows {ceiling[dimension]}",
                    # `dimension` is what a caller appends to "budget." for a trace reason.
                    {
                        "dimension": dimension,
                        "needed": needed[dimension],
                        "budget": ceiling[dimension],
                        "used": spent[dimension],
                    },
                )
