"""Domain types for PAH.

Pure data plus the shared predicates the engine and the stage function use.
No I/O, no imports outside the standard library — the same constraint the
TypeScript prototype keeps on `src/core`, and the reason its logic ports
faithfully (`docs/pah-dev-doc-v1.md` §4.2).

The one non-obvious rule: a Ticket stores no status enum. What a person sees is
derived from stored facts by `pah.core.stage.ticket_stage`, so a restart cannot
make the state lie.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

__all__ = [
    "Assembly",
    "Assignment",
    "AssignmentState",
    "Attention",
    "Budget",
    "BudgetUsed",
    "Cost",
    "FailureCertificate",
    "Mandate",
    "RepairPlan",
    "Report",
    "ReportState",
    "Round",
    "RoundVerdict",
    "Ticket",
    "TicketEnd",
    "latest_assignment",
    "latest_report",
    "latest_round",
    "live",
    "standing_report",
]

AssignmentState = Literal["pending", "running", "ended"]

#: What an executor reports. `blocked` and `failed` are what the worker says,
#: not a Ticket state; `waits_for_person` reads them.
ReportState = Literal["progress", "waiting", "ready", "blocked", "failed", "dropped"]

RoundVerdict = Literal["pass", "fail", "infeasible", "error", "aborted"]


@dataclass(frozen=True)
class Budget:
    """Hard ceilings. A ceiling, never a target — an agent that reads a budget
    as a target will keep looking for defects until it runs out."""

    rounds: int = 5
    geom_calls: int = 200
    sim_calls: int = 40


@dataclass
class BudgetUsed:
    rounds: int = 0
    geom_calls: int = 0
    sim_calls: int = 0

    def spend(self, *, rounds: int = 0, geom_calls: int = 0, sim_calls: int = 0) -> None:
        self.rounds += rounds
        self.geom_calls += geom_calls
        self.sim_calls += sim_calls

    def exhausted(self, budget: Budget) -> bool:
        return (
            self.rounds >= budget.rounds
            or self.geom_calls >= budget.geom_calls
            or self.sim_calls >= budget.sim_calls
        )


@dataclass(frozen=True)
class Cost:
    geom_calls: int = 0
    sim_calls: int = 0
    wall_s: float = 0.0
    llm_calls: int = 0
    tokens: int = 0


@dataclass(frozen=True)
class RepairPlan:
    """A proposal to change geometry. Produced by a policy or a person, never
    by a verifier: a verifier may only say what is wrong."""

    id: str
    operator: str
    targets: tuple[str, ...]
    params: dict
    cost_est: Cost = field(default_factory=Cost)
    proposed_by: Literal["rule", "case", "bandit", "llm", "human"] = "rule"
    reason: str = ""


@dataclass(frozen=True)
class FailureCertificate:
    """Structured evidence. `observed_*` may only be written by a registered
    verifier; `hypotheses` carry what an agent suspects and stays untested.

    `family="environment"` marks a verifier that could not run (out of memory,
    crashed engine). It is not a physical failure and must be excluded from any
    failure-rate denominator — the same rule that keeps a broken tool out of the
    numbers in `docs/pah-dev-doc-v1.md` §4.5.
    """

    code: str
    family: Literal["geometry", "assembly", "structural", "manufacturing", "environment"]
    locus: dict
    magnitude: dict
    observed_by: str
    confidence: float = 1.0
    hypotheses: tuple[dict, ...] = ()
    evidence_ref: str | None = None


@dataclass
class Round:
    """One attempt: propose -> materialize -> check -> simulate -> judge."""

    id: str
    index: int
    plan: RepairPlan
    stage: Literal[
        "proposed", "materialized", "checked", "simulated", "judged"
    ] = "proposed"
    verdict: RoundVerdict | None = None
    evidence: list[FailureCertificate] = field(default_factory=list)
    cost: Cost = field(default_factory=Cost)
    artifacts: dict[str, str] = field(default_factory=dict)
    edit_cost: float = 0.0
    preserve_ratio: float = 1.0

    @property
    def judged(self) -> bool:
        return self.verdict is not None


@dataclass(frozen=True)
class Report:
    state: ReportState
    summary: str
    at: str


@dataclass
class Assignment:
    id: str
    state: AssignmentState = "pending"
    created_at: str = ""
    started_at: str | None = None
    ended_at: str | None = None
    ended_reason: str | None = None
    report: Report | None = None
    idle_at: str | None = None
    #: When Inputs last reached the worker; at or after `report.at` means the
    #: worker has been busy with something newer than that report.
    delivered_at: str | None = None


@dataclass(frozen=True)
class Attention:
    """A question for a person. The only way a Ticket asks for one."""

    id: str
    question: str
    at: str = ""


@dataclass(frozen=True)
class Mandate:
    """Why the work may be done, and how far it may go. Immutable: a wider
    scope needs a new Ticket."""

    strength: Literal["explicit", "inferred"]
    basis: dict
    scope_parts: tuple[str, ...]
    scope_operations: tuple[str, ...]
    frozen_parts: tuple[str, ...]
    budget: Budget
    policy_version: str


@dataclass(frozen=True)
class TicketEnd:
    at: str
    by: Literal["executor", "human"]
    outcome: Literal["repaired", "exhausted", "infeasible", "dropped", "stopped"]
    reason: str = ""


@dataclass
class Ticket:
    id: str
    lineage: str
    mandate: Mandate
    rounds: list[Round] = field(default_factory=list)
    assignments: list[Assignment] = field(default_factory=list)
    inputs: list[dict] = field(default_factory=list)
    undelivered: list[str] = field(default_factory=list)
    attention: Attention | None = None
    ended: TicketEnd | None = None
    delivery_failed: bool = False
    signoff_required: bool = False
    signoff_at: str | None = None
    created_at: str = ""
    updated_at: str = ""

    @property
    def used(self) -> BudgetUsed:
        return BudgetUsed(
            rounds=sum(1 for r in self.rounds),
            geom_calls=sum(r.cost.geom_calls for r in self.rounds),
            sim_calls=sum(r.cost.sim_calls for r in self.rounds),
        )

    @property
    def passed(self) -> bool:
        return any(r.verdict == "pass" for r in self.rounds)


@dataclass(frozen=True)
class Assembly:
    """A versioned design snapshot. Immutable; every change makes a new
    version whose `parent` points back."""

    id: str
    version: int
    parts: tuple[dict, ...]
    joints: tuple[dict, ...]
    connectors: tuple[dict, ...]
    constraints: dict
    parent: str | None = None


# --------------------------------------------------------------------------
# shared predicates — the only place these rules may live
# --------------------------------------------------------------------------


def live(ticket: Ticket) -> bool:
    """A ticket is open while it has not ended."""
    return ticket.ended is None


def latest_assignment(ticket: Ticket) -> Assignment | None:
    return ticket.assignments[-1] if ticket.assignments else None


def latest_report(ticket: Ticket) -> Report | None:
    assignment = latest_assignment(ticket)
    return assignment.report if assignment else None


def latest_round(ticket: Ticket) -> Round | None:
    return ticket.rounds[-1] if ticket.rounds else None


def standing_report(ticket: Ticket) -> Report | None:
    """The report that still describes the ticket.

    A report stops standing once an input reached the worker after it, because
    the worker has had something new to act on since.
    """
    assignment = latest_assignment(ticket)
    if assignment is None or assignment.report is None:
        return None
    report = assignment.report
    if assignment.delivered_at is not None and assignment.delivered_at >= report.at:
        return None
    return report
