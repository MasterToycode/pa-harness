"""The repair loop: verify -> decide -> act -> re-verify -> record.

This module is the P0 orchestrator. It holds the loop's *shape* and none of its
domain rules: authorization and budget are the engine's, action legality is the
registry's, and failure evidence is the verifiers'. Everything it touches is a
protocol, so a deterministic policy, a pretrained model and a search strategy
are all interchangeable here — and so is a fake in a test.

The shape, from `docs/internal/repair-loop-spec.md` §1:

    assembly --verify--> certificates --policy--> plan --registry--> assembly'
        ^                                                                |
        +------------------------ record one Round ----------------------+

Four stop conditions, hard-coded (spec §3): no physical failure left, budget
spent, no applicable action, and a geometry cycle. A fifth, `max_rounds`, is a
backstop that keeps a buggy policy from spinning.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import Callable, Literal, Protocol

from pah.core.ids import key
from pah.core.model import (
    Assembly,
    Budget,
    BudgetUsed,
    Cost,
    FailureCertificate,
    RepairPlan,
    Round,
    Ticket,
)

__all__ = [
    "EpisodeResult",
    "LoopEngine",
    "LoopPolicy",
    "LoopRegistry",
    "LoopVerifiers",
    "Observer",
    "RepairLoop",
    "assembly_fingerprint",
    "hard_failures",
]

#: Environment-family certificates say the verifier could not run. They are not
#: physical failures and never keep a loop going (spec §2.1, doc §4.5).
_ENVIRONMENT = "environment"


class LoopEngine(Protocol):
    """The engine commands the loop uses. The engine enforces scope, frozen
    parts and budget; the loop never re-implements those checks."""

    def begin_round(self, ticket_id: str, plan: RepairPlan) -> Round: ...

    def stage_round(self, round_id: str, stage: str) -> None: ...

    def close_round(
        self,
        round_id: str,
        *,
        verdict: str,
        evidence: list[FailureCertificate],
        cost: Cost,
        artifacts: dict[str, str],
        edit_cost: float = 0.0,
        preserve_ratio: float = 1.0,
    ) -> Round: ...

    def end_ticket(self, ticket_id: str, outcome: str, reason: str = "") -> Ticket: ...


class LoopVerifiers(Protocol):
    """A cascade of verifiers, cheapest first. It returns evidence, never a
    verdict the loop would have to trust blindly."""

    def check(self, assembly: Assembly) -> list[FailureCertificate]: ...


class LoopRegistry(Protocol):
    """The action space: which operators may run on this evidence, and applying
    one legally."""

    def applicable(self, assembly: Assembly, certs: list[FailureCertificate]) -> list[str]: ...

    def apply(self, assembly: Assembly, plan: RepairPlan) -> Assembly: ...


class LoopPolicy(Protocol):
    """A decision. `None` means "no action I am willing to take", which is a
    normal answer, not an error."""

    name: str

    def propose(
        self,
        assembly: Assembly,
        certs: list[FailureCertificate],
        history: list[Round],
        budget_left: Budget,
    ) -> RepairPlan | None: ...


#: Receives `(kind, fields)` for every step worth keeping. `app.trace.Trace.emit`
#: satisfies it; tests pass a list appender.
Observer = Callable[[str, dict], None]


@dataclass(frozen=True)
class EpisodeResult:
    """What one loop run leaves behind. `verdict` is the loop's own word, kept
    distinct from the ticket's outcome vocabulary."""

    ticket_id: str
    verdict: Literal["repaired", "exhausted", "infeasible", "failed"]
    reason: str
    rounds: int
    assembly: Assembly
    certificates: tuple[FailureCertificate, ...]
    cost: Cost

    @property
    def repaired(self) -> bool:
        return self.verdict == "repaired"


def hard_failures(certs: list[FailureCertificate]) -> list[FailureCertificate]:
    """The certificates that mean the geometry is still wrong.

    Environment-family entries — a verifier that ran out of memory, an engine
    that crashed — are excluded, so a broken tool can never masquerade as a
    physical defect, and never as a pass either: they are simply dropped from
    both counts and reported separately.
    """
    return [c for c in certs if c.family != _ENVIRONMENT]


def assembly_fingerprint(assembly: Assembly) -> str:
    """A stable hash of everything a repair can change.

    Used for cycle detection: a plan that returns geometry already seen cannot
    make progress, so the loop stops instead of burning the budget on it.
    """
    parts = tuple(
        sorted((str(p.get("id")), _digest((p.get("geom") or {}).get("hash"))) for p in assembly.parts)
    )
    joints = tuple(sorted(_canonical(j) for j in assembly.joints))
    connectors = tuple(sorted(_canonical(c) for c in assembly.connectors))
    return key("assembly-fingerprint", parts, joints, connectors)


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, default=str, ensure_ascii=False)


def _digest(value: object) -> str:
    return str(value) if value is not None else ""


def _add_cost(total: Cost, extra: Cost) -> Cost:
    return Cost(
        geom_calls=total.geom_calls + extra.geom_calls,
        sim_calls=total.sim_calls + extra.sim_calls,
        wall_s=total.wall_s + extra.wall_s,
        llm_calls=total.llm_calls + extra.llm_calls,
        tokens=total.tokens + extra.tokens,
    )


def _spend(used: BudgetUsed, cost: Cost) -> BudgetUsed:
    return BudgetUsed(
        rounds=used.rounds + 1,
        geom_calls=used.geom_calls + cost.geom_calls,
        sim_calls=used.sim_calls + cost.sim_calls,
    )


def _left(budget: Budget, used: BudgetUsed) -> Budget:
    return Budget(
        rounds=max(0, budget.rounds - used.rounds),
        geom_calls=max(0, budget.geom_calls - used.geom_calls),
        sim_calls=max(0, budget.sim_calls - used.sim_calls),
    )


def _fits(cost: Cost, left: Budget) -> bool:
    """Whether a round's *declared* cost can be paid for before it starts.

    The engine still checks the authoritative arithmetic; this only keeps the
    episode from beginning a round that must be refused.
    """
    return cost.geom_calls <= left.geom_calls and cost.sim_calls <= left.sim_calls


def _spent_all(left: Budget) -> bool:
    return left.rounds <= 0 or left.geom_calls <= 0 or left.sim_calls <= 0


@dataclass
class RepairLoop:
    """Runs one ticket to a stop condition and returns what it left behind.

    The loop owns no state that survives the run: every round is written through
    the engine, so a process killed mid-episode resumes from the last closed
    round rather than from memory.
    """

    engine: LoopEngine
    verifiers: LoopVerifiers
    registry: LoopRegistry
    policy: LoopPolicy
    observer: Observer | None = None
    #: Backstop only; the real ceiling is the mandate's budget.
    max_rounds: int = 32
    _seen: set[str] = field(default_factory=set, init=False, repr=False)

    def run(self, ticket: Ticket, assembly: Assembly) -> EpisodeResult:
        """Drive `ticket` from `assembly` until it passes, runs out, or finds no
        action. The returned assembly is the last one that was verified."""
        self._seen = {assembly_fingerprint(assembly)}
        used = ticket.used
        budget = ticket.mandate.budget
        total_cost = Cost()
        # A ticket read back from the store already carries its rounds, so a run
        # that was killed resumes from the last closed round instead of repeating
        # it. The policy also sees that history, which is how it avoids retrying
        # an action that already failed.
        history: list[Round] = list(ticket.rounds)
        certs: list[FailureCertificate] = []

        self._emit("episode.begin", ticket=ticket.id, policy=self.policy.name, assembly=ticket.id)

        while True:
            certs, verified = self._verify(assembly, ticket.id, phase="before")
            if not verified:
                # A verifier that cannot run leaves the state unknown. Unknown is
                # never a pass and never a physical failure: the episode stops.
                return self._finish(
                    ticket, "failed", "verifier could not run", assembly, certs, total_cost, used
                )

            physical = hard_failures(certs)
            if not physical:
                return self._finish(
                    ticket, "repaired", "no physical failure", assembly, certs, total_cost, used
                )

            left = _left(budget, used)
            if _spent_all(left):
                return self._finish(
                    ticket, "exhausted", "budget spent", assembly, certs, total_cost, used
                )
            if len(history) >= self.max_rounds:
                return self._finish(
                    ticket, "exhausted", "max_rounds reached", assembly, certs, total_cost, used
                )

            plan = self.policy.propose(assembly, physical, history, left)
            if plan is None:
                return self._finish(
                    ticket, "infeasible", "no applicable action", assembly, certs, total_cost, used
                )

            if not _fits(plan.cost_est, left):
                # Starting a round that cannot be paid for would let the engine
                # refuse it halfway; saying so up front keeps the trace honest.
                return self._finish(
                    ticket, "exhausted", "next action exceeds budget", assembly, certs, total_cost, used
                )

            self._emit(
                "round.begin",
                ticket=ticket.id,
                round=len(history),
                operator=plan.operator,
                targets=list(plan.targets),
                policy=self.policy.name,
            )
            try:
                round_ = self.engine.begin_round(ticket.id, plan)
            except Exception as error:
                # The engine refused the round: scope, frozen parts or budget.
                # Nothing was consumed, so this ends the episode rather than
                # retrying the same illegal plan.
                self._emit(
                    "round.refused",
                    ticket=ticket.id,
                    operator=plan.operator,
                    error=f"{type(error).__name__}: {error}",
                )
                return self._finish(
                    ticket, "failed", f"round refused: {error}", assembly, certs, total_cost, used
                )

            try:
                new_assembly = self.registry.apply(assembly, plan)
            except Exception as error:  # an operator that cannot run is an error, not a fail
                self._emit(
                    "operator.error",
                    ticket=ticket.id,
                    round=round_.index,
                    operator=plan.operator,
                    error=f"{type(error).__name__}: {error}",
                )
                closed = self._close(round_, "error", [], plan.cost_est, new_assembly=None)
                history.append(closed)
                total_cost = _add_cost(total_cost, plan.cost_est)
                used = _spend(used, plan.cost_est)
                continue

            self.engine.stage_round(round_.id, "materialized")
            after, verified_after = self._verify(new_assembly, ticket.id, phase="after")

            fingerprint = assembly_fingerprint(new_assembly)
            cycled = fingerprint in self._seen
            self._seen.add(fingerprint)

            if not verified_after:
                # Unverified after repair is an error round, never a pass.
                verdict = "error"
            else:
                verdict = "fail" if hard_failures(after) else "pass"

            closed = self._close(round_, verdict, after, plan.cost_est, new_assembly=new_assembly)
            history.append(closed)
            total_cost = _add_cost(total_cost, plan.cost_est)
            used = _spend(used, plan.cost_est)
            self._emit(
                "round.close",
                ticket=ticket.id,
                round=closed.index,
                verdict=verdict,
                geom_calls=plan.cost_est.geom_calls,
                sim_calls=plan.cost_est.sim_calls,
            geom_hash=assembly_fingerprint(new_assembly),
            )

            if verdict == "pass":
                return self._finish(
                    ticket, "repaired", "verified after repair", new_assembly, after, total_cost, used
                )

            assembly = new_assembly
            certs = after

            if cycled and verdict == "fail":
                return self._finish(
                    ticket, "infeasible", "geometry cycle", assembly, certs, total_cost, used
                )

    # -- internals ---------------------------------------------------------

    def _verify(
        self, assembly: Assembly, ticket_id: str, *, phase: str
    ) -> tuple[list[FailureCertificate], bool]:
        """Runs the cascade.

        Returns the evidence and whether every verifier actually ran. The second
        value is the whole point: a crashed verifier leaves the state *unknown*,
        and unknown must never be read as a pass — that is the same rule the
        dataset scan uses for interpenetration (doc §4.8).
        """
        verified = True
        try:
            certs = list(self.verifiers.check(assembly))
        except Exception as error:
            verified = False
            self._emit(
                "verifier.error",
                ticket=ticket_id,
                phase=phase,
                error=f"{type(error).__name__}: {error}",
            )
            certs = [
                FailureCertificate(
                    code="verifier_error",
                    family="environment",
                    locus={"phase": phase},
                    magnitude={"error": f"{type(error).__name__}: {error}"},
                    observed_by="repair_loop",
                )
            ]
        self._emit(
            "verify",
            ticket=ticket_id,
            phase=phase,
            codes=[c.code for c in certs],
            physical=len(hard_failures(certs)),
            verified=verified,
        )
        return certs, verified

    def _close(
        self,
        round_: Round,
        verdict: str,
        certs: list[FailureCertificate],
        cost: Cost,
        *,
        new_assembly: Assembly | None,
    ) -> Round:
        artifacts = {"geom_hash": assembly_fingerprint(new_assembly)} if new_assembly else {}
        return self.engine.close_round(
            round_.id,
            verdict=verdict,
            evidence=list(certs),
            cost=cost,
            artifacts=artifacts,
        )

    def _finish(
        self,
        ticket: Ticket,
        verdict: Literal["repaired", "exhausted", "infeasible", "failed"],
        reason: str,
        assembly: Assembly,
        certs: list[FailureCertificate],
        cost: Cost,
        used: BudgetUsed,
    ) -> EpisodeResult:
        outcome = {
            "repaired": "repaired",
            "exhausted": "exhausted",
            "infeasible": "infeasible",
            "failed": "stopped",
        }[verdict]
        self.engine.end_ticket(ticket.id, outcome, reason)
        self._emit(
            "ticket.end",
            ticket=ticket.id,
            outcome=outcome,
            reason=reason,
            rounds=used.rounds,
            geom_calls=used.geom_calls,
            sim_calls=used.sim_calls,
        )
        return EpisodeResult(
            ticket_id=ticket.id,
            verdict=verdict,
            reason=reason,
            rounds=used.rounds,
            assembly=assembly,
            certificates=tuple(certs),
            cost=cost,
        )

    def _emit(self, kind: str, **fields: object) -> None:
        if self.observer is not None:
            self.observer(kind, dict(fields))
