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
from collections.abc import Iterable
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
    verdict: Literal["repaired", "exhausted", "infeasible", "failed", "cancelled"]
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


def _snapshot_key(assembly: Assembly) -> str:
    """A cache key that changes whenever the geometry or the design changes.

    `id` alone is not enough: nothing in the core model forces a caller to derive
    a fresh id per version, and a double that reuses the id would make a stale
    verdict look current. The fingerprint covers the parts' geometry hashes, the
    joints and the connectors; the id is kept alongside it so two different
    assemblies that happen to hash alike still get separate entries.
    """
    return f"{assembly.id}|{assembly_fingerprint(assembly)}"


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


def _cost_between(earlier: Cost, later: Cost) -> Cost:
    """`later - earlier`, clamped at zero per component.

    A meter can only move forward, but a metric that goes `int -> float` or is
    re-created mid-run must not produce a negative charge: a negative cost would
    let a round *earn* budget, which no report would survive.
    """
    return Cost(
        geom_calls=max(0, later.geom_calls - earlier.geom_calls),
        sim_calls=max(0, later.sim_calls - earlier.sim_calls),
        wall_s=max(0.0, later.wall_s - earlier.wall_s),
        llm_calls=max(0, later.llm_calls - earlier.llm_calls),
        tokens=max(0, later.tokens - earlier.tokens),
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
    #: Asked once per iteration, before any work is spent. A cancelled episode ends
    #: with `verdict="cancelled"` and closes its ticket as `stopped`, which is the
    #: engine's word for "a person ended this" — see `TicketEnd.outcome`. It cannot
    #: interrupt a call already in flight (an HTTP generation that has been submitted
    #: is billed whether or not we wait for it); what it guarantees is that **no new
    #: round starts** after the request, which is what a mis-click costs.
    cancel: Callable[[], bool] | None = None
    #: Backstop only; the real ceiling is the mandate's budget.
    max_rounds: int = 32
    _seen: set[str] = field(default_factory=set, init=False, repr=False)
    #: `assembly.id -> (certs, verified)` for the current run, so a snapshot is
    #: judged once. Assembly ids are content addressed, so a hit is the same
    #: geometry judged by the same verifiers, not a guess.
    _verified: dict[str, tuple[list[FailureCertificate], bool]] = field(
        default_factory=dict, init=False, repr=False
    )
    #: The cascade's meter as of the last charge, and the spent-but-uncharged
    #: remainder. Verification costs money; a round must pay for the checks that
    #: led to it, or an "equal budget" comparison rewards the expensive cascade.
    _charged: Cost = field(default_factory=Cost, init=False, repr=False)
    _pending: Cost = field(default_factory=Cost, init=False, repr=False)

    def run(
        self, ticket: Ticket, assembly: Assembly, *, seen: Iterable[str] = ()
    ) -> EpisodeResult:
        """Drive `ticket` from `assembly` until it passes, runs out, or finds no
        action. The returned assembly is the last one that was verified.

        `seen` is the fingerprint history a *resumed* episode must remember. Cycle
        detection is built on "have I been here before", and a fresh process that
        only knows its current snapshot will happily walk back into geometry the
        first run already ruled a cycle — the two runs then disagree on the round
        count, which is exactly the acceptance check that fails. Callers pass the
        fingerprints already written in the trace; the default empty set keeps the
        single-process behaviour unchanged.
        """
        self._seen = {assembly_fingerprint(assembly), *seen}
        self._verified = {}
        self._charged = self._meter_now()
        self._pending = Cost()
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
            if self.cancel is not None and self.cancel():
                return self._finish(
                    ticket, "cancelled", "cancelled by request", assembly, certs, total_cost, used
                )
            certs, verified = self._verified_or_run(assembly, ticket.id, phase="before")
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
                charged = self._charge(plan.cost_est)
                closed = self._close(round_, "error", [], charged, new_assembly=None)
                history.append(closed)
                total_cost = _add_cost(total_cost, charged)
                used = _spend(used, charged)
                continue

            self.engine.stage_round(round_.id, "materialized")
            after, verified_after = self._verified_or_run(
                new_assembly, ticket.id, phase="after"
            )

            fingerprint = assembly_fingerprint(new_assembly)
            cycled = fingerprint in self._seen
            self._seen.add(fingerprint)

            if not verified_after:
                # Unverified after repair is an error round, never a pass.
                verdict = "error"
            else:
                verdict = "fail" if hard_failures(after) else "pass"

            charged = self._charge(plan.cost_est)
            closed = self._close(round_, verdict, after, charged, new_assembly=new_assembly)
            history.append(closed)
            total_cost = _add_cost(total_cost, charged)
            used = _spend(used, charged)
            self._emit(
                "round.close",
                ticket=ticket.id,
                round=closed.index,
                verdict=verdict,
                geom_calls=charged.geom_calls,
                sim_calls=charged.sim_calls,
                llm_calls=charged.llm_calls,
                tokens=charged.tokens,
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

    def _verified_or_run(
        self, assembly: Assembly, ticket_id: str, *, phase: str
    ) -> tuple[list[FailureCertificate], bool]:
        """Verify `assembly`, reusing the last verdict for the same snapshot.

        Every round verifies its output and then the next iteration used to
        verify the *same* snapshot again — twice the expensive tier for one
        answer. The cache is keyed by the snapshot's content fingerprint, which
        is what a hit has to mean: the same geometry judged by the same
        verifiers. Keying on `assembly.id` alone would be a bet that every caller
        derives a fresh id per version, and one caller (a test double, a
        hand-built assembly) would silently turn a stale verdict into a pass.
        The reuse is still written to the trace, flagged `cached`: "we did not
        re-run it" is a fact a reader of the trace needs.
        """
        hit = self._verified.get(_snapshot_key(assembly))
        if hit is None:
            certs, verified = self._verify(assembly, ticket_id, phase=phase)
            # Only a run that actually answered is remembered. A verifier that
            # raised, or that reported an environment-family certificate, leaves
            # the state unknown — and the documented retry for a transient failure
            # (`docs/pah-dev-doc-v1.md` §4.5: the 388 -> 393 retry after an
            # intermittent OOM) is only possible if the next look is a fresh one.
            if verified:
                self._verified[_snapshot_key(assembly)] = (list(certs), verified)
            return certs, verified
        certs, verified = hit
        self._emit(
            "verify",
            ticket=ticket_id,
            phase=phase,
            codes=[cert.code for cert in certs],
            physical=len(hard_failures(certs)),
            verified=verified,
            cached=True,
        )
        return list(certs), verified

    def _meter_now(self) -> Cost:
        """What the cascade says it has spent, or zero when it says nothing.

        `LoopVerifiers` does not require a meter — a deterministic stub in a test
        has nothing to report — so the port is read defensively: a `spent`
        attribute or method returning a `Cost`, else nothing. A cascade that
        under-reports is not an error here; it simply cannot charge for work it
        did not declare, which is why the physics tier is required to report.
        """
        source = getattr(self.verifiers, "spent", None)
        if source is None:
            return Cost()
        value = source() if callable(source) else source
        if isinstance(value, Cost):
            return value
        if isinstance(value, dict):
            return Cost(
                geom_calls=int(value.get("geom_calls", 0) or 0),
                sim_calls=int(value.get("sim_calls", 0) or 0),
                wall_s=float(value.get("wall_s", 0.0) or 0.0),
                llm_calls=int(value.get("llm_calls", 0) or 0),
                tokens=int(value.get("tokens", 0) or 0),
            )
        return Cost()

    def _charge(self, estimate: Cost) -> Cost:
        """The round's real cost: what the plan estimated plus what verification spent.

        A round is charged after it closes, so the arithmetic the engine checks
        (`begin_round` refuses a plan the budget cannot pay for) still uses the
        declared estimate. The extra appears in the closed round's `cost`, which
        is what a budget report sums — estimating only the declared part would let
        an expensive verifier cascade look free.
        """
        now = self._meter_now()
        spent = _cost_between(self._charged, now)
        self._charged = now
        self._pending = _add_cost(self._pending, spent)
        charged = _add_cost(estimate, self._pending)
        self._pending = Cost()
        return charged

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
        # An environment-family certificate means a tool could not answer even when no
        # exception escaped — a cascade is allowed to catch per-verifier failures
        # itself (measured: a printability check that could not locate the interface
        # band). Unknown must never read as a pass, so it demotes the whole
        # verification here rather than only removing itself from the hard failures.
        if any(cert.family == "environment" for cert in certs):
            verified = False
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
        # Verification that no round ever paid for still happened: a clean design is
        # judged once and the episode ends with zero rounds, so without this the most
        # common successful episode would report zero geometric work.
        spent = _cost_between(self._charged, self._meter_now())
        self._charged = self._meter_now()
        total = _add_cost(_add_cost(cost, self._pending), spent)
        self._pending = Cost()
        outcome = {
            "repaired": "repaired",
            "exhausted": "exhausted",
            "infeasible": "infeasible",
            "failed": "stopped",
            "cancelled": "stopped",
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
            cost=total,
        )

    def _emit(self, kind: str, **fields: object) -> None:
        if self.observer is not None:
            self.observer(kind, dict(fields))
