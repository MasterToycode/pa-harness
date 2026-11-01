"""Run one repair episode end to end and leave its artifacts on disk.

This is the P0 smoke run: a real SQLite store, the real engine, the real trace
file and the real rule policy, over a synthetic two-part assembly. It exists so
that "the loop works" is a file you can open, not a claim in a message.

    python scripts/loop_demo.py --out runs/loop-demo
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from pah.app.store import Store
from pah.app.trace import Trace
from pah.core.engine import Engine
from pah.core.model import (
    Assembly,
    Budget,
    FailureCertificate,
    Mandate,
    RepairPlan,
    Ticket,
)
from pah.policy.loop import RepairLoop
from pah.policy.rule import RulePolicy

PASS_RADIUS = 0.80
OPERATORS = ("resize_connector", "relocate_connector", "swap_family")


class SyntheticVerifier:
    """A stand-in for the geometry cascade: it passes once the pin is thick enough."""

    def check(self, assembly: Assembly) -> list[FailureCertificate]:
        radius = float(assembly.connectors[0]["params"]["r"])
        if radius >= PASS_RADIUS:
            return []
        return [
            FailureCertificate(
                code="gap_too_wide",
                family="assembly",
                locus={"parts": ("p0", "p1"), "joint": "j0"},
                magnitude={"value": round(PASS_RADIUS - radius, 6), "unit": "mm"},
                observed_by="synthetic_verifier@1",
            )
        ]


class SyntheticRegistry:
    """Applies a plan by widening the connector, which is the only observable the
    loop needs in order to close."""

    def applicable(self, assembly: Assembly, certs: list[FailureCertificate]) -> list[str]:
        return list(OPERATORS) if certs else []

    def get(self, operator_id: str):
        return operator_id

    def apply(self, assembly: Assembly, plan: RepairPlan) -> Assembly:
        radius = float(assembly.connectors[0]["params"]["r"])
        step = 0.2 if plan.operator == "resize_connector" else 0.05
        return Assembly(
            id=assembly.id,
            version=assembly.version + 1,
            parts=assembly.parts,
            joints=assembly.joints,
            connectors=({"id": "c0", "family": "pin", "params": {"r": round(min(1.0, radius + step), 6)}},),
            constraints=assembly.constraints,
            parent=f"{assembly.id}@v{assembly.version}",
        )


def make_assembly(radius: float = 0.5) -> Assembly:
    return Assembly(
        id="asm_demo",
        version=0,
        parts=({"id": "p0", "geom": {"hash": "h0"}}, {"id": "p1", "geom": {"hash": "h1"}}),
        joints=({"id": "j0", "a": "p0", "b": "p1"},),
        connectors=({"id": "c0", "family": "pin", "params": {"r": radius}},),
        constraints={"process": "SLA"},
    )


def make_ticket(rounds: int) -> Ticket:
    return Ticket(
        id="t-demo",
        lineage="asm:demo",
        mandate=Mandate(
            strength="explicit",
            basis={"via": "demo"},
            scope_parts=("p0", "p1"),
            scope_operations=OPERATORS,
            frozen_parts=(),
            budget=Budget(rounds=rounds, geom_calls=20, sim_calls=10),
            policy_version="pv-demo",
        ),
        created_at="2026-01-01T00:00:00",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="runs/loop-demo")
    parser.add_argument("--rounds", type=int, default=4)
    args = parser.parse_args()

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    store = Store(out / "db.sqlite")
    engine = Engine(store)
    registry = SyntheticRegistry()
    trace_path = out / "trace.jsonl"

    try:
        with Trace(trace_path) as trace:
            ticket = make_ticket(args.rounds)
            engine.open_ticket(ticket)
            loop = RepairLoop(
                engine=engine,
                verifiers=SyntheticVerifier(),
                registry=registry,
                policy=RulePolicy(registry=registry),
                observer=lambda kind, fields: trace.emit(kind, **fields),
            )
            result = loop.run(store.tickets.get(ticket.id), make_assembly())
    finally:
        store.close()

    summary = {
        "verdict": result.verdict,
        "reason": result.reason,
        "rounds": result.rounds,
        "final_radius": result.assembly.connectors[0]["params"]["r"],
        "replay_hashes": Trace.replay_hashes(trace_path),
        "trace_rows": len(Trace(trace_path).rows()),
        "ticket_ended": store_ticket_outcome(out / "db.sqlite", ticket.id),
    }
    (out / "episode.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), "utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def store_ticket_outcome(db: Path, ticket_id: str) -> str | None:
    """Reopens the database to prove the outcome survived the process, not just
    the in-memory object."""
    store = Store(db)
    try:
        ended = store.tickets.get(ticket_id)
        return ended.ended.outcome if ended and ended.ended else None
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
