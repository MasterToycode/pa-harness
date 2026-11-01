"""SQLite persistence: aggregates as JSON rows, artifacts by content hash.

Two rules from `docs/pah-dev-doc-v1.md` §4.2 and §7.2 decide the shape of this
module.

One transaction per command. `atomic` opens a savepoint, nests savepoints, and
only the outermost release commits; a side effect registered with `after_commit`
runs after that commit and is discarded with the rollback. A command therefore
either fully happened or not at all, which is why the engine needs no lock.

Activity is derived, never stored. The `open` and `active` columns are generated
from the row's own JSON, so a crash cannot leave a status column disagreeing
with the ticket it describes, and a pass scans only the rows it can act on.

§7.2 names the hot columns; the rest of an aggregate rides along in that table's
`meta` JSON so a row still round-trips whole. Ticket and Round each carry
`meta` for exactly that reason, and no field of `pah.core.model` is dropped.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

from pah.core.model import (
    Assignment,
    Attention,
    Budget,
    Cost,
    FailureCertificate,
    Mandate,
    RepairPlan,
    Report,
    Round,
    Ticket,
    TicketEnd,
)

__all__ = [
    "SCHEMA_VERSION",
    "Decision",
    "SchemaError",
    "Store",
    "Thread",
    "content_address",
]

#: Bumped when a table definition changes; there is no migration framework, so
#: an older or newer file is refused rather than guessed at.
SCHEMA_VERSION = 1

_SCHEMA_KEY = "schema.version"

_log = logging.getLogger(__name__)

_TABLES = f"""
CREATE TABLE IF NOT EXISTS kv (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS trigger (
    id TEXT PRIMARY KEY,
    lineage TEXT NOT NULL,
    at TEXT NOT NULL,
    kind TEXT NOT NULL,
    payload TEXT NOT NULL,
    evidence_ref TEXT
);
CREATE TABLE IF NOT EXISTS thread (
    key TEXT PRIMARY KEY,
    assembly_id TEXT,
    pending TEXT NOT NULL DEFAULT '[]',
    evaluating TEXT,
    meta TEXT NOT NULL DEFAULT '{{}}'
);
CREATE TABLE IF NOT EXISTS ticket (
    id TEXT PRIMARY KEY,
    lineage TEXT NOT NULL,
    mandate TEXT NOT NULL,
    rounds TEXT NOT NULL DEFAULT '[]',
    assignments TEXT NOT NULL DEFAULT '[]',
    inputs TEXT NOT NULL DEFAULT '[]',
    ended TEXT NOT NULL DEFAULT 'null',
    meta TEXT NOT NULL DEFAULT '{{}}',
    open INT GENERATED ALWAYS AS (
        CASE WHEN json_extract(ended, '$') IS NULL THEN 1 END) VIRTUAL,
    active INT GENERATED ALWAYS AS (
        CASE WHEN json_extract(ended, '$') IS NULL AND (
            json_array_length(meta, '$.undelivered') > 0
            OR json_extract(assignments, '$[#-1].state') IN ('pending', 'running')
        ) THEN 1 END) VIRTUAL
);
CREATE TABLE IF NOT EXISTS round (
    id TEXT PRIMARY KEY,
    ticket_id TEXT NOT NULL,
    idx INT NOT NULL,
    plan TEXT NOT NULL,
    stage TEXT NOT NULL,
    artifacts TEXT NOT NULL DEFAULT '{{}}',
    evidence TEXT NOT NULL DEFAULT '[]',
    cost TEXT NOT NULL,
    verdict TEXT,
    meta TEXT NOT NULL DEFAULT '{{}}'
);
CREATE TABLE IF NOT EXISTS decision (
    id TEXT PRIMARY KEY,
    lineage TEXT NOT NULL,
    at TEXT NOT NULL,
    kind TEXT NOT NULL,
    reason TEXT NOT NULL DEFAULT '',
    audit TEXT NOT NULL DEFAULT '{{}}'
);
CREATE TABLE IF NOT EXISTS effect (
    kind TEXT NOT NULL,
    key TEXT NOT NULL,
    payload TEXT NOT NULL,
    seq INT NOT NULL,
    generation INT NOT NULL DEFAULT 0,
    attempts INT NOT NULL DEFAULT 0,
    due TEXT NOT NULL,
    error TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (kind, key)
);
CREATE TABLE IF NOT EXISTS artifact (
    hash TEXT PRIMARY KEY,
    path TEXT NOT NULL,
    format TEXT NOT NULL,
    bytes INT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS memory (
    signature TEXT NOT NULL,
    operator TEXT NOT NULL,
    n INT NOT NULL DEFAULT 0,
    success INT NOT NULL DEFAULT 0,
    cost_sum REAL NOT NULL DEFAULT 0,
    PRIMARY KEY (signature, operator)
);
CREATE INDEX IF NOT EXISTS ticket_open ON ticket(open) WHERE open IS NOT NULL;
CREATE INDEX IF NOT EXISTS ticket_active ON ticket(active) WHERE active IS NOT NULL;
CREATE INDEX IF NOT EXISTS round_ticket ON round(ticket_id, idx);
CREATE INDEX IF NOT EXISTS decision_lineage ON decision(lineage);
CREATE INDEX IF NOT EXISTS effect_due ON effect(kind, due, seq);
"""


class SchemaError(RuntimeError):
    """The file on disk was written by another schema version."""


# --------------------------------------------------------------------------
# JSON codecs — a row is the aggregate, so every field has a place to land
# --------------------------------------------------------------------------


def _dumps(value: Any) -> str:
    """Compact JSON text; the same value always produces the same text."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _loads(text: str) -> Any:
    return json.loads(text)


def _cost(data: dict) -> Cost:
    return Cost(**data)


def _plan(data: dict) -> RepairPlan:
    return RepairPlan(
        id=data["id"],
        operator=data["operator"],
        targets=tuple(data["targets"]),
        params=data["params"],
        cost_est=_cost(data["cost_est"]),
        proposed_by=data["proposed_by"],
        reason=data["reason"],
    )


def _certificate(data: dict) -> FailureCertificate:
    return FailureCertificate(
        code=data["code"],
        family=data["family"],
        locus=data["locus"],
        magnitude=data["magnitude"],
        observed_by=data["observed_by"],
        confidence=data["confidence"],
        hypotheses=tuple(data["hypotheses"]),
        evidence_ref=data["evidence_ref"],
    )


def _mandate(data: dict) -> Mandate:
    budget = data["budget"]
    return Mandate(
        strength=data["strength"],
        basis=data["basis"],
        scope_parts=tuple(data["scope_parts"]),
        scope_operations=tuple(data["scope_operations"]),
        frozen_parts=tuple(data["frozen_parts"]),
        budget=Budget(**budget),
        policy_version=data["policy_version"],
    )


def _round(data: dict) -> Round:
    return Round(
        id=data["id"],
        index=data["index"],
        plan=_plan(data["plan"]),
        stage=data["stage"],
        verdict=data["verdict"],
        evidence=[_certificate(c) for c in data["evidence"]],
        cost=_cost(data["cost"]),
        artifacts=dict(data["artifacts"]),
        edit_cost=data["edit_cost"],
        preserve_ratio=data["preserve_ratio"],
    )


def _report(data: dict) -> Report:
    return Report(state=data["state"], summary=data["summary"], at=data["at"])


def _assignment(data: dict) -> Assignment:
    report = data["report"]
    return Assignment(
        id=data["id"],
        state=data["state"],
        created_at=data["created_at"],
        started_at=data["started_at"],
        ended_at=data["ended_at"],
        ended_reason=data["ended_reason"],
        report=_report(report) if report is not None else None,
        idle_at=data["idle_at"],
        delivered_at=data["delivered_at"],
    )


def _ticket_meta(ticket: Ticket) -> dict:
    return {
        "undelivered": list(ticket.undelivered),
        "attention": asdict(ticket.attention) if ticket.attention else None,
        "delivery_failed": ticket.delivery_failed,
        "signoff_required": ticket.signoff_required,
        "signoff_at": ticket.signoff_at,
        "created_at": ticket.created_at,
        "updated_at": ticket.updated_at,
    }


def _ticket_columns(ticket: Ticket) -> tuple:
    """The stored columns of one ticket, in the order `_INSERT_TICKET` binds."""
    return (
        ticket.id,
        ticket.lineage,
        _dumps(asdict(ticket.mandate)),
        _dumps([asdict(r) for r in ticket.rounds]),
        _dumps([asdict(a) for a in ticket.assignments]),
        _dumps(ticket.inputs),
        _dumps(asdict(ticket.ended)) if ticket.ended is not None else "null",
        _dumps(_ticket_meta(ticket)),
    )


def _ticket(row: sqlite3.Row) -> Ticket:
    meta = _loads(row["meta"])
    ended = _loads(row["ended"])
    attention = meta["attention"]
    return Ticket(
        id=row["id"],
        lineage=row["lineage"],
        mandate=_mandate(_loads(row["mandate"])),
        rounds=[_round(d) for d in _loads(row["rounds"])],
        assignments=[_assignment(d) for d in _loads(row["assignments"])],
        inputs=_loads(row["inputs"]),
        undelivered=list(meta["undelivered"]),
        attention=Attention(**attention) if attention is not None else None,
        ended=TicketEnd(**ended) if ended is not None else None,
        delivery_failed=meta["delivery_failed"],
        signoff_required=meta["signoff_required"],
        signoff_at=meta["signoff_at"],
        created_at=meta["created_at"],
        updated_at=meta["updated_at"],
    )


def _round_columns(ticket_id: str, rnd: Round) -> tuple:
    residual = {"edit_cost": rnd.edit_cost, "preserve_ratio": rnd.preserve_ratio}
    return (
        rnd.id,
        ticket_id,
        rnd.index,
        _dumps(asdict(rnd.plan)),
        rnd.stage,
        _dumps(rnd.artifacts),
        _dumps([asdict(c) for c in rnd.evidence]),
        _dumps(asdict(rnd.cost)),
        rnd.verdict,
        _dumps(residual),
    )


def _round_row(row: sqlite3.Row) -> Round:
    residual = _loads(row["meta"])
    data = {
        "id": row["id"],
        "index": row["idx"],
        "plan": _loads(row["plan"]),
        "stage": row["stage"],
        "verdict": row["verdict"],
        "evidence": _loads(row["evidence"]),
        "cost": _loads(row["cost"]),
        "artifacts": _loads(row["artifacts"]),
        "edit_cost": residual["edit_cost"],
        "preserve_ratio": residual["preserve_ratio"],
    }
    return _round(data)


_INSERT_TICKET = """
INSERT OR REPLACE INTO ticket (id, lineage, mandate, rounds, assignments, inputs, ended, meta)
VALUES (?, ?, ?, ?, ?, ?, ?, ?)
"""

_INSERT_ROUND = """
INSERT OR REPLACE INTO round
    (id, ticket_id, idx, plan, stage, artifacts, evidence, cost, verdict, meta)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


# --------------------------------------------------------------------------
# records the store owns, because core's model has no use for them yet
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Thread:
    """One delivery thread: the work queued for an assembly, and what is mid-flight."""

    key: str
    assembly_id: str | None = None
    pending: list[dict] = field(default_factory=list)
    evaluating: dict | None = None
    meta: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Decision:
    """Why an automatic choice was made. Written once, never rewritten."""

    id: str
    lineage: str
    at: str
    kind: str
    reason: str = ""
    audit: dict = field(default_factory=dict)


def content_address(data: bytes) -> str:
    """The sha256 of `data` in hex; equal bytes always name the same artifact."""
    return hashlib.sha256(data).hexdigest()


def _stamp(at: str | None) -> str:
    """`at` when given, else now in UTC — the only place this module reads a clock."""
    return at if at is not None else datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------
# the store
# --------------------------------------------------------------------------


class Store:
    """One SQLite file holding the tickets, their rounds, artifacts and kv state.

    Every table is written through `atomic`; nothing here decides domain rules.
    """

    def __init__(self, path: str | Path) -> None:
        """Opens or creates the database at `path`, refusing any other schema version."""
        path = Path(path)
        if path.parent and not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        # Autocommit mode: transactions are the savepoints `atomic` opens, so a
        # statement outside `atomic` lands immediately and never half-commits.
        self.db = sqlite3.connect(str(path), isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self._callbacks: list[list[Callable[[], None]]] = []
        self.tickets = Tickets(self)
        self.rounds = Rounds(self)
        self.threads = Threads(self)
        self.decisions = Decisions(self)
        self.kv = KV(self)
        self.artifacts = Artifacts(self)
        self._prepare()

    def _prepare(self) -> None:
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        row = self.db.execute("SELECT value FROM kv WHERE key = ?", (_SCHEMA_KEY,)).fetchone()
        if row is not None and _loads(row["value"]) != SCHEMA_VERSION:
            found = _loads(row["value"])
            self.db.close()
            raise SchemaError(
                f"the database is at schema {found}, this build writes {SCHEMA_VERSION}"
            )
        with self.atomic():
            # One statement at a time: `executescript` would commit the
            # savepoint `atomic` just opened.
            for statement in _TABLES.split(";"):
                if statement.strip():
                    self.db.execute(statement)
            self.kv.set(_SCHEMA_KEY, SCHEMA_VERSION)

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # -- transactions ------------------------------------------------------

    @contextmanager
    def atomic(self) -> Iterator[None]:
        """Runs the block as one transaction: nested calls are savepoints, the
        outermost release commits, and any exception rolls everything back."""
        level = len(self._callbacks)
        name = f"pah_atomic_{level}"
        self.db.execute(f"SAVEPOINT {name}")
        self._callbacks.append([])
        try:
            yield
        except BaseException:
            self.db.execute(f"ROLLBACK TO {name}")
            self.db.execute(f"RELEASE {name}")
            self._callbacks.pop()
            raise
        self.db.execute(f"RELEASE {name}")
        actions = self._callbacks.pop()
        if self._callbacks:
            # A nested block handed its callbacks up; only the outermost runs them.
            self._callbacks[-1].extend(actions)
            return
        for action in actions:
            try:
                action()
            except Exception:  # a callback may not undo a committed transaction
                _log.exception("after_commit callback failed")

    def after_commit(self, fn: Callable[[], None]) -> None:
        """Runs `fn` once the outermost transaction committed, immediately when
        there is none, and never when the enclosing transaction rolls back."""
        if self._callbacks:
            self._callbacks[-1].append(fn)
        else:
            fn()

    # -- repositories ------------------------------------------------------
    #: Built in `__init__` so a caller reads `store.tickets.get(...)` and never
    #: remembers a handle; each one borrows this store's connection.
    tickets: Tickets
    rounds: Rounds
    threads: Threads
    decisions: Decisions
    kv: KV
    artifacts: Artifacts

    # -- artifacts ---------------------------------------------------------

    def put_artifact(
        self, data: bytes, fmt: str, root: str | Path, *, at: str | None = None
    ) -> str:
        """Stores `data` at `root/<hash[:2]>/<hash>.<fmt>` and returns its hash;
        the same bytes are written to disk only once."""
        digest = content_address(data)
        target = Path(root) / digest[:2] / f"{digest}.{fmt}"
        row = self.db.execute("SELECT path FROM artifact WHERE hash = ?", (digest,)).fetchone()
        if row is not None and Path(row["path"]).exists():
            return digest
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(data)
        with self.atomic():
            self.db.execute(
                "INSERT OR REPLACE INTO artifact (hash, path, format, bytes, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (digest, str(target), fmt, len(data), _stamp(at)),
            )
        return digest


class Tickets:
    """Reads and writes whole tickets; the active sets come from generated columns."""

    def __init__(self, store: Store) -> None:
        self._store = store

    def get(self, ticket_id: str) -> Ticket | None:
        """The whole ticket, or None when no ticket has that id."""
        row = self._store.db.execute("SELECT * FROM ticket WHERE id = ?", (ticket_id,)).fetchone()
        return None if row is None else _ticket(row)

    def save(self, ticket: Ticket) -> None:
        """Writes the whole aggregate, so the row can never hold half an update."""
        with self._store.atomic():
            self._store.db.execute(_INSERT_TICKET, _ticket_columns(ticket))

    def open(self) -> list[Ticket]:
        """Every ticket that has not ended, oldest row first."""
        rows = self._store.db.execute(
            "SELECT * FROM ticket WHERE open = 1 ORDER BY rowid"
        ).fetchall()
        return [_ticket(row) for row in rows]

    def active(self) -> list[Ticket]:
        """Open tickets that want work: undelivered inputs or a live assignment."""
        rows = self._store.db.execute(
            "SELECT * FROM ticket WHERE active = 1 ORDER BY rowid"
        ).fetchall()
        return [_ticket(row) for row in rows]


class Rounds:
    """Reads and writes rounds, keyed by their ticket and ordered by `index`."""

    def __init__(self, store: Store) -> None:
        self._store = store

    def save(self, ticket_id: str, rnd: Round) -> None:
        """Writes the round under `ticket_id`; the row replaces any earlier one."""
        with self._store.atomic():
            self._store.db.execute(_INSERT_ROUND, _round_columns(ticket_id, rnd))

    def by_ticket(self, ticket_id: str) -> list[Round]:
        """The ticket's rounds in `index` order, which is the order they happened."""
        rows = self._store.db.execute(
            "SELECT * FROM round WHERE ticket_id = ? ORDER BY idx", (ticket_id,)
        ).fetchall()
        return [_round_row(row) for row in rows]


class Threads:
    """Reads and writes threads; `pending` and `evaluating` decide what is live."""

    def __init__(self, store: Store) -> None:
        self._store = store

    def get(self, key: str) -> Thread | None:
        """The thread at `key`, or None when nothing has written it."""
        row = self._store.db.execute("SELECT * FROM thread WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        return Thread(
            key=row["key"],
            assembly_id=row["assembly_id"],
            pending=_loads(row["pending"]),
            evaluating=_loads(row["evaluating"]) if row["evaluating"] is not None else None,
            meta=_loads(row["meta"]),
        )

    def save(self, thread: Thread) -> None:
        """Writes the whole thread at `thread.key`, replacing any earlier row."""
        with self._store.atomic():
            self._store.db.execute(
                "INSERT OR REPLACE INTO thread (key, assembly_id, pending, evaluating, meta)"
                " VALUES (?, ?, ?, ?, ?)",
                (
                    thread.key,
                    thread.assembly_id,
                    _dumps(thread.pending),
                    _dumps(thread.evaluating) if thread.evaluating is not None else None,
                    _dumps(thread.meta),
                ),
            )


class Decisions:
    """Append-only record of automatic choices and the audit trail behind them."""

    def __init__(self, store: Store) -> None:
        self._store = store

    def get(self, decision_id: str) -> Decision | None:
        """The decision at `decision_id`, or None when it was never recorded."""
        row = self._store.db.execute(
            "SELECT * FROM decision WHERE id = ?", (decision_id,)
        ).fetchone()
        if row is None:
            return None
        return Decision(
            id=row["id"],
            lineage=row["lineage"],
            at=row["at"],
            kind=row["kind"],
            reason=row["reason"],
            audit=_loads(row["audit"]),
        )

    def save(self, decision: Decision) -> None:
        """Writes the decision at `decision.id`, replacing an earlier record of it."""
        with self._store.atomic():
            self._store.db.execute(
                "INSERT OR REPLACE INTO decision (id, lineage, at, kind, reason, audit)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (
                    decision.id,
                    decision.lineage,
                    decision.at,
                    decision.kind,
                    decision.reason,
                    _dumps(decision.audit),
                ),
            )


class KV:
    """Small JSON values the engine keeps between runs; no schema, no history."""

    def __init__(self, store: Store) -> None:
        self._store = store

    def get(self, key: str, default: Any = None) -> Any:
        """The value at `key`, or `default` when nothing is stored there."""
        row = self._store.db.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        return default if row is None else _loads(row["value"])

    def set(self, key: str, value: Any) -> None:
        """Stores `value` at `key`; a later `get` returns an equal value."""
        with self._store.atomic():
            self._store.db.execute(
                "INSERT OR REPLACE INTO kv (key, value) VALUES (?, ?)", (key, _dumps(value))
            )

    def delete(self, key: str) -> None:
        """Removes `key`; deleting an absent key is not an error."""
        with self._store.atomic():
            self._store.db.execute("DELETE FROM kv WHERE key = ?", (key,))


class Artifacts:
    """Content-addressed blobs, so a round names geometry by hash (§7.3)."""

    def __init__(self, store: Store) -> None:
        self._store = store

    def put(self, data: bytes, fmt: str, root: str | Path, *, at: str | None = None) -> str:
        """Stores `data` under `root` and returns its content address."""
        return self._store.put_artifact(data, fmt, root, at=at)

    def get(self, digest: str) -> bytes | None:
        """The bytes of the registered artifact, or None when none is registered."""
        row = self._store.db.execute(
            "SELECT path FROM artifact WHERE hash = ?", (digest,)
        ).fetchone()
        if row is None:
            return None
        path = Path(row["path"])
        return path.read_bytes() if path.exists() else None
