"""Durable side effects: queued inside a command, run after it committed.

The five rules are the ones `docs/pah-dev-doc-v1.md` §4.2 lifts from the
prototype, and each is load-bearing:

- one row per `(kind, key)`, so asking twice for the same side effect queues it
  once; the payload is replaced and `generation` grows, while `seq` keeps the
  position the row already earned in the queue;
- `generation` is read before the handler runs and matched again after, so an
  attempt whose payload was replaced while it ran settles nothing;
- `Later` means "not now" — it postpones the row and charges no attempt, so a
  full worker slot is never mistaken for a failing one;
- any other exception charges one attempt, records the error, and doubles the
  wait from 2s up to 60s;
- three consecutive failures on one kind make its health `error`, which is how a
  broken tool is kept out of the failure rate (§4.5).

Nothing here runs on its own: `process_due` is the only clock-driven entry
point, so tests pass a fixed time and the engine keeps one write path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Literal

from pah.app.store import Store

__all__ = [
    "BASE_DELAY_S",
    "FAILURES_BEFORE_ERROR",
    "LATER_DELAY_S",
    "MAX_DELAY_S",
    "Effects",
    "Health",
    "Later",
]

#: First retry wait, and the ceiling the doubling stops at.
BASE_DELAY_S = 2.0
MAX_DELAY_S = 60.0
#: What `Later` postpones by — the initial delay, since "not now" usually means
#: a slot frees up soon.
LATER_DELAY_S = 2.0
#: Consecutive failures after which a kind reports `error`.
FAILURES_BEFORE_ERROR = 3

_Run = Callable[[Any, str], None]


class Later(Exception):
    """Raises when an effect cannot run now: the row waits, no attempt is charged."""


@dataclass(frozen=True)
class Health:
    """What `health` reports for one effect kind; `error` means the tool is broken."""

    kind: str
    state: Literal["ready", "error"]
    detail: str
    attempts: int
    waiting: int


@dataclass(frozen=True)
class _Handler:
    run: _Run
    limit: int


def _stamp(at: datetime) -> str:
    """Fixed-width UTC text, so SQLite's text order is chronological order.

    A naive datetime is read as UTC rather than as local time, so the same
    argument names the same instant on every machine.
    """
    if at.tzinfo is None:
        at = at.replace(tzinfo=timezone.utc)
    return at.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")


class Effects:
    """The durable queue in the `effect` table; one handler per kind, run by hand."""

    def __init__(self, store: Store, *, clock: Callable[[], datetime] | None = None) -> None:
        """Binds the queue to `store`; `clock` is the only source of "now"."""
        self._store = store
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._handlers: dict[str, _Handler] = {}
        self._drained = False
        row = store.db.execute("SELECT IFNULL(MAX(seq), 0) FROM effect").fetchone()
        self._seq = int(row[0])

    # -- queue -------------------------------------------------------------

    def enqueue(
        self, kind: str, key: str, payload: Any, *, delay: float = 0.0
    ) -> None:
        """Queues one effect per `(kind, key)`: a repeat replaces the payload and
        becomes visible to a running attempt as a higher `generation`."""
        due = _stamp(self._clock() + timedelta(seconds=delay))
        self._seq += 1
        with self._store.atomic():
            self._store.db.execute(
                """
                INSERT INTO effect (kind, key, payload, seq, generation, attempts, due, error)
                VALUES (?, ?, ?, ?, 0, 0, ?, '')
                ON CONFLICT(kind, key) DO UPDATE SET
                    payload = excluded.payload,
                    generation = effect.generation + 1,
                    attempts = 0,
                    error = '',
                    due = excluded.due
                """,
                (kind, key, json.dumps(payload, ensure_ascii=False), self._seq, due),
            )

    def handle(self, kind: str, run: _Run, *, limit: int = 1) -> None:
        """Registers the one handler for `kind`; a pass runs at most `limit` of
        its rows, in `seq` order, so `limit=0` holds the kind without dropping it."""
        if kind in self._handlers:
            raise ValueError(f"{kind!r} already has an effect handler")
        self._handlers[kind] = _Handler(run, max(0, limit))

    def process_due(self, now: datetime | None = None) -> int:
        """Runs the due rows of every handled kind, oldest `seq` first, and
        returns how many handlers ran; nothing runs after `drain`."""
        if self._drained:
            return 0
        at = now if now is not None else self._clock()
        stamp = _stamp(at)
        ran = 0
        for kind, handler in list(self._handlers.items()):
            rows = self._store.db.execute(
                "SELECT key, payload, seq, generation, attempts FROM effect"
                " WHERE kind = ? AND due <= ? ORDER BY seq LIMIT ?",
                (kind, stamp, handler.limit),
            ).fetchall()
            for row in rows:
                self._attempt(kind, handler, row, at)
                ran += 1
        return ran

    def _attempt(self, kind: str, handler: _Handler, row: Any, at: datetime) -> None:
        """Runs one row and settles it only while its `generation` is still the
        one it started with; a replaced payload wins."""
        own = "kind = ? AND key = ? AND seq = ? AND generation = ?"
        ident = (kind, row["key"], row["seq"], row["generation"])
        try:
            handler.run(json.loads(row["payload"]), row["key"])
        except Later:
            # "Not now": postpone only, so a full slot costs no retry.
            due = _stamp(at + timedelta(seconds=LATER_DELAY_S))
            self._store.db.execute(
                f"UPDATE effect SET due = ? WHERE {own}", (due, *ident)
            )
        except Exception as error:
            attempts = int(row["attempts"]) + 1
            delay = min(MAX_DELAY_S, BASE_DELAY_S * 2 ** min(int(row["attempts"]), 16))
            self._store.db.execute(
                f"UPDATE effect SET attempts = ?, error = ?, due = ? WHERE {own}",
                (attempts, str(error)[:1000], _stamp(at + timedelta(seconds=delay)), *ident),
            )
        else:
            self._store.db.execute(f"DELETE FROM effect WHERE {own}", ident)

    # -- state -------------------------------------------------------------

    def pending(self, kind: str | None = None) -> int:
        """Counts queued rows, for one kind or for all of them."""
        if kind is None:
            row = self._store.db.execute("SELECT COUNT(*) FROM effect").fetchone()
        else:
            row = self._store.db.execute(
                "SELECT COUNT(*) FROM effect WHERE kind = ?", (kind,)
            ).fetchone()
        return int(row[0])

    def idle(self, kind: str | None = None) -> bool:
        """True when nothing is queued for `kind` (or for any kind)."""
        return self.pending(kind) == 0

    def drain(self) -> None:
        """Stops running effects; every queued row stays in the table for the next run."""
        self._drained = True

    def health(self, kind: str) -> Health:
        """Reports `error` while `kind`'s worst row has failed three times running;
        a successful run removes the row, which is what ends the streak."""
        row = self._store.db.execute(
            "SELECT COUNT(*), IFNULL(MAX(attempts), 0),"
            " (SELECT error FROM effect WHERE kind = ? AND attempts > 0"
            "  ORDER BY seq DESC LIMIT 1)"
            " FROM effect WHERE kind = ?",
            (kind, kind),
        ).fetchone()
        waiting, attempts = int(row[0]), int(row[1])
        error = row[2] or ""
        detail = f"{waiting} waiting" + (f"; the last failure: {error}" if error else "")
        state: Literal["ready", "error"] = (
            "error" if attempts >= FAILURES_BEFORE_ERROR else "ready"
        )
        return Health(
            kind=kind, state=state, detail=detail, attempts=attempts, waiting=waiting
        )
