"""What the domain core is allowed to depend on.

`core` holds the rules; it may name the shape of a store but may never import
one. Everything here is a `Protocol`, so `pah.app.store.Store` satisfies it by
having the right methods rather than by inheriting from anything, and the test
double in `tests/test_engine.py` satisfies it the same way. That keeps `core`
free of I/O and keeps the engine testable against a store that rolls back.

`Clock` is a port for the same reason. A command stamps times; a test that cannot
choose those times cannot pin the bytes it wrote, and a run whose timestamps
come from an ambient clock cannot be replayed.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from datetime import datetime, timezone
from typing import Callable, Protocol

from pah.core.model import Round, Ticket

__all__ = ["Clock", "Repositories", "RoundsRepository", "TicketsRepository", "utc_now"]

#: Returns an ISO-8601 instant. Injected so one run's timestamps are reproducible.
Clock = Callable[[], str]


def utc_now() -> str:
    """The current UTC instant as ISO-8601, the stamp the engine uses by default."""
    return datetime.now(timezone.utc).isoformat()


class TicketsRepository(Protocol):
    """Whole tickets as aggregates: one read and one write, never a field at a time."""

    def get(self, ticket_id: str) -> Ticket | None: ...

    def save(self, ticket: Ticket) -> None: ...

    def open(self) -> list[Ticket]: ...

    def active(self) -> list[Ticket]: ...


class RoundsRepository(Protocol):
    """Rounds as their own rows, reachable only through the ticket that owns them."""

    def by_ticket(self, ticket_id: str) -> list[Round]: ...

    def save(self, ticket_id: str, rnd: Round) -> None: ...


class Repositories(Protocol):
    """A store the engine may write: two repositories and the transaction around them."""

    tickets: TicketsRepository
    rounds: RoundsRepository

    def atomic(self) -> AbstractContextManager[None]: ...
