"""The append-only episode trace: `runs/<run>/trace.jsonl`, one event per line.

`docs/internal/repair-loop-spec.md` §2.4 fixes the event kinds and the two
promises this module is judged on: **every line parses on its own**, and
`replay_hashes` yields the same sequence for the same case, policy and seed.

Choices that decide what happens after a crash
----------------------------------------------
* `seq` is assigned here, never passed in. It restarts from the highest sequence
  number already on disk, so a process that reopens the file continues the count
  instead of repeating it: 1-based and strictly increasing.
* Every line is flushed before `emit` returns. A killed process loses the line it
  was writing; it cannot lose the lines it had written.
* A torn last line — the one case a kill during `write` can produce — is closed off
  with a newline when the file is opened for appending, so the next event gets its
  own line. The torn line stays in the file and is reported as corrupt rather than
  silently deleted: a trace that quietly drops rows is worse than one that says
  "line 7 is unreadable", because `replay_hashes` is an acceptance check.
* `rows()` raises `TraceCorrupt` on anything it cannot read;
  `rows(skip_corrupt=True)` returns the rest. There is deliberately no middle mode
  that returns rows *and* a count: a caller either needs the whole file or has
  decided to carry on without it, and a count that is easy to ignore invites the
  second decision under the appearance of the first.

`geom_hash` is not interpreted here. `replay_hashes` is the one accessor that knows
its name, because §5 compares that sequence between two runs and nothing else in
the loop should care how geometry is hashed.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pah.core.model import FailureCertificate

__all__ = ["EpisodeRecorder", "KINDS", "Trace", "TraceCorrupt"]

#: The closed set of event kinds in spec §2.4. An unlisted kind is refused at
#: `emit` rather than written: a typo that becomes a new event type is a schema
#: change nobody reviewed, and the trace is the audit record the loop is judged on.
KINDS = frozenset(
    {
        "episode.begin",
        "round.begin",
        "round.stage",
        "round.refused",
        "verify",
        "round.close",
        "ticket.end",
        "operator.error",
        "verifier.error",
    }
)


class TraceCorrupt(RuntimeError):
    """A trace line that is not a JSON object. Raised by `rows()` unless skipped."""


def _stamp(at: datetime) -> str:
    """Fixed-width UTC text, so text order is chronological order.

    The same format `app/effects.py` uses, and the same rule: a naive datetime is
    read as UTC, so the same argument names the same instant on every machine.
    """
    if at.tzinfo is None:
        at = at.replace(tzinfo=timezone.utc)
    return at.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")


def _jsonable(value: Any) -> Any:
    """Serialize the domain records a caller will hand to `emit`.

    `plan` is a `RepairPlan`, `cost` a `Cost`, `certs` a list of
    `FailureCertificate` — the recorder serializes them so the loop never has to
    pre-chew its own audit trail, and so failing to serialize cannot be the reason a
    round is lost.
    """
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=str)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"{type(value).__name__} is not traceable")


def _read(path: Path, *, skip_corrupt: bool) -> list[dict]:
    """Every row in `path`, in file order. Blank lines are ignored, not corrupt."""
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return []
    rows: list[dict] = []
    for number, raw in enumerate(data.split(b"\n"), start=1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            if skip_corrupt:
                continue
            excerpt = raw[:120].decode("utf-8", "replace")
            raise TraceCorrupt(
                f"{path}: line {number} is not readable JSON ({error}): {excerpt!r}"
            ) from error
        if not isinstance(row, dict):
            if skip_corrupt:
                continue
            excerpt = raw[:120].decode("utf-8", "replace")
            raise TraceCorrupt(
                f"{path}: line {number} is not a JSON object: {excerpt!r}"
            )
        rows.append(row)
    return rows


def _max_seq(path: Path) -> int:
    """The highest `seq` already on disk, so `emit` continues the count."""
    highest = 0
    for row in _read(path, skip_corrupt=True):
        try:
            highest = max(highest, int(row.get("seq") or 0))
        except (TypeError, ValueError):
            continue
    return highest


class Trace:
    """Writes one append-only JSONL file; `seq` and `at` are the trace's, not yours."""

    def __init__(
        self, path: str | Path, *, clock: Callable[[], datetime] | None = None
    ) -> None:
        """Opens `path` for appending, continuing `seq` from what is already there.

        `clock` is the only source of "now", so a test can pin `at` and a replay can
        compare two runs without a tolerance.
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._close_torn_tail()
        self._seq = _max_seq(self.path)
        self._file = self.path.open("a", encoding="utf-8")

    def _close_torn_tail(self) -> None:
        """Separates a line a killed process left unterminated from the next one.

        Nothing is rewritten: one newline is appended, and the torn line keeps its
        place in the file as a corrupt row.
        """
        try:
            data = self.path.read_bytes()
        except FileNotFoundError:
            return
        if data and not data.endswith(b"\n"):
            with self.path.open("ab") as handle:
                handle.write(b"\n")

    # -- writing -----------------------------------------------------------

    def emit(self, kind: str, **fields: Any) -> dict:
        """Appends one event and flushes it; returns the row as it was written.

        The returned row is the JSON round-trip of what went to disk, so a caller
        comparing it with `rows()` sees identical types (tuples are already lists in
        the file).
        """
        if kind not in KINDS:
            raise ValueError(
                f"unknown trace event {kind!r}; the closed set is {sorted(KINDS)}"
            )
        reserved = sorted({"seq", "at"} & set(fields))
        if reserved:
            raise ValueError(f"{reserved!r} are assigned by the trace, not the caller")
        self._seq += 1
        row = {"seq": self._seq, "at": _stamp(self._clock()), "kind": kind, **fields}
        line = json.dumps(row, ensure_ascii=False, default=_jsonable)
        self._file.write(line + "\n")
        self._file.flush()
        return json.loads(line)

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> Trace:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- reading -----------------------------------------------------------

    def rows(self, *, skip_corrupt: bool = False) -> list[dict]:
        """Every row already on disk, including the ones this process just wrote.

        A corrupt line raises `TraceCorrupt` with its number and an excerpt; with
        `skip_corrupt=True` it is dropped and the rest are returned.
        """
        if not self._file.closed:
            self._file.flush()
        return _read(self.path, skip_corrupt=skip_corrupt)

    @staticmethod
    def replay_hashes(path: str | Path) -> list[str]:
        """The `geom_hash` of every row that has one, in file order.

        This is the acceptance accessor: two runs of the same case with the same
        policy must produce equal lists (§5). A trace with a corrupt line is not
        replayable, so the default `rows()` behaviour — raise — is the one used here.
        """
        hashes: list[str] = []
        for row in _read(Path(path), skip_corrupt=False):
            value = row.get("geom_hash")
            if isinstance(value, str) and value:
                hashes.append(value)
        return hashes


class EpisodeRecorder:
    """The one convenience on top of `Trace`: a `verify` row with structured certs.

    A certificate contains tuples and nested dicts; `str()`-ing it into a log line
    is how evidence stops being evidence. `verify` writes each certificate as a row
    of its own fields, and derives `verifier_ids` from `observed_by` when the caller
    does not pass them — a caller cannot accidentally claim a verifier that never
    produced the evidence it is logging.
    """

    def __init__(self, trace: Trace) -> None:
        self._trace = trace

    def verify(
        self,
        *,
        round: int,
        certs: Iterable[FailureCertificate | Mapping[str, Any]],
        verifier_ids: Iterable[str] | None = None,
    ) -> dict:
        """Appends one `verify` event and returns the row as written."""
        rows = [_certificate(cert) for cert in certs]
        if verifier_ids is None:
            observed = sorted(
                {str(row["observed_by"]) for row in rows if row.get("observed_by")}
            )
        else:
            observed = [str(verifier_id) for verifier_id in verifier_ids]
        return self._trace.emit(
            "verify", round=round, certs=rows, verifier_ids=observed
        )


def _certificate(cert: FailureCertificate | Mapping[str, Any]) -> dict:
    """One certificate as a plain row: a dataclass is expanded, a mapping is copied."""
    if is_dataclass(cert) and not isinstance(cert, type):
        return asdict(cert)
    if isinstance(cert, Mapping):
        return dict(cert)
    raise TypeError(
        "a certificate must be a FailureCertificate or a mapping, not "
        f"{type(cert).__name__}"
    )
