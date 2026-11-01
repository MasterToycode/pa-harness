"""The trace's promises: one parseable JSON object per line, a sequence that
survives a restart, and a `geom_hash` replay that two runs must agree on.

The crash tests are the point of the module. A recorder that is only correct on a
clean exit records the runs nobody needs to debug; the interesting trace is the one
whose process was killed mid-round, and it has to stay readable, keep its row
numbers unique, and keep reporting the line that got torn.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from pah.app.trace import KINDS, EpisodeRecorder, Trace, TraceCorrupt
from pah.core.model import Cost, FailureCertificate

FIXED = datetime(2026, 11, 1, 12, 0, 0, tzinfo=timezone.utc)
AT = "2026-11-01T12:00:00.000000+00:00"


@pytest.fixture()
def path(tmp_path):
    return tmp_path / "runs" / "ep-1" / "trace.jsonl"


@pytest.fixture()
def trace(path):
    with Trace(path, clock=lambda: FIXED) as writing:
        yield writing


def verify_certificate(
    observed_by: str = "connector_feasible@1", code: str = "gap_too_wide"
) -> FailureCertificate:
    return FailureCertificate(
        code=code,
        family="geometry",
        locus={"parts": ["p1", "p6"], "joint": "j6"},
        magnitude={"value": 0.9, "unit": "mm", "threshold": None, "direction": "gt"},
        observed_by=observed_by,
        hypotheses=({"text": "wall"},),
    )


# --------------------------------------------------------------------------- #
# Sequence and clock
# --------------------------------------------------------------------------- #
def test_seq_starts_at_one_and_increases(trace):
    for index in range(3):
        trace.emit("round.stage", round=index, stage="materialized")

    assert [row["seq"] for row in trace.rows()] == [1, 2, 3]


def test_at_comes_from_the_injected_clock(trace):
    row = trace.emit("round.begin", ticket="t1", round=0)

    assert row["at"] == AT
    assert all(other["at"] == AT for other in trace.rows())


def test_the_clock_is_read_once_per_event(path):
    ticks = iter(FIXED + timedelta(seconds=offset) for offset in range(3))
    with Trace(path, clock=lambda: next(ticks)) as writing:
        rows = [
            writing.emit("round.stage", round=index, stage="materialized")
            for index in range(3)
        ]

    assert [row["at"] for row in rows] == [
        "2026-11-01T12:00:00.000000+00:00",
        "2026-11-01T12:00:01.000000+00:00",
        "2026-11-01T12:00:02.000000+00:00",
    ]


def test_seq_and_at_are_the_trace_s_own(trace):
    for field in ("seq", "at"):
        with pytest.raises(ValueError, match="assigned by the trace"):
            trace.emit("verify", **{field: 1})

    assert trace.emit("verify", round=0, certs=[])["seq"] == 1  # nothing was burned
    assert [row["seq"] for row in trace.rows()] == [1]


def test_unknown_event_kinds_are_refused(trace):
    with pytest.raises(ValueError, match="unknown trace event"):
        trace.emit("round.start", round=0)

    assert trace.rows() == []


def test_the_closed_set_is_the_nine_kinds_the_loop_emits():
    # `episode.begin` and `round.refused` joined the set when `policy/loop.py` was
    # written; the seven-kind version of this test predated it and was the only
    # red test in the suite. The trace refuses any kind outside this set, so the
    # set and the loop's emissions must stay in step.
    assert KINDS == {
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


# --------------------------------------------------------------------------- #
# One line, one parse
# --------------------------------------------------------------------------- #
def test_every_line_parses_on_its_own(trace, path):
    trace.emit(
        "round.begin",
        ticket="t1",
        round=0,
        plan={"operator": "resize_connector", "targets": ["p1", "p6"]},
        policy="rule",
    )
    trace.emit("round.stage", round=0, stage="materialized", geom_hash="h1")
    trace.emit("round.close", round=0, verdict="fail", cost=Cost(geom_calls=2, wall_s=1.5))
    trace.emit("ticket.end", outcome="exhausted", reason="budget.rounds")

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    assert [row["seq"] for row in rows] == [1, 2, 3, 4]
    assert [row["kind"] for row in rows] == [
        "round.begin",
        "round.stage",
        "round.close",
        "ticket.end",
    ]
    assert rows[0]["plan"]["operator"] == "resize_connector"
    assert rows[2]["cost"] == {
        "geom_calls": 2,
        "sim_calls": 0,
        "wall_s": 1.5,
        "llm_calls": 0,
        "tokens": 0,
    }


def test_emit_returns_the_row_as_written(trace):
    written = trace.emit("ticket.end", outcome="repaired", reason="")

    assert written == trace.rows()[-1]
    assert written["seq"] == 1 and written["kind"] == "ticket.end"


# --------------------------------------------------------------------------- #
# Corruption
# --------------------------------------------------------------------------- #
def test_a_corrupt_line_raises_unless_skipped(path):
    with Trace(path, clock=lambda: FIXED) as writer:
        writer.emit("round.begin", round=0)
        writer.emit("round.close", round=0, verdict="fail")
    with path.open("a", encoding="utf-8") as handle:
        handle.write("not json\n")
        handle.write('{"seq": 4, "kind": "ticket.end"}\n')

    with Trace(path, clock=lambda: FIXED) as reader:
        with pytest.raises(TraceCorrupt, match="line 3"):
            reader.rows()
        kept = reader.rows(skip_corrupt=True)

    assert [row["seq"] for row in kept] == [1, 2, 4]


def test_a_line_that_is_not_an_object_is_corrupt(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("42\n", encoding="utf-8")

    with Trace(path, clock=lambda: FIXED) as reader:
        with pytest.raises(TraceCorrupt, match="not a JSON object"):
            reader.rows()
        assert reader.rows(skip_corrupt=True) == []


# --------------------------------------------------------------------------- #
# Restart
# --------------------------------------------------------------------------- #
def test_reopening_continues_the_sequence(path):
    with Trace(path, clock=lambda: FIXED) as first:
        for index in range(3):
            first.emit("round.stage", round=index, stage="materialized")
    prefix = path.read_bytes()

    with Trace(path, clock=lambda: FIXED) as second:
        second.emit("ticket.end", outcome="exhausted")
        second.emit("ticket.end", outcome="exhausted")

    data = path.read_bytes()
    rows = [json.loads(line) for line in data.decode("utf-8").splitlines()]

    assert data.startswith(prefix)  # append-only: nothing already written moved
    assert [row["seq"] for row in rows] == [1, 2, 3, 4, 5]  # no repeats after restart


def test_a_torn_last_line_does_not_swallow_the_next_event(path):
    with Trace(path, clock=lambda: FIXED) as first:
        first.emit("round.begin", round=0)
    path.write_bytes(path.read_bytes() + b'{"seq": 2, "kind": "rou')  # killed mid-write

    with Trace(path, clock=lambda: FIXED) as second:
        written = second.emit("ticket.end", outcome="exhausted")

        assert written["seq"] == 2  # counted from the last readable row
        with pytest.raises(TraceCorrupt, match="line 2"):
            second.rows()
        assert len(second.rows(skip_corrupt=True)) == 2

    assert b'"rou' in path.read_bytes()  # the torn row is kept, not deleted


# --------------------------------------------------------------------------- #
# Replay
# --------------------------------------------------------------------------- #
def test_replay_hashes_extracts_geom_hashes_in_order(trace, path):
    trace.emit("round.stage", round=0, stage="materialized", geom_hash="h1")
    trace.emit("round.stage", round=1, stage="materialized", geom_hash="h2")
    trace.emit("round.close", round=1, verdict="fail")  # no geometry to hash
    trace.emit("round.stage", round=2, stage="materialized", geom_hash="h1")  # loop
    trace.emit("round.stage", round=3, stage="materialized", geom_hash=None)

    assert Trace.replay_hashes(path) == ["h1", "h2", "h1"]


def test_replay_hashes_refuses_a_corrupt_trace(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"seq": 1, "geom_hash": "h1"}\n{"seq": \n', encoding="utf-8")

    with pytest.raises(TraceCorrupt):
        Trace.replay_hashes(path)


# --------------------------------------------------------------------------- #
# EpisodeRecorder
# --------------------------------------------------------------------------- #
def test_episode_recorder_structures_the_certificates(trace):
    written = EpisodeRecorder(trace).verify(
        round=0,
        certs=[verify_certificate("v2"), verify_certificate("v1", "peg_detached")],
    )

    assert written["kind"] == "verify" and written["round"] == 0
    assert [cert["code"] for cert in written["certs"]] == ["gap_too_wide", "peg_detached"]
    assert written["certs"][0]["locus"] == {"parts": ["p1", "p6"], "joint": "j6"}
    assert written["certs"][0]["magnitude"]["value"] == 0.9
    assert written["certs"][0]["hypotheses"] == [{"text": "wall"}]  # tuple -> list on disk
    assert written["verifier_ids"] == ["v1", "v2"]  # derived, sorted, de-duplicated
    assert written == trace.rows()[-1]


def test_episode_recorder_keeps_explicit_verifier_ids(trace):
    written = EpisodeRecorder(trace).verify(
        round=1, certs=[{"code": "gap_too_wide", "observed_by": "v3"}], verifier_ids=["v9"]
    )

    assert written["verifier_ids"] == ["v9"]
    assert written["certs"] == [{"code": "gap_too_wide", "observed_by": "v3"}]


def test_episode_recorder_refuses_something_that_is_not_a_certificate(trace):
    with pytest.raises(TypeError, match="certificate"):
        EpisodeRecorder(trace).verify(round=0, certs=[42])  # type: ignore[list-item]
