"""The store's four promises: one transaction per command, callbacks only after
a commit, generated activity sets that cannot disagree with the row, and one
file per distinct artifact.

The transaction tests are the ones that matter for §4.2: a command must either
fully happen or not at all, and no `after_commit` callback may outlive a
rollback. The generated-column tests pin the rule a pass relies on — it may only
see tickets that are open and actually want work.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from pah.app.store import (
    SCHEMA_VERSION,
    Decision,
    SchemaError,
    Store,
    Thread,
    content_address,
)
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

MANDATE = Mandate(
    strength="explicit",
    basis={"via": "human_request"},
    scope_parts=("p1", "p3"),
    scope_operations=("resize_interface",),
    frozen_parts=("p0",),
    budget=Budget(rounds=5, geom_calls=200, sim_calls=40),
    policy_version="pv-test",
)

AT = "2026-01-01T00:00:00+00:00"


@pytest.fixture
def store(tmp_path):
    store = Store(tmp_path / "db.sqlite")
    yield store
    store.close()


def make_ticket(**overrides) -> Ticket:
    ticket = Ticket(id="t1", lineage="asm:1", mandate=MANDATE, created_at="c0")
    for name, value in overrides.items():
        setattr(ticket, name, value)
    return ticket


def make_round(index: int, verdict=None) -> Round:
    plan = RepairPlan(id=f"p{index}", operator="resize_interface", targets=("p1",), params={})
    return Round(id=f"r{index}", index=index, plan=plan, verdict=verdict, cost=Cost())


def rich_ticket() -> Ticket:
    """A ticket that fills every field, so no column can silently drop one."""
    rnd = Round(
        id="r0",
        index=0,
        plan=RepairPlan(
            id="p0",
            operator="resize_interface",
            targets=("p1", "p3"),
            params={"dr": 0.3},
            cost_est=Cost(geom_calls=2, wall_s=1.5),
            proposed_by="rule",
            reason="overlap",
        ),
        stage="judged",
        verdict="fail",
        evidence=[
            FailureCertificate(
                code="overlap",
                family="assembly",
                locus={"parts": ["p0", "p1"]},
                magnitude={"mm": 0.8},
                observed_by="v_overlap",
                confidence=0.9,
                hypotheses=({"why": "datum"},),
                evidence_ref="ab12cd",
            )
        ],
        cost=Cost(geom_calls=2, sim_calls=1, wall_s=1.5),
        artifacts={"glb": "ab12cd"},
        edit_cost=0.25,
        preserve_ratio=0.75,
    )
    assignment = Assignment(
        id="a1",
        state="running",
        created_at="c1",
        started_at="s1",
        report=Report(state="progress", summary="ok", at="r1"),
        idle_at="i1",
        delivered_at="d1",
    )
    return Ticket(
        id="t1",
        lineage="asm:1",
        mandate=MANDATE,
        rounds=[rnd],
        assignments=[assignment],
        inputs=[{"kind": "human", "text": "hi"}],
        undelivered=["i2"],
        attention=Attention(id="q1", question="which part may I move?", at="q"),
        delivery_failed=False,
        signoff_required=True,
        created_at="c0",
        updated_at="u0",
    )


# -- transactions ----------------------------------------------------------


def test_an_outer_rollback_discards_nested_work(store):
    with pytest.raises(RuntimeError):
        with store.atomic():
            store.kv.set("outer", 1)
            with store.atomic():
                store.kv.set("inner", 2)
            raise RuntimeError("boom")
    assert store.kv.get("outer") is None
    assert store.kv.get("inner") is None


def test_an_inner_rollback_keeps_the_outer_work(store):
    with store.atomic():
        store.kv.set("kept", 1)
        with pytest.raises(RuntimeError):
            with store.atomic():
                store.kv.set("dropped", 2)
                raise RuntimeError("boom")
    assert store.kv.get("kept") == 1
    assert store.kv.get("dropped") is None


def test_after_commit_waits_for_the_outermost_commit(store):
    seen = []
    with store.atomic():
        store.kv.set("k", 1)
        with store.atomic():
            store.after_commit(lambda: seen.append(store.kv.get("k")))
        assert seen == []  # the inner block released a savepoint, not the transaction
    assert seen == [1]


def test_after_commit_is_discarded_with_the_rollback(store):
    seen = []
    with pytest.raises(RuntimeError):
        with store.atomic():
            store.after_commit(lambda: seen.append("ran"))
            raise RuntimeError("boom")
    assert seen == []


def test_after_commit_outside_a_transaction_runs_now(store):
    seen = []
    store.after_commit(lambda: seen.append("ran"))
    assert seen == ["ran"]


# -- derived activity sets -------------------------------------------------


def test_an_open_ticket_is_not_active_until_it_wants_work(store):
    store.tickets.save(make_ticket())
    assert [t.id for t in store.tickets.open()] == ["t1"]
    assert store.tickets.active() == []

    ticket = store.tickets.get("t1")
    ticket.assignments.append(Assignment(id="a1", state="pending"))
    store.tickets.save(ticket)
    assert [t.id for t in store.tickets.active()] == ["t1"]


def test_undelivered_inputs_make_a_ticket_active(store):
    store.tickets.save(make_ticket(undelivered=["i1"]))
    assert [t.id for t in store.tickets.active()] == ["t1"]


def test_an_ended_ticket_leaves_both_sets(store):
    ended = TicketEnd(at="e", by="human", outcome="stopped")
    store.tickets.save(make_ticket(undelivered=["i1"], ended=ended))
    assert store.tickets.open() == []
    assert store.tickets.active() == []


def test_the_generated_columns_are_the_row_not_a_copy(store):
    store.tickets.save(make_ticket(undelivered=["i1"]))
    row = store.db.execute("SELECT open, active FROM ticket WHERE id = 't1'").fetchone()
    assert (row["open"], row["active"]) == (1, 1)


def test_a_saved_ticket_reads_back_unchanged(store):
    ticket = rich_ticket()
    store.tickets.save(ticket)
    assert store.tickets.get("t1") == ticket


# -- rounds, threads, decisions -------------------------------------------


def test_rounds_come_back_in_index_order(store):
    store.rounds.save("t1", make_round(2))
    store.rounds.save("t1", make_round(0))
    assert [r.index for r in store.rounds.by_ticket("t1")] == [0, 2]


def test_a_saved_round_reads_back_unchanged(store):
    rnd = rich_ticket().rounds[0]
    store.rounds.save("t1", rnd)
    assert store.rounds.by_ticket("t1") == [rnd]


def test_threads_and_decisions_round_trip(store):
    thread = Thread(key="asm:1", assembly_id="v3", pending=[{"kind": "human"}], meta={"note": "x"})
    store.threads.save(thread)
    assert store.threads.get("asm:1") == thread

    decision = Decision(id="d1", lineage="asm:1", at=AT, kind="triaged", reason="static", audit={"a": 1})
    store.decisions.save(decision)
    assert store.decisions.get("d1") == decision


# -- kv --------------------------------------------------------------------


def test_kv_reads_writes_and_deletes(store):
    assert store.kv.get("missing") is None
    assert store.kv.get("missing", default=7) == 7
    store.kv.set("run", {"policy": "rule", "budgets": [5, 200]})
    assert store.kv.get("run") == {"policy": "rule", "budgets": [5, 200]}
    store.kv.delete("run")
    assert store.kv.get("run") is None
    store.kv.delete("run")  # deleting an absent key is not an error


# -- artifacts -------------------------------------------------------------


def test_the_same_bytes_are_written_once(store, tmp_path, monkeypatch):
    root = tmp_path / "artifacts"
    writes = []
    real_write = Path.write_bytes

    def counted(self, data):
        writes.append(self)
        return real_write(self, data)

    monkeypatch.setattr(Path, "write_bytes", counted)

    first = store.artifacts.put(b"glb-bytes", "glb", root, at=AT)
    second = store.artifacts.put(b"glb-bytes", "glb", root, at=AT)

    assert first == second == content_address(b"glb-bytes")
    assert writes == [root / first[:2] / f"{first}.glb"]
    assert sorted(p.name for p in (root / first[:2]).iterdir()) == [f"{first}.glb"]
    registered = store.db.execute(
        "SELECT format, bytes FROM artifact WHERE hash = ?", (first,)
    ).fetchone()
    assert (registered["format"], registered["bytes"]) == ("glb", len(b"glb-bytes"))
    assert store.artifacts.get(first) == b"glb-bytes"
    assert store.artifacts.get("0" * 64) is None


def test_different_bytes_get_different_artifacts(store, tmp_path):
    root = tmp_path / "artifacts"
    first = store.artifacts.put(b"a", "ply", root, at=AT)
    second = store.artifacts.put(b"b", "ply", root, at=AT)
    assert first != second
    assert {store.artifacts.get(first), store.artifacts.get(second)} == {b"a", b"b"}


# -- schema ----------------------------------------------------------------


def test_the_schema_version_is_recorded(store):
    assert store.kv.get("schema.version") == SCHEMA_VERSION


def test_a_foreign_schema_version_is_refused(tmp_path):
    path = tmp_path / "db.sqlite"
    Store(path).close()
    db = sqlite3.connect(path)
    db.execute("UPDATE kv SET value = ? WHERE key = ?", (json.dumps(SCHEMA_VERSION + 1), "schema.version"))
    db.commit()
    db.close()
    with pytest.raises(SchemaError):
        Store(path)


def test_the_database_is_wal_with_a_full_fsync(store):
    assert store.db.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    assert store.db.execute("PRAGMA synchronous").fetchone()[0] == 2  # FULL


def test_reopening_keeps_the_tickets(tmp_path):
    path = tmp_path / "db.sqlite"
    first = Store(path)
    first.tickets.save(make_ticket())
    first.close()
    with Store(path) as again:
        assert again.tickets.get("t1") is not None
        assert again.kv.get("schema.version") == SCHEMA_VERSION
