"""The effect queue's five rules, one test each: a repeat keeps its place but not
its payload, `Later` is not a failure, a failure backs off, a success settles,
and an attempt whose payload was replaced settles nothing.

Every test drives the clock itself — `process_due(now=...)` is the only way time
enters the queue — so a failure here names a rule, never a slow machine.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from pah.app.effects import (
    BASE_DELAY_S,
    MAX_DELAY_S,
    Effects,
    Later,
)
from pah.app.store import Store

T0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def at(hour: int, minute: int, second: int) -> str:
    """The exact text the queue stores for that instant, so a test can pin it."""
    moment = datetime(2026, 1, 1, hour, minute, second, tzinfo=timezone.utc)
    return moment.strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")


@pytest.fixture
def store(tmp_path):
    store = Store(tmp_path / "db.sqlite")
    yield store
    store.close()


@pytest.fixture
def effects(store):
    return Effects(store, clock=lambda: T0)


def row(store: Store, kind: str, key: str):
    return store.db.execute(
        "SELECT * FROM effect WHERE kind = ? AND key = ?", (kind, key)
    ).fetchone()


def due(store: Store, kind: str, key: str) -> datetime:
    return datetime.fromisoformat(row(store, kind, key)["due"])


# -- coalescing ------------------------------------------------------------


def test_a_repeat_keeps_its_place_and_raises_the_generation(effects, store):
    effects.enqueue("reply", "m1", {"n": 1})
    first = row(store, "reply", "m1")
    effects.enqueue("reply", "m1", {"n": 2})
    second = row(store, "reply", "m1")

    assert second["seq"] == first["seq"]
    assert second["generation"] == first["generation"] + 1
    assert json.loads(second["payload"]) == {"n": 2}
    assert effects.pending("reply") == 1


def test_the_queue_order_is_the_order_of_first_asking(effects, store):
    effects.enqueue("reply", "m2", {})
    effects.enqueue("reply", "m1", {})
    effects.enqueue("reply", "m2", {})  # a repeat must not jump the queue
    ran = []
    effects.handle("reply", lambda payload, key: ran.append(key), limit=2)
    effects.process_due(now=T0)
    assert ran == ["m2", "m1"]


# -- Later ------------------------------------------------------------------


def test_later_postpones_without_charging_an_attempt(effects, store):
    def not_now(payload, key):
        raise Later("all worker slots are busy")

    effects.handle("launch", not_now)
    effects.enqueue("launch", "a1", {})
    assert effects.process_due(now=T0) == 1

    waited = row(store, "launch", "a1")
    assert (waited["attempts"], waited["error"]) == (0, "")
    assert waited["due"] == at(12, 0, 2)
    assert waited["generation"] == 0


# -- failure ---------------------------------------------------------------


def test_failures_double_the_wait_and_keep_the_error(effects, store):
    def boom(payload, key):
        raise RuntimeError("no CUDA context")

    effects.handle("launch", boom)
    effects.enqueue("launch", "a1", {})

    assert effects.process_due(now=T0) == 1
    assert row(store, "launch", "a1")["attempts"] == 1
    assert row(store, "launch", "a1")["due"] == at(12, 0, 2)

    assert effects.process_due(now=T0 + timedelta(seconds=2)) == 1
    assert row(store, "launch", "a1")["attempts"] == 2
    assert row(store, "launch", "a1")["due"] == at(12, 0, 6)

    assert effects.process_due(now=T0 + timedelta(seconds=6)) == 1
    assert row(store, "launch", "a1")["attempts"] == 3
    assert row(store, "launch", "a1")["due"] == at(12, 0, 14)
    assert "no CUDA context" in row(store, "launch", "a1")["error"]


def test_the_wait_stops_doubling_at_the_ceiling(effects, store):
    def boom(payload, key):
        raise RuntimeError("x")

    effects.handle("launch", boom)
    effects.enqueue("launch", "a1", {})

    waits = []
    for _ in range(8):
        before = due(store, "launch", "a1")
        effects.process_due(now=before)
        waits.append((due(store, "launch", "a1") - before).total_seconds())

    assert waits[:3] == [BASE_DELAY_S, 2 * BASE_DELAY_S, 4 * BASE_DELAY_S]
    assert max(waits) == MAX_DELAY_S


# -- settling --------------------------------------------------------------


def test_a_success_settles_the_row(effects, store):
    done = []
    effects.handle("reply", lambda payload, key: done.append((payload, key)))
    effects.enqueue("reply", "m1", {"text": "hi"})

    assert effects.process_due(now=T0) == 1
    assert done == [({"text": "hi"}, "m1")]
    assert effects.idle("reply")


def test_a_replaced_payload_survives_the_attempt_it_interrupted(effects, store):
    def replace(payload, key):
        effects.enqueue("reply", key, {"n": payload["n"] + 1})

    effects.handle("reply", replace)
    effects.enqueue("reply", "m1", {"n": 1})
    assert effects.process_due(now=T0) == 1

    kept = row(store, "reply", "m1")
    assert kept is not None, "the old attempt deleted the newer payload"
    assert json.loads(kept["payload"]) == {"n": 2}
    assert kept["generation"] == 1


def test_a_replaced_payload_is_not_charged_the_older_failure(effects, store):
    def replace_then_fail(payload, key):
        effects.enqueue("reply", key, {"n": 2})
        raise RuntimeError("boom")

    effects.handle("reply", replace_then_fail)
    effects.enqueue("reply", "m1", {"n": 1})
    effects.process_due(now=T0)

    kept = row(store, "reply", "m1")
    assert (kept["attempts"], kept["error"]) == (0, "")
    assert kept["generation"] == 1


# -- health ----------------------------------------------------------------


def test_three_failures_report_error_and_a_success_clears_it(effects, store):
    broken = True

    def run(payload, key):
        if broken:
            raise RuntimeError("the simulator crashed")

    effects.handle("launch", run)
    effects.enqueue("launch", "a1", {})
    assert effects.health("launch").state == "ready"

    for _ in range(3):
        effects.process_due(now=due(store, "launch", "a1"))

    health = effects.health("launch")
    assert health.state == "error"
    assert health.attempts == 3
    assert "the simulator crashed" in health.detail

    broken = False
    effects.process_due(now=due(store, "launch", "a1"))
    assert effects.health("launch").state == "ready"


# -- queue state -----------------------------------------------------------


def test_pending_and_idle_track_the_queue(effects, store):
    assert effects.idle()
    assert effects.pending() == 0

    effects.enqueue("reply", "m1", {})
    effects.enqueue("launch", "a1", {})
    assert effects.pending() == 2
    assert effects.pending("reply") == 1
    assert not effects.idle()

    # Not yet due: the queue reports work, but nothing runs and nothing is lost.
    assert effects.process_due(now=T0 - timedelta(seconds=1)) == 0
    assert effects.pending() == 2


def test_a_pass_runs_at_most_the_handler_limit(effects, store):
    ran = []
    effects.handle("reply", lambda payload, key: ran.append(key), limit=2)
    for key in ("m1", "m2", "m3"):
        effects.enqueue("reply", key, {})

    assert effects.process_due(now=T0) == 2
    assert ran == ["m1", "m2"]
    assert effects.pending("reply") == 1


def test_a_zero_limit_holds_the_kind_without_dropping_rows(effects, store):
    ran = []
    effects.handle("launch", lambda payload, key: ran.append(key), limit=0)
    effects.enqueue("launch", "a1", {})

    assert effects.process_due(now=T0) == 0
    assert ran == []
    assert effects.pending("launch") == 1


def test_rows_of_an_unhandled_kind_are_left_alone(effects, store):
    effects.enqueue("reply", "m1", {})
    effects.enqueue("launch", "a1", {})
    effects.handle("reply", lambda payload, key: None)

    assert effects.process_due(now=T0) == 1
    assert effects.pending("launch") == 1


def test_drain_keeps_the_rows_for_the_next_run(effects, store):
    ran = []
    effects.handle("reply", lambda payload, key: ran.append(key))
    effects.enqueue("reply", "m1", {})

    effects.drain()
    assert effects.process_due(now=T0) == 0
    assert ran == []
    assert effects.pending("reply") == 1


def test_a_delayed_row_waits_for_its_due_time(effects, store):
    effects.handle("reply", lambda payload, key: None)
    effects.enqueue("reply", "m1", {}, delay=5)

    assert effects.process_due(now=T0) == 0
    assert effects.pending("reply") == 1
    assert effects.process_due(now=T0 + timedelta(seconds=5)) == 1


def test_a_queued_effect_lives_in_the_callers_transaction(store):
    """The row is written inside the command, so a rolled-back command queues nothing."""
    effects = Effects(store, clock=lambda: T0)
    with pytest.raises(RuntimeError):
        with store.atomic():
            effects.enqueue("reply", "m1", {})
            raise RuntimeError("boom")
    assert effects.pending() == 0

    with store.atomic():
        effects.enqueue("reply", "m1", {})
    assert effects.pending("reply") == 1


def test_queued_rows_survive_a_restart(tmp_path):
    path = tmp_path / "db.sqlite"
    first = Store(path)
    Effects(first, clock=lambda: T0).enqueue("reply", "m1", {"text": "hi"})
    first.close()

    with Store(path) as again:
        effects = Effects(again, clock=lambda: T0)
        done = []
        effects.handle("reply", lambda payload, key: done.append(payload))
        assert effects.process_due(now=T0) == 1
        assert done == [{"text": "hi"}]
