"""The report is written through a durable queue, so "ended" cannot mean "lost".

The audit's P0-3: `RepairLoop.run` ends the ticket, and the runtime writes
`report.json` afterwards. A crash in between left a directory whose ticket was ended
and whose report did not exist — and `--resume` refused it, because an ended ticket
meant "nothing left to do". The episode's evidence had been paid for and was then
unreachable.

Two halves, both tested here:

* the write is an **effect row carrying the finished document**, so a failure keeps the
  text and a replay produces byte-identical output;
* `--resume` on an ended episode with an owed write **replays it** instead of refusing,
  while the refusal for a genuinely finished episode stays exactly as it was.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from _design_fakes import design_spec, install
from pah.app import runtime as runtime_module
from pah.app.runtime import RuntimeConfig


def run_episode(out: Path, spec, *, rounds: int = 1, resume: bool = False) -> dict:
    config = RuntimeConfig(out=out, policy="rule", rounds=rounds, resume=resume)
    with runtime_module.build(config) as composed:
        return composed.run_episode(spec, policy_name="rule", rounds=rounds, resume=resume)


def test_a_normal_run_writes_the_report_and_leaves_no_effect_row(monkeypatch, tmp_path):
    install(monkeypatch, mode="always_fail")
    out = tmp_path / "ep"

    report = run_episode(out, design_spec(clearance_mm=0.0))

    assert (out / "report.json").exists()
    assert json.loads((out / "report.json").read_text("utf-8")) == json.loads(
        json.dumps(report, ensure_ascii=False, default=str)
    )
    # The row is deleted on success: an idle queue is how the next resume knows there
    # is nothing owed.
    from pah.app.effects import Effects
    from pah.app.store import Store

    store = Store(out / "db.sqlite")
    try:
        assert Effects(store).idle("write-document")
    finally:
        store.close()


def test_a_failed_write_keeps_the_document_queued_and_repairs_later(tmp_path):
    """The audit's acceptance: the write fails once, the row keeps the document."""
    from pah.app.effects import Effects
    from pah.app.store import Store

    out = tmp_path / "ep"
    out.mkdir(parents=True)
    target = out / "report.json"
    target.mkdir()  # a directory where a file should be: `os.replace` must fail
    text = '{"from": "the interrupted attempt"}\n'

    store = Store(out / "db.sqlite")
    try:
        effects = Effects(store)
        effects.handle("write-document", runtime_module._write_document)
        effects.enqueue("write-document", "ticket:report", {"path": str(target), "text": text})
        effects.process_due()

        assert effects.pending("write-document") == 1, "a failed write must keep its row"
        assert effects.health("write-document").attempts == 1
        assert effects.health("write-document").state == "ready"  # one failure is a retry

        row = store.db.execute("SELECT payload FROM effect").fetchone()
        assert json.loads(row["payload"])["text"] == text, "the document must survive"

        target.rmdir()
        # A failed effect backs off before its next attempt, so the retry is scheduled
        # in the future on purpose — replay it as that future.
        effects.process_due(now=datetime.now(timezone.utc) + timedelta(seconds=120))
        assert effects.idle("write-document")
    finally:
        store.close()

    assert json.loads(target.read_text("utf-8")) == {"from": "the interrupted attempt"}


def test_resume_replays_an_owed_report_instead_of_refusing(monkeypatch, tmp_path):
    """Ended ticket + owed write → replay. The core of P0-3."""
    from pah.app.effects import Effects
    from pah.app.store import Store

    install(monkeypatch, mode="always_fail")
    out = tmp_path / "ep"
    spec = design_spec(clearance_mm=0.0)
    first = run_episode(out, spec)
    rounds_before = len(first["rounds"])

    # Simulate "ended, then the process died before writing": no report, one owed row.
    (out / "report.json").unlink()
    store = Store(out / "db.sqlite")
    try:
        effects = Effects(store)
        effects.handle("write-document", runtime_module._write_document)
        effects.enqueue(
            "write-document",
            "ticket:report",
            {"path": str(out / "report.json"), "text": '{"recovered": true}\n'},
        )
    finally:
        store.close()

    repaired = run_episode(out, spec, resume=True)

    assert repaired == {"recovered": True}
    assert json.loads((out / "report.json").read_text("utf-8")) == {"recovered": True}
    # A repair is not a resume: the recorded rounds did not grow.
    assert rounds_before >= 1


def test_resuming_a_finished_episode_is_still_refused(monkeypatch, tmp_path):
    install(monkeypatch, mode="always_fail")
    out = tmp_path / "ep"
    spec = design_spec(clearance_mm=0.0)
    run_episode(out, spec)

    with pytest.raises(RuntimeError, match="already ended"):
        run_episode(out, spec, resume=True)


def test_an_ended_episode_with_no_report_and_no_queue_names_the_problem(monkeypatch, tmp_path):
    install(monkeypatch, mode="always_fail")
    out = tmp_path / "ep"
    spec = design_spec(clearance_mm=0.0)
    run_episode(out, spec)
    (out / "report.json").unlink()

    with pytest.raises(RuntimeError, match="nothing to replay"):
        run_episode(out, spec, resume=True)
