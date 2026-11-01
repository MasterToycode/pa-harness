"""Behaviour of the case-memory plugin: remember, rank, replay, delegate, bookkeep.

The claim this file is evidence for is the user's: *a new policy plugin makes the
harness better at repairing a failure it has already seen, without touching the
kernel*. So the tests are at two levels:

* the **ledger and the policy contract** — fast, no simulator, no materialiser;
* one **end-to-end episode pair** through the real `RepairLoop`, where episode 2
  closes the same failure in one round instead of two. That is the number a reader
  should believe, because the loop and the engine are the real ones; only the
  verifier and the operator registry are doubles.

Nothing here is trained, embedded or sampled: the only state is a SQLite table.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from pah.core.model import Budget, Cost, FailureCertificate, RepairPlan, Round, Ticket, Mandate
from pah.design.api import assembly_from_spec, spec_of
from pah.design.spec import DesignSpec, FitConstraints, InterfaceSpec, PartSpec
from pah.plugins import API_VERSION, KINDS, PluginHost
from pah.policy.case_memory import (
    CaseMemory,
    CaseMemoryPolicy,
    build_case_memory_policy,
    register,
    signature_of,
)
from pah.policy.loop import RepairLoop

# --------------------------------------------------------------------------- #
# Design-plane doubles
# --------------------------------------------------------------------------- #
SPEC = DesignSpec(
    spec_id="case-memory-demo",
    requirement="a peg that is too tight; the fix is the right clearance",
    parts=(
        PartSpec(id="body", role="body", size_mm=(60.0, 40.0, 12.0)),
        PartSpec(id="peg", role="insert", size_mm=(10.0, 10.0, 14.0)),
    ),
    interfaces=(
        InterfaceSpec(
            id="iface_0",
            kind="peg_hole",
            a="body",
            b="peg",
            params={
                "d_peg_mm": 10.0,
                "d_hole_mm": 10.10,
                "engage_mm": 8.0,
                "peg_len_mm": 14.0,
            },
        ),
    ),
    constraints=FitConstraints(min_clearance_mm=0.10, max_clearance_mm=0.60),
)


def assembly_for(spec: DesignSpec = SPEC, *, geometry: str = "sha256:seed"):
    return assembly_from_spec(
        spec=spec,
        geometry_paths={
            "body": "runs/demo/geometry/body.stl",
            "peg": "runs/demo/geometry/peg.stl",
        },
        geometry_hashes={"body": "sha256:body", "peg": geometry},
    )


def certificate(code: str = "insufficient_clearance", family: str = "geometry"):
    return FailureCertificate(
        code=code,
        family=family,  # type: ignore[arg-type]
        locus={"spec": SPEC.spec_id, "parts": ["body", "peg"], "interface": "iface_0"},
        magnitude={"value": 0.05, "unit": "mm", "threshold": 0.1, "direction": "lt"},
        observed_by="fit_clearance",
    )


class FakeInner:
    """A stand-in for `design_policies.RulePolicy`: a queue of plans, then `None`."""

    name = "rule"

    def __init__(self, *plans: RepairPlan | None) -> None:
        self.plans = list(plans)
        self.calls = 0

    def propose(self, assembly, certs, history, budget_left):
        self.calls += 1
        if not self.plans:
            return None
        return self.plans.pop(0)


def inner_plan(clearance_mm: float = 0.30, operator: str = "set_clearance") -> RepairPlan:
    return RepairPlan(
        id=f"inner-{operator}-{clearance_mm}",
        operator=operator,
        targets=("body", "peg"),
        params={
            "interface_id": "iface_0",
            "parts": ("body", "peg"),
            "clearance_mm": clearance_mm,
        },
        cost_est=Cost(geom_calls=1),
        proposed_by="rule",
    )


@pytest.fixture()
def memory() -> CaseMemory:
    ledger = CaseMemory(":memory:")
    yield ledger
    ledger.close()


def closed(plan: RepairPlan, verdict: str, *, index: int = 0) -> Round:
    return Round(id=f"r{index}", index=index, plan=plan, verdict=verdict)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# The ledger
# --------------------------------------------------------------------------- #
def test_signature_is_the_sorted_physical_codes() -> None:
    codes = signature_of(
        [
            certificate("insufficient_clearance"),
            certificate("insertion_path_blocked", family="geometry"),
            # An environment entry is a tool that could not run, not a failure class.
            certificate("physics_error", family="environment"),
        ]
    )
    assert codes == "insertion_path_blocked+insufficient_clearance"  # sorted


def test_pass_and_fail_both_land_in_the_ledger(memory: CaseMemory) -> None:
    """The ledger is append-only: a success is not overwritten by the next failure."""
    for outcome in ("pass", "fail", "error"):
        memory.record(
            content_key=SPEC.content_key(),
            kind="peg_hole",
            cert_codes="insufficient_clearance",
            operator="set_clearance",
            params={"clearance_mm": 0.3},
            targets=("body", "peg"),
            outcome=outcome,
            rounds_used=1,
        )
    rows = memory.rows()
    assert [row["outcome"] for row in rows] == ["pass", "fail", "error"]
    assert memory.stats() == {
        "total": 3,
        "by_outcome": {"pass": 1, "fail": 1, "error": 1},
        "by_class": {"peg_hole|insufficient_clearance": 3},
    }


def test_ledger_survives_on_disk(tmp_path: Path) -> None:
    """'Uses it more, gets better' needs the file to outlive the process."""
    path = tmp_path / "runs" / "case_memory.sqlite"
    first = CaseMemory(path)
    first.record(
        content_key=SPEC.content_key(),
        kind="peg_hole",
        cert_codes="insufficient_clearance",
        operator="set_clearance",
        params={"clearance_mm": 0.3},
        outcome="pass",
        rounds_used=1,
    )
    first.close()

    second = CaseMemory(path)
    candidates = second.candidates(kind="peg_hole", cert_codes="insufficient_clearance")
    assert [record.operator for record in candidates] == ["set_clearance"]
    second.close()


def test_ranking_prefers_the_rate_then_the_evidence(memory: CaseMemory) -> None:
    """2/2 outranks 1/1? No: rate first, then the count — and both beat 1/3."""
    for operator, outcome, times in (
        ("set_clearance", "pass", 1),
        ("resize_feature", "pass", 2),
        ("resize_insert", "fail", 2),
    ):
        for _ in range(times):
            memory.record(
                content_key=SPEC.content_key(),
                kind="peg_hole",
                cert_codes="insufficient_clearance",
                operator=operator,
                params={"clearance_mm": 0.3},
                outcome=outcome,
                rounds_used=1,
            )
    memory.record(
        content_key=SPEC.content_key(),
        kind="peg_hole",
        cert_codes="insufficient_clearance",
        operator="resize_insert",
        params={"clearance_mm": 0.3},
        outcome="pass",
        rounds_used=1,
    )
    ordered = memory.candidates(kind="peg_hole", cert_codes="insufficient_clearance")
    assert [record.operator for record in ordered] == [
        "resize_feature",  # 2/2
        "set_clearance",  # 1/1
        "resize_insert",  # 1/3
    ]
    assert ordered[0].rate == pytest.approx(1.0)
    # A never-successful action is not a candidate at all.
    assert all(record.passes > 0 for record in ordered)


def test_missing_class_returns_nothing(memory: CaseMemory) -> None:
    assert memory.candidates(kind="dovetail", cert_codes="insufficient_clearance") == []


# --------------------------------------------------------------------------- #
# The policy
# --------------------------------------------------------------------------- #
def test_first_episode_delegates_and_second_replays(memory: CaseMemory) -> None:
    """The task's headline case: the fix is remembered and reused the second time."""
    inner = FakeInner(inner_plan())
    episode_one = CaseMemoryPolicy(inner=inner, memory=memory)
    plan = episode_one.propose(assembly_for(), [certificate()], [], Budget())
    assert episode_one.last_source == "inner"
    assert plan is not None and plan.proposed_by == "rule"
    assert episode_one.flush([closed(plan, "pass")]) == 1  # what the caller does after the loop

    episode_two = CaseMemoryPolicy(inner=inner, memory=memory)
    replayed = episode_two.propose(assembly_for(geometry="sha256:other"), [certificate()], [], Budget())
    assert episode_two.last_source == "case_memory"
    assert replayed is not None
    assert replayed.proposed_by == "case"
    assert replayed.operator == "set_clearance"
    assert replayed.params["clearance_mm"] == pytest.approx(0.30)
    # The address came from the current design, not from the ledger.
    assert replayed.params["interface_id"] == "iface_0"
    assert tuple(replayed.targets) == ("body", "peg")
    assert "0/1 failed" not in replayed.reason and "succeeded 1/1" in replayed.reason
    assert inner.calls == 1  # the second episode never had to ask
    assert episode_two.last_record is not None and episode_two.last_record.passes == 1


def test_unseen_failure_delegates(memory: CaseMemory) -> None:
    """A different recurrence class is not assumed to share a fix."""
    inner = FakeInner(inner_plan(0.30, operator="resize_feature"))
    policy = CaseMemoryPolicy(inner=inner, memory=memory)
    policy.propose(assembly_for(), [certificate("insertion_path_blocked")], [], Budget())
    assert policy.last_source == "inner"
    assert inner.calls == 1


def test_clearance_outside_the_current_band_is_refused(memory: CaseMemory) -> None:
    """The same knob in a design with a tighter band must not be replayed."""
    memory.record(
        content_key=SPEC.content_key(),
        kind="peg_hole",
        cert_codes="insufficient_clearance",
        operator="set_clearance",
        params={"interface_id": "iface_0", "parts": ("body", "peg"), "clearance_mm": 0.45},
        outcome="pass",
        rounds_used=1,
    )
    tight = SPEC.edit(constraints=FitConstraints(min_clearance_mm=0.05, max_clearance_mm=0.20))
    inner = FakeInner(inner_plan(0.10))
    policy = CaseMemoryPolicy(inner=inner, memory=memory)
    plan = policy.propose(assembly_for(tight), [certificate()], [], Budget())
    assert policy.last_source == "inner"
    assert plan is not None and plan.proposed_by == "rule"
    assert policy.last_record is None


def test_unbounded_knob_is_refused(memory: CaseMemory) -> None:
    """A parameter this policy cannot bound is never handed to the operator."""
    memory.record(
        content_key=SPEC.content_key(),
        kind="peg_hole",
        cert_codes="insufficient_clearance",
        operator="set_clearance",
        params={"clearance_mm": 0.3, "mystery_mm": 7.0},
        outcome="pass",
        rounds_used=1,
    )
    inner = FakeInner(inner_plan(0.3))
    policy = CaseMemoryPolicy(inner=inner, memory=memory)
    policy.propose(assembly_for(), [certificate()], [], Budget())
    assert policy.last_source == "inner"


def test_part_addressed_operators_are_never_replayed(memory: CaseMemory) -> None:
    """`shift_part` names one part of another design; the memory stores no role."""
    memory.record(
        content_key=SPEC.content_key(),
        kind="peg_hole",
        cert_codes="geometric_interference",
        operator="shift_part",
        params={"part_id": "peg", "delta_mm": [0.0, 0.5, 0.0]},
        outcome="pass",
        rounds_used=1,
    )
    inner = FakeInner(inner_plan(0.3, operator="resize_insert"))
    policy = CaseMemoryPolicy(inner=inner, memory=memory)
    policy.propose(assembly_for(), [certificate("geometric_interference")], [], Budget())
    assert policy.last_source == "inner"


def test_already_tried_action_is_not_resubmitted(memory: CaseMemory) -> None:
    """Replaying the plan that just failed would burn a round and hit the cycle guard."""
    memory.record(
        content_key=SPEC.content_key(),
        kind="peg_hole",
        cert_codes="insufficient_clearance",
        operator="set_clearance",
        params={"interface_id": "iface_0", "parts": ("body", "peg"), "clearance_mm": 0.3},
        outcome="pass",
        rounds_used=1,
    )
    previous = RepairPlan(
        id="previous",
        operator="set_clearance",
        targets=("body", "peg"),
        params={"interface_id": "iface_0", "parts": ("body", "peg"), "clearance_mm": 0.3},
        proposed_by="case",
    )
    inner = FakeInner(inner_plan(0.5))
    policy = CaseMemoryPolicy(inner=inner, memory=memory)
    plan = policy.propose(
        assembly_for(), [certificate()], [closed(previous, "fail")], Budget()
    )
    assert policy.last_source == "inner"
    assert plan is not None and plan.params["clearance_mm"] == pytest.approx(0.50)


def test_error_rounds_are_stored_but_do_not_count(memory: CaseMemory) -> None:
    """An operator that raised is not evidence about the action."""
    inner = FakeInner(inner_plan(0.3), inner_plan(0.4))
    policy = CaseMemoryPolicy(inner=inner, memory=memory)
    first = policy.propose(assembly_for(), [certificate()], [], Budget())
    assert first is not None
    assert policy.flush([closed(first, "error")]) == 1
    rows = memory.rows()
    assert [row["outcome"] for row in rows] == ["error"]
    # `decided` excludes it, so it is not even a candidate.
    assert memory.candidates(kind="peg_hole", cert_codes="insufficient_clearance") == []


def test_flush_is_idempotent(memory: CaseMemory) -> None:
    """A resumed episode re-reads its own history; it must not double-count."""
    policy = CaseMemoryPolicy(inner=FakeInner(inner_plan(0.3)), memory=memory)
    plan = policy.propose(assembly_for(), [certificate()], [], Budget())
    assert plan is not None
    history = [closed(plan, "pass")]
    assert policy.flush(history) == 1
    assert policy.flush(history) == 0
    assert len(memory.rows()) == 1


def test_open_round_keeps_its_provenance(memory: CaseMemory) -> None:
    """A round with no verdict yet is not a record; flushing again later still works."""
    policy = CaseMemoryPolicy(inner=FakeInner(inner_plan(0.3)), memory=memory)
    plan = policy.propose(assembly_for(), [certificate()], [], Budget())
    assert plan is not None
    open_round = Round(id="r0", index=0, plan=plan)  # verdict is None
    assert policy.flush([open_round]) == 0
    open_round.verdict = "pass"
    assert policy.flush([open_round]) == 1


def test_no_ledger_entry_means_no_action_without_an_inner(memory: CaseMemory) -> None:
    """A policy with nothing to delegate to says `no action`, not `success`."""
    policy = CaseMemoryPolicy(inner=None, memory=memory)
    assert policy.propose(assembly_for(), [certificate()], [], Budget()) is None
    assert policy.last_source == "none"


def test_non_design_plane_assembly_delegates(memory: CaseMemory) -> None:
    """The M1 (case.json) plane has no spec, so there is no kind or band to key on."""
    from pah.core.model import Assembly

    legacy = Assembly(
        id="asm_legacy",
        version=1,
        parts=({"id": "p0"},),
        joints=(),
        connectors=(),
        constraints={},
    )
    inner = FakeInner(inner_plan(0.3))
    policy = CaseMemoryPolicy(inner=inner, memory=memory)
    assert policy.propose(legacy, [certificate()], [], Budget()) is not None
    assert policy.last_source == "inner"
    assert memory.rows() == []


# --------------------------------------------------------------------------- #
# The extension point: a plugin, not a kernel change
# --------------------------------------------------------------------------- #
def test_policy_satisfies_the_loop_protocol(memory: CaseMemory) -> None:
    """`RepairLoop` accepts this object because it *is* a `LoopPolicy`.

    `loop.LoopPolicy` is a plain `Protocol` (not `runtime_checkable`), so the check here
    is the duck-typed one the loop actually performs: a `name` and a four-argument
    `propose`. The end-to-end test below runs the real loop, which is the real proof.
    """
    policy = CaseMemoryPolicy(inner=None, memory=memory)
    assert policy.name == "case_memory"
    assert callable(policy.propose)
    assert list(inspect.signature(policy.propose).parameters) == [
        "assembly",
        "certs",
        "history",
        "budget_left",
    ]


def test_plugin_declares_the_contract_and_registers_one_policy(tmp_path: Path) -> None:
    """`pah.plugins` loads a module by three names; this asserts all three."""
    import pah.policy.case_memory as module

    assert module.api_version == API_VERSION
    assert module.kind in KINDS
    assert callable(module.register)

    host = PluginHost()
    before_verifiers = host.verifiers.ids()
    before_operators = host.operators.ids() if hasattr(host.operators, "ids") else ()
    register(host)
    assert "case_memory" in host.policies
    # Nothing outside `host.policies` was touched: no verifier, no operator, no connector.
    assert host.verifiers.ids() == before_verifiers
    if hasattr(host.operators, "ids"):
        assert host.operators.ids() == before_operators
    assert host.connectors == {}

    factory = host.policies["case_memory"]
    policy = factory(memory_path=tmp_path / "runs" / "case_memory.sqlite")
    assert isinstance(policy, CaseMemoryPolicy)
    assert policy.name == "case_memory"
    assert policy.inner is not None  # the rule policy, built against the host's operators
    assert Path(policy.memory.path).name == "case_memory.sqlite"


def test_plugin_loader_lists_it_as_a_builtin(tmp_path: Path) -> None:
    """The extension point, end to end: `pah.plugins.load` finds and mounts it.

    `plugins_builtin.BUILTIN_MODULES` names this module (T4 owns that file); loading a
    fresh host must produce a *policy* row with no error and a factory in
    `host.policies`. That — not a docstring — is what "a new capability without touching
    `core/`" means in this build.
    """
    from pah.plugins import load

    host = PluginHost()
    report = load(host)
    rows = [record for record in report.records if record.name == "pah.policy.case_memory"]
    assert len(rows) == 1
    row = rows[0]
    assert row.ok, row.error
    assert row.kind == "policy"
    assert row.source == "builtin"
    assert "policy=case_memory" in row.declarations
    assert "case_memory" in host.policies
    assert report.failed == ()


def test_factory_accepts_a_runtime_that_passes_extra_keywords(tmp_path: Path) -> None:
    """A runtime may pass `root`/`seed`/`episode`; the factory ignores what it does
    not need instead of failing at the composition root."""
    policy = build_case_memory_policy(
        memory_path=tmp_path / "mem.sqlite", root=tmp_path, seed=7, episode="e0"
    )
    assert isinstance(policy, CaseMemoryPolicy)
    assert policy.inner is None  # explicit: no inner was supplied


# --------------------------------------------------------------------------- #
# End to end through the real loop: 2 rounds the first time, 1 the second
# --------------------------------------------------------------------------- #
class _Engine:
    """The loop's engine contract, minimal: budget, round bookkeeping, close."""

    def __init__(self) -> None:
        self.rounds: list[Round] = []
        self.ended: list[tuple[str, str]] = []

    def begin_round(self, ticket_id: str, plan: RepairPlan) -> Round:
        round_ = Round(id=f"r{len(self.rounds)}", index=len(self.rounds), plan=plan)
        self.rounds.append(round_)
        return round_

    def stage_round(self, round_id: str, stage: str) -> None:
        return None

    def close_round(self, round_id, *, verdict, evidence, cost, artifacts, **_):
        for round_ in self.rounds:
            if round_.id == round_id:
                round_.verdict = verdict
                round_.evidence = list(evidence)
                round_.cost = cost
                round_.artifacts = dict(artifacts)
                return round_
        raise AssertionError(f"unknown round {round_id}")

    def end_ticket(self, ticket_id: str, outcome: str, reason: str = "") -> Ticket:
        self.ended.append((outcome, reason))
        return _ticket()


class _Verifiers:
    """Certificate batches, one per `check` call; the last batch repeats."""

    def __init__(self, *batches: list[FailureCertificate]) -> None:
        self.batches = list(batches) or [[]]
        self.calls = 0

    def check(self, assembly):
        batch = self.batches[min(self.calls, len(self.batches) - 1)]
        self.calls += 1
        return list(batch)


class _Operators:
    """Applies a plan by minting a new assembly version whose geometry hash moves."""

    def applicable(self, assembly, certs):
        return ["set_clearance"]

    def apply(self, assembly, plan):
        spec = spec_of(assembly)
        token = json.dumps(
            {"operator": plan.operator, "params": plan.params}, sort_keys=True, default=str
        )
        return assembly_from_spec(
            spec=spec,
            geometry_paths={
                "body": "runs/demo/geometry/body.stl",
                "peg": "runs/demo/geometry/peg.stl",
            },
            geometry_hashes={"body": "sha256:body", "peg": f"sha256:{token}"},
            parent=assembly.id,
            version=assembly.version + 1,
        )


def _ticket(rounds: int = 5) -> Ticket:
    return Ticket(
        id="t1",
        lineage="demo",
        mandate=Mandate(
            strength="explicit",
            basis={"via": "test"},
            scope_parts=("body", "peg"),
            scope_operations=("set_clearance",),
            frozen_parts=(),
            budget=Budget(rounds=rounds, geom_calls=100, sim_calls=10),
            policy_version="pv-case-memory",
        ),
    )


def _run_episode(policy, verifiers) -> tuple:
    engine = _Engine()
    loop = RepairLoop(
        engine=engine,
        verifiers=verifiers,
        registry=_Operators(),
        policy=policy,
    )
    result = loop.run(_ticket(), assembly_for())
    # `RepairLoop` returns on `pass` without asking again, so the winning round is only
    # in `engine.rounds`. A runtime credits it with this one line; the test is the caller.
    policy.flush(engine.rounds)
    return result, engine


def test_second_episode_closes_in_one_round(memory: CaseMemory) -> None:
    """The evidence: a failure met twice costs 2 rounds then 1, with the same kernel.

    Episode 1: the inner policy's first clearance (0.05) does not satisfy the fit, the
    second (0.30) does — two rounds, and the ledger learns `set_clearance -> 0.30`.
    Episode 2: the same failure is proposed *from memory* on round one.
    """
    inner = FakeInner(inner_plan(0.05), inner_plan(0.30))
    episode_one_policy = CaseMemoryPolicy(inner=inner, memory=memory)
    # The loop reuses a round's *after* verification as the next *before* check, so the
    # sequence is: fail, fail, pass — two rounds and three checks.
    result_one, engine_one = _run_episode(
        episode_one_policy, _Verifiers([certificate()], [certificate()], [])
    )
    assert result_one.verdict == "repaired"
    assert len(engine_one.rounds) == 2
    assert episode_one_policy.last_source in {"inner", "case_memory"}

    episode_two_policy = CaseMemoryPolicy(inner=FakeInner(inner_plan(0.05)), memory=memory)
    result_two, engine_two = _run_episode(
        episode_two_policy, _Verifiers([certificate()], [])
    )
    assert result_two.verdict == "repaired"
    assert len(engine_two.rounds) == 1
    assert engine_two.rounds[0].plan.operator == "set_clearance"
    assert engine_two.rounds[0].plan.params["clearance_mm"] == pytest.approx(0.30)
    assert engine_two.rounds[0].plan.proposed_by == "case"
    assert episode_two_policy.last_source == "case_memory"
    # The ledger holds all three attempts: the loser, the winner, and the replay of the
    # winner in episode 2 — which is the whole mechanism in one row list.
    rows = memory.rows()
    assert [row["outcome"] for row in rows] == ["fail", "pass", "pass"]
    assert [json.loads(row["params"])["clearance_mm"] for row in rows] == [0.05, 0.30, 0.30]
    assert {row["cert_codes"] for row in rows} == {"insufficient_clearance"}
    assert {row["kind"] for row in rows} == {"peg_hole"}
    assert memory.stats()["by_outcome"] == {"pass": 2, "fail": 1}
