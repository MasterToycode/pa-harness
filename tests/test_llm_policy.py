"""Tests for the model-backed policy (`policy/llm.py`).

The reproducibility boundary for a pipeline that calls a model is the
**recording**, not the model (`app/providers.ReplayLLM`), so every test here runs
from a JSONL file and needs no key and no network. What is locked:

* a well-formed reply becomes a plan the guarded registry accepts and the design
  really changes;
* the four ways a reply can be wrong — not JSON, an operator outside the legal
  list, parameters the operator's model refuses, targets that disagree — each
  return `None` with a non-empty `last_error`, and **nothing retries, falls back or
  guesses**;
* an abstention (`{"done": true}`) is *not* an error;
* the legal action list the model sees is the registry's, narrowed by what the
  connector kind can express.
"""

from __future__ import annotations

import json

import pytest

from pah.app.providers import ReplayLLM
from pah.core.model import Budget, FailureCertificate, RepairPlan, Round
from pah.design import spec_of
from pah.policy.design_ops import register_builtin
from pah.policy.design_policies import build
from pah.policy.llm import LlmPolicy, OUTPUT_SCHEMA
from pah.policy.loop import RepairLoop
from pah.policy.registry import OperatorRegistry
from test_design_ops import base_assembly, cert, make_spec  # noqa: E402
from test_design_policies import DesignChecks, FakeEngine, ticket  # noqa: E402

PARTS = ["body", "insert"]


# --------------------------------------------------------------------------- #
# Recording helpers — a jsonl of replies is the whole fake model
# --------------------------------------------------------------------------- #
def record(tmp_path, *replies, name: str = "recording.jsonl"):
    """Write one JSONL row per reply; a `dict` becomes text, a `str` is verbatim."""
    rows = []
    for reply in replies:
        text = reply if isinstance(reply, str) else json.dumps(reply)
        rows.append(json.dumps({"text": text, "model": "test-model", "usage": {"prompt_tokens": 120, "completion_tokens": 40}}))
    path = tmp_path / name
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def registry() -> OperatorRegistry:
    registry = OperatorRegistry()
    register_builtin(registry)
    return registry


def policy_for(tmp_path, *replies, ops=None, name="recording.jsonl") -> LlmPolicy:
    ops = ops or registry()
    return LlmPolicy(ReplayLLM(record(tmp_path, *replies, name=name)), registry=ops)


def propose(policy, assembly, *certs, history=()) -> RepairPlan | None:
    return policy.propose(
        assembly,
        list(certs) or [cert("insufficient_clearance", clearance_mm=0.0)],
        list(history),
        Budget(rounds=3, geom_calls=100, sim_calls=20),
    )


WANT_CLEARANCE = {
    "operator": "set_clearance",
    "params": {
        "interface_id": "iface_0",
        "parts": PARTS,
        "clearance_mm": 0.35,
    },
    "reason": "insufficient_clearance: move to the middle of the [0.10, 0.60] band",
}


# --------------------------------------------------------------------------- #
# The happy path: a reply becomes a legal plan
# --------------------------------------------------------------------------- #
def test_a_recorded_reply_becomes_a_plan_the_registry_accepts(tmp_path):
    ops = registry()
    assembly = base_assembly()
    policy = policy_for(tmp_path, WANT_CLEARANCE, ops=ops)

    plan = propose(policy, assembly, cert("insufficient_clearance"))

    assert plan is not None
    assert policy.last_error == ""
    assert plan.operator == "set_clearance"
    assert plan.targets == ("body", "insert")
    assert plan.proposed_by == "llm"
    assert plan.cost_est.llm_calls == 1
    assert plan.cost_est.tokens == 160
    repaired = ops.apply(assembly, plan)
    assert spec_of(repaired).constraints.in_band(spec_of(repaired).clearance_mm)
    assert spec_of(repaired).clearance_mm == pytest.approx(0.35)


def test_the_observation_carries_the_numbers_the_schemas_and_no_geometry(tmp_path):
    policy = policy_for(tmp_path, WANT_CLEARANCE)
    assembly = base_assembly()

    propose(policy, assembly, cert("insufficient_clearance", clearance_mm=0.0))

    request = policy.client.calls[0]["request"]
    seen = json.loads(request["user"])
    assert seen["design"]["kind"] == "peg_hole"
    assert seen["design"]["clearance_mm"] == 0.0
    assert seen["design"]["acceptance"]["min_clearance_mm"] == 0.10
    assert seen["design"]["interface_params"]["d_peg_mm"] == 10.0
    assert seen["failures"][0]["code"] == "insufficient_clearance"
    assert seen["budget_left"] == {"rounds": 3, "geom_calls": 100, "sim_calls": 20}
    assert request["json_schema"] == OUTPUT_SCHEMA
    assert request["temperature"] == 0.0
    # No path, no mesh, no vertex: the model is a planner, not a generator.
    blob = request["user"]
    assert '"geom"' not in blob and '"path"' not in blob and ".stl" not in blob
    assert "vertex" not in blob


def test_the_legal_action_list_is_the_registrys_narrowed_by_the_kind(tmp_path):
    ops = registry()
    policy = policy_for(tmp_path, {"done": True}, ops=ops)
    assembly = base_assembly(make_spec("dovetail"))

    propose(policy, assembly, cert("geometric_interference"))

    seen = json.loads(policy.client.calls[0]["request"]["user"])
    legal = seen["legal_actions"]
    assert "resize_insert" not in legal  # a dovetail has no insert-side number
    assert "shift_part" in legal
    assert set(legal) <= set(ops.applicable(assembly, [cert("geometric_interference")]))


def test_the_history_is_shown_so_a_failed_plan_is_not_resubmitted(tmp_path):
    policy = policy_for(tmp_path, {"done": True})
    tried = RepairPlan(
        id="p0",
        operator="set_clearance",
        targets=("body", "insert"),
        params={"interface_id": "iface_0", "parts": PARTS, "clearance_mm": 0.35},
    )
    history = [Round(id="r0", index=0, plan=tried, verdict="fail", evidence=[cert("insufficient_clearance")])]

    propose(policy, base_assembly(), history=history)

    seen = json.loads(policy.client.calls[0]["request"]["user"])
    assert seen["history"][0]["operator"] == "set_clearance"
    assert seen["history"][0]["verdict"] == "fail"
    assert seen["history"][0]["codes_after"] == ["insufficient_clearance"]


def test_a_targets_field_that_agrees_is_accepted(tmp_path):
    reply = {**WANT_CLEARANCE, "targets": PARTS}

    plan = propose(policy_for(tmp_path, reply), base_assembly())

    assert plan is not None and plan.targets == ("body", "insert")


# --------------------------------------------------------------------------- #
# The four rejections: each returns None, records why, and does not retry
# --------------------------------------------------------------------------- #
def test_an_unknown_operator_is_refused(tmp_path):
    reply = {**WANT_CLEARANCE, "operator": "move_vertex"}
    policy = policy_for(tmp_path, reply)

    plan = propose(policy, base_assembly())

    assert plan is None
    assert "move_vertex" in policy.last_error
    assert "legal actions" in policy.last_error
    assert len(policy.client.calls) == 1  # no retry, no fallback


def test_a_legal_operator_the_evidence_does_not_admit_is_refused(tmp_path):
    """`rotate_part` exists, but an `insufficient_clearance` round does not admit it."""
    reply = {
        "operator": "rotate_part",
        "params": {"part_id": "insert", "delta_deg": [15.0, 0.0, 0.0]},
    }
    policy = policy_for(tmp_path, reply)

    plan = propose(policy, base_assembly(), cert("insufficient_clearance"))

    assert plan is None
    assert "rotate_part" in policy.last_error


def test_parameters_the_operator_refuses_are_refused(tmp_path):
    """The model's payload goes through the operator's own pydantic model."""
    smuggled = {
        "operator": "set_clearance",
        "params": {**WANT_CLEARANCE["params"], "vertex": [0.0, 0.0, 0.0]},
    }
    wrong_type = {
        "operator": "set_clearance",
        "params": {**WANT_CLEARANCE["params"], "clearance_mm": "a hair"},
    }

    for reply in (smuggled, wrong_type):
        policy = policy_for(tmp_path, reply, name=f"{len(str(reply))}.jsonl")
        plan = propose(policy, base_assembly())
        assert plan is None
        assert "invalid" in policy.last_error


def test_targets_that_disagree_with_the_params_are_refused(tmp_path):
    reply = {**WANT_CLEARANCE, "targets": ["body"]}
    policy = policy_for(tmp_path, reply)

    plan = propose(policy, base_assembly())

    assert plan is None
    assert "targets" in policy.last_error


def test_a_reply_that_is_not_json_is_refused(tmp_path):
    policy = policy_for(tmp_path, "Sure! Here is the plan: use set_clearance.")

    plan = propose(policy, base_assembly())

    assert plan is None
    assert "not JSON" in policy.last_error


def test_a_json_array_is_not_a_plan(tmp_path):
    policy = policy_for(tmp_path, "[1, 2, 3]")

    assert propose(policy, base_assembly()) is None
    assert "not a JSON object" in policy.last_error


def test_a_parameter_set_the_design_cannot_hold_is_refused(tmp_path):
    """A legal payload can still describe an unbuildable part: a dovetail's per-side
    clearance cannot be negative, and `regenerate_interface` takes a free-form
    patch. The plan is checked against the snapshot before a round is spent."""
    ops = registry()
    assembly = base_assembly(make_spec("dovetail"))
    reply = {
        "operator": "regenerate_interface",
        "params": {
            "interface_id": "iface_0",
            "parts": PARTS,
            "params": {"clearance_mm": -0.2},
        },
    }
    policy = policy_for(tmp_path, reply, ops=ops)

    plan = propose(policy, assembly, cert("geometric_interference"))

    assert plan is None
    assert "does not build" in policy.last_error


def test_switching_to_the_kind_already_in_use_is_refused_by_the_model_layer(tmp_path):
    reply = {
        "operator": "switch_connector",
        "params": {"interface_id": "iface_0", "parts": PARTS, "kind": "peg_hole"},
    }
    policy = policy_for(tmp_path, reply)

    plan = propose(policy, base_assembly(), cert("insufficient_clearance"))

    assert plan is None
    assert "does not build" in policy.last_error


# --------------------------------------------------------------------------- #
# Abstention is an answer, not an error
# --------------------------------------------------------------------------- #
def test_an_abstention_is_not_an_error(tmp_path):
    policy = policy_for(tmp_path, {"done": True})

    plan = propose(policy, base_assembly())

    assert plan is None
    assert policy.last_error == ""


def test_a_null_operator_is_an_abstention_too(tmp_path):
    policy = policy_for(tmp_path, {"operator": None, "reason": "nothing will help"})

    assert propose(policy, base_assembly()) is None
    assert policy.last_error == ""


def test_no_evidence_means_no_call_to_the_model(tmp_path):
    """A round with nothing to repair must not spend a token."""
    policy = policy_for(tmp_path, WANT_CLEARANCE)

    assert policy.propose(base_assembly(), [], [], Budget()) is None
    assert policy.client.calls == []


def test_environment_evidence_alone_means_no_call_to_the_model(tmp_path):
    policy = policy_for(tmp_path, WANT_CLEARANCE)
    crashed = FailureCertificate(
        code="physics_error",
        family="environment",
        locus={"interface": "iface_0"},
        magnitude={},
        observed_by="physics_settle",
    )

    assert policy.propose(base_assembly(), [crashed], [], Budget()) is None
    assert policy.client.calls == []


def test_a_dead_provider_is_recorded_not_raised(tmp_path):
    policy = LlmPolicy(ReplayLLM(record(tmp_path, WANT_CLEARANCE)), registry=registry())
    propose(policy, base_assembly())  # consume the only recording
    policy.last_error = ""

    # The second call finds the recording exhausted: ReplayLLM raises ProviderError.
    assert propose(policy, base_assembly()) is None
    assert "provider error" in policy.last_error


# --------------------------------------------------------------------------- #
# The loop: one round, and the plan really repaired the design
# --------------------------------------------------------------------------- #
def run_loop(assembly, policy, ops, *, rounds: int = 3):
    engine = FakeEngine()
    loop = RepairLoop(
        engine=engine, verifiers=DesignChecks(), registry=ops, policy=policy
    )
    return loop.run(ticket(rounds), assembly), engine


def test_the_llm_policy_closes_the_loop_in_one_recorded_round(tmp_path):
    ops = registry()
    assembly = base_assembly()
    policy = policy_for(tmp_path, WANT_CLEARANCE, ops=ops)

    result, engine = run_loop(assembly, policy, ops)

    assert result.verdict == "repaired"
    assert result.rounds == 1
    assert engine.rounds[0].plan.operator == "set_clearance"
    assert engine.rounds[0].verdict == "pass"
    assert spec_of(result.assembly).constraints.in_band(spec_of(result.assembly).clearance_mm)


def test_a_refused_reply_stops_the_episode_rather_than_being_replaced(tmp_path):
    """No fallback: an unparseable reply is an `infeasible` episode, not a rule run."""
    ops = registry()
    policy = policy_for(tmp_path, "not json at all", ops=ops)

    result, engine = run_loop(base_assembly(), policy, ops)

    assert result.verdict == "infeasible"
    assert result.rounds == 0
    assert engine.rounds == []
    assert "not JSON" in policy.last_error


# --------------------------------------------------------------------------- #
# The factory T4 calls
# --------------------------------------------------------------------------- #
def test_build_wires_the_client_and_the_prompt(tmp_path):
    client = ReplayLLM(record(tmp_path, WANT_CLEARANCE))

    policy = build("llm", registry=registry(), llm=client)

    assert policy.name == "llm"
    assert policy.client is client
    assert "Repair policy" in policy.system_prompt
    assert propose(policy, base_assembly()) is not None


def test_build_takes_a_prompt_from_the_config_file(tmp_path):
    config = tmp_path / "pah.toml"
    prompt = tmp_path / "prompt.md"
    prompt.write_text("You are a different planner.", encoding="utf-8")
    config.write_text(
        f'[policy.llm]\nsystem_prompt_file = "{prompt.as_posix()}"\n', encoding="utf-8"
    )

    policy = build(
        "llm",
        registry=registry(),
        llm=ReplayLLM(record(tmp_path, {"done": True})),
        config_path=config,
    )

    assert policy.system_prompt == "You are a different planner."


def test_every_plan_the_policy_returns_is_bound_to_the_three_gates(tmp_path):
    """The same guarantee the rule policy has: a plan the model produced is either
    refused or accepted by the guarded registry, never silently mis-scoped."""
    ops = registry()
    assembly = base_assembly()
    cases = [
        ("insufficient_clearance", WANT_CLEARANCE),
        (
            "insufficient_clearance",
            {
                "operator": "resize_feature",
                "params": {"interface_id": "iface_0", "parts": PARTS, "delta_mm": 0.4},
            },
        ),
        (
            "insertion_path_blocked",
            {
                "operator": "shift_part",
                "params": {"part_id": "insert", "delta_mm": [0.0, 2.0, 0.0]},
            },
        ),
        (
            "insufficient_clearance",
            {
                "operator": "switch_connector",
                "params": {"interface_id": "iface_0", "parts": PARTS, "kind": "snap_fit"},
            },
        ),
        (
            "geometric_interference",
            {
                "operator": "resize_insert",
                "params": {"interface_id": "iface_0", "parts": PARTS, "delta_mm": -0.4},
            },
        ),
    ]
    for index, (code, reply) in enumerate(cases):
        policy = policy_for(tmp_path, reply, name=f"r{index}.jsonl", ops=ops)
        plan = propose(policy, assembly, cert(code))
        assert plan is not None, (reply, policy.last_error)
        ops.apply(assembly, plan)
