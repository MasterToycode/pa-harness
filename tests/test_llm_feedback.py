"""P3-2 feedback tests: what the model is told, and what the report admits about it.

Three claims are locked here, because each one is a way the "LLM feedback" line
could look implemented while being wrong:

1. **The numbers in `suggestions` are the rule's, not the model's.** The assertion
   is not "a hint is present" but "the hint equals the plan `RulePolicy` would
   propose" — both read the same ladder and the same band arithmetic, so a future
   edit that lets the model seed a suggestion fails this test.
2. **`history` says what an action *did*.** Every closed round carries the numbers
   the plan wrote, the failure codes before and after, and the verifier-measured
   clearance pair. "An action was tried" is not feedback.
3. **A dead provider is not a policy that found nothing to do.** The loop reports
   both as `infeasible`; the report has to separate them, or an equal-budget
   comparison charges the harness's breakage to the policy.

Everything runs offline: the client is a scripted double, and the compositions go
through `tests/_design_fakes.install`, so nothing here spends a token.
"""

from __future__ import annotations

import json

import pytest

from _design_fakes import design_spec, install  # noqa: E402  (pytest puts tests/ on sys.path)
from pah.app import runtime as runtime_module
from pah.app.providers import LLMResult, ProviderError, Usage
from pah.app.runtime import RuntimeConfig
from pah.core.model import Budget, FailureCertificate, Round
from pah.policy.design_ops import register_builtin
from pah.policy.design_policies import (
    DEFAULT_LADDER,
    RulePolicy,
    band_target,
    suggestions_for,
)
from pah.policy.llm import LlmPolicy
from pah.policy.registry import OperatorRegistry
from test_design_ops import base_assembly, make_spec  # noqa: E402
from test_design_policies import failed  # noqa: E402

BUDGET = Budget(rounds=3, geom_calls=100, sim_calls=20)


# --------------------------------------------------------------------------- #
# Doubles
# --------------------------------------------------------------------------- #
class ScriptedClient:
    """A provider double: scripted replies, counted calls, no network.

    A `dict` reply is serialised, a `str` is sent verbatim (so a malformed answer
    can be scripted), and an exception is raised (so a dead provider can be too).
    """

    provider = "scripted"

    def __init__(self, *replies: object) -> None:
        self._replies = list(replies) or [{"done": True}]
        self.calls = 0
        self.requests: list[dict] = []

    def complete(self, **kwargs: object) -> LLMResult:
        self.calls += 1
        self.requests.append(dict(kwargs))
        reply = self._replies[min(self.calls - 1, len(self._replies) - 1)]
        if isinstance(reply, BaseException):
            raise reply
        text = reply if isinstance(reply, str) else json.dumps(reply)
        return LLMResult(
            text=text, model="scripted", provider=self.provider, usage=Usage()
        )

    def observation(self, index: int = 0) -> dict:
        return json.loads(self.requests[index]["user"])


def registry() -> OperatorRegistry:
    ops = OperatorRegistry()
    register_builtin(ops)
    return ops


def cert(code: str, clearance_mm: float = 0.0) -> FailureCertificate:
    return FailureCertificate(
        code=code,
        family="geometry",
        locus={"interface": "iface_0"},
        magnitude={"clearance_mm": clearance_mm},
        observed_by="fake_fit",
    )


def assembly_band(assembly):
    """The acceptance band of the spec an assembly carries."""
    from pah.design.api import spec_of

    return spec_of(assembly).constraints


# --------------------------------------------------------------------------- #
# 1. suggestions: the rule's numbers, for the model to copy or argue with
# --------------------------------------------------------------------------- #
def test_the_observation_carries_the_rules_own_suggestion_numbers():
    """`params_hint` equals the plan the deterministic baseline would send.

    The same ladder and the same `band_target` arithmetic produce both, so this is
    the assertion that keeps the feedback honest: the model is handed a computed
    number, not an example number.
    """
    ops = registry()
    assembly = base_assembly()
    client = ScriptedClient({"done": True})
    policy = LlmPolicy(client, registry=ops)

    plan = policy.propose(assembly, [cert("insufficient_clearance")], [], BUDGET)

    assert plan is None  # the scripted model abstained
    seen = client.observation()
    suggested = seen["suggestions"]["insufficient_clearance"]
    hint = suggested["params_hint"]["set_clearance"]
    rule_plan = RulePolicy(registry=ops).propose(
        assembly, [cert("insufficient_clearance")], [], BUDGET
    )
    assert rule_plan is not None
    assert hint["clearance_mm"] == rule_plan.params["clearance_mm"]
    assert hint["clearance_mm"] == band_target(assembly_band(assembly), "mid")
    assert suggested["operators"][0] == "set_clearance"
    assert "band" in suggested["why"]


def test_a_suggestion_is_sendable_through_the_operators_own_model():
    """A hint the registry would refuse is worse than no hint: it invites a rejected
    call. Every suggested payload validates as the operator's own parameters."""
    ops = registry()
    spec = make_spec()
    suggestions = suggestions_for(["insufficient_clearance", "geometric_interference"], spec)

    for code, row in suggestions.items():
        assert row["operators"] == list(row["params_hint"])
        for operator_id, params in row["params_hint"].items():
            operator = ops.get(operator_id)
            validated = operator.params.model_validate(params)
            assert operator.targets(validated.model_dump()), (code, operator_id)
            if "parts" in validated.model_dump():
                assert set(validated.model_dump()["parts"]) == {"body", "insert"}
            assert code in DEFAULT_LADDER


def test_suggestions_respect_what_the_kind_can_express():
    """A dovetail has no insert-side number, so no hint offers one — and the
    observation only suggests operators the round actually admits."""
    dovetail = make_spec("dovetail")
    suggestions = suggestions_for(["geometric_interference"], dovetail)
    hinted = suggestions["geometric_interference"]["params_hint"]

    # The interference ladder starts with `resize_insert`, which a dovetail cannot
    # express, so the row falls through to the next specific rung.
    assert "resize_insert" not in hinted
    assert "shift_part" in hinted

    ops = registry()
    assembly = base_assembly(dovetail)
    client = ScriptedClient({"done": True})
    LlmPolicy(client, registry=ops).propose(
        assembly, [cert("geometric_interference")], [], BUDGET
    )
    seen = client.observation()
    assert set(seen["suggestions"]["geometric_interference"]["params_hint"]) <= set(
        seen["legal_actions"]
    )


def test_a_suggestion_already_tried_is_not_offered_again():
    """A hint that repeats a failed attempt invites the one action the history
    already ruled out. When every hint for a code is spent, the code gets no entry —
    which is the honest answer: the baseline has nothing new for it."""
    ops = registry()
    assembly = base_assembly()
    policy = LlmPolicy(ScriptedClient({"done": True}), registry=ops)
    first = RulePolicy(registry=ops).propose(
        assembly, [cert("insufficient_clearance")], [], BUDGET
    )
    history = [failed(first, "insufficient_clearance")]

    policy.propose(assembly, [cert("insufficient_clearance")], history, BUDGET)

    seen = policy.client.observation()["suggestions"]["insufficient_clearance"]
    assert "set_clearance" not in seen["params_hint"]  # the attempted rung is gone
    assert "resize_feature" in seen["params_hint"]  # the next rung remains
    assert seen["operators"][0] == "resize_feature"


def test_a_code_whose_whole_ladder_is_spent_gets_no_suggestion():
    ops = registry()
    assembly = base_assembly()
    policy = LlmPolicy(ScriptedClient({"done": True}), registry=ops)
    rule = RulePolicy(registry=ops)
    first = rule.propose(assembly, [cert("insufficient_clearance")], [], BUDGET)
    second = rule.propose(
        assembly, [cert("insufficient_clearance")], [failed(first)], BUDGET
    )
    history = [failed(first), failed(second)]

    policy.propose(assembly, [cert("insufficient_clearance")], history, BUDGET)

    # set_clearance and resize_feature are both spent; the fallback rungs are not
    # clearance moves, so the next suggestion can only come from them.
    suggestions = policy.client.observation()["suggestions"]["insufficient_clearance"]
    assert set(suggestions["params_hint"]) <= {"switch_connector", "regenerate_interface"}


def test_a_code_with_no_rule_still_gets_a_reason_and_the_fallback_rungs():
    suggestions = suggestions_for(["below_min_wall"], make_spec())

    row = suggestions["below_min_wall"]
    assert row["params_hint"]  # switch_connector / regenerate_interface
    assert "no measured size lever" in row["why"]


# --------------------------------------------------------------------------- #
# 2. history: what the action wrote, and how the codes moved
# --------------------------------------------------------------------------- #
def test_history_reports_the_change_and_the_code_delta():
    ops = registry()
    assembly = base_assembly()
    client = ScriptedClient({"done": True})
    policy = LlmPolicy(client, registry=ops)
    first = RulePolicy(registry=ops).propose(
        assembly, [cert("insufficient_clearance")], [], BUDGET
    )
    second = RulePolicy(registry=ops).propose(
        assembly,
        [cert("geometric_interference")],
        [failed(first, "insufficient_clearance")],
        BUDGET,
    )
    history = [
        Round(
            id="r0",
            index=0,
            plan=first,
            verdict="fail",
            evidence=[cert("geometric_interference")],
        ),
        Round(
            id="r1",
            index=1,
            plan=second,
            verdict="fail",
            evidence=[cert("insufficient_clearance", 0.05)],
        ),
    ]

    policy.propose(assembly, [cert("insufficient_clearance", 0.05)], history, BUDGET)

    rows = client.observation()["history"]
    assert [row["round"] for row in rows] == [0, 1]
    # The numbers each action asked to write, in the operator's own units.
    assert rows[0]["changed"] == {"clearance_mm": first.params["clearance_mm"]}
    assert rows[1]["changed"] == {"delta_mm": second.params.get("delta_mm")}
    # The code delta: round 0's before-set was never recorded, round 1's is round 0's after.
    assert rows[0]["codes_before"] is None
    assert rows[0]["cleared"] is None
    assert rows[1]["codes_before"] == ["geometric_interference"]
    assert rows[1]["cleared"] == ["geometric_interference"]
    assert rows[1]["appeared"] == ["insufficient_clearance"]
    # The verifier's own clearance pair, read off the fit certificates.
    assert rows[1]["measured_clearance_after_mm"] == pytest.approx(0.05)


def test_history_does_not_invent_a_before_value_it_does_not_have():
    """A resumed episode can only see closed rounds; inventing round 0's before-set
    would make its observation differ from an uninterrupted run's."""
    ops = registry()
    policy = LlmPolicy(ScriptedClient({"done": True}), registry=ops)
    plan = RulePolicy(registry=ops).propose(
        base_assembly(), [cert("insufficient_clearance")], [], BUDGET
    )

    policy.propose(
        base_assembly(),
        [cert("insufficient_clearance")],
        [Round(id="r0", index=0, plan=plan, verdict="fail", evidence=[])],
        BUDGET,
    )

    row = policy.client.observation()["history"][0]
    assert row["codes_before"] is None
    assert row["measured_clearance_before_mm"] is None
    assert row["codes_after"] == []


# --------------------------------------------------------------------------- #
# 3. a dead provider: reported, counted as environment, not as a policy failure
# --------------------------------------------------------------------------- #
def _episode(monkeypatch, tmp_path, client, *, rounds: int = 2, name: str = "ep") -> dict:
    """One composed episode over the fake stack, with a scripted decision layer."""
    ops = registry()
    policy = LlmPolicy(client, registry=ops)
    install(monkeypatch, policies={"llm": policy})
    # `policy_for("llm")` builds the provider client before it asks T3 for a policy;
    # stubbing the builder is what keeps this test offline (a real `pah.toml` would
    # otherwise be required, and the scripted client is already constructed).
    monkeypatch.setattr(runtime_module, "build_llm_client", lambda *_a, **_k: client)
    # After `install` so the real action space replaces the single-operator fake: the
    # policy validates against real parameter models, and the bridge applies for real.
    monkeypatch.setattr(runtime_module, "resolve_operators", lambda: ops)
    cfg = RuntimeConfig(out=tmp_path / name, policy="llm", rounds=rounds)
    with runtime_module.build(cfg) as composed:
        return composed.run_episode(
            design_spec(clearance_mm=0.0), policy_name="llm", rounds=rounds
        )


def test_a_provider_timeout_is_an_environment_error_not_a_policy_failure(monkeypatch, tmp_path):
    client = ScriptedClient(ProviderError("provider llm is unreachable: timed out"))

    report = _episode(monkeypatch, tmp_path, client)

    assert client.calls == 1  # one call, no retry
    assert report["episode"]["verdict"] == "infeasible"  # the loop may not know better
    errors = report["llm"]["errors"]
    assert errors == [
        {
            "code": "provider_timeout",
            "message": errors[0]["message"],
            "at_round": 0,
            "kind": "environment",
        }
    ]
    assert "timed out" in errors[0]["message"]
    assert report["llm"]["environment_errors"] == 1
    assert report["llm"]["answer_errors"] == 0
    assert report["counts"]["repaired_by_policy"] == 0
    assert report["counts"]["environment_errors"] >= 1
    assert report["counts"]["environment_error_sources"]["llm"] == 1


def test_an_unusable_reply_is_recorded_as_the_policys_own_failure(monkeypatch, tmp_path):
    """A model that answers garbage *is* the policy failing — the opposite call from
    a provider outage, and the reason the ledger carries a `kind`."""
    client = ScriptedClient("Sure! Here is the plan: use set_clearance.")

    report = _episode(monkeypatch, tmp_path, client)

    assert client.calls == 1  # no second attempt with a "closer" prompt
    errors = report["llm"]["errors"]
    assert [(row["code"], row["kind"]) for row in errors] == [("not_json", "answer")]
    assert errors[0]["at_round"] == 0
    assert report["llm"]["answer_errors"] == 1
    assert report["llm"]["environment_errors"] == 0
    assert report["counts"]["environment_error_sources"]["llm"] == 0
    assert report["counts"]["environment_errors"] == 0
    assert report["counts"]["repaired_by_policy"] == 0


def test_a_good_reply_still_repairs_and_leaves_no_error_rows(monkeypatch, tmp_path):
    """The feedback upgrade must not change the decision path: a usable reply is
    still accepted and still closes the loop."""
    client = ScriptedClient(
        {
            "operator": "set_clearance",
            "params": {
                "interface_id": "iface_0",
                "parts": ["body", "insert"],
                "clearance_mm": 0.4,
            },
            "reason": "insufficient_clearance: mid band",
        }
    )

    report = _episode(monkeypatch, tmp_path, client, rounds=2)

    assert report["episode"]["verdict"] == "repaired"
    assert report["counts"]["repaired_by_policy"] == 1
    assert report["llm"]["errors"] == []
    assert report["counts"]["environment_errors"] == 0
    assert report["rounds"][0]["proposed_by"] == "llm"


def test_the_ledger_is_per_episode_not_per_process(monkeypatch, tmp_path):
    """The runtime caches one policy per process; a second episode must not inherit
    the first one's refusals as its own."""
    client = ScriptedClient(ProviderError("timed out"))
    ops = registry()
    policy = LlmPolicy(client, registry=ops)
    install(monkeypatch, policies={"llm": policy})
    monkeypatch.setattr(runtime_module, "build_llm_client", lambda *_a, **_k: client)
    monkeypatch.setattr(runtime_module, "resolve_operators", lambda: ops)

    reports = []
    for name, rounds in (("first", 1), ("second", 1)):
        cfg = RuntimeConfig(out=tmp_path / name, policy="llm", rounds=rounds)
        with runtime_module.build(cfg) as composed:
            reports.append(
                composed.run_episode(
                    design_spec(clearance_mm=0.0), policy_name="llm", rounds=rounds
                )
            )

    assert [len(report["llm"]["errors"]) for report in reports] == [1, 1]
    assert len(policy.errors) == 2  # the ledger itself accumulates, the reports split
