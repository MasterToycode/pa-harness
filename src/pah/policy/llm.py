"""The model-backed repair policy: one JSON object per round, validated before it moves anything.

This is 方案 A (`docs/pah-dev-doc-v1.md` §4, `docs/contracts/v2-interfaces.md` §9):
the model is a **planner**, not a generator. It names an operator and the numbers
that operator takes; it never emits geometry, and there is no field in its output
type in which geometry could be smuggled. The geometry is the operators' job
(`policy/design_ops.py`), exactly as it is for the deterministic baseline, so a
win for this policy is a win for *deciding*, not for drawing.

Three rules, and each is a way this module could have been wrong
---------------------------------------------------------------
1. **The legal action list is the registry's, not the model's.** The observation
   carries `legal_actions` — the operators `OperatorRegistry.applicable` admits for
   this evidence, minus the ones the connector kind cannot express
   (`design_ops.expressible`). An operator id outside that list is refused after
   the call, so a model that invents `move_vertex` is answering a different
   question and is told so.
2. **Every rejection is recorded, never repaired.** A parse failure, an unknown
   operator, a parameter set that fails the operator's own pydantic model, or
   targets that disagree with the parameters: each returns `None` with
   `last_error` set. There is **no fallback** to another policy and no second
   attempt with a "closer" prompt. A run whose two calls went to different
   strategies is not one experiment, and the equal-budget comparison in the eval
   would be measuring the fallback.
3. **Abstaining is not failing.** `{"done": true}` (or a null operator) returns
   `None` with `last_error == ""`, because "no action is worth taking" is a
   legitimate answer (`loop.LoopPolicy`). A caller that cannot tell that apart
   from a failed call will report a provider outage as a model that found nothing
   to do — which is why `last_error` exists and why `propose` never raises.

What the model is shown
-----------------------
One compact JSON object (`LlmPolicy.observation`): the design's key numbers, the
failures with their magnitudes, the plans already tried and how they ended, the
legal action list with each operator's parameter schema, and the budget left.
Nothing else — no geometry, no mesh, no file path. The system prompt is
`prompts/repair_policy.md`; it is a file rather than a string so that prompt
changes are a reviewable diff and a recording stays interpretable.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pah.app.providers import LLMClient, LLMResult
from pah.core.ids import key
from pah.core.model import (
    Assembly,
    Budget,
    Cost,
    FailureCertificate,
    RepairPlan,
    Round,
)
from pah.design.api import spec_of
from pah.design.spec import DesignSpec
from pah.policy.design_ops import expressible
from pah.policy.registry import Operator

__all__ = ["LlmPolicy", "OUTPUT_SCHEMA", "default_system_prompt"]

_ENVIRONMENT = "environment"
_PROMPT_NAME = "repair_policy.md"

#: What one round's reply must look like. Sent as `json_schema` so a provider that#: supports structured output can enforce it; the client in `app/providers.py`
#: currently downgrades it to `response_format={"type": "json_object"}`, so this is
#: documentation for the model as much as a constraint.
OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "operator": {
            "type": ["string", "null"],
            "description": "one id from legal_actions, or null to abstain",
        },
        "params": {
            "type": "object",
            "description": "exactly the fields legal_actions[operator] requires",
        },
        "targets": {
            "type": "array",
            "items": {"type": "string"},
            "description": "optional; if given it must equal the targets the params compute",
        },
        "reason": {"type": "string", "description": "one sentence, for the trace"},
        "done": {
            "type": "boolean",
            "description": "true when no action in legal_actions is worth taking",
        },
    },
    "additionalProperties": False,
}


def _field_only_schema(model) -> dict:
    """The parameter model's schema with every `description` removed.

    The prose that documents an operator belongs to the humans reading this module:
    it explains *why* an operator exists and mentions things like vertices, files and
    tolerances. Sending that prose inside the observation is how geometry vocabulary
    leaks back into the model's context even though the action space has no geometry
    field in it — the contract's "the LLM's output type has no geometry" is about the
    *shape of the decision*, and the observation should hold the same line.

    Only structure survives: property names, types, enum values, required lists.
    """
    schema = model.model_json_schema()
    return _strip_descriptions(schema)


def _strip_descriptions(node):
    if isinstance(node, dict):
        return {
            key: _strip_descriptions(value)
            for key, value in node.items()
            if key not in ("description", "title")
        }
    if isinstance(node, list):
        return [_strip_descriptions(item) for item in node]
    return node


def default_system_prompt() -> str:
    """The prompt file, read once per process."""
    return (
        Path(__file__).with_name("prompts") / _PROMPT_NAME
    ).read_text(encoding="utf-8")


class LlmPolicy:
    """`loop.LoopPolicy` backed by a `LLMClient`. Stateless apart from `last_error`."""

    def __init__(
        self,
        client: LLMClient,
        *,
        registry,
        name: str = "llm",
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        geom_calls_per_round: int = 1,
    ) -> None:
        self.client = client
        self.registry = registry
        self.name = name
        self.system_prompt = system_prompt or default_system_prompt()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.geom_calls_per_round = geom_calls_per_round
        #: Why the last `propose` produced no plan. `""` means either "a plan came
        #: back" or "the model chose to abstain"; a caller that needs to tell those
        #: apart reads it, and the eval must, or a provider outage would be counted
        #: as a model that had nothing to do.
        self.last_error = ""

    # -- the decision ------------------------------------------------------
    def propose(
        self,
        assembly: Assembly,
        certs: list[FailureCertificate],
        history: list[Round],
        budget_left: Budget,
    ) -> RepairPlan | None:
        self.last_error = ""
        evidence = [cert for cert in certs if cert.family != _ENVIRONMENT]
        try:
            spec = spec_of(assembly)
        except KeyError as error:
            self.last_error = f"not a design-plane assembly: {error}"
            return None
        if not evidence:
            return None

        legal = self._legal(spec, assembly, evidence)
        if not legal:
            return None

        observation = self._observation(spec, evidence, history, budget_left, legal)
        try:
            result = self.client.complete(
                system=self.system_prompt,
                user=json.dumps(observation, ensure_ascii=False, default=str),
                json_schema=OUTPUT_SCHEMA,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as error:  # a dead provider is not an abstention
            self.last_error = f"provider error: {type(error).__name__}: {error}"
            return None
        return self._decode(result, legal, assembly)

    # -- what the model is shown ------------------------------------------
    def _legal(
        self, spec: DesignSpec, assembly: Assembly, evidence: Sequence[FailureCertificate]
    ) -> list[str]:
        """The operators this evidence admits and this connector kind can express."""
        return [
            operator_id
            for operator_id in self.registry.applicable(assembly, list(evidence))
            if expressible(spec.interface.kind, operator_id)
        ]

    def _observation(
        self,
        spec: DesignSpec,
        evidence: Sequence[FailureCertificate],
        history: Sequence[Round],
        budget_left: Budget,
        legal: Sequence[str],
    ) -> dict:
        band = spec.constraints
        return {
            "design": {
                "spec_id": spec.spec_id,
                "requirement": spec.requirement,
                "revision": spec.revision,
                "kind": spec.interface.kind,
                "interface_id": spec.interface.id,
                "parts": [spec.interface.a, spec.interface.b],
                "clearance_mm": round(spec.clearance_mm, 6),
                "acceptance": {
                    "min_clearance_mm": band.min_clearance_mm,
                    "max_clearance_mm": band.max_clearance_mm,
                    "min_wall_mm": band.min_wall_mm,
                    "target_fit": band.target_fit,
                    "process": band.process,
                },
                "interface_params": spec.interface.validated_params().model_dump(),
                "part_poses": [
                    {
                        "id": part.id,
                        "role": part.role,
                        "size_mm": list(part.size_mm),
                        "pose_mm": list(part.pose_mm),
                        "rot_deg": list(part.rot_deg),
                    }
                    for part in spec.parts
                ],
            },
            "failures": [
                {
                    "code": cert.code,
                    "family": cert.family,
                    "magnitude": dict(cert.magnitude),
                    "locus": dict(cert.locus),
                    "observed_by": cert.observed_by,
                }
                for cert in evidence
            ],
            "history": [
                {
                    "operator": item.plan.operator,
                    "targets": list(item.plan.targets),
                    "params": dict(item.plan.params),
                    "verdict": item.verdict,
                    "codes_after": sorted({cert.code for cert in item.evidence}),
                }
                for item in history
            ],
            "legal_actions": {
                operator_id: _field_only_schema(self.registry.get(operator_id).params)
                for operator_id in legal
            },
            "budget_left": {
                "rounds": budget_left.rounds,
                "geom_calls": budget_left.geom_calls,
                "sim_calls": budget_left.sim_calls,
            },
            "output": {
                "operator": "one id from legal_actions, or null",
                "params": "the fields that operator's schema requires",
                "targets": "optional; must equal what the params compute",
                "reason": "one sentence",
                "done": "true to abstain",
            },
        }

    # -- what comes back ---------------------------------------------------
    def _decode(
        self, result: LLMResult, legal: Sequence[str], assembly: Assembly
    ) -> RepairPlan | None:
        """The reply as a plan, or `None` with `last_error` saying why not."""
        try:
            payload = json.loads(result.text)
        except (TypeError, ValueError) as error:
            self.last_error = f"not JSON: {error}: {_excerpt(result.text)}"
            return None
        if not isinstance(payload, Mapping):
            self.last_error = f"reply is a {type(payload).__name__}, not a JSON object"
            return None
        if payload.get("done") is True or payload.get("operator") is None:
            return None  # an abstention, which is an answer

        operator_id = payload.get("operator")
        if not isinstance(operator_id, str) or not operator_id:
            self.last_error = f"operator is {operator_id!r}, not an operator id"
            return None
        if operator_id not in legal:
            self.last_error = (
                f"operator {operator_id!r} is not in this round's legal actions "
                f"{list(legal)}"
            )
            return None

        operator: Operator = self.registry.get(operator_id)
        try:
            params = operator.params.model_validate(payload.get("params") or {})
        except Exception as error:  # ValidationError, or a params field that is not a dict
            self.last_error = (
                f"params for {operator_id!r} are invalid: {type(error).__name__}: {error}"
            )
            return None

        payload_params = params.model_dump()
        targets = tuple(operator.targets(payload_params))
        declared = payload.get("targets")
        if declared is not None and tuple(str(item) for item in declared) != targets:
            self.last_error = (
                f"declared targets {tuple(declared)!r} disagree with the ones "
                f"{operator_id!r} computes from its params: {targets!r}"
            )
            return None

        plan = RepairPlan(
            id=key("plan", operator_id, targets, payload_params),
            operator=operator_id,
            targets=targets,
            params=payload_params,
            cost_est=Cost(
                geom_calls=self.geom_calls_per_round,
                llm_calls=1,
                tokens=result.usage.tokens,
            ),
            proposed_by="llm",
            reason=str(payload.get("reason") or "")[:500],
        )
        # The parameter model proves the *payload* is well formed. This proves the
        # *design* it describes exists: e.g. a dovetail patch asking for a negative
        # per-side clearance is a legal dict and an unbuildable part. Free to check
        # because an operator is pure (`design_ops` docstring), and worth checking
        # because a plan that cannot be built costs the round and then reports an
        # operator error instead of a decision.
        try:
            self.registry.apply(assembly, plan)
        except Exception as error:
            self.last_error = (
                f"the plan does not build on this snapshot: "
                f"{type(error).__name__}: {error}"
            )
            return None
        return plan


def _excerpt(text: object, limit: int = 200) -> str:
    """The first `limit` characters of a reply, for the error message."""
    return repr(str(text)[:limit])
