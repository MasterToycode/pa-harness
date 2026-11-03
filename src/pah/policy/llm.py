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
legal action list with each operator's parameter schema, the budget left, and —
since P3-2 — the rule baseline's *suggestions* for each observed code.

Two of those deserve their reasoning written down:

* **`suggestions` is the rule's arithmetic, not the model's.** For every code it
  observed, the model gets the deterministic ladder's row plus the exact parameter
  payloads that ladder would send (`design_policies.suggestions_for`). The point is
  not to make the model copy: it is to make the *numbers* come from a place that
  can be audited, so the only thing the model is deciding is *whether and which*
  action — the question RQ2 asks. A model that invents `clearance_mm = 0.2` because
  it sounds snug is not being measured on strategy, it is being measured on
  arithmetic it was never given.
* **`history` says what changed, not just what was tried.** Every closed round
  carries the numbers the plan wrote (`changed`), the failure codes before and after
  (`codes_before`/`codes_after`, `cleared`, `appeared`), and the *measured*
  clearance pair where a fit certificate recorded one. That is the feedback signal a
  repair loop needs: "the last action opened the fit by 0.35 mm and the fit code
  disappeared" is a different prompt from "an action was tried".

What is deliberately absent: geometry, meshes, file paths — and any claim that a
repair *worked*. Only a verifier may say that, and the model only ever sees its
certificates. The system prompt is `prompts/repair_policy.md`; it is a file rather
than a string so that prompt changes are a reviewable diff and a recording stays
interpretable.
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
from pah.design.api import (
    CODE_EXCESSIVE_CLEARANCE,
    CODE_INSUFFICIENT_CLEARANCE,
    spec_of,
)
from pah.design.spec import DesignSpec
from pah.policy.design_ops import expressible
from pah.policy.design_policies import params_signal, suggestions_for
from pah.policy.registry import Operator

__all__ = ["LlmPolicy", "OUTPUT_SCHEMA", "default_system_prompt"]

_ENVIRONMENT = "environment"
#: The two kinds of row `LlmPolicy.errors` holds. `environment` means the decision
#: never reached a model (a timeout, a refused key, an exhausted quota) — a tool
#: fault, which must not be counted against the policy. `answer` means a model
#: replied and the reply was unusable — that *is* the policy failing.
_KIND_ENVIRONMENT = "environment"
_KIND_ANSWER = "answer"
_PROMPT_NAME = "repair_policy.md"

#: What one round's reply must look like. Sent as `json_schema` so a provider that
#: supports structured output can enforce it; the client in `app/providers.py`
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
        #: Every refusal this policy has produced, one row each, in call order:
        #: ``{"code", "message", "at_round", "kind"}``. Kept as a ledger rather than
        #: only a `last_error` because the *loop* reports a provider timeout and a
        #: model that found nothing to do identically (`None` -> `infeasible`); it is
        #: this ledger that lets `app/runtime.py` put the difference in the report and
        #: keep a harness fault out of the policy's failure rate.
        self.errors: list[dict] = []

    # -- the decision ------------------------------------------------------
    def propose(
        self,
        assembly: Assembly,
        certs: list[FailureCertificate],
        history: list[Round],
        budget_left: Budget,
    ) -> RepairPlan | None:
        self.last_error = ""
        at_round = len(history)
        evidence = [cert for cert in certs if cert.family != _ENVIRONMENT]
        try:
            spec = spec_of(assembly)
        except KeyError as error:
            self._record(
                kind=_KIND_ENVIRONMENT,
                code="not_a_design_plane",
                message=f"not a design-plane assembly: {error}",
                at_round=at_round,
            )
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
            self._record(
                kind=_KIND_ENVIRONMENT,
                code=_provider_code(error),
                message=f"provider error: {type(error).__name__}: {error}",
                at_round=at_round,
            )
            return None
        return self._decode(result, legal, assembly, at_round)

    def _record(self, *, kind: str, code: str, message: str, at_round: int) -> None:
        """One refusal, in the ledger and in `last_error`.

        One call per round is the contract (`propose` never retries), so a row here
        is also the record of a round that produced no decision. The `kind` is what
        the accounting needs; the `code` is what a human scans first.
        """
        self.last_error = message
        self.errors.append(
            {
                "code": code,
                "message": message[:500],
                "at_round": int(at_round),
                "kind": kind,
            }
        )

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
            "history": _history(history),
            "suggestions": self._suggestions(spec, evidence, legal, history),
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

    def _suggestions(
        self,
        spec: DesignSpec,
        evidence: Sequence[FailureCertificate],
        legal: Sequence[str],
        history: Sequence[Round],
    ) -> dict:
        """The rule baseline's numbers for each observed code, narrowed to this round.

        `design_policies.suggestions_for` is the single source — the same ladder and
        the same band arithmetic the deterministic policy walks. Two filters live
        here and not there, because they are facts about *this* round rather than
        about the design:

        * `legal` — what the registry admits for this evidence;
        * already-tried plans — a hint that repeats a failed attempt is worse than no
          hint, because it invites the one action the history already ruled out. A
          code whose every suggestion has been attempted gets no entry at all, which
          is the honest answer: the baseline has nothing new for it.
        """
        allowed = set(legal)
        tried = _attempted(history)
        rows: dict[str, dict] = {}
        for code, row in suggestions_for(
            [cert.code for cert in evidence], spec, skip=tried
        ).items():
            hints = {
                operator_id: hint
                for operator_id, hint in row["params_hint"].items()
                if operator_id in allowed
            }
            if not hints:
                continue
            rows[code] = {
                "operators": [op for op in row["operators"] if op in hints],
                "params_hint": hints,
                "why": row["why"],
            }
        return rows

    # -- what comes back ---------------------------------------------------
    def _decode(
        self,
        result: LLMResult,
        legal: Sequence[str],
        assembly: Assembly,
        at_round: int,
    ) -> RepairPlan | None:
        """The reply as a plan, or `None` with the refusal recorded.

        Every branch below records before returning: a silent `None` is what turns a
        wrong answer into an episode that looks like a model with nothing to say.
        """

        def refuse(code: str, message: str) -> None:
            self._record(kind=_KIND_ANSWER, code=code, message=message, at_round=at_round)

        try:
            payload = json.loads(result.text)
        except (TypeError, ValueError) as error:
            refuse("not_json", f"not JSON: {error}: {_excerpt(result.text)}")
            return None
        if not isinstance(payload, Mapping):
            refuse(
                "not_an_object",
                f"reply is a {type(payload).__name__}, not a JSON object",
            )
            return None
        if payload.get("done") is True or payload.get("operator") is None:
            return None  # an abstention, which is an answer

        operator_id = payload.get("operator")
        if not isinstance(operator_id, str) or not operator_id:
            refuse("bad_operator_field", f"operator is {operator_id!r}, not an operator id")
            return None
        if operator_id not in legal:
            refuse(
                "unknown_operator",
                f"operator {operator_id!r} is not in this round's legal actions "
                f"{list(legal)}",
            )
            return None

        operator: Operator = self.registry.get(operator_id)
        try:
            params = operator.params.model_validate(payload.get("params") or {})
        except Exception as error:  # ValidationError, or a params field that is not a dict
            refuse(
                "invalid_params",
                f"params for {operator_id!r} are invalid: {type(error).__name__}: {error}",
            )
            return None

        payload_params = params.model_dump()
        targets = tuple(operator.targets(payload_params))
        declared = payload.get("targets")
        if declared is not None and tuple(str(item) for item in declared) != targets:
            refuse(
                "targets_mismatch",
                f"declared targets {tuple(declared)!r} disagree with the ones "
                f"{operator_id!r} computes from its params: {targets!r}",
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
            refuse(
                "unbuildable",
                f"the plan does not build on this snapshot: "
                f"{type(error).__name__}: {error}",
            )
            return None
        return plan


# --------------------------------------------------------------------------- #
# The feedback block: what the last actions did
# --------------------------------------------------------------------------- #
def _history(rounds: Sequence[Round]) -> list[dict]:
    """Each closed round as "what was written" plus "what the codes did".

    `codes_before` is the previous round's `codes_after`, and `None` for the first
    round: the loop hands a policy only closed rounds, and the codes round 0 faced
    were never recorded in one. That is a documented gap rather than a guess — a
    guessed value would make a resumed episode's observation differ from an
    uninterrupted one, and `--resume` is supposed to be invisible.

    `measured_clearance_*` is the verifier's own number where a fit certificate
    recorded one, which is why it is a *pair*: the before-value is the previous
    round's after-value, so the model reads "the last action took the built clearance
    from -0.1000 to +0.3500 mm and the fit code cleared" instead of "an action was
    tried".
    """
    rows: list[dict] = []
    previous_codes: list[str] | None = None
    previous_clearance: float | None = None
    for index, item in enumerate(rounds):
        codes_after = sorted({cert.code for cert in item.evidence})
        measured_after = _measured_clearance(item.evidence)
        rows.append(
            {
                "round": index,
                "operator": item.plan.operator,
                "targets": list(item.plan.targets),
                "params": dict(item.plan.params),
                "changed": _changed(item.plan),
                "verdict": item.verdict,
                "codes_before": previous_codes,
                "codes_after": codes_after,
                "cleared": (
                    None
                    if previous_codes is None
                    else sorted(set(previous_codes) - set(codes_after))
                ),
                "appeared": (
                    None
                    if previous_codes is None
                    else sorted(set(codes_after) - set(previous_codes))
                ),
                "measured_clearance_before_mm": previous_clearance,
                "measured_clearance_after_mm": measured_after,
            }
        )
        previous_codes = codes_after
        previous_clearance = measured_after
    return rows


def _changed(plan: RepairPlan) -> dict:
    """The numbers the plan asked to write, in the operator's own units.

    A from→to pair per round is not recoverable: a `Round` records the plan and the
    evidence *after* it, never the spec it was applied to. So this states the write
    (a target, a delta, a new kind) and `_history` adds the verifier-measured pair
    for the quantity the band is judged on. The alternative — reconstructing the
    chain of edits in the policy — would be a second source of truth for what
    happened, which is exactly what the verifier's numbers exist to replace.
    """
    params = dict(plan.params)
    operator = plan.operator
    if operator == "set_clearance":
        return {"clearance_mm": params.get("clearance_mm")}
    if operator in ("resize_feature", "resize_insert"):
        return {"delta_mm": params.get("delta_mm")}
    if operator == "shift_part":
        return {"pose_delta_mm": params.get("delta_mm")}
    if operator == "rotate_part":
        return {"rot_delta_deg": params.get("delta_deg")}
    if operator == "switch_connector":
        return {"kind": params.get("kind")}
    if operator == "regenerate_interface":
        return {"params": params.get("params")}
    return params


def _signature(operator: str, params: Mapping[str, object]) -> tuple[str, str]:
    """`(operator, canonical params)` — the identity of one attempt.

    `params_signal` is shared with `design_policies`' repeat guard: two spellings of
    "the same attempt" is how a repeat slips through one of them.
    """
    return operator, params_signal(params)


def _attempted(history: Sequence[Round]) -> set[tuple[str, str]]:
    """Every `(operator, params)` already tried, failures and errors alike."""
    return {
        _signature(str(item.plan.operator), dict(item.plan.params))
        for item in history
    }


def _measured_clearance(evidence: Sequence[FailureCertificate]) -> float | None:
    """The radial clearance a fit certificate measured, or `None`.

    Read from the certificate, never recomputed from the spec: the loop's whole
    discipline is that only a verifier may say what the geometry is, and this block
    is the one place a policy gets to look at what a verifier saw.
    """
    for cert in evidence:
        if cert.code not in (CODE_INSUFFICIENT_CLEARANCE, CODE_EXCESSIVE_CLEARANCE):
            continue
        value = (cert.magnitude or {}).get("clearance_mm")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return round(float(value), 6)
    return None


def _provider_code(error: BaseException) -> str:
    """A machine code for a decision that never reached a model.

    Timeout is worth its own code: it is the one provider fault where retrying the
    same episode later is the documented fix (`docs/contracts/v2-interfaces.md` §7),
    so a report that lumps it with an exhausted quota sends the reader to the wrong
    page.
    """
    text = f"{type(error).__name__} {error}".lower()
    return "provider_timeout" if "timeout" in text or "timed out" in text else "provider_error"


def _excerpt(text: object, limit: int = 200) -> str:
    """The first `limit` characters of a reply, for the error message."""
    return repr(str(text)[:limit])
