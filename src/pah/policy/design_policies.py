"""The design plane's baselines: a deterministic rule policy and a seeded random one.

`RulePolicy` is the P0 policy — the one the loop must close with before a learned
policy has anything to be compared against (`docs/internal/repair-loop-spec.md`
§2.3). `RandomPolicy` is the floor: same action space, same budget, no reasoning.
Both are `loop.LoopPolicy` (`name` + `propose`), so the loop cannot tell them
apart, which is the point.

What a rule is allowed to be
----------------------------
A rule reads **design intent and evidence** — the spec's numbers, the fit band,
the certificate codes — and produces a *number*. It never reads geometry and
never concludes anything: `propose` returning `None` ("no action I am willing to
take") is a normal answer, and only a verifier may say whether a repair worked.

Three properties earn their keep, in the order they can fail:

1. **It never invents an action.** Every candidate comes from the operator
   registry (`registry.applicable`), and the plan's targets are computed by the
   operator itself (`Operator.targets`), never paraphrased here. A rule that
   wrote its own targets would be one refactor away from a plan gate 3 refuses.
2. **It does not repeat itself.** A candidate already attempted at the same
   operator with the same targets and the same numbers is skipped, so a failing
   plan cannot be resubmitted until the budget runs out — the loop's cycle guard
   would otherwise catch it only after paying for a round.
3. **It asks whether the action is expressible at all.** The registry's
   preconditions never see parameters, so they cannot know that a dovetail has no
   insert-side dimension. `design_ops.expressible` does, and asking before
   proposing keeps an impossible repair out of the trace instead of turning it
   into an error round that teaches nothing.

The arithmetic is not here
--------------------------
`set_clearance` and the two resize rules compute a *clearance target* and hand it
to `design_ops`, which owns the translation into each kind's own parameter
(`params_with_clearance`, `delta_for_clearance`). A rule with its own copy of
"peg_hole wants d_hole = d_peg + 2c" would be the second source of truth for the
one number this whole plane argues about.

The ladder, and why a paradigm switch is last rather than first
---------------------------------------------------------------
Each code has a short list of *specific* repairs, then everyone shares the same
tail: `switch_connector` (a different fit paradigm) and `regenerate_interface`
(re-seat the interface from its intent). The tail is last on purpose: changing
the connector is the move that makes the previous evidence uninterpretable, so it
is what you do when the specific arithmetic has already failed — not what you do
because you could not decide.
"""

from __future__ import annotations

import json
import random
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

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
    CODE_DYNAMIC_PENETRATION,
    CODE_EXCESSIVE_CLEARANCE,
    CODE_GEOMETRIC_INTERFERENCE,
    CODE_INSUFFICIENT_CLEARANCE,
    CODE_INSERTION_PATH_BLOCKED,
    CODE_JOINT_UNSTABLE,
    DESIGN_OPERATOR_IDS,
    spec_of,
)
from pah.design.spec import DesignSpec, FitConstraints
from pah.policy.design_ops import (
    CONNECTOR_KINDS,
    delta_for_clearance,
    expressible,
    next_connector_kind,
    params_with_clearance,
    resize_field,
    resize_params,
)
from pah.policy.registry import Operator

__all__ = [
    "DEFAULT_LADDER",
    "FALLBACK_LADDER",
    "POLICY_NAMES",
    "RandomPolicy",
    "RulePolicy",
    "band_target",
    "build",
    "expressible",
    "load_ladder",
]

#: The names `build` accepts, published so a CLI or an eval can list them (and
#: refuse an unknown name) without keeping a second copy of this tuple.
POLICY_NAMES: tuple[str, ...] = ("rule", "random", "llm")

_ENVIRONMENT = "environment"

#: Which edge of the fit band a rule aims at, per triggering code.
#:
#: `mid` is the default because it is the point furthest from both verifier
#: inequalities, so a rounding error on the way to the materialiser cannot push
#: the answer back out of the band. `low` is for "too much": the defect is an
#: excess, and the smallest legal clearance is the repair that changes the design
#: least. `high` is for a blocked insertion path: the blockage is a
#: clearance-limited channel, so the repair is to open it as far as the band
#: allows — "it does not fit" is not answered by "make it tighter".
#:
#: The `high` choice is a rationale, not a current measurement: on 2026-11-01 the
#: then-current dovetail fixture needed clearance 0.55 for its path block to clear,
#: so `mid` (0.35) would have spent a round proving it was not enough. That fixture
#: has since been corrected upstream (it is clean at clearance 0.10 now), so the
#: rule is kept for the case where a blockage *is* clearance-limited.
_EDGE: dict[str, Literal["low", "mid", "high"]] = {
    CODE_INSUFFICIENT_CLEARANCE: "mid",
    CODE_EXCESSIVE_CLEARANCE: "low",
    CODE_GEOMETRIC_INTERFERENCE: "low",
    CODE_INSERTION_PATH_BLOCKED: "high",
}

#: Code -> the operators worth trying, most specific first. Every row is a
#: *priority*, not a permission: the registry decides legality (`applicable`), so
#: a rung whose preconditions do not fire for this code is skipped, not obeyed.
#:
#: `geometric_interference` rungs follow the contract's trigger table: the fit
#: verifier's codes are what make a *cavity* operator legal, so a lone overlap
#: certificate can only turn the insert down (`resize_insert`, negative delta) or
#: move it (`shift_part`). When the same design is also out of band — the §7 case
#: `d_peg=10.2, d_hole=10.0` — the cheaper fit certificate arrives first and
#: `set_clearance` takes the turn instead.
#:
#: `insertion_path_blocked` puts `resize_feature` first and the pose guesses after
#: it, which is this module's own rule applied honestly: a number with a known band
#: beats a 3-D guess. Opening the cavity widens the channel the insert travels
#: through, monotonically; rotating the insert does not — measured 2026-11-01, one
#: 15° tilt on a path-blocked dovetail added `geometric_interference` without
#: clearing the original code.
#:
#: The printability and physics codes have no specific rung: their repair is a
#: *shape* change whose size nobody has measured, so only the shared tail applies.
#: Inventing a number for them would be a policy claiming a conclusion it cannot
#: support.
DEFAULT_LADDER: dict[str, tuple[str, ...]] = {
    CODE_INSUFFICIENT_CLEARANCE: ("set_clearance", "resize_feature"),
    CODE_EXCESSIVE_CLEARANCE: ("set_clearance", "resize_insert"),
    CODE_GEOMETRIC_INTERFERENCE: ("resize_insert", "shift_part"),
    CODE_INSERTION_PATH_BLOCKED: ("resize_feature", "rotate_part", "shift_part"),
    "below_min_wall": (),
    "unsupported_overhang": (),
    CODE_JOINT_UNSTABLE: (),
    CODE_DYNAMIC_PENETRATION: (),
}

#: Appended to every row. `switch_connector` first because a paradigm change is
#: the only remaining move whose effect can be predicted without a physics model;
#: `regenerate_interface` second because in v1 it writes the numbers a specific
#: rule has usually already written, so it is the honest "nothing else left".
FALLBACK_LADDER: tuple[str, ...] = ("switch_connector", "regenerate_interface")

#: A deterministic in-plane offset ladder, in mm.
#:
#: Two properties, both learned the hard way. It is **deterministic**, because two
#: runs of the same case must be identical. And each rung's magnitude in an axis is
#: larger than the sum of the rungs before it in that axis, so **no two rungs can
#: cancel**: an earlier version tried `(+2, -2)` on the current pose, and the second
#: shift walked the part back to a pose already fingerprinted, which the loop
#: correctly read as a geometry cycle and ended the episode. A policy must not
#: propose an action that returns the design to where it has been.
#:
#: The cumulative poses are ``(+2,0,0) (-1,0,0) (-1,+2,0) (-1,-1,0) (+3,-1,0)
#: (+3,+3,0)`` — six distinct places, both axes, no repeats.
_SHIFT_LADDER: tuple[tuple[float, float, float], ...] = (
    (2.0, 0.0, 0.0),
    (-3.0, 0.0, 0.0),
    (0.0, 2.0, 0.0),
    (0.0, -3.0, 0.0),
    (4.0, 0.0, 0.0),
    (0.0, 4.0, 0.0),
)

#: The same idea for rotation, in degrees about the world axes. In-plane rotation
#: about Z cannot unblock a straight +Z approach, so only the two tilts appear;
#: magnitudes grow for the same no-cancellation reason as `_SHIFT_LADDER`.
_ROTATE_LADDER: tuple[tuple[float, float, float], ...] = (
    (15.0, 0.0, 0.0),
    (-22.0, 0.0, 0.0),
    (0.0, 15.0, 0.0),
    (0.0, -22.0, 0.0),
)


# --------------------------------------------------------------------------- #
# Shared arithmetic — the band, and the operator catalogue both policies use
# --------------------------------------------------------------------------- #
#: How far inside a band edge a target is placed, in millimetres.
#:
#: One nanometre — the trace's own precision — and it exists because of a float
#: fact, not a stylistic preference. The write path is ``d_hole = d_peg + 2c`` and
#: the read path is ``c = (d_hole - d_peg) / 2``; that round trip is *not* exact
#: for the round kinds (``10.0 + 2*0.1 - 10.0`` is ``0.19999999999999929``), so a
#: target sitting exactly on a band edge reads back a hair outside it and the fit
#: verifier — correctly — keeps saying "still broken". Measured here, on the
#: `d_peg=10.2, d_hole=10.0` case: aiming at ``min_clearance_mm=0.1`` produced a
#: built clearance of ``0.09999999999999964`` and one wasted round.
_EDGE_INSET_MM = 1e-6


def band_target(
    constraints: FitConstraints, where: Literal["low", "mid", "high"] = "mid"
) -> float:
    """A clearance inside the acceptance band, at `where`.

    `FitConstraints.in_band` is inclusive, so an edge *would* be a legal target —
    `_EDGE_INSET_MM` explains why the edges are inset anyway. A band too narrow to
    afford the inset (or a zero-width one) gets its midpoint: there is no room for
    an edge strategy, so the only target that survives the round trip is the one
    in the middle. Values are rounded to the six decimals the trace records.
    """
    low, high = constraints.min_clearance_mm, constraints.max_clearance_mm
    if where == "low" and high - low > 2.0 * _EDGE_INSET_MM:
        return round(low + _EDGE_INSET_MM, 6)
    if where == "high" and high - low > 2.0 * _EDGE_INSET_MM:
        return round(high - _EDGE_INSET_MM, 6)
    return round((low + high) / 2.0, 6)


class OperatorCatalogue(Protocol):
    """The slice of `OperatorRegistry` a policy needs — same shape as `policy.rule`."""

    def applicable(
        self, asm: Assembly, certs: Sequence[FailureCertificate]
    ) -> list[str]: ...

    def get(self, operator_id: str) -> Operator: ...


def _normalise(value: object) -> object:
    """A parameter payload as it is compared and recorded: floats to six decimals.

    Two plans are "the same plan" only if a person would call them the same, and
    `0.30000000000000004` versus `0.3` is the same plan — an arithmetic difference
    that came from *how* a number was computed, not from what was asked for. Six
    decimals is the trace's convention (`design_ops._mm`), so rounding here also
    keeps a plan's id stable across a rule policy and a replayed LLM policy that
    chose the same number.
    """
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        return round(value, 6) + 0.0  # `+ 0.0` folds -0.0 into 0.0
    if isinstance(value, Mapping):
        return {str(name): _normalise(item) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalise(item) for item in value]
    return str(value)


def _signal(params: Mapping[str, object]) -> str:
    """A stable text for one parameter payload.

    `json.dumps(..., sort_keys=True)` over the normalised payload is the whole
    trick: two plans are "the same plan" iff their operator, targets and this text
    agree, and a text comparison is exact where a float comparison would need a
    tolerance nobody agrees on.
    """
    return json.dumps(_normalise(dict(params)), sort_keys=True)


def _tried(history: Sequence[Round]) -> set[tuple[str, tuple[str, ...], str]]:
    """The `(operator, targets, params)` triples already attempted, failures included.

    Error rounds count: an operator that raised did not work, and retrying the
    same numbers is the definition of a wasted round.
    """
    return {
        (str(item.plan.operator), tuple(item.plan.targets), _signal(item.plan.params))
        for item in history
    }


def _spec(assembly: Assembly) -> DesignSpec | None:
    """The spec, or `None` for an assembly that is not on the design plane."""
    try:
        return spec_of(assembly)
    except KeyError:
        return None


def _design_plane_certs(certs: Sequence[FailureCertificate]) -> list[FailureCertificate]:
    """The physical evidence. An environment-family entry is a tool that could not
    run, not a defect, and must not be allowed to drive a repair."""
    return [cert for cert in certs if cert.family != _ENVIRONMENT]


def _interface_params(spec: DesignSpec, **fields: object) -> dict:
    """One interface operator's parameter payload: which interface, which parts."""
    return {
        "interface_id": spec.interface.id,
        "parts": (spec.interface.a, spec.interface.b),
        **fields,
    }


def _canonical(operator: Operator, params: Mapping[str, object]) -> dict:
    """`params` as the operator's own model records them, normalised for comparison.

    The payload is canonicalised *before* the plan is built so that the history
    signature of a plan this policy proposed equals the signature of the same plan
    read back from the store — a partial dict and a full one are the same plan.
    """
    return _normalise(operator.params.model_validate(dict(params)).model_dump())


def _plan(
    operator: Operator, params: Mapping[str, object], *, reason: str, geom_calls: int
) -> RepairPlan:
    payload = _canonical(operator, params)
    targets = tuple(operator.targets(payload))
    return RepairPlan(
        id=key("plan", operator.id, targets, payload),
        operator=operator.id,
        targets=targets,
        params=payload,
        cost_est=Cost(geom_calls=geom_calls),
        proposed_by="rule",
        reason=reason,
    )


# --------------------------------------------------------------------------- #
# Candidate builders — one per operator, each a small deterministic ladder
# --------------------------------------------------------------------------- #
#: `(spec, certificate) -> candidate parameter payloads`, in the order to try
#: them. An empty list means "this operator has nothing to say about this
#: evidence", which is different from "it has something to say that failed".
Candidates = Callable[[DesignSpec, FailureCertificate], list[dict]]


def _set_clearance(spec: DesignSpec, cert: FailureCertificate) -> list[dict]:
    target = band_target(spec.constraints, _EDGE.get(cert.code, "mid"))
    if abs(target - spec.clearance_mm) < 1e-9:
        return []  # the design is already where this rule would put it
    return [_interface_params(spec, clearance_mm=target)]


def _resize(operator_id: str) -> Candidates:
    def build(spec: DesignSpec, cert: FailureCertificate) -> list[dict]:
        kind = spec.interface.kind
        if not expressible(kind, operator_id):
            return []
        target = band_target(spec.constraints, _EDGE.get(cert.code, "mid"))
        delta = delta_for_clearance(kind, operator_id, spec.clearance_mm, target)
        if abs(delta) < 1e-9:
            return []
        return [_interface_params(spec, delta_mm=delta)]

    return build


def _shift_part(spec: DesignSpec, _cert: FailureCertificate) -> list[dict]:
    # The insert moves, never the body: the body's frame is what every other
    # number in the spec is written in, so moving it would re-interpret them all.
    return [
        {"part_id": spec.insert.id, "delta_mm": delta} for delta in _SHIFT_LADDER
    ]


def _rotate_part(spec: DesignSpec, _cert: FailureCertificate) -> list[dict]:
    return [
        {"part_id": spec.insert.id, "delta_deg": delta} for delta in _ROTATE_LADDER
    ]


def _switch_connector(spec: DesignSpec, _cert: FailureCertificate) -> list[dict]:
    kind = next_connector_kind(spec.interface.kind)
    if kind is None:
        return []
    # Empty `params`: the operator translates the current design into the new
    # kind. The policy has no business inventing a hook height from a peg
    # diameter, and the translation that does exist lives in `design_ops`.
    return [_interface_params(spec, kind=kind, params={})]


def _regenerate_interface(spec: DesignSpec, _cert: FailureCertificate) -> list[dict]:
    kind = spec.interface.kind
    if not expressible(kind, "set_clearance"):
        return []
    target = band_target(spec.constraints, "mid")
    payload = params_with_clearance(kind, spec.interface.params, target)
    if payload == spec.interface.model_for().model_validate(spec.interface.params).model_dump():
        return []
    return [_interface_params(spec, params=payload)]


_BUILDERS: dict[str, Candidates] = {
    "set_clearance": _set_clearance,
    "resize_feature": _resize("resize_feature"),
    "resize_insert": _resize("resize_insert"),
    "shift_part": _shift_part,
    "rotate_part": _rotate_part,
    "switch_connector": _switch_connector,
    "regenerate_interface": _regenerate_interface,
}


def load_ladder(path: str | Path) -> dict[str, tuple[str, ...]]:
    """A ladder override from a TOML file, merged over `DEFAULT_LADDER`.

    Shape — one row per certificate code, exactly the same vocabulary the code
    uses::

        [policy.rule.ladder]
        insufficient_clearance = ["set_clearance", "resize_feature", "switch_connector"]

    An operator id that is not one of the seven is a *load* error, not something
    to discover as a skipped candidate three rounds into a run.
    """
    data = tomllib.loads(Path(path).read_text("utf-8"))
    table = ((data.get("policy") or {}).get("rule") or {}).get("ladder") or {}
    merged = dict(DEFAULT_LADDER)
    for code, operator_ids in table.items():
        if not isinstance(operator_ids, list) or not all(
            isinstance(item, str) for item in operator_ids
        ):
            raise ValueError(
                f"policy.rule.ladder.{code} must be a list of operator ids, got "
                f"{operator_ids!r}"
            )
        unknown = sorted(set(operator_ids) - set(DESIGN_OPERATOR_IDS))
        if unknown:
            raise KeyError(
                f"policy.rule.ladder.{code} names unregistered operator(s) {unknown!r}; "
                f"the design action space is {list(DESIGN_OPERATOR_IDS)}"
            )
        merged[str(code)] = tuple(operator_ids)
    return merged


# --------------------------------------------------------------------------- #
# RulePolicy
# --------------------------------------------------------------------------- #
@dataclass
class RulePolicy:
    """Deterministic repair decisions read off the certificate codes.

    One round of `propose` picks the first candidate that is (a) legal for some
    certificate, (b) expressible on this connector kind, (c) not already
    attempted, and (d) changes the design. `None` when no such candidate exists —
    that is the policy saying "the action space is exhausted", and the loop
    records it as `infeasible`, not as a failure.
    """

    registry: OperatorCatalogue
    name: str = "rule"
    ladder: dict[str, tuple[str, ...]] = field(default_factory=lambda: dict(DEFAULT_LADDER))
    #: Declared cost of one rule-based round, for the loop's budget preview.
    geom_calls_per_round: int = 1

    def ladder_for(self, code: str) -> tuple[str, ...]:
        return tuple(self.ladder.get(code, ())) + FALLBACK_LADDER

    def propose(
        self,
        assembly: Assembly,
        certs: list[FailureCertificate],
        history: list[Round],
        budget_left: Budget,
    ) -> RepairPlan | None:
        spec = _spec(assembly)
        evidence = _design_plane_certs(certs)
        if spec is None or not evidence:
            return None

        applicable = set(self.registry.applicable(assembly, evidence))
        tried = _tried(history)
        for cert in evidence:
            for operator_id in self.ladder_for(cert.code):
                if operator_id not in applicable:
                    continue
                operator = self.registry.get(operator_id)
                builder = _BUILDERS.get(operator_id)
                if builder is None:  # a plugin operator: this policy has no rule for it
                    continue
                for params in builder(spec, cert):
                    plan = _plan(
                        operator,
                        params,
                        reason=f"{cert.code} -> {operator_id}",
                        geom_calls=self.geom_calls_per_round,
                    )
                    if (plan.operator, plan.targets, _signal(plan.params)) in tried:
                        continue
                    return plan
        return None


# --------------------------------------------------------------------------- #
# RandomPolicy
# --------------------------------------------------------------------------- #
@dataclass
class RandomPolicy:
    """The floor: same action space, same budget, no reasoning.

    Seeded, and seeded *per call* from `(seed, snapshot, round index)` rather than
    from a stream that advances inside the object: an episode resumed from the
    store must draw the same numbers it would have drawn without the interruption,
    or `--resume` would not reproduce the run (`docs/contracts/v2-interfaces.md`
    §7.2/§7.4).

    It avoids repeating an exact plan it already attempted. That is not the random
    baseline cheating: a repeated plan is a bricked round for *any* policy, so
    spending samples on one measures the loop, not the policy.
    """

    registry: OperatorCatalogue
    name: str = "random"
    seed: int = 0
    #: How many draws to spend looking for an untried, non-degenerate plan.
    samples: int = 8
    geom_calls_per_round: int = 1

    def propose(
        self,
        assembly: Assembly,
        certs: list[FailureCertificate],
        history: list[Round],
        budget_left: Budget,
    ) -> RepairPlan | None:
        spec = _spec(assembly)
        evidence = _design_plane_certs(certs)
        if spec is None or not evidence:
            return None

        legal = [
            operator_id
            for operator_id in self.registry.applicable(assembly, evidence)
            if expressible(spec.interface.kind, operator_id) and operator_id in _BUILDERS
        ]
        if not legal:
            return None

        rng = random.Random(f"{self.seed}|{assembly.id}|{len(history)}")
        tried = _tried(history)
        for _ in range(self.samples):
            operator_id = rng.choice(legal)
            params = _SAMPLE[operator_id](spec, rng)
            if params is None:
                continue
            operator = self.registry.get(operator_id)
            plan = _plan(
                operator,
                params,
                reason=f"random:{rng.random():.6f}",
                geom_calls=self.geom_calls_per_round,
            )
            if (plan.operator, plan.targets, _signal(plan.params)) in tried:
                continue
            return plan
        return None


# --------------------------------------------------------------------------- #
# Sampling — one per operator, all inside the operator's own parameter model
# --------------------------------------------------------------------------- #
#: How far a random draw may push each operator. Small on purpose: a random
#: baseline whose repairs are huge would fail for a reason that has nothing to do
#: with the loop, and the comparison would stop being about strategy.
_RESIZE_SPAN_MM = 0.6
_SHIFT_SPAN_MM = 3.0
_ROTATE_SPAN_DEG = 20.0


def _sample_clearance(spec: DesignSpec, rng: random.Random) -> dict:
    band = spec.constraints
    return _interface_params(spec, clearance_mm=round(rng.uniform(band.min_clearance_mm, band.max_clearance_mm), 3))


def _sample_resize(operator_id: str) -> Callable[[DesignSpec, random.Random], dict | None]:
    def sample(spec: DesignSpec, rng: random.Random) -> dict | None:
        kind = spec.interface.kind
        if not expressible(kind, operator_id):
            return None
        field = resize_field(kind, operator_id)[0]
        current = float(spec.interface.validated_params().model_dump()[field])
        delta = round(rng.uniform(-_RESIZE_SPAN_MM, _RESIZE_SPAN_MM), 3)
        if abs(delta) < 1e-3:
            return None  # a zero delta is not an action
        values = resize_params(kind, operator_id, spec.interface.params, delta)
        if values is None:
            return None  # the kind cannot build that; draw again
        # Report the delta that will actually be built: a clamp inside the kind's
        # bounds shortens the request, and a plan must not claim a move the design
        # did not make.
        realised = round(values[field] - current, 6)
        if abs(realised) < 1e-3:
            return None
        return _interface_params(spec, delta_mm=realised)

    return sample


def _sample_shift(spec: DesignSpec, rng: random.Random) -> dict | None:
    delta = tuple(
        round(rng.uniform(-_SHIFT_SPAN_MM, _SHIFT_SPAN_MM), 3) for _ in range(3)
    )
    if all(abs(value) < 1e-3 for value in delta):
        return None
    return {"part_id": spec.insert.id, "delta_mm": delta}


def _sample_rotate(spec: DesignSpec, rng: random.Random) -> dict | None:
    delta = tuple(
        round(rng.uniform(-_ROTATE_SPAN_DEG, _ROTATE_SPAN_DEG), 1) for _ in range(3)
    )
    if all(abs(value) < 0.1 for value in delta):
        return None
    return {"part_id": spec.insert.id, "delta_deg": delta}


def _sample_switch(spec: DesignSpec, rng: random.Random) -> dict | None:
    alternatives = [kind for kind in CONNECTOR_KINDS if kind != spec.interface.kind]
    if not alternatives:
        return None
    return _interface_params(spec, kind=rng.choice(alternatives), params={})


def _sample_regenerate(spec: DesignSpec, rng: random.Random) -> dict | None:
    kind = spec.interface.kind
    if not expressible(kind, "set_clearance"):
        return None
    payload = params_with_clearance(
        kind, spec.interface.params, band_target(spec.constraints, "mid")
    )
    # Engagement is the one lever a specific rule never touches, so a draw here
    # explores a part of the action space the rule ladder structurally cannot.
    payload["engage_mm"] = round(max(0.5, float(payload["engage_mm"]) + rng.uniform(0.5, 2.0)), 3)
    return _interface_params(spec, params=payload)


_SAMPLE: dict[str, Callable[[DesignSpec, random.Random], dict | None]] = {
    "set_clearance": _sample_clearance,
    "resize_feature": _sample_resize("resize_feature"),
    "resize_insert": _sample_resize("resize_insert"),
    "shift_part": _sample_shift,
    "rotate_part": _sample_rotate,
    "switch_connector": _sample_switch,
    "regenerate_interface": _sample_regenerate,
}


# --------------------------------------------------------------------------- #
# The one factory a composition root calls
# --------------------------------------------------------------------------- #
def build(
    name: str,
    *,
    registry: OperatorCatalogue,
    config_path: str | Path | None = None,
    seed: int = 0,
    llm=None,
):
    """`name` -> a `LoopPolicy`: `rule`, `random`, or `llm`.

    One entry point so a CLI or an eval never branches on policy names itself, and
    so a wrong name fails at construction instead of three rounds in.

    `llm` is the provider's `LLMClient` (`app.providers`), never constructed here:
    a policy that could reach the network on its own would make the manifest's
    provider record a lie. `name="llm"` with no client is a configuration error,
    not a silent `NullLLM` — a policy that always abstains would look like a model
    that found nothing to do.
    """
    if name == "rule":
        ladder = load_ladder(config_path) if config_path is not None else dict(DEFAULT_LADDER)
        return RulePolicy(registry=registry, ladder=ladder)
    if name == "random":
        return RandomPolicy(registry=registry, seed=seed)
    if name == "llm":
        from pah.app.providers import ProviderError  # local: keeps this module import-light

        if llm is None:
            raise ProviderError(
                "policy 'llm' needs a client: pass the provider's LLMClient (or a "
                "ReplayLLM). A missing client is a configuration error, not an abstention."
            )
        from pah.policy.llm import LlmPolicy

        return LlmPolicy(llm, registry=registry, system_prompt=_prompt_override(config_path))
    raise ValueError(
        f"unknown policy {name!r}; the design plane ships {POLICY_NAMES}"
    )


def _prompt_override(config_path: str | Path | None) -> str | None:
    """A system prompt from a config file, if one is given.

    Kept separate from the ladder loader because the ladder is a *rule* concept
    and this is an *LLM* one; one TOML file may carry both sections, and each
    policy reads only its own.
    """
    if config_path is None:
        return None
    data = tomllib.loads(Path(config_path).read_text("utf-8"))
    table = (data.get("policy") or {}).get("llm") or {}
    prompt = table.get("system_prompt_file")
    if not prompt:
        return None
    return Path(prompt).read_text("utf-8")
