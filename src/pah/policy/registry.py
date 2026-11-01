"""The operator registry: the closed action space a repair policy may choose from.

`docs/internal/repair-loop-spec.md` §2.2 fixes the shape; this module supplies the
guards the spec only implies, and the M1 ladder's three parameter operators as the
first entries.

Four promises, in the order they can fail
-----------------------------------------
1. **An unknown operator is a `KeyError`.** A policy that names an operator nobody
   registered did not invent an action — it tried to. Refusing it here is the code
   behind "the agent cannot invent an action".
2. **Parameters are validated by the operator's own pydantic model**, with
   `extra="forbid"`: a plan cannot smuggle a field the operator never reads, and a
   malformed plan raises `pydantic.ValidationError` at the boundary instead of
   inside a geometry call. The exception is **not** caught here: the engine turns it
   into a `UserError` (spec §4), because only the engine may decide what a rejected
   plan means.
3. **The targets a plan declares must equal the targets its parameters compute.**
   This is the anti-spoof check. Without it a plan could declare
   `targets=("p1",)` and then carry parameters that rewrite `p4`; `targets()` is a
   pure function of the *validated* parameters, so the comparison is over facts.
4. **A builtin re-checks the joint it is about to edit** against those same targets
   (`_joint_for`). Check 3 proves the plan is self-consistent, not that the assembly
   agrees: parameters could name joint `j6` and parts `p1,p6` while `j6` in fact
   joins `p1,p7`. One lookup closes that gap.

`applicable` is the enumeration half: an operator is applicable when **all** of its
preconditions hold for **at least one** certificate. No evidence, no action — an
empty `certs` returns `[]` rather than "anything is possible". The result is sorted
by operator id, so two calls with the same evidence return the same list in the
same order regardless of registration order.

What the builtin operators edit, and what they do not
-----------------------------------------------------
They edit one record of `Assembly.joints` — the `case.json` joint shape built by
`connector_feasible._joint` (`id`, `a`, `b`, `anchor_mm`, `n_ab`, `t_b_mm`,
`depth_max_mm`, `r_inplane_max_mm`), lengths in millimetres throughout. What they
produce is the *connector request* for that joint, kept under `record["connector"]`
(`r_mm`, `depth_mm`, `section`): exactly the `(r, depth, section)` triple
`connector.build` is called with, and exactly what `repair_script` already computes.

They do **not** build geometry. `connector.build` costs seconds to minutes and
belongs to the tool executor; the registry materialises the design change that call
will be made with, as a new `Assembly` version (deterministic id, `parent` pointing
back — `pah.core.ids`). Every number is reused from `repair_script`
(`closed_form_radius`, `gate_margin`, `_plane_basis`, `_POST_GATE_CODES`), never
re-derived: the module docstring there explains why a re-derived copy of the wall
cap or of the plane basis is a silent, expensive bug.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from pah.core import ids
from pah.core.model import Assembly, FailureCertificate, RepairPlan
from pah.policy import repair_script
from pah.tools.verifiers.connector_feasible import R_MIN_MM, SECTION

__all__ = [
    "Operator",
    "OperatorRegistry",
    "RelocateConnectorParams",
    "ResizeConnectorParams",
    "SwapFamilyParams",
    "register_builtin",
]


@dataclass(frozen=True)
class Operator:
    """One action a policy may propose: what it needs, what it touches, what it costs.

    `preconditions` are checked against `(assembly, certificate)` pairs, so a rule
    can depend on the evidence *and* on the snapshot it is evidence about.
    `targets` is the only source of truth for which parts a plan touches, and it
    reads validated parameters only — never the assembly, never the plan.
    """

    id: str
    params: type[BaseModel]
    preconditions: tuple[Callable[[Assembly, FailureCertificate], bool], ...]
    apply: Callable[[Assembly, dict], Assembly]
    targets: Callable[[dict], tuple[str, ...]]
    cost: Literal["cheap", "medium", "expensive"]
    family: Literal["geometry", "interface", "connector", "generative"]


class OperatorRegistry:
    """The registered action space. Build one per process with `register_builtin`."""

    def __init__(self) -> None:
        self._operators: dict[str, Operator] = {}

    # -- construction ------------------------------------------------------

    def register(self, operator: Operator) -> None:
        """Adds one operator. Re-registering an id is an error, not an update: a
        policy's action space must not change under it, and a silent replacement
        would make two traces with the same policy version disagree."""
        if operator.id in self._operators:
            raise ValueError(f"operator {operator.id!r} is already registered")
        if not isinstance(operator.params, type) or not issubclass(
            operator.params, BaseModel
        ):
            raise TypeError(
                f"operator {operator.id!r} must declare a pydantic model as `params`"
            )
        self._operators[operator.id] = operator

    def get(self, operator_id: str) -> Operator:
        """The registered operator. A missing id raises `KeyError` by design."""
        return self._operators[operator_id]

    def __contains__(self, operator_id: object) -> bool:
        return operator_id in self._operators

    # -- enumeration -------------------------------------------------------

    def applicable(
        self, asm: Assembly, certs: Iterable[FailureCertificate]
    ) -> list[str]:
        """Ids of the operators some certificate makes possible, sorted by id.

        "Applicable" means `all(preconditions)` for one certificate: a rule that
        needs two conditions must have both, and it may satisfy them on the same
        certificate. An operator with no preconditions is applicable to any
        evidence — the empty conjunction is true.
        """
        certificates = tuple(certs)
        if not certificates:
            return []
        return sorted(
            operator_id
            for operator_id, operator in self._operators.items()
            if any(
                all(pre(asm, cert) for pre in operator.preconditions)
                for cert in certificates
            )
        )

    # -- the guarded apply -------------------------------------------------

    def apply(self, asm: Assembly, plan: RepairPlan) -> Assembly:
        """`plan` materialised as a new `Assembly` version, or an exception.

        Three checks stand between a plan and the geometry: the operator must be
        registered, the parameters must satisfy its model, and the targets its
        parameters compute must equal the ones the plan declares.
        """
        operator = self.get(plan.operator)  # 1: KeyError on an invented action
        params = operator.params.model_validate(plan.params)  # 2: ValidationError
        computed = tuple(operator.targets(params.model_dump()))
        declared = tuple(plan.targets)
        if set(computed) != set(declared):
            raise ValueError(
                f"plan {plan.id!r} declares targets {declared!r}, but operator "
                f"{operator.id!r} computes {computed!r} from its parameters"
            )
        return operator.apply(asm, params.model_dump())


# --------------------------------------------------------------------------- #
# Parameters — one model per operator, so the plan's payload is typed at the edge
# --------------------------------------------------------------------------- #
class _JointParams(BaseModel):
    """The two fields every builtin needs: which joint, and which parts it touches.

    `parts` is the pair the joint connects. It is a parameter rather than something
    read from the assembly because `targets()` must be a pure function of the
    parameters: that is what makes checks 3 and 4 independent of each other.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    joint_id: str
    parts: tuple[str, ...]


class ResizeConnectorParams(_JointParams):
    """A closed-form radius for one joint (`repair_script.closed_form_radius`).

    `gap_mm` is measured evidence, not a chosen number: it is the `value` of the
    `gap_too_wide` certificate. The caps default to the joint's own record
    (`depth_max_mm`, `r_inplane_max_mm`) and may be overridden when a caller has a
    better measurement — the `d_cap` local-wall override is exactly that case.
    """

    gap_mm: float = Field(ge=0.0)
    t_b_mm: float | None = Field(default=None, gt=0.0)
    d_cap_mm: float | None = Field(default=None, gt=0.0)
    r_cap_mm: float | None = Field(default=None, gt=0.0)
    r_min_mm: float = Field(default=R_MIN_MM, gt=0.0)


class RelocateConnectorParams(_JointParams):
    """An in-plane move of the joint's anchor, `(du_mm, dv_mm)` off the recorded one.

    The two offsets are in the plane spanned by `repair_script._plane_basis`, built
    from the joint's `n_ab`. That basis is reused rather than re-derived: the M1
    pass that copied it by hand placed the pin somewhere the ray had not measured,
    and the gate self-calibration reported it as 103 false positives.
    """

    du_mm: float
    dv_mm: float


class SwapFamilyParams(_JointParams):
    """The connector family at this joint — the only lever that changes what the
    boolean removes once the pin and socket were actually built."""

    section: Literal["circle", "D"] = SECTION


# --------------------------------------------------------------------------- #
# Preconditions — pure predicates over (assembly, certificate)
# --------------------------------------------------------------------------- #
def _fires_on(*codes: str) -> Callable[[Assembly, FailureCertificate], bool]:
    """A precondition that accepts the certificate codes one operator can help."""
    allowed = frozenset(codes)

    def check(asm: Assembly, cert: FailureCertificate) -> bool:
        return cert.code in allowed

    return check


def _joint_is_in_the_assembly(asm: Assembly, cert: FailureCertificate) -> bool:
    """A certificate about a joint this assembly does not contain is stale evidence.

    The locus is where `connector_feasible` records the joint it judged
    (`locus["joint"]`), so this is a lookup, not a guess.
    """
    joint_id = str((cert.locus or {}).get("joint") or "")
    if not joint_id:
        return False
    return any(str(joint.get("id")) == joint_id for joint in asm.joints)


# --------------------------------------------------------------------------- #
# Shared apply plumbing
# --------------------------------------------------------------------------- #
def _targets(params: dict) -> tuple[str, ...]:
    """The parts the joint connects — the only targets any builtin declares."""
    return tuple(params["parts"])


def _joint_for(asm: Assembly, params: dict) -> dict:
    """The joint record to edit, after re-checking it against the plan's targets.

    Check 3 in `OperatorRegistry.apply` proves the plan agrees with itself. This
    proves the assembly agrees with the plan; without it, a plan for `j6` could
    carry the targets of `j7` and both checks would still pass.
    """
    joint_id = str(params["joint_id"])
    joint = next((j for j in asm.joints if str(j.get("id")) == joint_id), None)
    if joint is None:
        raise KeyError(f"assembly {asm.id!r} has no joint {joint_id!r}")
    joined = (joint.get("a"), joint.get("b"))
    if joined[0] is None or joined[1] is None:
        raise ValueError(
            f"joint {joint_id!r} records no 'a'/'b' parts, so its targets cannot be "
            "confirmed against the plan"
        )
    declared = tuple(str(part) for part in params["parts"])
    if set(declared) != {str(joined[0]), str(joined[1])}:
        raise ValueError(
            f"joint {joint_id!r} connects {joined!r}, not the plan's targets {declared!r}"
        )
    return joint


def _request(joint: dict) -> dict:
    """The connector request already on the record, if any."""
    return dict(joint.get("connector") or {})


def _number(explicit: float | None, joint: dict, field: str) -> float:
    """A parameter with the joint record as its fallback.

    Absent everywhere is `0.0`, which every `repair_script` bound reads as "no such
    limit" — the same convention `repair_edge` uses for `d_cap`.
    """
    if explicit is not None:
        return float(explicit)
    try:
        return float(joint.get(field) or 0.0)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"joint {joint.get('id')!r} has a non-numeric {field!r}"
        ) from error


def _length(value: float) -> float:
    """A millimetre value as it is written to a record.

    Rounded to the six decimals every M1 report uses, and `+ 0.0` so `-0.0` becomes
    `0.0`: the same anchor must have one spelling in the trace and one id.
    """
    return round(float(value), 6) + 0.0


def _rewrite(asm: Assembly, joint: dict, patch: dict) -> Assembly:
    """A new version with one joint's record replaced by `joint | patch`.

    The id is derived from the facts that make this version — base id, new version,
    joint, patch — so applying the same plan to the same assembly twice yields the
    same id, and two different plans cannot collide on one.
    """
    joint_id = str(joint["id"])
    record = {**joint, **patch}
    joints = tuple(record if str(j.get("id")) == joint_id else j for j in asm.joints)
    version = asm.version + 1
    return replace(
        asm,
        id=ids.key("assembly", asm.id, version, joint_id, patch),
        version=version,
        joints=joints,
        parent=asm.id,
    )


# --------------------------------------------------------------------------- #
# The three builtins
# --------------------------------------------------------------------------- #
def _apply_resize(asm: Assembly, params: dict) -> Assembly:
    """Write the largest radius that clears the wall cap and the depth cap.

    The arithmetic is `repair_script.closed_form_radius`, the same call
    `_plan_resize` makes, and the depth rule is `_plan_resize`'s
    `min(2r, d_cap)`: shortening the pin is never a repair, it is a different pin.
    """
    joint = _joint_for(asm, params)
    gap = float(params["gap_mm"])
    t_b = _number(params.get("t_b_mm"), joint, "t_b_mm")
    d_cap = _number(params.get("d_cap_mm"), joint, "depth_max_mm")
    r_cap = _number(params.get("r_cap_mm"), joint, "r_inplane_max_mm")
    r = repair_script.closed_form_radius(gap, t_b, d_cap, float(params["r_min_mm"]), r_cap)
    depth = min(2.0 * r, d_cap) if d_cap > 0 else 2.0 * r
    request = {**_request(joint), "r_mm": _length(r), "depth_mm": _length(depth)}
    return _rewrite(asm, joint, {"connector": request})


def _apply_relocate(asm: Assembly, params: dict) -> Assembly:
    """Move the anchor by `(du_mm, dv_mm)` in the joint's own plane.

    Same construction as `repair_script._plan_relocate`: the new world point is
    `base + du * e1 + dv * e2` with `(e1, e2)` from `_plane_basis(n_ab)`, so the
    position measured is the position built.
    """
    joint = _joint_for(asm, params)
    normal = [float(x) for x in (joint.get("n_ab") or ())]
    base = [float(x) for x in (joint.get("anchor_mm") or ())]
    if len(normal) != 3 or len(base) != 3:
        raise ValueError(
            f"joint {joint.get('id')!r} needs a 3-vector 'n_ab' and 'anchor_mm' "
            "in millimetres before it can be relocated"
        )
    e1, e2 = repair_script._plane_basis(normal)
    du, dv = float(params["du_mm"]), float(params["dv_mm"])
    anchor = [_length(base[i] + du * e1[i] + dv * e2[i]) for i in range(3)]
    return _rewrite(asm, joint, {"anchor_mm": anchor})


def _apply_swap(asm: Assembly, params: dict) -> Assembly:
    """Change the connector section at the current radius and position."""
    joint = _joint_for(asm, params)
    request = {**_request(joint), "section": params["section"]}
    return _rewrite(asm, joint, {"connector": request})


#: Post-gate codes: the pin and socket were built, so the family is the next lever.
#: Reused from `repair_script` instead of re-listed — a mirrored copy of this set is
#: exactly the drift that module's own docstring warns about.
_POST_GATE_CODES = repair_script._POST_GATE_CODES

_RESIZE_CONNECTOR = Operator(
    id="resize_connector",
    params=ResizeConnectorParams,
    preconditions=(_fires_on("gap_too_wide"), _joint_is_in_the_assembly),
    apply=_apply_resize,
    targets=_targets,
    cost="cheap",
    family="connector",
)

#: `relocate_connector` is the first rung for `no_material_hit` (no radius can
#: change a ray that hits nothing) and the second for `gap_too_wide` (a gap wider
#: than `d_cap` is only fixable by measuring somewhere else).
_RELOCATE_CONNECTOR = Operator(
    id="relocate_connector",
    params=RelocateConnectorParams,
    preconditions=(
        _fires_on("gap_too_wide", "no_material_hit"),
        _joint_is_in_the_assembly,
    ),
    apply=_apply_relocate,
    targets=_targets,
    cost="cheap",
    family="connector",
)

_SWAP_FAMILY = Operator(
    id="swap_family",
    params=SwapFamilyParams,
    preconditions=(_fires_on(*sorted(_POST_GATE_CODES)), _joint_is_in_the_assembly),
    apply=_apply_swap,
    targets=_targets,
    cost="cheap",
    family="connector",
)


def register_builtin(registry: OperatorRegistry) -> None:
    """Registers the M1 ladder's three parameter operators, as `repair_script` names them.

    This is where the ladder becomes an action space: `repair_script` still owns the
    arithmetic and the ordering (`next_operator`), while the registry owns what may
    be proposed and what a proposal must prove. Nothing here chooses an operator —
    that is the policy's job (§2.3), and keeping it out is what lets a rule policy
    and an LLM policy share one validated space.
    """
    for operator in (_RESIZE_CONNECTOR, _RELOCATE_CONNECTOR, _SWAP_FAMILY):
        registry.register(operator)
