"""The design plane's action space: seven operators that move *numbers*, never vertices.

Why an operator table instead of a geometry generator
-----------------------------------------------------
The v1 action space (`policy/registry.py`) edited one record of
`Assembly.joints`: a connector request on geometry the dataset already shipped.
Its ceiling was the dataset's — every repair was a better way to place a pin in
somebody else's mesh.

The design-plane operators edit a `DesignSpec` (`design/spec.py`) instead: the
interface's kind, its parameters, a part's pose. A spec edit is a *decision*, so
an agent can be held to it ("the hole became 10.5 mm, and 10.5 = 10.0 + 2·0.25").
That is `docs/contracts/v2-interfaces.md` §4, and the reason the LLM never gets
to name a vertex: there is no parameter in this table that *is* a vertex.

**An operator is pure.** `apply(assembly, params)` reads the spec off the
snapshot, edits it, and returns `assembly_from_spec(...)` with `parent` pointing
back and `version + 1`. It writes no file, builds no mesh, and does not know a
`Materializer` exists. That is a deliberate change from this module's first draft
(Lead, 2026-11-01): it lets an operator be tested without a run directory, it
keeps geometry construction in the composition root and the cascade adapter
where the verifiers can see it, and it makes the operator's output a function of
its input and nothing else. The geometry *identity* still moves with the numbers
because `assembly_from_spec` fills `geom.hash` from
`design.api.geometry_hash(spec, part_id)`; `geom.path` stays empty, which is the
signal that the cascade owes this snapshot a materialisation before anyone may
ask it for triangles.

The four gates, and the fifth one this module adds
--------------------------------------------------
`OperatorRegistry.apply` already refuses an unregistered operator (1), a
parameter set that fails the operator's own pydantic model (2), and a plan whose
declared targets disagree with the targets its parameters compute (3).
`registry._joint_for` adds the fourth check — that the joint in the assembly is
the joint the plan names — and `_checked_spec` here is the design-plane version
of it: the interface id and part pair the plan names must be the ones the spec
actually has, or the plan is about a design that is not in front of us.

`targets()` is a pure function of the validated parameters, which is why
`parts` is a *parameter* of every interface operator rather than something read
off the spec. `registry._JointParams` makes the same trade for the same reason:
if `targets()` read the assembly, check (3) would prove nothing.

Where each number comes from
----------------------------
Clearance is the one number the design plane argues about, and each kind spells
it differently: `(d_hole - d_peg)/2` for a peg, `(slot_w - hook_w)/2` for a snap
hook, a per-side `clearance_mm` for a dovetail. `spec.radial_clearance_mm` owns
the read direction. This module owns the *write* direction, in exactly one place
(`params_with_clearance`), because a second copy of that arithmetic is how two
modules start disagreeing about the same design — the failure mode
`design/spec.py` itself warns about for the read direction.

The consequence, stated once: growing the *cavity* opens the fit, growing the
*insert* closes it, and one millimetre of field buys half a millimetre of radial
clearance for the two round kinds but a whole one for the dovetail. `_RESIZE`
holds both facts together so the policy can invert them without re-deriving
them. `dovetail` exposes no insert-side number at all, so `resize_insert` is not
expressible there; `expressible` says so *before* a policy proposes it, instead
of the executor discovering it as an error round.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from pah.core.model import Assembly, FailureCertificate
from pah.design.api import (
    CODE_DYNAMIC_PENETRATION,
    CODE_EXCESSIVE_CLEARANCE,
    CODE_GEOMETRIC_INTERFERENCE,
    CODE_INSUFFICIENT_CLEARANCE,
    CODE_INSERTION_PATH_BLOCKED,
    CODE_JOINT_UNSTABLE,
    CODE_PHYSICS_ERROR,
    DESIGN_CODES,
    assembly_from_spec,
    spec_of,
)
from pah.design.spec import (
    DesignSpec,
    Vec3,
    builtin_params_model,
)
from pah.design.connectors import builder_for
from pah.policy.registry import Operator

__all__ = [
    "CONNECTOR_KINDS",
    "HARD_CODES",
    "RegenerateInterfaceParams",
    "ResizeFeatureParams",
    "ResizeInsertParams",
    "RotatePartParams",
    "SetClearanceParams",
    "ShiftPartParams",
    "SwitchConnectorParams",
    "delta_for_clearance",
    "expressible",
    "make_design_operators",
    "next_connector_kind",
    "params_with_clearance",
    "register_builtin",
    "register_design_operators",
    "resize_field",
    "resize_params",
    "switch_params",
]

# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #
#: The connector kinds this module's arithmetic understands, in a fixed order so
#: `switch_connector` picks the same alternative on every run.
CONNECTOR_KINDS: tuple[str, ...] = ("peg_hole", "snap_fit", "dovetail")

#: Every design-plane code except the environment one. `hard_failures` drops
#: `physics_error` before a policy ever sees it, so a policy's evidence is exactly
#: this set — derived here rather than copied, so the table and this tuple cannot
#: drift apart.
HARD_CODES: tuple[str, ...] = tuple(
    sorted(
        code
        for codes in DESIGN_CODES.values()
        for code in codes
        if code != CODE_PHYSICS_ERROR
    )
)

_CLEARANCE_CODES = (CODE_INSUFFICIENT_CLEARANCE, CODE_EXCESSIVE_CLEARANCE)


# --------------------------------------------------------------------------- #
# The clearance arithmetic — one function each way, and no second copy
# --------------------------------------------------------------------------- #
def _validated(kind: str, params: Mapping[str, object]) -> dict:
    """`params` as the kind's own model, fully populated.

    Partial dicts are legal on an `InterfaceSpec` (the model fills defaults at
    validation time), so every write below starts from the *full* set of numbers
    rather than patching a dict that may not carry the one being read.
    """
    model = builtin_params_model(kind)
    if model is None:
        raise KeyError(
            f"connector kind {kind!r} is not builtin; a plugin kind reaches the design "
            "operators only through `regenerate_interface` or `switch_connector`"
        )
    return model.model_validate(dict(params)).model_dump()


def _clamp_params(kind: str, params: Mapping[str, object]) -> dict | None:
    """`params` moved inside the kind's own declared bounds, or `None` if it cannot be.

    The bounds come from the parameter model's `Field(ge=…)` / `le=…` / `gt=…` /
    `lt=…` metadata rather than being restated here, so tightening a parameter
    model tightens the action space with no operator change.

    Inclusive bounds are clamped, exclusive ones are refused, and the difference is
    not cosmetic:

    * ``dovetail.clearance_mm >= 0`` is inclusive and 0 is a real dovetail — the
      tightest one the geometry can build. A target below it is a design the kind
      cannot *be*, so the closest buildable intent is 0.
    * ``d_peg_mm > 0`` is exclusive. Clamping to just under the bound builds a peg
      of diameter 5e-324: legal in the model, nonsense as a part. A request that
      would cross it is a mistake to refuse, not a number to re-interpret.
    """
    model = builtin_params_model(kind)
    if model is None:
        raise KeyError(f"connector kind {kind!r} is not builtin")
    values = dict(params)
    for name, field in model.model_fields.items():
        value = values.get(name)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        for bound in field.metadata:
            if hasattr(bound, "ge") and value < bound.ge:
                values[name] = bound.ge
                value = bound.ge
            elif hasattr(bound, "le") and value > bound.le:
                values[name] = bound.le
                value = bound.le
            elif hasattr(bound, "gt") and value <= bound.gt:
                return None
            elif hasattr(bound, "lt") and value >= bound.lt:
                return None
    return values


def params_with_clearance(
    kind: str, params: Mapping[str, object], clearance_mm: float
) -> dict:
    """The interface parameters that realise a target *radial* clearance.

    The one place this arithmetic lives, written out per kind rather than
    disguised as a scale factor, because the three lines are the design fact:

    * ``peg_hole``: ``d_hole = d_peg + 2c`` — clearance is half the diametral gap;
    * ``snap_fit``: ``slot_w = hook_w + 2c`` — the same halving, on the slot;
    * ``dovetail``: ``clearance_mm = c`` — already per side.

    A negative target is honoured for the two round kinds: an interference fit is a
    legal design and judging it is a verifier's job, not this module's. `dovetail`
    cannot express one (`_clamp_params`), so it lands at the tightest buildable
    dovetail instead.
    """
    values = _validated(kind, params)
    if kind == "peg_hole":
        values["d_hole_mm"] = _mm(float(values["d_peg_mm"]) + 2.0 * clearance_mm)
    elif kind == "snap_fit":
        values["slot_w_mm"] = _mm(float(values["hook_w_mm"]) + 2.0 * clearance_mm)
    elif kind == "dovetail":
        values["clearance_mm"] = _mm(clearance_mm)
    else:
        raise KeyError(f"no clearance arithmetic for connector kind {kind!r}")
    clamped = _clamp_params(kind, values)
    if clamped is None:
        raise ValueError(
            f"a radial clearance of {clearance_mm} mm is not expressible with connector "
            f"kind {kind!r}"
        )
    return clamped


#: Where each resize operator writes, and how much radial clearance one
#: millimetre of that field buys. Signed: growing the insert *closes* the fit,
#: which is why `resize_insert` is the repair for "too much clearance" and
#: `resize_feature` is the repair for "too little".
#:
#: `dovetail` appears once. The kind has one number per side and no separate
#: tongue/slot dimension, so there is no insert-side field to resize — see
#: `expressible`.
_RESIZE: dict[tuple[str, str], tuple[str, float]] = {
    ("peg_hole", "resize_feature"): ("d_hole_mm", +0.5),
    ("peg_hole", "resize_insert"): ("d_peg_mm", -0.5),
    ("snap_fit", "resize_feature"): ("slot_w_mm", +0.5),
    ("snap_fit", "resize_insert"): ("hook_w_mm", -0.5),
    ("dovetail", "resize_feature"): ("clearance_mm", +1.0),
}

#: The one dimension each kind calls "the size of the thing that goes in". Used
#: by `switch_connector` so a 10 mm peg becomes a 10 mm hook rather than a
#: default-sized one: the paradigm may change, the part that has to fit may not.
_SIZE_FIELD: dict[str, str] = {
    "peg_hole": "d_peg_mm",
    "snap_fit": "hook_w_mm",
    "dovetail": "width_mm",
}


def resize_field(kind: str, operator_id: str) -> tuple[str, float]:
    """`(field, clearance gained per +1 mm of it)` for one resize operator."""
    try:
        return _RESIZE[(kind, operator_id)]
    except KeyError:
        raise KeyError(
            f"connector kind {kind!r} exposes no field for {operator_id!r}; "
            f"expressible kinds are {sorted(k for k, _ in _RESIZE)}"
        ) from None


def delta_for_clearance(
    kind: str, operator_id: str, current_mm: float, target_mm: float
) -> float:
    """How far to move the operator's own field to take clearance to `target_mm`.

    The inverse of `apply`'s ``field += delta_mm``, which is why both sides read
    the same `_RESIZE` row: a rule that computed the delta from a second copy of
    the factor would be one refactor away from proposing a repair that moves the
    clearance the wrong way.
    """
    _field, gain = resize_field(kind, operator_id)
    return _mm((target_mm - current_mm) / gain)


def resize_params(
    kind: str, operator_id: str, params: Mapping[str, object], delta_mm: float
) -> dict | None:
    """The interface parameters after a resize, or `None` if the kind cannot build it.

    One function for both callers that need this question answered — the operator
    (`_resize_edit` raises) and a policy that must not propose an unbuildable
    design (it resamples). Same arithmetic either way, clamped into the kind's
    bounds by `_clamp_params`.
    """
    field, _gain = resize_field(kind, operator_id)
    values = _validated(kind, params)
    values[field] = _mm(float(values[field]) + float(delta_mm))
    return _clamp_params(kind, values)


def expressible(kind: str, operator_id: str) -> bool:
    """Whether `operator_id` has a number to write on this connector kind.

    `OperatorRegistry.applicable` answers "may this operator run on this
    evidence?" and cannot answer this: its preconditions never see the
    parameters, so they cannot know which kind the design is. Asking here keeps
    an inexpressible action out of the policy's candidate list rather than
    letting it fail as an error round that costs a budget unit and teaches
    nothing.
    """
    if operator_id in ("resize_feature", "resize_insert"):
        return (kind, operator_id) in _RESIZE
    if operator_id == "set_clearance":
        return builtin_params_model(kind) is not None
    return True


def next_connector_kind(current: str) -> str | None:
    """The next builtin kind, or `None` when there is nothing to switch to."""
    for kind in CONNECTOR_KINDS:
        if kind != current:
            return kind
    return None


# --------------------------------------------------------------------------- #
# Parameters — one strict model per operator, so a plan's payload is typed at the edge
# --------------------------------------------------------------------------- #
class _Strict(BaseModel):
    """`extra="forbid"` is the rule that makes the LLM schema meaningful: a plan
    cannot smuggle a field the operator never reads, and the failure is a
    `ValidationError` at the boundary instead of a surprise inside a geometry call."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class _InterfaceParams(_Strict):
    """Which interface, and which parts it joins.

    `parts` is a parameter and not a lookup because `targets()` must be a pure
    function of the parameters (module docstring; `registry._JointParams` made
    the same trade). Its type is a fixed pair: this plane's interfaces join
    exactly two parts, and a three-element `parts` would be a plan about a design
    that does not exist.
    """

    interface_id: str = Field(min_length=1)
    parts: tuple[str, str]


class SetClearanceParams(_InterfaceParams):
    """A target radial clearance in millimetres — the design intent, not a dimension."""

    clearance_mm: float = Field(ge=0.0)


class ResizeFeatureParams(_InterfaceParams):
    """Grow (positive) or shrink (negative) the body-side cavity, in millimetres."""

    delta_mm: float


class ResizeInsertParams(_InterfaceParams):
    """Grow (positive) or shrink (negative) the insert-side feature, in millimetres."""

    delta_mm: float


class ShiftPartParams(_Strict):
    """Translate one part by a world-frame offset in millimetres."""

    part_id: str = Field(min_length=1)
    delta_mm: Vec3


class RotatePartParams(_Strict):
    """Rotate one part by an extrinsic XYZ delta in degrees."""

    part_id: str = Field(min_length=1)
    delta_deg: Vec3


class SwitchConnectorParams(_InterfaceParams):
    """Change the connector kind — the only lever that changes *what the fit is*.

    `params` empty means "translate the current design into the new kind" (the
    operator does that arithmetic from the spec it was handed); a full parameter
    set means the caller has already decided the numbers and they are used
    verbatim, subject to the new kind's own model.
    """

    kind: str = Field(min_length=1)
    params: dict = Field(default_factory=dict)


class RegenerateInterfaceParams(_InterfaceParams):
    """Re-generate the interface from its intent — a *partial* parameter patch.

    In v1 this is "write these numbers": the fields are merged into the current
    ones and validated as the interface's own model. It is the slot a DiT adapter
    fills later, and it is deliberately the widest operator (it can change
    engagement, approach, feature placement) so that "the paradigm is right but
    its proportions are wrong" has somewhere to go.

    `prompt_hint` is how a *decision* reaches a *generative model* without the model
    ever seeing a vertex: when a part comes back hollow, no number in the design is
    wrong, so the only honest action is "generate it again, but ask for a solid".
    The hint is carried through `spec.provenance` to whichever adapter builds the
    geometry (`examples/hunyuan3d_materializer.py` appends it to its prompt), and it is
    allowed on its own — a regeneration that changes no number is exactly this case.
    """

    params: dict = Field(default_factory=dict)
    prompt_hint: str | None = Field(default=None, max_length=600)


# --------------------------------------------------------------------------- #
# Targets — pure functions of the validated parameters, nothing else
# --------------------------------------------------------------------------- #
def _interface_targets(params: dict) -> tuple[str, ...]:
    return tuple(params["parts"])


def _part_targets(params: dict) -> tuple[str, ...]:
    return (str(params["part_id"]),)


# --------------------------------------------------------------------------- #
# Preconditions
# --------------------------------------------------------------------------- #
def _fires_on(*codes: str) -> Callable[[Assembly, FailureCertificate], bool]:
    """A precondition that accepts the certificate codes one operator can help."""
    allowed = frozenset(codes)

    def check(_asm: Assembly, cert: FailureCertificate) -> bool:
        return cert.code in allowed

    return check


def _is_design_plane(asm: Assembly, _cert: FailureCertificate) -> bool:
    """A design-plane certificate is about a design-plane assembly.

    v1 has exactly one interface per spec, so "this assembly carries a spec" is
    already enough to know the certificate is about that interface. When N
    interfaces arrive this must start reading `cert.locus["interface"]`; reading
    it now would be a dependence on a field the verifiers are free not to write.
    """
    try:
        spec_of(asm)
    except KeyError:
        return False
    return True


# --------------------------------------------------------------------------- #
# Spec edits — the domain half of each operator, pure DesignSpec -> DesignSpec
# --------------------------------------------------------------------------- #
SpecEdit = Callable[[DesignSpec, dict], DesignSpec]


def _checked_spec(asm: Assembly, params: dict) -> DesignSpec:
    """The spec to edit, after proving it is the one the plan names.

    `OperatorRegistry.apply` proved the plan agrees with itself. This proves the
    assembly agrees with the plan — the design-plane form of `registry._joint_for`.
    """
    spec = spec_of(asm)
    iface = spec.interface
    declared_interface = str(params.get("interface_id") or "")
    if declared_interface and declared_interface != iface.id:
        raise KeyError(
            f"assembly {asm.id!r} has interface {iface.id!r}, not {declared_interface!r}"
        )
    declared_parts = tuple(str(part) for part in params.get("parts") or ())
    if declared_parts and set(declared_parts) != {iface.a, iface.b}:
        raise ValueError(
            f"interface {iface.id!r} joins {(iface.a, iface.b)!r}, not the plan's "
            f"targets {declared_parts!r}"
        )
    part_id = str(params.get("part_id") or "")
    if part_id and part_id not in {part.id for part in spec.parts}:
        raise KeyError(f"assembly {asm.id!r} has no part {part_id!r}")
    return spec


def _edit_set_clearance(spec: DesignSpec, params: dict) -> DesignSpec:
    """Write the target clearance into whichever field the kind spells it with."""
    iface = spec.interface
    return spec.with_interface(
        params=params_with_clearance(
            iface.kind, iface.params, float(params["clearance_mm"])
        )
    )


def _resize_edit(operator_id: str) -> SpecEdit:
    def edit(spec: DesignSpec, params: dict) -> DesignSpec:
        iface = spec.interface
        values = resize_params(
            iface.kind, operator_id, iface.params, float(params["delta_mm"])
        )
        if values is None:
            field = resize_field(iface.kind, operator_id)[0]
            raise ValueError(
                f"{operator_id} by {params['delta_mm']} mm would move {field!r} outside "
                f"what connector kind {iface.kind!r} allows, so it does not describe a "
                "buildable design"
            )
        return spec.with_interface(params=values)

    return edit


def _edit_shift_part(spec: DesignSpec, params: dict) -> DesignSpec:
    part = spec.part(str(params["part_id"]))
    delta = [float(value) for value in params["delta_mm"]]
    pose = tuple(_mm(part.pose_mm[axis] + delta[axis]) for axis in range(3))
    return spec.with_part(part.model_copy(update={"pose_mm": pose}))


def _edit_rotate_part(spec: DesignSpec, params: dict) -> DesignSpec:
    part = spec.part(str(params["part_id"]))
    delta = [float(value) for value in params["delta_deg"]]
    rot = tuple(_mm(part.rot_deg[axis] + delta[axis]) for axis in range(3))
    return spec.with_part(part.model_copy(update={"rot_deg": rot}))


def switch_params(spec: DesignSpec, kind: str, given: Mapping[str, object] | None) -> dict:
    """The parameter set a switch to `kind` will build, from `given` or translated.

    Translation keeps the three things that are facts about the *part*, not about
    the paradigm — how deep it engages, how far above it starts, and how big the
    thing that goes in is — and re-aims the fit at the middle of the band when the
    current clearance is not already acceptable. Anything else is the new kind's
    default, because inventing a hook height from a peg diameter is guesswork
    pretending to be design.

    A kind whose model demands a number the old design cannot supply is a loud
    `ValueError`, not a fabricated default: the caller must pass `params` instead.
    """
    model = builtin_params_model(kind)
    if model is None:
        raise KeyError(f"connector kind {kind!r} is not builtin and cannot be switched to")
    if given:
        return model.model_validate(dict(given)).model_dump()

    old_kind = spec.interface.kind
    old = _validated(old_kind, spec.interface.params)
    old_size = float(old.get(_SIZE_FIELD.get(old_kind, ""), 0.0) or 0.0)
    if old_size <= 0.0:
        raise ValueError(
            f"cannot translate a {old_kind} design into {kind!r}: the old interface carries "
            f"no {_SIZE_FIELD.get(old_kind)!r} to carry over, so pass that kind's parameters "
            "explicitly"
        )
    seed = {_SIZE_FIELD[kind]: _mm(old_size)}
    # The new kind's clearance field gets a placeholder: `params_with_clearance`
    # writes the real number a few lines down, and a kind like `peg_hole` declares
    # its clearance field as *required*, so it cannot be absent at construction.
    feature = (_RESIZE.get((kind, "resize_feature")) or (None,))[0]
    if feature:
        seed.setdefault(feature, _mm(old_size))
    translated = model(**seed).model_dump()
    for shared in ("engage_mm", "approach_mm"):
        if shared in old:
            translated[shared] = old[shared]

    band = spec.constraints
    target = (
        spec.clearance_mm
        if band.in_band(spec.clearance_mm)
        else (band.min_clearance_mm + band.max_clearance_mm) / 2.0
    )
    return params_with_clearance(kind, translated, target)


def _seat_the_insert(spec: DesignSpec, kind: str) -> DesignSpec:
    """Put the insert where the *new* kind says it seats.

    A kind change rewrites the interface but leaves `insert.pose_mm` wherever the
    old kind put it. Those two poses are not interchangeable: a peg's seated pose is
    inside a socket the hole was drilled for, while a dovetail's is half a tongue
    along the body's face. Keeping the old pose builds an intersection that the
    design never asked for — measured, peg_hole -> dovetail on the 2026-11-01
    fixtures: **1800.01 mm³** of seated overlap, which no operator could repair and
    which would have been reported as a blocking verifier failure.

    Only the new kind knows where its own insert seats, so the answer comes from
    `ConnectorBuilder.seated_pose_mm` rather than from a formula per kind here —
    a second copy of that placement is the same mistake written twice.
    """
    pose = tuple(_mm(value) for value in builder_for(kind).seated_pose_mm(spec))
    return spec.with_part(spec.insert.model_copy(update={"pose_mm": pose}))


def _edit_switch_connector(spec: DesignSpec, params: dict) -> DesignSpec:
    kind = str(params["kind"])
    if kind == spec.interface.kind:
        raise ValueError(
            f"switch_connector to {kind!r} is not a switch: the interface already uses it"
        )
    # `with_interface` replaces the parameter set wholesale on a kind change, which
    # is exactly what a paradigm switch needs: the old kind's fields are not
    # parameters of the new one and every kind's model refuses them.
    switched = spec.with_interface(
        kind=kind, params=switch_params(spec, kind, params.get("params") or None)
    )
    return _seat_the_insert(switched, kind)


def _edit_regenerate_interface(spec: DesignSpec, params: dict) -> DesignSpec:
    patch = dict(params.get("params") or {})
    hint = str(params.get("prompt_hint") or "").strip()
    if not patch and not hint:
        raise ValueError(
            "regenerate_interface needs at least one parameter or a prompt hint; a "
            "regeneration that changes nothing is not a repair"
        )
    updated = spec
    if patch:
        iface = spec.interface
        updated = spec.with_interface(
            params=_validated(iface.kind, {**iface.params, **patch})
        )
    if hint:
        # The hint travels with the design, not with the policy: whichever adapter
        # materialises this revision reads it from provenance, so replaying the spec
        # reproduces the same prompt.
        updated = updated.edit(
            provenance={**updated.provenance, "prompt_hint": hint}
        )
    return updated


# --------------------------------------------------------------------------- #
# Assembly plumbing
# --------------------------------------------------------------------------- #
def _mm(value: float) -> float:
    """A millimetre as it is recorded: six decimals, and `-0.0` folded to `0.0`.

    The same convention `registry._length` uses, so two traces of the same number
    have one spelling and one hash.
    """
    return round(float(value), 6) + 0.0


# --------------------------------------------------------------------------- #
# The seven operators
# --------------------------------------------------------------------------- #
def make_design_operators() -> list[Operator]:
    """The design plane's closed action space, built once per process.

    No materialiser and no output directory: an operator edits a spec and returns
    the next `Assembly` version (`docstring`, "An operator is pure"). Whoever owns
    the run — the composition root, or T2's cascade adapter — materialises the
    snapshot whose `geom.path` is still empty.
    """

    def apply_with(edit: SpecEdit) -> Callable[[Assembly, dict], Assembly]:
        def apply(asm: Assembly, params: dict) -> Assembly:
            spec = _checked_spec(asm, params)
            edited = edit(spec, params)
            return assembly_from_spec(
                spec=edited, parent=asm.id, version=asm.version + 1
            )

        return apply

    def part_operator(
        operator_id: str,
        params_model: type[BaseModel],
        edit: SpecEdit,
        *,
        codes: tuple[str, ...],
        cost: Literal["cheap", "medium"],
    ) -> Operator:
        return Operator(
            id=operator_id,
            params=params_model,
            preconditions=(_fires_on(*codes), _is_design_plane),
            apply=apply_with(edit),
            targets=_part_targets,
            cost=cost,
            family="geometry",
        )

    def interface_operator(
        operator_id: str,
        params_model: type[BaseModel],
        edit: SpecEdit,
        *,
        codes: tuple[str, ...],
        cost: Literal["cheap", "medium"],
        family: Literal["interface", "connector", "generative"],
    ) -> Operator:
        return Operator(
            id=operator_id,
            params=params_model,
            preconditions=(_fires_on(*codes), _is_design_plane),
            apply=apply_with(edit),
            targets=_interface_targets,
            cost=cost,
            family=family,
        )

    return [
        # The exact-target operator: one number in, the fit band out, no
        # iteration. It is cheap because the arithmetic is invertible, not
        # because the geometry is small.
        interface_operator(
            "set_clearance",
            SetClearanceParams,
            _edit_set_clearance,
            codes=_CLEARANCE_CODES,
            cost="cheap",
            family="interface",
        ),
        # The two delta operators: they change a *feature size*, so they also
        # move what the fit verifier measures (clearance) but say so in the
        # vocabulary a printer and a CAD feature tree use.
        interface_operator(
            "resize_feature",
            ResizeFeatureParams,
            _resize_edit("resize_feature"),
            codes=(CODE_INSUFFICIENT_CLEARANCE, CODE_INSERTION_PATH_BLOCKED),
            cost="cheap",
            family="interface",
        ),
        interface_operator(
            "resize_insert",
            ResizeInsertParams,
            _resize_edit("resize_insert"),
            codes=(CODE_GEOMETRIC_INTERFERENCE,),
            cost="cheap",
            family="interface",
        ),
        # Pose operators. Medium, not cheap: they move a whole part, so a repair
        # for one failure can open another one anywhere the part now points.
        part_operator(
            "shift_part",
            ShiftPartParams,
            _edit_shift_part,
            codes=(CODE_GEOMETRIC_INTERFERENCE, CODE_INSERTION_PATH_BLOCKED),
            cost="medium",
        ),
        part_operator(
            "rotate_part",
            RotatePartParams,
            _edit_rotate_part,
            codes=(CODE_INSERTION_PATH_BLOCKED,),
            cost="medium",
        ),
        interface_operator(
            "switch_connector",
            SwitchConnectorParams,
            _edit_switch_connector,
            codes=HARD_CODES,
            cost="medium",
            family="connector",
        ),
        interface_operator(
            "regenerate_interface",
            RegenerateInterfaceParams,
            _edit_regenerate_interface,
            codes=HARD_CODES,
            cost="medium",
            family="generative",
        ),
    ]


def register_builtin(registry) -> None:
    """Adds all seven design-plane operators to `registry`.

    Named for symmetry with `policy.registry.register_builtin`, which registers
    the dataset plane's three: whichever plane a composition root is on, "the
    builtin action space" is one call. Deliberately no filtering and no policy —
    which operators a run may use is the engine's mandate and the policy's
    business, not the table's.
    """
    for operator in make_design_operators():
        registry.register(operator)


#: The task-facing name for the same call; `register_builtin` is what T4's
#: runtime imports.
register_design_operators = register_builtin
