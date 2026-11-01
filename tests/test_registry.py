"""The registry's four promises, tested without geometry.

Every test here runs on a hand-built `Assembly` of plain dicts: the registry is the
action space, not a geometry layer, so the only things worth pinning are the guards
— an unknown operator, a malformed parameter, a plan whose parameters contradict its
declared targets, and a joint the plan names but the assembly does not agree with.

The one numeric test uses a real row (`asm_000000:7`): its `r_mm = 0.661631` and
`gate_slack_mm = 0.030381` were produced by `closed_form_radius` during M1, so the
registry reproducing them proves the adapter reuses the closed form instead of
re-deriving it.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from pah.core.model import Assembly, FailureCertificate, RepairPlan
from pah.policy import repair_script
from pah.policy.registry import (
    RelocateConnectorParams,
    ResizeConnectorParams,
    SwapFamilyParams,
    Operator,
    OperatorRegistry,
    register_builtin,
)

# --------------------------------------------------------------------------- #
# Fixtures: one assembly, one joint, one certificate
# --------------------------------------------------------------------------- #
#: `asm_000000:7` — gap/t_b/d_cap/r_cap and the radius the M1 pass picked.
GAP_MM = 0.004754
T_B_MM = 1.519051
D_CAP_MM = 1.592661
R_CAP_MM = 1.178991
M1_R_MM = 0.661631
M1_SLACK_MM = 0.030381


def joint_record(joint_id: str = "j6", a: str = "p1", b: str = "p6", **overrides) -> dict:
    """One `case.json` joint (`connector_feasible._joint`), lengths in millimetres."""
    record = {
        "id": joint_id,
        "edge_id": int(joint_id[1:]),
        "a": a,
        "b": b,
        "anchor_mm": [12.5, -3.25, 47.0],
        "n_ab": [0.0, 0.0, 1.0],
        "half_w_mm": 2.0,
        "half_h_mm": 1.5,
        "r_inplane_max_mm": R_CAP_MM,
        "depth_max_mm": D_CAP_MM,
        "t_b_mm": T_B_MM,
    }
    record.update(overrides)
    return record


def assembly_of(*joints: dict, version: int = 0) -> Assembly:
    return Assembly(
        id="asm_000000",
        version=version,
        parts=({"id": "p1"}, {"id": "p6"}, {"id": "p7"}),
        joints=tuple(joints) or (joint_record(),),
        connectors=(),
        constraints={"unit_mm": 1500.0},
    )


def certificate(
    code: str = "gap_too_wide",
    *,
    joint_id: str = "j6",
    parts: tuple[str, ...] = ("p1", "p6"),
) -> FailureCertificate:
    """The shape `connector_feasible.certificates_for` actually emits."""
    return FailureCertificate(
        code=code,
        family="geometry",
        locus={"parts": list(parts), "joint": joint_id, "edge": 6},
        magnitude={"value": GAP_MM, "unit": "mm", "threshold": None, "direction": "gt"},
        observed_by="connector_feasible@1",
    )


# --------------------------------------------------------------------------- #
# A stand-in operator, so the guards are tested before any builtin exists
# --------------------------------------------------------------------------- #
class DummyParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    part: str
    amount: float = 1.0


def make_operator(
    operator_id: str = "dummy",
    *,
    preconditions=(),
    calls: list | None = None,
    targets=None,
) -> Operator:
    """An operator that records every call instead of touching geometry."""
    seen: list = [] if calls is None else calls

    def apply(asm: Assembly, params: dict) -> Assembly:
        seen.append((asm.id, dict(params)))
        return replace(asm, version=asm.version + 1, parent=asm.id)

    return Operator(
        id=operator_id,
        params=DummyParams,
        preconditions=tuple(preconditions),
        apply=apply,
        targets=targets or (lambda params: (params["part"],)),
        cost="cheap",
        family="geometry",
    )


def plan_of(
    operator: str = "dummy",
    *,
    targets: tuple[str, ...] = ("p1",),
    params: dict | None = None,
) -> RepairPlan:
    return RepairPlan(
        id="plan-1",
        operator=operator,
        targets=targets,
        params={"part": "p1"} if params is None else params,
    )


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #
def test_register_then_get_and_contains():
    registry = OperatorRegistry()
    operator = make_operator()
    registry.register(operator)

    assert "dummy" in registry
    assert "dummy" not in OperatorRegistry()
    assert registry.get("dummy") is operator


def test_duplicate_id_is_refused_and_the_first_registration_stands():
    registry = OperatorRegistry()
    first = make_operator()
    registry.register(first)

    with pytest.raises(ValueError, match="already registered"):
        registry.register(make_operator())

    assert registry.get("dummy") is first


def test_an_operator_must_declare_a_pydantic_model():
    bad = Operator(
        id="bad",
        params=dict,  # type: ignore[arg-type]
        preconditions=(),
        apply=lambda asm, params: asm,
        targets=lambda params: (),
        cost="cheap",
        family="geometry",
    )
    with pytest.raises(TypeError, match="pydantic model"):
        OperatorRegistry().register(bad)


def test_unknown_operator_is_a_key_error():
    registry = OperatorRegistry()
    with pytest.raises(KeyError):
        registry.get("nope")
    with pytest.raises(KeyError):
        registry.apply(assembly_of(), plan_of("nope"))


# --------------------------------------------------------------------------- #
# The three guards on `apply`
# --------------------------------------------------------------------------- #
def test_parameters_are_validated_and_the_exception_escapes():
    registry = OperatorRegistry()
    registry.register(make_operator())

    with pytest.raises(ValidationError):
        registry.apply(assembly_of(), plan_of(params={}))  # `part` is required
    with pytest.raises(ValidationError):
        registry.apply(assembly_of(), plan_of(params={"part": "p1", "amount": "much"}))
    with pytest.raises(ValidationError):
        registry.apply(  # extra="forbid": no field the operator does not read
            assembly_of(), plan_of(params={"part": "p1", "smuggled": 1})
        )


def test_target_mismatch_is_refused_before_the_operator_runs():
    """The most important guard: the parameters say `p1`, the plan claims `p2`."""
    calls: list = []
    registry = OperatorRegistry()
    registry.register(make_operator(calls=calls))

    with pytest.raises(ValueError, match=r"\('p2',\).*\('p1',\)"):
        registry.apply(assembly_of(), plan_of(targets=("p2",)))

    assert calls == []


def test_matching_targets_get_the_validated_parameters():
    calls: list = []
    registry = OperatorRegistry()
    registry.register(make_operator(calls=calls))

    applied = registry.apply(assembly_of(), plan_of(params={"part": "p1", "amount": 2}))

    assert calls == [("asm_000000", {"part": "p1", "amount": 2.0})]
    assert applied.version == 1 and applied.parent == "asm_000000"


def test_targets_are_compared_as_sets_not_as_sequences():
    """Both sides may order the pair either way; only the membership is the fact."""
    registry = OperatorRegistry()
    registry.register(make_operator("pair", targets=lambda params: (params["part"], "p1")))

    applied = registry.apply(
        assembly_of(),
        RepairPlan(
            id="p", operator="pair", targets=("p1", "p6"), params={"part": "p6"}
        ),
    )
    assert applied.version == 1


# --------------------------------------------------------------------------- #
# Enumeration
# --------------------------------------------------------------------------- #
def test_applicable_needs_every_precondition():
    registry = OperatorRegistry()
    registry.register(make_operator("all_pass", preconditions=(lambda a, c: True,) * 2))
    registry.register(
        make_operator(
            "one_fails", preconditions=(lambda a, c: True, lambda a, c: False)
        )
    )

    assert registry.applicable(assembly_of(), [certificate()]) == ["all_pass"]


def test_applicable_is_sorted_and_does_not_depend_on_registration_order():
    registry = OperatorRegistry()
    for operator_id in ("c", "a", "b"):
        registry.register(make_operator(operator_id, preconditions=(lambda a, c: True,)))

    assert registry.applicable(assembly_of(), [certificate()]) == ["a", "b", "c"]


def test_applicable_without_evidence_is_empty():
    """No certificate, no claim about which action is possible."""
    registry = OperatorRegistry()
    registry.register(make_operator(preconditions=(lambda a, c: True,)))

    assert registry.applicable(assembly_of(), []) == []


def test_applicable_accepts_an_operator_one_of_the_certificates_supports():
    registry = OperatorRegistry()
    gap = certificate("gap_too_wide")
    material = certificate("no_material_hit")
    registry.register(make_operator("on_gap", preconditions=(lambda a, c: c is gap,)))
    registry.register(
        make_operator("on_material", preconditions=(lambda a, c: c is material,))
    )

    assert registry.applicable(assembly_of(), [gap, material]) == ["on_gap", "on_material"]


# --------------------------------------------------------------------------- #
# The builtins
# --------------------------------------------------------------------------- #
def builtin_registry() -> OperatorRegistry:
    registry = OperatorRegistry()
    register_builtin(registry)
    return registry


def test_register_builtin_registers_the_three_m1_operators():
    registry = builtin_registry()

    for operator_id in ("resize_connector", "relocate_connector", "swap_family"):
        assert operator_id in registry
        operator = registry.get(operator_id)
        assert operator.family == "connector"
        assert operator.cost == "cheap"
    assert registry.get("resize_connector").params is ResizeConnectorParams
    assert registry.get("relocate_connector").params is RelocateConnectorParams
    assert registry.get("swap_family").params is SwapFamilyParams


def resize_plan(**params) -> RepairPlan:
    merged = {
        "joint_id": "j6",
        "parts": ("p1", "p6"),
        "gap_mm": GAP_MM,
        "t_b_mm": T_B_MM,
        "d_cap_mm": D_CAP_MM,
        "r_cap_mm": R_CAP_MM,
    }
    merged.update(params)
    return RepairPlan(
        id="plan-resize",
        operator="resize_connector",
        targets=("p1", "p6"),
        params=merged,
    )


def test_resize_reproduces_the_m1_closed_form():
    """`asm_000000:7` picked `r = 0.661631` with 0.030381 mm of gate slack."""
    started = assembly_of()
    applied = builtin_registry().apply(started, resize_plan())

    request = applied.joints[0]["connector"]
    assert request["r_mm"] == pytest.approx(M1_R_MM, abs=1e-6)
    assert request["depth_mm"] == pytest.approx(2 * M1_R_MM, abs=1e-6)
    assert repair_script.gate_margin(
        GAP_MM, T_B_MM, D_CAP_MM, request["r_mm"]
    )[1] == pytest.approx(M1_SLACK_MM, abs=1e-6)

    assert applied.version == 1
    assert applied.parent == started.id
    assert applied.id != started.id
    # The input snapshot is untouched: `Assembly` is versioned, not edited in place.
    assert "connector" not in started.joints[0]


def test_resize_takes_a_cap_override_over_the_joint_record():
    """`d_cap_mm` beats `depth_max_mm`; a cap below the floor leaves only `r_min`."""
    applied = builtin_registry().apply(
        assembly_of(), resize_plan(d_cap_mm=0.08, r_min_mm=0.25)
    )

    assert applied.joints[0]["connector"]["r_mm"] == pytest.approx(0.25)
    assert applied.joints[0]["connector"]["depth_mm"] == pytest.approx(0.08)


def test_resize_appends_to_an_existing_connector_request():
    started = assembly_of(joint_record(connector={"section": "circle"}))
    applied = builtin_registry().apply(started, resize_plan())

    request = applied.joints[0]["connector"]
    assert request["section"] == "circle"
    assert request["r_mm"] == pytest.approx(M1_R_MM, abs=1e-6)


def test_relocate_moves_the_anchor_in_the_joint_plane():
    """`n_ab = +z` gives `e1 = -y`, `e2 = +x` (`repair_script._plane_basis`)."""
    applied = builtin_registry().apply(
        assembly_of(),
        RepairPlan(
            id="plan-move",
            operator="relocate_connector",
            targets=("p1", "p6"),
            params={"joint_id": "j6", "parts": ("p1", "p6"), "du_mm": 2.0, "dv_mm": 1.0},
        ),
    )

    assert applied.joints[0]["anchor_mm"] == pytest.approx([13.5, -5.25, 47.0])


def test_swap_family_changes_only_the_section():
    started = assembly_of(
        joint_record(connector={"r_mm": 0.25, "depth_mm": 0.5, "section": "D"})
    )
    applied = builtin_registry().apply(
        started,
        RepairPlan(
            id="plan-swap",
            operator="swap_family",
            targets=("p1", "p6"),
            params={"joint_id": "j6", "parts": ("p1", "p6"), "section": "circle"},
        ),
    )

    assert applied.joints[0]["connector"] == {
        "r_mm": 0.25,
        "depth_mm": 0.5,
        "section": "circle",
    }


def test_a_builtin_refuses_a_joint_that_does_not_touch_the_declared_parts():
    """Check 3 only proves the plan is self-consistent; this is the assembly's veto."""
    started = assembly_of(joint_record(a="p1", b="p7"))
    registry = builtin_registry()

    with pytest.raises(ValueError, match="not the plan's targets"):
        registry.apply(
            started,
            RepairPlan(
                id="plan-lying",
                operator="swap_family",
                targets=("p1", "p6"),
                params={"joint_id": "j6", "parts": ("p1", "p6")},
            ),
        )


def test_a_builtin_refuses_a_joint_that_is_not_in_the_assembly():
    registry = builtin_registry()
    with pytest.raises(KeyError, match="no joint"):
        registry.apply(
            assembly_of(),
            RepairPlan(
                id="plan-missing",
                operator="swap_family",
                targets=("p1", "p6"),
                params={"joint_id": "j99", "parts": ("p1", "p6")},
            ),
        )


def test_the_same_plan_on_the_same_base_is_the_same_version():
    """Determinism is what makes a trace replayable: same facts, same id."""
    registry = builtin_registry()
    first = registry.apply(assembly_of(), resize_plan())
    second = registry.apply(assembly_of(), resize_plan())

    assert first.id == second.id
    assert first.joints == second.joints


def test_builtin_preconditions_follow_the_ladder_codes():
    registry = builtin_registry()
    asm = assembly_of()

    assert registry.applicable(asm, [certificate("gap_too_wide")]) == [
        "relocate_connector",
        "resize_connector",
    ]
    assert registry.applicable(asm, [certificate("no_material_hit")]) == [
        "relocate_connector"
    ]
    assert registry.applicable(asm, [certificate("peg_detached")]) == ["swap_family"]
    assert registry.applicable(asm, [certificate("unknown_reason")]) == []


def test_stale_evidence_makes_no_operator_applicable():
    registry = builtin_registry()
    assert registry.applicable(
        assembly_of(), [certificate("gap_too_wide", joint_id="j99")]
    ) == []
