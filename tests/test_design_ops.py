"""Contract tests for the design plane's seven operators (`policy/design_ops.py`).

The operators are **pure** (Lead, 2026-11-01): `apply` reads a spec off the
snapshot, edits it, and returns the next `Assembly` version. No materialiser, no
output directory, no mesh. So nothing here needs a fake materialiser either — a
base snapshot is `assembly_from_spec(spec=..., version=1)` and that is the whole
fixture. What a materialiser would do with the resulting snapshot is T1's and
T2's business, and the one contract point this file does pin is that the
operator leaves `geom.path` empty and the *hash* already moved.

These helpers are reused by `test_design_policies.py` and `test_llm_policy.py`.
"""

from __future__ import annotations

import warnings

import pytest
import trimesh
from pydantic import ValidationError

from pah.core.model import Assembly, FailureCertificate, RepairPlan
from pah.design import DesignSpec, FitConstraints, InterfaceSpec, PartSpec
from pah.design.api import (
    DESIGN_OPERATOR_IDS,
    assembly_from_spec,
    spec_of,
)
from pah.design.spec import radial_clearance_mm
from pah.policy.design_ops import (
    CONNECTOR_KINDS,
    HARD_CODES,
    delta_for_clearance,
    expressible,
    make_design_operators,
    next_connector_kind,
    params_with_clearance,
    register_builtin,
    resize_field,
    switch_params,
)
from pah.policy.registry import OperatorRegistry

# --------------------------------------------------------------------------- #
# Fixtures: a spec, the snapshot it becomes, a registry, a legal plan
# --------------------------------------------------------------------------- #
DEFAULT_PARAMS: dict[str, dict] = {
    "peg_hole": {"d_peg_mm": 10.0, "d_hole_mm": 10.0},
    "snap_fit": {"hook_w_mm": 4.0, "slot_w_mm": 4.0},
    "dovetail": {"width_mm": 10.0, "clearance_mm": 0.05},
}


def make_spec(
    kind: str = "peg_hole",
    params: dict | None = None,
    *,
    spec_id: str = "demo",
    min_c: float = 0.10,
    max_c: float = 0.60,
) -> DesignSpec:
    return DesignSpec(
        spec_id=spec_id,
        requirement=f"a {kind} that holds",
        parts=(
            PartSpec(id="body", role="body", size_mm=(40.0, 40.0, 12.0)),
            PartSpec(
                id="insert",
                role="insert",
                size_mm=(10.0, 10.0, 20.0),
                pose_mm=(0.0, 0.0, 8.0),
            ),
        ),
        interfaces=(
            InterfaceSpec(
                id="iface_0",
                kind=kind,
                a="body",
                b="insert",
                params=dict(params or DEFAULT_PARAMS[kind]),
            ),
        ),
        constraints=FitConstraints(min_clearance_mm=min_c, max_clearance_mm=max_c),
    )


def base_assembly(spec: DesignSpec | None = None) -> Assembly:
    """The version-1 snapshot the loop starts from — purely, with no geometry."""
    return assembly_from_spec(spec=spec or make_spec(), version=1)


def build_registry() -> OperatorRegistry:
    registry = OperatorRegistry()
    register_builtin(registry)
    return registry


def plan_for(registry: OperatorRegistry, operator_id: str, params: dict) -> RepairPlan:
    """A plan whose targets are the ones the operator computes — legal by gate 3."""
    operator = registry.get(operator_id)
    model = operator.params.model_validate(params)
    return RepairPlan(
        id=f"plan-{operator_id}",
        operator=operator_id,
        targets=tuple(operator.targets(model.model_dump())),
        params=params,
    )


def apply_plan(
    registry: OperatorRegistry, assembly: Assembly, operator_id: str, params: dict
) -> Assembly:
    return registry.apply(assembly, plan_for(registry, operator_id, params))


def iface_parts(assembly: Assembly) -> tuple[str, ...]:
    spec = spec_of(assembly)
    return (spec.interface.a, spec.interface.b)


def clearance_plan(
    registry: OperatorRegistry, assembly: Assembly, clearance_mm: float
) -> RepairPlan:
    return plan_for(
        registry,
        "set_clearance",
        {
            "interface_id": "iface_0",
            "parts": iface_parts(assembly),
            "clearance_mm": clearance_mm,
        },
    )


def cert(code: str, **magnitude: float) -> FailureCertificate:
    """One piece of design-plane evidence, shaped like the verifiers' output."""
    return FailureCertificate(
        code=code,
        family="geometry",
        locus={"interface": "iface_0"},
        magnitude=dict(magnitude),
        observed_by="test",
    )


# --------------------------------------------------------------------------- #
# The table itself
# --------------------------------------------------------------------------- #
def test_the_seven_operators_are_the_contracted_seven():
    registry = build_registry()

    assert tuple(sorted(registry._operators)) == tuple(sorted(DESIGN_OPERATOR_IDS))
    for operator_id in DESIGN_OPERATOR_IDS:
        assert registry.get(operator_id).family in {
            "geometry",
            "interface",
            "connector",
            "generative",
        }
        assert registry.get(operator_id).cost in {"cheap", "medium"}


def test_every_parameter_model_forbids_a_field_it_does_not_read():
    """The rule that makes an LLM's schema meaningful: no smuggled geometry."""
    registry = build_registry()
    smuggled = {
        "interface_id": "iface_0",
        "parts": ("body", "insert"),
        "vertex": [1.0, 2.0, 3.0],
    }

    with pytest.raises(ValidationError):
        registry.get("set_clearance").params.model_validate(smuggled)


def test_an_unknown_operator_is_refused_by_the_registry():
    registry = build_registry()
    invented = RepairPlan(
        id="p", operator="move_vertex", targets=("body",), params={"x": 1.0}
    )

    with pytest.raises(KeyError):
        registry.apply(base_assembly(), invented)


def test_a_plan_whose_targets_do_not_match_its_params_is_refused():
    registry = build_registry()
    spoofed = RepairPlan(
        id="p",
        operator="set_clearance",
        targets=("body",),
        params={
            "interface_id": "iface_0",
            "parts": ("body", "insert"),
            "clearance_mm": 0.3,
        },
    )

    with pytest.raises(ValueError, match="declares targets"):
        registry.apply(base_assembly(), spoofed)


def test_a_plan_for_a_different_interface_is_refused_by_the_operator():
    """Gate 3 proves the plan agrees with itself; the operator proves the design does."""
    registry = build_registry()
    elsewhere = RepairPlan(
        id="p",
        operator="set_clearance",
        targets=("body", "insert"),
        params={
            "interface_id": "iface_9",
            "parts": ("body", "insert"),
            "clearance_mm": 0.3,
        },
    )

    with pytest.raises(KeyError, match="iface_0"):
        registry.apply(base_assembly(), elsewhere)


# --------------------------------------------------------------------------- #
# Clearance: the arithmetic the whole plane argues about
# --------------------------------------------------------------------------- #
def test_set_clearance_translates_the_target_into_each_kinds_own_parameter():
    """The three lines of `params_with_clearance`, asserted as numbers.

    peg_hole: d_hole = d_peg + 2c · snap_fit: slot_w = hook_w + 2c ·
    dovetail: clearance_mm = c
    """
    target = 0.25

    peg = params_with_clearance("peg_hole", {"d_peg_mm": 10.0, "d_hole_mm": 10.0}, target)
    assert peg["d_hole_mm"] == pytest.approx(10.5)
    assert (peg["d_hole_mm"] - peg["d_peg_mm"]) / 2.0 == pytest.approx(target)

    snap = params_with_clearance("snap_fit", {"hook_w_mm": 4.0, "slot_w_mm": 4.0}, target)
    assert snap["slot_w_mm"] == pytest.approx(4.5)
    assert (snap["slot_w_mm"] - snap["hook_w_mm"]) / 2.0 == pytest.approx(target)

    dove = params_with_clearance("dovetail", {"clearance_mm": 0.05}, target)
    assert dove["clearance_mm"] == pytest.approx(target)


@pytest.mark.parametrize("kind", ["peg_hole", "snap_fit", "dovetail"])
def test_set_clearance_lands_on_the_requested_clearance_in_the_built_spec(kind):
    registry = build_registry()
    assembly = base_assembly(make_spec(kind))

    result = apply_plan(registry, assembly, "set_clearance", clearance_plan(
        registry, assembly, 0.3
    ).params)

    assert spec_of(result).clearance_mm == pytest.approx(0.3)
    assert spec_of(result).constraints.in_band(spec_of(result).clearance_mm)


def test_a_negative_target_clearance_is_a_legal_interference_request():
    """Designing an interference fit is legal; judging it is the verifier's job."""
    assert params_with_clearance(
        "peg_hole", {"d_peg_mm": 10.0, "d_hole_mm": 10.0}, -0.1
    )["d_hole_mm"] == pytest.approx(9.8)


def test_a_dovetail_clearance_is_floored_at_zero():
    """`dovetail.clearance_mm >= 0` is an *inclusive* bound and 0 is a real
    dovetail, so a negative target lands on the tightest buildable fit instead of
    producing a spec the kind's own model refuses."""
    assert params_with_clearance("dovetail", {"clearance_mm": 0.1}, -0.1)[
        "clearance_mm"
    ] == 0.0

    registry = build_registry()
    assembly = base_assembly(make_spec("dovetail"))
    result = apply_plan(
        registry,
        assembly,
        "resize_feature",
        {
            "interface_id": "iface_0",
            "parts": iface_parts(assembly),
            "delta_mm": -0.5,  # would ask for -0.45
        },
    )
    assert spec_of(result).clearance_mm == 0.0


def test_a_resize_that_would_empty_a_feature_is_refused():
    """An *exclusive* bound is refused, not clamped: a peg of diameter 5e-324 is
    legal in the model and nonsense as a part."""
    registry = build_registry()
    assembly = base_assembly()

    with pytest.raises(ValueError, match="buildable"):
        apply_plan(
            registry,
            assembly,
            "resize_insert",
            {
                "interface_id": "iface_0",
                "parts": iface_parts(assembly),
                "delta_mm": -10.0,  # d_peg 10.0 -> 0.0, and the model wants > 0
            },
        )


# --------------------------------------------------------------------------- #
# The delta operators
# --------------------------------------------------------------------------- #
def test_resize_feature_opens_the_cavity_and_resize_insert_closes_the_fit():
    registry = build_registry()
    assembly = base_assembly()
    parts = iface_parts(assembly)

    wider = apply_plan(
        registry,
        assembly,
        "resize_feature",
        {"interface_id": "iface_0", "parts": parts, "delta_mm": 0.4},
    )
    # 0.4 mm of hole diameter is 0.2 mm of radial clearance.
    assert spec_of(wider).clearance_mm == pytest.approx(0.2)

    fatter = apply_plan(
        registry,
        assembly,
        "resize_insert",
        {"interface_id": "iface_0", "parts": parts, "delta_mm": 0.4},
    )
    assert spec_of(fatter).clearance_mm == pytest.approx(-0.2)


def test_delta_for_clearance_inverts_the_table():
    """The delta a policy computes, fed through ``field += delta``, lands on target."""
    for kind, current in (("peg_hole", -0.1), ("snap_fit", 0.0), ("dovetail", 0.05)):
        params = params_with_clearance(kind, make_spec(kind).interface.params, current)
        for operator_id in ("resize_feature", "resize_insert"):
            if not expressible(kind, operator_id):
                continue
            field = resize_field(kind, operator_id)[0]
            moved = dict(params)
            moved[field] = params[field] + delta_for_clearance(
                kind, operator_id, current, 0.25
            )
            assert radial_clearance_mm(kind, moved) == pytest.approx(0.25), (
                kind,
                operator_id,
                moved,
            )


def test_dovetail_has_no_insert_side_number():
    assert expressible("dovetail", "resize_feature")
    assert not expressible("dovetail", "resize_insert")
    assert expressible("peg_hole", "resize_insert")

    registry = build_registry()
    assembly = base_assembly(make_spec("dovetail"))
    with pytest.raises(KeyError, match="no field"):
        apply_plan(
            registry,
            assembly,
            "resize_insert",
            {
                "interface_id": "iface_0",
                "parts": iface_parts(assembly),
                "delta_mm": -0.1,
            },
        )


# --------------------------------------------------------------------------- #
# Pose operators
# --------------------------------------------------------------------------- #
def test_shift_part_moves_the_pose_and_leaves_the_interface_alone():
    registry = build_registry()
    assembly = base_assembly()

    result = apply_plan(
        registry, assembly, "shift_part", {"part_id": "insert", "delta_mm": (2.0, 0.0, 0.0)}
    )

    spec = spec_of(result)
    assert spec.part("insert").pose_mm == (2.0, 0.0, 8.0)
    assert spec.part("body").pose_mm == (0.0, 0.0, 0.0)
    assert spec.interface.params == spec_of(assembly).interface.params


def test_rotate_part_turns_the_pose():
    registry = build_registry()
    assembly = base_assembly()

    result = apply_plan(
        registry, assembly, "rotate_part", {"part_id": "insert", "delta_deg": (0.0, 15.0, 0.0)}
    )

    assert spec_of(result).part("insert").rot_deg == (0.0, 15.0, 0.0)


def test_a_pose_operator_naming_an_absent_part_is_refused():
    registry = build_registry()

    with pytest.raises(KeyError, match="no part"):
        apply_plan(
            registry, base_assembly(), "shift_part", {"part_id": "ghost", "delta_mm": (1.0, 0.0, 0.0)}
        )


# --------------------------------------------------------------------------- #
# The paradigm operators
# --------------------------------------------------------------------------- #
def test_switch_connector_translates_the_part_that_has_to_fit():
    """A 10.2 mm peg becomes a 10.2 mm hook, and the fit is re-aimed into the band."""
    registry = build_registry()
    assembly = base_assembly(make_spec(params={"d_peg_mm": 10.2, "d_hole_mm": 10.0}))

    result = apply_plan(
        registry,
        assembly,
        "switch_connector",
        {"interface_id": "iface_0", "parts": iface_parts(assembly), "kind": "snap_fit"},
    )

    spec = spec_of(result)
    assert spec.interface.kind == "snap_fit"
    assert spec.interface.params["hook_w_mm"] == pytest.approx(10.2)
    assert spec.constraints.in_band(spec.clearance_mm)
    # The old kind's fields are gone: they are not parameters of the new kind.
    assert "d_hole_mm" not in spec.interface.params


def test_switch_connector_takes_explicit_parameters_verbatim():
    registry = build_registry()
    assembly = base_assembly()

    result = apply_plan(
        registry,
        assembly,
        "switch_connector",
        {
            "interface_id": "iface_0",
            "parts": iface_parts(assembly),
            "kind": "dovetail",
            "params": {"width_mm": 8.0, "clearance_mm": 0.2},
        },
    )

    assert spec_of(result).interface.params["width_mm"] == pytest.approx(8.0)
    assert spec_of(result).clearance_mm == pytest.approx(0.2)


def test_switch_params_re_aims_an_out_of_band_fit():
    """The switch is a design decision, so it lands inside the band, not wherever
    the old number happened to be."""
    spec = make_spec(params={"d_peg_mm": 10.2, "d_hole_mm": 10.0})  # clearance -0.1

    translated = switch_params(spec, "snap_fit", None)

    assert spec.constraints.in_band(radial_clearance_mm("snap_fit", translated))


def test_switch_connector_reseats_the_insert_at_the_new_kinds_seated_pose():
    """A kind switch has to move the insert: the peg's seated pose is inside a
    socket, the dovetail's is half a tongue along the body's face."""
    from pah.design.connectors import builder_for

    registry = build_registry()
    assembly = base_assembly(make_spec(params={"d_peg_mm": 10.2, "d_hole_mm": 10.0}))

    result = apply_plan(
        registry,
        assembly,
        "switch_connector",
        {
            "interface_id": "iface_0",
            "parts": iface_parts(assembly),
            "kind": "dovetail",
        },
    )

    spec = spec_of(result)
    assert spec.interface.kind == "dovetail"
    assert spec.insert.pose_mm == tuple(builder_for("dovetail").seated_pose_mm(spec))
    assert spec.insert.pose_mm != spec_of(assembly).insert.pose_mm


def test_a_kind_switch_does_not_build_a_seated_intersection(tmp_path):
    """The acceptance the Lead set, on real geometry.

    Without the re-seat this exact switch measured **1800.01 mm³** of seated
    overlap (T1's `demo-peg-hole-interfere` -> dovetail, 2026-11-01): a blocking
    false positive manufactured by the operator, which no policy could repair
    because the defect was never in the design's intent.
    """
    from pah.design.materialize import materialize

    registry = build_registry()
    base = materialize(
        make_spec(params={"d_peg_mm": 10.2, "d_hole_mm": 10.0}), tmp_path / "base"
    )

    switched = apply_plan(
        registry,
        base.assembly,
        "switch_connector",
        {
            "interface_id": "iface_0",
            "parts": iface_parts(base.assembly),
            "kind": "dovetail",
        },
    )
    built = materialize(spec_of(switched), tmp_path / "switched")

    # A zero-volume intersection has no centroid, and trimesh says so with a
    # RuntimeWarning: expected here, and not a finding about the design.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        overlap = trimesh.boolean.intersection(
            [built.meshes["body"], built.meshes["insert"]], engine="manifold"
        )
        volume = (
            0.0
            if overlap is None or overlap.is_empty
            else abs(float(overlap.volume))
        )
    assert volume < 1e-3, f"seated overlap after the switch: {volume} mm3"


def test_switching_to_the_kind_already_in_use_is_refused():
    registry = build_registry()
    assembly = base_assembly()

    with pytest.raises(ValueError, match="not a switch"):
        apply_plan(
            registry,
            assembly,
            "switch_connector",
            {"interface_id": "iface_0", "parts": iface_parts(assembly), "kind": "peg_hole"},
        )


def test_next_connector_kind_never_returns_the_current_one():
    for kind in CONNECTOR_KINDS:
        assert next_connector_kind(kind) != kind
    assert next_connector_kind("a-plugin-kind") == CONNECTOR_KINDS[0]


def test_regenerate_interface_patches_the_current_parameters():
    registry = build_registry()
    assembly = base_assembly()

    result = apply_plan(
        registry,
        assembly,
        "regenerate_interface",
        {
            "interface_id": "iface_0",
            "parts": iface_parts(assembly),
            "params": {"d_hole_mm": 10.6, "engage_mm": 9.0},
        },
    )

    spec = spec_of(result)
    assert spec.interface.params["d_hole_mm"] == pytest.approx(10.6)
    assert spec.interface.params["engage_mm"] == pytest.approx(9.0)
    assert spec.interface.params["d_peg_mm"] == pytest.approx(10.0)  # untouched


def test_regenerating_nothing_is_refused():
    registry = build_registry()
    assembly = base_assembly()

    with pytest.raises(ValueError, match="changes nothing"):
        apply_plan(
            registry,
            assembly,
            "regenerate_interface",
            {"interface_id": "iface_0", "parts": iface_parts(assembly)},
        )


# --------------------------------------------------------------------------- #
# Purity: version chain, no I/O, hashes from the spec
# --------------------------------------------------------------------------- #
def test_every_apply_makes_one_new_version_pointing_back():
    registry = build_registry()
    assembly = base_assembly()

    result = registry.apply(assembly, clearance_plan(registry, assembly, 0.3))

    assert result.version == assembly.version + 1
    assert result.parent == assembly.id
    assert result.id != assembly.id
    assert assembly.version == 1  # the base was not mutated


def test_an_operator_writes_no_geometry_but_its_hash_already_moved():
    registry = build_registry()
    assembly = base_assembly()

    result = registry.apply(assembly, clearance_plan(registry, assembly, 0.3))

    before = {part["id"]: part["geom"]["hash"] for part in assembly.parts}
    after = {part["id"]: part["geom"]["hash"] for part in result.parts}
    assert all(part["geom"]["path"] == "" for part in result.parts)
    assert after != before  # the identity follows the numbers, not the file
    assert all(hash_.startswith("sha256:") for hash_ in after.values())


def test_the_operators_edit_specs_the_real_materialiser_can_build(tmp_path):
    """Integration with T1 (`pah.design.materialize`), added once it landed.

    Everything above is pure, and "the spec changed" and "the geometry changed" are
    different claims: only the second one is something a verifier can see. So this
    test takes an operator's output through the real trimesh builder and checks the
    triangles moved the way the number says they should.
    """
    from pah.design.materialize import materialize

    registry = build_registry()
    base = materialize(make_spec(), tmp_path / "base")  # peg_hole 10.0 / 10.0, clearance 0

    plan = clearance_plan(registry, base.assembly, 0.35)
    repaired = registry.apply(base.assembly, plan)
    assert all(part["geom"]["path"] == "" for part in repaired.parts)

    built = materialize(spec_of(repaired), tmp_path / "repaired")

    # The hole grew by 0.7 mm of diameter, so the body lost material and the insert
    # kept its own. Numbers, not just "a file appeared".
    assert built.meshes["body"].volume < base.meshes["body"].volume
    assert built.meshes["insert"].volume == pytest.approx(base.meshes["insert"].volume)
    assert spec_of(built.assembly).clearance_mm == pytest.approx(0.35)
    assert all(part["geom"]["path"] for part in built.assembly.parts)
    assert built.paths["body"] != base.paths["body"]


def test_two_different_designs_from_one_base_are_two_assemblies():
    """A best-of-n search applies two plans to the same base; they must not collide."""
    registry = build_registry()
    assembly = base_assembly()

    first = registry.apply(assembly, clearance_plan(registry, assembly, 0.2))
    second = registry.apply(assembly, clearance_plan(registry, assembly, 0.5))

    assert first.id != second.id
    assert first.parts[0]["geom"]["hash"] != second.parts[0]["geom"]["hash"]


def test_reapplying_the_same_plan_is_idempotent():
    """Same base + same numbers = the same snapshot, which is what resume needs."""
    registry = build_registry()
    assembly = base_assembly()
    plan = clearance_plan(registry, assembly, 0.3)

    assert registry.apply(assembly, plan).id == registry.apply(assembly, plan).id


def test_switching_kind_moves_the_hash_even_with_identical_numbers():
    """`geometry_hash` folds the kind in, so "same numbers, other paradigm" is new."""
    registry = build_registry()
    assembly = base_assembly()

    result = apply_plan(
        registry,
        assembly,
        "switch_connector",
        {"interface_id": "iface_0", "parts": iface_parts(assembly), "kind": "snap_fit"},
    )

    before = {part["id"]: part["geom"]["hash"] for part in assembly.parts}
    after = {part["id"]: part["geom"]["hash"] for part in result.parts}
    assert after != before


# --------------------------------------------------------------------------- #
# Preconditions: what the evidence has to be before an operator is even legal
# --------------------------------------------------------------------------- #
def test_preconditions_route_the_contracted_codes():
    registry = build_registry()
    assembly = base_assembly()

    insufficient = registry.applicable(assembly, [cert("insufficient_clearance")])
    assert "set_clearance" in insufficient
    assert "resize_feature" in insufficient
    assert "rotate_part" not in insufficient
    assert "switch_connector" in insufficient  # any hard code is a paradigm question

    blocked = registry.applicable(assembly, [cert("insertion_path_blocked")])
    assert "rotate_part" in blocked and "shift_part" in blocked

    assert registry.applicable(assembly, []) == []


def test_a_dataset_plane_assembly_is_not_design_plane_evidence():
    """The M0/M1 cases have no spec; a design operator must not pretend they do."""
    registry = build_registry()
    dataset = Assembly(
        id="asm_case",
        version=0,
        parts=({"id": "p0", "geom": {"hash": "h0"}},),
        joints=({"id": "j0", "a": "p0", "b": "p1"},),
        connectors=(),
        constraints={},
    )

    assert registry.applicable(dataset, [cert("insufficient_clearance")]) == []


def test_environment_family_evidence_is_never_a_repair_trigger():
    """`physics_error` is an environment code: it must not make an operator legal."""
    registry = build_registry()
    environment = FailureCertificate(
        code="physics_error",
        family="environment",
        locus={"interface": "iface_0"},
        magnitude={},
        observed_by="physics_settle",
    )

    assert registry.applicable(base_assembly(), [environment]) == []


def test_hard_codes_are_the_design_codes_minus_the_environment_one():
    assert "physics_error" not in HARD_CODES
    assert set(HARD_CODES) == {
        "insufficient_clearance",
        "excessive_clearance",
        "insufficient_material",
        "excess_material",
        "geometric_interference",
        "insertion_path_blocked",
        # Global assembly: two parts that share no interface can still collide, and
        # only the whole-assembly pass sees it.
        "global_interference",
        "below_min_wall",
        "unsupported_overhang",
        "joint_unstable",
        "dynamic_penetration",
    }


def test_the_operator_list_is_rebuilt_per_call_not_shared():
    """Two registries (two runs) must not collide on the same operator objects."""
    first = make_design_operators()
    second = make_design_operators()
    assert [op.id for op in first] == [op.id for op in second]
    assert first[0] is not second[0]
