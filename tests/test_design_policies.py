"""Tests for the design plane's baselines (`policy/design_policies.py`).

Two things are locked here. First the *numbers*: a rule that says
`insufficient_clearance -> set_clearance` must produce a clearance the design can
actually be built at and the fit band accepts, asserted as a value, not as "an
operator came back". Second the *loop behaviour*: rounds spent, actions proposed,
and the fact that a policy says `None` when the action space is exhausted instead
of proposing something it knows cannot help.

The verifiers below are fakes, but not arbitrary ones: `DesignChecks` mirrors what
`fit_clearance` + `interface_interference` will observe, in cascade order
(cheapest first), so a rule that only works because the evidence arrived in a
convenient order would fail here too.
"""

from __future__ import annotations

import pytest

from pah.app.providers import ProviderError
from pah.core.model import (
    Budget,
    FailureCertificate,
    Mandate,
    RepairPlan,
    Round,
    Ticket,
)
from pah.design import spec_of
from pah.policy.design_ops import register_builtin
from pah.policy.design_policies import (
    DEFAULT_LADDER,
    FALLBACK_LADDER,
    RandomPolicy,
    RulePolicy,
    band_target,
    build,
    expressible,
    load_ladder,
)
from pah.policy.loop import RepairLoop
from pah.policy.registry import OperatorRegistry
from test_design_ops import (  # noqa: E402  (pytest puts tests/ on sys.path)
    base_assembly,
    cert,
    make_spec,
)

KINDS = ("peg_hole", "snap_fit", "dovetail")


def rule_registry() -> OperatorRegistry:
    registry = OperatorRegistry()
    register_builtin(registry)
    return registry


def propose(policy, assembly, *certs, history=()) -> RepairPlan | None:
    return policy.propose(
        assembly,
        list(certs),
        list(history),
        Budget(rounds=3, geom_calls=100, sim_calls=20),
    )


def failed(plan: RepairPlan, code: str = "insufficient_clearance") -> Round:
    """A closed round in which `plan` did not work — the history a policy must read."""
    return Round(id=f"r-{plan.id}", index=0, plan=plan, verdict="fail", evidence=[cert(code)])


# --------------------------------------------------------------------------- #
# RulePolicy: the arithmetic, per certificate code
# --------------------------------------------------------------------------- #
def test_insufficient_clearance_is_repaired_at_the_middle_of_the_band():
    registry = rule_registry()
    assembly = base_assembly()  # peg_hole, clearance 0.0, band [0.10, 0.60]
    policy = RulePolicy(registry=registry)

    plan = propose(policy, assembly, cert("insufficient_clearance", clearance_mm=0.0))

    assert plan is not None
    assert plan.operator == "set_clearance"
    assert plan.params["clearance_mm"] == pytest.approx(0.35)  # (0.10 + 0.60) / 2
    # The number is not just in the plan: it is what the design becomes.
    repaired = registry.apply(assembly, plan)
    clearance = spec_of(repaired).clearance_mm
    assert clearance == pytest.approx(0.35)
    assert spec_of(repaired).constraints.in_band(clearance)


def test_excessive_clearance_is_repaired_at_the_lower_edge():
    registry = rule_registry()
    assembly = base_assembly(make_spec(params={"d_peg_mm": 10.0, "d_hole_mm": 11.0}))
    policy = RulePolicy(registry=registry)

    plan = propose(policy, assembly, cert("excessive_clearance", clearance_mm=0.5))

    assert plan is not None
    assert plan.operator == "set_clearance"
    assert plan.params["clearance_mm"] == pytest.approx(0.10, abs=1e-5)
    repaired = registry.apply(assembly, plan)
    assert spec_of(repaired).constraints.in_band(spec_of(repaired).clearance_mm)


def test_geometric_interference_is_pushed_out_by_turning_the_insert_down():
    """The §7 acceptance case: `d_peg=10.2, d_hole=10.0` is 0.1 mm of overlap.

    A lone overlap certificate only makes the *insert* operators legal (the
    contract's trigger table: the cavity operators belong to the fit verifier), so
    the repair is to turn the peg down by ``(|-0.1| + margin) / 0.5`` ≈ 0.4 mm —
    the overlap plus the margin the band asks for, in the parameter the kind has.
    The last ~2e-6 is the edge inset (`design_policies._EDGE_INSET_MM`), which is
    what keeps the built clearance from reading back a hair below the band.
    """
    registry = rule_registry()
    assembly = base_assembly(
        make_spec(params={"d_peg_mm": 10.2, "d_hole_mm": 10.0})
    )  # clearance -0.1: interference
    policy = RulePolicy(registry=registry)

    plan = propose(policy, assembly, cert("geometric_interference", volume_mm3=0.8))

    assert plan is not None
    assert plan.operator == "resize_insert"
    assert plan.params["delta_mm"] == pytest.approx(-0.4, abs=1e-5)
    repaired = registry.apply(assembly, plan)
    assert spec_of(repaired).clearance_mm == pytest.approx(0.10, abs=1e-5)
    assert spec_of(repaired).constraints.in_band(spec_of(repaired).clearance_mm)
    assert spec_of(repaired).interface.params["d_peg_mm"] == pytest.approx(9.8, abs=1e-5)


def test_an_out_of_band_overlap_is_repaired_by_the_cheaper_fit_certificate():
    """`d_peg=10.2, d_hole=10.0` also violates the band, and the fit certificate is
    cheaper, so it arrives first and `set_clearance` takes the turn — the other
    half of the §7 sentence ("set_clearance or resize_feature")."""
    registry = rule_registry()
    assembly = base_assembly(make_spec(params={"d_peg_mm": 10.2, "d_hole_mm": 10.0}))
    policy = RulePolicy(registry=registry)

    plan = propose(
        policy,
        assembly,
        cert("insufficient_clearance", clearance_mm=-0.1),
        cert("geometric_interference", volume_mm3=0.8),
    )

    assert plan is not None
    assert plan.operator == "set_clearance"
    assert spec_of(registry.apply(assembly, plan)).constraints.in_band(
        spec_of(registry.apply(assembly, plan)).clearance_mm
    )


def test_insertion_path_blocked_opens_the_channel_before_guessing_a_pose():
    """The ladder puts the number before the guess: opening the cavity can only
    widen the channel, while a tilt can make the path worse — and on the real
    cascade it did (T1's dovetail fixture went from "path blocked" to "path blocked
    and interfering" after one tilt)."""
    registry = rule_registry()
    assembly = base_assembly()
    policy = RulePolicy(registry=registry)
    code = cert("insertion_path_blocked")

    plan = propose(policy, assembly, code)

    assert plan is not None
    assert plan.operator == "resize_feature"
    # A blocked path wants the loosest fit the band allows, not the tightest.
    assert plan.params["delta_mm"] == pytest.approx(1.2, abs=1e-5)  # (0.60 - 0.0) / 0.5
    after_open = policy.propose(
        registry.apply(assembly, plan),
        [code],
        [failed(plan, "insertion_path_blocked")],
        Budget(rounds=3, geom_calls=100, sim_calls=20),
    )
    assert after_open is not None
    assert after_open.operator == "rotate_part"
    assert after_open.targets == ("insert",)  # the insert moves, never the body
    assert after_open.params["delta_deg"] == [15.0, 0.0, 0.0]


def test_the_pose_ladders_can_never_walk_back_to_a_visited_pose():
    """A ladder whose rungs cancel hands the loop a geometry cycle, which ends the
    episode: `(+2, -2)` on the current pose did exactly that on the real cascade.
    Each rung's magnitude exceeds the sum of the earlier ones in its axis, so the
    cumulative poses are pairwise distinct."""
    from pah.policy.design_policies import _ROTATE_LADDER, _SHIFT_LADDER

    for ladder in (_SHIFT_LADDER, _ROTATE_LADDER):
        seen = set()
        cumulative = [0.0, 0.0, 0.0]
        for rung in ladder:
            cumulative = [round(a + b, 6) for a, b in zip(cumulative, rung)]
            marker = tuple(cumulative)
            assert marker not in seen, (ladder, marker)
            seen.add(marker)


def test_a_printability_failure_falls_through_to_a_paradigm_switch():
    """No specific rung exists for a wall that is too thin, so the honest move is
    a different connector, not an invented wall thickness."""
    registry = rule_registry()
    assembly = base_assembly()
    policy = RulePolicy(registry=registry)

    plan = propose(policy, assembly, cert("below_min_wall"))

    assert plan is not None
    assert plan.operator == "switch_connector"
    assert plan.params["kind"] != spec_of(assembly).interface.kind
    assert plan.params["params"] == {}  # translated by the operator, not invented here
    assert spec_of(registry.apply(assembly, plan)).interface.kind == plan.params["kind"]


# --------------------------------------------------------------------------- #
# RulePolicy: the three properties the module claims
# --------------------------------------------------------------------------- #
def test_a_plan_that_already_failed_is_not_proposed_again():
    registry = rule_registry()
    assembly = base_assembly()  # clearance 0.0, mid is 0.35
    policy = RulePolicy(registry=registry)
    code = cert("insufficient_clearance", clearance_mm=0.0)

    first = propose(policy, assembly, code)
    second = propose(policy, assembly, code, history=[failed(first)])

    assert first is not None and second is not None
    assert second.operator == "resize_feature"  # the next rung, not the same plan
    assert (second.operator, second.params) != (first.operator, first.params)


def test_the_policy_returns_none_when_the_action_space_is_exhausted():
    """A registry with only the two clearance operators: once they are tried,
    nothing legal is left, and `None` is the right answer."""
    from pah.policy.design_ops import make_design_operators

    registry = OperatorRegistry()
    for operator in make_design_operators():
        if operator.id in {"set_clearance", "resize_feature"}:
            registry.register(operator)
    assembly = base_assembly()
    policy = RulePolicy(registry=registry)
    code = cert("insufficient_clearance", clearance_mm=0.0)

    first = propose(policy, assembly, code)
    second = propose(policy, assembly, code, history=[failed(first)])
    third = propose(policy, assembly, code, history=[failed(first), failed(second)])

    assert {first.operator, second.operator} == {"set_clearance", "resize_feature"}
    assert third is None


def test_no_evidence_is_no_action():
    registry = rule_registry()
    policy = RulePolicy(registry=registry)

    assert propose(policy, base_assembly()) is None


def test_environment_evidence_is_not_a_repair_trigger():
    registry = rule_registry()
    policy = RulePolicy(registry=registry)
    crashed = FailureCertificate(
        code="physics_error",
        family="environment",
        locus={"interface": "iface_0"},
        magnitude={"error": "MuJoCo exploded"},
        observed_by="physics_settle",
    )

    assert propose(policy, base_assembly(), crashed) is None


def test_a_dataset_plane_assembly_gets_no_design_action():
    from pah.core.model import Assembly

    registry = rule_registry()
    policy = RulePolicy(registry=registry)
    dataset = Assembly(
        id="asm_case",
        version=0,
        parts=({"id": "p0", "geom": {"hash": "h0"}},),
        joints=({"id": "j0", "a": "p0", "b": "p1"},),
        connectors=(),
        constraints={},
    )

    assert propose(policy, dataset, cert("insufficient_clearance")) is None


def test_a_design_already_at_the_target_is_not_touched():
    """No-op suppression: a clearance move that would change nothing is not a
    repair, so both clearance rungs are skipped and the paradigm gets the turn."""
    registry = rule_registry()
    at_mid = base_assembly(
        make_spec(params={"d_peg_mm": 10.0, "d_hole_mm": 10.7})  # clearance 0.35 = band mid
    )
    policy = RulePolicy(registry=registry)

    plan = propose(policy, at_mid, cert("insufficient_clearance"))

    assert plan is not None
    assert plan.operator not in {"set_clearance", "resize_feature"}
    repaired = registry.apply(at_mid, plan)
    assert spec_of(repaired).content_key() != spec_of(at_mid).content_key()


def test_every_rule_produces_an_applicable_plan_on_every_kind():
    """The property behind "it never invents an action": for every certificate code
    the ladder knows, on every connector kind, the plan either does not exist or
    the guarded registry accepts it and the design really changes."""
    registry = rule_registry()
    policy = RulePolicy(registry=registry)
    codes = [
        "insufficient_clearance",
        "excessive_clearance",
        "geometric_interference",
        "insertion_path_blocked",
        "below_min_wall",
        "unsupported_overhang",
        "joint_unstable",
        "dynamic_penetration",
    ]

    for kind in KINDS:
        assembly = base_assembly(make_spec(kind))
        for code in codes:
            plan = propose(policy, assembly, cert(code))
            if plan is None:
                continue
            assert expressible(kind, plan.operator)
            repaired = registry.apply(assembly, plan)
            assert spec_of(repaired).content_key() != spec_of(assembly).content_key(), (
                kind,
                code,
                plan.operator,
            )


def test_a_dovetail_never_gets_a_resize_insert_plan():
    """The registry cannot answer "is this expressible"; the policy must ask.

    Forced with a ladder that *starts* at `resize_insert`, which is the only way
    to see the filter: the dovetail has no insert-side number, so the rung is
    skipped and the paradigm switch takes the turn.
    """
    registry = rule_registry()
    assembly = base_assembly(make_spec("dovetail"))
    policy = RulePolicy(
        registry=registry,
        ladder={"geometric_interference": ("resize_insert",)},
    )

    plan = propose(policy, assembly, cert("geometric_interference"))

    assert plan is not None
    assert plan.operator == "switch_connector"


# --------------------------------------------------------------------------- #
# RulePolicy: the ladder is data
# --------------------------------------------------------------------------- #
def test_the_fallback_tail_is_appended_to_every_code():
    policy = RulePolicy(registry=rule_registry())

    for code in DEFAULT_LADDER:
        assert policy.ladder_for(code)[-len(FALLBACK_LADDER) :] == FALLBACK_LADDER


def test_a_config_file_can_override_one_row(tmp_path):
    path = tmp_path / "pah.toml"
    path.write_text(
        '[policy.rule.ladder]\ninsufficient_clearance = ["resize_feature"]\n',
        encoding="utf-8",
    )
    ladder = load_ladder(path)
    registry = rule_registry()
    policy = build("rule", registry=registry, config_path=path)

    plan = propose(policy, base_assembly(), cert("insufficient_clearance"))

    assert ladder["insufficient_clearance"] == ("resize_feature",)
    assert plan is not None and plan.operator == "resize_feature"
    assert policy.ladder["excessive_clearance"] == DEFAULT_LADDER["excessive_clearance"]


def test_a_config_file_naming_an_unregistered_operator_is_refused_at_load(tmp_path):
    path = tmp_path / "pah.toml"
    path.write_text(
        '[policy.rule.ladder]\ninsufficient_clearance = ["move_vertex"]\n',
        encoding="utf-8",
    )

    with pytest.raises(KeyError, match="move_vertex"):
        load_ladder(path)


def test_band_target_places_the_edges_just_inside_the_band():
    """Aims at the edges, but not *on* them — the round trip through the kind's own
    parameter is not float-exact (`_EDGE_INSET_MM`)."""
    constraints = make_spec().constraints

    low, high, mid = (
        band_target(constraints, "low"),
        band_target(constraints, "high"),
        band_target(constraints, "mid"),
    )

    assert low == pytest.approx(0.10, abs=1e-5)
    assert high == pytest.approx(0.60, abs=1e-5)
    assert mid == pytest.approx(0.35)
    assert constraints.min_clearance_mm < low < high < constraints.max_clearance_mm
    for value in (low, mid, high):
        assert constraints.in_band(value)


def test_band_target_falls_back_to_the_middle_of_a_narrow_band():
    spec = make_spec(min_c=0.10, max_c=0.1000001)
    assert band_target(spec.constraints, "low") == pytest.approx(0.10000005)


# --------------------------------------------------------------------------- #
# RandomPolicy
# --------------------------------------------------------------------------- #
def test_random_is_reproducible_from_the_snapshot_and_the_round_index():
    registry = rule_registry()
    assembly = base_assembly()
    code = cert("insufficient_clearance")

    first = propose(RandomPolicy(registry=registry, seed=7), assembly, code)
    second = propose(RandomPolicy(registry=registry, seed=7), assembly, code)

    assert first is not None and second is not None
    assert (first.operator, first.params) == (second.operator, second.params)


def test_random_only_draws_from_the_applicable_set():
    registry = rule_registry()
    assembly = base_assembly()
    code = cert("insertion_path_blocked")
    applicable = set(registry.applicable(assembly, [code]))

    for seed in range(20):
        plan = propose(RandomPolicy(registry=registry, seed=seed), assembly, code)
        assert plan is not None
        assert plan.operator in applicable


def test_random_plans_always_pass_the_guards():
    registry = rule_registry()

    for kind in KINDS:
        assembly = base_assembly(make_spec(kind))
        for code in ("insufficient_clearance", "geometric_interference", "insertion_path_blocked"):
            for seed in range(6):
                plan = propose(RandomPolicy(registry=registry, seed=seed), assembly, cert(code))
                if plan is None:
                    continue
                registry.apply(assembly, plan)  # raises if the plan is not legal


def test_random_never_repeats_a_plan_visible_in_the_history():
    registry = rule_registry()
    assembly = base_assembly()
    code = cert("insufficient_clearance")
    first = propose(RandomPolicy(registry=registry, seed=3), assembly, code)
    assert first is not None

    for seed in range(12):
        again = propose(
            RandomPolicy(registry=registry, seed=seed),
            assembly,
            code,
            history=[failed(first)],
        )
        assert again is None or again.id != first.id


def test_random_gives_up_when_its_sample_budget_is_spent():
    """The dedupe is bounded by `samples`; when every draw is taken it says `None`
    rather than spending the round on a plan already known to fail."""
    registry = rule_registry()

    exhausted = RandomPolicy(registry=registry, seed=4, samples=0)

    assert propose(exhausted, base_assembly(), cert("insufficient_clearance")) is None


def test_random_returns_nothing_without_evidence():
    assert propose(RandomPolicy(registry=rule_registry()), base_assembly()) is None


# --------------------------------------------------------------------------- #
# The factory
# --------------------------------------------------------------------------- #
def test_build_returns_the_named_policy():
    registry = rule_registry()

    assert build("rule", registry=registry).name == "rule"
    assert build("random", registry=registry, seed=2).name == "random"
    assert build("random", registry=registry, seed=2).seed == 2


def test_build_refuses_an_unknown_policy_name():
    with pytest.raises(ValueError, match="unknown policy"):
        build("best-of-n", registry=rule_registry())


def test_build_refuses_llm_without_a_client():
    """A missing provider is a configuration error, not an abstention: returning
    `None` here would report "no applicable action" for a run that never asked a
    model anything."""
    with pytest.raises(ProviderError):
        build("llm", registry=rule_registry())


# --------------------------------------------------------------------------- #
# The loop: rounds spent and actions proposed
# --------------------------------------------------------------------------- #
class FakeEngine:
    """Enough engine to run `RepairLoop`; scope and budget accounting are the real
    engine's job and are tested there."""

    def __init__(self) -> None:
        self.rounds: list[Round] = []
        self.ended: list[tuple[str, str]] = []

    def begin_round(self, ticket_id: str, plan: RepairPlan) -> Round:
        round_ = Round(id=f"r{len(self.rounds)}", index=len(self.rounds), plan=plan)
        self.rounds.append(round_)
        return round_

    def stage_round(self, round_id: str, stage: str) -> None:
        next(item for item in self.rounds if item.id == round_id).stage = stage

    def close_round(
        self, round_id, *, verdict, evidence, cost, artifacts, edit_cost=0.0, preserve_ratio=1.0
    ) -> Round:
        round_ = next(item for item in self.rounds if item.id == round_id)
        round_.verdict = verdict
        round_.evidence = list(evidence)
        round_.cost = cost
        round_.artifacts = dict(artifacts)
        return round_

    def end_ticket(self, ticket_id: str, outcome: str, reason: str = "") -> Ticket:
        self.ended.append((outcome, reason))
        return ticket(3)


class DesignChecks:
    """What `fit_clearance` + `interface_interference` will observe, in cascade order.

    Cheapest first (the fit) and then the boolean overlap, which is the order
    `DESIGN_VERIFIER_IDS` fixes — so a rule that depends on a different arrival
    order is caught here rather than in the eval.
    """

    def __init__(self) -> None:
        self.calls = 0

    def check(self, assembly) -> list[FailureCertificate]:
        self.calls += 1
        spec = spec_of(assembly)
        clearance = spec.clearance_mm
        band = spec.constraints
        found: list[FailureCertificate] = []
        # The fit is cheap and runs first (`DESIGN_VERIFIER_IDS`), so a fit
        # certificate outranks the boolean overlap in the evidence list.
        if not band.in_band(clearance):
            code = (
                "insufficient_clearance"
                if clearance < band.min_clearance_mm
                else "excessive_clearance"
            )
            found.append(cert(code, clearance_mm=round(clearance, 6)))
        if clearance < 0.0:
            found.append(cert("geometric_interference", volume_mm3=round(-clearance, 6)))
        return found


def ticket(rounds: int = 3) -> Ticket:
    return Ticket(
        id="t-design",
        lineage="spec:demo",
        mandate=Mandate(
            strength="explicit",
            basis={"via": "test"},
            scope_parts=("body", "insert"),
            scope_operations=("set_clearance",),
            frozen_parts=(),
            budget=Budget(rounds=rounds, geom_calls=50, sim_calls=10),
            policy_version="pv-design-rule",
        ),
        created_at="2026-01-01T00:00:00",
    )


def run_loop(assembly, policy, registry, *, rounds: int = 3):
    engine = FakeEngine()
    verifiers = DesignChecks()
    loop = RepairLoop(engine=engine, verifiers=verifiers, registry=registry, policy=policy)
    result = loop.run(ticket(rounds), assembly)
    return result, engine


@pytest.mark.parametrize("kind", KINDS)
def test_the_rule_policy_closes_a_fit_failure_in_one_round(kind):
    registry = rule_registry()
    assembly = base_assembly(make_spec(kind))  # every kind starts at or below the band

    result, engine = run_loop(assembly, RulePolicy(registry=registry), registry)

    assert result.verdict == "repaired"
    assert result.rounds == 1
    assert engine.rounds[0].verdict == "pass"
    assert engine.rounds[0].plan.operator == "set_clearance"
    assert spec_of(result.assembly).constraints.in_band(spec_of(result.assembly).clearance_mm)


def test_the_rule_policy_closes_the_acceptance_interference_case():
    """`d_peg=10.2, d_hole=10.0` — the case §7 of the contract asks for."""
    registry = rule_registry()
    assembly = base_assembly(make_spec(params={"d_peg_mm": 10.2, "d_hole_mm": 10.0}))

    result, engine = run_loop(assembly, RulePolicy(registry=registry), registry)

    assert result.verdict == "repaired"
    assert result.rounds == 1
    assert engine.rounds[0].plan.operator == "set_clearance"
    assert spec_of(result.assembly).clearance_mm == pytest.approx(0.35)
    assert result.certificates == ()


def test_the_rule_policy_reports_infeasible_rather_than_spending_the_budget():
    registry = OperatorRegistry()  # no operators at all: nothing is legal
    result, engine = run_loop(base_assembly(), RulePolicy(registry=registry), registry)

    assert result.verdict == "infeasible"
    assert result.rounds == 0
    assert engine.rounds == []
    assert engine.ended == [("infeasible", "no applicable action")]


def test_the_random_policy_also_closes_the_loop_without_illegal_plans():
    registry = rule_registry()
    assembly = base_assembly()

    result, engine = run_loop(assembly, RandomPolicy(registry=registry, seed=11), registry, rounds=4)

    assert result.verdict in {"repaired", "exhausted", "infeasible"}
    assert result.rounds <= 4
    assert all(round_.verdict != "error" for round_ in engine.rounds), [
        (round_.plan.operator, round_.verdict) for round_ in engine.rounds
    ]


def test_both_baselines_are_interchangeable_at_the_loop_boundary():
    """The loop reads only `name` and `propose`; that is what makes the
    equal-budget comparison in the eval meaningful."""
    registry = rule_registry()
    for policy in (RulePolicy(registry=registry), RandomPolicy(registry=registry, seed=0)):
        result, _engine = run_loop(base_assembly(), policy, registry, rounds=2)
        assert hasattr(policy, "name") and isinstance(policy.name, str)
        assert result.ticket_id == "t-design"
