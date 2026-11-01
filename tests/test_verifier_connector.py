"""Contract tests for `connector_feasible` (dev doc §4.8, contract §B).

Two layers, tested separately because they fail for different reasons:

* **translation** — `classify` / `certificates_for`. Pure, instant, and where the
  149-case "anchor gap filed as a geometry bug" trap lives.
* **the real thing** — the fixtures actually run `connector.build` on real
  geometry, so they cost ~25 s. That is the point: §4.8 makes the fixture pass a
  runtime gate, and a gate nobody ever opens is decoration.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from pah.tools.verifiers import Registry
from pah.tools.verifiers import connector_feasible as CF

# --------------------------------------------------------------------------- #
# fixtures that never touch the disk
# --------------------------------------------------------------------------- #
FAIL_CASES = [
    # (reason, detail, expected code) — every row of contract §B, one each.
    ("unreatable_too_small", "r_max=0.1789mm(vs r_min 0.25)", "unreatable_too_small"),
    ("host_geometry_unreliable", "a:decimate_ok=False,vol_err=3.1%", "host_geometry_unreliable"),
    ("bool_failed", "no_manifold3d", "no_manifold3d"),
    ("bool_failed", "union_status(NoError_Degenerate,faces=0)", "bool_failed"),
    ("open_after_bool", "a(bnd1=3,nm=2)", "open_after_bool"),
    ("peg_detached", "fragmented_a(share=0.9120<0.9990,ncomp=3/1)", "peg_detached"),
    ("socket_breaks_wall", "socket_no_effect(dv_b=0,expected=1e-9)", "socket_breaks_wall"),
    (
        "degenerate_geometry",
        "degenerate_other(gap_too_wide_for_wall(gap=5.0052mm,need=7.2764mm))",
        "gap_too_wide",
    ),
    ("degenerate_geometry", "degenerate_other(no_material_hit(a=1,b=0))", "no_material_hit"),
    ("degenerate_geometry", "degenerate_thin_wall(wall_too_thin(t_b=0.3mm))", "degenerate_thin_wall"),
    ("degenerate_geometry", "degenerate_tangent(peg_tip_inside_a)", "degenerate_tangent"),
    (
        "degenerate_geometry",
        "degenerate_at_patch_edge(socket_cap_not_outside_b)",
        "degenerate_at_patch_edge",
    ),
    ("degenerate_geometry", "peg_tip_inside_a(tip_t=1e-4)", "degenerate_other"),
    ("degenerate_geometry", "degenerate_other(ray_parity_odd(a=1,b=0))", "degenerate_other"),
]


def _joint(edge_id: int = 1) -> dict:
    """One synthetic `case.json` joint, all `*_mm` fields at the 225 mm scale."""
    return {
        "id": f"j{edge_id}",
        "edge_id": edge_id,
        "eid": edge_id,
        "a": "p0",
        "b": "p1",
        "a_index": 0,
        "b_index": 1,
        "center_mm": [0.0, 0.0, 0.0],
        "anchor_mm": [1000.0, 2000.0, 3000.0],
        "n_ab": [0.0, 0.0, 1.0],
        "half_w_mm": 3000.0,
        "half_h_mm": 2000.0,
        "r_inplane_max_mm": 1500.0,
        "depth_max_mm": 900.0,
        "t_a_mm": 1500.0,
        "t_b_mm": 1200.0,
        "limiter": "in_plane",
        "wall_irrelevant": True,
        "anchor_gap_mm": 0.01,
        "measurement_lod": "fine",
        "interface_type": "coincident",
        "edge_quality": "good",
        "feasible": True,
        "thickness_failed": False,
        "n_parts": 2,
    }


def _case(*joints: dict) -> CF.Case:
    return CF.Case(
        case_id="asm_synthetic",
        parts=(
            {"id": "p0", "pre_decimate": "part_000.ply"},
            {"id": "p1", "pre_decimate": "part_001.ply"},
        ),
        joints=tuple(joints),
        scale_mm=CF.SCALE_MM,
        baseline=dict(CF.BASELINE),
    )


def _box(size: float = 1.0):
    """A stand-in mesh: only `.bounds` is ever read by the code under test."""
    return types.SimpleNamespace(bounds=(np.zeros(3), np.full(3, size)))


@pytest.fixture
def no_geometry(monkeypatch):
    """Replaces the two I/O seams so a test can drive the whole pipeline instantly."""
    monkeypatch.setattr(CF, "_load_parts", lambda case: ((_box(), _box()), 1.0))

    def fake_build(*, parts, joint, edge, theta, unit_mm, host_quality):
        return {"ok": False, "reason": "degenerate_geometry",
                "detail": "degenerate_other(no_material_hit(a=1,b=0))"}

    monkeypatch.setattr(CF, "_call_build", fake_build)
    return fake_build


# --------------------------------------------------------------------------- #
# translation: reason -> code (contract §B)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(("reason", "detail", "expected"), FAIL_CASES)
def test_every_reason_maps_to_its_registered_code(reason, detail, expected):
    code, mechanism = CF.classify({"reason": reason, "detail": detail})
    assert code == expected
    assert code in CF.CODES, f"{code} is not in the closed vocabulary"
    assert mechanism, "the source's own name for the cause must survive"


def test_promoted_causes_beat_the_degen_subcode():
    """`degenerate_other(gap_too_wide_for_wall(...))` is an anchor gap, not geometry."""
    result = {
        "reason": "degenerate_geometry",
        "degen_code": "degenerate_other",
        "detail": "degenerate_other(gap_too_wide_for_wall(gap=5.0mm,need=7.3mm))",
    }
    code, mechanism = CF.classify(result)
    assert (code, mechanism) == ("gap_too_wide", "gap_too_wide_for_wall")
    assert CF._FAMILY[code] == "assembly"


def test_families_match_the_documented_split():
    """§B's 族 column: three assembly causes, three manufacturing ones, rest geometry."""
    assert CF._FAMILY["gap_too_wide"] == "assembly"
    assert CF._FAMILY["no_material_hit"] == "assembly"
    assert CF._FAMILY["unreatable_too_small"] == "assembly"
    for code in ("open_after_bool", "peg_detached", "socket_breaks_wall"):
        assert CF._FAMILY[code] == "manufacturing"
    for code in ("degenerate_thin_wall", "degenerate_tangent",
                 "degenerate_at_patch_edge", "degenerate_other",
                 "bool_failed", "no_manifold3d", "host_geometry_unreliable"):
        assert CF._FAMILY[code] == "geometry"


def test_a_passing_edge_yields_no_certificate():
    assert CF.certificates_for(_joint(), {"ok": True, "reason": "", "detail": ""}, 1500.0) == []


def test_certificate_shape_is_usable_by_the_core_model():
    certificates = CF.certificates_for(
        _joint(7),
        {"reason": "unreatable_too_small", "detail": "r_max=0.17mm", "r_cap": 1e-4},
        1500.0,
    )
    certificate = certificates[0]
    assert certificate.code == "unreatable_too_small"
    assert certificate.family == "assembly"
    assert certificate.locus["parts"] == ["p0", "p1"]
    assert certificate.locus["joint"] == "j7"
    assert certificate.locus["edge"] == 7
    assert certificate.magnitude["unit"] == "mm"
    assert certificate.magnitude["threshold"] == CF.R_MIN_MM
    assert certificate.observed_by.startswith("connector_feasible@")
    assert certificate.confidence == 1.0
    assert certificate.hypotheses == ()
    assert certificate.evidence_ref == "edges_v2.json#id=7;t_b=" + CF.T_B_SOURCE


# --------------------------------------------------------------------------- #
# unknown reasons must never be dropped (task hard rule)
# --------------------------------------------------------------------------- #
def test_unknown_reason_is_carried_not_dropped(no_geometry, monkeypatch):
    """A reason the table has never seen becomes evidence, not silence."""
    detail = "mystery_mechanism(wobble=3.5)"

    def surprise(**kwargs):
        return {"ok": False, "reason": "brand_new_reason", "detail": detail}

    monkeypatch.setattr(CF, "_call_build", surprise)
    verifier = CF.ConnectorFeasible()
    certificates = verifier.check(_case(_joint(11)))

    assert len(certificates) == 1
    certificate = certificates[0]
    assert certificate.code == "unknown_reason"
    assert certificate.code in verifier.spec.codes
    # the escape hatch is registered so it can be *emitted*, not so it can be trusted
    assert certificate.confidence == 0.0
    carried = " ".join(str(h["text"]) for h in certificate.hypotheses)
    assert "brand_new_reason" in carried
    assert detail in carried
    assert all(h["untested"] is True for h in certificate.hypotheses)


def test_known_reason_is_not_marked_unknown(no_geometry):
    certificates = CF.ConnectorFeasible().check(_case(_joint()))
    assert [c.code for c in certificates] == ["no_material_hit"]
    assert certificates[0].confidence == 1.0


def test_a_raising_source_is_blamed_on_the_adapter_not_the_geometry(monkeypatch):
    """Exception translation must not disguise an adapter bug as `bool_failed`."""
    def exploding():
        class Boom:
            @staticmethod
            def build(*args, **kwargs):
                raise RuntimeError("adapter wired the arguments wrong")

        return Boom()

    monkeypatch.setattr(CF, "_connector", exploding)
    result = CF._call_build(
        parts=(None, None),
        joint=_joint(),
        edge=CF.to_edge_dict(_joint(), 1500.0),
        theta=(0.0, 0.0, 0.0, 0.0, 1.0, 1e-3, 2e-3),
        unit_mm=1500.0,
        host_quality={"a": None, "b": None},
    )
    assert result["ok"] is False
    assert result["reason"] == "adapter_raised"
    assert "adapter wired the arguments wrong" in result["detail"]
    code, _ = CF.classify(result)
    assert code == "unknown_reason"  # carried, not silently re-labelled


# --------------------------------------------------------------------------- #
# OOM retry: the machine's failure is not the design's failure
# --------------------------------------------------------------------------- #
_OK = {"ok": True, "reason": "", "detail": "", "r": 1e-4}


def _oom_wrapped(mib: int = 56) -> dict:
    """Exactly the shape `connector.py:652-659` returns when `manifold3d` is starved.

    That catch-all wraps the `MemoryError` into a *string*, so a plain
    `except MemoryError` around the call would never see it — the gold rows lose
    five edges to this shape (`asm_000097` ids 912/916/917/919/920).
    """
    return {
        "ok": False,
        "reason": "bool_failed",
        "detail": f"exception:MemoryError(Unable to allocate {mib}.0 MiB for an array)",
    }


class _ScriptedBuild:
    """A `connector.build` whose every call is scripted in order."""

    def __init__(self, *results):
        self._results = list(results)
        self.calls = 0

    def build(self, *args, **kwargs):
        self.calls += 1
        item = self._results[min(self.calls, len(self._results)) - 1]
        if isinstance(item, BaseException):
            raise item
        return item


@pytest.fixture
def retry(monkeypatch):
    """Drives `_call_build` against a scripted build and records the back-off."""
    CF.reset_oom_stats()
    slept: list[float] = []
    monkeypatch.setattr(CF, "_sleep", lambda seconds: slept.append(seconds))

    def install(*results) -> _ScriptedBuild:
        build = _ScriptedBuild(*results)
        monkeypatch.setattr(CF, "_connector", lambda: build)
        return build

    yield install, slept
    CF.reset_oom_stats()  # the counters are module state; leave them as found


def _build() -> dict:
    joint = _joint()
    return CF._call_build(
        parts=(None, None),
        joint=joint,
        edge=CF.to_edge_dict(joint, 1500.0),
        theta=(0.0, 0.0, 0.0, 0.0, 1.0, 1e-3, 2e-3),
        unit_mm=1500.0,
        host_quality={"a": None, "b": None},
    )


def test_a_raised_memory_error_is_retried_and_the_second_call_answers(retry):
    """The retry path must be *taken*, not merely present — one raise, then ok."""
    install, slept = retry
    build = install(MemoryError("Unable to allocate 56.0 MiB"), _OK)

    result = _build()

    assert result["ok"] is True
    assert build.calls == 2
    assert slept == [1.0], "the first retry waits 1 s, and only one retry was needed"
    assert CF.oom_stats() == {"retried": 1, "oom": 0}


def test_the_string_wrapped_memory_error_is_retried_too(retry):
    """The shape the source actually produces: `bool_failed/exception:MemoryError`."""
    install, slept = retry
    build = install(_oom_wrapped(), _OK)

    result = _build()

    assert result["ok"] is True
    assert build.calls == 2
    assert slept == [1.0]
    assert CF.oom_stats() == {"retried": 1, "oom": 0}


def test_a_persistent_oom_becomes_verifier_oom_not_bool_failed(retry):
    """Two retries, then a code of its own — the environment never files as geometry."""
    install, slept = retry
    build = install(_oom_wrapped())

    result = _build()

    assert build.calls == 3, "one attempt plus OOM_RETRIES retries"
    assert slept == [1.0, 3.0]
    assert result["ok"] is False
    assert result["reason"] == "verifier_oom"
    assert "exception:MemoryError" in result["detail"]

    code, mechanism = CF.classify(result)
    assert code == "verifier_oom"
    assert code in CF.CODES
    assert code != "bool_failed"
    assert mechanism
    assert CF.oom_stats() == {"retried": 2, "oom": 1}

    certificate = CF.certificates_for(_joint(912), result, 1500.0)[0]
    assert certificate.code == "verifier_oom"
    assert certificate.code != "bool_failed"
    assert certificate.magnitude == {}
    # A known cause, not an unclassified one: confidence 1.0, no hypotheses.
    assert certificate.confidence == 1.0
    assert certificate.hypotheses == ()


def test_an_ordinary_bool_failure_is_not_retried(retry):
    """Only OOM earns a retry; a `NoError_Degenerate` union is already the answer."""
    install, slept = retry
    build = install(
        {"ok": False, "reason": "bool_failed", "detail": "union_status(NoError_Degenerate,faces=0)"}
    )

    result = _build()

    assert result["reason"] == "bool_failed"
    assert build.calls == 1
    assert slept == []
    assert CF.oom_stats() == {"retried": 0, "oom": 0}


def test_the_verifier_emits_verifier_oom_for_a_starved_edge(monkeypatch):
    """End to end: a starved edge reaches the certificate as `verifier_oom`."""
    CF.reset_oom_stats()
    monkeypatch.setattr(CF, "_load_parts", lambda case: ((_box(), _box()), 1.0))
    monkeypatch.setattr(CF, "_sleep", lambda seconds: None)
    monkeypatch.setattr(CF, "_connector", lambda: _ScriptedBuild(_oom_wrapped()))
    try:
        certificates = CF.ConnectorFeasible().check(_case(_joint(912)))
    finally:
        CF.reset_oom_stats()

    assert [c.code for c in certificates] == ["verifier_oom"]
    assert certificates[0].locus["edge"] == 912


# --------------------------------------------------------------------------- #
# the scale protocol — the one constant that silently changes every number
# --------------------------------------------------------------------------- #
def test_coordinates_and_derived_lengths_use_different_frames():
    """`center_mm` is 150 mm-frame millimetres; `r_*_mm` are 225 mm-scale ones."""
    joint = _joint()
    edge = CF.to_edge_dict(joint, unit_mm=1500.0)
    # coordinates: fixed /1000, independent of unit_mm
    assert edge["anchor"] == [1.0, 2.0, 3.0]
    assert CF.to_edge_dict(joint, unit_mm=1000.0)["anchor"] == [1.0, 2.0, 3.0]
    # derived lengths: /unit_mm, so they follow the scale protocol
    assert edge["r_inplane_max"] == pytest.approx(1.0)
    assert CF.to_edge_dict(joint, unit_mm=1000.0)["r_inplane_max"] == pytest.approx(1.5)
    assert edge["t_b"] == pytest.approx(0.8)


def test_feasibility_ratio_is_scale_invariant():
    """Why the gold set survives a rescale: only `r_cap / r_min` decides (§18.4)."""
    joint = _joint()
    ratios = []
    for rule in (1500.0, 1000.0, 76.23):
        edge = CF.to_edge_dict(joint, unit_mm=rule)
        ratios.append(edge["r_inplane_max"] / (CF.R_MIN_MM / rule))
    assert ratios[0] == pytest.approx(ratios[1])
    assert ratios[1] == pytest.approx(ratios[2])


def test_scale_rule_recovers_the_declared_scale():
    """`225 / longest_side` — the 150 mm geometry with 225 mm-frame thresholds."""
    case = _case(_joint())
    assert CF._unit_mm(case, longest=0.15, rule="scale") == pytest.approx(1500.0)
    assert CF._unit_mm(case, longest=0.15, rule="1000") == 1000.0


# --------------------------------------------------------------------------- #
# §4.8: a verifier that disagrees with its fixtures gets disabled
# --------------------------------------------------------------------------- #
class _Stub:
    """A verifier whose verdict per fixture label is scripted, to exercise the gate."""

    def __init__(self, verifier_id: str, verdicts: dict[str, str], fixtures: dict):
        from pah.tools.verifiers import VerifierSpec

        self.spec = VerifierSpec(
            id=verifier_id, codes=("x",), cost_class="cheap", fixtures=fixtures
        )
        self._verdicts = verdicts

    def case_of(self, item):
        return item

    def check(self, case, assembly=None):
        if self._verdicts[case["label"]] == "pass":
            return []
        return [types.SimpleNamespace(code="x")]


def _stub(verifier_id: str, known_good: str, known_bad: str, **kwargs) -> _Stub:
    fixtures = {
        "known_good": [{"label": "g"}],
        "known_bad": [{"label": "b"}],
    }
    return _Stub(verifier_id, {"g": known_good, "b": known_bad}, fixtures, **kwargs)


def test_sanity_passes_a_verifier_that_agrees_with_itself():
    registry = Registry()
    registry.register(_stub("honest", known_good="pass", known_bad="fail"))
    assert registry.sanity() == {"honest": True}
    assert registry.disabled() == frozenset()


def test_sanity_disables_a_verifier_whose_known_good_fails():
    registry = Registry()
    registry.register(_stub("lying", known_good="fail", known_bad="fail"))
    assert registry.sanity() == {"lying": False}
    assert registry.disabled() == frozenset({"lying"})
    assert "known_good judged negative" in registry.why_disabled("lying")


def test_sanity_disables_a_verifier_that_never_fires():
    registry = Registry()
    registry.register(_stub("blind", known_good="pass", known_bad="pass"))
    assert registry.sanity() == {"blind": False}
    assert "known_bad judged positive" in registry.why_disabled("blind")


def test_sanity_refuses_a_verifier_without_fixtures():
    registry = Registry()
    registry.register(_Stub("bare", {}, {}))
    assert registry.sanity() == {"bare": False}
    assert registry.disabled() == frozenset({"bare"})


def test_registry_rejects_a_duplicate_id():
    registry = Registry()
    stub = _stub("dup", known_good="pass", known_bad="fail")
    registry.register(stub)
    with pytest.raises(ValueError, match="already registered"):
        registry.register(stub)


def test_registry_get_reports_what_is_registered():
    registry = Registry()
    registry.register(_stub("solo", known_good="pass", known_bad="fail"))
    assert registry.ids() == ("solo",)
    assert registry.get("solo").spec.id == "solo"
    with pytest.raises(KeyError, match="no verifier 'ghost'"):
        registry.get("ghost")


# --------------------------------------------------------------------------- #
# the real thing: fixtures against real geometry (≈25 s)
# --------------------------------------------------------------------------- #
requires_geometry = pytest.mark.skipif(
    not CF.GEOMETRY_ROOT.exists() or not CF.EDGE_FILE.exists(),
    reason=f"benchmark data not mounted at {CF.GEOMETRY_ROOT}",
)


def _fixture_params(kind: str):
    return [pytest.param(item, id=item["label"])
            for item in CF.ConnectorFeasible.spec.fixtures[kind]]


@requires_geometry
@pytest.mark.parametrize("item", _fixture_params("known_good"))
def test_known_good_fixtures_are_judged_positive(item):
    verifier = CF.ConnectorFeasible()
    assert verifier.check(verifier.case_of(item)) == []


@requires_geometry
@pytest.mark.parametrize("item", _fixture_params("known_bad"))
def test_known_bad_fixtures_are_judged_negative(item):
    verifier = CF.ConnectorFeasible()
    certificates = verifier.check(verifier.case_of(item))
    assert certificates, f"{item['label']} must be rejected"
    assert certificates[0].code in verifier.spec.codes
    assert certificates[0].code != "unknown_reason", (
        "a fixture must land on a known code, or the table has a hole"
    )


@requires_geometry
def test_registry_sanity_over_the_real_verifier():
    """The gate the kernel runs before trusting a number (§4.8)."""
    registry = Registry()
    registry.register(CF.ConnectorFeasible())
    assert registry.sanity() == {"connector_feasible": True}
    assert registry.disabled() == frozenset()
