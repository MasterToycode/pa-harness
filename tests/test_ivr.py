"""IVR's contract, on synthetic geometry only.

Every case here is built from primitives so the expected number is hand-computed
and the test never depends on the 100-assembly corpus (that is M0's job,
`docs/pah-dev-doc-v1.md` §4.8). The four cases §4.8 names as the reason the
evaluator exists are all present:

- two disjoint parts must read exactly 0, not "unknown";
- a 50 % overlap must equal the hand value to 1e-9;
- a fully nested part is where surface sampling lies, so both levels are pinned
  there (exact and occupancy are right, surface reads half the truth);
- a non-manifold shell must read unknown, with a reason, and never 0.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
import trimesh

from pah.tools import ivr
from pah.tools.ivr import IVRResult, ivr_exact, ivr_probe

#: `docs/internal/adapter-contract.md` §A: geometry is metres, so 1 m = 1000 mm.
UNIT_MM = 1000.0

#: A unit cube is the cheapest closed solid with an exact hand volume.
_SIDE = 1.0


def _cube(side: float = _SIDE, at: tuple[float, float, float] = (0.0, 0.0, 0.0)):
    mesh = trimesh.creation.box(extents=(side, side, side))
    mesh.apply_translation(at)
    return mesh


def _non_manifold_fin():
    """A cube with one extra triangle welded along an existing edge.

    That edge now has three incident faces and the fin leaves two free edges —
    an open, non-manifold shell, the case §4.8 says must never be read as 0.
    """
    box = _cube()
    apex = box.vertices[[0]] + np.array([[0.0, 0.0, 2.5]])
    vertices = np.vstack([box.vertices, apex])
    faces = np.vstack([box.faces, [[0, 1, len(box.vertices)]]])
    return trimesh.Trimesh(vertices, faces, process=False)


def _half_overlap():
    """Two unit cubes sharing exactly half their volume: truth IVR = 0.5 / 2."""
    return [("p0", _cube()), ("p1", _cube(at=(0.5, 0.0, 0.0)))]


# --------------------------------------------------------------------------- #
# 1. disjoint
# --------------------------------------------------------------------------- #
def test_disjoint_boxes_read_exactly_zero():
    """An AABB miss proves an empty intersection, so this pair is *known* 0."""
    parts = [("p0", _cube()), ("p1", _cube(at=(3.0, 0.0, 0.0)))]

    result = ivr_exact(parts)

    assert result.value == 0.0
    assert result.pairs == 1
    assert result.unknown_pairs == []
    assert result.detail["pairs"][0]["method"] == "bbox"


# --------------------------------------------------------------------------- #
# 2. half overlap, hand computation
# --------------------------------------------------------------------------- #
def test_half_overlap_equals_the_hand_value():
    """|A ∩ B| = 0.5, Σ|V| = 2, so IVR = 0.25 — to 1e-9 as §4.8 requires."""
    result = ivr_exact(_half_overlap())

    assert result.unknown_pairs == []
    assert abs(result.value - 0.25) < 1e-9
    assert result.detail["numerator"] == pytest.approx(0.5)
    assert result.detail["denominator"] == pytest.approx(2.0)


# --------------------------------------------------------------------------- #
# 3. full containment — the case that separates the levels
# --------------------------------------------------------------------------- #
def test_full_containment_is_exact_and_occupancy_agrees():
    """A 1³ cube inside a 2³ cube: truth 1 / 9, and occupancy sampling finds it."""
    parts = [("p0", _cube(2.0)), ("p1", _cube(1.0))]

    exact = ivr_exact(parts)
    probe = ivr_probe(parts)

    assert exact.value == pytest.approx(1.0 / 9.0)
    # The sample box is the inner cube's AABB, entirely inside both solids, so
    # occupancy is exact here rather than merely close.
    assert probe.value == pytest.approx(1.0 / 9.0)


def test_surface_sampling_markedly_underestimates_a_nested_part():
    """§4.8's blind spot, pinned: surface mode reads half of a full containment.

    Only the inner part's shell lies inside the outer one, so the evidence
    fraction is 0.5 and the estimate halves the truth. That is why a sampled
    number may never be filed as `ivr`.
    """
    parts = [("p0", _cube(2.0)), ("p1", _cube(1.0))]

    exact = ivr_exact(parts)
    surface = ivr_probe(parts, method="surface", seed=0)

    assert surface.value is not None
    assert surface.value < 0.6 * exact.value, "surface sampling must lose the nested volume"
    assert surface.detail["pairs"][0]["method"] == "surface"


def test_surface_sampling_never_reports_absence_as_zero():
    """No surface of either part inside the other: unknown, because it is also
    what a fully nested pair looks like when only the outer shell is sampled."""
    a = trimesh.creation.icosphere(subdivisions=2, radius=0.5)
    b = trimesh.creation.icosphere(subdivisions=2, radius=0.5)
    b.apply_translation((0.95, 0.95, 0.95))  # AABBs overlap, solids are apart
    parts = [("p0", a), ("p1", b)]

    surface = ivr_probe(parts, method="surface")
    occupancy = ivr_probe(parts, method="occupancy")

    assert surface.value is None
    assert len(surface.unknown_pairs) == 1
    assert "surface_no_evidence" in surface.unknown_pairs[0]
    assert occupancy.value == 0.0, "occupancy can say 0 within its resolution"
    assert occupancy.unknown_pairs == []


# --------------------------------------------------------------------------- #
# 4. non-manifold
# --------------------------------------------------------------------------- #
def test_non_manifold_mesh_is_unknown_not_zero():
    """The open/non-manifold shell is refused before the boolean is trusted."""
    parts = [("p0", _non_manifold_fin()), ("p1", _cube())]

    result = ivr_exact(parts)

    assert result.value is None
    assert result.pairs == 1
    assert len(result.unknown_pairs) == 1
    assert "open_surface" in result.unknown_pairs[0]
    assert result.detail["pairs"][0]["status"] == "unknown"


def test_engine_status_alone_is_enough_to_go_unknown(monkeypatch):
    """manifold3d answers `NotManifold` with a *valid-looking* empty solid.

    Reading its volume would file "unknown" as 0, so the status check is what
    must catch it. The pre-check is bypassed here to exercise exactly that
    branch — with it in place the pair never reaches the engine.
    """
    monkeypatch.setattr(ivr, "_surface_defect", lambda mesh: "")

    result = ivr_exact([("p0", _non_manifold_fin()), ("p1", _cube())])

    assert result.value is None
    assert "boolean:Error.NotManifold" in result.unknown_pairs[0]


# --------------------------------------------------------------------------- #
# 5. pairs narrow the check
# --------------------------------------------------------------------------- #
def test_pairs_parameter_decides_the_denominator():
    """Only the listed pair is checked, and only its parts enter the denominator."""
    parts = [
        ("p0", _cube()),
        ("p1", _cube(at=(0.5, 0.0, 0.0))),
        ("p2", _cube(10.0, at=(100.0, 0.0, 0.0))),  # disjoint, V = 1000
    ]

    narrowed = ivr_exact(parts, pairs=[("p0", "p1")])
    everything = ivr_exact(parts)

    assert narrowed.pairs == 1
    assert narrowed.value == pytest.approx(0.25)
    assert narrowed.detail["denominator"] == pytest.approx(2.0)
    assert set(narrowed.detail["part_volumes"]) == {"p0", "p1"}, "p2 must not dilute"
    # The same numerator over all three parts' volume is ~500x smaller.
    assert everything.pairs == 3
    assert everything.value == pytest.approx(0.5 / 1002.0)


def test_a_pair_listed_twice_is_counted_once():
    """§4.8 forbids summing a pair twice, in either order."""
    parts = _half_overlap()

    once = ivr_exact(parts, pairs=[("p0", "p1")])
    repeated = ivr_exact(parts, pairs=[("p0", "p1"), ("p1", "p0"), ("p1", "p0")])

    assert repeated.pairs == 1
    assert repeated.value == once.value


def test_unknown_pair_ids_are_a_caller_error():
    """A typo in the pair list must raise, not silently shrink the numerator."""
    with pytest.raises(ValueError, match="unknown part"):
        ivr_exact(_half_overlap(), pairs=[("p0", "p9")])


# --------------------------------------------------------------------------- #
# 6. units
# --------------------------------------------------------------------------- #
def test_millimetre_geometry_keeps_the_ratio_and_scales_volume():
    """IVR is dimensionless; the volumes in `detail` follow the input unit.

    Same design built in metres and in millimetres must agree, and the reported
    volumes must differ by exactly `unit_mm ** 3` = 1e9 (adapter contract §A).
    """
    metres = [("p0", _cube(0.1)), ("p1", _cube(0.1, at=(0.05, 0.0, 0.0)))]
    millimetres = [
        ("p0", _cube(100.0)),
        ("p1", _cube(100.0, at=(50.0, 0.0, 0.0))),
    ]

    m = ivr_exact(metres)
    mm = ivr_exact(millimetres)

    assert m.value == pytest.approx(0.25)
    assert mm.value == pytest.approx(m.value)
    assert mm.detail["part_volumes"]["p0"] == pytest.approx(100.0**3)
    assert mm.detail["part_volumes"]["p0"] == pytest.approx(
        m.detail["part_volumes"]["p0"] * UNIT_MM**3
    )


# --------------------------------------------------------------------------- #
# 7. nothing to check
# --------------------------------------------------------------------------- #
def test_empty_input_and_a_single_part_are_vacuously_zero():
    """`value=0.0` here is an empty sum, not a measurement.

    The numerator has no terms and no ratio is formed, which `pairs == 0` states
    outright — so this cannot be confused with "checked and clean". `None` would
    be worse: it would read as "the tool failed".
    """
    for result in (
        ivr_probe([]),
        ivr_exact([]),
        ivr_probe([("p0", _cube())], pairs=[]),
        ivr_exact([("p0", _cube())]),
    ):
        assert isinstance(result, IVRResult)
        assert result.value == 0.0
        assert result.pairs == 0
        assert result.unknown_pairs == []


# --------------------------------------------------------------------------- #
# the three hard rules, as plain assertions
# --------------------------------------------------------------------------- #
def test_levels_can_never_be_confused():
    """§4.8 rule 1 + rule 3: level, field name and the three numbers are attributes."""
    parts = _half_overlap()

    probe = ivr_probe(parts)
    exact = ivr_exact(parts)

    assert probe.level == "probe"
    assert probe.field_name == "penetration_probe"
    assert exact.level == "exact"
    assert exact.field_name == "ivr_exact"
    assert probe.field_name != exact.field_name
    # rule 3: a report needs no `detail` parsing to print all three numbers.
    assert isinstance(probe.value, float)
    assert probe.unknown_pairs == []
    assert probe.pairs == 1


def test_probe_is_reproducible_and_close_to_exact():
    """Same seed, same number; and the estimate stays near the measurement."""
    rotated = _cube()
    rotated.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 6, [0, 0, 1]))
    rotated.apply_translation((0.35, 0.15, 0.0))
    parts = [("p0", _cube()), ("p1", rotated)]
    exact = ivr_exact(parts).value
    assert exact is not None and exact > 0.0

    first = ivr_probe(parts, seed=7)
    second = ivr_probe(parts, seed=7)

    assert first.value == second.value, "the sampling must be a function of seed alone"
    for seed in (0, 7, 12345):
        value = ivr_probe(parts, seed=seed).value
        assert value is not None
        assert abs(value - exact) / exact < 0.05


def test_probe_is_fast_on_twenty_thousand_face_parts():
    """§4.8 wants the fast level in seconds at 2 x 20k faces.

    The bound is loose on purpose (measured: ~1 s at the default 512 samples).
    It exists to catch a sampling-strategy regression — re-building a ray index
    per point, or dropping the bounding-box cull, costs 10-100x, not 20 %.
    """
    a = trimesh.creation.icosphere(subdivisions=5, radius=0.5)
    b = trimesh.creation.icosphere(subdivisions=5, radius=0.5)
    b.apply_translation((0.3, 0.1, 0.0))
    assert len(a.faces) >= 20_000
    parts = [("p0", a), ("p1", b)]

    started = time.perf_counter()
    probe = ivr_probe(parts)
    elapsed = time.perf_counter() - started

    assert elapsed < 10.0, f"probe took {elapsed:.1f}s at 20k faces"
    assert probe.value is not None
    assert abs(probe.value - ivr_exact(parts).value) / probe.value < 0.10
    assert probe.detail["samples"] == 512
