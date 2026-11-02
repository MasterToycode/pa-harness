"""The bridge rule: a *span* limit, not a smaller area threshold.

Ruling of 2026-11-02 (audit P2-2). A 0.3 mm-wide step across a 6.4 mm bore is printed
by the first layer as a bridge; judging it as an unsupported overhang made every
decomposed piece "unprintable" (6.148 mm2 per part) for a feature no repair operator
can act on. False positives are as much an audit failure as overclaims — the harness
must not say a printable part is unprintable any more than the reverse.

The rule has two sides and this file pins both:

* **what it forgives** — a connected ceiling whose surface normal points down, whose
  angle is below the process's overhang limit, and whose *span* (the narrow dimension
  of what it covers) is at most `ProcessProfile.bridge_mm` is a bridge, not an
  overhang;
* **what it must still report** — a ceiling wider than the process can cross, even if
  its area is small. A rule that only measured area would pass a 10 x 10 mm recess
  that no FDM machine bridges, which is why the unit tests below compare ceilings of
  *similar area and different span* rather than "small passes, big fails".

Resin is the second half of the argument: `sla.bridge_mm` is 0, not 8, because a
vat of liquid cannot be spanned at all. The same geometry that is fine on FDM must
still fire on SLA.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import trimesh
from trimesh.boolean import difference
from trimesh.creation import box, cylinder

from pah.design.process import PROCESS_PROFILES
from pah.tools.verifiers.printability import (
    Ceiling,
    Printability,
    ceiling_regions,
    overhang_faces,
)

SECTIONS = 64
FDM_ANGLE = PROCESS_PROFILES["fdm"].overhang_deg
FDM_BRIDGE = PROCESS_PROFILES["fdm"].bridge_mm


# --------------------------------------------------------------------------- #
# Solid fixtures: four ceilings whose spans decide the answer
# --------------------------------------------------------------------------- #
def _plate(width: float = 40.0, depth: float = 40.0, height: float = 12.0) -> trimesh.Trimesh:
    return box(extents=(width, depth, height))


def _ring_cutter(outer_r: float, inner_r: float, height: float, z: float) -> trimesh.Trimesh:
    outer = cylinder(radius=outer_r, height=height, sections=SECTIONS)
    outer.apply_translation((0.0, 0.0, z))
    inner = cylinder(radius=inner_r, height=height * 2.0, sections=SECTIONS)
    inner.apply_translation((0.0, 0.0, z))
    return difference([outer, inner], engine="manifold")


def annular_groove(width_mm: float = 0.3, bore_mm: float = 6.4) -> trimesh.Trimesh:
    """A 0.3 mm-wide groove cut up into the bottom face, opening on a 6.4 mm bore.

    This is the shape that condemned the decomposition: the ceiling is a thin ring,
    and the printer crosses the ring's width, not the bore's diameter.
    """
    radius = bore_mm / 2.0
    cutter = _ring_cutter(radius, radius - width_mm, 6.0, -3.0)
    return difference([_plate(), cutter], engine="manifold")


def blind_bore(diameter_mm: float = 6.4) -> trimesh.Trimesh:
    """A blind bore cut up into the bottom face: a disc ceiling of `diameter_mm`."""
    cutter = cylinder(radius=diameter_mm / 2.0, height=6.0, sections=SECTIONS)
    cutter.apply_translation((0.0, 0.0, -3.0))
    return difference([_plate(), cutter], engine="manifold")


def recess(
    size_mm: tuple[float, float, float] = (10.0, 10.0, 3.0),
    plate_mm: tuple[float, float, float] = (40.0, 40.0, 12.0),
) -> trimesh.Trimesh:
    """A rectangular recess up into the bottom face: a `size_mm[0]`-wide ceiling."""
    cutter = box(extents=size_mm)
    cutter.apply_translation((0.0, 0.0, -plate_mm[2] / 2.0 + size_mm[2] / 2.0))
    return difference([_plate(*plate_mm), cutter], engine="manifold")


def both_ceilings() -> trimesh.Trimesh:
    """A part with a 6.4 mm bore and a 10 x 10 mm recess: bridged and reported at once."""
    bore = cylinder(radius=3.2, height=6.0, sections=SECTIONS)
    bore.apply_translation((-12.0, 0.0, -3.0))
    pocket = box(extents=(10.0, 10.0, 3.0))
    pocket.apply_translation((12.0, 0.0, -4.5))
    return difference([_plate(60.0, 40.0), bore, pocket], engine="manifold")


# --------------------------------------------------------------------------- #
# The criterion itself
# --------------------------------------------------------------------------- #
def test_the_short_span_ceiling_is_one_connected_region():
    """Connectedness, not per-triangle size: 128 facets are one ceiling."""
    regions = ceiling_regions(annular_groove(), min_angle_deg=FDM_ANGLE)

    assert len(regions) == 1
    groove = regions[0]
    assert isinstance(groove, Ceiling)
    assert groove.faces == 128
    assert groove.area_mm2 == pytest.approx(5.74, abs=0.1)
    # Narrower than the bore it sits on: the equal-area bound, not the footprint.
    assert groove.span_mm == pytest.approx(2.70, abs=0.05)


def test_a_span_inside_the_bridge_limit_is_not_an_overhang():
    mesh = annular_groove()

    # Without the span rule the ring is an overhang: that is the false positive.
    assert overhang_faces(mesh, min_angle_deg=FDM_ANGLE)[0] == pytest.approx(5.74, abs=0.1)
    assert FDM_BRIDGE == 8.0
    assert overhang_faces(mesh, min_angle_deg=FDM_ANGLE, bridge_mm=FDM_BRIDGE)[0] == 0.0


def test_a_deep_bore_ceiling_is_a_bridge_of_its_own_diameter():
    regions = ceiling_regions(blind_bore(6.4), min_angle_deg=FDM_ANGLE)

    assert len(regions) == 1
    assert regions[0].span_mm == pytest.approx(6.4, abs=0.05)
    assert overhang_faces(
        blind_bore(6.4), min_angle_deg=FDM_ANGLE, bridge_mm=FDM_BRIDGE
    )[0] == 0.0


def test_a_ceiling_wider_than_the_bridge_limit_is_still_reported():
    """The direction that must not be forgiven: 10 mm is past what FDM crosses."""
    mesh = recess((10.0, 10.0, 3.0))
    regions = ceiling_regions(mesh, min_angle_deg=FDM_ANGLE)

    assert len(regions) == 1
    assert regions[0].span_mm == pytest.approx(10.0, abs=0.05)
    assert overhang_faces(
        mesh, min_angle_deg=FDM_ANGLE, bridge_mm=FDM_BRIDGE
    )[0] == pytest.approx(100.0, abs=1.0)


def test_a_wide_but_narrow_opening_is_bridged_despite_more_area():
    """Area ordering and verdict ordering disagree, which is the whole point.

    The long slot has more than twice the ceiling area of the compact recess, and the
    span rule still lets it pass while reporting the recess: no area floor can do
    that, whatever number it is set to.
    """
    long_slot = recess((40.0, 6.0, 3.0), plate_mm=(70.0, 40.0, 12.0))  # 240 mm2, 6 mm span
    compact = recess((10.0, 10.0, 3.0))  # 100 mm2, 10 mm span
    slot_area = overhang_faces(long_slot, min_angle_deg=FDM_ANGLE)[0]
    compact_area = overhang_faces(compact, min_angle_deg=FDM_ANGLE)[0]
    assert slot_area > 2.0 * compact_area

    assert overhang_faces(long_slot, min_angle_deg=FDM_ANGLE, bridge_mm=FDM_BRIDGE)[0] == 0.0
    assert overhang_faces(compact, min_angle_deg=FDM_ANGLE, bridge_mm=FDM_BRIDGE)[0] > 0.0


def test_a_thin_ring_around_a_wide_bore_does_not_inherit_the_bore_diameter():
    """A footprint-only rule would call this 20 mm; the groove is 0.3 mm wide."""
    regions = ceiling_regions(annular_groove(0.3, 20.0), min_angle_deg=FDM_ANGLE)

    assert len(regions) == 1
    assert regions[0].span_mm < FDM_BRIDGE
    assert overhang_faces(
        annular_groove(0.3, 20.0), min_angle_deg=FDM_ANGLE, bridge_mm=FDM_BRIDGE
    )[0] == 0.0


def test_no_bridge_limit_keeps_the_plain_angle_rule():
    """`bridge_mm=0` is the search's mode: it ranks poses, it does not judge spans."""
    mesh = annular_groove()
    assert overhang_faces(mesh, min_angle_deg=FDM_ANGLE, bridge_mm=0.0)[0] > 0.0


# --------------------------------------------------------------------------- #
# The verifier: resin does not bridge
# --------------------------------------------------------------------------- #
def _view(mesh: trimesh.Trimesh, *, process: str, orientation=(0.0, 0.0, 0.0)):
    """A design view whose body is `mesh` and whose insert is a small peg on top."""
    from pah.design.api import DesignView
    from pah.design.spec import (
        DesignSpec,
        FitConstraints,
        InterfaceSpec,
        PartSpec,
        PegHoleParams,
    )

    peg = cylinder(radius=1.0, height=6.0, sections=16)
    peg.apply_translation((0.0, 0.0, 15.0))
    spec = DesignSpec(
        spec_id=f"bridge-{process}",
        requirement="a plate with one cavity ceiling, for the bridge rule",
        parts=(
            PartSpec(id="body", role="body", size_mm=tuple(mesh.extents)),
            PartSpec(id="insert", role="insert", size_mm=(2.0, 2.0, 6.0)),
        ),
        interfaces=(
            InterfaceSpec(
                id="iface_0",
                kind="peg_hole",
                a="body",
                b="insert",
                params=PegHoleParams(d_peg_mm=2.0, d_hole_mm=2.4).model_dump(),
            ),
        ),
        constraints=FitConstraints(
            min_clearance_mm=0.03,
            max_clearance_mm=0.60,
            process=process,
            print_orientation_deg=orientation,
        ),
    )
    return DesignView(spec=spec, meshes={"body": mesh, "insert": peg}, paths={})


def test_the_same_step_is_fine_on_fdm_and_blocking_on_resin():
    """`sla.bridge_mm` is 0: a vat is not bridged, so the step is a real overhang."""
    assert PROCESS_PROFILES["fdm"].bridge_mm == 8.0
    assert PROCESS_PROFILES["sla"].bridge_mm == 0.0
    mesh = annular_groove()

    fdm = Printability()
    assert fdm.check(_view(mesh, process="fdm")) == []
    # The forgiveness is recorded, not silent: the report can say what it let pass.
    assert fdm.bridged_overhangs and fdm.bridged_overhangs[0]["span_mm"] < FDM_BRIDGE

    sla = Printability()
    codes = [cert.code for cert in sla.check(_view(mesh, process="sla"))]
    assert "unsupported_overhang" in codes
    assert sla.bridged_overhangs == ()


def test_a_bridged_ceiling_is_reported_beside_a_real_one():
    """One part, two ceilings: the finding must carry what the span rule let pass."""
    mesh = both_ceilings()

    certificate = None
    for cert in Printability().check(_view(mesh, process="fdm")):
        if cert.code == "unsupported_overhang":
            certificate = cert
    assert certificate is not None, "the 10 x 10 recess must still be reported"

    magnitude = certificate.magnitude
    assert magnitude["area_mm2"] == pytest.approx(100.0, abs=2.0)  # the recess
    assert magnitude["bridge_mm"] == 8.0
    bridged = magnitude["bridged_ceilings"]
    assert len(bridged) == 1
    assert bridged[0]["span_mm"] < 8.0
    assert bridged[0]["area_mm2"] == pytest.approx(32.1, abs=1.0)


def test_a_clean_verdict_still_shows_what_the_span_rule_forgave(tmp_path):
    """The report must explain a clean printable result, not just be silent (C2).

    A decomposed run reaches `printable: 是` because a 6.4 mm ceiling was bridged. If
    the report said nothing, "we bridged a ceiling" and "we never looked" would read
    the same — the two-claims discipline the audit is about.
    """
    from pah.cli.main import main

    source = tmp_path / "plate.stl"
    source.write_bytes(_plate(60.0, 40.0, 12.0).export(file_type="stl"))
    out = tmp_path / "run"

    assert main(["decompose", str(source), "--parts", "3", "--out", str(out)]) == 0
    report = json.loads((out / "report.json").read_text("utf-8"))
    block = report["verdicts"]["printable"]

    assert block["ok"] is True
    assert block["blocking"] == []
    bridged = block["orientation"]["bridged_ceilings"]
    assert bridged, "a bridged ceiling has to appear somewhere in the report"
    assert all(0.0 < row["span_mm"] <= FDM_BRIDGE for row in bridged)
    assert all(row["area_mm2"] > 0.0 for row in bridged)

