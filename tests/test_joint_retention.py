"""Joint retention: form closure, a gravity-carried seat, or a real defect.

Audit P1-2's second half. `physics_settle` models a peg in a bore and refuses a view
whose wall is thin compared to the insert, so a decomposed pair (a plate around a small
socket) got no physics tier at all — every multi-part run said "physics not evaluated"
in one sentence (round 2 and 3 of `docs/remediation-plan.md`).

`joint_retention` answers the question that *is* answerable geometrically: pull the
mating part out along the interface axis and see whether anything catches it. Two ways a
joint stays put, and the difference matters:

* **form closure** — an undercut blocks the pull-out (snap hook, dovetail, lip);
* **the seat carries it** — gravity pushes the part *into* the socket, so a plain
  clearance fit does not fall.

What is deliberately not claimed: friction. A horizontal peg in a horizontal hole is
held by material and surface finish, which this verifier does not know; inventing a
coefficient would turn a guess into a certificate. Only a pull-out that gravity actively
helps is reported, because that is the case geometry provably cannot hold.
"""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from pah.design.api import DESIGN_CODES, DesignView
from pah.design.spec import DesignSpec, InterfaceSpec, PartSpec
from pah.tools.verifiers.joint_retention import CODES, JointRetention
from pah.tools.verifiers.design_cascade import default_registry


def _spec() -> DesignSpec:
    return DesignSpec(
        spec_id="retention",
        parts=(
            PartSpec(id="body", role="body", size_mm=(30.0, 30.0, 8.0)),
            PartSpec(id="peg", role="insert", size_mm=(4.0, 4.0, 10.0), pose_mm=(0.0, 0.0, 1.0)),
        ),
        interfaces=(
            InterfaceSpec(
                id="iface",
                kind="peg_hole",
                a="body",
                b="peg",
                params={"d_peg_mm": 4.0, "d_hole_mm": 4.4, "engage_mm": 4.0},
            ),
        ),
    )


def _pair(*, axis=(0.0, 0.0, 1.0), undercut: bool = False) -> DesignView:
    """A host with a blind socket and a peg in it, optionally with a lip that holds it.

    `undercut=True` shrinks the socket's mouth to 3.6 mm inside a 0.5 mm band, so the
    4.0 mm peg cannot be pulled straight out — form closure, no friction assumed.
    """
    plate = trimesh.creation.box(extents=(30.0, 30.0, 8.0))
    plate.apply_translation((0.0, 0.0, -4.0))  # top face at z = 0
    bore = trimesh.creation.cylinder(radius=2.2, height=8.0, sections=48)
    bore.apply_translation((0.0, 0.0, -4.0))  # blind socket: mouth at z=0, floor at z=-8
    host = trimesh.boolean.difference([plate, bore], engine="manifold")
    if undercut:
        lip = trimesh.creation.cylinder(radius=1.8, height=1.0, sections=48)
        lip.apply_translation((0.0, 0.0, -0.5))  # a 3.6 mm lip just inside the mouth
        host = trimesh.boolean.union([host, lip], engine="manifold")

    peg = trimesh.creation.cylinder(radius=2.0, height=10.0, sections=48)
    peg.apply_translation((0.0, 0.0, 1.0))  # tip at the socket floor

    axis_array = np.asarray(axis, dtype=float)
    if not np.allclose(axis_array, (0.0, 0.0, 1.0)):
        rotation = trimesh.geometry.align_vectors((0.0, 0.0, 1.0), axis_array)
        host.apply_transform(rotation)
        peg.apply_transform(rotation)

    return DesignView(
        spec=_spec(),
        meshes={"body": host, "peg": peg},
        mate_axis=tuple(float(v) for v in axis_array),
        approach_mm=6.0,
    )


# --------------------------------------------------------------------------- #
# The contract: a code, a fixture pair, and a place in the closed set
# --------------------------------------------------------------------------- #
def test_the_code_is_in_the_design_contract():
    assert CODES == ("joint_not_retained",)
    assert DESIGN_CODES["joint_retention"] == ("joint_not_retained",)


def test_the_registry_ships_it():
    assert "joint_retention" in default_registry().ids()


def test_its_fixtures_disagree_the_way_the_gate_requires():
    verifier = JointRetention()
    good, bad = verifier.fixtures()

    assert good and bad
    assert verifier.check(good[0]) == []
    found = verifier.check(bad[0])
    assert [cert.code for cert in found] == ["joint_not_retained"]


# --------------------------------------------------------------------------- #
# The three states
# --------------------------------------------------------------------------- #
def test_a_peg_standing_in_an_upward_socket_is_retained_by_its_seat():
    """Gravity pushes it into the socket: no certificate, and no note either."""
    verifier = JointRetention()

    assert verifier.check(_pair(axis=(0.0, 0.0, 1.0))) == []
    assert verifier.last_note == ""


def test_an_undercut_is_form_closure_and_is_retained_even_upside_down():
    verifier = JointRetention()

    assert verifier.check(_pair(axis=(0.0, 0.0, -1.0), undercut=True)) == []
    assert verifier.last_note == ""


def test_a_friction_only_joint_upside_down_is_reported_as_not_retained():
    """Nothing blocks the pull-out and gravity points that way: a real defect."""
    verifier = JointRetention()

    found = verifier.check(_pair(axis=(0.0, 0.0, -1.0)))

    assert [cert.code for cert in found] == ["joint_not_retained"]
    assert found[0].family == "assembly"
    assert found[0].magnitude["gravity_along_pull_mps2"] > 0.0
    assert found[0].magnitude["overlap_volume_mm3"] <= found[0].magnitude["min_volume_mm3"]


def test_a_horizontal_joint_is_neither_claimed_nor_condemned():
    """Friction depends on material and finish; this verifier does not model it.

    A zero gravity component along the pull-out is not evidence for either answer, so
    saying nothing is the only honest report.
    """
    verifier = JointRetention()

    assert verifier.check(_pair(axis=(1.0, 0.0, 0.0))) == []
    assert verifier.last_note == ""


def test_a_missing_mesh_is_not_evaluated_rather_than_a_verdict(monkeypatch):
    view = _pair()
    verifier = JointRetention()

    def boom(_part_id):
        raise FileNotFoundError("no mesh")

    monkeypatch.setattr(view, "mesh", boom)

    assert verifier.check(view) == []
    assert "not evaluated" in verifier.last_note
