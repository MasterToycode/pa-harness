"""The multi-part path: what an assembly records, and what still refuses to build.

`DesignSpec` now accepts N parts and M interfaces (`test_design_spec.py` pins that
contract). This file pins the two things downstream of it:

* the `Assembly` a decomposition's spec produces — every interface becomes a joint,
  and the two-part landmarks (`body_id`, `insert_id`) are *absent* when there is no
  single part holding that role, rather than pointing at whichever came first;
* the honest refusal from the trimesh materialiser, because "N parts, M interfaces"
  is representable today and buildable only after per-interface machining lands. A
  refusal that names the shape is a work item; a `ValueError` about `spec.interface`
  would read like a caller mistake.
"""

from __future__ import annotations

import pytest

from pah.design.api import assembly_from_spec, geometry_hash, spec_of
from pah.design.materialize import materialize
from pah.design.spec import DesignSpec, InterfaceSpec, PartSpec

#: base -> mid -> tip. The middle part takes the peg side of one interface and the
#: hole side of the next, which is the shape a decomposition returns and the reason
#: `role` cannot decide topology.
CHAIN = (
    PartSpec(id="base", role="body", size_mm=(60.0, 40.0, 12.0)),
    PartSpec(id="mid", role="part", size_mm=(20.0, 20.0, 20.0), pose_mm=(0.0, 0.0, 12.0)),
    PartSpec(id="tip", role="part", size_mm=(10.0, 10.0, 10.0), pose_mm=(0.0, 0.0, 32.0)),
)
PEG = {"d_peg_mm": 9.9, "d_hole_mm": 10.0}


def chain_spec() -> DesignSpec:
    return DesignSpec(
        spec_id="chain",
        requirement="three printable pieces",
        parts=CHAIN,
        interfaces=(
            InterfaceSpec(id="iface_base_mid", kind="peg_hole", a="base", b="mid", params=PEG),
            InterfaceSpec(id="iface_mid_tip", kind="peg_hole", a="mid", b="tip", params=PEG),
        ),
    )


# --------------------------------------------------------------------------- #
# The assembly record
# --------------------------------------------------------------------------- #
def test_every_interface_becomes_a_joint():
    spec = chain_spec()
    assembly = assembly_from_spec(spec=spec)

    assert [joint["id"] for joint in assembly.joints] == [
        "iface_base_mid",
        "iface_mid_tip",
    ]
    assert [joint["kind"] for joint in assembly.joints] == ["peg_hole", "peg_hole"]
    assert [(joint["a"], joint["b"]) for joint in assembly.joints] == [
        ("base", "mid"),
        ("mid", "tip"),
    ]
    # The assembly is a complete record of its own design: the spec round-trips.
    assert spec_of(assembly).content_key() == spec.content_key()


def test_every_part_carries_its_own_geometry_hash():
    spec = chain_spec()
    assembly = assembly_from_spec(spec=spec)

    hashes = {part["id"]: part["geom"]["hash"] for part in assembly.parts}
    assert list(hashes) == ["base", "mid", "tip"]
    assert len(set(hashes.values())) == 3, "two parts share a geometry hash"
    assert hashes["mid"] == geometry_hash(spec, "mid")


def test_the_two_part_landmarks_disappear_when_there_is_no_the_insert():
    """A report may say "the body" only when there is exactly one. On a chain the
    landmark is simply absent: a landmark that can be wrong is worse than none."""
    constraints = assembly_from_spec(spec=chain_spec()).constraints

    assert constraints["body_id"] == "base"  # exactly one part is tagged body
    assert "insert_id" not in constraints  # none is tagged insert
    assert constraints["body_id"] != "mid"


def test_a_two_part_spec_keeps_both_landmarks():
    """The convenience must not regress: the two-part report still has its ids."""
    spec = DesignSpec(
        spec_id="pair",
        parts=(
            PartSpec(id="body", role="body", size_mm=(60.0, 40.0, 12.0)),
            PartSpec(id="peg", role="insert", size_mm=(10.0, 10.0, 14.0), pose_mm=(0.0, 0.0, 2.0)),
        ),
        interfaces=(InterfaceSpec(id="i", kind="peg_hole", a="body", b="peg", params=PEG),),
    )
    constraints = assembly_from_spec(spec=spec).constraints

    assert (constraints["body_id"], constraints["insert_id"]) == ("body", "peg")


def test_a_spec_edit_moves_the_assembly_id():
    """The chain's identity is its numbers; editing one connection is a new design."""
    spec = chain_spec()
    edited = spec.with_interface(interface_id="iface_mid_tip", params={"d_peg_mm": 9.0})

    assert assembly_from_spec(spec=spec).id != assembly_from_spec(spec=edited).id


# --------------------------------------------------------------------------- #
# What the materialiser still refuses, and how it says so
# --------------------------------------------------------------------------- #
def test_n_part_materialisation_is_refused_by_name(tmp_path):
    with pytest.raises(NotImplementedError) as raised:
        materialize(chain_spec(), tmp_path / "run")

    message = str(raised.value)
    assert "3 parts and 2 interfaces" in message
    assert "two-part" in message
    assert not (tmp_path / "run" / "spec.json").exists(), (
        "a refused materialisation must not leave a run directory that looks built"
    )


def test_a_standalone_part_is_refused_with_a_different_reason(tmp_path):
    """One part, no interface: the design is legal, the *blank* is not a part."""
    solo = DesignSpec(
        spec_id="solo", parts=(PartSpec(id="only", role="part", size_mm=(20.0, 20.0, 5.0)),)
    )
    with pytest.raises(NotImplementedError) as raised:
        materialize(solo, tmp_path / "run")

    message = str(raised.value)
    assert "one part and no interfaces" in message
    assert "mesh materialiser" in message


def test_the_two_part_case_still_materialises(tmp_path):
    """The refusal above must not have widened: the shipped case still builds."""
    spec = DesignSpec(
        spec_id="pair",
        parts=(
            PartSpec(id="body", role="body", size_mm=(60.0, 40.0, 12.0)),
            PartSpec(id="peg", role="insert", size_mm=(10.0, 10.0, 14.0), pose_mm=(0.0, 0.0, 2.0)),
        ),
        interfaces=(
            InterfaceSpec(id="i", kind="peg_hole", a="body", b="peg", params={"d_peg_mm": 10.2, "d_hole_mm": 10.0}),
        ),
    )
    built = materialize(spec, tmp_path / "run")

    assert set(built.meshes) == {"body", "peg"}
    assert all(mesh.is_watertight for mesh in built.meshes.values())
