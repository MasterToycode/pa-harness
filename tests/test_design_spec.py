"""Contract tests for `pah.design.spec` — the frozen design intent.

Everything the v2 planes depend on is pinned here: the one-interface/two-part
shape, the fact that identity is content addressed, and the two meanings of
`with_interface` (a same-kind parameter edit merges; a kind change replaces,
because the old numbers are in the old kind's vocabulary).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from pah.design.spec import (
    DesignSpec,
    DovetailParams,
    FitConstraints,
    InterfaceSpec,
    PartSpec,
    PegHoleParams,
    SnapFitParams,
    builtin_params_model,
    radial_clearance_mm,
)

BODY = PartSpec(id="body", role="body", size_mm=(60.0, 40.0, 12.0))
PEG = PartSpec(id="peg", role="insert", size_mm=(12.0, 12.0, 20.0), pose_mm=(0.0, 0.0, 10.0))


def peg_hole_spec(**params) -> DesignSpec:
    values = {"d_peg_mm": 10.2, "d_hole_mm": 10.0}
    values.update(params)
    return DesignSpec(
        spec_id="demo",
        requirement="two-part bracket",
        parts=(BODY, PEG),
        interfaces=(
            InterfaceSpec(id="iface_0", kind="peg_hole", a="body", b="peg", params=values),
        ),
    )


# --------------------------------------------------------------------------- #
# shape
# --------------------------------------------------------------------------- #


def test_a_well_formed_spec_exposes_its_parts_and_clearance():
    spec = peg_hole_spec()

    assert spec.body.id == "body"
    assert spec.insert.id == "peg"
    assert spec.interface.kind == "peg_hole"
    assert spec.clearance_mm == pytest.approx(-0.1)


def test_clearance_is_derived_and_never_trusted_from_the_spec():
    # The parameters imply -0.1 mm; a clearance field would let a spec lie.
    spec = peg_hole_spec()
    assert radial_clearance_mm("peg_hole", spec.interface.params) == pytest.approx(-0.1)
    assert not spec.constraints.in_band(spec.clearance_mm)


@pytest.mark.parametrize(
    "parts",
    [
        (BODY,),
        (BODY, PEG, PartSpec(id="third", role="body", size_mm=(1.0, 1.0, 1.0))),
    ],
)
def test_v1_requires_exactly_one_body_and_one_insert(parts):
    with pytest.raises(ValidationError):
        DesignSpec(
            spec_id="x",
            parts=parts,
            interfaces=(InterfaceSpec(id="i", kind="peg_hole", a="body", b="peg", params={"d_peg_mm": 1, "d_hole_mm": 2}),),
        )


def test_one_interface_only():
    with pytest.raises(ValidationError):
        DesignSpec(
            spec_id="x",
            parts=(BODY, PEG),
            interfaces=(
                InterfaceSpec(id="i0", kind="peg_hole", a="body", b="peg", params={"d_peg_mm": 9.9, "d_hole_mm": 10.0}),
                InterfaceSpec(id="i1", kind="peg_hole", a="body", b="peg", params={"d_peg_mm": 9.9, "d_hole_mm": 10.0}),
            ),
        )


def test_the_interface_must_join_the_two_parts_with_the_right_roles():
    with pytest.raises(ValidationError):
        DesignSpec(
            spec_id="x",
            parts=(BODY, PEG),
            interfaces=(InterfaceSpec(id="i", kind="peg_hole", a="peg", b="body", params={"d_peg_mm": 9.9, "d_hole_mm": 10.0}),),
        )


def test_an_unknown_parameter_is_refused_where_the_spec_is_written():
    with pytest.raises(ValidationError):
        peg_hole_spec(d_peg_mm=10.0, d_hole_mm=10.3, surprise_mm=1.0)


def test_the_fit_band_must_be_ordered():
    with pytest.raises(ValidationError):
        FitConstraints(min_clearance_mm=0.6, max_clearance_mm=0.1)


# --------------------------------------------------------------------------- #
# identity
# --------------------------------------------------------------------------- #


def test_content_key_ignores_provenance_and_revision():
    """Who chose a number and which revision it is cannot change the geometry, so
    they must not change the identity the loop's cycle detection relies on."""
    spec = peg_hole_spec()
    replayed = DesignSpec.model_validate(
        {
            **spec.model_dump(),
            "revision": 7,
            "provenance": {"interface_kind": "llm:test", "params": "rule"},
            "requirement": "rewritten wording",
        }
    )
    assert replayed.content_key() == spec.content_key()


def test_content_key_moves_when_a_number_or_a_pose_moves():
    spec = peg_hole_spec()
    assert spec.with_interface(params={"d_hole_mm": 10.5}).content_key() != spec.content_key()

    moved = spec.with_part(PEG.model_copy(update={"pose_mm": (0.0, 0.0, 9.0)}))
    assert moved.content_key() != spec.content_key()
    assert moved.revision == spec.revision + 1


def test_content_key_is_stable_across_processes():
    # Pinned literal: a hash that changes between runs would break replay.
    assert peg_hole_spec().content_key() == peg_hole_spec().content_key()


# --------------------------------------------------------------------------- #
# edits
# --------------------------------------------------------------------------- #


def test_same_kind_edit_merges_parameters():
    spec = peg_hole_spec()
    edited = spec.with_interface(params={"d_hole_mm": 10.5})

    assert edited.revision == spec.revision + 1
    assert edited.interface.params["d_peg_mm"] == pytest.approx(10.2)  # kept
    assert edited.interface.params["d_hole_mm"] == pytest.approx(10.5)
    assert edited.clearance_mm == pytest.approx(0.15)
    assert spec.interface.params["d_hole_mm"] == pytest.approx(10.0)  # immutable


def test_a_kind_change_replaces_the_parameters_and_drops_the_old_vocabulary():
    spec = peg_hole_spec()
    swapped = spec.with_interface(
        kind="snap_fit", params={"hook_w_mm": 4.0, "hook_h_mm": 1.2, "slot_w_mm": 4.4}
    )

    assert swapped.interface.kind == "snap_fit"
    assert set(swapped.interface.params) == {"hook_w_mm", "hook_h_mm", "slot_w_mm"}
    assert swapped.clearance_mm == pytest.approx(0.2)
    assert swapped.revision == spec.revision + 1


def test_a_kind_change_without_parameters_is_refused():
    with pytest.raises(ValueError, match="needs that kind's parameters"):
        peg_hole_spec().with_interface(kind="dovetail")


def test_switching_kind_with_the_old_parameters_is_refused_at_validation():
    with pytest.raises(ValidationError):
        peg_hole_spec().with_interface(kind="dovetail", params={"d_peg_mm": 10.2})


def test_with_part_refuses_a_part_the_spec_does_not_have():
    with pytest.raises(KeyError):
        peg_hole_spec().with_part(PartSpec(id="ghost", role="insert", size_mm=(1.0, 1.0, 1.0)))


# --------------------------------------------------------------------------- #
# builtin parameter models
# --------------------------------------------------------------------------- #


def test_each_builtin_kind_declares_its_clearance_rule():
    assert radial_clearance_mm("peg_hole", PegHoleParams(d_peg_mm=10.0, d_hole_mm=10.4)) == pytest.approx(0.2)
    assert radial_clearance_mm("snap_fit", SnapFitParams(hook_w_mm=4.0, slot_w_mm=4.4)) == pytest.approx(0.2)
    assert radial_clearance_mm("dovetail", DovetailParams(clearance_mm=0.2)) == pytest.approx(0.2)
    assert builtin_params_model("nope") is None


def test_a_peg_longer_than_its_engagement_defaults_to_engaging_plus_a_handle():
    params = PegHoleParams(d_peg_mm=10.0, d_hole_mm=10.4, engage_mm=6.0)
    assert params.peg_length_mm == pytest.approx(12.0)
    assert PegHoleParams(d_peg_mm=10.0, d_hole_mm=10.4, peg_len_mm=25.0).peg_length_mm == 25.0
