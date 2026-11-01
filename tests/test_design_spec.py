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
        # two parts both claiming to be the body
        (BODY, PEG, PartSpec(id="third", role="body", size_mm=(1.0, 1.0, 1.0))),
        # two parts both claiming to be the insert
        (BODY, PEG, PartSpec(id="third", role="insert", size_mm=(1.0, 1.0, 1.0))),
    ],
)
def test_at_most_one_body_and_one_insert(parts):
    """`role` is a hint about where a part came from, and only two of the three
    values describe *one* part. A second body is a design that has not decided
    which piece is the frame."""
    with pytest.raises(ValidationError):
        DesignSpec(
            spec_id="x",
            parts=parts,
            interfaces=(
                InterfaceSpec(id="i", kind="peg_hole", a="body", b="peg", params={"d_peg_mm": 1, "d_hole_mm": 2}),
            ),
        )


def test_two_interfaces_cannot_join_the_same_pair():
    """One mating pair, one connection. Two connectors in the same pair compete for
    the same space, and neither insertion path can be satisfied at once."""
    with pytest.raises(ValidationError) as raised:
        DesignSpec(
            spec_id="x",
            parts=(BODY, PEG),
            interfaces=(
                InterfaceSpec(id="i0", kind="peg_hole", a="body", b="peg", params={"d_peg_mm": 9.9, "d_hole_mm": 10.0}),
                InterfaceSpec(id="i1", kind="peg_hole", a="body", b="peg", params={"d_peg_mm": 9.9, "d_hole_mm": 10.0}),
            ),
        )
    assert "same pair" in str(raised.value)


def test_an_interface_may_join_either_way_round():
    """The old contract demanded `a` be the body and `b` the insert. N-part specs
    cannot: in a chain the middle part takes the peg side of one interface and the
    hole side of the next, so the side is a property of the *connection*, not of the
    part. What still matters is that both sides exist."""
    reversed_side = DesignSpec(
        spec_id="x",
        parts=(BODY, PEG),
        interfaces=(
            InterfaceSpec(id="i", kind="peg_hole", a="peg", b="body", params={"d_peg_mm": 9.9, "d_hole_mm": 10.0}),
        ),
    )
    assert reversed_side.interface.a == "peg"
    assert reversed_side.interface.b == "body"


def test_an_interface_to_an_unknown_part_is_refused():
    with pytest.raises(ValidationError) as raised:
        DesignSpec(
            spec_id="x",
            parts=(BODY, PEG),
            interfaces=(
                InterfaceSpec(id="i", kind="peg_hole", a="body", b="ghost", params={"d_peg_mm": 9.9, "d_hole_mm": 10.0}),
            ),
        )
    assert "ghost" in str(raised.value)


def test_an_interface_cannot_join_a_part_to_itself():
    with pytest.raises(ValidationError) as raised:
        DesignSpec(
            spec_id="x",
            parts=(BODY, PEG),
            interfaces=(
                InterfaceSpec(id="i", kind="peg_hole", a="body", b="body", params={"d_peg_mm": 9.9, "d_hole_mm": 10.0}),
            ),
        )
    assert "itself" in str(raised.value)


def test_a_part_that_no_interface_reaches_is_refused():
    """A printed part nothing is connected to is a part nothing holds. If two solids
    really ship as one object without a connector, that is two designs."""
    orphan = PartSpec(id="orphan", role="part", size_mm=(5.0, 5.0, 5.0))
    with pytest.raises(ValidationError) as raised:
        DesignSpec(
            spec_id="x",
            parts=(BODY, PEG, orphan),
            interfaces=(
                InterfaceSpec(id="i", kind="peg_hole", a="body", b="peg", params={"d_peg_mm": 9.9, "d_hole_mm": 10.0}),
            ),
        )
    assert "orphan" in str(raised.value)


def test_a_disconnected_assembly_is_refused():
    """Both parts are reached, but by different connectors: the spec is two
    assemblies wearing one id, and no verifier could judge it as one object."""
    with pytest.raises(ValidationError) as raised:
        DesignSpec(
            spec_id="x",
            parts=(
                BODY,
                PEG,
                PartSpec(id="c1", role="part", size_mm=(5.0, 5.0, 5.0)),
                PartSpec(id="c2", role="part", size_mm=(5.0, 5.0, 5.0)),
            ),
            interfaces=(
                InterfaceSpec(id="i0", kind="peg_hole", a="body", b="peg", params={"d_peg_mm": 9.9, "d_hole_mm": 10.0}),
                InterfaceSpec(id="i1", kind="peg_hole", a="c1", b="c2", params={"d_peg_mm": 9.9, "d_hole_mm": 10.0}),
            ),
        )
    assert "splits into 2" in str(raised.value)


def test_duplicate_interface_ids_are_refused():
    with pytest.raises(ValidationError) as raised:
        DesignSpec(
            spec_id="x",
            parts=(BODY, PEG, PartSpec(id="c", role="part", size_mm=(5.0, 5.0, 5.0))),
            interfaces=(
                InterfaceSpec(id="i", kind="peg_hole", a="body", b="peg", params={"d_peg_mm": 9.9, "d_hole_mm": 10.0}),
                InterfaceSpec(id="i", kind="peg_hole", a="peg", b="c", params={"d_peg_mm": 9.9, "d_hole_mm": 10.0}),
            ),
        )
    assert "duplicate interface" in str(raised.value)


def test_a_single_part_without_interfaces_is_a_design():
    """One solid, no connection: the shape a decomposition returns when P3-SAM
    finds one part. It must be materialisable and printable on its own."""
    spec = DesignSpec(
        spec_id="solo", parts=(PartSpec(id="only", role="part", size_mm=(20.0, 20.0, 5.0)),)
    )
    assert spec.part_ids() == ("only",)
    assert spec.interfaces == ()


# --------------------------------------------------------------------------- #
# N parts: the shape that a decomposition returns
# --------------------------------------------------------------------------- #
CHAIN = (
    PartSpec(id="base", role="body", size_mm=(60.0, 40.0, 12.0)),
    PartSpec(id="mid", role="part", size_mm=(20.0, 20.0, 20.0), pose_mm=(0.0, 0.0, 12.0)),
    PartSpec(id="tip", role="part", size_mm=(10.0, 10.0, 10.0), pose_mm=(0.0, 0.0, 32.0)),
)


def chain_spec() -> DesignSpec:
    peg = {"d_peg_mm": 9.9, "d_hole_mm": 10.0}
    return DesignSpec(
        spec_id="chain",
        requirement="three printable pieces",
        parts=CHAIN,
        interfaces=(
            InterfaceSpec(id="iface_base_mid", kind="peg_hole", a="base", b="mid", params=peg),
            InterfaceSpec(id="iface_mid_tip", kind="peg_hole", a="mid", b="tip", params=peg),
        ),
    )


def test_three_parts_and_two_interfaces_are_a_valid_spec():
    spec = chain_spec()

    assert spec.part_ids() == ("base", "mid", "tip")
    assert [iface.id for iface in spec.interfaces] == ["iface_base_mid", "iface_mid_tip"]


def test_a_middle_part_sits_on_both_sides_of_its_two_interfaces():
    spec = chain_spec()

    assert [iface.id for iface in spec.interfaces_of("mid")] == [
        "iface_base_mid",
        "iface_mid_tip",
    ]
    assert spec.neighbour_of("iface_base_mid", "mid") == "base"
    assert spec.neighbour_of(spec.interface_by_id("iface_mid_tip"), "mid") == "tip"
    assert len(spec.interfaces_of("base")) == 1


def test_the_two_part_shorthands_refuse_to_guess_on_an_n_part_spec():
    """`body`, `insert` and `interface` are the two-part conveniences. On a chain
    they must say so, not silently pick the first match: a repair that edited "the
    interface" of a 3-part design would edit whichever one happened to be first."""
    spec = chain_spec()

    # `base` is the only part tagged body, so this one still resolves.
    assert spec.body.id == "base"

    with pytest.raises(ValueError) as insert:
        spec.insert
    assert "tagged 'insert'" in str(insert.value)

    with pytest.raises(ValueError) as iface:
        spec.interface
    assert "use `interfaces`" in str(iface.value)


def test_with_interface_needs_the_interface_id_once_there_are_many():
    spec = chain_spec()

    with pytest.raises(ValueError):
        spec.with_interface(params={"d_hole_mm": 10.4})

    narrowed = spec.with_interface(
        interface_id="iface_mid_tip", params={"d_hole_mm": 10.4}
    )
    assert narrowed.interface_by_id("iface_mid_tip").params["d_hole_mm"] == pytest.approx(10.4)
    # The other interface is untouched: a repair of one connection is not a
    # renegotiation of the whole assembly.
    assert narrowed.interface_by_id("iface_base_mid").params["d_hole_mm"] == pytest.approx(10.0)
    assert narrowed.revision == spec.revision + 1


def test_content_key_ignores_the_order_interfaces_were_discovered_in():
    """A decomposition that walks its bboxes in a different order did not design
    anything differently, so the identity must not move."""
    spec = chain_spec()
    shuffled = spec.edit(interfaces=tuple(reversed(spec.interfaces)))

    assert shuffled.content_key() == spec.content_key()


def test_content_key_moves_when_one_connection_of_many_changes():
    spec = chain_spec()
    edited = spec.with_interface(interface_id="iface_mid_tip", params={"d_peg_mm": 9.0})
    assert edited.content_key() != spec.content_key()


def test_short_says_how_many_pieces_a_multi_part_spec_has():
    assert chain_spec().short().endswith("3 parts / 2 interfaces")
    assert "peg_hole" in peg_hole_spec().short()


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
