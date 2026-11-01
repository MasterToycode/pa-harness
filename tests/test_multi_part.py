"""The N-part path end to end, on a real solid, with no network and no credits.

What is pinned here
-------------------
Every arrow of `pah decompose` that used to be a two-part assumption:

* splitting a solid into pieces whose volumes add back up to the whole, and whose
  faces are exact complements;
* deciding *which* pieces are joined and along which axis, including the cases that
  must be refused (not adjacent, adjacent on a corner, too thin to hold a peg);
* machining a socket into one piece and a peg onto another without breaking either
  solid's watertightness, and with the peg narrower than the hole by exactly the
  designed clearance;
* assembling the spec, materialising it, and verifying **each joint in its own
  two-part shadow** — the trick that let six verifiers be reused instead of rewritten;
* the verdict arithmetic: a check that did not run must land in `not_evaluated`
  even though its note is keyed by interface.

The geometry is a 60x40x12 plate — the same blank the two-part acceptance case uses,
so a regression in the N-part path shows up as a difference from a known solid.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import trimesh

from pah.app.runtime import _verdicts
from pah.design.decompose import (
    Decomposition,
    DecomposedPart,
    check_store,
    file_hash,
    mesh_hash,
    parts_from_decomposition,
    split_slabs,
    store_parts,
)
from pah.design.machining import MachiningError, add_peg, cut_socket
from pah.design.materialize import materialize
from pah.design.pairing import contact_of, propose_interfaces, spec_from_parts
from pah.design.pairs import pair_spec, pair_view
from pah.design.spec import DesignSpec, FitConstraints, PartSpec

PLATE = (60.0, 40.0, 12.0)


def plate() -> trimesh.Trimesh:
    return trimesh.creation.box(extents=PLATE)


def two_slabs(tmp_path: Path) -> tuple[DesignSpec, Path]:
    """The standard fixture: a plate cut in two along Z, pegged back together."""
    decomposition = split_slabs(plate(), count=2, axis=2)
    out = tmp_path / "run"
    # The spec names the blank *file*, so the files are written first.
    stored = store_parts(decomposition.parts, out)
    parts = parts_from_decomposition(
        decomposition, source_hashes={key: file_hash(path) for key, path in stored.items()}
    )
    pairing = propose_interfaces(parts)
    spec = spec_from_parts(
        spec_id="two-slabs",
        parts=parts,
        interfaces=pairing.interfaces,
        constraints=FitConstraints(print_orientation_deg=(0.0, 0.0, 0.0)),
    )
    return spec, out


# --------------------------------------------------------------------------- #
# The splitter
# --------------------------------------------------------------------------- #
def test_slabs_are_solids_that_add_back_up_to_the_whole():
    whole = plate()
    decomposition = split_slabs(whole, count=3, axis=2)

    assert decomposition.part_ids() == ("part_00", "part_01", "part_02")
    for part in decomposition.parts:
        assert part.mesh.is_watertight
        assert part.mesh.volume > 0.0
    assert sum(part.mesh.volume for part in decomposition.parts) == pytest.approx(
        whole.volume, rel=1e-9
    )
    # The cut axis is recorded, because "why is this split along Z" is a question a
    # report has to answer without a human re-running it.
    assert decomposition.provenance["axis"] == "z"


def test_the_splitter_names_itself_honestly():
    """It is not a semantic decomposition, and the provenance says so."""
    provenance = split_slabs(plate(), count=2).provenance

    assert provenance["method"] == "local-slabs"
    assert "not a semantic decomposition" in provenance["note"]


def test_an_open_shell_is_refused_before_it_becomes_parts():
    """A surface with no volume cannot be cut into printable pieces."""
    triangle = trimesh.Trimesh(
        vertices=[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0]],
        faces=[[0, 1, 2]],
        process=False,
    )
    with pytest.raises(ValueError) as raised:
        split_slabs(triangle, count=2)
    assert "closed solid" in str(raised.value)


def test_one_part_is_not_a_decomposition():
    with pytest.raises(ValueError):
        split_slabs(plate(), count=1)


# --------------------------------------------------------------------------- #
# Pairing
# --------------------------------------------------------------------------- #
def test_two_stacked_slabs_are_adjacent_along_z():
    parts = parts_from_decomposition(split_slabs(plate(), count=2, axis=2))
    contact = contact_of(*parts)

    assert contact is not None
    assert contact.axis == (0.0, 0.0, 1.0)  # from part_00 up to part_01
    assert contact.point_mm[2] == pytest.approx(0.0)  # the shared face is z = 0
    assert contact.gap_mm == pytest.approx(0.0)
    assert contact.overlap_mm == pytest.approx((60.0, 40.0))
    assert contact.thickness_mm == pytest.approx((6.0, 6.0))


def test_parts_that_do_not_touch_are_not_adjacent():
    left = PartSpec(id="left", role="part", size_mm=(10.0, 10.0, 10.0), pose_mm=(0.0, 0.0, 0.0))
    right = PartSpec(id="right", role="part", size_mm=(10.0, 10.0, 10.0), pose_mm=(20.0, 0.0, 0.0))

    assert contact_of(left, right, tolerance_mm=0.5) is None
    # ... and it becomes adjacent once the tolerance covers the 10 mm gap.
    assert contact_of(left, right, tolerance_mm=12.0) is not None


def test_parts_touching_on_a_corner_are_refused():
    """Two boxes diagonal to each other share an edge, not a face; a peg there has
    no surface to seat on."""
    first = PartSpec(id="a", role="part", size_mm=(10.0, 10.0, 10.0), pose_mm=(0.0, 0.0, 0.0))
    second = PartSpec(id="b", role="part", size_mm=(10.0, 10.0, 10.0), pose_mm=(10.0, 10.0, 0.0))

    assert contact_of(first, second) is None


def test_a_thin_pair_is_skipped_by_name_not_silently():
    """A 2 mm shelf cannot hold a peg that survives printing; the pairing says why."""
    thin = PartSpec(id="thin", role="part", size_mm=(40.0, 40.0, 2.0), pose_mm=(0.0, 0.0, 0.0))
    thick = PartSpec(id="thick", role="part", size_mm=(40.0, 40.0, 20.0), pose_mm=(0.0, 0.0, 11.0))

    report = propose_interfaces((thin, thick))

    assert report.interfaces == ()
    assert report.skipped
    assert {report.skipped[0][0], report.skipped[0][1]} == {"thin", "thick"}
    assert "peg" in report.reasons()


def test_the_proposed_peg_fits_the_hole_and_the_clearance_is_the_designs():
    parts = parts_from_decomposition(split_slabs(plate(), count=2, axis=2))
    report = propose_interfaces(parts, clearance_mm=0.25)

    assert len(report.interfaces) == 1
    params = report.interfaces[0].params
    radial = (params["d_hole_mm"] - params["d_peg_mm"]) / 2.0
    assert radial == pytest.approx(0.25)
    # The peg may use at most 60% of the thinnest part's thickness, so a socket
    # never leaves a paper-thin floor.
    assert params["engage_mm"] <= 0.6 * 6.0 + 1e-9
    assert params["d_peg_mm"] <= 0.6 * 6.0 + 1e-9


def test_a_chain_puts_the_peg_and_the_socket_on_opposite_faces_of_the_middle():
    parts = parts_from_decomposition(split_slabs(plate(), count=3, axis=2))
    interfaces = propose_interfaces(parts).interfaces

    assert [i.id for i in interfaces] == ["iface_part_00_part_01", "iface_part_01_part_02"]
    assert (interfaces[0].a, interfaces[0].b) == ("part_00", "part_01")
    assert (interfaces[1].a, interfaces[1].b) == ("part_01", "part_02")


# --------------------------------------------------------------------------- #
# Machining
# --------------------------------------------------------------------------- #
def test_a_socket_removes_exactly_the_cylinder_it_names():
    solid = plate()
    socket = cut_socket(
        solid, axis=(0.0, 0.0, 1.0), point=(0.0, 0.0, 6.0), d_hole_mm=6.0, depth_mm=8.0
    )

    assert socket.is_watertight
    expected = 3.141592653589793 * 3.0**2 * 8.0
    assert solid.volume - socket.volume == pytest.approx(expected, rel=0.02)


def test_a_peg_adds_material_and_stays_one_solid():
    solid = plate()
    with_peg = add_peg(
        solid, axis=(0.0, 0.0, 1.0), point=(0.0, 0.0, 6.0), d_peg_mm=6.0, engage_mm=5.0
    )

    assert with_peg.is_watertight
    assert with_peg.volume > solid.volume
    # The peg stands 5 mm proud of the face and buries a 2 mm root in the part, so
    # the solid grows by the root pad, not by the protrusion.
    assert with_peg.extents[2] == pytest.approx(12.0 + 2.0, rel=1e-6)


def test_machining_refuses_a_feature_that_would_split_the_part():
    """A cutter that removes the whole part is a design error, not a mesh to slice.

    Depth, not diameter, is what bites: a wide shallow cutter takes a slice off the
    top and leaves a perfectly good solid, so the refusal has to be tested with a
    cut that goes all the way through.
    """
    small = trimesh.creation.box(extents=(4.0, 4.0, 4.0))
    with pytest.raises(MachiningError):
        cut_socket(
            small, axis=(0.0, 0.0, 1.0), point=(0.0, 0.0, 2.0), d_hole_mm=20.0, depth_mm=10.0
        )


# --------------------------------------------------------------------------- #
# Materialisation and the per-pair shadow
# --------------------------------------------------------------------------- #
def test_materialising_a_decomposition_writes_every_part_and_records_every_axis(tmp_path):
    spec, out = two_slabs(tmp_path)
    design = materialize(spec, out)

    assert sorted(design.meshes) == ["part_00", "part_01"]
    for part_id, mesh in design.meshes.items():
        assert mesh.is_watertight, part_id
        assert (out / "geometry" / f"{part_id}.stl").exists()
    assert list(design.interfaces) == ["iface_part_00_part_01"]
    assert design.interfaces["iface_part_00_part_01"]["mate_axis"] == [0.0, 0.0, 1.0]
    # The blank store is a hash-checked input, not a convenience copy.
    assert set(check_store(spec, out)) == {"part_00", "part_01"}


def test_the_spec_and_the_blank_store_agree_or_the_run_refuses(tmp_path):
    spec, out = two_slabs(tmp_path)
    blank = out / "parts" / "part_00.stl"
    blank.write_bytes(trimesh.creation.box(extents=(5.0, 5.0, 5.0)).export(file_type="stl"))

    with pytest.raises(FileNotFoundError) as raised:
        check_store(spec, out)
    assert "part_00" in str(raised.value)


def test_the_pair_shadow_is_exactly_the_two_part_contract(tmp_path):
    spec, out = two_slabs(tmp_path)
    design = materialize(spec, out)
    interface_id = spec.interfaces[0].id
    pair = pair_view(design, interface_id)

    assert pair.spec.body.id == "part_00"  # host, dressed as the body
    assert pair.spec.insert.id == "part_01"
    assert pair.spec.interface.id == interface_id
    assert set(pair.view.meshes) == {"part_00", "part_01"}
    assert pair.view.mate_axis == (0.0, 0.0, 1.0)
    # The shadow is derived, not stored: a chain's second pair gets its own axes.
    assert pair_spec(spec, interface_id).content_key() != spec.content_key()


def test_each_pair_of_a_chain_is_judged_on_its_own(tmp_path):
    from pah.tools.verifiers.design_cascade import DesignVerifierCascade, default_registry

    decomposition = split_slabs(plate(), count=3, axis=2)
    out = tmp_path / "run"
    stored = store_parts(decomposition.parts, out)
    parts = parts_from_decomposition(
        decomposition, source_hashes={key: file_hash(path) for key, path in stored.items()}
    )
    spec = spec_from_parts(
        spec_id="chain",
        parts=parts,
        interfaces=propose_interfaces(parts).interfaces,
        # No build orientation declared, for the reason the CLI gives: a chain's
        # middle piece carries a peg pointing down, so printing it in assembly pose
        # is never the pose a person uses, and declaring one would manufacture an
        # overhang finding about a decision nobody made.
        constraints=FitConstraints(),
    )
    design = materialize(spec, out)

    cascade = DesignVerifierCascade(registry=default_registry(), root=out)
    certificates = cascade.check(design.assembly)

    assert certificates == [], [cert.code for cert in certificates]
    # Notes are keyed by interface, which is how a reader learns *which* joint had a
    # check that did not run.
    assert all("/" in key for key in cascade.notes), cascade.notes
    assert len({key.split("/")[0] for key in cascade.notes}) == 2


def test_a_check_that_did_not_run_on_one_joint_makes_the_claim_partial():
    """The verdict arithmetic must see interface-keyed notes.

    Measured 2026-11-02, first three-part run: notes were keyed
    `iface_.../physics_settle` while `_verdicts` matched bare verifier ids, so a run
    with two checks not run reported `validated: full`. Under-claiming is the safe
    direction and over-claiming is the bug this harness exists to prevent.
    """
    notes = {
        "iface_a_b/physics_settle": "not evaluated",
        "iface_a_b/printability": "overhang not evaluated",
    }
    verdicts = _verdicts([], notes, Path("/nonexistent/dir"))

    assert "physics_settle" in verdicts["assemblable"]["not_evaluated"]
    assert verdicts["assemblable"]["validated"] == "partial"
    assert verdicts["printable"]["validated"] == "partial"


def test_a_two_part_note_still_counts():
    """The bare-id form must keep working: the shipped two-part runs use it."""
    verdicts = _verdicts([], {"physics_settle": "not evaluated"}, Path("/nonexistent/dir"))

    assert verdicts["assemblable"]["not_evaluated"] == ["physics_settle"]


# --------------------------------------------------------------------------- #
# The command
# --------------------------------------------------------------------------- #
def test_pah_decompose_builds_a_printable_three_part_run(tmp_path, capsys):
    from pah.cli.main import main

    source = tmp_path / "plate.stl"
    source.write_bytes(plate().export(file_type="stl"))
    out = tmp_path / "run"

    code = main(["decompose", str(source), "--parts", "3", "--out", str(out)])

    assert code == 0, capsys.readouterr().out
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["run"]["parts"] == 3
    assert report["run"]["interfaces"] == 2
    assert report["verdicts"]["printable"]["watertight"] == {
        "part_00": True,
        "part_01": True,
        "part_02": True,
    }
    assert (out / "print" / "assembly.3mf").exists()


def test_pah_decompose_says_so_when_nothing_can_be_joined(tmp_path, capsys):
    """One part in, one part out: there is no pair, so there is no product."""
    from pah.cli.main import main

    source = tmp_path / "cube.stl"
    source.write_bytes(trimesh.creation.box(extents=(10.0, 10.0, 10.0)).export(file_type="stl"))

    code = main(["decompose", str(source), "--parts", "1", "--out", str(tmp_path / "run")])

    assert code == 1
    assert "at least 2" in capsys.readouterr().out


def test_mesh_hash_tracks_bytes_not_object_identity():
    first = plate()
    second = plate()

    assert mesh_hash(first) == mesh_hash(second)
    second.apply_translation((1.0, 0.0, 0.0))
    assert mesh_hash(first) != mesh_hash(second)


def test_a_decomposition_dataclass_round_trips_through_the_store(tmp_path):
    decomposition = Decomposition(
        parts=(DecomposedPart(id="only", mesh=plate()),),
        provenance={"method": "test"},
    )
    paths = store_parts(decomposition.parts, tmp_path)

    assert paths["only"].exists()
    assert mesh_hash(trimesh.load(paths["only"], force="mesh")) == mesh_hash(plate())
