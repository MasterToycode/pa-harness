"""Contract tests for the design plane's materialisation layer (T1).

What is asserted here is what every later v2 module stands on: a spec becomes two
closed, positive-volume, world-frame solids; the geometry actually encodes the
numbers (a 0.1 mm oversize peg interferes, a 0.15 mm undersize one does not, and
the socket really has a floor); and materialising twice is byte-for-byte the same
design, because `docs/contracts/v2-interfaces.md` §7.4 makes reproducibility a
release criterion rather than a virtue.

The tests are deliberately allowed to touch `trimesh` directly. A materialiser
that is only tested through its own hash would pass while writing garbage --
`geometry_hash` is a function of the spec, so it cannot notice a broken mesh.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import trimesh

from pah.design.api import geometry_hash, spec_of, view_of
from pah.design.connectors import available_kinds, builder_for, register_builder
from pah.design.materialize import materialize
from pah.design.spec import DesignSpec, PegHoleParams

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures" / "design"
FIXTURE_FILES = sorted(FIXTURES.glob("*.json"))
BUILTIN_KINDS = ("peg_hole", "snap_fit", "dovetail")


def fixture(name: str) -> DesignSpec:
    """The spec one shipped fixture describes."""
    return DesignSpec.model_validate_json((FIXTURES / name).read_text(encoding="utf-8"))


def intersection_volume(first: trimesh.Trimesh, second: trimesh.Trimesh) -> float:
    """How much of the two solids occupies the same points, in mm3.

    Two solids that *touch* over a face -- the seated peg on the socket floor, the
    dovetail handle on the plate -- have zero volume in common, and manifold still
    returns a **watertight zero-thickness sheet** for that case. `.volume` is not
    usable here: it goes through `mass_properties`, which divides by that zero and
    warns (the centre of mass of a sheet is not defined), so the same divergence
    integral is summed directly below. A verifier that judged interference on
    "the intersection is non-empty" would fire on every correctly seated design;
    the design-plane rule is a volume threshold, which is why this helper measures
    volume and nothing else.
    """
    overlap = trimesh.boolean.intersection([first, second], engine="manifold")
    if overlap is None or len(overlap.faces) == 0:
        return 0.0
    corners = overlap.triangles
    signed = np.einsum("ij,ij->i", corners[:, 0], np.cross(corners[:, 1], corners[:, 2]))
    return float(abs(signed.sum()) / 6.0)


def reach_at(mesh: trimesh.Trimesh, z: float, *, start: float = -100.0) -> float:
    """How far `mesh` reaches from the Z axis at height `z`, measured along X.

    Used where a bounding box cannot answer: the insert's AABB is set by its
    handle, so "how long is the exposed peg" has to be read off a section.
    """
    locations, _, _ = mesh.ray.intersects_location(
        np.array([[start, 0.0, z]]), np.array([[1.0, 0.0, 0.0]])
    )
    return float(np.max(np.abs(locations[:, 0])))


# --------------------------------------------------------------------------- #
# The registry
# --------------------------------------------------------------------------- #
def test_the_three_builtin_kinds_are_registered():
    assert set(BUILTIN_KINDS) <= set(available_kinds())


def test_an_unknown_kind_names_the_registered_alternatives():
    with pytest.raises(KeyError, match="peg_hole"):
        builder_for("magnets")


def test_a_kind_cannot_be_registered_twice():
    """Two builders for one kind would make geometry depend on import order."""

    class Duplicate:
        kind = "peg_hole"

        def params_model(self):
            return PegHoleParams

        def clearance_mm(self, params):
            return 0.0

        def build(self, spec):
            raise AssertionError("never called")

    with pytest.raises(ValueError, match="already registered"):
        register_builder(Duplicate())


# --------------------------------------------------------------------------- #
# Materialisation: layout, solids, axes
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("path", FIXTURE_FILES, ids=lambda path: path.stem)
def test_every_fixture_materialises_two_closed_solids(path: Path, tmp_path: Path):
    spec = DesignSpec.model_validate_json(path.read_text(encoding="utf-8"))
    assert spec.interface.kind in available_kinds()

    design = materialize(spec, tmp_path / "design")

    assert design.root == (tmp_path / "design").resolve()
    assert set(design.meshes) == {part.id for part in spec.parts}
    for part in spec.parts:
        mesh = design.meshes[part.id]
        assert mesh.is_watertight, f"{part.id} is an open shell"
        assert mesh.volume > 0.0, f"{part.id} has non-positive volume"

        written = Path(design.paths[part.id])
        assert written.is_absolute()
        assert written.exists()
        assert written.parent == design.root / "geometry"
        assert written.name == f"{part.id}.stl"

    saved = design.root / "spec.json"
    assert saved.exists()
    assert DesignSpec.model_validate_json(saved.read_text(encoding="utf-8")).content_key() == (
        spec.content_key()
    )


@pytest.mark.parametrize(
    ("name", "axis", "approach"),
    [
        ("demo-peg-hole-interfere.json", (0.0, 0.0, 1.0), 10.0),
        ("demo-snap-fit.json", (0.0, 0.0, 1.0), 12.0),
        ("demo-dovetail.json", (1.0, 0.0, 0.0), 22.0),
    ],
)
def test_the_materialised_design_reports_the_insertion_axis_and_approach(
    name, axis, approach, tmp_path: Path
):
    """The path verifier starts its sweep from these, so they are part of the port.

    They are copied off the builder's `ConnectorGeometry`, not re-derived from the
    kind name: dovetail slides along +X, and a view that only knew "peg_hole-style
    default +Z" would block a correct design.
    """
    spec = fixture(name)
    design = materialize(spec, tmp_path)

    assert design.mate_axis == pytest.approx(axis, abs=1e-9)
    assert design.approach_mm == pytest.approx(approach)
    view = design.view()
    assert tuple(view.mate_axis) == tuple(design.mate_axis)
    assert view.approach_mm == pytest.approx(design.approach_mm)
    # ... and the report is the builder's own, not a second opinion.
    built = builder_for(spec.interface.kind).build(spec)
    assert tuple(design.mate_axis) == pytest.approx(tuple(built.mate_axis), abs=1e-12)


#: Samples between `+approach` and the seated pose, matching the path verifier's
#: order of magnitude (its 14 steps are 0.43 mm at the default approach).
SWEEP_SAMPLES = 15


@pytest.mark.parametrize(
    "name", ["demo-peg-hole-clear.json", "demo-snap-fit.json", "demo-dovetail.json"]
)
def test_a_clean_design_has_a_clear_path_along_its_reported_axis(name, tmp_path: Path):
    """Sweeping the insert along `view.mate_axis` must not hit the body.

    This is the geometry half of `insertion_path`, asserted here so the axis
    contract is locked by T1 as well: the insert is translated from `+approach` to
    the seat, and no sampled position may share volume with the body. The seated
    pose itself is included -- it is a face contact, zero volume, not a penetration.
    """
    spec = fixture(name)
    design = materialize(spec, tmp_path)
    body, insert = design.meshes[spec.body.id], design.meshes[spec.insert.id]

    worst = 0.0
    for step in range(SWEEP_SAMPLES):
        height = design.approach_mm * (1.0 - step / (SWEEP_SAMPLES - 1))
        moved = insert.copy()
        moved.apply_translation([height * component for component in design.mate_axis])
        worst = max(worst, intersection_volume(body, moved))
    assert worst < 1e-3, f"{spec.interface.kind} is blocked along its own axis: {worst}"


def test_built_clearance_is_the_specs_clearance():
    """No second copy of the clearance arithmetic, for any kind."""
    for path in FIXTURE_FILES:
        spec = DesignSpec.model_validate_json(path.read_text(encoding="utf-8"))
        builder = builder_for(spec.interface.kind)
        assert builder.clearance_mm(spec.interface.validated_params()) == pytest.approx(
            spec.clearance_mm
        )


def test_the_blank_is_centred_and_its_pose_and_rotation_reach_world_coordinates(
    tmp_path: Path,
):
    """`pose_mm` names the blank's centre; the rotation is applied about it first."""
    spec = fixture("demo-peg-hole-clear.json")
    turned = spec.with_part(
        spec.body.model_copy(update={"rot_deg": (0.0, 90.0, 0.0), "pose_mm": (3.0, -4.0, 0.0)})
    )
    body = materialize(turned, tmp_path).meshes[turned.body.id]

    extents = body.bounds[1] - body.bounds[0]
    centre = (body.bounds[0] + body.bounds[1]) / 2.0
    assert extents == pytest.approx((12.0, 40.0, 60.0), abs=1e-6)
    assert centre == pytest.approx((3.0, -4.0, 0.0), abs=1e-6)


def test_the_insertion_axis_follows_the_inserts_rotation():
    spec = fixture("demo-peg-hole-clear.json")
    turned = spec.with_part(spec.insert.model_copy(update={"rot_deg": (0.0, 90.0, 0.0)}))
    built = builder_for(turned.interface.kind).build(turned)
    assert built.mate_axis == pytest.approx((1.0, 0.0, 0.0), abs=1e-9)


# --------------------------------------------------------------------------- #
# The demo: an oversize peg interferes, a clear one does not
# --------------------------------------------------------------------------- #
def test_the_oversize_peg_really_interferes(tmp_path: Path):
    spec = fixture("demo-peg-hole-interfere.json")
    params = spec.interface.validated_params()
    assert spec.clearance_mm == pytest.approx(-0.1)
    design = materialize(spec, tmp_path)
    overlap = intersection_volume(design.meshes["body"], design.meshes["insert"])
    # The full engagement is in play, so the sliver is the annulus between the two
    # diameters over `engage` -- about 25 mm3 for 10.2 in 10.0 over 8 mm.
    expected = math.pi * (params.d_peg_mm**2 - params.d_hole_mm**2) / 4.0 * params.engage_mm
    assert overlap == pytest.approx(expected, rel=0.05)
    assert spec.constraints.in_band(spec.clearance_mm) is False


def test_the_clear_peg_does_not_interfere(tmp_path: Path):
    spec = fixture("demo-peg-hole-clear.json")
    design = materialize(spec, tmp_path)
    overlap = intersection_volume(design.meshes["body"], design.meshes["insert"])
    assert overlap < 1e-3
    assert spec.constraints.in_band(spec.clearance_mm) is True


def test_the_socket_has_a_floor_and_the_volume_agrees(tmp_path: Path):
    spec = fixture("demo-peg-hole-interfere.json")
    params = spec.interface.validated_params()
    body = materialize(spec, tmp_path).meshes[spec.body.id]
    radius = params.d_hole_mm / 2.0
    top = spec.body.size_mm[2] / 2.0
    floor = top - params.engage_mm

    # Void just under the opening, material just under the floor: that pair is
    # what makes the socket blind. A through hole would leave the lower point
    # empty as well.
    assert not body.contains(np.array([[0.0, 0.0, top - 0.5]]))[0]
    assert body.contains(np.array([[0.0, 0.0, floor - 0.5]]))[0]
    # A point in the wall is material, so the test is about the socket and not
    # about `contains` always answering False.
    assert body.contains(np.array([[radius + 0.6, 0.0, top - 0.5]]))[0]

    # A ray down the socket axis enters material at the floor and leaves at the
    # blank's bottom face: the shallowest hit is the floor, exactly there.
    locations, _, _ = body.ray.intersects_location(
        np.array([[0.0, 0.0, top - 0.1]]), np.array([[0.0, 0.0, -1.0]])
    )
    assert len(locations) == 2
    assert np.max(locations[:, 2]) == pytest.approx(floor, abs=1e-6)

    # The socket removed a cylinder `engage` deep, not a through bore.
    blank = math.prod(spec.body.size_mm)
    assert blank - body.volume == pytest.approx(
        math.pi * radius**2 * params.engage_mm, rel=0.01
    )


def test_the_insert_seats_on_the_socket_floor(tmp_path: Path):
    spec = fixture("demo-peg-hole-clear.json")
    params = spec.interface.validated_params()
    design = materialize(spec, tmp_path)
    body, insert = design.meshes[spec.body.id], design.meshes[spec.insert.id]

    body_top = body.bounds[1][2]
    floor = body_top - params.engage_mm
    assert body_top == pytest.approx(spec.body.pose_mm[2] + spec.body.size_mm[2] / 2.0)
    # The insert's local origin is the peg's leading tip, so the seated pose puts
    # that tip on the socket floor: touching, never overlapping. The micromeasure
    # matters -- a pose a micron too deep would make the interference verifier
    # report a perfectly normal design.
    assert insert.bounds[0][2] == pytest.approx(floor, abs=1e-6)
    # The exposed peg really is `peg_length - engage` long: just below its top the
    # section is still the peg, and half a millimetre above it the section has
    # widened to the handle. The insert's AABB cannot answer this -- the handle is
    # what sets it -- so the length is read off two sections.
    exposed = params.peg_length_mm - params.engage_mm
    assert reach_at(insert, body_top + exposed - 0.5) == pytest.approx(
        params.d_peg_mm / 2.0, rel=0.01
    )
    assert reach_at(insert, body_top + exposed + 0.5) > 1.5 * params.d_peg_mm / 2.0
    assert intersection_volume(body, insert) < 1e-3


def test_a_part_that_is_not_one_closed_solid_is_refused():
    """Null geometry is named at the materialisation boundary, not left to a verifier.

    A shell or an empty boolean result would otherwise reach a verifier as
    "nothing observed", which the loop reads as a pass -- the most expensive class
    of failure the harness has (dev doc §4.8).
    """
    from pah.design.materialize import _require_solids

    spec = fixture("demo-peg-hole-clear.json")
    solid = trimesh.creation.box(extents=(10.0, 10.0, 10.0))

    with pytest.raises(ValueError, match="returned meshes for"):
        _require_solids(spec, {spec.body.id: solid})

    with pytest.raises(TypeError, match="not a Trimesh"):
        _require_solids(spec, {spec.body.id: solid, spec.insert.id: None})

    open_shell = solid.copy()
    open_shell.faces = open_shell.faces[:-1]
    with pytest.raises(ValueError, match="watertight"):
        _require_solids(spec, {spec.body.id: open_shell, spec.insert.id: solid})

    inverted = solid.copy()
    inverted.invert()
    with pytest.raises(ValueError, match="non-positive volume"):
        _require_solids(spec, {spec.body.id: inverted, spec.insert.id: solid})


# --------------------------------------------------------------------------- #
# Determinism and idempotence (contract §7.4)
# --------------------------------------------------------------------------- #
def test_two_materialisations_are_byte_identical(tmp_path: Path):
    spec = fixture("demo-snap-fit.json")
    first = materialize(spec, tmp_path / "a")
    second = materialize(spec, tmp_path / "b")

    assert first.assembly.id == second.assembly.id
    assert (first.root / "spec.json").read_bytes() == (second.root / "spec.json").read_bytes()
    for part in spec.parts:
        assert Path(first.paths[part.id]).read_bytes() == Path(second.paths[part.id]).read_bytes()
        recorded = next(row for row in first.assembly.parts if row["id"] == part.id)
        assert recorded["geom"]["hash"] == geometry_hash(spec, part.id)
        assert recorded["geom"]["path"] == first.paths[part.id]


def test_materialising_twice_into_one_directory_is_idempotent(tmp_path: Path):
    """Resume and retry re-materialise rounds that were already written."""
    spec = fixture("demo-dovetail.json")
    first = materialize(spec, tmp_path)
    before = {part_id: Path(path).read_bytes() for part_id, path in first.paths.items()}
    spec_before = (tmp_path / "spec.json").read_bytes()

    second = materialize(spec, tmp_path)

    assert second.assembly.id == first.assembly.id
    assert second.paths == first.paths
    for part_id, digest in before.items():
        assert Path(second.paths[part_id]).read_bytes() == digest
    assert (tmp_path / "spec.json").read_bytes() == spec_before


def test_geometry_hash_moves_only_when_the_design_does():
    spec = fixture("demo-peg-hole-clear.json")
    body_hash = geometry_hash(spec, "body")

    assert body_hash != geometry_hash(spec, "insert")
    assert body_hash == geometry_hash(spec, "body")  # a pure function of the spec

    # Revision and provenance are bookkeeping, not intent: re-recording them must
    # not look like a new design, or cycle detection would read a re-run as
    # progress.
    assert geometry_hash(spec.edit(revision=7), "body") == body_hash

    # A number the geometry is built from does move it.
    assert geometry_hash(spec.with_interface(params={"d_hole_mm": 10.4}), "body") != body_hash
    moved = spec.with_part(spec.insert.model_copy(update={"pose_mm": (2.0, 0.0, -2.0)}))
    assert geometry_hash(moved, "insert") != geometry_hash(spec, "insert")


# --------------------------------------------------------------------------- #
# Round trip through the disk
# --------------------------------------------------------------------------- #
def test_the_assembly_round_trips_the_spec_and_its_paths(tmp_path: Path):
    spec = fixture("demo-snap-fit.json")
    design = materialize(spec, tmp_path)

    assert spec_of(design.assembly).content_key() == spec.content_key()
    # The just-built meshes are handed to the view rather than parsed back.
    assert design.view().body.faces.shape == design.meshes[spec.body.id].faces.shape
    # An assembly read from elsewhere still finds its triangles.
    reloaded = view_of(design.assembly).mesh(spec.insert.id)
    assert reloaded.faces.shape == design.meshes[spec.insert.id].faces.shape


def test_the_written_stl_reads_back_as_the_same_solid(tmp_path: Path):
    spec = fixture("demo-peg-hole-clear.json")
    design = materialize(spec, tmp_path)
    for part in spec.parts:
        original = design.meshes[part.id]
        reloaded = trimesh.load(design.paths[part.id], force="mesh")
        assert reloaded.is_watertight
        assert len(reloaded.vertices) == len(original.vertices)
        assert len(reloaded.faces) == len(original.faces)
        assert reloaded.volume == pytest.approx(original.volume, rel=1e-6)


# --------------------------------------------------------------------------- #
# The other two kinds: the features they add are really there
# --------------------------------------------------------------------------- #
def test_the_snap_fit_chamber_widens_the_slot(tmp_path: Path):
    spec = fixture("demo-snap-fit.json")
    params = spec.interface.validated_params()
    design = materialize(spec, tmp_path)
    body = design.meshes[spec.body.id]

    blank = math.prod(spec.body.size_mm)
    slot = params.slot_w_mm**2 * spec.body.size_mm[2]
    play = params.radial_clearance_mm
    chamber_side = params.slot_w_mm + 2.0 * (params.hook_h_mm + play)
    chamber_height = params.hook_h_mm + 2.0 * play
    chamber = (chamber_side**2 - params.slot_w_mm**2) * chamber_height

    # More than the slot alone was removed, and by exactly the undercut chamber:
    # that is the feature the fit/path verifiers are supposed to see.
    assert blank - body.volume == pytest.approx(slot + chamber, rel=1e-6)
    assert intersection_volume(body, design.meshes[spec.insert.id]) < 1e-3


def test_the_dovetail_groove_is_a_sliding_channel(tmp_path: Path):
    spec = fixture("demo-dovetail.json")
    params = spec.interface.validated_params()
    design = materialize(spec, tmp_path)
    body, insert = design.meshes[spec.body.id], design.meshes[spec.insert.id]

    half_open = params.width_mm / 2.0
    half_floor = half_open + params.depth_mm * math.tan(math.radians(params.angle_deg))
    pocket = (half_open + half_floor) * params.depth_mm * (
        params.engage_mm + params.clearance_mm
    )
    assert math.prod(spec.body.size_mm) - body.volume == pytest.approx(pocket, rel=1e-6)

    # Seated: the tongue's leading face is where the pose puts it, it clears the
    # pocket, and the handle stands above the body's top face, which is what
    # makes the tongue impossible to lift out.
    assert insert.bounds[0][0] == pytest.approx(spec.insert.pose_mm[0], abs=1e-6)
    assert insert.bounds[1][2] > body.bounds[1][2]
    assert intersection_volume(body, insert) < 1e-3
