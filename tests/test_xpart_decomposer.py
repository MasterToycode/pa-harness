"""The X-Part adapter, exercised without spending credits.

Why this file exists at all
---------------------------
The adapter's expensive half (COS upload, submit, poll) cannot be in the test suite:
it costs money and needs a network. Its *other* half — repair a generated part into
a closed solid, rescale a normalised decomposition to millimetres, name the parts,
carry a provenance record — is where the real bugs live, and it can be tested on
meshes written by the local splitter.

So the offline loader (`load_decomposition_dir`) and the repair ladder (`_close`) are
covered directly, and the live path is covered by one recorded job whose facts are
asserted in `docs/multi-part.md` rather than by a test that runs it again.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import trimesh

from pah.design.decompose import split_slabs

#: `examples/` is not a package: the paid adapters live outside it on purpose.
EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _load_example():
    import importlib.util

    module_spec = importlib.util.spec_from_file_location(
        "_pah_example_xpart_decomposer", EXAMPLES / "xpart_decomposer.py"
    )
    assert module_spec and module_spec.loader
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def xpart():
    return _load_example()


def _write_parts(directory: Path, count: int = 2) -> None:
    """A decomposition directory shaped like the one the API returns."""
    directory.mkdir(parents=True, exist_ok=True)
    decomposition = split_slabs(trimesh.creation.box(extents=(60.0, 40.0, 12.0)), count=count)
    for part in decomposition.parts:
        part.mesh.export(directory / f"{part.id}.glb")


# --------------------------------------------------------------------------- #
# Replay a decomposition that is already on disk
# --------------------------------------------------------------------------- #
def test_a_directory_loads_as_a_decomposition_without_calling_anything(xpart, tmp_path):
    _write_parts(tmp_path / "parts")

    decomposition = xpart.load_decomposition_dir(tmp_path / "parts")

    assert len(decomposition.parts) == 2
    assert decomposition.provenance["method"] == "xpart-dir"
    assert decomposition.provenance["files"] == 2
    for part in decomposition.parts:
        assert part.mesh.is_watertight
        assert part.mesh.volume > 0.0


def test_a_normalised_decomposition_is_scaled_to_the_size_asked_for(xpart, tmp_path):
    """X-Part answers in normalised units; the caller says how big the object is.

    The measured job had a union extent of 1.049 — a person asking for a 120 mm
    object needs every part multiplied by ~114. Silent would be a part printed at
    1/100 scale, which looks like a slicer problem for an hour.
    """
    _write_parts(tmp_path / "parts")
    raw = xpart.load_decomposition_dir(tmp_path / "parts")
    union_before = trimesh.util.concatenate([part.mesh for part in raw.parts]).extents.max()

    scaled = xpart.load_decomposition_dir(tmp_path / "parts", target_size_mm=120.0)
    union_after = trimesh.util.concatenate([part.mesh for part in scaled.parts]).extents.max()

    assert scaled.provenance["scale_mm"] == pytest.approx(120.0 / union_before)
    assert union_after == pytest.approx(120.0, rel=1e-6)
    # Volumes scale cubically, which is the check that catches a scale applied to
    # the transform but not to the vertices.
    assert scaled.parts[0].mesh.volume == pytest.approx(
        raw.parts[0].mesh.volume * (120.0 / union_before) ** 3, rel=1e-6
    )


def test_an_unreadable_part_is_named_as_dropped(xpart, tmp_path):
    """A broken download must be reported, not silently subtracted from the count."""
    parts = tmp_path / "parts"
    parts.mkdir()
    trimesh.creation.box(extents=(10.0, 10.0, 10.0)).export(parts / "good.glb")
    (parts / "broken.glb").write_bytes(b"this is not a mesh")

    decomposition = xpart.load_decomposition_dir(parts)

    assert [part.id for part in decomposition.parts] == ["good"]
    assert decomposition.provenance["dropped"][0][0] == "broken.glb"
    assert "unreadable" in decomposition.provenance["dropped"][0][1]


def test_an_empty_directory_is_refused_with_the_path(xpart, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()

    with pytest.raises(FileNotFoundError) as raised:
        xpart.load_decomposition_dir(empty)
    assert str(empty) in str(raised.value)


def test_the_repair_ladder_prefers_filling_holes_over_remeshing(xpart):
    """A closed solid is returned untouched: no remesh, no silent detail loss."""
    solid = trimesh.creation.box(extents=(10.0, 10.0, 10.0))

    assert xpart._close(solid) is solid

    holed = trimesh.Trimesh(
        vertices=solid.vertices.copy(), faces=solid.faces[:-2].copy(), process=False
    )
    assert not holed.is_watertight
    repaired = xpart._close(holed)
    assert repaired is not None and repaired.is_watertight


def test_a_part_is_named_by_the_geometry_inside_it_unless_that_name_is_a_placeholder(
    xpart, tmp_path
):
    """The API's parts arrive as `part_N.glb` holding a geometry called `part_N`.

    That inner name is what the segmentation labels refer to, so it must survive. A
    GLB written by trimesh instead holds `geometry_0` — a placeholder, not an
    identity — and two of those in one directory would collide, so a placeholder
    name falls back to the file stem.
    """
    named = tmp_path / "part_7.glb"
    trimesh.creation.box(extents=(1.0, 1.0, 1.0)).export(named)
    assert xpart._inner_name(named) == "part_7"  # trimesh's placeholder, replaced by the stem

    placeholder = tmp_path / "scene.glb"
    trimesh.creation.box(extents=(1.0, 1.0, 1.0)).export(placeholder)
    assert xpart._inner_name(placeholder) == "scene"


def test_two_files_with_the_same_placeholder_name_stay_distinct(xpart, tmp_path):
    parts = tmp_path / "parts"
    parts.mkdir()
    third = trimesh.creation.box(extents=(10.0, 10.0, 10.0))
    (parts / "a.glb").parent.mkdir(parents=True, exist_ok=True)
    third.export(parts / "a.glb")
    third.copy().apply_translation((20.0, 0.0, 0.0)).export(parts / "b.glb")

    decomposition = xpart.load_decomposition_dir(parts)

    assert len({part.id for part in decomposition.parts}) == 2
