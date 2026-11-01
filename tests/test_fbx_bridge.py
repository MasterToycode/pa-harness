"""The FBX bridge against a real Blender, or skipped where Blender is absent.

Why these tests exist at all
----------------------------
Every other module in PAH can be tested with fakes, because its contract is a
Python object. This one's contract is a *file format*: the claim "STL in, FBX out,
same millimetres, one object per part" is only true if a real DCC application says
so. A stub would test our own reading of a format we do not write.

So the tests run Blender when it is installed and skip loudly when it is not —
`BlenderMissing` is a documented setup step, not a failure of the suite.
"""

from __future__ import annotations

import json

import pytest

from pah.tools.fbx import (
    BLENDER_ENV,
    BlenderMissing,
    FbxError,
    blender_exe,
    describe,
    from_fbx,
    to_fbx,
)

try:
    _BLENDER = blender_exe()
except BlenderMissing:
    _BLENDER = None

needs_blender = pytest.mark.skipif(
    _BLENDER is None,
    reason=f"Blender is not installed (set {BLENDER_ENV} to blender.exe to run these)",
)

#: The c1 acceptance geometry: a 60x40x12 blank and a 10.2 mm peg. Real output
#: from the materialiser, so the round trip is measured against what PAH writes.
BODY = "runs/acceptance/c1-episode/geometry/body.stl"
INSERT = "runs/acceptance/c1-episode/geometry/insert.stl"

#: FBX stores coordinates as float32, so 60 mm comes back as 60.0000019. That is
#: the format's resolution, not a bug in the bridge: demanding equality would be
#: demanding Blender write a format it is not writing. 1e-5 relative is ~0.6 µm on
#: a 60 mm part — three orders below the 0.1 mm fit band the design plane judges.
FBX_TOL = 1e-5


def _repo(path: str):
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    return root / path


# --------------------------------------------------------------------------- #
# Setup resolution — runs everywhere, Blender or not
# --------------------------------------------------------------------------- #
def test_describe_reports_a_row_either_way():
    row = describe()
    assert row["blender"] in ("ok", "missing")
    if row["blender"] == "ok":
        assert row["path"].endswith(("blender.exe", "blender"))
    else:
        assert "FBX" in row["detail"] or "Blender" in row["detail"]


def test_env_pointing_nowhere_names_the_variable(monkeypatch):
    monkeypatch.setenv(BLENDER_ENV, "D:/definitely/not/blender.exe")
    with pytest.raises(BlenderMissing) as raised:
        blender_exe()
    assert BLENDER_ENV in str(raised.value)


def test_missing_blender_names_the_download(monkeypatch):
    """The message is the install instruction; a bare 'not found' costs an hour."""
    monkeypatch.delenv(BLENDER_ENV, raising=False)
    monkeypatch.setattr("pah.tools.fbx._candidates", lambda: [])
    with pytest.raises(BlenderMissing) as raised:
        blender_exe()
    assert "blender/release" in str(raised.value)
    assert BLENDER_ENV in str(raised.value)


def test_missing_source_is_refused_before_blender_runs(tmp_path):
    with pytest.raises(FileNotFoundError):
        to_fbx([tmp_path / "nope.stl"], tmp_path / "out.fbx")
    with pytest.raises(FileNotFoundError):
        from_fbx(tmp_path / "nope.fbx", tmp_path / "parts")


def test_to_fbx_needs_at_least_one_source(tmp_path):
    with pytest.raises(ValueError):
        to_fbx([], tmp_path / "out.fbx")


# --------------------------------------------------------------------------- #
# The round trip — the actual claim
# --------------------------------------------------------------------------- #
@needs_blender
def test_round_trip_preserves_millimetres_and_part_names(tmp_path):
    """STL -> one multi-object FBX -> one STL per object, unchanged in mm.

    The assertion is *volume and extents*, not vertex counts: FBX quantises
    nothing, but Blender rewrites the mesh through its own importer, so the honest
    claim is "the solid is the same", which volume plus extents measures.
    """
    import trimesh

    fb = tmp_path / "assembly.fbx"
    names = to_fbx([_repo(BODY), _repo(INSERT)], fb)
    assert names == ["body", "insert"]
    assert fb.stat().st_size > 0

    parts = from_fbx(fb, tmp_path / "parts")
    assert sorted(parts) == ["body", "insert"]

    reference = trimesh.load(_repo(BODY), force="mesh")
    body = trimesh.load(parts["body"], force="mesh")
    assert body.is_watertight
    assert body.volume == pytest.approx(reference.volume, rel=FBX_TOL)
    assert body.extents == pytest.approx(reference.extents, rel=FBX_TOL)
    # 60 x 40 x 12 mm is the c1 blank; a unit-scale bug would show up here as
    # 60_000 or 0.06 and nothing else in the test would notice.
    assert body.extents.max() == pytest.approx(60.0, rel=FBX_TOL)


@needs_blender
def test_a_single_object_fbx_comes_back_as_one_part(tmp_path):
    import trimesh

    fb = tmp_path / "one.fbx"
    assert to_fbx([_repo(INSERT)], fb) == ["insert"]
    parts = from_fbx(fb, tmp_path / "parts")
    assert list(parts) == ["insert"]
    insert = trimesh.load(parts["insert"], force="mesh")
    assert insert.is_watertight
    assert insert.extents[2] == pytest.approx(14.0, rel=FBX_TOL)


@needs_blender
def test_the_bridge_reports_failure_instead_of_writing_a_bad_file(tmp_path):
    """An unreadable input must raise, not produce a 0-byte FBX that a job uploads."""
    broken = tmp_path / "broken.stl"
    broken.write_bytes(b"not an stl at all")
    with pytest.raises(FbxError):
        to_fbx([broken], tmp_path / "out.fbx")
    assert not (tmp_path / "out.fbx").exists()


def test_describe_is_json_serialisable():
    """`pah plugins`/a doctor row prints this; keep it plain data."""
    assert json.loads(json.dumps(describe()))
