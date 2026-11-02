"""No declared orientation must not mean "no answer".

Audit P2-2 / C2. `printability` evaluated overhangs only when the spec declared a build
orientation, so every multi-part run — where each piece is printed in its own best pose —
reported `overhang not evaluated`. The fix is a finite search: a printed part rests on a
face, and a cube has 24 of them.

The recommendation is deterministic (overhang first, then height, then candidate order),
because a recommendation that changed between two identical runs would be useless as
evidence. A **declared** orientation still wins: the search only fills the gap.
"""

from __future__ import annotations

import trimesh

from pah.design.orientation import CANDIDATES, Orientation, recommend
from pah.tools.verifiers._design_fixtures import standard_fixture
from pah.tools.verifiers.printability import Printability


def test_the_search_considers_the_24_face_down_poses():
    assert len(CANDIDATES) == 24
    assert len(set(CANDIDATES)) == 24


def test_a_tilted_plate_prefers_a_face_down_pose(tmp_path):
    """A 40x40x4 plate has zero overhang flat and a lot on edge."""
    plate = trimesh.creation.box(extents=(40.0, 40.0, 4.0))

    flat = recommend(plate, angle_deg=45.0)

    assert flat.overhang_area_mm2 <= 1e-6, flat
    assert flat.bed_area_mm2 > 100.0, flat
    assert flat.height_mm == 4.0


def test_the_recommendation_is_deterministic():
    mesh = trimesh.creation.box(extents=(30.0, 20.0, 6.0))

    first = recommend(mesh, angle_deg=45.0)
    second = recommend(mesh, angle_deg=45.0)

    assert first.angles == second.angles
    assert first.overhang_area_mm2 == second.overhang_area_mm2


def test_support_volume_is_labelled_an_estimate():
    """A support number that looks measured but is guessed is worse than none."""
    table = recommend(trimesh.creation.box(extents=(40.0, 40.0, 4.0)), angle_deg=45.0)

    assert "support_volume_mm3_estimate" in table.as_dict()
    assert "support_volume_mm3" not in table.as_dict()


# The verifier integration (no declaration -> search instead of "not evaluated") is NOT
# wired yet: turning it on changes six existing expectations about what an undeclared
# orientation means, and those need updating together with the report column. The search
# itself is landed and tested above; the flag stays off until the report can carry both
# columns ("declared" vs "recommended").
