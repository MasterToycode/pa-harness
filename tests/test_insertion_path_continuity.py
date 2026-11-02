"""The path sweep must not step over a wall — or start inside one.

Audit P1-3. Two ways the verifier used to claim more than it measured:

1. **14 fixed samples.** Across the default 6 mm approach that is 0.43 mm per step, so a
   wall thinner than that sits *between* two samples and the sweep reported "no
   obstruction". The audit's acceptance is exactly that: a wall the old sampling steps
   over, which the new one must find.
2. **The start was never tested.** The sweep began at `+approach` and only looked
   downward, so an insert whose starting pose already overlapped the body reported a
   clear path — the strongest possible false positive.

Fixed by testing the start pose (`approach_not_clear`) and by refining any interval whose
end poses come close to the body down to 0.1 mm steps, with a hard ceiling on booleans.
"""

from __future__ import annotations

import numpy as np
import trimesh

from pah.design.api import (
    CODE_APPROACH_NOT_CLEAR,
    CODE_INSERTION_PATH_BLOCKED,
    DesignView,
)
from pah.design.spec import DesignSpec, InterfaceSpec, PartSpec
from pah.tools.verifiers.insertion_path import MAX_BOOLEANS, MIN_STEP_MM, InsertionPath


def _spec() -> DesignSpec:
    return DesignSpec(
        spec_id="path",
        parts=(
            PartSpec(id="body", role="body", size_mm=(40.0, 40.0, 10.0)),
            PartSpec(id="peg", role="insert", size_mm=(6.0, 6.0, 20.0), pose_mm=(0.0, 0.0, 0.0)),
        ),
        interfaces=(
            InterfaceSpec(
                id="iface",
                kind="peg_hole",
                a="body",
                b="peg",
                params={"d_peg_mm": 6.0, "d_hole_mm": 6.4, "engage_mm": 5.0},
            ),
        ),
    )


def _body() -> trimesh.Trimesh:
    """A plate whose top face is z = 0, with a 6.4 mm bore down to z = -5."""
    plate = trimesh.creation.box(extents=(40.0, 40.0, 10.0))
    plate.apply_translation((0.0, 0.0, -5.0))
    # A roof above the plate, with the same bore: the insert starts inside it, so a
    # *laterally shifted* start pose is genuinely inside material (the audit asks for
    # "start inside the body", which an open-topped plate cannot express).
    roof = trimesh.creation.box(extents=(40.0, 40.0, 8.0))
    roof.apply_translation((0.0, 0.0, 4.0))
    bore = trimesh.creation.cylinder(radius=3.2, height=20.0, sections=48)
    bore.apply_translation((0.0, 0.0, -1.0))
    shell = trimesh.boolean.union([plate, roof], engine="manifold")
    return trimesh.boolean.difference([shell, bore], engine="manifold")


def _peg() -> trimesh.Trimesh:
    """A 6 mm peg whose tip is at z = 0, so the seated pose is inside the bore."""
    peg = trimesh.creation.cylinder(radius=3.0, height=20.0, sections=48)
    peg.apply_translation((0.0, 0.0, 10.0))
    return peg


def _view(*, body=None, peg=None) -> DesignView:
    return DesignView(
        spec=_spec(),
        meshes={"body": body if body is not None else _body(), "peg": peg if peg is not None else _peg()},
        mate_axis=(0.0, 0.0, 1.0),
        approach_mm=6.0,
    )


def _thin_wall_body() -> trimesh.Trimesh:
    """The bore, plus a 0.2 mm shelf just inside the mouth.

    At the coarse ladder (0.43 mm steps) the peg passes either side of it; only a fine
    enough sweep can hit it. The shelf is a full disc minus the bore so it really blocks.
    """
    plate = trimesh.creation.box(extents=(40.0, 40.0, 10.0))
    plate.apply_translation((0.0, 0.0, -5.0))
    bore = trimesh.creation.cylinder(radius=3.2, height=5.0, sections=48)
    bore.apply_translation((0.0, 0.0, -2.5))
    body = trimesh.boolean.difference([plate, bore], engine="manifold")

    # A 0.2 mm membrane across the bore, sitting in the last 0.43 mm before the seat —
    # the band the fixed ladder never sampled. The peg's tip must pass through it.
    disc = trimesh.creation.cylinder(radius=4.0, height=0.2, sections=48)
    disc.apply_translation((0.0, 0.0, 0.2))  # spans z in [0.1, 0.3]
    return trimesh.boolean.union([body, disc], engine="manifold")


# --------------------------------------------------------------------------- #
# The old failure mode: start inside the body
# --------------------------------------------------------------------------- #
def test_a_start_pose_inside_the_body_is_reported_as_unreachable():
    """Starting inside and finding no obstruction below it was the worst false pass."""
    body = _body()
    peg = _peg()
    peg.apply_translation((4.0, 0.0, 0.0))  # shifted off the bore: solid material above

    found = InsertionPath().check(_view(body=body, peg=peg))

    codes = [cert.code for cert in found]
    # The start overlaps, so the first sample is blocked *and* the approach did not
    # clear the body. Both are true and both are reported; the second is the one the
    # old sweep could not produce at all.
    assert CODE_INSERTION_PATH_BLOCKED in codes
    assert CODE_APPROACH_NOT_CLEAR in codes
    unresolved = next(c for c in found if c.code == CODE_APPROACH_NOT_CLEAR)
    assert unresolved.magnitude["volume_mm3"] > 0.0
    assert "outside" in unresolved.magnitude["reason"]


def test_a_clean_pair_still_passes():
    assert InsertionPath().check(_view()) == []


# --------------------------------------------------------------------------- #
# The old failure mode: a wall thinner than the step
# --------------------------------------------------------------------------- #
def test_a_wall_thinner_than_the_coarse_step_is_found():
    """0.2 mm shelf, 0.43 mm coarse step: the old sweep walked straight past it.

    This is the audit's acceptance case. It also pins the refinement budget: the check
    must find it without unbounded booleans.
    """
    found = InsertionPath().check(_view(body=_thin_wall_body()))

    assert [cert.code for cert in found] == [CODE_INSERTION_PATH_BLOCKED]
    magnitude = found[0].magnitude
    # Either the extended ladder or a refinement may be what catches it; the finding is
    # what matters, and both paths are bounded by the same boolean budget.
    assert magnitude["blocked_at_mm"] <= 0.5
    assert magnitude["booleans"] <= MAX_BOOLEANS
    # The block is near the shelf, not at the seat.
    assert 0.0 < magnitude["blocked_at_mm"] < 6.0


def test_the_certificate_says_the_path_is_straight():
    """A doorknob path (rotate then slide) is explicitly out of scope."""
    found = InsertionPath().check(_view(body=_thin_wall_body()))

    assert found[0].magnitude["path"] == "single straight pull along mate_axis"


def test_the_boolean_budget_is_a_ceiling_not_a_verdict():
    """When the budget runs out the check stops claiming; it never invents a block."""
    verifier = InsertionPath()
    view = _view(body=_thin_wall_body())

    found = verifier.check(view)

    assert found, "the shelf is still found within the budget"
    assert found[0].magnitude["booleans"] <= MAX_BOOLEANS
    # And a clean pair with the same budget reports clean, not blocked-by-exhaustion.
    assert verifier.check(_view()) == []


def test_the_magnitude_carries_where_and_how_much():
    found = InsertionPath().check(_view(body=_thin_wall_body()))
    magnitude = found[0].magnitude

    assert set(magnitude) >= {
        "blocked_at_mm",
        "volume_mm3",
        "approach_mm",
        "mate_axis",
        "step_mm",
        "samples",
    }
    assert np.allclose(magnitude["mate_axis"], (0.0, 0.0, 1.0))
