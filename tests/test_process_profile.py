"""One part, two processes, two answers — and the report names which one it used.

Audit P2-1. The process knowledge used to be split: an overhang table in the verifier and
a single `FitConstraints.min_wall_mm` that meant the same thing for an FDM nozzle and for
a CNC mill. A 0.5 mm wall is unprintable on FDM and ordinary in resin, so one number
cannot serve both, and a report that did not name the process could not be compared to
another conclusion.

The profile now lives in `pah.design.process`; the verifier takes the **stricter** of the
design's floor and the process's floor, records both in the certificate, and refuses
(`not evaluated`) when the process name has no profile instead of borrowing FDM's.
"""

from __future__ import annotations

import pytest
import trimesh

from pah.design.process import PROCESS_PROFILES, effective_min_wall_mm, profile_for
from pah.design.spec import FitConstraints


def test_every_process_the_spec_can_name_has_a_profile():
    """The spec's Literal and the table must not drift apart."""
    allowed = {"fdm", "sla", "sls", "mjf", "cnc"}

    assert set(PROCESS_PROFILES) == allowed
    for name in allowed:
        assert profile_for(name) is not None
        assert profile_for(name).min_wall_mm > 0.0


def test_the_profiles_actually_differ():
    """A table where every process shares the numbers would pass the test above."""
    walls = {name: profile.min_wall_mm for name, profile in PROCESS_PROFILES.items()}
    assert len(set(walls.values())) >= 4, walls
    assert walls["sla"] < walls["fdm"]
    assert PROCESS_PROFILES["cnc"].overhang_deg is None
    assert PROCESS_PROFILES["sls"].supports is False


def test_an_unknown_process_has_no_profile():
    assert profile_for("fdm-ish") is None
    assert profile_for("") is None


def test_the_effective_floor_is_the_stricter_of_design_and_process():
    fdm = PROCESS_PROFILES["fdm"]  # 1.2 mm floor
    sla = PROCESS_PROFILES["sla"]  # 0.4 mm floor

    # The design asks for less than FDM can make: the process wins.
    assert effective_min_wall_mm(FitConstraints(min_wall_mm=0.5, process="fdm"), fdm) == pytest.approx(1.2)
    # The design asks for more than resin needs: the design wins.
    assert effective_min_wall_mm(FitConstraints(min_wall_mm=0.8, process="sla"), sla) == pytest.approx(0.8)


# --------------------------------------------------------------------------- #
# The profile reaches the certificate
# --------------------------------------------------------------------------- #
def test_a_wall_violation_names_the_process_and_its_own_floor():
    """A reader must be able to tell which process's number convicted the part.

    The geometric end-to-end case (one part, two profiles, two verdicts) needs a
    fixture whose *measured* wall lands between two process floors; the fixture set
    does not have one yet, so the acceptance here is the threshold the verifier
    applies, taken from the profile and recorded in the evidence.
    """
    from pah.design.spec import FitConstraints
    from pah.tools.verifiers._design_fixtures import standard_fixture
    from pah.tools.verifiers.printability import Printability

    view = standard_fixture()
    # A floor no real wall can meet, so the certificate is guaranteed to fire and the
    # recorded numbers can be checked.
    view.spec = view.spec.edit(
        constraints=FitConstraints(min_wall_mm=50.0, process="sla", print_orientation_deg=(0.0, 0.0, 0.0))
    )
    found = Printability().check(view)

    wall = next(cert for cert in found if cert.code == "below_min_wall")
    assert wall.magnitude["process"] == "sla"
    # The design's own floor is the stricter one here, so it is what was applied.
    assert wall.magnitude["required_wall_mm"] == pytest.approx(50.0)


def test_the_verifier_carries_no_process_numbers_of_its_own():
    """The overhang table must be a view of the profile table, not a second copy."""
    from pah.design.process import PROCESS_PROFILES
    from pah.tools.verifiers.printability import PROCESS_MIN_ANGLE_FROM_HORIZONTAL_DEG

    assert PROCESS_MIN_ANGLE_FROM_HORIZONTAL_DEG == {
        name: profile.overhang_deg for name, profile in PROCESS_PROFILES.items()
    }
