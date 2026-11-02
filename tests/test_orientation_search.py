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

import json
import tempfile
from pathlib import Path

import pytest
import trimesh

from pah.design.orientation import CANDIDATES, Orientation, recommend
from pah.design.api import DesignView
from pah.tools.verifiers._design_fixtures import standard_fixture
from pah.tools.verifiers.printability import Printability

#: A shipped demo spec: the "no orientation declared" case a user actually runs.
DEMO_SPEC = Path(__file__).resolve().parents[1] / "fixtures" / "design" / "demo-peg-hole-clear.json"


def _scratch_dir() -> Path:
    """A throwaway run directory for the materialiser (one per call, never reused)."""
    return Path(tempfile.mkdtemp(prefix="pah-orient-test-"))


def _run_demo(out: Path, spec_path: Path | None = None) -> Path:
    """`pah run` one demo spec, returning the episode directory that holds the report."""
    from pah.cli.main import main

    if spec_path is None:
        spec_path = out.parent / "spec.json"
        spec_path.write_text(DEMO_SPEC.read_text("utf-8"), "utf-8")
    code = main(["run", "--spec", str(spec_path), "--policy", "rule", "--rounds", "2", "--out", str(out)])
    assert code == 0, "a clean demo spec must run to a verdict"
    return out


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


# --------------------------------------------------------------------------- #
# The verifier integration: no declaration searches instead of declining (P2-2)
# --------------------------------------------------------------------------- #
def demo_view(name: str = "demo-peg-hole-clear"):
    """The built solids of a shipped demo spec, the thing a user actually runs."""
    from pah.design.api import DesignView
    from pah.design.materialize import materialize
    from pah.design.spec import DesignSpec

    spec = DesignSpec.model_validate(json.loads(DEMO_SPEC.read_text("utf-8")))
    design = materialize(spec, _scratch_dir())
    return DesignView(
        spec=design.spec,
        root=design.root,
        meshes=dict(design.meshes),
        paths=dict(design.paths),
        mate_axis=design.mate_axis,
        approach_mm=design.approach_mm,
    )


def test_an_undeclared_demo_part_is_judged_in_a_recommended_pose():
    """`not evaluated` is gone: the part is judged in its best of 24 poses."""
    view = demo_view()
    assert view.spec.constraints.print_orientation_deg is None

    verifier = Printability()
    certificates = verifier.check(view)

    assert certificates == []
    assert verifier.overhang_evaluated is True
    assert verifier.orientation_source == "recommended"
    assert verifier.recommendation is not None
    assert verifier.orientation_deg == verifier.recommendation.angles
    # The report needs the trade the winner won, not just the winner.
    assert len(verifier.orientation_top) == 3
    assert verifier.orientation_top[0] is verifier.recommendation
    assert verifier.orientation_top[0].bed_area_mm2 >= 0.0
    assert verifier.last_note == ""


def test_a_declared_orientation_is_not_overruled_by_the_search():
    """A person's pose wins; the search only fills a gap (and says which it did)."""
    view = demo_view()
    declared = view.spec.edit(
        constraints=view.spec.constraints.model_copy(
            update={"print_orientation_deg": (0.0, 0.0, 0.0)}
        )
    )
    verifier = Printability()
    verifier.check(DesignView(spec=declared, meshes=dict(view.meshes)))

    assert verifier.overhang_evaluated is True
    assert verifier.orientation_source == "declared"
    assert verifier.orientation_deg == (0.0, 0.0, 0.0)
    assert verifier.recommendation is None
    assert verifier.orientation_top == ()


def test_the_report_prints_the_recommended_orientation_column(tmp_path):
    """C2: the report keeps "declared" and "recommended" as two different claims."""
    out = _run_demo(tmp_path / "ep0")

    report = json.loads((out / "report.json").read_text("utf-8"))
    block = report["verdicts"]["printable"]["orientation"]

    assert block["source"] == "recommended"
    assert len(block["top"]) == 3
    assert block["angles_deg"] is not None
    assert block["top"][0]["angles_deg"] == block["angles_deg"]
    # The overhang claim was made, so it must not still be listed as not evaluated.
    assert "printability" not in report["verdicts"]["printable"]["not_evaluated"]


def test_the_report_prints_the_declared_orientation_column(tmp_path):
    payload = json.loads(DEMO_SPEC.read_text("utf-8"))
    payload["constraints"]["print_orientation_deg"] = [0.0, 0.0, 0.0]
    spec_path = tmp_path / "declared.json"
    spec_path.write_text(json.dumps(payload), "utf-8")

    out = _run_demo(tmp_path / "ep1", spec_path)

    report = json.loads((out / "report.json").read_text("utf-8"))
    block = report["verdicts"]["printable"]["orientation"]

    assert block["source"] == "declared"
    assert block["angles_deg"] == [0.0, 0.0, 0.0]
    assert block["top"] == []  # no search ran, so there is no trade to show


def test_an_unreachable_cascade_reports_unknown_rather_than_a_guess():
    """A report must never invent an orientation it could not read."""
    from pah.app.runtime import _orientation_block

    assert _orientation_block(None) == {"source": "unknown"}
    assert _orientation_block(object()) == {"source": "unknown"}


@pytest.mark.parametrize(
    ("source", "angles", "evaluated"),
    [
        ("declared", [0.0, 0.0, 0.0], True),
        ("recommended", [0.0, 180.0, 0.0], True),
        ("none", None, False),  # subtractive: no rule to apply a pose to
        ("unknown", None, False),  # no report to read
    ],
)
def test_the_eval_layer_reads_the_column_not_the_declaration(source, angles, evaluated):
    """`manufacturing_evaluated` follows the run, not the spec (audit P2-2).

    Before the search, an undeclared spec was "not evaluated" — true then, false now,
    because the run evaluates a recommended pose. The eval summary prints this flag,
    so it has to agree with the report it summarises.
    """
    from pah.eval.runner import _manufacturing

    report = {
        "verdicts": {
            "printable": {
                "orientation": {"source": source, "angles_deg": angles, "top": []}
            }
        }
    }
    assert _manufacturing(report) == (evaluated, angles)
    assert _manufacturing(None) == (False, None)

