"""STEP export of the fit features (P2-5): the read-back is the evidence.

What is pinned here
-------------------
* **C5's numbers**: a peg-hole design exports a STEP; the file is imported again and
  the hole's and peg's diameters are measured *from the geometry* and compared with
  the spec within 0.01 mm. The test also tampers with the expected value to prove the
  measurement is real rather than a copy of the parameters.
* **The chain**: a three-part decomposed spec produces exactly one feature per
  interface, keyed by the interface id, with the host blanks declared as
  "not converted" — this module never turns a mesh into B-Rep.
* **The boundary**: a kernel failure is a `problems` row and no STEP file; a missing
  kernel is a named refusal; `--step` off leaves the manifest exactly as before
  (no key, flat `artifacts`), and a step failure never changes `printable`.
* **Two product classes**: with `--step`, `artifacts` entries carry `kind`
  (`print_mesh` / `engineering_step` / `manifest`).

Offline and bounded: build123d is local, there is no slicer and no network. The
STEP round-trip is the slow part (~1 s per export), so the fixtures are small.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("build123d", reason="STEP export needs the local B-Rep kernel")

from pah.cli.design import default_spec
from pah.cli.export import export_run
from pah.design.materialize import materialize
from pah.design.spec import FitConstraints
from pah.tools import step_export

FULL_VERDICTS = {
    "assemblable": {"ok": True, "validated": "full", "not_evaluated": [], "blocking": []},
    "printable": {"ok": True, "validated": "full", "not_evaluated": [], "blocking": []},
}

PARTIAL_VERDICTS = {
    "assemblable": {"ok": True, "validated": "full", "not_evaluated": [], "blocking": []},
    "printable": {
        "ok": True,
        "validated": "partial",
        "not_evaluated": ["printability"],
        "blocking": [],
    },
}


def peg_hole_spec(*, spec_id: str = "step-peg-hole"):
    """The proposal's defect, as a spec: a 10.2 mm peg ordered for a 10.0 mm hole.

    Hole 10.0 / peg 10.2 is the pair the audit's C5 names, so the read-back has two
    distinguishable diameters to find in one file.
    """
    spec = default_spec(kind="peg_hole", spec_id=spec_id)
    return spec.with_interface(params={"d_peg_mm": 10.2, "d_hole_mm": 10.0})


def run_dir(tmp_path: Path, spec, *, verdicts: dict = FULL_VERDICTS) -> Path:
    """A run directory as `pah run` leaves it, narrowed to what export reads."""
    run = tmp_path / "run"
    materialize(spec, run)
    (run / "report.json").write_text(
        json.dumps(
            {
                "schema": "pah-run-report/2",
                "run": {"generator": "local-parametric"},
                "verdicts": verdicts,
            }
        ),
        encoding="utf-8",
    )
    return run


# --------------------------------------------------------------------------- #
# C5: export, read back, compare
# --------------------------------------------------------------------------- #
def test_a_peg_hole_step_reproduces_the_spec_diameters(tmp_path):
    spec = peg_hole_spec()
    target = tmp_path / "engineering.step"

    result = step_export.export_features(spec, None, target)

    assert result.ok is True, result.problems
    assert result.path == str(target)
    assert target.exists() and target.stat().st_size > 0
    # The audit's numbers, straight off the parameters.
    feature = result.features[0]
    assert feature["interface_id"] == "iface_0"
    assert feature["kind"] == "peg_hole"
    assert feature["nominal_dims_mm"]["d_hole_mm"] == pytest.approx(10.0)
    assert feature["nominal_dims_mm"]["d_peg_mm"] == pytest.approx(10.2)

    check = result.dimension_check
    assert check["tolerance_mm"] == 0.01
    assert check["ok"] is True
    measured = {entry["dimension"]: entry for entry in check["checked"]}
    assert measured["hole_diameter_mm"]["spec_mm"] == pytest.approx(10.0)
    assert measured["peg_diameter_mm"]["spec_mm"] == pytest.approx(10.2)
    for entry in measured.values():
        assert entry["measured_mm"] is not None
        assert entry["error_mm"] <= 0.01
        assert entry["ok"] is True
    # The measurement is from cylindrical faces in the file, not from the parameters.
    assert sorted(check["observed_diameters_mm"]) == pytest.approx([10.0, 10.2])


def test_the_dimension_check_reads_the_file_not_the_parameters(tmp_path):
    """Tempting a pass: ask for a diameter the STEP does not contain."""
    path = tmp_path / "engineering.step"
    assert step_export.export_features(peg_hole_spec(), None, path).ok is True

    check = step_export.measure_step(
        path,
        [
            {
                "interface_id": "iface_0",
                "dimension": "hole_diameter_mm",
                "spec_mm": 9.7,  # the file has 10.0 (and a 10.2 peg)
            }
        ],
    )

    assert check["ok"] is False
    entry = check["checked"][0]
    assert entry["measured_mm"] == pytest.approx(10.0)
    assert entry["error_mm"] == pytest.approx(0.3)


def test_the_step_and_the_mesh_are_within_the_audit_tolerance(tmp_path):
    """C5's second half: STL vs STEP fit dimension within 0.01 mm.

    The STL is a 64-gon inscribed in the nominal circle, so the manifest reports the
    bound that tessellation implies — radial, not a caliper measurement — and it must
    stay inside the audit's 0.01 mm.
    """
    result = step_export.export_features(peg_hole_spec(), None, tmp_path / "e.step")

    bound = result.dimension_check["stl_step_bound"]
    assert bound["max_stl_vs_step_radial_mm"] <= 0.01
    assert bound["ok"] is True
    assert "inscribed" in bound["basis"]


# --------------------------------------------------------------------------- #
# The three-part chain: one feature per interface, blanks stay meshes
# --------------------------------------------------------------------------- #
def test_a_three_part_chain_exports_one_feature_per_interface(tmp_path):
    import trimesh

    from pah.design.decompose import (
        file_hash,
        parts_from_decomposition,
        split_slabs,
        store_parts,
    )
    from pah.design.pairing import propose_interfaces, spec_from_parts

    decomposition = split_slabs(
        trimesh.creation.box(extents=(60.0, 40.0, 12.0)), count=3, axis=2
    )
    run = tmp_path / "chain"
    stored = store_parts(decomposition.parts, run)
    parts = parts_from_decomposition(
        decomposition,
        source_hashes={key: file_hash(path) for key, path in stored.items()},
    )
    pairing = propose_interfaces(parts)
    spec = spec_from_parts(
        spec_id="chain-3",
        parts=parts,
        interfaces=pairing.interfaces,
        constraints=FitConstraints(print_orientation_deg=(0.0, 0.0, 0.0)),
    )

    result = step_export.export_features(spec, None, run / "engineering.step")

    assert result.ok is True, result.problems
    expected_ids = [interface.id for interface in pairing.interfaces]
    assert len(expected_ids) == 2
    assert [feature["interface_id"] for feature in result.features] == expected_ids
    # Decomposed blanks are never converted: their negative features travel as void
    # solids and the bodies are declared as untouched meshes.
    assert all(
        feature["negative_representation"] == "void_solid" for feature in result.features
    )
    representations = {body["part_id"]: body["representation"] for body in result.bodies}
    assert set(representations) == {part.id for part in parts}
    assert set(representations.values()) == {"mesh_blank_not_converted"}
    # Every interface's hole and peg diameter was read back from the file.
    assert len(result.dimension_check["checked"]) == 4
    assert all(entry["ok"] for entry in result.dimension_check["checked"])


# --------------------------------------------------------------------------- #
# Failure paths: no silent success, no half-written file
# --------------------------------------------------------------------------- #
def test_a_kernel_failure_becomes_a_problem_and_writes_no_file(tmp_path, monkeypatch):
    def explode(*_args, **_kwargs):
        raise RuntimeError("the kernel exploded")

    monkeypatch.setattr(step_export, "_features_for", explode)
    target = tmp_path / "engineering.step"

    result = step_export.export_features(peg_hole_spec(), None, target)

    assert result.ok is False
    assert result.path is None
    assert any("kernel exploded" in problem for problem in result.problems)
    assert not target.exists()


def test_a_missing_kernel_is_a_named_refusal(tmp_path, monkeypatch):
    def missing():
        raise step_export.StepExportError(
            "build123d/OCP is not importable, so a STEP cannot be written"
        )

    monkeypatch.setattr(step_export, "_build123d", missing)

    with pytest.raises(step_export.StepExportError, match="build123d"):
        step_export.export_features(peg_hole_spec(), None, tmp_path / "e.step")


def test_a_cylinder_missing_from_the_file_fails_the_check(tmp_path):
    """A STEP without the cylinder cannot vouch for the dimension."""
    path = tmp_path / "empty.step"
    # A box only: no cylindrical face anywhere.
    from build123d import Box, export_step

    export_step(Box(10.0, 10.0, 10.0), str(path))

    check = step_export.measure_step(
        path,
        [{"interface_id": "iface_0", "dimension": "hole_diameter_mm", "spec_mm": 10.0}],
    )

    assert check["ok"] is False
    assert check["checked"][0]["measured_mm"] is None
    assert any("no cylindrical face" in problem for problem in check["problems"])


# --------------------------------------------------------------------------- #
# The export manifest
# --------------------------------------------------------------------------- #
def test_step_off_leaves_the_manifest_and_its_artifacts_unchanged(tmp_path):
    """The capability must not rewrite manifests for runs that did not ask for it."""
    run = run_dir(tmp_path, peg_hole_spec())
    out = tmp_path / "print"

    manifest = export_run(run, out=out)

    assert "engineering_step" not in manifest
    assert manifest["printable"] is True
    assert manifest["slicing"] is None
    assert manifest["artifacts"] and all(isinstance(item, str) for item in manifest["artifacts"])
    assert not (out / "engineering.step").exists()


def test_step_adds_an_engineering_block_without_touching_printable(tmp_path):
    run = run_dir(tmp_path, peg_hole_spec())
    out = tmp_path / "print"

    manifest = export_run(run, out=out, step=True)

    engineering = manifest["engineering_step"]
    assert engineering["ok"] is True, engineering["problems"]
    assert Path(engineering["path"]) == out / "engineering.step"
    assert (out / "engineering.step").exists()
    assert engineering["features"][0]["interface_id"] == "iface_0"
    assert engineering["dimension_check"]["ok"] is True
    # The print claim is a separate question and must not move.
    assert manifest["printable"] is True
    # Two product classes, each named.
    kinds = {item["kind"] for item in manifest["artifacts"]}
    assert {"print_mesh", "engineering_step", "manifest"} <= kinds
    assert all("path" in item for item in manifest["artifacts"])


def test_a_step_failure_does_not_make_the_print_claim_false(tmp_path, monkeypatch):
    def missing():
        raise step_export.StepExportError("build123d/OCP is not importable")

    monkeypatch.setattr(step_export, "_build123d", missing)
    run = run_dir(tmp_path, peg_hole_spec())
    out = tmp_path / "print"

    manifest = export_run(run, out=out, step=True)

    assert manifest["printable"] is True
    assert manifest["engineering_step"]["ok"] is False
    assert manifest["engineering_step"]["path"] is None
    assert any("build123d" in problem for problem in manifest["problems"])
    assert not (out / "engineering.step").exists()


def test_a_partial_report_keeps_its_print_refusal_even_with_a_valid_step(tmp_path):
    """Only append: a STEP must not rescue a print claim that did not validate."""
    run = run_dir(tmp_path, peg_hole_spec(), verdicts=PARTIAL_VERDICTS)
    out = tmp_path / "print"

    manifest = export_run(run, out=out, step=True)

    assert manifest["printable"] is False
    assert manifest["engineering_step"]["ok"] is True
    assert any("partial" in reason for reason in manifest["readiness"]["printable"]["reasons"])


def test_a_missing_spec_leaves_a_stepless_but_honest_block(tmp_path):
    run = run_dir(tmp_path, peg_hole_spec())
    (run / "spec.json").unlink()

    manifest = export_run(run, out=tmp_path / "print", step=True)

    assert manifest["engineering_step"]["ok"] is False
    assert manifest["engineering_step"]["path"] is None
    assert any("spec.json" in problem for problem in manifest["engineering_step"]["problems"])


def test_the_cli_writes_the_step_and_reports_it(tmp_path, capsys):
    from pah.cli.main import main

    run = run_dir(tmp_path, peg_hole_spec())
    out = tmp_path / "print"

    code = main(["export", str(run), "--step", "--out", str(out)])

    assert code == 0
    printed = capsys.readouterr().out
    assert "engineering" in printed
    manifest = json.loads((out / "print-manifest.json").read_text("utf-8"))
    assert manifest["engineering_step"]["ok"] is True
    assert manifest["printable"] is True


def test_the_cli_exits_one_when_the_asked_for_step_fails(tmp_path, monkeypatch, capsys):
    from pah.cli.main import main

    def missing():
        raise step_export.StepExportError("build123d/OCP is not importable")

    monkeypatch.setattr(step_export, "_build123d", missing)
    run = run_dir(tmp_path, peg_hole_spec())

    code = main(["export", str(run), "--step", "--out", str(tmp_path / "print")])

    assert code == 1  # printable held, the requested engineering artefact did not
