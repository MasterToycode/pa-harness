"""The export gate: a file that says "printable" must have been checked.

The audit (`docs/remediation-plan.md` P0-2) found `export_run` deciding
`manifest["printable"] = not problems` where `problems` only listed file-level
trouble — so a report with `validated: partial` (no build orientation declared, so
overhang was never evaluated) still exported `printable: true`. That is the most
direct overclaim the harness had: the manifest is the thing a person hands to a
printer.

The rule now: a claim is exportable only when the report says `ok is true`,
`validated == "full"`, and `not_evaluated` is empty. `--for-print` turns the rule into
an exit code and decides **before** writing, so a refused export leaves no stray STL
that could be mistaken for a go-ahead.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import trimesh

from pah.cli.export import export_run


def _run_dir(tmp_path: Path, verdicts: dict, *, watertight: bool = True) -> Path:
    run = tmp_path / "run"
    geometry = run / "geometry"
    geometry.mkdir(parents=True)
    part = trimesh.creation.box(extents=(20.0, 20.0, 10.0))
    if not watertight:
        part = trimesh.Trimesh(part.vertices.copy(), part.faces[:-2].copy(), process=False)
    (geometry / "left.stl").write_bytes(part.export(file_type="stl"))
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


FULL = {
    "assemblable": {"ok": True, "validated": "full", "not_evaluated": [], "blocking": []},
    "printable": {
        "ok": True,
        "validated": "full",
        "not_evaluated": [],
        "blocking": [],
        "watertight": {"left": True},
    },
}

PARTIAL = {
    "assemblable": {
        "ok": True,
        "validated": "partial",
        "not_evaluated": ["physics_settle"],
        "blocking": [],
    },
    "printable": {
        "ok": True,
        "validated": "partial",
        "not_evaluated": ["printability"],
        "blocking": [],
        "watertight": {"left": True},
    },
}

UNKNOWN = {
    "assemblable": {
        "ok": None,
        "unknown": True,
        "validated": "unknown",
        "not_evaluated": ["physics_settle"],
        "blocking": [],
    },
    "printable": {
        "ok": None,
        "unknown": True,
        "validated": "unknown",
        "not_evaluated": ["physics_settle"],
        "blocking": [],
        "watertight": {"left": True},
    },
}


def test_a_fully_validated_run_is_both_printable_and_assembly_ready(tmp_path: Path):
    manifest = export_run(_run_dir(tmp_path, FULL))

    assert manifest["printable"] is True
    assert manifest["assembly_ready"] is True
    assert manifest["readiness"]["printable"]["reasons"] == []
    assert manifest["wrote_files"] is True


def test_a_partial_run_does_not_export_as_printable(tmp_path: Path):
    """The audit's exact case: ok is true, but the checks did not all run."""
    manifest = export_run(_run_dir(tmp_path, PARTIAL))

    assert manifest["printable"] is False
    assert manifest["assembly_ready"] is False
    reasons = manifest["readiness"]["printable"]["reasons"]
    assert any("partial" in reason for reason in reasons)
    assert any("printability" in reason for reason in reasons)
    # The files are still written: a person may want the meshes for inspection.
    assert manifest["wrote_files"] is True
    assert manifest["artifacts"]


def test_an_unknown_run_does_not_export_as_printable_either(tmp_path: Path):
    manifest = export_run(_run_dir(tmp_path, UNKNOWN))

    assert manifest["printable"] is False
    assert any("unknown" in reason for reason in manifest["readiness"]["printable"]["reasons"])


def test_a_missing_report_is_not_a_pass(tmp_path: Path):
    run = _run_dir(tmp_path, FULL)
    (run / "report.json").unlink()

    manifest = export_run(run)

    assert manifest["printable"] is False
    assert any("unknown" in reason for reason in manifest["readiness"]["printable"]["reasons"])


def test_a_non_watertight_part_blocks_both_claims(tmp_path: Path):
    manifest = export_run(_run_dir(tmp_path, FULL, watertight=False))

    assert manifest["printable"] is False
    assert manifest["assembly_ready"] is False
    assert any("not watertight" in problem for problem in manifest["problems"])


def test_strict_mode_refuses_before_writing_anything(tmp_path: Path):
    """先判后写: a refused export must leave no artefact someone could print."""
    run = _run_dir(tmp_path, PARTIAL)
    out = tmp_path / "print"

    manifest = export_run(run, out=out, strict=True)

    assert manifest["printable"] is False
    assert manifest["wrote_files"] is False
    assert manifest["artifacts"] == []
    assert not out.exists() or not any(out.iterdir())


def test_strict_mode_writes_when_the_claim_holds(tmp_path: Path):
    run = _run_dir(tmp_path, FULL)
    out = tmp_path / "print"

    manifest = export_run(run, out=out, strict=True)

    assert manifest["wrote_files"] is True
    assert (out / "left.stl").exists()
    assert (out / "assembly.3mf").exists()
    assert (out / "print-manifest.json").exists()


def test_the_manifest_pins_the_report_it_was_derived_from(tmp_path: Path):
    """`report_sha256` is what stops an export and a report drifting apart."""
    run = _run_dir(tmp_path, FULL)
    expected = __import__("hashlib").sha256((run / "report.json").read_bytes()).hexdigest()

    manifest = export_run(run)

    assert manifest["report_sha256"] == expected
    assert manifest["generator"] == "local-parametric"


def test_pah_export_for_print_exits_three_on_a_partial_report(tmp_path: Path):
    from pah.cli.main import main

    run = _run_dir(tmp_path, PARTIAL)
    out = tmp_path / "print"

    code = main(["export", str(run), "--for-print", "--out", str(out)])

    assert code == 3
    assert not out.exists() or not any(out.iterdir())


def test_pah_export_for_print_succeeds_on_a_full_report(tmp_path: Path):
    from pah.cli.main import main

    run = _run_dir(tmp_path, FULL)
    out = tmp_path / "print"

    code = main(["export", str(run), "--for-print", "--out", str(out)])

    assert code == 0
    assert (out / "print-manifest.json").exists()
