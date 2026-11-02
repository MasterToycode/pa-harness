"""Report coverage: three states, and no path from "did not run" to "passed".

The audit (`docs/remediation-plan.md` P0-1) found three ways the verdict block said
more than the evidence supported:

1. an N-part run skipped `insert_material` per interface but only wrote a note for
   `physics_settle`, so the skipped check counted as run;
2. a verifier its own sanity gate had disabled was skipped without a note, so
   "disabled" and "passed" produced identical `validated` strings;
3. environment certificates were filtered out *before* `ok` was computed, so a
   crashed physics verifier produced `assemblable.ok: true, validated: full` inside
   the same report whose episode verdict was `failed`.

These tests pin the fixes at the level the audit asked for: the report block, not the
individual verifier.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pah.app.runtime import _did_not_run, _verdicts, verdict_text
from pah.core.model import FailureCertificate


def _certificate(code: str = "simulation_failed", family: str = "environment") -> FailureCertificate:
    return FailureCertificate(
        code=code,
        family=family,  # type: ignore[arg-type]
        locus={"spec": "s"},
        magnitude={},
        observed_by="physics_settle",
    )


# --------------------------------------------------------------------------- #
# (3) environment certificate -> unknown, never "ok"
# --------------------------------------------------------------------------- #
def test_an_environment_certificate_makes_both_claims_unknown(tmp_path: Path):
    verdicts = _verdicts([_certificate()], {}, tmp_path)

    for name in ("assemblable", "printable"):
        block = verdicts[name]
        assert block["ok"] is None, f"{name} claimed {block['ok']!r} while a check crashed"
        assert block["unknown"] is True
        assert block["validated"] == "unknown"
        assert "physics_settle" in block["not_evaluated"]

    assert verdicts["environment_errors"][0]["observed_by"] == "physics_settle"
    assert verdict_text(verdicts["assemblable"]) == "未知"


def test_a_physical_defect_still_reads_as_no_and_not_as_unknown(tmp_path: Path):
    certificate = _certificate(code="geometric_interference", family="assembly")

    verdicts = _verdicts([certificate], {}, tmp_path)

    assert verdicts["assemblable"]["ok"] is False
    assert verdicts["assemblable"]["unknown"] is False
    assert verdicts["assemblable"]["blocking"] == ["geometric_interference"]
    assert verdict_text(verdicts["assemblable"]) == "否"


def test_a_clean_full_run_is_the_only_thing_that_reads_as_yes(tmp_path: Path):
    verdicts = _verdicts([], {}, tmp_path)

    assert verdicts["assemblable"]["ok"] is True
    assert verdicts["assemblable"]["validated"] == "full"
    assert verdict_text(verdicts["assemblable"]) == "是"


# --------------------------------------------------------------------------- #
# (1) an N-part run's skipped checks are named, and `validated` says partial
# --------------------------------------------------------------------------- #
def test_a_skipped_pair_check_is_listed_and_downgrades_the_claim(tmp_path: Path):
    notes = {
        "iface_a_b/insert_material": "not evaluated: decomposed part has no reference",
        "iface_a_b/physics_settle": "not evaluated: settle model is peg-in-bore",
    }

    verdicts = _verdicts([], notes, tmp_path)

    assert "insert_material" in verdicts["assemblable"]["not_evaluated"]
    assert "physics_settle" in verdicts["assemblable"]["not_evaluated"]
    assert verdicts["assemblable"]["validated"] == "partial"


def test_the_pair_skip_table_is_the_only_source_of_both_skip_and_note():
    """One table, read twice. The audit's bug was the two drifting apart."""
    from pah.tools.verifiers.design_cascade import _PAIR_SKIPS

    assert set(_PAIR_SKIPS) == {"insert_material", "physics_settle"}
    for reason in _PAIR_SKIPS.values():
        assert reason.startswith("not evaluated:")
        assert len(reason) > 40, "a skip reason has to say why, not just that"


def test_did_not_run_accepts_both_key_shapes():
    notes = {"printability": "x", "iface_1/insert_material": "y", "other": "z"}

    assert _did_not_run(("printability", "insert_material", "fit_clearance"), notes) == [
        "insert_material",
        "printability",
    ]
    assert _did_not_run(("fit_clearance",), notes) == []


# --------------------------------------------------------------------------- #
# (2) a sanity-disabled verifier is not evaluated either
# --------------------------------------------------------------------------- #
def test_a_disabled_verifier_is_recorded_as_not_run(tmp_path: Path):
    import trimesh

    from pah.design.decompose import (
        DecomposedPart,
        Decomposition,
        file_hash,
        parts_from_decomposition,
        store_parts,
    )
    from pah.design.materialize import materialize
    from pah.design.pairing import propose_interfaces, spec_from_parts
    from pah.tools.verifiers.design_cascade import (
        DesignVerifierCascade,
        default_registry,
    )

    registry = default_registry()
    decomposition = Decomposition(
        parts=(
            DecomposedPart(id="left", mesh=trimesh.creation.box(extents=(20.0, 20.0, 10.0))),
            DecomposedPart(
                id="right",
                mesh=trimesh.creation.box(
                    extents=(20.0, 20.0, 10.0),
                    transform=trimesh.transformations.translation_matrix((0.0, 0.0, 10.0)),
                ),
            ),
        ),
        provenance={"method": "test"},
    )
    out = tmp_path / "run"
    stored = store_parts(decomposition.parts, out)
    parts = parts_from_decomposition(
        decomposition, source_hashes={key: file_hash(path) for key, path in stored.items()}
    )
    spec = spec_from_parts(
        spec_id="disabled-printability",
        parts=parts,
        interfaces=propose_interfaces(parts).interfaces,
    )
    design = materialize(spec, out)

    cascade = DesignVerifierCascade(registry=registry, root=out)
    # Disable *after* construction: the cascade runs the registry's own sanity gate
    # when it is built, which would re-enable a verifier whose fixtures now pass.
    cascade.registry._disabled.add("printability")
    cascade.registry._reasons["printability"] = "its own fixtures disagreed"
    certificates = cascade.check(design.assembly)

    assert certificates == []
    # Notes are keyed `<interface>/<verifier>` on the per-pair path.
    disabled_note = [key for key in cascade.notes if key.endswith("/printability")]
    assert disabled_note, cascade.notes
    assert "disabled" in cascade.notes[disabled_note[0]]
    verdicts = _verdicts(certificates, cascade.notes, out / "geometry")
    assert verdicts["printable"]["validated"] == "partial"
    assert "printability" in verdicts["printable"]["not_evaluated"]


# --------------------------------------------------------------------------- #
# The whole CLI, on the case the audit actually ran
# --------------------------------------------------------------------------- #
def test_a_three_part_run_never_claims_full_validation(tmp_path: Path, capsys):
    import json as _json

    import trimesh

    from pah.cli.main import main

    source = tmp_path / "plate.stl"
    source.write_bytes(trimesh.creation.box(extents=(60.0, 40.0, 12.0)).export(file_type="stl"))
    out = tmp_path / "run"

    assert main(["decompose", str(source), "--parts", "3", "--out", str(out)]) == 0

    report = _json.loads((out / "report.json").read_text(encoding="utf-8"))
    assembly = report["verdicts"]["assemblable"]
    assert "physics_settle" in assembly["not_evaluated"]
    assert assembly["validated"] == "partial"
    assert assembly["ok"] is True  # nothing proved a defect; it just is not "full"

    printed = capsys.readouterr().out
    assert "(partial)" in printed
