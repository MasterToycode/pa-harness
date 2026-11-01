"""The chain from "the part is hollow" to "the generator is asked differently".

Three pieces, and the test proves they are connected rather than merely present:

    insert_material certificate  ->  RulePolicy proposes regenerate_interface with a
    prompt_hint  ->  the operator writes the hint into spec.provenance  ->  the DiT
    adapter reads it and puts it in the model's prompt

Also: `pah export` writes what a printer takes (STL/3MF) and refuses to call a run
printable when the report says otherwise.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pah.core.model import Budget, FailureCertificate
from pah.design.api import assembly_from_spec, spec_of
from pah.policy.design_ops import register_builtin
from pah.policy.design_policies import MATERIAL_PROMPT_HINT, RulePolicy
from pah.policy.registry import OperatorRegistry

REPO = Path(__file__).resolve().parents[1]


def _assembly():
    from pah.cli.design import default_spec

    return assembly_from_spec(spec=default_spec(kind="peg_hole", requirement="a peg"))


def _material_cert() -> FailureCertificate:
    return FailureCertificate(
        code="insufficient_material",
        family="geometry",
        locus={"part": "insert"},
        magnitude={"volume_ratio": 0.29},
        observed_by="insert_material",
    )


# --------------------------------------------------------------------------- #
# the chain
# --------------------------------------------------------------------------- #
def test_a_hollow_part_is_repaired_by_asking_the_generator_again():
    registry = OperatorRegistry()
    register_builtin(registry)
    plan = RulePolicy(registry=registry).propose(
        _assembly(), [_material_cert()], [], Budget(rounds=3, geom_calls=100, sim_calls=10)
    )

    assert plan is not None
    assert plan.operator == "regenerate_interface"
    assert plan.params["params"] == {}  # no number was wrong
    assert plan.params["prompt_hint"] == MATERIAL_PROMPT_HINT
    assert "insufficient_material" in plan.reason


def test_the_hint_travels_through_the_design_not_through_the_policy():
    """Replaying the spec must reproduce the same prompt, so the hint is provenance."""
    registry = OperatorRegistry()
    register_builtin(registry)
    plan = RulePolicy(registry=registry).propose(
        _assembly(), [_material_cert()], [], Budget(rounds=3, geom_calls=100, sim_calls=10)
    )

    after = registry.apply(_assembly(), plan)

    assert spec_of(after).provenance["prompt_hint"] == MATERIAL_PROMPT_HINT


def test_the_generator_prompt_carries_the_hint():
    from examples.hunyuan3d_materializer import _prompt_for
    from pah.cli.design import default_spec

    spec = default_spec(kind="peg_hole", requirement="a peg").edit(
        provenance={"prompt_hint": MATERIAL_PROMPT_HINT}
    )

    prompt = _prompt_for(spec)

    assert "SOLID part" in prompt
    assert "no hollow shell" in prompt


def test_a_spec_without_a_hint_does_not_get_one_invented():
    from examples.hunyuan3d_materializer import _prompt_for
    from pah.cli.design import default_spec

    prompt = _prompt_for(default_spec(kind="peg_hole", requirement="a peg"))

    assert "SOLID part" not in prompt


# --------------------------------------------------------------------------- #
# export
# --------------------------------------------------------------------------- #
def _fake_run(tmp_path: Path, *, printable: bool) -> Path:
    import trimesh

    run = tmp_path / "run"
    geometry = run / "geometry"
    geometry.mkdir(parents=True)
    trimesh.creation.box((40, 20, 6)).export(geometry / "body.stl")
    peg = trimesh.creation.cylinder(radius=4, height=14, sections=48)
    peg.export(geometry / "insert.stl")
    spec = json.loads(
        (REPO / "fixtures/design/demo-peg-hole-clear.json").read_text("utf-8")
    )
    (run / "spec.json").write_text(json.dumps(spec), "utf-8")
    report = {
        "run": {"generator": "test:python"},
        "verdicts": {
            "printable": {
                "ok": printable,
                "blocking": [] if printable else ["insufficient_material"],
                "not_evaluated": [],
                "watertight": {"body": True, "insert": True},
                "validated": "full",
            }
        },
    }
    (run / "report.json").write_text(json.dumps(report), "utf-8")
    return run


def test_export_writes_a_3mf_and_a_manifest_for_a_printable_run(tmp_path, capsys):
    from pah.cli.main import main

    run = _fake_run(tmp_path, printable=True)

    code = main(["export", str(run), "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert code == 0
    assert payload["printable"] is True
    assert payload["units"] == "mm"
    assert (run / "print" / "assembly.3mf").exists()
    assert (run / "print" / "print-manifest.json").exists()
    assert set(payload["parts"]) == {"body", "insert"}


def test_export_refuses_to_call_a_run_printable_when_the_report_says_no(tmp_path, capsys):
    from pah.cli.main import main

    run = _fake_run(tmp_path, printable=False)

    code = main(["export", str(run), "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert code == 1
    assert payload["printable"] is False
    assert any("insufficient_material" in problem for problem in payload["problems"])


def test_glb_and_urdf_are_refused_as_printing_formats(capsys):
    """The manifest says it, and the command enforces it: neither is an order form."""
    from pah.cli.main import main

    code = main(["export", "runs/verdict-test", "--formats", "glb"])
    output = capsys.readouterr().out

    assert code != 0
    assert "not printing formats" in output
