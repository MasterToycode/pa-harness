"""`pah providers doctor` and the python-generator provider: the integration seam.

These tests cover the seam, not a model. What they pin down:

* a `[gen.providers.X] kind="python"` entry resolves to the callable it names, and
  each way of getting that wrong has its own sentence (no target, unimportable
  module, missing attribute, non-callable);
* the runtime's materialiser resolution returns the configured generator when one
  is configured, and T1's parametric builder when none is — never a silent mix;
* `pah providers doctor` exits 0/1 as the checks pass or fail, and `--call` really
  invokes the configured thing;
* an episode whose `[gen]` names a python adapter runs with it, which is the proof
  that a DiT/CAD adapter is a drop-in rather than a fork.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pah.app.providers import ProviderConfig, ProviderError, resolve_python_target

REPO = Path(__file__).resolve().parents[1]


def _config(target: str) -> ProviderConfig:
    return ProviderConfig(id="dit", kind="python", compat={"target": target})


# --------------------------------------------------------------------------- #
# target resolution
# --------------------------------------------------------------------------- #
def test_a_python_gen_target_resolves_to_the_callable_it_names():
    target = resolve_python_target(
        _config("examples.dit_materializer:stub_materialize")
    )
    assert callable(target)


def test_a_dotted_target_is_accepted_too():
    target = resolve_python_target(_config("examples.dit_materializer.stub_materialize"))
    assert callable(target)


@pytest.mark.parametrize(
    "target, fragment",
    [
        ("", "declares no `compat.target`"),
        ("definitely_not_a_module:materialize", "cannot import"),
        ("examples.dit_materializer:no_such_function", "has no attribute"),
        ("examples.dit_materializer:REPO", "not a callable"),
    ],
)
def test_every_way_of_getting_the_target_wrong_has_its_own_sentence(target, fragment):
    with pytest.raises(ProviderError, match=fragment):
        resolve_python_target(_config(target))


# --------------------------------------------------------------------------- #
# the composition root picks the configured generator, or T1
# --------------------------------------------------------------------------- #
def _write_config(tmp_path: Path, *, gen_target: str | None) -> Path:
    lines = ["[gen]", "default = \"dit\"", "", "[gen.providers.dit]"]
    if gen_target is None:
        lines.append('kind = "local-parametric"')
    else:
        lines.extend(['kind = "python"', f'compat = {{ target = "{gen_target}" }}'])
    path = tmp_path / "pah.toml"
    path.write_text("\n".join(lines) + "\n", "utf-8")
    return path


def test_no_gen_section_means_the_parametric_builder(tmp_path):
    from pah.app.runtime import resolve_generator

    path = tmp_path / "pah.toml"
    path.write_text("[llm]\ndefault = \"none\"\n", "utf-8")
    resolved = resolve_generator(path)
    assert resolved.__module__ == "pah.design.materialize"


def test_a_configured_python_generator_is_the_one_the_runtime_uses(tmp_path):
    from pah.app.runtime import resolve_generator

    path = _write_config(tmp_path, gen_target="examples.dit_materializer:stub_materialize")
    resolved = resolve_generator(path)
    assert resolved.__name__ == "stub_materialize"


def test_a_declared_but_broken_generator_refuses_instead_of_falling_back(tmp_path):
    """The report names the generator; a quiet fallback would make it a lie."""
    from pah.app.runtime import resolve_generator

    path = _write_config(tmp_path, gen_target="no_such_module:materialize")
    with pytest.raises(ProviderError, match="cannot import"):
        resolve_generator(path)


# --------------------------------------------------------------------------- #
# pah providers doctor
# --------------------------------------------------------------------------- #
def _doctor(*args: str) -> tuple[int, dict]:
    from pah.cli.main import main

    code = main(["providers", "doctor", *args, "--json"])
    return code, {}


def test_doctor_reports_the_llm_side_without_spending_a_call(tmp_path, capsys):
    from pah.cli.main import main

    config = tmp_path / "pah.toml"
    config.write_text(
        "\n".join(
            [
                "[llm]",
                'default = "fake"',
                "[llm.providers.fake]",
                'kind = "openai-compatible"',
                'base_url = "http://127.0.0.1:9/v1"',
                'api_key_env = "PAH_TEST_KEY_NOT_SET"',
                'model = "test-model"',
                'billing = "free"',
            ]
        )
        + "\n",
        "utf-8",
    )
    code = main(["providers", "doctor", "--role", "llm", "--config", str(config), "--json"])
    payload = json.loads(capsys.readouterr().out)
    # The key is not in this shell, so the credential check fails — which is exactly
    # the answer a first integration needs on line one.
    assert code == 1
    rows = {row["check"]: row["ok"] for row in payload["sections"]["llm"]["checks"]}
    assert rows["credential"] is False
    assert rows["provider configured"] is True
    assert rows["billing declared"] is True


def test_doctor_passes_a_generator_self_test_with_the_stub(tmp_path, capsys):
    from pah.cli.main import main

    config = _write_config(tmp_path, gen_target="examples.dit_materializer:stub_materialize")
    code = main(
        [
            "providers", "doctor", "--role", "gen", "--call",
            "--config", str(config),
            "--spec", str(REPO / "fixtures" / "design" / "demo-peg-hole-clear.json"),
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert code == 0, payload
    call = payload["sections"]["gen"]["call"]
    assert call["ok"] is True
    assert set(call["parts"]) == {"body", "insert"}
    assert all(part["watertight"] for part in call["parts"].values())


# --------------------------------------------------------------------------- #
# an episode really runs on a configured python generator
# --------------------------------------------------------------------------- #
def test_an_episode_materialises_through_the_configured_generator(tmp_path):
    """The port is a drop-in: same report, different `[gen]` entry."""
    from pah.cli.main import main

    config = _write_config(tmp_path, gen_target="examples.dit_materializer:stub_materialize")
    out = tmp_path / "ep"
    code = main(
        [
            "run",
            "--spec", str(REPO / "fixtures" / "design" / "demo-peg-hole-clear.json"),
            "--policy", "rule",
            "--rounds", "2",
            "--out", str(out),
            "--config", str(config),
        ]
    )
    assert code == 0
    report = json.loads((out / "report.json").read_text("utf-8"))
    assert report["episode"]["verdict"] in ("repaired", "exhausted", "infeasible")
    assert report["geometry"]["complete"] is True

