"""The paid-generation gate: configuration alone must not be able to spend money.

Why this file exists
--------------------
On 2026-11-02 a read-only audit ran `scripts/acceptance.py` against the default
`pah.toml` (`[gen] default = "hunyuan"`). Two mechanisms combined to call a paid
service: the adapter loaded `.env` at import time, and resolving a generator did not
ask anyone. One real 混元生3D job was submitted before the run was killed.

These tests pin the two halves of the fix, each on the layer that failed:

* **no import-time credential loading** — checked with `ast` (the mechanism) and by
  importing the adapter with a recording `load_env_file` (the behaviour);
* **no resolve without a yes** — checked against the socket layer, so "zero network"
  is a claim about the process, not about a mock.

The gate is judged on what a provider *declares* (a billing unit, or a vendor adapter
under `examples/`), never on its id — a test that hardcoded `"hunyuan"` would pass
while the next vendor adapter spent money.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import socket
import sys
from pathlib import Path

import pytest

from pah.app.providers import (
    PAID_GENERATION_VAR,
    PaidGenerationRefused,
    ProviderConfig,
    generation_is_paid,
    paid_generation_allowed,
    resolve_generator,
)

REPO = Path(__file__).resolve().parents[1]
SPEC = REPO / "fixtures" / "design" / "demo-peg-hole-clear.json"
HUNYUAN = REPO / "examples" / "hunyuan3d_materializer.py"

#: A stand-in for a vendor adapter. It re-exports the shipped parametric generator, so
#: an authorised run really completes while nothing in it can reach the network — the
#: property that lets a *paid* provider be tested offline.
PAID_ADAPTER = '''\
"""A paid-provider test double: parametric output, zero network."""

from pah.design.materialize import materialize

__all__ = ["materialize"]
'''


@pytest.fixture(autouse=True)
def _closed_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test starts with the gate closed.

    `pah run --allow-paid-gen` writes the variable into the process environment on
    purpose; without this, one test's yes would authorise the next test's paid
    provider, and the isolation the tests assert would be gone.
    """
    monkeypatch.delenv(PAID_GENERATION_VAR, raising=False)


def _paid_config(tmp_path: Path, *, billing: str = "per_call") -> Path:
    """A `pah.toml` whose only generator declares a paid billing unit."""
    (tmp_path / "paid_adapter.py").write_text(PAID_ADAPTER, "utf-8")
    config = tmp_path / "pah.toml"
    config.write_text(
        "\n".join(
            [
                "[gen]",
                'default = "paid_stub"',
                "",
                "[gen.providers.paid_stub]",
                'kind = "python"',
                f'billing = "{billing}"',
                'compat = { target = "paid_adapter.py:materialize" }',
            ]
        )
        + "\n",
        "utf-8",
    )
    return config


# --------------------------------------------------------------------------- #
# the refusal
# --------------------------------------------------------------------------- #
def test_a_billing_provider_is_refused_without_the_flag(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = _paid_config(tmp_path)

    with pytest.raises(PaidGenerationRefused) as error:
        resolve_generator(config)

    assert isinstance(error.value, RuntimeError)  # what the CLI's exit 3 is built on
    message = str(error.value)
    assert "--allow-paid-gen" in message
    assert "PAH_ALLOW_PAID_GEN" in message
    assert "不会调用" in message  # "this call does not happen" is the actionable half


def test_the_runtime_resolver_meets_the_same_gate(tmp_path, monkeypatch):
    """`pah run` resolves through `pah.app.runtime.resolve_generator`; it must refuse
    too, or the gate would only cover the spellings nobody uses."""
    monkeypatch.chdir(tmp_path)
    config = _paid_config(tmp_path)

    from pah.app.runtime import resolve_generator as runtime_resolve_generator

    with pytest.raises(PaidGenerationRefused):
        runtime_resolve_generator(config)


def test_the_flag_opens_the_gate(tmp_path, monkeypatch):
    monkeypatch.setenv(PAID_GENERATION_VAR, "1")
    monkeypatch.chdir(tmp_path)
    config = _paid_config(tmp_path)

    resolved = resolve_generator(config)

    assert resolved is not None
    assert resolved.__module__ == "pah.design.materialize"


def test_the_two_documented_spellings_open_the_gate():
    assert paid_generation_allowed({PAID_GENERATION_VAR: "1"}) is True
    assert paid_generation_allowed({PAID_GENERATION_VAR: "true"}) is True
    assert paid_generation_allowed({PAID_GENERATION_VAR: "TRUE"}) is True
    # A typo must leave it closed: the cost of refusing is one flag, the other way
    # round is real money.
    assert paid_generation_allowed({PAID_GENERATION_VAR: "yes"}) is False
    assert paid_generation_allowed({}) is False


# --------------------------------------------------------------------------- #
# the criterion: declared, never named
# --------------------------------------------------------------------------- #
def test_a_vendor_adapter_path_is_paid_even_without_a_billing_line():
    """The shipped `pah.toml` names the Hunyuan adapter by path and declares no
    billing; the path under `examples/` is itself the declaration."""
    config = ProviderConfig(
        id="hunyuan",
        kind="python",
        compat={"target": "examples/hunyuan3d_materializer.py:materialize"},
    )
    assert generation_is_paid(config) is True


def test_an_absolute_path_to_an_example_adapter_is_still_paid():
    """A Windows drive letter contains a colon, and `target` splits on one. If the
    split took the first colon, an absolute path would classify as a module name and
    the gate would let a paid adapter through."""
    target = f"{REPO / 'examples' / 'hunyuan3d_materializer.py'}:materialize"
    config = ProviderConfig(id="hunyuan", kind="python", compat={"target": target})
    assert generation_is_paid(config) is True


def test_a_local_model_imported_by_module_is_not_paid():
    """A DiT a user runs on their own GPU is resolved by module name and declares no
    cost; the gate must not block it just because the adapter is a python kind."""
    config = ProviderConfig(
        id="dit", kind="python", compat={"target": "my_dit.materialize:materialize"}
    )
    assert generation_is_paid(config) is False


def test_local_parametric_is_free_by_construction():
    assert generation_is_paid(ProviderConfig(id="local", kind="local-parametric")) is False


def test_billing_free_is_a_declaration_of_no_cost():
    config = ProviderConfig(
        id="stub", kind="python", billing="free",
        compat={"target": "my_dit.materialize:materialize"},
    )
    assert generation_is_paid(config) is False


# --------------------------------------------------------------------------- #
# the CLI: exit 3, no report, no socket
# --------------------------------------------------------------------------- #
def test_pah_run_refuses_a_paid_generator_without_the_flag(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    config = _paid_config(tmp_path)
    out = tmp_path / "out"

    from pah.cli.main import main

    code = main(
        [
            "run", "--spec", str(SPEC), "--policy", "rule", "--rounds", "1",
            "--out", str(out), "--config", str(config),
        ]
    )

    assert code == 3
    assert not (out / "report.json").exists()
    assert "--allow-paid-gen" in capsys.readouterr().out


def test_a_refused_run_opens_no_socket(tmp_path, monkeypatch):
    """Asserted on `socket`, not on `urllib`: the claim is about the process, and a
    mock of one caller would let every other caller through."""
    monkeypatch.chdir(tmp_path)
    config = _paid_config(tmp_path)
    attempts: list[object] = []

    def _forbidden(*args, **kwargs):
        attempts.append(args)
        raise AssertionError("the paid gate must refuse before any socket is opened")

    monkeypatch.setattr(socket, "socket", _forbidden)
    monkeypatch.setattr(socket, "create_connection", _forbidden)

    from pah.cli.main import main

    code = main(
        [
            "run", "--spec", str(SPEC), "--policy", "rule", "--rounds", "1",
            "--out", str(tmp_path / "out"), "--config", str(config),
        ]
    )

    assert code == 3
    assert attempts == []


def test_an_authorised_run_records_the_paid_generator(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = _paid_config(tmp_path)
    out = tmp_path / "out"

    from pah.cli.main import main

    code = main(
        [
            "run", "--spec", str(SPEC), "--policy", "rule", "--rounds", "2",
            "--out", str(out), "--config", str(config), "--allow-paid-gen",
        ]
    )

    assert code == 0
    report = json.loads((out / "report.json").read_text("utf-8"))
    assert report["run"]["paid_generation"] is True
    assert report["run"]["paid_provider"] == "paid_stub:python"
    assert report["run"]["paid_billing"] == "per_call"


def test_decompose_refuses_xpart_without_the_flag(tmp_path, monkeypatch, capsys):
    """X-Part is a paid decomposer, not a generator — same gate, same refusal. The
    old behaviour printed a cost warning and went ahead."""
    from pah.cli.main import main

    out = tmp_path / "decomposed"
    code = main(
        ["decompose", str(tmp_path / "object.stl"), "--decomposer", "xpart", "--out", str(out)]
    )

    assert code == 3
    assert "--allow-paid-gen" in capsys.readouterr().out
    assert not out.exists()


def test_build_refuses_before_it_creates_a_run_directory(tmp_path, monkeypatch, capsys):
    """A refusal must leave no half-run behind: the gate runs before `--out` exists."""
    monkeypatch.chdir(tmp_path)
    config = _paid_config(tmp_path)
    out = tmp_path / "build"

    from pah.cli.main import main

    code = main(["build", "a peg", "--out", str(out), "--config", str(config), "--no-preview"])

    assert code == 3
    assert "--allow-paid-gen" in capsys.readouterr().out
    assert not out.exists()


def test_eval_refuses_before_the_first_cell(tmp_path, monkeypatch, capsys):
    """Every cell would be another paid generation, so the refusal comes first."""
    monkeypatch.chdir(tmp_path)
    config = _paid_config(tmp_path)
    out = tmp_path / "eval"

    from pah.cli.main import main

    code = main(
        [
            "eval", "--suite", str(SPEC), "--policies", "rule", "--rounds", "1",
            "--out", str(out), "--config", str(config),
        ]
    )

    assert code == 3
    assert "--allow-paid-gen" in capsys.readouterr().out
    assert not (out / "results.jsonl").exists()


def test_doctor_call_refuses_a_paid_generator_without_the_flag(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    config = _paid_config(tmp_path)

    from pah.cli.main import main

    code = main(
        ["providers", "doctor", "--role", "gen", "--call", "--config", str(config), "--json"]
    )
    payload = json.loads(capsys.readouterr().out)

    assert code == 1
    call = payload["sections"]["gen"]["call"]
    assert call["ok"] is False
    assert "--allow-paid-gen" in call["error"]


def test_doctor_still_inspects_a_paid_generator_for_free(tmp_path, monkeypatch, capsys):
    """Without `--call` nothing is spent, so the import check must still answer — a
    doctor that refuses to look cannot explain what a paid config would do."""
    monkeypatch.chdir(tmp_path)
    config = _paid_config(tmp_path)

    from pah.cli.main import main

    code = main(["providers", "doctor", "--role", "gen", "--config", str(config), "--json"])
    section = json.loads(capsys.readouterr().out)["sections"]["gen"]

    assert code == 0
    assert section["paid"] is True
    assert section["call"] is None
    assert any(row["check"] == "target importable" for row in section["checks"])


# --------------------------------------------------------------------------- #
# the import regression: no `.env` read, no work at import
# --------------------------------------------------------------------------- #
def _call_name(call: ast.Call) -> str:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return ""


def test_the_hunyuan_adapter_does_no_work_at_import_time():
    """The mechanism of the incident, pinned with `ast`.

    A grep for `load_env_file` would also match a call inside a function, which is
    legitimate; only a *module-level* call executes on import, so only that is banned.
    """
    tree = ast.parse(HUNYUAN.read_text("utf-8"))
    banned = {"load_env_file", "_bypass_proxy_for_service"}
    offenders: list[tuple[int, str]] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue  # a def's body runs when it is called, not when it is imported
        for inner in ast.walk(node):
            if isinstance(inner, ast.Call) and _call_name(inner) in banned:
                offenders.append((inner.lineno, _call_name(inner)))

    assert offenders == []
    # Not even imported: the adapter reads credentials from the process environment
    # that `pah` already filled, and has no business holding the loader.
    assert "load_env_file" not in HUNYUAN.read_text("utf-8")


def test_importing_the_hunyuan_adapter_loads_no_dotenv(monkeypatch):
    """The behaviour behind the AST check: import it with a recording loader."""
    from pah.app import env as env_module

    loads: list[tuple] = []
    monkeypatch.setattr(
        env_module, "load_env_file", lambda *a, **k: loads.append(a) or {}
    )

    spec = importlib.util.spec_from_file_location("pah_paid_gate_probe", HUNYUAN)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "pah_paid_gate_probe", module)
    spec.loader.exec_module(module)

    assert loads == []
    assert callable(module.materialize)
