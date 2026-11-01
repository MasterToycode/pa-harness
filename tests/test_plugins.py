"""The plugin surface: three load sources, one catalogue, and no silent failures.

What is pinned here
-------------------
* the builtin catalogue matches the frozen contract sets, with **no duplicate rows** —
  the contract says registering the same id twice is an error, and the listing must
  not paper over one;
* a `pah.toml` module and an entry point register into the same host and are reported
  with their own source;
* every way a plugin can be wrong (`api_version`, no `register`, a raising `register`,
  registering nothing, declaring one kind and landing another, a duplicate id) becomes
  a **failed row with a sentence**, never a traceback that takes `pah plugins list`
  down — that command is the diagnostic, so it has to survive the thing it diagnoses;
* the CLI exit codes: `0` all loaded, `1` something failed, `3` the config itself is
  broken.
"""

from __future__ import annotations

import importlib
import json
import textwrap
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from pah import plugins
from pah.cli.main import main
from pah.design.api import DESIGN_CODES, DESIGN_OPERATOR_IDS

REPO = Path(__file__).resolve().parents[1]

GOOD_POLICY_PLUGIN = """
api_version = "1"
kind = "policy"

def register(host):
    host.policies["remembered"] = lambda **kwargs: object()
"""

DUPLICATE_OPERATOR_PLUGIN = """
api_version = "1"
kind = "operator"

from pydantic import BaseModel

from pah.policy.registry import Operator


class Params(BaseModel):
    place: str = "x"


def register(host):
    operator = Operator(
        id="dup_operator",
        params=Params,
        preconditions=(),
        apply=lambda assembly, params: assembly,
        targets=lambda params: (),
        cost="cheap",
        family="connector",
    )
    host.operators.register(operator)
    host.operators.register(operator)
"""


def write_module(directory: Path, name: str, source: str) -> str:
    """Writes `source` under a unique module name and returns it.

    The name has to be unique per test: `importlib` caches modules by name, so two
    parametrised cases writing `bad_plugin.py` into different temp directories would
    still import the first file's module the second time — a test that silently
    measures its own cache.
    """
    unique = f"{name}_{uuid.uuid4().hex[:8]}"
    (directory / f"{unique}.py").write_text(textwrap.dedent(source), "utf-8")
    return unique


def write_config(directory: Path, modules: list[str]) -> Path:
    listed = ", ".join(f'"{module}"' for module in modules)
    config = directory / "pah.toml"
    config.write_text(f"[plugins]\nmodules = [{listed}]\n", "utf-8")
    return config


def fresh_host(monkeypatch, directory: Path) -> plugins.PluginHost:
    """A host plus a sys.path that can import modules written into `directory`."""
    monkeypatch.syspath_prepend(str(directory))
    importlib.invalidate_caches()
    return plugins.PluginHost()


# --------------------------------------------------------------------------- #
# The builtin catalogue
# --------------------------------------------------------------------------- #
def test_builtin_rows_match_the_frozen_sets_and_do_not_duplicate():
    records = plugins.builtin_records()
    keys = [(record.kind, record.name) for record in records]
    assert len(keys) == len(set(keys)), "the catalogue must not list an id twice"

    assert {record.name for record in records if record.kind == "operator"} == set(
        DESIGN_OPERATOR_IDS
    )
    assert {record.name for record in records if record.kind == "verifier"} == set(
        DESIGN_CODES
    )
    assert {record.name for record in records if record.kind == "connector"} == {
        "peg_hole",
        "snap_fit",
        "dovetail",
    }
    assert {record.name for record in records if record.kind == "policy"} == set(
        plugins.BUILTIN_POLICIES
    )
    # 3 connectors + 5 verifiers + 7 operators + 3 policies; a plugin module the
    # checkout ships adds one more row, none of which are contract constants.
    assert len(records) == 18
    assert all(record.source == "builtin" for record in records)
    assert all(record.ok for record in records)


def test_the_builtin_module_chain_is_discoverable_without_executing_it():
    from pah import plugins_builtin

    assert "pah.policy.case_memory" in plugins_builtin.BUILTIN_MODULES
    # `builtin_modules()` is allowed to be empty: a module that has not landed is not
    # a broken plugin, and the chain must not turn `pah plugins list` red for it.
    assert isinstance(plugins_builtin.builtin_modules(), tuple)


# --------------------------------------------------------------------------- #
# Loading from pah.toml
# --------------------------------------------------------------------------- #
def test_a_toml_module_registers_and_reports_its_source(monkeypatch, tmp_path):
    module = write_module(tmp_path, "policy_plugin", GOOD_POLICY_PLUGIN)
    host = fresh_host(monkeypatch, tmp_path)
    config = write_config(tmp_path, [module])

    report = plugins.load(host, config_path=config, include_entry_points=False)

    record = next(row for row in report.records if row.name == module)
    assert record.ok, record.error
    assert record.source == "toml"
    assert record.kind == "policy"
    assert record.api_version == "1"
    assert record.declarations == ("policy=remembered",)
    assert "remembered" in host.policies
    assert not report.failed


def test_a_missing_config_file_is_a_clear_refusal(tmp_path):
    with pytest.raises(plugins.PluginError, match="does not exist"):
        plugins.read_config_modules(tmp_path / "nope.toml")


def test_a_config_that_is_not_toml_is_a_clear_refusal(tmp_path):
    config = tmp_path / "pah.toml"
    config.write_text("[plugins\nmodules = [", "utf-8")
    with pytest.raises(plugins.PluginError, match="not valid TOML"):
        plugins.read_config_modules(config)


def test_modules_must_be_a_list_of_strings(tmp_path):
    config = tmp_path / "pah.toml"
    config.write_text('[plugins]\nmodules = "one_module"\n', "utf-8")
    with pytest.raises(plugins.PluginError, match="list of module paths"):
        plugins.read_config_modules(config)


# --------------------------------------------------------------------------- #
# Every way a plugin can be wrong is a row
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "source,expected",
    [
        ('api_version = "2"\nkind = "policy"\n', "api_version"),
        ('api_version = "1"\nkind = "weld"\n', "declares kind"),
        ('api_version = "1"\nkind = "policy"\n', "no register"),
        (
            'api_version = "1"\nkind = "policy"\ndef register(host):\n    raise RuntimeError("boom")\n',
            "register(host) raised RuntimeError: boom",
        ),
        (
            'api_version = "1"\nkind = "policy"\ndef register(host):\n    pass\n',
            "added nothing",
        ),
        (
            'api_version = "1"\nkind = "verifier"\ndef register(host):\n'
            '    host.policies["sneaky"] = lambda **kwargs: object()\n',
            "but registered",
        ),
    ],
)
def test_a_wrong_plugin_is_a_failed_row_not_a_crash(monkeypatch, tmp_path, source, expected):
    module = write_module(tmp_path, "bad_plugin", source)
    host = fresh_host(monkeypatch, tmp_path)
    config = write_config(tmp_path, [module])

    report = plugins.load(host, config_path=config, include_entry_points=False)

    record = next(row for row in report.records if row.name == module)
    assert not record.ok
    assert expected in record.error
    assert report.failed == (record,)


def test_a_duplicate_registration_is_refused_not_overwritten(monkeypatch, tmp_path):
    """The contract says a repeated id is an error; a registry that quietly replaced an
    action would let two traces with the same policy disagree."""
    module = write_module(tmp_path, "duplicate_plugin", DUPLICATE_OPERATOR_PLUGIN)
    host = fresh_host(monkeypatch, tmp_path)
    config = write_config(tmp_path, [module])

    report = plugins.load(host, config_path=config, include_entry_points=False)

    record = next(row for row in report.records if row.name == module)
    assert not record.ok
    assert "already registered" in record.error


def test_an_import_error_names_the_module_and_the_cause(monkeypatch, tmp_path):
    host = fresh_host(monkeypatch, tmp_path)
    config = write_config(tmp_path, ["a_module_that_does_not_exist"])

    report = plugins.load(host, config_path=config, include_entry_points=False)

    record = next(row for row in report.records if row.name == "a_module_that_does_not_exist")
    assert not record.ok
    assert "import failed" in record.error
    assert "ModuleNotFoundError" in record.error


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #
def test_an_entry_point_plugin_is_loaded_and_labelled(monkeypatch, tmp_path):
    module = SimpleNamespace(
        __name__="installed_plugin",
        api_version="1",
        kind="policy",
        register=lambda host: host.policies.__setitem__("from_entry", lambda **kwargs: object()),
    )
    monkeypatch.setattr(plugins, "entry_point_plugins", lambda: [("entrypoint:installed", module)])
    host = plugins.PluginHost()

    report = plugins.load(host, include_entry_points=True)

    record = next(row for row in report.records if row.name == "entrypoint:installed")
    assert record.ok, record.error
    assert record.source == "entrypoint"
    assert record.declarations == ("policy=from_entry",)
    assert "from_entry" in host.policies


# --------------------------------------------------------------------------- #
# pah plugins list
# --------------------------------------------------------------------------- #
def test_plugins_list_prints_every_kind(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)  # no ambient pah.toml: the listing is the builtins
    rc = main(["plugins", "list", "--no-entry-points"])
    output = capsys.readouterr().out

    assert rc == 0
    # Every kind that has a row is named; `provider` has no row here because no
    # provider is configured, and the listing is a catalogue of what exists.
    for kind in ("connector", "verifier", "operator", "policy"):
        assert kind in output
    assert "plugin row(s)" in output


def test_plugins_list_json_names_the_whole_contract(monkeypatch, tmp_path, capsys):
    from pah import plugins_builtin

    monkeypatch.chdir(tmp_path)
    assert main(["plugins", "list", "--no-entry-points", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["api_version"] == plugins.API_VERSION
    assert payload["kinds"] == list(plugins.KINDS)
    assert payload["failed"] == 0
    # The eighteen contract rows plus whatever builtin plugin modules this checkout
    # ships — the chain is read from `plugins_builtin`, so a module landing tomorrow
    # does not break this test.
    assert len(payload["plugins"]) == 18 + len(plugins_builtin.builtin_modules())


def test_plugins_list_exits_1_when_a_declared_plugin_fails(monkeypatch, tmp_path, capsys):
    host_config = write_config(tmp_path, ["no_such_plugin_module"])
    rc = main(["plugins", "list", "--no-entry-points", "--config", str(host_config)])
    output = capsys.readouterr().out

    assert rc == 1
    assert "FAIL" in output
    assert "import failed" in output


def test_plugins_list_exits_3_when_the_config_is_broken(tmp_path, capsys):
    config = tmp_path / "pah.toml"
    config.write_text("[plugins\n", "utf-8")
    rc = main(["plugins", "list", "--no-entry-points", "--config", str(config)])
    assert rc == 3
    assert "not valid TOML" in capsys.readouterr().out
