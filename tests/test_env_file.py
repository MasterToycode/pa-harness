"""The local `.env`: paste a key once, not in every shell.

`pah` reads credentials from the environment only, which keeps keys out of the
repository but makes them awkward to use. `.env` is the compromise the tests here
pin down: gitignored, read at CLI start, and **an already-set environment variable
always wins**, so a one-off export still overrides the file and a shared checkout
can keep it empty.
"""

from __future__ import annotations

import json

from pah.app.env import ENV_FILE_VAR, load_env_file, parse_env_text


def test_parsing_the_subset_people_actually_type():
    text = """
    # a comment
    DEEPSEEK_API_KEY=sk-abc
    export QWEN_API_KEY="sk-qwen"
    SPACED = value with spaces
    broken-line
    """
    assert parse_env_text(text) == {
        "DEEPSEEK_API_KEY": "sk-abc",
        "QWEN_API_KEY": "sk-qwen",
        "SPACED": "value with spaces",
    }


def test_the_file_fills_gaps_and_never_overrides_a_shell_variable(tmp_path):
    path = tmp_path / ".env"
    path.write_text("DEEPSEEK_API_KEY=from-file\nQWEN_API_KEY=from-file\n", "utf-8")
    environ = {"DEEPSEEK_API_KEY": "from-shell"}

    added = load_env_file(path, environ=environ)

    assert added == {"QWEN_API_KEY": "from-file"}
    assert environ["DEEPSEEK_API_KEY"] == "from-shell"


def test_a_missing_file_is_normal(tmp_path):
    assert load_env_file(tmp_path / "absent", environ={}) == {}


def test_the_cli_loads_the_file_and_the_provider_check_sees_it(tmp_path, monkeypatch, capsys):
    """End to end: `.env` on disk -> credential check PASS, with no export at all."""
    from pah.cli.main import main

    variable = "PAH_TEST_ENV_FILE_KEY"
    monkeypatch.delenv(variable, raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(f"{variable}=from-dotenv\n", "utf-8")
    monkeypatch.setenv(ENV_FILE_VAR, str(env_file))

    config = tmp_path / "pah.toml"
    config.write_text(
        "\n".join(
            [
                "[llm]",
                'default = "fake"',
                "[llm.providers.fake]",
                'kind = "openai-compatible"',
                'base_url = "http://127.0.0.1:9/v1"',
                f'api_key_env = "{variable}"',
                'model = "test-model"',
                'billing = "free"',
            ]
        )
        + "\n",
        "utf-8",
    )

    code = main(["providers", "doctor", "--role", "llm", "--config", str(config), "--json"])
    payload = json.loads(capsys.readouterr().out)
    rows = {row["check"]: row["ok"] for row in payload["sections"]["llm"]["checks"]}

    assert code == 0, payload
    assert rows["credential"] is True


def test_the_loaded_names_are_reported_and_no_value_is_printed(tmp_path, monkeypatch, capsys):
    """"Did my key load?" must be answerable without reading source — and without
    the value ever reaching a log."""
    from pah.cli.main import main

    variable = "PAH_TEST_SILENT_KEY"
    monkeypatch.delenv(variable, raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(f"{variable}=very-secret-value\n", "utf-8")
    monkeypatch.setenv(ENV_FILE_VAR, str(env_file))

    main(["providers", "doctor", "--role", "gen", "--json"])
    captured = capsys.readouterr()

    assert variable in captured.err
    assert "very-secret-value" not in captured.out
    assert "very-secret-value" not in captured.err
