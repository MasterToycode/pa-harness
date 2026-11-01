"""Tests must not see the developer's machine configuration.

`pah.toml` and `.env` are per-machine: they name which generator this checkout uses
and which provider it talks to. A test that picks them up stops being about the code —
it calls a paid API, or silently runs on a DiT where the test assumed the parametric
builder, and then fails for a reason that has nothing to do with the change under
review.

This fixture installs a minimal, isolated config for **every** test:

* `PAH_CONFIG` points at a temp file with the parametric generator and no LLM;
* `PAH_ENV_FILE` points at a file that does not exist, so real credentials never leak
  into a test's environment.

A test that wants its own configuration still wins, because `resolve_config_path`
prefers the explicit `--config`, and a test may set `PAH_CONFIG` / `PAH_ENV_FILE`
itself (monkeypatch.setenv applies after this fixture's defaults).
"""

from __future__ import annotations

from pathlib import Path

import pytest

ISOLATED_CONFIG = """\
[llm]
default = "none"

[gen]
default = "local"

[gen.providers.local]
kind = "local-parametric"
"""


@pytest.fixture(autouse=True)
def isolated_machine_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    config = tmp_path / "pah.isolated.toml"
    config.write_text(ISOLATED_CONFIG, encoding="utf-8")
    monkeypatch.setenv("PAH_CONFIG", str(config))
    monkeypatch.setenv("PAH_ENV_FILE", str(tmp_path / "no-such.env"))
    return config
