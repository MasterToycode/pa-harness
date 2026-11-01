"""Load a local `.env` so a key is pasted once instead of exported every shell.

Why this exists
---------------
`pah` reads credentials from the environment only — a key in a config file is a key
that ends up in a repository. That rule is right, but by itself it forces a key to be
re-exported in every new terminal, which is where keys leak into scrollback and chat
logs. A `.env` file fixes the ergonomics without weakening the rule: it is gitignored,
read once at CLI start, and its values go into `os.environ` — never printed, never
copied into a report or a trace.

Precedence
----------
An already-set environment variable **wins** over the file. So a one-off
`$env:DEEPSEEK_API_KEY=...` in a shell still overrides the file for that session, and
a shared machine can keep `.env` empty. The file only fills in what is missing.

Where it is read
----------------
`$PAH_ENV_FILE` if set, else `.env` in the current directory (the repo root when you
run `pah` from the checkout). A missing file is normal and not an error: the default
configuration needs no credentials at all.

Format: one `NAME=VALUE` per line, `#` comments, optional `export `, optional
surrounding quotes. That is the subset people actually type; a full dotenv grammar
would be a dependency for no benefit here.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["ENV_FILE_VAR", "load_env_file", "parse_env_text"]

#: Points `pah` at a `.env` outside the current directory (CI, a shared checkout).
ENV_FILE_VAR = "PAH_ENV_FILE"

#: Names that look like credentials — used only to say *which* names were filled,
#: never their values.
_SEPARATORS = "="


def _default_path() -> Path:
    override = os.environ.get(ENV_FILE_VAR, "")
    return Path(override) if override else Path(".env")


def parse_env_text(text: str) -> dict[str, str]:
    """The `NAME=VALUE` pairs in `text`, in file order, last one winning."""
    pairs: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        name, separator, value = line.partition(_SEPARATORS)
        if not separator:
            continue
        name = name.strip()
        if not name or not name.replace("_", "").isalnum():
            continue  # not a variable name; a typo must not become one
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        pairs[name] = value
    return pairs


def load_env_file(path: str | Path | None = None, *, environ: dict | None = None) -> dict[str, str]:
    """Fill `environ` (default `os.environ`) from the file; returns what was added.

    Returns the names it actually set. A caller can log them; nothing here ever
    returns or prints a value. Missing file, unreadable file and empty file are all
    "nothing to add" rather than errors, because the builtin configuration runs with
    no credentials at all.
    """
    target = Path(path) if path is not None else _default_path()
    environment = environ if environ is not None else os.environ
    try:
        text = target.read_text("utf-8")
    except (FileNotFoundError, NotADirectoryError, PermissionError, OSError):
        return {}
    added: dict[str, str] = {}
    for name, value in parse_env_text(text).items():
        if environment.get(name):  # an explicit shell variable wins
            continue
        environment[name] = value
        added[name] = value
    return added
