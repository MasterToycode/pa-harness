"""The builtin plugin chain: shipped modules that arrive as plugins, not as core edits.

`docs/contracts/v2-interfaces.md` §6 names three ways a module enters the harness, and
this file is the first one: **it is the explicit list of what the checkout ships**.
A builtin plugin is an ordinary plugin module (`api_version = "1"`, `kind`, and
`register(host)`), listed here so that `pah plugins list` can show it with
`source=builtin` — no entry point install, no user config.

Why the list is filtered instead of imported
--------------------------------------------
A module in `BUILTIN_MODULES` that is not installed is **skipped, not failed**. A
builtin means "what this build has", and a teammate's module that lands tomorrow is
not a defect today. The filter is `importlib.util.find_spec`, which answers "is the
file there" without executing the module — so listing the catalogue stays cheap and a
module that is present but raises still shows up as a failed row (the import happens
in `plugins.load`, and the failure is reported there).

Adding one is one line: put the module path in `BUILTIN_MODULES`. That is the whole
demonstration the contract asks for — a new capability (case memory, a DiT adapter, a
self-iteration loop) arrives without touching `core/` or any other frozen file.
"""

from __future__ import annotations

import importlib.util

__all__ = ["BUILTIN_MODULES", "builtin_modules"]

#: Modules this checkout ships as plugins, in listing order. Each one must declare
#: `api_version = "1"`, a `kind` from `pah.plugins.KINDS`, and `register(host)`.
BUILTIN_MODULES: tuple[str, ...] = (
    # T5's case memory: a policy plugin that remembers which repairs worked.
    "pah.policy.case_memory",
)


def builtin_modules() -> tuple[str, ...]:
    """The subset of `BUILTIN_MODULES` this installation actually has."""
    found: list[str] = []
    for module_name in BUILTIN_MODULES:
        try:
            if importlib.util.find_spec(module_name) is not None:
                found.append(module_name)
        except (ImportError, ModuleNotFoundError, ValueError):
            # A missing parent package is the same fact as a missing module: absent.
            continue
    return tuple(found)
