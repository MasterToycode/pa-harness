"""The plugin surface: one module in, one kind registered — the harness's growth port.

Why this exists
---------------
The harness is meant to keep growing (a DiT adapter, case memory, a
self-iteration mechanism), and `docs/contracts/v2-interfaces.md` §6 draws the
consequence: a new capability must arrive as a **plugin**, never as an edit to
`core/` or to any other frozen module. That only works if the harness has one
place that answers four questions at runtime:

1. *what is loaded* — a catalogue with kinds, so `pah plugins list` can print it;
2. *where it came from* — builtin, `pah.toml`, or an entry point;
3. *what it declares* — the certificate codes, operator ids, kinds, policy names
   it actually added (read off the registries, not trusted from a docstring);
4. *what broke* — a module that fails to import, declares the wrong
   `api_version`, or raises inside `register()` is recorded as a failed row, not
   as a traceback that takes the CLI down with it.

The three sources, in the order they win
----------------------------------------
* **builtin** — `BUILTIN_POLICIES` plus the closed sets the frozen contracts
  already fix (`DESIGN_CODES`, `DESIGN_OPERATOR_IDS`, the connector kinds in
  `design/spec.py`). Listing them is how `pah plugins list` answers "what does
  this build ship?" without importing a single optional module.
* **toml** — `[plugins] modules = ["mypkg.mod", ...]` in `pah.toml`.
* **entry point** — `importlib.metadata` group `pah.plugins`, so an installed
  package is discovered without editing anyone's config.

A plugin module declares exactly two module-level names and one callable::

    api_version = "1"          # required; the contract version it was written against
    kind = "policy"            # one of KINDS
    def register(host): ...    # adds to host.connectors / verifiers / operators /
                               # policies / providers

`api_version` is checked before `register` runs: a plugin written against a
future contract is refused by name, which is the difference between "your module
is incompatible" and a TypeError three frames deep in someone's registration.

Why the failure is a row and not an exception
---------------------------------------------
`pah plugins list` is a diagnostic: an operator whose plugin is broken needs the
list of what loaded *and* what did not. `load()` therefore never raises for a
bad plugin — it returns a record with `error` set. Only a malformed `pah.toml`
raises `PluginError`, because that is a broken request rather than a broken
implementation, and the run should stop before it starts. A caller that needs a
guarantee (the runtime does, before it runs an episode) checks
`report.failed` and decides; that decision lives in `app/runtime.py`, not here.

What a plugin may not do
------------------------
Declare itself under a kind outside `KINDS`, claim a different `api_version`,
or register something under a name it did not declare as its kind. The last rule
is what stops a "policy" plugin from quietly mounting an operator nobody audited;
the record carries both what was declared and what landed.
"""

from __future__ import annotations

import importlib
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps this module import-light
    from pah.design.api import DesignVerifierRegistry
    from pah.policy.registry import OperatorRegistry

__all__ = [
    "API_VERSION",
    "BUILTIN_POLICIES",
    "CONFIG_ENV",
    "ENTRY_POINT_GROUP",
    "KINDS",
    "PluginError",
    "PluginHost",
    "PluginLoadReport",
    "PluginRecord",
    "builtin_records",
    "entry_point_plugins",
    "load",
    "read_config_modules",
    "resolve_config_path",
    "snapshot",
]

#: The contract version this build serves. `docs/contracts/v2-interfaces.md` §6.
API_VERSION = "1"

#: The closed set of plugin kinds. A module that declares anything else is a row
#: with an error, never a new kind: a typo would otherwise become a schema change.
KINDS = ("connector", "verifier", "operator", "policy", "provider")

#: `importlib.metadata` group an installed package uses to announce itself.
ENTRY_POINT_GROUP = "pah.plugins"

#: Environment variable that points at a `pah.toml` when `--config` is absent.
CONFIG_ENV = "PAH_CONFIG"

#: Policy names this build resolves without any plugin. `llm` is listed because
#: the CLI accepts it; it still refuses to run without a configured provider,
#: which is a configuration fact rather than a missing module.
BUILTIN_POLICIES = ("rule", "random", "llm")


class PluginError(RuntimeError):
    """A plugin request that cannot be honoured: a broken `pah.toml`, no module."""


# --------------------------------------------------------------------------- #
# The registries a plugin is allowed to touch
# --------------------------------------------------------------------------- #
@dataclass
class PluginHost:
    """The five registries a plugin's `register(host)` may add to.

    Built by the composition root (`app/runtime.py`) from the real modules, so a
    plugin registers into the same objects an episode will use. `pah plugins
    list` builds an empty host instead: discovering what a plugin *would*
    register and registering it into live state are the same call, which is what
    keeps the listing honest.
    """

    connectors: dict[str, object] = field(default_factory=dict)
    verifiers: DesignVerifierRegistry = field(default_factory=lambda: _verifier_registry())
    operators: OperatorRegistry = field(default_factory=lambda: _operator_registry())
    #: name -> factory the runtime calls as `factory(**kwargs)`; the contract for
    #: a policy plugin is "produces a `policy.loop.LoopPolicy`".
    policies: dict[str, Callable[..., object]] = field(default_factory=dict)
    #: provider id -> `app.providers.ProviderConfig` (or anything the runtime can
    #: hand to `providers.build_client`).
    providers: dict[str, object] = field(default_factory=dict)


def _verifier_registry() -> DesignVerifierRegistry:
    from pah.design.api import DesignVerifierRegistry  # noqa: PLC0415 (lazy: trimesh on import)

    return DesignVerifierRegistry()


def _operator_registry() -> OperatorRegistry:
    from pah.policy.registry import OperatorRegistry  # noqa: PLC0415 (lazy: scipy on import)

    return OperatorRegistry()


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PluginRecord:
    """One row of `pah plugins list`: what, from where, declaring what, and why not.

    `name` is the module path for a loaded module and the shipped identifier for a
    builtin row, because that is what a reader needs in order to grep for it.

    `optional` marks a **builtin** module whose failure is a warning rather than a
    refusal. The distinction is about who promised what: `pah.toml` and an entry point
    are a caller's declaration, so a broken one must stop a run before it starts; the
    builtin chain is the checkout's own list, and a teammate's module landing over
    several minutes must not take `pah run` down in the meantime.
    """

    name: str
    kind: str
    source: str  # "builtin" | "toml" | "entrypoint"
    api_version: str = API_VERSION
    module: str = ""
    declarations: tuple[str, ...] = ()
    error: str = ""
    optional: bool = False

    @property
    def ok(self) -> bool:
        return not self.error

    @property
    def blocking(self) -> bool:
        """A failure that must stop an episode: declared by the caller, not shipped."""
        return bool(self.error) and not self.optional


@dataclass(frozen=True)
class PluginLoadReport:
    """Every row, in a stable order: builtins first, then toml, then entry points."""

    records: tuple[PluginRecord, ...] = ()

    @property
    def failed(self) -> tuple[PluginRecord, ...]:
        return tuple(record for record in self.records if not record.ok)

    @property
    def blocking(self) -> tuple[PluginRecord, ...]:
        """Failures a run must refuse for: a declared plugin that did not load."""
        return tuple(record for record in self.records if record.blocking)

    @property
    def warnings(self) -> tuple[PluginRecord, ...]:
        """Optional (builtin) failures, reported but not fatal."""
        return tuple(record for record in self.records if not record.ok and record.optional)

    @property
    def loaded(self) -> tuple[PluginRecord, ...]:
        return tuple(record for record in self.records if record.ok)

    def by_kind(self) -> dict[str, list[PluginRecord]]:
        grouped: dict[str, list[PluginRecord]] = {}
        for record in self.records:
            grouped.setdefault(record.kind, []).append(record)
        return grouped


# --------------------------------------------------------------------------- #
# What ships in the box
# --------------------------------------------------------------------------- #
def builtin_records() -> tuple[PluginRecord, ...]:
    """The capability catalogue the frozen contracts already fix.

    Read from `design/api.py` and `design/spec.py` rather than from the modules
    that implement them, so the listing works even while those modules are still
    being written — and cannot claim a code or an operator id that the contract
    does not assign.
    """
    from typing import get_args

    from pah.design.api import DESIGN_CODES, DESIGN_OPERATOR_IDS
    from pah.design.spec import ConnectorKindName

    records: list[PluginRecord] = []
    for kind_name in sorted(get_args(ConnectorKindName)):
        records.append(
            PluginRecord(
                name=kind_name,
                kind="connector",
                source="builtin",
                declarations=(f"kind={kind_name}",),
            )
        )
    for verifier_id, codes in DESIGN_CODES.items():
        records.append(
            PluginRecord(
                name=verifier_id,
                kind="verifier",
                source="builtin",
                declarations=tuple(f"code={code}" for code in codes),
            )
        )
    for operator_id in DESIGN_OPERATOR_IDS:
        records.append(
            PluginRecord(
                name=operator_id,
                kind="operator",
                source="builtin",
                declarations=(f"operator={operator_id}",),
            )
        )
    for policy_name in BUILTIN_POLICIES:
        records.append(
            PluginRecord(
                name=policy_name,
                kind="policy",
                source="builtin",
                declarations=(f"policy={policy_name}",),
            )
        )
    return tuple(records)


# --------------------------------------------------------------------------- #
# Where a plugin module is declared
# --------------------------------------------------------------------------- #
def resolve_config_path(explicit: str | Path | None = None) -> Path | None:
    """The `pah.toml` to read: `--config`, then `$PAH_CONFIG`, then `./pah.toml`.

    `None` means "no config", which is valid: the builtin plane needs none.
    """
    if explicit:
        return Path(explicit)
    from_env = os.environ.get(CONFIG_ENV, "")
    if from_env:
        return Path(from_env)
    local = Path("pah.toml")
    return local if local.exists() else None


def read_config_modules(config_path: str | Path | None = None) -> tuple[list[str], Path | None]:
    """The `[plugins] modules` list from `pah.toml`, plus the file it came from.

    A malformed file raises `PluginError` naming the file: a config the harness
    cannot parse is a broken request, and silently running with no plugins would
    make a run look reproducible when it is not.
    """
    path = resolve_config_path(config_path)
    if path is None:
        return [], None
    if not path.exists():
        raise PluginError(f"config file {path} does not exist")
    try:
        data = tomllib.loads(path.read_text("utf-8"))
    except tomllib.TOMLDecodeError as error:
        raise PluginError(f"{path} is not valid TOML: {error}") from error
    section = data.get("plugins", {})
    if not isinstance(section, dict):
        raise PluginError(f"{path}: [plugins] must be a table, got {type(section).__name__}")
    modules = section.get("modules", [])
    if not isinstance(modules, list) or not all(isinstance(item, str) and item for item in modules):
        raise PluginError(f"{path}: [plugins] modules must be a list of module paths")
    return list(modules), path


def entry_point_plugins() -> list[tuple[str, object]]:
    """`(label, loaded object)` for every entry point in `pah.plugins`.

    Import errors are reported by `load()`, not here: discovery must be able to
    list a broken entry point by name.
    """
    from importlib.metadata import entry_points  # noqa: PLC0415 (stdlib, kept local for symmetry)

    found: list[tuple[str, object]] = []
    for point in entry_points(group=ENTRY_POINT_GROUP):
        label = f"entrypoint:{point.name}"
        try:
            found.append((label, point.load()))
        except Exception as error:  # noqa: BLE001 - the row carries the failure
            found.append((label, error))
    return sorted(found, key=lambda item: item[0])


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def snapshot(host: PluginHost) -> dict[str, tuple[str, ...]]:
    """What the host holds right now, per kind — the before/after of one plugin.

    The operator registry has no public enumeration in v1, so its private dict is
    read here rather than reimplementing a catalogue that would then disagree with
    it. `ids()` is preferred the day the registry grows one.
    """
    verifier_ids = getattr(host.verifiers, "ids", lambda: ())()
    operators = getattr(host.operators, "ids", None)
    if callable(operators):
        operator_ids = operators()
    else:
        operator_ids = tuple(getattr(host.operators, "_operators", {}).keys())
    return {
        "connector": tuple(sorted(host.connectors)),
        "verifier": tuple(sorted(verifier_ids)),
        "operator": tuple(sorted(operator_ids)),
        "policy": tuple(sorted(host.policies)),
        "provider": tuple(sorted(host.providers)),
    }


def _declarations(host: PluginHost, before: Mapping[str, tuple[str, ...]]) -> tuple[str, ...]:
    """What a plugin actually added, named the way a report cites it."""
    after = snapshot(host)
    lines: list[str] = []
    for kind in KINDS:
        for name in after.get(kind, ()):
            if name in before.get(kind, ()):
                continue
            if kind == "verifier":
                try:
                    codes = tuple(host.verifiers.get(name).spec.codes)
                except Exception:  # a registry that cannot describe it still gets a row
                    codes = ()
                lines.extend(f"code={code}" for code in codes)
                lines.append(f"verifier={name}")
            elif kind == "connector":
                lines.append(f"kind={name}")
            elif kind == "operator":
                lines.append(f"operator={name}")
            elif kind == "policy":
                lines.append(f"policy={name}")
            else:
                lines.append(f"provider={name}")
    return tuple(sorted(set(lines)))


def _register_into(host: PluginHost, obj: object, *, label: str) -> PluginRecord:
    """Import/validate/reigster one plugin object; always returns a record."""
    module = getattr(obj, "__name__", "") or label
    if isinstance(obj, BaseException):
        return PluginRecord(
            name=label,
            kind="",
            source="entrypoint",
            module=module,
            error=f"could not load: {type(obj).__name__}: {obj}",
        )
    kind = str(getattr(obj, "kind", "") or "")
    api_version = str(getattr(obj, "api_version", "") or "")
    if kind not in KINDS:
        return PluginRecord(
            name=label,
            kind=kind,
            source="",
            module=module,
            api_version=api_version,
            error=f"declares kind {kind!r}; the contract allows {KINDS}",
        )
    if api_version != API_VERSION:
        return PluginRecord(
            name=label,
            kind=kind,
            source="",
            module=module,
            api_version=api_version,
            error=(
                f"declares api_version {api_version!r}; this harness serves "
                f"{API_VERSION!r} (upgrade the plugin, not the harness)"
            ),
        )
    register = getattr(obj, "register", None)
    if not callable(register):
        return PluginRecord(
            name=label,
            kind=kind,
            source="",
            module=module,
            api_version=api_version,
            error="has no register(host) callable",
        )
    before = snapshot(host)
    try:
        register(host)
    except Exception as error:  # noqa: BLE001 - a broken plugin is a row, not a crash
        return PluginRecord(
            name=label,
            kind=kind,
            source="",
            module=module,
            api_version=api_version,
            error=f"register(host) raised {type(error).__name__}: {error}",
        )
    declarations = _declarations(host, before)
    added_kinds = {
        "connector" if line.startswith("kind=") else line.split("=", 1)[0]
        for line in declarations
        if line.startswith(("kind=", "verifier=", "operator=", "policy=", "provider="))
    }
    if not declarations:
        return PluginRecord(
            name=label,
            kind=kind,
            source="",
            module=module,
            api_version=api_version,
            error="register(host) added nothing to the host",
        )
    if kind not in added_kinds:
        return PluginRecord(
            name=label,
            kind=kind,
            source="",
            module=module,
            api_version=api_version,
            declarations=declarations,
            error=f"declares kind {kind!r} but registered {sorted(added_kinds)!r}",
        )
    return PluginRecord(
        name=label,
        kind=kind,
        source="",
        module=module,
        api_version=api_version,
        declarations=declarations,
    )


def load(
    host: PluginHost,
    *,
    config_path: str | Path | None = None,
    include_entry_points: bool = True,
) -> PluginLoadReport:
    """Load every declared plugin into `host`; never raises for a bad plugin.

    Order is fixed — contract rows, then the builtin module chain, then `pah.toml`
    modules in declared order, then entry points by name — so two runs of
    `pah plugins list` print the same table.
    """
    records: list[PluginRecord] = list(builtin_records())

    # Source 1, part two: the shipped plugin modules (see `plugins_builtin.py`).
    # A module that is present but broken is a row; for the builtin chain it is a
    # warning rather than a refusal, so a teammate's in-flight module cannot take
    # `pah run` down while the rest of the tree is green.
    for module_path in _builtin_modules():
        records.append(_load_module(module_path, "builtin", host, optional=True))

    modules, _ = read_config_modules(config_path)
    for module_path in modules:
        records.append(_load_module(module_path, "toml", host))

    if include_entry_points:
        for label, obj in entry_point_plugins():
            record = _register_into(host, obj, label=label)
            records.append(
                PluginRecord(
                    name=record.name,
                    kind=record.kind,
                    source="entrypoint",
                    api_version=record.api_version,
                    module=record.module,
                    declarations=record.declarations,
                    error=record.error,
                )
            )

    return PluginLoadReport(records=tuple(records))


def _builtin_modules() -> tuple[str, ...]:
    """The shipped plugin modules, or none when `plugins_builtin.py` is absent."""
    try:
        from pah import plugins_builtin  # noqa: PLC0415
    except Exception:  # noqa: BLE001 - the chain is optional; the contract rows are not
        return ()
    discover = getattr(plugins_builtin, "builtin_modules", None)
    if not callable(discover):
        return ()
    try:
        return tuple(discover())
    except Exception:  # noqa: BLE001 - a broken chain must not take the catalogue down
        return ()


def _load_module(
    module_path: str, source: str, host: PluginHost, *, optional: bool = False
) -> PluginRecord:
    """Import `module_path` and register it, recording the source that named it."""
    try:
        module = importlib.import_module(module_path)
    except Exception as error:  # noqa: BLE001 - the row carries the failure
        return PluginRecord(
            name=module_path,
            kind="",
            source=source,
            module=module_path,
            error=f"import failed: {type(error).__name__}: {error}",
            optional=optional,
        )
    record = _register_into(host, module, label=module_path)
    return PluginRecord(
        name=record.name,
        kind=record.kind,
        source=source,
        api_version=record.api_version,
        module=record.module,
        declarations=record.declarations,
        error=record.error,
        optional=optional,
    )
