"""The composition root: one place that turns the modules into a runnable episode.

Why a composition root at all
-----------------------------
Every other v2 module is written against a protocol and knows nothing about its
neighbours: the operators edit a `DesignSpec`, the cascade materialises and judges
it, the loop drives verify → decide → apply. Something has to hold them and say
"this is the process". That is this module, and it is deliberately the only v2 file
allowed to import across all four planes — the CLI, the eval runner and the tests
all come through here instead of wiring themselves.

Three rules decide what is in it
--------------------------------
1. **No domain rules.** `Runtime` does not decide what to repair, whether a design
   is good, or what an operator means. It opens a store, builds a ticket, calls the
   loop, and writes down what happened. Everything a reader might call a rule lives
   in `core/`, `design/api.py`, the operators or the verifiers.
2. **Missing neighbours fail as one sentence, not a traceback.** T1/T2/T3 are
   written in parallel with this file, so every cross-plane import is lazy and
   wrapped: the run stops with `RuntimeError("the design-plane operator table (T3)
   has not landed yet: module 'pah.policy.design_ops' is absent ...")` instead of an
   `ImportError` from three frames down. `cli/run.py` prints it and exits 3.
3. **The runtime owns the episode, not the geometry.** Operators are pure
   (`docs/contracts/v2-interfaces.md` §2) and `geom.hash` is
   `design/api.geometry_hash(spec, part_id)` — spec-only by contract. So this
   module never hashes an STL and never builds one: it writes the spec each round
   produced, hands the pure `Assembly` to the loop, and lets T2's cascade
   materialise when it needs triangles. That is what makes a resumed episode
   reproduce the same `replay_hashes` even after the run directory moved.

What an episode leaves on disk
------------------------------
    <out>/spec.json              the design intent the episode ended with
    <out>/spec-initial.json      the spec it started from (fresh runs)
    <out>/trace.jsonl            append-only events, `seq` continuing across resume
    <out>/db.sqlite              tickets and rounds; the resume anchor
    <out>/geometry/*.stl         written by the verifier cascade's materialiser
    <out>/round-<n>/spec.json    what round n produced — the state a resume rebuilds
    <out>/report.json            one JSON document: outcome, budget, evidence, trace

`round-<n>` is keyed by the engine's own round index, so a hole in the numbering is
a round whose operator never produced a design, and the report says which.
"""

from __future__ import annotations

import importlib
import inspect
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from pah.app.store import Store
from pah.app.trace import Trace
from pah.core.engine import Engine
from pah.core.ids import key
from pah.core.model import (
    Assembly,
    Budget,
    Cost,
    FailureCertificate,
    Mandate,
    RepairPlan,
    Round,
    Ticket,
)
from pah.design.api import (
    SPEC_SLOT,
    assembly_from_spec,
    geometry_hash,
    hard_codes,
    spec_of,
)
from pah.design.spec import DesignSpec
from pah.policy.loop import (
    EpisodeResult,
    RepairLoop,
    assembly_fingerprint,
    hard_failures,
)

__all__ = [
    "FAULTS",
    "FaultyCascade",
    "Runtime",
    "RuntimeConfig",
    "SpecBridge",
    "budget_for",
    "build",
    "build_llm_client",
    "require_provider",
    "resolve_cascade",
    "resolve_design_registry",
    "resolve_materializer",
    "resolve_operators",
    "resolve_policy",
]

#: What `--inject-fault` accepts. `verifier` fails the *second* verification call,
#: which is the first post-repair one: the loop records that round as `error`,
#: keeps going, and the episode still reaches a stop condition. Failing the first
#: call instead would end the episode before any round exists, which proves
#: nothing about the loop's recovery.
FAULTS = ("verifier",)

#: The `kv` key that records which ticket a run directory belongs to. `build` has to
#: create the SQLite file before an episode starts, so "db.sqlite exists" cannot
#: mean "this directory already holds a run" — this marker can.
RUN_MARKER = "pah.run.ticket"

#: A known limitation of resuming, recorded in every resumed report rather than left
#: for a reader to discover from a budget table. `policy/loop.py:_charge` reads the
#: cascade's meter, which a new process starts at zero; verification done between the
#: last recorded round and the first resumed round is therefore charged to the resumed
#: round. Geometry hashes are unaffected, so replay and case outcomes still compare.
RESUME_CHARGE_NOTE = (
    "resume: the cascade meter restarts at 0, so verification costs between the last "
    "recorded round and the first resumed round are charged to the resumed round; "
    "geometry hashes and verdicts are unaffected"
)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RuntimeConfig:
    """Everything `pah run` decides before it composes anything."""

    out: Path
    #: Where to read `[plugins] modules` from. `None` means `../pah.toml` or
    #: `$PAH_CONFIG`, resolved by `pah.plugins.resolve_config_path`.
    config_path: Path | None = None
    seed: int = 0
    geom_calls: int | None = None
    sim_calls: int | None = None
    inject_fault: str | None = None
    stop_after_round: int | None = None
    #: Kept for the report only; the episode itself is driven by `run_episode`.
    policy: str = "rule"
    rounds: int = 3
    resume: bool = False


def budget_for(rounds: int, *, geom_calls: int | None = None, sim_calls: int | None = None) -> Budget:
    """The mandate a run gets for `rounds` rounds.

    Only `rounds` is a user-facing number. The two call dimensions get headroom
    because they are *declared* by the policy before the round runs and then
    *charged* with the verifier cascade's real spend (`policy/loop.py:_charge`): a
    budget one call short would refuse a legal round and the refusal would read as a
    policy failure instead of an accounting artifact. A caller that wants a tighter
    ceiling passes one explicitly.
    """
    if rounds < 1:
        raise ValueError(f"rounds must be >= 1, got {rounds}")
    return Budget(
        rounds=rounds,
        geom_calls=geom_calls if geom_calls is not None else max(16, 8 * rounds),
        sim_calls=sim_calls if sim_calls is not None else max(8, 4 * rounds),
    )


# --------------------------------------------------------------------------- #
# Cross-plane resolution: lazy, and one sentence when a neighbour is absent
# --------------------------------------------------------------------------- #
def _require(module_name: str, what: str) -> Any:
    """Import `module_name` for `what`, or raise a `RuntimeError` a human can act on.

    A missing module and a missing third-party dependency read completely
    differently to whoever sees the message, so they are told apart: the first means
    "a teammate has not landed it yet", the second means "this install is broken".
    """
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        absent = error.name or ""
        if absent == module_name or module_name.startswith(absent + "."):
            raise RuntimeError(
                f"{what} has not landed yet: module {module_name!r} is absent. "
                "The harness refuses to run a stubbed episode; see "
                "docs/contracts/v2-interfaces.md §8 for who is writing it."
            ) from error
        raise RuntimeError(
            f"{what} ({module_name!r}) cannot be imported because its dependency "
            f"{absent!r} is missing from this environment"
        ) from error
    except Exception as error:  # a module that exists but blows up on import
        raise RuntimeError(
            f"{what} ({module_name!r}) failed to import: {type(error).__name__}: {error}"
        ) from error


def resolve_materializer() -> Any:
    """T1: `pah.design.materialize.materialize(spec, out_dir) -> MaterializedDesign`.

    The runtime does not call it — T2's cascade does, when the assembly it is asked
    to judge has no `geom.path` yet. It is resolved here anyway, as a preflight, so
    that a missing T1 is reported as "the geometry materialiser has not landed yet"
    rather than as a constructor failure inside the cascade.
    """
    module = _require("pah.design.materialize", "the geometry materialiser (T1)")
    materialize = getattr(module, "materialize", None)
    if not callable(materialize):
        raise RuntimeError(
            "the geometry materialiser (T1) does not expose "
            "`materialize(spec, out_dir)` in pah.design.materialize"
        )
    return materialize


def resolve_design_registry() -> Any:
    """T2: the design-plane verifier catalogue, not yet gate-checked.

    The canonical entry point is `design_cascade.default_registry()`; the second
    spelling is accepted because the module was written in parallel and a naming
    drift there must not cost an integration day on this side.
    """
    module = _require(
        "pah.tools.verifiers.design_cascade", "the design-plane verifier cascade (T2)"
    )
    for name in ("default_registry", "build_registry"):
        factory = getattr(module, name, None)
        if callable(factory):
            return factory()
    raise RuntimeError(
        "the design-plane verifier cascade (T2) exposes neither "
        "`default_registry()` nor `build_registry()` in pah.tools.verifiers.design_cascade"
    )


def resolve_cascade(registry: Any, root: str | Path) -> Any:
    """T2: `DesignVerifierCascade(registry, root)`, the loop's `LoopVerifiers`.

    `root` is where the cascade materialises `root/geometry/<part_id>.stl` when it
    is handed a still-pure assembly; the runtime passes the run directory so those
    files are the run's `geometry/` deliverable. The cascade also carries
    `spent: Cost`, which `policy/loop.py` charges to each round — an un-metered
    cascade would make the physics tier look free and unfair in an equal-budget
    comparison.
    """
    module = _require(
        "pah.tools.verifiers.design_cascade", "the design-plane verifier cascade (T2)"
    )
    cls = getattr(module, "DesignVerifierCascade", None)
    if cls is not None:
        return cls(registry, Path(root))
    for name in ("build", "cascade"):
        factory = getattr(module, name, None)
        if callable(factory):
            return factory(registry, Path(root))
    raise RuntimeError(
        "the design-plane verifier cascade (T2) exposes no "
        "`DesignVerifierCascade(registry, root)` in pah.tools.verifiers.design_cascade"
    )


def resolve_operators() -> Any:
    """T3: an `OperatorRegistry` holding the seven design-plane operators."""
    from pah.policy.registry import OperatorRegistry  # noqa: PLC0415

    module = _require("pah.policy.design_ops", "the design-plane operator table (T3)")
    register = None
    for name in ("register_builtin", "register_operators", "register_all"):
        candidate = getattr(module, name, None)
        if callable(candidate):
            register = candidate
            break
    if register is None:
        raise RuntimeError(
            "the design-plane operator table (T3) exposes no `register_builtin(registry)` "
            "in pah.policy.design_ops"
        )
    registry = OperatorRegistry()
    register(registry)
    if not getattr(registry, "_operators", None):
        raise RuntimeError(
            "the design-plane operator table (T3) registered no operators; refusing "
            "to run an episode with an empty action space"
        )
    return registry


def _provider_config(config_path: str | Path | None) -> Any:
    """The configured `llm` provider, or raise with the reason it is absent.

    Two failures are refused here rather than at the first decision: no provider
    configured (the CLI has nothing to ask), and a provider whose key environment
    variable is unset (the call would fail after the episode had already paid for its
    first verification). The key itself is never read, printed or stored — only its
    variable name, which is what `app/providers.py` documents.
    """
    from pah.app.providers import ProviderError, load_registry  # noqa: PLC0415
    from pah.plugins import resolve_config_path  # noqa: PLC0415

    path = resolve_config_path(config_path)
    registry = load_registry(path if path is not None else "pah.toml")
    try:
        config = registry.choose("llm")
    except ProviderError as error:
        where = str(path) if path is not None else "no pah.toml found"
        raise RuntimeError(
            f"`--policy llm` needs a default llm provider in pah.toml ({where}): {error}"
        ) from error
    key_env = str(getattr(config, "api_key_env", "") or "")
    if key_env and not os.environ.get(key_env):
        raise RuntimeError(
            f"llm provider {config.id!r} needs ${key_env}, which is not set. The run "
            "would fail on its first decision, so it is refused here instead; set the "
            "variable or point --config at a provider that needs no key."
        )
    return config


def require_provider(config_path: str | Path | None = None) -> Any:
    """Preflight for `--policy llm`: fails with a sentence, before anything is built."""
    return _provider_config(config_path)


def build_llm_client(config_path: str | Path | None = None) -> Any:
    """The `LLMClient` a model-backed policy uses, built from `pah.toml`."""
    from pah.app.providers import build_client  # noqa: PLC0415

    return build_client(_provider_config(config_path))


def resolve_policy(
    name: str,
    *,
    operators: Any,
    config_path: str | Path | None = None,
    seed: int = 0,
    llm: Any = None,
) -> Any:
    """T3: a `policy.loop.LoopPolicy` for `name` (`rule`, `random`, `llm`).

    The builder is called with the parameter names it actually declares rather than
    a fixed signature: the design-plane policies are being written in parallel, and
    a policy that wants `(registry, seed)` and one that wants
    `(registry, config_path, llm)` are both legal. Unknown names are refused with
    the list the module publishes, because silently falling back to `rule` would
    make a comparison table lie about which policy produced a row.
    """
    module = _require("pah.policy.design_policies", "the design-plane policies (T3)")
    build = None
    for attr in ("build", "build_policy", "make_policy"):
        candidate = getattr(module, attr, None)
        if callable(candidate):
            build = candidate
            break
    if build is None:
        raise RuntimeError(
            "the design-plane policies (T3) expose no `build(name, ...)` in "
            "pah.policy.design_policies"
        )
    available = getattr(module, "POLICY_NAMES", None)
    if available is not None and name not in available:
        raise RuntimeError(
            f"unknown policy {name!r}; pah.policy.design_policies offers {tuple(available)}"
        )
    kwargs = _builder_kwargs(
        build, operators=operators, config_path=config_path, seed=seed, llm=llm
    )
    return build(name, **kwargs)


def _builder_kwargs(
    build: Callable[..., Any],
    *,
    operators: Any,
    config_path: str | Path | None,
    seed: int,
    llm: Any,
) -> dict[str, Any]:
    """Map this module's canonical names onto the builder's declared parameters."""
    try:
        declared = set(inspect.signature(build).parameters)
    except (TypeError, ValueError):  # a builtin or a C callable: try the canonical set
        declared = {"registry", "config_path", "seed", "llm"}
    kwargs: dict[str, Any] = {}
    if "registry" in declared:
        kwargs["registry"] = operators
    elif "operators" in declared:
        kwargs["operators"] = operators
    if "config_path" in declared:
        kwargs["config_path"] = config_path
    if "seed" in declared:
        kwargs["seed"] = seed
    if "llm" in declared and llm is not None:
        kwargs["llm"] = llm
    return kwargs


# --------------------------------------------------------------------------- #
# The design plane -> loop adapters
# --------------------------------------------------------------------------- #
class FaultyCascade:
    """A cascade that fails on purpose once — the §4 "injected verifier" acceptance.

    A real verifier crash is indistinguishable from a bug, and a test that induces
    one by editing a module tests the edit, not the loop. This wrapper is the
    harness's own fault injector: it raises on the *Nth* `check` call, and the
    loop's existing `verifier.error` path turns that into a round marked `error`
    with the episode continuing. `fired` records whether the fault was ever reached,
    because an injection that silently never fires proves nothing.
    """

    def __init__(self, inner: Any, *, at_call: int = 2, message: str = "injected verifier fault") -> None:
        if at_call < 1:
            raise ValueError(f"at_call must be >= 1, got {at_call}")
        self._inner = inner
        self._at = at_call
        self._message = message
        self.calls = 0
        self.fired = 0

    @property
    def spent(self) -> Any:
        """The wrapped cascade's meter; injection costs nothing of its own."""
        return getattr(self._inner, "spent", Cost())

    @property
    def notes(self) -> Any:
        """Capability boundaries the wrapped cascade declared ("not evaluated: ...")."""
        return getattr(self._inner, "notes", {})

    def check(self, assembly: Assembly) -> list[FailureCertificate]:
        self.calls += 1
        if self.calls == self._at:
            self.fired += 1
            raise RuntimeError(f"{self._message} (call {self.calls})")
        return self._inner.check(assembly)


class _StopAfterRound(Exception):
    """Raised by the observer to stop a healthy episode mid-flight, for the resume demo.

    It is not a failure: the round that triggered it is already closed in the store,
    so `--resume` continues from it exactly as it would after a kill.
    """


def state_assembly(spec: DesignSpec, *, parent: str | None, version: int) -> Assembly:
    """The pure `Assembly` for a spec: hashes from the spec, no paths, no disk.

    `geom.path` stays empty on purpose — that empty path is the signal T2's cascade
    uses to materialise before showing a verifier any triangles
    (`docs/contracts/v2-interfaces.md` §2). Building the state this way is also why
    `--resume` works from `round-<n>/spec.json` alone.
    """
    return assembly_from_spec(spec=spec, parent=parent, version=version)


@dataclass
class SpecBridge:
    """The design plane seen through the loop's `LoopRegistry` protocol.

        operator(assembly, plan) -> assembly carrying the edited spec
        spec_of(that)            -> the new `DesignSpec`
        round-<n>/spec.json      -> the state a resume rebuilds

    The operator is pure and the cascade materialises, so this bridge's whole job is
    to keep the round directories and the trace aligned with the engine's round
    indices. That alignment is the resume contract: `round-<n>/spec.json` is what
    round `n` produced, and nothing else is needed to rebuild the state it reached.
    """

    operators: Any
    out_dir: Path
    #: The engine index the next produced round will carry. On a fresh episode this
    #: is 0; on a resume it is the number of rounds the store already holds, so the
    #: directory names stay aligned with `round.index` even across processes.
    start_round: int = 0
    emit: Callable[..., None] | None = None
    produced: int = field(default=0, init=False)
    refused: int = field(default=0, init=False)

    def applicable(self, assembly: Assembly, certs: list[FailureCertificate]) -> list[str]:
        return list(self.operators.applicable(assembly, certs))

    def apply(self, assembly: Assembly, plan: RepairPlan) -> Assembly:
        index = self.start_round + self.refused + self.produced
        try:
            edited = self.operators.apply(assembly, plan)
        except Exception:
            # The round still consumed its engine index; the next produced round is
            # one further along. Counting it keeps `round-<n>` honest.
            self.refused += 1
            raise
        try:
            spec = spec_of(edited)
        except KeyError as error:
            raise RuntimeError(
                f"operator {plan.operator!r} returned an assembly without a design "
                f"spec under {SPEC_SLOT!r}; the design-plane operators must edit the "
                "spec, not the part records"
            ) from error
        self.produced += 1
        self._write_spec(self.out_dir / f"round-{index}", spec)
        if self.emit is not None:
            self.emit(
                "round.stage",
                round=index,
                stage="materialized",
                spec=spec.content_key(),
                revision=spec.revision,
            )
        return edited

    @staticmethod
    def _write_spec(directory: Path, spec: DesignSpec) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "spec.json").write_text(
            json.dumps(spec.model_dump(), ensure_ascii=False, indent=2, default=str) + "\n",
            "utf-8",
        )


# --------------------------------------------------------------------------- #
# The runtime
# --------------------------------------------------------------------------- #
class Runtime:
    """One composed process: store, engine, trace, cascade, operators.

    `build(cfg)` returns one of these bound to `cfg.out`; every `run_episode` on it
    writes into the same directory, which is what makes `--resume` a matter of
    reopening the store. The collaborators are injected rather than constructed here
    so a test can compose the same shape with contract-faithful fakes and exercise
    the resume/replay logic without waiting for T1–T3.
    """

    def __init__(
        self,
        cfg: RuntimeConfig,
        *,
        store: Store,
        engine: Engine,
        trace: Trace,
        verifiers: Any,
        operators: Any,
        design_registry: Any = None,
        policies: Mapping[str, Callable[..., Any]] | None = None,
        plugin_report: Any = None,
        notes: Sequence[str] = (),
    ) -> None:
        self.cfg = cfg
        self.out = Path(cfg.out)
        self.store = store
        self.engine = engine
        self.trace = trace
        self.verifiers = verifiers
        self.operators = operators
        self.design_registry = design_registry
        self.policies = dict(policies or {})
        self.plugin_report = plugin_report
        self.notes = list(notes)
        self._policy_cache: dict[str, Any] = {}

    # -- lifecycle ---------------------------------------------------------
    def close(self) -> None:
        """Flushes and releases the store and the trace. Safe to call twice."""
        try:
            self.trace.close()
        finally:
            self.store.close()

    def __enter__(self) -> Runtime:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # -- policies ----------------------------------------------------------
    def policy_for(self, name: str) -> Any:
        """The policy instance for `name`, built once per runtime.

        A plugin-provided factory wins over T3's builder: the plugin surface is the
        documented way to add a policy, and a plugin that shadows a builtin name is
        reported by `pah plugins list` before it is used.
        """
        if name in self._policy_cache:
            return self._policy_cache[name]
        factory = self.policies.get(name)
        if factory is not None:
            kwargs = _builder_kwargs(
                factory,
                operators=self.operators,
                config_path=self.cfg.config_path,
                seed=self.cfg.seed,
                llm=None,
            )
            policy = factory(**kwargs)
        else:
            llm = build_llm_client(self.cfg.config_path) if name == "llm" else None
            try:
                policy = resolve_policy(
                    name,
                    operators=self.operators,
                    config_path=self.cfg.config_path,
                    seed=self.cfg.seed,
                    llm=llm,
                )
            except ValueError as error:
                # A typo must say what *is* available, including the plugin names:
                # otherwise the extension point is invisible at the moment it matters.
                loaded = ", ".join(sorted(self.policies)) or "none"
                raise RuntimeError(
                    f"{error} Policy plugins loaded in this run: {loaded}."
                ) from error
        self._policy_cache[name] = policy
        return policy

    # -- the episode -------------------------------------------------------
    def run_episode(
        self,
        spec: DesignSpec,
        *,
        policy_name: str | None = None,
        rounds: int | None = None,
        out_dir: str | Path | None = None,
        resume: bool | None = None,
    ) -> dict:
        """Run one episode and return the report that was written to `report.json`.

        The report is the return value as well as a file because the eval runner and
        the tests need the same facts as the CLI, and a second formatter would be a
        second opportunity for the two to disagree.
        """
        policy_name = policy_name or self.cfg.policy
        rounds = int(rounds if rounds is not None else self.cfg.rounds)
        out_dir = Path(out_dir) if out_dir is not None else self.out
        resume = bool(self.cfg.resume if resume is None else resume)

        if out_dir.resolve() != self.out.resolve():
            raise RuntimeError(
                f"runtime is bound to {self.out} but asked to run in {out_dir}; "
                "build one runtime per run directory"
            )
        out_dir.mkdir(parents=True, exist_ok=True)
        db_path = out_dir / "db.sqlite"
        marker = self.store.kv.get(RUN_MARKER)
        if not resume and marker:
            raise RuntimeError(
                f"{out_dir} already holds run {marker!r}; pass --resume to continue "
                "it, or choose a different --out"
            )

        policy = self.policy_for(policy_name)
        ticket_id = key("ticket", spec.spec_id, spec.content_key(), policy_name)
        started = time.monotonic()

        if resume:
            if not marker:
                raise RuntimeError(
                    f"nothing to resume in {out_dir}: no ticket was ever recorded "
                    "there. A --resume must point at the directory of an interrupted run."
                )
            if marker != ticket_id:
                raise RuntimeError(
                    f"the run in {out_dir} belongs to ticket {marker!r}, not to the "
                    f"spec/policy pair being resumed (spec {spec.spec_id!r}, policy "
                    f"{policy_name!r}). The spec file may have been edited; resume with "
                    "the spec the run started from."
                )
            ticket = self.store.tickets.get(ticket_id)
            if ticket is None:
                raise RuntimeError(
                    f"nothing to resume in {out_dir}: ticket {ticket_id!r} is recorded "
                    "as the run's owner but cannot be read back"
                )
            if ticket.ended is not None:
                raise RuntimeError(
                    f"the episode in {out_dir} already ended with outcome "
                    f"{ticket.ended.outcome!r}; there is nothing left to resume"
                )
            dangling = self._close_dangling_rounds(ticket_id)
            ticket = self.store.tickets.get(ticket_id)
            budget = ticket.mandate.budget
            assembly, start_round = self._resume_state(out_dir, ticket, spec)
            notes = [f"resumed from round {start_round}", RESUME_CHARGE_NOTE]
            if rounds != budget.rounds:
                notes.append(
                    f"resume keeps the recorded budget (rounds={budget.rounds}); "
                    f"--rounds {rounds} is recorded but not applied"
                )
            if dangling:
                notes.append(f"closed {dangling} interrupted round(s) as 'aborted'")
        else:
            budget = budget_for(
                rounds, geom_calls=self.cfg.geom_calls, sim_calls=self.cfg.sim_calls
            )
            ticket = Ticket(
                id=ticket_id,
                lineage=f"design:{spec.spec_id}",
                mandate=Mandate(
                    strength="explicit",
                    basis={
                        "via": "pah run",
                        "spec": spec.spec_id,
                        "content_key": spec.content_key(),
                        "rounds": rounds,
                        "policy": policy_name,
                        "seed": self.cfg.seed,
                    },
                    scope_parts=tuple(part.id for part in spec.parts),
                    scope_operations=tuple(sorted(self._operator_ids())),
                    frozen_parts=(),
                    budget=budget,
                    policy_version=f"{policy_name}@1",
                ),
                created_at="",
            )
            self.engine.open_ticket(ticket)
            self.store.kv.set(RUN_MARKER, ticket_id)
            assembly = state_assembly(spec, parent=None, version=1)
            start_round = 0
            notes = []
            self._write_spec_file(out_dir / "spec.json", spec)
            self._write_spec_file(out_dir / "spec-initial.json", spec)

        bridge = SpecBridge(
            operators=self.operators,
            out_dir=out_dir,
            start_round=start_round,
            emit=self._emit,
        )
        stopped_at: list[int] = []
        observer = self._observer(stopped_at)
        loop = RepairLoop(
            engine=self.engine,
            verifiers=self.verifiers,
            registry=bridge,
            policy=policy,
            observer=observer,
            # The mandate is the real ceiling; this only stops a runaway policy.
            max_rounds=max(1, budget.rounds),
        )
        result: EpisodeResult | None = None
        stopped = False
        try:
            result = self._run_loop(loop, ticket, assembly, spec)
        except _StopAfterRound:
            stopped = True

        ticket = self.store.tickets.get(ticket_id)
        if ticket is None:  # pragma: no cover - the store cannot lose an open ticket
            raise RuntimeError(f"ticket {ticket_id!r} disappeared from {db_path}")

        # The final design intent: the assembly the loop left behind, or (on a clean
        # stop) the state the last produced round wrote.
        if result is not None:
            final_spec = spec_of(result.assembly)
            certs = list(result.certificates)
            verdict = result.verdict
            reason = result.reason
        else:
            final_spec = self._current_spec(out_dir, ticket, spec)
            certs = []
            verdict = "stopped"
            reason = f"stopped after {stopped_at[-1]} round(s) for the resume demo"
        self._write_spec_file(out_dir / "spec.json", final_spec)

        report = self._report(
            spec=spec,
            final_spec=final_spec,
            ticket=ticket,
            policy_name=policy_name,
            rounds_requested=rounds,
            resume=resume,
            stopped=stopped,
            verdict=verdict,
            reason=reason,
            certificates=certs,
            bridge=bridge,
            budget=budget,
            wall_s=time.monotonic() - started,
            notes=notes,
        )
        (out_dir / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", "utf-8"
        )
        return report

    # -- resume plumbing ---------------------------------------------------
    def _run_loop(
        self, loop: RepairLoop, ticket: Ticket, assembly: Assembly, spec: DesignSpec
    ) -> EpisodeResult:
        """`loop.run`, handing a resumed episode the fingerprint history it cannot recall.

        Cycle detection is "have I been here before"; a new process only knows the
        snapshot it was handed, so a resumed episode would walk back into geometry the
        original run had already rejected and take one extra round. `seen` is the
        kernel's documented fix (`policy/loop.py:RepairLoop.run`), and the history is
        entirely recoverable: every closed round recorded its `geom_hash` artifact,
        and the initial snapshot is a pure function of the spec the caller passed (the
        ticket id already proves it is the same content).

        The signature is checked because `loop.py` is frozen kernel: a tree pinned to
        the older kernel still runs, with the one-extra-round limit documented in the
        report notes instead of a `TypeError`.
        """
        seen = [
            str(rnd.artifacts.get("geom_hash"))
            for rnd in ticket.rounds
            if rnd.artifacts.get("geom_hash")
        ]
        seen.append(assembly_fingerprint(state_assembly(spec, parent=None, version=1)))
        if "seen" in inspect.signature(RepairLoop.run).parameters:
            return loop.run(ticket, assembly, seen=seen)
        return loop.run(ticket, assembly)

    def _close_dangling_rounds(self, ticket_id: str) -> int:
        """A process killed mid-round leaves a round with no verdict.

        The loop would otherwise count it against the budget and never resolve it,
        and the report would show a round that neither passed nor failed. Closing it
        as `aborted` is the honest record: it happened, it was interrupted, and its
        evidence is lost. The round is *not* rerun, because a half-applied operator
        may already have written state.
        """
        ticket = self.store.tickets.get(ticket_id)
        if ticket is None:
            return 0
        dangling = [rnd for rnd in ticket.rounds if rnd.verdict is None]
        for rnd in dangling:
            try:
                self.engine.close_round(
                    rnd.id,
                    verdict="aborted",
                    evidence=[],
                    cost=Cost(),
                    artifacts={},
                )
            except Exception as error:  # noqa: BLE001 - a refusal means it was already closed
                raise RuntimeError(
                    f"could not close the interrupted round {rnd.id!r} in {ticket_id!r}: "
                    f"{type(error).__name__}: {error}"
                ) from error
        return len(dangling)

    def _resume_state(
        self, out_dir: Path, ticket: Ticket, spec: DesignSpec
    ) -> tuple[Assembly, int]:
        """Rebuild the state the interrupted episode had reached, from its spec files.

        The spec is read back from `round-<n>/spec.json` — the file the episode wrote
        when it produced that round — and turned into a pure `Assembly`. Nothing is
        read back from geometry: `geom.hash` is spec-only by contract, so the resumed
        run computes the same fingerprint the interrupted one would have.
        """
        closed = self._rounds_with_design(ticket)
        start_round = len(ticket.rounds)
        if not closed:
            return state_assembly(spec, parent=None, version=1), 0
        last = max(rnd.index for rnd in closed)
        spec_path = out_dir / f"round-{last}" / "spec.json"
        if not spec_path.exists():
            raise RuntimeError(
                f"cannot resume: {spec_path} is missing, so the design that round "
                f"{last} produced is unknown. Refusing to guess it from the trace."
            )
        state_spec = DesignSpec.model_validate_json(spec_path.read_text("utf-8"))
        return state_assembly(state_spec, parent=None, version=last + 2), start_round

    @staticmethod
    def _rounds_with_design(ticket: Ticket) -> list[Round]:
        """Rounds that actually produced a design — the only ones a resume can rebuild.

        Not "every closed round": an operator that raised closes its round as `error`
        with no artefact, and a round a killed process abandoned is closed as
        `aborted` on the next start. Neither wrote a `round-<n>/spec.json`, so neither
        can be the state a resume starts from.
        """
        return [rnd for rnd in ticket.rounds if rnd.artifacts.get("geom_hash")]

    def _current_spec(self, out_dir: Path, ticket: Ticket, spec: DesignSpec) -> DesignSpec:
        """The last spec a produced round wrote, or the episode's input spec."""
        produced = self._rounds_with_design(ticket)
        if produced:
            path = out_dir / f"round-{max(rnd.index for rnd in produced)}" / "spec.json"
            if path.exists():
                return DesignSpec.model_validate_json(path.read_text("utf-8"))
        return spec

    # -- trace -------------------------------------------------------------
    def _emit(self, kind: str, **fields: Any) -> None:
        """The runtime's own trace rows (round artefacts), separate from the loop's."""
        self.trace.emit(kind, **fields)

    def _observer(self, stopped_at: list[int]) -> Callable[[str, dict], None]:
        """Forward loop events to the trace, and honour `--stop-after-round`.

        Stopping raises out of the observer *after* the round was closed in the
        store: the episode is resumable, not half-written. The exception is the
        runtime's own control flow, never visible as a trace event.
        """
        limit = self.cfg.stop_after_round
        seen = {"closed": 0}

        def observe(kind: str, fields: dict) -> None:
            self.trace.emit(kind, **fields)
            if kind == "round.close":
                seen["closed"] += 1
                stopped_at.append(seen["closed"])
                if limit is not None and seen["closed"] >= limit:
                    raise _StopAfterRound(seen["closed"])

        return observe

    # -- geometry ----------------------------------------------------------
    def _geometry(self, out_dir: Path, final_spec: DesignSpec) -> dict:
        """What the cascade's materialiser left in `<out>/geometry/`.

        Reported rather than produced: the runtime does not build geometry. `exists`
        per part is the check that the cascade really materialised before judging —
        a run whose verifiers saw no triangles has no evidence worth the name.
        """
        directory = out_dir / "geometry"
        parts: dict[str, dict] = {}
        for part in final_spec.parts:
            path = directory / f"{part.id}.stl"
            parts[part.id] = {
                "path": str(path),
                "hash": geometry_hash(final_spec, part.id),
                "exists": path.exists(),
            }
        files = sorted(item.name for item in directory.glob("*.stl")) if directory.exists() else []
        return {
            "dir": str(directory),
            "files": files,
            "parts": parts,
            "complete": bool(parts) and all(detail["exists"] for detail in parts.values()),
        }

    # -- report ------------------------------------------------------------
    def _operator_ids(self) -> tuple[str, ...]:
        ids = getattr(self.operators, "ids", None)
        if callable(ids):
            return tuple(ids())
        return tuple(sorted(getattr(self.operators, "_operators", {})))

    def _verifier_spend(self) -> dict[str, int]:
        """The cascade's own meter, if it keeps one (`policy/loop.py:_meter_now`)."""
        source = getattr(self.verifiers, "spent", None)
        value = source() if callable(source) else source
        if isinstance(value, Cost):
            return {"geom_calls": value.geom_calls, "sim_calls": value.sim_calls}
        if isinstance(value, dict):
            return {
                "geom_calls": int(value.get("geom_calls", 0) or 0),
                "sim_calls": int(value.get("sim_calls", 0) or 0),
            }
        return {"geom_calls": 0, "sim_calls": 0}

    def _verifier_notes(self) -> dict[str, str]:
        """What the cascade says it could *not* evaluate, and why.

        A capability boundary is a third state beside "passed" and "failed": the
        physics tier that could not build a body, the overhang check with no declared
        build orientation. Folding it into either count is how a report starts claiming
        more than the tools measured, so it gets its own section and its own count.
        """
        source = getattr(self.verifiers, "notes", None)
        value = source() if callable(source) else source
        if not isinstance(value, Mapping):
            return {}
        return {str(key): str(reason) for key, reason in value.items()}

    def _report(
        self,
        *,
        spec: DesignSpec,
        final_spec: DesignSpec,
        ticket: Ticket,
        policy_name: str,
        rounds_requested: int,
        resume: bool,
        stopped: bool,
        verdict: str,
        reason: str,
        certificates: Sequence[FailureCertificate],
        bridge: SpecBridge,
        budget: Budget,
        wall_s: float,
        notes: Sequence[str],
    ) -> dict:
        rounds = list(ticket.rounds)
        used = ticket.used
        environment_errors = sum(
            1
            for rnd in rounds
            for cert in rnd.evidence
            if cert.family == "environment"
        ) + sum(1 for cert in certificates if cert.family == "environment")
        unknown_rounds = [rnd.index for rnd in rounds if rnd.verdict in ("error", "aborted")]
        error_rounds = [rnd.index for rnd in rounds if rnd.verdict == "error"]
        verifier_errors = [
            row
            for row in self.trace.rows(skip_corrupt=True)
            if row.get("kind") == "verifier.error"
        ]
        disabled: dict[str, str] = {}
        if self.design_registry is not None:
            disabled = {
                verifier_id: self.design_registry.why_disabled(verifier_id)
                for verifier_id in sorted(self.design_registry.disabled())
            }
        geometry = self._geometry(self.out, final_spec)
        verifier_notes = self._verifier_notes()
        return {
            "schema": "pah-run-report/1",
            # 口径隔离: a design-plane report never carries a dataset number. The
            # field is explicit so a future reader cannot merge the two by accident.
            "plane": "design",
            "dataset_runs_touched": [],
            "run": {
                "out": str(self.out),
                "policy": policy_name,
                "rounds_requested": rounds_requested,
                "resume": resume,
                "seed": self.cfg.seed,
                "inject_fault": self.cfg.inject_fault,
                "stop_after_round": self.cfg.stop_after_round,
                "status": "stopped" if stopped else "completed",
                "wall_s": round(wall_s, 3),
            },
            "ticket": {
                "id": ticket.id,
                "outcome": ticket.ended.outcome if ticket.ended else None,
                "reason": ticket.ended.reason if ticket.ended else "",
            },
            "episode": {"verdict": verdict, "reason": reason, "rounds": len(rounds)},
            "budget": {
                "rounds": {
                    "budget": budget.rounds,
                    "used": used.rounds,
                    "left": max(0, budget.rounds - used.rounds),
                },
                "geom_calls": {
                    "budget": budget.geom_calls,
                    "charged": used.geom_calls,
                    "left": max(0, budget.geom_calls - used.geom_calls),
                },
                "sim_calls": {
                    "budget": budget.sim_calls,
                    "charged": used.sim_calls,
                    "left": max(0, budget.sim_calls - used.sim_calls),
                },
                # The cascade's meter, kept beside the ledger rather than inside it:
                # the ledger is what the engine enforced, the meter is what the tools
                # actually ran.
                "verifier_spend": self._verifier_spend(),
                "rounds_produced": bridge.produced,
                "rounds_without_design": bridge.refused,
            },
            "design": {
                "spec_id": spec.spec_id,
                "kind": final_spec.interface.kind,
                "requirement": final_spec.requirement,
                "initial_content_key": spec.content_key(),
                "final_content_key": final_spec.content_key(),
                "initial_clearance_mm": round(spec.clearance_mm, 6),
                "final_clearance_mm": round(final_spec.clearance_mm, 6),
                "final_revision": final_spec.revision,
                "provenance": dict(final_spec.provenance),
                "fit_band_mm": [
                    final_spec.constraints.min_clearance_mm,
                    final_spec.constraints.max_clearance_mm,
                ],
            },
            "rounds": [
                {
                    "index": rnd.index,
                    "operator": rnd.plan.operator,
                    "params": dict(rnd.plan.params),
                    "proposed_by": rnd.plan.proposed_by,
                    "reason": rnd.plan.reason,
                    "verdict": rnd.verdict,
                    "stage": rnd.stage,
                    "codes": sorted({cert.code for cert in rnd.evidence}),
                    "environment_errors": sum(
                        1 for cert in rnd.evidence if cert.family == "environment"
                    ),
                    "geom_calls": rnd.cost.geom_calls,
                    "sim_calls": rnd.cost.sim_calls,
                    "llm_calls": rnd.cost.llm_calls,
                    "tokens": rnd.cost.tokens,
                    "geom_hash": rnd.artifacts.get("geom_hash", ""),
                }
                for rnd in rounds
            ],
            "llm": {
                # What the decision layer cost, summed from closed rounds. Money is
                # not re-derived here: the provider adapter declares its billing unit
                # (`Usage.billing`) and only `per_token` has a price, so calls and
                # tokens are the comparable numbers across providers.
                "calls": sum(rnd.cost.llm_calls for rnd in rounds),
                "tokens": sum(rnd.cost.tokens for rnd in rounds),
                "billing_note": "calls and tokens are the comparison unit; money is provider-declared",
            },
            "counts": {
                # Whether the *policy* closed the loop on this design. Not a dataset
                # yield, and never added to the two columns below.
                "repaired_by_policy": int(verdict == "repaired"),
                "physical_failures_final": len(hard_codes(certificates)),
                "final_codes": sorted({cert.code for cert in certificates}),
                # Tool health, kept strictly apart from the repair outcome.
                "environment_errors": environment_errors,
                "verifier_error_events": len(verifier_errors),
                # State that could not be established: an errored/aborted round, or a
                # final verdict of `failed`. Never counted as a pass.
                "unknown": len(unknown_rounds) + int(verdict == "failed"),
                "error_rounds": error_rounds,
                # Capability boundaries: checks the cascade says it did not run. A
                # third state — not a physical failure, not a tool fault, not a pass.
                "not_evaluated": len(verifier_notes),
                "not_evaluated_by": sorted(verifier_notes),
            },
            "verifiers": {
                "enabled": list(self.design_registry.ids())
                if self.design_registry is not None
                else [],
                "disabled": disabled,
                "notes": verifier_notes,
            },
            "trace": {
                "path": str(self.trace.path),
                "rows": len(self.trace.rows(skip_corrupt=True)),
                "replay_hashes": Trace.replay_hashes(self.trace.path),
            },
            "geometry": geometry,
            "plugins": {
                "loaded": len(self.plugin_report.loaded) if self.plugin_report else 0,
                "failed": [
                    {"name": record.name, "error": record.error}
                    for record in (self.plugin_report.blocking if self.plugin_report else ())
                ],
                "warnings": [
                    {"name": record.name, "error": record.error}
                    for record in (self.plugin_report.warnings if self.plugin_report else ())
                ],
            },
            "notes": list(notes) + list(self.notes),
        }

    @staticmethod
    def _write_spec_file(path: Path, spec: DesignSpec) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(spec.model_dump(), ensure_ascii=False, indent=2, default=str) + "\n",
            "utf-8",
        )


# --------------------------------------------------------------------------- #
# Building one
# --------------------------------------------------------------------------- #
def build(cfg: RuntimeConfig) -> Runtime:
    """Compose a runtime for `cfg.out`.

    The order is a dependency order, not a preference: the neighbours are resolved
    *before* anything is created on disk (a refused command must not leave a
    half-run directory), plugins register into the real registries before the
    verifier gate runs (so a plugin verifier is gated like a builtin one), and the
    gate runs before the cascade is built (a verifier that disagrees with its own
    fixtures must not influence a single certificate).
    """
    from pah import plugins  # noqa: PLC0415  (keeps this module importable without the plugin layer)

    out = Path(cfg.out)
    resolve_materializer()
    design_registry = resolve_design_registry()
    operators = resolve_operators()

    host = plugins.PluginHost(verifiers=design_registry, operators=operators)
    plugin_report = plugins.load(host, config_path=cfg.config_path)
    if plugin_report.blocking:
        names = ", ".join(record.name for record in plugin_report.blocking)
        raise plugins.PluginError(
            f"declared plugin(s) failed to load: {names}; run `pah plugins list` for "
            "the reason. Refusing to run an episode whose action space differs from "
            "the one requested."
        )
    # A builtin module that is mid-flight is a warning, not a refusal: the episode runs
    # without that capability, and the report says so instead of letting a reader
    # assume the capability was there.
    plugin_notes = [
        f"plugin warning: {record.name}: {record.error}"
        for record in plugin_report.warnings
    ]

    design_registry.sanity()
    # The cascade materialises into `root/geometry`, so the directory has to exist
    # by the time it is constructed — but not before every neighbour resolved.
    out.mkdir(parents=True, exist_ok=True)
    verifiers = resolve_cascade(design_registry, out)
    if cfg.inject_fault == "verifier":
        # Fail the first post-repair verification, not the initial one: the episode
        # must have a round to mark as `error` before it recovers.
        verifiers = FaultyCascade(verifiers, at_call=2)
    elif cfg.inject_fault:
        raise RuntimeError(
            f"unknown --inject-fault {cfg.inject_fault!r}; this build injects {FAULTS}"
        )

    store = Store(out / "db.sqlite")
    try:
        trace = Trace(out / "trace.jsonl")
    except BaseException:
        store.close()
        raise
    return Runtime(
        cfg,
        store=store,
        engine=Engine(store),
        trace=trace,
        verifiers=verifiers,
        operators=operators,
        design_registry=design_registry,
        policies=host.policies,
        plugin_report=plugin_report,
        notes=plugin_notes,
    )
