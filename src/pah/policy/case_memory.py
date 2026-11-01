"""Case memory: the harness gets better at repairing *the same failure* by remembering.

Why this is a plugin and not a change to the loop
--------------------------------------------------
The user's requirement for the self-iteration mechanism is "用得越多，生成可装配部件的能力越强",
and `docs/contracts/v2-interfaces.md` §6 draws the hard consequence: a new capability must
arrive as **a plugin that registers into a registry**, never as an edit to `core/` or to
another owner's module. So this file implements `policy.loop.LoopPolicy` — the exact
protocol `RepairLoop` already talks to — and `register(host)` puts a factory into
`host.policies["case_memory"]`. `core/engine.py`, `core/model.py` and `policy/loop.py` are
untouched: the loop cannot tell this policy from the deterministic rule policy, which is
the point. That is the evidence the extension point is real.

What it does (and what it deliberately does not)
-----------------------------------------------
It keeps a **ledger** in SQLite: one row per attempt —

    (spec.content_key, kind, cert_codes, plan.operator, plan.params, outcome, rounds_used)

— and on the next failure of the same *shape* it proposes the action that worked before,
ranked by success rate. Retrieval is SQL (`GROUP BY operator, params`, `HAVING successes > 0`,
`ORDER BY successes/decided DESC`); nothing is embedded, nothing is trained, and there is no
model in this file. That is not a limitation to apologise for: an offline ledger is exactly
what makes the mechanism *auditable* ("this plan came from 3/4 similar cases") and what keeps
the policy inside the "no training in the harness" rule (`docs/pah-dev-doc-v1.md` §6, contract §9).

Four decisions worth stating
----------------------------
1. **The address comes from the current design; only the knobs are remembered.** A stored
   `interface_id` / `parts` names parts of *another* spec, so replaying it verbatim would
   address geometry that is not there. The memory stores "set clearance to 0.30", and the
   current spec says *which* interface.
2. **A knob with no declared bound is refused, not replayed.** `clearance_mm` is bounded by
   the current spec's `FitConstraints` band, `delta_mm` by half the smallest part, `kind` by
   the builtin kind set, and an operator's own pydantic model has the last word when the
   registry is available. An operator this policy cannot bound (`shift_part`, `rotate_part`,
   whose payload is a world-frame vector about *one named part*) is never replayed: the
   memory does not store part roles, and guessing one is how a fix from case A lands on the
   wrong part of case B.
3. **The match is exact on `(kind, sorted certificate codes)`.** A repair that worked for
   `insufficient_clearance` is not assumed to work for
   `insufficient_clearance + insertion_path_blocked`. Fuzzy matching is a later change; doing
   it silently here would make the ledger's score meaningless, because "success rate" would
   no longer describe one recurrence class.
4. **The outcome is read off the next round, and the last round needs an explicit flush.**
   `RepairLoop` appends a closed `Round` to the history *before* asking for the next plan, so
   `propose` records the previous attempt's verdict on entry. The one round it never sees is
   the last one — the loop returns on `pass` without asking again. `flush(history)` is public
   for exactly that reason: a caller that wants the winning action credited calls it once
   after the episode. Without it, every episode's final success is lost and the memory only
   ever learns from failures. `EpisodeResult` carries no rounds, so the runtime should pass
   `ticket.rounds` (one line, and T4 owns `app/runtime.py`).

The scoring, in one line: `rate = passes / (passes + fails)`; ties broken by more passes, then
by fewer rounds used (a fix that works in round 1 is worth more than one that needs three),
then by recency. Error rounds are stored but excluded from the rate — an operator that raised
is not evidence that the action is wrong.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, get_args

from pah.core.ids import key
from pah.core.model import Assembly, Budget, Cost, FailureCertificate, RepairPlan, Round
from pah.design.api import spec_of
from pah.design.spec import ConnectorKindName, DesignSpec, builtin_params_model

__all__ = [
    "API_VERSION",
    "BUILTIN_PARAMS_KINDS",
    "CaseMemory",
    "CaseMemoryPolicy",
    "CaseRecord",
    "DEFAULT_MEMORY_PATH",
    "OUTCOMES",
    "PLUGIN_NAME",
    "api_version",
    "build_case_memory_policy",
    "kind",
    "register",
    "signature_of",
]

# --------------------------------------------------------------------------- #
# Plugin declarations (`docs/contracts/v2-interfaces.md` §6)
# --------------------------------------------------------------------------- #
#: The contract version this module was written against. A future harness that
#: bumps the API refuses this module *by name* instead of failing three frames deep.
api_version = API_VERSION = "1"

#: What lands in `host.policies`. A "policy" plugin may only add policies.
kind = "policy"

#: The name a runtime asks for: `--policy case_memory`, `pah.toml`, or `host.policies[...]`.
PLUGIN_NAME = "case_memory"

#: Outcomes a row can carry. `error` is stored and excluded from the rate.
OUTCOMES = ("pass", "fail", "error")

#: Where the ledger lives when the caller gives no path. The design plane's runs go
#: under `runs/`, and this file is deliberately one per workspace rather than one per
#: episode: the whole point is that episode *n+1* can read episode *n*.
DEFAULT_MEMORY_PATH = Path("runs") / "case_memory.sqlite"

#: The addressing fields a plan carries. They are *replaced* from the current spec.
_ADDRESS_KEYS = ("interface_id", "parts", "part_id")

#: Operators whose payload names a world-frame vector about one part. Not replayable:
#: the memory stores no part role, so the same numbers could land on the wrong part.
_PART_ADDRESSED = frozenset({"shift_part", "rotate_part"})

#: The connector kinds this build can materialise. A remembered switch_connector
#: whose target kind is not here is refused rather than handed to a registry that
#: would raise on it.
BUILTIN_PARAMS_KINDS = frozenset(get_args(ConnectorKindName))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS case_memory (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    spec_content_key  TEXT    NOT NULL,
    spec_id           TEXT    NOT NULL DEFAULT '',
    kind              TEXT    NOT NULL,
    cert_codes        TEXT    NOT NULL,
    operator          TEXT    NOT NULL,
    params            TEXT    NOT NULL,
    targets           TEXT    NOT NULL DEFAULT '[]',
    outcome           TEXT    NOT NULL,
    rounds_used       INTEGER NOT NULL DEFAULT 0,
    created_at        TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS case_memory_lookup ON case_memory (kind, cert_codes);
CREATE INDEX IF NOT EXISTS case_memory_spec   ON case_memory (spec_content_key);
"""


# --------------------------------------------------------------------------- #
# Small pure helpers
# --------------------------------------------------------------------------- #
def _normalise(value: object) -> object:
    """A payload as it is stored and compared: floats to six decimals, tuples to lists.

    The same convention `design_policies._normalise` uses for plan signatures, so a
    plan recorded here and the same plan read back from a store compare equal. Rounding
    is about *how* a float was computed, not *what* was asked for.
    """
    if isinstance(value, bool) or value is None or isinstance(value, int):
        return value
    if isinstance(value, float):
        return round(value, 6) + 0.0  # `+ 0.0` folds -0.0 into 0.0
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return {str(name): _normalise(item) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalise(item) for item in value]
    return str(value)


def _signal(params: Mapping[str, object]) -> str:
    """A stable text for one parameter payload (JSON, sorted keys, normalised)."""
    return json.dumps(_normalise(dict(params)), sort_keys=True)


def signature_of(certificates: Iterable[FailureCertificate]) -> str:
    """The recurrence class of a piece of evidence: its physical codes, sorted.

    Environment-family certificates are excluded: "the simulator could not run" is not
    a failure class a repair can be remembered against, and letting it into the
    signature would make the same physical failure match two different keys.
    """
    codes = sorted(
        {str(cert.code) for cert in certificates if cert.family != "environment"}
    )
    return "+".join(codes)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _spec_or_none(assembly: Assembly) -> DesignSpec | None:
    """The design spec, or `None` for an assembly that is not on the design plane."""
    try:
        return spec_of(assembly)
    except KeyError:
        return None


# --------------------------------------------------------------------------- #
# The ledger
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CaseRecord:
    """One ranked candidate: an action, and how it has done on this failure class."""

    operator: str
    params: dict
    passes: int
    decided: int
    rounds_sum: int
    last_used: str

    @property
    def rate(self) -> float:
        return self.passes / self.decided if self.decided else 0.0

    @property
    def mean_rounds(self) -> float:
        return self.rounds_sum / self.decided if self.decided else 0.0

    def describe(self) -> str:
        return (
            f"{self.operator}({', '.join(f'{k}={v}' for k, v in sorted(self.params.items()))}) "
            f"@{self.passes}/{self.decided}"
        )


class CaseMemory:
    """The SQLite ledger. One file, one table, no schema surprises.

    `path` is whatever the caller gives (a `runs/` file is the convention); `":memory:"`
    is allowed and is what the tests use when they do not care to leave a file behind.
    The connection is opened here and committed per row, so a process killed between
    rounds loses at most the round it was in — which is the same guarantee the loop's
    round bookkeeping gives.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        target = str(path if path is not None else DEFAULT_MEMORY_PATH)
        self.path = Path(target)
        if target != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(target)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        #: Rows written by this instance, for a caller that wants the evidence.
        self.recorded = 0

    # -- writing -----------------------------------------------------------
    def record(
        self,
        *,
        content_key: str,
        kind: str,
        cert_codes: str,
        operator: str,
        params: Mapping[str, object],
        targets: Sequence[str] = (),
        outcome: str,
        rounds_used: int,
        spec_id: str = "",
    ) -> None:
        """Append one attempt. `outcome` is one of `OUTCOMES`; no row is ever updated.

        Append-only on purpose: "this action has worked 3 of 4 times" must stay
        answerable after the fifth attempt fails, and an UPDATE would quietly rewrite
        history into whatever the last run believed.
        """
        if outcome not in OUTCOMES:
            raise ValueError(f"outcome must be one of {OUTCOMES!r}, got {outcome!r}")
        self._conn.execute(
            "INSERT INTO case_memory"
            " (spec_content_key, spec_id, kind, cert_codes, operator, params, targets,"
            "  outcome, rounds_used, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                str(content_key),
                str(spec_id),
                str(kind),
                str(cert_codes),
                str(operator),
                _signal(params),
                json.dumps([str(item) for item in targets]),
                str(outcome),
                int(rounds_used),
                _utc_now(),
            ),
        )
        self._conn.commit()
        self.recorded += 1

    # -- reading -----------------------------------------------------------
    def candidates(
        self, *, kind: str, cert_codes: str, limit: int = 8
    ) -> list[CaseRecord]:
        """Proven actions for one failure class, best first.

        `HAVING passes > 0` is the task's "successes only" rule; the ordering is
        success rate, then more evidence, then fewer rounds, then recency. `params`
        is grouped as the stored JSON text, so two payloads that differ only in float
        noise are one candidate.
        """
        rows = self._conn.execute(
            """
            SELECT operator, params,
                   SUM(outcome = 'pass')                 AS passes,
                   SUM(outcome IN ('pass', 'fail'))      AS decided,
                   SUM(rounds_used)                      AS rounds_sum,
                   MAX(created_at)                       AS last_used
              FROM case_memory
             WHERE kind = ? AND cert_codes = ?
             GROUP BY operator, params
            HAVING passes > 0 AND decided > 0
             ORDER BY (CAST(passes AS REAL) / decided) DESC,
                      passes DESC,
                      (CAST(rounds_sum AS REAL) / decided) ASC,
                      last_used DESC
             LIMIT ?
            """,
            (str(kind), str(cert_codes), int(limit)),
        ).fetchall()
        return [
            CaseRecord(
                operator=str(row["operator"]),
                params=json.loads(row["params"]),
                passes=int(row["passes"]),
                decided=int(row["decided"]),
                rounds_sum=int(row["rounds_sum"]),
                last_used=str(row["last_used"]),
            )
            for row in rows
        ]

    def rows(self) -> list[dict]:
        """Every row, oldest first — the audit view a report or a test reads."""
        return [
            dict(row)
            for row in self._conn.execute(
                "SELECT * FROM case_memory ORDER BY id ASC"
            ).fetchall()
        ]

    def stats(self) -> dict:
        """Counts by outcome and by failure class, for the run report."""
        by_outcome = {
            str(row["outcome"]): int(row["n"])
            for row in self._conn.execute(
                "SELECT outcome, COUNT(*) AS n FROM case_memory GROUP BY outcome"
            )
        }
        by_class = {
            f"{row['kind']}|{row['cert_codes']}": int(row["n"])
            for row in self._conn.execute(
                "SELECT kind, cert_codes, COUNT(*) AS n FROM case_memory"
                " GROUP BY kind, cert_codes"
            )
        }
        return {"total": sum(by_outcome.values()), "by_outcome": by_outcome, "by_class": by_class}

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "CaseMemory":
        return self

    def __exit__(self, *exception: object) -> None:
        self.close()


# --------------------------------------------------------------------------- #
# The policy
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _Pending:
    """The provenance of a plan this policy issued, waiting for its round's verdict.

    The failure signature is stashed at *proposal* time because that is when it is
    known: by the time the round closes, the certificates describe the design *after*
    the repair, and a ledger keyed on those would answer the wrong question.
    """

    content_key: str
    spec_id: str
    kind: str
    signature: str
    operator: str
    params: dict
    targets: tuple[str, ...]
    rounds_used: int


@dataclass
class CaseMemoryPolicy:
    """`loop.LoopPolicy` that consults the ledger before it consults its inner policy.

    `inner` is delegated to on every miss (an unseen failure, an unbound knob, a
    candidate already tried this episode) and may be `None` — then a miss is `None`,
    which the loop reads as "no action", not as success. `registry` is optional and
    is only used to validate a replayed payload against the operator's own model and
    to read `operator.targets(params)`; without it the policy still works, and a bad
    replay becomes an `error` round the ledger records.
    """

    inner: Any = None
    memory: CaseMemory | None = None
    registry: Any = None
    name: str = PLUGIN_NAME
    #: Declared cost of one round, used by the loop's budget preview. A replayed plan
    #: costs exactly one materialisation, same as a rule-based one.
    geom_calls_per_round: int = 1
    _pending: dict[str, _Pending] = field(default_factory=dict, init=False, repr=False)
    #: Where the last proposal came from: `"case_memory"`, `"inner"` or `"none"`.
    last_source: str = field(default="none", init=False)
    #: The record the last proposal replayed, for a trace to cite.
    last_record: CaseRecord | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        # A ledger is opened on first use, not in `__init__`: constructing a policy must
        # not create a file, or importing a plugin would write to whatever directory the
        # process happens to be in.
        if self.memory is None:
            self.memory = CaseMemory()

    # -- the loop's entry point -------------------------------------------
    def propose(
        self,
        assembly: Assembly,
        certs: list[FailureCertificate],
        history: list[Round],
        budget_left: Budget,
    ) -> RepairPlan | None:
        """Replay the best proven action for this failure class, else ask `inner`."""
        self.last_source = "none"
        self.last_record = None
        self.flush(history)
        spec = _spec_or_none(assembly)
        evidence = [cert for cert in certs if cert.family != "environment"]
        signature = signature_of(evidence)
        if spec is not None and signature:
            plan = self._replay(spec, signature, history)
            if plan is not None:
                self.last_source = "case_memory"
                self._remember(spec, signature, plan, history)
                return plan
        plan = self._delegate(assembly, certs, history, budget_left)
        if plan is not None:
            self.last_source = "inner"
            if spec is not None:
                self._remember(spec, signature, plan, history)
        return plan

    # -- bookkeeping -------------------------------------------------------
    def flush(self, history: Sequence[Round]) -> int:
        """Record the verdict of every round this policy proposed that has closed.

        Idempotent per plan id: a round already recorded is popped from `_pending`, so
        calling this twice (a resumed episode, a defensive caller) writes nothing twice.
        Returns the number of rows written — which is the number of attempts credited.
        """
        written = 0
        for round_ in history:
            plan_id = str(getattr(round_.plan, "id", ""))
            pending = self._pending.get(plan_id)
            if pending is None:
                continue
            verdict = getattr(round_, "verdict", None)
            if verdict is None:  # the round is still open; keep the provenance
                continue
            self._pending.pop(plan_id, None)
            self.memory.record(
                content_key=pending.content_key,
                spec_id=pending.spec_id,
                kind=pending.kind,
                cert_codes=pending.signature,
                operator=pending.operator,
                params=pending.params,
                targets=pending.targets,
                outcome="pass" if verdict == "pass" else ("fail" if verdict == "fail" else "error"),
                rounds_used=pending.rounds_used,
            )
            written += 1
        return written

    def _remember(
        self, spec: DesignSpec, signature: str, plan: RepairPlan, history: Sequence[Round]
    ) -> None:
        """Stash the plan's provenance so `flush` can credit or blame it."""
        self._pending[str(plan.id)] = _Pending(
            content_key=spec.content_key(),
            spec_id=spec.spec_id,
            kind=spec.interface.kind,
            signature=signature,
            operator=str(plan.operator),
            params=dict(plan.params),
            targets=tuple(str(item) for item in plan.targets),
            rounds_used=len(history) + 1,
        )

    # -- replay ------------------------------------------------------------
    def _replay(
        self, spec: DesignSpec, signature: str, history: Sequence[Round]
    ) -> RepairPlan | None:
        """The highest-scoring remembered action that is legal *here*."""
        tried = {
            (str(round_.plan.operator), _signal(round_.plan.params)) for round_ in history
        }
        for record in self.memory.candidates(
            kind=spec.interface.kind, cert_codes=signature
        ):
            if record.operator in _PART_ADDRESSED:
                continue
            params = _adapt_params(record.params, spec)
            if params is None:
                continue
            if (record.operator, _signal(params)) in tried:
                continue
            plan = self._plan(record, params, spec)
            if plan is None:
                continue
            if (str(plan.operator), _signal(plan.params)) in tried:
                continue
            self.last_record = record
            return plan
        return None

    def _plan(
        self, record: CaseRecord, params: dict, spec: DesignSpec
    ) -> RepairPlan | None:
        """The `RepairPlan` for a replayed payload, validated by the operator's model."""
        targets: tuple[str, ...] = (spec.interface.a, spec.interface.b)
        if self.registry is not None:
            try:
                operator = self.registry.get(record.operator)
                params = dict(operator.params.model_validate(params).model_dump())
                targets = tuple(str(item) for item in operator.targets(params))
            except Exception:  # noqa: BLE001 - an operator that cannot validate this is skipped
                return None
        return RepairPlan(
            id=key("plan", record.operator, targets, _signal(params)),
            operator=record.operator,
            targets=targets,
            params=params,
            cost_est=Cost(geom_calls=self.geom_calls_per_round),
            proposed_by="case",
            reason=(
                f"case memory: {record.operator} succeeded {record.passes}/{record.decided} "
                f"on {record.passes + (record.decided - record.passes)} seen "
                f"{'attempt' if record.decided == 1 else 'attempts'}"
            ),
        )

    def _delegate(
        self,
        assembly: Assembly,
        certs: list[FailureCertificate],
        history: list[Round],
        budget_left: Budget,
    ) -> RepairPlan | None:
        if self.inner is None:
            return None
        return self.inner.propose(assembly, certs, history, budget_left)


# --------------------------------------------------------------------------- #
# Parameter adaptation: the knobs travel, the addresses are rebuilt
# --------------------------------------------------------------------------- #
def _delta_cap_mm(spec: DesignSpec) -> float:
    """A feature cannot be resized by more than half the smallest part dimension."""
    return 0.5 * min(min(float(value) for value in part.size_mm) for part in spec.parts)


def _adapt_params(params: Mapping[str, object], spec: DesignSpec) -> dict | None:
    """Remembered knobs -> a payload for **this** spec, or `None` if it cannot be bound.

    Every branch is a decision, and the fall-through is the important one: a knob this
    function has no bound for is refused. Replaying `{"foo_mm": 3}` because it once
    worked is how a policy smuggles a number out of one design into another.
    """
    out: dict[str, object] = {}
    for name, value in params.items():
        if name in _ADDRESS_KEYS:
            continue  # addressing is rebuilt below from the current spec
        if name == "clearance_mm":
            low = float(spec.constraints.min_clearance_mm)
            high = float(spec.constraints.max_clearance_mm)
            try:
                target = float(value)
            except (TypeError, ValueError):
                return None
            if not (low - 1e-9 <= target <= high + 1e-9):
                return None  # outside the band the *current* task asks for
            out["clearance_mm"] = target
        elif name == "delta_mm":
            cap = _delta_cap_mm(spec)
            if isinstance(value, (list, tuple)):
                values = [float(item) for item in value]
                if any(abs(item) > cap for item in values):
                    return None
                out["delta_mm"] = values
            else:
                scalar = float(value)
                if abs(scalar) > cap:
                    return None
                out["delta_mm"] = scalar
        elif name == "delta_deg":
            try:
                values = [float(item) for item in value]  # type: ignore[union-attr]
            except (TypeError, ValueError):
                return None
            if any(abs(item) > 90.0 for item in values):
                return None
            out["delta_deg"] = values
        elif name == "kind":
            if str(value) not in BUILTIN_PARAMS_KINDS:
                return None  # a kind this build does not know how to build
            out["kind"] = str(value)
        elif name == "params":
            if not isinstance(value, Mapping):
                return None
            out["params"] = {str(name_): item for name_, item in value.items()}
        else:
            return None  # no declared bound: never replayed
    if "params" in out:
        # A connector-parameter patch has to still describe a buildable interface on
        # *this* design, so it is validated against the kind's own frozen model.
        target_kind = str(out.get("kind") or spec.interface.kind)
        model = builtin_params_model(target_kind)
        if model is None:
            return None
        base = dict(spec.interface.params) if target_kind == spec.interface.kind else {}
        try:
            model.model_validate({**base, **out["params"]})
        except Exception:  # noqa: BLE001 - an unbuildable patch is refused, not replayed
            return None
    out["interface_id"] = spec.interface.id
    out["parts"] = (spec.interface.a, spec.interface.b)
    return out


# --------------------------------------------------------------------------- #
# Plugin entry point
# --------------------------------------------------------------------------- #
def build_case_memory_policy(
    *,
    inner: Any = None,
    memory: CaseMemory | None = None,
    memory_path: str | Path | None = None,
    registry: Any = None,
    **ignored: object,
) -> CaseMemoryPolicy:
    """The factory `host.policies["case_memory"]` points at.

    `pah.plugins` calls a policy factory as `factory(**kwargs)`; everything this
    factory does not recognise is ignored on purpose, so a runtime that passes
    `root=`/`seed=`/`episode=` does not have to know this module's signature. Pass
    `memory_path` (or `memory`) to choose the ledger; the default is
    `runs/case_memory.sqlite`, overridable with `$PAH_CASE_MEMORY`.
    """
    if memory is None:
        import os  # noqa: PLC0415 - only needed on this path

        target = memory_path or os.environ.get("PAH_CASE_MEMORY") or DEFAULT_MEMORY_PATH
        memory = CaseMemory(target)
    return CaseMemoryPolicy(
        inner=inner if inner is not None else ignored.get("inner"),
        memory=memory,
        registry=registry,
    )


def register(host: Any) -> None:
    """`pah.plugins` entry point: publish one policy factory, and nothing else.

    The inner policy is built against the host's live operator registry, so a replayed
    plan is validated by the same operators a rule-based plan would use. That is the
    whole registration: one dict entry. No core module is imported for side effects,
    no registry is mutated outside `host.policies`, and `kind = "policy"` above is the
    declaration `pah.plugins` checks this against.
    """
    operators = getattr(host, "operators", None)
    host.policies[PLUGIN_NAME] = _factory_for(inner=_default_inner(operators), registry=operators)


def _factory_for(*, inner: Any, registry: Any):
    """A `factory(**kwargs)` that keeps the registry-bound inner policy."""

    def factory(**kwargs: object) -> CaseMemoryPolicy:
        kwargs.setdefault("inner", inner)
        kwargs.setdefault("registry", registry)
        return build_case_memory_policy(**kwargs)

    return factory


def _default_inner(operators: Any) -> Any:
    """The deterministic policy this one delegates to, best available in this build.

    The design plane's own rule policy is preferred because its ladders are written
    against design-plane codes; `policy.rule.RulePolicy` (the M1 policy) is the
    fallback for a tree where the design policies have not landed. `None` means the
    plugin still loads and simply proposes nothing on a miss — a policy that cannot
    delegate is a weaker policy, not a broken import.
    """
    for module_name, attribute in (
        ("pah.policy.design_policies", "RulePolicy"),
        ("pah.policy.rule", "RulePolicy"),
    ):
        try:
            module = __import__(module_name, fromlist=[attribute])
            policy_class = getattr(module, attribute)
            return policy_class(registry=operators)
        except Exception:  # noqa: BLE001 - a missing policy is a fallback, not a crash
            continue
    return None
