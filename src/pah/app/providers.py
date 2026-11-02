"""Provider registry: one `kind` covers every OpenAI-compatible service.

Most services a project like this talks to — the domestic APIs, coding and token
plans, OpenRouter, a local vLLM or llama-server — speak the same wire protocol.
So there is one adapter for all of them and the differences live in
configuration, not in code.

Three rules separate this from a product harness:

- **No silent fallback.** A run whose two calls went to different providers is
  not one experiment. `select` refuses; a caller that wants a fallback asks for
  it and the choice is recorded in the manifest.
- **Billing is declared, never inferred.** A token plan and a subscription have
  no per-token price, so cost is always recorded as tokens plus a billing unit.
  Comparing runs across providers compares tokens and call counts, not money.
- **Generation that can spend money is opt-in, once.** A provider that declares a
  billing unit, or whose adapter is one of the vendor adapters under `examples/`,
  refuses to resolve unless this process carries `PAH_ALLOW_PAID_GEN` — the flag a
  human types. Configuration alone must never be able to spend money: on 2026-11-02
  an audit script that only meant to read a report called a cloud generator, because
  `pah.toml` was enough to make it so.
"""

from __future__ import annotations

import json
import os
import tomllib
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Protocol

__all__ = [
    "Billing",
    "LLMClient",
    "LLMResult",
    "LocalParametric",
    "NullLLM",
    "OpenAICompatible",
    "PAID_GENERATION_VAR",
    "PaidGenerationRefused",
    "ProviderConfig",
    "ProviderError",
    "Registry",
    "ReplayLLM",
    "Transport",
    "Usage",
    "chosen_gen_provider",
    "generation_is_paid",
    "load_python_target",
    "load_registry",
    "paid_generation_allowed",
    "paid_generation_record",
    "require_paid_generation",
    "resolve_generator",
    "resolve_python_target",
]

#: How a provider charges. `per_token` is the only one with a price; the rest
#: consume a quota, a subscription or nothing.
Billing = Literal["per_token", "quota", "subscription", "free", "gpu_seconds", "per_call"]

#: The one environment variable that authorises a paid generation for this process.
#: A CLI flag sets it; nothing in configuration can, so a copied `pah.toml` — or a
#: script that thought it was only reading files — cannot spend money by accident.
PAID_GENERATION_VAR = "PAH_ALLOW_PAID_GEN"

#: The spellings that open the gate. Everything else, including a typo, leaves it
#: closed: refusing a run costs one flag, an unintended generation costs money.
_ALLOW_PAID_VALUES = frozenset({"1", "true"})

#: Where the vendor-backed adapters live (`hunyuan3d_materializer.py`,
#: `xpart_decomposer.py`). They sit outside the package on purpose — `pah decompose`
#: loads one by path — so a `compat.target` under this directory is itself the
#: declaration that executing it calls a service which bills.
_EXAMPLES_DIR = Path(__file__).resolve().parents[3] / "examples"

_KINDS = (
    "openai-compatible",
    "codex-oauth",
    "local-parametric",
    "http-repair",
    # A local Python callable, imported by name. This is the door a DiT adapter
    # walks through: `[gen.providers.dit] kind = "python"` plus
    # `compat = { target = "my_module:materialize" }`. Kept in the same registry as
    # the HTTP providers so a run's manifest can name *which* generator produced a
    # part, and so "no silent fallback" applies to geometry exactly as it does to
    # decisions.
    "python",
)

#: `(url, headers, body_bytes) -> parsed JSON dict`. Injected so tests never
#: reach the network and a caller can swap in a recording transport.
Transport = Callable[[str, dict[str, str], bytes], dict[str, Any]]


class ProviderError(RuntimeError):
    """A provider is missing, misconfigured, or refused the call."""


class PaidGenerationRefused(ProviderError):
    """A paid generator was resolved without this process being told to pay for it.

    Subclasses `ProviderError` because every caller that already turns a provider
    refusal into one sentence and exit 3 — the composition root, `pah run`, the
    doctor — must do the same here without learning a second error class. `provider`
    and `billing` carry the facts a report or a log needs when the refusal is shown.
    """

    def __init__(self, message: str, *, provider: str = "", billing: str = "") -> None:
        super().__init__(message)
        self.provider = provider
        self.billing = billing


@dataclass(frozen=True)
class Usage:
    """What one call cost. `billing` names the unit `cost` is counted in, so a
    quota-based plan never has to pretend it has a price."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    billing: Billing = "free"
    cost: float = 0.0

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True)
class LLMResult:
    text: str
    model: str
    usage: Usage
    provider: str
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProviderConfig:
    id: str
    kind: str
    base_url: str = ""
    api_key_env: str = ""
    model: str = ""
    billing: Billing = "free"
    pricing: dict[str, float] = field(default_factory=dict)
    compat: dict[str, Any] = field(default_factory=dict)
    credentials_file: str = ""


@dataclass(frozen=True)
class Registry:
    """The parsed `pah.toml`: every provider, and which one `role` uses."""

    providers: dict[str, ProviderConfig]
    defaults: dict[str, str]
    path: Path | None = None

    def choose(self, role: str, provider_id: str | None = None) -> ProviderConfig:
        """The provider `role` must use, or a refusal.

        There is no fallback: a run that silently changes provider is not one
        experiment. Pass `provider_id` to pin it explicitly.
        """
        chosen = provider_id or self.defaults.get(role)
        if not chosen:
            raise ProviderError(f"no default provider configured for {role!r}")
        if chosen not in self.providers:
            known = ", ".join(sorted(self.providers)) or "none"
            raise ProviderError(f"unknown provider {chosen!r} for {role!r}; configured: {known}")
        return self.providers[chosen]


def load_registry(path: str | Path) -> Registry:
    """Reads `pah.toml`. A missing file yields an empty registry, which is valid:
    M0–M2 run with no external service at all."""
    path = Path(path)
    if not path.exists():
        return Registry(providers={}, defaults={}, path=None)
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    providers: dict[str, ProviderConfig] = {}
    for role in ("llm", "gen"):
        section = data.get(role, {})
        for provider_id, raw in section.get("providers", {}).items():
            kind = raw.get("kind", "")
            if kind not in _KINDS:
                raise ProviderError(
                    f"{role}.providers.{provider_id}: kind {kind!r} is not one of {', '.join(_KINDS)}"
                )
            providers[provider_id] = ProviderConfig(
                id=provider_id,
                kind=kind,
                base_url=raw.get("base_url", ""),
                api_key_env=raw.get("api_key_env", ""),
                model=raw.get("model", ""),
                billing=raw.get("billing", "free"),
                pricing=raw.get("pricing", {}),
                compat=raw.get("compat", {}),
                credentials_file=raw.get("credentials_file", ""),
            )
    defaults = {role: data.get(role, {}).get("default", "") for role in ("llm", "gen")}
    return Registry(providers=providers, defaults=defaults, path=path)


# --------------------------------------------------------------------------
# the paid-generation gate
# --------------------------------------------------------------------------


def chosen_gen_provider(config_path: str | Path | None = None) -> ProviderConfig | None:
    """The provider `[gen] default` names, or `None` when nothing is configured.

    `None` covers both "no default" and "the default names a provider this file does
    not define": neither can generate anything, so neither is a paid decision.
    """
    from pah.plugins import resolve_config_path  # noqa: PLC0415 - plugin layer is optional

    path = resolve_config_path(config_path)
    registry = load_registry(str(path) if path is not None else "pah.toml")
    try:
        return registry.choose("gen")
    except ProviderError:
        return None


def paid_generation_allowed(environ: dict[str, str] | None = None) -> bool:
    """Whether this process was told, explicitly, that a paid call may happen."""
    environment = environ if environ is not None else os.environ
    return str(environment.get(PAID_GENERATION_VAR, "")).strip().lower() in _ALLOW_PAID_VALUES


def generation_is_paid(config: ProviderConfig) -> bool:
    """Whether using `config` as the generator can spend money.

    Read from what the provider *declares*, never from its id, so a future
    `hunyuan2` (or a local adapter borrowing the name) is classified by its own
    config rather than by a list someone has to remember to update:

    * `kind = "local-parametric"` is the shipped generator: free by construction;
    * a `billing` unit other than unset/`free` is a declared cost;
    * a `compat.target` that is a file under this checkout's `examples/` is the
      vendor-adapter convention — those adapters cannot run without a paid service.
    """
    if config.kind == "local-parametric":
        return False
    if _declared_billing(config):
        return True
    return _targets_examples_adapter(str((config.compat or {}).get("target", "") or ""))


def _declared_billing(config: ProviderConfig) -> str:
    """The billing unit this provider *declares*, or `""` for the default.

    `ProviderConfig.billing` defaults to `"free"`, so "declares nothing" and
    "declares free" read the same there. Treating the default as no declaration is
    what keeps the refusal honest: without it, a vendor adapter that never wrote a
    billing line would be described as `billing=free` while it is charging money.
    """
    billing = str(config.billing or "").strip()
    return billing if billing and billing != "free" else ""


def _targets_examples_adapter(target: str) -> bool:
    """`compat.target` names a file in this checkout's `examples/`, in either spelling.

    The checkout-relative spelling the shipped `pah.toml` uses
    (`examples/hunyuan3d_materializer.py:materialize`) is matched as written, so the
    gate does not depend on the current directory; an absolute spelling is matched
    after resolving it. The split is at the **last** colon: a Windows drive letter is
    one too, and classifying a paid adapter as free is the one mistake that costs money.
    """
    module_name = target.rpartition(":")[0].strip() or target.strip()
    if not _looks_like_a_path(module_name):
        return False
    if module_name.replace("\\", "/").startswith("examples/"):
        return True
    path = Path(module_name).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    try:
        return path.resolve().is_relative_to(_EXAMPLES_DIR)
    except (OSError, ValueError):
        return False


def require_paid_generation(config: ProviderConfig, *, action: str = "") -> None:
    """Refuse a paid generator unless this process was explicitly authorised.

    The sentence states what did *not* happen — no call, no charge — because after an
    incident like the one this gate exists for, the first question is whether the
    refusal already cost something. It names both ways to authorise: the flag for the
    command being run, and the variable for a wrapper that has no flag (`pah serve`,
    a shell script).
    """
    if not generation_is_paid(config) or paid_generation_allowed():
        return
    provider = f"{config.id}:{config.kind}"
    declared = _declared_billing(config)
    if declared:
        reason, detail = "声明了计费单位", f"billing={declared}"
    else:
        reason, detail = "是 examples/ 下的 vendor 适配器", "billing undeclared"
    where = f"（{action}）" if action else ""
    raise PaidGenerationRefused(
        f"付费生成闸已关闭{where}：生成器 {provider!r} {reason}（{detail}）。"
        "这次不会调用任何生成服务，也不会产生费用；"
        "要真的生成请加 --allow-paid-gen（或设置 PAH_ALLOW_PAID_GEN=1）；"
        "只想跑离线流程，就把 pah.toml 里的 [gen] default 换成 local。",
        provider=provider,
        billing=declared or detail,
    )


def paid_generation_record(
    config_path: str | Path | None = None, *, action: str = ""
) -> dict[str, Any] | None:
    """The report fields an authorised paid generator requires, or `None` for free.

    Recording and permitting are one decision on purpose: a report that carries
    `paid_generation: true` cannot exist unless the gate was opened, and an opened
    gate cannot leave the report silent about who was (or could be) charged. A paid
    provider the run cannot execute (`kind` is not `python`) still closes the gate
    but is not recorded, because no money can move through it.
    """
    config = chosen_gen_provider(config_path)
    if config is None or not generation_is_paid(config):
        return None
    require_paid_generation(config, action=action)
    if config.kind != "python":
        return None
    return {
        "paid_generation": True,
        "provider": f"{config.id}:{config.kind}",
        "billing": _declared_billing(config) or "undeclared (vendor adapter)",
    }


# --------------------------------------------------------------------------
# clients
# --------------------------------------------------------------------------


class LLMClient(Protocol):
    """One turn of judgement. Stateless by design: a repair decision needs no
    conversation, and stateless calls are cheap to replay and to audit."""

    provider: str

    def complete(
        self,
        *,
        system: str,
        user: str,
        json_schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
        seed: int | None = None,
        max_tokens: int = 2048,
    ) -> LLMResult: ...


class NullLLM:
    """The default when nothing is configured. It refuses rather than inventing
    an answer, so a run that reaches the model layer by accident fails loudly."""

    provider = "null"

    def complete(self, **_kwargs: Any) -> LLMResult:
        raise ProviderError(
            "this run has no LLM provider configured; the decision fell through to "
            "the model layer, which means the rule layer found no candidate"
        )


class LocalParametric:
    """The generator used by M0–M2: no network, no model, just the operator
    registry. It exists so the checkpoint `gen=param` is explicit in a manifest."""

    provider = "local-parametric"

    def repair(self, **_kwargs: Any) -> None:
        raise ProviderError("LocalParametric is driven by the operator registry, not called directly")


def _http_transport(url: str, headers: dict[str, str], body: bytes) -> dict[str, Any]:
    """The one place this package opens a socket."""
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")[:800]
        raise ProviderError(f"{url} answered HTTP {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise ProviderError(f"{url} is unreachable: {error.reason}") from error


class OpenAICompatible:
    """Every `openai-compatible` provider: domestic APIs, coding and token plans,
    OpenRouter, a local vLLM or llama-server. Configuration carries the
    differences; this class carries none."""

    def __init__(
        self,
        config: ProviderConfig,
        *,
        transport: Transport = _http_transport,
        environ: dict[str, str] | None = None,
    ) -> None:
        self.config = config
        self.provider = config.id
        self._transport = transport
        self._environ = environ if environ is not None else os.environ

    def _key(self) -> str:
        """Reads the key by variable name at call time, so a rotated key needs no
        restart and a missing key fails before the request is built."""
        if not self.config.api_key_env:
            return ""  # loopback services such as vLLM take no key
        key = self._environ.get(self.config.api_key_env, "")
        if not key:
            raise ProviderError(
                f"provider {self.provider!r} needs ${self.config.api_key_env}, which is unset"
            )
        return key

    def complete(
        self,
        *,
        system: str,
        user: str,
        json_schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
        seed: int | None = None,
        max_tokens: int = 2048,
    ) -> LLMResult:
        body: dict[str, Any] = {
            "model": self.config.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": temperature,
            "max_tokens": int(self.config.compat.get("max_tokens", max_tokens)),
        }
        if seed is not None:
            body["seed"] = seed
        if json_schema is not None:
            body["response_format"] = {"type": "json_object"}
        for name in ("top_p", "thinking_format", "reasoning"):
            if name in self.config.compat:
                body[name] = self.config.compat[name]

        headers = {"Content-Type": "application/json"}
        key = self._key()
        if key:
            headers["Authorization"] = f"Bearer {key}"

        url = self.config.base_url.rstrip("/") + "/chat/completions"
        raw = self._transport(url, headers, json.dumps(body).encode("utf-8"))
        return self._parse(raw)

    def _parse(self, raw: dict[str, Any]) -> LLMResult:
        try:
            text = raw["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as error:
            raise ProviderError(f"{self.provider}: response has no message content") from error
        usage_raw = raw.get("usage") or {}
        prompt = int(usage_raw.get("prompt_tokens", 0))
        completion = int(usage_raw.get("completion_tokens", 0))
        return LLMResult(
            text=text,
            model=raw.get("model", self.config.model),
            provider=self.provider,
            usage=Usage(
                prompt_tokens=prompt,
                completion_tokens=completion,
                billing=self.config.billing,
                cost=self._cost(prompt, completion),
            ),
            raw=raw,
        )

    def _cost(self, prompt: int, completion: int) -> float:
        """Money only exists for `per_token`; every other billing unit reports 0.0
        and is accounted by tokens and call counts instead."""
        if self.config.billing != "per_token":
            return 0.0
        price_in = float(self.config.pricing.get("in_usd_per_mtok", 0.0))
        price_out = float(self.config.pricing.get("out_usd_per_mtok", 0.0))
        return (prompt * price_in + completion * price_out) / 1_000_000.0


class ReplayLLM:
    """Replays recorded calls in order, so a run reproduces without the network.

    This is what makes "same seed, same result" true for a pipeline that calls a
    model: the recording, not the model, is the reproducibility boundary.
    """

    provider = "replay"

    def __init__(self, path: str | Path) -> None:
        self._rows = [json.loads(line) for line in Path(path).read_text("utf-8").splitlines() if line.strip()]
        self._index = 0
        self.calls: list[dict[str, Any]] = []

    def complete(self, **kwargs: Any) -> LLMResult:
        if self._index >= len(self._rows):
            raise ProviderError(
                f"replay exhausted after {len(self._rows)} calls; the run asked for another"
            )
        row = self._rows[self._index]
        self.calls.append({"request": kwargs, "response": row})
        self._index += 1
        usage = row.get("usage", {})
        return LLMResult(
            text=row["text"],
            model=row.get("model", "replay"),
            provider="replay",
            usage=Usage(
                prompt_tokens=int(usage.get("prompt_tokens", 0)),
                completion_tokens=int(usage.get("completion_tokens", 0)),
                billing="free",
            ),
        )


def build_client(
    config: ProviderConfig,
    *,
    transport: Transport = _http_transport,
    environ: dict[str, str] | None = None,
) -> LLMClient:
    """One factory per `kind`, so a caller never branches on the provider name."""
    if config.kind == "openai-compatible":
        return OpenAICompatible(config, transport=transport, environ=environ)
    if config.kind == "local-parametric":
        return NullLLM()
    if config.kind == "python":
        raise ProviderError(
            f"provider {config.id!r} is a python generator, not a chat client; "
            "resolve it with `resolve_python_target` instead"
        )
    if config.kind == "codex-oauth":
        raise ProviderError(
            "codex-oauth is not wired yet: it reuses the subscription quota DSH keeps in "
            "credentials_file. Add it when the decision layer starts using a model."
        )
    raise ProviderError(f"kind {config.kind!r} has no client")


def resolve_generator(config_path: str | Path | None = None) -> Callable[..., Any] | None:
    """The python generator `[gen]` configures, gated; `None` when it configures none.

    `None` is not a fallback: the composition root decides what "no configured
    generator" means (T1's parametric materialiser, in `pah.app.runtime`). This
    function owns the *decision to spend*, which is why it lives beside the provider
    registry and not in the composition root — and both spellings end in
    `resolve_python_target`, so no caller can resolve a vendor adapter past the gate.
    """
    config = chosen_gen_provider(config_path)
    if config is None or config.kind != "python":
        return None
    return resolve_python_target(config)


def resolve_python_target(config: ProviderConfig) -> Callable[..., Any]:
    """The `Materializer` port's callable, refused when it is paid and unauthorised.

    The gate sits here because this is the one function every composition root
    already calls to turn `[gen]` into geometry: resolving a vendor adapter is the
    moment a run decides to spend, so no caller has to remember to re-check.
    """
    require_paid_generation(config)
    return load_python_target(config)


def load_python_target(config: ProviderConfig) -> Callable[..., Any]:
    """Import the callable an `[gen.providers.<id>] kind="python"` entry names.

    This is the import half without the payment gate: `pah providers doctor` may
    import an adapter for free — importing is not calling — and reports the gate's
    state beside it. A run must enter through `resolve_python_target` instead.

    `compat = { target = "package.module:function" }`. Two spellings are supported,
    because a first integration and a released adapter are different situations:

    * **an importable module** (`"my_dit.materialize:materialize"`, or dotted as
      `"my_dit.materialize.materialize"`) — what a package you installed should use;
    * **a file path** (`"examples/dit_materializer.py:stub_materialize"`) — what lets
      you test an adapter from a checkout without packaging it, which is the whole
      point of a first test.

    Failures each get their own sentence, because this is the seam where a user's
    adapter either works or does not:

    * no target declared — a configuration mistake;
    * the module/file will not import — usually a missing dependency of the adapter;
    * the attribute is missing or not callable — usually a model object where a
      function was expected.

    Deliberately **not** a fallback: if a run asked for a DiT generator and the
    adapter cannot load, the run must stop rather than quietly produce trimesh parts
    while the report says a DiT generated them.
    """
    raw = str((config.compat or {}).get("target", "") or "").strip()
    if not raw:
        raise ProviderError(
            f"gen provider {config.id!r} has kind 'python' but declares no "
            "`compat.target`; write it as `compat = { target = \"module:function\" }`"
        )
    # Split on the **last** colon: a Windows drive letter is a colon too, and
    # `D:\checkout\examples\adapter.py:materialize` split on the first one yields the
    # module `D` and an attribute that is a path — a config that can never resolve,
    # which the P0-4 review found while adding the payment gate. No colon at all means
    # the dotted spelling `pkg.module.function`.
    if ":" in raw:
        module_name, _, attribute = raw.rpartition(":")
        if "/" in attribute or "\\" in attribute:
            # The last colon was a drive letter, not the separator: keep the first split
            # so the error below names the target the user actually wrote.
            module_name, _, attribute = raw.partition(":")
    else:
        module_name, _, attribute = raw.rpartition(".")
    if not module_name or not attribute:
        raise ProviderError(
            f"gen provider {config.id!r} target {raw!r} is not 'module:function' or "
            "'path/to/file.py:function'"
        )

    obj = (
        _from_file(module_name, attribute, config)
        if _looks_like_a_path(module_name)
        else _from_module(module_name, attribute, config)
    )
    if not callable(obj):
        raise ProviderError(
            f"gen provider {config.id!r}: target {raw!r} is "
            f"{type(obj).__name__}, not a callable `(spec, out_dir) -> MaterializedDesign`"
        )
    return obj


def _looks_like_a_path(module: str) -> bool:
    """A path, not a module name: `.py` suffix or a directory separator."""
    return module.endswith(".py") or "/" in module or "\\" in module


def _from_module(module_name: str, attribute: str, config: ProviderConfig) -> Any:
    from importlib import import_module  # noqa: PLC0415

    try:
        module = import_module(module_name)
    except Exception as error:  # noqa: BLE001 - the sentence is the product here
        raise ProviderError(
            f"gen provider {config.id!r}: cannot import {module_name!r}: "
            f"{type(error).__name__}: {error}. Put the adapter on `sys.path` (install it "
            "or set PYTHONPATH), or point `compat.target` at a file path instead."
        ) from error
    obj = getattr(module, attribute, None)
    if obj is None:
        raise ProviderError(
            f"gen provider {config.id!r}: module {module_name!r} has no attribute "
            f"{attribute!r}"
        )
    return obj


def _from_file(path_str: str, attribute: str, config: ProviderConfig) -> Any:
    """Load `attribute` from a .py file — the checkout-path spelling."""
    from importlib.util import module_from_spec, spec_from_file_location  # noqa: PLC0415

    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.exists():
        raise ProviderError(
            f"gen provider {config.id!r}: adapter file {path} does not exist "
            "(relative paths are resolved against the current directory)"
        )
    name = f"pah_plugin_{abs(hash(str(path)))}"
    spec = spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ProviderError(f"gen provider {config.id!r}: {path} is not importable as Python")
    module = module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as error:  # noqa: BLE001 - the sentence is the product here
        raise ProviderError(
            f"gen provider {config.id!r}: adapter {path} raised on import: "
            f"{type(error).__name__}: {error}"
        ) from error
    obj = getattr(module, attribute, None)
    if obj is None:
        raise ProviderError(
            f"gen provider {config.id!r}: {path} has no attribute {attribute!r}"
        )
    return obj
