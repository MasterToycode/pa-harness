"""Provider registry: one `kind` covers every OpenAI-compatible service.

Most services a project like this talks to — the domestic APIs, coding and token
plans, OpenRouter, a local vLLM or llama-server — speak the same wire protocol.
So there is one adapter for all of them and the differences live in
configuration, not in code.

Two rules separate this from a product harness:

- **No silent fallback.** A run whose two calls went to different providers is
  not one experiment. `select` refuses; a caller that wants a fallback asks for
  it and the choice is recorded in the manifest.
- **Billing is declared, never inferred.** A token plan and a subscription have
  no per-token price, so cost is always recorded as tokens plus a billing unit.
  Comparing runs across providers compares tokens and call counts, not money.
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
    "ProviderConfig",
    "ProviderError",
    "Registry",
    "ReplayLLM",
    "Transport",
    "Usage",
    "load_registry",
    "resolve_python_target",
]

#: How a provider charges. `per_token` is the only one with a price; the rest
#: consume a quota, a subscription or nothing.
Billing = Literal["per_token", "quota", "subscription", "free", "gpu_seconds", "per_call"]

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


def resolve_python_target(config: ProviderConfig) -> Callable[..., Any]:
    """The callable an `[gen.providers.<id>] kind="python"` entry names.

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
    module_name, _, attribute = raw.partition(":")
    if not attribute:  # accept "pkg.mod.func" by splitting on the last dot
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
