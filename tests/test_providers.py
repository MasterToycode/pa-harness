"""Contract tests for the provider registry.

The rules worth locking down are the ones that keep a run honest: no silent
fallback, keys read by variable name at call time, and money that only exists
for per-token billing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pah.app.providers import (
    NullLLM,
    OpenAICompatible,
    ProviderConfig,
    ProviderError,
    ReplayLLM,
    Usage,
    build_client,
    load_registry,
)

TOML = """
[llm]
default = "local"

[llm.providers.local]
kind = "openai-compatible"
base_url = "http://127.0.0.1:8000/v1"
model = "local-model"
billing = "free"

[llm.providers.volcengine]
kind = "openai-compatible"
base_url = "https://ark.example.com/api/coding/v3/"
api_key_env = "VOLC_KEY"
model = "doubao-seed-2.0-pro"
billing = "quota"
compat = { thinking_format = "deepseek", max_tokens = 4096 }

[llm.providers.paid]
kind = "openai-compatible"
base_url = "https://api.example.com/v1"
api_key_env = "PAID_KEY"
model = "paid-model"
billing = "per_token"
pricing = { in_usd_per_mtok = 1.0, out_usd_per_mtok = 2.0 }

[gen]
default = "param"

[gen.providers.param]
kind = "local-parametric"
"""


class Transport:
    """Records the request and answers a canned completion."""

    def __init__(self, answer: dict | None = None) -> None:
        self.url = ""
        self.headers: dict[str, str] = {}
        self.body: dict = {}
        self.calls = 0
        self._answer = answer or {
            "model": "answered-by",
            "choices": [{"message": {"content": '{"operator":"resize_interface"}'}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20},
        }

    def __call__(self, url: str, headers: dict[str, str], body: bytes) -> dict:
        self.calls += 1
        self.url = url
        self.headers = headers
        self.body = json.loads(body)
        return self._answer


@pytest.fixture()
def registry(tmp_path: Path):
    path = tmp_path / "pah.toml"
    path.write_text(TOML, "utf-8")
    return load_registry(path)


def test_missing_file_is_an_empty_registry(tmp_path: Path):
    reg = load_registry(tmp_path / "nope.toml")
    assert reg.providers == {}
    assert reg.path is None


def test_parses_providers_and_defaults(registry):
    assert set(registry.providers) == {"local", "volcengine", "paid", "param"}
    assert registry.defaults["llm"] == "local"
    assert registry.providers["volcengine"].billing == "quota"
    assert registry.providers["volcengine"].compat["thinking_format"] == "deepseek"


def test_unknown_kind_is_refused(tmp_path: Path):
    path = tmp_path / "bad.toml"
    path.write_text('[llm.providers.x]\nkind = "magic"\n', "utf-8")
    with pytest.raises(ProviderError, match="is not one of"):
        load_registry(path)


def test_choose_uses_the_default(registry):
    assert registry.choose("llm").id == "local"


def test_choose_pins_an_explicit_provider(registry):
    assert registry.choose("llm", "paid").id == "paid"


def test_choose_does_not_fall_back(registry):
    """A run that silently changes provider is not one experiment."""
    with pytest.raises(ProviderError, match="unknown provider"):
        registry.choose("llm", "not-configured")


def test_openai_compatible_sends_the_declared_shape(registry):
    transport = Transport()
    client = OpenAICompatible(registry.providers["volcengine"], transport=transport, environ={"VOLC_KEY": "k"})
    client.complete(system="s", user="u", json_schema={"type": "object"}, seed=7)

    assert transport.url == "https://ark.example.com/api/coding/v3/chat/completions"
    assert transport.headers["Authorization"] == "Bearer k"
    assert transport.body["model"] == "doubao-seed-2.0-pro"
    assert transport.body["seed"] == 7
    assert transport.body["response_format"] == {"type": "json_object"}
    assert transport.body["thinking_format"] == "deepseek"
    assert transport.body["max_tokens"] == 4096  # compat wins over the call default
    assert transport.body["messages"][0] == {"role": "system", "content": "s"}


def test_loopback_needs_no_key(registry):
    transport = Transport()
    OpenAICompatible(registry.providers["local"], transport=transport, environ={}).complete(
        system="s", user="u"
    )
    assert "Authorization" not in transport.headers
    assert transport.url == "http://127.0.0.1:8000/v1/chat/completions"


def test_missing_key_fails_before_the_request(registry):
    transport = Transport()
    with pytest.raises(ProviderError, match=r"\$VOLC_KEY"):
        OpenAICompatible(registry.providers["volcengine"], transport=transport, environ={}).complete(
            system="s", user="u"
        )
    assert transport.calls == 0


def test_per_token_is_priced(registry):
    transport = Transport()
    result = OpenAICompatible(registry.providers["paid"], transport=transport, environ={"PAID_KEY": "k"}).complete(
        system="s", user="u"
    )
    assert result.usage.billing == "per_token"
    assert result.usage.tokens == 120
    # 100*1.0 + 20*2.0 micro-dollars per million
    assert result.usage.cost == pytest.approx(140 / 1_000_000)


def test_quota_is_not_priced(registry):
    """A quota plan has no per-token price; inventing one would corrupt any
    cross-provider comparison."""
    transport = Transport()
    result = OpenAICompatible(
        registry.providers["volcengine"], transport=transport, environ={"VOLC_KEY": "k"}
    ).complete(system="s", user="u")
    assert result.usage.billing == "quota"
    assert result.usage.cost == 0.0
    assert result.usage.tokens == 120


def test_response_without_content_is_refused(registry):
    transport = Transport({"choices": []})
    with pytest.raises(ProviderError, match="no message content"):
        OpenAICompatible(registry.providers["local"], transport=transport, environ={}).complete(
            system="s", user="u"
        )


def test_null_client_refuses_instead_of_inventing():
    with pytest.raises(ProviderError, match="no LLM provider configured"):
        NullLLM().complete(system="s", user="u")


def test_build_client_dispatch(registry):
    assert isinstance(build_client(registry.providers["local"]), OpenAICompatible)
    assert isinstance(build_client(registry.providers["param"]), NullLLM)


def test_codex_oauth_says_what_is_missing():
    with pytest.raises(ProviderError, match="codex-oauth is not wired yet"):
        build_client(ProviderConfig(id="codex", kind="codex-oauth"))


def test_replay_returns_recorded_calls_in_order(tmp_path: Path):
    path = tmp_path / "calls.jsonl"
    rows = [
        {"text": "first", "model": "m", "usage": {"prompt_tokens": 5, "completion_tokens": 1}},
        {"text": "second", "model": "m", "usage": {}},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows), "utf-8")
    client = ReplayLLM(path)

    assert client.complete(system="s", user="u").text == "first"
    second = client.complete(system="s", user="u2")
    assert second.text == "second"
    assert second.usage.tokens == 0
    assert len(client.calls) == 2


def test_replay_refuses_to_run_past_its_recording(tmp_path: Path):
    path = tmp_path / "one.jsonl"
    path.write_text(json.dumps({"text": "only"}), "utf-8")
    client = ReplayLLM(path)
    client.complete(system="s", user="u")
    with pytest.raises(ProviderError, match="replay exhausted"):
        client.complete(system="s", user="u")


def test_usage_tokens_add_up():
    assert Usage(prompt_tokens=3, completion_tokens=4).tokens == 7
