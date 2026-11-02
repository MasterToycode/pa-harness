"""`pah providers doctor` — check the two external services before spending a run.

Why this command exists
-----------------------
A first integration always fails in one of four places, and they look identical
from `pah run`'s output: the config file is not the one you edited, the key
environment variable is not in *this* shell, the model name is wrong for the
endpoint, or the generator adapter imports but is not the callable you think it is.
Diagnosing that through an episode costs a lot of noise for a one-line answer, so
this command prints the five facts that distinguish them, one service at a time:

    pah providers doctor                # config + key/import checks, no network
    pah providers doctor --call         # also spend one tiny call per service
    pah providers doctor --role gen     # only the generator (DiT) side
    pah providers doctor --role gen --call --allow-paid-gen   # call a vendor adapter

`--call` is deliberately explicit: it is the only thing here that can cost money
or load a model. Calling a generator that bills — one whose config declares a
billing unit, or whose adapter is a vendor adapter under `examples/` — additionally
needs `--allow-paid-gen`, the same gate `pah run` uses; without it the check fails
with the reason and makes no request. The default run touches nothing remote and
imports nothing heavy beyond what the check itself needs.

Exit codes follow `cli/main.py`: 0 all checks passed, 1 a check said no, 3 crash.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path

EXIT_OK = 0
EXIT_NO = 1
EXIT_CRASH = 3

#: The fixture `--call` materialises when the caller names no `--spec`. Small and
#: already carries a valid band, so a green check means the adapter works, not that
#: the design happened to be easy.
DEFAULT_SPEC = "fixtures/design/demo-peg-hole-clear.json"


def add_parser(groups: argparse._SubParsersAction) -> None:
    """Mounts `pah providers <command>` onto the root parser."""
    providers = groups.add_parser(
        "providers", help="external service checks (LLM endpoint, generator adapter)"
    )
    commands = providers.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser(
        "doctor",
        help="print how each service is configured, and optionally call it once",
    )
    doctor.add_argument(
        "--role",
        default="all",
        choices=["all", "llm", "gen"],
        help="which side to check (default: all)",
    )
    doctor.add_argument("--config", default=None, help="path to pah.toml")
    doctor.add_argument(
        "--call",
        action="store_true",
        help="actually call each service once (may cost money / load a model)",
    )
    doctor.add_argument(
        "--allow-paid-gen",
        dest="allow_paid_gen",
        action="store_true",
        help="authorise a generator that costs money; required for --call on one",
    )
    doctor.add_argument(
        "--spec", default=DEFAULT_SPEC, help="spec used by the generator self-test"
    )
    doctor.add_argument("--json", action="store_true", help="machine-readable output")
    doctor.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    from pah.app.providers import PAID_GENERATION_VAR  # noqa: PLC0415
    from pah.plugins import PluginError, resolve_config_path  # noqa: PLC0415

    if args.allow_paid_gen:
        # The same flag, and the same one-line authorisation, as `pah run`: `--call`
        # on a vendor adapter spends a generation, so it needs the same yes.
        os.environ[PAID_GENERATION_VAR] = "1"
    try:
        path = resolve_config_path(args.config)
        sections: dict[str, dict] = {}
        failures = 0
        if args.role in ("all", "llm"):
            section, failed = _check_llm(path, call=args.call)
            sections["llm"] = section
            failures += int(failed)
        if args.role in ("all", "gen"):
            section, failed = _check_gen(path, call=args.call, spec_path=args.spec)
            sections["gen"] = section
            failures += int(failed)
    except PluginError as error:
        print(f"pah providers doctor: {error}")
        return EXIT_NO
    except Exception:  # noqa: BLE001 - a crash must be exit 3, not a bare traceback
        traceback.print_exc()
        return EXIT_CRASH

    payload = {
        "config": str(path) if path is not None else "",
        "config_found": path is not None and Path(path).exists(),
        "sections": sections,
        "failures": failures,
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=1))
    else:
        _print(payload)
    return EXIT_OK if failures == 0 else EXIT_NO


# --------------------------------------------------------------------------- #
# the two checks
# --------------------------------------------------------------------------- #
def _check_llm(config_path: Path | None, *, call: bool) -> tuple[dict, int]:
    """Everything that decides whether `--policy llm` can run, plus one real call."""
    from pah.app.providers import ProviderError, build_client, load_registry  # noqa: PLC0415

    section: dict = {"role": "llm", "configured": False, "checks": [], "call": None}
    registry = load_registry(str(config_path) if config_path is not None else "pah.toml")
    try:
        config = registry.choose("llm")
    except ProviderError as error:
        section["checks"].append({"check": "provider configured", "ok": False, "detail": str(error)})
        return section, 1

    key_env = str(getattr(config, "api_key_env", "") or "")
    key_present = (not key_env) or bool(os.environ.get(key_env))
    rows = [
        ("provider configured", True, f"{config.id} (kind={config.kind})"),
        ("base_url", bool(config.base_url), config.base_url or "<empty: loopback client>"),
        ("model", bool(config.model), config.model or "<empty>"),
        ("billing declared", True, config.billing),
        (
            "credential",
            key_present,
            f"${key_env} is set" if key_env and key_present else
            (f"${key_env} is NOT set in this shell" if key_env else "no key needed"),
        ),
    ]
    section.update(
        {
            "configured": True,
            "provider": config.id,
            "kind": config.kind,
            "model": config.model,
            "base_url": config.base_url,
            "billing": config.billing,
            "api_key_env": key_env,
        }
    )
    section["checks"] = [{"check": name, "ok": ok, "detail": detail} for name, ok, detail in rows]
    failed = sum(1 for _, ok, _ in rows if not ok)

    if call and failed == 0:
        section["call"] = _call_llm(build_client(config))
        failed += int(not section["call"]["ok"])
    return section, failed


def _call_llm(client) -> dict:
    """One completion, billed as one call: the cheapest proof the endpoint answers."""
    started = time.perf_counter()
    try:
        result = client.complete(
            system="You are a connectivity probe for a CAD repair harness.",
            user="Reply with exactly: OK",
            temperature=0.0,
            max_tokens=8,
        )
    except Exception as error:  # noqa: BLE001 - the sentence is the product here
        return {"ok": False, "error": f"{type(error).__name__}: {error}"}
    elapsed = round(time.perf_counter() - started, 3)
    return {
        "ok": bool(result.text.strip()),
        "text": result.text.strip()[:80],
        "model": result.model,
        "prompt_tokens": result.usage.prompt_tokens,
        "completion_tokens": result.usage.completion_tokens,
        "billing": result.usage.billing,
        "cost": result.usage.cost,
        "wall_s": elapsed,
    }


def _check_gen(config_path: Path | None, *, call: bool, spec_path: str) -> tuple[dict, int]:
    """Whether a DiT/CAD adapter is configured, importable, and shape-correct."""
    from pah.app.providers import (  # noqa: PLC0415
        PaidGenerationRefused,
        ProviderError,
        generation_is_paid,
        load_python_target,
        load_registry,
        paid_generation_allowed,
        require_paid_generation,
    )

    registry = load_registry(str(config_path) if config_path is not None else "pah.toml")
    section: dict = {
        "role": "gen",
        "configured": False,
        "default": registry.defaults.get("gen", ""),
        "paid": False,
        "checks": [],
        "call": None,
    }
    try:
        config = registry.choose("gen")
    except ProviderError as error:
        # Not a failure: the parametric builder is the default, not a fallback.
        section["checks"].append(
            {"check": "generator configured", "ok": True, "detail": f"not configured, using the parametric builder ({error})"}
        )
        section["using"] = "local-parametric"
        return section, 0

    section.update(
        {"configured": True, "provider": config.id, "kind": config.kind,
         "target": str((config.compat or {}).get("target", "") or "")}
    )
    if config.kind == "local-parametric":
        # The shipped default: parts come from the parametric builder, which is a
        # legitimate generator, not a missing one. Calling that a failure would teach
        # a reader to ignore the FAIL column.
        section["checks"].append(
            {"check": "generator kind", "ok": True,
             "detail": "parametric builder (no external generator configured)"}
        )
        section["using"] = "local-parametric"
        return section, 0
    if config.kind != "python":
        section["checks"].append(
            {"check": "generator kind", "ok": False,
             "detail": f"kind={config.kind!r}; the generator port takes kind='python' "
                       "with compat.target = 'module:function' or 'path/to/file.py:function'"}
        )
        return section, 1

    # Informational, not a failure: a paid generator is a legitimate configuration.
    # What a reader needs to know before `--call` is whether the gate is open.
    paid = generation_is_paid(config)
    section["paid"] = paid
    section["checks"].append(
        {
            "check": "paid generator",
            "ok": True,
            "detail": (
                "declared paid — gate open (authorised)"
                if paid_generation_allowed()
                else "declared paid — gate closed: --call needs --allow-paid-gen"
            )
            if paid
            else "free path: no service is called",
        }
    )
    try:
        # Import only: a doctor check must stay free even when the adapter is paid.
        # It is the *call* below that the gate governs.
        target = load_python_target(config)
    except ProviderError as error:
        section["checks"].append({"check": "target importable", "ok": False, "detail": str(error)})
        return section, 1
    section["checks"].append(
        {"check": "target importable", "ok": True, "detail": f"{section['target']} -> {getattr(target, '__module__', '?')}"}
    )
    section["using"] = config.id
    failed = 0
    if call:
        try:
            require_paid_generation(config, action="pah providers doctor --call")
        except PaidGenerationRefused as error:
            # The refusal is the call's result, not a crash: exit 1 with the sentence
            # that says no request was made and which flag would authorise one.
            section["call"] = {"ok": False, "error": str(error)}
            failed = 1
        else:
            section["call"] = _call_gen(target, spec_path)
            failed += int(not section["call"]["ok"])
    return section, failed


def _call_gen(target, spec_path: str) -> dict:
    """Run the adapter on one small spec and check the shape of what came back.

    The checks are the contract's, not the adapter's claims: two world-frame
    meshes, watertight, and an `assembly` whose parts carry a geometry hash. A
    generator that returns "something" is not integrated; one that returns the
    contract's object is.
    """
    import tempfile  # noqa: PLC0415

    path = Path(spec_path)
    if not path.exists():
        return {"ok": False, "error": f"spec not found: {spec_path}"}
    try:
        from pah.design.spec import DesignSpec  # noqa: PLC0415

        spec = DesignSpec.model_validate(json.loads(path.read_text("utf-8")))
        out = Path(tempfile.mkdtemp(prefix="pah-doctor-gen-"))
        started = time.perf_counter()
        design = target(spec, out)
        elapsed = round(time.perf_counter() - started, 3)
    except Exception as error:  # noqa: BLE001 - the sentence is the product here
        return {"ok": False, "error": f"{type(error).__name__}: {error}"}

    missing = [
        name
        for name in ("spec", "assembly", "meshes", "paths")
        if getattr(design, name, None) is None
    ]
    if missing:
        return {
            "ok": False,
            "error": f"returned {type(design).__name__} without {missing}; expected "
                     "MaterializedDesign(spec, assembly, meshes, paths, root)",
        }
    parts = {
        part_id: {
            "watertight": bool(mesh.is_watertight),
            "volume_mm3": round(float(mesh.volume), 2),
            "faces": int(len(mesh.faces)),
            "path": str(design.paths.get(part_id, "")),
        }
        for part_id, mesh in design.meshes.items()
    }
    watertight = all(row["watertight"] for row in parts.values())
    has_hashes = all(
        (part.get("geom") or {}).get("hash") for part in design.assembly.parts
    )
    ok = bool(parts) and watertight and has_hashes and len(parts) == 2
    detail = ""
    if not ok:
        detail = (
            f"parts={sorted(parts)} watertight={watertight} geometry_hash={has_hashes}; "
            "the contract needs two watertight parts with content-addressed hashes"
        )
    return {"ok": ok, "parts": parts, "wall_s": elapsed, "detail": detail}


# --------------------------------------------------------------------------- #
# printing
# --------------------------------------------------------------------------- #
def _print(payload: dict) -> None:
    where = payload["config"] or "<no pah.toml found>"
    print(f"pah providers doctor  ·  config = {where}\n")
    for role in ("llm", "gen"):
        section = payload["sections"].get(role)
        if section is None:
            continue
        title = "LLM (decision layer)" if role == "llm" else "GEN (geometry layer)"
        print(f"[{role}] {title}")
        if section.get("provider"):
            print(f"  provider   {section['provider']}  kind={section.get('kind')}")
        if role == "llm" and section.get("model"):
            print(f"  model      {section['model']}   base_url={section.get('base_url')}")
        if role == "gen":
            print(f"  default    {section.get('default') or '<unset>'}"
                  f"   using={section.get('using')}")
            if section.get("target"):
                print(f"  target     {section['target']}")
        for check in section["checks"]:
            mark = "PASS" if check["ok"] else "FAIL"
            print(f"  {mark}  {check['check']:20s} {check['detail']}")
        call = section.get("call")
        if call is not None:
            if call.get("ok"):
                if role == "llm":
                    print(f"  PASS  live call           {call['text']!r} in {call['wall_s']}s"
                          f"  tokens={call['prompt_tokens']}+{call['completion_tokens']}"
                          f"  billing={call['billing']}")
                else:
                    print(f"  PASS  live materialise    {call['wall_s']}s parts="
                          f"{ {k: v['volume_mm3'] for k, v in call['parts'].items()} }")
            else:
                print(f"  FAIL  live call           {call.get('error') or call.get('detail')}")
        print()
    print(f"{payload['failures']} failing check(s)")
