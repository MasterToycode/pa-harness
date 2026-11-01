"""`pah design new` and `pah design sanity` — the entrance and the health check.

    pah design new --requirement "a press-fit peg that holds a bracket" \
                   --kind peg_hole --out runs/demo
    pah design sanity [--json]

`design new` turns a sentence into the one artifact the rest of v2 consumes: a
`DesignSpec`. It is a *generator*, not a planner: it picks a complete set of numbers
for the requested connector kind, writes them down, and records in `provenance` who
picked what. That provenance field is the reason this command is worth having — a
spec whose peg diameter came from a model and whose pose came from a default must
say so, or a later report cannot tell which decisions were measured.

With an LLM configured in `pah.toml` the *parameters* may come from the model; the
part envelopes, poses and constraints stay deterministic, because a model's job here
is the connector arithmetic (`docs/contracts/v2-interfaces.md` §9.3: no geometry
fields in an LLM output). With no provider — the normal case on this machine — the
deterministic defaults are used and the provenance says exactly that, including the
reason there was no model.

`design sanity` is the §4.8 gate for the *design* plane: every verifier ships a
`known_good`/`known_bad` pair, and one that disagrees with its own fixtures is
disabled before an episode can cite it. Same table as `pah verify sanity`, one plane
over.

Exit codes: `0` the thing asked for was true, `1` a verifier disagrees with its
fixtures (sanity) or the spec could not be written, `3` the request or the
environment is broken (unknown kind, T2 not landed, bad output path).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pah.cli.main import EXIT_CRASH, EXIT_NO, EXIT_OK
from pah.design.spec import (
    DesignSpec,
    FitConstraints,
    InterfaceSpec,
    PartSpec,
    builtin_params_model,
    radial_clearance_mm,
)

__all__ = ["add_parser", "default_parts", "default_spec", "KINDS"]

#: The connector kinds `design new` can start from. Read off the spec module rather
#: than re-listed, so a kind added to `design/spec.py` shows up here for free.
KINDS = tuple(
    name for name in ("peg_hole", "snap_fit", "dovetail") if builtin_params_model(name)
)

#: Deterministic connector numbers, per kind. They are deliberately *inside* the
#: default fit band (0.10–0.60 mm): `design new` should hand the loop a design worth
#: verifying, not a broken one. The broken Peg-Hole from the proposal
#: (`d_peg=10.2, d_hole=10.0`) is a fixture of the repair story, not a default.
_PARAMS: dict[str, dict] = {
    "peg_hole": {
        "d_peg_mm": 10.0,
        "d_hole_mm": 10.2,  # 0.10 mm radial clearance
        "engage_mm": 8.0,
        "approach_mm": 10.0,
        "feature_xy_mm": (0.0, 0.0),
    },
    "snap_fit": {"approach_mm": 12.0},  # hook 4.0 into slot 4.2 -> 0.10 mm
    "dovetail": {"approach_mm": 22.0},  # 0.10 mm per side
}

#: Part envelopes, per kind: `(body_size, insert_size)`. Poses are derived by
#: `default_parts` so the insert sits on the body's top face — the one convention
#: the frozen spec does not state, recorded in the spec's provenance so a materialiser
#: with a different one is a visible disagreement instead of a silent overlap.
_SIZES: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    "peg_hole": ((60.0, 40.0, 12.0), (22.0, 22.0, 20.0)),
    "snap_fit": ((60.0, 40.0, 12.0), (8.0, 8.0, 14.0)),
    "dovetail": ((60.0, 40.0, 16.0), (20.0, 30.0, 12.0)),
}

#: How the insert is seated, per kind, copied from the shipped `fixtures/design/*`
#: specs rather than invented: those are the designs T1's builders are tested against,
#: and a generator that placed the insert differently would produce a spec the loop
#: repairs into a shape nobody validated. The pose is the **seated** pose (contract:
#: `approach_mm` is how far above it the insertion path starts).
_INSERT_POSE: dict[str, tuple[float, float, float]] = {
    "peg_hole": (0.0, 0.0, -2.0),
    "snap_fit": (0.0, 0.0, -3.0),
    "dovetail": (10.0, 0.0, 8.0),
}

#: How the deterministic parts were laid out; copied into `provenance["pose"]`.
POSE_RULE = "part poses copied from fixtures/design; insert pose is the seated pose"


def add_parser(groups: argparse._SubParsersAction) -> None:
    """Mounts the `design` group under the root parser (called from `cli/main.py`)."""
    design = groups.add_parser("design", help="design-plane tools")
    commands = design.add_subparsers(dest="command", required=True)

    new = commands.add_parser("new", help="write an initial DesignSpec")
    new.add_argument("--requirement", default="", help="the sentence the design answers")
    new.add_argument("--kind", default="peg_hole", choices=list(KINDS))
    new.add_argument("--spec-id", dest="spec_id", default=None)
    new.add_argument("--out", required=True, help="directory to write spec.json into")
    new.add_argument("--config", default=None, help="pah.toml for the optional LLM provider")
    new.add_argument(
        "--no-llm",
        dest="no_llm",
        action="store_true",
        help="use the deterministic defaults even when a provider is configured",
    )
    new.add_argument("--json", action="store_true", help="print the spec as JSON")
    new.set_defaults(handler=_new)

    sanity = commands.add_parser(
        "sanity", help="run every design verifier's known_good/known_bad fixtures (§4.8)"
    )
    sanity.add_argument("--json", action="store_true", help="machine-readable output")
    sanity.set_defaults(handler=_sanity)


# --------------------------------------------------------------------------- #
# pah design new
# --------------------------------------------------------------------------- #
def default_parts(
    kind: str, insert_pose: tuple[float, float, float] | None = None
) -> tuple[PartSpec, PartSpec]:
    """The two blanks for `kind`: a body at the origin and a seated insert.

    `insert_pose` is what the kind's builder says the seated pose is; without it the
    fixture-copied pose is used so the spec is still well formed.
    """
    if kind not in _SIZES:
        raise KeyError(f"{kind!r} has no default part envelope")
    body_size, insert_size = _SIZES[kind]
    body = PartSpec(id="body", role="body", size_mm=body_size, pose_mm=(0.0, 0.0, 0.0))
    insert = PartSpec(
        id="insert",
        role="insert",
        size_mm=insert_size,
        pose_mm=_INSERT_POSE[kind] if insert_pose is None else insert_pose,
    )
    return body, insert


def default_spec(
    *,
    kind: str,
    requirement: str = "",
    spec_id: str | None = None,
    params: dict | None = None,
    provenance: dict | None = None,
    insert_pose: tuple[float, float, float] | None = None,
) -> DesignSpec:
    """A complete, valid `DesignSpec` for `kind`.

    The interface is constructed with the *target* kind's complete parameter set;
    `DesignSpec.with_interface` is deliberately not used here because it replaces an
    existing parameter set, and there is no previous design to preserve.
    """
    if kind not in KINDS:
        raise KeyError(f"unknown connector kind {kind!r}; this build offers {KINDS}")
    body, insert = default_parts(kind, insert_pose)
    model = builtin_params_model(kind)
    values = dict(_PARAMS[kind])
    values.update(params or {})
    validated = model.model_validate(values).model_dump()
    spec = DesignSpec(
        spec_id=spec_id or f"design-{kind}",
        requirement=requirement,
        parts=(body, insert),
        interfaces=(
            InterfaceSpec(id="iface_0", kind=kind, a=body.id, b=insert.id, params=validated),
        ),
        constraints=FitConstraints(),
        provenance=dict(provenance or {}),
    )
    # Read it back once so a spec that silently lost a field cannot be written: the
    # clearance is the number every later report is judged on.
    assert spec.clearance_mm == radial_clearance_mm(kind, validated)
    return spec


def seated_pose(spec: DesignSpec) -> tuple[tuple[float, float, float] | None, str]:
    """The insert pose `spec`'s connector kind asks for, and who decided it.

    The seated pose is a property of the connector, not of the generator: a peg hangs
    below its plate while a dovetail sits beside the body. A single shared default is
    how `design new --kind X` produces a phantom interference for one kind and a gap
    for another (T1 measured an 1800 mm^3 false overlap). T1's builder exposes
    `seated_pose_mm(spec)`; this function asks it and records the answer, falling back
    to the fixture-copied pose — **with the reason in `provenance`** — while T1 is
    still landing.
    """
    kind = spec.interface.kind
    try:
        from pah.design.connectors import builder_for  # noqa: PLC0415 - T1 may be absent
    except Exception as error:  # noqa: BLE001 - a missing T1 is a documented fallback
        return None, f"fixtures/design defaults (builder not importable: {type(error).__name__})"
    try:
        builder = builder_for(kind)
    except Exception as error:  # noqa: BLE001 - an unknown kind is T1's refusal to make
        return None, f"fixtures/design defaults (builder_for({kind!r}) failed: {error})"
    seated = getattr(builder, "seated_pose_mm", None)
    if not callable(seated):
        return None, f"fixtures/design defaults ({kind} builder has no seated_pose_mm yet)"
    try:
        pose = seated(spec)
    except Exception as error:  # noqa: BLE001 - a builder that cannot seat is not fatal here
        return None, (
            f"fixtures/design defaults (seated_pose_mm failed: "
            f"{type(error).__name__}: {error})"
        )
    return tuple(float(value) for value in pose), f"builder:{kind}.seated_pose_mm"


def _new(args: argparse.Namespace) -> int:
    """Writes `out/spec.json`, choosing parameters from a model when one is configured."""
    provenance = {
        "via": "pah design new",
        "kind": f"cli:{args.kind}",
        "parts": "deterministic-defaults",
        "pose": POSE_RULE,
        "constraints": "deterministic-defaults",
    }
    params: dict | None = None
    if not args.no_llm:
        params, picked = _params_from_llm(args.kind, args.requirement, args.config)
        provenance.update(picked)
    if params is None:
        provenance.setdefault("params", "deterministic-defaults")
    try:
        # Two passes: the seated pose is the connector kind's answer, and it is asked
        # on a provisional spec that already carries the chosen parameters. The second
        # build is the one written, so `provenance["pose"]` records who decided.
        provisional = default_spec(
            kind=args.kind,
            requirement=args.requirement,
            spec_id=args.spec_id,
            params=params,
            provenance=provenance,
        )
        pose, how = seated_pose(provisional)
        provenance["pose"] = how
        spec = default_spec(
            kind=args.kind,
            requirement=args.requirement,
            spec_id=args.spec_id,
            params=params,
            provenance=provenance,
            insert_pose=pose,
        )
    except Exception as error:  # noqa: BLE001 - a bad default is a request error
        print(f"pah design new: cannot build a {args.kind!r} spec: {type(error).__name__}: {error}")
        return EXIT_NO

    out = Path(args.out)
    try:
        out.mkdir(parents=True, exist_ok=True)
        target = out / "spec.json"
        target.write_text(
            json.dumps(spec.model_dump(), ensure_ascii=False, indent=2, default=str) + "\n",
            "utf-8",
        )
    except OSError as error:
        print(f"pah design new: cannot write into {out}: {error}")
        return EXIT_CRASH

    if args.json:
        print(json.dumps(spec.model_dump(), ensure_ascii=False, indent=1, default=str))
    else:
        print(f"wrote {target}")
        print(f"  {spec.short()}")
        for key, value in provenance.items():
            print(f"  {key:12s} {value}")
    return EXIT_OK


def _params_from_llm(
    kind: str, requirement: str, config_path: str | None
) -> tuple[dict | None, dict]:
    """`(params, provenance)` from the configured model, or `(None, why not)`.

    Only the connector arithmetic comes from the model. It is validated by the kind's
    own pydantic model before it is used, and any failure falls back to the
    deterministic defaults *with the reason written down* — a fallback that leaves no
    trace is how a run starts claiming a model chose something it did not.
    """
    try:
        from pah.app import runtime  # noqa: PLC0415 - keeps `pah --help` light

        client = runtime.build_llm_client(Path(config_path) if config_path else None)
    except RuntimeError as error:
        return None, {"params": f"deterministic-defaults (llm unavailable: {error})"}
    model = builtin_params_model(kind)
    defaults = {name: detail.default for name, detail in model.model_fields.items()}
    system = (
        "You are a mechanical design assistant. Return only a JSON object "
        '{"params": {...}} whose keys are the fields the user lists. Millimetres.'
    )
    user = (
        f"Connector kind: {kind}.\n"
        f"Parameter fields and defaults: {json.dumps(defaults, default=str)}\n"
        f"Requirement: {requirement or '(none given)'}\n"
        "Choose values that give a slip fit: 0.10-0.60 mm radial clearance."
    )
    try:
        result = client.complete(system=system, user=user, json_schema={"type": "object"})
        payload = json.loads(result.text)
        chosen = payload.get("params")
        if not isinstance(chosen, dict):
            raise ValueError("the model returned no 'params' object")
        validated = model.model_validate(chosen).model_dump()
    except Exception as error:  # noqa: BLE001 - any model failure falls back, loudly
        return None, {
            "params": f"deterministic-defaults (llm failed: {type(error).__name__}: {error})"
        }
    return validated, {
        "params": f"llm:{result.provider}",
        "llm_model": result.model,
        "llm_tokens": result.usage.tokens,
        "llm_provider": result.provider,
    }


# --------------------------------------------------------------------------- #
# pah design sanity
# --------------------------------------------------------------------------- #
def _sanity(args: argparse.Namespace) -> int:
    """The design plane's §4.8 gate: fixtures disagree -> the verifier is disabled."""
    from pah.app import runtime  # noqa: PLC0415 - keeps `pah --help` light

    try:
        registry = runtime.resolve_design_registry()
        results = registry.sanity()
    except RuntimeError as error:
        print(f"pah design sanity: {error}")
        return EXIT_CRASH

    payload = {
        "verifiers": {
            verifier_id: {
                "ok": ok,
                "cost_class": registry.get(verifier_id).spec.cost_class,
                "codes": list(registry.get(verifier_id).spec.codes),
                "why": registry.why_disabled(verifier_id),
            }
            for verifier_id, ok in results.items()
        },
        "disabled": sorted(registry.disabled()),
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=1))
    else:
        if not payload["verifiers"]:
            print("  (no design verifiers registered)")
        for verifier_id, detail in payload["verifiers"].items():
            mark = "PASS" if detail["ok"] else "FAIL"
            print(
                f"  {mark}  {verifier_id:22s} {detail['cost_class']:9s} "
                f"{len(detail['codes'])} code(s): {', '.join(detail['codes'])}"
            )
            if detail["why"]:
                print(f"        disabled: {detail['why']}")
        print(
            f"\n{len(results)} design verifier(s), "
            f"{len(payload['disabled'])} disabled"
        )
    return EXIT_OK if not payload["disabled"] else EXIT_NO
