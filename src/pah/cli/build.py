"""`pah build "a prompt"` — one command from an idea to a part you can look at.

The product entry point the harness was missing. It composes what already existed, in
the order it has to happen:

    prompt ─▶ DesignSpec ─▶ episode (generate · verify · repair) ─▶ preview.png/.html
                                       │
                                       └─▶ report.json + trace.jsonl + geometry/*.stl

Design decisions worth knowing:

* **the prompt becomes `spec.requirement`**, which is what the generator builds its
  own prompt from and what the report shows. It is *not* sent to the LLM by default:
  every generation is the expensive part, so `--design llm` ("let the model pick the
  connector parameters too") is an explicit opt-in.
* **`--rounds 1` is the default here**, unlike `pah run` (3): with an external
  generator, each round is another paid generation, and a first look should cost one.
* every artifact path is printed and the preview is rendered in the same command,
  because "where is my part" is a product question, not a debugging one.

`execute()` is the same pipeline without printing — it is what `pah serve` runs in its
worker thread, so the page and the CLI can never drift apart.
"""

from __future__ import annotations

import argparse
import json
import time
import webbrowser
from pathlib import Path
from typing import Callable

EXIT_OK = 0
EXIT_NO = 1
EXIT_CRASH = 3

Progress = Callable[[str], None]


# --------------------------------------------------------------------------- #
# the pipeline
# --------------------------------------------------------------------------- #
def execute(
    prompt: str,
    *,
    kind: str = "peg_hole",
    rounds: int = 1,
    policy: str = "rule",
    out: str | Path | None = None,
    config: str | None = None,
    seed: int = 0,
    design: str = "default",
    preview: bool = True,
    progress: Progress | None = None,
) -> dict:
    """Run one build and return everything a caller (CLI or page) needs.

    Raises on a refusal — a bad kind, an uncomposable runtime, a failed episode — so
    the caller prints one sentence rather than a two-line report about nothing.
    """
    note = progress or (lambda _line: None)
    target = Path(out) if out else Path("runs") / f"build-{time.strftime('%Y%m%d-%H%M%S')}"
    target.mkdir(parents=True, exist_ok=True)
    note(f"run directory {target}")

    from pah.cli.design import default_spec, seated_pose  # noqa: PLC0415

    provisional = default_spec(kind=kind, requirement=prompt)
    pose, pose_source = seated_pose(provisional)
    spec = default_spec(
        kind=kind,
        requirement=prompt,
        insert_pose=pose,
        provenance={
            "via": "pah build",
            "prompt": prompt,
            "pose": pose_source,
            "design": design,
        },
    )
    note(f"design {kind} · clearance {spec.clearance_mm:+.3f} mm")

    if design == "llm":
        from pah.cli.design import _params_from_llm  # noqa: PLC0415

        chosen, provenance = _params_from_llm(kind, prompt, config)
        if chosen:
            spec = spec.with_interface(params=chosen).edit(
                provenance={**spec.provenance, **provenance}
            )
            note(f"llm chose params: {chosen}")
        else:
            note(f"no LLM choice ({provenance}); keeping deterministic params")

    spec_path = target / "spec.json"
    spec_path.write_text(
        json.dumps(spec.model_dump(), ensure_ascii=False, indent=1) + "\n", "utf-8"
    )

    from pah.app import runtime  # noqa: PLC0415

    cfg = runtime.RuntimeConfig(
        out=target,
        config_path=Path(config) if config else None,
        seed=seed,
        policy=policy,
        rounds=rounds,
    )
    composed = runtime.build(cfg)
    try:
        note(f"episode: policy={policy} rounds≤{rounds} generator={cfg_generator(cfg)}")
        started = time.perf_counter()
        report = composed.run_episode(
            spec, policy_name=policy, rounds=rounds, out_dir=target, resume=False
        )
        elapsed = time.perf_counter() - started
    finally:
        composed.close()
    episode = report.get("episode", {})
    note(f"verdict {episode.get('verdict')} in {elapsed:.1f}s "
         f"({episode.get('rounds')} round(s))")

    result: dict = {
        "out": str(target),
        "spec_path": str(spec_path),
        "report_path": str(target / "report.json"),
        "trace_path": str(target / "trace.jsonl"),
        "verdict": episode.get("verdict"),
        "reason": episode.get("reason"),
        "rounds": episode.get("rounds", 0),
        "generator": report.get("run", {}).get("generator"),
        "elapsed_s": round(elapsed, 1),
        "parts": [
            {"id": part_id, "path": str(detail.get("path"))}
            for part_id, detail in report.get("geometry", {}).get("parts", {}).items()
        ],
        "notes": [
            f"{verifier_id}: {text}"
            for verifier_id, text in (report.get("verifiers", {}).get("notes") or {}).items()
        ],
        "report": report,
        "preview_png": None,
        "preview_html": None,
    }

    if preview:
        try:
            from pah.cli.preview import preview_run  # noqa: PLC0415

            png, html = preview_run(target)
            result["preview_png"] = str(png)
            result["preview_html"] = str(html)
            note(f"preview {png.name} + {html.name}")
        except Exception as error:  # noqa: BLE001 - the STLs are the deliverable
            note(f"preview failed: {type(error).__name__}: {error}")
    return result


def cfg_generator(cfg) -> str:
    """The generator label the run will claim, for the log line before it starts."""
    try:
        from pah.app.providers import ProviderError, load_registry  # noqa: PLC0415
        from pah.plugins import resolve_config_path  # noqa: PLC0415

        path = resolve_config_path(cfg.config_path)
        registry = load_registry(path if path is not None else "pah.toml")
        chosen = registry.choose("gen")
        return f"{chosen.id}:{chosen.kind}"
    except Exception:  # noqa: BLE001 - a label must never stop a build
        return "local-parametric"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def add_parser(groups: argparse._SubParsersAction) -> None:
    build = groups.add_parser(
        "build", help="prompt → design → generate → verify → repaired part + preview"
    )
    build.add_argument("prompt", help="what you want, in one sentence")
    build.add_argument(
        "--kind",
        default="peg_hole",
        help="connector kind: peg_hole|snap_fit|dovetail (default peg_hole)",
    )
    build.add_argument("--rounds", type=int, default=1, help="generation rounds (default 1)")
    build.add_argument(
        "--policy",
        default="rule",
        help="decision policy for repairs: rule|random|llm (default rule)",
    )
    build.add_argument(
        "--design",
        default="default",
        choices=["default", "llm"],
        help="'llm' lets the model choose the connector parameters too (costs tokens)",
    )
    build.add_argument("--out", default=None, help="run directory (default runs/build-<time>)")
    build.add_argument("--config", default=None, help="pah.toml to use")
    build.add_argument("--seed", type=int, default=0)
    build.add_argument("--no-preview", action="store_true", help="skip PNG/HTML rendering")
    build.add_argument(
        "--open", action="store_true", help="open the interactive preview in your browser"
    )
    build.add_argument("--json", action="store_true", help="print the report as JSON")
    build.set_defaults(handler=_build)


def _build(args: argparse.Namespace) -> int:
    from pah.cli.run import _print_report  # noqa: PLC0415

    try:
        result = execute(
            args.prompt,
            kind=args.kind,
            rounds=args.rounds,
            policy=args.policy,
            out=args.out,
            config=args.config,
            seed=args.seed,
            design=args.design,
            preview=not args.no_preview,
            progress=lambda line: None if args.json else print(f"  · {line}"),
        )
    except Exception as error:  # noqa: BLE001 - one sentence, not a stack
        print(f"pah build: {error}")
        return EXIT_CRASH

    if args.json:
        print(json.dumps(result["report"], ensure_ascii=False, indent=1))
    else:
        _print_report(result["report"])
        _print_artifacts(result)
    if args.open and result.get("preview_html"):
        webbrowser.open(Path(result["preview_html"]).resolve().as_uri())
    return EXIT_OK if result["verdict"] != "failed" else EXIT_NO


def _print_artifacts(result: dict) -> None:
    """The "where is my part" block: paths first, then the numbers that matter."""
    print()
    print("=" * 66)
    print("  your part")
    print("=" * 66)
    for part in result["parts"]:
        print(f"  {part['id']:8s} {part['path']}")
    if result.get("preview_png"):
        print(f"  preview  {result['preview_png']}   ← 四视图，直接打开看")
    if result.get("preview_html"):
        print(f"  3D 视图  {result['preview_html']}   ← 双击浏览器打开，可旋转缩放")
    print(f"  spec     {result['spec_path']}")
    print(f"  report   {result['report_path']}")
    print(f"  trace    {result['trace_path']}")
    print()
    print(f"  generator {result['generator']}   verdict {result['verdict']}   "
          f"rounds {result['rounds']}   {result['elapsed_s']}s")
    for note in result["notes"]:
        print(f"  未评估 · {note[:120]}")
    print()
