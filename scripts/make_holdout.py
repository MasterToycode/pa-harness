"""Generate the independent evaluation suite (`fixtures/design/eval-holdout/`).

Why a generator instead of hand-written JSON
--------------------------------------------
The holdout exists to answer "does the policy generalise beyond the demos it was
tuned on". Hand-written cases would drift toward whatever the verifiers happen to
flag today; a seeded generator produces the same 24 designs on every run, spreads
them across connector kinds, fit bands, sizes and defect classes, and makes the
*construction rule* auditable: a reader can see that "clean" means "clearance inside
the band and wall above the minimum", not "whatever passed".

What it produces
----------------
* eight `peg_hole`, eight `snap_fit`, eight `dovetail` cases;
* per kind: four **clean** (clearance in the declared fit band, wall above
  `min_wall_mm`) and four with a deliberate defect (clearance below the band, above
  it, a negative clearance for peg_hole, or a wall below the minimum);
* every case is a plain `DesignSpec` JSON — the same shape as `fixtures/design/*.json`
  — with `print_orientation_deg` alternating between declared and undeclared so the
  report's "declared vs recommended" columns are exercised;
* a `README.md` naming the suite as evaluation-only and tabulating the intent.

The intent is *not* a promise about the verifiers: physics or the orientation search
can add a finding to a clean design, and that is reported as observed rather than
tuned away. The rule for "clean" is stated above and is the only curation.

Deterministic: seed 20261103, sorted traversal, no clock, no hashing order. Re-run
`python scripts/make_holdout.py` from the repo root to reproduce every byte.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

from pah.design.connectors import builder_for
from pah.design.spec import (
    DesignSpec,
    DovetailParams,
    FitConstraints,
    InterfaceSpec,
    PartSpec,
    PegHoleParams,
    SnapFitParams,
    radial_clearance_mm,
)

#: Fixed so the suite is reproducible; changing it changes the evidence.
SEED = 20261103

#: Where the suite lives, relative to the repository root.
SUITE_DIR = Path("fixtures/design/eval-holdout")

#: The bands the cases declare. Different from the demos on purpose (the demos use
#: 0.10–0.60 and 0.04–0.60); a holdout that reused them would be the same experiment.
BANDS: tuple[tuple[float, float], ...] = (
    (0.06, 0.22),
    (0.12, 0.35),
    (0.20, 0.50),
    (0.30, 0.65),
)

#: Wall minima to declare. A clean case must keep its wall above this by a margin.
WALLS: tuple[float, ...] = (1.0, 1.5, 2.0)
PROCESSES: tuple[str, ...] = ("fdm", "fdm", "sla", "mjf")

#: Body envelopes to draw from; `peg_hole` reads the body blank, the insert is an
#: envelope hint only (`design/connectors/peg_hole.py` builds the peg from params).
BODIES: tuple[tuple[float, float, float], ...] = (
    (50.0, 35.0, 10.0),
    (60.0, 40.0, 12.0),
    (45.0, 30.0, 8.0),
    (70.0, 45.0, 14.0),
    (55.0, 35.0, 16.0),
)

#: Insert envelopes, per kind: a hint for the Assembly/report, not the shape (the
#: builder owns the solid). Do not make these meaningful.
INSERTS: dict[str, tuple[float, float, float]] = {
    "peg_hole": (12.0, 12.0, 22.0),
    "snap_fit": (8.0, 8.0, 14.0),
    "dovetail": (20.0, 30.0, 12.0),
}


@dataclass(frozen=True)
class Case:
    """One generated design and the construction rule that produced it."""

    spec: DesignSpec
    intent: str  # "clean" | "tight" | "loose" | "negative" | "thin_wall"
    kind: str
    clearance_mm: float
    band: tuple[float, float]
    wall_margin_mm: float  # analytic wall on the thinnest side, mm (peg_hole only)


def _round(value: float) -> float:
    return round(float(value), 3)


def _wall_margin(body: tuple[float, float, float], params: dict) -> float:
    """The thinnest wall between a centred or offset bore and the body's outside.

    Analytic, because the generator decides placement before any verifier runs: a
    "clean" case must be clean by construction, not by luck. It is a margin, not a
    measurement — the printability verifier measures the built mesh and may disagree
    in the last fraction of a millimetre, which is reported rather than hidden.
    """
    half_x, half_y = body[0] / 2.0, body[1] / 2.0
    radius = float(params.get("d_hole_mm", params.get("slot_w_mm", 0.0))) / 2.0
    fx, fy = params.get("feature_xy_mm", (0.0, 0.0))
    return min(
        half_x - abs(float(fx)) - radius,
        half_y - abs(float(fy)) - radius,
    )


def _peg_hole(rng: random.Random, index: int, defect: str | None) -> Case:
    body = rng.choice(BODIES)
    band = rng.choice(BANDS)
    if defect == "thin_wall":
        d_hole = _round(rng.uniform(16.0, 22.0))
        half_x = body[0] / 2.0
        margin = 0.6  # thin by construction: below every WALLS entry
        fx = _round(half_x - d_hole / 2.0 - margin)
        fy = 0.0
        clearance = _round(rng.uniform(band[0] + 0.03, band[1] - 0.03))
        d_peg = _round(d_hole - 2.0 * clearance)
        engage = _round(rng.uniform(4.0, min(6.0, body[2] / 2.0)))
    else:
        d_peg = _round(rng.uniform(6.0, 14.0))
        if defect == "negative":
            clearance = -round(rng.uniform(0.06, 0.14), 3)
        elif defect == "tight":
            clearance = _round(band[0] - rng.uniform(0.03, 0.07))
        elif defect == "loose":
            clearance = _round(band[1] + rng.uniform(0.10, 0.20))
        else:
            clearance = _round(rng.uniform(band[0] + 0.03, band[1] - 0.03))
        d_hole = _round(d_peg + 2.0 * clearance)
        fx = _round(rng.uniform(-0.25, 0.25) * (body[0] / 2.0 - d_hole / 2.0))
        fy = _round(rng.uniform(-0.25, 0.25) * (body[1] / 2.0 - d_hole / 2.0))
        engage = _round(rng.uniform(5.0, min(9.0, body[2] / 2.0)))
    approach = _round(rng.uniform(8.0, 12.0))
    params = {
        "d_peg_mm": max(d_peg, 1.0),
        "d_hole_mm": max(d_hole, 1.0),
        "engage_mm": engage,
        "approach_mm": approach,
        "feature_xy_mm": (fx, fy),
    }
    wall = _wall_margin(body, params)
    return _case("peg_hole", body, band, params, clearance, defect, wall, index)


def _snap_fit(rng: random.Random, index: int, defect: str | None) -> Case:
    body = rng.choice(BODIES)
    band = rng.choice(BANDS)
    hook_w = _round(rng.uniform(2.8, 5.0))
    if defect == "tight":
        clearance = _round(band[0] - rng.uniform(0.03, 0.07))
    elif defect == "loose":
        clearance = _round(band[1] + rng.uniform(0.10, 0.20))
    else:
        clearance = _round(rng.uniform(band[0] + 0.03, band[1] - 0.03))
    slot_w = _round(hook_w + 2.0 * clearance)
    params = {
        "hook_w_mm": hook_w,
        "hook_h_mm": _round(rng.uniform(0.8, 1.6)),
        "beam_len_mm": _round(rng.uniform(8.0, 14.0)),
        "beam_thk_mm": _round(rng.uniform(1.2, 1.8)),
        "slot_w_mm": slot_w,
        "engage_mm": _round(rng.uniform(5.0, 8.0)),
        "approach_mm": _round(rng.uniform(10.0, 14.0)),
    }
    return _case("snap_fit", body, band, params, clearance, defect, float("inf"), index)


def _dovetail(rng: random.Random, index: int, defect: str | None) -> Case:
    body = rng.choice(BODIES)
    band = rng.choice(BANDS)
    if defect == "tight":
        clearance = _round(max(band[0] - rng.uniform(0.03, 0.07), 0.0))
    elif defect == "loose":
        clearance = _round(band[1] + rng.uniform(0.10, 0.20))
    else:
        clearance = _round(rng.uniform(band[0] + 0.03, band[1] - 0.03))
    engage = _round(rng.uniform(12.0, min(22.0, body[0] / 2.0)))
    params = {
        "width_mm": _round(rng.uniform(8.0, 14.0)),
        "depth_mm": _round(rng.uniform(4.0, 7.0)),
        "angle_deg": _round(rng.uniform(8.0, 18.0)),
        "clearance_mm": clearance,
        "engage_mm": engage,
        "approach_mm": _round(engage + rng.uniform(2.0, 5.0)),
    }
    return _case("dovetail", body, band, params, clearance, defect, float("inf"), index)


def _case(
    kind: str,
    body: tuple[float, float, float],
    band: tuple[float, float],
    params: dict,
    clearance: float,
    defect: str | None,
    wall_margin: float,
    index: int,
) -> Case:
    """Build the spec, then move the insert to the pose this kind says is seated.

    The seat is asked of the builder *after* the parameters are chosen: a peg's seat
    is `top - engage`, so varying `engage` without recomputing the pose would put the
    insert in the previous design's place — the failure mode
    `connector.seated_pose_mm` exists to prevent.
    """
    intent = defect or "clean"
    min_wall = WALLS[0] if defect == "thin_wall" else WALLS[min(index, len(WALLS) - 1)]
    declared = index % 2 == 0
    constraints = FitConstraints(
        min_clearance_mm=band[0],
        max_clearance_mm=band[1],
        min_wall_mm=2.0 if defect == "thin_wall" else min_wall,
        process=PROCESSES[index % len(PROCESSES)],
        target_fit="slip" if band[0] < 0.25 else "snug",
        print_orientation_deg=(0.0, 0.0, 0.0) if declared else None,
    )
    spec_id = f"holdout-{kind.replace('_', '-')}-{intent.replace('_', '-')}-{index:02d}"
    provenance = {
        "generator": "scripts/make_holdout.py",
        "seed": SEED,
        "intent": intent,
        "construction": (
            "clearance inside the declared band and wall above min_wall_mm"
            if intent == "clean"
            else f"deliberate {intent} defect; the rest of the design is ordinary"
        ),
        "suite": "eval-holdout — evaluation only, never tuning",
    }
    body_part = PartSpec(id="body", role="body", size_mm=body)
    insert_part = PartSpec(id="insert", role="insert", size_mm=INSERTS[kind])
    provisional = DesignSpec(
        spec_id=spec_id,
        requirement=(
            f"holdout {intent} case #{index}: a {kind.replace('_', ' ')} joint on a "
            f"{body[0]:.0f}x{body[1]:.0f}x{body[2]:.0f} mm plate"
        ),
        parts=(body_part, insert_part),
        interfaces=(
            InterfaceSpec(id="iface_0", kind=kind, a="body", b="insert", params=params),
        ),
        constraints=constraints,
        provenance=provenance,
    )
    seat = builder_for(kind).seated_pose_mm(provisional)
    spec = DesignSpec(
        spec_id=spec_id,
        requirement=provisional.requirement,
        parts=(body_part, insert_part.model_copy(update={"pose_mm": seat})),
        interfaces=provisional.interfaces,
        constraints=constraints,
        provenance=provenance,
    )
    actual = radial_clearance_mm(kind, spec.interface.params)
    return Case(
        spec=spec,
        intent=intent,
        kind=kind,
        clearance_mm=actual,
        band=band,
        wall_margin_mm=wall_margin,
    )


#: Per kind, the four clean cases then four defects. Order is fixed so the seed maps
#: deterministically: `index` 0..7 per kind, `index % 2` decides declared orientation.
DEFECTS: dict[str, tuple[str | None, ...]] = {
    "peg_hole": (None, None, None, None, "tight", "loose", "negative", "thin_wall"),
    "snap_fit": (None, None, None, None, "tight", "tight", "loose", "loose"),
    "dovetail": (None, None, None, None, "tight", "tight", "loose", "loose"),
}


def generate() -> list[Case]:
    """Every case, in a fixed order; no global random state."""
    rng = random.Random(SEED)
    cases: list[Case] = []
    for kind, builders in (
        ("peg_hole", _peg_hole),
        ("snap_fit", _snap_fit),
        ("dovetail", _dovetail),
    ):
        for index, defect in enumerate(DEFECTS[kind]):
            cases.append(builders(rng, index, defect))
    seen: set[str] = set()
    for case in cases:
        if case.spec.spec_id in seen:
            raise RuntimeError(f"duplicate spec_id {case.spec.spec_id!r}")
        seen.add(case.spec.spec_id)
    return cases


def write(cases: list[Case], suite_dir: Path, *, verify: bool) -> None:
    """Writes the JSON files and the README; optionally proves each case materialises."""
    suite_dir.mkdir(parents=True, exist_ok=True)
    for old in sorted(suite_dir.glob("*.json")):
        old.unlink()
    for case in cases:
        path = suite_dir / f"{case.spec.spec_id}.json"
        path.write_text(
            json.dumps(case.spec.model_dump(mode="json"), indent=2, sort_keys=True, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        if verify:
            _materialises(case.spec)
    (suite_dir / "README.md").write_text(_readme(cases), "utf-8")


def _materialises(spec: DesignSpec) -> None:
    """Materialise into a scratch directory too; a suite of unbuildable specs is a trap."""
    import tempfile  # noqa: PLC0415 - only the verification path needs it

    from pah.design.materialize import materialize  # noqa: PLC0415

    with tempfile.TemporaryDirectory(prefix="pah-holdout-") as scratch:
        design = materialize(spec, Path(scratch))
    if not design.meshes:
        raise RuntimeError(f"{spec.spec_id} materialised no meshes")


def _readme(cases: list[Case]) -> str:
    clean = sum(1 for case in cases if case.intent == "clean")
    lines = [
        "# Evaluation holdout — not for tuning",
        "",
        "These designs exist to answer one question: does a policy that looked good on",
        "the four demos also close *unseen* designs? Nothing here may be used to choose",
        "a policy's parameters, threshold or ladder; if it were, the comparison would be",
        "measuring the author instead of the agent.",
        "",
        f"- generated by `scripts/make_holdout.py` with seed `{SEED}` (deterministic; re-running reproduces every byte)",
        f"- {len(cases)} cases: {clean} built clean, {len(cases) - clean} with one deliberate defect",
        "- 'clean' means: clearance inside the declared fit band, and wall above `min_wall_mm`",
        "  (peg_hole uses an analytic wall margin; the other two kinds are governed by their",
        "  slot/groove dimensions). 'Defect' is one of `tight` / `loose` / `negative` /",
        "  `thin_wall`; everything else about the design is ordinary.",
        "- `print_orientation_deg` alternates between declared and undeclared, so the",
        "  report's declared-vs-recommended columns are both exercised.",
        "",
        "The intent is a construction rule, not a prediction: a clean design can still be",
        "flagged by the physics tier or the orientation search. Observed verdicts are",
        "reported next to the intent, never tuned into a pass.",
        "",
        "| case | kind | intent | band (mm) | clearance (mm) | wall margin (mm) | orientation |",
        "|---|---|---|---|---|---|---|",
    ]
    for case in cases:
        declared = "declared" if case.spec.constraints.print_orientation_deg is not None else "none"
        wall = "-" if math.isinf(case.wall_margin_mm) else f"{case.wall_margin_mm:.2f}"
        lines.append(
            f"| `{case.spec.spec_id}` | {case.kind} | {case.intent} | "
            f"{case.band[0]:.2f}–{case.band[1]:.2f} | {case.clearance_mm:+.3f} | "
            f"{wall} | {declared} |"
        )
    lines.append("")
    lines.append("Run it with:")
    lines.append("")
    lines.append("```")
    lines.append("pah eval --suite holdout --policies rule,random --rounds 3 --out runs/eval-holdout")
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(SUITE_DIR), help="suite directory")
    parser.add_argument(
        "--no-verify",
        dest="verify",
        action="store_false",
        help="skip the materialise-every-case check (faster, less proof)",
    )
    parser.set_defaults(verify=True)
    args = parser.parse_args()
    cases = generate()
    write(cases, Path(args.out), verify=args.verify)
    clean = sum(1 for case in cases if case.intent == "clean")
    print(
        f"wrote {len(cases)} cases to {args.out} "
        f"({clean} clean, {len(cases) - clean} deliberate defects; seed {SEED})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
