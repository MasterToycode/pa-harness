"""Generate the stress suite: four deliberately broken designs, from the fixtures.

    python scripts/make_stress_suite.py [--out fixtures/design/stress]

`fixtures/design` holds one broken case and three clean ones, so an equal-budget
comparison over it is degenerate: both policies repair 4/4 and the table says
nothing. The stress suite keeps the same four *kinds* and breaks each one in a
different way, so the policies actually have to choose:

| file | what is wrong | which verifier should say so |
|---|---|---|
| `stress-peg-hole-loose` | clearance 0.9 mm, above the band | `fit_clearance` (excessive) |
| `stress-peg-hole-oversize` | peg 1.4 mm larger than the hole | interference / path / physics |
| `stress-snap-fit-swelled` | hook wider than its slot | interference |
| `stress-dovetail-tight` | dovetail clearance 0.0 mm | `fit_clearance` (insufficient) |

The insert pose is recomputed with `builder_for(kind).seated_pose_mm(...)`, never
hand-edited: a parameter change moves where the insert seats, and a stale pose
would show up as a 1000+ mm3 interference that no design intent asked for.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from pah.design.connectors import builder_for  # noqa: E402
from pah.design.spec import DesignSpec  # noqa: E402

FIXTURES = REPO / "fixtures" / "design"
OUT = FIXTURES / "stress"

#: `(base fixture, new spec id, requirement, param overrides)`.
CASES: tuple[tuple[str, str, str, dict], ...] = (
    (
        "demo-peg-hole-clear.json",
        "stress-peg-hole-loose",
        "A peg that wobbles: 0.9 mm of radial clearance in a slot designed for slip fit.",
        {"d_peg_mm": 8.2, "d_hole_mm": 10.0},
    ),
    (
        "demo-peg-hole-interfere.json",
        "stress-peg-hole-oversize",
        "A peg 1.4 mm larger than its socket: the classic press-fit-gone-wrong.",
        {"d_peg_mm": 11.4, "d_hole_mm": 10.0},
    ),
    (
        "demo-snap-fit.json",
        "stress-snap-fit-swelled",
        "A hook wider than the slot it must pass through.",
        {"hook_w_mm": 4.9, "slot_w_mm": 4.2},
    ),
    (
        "demo-dovetail.json",
        "stress-dovetail-tight",
        "A dovetail with zero clearance: it cannot slide.",
        {"clearance_mm": 0.0},
    ),
)


def build(base: Path, spec_id: str, requirement: str, params: dict) -> DesignSpec:
    spec = DesignSpec.model_validate(json.loads(base.read_text("utf-8")))
    spec = spec.with_interface(params=params)
    seated = builder_for(spec.interface.kind).seated_pose_mm(spec)
    insert = spec.insert.model_copy(update={"pose_mm": tuple(float(x) for x in seated)})
    return spec.edit(
        spec_id=spec_id,
        requirement=requirement,
        parts=tuple(insert if part.id == insert.id else part for part in spec.parts),
        provenance={
            "via": "scripts/make_stress_suite.py",
            "base": base.name,
            "severity": "deliberately broken",
            "pose": f"builder:{spec.interface.kind}.seated_pose_mm",
        },
        revision=1,
    )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args(argv[1:])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for base_name, spec_id, requirement, params in CASES:
        base = FIXTURES / base_name
        if not base.exists():
            print(f"missing base fixture {base}")
            return 2
        spec = build(base, spec_id, requirement, params)
        path = out / f"{spec_id}.json"
        path.write_text(
            json.dumps(spec.model_dump(), ensure_ascii=False, indent=1) + "\n", "utf-8"
        )
        print(
            f"{path.name:34s} kind={spec.interface.kind:9s} "
            f"clearance={spec.clearance_mm:+.3f} mm  params={params}"
        )
    print(f"\nwrote {len(CASES)} spec(s) to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
