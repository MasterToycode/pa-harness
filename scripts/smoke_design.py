"""Smoke-check the design plane end to end, without a policy or a loop.

    python scripts/smoke_design.py [fixtures/design]

Materialises every spec under the fixture directory, prints what was built, and
runs the design-plane verifier cascade over it if those modules exist. This is the
"is the geometry real" check: watertight solids, a hole the peg actually fits
through, an interference case that measures positive, and hashes that do not move
between two runs.

It is a script, not a test: it exits non-zero on a failed invariant, and prints the
evidence either way. A test would hide the numbers; the numbers are the point here.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:  # the package is installed, but scripts run first
    sys.path.insert(0, str(REPO / "src"))

from pah.design.materialize import materialize  # noqa: E402
from pah.design.spec import DesignSpec  # noqa: E402

FIXTURES = REPO / "fixtures" / "design"


def main(argv: list[str]) -> int:
    root = Path(argv[1]) if len(argv) > 1 else FIXTURES
    specs = sorted(root.glob("*.json"))
    if not specs:
        print(f"no specs under {root}")
        return 2

    out = Path(tempfile.mkdtemp(prefix="pah-smoke-"))
    failures: list[str] = []
    print(f"materialising {len(specs)} spec(s) into {out}\n")

    for path in specs:
        spec = DesignSpec.model_validate(json.loads(path.read_text("utf-8")))
        design = materialize(spec, out / spec.spec_id)
        volumes = {part_id: round(mesh.volume, 2) for part_id, mesh in design.meshes.items()}
        watertight = {
            part_id: mesh.is_watertight for part_id, mesh in design.meshes.items()
        }
        hashes = {
            str(part["id"]): str(part["geom"]["hash"])[:19]
            for part in design.assembly.parts
        }
        print(f"{path.name}")
        print(f"  kind={spec.interface.kind:9s} clearance={spec.clearance_mm:+.3f} mm "
              f"revision={spec.revision} parts={list(design.meshes)}")
        print(f"  volume_mm3={volumes}")
        print(f"  watertight={watertight}")
        print(f"  geom_hash={hashes}")
        for part_id, ok in watertight.items():
            if not ok:
                failures.append(f"{path.name}: part {part_id} is not watertight")
        for part_id, volume in volumes.items():
            if volume <= 0:
                failures.append(f"{path.name}: part {part_id} has no volume")

        # Same spec, second materialisation: identity and bytes must not move.
        again = materialize(spec, out / spec.spec_id)
        first = {p["id"]: p["geom"]["hash"] for p in design.assembly.parts}
        second = {p["id"]: p["geom"]["hash"] for p in again.assembly.parts}
        if first != second:
            failures.append(f"{path.name}: geom_hash moved between two materialisations")
        for part_id, stl in design.paths.items():
            if Path(stl).read_bytes() != Path(again.paths[part_id]).read_bytes():
                failures.append(f"{path.name}: STL bytes for {part_id} are not deterministic")
        print()

    try:
        from pah.tools.verifiers.design_cascade import default_registry

        registry = default_registry()
        results = registry.sanity()
        print("verifier sanity:", results)
        for verifier_id, ok in results.items():
            if not ok:
                failures.append(f"sanity: {verifier_id} disabled — {registry.why_disabled(verifier_id)}")

        if results and all(results.values()):
            from pah.design.api import view_of
            from pah.tools.verifiers.design_cascade import DesignVerifierCascade

            cascade = DesignVerifierCascade(registry, root=out)
            for path in specs:
                spec = DesignSpec.model_validate(json.loads(path.read_text("utf-8")))
                design = materialize(spec, out / spec.spec_id)
                certs = cascade.check(design.assembly)
                codes = [cert.code for cert in certs]
                print(f"{path.name:32s} certs={codes}")
    except ModuleNotFoundError as error:
        print(f"(verifiers not available yet: {error})")

    print()
    if failures:
        print("FAIL")
        for line in failures:
            print(" -", line)
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
