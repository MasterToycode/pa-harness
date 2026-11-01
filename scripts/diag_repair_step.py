"""Diagnose one repair step end to end, printing the numbers at each stage.

    python scripts/diag_repair_step.py [spec.json]

Used when an episode reports a failure code that the design change should have
removed: it isolates whether the stale thing is the design edit, the assembly
that carries it, or the geometry the verifier was shown. Each stage prints what
it saw, so the first wrong number names the module.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

import trimesh  # noqa: E402

from pah.design.api import spec_of, view_of  # noqa: E402
from pah.design.materialize import materialize  # noqa: E402
from pah.design.spec import DesignSpec  # noqa: E402
from pah.policy import design_ops  # noqa: E402
from pah.policy.registry import OperatorRegistry  # noqa: E402


def intersection_volume(a: trimesh.Trimesh, b: trimesh.Trimesh) -> float:
    try:
        out = trimesh.boolean.intersection([a, b], engine="manifold")
    except Exception as error:  # noqa: BLE001 - the number is the diagnostic
        print(f"    boolean failed: {type(error).__name__}: {error}")
        return float("nan")
    if out is None:
        return 0.0
    return float(out.volume)


def main(argv: list[str]) -> int:
    spec_path = Path(argv[1]) if len(argv) > 1 else REPO / "fixtures" / "design" / "demo-peg-hole-interfere.json"
    spec = DesignSpec.model_validate(json.loads(spec_path.read_text("utf-8")))
    root = Path(tempfile.mkdtemp(prefix="pah-diag-"))

    print(f"spec   {spec.spec_id} kind={spec.interface.kind} clearance={spec.clearance_mm:+.3f}")
    built = materialize(spec, root / "r1")
    print(f"  built  hashes={[p['geom']['hash'][:14] for p in built.assembly.parts]}")
    print(f"  inter  {intersection_volume(built.meshes[spec.body.id], built.meshes[spec.insert.id]):.4f} mm3")

    registry = OperatorRegistry()
    design_ops.register_builtin(registry)
    plan = _plan_for(spec, registry)
    print(f"plan   operator={plan.operator} params={plan.params}")

    after_assembly = registry.apply(built.assembly, plan)
    after = spec_of(after_assembly)
    print(f"  spec'  revision={after.revision} clearance={after.clearance_mm:+.3f} "
          f"key={after.content_key()[:18]}")
    print(f"  assem' version={after_assembly.version} parent_ok={after_assembly.parent == built.assembly.id}")
    print(f"  assem' hashes={[p['geom']['hash'][:14] for p in after_assembly.parts]}")
    print(f"  assem' paths={[ (p['geom'].get('path') or '<empty>')[-22:] for p in after_assembly.parts]}")

    rebuilt = materialize(after, root / "r2")
    print(f"  rebuilt hashes={[p['geom']['hash'][:14] for p in rebuilt.assembly.parts]}")
    print(f"  rebuilt inter {intersection_volume(rebuilt.meshes[after.body.id], rebuilt.meshes[after.insert.id]):.4f} mm3")

    view = view_of(after_assembly, root / "r2")
    print(f"  view   meshes_from={'paths' if view.paths else 'root'}")
    try:
        body, insert = view.body, view.insert
        print(f"  view   inter {intersection_volume(body, insert):.4f} mm3 "
              f"body_tris={len(body.faces)} insert_tris={len(insert.faces)}")
    except Exception as error:  # noqa: BLE001
        print(f"  view   failed: {type(error).__name__}: {error}")
    return 0


def _plan_for(spec: DesignSpec, registry: OperatorRegistry):
    """The rule policy's answer for an interference certificate, if it has one."""
    from pah.core.model import FailureCertificate
    from pah.policy.design_policies import RulePolicy

    certs = [
        FailureCertificate(
            code="geometric_interference",
            family="assembly",
            locus={"part": spec.insert.id},
            magnitude={"volume_mm3": 25.34},
            observed_by="diag",
        )
    ]
    policy = RulePolicy(registry=registry)
    plan = policy.propose(_empty_assembly(spec), certs, [], _budget())
    if plan is None:
        raise SystemExit("the rule policy proposed nothing for geometric_interference")
    return plan


def _empty_assembly(spec: DesignSpec):
    from pah.design.api import assembly_from_spec

    return assembly_from_spec(spec=spec)


def _budget():
    from pah.core.model import Budget

    return Budget(rounds=3, geom_calls=200, sim_calls=40)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
