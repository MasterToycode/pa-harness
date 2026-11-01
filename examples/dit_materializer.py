"""Template: a DiT (or CAD kernel) as PAH's geometry generator.

Copy this file somewhere importable (e.g. `my_dit/materialize.py`), fill in
`generate_insert`, and point `pah.toml` at it:

    [gen]
    default = "dit"

    [gen.providers.dit]
    kind = "python"
    compat = { target = "my_dit.materialize:materialize" }

Then:

    pah providers doctor --role gen            # config + target importable
    pah providers doctor --role gen --call     # run it once on one small spec
    pah run --spec fixtures/design/demo-peg-hole-clear.json --policy rule --rounds 3 --out runs/try

What the harness expects from you
---------------------------------
`materialize(spec, out_dir) -> MaterializedDesign`, where:

* both parts are **watertight world-frame meshes** (mm), i.e. the pose in the spec
  is already applied — the verifiers measure the assembly as it sits;
* the `assembly` is built with `pah.design.api.assembly_from_spec(...)`, which is
  what records the content-addressed `geom.hash` the loop's cycle detection and
  reproducibility checks rely on;
* the STLs are written under `out_dir/geometry/<part_id>.stl`, because a resumed
  episode re-reads them instead of re-running your model.

This template keeps the **body parametric** (it is a plate with a socket; no
generative model needed) and asks the DiT for the **insert** only. That is the
smallest honest first integration: one generated part, everything else as before,
so a failure is your adapter's and not the harness's.

Honesty rules baked in
----------------------
`generate_insert` raises when the model is not wired. There is **no silent
fallback to the parametric pegs**: a run whose report says a DiT generated the
parts, while trimesh did, is a number nobody can reproduce. `--selftest` proves the
plumbing with an explicit stub instead.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import trimesh

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:  # works from a checkout without `pip install -e .`
    sys.path.insert(0, str(REPO / "src"))

from pah.app.env import load_env_file  # noqa: E402
from pah.design.api import MaterializedDesign, assembly_from_spec  # noqa: E402
from pah.design.connectors import builder_for  # noqa: E402
from pah.design.spec import DesignSpec  # noqa: E402

# A bare `python examples/...py` does not go through `pah`'s entry point, so it loads
# `.env` itself; missing-only means an explicit shell export still wins.
load_env_file(REPO / ".env")

#: Where your model lives. Read at call time, never at import: a rotation of the
#: endpoint or a moved checkpoint must not need a code change.
MODEL_ENV = "DIT_MODEL"


# --------------------------------------------------------------------------- #
# the port
# --------------------------------------------------------------------------- #
def materialize(spec: DesignSpec, out_dir: str | Path) -> MaterializedDesign:
    """The `Materializer` the harness calls: spec in, two solids out."""
    return _build(spec, Path(out_dir), generate=generate_insert)


def generate_insert(
    spec: DesignSpec, reference: trimesh.Trimesh
) -> trimesh.Trimesh:
    """**Replace this with your model.** World-frame insert mesh, millimetres.

    `reference` is the parametric insert the harness would otherwise have used:
    same part, same seated pose, same units. Use it to place your model's output
    (`place_like`) or to sanity-check that your frame convention agrees with the
    spec's before the harness measures two unrelated solids.

    Read `MODEL_ENV` / other env vars here and call your model — local torch, an
    HTTP endpoint, a queue — the harness does not care which. Two things it does
    care about:

    * return a **watertight** mesh (`mesh.is_watertight`), or the verifiers will
      refuse it (correctly: an open shell has no interior to collide or measure);
    * return it in **world coordinates at the seated pose**, not the model's
      canonical frame.

    A quick local model skeleton::

        path = os.environ.get(MODEL_ENV)
        if not path:
            raise RuntimeError(...)
        model = _load(path)                     # cached at module level in practice
        out = model.generate(prompt=spec.requirement, params=spec.interface.params)
        return place_like(trimesh.Trimesh(vertices=out.v, faces=out.f), reference)
    """
    raise RuntimeError(
        "DiT is not wired in this template. Two options:\n"
        "  * run `python examples/dit_materializer.py --selftest` to prove the plumbing\n"
        "    with a stub insert and no model;\n"
        "  * edit `generate_insert` in your copy of this file to call your model and\n"
        f"    return a watertight world-frame mesh (models found via ${MODEL_ENV})."
    )


def place_like(mesh: trimesh.Trimesh, reference: trimesh.Trimesh) -> trimesh.Trimesh:
    """Move a model output onto the reference part's frame: XY centrelines + bottom Z.

    Deliberately the crudest placement that can be honest: it aligns the part's
    bounding box in XY and its lowest point in Z. It does **not** rotate, so a model
    whose canonical axis is not +Z will come back lying down — keep this for the
    first test, and replace it with your model's own pose protocol (or make the
    model emit world coordinates) before reporting numbers.
    """
    placed = mesh.copy()
    target = reference.bounds
    current = placed.bounds
    shift = [
        (target[0][0] + target[1][0]) / 2 - (current[0][0] + current[1][0]) / 2,
        (target[0][1] + target[1][1]) / 2 - (current[0][1] + current[1][1]) / 2,
        target[0][2] - current[0][2],
    ]
    placed.apply_translation(shift)
    return placed


# --------------------------------------------------------------------------- #
# the shared build path (template code — copy it as-is)
# --------------------------------------------------------------------------- #
def stub_materialize(spec: DesignSpec, out_dir: str | Path) -> MaterializedDesign:
    """A **plumbing test only**: identical to the parametric builder, no model.

    Exists so `pah providers doctor --role gen --call` and `pah run` can be proven
    end to end before a model exists (`compat.target = "examples.dit_materializer:stub_materialize"`).
    It is *not* a generator and must never appear in a run whose numbers are
    reported: the report would say a python generator produced parts that are the
    parametric ones.
    """
    return _build(spec, Path(out_dir), generate=lambda _spec, reference: reference.copy())


def _build(
    spec: DesignSpec, out_dir: Path, *, generate
) -> MaterializedDesign:
    builder = builder_for(spec.interface.kind)
    geometry = builder.build(spec)
    meshes = dict(geometry.meshes)

    insert_id = spec.insert.id
    meshes[insert_id] = generate(spec, meshes[insert_id])
    if not meshes[insert_id].is_watertight:
        raise ValueError(
            f"generator returned a non-watertight mesh for part {insert_id!r}; a shell "
            "with no closed interior cannot be measured by the verifiers"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    geometry_dir = out_dir / "geometry"
    geometry_dir.mkdir(exist_ok=True)
    paths: dict[str, str] = {}
    for part_id, mesh in meshes.items():
        path = geometry_dir / f"{part_id}.stl"
        mesh.export(path)
        paths[part_id] = str(path.resolve())

    (out_dir / "spec.json").write_text(
        json.dumps(spec.model_dump(), ensure_ascii=False, indent=1) + "\n", "utf-8"
    )
    assembly = assembly_from_spec(
        spec=spec,
        geometry_paths=paths,
        mate_axis=geometry.mate_axis,
        approach_mm=geometry.approach_mm,
    )
    return MaterializedDesign(
        spec=spec,
        assembly=assembly,
        meshes=meshes,
        paths=paths,
        root=out_dir,
        mate_axis=geometry.mate_axis,
        approach_mm=geometry.approach_mm,
    )


# --------------------------------------------------------------------------- #
# selftest: prove the plumbing before the model exists
# --------------------------------------------------------------------------- #
def _selftest(spec_path: str) -> int:
    """Materialise one spec with a stub insert, then say exactly what is left to do.

    The stub returns the *parametric* insert, so this run proves the adapter's
    contract (write STLs, build an assembly with the right hashes, two watertight
    parts) without pretending a model ran. If this passes, the only remaining work
    is inside `generate_insert`.
    """
    spec = DesignSpec.model_validate(json.loads(Path(spec_path).read_text("utf-8")))
    out = Path(tempfile.mkdtemp(prefix="pah-dit-selftest-"))
    design = _build(spec, out, generate=lambda _spec, reference: reference.copy())

    print(f"spec        {spec.spec_id}  kind={spec.interface.kind}  "
          f"clearance={spec.clearance_mm:+.3f} mm")
    print(f"out         {out}")
    for part_id, mesh in design.meshes.items():
        print(f"part {part_id:8s} watertight={mesh.is_watertight} "
              f"volume={mesh.volume:.1f} mm3 faces={len(mesh.faces)}")
    print(f"assembly    {design.assembly.id}  parts={len(design.assembly.parts)}")
    print(f"hashes      {[p['geom']['hash'][:18] for p in design.assembly.parts]}")
    print(f"mate_axis   {design.mate_axis}  approach={design.approach_mm} mm")
    print("\nplumbing OK. Replace `generate_insert` with your model; do not change "
          "`_build`.")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--selftest", action="store_true", help="prove the adapter plumbing with a stub")
    parser.add_argument(
        "--spec",
        default=str(REPO / "fixtures" / "design" / "demo-peg-hole-clear.json"),
        help="spec used by --selftest",
    )
    args = parser.parse_args(argv[1:])
    if args.selftest:
        return _selftest(args.spec)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
