"""Geometry materialisation: a `DesignSpec` in, two solids plus an `Assembly` out.

This is the seam where the design plane touches the disk, so it is deliberately
dumb. `builder_for(spec.interface.kind).build(spec)` decides the shapes; this
module decides only the *artefacts*: where the STLs go, what the assembly records,
and what has to be true before either is written down.

Why the hash is the spec's, not the STL's
-----------------------------------------
`geometry_hash` lives in `pah.design.api` and is re-exported here rather than
reimplemented: it is a function of `spec.content_key()`, the part id and the
kind, so the same numbers hash the same even if the run directory moved or the
kernel that triangulated them was upgraded. Two materialisers (trimesh today, a
CAD kernel or a DiT adapter tomorrow) must agree on identity for the same
numbers, or a cache keyed on geometry misses forever. A second copy of that
formula in this file is exactly how the two would drift, and the loop's cycle
detection -- which reads `geom.hash` -- would quietly stop noticing edits.

What this module guarantees (contract §7.4)
------------------------------------------
* **Determinism.** Same spec, same bytes. Nothing here reads a clock, a random
  seed or an unordered collection; the STL bytes are compared directly in the
  tests, because a test that only compared hashes would pass even if the files
  differed.
* **Idempotence.** A second call on the same `out_dir` overwrites, keeps the
  same hashes and the same assembly id, and does not raise: resume and retry both
  re-materialise a round that was already written.
* **Layout.** `out_dir/spec.json` plus `out_dir/geometry/<part_id>.stl`. The
  layout is part of the interface (`MaterializedDesign.paths` and `DesignView`
  both assume it), so it does not move.
* **The insertion facts travel with the design.** `mate_axis` and `approach_mm`
  are copied off the `ConnectorGeometry` the builder returned, so a verifier
  reads `view.mate_axis` instead of re-deriving an axis from a kind name. Dovetail
  is why: it slides along +X, and a verifier that assumed +Z would drive the
  tongue through the plate and call a correct design blocked.

Why mesh sanity is checked here
-------------------------------
A connector kind that returns a shell, a scene or an empty boolean result is a
builder bug, and the cheapest place to say so is the boundary where the mesh
becomes evidence. Letting it through would surface as a verifier that "passed" a
hole with no wall, which is the most expensive class of failure the harness has
(`docs/pah-dev-doc-v1.md` §4.8).
"""

from __future__ import annotations

import json
from pathlib import Path

import trimesh

from pah.design.api import (
    MaterializedDesign,
    assembly_from_spec,
    geometry_hash,
)
from pah.design.connectors import builder_for
from pah.design.spec import DesignSpec

__all__ = ["geometry_hash", "materialize"]


def materialize(spec: DesignSpec, out_dir: Path) -> MaterializedDesign:
    """Build `spec` and write the artefacts a verifier and a report read.

    The returned design keeps the meshes it just built, so a cascade can judge the
    same triangles that were written instead of parsing them back; `paths` are
    absolute, because an `Assembly` is consumed from whichever directory the
    episode is running in.
    """
    root = Path(out_dir).resolve()
    geometry_dir = root / "geometry"
    geometry_dir.mkdir(parents=True, exist_ok=True)

    built = builder_for(spec.interface.kind).build(spec)
    _require_solids(spec, built.meshes)

    # `mode="json"` keeps the payload to types `DesignSpec.model_validate` reads
    # back exactly, so a run directory is a complete record of its own design.
    spec_path = root / "spec.json"
    spec_path.write_text(
        json.dumps(spec.model_dump(mode="json"), indent=2, sort_keys=True, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    paths: dict[str, str] = {}
    for part in spec.parts:
        target = geometry_dir / f"{part.id}.stl"
        target.write_bytes(built.meshes[part.id].export(file_type="stl"))
        paths[part.id] = str(target)

    # The axis and the approach are recorded on the assembly as well as returned,
    # so a consumer that only has the assembly (`view_of` in the cascade, a CLI
    # report, a resumed episode) reads the kind's real insertion facts instead of
    # falling back to +Z. Dovetail is why: it slides along +X.
    assembly = assembly_from_spec(
        spec=spec,
        geometry_paths=paths,
        mate_axis=built.mate_axis,
        approach_mm=built.approach_mm,
    )
    return MaterializedDesign(
        spec=spec,
        assembly=assembly,
        meshes=dict(built.meshes),
        paths=paths,
        root=root,
        mate_axis=built.mate_axis,
        approach_mm=built.approach_mm,
    )


def _require_solids(spec: DesignSpec, meshes: dict[str, trimesh.Trimesh]) -> None:
    """Refuse to write a part that is not one closed, positive-volume solid.

    Keys are checked as a set, not one by one: a builder that invents a third
    part or drops one turns every later "which part is this?" into a guess.
    """
    expected = {part.id for part in spec.parts}
    if set(meshes) != expected:
        raise ValueError(
            f"materialiser for {spec.interface.kind!r} returned meshes for "
            f"{sorted(meshes)!r}, but the spec's parts are {sorted(expected)!r}"
        )
    for part in spec.parts:
        mesh = meshes[part.id]
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError(
                f"part {part.id!r} came back as {type(mesh).__name__}, not a Trimesh "
                "(an empty boolean result usually means the feature removed the part)"
            )
        if not mesh.is_watertight:
            raise ValueError(
                f"part {part.id!r} is not watertight after materialisation; a verifier "
                "cannot tell inside from outside on an open shell"
            )
        if not mesh.volume > 0.0:
            raise ValueError(f"part {part.id!r} has non-positive volume {mesh.volume!r}")
