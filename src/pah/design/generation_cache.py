"""Generated geometry on disk, addressed by the request that produced it.

Why this exists
---------------
The cascade caches materialisations **in memory**, keyed by the generation request
(`design_cascade._materialise`). That is enough inside one process and useless across
two: a resumed episode builds a fresh cascade, finds an empty cache, and calls the
generator again — a second paid sample of the same request, and one the report cannot
distinguish from the first. The audit measured the shape of it (§5.4): `dit_materializer`
documents "resume re-reads the STL, it does not re-run the model", while the runtime
does exactly the opposite for generative adapters.

So the artefact is written next to the run and addressed by the same key:

    <out>/generation/<key>/manifest.json
    <out>/generation/<key>/<part_id>.stl

`load` returns a `MaterializedDesign` for a key that is already there; `save` writes one.
A resumed run therefore pays for a request once, per run directory, no matter how many
processes touch it.

What this is not
----------------
It is not a claim that generation is reproducible. The ledger stores the `output_hash`
of what came back, so two different samples of the same request are *distinguishable*;
only a generator that declares `non_deterministic = False` can claim the same request
yields the same mesh. X-Part and Hunyuan both sample, so their reports say
`deterministic_generator: false` and the eval has to stratify on it.
"""

from __future__ import annotations

import json
from pathlib import Path

import trimesh

__all__ = ["GENERATION_DIR", "load", "save"]

GENERATION_DIR = "generation"


def _key_dir(root: Path, key: str) -> Path:
    # The key is a `sha256:...` string; the colon is not portable in a path component.
    return Path(root) / GENERATION_DIR / key.replace(":", "-")


def load(root: Path, key: str):
    """A `MaterializedDesign` for `key`, or `None` when this run has not seen it.

    A missing or unreadable manifest is a cache miss, not an error: the caller then
    asks the generator, which is the expensive-but-correct path. A manifest whose STL
    is gone is also a miss — half a ledger is worse than none.
    """
    from pah.design.api import MaterializedDesign, assembly_from_spec

    directory = _key_dir(Path(root), key)
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None

    paths = manifest.get("paths") or {}
    meshes: dict[str, trimesh.Trimesh] = {}
    for part_id, path in paths.items():
        location = Path(path)
        if not location.exists():
            return None
        meshes[part_id] = trimesh.load(location, force="mesh")
    if not meshes:
        return None

    from pah.design.spec import DesignSpec

    spec = DesignSpec.model_validate(manifest["spec"])
    assembly = assembly_from_spec(
        spec=spec,
        geometry_paths=paths,
        mate_axis=manifest.get("mate_axis"),
        approach_mm=manifest.get("approach_mm"),
        interfaces=manifest.get("interfaces") or None,
    )
    generation = manifest.get("generation")
    if generation:
        from dataclasses import replace

        assembly = replace(
            assembly, constraints={**(assembly.constraints or {}), "generation": generation}
        )
    return MaterializedDesign(
        spec=spec,
        assembly=assembly,
        meshes=meshes,
        paths=paths,
        root=Path(root),
        mate_axis=tuple(manifest.get("mate_axis") or (0.0, 0.0, 1.0)),
        approach_mm=float(manifest.get("approach_mm") or 6.0),
        interfaces=manifest.get("interfaces") or {},
    )


def save(root: Path, key: str, design) -> Path:
    """Persist `design` under `key`; returns the directory.

    Written through a temporary directory and renamed, so a crash mid-write cannot
    leave a manifest that points at half the parts — the same rule the report effect
    follows, for the same reason.
    """
    directory = _key_dir(Path(root), key)
    staging = directory.with_name(directory.name + ".tmp")
    if staging.exists():
        for stale in staging.glob("*"):
            stale.unlink()
        staging.rmdir()
    staging.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for part_id, mesh in design.meshes.items():
        target = staging / f"{part_id}.stl"
        target.write_bytes(mesh.export(file_type="stl"))
        paths[part_id] = str(directory / target.name)  # final path, not the staging one
    generation = (design.assembly.constraints or {}).get("generation")
    manifest = {
        "schema": "pah-generation/1",
        "key": key,
        "spec": design.spec.model_dump(mode="json"),
        "paths": paths,
        "mate_axis": list(design.mate_axis),
        "approach_mm": float(design.approach_mm),
        "interfaces": design.interfaces or {},
        "generation": generation,
    }
    (staging / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", "utf-8"
    )
    if directory.exists():
        for stale in directory.glob("*"):
            stale.unlink()
        directory.rmdir()
    staging.rename(directory)
    return directory
