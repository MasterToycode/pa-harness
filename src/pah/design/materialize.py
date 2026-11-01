"""Geometry materialisation: a `DesignSpec` in, one solid per part plus an `Assembly` out.

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

    if any(part.source for part in spec.parts):
        return _materialize_decomposed(spec, root, geometry_dir)

    _require_buildable(spec)
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


def _materialize_decomposed(
    spec: DesignSpec, root: Path, geometry_dir: Path
) -> MaterializedDesign:
    """Machine every interface into the decomposed blanks and write the run.

    The blanks come from `<root>/parts/`, hash-checked against `PartSpec.source`.
    Each interface then cuts a socket into its host and grows a peg on its mating
    part — sequentially, because a part in the middle of a chain carries both, and
    a boolean per interface is the only order-independent way to get there.

    The per-interface frames are recorded on the assembly, not just returned: a
    verifier looking at one pair needs *that pair's* axis. With one global axis,
    every pair in a chain would be swept in the same direction, and a joint whose
    normal points elsewhere would be reported blocked.
    """
    from pah.design.decompose import check_store
    from pah.design.machining import add_peg, cut_socket
    from pah.design.pairing import contact_of

    parametric = sorted(part.id for part in spec.parts if not part.source)
    if parametric:
        raise ValueError(
            f"parts {parametric!r} have no `source`, so they are not decomposed "
            "blanks; a spec may not mix decomposed parts with parametric ones"
        )

    blanks = check_store(spec, root)
    meshes = {
        part.id: _as_solid(trimesh.load(blanks[part.id], force="mesh"), part.id)
        for part in spec.parts
    }

    facts: dict[str, dict] = {}
    for interface in spec.interfaces:
        host = spec.part(interface.a)
        guest = spec.part(interface.b)
        contact = contact_of(host, guest)
        if contact is None:
            raise ValueError(
                f"interface {interface.id!r} is between {host.id!r} and {guest.id!r}, "
                "but their envelopes are no longer adjacent; the spec's poses and the "
                "blanks disagree, which means the spec was edited after decomposition"
            )
        axis, point = contact.axis, contact.point_mm
        params = interface.validated_params()
        meshes[host.id] = cut_socket(
            meshes[host.id],
            axis=axis,
            point=point,
            d_hole_mm=params.d_hole_mm,
            depth_mm=params.engage_mm,
        )
        meshes[guest.id] = add_peg(
            meshes[guest.id],
            axis=axis,
            point=point,
            d_peg_mm=params.d_peg_mm,
            engage_mm=params.engage_mm,
        )
        facts[interface.id] = {
            "mate_axis": list(axis),
            "approach_mm": float(params.approach_mm),
        }

    _require_solids(spec, meshes)

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
        target.write_bytes(meshes[part.id].export(file_type="stl"))
        paths[part.id] = str(target)

    assembly = assembly_from_spec(
        spec=spec,
        geometry_paths=paths,
        interfaces=facts,
    )
    first = next(iter(facts.values()), None) or {}
    return MaterializedDesign(
        spec=spec,
        assembly=assembly,
        meshes=dict(meshes),
        paths=paths,
        root=root,
        mate_axis=tuple(first.get("mate_axis", (0.0, 0.0, 1.0))),
        approach_mm=float(first.get("approach_mm", 6.0)),
        interfaces=facts,
    )


def _as_solid(mesh: trimesh.Trimesh, part_id: str) -> trimesh.Trimesh:
    """A loaded blank as one closed solid, or a refusal naming the part."""
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"blank for part {part_id!r} is a {type(mesh).__name__}, not a mesh")
    if not mesh.is_watertight or mesh.volume <= 0.0:
        raise ValueError(
            f"blank for part {part_id!r} is not a closed solid "
            f"(watertight={mesh.is_watertight}, volume={mesh.volume!r})"
        )
    return mesh


def _require_buildable(spec: DesignSpec) -> None:
    """Refuse, with the reason, the shapes this materialiser cannot build yet.

    `ConnectorBuilder.build` is a **two-part** port: it returns one solid per part
    from one interface. A spec can now hold N parts and M interfaces (a
    decomposition returns many pieces), and building those needs the feature applied
    per interface onto a shared blank — the piece of work that is not written yet.

    Saying so here, by name and with the count, beats the alternative: letting it
    fall through would surface as `spec.interface`'s "use `interfaces`" message,
    which reads like a caller mistake and sends the reader to the wrong file. A
    single-part spec lands here too, because there is no interface to take a kind
    from.
    """
    if len(spec.parts) == 1 and not spec.interfaces:
        raise NotImplementedError(
            f"spec {spec.spec_id!r} has one part and no interfaces; this materialiser "
            "builds connected assemblies (it needs a connector kind to know what to "
            "cut). A standalone part needs a mesh materialiser, not a blank."
        )
    if len(spec.interfaces) > 1:
        raise NotImplementedError(
            f"spec {spec.spec_id!r} has {len(spec.parts)} parts and "
            f"{len(spec.interfaces)} interfaces; the trimesh materialiser builds the "
            "two-part, one-interface case. N-part machining (apply each interface's "
            "feature onto the shared blank of both parts, then union per part) is not "
            "written yet — `docs/architecture.md` tracks it as the multi-part step."
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
