"""The script Blender runs for `pah.tools.fbx`. Not importable outside Blender.

Runs headless: `blender --background --factory-startup --python _blender_fbx.py -- <json>`.
Everything it needs arrives in that one JSON argument, because a Blender subprocess
cannot import `pah` (different interpreter, different site-packages) and a shared
module would be a second copy of the naming rules.

Two directions, because the format gap is symmetric:

* **to** — PAH's STL/GLB/OBJ files -> one multi-object FBX, which is the only
  input `SubmitHunyuan3DPartJob` accepts;
* **from** — one multi-object FBX/OBJ (a decomposition result) -> one STL per
  object, which is what the design plane wants per part.

Object names are preserved in both directions. A decomposition that comes back as
`part_000`, `part_001` must land on disk as `part_000.stl`, `part_001.stl`, or the
bbox/segmentation data that arrived alongside it can never be joined back to a
part.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _args() -> dict:
    argv = sys.argv
    payload = argv[argv.index("--") + 1] if "--" in argv else "{}"
    return json.loads(payload)


def _clear_scene() -> None:
    import bpy

    bpy.ops.wm.read_factory_settings(use_empty=True)


def _mesh_objects() -> list:
    import bpy

    return [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]


def _import_any(path: Path) -> list:
    """Import one file and return the mesh objects it created, renamed to its stem.

    A single-object import is named after the file: `body.stl` becomes `body`.
    A multi-object import (GLB/OBJ scene) keeps its own names, because those names
    *are* the part identity a decomposition is keyed on.
    """
    import bpy

    before = set(bpy.context.scene.objects)
    suffix = path.suffix.lower()
    if suffix == ".stl":
        bpy.ops.wm.stl_import(filepath=str(path))
    elif suffix in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=str(path))
    elif suffix == ".obj":
        bpy.ops.wm.obj_import(filepath=str(path))
    elif suffix == ".ply":
        bpy.ops.wm.ply_import(filepath=str(path))
    else:
        raise SystemExit(f"unsupported input format: {path.name}")

    created = [obj for obj in bpy.context.scene.objects if obj not in before and obj.type == "MESH"]
    if len(created) == 1:
        created[0].name = path.stem
        created[0].data.name = path.stem
    return created


def _bake_world(obj) -> None:
    """Fold the object's world matrix into its mesh data.

    Not cosmetic: some importers (glTF among them) express their conversion as a
    transform on a parent object and leave the mesh local, and the STL exporter
    writes *local* coordinates. Without baking, an object that looks right in the
    scene exports shifted or rotated. After it, every object's mesh data is world
    space, so whichever exporter runs next cannot disagree with the scene.
    """
    import numpy as np

    matrix = np.array(obj.matrix_world, dtype=float)
    if np.allclose(matrix, np.eye(4)):
        return
    obj.data.transform(matrix)
    obj.matrix_world.identity()


def _export_fbx(out: Path, global_scale: float) -> None:
    import bpy

    for obj in _mesh_objects():
        _bake_world(obj)
    out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.fbx(
        filepath=str(out),
        use_selection=False,
        object_types={"MESH"},
        global_scale=global_scale,
        apply_unit_scale=True,
        apply_scale_options="FBX_SCALE_ALL",
        # Blender's own FBX convention; a viewer that disagrees sees a rotated
        # model, not a broken one, so the default is the least surprising choice.
        axis_forward="-Z",
        axis_up="Y",
        mesh_smooth_type="FACE",
        use_mesh_modifiers=True,
        bake_space_transform=False,
    )


def _export_stl(obj, out: Path, global_scale: float) -> None:
    import bpy

    _bake_world(obj)
    out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.wm.stl_export(
        filepath=str(out),
        export_selected_objects=True,
        global_scale=global_scale,
        apply_modifiers=True,
        ascii_format=False,
    )


def main() -> int:
    args = _args()
    mode = args["mode"]
    global_scale = float(args.get("global_scale", 1.0))
    _clear_scene()

    if mode == "to":
        inputs = [Path(p) for p in args["inputs"]]
        for path in inputs:
            if not path.exists():
                raise SystemExit(f"input does not exist: {path}")
            _import_any(path)
        names = [obj.name for obj in _mesh_objects()]
        if not names:
            raise SystemExit("nothing to export: no mesh objects imported")
        _export_fbx(Path(args["out"]), global_scale)
        print(json.dumps({"ok": True, "objects": names, "out": args["out"]}))
        return 0

    if mode == "from":
        source = Path(args["source"])
        suffix = source.suffix.lower()
        if suffix in (".fbx",):
            import bpy

            bpy.ops.import_scene.fbx(filepath=str(source))
        elif suffix in (".glb", ".gltf", ".obj", ".stl", ".ply"):
            _import_any(source)
        else:
            raise SystemExit(f"unsupported input format: {source.name}")

        out_dir = Path(args["out_dir"])
        written = []
        for obj in _mesh_objects():
            target = out_dir / f"{obj.name}.stl"
            _export_stl(obj, target, global_scale)
            written.append({"name": obj.name, "stl": str(target)})
        if not written:
            raise SystemExit("the file carries no mesh objects")
        print(json.dumps({"ok": True, "parts": written}))
        return 0

    raise SystemExit(f"unknown mode {mode!r}")


if __name__ == "__main__":
    raise SystemExit(main())
