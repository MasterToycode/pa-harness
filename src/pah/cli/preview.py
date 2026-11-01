"""`pah preview` — turn a run's STLs into something you can actually look at.

Two artifacts, because they answer two different questions:

* `preview.png` — four orthographic/iso views rendered with matplotlib, so the part
  can be checked in a terminal-era workflow, a paper figure, or a chat message. No
  browser, no GPU, no extra dependency beyond matplotlib.
* `preview.html` — a single self-contained page with a rotate/zoom viewer. The STLs
  are embedded as `data:` URIs, so the file works by double-clicking it, offline,
  with no server and no CORS problem.

Shading is per-face from the triangle normal against a fixed light: enough to read the
shape, deliberately not a renderer. Faces are subsampled when a part is dense — a
60 k-face generated insert would otherwise make the PNG take minutes.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import trimesh

__all__ = ["render_html", "render_png", "preview_run"]

#: Faces per part drawn in the PNG. Above this the image is subsampled: the point is
#: silhouette and shading, not every triangle.
MAX_FACES_PNG = 4000

#: A fixed key light, so two runs of the same part render identically.
_LIGHT = np.array([0.35, -0.55, 0.75])
_LIGHT = _LIGHT / np.linalg.norm(_LIGHT)


def _load_parts(run_dir: Path) -> dict[str, trimesh.Trimesh]:
    geometry = run_dir / "geometry"
    if not geometry.exists():
        raise FileNotFoundError(f"{run_dir} has no geometry/ directory")
    parts: dict[str, trimesh.Trimesh] = {}
    for path in sorted(geometry.glob("*.stl")):
        mesh = trimesh.load(path, force="mesh")
        if isinstance(mesh, trimesh.Trimesh) and len(mesh.faces):
            parts[path.stem] = mesh
    if not parts:
        raise FileNotFoundError(f"{geometry} contains no readable STL")
    return parts


def _sample_faces(mesh: trimesh.Trimesh, limit: int) -> np.ndarray:
    faces = mesh.faces
    if len(faces) <= limit:
        return faces
    step = max(1, len(faces) // limit)
    return faces[::step][:limit]


def _shade(mesh: trimesh.Trimesh, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Triangle centres and per-face grey levels for the sampled faces."""
    triangles = mesh.vertices[faces]
    centres = triangles.mean(axis=1)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    lengths[lengths == 0.0] = 1.0
    normals = normals / lengths[:, None]
    intensity = 0.35 + 0.65 * np.clip(normals @ _LIGHT, 0.0, 1.0)
    return centres, intensity


def _camera_basis(elev_deg: float, azim_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Camera right/up/depth axes for a view, from a spherical camera position.

    Built with a look-at basis rather than Euler rotations: the first attempt used
    Euler angles and the top view projected the z axis into the screen, so the plate
    drifted away from the insert. These axes work at the poles by construction.
    """
    elev = np.radians(elev_deg)
    azim = np.radians(azim_deg)
    position = np.array(
        [np.cos(elev) * np.cos(azim), np.cos(elev) * np.sin(azim), np.sin(elev)]
    )
    forward = -position / np.linalg.norm(position)  # where the camera looks
    up_hint = np.array([0.0, 0.0, 1.0])
    if abs(float(forward @ up_hint)) > 0.999:
        up_hint = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, up_hint)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    return right, up, -forward  # depth grows toward the camera


def _decimate(mesh: trimesh.Trimesh, limit: int) -> trimesh.Trimesh:
    """A closed approximation with at most ~`limit` faces.

    Sampling every n-th *triangle* leaves gaps: a 64 k-face generated insert became a
    speckled ghost in the first render. Decimation keeps the surface closed, which is
    what shading and depth need.
    """
    if len(mesh.faces) <= limit:
        return mesh
    try:
        return mesh.simplify_quadric_decimation(face_count=limit)
    except Exception:  # noqa: BLE001 - a failed decimation must not lose the render
        return mesh


def _rasterize(
    parts: dict[str, trimesh.Trimesh],
    colours: list[tuple[int, int, int]],
    *,
    elev: float,
    azim: float,
    size: int,
    span: float,
    centre: np.ndarray,
) -> np.ndarray:
    """A z-buffered orthographic render: one RGB array, no 3D library.

    matplotlib's `Poly3DCollection` sorts whole collections by depth, so a plate's
    top face can be drawn over the near wall and the part reads as a crossed slab
    (observed in the first preview). A per-pixel depth buffer has no ordering problem.
    """
    right, up, depth_axis = _camera_basis(elev, azim)
    pixels = np.zeros((size, size, 3), dtype=np.float32)
    depth = np.full((size, size), -np.inf, dtype=np.float32)
    scale = (size * 0.86) / max(span, 1e-9)
    light = np.array([0.35, -0.55, 0.75])
    light = light / np.linalg.norm(light)

    for order, (_, mesh) in enumerate(parts.items()):
        colour = np.array(colours[order % len(colours)], dtype=np.float32) / 255.0
        source = _decimate(mesh, MAX_FACES_PNG)
        triangles = source.vertices[source.faces] - centre
        sx = triangles @ right
        sy = triangles @ up
        sz = triangles @ depth_axis
        screen = np.stack([sx * scale + size / 2.0, size / 2.0 - sy * scale], axis=2)
        normal = np.cross(
            triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
        )
        lengths = np.linalg.norm(normal, axis=1)
        lengths[lengths == 0.0] = 1.0
        normal = normal / lengths[:, None]
        intensity = 0.30 + 0.70 * np.abs(normal @ light)

        for index in range(len(source.faces)):
            x0, y0 = screen[index].min(axis=0)
            x1, y1 = screen[index].max(axis=0)
            ix0, ix1 = max(int(np.floor(x0)), 0), min(int(np.ceil(x1)), size - 1)
            iy0, iy1 = max(int(np.floor(y0)), 0), min(int(np.ceil(y1)), size - 1)
            if ix1 < ix0 or iy1 < iy0:
                continue
            xs = np.arange(ix0, ix1 + 1) + 0.5
            ys = np.arange(iy0, iy1 + 1) + 0.5
            grid_x, grid_y = np.meshgrid(xs, ys)
            ax, ay = screen[index, 0]
            bx, by = screen[index, 1]
            cx, cy = screen[index, 2]
            denominator = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
            if abs(denominator) < 1e-12:
                continue
            w0 = ((by - cy) * (grid_x - cx) + (cx - bx) * (grid_y - cy)) / denominator
            w1 = ((cy - ay) * (grid_x - cx) + (ax - cx) * (grid_y - cy)) / denominator
            w2 = 1.0 - w0 - w1
            inside = (w0 >= 0.0) & (w1 >= 0.0) & (w2 >= 0.0)
            if not inside.any():
                continue
            z = w0 * sz[index, 0] + w1 * sz[index, 1] + w2 * sz[index, 2]
            window = depth[iy0 : iy1 + 1, ix0 : ix1 + 1]
            closer = inside & (z > window)
            if not closer.any():
                continue
            window[closer] = z[closer]
            shade = colour * float(intensity[index])
            pixels[iy0 : iy1 + 1, ix0 : ix1 + 1][closer] = np.clip(shade, 0.0, 1.0)
    return (pixels * 255.0).astype(np.uint8)


def render_png(run_dir: str | Path, path: str | Path | None = None) -> Path:
    """Four-view shaded render of every part in `run_dir/geometry`."""
    from PIL import Image, ImageDraw  # noqa: PLC0415 - only the render path pays for it

    run = Path(run_dir)
    parts = _load_parts(run)
    target = Path(path) if path is not None else run / "preview.png"

    colours = [(122, 166, 194), (194, 163, 122), (143, 191, 143), (180, 143, 191)]
    low = np.min([m.bounds[0] for m in parts.values()], axis=0)
    high = np.max([m.bounds[1] for m in parts.values()], axis=0)
    centre = (low + high) / 2.0
    span = float(np.max(high - low)) or 1.0

    tile = 560
    gap = 8
    views = [
        (22.0, -60.0, "isometric"),
        (0.0, -90.0, "front"),
        (0.0, 0.0, "side"),
        (90.0, -90.0, "top"),
    ]
    width = tile * 2 + gap * 3
    height = tile * 2 + gap * 3 + 34
    canvas = Image.new("RGB", (width, height), (16, 21, 28))
    draw = ImageDraw.Draw(canvas)
    draw.text((gap + 4, 10), f"PAH preview · {run.name}", fill=(223, 231, 240))
    for index, (elev, azim, title) in enumerate(views):
        tile_pixels = _rasterize(
            parts, colours, elev=elev, azim=azim, size=tile, span=span, centre=centre
        )
        column = index % 2
        row = index // 2
        left = gap + column * (tile + gap)
        top = 34 + gap + row * (tile + gap)
        canvas.paste(Image.fromarray(tile_pixels), (left, top))
        draw.text((left + 6, top + 6), title, fill=(139, 160, 181))
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target)
    return target


_VIEWER = """<!doctype html>
<meta charset="utf-8">
<title>PAH · {name}</title>
<style>
  body {{ margin:0; font-family: ui-sans-serif, system-ui, sans-serif; background:#10151c; color:#dfe7f0; }}
  header {{ padding:12px 18px; border-bottom:1px solid #22303f; display:flex; gap:18px; align-items:baseline; }}
  h1 {{ font-size:15px; margin:0; font-weight:600; }}
  .meta {{ font-size:12px; color:#8ba0b5; }}
  #view {{ position:fixed; inset:52px 0 0 0; }}
  .hint {{ position:fixed; left:18px; bottom:14px; font-size:12px; color:#7f93a8; }}
</style>
<header>
  <h1>{name}</h1>
  <span class="meta">{parts} · 鼠标左键旋转 / 滚轮缩放 / 右键平移</span>
</header>
<div id="view"></div>
<div class="hint">拖动即可看整体；这是预览页，几何文件在 run 目录的 geometry/ 下。</div>
<script type="importmap">
{{ "imports": {{
  "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
  "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
}} }}
</script>
<script type="module">
import * as THREE from "three";
import {{ OrbitControls }} from "three/addons/controls/OrbitControls.js";
import {{ STLLoader }} from "three/addons/loaders/STLLoader.js";

const parts = {payload};
const el = document.getElementById("view");
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x10151c);
const camera = new THREE.PerspectiveCamera(45, innerWidth / innerHeight, 0.1, 5000);
camera.position.set(70, -90, 60);
const renderer = new THREE.WebGLRenderer({{ antialias: true }});
renderer.setSize(innerWidth, innerHeight);
el.appendChild(renderer.domElement);
const controls = new OrbitControls(camera, renderer.domElement);
scene.add(new THREE.AmbientLight(0xffffff, 0.55));
const key = new THREE.DirectionalLight(0xffffff, 1.15); key.position.set(1, -1, 1.4); scene.add(key);
const rim = new THREE.DirectionalLight(0x88aaff, 0.5); rim.position.set(-1, 1, -0.6); scene.add(rim);
const colours = [0x7aa6c2, 0xc2a37a, 0x8fbf8f, 0xb48fbf];
const loader = new STLLoader();
const group = new THREE.Group();
parts.forEach((part, i) => {{
  const binary = Uint8Array.from(atob(part.stl), c => c.charCodeAt(0)).buffer;
  const geometry = loader.parse(binary);
  geometry.computeVertexNormals();
  const material = new THREE.MeshStandardMaterial({{
    color: colours[i % colours.length], metalness: 0.15, roughness: 0.55, flatShading: false
  }});
  group.add(new THREE.Mesh(geometry, material));
}});
scene.add(group);
const box = new THREE.Box3().setFromObject(group);
const size = box.getSize(new THREE.Vector3()).length();
group.position.sub(box.getCenter(new THREE.Vector3()));
camera.position.set(size * 0.9, -size * 1.1, size * 0.8);
controls.target.set(0, 0, 0);
controls.update();
addEventListener("resize", () => {{
  camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
}});
(function loop() {{ requestAnimationFrame(loop); controls.update(); renderer.render(scene, camera); }})();
</script>
"""


def render_html(
    run_dir: str | Path, path: str | Path | None = None, *, max_faces: int = 60000
) -> Path:
    """A single-file viewer with the STLs embedded, so it opens offline."""
    run = Path(run_dir)
    parts = _load_parts(run)
    payload = []
    for part_id, mesh in parts.items():
        simplified = mesh
        if len(mesh.faces) > max_faces:
            try:  # trimesh 5 wants an explicit `face_count`, not a reduction
                simplified = mesh.simplify_quadric_decimation(face_count=max_faces)
            except Exception:  # noqa: BLE001 - a failed decimation must not lose the view
                simplified = mesh
        payload.append(
            {"id": part_id, "stl": base64.b64encode(simplified.export(file_type="stl")).decode("ascii")}
        )
    target = Path(path) if path is not None else run / "preview.html"
    target.write_text(
        _VIEWER.format(
            name=run.name,
            parts=" + ".join(f"{p['id']} ({len(parts[p['id']].faces)} faces)" for p in payload),
            payload=json.dumps(payload),
        ),
        "utf-8",
    )
    return target


def preview_run(run_dir: str | Path) -> tuple[Path, Path]:
    """Both artifacts for one run directory."""
    run = Path(run_dir)
    return render_png(run), render_html(run)


# --------------------------------------------------------------------------- #
# `pah preview <run>` — the CLI wrapper
# --------------------------------------------------------------------------- #
def add_parser(groups: "argparse._SubParsersAction") -> None:
    import argparse  # noqa: PLC0415 - only needed when mounting

    preview = groups.add_parser("preview", help="render a run's STLs into PNG + HTML")
    preview.add_argument("run", help="a run directory that contains geometry/")
    preview.add_argument("--out", default=None, help="PNG path (default <run>/preview.png)")
    preview.add_argument("--open", action="store_true", help="open the 3D view in a browser")
    preview.add_argument("--json", action="store_true")
    preview.set_defaults(handler=_preview)


def _preview(args: "argparse.Namespace") -> int:
    import json  # noqa: PLC0415
    import webbrowser  # noqa: PLC0415
    from pathlib import Path as _Path  # noqa: PLC0415

    run = _Path(args.run)
    try:
        png = render_png(run, args.out)
        html = render_html(run)
    except FileNotFoundError as error:
        print(f"pah preview: {error}")
        return 1
    payload = {"run": str(run), "png": str(png), "html": str(html)}
    print(json.dumps(payload, ensure_ascii=False, indent=1) if args.json else
          f"png   {png}\nhtml  {html}\n\n双击 html 可旋转缩放；png 是四视图。")
    if args.open:
        webbrowser.open(html.resolve().as_uri())
    return 0
