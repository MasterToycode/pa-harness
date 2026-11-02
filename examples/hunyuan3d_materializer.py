"""Hunyuan3D (腾讯混元生3D) as PAH's geometry generator — paste two keys, done.

Tencent Cloud does not use one OpenAI-style `sk-...` key: it uses **SecretId +
SecretKey** and a TC3-HMAC-SHA256 signature, version `2025-05-13`, endpoint
`ai3d.tencentcloudapi.com` (文本/图片生成 3D 任务，异步: Submit → Query → 下载)。
This adapter carries that protocol so the harness only sees
`materialize(spec, out_dir) -> MaterializedDesign`.

Where the keys come from — this process's environment, never this module. `pah`
fills that environment once at startup from `D:\\yanyi\\pa-harness\\.env`（已
gitignore）:

    TENCENTCLOUD_SECRET_ID=AKID...
    TENCENTCLOUD_SECRET_KEY=...
    TENCENTCLOUD_REGION=ap-guangzhou

**This file must not read `.env` itself.** It used to, at import time, and that is
exactly how a read-only audit script spent real money: the import loaded credentials
the run had never been authorised to use, `pah.toml` resolved `hunyuan:python`, and
one acceptance check called the paid service. An import must be free of side effects;
credentials arrive from `pah`'s one loader (`pah.app.env`), or from an explicit
export. A bare `python examples/hunyuan3d_materializer.py` has neither, so export the
two variables before running it directly.

拿钥匙的地方：腾讯云控制台 → 访问管理 CAM → API 密钥管理 → 新建密钥；并在
[混元生3D 控制台](https://console.cloud.tencent.com/ai3d) 开通服务（按任务计费）。

Wire it up in `pah.toml` (and declare the billing unit — that is what the paid gate
reads):

    [gen]
    default = "hunyuan"

    [gen.providers.hunyuan]
    kind = "python"
    billing = "per_call"
    compat = { target = "examples/hunyuan3d_materializer.py:materialize" }

Without `--allow-paid-gen` (or `PAH_ALLOW_PAID_GEN=1`), `pah run` / `pah build` /
`pah eval` refuse this provider and say why — configuration alone cannot spend money.

Then, in order (each step is cheap and fails for one reason only):

    python examples/hunyuan3d_materializer.py --check       # SDK/keys 就绪了吗（不花钱，不读 .env）
    pah providers doctor --role gen                         # 同一项检查，经统一入口读 .env
    pah providers doctor --role gen --call --allow-paid-gen # 真生成一次（会花钱，见下）
    pah run --spec fixtures/design/demo-peg-hole-interfere.json \\
            --policy rule --rounds 3 --out runs/hunyuan --allow-paid-gen

Two honesty rules this file keeps
--------------------------------
1. **没有 key 就抛错，不退回参数化几何。** report 的 `run.generator` 会写 `hunyuan:python`，
   如果实际是 trimesh 造的，这个字段就是假的。
2. **接口尺寸必须由参数决定，不由模型决定。** 生成模型给不了「孔径 10.0 mm」；所以本体
   （带孔/槽的 body）仍然参数化，模型只生成 insert 的自由形状，然后按 `d_peg` 做**等比**缩放
   并对齐到 seated pose。缩放是形状保持的，非等比会把销子压扁。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np
import trimesh

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from pah.design.api import MaterializedDesign, assembly_from_spec  # noqa: E402
from pah.design.connectors import builder_for  # noqa: E402
from pah.design.spec import DesignSpec  # noqa: E402

# Importing this module reads no `.env`, mutates no environment variable and opens no
# socket. `pah` has already loaded credentials by the time an adapter is resolved, and
# an import that loads them again is how the paid API got called by accident. The
# proxy fix therefore runs at call time (`_generate`), not here.


def _bypass_proxy_for_service() -> None:
    """Keep Tencent Cloud off this machine's dev proxy.

    `HTTPS_PROXY` is set globally here (127.0.0.1:7890, a foreign-network proxy) and
    its `NO_PROXY` only lists DeepSeek. Tencent's endpoint is domestic, so proxying it
    makes the TLS connection reset (`ConnectionResetError 10054`) before any request
    is signed — which looks like a credential problem and is not one. This appends the
    service's domains to `NO_PROXY` for **this process only**: an explicit proxy for
    those hosts still wins if you set one, because the list is only extended.

    Applies to the download host as well: generated files come from COS
    (`myqcloud.com`), and a proxied download fails the same way.
    """
    hosts = ("tencentcloudapi.com", "tencentcloud.com", "myqcloud.com")
    current = os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""
    entries = [part.strip() for part in current.split(",") if part.strip()]
    for host in hosts:
        for pattern in (host, f"*.{host}"):
            if pattern not in entries:
                entries.append(pattern)
    joined = ",".join(entries)
    os.environ["NO_PROXY"] = joined
    os.environ["no_proxy"] = joined


SECRET_ID_ENV = "TENCENTCLOUD_SECRET_ID"
SECRET_KEY_ENV = "TENCENTCLOUD_SECRET_KEY"
REGION_ENV = "TENCENTCLOUD_REGION"
DEFAULT_REGION = "ap-guangzhou"

#: 提交/查询用的动作名。SDK 里真实存在的只有两组（没有 `SubmitHunyuanTo3DJob`，
#: 那是按文档猜的、会被 SDK 拒绝）：极速版 `...RapidJob` 与专业版 `...ProJob`。
#: 模型版本再通过请求里的 `Model` 参数选——**每个档位支持的版本不同**，例如专业版
#: 只接受 `[3.0, 3.1]`，写 2.5 会被参数校验拒掉（`InvalidResultFormat`）。
TIERS: dict[str, tuple[str, str]] = {
    "rapid": ("SubmitHunyuanTo3DRapidJob", "QueryHunyuanTo3DRapidJob"),
    "pro": ("SubmitHunyuanTo3DProJob", "QueryHunyuanTo3DProJob"),
}
DEFAULT_TIER = "rapid"
TIER_ENV = "HUNYUAN3D_TIER"
MODEL_ENV = "HUNYUAN3D_MODEL"
API_VERSION = "2025-05-13"

#: 轮询上限：极速版通常 10-60 s；专业版可能几分钟。超过就报错，不无限等。
POLL_TIMEOUT_S = 300
POLL_INTERVAL_S = 3.0


# --------------------------------------------------------------------------- #
# the port
# --------------------------------------------------------------------------- #
def materialize(spec: DesignSpec, out_dir: str | Path) -> MaterializedDesign:
    """Spec in, two solids out: body parametric, insert generated by Hunyuan3D."""
    out = Path(out_dir)
    builder = builder_for(spec.interface.kind)
    geometry = builder.build(spec)
    meshes = dict(geometry.meshes)

    insert_id = spec.insert.id
    prompt = _prompt_for(spec)
    generated = _generate(prompt)
    meshes[insert_id] = _fit_to_interface(generated, reference=meshes[insert_id], spec=spec)

    if not meshes[insert_id].is_watertight:
        raise ValueError(
            "hunyuan3d returned a mesh that could not be made watertight after repair; "
            "a shell with no closed interior cannot be measured by the verifiers"
        )

    out.mkdir(parents=True, exist_ok=True)
    geometry_dir = out / "geometry"
    geometry_dir.mkdir(exist_ok=True)
    paths: dict[str, str] = {}
    for part_id, mesh in meshes.items():
        path = geometry_dir / f"{part_id}.stl"
        mesh.export(path)
        paths[part_id] = str(path.resolve())

    (out / "spec.json").write_text(
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
        root=out,
        mate_axis=geometry.mate_axis,
        approach_mm=geometry.approach_mm,
    )


# The four facts a *generative* materialiser has to declare (pah.design.api:
# materializer_identity). Without them a hint-only repair reads as "no change" and
# the loop calls it a geometry cycle instead of asking the model again — audit P1-4.
materialize.generator_id = "hunyuan3d"  # type: ignore[attr-defined]
materialize.supports_prompt_hint = True  # type: ignore[attr-defined]
materialize.non_deterministic = True  # type: ignore[attr-defined]
materialize.model_version = (  # type: ignore[attr-defined]
    os.environ.get(MODEL_ENV, "").strip() or os.environ.get(TIER_ENV, "").strip() or None
)


# --------------------------------------------------------------------------- #
# credentials + the service
# --------------------------------------------------------------------------- #
def _credentials() -> tuple[str, str, str]:
    secret_id = os.environ.get(SECRET_ID_ENV, "").strip()
    secret_key = os.environ.get(SECRET_KEY_ENV, "").strip()
    region = os.environ.get(REGION_ENV, "").strip() or DEFAULT_REGION
    if not secret_id or not secret_key:
        raise RuntimeError(
            f"Hunyuan3D needs ${SECRET_ID_ENV} and ${SECRET_KEY_ENV} in this process. "
            "`pah` loads them from D:\\yanyi\\pa-harness\\.env at startup, so run "
            "through `pah providers doctor --role gen` (or `pah run --allow-paid-gen`); "
            "a bare `python examples/hunyuan3d_materializer.py` must export them first, "
            "because this module deliberately does not read `.env` itself."
        )
    return secret_id, secret_key, region


def _client():
    """The official SDK client. A missing package is one sentence, not a traceback."""
    secret_id, secret_key, region = _credentials()
    try:
        from tencentcloud.ai3d.v20250513 import ai3d_client  # noqa: PLC0415
        from tencentcloud.common import credential  # noqa: PLC0415
        from tencentcloud.common.profile.client_profile import ClientProfile  # noqa: PLC0415
        from tencentcloud.common.profile.http_profile import HttpProfile  # noqa: PLC0415
    except ImportError as error:  # the package is optional until you use this adapter
        raise RuntimeError(
            "缺少腾讯云 SDK：请先 `pip install tencentcloud-sdk-python`。"
            f"（{type(error).__name__}: {error}）"
        ) from error
    http = HttpProfile()
    http.endpoint = "ai3d.tencentcloudapi.com"
    profile = ClientProfile()
    profile.httpProfile = http
    return ai3d_client.Ai3dClient(credential.Credential(secret_id, secret_key), region, profile)


def _prompt_for(spec: DesignSpec) -> str:
    """What we ask the model for: the insert's free shape, never the mating feature.

    接口尺寸由 spec 决定（`_fit_to_interface` 会等比缩放到 `d_peg_mm`），所以提示词里
    不写毫米数——写了也没用，模型不听；写了反而让人以为模型能保证尺寸。
    """
    kind = spec.interface.kind
    shape = {
        "peg_hole": "a straight cylindrical pin with a slightly chamfered tip",
        "snap_fit": "a cantilever snap hook with a rounded lead-in",
        "dovetail": "a dovetail tongue with a small handle block",
    }.get(kind, "a mechanical insert")
    # A decision (rule or LLM) can send a material instruction through the design's
    # provenance — this is the one channel by which an agent guides the generator
    # without ever touching geometry. Measured reason: the first generated pin was a
    # hollow tube, and no numeric parameter was wrong.
    hint = str((spec.provenance or {}).get("prompt_hint") or "").strip()
    hint_clause = f" {hint}." if hint else ""
    return (
        f"a single 3D printable mechanical part: {shape}. "
        f"Purpose: {spec.requirement or 'join two parts'}. "
        "Centered on the origin, upright along +Z, no text, no base, no support "
        "structures. "
        # The requirement text often carries numbers ("9.7 mm peg in a 10.0 mm
        # socket"). A generative model cannot hit a tolerance, and a prompt that
        # asks it to would only produce a shape that fails the fit verifier for a
        # reason nobody can act on. The fit is applied *after* generation
        # (`_fit_to_interface`), so say so here instead of pretending.
        "Ignore any numeric tolerances in that description: the mating fit is "
        "applied after generation, so only the overall shape matters."
        f"{hint_clause}"
    )


def _tier() -> str:
    """Which action pair to call. `pro` is highest precision, `rapid` fastest."""
    tier = os.environ.get(TIER_ENV, "").strip() or DEFAULT_TIER
    if tier not in TIERS:
        raise RuntimeError(
            f"{TIER_ENV}={tier!r} is not one of {sorted(TIERS)}; "
            "standard|rapid|pro select different action pairs on the same service"
        )
    return tier


def _model() -> str:
    """The model version, if pinned. Empty means "let the service pick its default".

    This is the answer to "which model am I integrating": Hunyuan3D's API chooses the
    *version* through this parameter (`HY-3D-2.5`, `HY-3D-3.0`, `HY-3D-3.1`, ...)
    while the *tier* chooses the action pair. Leaving it empty means a report cannot
    claim which version produced a part, so a run that matters should pin it.
    """
    return os.environ.get(MODEL_ENV, "").strip()


def _submit_payload(prompt: str, *, model: str = "", image_url: str | None = None) -> dict:
    """The submit body, kept pure so a test can assert what we would send.

    `Model` is only included when pinned: sending an empty string would ask the
    service to look up a model called "", which is a different request than "default".
    """
    payload: dict[str, object] = {"ResultFormat": "GLB"}
    if image_url:
        payload["ImageUrl"] = image_url
    else:
        payload["Prompt"] = prompt
    if model:
        payload["Model"] = model
    return payload


def _generate(prompt: str, *, image_url: str | None = None) -> trimesh.Trimesh:
    """Submit → poll → download → mesh. Every failure is a sentence, not a stack."""
    # At the moment of the call, not at import: an import must not mutate this
    # process's proxy settings, and a run that never generates pays no attention.
    _bypass_proxy_for_service()
    client = _client()
    submit_action, query_action = TIERS[_tier()]
    job = _call(client, submit_action, _submit_payload(prompt, model=_model(), image_url=image_url))
    job_id = str(job.get("JobId") or job.get("JobID") or "")
    if not job_id:
        raise RuntimeError(f"hunyuan3d submit returned no JobId: {json.dumps(job)[:400]}")

    deadline = time.monotonic() + POLL_TIMEOUT_S
    while time.monotonic() < deadline:
        result = _call(client, query_action, {"JobId": job_id})
        status = str(result.get("Status") or result.get("StatusString") or "").upper()
        if status in ("DONE", "SUCCESS", "SUCCEEDED", "COMPLETED"):
            url = _first_file_url(result)
            if not url:
                raise RuntimeError(
                    f"hunyuan3d job {job_id} finished but returned no file url: "
                    f"{json.dumps(result)[:600]}"
                )
            return _load_mesh(url)
        if status in ("FAIL", "FAILED", "ERROR"):
            raise RuntimeError(
                f"hunyuan3d job {job_id} failed: "
                f"{result.get('ErrorCode') or ''} {result.get('ErrorMessage') or ''}".strip()
            )
        time.sleep(POLL_INTERVAL_S)
    raise RuntimeError(
        f"hunyuan3d job {job_id} did not finish within {POLL_TIMEOUT_S}s; "
        "raise POLL_TIMEOUT_S for the pro version, or check the console for the job"
    )


def _call(client, action: str, payload: dict) -> dict:
    """One signed JSON call, with the SDK's typed request/response kept at the edge.

    SDK failures are re-raised as one sentence. The first failure this adapter ever
    hit was a reset connection caused by a local proxy — reported by the SDK as
    `ClientNetworkError`, which reads like a credential problem and wastes an hour.
    """
    import importlib  # noqa: PLC0415

    module = importlib.import_module("tencentcloud.ai3d.v20250513.models")
    request = getattr(module, f"{action}Request")()
    request.from_json_string(json.dumps(payload))
    try:
        response = getattr(client, action)(request)
    except Exception as error:  # noqa: BLE001 - the sentence is the product here
        code = str(getattr(error, "code", "") or type(error).__name__)
        message = str(getattr(error, "message", "") or error)
        hint = ""
        if "Network" in code or "Reset" in message or "10054" in message:
            hint = (
                " 网络层失败：这台机器设了 HTTPS_PROXY，腾讯云域名必须绕过它。"
                "适配器已对腾讯云域名自动绕过；若仍失败，检查代理/防火墙，"
                "或把 *.tencentcloudapi.com,*.myqcloud.com 加进 NO_PROXY。"
            )
        raise RuntimeError(f"hunyuan3d {action} 调用失败 [{code}]: {message}.{hint}") from error
    return json.loads(response.to_json_string())


def _first_file_url(payload: dict) -> str:
    """The first URL in the response that looks like a 3D file.

    The schema has moved between versions (`ResultFile3Ds`, `FileUrl`, `Url`), so
    this scans instead of guessing a fixed path: a rename upstream must not look
    like "the model produced nothing".
    """
    suffixes = (".glb", ".gltf", ".obj", ".stl", ".fbx", ".usdz")
    stack = [payload]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            for key, value in node.items():
                if isinstance(value, str) and value.startswith(("http://", "https://")):
                    if key.lower().endswith(("url", "uri")) or value.lower().split("?")[0].endswith(suffixes):
                        return value
                elif isinstance(value, (dict, list)):
                    stack.append(value)
        elif isinstance(node, list):
            stack.extend(node)
    return ""


def _load_mesh(url: str) -> trimesh.Trimesh:
    """Download and read whatever 3D format came back, then close it."""
    with urllib.request.urlopen(url, timeout=120) as response:
        data = response.read()
    with tempfile.NamedTemporaryFile(suffix=_suffix_of(url), delete=False) as handle:
        handle.write(data)
        path = Path(handle.name)
    try:
        loaded = trimesh.load(path, force="mesh")
    finally:
        path.unlink(missing_ok=True)
    if isinstance(loaded, trimesh.Scene):
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    mesh = loaded
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    if not mesh.is_watertight:
        trimesh.repair.fill_holes(mesh)
        trimesh.repair.fix_normals(mesh)
    return mesh


def _suffix_of(url: str) -> str:
    """The file extension the download URL implies, so the loader picks a parser."""
    clean = url.split("?")[0].lower()
    for suffix in (".glb", ".gltf", ".obj", ".stl", ".fbx", ".usdz"):
        if clean.endswith(suffix):
            return suffix
    return ".glb"


def _close_by_voxels(mesh: trimesh.Trimesh, *, samples: int = 120) -> trimesh.Trimesh:
    """Voxel-remesh an open shell into a watertight solid.

    Generative meshes come back as open shells (measured on the first real job:
    50 043 faces, 72 disconnected components, volume 0.045 of the bounding box). A
    verifier cannot measure a shell — no closed interior means no interference
    volume and no penetration — so the adapter has to close it before the harness
    sees it. Voxelise → fill → marching cubes is the cheapest operation that always
    answers with a manifold, at the cost of losing sharp detail below one voxel.

    `samples` is per *longest* side: 120 keeps a part of 0.35 units at ~0.003 units
    per voxel, which measured out to a 27 k-face insert — detailed enough for the
    geometry checks and cheap enough to rebuild every round.
    """
    extent = float(max(mesh.extents))
    if extent <= 0.0:
        raise ValueError("generated mesh has zero extent and cannot be closed")
    pitch = extent / float(samples)
    grid = mesh.voxelized(pitch=pitch).fill()
    closed = grid.marching_cubes
    if closed is None or len(closed.faces) == 0:
        raise ValueError("voxel remesh produced no surface; the model output is unusable")
    closed.apply_scale(pitch)
    closed.merge_vertices()
    trimesh.repair.fix_normals(closed)

    # A voxel fill of a surface with small holes produces a solid; of a shell missing
    # whole caps it produces a thin skin instead, which *looks* watertight and would
    # sail through every verifier with a 1 mm3 "part". Measured: the real Hunyuan
    # output closed to 27% of its convex hull (a tapered pin, plausible), while a
    # cylinder with two faces torn out closed to 7% (a skin). The ratio is the only
    # cheap way to tell those apart.
    hull_volume = float(mesh.convex_hull.volume)
    ratio = float(closed.volume) / hull_volume if hull_volume > 0.0 else 0.0
    if ratio < 0.25:
        raise ValueError(
            "generated mesh cannot be closed into a solid: voxel closing produced a "
            f"shell ({ratio:.0%} of its convex hull). The model output is missing too "
            "much surface; try a plainer prompt, or repair it with an external "
            "remesher before handing it to the harness."
        )
    return closed


def _align_long_axis_to_z(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Rotate the part so its longest direction is +Z, cross-section in XY.

    Generative models do not respect a requested orientation: the first real job
    came back lying along Y, so "scale the cross-section to the peg diameter" would
    have scaled the *length* and produced a 2.8 mm pin. Principal-axis alignment is
    the cheapest way to make the part's own frame agree with the insert's.
    """
    vertices = mesh.vertices - mesh.centroid
    if len(vertices) < 4:
        return mesh.copy()
    values, vectors = np.linalg.eigh(np.cov(vertices.T))
    order = np.argsort(values)
    z_axis = vectors[:, order[-1]]
    x_axis = vectors[:, order[-2]]
    y_axis = np.cross(z_axis, x_axis)
    # Rows, not columns: a point is re-expressed in the new basis as `R @ v`, so the
    # long-axis vector must be the *third row* to land on +Z. Written as columns it
    # maps the long axis onto a radial direction instead (measured: a 40 mm pin
    # lying along Y came back 40 mm along X).
    rotation = np.vstack([x_axis, y_axis, z_axis])
    if np.linalg.det(rotation) < 0.0:
        rotation[1] *= -1.0
    transform = np.block([[rotation, np.zeros((3, 1))], [np.zeros((1, 3)), 1.0]])
    aligned = mesh.copy()
    aligned.apply_transform(transform)
    return aligned


def _fit_to_interface(
    generated: trimesh.Trimesh, *, reference: trimesh.Trimesh, spec: DesignSpec
) -> trimesh.Trimesh:
    """Close, orient, scale to the designed diameter, seat, and bound to the envelope.

    Four steps, and each one exists because the first real generation failed without it:

    1. **close** — a generative mesh is an open shell; nothing downstream can measure it;
    2. **orient** — the part comes back lying along whatever axis its own frame used
       (measured: along Y), so scaling before alignment distorts it;
    3. **scale uniformly** so the *smaller* cross-section extent equals `d_peg_mm` —
       the smaller one is the shank, the larger one is whatever detail the model added,
       and scaling the larger would make the shank undersized (a loose fit) while
       scaling the length would make it a 2.8 mm pin;
    4. **bound to the designed insert envelope** — a generated pin measured 33 mm
       long against a 14 mm design; cutting it to the envelope keeps the interface
       exact and stops a decorative tail from becoming an interference the policy
       would then try to "repair".
    """
    mesh = generated.copy()
    if not mesh.is_watertight:
        mesh = _close_by_voxels(mesh)
    mesh = _align_long_axis_to_z(mesh)
    mesh.apply_translation(-mesh.bounds.mean(axis=0))

    params = spec.interface.validated_params()
    target_d = float(getattr(params, "d_peg_mm", 0.0) or 0.0)
    if target_d <= 0.0:
        target_d = float(min(reference.extents[0], reference.extents[1]))
    shank = float(min(mesh.extents[0], mesh.extents[1]))
    if shank > 1e-9:
        mesh.apply_scale(target_d / shank)

    target = reference.bounds
    current = mesh.bounds
    mesh.apply_translation(
        [
            (target[0][0] + target[1][0]) / 2 - (current[0][0] + current[1][0]) / 2,
            (target[0][1] + target[1][1]) / 2 - (current[0][1] + current[1][1]) / 2,
            target[0][2] - current[0][2],
        ]
    )

    oversized = any(
        mesh.extents[i] > reference.extents[i] + 1e-6 for i in range(3)
    )
    if oversized:
        envelope = trimesh.creation.box(extents=reference.extents)
        envelope.apply_translation(reference.bounds.mean(axis=0))
        try:
            cut = trimesh.boolean.intersection([mesh, envelope], engine="manifold")
        except Exception:  # noqa: BLE001 - a boolean failure falls back to the uncut shape
            cut = None
        if cut is not None and cut.is_watertight and float(cut.volume) > 0.0:
            mesh = cut
    return mesh


# --------------------------------------------------------------------------- #
# --check: is the environment ready? (spends nothing)
# --------------------------------------------------------------------------- #
def _check() -> int:
    problems: list[str] = []
    for name in (SECRET_ID_ENV, SECRET_KEY_ENV):
        if not os.environ.get(name):
            problems.append(
                f"{name} is not set in this process (pah loads .env at start; a bare "
                "script must export it)"
            )
    try:
        import tencentcloud.ai3d.v20250513  # noqa: F401,PLC0415

        sdk = "tencentcloud-sdk-python present"
    except ImportError:
        sdk = "MISSING: pip install tencentcloud-sdk-python"
        problems.append(sdk)
    region = os.environ.get(REGION_ENV) or DEFAULT_REGION
    try:
        tier = _tier()
        submit_action, query_action = TIERS[tier]
    except RuntimeError as error:
        problems.append(str(error))
        tier, submit_action, query_action = "?", "?", "?"
    model = _model() or "<service default>"
    print(f"endpoint   ai3d.tencentcloudapi.com  version={API_VERSION}  region={region}")
    print(f"sdk        {sdk}")
    print(f"tier       {tier}   (HUNYUAN3D_TIER=standard|rapid|pro)")
    print(f"model      {model}   (HUNYUAN3D_MODEL, e.g. HY-3D-2.5 / HY-3D-3.1)")
    print(f"actions    {submit_action} -> {query_action}")
    for problem in problems:
        print(f"FAIL       {problem}")
    print(
        f"\n{len(problems)} problem(s). --check never calls the service (no cost) "
        "and never reads .env."
    )
    return 0 if not problems else 1


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="verify SDK + keys, no network call")
    parser.add_argument("--prompt", default="a straight cylindrical pin with a chamfered tip",
                        help="plain generation prompt (costs one job)")
    parser.add_argument("--out", default="runs/hunyuan-smoke", help="where to write the mesh")
    args = parser.parse_args(argv[1:])
    if args.check:
        return _check()
    if args.prompt:
        # Explicit generation: this is the only path that spends money, and it is
        # never on the default route.
        mesh = _generate(args.prompt)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        path = out / "generated_insert.stl"
        mesh.export(path)
        print(f"wrote {path}  watertight={mesh.is_watertight} "
              f"volume={mesh.volume:.1f} faces={len(mesh.faces)}")
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

