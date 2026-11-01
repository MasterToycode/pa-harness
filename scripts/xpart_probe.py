"""One real 组件生成 job, to settle what the output actually is before wiring it in.

Why this script exists
----------------------
The harness needs to know one thing the API docs do not say: does a decomposed object
come back as **one multi-object file**, as **N separate files**, or as **one merged
mesh**? Everything about how `Decomposer` consumes the result depends on the answer,
and the docs only say "generated file URLs".

This script therefore **spends credits exactly once**: upload -> submit -> poll ->
download -> inspect, with the JobId written to disk the moment submit returns, so a
crash or a timeout resumes by querying that job instead of paying again. Re-running
after a successful submit reuses the recorded JobId unless `--force-submit` is given.

Artifacts all land in `--out` (default `D:\\Temp\\pah-xpart-probe\\`):

    attempt.json   written *before* submit, so an interrupted call is visible
    state.json     JobId + timings; what a resume reads
    submit.json    raw SubmitHunyuan3DPartJob response
    query.json     raw final QueryHunyuan3DPartJob response (signed URLs, 24h)
    upload.json    bucket + URL prefix (signature stripped)
    part_*.{fbx,obj,...}, part_segmentation_info.json
    summary.json   everything below, in one file, plus the mesh inspection

Usage:

    python scripts/xpart_probe.py D:\\Temp\\pah-fbx\\000.fbx
    python scripts/xpart_probe.py --job-id 1339...        # query an existing job, no cost
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from pah.app.env import load_env_file  # noqa: E402
from pah.tools.tencent_ai3d import (  # noqa: E402
    SECRET_ID_ENV,
    SECRET_KEY_ENV,
    download_results,
    submit_part_job,
    upload_to_cos,
    wait_part_job,
)

DEFAULT_INPUT = Path("D:/Temp/pah-fbx/000.fbx")
DEFAULT_OUT = Path("D:/Temp/pah-xpart-probe")

#: The one place the credentials are looked for, matching what `pah` reads.
ENV_FILE = REPO / ".env"


def _json_write(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", "utf-8")
    return path


def _url_prefix(url: str) -> str:
    """Scheme + host + path. Kept so a result can be described without leaking a
    signature into a report or a log."""
    return url.split("?", 1)[0]


def _verify_url(url: str) -> dict:
    """Fetch the first kilobyte of the signed URL before submitting a paid job.

    The service reads that URL from the public internet; if it is not reachable, the
    job fails after being submitted. Reading it locally is the cheapest way to catch a
    private bucket, a wrong region, or a signature that cannot be validated.
    """
    import urllib.request  # noqa: PLC0415

    request = urllib.request.Request(url, headers={"Range": "bytes=0-1023"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return {"ok": True, "status": int(response.status), "bytes_read": len(response.read())}
    except Exception as error:  # noqa: BLE001
        return {"ok": False, "error": f"{type(error).__name__}: {error}"}


def _log(message: str) -> None:
    print(message, flush=True)


# --------------------------------------------------------------------------- #
# inspection: the actual question this probe answers
# --------------------------------------------------------------------------- #
def _mesh_facts(mesh) -> dict:
    return {
        "faces": int(len(mesh.faces)),
        "vertices": int(len(mesh.vertices)),
        "watertight": bool(mesh.is_watertight),
        "extents": [round(float(value), 4) for value in mesh.extents],
        # World-space bbox: the parts only assemble if they share one frame, and the
        # only honest evidence for that is where each part actually sits.
        "bounds": [
            [round(float(value), 4) for value in mesh.bounds[0]],
            [round(float(value), 4) for value in mesh.bounds[1]],
        ],
        "volume": round(float(abs(mesh.volume)), 4) if mesh.is_watertight else None,
    }


def _inspect_with_trimesh(path: Path) -> dict:
    """Scene vs Trimesh, per-object facts, and the merged view the harness would see."""
    import trimesh  # noqa: PLC0415 - imported here so the probe reports a missing dep

    facts: dict = {"path": str(path), "bytes": path.stat().st_size}
    try:
        loaded = trimesh.load(path, process=False)
    except Exception as error:  # noqa: BLE001 - the probes's product is a diagnosis
        return {**facts, "readable": False, "error": f"{type(error).__name__}: {error}"}

    facts["readable"] = True
    facts["trimesh_type"] = type(loaded).__name__
    if isinstance(loaded, trimesh.Scene):
        geometry = dict(loaded.geometry)
        facts["geometry_count"] = len(geometry)
        facts["object_names"] = list(geometry)
        facts["objects"] = {
            name: _mesh_facts(mesh) for name, mesh in geometry.items()
        }
        facts["scene_watertight_all"] = all(mesh.is_watertight for mesh in geometry.values())
    else:
        facts["geometry_count"] = 1
        facts["object_names"] = [Path(path).stem]
        facts["objects"] = {Path(path).stem: _mesh_facts(loaded)}

    try:
        merged = trimesh.load(path, force="mesh")
        facts["merged"] = _mesh_facts(merged)
    except Exception as error:  # noqa: BLE001
        facts["merged"] = {"error": f"{type(error).__name__}: {error}"}
    return facts


def _inspect_fbx_with_blender(path: Path, work_dir: Path) -> dict:
    """trimesh cannot read FBX; Blender can, and the harness already depends on it."""
    from pah.tools import fbx  # noqa: PLC0415

    facts: dict = {"path": str(path), "bytes": path.stat().st_size, "trimesh_type": "FBX"}
    try:
        parts = fbx.from_fbx(path, work_dir)
    except Exception as error:  # noqa: BLE001
        return {**facts, "readable": False, "error": f"{type(error).__name__}: {error}"}
    facts["readable"] = True
    facts["geometry_count"] = len(parts)
    facts["object_names"] = list(parts)
    facts["objects"] = {
        name: _inspect_with_trimesh(Path(stl)) for name, stl in parts.items()
    }
    return facts


def inspect_artifact(path: Path, work_dir: Path) -> dict:
    """One downloaded artifact -> facts. ZIPs are unpacked first: a part bundle is a
    plausible answer and would otherwise be reported as "unreadable"."""
    suffix = path.suffix.lower()
    if suffix == ".zip":
        target = work_dir / f"{path.stem}_unzipped"
        with zipfile.ZipFile(path) as archive:
            archive.extractall(target)
        inner = [
            inspect_artifact(item, work_dir)
            for item in sorted(target.rglob("*"))
            if item.is_file() and item.suffix.lower() in (".fbx", ".obj", ".glb", ".stl", ".ply")
        ]
        return {"path": str(path), "bytes": path.stat().st_size, "zip": True, "contents": inner}
    if suffix == ".fbx":
        return _inspect_fbx_with_blender(path, work_dir)
    if suffix in (".obj", ".glb", ".gltf", ".stl", ".ply", ".off"):
        return _inspect_with_trimesh(path)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "readable": False,
        "error": f"no mesh inspector for {suffix or '<none>'}",
    }


def _segmentation_summary(path: Path) -> dict:
    """What the segmentation payload says, decoded rather than assumed.

    The field is documented as "分割数据" with no schema, and the first real job came
    back as a binary `LSEG` blob — which a `json.load` reports as a UTF-8 error that
    reads like corruption. `parse_segmentation` is the only thing that knows better.
    """
    from pah.tools.tencent_ai3d import Ai3dError, parse_segmentation  # noqa: PLC0415

    summary: dict = {"path": str(path), "bytes": path.stat().st_size}
    try:
        decoded = parse_segmentation(path.read_bytes())
    except (Ai3dError, UnicodeDecodeError) as error:
        return {**summary, "error": f"{type(error).__name__}: {error}"}
    summary["format"] = decoded["format"]
    if decoded["format"] == "LSEG":
        labels = decoded["labels"]
        summary.update(
            {
                "version": decoded["version"],
                "count": decoded["count"],
                "label_values": sorted(int(key) for key in decoded["histogram"]),
                "faces_per_label": decoded["histogram"],
                "head_bytes": path.read_bytes()[:16].hex(),
            }
        )
        # The label count is what a Decomposer would index, so say whether it can be
        # joined straight to the submitted mesh rather than leaving that to a guess.
        summary["labels_look_one_per_input_face"] = len(labels) == decoded["count"]
    else:
        summary["data_head"] = json.dumps(decoded["data"], ensure_ascii=False)[:600]
    return summary


def _frame_report(inspection: list[dict]) -> dict:
    """Do the parts assemble as-is, or is each one re-centred at the origin?

    Answered from world-space bboxes, not from faith: if every part is centred on the
    same point, the service normalised each part and a registration step is required
    before any assembly check can mean anything.
    """
    import numpy as np  # noqa: PLC0415

    bounds = [
        item["merged"]["bounds"]
        for item in inspection
        if isinstance(item, dict) and isinstance(item.get("merged"), dict) and item["merged"].get("bounds")
    ]
    report: dict = {"parts_with_bounds": len(bounds)}
    if not bounds:
        report["shared_frame"] = None
        return report
    lo = np.min([box[0] for box in bounds], axis=0)
    hi = np.max([box[1] for box in bounds], axis=0)
    centers = np.array([(np.array(box[0]) + np.array(box[1])) / 2.0 for box in bounds])
    spread = float(np.max(np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)))
    union = hi - lo
    report.update(
        {
            "union_min": [round(float(value), 4) for value in lo],
            "union_max": [round(float(value), 4) for value in hi],
            "union_extents": [round(float(value), 4) for value in union],
            "center_spread": round(spread, 4),
            # 5% of the longest side: tighter than that and "shared frame" and
            # "every part re-centred" are indistinguishable in the data.
            "shared_frame": bool(spread > 0.05 * float(np.max(union))),
        }
    )
    return report


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #
def _headline(summary: dict) -> str:
    """The five lines the harness needs, before the 20 kB of raw JSON.

    Everything else in `summary.json` is evidence; this is the answer to "what shape is
    the output?" — the question the whole probe exists to settle.
    """
    inspection = summary["inspection"]
    kinds = sorted({str(item.get("trimesh_type")) for item in inspection})
    geometries = [item.get("geometry_count") for item in inspection if item.get("readable")]
    names = [name for item in inspection for name in item.get("object_names") or []]
    solids = [
        bool(facts.get("watertight"))
        for item in inspection
        for facts in (item.get("objects") or {}).values()
        if isinstance(facts, dict)
    ]
    return "\n".join(
        [
            f"shape     = {summary['result_file_count']} files, types="
            f"{sorted({str(item['type']) for item in inspection})}",
            f"per file  = trimesh {kinds}, geometry_count={geometries}",
            f"objects   = {names}",
            f"watertight= {sum(solids)}/{len(solids)} meshes as loaded",
            f"frame     = {json.dumps(summary['frame'], ensure_ascii=False)}",
            f"segment   = {json.dumps({k: v for k, v in (summary['segmentation'] or {}).items() if k != 'faces_per_label'}, ensure_ascii=False)}",
        ]
    )


def _cost_line(staged: bool, post_process: bool) -> str:
    return (
        "cost      = 1 次组件生成任务（后付费积分制）"
        f" + staged={'开' if staged else '关'} + post_process={'开(+20 积分)' if post_process else '关'}"
        " → 本次只提交 1 次，脚本内部不重试提交"
    )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("fbx", nargs="?", default=str(DEFAULT_INPUT), help="input FBX")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="artifact directory")
    parser.add_argument("--timeout", type=float, default=900.0, help="poll budget (s)")
    parser.add_argument("--interval", type=float, default=10.0, help="poll interval (s)")
    parser.add_argument("--staged", action="store_true", help="enable staged generation")
    parser.add_argument("--post-process", action="store_true", help="enable post process (+20 credits)")
    parser.add_argument("--job-id", default="", help="query an already submitted job; no cost")
    parser.add_argument(
        "--force-submit",
        action="store_true",
        help="submit even if state.json already records a job (spends credits again)",
    )
    args = parser.parse_args(argv[1:])

    load_env_file(ENV_FILE)
    started = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    _log(f"input     {args.fbx}")
    _log(f"out       {out}")
    _log(
        f"env       {ENV_FILE}  keys: "
        + ", ".join(
            f"{name}={'set' if os.environ.get(name) else 'MISSING'}"
            for name in (SECRET_ID_ENV, SECRET_KEY_ENV)
        )
    )
    _log(_cost_line(args.staged, args.post_process))

    state_path = out / "state.json"
    state: dict = json.loads(state_path.read_text("utf-8")) if state_path.is_file() else {}
    job_id = args.job_id or ("" if args.force_submit else state.get("job_id") or "")
    upload_info: dict = state.get("upload", {}) if job_id else {}
    submit_seconds: float | None = state.get("submit_seconds") if job_id else None

    if not job_id:
        source = Path(args.fbx)
        if not source.is_file():
            _log(f"STOP      输入文件不存在：{source}")
            return 2

        _json_write(out / "attempt.json", {"attempted_at": time.time(), "input": str(source)})
        _log("upload    -> COS ...")
        uploaded = time.time()
        url = upload_to_cos(source)
        upload_info = {
            "url_prefix": _url_prefix(url),
            "bucket": _url_prefix(url).split("//", 1)[-1].split(".", 1)[0],
            "seconds": round(time.time() - uploaded, 2),
        }
        _log(f"upload    ok  {time.time() - uploaded:.1f}s  {upload_info['bucket']}")

        reachable = _verify_url(url)
        upload_info["url_reachable"] = reachable
        _log(f"preflight signed URL  {json.dumps(reachable, ensure_ascii=False)}")
        _json_write(out / "upload.json", upload_info)
        if not reachable.get("ok"):
            _log("STOP      签名 URL 本地都读不到，提交只会白花钱；先修 COS/URL，不提交任务。")
            return 3

        _log("submit    -> SubmitHunyuan3DPartJob ...")
        submitted = time.time()
        job_id = submit_part_job(
            url, staged=args.staged, post_process=args.post_process
        )
        submit_seconds = round(time.time() - submitted, 2)
        # Written before anything else can fail: the JobId is the only thing a
        # resume must not lose, because re-submitting costs credits.
        _json_write(
            out / "state.json",
            {
                "job_id": job_id,
                "submitted_at": time.time(),
                "submit_seconds": submit_seconds,
                "input": str(source),
                "staged": args.staged,
                "post_process": args.post_process,
                "upload": upload_info,
            },
        )
        _json_write(
            out / "submit.json",
            {"JobId": job_id, "seconds": submit_seconds, "input_url_prefix": _url_prefix(url)},
        )
        _log(f"submit    ok  JobId={job_id}  {submit_seconds:.1f}s")
    else:
        _log(f"resume    JobId={job_id}（已在 {state_path.name} 里；不再提交，不花钱）")

    _log(f"query     polling every {args.interval:.0f}s, budget {args.timeout:.0f}s ...")
    polled = time.time()
    last = {"status": ""}

    def progress(response: dict) -> None:
        status = str(response.get("Status") or "").upper()
        if status != last["status"]:
            last["status"] = status
            _log(f"query     {time.time() - polled:6.1f}s  Status={status or '<empty>'}")

    final = wait_part_job(
        job_id, timeout_s=args.timeout, interval_s=args.interval, on_poll=progress
    )
    query_seconds = round(time.time() - polled, 2)
    _json_write(out / "query.json", final)

    entries = final.get("ResultFile3Ds") or []
    _log(
        f"query     done  Status={final.get('Status')}  "
        f"ResultFile3Ds={len(entries)}  PartSegmentationInfoUrl="
        f"{'yes' if final.get('PartSegmentationInfoUrl') else 'no'}  {query_seconds:.0f}s"
    )

    _log("download  -> all result files ...")
    downloaded = download_results(final, out)
    for item in downloaded["files"]:
        path = Path(item["path"])
        _log(f"download  {path.name:24s} {item['type'] or '?':8s} {path.stat().st_size / 1e6:8.2f} MB")
    if downloaded["segmentation"]:
        seg = Path(downloaded["segmentation"]["path"])
        _log(f"download  {seg.name:24s} {'JSON':8s} {seg.stat().st_size / 1e6:8.2f} MB")

    _log("inspect   -> trimesh / Blender ...")
    inspection = [
        inspect_artifact(Path(item["path"]), out / "inspect")
        for item in downloaded["files"]
    ]
    segmentation_summary = (
        _segmentation_summary(Path(downloaded["segmentation"]["path"]))
        if downloaded["segmentation"]
        else None
    )

    summary = {
        "input": str(args.fbx),
        "started_at": started,
        "finished_at": time.time(),
        "elapsed_s": round(time.time() - started, 1),
        "upload": upload_info,
        "job": {
            "job_id": job_id,
            "submit_seconds": submit_seconds,
            "query_seconds": query_seconds,
            "staged": state.get("staged", args.staged),
            "post_process": state.get("post_process", args.post_process),
        },
        "status": final.get("Status"),
        "error_code": final.get("ErrorCode"),
        "error_message": final.get("ErrorMessage"),
        "result_file_count": len(entries),
        "result_files": [
            {
                "type": item["type"],
                "url_prefix": _url_prefix(item["url"]),
                "path": item["path"],
                "bytes": Path(item["path"]).stat().st_size,
            }
            for item in downloaded["files"]
        ],
        "segmentation": segmentation_summary,
        "frame": _frame_report(inspection),
        "inspection": inspection,
    }
    summary_path = _json_write(out / "summary.json", summary)

    _log("")
    _log(_headline(summary))
    _log("")
    _log(json.dumps(summary, ensure_ascii=False, indent=1))
    _log("")
    _log("files")
    for path in sorted(out.iterdir()):
        if path.is_file():
            _log(f"  {path.name:32s} {path.stat().st_size / 1e6:8.3f} MB")
    _log(f"summary   {summary_path}")
    return 0 if str(final.get("Status") or "").upper() in ("DONE", "SUCCESS") else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
