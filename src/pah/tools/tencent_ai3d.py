"""Tencent Cloud 组件生成 (X-Part + P3-SAM) client: one whole object in, parts out.

Why this module exists
---------------------
`SubmitHunyuan3DPartJob` is the only cloud action that returns an object *already
split into parts* — X-Part's decomposition plus P3-SAM's part meshes — which is
exactly what the N-part harness needs and exactly what a text-to-3D generator
cannot give. The protocol is not an SDK one-liner: upload, submit, poll, download.

Three properties of that API drive everything below, and none are optional:

1. **The input is a URL, never bytes.** The part action has no Base64 field (the
   image and prompt actions do) and the only container it accepts is FBX. So this
   module owns a COS upload and a 24-hour pre-signed URL. The bucket stays private:
   a signed URL is all the service needs, and a public-read bucket would publish
   every model ever generated.
2. **A COS bucket name carries the account APPID, which is not the Uin.** STS
   reports a Uin; CAM's `GetUserAppId` reports the APPID that bucket names use.
   Getting this wrong is a `create_bucket` failure that reads like a permissions
   problem, so `PAH_COS_APPID` exists as the escape hatch.
3. **The domain must bypass the developer proxy.** This machine carries a global
   `HTTPS_PROXY` for foreign-network access; routed through it, connections to
   `*.tencentcloudapi.com` reset (WinError 10054) *before* a request is signed,
   which surfaces as a credential error and is not one. `_bypass_proxy_for_service`
   extends `NO_PROXY` for this process only, and covers the COS download host too.

Cost discipline
---------------
`submit_part_job` spends credits; `post_process=True` spends 20 more. Nothing here
retries a submit — a network failure after `Submit` may already have created a job,
so the caller keeps the JobId and queries it. Result URLs expire in 24 hours, so
`download_results` belongs in the same run as the submit that produced them.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import struct
import time
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Any, Callable

__all__ = [
    "API_VERSION",
    "BUCKET_PREFIX",
    "ENDPOINT",
    "Ai3dError",
    "CosError",
    "PartJobFailed",
    "PartJobTimeout",
    "download_results",
    "parse_segmentation",
    "query_part_job",
    "submit_part_job",
    "upload_to_cos",
    "wait_part_job",
]

#: The service version the request/response shapes below were verified against.
API_VERSION = "2025-05-13"
ENDPOINT = "ai3d.tencentcloudapi.com"

SECRET_ID_ENV = "TENCENTCLOUD_SECRET_ID"
SECRET_KEY_ENV = "TENCENTCLOUD_SECRET_KEY"
REGION_ENV = "TENCENTCLOUD_REGION"
#: Overrides the APPID lookup when CAM is not readable by this key.
APPID_ENV = "PAH_COS_APPID"
DEFAULT_REGION = "ap-guangzhou"

#: Bucket name without the APPID suffix, which COS appends as `-<appid>`.
BUCKET_PREFIX = "pah-parts-"

#: The only model version the part action currently documents.
MODEL = "1.5"

#: Statuses that mean "keep polling". Anything else is terminal.
_PENDING = {"WAIT", "RUN", "RUNNING", "PENDING", "INIT", ""}
_FAILED = {"FAIL", "FAILED", "ERROR"}
_DONE = {"DONE", "SUCCESS", "SUCCEEDED", "COMPLETED"}

#: Extensions worth preserving when a result is named after its URL.
_MODEL_SUFFIXES = {".fbx", ".glb", ".gltf", ".obj", ".stl", ".ply", ".usdz", ".zip"}


class Ai3dError(RuntimeError):
    """Any service-level refusal, already reduced to one actionable sentence."""


class PartJobFailed(Ai3dError):
    """The job reached FAIL. Keeps the code and message separate so a caller can
    branch on `ErrorCode` (a parameter error is worth fixing) without re-parsing text."""

    def __init__(self, job_id: str, code: str, message: str) -> None:
        self.job_id = job_id
        self.code = code
        self.message = message
        super().__init__(f"组件生成任务 {job_id} 失败 [{code}]: {message}")


class PartJobTimeout(Ai3dError):
    """The job is still running. The JobId is the only thing that must not be lost."""


class CosError(Ai3dError):
    """Upload-side failure (SDK, credentials, bucket, object)."""


# --------------------------------------------------------------------------- #
# the proxy trap
# --------------------------------------------------------------------------- #
def _bypass_proxy_for_service() -> None:
    """Keep Tencent Cloud off this machine's foreign-network proxy.

    `NO_PROXY` is only *extended*: an explicit proxy entry for these hosts still
    wins, because this function never removes anything. Applied at import time
    because the first request is the one that has to work — a caller cannot be
    expected to remember the incantation before every probe.
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


_bypass_proxy_for_service()


# --------------------------------------------------------------------------- #
# credentials, clients
# --------------------------------------------------------------------------- #
def _credentials() -> tuple[str, str, str]:
    """SecretId, SecretKey, region. Error text names variables, never values."""
    secret_id = os.environ.get(SECRET_ID_ENV, "").strip()
    secret_key = os.environ.get(SECRET_KEY_ENV, "").strip()
    region = os.environ.get(REGION_ENV, "").strip() or DEFAULT_REGION
    if not secret_id or not secret_key:
        raise Ai3dError(
            f"组件生成需要 ${SECRET_ID_ENV} 和 ${SECRET_KEY_ENV}。把它们写进仓库根目录的 "
            ".env（pah 启动时会读，脚本里要自己 load_env_file()），"
            "或先在 https://console.cloud.tencent.com/cam/capi 新建一对密钥。"
        )
    return secret_id, secret_key, region


def _ai3d_client(client: Any = None) -> Any:
    """The SDK client, or the one injected by a test. Imports stay here so importing
    this module never costs the `tencentcloud` import."""
    if client is not None:
        return client
    secret_id, secret_key, region = _credentials()
    try:
        from tencentcloud.ai3d.v20250513 import ai3d_client
        from tencentcloud.common import credential
        from tencentcloud.common.profile.client_profile import ClientProfile
        from tencentcloud.common.profile.http_profile import HttpProfile
    except ImportError as error:  # the package is optional until this module is used
        raise Ai3dError(
            "缺少腾讯云 SDK：请先 `pip install tencentcloud-sdk-python`。"
            f"（{type(error).__name__}: {error}）"
        ) from error
    http = HttpProfile()
    http.endpoint = ENDPOINT
    profile = ClientProfile()
    profile.httpProfile = http
    return ai3d_client.Ai3dClient(
        credential.Credential(secret_id, secret_key), region, profile
    )


def _call(client: Any, action: str, payload: dict) -> dict:
    """One signed call, with the SDK's typed request/response kept at the edge.

    A raw SDK exception is re-raised as one sentence: the first failure this
    integration hit was `ClientNetworkError` from the local proxy, which reads like
    a bad key and sends people to rotate credentials that are fine.
    """
    module = importlib.import_module(f"tencentcloud.ai3d.v{API_VERSION.replace('-', '')}.models")
    request = getattr(module, f"{action}Request")()
    request.from_json_string(json.dumps(payload))
    try:
        response = getattr(client, action)(request)
    except Exception as error:  # noqa: BLE001 - one sentence is the product here
        code = str(getattr(error, "code", "") or type(error).__name__)
        message = str(getattr(error, "message", "") or error)
        hint = ""
        if "Network" in code or "Reset" in message or "10054" in message:
            hint = (
                " 网络层失败：这台机器的 HTTPS_PROXY 会重置腾讯云连接；"
                "本模块已在导入时把 *.tencentcloudapi.com / *.myqcloud.com 加进 NO_PROXY，"
                "若仍失败请检查代理或防火墙。"
            )
        raise Ai3dError(f"{action} 调用失败 [{code}]: {message}.{hint}") from error
    return json.loads(response.to_json_string())


# --------------------------------------------------------------------------- #
# submit / query / wait
# --------------------------------------------------------------------------- #
def _submit_payload(
    url: str, *, staged: bool, post_process: bool, model: str = MODEL
) -> dict:
    """The submit body, pure so a test can assert what would be sent.

    `File.Type` is mandatory and only `FBX` is accepted; the URL has to be a URL
    (there is no byte field), which is why `upload_to_cos` exists.
    """
    if not str(url).startswith(("http://", "https://")):
        raise ValueError(
            "组件生成的 File 只接受 URL（没有 Base64 通道），先 upload_to_cos() 再提交；"
            f"收到的是 {url[:80]!r}"
        )
    return {
        "File": {"Url": url, "Type": "FBX"},
        "Model": model,
        "EnableStagedGeneration": bool(staged),
        "EnablePostProcess": bool(post_process),
    }


def submit_part_job(
    url: str, *, staged: bool = False, post_process: bool = False, client: Any = None
) -> str:
    """Submit one part-generation job and return its JobId. **This spends credits.**

    `staged=True` splits the job in two (segmentation first, editable, then part
    meshes); `post_process=True` adds 20 credits and returns a single merged model
    link instead of the part files — which is the opposite of what the harness wants,
    so it defaults off.
    """
    client = _ai3d_client(client)
    payload = _submit_payload(url, staged=staged, post_process=post_process)
    response = _call(client, "SubmitHunyuan3DPartJob", payload)
    job_id = str(response.get("JobId") or "").strip()
    if not job_id:
        raise Ai3dError(
            "SubmitHunyuan3DPartJob 没有返回 JobId，无法查询任务："
            f"{json.dumps(response, ensure_ascii=False)[:400]}"
        )
    return job_id


def query_part_job(job_id: str, *, client: Any = None, **_: Any) -> dict:
    """Query one job. Returns the response dict verbatim: `Status`, `ErrorCode`,
    `ErrorMessage`, `ResultFile3Ds` and `PartSegmentationInfoUrl`."""
    client = _ai3d_client(client)
    if not str(job_id).strip():
        raise ValueError("query_part_job 需要一个 JobId")
    return _call(client, "QueryHunyuan3DPartJob", {"JobId": job_id})


def _status_of(result: dict) -> str:
    return str(result.get("Status") or result.get("StatusString") or "").strip().upper()


def wait_part_job(
    job_id: str,
    *,
    timeout_s: float = 900.0,
    interval_s: float = 10.0,
    client: Any = None,
    on_poll: Callable[[dict], None] | None = None,
) -> dict:
    """Poll until DONE (returns the final response) or FAIL (raises `PartJobFailed`).

    Polling is cheap and never resubmits; a timeout raises `PartJobTimeout` carrying
    the JobId, because re-running a query is free while re-running a submit is not.
    `on_poll` sees every response, so a several-minute job can print progress instead
    of looking hung.
    """
    client = _ai3d_client(client)
    deadline = time.monotonic() + float(timeout_s)
    result: dict = {}
    while True:
        result = query_part_job(job_id, client=client)
        if on_poll is not None:
            on_poll(result)
        status = _status_of(result)
        if status in _DONE:
            return result
        if status in _FAILED:
            raise PartJobFailed(
                job_id,
                str(result.get("ErrorCode") or "").strip(),
                str(result.get("ErrorMessage") or "").strip(),
            )
        if status not in _PENDING:
            raise Ai3dError(
                f"任务 {job_id} 返回了未知状态 {status!r}：" f"{json.dumps(result, ensure_ascii=False)[:400]}"
            )
        if time.monotonic() >= deadline:
            raise PartJobTimeout(
                f"任务 {job_id} 在 {timeout_s:.0f}s 内没有结束（最后状态 {status or 'UNKNOWN'}）。"
                "不要重新提交：用同一个 JobId 再查一次。"
            )
        time.sleep(float(interval_s))


# --------------------------------------------------------------------------- #
# download
# --------------------------------------------------------------------------- #
def _http_get(url: str) -> bytes:
    """Download one result file. 24h-signed COS URLs, so no credentials needed."""
    with urllib.request.urlopen(url, timeout=180) as response:
        return response.read()


def _suffix_for(entry: dict, url: str) -> str:
    """The extension for a downloaded file: the URL's own, else the declared `Type`.

    COS URLs are signed, so the path is still the honest place to look; `Type` is
    the fallback because the service declares the format there and a proxy or a CDN
    can rewrite the path.
    """
    path_suffix = PurePosixPath(url.split("?")[0]).suffix.lower()
    if path_suffix in _MODEL_SUFFIXES:
        return path_suffix
    declared = str(entry.get("Type") or "").strip().lower()
    if declared and len(declared) <= 5 and declared.isalnum():
        return f".{declared}"
    return ".bin"


def download_results(
    result: dict, out_dir: str | Path, *, fetch: Callable[[str], bytes] | None = None
) -> dict:
    """Download every `ResultFile3Ds` entry plus the segmentation blob.

    Files are named `part_00.<ext>`, `part_01.<ext>` in service order rather than by
    URL basename: the basenames are opaque COS keys, and part order is the only
    ordering the service promises. Returns `{"dir", "files": [{type,url,path}],
    "segmentation": {url,path} | None}` — with URLs kept, because a result whose
    download looked short can be re-fetched without paying for another job.

    The segmentation payload is **not JSON** (measured: a `LSEG` binary header plus
    one label byte per face), so it is written as `.bin` and left to
    `parse_segmentation` rather than being named after a format it does not have.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    getter = fetch if fetch is not None else _http_get

    files: list[dict] = []
    for index, entry in enumerate(result.get("ResultFile3Ds") or []):
        url = str(entry.get("Url") or "").strip()
        if not url:
            raise Ai3dError(
                f"第 {index} 个 ResultFile3D 没有 Url：{json.dumps(entry, ensure_ascii=False)[:300]}"
            )
        path = out / f"part_{index:02d}{_suffix_for(entry, url)}"
        path.write_bytes(getter(url))
        files.append({"type": entry.get("Type"), "url": url, "path": str(path)})

    segmentation = None
    seg_url = str(result.get("PartSegmentationInfoUrl") or "").strip()
    if seg_url:
        seg_path = out / "part_segmentation_info.bin"
        seg_path.write_bytes(getter(seg_url))
        segmentation = {"url": seg_url, "path": str(seg_path)}

    return {"dir": str(out), "files": files, "segmentation": segmentation}


def parse_segmentation(payload: bytes) -> dict:
    """Decode `PartSegmentationInfoUrl`: per-face part labels, in two known shapes.

    Measured against one real job: the download is **not JSON**. It is

        b"LSEG" | uint32-LE version (1) | uint32-LE count | count * uint8 label

    with `label` being the part number — the same number the result GLBs carry as
    their inner object name (`part_0` ... `part_18`). One byte per face is why the
    payload is a few hundred kB and why a JSON parser reports "invalid UTF-8".

    A JSON body is returned as-is under `"format": "json"`: the field is documented
    as "分割数据" without a schema, so a version that starts sending JSON must not be
    reported as corruption.
    """
    data = bytes(payload)
    if data[:1] in (b"{", b"["):
        return {"format": "json", "data": json.loads(data.decode("utf-8"))}
    if len(data) < 12 or data[:4] != b"LSEG":
        raise Ai3dError(
            "无法识别 PartSegmentationInfo：既不是 LSEG 二进制（magic 'LSEG'），也不是 JSON。"
            f"前 16 字节：{data[:16].hex()}"
        )
    version, count = struct.unpack("<II", data[4:12])
    labels = data[12 : 12 + count]
    if len(labels) != count:
        raise Ai3dError(
            f"LSEG 头部声明 {count} 个标签，实际只有 {len(labels)} 字节（文件被截断？）"
        )
    histogram: dict[str, int] = {}
    for value in labels:
        histogram[str(value)] = histogram.get(str(value), 0) + 1
    return {
        "format": "LSEG",
        "version": int(version),
        "count": int(count),
        "labels": labels,
        "histogram": histogram,
    }


# --------------------------------------------------------------------------- #
# COS upload (the only way to give the service a file)
# --------------------------------------------------------------------------- #
def _cos_code(error: Exception) -> str:
    """The COS error code, from a `CosServiceError` or any lookalike."""
    getter = getattr(error, "get_error_code", None)
    if callable(getter):
        try:
            return str(getter() or "")
        except Exception:  # noqa: BLE001 - a fake or a bare exception may not cooperate
            pass
    raw = getattr(error, "code", "")
    return str(raw or type(error).__name__)


def _cos_detail(error: Exception) -> str:
    """Code + status + message, in one line, without guessing at a type."""
    parts = [_cos_code(error)]
    status = getattr(error, "get_status_code", None)
    if callable(status):
        try:
            parts.append(str(status() or ""))
        except Exception:  # noqa: BLE001
            pass
    message = getattr(error, "get_error_msg", None)
    text = str(message() or "") if callable(message) else str(error)
    parts.append(text)
    return " ".join(part for part in parts if part)


def _cos_appid() -> str:
    """The account APPID that COS bucket names are suffixed with.

    Not the Uin: `GetCallerIdentity` returns `100041611714`, while the APPID in
    `pah-parts-1350645027` is a different number. CAM's `GetUserAppId` is the API
    that answers this directly; an explicit `PAH_COS_APPID` skips it for keys that
    cannot read CAM.
    """
    override = os.environ.get(APPID_ENV, "").strip()
    if override:
        return override
    secret_id, secret_key, region = _credentials()
    try:
        from tencentcloud.cam.v20190116 import cam_client, models
        from tencentcloud.common import credential
        from tencentcloud.common.profile.client_profile import ClientProfile
        from tencentcloud.common.profile.http_profile import HttpProfile
    except ImportError as error:
        raise CosError(
            "缺少腾讯云 SDK，无法查询 APPID：请先 `pip install tencentcloud-sdk-python`，"
            f"或直接设置 ${APPID_ENV}。（{type(error).__name__}: {error}）"
        ) from error
    http = HttpProfile()
    http.endpoint = "cam.tencentcloudapi.com"
    profile = ClientProfile()
    profile.httpProfile = http
    client = cam_client.CamClient(
        credential.Credential(secret_id, secret_key), region, profile
    )
    try:
        response = client.GetUserAppId(models.GetUserAppIdRequest())
    except Exception as error:  # noqa: BLE001
        raise CosError(
            f"查询 APPID 失败（CAM GetUserAppId）：{_cos_detail(error)}。"
            f"若这把密钥没有 CAM 读权限，直接设置 ${APPID_ENV}=<APPID>。"
        ) from error
    appid = str(json.loads(response.to_json_string()).get("AppId") or "").strip()
    if not appid:
        raise CosError(f"CAM GetUserAppId 没有返回 AppId；请设置 ${APPID_ENV}")
    return appid


def _cos_client(region: str | None = None) -> Any:
    """A COS client for `region`. The COS SDK is imported lazily for the same reason
    the ai3d SDK is: importing this module must not require an optional package."""
    secret_id, secret_key, default_region = _credentials()
    try:
        from qcloud_cos import CosConfig, CosS3Client
    except ImportError as error:
        raise CosError(
            "缺少 COS SDK：请先 `pip install cos-python-sdk-v5`。"
            f"（{type(error).__name__}: {error}）"
        ) from error
    config = CosConfig(
        Region=region or default_region,
        SecretId=secret_id,
        SecretKey=secret_key,
    )
    return CosS3Client(config)


def _ensure_bucket(client: Any, bucket: str) -> None:
    """Create the bucket if it is not there yet, and only ignore "it is there".

    Not `head_bucket` + create: one call that treats two known "already mine" codes
    as success is less code and less likely to race. Any other code is loud, because
    a silently missing bucket turns into a signing error that looks like a bad key.
    """
    try:
        client.create_bucket(Bucket=bucket)
    except Exception as error:  # noqa: BLE001
        code = _cos_code(error)
        if code in ("BucketAlreadyExists", "BucketAlreadyOwnedByYou", "BucketAlreadyExist"):
            return
        raise CosError(
            f"COS create_bucket({bucket}) 失败：{_cos_detail(error)}。"
            "若提示名字里的 APPID 不对，检查 PAH_COS_APPID。"
        ) from error


def _object_key(source: Path, data: bytes) -> str:
    """`jobs/<utc stamp>-<content hash>.fbx`: unique per content, sortable by time."""
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    digest = hashlib.sha256(data).hexdigest()[:10]
    suffix = source.suffix.lower() or ".fbx"
    return f"jobs/{stamp}-{digest}{suffix}"


def upload_to_cos(
    path: str | os.PathLike,
    *,
    bucket: str | None = None,
    region: str | None = None,
    expire_s: int = 86400,
    key: str | None = None,
    cos: Any = None,
) -> str:
    """Upload a local FBX to COS and return a URL the service can read.

    The object is written to a **private** bucket and handed out as a pre-signed GET
    valid for `expire_s` (24h, the API's own retention for input URLs). Public read is
    never turned on: the URL is the capability, and it dies on its own.
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"要上传的文件不存在：{source}")
    data = source.read_bytes()
    client = cos if cos is not None else _cos_client(region)
    target = bucket or f"{BUCKET_PREFIX}{_cos_appid()}"
    object_key = key or _object_key(source, data)

    _ensure_bucket(client, target)
    try:
        client.put_object(Bucket=target, Body=data, Key=object_key)
    except Exception as error:  # noqa: BLE001
        raise CosError(
            f"COS put_object({target}/{object_key}) 失败：{_cos_detail(error)}"
        ) from error

    try:
        return str(
            client.get_presigned_url(
                Bucket=target, Key=object_key, Method="GET", Expired=int(expire_s)
            )
        )
    except Exception as error:  # noqa: BLE001
        raise CosError(
            f"COS get_presigned_url({target}/{object_key}) 失败：{_cos_detail(error)}"
        ) from error
