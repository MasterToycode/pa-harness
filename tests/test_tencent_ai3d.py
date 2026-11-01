"""组件生成 client: everything except the paid call, with the SDK faked.

The real service is behind two paywalls (a Tencent Cloud account and per-job
credits), so nothing here may touch the network. Every test injects a fake SDK
client instead: the request objects are still the real generated models, which is
what makes "we send FBX + a URL" a meaningful assertion rather than a tautology.

What is left untested is only the network round-trip itself.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pah.tools import tencent_ai3d
from pah.tools.tencent_ai3d import (
    Ai3dError,
    CosError,
    PartJobFailed,
    PartJobTimeout,
    download_results,
    parse_segmentation,
    query_part_job,
    submit_part_job,
    upload_to_cos,
    wait_part_job,
)

# --------------------------------------------------------------------------- #
# fakes: the two SDK clients, nothing else
# --------------------------------------------------------------------------- #
class FakeResponse:
    """What the generated SDK returns: an object that serialises itself to JSON."""

    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def to_json_string(self) -> str:
        return json.dumps(self._payload)


class FakeAi3d:
    """Records the real request objects; replays scripted query responses."""

    def __init__(self, *, submit: dict | None = None, queries: list[dict] | None = None) -> None:
        self.submit_response = submit if submit is not None else {"JobId": "job-1"}
        self.queries = list(queries or [{"Status": "DONE"}])
        self.submitted: list[object] = []
        self.queried: list[str] = []

    def SubmitHunyuan3DPartJob(self, request):  # noqa: N802 - the SDK's own name
        self.submitted.append(request)
        return FakeResponse(self.submit_response)

    def QueryHunyuan3DPartJob(self, request):  # noqa: N802
        self.queried.append(request.JobId)
        payload = self.queries.pop(0) if len(self.queries) > 1 else self.queries[0]
        return FakeResponse(payload)


class FakeServiceError(Exception):
    """A `CosServiceError` lookalike: the client reads code/message off it."""

    def __init__(self, code: str, message: str = "", status: int = 403) -> None:
        super().__init__(message or code)
        self._code = code
        self._message = message or code
        self._status = status

    def get_error_code(self) -> str:
        return self._code

    def get_error_msg(self) -> str:
        return self._message

    def get_status_code(self) -> int:
        return self._status


class FakeCos:
    """A private-bucket, pre-signed-URL COS. `put_bucket_acl` raises if ever called."""

    def __init__(self, *, create_error: Exception | None = None) -> None:
        self.create_error = create_error
        self.created: list[str] = []
        self.puts: list[dict] = []
        self.presigned: list[dict] = []

    def create_bucket(self, Bucket):  # noqa: N802, N803 - COS SDK naming
        self.created.append(Bucket)
        if self.create_error is not None:
            raise self.create_error
        return {}

    def put_object(self, Bucket, Body, Key):  # noqa: N802, N803
        self.puts.append({"bucket": Bucket, "body": Body, "key": Key})
        return {}

    def get_presigned_url(self, Bucket, Key, Method, Expired):  # noqa: N802, N803
        self.presigned.append(
            {"bucket": Bucket, "key": Key, "method": Method, "expired": Expired}
        )
        return f"https://{Bucket}.cos.ap-guangzhou.myqcloud.com/{Key}?sign=stub"

    def put_bucket_acl(self, **_):  # noqa: D401 - a guard, not a feature
        raise AssertionError("the bucket must stay private; only the URL is handed out")


@pytest.fixture()
def fbx(tmp_path: Path) -> Path:
    path = tmp_path / "000.fbx"
    path.write_bytes(b"Kaydara FBX Binary\x00" * 128)
    return path


# --------------------------------------------------------------------------- #
# submit: what actually goes on the wire
# --------------------------------------------------------------------------- #
def test_submit_sends_an_fbx_url_and_returns_the_job_id():
    fake = FakeAi3d(submit={"JobId": "job-42", "RequestId": "req"})

    job_id = submit_part_job("https://b.cos.ap-guangzhou.myqcloud.com/jobs/a.fbx?sign=1", client=fake)

    assert job_id == "job-42"
    sent = fake.submitted[0]
    assert sent.File.Type == "FBX"  # the only accepted container
    assert sent.File.Url.endswith("a.fbx?sign=1")
    assert sent.Model == "1.5"
    assert sent.EnableStagedGeneration is False
    assert sent.EnablePostProcess is False


def test_an_empty_job_id_is_an_error_not_a_job_to_poll():
    fake = FakeAi3d(submit={"RequestId": "req"})

    with pytest.raises(Ai3dError, match="JobId"):
        submit_part_job("https://x/a.fbx", client=fake)


def test_the_two_costly_flags_are_off_by_default_and_pass_through_when_set():
    fake = FakeAi3d(submit={"JobId": "j"})
    submit_part_job("https://x/a.fbx", staged=True, post_process=True, client=fake)

    sent = fake.submitted[0]
    assert sent.EnableStagedGeneration is True
    assert sent.EnablePostProcess is True


def test_a_local_path_is_refused_before_it_costs_a_job(tmp_path: Path):
    """There is no byte channel on this action; failing locally beats a paid 400."""
    missing = tmp_path / "000.fbx"
    with pytest.raises(ValueError, match="upload_to_cos"):
        submit_part_job(str(missing), client=FakeAi3d())


# --------------------------------------------------------------------------- #
# query + poll
# --------------------------------------------------------------------------- #
def test_query_returns_the_response_verbatim():
    payload = {
        "Status": "DONE",
        "ResultFile3Ds": [{"Type": "FBX", "Url": "https://x/a.fbx"}, {"Type": "OBJ", "Url": "https://x/b.obj"}],
        "PartSegmentationInfoUrl": "https://x/seg.json",
        "RequestId": "req-1",
    }
    result = query_part_job("job-1", client=FakeAi3d(queries=[payload]))

    assert result == payload
    assert len(result["ResultFile3Ds"]) == 2


def test_wait_polls_until_done_and_returns_the_last_response():
    done = {"Status": "DONE", "ResultFile3Ds": [{"Type": "FBX", "Url": "https://x/a.fbx"}]}
    fake = FakeAi3d(queries=[{"Status": "WAIT"}, {"Status": "RUN"}, done])

    result = wait_part_job("job-1", timeout_s=5, interval_s=0, client=fake)

    assert result == done
    assert fake.queried == ["job-1", "job-1", "job-1"]


def test_every_poll_is_reported_so_a_long_job_is_not_a_silent_wait():
    fake = FakeAi3d(queries=[{"Status": "WAIT"}, {"Status": "RUN"}, {"Status": "DONE"}])
    seen: list[str] = []

    wait_part_job("job-1", timeout_s=5, interval_s=0, client=fake, on_poll=lambda r: seen.append(r["Status"]))

    assert seen == ["WAIT", "RUN", "DONE"]


def test_a_failed_job_raises_with_both_the_code_and_the_message():
    fake = FakeAi3d(
        queries=[
            {"Status": "RUN"},
            {
                "Status": "FAIL",
                "ErrorCode": "InvalidParameter.File",
                "ErrorMessage": "the file must be FBX",
            },
        ]
    )

    with pytest.raises(PartJobFailed) as error:
        wait_part_job("job-1", timeout_s=5, interval_s=0, client=fake)

    assert error.value.code == "InvalidParameter.File"
    assert error.value.message == "the file must be FBX"
    assert error.value.job_id == "job-1"
    assert "InvalidParameter.File" in str(error.value)
    assert "the file must be FBX" in str(error.value)


def test_a_timeout_keeps_the_job_id_and_never_submits_again():
    """Re-querying is free; re-submitting spends credits, so the failure text has to
    tell the next run what to query instead of what to submit."""
    fake = FakeAi3d(queries=[{"Status": "RUN"}])

    with pytest.raises(PartJobTimeout) as error:
        wait_part_job("job-running", timeout_s=0, interval_s=0, client=fake)

    assert "job-running" in str(error.value)
    assert "不要重新提交" in str(error.value)
    assert fake.submitted == []


def test_an_unknown_status_is_loud_rather_than_polled_forever():
    fake = FakeAi3d(queries=[{"Status": "TEAPOT"}])

    with pytest.raises(Ai3dError, match="未知状态"):
        wait_part_job("job-1", timeout_s=1, interval_s=0, client=fake)


# --------------------------------------------------------------------------- #
# download
# --------------------------------------------------------------------------- #
def _downloader(payloads: dict[str, bytes]):
    seen: list[str] = []

    def fetch(url: str) -> bytes:
        seen.append(url)
        return payloads[url]

    fetch.seen = seen  # type: ignore[attr-defined]
    return fetch


def test_every_result_file_and_the_segmentation_are_written(tmp_path: Path):
    result = {
        "Status": "DONE",
        "ResultFile3Ds": [
            {"Type": "FBX", "Url": "https://cos/a.fbx?sign=1"},
            {"Type": "OBJ", "Url": "https://cos/b.obj?sign=2"},
        ],
        "PartSegmentationInfoUrl": "https://cos/seg?sign=3",
    }
    fetch = _downloader(
        {
            "https://cos/a.fbx?sign=1": b"FBX-A",
            "https://cos/b.obj?sign=2": b"OBJ-B",
            "https://cos/seg?sign=3": b'{"parts": 3}',
        }
    )

    written = download_results(result, tmp_path / "out", fetch=fetch)

    assert len(written["files"]) == 2
    assert Path(written["files"][0]["path"]).name == "part_00.fbx"
    assert Path(written["files"][1]["path"]).name == "part_01.obj"
    assert Path(written["files"][0]["path"]).read_bytes() == b"FBX-A"
    # URLs are kept: a short download can be retried without paying for a new job.
    assert written["files"][0]["url"].startswith("https://cos/a.fbx")

    segmentation = written["segmentation"]
    assert segmentation is not None
    assert Path(segmentation["path"]).name == "part_segmentation_info.bin"
    assert json.loads(Path(segmentation["path"]).read_text("utf-8")) == {"parts": 3}
    assert fetch.seen == [
        "https://cos/a.fbx?sign=1",
        "https://cos/b.obj?sign=2",
        "https://cos/seg?sign=3",
    ]


def test_a_suffixless_signed_url_falls_back_to_the_declared_type(tmp_path: Path):
    """CDNs rewrite paths; `Type` is the service's own statement about the format."""
    result = {"ResultFile3Ds": [{"Type": "GLB", "Url": "https://cos/object?sign=1"}]}

    written = download_results(result, tmp_path, fetch=_downloader({"https://cos/object?sign=1": b"x"}))

    assert Path(written["files"][0]["path"]).suffix == ".glb"


def test_a_result_with_no_files_writes_nothing_and_says_so(tmp_path: Path):
    written = download_results({"Status": "DONE", "ResultFile3Ds": []}, tmp_path, fetch=_downloader({}))

    assert written["files"] == []
    assert written["segmentation"] is None


def test_a_file_entry_without_a_url_is_an_error_not_a_skipped_file():
    """Silently dropping it would report "2 parts" for a job that returned 3."""
    result = {"ResultFile3Ds": [{"Type": "FBX"}, {"Type": "FBX", "Url": "https://cos/b.fbx"}]}

    with pytest.raises(Ai3dError, match="没有 Url"):
        download_results(result, ".", fetch=_downloader({}))


# --------------------------------------------------------------------------- #
# the segmentation blob is binary, and that is not an error
# --------------------------------------------------------------------------- #
def test_the_lseg_blob_decodes_to_one_label_per_face():
    """Measured on the first real job: `PartSegmentationInfoUrl` is not JSON, it is
    b"LSEG" | version | count | count * uint8. Reading it as UTF-8 fails on byte 9."""
    payload = b"LSEG" + (1).to_bytes(4, "little") + (6).to_bytes(4, "little") + bytes([0, 0, 1, 2, 2, 2])

    decoded = parse_segmentation(payload)

    assert decoded["format"] == "LSEG"
    assert decoded["version"] == 1
    assert decoded["count"] == 6
    assert decoded["labels"] == bytes([0, 0, 1, 2, 2, 2])
    assert decoded["histogram"] == {"0": 2, "1": 1, "2": 3}


def test_a_json_segmentation_payload_is_still_accepted():
    """The field is documented without a schema; a future version that sends JSON
    must not be reported as corruption."""
    decoded = parse_segmentation(b'{"parts": [{"id": 0}]}')

    assert decoded["format"] == "json"
    assert decoded["data"] == {"parts": [{"id": 0}]}


def test_an_unknown_segmentation_blob_is_loud_about_what_it_saw():
    with pytest.raises(Ai3dError, match="LSEG"):
        parse_segmentation(b"\x00\x01\x02\x03 unrelated")

    with pytest.raises(Ai3dError):
        parse_segmentation(b"LSEG" + (1).to_bytes(4, "little") + (9).to_bytes(4, "little") + b"ab")


# --------------------------------------------------------------------------- #
# upload: private bucket, signed URL
# --------------------------------------------------------------------------- #
def test_upload_creates_the_bucket_puts_the_file_and_returns_a_signed_url(fbx: Path):
    fake = FakeCos()

    url = upload_to_cos(fbx, bucket="pah-parts-1350645027", cos=fake, expire_s=86400)

    assert fake.created == ["pah-parts-1350645027"]
    put = fake.puts[0]
    assert put["bucket"] == "pah-parts-1350645027"
    assert put["body"] == fbx.read_bytes()
    assert put["key"].startswith("jobs/") and put["key"].endswith(".fbx")
    assert fake.presigned[0]["method"] == "GET"
    assert fake.presigned[0]["expired"] == 86400
    assert url.startswith("https://pah-parts-1350645027.cos.ap-guangzhou.myqcloud.com/jobs/")


def test_an_existing_bucket_is_not_an_error(fbx: Path):
    fake = FakeCos(create_error=FakeServiceError("BucketAlreadyOwnedByYou", "mine"))

    url = upload_to_cos(fbx, bucket="pah-parts-1", cos=fake)

    assert "sign=" in url
    assert len(fake.puts) == 1


def test_a_bucket_failure_that_is_not_already_mine_names_the_code_and_bucket(fbx: Path):
    fake = FakeCos(create_error=FakeServiceError("AccessDenied", "no cos:PutBucket", 403))

    with pytest.raises(CosError) as error:
        upload_to_cos(fbx, bucket="pah-parts-1", cos=fake)

    message = str(error.value)
    assert "AccessDenied" in message
    assert "pah-parts-1" in message
    assert fake.puts == []  # nothing uploaded into a bucket that said no


def test_the_object_key_changes_with_the_content(fbx: Path, tmp_path: Path):
    other = tmp_path / "other.fbx"
    other.write_bytes(b"different bytes")

    first = upload_to_cos(fbx, bucket="b", cos=FakeCos())
    second = upload_to_cos(other, bucket="b", cos=FakeCos())

    assert first.split("?")[0] != second.split("?")[0]


# --------------------------------------------------------------------------- #
# refusals (still no network)
# --------------------------------------------------------------------------- #
def test_missing_keys_name_the_variables_and_the_dotenv(monkeypatch):
    monkeypatch.delenv(tencent_ai3d.SECRET_ID_ENV, raising=False)
    monkeypatch.delenv(tencent_ai3d.SECRET_KEY_ENV, raising=False)

    with pytest.raises(Ai3dError) as error:
        submit_part_job("https://x/a.fbx")

    message = str(error.value)
    assert tencent_ai3d.SECRET_ID_ENV in message
    assert tencent_ai3d.SECRET_KEY_ENV in message
    assert ".env" in message


def test_upload_without_keys_fails_before_any_bucket_call(fbx: Path, monkeypatch):
    monkeypatch.delenv(tencent_ai3d.SECRET_ID_ENV, raising=False)
    monkeypatch.delenv(tencent_ai3d.SECRET_KEY_ENV, raising=False)

    with pytest.raises(Ai3dError, match=tencent_ai3d.SECRET_KEY_ENV):
        upload_to_cos(fbx)


def test_a_missing_local_file_is_a_file_error_not_a_cloud_error(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        upload_to_cos(tmp_path / "nope.fbx", bucket="b", cos=FakeCos())


# --------------------------------------------------------------------------- #
# the proxy trap (measured: reset connection, reported as a credential error)
# --------------------------------------------------------------------------- #
def test_tencent_domains_bypass_the_developer_proxy(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "api.deepseek.com")
    monkeypatch.delenv("no_proxy", raising=False)

    tencent_ai3d._bypass_proxy_for_service()

    entries = {"api.deepseek.com"} | {part.strip() for part in tencent_ai3d.os.environ["NO_PROXY"].split(",")}
    for host in ("tencentcloudapi.com", "myqcloud.com"):
        assert host in entries
        assert f"*.{host}" in entries
    assert tencent_ai3d.os.environ["no_proxy"] == tencent_ai3d.os.environ["NO_PROXY"]


def test_importing_the_module_already_set_the_bypass():
    entries = {part.strip() for part in tencent_ai3d.os.environ.get("NO_PROXY", "").split(",")}
    assert "tencentcloudapi.com" in entries


# --------------------------------------------------------------------------- #
# the contract this module was verified against
# --------------------------------------------------------------------------- #
def test_the_service_version_is_pinned():
    """A version is not a detail: an adapter that forgets which one it speaks drifts
    silently, and the drift shows up as a shape change in a report."""
    assert tencent_ai3d.API_VERSION == "2025-05-13"
    assert tencent_ai3d.ENDPOINT == "ai3d.tencentcloudapi.com"
    assert tencent_ai3d.MODEL == "1.5"
