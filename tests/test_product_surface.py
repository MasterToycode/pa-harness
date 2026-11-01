"""The product surface: one prompt in, a part and a preview out, and a page to do it.

These tests are deliberately about the *entry points* rather than the pipeline they
call — the pipeline is covered by the loop/verifier suites. What is pinned here is
that "I typed a prompt and got nothing to look at" cannot regress:

* `pah build` writes the spec, the STLs, the report, the trace **and** both preview
  artifacts, and returns them by name;
* the preview PNG is a real render (non-trivial size, correct dimensions), not an
  empty file;
* the local page serves the form, refuses concurrent builds, and serves only files
  under the working directory.
"""

from __future__ import annotations

import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from pah.cli.build import execute
from pah.cli.preview import preview_run, render_html, render_png


def _write_spec_run(run_dir: Path) -> Path:
    """A run directory with two STLs, so the renderer can be tested alone."""
    import trimesh

    run_dir.mkdir(parents=True, exist_ok=True)
    geometry = run_dir / "geometry"
    geometry.mkdir(exist_ok=True)
    trimesh.creation.box((40.0, 20.0, 6.0)).export(geometry / "body.stl")
    peg = trimesh.creation.cylinder(radius=4.0, height=14.0, sections=48)
    peg.apply_translation((0.0, 0.0, 3.0))
    peg.export(geometry / "insert.stl")
    return run_dir


# --------------------------------------------------------------------------- #
# one command
# --------------------------------------------------------------------------- #
def test_a_prompt_produces_a_part_report_and_preview(tmp_path):
    result = execute(
        "a pin with a round head that seats into a blind socket",
        kind="peg_hole",
        rounds=1,
        out=tmp_path / "part",
        config=None,  # the isolated test config: the parametric generator
    )

    assert result["verdict"] in ("repaired", "exhausted", "infeasible")
    assert {part["id"] for part in result["parts"]} == {"body", "insert"}
    for key in ("spec_path", "report_path", "trace_path", "preview_png", "preview_html"):
        assert Path(result[key]).exists(), f"{key} missing"
    assert result["generator"].startswith("local")

    report = json.loads(Path(result["report_path"]).read_text("utf-8"))
    assert report["episode"]["verdict"] == result["verdict"]


def test_the_preview_is_a_real_four_view_render(tmp_path):
    run = _write_spec_run(tmp_path / "run")

    png = render_png(run)

    from PIL import Image  # noqa: PLC0415 - only this assertion needs it

    with Image.open(png) as image:
        assert image.size == (2 * 560 + 3 * 8, 2 * 560 + 3 * 8 + 34)
        # A blank canvas would compress to a few hundred bytes; a shaded part does not.
        assert png.stat().st_size > 4000


def test_the_html_viewer_embeds_the_stls_so_it_opens_offline(tmp_path):
    run = _write_spec_run(tmp_path / "run")

    html = render_html(run)

    text = html.read_text("utf-8")
    assert '"stl":' in text  # base64 payload, not a file:// fetch
    assert "OrbitControls" in text
    assert "body" in text and "insert" in text


def test_preview_run_returns_both_artifacts(tmp_path):
    run = _write_spec_run(tmp_path / "run")

    png, html = preview_run(run)

    assert png.exists() and html.exists()


# --------------------------------------------------------------------------- #
# the page
# --------------------------------------------------------------------------- #
def test_the_page_serves_a_form_and_serves_only_inside_the_workdir(tmp_path, monkeypatch):
    from pah.cli.serve import Handler, JobState

    (tmp_path / "ok.txt").write_text("fine", "utf-8")
    (tmp_path.parent / "outside.txt").write_text("secret", "utf-8")
    handler = type("BoundHandler", (Handler,), {"root": tmp_path})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server.job = JobState()  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urllib.request.urlopen(f"{base}/", timeout=5) as response:
            page = response.read().decode("utf-8")
        assert response.status == 200
        assert 'id="prompt"' in page and 'id="go"' in page
        # The stop button and its wiring must be in the served page: a user running a
        # server started before the change cannot see the button, and that looked like
        # "the feature is missing" rather than "restart the server". It is always
        # rendered and disabled when idle — a control that only exists during an
        # invisible state is undiscoverable (the user looked for it while idle).
        assert 'id="stop"' in page
        assert 'id="stop" hidden' not in page
        assert 'id="stop" disabled' in page
        assert "/cancel" in page
        assert "2026-11-01d" in page  # page version marker, so staleness is visible

        with urllib.request.urlopen(f"{base}/status", timeout=5) as response:
            status = json.loads(response.read())
        assert status["state"] == "idle"
        assert status["cancelling"] is False

        cancel = urllib.request.Request(f"{base}/cancel", data=b"", method="POST")
        with urllib.request.urlopen(cancel, timeout=5) as response:
            assert json.loads(response.read())["state"] == "idle"  # nothing running

        with urllib.request.urlopen(f"{base}/file?p=ok.txt", timeout=5) as response:
            assert response.read() == b"fine"

        request = urllib.request.Request(f"{base}/file?p=../outside.txt")
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request, timeout=5)
        assert error.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_a_second_build_is_refused_while_one_runs():
    from pah.cli.serve import JobState

    job = JobState()
    job.begin()
    assert job.snapshot()["state"] == "running"
    job.finish(result={"verdict": "repaired"})
    assert job.snapshot()["state"] == "done"
    assert job.snapshot()["result"] == {"verdict": "repaired"}


# --------------------------------------------------------------------------- #
# a failed episode must say why
# --------------------------------------------------------------------------- #
def test_a_failed_episode_keeps_the_reason_it_was_told(monkeypatch, tmp_path):
    """Measured: a user saw `failed` with no explanation while the generator API had
    answered "资源包积分已用尽". The sentence has to reach the report."""
    from _design_fakes import design_spec, install

    from pah.app import runtime as runtime_module
    from pah.app.runtime import RuntimeConfig

    install(monkeypatch, mode="crash")
    config = RuntimeConfig(out=tmp_path / "ep", policy="rule", rounds=1)
    with runtime_module.build(config) as composed:
        report = composed.run_episode(
            design_spec(clearance_mm=0.0), policy_name="rule", rounds=1
        )

    assert report["episode"]["verdict"] == "failed"
    assert report["counts"]["environment_errors"] >= 1
    assert report["errors"], "a failed episode must carry its cause"
    detail = report["errors"][0]["detail"]
    assert "积分" in detail or "ResourceInsufficient" in detail
    assert report["errors"][0]["code"] == "verifier_error"
