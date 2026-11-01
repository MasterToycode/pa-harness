"""`pah serve` — the local page where you type a prompt and watch the harness work.

Why a server instead of more CLI flags: the pipeline takes seconds to minutes (a
generation call, several verifiers), produces four kinds of artifact, and is easier to
understand while it runs than after. A page gives one input, a live log, and the part
in front of you — and it needs no dependencies beyond the standard library, so it
works offline and cannot rot.

    pah serve                 # http://127.0.0.1:8321
    pah serve --open          # ... and open the browser

Bound to localhost on purpose: this is a control surface for paid API calls, not a
service. One job at a time, because two concurrent generations would make the log —
and the "which part am I looking at" question — ambiguous.

Routes: `/` (page), `POST /build` (start), `GET /status` (poll), `GET /file?p=...`
(artifacts under the working directory only).
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import traceback
import urllib.parse
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

EXIT_OK = 0
EXIT_CRASH = 3

KINDS = ("peg_hole", "snap_fit", "dovetail")
POLICIES = ("rule", "random", "llm")


class JobState:
    """The single in-flight build: status, log lines, and the last result."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.state = "idle"  # idle | running | done | error
        self.log: list[str] = []
        self.result: dict | None = None
        self.started = 0.0
        self.finished = 0.0
        #: Set by the page's stop button; the loop asks it once per round. It cannot
        #: abort a generation that has already been submitted (that one is billed
        #: either way), but it guarantees no *further* round starts — which is what a
        #: mis-click actually costs.
        self.cancel_event = threading.Event()

    def begin(self) -> None:
        with self.lock:
            self.state = "running"
            self.log = ["starting…"]
            self.result = None
            self.started = time.time()
            self.finished = 0.0
            self.cancel_event.clear()

    def cancel(self) -> None:
        """Ask the running build to stop after the current round."""
        with self.lock:
            if self.state != "running":
                return
            self.cancel_event.set()
            self.log.append("cancel requested: no further round will start "
                            "(a generation already submitted is billed)")

    def note(self, line: str) -> None:
        with self.lock:
            self.log.append(line)
            self.log = self.log[-200:]

    def finish(self, *, result: dict | None, error: str | None = None) -> None:
        with self.lock:
            self.state = "error" if error else "done"
            self.result = result
            if error:
                self.log.append(f"FAILED: {error}")
            self.finished = time.time()
            self.cancel_event.clear()

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "state": self.state,
                "log": list(self.log),
                "result": self.result,
                "cancelling": self.cancel_event.is_set(),
                "elapsed": round((self.finished or time.time()) - self.started, 1)
                if self.started
                else 0.0,
            }


def _run_job(job: JobState, fields: dict[str, str]) -> None:
    """Worker thread: compose the same pipeline `pah build` runs, logging as it goes."""
    from pah.cli.build import execute  # noqa: PLC0415

    job.begin()
    try:
        result = execute(
            fields.get("prompt", "").strip(),
            kind=fields.get("kind", "peg_hole"),
            rounds=int(fields.get("rounds", "1") or 1),
            policy=fields.get("policy", "rule"),
            out=fields.get("out") or None,
            config=fields.get("config") or None,
            progress=job.note,
            cancel=job.cancel_event.is_set,
        )
        job.finish(result=result)
    except Exception as error:  # noqa: BLE001 - the page shows the sentence
        job.note(traceback.format_exc(limit=3))
        job.finish(result=None, error=f"{type(error).__name__}: {error}")


_PAGE = """<!doctype html>
<html lang="zh"><meta charset="utf-8">
<title>PAH · 生成可装配部件</title>
<style>
  :root {{ color-scheme: dark; }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; background:#0f141b; color:#dfe7f0;
         font-family: ui-sans-serif, system-ui, "Segoe UI", sans-serif; }}
  header {{ padding:18px 26px; border-bottom:1px solid #22303f; display:flex; gap:16px;
           align-items:baseline; }}
  h1 {{ font-size:17px; margin:0; font-weight:600; letter-spacing:.2px; }}
  .sub {{ font-size:12px; color:#8ba0b5; }}
  main {{ display:grid; grid-template-columns: 430px 1fr; gap:22px; padding:22px 26px 40px; }}
  label {{ display:block; font-size:12px; color:#8ba0b5; margin:14px 0 6px; }}
  textarea, select, input {{ width:100%; background:#141b24; color:#dfe7f0;
      border:1px solid #263444; border-radius:8px; padding:10px 12px; font-size:13px;
      font-family:inherit; }}
  textarea {{ min-height:96px; resize:vertical; line-height:1.5; }}
  .row {{ display:flex; gap:10px; }}
  .row > div {{ flex:1; }}
  button {{ margin-top:16px; width:100%; padding:11px 14px; border-radius:8px;
      border:1px solid #2f6f9f; background:#1b4b6e; color:#eaf4ff; font-size:14px;
      font-weight:600; cursor:pointer; }}
  button:disabled {{ opacity:.55; cursor:progress; }}
  #stop {{ border-color:#a35a3a; background:#5a2f1e; color:#ffe0cf; margin-top:8px; }}
  #stop[hidden] {{ display:none; }}
  #status {{ font-size:12px; color:#8ba0b5; margin-top:10px; min-height:16px; }}
  #log {{ margin-top:12px; background:#0b1017; border:1px solid #1d2836; border-radius:8px;
      padding:10px; font-family: ui-monospace, Consolas, monospace; font-size:11.5px;
      line-height:1.5; color:#9fb3c8; height:220px; overflow:auto; white-space:pre-wrap; }}
  .panel {{ background:#121a23; border:1px solid #22303f; border-radius:10px; padding:16px; }}
  .panel h2 {{ font-size:13px; margin:0 0 10px; color:#b8c8d8; font-weight:600; }}
  #preview {{ width:100%; border-radius:8px; background:#0b1017; display:block; }}
  #artifacts {{ font-size:12px; line-height:1.9; margin-top:12px; }}
  #artifacts a {{ color:#7fc0e8; text-decoration:none; }}
  #artifacts a:hover {{ text-decoration:underline; }}
  #verdict {{ display:inline-block; padding:3px 9px; border-radius:999px; font-size:12px;
      border:1px solid #2f6f9f; color:#bde3ff; margin-left:8px; }}
  #verdict.bad {{ border-color:#9f4a2f; color:#ffc9b3; }}
  .note {{ font-size:12px; color:#7f93a8; margin-top:8px; }}
</style>
<header>
  <h1>PAH · 生成可装配部件</h1>
  <span class="sub">描述你想要的零件 → 模型生成 → 六类验证 → 不合格自动修 → 出图</span>
</header>
<main>
  <section class="panel">
    <h2>1 · 需求</h2>
    <label for="prompt">一句话描述</label>
    <textarea id="prompt">a 3D-printable bracket joint: a chamfered pin with a flat round head sliding into a blind socket</textarea>
    <div class="row">
      <div><label for="kind">连接件</label>
        <select id="kind">{kinds}</select></div>
      <div><label for="rounds">轮数（每轮一次生成）</label>
        <input id="rounds" type="number" min="1" max="5" value="1"></div>
    </div>
    <div class="row">
      <div><label for="policy">修复策略</label>
        <select id="policy">{policies}</select></div>
      <div><label for="config">pah.toml</label>
        <input id="config" placeholder="（默认 ./pah.toml）" value="{config}"></div>
    </div>
    <button id="go">开始生成</button>
    <button id="stop" hidden>停止（不再开始新的轮次）</button>
    <div id="status">就绪</div>
    <div id="log" hidden></div>
    <div class="note">每轮会调用一次生成服务（混元）。一次 build 最多 <b>1 + 轮数</b> 次生成：
      首轮物化 1 次，之后每个修复轮各 1 次。点错了可以按「停止」，但已经提交给云端的那一次仍会计费。</div>
  </section>
  <section class="panel">
    <h2>2 · 结果</h2>
    <img id="preview" alt="预览会显示在这里" src="">
    <div id="artifacts">还没有结果。</div>
  </section>
</main>
<script>
const $ = id => document.getElementById(id);
let timer = null;

async function start() {{
  const body = new URLSearchParams({{
    prompt: $("prompt").value, kind: $("kind").value, rounds: $("rounds").value,
    policy: $("policy").value, config: $("config").value
  }});
  $("go").disabled = true; $("status").textContent = "提交中…";
  $("log").hidden = false; $("log").textContent = "";
  await fetch("/build", {{ method: "POST", body }});
  poll();
}}

function poll() {{
  clearTimeout(timer);
  fetch("/status").then(r => r.json()).then(s => {{
    $("log").textContent = s.log.join("\\n");
    $("log").scrollTop = $("log").scrollHeight;
    $("status").textContent = s.state === "running" ? `生成中… ${{s.elapsed}}s` :
      (s.state === "done" ? `完成，用时 ${{s.elapsed}}s` :
       (s.state === "error" ? `失败，用时 ${{s.elapsed}}s` : "就绪"));
    // A finished job survives a page reload: the server keeps the last result, so
    // refreshing (or reopening the page later) still shows the part you just built.
    if (s.result) renderResult(s.result);
    if (s.state === "running") {{
      $("stop").hidden = false;
      $("stop").disabled = !!s.cancelling;
      $("stop").textContent = s.cancelling ? "正在停止…" : "停止（不再开始新的轮次）";
      timer = setTimeout(poll, 1000);
      return;
    }}
    $("stop").hidden = true;
    $("go").disabled = false;
  }});
}}

function renderResult(r) {{
  const stamp = Date.now();
  $("log").hidden = false;
  $("preview").src = "/file?p=" + encodeURIComponent(r.preview_png || "") + "&t=" + stamp;
  const verdict = r.verdict || "?";
  const bad = (verdict === "failed" || verdict === "exhausted") ? "bad" : "";
  $("artifacts").innerHTML = `
    <div>判定 <span id="verdict" class="${{bad}}">${{verdict}}</span>
         · 轮数 ${{r.rounds}} · 生成器 ${{r.generator || "?"}} · ${{r.elapsed_s || "?"}}s</div>
    <div>网格 <a href="/file?p=${{encodeURIComponent(r.preview_html || "")}}" target="_blank">在浏览器里旋转查看</a></div>
    <div>STL：${{(r.parts || []).map(p => `<a href="/file?p=${{encodeURIComponent(p.path)}}">${{p.id}}</a>`).join(" · ")}}</div>
    <div>报告 <a href="/file?p=${{encodeURIComponent(r.report_path || "")}}">report.json</a>
         · <a href="/file?p=${{encodeURIComponent(r.trace_path || "")}}">trace.jsonl</a>
         · <a href="/file?p=${{encodeURIComponent(r.spec_path || "")}}">spec.json</a></div>
    ${{(r.notes || []).map(n => `<div class="note">未评估 · ${{n}}</div>`).join("")}}
  `;
}}

$("go").onclick = start;
$("stop").onclick = () => {{
  $("stop").disabled = true;
  $("stop").textContent = "正在停止…";
  fetch("/cancel", {{ method: "POST" }}).then(poll);
}};
poll();   // pick up a result that already exists
</script>
"""


class Handler(BaseHTTPRequestHandler):
    server_version = "pah/1"
    root: Path = Path(".")

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - stdlib signature
        return  # the page has its own log; stderr stays readable

    # -- helpers -----------------------------------------------------------
    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: dict, status: int = 200) -> None:
        self._send(status, json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _under_root(self, raw: str) -> Path | None:
        """A path the page may serve: inside the working directory, and existing."""
        if not raw:
            return None
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = self.root / candidate
        try:
            resolved = candidate.resolve()
        except OSError:
            return None
        if not str(resolved).startswith(str(self.root.resolve())):
            return None
        return resolved if resolved.exists() else None

    # -- routes ------------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/":
            page = _PAGE.format(
                kinds="".join(f'<option value="{k}">{k}</option>' for k in KINDS),
                policies="".join(f'<option value="{p}">{p}</option>' for p in POLICIES),
                config="pah.toml",
            )
            self._send(HTTPStatus.OK, page.encode("utf-8"), "text/html; charset=utf-8")
            return
        if parsed.path == "/status":
            self._json(self.server.job.snapshot())  # type: ignore[attr-defined]
            return
        if parsed.path == "/file":
            query = urllib.parse.parse_qs(parsed.query)
            target = self._under_root((query.get("p") or [""])[0])
            if target is None:
                self._json({"error": "no such artifact"}, 404)
                return
            suffix = target.suffix.lower()
            kind = {
                ".png": "image/png",
                ".html": "text/html; charset=utf-8",
                ".json": "application/json; charset=utf-8",
                ".jsonl": "text/plain; charset=utf-8",
                ".stl": "model/stl",
                ".glb": "model/gltf-binary",
                ".txt": "text/plain; charset=utf-8",
            }.get(suffix, "application/octet-stream")
            self._send(HTTPStatus.OK, target.read_bytes(), kind)
            return
        self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:  # noqa: N802 - stdlib naming
        parsed = urllib.parse.urlparse(self.path)
        job: JobState = self.server.job  # type: ignore[attr-defined]
        if parsed.path == "/cancel":
            job.cancel()
            self._json(job.snapshot())
            return
        if parsed.path != "/build":
            self._json({"error": "not found"}, 404)
            return
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length).decode("utf-8", "replace")
        fields = {k: v[0] for k, v in urllib.parse.parse_qs(body).items()}
        with job.lock:
            running = job.state == "running"
        if running:
            self._json({"error": "a build is already running"}, 409)
            return
        threading.Thread(target=_run_job, args=(job, fields), daemon=True).start()
        self._json({"started": True})


def add_parser(groups: argparse._SubParsersAction) -> None:
    serve = groups.add_parser("serve", help="local page: type a prompt, watch the part appear")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8321)
    serve.add_argument("--config", default="pah.toml", help="pre-filled pah.toml path")
    serve.add_argument("--no-open", action="store_true")
    serve.set_defaults(handler=_serve)


def _serve(args: argparse.Namespace) -> int:
    global _PAGE
    _PAGE = _PAGE.replace("{config}", "") if False else _PAGE  # keep placeholder single-source
    handler = type("BoundHandler", (Handler,), {"root": Path.cwd()})
    httpd = ThreadingHTTPServer((args.host, args.port), handler)
    httpd.job = JobState()  # type: ignore[attr-defined]
    url = f"http://{args.host}:{args.port}/"
    print(f"pah serve: {url}   (Ctrl+C 停止)")
    print(f"pah.toml : {Path(args.config).resolve()}")
    if not args.no_open:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\npah serve: stopped")
    finally:
        httpd.server_close()
    return EXIT_OK
