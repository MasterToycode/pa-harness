"""FBX bridge to Blender: the one format gap between PAH and 混元's part API.

Why this is a separate module
-----------------------------
`SubmitHunyuan3DPartJob` — the cloud face of P3-SAM + X-Part, the only API that
returns a whole object as *separate printable parts* — accepts **only FBX**. PAH
speaks STL and GLB, and no pure-Python writer exists for FBX (trimesh has no FBX
writer; assimp imports FBX but cannot export it; there is no `fbx`/`fbx-sdk` wheel
on PyPI). So the conversion needs a real DCC application, and Blender is the one
that runs headless, free, and scriptable.

That makes Blender an *external tool*, on the same footing as MuJoCo: optional,
resolved by name, and loud when missing. Nothing here silently degrades to a
different format — a 3D job submitted with the wrong container fails server-side
with an error nobody can debug.

Two directions
--------------
* `to_fbx` — PAH parts (`body.stl`, `insert.stl`) -> one multi-object FBX, which
  is what the API's `File` field takes.
* `from_fbx` — one multi-object FBX/OBJ from a decomposition -> one STL per
  object, keyed by the object's own name, so per-part bbox/segmentation data can
  still be joined back to a part.

Units
-----
`global_scale` multiplies on the way out. PAH designs in millimetres; the API
auto-normalises the scale it receives, but a manifest that says what was sent is
the only way to explain a result that comes back sized differently.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

__all__ = [
    "BLENDER_ENV",
    "BlenderMissing",
    "FbxError",
    "blender_exe",
    "from_fbx",
    "to_fbx",
]

#: Environment variable that points at `blender.exe` when it is not on PATH.
BLENDER_ENV = "PAH_BLENDER"

#: The script Blender executes. Resolved next to this file so a wheel install works.
_SCRIPT = Path(__file__).with_name("_blender_fbx.py")


class BlenderMissing(RuntimeError):
    """No Blender binary. Carries the exact fix, because this is a setup step."""


class FbxError(RuntimeError):
    """Blender ran and refused, or produced nothing. Carries its stderr tail."""


def _candidates() -> list[Path]:
    """Where Blender actually lives on this kind of machine, newest last."""
    found: list[Path] = []
    on_path = shutil.which("blender")
    if on_path:
        found.append(Path(on_path))
    for root in (
        os.environ.get("ProgramFiles"),
        os.environ.get("ProgramW6432"),
        os.environ.get("LOCALAPPDATA"),
    ):
        if not root:
            continue
        for pattern in (
            "Blender Foundation/Blender */blender.exe",
            "Programs/Blender Foundation/Blender */blender.exe",
        ):
            found.extend(sorted(Path(root).glob(pattern)))
    return found


def blender_exe(explicit: str | os.PathLike | None = None) -> Path:
    """The Blender to run: `explicit`, then `$PAH_BLENDER`, then PATH, then a scan.

    Raises `BlenderMissing` with the download URL rather than returning `None`: a
    caller that has to check for `None` will one day forget, and the failure would
    surface as an opaque `FileNotFoundError` from `subprocess`.
    """
    for candidate in (explicit, os.environ.get(BLENDER_ENV)):
        if candidate:
            path = Path(candidate)
            if path.is_file():
                return path
            raise BlenderMissing(
                f"{BLENDER_ENV} points at {path}, which is not a file. Point it at "
                "blender.exe, or unset it to let PAH search."
            )
    candidates = _candidates()
    if candidates:
        return candidates[-1]
    raise BlenderMissing(
        "Blender is required to write FBX (混元组件生成 only accepts FBX) and none "
        "was found. Download the portable Windows build from "
        "https://mirrors.tuna.tsinghua.edu.cn/blender/release/Blender4.5/ , unzip "
        f"it anywhere, and set {BLENDER_ENV} to the blender.exe inside."
    )


def _run_blender(payload: dict, timeout: float) -> dict:
    """Run the bridge script headless and return the JSON it printed last."""
    exe = blender_exe()
    command = [
        str(exe),
        "--background",
        "--factory-startup",
        "--python",
        str(_SCRIPT),
        "--",
        json.dumps(payload),
    ]
    try:
        done = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.TimeoutExpired as expired:
        raise FbxError(
            f"Blender did not finish in {timeout:.0f}s converting "
            f"{payload.get('inputs') or payload.get('source')!r}"
        ) from expired

    if done.returncode != 0:
        tail = "\n".join((done.stderr or done.stdout or "").strip().splitlines()[-12:])
        raise FbxError(
            f"Blender exited {done.returncode} for mode {payload.get('mode')!r}. "
            f"Last lines:\n{tail}"
        )

    for line in reversed((done.stdout or "").splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    tail = "\n".join((done.stdout or "").strip().splitlines()[-12:])
    raise FbxError(f"Blender printed no result payload. Last lines:\n{tail}")


def to_fbx(
    sources: list[str | os.PathLike],
    out: str | os.PathLike,
    *,
    global_scale: float = 1.0,
    timeout: float = 900.0,
) -> list[str]:
    """Import `sources` into one scene and write a single multi-object FBX.

    Returns the object names it wrote, in scene order — a manifest entry that lets
    a report say which parts were actually sent, not which ones were meant to be.
    Object names come from the file stems, so `body.stl` arrives as `body`.
    """
    paths = [Path(source) for source in sources]
    if not paths:
        raise ValueError("to_fbx needs at least one source file")
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"to_fbx sources do not exist: {missing}")
    payload = {
        "mode": "to",
        "inputs": [str(path.resolve()) for path in paths],
        "out": str(Path(out).resolve()),
        "global_scale": float(global_scale),
    }
    result = _run_blender(payload, timeout)
    return [str(name) for name in result.get("objects", [])]


def from_fbx(
    source: str | os.PathLike,
    out_dir: str | os.PathLike,
    *,
    global_scale: float = 1.0,
    timeout: float = 900.0,
) -> dict[str, str]:
    """Split one multi-object FBX/OBJ/GLB into `{object name: stl path}`.

    The object names are the identity a decomposition is keyed on. They are
    returned verbatim — not slugified — because the segmentation JSON from the same
    job uses them, and a renamed part is a part whose bbox no longer matches.
    """
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"from_fbx source does not exist: {path}")
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    payload = {
        "mode": "from",
        "source": str(path.resolve()),
        "out_dir": str(target.resolve()),
        "global_scale": float(global_scale),
    }
    result = _run_blender(payload, timeout)
    parts = result.get("parts") or []
    if not parts:
        raise FbxError(f"{path.name} carried no mesh objects")
    return {str(item["name"]): str(item["stl"]) for item in parts}


def describe() -> dict[str, str]:
    """One row for a doctor table: which Blender would be used, or why none is."""
    try:
        exe = blender_exe()
    except BlenderMissing as error:
        return {"blender": "missing", "detail": str(error).splitlines()[0]}
    return {"blender": "ok", "path": str(exe)}


if __name__ == "__main__":  # a human checking the setup, not a code path
    print(json.dumps(describe(), indent=2, ensure_ascii=False))
    sys.exit(0 if describe().get("blender") == "ok" else 1)
