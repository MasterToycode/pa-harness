"""`connector_feasible`: adapter over `rl_assembly` `connector.py`.

Reuse boundary (§2.4): parameter translation, return-value translation, exception
translation. Nothing here decides geometry — every code below renames a verdict
`connector.build` already reached, and the one place this module computes
anything (`_theta`) is copied verbatim from the benchmark that produced the
gold standard (`tests/asm_final.py:209-213`).

Two facts about the source data drive the whole translation, both measured, both
documented in `docs/internal/adapter-contract.md`:

1. `edges_v2.json` mixes two length frames. `center_mm` / `anchor_mm` are
   millimetres *in the 150 mm base frame*; `r_*_mm` / `t_*_mm` / `half_*_mm` are
   millimetres *at the 225 mm final scale*. So coordinates divide by 1000 and
   derived lengths divide by `unit_mm` — the same `unit_mm` handed to
   `connector.build`, which is what keeps `r_min` honest.
2. Booleans and thickness must share the undecimated geometry
   (`assemblies_pre_decimate`, 150 mm). That is settled in
   `research/connector-boolean-report.md` §18.3 and is why `unit_mm` is `1500`
   here rather than `1000` (see `SCALE_MM` below).

One fact about the *machine* does too: the gold run's 388 successes include five
edges it lost to `MemoryError` (`asm_000097` ids 912/916/917/919/920), so 388 is
only stable to ±5. `_call_build` therefore retries an out-of-memory verdict twice
before recording `verifier_oom` — a code of its own, counted apart from
`bool_failed`, because an environment failure and a geometry failure are not the
same finding.
"""

from __future__ import annotations

import datetime as dt
import functools
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterator, Literal

from pah.core.model import Assembly, FailureCertificate

from .base import Case, VerifierSpec

__all__ = [
    "CODES",
    "EDGE_FILE",
    "GEOMETRY_ROOT",
    "OOM_RETRIES",
    "ConnectorFeasible",
    "case_parts",
    "classify",
    "is_oom",
    "load_cases",
    "oom_stats",
    "reset_oom_stats",
    "to_edge_dict",
]

# --------------------------------------------------------------------------- #
# Asset locations — read-only, overridable so a run can be pointed at a copy
# --------------------------------------------------------------------------- #
RL_REPO = Path(os.environ.get("PAH_RL_ASSEMBLY", r"D:\yanyi\rl_assembly"))
RL_CODE = RL_REPO / "code"
#: `data/` is a junction to `E:`; go through it so the path survives a machine
#: move (`adapter-contract.md` §E10).
GEOMETRY_ROOT = Path(
    os.environ.get("PAH_GEOMETRY", str(RL_REPO / "data" / "assemblies_pre_decimate"))
)
EDGE_FILE = Path(
    os.environ.get(
        "PAH_EDGES", str(RL_REPO / "research" / "contact-graph-anchor" / "edges_v2.json")
    )
)

# --------------------------------------------------------------------------- #
# The scale protocol (measured, not chosen — see the module docstring)
# --------------------------------------------------------------------------- #
#: `edges_v2.json` declares `scale_mm: 225` and computes its derived lengths at
#: that scale, while the geometry it must be combined with is the 150 mm
#: undecimated set. One data unit there (a metre) is therefore
#: `SCALE_MM / longest_side` millimetres of the 225 mm frame — measured ≈1500.
#:
#: The two candidates that lose: 1000 (right only for the decimated 225 mm
#: `assemblies/`, which §18.3 forbids as a boolean host, and 1.5× too strict here)
#: and `meta["scale"] * 1000` ≈ 76 (metres per *original Hy3D source unit* — an
#: unrelated quantity that inflates `r_min` 19.7×). Both were run on 5 assemblies;
#: measured 24 / 20 / 4 successes out of the same 38 edges, and only this rule
#: agrees with the gold rows edge for edge: `runs/m0-connector/scale-probe.md`.
SCALE_MM = 225.0
#: Print floor, stated at the 225 mm scale (`connector.R_MIN_MM`).
R_MIN_MM = 0.25
#: `connector.py:83-85` — depth carries no threshold of its own; the wall gate
#: `r <= t_b / 2.25` is the only constraint. Passing `None` would re-introduce
#: `1.8 * r_min` and change the feasible set, so pass it explicitly.
DEPTH_MIN_MM = 0.0
SECTION = "D"

#: `edges_v2.json` 坐标口径：150 mm 基准帧毫米 -> 米。与 `unit_mm` 无关。
COORD_MM_PER_UNIT = 1000.0

#: Where the `t_b` this adapter divides came from, carried in `evidence_ref` so a
#: reader never has to guess which of the two wall semantics a row was measured
#: under (`adapter-contract.md` §E7): `edges_v2.json` supersedes
#: `edges_final.json`, whose wall definition spanned cavities.
T_B_SOURCE = "edges_v2/contact_graph._thickness"

#: The frozen numbers (`pah-dev-doc-v1.md` §8.1) this verifier must reproduce.
BASELINE = {"feasible_edges": 705, "success": 388, "rate": 0.550}

UnitMmRule = Literal["scale", "1000", "meta_scale"]

# --------------------------------------------------------------------------- #
# reason -> FailureCertificate.code  (§3.4 vocabulary, contract §B)
# --------------------------------------------------------------------------- #
#: Top-level `reason` values `connector.build` can return, renamed where §3.4
#: already owns the word. `degenerate_geometry` is refined by its `detail` prefix
#: below — leaving it collapsed would file 149 anchor-gap cases as geometry bugs
#: (`connector-boolean-report.md` §16).
_REASON_CODE = {
    "unreatable_too_small": "unreatable_too_small",
    "host_geometry_unreliable": "host_geometry_unreliable",
    "bool_failed": "bool_failed",
    "open_after_bool": "open_after_bool",
    "peg_detached": "peg_detached",
    "socket_breaks_wall": "socket_breaks_wall",
    "degenerate_geometry": "degenerate_other",
    # Emitted by `_call_build` alone, never by the source: the retries ran out
    # while the machine, not the design, was the problem.
    "verifier_oom": "verifier_oom",
}

#: `degenerate_geometry`'s four `detail` prefixes (`connector.py:101-107`).
_DEGEN_CODES = (
    "degenerate_tangent",
    "degenerate_thin_wall",
    "degenerate_at_patch_edge",
    "degenerate_other",
)

#: Causes the source buries in `detail` that §3.4 has a first-class word for.
#: `gap_too_wide_for_wall` is 166/705 of the gold denominator — the single largest
#: cause, and NOT a geometry bug.
_PROMOTED = {
    "gap_too_wide_for_wall": "gap_too_wide",
    "no_material_hit": "no_material_hit",
    "no_manifold3d": "no_manifold3d",
}

_FAMILY = {
    "unreatable_too_small": "assembly",
    "gap_too_wide": "assembly",
    "no_material_hit": "assembly",
    "degenerate_thin_wall": "geometry",
    "degenerate_tangent": "geometry",
    "degenerate_at_patch_edge": "geometry",
    "degenerate_other": "geometry",
    "bool_failed": "geometry",
    "no_manifold3d": "geometry",
    "host_geometry_unreliable": "geometry",
    "open_after_bool": "manufacturing",
    "peg_detached": "manufacturing",
    "socket_breaks_wall": "manufacturing",
    # The escape hatch (§4.8): an unclassified reason is registered so it can be
    # emitted, never so it can be trusted — `confidence` is 0.0 and the raw text
    # rides in `hypotheses`.
    "unknown_reason": "geometry",
    # Not a design fault: the verifier never got an answer because an allocation
    # failed. §3.4 fixes four families and none of them is "environment", so this
    # takes the `unknown_reason` slot — what keeps it out of `bool_failed` is the
    # *code*, which is what every report counts. Widening the family vocabulary
    # is a `pah.core` change, deliberately not made here.
    "verifier_oom": "geometry",
}

#: Everything `check` may emit. `unknown_reason` is in the closed set on purpose:
#: dropping it would silently delete evidence, which is the one thing a verifier
#: must never do.
CODES: tuple[str, ...] = tuple(_FAMILY)


# --------------------------------------------------------------------------- #
# Source provenance
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=1)
def observed_by() -> str:
    """`connector_feasible@<git describe | mtime>` — which build was measured."""
    source = RL_CODE / "assembly_rl" / "connector.py"
    try:
        done = subprocess.run(
            ["git", "-C", str(RL_CODE), "describe", "--always", "--dirty"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if done.returncode == 0 and done.stdout.strip():
            return f"connector_feasible@{done.stdout.strip()}"
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        when = dt.datetime.fromtimestamp(source.stat().st_mtime, dt.timezone.utc)
    except OSError:
        return "connector_feasible@unknown"
    return f"connector_feasible@{when.strftime('%Y%m%dT%H%M%SZ')}"


@functools.lru_cache(maxsize=1)
def _connector() -> Any:
    """Imports `assembly_rl.connector` without touching a byte of it."""
    if str(RL_CODE) not in sys.path:
        sys.path.insert(0, str(RL_CODE))
    from assembly_rl import connector  # noqa: PLC0415  (import is the side effect)

    return connector


# --------------------------------------------------------------------------- #
# case.json construction
# --------------------------------------------------------------------------- #
def _in_split(edge: dict, split: str) -> bool:
    """Which edges a probe split wants.

    `feasible` is the gold denominator: `feasible ∧ ¬thickness_failed` at
    `measurement_lod == "fine"` — 705 of 953. `coarse` edges belong to
    `asm_000099`, which has no undecimated geometry; the source excludes them and
    so does the denominator (§19.1).
    """
    if split == "all":
        return True
    if split == "feasible":
        return bool(
            edge.get("feasible")
            and not edge.get("thickness_failed")
            and edge.get("measurement_lod") == "fine"
        )
    raise ValueError(f"unknown split {split!r}; use 'feasible' or 'coarse' or 'all'")


def _joint(edge: dict, meta: dict) -> dict:
    """One `edges_v2.json` record as a `case.json` joint.

    The `*_mm` fields are copied verbatim, provenance and all: dividing by
    `unit_mm` happens once, in `to_edge_dict`, so there is exactly one place
    where a scale can be got wrong.
    """
    return {
        "id": f"j{edge['id']}",
        "edge_id": int(edge["id"]),
        "eid": int(edge["eid"]),
        "a": f"p{edge['a']}",
        "b": f"p{edge['b']}",
        "a_index": int(edge["a"]),
        "b_index": int(edge["b"]),
        "center_mm": list(edge["center_mm"]),
        "anchor_mm": list(edge["anchor_mm"]),
        "n_ab": list(edge["n_ab"]),
        "half_w_mm": float(edge["half_w_mm"]),
        "half_h_mm": float(edge["half_h_mm"]),
        "r_inplane_max_mm": float(edge["r_inplane_max_mm"]),
        "depth_max_mm": float(edge["depth_max_mm"]),
        "t_a_mm": float(edge["t_a_mm"]),
        "t_b_mm": float(edge["t_b_mm"]),
        "limiter": edge.get("limiter"),
        "wall_irrelevant": bool(edge.get("wall_irrelevant")),
        "anchor_gap_mm": float(edge.get("anchor_gap_mm") or 0.0),
        "measurement_lod": edge.get("measurement_lod"),
        "interface_type": edge.get("interface_type"),
        "edge_quality": edge.get("edge_quality"),
        "feasible": bool(edge.get("feasible")),
        "thickness_failed": bool(edge.get("thickness_failed")),
        "n_parts": int(meta.get("n_parts") or 0),
    }


def _part_record(part: dict, case_id: str) -> dict:
    """One `meta.json` part as a `case.json` part, both geometry paths spelled out."""
    name = part["file"]
    return {
        "id": f"p{part['id']}",
        "file": name,
        "path": str(GEOMETRY_ROOT / case_id / "parts" / name),
        "pre_decimate": str(GEOMETRY_ROOT / case_id / "parts" / name),
        "frozen": True,
        # kept so `host_quality` can be handed through unchanged (§B: the field
        # is read but never written by this pipeline, so it never fires).
        "meta": part,
    }


def load_cases(
    path: str | Path = EDGE_FILE,
    *,
    split: str = "feasible",
    limit: int | None = None,
) -> list[Case]:
    """Reads `edges_v2.json` into one `Case` per assembly.

    `limit` counts **assemblies**, not edges, because geometry loads per assembly
    and 724 undecimated parts do not fit in memory at once
    (`tests/asm_final.py:22`).
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    grouped: dict[str, list[dict]] = {}
    for edge in data["edges"]:
        if _in_split(edge, split):
            grouped.setdefault(edge["assembly_id"], []).append(edge)

    cases: list[Case] = []
    for case_id in sorted(grouped)[: None if limit is None else int(limit)]:
        meta = json.loads(
            (GEOMETRY_ROOT / case_id / "meta.json").read_text(encoding="utf-8")
        )
        cases.append(
            Case(
                case_id=case_id,
                parts=tuple(_part_record(p, case_id) for p in meta["parts"]),
                joints=tuple(_joint(e, meta) for e in grouped[case_id]),
                constraints=_constraints(case_id),
                scale_mm=float(data.get("scale_mm") or SCALE_MM),
                scale_k=float(data.get("k") or 0.7),
                baseline=dict(BASELINE),
            )
        )
    return cases


def _constraints(case_id: str) -> dict:
    """`constraints.json` plus the two thresholds this pipeline actually uses."""
    path = GEOMETRY_ROOT / case_id / "constraints.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raw = {}
    return {**raw, "r_min_mm": R_MIN_MM, "depth_min_mm": DEPTH_MIN_MM, "process": "SLA"}


def to_edge_dict(joint: dict, unit_mm: float) -> dict:
    """`case.json` joint -> the `edge` dict `connector.build` expects, in data units.

    Coordinates divide by the file's own coordinate rule (150 mm base frame,
    millimetres -> metres). Derived lengths divide by `unit_mm`, the same number
    `connector` receives as `unit_mm`, so `r_min_mm / unit_mm` lands in the same
    units as `r_inplane_max` and the gate stays scale-invariant (§18.4).
    """
    return {
        "center": [x / COORD_MM_PER_UNIT for x in joint["center_mm"]],
        "anchor": [x / COORD_MM_PER_UNIT for x in joint["anchor_mm"]],
        "n_ab": list(joint["n_ab"]),
        "half_w": joint["half_w_mm"] / unit_mm,
        "half_h": joint["half_h_mm"] / unit_mm,
        "r_inplane_max": joint["r_inplane_max_mm"] / unit_mm,
        "depth_max": joint["depth_max_mm"] / unit_mm,
        "t_a": joint["t_a_mm"] / unit_mm,
        "t_b": joint["t_b_mm"] / unit_mm,
        "interface_type": joint.get("interface_type"),
        "edge_quality": joint.get("edge_quality"),
    }


def _theta(joint: dict, edge: dict, unit_mm: float) -> tuple[float, ...]:
    """The `(u, v, nx, ny, nz, r, depth)` the policy asks for.

    Copied from `tests/asm_final.py:209-213`: aspect 2:1 pin, never below the
    print floor. Changing this changes the success count, so it is not a free
    parameter.
    """
    r_min = R_MIN_MM / unit_mm
    r_cap, d_cap = edge["r_inplane_max"], edge["depth_max"]
    r = float(min(r_cap, 0.5 * d_cap)) if d_cap > 0 else float(r_cap)
    r = max(r, r_min)
    depth = float(min(2.0 * r, d_cap)) if d_cap > 0 else 2.0 * r
    n = edge["n_ab"]
    return (0.0, 0.0, n[0], n[1], n[2], r, max(depth, 1e-12))


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def _load_parts(case: Case) -> tuple[tuple[Any, ...], float]:
    """Loads one assembly's undecimated parts and its longest side, in metres."""
    return _load_parts_cached(
        case.case_id,
        tuple(p.get("pre_decimate") or p.get("path") for p in case.parts),
    )


def case_parts(case: Case) -> tuple[tuple[Any, ...], float]:
    """The meshes this case is measured on — the same objects, from the same cache.

    Public because the probe's IVR pass reads exactly these meshes
    (`assemblies_pre_decimate`, metres, `docs/internal/adapter-contract.md` §A):
    loading a second copy to measure interpenetration would double the resident
    geometry for no gain.
    """
    return _load_parts(case)


@functools.lru_cache(maxsize=1)
def _load_parts_cached(
    case_id: str, sources: tuple[str, ...]
) -> tuple[tuple[Any, ...], float]:
    """`process=False` is deliberate: `process=True` merges vertices and breaks
    the non-manifold structure `bnd1 == 0` is measured on.

    Cached at **1**, not 2: the largest assembly is 175 MB on disk and several GB
    in memory, and a worker that holds two of them while three siblings do the
    same is what turns a 20-minute probe into a swap storm. One is all the
    fixture pass needs, since it asks for one case two edges in a row.
    """
    import numpy as np  # noqa: PLC0415
    import trimesh  # noqa: PLC0415  (kept out of module import; it is slow)

    meshes = tuple(trimesh.load(source, process=False) for source in sources)
    lo = np.min([m.bounds[0] for m in meshes], axis=0)
    hi = np.max([m.bounds[1] for m in meshes], axis=0)
    return meshes, float(np.max(hi - lo))


@functools.lru_cache(maxsize=8)
def _meta_scale_mm(case_id: str) -> float:
    """`meta["scale"] * 1000` — metres per original Hy3D source unit."""
    meta = json.loads(
        (GEOMETRY_ROOT / case_id / "meta.json").read_text(encoding="utf-8")
    )
    return float(meta["scale"]) * 1000.0


def _unit_mm(case: Case, longest: float, rule: UnitMmRule) -> float:
    """Millimetres per data unit, under the requested rule."""
    if rule == "scale":
        return float(case.scale_mm) / max(longest, 1e-12)
    if rule == "1000":
        return 1000.0
    if rule == "meta_scale":
        return _meta_scale_mm(case.case_id)
    raise ValueError(f"unknown unit_mm rule {rule!r}")


# --------------------------------------------------------------------------- #
# The call itself — one seam, so the mapping below is testable without geometry
# --------------------------------------------------------------------------- #
#: `MemoryError` retry policy: two retries, 1 s then 3 s. The failure is
#: transient, not geometric — on `asm_000097` the gold run lost five edges to
#: `exception:MemoryError(Unable to allocate 56.0 / 14.0 MiB)` while a re-run with
#: one assembly per worker got all five. Waiting matters because the failed
#: attempt's temporaries are released before the next try: an immediate retry at
#: the same ceiling fails the same way.
OOM_RETRIES = (1.0, 3.0)

#: Per-process counters. `iter_joint_results`' callers read them as deltas around
#: every edge, so `retried_n` / `oom_n` in a report are attributable instead of
#: inferred from a number that silently swallowed the retry.
_OOM = {"retried": 0, "oom": 0}

#: Indirection so a test can drive the retry path without spending 4 s asleep.
_sleep = time.sleep


def oom_stats() -> dict[str, int]:
    """This process's `{"retried": <retry attempts>, "oom": <edges still OOM>}`."""
    return dict(_OOM)


def reset_oom_stats() -> None:
    """Zeroes the counters — `probe`'s worker reads deltas, tests read totals."""
    _OOM["retried"] = 0
    _OOM["oom"] = 0


def is_oom(result: dict) -> bool:
    """Whether `result` says "the machine ran out of memory", not "the design failed".

    Two shapes reach this module. A `MemoryError` raised *through* the call is
    caught by `_build_once` and reported as `adapter_raised`; and the source's own
    catch-all (`connector.py:652-659`) turns it into a string —
    `bool_failed / exception:MemoryError(Unable to allocate ...)`, which is
    literally what the gold rows recorded, and the shape a plain
    `except MemoryError` would miss. Both are the environment, so both are
    retried and neither is ever filed as `bool_failed`.
    """
    if result.get("ok"):
        return False
    return "MemoryError" in str(result.get("detail") or "")


def _call_build(
    *,
    parts: tuple[Any, ...],
    joint: dict,
    edge: dict,
    theta: tuple[float, ...],
    unit_mm: float,
    host_quality: dict,
) -> dict:
    """`connector.build`, retried up to `OOM_RETRIES` times when it runs out of memory.

    A final OOM becomes `reason="verifier_oom"` — a registered code of its own,
    because "the machine could not decide" and "the geometry is not buildable"
    are different findings and a report that merges them teaches the loop the
    wrong lesson (§4.8).
    """
    attempts = len(OOM_RETRIES) + 1
    for attempt in range(1, attempts + 1):
        result = _build_once(
            parts=parts,
            joint=joint,
            edge=edge,
            theta=theta,
            unit_mm=unit_mm,
            host_quality=host_quality,
        )
        if not is_oom(result) or attempt == attempts:
            break
        _OOM["retried"] += 1
        gc.collect()  # hand back what the failed attempt allocated, then wait
        _sleep(OOM_RETRIES[attempt - 1])
    if not is_oom(result):
        return result
    _OOM["oom"] += 1
    return {
        "ok": False,
        "reason": "verifier_oom",
        "detail": f"memory_error_after_{attempts}_attempts({result.get('detail', '')})",
    }


def _build_once(
    *,
    parts: tuple[Any, ...],
    joint: dict,
    edge: dict,
    theta: tuple[float, ...],
    unit_mm: float,
    host_quality: dict,
) -> dict:
    """One `connector.build` call with the parameters the gold run used, nothing more.

    `build` is documented never to raise (`connector.py:29`), so the `except` here
    is exception *translation*, not defensive programming — and it deliberately
    reports an adapter fault rather than `bool_failed`, because blaming the
    geometry for this module's bug is the one thing §4.8 forbids.
    """
    connector = _connector()
    try:
        return connector.build(
            parts[joint["a_index"]],
            parts[joint["b_index"]],
            edge,
            theta,
            clamp=True,
            # §19.1 ran with STL round-trip off; turning it on would add a second,
            # stricter failure mode and stop this number from being the same number.
            verify_stl=False,
            overlap_check=True,
            unit_mm=unit_mm,
            r_min_mm=R_MIN_MM,
            depth_min_mm=DEPTH_MIN_MM,
            host_quality=host_quality,
            section=SECTION,
            # `anchor` is on the record; re-snapping it would move a point that is
            # already on the interface (`connector.py:483-495`).
            snap=None,
        )
    except Exception as exc:  # noqa: BLE001  (translated, never swallowed silently)
        return {
            "ok": False,
            "reason": "adapter_raised",
            "detail": f"{type(exc).__name__}: {exc}",
        }


# --------------------------------------------------------------------------- #
# Return-value translation
# --------------------------------------------------------------------------- #
def _split_detail(detail: str) -> tuple[str, str]:
    """`"degenerate_other(gap_too_wide_for_wall(gap=...))"` -> (head, inside)."""
    text = (detail or "").strip()
    if not text:
        return "", ""
    head, sep, rest = text.partition("(")
    return head, (rest[:-1] if sep and rest.endswith(")") else rest)


def classify(result: dict) -> tuple[str, str]:
    """`connector.build`'s failure dict -> `(code, mechanism)`.

    `mechanism` is whatever the source called the cause, kept so nothing the
    adapter cannot promote is lost. Promotion order matters: a
    `degenerate_other(gap_too_wide_for_wall(...))` must become `gap_too_wide`,
    not stay a geometry failure — that mis-file is the documented 149-case trap.
    """
    reason = str(result.get("reason") or "")
    head, inner = _split_detail(str(result.get("detail") or ""))
    if head in _DEGEN_CODES:
        inner_head, _ = _split_detail(inner)
        if inner_head in _PROMOTED:
            return _PROMOTED[inner_head], inner_head
        # `degen_code` is authoritative when present; the prefix is the fallback
        # for the direct `_fail("degenerate_geometry", ...)` sites.
        code = str(result.get("degen_code") or head)
        return (code if code in _FAMILY else "unknown_reason"), inner_head or head
    if head in _PROMOTED:
        return _PROMOTED[head], head
    code = _REASON_CODE.get(reason)
    if code is None:
        return "unknown_reason", head or reason
    # The tail of a `detail` never changes the family for a reason that already
    # names one; it is only the mechanism label (`fragmented_a`, `union_status`).
    return code, head or reason


def _mm(result: dict, key: str, unit_mm: float) -> float | None:
    """A reported data-unit length as millimetres at the 225 mm scale."""
    value = result.get(key)
    if value is None:
        return None
    try:
        return float(value) * unit_mm
    except (TypeError, ValueError):
        return None


def _magnitude(code: str, result: dict, unit_mm: float) -> dict:
    """The number that made the verdict fire, in mm — `{}` when nothing measured."""
    r_mm = _mm(result, "r", unit_mm)
    if code == "unreatable_too_small":
        return {
            "value": _mm(result, "r_cap", unit_mm),
            "unit": "mm",
            "threshold": R_MIN_MM,
            "direction": "lt",
        }
    if code == "gap_too_wide":
        return {
            "value": _mm(result, "gap", unit_mm),
            "unit": "mm",
            "threshold": None,
            "direction": "gt",
        }
    if code == "degenerate_thin_wall":
        return {
            "value": _mm(result, "t_b", unit_mm),
            "unit": "mm",
            "threshold": None if r_mm is None else 2.25 * r_mm,
            "direction": "lt",
        }
    if code == "socket_breaks_wall":
        return {
            "value": _mm(result, "wall_remaining", unit_mm),
            "unit": "mm",
            "threshold": None if r_mm is None else 0.25 * r_mm,
            "direction": "lt",
        }
    return {}


def certificates_for(
    joint: dict, result: dict, unit_mm: float, *, edge_file: str | Path = EDGE_FILE
) -> list[FailureCertificate]:
    """One `connector.build` result as evidence; `[]` when the edge passed."""
    if result.get("ok"):
        return []
    code, mechanism = classify(result)
    reason = str(result.get("reason") or "")
    detail = str(result.get("detail") or "")
    unknown = code == "unknown_reason"
    return [
        FailureCertificate(
            code=code,
            family=_FAMILY[code],
            locus={
                "parts": [joint["a"], joint["b"]],
                "joint": joint["id"],
                "edge": joint.get("edge_id"),
                # the source's own name for the cause: `gap_too_wide_for_wall`,
                # `ray_parity_odd`, `peg_tip_inside_a`, ...
                "mechanism": mechanism,
                "detail": detail,
            },
            magnitude=_magnitude(code, result, unit_mm),
            observed_by=observed_by(),
            # A registered-but-unclassified reason is carried, not believed.
            confidence=0.0 if unknown else 1.0,
            hypotheses=(
                (
                    {
                        "text": f"{reason}: {detail}",
                        "proposed_by": "verifier",
                        "untested": True,
                        "note": "reason absent from the adapter's table",
                    },
                )
                if unknown
                else ()
            ),
            evidence_ref=(
                f"{Path(edge_file).name}#id={joint.get('edge_id')};t_b={T_B_SOURCE}"
            ),
        )
    ]


# --------------------------------------------------------------------------- #
# The verifier
# --------------------------------------------------------------------------- #
class ConnectorFeasible:
    """Can a printable peg/socket pair be built on each of the case's edges?

    `known_good` / `known_bad` are edges whose verdicts the frozen gold run
    (`tests/_final_rows_Nnew_Tnew.json`) already recorded, so a silent change on
    either side of the adapter shows up as a self-check failure rather than as a
    slowly drifting success rate.
    """

    spec = VerifierSpec(
        id="connector_feasible",
        codes=CODES,
        cost_class="expensive",
        fixtures={
            "known_good": [
                # 2 parts / 9.5 MB; gold: ok=True at r_eff 0.36 mm.
                {"case_id": "asm_000028", "ab": [0, 1], "label": "asm_000028:0-1"},
            ],
            "known_bad": [
                # gold: unreatable_too_small, r_max 0.1789 mm < 0.25 mm floor.
                {"case_id": "asm_000043", "ab": [0, 1], "label": "asm_000043:0-1"},
                # gold: degenerate_geometry(gap_too_wide_for_wall) -> gap_too_wide.
                {"case_id": "asm_000043", "ab": [0, 5], "label": "asm_000043:0-5"},
            ],
        },
    )

    def __init__(
        self,
        *,
        unit_mm_rule: UnitMmRule = "scale",
        edge_file: str | Path = EDGE_FILE,
    ) -> None:
        self.unit_mm_rule = unit_mm_rule
        self.edge_file = Path(edge_file)

    # -- fixtures ----------------------------------------------------------

    def case_of(self, item: dict) -> Case:
        """The single-edge case a fixture names."""
        wanted = tuple(int(i) for i in item["ab"])
        for case in load_cases(self.edge_file, split="all"):
            if case.case_id != item["case_id"]:
                continue
            joints = tuple(
                j for j in case.joints if (j["a_index"], j["b_index"]) == wanted
            )
            if not joints:
                raise KeyError(f"{item['case_id']} has no edge {wanted[0]}-{wanted[1]}")
            return Case(
                case_id=case.case_id,
                parts=case.parts,
                joints=joints,
                constraints=case.constraints,
                scale_mm=case.scale_mm,
                scale_k=case.scale_k,
                baseline=case.baseline,
            )
        raise KeyError(f"no such case {item['case_id']!r} in {self.edge_file}")

    # -- the check ---------------------------------------------------------

    def check(
        self, case: Case, assembly: Assembly | None = None
    ) -> list[FailureCertificate]:
        """Every edge of `case` that `connector.build` could not materialise."""
        return [
            certificate
            for joint, result, unit_mm in iter_joint_results(
                case, unit_mm_rule=self.unit_mm_rule
            )
            for certificate in certificates_for(
                joint, result, unit_mm, edge_file=self.edge_file
            )
        ]


def _quality(case: Case, part_id: str) -> dict | None:
    """The `meta.json` part record `connector.host_unreliable` reads."""
    for part in case.parts:
        if part["id"] == part_id:
            return part.get("meta")
    return None


def iter_joint_results(
    case: Case, *, unit_mm_rule: UnitMmRule = "scale"
) -> Iterator[tuple[dict, dict, float]]:
    """`(joint, build_result, unit_mm)` per edge — the probe's raw material.

    Same call as `check`, but the un-translated result is handed back so the
    probe can compare against the gold rows reason by reason instead of only
    counting certificates.
    """
    parts, longest = _load_parts(case)
    unit_mm = _unit_mm(case, longest, unit_mm_rule)
    for joint in case.joints:
        edge = to_edge_dict(joint, unit_mm)
        result = _call_build(
            parts=parts,
            joint=joint,
            edge=edge,
            theta=_theta(joint, edge, unit_mm),
            unit_mm=unit_mm,
            host_quality={
                "a": _quality(case, joint["a"]),
                "b": _quality(case, joint["b"]),
            },
        )
        yield joint, result, unit_mm
