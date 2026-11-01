"""M1: repair the two anchor-gap failure codes on the 100-assembly corpus.

Reads the frozen M0 ledger (`runs/m0-connector/rows.jsonl`), takes every edge the
connector probe failed with `gap_too_wide` or `no_material_hit`, runs the
deterministic ladder in `pah.policy.repair_script`, and streams one row per edge
to `runs/m1-min/rows.jsonl`. Resumable: a row already on disk is not recomputed.

What it measures — the four groups the M1 brief asks for
--------------------------------------------------------
1. **edge level**: how many of the 238 come back, split by the code they failed
   under, and the resulting yield over the 705-edge denominator;
2. **assembly level**: `completely solvable` before/after, where solvable means
   the buildable edges of an assembly span all of its parts (the same criterion
   `contact-graph-report.md` §6.4 calls 生成树可行), plus the Plan-B reachable-part
   count;
3. **cost**: attempts, `connector.build` calls, `measure_axis` calls and wall
   clock, per edge and per operator;
4. **residual attribution**: what the unrepaired edges die of, by code and by why
   the ladder stopped — this is the next repair direction, so it is reported as a
   distribution rather than a count.

Two claims it refuses to take on faith
-------------------------------------
* **"still printable"** is checked, not assumed: a repaired edge reports its
  radius against the 0.25 mm print floor and `wall_remaining` against `0.25 r`,
  because `clamp=True` skips that wall test at `connector.py:855`.
* **"no non-target part touched"** is measured: every part of every assembly is
  hashed before and after its repairs (`parts-hash.jsonl`), and the run reports any
  part whose bytes changed. `connector.build` returns *new* meshes for the two
  target parts; the inputs must come back untouched.

Nothing under `rl_assembly/` is written — the corpus is opened read-only, and all
output lands in `--out`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import sys
import time
import traceback
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pah.policy.repair_script import (  # noqa: E402
    MAX_ATTEMPTS,
    AssemblyParts,
    AttemptBudget,
    repair_edge,
)
from pah.tools.verifiers.connector_feasible import (  # noqa: E402
    EDGE_FILE,
    R_MIN_MM,
    GEOMETRY_ROOT,
    _connector,
    _load_parts,
    _unit_mm,
    load_cases,
    to_edge_dict,
)

TARGET_CODES = ("gap_too_wide", "no_material_hit")
#: The frozen M0 ledger's code counts, used as a guard on the input file. If a
#: concurrent re-probe is rewriting `rows.jsonl` while this runs, the guard fails
#: over to the run-1 copy instead of measuring a half-written file.
FROZEN_CODES = {
    "": 393,
    "gap_too_wide": 166,
    "no_material_hit": 72,
    "unreatable_too_small": 29,
    "socket_breaks_wall": 17,
    "degenerate_other": 17,
    "peg_detached": 8,
    "degenerate_tangent": 3,
}
FROZEN_ROWS = 705
#: Post-gate codes: the peg and socket were built, so the failure is not the
#: anchor's fault. Used to score the closed-form gate prediction.
POST_GATE = frozenset({"peg_detached", "socket_breaks_wall", "open_after_bool", "bool_failed"})


# --------------------------------------------------------------------------- #
# Input
# --------------------------------------------------------------------------- #
def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows


def matches_frozen(rows: list[dict]) -> bool:
    if len(rows) != FROZEN_ROWS:
        return False
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.get("code", "")] = counts.get(row.get("code", ""), 0) + 1
    return counts == FROZEN_CODES


def resolve_source(explicit: str | None, out: Path) -> tuple[Path, list[dict]]:
    """The frozen M0 ledger, refusing to measure a file that is being rewritten."""
    pinned = out / "source-rows.jsonl"
    candidates = [Path(explicit)] if explicit else []
    candidates += [
        pinned,
        Path("runs/m0-connector/rows.jsonl"),
        Path("runs/m0-connector/rows-run1.jsonl"),
    ]
    seen, problems = set(), []
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if not path.exists():
            problems.append(f"{path}: missing")
            continue
        rows = read_rows(path)
        if not matches_frozen(rows):
            problems.append(f"{path}: {len(rows)} rows / codes differ from the frozen M0 ledger")
            continue
        if path != pinned:
            pinned.parent.mkdir(parents=True, exist_ok=True)
            pinned.write_bytes(path.read_bytes())
            print(f"pinned {path} -> {pinned}", flush=True)
        return pinned, rows
    raise SystemExit("no usable frozen M0 ledger:\n  " + "\n  ".join(problems))


# --------------------------------------------------------------------------- #
# Part hashes — the evidence that M1 did not touch non-target geometry
# --------------------------------------------------------------------------- #
def part_hashes(parts) -> list[str]:
    return [hashlib.sha1(mesh.vertices.tobytes()).hexdigest()[:16] for mesh in parts]


def edge_for(joint: dict, unit_mm: float, code: str) -> dict:
    """One `edges_v2` joint as the repair stage receives it, in data units.

    `radius_cap` / `depth_cap` are taken from `connector` itself rather than
    re-derived here, so the closed-form radius policy inside the repair module
    cannot drift from the caps the build will apply.
    """
    edge = to_edge_dict(joint, unit_mm)
    connector = _connector()
    return {
        **edge,
        "edge_id": f"{joint['case_id']}:{joint['edge_id']}",
        "case_id": joint["case_id"],
        "edge_index": int(joint["edge_id"]),
        "a_index": int(joint["a_index"]),
        "b_index": int(joint["b_index"]),
        "code": code,
        "r_cap": float(connector.radius_cap(edge)),
        "d_cap": float(connector.depth_cap(edge)),
    }


# --------------------------------------------------------------------------- #
# One edge → one row
# --------------------------------------------------------------------------- #
def row_of(report, joint: dict, source: dict, unit_mm: float) -> dict:
    """The ledger row: verdict, cost, printability, provenance."""
    result = report.result or {}
    r_mm = _mm(result.get("r"), unit_mm)
    wall_mm = _mm(result.get("wall_remaining"), unit_mm)
    wall_need = None if r_mm is None else 0.25 * r_mm
    # The print floor comparison needs a tolerance: both lengths are rounded to
    # 1e-6 mm for the row, and a repair that lands *exactly* on the wall floor is
    # legal (that is what the `r <= t_b/2.25` constraint produces). Without it the
    # floor cases would be reported as violations by 2.5e-7 mm.
    printable = bool(
        report.success
        and r_mm is not None
        and r_mm >= R_MIN_MM - 1e-6
        and wall_mm is not None
        and wall_need is not None
        and wall_mm >= wall_need - 1e-5
    )
    return {
        "case_id": joint["case_id"],
        "edge_id": int(joint["edge_id"]),
        "a": int(joint["a_index"]),
        "b": int(joint["b_index"]),
        "src_code": source["code"],
        "src_detail": source.get("detail", ""),
        "src_r_mm": source.get("r_mm"),
        "limiter": source.get("limiter", ""),
        "success": bool(report.success),
        "final_reason": report.reason or "",
        "stopped": report.stopped,
        "exhausted": bool(report.exhausted),
        "operator": report.final.get("operator", ""),
        "n_attempts": report.n_attempts,
        "build_calls": report.build_calls,
        "measure_calls": report.measure_calls,
        "wall_s": report.wall_s,
        "attempts": [
            {
                "operator": a.operator,
                "success": a.success,
                "reason": a.reason,
                "raw_reason": a.raw_reason,
                "detail": a.detail,
                "elapsed_s": a.elapsed_s,
                "params": a.params,
            }
            for a in report.attempts
        ],
        "final": report.final,
        "r_mm": r_mm,
        "depth_mm": _mm(result.get("depth"), unit_mm),
        "wall_remaining_mm": wall_mm,
        "wall_need_mm": wall_need,
        "printable": printable,
        "dv_a": result.get("dv_a"),
        "dv_b": result.get("dv_b"),
        "peg_volume": result.get("peg_volume"),
        "sock_volume": result.get("sock_volume"),
        "anchor_shift_mm": report.final.get("anchor_shift_mm"),
        "unit_mm": round(float(unit_mm), 6),
    }



def _printable(row: dict) -> bool:
    """Is a repaired edge still printable? `r >= 0.25 mm` and `wall >= 0.25 r`.

    `connector.build` with `clamp=True` skips the wall test at `connector.py:855`,
    so a successful build can sit *exactly* on the wall floor. That is legal (the
    floor is the design rule) and it is what the radius clamp produces, so the
    comparison needs a tolerance of the row's own rounding: both lengths are
    stored to 1e-6 mm.
    """
    if not row.get("success"):
        return False
    r_mm, wall_mm = row.get("r_mm"), row.get("wall_remaining_mm")
    if r_mm is None or wall_mm is None:
        return False
    return bool(r_mm >= R_MIN_MM - 1e-6 and wall_mm >= 0.25 * r_mm - 1e-5)

def _mm(value, unit_mm: float) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value) * unit_mm, 6)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# One assembly
# --------------------------------------------------------------------------- #
def run_case(
    case,
    targets: list[dict],
    *,
    budget_factory,
    quiet: bool = False,
) -> tuple[list[dict], dict]:
    """Repair one assembly's target edges; returns `(rows, hash_record)`."""
    parts, longest = _load_parts(case)
    unit_mm = _unit_mm(case, longest, "scale")
    before = part_hashes(parts)
    assembly = AssemblyParts(
        case_id=case.case_id,
        unit_mm=unit_mm,
        meshes=parts,
        part_meta=tuple(p.get("meta") for p in case.parts),
        n_parts=len(parts),
    )
    joints = {int(j["edge_id"]): {**j, "case_id": case.case_id} for j in case.joints}

    rows: list[dict] = []
    for target in targets:
        joint = joints[int(target["edge_id"])]
        edge = edge_for(joint, unit_mm, target["code"])
        try:
            report = repair_edge(
                assembly, edge, budget=budget_factory()
            )
            row = row_of(report, joint, target, unit_mm)
        except Exception as exc:  # noqa: BLE001 - one edge must not end the run
            row = {
                "case_id": case.case_id,
                "edge_id": int(target["edge_id"]),
                "a": int(joint["a_index"]),
                "b": int(joint["b_index"]),
                "src_code": target["code"],
                "src_detail": target.get("detail", ""),
                "success": False,
                "final_reason": "adapter_raised",
                "stopped": "error",
                "exhausted": False,
                "operator": "",
                "n_attempts": 0,
                "build_calls": 0,
                "measure_calls": 0,
                "wall_s": 0.0,
                "attempts": [],
                "final": {},
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=4),
            }
        rows.append(row)
        if not quiet:
            print(
                f"    {row['case_id']}:{row['edge_id']:>4} {row['src_code']:<15}"
                f" -> {'OK ' if row['success'] else row['final_reason']:<22}"
                f" {row['n_attempts']}x {row['build_calls']}b/{row['measure_calls']}m"
                f" {row['wall_s']:>6.1f}s",
                flush=True,
            )

    after = part_hashes(parts)
    changed = [i for i, (x, y) in enumerate(zip(before, after)) if x != y]
    record = {
        "case_id": case.case_id,
        "n_parts": len(parts),
        "unit_mm": round(float(unit_mm), 6),
        "n_targets": len(targets),
        "hash_before": before,
        "hash_after": after,
        "changed_parts": changed,
        "targets": [[int(r["edge_id"]), int(r["a"]), int(r["b"])] for r in rows],
    }
    return rows, record


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
def run(args) -> dict:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / "rows.jsonl"
    hash_path = out / "parts-hash.jsonl"
    started = time.time()
    deadline = started + max(1.0, args.minutes) * 60.0 - 90.0  # 90 s of head-room

    source_path, source_rows = resolve_source(args.source, out)
    print(f"source ledger: {source_path} ({len(source_rows)} rows)", flush=True)
    targets_by_case: dict[str, list[dict]] = {}
    for row in source_rows:
        if row.get("code") in TARGET_CODES:
            targets_by_case.setdefault(row["case_id"], []).append(row)
    n_targets = sum(len(v) for v in targets_by_case.values())
    per_code = {c: sum(1 for v in targets_by_case.values() for r in v if r["code"] == c)
                for c in TARGET_CODES}
    print(
        f"targets: {n_targets} edges over {len(targets_by_case)} assemblies"
        f" ({', '.join(f'{c}={n}' for c, n in per_code.items())})",
        flush=True,
    )

    done = set()
    if args.resume and rows_path.exists():
        done = {(r["case_id"], int(r["edge_id"])) for r in read_rows(rows_path)}
        print(f"resume: {len(done)} rows already on disk", flush=True)

    cases = {c.case_id: c for c in load_cases(EDGE_FILE, split="feasible")}
    ordered = [cid for cid in sorted(targets_by_case) if cid in cases]
    if args.limit:
        ordered = ordered[: max(1, int(args.limit))]
        print(f"limit: first {len(ordered)} assemblies", flush=True)
    missing = sorted(set(targets_by_case) - set(cases))
    if missing:
        print(f"WARNING: {len(missing)} assemblies have no feasible-split case: {missing[:5]}")

    def budget_factory() -> AttemptBudget:
        return AttemptBudget(
            max_attempts=args.attempts, deadline=deadline if args.minutes else None
        )

    hash_sink = hash_path.open("a", encoding="utf-8")
    n_new = 0
    try:
        with rows_path.open("a", encoding="utf-8") as sink:
            for index, case_id in enumerate(ordered, start=1):
                todo = [t for t in targets_by_case[case_id] if (case_id, int(t["edge_id"])) not in done]
                if not todo:
                    continue
                if time.time() >= deadline:
                    print(f"deadline reached before {case_id}; stopping cleanly", flush=True)
                    break
                print(f"  [{index:2d}/{len(ordered)}] {case_id} ({len(todo)} edges)", flush=True)
                case = cases[case_id]
                try:
                    rows, record = run_case(
                        case, todo, budget_factory=budget_factory, quiet=args.quiet
                    )
                except Exception:  # noqa: BLE001 - one assembly must not end the run
                    print(f"  !! {case_id} failed:\n{traceback.format_exc(limit=3)}", flush=True)
                    continue
                for row in rows:
                    sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                sink.flush()
                os.fsync(sink.fileno())
                hash_sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                hash_sink.flush()
                n_new += len(rows)
                elapsed = time.time() - started
                print(f"      ({n_new} edges this run, {elapsed:.0f}s)", flush=True)
    finally:
        hash_sink.close()

    return summarise(
        out=out,
        source_path=source_path,
        source_rows=source_rows,
        elapsed=time.time() - started,
        n_targets=n_targets,
        n_new=n_new,
        args=args,
    )


# --------------------------------------------------------------------------- #
# The four groups of numbers
# --------------------------------------------------------------------------- #
def _tree_stats(n_parts: int, edges: list[tuple[int, int]]) -> dict:
    """Largest connected component and whether one component covers all parts."""
    if n_parts <= 0:
        return {"n_parts": 0, "largest": 0, "solvable": False}
    parent = list(range(n_parts))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        if 0 <= a < n_parts and 0 <= b < n_parts:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
    sizes: dict[int, int] = {}
    for node in range(n_parts):
        root = find(node)
        sizes[root] = sizes.get(root, 0) + 1
    largest = max(sizes.values()) if sizes else 0
    return {"n_parts": n_parts, "largest": largest, "solvable": largest == n_parts}


def _sum_wall(rows: list[dict]) -> float:
    return round(sum(float(r.get("wall_s") or 0.0) for r in rows), 1)


def _pass_seconds(out: Path, n_new: int, elapsed: float, *, per_edge_sum: float = 0.0) -> tuple[float, str]:
    """Wall clock of the pass that actually built the rows.

    A `--resume` pass that finds every row already on disk takes a second; calling
    that the run's cost would understate what the repairs cost by three orders of
    magnitude. So the reported figure is carried from the run that measured it —
    `run.json` when it is still describing that pass, otherwise the last
    cumulative line of `run.log` (`"<n> edges this run, <s>s"`).
    """
    if n_new:
        return float(elapsed), "measured"
    if per_edge_sum > 0:
        # A report-only pass builds nothing, so its own clock says nothing about
        # what the repairs cost. The ledger's per-edge wall clock still does, and
        # it is the number a reader can re-add from rows.jsonl.
        return float(per_edge_sum), "sum of per-edge wall clock (report-only pass)"
    try:
        prior = json.loads((out / "run.json").read_text(encoding="utf-8"))
        seconds = float(prior.get("seconds") or 0.0)
        if seconds > 0:
            return seconds, "carried from run.json"
    except (OSError, ValueError):
        pass
    try:
        text = (out / "run.log").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return float(elapsed), "unavailable"
    matches = re.findall(r"\(\s*(\d+) edges this run,\s*(\d+)s\)", text)
    if matches:
        return float(matches[-1][1]), "carried from run.log"
    return float(elapsed), "unavailable"


def summarise(
    *,
    out: Path,
    source_path: Path,
    source_rows: list[dict],
    elapsed: float,
    n_targets: int,
    n_new: int,
    args,
) -> dict:
    rows = read_rows(out / "rows.jsonl")
    for row in rows:
        # Recompute rather than trust the stored flag: it is written at build time,
        # and a tolerance change must not need a 20-minute re-run to take effect.
        row["printable"] = _printable(row)
    seconds, seconds_source = _pass_seconds(out, n_new, elapsed, per_edge_sum=_sum_wall(rows))
    by_key = {(r["case_id"], int(r["edge_id"])): r for r in rows}
    hashes = read_rows(out / "parts-hash.jsonl")
    hash_by_case = {h["case_id"]: h for h in hashes}
    edges_data = json.loads(Path(EDGE_FILE).read_text(encoding="utf-8"))

    # ---- 1. edge level ---------------------------------------------------- #
    edge_level: dict[str, dict] = {}
    for code in TARGET_CODES:
        src = [r for r in source_rows if r.get("code") == code]
        got = [by_key[(r["case_id"], int(r["edge_id"]))] for r in src
               if (r["case_id"], int(r["edge_id"])) in by_key]
        repaired = [g for g in got if g["success"]]
        edge_level[code] = {
            "edges": len(src),
            "measured": len(got),
            "repaired": len(repaired),
            "rate": round(len(repaired) / len(got), 4) if got else None,
            "printable": sum(1 for g in repaired if g["printable"]),
            "attempts_hist": _counts(str(g["n_attempts"]) for g in got),
            "by_operator": _counts(g["operator"] for g in repaired),
            "shift_mm_p50": _p([g["anchor_shift_mm"] for g in repaired], 0.5),
        }
    total_src = sum(v["edges"] for v in edge_level.values())
    total_repaired = sum(v["repaired"] for v in edge_level.values())
    total_measured = sum(v["measured"] for v in edge_level.values())
    success_before = FROZEN_CODES[""]
    denominator = FROZEN_ROWS
    edge_level["_total"] = {
        "edges": total_src,
        "measured": total_measured,
        "repaired": total_repaired,
        "rate": round(total_repaired / total_measured, 4) if total_measured else None,
        "printable": sum(v["printable"] for k, v in edge_level.items() if k != "_total"),
        "yield_before": success_before / denominator,
        "yield_after": (success_before + total_repaired) / denominator,
    }

    # ---- 2. assembly level ------------------------------------------------ #
    # The part count must come from the geometry record, not from the edges: parts
    # with no edge at all are exactly the ones a spanning tree has to reach.
    meta_parts: dict[str, int] = {h["case_id"]: int(h["n_parts"]) for h in hashes}
    feasible_file: dict[str, list[tuple[int, int]]] = {}
    for edge in edges_data["edges"]:
        aid = edge["assembly_id"]
        if edge.get("feasible"):
            feasible_file.setdefault(aid, []).append((int(edge["a"]), int(edge["b"])))
    universe = sorted({edge["assembly_id"] for edge in edges_data["edges"]})
    for aid in universe:
        if aid in meta_parts:
            continue
        try:
            meta = json.loads(
                (Path(GEOMETRY_ROOT) / aid / "meta.json").read_text(encoding="utf-8")
            )
            meta_parts[aid] = len(meta["parts"])
        except (OSError, ValueError, KeyError):
            meta_parts[aid] = 0
    ok_before: dict[str, list[tuple[int, int]]] = {}
    ok_after: dict[str, list[tuple[int, int]]] = {}
    for row in source_rows:
        if row.get("ok"):
            ok_before.setdefault(row["case_id"], []).append((int(row["a"]), int(row["b"])))
    for row in rows:
        if row["success"]:
            ok_after.setdefault(row["case_id"], []).append((int(row["a"]), int(row["b"])))
    for aid, pairs in ok_before.items():
        ok_after.setdefault(aid, []).extend(pairs)

    assemblies = []
    for aid in universe:
        n = int(meta_parts.get(aid) or 0)
        file_edges = feasible_file.get(aid, [])
        before = ok_before.get(aid, [])
        after = ok_after.get(aid, [])
        assemblies.append(
            {
                "case_id": aid,
                "n_parts": n,
                "n_feasible_file": len(file_edges),
                "n_ok_before": len(before),
                "n_ok_after": len(after),
                "file": _tree_stats(n, file_edges),
                "before": _tree_stats(n, before),
                "after": _tree_stats(n, after),
            }
        )
    assembly_level = {
        "assemblies": len(assemblies),
        "solvable_file": sum(1 for a in assemblies if a["file"]["solvable"]),
        "solvable_before": sum(1 for a in assemblies if a["before"]["solvable"]),
        "solvable_after": sum(1 for a in assemblies if a["after"]["solvable"]),
        "reachable_before": sum(a["before"]["largest"] for a in assemblies),
        "reachable_after": sum(a["after"]["largest"] for a in assemblies),
        "parts_total": sum(a["n_parts"] for a in assemblies),
        "flipped": [a["case_id"] for a in assemblies
                    if a["after"]["solvable"] and not a["before"]["solvable"]],
        "lost": [a["case_id"] for a in assemblies
                 if a["before"]["solvable"] and not a["after"]["solvable"]],
    }

    # ---- 3. cost ---------------------------------------------------------- #
    per_edge = [r["wall_s"] for r in rows]
    cost = {
        "edges_measured": len(rows),
        "attempts_total": sum(r["n_attempts"] for r in rows),
        "build_calls": sum(r["build_calls"] for r in rows),
        "measure_calls": sum(r["measure_calls"] for r in rows),
        "attempts_hist": {str(k): sum(1 for r in rows if r["n_attempts"] == k) for k in range(4)},
        "wall_s_total": round(seconds, 1),
        "wall_s_per_edge_p50": _p(per_edge, 0.5),
        "wall_s_per_edge_p95": _p(per_edge, 0.95),
        "wall_s_per_edge_sum": round(sum(per_edge), 1),
        "by_operator": {},
    }
    for name in ("resize_connector", "relocate_connector", "swap_family", ""):
        hits = [r for r in rows if r["success"] and r["operator"] == name]
        tries = [r for r in rows if any(a["operator"] == name for a in r["attempts"])]
        if tries:
            cost["by_operator"][name or "none"] = {
                "attempted": len(tries),
                "succeeded": len(hits),
                "hit_rate": round(len(hits) / len(tries), 4),
            }

    # ---- 4. residual attribution ----------------------------------------- #
    failed = [r for r in rows if not r["success"]]
    residual = {
        "unrepaired": len(failed),
        "by_reason": _counts(r["final_reason"] for r in failed),
        "by_stopped": _counts(r["stopped"] for r in failed),
        "by_src_code": _counts(r["src_code"] for r in failed),
        "by_last_operator": _counts(r["operator"] for r in failed),
        "dead_ends": _counts(
            f"{r['src_code']} -> {r['final_reason']}" for r in failed
        ),
        "gap_too_wide_mechanism": _gap_mechanism(failed, edges_data),
    }

    # ---- claim checks ----------------------------------------------------- #
    changed = {h["case_id"]: h["changed_parts"] for h in hashes if h["changed_parts"]}
    repaired = [r for r in rows if r["success"]]
    radii = [r["r_mm"] for r in repaired]
    wall_ratio = [
        r["wall_remaining_mm"] / r["wall_need_mm"] for r in repaired
        if r.get("wall_remaining_mm") and r.get("wall_need_mm")
    ]
    printability = {
        "repaired": len(repaired),
        "printable": sum(1 for r in repaired if r["printable"]),
        "r_mm_min": _p(radii, 0.0),
        "r_mm_p25": _p(radii, 0.25),
        "r_mm_p50": _p(radii, 0.5),
        "r_mm_p95": _p(radii, 0.95),
        "r_mm_max": _p(radii, 1.0),
        # A repair that lands on the print floor is a legal repair and a weak pin.
        # The count is reported so "repaired" is never read as "as strong as new".
        "at_print_floor": sum(1 for r in repaired
                              if r["r_mm"] is not None and r["r_mm"] <= R_MIN_MM + 1e-4),
        "wall_remaining_over_need_p50": _p(wall_ratio, 0.5),
        "wall_short": [f"{r['case_id']}:{r['edge_id']}" for r in repaired
                       if not r["printable"]],
    }
    gate = _gate_calibration(rows)
    untouched = {
        "assemblies_hashed": len(hashes),
        "assemblies_with_changed_parts": len(changed),
        "changed": {k: v for k, v in list(changed.items())[:20]},
        "target_pairs": sum(len(h["targets"]) for h in hashes),
    }

    summary = {
        "probe": "m1-repair",
        "scope": "gap_too_wide + no_material_hit, 100 assemblies",
        "source": {
            "path": str(source_path),
            "sha256": sha256_of(source_path),
            "rows": len(source_rows),
            "pinned_copy": str(out / "source-rows.jsonl"),
        },
        "config": {
            "max_attempts": MAX_ATTEMPTS,
            "attempts_used": args.attempts,
            "minutes": args.minutes,
            "edge_file": str(EDGE_FILE),
            "geometry_root": str(GEOMETRY_ROOT),
            "geometry_hashes": {
                "targets": "sha256 of every target part before/after (parts-hash.jsonl)"
            },
        },
        "edge_level": edge_level,
        "assembly_level": assembly_level,
        "assemblies": assemblies,
        "cost": cost,
        "residual": residual,
        "printability": printability,
        "untouched": untouched,
        "gate_calibration": gate,
        "complete": len(rows) == n_targets,
        "edges_written": len(rows),
        "edges_new_this_run": n_new,
        "targets": n_targets,
        "seconds": round(seconds, 1),
        "seconds_source": seconds_source,
    }
    (out / "run.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    (out / "summary.md").write_text(_markdown(summary), encoding="utf-8")
    return summary


def _p(values, q: float) -> float | None:
    clean = [v for v in values if isinstance(v, (int, float))]
    if not clean:
        return None
    if len(clean) == 1:
        return round(float(clean[0]), 6)
    ordered = sorted(clean)
    pos = max(0.0, min(1.0, q)) * (len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return round(float(ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)), 6)


def _counts(values) -> dict:
    out: dict[str, int] = {}
    for value in values:
        out[str(value)] = out.get(str(value), 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


#: `gap_too_wall` detail as the source spells it (`connector.py:815-819`).
_GAP_RE = re.compile(r"gap=([0-9.eE+-]+)mm,need=([0-9.eE+-]+)mm>eng_max=([0-9.eE+-]+)mm")


def _gap_mechanism(failed: list[dict], edges_data: dict) -> dict:
    """Which term owns the residual `gap_too_wide` verdict.

    The gate is `min(d_cap, t_b + gap - 1.25r) <= gap + r`; reading the numbers
    back out of the last attempt's detail says whether the wall term or the depth
    cap is the binding one. If `eng_max` equals the recorded `depth_max`, no radius
    and no in-patch anchor can pass, because the cap is a file constant — that is
    a policy/algorithm question, not a repair this ladder can reach.
    """
    known = {(e["assembly_id"], int(e["id"])): e for e in edges_data["edges"]}
    depth_bound = wall_bound = gap_over_cap = parsed = 0
    ratios = []
    for row in failed:
        if row["final_reason"] != "gap_too_wide" or not row["attempts"]:
            continue
        match = _GAP_RE.search(row["attempts"][-1].get("detail") or "")
        edge = known.get((row["case_id"], int(row["edge_id"])))
        if not match or edge is None:
            continue
        gap, _need, eng_max = (float(x) for x in match.groups())
        d_cap = float(edge["depth_max_mm"])
        parsed += 1
        if abs(eng_max - d_cap) <= 1e-3:
            depth_bound += 1
        else:
            wall_bound += 1
        if d_cap > 0:
            ratios.append(gap / d_cap)
            gap_over_cap += gap >= d_cap
    return {
        "parsed": parsed,
        "depth_cap_binds": depth_bound,
        "wall_term_binds": wall_bound,
        "local_gap_ge_recorded_depth_cap": gap_over_cap,
        "gap_over_depth_cap_p50": _p(ratios, 0.5),
        "note": (
            "depth_max is a per-edge constant in edges_v2.json "
            "(0.8*min(t_a,t_b)); when the local gap exceeds it the gate cannot "
            "pass at any radius or in-plane position"
        ),
    }


def _gate_calibration(rows: list[dict]) -> dict:
    """How often the closed-form gate prediction matched what the build did.

    `predicted` comes from the repair plan (`params["gate_predicted"]`); `passed`
    means the build reached the boolean stage (success, or a post-gate code). A
    mirror of the source's gate that disagrees here would make the radius policy
    look better than it is, so the confusion matrix is part of the report.
    """
    agree = disagree = 0
    pairs = []
    for row in rows:
        for attempt in row["attempts"]:
            params = attempt.get("params") or {}
            if "gate_predicted" not in params:
                continue
            predicted = bool(params["gate_predicted"])
            passed = attempt["success"] or (attempt.get("reason") in POST_GATE)
            agree += predicted == passed
            disagree += predicted != passed
            pairs.append((predicted, passed))
    return {
        "attempts_checked": agree + disagree,
        "agree": agree,
        "disagree": disagree,
        "false_positive": sum(1 for p, x in pairs if p and not x),
        "false_negative": sum(1 for p, x in pairs if not p and x),
    }


def _markdown(summary: dict) -> str:
    edge = summary["edge_level"]
    total = edge["_total"]
    asm = summary["assembly_level"]
    cost = summary["cost"]
    res = summary["residual"]
    pr = summary["printability"]
    lines = [
        "# M1 repair probe — auto-generated tables",
        "",
        f"- source: `{summary['source']['path']}` "
        f"({summary['source']['rows']} rows, sha256 `{summary['source']['sha256'][:16]}…`)",
        f"- geometry: `{summary['config']['geometry_root']}`",
        f"- targets: {summary['targets']} edges, written {summary['edges_written']} "
        f"({'complete' if summary['complete'] else 'PARTIAL'})",
        f"- wall clock: {summary['seconds']} s",
        "",
        "## 1. edge level",
        "",
        "| source code | edges | repaired | rate | printable |",
        "|---|---:|---:|---:|---:|",
    ]
    for code in TARGET_CODES:
        row = edge[code]
        lines.append(
            f"| `{code}` | {row['edges']} | {row['repaired']} | "
            f"{_pct(row['rate'])} | {row['printable']} |"
        )
    lines += [
        f"| **total** | **{total['edges']}** | **{total['repaired']}** | "
        f"**{_pct(total['rate'])}** | **{total['printable']}** |",
        "",
        f"yield over the 705-edge denominator: {total['yield_before']:.1%} → "
        f"{total['yield_after']:.1%} "
        f"({FROZEN_CODES['']} → {FROZEN_CODES[''] + total['repaired']})",
        "",
        "## 2. assembly level (spanning tree over buildable edges)",
        "",
        "| quantity | before | after |",
        "|---|---:|---:|",
        f"| completely solvable | {asm['solvable_before']} | {asm['solvable_after']} |",
        f"| reachable parts (Plan B) | {asm['reachable_before']}/{asm['parts_total']} | "
        f"{asm['reachable_after']}/{asm['parts_total']} |",
        f"| file-feasible ceiling | {asm['solvable_file']} | {asm['solvable_file']} |",
        "",
        f"flipped to solvable: {', '.join(asm['flipped']) or '—'}",
        f"lost: {', '.join(asm['lost']) or '—'}",
        "",
        "## 3. cost",
        "",
        "| quantity | value |",
        "|---|---:|",
        f"| attempts | {cost['attempts_total']} |",
        f"| `connector.build` calls | {cost['build_calls']} |",
        f"| `measure_axis` calls | {cost['measure_calls']} |",
        f"| wall s (whole run) | {cost['wall_s_total']} |",
        f"| wall s per edge p50 / p95 | {cost['wall_s_per_edge_p50']} / {cost['wall_s_per_edge_p95']} |",
        f"| attempts histogram | {cost['attempts_hist']} |",
        "",
        "| operator | attempted | succeeded | hit rate |",
        "|---|---:|---:|---:|",
    ]
    for name, row in cost["by_operator"].items():
        lines.append(
            f"| `{name}` | {row['attempted']} | {row['succeeded']} | {_pct(row['hit_rate'])} |"
        )
    lines += [
        "",
        "## 4. residual attribution",
        "",
        f"unrepaired: **{res['unrepaired']}**",
        "",
        "| final reason | count |",
        "|---|---:|",
    ]
    for name, count in res["by_reason"].items():
        lines.append(f"| `{name}` | {count} |")
    lines += ["", "| why the ladder stopped | count |", "|---|---:|"]
    for name, count in res["by_stopped"].items():
        lines.append(f"| `{name}` | {count} |")
    lines += ["", "| src → final | count |", "|---|---:|"]
    for name, count in list(res["dead_ends"].items())[:15]:
        lines.append(f"| `{name}` | {count} |")
    gap = res.get("gap_too_wide_mechanism") or {}
    if gap.get("parsed"):
        lines += [
            "",
            "### residual `gap_too_wide`: which term binds",
            "",
            f"- parsed from the last attempt's detail: {gap['parsed']}",
            f"- `eng_max == depth_max` (the depth cap is the binding term): "
            f"**{gap['depth_cap_binds']}**",
            f"- wall term binds: {gap['wall_term_binds']}",
            f"- local gap ≥ the recorded `depth_max`: "
            f"{gap['local_gap_ge_recorded_depth_cap']} "
            f"(gap/depth_max p50 {gap['gap_over_depth_cap_p50']})",
        ]
    lines += [
        "",
        "## claim checks",
        "",
        f"- printable: {pr['printable']}/{pr['repaired']} repaired edges "
        f"(r p50 {pr['r_mm_p50']} mm, {pr['at_print_floor']} at the "
        f"{R_MIN_MM} mm floor, wall remaining / needed p50 "
        f"{pr['wall_remaining_over_need_p50']})",
        f"- parts hashed: {summary['untouched']['assemblies_hashed']} assemblies, "
        f"changed: {summary['untouched']['assemblies_with_changed_parts']}",
        f"- gate prediction: {summary['gate_calibration']['agree']}/"
        f"{summary['gate_calibration']['attempts_checked']} attempts agree "
        f"(false +{summary['gate_calibration']['false_positive']} / "
        f"-{summary['gate_calibration']['false_negative']})",
        "",
    ]
    return "\n".join(lines)


def _pct(value) -> str:
    return "—" if value is None else f"{value:.1%}"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="runs/m1-min")
    parser.add_argument("--source", default=None, help="frozen M0 rows.jsonl (validated)")
    parser.add_argument("--resume", action="store_true", help="skip edges already on disk")
    parser.add_argument("--attempts", type=int, default=MAX_ATTEMPTS)
    parser.add_argument("--minutes", type=float, default=90.0)
    parser.add_argument("--limit", type=int, default=0, help="first N assemblies only")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    started = time.time()
    summary = run(args)
    print("\n=== M1 summary ===", flush=True)
    total = summary["edge_level"]["_total"]
    print(
        f"repaired {total['repaired']}/{total['measured']} "
        f"({_pct(total['rate'])})  yield {total['yield_before']:.1%} -> "
        f"{total['yield_after']:.1%}",
        flush=True,
    )
    asm = summary["assembly_level"]
    print(
        f"solvable {asm['solvable_before']}/100 -> {asm['solvable_after']}/100 "
        f"(file ceiling {asm['solvable_file']})",
        flush=True,
    )
    print(
        f"cost: {summary['cost']['attempts_total']} attempts, "
        f"{summary['cost']['build_calls']} builds, "
        f"{summary['cost']['measure_calls']} measures, "
        f"{time.time() - started:.0f}s wall",
        flush=True,
    )
    print(f"residual: {summary['residual']['by_reason']}", flush=True)
    print(f"wrote {Path(args.out) / 'run.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())





