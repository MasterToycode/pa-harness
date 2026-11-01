"""M1 variant: does the local single wall unlock the depth-capped `gap_too_wide` edges?

The experiment `runs/m1-min/report.md` §4.1 asks for, run **beside** the baseline
and never merged into it:

* baseline arm = the original M1 configuration, unchanged (`repair_edge` with the
  `edges_v2` edge dict). Re-run here so the paired comparison is measured in one
  process, and so "the baseline is reproduced bit for bit" is a check rather than
  a claim.
* variant arm = the same ladder, with one data-layer change: `depth_max` / `d_cap`
  replaced by `0.8 ×` the wall measured at the anchor on the socket side
  (`pah.policy.dcap`), plus a residual-wall guard that vetoes any build which eats
  into that wall.

Nothing is written back to `edges_v2.json` or to `rl_assembly/`; the override is a
copy of the edge dict, in memory, for the length of one edge. Every row carries
`dcap_source` so a variant row can never be read as a golden-protocol row.

Usage
-----
    python scripts/m1_dcap.py --out runs/m1-dcap --resume          # measure
    python scripts/m1_dcap.py --out runs/m1-dcap --compare         # baseline vs M1
    python scripts/m1_dcap.py --out runs/m1-dcap --summarize       # the six groups

The six groups `--summarize` reports are the ones the brief names: repairs gained,
`socket_breaks_wall` before/after, IVR of the repaired edges, the residual-wall
distribution, the other failure codes' transitions, and the McNemar pairing.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
import traceback
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pah.policy.dcap import (  # noqa: E402
    WALL_FRAC,
    WALL_FRAC_FLOOR_MM,
    depth_cap_mm,
    override_depth_cap,
    probe_local_wall,
    residual_wall_need_mm,
    thin_wall_veto,
)
from pah.policy.repair_script import (  # noqa: E402
    MAX_ATTEMPTS,
    AssemblyParts,
    AttemptBudget,
    _connector_build,
    repair_edge,
)
from pah.tools.ivr import ivr_exact  # noqa: E402
from pah.tools.verifiers.connector_feasible import (  # noqa: E402
    EDGE_FILE,
    R_MIN_MM,
    _connector,
    _load_parts,
    _unit_mm,
    load_cases,
    to_edge_dict,
)

TARGET_CODES = ("gap_too_wide", "no_material_hit")
#: The 238 M1 target edges over the 705 feasible ones — the denominator both arms
#: are run on, so the pairing has a fixed population.
M1_ROWS = 238
#: `pah-dev-doc-v1.md` §5.3's rule table ends at three attempts (`MAX_ATTEMPTS`).
#:
#: Three arms, because one number cannot answer the brief's question honestly:
#:
#: * `baseline`  — the M1 configuration, unchanged. The control, and the proof
#:   that the re-run reproduces `runs/m1-min/rows.jsonl`.
#: * `cap_only`  — the data-layer change alone (`d_cap` from the local single
#:   wall). Isolating the lever means isolating it from the wall standard.
#: * `variant`   — `cap_only` plus the brief's residual-wall floor: the plan is
#:   searched under `max(0.25 mm, 20 % of the local wall)` and a build that eats
#:   into it is vetoed. This is the protocol the brief specifies.
ARMS = ("baseline", "cap_only", "variant")
ARMS_BOTH = ("baseline", "variant")


# --------------------------------------------------------------------------- #
# Input
# --------------------------------------------------------------------------- #
def read_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows


def m1_targets(path: Path, limit_assemblies: int | None = None) -> list[dict]:
    """The 238 edges M1 worked on, in the order `runs/m1-min/rows.jsonl` has them."""
    rows = read_rows(path)
    picked = [r for r in rows if r.get("src_code") in TARGET_CODES]
    if len(picked) != M1_ROWS:
        raise SystemExit(
            f"{path}: expected the frozen {M1_ROWS} M1 target rows, found {len(picked)}"
            f" of {len(rows)}"
        )
    if limit_assemblies:
        keep = sorted({r["case_id"] for r in picked})[: int(limit_assemblies)]
        picked = [r for r in picked if r["case_id"] in set(keep)]
    return picked


def resolve_source(explicit: str | None, out: Path) -> Path:
    """Pin the frozen M1 ledger into the run directory, once.

    Both arms are described relative to the same 238 M1 rows, so the run keeps its
    own copy: a later M1 re-run cannot change what this experiment was paired
    against.
    """
    pinned = out / "source-rows.jsonl"
    if pinned.exists():
        return pinned
    source = Path(explicit or "runs/m1-min/rows.jsonl")
    if not source.exists():
        raise SystemExit(f"missing M1 ledger: {source}")
    out.mkdir(parents=True, exist_ok=True)
    pinned.write_bytes(source.read_bytes())
    print(f"pinned {source} -> {pinned}", flush=True)
    return pinned


def ledger(args, out: Path) -> Path:
    pinned = out / "source-rows.jsonl"
    return pinned if pinned.exists() else Path(args.source)


def _src_code(target: dict) -> str:
    """The failure code an M1 row came in with (`src_code` on `runs/m1-min` rows)."""
    return str(target.get("src_code") or target.get("code") or "")


def edge_for(joint: dict, unit_mm: float, code: str) -> dict:
    """`edges_v2` joint -> the edge dict the repair stage receives, in data units.

    Copied field for field from `scripts/m1_repair.edge_for` on purpose: the
    baseline arm has to see exactly the dict M1 saw, so the caps come from
    `connector` itself rather than from a second derivation.
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
# The two arms
# --------------------------------------------------------------------------- #
class BuildTap:
    """`connector.build` through M1's parameters, remembering the last result.

    The tap exists for two reasons: the IVR needs the two output meshes (the
    scalar-only report drops them), and the variant arm has to be able to veto a
    build *after* it ran. It calls `repair_script._connector_build` — the exact
    function the ladder uses by default — so tapping does not change the build.
    """

    def __init__(self, *, wall_mm: float | None, unit_mm: float) -> None:
        self.wall_mm = wall_mm
        self.unit_mm = unit_mm
        self.last: dict | None = None
        self.guard_trips = 0
        self.wall_need_mm: float | None = None

    def __call__(self, assembly, edge, theta, *, section: str) -> dict:
        result = _connector_build(assembly, edge, theta, section=section)
        if self.wall_mm is not None:
            veto = thin_wall_veto(result, unit_mm=self.unit_mm, wall_mm=self.wall_mm)
            if veto is not None:
                self.guard_trips += 1
                self.wall_need_mm = veto.get("wall_need_mm")
                self.last = None  # the meshes are not a design, they are rejected
                return veto
        need = result.get("wall_remaining")
        if need is not None and self.wall_mm is not None:
            build_wall = result.get("t_b")
            ref = self.wall_mm if build_wall is None else min(
                self.wall_mm, float(build_wall) * self.unit_mm
            )
            self.wall_need_mm = residual_wall_need_mm(ref)
        self.last = result
        return result


def ivr_of(result: dict, assembly_parts: tuple, unit_mm: float) -> dict | None:
    """Pair-local IVR of one repaired edge: `vol(A'∩B')/(vol A'+vol B')`.

    Also returns the numerator over the assembly's total part volume, which is the
    edge's contribution to the §4.8 assembly IVR (`Σ vol(V_i∩V_j) / Σ_i vol(V_i)`)
    if that pair is in `E_check`. Both are reported; neither is called "the IVR".
    """
    a_out, b_out = result.get("a_out"), result.get("b_out")
    if a_out is None or b_out is None:
        return None
    try:
        reading = ivr_exact([("a", a_out), ("b", b_out)], pairs=[("a", "b")])
    except Exception as exc:  # noqa: BLE001 - an IVR failure must not end the pass
        return {"error": f"{type(exc).__name__}: {exc}"}
    detail = reading.detail
    numerator = float(detail.get("numerator") or 0.0)
    bbox = None
    for record in detail.get("pairs") or []:
        if record.get("bbox_volume") is not None:
            bbox = float(record["bbox_volume"])
    total = 0.0
    for mesh in assembly_parts:
        try:
            total += abs(float(mesh.volume))
        except Exception:  # noqa: BLE001
            total = 0.0
            break
    contrib = numerator / total if total > 0.0 else None
    contrib_hi = (numerator + bbox) / total if (bbox is not None and total > 0.0) else contrib
    return {
        "pair_ivr": reading.value,
        "numerator": numerator,
        "denominator": float(detail.get("denominator") or 0.0),
        "ivr_contrib": contrib,
        "ivr_contrib_hi": contrib_hi,
        "n_unknown": len(reading.unknown_pairs),
        "unknown": list(reading.unknown_pairs),
        "field": reading.field_name,
    }


def run_edge(
    assembly: AssemblyParts,
    parts: tuple,
    joint: dict,
    unit_mm: float,
    target: dict,
    *,
    arms: tuple[str, ...],
    attempts: int,
    deadline: float | None,
    ivr: bool,
) -> list[dict]:
    """Both arms for one edge, sharing the loaded assembly and the wall probe."""
    edge = edge_for(joint, unit_mm, _src_code(target))
    probe = probe_local_wall(
        parts[int(joint["a_index"])],
        parts[int(joint["b_index"])],
        edge["anchor"],
        edge["n_ab"],
        unit_mm=unit_mm,
    )
    file_record = {
        "t_a_mm": round(float(joint["t_a_mm"]), 6),
        "t_b_mm": round(float(joint["t_b_mm"]), 6),
        "depth_max_mm": round(float(joint["depth_max_mm"]), 6),
        "anchor_gap_mm": round(float(joint.get("anchor_gap_mm") or 0.0), 6),
        "interface_type": joint.get("interface_type"),
        "limiter": joint.get("limiter"),
        "wall_irrelevant": bool(joint.get("wall_irrelevant")),
        "half_w_mm": round(float(joint["half_w_mm"]), 6),
        "half_h_mm": round(float(joint["half_h_mm"]), 6),
    }
    wall_record = {
        "wall_mm": probe.wall_mm,
        "span_mm": probe.span_mm,
        "cavity_after_mm": probe.cavity_after_mm,
        "n_layers": probe.n_layers,
        "rays": probe.rays,
        "n_valid": probe.n_valid,
        "centre_ok": probe.centre_ok,
        "per_ray_mm": list(probe.per_ray_mm),
        "detail": probe.detail,
    }

    def ladder(arm: str, edge_now: dict, source: str, *, guard: bool) -> dict:
        tap = BuildTap(wall_mm=probe.wall_mm if guard else None, unit_mm=unit_mm)
        started = time.perf_counter()
        # The variant plans *under* the residual-wall floor, so it shrinks the pin
        # instead of eating the wall; the build port's veto is the last line of
        # defence for whatever the closed form cannot see. The baseline passes
        # neither, which is what makes it the M1 row.
        floor = (
            {"wall_need_abs": R_MIN_MM / unit_mm, "wall_need_frac": WALL_FRAC}
            if guard
            else {}
        )
        try:
            report = repair_edge(
                assembly,
                edge_now,
                budget=AttemptBudget(max_attempts=attempts, deadline=deadline),
                build=tap,
                **floor,
            )
            error = ""
        except Exception as exc:  # noqa: BLE001 - one edge must not end the pass
            report = None
            error = f"{type(exc).__name__}: {exc}"
            traceback.print_exc(limit=3)
        elapsed = time.perf_counter() - started
        row = _row(
            arm=arm,
            joint=joint,
            target=target,
            unit_mm=unit_mm,
            edge=edge,
            edge_now=edge_now,
            dcap_source=source,
            report=report,
            probe=probe,
            wall_record=wall_record,
            file_record=file_record,
            tap=tap,
            elapsed=elapsed,
            error=error,
        )
        if error:
            row["traceback"] = traceback.format_exc(limit=4)
        if ivr and row.get("success") and tap.last is not None:
            row["ivr"] = ivr_of(tap.last, parts, unit_mm)
        tap.last = None
        return row

    rows: list[dict] = []
    baseline_row: dict | None = None
    for arm in arms:
        if arm == "baseline":
            row = ladder(arm, edge, "file_const", guard=False)
            baseline_row = row
        elif probe.wall_mm is None:
            # No measurable wall at the anchor → overriding the cap would be a
            # guess, so the lever does not apply to this edge. Reuse the baseline
            # ladder's verdict (the treated edge dict *is* the baseline dict) and
            # say so, instead of spending a second identical ladder.
            if baseline_row is not None:
                row = {
                    **baseline_row,
                    "arm": arm,
                    "dcap_source": "unmeasured",
                    "reused_baseline": True,
                    "override": None,
                    "floor_on": arm == "variant",
                }
            else:
                row = ladder(arm, edge, "unmeasured", guard=False)
        else:
            row = ladder(
                arm,
                override_depth_cap(edge, wall_mm=probe.wall_mm, unit_mm=unit_mm),
                "local_single_wall",
                # `cap_only` changes the cap and nothing else; `variant` is the
                # brief's protocol, which also plans under the wall floor.
                guard=arm == "variant",
            )
        rows.append(row)
    return rows


def _row(
    *,
    arm: str,
    joint: dict,
    target: dict,
    unit_mm: float,
    edge: dict,
    edge_now: dict,
    dcap_source: str,
    report,
    probe,
    wall_record: dict,
    file_record: dict,
    tap: BuildTap,
    elapsed: float,
    error: str,
) -> dict:
    """One ledger row: the ladder's verdict plus the measurement it was built on."""
    result = (report.result if report else {}) or {}
    r_mm = _mm(result.get("r"), unit_mm)
    wall_mm = _mm(result.get("wall_remaining"), unit_mm)
    wall_need = None if r_mm is None else 0.25 * r_mm
    printable = bool(
        report is not None
        and report.success
        and r_mm is not None
        and r_mm >= R_MIN_MM - 1e-6
        and wall_mm is not None
        and wall_need is not None
        and wall_mm >= wall_need - 1e-5
    )
    floor = None
    if probe.wall_mm is not None:
        floor = residual_wall_need_mm(probe.wall_mm)
    row = {
        "arm": arm,
        "case_id": joint["case_id"],
        "edge_id": int(joint["edge_id"]),
        "a": int(joint["a_index"]),
        "b": int(joint["b_index"]),
        "src_code": _src_code(target),
        "src_detail": target.get("detail", ""),
        "src_r_mm": target.get("r_mm"),
        "limiter": target.get("limiter", ""),
        "dcap_source": dcap_source,
        "floor_on": bool(tap.wall_mm is not None),
        "reused_baseline": False,
        "success": bool(report.success) if report is not None else False,
        "final_reason": (report.reason or "") if report is not None else "adapter_raised",
        "stopped": report.stopped if report is not None else "error",
        "exhausted": bool(report.exhausted) if report is not None else False,
        "operator": (report.final.get("operator", "") if report is not None else ""),
        "n_attempts": report.n_attempts if report is not None else 0,
        "build_calls": report.build_calls if report is not None else 0,
        "measure_calls": report.measure_calls if report is not None else 0,
        "wall_s": round(float(elapsed), 4),
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
            for a in (report.attempts if report is not None else ())
        ],
        "final": report.final if report is not None else {},
        "r_mm": r_mm,
        "depth_mm": _mm(result.get("depth"), unit_mm),
        "wall_remaining_mm": wall_mm,
        "wall_need_mm": wall_need,
        "residual_wall_floor_mm": round(floor, 6) if floor is not None else None,
        "residual_wall_margin_mm": (
            None if (wall_mm is None or floor is None) else round(wall_mm - floor, 6)
        ),
        "printable": printable,
        "guard_trips": int(tap.guard_trips),
        "guard_need_mm": tap.wall_need_mm,
        "dv_a": result.get("dv_a"),
        "dv_b": result.get("dv_b"),
        "peg_volume": result.get("peg_volume"),
        "sock_volume": result.get("sock_volume"),
        "anchor_shift_mm": (report.final.get("anchor_shift_mm") if report is not None else None),
        "file": file_record,
        "wall": wall_record,
        "override": (
            None
            if dcap_source != "local_single_wall"
            else {
                "d_cap_file_mm": round(float(edge["d_cap"]) * unit_mm, 6),
                "d_cap_local_mm": round(float(edge_now["d_cap"]) * unit_mm, 6),
                "depth_max_local_mm": round(float(edge_now["depth_max"]) * unit_mm, 6),
                "wall_mm": probe.wall_mm,
            }
        ),
        "built_dcap_mm": round(float(edge_now["d_cap"]) * unit_mm, 6),
        "unit_mm": round(float(unit_mm), 6),
    }
    if error:
        row["error"] = error
    return row


def _mm(value, unit_mm: float) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value) * unit_mm, 6)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# The pass
# --------------------------------------------------------------------------- #
def run(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / "rows.jsonl"
    started = time.time()
    deadline = started + max(1.0, args.minutes) * 60.0 - 90.0 if args.minutes else None

    targets = m1_targets(resolve_source(args.source, out), args.limit)
    by_case: dict[str, list[dict]] = {}
    for row in targets:
        by_case.setdefault(row["case_id"], []).append(row)
    arms = {
        "all": ARMS,
        "both": ARMS_BOTH,
        "baseline": ("baseline",),
        "cap_only": ("cap_only",),
        "variant": ("variant",),
    }[args.arms]
    done = {
        (r["arm"], r["case_id"], int(r["edge_id"]))
        for r in read_rows(rows_path)
    }
    if not args.resume:
        done = set()
    print(
        f"targets: {len(targets)} edges over {len(by_case)} assemblies; arms={arms};"
        f" {len(done)} rows already on disk",
        flush=True,
    )

    cases = {c.case_id: c for c in load_cases(EDGE_FILE, split="feasible")}
    ordered = [cid for cid in sorted(by_case) if cid in cases]
    missing = sorted(set(by_case) - set(cases))
    if missing:
        print(f"WARNING: {len(missing)} assemblies have no feasible case: {missing[:5]}", flush=True)

    n_new = 0
    with rows_path.open("a", encoding="utf-8") as sink:
        for index, case_id in enumerate(ordered, start=1):
            todo = [
                t for t in by_case[case_id]
                if any((arm, case_id, int(t["edge_id"])) not in done for arm in arms)
            ]
            if not todo:
                continue
            if deadline is not None and time.time() >= deadline:
                print(f"deadline reached before {case_id}; stopping cleanly", flush=True)
                break
            print(f"  [{index:2d}/{len(ordered)}] {case_id} ({len(todo)} edges)", flush=True)
            case = cases[case_id]
            try:
                parts, longest = _load_parts(case)
                unit_mm = _unit_mm(case, longest, "scale")
                assembly = AssemblyParts(
                    case_id=case.case_id,
                    unit_mm=unit_mm,
                    meshes=parts,
                    part_meta=tuple(p.get("meta") for p in case.parts),
                    n_parts=len(parts),
                )
                joints = {int(j["edge_id"]): {**j, "case_id": case.case_id}
                          for j in case.joints}
            except Exception:  # noqa: BLE001 - one assembly must not end the pass
                print(f"  !! {case_id} failed to load:\n{traceback.format_exc(limit=3)}",
                      flush=True)
                continue
            for target in todo:
                joint = joints[int(target["edge_id"])]
                rows = run_edge(
                    assembly,
                    parts,
                    joint,
                    unit_mm,
                    target,
                    arms=arms,
                    attempts=args.attempts,
                    deadline=deadline,
                    ivr=args.ivr == "repaired",
                )
                for row in rows:
                    key = (row["arm"], row["case_id"], int(row["edge_id"]))
                    if key in done:
                        continue
                    sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                    done.add(key)
                    n_new += 1
                    print(
                        f"    {row['arm']:<8} {row['case_id']}:{row['edge_id']:>4}"
                        f" {'OK ' if row['success'] else row['final_reason'][:20]:<21}"
                        f" wall={row['wall'].get('wall_mm')} src={row['dcap_source']}"
                        f" {row['n_attempts']}x {row['wall_s']:>6.1f}s",
                        flush=True,
                    )
                sink.flush()
                os.fsync(sink.fileno())
            del parts
    elapsed = time.time() - started
    print(f"\n{n_new} rows written to {rows_path} in {elapsed:.0f}s", flush=True)
    return 0


# --------------------------------------------------------------------------- #
# Baseline vs M1: the regression check
# --------------------------------------------------------------------------- #
#: Deterministic fields the ladder produces. Timing is excluded by construction —
#: `wall_s` / `elapsed_s` can never match — and `measure_calls` is included because
#: the number of ray passes is a property of the plan, not of the machine.
COMPARE_FIELDS = (
    "src_code", "success", "final_reason", "stopped", "exhausted", "operator",
    "n_attempts", "build_calls", "measure_calls", "final", "r_mm", "depth_mm",
    "wall_remaining_mm", "wall_need_mm", "printable", "dv_a", "dv_b",
    "peg_volume", "sock_volume", "anchor_shift_mm",
)


def compare(args) -> int:
    out = Path(args.out)
    mine = [r for r in read_rows(out / "rows.jsonl") if r["arm"] == "baseline"]
    ref = {(r["case_id"], int(r["edge_id"])): r for r in read_rows(ledger(args, out))}
    print(f"baseline rows: {len(mine)}; M1 rows: {len(ref)}")
    mismatches: list[dict] = []
    checked = 0
    for row in mine:
        key = (row["case_id"], int(row["edge_id"]))
        other = ref.get(key)
        if other is None:
            mismatches.append({"key": key, "field": "<missing in M1>"})
            continue
        checked += 1
        for field in COMPARE_FIELDS:
            a, b = row.get(field), other.get(field)
            if field in ("dv_a", "dv_b", "peg_volume", "sock_volume") and _close(a, b):
                continue
            if a != b:
                mismatches.append({"key": key, "field": field, "baseline": a, "m1": b})
        for i, (mine_attempt, ref_attempt) in enumerate(
            zip(row.get("attempts") or [], other.get("attempts") or [])
        ):
            for field in ("operator", "success", "reason", "raw_reason", "params"):
                a = mine_attempt.get(field)
                b = ref_attempt.get(field)
                if a != b:
                    mismatches.append(
                        {"key": key, "field": f"attempts[{i}].{field}", "baseline": a, "m1": b}
                    )
        if len(row.get("attempts") or []) != len(other.get("attempts") or []):
            mismatches.append({
                "key": key, "field": "attempts.length",
                "baseline": len(row.get("attempts") or []),
                "m1": len(other.get("attempts") or []),
            })
    report = {
        "checked_edges": checked,
        "baseline_rows": len(mine),
        "m1_rows": len(ref),
        "mismatches": len(mismatches),
        # `identical` is the claim the brief asks for: every field the ladder
        # decides matches, edge for edge. `complete` says whether all 238 were run.
        "identical": not mismatches and checked == len(mine) and checked > 0,
        "complete": checked == M1_ROWS,
        "details": mismatches[:40],
    }
    (out / "compare.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "details"}, ensure_ascii=False))
    for item in mismatches[:10]:
        print("  MISMATCH", item)
    return 0 if report["identical"] else 1


def _close(a, b) -> bool:
    try:
        return math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-15)
    except (TypeError, ValueError):
        return a == b


# --------------------------------------------------------------------------- #
# The six groups
# --------------------------------------------------------------------------- #
def _paired(baseline: dict, variant: dict) -> dict:
    """McNemar cells over the edges both arms ran."""
    keys = sorted(set(baseline) & set(variant))
    b = c = both_ok = both_bad = 0
    for key in keys:
        bs, vs = bool(baseline[key]["success"]), bool(variant[key]["success"])
        if vs and not bs:
            b += 1
        elif bs and not vs:
            c += 1
        elif vs and bs:
            both_ok += 1
        else:
            both_bad += 1
    return {
        "n": len(keys),
        "variant_only": b,
        "baseline_only": c,
        "both_success": both_ok,
        "both_fail": both_bad,
        "mcnemar_exact_p": _mcnemar_p(b, c),
        "baseline_success": both_ok + c,
        "variant_success": both_ok + b,
    }


def _mcnemar_p(b: int, c: int) -> float | None:
    """Exact two-sided McNemar p (binomial on the discordant pairs)."""
    n = b + c
    if n == 0:
        return None
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2.0**n)
    return min(1.0, 2.0 * tail)


def _codes(rows: dict) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows.values():
        code = "" if row.get("success") else (row.get("final_reason") or "")
        out[code] = out.get(code, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def _quantiles(values: list[float]) -> dict | None:
    vals = sorted(float(v) for v in values)
    if not vals:
        return None
    def q(p: float) -> float:
        if len(vals) == 1:
            return vals[0]
        pos = p * (len(vals) - 1)
        lo, hi = int(math.floor(pos)), int(math.ceil(pos))
        if lo == hi:
            return vals[lo]
        return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)
    return {
        "n": len(vals),
        "min": round(vals[0], 6),
        "p05": round(q(0.05), 6),
        "p50": round(q(0.5), 6),
        "p95": round(q(0.95), 6),
        "max": round(vals[-1], 6),
    }


# --------------------------------------------------------------------------- #
# The six groups
# --------------------------------------------------------------------------- #
#: Which treatment arm each table is reported for, and against what.
TREATMENTS = (
    ("cap_only", "baseline"),
    ("variant", "baseline"),
    ("variant", "cap_only"),
)


def _paired(left: dict, right: dict) -> dict:
    """McNemar cells for `right` (the treatment) against `left` (the control).

    `b` = the treatment repaired it and the control did not; `c` = the other way
    round. Concordant edges are counted but say nothing about the difference.
    """
    keys = sorted(set(left) & set(right))
    b = c = both_ok = both_bad = 0
    for key in keys:
        lo, rt = bool(left[key]["success"]), bool(right[key]["success"])
        if rt and not lo:
            b += 1
        elif lo and not rt:
            c += 1
        elif rt:
            both_ok += 1
        else:
            both_bad += 1
    return {
        "n": len(keys),
        "treatment_only": b,
        "control_only": c,
        "both_success": both_ok,
        "both_fail": both_bad,
        "mcnemar_exact_p": _mcnemar_p(b, c),
        "control_success": both_ok + c,
        "treatment_success": both_ok + b,
    }


def _mcnemar_p(b: int, c: int) -> float | None:
    """Exact two-sided McNemar p (binomial on the discordant pairs)."""
    n = b + c
    if n == 0:
        return None
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2.0**n)
    return min(1.0, 2.0 * tail)


def _codes(rows: dict) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows.values():
        code = "" if row.get("success") else (row.get("final_reason") or "")
        out[code] = out.get(code, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def _quantiles(values: list[float]) -> dict | None:
    vals = sorted(float(v) for v in values)
    if not vals:
        return None

    def q(p: float) -> float:
        if len(vals) == 1:
            return vals[0]
        pos = p * (len(vals) - 1)
        lo, hi = int(math.floor(pos)), int(math.ceil(pos))
        if lo == hi:
            return vals[lo]
        return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)

    return {
        "n": len(vals),
        "min": round(vals[0], 6),
        "p05": round(q(0.05), 6),
        "p50": round(q(0.5), 6),
        "p95": round(q(0.95), 6),
        "max": round(vals[-1], 6),
    }


def _expected_success(arm_rows: dict) -> dict:
    """The success code a `connector.build` would return if the wall were ignored."""
    return arm_rows


def summarize(args) -> int:
    out = Path(args.out)
    rows = read_rows(out / "rows.jsonl")
    by_arm: dict[str, dict] = {}
    for arm in ARMS:
        by_arm[arm] = {(r["case_id"], int(r["edge_id"])): r for r in rows if r["arm"] == arm}
    have = {arm: len(rows_in) for arm, rows_in in by_arm.items() if rows_in}
    if "baseline" not in have:
        raise SystemExit(f"{out/'rows.jsonl'}: the baseline arm is missing")
    treatments = [t for t in TREATMENTS if t[0] in have and t[1] in have]

    m1 = {(r["case_id"], int(r["edge_id"])): r for r in read_rows(ledger(args, out))}
    # The three denominators the brief names: the 73 that ended on `gap_too_wide`,
    # the 80 still on one of the two codes, and the whole 238-edge M1 population.
    groups = {
        "g73": {k for k, r in m1.items() if not r["success"]
                and r["final_reason"] == "gap_too_wide"},
        "e80": {k for k, r in m1.items() if not r["success"]
                and r["final_reason"] in ("gap_too_wide", "no_material_hit")},
        "all238": set(m1),
    }
    groups = {
        name: (keys & set(by_arm["baseline"]) & set.intersection(*[set(by_arm[a]) for a, _ in treatments]))
        for name, keys in groups.items()
    }

    analysis: dict = {
        "out": str(out),
        "rows": have,
        "arms": list(have),
        "groups": {name: len(keys) for name, keys in groups.items()},
    }

    # --- 1. repairs gained, per treatment and denominator ----------------
    repairs: dict = {}
    for treatment, control in treatments:
        repairs[f"{treatment}_vs_{control}"] = {
            name: _paired({k: by_arm[control][k] for k in keys},
                          {k: by_arm[treatment][k] for k in keys})
            for name, keys in groups.items()
        }
        # `c` is not one thing: an edge the control repaired and the treatment did
        # not can be the wall standard biting (the brief's floor) or the cap
        # change going the wrong way. Splitting it keeps the two from being read
        # as one regression.
        split = {"floor_veto": 0, "other": 0}
        for key in groups["all238"]:
            if not by_arm[control][key]["success"] or by_arm[treatment][key]["success"]:
                continue
            row = by_arm[treatment][key]
            if row.get("final_reason") == "degenerate_thin_wall" or int(
                row.get("guard_trips") or 0
            ):
                split["floor_veto"] += 1
            else:
                split["other"] += 1
        repairs[f"{treatment}_vs_{control}"]["control_only_split"] = split
    analysis["repairs"] = repairs

    # --- 2. socket_breaks_wall before/after ------------------------------
    analysis["socket_breaks_wall"] = {
        arm: {
            "socket_breaks_wall": sum(
                1 for r in rows_in.values() if r.get("final_reason") == "socket_breaks_wall"
            ),
            "degenerate_thin_wall": sum(
                1 for r in rows_in.values() if r.get("final_reason") == "degenerate_thin_wall"
            ),
            "success": sum(1 for r in rows_in.values() if r.get("success")),
            "n": len(rows_in),
        }
        for arm, rows_in in by_arm.items() if rows_in
    }

    # --- 3. IVR of the repaired edges ------------------------------------
    def ivr_block(arm_arm: str) -> dict:
        rows_in = [r for r in by_arm[arm_arm].values() if r.get("success") and r.get("ivr")]
        vals = [r["ivr"].get("pair_ivr") for r in rows_in if r["ivr"].get("pair_ivr") is not None]
        contribs = [r["ivr"].get("ivr_contrib") for r in rows_in
                    if r["ivr"].get("ivr_contrib") is not None]
        highs = [r["ivr"].get("ivr_contrib_hi") for r in rows_in
                 if r["ivr"].get("ivr_contrib_hi") is not None]
        return {
            "n_repaired": len(rows_in),
            "n_known": len(vals),
            "n_unknown": sum(1 for r in rows_in if (r["ivr"].get("n_unknown") or 0) > 0),
            "n_error": sum(1 for r in rows_in if r["ivr"].get("error")),
            "pair_ivr": _quantiles(vals),
            "contrib_lo": min(contribs + [0.0]) if contribs else None,
            "contrib_hi": max(highs) if highs else None,
            "contrib_p50": statistics.median(contribs) if contribs else None,
        }
    analysis["ivr"] = {arm: ivr_block(arm) for arm in have}

    # --- 4. other failure codes, per code --------------------------------
    analysis["codes"] = {
        arm: _codes(rows_in) for arm, rows_in in by_arm.items() if rows_in
    }
    transitions: dict[str, dict[str, int]] = {}
    for treatment, control in treatments:
        counts: dict[str, int] = {}
        for key in groups["all238"]:
            before = "" if by_arm[control][key].get("success") else (
                by_arm[control][key].get("final_reason") or ""
            )
            after = "" if by_arm[treatment][key].get("success") else (
                by_arm[treatment][key].get("final_reason") or ""
            )
            if before != after:
                label = f"{before or 'ok'} -> {after or 'ok'}"
                counts[label] = counts.get(label, 0) + 1
        transitions[f"{treatment}_vs_{control}"] = dict(
            sorted(counts.items(), key=lambda kv: -kv[1])
        )
    analysis["transitions"] = transitions

    # --- 5. residual wall ------------------------------------------------
    def wall_block(arm_arm: str) -> dict:
        rows_in = [r for r in by_arm[arm_arm].values() if r.get("success")]
        margins = [r["residual_wall_margin_mm"] for r in rows_in
                   if r.get("residual_wall_margin_mm") is not None]
        remain = [r["wall_remaining_mm"] for r in rows_in
                  if r.get("wall_remaining_mm") is not None]
        floors = [r["residual_wall_floor_mm"] for r in rows_in
                  if r.get("residual_wall_floor_mm") is not None]
        near = 0
        for r in rows_in:
            margin, floor = r.get("residual_wall_margin_mm"), r.get("residual_wall_floor_mm")
            if margin is not None and floor:
                near += 1 if abs(margin) <= 0.01 * float(floor) else 0
        return {
            "n": len(rows_in),
            "wall_remaining_mm": _quantiles(remain),
            "margin_mm": _quantiles(margins),
            "floors_mm": _quantiles(floors),
            # 踩线 = the build sits exactly on its floor; "near" = within 1 % of it.
            "at_floor": sum(1 for m in margins if abs(m) <= 1e-6),
            "near_floor": near,
            "below_floor": sum(1 for m in margins if m < -1e-6),
            "guard_trips": sum(int(r.get("guard_trips") or 0) for r in by_arm[arm_arm].values()),
            "depth_mm": _quantiles([r["depth_mm"] for r in rows_in
                                    if r.get("depth_mm") is not None]),
            "r_mm": _quantiles([r["r_mm"] for r in rows_in if r.get("r_mm") is not None]),
        }
    analysis["residual_wall"] = {arm: wall_block(arm) for arm in have}

    # --- the measurement itself (the evidence for the hypothesis) --------
    measured = [
        r for r in by_arm[treatments[-1][0]].values()
        if r.get("wall", {}).get("wall_mm") is not None and r.get("override")
    ]
    ratios = [
        row["wall"]["wall_mm"] / row["file"]["t_b_mm"]
        for row in measured if (row["file"]["t_b_mm"] or 0) > 0
    ]
    span_ratios = [
        row["wall"]["wall_mm"] / row["wall"]["span_mm"]
        for row in measured if row["wall"].get("span_mm")
    ]
    analysis["measurement"] = {
        "n_measured": len(measured),
        "n_unmeasured": sum(
            1 for r in by_arm[treatments[-1][0]].values()
            if r.get("wall", {}).get("wall_mm") is None
        ),
        "n_targets": len(by_arm[treatments[-1][0]]),
        "wall_over_file_t_b": _quantiles(ratios),
        "wall_over_span": _quantiles(span_ratios),
        "d_cap_gain": _quantiles([
            row["override"]["d_cap_local_mm"] / row["override"]["d_cap_file_mm"]
            for row in measured if row["override"]["d_cap_file_mm"] > 0
        ]),
        "d_cap_shrink": sum(
            1 for row in measured
            if row["override"]["d_cap_local_mm"] < row["override"]["d_cap_file_mm"]
        ),
        "wall_mm": _quantiles([row["wall"]["wall_mm"] for row in measured]),
        "n_with_cavity_after": sum(
            1 for row in measured if row["wall"].get("cavity_after_mm")
        ),
        "n_multi_layer": sum(1 for row in measured if (row["wall"].get("n_layers") or 0) > 1),
    }

    (out / "analysis.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    text = _summary_md(analysis)
    (out / "summary.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


def _summary_md(a: dict) -> str:
    def pct(x: int, n: int) -> str:
        return f"{x}/{n} = {100.0 * x / n:.1f}%" if n else f"{x}/0"

    lines = [
        "# M1-dcap 变体：边级结果（自动生成）",
        "",
        f"行数（按臂）：{a['rows']}；分母组：{a['groups']}",
        "",
        "## 1. 修复成功数（配对）",
        "",
        "| 处理 vs 对照 | 组 | n | 对照成功 | 处理成功 | b（仅处理成功） | c（仅对照成功） | McNemar p |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, per_group in a["repairs"].items():
        for name, block in per_group.items():
            if name == "control_only_split":
                continue
            p = block["mcnemar_exact_p"]
            lines.append(
                f"| {label} | {name} | {block['n']} |"
                f" {pct(block['control_success'], block['n'])} |"
                f" {pct(block['treatment_success'], block['n'])} | {block['treatment_only']} |"
                f" {block['control_only']} | {('n/a' if p is None else f'{p:.3g}')} |"
            )
    lines += ["", "c 的归因（238 全集）：", ""]
    for label, per_group in a["repairs"].items():
        split = per_group.get("control_only_split") or {}
        lines.append(
            f"- {label}：c = {per_group['all238']['control_only']}"
            f"（被壁厚守则否掉 {split.get('floor_veto', 0)}，其余 {split.get('other', 0)}）"
        )

    lines += ["", "## 2. `socket_breaks_wall` 与壁厚否决", "",
              "| 臂 | 成功 | `socket_breaks_wall` | `degenerate_thin_wall` | n |",
              "|---|---:|---:|---:|---:|"]
    for arm, block in a["socket_breaks_wall"].items():
        lines.append(
            f"| {arm} | {block['success']} | {block['socket_breaks_wall']} |"
            f" {block['degenerate_thin_wall']} | {block['n']} |"
        )

    lines += ["", "## 3. IVR（修好的边；pair-local 与装配级贡献）", ""]
    for arm, block in a["ivr"].items():
        q = block["pair_ivr"]
        lines.append(
            f"- **{arm}**：n={block['n_repaired']}，已知 {block['n_known']}，"
            f"unknown {block['n_unknown']}，error {block['n_error']}；"
            f"pair IVR p50={_fmt(q and q['p50'])}（min {_fmt(q and q['min'])}/"
            f"max {_fmt(q and q['max'])}）；装配级贡献 [lo, hi] ="
            f" [{_fmt(block['contrib_lo'])}, {_fmt(block['contrib_hi'])}]"
        )

    lines += ["", "## 4. 失败码前后对比（238 全集）", "",
              "| 码 | " + " | ".join(a["codes"]) + " |",
              "|---|" + "---:|" * len(a["codes"])]
    all_codes = sorted({code for block in a["codes"].values() for code in block})
    for code in all_codes:
        cells = " | ".join(str(a["codes"][arm].get(code, 0)) for arm in a["codes"])
        lines.append(f"| `{code or 'ok'}` | {cells} |")
    lines += ["", "变化（按边、每对配对）：", ""]
    for label, counts in a["transitions"].items():
        lines.append(f"- {label}：" + ("；".join(f"`{k}` {v}" for k, v in counts.items()) or "无变化"))

    lines += ["", "## 5. 最小剩余壁厚（修好的边）", ""]
    for arm, block in a["residual_wall"].items():
        q, m, f = block["wall_remaining_mm"], block["margin_mm"], block["floors_mm"]
        lines.append(
            f"- **{arm}**：n={block['n']}；剩余壁厚 p50={_fmt(q and q['p50'])}"
            f" p05={_fmt(q and q['p05'])} min={_fmt(q and q['min'])}；"
            f"floor p50={_fmt(f and f['p50'])}；余量 p50={_fmt(m and m['p50'])}"
            f" min={_fmt(m and m['min'])}；踩线（=floor）{block['at_floor']}，"
            f"距 floor ≤1% {block['near_floor']}，低于 floor {block['below_floor']}；"
            f"r p50={_fmt((block['r_mm'] or {}).get('p50'))}，"
            f"eng p50={_fmt((block['depth_mm'] or {}).get('p50'))}；"
            f"守卫生效 {block['guard_trips']} 次"
        )

    lines += ["", "## 6. 测量本身（假设的证据）", ""]
    mm = a["measurement"]
    lines.append(
        f"- 可测 {mm['n_measured']} / {mm['n_targets']} 条边（其余在锚点测不到 socket 侧材质，"
        f"处理不适用、沿用对照判定）；墙后有空腔 {mm['n_with_cavity_after']} 条，"
        f"多层 {mm['n_multi_layer']} 条；局部 d_cap 比文件 d_cap 变小 {mm['d_cap_shrink']} 条"
    )
    for label, key in (("local wall / file t_b", "wall_over_file_t_b"),
                       ("local wall / span", "wall_over_span"),
                       ("local d_cap / file d_cap", "d_cap_gain"),
                       ("local wall (mm)", "wall_mm")):
        q = mm[key]
        if q:
            lines.append(
                f"- {label}：p05 {_fmt(q['p05'])} / p50 {_fmt(q['p50'])} / p95 {_fmt(q['p95'])}"
                f" / min {_fmt(q['min'])} / max {_fmt(q['max'])}"
            )
    return "\n".join(lines) + "\n"


def _fmt(value) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.4g}"
    except (TypeError, ValueError):
        return str(value)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="runs/m1-dcap")
    parser.add_argument("--source", default="runs/m1-min/rows.jsonl")
    parser.add_argument(
        "--arms", choices=("all", "both", "baseline", "cap_only", "variant"), default="all"
    )
    parser.add_argument("--attempts", type=int, default=MAX_ATTEMPTS)
    parser.add_argument("--minutes", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=0, help="max assemblies")
    parser.add_argument("--ivr", choices=("repaired", "none"), default="repaired")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args(argv)
    if args.summarize:
        return summarize(args)
    if args.compare:
        return compare(args)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
