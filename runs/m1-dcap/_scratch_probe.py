"""Recon 2 (read-only): the local single wall on the 73, before any repair runs.

For each of the 73 (plus the 7 `no_material_hit`) edges: measure B's first layer at
the recorded anchor, and report the file's `depth_max` / `t_b` beside it. No builds,
no writes.

Run:  python runs/m1-dcap/_scratch_probe.py [max_assemblies]
"""

from __future__ import annotations

import json
import statistics as st
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pah.policy.dcap import (  # noqa: E402
    depth_cap_mm,
    probe_local_wall,
    residual_wall_need_mm,
)
from pah.tools.verifiers.connector_feasible import (  # noqa: E402
    EDGE_FILE,
    _load_parts,
    _unit_mm,
    load_cases,
    to_edge_dict,
)

ROWS = Path(__file__).resolve().parents[1] / "m1-min" / "rows.jsonl"


def main() -> None:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    rows = [json.loads(x) for x in ROWS.read_text(encoding="utf-8").splitlines() if x.strip()]
    targets = [r for r in rows if not r["success"]
               and r["final_reason"] in ("gap_too_wide", "no_material_hit")]
    print(f"targets: {len(targets)}")
    by_case: dict[str, list[dict]] = {}
    for r in targets:
        by_case.setdefault(r["case_id"], []).append(r)
    cases = {c.case_id: c for c in load_cases(EDGE_FILE, split="feasible")}

    print(f"{'edge':<20} {'src->fin':<28} {'t_a':>8} {'t_b':>9} {'d_cap':>8} "
          f"{'gap':>8} {'wall':>9} {'span':>10} {'cav':>7} {'0.8wall':>8} "
          f"{'need':>8} {'pred':>5}")
    walls = []
    t0 = time.time()
    for case_id in sorted(by_case)[:limit]:
        case = cases[case_id]
        parts, longest = _load_parts(case)
        unit_mm = _unit_mm(case, longest, "scale")
        joints = {int(j["edge_id"]): {**j, "case_id": case.case_id} for j in case.joints}
        for row in by_case[case_id]:
            joint = joints[row["edge_id"]]
            edge = to_edge_dict(joint, unit_mm)
            probe = probe_local_wall(
                parts[int(joint["a_index"])],
                parts[int(joint["b_index"])],
                edge["anchor"],
                edge["n_ab"],
                unit_mm=unit_mm,
            )
            gaps = [a["params"].get("gap_mm") for a in (row.get("attempts") or [])
                    if a["params"].get("gap_mm") is not None]
            gap = min(gaps) if gaps else float("nan")
            wall = probe.wall_mm
            cap = depth_cap_mm(wall) if wall else None
            need = gap + 0.25
            pred = "" if not wall else ("PASS" if cap > need else "fail")
            if wall:
                walls.append((wall, joint["t_b_mm"], probe.span_mm, joint["t_a_mm"]))
            print(
                f"{case_id}:{row['edge_id']:<7} {row['src_code'][:14] + '->' + row['final_reason'][:12]:<28}"
                f" {joint['t_a_mm']:>8.3f} {joint['t_b_mm']:>9.3f} {joint['depth_max_mm']:>8.3f}"
                f" {gap:>8.3f} {(wall if wall else float('nan')):>9.4f}"
                f" {(probe.span_mm if probe.span_mm else float('nan')):>10.3f}"
                f" {(probe.cavity_after_mm if probe.cavity_after_mm else float('nan')):>7.3f}"
                f" {(cap if cap else float('nan')):>8.3f} {need:>8.3f} {pred:>5}"
                f"  [{probe.n_valid}/{probe.rays} {probe.detail[:24]}]"
            )
    print(f"\n{len(walls)} measured edges, {time.time() - t0:.1f}s")
    if walls:
        ratio = sorted(w / tb for w, tb, _s, _ta in walls if tb > 0)
        print("wall / file_t_b: p05 %.3f p50 %.3f p95 %.3f" % (
            st.quantiles(ratio, n=20)[0], st.median(ratio), st.quantiles(ratio, n=20)[-1]))
        span_ratio = sorted(w / s for w, _tb, s, _ta in walls if s)
        print("wall / span    : p05 %.3f p50 %.3f p95 %.3f" % (
            st.quantiles(span_ratio, n=20)[0], st.median(span_ratio),
            st.quantiles(span_ratio, n=20)[-1]))


if __name__ == "__main__":
    main()
