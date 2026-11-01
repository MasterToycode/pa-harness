"""Recon (read-only): what does the ray actually see at the anchor on the socket side?

Prints, for a few of the 73 unrepaired `gap_too_wide` edges, the file-level walls
and the ray-level material intervals of both parts in mm. No writes, no repair.

Run:  python runs/m1-dcap/_scratch_recon.py [n_edges]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np  # noqa: E402

from pah.tools.verifiers.connector_feasible import (  # noqa: E402
    EDGE_FILE,
    _connector,
    _load_parts,
    _unit_mm,
    load_cases,
    to_edge_dict,
)

ROWS = Path(__file__).resolve().parents[1] / "m1-min" / "rows.jsonl"


def mm(x, u):
    return None if x is None else round(float(x) * u, 4)


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    rows = [json.loads(line) for line in ROWS.read_text(encoding="utf-8").splitlines() if line.strip()]
    targets = [r for r in rows if not r["success"] and r["final_reason"] == "gap_too_wide"]
    print(f"{len(targets)} edges with final_reason=gap_too_wide")
    by_case: dict[str, list[dict]] = {}
    for r in targets:
        by_case.setdefault(r["case_id"], []).append(r)
    picked = []
    for case_id in sorted(by_case):
        picked.append(by_case[case_id][0])
        if len(picked) >= n:
            break

    cases = {c.case_id: c for c in load_cases(EDGE_FILE, split="feasible")}
    conn = _connector()
    for row in picked:
        case = cases[row["case_id"]]
        parts, longest = _load_parts(case)
        unit_mm = _unit_mm(case, longest, "scale")
        joints = {int(j["edge_id"]): {**j, "case_id": case.case_id} for j in case.joints}
        joint = joints[row["edge_id"]]
        edge = to_edge_dict(joint, unit_mm)
        a_i, b_i = int(joint["a_index"]), int(joint["b_index"])
        anchor = np.asarray(edge["anchor"], dtype=float)
        normal = np.asarray(edge["n_ab"], dtype=float)
        normal = normal / np.linalg.norm(normal)

        print("=" * 100)
        print(
            f"{row['case_id']}:{row['edge_id']}  src={row['src_code']}"
            f"  final={row['final_reason']}  unit_mm={unit_mm:.3f}"
        )
        print(
            f"  file: t_a={joint['t_a_mm']:.4f} t_b={joint['t_b_mm']:.4f}"
            f" depth_max={joint['depth_max_mm']:.4f} r_inplane_max={joint['r_inplane_max_mm']:.4f}"
            f" anchor_gap={joint['anchor_gap_mm']:.4f} iface={joint['interface_type']}"
            f" limiter={joint['limiter']} wall_irrelevant={joint['wall_irrelevant']}"
        )
        print(
            f"  patch: half_w={joint['half_w_mm']:.3f} half_h={joint['half_h_mm']:.3f}"
            f"  a={a_i} b={b_i}  n_ab={[round(x, 3) for x in joint['n_ab']]}"
        )
        theta = (0.0, 0.0, *normal.tolist(), 0.25 / unit_mm, 0.45 / unit_mm)
        read = conn.measure_axis(parts[a_i], parts[b_i], edge, theta)
        iv_a, odd_a = conn.line_intervals(parts[a_i], read.get("p_plane"), normal)
        iv_b, odd_b = conn.line_intervals(parts[b_i], read.get("p_plane"), normal)
        print(f"  ray ok={read.get('ok')} reason={read.get('reason')} detail={read.get('detail')}")
        print(f"  ray: gap={mm(read.get('gap'), unit_mm)} t_a={mm(read.get('t_a'), unit_mm)}"
              f" t_b={mm(read.get('t_b'), unit_mm)} anchor_shift={mm(read.get('anchor_shift'), unit_mm)}")
        print(f"  iv_a (odd={odd_a}): {[(mm(a, unit_mm), mm(b, unit_mm)) for a, b in iv_a]}")
        print(f"  iv_b (odd={odd_b}): {[(mm(a, unit_mm), mm(b, unit_mm)) for a, b in iv_b]}")
        # socket geometry at the last attempted r
        last_r = None
        for att in row.get("attempts") or []:
            last_r = att["params"].get("r_mm")
        print(f"  last attempt r_mm={last_r} depth_max_mm={row['final'].get('d_cap_mm')}"
              f"  last detail={ (row.get('attempts') or [{}])[-1].get('detail', '')[:120] }")


if __name__ == "__main__":
    main()
