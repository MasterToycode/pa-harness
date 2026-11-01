"""Scratch: does a closed-form radius fix rescue `gap_too_wide`?

Not a deliverable; it answers one question before the repair module is written:
on a real edge, is `eng_max <= need` fixed by asking for a smaller r, and how
expensive is `measure_axis` vs `build`?

Run: python runs/m1-min/_scratch_recon.py asm_000000
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, r"D:\yanyi\pa-harness\src")

from pah.tools.verifiers.connector_feasible import (  # noqa: E402
    _connector,
    _load_parts,
    _theta,
    _unit_mm,
    load_cases,
    to_edge_dict,
)


def main(case_id: str, wanted: list[int]) -> None:
    cases = {c.case_id: c for c in load_cases(split="feasible")}
    case = cases[case_id]
    t0 = time.time()
    parts, longest = _load_parts(case)
    unit_mm = _unit_mm(case, longest, "scale")
    print(f"loaded {len(parts)} parts in {time.time()-t0:.1f}s  unit_mm={unit_mm:.1f}")
    c = _connector()

    for joint in case.joints:
        if wanted and joint["edge_id"] not in wanted:
            continue
        edge = to_edge_dict(joint, unit_mm)
        theta = _theta(joint, edge, unit_mm)
        t1 = time.time()
        m = c.measure_axis(parts[joint["a_index"]], parts[joint["b_index"]], edge, theta)
        dt_m = time.time() - t1
        print(f"\n--- edge {joint['edge_id']}  measure {dt_m:.2f}s")
        print("   ", {k: (round(v * unit_mm, 4) if isinstance(v, float) else v)
                      for k, v in m.items() if k in ("ok", "detail", "gap", "t_a", "t_b", "penetration")})
        if not m.get("ok"):
            continue
        gap, t_b, t_a = m["gap"], m["t_b"], m["t_a"]
        d_cap = edge["depth_max"]
        r_cap = edge["r_inplane_max"]
        r_min = 0.25 / unit_mm
        print(f"    gap={gap*unit_mm:.4f}mm t_a={t_a*unit_mm:.3f} t_b={t_b*unit_mm:.3f} "
              f"d_cap={d_cap*unit_mm:.4f} r_cap={r_cap*unit_mm:.4f} r_min={r_min*unit_mm:.3f}")

        def predict(r):
            budget = 1.25 * r
            eng_max = min(d_cap, t_b + gap - budget) if t_b + gap - budget > 0 else -1
            need = gap + r if gap > 0 else 0.0
            return eng_max > need, need * unit_mm, eng_max * unit_mm

        for label, r in (("as-asked", theta[5]), ("r_min", r_min),
                         ("0.95*wallcap", 0.95 * t_b / 2.25)):
            ok_p, need, eng_max = predict(r)
            print(f"    r={r*unit_mm:.4f}mm  predict_pass={ok_p}  need={need:.4f} eng_max={eng_max:.4f}")
            depth = min(2 * r, d_cap)
            th = (0.0, 0.0, *edge["n_ab"], r, depth)
            t2 = time.time()
            res = c.build(parts[joint["a_index"]], parts[joint["b_index"]], edge, th,
                          clamp=True, verify_stl=False, overlap_check=True,
                          unit_mm=unit_mm, r_min_mm=0.25, depth_min_mm=0.0,
                          section="D", snap=None)
            print(f"    [{label}] build {time.time()-t2:.2f}s -> ok={res.get('ok')} "
                  f"reason={res.get('reason')} detail={str(res.get('detail'))[:110]}")
            if res.get("ok"):
                print("        r_mm=", round(res["r"] * unit_mm, 4),
                      " wall_remaining_mm=", round(res["wall_remaining"] * unit_mm, 4),
                      " dv_a=", res.get("dv_a"), " dv_b=", res.get("dv_b"))


if __name__ == "__main__":
    cid = sys.argv[1] if len(sys.argv) > 1 else "asm_000000"
    ids = [int(x) for x in sys.argv[2:]]
    main(cid, ids)
