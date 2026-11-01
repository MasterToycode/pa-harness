"""Read-only analysis of runs/m1-min/rows.jsonl for the M1 report.

Not a deliverable: it re-reads the ledger and prints the breakdowns the report
quotes, so every number in the prose can be regenerated without re-running a
build. Nothing is written.
"""
from __future__ import annotations

import collections
import json
import pathlib
import statistics

OUT = pathlib.Path(__file__).resolve().parent
ROWS = OUT / "rows.jsonl"


def rows() -> list[dict]:
    return [json.loads(l) for l in ROWS.read_text(encoding="utf-8").splitlines() if l.strip()]


def pct(a: int, b: int) -> str:
    return f"{a}/{b} = {a / b:.1%}" if b else "n/a"


def q(values, p):
    clean = sorted(v for v in values if isinstance(v, (int, float)))
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    pos = p * (len(clean) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(clean) - 1)
    return round(clean[lo] + (clean[hi] - clean[lo]) * (pos - lo), 4)


def main() -> None:
    rs = rows()
    print(f"rows {len(rs)}")
    ok = [r for r in rs if r["success"]]
    print(f"repaired {pct(len(ok), len(rs))}")
    for code in ("gap_too_wide", "no_material_hit"):
        sub = [r for r in rs if r["src_code"] == code]
        good = [r for r in sub if r["success"]]
        print(f"  {code:16} {pct(len(good), len(sub))}")

    print("\nby operator that succeeded:")
    for name, count in collections.Counter(r["operator"] for r in ok).most_common():
        print(f"  {name:20} {count}")

    print("\nby src_code x winning operator:")
    for code in ("gap_too_wide", "no_material_hit"):
        for name, count in collections.Counter(
            r["operator"] for r in ok if r["src_code"] == code
        ).most_common():
            print(f"  {code:16} {name:20} {count}")

    print("\nby src_code x attempts:")
    for code in ("gap_too_wide", "no_material_hit"):
        hist = collections.Counter(r["n_attempts"] for r in rs if r["src_code"] == code)
        print(f"  {code:16} {dict(sorted(hist.items()))}")

    print("\nradius of repaired edges (mm):")
    rr = [r["r_mm"] for r in ok]
    print(f"  min {q(rr,0)} p25 {q(rr,0.25)} p50 {q(rr,0.5)} p75 {q(rr,0.75)} p95 {q(rr,0.95)} max {q(rr,1)}")
    floor = [r for r in ok if r["r_mm"] is not None and r["r_mm"] <= 0.2501]
    print(f"  at the 0.25 mm print floor: {pct(len(floor), len(ok))}")
    ratios = [r["wall_remaining_mm"] / r["wall_need_mm"] for r in ok
              if r.get("wall_remaining_mm") and r.get("wall_need_mm")]
    print(f"  wall_remaining / (0.25 r): min {q(ratios,0)} p50 {q(ratios,0.5)}")
    print(f"  printable: {pct(sum(1 for r in ok if r['printable']), len(ok))}")

    print("\nshift of repaired edges (mm):")
    sh = [r["anchor_shift_mm"] for r in ok]
    print(f"  p50 {q(sh,0.5)} p95 {q(sh,0.95)} max {q(sh,1)}  moved>0: {sum(1 for s in sh if s and s > 1e-9)}")

    print("\nresidual:")
    bad = [r for r in rs if not r["success"]]
    for name, count in collections.Counter(r["final_reason"] for r in bad).most_common():
        print(f"  {name or '(none)':22} {count}")
    print("  by stopped:", dict(collections.Counter(r["stopped"] for r in bad)))
    print("  by src x final:")
    for name, count in collections.Counter(
        f"{r['src_code']}->{r['final_reason']}" for r in bad
    ).most_common():
        print(f"    {name:40} {count}")

    print("\nattempts where a gate prediction was made:")
    agree = disagree = fp = fn = 0
    for r in rs:
        for a in r["attempts"]:
            params = a.get("params") or {}
            if "gate_predicted" not in params:
                continue
            predicted = bool(params["gate_predicted"])
            passed = a["success"] or a["reason"] in {
                "peg_detached", "socket_breaks_wall", "open_after_bool", "bool_failed"}
            agree += predicted == passed
            fp += predicted and not passed
            fn += (not predicted) and passed
            disagree += predicted != passed
    print(f"  agree {agree} disagree {disagree} (false+ {fp} false- {fn})")

    print("\nparameter spread of winning plans:")
    print("  sections:", dict(collections.Counter(
        (r["final"].get("params") or {}).get("section") for r in ok)))

    print("\ncost:")
    print("  builds", sum(r["build_calls"] for r in rs),
          "measures", sum(r["measure_calls"] for r in rs),
          "attempts", sum(r["n_attempts"] for r in rs))
    print("  wall per edge p50", q([r["wall_s"] for r in rs], 0.5),
          "p95", q([r["wall_s"] for r in rs], 0.95))
    slow = sorted(rs, key=lambda r: -r["wall_s"])[:5]
    for r in slow:
        print(f"    slow {r['case_id']}:{r['edge_id']} {r['wall_s']}s {r['build_calls']}b/{r['measure_calls']}m")

    # does the residual correlate with the recorded interface type / limiter?
    edges_file = pathlib.Path(r"D:\yanyi\rl_assembly\research\contact-graph-anchor\edges_v2.json")
    data = json.loads(edges_file.read_text(encoding="utf-8"))
    info = {(e["assembly_id"], e["id"]): e for e in data["edges"]}
    print("\ninterface_type x verdict:")
    for code in ("gap_too_wide", "no_material_hit"):
        for kind in ("coincident", "near"):
            sub = [r for r in rs if r["src_code"] == code
                   and (info.get((r["case_id"], r["edge_id"])) or {}).get("interface_type") == kind]
            good = [r for r in sub if r["success"]]
            print(f"  {code:16} {kind:11} {pct(len(good), len(sub))}")
    print("\nresidual gap_too_wide: is the depth cap the binding term?")
    import re
    pat = re.compile(r"gap=([0-9.eE+-]+)mm,need=([0-9.eE+-]+)mm>eng_max=([0-9.eE+-]+)mm")
    depth_bound = 0
    for r in rs:
        if r["success"] or r["final_reason"] != "gap_too_wide":
            continue
        detail = (r["attempts"][-1]["detail"] if r["attempts"] else "")
        m = pat.search(detail)
        e = info.get((r["case_id"], r["edge_id"])) or {}
        if m and e:
            gap, need, eng = (float(x) for x in m.groups())
            if abs(eng - e["depth_max_mm"]) < 1e-6:
                depth_bound += 1
    print(f"  last attempt's eng_max == the file's depth_max: {depth_bound} of "
          f"{sum(1 for r in rs if not r['success'] and r['final_reason'] == 'gap_too_wide')}")



if __name__ == "__main__":
    main()
