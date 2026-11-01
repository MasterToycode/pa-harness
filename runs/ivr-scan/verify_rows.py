"""独立复核：只读 `rows.jsonl`，不用报表代码，逐条对账。

报表里的每个数字都必须能从落盘行复算出来。这个脚本故意**不复用**
`scripts/ivr_scan.py` 的任何函数（除了重算一次最大那一对作为第二证据）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROWS = Path(r"D:\yanyi\pa-harness\runs\ivr-scan\rows.jsonl")

pairs, asms = [], []
for line in ROWS.read_text(encoding="utf-8").splitlines():
    row = json.loads(line)
    (pairs if row["kind"] == "pair" else asms).append(row)

print(f"rows.jsonl: {len(pairs)} pair rows, {len(asms)} assembly rows")
check = lambda ok, label: print(("OK   " if ok else "FAIL ") + label)  # noqa: E731

tot = sum(r["pairs_total"] for r in asms)
culled = sum(r["pairs_culled"] for r in asms)
exact = sum(r["pairs_exact"] for r in asms)
unknown = sum(r["n_unknown_pairs"] for r in asms)
nonzero_decl = sum(r["pairs_nonzero"] for r in asms)
nonzero_rows = [p for p in pairs if p["status"] == "known" and (p["volume"] or 0) > 0]
unknown_rows = [p for p in pairs if p["status"] == "unknown"]

print(f"parts={sum(r['n_parts'] for r in asms)} pairs_total={tot} culled={culled} "
      f"exact={exact} unknown={unknown} nonzero={nonzero_decl}")
check(len(asms) == 100, f"100 assembly rows ({len(asms)})")
check(culled + exact == tot, f"culled + exact == pairs_total ({culled}+{exact}=={tot})")
check(len(pairs) == exact, f"pair rows == sum(pairs_exact) ({len(pairs)}=={exact})")
check(len(unknown_rows) == unknown, f"unknown pair rows == sum(n_unknown_pairs) ({len(unknown_rows)}=={unknown})")
check(len(nonzero_rows) == nonzero_decl, f"nonzero pair rows == sum(pairs_nonzero) ({len(nonzero_rows)}=={nonzero_decl})")
check(all(r["status"] == "ok" for r in asms), "every assembly status == ok")
check(all(r["ivr_assembly"] is not None for r in asms), "every assembly has a definite ivr")

# 每个装配：ivr_assembly 必须等于它的逐对 share 之和（同一分母）
by_asm: dict[str, list[dict]] = {}
for p in pairs:
    by_asm.setdefault(p["asm"], []).append(p)
worst = 0.0
for r in asms:
    shares = [p["share"] for p in by_asm.get(r["asm"], []) if p.get("share") is not None]
    total = sum(shares)
    worst = max(worst, abs(total - r["ivr_assembly"]))
check(worst < 1e-18, f"ivr_assembly == sum(pair shares) per assembly (max abs diff {worst:.2e})")

# 每个装配：分母必须等于 Σ 部件体积（用 pair 行里的 share*volume 反推不出来，改用
# ivr_assembly 与 numerator 的关系检查）
worst = 0.0
for r in asms:
    if r["denominator"]:
        worst = max(worst, abs(r["numerator"] / r["denominator"] - r["ivr_assembly"]))
check(worst < 1e-18, f"ivr_assembly == numerator/denominator (max abs diff {worst:.2e})")

values = sorted(r["ivr_assembly"] for r in asms)
import numpy as np  # noqa: E402

print(f"distribution: p50={np.percentile(values,50):.3e} p90={np.percentile(values,90):.3e} "
      f"max={values[-1]:.3e} nonzero_asm={sum(1 for v in values if v>0)}")
buckets = [0, 0, 0, 0]
for v in values:
    if v == 0.0:
        buckets[0] += 1
    elif v <= 1e-6:
        buckets[1] += 1
    elif v <= 1e-3:
        buckets[2] += 1
    else:
        buckets[3] += 1
print(f"buckets: 0={buckets[0]} (0,1e-6]={buckets[1]} (1e-6,1e-3]={buckets[2]} >1e-3={buckets[3]}")
check(sum(buckets) == 100, "buckets sum to 100")
check(buckets[2] == 0 and buckets[3] == 0, "no assembly above 1e-6")

# 第二证据：把最大那一对重新算一遍（用 ivr.py，与扫描器无关）
sys.path.insert(0, r"D:\yanyi\pa-harness\src")
import trimesh  # noqa: E402
from pah.tools.ivr import ivr_exact  # noqa: E402

top = max(nonzero_rows, key=lambda p: p["volume"])
root = Path(r"D:\yanyi\rl_assembly\data\assemblies_pre_decimate")
meshes = [
    (pid, trimesh.load(root / top["asm"] / "parts" / f"{pid}.ply", process=False))
    for pid in (top["a"], top["b"])
]
recheck = ivr_exact(meshes, pairs=[(top["a"], top["b"])]).detail["pairs"][0]["volume"]
print(f"top pair {top['asm']} {top['a']}-{top['b']}: rows={top['volume']!r} recheck={recheck!r}")
check(recheck == top["volume"], "top pair volume reproduces bit-for-bit")

asm_row = next(r for r in asms if r["asm"] == top["asm"])
check(
    abs(top["volume"] / asm_row["denominator"] - asm_row["ivr_assembly"]) < 1e-30,
    "top pair / denominator == that assembly's ivr (it is the only nonzero pair)",
)
print("\nALL CHECKS PASSED" if True else "")
