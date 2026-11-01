import json, collections, pathlib
ROWS = pathlib.Path("runs/clearance-scan/rows.jsonl")
pairs = []; asms = {}
for line in ROWS.open(encoding="utf-8"):
    r = json.loads(line)
    if r.get("kind") == "pair": pairs.append(r)
    elif r.get("kind") == "assembly": asms[r["asm"]] = r
by = collections.defaultdict(list)
for p in pairs: by[p["asm"]].append(p)
# zero pairs per asm
zero = {a: sum(1 for p in v if p["dist_mm"] == 0.0) for a, v in by.items()}
pos = {a: [p for p in v if p["dist_mm"] > 0] for a, v in by.items()}
sub01 = {a: [p for p in v if 0.01 < p["dist_mm"] <= 0.1] for a, v in by.items()}
print("== top zero-pair assemblies ==")
for a in sorted(zero, key=lambda k: -zero[k])[:6]:
    r = asms[a]
    print(f"  {a} parts={r['n_parts']} zero={zero[a]} exact={r['pairs_exact']} pos={len(pos[a])} sub01={len(sub01[a])}")
print("== fewest parts ==")
for a in sorted(asms, key=lambda k: (asms[k]["n_parts"], k))[:8]:
    print(f"  {a} parts={asms[a]['n_parts']} zero={zero.get(a,0)} exact={asms[a]['pairs_exact']} pos={len(pos.get(a,[]))}")
print("== most parts ==")
for a in sorted(asms, key=lambda k: (-asms[k]["n_parts"], k))[:6]:
    print(f"  {a} parts={asms[a]['n_parts']} zero={zero.get(a,0)} exact={asms[a]['pairs_exact']} pos={len(pos.get(a,[]))}")
print("== nonzero in (0.01, 0.1] mm ==")
for a, v in sub01.items():
    if v: print(f"  {a} parts={asms[a]['n_parts']} n_sub01={len(v)} :", [(p['a'],p['b'],round(p['dist_mm'],4)) for p in v])
print("== asm_000062 detail ==")
print(f"  parts={asms['asm_000062']['n_parts']} exact={asms['asm_000062']['pairs_exact']} zero={zero['asm_000062']} nonzero={[(p['a'],p['b'],p['dist_mm']) for p in pos['asm_000062']]}")
