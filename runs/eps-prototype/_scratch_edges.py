import json, collections, pathlib
rows = [json.loads(l) for l in pathlib.Path("runs/m0-connector/rows.jsonl").open(encoding="utf-8")]
ok = [r for r in rows if r.get("ok")]
by_asm = collections.defaultdict(list)
for r in rows: by_asm[r["case_id"]].append(r)
ok_by_asm = collections.defaultdict(list)
for r in ok: ok_by_asm[r["case_id"]].append(r)

# zero-gap pairs
zero = collections.defaultdict(set)
for line in pathlib.Path("runs/clearance-scan/rows.jsonl").open(encoding="utf-8"):
    r = json.loads(line)
    if r.get("kind")=="pair" and r["dist_mm"]==0.0:
        zero[r["asm"]].add(frozenset((r["a"],r["b"])))

cands = ["asm_000062","asm_000003","asm_000073","asm_000007","asm_000098","asm_000047","asm_000087","asm_000026","asm_000093","asm_000012","asm_000021"]
for a in cands:
    n_edges = len(by_asm.get(a, []))
    n_ok = len(ok_by_asm.get(a, []))
    okpairs = [f"p{r['a']}-p{r['b']}" for r in ok_by_asm.get(a, [])]
    zok = sum(1 for r in ok_by_asm.get(a,[]) if frozenset((f"part_{r['a']:03d}", f"part_{r['b']:03d}")) in zero[a])
    print(f"{a}: edges={n_edges} ok={n_ok} ok_on_zero_gap={zok} ok_pairs={okpairs}")
