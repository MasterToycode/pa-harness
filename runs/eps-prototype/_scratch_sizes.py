import time, numpy as np, trimesh
from pathlib import Path
from scipy.spatial import cKDTree
G = Path(r"D:\yanyi\rl_assembly\data\assemblies_pre_decimate")
ma = trimesh.load(G/"asm_000062"/"parts"/"part_000.ply", process=False)
mb = trimesh.load(G/"asm_000062"/"parts"/"part_008.ply", process=False)
# max edge length of B (rigorous radius: if p is inside a B triangle, nearest B vertex is within its longest edge)
e = np.linalg.norm(np.asarray(mb.vertices)[np.asarray(mb.edges_unique)[:,0]] - np.asarray(mb.vertices)[np.asarray(mb.edges_unique)[:,1]], axis=1)
print(f"B edge len mm: max={e.max()*1000:.4f} p999={np.percentile(e,99.9)*1000:.4f} p50={np.percentile(e,50)*1000:.5f}")
tb = cKDTree(mb.vertices)
t0=time.perf_counter(); d,_ = tb.query(np.asarray(ma.vertices), k=1); print(f"kd query A->Bvert {time.perf_counter()-t0:.2f}s")
for R in (e.max(), np.percentile(e,99.9)):
    print(f"  A verts within {R*1000:.4f} mm of a B vertex: {int((d<=R).sum())} / {len(ma.vertices)}")
# how many B tris in the near region
print("B bbox", np.round(mb.bounds,4).tolist())
sel = d <= e.max()
print("cand bbox", np.round(np.asarray(ma.vertices)[sel].min(0),4).tolist(), np.round(np.asarray(ma.vertices)[sel].max(0),4).tolist())
