"""Scratch: characterise every zero-gap pair of the 5 candidate assemblies.

Question this answers (it decides the algorithm): are the contacts **large coplanar
cut faces** (the "partition" picture in the task) or **curve/tangency contacts**
(a few hundred exactly-coincident vertices, no coplanar face)?

For each zero pair:
  n_zero_verts_a   A's vertices whose exact distance to B's surface is <= 1e-6 m
  faces_fully_on   A's faces whose 3 vertices are all on B
  area_on_mm2      area of those faces
  pca_ratio        eigenvalue ratio of the contact point set (1D curve? 2D sheet?)
  scatter_um       max |offset| of the contact points from their best-fit plane
  normal_spread    cos spread of the contact faces' normals
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

GEOM = Path(r"D:\yanyi\rl_assembly\data\assemblies_pre_decimate")
ROWS = Path(r"runs/clearance-scan/rows.jsonl")


def point_tri_distance(points, tris):
    a = tris[None, :, 0, :]
    b = tris[None, :, 1, :]
    c = tris[None, :, 2, :]
    ab = b - a
    ac = c - a
    ap = points[:, None, :] - a
    n = np.cross(ab, ac)
    n2 = np.einsum("ijk,ijk->ij", n, n)
    nd = n2 > 1e-30
    d = np.einsum("ijk,ijk->ij", n, ap)
    scale = np.where(nd, d / np.where(nd, n2, 1.0), 0.0)
    proj = points[:, None, :] - scale[:, :, None] * n
    v0, v1 = ab, ac
    v2 = proj - a
    d00 = np.einsum("ijk,ijk->ij", v0, v0)
    d01 = np.einsum("ijk,ijk->ij", v0, v1)
    d11 = np.einsum("ijk,ijk->ij", v1, v1)
    d20 = np.einsum("ijk,ijk->ij", v2, v0)
    d21 = np.einsum("ijk,ijk->ij", v2, v1)
    den = d00 * d11 - d01 * d01
    usable = np.abs(den) > 1e-30
    den_s = np.where(usable, den, 1.0)
    bv = (d11 * d20 - d01 * d21) / den_s
    bw = (d00 * d21 - d01 * d20) / den_s
    inside = usable & (bv >= 0) & (bw >= 0) & (bv + bw <= 1.0)
    best = np.where(inside, np.linalg.norm(proj - points[:, None, :], axis=2), np.inf)
    for s0, s1 in ((a, b), (b, c), (c, a)):
        edge = s1 - s0
        e2 = np.einsum("ijk,ijk->ij", edge, edge)
        t = np.einsum("ijk,ijk->ij", points[:, None, :] - s0, edge) / np.where(e2 > 1e-30, e2, 1.0)
        t = np.clip(t, 0.0, 1.0)
        seg = s0 + t[:, :, None] * edge
        best = np.minimum(best, np.linalg.norm(seg - points[:, None, :], axis=2))
    return best


class SurfaceIndex:
    def __init__(self, mesh):
        self.tris = np.ascontiguousarray(mesh.triangles)
        self.cen = self.tris.mean(axis=1)
        self.rad = np.linalg.norm(self.tris - self.cen[:, None, :], axis=2).max(axis=1)
        self.r_max = float(self.rad.max()) if len(self.rad) else 0.0
        self.tree = cKDTree(self.cen)

    def distance(self, points, k=32):
        out = np.full(len(points), np.inf)
        todo = np.arange(len(points))
        kk = k
        while len(todo):
            take = min(kk, len(self.cen))
            dist, idx = self.tree.query(points[todo], k=take)
            dist = np.atleast_2d(dist)
            idx = np.atleast_2d(idx)
            for row, pi in enumerate(todo):
                d = point_tri_distance(points[pi][None, :], self.tris[idx[row]])
                out[pi] = min(out[pi], float(d.min()))
            far = dist[:, -1]
            stop = (far - self.r_max > out[todo]) | (take >= len(self.cen))
            todo = todo[~stop]
            kk = min(kk * 4, len(self.cen))
            if take >= len(self.cen):
                break
        return out


def contact_stats(mesh_a, mesh_b, tol=1e-6):
    va = np.asarray(mesh_a.vertices, dtype=float)
    vb = np.asarray(mesh_b.vertices, dtype=float)
    fa = np.asarray(mesh_a.faces)
    dvert, _ = cKDTree(vb).query(va, k=1, workers=-1)
    e = np.linalg.norm(vb[np.asarray(mesh_b.edges_unique)[:, 0]] - vb[np.asarray(mesh_b.edges_unique)[:, 1]], axis=1)
    cand = np.flatnonzero(dvert <= float(e.max()) + tol)
    d = np.full(len(va), np.inf)
    if len(cand):
        d[cand] = SurfaceIndex(mesh_b).distance(va[cand])
    on = d <= tol
    faces_on = np.all(on[fa], axis=1)
    pts = va[on]
    row = {
        "n_zero_verts": int(on.sum()),
        "n_faces_full": int(faces_on.sum()),
        "area_on_mm2": float(mesh_a.area_faces[faces_on].sum() * 1e6),
        "area_share_pct": float(100 * mesh_a.area_faces[faces_on].sum() / mesh_a.area) if mesh_a.area else 0.0,
    }
    if len(pts) >= 4:
        centre = pts.mean(0)
        cov = np.cov((pts - centre).T)
        w, v = np.linalg.eigh(cov)
        w = np.maximum(w, 1e-30)
        normal = v[:, 0]
        off = (pts - centre) @ normal
        row["pca_ratio_2d"] = float(w[1] / w[2])     # ~1 => sheet, <<1 => curve
        row["pca_ratio_1d"] = float(w[0] / w[2])
        row["scatter_um"] = float(np.abs(off).max() * 1e6)
        row["extent_mm"] = float(np.sqrt(w[2]) * 2e3)
        if faces_on.any():
            fn = np.asarray(mesh_a.face_normals)[faces_on]
            nav = fn.mean(0)
            nav /= max(np.linalg.norm(nav), 1e-12)
            cos = fn @ nav
            row["normal_cos_min"] = float(cos.min())
            row["normal_cos_p50"] = float(np.percentile(cos, 50))
    return row, d


def main(assemblies):
    pairs = []
    for line in ROWS.open(encoding="utf-8"):
        r = json.loads(line)
        if r.get("kind") == "pair" and r["asm"] in assemblies and r["dist_mm"] == 0.0:
            pairs.append(r)
    cache = {}
    out = []
    t_all = time.perf_counter()
    for r in pairs:
        key = r["asm"]
        load = lambda p: trimesh.load(GEOM / key / "parts" / f"{p}.ply", process=False)
        ma = cache.setdefault((key, r["a"]), None) or load(r["a"])
        cache[(key, r["a"])] = ma
        mb = cache.setdefault((key, r["b"]), None) or load(r["b"])
        cache[(key, r["b"])] = mb
        t0 = time.perf_counter()
        row, _ = contact_stats(ma, mb)
        row.update({"asm": key, "a": r["a"], "b": r["b"], "s": round(time.perf_counter() - t0, 2)})
        out.append(row)
        print(f"{key} {r['a']}-{r['b']}: verts={row['n_zero_verts']:5d} "
              f"faces={row['n_faces_full']:5d} area={row['area_on_mm2']:.4f}mm2 "
              f"pca2d={row.get('pca_ratio_2d', float('nan')):.3f} "
              f"scatter={row.get('scatter_um', float('nan')):.2f}um "
              f"extent={row.get('extent_mm', float('nan')):.2f}mm "
              f"({row['s']}s)", flush=True)
    Path("runs/eps-prototype/_scratch_contacts.json").write_text(
        json.dumps(out, indent=1), encoding="utf-8")
    print(f"\n{len(out)} pairs in {time.perf_counter()-t_all:.0f}s")
    if out:
        nz = np.array([o["n_zero_verts"] for o in out])
        ar = np.array([o["area_on_mm2"] for o in out])
        print(f"n_zero_verts: min={nz.min()} p50={np.percentile(nz,50):.0f} p90={np.percentile(nz,90):.0f} max={nz.max()}")
        print(f"area_on_mm2 : min={ar.min():.4f} p50={np.percentile(ar,50):.4f} p90={np.percentile(ar,90):.4f} max={ar.max():.3f}")
        print(f"pairs with area>0.01mm2: {int((ar>0.01).sum())} / {len(ar)}")


if __name__ == "__main__":
    asms = sys.argv[1:] or ["asm_000062", "asm_000073", "asm_000007", "asm_000098", "asm_000047"]
    main(asms)
