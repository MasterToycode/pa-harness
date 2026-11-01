"""Scratch: interface structure for zero-gap pairs, with the fast candidate path.

Read-only. For one pair it prints:
  1. exact A-vertex -> B-surface distances (rigorous candidate bound + exact kernel),
  2. the contact patch: face count, area, planarity, normal spread,
  3. the wall depth behind the patch (the gate input),
  4. whether the two triangulations share vertices on the patch.
"""

import sys
import time
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

G = Path(r"D:\yanyi\rl_assembly\data\assemblies_pre_decimate")


def point_tri_distance(points, tris):
    """(N,3) points vs (M,3,3) tris -> (N,M) exact point-triangle distances."""
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


def probe(asm, pa, pb, tol=1e-6):
    ma = trimesh.load(G / asm / "parts" / f"{pa}.ply", process=False)
    mb = trimesh.load(G / asm / "parts" / f"{pb}.ply", process=False)
    print(f"\n=== {asm} {pa}({len(ma.faces)}f/{len(ma.vertices)}v) vs {pb}({len(mb.faces)}f) ===")
    va = np.asarray(ma.vertices, dtype=float)
    vb = np.asarray(mb.vertices, dtype=float)

    # vertex-coincidence channel
    t0 = time.perf_counter()
    dvert, _ = cKDTree(vb).query(va, k=1, workers=-1)
    print(f"  exact A->Bvertex coincidence: zero={int((dvert<=1e-9).sum())} "
          f"<=1e-6={int((dvert<=tol).sum())} ({time.perf_counter()-t0:.1f}s)")

    # rigorous candidate set: p inside a B triangle => nearest B vertex within its longest edge
    e = np.linalg.norm(vb[np.asarray(mb.edges_unique)[:, 0]] - vb[np.asarray(mb.edges_unique)[:, 1]], axis=1)
    R = float(e.max())
    cand = np.flatnonzero(dvert <= R + tol)
    print(f"  B max edge {R*1000:.4f} mm -> candidates {len(cand)} / {len(va)}")

    idx_b = SurfaceIndex(mb)
    t0 = time.perf_counter()
    dsurf = np.full(len(va), np.inf)
    if len(cand):
        dsurf[cand] = idx_b.distance(va[cand])
    print(f"  exact A-vertex->B-surface ({time.perf_counter()-t0:.2f}s): "
          f"zero={int((dsurf<=1e-9).sum())} <=1e-6={int((dsurf<=tol).sum())} "
          f"<=1e-5={int((dsurf<=1e-5).sum())}")

    on = dsurf <= tol
    faces_on = np.all(on[np.asarray(ma.faces)], axis=1)
    print(f"  A faces fully on B: {int(faces_on.sum())} / {len(ma.faces)} "
          f"area={ma.area_faces[faces_on].sum()*1e6:.4f} mm^2 "
          f"({100*ma.area_faces[faces_on].sum()/ma.area:.2f}% of A)")
    if faces_on.any():
        fn = ma.face_normals[faces_on]
        navg = fn.mean(axis=0)
        navg /= np.linalg.norm(navg)
        cos = fn @ navg
        print(f"  patch normal {np.round(navg,4).tolist()} cos spread min={cos.min():.4f} "
              f"p05={np.percentile(cos,5):.4f} p50={np.percentile(cos,50):.4f}")
        pts = va[np.flatnonzero(on)]
        off = (pts - pts.mean(0)) @ navg
        print(f"  patch plane scatter: max|off|={np.abs(off).max()*1e6:.3f} um "
              f"(n_pts={len(pts)})")
        print(f"  patch bbox mm: {np.round(pts.min(0)*1000,3).tolist()} .. {np.round(pts.max(0)*1000,3).tolist()}")
        # wall depth behind patch along -navg
        depth = (pts[:, None, :] - va[None, :, :]) @ navg
        depth = np.where(depth > 1e-6, depth, np.inf)
        per = depth.min(axis=1)
        fin = per[np.isfinite(per)]
        if len(fin):
            print(f"  wall depth behind patch mm: min={per.min()*1000:.4f} "
                  f"p05={np.percentile(fin,5)*1000:.4f} p50={np.percentile(fin,50)*1000:.4f} max={fin.max()*1000:.3f}")
        else:
            print("  wall depth behind patch: no vertex strictly behind the plane")
    # B-side patch size
    dvert_b, _ = cKDTree(va).query(vb, k=1, workers=-1)
    e_a = np.linalg.norm(va[np.asarray(ma.edges_unique)[:, 0]] - va[np.asarray(ma.edges_unique)[:, 1]], axis=1)
    Ra = float(e_a.max())
    cb = np.flatnonzero(dvert_b <= Ra + tol)
    ds_b = np.full(len(vb), np.inf)
    if len(cb):
        ds_b[cb] = SurfaceIndex(ma).distance(vb[cb])
    onb = ds_b <= tol
    faces_on_b = np.all(onb[np.asarray(mb.faces)], axis=1)
    print(f"  B faces fully on A: {int(faces_on_b.sum())} / {len(mb.faces)} "
          f"area={mb.area_faces[faces_on_b].sum()*1e6:.4f} mm^2")
    return dsurf, faces_on


if __name__ == "__main__":
    pairs = [("asm_000062", "part_000", "part_008"), ("asm_000062", "part_001", "part_004")]
    for a, p, q in pairs:
        probe(a, p, q)
