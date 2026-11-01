"""Scratch: what does a zero-gap interface actually look like?

Read-only exploration. Answers, for one pair:
  1. how many A-vertices sit exactly on B's surface (tol 1e-9 / 1e-6 / 1e-5)?
  2. is the contact patch planar, and what fraction of A's faces are involved?
  3. are the two triangulations identical on the patch (shared vertices) or not?
  4. how thick is the material behind the patch (the wall-thickness gate input)?
"""

import sys
import time
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

GEOM = Path(r"D:\yanyi\rl_assembly\data\assemblies_pre_decimate")


def point_tri_distance(points: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Exact distance from each point to each triangle of `tris` (M,3,3).

    `points` (N,3) vs `tris` (M,3,3) -> (N,M).  Vectorised over both axes.
    Project-to-plane then clamp to the three edges, same as the clearance scan's
    kernel (which is the reference implementation for this corpus).
    """
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

    v0 = ab
    v1 = ac
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

    best = np.full(points.shape[0:1] + (tris.shape[0],), np.inf)
    best = np.where(inside, np.linalg.norm(proj - points[:, None, :], axis=2), best)

    for s0, s1 in ((a, b), (b, c), (c, a)):
        edge = s1 - s0
        e2 = np.einsum("ijk,ijk->ij", edge, edge)
        t = np.einsum("ijk,ijk->ij", points[:, None, :] - s0, edge) / np.where(e2 > 1e-30, e2, 1.0)
        t = np.clip(t, 0.0, 1.0)
        seg = s0 + t[:, :, None] * edge
        dd = np.linalg.norm(seg - points[:, None, :], axis=2)
        best = np.minimum(best, dd)
    return best


def surface_distance(points, mesh, *, k=16, radius_slack=1e-9):
    """Exact min distance point->surface, restricted by a centroid KD-tree bound.

    For each point: candidate triangles = k nearest centroids; expand k until
    `dist(point, k-th centroid) - r_cut > best` (rigorous: a triangle whose
    centroid is farther than best + r_i cannot be the nearest one).
    """
    tris = np.ascontiguousarray(mesh.triangles)
    cen = tris.mean(axis=1)
    rad = np.linalg.norm(tris - cen[:, None, :], axis=2).max(axis=1)
    tree = cKDTree(cen)
    r_cut = float(rad.max())
    out = np.full(len(points), np.inf)
    todo = np.arange(len(points))
    kk = k
    while len(todo):
        take = min(kk, len(cen))
        dist, idx = tree.query(points[todo], k=take)
        dist = np.atleast_2d(dist)
        idx = np.atleast_2d(idx)
        d = np.empty(len(todo))
        for row in range(len(todo)):
            d[row] = point_tri_distance(points[todo[row]][None, :], tris[idx[row]]).min()
        out[todo] = np.minimum(out[todo], d)
        # rigorous stop: farther centroids have lower bound > current best
        far = dist[:, -1]
        stop = (far - r_cut > out[todo]) | (take >= len(cen))
        todo = todo[~stop]
        kk = min(kk * 4, len(cen))
        if take >= len(cen):
            break
    return out


def report_pair(asm, pa, pb, tol_list=(1e-9, 1e-6, 1e-5)):
    ma = trimesh.load(GEOM / asm / "parts" / f"{pa}.ply", process=False)
    mb = trimesh.load(GEOM / asm / "parts" / f"{pb}.ply", process=False)
    print(f"\n=== {asm}  {pa}({len(ma.faces)}f) vs {pb}({len(mb.faces)}f) ===")
    # 1. exact vertex coincidence
    tree_b = cKDTree(mb.vertices)
    dv, _ = tree_b.query(ma.vertices, k=1)
    for tol in tol_list:
        print(f"  A vertices within {tol:g} m of a B vertex: {int((dv <= tol).sum())} / {len(ma.vertices)}")
    tree_a = cKDTree(ma.vertices)
    dva, _ = tree_a.query(mb.vertices, k=1)
    print(f"  B vertices within 1e-9 m of an A vertex: {int((dva <= 1e-9).sum())} / {len(mb.vertices)}")

    # 2. exact distance from every A vertex to B's surface
    t0 = time.perf_counter()
    dsurf = surface_distance(np.asarray(ma.vertices), mb)
    print(f"  A-vertex -> B-surface exact distance: {time.perf_counter()-t0:.1f}s "
          f"zero(<1e-9)={int((dsurf <= 1e-9).sum())} "
          f"<1e-6={int((dsurf <= 1e-6).sum())} <1e-5={int((dsurf <= 1e-5).sum())}")
    verts_on = np.flatnonzero(dsurf <= 1e-6)
    print(f"  patches: n_on_b={len(verts_on)} of {len(ma.vertices)} ({100*len(verts_on)/len(ma.vertices):.1f}%)")

    # 3. faces fully on B (all 3 verts on B) and their normals
    fon = np.all(dsurf[ma.faces] <= 1e-6, axis=1)
    print(f"  faces fully on B: {int(fon.sum())} / {len(ma.faces)} "
          f"area share={ma.area_faces[fon].sum()/ma.area*100:.1f}%")
    if fon.any():
        nrm = ma.face_normals[fon]
        n_avg = nrm.mean(axis=0)
        n_avg = n_avg / max(np.linalg.norm(n_avg), 1e-12)
        cos = nrm @ n_avg
        print(f"  patch mean normal={np.round(n_avg,3)} ; normal spread: "
              f"min cos={cos.min():.4f} p05={np.percentile(cos,5):.4f} p50={np.percentile(cos,50):.4f}")
        # plane fit scatter
        pts = np.asarray(ma.vertices)[verts_on]
        off = (pts - pts.mean(axis=0)) @ n_avg
        print(f"  patch plane scatter: max |off|={np.abs(off).max()*1000:.6f} mm")
    # 4. material depth behind the patch along -normal
    if fon.any():
        n_avg = ma.face_normals[fon].mean(axis=0)
        n_avg = n_avg / np.linalg.norm(n_avg)
        pts = np.asarray(ma.vertices)[verts_on]
        depth = (pts[:, None, :] - np.asarray(ma.vertices)[None, :, :]) @ n_avg
        depth = depth[:, :, None] if depth.ndim == 2 else depth
        # min positive depth > 1e-6
        d2 = np.einsum("ij,ij->i", np.zeros(0, 3), np.zeros(0, 3)) if False else None
        flat = ((pts[:, None, :] - np.asarray(ma.vertices)[None, :, :]) @ n_avg)
        flat = np.where(flat > 1e-6, flat, np.inf)
        per = flat.min(axis=1)
        print(f"  wall depth behind patch (min per vertex, mm): "
              f"min={per.min()*1000:.4f} p05={np.percentile(per[np.isfinite(per)],5)*1000:.4f} "
              f"p50={np.percentile(per[np.isfinite(per)],50)*1000:.4f}")
        print(f"  bbox of parts: A {np.round(ma.bounds,4).tolist()} B {np.round(mb.bounds,4).tolist()}")


if __name__ == "__main__":
    report_pair("asm_000062", "part_000", "part_008")
