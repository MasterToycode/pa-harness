"""Epsilon clearance injection: a **local** normal offset on the contact set.

`docs/pah-dev-doc-v1.md` §4.8 records why this step exists at all:

> 初始穿模被引擎直接拒 | 恰好接触（`y_min=0`）即 `is_valid()=False` | 装配态进入
> 仿真前必须做 `epsilon` 预处理留缝；这一步失败记 `precondition_failed`

and `runs/clearance-scan/report.md` measures the size of the problem: of the 996
part pairs that could be measured, **898 (90.16 %) have a surface-to-surface
minimum distance of exactly 0 m**, spread over **100 / 100** assemblies. The data
is a *partition* of one solid: the pieces were cut out of it, so the cut surfaces
meet exactly. Partition correctness wants the gap to be 0; assembly feasibility
wants it to be > 0. The two requirements cannot both hold, so an epsilon is a
**required pre-processing step for evaluation**, not an optional cleanup.

What this module does **not** do, deliberately:

- **no global erode** and **no Minkowski sum**. A uniform erode would move every
  surface of every part (including the surfaces that carry the connectors),
  change the part's global size, and destroy the gaps the design already has. The
  offset here touches only vertices that are *provably* in contact.
- **no re-triangulation**. Vertices move; faces keep their indices and winding.
  Face count, edge-to-face incidence and Euler characteristic are therefore
  unchanged by construction, which is what makes the before/after watertightness
  comparison meaningful instead of a restatement of "the code kept the faces".
- **no epsilon constant**. `eps_mm` is a *manufacturing parameter* of the print
  process (a clearance a printer can actually produce), passed in by the caller
  and reported with every result. Picking it is an evaluation-config decision
  (§4.8: ε is frozen together with `policy_version`), not something this module
  may hard-code.

## What a "contact" actually is on this corpus (measured, not assumed)

The task description pictures a large coplanar cut face. On the delivered
geometry it is usually a **tangency curve**: the two pieces are independent
reconstructions, and the boolean trim leaves them meeting along the curve where
their surfaces touch — a few hundred vertices whose exact point-to-surface
distance is 0, and almost no face lying fully on the other part
(`runs/eps-prototype/report.md` §2 has the measurement). The offset is therefore
defined on **vertices**, each moving along its own surface normal, rather than on
a fitted plane. That definition degenerates correctly to the coplanar case (all
incident faces coplanar ⇒ the vertex normal *is* the interface normal) and stays
defined on a curve, where a plane fit would not exist.

## The two protections, and what they are for

1. **Wall gate** — a displacement may not exceed `wall_fraction` (default 10 %) of
   the *local* wall thickness at that vertex. Thickness is measured by casting a
   ray from the vertex along the inward normal and taking the first hit on the
   part's own surface (`ray_thickness`), i.e. the material actually behind the
   surface. A vertex that fails keeps its position and is counted `skipped_thin`.
   Moving a vertex further than the material behind it is exactly how a dent
   becomes a hole.
2. **Whole-part revert** — after the offset the part is re-checked: boundary edge
   count, Euler characteristic, `is_watertight` and face count must be unchanged,
   the volume must stay finite and positive, no face may have flipped, and no
   *new* self-intersecting face pair may appear. If any check fails the part is
   restored to its **original** geometry and recorded `rejected`. The task's
   wording is "整件回退到原始", so the revert is whole-part, not per-patch: a part
   that cannot take the offset anywhere takes it nowhere, and the pairs it
   belongs to keep their zero gap.

Both protections are *reported per part*, because a silent skip is
indistinguishable from a successful offset in every downstream number.

## Units

Geometry is **metres** (`docs/internal/adapter-contract.md` §A). The public knob
is `eps_mm` because every report and every manufacturing spec speaks millimetres;
`eps_m` is derived once at the top of `inject_epsilon`. Reported lengths are mm,
areas mm², volumes mm³ or a ratio.

## Purity

`inject_epsilon` does no I/O and mutates no input mesh: it returns new
`trimesh.Trimesh` objects. That is what lets the prototype script write the
result to a scratch directory and re-measure it with the *unmodified*
`scripts/clearance_scan.py` and `scripts/ivr_scan.py` judges rather than with a
second implementation that could disagree with them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Literal, Sequence

import numpy as np
import trimesh
from scipy.spatial import cKDTree

__all__ = [
    "DEFAULT_MAX_THICKNESS_M",
    "DEFAULT_WALL_FRACTION",
    "Injection",
    "PartOutcome",
    "TOUCH_TOL_M",
    "boundary_edges",
    "contact_vertices",
    "dilate_rings",
    "inject_epsilon",
    "mesh_health",
    "point_triangle_distance",
    "ray_thickness",
    "self_intersections",
    "surface_distance",
    "vertex_normals",
]

#: A vertex whose exact distance to the other surface is at most this is "on" it.
#: The clearance scan calls a pair touching at `1e-9 m`, but *vertices* of the two
#: parts are only bit-identical on the intersection curve; a vertex in the middle
#: of a coplanar interface reads a distance of order 1e-17 m through floating
#: point noise. 1e-6 m = 0.001 mm is three orders of magnitude below the thinnest
#: positive gap this corpus contains (0.029 mm, `asm_000098`), so it cannot pull a
#: genuinely separated pair into the contact set.
TOUCH_TOL_M = 1e-6

#: The wall gate's fraction. §4.8 does not name a number; 10 % is the task's and
#: is a standard DfAM rule of thumb, not a law of nature.
DEFAULT_WALL_FRACTION = 0.10

#: How far the thickness ray may search before "no opposite surface within this
#: distance" is reported as "thicker than that". 20 mm is larger than the thickest
#: feature in the corpus (parts are 30-150 mm across) while keeping the candidate
#: ball query small.
DEFAULT_MAX_THICKNESS_M = 0.020


# --------------------------------------------------------------------------- #
# exact point-to-triangle distance (the reference kernel of this corpus)
# --------------------------------------------------------------------------- #
def point_triangle_distance(points: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Exact distance from every point to every triangle.

    `points (N,3)` × `tris (M,3,3)` → `(N,M)`, in the input's units.

    Method: project onto the triangle's plane and, if the projection falls inside
    the triangle, take it; otherwise take the minimum of the three point-to-edge
    distances. That *is* the exact point-to-triangle distance, not a sample — the
    same construction as `scripts/clearance_scan.py:point_triangle_distance`,
    vectorised over both axes instead of one point at a time, because the contact
    search evaluates a few hundred points against a few thousand candidate
    triangles and the scalar version costs ~1 ms per point.

    A degenerate (zero-area) triangle has no plane; its `inf` projection candidate
    is dropped and the edge distances take over, which is correct: a degenerate
    triangle *is* its longest edge.
    """
    points = np.asarray(points, dtype=np.float64)
    tris = np.asarray(tris, dtype=np.float64)
    n_points = len(points)
    n_tris = len(tris)
    if n_points == 0 or n_tris == 0:
        return np.zeros((n_points, n_tris))

    a = tris[None, :, 0, :]
    b = tris[None, :, 1, :]
    c = tris[None, :, 2, :]
    ab = b - a
    ac = c - a
    ap = points[:, None, :] - a
    normal = np.cross(ab, ac)
    n2 = np.einsum("ijk,ijk->ij", normal, normal)
    usable_plane = n2 > 1e-30
    plane_d = np.einsum("ijk,ijk->ij", normal, ap)
    scale = np.where(usable_plane, plane_d / np.where(usable_plane, n2, 1.0), 0.0)
    proj = points[:, None, :] - scale[:, :, None] * normal

    v2 = proj - a
    d00 = np.einsum("ijk,ijk->ij", ab, ab)
    d01 = np.einsum("ijk,ijk->ij", ab, ac)
    d11 = np.einsum("ijk,ijk->ij", ac, ac)
    d20 = np.einsum("ijk,ijk->ij", v2, ab)
    d21 = np.einsum("ijk,ijk->ij", v2, ac)
    den = d00 * d11 - d01 * d01
    usable_den = np.abs(den) > 1e-30
    den_safe = np.where(usable_den, den, 1.0)
    bary_v = (d11 * d20 - d01 * d21) / den_safe
    bary_w = (d00 * d21 - d01 * d20) / den_safe
    inside = (
        usable_plane
        & usable_den
        & (bary_v >= 0.0)
        & (bary_w >= 0.0)
        & (bary_v + bary_w <= 1.0)
    )
    best = np.where(inside, np.linalg.norm(proj - points[:, None, :], axis=2), np.inf)

    for s0, s1 in ((a, b), (b, c), (c, a)):
        edge = s1 - s0
        edge_len2 = np.einsum("ijk,ijk->ij", edge, edge)
        t = np.einsum("ijk,ijk->ij", points[:, None, :] - s0, edge) / np.where(
            edge_len2 > 1e-30, edge_len2, 1.0
        )
        t = np.clip(t, 0.0, 1.0)
        closest = s0 + t[:, :, None] * edge
        best = np.minimum(best, np.linalg.norm(closest - points[:, None, :], axis=2))
    return best


# --------------------------------------------------------------------------- #
# exact point-to-surface distance with a rigorous branch bound
# --------------------------------------------------------------------------- #
class SurfaceIndex:
    """A mesh's triangles indexed for exact "how far is the nearest surface" queries.

    For every triangle take the distance `r_i` from its centroid to its farthest
    vertex. Then for any point `p` and any triangle `i`:

        dist(p, tri_i) >= |p - centroid_i| - r_i

    so once `|p - centroid_(k)| - r_max > best_so_far`, no triangle outside the
    first `k` centroids can beat the answer and the search stops. That is what
    makes the result **exact** rather than "restricted to the k nearest
    centroids": the bound proves the discarded triangles cannot win.

    `r_max` is the global maximum triangle radius — the conservative choice. A
    per-candidate radius would prune earlier, but an error in the bound reports
    "not in contact" for a point that is, and that error is invisible downstream.
    """

    __slots__ = ("tris", "centroids", "radius", "r_max", "tree", "n_faces")

    def __init__(self, mesh: trimesh.Trimesh) -> None:
        self.tris = np.ascontiguousarray(np.asarray(mesh.triangles, dtype=np.float64))
        self.n_faces = len(self.tris)
        if self.n_faces == 0:
            self.centroids = np.zeros((0, 3))
            self.radius = np.zeros(0)
            self.r_max = 0.0
            self.tree = cKDTree(np.zeros((1, 3)))
            return
        self.centroids = self.tris.mean(axis=1)
        self.radius = np.linalg.norm(
            self.tris - self.centroids[:, None, :], axis=2
        ).max(axis=1)
        self.r_max = float(self.radius.max())
        self.tree = cKDTree(self.centroids)

    def nearest(
        self, points: np.ndarray, *, k: int = 32
    ) -> tuple[np.ndarray, np.ndarray]:
        """Exact `(distance, triangle_index)` of the nearest surface patch.

        The triangle index is what carries the *direction*: the outward normal of
        the triangle the point sits on is the true surface normal there, whereas a
        crease-averaged vertex normal is not (see `contact_directions`).
        """
        points = np.asarray(points, dtype=np.float64)
        out = np.full(len(points), np.inf)
        out_index = np.full(len(points), -1, dtype=np.int64)
        if self.n_faces == 0 or len(points) == 0:
            return out, out_index
        pending = np.arange(len(points))
        take = min(k, self.n_faces)
        while len(pending):
            far, index = self.tree.query(points[pending], k=take)
            far = np.atleast_2d(np.asarray(far, dtype=np.float64))
            index = np.atleast_2d(index)
            for row, point_index in enumerate(pending):
                found = point_triangle_distance(
                    points[point_index][None, :], self.tris[index[row]]
                )
                best = int(np.argmin(found))
                if found[best] < out[point_index]:
                    out[point_index] = float(found[best])
                    out_index[point_index] = int(index[row][best])
            if take >= self.n_faces:
                break
            # Outside the first `take` centroids a triangle's lower bound is
            # |p - centroid_take| - r_max; where that exceeds the best, it is done.
            settled = (far[:, -1] - self.r_max > out[pending]) | (take >= self.n_faces)
            pending = pending[~settled]
            take = min(take * 4, self.n_faces)
        return out, out_index

    def distance(self, points: np.ndarray, *, k: int = 32) -> np.ndarray:
        """Exact minimum distance from each point to this surface (metres)."""
        return self.nearest(points, k=k)[0]


def surface_distance(points: np.ndarray, mesh: trimesh.Trimesh) -> np.ndarray:
    """Exact distance from `points` to `mesh`'s surface (metres)."""
    return SurfaceIndex(mesh).distance(points)


# --------------------------------------------------------------------------- #
# which vertices are in contact
# --------------------------------------------------------------------------- #
def _max_edge_length(mesh: trimesh.Trimesh) -> float:
    """Longest edge of the mesh, in metres — the contact candidate-culling radius."""
    edges = np.asarray(mesh.edges_unique)
    if len(edges) == 0:
        return 0.0
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    return float(lengths.max())


def contact_vertices(
    mesh: trimesh.Trimesh,
    other: trimesh.Trimesh,
    *,
    tol_m: float = TOUCH_TOL_M,
    other_index: SurfaceIndex | None = None,
) -> np.ndarray:
    """Indices of `mesh`'s vertices whose exact distance to `other` is ≤ `tol_m`.

    Two channels, cheapest first:

    1. **Vertex coincidence** (`cKDTree` on the other's vertices). On a boolean cut
       the intersection curve is computed from the same input triangles on both
       sides, so the two parts share those vertices *exactly*; this finds most of
       the contact set at C speed.
    2. **Exact surface distance** for the candidates only. If `p` is within
       `tol_m` of the other's surface it lies inside one of the other's
       triangles, and the nearest vertex of *that* triangle is at most its longest
       edge away. Restricting the exact query to vertices within
       `max_edge_length + tol_m` of some vertex of the other is therefore
       **lossless**; on this corpus it shrinks 240 262 candidates to 340.

    The returned set depends only on where the two surfaces are, not on how the
    interface happens to be triangulated — which is what makes a curve-contact
    offset possible at all.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if len(vertices) == 0 or len(other.faces) == 0:
        return np.zeros(0, dtype=np.int64)
    other_vertices = np.asarray(other.vertices, dtype=np.float64)
    nearest, _ = cKDTree(other_vertices).query(vertices, k=1, workers=-1)
    radius = _max_edge_length(other) + tol_m
    candidates = np.flatnonzero(nearest <= radius)
    if len(candidates) == 0:
        return np.zeros(0, dtype=np.int64)
    index = other_index if other_index is not None else SurfaceIndex(other)
    distances = index.distance(vertices[candidates])
    return candidates[distances <= tol_m].astype(np.int64)


def vertex_normals(mesh: trimesh.Trimesh, vertex_index: np.ndarray) -> np.ndarray:
    """Area-weighted unit surface normal at each of `vertex_index`.

    Weighting by triangle area is the standard choice and keeps the normal
    insensitive to how densely a flat region happens to be tessellated. On a
    coplanar interface every incident face contributes the same direction, so this
    reduces to the interface normal; on a tangency curve — which is what this
    corpus actually has — it is the only direction that is defined.

    A vertex with no usable incident face returns a zero vector, and the caller
    must treat a zero normal as "cannot offset here".
    """
    faces = np.asarray(mesh.faces)
    vertex_index = np.asarray(vertex_index, dtype=np.int64)
    if len(faces) == 0 or len(vertex_index) == 0:
        return np.zeros((len(vertex_index), 3))
    weighted = np.asarray(mesh.area_faces, dtype=np.float64)[:, None] * np.asarray(
        mesh.face_normals, dtype=np.float64
    )
    flat = faces.reshape(-1)
    order = np.argsort(flat, kind="stable")
    flat_sorted = flat[order]
    weighted_sorted = weighted.reshape(-1, 3)[order]
    accumulated = np.empty((len(mesh.vertices), 3))
    for axis in range(3):
        accumulated[:, axis] = np.bincount(
            flat_sorted, weights=weighted_sorted[:, axis], minlength=len(mesh.vertices)
        )
    out = accumulated[vertex_index]
    length = np.linalg.norm(out, axis=1)
    good = length > 1e-30
    out[good] /= length[good, None]
    out[~good] = 0.0
    return out


def contact_directions(
    mesh: trimesh.Trimesh,
    other: trimesh.Trimesh,
    contact: np.ndarray,
    *,
    other_index: SurfaceIndex | None = None,
) -> np.ndarray:
    """Unit directions that move `contact` vertices of `mesh` away from `other`.

    The direction is the **outward normal of the triangle of `other` that the
    vertex actually sits on** — the normal of the surface being touched, i.e. the
    literal reading of "沿切面法向". The mesh's own crease-averaged vertex normal
    is *not* used for the direction: at a sharp edge, three faces of very
    different orientation meet, and their area-weighted average can be 60° off the
    interface, which would spend most of a small epsilon sideways instead of
    opening the gap. The own normal is used only for the **sign**, once per
    (part, other) pair and averaged over the whole contact set, because at a
    contact the two surfaces face each other and their outward normals must be
    anti-parallel:

        mean_i dot( n_other_i , n_self_i ) < 0    → the winding agrees, keep the sign
        mean_i dot( n_other_i , n_self_i ) > 0    → the source mesh is wound the
                                                    other way, flip every direction

    Deciding the sign once from the mean, rather than per vertex, is what makes
    this robust: individual contact vertices on a crease are exactly where a
    per-vertex decision would be ambiguous.

    A vertex already inside `other` (which the corpus does not contain — IVR is 0
    everywhere) would be pushed further in by this rule; that case is `other`'s
    penetration to report, not something a clearance offset can fix.
    """
    contact = np.asarray(contact, dtype=np.int64)
    if len(contact) == 0:
        return np.zeros((0, 3))
    index = other_index if other_index is not None else SurfaceIndex(other)
    points = np.asarray(mesh.vertices, dtype=np.float64)[contact]
    _, triangle = index.nearest(points)
    usable = triangle >= 0
    directions = np.zeros((len(contact), 3))
    if not usable.any():
        return directions
    face_normals = np.asarray(other.face_normals, dtype=np.float64)
    directions[usable] = face_normals[triangle[usable]]
    own = vertex_normals(mesh, contact[usable])
    facing = np.einsum("ij,ij->i", directions[usable], own)
    if float(facing.mean()) > 0.0:
        directions[usable] *= -1.0
    return directions


def dilate_rings(mesh: trimesh.Trimesh, vertex_index: np.ndarray, rings: int) -> np.ndarray:
    """`vertex_index` plus every vertex within `rings` edge hops of it.

    Opt-in. The reason it exists: a contact curve can run through the *interior*
    of a triangle whose three vertices are all just outside the tolerance. Such a
    triangle would not move and the surfaces would stay in contact at that spot.
    With one ring of dilation the triangles that contain contact vertices are
    covered far more often than not.
    """
    vertex_index = np.asarray(vertex_index, dtype=np.int64)
    if rings <= 0 or len(vertex_index) == 0:
        return vertex_index
    edges = np.asarray(mesh.edges_unique)
    if len(edges) == 0:
        return vertex_index
    both = np.vstack([edges, edges[:, ::-1]])
    order = np.argsort(both[:, 0], kind="stable")
    neighbour_sorted = both[order, 1]
    source_sorted = both[order, 0]
    seen = np.zeros(len(mesh.vertices), dtype=bool)
    seen[vertex_index] = True
    frontier = np.unique(vertex_index)
    for _ in range(rings):
        if len(frontier) == 0:
            break
        lo = np.searchsorted(source_sorted, frontier)
        hi = np.searchsorted(source_sorted, frontier, side="right")
        counts = np.maximum(hi - lo, 0)
        total = int(counts.sum())
        if total == 0:
            break
        offsets = np.repeat(lo, counts) + (np.arange(total) - np.repeat(
            np.concatenate(([0], np.cumsum(counts)[:-1])), counts
        ))
        gathered = neighbour_sorted[offsets]
        gathered = gathered[~seen[gathered]]
        if len(gathered) == 0:
            break
        gathered = np.unique(gathered)
        seen[gathered] = True
        frontier = gathered
    return np.flatnonzero(seen).astype(np.int64)


# --------------------------------------------------------------------------- #
# local wall thickness: cast the inward normal ray at the part's own surface
# --------------------------------------------------------------------------- #
def ray_thickness(
    mesh: trimesh.Trimesh,
    origins: np.ndarray,
    directions: np.ndarray,
    *,
    max_distance_m: float = DEFAULT_MAX_THICKNESS_M,
    skip_faces: Sequence[np.ndarray] | None = None,
) -> np.ndarray:
    """First hit of each ray `origin + t*direction` on `mesh`'s own surface.

    Returns `t` in metres, `inf` where the ray leaves `max_distance_m` without a
    hit ("thicker than the search radius").

    Rays start *on* the surface, so the caller's own incident triangles would be
    hit at `t ≈ 0`; `skip_faces` names, per ray, the face indices to ignore (the
    vertex's 1-ring), and a `t` floor of 1e-9 m removes the rest of the self-hit
    noise.

    Möller–Trumbore, vectorised over candidate triangles per ray, with candidates
    pruned by a centroid KD-tree ball of radius `max_distance_m + max_triangle_r`.
    `trimesh.ray` is not used: without `embreex` installed (this machine) trimesh
    falls back to a Python loop at ~1 ms per ray, which is 50× slower than the
    vectorised kernel here for a few hundred rays.
    """
    origins = np.asarray(origins, dtype=np.float64)
    directions = np.asarray(directions, dtype=np.float64)
    out = np.full(len(origins), np.inf)
    if len(origins) == 0 or len(mesh.faces) == 0:
        return out
    tris = np.ascontiguousarray(np.asarray(mesh.triangles, dtype=np.float64))
    centroids = tris.mean(axis=1)
    radius = np.linalg.norm(tris - centroids[:, None, :], axis=2).max(axis=1)
    r_max = float(radius.max())
    tree = cKDTree(centroids)

    for row, (origin, direction) in enumerate(zip(origins, directions)):
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-30:
            continue
        unit = direction / norm
        candidates = tree.query_ball_point(origin, max_distance_m + r_max)
        if not candidates:
            continue
        candidates = np.asarray(candidates, dtype=np.int64)
        if skip_faces is not None:
            skip = np.asarray(skip_faces[row], dtype=np.int64)
            if len(skip):
                candidates = candidates[~np.isin(candidates, skip)]
        if len(candidates) == 0:
            continue
        hit = _ray_triangle_distance(origin, unit, tris[candidates], max_distance_m)
        finite = hit[np.isfinite(hit)]
        if len(finite):
            out[row] = float(finite.min())
    return out


def _ray_triangle_distance(
    origin: np.ndarray, unit: np.ndarray, tris: np.ndarray, max_t: float
) -> np.ndarray:
    """Möller–Trumbore hit distance per triangle, `inf` where the ray misses.

    Barycentric containment is inclusive so a ray grazing an edge still counts: a
    missed hit here makes a thin wall look thick and silently disables the gate.
    """
    v0 = tris[:, 0]
    edge1 = tris[:, 1] - v0
    edge2 = tris[:, 2] - v0
    pvec = np.cross(unit[None, :], edge2)
    det = np.einsum("ij,ij->i", edge1, pvec)
    parallel = np.abs(det) < 1e-30
    inv_det = np.where(parallel, 0.0, 1.0 / np.where(parallel, 1.0, det))
    tvec = origin[None, :] - v0
    u = np.einsum("ij,ij->i", tvec, pvec) * inv_det
    qvec = np.cross(tvec, edge1)
    v = np.einsum("j,ij->i", unit, qvec) * inv_det
    t = np.einsum("ij,ij->i", edge2, qvec) * inv_det
    inside = (
        ~parallel
        & (u >= -1e-12)
        & (v >= -1e-12)
        & (u + v <= 1.0 + 1e-12)
        & (t > 1e-9)
        & (t <= max_t)
    )
    return np.where(inside, t, np.inf)


# --------------------------------------------------------------------------- #
# self-intersection: new face pairs after the offset
# --------------------------------------------------------------------------- #
def self_intersections(
    mesh: trimesh.Trimesh,
    faces: np.ndarray | None = None,
    *,
    pad_m: float = 0.0,
) -> tuple[int, list[tuple[int, int]]]:
    """Count intersecting face pairs of `mesh`, among faces involving `faces`.

    `faces` restricts the *first* member of every pair (default: every face). The
    second member is pruned twice, both rigorously:

    1. its AABB must meet the AABB of the whole `faces` set grown by `pad_m`
       (`pad_m` carries the displacement, so a face that moved is still inside);
    2. the two faces' own AABBs must overlap — two triangles can only intersect
       if their bounding boxes do.

    Faces that share a vertex are adjacent by construction and are excluded;
    counting them would report every closed mesh as fully self-intersecting.

    **Coplanar overlap is not detected.** The test is six segment-triangle
    crossings (each triangle's three edges against the other triangle), which is
    complete for non-coplanar pairs and degenerate for coplanar ones. That is a
    stated limitation, not an oversight: the failure mode this guards against is a
    dent piercing the opposite wall, which is transversal.
    """
    tris = np.ascontiguousarray(np.asarray(mesh.triangles, dtype=np.float64))
    face_ids = np.asarray(mesh.faces)
    n_faces = len(tris)
    if n_faces < 2:
        return 0, []
    first = np.arange(n_faces) if faces is None else np.unique(np.asarray(faces, dtype=np.int64))
    first = first[(first >= 0) & (first < n_faces)]
    if len(first) == 0:
        return 0, []

    lo = tris.min(axis=1)
    hi = tris.max(axis=1)
    region_lo = lo[first].min(axis=0) - pad_m
    region_hi = hi[first].max(axis=0) + pad_m
    near = np.flatnonzero(
        np.all(lo <= region_hi, axis=1) & np.all(hi >= region_lo, axis=1)
    )
    if len(near) == 0:
        return 0, []

    # Chunked outer AABB test: (moved face, candidate face) pairs whose boxes meet.
    left: list[np.ndarray] = []
    right: list[np.ndarray] = []
    chunk = max(1, int(4_000_000 // max(len(near), 1)))
    for start in range(0, len(first), chunk):
        block = first[start : start + chunk]
        overlap = np.all(
            (lo[block][:, None, :] <= hi[near][None, :, :])
            & (hi[block][:, None, :] >= lo[near][None, :, :]),
            axis=2,
        )
        rows, columns = np.nonzero(overlap)
        if len(rows):
            left.append(block[rows])
            right.append(near[columns])
    if not left:
        return 0, []
    ia = np.concatenate(left)
    ib = np.concatenate(right)
    keep = ia != ib
    ia, ib = ia[keep], ib[keep]
    if len(ia) == 0:
        return 0, []
    pairs = np.unique(np.stack([ia, ib], axis=1), axis=0)
    ia, ib = pairs[:, 0], pairs[:, 1]

    adjacent = _shares_vertex(face_ids[ia], face_ids[ib])
    ia, ib = ia[~adjacent], ib[~adjacent]
    if len(ia) == 0:
        return 0, []
    hit = _triangles_intersect(tris[ia], tris[ib])
    indices = np.flatnonzero(hit)
    samples = [(int(ia[i]), int(ib[i])) for i in indices[:20]]
    return int(indices.size), samples


def _shares_vertex(fa: np.ndarray, fb: np.ndarray) -> np.ndarray:
    """Whether each face pair shares at least one vertex index."""
    out = np.zeros(len(fa), dtype=bool)
    for column in range(3):
        for other in range(3):
            out |= fa[:, column] == fb[:, other]
    return out


def _triangles_intersect(ta: np.ndarray, tb: np.ndarray) -> np.ndarray:
    """Six segment-triangle crossings per pair; see `self_intersections`."""
    hit = np.zeros(len(ta), dtype=bool)
    for source, target in ((ta, tb), (tb, ta)):
        for corner in range(3):
            origin = source[:, corner]
            direction = source[:, (corner + 1) % 3] - origin
            length = np.linalg.norm(direction, axis=1)
            unit = direction / np.where(length > 1e-30, length, 1.0)[:, None]
            hit |= _segment_hits_triangle(origin, unit, length, target)
    return hit


def _segment_hits_triangle(
    origin: np.ndarray, unit: np.ndarray, length: np.ndarray, tris: np.ndarray
) -> np.ndarray:
    """Per row: does the segment `origin → origin + length*unit` cross `tris[row]`?"""
    v0 = tris[:, 0]
    edge1 = tris[:, 1] - v0
    edge2 = tris[:, 2] - v0
    pvec = np.cross(unit, edge2)
    det = np.einsum("ij,ij->i", edge1, pvec)
    parallel = np.abs(det) < 1e-30
    inv_det = np.where(parallel, 0.0, 1.0 / np.where(parallel, 1.0, det))
    tvec = origin - v0
    u = np.einsum("ij,ij->i", tvec, pvec) * inv_det
    qvec = np.cross(tvec, edge1)
    v = np.einsum("ij,ij->i", unit, qvec) * inv_det
    t = np.einsum("ij,ij->i", edge2, qvec) * inv_det
    tol = 1e-12
    return (
        ~parallel
        & (u >= -tol)
        & (v >= -tol)
        & (u + v <= 1.0 + tol)
        & (t > tol)
        & (t <= length + tol)
    )


# --------------------------------------------------------------------------- #
# mesh health, measured so the before/after comparison is explicit
# --------------------------------------------------------------------------- #
def boundary_edges(mesh: trimesh.Trimesh) -> int:
    """Edges with exactly one incident face. 0 = closed shell.

    `is_watertight` is *not* the gate: this corpus's parts are closed but contain
    0.01-0.1 % non-manifold edges (3+ faces), which makes `is_watertight` False
    for 25 of 26 real parts (`docs/internal/adapter-contract.md` §E6). The
    boundary-edge count is the property that actually distinguishes "closed" from
    "open" here. Both are reported anyway.
    """
    if len(mesh.faces) == 0:
        return -1
    counts = np.bincount(np.asarray(mesh.edges_unique_inverse))
    return int(np.count_nonzero(counts == 1))


def mesh_health(mesh: trimesh.Trimesh) -> dict:
    """The numbers the watertightness row of the report needs."""
    has_faces = len(mesh.faces) > 0
    return {
        "is_watertight": bool(mesh.is_watertight),
        "euler_number": int(mesh.euler_number),
        "boundary_edges": boundary_edges(mesh),
        "n_faces": int(len(mesh.faces)),
        "n_vertices": int(len(mesh.vertices)),
        "volume_m3": float(abs(mesh.volume)) if has_faces else 0.0,
    }


# --------------------------------------------------------------------------- #
# the operation
# --------------------------------------------------------------------------- #
@dataclass
class PartOutcome:
    """Everything that happened to one part, including what did not happen."""

    part: str
    contact_vertices: int = 0
    offset_vertices: int = 0
    skipped_thin: int = 0
    rejected: bool = False
    reject_reason: str = ""
    max_displacement_mm: float = 0.0
    mean_displacement_mm: float = 0.0
    min_displacement_mm: float = 0.0
    volume_before_m3: float = 0.0
    volume_after_m3: float = 0.0
    volume_change_pct: float = 0.0
    health_before: dict = field(default_factory=dict)
    health_after: dict = field(default_factory=dict)
    self_intersections_before: int = 0
    self_intersections_after: int = 0
    flipped_faces: int = 0
    thickness_min_mm: float = math.inf
    thickness_p50_mm: float = math.inf
    pairs: list[str] = field(default_factory=list)

    @property
    def new_self_intersections(self) -> int:
        return max(self.self_intersections_after - self.self_intersections_before, 0)

    def as_row(self) -> dict:
        row = {key: value for key, value in self.__dict__.items()}
        row["new_self_intersections"] = self.new_self_intersections
        row["thickness_min_mm"] = (
            None if not math.isfinite(self.thickness_min_mm) else self.thickness_min_mm
        )
        row["thickness_p50_mm"] = (
            None if not math.isfinite(self.thickness_p50_mm) else self.thickness_p50_mm
        )
        row["pairs"] = list(self.pairs)
        return row


@dataclass
class Injection:
    """The result of one `inject_epsilon` call."""

    eps_mm: float
    mode: str
    wall_fraction: float
    tol_m: float
    dilate: int
    meshes: dict[str, trimesh.Trimesh]
    parts: dict[str, PartOutcome]
    pairs: list[dict]
    moved: dict[str, np.ndarray]

    def summary(self) -> dict:
        parts = list(self.parts.values())
        return {
            "eps_mm": self.eps_mm,
            "mode": self.mode,
            "wall_fraction": self.wall_fraction,
            "tol_m": self.tol_m,
            "dilate": self.dilate,
            "n_pairs": len(self.pairs),
            "n_parts": len(parts),
            "n_parts_offset": sum(1 for p in parts if p.offset_vertices > 0),
            "n_parts_rejected": sum(1 for p in parts if p.rejected),
            "n_contact_vertices": sum(p.contact_vertices for p in parts),
            "n_offset_vertices": sum(p.offset_vertices for p in parts),
            "n_skipped_thin": sum(p.skipped_thin for p in parts),
            "n_flipped_faces": sum(p.flipped_faces for p in parts),
            "n_new_self_intersections": sum(p.new_self_intersections for p in parts),
            "max_displacement_mm": max((p.max_displacement_mm for p in parts), default=0.0),
            "volume_change_pct": _total_volume_change(parts),
        }


def _total_volume_change(parts: list[PartOutcome]) -> float:
    before = sum(p.volume_before_m3 for p in parts)
    after = sum(p.volume_after_m3 for p in parts)
    return 100.0 * (after - before) / before if before > 0 else 0.0


def inject_epsilon(
    parts: dict[str, trimesh.Trimesh],
    pairs: Iterable[tuple[str, str]],
    *,
    eps_mm: float,
    mode: Literal["split", "single"] = "split",
    wall_fraction: float = DEFAULT_WALL_FRACTION,
    tol_m: float = TOUCH_TOL_M,
    dilate: int = 0,
    max_thickness_m: float = DEFAULT_MAX_THICKNESS_M,
    check_self_intersection: bool = True,
) -> Injection:
    """Open `eps_mm` of clearance at every contact between the named part pairs.

    `mode` decides how the epsilon is apportioned:

    - `"split"` — each side recedes `eps/2`. Symmetric, so neither part loses more
      material than the other and the correction is never more than half the
      requested clearance on either side.
    - `"single"` — only the **first** part named in each pair recedes, by the full
      `eps`. One side keeps its original surface, so half as many parts are edited
      and the edited half carries the whole offset. Which of the two is gentler on
      the geometry is exactly what the prototype measures.

    Displacements from several pairs are **accumulated before anything moves**. A
    vertex where three parts meet belongs to the contact set of two pairs and has
    to retreat from both; applying the pairs one after another while re-reading
    the mesh in between would make the result depend on pair order, which would
    make the experiment unreproducible. Every contact set is therefore found on
    the *original* geometry, and each vertex moves once, along the sum of its
    contributions.

    Returns an `Injection` with the new meshes plus per-part and per-pair records.
    Inputs are never mutated.
    """
    if not np.isfinite(eps_mm) or eps_mm < 0.0:
        raise ValueError(f"eps_mm must be a finite value >= 0, got {eps_mm!r}")
    if mode not in ("split", "single"):
        raise ValueError(f"unknown mode {mode!r}; use 'split' or 'single'")
    if not 0.0 < wall_fraction <= 1.0:
        raise ValueError(f"wall_fraction must be in (0, 1], got {wall_fraction!r}")
    if dilate < 0:
        raise ValueError(f"dilate must be >= 0, got {dilate!r}")

    ordered: list[tuple[str, str]] = []
    for entry in pairs:
        a, b = (str(pid) for pid in entry)
        if a == b:
            raise ValueError(f"pair ({a!r}, {b!r}) names the same part twice")
        for pid in (a, b):
            if pid not in parts:
                raise KeyError(f"pair names unknown part {pid!r}")
        ordered.append((a, b))
    eps_m = eps_mm / 1000.0

    indexes = {pid: SurfaceIndex(mesh) for pid, mesh in parts.items()}
    cache: dict[tuple[str, str], np.ndarray] = {}

    def contact_of(pid: str, other: str) -> np.ndarray:
        key = (pid, other)
        if key not in cache:
            cache[key] = contact_vertices(
                parts[pid], parts[other], tol_m=tol_m, other_index=indexes[other]
            )
        return cache[key]

    # ---- phase 1: every contact set, on the original geometry ---------------
    plan: dict[str, list[tuple[np.ndarray, np.ndarray, float]]] = {}
    pair_rows: list[dict] = []
    for a, b in ordered:
        side_a = eps_m if mode == "single" else eps_m / 2.0
        side_b = 0.0 if mode == "single" else eps_m / 2.0
        row = {
            "a": a,
            "b": b,
            "eps_side_mm": [side_a * 1000.0, side_b * 1000.0],
            "a_contact_vertices": 0,
            "b_contact_vertices": 0,
        }
        for pid, other, side, key in (
            (a, b, side_a, "a_contact_vertices"),
            (b, a, side_b, "b_contact_vertices"),
        ):
            contact = contact_of(pid, other)
            row[key] = int(len(contact))
            if side <= 0.0 or len(contact) == 0:
                continue
            directions = contact_directions(
                parts[pid], parts[other], contact, other_index=indexes[other]
            )
            usable = np.linalg.norm(directions, axis=1) > 0.5
            if not usable.any():
                continue
            plan.setdefault(pid, []).append((contact[usable], directions[usable], side))
        pair_rows.append(row)

    # ---- phase 2: dilate, gate, accumulate ---------------------------------
    outcomes: dict[str, PartOutcome] = {}
    new_meshes: dict[str, trimesh.Trimesh] = {}
    moved_masks: dict[str, np.ndarray] = {}

    for pid, mesh in parts.items():
        outcome = PartOutcome(part=pid)
        outcome.volume_before_m3 = float(abs(mesh.volume))
        outcome.health_before = mesh_health(mesh)
        outcome.pairs = sorted(set(_neighbours(ordered, pid)))
        stack = plan.get(pid, [])

        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        displacement = np.zeros_like(vertices)
        moved = np.zeros(len(vertices), dtype=bool)
        thickness_all: list[np.ndarray] = []

        for contact, directions, side in stack:
            if dilate > 0:
                expanded = dilate_rings(mesh, contact, dilate)
                extra = np.setdiff1d(expanded, contact)
                if len(extra):
                    # A dilated vertex has no contact of its own, so it inherits
                    # the direction of the nearest real contact vertex. Solid-name
                    # localisation keeps the dent's wall parallel to itself.
                    _, nearest = cKDTree(vertices[contact]).query(
                        vertices[extra], k=1, workers=-1
                    )
                    expanded_dirs = np.vstack([directions, directions[nearest]])
                    expanded = np.concatenate([contact, extra])
                    order = np.argsort(expanded, kind="stable")
                    expanded = expanded[order]
                    expanded_dirs = expanded_dirs[order]
                else:
                    expanded, expanded_dirs = contact, directions
            else:
                expanded, expanded_dirs = contact, directions
            outcome.contact_vertices += int(len(expanded))
            # Wall gate: displacement <= wall_fraction * local thickness.
            thickness_mm = (
                ray_thickness(
                    mesh,
                    vertices[expanded],
                    -expanded_dirs,
                    max_distance_m=max(max_thickness_m, eps_m * 4.0),
                    skip_faces=_incident_faces(mesh, expanded),
                )
                * 1000.0
            )
            thickness_all.append(thickness_mm)
            keep = thickness_mm * wall_fraction >= side * 1000.0
            outcome.skipped_thin += int(np.count_nonzero(~keep))
            if not keep.any():
                continue
            contribution = np.zeros_like(displacement)
            contribution[expanded[keep]] = expanded_dirs[keep] * side
            displacement += contribution
            moved[expanded[keep]] = True

        outcome.offset_vertices = int(np.count_nonzero(moved))
        if thickness_all:
            finite = np.concatenate(thickness_all)
            finite = finite[np.isfinite(finite)]
            if len(finite):
                outcome.thickness_min_mm = float(finite.min())
                outcome.thickness_p50_mm = float(np.percentile(finite, 50))

        magnitude = np.linalg.norm(displacement, axis=1)
        if outcome.offset_vertices:
            outcome.max_displacement_mm = float(magnitude.max() * 1000.0)
            outcome.mean_displacement_mm = float(magnitude[moved].mean() * 1000.0)
            outcome.min_displacement_mm = float(magnitude[moved].min() * 1000.0)

        if outcome.offset_vertices == 0:
            outcome.volume_after_m3 = outcome.volume_before_m3
            outcome.health_after = dict(outcome.health_before)
            outcomes[pid] = outcome
            new_meshes[pid] = mesh
            moved_masks[pid] = moved
            continue

        candidate = trimesh.Trimesh(
            vertices=vertices + displacement, faces=np.asarray(mesh.faces), process=False
        )
        outcome.volume_after_m3 = float(abs(candidate.volume))
        outcome.health_after = mesh_health(candidate)
        outcome.volume_change_pct = (
            100.0 * (outcome.volume_after_m3 - outcome.volume_before_m3)
            / outcome.volume_before_m3
            if outcome.volume_before_m3 > 0
            else 0.0
        )
        moved_faces = np.flatnonzero(np.any(moved[np.asarray(mesh.faces)], axis=1))
        outcome.flipped_faces = _flipped_faces(mesh, candidate, moved_faces)
        if check_self_intersection:
            pad = max(outcome.max_displacement_mm / 1000.0, 1e-9) + 1e-9
            before_hits, _ = self_intersections(mesh, moved_faces, pad_m=pad)
            after_hits, _ = self_intersections(candidate, moved_faces, pad_m=pad)
            outcome.self_intersections_before = before_hits
            outcome.self_intersections_after = after_hits

        reason = _health_violation(outcome)
        if reason:
            outcome.rejected = True
            outcome.reject_reason = reason
            #: The task's sentence is "整件回退到原始": on any violation the part
            #: goes back to the mesh it arrived with, never to a partial offset.
            outcome.health_after = dict(outcome.health_before)
            outcome.volume_after_m3 = outcome.volume_before_m3
            outcome.volume_change_pct = 0.0
            outcome.max_displacement_mm = 0.0
            outcome.mean_displacement_mm = 0.0
            outcome.min_displacement_mm = 0.0
            outcome.self_intersections_after = outcome.self_intersections_before
            outcome.flipped_faces = 0
            outcome.offset_vertices = 0
            new_meshes[pid] = mesh
            moved_masks[pid] = np.zeros(len(vertices), dtype=bool)
        else:
            new_meshes[pid] = candidate
            moved_masks[pid] = moved
        outcomes[pid] = outcome

    # ---- phase 3: per-pair accounting, from what actually moved ------------
    for row in pair_rows:
        a, b = row["a"], row["b"]
        for pid, other, key in ((a, b, "a_offset_vertices"), (b, a, "b_offset_vertices")):
            contact = cache.get((pid, other))
            moved = moved_masks.get(pid)
            if contact is None or moved is None:
                row[key] = 0
            else:
                row[key] = int(np.count_nonzero(moved[contact]))
        row["a_rejected"] = outcomes[a].rejected
        row["b_rejected"] = outcomes[b].rejected

    return Injection(
        eps_mm=float(eps_mm),
        mode=mode,
        wall_fraction=float(wall_fraction),
        tol_m=float(tol_m),
        dilate=int(dilate),
        meshes=new_meshes,
        parts=outcomes,
        pairs=pair_rows,
        moved=moved_masks,
    )


def _neighbours(ordered: Sequence[tuple[str, str]], pid: str) -> list[str]:
    """The other parts `pid` shares a treated pair with."""
    out = []
    for a, b in ordered:
        if a == pid:
            out.append(b)
        elif b == pid:
            out.append(a)
    return out


def _incident_faces(mesh: trimesh.Trimesh, vertex_index: np.ndarray) -> list[np.ndarray]:
    """Face indices incident to each vertex, in `vertex_index` order."""
    faces = np.asarray(mesh.faces)
    flat = faces.reshape(-1)
    face_of = np.repeat(np.arange(len(faces)), 3)
    order = np.argsort(flat, kind="stable")
    flat_sorted = flat[order]
    face_sorted = face_of[order]
    starts = np.searchsorted(flat_sorted, vertex_index)
    ends = np.searchsorted(flat_sorted, vertex_index, side="right")
    return [face_sorted[start:end] for start, end in zip(starts, ends)]


def _flipped_faces(
    before: trimesh.Trimesh, after: trimesh.Trimesh, faces: np.ndarray
) -> int:
    """How many of `faces` had their winding inverted by the offset.

    A thin wall folded over itself inverts the triangles on the fold, and an
    inverted triangle is the cheapest visible trace of a broken surface that still
    counts as "closed" (same faces, same edges, same Euler number).
    """
    if len(faces) == 0:
        return 0
    before_normals = np.asarray(before.face_normals)[faces]
    after_normals = np.asarray(after.face_normals)[faces]
    dot = np.einsum("ij,ij->i", before_normals, after_normals)
    lengths = np.linalg.norm(after_normals, axis=1)
    return int(np.count_nonzero((dot < 0.0) & (lengths > 0)))


def _health_violation(outcome: PartOutcome) -> str:
    """The first reason this part must be reverted, or "" when it may stand."""
    before, after = outcome.health_before, outcome.health_after
    if after["n_faces"] != before["n_faces"]:
        return f"n_faces {before['n_faces']}->{after['n_faces']}"
    if after["boundary_edges"] != before["boundary_edges"]:
        return f"boundary_edges {before['boundary_edges']}->{after['boundary_edges']}"
    if after["euler_number"] != before["euler_number"]:
        return f"euler_number {before['euler_number']}->{after['euler_number']}"
    if after["is_watertight"] != before["is_watertight"]:
        return f"is_watertight {before['is_watertight']}->{after['is_watertight']}"
    if not math.isfinite(outcome.volume_after_m3) or outcome.volume_after_m3 <= 0.0:
        return "volume_non_positive"
    if outcome.flipped_faces > 0:
        return f"inverted_faces={outcome.flipped_faces}"
    if outcome.new_self_intersections > 0:
        return f"new_self_intersections={outcome.new_self_intersections}"
    return ""
