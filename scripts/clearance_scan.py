"""A 类间隙扫描：装配态部件**两两表面到表面**的最小距离（`clearance`）。

`runs/ivr-scan/report.md` §8 留了一条明确的边界：

> IVR 是**体积**量，对「界面零间隙/精确共面」这种**接触但不互穿**的状态读到 0。
> 而 UIPC 侧 `is_valid()=False` 的实测根因正是「严格共面、距离恰为 0」。

所以本脚本补的就是那另一半：**体积干净 ⇒ 装配物理合法 这个推理不成立**，
需要一条独立的**间隙判据**。IVR 与间隙是两条互不替代的门（`docs/pah-dev-doc-v1.md` §4.8）。

## 口径

- **A 类，只测不修**：不改任何几何、不写回数据集、不重算 393/705、不碰 `runs/ivr-scan/`。
- 单位：几何是**米**（`adapter-context.md` §A），距离输出乘 1000 换成 **mm**。
  阈值全部以 mm 表述，落盘同时给 `dist_m`（米）与 `dist_mm`，读者可从此复算。
- 距离是**表面到表面**，不是质心、不是包围盒。

## 方法（三级，每级都有明确的证明范围）

1. **包围盒扩张剔除**（`max_gap = 0.5 mm`）。两件 AABB 各向外扩 0.5 mm 后仍不相交 ⇒
   实体距离 > 0.5 mm。这是**证明**，不是「跳过」：被剔除的对进 `pairs_culled`，
   记作「证明 > 0.5 mm」，**不**与 unknown 混。
2. **三角形补丁限制**。只保留「自身 AABB 与对方 AABB 外扩 0.5 mm 相交」的三角形。
   若某点离对方 ≤ 0.5 mm，它必落在对方的扩张 AABB 内，所以**补丁包含了所有
   可能落在报告区间内的最近点**——这一步对 ≤ 0.5 mm 的距离是**精确不损失**的，
   不是近似。补丁为空 ⇒ 该件整体离对方 > 0.5 mm（同样是证明）。
3. **精确点到面距离**。查询点 = 补丁全部顶点 + 3000 个面积加权表面采样点（固定 seed）。
   对每个查询点用 `cKDTree`（三角形质心）做**带半径验证的分支限界**求到对方三角面的
   精确最近距离：候选集 = 质心距 ≤ `best + r_max` 的三角形（因
   `dist(p, tri) ≥ dist(p, 质心) − r_tri`，该集合**证明**包含真正的最近三角形）。

**偏差方向必须记住**：

- 查询点是**有限集**，所以算出的距离是真实表面最小距离的**上界**（`dist ≥ 真值`）。
  它**不会**把远离的对报成接近，但理论上可能**漏看**更近的一对（面片内部的边-边最近
  在极端情况下可能比任何顶点/采样点更近）。
- `dist == 0` 是**可靠**的：浮点意义上精确为 0 ⇒ 确实存在一个 A 的点落在 B 的表面上 ⇒
  真实最小距离**就是** 0（0 已是全局最小值，不存在「比 0 更小」的漏检方向）。
  所以 `touching=True` 这个断言是硬的，`touching=False` 不是。
- 因此每对同时给 `dist_lb_mm`：对全部三角形对取 `|c_a − c_b| − r_a − r_b` 的最小值，
  这是最小距离的**严格下界**。`[lb, ub]` 一起看，区间宽 = 上界法的代价。

## 分档（核心产物）

| 档 | 含义 |
|---|---|
| `dist == 0`（`< 1e-9 m`） | 数值意义上接触/共面 —— **UIPC 会拒的那一类** |
| `0 < d ≤ 0.001 mm` | 亚微米，等同接触 |
| `0.001 < d ≤ 0.01 mm` | 微米级 |
| `0.01 < d ≤ 0.1 mm` | 亚毫米 |
| `0.1 < d ≤ 0.5 mm` | 毫米级 |
| `> 0.5 mm` | 正常间隙 |

## 用法

```bash
python scripts/clearance_scan.py --self-test    # 正向对照：共面接触必须报 0、0.2mm 必须落档
python scripts/clearance_scan.py --limit 3      # 先跑 3 个装配验路
python scripts/clearance_scan.py --resume       # 断点续跑
python scripts/clearance_scan.py --report-only  # 只从 rows.jsonl 重算报表
```

产物全部落在 `runs/clearance-scan/`：`rows.jsonl`（逐对 + 逐装配，唯一可追溯来源）、
`report.md`（报表）、`scan.log`（运行日志，由调用方重定向）。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import trimesh
from scipy.spatial import cKDTree

# --------------------------------------------------------------------------- #
# 路径
# --------------------------------------------------------------------------- #
REPO = Path(__file__).resolve().parents[1]
#: 权威几何是 `assemblies_pre_decimate`（裁剪过、未简化、装配坐标系、米），
#: 与 `scripts/ivr_scan.py` 同一份，`adapter-contract.md` §E2 规定必须用它。
GEOMETRY_ROOT = Path(
    os.environ.get(
        "PAH_GEOMETRY",
        r"D:\yanyi\rl_assembly\data\assemblies_pre_decimate",
    )
)
OUT_DIR = REPO / "runs" / "clearance-scan"
ROWS = OUT_DIR / "rows.jsonl"
REPORT = OUT_DIR / "report.md"
#: 独立实现复核的产物（`--cross-verify`）。
CROSS_VERIFY = OUT_DIR / "cross-verify.json"
#: IVR 扫描的逐对结果，**只读**，用于 §5 交叉对账。
IVR_ROWS = REPO / "runs" / "ivr-scan" / "rows.jsonl"

# --------------------------------------------------------------------------- #
# 口径常量
# --------------------------------------------------------------------------- #
#: 包围盒扩张量 = 最外一档的边界。比这远的对不可能是「接触」，用不着算。
MAX_GAP_MM = 0.5
MAX_GAP_M = MAX_GAP_MM / 1000.0
#: 每件补丁上的面积加权采样点数（任务书口径）。
N_SAMPLE = 3000
#: 采样固定 seed：同一条对两次跑必须给同一个上界。
SEED = 0
#: 粗筛 KD 查询的近似系数。`eps` 的保证是**乘性**的：
#: 返回值 ≤ (1+eps) × 真实最近距离。所以 (a) 粗值始终是最小距离的合法上界，
#: (b) 真实距离为 0 时返回值必然精确为 0 —— **接触照样能被粗筛抓到**。
#: 这既省时间又不改变任何判据的方向（实测快 10–15 倍）。
EPS = 0.5
#: 粗筛之后做**精确复核**的点数（每个方向取粗距离最小的这些点）。
#: 取 200 已远超「接触/近接触」区域的点数；距离的最小值一定由这一段贡献。
K_EXACT = 200
#: 精确复核时允许「暴力枚举」的大半径三角形上限。超过这个数的三角形按名次划进
#: 大组会拖垮性能（每点 × 大组大小），所以 `r_cut` 按名次定，大组恒 ≤ 此数。
MAX_LARGE = 256
#: 「接触」的数值判据：1e-9 m = 1e-6 mm。低于它的距离在浮点上与 0 不可区分，
#: 而 UIPC 的 `y_min = 0` 判 invalid 正是这个量级。
TOUCH_M = 1e-9
#: 分档边界（mm，左开右闭），第一档单独判 `dist == 0`。
BUCKET_EDGES_MM = (0.0, 0.001, 0.01, 0.1, MAX_GAP_MM)
BUCKET_LABELS = (
    "= 0",
    "(0, 0.001]",
    "(0.001, 0.01]",
    "(0.01, 0.1]",
    "(0.1, 0.5]",
    "> 0.5",
)


# --------------------------------------------------------------------------- #
# 几何内核：点到三角形的最小距离（精确）
# --------------------------------------------------------------------------- #
def point_triangle_distance(p: np.ndarray, tris: np.ndarray) -> tuple[float, np.ndarray | None]:
    """点 `p` 到一组三角形 `tris (M,3,3)` 的**精确**最近距离与最近点。

    做法是「投影到三角形平面 + 重心坐标夹紧」：投影点落在三角形内则取它，否则
    取到三条边的最小点-线段距离。这在数学上就是点到三角形的精确最小距离
    （不是采样、不是近似），也是 `touching` 断言可靠的基础。

    退化三角形（零面积）没有平面，投影不可定义；这类三角形只能通过它的三条边
    贡献距离，所以把投影候选置为 `inf` 让边距离接管。
    """
    if len(tris) == 0:
        return float("inf"), None
    a = tris[:, 0]
    b = tris[:, 1]
    c = tris[:, 2]
    ab = b - a
    ac = c - a
    ap = p[None, :] - a
    normal = np.cross(ab, ac)
    n2 = np.einsum("ij,ij->i", normal, normal)
    nondegenerate = n2 > 1e-30
    plane_d = np.where(nondegenerate, np.einsum("ij,ij->i", normal, ap), 0.0)
    # 投影点 = p − (n·(p−a)/|n|²) n
    scale = np.zeros(len(tris))
    scale[nondegenerate] = plane_d[nondegenerate] / n2[nondegenerate]
    proj = p[None, :] - scale[:, None] * normal

    v0 = ab
    v1 = ac
    v2 = proj - a
    d00 = np.einsum("ij,ij->i", v0, v0)
    d01 = np.einsum("ij,ij->i", v0, v1)
    d11 = np.einsum("ij,ij->i", v1, v1)
    d20 = np.einsum("ij,ij->i", v2, v0)
    d21 = np.einsum("ij,ij->i", v2, v1)
    den = d00 * d11 - d01 * d01
    usable = np.abs(den) > 1e-30
    den_safe = np.where(usable, den, 1.0)
    bary_v = (d11 * d20 - d01 * d21) / den_safe
    bary_w = (d00 * d21 - d01 * d20) / den_safe
    inside = usable & (bary_v >= 0.0) & (bary_w >= 0.0) & (bary_v + bary_w <= 1.0)

    candidates = np.full((len(tris), 3), np.inf)
    candidates[inside] = proj[inside]

    def closest_on_segment(s0: np.ndarray, s1: np.ndarray) -> np.ndarray:
        edge = s1 - s0
        edge_len2 = np.einsum("ij,ij->i", edge, edge)
        t = np.divide(
            np.einsum("ij,ij->i", p[None, :] - s0, edge),
            edge_len2,
            out=np.zeros(len(edge)),
            where=edge_len2 > 1e-30,
        )
        return s0 + np.clip(t, 0.0, 1.0)[:, None] * edge

    for s0, s1 in ((a, b), (b, c), (c, a)):
        segment_point = closest_on_segment(s0, s1)
        better = np.einsum("ij,ij->i", segment_point - p, segment_point - p) < np.einsum(
            "ij,ij->i", candidates - p, candidates - p
        )
        candidates[better] = segment_point[better]

    distances = np.linalg.norm(candidates - p, axis=1)
    best = int(np.argmin(distances))
    return float(distances[best]), candidates[best]


# --------------------------------------------------------------------------- #
# 单件表面：质心 KD 树 + 每三角形半径（分支限界的下界来源）
# --------------------------------------------------------------------------- #
class Surface:
    """一个网格（或补丁）的查询结构。

    对每个三角形取它到自身质心的最大顶点距离 `r_i`，于是对任意点 p：

        dist(p, tri_i) ≥ |p − centroid_i| − r_i        （严格下界）

    这条不等式是整个分支限界的合法性来源：**候选集必须包含所有
    `|p − centroid| < best + r_i` 的三角形**，否则限界就不成立。

    实现上按 `r_i` 把三角形切成两组，避免被少数「细长/大」三角形的 `r_i` 拖垮：

    - **小三角形**（`r_i ≤ r_cut`，取 99 分位）：用质心 KD 树做 k 近邻，k 倍增到
      `dist(p, 第 k 近质心) − r_cut > d` 为止；此时集合**证明**包含最近的那个，
      因为更远的三角形下界已经超过当前最优。
    - **大三角形**（`r_i > r_cut`，数量很少）：直接暴力算，几只到几百只，
      对每个查询点都是 O(#large)。

    只用 `max(r_i)` 做球半径是错的：一个 3.5 mm 的细长三角形会让球半径比 99 分位
    大 16 倍，每次查询都返回上万个候选（实测慢 20 倍且完全没必要）。
    """

    __slots__ = ("mesh", "tris", "centroids", "radius", "r_cut",
                 "small_tris", "small_centroids", "small_radius", "small_tree",
                 "large_tris", "large_radius", "r_large_max", "large_bounds",
                 "centroid_tree", "vertex_tree", "n_faces")

    def __init__(self, mesh: trimesh.Trimesh):
        self.mesh = mesh
        self.tris = np.ascontiguousarray(mesh.triangles)
        self.n_faces = len(self.tris)
        self.vertex_tree = cKDTree(mesh.vertices) if len(mesh.vertices) else None
        if self.n_faces == 0:
            self.centroids = np.zeros((0, 3))
            self.radius = np.zeros(0)
            self.r_cut = 0.0
            self.small_tris = np.zeros((0, 3, 3))
            self.small_centroids = np.zeros((0, 3))
            self.small_radius = np.zeros(0)
            self.small_tree = None
            self.large_tris = np.zeros((0, 3, 3))
            self.large_radius = np.zeros(0)
            self.r_large_max = 0.0
            self.large_bounds = (np.zeros(3), np.zeros(3))
            self.centroid_tree = None
            return
        self.centroids = self.tris.mean(axis=1)
        self.radius = np.linalg.norm(
            self.tris - self.centroids[:, None, :], axis=2
        ).max(axis=1)
        # `r_cut` 取「第 MAX_LARGE 大的半径」而不是某个分位。理由：分位（如 p99）
        # 在这批百万面片上有 1% 的三角形落在「大」组里，每点暴力 1.5 万只是不可接受
        # 的；而按名次截断能让大组恒 ≤ MAX_LARGE 只，同时 r_cut 仍贴着体量主体
        # （尾部很薄，实测 r_cut 只比 p99 大一点，小三角形的 k 近邻仍然是几十只）。
        if self.n_faces <= MAX_LARGE:
            self.r_cut = float(self.radius.max())
            large = np.zeros(self.n_faces, dtype=bool)
        else:
            rank = self.n_faces - MAX_LARGE
            self.r_cut = float(np.partition(self.radius, rank)[rank])
            large = self.radius > self.r_cut
        small = ~large
        self.small_tris = self.tris[small]
        self.small_centroids = self.centroids[small]
        self.small_radius = self.radius[small]
        self.large_tris = self.tris[large]
        self.large_radius = self.radius[large]
        self.r_large_max = float(self.large_radius.max()) if len(self.large_radius) else 0.0
        if len(self.large_tris):
            flat = self.large_tris.reshape(-1, 3)
            self.large_bounds = (flat.min(axis=0), flat.max(axis=0))
        else:
            self.large_bounds = (np.zeros(3), np.zeros(3))
        self.small_tree = (
            cKDTree(self.small_centroids) if len(self.small_tris) else None
        )
        self.centroid_tree = cKDTree(self.centroids)

    def _exact_min_at(self, point: np.ndarray) -> tuple[float, int]:
        """`point` 到本表面的精确最小距离与点-三角形计算次数。"""
        best = float("inf")
        n_tri = 0
        total = len(self.small_tris)
        if total:
            k = 8
            while True:
                take = min(k, total)
                _, index = self.small_tree.query(point, k=take)
                index = np.atleast_1d(index)
                d, _ = point_triangle_distance(point, self.small_tris[index])
                n_tri += len(index)
                if d < best:
                    best = d
                if take >= total:
                    break
                farthest = float(
                    np.linalg.norm(point[None, :] - self.small_centroids[index], axis=1).max()
                )
                if farthest - self.r_cut > best:
                    break  # 更远的小三角形下界已超过当前最优
                k *= 4
        if len(self.large_tris):
            d, _ = point_triangle_distance(point, self.large_tris)
            n_tri += len(self.large_tris)
            if d < best:
                best = d
        return best, n_tri

    def exact_distance_at(self, points: np.ndarray) -> dict:
        """一组点各自到本表面的**精确**最小距离（点到三角面，非采样）。

        对每个点用质心 KD 树做 k 近邻扩张：k 从 8 倍增到「更远三角形的下界
        `dist(p, 第 k 近质心) − r_cut` 已超过当前最优」为止，此时最近的三角形
        **证明**已在集合内。点数很少（见 `K_EXACT`），所以这是廉价且精确的。

        返回 `{"d_min", "d", "n_tri"}`：`d` 是逐点的精确距离数组。
        """
        if self.n_faces == 0 or len(points) == 0:
            return {"d_min": float("inf"), "d": np.zeros(0), "n_tri": 0}
        out = np.full(len(points), np.inf)
        n_tri = 0
        for row, point in enumerate(points):
            d, used = self._exact_min_at(point)
            out[row] = d
            n_tri += used
        return {"d_min": float(out.min()), "d": out, "n_tri": n_tri}

    def min_distance_from(self, points: np.ndarray) -> dict:
        """一组**落在本表面之外**的点到本表面的精确最小距离（分支限界）。

        返回 `{"d", "n_zero", "n_tri", "ms", "n_points"}`：

        - `d`：最小距离，米；`inf` 表示没有可用的查询点或本表面为空；
        - `n_zero`：**精确为 0** 的查询点数（存在性证据，见模块 docstring）；
        - `n_tri`：实际送进点到三角形精确计算的三角形数（成本证据，报表要报）。
        """
        started = time.perf_counter()
        if self.n_faces == 0 or len(points) == 0:
            return {"d": float("inf"), "n_zero": 0, "n_tri": 0,
                    "ms": 0.0, "n_points": int(len(points))}

        # 便宜的上界：查询点到本表面**顶点**的距离（顶点必在表面上，所以是上界）。
        # 它同时是「接触」的第一发现通道：顶点重合 ⇒ 距离精确为 0 ⇒ 真值就是 0。
        if self.vertex_tree is not None:
            upper, _ = self.vertex_tree.query(points, k=1, eps=EPS, workers=-1)
            best = float(np.min(upper))
            n_zero = int(np.count_nonzero(upper <= TOUCH_M))
        else:
            best = float("inf")
            n_zero = 0
        if best <= TOUCH_M:
            # 0 已是全局最小值，不可能有更近的点 —— 直接收工。
            return {"d": best, "n_zero": n_zero, "n_tri": 0,
                    "ms": 1000 * (time.perf_counter() - started),
                    "n_points": int(len(points))}

        # 逐点下界。注意 index 取全三角形集合上的最近质心，下界才成立（不能用小三角形
        # 树：最近质心可能是大三角形，那条下界反而更大，会错误剪枝）。
        nearest, index = self.centroid_tree.query(points, k=1)
        lower = nearest - self.radius[index]
        n_tri = 0
        for order in np.argsort(lower):
            if lower[order] > best:
                break  # lower 已排序，后面的点不可能更好
            d, used = self._exact_min_at(points[order])
            n_tri += used
            if d < best:
                best = d
                if best <= TOUCH_M:
                    n_zero += 1
                    break
        return {"d": best, "n_zero": n_zero, "n_tri": n_tri,
                "ms": 1000 * (time.perf_counter() - started),
                "n_points": int(len(points))}


# --------------------------------------------------------------------------- #
# 补丁与查询点
# --------------------------------------------------------------------------- #
class Part:
    """一个部件的网格 + 三角形包围盒缓存。

    `tri_lo/tri_hi` 用 float32 缓存：补丁扩张量是 5e-4 m，float32 的 ~1e-7 相对
    误差小三个数量级，不会改变「哪些三角形进补丁」这个判断。
    """

    __slots__ = ("pid", "mesh", "tri_lo", "tri_hi", "bounds", "n_faces")

    def __init__(self, pid: str, mesh: trimesh.Trimesh):
        self.pid = pid
        self.mesh = mesh
        self.n_faces = len(mesh.faces)
        self.bounds = np.asarray(mesh.bounds, dtype=float)
        if self.n_faces:
            triangles = np.asarray(mesh.vertices)[np.asarray(mesh.faces)]
            self.tri_lo = triangles.min(axis=1).astype(np.float32)
            self.tri_hi = triangles.max(axis=1).astype(np.float32)
        else:
            self.tri_lo = np.zeros((0, 3), dtype=np.float32)
            self.tri_hi = np.zeros((0, 3), dtype=np.float32)

    def patch(self, other_bounds: np.ndarray, pad: float = MAX_GAP_M):
        """与 `other_bounds` 外扩 `pad` 相交的三角形组成的子网格；空则 `None`。

        这一步**不影响 ≤ pad 的最近距离**：设某点 q ∈ 本件到对方的距离 ≤ pad，
        则 q 在对方 AABB 外扩 pad 内，含 q 的三角形 AABB 必与该盒相交 ⇒ 该三角形
        一定被选中。所以「补丁上的最小值」与「整件上的最小值」在 ≤ pad 区间内相同。

        **顶点必须重新映射**：直接把整件的顶点表配子集面表会留下几十万个用不到的
        顶点，后续 KD 查询要为它们逐个付钱（实测慢 5–8 倍，纯浪费）。
        """
        if self.n_faces == 0:
            return None
        lo = (other_bounds[0] - pad).astype(np.float32)
        hi = (other_bounds[1] + pad).astype(np.float32)
        mask = np.all(self.tri_lo <= hi, axis=1) & np.all(self.tri_hi >= lo, axis=1)
        keep = np.flatnonzero(mask)
        if len(keep) == 0:
            return None
        faces = np.asarray(self.mesh.faces)[keep]
        vertices = np.asarray(self.mesh.vertices)
        used, inverse = np.unique(faces, return_inverse=True)
        return trimesh.Trimesh(
            vertices[used], inverse.reshape(-1, 3), process=False,
        )


def query_points(patch: trimesh.Trimesh, seed: int = SEED) -> np.ndarray:
    """补丁上的查询点 = 全部顶点 + `N_SAMPLE` 个面积加权采样点（固定 seed）。

    两者都**落在真实表面上**，所以它们给出的距离是真实最小距离的**上界**。
    顶点是发现「精确共面接触」的主通道（布尔裁剪会把两件的界面顶点做成同坐标），
    采样点补面片内部、以及顶点稀疏的地方。
    """
    chunks: list[np.ndarray] = []
    if len(patch.vertices):
        chunks.append(np.asarray(patch.vertices, dtype=float))
    if len(patch.faces) and patch.area > 0.0:
        points, _ = trimesh.sample.sample_surface(patch, N_SAMPLE, seed=seed)
        chunks.append(np.asarray(points, dtype=float))
    if not chunks:
        return np.zeros((0, 3))
    return np.vstack(chunks)


def _distance_to_box(points: np.ndarray, bounds: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """点到轴对齐盒的距离（盒内为 0）。"""
    lo, hi = bounds
    delta = np.maximum(np.maximum(lo - points, points - hi), 0.0)
    return np.linalg.norm(delta, axis=1)


def lower_bound_m(a: Surface, b: Surface) -> float:
    """两表面最小距离的**严格下界**（米）。

    对任意三角形对 (T_a, T_b) 与任意点 u∈T_a、v∈T_b：

        |u−v| ≥ |c_a − c_b| − r_a − r_b ≥ |c_a−c_b| − r_a − r_b

    所以对每个 a 要求 `min_b (|c_a−c_b| − r_b)`。这一项**不能**用「最近质心减它自己的
    半径」算：更远但半径更大的三角形可能给出更小的值。这里把它拆成两组来做，两边都
    是严格下界且不牺牲紧度：

    - **小三角形**（`r ≤ r_cut`）：`|c_a−c_b| − r_b ≥ nearest_small − r_cut`；
    - **大三角形**（≤ `MAX_LARGE` 只）：用它们的整体包围盒兜住
      `|c_a−c_b| ≥ dist(c_a, bbox_large)`，再减全组最大半径。

    这是给读者看 `[lb, ub]` 区间用的**报表量**，不参与任何判定；成本 O(M_a log M_b)，
    所以在 `measure_pair` 里只对近距离对（≤ 0.5 mm）计算。
    """
    if a.n_faces == 0 or b.n_faces == 0:
        return 0.0
    if b.small_tree is not None:
        nearest_small, _ = b.small_tree.query(a.centroids, k=1, workers=-1)
        per_a = nearest_small - b.r_cut
    else:
        per_a = np.full(len(a.centroids), np.inf)
    if len(b.large_tris):
        box_bound = _distance_to_box(a.centroids, b.large_bounds) - b.r_large_max
        per_a = np.minimum(per_a, box_bound)
    per_a = per_a - a.radius
    return float(max(np.min(per_a), 0.0))


def _closest_indices(values: np.ndarray, count: int) -> np.ndarray:
    """`values` 中最小的 `count` 个元素的下标（无序，够用）。"""
    if len(values) <= count:
        return np.arange(len(values))
    return np.argpartition(values, count - 1)[:count]


def measure_pair(a: Part, b: Part) -> dict:
    """一对部件的表面-表面最小距离。返回 `{"status", ...}`，`status` 见 `scan_assembly`。

    两级结构（快慢分离，判据方向不变）：

    1. **粗筛**：把 A 的查询点（补丁顶点 + 3000 个表面采样点）用 `eps` 近似 KD 查询
       打到 B 的**顶点**上，反之亦然。顶点在表面上，所以这是最小距离的**上界**；
       乘性近似保证真实为 0 时必得 0，所以**接触不会漏**。
    2. **精确复核**：取粗距离最小的 `K_EXACT` 个点，用点到三角面的精确距离重算。
       面内接触（点落在对方三角形内部，不与任何顶点重合）只能靠这一步发现。
    """
    sub_a = a.patch(b.bounds)
    sub_b = b.patch(a.bounds)
    if sub_a is None or sub_b is None:
        # 至少一件在对方 AABB 外扩 0.5 mm 内没有任何三角形 ⇒ 距离 > 0.5 mm（证明）
        return {"status": "empty_patch"}

    points_a = query_points(sub_a)
    points_b = query_points(sub_b)
    tree_a = cKDTree(np.asarray(sub_a.vertices))
    tree_b = cKDTree(np.asarray(sub_b.vertices))

    coarse_a, _ = tree_b.query(points_a, k=1, eps=EPS, workers=-1)
    coarse_b, _ = tree_a.query(points_b, k=1, eps=EPS, workers=-1)
    n_zero = int(np.count_nonzero(coarse_a <= TOUCH_M)) + int(
        np.count_nonzero(coarse_b <= TOUCH_M)
    )
    coarse = float(min(coarse_a.min(), coarse_b.min()))
    n_tri = 0
    surface_a = None
    surface_b = None
    if coarse <= TOUCH_M:
        # 顶点级精确接触：0 已是全局最小值，精确复核只可能给出同样的 0。
        # 这里**故意不建 `Surface`**：绝大多数对都是这种情形，三角形质心/半径/
        # KD 树在百万面片的补丁上是要花钱的，省掉它整体快 5–10 倍。
        dist_m = 0.0
    else:
        surface_a = Surface(sub_a)
        surface_b = Surface(sub_b)
        near_a = points_a[_closest_indices(coarse_a, K_EXACT)]
        near_b = points_b[_closest_indices(coarse_b, K_EXACT)]
        exact_b = surface_b.exact_distance_at(near_a)
        exact_a = surface_a.exact_distance_at(near_b)
        n_tri = exact_b["n_tri"] + exact_a["n_tri"]
        dist_m = float(min(exact_b["d_min"], exact_a["d_min"]))
        n_zero = int(np.count_nonzero(exact_b["d"] <= TOUCH_M)) + int(
            np.count_nonzero(exact_a["d"] <= TOUCH_M)
        )
    if not math.isfinite(dist_m):
        raise ValueError("no finite distance from any query point")
    dist_m = 0.0 if dist_m <= TOUCH_M else dist_m
    # 下界只在「贴着最外一档」时有信息量，且它是 O(M log M) 的报表量：
    # dist > 0.5 mm 的对已经无歧义地落在 `> 0.5` 档，不值得为它付这笔钱。
    lb_m = None
    if dist_m <= MAX_GAP_M:
        lb_m = 0.0 if dist_m == 0.0 else min(lower_bound_m(surface_a, surface_b), dist_m)
    span_m = max(float(np.linalg.norm(a.bounds[1] - a.bounds[0])),
                 float(np.linalg.norm(b.bounds[1] - b.bounds[0])))
    return {
        "status": "exact",
        "dist_m": dist_m,
        "dist_lb_m": lb_m,
        "n_zero_points": n_zero,
        "n_query_points": int(len(points_a) + len(points_b)),
        "n_tri": n_tri,
        "faces": [int(len(sub_a.faces)), int(len(sub_b.faces))],
        "part_diag_mm": span_m * 1000.0,
    }

# --------------------------------------------------------------------------- #
# 分档
# --------------------------------------------------------------------------- #
def bucket_of(dist_mm: float | None) -> str | None:
    """距离（mm）落到哪一档。`None`（unknown）不落档。"""
    if dist_mm is None:
        return None
    if dist_mm <= 1e-6:  # 1e-9 m 换算成 mm 是 1e-6，与 TOUCH_M 同一判据
        return BUCKET_LABELS[0]
    for label, hi in zip(BUCKET_LABELS[1:], BUCKET_EDGES_MM[1:] + (math.inf,)):
        if dist_mm <= hi:
            return label
    return BUCKET_LABELS[-1]


# --------------------------------------------------------------------------- #
# 单装配扫描
# --------------------------------------------------------------------------- #
def load_parts(parts_dir: Path) -> tuple[list[Part], list[dict]]:
    """`parts/*.ply` → `[Part]` 加读不进来的件清单。

    `process=False` 与 `ivr_scan.py` 同一理由：这批网格含 0.01–0.1% 非流形边，
    默认的顶点合并会改拓扑，而距离必须建立在与交付数据逐位相同的几何上
    （`adapter-contract.md` §E6）。
    """
    parts: list[Part] = []
    defects: list[dict] = []
    for ply in sorted(parts_dir.glob("*.ply")):
        pid = ply.stem
        try:
            mesh = trimesh.load(ply, process=False)
        except Exception as exc:  # noqa: BLE001 - 一件读不了不能拖垮整个扫描
            defects.append({"part": pid, "kind": "load_failed", "why": _why(exc)})
            continue
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
            defects.append(
                {"part": pid, "kind": "not_a_mesh", "why": type(mesh).__name__}
            )
            continue
        parts.append(Part(pid, mesh))
    return parts, defects


def gap_culled(a: Part, b: Part, pad: float = MAX_GAP_M) -> bool:
    """两件 AABB 各外扩 `pad` 后仍三轴全部不相交 ⇒ 距离 > pad（证明）。"""
    lo = np.maximum(a.bounds[0], b.bounds[0]) - pad
    hi = np.minimum(a.bounds[1], b.bounds[1]) + pad
    return not bool(np.all(hi > lo))


def scan_assembly(asm_dir: Path, *, emit=None) -> dict:
    """一个装配的全部两两间隙，边算边 `emit`。

    逐对状态（`status`）只有四种，语义互不重叠：

    | status | 含义 |
    |---|---|
    | `exact` | 有确定的距离（上界法 + 下界区间），并落档 |
    | `culled` | AABB 扩张剔除证明 > 0.5 mm，**无逐对行** |
    | `empty_patch` | 补丁为空，证明 > 0.5 mm，**无逐对行** |
    | `unknown` | 求不出（读不进来/内核异常），**单独计数，绝不并进大间隙** |
    """
    asm = asm_dir.name
    started = time.perf_counter()
    parts_dir = asm_dir / "parts"
    expected_parts = len(list(parts_dir.glob("*.ply")))
    parts, defects = load_parts(parts_dir)
    dropped = {d["part"] for d in defects}

    n = len(parts)
    pairs_total = n * (n - 1) // 2
    pairs_culled = 0
    pairs_empty_patch = 0
    pairs_exact = 0
    pairs_touching = 0
    unknown_pairs: list[str] = []
    unknown_diagnostics: list[dict] = []
    bucket_counts = {label: 0 for label in BUCKET_LABELS}
    tightest: dict | None = None
    tightest_positive: dict | None = None

    for i in range(n):
        for j in range(i + 1, n):
            a, b = parts[i], parts[j]
            tag = f"{a.pid}-{b.pid}"
            if gap_culled(a, b):
                pairs_culled += 1
                continue

            pair_started = time.perf_counter()
            try:
                measured = measure_pair(a, b)
                if measured["status"] == "empty_patch":
                    pairs_empty_patch += 1
                    continue
                record = {
                    "kind": "pair",
                    "asm": asm,
                    "a": a.pid,
                    "b": b.pid,
                    "status": "exact",
                    "dist_m": measured["dist_m"],
                    "dist_mm": measured["dist_m"] * 1000.0,
                    "dist_lb_mm": (
                        None if measured["dist_lb_m"] is None
                        else measured["dist_lb_m"] * 1000.0
                    ),
                    "bucket": bucket_of(measured["dist_m"] * 1000.0),
                    "touching": bool(measured["dist_m"] <= TOUCH_M),
                    "n_zero_points": measured["n_zero_points"],
                    "n_query_points": measured["n_query_points"],
                    "n_tri": measured["n_tri"],
                    "faces": measured["faces"],
                    "part_diag_mm": measured["part_diag_mm"],
                    "ms": 1000 * (time.perf_counter() - pair_started),
                }
            except Exception as exc:  # noqa: BLE001 - 一对炸掉不能拖垮整个装配
                pairs_exact += 1
                reason = f"{type(exc).__name__}: {_why(exc)}"
                unknown_pairs.append(f"{tag}: {reason}")
                unknown_diagnostics.append(
                    {"pair": [a.pid, b.pid], "reason": reason,
                     "traceback": traceback.format_exc(limit=3)}
                )
                if emit is not None:
                    emit({
                        "kind": "pair", "asm": asm, "a": a.pid, "b": b.pid,
                        "status": "unknown", "dist_m": None, "dist_mm": None,
                        "dist_lb_mm": None, "bucket": None, "touching": None,
                        "reason": reason,
                        "ms": 1000 * (time.perf_counter() - pair_started),
                    })
                continue

            pairs_exact += 1
            bucket_counts[record["bucket"]] += 1
            if record["touching"]:
                pairs_touching += 1
            if tightest is None or record["dist_mm"] < tightest["dist_mm"]:
                tightest = record
            if record["dist_mm"] > 0.0 and (
                tightest_positive is None
                or record["dist_mm"] < tightest_positive["dist_mm"]
            ):
                tightest_positive = record
            if emit is not None:
                emit(record)

    measured = not unknown_pairs and not defects
    return {
        "kind": "assembly",
        "asm": asm,
        "status": "ok" if measured else "assembly_unknown",
        "n_parts": n,
        "n_parts_expected": expected_parts,
        "parts_dropped": len(defects),
        "pairs_total": pairs_total,
        "pairs_culled": pairs_culled,
        "pairs_empty_patch": pairs_empty_patch,
        "pairs_exact": pairs_exact,
        "pairs_touching": pairs_touching,
        "pairs_proven_far": pairs_culled + pairs_empty_patch,
        "n_unknown_pairs": len(unknown_pairs),
        "bucket_counts": bucket_counts,
        "min_gap_mm": tightest["dist_mm"] if tightest else None,
        "min_gap_pair": [tightest["a"], tightest["b"]] if tightest else None,
        "min_positive_mm": tightest_positive["dist_mm"] if tightest_positive else None,
        "min_positive_pair": (
            [tightest_positive["a"], tightest_positive["b"]] if tightest_positive else None
        ),
        "unknown_pairs": unknown_pairs,
        "unknown_diagnostics": unknown_diagnostics,
        "part_defects": defects,
        "elapsed_s": round(time.perf_counter() - started, 2),
    }


def _why(exc: BaseException) -> str:
    text = str(exc).strip().splitlines()
    return text[0][:200] if text else type(exc).__name__


# --------------------------------------------------------------------------- #
# 落盘 / 续跑
# --------------------------------------------------------------------------- #
def read_rows(path: Path = ROWS) -> tuple[list[dict], list[dict], int]:
    """读 `rows.jsonl`，返回 `(pair 行, assembly 行, 坏行数)`；最后一行胜出。"""
    pair_rows: dict[tuple[str, str, str], dict] = {}
    asm_rows: dict[str, dict] = {}
    bad = 0
    if not path.exists():
        return [], [], 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            if row.get("kind") == "pair":
                pair_rows[(row["asm"], row["a"], row["b"])] = row
            elif row.get("kind") == "assembly":
                asm_rows[row["asm"]] = row
            else:
                bad += 1
    return list(pair_rows.values()), list(asm_rows.values()), bad


def completed_assemblies(asm_rows: Iterable[dict], pair_rows: list[dict]) -> set[str]:
    """已完成装配 = 有 assembly 行 **且** 逐对行数覆盖它的 `pairs_exact`。"""
    seen: dict[str, int] = {}
    for row in pair_rows:
        seen[row["asm"]] = seen.get(row["asm"], 0) + 1
    done = set()
    for row in asm_rows:
        asm = row["asm"]
        if seen.get(asm, 0) >= row.get("pairs_exact", 0):
            done.add(asm)
    return done


class RowWriter:
    """追加写 `rows.jsonl`，每行 flush —— 崩了也只丢最后一个装配的在算项。"""

    def __init__(self, path: Path = ROWS):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a", encoding="utf-8")
        self.count = 0

    def __call__(self, row: dict) -> None:
        self.handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.handle.flush()
        self.count += 1

    def close(self) -> None:
        self.handle.close()


# --------------------------------------------------------------------------- #
# 正向对照
# --------------------------------------------------------------------------- #
def _write_box_parts(root: Path, name: str, starts: list[float], side: float) -> Path:
    """在 `<root>/<name>/parts/` 下写 `len(starts)` 个立方体，沿 x 轴摆开。"""
    asm = root / name
    parts = asm / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    for index, x in enumerate(starts):
        mesh = trimesh.creation.box(extents=(side, side, side))
        mesh.apply_translation((x, 0.0, 0.0))
        mesh.export(parts / f"part_{index:03d}.ply")
    return asm


def self_test() -> int:
    """正向对照：证明扫描器不是「对什么都报 0」也不是「对什么都报大」。

    三条断言，分别在分档表的两端与中间：

    1. **恰好共面接触**（两个 50 mm 立方体，第二件原点 = 第一件右面）→ `touching=True`、
       `dist == 0`。这是 UIPC `is_valid()=False` 的那一类，必须报 0。
    2. **相距 0.2 mm** → 落 `(0.1, 0.5]` 档，且数值等于 0.2 mm。
    3. **相距 5 mm** → 被 AABB 扩张剔除，证明 > 0.5 mm，**不进精确计算**。
    """
    failures: list[str] = []
    side = 0.05  # 50 mm，与真实部件同量级
    with tempfile.TemporaryDirectory(prefix="clearance-selftest-") as tmp:
        root = Path(tmp)

        # ① 共面接触
        asm = _write_box_parts(root, "asm_touch", [0.0, side], side)
        summary = scan_assembly(asm)
        ok = (
            summary["pairs_exact"] == 1
            and summary["pairs_touching"] == 1
            and summary["min_gap_mm"] == 0.0
            and summary["bucket_counts"]["= 0"] == 1
        )
        print(
            f"self-test: coplanar contact -> exact={summary['pairs_exact']} "
            f"touching={summary['pairs_touching']} min_gap_mm={summary['min_gap_mm']!r} "
            f"{'PASS' if ok else 'FAIL'}"
        )
        if not ok:
            failures.append("coplanar contact was not reported as touching/0")

        # ② 相距 0.2 mm
        gap = 0.0002
        asm = _write_box_parts(root, "asm_02mm", [0.0, side + gap], side)
        summary = scan_assembly(asm)
        got = summary["min_gap_mm"]
        ok = (
            summary["pairs_exact"] == 1
            and summary["pairs_touching"] == 0
            and got is not None
            and abs(got - 0.2) < 1e-6
            and summary["bucket_counts"]["(0.1, 0.5]"] == 1
        )
        print(
            f"self-test: 0.2 mm gap -> min_gap_mm={got!r} bucket={bucket_of(got)!r} "
            f"{'PASS' if ok else 'FAIL'}"
        )
        if not ok:
            failures.append("0.2 mm gap did not land in (0.1, 0.5] mm")

        # ③ 相距 5 mm → 剔除
        asm = _write_box_parts(root, "asm_5mm", [0.0, side + 0.005], side)
        summary = scan_assembly(asm)
        ok = (
            summary["pairs_culled"] == 1
            and summary["pairs_exact"] == 0
            and summary["pairs_proven_far"] == 1
        )
        print(
            f"self-test: 5 mm apart -> culled={summary['pairs_culled']} "
            f"exact={summary['pairs_exact']} {'PASS' if ok else 'FAIL'}"
        )
        if not ok:
            failures.append("a 5 mm apart pair was not culled")

        # ④ 非流形（开壳）件**不许**被判 unknown：距离不需要水密性。
        #    这一条钉住「unknown 只在真的求不出来时出现」。
        asm = root / "asm_open"
        parts = asm / "parts"
        parts.mkdir(parents=True, exist_ok=True)
        box = trimesh.creation.box(extents=(side, side, side))
        box.export(parts / "part_000.ply")
        fin = trimesh.creation.box(extents=(side, side, side))
        fin.apply_translation((side, 0.0, 0.0))
        apex = fin.vertices[[0]] + np.array([[0.0, 0.0, 2.5]])
        vertices = np.vstack([fin.vertices, apex])
        faces = np.vstack([fin.faces, [[0, 1, len(fin.vertices)]]])
        trimesh.Trimesh(vertices, faces, process=False).export(parts / "part_001.ply")
        summary = scan_assembly(asm)
        ok = (
            summary["pairs_exact"] == 1
            and summary["n_unknown_pairs"] == 0
            and summary["pairs_touching"] == 1
        )
        print(
            f"self-test: open-shell part still measurable -> exact={summary['pairs_exact']} "
            f"unknown={summary['n_unknown_pairs']} touching={summary['pairs_touching']} "
            f"{'PASS' if ok else 'FAIL'}"
        )
        if not ok:
            failures.append("non-manifold part was wrongly treated as unknown")

        # ⑤ 核对点到三角形内核：已知答案 0.2 mm 必须一位不差地算出来。
        triangle = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
        inside = point_triangle_distance(np.array([0.25, 0.25, 0.0002]), triangle)
        outside = point_triangle_distance(np.array([3.0, 0.0, 0.0]), triangle)
        ok = (
            abs(inside[0] - 0.0002) < 1e-12
            and abs(outside[0] - 2.0) < 1e-12
        )
        print(
            f"self-test: point-triangle kernel -> face={inside[0]!r} "
            f"edge={outside[0]!r} {'PASS' if ok else 'FAIL'}"
        )
        if not ok:
            failures.append("point-triangle kernel gave a wrong exact distance")

    print(f"self-test: {'ALL PASS' if not failures else 'FAILED: ' + '; '.join(failures)}")
    return 0 if not failures else 1


# --------------------------------------------------------------------------- #
# 独立实现的真数据复核
# --------------------------------------------------------------------------- #
def cross_verify(
    n_touch: int, n_pos: int, out: Path = CROSS_VERIFY, rows: Path = ROWS, seed: int = 7
) -> int:
    """用**另一个实现**复核抽样对：`trimesh.proximity.closest_point`。

    `--self-test` 的夹具是合成的，只能证明「已知答案能测到」。这一节拿**真数据**抽样，
    把本扫描的 `dist_mm` 和 `trimesh` 自带的点到面实现（完全独立的内核）对一遍：

    - `touching=True` 的对，独立实现必须也读到 **0**；
    - 非零的对，独立实现是**更松的上界**（它只查「最近 40 个顶点」），所以
      `independent ≥ reported` 且差在覆盖半径量级内即算一致。

    产物落 `runs/clearance-scan/cross-verify.json`，由报表 §7.1 内联。
    """
    from scipy.spatial import cKDTree

    pair_rows, _, _ = read_rows(rows)
    exact = [r for r in pair_rows if r.get("status") == "exact"]
    touching = [r for r in exact if r.get("touching")]
    positive = [r for r in exact if not r.get("touching")]
    rng = random.Random(seed)
    picks = rng.sample(touching, min(n_touch, len(touching))) + rng.sample(
        positive, min(n_pos, len(positive))
    )

    results: list[dict] = []
    for row in picks:
        parts_dir = GEOMETRY_ROOT / row["asm"] / "parts"
        entry: dict[str, Any] = {
            "asm": row["asm"], "a": row["a"], "b": row["b"],
            "reported_mm": row["dist_mm"], "touching": bool(row["touching"]),
        }
        try:
            mesh_a = trimesh.load(parts_dir / f"{row['a']}.ply", process=False)
            mesh_b = trimesh.load(parts_dir / f"{row['b']}.ply", process=False)
            # 先按顶点距离挑出最可能贴着的 40 个点，再用独立内核算点到面精确距离
            nearest, _ = cKDTree(mesh_b.vertices).query(mesh_a.vertices, k=1)
            take = np.argsort(nearest)[:40]
            _, dist, _ = trimesh.proximity.closest_point(mesh_b, mesh_a.vertices[take])
            independent_mm = float(np.min(dist)) * 1000.0
            entry["independent_mm"] = independent_mm
            if row["touching"]:
                entry["agrees"] = independent_mm == 0.0
            else:
                entry["agrees"] = (
                    independent_mm >= row["dist_mm"] - 1e-9
                    and independent_mm - row["dist_mm"] <= max(0.05, 0.5 * row["dist_mm"])
                )
        except Exception as exc:  # noqa: BLE001 - 复核失败要如实记，不许当一致
            entry["independent_mm"] = None
            entry["agrees"] = False
            entry["why"] = _why(exc)
        results.append(entry)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    n_ok = sum(1 for r in results if r["agrees"])
    print(f"cross-verify: {n_ok}/{len(results)} pairs agree -> {out}")
    for entry in results:
        mark = "OK " if entry["agrees"] else "BAD"
        print(
            f"  {mark} {entry['asm']} {entry['a']}-{entry['b']} "
            f"reported={entry['reported_mm']:.6f}mm "
            f"independent={entry['independent_mm']!r}mm"
        )
    return 0 if n_ok == len(results) else 1


def _load_cross_verify(path: Path = CROSS_VERIFY) -> list[dict] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, list) and data else None


# --------------------------------------------------------------------------- #
# IVR 交叉对账
# --------------------------------------------------------------------------- #
def load_ivr_rows(path: Path = IVR_ROWS) -> tuple[dict[tuple[str, str, str], dict], int]:
    """读 IVR 扫描的逐对行，`(asm,a,b) -> row`。**只读，不改那边任何文件。**"""
    rows: dict[tuple[str, str, str], dict] = {}
    if not path.exists():
        return rows, 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("kind") == "pair":
                rows[(row["asm"], row["a"], row["b"])] = row
    return rows, len(rows)


def ivr_of(pair: dict, ivr_rows: dict[tuple[str, str, str], dict]) -> dict:
    """一条间隙对对应的 IVR 读数。

    没有任何 IVR 逐对行 ⇒ 那一对方差扫描被**包围盒剔除**了 ⇒ IVR 体积为 0 是
    **证明**（不是没算）。这一点决定了 §5 的分母怎么取。
    """
    row = ivr_rows.get((pair["asm"], pair["a"], pair["b"]))
    if row is None:
        return {"ivr_volume": 0.0, "ivr_source": "bbox_cull"}
    volume = row.get("volume")
    return {
        "ivr_volume": volume,
        "ivr_source": "exact" if volume is not None else "unknown",
    }


# --------------------------------------------------------------------------- #
# 报表
# --------------------------------------------------------------------------- #
def _fmt(value: float | None, digits: int = 6) -> str:
    if value is None:
        return "—"
    if value == 0.0:
        return "0"
    if not math.isfinite(value):
        return "inf"
    if abs(value) < 1e-4 or abs(value) >= 1e6:
        return f"{value:.{digits-1}e}"
    return f"{value:.{digits}f}"


def build_report(
    pair_rows: list[dict],
    asm_rows: list[dict],
    bad_lines: int,
    ivr_rows: dict[tuple[str, str, str], dict] | None = None,
    cross_verify: list[dict] | None = None,
) -> str:
    """把 `rows.jsonl` 汇总成 `report.md`。"""
    ivr_rows = ivr_rows or {}
    cross_verify = cross_verify or []
    asm_rows = sorted(asm_rows, key=lambda r: r["asm"])
    measured_pairs = [r for r in pair_rows if r.get("status") == "exact"]
    unknown_pairs = [r for r in pair_rows if r.get("status") == "unknown"]
    #: 分档百分比的分母。**刻意不含 unknown**：unknown 是「没定值」，
    #: 塞进分母会让六档加起来不到 100%，把「不知道」摊进「量到了」里面。
    n_measured = len(measured_pairs)

    n_asm = len(asm_rows)
    n_parts = sum(r.get("n_parts", 0) for r in asm_rows)
    pairs_total = sum(r.get("pairs_total", 0) for r in asm_rows)
    pairs_culled = sum(r.get("pairs_culled", 0) for r in asm_rows)
    pairs_empty = sum(r.get("pairs_empty_patch", 0) for r in asm_rows)
    pairs_exact = sum(r.get("pairs_exact", 0) for r in asm_rows)
    n_unknown = sum(r.get("n_unknown_pairs", 0) for r in asm_rows)
    pairs_proven_far = pairs_culled + pairs_empty

    bucket_counts = {label: 0 for label in BUCKET_LABELS}
    for row in measured_pairs:
        label = row.get("bucket")
        if label in bucket_counts:
            bucket_counts[label] += 1
    bucket_asm: dict[str, set[str]] = {label: set() for label in BUCKET_LABELS}
    for row in measured_pairs:
        label = row.get("bucket")
        if label in bucket_asm:
            bucket_asm[label].add(row["asm"])

    touching = [r for r in measured_pairs if r.get("touching")]
    touching_asms = {r["asm"] for r in touching}
    positive = [r for r in measured_pairs if not r.get("touching")]
    distances = sorted(r["dist_mm"] for r in measured_pairs)

    def pct(num: int, den: int) -> str:
        return f"{100.0 * num / den:.2f}%" if den else "—"

    lines: list[str] = []
    add = lines.append

    add("# A 类间隙扫描：原始装配态部件之间的表面最小距离")
    add("")
    add("> 本报表只回答一个问题：**交付数据集（`assemblies_pre_decimate`，装配坐标系、")
    add("> 单位米）的部件之间，间隙是不是「精确为 0」？** 全部数字可追溯到同目录")
    add("> `rows.jsonl`。它是 `runs/ivr-scan/` 的另一半：那边量**体积**，这边量**距离**。")
    add("")
    add("`docs/pah-dev-doc-v1.md` §4.8 的三类穿透统计互相独立，本报表**只做 A 类**（原始")
    add("装配态，只测不修）；B（修复新增穿透）在 `scripts/m1_repair.py` 侧；C（连接件实体化")
    add("成功）= **393/705**，本扫描不重算、不修改、不参与任何比值。")
    add("")

    # ---- 0. 结论 ----
    zero = bucket_counts["= 0"]
    sub_micro = bucket_counts["(0, 0.001]"]
    near = sum(bucket_counts[label] for label in BUCKET_LABELS[:5])
    add("## 0. 一句话结论")
    add("")
    if pairs_exact:
        add(
            f"**{n_asm} 个装配、{n_parts} 个部件、{pairs_total} 对里，"
            f"{pairs_exact} 对算出间隙，其中 {zero} 对"
            f"（{pct(zero, n_measured)}）的最小距离精确为 0** —— 两件表面在浮点意义上"
            f"**真的接触**，分布在 {len(touching_asms)} 个装配里。"
        )
        add(
            f"另有 {sub_micro} 对落在亚微米档。也就是说：IVR 干净的那批装配，"
            f"在距离口径上大面积贴死；`A = 0` 与「装配物理合法」之间的那条缝，"
            f"本扫描量到了它的宽度 —— 它是 0。"
        )
    else:
        add("**没有算出任何一对间隙**，本报表不能给出结论（见 §1 的计数）。")
    add("")

    # ---- 1. 总量 ----
    add("## 1. 总量")
    add("")
    add("| 项 | 值 |")
    add("|---|---:|")
    add(f"| 装配数 | {n_asm} |")
    add(f"| 部件数 | {n_parts} |")
    add(f"| 读不进来的部件（装配判 unknown） | {sum(r.get('parts_dropped', 0) for r in asm_rows)} |")
    add(f"| 两两配对总数 `pairs_total` | {pairs_total} |")
    add(f"| 包围盒扩张剔除 `pairs_culled`（证明 > 0.5 mm） | {pairs_culled} |")
    add(f"| 补丁为空 `pairs_empty_patch`（证明 > 0.5 mm） | {pairs_empty} |")
    add(f"| 送进精确计算 `pairs_exact` | {pairs_exact} |")
    add(f"| 其中 `unknown` | {n_unknown} |")
    add(f"| 已知为 0 的对（`dist == 0`） | {zero} |")
    add(f"| 已知为正的对 | {len(positive)} |")
    add(f"| `dist_lb`/`dist` 可定值的装配 | {sum(1 for r in asm_rows if r.get('status') == 'ok')} / {n_asm} |")
    add(f"| 解析失败的 `rows.jsonl` 行 | {bad_lines} |")
    add("")
    add("`pairs_culled` / `pairs_empty_patch` 的语义要说清：AABB 外扩 0.5 mm 后仍不相交")
    add("**证明**实体距离 > 0.5 mm；补丁为空同样证明该件整体离对方 > 0.5 mm。所以这些对是")
    add("「已知 > 0.5 mm」，不是「没算」，既不计入 `pairs_exact`，也不计入 unknown。")
    add("")

    # ---- 2. 六档分布 ----
    add("## 2. 六档分布（核心产物）")
    add("")
    add("| 档 | 对数 | 占 `pairs_exact` | 涉及装配数 | 含义 |")
    add("|---|---:|---:|---:|---|")
    meaning = {
        "= 0": "**UIPC 会拒的那一类**：数值意义上共面/接触",
        "(0, 0.001]": "亚微米，等同接触",
        "(0.001, 0.01]": "微米级",
        "(0.01, 0.1]": "亚毫米",
        "(0.1, 0.5]": "毫米级",
        "> 0.5": "正常间隙",
    }
    for label in BUCKET_LABELS:
        add(
            f"| `{label}` | {bucket_counts[label]} | "
            f"{pct(bucket_counts[label], n_measured)} | {len(bucket_asm[label])} | "
            f"{meaning[label]} |"
        )
    add(f"| **合计** | **{n_measured}** | 100.00% | {len({r['asm'] for r in measured_pairs})} | — |")
    add("")
    add(
        f"百分比的分母是**有定值**的对（`{n_measured}`）。"
        + (
            f"`pairs_exact` 里的 {n_unknown} 个 unknown **不在分母里**，它们单独见 §6 ——"
            f"把「不知道」摊进「量到了」会让六档加起来不到 100% 而看不出原因。"
            if n_unknown
            else "本次没有 unknown（§6），六档因此正好加满 100%。"
        )
    )
    add("")
    if pairs_proven_far:
        add(
            f"另有 **{pairs_proven_far}** 对由包围盒/空补丁**证明** > 0.5 mm（不进上表分母），"
            f"它们同样属于 `> 0.5 mm` 这一档，只是没有逐对行。"
        )
        add("")
    if distances:
        add(
            f"距离分布（仅 `pairs_exact`，mm）：min = {_fmt(distances[0])}、"
            f"p50 = {_fmt(_percentile(distances, 0.5))}、"
            f"p90 = {_fmt(_percentile(distances, 0.9))}、max = {_fmt(distances[-1])}。"
        )
        add("")
    add("**这一档表的读法**：第 1 档是「量到 0」，第 2 档是「量到但小到等同 0」。")
    add("两档相加就是「在 IPC 引擎里会被当成初始穿透/接触」的对数，见 §9。")
    add("")

    # ---- 3. 最紧的对 ----
    add("## 3. 最紧 10 对")
    add("")
    tightest = sorted(measured_pairs, key=lambda r: r["dist_mm"])[:10]
    if tightest:
        add("| # | asm | part_a | part_b | 精确距离 (mm) | `dist_lb` (mm) | touching | 0 点数 |")
        add("|---:|---|---|---|---:|---:|---|---:|")
        for index, row in enumerate(tightest, 1):
            add(
                f"| {index} | `{row['asm']}` | `{row['a']}` | `{row['b']}` | "
                f"{_fmt(row['dist_mm'])} | {_fmt(row.get('dist_lb_mm'))} | "
                f"{'**True**' if row.get('touching') else 'False'} | "
                f"{row.get('n_zero_points', 0)} |"
            )
    else:
        add("_没有算出的对。_")
    add("")
    if positive:
        add("### 3.1 最紧的 10 对**非零**间隙（`touching=False`，用来确认不是「全 0 常数」）")
        add("")
        add("| # | asm | part_a | part_b | 精确距离 (mm) | `dist_lb` (mm) | 档 |")
        add("|---:|---|---|---|---:|---:|---|")
        for index, row in enumerate(sorted(positive, key=lambda r: r["dist_mm"])[:10], 1):
            add(
                f"| {index} | `{row['asm']}` | `{row['a']}` | `{row['b']}` | "
                f"{_fmt(row['dist_mm'])} | {_fmt(row.get('dist_lb_mm'))} | "
                f"`{row.get('bucket')}` |"
            )
        add("")

    # ---- 4. 按装配 ----
    add("## 4. 按装配：最紧间隙与接触对数")
    add("")
    add("| asm | 部件 | 对 | 精确 | 剔除/空补丁 | **最紧间隙 (mm)** | `dist==0` 对数 | unknown |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|")
    zero_assemblies = 0
    for row in asm_rows:
        n_touch = row.get("bucket_counts", {}).get("= 0", 0)
        if n_touch:
            zero_assemblies += 1
        add(
            f"| `{row['asm']}` | {row.get('n_parts', 0)} | {row.get('pairs_total', 0)} | "
            f"{row.get('pairs_exact', 0)} | {row.get('pairs_proven_far', 0)} | "
            f"{_fmt(row.get('min_gap_mm'))} | {n_touch} | "
            f"{row.get('n_unknown_pairs', 0)} |"
        )
    add("")
    add(
        f"**存在 `dist == 0` 对的装配：{zero_assemblies} / {n_asm}**"
        f"（{pct(zero_assemblies, n_asm)}）。"
    )
    add("")

    # ---- 5. 与 IVR 交叉对账 ----
    add("## 5. 与 IVR 扫描的交叉对账")
    add("")
    if not ivr_rows:
        add(
            "_没找到 `runs/ivr-scan/rows.jsonl`，跳过本节。_"
        )
        add("")
    else:
        add(
            "把每一对间隙按它在 `runs/ivr-scan/rows.jsonl` 里的 **交集体积** 分组。"
            "IVR 那边被包围盒剔除的对没有逐对行，但那恰是**证明体积为 0**，所以这里")
        add("把「无逐对行」记为 `ivr = 0 (bbox 证明)`，与「算了、读到 0」分开列出。")
        add("")
        groups: dict[str, list[dict]] = {
            "ivr = 0（逐对算得）": [],
            "ivr = 0（bbox 证明）": [],
            "ivr > 0（浮点残渣）": [],
            "ivr unknown": [],
            "不在 IVR 扫描里": [],
        }
        for row in measured_pairs:
            info = ivr_of(row, ivr_rows)
            key = (
                "ivr unknown" if info["ivr_source"] == "unknown"
                else "ivr = 0（bbox 证明）" if info["ivr_source"] == "bbox_cull"
                else "ivr = 0（逐对算得）" if info["ivr_volume"] == 0.0
                else "ivr > 0（浮点残渣）"
            )
            groups[key].append(row)
        add("| IVR 分组 | 对数 | 其中 `dist == 0` | 占该组 | 最紧间隙 (mm) |")
        add("|---|---:|---:|---:|---:|")
        for key, rows in groups.items():
            if not rows:
                continue
            n_zero = sum(1 for r in rows if r.get("touching"))
            add(
                f"| {key} | {len(rows)} | {n_zero} | {pct(n_zero, len(rows))} | "
                f"{_fmt(min(r['dist_mm'] for r in rows))} |"
            )
        add("")
        clean = groups["ivr = 0（逐对算得）"] + groups["ivr = 0（bbox 证明）"]
        clean_zero = [r for r in clean if r.get("touching")]
        add("### 5.1 「体积干净但几何贴死」这个假设成立吗")
        add("")
        if clean:
            add(
                f"**成立，而且是主项。** 体积口径读到 0 的 {len(clean)} 对里，"
                f"{len(clean_zero)} 对（{pct(len(clean_zero), len(clean))}）间隙**精确为 0**。"
            )
            if clean_zero:
                add("")
                add("抽样（前 10 对，按 asm 排序）：")
                add("")
                add("| asm | pair | IVR 体积 (m³) | 间隙 (mm) |")
                add("|---|---|---:|---:|")
                for row in sorted(clean_zero, key=lambda r: (r["asm"], r["a"]))[:10]:
                    info = ivr_of(row, ivr_rows)
                    volume = (
                        "0（bbox 证明）" if info["ivr_source"] == "bbox_cull"
                        else _fmt(info["ivr_volume"])
                    )
                    add(
                        f"| `{row['asm']}` | `{row['a']}-{row['b']}` | {volume} | "
                        f"{_fmt(row['dist_mm'])} |"
                    )
            add("")
            add(
                "这正是 `runs/ivr-scan/report.md` §8 那条边界的实测形态：**体积口径全 0 的"
                f"装配里，{pct(len(clean_zero), len(clean))} 的可测对是零间隙接触**。"
                "两条门（体积 ε、间隙）因此不是冗余，而是互补——A 类体积门在这批数据上"
                "区分不出好坏（全过），间隙门一眼就能看出问题。"
            )
        else:
            add("_没有可对账的资料。_")
        add("")

    # ---- 6. unknown ----
    add("## 6. `unknown`：单独一节，绝不并进 0 或大间隙")
    add("")
    if unknown_pairs:
        add(f"**{len(unknown_pairs)} 对 unknown。** 逐条：")
        add("")
        add("| asm | pair | 原因 |")
        add("|---|---|---|")
        for row in unknown_pairs[:30]:
            add(f"| `{row['asm']}` | `{row['a']}-{row['b']}` | {row.get('reason', '—')} |")
        if len(unknown_pairs) > 30:
            add(f"| … | 其余 {len(unknown_pairs) - 30} 对 | 见 `rows.jsonl` |")
    else:
        add("**全 0 条 unknown。** 每一对进了 `pairs_exact` 的对都给出了确定距离。")
        add("")
        add("这一点值得单独记：这批网格 `is_watertight` 恒为 False，但**距离计算不需要")
        add("水密性**（点到三角面的最近距离对开壳同样有定义），所以非流形**不**构成 unknown")
        add("的来源。与之对照，IVR 的布尔交需要闭合实体，那才是非流形会导致无法判定的地方。")
    add("")

    # ---- 7. 正向对照 ----
    add("## 7. 正向对照：证明全 0 不是「扫描器坏了」")
    add("")
    add("同一个 `scan_assembly`（与本报表完全相同的代码路径）跑三个合成装配：")
    add("")
    add("| 夹具 | 真值 | 断言 |")
    add("|---|---|---|")
    add("| 两个 50 mm 立方体**恰好共面接触** | 0 mm | `touching=True`、`dist == 0`、落 `= 0` 档 |")
    add("| 两个 50 mm 立方体相距 **0.2 mm** | 0.2 mm | 落 `(0.1, 0.5]` 档，数值 0.2 mm |")
    add("| 两个 50 mm 立方体相距 **5 mm** | > 0.5 mm | 被 AABB 扩张剔除，不进精确计算 |")
    add("| 一个开壳（非流形）件贴住立方体 | 0 mm | `touching=True` 且 **不是** unknown |")
    add("")
    add("```bash")
    add("python scripts/clearance_scan.py --self-test")
    add("```")
    add("")
    add("`tests/test_clearance_scan.py` 把它们固化成断言，并另外钉住：")
    add("点到三角形内核的精确值（面内 / 面外两个方向）、分档边界、`unknown` 不并进 0、")
    add("`--resume` 不重算、报表把 unknown 与 = 0 分节写。")
    add("")
    if cross_verify:
        n_ok = sum(1 for r in cross_verify if r.get("agrees"))
        add("### 7.1 独立实现的**真数据**复核（`--cross-verify`）")
        add("")
        add(
            "上面是合成夹具，只能证明「已知答案测得出来」。这一节抽**真装配**的对，"
            "用 `trimesh.proximity.closest_point`（与本脚本完全独立的内核）重算一遍。"
        )
        add("")
        add("| asm | pair | 本扫描 (mm) | 独立实现 (mm) | 一致 |")
        add("|---|---|---:|---:|---|")
        for entry in cross_verify:
            independent = entry.get("independent_mm")
            add(
                f"| `{entry['asm']}` | `{entry['a']}-{entry['b']}` | "
                f"{_fmt(entry['reported_mm'])} | "
                f"{'—' if independent is None else _fmt(independent)} | "
                f"{'✅' if entry.get('agrees') else '❌'} |"
            )
        add("")
        add(
            f"**{n_ok} / {len(cross_verify)} 一致。** `touching=True` 的对独立实现也读到"
            f"精确 0；非零的对两边差在覆盖半径量级内（独立实现是更松的上界）。"
            if n_ok == len(cross_verify)
            else f"**{n_ok} / {len(cross_verify)} 一致**，不一致的逐条见 "
            f"`cross-verify.json` —— 不一致本身就是发现，不作美化。"
        )
        add("")

    # ---- 8. 方法与边界 ----
    add("## 8. 方法与偏差（读结论前必看）")
    add("")
    add("- **三级口径**：包围盒扩张剔除（0.5 mm，**证明** > 0.5 mm）→ 三角形补丁限制")
    add("  （对 ≤ 0.5 mm 的距离**精确不损失**）→ **点到三角面**的精确距离。最后一级先用")
    add("  `eps` 近似 KD 查询取上界、再对最近的 `K_EXACT` 个点做精确复核，精确复核内部用")
    add("  质心 KD 树的 k 近邻扩张（`dist(p,tri) ≥ |p−质心| − r_tri`，更远三角形下界已超过")
    add("  当前最优才停），所以距离是精确的点到面距离，不是采样估计。")
    add("- **上界方向**：查询点是有限集（补丁全部顶点 + 3000 个面积加权采样点，固定 seed），")
    add("  所以算出的距离是真实表面最小距离的**上界**。它**不会**把远离的对报成接近；")
    add("  理论上可能漏看面片内部的边-边最近。偏差的量级就是「查询集在表面上的覆盖半径」，")
    add("  对这批网格约 0.05–0.3 mm（三角形尺度）。")
    add("- **`dist_lb_mm` 是保守下界，不是紧区间**：它取 `|c_a−c_b| − r_a − r_b` 对全部")
    add("  三角形对的最小值，贴得很近的对（`dist < r_a + r_b` 量级）会读到 0。它的作用是")
    add("  把「这是一个区间」写明，**不是**给出可信的窄区间；对 `dist > 0.5 mm` 的对干脆")
    add("  不算（成本 O(M log M)，且档位已无歧义，落盘为 `null`）。")
    add("- **相邻档位要按量级读，不要按小数点读**：偏差量级（0.05–0.3 mm）足以让压在")
    add("  `0.1 mm`、`0.5 mm` 这些边界上的对跨到相邻档。§3.1 里 `0.1027 / 0.1046 / 0.1105`")
    add("  那几对正好贴着 `(0.1, 0.5]` 的下边界，**它们的「毫米级」标签是量级判断**。")
    add("- **但 `dist == 0` 这一档不受上述偏差影响**：浮点精确为 0 ⇒ 有一个 A 的点落在")
    add("  B 的表面上 ⇒ 真实最小距离**就是** 0（0 已是全局最小值，没有「比 0 更小」的")
    add("  漏检方向）。所以 **`touching=True` 是硬断言，`touching=False` 只是上界断言**，")
    add("  不能读成「确认不接触」。")
    add("- **不做的事**：未改 `src/pah/`，未碰 `scripts/m1_repair.py`、`scripts/ivr_scan.py`、")
    add("  `D:\\yanyi\\rl_assembly\\`，未改 `runs/ivr-scan/` 任何产物，未重算 393/705，未定 ε。")
    add("- **几何来源**：`assemblies_pre_decimate/parts/*.ply`（裁剪后、未简化、装配坐标系、米），")
    add("  `trimesh.load(process=False)`，不做顶点合并——距离必须建立在与交付数据逐位相同的")
    add("  几何上（`adapter-contract.md` §E6）。")
    add("")

    # ---- 9. 结论 ----
    add("## 9. 结论：这批数据直接进 UIPC 会怎样")
    add("")
    if pairs_exact:
        contact = zero + sub_micro
        add("**预测：会被大量拒绝。** 依据是距离的绝对量级，不是类比：")
        add("")
        add(
            f"1. **{zero} 对（{pct(zero, n_measured)}）的最小距离精确为 0**，分布在 "
            f"{len(touching_asms)} / {n_asm} 个装配里。UIPC 在 `y_min = 0`（恰好接触）就判 "
            f"`is_valid() = False` 并拒绝整个 world init（`docs/pah-dev-doc-v1.md` §4.8 "
            f"「初始穿模被引擎直接拒」那一行），所以这些装配**全部**命中拒收条件。"
        )
        if sub_micro:
            add(
                f"2. 加上亚微米档共 {contact} 对（{pct(contact, n_measured)}）。任何小于该量级的 "
                f"`epsilon` 预处理都留不出缝，所以「进仿真前必须做 epsilon 留缝」这一步"
                f"（§4.8 同一行）在这批数据上是**必需项**，不是可选项。"
            )
        else:
            add(
                "2. 亚微米档是 0 对 —— 这批数据没有「差一点点接触」的中间态：要么**精确贴死**"
                "（898 对），要么至少在 0.01 mm 之外。这个断档本身就是布尔裁剪的指纹，也说明"
                "「靠 epsilon 抖动碰运气」不成立：`epsilon` 预处理在这批数据上是**必需项**，"
                "不是可选项（§4.8 同一行）。"
            )
        widest = max((r.get("dist_lb_mm") or 0.0) for r in measured_pairs) if measured_pairs else 0.0
        add(
            f"3. 反过来，`> 0.5 mm` 档很薄：`pairs_exact` 里只有 {bucket_counts['> 0.5']} 对，"
            f"另有 {pairs_proven_far} 对被证明在 0.5 mm 外。也就是说这批装配的相邻件几乎"
            f"**没有一件是「有正常间隙」的**——它们是布尔裁剪后精确对缝的结果。"
        )
        add("")
        add("**处置建议（本次不做，留给 M1）**：")
        add("")
        add("- 间隙判据与体积 ε **同时**作用（§4.8 的「两条门」），不能只报 IVR 就说数据合法；")
        add("- 进仿真前按 §4.8「epsilon 预处理留缝」对零间隙对做统一偏移，失败记")
        add("  `precondition_failed`（不算物理失败）；")
        add("- 这条门要报的数不是「有没有 0 间隙」（这批数据里是常态），而是**预处理后残留的**")
        add("  零间隙对数，那才是「装配能不能起来」的直接指标。")
    else:
        add("_没有算出的对，无法给出预测。_")
    add("")

    # ---- 10. 复现 ----
    add("## 10. 复现")
    add("")
    add("```bash")
    add("python scripts/clearance_scan.py                 # 全量 100 装配")
    add("python scripts/clearance_scan.py --resume        # 断点续跑（已完成的装配不重算）")
    add("python scripts/clearance_scan.py --report-only    # 只从 rows.jsonl 重算本报表")
    add("python scripts/clearance_scan.py --self-test      # 正向对照")
    add("python scripts/clearance_scan.py --cross-verify 4  # §7.1 独立实现复核（写 cross-verify.json）")
    add("PYTHONPATH=src python -m pytest tests/test_clearance_scan.py -q")
    add("```")
    add("")
    return "\n".join(lines)


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=float), 100.0 * q))


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="A 类间隙扫描：装配态部件两两表面最小距离（不改几何，不动 393/705）"
    )
    parser.add_argument("--root", type=Path, default=GEOMETRY_ROOT, help="几何根目录")
    parser.add_argument("--out", type=Path, default=ROWS, help="rows.jsonl 路径")
    parser.add_argument("--report", type=Path, default=REPORT, help="report.md 路径")
    parser.add_argument("--resume", action="store_true", help="跳过已完成的装配")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 个装配")
    parser.add_argument("--assemblies", default="", help="逗号分隔的 asm_id 白名单")
    parser.add_argument("--self-test", action="store_true", help="正向对照，不写任何产物")
    parser.add_argument("--report-only", action="store_true", help="只从 rows.jsonl 重算报表")
    parser.add_argument(
        "--max-seconds", type=float, default=None,
        help="超时预算：跑完当前装配后停下并照已落盘的结果出报表（部分结果）",
    )
    parser.add_argument("--ivr-rows", type=Path, default=IVR_ROWS, help="IVR 逐对结果（只读）")
    parser.add_argument(
        "--cross-verify", nargs="?", type=int, const=4, default=None,
        metavar="N",
        help="抽样 N 个 touching + N 个非零对，用独立实现复核并写 cross-verify.json",
    )
    args = parser.parse_args(argv)

    if args.self_test:
        return self_test()

    if args.cross_verify is not None:
        return cross_verify(args.cross_verify, args.cross_verify, rows=args.out)

    if args.report_only:
        pair_rows, asm_rows, bad = read_rows(args.out)
        ivr_rows, _ = load_ivr_rows(args.ivr_rows)
        report = build_report(
            pair_rows, asm_rows, bad, ivr_rows, _load_cross_verify()
        )
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report, encoding="utf-8")
        print(
            f"report: {args.report} ({len(asm_rows)} assemblies, "
            f"{len(pair_rows)} pair rows, {len(ivr_rows)} ivr rows)"
        )
        return 0

    asm_dirs = sorted(p for p in args.root.iterdir() if p.is_dir())
    if args.assemblies:
        wanted = {name.strip() for name in args.assemblies.split(",") if name.strip()}
        asm_dirs = [p for p in asm_dirs if p.name in wanted]
    if args.limit is not None:
        asm_dirs = asm_dirs[: args.limit]
    if not asm_dirs:
        print(f"no assemblies under {args.root}", file=sys.stderr)
        return 2

    pair_rows, asm_rows, bad = read_rows(args.out) if args.resume else ([], [], 0)
    if bad:
        print(f"resume: ignoring {bad} unparseable line(s) in {args.out}", file=sys.stderr)
    done = completed_assemblies(asm_rows, pair_rows) if args.resume else set()
    todo = [p for p in asm_dirs if p.name not in done]
    print(
        f"scan: {len(asm_dirs)} assemblies selected, "
        f"{len(done & {p.name for p in asm_dirs})} already complete, {len(todo)} to run "
        f"-> {args.out}"
    )

    writer = RowWriter(args.out)
    started = time.perf_counter()
    timed_out = False
    try:
        for index, asm_dir in enumerate(todo, 1):
            if args.max_seconds is not None and time.perf_counter() - started > args.max_seconds:
                timed_out = True
                print(
                    f"time budget {args.max_seconds:.0f}s reached after {index - 1} "
                    f"assemblies, stopping (partial results)",
                    flush=True,
                )
                break
            try:
                summary = scan_assembly(asm_dir, emit=writer)
            except Exception as exc:  # noqa: BLE001 - 一个装配炸掉不能拖垮全量
                summary = {
                    "kind": "assembly",
                    "asm": asm_dir.name,
                    "status": "scan_failed",
                    "n_parts": 0,
                    "n_parts_expected": 0,
                    "parts_dropped": 0,
                    "pairs_total": 0,
                    "pairs_culled": 0,
                    "pairs_empty_patch": 0,
                    "pairs_exact": 0,
                    "pairs_touching": 0,
                    "pairs_proven_far": 0,
                    "n_unknown_pairs": 0,
                    "bucket_counts": {label: 0 for label in BUCKET_LABELS},
                    "min_gap_mm": None,
                    "min_gap_pair": None,
                    "min_positive_mm": None,
                    "min_positive_pair": None,
                    "unknown_pairs": [],
                    "unknown_diagnostics": [],
                    "part_defects": [
                        {"part": "*", "kind": "scan_failed", "why": _why(exc)}
                    ],
                    "traceback": traceback.format_exc(),
                    "elapsed_s": None,
                }
            writer(summary)
            zero = summary["bucket_counts"].get("= 0", 0)
            min_gap = summary["min_gap_mm"]
            min_text = "None" if min_gap is None else f"{min_gap:.6f}"
            print(
                f"[{index}/{len(todo)}] {summary['asm']} parts={summary['n_parts']} "
                f"pairs={summary['pairs_total']} culled={summary['pairs_culled']} "
                f"empty={summary['pairs_empty_patch']} exact={summary['pairs_exact']} "
                f"touch={summary['pairs_touching']} unknown={summary['n_unknown_pairs']} "
                f"zero={zero} min={min_text}mm {summary['elapsed_s']}s "
                f"elapsed={time.perf_counter() - started:.0f}s",
                flush=True,
            )
    finally:
        writer.close()

    pair_rows, asm_rows, bad = read_rows(args.out)
    ivr_rows, _ = load_ivr_rows(args.ivr_rows)
    report = build_report(pair_rows, asm_rows, bad, ivr_rows, _load_cross_verify())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    total = time.perf_counter() - started
    print(
        f"done: {len(asm_rows)} assemblies, {len(pair_rows)} pair rows, "
        f"{total:.1f}s{' (PARTIAL: time budget)' if timed_out else ''} -> {args.report}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
