"""`scripts/clearance_scan.py` 的契约：A 类间隙扫描必须能把**零间隙**量出来。

本文件的存在理由只有一条 —— `runs/ivr-scan/` 报了「体积全 0」，而
**体积 0 不等于装配物理合法**（UIPC 在 `y_min = 0` 就判 `is_valid() = False`）。
所以第一个断言必须是：两个**恰好共面接触**的立方体，必须报 `touching=True`、
`dist == 0`。第二条钉住分档的另一端：相距 0.2 mm 的必须落 `(0.1, 0.5]` 档。

其余几条钉的是这个扫描器最容易悄悄说谎的四个地方：

- **采样下限陷阱**：只做「点到点」采样的话，两件贴死也会被报成采样间距
  （几百微米），零间隙会被系统性漏掉。`test_face_interior_contact_is_found`
  用一个「顶点互不重合、面内接触」的夹具证明走的是**点到面**而不是点到点。
- **unknown 不许折成 0 也不许折成大间隙**：读不进来的件让整条装配 unknown。
- **非流形不是 unknown 的理由**：距离不需要水密性，开壳件照样要给出确定距离。
- **剔除是证明不是跳过**：AABB 外扩 0.5 mm 不相交的对必须进 `pairs_culled`，
  且不进 `pairs_exact` 分母。

跑：`python -m pytest tests/test_clearance_scan.py -q`
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh

REPO = Path(__file__).resolve().parents[1]
#: `pah` 是 src 布局、不装进环境也能测（与仓里其它测试同一个约定）。
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

_SPEC = importlib.util.spec_from_file_location(
    "clearance_scan", REPO / "scripts" / "clearance_scan.py"
)
assert _SPEC and _SPEC.loader
clearance_scan = importlib.util.module_from_spec(_SPEC)
sys.modules["clearance_scan"] = clearance_scan
_SPEC.loader.exec_module(clearance_scan)

SIDE = 0.05  # 50 mm，与真实部件同量级


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def _write_boxes(asm_dir: Path, starts: list[tuple[float, float, float]],
                 extents: list[tuple[float, float, float]] | None = None) -> None:
    """在 `asm_dir/parts/` 下写出若干立方体/长方体。"""
    parts = asm_dir / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    extents = extents or [(SIDE, SIDE, SIDE)] * len(starts)
    for index, (start, extent) in enumerate(zip(starts, extents)):
        mesh = trimesh.creation.box(extents=extent)
        mesh.apply_translation(np.asarray(start) + np.asarray(extent) / 2.0)
        mesh.export(parts / f"part_{index:03d}.ply")


# --------------------------------------------------------------------------- #
# 1. 正向对照：零间隙必须量到 0，0.2 mm 必须落对档
# --------------------------------------------------------------------------- #
def test_coplanar_contact_is_reported_as_touching(tmp_path):
    """两个 50 mm 立方体**恰好共面接触**：真值 0，必须 `touching=True`。

    这就是 UIPC `is_valid()=False` 的那一类几何。扫描器报不出它，整条结论链就断了。
    """
    asm = tmp_path / "asm_touch"
    _write_boxes(asm, [(0.0, 0.0, 0.0), (SIDE, 0.0, 0.0)])

    summary = clearance_scan.scan_assembly(asm)

    assert summary["n_parts"] == 2
    assert summary["pairs_total"] == 1
    assert summary["pairs_culled"] == 0
    assert summary["pairs_exact"] == 1
    assert summary["pairs_touching"] == 1
    assert summary["min_gap_mm"] == 0.0
    assert summary["bucket_counts"]["= 0"] == 1
    assert summary["n_unknown_pairs"] == 0


def test_a_02mm_gap_lands_in_the_right_bucket(tmp_path):
    """相距 0.2 mm：必须落 `(0.1, 0.5]` 档且数值等于 0.2 mm。

    容差 1e-6 mm：PLY 顶点按 float32 存，0.05 进来会变成 0.050000000745…，
    这是**输入精度**的代价，不是算法的。
    """
    asm = tmp_path / "asm_02mm"
    _write_boxes(asm, [(0.0, 0.0, 0.0), (SIDE + 0.0002, 0.0, 0.0)])

    summary = clearance_scan.scan_assembly(asm)

    assert summary["pairs_exact"] == 1
    assert summary["pairs_touching"] == 0
    assert summary["min_gap_mm"] == pytest.approx(0.2, abs=1e-6)
    assert summary["bucket_counts"]["(0.1, 0.5]"] == 1


def test_a_5mm_gap_is_culled_and_counted_as_proven_far(tmp_path):
    """相距 5 mm：AABB 外扩 0.5 mm 后仍不相交 ⇒ 证明 > 0.5 mm，不进精确计算。"""
    asm = tmp_path / "asm_5mm"
    _write_boxes(asm, [(0.0, 0.0, 0.0), (SIDE + 0.005, 0.0, 0.0)])

    summary = clearance_scan.scan_assembly(asm)

    assert summary["pairs_culled"] == 1
    assert summary["pairs_exact"] == 0
    assert summary["pairs_proven_far"] == 1
    assert summary["pairs_touching"] == 0
    # 剔除的对不许有「最紧间隙」——没有量过就没有值
    assert summary["min_gap_mm"] is None


def test_self_test_entry_point_passes():
    """`--self-test` 是给人跑的一行命令，它自己必须绿。"""
    assert clearance_scan.self_test() == 0


# --------------------------------------------------------------------------- #
# 2. 面内接触：证明走的是「点到面」，不是「点到点」
# --------------------------------------------------------------------------- #
def test_face_interior_contact_is_found(tmp_path):
    """小立方体坐在大立方体顶面上，**顶点互不重合**：必须报 0。

    这条是本扫描器最关键的一条断言。只做「点到点」采样的话，两件贴死也会被报成
    采样间距（这里会报 ~1e-2 m 量级），零间隙被系统性漏掉；只有点到**面**的精确
    距离才能给出 0。夹具刻意让两件的顶点在同一平面内错开，确保不可能靠顶点重合
    「碰巧」命中。
    """
    asm = tmp_path / "asm_face_contact"
    _write_boxes(
        asm,
        [(0.0, 0.0, 0.0), (0.010, 0.010, SIDE)],        # 大件顶面 z = 0.05
        [(SIDE, SIDE, SIDE), (0.020, 0.020, 0.020)],     # 小件底面 z = 0.05
    )

    summary = clearance_scan.scan_assembly(asm)
    assert summary["pairs_exact"] == 1
    assert summary["pairs_touching"] == 1, "面内共面接触没被量到 0"
    assert summary["min_gap_mm"] == 0.0

    # 夹具自检：两件的顶点集合必须真的没有重合点，否则这条测试证明不了
    # 「走的是点到面」——它只能证明顶点重合也能被查到。
    parts, _ = clearance_scan.load_parts(asm / "parts")
    a = _patch_of(parts[0], parts[1])
    b = _patch_of(parts[1], parts[0])
    from scipy.spatial import cKDTree

    nearest, _ = cKDTree(np.asarray(b.vertices)).query(np.asarray(a.vertices), k=1)
    assert nearest.min() > 1e-6, (
        f"夹具里出现了相距 {nearest.min():.3e} m 的顶点对，"
        "这样命中 0 可能只是顶点重合，证不出面内接触这条路"
    )


def _patch_of(part, other) -> trimesh.Trimesh:
    patch = part.patch(other.bounds)
    assert patch is not None
    return patch


# --------------------------------------------------------------------------- #
# 3. 内核与分档的边界
# --------------------------------------------------------------------------- #
def test_point_triangle_kernel_is_exact():
    """点到三角形的精确距离：面内 / 边上 / 顶点 / 面外四个位置。"""
    triangle = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])

    face = clearance_scan.point_triangle_distance(np.array([0.25, 0.25, 2e-4]), triangle)
    assert face[0] == pytest.approx(2e-4, abs=1e-15)

    edge = clearance_scan.point_triangle_distance(np.array([0.5, -3e-4, 0.0]), triangle)
    assert edge[0] == pytest.approx(3e-4, abs=1e-15)

    vertex = clearance_scan.point_triangle_distance(np.array([-0.0, -0.0, 7e-4]), triangle)
    assert vertex[0] == pytest.approx(7e-4, abs=1e-15)

    far = clearance_scan.point_triangle_distance(np.array([3.0, 0.0, 0.0]), triangle)
    assert far[0] == pytest.approx(2.0, abs=1e-15)


def test_bucket_boundaries_are_left_open_right_closed():
    """分档边界：0 单独一档，其余左开右闭。"""
    assert clearance_scan.bucket_of(0.0) == "= 0"
    assert clearance_scan.bucket_of(0.0005) == "(0, 0.001]"
    assert clearance_scan.bucket_of(0.001) == "(0, 0.001]"
    assert clearance_scan.bucket_of(0.001 + 1e-9) == "(0.001, 0.01]"
    assert clearance_scan.bucket_of(0.01) == "(0.001, 0.01]"
    assert clearance_scan.bucket_of(0.0501) == "(0.01, 0.1]"
    assert clearance_scan.bucket_of(0.1) == "(0.01, 0.1]"
    assert clearance_scan.bucket_of(0.2) == "(0.1, 0.5]"
    assert clearance_scan.bucket_of(0.5) == "(0.1, 0.5]"
    assert clearance_scan.bucket_of(0.5001) == "> 0.5"
    assert clearance_scan.bucket_of(None) is None


def test_distance_to_box_is_zero_inside():
    """盒内为 0，盒外为到最近面的欧氏距离。"""
    bounds = (np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]))
    inside = clearance_scan._distance_to_box(np.array([[0.5, 0.5, 0.5]]), bounds)
    assert inside[0] == 0.0
    outside = clearance_scan._distance_to_box(np.array([[2.0, 0.5, 0.5]]), bounds)
    assert outside[0] == pytest.approx(1.0)
    corner = clearance_scan._distance_to_box(np.array([[1.0, 1.0, 2.0]]), bounds)
    assert corner[0] == pytest.approx(1.0)


def test_lower_bound_never_exceeds_the_measured_distance(tmp_path):
    """`dist_lb ≤ dist` 必须恒成立 —— 上下界打架就是「区间」在撒谎。"""
    asm = tmp_path / "asm_bracket"
    _write_boxes(asm, [(0.0, 0.0, 0.0), (SIDE + 0.0002, 0.0, 0.0)])

    pair_rows, _, _ = clearance_scan.read_rows(_scan_to_rows(asm, tmp_path))

    assert len(pair_rows) == 1
    row = pair_rows[0]
    assert row["dist_lb_mm"] is not None
    assert row["dist_lb_mm"] <= row["dist_mm"]
    assert row["dist_lb_mm"] <= 0.2


def test_lower_bound_is_skipped_for_wide_pairs(tmp_path):
    """距离 > 0.5 mm 的对不给下界（它是 O(M log M) 的报表量，且档位已无歧义）。

    夹具要同时满足两件事，否则测的不是这条分支：AABB 外扩 0.5 mm 后仍相交
    （不被剔除），而**表面距离** > 0.5 mm。沿 x、y 各错开 0.4 mm 正好满足：
    平行面在 y 上完全不重叠，最近点落在两条角边上，距离 √2×0.4 ≈ 0.57 mm > 0.5 mm，
    而两个方向的 AABB 间隙都只有 0.4 mm < 0.5 mm。
    """
    asm = tmp_path / "asm_wide"
    _write_boxes(asm, [(0.0, 0.0, 0.0), (SIDE + 0.0004, SIDE + 0.0004, 0.0)])

    pair_rows, _, _ = clearance_scan.read_rows(_scan_to_rows(asm, tmp_path))

    assert len(pair_rows) == 1, "这对必须没被 AABB 剔除，否则测不到「宽对不给下界」"
    row = pair_rows[0]
    assert row["dist_mm"] > 0.5, f"夹具没造出 > 0.5 mm 的距离：{row['dist_mm']}"
    assert row["bucket"] == "> 0.5"
    assert row["dist_lb_mm"] is None


def _scan_to_rows(asm: Path, tmp_path: Path) -> Path:
    """把单个装配扫进一个临时 `rows.jsonl` 并返回路径。"""
    out = tmp_path / f"{asm.name}.jsonl"
    writer = clearance_scan.RowWriter(out)
    try:
        clearance_scan.scan_assembly(asm, emit=writer)
    finally:
        writer.close()
    return out


# --------------------------------------------------------------------------- #
# 4. unknown 与「非流形不是 unknown 的理由」
# --------------------------------------------------------------------------- #
def test_a_part_that_cannot_be_loaded_makes_the_assembly_unknown(tmp_path):
    """读不进来的件让整条装配 unknown，且这些对**不落任何档**。

    这一条防的是最隐蔽的一种失败：少一件、分母与配对少一整列，剩下的对照样
    报出一堆好看的 0，而「不知道」被静默消化掉了。
    """
    asm = tmp_path / "asm_drop"
    _write_boxes(asm, [(0.0, 0.0, 0.0), (SIDE, 0.0, 0.0)])
    (asm / "parts" / "part_002.ply").write_text(
        "ply\nformat ascii 1.0\nelement vertex 1\n"
        "property float x\nproperty float y\nproperty float z\n"
        "end_header\n0 0 0\n",
        encoding="ascii",
    )

    summary = clearance_scan.scan_assembly(asm)

    assert summary["n_parts_expected"] == 3
    assert summary["parts_dropped"] == 1
    assert summary["status"] == "assembly_unknown"
    # 剩下的一对还是会算出来，但装配级结论必须带 unknown 标志
    assert summary["pairs_exact"] == 1


def test_non_manifold_part_is_measurable_not_unknown(tmp_path):
    """开壳（非流形）件照样要给确定距离：距离不需要水密性。

    与 IVR 的分工要分清：布尔交需要闭合实体，那才是非流形会导致无法判定的地方。
    """
    asm = tmp_path / "asm_open"
    parts = asm / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    box = trimesh.creation.box(extents=(SIDE, SIDE, SIDE))
    box.export(parts / "part_000.ply")
    fin = trimesh.creation.box(extents=(SIDE, SIDE, SIDE))
    fin.apply_translation((SIDE, 0.0, 0.0))
    apex = fin.vertices[[0]] + np.array([[0.0, 0.0, 2.5]])
    vertices = np.vstack([fin.vertices, apex])
    faces = np.vstack([fin.faces, [[0, 1, len(fin.vertices)]]])
    trimesh.Trimesh(vertices, faces, process=False).export(parts / "part_001.ply")

    summary = clearance_scan.scan_assembly(asm)

    assert summary["pairs_exact"] == 1
    assert summary["n_unknown_pairs"] == 0
    assert summary["status"] == "ok"
    assert summary["pairs_touching"] == 1


# --------------------------------------------------------------------------- #
# 5. 落盘 / 续跑 / 报表
# --------------------------------------------------------------------------- #
def test_scan_writes_rows_and_a_report_end_to_end(tmp_path):
    """一次 `main` 跑两个装配：rows.jsonl 可回读，报表六档齐全且 unknown 单独成节。"""
    root = tmp_path / "geom"
    _write_boxes(root / "asm_000001", [(0.0, 0.0, 0.0), (SIDE, 0.0, 0.0)])
    _write_boxes(root / "asm_000000", [(0.0, 0.0, 0.0), (SIDE + 0.0002, 0.0, 0.0)])
    out = tmp_path / "rows.jsonl"
    report = tmp_path / "report.md"

    code = clearance_scan.main(
        ["--root", str(root), "--out", str(out), "--report", str(report)]
    )
    assert code == 0

    pair_rows, asm_rows, bad = clearance_scan.read_rows(out)
    assert bad == 0
    assert {r["asm"] for r in asm_rows} == {"asm_000000", "asm_000001"}
    measured = {r["asm"]: r for r in asm_rows}
    assert measured["asm_000001"]["pairs_touching"] == 1
    assert measured["asm_000000"]["min_gap_mm"] == pytest.approx(0.2, abs=1e-6)

    text = report.read_text(encoding="utf-8")
    for label in clearance_scan.BUCKET_LABELS:
        assert f"| `{label}` |" in text, f"报表缺 {label} 这一档"
    assert "unknown" in text
    assert "A 类" in text


def test_resume_does_not_recompute_a_completed_assembly(tmp_path):
    """`--resume` 跳过已完成的装配：第二次跑不再追加任何行。"""
    root = tmp_path / "geom"
    _write_boxes(root / "asm_000001", [(0.0, 0.0, 0.0), (SIDE, 0.0, 0.0)])
    out = tmp_path / "rows.jsonl"
    report = tmp_path / "report.md"

    first = ["--root", str(root), "--out", str(out), "--report", str(report)]
    assert clearance_scan.main(first) == 0
    before = out.read_text(encoding="utf-8").count("\n")

    assert clearance_scan.main(first + ["--resume"]) == 0
    assert out.read_text(encoding="utf-8").count("\n") == before


def test_completed_assemblies_requires_the_pair_rows(tmp_path):
    """只有汇总行、没有逐对行的装配不算完成 —— 否则续跑会留下缺项。"""
    asm_rows = [{"asm": "asm_x", "pairs_exact": 2}]
    assert clearance_scan.completed_assemblies(asm_rows, []) == set()
    partial = [{"asm": "asm_x", "a": "p0", "b": "p1"}]
    assert clearance_scan.completed_assemblies(asm_rows, partial) == set()
    full = [{"asm": "asm_x", "a": "p0", "b": "p1"},
            {"asm": "asm_x", "a": "p0", "b": "p2"}]
    assert clearance_scan.completed_assemblies(asm_rows, full) == {"asm_x"}


def test_bad_lines_are_counted_not_fatal(tmp_path):
    """写一半崩掉的残行只计数，不拖垮整份 rows.jsonl。"""
    out = tmp_path / "rows.jsonl"
    good = json.dumps({"kind": "assembly", "asm": "asm_0", "pairs_exact": 0})
    truncated = '{"kind": "pair", "asm": "asm_0", "a": "p0"'
    out.write_text(f"{good}\n{truncated}\n", encoding="utf-8")

    pair_rows, asm_rows, bad = clearance_scan.read_rows(out)

    assert bad == 1
    assert len(asm_rows) == 1
    assert pair_rows == []


# --------------------------------------------------------------------------- #
# 6. 与 IVR 交叉对账
# --------------------------------------------------------------------------- #
def test_ivr_bbox_cull_counts_as_a_proven_zero():
    """IVR 那边被包围盒剔除的对没有逐对行，但那恰是**证明体积为 0**。

    这条决定了 §5 的分母怎么取：把「没行」当成「没算」会把最大的那一块丢掉。
    """
    info = clearance_scan.ivr_of(
        {"asm": "asm_0", "a": "p0", "b": "p1"}, {}
    )
    assert info["ivr_source"] == "bbox_cull"
    assert info["ivr_volume"] == 0.0

    measured = {("asm_0", "p0", "p1"): {"volume": 0.0}}
    assert clearance_scan.ivr_of(
        {"asm": "asm_0", "a": "p0", "b": "p1"}, measured
    )["ivr_source"] == "exact"

    unknown = {("asm_0", "p0", "p1"): {"volume": None}}
    assert clearance_scan.ivr_of(
        {"asm": "asm_0", "a": "p0", "b": "p1"}, unknown
    )["ivr_source"] == "unknown"


def test_report_cross_tabulates_ivr_zero_against_zero_clearance():
    """「体积干净但几何贴死」这一栏必须能在报表里被数出来。"""
    asm_rows = [
        {
            "kind": "assembly", "asm": "asm_000000", "status": "ok",
            "n_parts": 2, "parts_dropped": 0, "pairs_total": 1,
            "pairs_culled": 0, "pairs_empty_patch": 0, "pairs_exact": 1,
            "pairs_touching": 1, "pairs_proven_far": 0, "n_unknown_pairs": 0,
            "bucket_counts": {**{l: 0 for l in clearance_scan.BUCKET_LABELS}, "= 0": 1},
            "min_gap_mm": 0.0, "min_gap_pair": ["part_000", "part_001"],
            "min_positive_mm": None, "min_positive_pair": None,
            "unknown_pairs": [], "unknown_diagnostics": [], "part_defects": [],
            "elapsed_s": 0.1,
        }
    ]
    pair_rows = [
        {
            "kind": "pair", "asm": "asm_000000", "a": "part_000", "b": "part_001",
            "status": "exact", "dist_m": 0.0, "dist_mm": 0.0, "dist_lb_mm": 0.0,
            "bucket": "= 0", "touching": True, "n_zero_points": 12,
            "n_query_points": 100, "n_tri": 0, "faces": [2, 2], "ms": 1.0,
        }
    ]
    # IVR 行里只有**别的**对，本对没有逐对行 ⇒ 那一对是被 bbox 剔除的，体积为 0 是证明。
    # 传空 dict 不行：那等于「文件不存在」，报表会整节跳过。
    ivr_rows = {("asm_999999", "part_000", "part_001"): {"volume": 0.0}}

    text = clearance_scan.build_report(pair_rows, asm_rows, 0, ivr_rows)

    assert "## 5. 与 IVR 扫描的交叉对账" in text
    assert "ivr = 0（bbox 证明）" in text
    assert "成立，而且是主项" in text
