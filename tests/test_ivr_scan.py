"""`scripts/ivr_scan.py` 的契约：A 类扫描对**已知重叠**必须能测出来。

本文件的存在理由只有一条 —— **全量扫描报出全 0 时，必须能排除「扫描器坏了」**。
所以第一个测试就是一个已知答案的合成装配：两个单位立方体错开半个身位，
`|A∩B| = 0.5`、`Σ|V| = 2`、`ivr = 0.25`，手算值，容差 1e-9。

其余几条钉的是「不让 0 把不知道吃进去」和「口径不许在这里另造」：

- AABB 剔除判据与 `ivr.py` 的 `_overlap_extent` 逐字一致（否则会出现「本脚本送进布尔、
  引擎内部却把它剔除成 0」的漂移）；
- 逐对调用的聚合结果与整批 `ivr_exact` 的 `value` **逐位相等**（口径不另造的证据）；
- 非流形件读 unknown、装配级 `ivr_assembly` 变 `None`，且区间 `[lo, hi]` 给出上界；
- `--resume` 不重算已完成的装配，且报表把 unknown 与 0 分节写。

跑：`python -m pytest tests/test_ivr_scan.py -q`
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh

REPO = Path(__file__).resolve().parents[1]
#: `pah` 是 src 布局、不装进环境也能测（与仓里其它测试同一个约定）：
#: 显式把 `src` 放进 `sys.path`，不依赖 `scripts/ivr_scan.py` 里那句插入的副作用。
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

_SPEC = importlib.util.spec_from_file_location(
    "ivr_scan", REPO / "scripts" / "ivr_scan.py"
)
assert _SPEC and _SPEC.loader
ivr_scan = importlib.util.module_from_spec(_SPEC)
sys.modules["ivr_scan"] = ivr_scan
_SPEC.loader.exec_module(ivr_scan)

from pah.tools import ivr  # noqa: E402
from pah.tools.ivr import ivr_exact  # noqa: E402


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def _write_cubes(asm_dir: Path, shifts: list[float], side: float = 1.0) -> None:
    """在 `asm_dir/parts/` 下写出 `len(shifts)` 个单位立方体，沿 x 轴错开。"""
    parts = asm_dir / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    for index, shift in enumerate(shifts):
        mesh = trimesh.creation.box(extents=(side, side, side))
        mesh.apply_translation((shift, 0.0, 0.0))
        mesh.export(parts / f"part_{index:03d}.ply")


def _write_non_manifold_pair(asm_dir: Path) -> None:
    """一个立方体 + 一个带鳍的开壳件：后者没有可信体积，必须读 unknown。"""
    parts = asm_dir / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    box.export(parts / "part_000.ply")

    fin = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    apex = fin.vertices[[0]] + np.array([[0.0, 0.0, 2.5]])
    vertices = np.vstack([fin.vertices, apex])
    faces = np.vstack([fin.faces, [[0, 1, len(fin.vertices)]]])
    trimesh.Trimesh(vertices, faces, process=False).export(parts / "part_001.ply")


# --------------------------------------------------------------------------- #
# 1. 正向对照：已知重叠必须被量到
# --------------------------------------------------------------------------- #
def test_known_half_overlap_is_measured_as_a_quarter(tmp_path):
    """两个单位立方体错开 0.5：真值 0.5 / 2 = 0.25，容差 1e-9。

    这是「全量报全 0」时唯一能排除「扫描器坏了」的那条证据，也是本文件的第一断言。
    """
    asm = tmp_path / "asm_overlap"
    _write_cubes(asm, [0.0, 0.5])

    summary = ivr_scan.scan_assembly(asm)

    assert summary["n_parts"] == 2
    assert summary["pairs_total"] == 1
    assert summary["pairs_culled"] == 0
    assert summary["pairs_exact"] == 1
    assert summary["n_unknown_pairs"] == 0
    assert summary["ivr_assembly"] == pytest.approx(0.25, abs=1e-9)
    assert summary["pairs_nonzero"] == 1
    assert summary["max_pair"] == ["part_000", "part_001"]


def test_a_deeper_overlap_reads_a_bigger_number(tmp_path):
    """错开 0.9：|A∩B| = 0.1 → 0.05。扫描器不是只会读 0.25 这个常数。

    容差放在 1e-6 而不是 1e-9：PLY 顶点按 float32 存，0.9 进来会变成 0.89999997，
    这是**输入精度**的代价，不是算法的。0.5 那个用例全程可精确表示，所以还是 1e-9。
    """
    asm = tmp_path / "asm_deep"
    _write_cubes(asm, [0.0, 0.9])

    assert ivr_scan.scan_assembly(asm)["ivr_assembly"] == pytest.approx(0.05, abs=1e-6)


def test_self_test_entry_point_passes():
    """`--self-test` 是给人跑的一行命令，它自己必须绿。"""
    assert ivr_scan.self_test() == 0


# --------------------------------------------------------------------------- #
# 2. 包围盒剔除：证明为 0，不是「跳过」
# --------------------------------------------------------------------------- #
def test_disjoint_pair_is_culled_without_a_boolean(tmp_path):
    """AABB 不相交是实体不相交的证明，所以这对是已知 0，且不进 `pairs_exact`。"""
    asm = tmp_path / "asm_far"
    _write_cubes(asm, [0.0, 3.0])

    summary = ivr_scan.scan_assembly(asm)

    assert summary["pairs_culled"] == 1
    assert summary["pairs_exact"] == 0
    assert summary["ivr_assembly"] == 0.0
    assert summary["pairs_proven_zero"] == 1


def test_cull_rule_matches_ivr_overlap_extent():
    """剔除判据必须与 `ivr.py:_overlap_extent` 逐字一致。

    两者不一致会出现这种漂移：本脚本把一对送进 `ivr_exact`，引擎内部却按自己的
    判据把它剔除成 `method="bbox"`，于是 `pairs_exact` 数对了、数值也对，但「什么
    算算过」和「什么算证明」两个口径已经分家。
    """
    rng = np.random.default_rng(0)
    for _ in range(50):
        sides = rng.uniform(0.2, 1.5, size=2)
        offsets = rng.uniform(-2.0, 2.0, size=(2, 3))
        a = trimesh.creation.box(extents=(sides[0],) * 3)
        b = trimesh.creation.box(extents=(sides[1],) * 3)
        a.apply_translation(offsets[0])
        b.apply_translation(offsets[1])

        assert ivr_scan.aabb_misses(a, b) == (ivr._overlap_extent(a, b) is None)


# --------------------------------------------------------------------------- #
# 3. 口径不另造：逐对聚合 == 整批 ivr_exact
# --------------------------------------------------------------------------- #
def test_per_pair_aggregation_equals_the_batch_measurement(tmp_path):
    """一次一对地算、再自己求和，必须等于 `ivr_exact` 整批给出的 `value`。

    这是「不另造归一化」的证据：本脚本唯一做的事是把 per-pair 的分子加起来、
    除以全部部件体积，分母/求和顺序都与 `ivr.py:_measure` 相同。
    """
    asm = tmp_path / "asm_three"
    _write_cubes(asm, [0.0, 0.5, 5.0])  # 前两件重叠一半，第三件离得远

    summary = ivr_scan.scan_assembly(asm)

    parts, _ = ivr_scan.load_parts(asm / "parts")
    batch = ivr_exact(parts)
    assert batch.value is not None
    assert summary["ivr_assembly"] == pytest.approx(batch.value, rel=1e-12)
    assert summary["pairs_total"] == 3
    assert summary["pairs_culled"] == 2  # p0-p2 与 p1-p2
    assert summary["ivr_assembly"] == pytest.approx(0.5 / 3.0, rel=1e-12)


# --------------------------------------------------------------------------- #
# 4. unknown 不许当成 0
# --------------------------------------------------------------------------- #
def test_non_manifold_pair_is_unknown_and_the_assembly_goes_none(tmp_path):
    """开壳件必须读 unknown：`ivr_assembly` 变 `None`，不是 0。

    同时检查 `[lo, hi]`：`lo` 把 unknown 记 0，`hi` 给它体积上界。区间必须存在，
    因为「不知道」的代价就是这条宽度（§4.8 报表给上下界）。
    """
    asm = tmp_path / "asm_bad"
    _write_non_manifold_pair(asm)

    summary = ivr_scan.scan_assembly(asm)

    assert summary["n_unknown_pairs"] == 1
    assert summary["ivr_assembly"] is None, "unknown 绝不能被折成 0"
    assert summary["status"] == "assembly_unknown"
    assert summary["ivr_lo"] == 0.0
    assert summary["ivr_hi"] is not None and summary["ivr_hi"] > 0.0
    assert "open_surface" in summary["unknown_pairs"][0]


@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # 退化三角形求质心会除零，是夹具自身
def test_a_part_without_volume_makes_its_pairs_unknown(tmp_path):
    """没有可信体积的件（这里是一片退化的零面积三角形）让相关对 unknown。

    退化三角形能读成 `Trimesh`（有 1 个面）但体积为 0，所以它走的是
    `part_volume -> None -> no_solid_volume` 这条路，而不是「读不进来」那条。
    """
    asm = tmp_path / "asm_degenerate"
    _write_cubes(asm, [0.0, 0.5])
    flat = trimesh.Trimesh(
        vertices=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        faces=[[0, 1, 2]],
        process=False,
    )
    flat.export(asm / "parts" / "part_002.ply")

    summary = ivr_scan.scan_assembly(asm)

    assert summary["n_parts"] == 3
    assert summary["n_unknown_pairs"] == 2  # part_002 与另两件各一对
    assert summary["ivr_assembly"] is None
    assert all("no_solid_volume" in reason for reason in summary["unknown_pairs"])


def test_a_part_that_cannot_be_loaded_voids_the_whole_assembly(tmp_path):
    """读不进来的件**不许**从分母里静默消失。

    这一条是真踩过的坑：没有面的 PLY 会被 `trimesh` 读成 `PointCloud`，在旧实现里
    它直接被丢掉，剩下的件照常算出一个 0.25，而分母少了那件、配对也少了一整列——
    数字看着完全正常。所以缺陷必须让整条装配 `ivr_assembly = None`。
    """
    asm = tmp_path / "asm_drop"
    _write_cubes(asm, [0.0, 0.5])
    (asm / "parts" / "part_002.ply").write_text(
        "ply\nformat ascii 1.0\nelement vertex 1\n"
        "property float x\nproperty float y\nproperty float z\n"
        "end_header\n0 0 0\n",
        encoding="ascii",
    )

    summary = ivr_scan.scan_assembly(asm)

    assert summary["n_parts_expected"] == 3
    assert summary["parts_dropped"] == 1
    assert summary["status"] == "assembly_unknown"
    assert summary["ivr_assembly"] is None, "缺件的分母算出来的 IVR 是个假数"
    assert summary["ivr_lo"] is None and summary["ivr_hi"] is None


# --------------------------------------------------------------------------- #
# 5. 落盘与续跑
# --------------------------------------------------------------------------- #
def test_scan_writes_rows_and_a_report_end_to_end(tmp_path):
    """一次 `main` 跑两个装配：rows.jsonl 可回读，report.md 把 unknown 单独成节。"""
    root = tmp_path / "geom"
    _write_cubes(root / "asm_000001", [0.0, 0.5])
    _write_non_manifold_pair(root / "asm_000000")
    out = tmp_path / "rows.jsonl"
    report = tmp_path / "report.md"

    code = ivr_scan.main(["--root", str(root), "--out", str(out), "--report", str(report)])
    assert code == 0

    pair_rows, asm_rows, bad = ivr_scan.read_rows(out)
    assert bad == 0
    assert {r["asm"] for r in asm_rows} == {"asm_000000", "asm_000001"}
    measured = {r["asm"]: r for r in asm_rows}
    assert measured["asm_000001"]["ivr_assembly"] == pytest.approx(0.25, abs=1e-9)
    assert measured["asm_000000"]["ivr_assembly"] is None

    text = report.read_text(encoding="utf-8")
    assert "A 类" in text
    assert "未知" in text or "unknown" in text
    # unknown 的数字不许出现在「= 0」档里
    assert "| `asm_000000` | unknown |" in text
    assert "| = 0 |" in text


def test_resume_does_not_recompute_a_completed_assembly(tmp_path):
    """`--resume` 跳过已完成的装配：第二次跑不再追加任何 pair 行。"""
    root = tmp_path / "geom"
    _write_cubes(root / "asm_000001", [0.0, 0.5])
    out = tmp_path / "rows.jsonl"
    report = tmp_path / "report.md"

    first = ["--root", str(root), "--out", str(out), "--report", str(report)]
    assert ivr_scan.main(first) == 0
    before = out.read_text(encoding="utf-8").count("\n")

    assert ivr_scan.main(first + ["--resume"]) == 0
    assert out.read_text(encoding="utf-8").count("\n") == before


def test_a_half_written_assembly_row_is_not_treated_as_complete(tmp_path):
    """只有汇总行、没有逐对行的装配不算完成 —— 否则续跑会留下缺项的分母。"""
    asm_rows = [{"asm": "asm_x", "pairs_exact": 3}]
    assert ivr_scan.completed_assemblies(asm_rows, []) == set()
    partial = [{"asm": "asm_x", "a": "p0", "b": "p1"}]
    assert ivr_scan.completed_assemblies(asm_rows, partial) == set()
    full = [
        {"asm": "asm_x", "a": "p0", "b": f"p{index}"} for index in range(1, 4)
    ]
    assert ivr_scan.completed_assemblies(asm_rows, full) == {"asm_x"}


def test_bad_lines_are_counted_not_fatal(tmp_path):
    """崩在写一半留下的残行只计数，不拖垮整份 rows.jsonl。"""
    out = tmp_path / "rows.jsonl"
    good = json.dumps({"kind": "assembly", "asm": "asm_0", "pairs_exact": 0})
    truncated = '{"kind": "pair", "asm": "asm_0", "a": "p0"'  # 写到一半就崩了
    out.write_text(f"{good}\n{truncated}\n", encoding="utf-8")

    pair_rows, asm_rows, bad = ivr_scan.read_rows(out)

    assert bad == 1
    assert len(asm_rows) == 1
    assert pair_rows == []


def test_last_duplicate_row_wins(tmp_path):
    """同一个键重复出现时以最后一条为准，这样重跑只需追加、不必改文件。"""
    out = tmp_path / "rows.jsonl"
    rows = [
        {"kind": "assembly", "asm": "asm_0", "ivr_assembly": 0.5},
        {"kind": "assembly", "asm": "asm_0", "ivr_assembly": 0.0},
    ]
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    _, asm_rows, _ = ivr_scan.read_rows(out)

    assert len(asm_rows) == 1
    assert asm_rows[0]["ivr_assembly"] == 0.0


# --------------------------------------------------------------------------- #
# 6. 对照：裁剪前原始件（`parts/` 不在子层，PLY 直接摆在目录根下）
# --------------------------------------------------------------------------- #
def test_parts_dir_override_reads_a_flat_directory(tmp_path):
    """`data/objects/<id>/` 的 PLY 摆在根下、没有 `parts/` 子层，必须能扫。"""
    flat = tmp_path / "0000c32fde7f45efb8d14e8ba737d50c"
    flat.mkdir()
    for index, shift in enumerate((0.0, 0.5)):
        mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
        mesh.apply_translation((shift, 0.0, 0.0))
        mesh.export(flat / f"part_{index}.ply")

    summary = ivr_scan.scan_assembly(flat, parts_dir=flat)

    assert summary["n_parts"] == 2
    assert summary["ivr_assembly"] == pytest.approx(0.25, abs=1e-9)


def test_equiv_ratio_translates_a_volume_into_a_length_fraction():
    """1 m³ 的交集体积在 10 m 对角线上 = 0.1 = 10%（等体积立方体边长）。"""
    assert ivr_scan._equiv_ratio(1.0, 10.0) == pytest.approx(0.1)
    for argument in ((None, 1.0), (1.0, None), (0.0, 1.0), (1.0, 0.0)):
        assert ivr_scan._equiv_ratio(*argument) is None


def test_report_inlines_the_cross_check_when_it_exists(tmp_path):
    """有对照数据时报表必须有 §6.1，且 `None` 不许被打成 0。"""
    cross_check = [
        {
            "asm": "asm_000000",
            "object_id": "0" * 32,
            "raw": {
                "status": "ok",
                "n_parts": 10,
                "pairs_exact": 18,
                "pairs_nonzero": 15,
                "ivr_assembly": 0.02888,
                "equiv_edge_over_diagonal": 0.03915,
            },
            "trimmed": {
                "status": "ok",
                "n_parts": 8,
                "pairs_exact": 11,
                "pairs_nonzero": 0,
                "ivr_assembly": 0.0,
                "equiv_edge_over_diagonal": None,
            },
        }
    ]
    asm_rows = [
        {
            "kind": "assembly",
            "asm": "asm_000000",
            "status": "ok",
            "n_parts": 8,
            "parts_dropped": 0,
            "pairs_total": 28,
            "pairs_culled": 17,
            "pairs_exact": 11,
            "n_unknown_pairs": 0,
            "pairs_nonzero": 0,
            "pairs_proven_zero": 28,
            "ivr_assembly": 0.0,
            "ivr_lo": 0.0,
            "ivr_hi": 0.0,
            "ivr_max_pair": 0.0,
            "max_pair": None,
            "max_pair_volume": None,
            "numerator": 0.0,
            "denominator": 1.0,
            "diagonal": 0.16,
            "unknown_pairs": [],
            "unknown_diagnostics": [],
            "part_defects": [],
        }
    ]

    text = ivr_scan.build_report([], asm_rows, 0, cross_check=cross_check)

    assert "### 6.1" in text
    assert "2.888e-02" in text
    assert "3.915%" in text
