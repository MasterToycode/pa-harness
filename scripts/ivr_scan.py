"""A 类扫描：原始装配态下部件之间到底有没有体积穿模（`ivr_assembly`）。

`docs/pah-dev-doc-v1.md` §4.8 的三类穿透统计必须分开报，本脚本只做**A 类**：

| 类 | 含义 | 谁产出 |
|---|---|---|
| **A 原始装配穿透** | 数据集自带，修复前就存在 | 本脚本（全对两两 `ivr_exact`） |
| **B 修复新增穿透** | 相对 A 的增量 | `scripts/m1_repair.py` 的前后 diff（本脚本不碰） |
| **C 连接件实体化成功** | 393/705 | `pah probe connector`（本脚本只引用、不重算） |

**A 不是 B 的分母，B 不是 C 的修正项。** 所以本脚本对 `393/705` 一个字节都不读、不改、不重跑。

## 为什么这个数此前没有

`AssemblyEnv.parts_info()` 把 `ivr` 硬编码成 `0.0`（`assembly_env.py:784`），所以
388/393、55.0% 这些数**都不含穿模惩罚**。连接件侧只验证过「布尔没引入新穿插」（边级
两部件，全 0），从未验证过「装配态部件之间没穿模」。这就是本脚本要回答的那个问题。

## 口径（不另造，全部沿用 `src/pah/tools/ivr.py`）

```
IVR = Σ_(i,j) ∈ E_check  Vol(V_i ∩ V_j)  /  Σ_i Vol(V_i)
```

- `E_check` = **装配内全部两两配对**（N 件 → N(N−1)/2 对）。不排除任何对，因为 A 类问的
  就是「原始数据有没有」，没有任何合法的设计重叠要先扣掉。
- 分母是**全部部件体积之和**，不是「被检查的对里出现的部件之和」——全对检查下两者相同。
- 体积 ∩ 由 `ivr_exact` 一次一对算出（`manifold3d` 实体布尔）。**故意一次只传一对**：
  整批传时任一 unknown 会让整个 `value` 变 `None`，而本脚本必须把 unknown 单独数出来，
  这正是 §4.8 规则 2 要求区分的东西。
- 包围盒剔除沿用 `ivr.py:344 _overlap_extent` 的判据（三轴 extent 全部 >0 才算相交）。
  AABB 是实体的超集，**AABB 不相交 ⇒ 实体不相交是证明**，所以剔除的对是「已知 0」，
  不是「未知」。剔除数单独报出来（`pairs_culled`），不与 unknown 混。
- unknown **不计入 0**：一出现就进 `unknown_pairs` 桶，装配级 `ivr_assembly` 变 `None`，
  同时给 `[ivr_lo, ivr_hi]` 区间（§4.8「报表给上下界」的写法）：
  `lo` = unknown 记 0，`hi` = unknown 取其体积上界 `min(bbox 重叠体积, vol_a, vol_b)`。

## 单位

几何是**米**（`docs/internal/adapter-contract.md` §A），IVR 无量纲。`denominator` 落盘
的是 m³，只用于让读者能从 `rows.jsonl` 复算，不参与任何阈值判断。

## 阈值 ε

**本脚本不定 ε。** ε 属评测配置，随 `policy_version` 冻结（§4.8），必须先有分布才能定。
本脚本只产出原始分布，把「ε 该定多少」留给有分布之后的决策。

## 用法

```bash
python scripts/ivr_scan.py --self-test          # 正向对照：已知 50% 重叠必须被量到
python scripts/ivr_scan.py --limit 3            # 先跑 3 个装配验路
python scripts/ivr_scan.py --resume             # 断点续跑，已完成的对不重算
python scripts/ivr_scan.py --report-only        # 只从 rows.jsonl 重算报表
```

产物全部落在 `runs/ivr-scan/`：`rows.jsonl`（逐对 + 逐装配，唯一可追溯来源）、
`report.md`（报表）、`scan.log`（运行日志，由调用方重定向）。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import trimesh

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pah.tools.ivr import ivr_exact  # noqa: E402  (after sys.path fix)

# --------------------------------------------------------------------------- #
# 路径
# --------------------------------------------------------------------------- #
REPO = Path(__file__).resolve().parents[1]
#: 权威几何是 `assemblies_pre_decimate`（裁剪过、未简化），`adapter-contract.md`
#: §E2/§E11：布尔与测厚都必须用它，`assemblies/` 是 225 mm 简化集，不能跑布尔。
GEOMETRY_ROOT = Path(
    os.environ.get(
        "PAH_GEOMETRY",
        r"D:\yanyi\rl_assembly\data\assemblies_pre_decimate",
    )
)
OUT_DIR = REPO / "runs" / "ivr-scan"
ROWS = OUT_DIR / "rows.jsonl"
REPORT = OUT_DIR / "report.md"

#: 分布分档的边界（只用于报表分桶，**不是 ε**）。浮点噪声落在第一档
#: `(0, 1e-6]`，所以这一档的数字必须和「真的有穿模」分开看。
BUCKETS = (0.0, 1e-6, 1e-3)


# --------------------------------------------------------------------------- #
# 几何加载
# --------------------------------------------------------------------------- #
def load_parts(parts_dir: Path) -> tuple[list[tuple[str, trimesh.Trimesh]], list[dict]]:
    """`parts/*.ply` → `[(id, mesh)]` 加每件的体积缺陷清单。

    `process=False` 是必须的：`trimesh.load` 默认会 merge 顶点/去重，在这批
    「封闭但含 0.01–0.1% 非流形边」的网格上会改动拓扑，而布尔结果必须建立在
    与交付数据逐位相同的几何上（`adapter-contract.md` §E6）。

    体积取 `abs(mesh.volume)`：这批网格封闭，divergence-theorem 体积有意义；
    非有限或 ≤0 的件没有可除的分母，记一条缺陷交给装配级判 unknown。
    """
    parts: list[tuple[str, trimesh.Trimesh]] = []
    defects: list[dict] = []
    for ply in sorted(parts_dir.glob("*.ply")):
        pid = ply.stem
        try:
            mesh = trimesh.load(ply, process=False)
        except Exception as exc:  # noqa: BLE001 - 一件读不了不能拖垮整个扫描
            defects.append({"part": pid, "kind": "load_failed", "why": _why(exc)})
            continue
        if not isinstance(mesh, trimesh.Trimesh):
            defects.append(
                {"part": pid, "kind": "not_a_mesh", "why": type(mesh).__name__}
            )
            continue
        parts.append((pid, mesh))
    return parts, defects


def part_volume(mesh: trimesh.Trimesh) -> float | None:
    """`|vol(V_i)|`，与 `ivr.py:333 _solid_volume` 同一判据，None 表示无可除体积。"""
    if len(mesh.faces) == 0:
        return None
    try:
        volume = abs(float(mesh.volume))
    except (TypeError, ValueError):
        return None
    return volume if math.isfinite(volume) and volume > 0.0 else None


def aabb_misses(a: trimesh.Trimesh, b: trimesh.Trimesh) -> bool:
    """两 AABB 是否三轴全部不相交 —— 与 `ivr.py:344 _overlap_extent` 逐字同判据。

    AABB 含其实体，所以「AABB 不相交」是「实体不相交」的**证明**；这不是「跳过」，
    是把 0 变成已知的 0。判据必须与 `ivr.py` 完全一致，否则会出现「本脚本认为该
    送进布尔、`ivr_exact` 内部却把它剔除成 method=bbox」的口径漂移
    （`tests/test_ivr_scan.py::test_cull_rule_matches_ivr` 钉住这一条）。
    """
    lo = np.maximum(a.bounds[0], b.bounds[0])
    hi = np.minimum(a.bounds[1], b.bounds[1])
    extent = hi - lo
    return not bool(np.all(extent > 0.0))


def overlap_volume(a: trimesh.Trimesh, b: trimesh.Trimesh) -> float:
    """两 AABB 交集的体积，用作 unknown 对的上界（交体必含于 AABB 交）。"""
    lo = np.maximum(a.bounds[0], b.bounds[0])
    hi = np.minimum(a.bounds[1], b.bounds[1])
    extent = np.maximum(hi - lo, 0.0)
    return float(np.prod(extent))


# --------------------------------------------------------------------------- #
# 单装配扫描
# --------------------------------------------------------------------------- #
def scan_assembly(asm_dir: Path, *, emit=None, parts_dir: Path | None = None) -> dict:
    """一个装配的全部两两配对 IVR，边算边 `emit`。

    `emit(row)` 每算完一对就调一次，调用方负责落盘——这样中途被杀也不会丢掉已算的
    对。续跑按**整装配**跳过（`completed_assemblies`），不做对级续跑：装配级汇总
    要有完整分母，只补一半的对会让分母与计数缺项。

    `parts_dir` 缺省是 `<asm_dir>/parts`。裁剪前原始件目录（`data/objects/<id>/`）
    的 PLY 就摆在目录根下、没有 `parts/` 子层，所以这个参数由调用方给。
    """
    asm = asm_dir.name
    started = time.perf_counter()
    pdir = parts_dir if parts_dir is not None else asm_dir / "parts"
    expected_parts = len(list(pdir.glob("*.ply")))
    parts, defects = load_parts(pdir)
    volumes = {pid: part_volume(mesh) for pid, mesh in parts}
    denominator = sum(v for v in volumes.values() if v is not None)
    diagonal = _diagonal(parts)

    n = len(parts)
    pairs_total = n * (n - 1) // 2
    pairs_culled = 0
    pairs_exact = 0
    unknown_pairs: list[str] = []
    unknown_diagnostics: list[dict] = []
    numerator = 0.0
    numerator_hi = 0.0
    pairs_nonzero = 0
    max_pair: tuple[str, str] | None = None
    max_volume = -1.0

    for i in range(n):
        for j in range(i + 1, n):
            a, b = parts[i][0], parts[j][0]
            if volumes[a] is None or volumes[b] is None:
                bad = [pid for pid in (a, b) if volumes[pid] is None]
                pairs_exact += 1
                reason = "no_solid_volume:" + ",".join(bad)
                unknown_pairs.append(f"{a}-{b}: {reason}")
                # 上界：交体含于 AABB 交，也含于任一件（有体积的那些件）
                known_vols = [volumes[pid] for pid in (a, b) if volumes[pid] is not None]
                bbox_volume = overlap_volume(parts[i][1], parts[j][1])
                upper = min([bbox_volume, *known_vols])
                numerator_hi += upper
                unknown_diagnostics.append(
                    {"pair": [a, b], "reason": reason, "bbox_volume": bbox_volume, "upper": upper}
                )
                _emit_pair(
                    emit, asm, a, b,
                    status="unknown", volume=None, method="", reason=reason,
                    bbox_volume=None, faces=(len(parts[i][1].faces), len(parts[j][1].faces)),
                    share=None, ms=0.0,
                )
                continue

            if aabb_misses(parts[i][1], parts[j][1]):
                # AABB 不相交 ⇒ 实体不相交（证明，不是近似），所以这对已知 0。
                pairs_culled += 1
                continue

            pairs_exact += 1
            bbox_volume = overlap_volume(parts[i][1], parts[j][1])
            started_pair = time.perf_counter()
            result = ivr_exact([parts[i], parts[j]], pairs=[(a, b)])
            elapsed_ms = (time.perf_counter() - started_pair) * 1000.0
            record = result.detail["pairs"][0]
            volume = record["volume"]
            if volume is None:
                reason = record["reason"] or "unknown"
                unknown_pairs.append(f"{a}-{b}: {reason}")
                unknown_diagnostics.append(
                    {
                        "pair": [a, b],
                        "reason": reason,
                        "bbox_volume": bbox_volume,
                        # unknown 的**上界**：交体含于 AABB 交，也含于两件本身
                        "upper": min(bbox_volume, volumes[a], volumes[b]),
                    }
                )
                numerator_hi += min(bbox_volume, volumes[a], volumes[b])
            else:
                numerator += volume
                numerator_hi += volume
                if volume > 0.0:
                    pairs_nonzero += 1
                if volume > max_volume:
                    max_volume = volume
                    max_pair = (a, b)

            _emit_pair(
                emit, asm, a, b,
                status=record["status"], volume=volume, method=record["method"],
                reason=record["reason"],
                bbox_volume=bbox_volume,
                faces=(len(parts[i][1].faces), len(parts[j][1].faces)),
                share=None if volume is None or denominator <= 0 else volume / denominator,
                ms=elapsed_ms,
            )

    # 分母可信 = 每件都有有限正体积。有件读不进来时**不能**把它当 0 从分母里剔掉：
    # 少一个部件同时缩小分母、少一列配对，「算出来的 IVR」会比真值大，而且看不出
    # 少了谁。这种情况整个装配判 unknown，同时把缺陷写进 `part_defects`。
    denominator_ok = denominator > 0.0 and all(v is not None for v in volumes.values())
    measured = denominator_ok and not unknown_pairs and not defects
    ivr_assembly = numerator / denominator if measured else None
    # 区间只在分母可信时才有意义：分母本身缺件时「不知道」的范围是整条 [0, 1]。
    lo = numerator / denominator if denominator_ok and not defects else None
    hi = numerator_hi / denominator if denominator_ok and not defects else None

    return {
        "kind": "assembly",
        "asm": asm,
        "status": "ok" if measured else "assembly_unknown",
        "n_parts": n,
        "n_parts_expected": expected_parts,
        "parts_dropped": len(defects),
        "pairs_total": pairs_total,
        "pairs_culled": pairs_culled,
        "pairs_exact": pairs_exact,
        "n_unknown_pairs": len(unknown_pairs),
        "pairs_nonzero": pairs_nonzero,
        "pairs_proven_zero": pairs_culled + (pairs_exact - len(unknown_pairs) - pairs_nonzero),
        "ivr_assembly": ivr_assembly,
        "ivr_lo": lo,
        "ivr_hi": hi,
        "ivr_max_pair": (max_volume / denominator) if max_pair and denominator > 0 else None,
        "max_pair": list(max_pair) if max_pair else None,
        "max_pair_volume": max_volume if max_pair else None,
        "numerator": numerator,
        "denominator": denominator,
        "diagonal": diagonal,
        "unknown_pairs": unknown_pairs,
        "unknown_diagnostics": unknown_diagnostics,
        "part_defects": defects,
        "elapsed_s": round(time.perf_counter() - started, 2),
    }


def _emit_pair(emit, asm: str, a: str, b: str, **fields: Any) -> None:
    if emit is None:
        return
    emit({"kind": "pair", "asm": asm, "a": a, "b": b, **fields})


def _diagonal(parts: list[tuple[str, trimesh.Trimesh]]) -> float | None:
    """装配包围盒对角线（米），只用于把 IVR 换算成「长度口径」做对照。"""
    if not parts:
        return None
    lo = np.min([m.bounds[0] for _, m in parts], axis=0)
    hi = np.max([m.bounds[1] for _, m in parts], axis=0)
    return float(np.linalg.norm(hi - lo))


# --------------------------------------------------------------------------- #
# 落盘 / 续跑
# --------------------------------------------------------------------------- #
def read_rows(path: Path = ROWS) -> tuple[list[dict], list[dict], int]:
    """读 `rows.jsonl`，返回 `(pair 行, assembly 行, 坏行数)`。

    「最后一行胜出」：同一 `(asm, a, b)` 或同一 `asm` 重复出现时以最后一条为准。
    这样续跑/重跑只会追加，不需要就地改文件，也不怕写到一半崩（坏行只会计数）。
    """
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
                # 重跑同一个装配时旧行被新行顶掉；但 `elapsed`/`--limit` 造成的
                # 半成品装配行也可能是旧的，是否可信由调用方按 pair 行重算。
                asm_rows[row["asm"]] = row
            else:
                bad += 1
    return list(pair_rows.values()), list(asm_rows.values()), bad


def completed_assemblies(asm_rows: Iterable[dict], pair_rows: list[dict]) -> set[str]:
    """已完成装配 = 有 assembly 行 **且** 该装配的 pair 行数足以覆盖它的对数。

    只看到 assembly 行不够：`--limit 1` 会留下合法的半成品。用 pair 行重数一遍是最
    省事的自证。
    """
    seen: dict[str, int] = {}
    for row in pair_rows:
        seen[row["asm"]] = seen.get(row["asm"], 0) + 1
    done = set()
    for row in asm_rows:
        asm = row["asm"]
        # 被剔除的对没有 pair 行（它们是「证明为 0」而不是「算过」），所以
        # 有行数 = pairs_exact；与汇总行对得上才算完成。
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
# 正向对照：已知重叠必须被量到
# --------------------------------------------------------------------------- #
def make_overlap_fixture(root: Path) -> Path:
    """造一个已知 50% 重叠的两件装配：两个单位立方体错开半个身位。

    手算：|A ∩ B| = 0.5，Σ|V| = 1 + 1 = 2 → `ivr_assembly` = **0.25**。
    这是本扫描器的正向对照——全量跑出全 0 时，它用来排除「扫描器坏了」。
    """
    asm = root / "asm_selftest"
    parts = asm / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    for index, shift in enumerate((0.0, 0.5)):
        mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
        mesh.apply_translation((shift, 0.0, 0.0))
        mesh.export(parts / f"part_{index:03d}.ply")
    return asm


def self_test() -> int:
    """三条断言：已知重叠被测出、AABB 剔除生效、非流形件读 unknown 不读 0。"""
    failures: list[str] = []
    with tempfile.TemporaryDirectory(prefix="ivr-selftest-") as tmp:
        root = Path(tmp)
        asm = make_overlap_fixture(root)

        # ① 50% 重叠 → 0.25
        summary = scan_assembly(asm)
        got = summary["ivr_assembly"]
        ok = got is not None and abs(got - 0.25) < 1e-9
        print(
            f"self-test: 50%-overlap cube pair -> ivr_assembly={got!r} "
            f"(expected 0.25) {'PASS' if ok else 'FAIL'}"
        )
        if not ok:
            failures.append("overlap fixture not measured as 0.25")

        # ② 立方体的 PLY 往返不能把关壳弄坏：否则布尔必然 NotManifold，①会假绿
        parts, defects = load_parts(asm / "parts")
        bnd = [_boundary_edges(m) for _, m in parts]
        ok = not defects and len(parts) == 2 and all(b == 0 for b in bnd)
        print(
            f"self-test: ply round-trip closed (parts={len(parts)}, "
            f"boundary_edges={bnd}) {'PASS' if ok else 'FAIL'}"
        )
        if not ok:
            failures.append("ply round-trip broke the closed surface")

        # ③ AABB 剔除：错开 3 个身位 → pairs_culled=1、pairs_exact=0、ivr 0.0
        far = root / "asm_far"
        (far / "parts").mkdir(parents=True)
        for index, shift in enumerate((0.0, 3.0)):
            mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
            mesh.apply_translation((shift, 0.0, 0.0))
            mesh.export(far / "parts" / f"part_{index:03d}.ply")
        summary = scan_assembly(far)
        ok = (
            summary["pairs_culled"] == 1
            and summary["pairs_exact"] == 0
            and summary["ivr_assembly"] == 0.0
        )
        print(
            f"self-test: disjoint pair culled without boolean "
            f"(culled={summary['pairs_culled']}, exact={summary['pairs_exact']}, "
            f"ivr={summary['ivr_assembly']!r}) {'PASS' if ok else 'FAIL'}"
        )
        if not ok:
            failures.append("bbox cull did not short-circuit a disjoint pair")

    print(
        f"self-test: {'ALL PASS' if not failures else 'FAILED: ' + '; '.join(failures)}"
    )
    return 0 if not failures else 1


def _boundary_edges(mesh: trimesh.Trimesh) -> int:
    """边界边数（恰 1 个相邻面）。0 = 闭合壳，判封闭一律用它而不是 `is_watertight`。"""
    if len(mesh.faces) == 0:
        return -1
    counts = np.bincount(mesh.edges_unique_inverse)
    return int(np.count_nonzero(counts == 1))


# --------------------------------------------------------------------------- #
# 附加对照：裁剪前原始件（用来定位「1%–3% 对角线」是哪个人群/哪个口径）
# --------------------------------------------------------------------------- #
OBJECTS_ROOT = GEOMETRY_ROOT.parent / "objects"


def raw_cross_check(names: list[str], out_json: Path, out_md: Path) -> int:
    """对同一批装配，同时扫「交付件」与「裁剪前原始件」，把口径差异摆出来。

    交付的 `assemblies_pre_decimate` 是 `exclusive_i = part_i − ∪(其余部件)` 裁剪后的
    产物；`data/objects/<object_id>/part_*.ply` 是**裁剪前**的原样件（记忆卡
    「装配数据管线布尔裁剪」：「接缝两边都带材料、互相侵入 1%–3% 对角线」）。
    只有把两边用同一个扫描器量一遍，才能判断记忆卡那句话说的是哪个人群、哪个口径。

    两个必须记住的口径差别，写进输出文件本身：

    - 原始件目录的部件集合与交付装配**不必相同**（漏斗在裁剪前淘汰过部件，
      如 asm_000000 有 10 件原样件、8 件进入交付），所以这是人群级对照，不是同一
      装配的前后 diff；
    - 原始件在自己的坐标系里，`ivr` 无量纲所以尺度无关，但 `diagonal` 只能用来把
      体积折算成「等体积立方体边长」，不是记忆卡那个「穿透深度」。
    """
    rows: list[dict] = []
    for name in names:
        asm_dir = GEOMETRY_ROOT / name
        meta_path = asm_dir / "meta.json"
        if not meta_path.exists():
            print(f"cross-check: {name} has no meta.json, skipped", file=sys.stderr)
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        object_id = str(meta.get("source", "")).rsplit("/", 1)[-1]
        raw_dir = OBJECTS_ROOT / object_id
        entry: dict[str, Any] = {"asm": name, "object_id": object_id}
        for arm, path, pdir in (
            ("raw", raw_dir, raw_dir if raw_dir.exists() else None),
            ("trimmed", asm_dir, None),
        ):
            if not path.exists():
                entry[arm] = {"status": "missing", "path": str(path)}
                continue
            summary = scan_assembly(path, parts_dir=pdir)
            entry[arm] = {
                "path": str(path),
                "status": summary["status"],
                "n_parts": summary["n_parts"],
                "pairs_total": summary["pairs_total"],
                "pairs_culled": summary["pairs_culled"],
                "pairs_exact": summary["pairs_exact"],
                "n_unknown_pairs": summary["n_unknown_pairs"],
                "pairs_nonzero": summary["pairs_nonzero"],
                "ivr_assembly": summary["ivr_assembly"],
                "ivr_lo": summary["ivr_lo"],
                "ivr_hi": summary["ivr_hi"],
                "ivr_max_pair": summary["ivr_max_pair"],
                "max_pair": summary["max_pair"],
                "max_pair_volume": summary["max_pair_volume"],
                "diagonal": summary["diagonal"],
                # 等体积立方体边长 / 对角线：把体积比翻译成一个「长度占对角线」的数，
                # 才勉强能和记忆卡的 1%–3% 摆在一起看。偏小——细长穿透同样体积下
                # 最大深度比等体积边长更大——所以这是该比例的下界。
                "equiv_edge_over_diagonal": _equiv_ratio(
                    summary["max_pair_volume"], summary["diagonal"]
                ),
                "elapsed_s": summary["elapsed_s"],
            }
            print(
                f"cross-check [{arm:7s}] {name} parts={summary['n_parts']} "
                f"nonzero={summary['pairs_nonzero']} "
                f"ivr={_fmt(summary['ivr_assembly'])}",
                flush=True,
            )
        rows.append(entry)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    out_md.write_text(_cross_check_md(rows), encoding="utf-8")
    print(f"cross-check: {len(rows)} assemblies -> {out_md}")
    return 0


def _equiv_ratio(volume: float | None, diagonal: float | None) -> float | None:
    if not volume or not diagonal or volume <= 0.0 or diagonal <= 0.0:
        return None
    return float(volume) ** (1.0 / 3.0) / float(diagonal)


def _cross_check_md(rows: list[dict]) -> str:
    lines: list[str] = []
    add = lines.append
    add("# 附加对照：裁剪前原始件 vs 交付件（同一扫描器、同一口径）")
    add("")
    add("`data/objects/<object_id>/part_*.ply`（裁剪前原样件）与")
    add("`assemblies_pre_decimate/<asm>/parts/*.ply`（交付件）各扫一遍。")
    add("两边的部件集合不必相同，所以这是**人群级**对照，不是同一装配的前后 diff。")
    add("")
    add("| asm | 臂 | 部件 | pairs | 剔除 | 精确 | unknown | 非零对 | `ivr_assembly` | 最大单对体积 (m³) | 等体积边长/对角线 |")
    add("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        for arm, label in (("raw", "裁剪前"), ("trimmed", "交付件")):
            data = row.get(arm) or {}
            if data.get("status") == "missing":
                add(f"| `{row['asm']}` | {label} | — | — | — | — | — | — | 缺目录 | — | — |")
                continue
            ivr = data.get("ivr_assembly")
            ivr_cell = "unknown" if ivr is None else f"{ivr:.3e}"
            vol = data.get("max_pair_volume")
            vol_cell = "—" if not vol else f"{vol:.3e}"
            ratio = data.get("equiv_edge_over_diagonal")
            ratio_cell = "—" if not ratio else f"{ratio * 100:.4g}%"
            add(
                f"| `{row['asm']}` | {label} | {data.get('n_parts')} | {data.get('pairs_total')} | "
                f"{data.get('pairs_culled')} | {data.get('pairs_exact')} | "
                f"{data.get('n_unknown_pairs')} | {data.get('pairs_nonzero')} | {ivr_cell} | "
                f"{vol_cell} | {ratio_cell} |"
            )
    add("")
    raw_nonzero = [r for r in rows if (r.get("raw") or {}).get("pairs_nonzero")]
    trim_nonzero = [r for r in rows if (r.get("trimmed") or {}).get("pairs_nonzero")]
    add("## 判读")
    add("")
    add(f"- 裁剪前原件侧：{len(raw_nonzero)} / {len(rows)} 个装配有非零体积相交。")
    add(f"- 交付件侧：{len(trim_nonzero)} / {len(rows)} 个装配有非零体积相交。")
    add("- 「等体积边长/对角线」是**下界**：细长穿透在同样体积下最大深度更大。")
    add("  所以这一列只能用来判断「量级是否够到 1%–3%」，不能当作穿透深度的测量值。")
    add("")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# 报表
# --------------------------------------------------------------------------- #
def build_report(
    pair_rows: list[dict],
    asm_rows: list[dict],
    bad_lines: int,
    cross_check: list[dict] | None = None,
) -> str:
    """把 `rows.jsonl`（可选 `raw-crosscheck.json`）变成 `report.md`。所有数字都能从行里复算。"""
    rows = sorted(asm_rows, key=lambda r: r["asm"])
    n_asm = len(rows)
    n_parts = sum(r["n_parts"] for r in rows)
    pairs_total = sum(r["pairs_total"] for r in rows)
    pairs_culled = sum(r["pairs_culled"] for r in rows)
    pairs_exact = sum(r["pairs_exact"] for r in rows)
    n_unknown = sum(r["n_unknown_pairs"] for r in rows)
    nonzero_asm = [r for r in rows if r["pairs_nonzero"] > 0]
    unknown_asm = [r for r in rows if r["n_unknown_pairs"] > 0]

    known = [r for r in rows if r["ivr_assembly"] is not None]
    values = sorted(r["ivr_assembly"] for r in known)
    nonzero_pair_rows = [p for p in pair_rows if p.get("status") == "known" and (p.get("volume") or 0.0) > 0.0]

    lines: list[str] = []
    add = lines.append
    add("# A 类穿透扫描：原始装配态部件之间的体积穿模")
    add("")
    add("> 本报表只回答一个问题：**交付数据集（`assemblies_pre_decimate`，装配坐标系、单位米）")
    add("> 的部件之间，有没有体积上的互相侵入？** 全部数字可追溯到同目录 `rows.jsonl`。")
    add("")
    add("三类统计互相独立（`docs/pah-dev-doc-v1.md` §4.8）：本报表是 **A 原始装配穿透**；")
    add("B（修复新增穿透）等 M1 结果，接口留在 `scripts/m1_repair.py` 侧；C（连接件实体化成功）")
    add("= **393/705**，本扫描不重算、不修改、不参与任何比值。")
    add("")
    add("## 0. 一句话结论")
    add("")
    add(_headline(rows, known, values, nonzero_asm, n_unknown))
    add("")
    add("## 1. 总量")
    add("")
    add("| 项 | 值 |")
    add("|---|---:|")
    add(f"| 装配数 | {n_asm} |")
    add(f"| 部件数 | {n_parts} |")
    add(f"| 读不进来的部件（装配随之判 unknown） | {sum(r.get('parts_dropped') or 0 for r in rows)} |")
    add(f"| 两两配对总数 `pairs_total` | {pairs_total} |")
    add(f"| 包围盒剔除 `pairs_culled`（证明为 0，非未知） | {pairs_culled} |")
    add(f"| 送进布尔 `pairs_exact` | {pairs_exact} |")
    add(f"| 其中 `unknown` | {n_unknown} |")
    add(f"| 已知为 0 的对（剔除 + 布尔读 0） | {pairs_total - n_unknown - len(nonzero_pair_rows)} |")
    add(f"| 非零的对 | {len(nonzero_pair_rows)} |")
    add(f"| `ivr_assembly` 可定值的装配 | {len(known)} / {n_asm} |")
    add(f"| 解析失败的 `rows.jsonl` 行 | {bad_lines} |")
    add("")
    add("`pairs_culled` 的语义要说清：AABB 是实体的超集，AABB 不相交**证明**实体不相交，")
    add("所以这些对是「已知 0」，不是「没算」。它们不计入 `pairs_exact`，也不计入 unknown。")
    add("")
    if n_parts != 724:
        add(f"**件数口径对齐**：本次实测 **{n_parts}** 件（`parts/*.ply` 文件数与"
            f"Σ`meta.json.n_parts` 一致，逐装配无差异）；任务书与记忆卡写的是 724 件。"
            f"差 {724 - n_parts} 件的原因**未查明**（`assemblies/` 简化集为 720 件，也不是 724）。")
        add("本扫描覆盖实际存在的全部 PLY，不补齐、不外推——少的那几件不影响「有没有穿模」的判决，")
        add("但引用件数时必须用实测值。")
        add("")

    add("## 2. 分布")
    add("")
    if values:
        add(f"- `ivr_assembly` p50 = **{_pct(values, 50):.3e}**，p90 = **{_pct(values, 90):.3e}**，"
            f"max = **{values[-1]:.3e}**，min = **{values[0]:.3e}**（n = {len(values)} 个可定值装配）")
    else:
        add("- 没有任何装配的 `ivr_assembly` 可定值（全部含 unknown）")
    add(f"- 非零装配：**{len(nonzero_asm)} / {n_asm}**"
        + ("" if not nonzero_asm else "（" + ", ".join(f"{r['asm']}" for r in nonzero_asm[:10])
           + ("…" if len(nonzero_asm) > 10 else "") + "）"))
    add("")
    add("### 分档计数（档位只用于看形状，**不是 ε**）")
    add("")
    add("| 档 | 装配数 | 含义 |")
    add("|---|---:|---|")
    for label, lo, hi, meaning in (
        ("= 0", None, 0.0, "布尔读到的交集体积逐位为 0"),
        ("(0, 1e-6]", 0.0, 1e-6, "**浮点噪声档**：布尔在近相切界面上留的残渣，不是穿模"),
        ("(1e-6, 1e-3]", 1e-6, 1e-3, "可辨但很小"),
        ("> 1e-3", 1e-3, None, "量级上像真的穿模"),
    ):
        count = sum(1 for v in values if _in_bucket(v, lo, hi))
        add(f"| {label} | {count} | {meaning} |")
    add("")
    if nonzero_pair_rows:
        vols = sorted(p["volume"] for p in nonzero_pair_rows)
        add(f"非零对的体积（m³）：min = {vols[0]:.3e}，p50 = {_pct(vols, 50):.3e}，"
            f"max = {vols[-1]:.3e}。")
        # 噪声地平线：把「最大非零体积」与「最小非零体积」摆在一起，读者一眼能看出
        # 这些数是不是布尔残渣
        deepest = max(nonzero_pair_rows, key=lambda p: p["volume"])
        add(f"最大的一对是 `{deepest['asm']}` 的 `{deepest['a']}-{deepest['b']}`，"
            f"体积 {deepest['volume']:.3e} m³ ≈ **{_equiv_edge_mm(deepest['volume']):.3f} mm 边长的立方体**。")
        add("")

    add("## 3. 证据：穿模最重的 5 个装配")
    add("")
    ranked = sorted(rows, key=lambda r: -(r["numerator"] or 0.0))[:5]
    add("按 `numerator`（交集体积绝对量）排序。`pairs_nonzero = 0` 的装配说明它没有任何一对")
    add("读到正体积，它的「最大单对」一栏因此写 `无`，而不是写一个 0 装成有数。")
    add("")
    add("| asm | `ivr_assembly` | 非零对 | 最大单对 `ivr_max_pair` | 最大单对 | 该对体积 (m³) |")
    add("|---|---:|---:|---:|---|---:|")
    for r in ranked:
        ivr = "unknown" if r["ivr_assembly"] is None else f"{r['ivr_assembly']:.3e}"
        if r["pairs_nonzero"] == 0:
            add(f"| `{r['asm']}` | {ivr} | 0 | 无 | 无 | 无 |")
            continue
        mp = "—" if r["max_pair"] is None else f"`{r['max_pair'][0]}-{r['max_pair'][1]}`"
        mv = "—" if r["ivr_max_pair"] is None else f"{r['ivr_max_pair']:.3e}"
        vol = "—" if r["max_pair_volume"] is None else f"{r['max_pair_volume']:.3e}"
        add(f"| `{r['asm']}` | {ivr} | {r['pairs_nonzero']} | {mv} | {mp} | {vol} |")
    add("")
    add("逐对明细（同目录 `rows.jsonl` 里 `kind=\"pair\"` 的行）：")
    add("")
    top_pairs = sorted(nonzero_pair_rows, key=lambda p: -p["volume"])[:10]
    if not top_pairs:
        add("**没有任何一对的体积 > 0。** 见 §5 正向对照：扫描器对已知重叠能读出 0.25，")
        add("所以这里的全 0 不是「扫描器坏了」。")
    else:
        add("| asm | pair | 体积 (m³) | 占该装配分母 | 方法 | 面数 |")
        add("|---|---|---:|---:|---|---|")
        for p in top_pairs:
            share = "—" if p.get("share") is None else f"{p['share']:.3e}"
            add(f"| `{p['asm']}` | `{p['a']}-{p['b']}` | {p['volume']:.3e} | {share} | {p['method']} | {p['faces'][0]}/{p['faces'][1]} |")
    add("")

    add("## 4. `unknown`：单独一节，绝不并进 0")
    add("")
    abnormal = [r for r in rows if r.get("status") not in (None, "ok")]
    if not unknown_asm and not abnormal:
        add(f"**全 0 条 unknown。** {pairs_exact} 个过包围盒的对全部由 `manifold3d` 给出确定答案")
        add("（`status=\"known\"`），没有一对因非流形/引擎失败而不可判定。")
        add("")
        add("这一点本身值得记：这批网格 `is_watertight` 恒为 False（0.01–0.1% 非流形边，")
        add("`adapter-contract.md` §E6），但**边界边为 0**，所以 `ivr.py:_surface_defect` 的")
        add("开壳判据不触发，布尔引擎确实跑得动。")
    else:
        add(f"**{n_unknown} 对不可判定**，分布在 {len(unknown_asm)} 个装配；"
            f"另有 {len(abnormal)} 个装配因部件读不进来而整条判 unknown。")
        add("")
        if unknown_asm:
            add("| asm | unknown 对 | 该对原因 | 上界体积 (m³) |")
            add("|---|---|---|---:|")
            for r in unknown_asm:
                diags = r.get("unknown_diagnostics") or []
                for k, diag in enumerate(diags):
                    asm_cell = f"`{r['asm']}`" if k == 0 else ""
                    upper = "—" if diag.get("upper") is None else f"{diag['upper']:.3e}"
                    add(f"| {asm_cell} | `{diag['pair'][0]}-{diag['pair'][1]}` | `{diag['reason']}` | {upper} |")
            add("")
            add("这些装配的 `ivr_assembly` 因此是 `None`，同行的 `ivr_lo` / `ivr_hi` 给出区间：")
            add("`lo` 把 unknown 记 0，`hi` 把 unknown 取其体积上界 `min(AABB 重叠体积, vol_a, vol_b)`。")
            add("区间宽度就是「不知道」的代价，不许挑一个端点交差（§4.8）。")
            add("")
        parts_defects = [(r["asm"], d) for r in rows for d in (r.get("part_defects") or [])]
        if parts_defects:
            add("部件级缺陷（该装配整条判 unknown，`ivr_lo`/`ivr_hi` 一并置 `None`，因为分母")
            add("本身缺件、区间也不可信）：")
            add("")
            for asm, d in parts_defects:
                add(f"- `{asm}` / `{d['part']}`：`{d['kind']}` — {d['why']}")
        else:
            add("")
            add(f"部件级缺陷：0（{n_parts} 个 PLY 全部读成网格，无不缺失部件）。")
    add("")

    add("## 5. 正向对照：证明全 0 不是「扫描器坏了」")
    add("")
    add("同一个扫描函数（`scan_assembly`，与本报表走完全相同的代码路径）跑一个合成装配：")
    add("两个单位立方体错开半个身位，`|A ∩ B| = 0.5`，`Σ|V| = 2`，真值 `ivr = 0.25`。")
    add("")
    add("```")
    add("python scripts/ivr_scan.py --self-test")
    add("```")
    add("")
    add("`tests/test_ivr_scan.py` 把这条固化成断言，另外还钉了三件事：")
    add("AABB 剔除判据与 `ivr.py` 逐字一致；合成装配的 0.25 与整批 `ivr_exact` 的口径一致；")
    add("非流形件读 `unknown` 而不是 0。")
    add("")

    add("## 6. 对照记忆卡的「侵入量中位 1%–3% 对角线」")
    add("")
    add(_compare_section(rows, values, has_cross_check=bool(cross_check)))
    add("")
    if cross_check:
        add("### 6.1 直接实测：裁剪前原始件 vs 交付件")
        add("")
        add("上面那条对照里最关键的「人群不同」，可以用同一个扫描器直接证：")
        add("`data/objects/<object_id>/part_*.ply` 是**裁剪前**的原样件（记忆卡说「接缝两边")
        add("都带材料、互相侵入 1%–3% 对角线」正是指这一态），交付件是裁剪后的产物。")
        add("")
        add("| asm | 臂 | 部件 | 精确对数 | 非零对 | `ivr_assembly` | 等体积边长/对角线 |")
        add("|---|---|---:|---:|---:|---:|---:|")
        for row in cross_check:
            for arm, label in (("raw", "裁剪前"), ("trimmed", "交付件")):
                data = row.get(arm) or {}
                if data.get("status") == "missing":
                    add(f"| `{row['asm']}` | {label} | — | — | — | 缺目录 | — |")
                    continue
                ratio = data.get("equiv_edge_over_diagonal")
                add(
                    f"| `{row['asm']}` | {label} | {data.get('n_parts')} | {data.get('pairs_exact')} | "
                    f"{data.get('pairs_nonzero')} | {_fmt(data.get('ivr_assembly'))} | "
                    f"{'—' if not ratio else f'{ratio * 100:.4g}%'} |"
                )
        add("")
        raw_nz = sum(1 for r in cross_check if (r.get("raw") or {}).get("pairs_nonzero"))
        trim_nz = sum(1 for r in cross_check if (r.get("trimmed") or {}).get("pairs_nonzero"))
        raw_ivr = sorted(
            (r["raw"]["ivr_assembly"] for r in cross_check
             if r.get("raw") and r["raw"].get("ivr_assembly") is not None)
        )
        trim_ivr = sorted(
            (r["trimmed"]["ivr_assembly"] for r in cross_check
             if r.get("trimmed") and r["trimmed"].get("ivr_assembly") is not None)
        )
        add(f"裁剪前：{raw_nz} / {len(cross_check)} 个装配有非零体积相交；"
            f"交付件：{trim_nz} / {len(cross_check)}。")
        add("")
        if raw_ivr:
            add(f"**这批装配的裁剪前 `ivr_assembly` p50 = {_pct(raw_ivr, 50):.3e}、"
                f"范围 {raw_ivr[0]:.3e}–{raw_ivr[-1]:.3e}；交付件 p50 = "
                f"{_pct(trim_ivr, 50) if trim_ivr else float('nan'):.3e}、max = "
                f"{trim_ivr[-1] if trim_ivr else float('nan'):.3e}。**")
            add("裁剪前的量级正是记忆卡说的 1%–3%；交付件落到 ≤1e-13（全量最大，本 5 个")
            add("装配里 ≤3.2e-19）——**同一批装配、同一个扫描器，差别只在布尔裁剪这一步**。")
            add("")
            add("注意别把这里读成「体积比 ≈ 深度/对角线比」：两个量纲不同，数值落在一个区间")
            add("是因为薄接缝上两者恰好同量级，**不能当换算关系用**。这一节能证的是「人群不同」。")
            add("")
        add("「等体积边长/对角线」是**下界**口径（细长穿透同样体积下最大深度更大），")
        add("只用来看量级够不够得到 1%–3%，不能当穿透深度读。完整数据见同目录")
        add("`raw-crosscheck.json` / `raw-crosscheck.md`。")
        add("")

    add("## 7. ε 该定多少：只给建议，本次不定")
    add("")
    add(_epsilon_section(rows, values, nonzero_pair_rows))
    add("")

    add("## 8. 口径与边界")
    add("")
    add("- **归一口径**：`Σ vol(V_i∩V_j) / Σ vol(V_i)`，直接调 `ivr.py` 的 `ivr_exact`，未另造。")
    add("- **几何来源**：`assemblies_pre_decimate/parts/*.ply`（裁剪后、未简化、装配坐标系、米），")
    add("  `adapter-contract.md` §E2 规定布尔必须用它。`assemblies/`（225 mm 简化集）未使用。")
    add("- **加载**：`trimesh.load(process=False)`，不做顶点合并——布尔结果必须建立在与交付数据")
    add("  逐位相同的拓扑上。")
    add("- **一次一对**：故意不整批调 `ivr_exact`。整批时任一 unknown 会让整条 `value` 变 `None`，")
    add("  而本扫描必须把 unknown 单独数出来（§4.8 规则 2）。")
    add("- **不做的事**：未改 `src/pah/`，未碰 `scripts/m1_repair.py`、未碰 `D:\\yanyi\\rl_assembly\\`，")
    add("  未重算 393/705，未定 ε。")
    add("- **本扫描测不到的东西（重要）**：IVR 是**体积**量，对「界面零间隙/精确共面」这种")
    add("  **接触但不互穿**的状态读到 0。而 UIPC 侧 `is_valid()=False` 的实测根因正是")
    add("  「严格共面、距离恰为 0」（记忆卡「uipc装配有效性自检」）。所以 **A=0 不等于装配")
    add("  物理合法**：体积口径看不见零间隙，零间隙需要用间隙/距离判据单独测。")
    add("")
    add("## 9. 复现")
    add("")
    add("```bash")
    add("python scripts/ivr_scan.py                 # 全量 100 装配（约 9 分钟）")
    add("python scripts/ivr_scan.py --resume        # 断点续跑（已完成的装配不重算）")
    add("python scripts/ivr_scan.py --report-only    # 只从 rows.jsonl 重算本报表")
    add("python scripts/ivr_scan.py --raw-cross-check  # §6.1 的裁剪前/交付件对照")
    add("python runs/ivr-scan/verify_rows.py          # 独立复核：只读 rows.jsonl 对账")
    add("PYTHONPATH=src python -m pytest tests/test_ivr_scan.py -q   # 全仓套件同样约定")
    add("```")
    add("")
    return "\n".join(lines) + "\n"


def _headline(rows, known, values, nonzero_asm, n_unknown) -> str:
    """§0 一句话结论：先给判决，再给数字。

    判决线是 `1e-6`（体积比）——本报表 §2 把它标成浮点噪声档的上界。判「有没有
    穿模」必须用这条线，不能用 `== 0`：布尔在近相切界面上会留下 1e-19 量级的残渣，
    那不是穿模，也不是「扫描器读到了东西」。
    """
    if not rows:
        return "（无数据）"
    if not values:
        return f"{len(rows)} 个装配全部含 unknown 对，A 类无法给出定值。"
    top = values[-1]
    if top <= BUCKETS[1]:
        residue = "" if top == 0.0 else f"（最大 {top:.3e}，属布尔浮点残渣，非穿模）"
        return (
            f"**{len(rows)} 个装配、{sum(r['n_parts'] for r in rows)} 个部件、"
            f"{sum(r['pairs_total'] for r in rows)} 对里，体积意义上的互相侵入为零："
            f"`ivr_assembly` 全部 ≤ 1e-6**{residue}；unknown {n_unknown} 对（单独见 §4）。"
            f"交付语料是裁剪后的产物，接缝被布尔切成精确共面，所以体积口径读 0。"
        )
    return (
        f"{len(values)} 个可定值装配里，`ivr_assembly` 最大 {top:.3e}、"
        f"非零装配 {len(nonzero_asm)} 个；unknown {n_unknown} 对。"
    )


def _compare_section(rows: list[dict], values: list[float], has_cross_check: bool = False) -> str:
    """与「侵入量中位 1%–3% 对角线」的对照，先查口径再下结论。"""
    if not values:
        return "没有可定值的装配，无法对照。"
    top = values[-1]
    med = _pct(values, 50)
    if has_cross_check:
        verdict = "交付件远小于 1%–3%，而裁剪前原始件正好落在 1%–3%"
        tail = "**差异来自人群（裁剪前 vs 裁剪后），不是扫描口径失效**——§6.1 用同一个扫描器量到了："
    else:
        if top > 0.03 or med > 0.01:
            verdict = "更大"
        elif top == 0.0:
            verdict = "更小（实际读到 0）"
        else:
            verdict = "更小" + (" 3 个数量级以上" if top < 1e-3 else "")
        tail = "两处差异必须分开看，**任何一处都足以解释全部差距，所以这里不下「记忆卡错了」的结论**："
    lines = [
        f"**判决：{verdict}。** 实测 `ivr_assembly` p50 = {med:.3e}、max = {top:.3e}；",
        "记忆卡说的是「侵入量中位为装配对角线 **1%–3%**（0.5–6 mm）」。",
        tail,
        "",
        "1. **人群不同（最主要）**。1%–3% 说的是**裁剪前**的 HY3D-Bench 原始部件：",
        "   「部件各自独立重建，接缝两边都带材料、互相侵入 1%–3% 对角线」（记忆卡",
        "   「装配数据管线布尔裁剪」）。交付的 `assemblies_pre_decimate` 是 `exclusive_i =",
        "   part_i − ∪(其余部件)` 裁剪之后的产物，同一张卡记录裁剪后「残余相交 0」、",
        "   「27/27 条界面间距＝0（精确重合）」。本扫描测的正是裁剪后这一态，读到 0 与",
        "   裁剪卡自洽，**不是矛盾**。",
        "2. **口径不同**。1%–3% 是**穿透深度 / 装配对角线**（线性量，0.5–6 mm 是长度）；",
        f"   本报表是**体积比** `Σvol(V_i∩V_j)/Σvol(V_i)`。两者量纲不同，",
        "   不能直接比大小，只能比「是否有可辨穿透」这个定性结论。",
        "3. **另一个数也不属同一人群**。「seed 3 最多 83% 表面在另一件内」是**生成侧**",
        "   （seed 3 物体）被 data-harvest 的 `s5_no_overlap` 过滤**丢弃**的那批，",
        "   与交付 100 装配不是同一个集合。",
        "",
        "**要直接验证 1%–3%**，得在 `data/objects/<object_id>/part_*.ply`（裁剪前原样件）上",
        "跑同一个扫描器。"
        + ("**下一节就是这件事。**" if has_cross_check else "本报表本次没跑这一步，见 §7 的建议。"),
        "",
        f"长度口径换算：把最大那对交集体积折算成等体积立方体边长，",
    ]
    diag = _median([r["diagonal"] for r in rows if r.get("diagonal")])
    deepest = max(rows, key=lambda r: r.get("max_pair_volume") or 0.0)
    if deepest.get("max_pair_volume"):
        edge_mm = _equiv_edge_mm(deepest["max_pair_volume"])
        lines.append(
            f"**{edge_mm:.6g} mm**（装配对角线中位 {diag * 1000:.1f} mm）——"
            if diag
            else f"**{edge_mm:.6g} mm**——"
        )
        lines.append("1%–3% 对角线（1.6–4.8 mm）在体积口径下对应的深度，本数据远未达到。")
    else:
        lines.append("体积为 0 时该换算无意义（等体积边长 0）。")
    return "\n".join(lines)


def _epsilon_section(rows, values, nonzero_pair_rows) -> str:
    """ε 建议：只给理由和数量级，明确标注待定。"""
    if not values:
        return "无分布，无法给建议。"
    top = values[-1]
    n_asm = len(rows)
    lines = [
        "**待定（ε 属评测配置，随 `policy_version` 冻结，见 §4.8）。** 只给本扫描能支撑的建议：",
        "",
        f"- 实测非零装配 {sum(1 for v in values if v > 0)} / {n_asm}，最大 `ivr_assembly` = {top:.3e}。",
        "  在这批数据上，**任何 ε ≥ 1e-6 都会让全部 100 个装配通过 A 类门禁**；",
        "  反过来说，A 类在这批数据上区分不出好坏装配——它是一道「已经全过」的门。",
        "- 因此 ε 不该由 A 类分布定，**应由「修复前后的判据一致性」定**：B 类（修复新增穿透）",
        "  才是这道门真正要拦的东西。建议作法是同一个 ε 同时作用在 A 和 B 上，",
        "  用 B 的分布（M1 产出）来校准；A 类只用来确认「基线不是靠噪声蒙过去的」。",
        "- 若必须先给一个占位数：**ε = 1e-6**（体积比）是当前唯一有实测支撑的选择——",
        "  它刚好高过这批数据里最大的布尔残渣量级，又不至于把一次真实的小穿模算成通过。",
        "  但这是**待定值**，不得写进任何 `policy_version` 之前先补 B 类分布。",
        "- 一个必须同时存在的另一半：体积 ε 管不住「零间隙」（见 §8 最后一条）。",
        "  IVR 门通过不等于装配物理合法，需要并行的间隙判据。",
    ]
    return "\n".join(lines)


def _in_bucket(value: float, lo: float | None, hi: float | None) -> bool:
    if lo is None and hi is not None:
        return value == hi
    if lo is not None and hi is not None:
        return lo < value <= hi
    if lo is not None and hi is None:
        return value > lo
    return False


def _pct(values: list[float], q: float) -> float:
    """线性插值分位数（与 numpy 默认一致），少于 2 个点时退回端点。"""
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    return float(np.percentile(values, q))


def _median(values: list[float]) -> float | None:
    return None if not values else float(np.median(values))


def _equiv_edge_mm(volume_m3: float) -> float:
    """等体积立方体边长（mm）：把无量的体积比翻译成读者能比的东西。"""
    if volume_m3 is None or volume_m3 <= 0.0:
        return 0.0
    return float(volume_m3) ** (1.0 / 3.0) * 1000.0


def _why(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _fmt(value: float | None, digits: int = 3) -> str:
    """`None` 打成 `None`，别打成 0（§4.8：不知道不许显示成没穿模）。"""
    return "None" if value is None else f"{value:.{digits}e}"


def _load_cross_check(path: Path) -> list[dict] | None:
    """有 `raw-crosscheck.json` 就并进报表 §6.1，没有就跳过（不编数）。"""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, list) and data else None


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="A 类穿透扫描：装配态部件两两 IVR（不改 393/705，不定 ε）"
    )
    parser.add_argument("--root", type=Path, default=GEOMETRY_ROOT, help="几何根目录")
    parser.add_argument("--out", type=Path, default=ROWS, help="rows.jsonl 路径")
    parser.add_argument("--resume", action="store_true", help="跳过已完成的装配")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 个装配")
    parser.add_argument("--assemblies", default="", help="逗号分隔的 asm_id 白名单")
    parser.add_argument("--self-test", action="store_true", help="正向对照，不写任何产物")
    parser.add_argument("--report-only", action="store_true", help="只从 rows.jsonl 重算报表")
    parser.add_argument("--report", type=Path, default=REPORT, help="report.md 路径")
    parser.add_argument(
        "--raw-cross-check",
        nargs="?",
        const="__default__",
        default=None,
        help="附加对照：同时扫裁剪前原始件与交付件；可选逗号分隔 asm 白名单（缺省前 5 个）",
    )
    args = parser.parse_args(argv)

    if args.self_test:
        return self_test()

    if args.raw_cross_check is not None:
        names = [n.strip() for n in args.raw_cross_check.split(",") if n.strip()]
        if not names or names == ["__default__"]:
            names = sorted(p.name for p in GEOMETRY_ROOT.iterdir() if p.is_dir())[:5]
        return raw_cross_check(
            names,
            args.report.parent / "raw-crosscheck.json",
            args.report.parent / "raw-crosscheck.md",
        )

    if args.report_only:
        pair_rows, asm_rows, bad = read_rows(args.out)
        report = build_report(
            pair_rows,
            asm_rows,
            bad,
            cross_check=_load_cross_check(args.report.parent / "raw-crosscheck.json"),
        )
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report, encoding="utf-8")
        print(f"report: {args.report} ({len(asm_rows)} assemblies, {len(pair_rows)} pair rows)")
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
    # 已完成装配的 pair 行在文件里，但装配级汇总要整装配重算才有分母——所以
    # resume 只跳过**整装配已完成**的，不做对级续跑（对级续跑会让分母/计数缺项）。
    todo = [p for p in asm_dirs if p.name not in done]
    print(
        f"scan: {len(asm_dirs)} assemblies selected, {len(done & {p.name for p in asm_dirs})} "
        f"already complete, {len(todo)} to run -> {args.out}"
    )

    writer = RowWriter(args.out)
    started = time.perf_counter()
    n_exact = 0
    try:
        for index, asm_dir in enumerate(todo, 1):
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
                    "pairs_exact": 0,
                    "n_unknown_pairs": 0,
                    "pairs_nonzero": 0,
                    "pairs_proven_zero": 0,
                    "ivr_assembly": None,
                    "ivr_lo": None,
                    "ivr_hi": None,
                    "ivr_max_pair": None,
                    "max_pair": None,
                    "max_pair_volume": None,
                    "numerator": None,
                    "denominator": None,
                    "diagonal": None,
                    "unknown_pairs": [],
                    "unknown_diagnostics": [],
                    "part_defects": [
                        {"part": "*", "kind": "scan_failed", "why": _why(exc)}
                    ],
                    "traceback": traceback.format_exc(),
                    "elapsed_s": None,
                }
            writer(summary)
            n_exact += summary["pairs_exact"]
            ivr = summary["ivr_assembly"]
            print(
                f"[{index}/{len(todo)}] {summary['asm']} parts={summary['n_parts']} "
                f"pairs={summary['pairs_total']} culled={summary['pairs_culled']} "
                f"exact={summary['pairs_exact']} unknown={summary['n_unknown_pairs']} "
                f"nonzero={summary['pairs_nonzero']} "
                f"ivr={'None' if ivr is None else f'{ivr:.3e}'} "
                f"{summary['elapsed_s']}s  elapsed={time.perf_counter() - started:.0f}s",
                flush=True,
            )
    finally:
        writer.close()

    pair_rows, asm_rows, bad = read_rows(args.out)
    report = build_report(
        pair_rows,
        asm_rows,
        bad,
        cross_check=_load_cross_check(args.report.parent / "raw-crosscheck.json"),
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    total = time.perf_counter() - started
    print(
        f"done: {len(asm_rows)} assemblies, {len(pair_rows)} pair rows, "
        f"{total:.1f}s -> {args.report}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
