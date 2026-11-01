# `unit_mm` 口径实测（5 装配，`--cases 5`）

问题来自 `docs/internal/adapter-contract.md` §E1 的「待拍板 1」：`unit_mm` 取 1000 还是
`meta["scale"] × 1000`？两者差 12.7 倍，而 `connector.feasibility` 用它把 `r_min_mm` 换算成
数据单位（`r_min = r_min_mm / unit_mm`），因此它同时决定**可行边集合**与**成功率**。

## 做法

同一批几何（`assemblies_pre_decimate`，未简化 150 mm）、同一条流水线
（`pah probe connector --split feasible --cases 5`）、只改 `--unit-mm`：

| 变体 | 派生长度除数 | `unit_mm` 传给 `connector.build` |
|---|---|---|
| `scale` | `225 / longest_side` | 同左 |
| `1000` | 1000 | 1000 |
| `meta_scale` | `meta["scale"] × 1000` | 同左 |

三者都是**自洽的一套口径**（除数与 `unit_mm` 用同一个量），差的只是这一套选的是哪个尺度。

## 结果（5 装配 = `asm_000000..asm_000004`，38 条可行边）

| `unit_mm` 口径 | 实测 `unit_mm` | 可行边 | 成功 | 成功率 | 与金标准逐边不符 | `unreatable_too_small` |
|---|---:|---:|---:|---:|---:|---:|
| **`scale`（采用）** | **1500.0000** | 38 | **24** | **63.2%** | **0 / 38** | 1 |
| `1000` | 1000 | 38 | 20 | 52.6% | 8 / 38 | 1 |
| `meta_scale` | 76.23 | 38 | 4 | 10.5% | 22 / 38 | 28 |

金标准在同样 5 个装配上的数字（`_final_rows_Nnew_Tnew.json`，分母 `fine ∧ feasible ∧ ¬thickness_failed`）：
**38 条可行边、24 成功**。只有 `scale` 复现，且是逐边一致，不是只在总数上碰巧相等。

原始产物：`runs/_scale-probe/{scale,1000,meta_scale}/run.json` 与同目录 `rows.jsonl`。
`--jobs 5` 的 `scale` 跑用时 81.4 s；另两个变体是断点续跑（`rows.jsonl` 已在盘上，0.0 s 收口），
顺带验证了续跑。

## 结论：两个候选都错，正确的是 `225 / longest_side ≈ 1500`

1. **`meta["scale"] × 1000`（76.2）是量纲不对**，不是「差一点」。`meta["scale"]` 是
   「米 / 原始 Hy3D 源单位」（`0.225 / 0.11795 = 1.9076` 这类比值的倒数），与「米 / 存储单位」
   无关。用它当 `unit_mm`，`r_min` 被放大约 19.7 倍（1500 / 76.2），38 条边里 28 条被
   判成 `unreatable_too_small`——而金标准里它们有 24 条能成功。
2. **`1000` 的量纲对（米 → 毫米），但尺度对错了。** 1000 是 225 mm 归一化几何（`assemblies/`）
   的正确口径；`assemblies/` 是简化件，`connector-boolean-report.md` §18.3 实测禁止用它做
   布尔宿主（`t_b` 最多差 4.5 倍）。布尔与测厚必须同源于 150 mm 的
   `assemblies_pre_decimate`，因此 1 个数据单位（那里的 1 米）等于**225 尺度下** 1500 mm。
3. `r_min` 与 `r_cap` 必须用**同一个** `unit_mm`。`edges_v2.json` 的派生长度
   （`r_*_mm` / `t_*_mm` / `half_*_mm`）写在 225 尺度，坐标（`center_mm` / `anchor_mm`）写在
   150 基准帧——文件头两句话说得明确（`coordinate_rule` 与 `derived_length_rule`）。
   代码里这两件事分成两个常量：`to_edge_dict` 的坐标除以固定的
   `COORD_MM_PER_UNIT = 1000`，派生长度除以 `unit_mm`。混用这两个除数正是
   `adapter-contract.md` §E2 警告的「两套尺度混用」。

判定为尺度不变量：`feasible` 的条件只依赖 `r_cap / r_min`
= `r_inplane_max_mm / r_min_mm`，与尺度无关（`connector-boolean-report.md` §18.4 的
`test_build_scale_invariance`）。所以上表三行「可行边」都是 38——差别不在可行性判定，
而在**喂给布尔的几何尺寸**：`1000` 让 peg 相对几何大 1.5 倍（多出 3 条 `socket_breaks_wall`），
`meta_scale` 让它大约 20 倍（28 条直接判放不下）。这就是为什么必须看**成功率**而不是只看
可行边数。

代码常量旁的注释见 `src/pah/tools/verifiers/connector_feasible.py` 的 `SCALE_MM` 与
`to_edge_dict` 的 docstring。
