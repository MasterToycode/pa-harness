# M0 切片：`connector_feasible` 适配器与金标准对照

产物目录。代码在 `src/pah/tools/verifiers/` 与 `src/pah/cli/`，测试在
`tests/test_verifier_connector.py`。`rl_assembly/` 全程只读（import + 子进程查 git 描述，零写入）。

## 验收对照表（`report.md` 为 `pah probe` 自动生成）

| 量 | 本次实测 | 金标准 | 差 |
|---|---:|---:|---:|
| `edges_total` | 953 | — | — |
| `feasible`（分母，`feasible ∧ ¬thickness_failed ∧ lod=fine`） | **705** | 705 | **0** |
| `success` | **393** | 388 | +5 |
| `rate` | 55.7% | 55.0% | +0.7pp |
| 扣掉环境相关边后的 `success` | **388** | 388 | **0** |

**逐边比对（705 条）：0 条属于「同一判断」之外的差异，5 条差异全部是同一个原因。**

| 边 | 本适配器 | 金标准 | 金标准 detail |
|---|---|---|---|
| `asm_000097:0-1`（id 912） | ok | fail | `exception:MemoryError(Unable to allocate 56.0 MiB ...)` |
| `asm_000097:1-2`（id 916） | ok | fail | `exception:MemoryError(Unable to allocate 14.0 MiB ...)` |
| `asm_000097:1-3`（id 917） | ok | fail | `exception:MemoryError(...)` |
| `asm_000097:1-5`（id 919） | ok | fail | `exception:MemoryError(...)` |
| `asm_000097:1-6`（id 920） | ok | fail | `exception:MemoryError(...)` |

这 5 条边正是 `connector-boolean-report.md` §19.5.1 自己标注的「属环境问题而非几何问题」的
`bool_failed`——那次运行进程内存打满，5 条边在 `manifold3d` 分配 14–56 MiB 时抛 `MemoryError`。
本次 `--jobs 3`、每 worker 只持 1 个装配，同一批边全部成功。`388 + 5 = 393`，帐是对得上的。

**推论（对 M0 的口径有影响）**：金标准 388 里含 5 条内存相关失败，重跑不可稳定复现；
可复现的部分是 **388 / 705 = 55.0%（±5 条）**。要让基线确定，需要内存门槛或对 `asm_000097`
的重试策略，否则每次复现都会在 388–393 之间浮动。

其余码分布与金标准逐项一致（金标准 §19.4 的 7 类对照）：

| 码 | 本次 | 金标准 reason 归并后 |
|---|---:|---:|
| `gap_too_wide` | 166 | 166 |
| `no_material_hit` | 72 | 72 |
| `unreatable_too_small` | 29 | 29 |
| `socket_breaks_wall` | 17 | 17 |
| `peg_detached` | 8 | 8 |
| `degenerate_other` | 17 | （`degenerate_geometry` 合计 258，本次 258 = 166+72+17+3） |
| `degenerate_tangent` | 3 | 同上 |
| `bool_failed` | 0 | 5（即上表 5 条 `MemoryError`） |

## 规模与口径

- 数据：`research/contact-graph-anchor/edges_v2.json`（953 边 / 100 装配）；几何
  `data/assemblies_pre_decimate`（未简化，150 mm，2.28 GB）；分母涉及 93 个装配（7 个无可行边）。
- `unit_mm` 实测为 **`225 / longest_side ≈ 1500.0000`**，不是 1000 也不是
  `meta["scale"] × 1000 ≈ 76`。三者在同样 5 个装配上的实测与证据见 `scale-probe.md`。
- 与金标准的对齐参数：`clamp=True`、`verify_stl=False`、`overlap_check=True`、
  `depth_min_mm=0.0`、`section="D"`、`snap=None`、`host_quality` 透传。
- 全量耗时 **1047.7 s**（`--jobs 3`，`_load_parts` 缓存上限 1）。首次用 `--jobs 4` + 缓存上限 2
  时 free RAM 掉到 0.9 GB，头 4 个装配跑了 10 min 以上还没收口（swap），中止后改 3 worker 重跑；
  93 个装配 / 705 条边全程 17.5 min。16 GB 机器上 `--jobs` 不宜超过 3。

## 复跑

```powershell
$env:PYTHONPATH='D:\yanyi\pa-harness\src'
python -m pah.cli.main verify sanity                     # 3 个 fixture，约 25 s
python -m pah.cli.main probe connector --split feasible --cases all --jobs 3 --out runs/m0-connector
```

`rows.jsonl` 逐边落盘 + 断点续跑；同一 `--out` 换口径会被 `probe.json`/`run.json` 的协议比对拒绝。

## 文件

| 文件 | 内容 |
|---|---|
| `report.md` | `pah probe` 自动生成的对照表（本次验收结论） |
| `run.json` | 机器可读的同内容 + 逐边差异明细 |
| `rows.jsonl` | 705 条边的逐边结果（`ok/reason/detail/degen_code/code/unit_mm/r_mm/...`） |
| `scale-probe.md` | `unit_mm` 三口径 × 5 装配的实测判定 |
| `../_scale-probe/{scale,1000,meta_scale}/` | 上面三口径的原始 `run.json` 与 `rows.jsonl` |
| `../m0-connector.log` | 全量运行日志 |

## 2026-11-01 补跑：`MemoryError` 重试 + `ivr_exact` 入表

上面各表描述的是第一次运行（17:38 报告）。补跑后本目录变成两代产物并存：

| 文件 | 内容 |
|---|---|
| `report.md` | **第二次运行**的自动报告 + 末尾「追记」一节（重试结果、IVR 三个数、耗时对比、第一次运行的 5 条差异原文） |
| `report-run1.md` / `rows-run1.jsonl` / `run-run1.json` | 第一次运行的原件，逐字保留 |
| `rows.jsonl` / `run.json` | 第二次运行的逐边结果（新增 `retried_n` / `oom` / `ivr_value` / `ivr_method` 字段） |
| `run.log` | 第二次运行的控制台日志 |

补跑结论（`--jobs 3`，1217.6 s）：

- `retried_n = 2`、`oom_n = 0`：两条边各撞 1 次 `MemoryError`，重试后成功；没有边留在 `verifier_oom`；
- `success = 393`，扣掉金标准侧 5 条环境边 = 388 → 388 与 393 都**可归因**，不再 ±5 浮动；
- `success_original = 393`、`success_ivr_validated ∈ [393, 393]`、`n_unknown = 0`；
  360 对走 `manifold` 布尔、33 对由 AABB 不相交证 0；非零 IVR 最大 4.45e-13（远低于 ε=1e-6）。

复跑命令不变：`python -m pah.cli.main probe connector --split feasible --cases all --jobs 3 --out runs/m0-connector`；
`rows.jsonl` 已存在时会断点续跑，不会再重测。

