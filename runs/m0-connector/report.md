# M0 connector probe

- edges file: `D:\yanyi\rl_assembly\research\contact-graph-anchor\edges_v2.json`
- geometry: `D:\yanyi\rl_assembly\data\assemblies_pre_decimate`
- split: `feasible`, scope: `gold`, assemblies: 93, unit_mm rule: `scale`
- wall clock: 1217.6 s

| quantity | probe | gold | delta |
|---|---:|---:|---:|
| edges_total | 953 | — | — |
| feasible | 705 | 705 | +0 |
| success | 393 | 388 | +5 |
| rate | 55.7% | 55.0% | +0.7% |

**REPRODUCED-MODULO-ENVIRONMENT** (705/705 edges ran)

- `unreatable_too_small`: 29 (gold 29)
- adapter-feasible: 676
- environment-dependent edges (gold lost them to `MemoryError`, this run did not): 5 -> adjusted success 388

## failure codes

- `gap_too_wide`: 166
- `no_material_hit`: 72
- `unreatable_too_small`: 29
- `socket_breaks_wall`: 17
- `degenerate_other`: 17
- `peg_detached`: 8
- `degenerate_tangent`: 3

## `MemoryError` retries

- `retried_n`: 2 (retry attempts made after an out-of-memory verdict)
- `oom_n`: 0 (edges still recorded `verifier_oom`)
- `verifier_oom` failures: 0 — counted apart from `bool_failed`, never merged: one is the machine, the other is the geometry

## IVR (`ivr_exact`, ε = 1e-06)

| quantity | value |
|---|---:|
| `success_original` | 393 |
| `success_ivr_validated` ∈ [`lo`, `hi`] | [393, 393] |
| `n_unknown` | 0 |
| measured (`ivr` known) | 393 |
| over ε (success but interpenetrating) | 0 |

how each reading was produced: `manifold` 360, `bbox` 33

`lo` counts `unknown` as not IVR-validated, `hi` counts it as validated; the interval is the answer and `n_unknown` says how much of it is the boolean engine's silence rather than a measurement (§4.8).

## edge-by-edge vs the frozen gold rows

- compared: 705, mismatches: 5
- `asm_000097:912` adapter=ok gold=fail `exception:MemoryError(Unable to allocate 56.0 MiB for an arr`
- `asm_000097:916` adapter=ok gold=fail `exception:MemoryError(Unable to allocate 14.0 MiB for an arr`
- `asm_000097:917` adapter=ok gold=fail `exception:MemoryError(Unable to allocate 14.0 MiB for an arr`
- `asm_000097:919` adapter=ok gold=fail `exception:MemoryError(Unable to allocate 14.0 MiB for an arr`
- `asm_000097:920` adapter=ok gold=fail `exception:MemoryError(Unable to allocate 56.0 MiB for an arr`

---

## 追记：`MemoryError` 重试 + `ivr_exact` 入表（2026-11-01 18:10）

本节人工追加。上面的对照表、失败码、逐边差异全部是 `pah probe` 自动生成的原文，未改动。

### 重试后的确定数

| 量 | 值 | 说明 |
|---|---:|---|
| `retried_n` | **2** | 重试尝试次数。`asm_000079:759`、`asm_000082:780` 各撞 1 次 `MemoryError`，各重试 1 次后成功 |
| `oom_n` | **0** | 没有边最终落在 `verifier_oom` |
| `verifier_oom` | 0 | 与 `bool_failed`（0）**分开计数，不合并**：一个是环境，一个是几何 |
| `success` | **393** | 与首次运行相同 |
| 扣环境边后 | **388** | 393 − 5 条金标准侧环境失败 |

**388 与 393 现在都有归属，不再浮动**：

- 金标准侧 5 条（`asm_000097` id 912/916/917/919/920）是**金标准那次运行**的 OOM，本次仍全部成功；
- 本次运行**自己**撞上 2 次 OOM，各重试 1 次后成功。这两条边在首次运行（17:38）里是一次通过的
  （首次 `bool_failed` 计 0），所以 OOM 是**间歇性**的——这正是「388 ± 5」的来源，不是几何判据不稳；
- 若两次重试（1 s、3 s）仍失败，该边会记 `verifier_oom` 并进 `oom_n`，不会再静默变成 `bool_failed`。

### IVR（§4.8 要求的三个数）

| 量 | 值 |
|---|---:|
| `success_original` | 393 |
| `success_ivr_validated` ∈ [`lo`, `hi`] | **[393, 393]** |
| `n_unknown` | 0 |
| 已知读数 | 393 |
| 超过 ε 的成功边 | 0 |

读数方式：`manifold`（实体布尔交集）360 对、`bbox`（AABB 不相交即证 0）33 对。

393 个 IVR 里 388 个恰为 0，5 个非零且最大 **4.45e-13**，比 ε = 1e-6 低 7 个数量级——
是 `manifold3d` 在接触面留下的浮点残差，不是设计互穿。

**区间宽度为 0 不是因为挑了端点**：`n_unknown` 实测为 0，即 360 次布尔交集全部返回 `NoError`，
没有一条被引擎沉默吞掉。规则写在 `_ivr_summary`：`lo` 把 unknown 记为未通过，`hi` 记为通过。

口径与代价：

- 只对**成功**边所在的部件对跑 `ivr_exact`（一条边一对）；失败边不算成功，不消耗 55 ms–4.6 s；
- 装配级缓存键 = 两个 part id 排序后 `|` 拼接，同一对部件只算一次（本次 393 条边对应 393 个不同对，命中 0 次）；
- 几何与布尔同源（`assemblies_pre_decimate`，单位米），IVR 是比值，不做毫米换算；
- ε 是**评测配置**（§4.8），写死在 `src/pah/cli/probe.py` 的 `IVR_EPS`，与 `policy_version` 一起冻结。

### 耗时

| | 首次（17:38 的报告） | 本次 |
|---|---:|---:|
| wall clock | 1047.7 s | **1217.6 s** |

+169.9 s（+16%）。其中 IVR 约 150 s（393 对：最小装配 0.04 s、中位 0.68 s、最大 4.6 s / 236 万面），
其余是两次重试各 1 s 的等待与重跑布尔。

### 历史：重试前的第一次运行（原文，2026-11-01 17:38）

全文逐字保留在 `report-run1.md`。当时那 5 条差异的段落原文是：

```markdown
## edge-by-edge vs the frozen gold rows

- compared: 705, mismatches: 5
- `asm_000097:912` adapter=ok gold=fail `exception:MemoryError(Unable to allocate 56.0 MiB for an arr`
- `asm_000097:916` adapter=ok gold=fail `exception:MemoryError(Unable to allocate 14.0 MiB for an arr`
- `asm_000097:917` adapter=ok gold=fail `exception:MemoryError(Unable to allocate 14.0 MiB for an arr`
- `asm_000097:919` adapter=ok gold=fail `exception:MemoryError(Unable to allocate 14.0 MiB for an arr`
- `asm_000097:920` adapter=ok gold=fail `exception:MemoryError(Unable to allocate 56.0 MiB for an arr`
```

首次运行的其余段落（`success 393 / gold 388`、`REPRODUCED-MODULO-ENVIRONMENT`、
码分布 `gap_too_wide 166` … `degenerate_tangent 3`、`bool_failed 0`）与本次逐项相同，
`rows-run1.jsonl` / `run-run1.json` 是那次运行的原始产物。

