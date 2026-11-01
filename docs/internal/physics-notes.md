# 物理沉降档实测记录（`physics_settle`）

> 2026-11-01 · T5 · 对应交付 `src/pah/tools/mujoco_settle.py` + `src/pah/tools/verifiers/physics_settle.py`
> 这份文档只记**实测数字**，不记愿望。数字全部可以用文末两条命令复现。

---

## 0. 一句话

静态无干涉 ≠ 动力学稳定（`docs/pah-dev-doc-v1.md` §6）。物理档是验证器级联最贵的一档：
把 body 焊死在世界上、insert 放开、重力 -Z、跑 0.8 s 真实动力学，然后量三件事——
**横向游隙、接触穿深、有没有离开座位**。

---

## 1. 模型（以及它所有近似，逐条列出）

| 项 | 取值 | 理由 / 近似误差 |
|---|---|---|
| 单位 | 设计平面 mm；仿真 m；唯一换算 `MM_PER_M = 1e-3`，只在建局部坐标系时用一次 | 写错就是 1000× 质量错误 |
| 局部坐标系 | +Z = `view.mate_axis`；原点 = insert 落座质心 | 重力仍是世界 -Z；"哪边是拔出方向"从 `view.mate_axis` 读 |
| 腔体 | **24 个凸盒**，每盒内表面过实测壁点，用该点**所在三角面的法向**定向 | MuJoCo 无法把凹孔当一个 geom；用面法向而不是射线方向，否则斜面（槽底）会变成斜平面，insert 能沉进材料 2 mm |
| 多边形误差 | 顶点在 `R/cos(π/24)`，即比真实壁**远 0.86%**（本 fixture：0.0444 mm） | 偏向"判不稳"，不会制造穿深（每条 facet 是凸腔的支撑平面） |
| 壁半径 | 沿 engaged 段取多个高度，**每个方向取最小值**（最紧处） | 中段更宽的孔会被当窄孔模拟，同样偏保守 |
| 壁采样 | 24 方向 × 5 高度 = 120 条射线，一次向量化 raycast | trimesh 射线核 |
| insert | 圆截面（周长差 ≤2%）→ MuJoCo 精确 `cylinder`；否则 → **凸包 mesh** | 凸包 = MuJoCo 对任何 mesh geom 的碰撞表示，所以**非凸 insert 直接拒绝**（`physics_error`），不把倒扣的保持特征平均掉 |
| 接触 | `solref="0.004 1"`、`solimp="0.95 0.99 0.001"`、elliptic、implicitfast | 求解器柔度：1.4 g insert 静置下沉 **0.0032 mm**（这就是 `penetration_mm` 的噪声底） |
| 材料 | density 1240 kg/m³（PLA）、摩擦 0.5 | |
| 时间步 | 0.8 s × 2e-4 s = **4000 步**，固定步长、无随机 | 同 view 两次跑逐位相同 |
| 释放姿态 | insert 从**设计间隙允许的最坏横向位置**释放 | 关节的游隙本来就 ≤ 间隙；实测 drift ≈ clearance |
| 支撑 | 从 insert 质心沿重力打一条射线，命中处放支撑板；**打不中 = 无底（通孔），让它掉** | 单条射线的偏心支撑会被读成无支撑，已知局限 |

**不建模**（避免把 pass 读过头）：插入动作（那是 `insertion_path`）、body 除界面区外的其余部分、
塑性/过盈应力/热效应、真实打印的各向异性。

---

## 2. 实测数字（本机 MuJoCo 3.17.0，CPU）

fixture 几何：body = 60×40×12 板，居中立孔；insert = d10 × 14 长的销；盲孔孔深 = engage；销底坐孔底。

| 案例 | 判定 | drift_mm | 穿深_mm | 轴向_mm | 倾角° | 咬合_mm | 有支撑 | 仿真耗时 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 盲孔 0.15 间隙 / engage 8 | **通过** | 0.1498 | 0.0032 | -0.002 | 0.00 | 8.00 | 是 | 110 ms |
| 盲孔 0.20 间隙 / engage 8 | **通过** | 0.1998 | 0.0032 | -0.002 | 0.00 | 8.00 | 是 | 111 ms |
| 盲孔 0.15 间隙 / engage **0.8** | **通过** | 0.1499 | 0.0038 | -0.002 | 0.00 | 0.80 | 是 | 105 ms |
| 盲孔 **2.0 间隙** / engage 8 | `joint_unstable`（`lateral_drift`） | **1.9998** | 0.0032 | -0.002 | 0.00 | 8.00 | 是 | 75 ms |
| **过盈** 0.05（孔比销小 0.1） | `dynamic_penetration` | 0.0005 | **0.0531** | -0.002 | 0.03 | 8.00 | 是 | 405 ms |
| **通孔** 0.15 间隙 | `joint_unstable`（`drop_through`） | 2.3765 | 0.0027 | **-3138.7** | 16.02 | 2.00 | **否** | 30 ms |

读法：

- **drift ≈ clearance**，这不是巧合：insert 从最坏姿态释放，游隙本身就是关节能给出的位移上界。
  物理档给出的不是"几何上允许多少"，而是"重力下它真的待在了哪里"——两者在紧配合上一致，
  在松配合上差的是"会不会翻倒/滑出去"，那由穿深、轴向、倾角三条独立量兜住。
- **穿深 0.0032 mm 是求解器柔度**，不是设计缺陷；过盈案例的 0.0531 mm 与 0.05 mm 设计过盈
  一一对应，所以这条量是可信的。
- **通孔 → 掉落 3.14 m**：0.8 s 自由落体 `0.5·9.81·0.8² = 3.14 m`，与实测一致。这条**是真实结论，
  不是模型 bug**：无轴向保持的通孔销在重力下没有任何东西托住它。

---

## 3. 四个出厂 fixture 上的物理档（`pah design sanity` 与 `pah run` 实测）

`fixtures/design/*.json` 经 T1 的 materializer 物化后，物理档的实测：

| fixture | 设计间隙 | 物理档结果 | drift | 穿深 | 量到的壁/销半径 | 仿真耗时 |
|---|---|---:|---:|---:|---:|---:|
| `demo-peg-hole-clear` | +0.150 | **无证书（通过）** | 0.150 mm | 0.0058 mm | 4.995 / 4.850 | 103 ms |
| `demo-peg-hole-interfere` | **-0.100** | **`dynamic_penetration`** | 0.001 mm | **0.1072 mm** | 4.995 / 5.100 | 407 ms |
| `demo-dovetail` | +0.100 | 未评估（note，见 §4） | — | — | — | — |
| `demo-snap-fit` | +0.100 | 未评估（note，见 §4） | — | — | — | — |

读法：

- **过盈案例的 0.1072 mm 穿深 ≈ 设计的 0.1 mm 单边过盈**（多出的 0.007 是求解器柔度 + 多边形近似，
  两者都在 magnitude 里）。这条把「设计上过盈」和「动力学上被挤压」连成同一个数。
- 测得壁半径 4.995 vs d_hole/2 = 5.000：差 0.005 mm 是 96 边形孔的弦误差 + 24 采样，
  与 §1 的 0.86% 误差同源，量级一致。
- 两个非圆销（dovetail / snap_fit）的凸包分别是实体的 **1.243×** 与 **1.621×**，物理档**不评**（§4）。
  这是诚实分层：**未评估 ≠ 通过**，报表里 `not_evaluated` 独立计数（T4 的 report 已经这么报）。

---

## 4. 能力边界：`note` 而不是证书

2026-11-01 实测发现：4 个 fixture 一度全部 `physics_error`（reason=`non_convex_insert`）。
`physics_error` 是 `family="environment"`，内核语义是「未知不能算通过」→ **整个 episode 直接 failed**，
于是任何案例都不可能 repaired。定性错误在于把**能力边界**当成了**工具故障**。三者必须分开：

| 类别 | 例子 | 输出 | 谁消费 |
|---|---|---|---|
| **物理缺陷** | 松配 2.0 mm、过盈 -0.1、通孔掉落 | `joint_unstable` / `dynamic_penetration` 证书 | 策略、失败率、报表 |
| **能力边界** | 非凸 insert（钩子倒扣被凸包平均掉）、非水密、不在模型域 | `check()` 返回 `[]` + `verifier.last_note = "physics not evaluated: <reason>: <detail>"` | 报表的 `not_evaluated` 独立桶 |
| **工具故障** | MuJoCo 装不上、模型编译失败、数值发散、超预算 | `physics_error` 证书（`family="environment"`） | 只进 `environment_errors`，不进失败率、不算通过 |

落地：`physics_settle.MODEL_BOUNDARY_REASONS` 列出的 reason → note；其余 `SettleError` → `physics_error`。
报表侧只需要读两个属性：T2 的 `DesignVerifierCascade.notes()` 收 `verifier.last_note`，
计费读 `verifier.last_sim_calls`（跑成功 1 / 没跑 0）。T4 的 report 实测已显示
`not_evaluated=2`（printability + physics_settle）并逐条列出原因。

**代价必须写在明面上**：`check() == []` 现在有两种含义——「量了，没问题」和「没量」。
区分只能靠 `last_note`；任何把 `[]` 直接当「物理通过」的报表都是在虚报覆盖率。

---

## 5. 为什么 peg_hole 走**盲孔**口径（lead 2026-11-01 拍板）

- 盲孔（孔深 = `engage_mm`，销底坐孔底）语义自洽：`peg_len = engage + 6` 的 6 mm 正好露在外部；
  3D 打印件里"销插进盲孔靠配合定位"是真实的一种配合。
- 通孔 + 无销头在重力下**必掉**，物理档会（正确地）判 `joint_unstable`。如果每个 peg_hole 都这样，
  端到端演示会永远停在物理档——那不是物理档太严，是**设计上确实没有轴向保持**。
- 所以：`pull_out` / `drop_through` 两条代码路径**保留**（它们是将来 `peg_hole_through` 带销头肩部
  这个 kind 的核心价值），只是当前 blind 几何不会触发。`tests/test_physics_settle.py` 里有专门的
  通孔坏例覆盖它们。

## 6. 为什么"engage 太浅"在盲孔上**不会**判不稳（诚实说明）

实测（上表第 3 行）：盲孔 0.15 间隙、engage 仅 0.8 mm → drift 0.1499、倾角 0.00°，**稳定**。

这不是漏判，是物理：销站在孔底上，质心在自己的支撑多边形上方，**纯重力在任何咬合深度都无法让它翻倒**。
让浅咬合变得危险的是"没有支撑"（通孔 / 悬空），那由 `supported=False` + `drop_through` 覆盖。
因此本档**不设咬合深度阈值**——设了就是拿一个不成立的物理学假设换一个好看的失败率。

## 7. 阈值从哪来（全部可从 `FitConstraints` 推导，不写死）

| 阈值 | 公式 | 默认值 | 理由 |
|---|---|---|---|
| `drift_mm` | `1.7 × constraints.max_clearance_mm`（下限 0.2） | 1.02 | 配合带顶格 0.6 mm；1.7× 保证物理档**永不反驳** fit 档已接受的配合；1.02 mm 也是 10 mm 销的 10%（销开始"不定位"的量级） |
| `penetration_mm` | `min(0.03, min_clearance/3)`（下限 0.01） | 0.03 | 求解器柔度 0.0032 mm 的 3–9 倍；且是配合带**最小**间隙 0.1 的 1/3，合法间隙设计够不到 |
| `pull_out_mm` | 绝对 0.5 | 0.5 | 与配合带上限同量级，"离开了座位"不是配合公差 |
| `tilt_deg` | 绝对 15 | 15 | 15° 是楔住不是落座；次要触发（翻倒通常先体现在 drift） |

改配合带 → 物理阈值一起动（`tolerances_for()` 是唯一换算点），不会出现两个验证器互相打脸。

## 8. 工具故障（`physics_error`，`family="environment"`）

不是失败、不算通过、不进任何失败率（契约 §3）。**只有工具本身坏了才走这里**；
几何超出模型域（非凸、非水密、不是一对落座件）走 §4 的 note，不发证书：

| reason | 触发 |
|---|---|
| `mujoco_unavailable` | `import mujoco` 失败（本模块 lazy import，不会连累注册表） |
| `raycast_failed` | trimesh 射线核故障 |
| `model_compile_failed` | MJCF 编译失败 |
| `step_failed` / `diverged` | `mj_step` 抛异常 / 位置出现非有限值 |
| `wall_budget_exceeded` | 单次 check 超过 4.5 s 硬上限 |
| `unexpected_failure` | 验证器自身未预期异常（包了一层 `except Exception`） |

## 9. 怎么接进级联（T2 已实现，此处只记口径）

`design_cascade.default_registry()` 通过 `importlib` 可选注册 `physics_settle.PhysicsSettle()`；
`cost_class == "expensive"` 唯一按 `sim_calls` 计费。两个属性是报表侧唯一接口：

- `verifier.last_sim_calls`：跑成功 1、没跑 0 —— 精确记账读它（成本类计费会把一次工具故障也记 1 sim）；
- `verifier.last_note`：非空表示**未评估**（§4）。T2 的 `DesignVerifierCascade.notes()` 已经把它
  收进 `not_evaluated` 独立桶；T4 的 report 实测显示 `not_evaluated=2`（printability + physics_settle）
  并逐条列出原因。

`DesignView.mate_axis` / `approach_mm` 由 T1 的 materializer 填、`assembly_from_spec` 记进
`assembly.constraints`、`view_of()` 读回来；`physics_settle` 用 `view.mate_axis`，
只有在 view 没声明（零向量）时才回退到按 kind 的兜底表（`MATE_AXIS_BY_KIND`，dovetail = +X）。

## 10. 复现

```powershell
cd D:\yanyi\pa-harness
python -m pytest tests/test_physics_settle.py tests/test_case_memory.py -q   # 43 项，~7 s
python -m pah.cli.main design sanity                                        # 5 个验证器全 PASS
python -m pah.cli.main run --spec fixtures\design\demo-peg-hole-clear.json --policy rule --rounds 2 --out runs/probe
# report 里 verifier_spend.sim_calls = 1 且没有 physics 证书 = 物理档真的跑了并通过
```
