# PAH 架构（v2 · 设计平面）

> 状态：2026-11-01 v2 第一版在建 · 契约见 `docs/contracts/v2-interfaces.md`
> 上游设计：`docs/pah-dev-doc-v1.md`（三支柱）、`docs/internal/repair-loop-spec.md`（闭环规格）

## 1. 一句话

PAH 是一个**有状态、可复现、可扩展**的 Python 内核 + CLI：它把「设计意图 → 几何物化 → 证据 → 决策 → 动作 → 再验证」组织成受控闭环，让预训练 Agent 在**可信物理证据**和**硬约束**下修改 3D 装配设计。

PAH **不是**：新的 3D 生成模型；CAD 前端；某个数据集的连接件算法；RL 训练框架。

## 2. 两个平面，一个内核

```
┌─────────────────────────── 数据集平面（v1，冻结） ───────────────────────────┐
│ case.json（已有网格 + 接触边）→ connector_feasible → 55.7% 实体化            │
│ 证据价值：适配器忠实（705 边逐边 0 差异）、策略可评测（124/238）              │
│ 状态：只读。不再投入。runs/m0-connector、runs/m1-min 冻结在原处               │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │ 共用
                                    ▼
┌─────────────────────────── 设计平面（v2，主线） ─────────────────────────────┐
│ DesignSpec（连接方式 + 参数 + 位姿 + 公差）                                   │
│        │ materialize()                                                       │
│        ▼                                                                     │
│  几何（body / insert，确定性、内容寻址）                                      │
│        │ 验证器级联（cheap→expensive）                                        │
│        ▼                                                                     │
│  FailureCertificate[]（只有验证器能写 observed；environment 族 = 未知）       │
│        │ Policy（rule / random / best-of-n / LLM / case-memory）              │
│        ▼                                                                     │
│  RepairPlan（结构化动作：算子 + targets + 参数 + 理由；没有几何字段）          │
│        │ OperatorRegistry（未注册算子一律拒绝）                               │
│        ▼                                                                     │
│  DesignSpec'（新 revision）→ 回到物化                                          │
└──────────────────────────────────────────────────────────────────────────────┘
```

**两个平面共用同一个内核**：`core/`（实体、不变量、唯一写者）、`app/`（store / effects / trace / providers）、`policy/loop.py`（循环驱动器，只认四个 Protocol）。

## 3. 三层与三根支柱

| 层 | 内容 | 代码 |
|---|---|---|
| **L1 Physics-Aware Super-Agent**（管任务） | Ticket / Mandate / Assignment / Round / Budget / Trace / Effects | `core/*`、`app/*`、`policy/loop.py` |
| **L2 Physics-Grounded Repair Policy**（管决策） | 失败证书 → 修复计划的四层漏斗：规则 → 案例检索 → bandit → LLM | `policy/design_policies.py`、`policy/llm.py`、`policy/case_memory.py` |
| **L3 Physics-Conditioned Generator**（管局部几何） | 参数化物化（今天）/ CAD 内核（可插）/ 预训练 DiT（未来，适配器） | `design/connectors/*`、`design/materialize.py`、`Generator` 端口 |

## 4. 四条不变量（从 dsh-superagent 移植，与领域无关）

1. **内核零 I/O、同步、唯一写者**：`core/` 不做技术判断、不碰几何；状态由事实**推导**（`stage.py` 是纯函数），崩溃后状态不会撒谎。
2. **证据先于修复**：只有注册过且通过自检（known_good/known_bad）的验证器能写 `FailureCertificate`；策略只能提议。
3. **副作用先落库后执行**：`Effects` + `afterCommit`，重启不丢任务、不重放。
4. **预算是第一类对象**：rounds / geom_calls / sim_calls 三维硬顶；**验证的开销也记到轮次成本上**（`policy/loop.py::_charge`），否则等预算比较会奖励昂贵的验证级联。

## 5. 关键机制（v2 新增）

| 机制 | 落点 | 为什么 |
|---|---|---|
| 设计意图可哈希 | `DesignSpec.content_key()` / `geometry_hash()` | 环检测与恢复都靠内容身份，不靠时间戳或路径 |
| 几何确定性物化 | `design/materialize.py` | 同一 spec 必然得到同一 hash → 可复现、可跨 out_dir |
| 验证缓存 | `loop.py::_verified_or_run` | 每轮输出不再被验证两次；**失败/环境族不缓存**，瞬时故障可重试 |
| 验证成本记账 | `loop.py::_charge` + `Cascade.spent` | 物理档不免费，等预算公平 |
| 口径隔离 | 报表分目录/分命名 | 「连接件实体化成功」≠「完整装配可行性」，永不合并 |
| 未知不等于通过 | `family="environment"` | IVR/仿真/工具失败返回 unknown，不算 0 也不算通过 |

## 6. 扩展点（「持续维护、不断加模块」的落点）

五种插件，都不许改 `core/`：

| kind | 接口 | 加一个要做什么 |
|---|---|---|
| `connector` | `ConnectorBuilder` | 实现 `build(spec) -> ConnectorGeometry` + `params_model()`，注册 |
| `verifier` | `DesignVerifier` | 实现 `check(view)` + `fixtures()`（自检对强制），在 `DESIGN_CODES` 里占一个码组 |
| `operator` | `Operator`（v1 registry 同一形状） | 参数模型 + `apply(asm, params) -> Assembly`（纯 spec 编辑）+ `targets()` |
| `policy` | `LoopPolicy.propose(...)` | 一个 `name` + 一个提议函数，返回 `RepairPlan \| None` |
| `provider` | `LLMClient.complete(...)` | 走 `app/providers.py` 的注册表（不静默 fallback） |
| `decomposer` | `Decomposer.decompose(whole) -> Decomposition` | 整体网格进、N 个水密件出；本地切分器与腾讯云 X-Part 都是这个接口（见 `docs/multi-part.md`） |

加载：内置显式注册 / `pah.toml [plugins] modules` / Python entry points（group `pah.plugins`）；`pah plugins list` 可审计。详见 `docs/plugin-api.md`。

## 7. 自我迭代的路线（接口已留，机制未做）

用户要的「越用越强」不走训练，走**经验检索**与**策略阶梯**：

1. **Case memory**（已实现第一版）：`(kind, 证书码, 动作) → 成败次数`，SQLite 记账，检索优先复用成功动作——这是插件，不是内核改动。
2. **Bandit / 检索排序**：同一张表换打分函数。
3. **BC / Offline RL**：只导出轨迹（`trace.jsonl` 已经是训练格式），训练发生在 PAH 之外。
4. **DiT 适配器**：`Generator` 端口已定义，接入前必须先验证模型能否吃区域条件输入（doc §11 风险表）。

**纪律**：任何阶段都不改「证据先于修复」与「验证器唯一写者」。

## 8. 目录（v2 现状）

```text
src/pah/
├── core/          # model, engine, stage, ids, ports, errors   ← 不变量
├── app/           # store, effects, trace, providers, runtime    ← 组合与 I/O
├── design/        # spec, api(冻结契约), connectors/, materialize,
│                  # decompose(Decomposer 端口), pairing, machining, pairs ← 多件
├── policy/        # loop, registry, rule, design_ops, design_policies, llm, case_memory
├── tools/
│   ├── verifiers/ # base, connector_feasible(v1 冻结), fit, interference, path, printability, physics, design_cascade
│   ├── fbx.py     # Blender 无头调用：STL/GLB ⇄ FBX（混元组件生成只收 FBX）
│   ├── tencent_ai3d.py  # 组件生成 API 客户端（COS 上传 + submit/poll/download）
│   ├── ivr.py, eps.py   ← v1 冻结
│   └── physics/   # MuJoCo 沉降实验
├── eval/          # runner, report（等预算对比）
├── plugins.py     # 插件加载器
└── cli/           # main, run, build, decompose, design, eval, export, fbx, plugins, probe(v1)
```

## 9. 验收（v2 第一版）

小案例集（3 合成 + 2 真实），六条：能跑完并留 trace / 中断恢复一致 / 预算恰好停在 N / 同 spec 两次跑出相同 `geom_hash` 序列 / 注入必崩验证器不崩循环且该轮记 error / 报表口径分列。

**良率多少不影响这个判定。** v2 的第一性目标是「闭环可信」，良率是下一阶段的评估指标。
