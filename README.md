# PAH — Physics-Aware Harness

给 3D 可装配部件做的**agentic 修复闭环**：设计意图 → 几何物化 → 物理/几何证据 → 决策 → 动作 → 再验证。有状态、可复现、可扩展。

**先读这三份**：[docs/architecture.md](docs/architecture.md)（系统怎么搭）· [docs/contracts/v2-interfaces.md](docs/contracts/v2-interfaces.md)（接口与分工）· [docs/pah-dev-doc-v1.md](docs/pah-dev-doc-v1.md)（三支柱与 RQ）

## 它是什么 / 不是什么

| 是 | 不是 |
|---|---|
| 决策—工具—验证的闭环内核（Ticket / Mandate / Round / Budget / Trace） | 新的 3D 生成模型 |
| 让预训练 Agent 在**可信证据**与**硬约束**下改设计 | CAD 前端 |
| 可换成 DiT/CAD 的生成层**端口** | RL 训练框架 |
| 等预算可比、可复现的评测底座 | 某个数据集的连接件算法 |

核心不变量：**证据先于修复**（只有通过自检的验证器能写观测）、**预算是第一类对象**（含验证开销）、**内核零 I/O 且状态由事实派生**、**副作用先落库后执行**。

## 快速开始

```bash
pip install -e .                      # pah 命令
python -m pytest -q                   # 637 tests

pah design sanity                     # 5 个设计平面验证器自检（known_good/known_bad）
pah design new --kind peg_hole --out runs/demo

# 演示：插销 10.2 / 孔径 10.0 → 干涉 → 一轮修好
pah run --spec fixtures/design/demo-peg-hole-interfere.json --policy rule --rounds 3 --out runs/demo-rule
pah run --spec fixtures/design/demo-peg-hole-interfere.json --policy llm  --rounds 3 --out runs/demo-llm   # 需 DEEPSEEK_API_KEY

pah eval --suite fixtures/design/stress --policies rule,random --rounds 3 --out runs/eval-stress
python scripts/acceptance.py          # 六条 P0 验收判据（一条命令）
```

## 现在的实测状态（2026-11-01）

| 项 | 数字 |
|---|---|
| 契约案例 `peg 10.2 / hole 10.0` | rule **1 轮修复**（`set_clearance 0.35`），4 个失败码全清；LLM 策略同样 1 轮（真实调用，4254 tokens） |
| 等预算对比（4 个故意做坏的设计） | rule **4/4**、均值 1.00 轮 / 11 次几何调用；random **2/4**、2.25 轮 / 18.5 次 |
| P0 六条判据 | `scripts/acceptance.py` **6/6**（含中断恢复 hash 逐项相同、预算恰停、注入故障不崩） |
| 验证器 | 5 个（间隙 / 干涉 / 插入路径 / 可打印性 / MuJoCo 沉降），均有 known_good+known_bad 自检 |
| 测试 | **637 passed** |

口径纪律：`repaired_by_policy`、`environment_errors`、`not_evaluated`（能力边界，如未声明打印朝向、非凸件不评物理）三列**永不合并**；`not_evaluated` 不等于通过。

## 目录

```text
src/pah/
├── core/       # 实体、不变量、唯一写者（零 I/O）
├── app/        # store / effects / trace / providers / runtime（组合根）
├── design/     # DesignSpec + 三种连接件几何 + 物化 + 冻结契约 api.py
├── policy/     # 循环驱动器、算子表、rule / random / llm / case_memory
├── tools/      # 验证器（含 MuJoCo 沉降）、IVR、几何适配
├── eval/       # 等预算 runner + 分列报表
├── plugins.py  # 插件加载器（builtin / pah.toml / entry point）
└── cli/        # run / design / eval / plugins / probe
```
扩展一个新模块（DiT 适配器、新连接件、新验证器、新策略）不需要改 `core/`，见 [docs/plugin-api.md](docs/plugin-api.md)。

## 只读 / 冻结

- 数据集平面：`tools/verifiers/connector_feasible.py`、`policy/repair_script.py`、`policy/dcap.py`、`tools/eps.py`、`tools/ivr.py`、`runs/m0-connector/`、`runs/m1-min/` —— 它们是「适配器忠实」的证据（705 边逐边 0 差异、393 成功），不再投入。
- 外部数据目录（`D:\yanyi\rl_assembly\data\`）只读；新产物一律写 `runs/<id>/`。

## 下一步（未做，不是遗漏）

1. **评测集扩充**：正式结论要独立测试集（100 个 dev 集只够做探针；n=20 时 Wilson CI ≈ ±20 点）。
2. **LLM 策略的提示词工程**：目前只跑过 1 个真实调用 + ReplayLLM 录制。
3. **DiT 适配器**：`Generator` 端口已留；接入前必须先用 1 个样本验证模型能吃什么条件（见 doc §11 风险表）。
4. **case memory 落账**：`policy.flush(ticket.rounds)` 一行未接（Pass 轮不进记忆账），跨案例检索未做排序实验。
