# Repair Loop 规格（P0）

**目的**：把 harness 从「能跑单点验证器」推进到「有一条完整、可靠、可复现的 Agentic Repair Loop」。

**判定标准不是良率，是闭环。** 一个 3×5 的案例集能稳定跑完、能中断恢复、能量出预算账、能逐轮复现，就算这项能力完成。良率是下一阶段的评估指标，不是本阶段的验收条件。

每个模块必须能回答：**它提供的是 harness 的哪项通用能力**。不能回答的模块不进 P0。

---

## 1. 闭环形状

```
        ┌──────────────────────────────────────────────────────────────┐
        │                                                              │
Failed  │  Verification (cascade)                                      │
Assembly│        ↓                                                     │
State   │  FailureCertificate[]                                        │
   │    │        ↓                                                     │
   │    │  Policy: propose(asm, certs, history, budget_left)          │
   │    │        ↓                                                     │
   │    │  RepairPlan ──(scope/budget check by Engine)──▶ reject      │
   │    │        ↓                                                     │
   │    │  ToolExecutor: apply(asm, plan) ──▶ Assembly v+1            │
   │    │        ↓                                                     │
   │    │  Engine.record_round(...) ──▶ Round + Trace                 │
   │    └────────┘                                                     │
        │
        └──▶ EpisodeResult{assembly, certs, trace, verdict, budget_used}
```

**六个 P0 模块与它们各自的通用能力**：

| 模块 | 通用能力 | 不做什么 |
|---|---|---|
| `core/engine.py` | 唯一的领域写者：状态转移与不变量执行（范围/冻结/预算/幂等） | 不做技术判断，不碰几何 |
| `tools/verifiers/` | 统一的失败证据来源；验证器自检门禁 | 不决定修什么 |
| `policy/registry.py` | 动作空间可枚举、可校验、可组合的注册表 | 不选择动作 |
| `policy/loop.py` | 策略可替换的循环驱动器（规则 / LLM / 搜索同接口） | 不含领域规则 |
| `app/trace.py` | 全轨迹可复现：每轮输入输出与成本 | 不判定对错 |
| `cli/run.py` | 一条命令跑完一个 episode，含预算与恢复 | 不含策略 |

---

## 2. 接口（先定死，再实现）

### 2.1 Engine 的命令面

只暴露必需的命令，全部同步、单事务、幂等：

```python
class Engine:
    def open_ticket(case_id, mandate) -> Ticket
    def begin_round(ticket_id, plan) -> Round          # 校验范围/冻结/预算，失败抛 UserError
    def stage_round(round_id, stage: Literal["materialized","checked","simulated"]) -> None
    def close_round(round_id, *, verdict, evidence, cost, artifacts, edit_cost, preserve_ratio) -> Round
    def end_ticket(ticket_id, outcome, reason) -> Ticket
    def note_ticket(ticket_id, text) -> None           # 人工备注，不授予权限
```

**不变量（每次 `begin_round` 都查）**：
1. `plan.operator ∈ mandate.scope_operations`
2. `plan.targets ⊆ mandate.scope_parts` 且不与 `mandate.frozen_parts` 相交
3. `ticket.used + plan.cost_est ≤ mandate.budget`（三维度都要过）
4. `ticket.ended is None`

违反抛 `UserError(code)`，**不产生 Round**，**不消耗预算**。

### 2.2 Operator 注册表

```python
@dataclass(frozen=True)
class Operator:
    id: str
    params: type[BaseModel]
    preconditions: tuple[Callable[[Assembly, FailureCertificate], bool], ...]
    apply: Callable[[Assembly, dict], Assembly]
    targets: Callable[[dict], tuple[str, ...]]
    cost: Literal["cheap", "medium", "expensive"]
    family: Literal["geometry", "interface", "connector", "generative"]

class OperatorRegistry:
    def register(op) -> None
    def get(op_id) -> Operator
    def applicable(asm, certs) -> list[str]        # 前置条件全过的算子
    def apply(asm, plan) -> Assembly               # 校验参数后调用
```

参数用 pydantic 校验；**未注册的算子一律拒绝**——这是「Agent 不能发明动作」的代码落点。

### 2.3 Policy

```python
class Policy(Protocol):
    name: str
    def propose(self, asm, certs, history, budget_left) -> RepairPlan | None
```
`None` = 「无可行动作」，循环转人工。

三个实现：
- `RulePolicy`（P0，确定性）
- `LlmPolicy`（P1，走 `app/providers.py`）
- `RandomPolicy`（P1，baseline）

### 2.4 Trace

`runs/<run>/trace.jsonl`，一行一个事件，append-only：

```jsonc
{"seq":1,"at":"...","kind":"round.begin","ticket":"t1","round":0,"plan":{...},"policy":"rule"}
{"seq":2,"at":"...","kind":"round.stage","round":0,"stage":"materialized","artifacts":{"gen":"sha256:.."}}
{"seq":3,"at":"...","kind":"verify","round":0,"certs":[{"code":"gap_too_wide",...}],"verifier_ids":["connector_feasible@.."]}
{"seq":4,"at":"...","kind":"round.close","round":0,"verdict":"fail","cost":{...}}
{"seq":5,"at":"...","kind":"ticket.end","outcome":"exhausted","reason":"budget.rounds"}
```

**要求**：任意一行都能独立解析；`replay(trace)` 能重放出同一 `geom_hash` 序列。

---

## 3. 循环的停止条件（硬编码）

1. `verdict == pass` —— 全部硬验证器通过
2. 预算耗尽任一维度
3. `policy.propose` 返回 `None`
4. 环检测：新 `geom_hash` 命中历史 → 回滚该分支并停
5. 硬上限：`max_rounds` 兜底

停 = `end_ticket(outcome ∈ {repaired, exhausted, infeasible})`。

---

## 4. 异常与恢复（验收必须覆盖）

| 情形 | 行为 |
|---|---|
| 验证器抛异常 | 记 `environment` 族 `verifier_error`，该轮 `verdict=error`，**不进失败率**，继续下一轮或停 |
| 算子抛异常 | 同上，`operator_error` |
| 进程中断 | 重启后从 `store` 里最后一个 `closed` Round 之后继续；**已 close 的 Round 不许重跑** |
| 预算耗尽 | `end_ticket(exhausted)`，保留全部证据与轨迹 |
| 参数校验失败 | `UserError`，不产生 Round，不消耗预算 |
| 同一算子连续两次同一 plan | 环检测触发，停 |

---

## 5. 验收（M2 的 Definition of Done）

在一个**小案例集**上跑通，不是 100 个：

- **3 个真实装配案例**（从 `runs/m0-connector/rows.jsonl` 取有失败的）+ 2 个合成小案例（2 部件，秒级）；
- 每个案例：能跑完、有完整 trace、有最终结构；
- **中断恢复**：跑一半 kill 掉，重启后从断点继续且结果与不中断时一致；
- **预算**：给定 `rounds=3` 必须恰好停在 3，账目对得上；
- **可复现**：同 case + 同 seed + 同 policy 跑两次，trace 的 `geom_hash` 序列逐项相同；
- **异常**：注入一个必然抛异常的验证器，循环不崩、该轮记 `error`、后续轮继续。

**以上六条全过 = P0 完成。** 良率多少不影响这个判定。

---

## 6. 与数据集的关系（口径隔离）

- `runs/m0-connector/`（393/705）与 `runs/m1-min/`（124/238，19→25）**冻结在原处**，作为工具可信性验证与策略评测数据。
- Repair Loop 的验收**不依赖**这两份结果的任何数字。
- **连接件实体化成功率 ≠ 完整装配可行性**。前者是「能不能造出一个跨得过切缝的零件」，后者还要求生成树连通、可打印、可仿真。两套数分目录、分报表、分命名，**永不合并成一个「成功率」**。
- epsilon 留缝与 `d_cap` 专项**不在 P0 路径上**，结果进附录。

---

## 7. 里程碑重排

| 优先级 | 模块 | 状态 |
|---|---|---|
| **P0** | `core/engine.py`（Round 生命周期 + 不变量） | 待建 |
| **P0** | `policy/registry.py`（算子注册表 + pydantic 参数） | 待建 |
| **P0** | `policy/loop.py`（循环驱动器） | 待建 |
| **P0** | `app/trace.py`（全轨迹） | 待建 |
| **P0** | `cli/run.py`（一条命令跑 episode） | 待建 |
| **P1** | `LlmPolicy` + `providers.py` 接线 | 端口已就绪 |
| **P1** | `Evaluation Runner`（规则 vs LLM vs 随机，等预算） | 待建 |
| P2 | DiT Adapter / Manufacturing Validator | 不做 |
