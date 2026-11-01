# PAH — Physics-Aware Harness

装配可行性修复 harness。给 3D 可装配部件一个「物理失败 → 结构化证据 → 修复动作 → 再验证」的有状态闭环。

**设计文档：[docs/pah-dev-doc-v1.md](docs/pah-dev-doc-v1.md)**（先读 §0 一页纸，再看 §4 任务管理 / §5 决策 / §6 局部修改）

## 它不是什么

- 不是通用 3D 修复 harness（那条路已判死，撞 GATOR / Procedura）
- 不训练 DiT（只定义它的接口；v1 用参数化算子）
- 不是 RL 框架（只导出轨迹）

## 它是什么

三层可插拔内核，每一层对应论文一个 RQ：

| 层 | 端口 | 换它 = 换哪条实验 |
|---|---|---|
| `policy/` | `propose(assembly, failures, history, budget) -> RepairPlan` | RQ2 策略对比 |
| `tools/generators/` | `repair(assembly, plan, ctx) -> GenResult` | RQ3 局部 vs 全局 |
| `executors/` | `start/deliver/stop` | RQ4 串行 vs 并发 |

内核本身（`src/pah/core/`）是纯域：零 I/O、零第三方依赖、唯一写者、状态由事实派生。这是从 `dsh-superagent-prototype` 搬来的四条不变量，也是崩溃后状态不会撒谎的原因。

## 当前状态

- [x] 设计文档 v1
- [x] 域类型 + 阶段推导 + 内容哈希 ID（`pytest tests/` 全绿）
- [ ] M0 骨架与基线对齐：4 个验证器适配器 + `pah probe`，在 100 装配上跑出 388/705、86.8%、53/100
- [ ] M1 上帝返修探针
- [ ] M2 规则策略闭环

## 跑测试

```bash
python -m pytest
```

## 数据只读

`D:\yanyi\rl_assembly\data\` 只被 import 或子进程调用，**永不复制、永不写回**。新产物一律写 `runs/<id>/`。
