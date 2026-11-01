# 插件 API —— 新增一个模块要做什么

> 契约依据：`docs/contracts/v2-interfaces.md` §6（插件面）与 §9（三条纪律）。
> 代码依据：`src/pah/plugins.py`、`src/pah/plugins_builtin.py`、`src/pah/app/runtime.py`。
> 已落地的样例：`src/pah/policy/case_memory.py`（T5，`kind="policy"`）。

这份文档是**用户说的那个入口**：以后要加「自我迭代机制」、case memory、DiT 适配器、
新的几何内核、新的验证器，都从这里开始。**加模块不需要改 `core/`，也不需要改任何冻结文件。**

---

## 0. 一句话版本

新建一个普通 Python 模块，模块级写三样东西，然后让 harness 找到它：

```python
api_version = "1"                 # 声明你写的契约版本；不匹配会被按名字拒绝
kind = "policy"                   # 五选一：connector / verifier / operator / policy / provider
def register(host):               # 唯一入口；往 host 的五本注册表里加东西
    host.policies["my_policy"] = my_factory
```

找不到你的模块时不会有任何「静默不生效」：`pah plugins list` 会把它列成 FAIL 并说明
是哪个模块、哪一行原因（import 失败 / api_version 不符 / 没有 register / register 抛异常 /
什么都没注册 / 声明的 kind 和实际注册的不一致）。

---

## 1. 三步 checklist

### 第一步：决定 kind

| kind | 你要提供 | 注册到 | 最小可跑样例 |
|---|---|---|---|
| `connector` | 一个 `ConnectorBuilder`（`kind` / `params_model()` / `clearance_mm()` / `build(spec)`） | `host.connectors[kind] = builder` | 一种新连接件（如榫卯、卡扣变体） |
| `verifier` | 一个 `DesignVerifier`（`spec: VerifierSpec` + `check(view)` + `fixtures()`） | `host.verifiers.register(v)` | 新判据（如圆角半径、装配顺序约束） |
| `operator` | 一个 `policy.registry.Operator`（`id` / `params` / `preconditions` / `apply` / `targets` / `cost` / `family`） | `host.operators.register(op)` | 新修复动作（如改倒角、加圆角） |
| `policy` | 一个工厂 `factory(**, ...) -> LoopPolicy`（有 `name` 和 `propose(...)`） | `host.policies[name] = factory` | **自我迭代机制**、case memory、best-of-N 搜索 |
| `provider` | 一个 `app.providers` 能用的 client（或 `ProviderConfig`） | `host.providers[id] = ...` | 新的模型服务接入 |

> 一个模块只能声明一个 `kind`，但 `register` 里可以注册多个条目（例如一个 bundle）。
> 如果实际注册的 kind 与声明的完全不相交，会被判 FAIL —— 这是防止「policy 插件偷偷装了一个没人审计的 operator」。

### 第二步：写模块

硬性要求（每条都在代码里有对应检查）：

- `api_version = "1"`：写死的字符串。harness 升级契约版本时，旧插件会被**按名字**拒绝，而不是在三层调用栈里 TypeError。
- `register(host)` 必须存在且可调用；它只允许碰 `host` 暴露的五个对象。
- **不要 import `pah.core.*` 来改状态**：core 只有类型和规则。要改变几何就产出新的 `DesignSpec` / 新 `Assembly` 版本，让组合根去物化。
- **不要自己写文件到 episode 目录**：`round-<n>/spec.json`、`trace.jsonl`、`geometry/` 归 runtime 和物化器；插件写自己的状态（如 case memory 的 SQLite）请放在 `runs/` 或环境变量指定的路径。

按 kind 的额外纪律：

- **verifier**：必须带 `known_good` / `known_bad` 一对 fixtures（`DesignVerifierRegistry.sanity()` 会跑；不过就被禁用）。只能写 `observed_*`，不许给结论。抛异常 = 未知（`environment` 族），**永远不算通过**。
- **operator**：`apply(asm, plan)` **必须是纯的**——只改 spec（写进 `assembly.constraints["design_spec"]`），用 `assembly_from_spec(..., parent=旧id, version+1)` 造新版本。不写盘、不建几何、不调 materializer。`targets()` 必须是**参数的纯函数**；`RepairPlan` 的 `targets` 和 `params["parts"]` 必须一致，否则被 `OperatorRegistry` 的防伪闸拒绝。
- **policy**：只能**提议**，不能宣告成功。返回 `None` = 「没有我愿意做的动作」，这是正常答案。要读历史就从 `history: list[Round]` 读，不要自己存跨 episode 的隐式状态（case memory 是显式 SQLite 账本，不是模块全局变量）。
- **connector**：`build(spec)` 返回两个世界坐标网格 + `mate_axis` + `approach_mm`；几何身份由 `design/api.geometry_hash` 决定，**不要**改成按 STL 字节哈希。
- **provider**：凭据只写环境变量名，**永不落盘、永不打印**；计费形态必须声明（`billing`），别假装有价格。

### 第三步：让 harness 找到它（三选一）

1. **内置**（随仓库发布）：把模块路径加进 `src/pah/plugins_builtin.py` 的 `BUILTIN_MODULES`。
   文件还没落地时会被跳过，不会让 `pah plugins list` 变红。`case_memory` 就是这么挂的。
2. **配置**（本机/实验用）：`pah.toml` 里
   ```toml
   [plugins]
   modules = ["mypkg.my_module", "mypkg.self_iteration"]
   ```
   然后 `pah plugins list --config pah.toml`（或设 `$PAH_CONFIG`）。
3. **入口点**（装成包给别人用）：在 `pyproject.toml` 里
   ```toml
   [project.entry-points."pah.plugins"]
   my_module = "mypkg.my_module"
   ```

---

## 2. 验证你的插件

```bash
pah plugins list                      # 看到你的行：kind / source / api_version / 声明的算子 id 或证书码
pah plugins list --json               # 机器可读，适合 CI
pah design sanity                     # 若你是 verifier：跑 §4.8 门禁
pah run --spec fixtures/design/demo-peg-hole-clear.json --policy 你的名字 --rounds 2 --out runs/try
pah eval --suite fixtures/design --policies rule,你的名字 --rounds 3 --out runs/eval
```

`pah run` 里 policy 插件的名字和内置 `rule|random|llm` 完全同权；解析顺序是
**插件优先，然后 T3 的 builder**。写错名字会 exit 3 并列出「本次运行加载了哪些插件」。

---

## 3. 失败长什么样（都是 FAIL 行，不会崩 CLI）

| 现象 | 你会看到的 `error` |
|---|---|
| 模块 import 失败 | `import failed: ModuleNotFoundError: No module named '...'` |
| 契约版本不符 | `declares api_version '2'; this harness serves '1' (upgrade the plugin, not the harness)` |
| kind 拼错 | `declares kind 'weld'; the contract allows ('connector', 'verifier', 'operator', 'policy', 'provider')` |
| 忘了写 register | `has no register(host) callable` |
| register 里崩了 | `register(host) raised RuntimeError: ...` |
| register 什么也没加 | `register(host) added nothing to the host` |
| 声明 kind 与实际不符 | `declares kind 'verifier' but registered ['policy']` |
| 重复注册同一个 id | 注册表自己抛 `... is already registered`（**不是覆盖**，按契约必须报错） |

**WARN 与 FAIL 的区别**（`pah plugins list` 的 STATUS 列）：

- `FAIL` = 你在 `pah.toml` / entry point 里**声明**了这个插件，但它没加载起来。
  这是请求与实现不一致：`pah run` / `pah eval` 会直接拒绝开跑（exit 3），
  因为一个动作空间和请求不同的 episode，其数字不能进对比表。
- `WARN` = `plugins_builtin.py` 里列的**随仓库发布**的模块没加载起来。它是警告：
  那次 episode 少一项能力，report 的 `plugins.warnings` 和 `notes` 会写下来，但不会
  因为一个还没落地的模块把整个 CLI 拖红。落地后自动变成 OK，不需要改代码。

`pah plugins list` 的退出码：`0` 没有 FAIL 行（WARN 不影响），`1` 有 FAIL 行（表里写了原因），
`3` 请求本身坏了（`pah.toml` 不是 TOML、`--config` 指向不存在的文件）。

---

## 4. 加「自我迭代机制」怎么落

按契约，自我迭代不属于 `core/`，它属于 policy 层的插件。推荐形状（与 `case_memory` 同构）：

```python
# src/pah/policy/self_iteration.py
api_version = "1"
kind = "policy"

def register(host):
    host.policies["self_iteration"] = lambda **kwargs: SelfIteratingPolicy(
        inner=_rule_policy(host.operators),      # 会做事的底层策略
        ledger=AttemptLedger(_default_path()),   # 显式落盘的尝试账本
        registry=host.operators,
    )
```

- **读什么**：`propose(assembly, certs, history, budget_left)` 的四个参数就是全部合法输入。
  `history` 里有每一轮的计划、verdict、证据和成本——「上次同类证据用了哪个算子、花了多少、成没成」从这里学。
- **写什么**：只在 `register` 时打开自己的账本（SQLite/JSONL 皆可），不要改 `Assembly`/`Trace`/store。
- **不许**：改 `core/engine.py` 的预算、改验证器的判定、把「模型建议」写成 verdict。
- **怎么证明它有效**：`pah eval --policies rule,self_iteration --rounds 3`，看
  `repaired_by_policy` 与 `mean_geom_calls` 两列；口径列（`environment_errors`/`unknown`）单列，
  不要把工具健康混进成功率。

---

## 5. 三条不许破的纪律（摘自契约 §9）

1. **冻结模块只读**：`core/*`、`app/{store,effects,trace}.py`、`policy/{loop,registry}.py`、
   `design/{spec,api}.py` 只许 import。要加能力 = 加插件，不是改内核。
2. **验证器不给结论**：只有验证器能写 `observed_*`；策略只能提议。
3. **不训练模型**：决策只是结构化动作，LLM 的输出类型里没有几何字段。
```
