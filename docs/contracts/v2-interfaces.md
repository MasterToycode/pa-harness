# PAH v2 接口契约（设计平面）

> 冻结时间：2026-11-01 · 状态：**冻结，改动须经 Lead**
> 契约代码：`src/pah/design/spec.py`、`src/pah/design/api.py`（这两份文件是唯一事实来源；本文只做索引与分工）
> 上游设计：`docs/pah-dev-doc-v1.md` §4–§8、`docs/internal/repair-loop-spec.md`

---

## 0. 这一版在解决什么

v1 的闭环已经有了（engine / registry / loop / trace / verifiers），但它的**输入**是 `case.json`：已经存在的网格 + 已经存在的接触边。于是所有精力都被数据集里的几何缺陷吃掉，Agent 没有可决策的东西。

v2 把主向量换掉：**输入变成 `DesignSpec`——设计意图本身**（连接方式、参数、位姿、公差），几何由它确定性地物化出来。这样：

- Agent 决策的是「选哪种连接、把孔径改到多少、要不要换类型」，不是「补哪条边」；
- 验证器量的是**它刚建出来的东西**，不是数据集里别人的东西；
- 数据集（M0/M1）降级为「适配器忠实性证据」，不再挡主线。

一句话：**v2 的目标是「需求 → 可装配部件」的闭环，不是「数据集良率」。**

---

## 1. 平面的三个对象

```
DesignSpec ──materialize()──▶ MaterializedDesign{assembly, meshes, paths}
                                       │
                                       ▼
                        DesignView ──verifiers──▶ FailureCertificate[]
                                       │
                                       ▼
                              Policy ──▶ RepairPlan ──▶ SpecOperatorRegistry
                                       │                      │
                                       └──── 新的 DesignSpec ──┘（下一轮）
```

| 对象 | 位置 | 归属 | 谁只能读 |
|---|---|---|---|
| `DesignSpec` / `PartSpec` / `InterfaceSpec` / `FitConstraints` / 三种连接参数模型 | `design/spec.py` | Lead（冻结） | 全员 |
| `ConnectorBuilder` / `ConnectorGeometry` / `DesignView` / `DesignVerifier*` / `MaterializedDesign` / `Materializer` / `assembly_from_spec` / `spec_of` | `design/api.py` | Lead（冻结） | 全员 |
| 三个内置连接件的几何 | `design/connectors/*.py`、`design/materialize.py` | **T1** | 全员 |
| 五个设计平面验证器 + 级联适配器 | `tools/verifiers/{fit,interference,path,printability,physics}*.py` | **T2** | 全员 |
| 设计平面算子 + 策略（rule/random/best-of-n/llm/case） | `policy/design_ops.py`、`policy/design_policies.py`、`policy/llm.py` | **T3** | 全员 |
| 组合根 / CLI / eval / 插件系统 | `app/runtime.py`、`cli/{run,design,eval,plugins}.py`、`eval/`、`plugins.py` | **T4** | 全员 |

**禁止**：任何人改动 `tools/verifiers/connector_feasible.py`、`policy/repair_script.py`、`policy/dcap.py`、`tools/eps.py`、`tools/ivr.py`（M0/M1 冻结证据），以及 `runs/m0-connector/`、`runs/m1-min/` 下的任何数字。

---

## 2. 冻结接口（摘要，签名以代码为准）

```python
# design/spec.py
DesignSpec(spec_id, requirement, parts: tuple[PartSpec,...], interfaces: tuple[InterfaceSpec,...],
           constraints: FitConstraints, revision=1, provenance={}, api_version="1")
  .body / .insert / .interface / .clearance_mm
  .edit(**fields) / .with_interface(kind=?, params=?) / .with_part(PartSpec)
  .content_key() -> "sha256:..."      # 只覆盖能改变几何的字段
InterfaceSpec(id, kind, a, b, params: dict).validated_params() -> BaseModel
PartSpec(id, role: "body"|"insert", size_mm, pose_mm, rot_deg, material)
FitConstraints(min_clearance_mm, max_clearance_mm, min_wall_mm, process, target_fit).in_band(mm)
PegHoleParams(d_peg_mm, d_hole_mm, engage_mm, approach_mm, peg_len_mm?, feature_xy_mm)
SnapFitParams(hook_w_mm, hook_h_mm, beam_len_mm, beam_thk_mm, slot_w_mm, engage_mm, approach_mm)
DovetailParams(width_mm, depth_mm, angle_deg, clearance_mm, engage_mm, approach_mm)

# design/api.py
ConnectorBuilder: kind; params_model(); clearance_mm(params); build(spec) -> ConnectorGeometry
ConnectorGeometry(meshes: dict[part_id, Trimesh], mate_axis=(0,0,1), approach_mm)
MaterializedDesign(spec, assembly, meshes, paths, root).view() -> DesignView
Materializer = Callable[[DesignSpec, Path], MaterializedDesign]
DesignView(spec, root?, meshes?).mesh(id) / .body / .insert        # 惰性加载 root/geometry/<id>.stl
DesignVerifier: spec: VerifierSpec; check(view) -> list[FailureCertificate]; fixtures() -> (good, bad)
DesignVerifierRegistry: register/get/ids/sanity/check(view, stop_on_hard=False)/disabled
assembly_from_spec(spec=, geometry_paths=, geometry_hashes=, parent=None, version=1) -> Assembly
spec_of(assembly) -> DesignSpec
```

**顺带被继承、不许改的行为**：`Assembly` 是指数，`RepairLoop`（`policy/loop.py`）不用改——它只认 `LoopEngine/LoopVerifiers/LoopRegistry/LoopPolicy` 四个 Protocol，v2 全部满足。

---

## 3. 证书码表（设计平面，闭集）

| 验证器 id | cost | 码 | 判据 |
|---|---|---|---|
| `fit_clearance` | cheap | `insufficient_clearance` `excessive_clearance` | 设计间隙 vs `constraints` 的 [min,max] 带 |
| `interface_interference` | medium | `geometric_interference` | 装配 pose 下两实体布尔交体积 > 1e-6 mm³ |
| `insertion_path` | medium | `insertion_path_blocked` | 沿 mate_axis 从 approach 到 seat 采样扫掠，任一位置有交 |
| `printability` | medium | `below_min_wall` `unsupported_overhang` | 孔壁到外表面最小厚度、悬垂角 vs 工艺 |
| `physics_settle` | expensive | `joint_unstable` `dynamic_penetration` `physics_error` | MuJoCo 重力沉降后相对位移/穿深 |

`VerifierSpec(id, codes, cost_class, fixtures)` 复用 `tools/verifiers/base.py` 的那一个；`fixtures` 由各验证器自己的 `fixtures()` 返回 `(known_good, known_bad)` 两份 `DesignView`。

`physics_error` 与 `verifier_error` 一律 `family="environment"`：**不进失败率，不算通过**。

---

## 4. 动作空间（设计平面算子，7 个）

| 算子 id | 参数 | 触发码 | targets |
|---|---|---|---|
| `set_clearance` | `interface_id, clearance_mm` | `insufficient_clearance` `excessive_clearance` | [a,b] |
| `resize_feature` | `interface_id, delta_mm` | `insufficient_clearance` `insertion_path_blocked` | [a,b] |
| `resize_insert` | `interface_id, delta_mm` | `geometric_interference` | [a,b] |
| `shift_part` | `part_id, delta_mm[3]` | `geometric_interference` `insertion_path_blocked` | [part_id] |
| `rotate_part` | `part_id, delta_deg[3]` | `insertion_path_blocked` | [part_id] |
| `switch_connector` | `interface_id, kind, params` | 任何 hard code（换范式） | [a,b] |
| `regenerate_interface` | `interface_id, params`（局部再生成；将来 DiT 的挂点） | 任何 hard code | [a,b] |

规则：**未注册的算子一律拒绝**；参数用 pydantic `extra="forbid"` 校验；`targets()` 必须是参数的纯函数（复用 `policy/registry.py` 的防伪校验思路）。

---

## 5. CLI 面（第一版就这些）

```text
pah design new  --requirement "..." --kind peg_hole --out runs/demo     # 需求 → spec.json（可选 LLM）
pah run         --spec runs/demo/spec.json --policy rule --rounds 3 --out runs/demo/ep0 [--resume]
pah eval        --suite fixtures/design --policies rule,random,llm --rounds 3 --out runs/eval1
pah plugins list
pah verify sanity                      # 既有：数据集平面验证器自检
pah design sanity                      # 新增：设计平面验证器自检（known_good/known_bad）
```

退出码沿用 `cli/main.py` 的 0/1/2/3。

---

## 6. 插件面（可持续维护的落点）

一个插件 = 一个模块，声明 `api_version = "1"` 与 `kind ∈ {connector, verifier, operator, policy, provider}`，提供 `register(registry)` 或一个工厂。加载方式（全部可选）：

1. 内置：`pah/plugins_builtin.py` 里显式注册；
2. 配置：`pah.toml` 的 `[plugins] modules = ["mypkg.mod"]`；
3. 入口点：`importlib.metadata.entry_points(group="pah.plugins")`。

`pah plugins list` 打印每个插件的 kind、来源、api_version、声明的码/算子。**新增模块（DiT 适配器、case memory、自我迭代机制）不得修改 `core/`**——只能新增插件并注册。

---

## 7. 验收（本轮 Definition of Done）

小案例集：**3 个合成设计（三种连接件）+ 2 个真实装配（从 `runs/m0-connector` 取有失败的）**，不是 100 个。

1. `pah run` 能跑完一个 episode：spec → 失败证书 → 算子 → 新材料化 → 再验证 → 结论，全程有 trace；
2. **中断恢复**：跑一半 kill，重启 `--resume`，结果与不中断一致（`geom_hash` 序列相同）；
3. **预算**：`--rounds 3` 恰好在 3 停，账目对得上；
4. **可复现**：同 spec + 同 seed + 同 policy 跑两次，`geom_hash` 序列逐项相同；
5. **异常**：注入必抛异常的验证器，循环不崩、该轮 `error`、后续轮继续；
6. **口径隔离**：报表里「设计平面修复结果」与 M0/M1 数据集数字分目录、分命名，永不合并。

**验收案例的主戏**：`PegHoleParams(d_peg=10.2, d_hole=10.0)` → 判 `geometric_interference` + `insufficient_clearance` → 策略给出 `set_clearance` 或 `resize_feature` → 再验证通过 → 导出可用 STL。这就是提案里的 Peg-Hole 演示，但它是**建出来的**，不是从数据集里挑出来的。

---

## 8. 分工与写入范围（并行开发）

| 成员 | 写入范围 | 首交付 |
|---|---|---|
| **T1 设计几何** | `src/pah/design/connectors/**`、`src/pah/design/materialize.py`、`tests/test_design_*.py`、`fixtures/design/*.json` | `materialize(spec, out_dir)` 跑通三种连接件 + `pah design sanity` 可用的 fixtures |
| **T2 验证器** | `src/pah/tools/verifiers/{fit,interference,insertion_path,printability,physics}*.py`、`src/pah/tools/verifiers/design_cascade.py`、`tests/test_design_verifiers.py` | 五个验证器 + `fixtures()` 自检对 + 级联适配器（`check(assembly)` 接口） |
| **T3 策略与算子** | `src/pah/policy/design_ops.py`、`src/pah/policy/design_policies.py`、`src/pah/policy/llm.py`、`tests/test_design_policy*.py` | 7 个算子 + rule/random/best-of-n/llm 四个策略，同接口 |
| **T4 运行时/CLI/评测/插件** | `src/pah/app/runtime.py`、`src/pah/cli/{run,design,eval,plugins}.py`、`src/pah/eval/**`、`src/pah/plugins.py`、`docs/plugin-api.md` | `pah run` 端到端 + `pah eval` 等预算对比 + 插件加载器 |

**不重叠、不互相改文件。** 需要别人改动时，发消息给 Lead，不直接动。

---

## 9. 三条不许破的纪律

1. **冻结模块只读**（§1 末）；数据集数字不许改口径、不许合并。
2. **验证器不给结论**：只有验证器能写 `observed_*`；策略只能提议，不能宣告成功。
3. **不训练模型**：v2 只用预训练 LLM 做决策（方案 A），DiT 只留适配器接口；LLM 的输出类型里没有几何字段。
