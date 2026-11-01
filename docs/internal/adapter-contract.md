# 适配器契约：`rl_assembly` 四模块的真实调用面

只读分析产物。所有结论要么带 `file:line`，要么带数据文件路径；「注释/报告说的」与「代码实际做的」冲突时**两者都写**。
字段名对齐 `src/pah/core/model.py:104`（`FailureCertificate`）。本文件不改动 `rl_assembly` 任何代码。

**先说三条与任务书假设不符的事实**：

1. 两个被点名的数据文件**不存在**：`data/assemblies/*/` 里没有 `contact_graph.json`，也没有 `instruction.json`（全仓 `glob` 零命中）。接触图是**现算**的，设计意图在场的是 `constraints.json`（4 键）与代码里的 `env.INSTRUCTIONS`。
2. 题面的「12 条 reason」里有 5 条**不是** `reason` 字段的值，而是 `detail` 字符串前缀或 `degen_code` 字段。顶层 `reason` 只有 7 个。
3. `data/assemblies` 是 `E:\rl_assembly-data` 的 junction（`Get-Item ... .Target = E:\rl_assembly-data`），几何单位是**米**（实测 `part_000.ply` 体积与 `meta.json` 体积逐位相等，最长边 0.225 m）。仓库里同时存在**两套尺度**：`assemblies/`（225 mm，diag 0.326 m）与 `assemblies_pre_decimate/`（150 mm，diag 0.217 m）。

---

## A. 四个验证器的真实签名

### A1. `connector.py` —— 连接件实体化

```python
build(part_a, part_b, edge: dict, theta, mode: str = "peg_socket", *,
      clamp: bool = True, verify_stl: bool = True, overlap_check: bool = True,
      sections: int = SECTIONS, unit_mm: float | None = None,
      r_min_mm: float = R_MIN_MM, depth_min_mm: float | None = None,
      host_quality: dict | None = None, section: str = "D",
      snap: bool | None = None) -> dict          # connector.py:575-580
```

配套入口：`feasibility(edge, unit_mm, r_min_mm=0.25, depth_min_mm=None, t_a=None, t_b=None)`（`:414`）、`radius_cap`（`:373`）、`depth_cap`（`:394`）、`measure_axis(part_a, part_b, edge, theta=None)`（`:498`，纯量测不跑布尔）、`infer_unit_mm(*meshes)`（`:448`）、`host_unreliable(info)`（`:121`）、`line_intervals(mesh, origin, n, tol=None) -> (list[(t0,t1)], odd)`（`:227`）、`edge_census`/`fragment_share`/`stl_roundtrip`（`:187/:196/:205`）。

- **输入几何**：两个 `trimesh.Trimesh`，**同一装配坐标系、米**。`edge` 需含 `center`（数据单位/米）；可选 `n_ab`、`anchor`、`half_w`、`half_h`、`r_inplane_max`、`depth_max`、`t_a`、`t_b`、`r_max`（`_resolve_frame`，`:534-569`）。`theta = (u, v, nx, ny, nz, r, depth)`，7 维，`r`/`depth` 是**数据单位**（米），`depth` 是「露出 A 外表面的功能啮合长度」，不是绝对插入深度（`:590-593`）。无 `numpy` shape 约束以外的结构体，全部是 `dict`。
- **输入语义**：`n_ab` 由 A 指向 B（`contact_graph` 保证，`test_contact_graph.py:95`）。`anchor` 存在时 `(u,v)=(0,0)` 以它为基准，否则退回 `center`（`:562-569`）。
- **返回**：**永不抛异常**（`:29` 明文规定，`:652-659` 兜底 `except`）。成功 `{"a_out", "b_out", "ok": True, "reason": "", "case": "ok", "feasible": True, ...diag}`（`:955-959`）；失败 `{"a_out": None, "b_out": None, "ok": False, "reason": <7 类之一>, "case": reason, "detail": str, ...}`（`_fail`，`:969-973`）。诊断字段随路径变化：`unreatable_too_small` 路径给 `r_cap,d_cap,t_a,t_b,anchor_shift`；进入布尔的路径给 `r,depth,gap,penetration,base_t,tip_t,sock_bot_t,wall_remaining,r_clamped,depth_clamped,degen_code,nm_a,nm_b,dv_a,dv_b,...`（`:833-843`、`:770-786`）。
- **副作用**：无文件 IO（`stl_roundtrip` 只在 `BytesIO` 里往返）、无全局状态、不需要 GPU、不读 cwd。硬依赖 `manifold3d`（缺则 `bool_failed/no_manifold3d`，`:621`）、`trimesh` 的 `mesh.ray`（需 rtree）、`shapely`（D 形多边形，`:350-353`）。
- **耗时**：报告实测「`to_manifold` 0.03–0.7 s；union+diff 合计 0.1–5 s；`measure_axis` 首次 7 s、之后 0.03 s——首次慢是射线索引一次性构建」（`research/connector-boolean-report.md:229`，15 万面部件）；全量 946 条边约 20 分钟（同文件 `:641`），即 ~1.2 s/边。代码内无耗时注释。

### A2. `contact_graph.py` + `refine_edges.py` —— 接触图与锚点

```python
build(parts: list[trimesh.Trimesh], diagonal: float, ground_parts: list[int],
      delta: float | None = None, *, eps: float = 0.0,
      fine_parts: list[trimesh.Trimesh] | None = None) -> dict   # contact_graph.py:1235-1237
build_graph(..., level: str = "mesh", *, eps=0.0, fine_parts=None)  # :1275
build_with_stats(parts, diagonal, ground_parts, delta=None, level="mesh", *, eps=0.0,
                 fine_parts=None) -> tuple[dict, dict]            # :1288
assembly_diagonal(parts) -> float                                 # :1296
```

```python
refine(edges_raw, assembly_dir, *, k=K, scale_mm=SCALE_MM, base_mm=BASE_MM,
       r_min_mm=R_MIN_MM, thickness_root=None, wall_ratio=WALL_RATIO,
       anchor_tol_mm=ANCHOR_TOL_MM) -> dict      # refine_edges.py:117-120
```

- **输入几何**：`parts` 是**装配坐标系、米**、互不穿插的裁剪件；`diagonal` 由 `assembly_diagonal` 或 `meta["diagonal"]` 给；`ground_parts` 是接地件下标（来自 `meta["ground_parts"]`）。`delta` 缺省 `max(0.005·diagonal, 2·eps)`（`:1010`）。`fine_parts` 必须是**同序、同长**的未简化件（`:1025-1028`），只供测厚射线使用（`:1259-1265`）。
- **返回（`_build`，`:1209-1215`）**：`{"diagonal", "delta", "ground_parts", "nodes", "edges"}`。`nodes[i] = {id, mass, centroid[3], thickness}`（`:1202-1207`），`mass = |volume| × 1000 kg/m³`（`:94, 236`）。`edges[i]` 的键**精确等于** `{id, a, b, area, n_ab[3], center[3], anchor[3], fitness, half_w, half_h, r_inplane_raw, r_inplane_max, anchor_gap, anchor_fallback, inplane_method, depth_max, t_a, t_b, n_samples, n_patches, patch_area_share, interface_type, measurement_lod, thickness_failed, edge_quality, load}`（`:1149-1181`；`test_contact_graph.py:413-419` 是等价的权威清单）。`a < b`，`half_w >= half_h`，`depth_max == 0.8·min(t_a,t_b)`（`:1163`）。
- **返回（`stats`）**：`{level, n_parts, n_edges, n_candidate_pairs, thickness_fallbacks, thickness_rejections, anchor_fallback_count, mean_samples_per_edge}`（`:1216-1226`）。`thickness_rejections` 的键是 `no_hit` / `grazing` / `no_span` / `all_slivers` / `zero` / `too_thick` / `no_mid`（`:866-886`、`:906`）——**不是** harness 词表。
- **失败方式**：**抛 `ValueError`**。`diagonal<=0`、`eps<0`、`delta<=0`、`level` 非法、`fine_parts` 长度不符、接地件越界（`:1005-1032`）；单条边测厚失败不抛，落成 `t=0 + thickness_failed=True`（`:1138-1146`）。
- **副作用**：纯几何、无文件 IO（`:7`）。有对象级缓存：`mesh.ray` 由 trimesh 挂在 mesh 上、`_Part` 的 KD 树按需构建（`:247-289`）——**复用同一批 `Trimesh` 实例**是性能前提。
- **耗时**：单装配端到端 mean 19.0 s / max 110 s，100 装配 1907 s（`research/contact-graph-report.md:187`）。
- **`refine` 额外做的三件「项目决策」**：`r_eff = min(k·r_inplane_raw, t_b/2.25) ≥ 0.25 mm`（`:162-193`）、锚点自检 `≤ 0.05 mm`（`:205-231`）、毫米换算 `×1000×f`（`f = 225/150 = 1.5`，`:162`）。**它自己会重新 `cg.build` 一遍**（`:160`），传入的 `edges_raw` 只用来取 `(a,b)`（`:126-130`）。

### A3. `assembly_env.py` —— MuJoCo 装配仿真

```python
load_assembly_parts(assembly_dir, *, clearance=3e-3, use_fine=False) -> (list[PartInfo], dict)  # :100-101
load_contact_graph(path, assembly_id) -> list[dict]          # :237
edges_from_records(records, meta, parts) -> list[EdgeInfo]   # :279
synthesize_edges(parts, meta, *, delta_fraction=0.005)       # :145（兜底，非正式数据）
normalize_connectors(connectors, edges) -> ConnectorSpec     # :366
d_shape_peg(radius, length, flat_depth=None, ...)            # :442

AssemblyEnv(assembly_dir, contact_graph_path=None, *, backend="mujoco", clearance=3e-3,
            sim_seconds=SIM_SECONDS, dt=DT, use_fine=False, instr=1, edge_ids=None,
            friction=None, n_pegs_per_edge=1, peg_model="rigid",
            peg_stiffness_scale=1.0, edge_scale=1.0)         # :482-488
AssemblyEnv.evaluate(connectors, *, use_connectors=True, gravity_flip=False,
                     include_ground=True, sim_seconds=None, noise=0.0,
                     generator=None) -> dict                 # :705-708
AssemblyEnv.build_mjcf(connectors, *, use_connectors=True, gravity_flip=False,
                       include_ground=True, stiffness_scale=1.0) -> (str, np.ndarray)  # :564-566
AssemblyEnv.parts_info() -> dict   # :778      AssemblyEnv.summary() -> dict  # :788
```

- **输入**：`assembly_dir` 下必须有 `meta.json`（读 `diagonal`、`ground_parts`、`parts[].{file,mass,volume}`，`:108-111`、`:511`）与 `parts/part_*.ply`。`connectors` 接受三种形态：`{edge_id: theta}`（theta 同上 7 维，米）、`{u,v,direction,r,l}` 数组字典（`decode_action` 输出）、空 `{}`/`None`（= 不装）（`:366-414`）。`r`、`l` 会被**夹到 `EdgeInfo.r_cap`/`l_max`**（`:397-398`）。
- **返回**：键与玩具 `env.evaluate` 同构——`capacity, required, shortfall, pair_loss, global_loss, stable, effort, reward`，另有 `disp[P]`、`edge_disp[E]`、`steps`、`wall_s`、`ncon`、`pos0`、`pos1`、`stability_score`（`:756-775`）。`disp = ‖相对位移‖/diagonal`，**先扣掉整体刚体平移**（`:731`）。`stable = float(disp.max() < TAU_P)`，`TAU_P = 0.059`（`env.py:31`）。单位：装配目录一律米/千克（`:18`）。
- **失败方式**：`backend != "mujoco"` 直接 `NotImplementedError`（`:668-671`）；几何缺失走 `synthesize_edges` 兜底而不是报错（`:516-517`）；数值发散（NaN/Inf）被替换成「远在天边」的 100× 对角线，从而在指标里表现为散架（`:697-701`）。
- **副作用**：**会改网格**——`load_assembly_parts` 对每件做 `(1-clearance)` 质心等比缩放并把原点搬到质心（`:120-122`）。碰撞几何用**凸包**并缓存（`:124-129`）。无文件 IO、无 GPU（MuJoCo 纯 CPU）。每次 `evaluate` 重建 MJCF 字符串与 `MjModel`，无缓存。
- **耗时**：真实装配 31–409 ms/次（随部件数/接触数增长），8 边装配一次完整标签 59 s（`research/sim-env-report.md:152-162`）。**注意**：设计文档 §2.4 的「rollout 19 ms」属于玩具后端 `mujoco_env.py`（3 体合成场景，`research/c6-mujoco-mini-rollout.md:148,154`），**不能**外推到真实装配。

### A4. `anticheat.py` —— 词典序门控（纯函数）

```python
stability_score(result: dict, tau_p: float = 0.059) -> float          # :128
task_reward(result: dict) -> float                                    # :140
check_hard_constraints(result: dict, parts_info: dict,
                       cfg: GateConfig = DEFAULT_CONFIG) -> dict      # :152-153
eval_reversed_gravity(env, connectors: dict, *, cfg=DEFAULT_CONFIG) -> dict   # :228-229
connector_contribution(env, connectors: dict, *, cfg=DEFAULT_CONFIG) -> dict  # :258-259
gated_reward(result: dict, parts_info: dict, *, reversed_gravity=None,
             contribution=None, cfg=DEFAULT_CONFIG) -> dict           # :293-296
audit_suspicion(samples: Sequence[dict], *, cfg=DEFAULT_CONFIG) -> dict       # :406
stratified_failure(result, parts_info, edges, ...)                    # :473
```

- **输入**：`result` 只要求含 `disp`（退化时用 `shortfall` 反推，`:109-115`）；`parts_info` 键 `n_parts_expected / n_parts_actual / ivr / cd_delta / size_ratios`（`:160-165`）。`env` 只需满足 `evaluate(connectors, *, use_connectors, gravity_flip, include_ground) -> dict` 的鸭子类型（`:31-40`）。
- **返回**：`gated_reward` 给 `{reward, gate ∈ {1,2,3,4}, gate_name, passed, hard, reversed_gravity, contribution, task_reward, diagnostics, detail}`（`:324-386`）。门 1 罚 `-10·(1+magnitude)`、门 3 罚 `-8·(1+shortfall)`（`GateConfig.c_big/c_med`，`:72-73`）。`eval_reversed_gravity` 与 `connector_contribution` 各自**真跑 rollout**（后者跑两次，`:268-269`）。
- **失败方式**：不抛异常、不用 reason 字段，用 `gate` 整数 + `passed` 布尔 + `detail`。
- **副作用**：无。`GateConfig` 是 `frozen dataclass`（`:53-54`），改造阈值只能新建实例。
- **耗时**：纯计算 + 2×`env.evaluate`。

---

## B. 12 条 reason 的精确语义

先纠一处：**顶层 `reason` 只有 7 个值**（`unreatable_too_small` / `bool_failed` / `open_after_bool` / `peg_detached` / `socket_breaks_wall` / `degenerate_geometry` / `host_geometry_unreliable`，`tests/asm_bench.py:33-35` 与 `_fail` 调用点互证）。其余 5 条藏在 `detail` 或 `degen_code` 里。**适配器必须解析 `detail` 前缀，否则会把 149 条「锚点 gap」错记成「几何退化」**（`connector-boolean-report.md:421-427`）。

| 词 | 判定位置 | 数值判据 | 族 |
|---|---|---|---|
| `unreatable_too_small` | `connector.py:707`（早退，edge 已有 `r_inplane_max`/`depth_max` 时）、`:749`（射线量完补判）、`:763`（局部壁厚夹后） | `r_cap < r_min` 或 `d_cap < d_min`；`r_min = r_min_mm/unit_mm`（默认 0.25 mm），`d_min = 1.8·r_min = 0.45 mm`（`:86, 425`）；或 `r_wall_max = t_b/2.25 < r_min`（`:759-761`） | assembly（`feasible=False`，**不是失败**，`test_connector.py:418`） |
| `degenerate_thin_wall` | `:774-775` | `t_b < (1 + BOTTOM_GAP_R + MIN_WALL_R)·r = 2.25·r` | geometry（实为壁厚约束） |
| `degenerate_tangent` | `:780-784`，量测在 `:110-118` | 锚点处宿主面法向与轴向 `|cos| < TANGENT_COS = 0.087`（≈85°），A 或 B 任一侧 | geometry |
| `degenerate_at_patch_edge` | `:776-778` | `|u| > 0.85·half_w` 或 `|v| > 0.85·half_h`（`PATCH_EDGE_FRAC=0.85`，`:107`） | geometry（实测 0 条，报告 `§14`） |
| `degenerate_other` | 兜底（`:773`），把底层原因塞进括号内 detail | 见下 | geometry |
| `bool_failed` | `:621`（无 manifold3d）、`:891/894`（并集异常/状态）、`:899/902`（差集异常/状态）、`:655`（任何未预期异常） | `"NoError" not in status` 或输出 0 面 | geometry（引擎层；建议单独标 `tool_error`） |
| `open_after_bool` | `:909-910`（布尔后）、`:950-952`（STL 往返后） | `bnd1 != 0`（边界边数，**不是** `is_watertight`） | manufacturing（打印链闭合性） |
| `peg_detached` | `:844-847`（`peg_base_outside_a`）、`:918-924`（tag=="a" 的 `fragmented_a`）、`:930-933`（`peg_swallowed`） | 基点不在 A 的材质区间；或连块数变多且主块份额 `< 0.999` / 主块份额掉 >5 个百分点；或 `dv_a < 0.02·(πr²·啮合长度)` | manufacturing |
| `socket_breaks_wall` | `:855-858`（仅 `clamp=False`）、`:918-924`（tag=="b"）、`:937-940`（`socket_no_effect`）、`:941-944`（`socket_overcut`） | `wall_remain < 0.25·r`；或 `|dv_b| < 0.02·理论挖除体积`；或 `|dv_b| > 3.0×理论值` | manufacturing |
| `no_manifold3d` | **不是 reason**：`_fail("bool_failed","no_manifold3d")`（`:621`） | `import manifold3d` 失败 | geometry（环境缺陷） |
| `host_geometry_unreliable` | `:689-699` | `host_quality[k].decimate_ok is False` 或 `decimate_vol_err_pct > 2.0`（`DECIMATE_VOL_ERR_PCT`，`:97, 132`） | geometry（数据可信度） |
| `gap_too_wide` | **不是 reason**：`degenerate_other(gap_too_wide_for_wall(...))`（`:815-819`） | `eng_max = min(d_cap, t_b+gap-1.25r) <= need = gap+r`；`gap = b_near - a_front` | assembly |
| `no_material_hit` | **不是 reason**：`degenerate_other(no_material_hit(...))`（`:515, :728`） | `_nearest` 返回 `None`（该射线上没有任何有效材质区间），或 `line_intervals` 奇偶退化（`ray_parity_odd`） | geometry |

`degenerate_other` 的底层原因（detail 括号内）：`ray_parity_odd`、`no_material_hit`、`peg_in_overlap_zone`（`:795` 穿透吃光 `penetration >= t_b`；`:878` 占用率 `peg∩A/peg >= 0.95`）、`socket_region_empty`（`socket∩B/socket <= 0.05`，`:882`）、`wall_too_thin`（`:809`）、`gap_too_wide_for_wall`（`:815`）、`peg_tip_inside_a`（`:848`）、`socket_cap_not_outside_b`（`:851`）、`bad_theta`/`bad_edge`/`unknown_mode`/`degenerate_peg`（`:670/552/642/822`）。实测分布：204 条可行边失败里 `degenerate_other` 182（89.2%），其中 133 条是 `gap_too_wide_for_wall`（`connector-boolean-report.md:399-419`）。

---

## C. 适配器映射表

统一签名：`check(case, assembly) -> list[FailureCertificate]`。`locus` 至少带 `parts:[a,b]` 与 `joint: "{a}-{b}"`；`magnitude` 用 `{value, unit:"mm", threshold, direction}`。

| 现有函数 | 适配器方法 | 用什么参数 | 输出 `code` | 前置条件 |
|---|---|---|---|---|
| `connector.feasibility`（`:414`） | `ConnectorFeasible.fast_check` | `edge`、`unit_mm=1000`、`r_min_mm=0.25`、`depth_min_mm=0.45` | `unreatable_too_small` | edge 必须含 `r_inplane_max`（否则只能退回壁厚口径，会漏判面内空间）；毫秒级，先跑 |
| `connector.build`（`:575`） | `ConnectorFeasible.materialize` | 同上 + `theta`（由 policy 给 `u,v,dir`，`r=min(r_cap,0.5·d_cap)`，`d=2r` 的 aspect 口径见 `tests/connector_bench.py:115-129`）、`clamp=True`、`verify_stl=True`、`overlap_check=True`、`host_quality` | 另 6 条 + 4 个 degen 子码；解析 `detail`/`degen_code` | 几何 = **`assemblies_pre_decimate`**（测厚同源），`unit_mm=1000`，`edge` 已把 `anchor_mm/center_mm` 除以 1000 |
| `contact_graph.build`（`:1235`） | `ContactGraph.extract` | `parts`（`assemblies_pre_decimate/parts`）、`diagonal=meta["diagonal"]`、`ground_parts=meta["ground_parts"]`、`fine_parts=None` | 无 code（前置产物）；`thickness_failed=True` → `no_material_hit` / `wall_too_thin` | 只能用**裁剪后**件；`eps` 只影响 `delta` 与 `coincident`，不移动几何 |
| `contact_graph.build` 的 `edges` | `Connectivity.check` | 自算：`feasible_edges`（`edges_v2.json` 或 `feasibility.json` 的 `feasible_edges`）上的连通分量数 | `disconnected` | **必须自己算**：四模块都没有生成树判据（`feasibility.json` 是离线产物） |
| `assembly_env.AssemblyEnv.evaluate`（`:482/:705`） | `Stability.rollout` | `AssemblyEnv(dir, edges_json, edge_ids=feasible_edges, peg_stiffness_scale=(s/0.15)², edge_scale=s/0.15)`；`connectors` 来自 `decode_action` | `load_exceeded`（`disp.max() ≥ 0.059` 且无连接件也稳）、`disconnected`（可行边子图不连通） | 边表须是 `contact_graph.build` 原始格式（米 + `center/n_ab/r_max/l_max`）；**`edges_v2.json` 不能直接喂** |
| `anticheat.eval_reversed_gravity`（`:228`） | `Stability.reversed` | `env`、`connectors` | 诊断量（默认不作硬门，`GateConfig.reversed_gate="diagnostic"`，`:62`） | 无 |
| `anticheat.connector_contribution`（`:258`） | `AntiCheat.contribution` | 同上 | `connector_unnecessary`（`delta < 0.10` 或 `score_without > 0.50`） | 两次 rollout |
| `anticheat.check_hard_constraints`（`:152`） | `AntiCheat.hard` | `result`、`parts_info` | `interpenetration`（IVR）、`volume_exceeded`（尺寸失真） | **`ivr` 必须 harness 自己算**，见下 |
| `anticheat.gated_reward`（`:293`） | `AntiCheat.judge` | 上述三项结果 | 映射 `gate=1/3/4` 到 verdict | 无 |
| `refine_edges.refine`（`:117`） | 暂不封装 | — | — | 它会重跑一遍 `cg.build`（`:160`），套在 `ContactGraph.extract` 外层等于双倍耗时；只需要它的换算公式（`:162-203`）就自己算 |

### 现有代码给不出来、必须 harness 自己补的信息

1. **穿插体积比 IVR**。`AssemblyEnv.parts_info()` 硬编码 `"ivr": 0.0`（`assembly_env.py:784`，注释「裁剪后按构造零重叠」），`check_hard_constraints` 因此对真实数据**永不触发**。§4.8「假稳定 #1」的对策在现有实现里是空的。可复用的只有私有的 `connector._inter_volume`（`:178`），需要显式包装。
2. **`is_valid()` 硬门禁**。MuJoCo 路径没有 `is_valid`（`assembly_env.py` 全文无此符号）；`backend="uipc"` 直接 `NotImplementedError`（`:668`）。真正的 `world.is_valid()` 只在 `research/uipc_asm_validity.py:108`（`world.init` 之后、不 `advance`）出现，且要 CUDA。所以「invalid_sample 独立计数桶」这一层在 MuJoCo 上无法实现，必须由 harness 决定记 `precondition_failed`。
3. **插入路径 / 可达性**。四模块都没有扫掠、方向可达或 `insertion_blocked` 判据（`grep insert|sweep|path_check|collision_check` 在四个文件里只命中注释）。`theta` 里的方向只是杆轴，不是装配路径。
4. **保持力 / 承载**。只有解析代理量：`capacity = KAPPA·quality`、`required = LOAD_SAFETY·load·total_mass`（`env.py:43-45`、`assembly_env.py:748-750`），`quality = r_norm·l_norm·align·posq`（`:559-562`）、焊点刚度 `PEG_K_UNIT·q`（`:42, 620`）。没有牛顿量、没有力报表。
5. **失败 locus**。`connector.build` 从不返回 `a`/`b`（`_fail`，`:969`），部分边的 `locus` 只能靠适配器自己持有 `edge→(a,b)` 映射。
6. **装配级词表**。`disconnected` / `overhang` / `volume_exceeded` / `torsion_free` 在四模块中**均不存在**；`constraints.json` 的 `printer_volume: 250`、`max_insertion: 0.4` 无人读取（`data/asm_io.py:53-60` 只做格式校验）。
7. **设计意图**。没有 `instruction.json`。意图来自 `constraints.json["removable"]` 与代码里的 `INSTRUCTIONS = {0:("permanent",0.85), 1:("removable",0.35)}`（`env.py:33-38`），`AssemblyEnv(instr=1)` 默认「可拆」——这个默认值必须由 harness 显式从 `constraints.json` 推导，否则每个装配都被当成可拆。

---

## D. 自检 fixture 清单（§4.8）

样本都取自 100 装配，理由给的是**已落盘的实测证据**，不是我猜的。

| 验证器 | known_good | known_bad | 为什么选它 |
|---|---|---|---|
| `ConnectorFeasible` | `asm_000012` edge(0,1) | `asm_000089` 三条边 | `asm_000012`：2 部件/1 边，`fitness 0.998`、`interface_type=coincident`、`measurement_lod=fine`、`r_eff 0.644 mm`、`limiter=in_plane`（`research/contact-graph-anchor/edges_v2.json`）——秒级、单边、判据干净，最适合做「必须判正」。`asm_000089`：3 部件/3 边全部 `t_b=0`、`thickness_failed`、`r_eff=0`（同文件），`feasibility.json` 记 `n_feasible=0`——必然判负，且失败链路确定（`unreatable_too_small`） |
| `ContactGraph` | `asm_000012`（1 边 == `meta.n_contact_pairs=1`） | `asm_000089`（全边 `t_b=0`）；`asm_000068`（`rejections{zero:3,no_hit:34,grazing:1}`） | known_good 的边数能与 `meta.json` 的 `n_contact_pairs` 对齐，是最好的自动断言；`asm_000012` 那条边还有 `n_patches=33 / patch_area_share=0.136`，顺带覆盖「多块接触取最大块」分支（`contact_graph.py:1067-1075`）。known_bad 选 `t_b=0` 与测厚拒绝计数两类 |
| `Stability` | `asm_000000`（8 部件/11 边：无连接件必塌、满配必稳、门 4 通过，`tests/test_assembly_env.py:120-172` 已固化） | `asm_000062`@150 mm（弱连接件：`stable=1.0` 但贡献增量 **0.022**）；`asm_000034/064/066`（可行边子图不连通） | 假稳定活样本与「物理上装不成」两类失败各一。`asm_000062` 的实测值有三处独立记录：`sim-env-report.md:89-118`、`test_assembly_env.py:186-216`、`test_anticheat.py:352-387`。**注意** `test_assembly_env.py` 用 `pytest.mark.skipif` 挂在 `tests/_edges_assemblies.json` 上，该缓存当前**不存在**，直接跑会整组 skip |
| `AntiCheat` | `asm_000000` 满配 → `gate=4` | `asm_000062` 弱配 → `gate=3`、`reward=-8.627`；`asm_000021` 不装连接件就 `score_without=1.0`（装饰品） | 门 1 目前无法用真实数据造出负样本（IVR 恒 0），门 3 是唯一有真实正负样本的门。`asm_000021` 是 `sim-env-report.md:205` 的模式 A 代表 |

**首批 fixture 的载体**：由于没有 `contact_graph.json`，每个 fixture 都要么现跑一次 `contact_graph.build`（实测 mean 19 s/装配），要么把 `_edges_assemblies.json` 这类缓存重新生成一次（`tests/asm_bench.py:74-97`；100 装配约 10 分钟建边 + 15 分钟布尔，`connector-boolean-report.md:472`）。建议 fixture 直接**冻存**成小 JSON（per-assembly edges + 部件 sha256），否则自检本身就要半分钟起。

---

## E. 陷阱清单

1. **mm vs m 的三套口径并存**。① 权威口径：网格是**米**，`×1000` 得 mm（`research/contact-graph-batch100.py:171-175`、`refine_edges.py:165-189`）；实测 `part_000.ply` 体积 == `meta.json` 体积，最长边 0.225。② `connector.infer_unit_mm` 用 `TARGET_LONG_SIDE_MM=225 / longest_side`（`connector.py:87, 448-450`）= 1000，自洽；但它的 docstring 写「归一到 150 mm」（`:449`），**注释过时**。③ `tests/asm_bench.py:67-70` 读 `meta["scale"]×1000 ≈ 118 mm/单位`——`meta["scale"]`（0.118）其实是「米 / 原始 Hy3D 源单位」：`0.225 / 0.11795 = 1.9076`（asm_000012）、`0.225 / 0.11440 = 1.9667`（asm_000062），即原始物体最长边约 1.91–1.97 个源单位，与 `tests/test_connector.py:21` 的 `UNIT_MM = 150.0/1.9678` 同源。它**不是**「米/存储单位」。照抄这个分支会让 `r_min` 放大 8.5 倍、把可行边判成不可行。`connector_bench.py:70-73` 的写法是对的。
2. **两套尺度混用**。`assemblies/`（225 mm，diag 0.326 m）与 `assemblies_pre_decimate/`（150 mm，diag 0.217 m）单位同为米但**尺度差 1.5×**。布尔与测厚必须都用 `pre_decimate`（`refine_edges.py:107-111` 的 `boolean_geometry_rule` 明文），把 225 mm 的件塞进 150 mm 的边参数里会得到不一致的 `t_b`（decimation 会把 seam 塌掉，`t_b` 最多差 4.5×）。
3. **`edges_v2.json` 不能直接喂 `AssemblyEnv`**。它是毫米制 + `center_mm`/`anchor_mm` 键（header 明文：「divide by 1000 to get metres」），而 `edges_from_records` 只读 `center`（`assembly_env.py:312`），缺省 `[0,0,0]`——所有 peg 会装到原点，仿真仍然「跑得通」但不报错。同理 `r_max`/`l_max` 缺省成 `0.4·thick`/`max(thick, 0.005)`（`:319-320`）。
4. **`host_geometry_unreliable` 在当前数据上永不触发**。`host_unreliable()` 读 `decimate_ok` / `decimate_vol_err_pct`（`connector.py:121-132`），但**全仓所有 `meta.json` 都不含这两个键**（对全部 `meta.json` 做 `Select-String` 零命中；`data/asm_io.py:42-52` 的 `PART_REQUIRED` 也不含），`asm_io.part_record` 从不写它们。`research/_asm_rows.json` 里 14 条 `host_geometry_unreliable` 记录无法在今天的交付数据上复现。
5. **`is_valid()` 语义只属于 UIPC**。MuJoCo 无此概念；`backend="uipc"` 抛异常（`assembly_env.py:668-671`）。UIPC 版本里 `world.is_valid()` 在 `world.init` 之后、`advance` **之前**调用（`research/uipc_asm_validity.py:81,108`），`False` 意味着后续 `advance` 退化成静默空操作（`data/hy3d_screen.py:62` 的原话），这正是 §4.8「假稳定 #2」的机制，**不能**把它读成 `Δp≈0`。
6. **`is_watertight` 不可用**。这批网格有 0.01%–0.1% 非流形边，`is_watertight` 恒 False 但表面封闭；判定一律用 `bnd1 == 0`（`connector.py:22-28`、`data/asm_io.py:70-90`、`test_connector.py:92-97`）。`trimesh.boolean` 会因水密预检直接抛错，必须走 `manifold3d`（`:25-27`）。
7. **`t_b` 语义在两个产出方之间不一致**。`contact_graph._thickness` 取「从界面进去的**第一段材质**」（`:846-886`，含 `grazing<0.2`、`SLIVER_RATIO=1e-3`、`MIN_WALL=1e-5`、`too_thick>min(diagonal, extent)` 四道闸）。而报告实测落盘 `t_b` 更接近「首段起点到末段终点的**总跨度（含空腔）**」（`connector-boolean-report.md:677-689`，两条边差 0.5%），并明确指出「含空腔的厚度对盲孔深度约束是错的」。两者谁是今天 `edges_v2.json` 的真实语义，**未核实**；适配器应把 `t_b` 来源写进 `evidence_ref`。
8. **`_nearest` 的表层擦过残片**。射线起点落在表面时会切出 1e-4–1e-3 mm 的伪区间，曾把 64 条边误判成 `unreatable`（低估 10³–10⁶ 倍）。修法是丢掉「薄于同射线最厚区间 ×1e-3」的区间（`connector.py:274-297`）。**仍未清干净**：这个修法依赖「同一条射线上存在更厚区间」，当整条射线都只擦过表面时会退回原始列表（`:292`）。
9. **`clearance` 改了网格**。`AssemblyEnv` 加载时对每件等比缩小 0.3% 并把原点搬到质心（`assembly_env.py:120-122`），碰撞几何换成凸包（`:124-129`）。`PartInfo.mesh` **不能**拿去跑布尔或测厚——那是仿真专用几何。
10. **cwd / sys.path / 路径**。四个模块都以 `assembly_rl.*` 包名导入，调用方必须把 `D:\yanyi\rl_assembly\code` 放进 `sys.path`（`tests/label_quality_bench.py:28-31`、`test_assembly_env.py:18-19`）。`data/` 是 `E:\` 的 junction，`refine_edges.header()` 会把 `E:\rl_assembly-data\...` 这类绝对路径写进产物（`:107`），跨机复现会指向不存在的盘。
11. **`parts_fine/` 不存在**。交付语料 100 个装配中 **0 个**有 `parts_fine/`（实测 `Get-ChildItem -Directory | Where Test-Path parts_fine` = 0）。`AssemblyEnv(use_fine=True)` 会**静默回退**到 `parts/`（`:113-114`），`refine_edges._load_parts` 的警告（`:63-68`）已过时。真·未简化几何只有 `assemblies_pre_decimate/parts/`。
12. **`parts_info()`/`summary()` 的常量**。`ivr=0.0`、`size_ratios=[1.0]*n`、`cd_delta=0.0`（`:780-786`）——门 1 的四个子判据里三个是常量，只有 `n_parts` 是真的。
13. **`EdgeInfo.r_cap` 与连接件判据不同源**。env 侧 `r_cap = 0.4·min(t_a,t_b)`（`:88-90`，`test_assembly_env.py:66` 把它固化了），连接件侧是 `min(r_inplane_max, 0.5·min(half_w,half_h))` 与 `t_b/2.25`（`connector.py:373-391, 759`）。env 的夹取**不包含面内空间约束**，因此 `normalize_connectors` 可以放行连接件侧认为不可行的 `r`。门应该用 `feasibility.json` 的 `feasible_edges`（`label_quality_bench.py:49-52` 就是这么做的）。
14. **重力方向是字面翻转**。`gravity = "0 0 9.81" if gravity_flip else "0 0 -9.81"`（`:640`），且反重力测试同时 `include_ground=False`（`:235`）；`evaluate` 在算 `disp` 前扣掉整体刚体平移（`:731`），所以**绝对位移从来不是指标**。
15. **死代码/过时名字**：`MAX_EXTEND_STEPS`（`connector.py:74`）与 `_contains`（`:962`）零引用；`effective_radius_cap`/`feasible_radius` 是别名（`:444-445`）；`EdgeInfo.r_cap` 的注释「与连接件口径一致」（`:89`）与实际不一致（见 13）。
16. **`degenerate_geometry` 名不副实**：204 条失败里 149 条（73%）根因是锚点 gap 而不是几何退化（`connector-boolean-report.md:425-427`）。按字面把它归入 `family=geometry` 会把「界面没贴合」这一最大单因（`gap_too_wide` 166 条 + `no_material_hit` 72 条 = 33.8%，同文件 `:632-633`）错记成几何问题。

---

### 待拍板（会改适配器形态）

1. **`unit_mm` 用 1000 还是 `meta["scale"]×1000`**：本文按 1000（有体积/边长实测与 `contact-graph-batch100.py` 的换算支撑）。但 `asm_bench` 的分支会算出 ~118，而它的历史结论「705 条可行边 / 55% 成功率」（`connector-boolean-report.md:645-653`）用 118 是算不出来的——所以那次运行实际取的是哪个 `unit_mm`，**未核实**。两者不能混用，必须先在 harness 里定一个。
2. **`t_b` 要不要按「首段材质」重算**：报告建议接触图改语义并另设 `t_span`（`:686-689`），改它等于改算法，按 §2.4 铁律需请示。
3. **`open_after_bool` / `peg_detached` / `socket_breaks_wall` 归 `manufacturing` 还是 `structural`**：dev doc §3.4 的 v0 词表里 manufacturing 只有 `overhang` / `volume_exceeded`，这三条是新增语义。
