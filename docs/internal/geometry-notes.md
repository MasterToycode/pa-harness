# 几何约定（v2 设计平面 · T1 物化层）

> 冻结依据：`docs/contracts/v2-interfaces.md` §2、`src/pah/design/spec.py`、`src/pah/design/api.py`
> 实现：`src/pah/design/connectors/**`、`src/pah/design/materialize.py`
> 读者：T2（验证器）、T3（算子/策略）、T4（运行时/CLI）、将来的 CAD/DiT 后端

本文只写「代码里看不出来的约定」和「为什么」。形状参数本身以 `spec.py` 的 pydantic 模型为准。

---

## 1. 坐标系、位姿、世界坐标

- 单位一律 **mm / 度**，无 unit 字段。
- **blank 以 part 局部原点为中心**（等同 `trimesh.creation.box`）。所以 `pose_mm` 指 blank 中心，
  `feature_xy_mm` 也从同一个中心量。没选「角点对齐」，是因为那样「body 顶面」会写成
  `pose.z + size.z/2` 而其余特征还得跟着换参照，一个约定两种算法就会漂。
- **位姿**：`T = Translate(pose_mm) @ R`，其中 `R = Rz(rz) @ Ry(ry) @ Rx(rx)`（extrinsic XYZ，
  绕局部原点），即 `spec.py` docstring 的口径。`ConnectorBuilderBase.build` 负责应用，
  **builder 只写局部坐标**，插件的 `build_local` 也一样。
- **所有交出去的 mesh 都是 world frame**。`MaterializedDesign.meshes` / `.paths` / `DesignView.mesh()`
  拿到的都是世界坐标，验证器不需要、也不应该再做变换。

### insert 的局部原点 = 公特征的引导端

| kind | insert 局部原点 | seated 关系 | `mate_axis`（局部） |
|---|---|---|---|
| `peg_hole` | 销底面圆心（先入孔的那端） | 销底面 = 孔底 = body 顶面 − `engage_mm` | +Z |
| `snap_fit` | 钩头底面（先入槽的那端） | 钩头顶面 = 倒扣腔顶（含 `play` 间隙） | +Z |
| `dovetail` | 榫头前侧面 ∩ 榫头顶面 | 榫头前侧面 = 槽底 + `clearance_mm`；榫头顶面 = body 顶面 | +X |

- `insert.pose_mm` 是 **seated（完全插入）姿态**，由 spec 作者写准；builder 不推导、不纠正位姿
  （否则 `shift_part` 这类算子就无法制造位移让验证器去测）。
- `ConnectorGeometry.mate_axis` 是上面局部轴经 **insert 自身旋转**后的世界单位向量；
  `approach_mm` 是沿 `+mate_axis` 从 seated 往外退的起点。插入路径验证器从 `+approach` 扫到 `0`。
- 这两个量**跟着设计走**：`materialize` 把 `ConnectorGeometry` 的值抄进
  `MaterializedDesign.mate_axis/.approach_mm`，`.view()` 再透传给 `DesignView.mate_axis/.approach_mm`；
  同一个值还写进 `assembly.constraints`（`mate_axis`/`approach_mm`），所以只拿到 `Assembly` 的
  消费者（`view_of`、CLI、resume）也能拿到真轴。**验证器读 `view.mate_axis`，不要硬编码 +Z、
  也不要从 kind 名重推**（dovetail 是 +X，硬编码 +Z 会把正确的滑入配合判成阻塞）。
  `builder_for(kind).build(spec)` 仍是唯一权威来源。
- **换 kind 必须重定位姿**，而只有 kind 知道该放哪：`builder.seated_pose_mm(spec) -> Vec3` 返回
  「完全插入时 insert 原点的世界坐标」。T3 的 `switch_connector`、T4 的 `pah design new` 调它。
  这个返回值必须与 fixture 里手写的 `insert.pose_mm` 一致——测试 `test_the_kind_knows_where_its_insert_seats`
  就是拿四个 fixture 互校这条。不这么做的话，peg_hole → dovetail 的切换会把 insert 留在
  `(0,0,-2)`（dovetail 应为 `(10,0,8)`），seated 交体积 **1800.01 mm³**，变成策略永远修不好的假干涉。

---

## 2. 三种 kind 的精确形状（局部坐标）

### peg_hole —— 盲孔 socket

- body：blank − 圆柱盲孔（半径 `d_hole/2`，**孔深 = `engage_mm`**，开口在顶面，孔底 = 顶面 − engage）。
- insert：**单根圆柱销**（半径 `d_peg/2`，高 `peg_length_mm`，默认 `engage + 6`）。没有手柄板。
- **insert 是纯凸体（`convex_hull.volume / volume = 1.000`）**。这是物理档能跑的前提：MuJoCo 只碰
  凸包，销 + 宽手柄板那样非凸的 insert 会被凸包抹掉台阶（实测 hull/volume = 1.681，物理验证器
  正确地拒绝了）。要么凸分解（贵、不是每种形状都可行），要么让销就是销——选后者，顺便去掉一个
  spec 里根本没有的数（`2·d_peg` 见方的手柄是物化器自己发明的，不是设计意图）。
- **`insert.size_mm` 在 peg_hole 下只是包络提示，不参与几何**：builder 只读 interface 参数和
  body 的 blank。这个字段仍会写进 `Assembly`，报表里显示的就是「这块零件名义多大」。
- **为什么盲孔而不是通孔**：无销头的销穿过通孔后没有任何东西托住它，物理沉降会整根掉出去，
  这不是设计缺陷、是模型缺了底面支撑，会让每个 peg_hole 案例都过不了物理档。盲孔是真实配合
  （销靠配合定位），且给 insertion path 一条直线。通孔 + 销头将来做成**新 kind**`peg_hole_through`，
  不加 flag。
- seated：销底面与孔底**精确共面**（不是多插 1e-6）。销露出顶面 `peg_length_mm − engage_mm`。

### snap_fit —— 悬臂钩 + 倒扣腔

- body：blank − 贯通方槽（`slot_w × slot_w`，沿 Z 贯穿）− 倒扣腔
  （边长 `slot_w + 2(hook_h + play)`，高 `hook_h + 2·play`，腔顶 = `body_top − engage + play`），
  `play = (slot_w − hook_w)/2` 就是参数模型报的那份 clearance。
- insert：钩头（`hook_w × hook_w × hook_h`，底面在局部 z=0）+ 悬臂梁
  （`beam_thk × hook_w × beam_len`，从钩头顶面往上）。
- **近似（写在 docstring 里，也写在这）**：insert 是**刚体**，画的是「已经弹进倒扣之后」的形状。
  真实 snap 的钩头自由状态比槽宽，靠弹性变形才能穿过；刚体画那个自由形状根本插不进去，路径
  验证器拒绝它是对的。所以模型保持 `hook_w ≤ slot_w`，把倒扣做深 `hook_h`，不模拟弹簧。
  **它证明**：两个实体可制造、seated 不干涉、直线插入路径通畅。**它不证明**：保持力/拔出力——
  那是物理档（刚度、变形）的事，设计平面任何验证器都不断言。
- seated：钩头在腔内四周留 `play`，**没有贴面接触**。

### dovetail —— 梯形滑榫

- body：blank − 梯形槽（沿 X 的盲槽，开口在 **body 的 +X 面**和顶面，`pocket_len = engage + clearance`）。
  截面（Y/Z）：开口半宽 `width/2`，到槽底每侧加 `depth·tan(angle)`；**上窄下宽**才是「拔不出来」。
- insert：同角梯形榫（每侧比槽小 `clearance_mm`，深 `depth − clearance`，长 `engage_mm`）
  + 手柄（`engage × max(3·width, 20) × max(0.8·depth, 6)`），榫头顶面与手柄底面共面。
- seated：榫头前侧面 = 槽底 + `clearance`；**手柄底面坐在 body 顶面上**（贴面接触，见 §3）；
  榫头三条边各留 `clearance`，无接触。

### 物理档的凸性口径

| kind | insert 是否凸 | 物理档 |
|---|---|---|
| `peg_hole` | 是（`hull/volume = 1.000`） | 正常评估沉降 |
| `snap_fit` | 否（钩头 + 悬臂梁） | `not_evaluated`（T5 记「不评估」，不报 `physics_error`） |
| `dovetail` | 否（梯形榫 + 手柄） | 同上 |

snap/dovetail 保留手柄是必要的（卡扣与滑入需要夹持段），它们由几何/干涉/路径验证器覆盖；
**物理档不评估 ≠ 通过**，报表里必须分开写，不能把 `not_evaluated` 折进成功率。

### 圆柱分辨率

peg / 孔用 **64 边形**内接圆柱：10 mm 直径上比公称面小 6 µm，比任何 `FitConstraints` 能表达的
配合带低一个数量级，所以「公称直径就是契约」。**固定边数是确定性要求**：交给库默认值，STL 字节
就会随 trimesh 版本变。

---

## 3. 贴面接触 = 水密零厚度薄片（T2 必读）

seated 姿态下允许**且必然**出现贴面接触：

| 位置 | 接触 |
|---|---|
| peg 底面 ↔ 盲孔孔底 | 面接触（圆盘） |
| dovetail 手柄底面 ↔ body 顶面 | 面接触（矩形） |
| snap_fit | 无（处处留 `play`） |

manifold 对这种相交返回的是**零厚度薄片**，而且它 `is_watertight=True`、`len(faces) > 0`。
因此：

- `interface_interference` **必须按体积阈值判**（contract §3：>1e-6 mm³ 算干涉），
  **不要**用 `len(faces) > 0` 或 `is_watertight` 当判据，否则每个正常 seated 的 peg_hole / dovetail
  都会误报 `geometric_interference`。
- trimesh 5.1 在那种薄片上取 `.volume` 会走 `mass_properties`、除零并抛
  `RuntimeWarning: invalid value encountered in divide`（薄片质心无定义）。不想要警告就自己用
  散度定理求和：`|Σ v0·(v1×v2)|/6`，实测值为 0。
- **不要**为了「消掉薄片」在几何里塞 spec 没有的缝：那个缝会变成第二个真值来源。

---

## 4. 确定性、幂等、hash

- `geometry_hash` 在 `pah.design.api`（**唯一实现**，T1 只 re-export）。它是
  `sha256(f"{spec.content_key()}|{part_id}|{kind}")[:32]`，**不是** STL 的哈希：
  - loop 的环检测/恢复要能只凭 `spec.json` 复现 hash，与网格是否重建、目录是否搬家无关；
  - 将来换 CAD/DiT 后端，同样数字必须同 hash，否则缓存永远 miss。
- 物化层不发散：无时钟、无随机、无无序集合迭代；`spec.json` 用 `sort_keys=True` + LF 写出；
  STL 导出字节可复现（测试直接比字节，不只比 hash——只比 hash 的测试在网格坏掉时也会通过）。
- **幂等**：同 spec 同 `out_dir` 二次调用覆盖写、hash/assembly id 不变、不报错（恢复/重试会重调）。
- 目录约定别改：`out_dir/spec.json`、`out_dir/geometry/<part_id>.stl`。`geom.path` 为空 = 尚未物化。

---

## 5. fixtures（`fixtures/design/*.json`）

| 文件 | kind | 关键数字 | 期望证据 |
|---|---|---|---|
| `demo-peg-hole-interfere.json` | peg_hole | d_peg 10.2 / d_hole 10.0 / engage 8 / approach 10 | seated 交 ≈ 25.34 mm³（>0.5）；clearance −0.1 出配合带 |
| `demo-peg-hole-clear.json` | peg_hole | 9.7 / 10.0 / engage 8 | seated 交 = 0；clearance 0.15 在默认 slip 带 [0.10, 0.60] 内 |
| `demo-snap-fit.json` | snap_fit | hook 4×1、梁 12×1.6、槽 4.2、engage 8 | body 减料 = 槽 + 倒扣腔；seated 交 = 0 |
| `demo-dovetail.json` | dovetail | width 10 / depth 6 / 12° / engage 20 / approach 22 | body 减料 = 梯形槽体积；seated 交 = 0；手柄在顶面上方 |

`approach_mm` 的取值让路径从**完全退出**开始扫：peg 10 ≥ engage 8、snap 12 ≥ engage+hook、dovetail 22 ≥ engage+clearance。

fixtures 一律不设 `FitConstraints.print_orientation_deg`（默认 `None`）：不评悬垂、只评壁厚。
印床朝向是工艺决策，spec 没给就不要在 fixture 里编一个，否则测的是编出来的朝向而不是几何。
