# 多件装配（X-Part 式）——从「一个整体」到「一组能拼起来、能打印的零件」

> 这是 v2.1 的主线：设计平面从「两件 + 一个接口」扩到「N 件 + M 接口」。
> 本文记录**这条路怎么走通、边界在哪、哪些数字是实测的**。契约细节在
> `docs/contracts/v2-interfaces.md`，物理与几何的坑在 `docs/internal/`。

---

## 1. 一条命令

```bash
# 免费，本地切分：把一个整体等分成 N 块，自动造销孔，逐对验证，导出 3MF
pah decompose runs/acceptance/c1-episode/geometry/body.stl --parts 3 --out runs/decompose-3

# 花钱，走腾讯云「组件生成」（X-Part + P3-SAM 的云版本）
pah decompose whole.glb --decomposer xpart --size-mm 120 --out runs/xpart-1

# 免费，重放一单已经付过费的分解结果
pah decompose whole.glb --decomposer dir:D:\Temp\pah-xpart-probe --size-mm 120 --max-interfaces 4
```

产物：`runs/<id>/{spec.json, parts/, geometry/, print/{*.stl, assembly.3mf, print-manifest.json}, report.json}`。

---

## 2. 管线

```
整体网格
  ├─(A) Decomposer 端口 ──► N 个水密实体
  │      ├ local : 沿最长轴等分（免费、可控、切面精确互补）
  │      ├ xpart : 腾讯云 SubmitHunyuan3DPartJob（真实生成，见 §4）
  │      └ dir   : 重放已下载的分解目录（免费）
  ├─(B) pairing ──► 相邻判定 + 每个接触面一个 peg_hole
  ├─(C) DesignSpec(N 件, M 接口) ──► materialize：逐接口在网格上挖孔/长销
  ├─(D) 逐对验证 ──► 每个接口一套「两件影子」，复用原有 6 个验证器
  └─(E) 导出 ──► 每件 STL + 一个多对象 3MF + 打印清单
```

**为什么验证器不用重写**：6 个验证器都是「两个实体沿一个轴」的物理判断。
`design/pairs.py` 为每个接口构造一个只含这两件的 `DesignSpec`（角色改写成
`body`/`insert`）和一个只含这两块网格的 `DesignView`，并带上**这个接口自己的**
`mate_axis`。验证器看不出区别，于是零改动复用。这是这一轮最省事的决定，
也是最容易做错的地方——一个全局轴会让所有接口按同一方向扫，指向别处的接口
全被判「插入路径受阻」。

---

## 3. 分工边界：谁负责什么

| 环节 | 谁做 | 为什么 |
|---|---|---|
| 语义分解（这是什么零件） | X-Part | 生成模型的事 |
| 零件完整实体 | X-Part | 同上 |
| **配合面尺寸**（间隙、孔径、销长） | **PAH** | 生成模型不保证，也不该保证 |
| **可打印性**（壁厚、悬垂、水密） | **PAH** | 需要测量，不是生成 |
| **能否拼起来**（干涉、插入路径） | **PAH** | 同上 |
| 修复决策 | agent（LLM 只出结构化动作） | 项目主线 |

X-Part 的原论文自己也写：部件之间**只贴合、不固接**（parts are only touching,
not fastened）。所以「打印出来能拼」这件事，生成模型只做了一半。

---

## 4. 云 API 实测（2026-11-01，JobId 1508519036782002176）

输入：X-Part 官方样例 `000.glb`（50 万面，非水密，extents 1.05×1.04×0.25）→
Blender 转 FBX（15.4 MB）→ COS 预签名 URL → `SubmitHunyuan3DPartJob`。

| 事实 | 实测值 | 对管线的影响 |
|---|---|---|
| 提交到完成 | 1 次调用，几分钟 | 要轮询 `QueryHunyuan3DPartJob` |
| 输出文件数 | **19 个，每件一个 GLB** | 不是「一个多对象 OBJ」，件身份=文件 |
| 文件内命名 | 每个 GLB 一个 geometry，名 `part_N` | 用内层名字做 `PartSpec.id` |
| 单位 | **归一化到 ~1.0**，不是 mm | 必须按 `--size-mm` 缩放 |
| 水密性 | **多数件不水密**；补洞后部分仍不水密 | 必须修：补洞 → 体素重建兜底 |
| 件间关系 | **30 对 bbox 重叠 >1 mm** | 不是干净的体积划分，装配会干涉 |
| 分割数据 | 单独一份 face label（515,145 项，与提交面数不完全对应） | 用于编辑/合并，暂未接 |

因此 `examples/xpart_decomposer.py` 做四件事：修（三级，逐级记录）、缩（按目标 mm）、
命名（用内层 `part_N`）、**记录丢掉了哪些件**（不静默跳过）。

### 4.1 把真实结果放进管线跑一遍（2026-11-02，实测）

```bash
pah decompose 000.glb --decomposer dir:D:\Temp\pah-xpart-probe --size-mm 120 --max-interfaces 2
```

| 阶段 | 结果 |
|---|---|
| 载入 19 个 GLB | 19 件全部修复成水密实体；`scale_mm = 113.97`（union 1.053 → 120 mm） |
| 自动配对 | 找到多对相邻；接口上限 2，其余带原因跳过（「beyond the interface limit」） |
| 组装成 spec | 4 件 3 接口中取最大连通组 → **2 件 1~2 接口**，15 件被列为 `unconnected_parts` |
| 逐对验证（454 s） | **assemblable 否**：`geometric_interference` + `insertion_path_blocked`（两个接口都有）<br>**printable 否**：`below_min_wall` |
| 导出 | 3MF + 逐件 STL 照常写出（可检查，但报告已判否） |

**这个「否」是正确结论，不是 bug**：X-Part 的件互相嵌入、壁厚小于 1.2 mm，
打印出来既塞不进也立不住。这一轮交付的是**测出这件事的能力**；把它修好（分区去重叠、
补厚壁）是下一阶段 agent 修复循环的工作。

代价参考：`--no-verify` 全流程 11~14 s；带完整验证（干涉 + 插入路径 + 壁厚 + 间隙）
在 4 万面/件的量级上约 454 s —— 插入路径一个接口就要做 14 次布尔。

---

## 5. 验证状态：哪些是「证明」，哪些是「没测」

`report.json` 的 `verdicts` 只有两个字段，但每个都带 `validated`：

| 检查 | 两件参数化件 | 分解件（本轮） |
|---|---|---|
| `fit_clearance` | ✅ | ✅ 逐接口 |
| `interface_interference` | ✅ | ✅ 逐接口 |
| `insertion_path` | ✅ | ✅ 逐接口 |
| `printability`（壁厚） | ✅ | ✅ 逐接口 |
| `printability`（悬垂） | ✅（声明了打印朝向时） | ⚠️ **未评估**：N 件的每块有自己的摆放姿态，一个全局朝向不是它 |
| `insert_material` | ✅ | ⚠️ **不适用**：它拿生成件和参数化参考比体积，分解件没有那个参考 |
| `physics_settle` | ✅ | ⚠️ **不适用**：它的模型是「销在孔里」，分解件是「板叠板」 |

所以分解件的结论是 `assemblable 是 (partial)` / `printable 是 (partial)`，
`not_evaluated` 里列着没跑的那几项。**这个词是交付物的一部分**：它说清楚了
哪几句话有证据。

下一步要做的是**接合保持性（joint retention）**的物理检查：把配合件沿接口轴
放开，看它会不会掉——那才是「这块板能不能挂在那块板上」的正确模型。已记在
`docs/architecture.md` §7 的路线里。

---

## 6. 已知限制（非遗漏）

1. **本地切分器不是语义分解**：它沿最长轴等分，齿轮箱出来是切片，不是「轴 + 壳」。
   `provenance.note` 里就是这么写的。
2. **相邻判定是 bbox 事实**：包着别人的 C 形件判不出来。文档写在
   `design/pairing.py` 的模块头里，`contact_of` 返回它用到的每个数字。
3. **重叠对（X-Part 常见）会被配上但被判干涉**：这是正确结论（它们确实占了同一块
   空间），不是 bug；要修得先做「分区修复」（按顺序把重叠判给先到者）。
4. **接口数量默认上限 12**：19 件的分解提出 30 个接触面，每个接触面要两刀布尔
   加一整套验证，在 30 万面的网格上要好几分钟。上限是显式参数，被跳过的对有
   原因字符串。
5. **`pah decompose --decomposer xpart` 会花钱**：一单提交即计费，取消不退款。

---

## 7. 这一轮的四个真 bug（都已被测试钉住）

| bug | 现象 | 修法 |
|---|---|---|
| 角色假设 | 中间件既当销又当孔，两件契约的「a 必须是 body」直接拒绝 | 放开 a/b 方向，方向是**连接的属性**不是零件的属性 |
| 逐对 note 的 key | 三件跑出 `validated: full`，而 physics/printability 其实没跑 | `_did_not_run` 同时认 `<接口>/<验证器>` 和裸验证器名 |
| 网格哈希 | STL 写出再读回**不保证顶点顺序**，同一实体两个哈希，空白件库看起来被篡改 | 规范序顶点（1e-6 mm 取整）+ 面数 + 体积 |
| 最小销径 | 12 mm 板三等分后每块 4 mm，60% 只有 2.4 mm，被 3.0 mm 下限**静默拒绝连接** | 下限降到 2.0 mm；真正的护栏是壁厚预算，不是这个常数 |
