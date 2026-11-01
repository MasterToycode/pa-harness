# 首次接入 LLM 与 DiT（约 30 分钟）

> 目标：**不改任何内核代码**，把你自己的 LLM API 和 DiT 模型接进 PAH，跑通第一次「需求 → 证据 → 动作 → 再验证」。
> 依据：`docs/contracts/v2-interfaces.md` §6（插件面）、`docs/plugin-api.md`（新增模块 checklist）。
> 全程可自检：`pah providers doctor`。

---

## 0. 两条接入路线（先看这张表，决定你属于哪种）

| 你要接的东西 | 它在闭环里的角色 | 接入方式 | 代码位置 |
|---|---|---|---|
| **LLM API** | **决策**（Planner/Diagnoser/Controller）：选连接件类型、改哪个参数、改多少 | 配置即可（`pah.toml` 的 `[llm]`） + 内置 `--policy llm` | 不用写代码 |
| **DiT / CAD 内核** | **生成**（把 `DesignSpec` 变成两个实体） | 写一个 `materialize(spec, out_dir)` 函数，在 `[gen]` 里按 `python` 注册 | 复制 `examples/dit_materializer.py` |

关键边界（也是论文口径）：**LLM 的输出里没有几何字段**，它只输出结构化动作（算子 + 目标 + 参数）；
几何永远由生成层执行。所以 LLM 和 DiT 是两个独立的口，可以分别换、分别测。

---

## 1. 接入 LLM API（5 分钟）

### 1.1 放 key（写一次 `.env`，不用每个 shell 导出）

```powershell
cd D:\yanyi\pa-harness
copy .env.example .env        # 若还没有 .env
notepad .env                  # 填一行：DEEPSEEK_API_KEY=sk-...
```

`.env` 已被 `.gitignore` 忽略；`pah` 每次启动会读当前目录的 `.env`，并把**变量名**打到 stderr
（不打印值），例如 `pah: loaded 1 variable(s) from .env: DEEPSEEK_API_KEY`。

优先级：**已经存在的环境变量赢**。所以临时覆盖仍然有效：

```powershell
$env:DEEPSEEK_API_KEY = "sk-..."     # 只对当前 shell 生效，覆盖 .env
```

也可以用 `$PAH_ENV_FILE` 指向别的文件（CI / 共享 checkout）。
**不要**把 key 写进 `pah.toml`——那里只认变量名，写值不生效。

### 1.2 写 `pah.toml`（已在 `.gitignore` 里，不会被提交）

```toml
[llm]
default = "deepseek"

[llm.providers.deepseek]
kind        = "openai-compatible"
base_url    = "https://api.deepseek.com/v1"
api_key_env = "DEEPSEEK_API_KEY"     # 只写变量名
model       = "deepseek-chat"
billing     = "per_token"
pricing     = { in_usd_per_mtok = 0.28, out_usd_per_mtok = 1.12 }   # 会计口径
```

换厂商只改这一段（`kind` 都是 `openai-compatible`）：

| 厂商 | base_url 示例 | model 示例 |
|---|---|---|
| DeepSeek | `https://api.deepseek.com/v1` | `deepseek-chat` |
| 火山方舟 | `https://ark.cn-beijing.volces.com/api/v3` | 你的 endpoint id |
| 千问 / DashScope | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-max` |
| OpenRouter | `https://openrouter.ai/api/v1` | `deepseek/deepseek-chat` |
| 本地 vLLM / llama-server | `http://127.0.0.1:8000/v1` | 你起的模型名（可不要 key） |

### 1.3 三步验证（由便宜到贵）

```powershell
pah providers doctor --role llm              # 只读配置：5 项 PASS/FAIL，不花钱
pah providers doctor --role llm --call       # 花一次极小调用（约 22 tokens）
pah run --spec fixtures/design/demo-peg-hole-interfere.json --policy llm --rounds 3 --out runs/first-llm
```

通过长这样（实测）：

```
PASS  credential           $DEEPSEEK_API_KEY is set
PASS  live call           'OK' in 0.94s  tokens=21+1  billing=per_token
0 failing check(s)
```

跑完看**三个地方**，缺一个就是没接通：

| 看哪里 | 期望 |
|---|---|
| `runs/first-llm/report.json` → `llm` | `{"calls": 1, "tokens": 4254, ...}` —— 钱和 token 有账 |
| 同文件 → `rounds[0].proposed_by` | `"llm"`（不是 `"rule"`） |
| 同文件 → `rounds[0].reason` | 模型自己写的诊断理由（例："peg is 0.1 mm oversized per side…"） |

第一次实测就是 1 轮 `set_clearance(0.35)` 修好，4 个失败码全清。

### 1.4 失败长什么样（都是 exit 3 + 一句话，不会假装跑通）

| 现象 | 说明 |
|---|---|
| `needs $DEEPSEEK_API_KEY, which is not set` | 当前 shell 没这个变量（换窗口/IDE 常见） |
| `no default llm provider configured for 'llm'` | `pah.toml` 没找到或没 `[llm] default` |
| `unknown provider 'x' for 'llm'` | `default` 写成了不存在的 id |
| `answered HTTP 401 / 404` | key 或 `base_url`/`model` 不对（doctor `--call` 会直接暴露） |

---

## 2. 接入 DiT / CAD 内核（15 分钟）

### 2.1 它接在哪一层

```
DesignSpec（LLM 选的参数）
      │  ← 你接的地方：Materializer 端口
      ▼
materialize(spec, out_dir) → MaterializedDesign{ assembly, meshes, paths }
      ▼
验证器（间隙 / 干涉 / 插入路径 / 可打印性 / MuJoCo 沉降）
```

契约要求你的函数做到四件事：

1. **两个零件都是 watertight 的世界坐标网格**（mm），也就是 spec 里的位姿已经应用；
2. 用 `pah.design.api.assembly_from_spec(...)` 造 `assembly`（几何身份 hash 依赖它）；
3. STL 写到 `out_dir/geometry/<part_id>.stl`（中断恢复会重读，不重跑模型）；
4. 同一个 spec 两次物化结果一致（`geom_hash` 与 STL 字节）——这是可复现验收的一部分。

### 2.2 复制模板并填一个函数

```powershell
copy examples\dit_materializer.py my_dit_materialize.py
```

模板已经把「参数化造 body + 只让 DiT 生成 insert + 写盘 + 造 assembly + 传 mate_axis」写好了，
**你只改 `generate_insert(spec, reference)`**：

```python
def generate_insert(spec, reference):
    # reference = 参数化销（世界坐标、已就位、mm），用来对齐坐标系
    model = _load_model()                     # 建议模块级缓存
    out = model.generate(                     # 你的调用：本地 torch 或 HTTP 都行
        prompt=spec.requirement,
        params=spec.interface.params,
    )
    mesh = trimesh.Trimesh(vertices=out.v, faces=out.f)
    return place_like(mesh, reference)        # 或模型直接输出世界坐标就不用这步
```

不要写「模型不可用就用参数化几何顶上」——那会让 report 说的生成器和实际不符。模板里
`generate_insert` 默认就是**抛错**，这是故意的。

### 2.3 注册到 `pah.toml`

```toml
[gen]
default = "dit"

[gen.providers.dit]
kind   = "python"
# 两种写法都支持：
#   1) 文件路径（首次测试最省事，不用打包）
compat = { target = "my_dit_materialize.py:materialize" }
#   2) 可 import 的模块（正式发布用；需 pip install -e 或 PYTHONPATH）
# compat = { target = "my_dit.materialize:materialize" }
```

### 2.4 三步验证

```powershell
python examples\dit_materializer.py --selftest     # 用桩销证明管线通，不加载模型
pah providers doctor --role gen --call             # 用你的适配器物化一次，查 2 件水密 + hash
pah run --spec fixtures/design/demo-peg-hole-clear.json --policy rule --rounds 3 --out runs/first-dit
```

`--selftest` 通过但 `doctor --call` 失败，问题一定在 `generate_insert` 或 import，不在 harness。
两者都过之后，`pah run` 走的就是你的模型（report 的 `geometry.parts[*].hash` 会变）。

### 2.5 具体到混元生 3D（腾讯云）：它不是一个 key

腾讯云用的是 **SecretId + SecretKey** 两把，走 TC3 签名，域名 `ai3d.tencentcloudapi.com`，
版本 `2025-05-13`，流程是 **Submit → Query 轮询 → 下载文件**（[接口文档](https://cloud.tencent.cn/document/api/1804/123447)）。
适配器我已经写好：`examples/hunyuan3d_materializer.py`。

**1）拿钥匙**：腾讯云控制台 → 访问管理 CAM → API 密钥管理 → 新建密钥（得到 SecretId / SecretKey）；
再到 [混元生3D 控制台](https://console.cloud.tencent.com/ai3d) 开通服务（按任务计费）。

**2）写进 `.env`**（pah 启动时自动读）：

```ini
TENCENTCLOUD_SECRET_ID=AKID...
TENCENTCLOUD_SECRET_KEY=...
TENCENTCLOUD_REGION=ap-guangzhou
```

**3）三步验证（第 1 步不花钱）**：

```powershell
python examples\hunyuan3d_materializer.py --check        # SDK + 两把 key 就绪？不调用服务
pah providers doctor --role gen --call                   # 真生成一次（会花一个任务的费用）
pah run --spec fixtures\design\demo-peg-hole-interfere.json --policy rule --rounds 3 --out runs/hunyuan
```

**3.5）「接的是不是 2.5」——两层选择，别混**

| 层 | 选什么 | 怎么设 | 取值 |
|---|---|---|---|
| 档位 | 调哪组 action | `.env` 的 `HUNYUAN3D_TIER` | `standard`（默认）/ `rapid`（极速）/ `pro`（最高精度、PBR、多视角） |
| 版本 | 请求里的 `Model` 参数 | `.env` 的 `HUNYUAN3D_MODEL` | `HY-3D-2.5` / `HY-3D-3.0` / `HY-3D-3.1` 等，**以你开通后接口返回的取值为准** |

不设 `HUNYUAN3D_MODEL` = 用服务默认版本——那篇报告就不能声称「用的是 2.5」。
要锁定就在 `.env` 里写：

```ini
HUNYUAN3D_MODEL=HY-3D-2.5
HUNYUAN3D_TIER=pro
```

`python examples\hunyuan3d_materializer.py --check` 会把 tier / model / action 三行打出来，
一眼确认你接的到底是哪一层。

`--check` 在没填 key 时会明确报 `TENCENTCLOUD_SECRET_ID is not set`，填完变 0 problem。

**4）`pah.toml` 指过去**：

```toml
[gen]
default = "hunyuan"
[gen.providers.hunyuan]
kind = "python"
compat = { target = "examples/hunyuan3d_materializer.py:materialize" }
```

**两条口径**（适配器里写死了，不是建议）：
- **接口尺寸由 spec 决定，不由模型决定**：模型只生成 insert 的自由形状，按 `d_peg_mm` **等比**缩放到配合尺寸，再对齐 seated pose。非等比缩放会把销子压扁、骗过间隙验证器。
- **没有 key 就抛错**，绝不退回参数化几何——`report.run.generator` 写的是 `hunyuan:python`，偷偷换实现就等于伪造实验条件。

### 2.6 本地 torch 模型注意三件事

1. **模块级缓存模型**，别每次 `materialize` 都加载（一次 episode 会物化多轮）；
2. 设备/权重路径从环境变量读（`DIT_MODEL`、`DIT_DEVICE=cuda`），不要让代码里写死；
3. 输出先转 `trimesh.Trimesh` 再检查 `is_watertight`——不水密的壳验证器会正确拒绝它。

### 2.6 成本与已知限制（诚实口径）

- 级联目前把**每次物化记 1 次 `geom_calls`**，不区分「trimesh 秒级」还是「DiT 十秒」；
  等预算对比时，重生成器的真实代价被低估。修法：给 `Materializer` 加成本回调，进 P1。
- 生成层只能替换**物化**，不能替换**决策**；决策换的是 policy 插件。
- 非凸 insert 的物理档会记 `not_evaluated`（MuJoCo 只碰凸包），不是失败也不是通过。

---

## 3. 第一次组合：LLM 决策 + DiT 物化

```powershell
pah run --spec fixtures/design/demo-peg-hole-interfere.json `
        --policy llm --rounds 3 --out runs/first-both --config pah.toml
```

然后读 `runs/first-both/report.json`：

| 字段 | 含义 |
|---|---|
| `rounds[*].proposed_by` | `llm` = 决策来自模型 |
| `rounds[*].operator` / `params` | 模型提议的结构化动作 |
| `llm.calls` / `llm.tokens` | 这次实验的模型成本 |
| `budget.geom_calls.charged` | 几何调用账（含物化） |
| `counts.repaired_by_policy` / `environment_errors` / `not_evaluated` | **三列永不相加** |
| `geometry.parts[*].hash` | 证明几何来自你的生成器（换适配器 hash 会变） |

---

## 4. 排错顺序（照这个顺序查，别跳）

1. `pah providers doctor` —— 配置/凭据/目标函数在不在；
2. `pah providers doctor --call` —— 服务真的应答吗（这一层过了，后面才可能是 harness 的问题）；
3. `pah design sanity` —— 验证器自检（5 个全 PASS 才可信）；
4. `python scripts/smoke_design.py` —— 物化 + 验证整条链在参数化几何下是否正常；
5. `pah run ... --rounds 1` —— 最小 episode，看 `report.json` 的 `rounds[0].verdict`；
6. `pah run ... --resume` / `--inject-fault verifier` —— 恢复与异常两条验收；
7. 最后才怀疑 harness 本身：`python scripts/acceptance.py` 六条判据应全 PASS。

> 一条经验：**`--call` 都过不了的，不要去看 trace**。先让服务应答，再谈闭环。
