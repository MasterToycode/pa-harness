# 首次接入实测记录（2026-11-01）

> 目的：证明「LLM 决策层 + 生成器适配层」两个口都能在真栈上跑通，并留下可复现的命令与输出。
> 环境：Windows / Python 3.13 / `pah` 已 `pip install -e .` / 仓库 `D:\yanyi\pa-harness`
> 结论：**LLM 侧全通（真实调用）；生成器侧管线全通（用桩适配器），但真模型未接——需要你的 3D API key 或本地权重。**

---

## 1. LLM 决策层

### 1.1 未设 key（应该明确失败，而不是假装跑通）

```powershell
$env:DEEPSEEK_API_KEY=''
pah providers doctor --role llm
```

```
PASS  provider configured  deepseek (kind=openai-compatible)
PASS  base_url             https://api.deepseek.com/v1
PASS  model                deepseek-chat
PASS  billing declared     per_token
FAIL  credential           $DEEPSEEK_API_KEY is NOT set in this shell
1 failing check(s)          exit=1
```

### 1.2 设 key 后（配置自检全过）

```
PASS  credential           $DEEPSEEK_API_KEY is set
0 failing check(s)          exit=0
```

### 1.3 真实探针调用（一次极小 completion）

```powershell
pah providers doctor --role llm --call
```

```
PASS  live call           'OK' in 1.07s  tokens=21+1  billing=per_token
0 failing check(s)          exit=0
```

### 1.4 LLM 驱动的修复闭环

```powershell
pah run --spec fixtures/design/demo-peg-hole-interfere.json --policy llm --rounds 3 --out runs/first-llm
```

```
round 0  pass  set_clearance     geom=11 sim=2
verdict   repaired (verified after repair)     rounds 1/3
counts    repaired_by_policy=1  environment_errors=0  unknown=0  not_evaluated=1
```

`runs/first-llm/report.json`：

| 字段 | 实测值 | 含义 |
|---|---|---|
| `rounds[0].proposed_by` | `"llm"` | 决策来自模型，不是规则回落 |
| `rounds[0].operator` | `set_clearance` | 结构化动作（几何字段不存在） |
| `llm.calls` / `llm.tokens` | `1` / `4258` | 本次实验的模型成本，有账 |

**判定：LLM 接入完成。** 换厂商只改 `pah.toml` 的 `[llm.providers.*]` 五项，不碰代码。

---

## 2. 生成器（DiT/CAD）层

> 真模型未接：手上没有 3D 生成 API key，也没有你的本地权重。所以这里验证的是**从配置到物化再到验证的整条管线**，用模板自带的桩适配器（显式声明「不是生成器」）。真模型只需要替换一个函数。

### 2.1 适配器自检（桩销，不加载模型）

```powershell
python examples/dit_materializer.py --selftest
```

```
spec        demo-peg-hole-clear  kind=peg_hole  clearance=+0.150 mm
part body     watertight=True volume=28172.7 mm3 faces=270
part insert   watertight=True volume=1032.9 mm3 faces=256
assembly    bfa3b7886e62484c9b77ea9e5097969f  parts=2
hashes      ['sha256:65162057a37', 'sha256:b69249a4732']
mate_axis   (0.0, 0.0, 1.0)  approach=10.0 mm
plumbing OK.
```

### 2.2 让 harness 从配置里解析到适配器并真物化一次

`$TEMP\pah-dit-test\pah.toml`：

```toml
[gen]
default = "dit"

[gen.providers.dit]
kind = "python"
compat = { target = "examples/dit_materializer.py:stub_materialize" }
```

```powershell
pah providers doctor --role gen --call --config $TEMP\pah-dit-test\pah.toml
```

```
provider   dit  kind=python      target  examples/dit_materializer.py:stub_materialize
PASS  target importable
PASS  live materialise    0.007s parts={'body': 28172.69, 'insert': 1032.91}
0 failing check(s)          exit=0
```

### 2.3 完整 episode 走配置生成器（含修复闭环）

```powershell
pah run --spec fixtures/design/demo-peg-hole-interfere.json --policy rule `
        --rounds 3 --out runs/first-dit-repair --config $TEMP\pah-dit-test\pah.toml
```

```
verdict   repaired     rounds 1/3
generator: dit:python | replay hashes: ['9edff9d1287a0ca83b588eb524fa18b7']
final geom: {'body': 'sha256:2dcf8cf45', 'insert': 'sha256:8f4144919'}
```

`run.generator = "dit:python"` 是这次跑「几何确实来自配置的生成器」的凭据；换真模型后 hash 会变。

### 2.4 反例：配了但加载不起来（exit 3，绝不静默退回参数化）

```
pah run: gen provider 'broken': adapter file D:\yanyi\pa-harness\no_such_module.py does not exist
exit=3
```

---

## 3. 真模型接进来还差什么（清单）

| 项 | 现在的状态 | 你要做的 |
|---|---|---|
| 3D 生成服务 | 未接 | 申请腾讯混元生 3D 或百炼 Tripo（或本地 Hunyuan3D-2 mini / TripoSR） |
| `generate_insert(spec, reference)` | 抛错占位 | 调服务 → 拿 GLB/OBJ → `trimesh.load` → 尺度锚定 → 朝向对齐 → 水密修复 → 返回 |
| 尺度锚定 | 无 | 用 spec 的已知尺寸归一（销径/槽宽必须等于参数） |
| 朝向 | `place_like` 只对齐 XY/最低 Z，不旋转 | 换成你模型的位姿协议，或让模型直接输出世界坐标 |
| 水密 | 模板会拒绝非水密 | 体素重建 / manifold 布尔补全，再交给验证器 |
| 复现性 | 云 API 会静默改版 | 正式数字用本地固定权重；report 的 `run.generator` 只能记 id，版本请写进 `spec.provenance` |

---

## 4. 本次实测成本

| 项 | 次数 | tokens |
|---|---|---|
| LLM 探针（doctor --call） | 2 | 44 |
| LLM 驱动的 episode | 1 | 4258 |
| 生成器物化 | 3 | 0（本地） |

DeepSeek 按 `per_token` 声明价计，约 **$0.005**（¥0.04 以内）。

---

## 5. 复现命令（一条一条贴）

```powershell
cd D:\yanyi\pa-harness
$env:DEEPSEEK_API_KEY = "sk-..."                  # 只进环境变量

pah providers doctor --role llm --call
pah run --spec fixtures/design/demo-peg-hole-interfere.json --policy llm --rounds 3 --out runs/first-llm

python examples\dit_materializer.py --selftest
copy examples\dit_materializer.py my_dit_materialize.py
#   ← 只改这个文件里的 generate_insert()
#   ← pah.toml 里把 [gen] 指到 my_dit_materialize.py:materialize
pah providers doctor --role gen --call
pah run --spec fixtures/design/demo-peg-hole-interfere.json --policy llm --rounds 3 --out runs/first-both
```
