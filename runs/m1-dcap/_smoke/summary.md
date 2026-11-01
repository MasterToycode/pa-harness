# M1-dcap 变体：边级结果（自动生成）

行数（按臂）：{'baseline': 7, 'cap_only': 7, 'variant': 7}；分母组：{'g73': 2, 'e80': 2, 'all238': 7}

## 1. 修复成功数（配对）

| 处理 vs 对照 | 组 | n | 对照成功 | 处理成功 | b（仅处理成功） | c（仅对照成功） | McNemar p |
|---|---|---:|---:|---:|---:|---:|---:|
| cap_only_vs_baseline | g73 | 2 | 0/2 = 0.0% | 0/2 = 0.0% | 0 | 0 | n/a |
| cap_only_vs_baseline | e80 | 2 | 0/2 = 0.0% | 0/2 = 0.0% | 0 | 0 | n/a |
| cap_only_vs_baseline | all238 | 7 | 5/7 = 71.4% | 5/7 = 71.4% | 0 | 0 | n/a |
| variant_vs_baseline | g73 | 2 | 0/2 = 0.0% | 0/2 = 0.0% | 0 | 0 | n/a |
| variant_vs_baseline | e80 | 2 | 0/2 = 0.0% | 0/2 = 0.0% | 0 | 0 | n/a |
| variant_vs_baseline | all238 | 7 | 5/7 = 71.4% | 5/7 = 71.4% | 0 | 0 | n/a |
| variant_vs_cap_only | g73 | 2 | 0/2 = 0.0% | 0/2 = 0.0% | 0 | 0 | n/a |
| variant_vs_cap_only | e80 | 2 | 0/2 = 0.0% | 0/2 = 0.0% | 0 | 0 | n/a |
| variant_vs_cap_only | all238 | 7 | 5/7 = 71.4% | 5/7 = 71.4% | 0 | 0 | n/a |

c 的归因（238 全集）：

- cap_only_vs_baseline：c = 0（被壁厚守则否掉 0，其余 0）
- variant_vs_baseline：c = 0（被壁厚守则否掉 0，其余 0）
- variant_vs_cap_only：c = 0（被壁厚守则否掉 0，其余 0）

## 2. `socket_breaks_wall` 与壁厚否决

| 臂 | 成功 | `socket_breaks_wall` | `degenerate_thin_wall` | n |
|---|---:|---:|---:|---:|
| baseline | 5 | 0 | 0 | 7 |
| cap_only | 5 | 0 | 0 | 7 |
| variant | 5 | 0 | 0 | 7 |

## 3. IVR（修好的边；pair-local 与装配级贡献）

- **baseline**：n=5，已知 5，unknown 0，error 0；pair IVR p50=0（min 0/max 3e-06）；装配级贡献 [lo, hi] = [0, 1.062]
- **cap_only**：n=5，已知 5，unknown 0，error 0；pair IVR p50=0（min 0/max 4.3e-05）；装配级贡献 [lo, hi] = [0, 1.062]
- **variant**：n=5，已知 5，unknown 0，error 0；pair IVR p50=0（min 0/max 2.4e-05）；装配级贡献 [lo, hi] = [0, 1.062]

## 4. 失败码前后对比（238 全集）

| 码 | baseline | cap_only | variant |
|---|---:|---:|---:|
| `ok` | 5 | 5 | 5 |
| `gap_too_wide` | 2 | 2 | 2 |

变化（按边、每对配对）：

- cap_only_vs_baseline：无变化
- variant_vs_baseline：无变化
- variant_vs_cap_only：无变化

## 5. 最小剩余壁厚（修好的边）

- **baseline**：n=5；剩余壁厚 p50=1.018 p05=0.2277 min=0.1654；floor p50=0.6946；余量 p50=0.1829 min=-0.3989；踩线（=floor）0，距 floor ≤1% 0，低于 floor 2；r p50=0.9766，eng p50=2.157；守卫生效 0 次
- **cap_only**：n=5；剩余壁厚 p50=0.4768 p05=0.1877 min=0.1654；floor p50=0.6946；余量 p50=-0.3178 min=-0.5324；踩线（=floor）0，距 floor ≤1% 0，低于 floor 4；r p50=1.907，eng p50=2.33；守卫生效 0 次
- **variant**：n=5；剩余壁厚 p50=1.265 p05=0.498 min=0.4388；floor p50=0.6946；余量 p50=0.3052 min=0.135；踩线（=floor）0，距 floor ≤1% 0，低于 floor 0；r p50=1.557，eng p50=1.892；守卫生效 0 次

## 6. 测量本身（假设的证据）

- 可测 6 / 7 条边（其余在锚点测不到 socket 侧材质，处理不适用、沿用对照判定）；墙后有空腔 2 条，多层 2 条；局部 d_cap 比文件 d_cap 变小 2 条
- local wall / file t_b：p05 0.5339 / p50 1.017 / p95 1.897 / min 0.4575 / max 2.153
- local wall / span：p05 0.05761 / p50 1 / p95 1 / min 0.03677 / max 1
- local d_cap / file d_cap：p05 0.5339 / p50 1.205 / p95 2.348 / min 0.4575 / max 2.414
- local wall (mm)：p05 0.9818 / p50 2.043 / p95 5.488 / min 0.9779 / max 5.858
