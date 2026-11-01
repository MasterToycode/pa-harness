# M1 repair probe — auto-generated tables

- source: `runs\m1-min\source-rows.jsonl` (705 rows, sha256 `b7e0faf03f00c506…`)
- geometry: `D:\yanyi\rl_assembly\data\assemblies_pre_decimate`
- targets: 238 edges, written 238 (complete)
- wall clock: 1098.8 s

## 1. edge level

| source code | edges | repaired | rate | printable |
|---|---:|---:|---:|---:|
| `gap_too_wide` | 166 | 62 | 37.4% | 57 |
| `no_material_hit` | 72 | 0 | 0.0% | 0 |
| **total** | **238** | **62** | **26.1%** | **57** |

yield over the 705-edge denominator: 55.7% → 64.5% (393 → 455)

## 2. assembly level (spanning tree over buildable edges)

| quantity | before | after |
|---|---:|---:|
| completely solvable | 19 | 20 |
| reachable parts (Plan B) | 393/696 | 432/696 |
| file-feasible ceiling | 53 | 53 |

flipped to solvable: asm_000072
lost: —

## 3. cost

| quantity | value |
|---|---:|
| attempts | 402 |
| `connector.build` calls | 402 |
| `measure_axis` calls | 3213 |
| wall s (whole run) | 1098.8 |
| wall s per edge p50 / p95 | 1.84855 / 17.412 |
| attempts histogram | {'0': 7, '1': 62, '2': 167, '3': 2} |

| operator | attempted | succeeded | hit rate |
|---|---:|---:|---:|
| `resize_connector` | 227 | 57 | 25.1% |
| `relocate_connector` | 173 | 5 | 2.9% |
| `swap_family` | 2 | 0 | 0.0% |

## 4. residual attribution

unrepaired: **176**

| final reason | count |
|---|---:|
| `no_material_hit` | 95 |
| `gap_too_wide` | 67 |
| `unreatable_too_small` | 7 |
| `degenerate_other` | 5 |
| `bool_failed` | 1 |
| `socket_breaks_wall` | 1 |

| why the ladder stopped | count |
|---|---:|
| `rungs` | 161 |
| `terminal` | 8 |
| `no_plan` | 7 |

| src → final | count |
|---|---:|
| `no_material_hit -> no_material_hit` | 60 |
| `gap_too_wide -> gap_too_wide` | 59 |
| `gap_too_wide -> no_material_hit` | 35 |
| `no_material_hit -> gap_too_wide` | 8 |
| `gap_too_wide -> degenerate_other` | 5 |
| `no_material_hit -> unreatable_too_small` | 4 |
| `gap_too_wide -> unreatable_too_small` | 3 |
| `gap_too_wide -> bool_failed` | 1 |
| `gap_too_wide -> socket_breaks_wall` | 1 |

## claim checks

- printable: 57/62 repaired edges (r p50 0.615663 mm, wall remaining p50 0.450936 mm)
- parts hashed: 71 assemblies, changed: 0
- gate prediction: 297/402 attempts agree (false +103 / -2)
