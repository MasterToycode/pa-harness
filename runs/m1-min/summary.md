# M1 repair probe — auto-generated tables

- source: `runs\m1-min\source-rows.jsonl` (705 rows, sha256 `b7e0faf03f00c506…`)
- geometry: `D:\yanyi\rl_assembly\data\assemblies_pre_decimate`
- targets: 238 edges, written 238 (complete)
- wall clock: 1412.2 s

## 1. edge level

| source code | edges | repaired | rate | printable |
|---|---:|---:|---:|---:|
| `gap_too_wide` | 166 | 99 | 59.6% | 99 |
| `no_material_hit` | 72 | 25 | 34.7% | 25 |
| **total** | **238** | **124** | **52.1%** | **124** |

yield over the 705-edge denominator: 55.7% → 73.3% (393 → 517)

## 2. assembly level (spanning tree over buildable edges)

| quantity | before | after |
|---|---:|---:|
| completely solvable | 19 | 25 |
| reachable parts (Plan B) | 399/722 | 484/722 |
| file-feasible ceiling | 53 | 53 |

flipped to solvable: asm_000001, asm_000012, asm_000021, asm_000069, asm_000071, asm_000072
lost: —

## 3. cost

| quantity | value |
|---|---:|
| attempts | 379 |
| `connector.build` calls | 379 |
| `measure_axis` calls | 3213 |
| wall s (whole run) | 1412.2 |
| wall s per edge p50 / p95 | 3.5643 / 19.9947 |
| attempts histogram | {'0': 7, '1': 95, '2': 124, '3': 12} |

| operator | attempted | succeeded | hit rate |
|---|---:|---:|---:|
| `resize_connector` | 194 | 58 | 29.9% |
| `relocate_connector` | 173 | 66 | 38.1% |
| `swap_family` | 12 | 0 | 0.0% |

## 4. residual attribution

unrepaired: **114**

| final reason | count |
|---|---:|
| `gap_too_wide` | 73 |
| `unreatable_too_small` | 18 |
| `degenerate_other` | 8 |
| `no_material_hit` | 7 |
| `peg_detached` | 3 |
| `degenerate_tangent` | 3 |
| `socket_breaks_wall` | 2 |

| why the ladder stopped | count |
|---|---:|
| `rungs` | 89 |
| `terminal` | 18 |
| `no_plan` | 7 |

| src → final | count |
|---|---:|
| `gap_too_wide -> gap_too_wide` | 53 |
| `no_material_hit -> gap_too_wide` | 20 |
| `no_material_hit -> unreatable_too_small` | 12 |
| `no_material_hit -> no_material_hit` | 7 |
| `gap_too_wide -> unreatable_too_small` | 6 |
| `gap_too_wide -> degenerate_other` | 5 |
| `no_material_hit -> degenerate_tangent` | 3 |
| `no_material_hit -> degenerate_other` | 3 |
| `no_material_hit -> peg_detached` | 2 |
| `gap_too_wide -> socket_breaks_wall` | 2 |
| `gap_too_wide -> peg_detached` | 1 |

### residual `gap_too_wide`: which term binds

- parsed from the last attempt's detail: 73
- `eng_max == depth_max` (the depth cap is the binding term): **73**
- wall term binds: 0
- local gap ≥ the recorded `depth_max`: 57 (gap/depth_max p50 1.762776)

## claim checks

- printable: 124/124 repaired edges (r p50 0.61213 mm, 0 at the 0.25 mm floor, wall remaining / needed p50 3.127586)
- parts hashed: 71 assemblies, changed: 0
- gate prediction: 369/379 attempts agree (false +10 / -0)
