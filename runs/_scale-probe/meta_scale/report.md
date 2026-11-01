# M0 connector probe

- edges file: `D:\yanyi\rl_assembly\research\contact-graph-anchor\edges_v2.json`
- geometry: `D:\yanyi\rl_assembly\data\assemblies_pre_decimate`
- split: `feasible`, scope: `subset`, assemblies: 5, unit_mm rule: `meta_scale`
- wall clock: 0.0 s

| quantity | probe | gold | delta |
|---|---:|---:|---:|
| edges_total | 953 | — | — |
| feasible | 38 | 705 | -667 |
| success | 4 | 388 | -384 |
| rate | 10.5% | 55.0% | -44.5% |

**SUBSET** (38/38 edges ran)

- `unreatable_too_small`: 28 (gold 29)
- adapter-feasible: 10

## failure codes

- `unreatable_too_small`: 28
- `gap_too_wide`: 4
- `no_material_hit`: 1
- `socket_breaks_wall`: 1

## edge-by-edge vs the frozen gold rows

- compared: 38, mismatches: 22
- `asm_000000:3` adapter=unreatable_too_small gold=ok ``
- `asm_000000:5` adapter=unreatable_too_small gold=ok ``
- `asm_000004:48` adapter=unreatable_too_small gold=ok ``
- `asm_000001:12` adapter=unreatable_too_small gold=ok ``
- `asm_000001:14` adapter=unreatable_too_small gold=ok ``
- `asm_000001:16` adapter=unreatable_too_small gold=ok ``
- `asm_000001:17` adapter=unreatable_too_small gold=ok ``
- `asm_000002:20` adapter=degenerate_geometry gold=ok ``
- `asm_000002:22` adapter=unreatable_too_small gold=ok ``
- `asm_000003:30` adapter=unreatable_too_small gold=ok ``
