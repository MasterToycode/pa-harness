# M0 connector probe

- edges file: `D:\yanyi\rl_assembly\research\contact-graph-anchor\edges_v2.json`
- geometry: `D:\yanyi\rl_assembly\data\assemblies_pre_decimate`
- split: `feasible`, scope: `subset`, assemblies: 5, unit_mm rule: `1000`
- wall clock: 0.0 s

| quantity | probe | gold | delta |
|---|---:|---:|---:|
| edges_total | 953 | — | — |
| feasible | 38 | 705 | -667 |
| success | 20 | 388 | -368 |
| rate | 52.6% | 55.0% | -2.4% |

**SUBSET** (38/38 edges ran)

- `unreatable_too_small`: 1 (gold 29)
- adapter-feasible: 37

## failure codes

- `gap_too_wide`: 10
- `socket_breaks_wall`: 3
- `degenerate_other`: 2
- `no_material_hit`: 1
- `degenerate_tangent`: 1
- `unreatable_too_small`: 1

## edge-by-edge vs the frozen gold rows

- compared: 38, mismatches: 8
- `asm_000000:2` adapter=ok gold=fail `fragmented_b(share=0.6543<0.6677,ncomp=3/2)`
- `asm_000000:5` adapter=degenerate_geometry gold=ok ``
- `asm_000001:12` adapter=degenerate_geometry gold=ok ``
- `asm_000001:15` adapter=ok gold=fail `degenerate_other(gap_too_wide_for_wall(gap=1.1800mm,need=2.2`
- `asm_000001:17` adapter=degenerate_geometry gold=ok ``
- `asm_000003:34` adapter=socket_breaks_wall gold=ok ``
- `asm_000003:36` adapter=socket_breaks_wall gold=ok ``
- `asm_000003:37` adapter=socket_breaks_wall gold=ok ``
