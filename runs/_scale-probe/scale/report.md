# M0 connector probe

- edges file: `D:\yanyi\rl_assembly\research\contact-graph-anchor\edges_v2.json`
- geometry: `D:\yanyi\rl_assembly\data\assemblies_pre_decimate`
- split: `feasible`, scope: `subset`, assemblies: 5, unit_mm rule: `scale`
- wall clock: 0.0 s

| quantity | probe | gold | delta |
|---|---:|---:|---:|
| edges_total | 953 | — | — |
| feasible | 38 | 705 | -667 |
| success | 24 | 388 | -364 |
| rate | 63.2% | 55.0% | +8.2% |

**SUBSET** (38/38 edges ran)

- `unreatable_too_small`: 1 (gold 29)
- adapter-feasible: 37

## failure codes

- `gap_too_wide`: 8
- `degenerate_other`: 2
- `socket_breaks_wall`: 1
- `no_material_hit`: 1
- `degenerate_tangent`: 1
- `unreatable_too_small`: 1

## edge-by-edge vs the frozen gold rows

- compared: 38, mismatches: 0
