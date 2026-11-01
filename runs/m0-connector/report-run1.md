# M0 connector probe

- edges file: `D:\yanyi\rl_assembly\research\contact-graph-anchor\edges_v2.json`
- geometry: `D:\yanyi\rl_assembly\data\assemblies_pre_decimate`
- split: `feasible`, scope: `gold`, assemblies: 93, unit_mm rule: `scale`
- wall clock: 1047.7 s

| quantity | probe | gold | delta |
|---|---:|---:|---:|
| edges_total | 953 | — | — |
| feasible | 705 | 705 | +0 |
| success | 393 | 388 | +5 |
| rate | 55.7% | 55.0% | +0.7% |

**REPRODUCED-MODULO-ENVIRONMENT** (705/705 edges ran)

- `unreatable_too_small`: 29 (gold 29)
- adapter-feasible: 676
- environment-dependent edges (gold lost them to `MemoryError`, this run did not): 5 -> adjusted success 388

## failure codes

- `gap_too_wide`: 166
- `no_material_hit`: 72
- `unreatable_too_small`: 29
- `socket_breaks_wall`: 17
- `degenerate_other`: 17
- `peg_detached`: 8
- `degenerate_tangent`: 3

## edge-by-edge vs the frozen gold rows

- compared: 705, mismatches: 5
- `asm_000097:912` adapter=ok gold=fail `exception:MemoryError(Unable to allocate 56.0 MiB for an arr`
- `asm_000097:916` adapter=ok gold=fail `exception:MemoryError(Unable to allocate 14.0 MiB for an arr`
- `asm_000097:917` adapter=ok gold=fail `exception:MemoryError(Unable to allocate 14.0 MiB for an arr`
- `asm_000097:919` adapter=ok gold=fail `exception:MemoryError(Unable to allocate 14.0 MiB for an arr`
- `asm_000097:920` adapter=ok gold=fail `exception:MemoryError(Unable to allocate 56.0 MiB for an arr`
