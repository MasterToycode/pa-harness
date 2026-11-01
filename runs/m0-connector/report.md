# M0 connector probe

- edges file: `D:\yanyi\rl_assembly\research\contact-graph-anchor\edges_v2.json`
- geometry: `D:\yanyi\rl_assembly\data\assemblies_pre_decimate`
- split: `feasible`, scope: `subset`, assemblies: 1, unit_mm rule: `scale`
- wall clock: 1217.6 s

| quantity | probe | gold | delta |
|---|---:|---:|---:|
| edges_total | 953 | — | — |
| feasible | 6 | 705 | -699 |
| success | 2 | 388 | -386 |
| rate | 33.3% | 55.0% | -21.7% |

**SUBSET** (6/6 edges ran)

- `unreatable_too_small`: 0 (gold 29)
- adapter-feasible: 6

## failure codes

- `gap_too_wide`: 3
- `socket_breaks_wall`: 1

## `MemoryError` retries

- `retried_n`: 0 (retry attempts made after an out-of-memory verdict)
- `oom_n`: 0 (edges still recorded `verifier_oom`)
- `verifier_oom` failures: 0 — counted apart from `bool_failed`, never merged: one is the machine, the other is the geometry

## IVR (`ivr_exact`, ε = 1e-06)

| quantity | value |
|---|---:|
| `success_original` | 2 |
| `success_ivr_validated` ∈ [`lo`, `hi`] | [2, 2] |
| `n_unknown` | 0 |
| measured (`ivr` known) | 2 |
| over ε (success but interpenetrating) | 0 |

how each reading was produced: `manifold` 2

`lo` counts `unknown` as not IVR-validated, `hi` counts it as validated; the interval is the answer and `n_unknown` says how much of it is the boolean engine's silence rather than a measurement (§4.8).

## edge-by-edge vs the frozen gold rows

- compared: 6, mismatches: 0
