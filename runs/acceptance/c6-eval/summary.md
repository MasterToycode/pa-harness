# PAH v2 evaluation — design plane

- suite: `D:\yanyi\pa-harness\fixtures\design`
- policies: `rule`, `random`
- rounds per episode: **2** (equal budget for every cell)
- cells: 16 (16 completed, 0 harness errors)

## 口径（读数字之前先读这一段）

- This table is the **design plane only** (`DesignSpec → geometry → certificates → policy → re-materialise → re-verify`). No number here comes from `runs/m0-connector` or `runs/m1-min`; those are frozen dataset evidence with their own reports and are never merged into these columns.
- `repaired_by_policy` is an outcome; `environment_errors` and `unknown` are tool health; `harness_errors` is a cell that never ran. **The columns are never summed** — a crashed verifier must not look like a worse policy.
- `mean_*` columns are over completed cells only. A mean that included the error cells would understate every policy by the same invented amount.
- manufacturing_validated: not evaluated (no print orientation declared) — that is not a claim that the designs are printable.

## Per policy

| policy | cells | repaired_by_policy | mean_rounds | mean_geom_charged | mean_geom_verified | mean_sim_charged | mean_sim_verified | environment_errors | unknown | harness_errors |
|---|---|---|---|---|---|---|---|---|---|---|
| `rule` | 8 | 8/8 (100.0%) | 0.62 | 6.88 | 8.12 | 0.75 | 0.88 | 0 | 0 | 0 |
| `random` | 8 | 6/8 (75.0%) | 1.00 | 9.12 | 10.00 | 0.88 | 1.00 | 0 | 0 | 0 |

## Per case

| case | kind | action_space | `rule` | `random` |
|---|---|---|---|---|
| `demo-dovetail` | dovetail | 6 | repaired | repaired |
| `demo-peg-hole-clear` | peg_hole | 7 | repaired | repaired |
| `demo-peg-hole-interfere` | peg_hole | 7 | repaired | repaired |
| `demo-snap-fit` | snap_fit | 7 | repaired | repaired |
| `stress.stress-dovetail-tight` | dovetail | 6 | repaired | repaired |
| `stress.stress-peg-hole-loose` | peg_hole | 7 | repaired | repaired |
| `stress.stress-peg-hole-oversize` | peg_hole | 7 | repaired | exhausted |
| `stress.stress-snap-fit-swelled` | snap_fit | 7 | repaired | exhausted |

## How to read one cell

`repaired`/`exhausted`/`infeasible`/`error`/`failed` are the loop's verdicts (`policy/loop.py`); `harness_error` marks a cell that never produced an episode, with the exception in `results.jsonl`. `action_space` counts the operators that can express the case's connector kind — a smaller number is an explanation for a lower repair rate, not a worse policy.
