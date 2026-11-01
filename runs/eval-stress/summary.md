# PAH v2 evaluation — design plane

- suite: `fixtures/design/stress`
- policies: `rule`, `random`
- rounds per episode: **3** (equal budget for every cell)
- cells: 8 (8 completed, 0 harness errors)

## 口径（读数字之前先读这一段）

- This table is the **design plane only** (`DesignSpec → geometry → certificates → policy → re-materialise → re-verify`). No number here comes from `runs/m0-connector` or `runs/m1-min`; those are frozen dataset evidence with their own reports and are never merged into these columns.
- `repaired_by_policy` is an outcome; `environment_errors` and `unknown` are tool health; `harness_errors` is a cell that never ran. **The columns are never summed** — a crashed verifier must not look like a worse policy.
- `mean_*` columns are over completed cells only. A mean that included the error cells would understate every policy by the same invented amount.
- manufacturing_validated: not evaluated (no print orientation declared) — that is not a claim that the designs are printable.

## Per policy

| policy | cells | repaired_by_policy | mean_rounds | mean_geom_charged | mean_geom_verified | mean_sim_charged | mean_sim_verified | environment_errors | unknown | harness_errors |
|---|---|---|---|---|---|---|---|---|---|---|
| `rule` | 4 | 4/4 (100.0%) | 1.00 | 11.00 | 10.00 | 1.00 | 1.00 | 0 | 0 | 0 |
| `random` | 4 | 2/4 (50.0%) | 2.25 | 18.50 | 16.25 | 1.50 | 1.50 | 0 | 0 | 0 |

## Per case

| case | kind | action_space | `rule` | `random` |
|---|---|---|---|---|
| `stress-dovetail-tight` | dovetail | 6 | repaired | repaired |
| `stress-peg-hole-loose` | peg_hole | 7 | repaired | repaired |
| `stress-peg-hole-oversize` | peg_hole | 7 | repaired | exhausted |
| `stress-snap-fit-swelled` | snap_fit | 7 | repaired | exhausted |

## How to read one cell

`repaired`/`exhausted`/`infeasible`/`error`/`failed` are the loop's verdicts (`policy/loop.py`); `harness_error` marks a cell that never produced an episode, with the exception in `results.jsonl`. `action_space` counts the operators that can express the case's connector kind — a smaller number is an explanation for a lower repair rate, not a worse policy.
