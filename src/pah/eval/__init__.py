"""The equal-budget evaluation plane.

`runner` loads a suite, runs one episode per (case, policy) under the same budget and
returns plain rows; `report` writes `results.jsonl` and the `summary.md` that keeps
outcome, tool health and harness errors in separate columns.

Nothing here is imported by the runtime or the loop: evaluation is a consumer of the
harness, and a harness that depended on its own benchmark would be measuring itself.
"""
