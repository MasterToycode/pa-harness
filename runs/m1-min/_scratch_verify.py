"""Scratch: does the fixed plane basis rescue the two `asm_000006` anchor misses?

The same two edges that `_scratch_grid.py` rescued, now through the module
instead of a hand-rolled search. Run: python runs/m1-min/_scratch_verify.py
"""
from __future__ import annotations

import sys

sys.path.insert(0, r"D:\yanyi\pa-harness\src")
sys.path.insert(0, r"D:\yanyi\pa-harness\scripts")

from m1_repair import edge_for, run_case  # noqa: E402  (scripts/ on the path)
from pah.policy.repair_script import AttemptBudget  # noqa: E402
from pah.tools.verifiers.connector_feasible import load_cases  # noqa: E402

WANT = {"asm_000006": [58, 64], "asm_000002": [25], "asm_000044": [None]}


def main() -> None:
    cases = {c.case_id: c for c in load_cases(split="feasible")}
    for case_id, edge_ids in WANT.items():
        case = cases[case_id]
        targets = [
            {"edge_id": j["edge_id"], "code": "no_material_hit", "detail": ""}
            for j in case.joints
            if (edge_ids[0] is None or int(j["edge_id"]) in edge_ids)
        ]
        # only the ones that actually failed with no_material_hit in the M0 ledger
        import json
        ledger = {
            (r["case_id"], int(r["edge_id"]))
            for r in (
                json.loads(line)
                for line in open(r"D:\yanyi\pa-harness\runs\m1-min\source-rows.jsonl",
                                 encoding="utf-8")
                if line.strip()
            )
            if r["code"] == "no_material_hit"
        }
        targets = [t for t in targets if (case_id, int(t["edge_id"])) in ledger]
        if not targets:
            print(f"{case_id}: no target edges in this list")
            continue
        print(f"\n=== {case_id}: {[t['edge_id'] for t in targets]}")
        rows, record = run_case(
            case, targets, budget_factory=lambda: AttemptBudget(max_attempts=3)
        )
        for row in rows:
            print(
                f"  {row['case_id']}:{row['edge_id']} -> "
                f"{'OK' if row['success'] else row['final_reason']} "
                f"ops={[a['operator'] for a in row['attempts']]} "
                f"r={row['r_mm']} shift={row['anchor_shift_mm']} printable={row['printable']}"
            )
        print(f"  changed parts: {record['changed_parts']}")


if __name__ == "__main__":
    main()

