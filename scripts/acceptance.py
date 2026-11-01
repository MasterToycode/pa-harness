"""The six P0 acceptance checks, run against the real CLI.

    python scripts/acceptance.py [--out runs/acceptance] [--policy rule]

`docs/contracts/v2-interfaces.md` §7 fixes the criteria; this script is the
executable form of them, so the claim "P0 is done" is reproducible instead of
recalled:

1. an episode runs to a conclusion with a complete trace and a final design;
2. a restart from a clean stop reproduces the uninterrupted run's hash sequence;
3. the round budget stops the episode exactly where it says;
4. two runs of the same spec/policy/seed produce identical `geom_hash` sequences;
5. an injected verifier crash does not end the episode by crashing it;
6. the eval report keeps repaired-vs-environment counts in separate columns.

**Yield is not an acceptance criterion.** Check 1 asks that the loop closed and
left evidence, not that the repair succeeded — the doc is explicit that a policy
result is the next stage's measurement, not this stage's gate. The verdict is
printed so a reader can see it without mistaking it for a pass/fail.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
INTERFERE = REPO / "fixtures" / "design" / "demo-peg-hole-interfere.json"
SUITE = REPO / "fixtures" / "design"

#: What a check prints when it cannot run at all (a missing module, a broken CLI).
SKIP = "skip"


def _run_cli(*args: str, cwd: Path = REPO) -> subprocess.CompletedProcess[str]:
    """Run the real CLI in a child process, with stdout forced to UTF-8.

    Without `PYTHONIOENCODING` a piped stdout on a Chinese Windows locale is
    cp936, and the CLI's boxed report (═ / █) raises `UnicodeEncodeError` the
    moment it prints — which surfaces as exit 3 and looks like a crash in the
    harness rather than an encoding mismatch in the caller. The user-facing
    behaviour in a terminal is unaffected; only a captured pipe needs this.
    """
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    return subprocess.run(
        [sys.executable, "-m", "pah.cli.main", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )


def _report(out: Path) -> dict:
    return json.loads((out / "report.json").read_text("utf-8"))


def _hashes(out: Path) -> list[str]:
    rows = []
    for line in (out / "trace.jsonl").read_text("utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if isinstance(row.get("geom_hash"), str) and row["geom_hash"]:
            rows.append(row["geom_hash"])
    return rows


def _result(name: str, ok: bool, detail: str) -> dict:
    return {"check": name, "ok": ok, "detail": detail}


# --------------------------------------------------------------------------- #
# the six checks
# --------------------------------------------------------------------------- #
def check_runs_to_a_conclusion(root: Path, policy: str) -> dict:
    out = root / "c1-episode"
    shutil.rmtree(out, ignore_errors=True)
    proc = _run_cli(
        "run", "--spec", str(INTERFERE), "--policy", policy, "--rounds", "3",
        "--out", str(out),
    )
    if proc.returncode != 0:
        return _result("1 episode completes", False, f"exit {proc.returncode}: {proc.stderr.strip()[:400]}")
    report = _report(out)
    rows = len((out / "trace.jsonl").read_text("utf-8").splitlines())
    design = report.get("design", {})
    geometry = report.get("geometry", {})
    ok = (
        report["run"]["status"] == "completed"
        and rows > 0
        and bool(design.get("final_content_key"))
        and geometry.get("complete") is True
        and report["episode"]["rounds"] >= 1
    )
    detail = (
        f"verdict={report['episode']['verdict']} rounds={report['episode']['rounds']} "
        f"trace_rows={rows} geometry={'complete' if geometry.get('complete') else 'MISSING'} "
        f"final_revision={design.get('final_revision')}"
    )
    return _result("1 episode completes", ok, detail)


def check_reproducible(root: Path, policy: str) -> dict:
    first, second = root / "c4-a", root / "c4-b"
    for out in (first, second):
        shutil.rmtree(out, ignore_errors=True)
        proc = _run_cli(
            "run", "--spec", str(INTERFERE), "--policy", policy, "--rounds", "3",
            "--out", str(out), "--seed", "7",
        )
        if proc.returncode != 0:
            return _result("4 reproducible", False, f"exit {proc.returncode}: {proc.stderr.strip()[:300]}")
    left, right = _hashes(first), _hashes(second)
    return _result(
        "4 reproducible",
        bool(left) and left == right,
        f"{len(left)} hashes, identical={left == right}",
    )


def check_budget(root: Path, policy: str) -> dict:
    """One round asked for is one round spent — the ceiling is not a target."""
    out = root / "c3-budget"
    shutil.rmtree(out, ignore_errors=True)
    proc = _run_cli(
        "run", "--spec", str(INTERFERE), "--policy", policy, "--rounds", "1",
        "--out", str(out), "--inject-fault", "verifier",
    )
    if proc.returncode != 0:
        return _result("3 budget stops exactly", False, f"exit {proc.returncode}: {proc.stderr.strip()[:300]}")
    report = _report(out)
    used = report["budget"]["rounds"]["used"]
    over = (
        report["budget"]["geom_calls"]["charged"] > report["budget"]["geom_calls"]["budget"]
        or report["budget"]["sim_calls"]["charged"] > report["budget"]["sim_calls"]["budget"]
    )
    return _result(
        "3 budget stops exactly",
        used == 1 and not over,
        f"rounds used={used} of 1; geom={report['budget']['geom_calls']}; "
        f"sim={report['budget']['sim_calls']}",
    )


def check_resume(root: Path, policy: str) -> dict:
    """A clean stop plus `--resume` must land on the same hash sequence."""
    whole = root / "c2-whole"
    shutil.rmtree(whole, ignore_errors=True)
    proc = _run_cli(
        "run", "--spec", str(INTERFERE), "--policy", policy, "--rounds", "3",
        "--out", str(whole),
    )
    if proc.returncode != 0:
        return _result("2 resume matches", False, f"uninterrupted exit {proc.returncode}")

    part = root / "c2-part"
    shutil.rmtree(part, ignore_errors=True)
    stop = _run_cli(
        "run", "--spec", str(INTERFERE), "--policy", policy, "--rounds", "3",
        "--out", str(part), "--stop-after-round", "1",
    )
    if stop.returncode != 0:
        return _result("2 resume matches", False, f"stop exit {stop.returncode}: {stop.stderr.strip()[:300]}")
    resumed = _run_cli(
        "run", "--spec", str(INTERFERE), "--policy", policy, "--rounds", "3",
        "--out", str(part), "--resume",
    )
    if resumed.returncode != 0:
        return _result("2 resume matches", False, f"resume exit {resumed.returncode}: {resumed.stderr.strip()[:300]}")

    left, right = _hashes(whole), _hashes(part)
    return _result(
        "2 resume matches",
        bool(left) and left == right,
        f"uninterrupted={left} resumed={right}",
    )


def check_fault_injection(root: Path, policy: str) -> dict:
    out = root / "c5-fault"
    shutil.rmtree(out, ignore_errors=True)
    proc = _run_cli(
        "run", "--spec", str(INTERFERE), "--policy", policy, "--rounds", "3",
        "--out", str(out), "--inject-fault", "verifier",
    )
    if proc.returncode != 0:
        return _result("5 fault injection survives", False, f"exit {proc.returncode}: {proc.stderr.strip()[:300]}")
    report = _report(out)
    counts = report["counts"]
    trace = (out / "trace.jsonl").read_text("utf-8")
    ok = (
        counts["verifier_error_events"] >= 1
        and counts["unknown"] >= 1
        and "verifier.error" in trace
        and report["episode"]["rounds"] >= 1
    )
    return _result(
        "5 fault injection survives",
        ok,
        f"verifier_error_events={counts['verifier_error_events']} unknown={counts['unknown']} "
        f"rounds={report['episode']['rounds']} environment_errors={counts['environment_errors']}",
    )


def check_eval_columns(root: Path) -> dict:
    out = root / "c6-eval"
    shutil.rmtree(out, ignore_errors=True)
    proc = _run_cli(
        "eval", "--suite", str(SUITE), "--policies", "rule,random",
        "--rounds", "2", "--out", str(out),
    )
    # `pah eval` follows `cli/main.py`'s vocabulary: 0 = every cell completed,
    # 1 = it ran but a cell did not. Both have to have written their report.
    if proc.returncode not in (0, 1) or not (out / "results.jsonl").exists():
        return _result("6 eval keeps its columns", False, f"exit {proc.returncode}: {proc.stderr.strip()[:400]}")
    summary = (out / "summary.md").read_text("utf-8")
    rows = [
        json.loads(line)
        for line in (out / "results.jsonl").read_text("utf-8").splitlines()
        if line.strip()
    ]
    # The first line is a `{"schema": ...}` manifest, not a cell; counting it would
    # report one more experiment than ran.
    cells = [row for row in rows if row.get("policy") or row.get("verdict")]
    merged_claim = "success_rate" in summary or "成功率" in summary
    ok = (
        bool(cells)
        and ("repaired" in summary)
        and ("environment_errors" in summary or "environment" in summary)
        and not merged_claim
    )
    return _result(
        "6 eval keeps its columns",
        ok,
        f"cells={len(cells)} columns_ok={ok} merged_success_rate_column={merged_claim}",
    )


# --------------------------------------------------------------------------- #
def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(REPO / "runs" / "acceptance"))
    parser.add_argument("--policy", default="rule", choices=["rule", "random", "llm"])
    args = parser.parse_args(argv[1:])

    if not INTERFERE.exists():
        print(f"missing fixture {INTERFERE}")
        return 2
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)

    checks = [
        check_runs_to_a_conclusion(root, args.policy),
        check_resume(root, args.policy),
        check_budget(root, args.policy),
        check_reproducible(root, args.policy),
        check_fault_injection(root, args.policy),
        check_eval_columns(root),
    ]
    print(f"PAH P0 acceptance · policy={args.policy} · out={root}\n")
    width = max(len(c["check"]) for c in checks)
    for check in checks:
        mark = "PASS" if check["ok"] else "FAIL"
        print(f"  {mark}  {check['check']:<{width}}  {check['detail']}")
    failed = [c for c in checks if not c["ok"]]
    print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
