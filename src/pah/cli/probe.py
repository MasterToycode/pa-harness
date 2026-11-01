"""`pah probe`: run one verifier across the benchmark and stream the rows to disk.

The shape is copied from the source benches (§2.4: "流式载入/增量写盘/断点续跑
的做法直接抄") — worth copying because they are the difference between a 20-minute
run that survives a Ctrl-C and one that does not.

Two rules this module keeps:

* Geometry loads one assembly at a time. 724 undecimated parts do not fit in
  memory (`tests/asm_final.py:22`), so the unit of work is the assembly, not the
  edge, and `--cases N` counts assemblies for that reason.
* The rows land in `rows.jsonl` as they arrive and are re-read on the next run,
  so a resumed probe adds rows instead of starting over.

A row carries more than the connector verdict, because §4.8 refuses a baseline
with the holes hidden: `retried_n` / `oom` record what the machine cost, and a
successful edge also carries the `ivr_exact` of its part pair. The report turns
those into the three numbers §4.8 requires — `success_original`,
`success_ivr_validated ∈ [lo, hi]` and `n_unknown` — and keeps `verifier_oom`
counted apart from `bool_failed`.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from pah.tools.ivr import ivr_exact
from pah.tools.verifiers import default_registry
from pah.tools.verifiers.connector_feasible import (
    BASELINE,
    EDGE_FILE,
    GEOMETRY_ROOT,
    RL_REPO,
    Case,
    case_parts,
    certificates_for,
    load_cases,
    iter_joint_results,
    oom_stats,
)

__all__ = ["DEFAULT_GOLD", "IVR_EPS", "run"]

#: The frozen per-edge verdicts the gold standard was computed from. Read-only;
#: absent is fine, it only costs the edge-by-edge diff.
DEFAULT_GOLD = RL_REPO / "code" / "assembly_rl" / "tests" / "_final_rows_Nnew_Tnew.json"

#: `ivr_exact <= IVR_EPS` is what "IVR-validated" means (`pah-dev-doc-v1.md` §4.8).
#: ε is evaluation config, not a policy knob: 1e-6 of a pair's total solid volume
#: is orders of magnitude above boolean noise on closed parts and well below any
#: designed interference fit, so it answers "did these two solids interpenetrate"
#: rather than "did they touch".
IVR_EPS = 1e-6

_PROGRESS_EVERY = 5


def run(
    *,
    out: str | Path,
    split: str = "feasible",
    cases: int | None = None,
    jobs: int = 8,
    unit_mm_rule: str = "scale",
    edge_file: str | Path = EDGE_FILE,
    gold: str | Path | None = DEFAULT_GOLD,
    quiet: bool = False,
) -> dict:
    """Runs the connector probe and returns the comparison report."""
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "rows.jsonl"
    edge_path = Path(edge_file)
    if not edge_path.exists():
        raise FileNotFoundError(f"edge file not found: {edge_path}")
    if not GEOMETRY_ROOT.exists():
        raise FileNotFoundError(f"geometry root not found: {GEOMETRY_ROOT}")

    everything = json.loads(edge_path.read_text(encoding="utf-8"))
    selected = load_cases(edge_path, split=split, limit=cases)
    _assert_same_protocol(out_dir, split, unit_mm_rule, edge_path)
    done = _done_rows(rows_path)
    done_edges = {(r["case_id"], r["edge_id"]) for r in done}

    todo = [c for c in selected if any((c.case_id, j["edge_id"]) not in done_edges for j in c.joints)]
    total_edges = sum(len(c.joints) for c in selected)
    if not quiet:
        print(
            f"edges: {len(everything['edges'])} total, "
            f"{total_edges} in split={split}, {len(selected)} assemblies",
            flush=True,
        )
        print(
            f"geometry: {GEOMETRY_ROOT}\nunit_mm rule: {unit_mm_rule}",
            flush=True,
        )
        if done:
            print(f"resume: {len(done)} rows already on disk, {len(todo)} assemblies to go", flush=True)

    started = time.time()
    with rows_path.open("a", encoding="utf-8") as sink:
        if jobs <= 1:
            _run_serial(todo, sink, done, unit_mm_rule, edge_path, quiet)
        else:
            _run_pool(todo, sink, done, unit_mm_rule, edge_path, jobs, quiet)

    rows = _done_rows(rows_path)
    report = _report(
        rows, len(everything["edges"]), selected, split, unit_mm_rule, edge_path, gold,
        _spent(time.time() - started, todo, out_dir), limited=cases is not None,
    )
    report["report_only"] = not todo
    (out_dir / "run.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    (out_dir / "report.md").write_text(_markdown(report), encoding="utf-8")
    return report


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #
def _worker(payload: dict) -> list[dict]:
    """One assembly, every selected edge. Module-level so spawn can pickle it."""
    case: Case = payload["case"]
    rule: str = payload["unit_mm_rule"]
    edge_file = payload["edge_file"]
    # The IVR pass reads the meshes the boolean pass already loaded: same objects,
    # same cache, so this costs no extra resident geometry.
    parts, _longest = case_parts(case)
    ivr_cache: dict[str, dict] = {}
    rows: list[dict] = []
    before = oom_stats()
    for joint, result, unit_mm in iter_joint_results(case, unit_mm_rule=rule):
        certificates = certificates_for(joint, result, unit_mm, edge_file=edge_file)
        row = _row(case, joint, result, unit_mm, certificates)
        after = oom_stats()
        row["retried_n"] = after["retried"] - before["retried"]
        row["oom"] = after["oom"] - before["oom"]
        before = after
        if row["ok"]:
            row.update(_ivr_fields(case, parts, joint, ivr_cache))
        rows.append(row)
    return rows


def _ivr_fields(case: Case, parts: tuple, joint: dict, cache: dict) -> dict:
    """`ivr_exact` for the part pair a successful edge sits on (§4.8: exact level).

    Only successful edges are measured — a failed edge has no assembly to
    validate, so spending 55 ms-4 s on it would buy nothing. The cache key is the
    two part ids sorted and joined, so a second edge between the same parts in
    the same assembly is one measurement, not two; nothing about the key depends
    on edge order or on which id came first.
    """
    a_index, b_index = joint["a_index"], joint["b_index"]
    a_id = str(case.parts[a_index]["id"])
    b_id = str(case.parts[b_index]["id"])
    key = "|".join(sorted((a_id, b_id)))
    if key not in cache:
        try:
            reading = ivr_exact(
                [(a_id, parts[a_index]), (b_id, parts[b_index])],
                pairs=[(a_id, b_id)],
            )
            unknown = "" if reading.value is not None else "; ".join(reading.unknown_pairs)
            # "bbox" and "manifold" are different claims: an AABB miss *proves* an
            # empty intersection, the boolean *measures* it (`ivr.py`), and a
            # reader deserves to know which produced the zero.
            method = str((reading.detail.get("pairs") or [{}])[0].get("method") or "")
        except Exception as exc:  # noqa: BLE001 - §4.8 rule 2: unmeasurable is unknown
            reading, unknown, method = None, f"adapter:{type(exc).__name__}: {exc}", ""
        cache[key] = {
            # `None` means unknown, never "no overlap" — `ivr.py`'s contract.
            "ivr_value": None if reading is None else reading.value,
            "ivr_method": method,
            "ivr_unknown": unknown,
        }
    return dict(cache[key])


def _row(case: Case, joint: dict, result: dict, unit_mm: float, certificates: list) -> dict:
    return {
        "case_id": case.case_id,
        "edge_id": joint["edge_id"],
        "a": joint["a_index"],
        "b": joint["b_index"],
        "ok": bool(result.get("ok")),
        "reason": str(result.get("reason") or ""),
        "detail": str(result.get("detail") or ""),
        "degen_code": str(result.get("degen_code") or ""),
        "code": certificates[0].code if certificates else "",
        "unit_mm": round(float(unit_mm), 6),
        "r_mm": _scale(result.get("r"), unit_mm),
        "depth_mm": _scale(result.get("depth"), unit_mm),
        "feasible_file": joint["feasible"],
        "thickness_failed": joint["thickness_failed"],
        "limiter": joint["limiter"],
    }


def _scale(value, unit_mm: float) -> float | None:
    return None if value is None else round(float(value) * unit_mm, 6)


def _run_serial(todo, sink, done, rule, edge_path, quiet) -> None:
    for index, case in enumerate(todo, start=1):
        for row in _worker(
            {"case": case, "unit_mm_rule": rule, "edge_file": str(edge_path)}
        ):
            _emit(sink, row, done)
        _tick(index, len(todo), case, quiet)


def _run_pool(todo, sink, done, rule, edge_path, jobs, quiet) -> None:
    if not todo:
        return
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = {
            pool.submit(
                _worker,
                {"case": case, "unit_mm_rule": rule, "edge_file": str(edge_path)},
            ): case
            for case in todo
        }
        for index, future in enumerate(as_completed(futures), start=1):
            for row in future.result():
                _emit(sink, row, done)
            _tick(index, len(todo), futures[future], quiet)


def _emit(sink, row: dict, done) -> None:
    """Appends one row unless the resume set already holds it."""
    if (row["case_id"], row["edge_id"]) in done:
        return
    sink.write(json.dumps(row, ensure_ascii=False) + "\n")
    sink.flush()
    os.fsync(sink.fileno())


def _tick(index: int, total: int, case: Case, quiet: bool) -> None:
    if quiet or index % _PROGRESS_EVERY and index != total:
        return
    print(f"  [{index:3d}/{total}] {case.case_id} ({len(case.joints)} edges)", flush=True)


def _done_rows(rows_path: Path) -> list[dict]:
    """Every row already on disk; a torn last line is dropped, not fatal."""
    if not rows_path.exists():
        return []
    rows = []
    for line in rows_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


_PROTOCOL_KEYS = ("split", "unit_mm_rule", "edge_file")


def _assert_same_protocol(
    out_dir: Path, split: str, rule: str, edge_file: Path
) -> None:
    """Refuses to resume into rows measured under a different scale protocol.

    Resuming is what makes a 25-minute probe repeatable; resuming across a
    `--unit-mm` change would blend two scales into one `rows.jsonl` and produce a
    number nobody could attribute. The guard reads whichever marker already
    exists — `probe.json` from a run in flight, `run.json` from a finished one.
    """
    current = {"split": split, "unit_mm_rule": rule, "edge_file": str(edge_file)}
    for name in ("probe.json", "run.json"):
        marker = out_dir / name
        if not marker.exists():
            continue
        try:
            previous = json.loads(marker.read_text(encoding="utf-8"))
        except ValueError:
            continue
        previous = {key: previous.get(key) for key in _PROTOCOL_KEYS}
        if previous != current:
            raise ValueError(
                f"{out_dir} holds rows measured under {previous}; refusing to add "
                f"{current}. Pick a new --out directory."
            )
        return
    (out_dir / "probe.json").write_text(
        json.dumps(current, ensure_ascii=False, indent=1), encoding="utf-8"
    )


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #
def _oom_totals(rows: list[dict]) -> tuple[int, int]:
    """`(retried_n, oom_n)` over `rows`.

    `retried_n` counts retry attempts — one edge retried twice contributes two —
    and `oom_n` counts edges that ended `verifier_oom`. Both are read from the
    rows, so a run resumed over older rows reports what it actually has instead
    of a zero nobody measured.
    """
    retried = sum(int(r.get("retried_n") or 0) for r in rows)
    oom = sum(int(r.get("oom") or 0) for r in rows)
    return retried, oom


def _ivr_summary(rows: list[dict]) -> dict:
    """The §4.8 report numbers, from the `ivr_exact` each successful edge measured.

        lo = success_original ∧ (ivr_exact ≤ ε)                      # unknown -> fail
        hi = success_original ∧ (ivr_exact ≤ ε  ∨  ivr is unknown)   # unknown -> pass

    `n_unknown` is reported on its own and folded into neither endpoint: the
    interval *is* the answer, and its width is a finding about the geometry
    toolchain, not a number to round away. A success row carrying no IVR field at
    all (written before this probe measured IVR) counts in `missing`, not as
    unknown — "not measured" and "could not be decided" are different claims.
    """
    successes = [r for r in rows if r["ok"]]
    measured = [r for r in successes if "ivr_value" in r]
    known = [r["ivr_value"] for r in measured if r["ivr_value"] is not None]
    n_unknown = len(measured) - len(known)
    lo = sum(1 for value in known if value <= IVR_EPS)
    methods: dict[str, int] = {}
    for row in measured:
        name = str(row.get("ivr_method") or "unknown")
        methods[name] = methods.get(name, 0) + 1
    return {
        "level": "exact",
        "field": "ivr_exact",
        "eps": IVR_EPS,
        "checked": len(measured),
        "missing": len(successes) - len(measured),
        "known": len(known),
        "over_eps": len(known) - lo,
        "methods": dict(sorted(methods.items(), key=lambda kv: -kv[1])),
        "n_unknown": n_unknown,
        "lo": lo,
        "hi": lo + n_unknown,
    }


def _spent(elapsed: float, todo: list, out_dir: Path) -> float:
    """Wall clock of the pass that measured the rows, not of a report-only re-read.

    Reopening a finished run directory to regenerate `report.md` takes a second;
    reporting that as the run's cost would misstate how expensive the verifier is.
    """
    if todo:
        return elapsed
    try:
        prior = json.loads((out_dir / "run.json").read_text(encoding="utf-8"))
        return float(prior.get("seconds") or elapsed)
    except (OSError, ValueError):
        return elapsed


def _report(
    rows, edges_total, selected, split, rule, edge_path, gold, elapsed, *, limited: bool
) -> dict:
    wanted = {(c.case_id, j["edge_id"]) for c in selected for j in c.joints}
    ran = [r for r in rows if (r["case_id"], r["edge_id"]) in wanted]

    success = sum(1 for r in ran if r["ok"])
    unreatable = sum(1 for r in ran if r["reason"] == "unreatable_too_small")
    host_bad = sum(1 for r in ran if r["reason"] == "host_geometry_unreliable")
    codes: dict[str, int] = {}
    for row in ran:
        if not row["ok"]:
            codes[row["code"]] = codes.get(row["code"], 0) + 1

    denominator = len(wanted)
    rate = success / denominator if denominator else 0.0
    retried_n, oom_n = _oom_totals(ran)
    ivr = _ivr_summary(ran)
    gold_cmp = {
        "feasible": BASELINE["feasible_edges"],
        "success": BASELINE["success"],
        "rate": BASELINE["rate"],
    }
    delta = {
        "feasible": denominator - gold_cmp["feasible"],
        "success": success - gold_cmp["success"],
        "rate": round(rate - gold_cmp["rate"], 6),
    }
    complete = len(ran) == denominator
    # The gold columns only describe the *whole* feasible split; a `--cases N`
    # run is a link check, and calling its rate a regression would be a lie.
    gold_scope = split == "feasible" and not limited
    edge_diff = _edge_diff(ran, gold)

    report = {
        "probe": "connector",
        "split": split,
        "scope": "gold" if gold_scope else "subset",
        "unit_mm_rule": rule,
        "edge_file": str(edge_path),
        "geometry_root": str(GEOMETRY_ROOT),
        "edges_total": edges_total,
        "edges_in_split": denominator,
        "assemblies": len(selected),
        "ran": len(ran),
        "complete": complete,
        "success": success,
        "rate": round(rate, 6),
        "unreatable_too_small": unreatable,
        "host_geometry_unreliable": host_bad,
        "adapter_feasible": len(ran) - unreatable - host_bad,
        "codes": dict(sorted(codes.items(), key=lambda kv: -kv[1])),
        "success_original": success,
        "success_ivr_validated": {"lo": ivr["lo"], "hi": ivr["hi"]},
        "n_unknown": ivr["n_unknown"],
        "ivr": {key: value for key, value in ivr.items() if key not in ("lo", "hi", "n_unknown")},
        "retried_n": retried_n,
        "oom_n": oom_n,
        "gold": gold_cmp,
        "delta": delta,
        "edge_diff": edge_diff,
        "seconds": round(elapsed, 1),
    }
    report.update(_verdict(report, edge_diff))
    report["reproduced"] = report["verdict"] in (
        "reproduced",
        "reproduced-modulo-environment",
    )
    return report


def _edge_diff(rows: list[dict], gold: str | Path | None) -> dict:
    """Edge-by-edge agreement with the frozen gold rows, when they are readable."""
    if gold is None:
        return {}
    path = Path(gold)
    if not path.exists():
        return {"available": False}
    try:
        blocked = json.loads(path.read_text(encoding="utf-8"))["rows"]
    except (OSError, ValueError, KeyError):
        return {"available": False}
    reference = {
        (r["asm"], int(r["eid"])): r for r in blocked if "eid" in r
    }
    mismatches = []
    compared = 0
    for row in rows:
        key = (row["case_id"], row["edge_id"])
        if key not in reference:
            continue
        compared += 1
        if bool(reference[key]["ok"]) != row["ok"]:
            mismatches.append(
                {
                    "case_id": row["case_id"],
                    "edge_id": row["edge_id"],
                    "adapter": "ok" if row["ok"] else row["reason"],
                    "gold": "ok" if reference[key]["ok"] else "fail",
                    "gold_detail": str(reference[key].get("detail") or ""),
                }
            )
    return {
        "available": True,
        "compared": compared,
        "mismatches": len(mismatches),
        "details": mismatches,
        "examples": mismatches[:10],
    }


#: Gold failures the source itself attributes to the machine, not to geometry
#: (`connector-boolean-report.md` §19.5.1: "属环境问题而非几何问题"). Reproducing
#: such an edge is not a disagreement about the algorithm, so the verdict says so
#: instead of pretending the run missed the baseline.
_ENVIRONMENT_PREFIXES = ("exception:MemoryError",)


def _is_environment_failure(detail: str) -> bool:
    return detail.startswith(_ENVIRONMENT_PREFIXES)


def _verdict(report: dict, edge_diff: dict) -> dict:
    """`reproduced` / `reproduced-modulo-environment` / `mismatch` / `partial`."""
    if not report["complete"]:
        return {"verdict": "partial", "adjusted_success": report["success"]}
    if report["scope"] != "gold":
        return {"verdict": "subset", "adjusted_success": report["success"]}
    details = edge_diff.get("details") or []
    # A success this run got where gold recorded a machine failure is not a
    # better adapter, it is a bigger machine.
    environment = [m for m in details if m["adapter"] == "ok"
                   and _is_environment_failure(m["gold_detail"])]
    adjusted = report["success"] - len(environment)
    exact = not details and report["delta"] == {"feasible": 0, "success": 0, "rate": 0.0}
    modulo = (
        not exact
        and report["delta"]["feasible"] == 0
        and len(environment) == len(details)
        and adjusted == report["gold"]["success"]
    )
    return {
        "verdict": "reproduced" if exact else ("reproduced-modulo-environment" if modulo else "mismatch"),
        "adjusted_success": adjusted,
        "environment_edges": len(environment),
    }


def _markdown(report: dict) -> str:
    """The acceptance table, as the run directory's own summary."""
    gold = report["gold"]
    diff = report["edge_diff"]
    lines = [
        "# M0 connector probe",
        "",
        f"- edges file: `{report['edge_file']}`",
        f"- geometry: `{report['geometry_root']}`",
        f"- split: `{report['split']}`, scope: `{report['scope']}`, "
        f"assemblies: {report['assemblies']}, unit_mm rule: `{report['unit_mm_rule']}`",
        f"- wall clock: {report['seconds']} s",
        "",
        "| quantity | probe | gold | delta |",
        "|---|---:|---:|---:|",
        f"| edges_total | {report['edges_total']} | — | — |",
        f"| feasible | {report['edges_in_split']} | {gold['feasible']} | {report['delta']['feasible']:+d} |",
        f"| success | {report['success']} | {gold['success']} | {report['delta']['success']:+d} |",
        f"| rate | {report['rate']:.1%} | {gold['rate']:.1%} | {report['delta']['rate']:+.1%} |",
        "",
        f"**{report['verdict'].upper()}**"
        f" ({report['ran']}/{report['edges_in_split']} edges ran)",
        "",
        f"- `unreatable_too_small`: {report['unreatable_too_small']} (gold 29)",
        f"- adapter-feasible: {report['adapter_feasible']}",
    ]
    if report.get("environment_edges"):
        lines.append(
            f"- environment-dependent edges (gold lost them to `MemoryError`, this "
            f"run did not): {report['environment_edges']} -> adjusted success "
            f"{report['adjusted_success']}"
        )
    lines += [
        "",
        "## failure codes",
        "",
    ]
    for code, count in report["codes"].items():
        lines.append(f"- `{code}`: {count}")
    lines += [
        "",
        "## `MemoryError` retries",
        "",
        f"- `retried_n`: {report['retried_n']} (retry attempts made after an "
        f"out-of-memory verdict)",
        f"- `oom_n`: {report['oom_n']} (edges still recorded `verifier_oom`)",
        f"- `verifier_oom` failures: {report['codes'].get('verifier_oom', 0)} — "
        f"counted apart from `bool_failed`, never merged: one is the machine, "
        f"the other is the geometry",
    ]
    ivr = report["ivr"]
    coverage = ""
    if ivr["missing"]:
        coverage = (
            f"\n- **coverage gap**: {ivr['missing']} success row(s) carry no IVR "
            f"— they predate this measurement and are excluded from `[lo, hi]`"
        )
    lines += [
        "",
        f"## IVR (`{ivr['field']}`, ε = {ivr['eps']:g})",
        "",
        "| quantity | value |",
        "|---|---:|",
        f"| `success_original` | {report['success_original']} |",
        f"| `success_ivr_validated` ∈ [`lo`, `hi`] | "
        f"[{report['success_ivr_validated']['lo']}, {report['success_ivr_validated']['hi']}] |",
        f"| `n_unknown` | {report['n_unknown']} |",
        f"| measured (`ivr` known) | {ivr['known']} |",
        f"| over ε (success but interpenetrating) | {ivr['over_eps']} |",
        "",
        "how each reading was produced: "
        + ", ".join(f"`{name}` {count}" for name, count in ivr["methods"].items()),
        "",
        "`lo` counts `unknown` as not IVR-validated, `hi` counts it as validated; "
        "the interval is the answer and `n_unknown` says how much of it is the "
        "boolean engine's silence rather than a measurement (§4.8)." + coverage,
    ]
    if diff.get("available"):
        lines += [
            "",
            "## edge-by-edge vs the frozen gold rows",
            "",
            f"- compared: {diff['compared']}, mismatches: {diff['mismatches']}",
        ]
        for example in diff["examples"]:
            lines.append(
                f"- `{example['case_id']}:{example['edge_id']}` "
                f"adapter={example['adapter']} gold={example['gold']} "
                f"`{example['gold_detail'][:60]}`"
            )
    elif diff:
        lines += ["", "_gold rows not readable; edge-by-edge diff skipped_"]
    return "\n".join(lines) + "\n"
