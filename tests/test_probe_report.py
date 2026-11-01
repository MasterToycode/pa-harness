"""The probe's report arithmetic — the numbers §4.8 fixes, without the geometry.

`pah probe connector` needs 705 edges of real meshes and ~17 minutes; the
*reporting* needs only the rows it already wrote. `_report` is pure once rows
exist, so the three numbers this milestone is measured on are pinned here:

- `success_original` — the existing definition, untouched;
- `success_ivr_validated ∈ [lo, hi]` — `unknown` counted as failed for `lo` and as
  validated for `hi`, never folded into one endpoint;
- `n_unknown` — the boolean engine's silence, reported on its own.

Plus the two counters that make an environment failure visible instead of
letting it disappear into `bool_failed`: `retried_n` and `oom_n`.
"""

from __future__ import annotations

import json

import pytest

from pah.cli import probe
from pah.tools.verifiers.connector_feasible import Case

CASE = "asm_000000"

#: Distinguishes "this row carries no IVR" from a measured `0.0`, which is a real
#: and expected value (two solids that only touch read exactly zero).
_NOT_MEASURED = object()


def _row(
    edge_id: int,
    *,
    ok: bool,
    code: str = "",
    ivr=_NOT_MEASURED,
    method: str = "bbox",
    retried: int = 0,
    oom: int = 0,
) -> dict:
    """One `rows.jsonl` record. `ivr=None` means unknown; omitted means not measured."""
    row = {
        "case_id": CASE,
        "edge_id": edge_id,
        "a": 0,
        "b": 1,
        "ok": ok,
        "reason": "" if ok else "bool_failed",
        "detail": "",
        "code": code,
        "retried_n": retried,
        "oom": oom,
    }
    if ivr is not _NOT_MEASURED:
        row["ivr_value"] = ivr
        row["ivr_method"] = method
    return row


def _report(rows: list[dict], **overrides) -> dict:
    selected = [
        Case(case_id=CASE, joints=tuple({"edge_id": row["edge_id"]} for row in rows))
    ]
    return probe._report(
        rows,
        overrides.pop("edges_total", len(rows)),
        selected,
        overrides.pop("split", "feasible"),
        "scale",
        "edges_v2.json",
        overrides.pop("gold", None),
        1.0,
        limited=overrides.pop("limited", False),
        **overrides,
    )


# --------------------------------------------------------------------------- #
# the three numbers
# --------------------------------------------------------------------------- #
def test_unknown_widens_the_interval_instead_of_being_decided():
    """2 measured (one clean, one interpenetrating) + 1 unknown -> [1, 2]."""
    rows = [
        _row(1, ok=True, ivr=0.0),
        _row(2, ok=True, ivr=0.5),  # successful but interpenetrating -> not validated
        _row(3, ok=True, ivr=None),  # boolean could not decide -> unknown
        _row(4, ok=False, code="verifier_oom"),
        _row(5, ok=False, code="bool_failed"),
    ]

    report = _report(rows)

    assert report["success"] == 3
    assert report["success_original"] == 3
    assert report["success_ivr_validated"] == {"lo": 1, "hi": 2}
    assert report["n_unknown"] == 1
    assert report["ivr"]["known"] == 2
    assert report["ivr"]["over_eps"] == 1
    assert report["ivr"]["missing"] == 0
    # `lo` never silently counts the unknown as a pass, and `hi` never drops it.
    assert report["success_ivr_validated"]["lo"] < report["success_ivr_validated"]["hi"]


def test_eps_is_the_boundary_and_inclusive():
    """`ivr_exact == ε` is validated; a hair above is not — the comparison is `<=`."""
    rows = [
        _row(1, ok=True, ivr=probe.IVR_EPS),
        _row(2, ok=True, ivr=probe.IVR_EPS * 1.000001),
    ]

    report = _report(rows)

    assert report["success_ivr_validated"] == {"lo": 1, "hi": 1}
    assert report["n_unknown"] == 0
    assert report["ivr"]["over_eps"] == 1


def test_a_success_without_ivr_is_missing_not_unknown():
    """A row written before the probe measured IVR must not read as "unknown"."""
    rows = [
        _row(1, ok=True, ivr=0.0),
        _row(2, ok=True),  # no `ivr_value` key at all
        _row(3, ok=True, ivr=None),
    ]

    report = _report(rows)

    assert report["success_original"] == 3
    assert report["ivr"]["checked"] == 2
    assert report["ivr"]["missing"] == 1
    assert report["n_unknown"] == 1, "not-measured is not the same as could-not-decide"
    assert report["success_ivr_validated"] == {"lo": 1, "hi": 2}


def test_every_success_measured_is_a_closed_interval():
    """The M0 shape: all successes measured, no unknown -> `lo == hi`."""
    rows = [_row(edge, ok=True, ivr=0.0) for edge in range(1, 4)]

    report = _report(rows)

    assert report["success_ivr_validated"] == {"lo": 3, "hi": 3}
    assert report["n_unknown"] == 0
    assert report["ivr"]["known"] == 3


# --------------------------------------------------------------------------- #
# retries and OOMs are counted, not swallowed
# --------------------------------------------------------------------------- #
def test_retry_counters_survive_into_the_report():
    rows = [
        _row(1, ok=True, ivr=0.0, retried=1),
        _row(2, ok=False, code="verifier_oom", retried=2, oom=1),
        _row(3, ok=False, code="bool_failed"),
    ]

    report = _report(rows)

    assert report["retried_n"] == 3, "retry *attempts*, not edges retried"
    assert report["oom_n"] == 1


def test_oom_and_bool_failed_are_two_buckets():
    """An environment failure and a geometry failure must never merge."""
    rows = [
        _row(1, ok=False, code="verifier_oom", oom=1),
        _row(2, ok=False, code="bool_failed"),
        _row(3, ok=False, code="bool_failed"),
    ]

    report = _report(rows)

    assert report["codes"] == {"bool_failed": 2, "verifier_oom": 1}
    assert report["oom_n"] == 1


def test_old_rows_without_the_new_keys_still_report():
    """Resuming over rows written before this change must not crash the report."""
    rows = [
        {"case_id": CASE, "edge_id": 1, "ok": True, "reason": "", "detail": "", "code": ""},
        {"case_id": CASE, "edge_id": 2, "ok": False, "reason": "bool_failed",
         "detail": "union_status(NoError_Degenerate,faces=0)", "code": "bool_failed"},
    ]

    report = _report(rows)

    assert report["success_original"] == 1
    assert report["retried_n"] == 0
    assert report["oom_n"] == 0
    assert report["ivr"]["missing"] == 1


# --------------------------------------------------------------------------- #
# the markdown a reader actually opens
# --------------------------------------------------------------------------- #
def test_markdown_prints_the_three_numbers_and_both_buckets():
    rows = [
        _row(1, ok=True, ivr=0.0, method="bbox"),
        _row(2, ok=True, ivr=None, method="manifold"),
        _row(3, ok=False, code="verifier_oom", oom=1, retried=2),
        _row(4, ok=False, code="bool_failed"),
    ]

    text = probe._markdown(_report(rows))

    assert "`success_original` | 2" in text
    assert "[1, 2]" in text, "the interval must be printed as an interval"
    assert "`n_unknown` | 1" in text
    assert "`retried_n`: 2" in text
    assert "`oom_n`: 1" in text
    # a proved zero and a measured zero are not the same evidence
    assert "how each reading was produced: `bbox` 1, `manifold` 1" in text
    # both codes get their own line; neither is renamed into the other
    assert "- `bool_failed`: 1" in text
    assert "- `verifier_oom`: 1" in text
    json.dumps(text)  # the markdown is a plain string, not a lazy object


def test_markdown_flags_a_coverage_gap():
    """A partially measured IVR pass must say so where a reader will see it."""
    rows = [_row(1, ok=True, ivr=0.0), _row(2, ok=True)]

    text = probe._markdown(_report(rows))

    assert "coverage gap" in text


@pytest.mark.parametrize("missing", [False, True])
def test_run_json_is_serialisable_whatever_the_ivr_outcome(missing):
    """`run.json` is written with `json.dumps`; a raw numpy/None value would break it."""
    rows = [_row(1, ok=True, ivr=None if missing else 0.0)]

    report = _report(rows)

    assert json.loads(json.dumps(report, ensure_ascii=False))["n_unknown"] == int(missing)
