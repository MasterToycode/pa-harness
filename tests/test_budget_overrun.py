"""Budget overspend has to be visible, not clamped to zero.

The ledger's semantics are deliberate: `close_round` records what a round *actually*
cost and never refuses it, and the overspend shows up on the **next** `begin_round`.
That is the right shape for a repair loop — cancelling a round mid-way would throw
away evidence already paid for — but it makes the report the only place a person can
see that the budget was exceeded at all.

Before the fix (audit P0-5), the report clamped `left` to 0 and said nothing else, so
"spent exactly the last call" and "spent three more than it had" produced identical
numbers. Equal-budget comparisons across policies then silently included cells that
had overspent.
"""

from __future__ import annotations

from pah.app.runtime import budget_block
from pah.core.model import Budget, BudgetUsed


def test_a_run_within_budget_reports_no_overrun():
    block = budget_block(
        Budget(rounds=3, geom_calls=10, sim_calls=2),
        BudgetUsed(rounds=2, geom_calls=7, sim_calls=1),
    )

    assert block["geom_calls"]["left"] == 3
    assert block["geom_calls"]["overrun"] == 0
    assert block["overrun_total"] == 0


def test_an_overspent_dimension_reports_zero_left_and_a_positive_overrun():
    block = budget_block(
        Budget(rounds=3, geom_calls=10, sim_calls=2),
        BudgetUsed(rounds=3, geom_calls=13, sim_calls=2),
    )

    assert block["geom_calls"]["left"] == 0  # clamped, as before
    assert block["geom_calls"]["overrun"] == 3  # and the truth, which was missing
    assert block["overrun_total"] == 3
    # The dimensions that stayed inside still read clean.
    assert block["rounds"]["overrun"] == 0
    assert block["sim_calls"]["overrun"] == 0


def test_overrun_total_sums_every_dimension():
    block = budget_block(
        Budget(rounds=1, geom_calls=1, sim_calls=1),
        BudgetUsed(rounds=2, geom_calls=4, sim_calls=3),
    )

    assert block["overrun_total"] == 1 + 3 + 2


def test_the_eval_row_carries_the_overrun_columns():
    """Equal-budget cells are only comparable when overspend can be filtered out."""
    import inspect

    from pah.eval import runner

    source = inspect.getsource(runner)
    for column in ("budget_overrun_total", "geom_overrun", "sim_overrun"):
        assert column in source, f"eval row lost the {column} column"
