"""Deterministic statistical-contract tests for offline comparator metrics."""

from __future__ import annotations

from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from topology_gate.backtest import calculate_metrics  # noqa: E402


def test_one_sided_regret_is_zero_when_strategy_beats_comparator() -> None:
    metrics = calculate_metrics(
        positions=[-1.0],
        realized_returns=[-0.2],
        optimal_position=[1.0],
        periods_per_year=1,
    )

    assert metrics.absolute_comparator_discrepancy == 0.4
    assert metrics.one_sided_utility_regret == 0.0
    # Compatibility only; this is not the supported regret definition.
    assert metrics.dynamic_regret == 0.4


def test_comparator_costs_use_the_same_utility_boundary() -> None:
    metrics = calculate_metrics(
        positions=[0.0],
        realized_returns=[0.10],
        optimal_position=[1.0],
        comparator_transaction_costs=[0.05],
        periods_per_year=1,
    )

    assert metrics.absolute_comparator_discrepancy == 0.05
    assert metrics.one_sided_utility_regret == 0.05


def test_unevaluated_rows_cannot_enter_path_metrics_or_drawdown() -> None:
    metrics = calculate_metrics(
        positions=[1.0, 1.0],
        realized_returns=[100.0, 1.0],
        transaction_costs=[50.0, 0.0],
        evaluated=[False, True],
        periods_per_year=1,
    )

    assert metrics.n_evaluated == 1
    assert metrics.gross_return == 1.0
    assert metrics.net_return == 1.0
    assert metrics.total_transaction_cost == 0.0
    assert metrics.max_drawdown == 0.0


def test_expected_returns_are_the_declared_comparator_basis() -> None:
    metrics = calculate_metrics(
        positions=[1.0],
        realized_returns=[0.10],
        expected_returns=[0.20],
        optimal_position=[-1.0],
        periods_per_year=1,
    )

    # The comparator metric uses expected returns when supplied: -0.20 versus
    # +0.20.  Realized PnL remains 0.10 in the ordinary path metrics.
    assert metrics.absolute_comparator_discrepancy == 0.4
    assert metrics.one_sided_utility_regret == 0.0
    assert metrics.net_return == 0.1


def test_infeasible_or_nonfinite_comparator_is_rejected() -> None:
    with pytest.raises(ValueError, match="infeasible"):
        calculate_metrics(
            positions=[0.0],
            realized_returns=[0.1],
            optimal_position=[1.1],
        )

    with pytest.raises(ValueError, match="finite"):
        calculate_metrics(
            positions=[0.0],
            realized_returns=[0.1],
            optimal_position=[np.inf],
        )


def test_metric_and_detector_claims_are_explicitly_narrowed_in_docs() -> None:
    runbook = Path("docs/production-runbook.md").read_text(encoding="utf-8")
    architecture = Path("docs/architecture.md").read_text(encoding="utf-8")
    validity = Path("docs/statistical-validity.md").read_text(encoding="utf-8")

    combined = "\n".join((runbook, architecture, validity)).lower()
    assert "absolute comparator discrepancy" in combined
    assert "one-sided utility regret" in combined
    assert "knn" in combined
    assert "normalized-laplacian" in combined
    assert "not a persistent laplacian" in combined
    assert "exploratory" in combined
