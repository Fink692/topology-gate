"""Focused offline tests for the synthetic regime and walk-forward engine."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from topology_gate.backtest import (  # noqa: E402
    WalkForwardBacktest,
    WalkForwardConfig,
    calculate_metrics,
)
from topology_gate.synthetic import (  # noqa: E402
    SyntheticRegimeProcess,
    TimeIndexedFeatures,
    TimeIndexedLabels,
    generate_synthetic_regimes,
)


def test_synthetic_process_is_repeatable_and_exposes_known_shifts() -> None:
    process = SyntheticRegimeProcess(
        n_steps=30,
        n_features=2,
        change_points=(10, 20),
        seed=123,
        label_delay=2,
        feature_noise=0.0,
        return_noise=0.0,
    )
    first = process.generate()
    second = process.generate()

    np.testing.assert_array_equal(first.features.values, second.features.values)
    np.testing.assert_array_equal(first.labels.values, second.labels.values)
    np.testing.assert_array_equal(first.realized_returns, second.realized_returns)
    assert first.change_points == (10, 20)
    assert first.labels.available_at[9] == 11
    assert first.labels.available_at[29] == 31
    assert first.regime_ids[0] == first.regime_ids[9]
    assert first.regime_ids[10] != first.regime_ids[9]


def test_walk_forward_never_uses_unavailable_or_current_labels() -> None:
    n = 9
    features = TimeIndexedFeatures.from_array(np.arange(n, dtype=float).reshape(-1, 1))
    labels = TimeIndexedLabels.from_array(np.arange(n, dtype=float), available_at=None)
    calls = []

    def predictor(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> float:
        calls.append((x_train[:, 0].astype(int).tolist(), y_train.astype(int).tolist(), int(x_test[0, 0])))
        return 1.0

    result = WalkForwardBacktest(
        WalkForwardConfig(initial_train_size=3, label_delay=2, min_train_size=0)
    ).run(features, labels, predictor=predictor)

    assert result.training_positions[3] == (0,)
    assert result.training_positions[4] == (0, 1)
    assert result.training_positions[8] == (0, 1, 2, 3, 4, 5)
    assert all(max(train_rows, default=-1) < test_row for train_rows, _, test_row in calls)
    assert len(calls) == n - 3


def test_transaction_costs_and_metrics_are_path_based() -> None:
    features = TimeIndexedFeatures.from_array(np.zeros((4, 1)))
    labels = TimeIndexedLabels.from_array(np.ones(4))

    result = WalkForwardBacktest(
        WalkForwardConfig(
            initial_train_size=0,
            min_train_size=0,
            transaction_cost=0.01,
            periods_per_year=1,
        )
    ).run(
        features,
        labels,
        realized_returns=np.full(4, 0.1),
        predictor=lambda x_train, y_train, x_test: 1.0,
        optimal_position=np.ones(4),
        expected_returns=np.full(4, 0.1),
    )

    assert result.metrics.total_transaction_cost == 0.01
    assert result.metrics.turnover == 1.0
    np.testing.assert_allclose(result.net_returns, [0.09, 0.1, 0.1, 0.1])
    assert result.metrics.dynamic_regret == 0.01
    assert result.metrics.max_drawdown == 0.0

    direct = calculate_metrics(
        [1.0, -1.0, 1.0],
        [0.2, -0.2, 0.2],
        labels=[1.0, -1.0, 1.0],
        optimal_position=[1.0, 1.0, 1.0],
        periods_per_year=1,
    )
    assert direct.information_coefficient > 0.9
    assert direct.dynamic_regret == 0.4


def test_known_shift_detection_and_baseline_false_promotion() -> None:
    dataset = generate_synthetic_regimes(
        n_steps=12,
        n_features=1,
        change_points=(6,),
        seed=5,
        label_delay=1,
        feature_noise=0.0,
        return_noise=0.0,
        return_magnitude=0.1,
    )
    result = WalkForwardBacktest(
        WalkForwardConfig(initial_train_size=0, min_train_size=0, promotion_window=3)
    ).run(
        dataset,
        realized_returns=np.full(12, 0.1),
        expected_returns=np.full(12, 0.1),
        predictor=lambda x_train, y_train, x_test: 1.0 if x_test[0, 0] > 0 else -1.0,
        baseline_hooks={"opposite": lambda row: -1.0 if row[0, 0] > 0 else 1.0},
    )

    assert result.metrics.detection_delays == (0,)
    assert result.baseline_comparisons["opposite"].promoted
    assert result.baseline_comparisons["opposite"].false_promotion
    assert result.metrics.false_promotions == 1
