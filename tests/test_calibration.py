"""Reproducible null and shift-calibration harness tests."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from topology_gate.calibration import (
    CalibrationConfig,
    calibrate_null,
    calibrate_shift,
)
from topology_gate.topology import RollingTopologyDetector, TopologyConfig


class _ThresholdDetector:
    config_identity = "test-threshold-detector:v1"

    def detect(self, observations: np.ndarray) -> SimpleNamespace:
        alarms = np.asarray(observations[:, 0] > 3.0, dtype=bool)
        return SimpleNamespace(alarms=alarms)


def _zeros(rng: np.random.Generator, horizon: int, n_features: int) -> np.ndarray:
    del rng
    return np.zeros((horizon, n_features), dtype=float)


def _shifted(rng: np.random.Generator, horizon: int, n_features: int) -> np.ndarray:
    values = rng.normal(0.0, 0.1, size=(horizon, n_features))
    values[horizon // 2 :, 0] += 4.0
    return values


def test_null_calibration_is_deterministic_and_uncertainty_is_explicit() -> None:
    config = CalibrationConfig(trials=32, horizon=16, n_features=1, seed=11)
    first = calibrate_null(_ThresholdDetector, _zeros, config=config)
    second = calibrate_null(_ThresholdDetector, _zeros, config=config)

    assert first == second
    assert first.false_alarm_count == 0
    assert first.false_alarm_rate == 0.0
    assert first.false_alarm_ci_high > 0.0
    assert first.censored_run_fraction == 1.0
    assert first.average_run_length == 17.0
    assert first.to_dict()["config_identity"] == first.config_identity


def test_shift_calibration_reports_power_and_censored_delay() -> None:
    result = calibrate_shift(
        _ThresholdDetector,
        _shifted,
        shift_index=8,
        config=CalibrationConfig(trials=32, horizon=16, n_features=1, seed=5),
    )
    assert result.detection_count == 32
    assert result.detection_rate == 1.0
    assert result.detection_ci_low > 0.8
    assert result.mean_delay_with_censoring == 0.0
    assert result.censored_fraction == 0.0


def test_actual_topology_detector_can_be_calibrated_on_a_small_null() -> None:
    def detector_factory() -> RollingTopologyDetector:
        return RollingTopologyDetector(
            TopologyConfig(
                embedding_dim=1,
                cloud_window=8,
                graph_neighbors=2,
                n_eigenvalues=2,
                min_points=4,
                calibration_window=8,
                calibration_min_periods=3,
                threshold=5.0,
            )
        )

    result = calibrate_null(
        detector_factory,
        lambda rng, horizon, features: rng.normal(
            0.0, 0.1, size=(horizon, features)
        ),
        config=CalibrationConfig(trials=3, horizon=16, n_features=1, seed=3),
    )
    assert result.trials == 3
    assert len(result.first_alarm_steps) == 3
    assert len(result.detector_identity) == 64


def test_calibration_rejects_bad_factories_and_limits() -> None:
    with pytest.raises(ValueError, match="horizon"):
        CalibrationConfig(horizon=1)
    with pytest.raises(ValueError, match="shape"):
        calibrate_null(
            _ThresholdDetector,
            lambda rng, horizon, features: np.zeros((horizon, features + 1)),
            config=CalibrationConfig(trials=1, horizon=4, n_features=1),
        )
