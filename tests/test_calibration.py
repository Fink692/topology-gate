"""Reproducible null and shift-calibration harness tests."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from topology_gate.calibration import (
    CalibrationCertificate,
    CalibrationConfig,
    EProcessCalibrationConfig,
    PromotionCalibrationConfig,
    StationaryBlockBootstrap,
    ThresholdCalibrationResult,
    calibrate_eprocess_null,
    calibrate_null,
    calibrate_promotion_null,
    calibrate_shift,
    calibrate_threshold,
)
from topology_gate.topology import RollingTopologyDetector, TopologyConfig


class _ThresholdDetector:
    config_identity = "test-threshold-detector:v1"

    def detect(self, observations: np.ndarray) -> SimpleNamespace:
        alarms = np.asarray(observations[:, 0] > 3.0, dtype=bool)
        return SimpleNamespace(alarms=alarms)


class _ParameterizedThresholdDetector:
    def __init__(self, threshold: float) -> None:
        self.threshold = threshold
        self.config_identity = f"parameterized-threshold:{threshold:.6f}"

    def detect(self, observations: np.ndarray) -> SimpleNamespace:
        alarms = np.asarray(observations[:, 0] > self.threshold, dtype=bool)
        return SimpleNamespace(alarms=alarms)


class _ConstantPointFactory:
    def __init__(self, split: str) -> None:
        self.identity = f"constant-point:{split}:v1"

    def __call__(
        self, rng: np.random.Generator, horizon: int, n_features: int
    ) -> np.ndarray:
        del rng
        return np.full((horizon, n_features), 0.75, dtype=float)


class _RademacherScores:
    identity = "rademacher-score-null:v1"

    def __call__(self, rng: np.random.Generator, horizon: int) -> np.ndarray:
        return np.where(rng.integers(0, 2, size=horizon) == 0, -1.0, 1.0)


class _RademacherPromotionScores:
    identity = "rademacher-promotion-score-null:v1"

    def __call__(
        self,
        rng: np.random.Generator,
        horizon: int,
        challengers: int,
    ) -> np.ndarray:
        return np.where(
            rng.integers(0, 2, size=(horizon, challengers)) == 0,
            -1.0,
            1.0,
        )


class _AllPositivePromotionScores:
    identity = "all-positive-promotion-score:v1"

    def __call__(
        self,
        rng: np.random.Generator,
        horizon: int,
        challengers: int,
    ) -> np.ndarray:
        del rng
        return np.ones((horizon, challengers), dtype=float)


class _AllNegativePromotionScores:
    identity = "all-negative-promotion-score-null:v1"

    def __call__(
        self,
        rng: np.random.Generator,
        horizon: int,
        challengers: int,
    ) -> np.ndarray:
        del rng
        return -np.ones((horizon, challengers), dtype=float)


class _SecondEpochPositivePromotionScores:
    identity = "second-epoch-positive-promotion-score:v1"

    def __call__(
        self,
        rng: np.random.Generator,
        horizon: int,
        challengers: int,
    ) -> np.ndarray:
        del rng
        return np.stack(
            (
                -np.ones((horizon, challengers), dtype=float),
                np.ones((horizon, challengers), dtype=float),
            )
        )


def _zeros(rng: np.random.Generator, horizon: int, n_features: int) -> np.ndarray:
    del rng
    return np.zeros((horizon, n_features), dtype=float)


def _shifted(rng: np.random.Generator, horizon: int, n_features: int) -> np.ndarray:
    values = rng.normal(0.0, 0.1, size=(horizon, n_features))
    values[horizon // 2 :, 0] += 4.0
    return values


def _parameterized_threshold_factory(
    threshold: float,
) -> _ParameterizedThresholdDetector:
    return _ParameterizedThresholdDetector(threshold)


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


def test_threshold_calibration_selects_on_calibration_and_certifies_evaluation() -> None:
    calibration = CalibrationConfig(trials=32, horizon=16, n_features=1, seed=11)
    evaluation = CalibrationConfig(trials=32, horizon=16, n_features=1, seed=12)
    calibration_factory = _ConstantPointFactory("calibration")
    evaluation_factory = _ConstantPointFactory("evaluation")
    first = calibrate_threshold(
        _parameterized_threshold_factory,
        calibration_factory,
        evaluation_factory,
        detector_family_identity="parameterized-threshold-family:v1",
        candidate_thresholds=(0.5, 1.0),
        calibration_config=calibration,
        evaluation_config=evaluation,
        max_false_alarm_rate=0.2,
    )
    second = calibrate_threshold(
        _parameterized_threshold_factory,
        calibration_factory,
        evaluation_factory,
        detector_family_identity="parameterized-threshold-family:v1",
        candidate_thresholds=(0.5, 1.0),
        calibration_config=calibration,
        evaluation_config=evaluation,
        max_false_alarm_rate=0.2,
    )

    assert isinstance(first, ThresholdCalibrationResult)
    assert first == second
    assert first.selected_threshold == 1.0
    assert first.selected_index == 1
    assert first.calibration_results[0].false_alarm_rate == 1.0
    assert first.calibration_results[1].false_alarm_rate == 0.0
    assert first.calibration_observation_identity == calibration_factory.identity
    assert first.evaluation_observation_identity == evaluation_factory.identity
    assert first.evaluation_result.seed == evaluation.seed
    assert first.approved
    certificate = first.to_certificate()
    assert certificate.approved
    assert certificate.selection_identity == first.identity
    assert first.to_dict()["version"] == 2
    assert first.to_dict()["identity"] == first.identity


def test_threshold_calibration_rejects_unapproved_selection_or_reused_seed() -> None:
    config = CalibrationConfig(trials=16, horizon=8, n_features=1, seed=11)
    with pytest.raises(ValueError, match="no candidate threshold"):
        calibrate_threshold(
            _parameterized_threshold_factory,
            _ConstantPointFactory("calibration"),
            _ConstantPointFactory("evaluation"),
            detector_family_identity="parameterized-threshold-family:v1",
            candidate_thresholds=(0.5,),
            calibration_config=config,
            evaluation_config=CalibrationConfig(
                trials=16, horizon=8, n_features=1, seed=12
            ),
            max_false_alarm_rate=0.2,
        )
    with pytest.raises(ValueError, match="distinct seeds"):
        calibrate_threshold(
            _parameterized_threshold_factory,
            _ConstantPointFactory("calibration"),
            _ConstantPointFactory("evaluation"),
            detector_family_identity="parameterized-threshold-family:v1",
            candidate_thresholds=(1.0,),
            calibration_config=config,
            evaluation_config=config,
            max_false_alarm_rate=0.2,
        )
    factory = _ConstantPointFactory("same")
    with pytest.raises(ValueError, match="distinct observation factories"):
        calibrate_threshold(
            _parameterized_threshold_factory,
            factory,
            factory,
            detector_family_identity="parameterized-threshold-family:v1",
            candidate_thresholds=(1.0,),
            calibration_config=config,
            evaluation_config=CalibrationConfig(
                trials=16, horizon=8, n_features=1, seed=12
            ),
            max_false_alarm_rate=0.2,
        )


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


def test_null_certificate_requires_its_conservative_bound_to_pass() -> None:
    result = calibrate_null(
        _ThresholdDetector,
        _zeros,
        config=CalibrationConfig(trials=32, horizon=16, n_features=1, seed=11),
    )
    approved = result.to_certificate(max_false_alarm_rate=0.2)
    rejected = result.to_certificate(max_false_alarm_rate=0.05)

    assert isinstance(approved, CalibrationCertificate)
    assert approved.approved
    assert not rejected.approved
    assert approved.identity != rejected.identity
    assert approved.to_dict()["approved"] is True
    assert approved.to_dict()["version"] == 2
    assert approved.to_dict()["selection_identity"] is None


def test_certificate_rejects_rates_or_intervals_inconsistent_with_trial_count() -> None:
    with pytest.raises(ValueError, match="false_alarm_rate"):
        CalibrationCertificate(
            detector_identity="detector:v1",
            null_config_identity="null:v1",
            trials=100,
            horizon=16,
            false_alarm_count=0,
            false_alarm_rate=0.01,
            false_alarm_ci_high=0.03699349820698568,
            max_false_alarm_rate=0.05,
        )
    with pytest.raises(ValueError, match="false_alarm_ci_high"):
        CalibrationCertificate(
            detector_identity="detector:v1",
            null_config_identity="null:v1",
            trials=100,
            horizon=16,
            false_alarm_count=0,
            false_alarm_rate=0.0,
            false_alarm_ci_high=0.01,
            max_false_alarm_rate=0.05,
        )


def test_stationary_block_bootstrap_is_seeded_and_identity_bound() -> None:
    source = np.arange(60.0).reshape(30, 2)
    factory = StationaryBlockBootstrap(
        source,
        block_length=4,
        source_id="ar1-reference:v1",
    )
    first = factory(np.random.default_rng(17), 24, 2)
    second = factory(np.random.default_rng(17), 24, 2)

    np.testing.assert_array_equal(first, second)
    assert first.shape == (24, 2)
    assert factory.to_dict()["identity"] == factory.identity
    assert factory.identity
    assert np.all(np.isin(first, source))

    with pytest.raises(ValueError, match="feature dimension"):
        factory(np.random.default_rng(17), 24, 1)


def test_eprocess_null_calibration_is_reproducible_and_stops_at_crossing() -> None:
    config = EProcessCalibrationConfig(
        trials=256,
        horizon=256,
        alpha=0.05,
        eta=0.5,
        seed=19,
    )
    first = calibrate_eprocess_null(_RademacherScores(), config=config)
    second = calibrate_eprocess_null(_RademacherScores(), config=config)

    assert first == second
    assert first.score_factory_identity == _RademacherScores.identity
    assert first.threshold == 20.0
    assert first.threshold_crossing_count < first.trials
    assert first.threshold_crossing_ci_high < 0.1
    assert first.to_dict()["config_identity"] == first.config_identity
    assert all(0 <= step <= config.horizon for step in first.first_crossing_steps)


def test_eprocess_null_calibration_rejects_unbounded_or_malformed_scores() -> None:
    with pytest.raises(ValueError, match="outside"):
        calibrate_eprocess_null(
            lambda rng, horizon: np.full(horizon, 1.1),
            config=EProcessCalibrationConfig(trials=2, horizon=4),
        )
    with pytest.raises(ValueError, match="one-dimensional"):
        calibrate_eprocess_null(
            lambda rng, horizon: np.zeros((horizon, 1)),
            config=EProcessCalibrationConfig(trials=2, horizon=4),
        )
    with pytest.raises(ValueError, match="eta"):
        EProcessCalibrationConfig(eta=1.1)


def test_promotion_null_calibration_binds_gate_allocation_and_selection_order() -> None:
    config = PromotionCalibrationConfig(
        trials=8,
        horizon=16,
        challengers=3,
        alpha=0.05,
        eta=0.5,
        seed=23,
    )
    result = calibrate_promotion_null(_AllPositivePromotionScores(), config=config)

    assert result.threshold_crossing_count == config.trials
    assert result.first_promotion_epochs == (0,) * config.trials
    assert result.first_promotion_steps == (10,) * config.trials
    assert result.first_promoted_challenger_indices == (0,) * config.trials
    assert result.challenger_alpha_allocations == pytest.approx(
        (0.0125, 0.00625, 0.003125)
    )
    assert result.challenger_thresholds == pytest.approx(
        (80.0, 160.0, 320.0)
    )
    assert result.to_dict()["config_identity"] == result.config_identity


def test_promotion_null_calibration_is_reproducible_and_negative_null_stays_closed() -> None:
    config = PromotionCalibrationConfig(
        trials=64,
        horizon=128,
        challengers=3,
        alpha=0.05,
        eta=0.5,
        seed=29,
    )
    first = calibrate_promotion_null(_RademacherPromotionScores(), config=config)
    second = calibrate_promotion_null(_RademacherPromotionScores(), config=config)
    negative = calibrate_promotion_null(_AllNegativePromotionScores(), config=config)

    assert first == second
    assert first.score_factory_identity == _RademacherPromotionScores.identity
    assert len(first.first_promotion_steps) == config.trials
    assert first.threshold_crossing_count <= config.trials
    assert negative.threshold_crossing_count == 0
    assert negative.first_promotion_epochs == (1,) * config.trials
    assert negative.first_promotion_steps == (config.horizon,) * config.trials
    assert negative.first_promoted_challenger_indices == (-1,) * config.trials


def test_promotion_null_calibration_rejects_wrong_dimensions_and_limits() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        calibrate_promotion_null(
            lambda rng, horizon, challengers: np.zeros(horizon),
            config=PromotionCalibrationConfig(trials=1, horizon=4, challengers=2),
        )
    with pytest.raises(ValueError, match="shaped"):
        calibrate_promotion_null(
            lambda rng, horizon, challengers: np.zeros((horizon, challengers + 1)),
            config=PromotionCalibrationConfig(trials=1, horizon=4, challengers=2),
        )
    with pytest.raises(ValueError, match="challengers"):
        PromotionCalibrationConfig(challengers=0)


def test_promotion_null_calibration_spends_a_new_epoch_alpha_share_after_reset() -> None:
    result = calibrate_promotion_null(
        _SecondEpochPositivePromotionScores(),
        config=PromotionCalibrationConfig(
            trials=4,
            horizon=16,
            challengers=2,
            epochs=2,
            alpha=0.05,
            eta=0.5,
            seed=37,
        ),
    )

    assert result.threshold_crossing_count == 4
    assert result.first_promotion_epochs == (1, 1, 1, 1)
    assert result.first_promotion_steps == (12, 12, 12, 12)
    assert result.first_promoted_challenger_indices == (0, 0, 0, 0)
    assert result.challenger_alpha_schedule[0] == pytest.approx((0.0125, 0.00625))
    assert result.challenger_alpha_schedule[1] == pytest.approx((0.00625, 0.003125))


def test_calibration_result_records_observation_factory_identity() -> None:
    factory = StationaryBlockBootstrap(
        np.arange(40.0),
        block_length=3,
        source_id="serial-null:v1",
    )
    result = calibrate_null(
        _ThresholdDetector,
        factory,
        config=CalibrationConfig(trials=8, horizon=12, n_features=1, seed=4),
    )

    assert result.observation_identity == factory.identity
    assert result.to_dict()["observation_identity"] == factory.identity
