"""Tests for endogenous Wasserstein robustness."""

from __future__ import annotations

import pytest

from topology_gate import EndogenousWassersteinLinearLearner as PublicWassersteinLearner
from topology_gate.wasserstein import (
    WASSERSTEIN_SCHEMA,
    WASSERSTEIN_VERSION,
    EndogenousWassersteinLinearLearner,
    WassersteinRobustConfig,
)


def _config() -> WassersteinRobustConfig:
    return WassersteinRobustConfig(
        n_features=2,
        learning_rate=0.05,
        radius_floor=0.1,
        radius_sensitivity=0.2,
        radius_max=0.5,
        gradient_clip=2.0,
    )


def test_radius_is_prediction_time_and_monotone() -> None:
    learner = EndogenousWassersteinLinearLearner(_config())
    assert learner.radius(0.0) == 0.1
    assert learner.radius(1.0) == pytest.approx(0.3)
    assert learner.radius(100.0) == pytest.approx(0.5)


def test_robust_update_and_objective_are_finite() -> None:
    learner = EndogenousWassersteinLinearLearner(_config())
    before = learner.robust_objective((1.0, 2.0), 3.0, 0.0)
    update = learner.observe((1.0, 2.0), 3.0, 2.0)
    assert before[0] == 0.0
    assert update.step == 1
    assert update.radius == 0.5
    assert update.robust_objective > 0.0
    assert len(update.coefficients) == 2


def test_state_round_trip_and_tamper_rejection() -> None:
    learner = EndogenousWassersteinLinearLearner(_config())
    learner.observe((1.0, 0.0), 1.0, 0.0)
    state = learner.state_dict()
    assert state["schema"] == WASSERSTEIN_SCHEMA
    assert state["version"] == WASSERSTEIN_VERSION
    restored = EndogenousWassersteinLinearLearner.from_state_dict(state)
    assert restored.state_dict() == state
    tampered = dict(state)
    tampered["step"] = 99
    with pytest.raises(ValueError, match="digest"):
        EndogenousWassersteinLinearLearner.from_state_dict(tampered)


def test_invalid_inputs_fail_closed() -> None:
    learner = EndogenousWassersteinLinearLearner(_config())
    with pytest.raises(ValueError, match="exactly"):
        learner.predict((1.0,))
    with pytest.raises(ValueError, match="feature_abs_bound"):
        learner.predict((101.0, 0.0))
    with pytest.raises(ValueError, match="finite"):
        learner.observe((1.0, 0.0), 1.0, float("nan"))
    with pytest.raises(ValueError, match="at least"):
        learner.radius(-1.0)


def test_config_round_trip() -> None:
    config = _config()
    assert WassersteinRobustConfig.from_dict(config.to_dict()) == config
    assert PublicWassersteinLearner is EndogenousWassersteinLinearLearner
