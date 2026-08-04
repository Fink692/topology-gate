"""Tests for the non-topological mean/covariance CUSUM baseline."""

from __future__ import annotations

import numpy as np
import pytest

from topology_gate.cpd import (
    MeanCovarianceCUSUM,
    MeanCovarianceCUSUMConfig,
)


def test_mean_covariance_cusum_uses_prior_block_and_detects_shift() -> None:
    detector = MeanCovarianceCUSUM(
        MeanCovarianceCUSUMConfig(
            n_features=1,
            block_window=4,
            threshold=2.0,
            forgetting_lambda_min=0.8,
            forgetting_lambda_max=0.99,
        )
    )
    for _ in range(8):
        result = detector.observe([0.0])
    assert result.ready
    assert not result.alarm
    shifted = [detector.observe([5.0]) for _ in range(4)]
    assert shifted[-1].alarm
    assert shifted[-1].score >= 2.0
    assert 0.8 <= shifted[-1].forgetting_factor <= 0.99


def test_mean_covariance_batch_and_stream_round_trip() -> None:
    config = MeanCovarianceCUSUMConfig(n_features=2, block_window=2)
    values = np.vstack((np.zeros((4, 2)), np.ones((4, 2))))
    first = MeanCovarianceCUSUM(config)
    batch = first.detect(values)
    snapshot = first.stream_state_dict()
    second = MeanCovarianceCUSUM(config)
    second.load_stream_state_dict(snapshot)
    expected = first.observe([1.0, 1.0])
    actual = second.observe([1.0, 1.0])
    assert batch.alarms.shape == (8,)
    assert expected == actual


def test_mean_covariance_state_rejects_identity_and_malformed_rows() -> None:
    detector = MeanCovarianceCUSUM(MeanCovarianceCUSUMConfig(n_features=1))
    state = detector.stream_state_dict()
    state["config_identity"] = "wrong"
    with pytest.raises(ValueError, match="identity"):
        detector.load_stream_state_dict(state)
    with pytest.raises(ValueError, match="feature width"):
        detector.detect(np.zeros((4, 2)))


def test_mean_covariance_config_rejects_invalid_lambda_bounds() -> None:
    with pytest.raises(ValueError, match="lambda bounds"):
        MeanCovarianceCUSUMConfig(
            n_features=1,
            forgetting_lambda_min=0.9,
            forgetting_lambda_max=0.8,
        )
