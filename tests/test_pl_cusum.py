"""Tests for the exploratory persistent-spectrum CUSUM controller."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from topology_gate.calibration import CalibrationConfig, calibrate_null, calibrate_shift
from topology_gate.persistent import (
    PersistentLaplacianBackend,
    PersistentLaplacianConfig,
)
from topology_gate.pl_cusum import (
    PersistentCUSUMConfig,
    PersistentCUSUMError,
    PersistentLaplacianCUSUM,
)


def backend() -> PersistentLaplacianBackend:
    return PersistentLaplacianBackend(
        PersistentLaplacianConfig(
            max_vertices=4,
            max_simplices=100,
            q=0,
            n_eigenvalues=4,
        )
    )


def config(*, threshold: float = 1.0) -> PersistentCUSUMConfig:
    return PersistentCUSUMConfig(
        cloud_window=4,
        min_points=4,
        backend_eigenvalues=4,
        positive_spectrum_width=2,
        betti_dimensions=(0, 1),
        calibration_window=2,
        calibration_min_periods=2,
        drift=0.0,
        threshold=threshold,
        forgetting_lambda_min=0.8,
        forgetting_lambda_max=0.99,
    )


def rows() -> tuple[tuple[float, float], ...]:
    return (
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (0.0, 1.0),
        (0.0, 0.0),
        (1.0, 0.0),
        (2.0, 0.0),
        (3.0, 0.0),
    )


def test_persistent_cusum_is_causal_and_emits_backend_provenance() -> None:
    controller = PersistentLaplacianCUSUM(backend(), config())
    observations = [controller.observe(row) for row in rows()]

    assert [item.reason for item in observations[:3]] == [
        "insufficient_point_cloud"
    ] * 3
    assert observations[3].reason == "calibration_warmup"
    assert observations[5].ready
    assert observations[7].alarm
    assert observations[7].score > observations[5].score
    assert observations[7].betti_numbers == (1, 0)
    assert len(observations[7].positive_eigenvalues) == 2
    assert len(observations[7].backend_evidence_digest or "") == 64
    assert observations[7].raw_features == observations[7].state
    assert observations[7].whitened_features == observations[7].standardized_state
    assert 0.8 <= observations[7].forgetting_factor <= 0.99


def test_persistent_cusum_chunked_restore_matches_one_shot() -> None:
    path = rows()
    one_shot = PersistentLaplacianCUSUM(backend(), config())
    expected = [one_shot.observe(row).to_dict() for row in path]

    first = PersistentLaplacianCUSUM(backend(), config())
    actual = [first.observe(row).to_dict() for row in path[:6]]
    checkpoint = copy.deepcopy(first.stream_state_dict())
    resumed = PersistentLaplacianCUSUM(backend(), config())
    resumed.load_stream_state_dict(checkpoint)
    actual.extend(resumed.observe(row).to_dict() for row in path[6:])

    assert actual == expected
    assert resumed.stream_state_dict() == one_shot.stream_state_dict()


def test_persistent_cusum_batch_facade_integrates_with_calibration_harness() -> None:
    def factory() -> PersistentLaplacianCUSUM:
        return PersistentLaplacianCUSUM(backend(), config(threshold=100.0))

    def null_observations(rng: np.random.Generator, horizon: int, features: int) -> np.ndarray:
        return rng.normal(size=(horizon, features))

    calibration = CalibrationConfig(trials=4, horizon=8, n_features=2, seed=17)
    null_result = calibrate_null(factory, null_observations, config=calibration)
    shift_result = calibrate_shift(
        factory,
        null_observations,
        shift_index=4,
        config=calibration,
    )

    batch = factory().detect(null_observations(np.random.default_rng(3), 8, 2))
    assert len(batch.alarms) == 8
    assert len(batch.scores) == 8
    assert null_result.detector_identity == factory().config_identity
    assert shift_result.detector_identity == factory().config_identity
    assert len(null_result.first_alarm_steps) == 4
    assert len(shift_result.detection_delays) == 4


def test_persistent_cusum_state_schema_is_strict_and_restore_is_atomic() -> None:
    controller = PersistentLaplacianCUSUM(backend(), config())
    for row in rows()[:6]:
        controller.observe(row)
    before = copy.deepcopy(controller.stream_state_dict())

    unknown = copy.deepcopy(before)
    unknown["unexpected"] = True
    with pytest.raises(PersistentCUSUMError, match="state fields"):
        controller.load_stream_state_dict(unknown)
    assert controller.stream_state_dict() == before

    tampered = copy.deepcopy(before)
    tampered["history"][0]["state"] = [99.0]
    with pytest.raises(PersistentCUSUMError, match="state history dimension"):
        controller.load_stream_state_dict(tampered)
    assert controller.stream_state_dict() == before


class BadBackend:
    identity = "test.bad-backend:v1"
    max_vertices = 4
    n_eigenvalues = 4

    def __call__(self, point_cloud: object, n_eigenvalues: int) -> object:
        del point_cloud, n_eigenvalues
        return (0.0, 1.0)


def test_persistent_cusum_rolls_back_when_backend_contract_fails() -> None:
    controller = PersistentLaplacianCUSUM(BadBackend(), config())
    for row in rows()[:3]:
        controller.observe(row)
    before = controller.stream_state_dict()

    with pytest.raises(PersistentCUSUMError, match="must return"):
        controller.observe(rows()[3])

    assert controller.stream_state_dict() == before


def test_persistent_cusum_configuration_rejects_incompatible_width() -> None:
    with pytest.raises(PersistentCUSUMError, match="positive_spectrum_width"):
        PersistentCUSUMConfig(backend_eigenvalues=2, positive_spectrum_width=3)

    with pytest.raises(PersistentCUSUMError, match="stable identity"):
        PersistentLaplacianCUSUM(lambda point_cloud, width: (0.0, 1.0))
