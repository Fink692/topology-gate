"""Focused tests for the causal rolling topology/change-detection slice."""

from __future__ import annotations

import math
import random

import pytest

import topology_gate.topology as topology
from topology_gate.persistent import (
    PersistentLaplacianBackend,
    PersistentLaplacianConfig,
)
from topology_gate.topology import (
    RollingTopologyDetector,
    TopologyConfig,
    cusum_scores,
    point_cloud_features,
    robust_whiten,
    rolling_point_cloud,
    spectral_summary,
)


def _as_rows(value):
    return value.tolist() if hasattr(value, "tolist") else value


def _as_values(value):
    value = _as_rows(value)
    if isinstance(value, list):
        return value
    return list(value)


def _assert_finite(value):
    value = _as_rows(value)
    if isinstance(value, (list, tuple)):
        for item in value:
            _assert_finite(item)
    else:
        assert math.isfinite(float(value))


def _series(length=180, seed=7):
    rng = random.Random(seed)
    return [
        0.25 * math.sin(index / 8.0)
        + 0.04 * math.cos(index / 3.0)
        + rng.gauss(0.0, 0.025)
        for index in range(length)
    ]


def _config(**overrides):
    values = dict(
        embedding_dim=3,
        cloud_window=24,
        point_stride=1,
        graph_neighbors=4,
        n_eigenvalues=3,
        min_points=12,
        calibration_window=48,
        calibration_min_periods=12,
        threshold=8.0,
        drift=2.0,
        decay=1.0,
    )
    values.update(overrides)
    return TopologyConfig(**values)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"embedding_dim": 0},
        {"cloud_window": 1},
        {"graph_neighbors": 24},
        {"n_eigenvalues": 25},
        {"min_points": 4},
        {"calibration_min_periods": 49},
        {"threshold": 0},
        {"decay": 0},
        {"forgetting_lambda_min": 0.99, "forgetting_lambda_max": 0.90},
        {"forgetting_lambda_max": 1.1},
        {"forgetting_sensitivity": -1.0},
        {"input_kind": "levels"},
    ],
)
def test_configuration_rejects_unsafe_parameters(kwargs):
    with pytest.raises(ValueError):
        _config(**kwargs)


def test_detector_is_deterministic_and_warmup_is_finite():
    values = _series()
    detector = RollingTopologyDetector(_config())
    first = detector.detect(values)
    second = detector.detect(values)

    assert _as_values(first.features) == _as_values(second.features)
    assert _as_values(first.whitened_features) == _as_values(second.whitened_features)
    assert _as_values(first.scores) == _as_values(second.scores)
    assert _as_values(first.alarms) == _as_values(second.alarms)
    assert first.method == "knn_normalized_laplacian_approximation"
    assert len(first.features) == len(values)
    assert len(first.features[0]) == _config().feature_count
    warmup_end = _config().embedding_dim - 1 + _config().min_points - 1
    assert not any(first.valid[:warmup_end])
    assert all(not calibrated or valid
               for calibrated, valid in zip(first.calibrated, first.valid))
    for output in (
        first.features,
        first.whitened_features,
        first.innovation,
        first.scores,
        first.calibration_location,
        first.calibration_scale,
    ):
        _assert_finite(output)


def test_prefix_is_causal_and_future_changes_cannot_leak_backwards():
    prefix = _series(125, seed=11)
    altered_future = prefix + [8.0 * math.sin(index) for index in range(40)]
    detector = RollingTopologyDetector(_config())
    short = detector.detect(prefix)
    long = detector.detect(altered_future)

    for short_values, long_values in (
        (short.features, long.features),
        (short.whitened_features, long.whitened_features),
        (short.innovation, long.innovation),
        (short.scores, long.scores),
        (short.valid, long.valid),
        (short.calibrated, long.calibrated),
    ):
        assert _as_values(short_values) == _as_values(long_values)[: len(prefix)]


def test_point_cloud_is_causal_and_spectral_summary_is_permutation_stable():
    values = [float(index * index % 11) for index in range(40)]
    cloud = rolling_point_cloud(values, end_index=25, embedding_dim=3,
                                cloud_window=10, point_stride=2)
    future_mutated = values[:26] + [999.0] * 14
    mutated_cloud = rolling_point_cloud(future_mutated, end_index=25,
                                        embedding_dim=3, cloud_window=10,
                                        point_stride=2)
    assert cloud == mutated_cloud

    summary = spectral_summary(cloud, n_eigenvalues=4, graph_neighbors=3)
    reversed_summary = spectral_summary(tuple(reversed(cloud)), n_eigenvalues=4,
                                        graph_neighbors=3)
    assert summary.method == "knn_normalized_laplacian_approximation"
    assert summary.eigenvalues == pytest.approx(reversed_summary.eigenvalues, abs=1e-10)
    assert summary.spectral_entropy == pytest.approx(
        reversed_summary.spectral_entropy, abs=1e-10
    )
    assert 0.0 <= summary.spectral_entropy <= 1.0
    assert all(0.0 <= value <= 2.0 for value in summary.eigenvalues)


def test_robust_whitening_resists_one_large_reference_outlier():
    reference = [[float(index), float(index) * 0.5] for index in range(20)]
    reference.append([10_000.0, -10_000.0])
    result = robust_whiten(
        [[5.0, 2.5]],
        reference=reference,
        scale_floor=1e-6,
        ridge=1e-4,
    )
    assert result.location[0] < 20.0
    assert result.location[1] < 20.0
    assert len(result.values) == 1
    _assert_finite(result.values)
    _assert_finite(result.whitening_matrix)


def test_cusum_recursion_and_invalid_rows_are_explicit():
    result = cusum_scores(
        [0.0, 3.0, 3.0, 0.0, 3.0],
        drift=1.0,
        threshold=3.0,
        decay=1.0,
        valid=[False, True, True, False, True],
    )
    assert _as_values(result.scores) == pytest.approx([0.0, 2.0, 4.0, 0.0, 2.0])
    assert _as_values(result.alarms) == [False, False, True, False, False]
    with pytest.raises(ValueError):
        cusum_scores([1.0, -1.0])


def test_detector_reacts_to_a_large_volatility_regime_after_calibration():
    rng = random.Random(19)
    values = [rng.gauss(0.0, 0.04) for _ in range(120)]
    values.extend(rng.gauss(0.0, 0.7) for _ in range(80))
    test_config = _config(embedding_dim=2, threshold=9.0)
    result = RollingTopologyDetector(test_config).detect(values)

    baseline_scores = _as_values(result.scores)[80:110]
    changed_scores = _as_values(result.scores)[120:150]
    assert max(baseline_scores) < test_config.threshold
    assert max(changed_scores) > test_config.threshold
    assert sum(bool(value) for value in _as_values(result.alarms)[120:]) > 0


def test_optional_persistent_backend_seam_is_visible_and_causal():
    calls = []

    def backend(cloud, n_eigenvalues):
        calls.append((len(cloud), n_eigenvalues, tuple(cloud[-1])))
        return [0.0, 0.4, 0.9][:n_eigenvalues]

    summary = spectral_summary(
        [[0.0, 0.0], [1.0, 0.5], [2.0, 1.0]],
        n_eigenvalues=3,
        graph_neighbors=2,
        persistent_laplacian_backend=backend,
    )
    assert summary.method == "persistent_laplacian_backend"
    assert summary.backend_name == "test_optional_persistent_backend_seam_is_visible_and_causal.<locals>.backend"
    assert summary.is_approximation is False
    assert calls and calls[0][1] == 3

    detector = RollingTopologyDetector(
        _config(persistent_laplacian_backend=backend)
    )
    result = detector.detect(_series(90))
    assert result.method == "persistent_laplacian_backend"
    assert calls[-1][0] <= _config().cloud_window
    _assert_finite(result.features)


def test_configured_exact_persistent_backend_is_causal_and_checkpointable():
    backend = PersistentLaplacianBackend(
        PersistentLaplacianConfig(
            max_vertices=8,
            max_simplices=500,
            q=0,
            n_eigenvalues=2,
        )
    )
    config = _config(
        embedding_dim=1,
        cloud_window=8,
        graph_neighbors=2,
        n_eigenvalues=2,
        min_points=4,
        calibration_window=8,
        calibration_min_periods=2,
        persistent_laplacian_backend=backend,
    )
    values = _series(14, seed=31)
    detector = RollingTopologyDetector(config)
    short = detector.detect(values[:10])
    long = detector.detect(values)

    assert short.method == "persistent_laplacian_backend"
    assert detector.backend_identity == backend.identity
    assert _as_values(short.features) == _as_values(long.features)[:10]
    assert _as_values(short.scores) == _as_values(long.scores)[:10]
    short_digests = list(short.backend_evidence_digests)
    long_digests = list(long.backend_evidence_digests)
    assert short_digests == long_digests[:10]
    assert short_digests[-1] is not None
    assert all(value is None or len(value) == 64 for value in short_digests)

    streaming = RollingTopologyDetector(config)
    for value in values[:7]:
        streaming.observe([value])
    snapshot = streaming.stream_state_dict()
    expected = [streaming.observe([value]) for value in values[7:]]

    restored = RollingTopologyDetector(config)
    restored.load_stream_state_dict(snapshot)
    actual = [restored.observe([value]) for value in values[7:]]
    assert expected == actual


def test_persistent_backend_rejects_malformed_evidence_digest():
    class MalformedEvidenceBackend:
        def __call__(self, cloud, n_eigenvalues):
            del cloud, n_eigenvalues
            return type(
                "MalformedResult",
                (),
                {"eigenvalues": (0.0, 1.0), "evidence_digest": "not-a-digest"},
            )()

    with pytest.raises(ValueError, match="evidence digest"):
        spectral_summary(
            [[0.0], [1.0], [2.0]],
            n_eigenvalues=2,
            graph_neighbors=2,
            persistent_laplacian_backend=MalformedEvidenceBackend(),
        )


def test_configured_exact_backend_rejects_incompatible_detector_budgets():
    backend = PersistentLaplacianBackend(
        PersistentLaplacianConfig(
            max_vertices=4,
            max_simplices=100,
            q=0,
            n_eigenvalues=2,
        )
    )
    with pytest.raises(ValueError, match="cloud_window"):
        RollingTopologyDetector(
            _config(
                cloud_window=8,
                n_eigenvalues=2,
                min_points=5,
                persistent_laplacian_backend=backend,
            )
        )

    wider_backend = PersistentLaplacianBackend(
        PersistentLaplacianConfig(
            max_vertices=8,
            max_simplices=500,
            q=0,
            n_eigenvalues=3,
        )
    )
    with pytest.raises(ValueError, match="n_eigenvalues"):
        RollingTopologyDetector(
            _config(
                cloud_window=8,
                n_eigenvalues=2,
                min_points=5,
                persistent_laplacian_backend=wider_backend,
            )
        )


def test_exact_backend_failure_does_not_consume_stream_observation():
    class FailingBackend:
        max_vertices = 8
        n_eigenvalues = 2
        identity = "test.failing-persistent-backend:v1"

        def __call__(self, cloud, n_eigenvalues):
            if len(cloud) >= 4:
                raise ValueError("declared exact resource failure")
            return [0.0, 0.5]

    config = _config(
        embedding_dim=1,
        cloud_window=4,
        graph_neighbors=2,
        n_eigenvalues=2,
        min_points=3,
        calibration_window=6,
        calibration_min_periods=2,
        persistent_laplacian_backend=FailingBackend(),
    )
    detector = RollingTopologyDetector(config)
    detector.observe([0.0])
    detector.observe([0.1])
    detector.observe([0.2])
    before = detector.stream_state_dict()

    with pytest.raises(ValueError, match="declared exact resource failure"):
        detector.observe([0.3])

    assert detector.stream_state_dict() == before


def test_dependency_light_fallback_runs_without_numpy(monkeypatch):
    monkeypatch.setattr(topology, "_np", None)
    result = RollingTopologyDetector(
        _config(cloud_window=12, graph_neighbors=3, n_eigenvalues=2,
                min_points=8, calibration_window=20, calibration_min_periods=6)
    ).detect(_series(60, seed=3))
    assert len(result.features) == 60
    assert len(result.features[0]) == 13
    assert all(isinstance(value, bool) for value in result.alarms)
    _assert_finite(result.scores)


def test_nonfinite_input_is_rejected_instead_of_leaking_nan():
    detector = RollingTopologyDetector(_config())
    with pytest.raises(ValueError, match="NaN or infinity"):
        detector.detect([0.0, 1.0, float("nan"), 2.0])
    with pytest.raises(ValueError, match="NaN or infinity"):
        point_cloud_features([[0.0, 1.0], [float("inf"), 2.0]])


def test_random_finite_inputs_preserve_output_invariants():
    rng = random.Random(123)
    detector = RollingTopologyDetector(
        _config(embedding_dim=2, cloud_window=16, graph_neighbors=3,
                n_eigenvalues=2, min_points=8, calibration_window=24,
                calibration_min_periods=6)
    )
    for _ in range(5):
        values = [rng.uniform(-2.0, 2.0) for _ in range(70)]
        result = detector.detect(values)
        assert all(value >= 0.0 for value in _as_values(result.innovation))
        assert all(value >= 0.0 for value in _as_values(result.scores))
        assert all(count <= detector.config.cloud_window for count in result.point_counts)
        assert all(
            bool(calibrated) <= bool(valid)
            for calibrated, valid in zip(result.calibrated, result.valid)
        )
        _assert_finite(result.features)
