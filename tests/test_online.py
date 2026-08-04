"""End-to-end causal composition tests."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from topology_gate.config import ModelConfig
from topology_gate.observability import AuditEvent, AuditLog, MetricsRegistry
from topology_gate.online import (
    OnlineRunConfig,
    OnlineStreamState,
    PendingLabelRecord,
    run_recursive_rls,
)
from topology_gate.rls import RLS, RLSConfig
from topology_gate.synthetic import generate_synthetic_regimes
from topology_gate.topology import RollingTopologyDetector, TopologyConfig


def _components() -> tuple[RollingTopologyDetector, RLS]:
    detector = RollingTopologyDetector(
        TopologyConfig(
            embedding_dim=1,
            cloud_window=8,
            graph_neighbors=2,
            n_eigenvalues=2,
            min_points=4,
            calibration_window=12,
            calibration_min_periods=3,
            drift=0.1,
            threshold=2.0,
            forgetting_lambda_min=0.8,
            forgetting_lambda_max=0.99,
        )
    )
    learner = RLS(RLSConfig(n_features=1, lambda_min=0.8, lambda_max=1.0))
    return detector, learner


class _AggressiveDetector:
    config_identity = "test-aggressive-detector:v1"
    config = SimpleNamespace(forgetting_lambda_max=0.99)

    def __init__(self) -> None:
        self.steps = 0

    def reset_stream(self) -> None:
        self.steps = 0

    def observe(self, observation: np.ndarray) -> SimpleNamespace:
        assert observation.shape == (1,)
        self.steps += 1
        return SimpleNamespace(
            score=1.0,
            alarm=True,
            ready=True,
            forgetting_factor=0.8,
        )

    def stream_state_dict(self) -> dict[str, object]:
        return {"steps": self.steps}


def test_online_forgetting_requires_matching_approved_calibration() -> None:
    features = np.ones((4, 1), dtype=float)
    outcomes = np.zeros(4, dtype=float)

    uncertified = run_recursive_rls(
        features,
        outcomes,
        learner=RLS(RLSConfig(n_features=1, lambda_min=0.8, lambda_max=1.0)),
        detector=_AggressiveDetector(),
    )
    np.testing.assert_allclose(uncertified.forgetting_factors, 0.99)
    assert uncertified.acceleration_authorized is not None
    assert not np.any(uncertified.acceleration_authorized)
    assert uncertified.metrics["accelerated_forgetting_count"] == 0.0

    approved = run_recursive_rls(
        features,
        outcomes,
        learner=RLS(RLSConfig(n_features=1, lambda_min=0.8, lambda_max=1.0)),
        detector=_AggressiveDetector(),
        calibration=SimpleNamespace(
            detector_identity=_AggressiveDetector.config_identity,
            identity="test-certificate:v1",
            approved=True,
        ),
    )
    np.testing.assert_allclose(approved.forgetting_factors, 0.8)
    assert approved.acceleration_authorized is not None
    assert np.all(approved.acceleration_authorized)
    assert approved.calibration_identity == "test-certificate:v1"
    assert approved.metrics["accelerated_forgetting_count"] == 4.0

    with pytest.raises(ValueError, match="does not match detector identity"):
        run_recursive_rls(
            features[:1],
            outcomes[:1],
            learner=RLS(RLSConfig(n_features=1)),
            detector=_AggressiveDetector(),
            calibration=SimpleNamespace(
                detector_identity="wrong-detector:v1",
                identity="test-certificate:v1",
                approved=True,
            ),
        )


def test_recursive_runner_is_deterministic_and_respects_delayed_labels() -> None:
    dataset = generate_synthetic_regimes(
        n_steps=48,
        n_features=1,
        change_points=(24,),
        seed=41,
        label_delay=2,
        feature_noise=0.02,
        return_noise=0.01,
    )
    detector_a, learner_a = _components()
    detector_b, learner_b = _components()
    config = OnlineRunConfig(label_delay=2, transaction_cost_bps=3.0)
    first = run_recursive_rls(
        dataset.features.values,
        dataset.labels.values,
        realized_returns=dataset.realized_returns,
        market_states=dataset.features.values,
        detector=detector_a,
        learner=learner_a,
        config=config,
        shift_points=dataset.change_points,
    )
    second = run_recursive_rls(
        dataset.features.values,
        dataset.labels.values,
        realized_returns=dataset.realized_returns,
        market_states=dataset.features.values,
        detector=detector_b,
        learner=learner_b,
        config=config,
        shift_points=dataset.change_points,
    )
    np.testing.assert_array_equal(first.predictions, second.predictions)
    np.testing.assert_array_equal(first.forgetting_factors, second.forgetting_factors)
    np.testing.assert_array_equal(first.alarms, second.alarms)
    assert not np.any(first.update_steps[-2:])
    assert np.all(first.forgetting_factors >= 0.8)
    assert np.all(first.forgetting_factors <= 0.99)
    np.testing.assert_array_equal(first.realized_returns, dataset.realized_returns)


def test_detector_stream_snapshot_replays_exactly() -> None:
    detector, _ = _components()
    observations = np.linspace(-1.0, 1.0, 20).reshape(-1, 1)
    for row in observations[:12]:
        detector.observe(row)
    snapshot = detector.stream_state_dict()
    expected = [detector.observe(row) for row in observations[12:]]

    restored, _ = _components()
    restored.load_stream_state_dict(snapshot)
    actual = [restored.observe(row) for row in observations[12:]]
    assert expected == actual


def test_configuration_fingerprint_and_audit_are_json_safe(tmp_path) -> None:
    detector, _ = _components()
    model_config = ModelConfig(topology=detector.config, rls=RLSConfig(n_features=1))
    assert len(model_config.fingerprint()) == 64

    audit = AuditLog(max_events=2)
    audit.append(AuditEvent("detector", 0, {"score": 0.0}))
    audit.append(AuditEvent("promotion", 1, {"promoted": False}))
    audit.append(AuditEvent("prediction", 2, {"position": 0.25}))
    destination = tmp_path / "audit.jsonl"
    audit.to_jsonl(destination)
    assert len(destination.read_text(encoding="utf-8").splitlines()) == 2

    metrics = MetricsRegistry()
    metrics.increment("predictions")
    with metrics.timer("step"):
        pass
    assert metrics.snapshot()["counters"]["predictions"] == 1


def test_irregular_label_availability_is_causal_and_terminal_pending_is_visible() -> None:
    detector, learner = _components()
    features = np.arange(8, dtype=float).reshape(-1, 1)
    availability = np.array([2, 5, 3, 7, 6, 9, 10, 11])
    result = run_recursive_rls(
        features,
        np.arange(8, dtype=float),
        realized_returns=np.ones(8),
        learner=learner,
        detector=detector,
        config=OnlineRunConfig(max_pending_labels=16),
        label_available_at=availability,
    )

    assert result.pending_labels
    assert [item.source_step for item in result.pending_labels] == [5, 6, 7]
    assert result.pending_labels[0].available_step == 9
    assert result.update_steps[0]
    assert result.update_steps[1]
    assert result.update_steps[2]
    assert result.stream_state is not None
    restored = OnlineStreamState.from_state_dict(result.stream_state.state_dict())
    assert restored == result.stream_state
    unknown_state = dict(result.stream_state.state_dict())
    unknown_state["unmodeled_field"] = "reject"
    with pytest.raises(ValueError, match="unknown or missing"):
        OnlineStreamState.from_state_dict(unknown_state)
    unknown_pending = dict(result.pending_labels[0].state_dict())
    unknown_pending["unmodeled_field"] = "reject"
    with pytest.raises(ValueError, match="unknown or missing"):
        PendingLabelRecord.from_state_dict(unknown_pending)
    with pytest.raises(ValueError, match="strictly after"):
        run_recursive_rls(
            features[:3],
            np.arange(3, dtype=float),
            learner=RLS(1),
            label_available_at=[0, 2, 3],
        )
    with pytest.raises(ValueError, match="duplicate"):
        OnlineStreamState(
            next_step=3,
            previous_position=0.0,
            pending_labels=(
                PendingLabelRecord(0, 4, (1.0,), 1.0, 1.0),
                PendingLabelRecord(0, 5, (1.0,), 1.0, 1.0),
            ),
            feature_count=1,
        )


def test_chunked_delayed_replay_matches_one_shot_and_delay_is_integer() -> None:
    features = np.arange(8, dtype=float).reshape(-1, 1)
    outcomes = np.linspace(0.1, 0.8, 8)
    one_shot_learner = RLS(RLSConfig(n_features=1, lambda_min=0.8, lambda_max=1.0))
    one_shot = run_recursive_rls(
        features,
        outcomes,
        learner=one_shot_learner,
        config=OnlineRunConfig(label_delay=2),
    )

    chunked_learner = RLS(RLSConfig(n_features=1, lambda_min=0.8, lambda_max=1.0))
    first = run_recursive_rls(
        features[:4],
        outcomes[:4],
        learner=chunked_learner,
        config=OnlineRunConfig(label_delay=2),
    )
    second = run_recursive_rls(
        features[4:],
        outcomes[4:],
        learner=chunked_learner,
        config=OnlineRunConfig(label_delay=2, reset_state=False),
        initial_state=first.stream_state,
    )

    np.testing.assert_allclose(second.predictions, one_shot.predictions[4:])
    assert chunked_learner.state_dict() == one_shot_learner.state_dict()
    assert second.stream_state is not None
    assert second.stream_state.next_step == 8
    assert second.pending_labels == one_shot.pending_labels

    with pytest.raises(ValueError, match="integer"):
        OnlineRunConfig(label_delay=1.5)  # type: ignore[arg-type]


def test_strict_economic_mode_rejects_implicit_target_returns() -> None:
    with pytest.raises(ValueError, match="realized_returns are required"):
        run_recursive_rls(
            np.ones((3, 1)),
            np.ones(3),
            learner=RLS(RLSConfig(n_features=1)),
            config=OnlineRunConfig(require_realized_returns=True),
        )
