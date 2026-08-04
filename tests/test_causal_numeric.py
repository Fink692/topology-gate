"""Integration tests for the timestamped numerical worker."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from topology_gate.asof import (
    AsOfBook,
    LabelObservation,
    MarketObservation,
    UniverseMembership,
)
from topology_gate.calibration import CalibrationCertificate
from topology_gate.causal_numeric import (
    CausalFeaturePlan,
    CausalNumericError,
    CausalRLSConfig,
    FeatureBinding,
    run_causal_rls_replay,
)
from topology_gate.persistent import (
    PersistentLaplacianBackend,
    PersistentLaplacianConfig,
)
from topology_gate.replay import ReplayStatus
from topology_gate.rls import RLS, RLSConfig
from topology_gate.topology import RollingTopologyDetector, TopologyConfig


def observation(
    record_id: str,
    available: int,
    value: float,
    *,
    event_time: int | None = None,
) -> MarketObservation:
    return MarketObservation(
        record_id=record_id,
        instrument_id="ES",
        event_time=available if event_time is None else event_time,
        available_time=available,
        source_revision=0,
        ingest_sequence=0,
        fields={"x": value, "state": value},
    )


def label(target: str, available: int, value: float | None, status: str = "observed") -> LabelObservation:
    return LabelObservation(
        label_id=f"label-{target}",
        target_id=target,
        event_time=available - 1,
        available_time=available,
        received_time=available,
        status=status,
        value=value,
        source_revision=0,
    )


def plan(*, strict: bool = True) -> CausalFeaturePlan:
    bindings = {
        target: (FeatureBinding("m1", ("x",), "ES"),)
        for target in ("t1", "t2", "t3")
    }
    states = {
        target: (FeatureBinding("m1", ("state",), "ES"),)
        for target in ("t1", "t2", "t3")
    }
    return CausalFeaturePlan(
        bindings,
        state_bindings_by_target=states,
        require_membership=strict,
    )


def detector() -> RollingTopologyDetector:
    return RollingTopologyDetector(
        TopologyConfig(
            embedding_dim=1,
            cloud_window=8,
            graph_neighbors=2,
            n_eigenvalues=2,
            min_points=4,
            calibration_window=8,
            calibration_min_periods=3,
            drift=0.1,
            threshold=2.0,
            forgetting_lambda_min=0.8,
            forgetting_lambda_max=0.99,
        )
    )


def exact_detector() -> RollingTopologyDetector:
    backend = PersistentLaplacianBackend(
        PersistentLaplacianConfig(
            max_vertices=4,
            max_simplices=100,
            q=0,
            n_eigenvalues=2,
        )
    )
    return RollingTopologyDetector(
        TopologyConfig(
            embedding_dim=1,
            cloud_window=4,
            graph_neighbors=2,
            n_eigenvalues=2,
            min_points=3,
            calibration_window=6,
            calibration_min_periods=2,
            persistent_laplacian_backend=backend,
        )
    )


class ReadyDetector:
    config_identity = "ready-detector:v1"
    config = SimpleNamespace(forgetting_lambda_max=0.99)

    def __init__(self) -> None:
        self.calls = 0

    def observe(self, state_features: np.ndarray) -> SimpleNamespace:
        assert state_features.shape == (1,)
        self.calls += 1
        return SimpleNamespace(
            score=1.0,
            alarm=True,
            ready=True,
            forgetting_factor=0.8,
            method="ready-test",
        )

    def stream_state_dict(self) -> dict[str, object]:
        return {"calls": self.calls}

    def validate_stream_state_dict(self, state: dict[str, object]) -> dict[str, object]:
        if set(state) != {"calls"}:
            raise ValueError("invalid ready detector state")
        return {"calls": int(state["calls"])}

    def load_stream_state_dict(self, state: dict[str, object]) -> None:
        self.calls = int(state["calls"])


class MalformedDigestDetector(ReadyDetector):
    def observe(self, state_features: np.ndarray) -> SimpleNamespace:
        result = super().observe(state_features)
        result.backend_evidence_digest = "bad"
        return result


def make_book(*, missing: bool = False) -> AsOfBook:
    return AsOfBook(
        observations=(observation("m1", 1, 1.0),),
        universe=(UniverseMembership("ES", 0, 100, 0, 0, 0),),
        labels=(label("t1", 3, None if missing else 2.0, "missing" if missing else "observed"),),
    )


def make_learner() -> RLS:
    return RLS(RLSConfig(n_features=1, lambda_min=0.8, lambda_max=1.0))


class NonFiniteLearner:
    lambda_min = 0.5
    lambda_max = 1.0
    n_features = 1

    def predict(self, features: np.ndarray) -> float:
        del features
        return float("nan")

    def update(
        self, features: np.ndarray, target: float, *, forgetting_factor: float
    ) -> None:
        del features, target, forgetting_factor

    def state_dict(self) -> dict[str, object]:
        return {"kind": "non-finite"}

    def load_state_dict(self, state: dict[str, object]) -> None:
        if state != {"kind": "non-finite"}:
            raise ValueError("invalid non-finite learner state")


def test_timestamped_detector_rls_path_is_causal_and_freezes_factor() -> None:
    result = run_causal_rls_replay(
        make_book(),
        (1, 2, 4),
        ("t1", "t2", "t3"),
        plan=plan(),
        learner=make_learner(),
        detector=detector(),
        model_config=CausalRLSConfig(model_id="numeric"),
    )

    assert result.predictions.shape == (3,)
    assert result.positions.shape == (3,)
    assert result.detector_scores.shape == (3,)
    assert result.forgetting_factors.shape == (3,)
    assert result.replay.resolutions[0].status is ReplayStatus.OBSERVED
    assert result.replay.resolutions[0].settlement_time == 4
    assert result.pending_target_ids == ("t2", "t3")
    assert all(0.8 <= value <= 0.99 for value in result.forgetting_factors)


def test_exact_topology_artifact_digest_reaches_causal_step_telemetry() -> None:
    result = run_causal_rls_replay(
        make_book(),
        (1, 2, 4),
        ("t1", "t2", "t3"),
        plan=plan(),
        learner=make_learner(),
        detector=exact_detector(),
        model_config=CausalRLSConfig(model_id="exact-telemetry"),
    )

    assert result.topology_evidence_digests[:2] == (None, None)
    assert result.topology_evidence_digests[2] is not None
    assert len(result.topology_evidence_digests[2]) == 64
    assert result.steps[-1].topology_evidence_digest == result.topology_evidence_digests[2]


def test_chunked_timestamped_replay_matches_one_shot_after_state_restore() -> None:
    one_shot = run_causal_rls_replay(
        make_book(),
        (1, 2, 4),
        ("t1", "t2", "t3"),
        plan=plan(),
        learner=make_learner(),
        detector=detector(),
        model_config=CausalRLSConfig(model_id="numeric"),
    )
    first = run_causal_rls_replay(
        make_book(),
        (1, 2),
        ("t1", "t2"),
        plan=plan(),
        learner=make_learner(),
        detector=detector(),
        model_config=CausalRLSConfig(model_id="numeric"),
    )
    second = run_causal_rls_replay(
        make_book(),
        (4,),
        ("t3",),
        plan=plan(),
        learner=make_learner(),
        detector=detector(),
        model_config=CausalRLSConfig(model_id="numeric"),
        model_state=first.state.model_state,
        initial_state=first.state,
    )

    np.testing.assert_allclose(
        np.r_[first.predictions, second.predictions], one_shot.predictions
    )
    np.testing.assert_allclose(
        np.r_[first.positions, second.positions], one_shot.positions
    )
    np.testing.assert_allclose(
        np.r_[first.forgetting_factors, second.forgetting_factors],
        one_shot.forgetting_factors,
    )
    assert second.prediction_start == 2
    assert second.all_predictions == one_shot.all_predictions
    assert second.state.model_state == one_shot.state.model_state


def test_missing_label_is_settled_without_an_rls_update_or_pending_leak() -> None:
    result = run_causal_rls_replay(
        make_book(missing=True),
        (1, 4),
        ("t1", "t2"),
        plan=plan(),
        learner=make_learner(),
        model_config=CausalRLSConfig(model_id="numeric"),
    )
    assert result.replay.resolutions[0].status is ReplayStatus.MISSING
    assert result.replay.resolutions[0].reason == "label status is missing"
    assert result.pending_target_ids == ("t2",)


def test_strict_feature_plan_rejects_future_event_and_missing_membership() -> None:
    with pytest.raises(ValueError, match="event_time cannot be after"):
        observation("future", 3, 1.0, event_time=4)

    no_membership = AsOfBook(observations=(observation("m1", 1, 1.0),))
    with pytest.raises(CausalNumericError, match="not in the point-in-time universe"):
        run_causal_rls_replay(
            no_membership,
            (1,),
            ("t1",),
            plan=plan(),
            learner=make_learner(),
            model_config=CausalRLSConfig(model_id="membership"),
        )


def test_detector_forgetting_acceleration_requires_an_approved_certificate() -> None:
    neutral = run_causal_rls_replay(
        make_book(),
        (1, 2, 4),
        ("t1", "t2", "t3"),
        plan=plan(),
        learner=make_learner(),
        detector=ReadyDetector(),
        model_config=CausalRLSConfig(model_id="neutral"),
    )
    assert all(not step.acceleration_authorized for step in neutral.steps)
    assert all(step.forgetting_factor == 0.99 for step in neutral.steps)

    certificate = CalibrationCertificate(
        detector_identity=ReadyDetector.config_identity,
        null_config_identity="declared-null:v1",
        trials=100,
        horizon=16,
        false_alarm_count=0,
        false_alarm_rate=0.0,
        false_alarm_ci_high=0.03699349820698568,
        max_false_alarm_rate=0.05,
    )
    authorized = run_causal_rls_replay(
        make_book(),
        (1, 2, 4),
        ("t1", "t2", "t3"),
        plan=plan(),
        learner=make_learner(),
        detector=ReadyDetector(),
        calibration=certificate,
        model_config=CausalRLSConfig(model_id="authorized"),
    )
    assert all(step.acceleration_authorized for step in authorized.steps)
    assert all(step.forgetting_factor == 0.8 for step in authorized.steps)

    with pytest.raises(CausalNumericError, match="detector identity"):
        run_causal_rls_replay(
            make_book(),
            (1,),
            ("t1",),
            plan=plan(),
            learner=make_learner(),
            detector=ReadyDetector(),
            calibration=CalibrationCertificate(
                detector_identity="wrong-detector",
                null_config_identity="declared-null:v1",
                trials=100,
                horizon=16,
                false_alarm_count=0,
                false_alarm_rate=0.0,
                false_alarm_ci_high=0.03699349820698568,
                max_false_alarm_rate=0.05,
            ),
            model_config=CausalRLSConfig(model_id="wrong"),
        )


def test_causal_telemetry_rejects_a_malformed_topology_digest() -> None:
    with pytest.raises(CausalNumericError, match="topology evidence digest"):
        run_causal_rls_replay(
            make_book(),
            (1,),
            ("t1",),
            plan=plan(),
            learner=make_learner(),
            detector=MalformedDigestDetector(),
            model_config=CausalRLSConfig(model_id="bad-digest"),
        )


def test_factor_bounds_are_checked_before_label_settlement() -> None:
    with pytest.raises(CausalNumericError, match="learner lambda bounds"):
        run_causal_rls_replay(
            make_book(),
            (1,),
            ("t1",),
            plan=plan(),
            learner=make_learner(),
            model_config=CausalRLSConfig(
                model_id="bad-factor", default_forgetting_factor=0.5
            ),
        )


def test_invalid_learner_output_cannot_create_a_position() -> None:
    result = run_causal_rls_replay(
        make_book(),
        (1,),
        ("t1",),
        plan=plan(),
        learner=NonFiniteLearner(),
        model_config=CausalRLSConfig(model_id="non-finite"),
    )
    assert np.isnan(result.predictions[0])
    assert result.positions.tolist() == [0.0]
    assert result.replay.predictions[0].status is ReplayStatus.INVALID
