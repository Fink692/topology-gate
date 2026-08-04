"""Adversarial tests for authenticated causal state checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from topology_gate.checkpoint import (
    CheckpointCompatibilityError,
    CheckpointEnvelope,
    CheckpointError,
    CheckpointIntegrityError,
    checkpoint_from_components,
    load_checkpoint,
    restore_component_states,
    save_checkpoint,
)
from topology_gate.promotion import PromotionStateMachine
from topology_gate.rls import RLS
from topology_gate.topology import RollingTopologyDetector, TopologyConfig


def test_hmac_envelope_round_trip_and_tamper_rejection() -> None:
    key = b"checkpoint-test-key-0123456789"
    envelope = CheckpointEnvelope.create(
        package_version="0.1.0",
        config_fingerprint="cfg",
        backend_identity="graph:v1",
        dependency_fingerprint="deps",
        online_state={"next_step": 3},
        hmac_key=key,
    )
    restored = CheckpointEnvelope.from_json(
        envelope.to_json(),
        hmac_key=key,
        expected_config_fingerprint="cfg",
    )
    assert restored.online_state == {"next_step": 3}
    assert restored.integrity_algorithm == "hmac-sha256"

    tampered = json.loads(envelope.to_json())
    tampered["online_state"]["next_step"] = 4
    with pytest.raises(CheckpointIntegrityError):
        CheckpointEnvelope.from_dict(tampered, hmac_key=key)
    with pytest.raises(CheckpointCompatibilityError):
        CheckpointEnvelope.from_json(
            envelope.to_json(),
            hmac_key=key,
            expected_config_fingerprint="other",
        )
    with pytest.raises(CheckpointIntegrityError):
        CheckpointEnvelope.from_json(envelope.to_json())


def test_checkpoint_file_is_atomic_and_root_constrained(tmp_path: Path) -> None:
    envelope = CheckpointEnvelope.create(
        package_version="0.1.0",
        config_fingerprint="cfg",
        backend_identity="graph:v1",
        dependency_fingerprint="deps",
        online_state={"next_step": 1},
        allow_untrusted=True,
    )
    destination = save_checkpoint(envelope, tmp_path / "run.json", approved_root=tmp_path)
    assert load_checkpoint(destination, allow_untrusted=True).online_state == {"next_step": 1}
    with pytest.raises(CheckpointError):
        save_checkpoint(envelope, tmp_path.parent / "escape.json", approved_root=tmp_path)


def test_component_state_is_validated_detached_before_restore() -> None:
    detector = RollingTopologyDetector(
        TopologyConfig(
            embedding_dim=1,
            cloud_window=8,
            graph_neighbors=2,
            n_eigenvalues=2,
            min_points=3,
            calibration_window=8,
            calibration_min_periods=2,
        )
    )
    for value in ([0.0], [0.1], [0.2], [0.3], [0.4], [0.5]):
        detector.observe(value)
    learner = RLS(1, ridge=1.0, forgetting_factor=0.95)
    learner.update([1.0], 0.5)
    promotion = PromotionStateMachine("candidate", alpha=0.1, eta=0.5)
    promotion.observe_score(0.25)

    envelope = checkpoint_from_components(
        package_version="0.1.0",
        config_fingerprint="cfg",
        backend_identity=detector.backend_identity,
        dependency_fingerprint="deps",
        learner=learner,
        detector=detector,
        online_state={"next_step": 6},
        promotion=promotion,
        hmac_key=b"checkpoint-test-key-0123456789",
    )
    restored = restore_component_states(
        envelope,
        learner=learner,
        detector=detector,
        promotion=promotion,
        hmac_key=b"checkpoint-test-key-0123456789",
        expected_package_version="0.1.0",
        expected_config_fingerprint="cfg",
        expected_backend_identity=detector.backend_identity,
        expected_dependency_fingerprint="deps",
    )
    assert restored["learner"].state_dict() == learner.state_dict()
    assert restored["detector_state"] == detector.stream_state_dict()
    assert restored["promotion"].state_dict() == promotion.state_dict()


def test_component_restore_rejects_untrusted_or_unmatched_envelopes() -> None:
    with pytest.raises(CheckpointIntegrityError, match="hmac_key"):
        CheckpointEnvelope.create(
            package_version="0.1.0",
            config_fingerprint="cfg",
            backend_identity="graph:v1",
            dependency_fingerprint="deps",
        )
    envelope = CheckpointEnvelope.create(
        package_version="0.1.0",
        config_fingerprint="cfg",
        backend_identity="graph:v1",
        dependency_fingerprint="deps",
        allow_untrusted=True,
    )
    with pytest.raises(CheckpointIntegrityError, match="HMAC"):
        restore_component_states(
            envelope,
            hmac_key=None,
            expected_package_version="0.1.0",
            expected_config_fingerprint="cfg",
            expected_backend_identity="graph:v1",
            expected_dependency_fingerprint="deps",
        )

    authenticated = CheckpointEnvelope.create(
        package_version="0.1.0",
        config_fingerprint="cfg",
        backend_identity="graph:v1",
        dependency_fingerprint="deps",
        hmac_key=b"checkpoint-test-key-0123456789",
    )
    with pytest.raises(CheckpointCompatibilityError, match="config"):
        restore_component_states(
            authenticated,
            hmac_key=b"checkpoint-test-key-0123456789",
            expected_package_version="0.1.0",
            expected_config_fingerprint="other",
            expected_backend_identity="graph:v1",
            expected_dependency_fingerprint="deps",
        )
