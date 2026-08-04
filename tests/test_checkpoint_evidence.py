"""Focused tests for optional authenticated evidence-ledger checkpoint state."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, ClassVar, Mapping

import pytest

from topology_gate.checkpoint import (
    CheckpointCompatibilityError,
    CheckpointEnvelope,
    CheckpointError,
    CheckpointIntegrityError,
    checkpoint_from_components,
    restore_component_states,
)
from topology_gate.evidence import EvidenceLedger
from topology_gate.promotion import PromotionGate

KEY = b"checkpoint-evidence-test-key-0123456789"


class _EvidenceComponent:
    factory_calls: ClassVar[int] = 0

    def __init__(self) -> None:
        self.marker = "original"

    def state_dict(self) -> Mapping[str, Any]:
        return {"marker": self.marker, "pending": ["p-1", "p-2"]}

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "_EvidenceComponent":
        cls.factory_calls += 1
        candidate = cls()
        candidate.marker = str(state["marker"])
        return candidate


def _create(**kwargs: Any) -> CheckpointEnvelope:
    return CheckpointEnvelope.create(
        package_version="0.1.0",
        config_fingerprint="cfg",
        backend_identity="graph:v1",
        dependency_fingerprint="deps",
        hmac_key=KEY,
        **kwargs,
    )


def test_evidence_state_round_trips_and_is_in_the_authenticated_payload() -> None:
    state = {"pending": [{"prediction_id": "p-1"}], "epoch": 7}
    envelope = _create(evidence_state=state)

    assert envelope.payload_dict()["evidence_state"] == state
    assert "evidence_state" in envelope.to_dict()
    restored = CheckpointEnvelope.from_json(envelope.to_json(), hmac_key=KEY)
    assert restored.evidence_state == state

    tampered = replace(envelope, evidence_state={"pending": [], "epoch": 8})
    with pytest.raises(CheckpointIntegrityError):
        tampered.verify(KEY)


def test_manifest_digest_is_authenticated_and_checked_at_restore() -> None:
    envelope = _create(manifest_digest="manifest-001")
    restored = CheckpointEnvelope.from_json(
        envelope.to_json(),
        hmac_key=KEY,
        expected_manifest_digest="manifest-001",
    )
    assert restored.manifest_digest == "manifest-001"
    with pytest.raises(CheckpointCompatibilityError):
        CheckpointEnvelope.from_json(
            envelope.to_json(),
            hmac_key=KEY,
            expected_manifest_digest="manifest-002",
        )


def test_checkpoint_from_components_accepts_evidence_component_or_state() -> None:
    component = _EvidenceComponent()
    envelope = checkpoint_from_components(
        package_version="0.1.0",
        config_fingerprint="cfg",
        backend_identity="graph:v1",
        dependency_fingerprint="deps",
        evidence=component,
        hmac_key=KEY,
    )
    assert envelope.evidence_state == component.state_dict()

    explicit = checkpoint_from_components(
        package_version="0.1.0",
        config_fingerprint="cfg",
        backend_identity="graph:v1",
        dependency_fingerprint="deps",
        evidence_state={"resolved": 3},
        hmac_key=KEY,
    )
    assert explicit.evidence_state == {"resolved": 3}

    with pytest.raises(CheckpointError, match="either evidence"):
        checkpoint_from_components(
            package_version="0.1.0",
            config_fingerprint="cfg",
            backend_identity="graph:v1",
            dependency_fingerprint="deps",
            evidence=component,
            evidence_state={"resolved": 3},
            hmac_key=KEY,
        )


def test_restore_returns_detached_evidence_state_and_component() -> None:
    _EvidenceComponent.factory_calls = 0
    envelope = _create(evidence_state={"marker": "restored", "nested": {"x": [1]}})
    supplied = _EvidenceComponent()

    restored = restore_component_states(
        envelope,
        evidence=supplied,
        hmac_key=KEY,
        expected_package_version="0.1.0",
        expected_config_fingerprint="cfg",
        expected_backend_identity="graph:v1",
        expected_dependency_fingerprint="deps",
    )

    assert _EvidenceComponent.factory_calls == 1
    assert restored["evidence"] is not supplied
    assert restored["evidence"].marker == "restored"
    assert restored["evidence_state"] == envelope.evidence_state
    assert restored["evidence_state"] is not envelope.evidence_state
    assert restored["evidence_state"]["nested"] is not envelope.evidence_state["nested"]
    assert supplied.marker == "original"


def test_restore_does_not_construct_evidence_before_compatibility_or_integrity() -> None:
    _EvidenceComponent.factory_calls = 0
    envelope = _create(evidence_state={"marker": "restored"})
    supplied = _EvidenceComponent()
    common = {
        "envelope": envelope,
        "evidence": supplied,
        "hmac_key": KEY,
        "expected_package_version": "0.1.0",
        "expected_config_fingerprint": "cfg",
        "expected_backend_identity": "graph:v1",
        "expected_dependency_fingerprint": "deps",
    }

    with pytest.raises(CheckpointCompatibilityError):
        restore_component_states(
            **{**common, "expected_config_fingerprint": "wrong"}
        )
    assert _EvidenceComponent.factory_calls == 0

    tampered = replace(envelope, evidence_state={"marker": "tampered"})
    with pytest.raises(CheckpointIntegrityError):
        restore_component_states(**{**common, "envelope": tampered})
    assert _EvidenceComponent.factory_calls == 0


def test_checkpoints_without_evidence_keep_the_legacy_wire_payload() -> None:
    envelope = _create(online_state={"next_step": 4})
    payload = envelope.to_dict()
    assert "evidence_state" not in payload

    restored = CheckpointEnvelope.from_dict(payload, hmac_key=KEY)
    assert restored.evidence_state is None
    assert restored.online_state == {"next_step": 4}


def test_evidence_state_must_be_json_safe() -> None:
    with pytest.raises(CheckpointError, match="unsupported type"):
        _create(evidence_state={"bad": object()})


def test_real_evidence_ledger_restores_against_detached_promotion_gate() -> None:
    gate = PromotionGate(alpha=0.1, eta=0.5)
    gate.register_challenger("candidate")
    ledger = EvidenceLedger(gate, "candidate")
    ledger.record_prediction(
        prediction_id="p-0",
        decision_step=0,
        label_available_step=2,
        challenger_prediction=0.2,
        incumbent_prediction=0.1,
        model_fingerprint="m:v1",
        feature_fingerprint="f:v1",
    )
    envelope = checkpoint_from_components(
        package_version="0.1.0",
        config_fingerprint="cfg",
        backend_identity="graph:v1",
        dependency_fingerprint="deps",
        promotion=gate,
        evidence=ledger,
        hmac_key=KEY,
    )

    restored = restore_component_states(
        envelope,
        promotion=gate,
        evidence=ledger,
        eta=0.5,
        hmac_key=KEY,
        expected_package_version="0.1.0",
        expected_config_fingerprint="cfg",
        expected_backend_identity="graph:v1",
        expected_dependency_fingerprint="deps",
    )

    assert restored["promotion"] is not gate
    assert restored["evidence"] is not ledger
    assert restored["evidence"].state_dict() == ledger.state_dict()
