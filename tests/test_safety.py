"""Safety-boundary tests for finite data, resource caps, and audit export."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from topology_gate.config import (
    DataValidationError,
    ModelConfig,
    ResourceLimitError,
)
from topology_gate.observability import AuditEvent, AuditLog
from topology_gate.promotion import EProcess
from topology_gate.rls import MAX_RLS_FEATURES, RLS, RLSConfig
from topology_gate.synthetic import TimeIndexedFeatures, generate_synthetic_regimes
from topology_gate.topology import TopologyConfig


def test_nested_audit_secrets_are_redacted_and_export_is_atomic(tmp_path: Path) -> None:
    log = AuditLog(approved_root=tmp_path)
    log.append(
        AuditEvent(
            event_type="promotion",
            step=1,
            payload={
                "api_key": "SECRET",
                "nested": {"password": "SECRET2", "safe": 3},
            },
        )
    )
    payload = log.events()[0].to_dict()["payload"]
    assert payload == {
        "api_key": "[REDACTED]",
        "nested": {"password": "[REDACTED]", "safe": 3},
    }
    destination = log.approved_root / "audit.jsonl"
    log.to_jsonl(destination)
    exported = destination.read_text(encoding="utf-8")
    assert "SECRET" not in exported
    assert json.loads(exported)["payload"]["nested"]["safe"] == 3

    with pytest.raises(ValueError, match="escapes"):
        log.to_jsonl(tmp_path.parent / "audit.jsonl")


def test_audit_rejects_nonfinite_and_oversized_payloads(tmp_path: Path) -> None:
    log = AuditLog(approved_root=tmp_path, max_event_bytes=256)
    with pytest.raises(ValueError, match="non-finite"):
        log.append(AuditEvent("bad", 0, {"value": float("nan")}))
    with pytest.raises(ValueError, match="size limit"):
        log.append(AuditEvent("large", 0, {"value": "x" * 1000}))


def test_promotion_audit_metadata_is_redacted_at_creation() -> None:
    process = EProcess(alpha=0.1, eta=0.5)
    update = process.update(
        0.0,
        metadata={"token": "SECRET", "nested": {"password": "SECRET2"}},
    )
    assert update.audit_record.metadata == {
        "token": "[REDACTED]",
        "nested": {"password": "[REDACTED]"},
    }


def test_synthetic_and_dense_rls_boundaries_reject_bad_inputs() -> None:
    with pytest.raises(DataValidationError):
        TimeIndexedFeatures(index=(0, 1), values=np.array([[0.0], [np.nan]]))
    with pytest.raises((ValueError, ResourceLimitError)):
        generate_synthetic_regimes(n_steps=32, n_features=2, feature_noise=float("nan"))
    with pytest.raises(ValueError):
        RLS(MAX_RLS_FEATURES + 1)
    with pytest.raises(ValueError):
        RLSConfig(n_features=MAX_RLS_FEATURES + 1)


def test_callable_backend_has_a_stable_configuration_identity() -> None:
    def backend(_cloud, _count):
        return [0.0]

    config = ModelConfig(
        topology=TopologyConfig(
            embedding_dim=1,
            cloud_window=8,
            graph_neighbors=2,
            n_eigenvalues=1,
            min_points=3,
            calibration_window=8,
            calibration_min_periods=2,
            persistent_laplacian_backend=backend,
        ),
        rls=RLSConfig(n_features=1, lambda_min=0.9, lambda_max=1.0),
    )
    assert len(config.fingerprint()) == 64
    assert config.to_dict()["topology"]["persistent_laplacian_backend"]["callable"]
