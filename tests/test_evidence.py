"""Adversarial tests for the frozen promotion evidence ledger."""

from __future__ import annotations

import pytest

from topology_gate.evidence import EvidenceLedger
from topology_gate.promotion import PromotionGate


def _ledger(*, burn_in: int = 0) -> EvidenceLedger:
    gate = PromotionGate(alpha=0.1, eta=0.5)
    gate.register_challenger("candidate")
    return EvidenceLedger(gate, "candidate", burn_in=burn_in)


def _record(ledger: EvidenceLedger, index: int) -> None:
    ledger.record_prediction(
        prediction_id=f"p-{index}",
        decision_step=index,
        label_available_step=index + 2,
        challenger_prediction=0.2,
        incumbent_prediction=0.1,
        model_fingerprint="model:v1",
        feature_fingerprint="features:v1",
    )


def test_burn_in_and_resolution_feed_only_frozen_evidence() -> None:
    ledger = _ledger(burn_in=1)
    _record(ledger, 0)
    _record(ledger, 1)

    warmup = ledger.resolve_label(
        prediction_id="p-0",
        label_id="y-0",
        label_available_step=2,
        challenger_utility=1.0,
        incumbent_utility=0.0,
    )
    assert warmup.burn_in
    assert warmup.decision is None

    settled = ledger.resolve_label(
        prediction_id="p-1",
        label_id="y-1",
        label_available_step=3,
        challenger_utility=1.0,
        incumbent_utility=0.0,
        metadata={"nested": {"api_key": "secret"}},
    )
    assert settled.accepted
    assert settled.decision is not None
    assert ledger.resolved_count == 2
    assert ledger.pending_prediction_ids == ()
    exported = settled.to_dict()
    assert exported["decision"]["audit_record"]["metadata"]["nested"]["api_key"] == "[REDACTED]"
    state = ledger.state_dict()
    assert state["resolution_count"] == 2
    assert len(state["resolutions"]) == 2


def test_ledger_rejects_duplicates_reordering_early_labels_and_eta_override() -> None:
    ledger = _ledger()
    _record(ledger, 0)
    with pytest.raises(ValueError, match="strictly increasing"):
        ledger.record_prediction(
            prediction_id="p-1",
            decision_step=0,
            label_available_step=3,
            challenger_prediction=0.0,
            incumbent_prediction=0.0,
            model_fingerprint="m",
            feature_fingerprint="f",
        )
    with pytest.raises(ValueError, match="before its declared"):
        ledger.resolve_label(
            prediction_id="p-0",
            label_id="y-0",
            label_available_step=1,
            challenger_utility=0.0,
            incumbent_utility=0.0,
        )

    ledger.resolve_label(
        prediction_id="p-0",
        label_id="y-0",
        label_available_step=2,
        challenger_utility=0.0,
        incumbent_utility=0.0,
    )
    with pytest.raises(ValueError, match="already been resolved"):
        ledger.resolve_label(
            prediction_id="p-0",
            label_id="y-duplicate",
            label_available_step=3,
            challenger_utility=0.0,
            incumbent_utility=0.0,
        )


def test_ledger_requires_registered_challenger_and_rejects_non_json_metadata() -> None:
    gate = PromotionGate()
    with pytest.raises(ValueError, match="registered"):
        EvidenceLedger(gate, "missing")
    gate.register_challenger("candidate")
    ledger = EvidenceLedger(gate, "candidate")
    _record(ledger, 0)
    with pytest.raises(ValueError, match="JSON-safe"):
        ledger.resolve_label(
            prediction_id="p-0",
            label_id="y-0",
            label_available_step=2,
            challenger_utility=0.0,
            incumbent_utility=0.0,
            metadata={"bad": object()},
        )
