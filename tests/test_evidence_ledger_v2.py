"""Adversarial tests for the certified evidence-ledger contract."""

from __future__ import annotations

from topology_gate.evidence import EvidenceLedger, PromotionEvidenceConfig
from topology_gate.promotion import PromotionGate


def _config() -> PromotionEvidenceConfig:
    return PromotionEvidenceConfig(
        run_id="run-001",
        family_id="family-001",
        incumbent_id="incumbent",
        eta_policy_id="eta.prior-observations.v1",
        missing_label_policy_id="missing.predictable.v1",
        package_version="0.1.0",
        config_fingerprint="cfg-001",
        backend_identity="persistent.v1",
        dependency_fingerprint="deps-001",
        manifest_digest="manifest-001",
        global_alpha=0.1,
        initial_wealth=1.0,
        score_bound=1.0,
        certified=True,
    )


def _ledger(eta_policy) -> tuple[PromotionGate, EvidenceLedger]:
    gate = PromotionGate(alpha=0.1, eta=0.5)
    gate.register_challenger("candidate")
    ledger = EvidenceLedger(
        gate,
        "candidate",
        config=_config(),
        eta_policy=eta_policy,
        burn_in=0,
    )
    return gate, ledger


def _record(ledger: EvidenceLedger, index: int, *, availability: int | None = None) -> None:
    ledger.record_prediction(
        prediction_id=f"p-{index}",
        decision_step=index,
        label_available_step=index + 2 if availability is None else availability,
        challenger_prediction=0.25,
        incumbent_prediction=0.10,
        model_fingerprint="model:v1",
        feature_fingerprint="features:v1",
        target_id=f"target-{index}",
    )


def test_eta_is_resolved_before_label_and_is_frozen_in_receipt() -> None:
    contexts: list[dict[str, object]] = []

    def eta_policy(context):
        assert "label" not in context
        assert "challenger_utility" not in context
        contexts.append(dict(context))
        return 0.25 if int(context["prior_observations"]) == 0 else 0.5

    gate, ledger = _ledger(eta_policy)
    prediction = ledger.record_prediction(
        prediction_id="p-0",
        decision_step=0,
        label_available_step=2,
        challenger_prediction=0.2,
        incumbent_prediction=0.1,
        model_fingerprint="model:v1",
        feature_fingerprint="features:v1",
    )

    assert prediction.eta == 0.25
    assert prediction.eta_policy_id == "eta.prior-observations.v1"
    assert len(contexts) == 1
    resolution = ledger.resolve_label(
        prediction_id="p-0",
        label_id="y-0",
        label_available_step=2,
        challenger_utility=1.0,
        incumbent_utility=0.0,
    )
    assert resolution.decision is not None
    assert resolution.decision.eta == 0.25
    assert gate.challenger_state("candidate").observations == 1


def test_out_of_order_arrivals_settle_in_prediction_order() -> None:
    gate, ledger = _ledger(0.25)
    _record(ledger, 0, availability=2)
    _record(ledger, 1, availability=3)

    ledger.ingest_label(
        prediction_id="p-1",
        label_id="y-1",
        label_available_step=3,
        received_step=3,
        challenger_utility=1.0,
        incumbent_utility=0.0,
    )
    assert ledger.settle_ready(at_step=3) == ()
    assert gate.challenger_state("candidate").observations == 0

    ledger.ingest_label(
        prediction_id="p-0",
        label_id="y-0",
        label_available_step=2,
        received_step=4,
        challenger_utility=1.0,
        incumbent_utility=0.0,
    )
    settled = ledger.settle_ready(at_step=4)
    assert [value.prediction_id for value in settled] == ["p-0", "p-1"]
    assert [value.evidence_index for value in settled] == [0, 1]
    assert [value.decision.observation for value in settled if value.decision] == [1, 2]


def test_missing_label_is_visible_and_does_not_update_wealth() -> None:
    gate, ledger = _ledger(0.25)
    _record(ledger, 0)
    before = gate.challenger_state("candidate")

    resolution = ledger.mark_missing(
        prediction_id="p-0",
        label_id="y-0",
        label_available_step=2,
        reason="vendor timeout",
        expired=True,
    )

    after = gate.challenger_state("candidate")
    assert resolution.status == "expired"
    assert not resolution.accepted
    assert resolution.reason == "vendor timeout"
    assert after.e_value == before.e_value
    assert after.observations == before.observations == 0
    assert ledger.pending_prediction_ids == ()
    assert ledger.resolutions[0].status == "expired"


def test_state_round_trip_restores_pending_receipt_without_mutating_gate() -> None:
    gate, ledger = _ledger(0.25)
    _record(ledger, 0)
    ledger_state = ledger.state_dict()
    gate_state = gate.state_dict()

    restored_gate = PromotionGate.from_state_dict(gate_state)
    restored = EvidenceLedger.from_state_dict(
        ledger_state,
        gate=restored_gate,
        eta_policy=0.25,
    )

    assert restored.state_dict() == ledger_state
    assert restored_gate.state_dict() == gate_state
    resolution = restored.resolve_label(
        prediction_id="p-0",
        label_id="y-0",
        label_available_step=2,
        challenger_utility=1.0,
        incumbent_utility=0.0,
    )
    assert resolution.decision is not None
    assert restored_gate.challenger_state("candidate").observations == 1
