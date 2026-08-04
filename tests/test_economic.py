"""Fail-closed economic-path contract tests."""

from __future__ import annotations

import pytest

from topology_gate.economic import (
    EconomicDecision,
    EconomicEvaluationConfig,
    EconomicEvaluationError,
    EconomicEvidence,
    ExecutionCost,
    RealizedReturn,
    evaluate_economic_evidence_path,
    evaluate_economic_path,
)


def decision(target: str, time: int, position: float, *, evaluated: bool = True, reason: str | None = None) -> EconomicDecision:
    return EconomicDecision(
        decision_id=f"d-{target}",
        target_id=target,
        decision_time=time,
        position=position,
        evaluated=evaluated,
        reason=reason,
    )


def realized(target: str, time: int, value: float) -> RealizedReturn:
    return RealizedReturn(
        target_id=target,
        decision_time=time,
        realization_time=time + 1,
        available_time=time + 2,
        value=value,
    )


def cost(target: str, time: int) -> ExecutionCost:
    return ExecutionCost(
        target_id=target,
        decision_time=time,
        execution_time=time,
        available_time=time,
        cost_model_id="rates-v1",
        fee_rate=0.01,
        spread_rate=0.02,
        slippage_rate=0.03,
        impact_rate=0.04,
        other_rate=0.05,
    )


def capacity_cost(target: str, time: int, capacity: float) -> ExecutionCost:
    base = cost(target, time)
    return ExecutionCost(
        target_id=base.target_id,
        decision_time=base.decision_time,
        execution_time=base.execution_time,
        available_time=base.available_time,
        cost_model_id=base.cost_model_id,
        fee_rate=base.fee_rate,
        spread_rate=base.spread_rate,
        slippage_rate=base.slippage_rate,
        impact_rate=base.impact_rate,
        other_rate=base.other_rate,
        source_revision=base.source_revision,
        capacity_limit=capacity,
    )


def test_economic_path_separates_realized_returns_and_cost_components() -> None:
    result = evaluate_economic_path(
        (decision("t1", 1, 0.5), decision("t2", 2, -0.5)),
        {"t1": realized("t1", 1, 0.10), "t2": realized("t2", 2, -0.20)},
        {"t1": cost("t1", 1), "t2": cost("t2", 2)},
        config=EconomicEvaluationConfig(evaluation_id="eval", cost_model_id="rates-v1"),
    )

    # Total rate is 0.15: turnover is 0.5 on entry and 1.0 on the flip.
    assert result.rows[0].turnover == 0.5
    assert result.rows[0].gross_return == 0.05
    assert result.rows[0].total_cost == pytest.approx(0.075)
    assert result.rows[0].net_return == pytest.approx(-0.025)
    assert result.rows[1].turnover == 1.0
    assert result.rows[1].gross_return == 0.1
    assert result.rows[1].total_cost == pytest.approx(0.15)
    assert result.rows[1].net_return == pytest.approx(-0.05)
    assert result.metrics["evaluated_count"] == 2.0
    assert result.metrics["total_net_return"] == pytest.approx(-0.075)
    assert len(result.digest) == 64


def test_abstentions_are_visible_and_cannot_hide_an_open_position() -> None:
    result = evaluate_economic_path(
        (
            decision("t1", 1, 0.0, evaluated=False, reason="topology_not_ready"),
            decision("t2", 2, 0.0, evaluated=False, reason="missing_feature"),
        ),
        {},
        {},
        config=EconomicEvaluationConfig(cost_model_id="rates-v1"),
    )
    assert result.metrics["abstained_count"] == 2.0
    assert result.net_returns == (0.0, 0.0)

    with pytest.raises(EconomicEvaluationError, match="non-zero position"):
        evaluate_economic_path(
            (decision("t1", 1, 0.5), decision("t2", 2, 0.0, evaluated=False, reason="abstain")),
            {"t1": realized("t1", 1, 0.1)},
            {"t1": cost("t1", 1)},
            config=EconomicEvaluationConfig(cost_model_id="rates-v1"),
        )


def test_missing_returns_costs_and_identity_mismatches_fail_closed() -> None:
    with pytest.raises(EconomicEvaluationError, match="realized return is missing"):
        evaluate_economic_path(
            (decision("t1", 1, 0.5),),
            {},
            {"t1": cost("t1", 1)},
            config=EconomicEvaluationConfig(cost_model_id="rates-v1"),
        )
    with pytest.raises(EconomicEvaluationError, match="execution cost is missing"):
        evaluate_economic_path(
            (decision("t1", 1, 0.5),),
            {"t1": realized("t1", 1, 0.1)},
            {},
            config=EconomicEvaluationConfig(cost_model_id="rates-v1"),
        )
    with pytest.raises(EconomicEvaluationError, match="cost model identity"):
        evaluate_economic_path(
            (decision("t1", 1, 0.5),),
            {"t1": realized("t1", 1, 0.1)},
            {"t1": cost("t1", 1)},
            config=EconomicEvaluationConfig(cost_model_id="other"),
        )


def test_invalid_cost_or_time_data_is_rejected_before_evaluation() -> None:
    with pytest.raises(EconomicEvaluationError, match="non-negative"):
        ExecutionCost(
            target_id="t1",
            decision_time=1,
            execution_time=1,
            available_time=1,
            cost_model_id="rates-v1",
            spread_rate=-0.01,
        )
    with pytest.raises(EconomicEvaluationError, match="available_time"):
        ExecutionCost(
            target_id="t1",
            decision_time=1,
            execution_time=1,
            available_time=2,
            cost_model_id="rates-v1",
        )
    with pytest.raises(EconomicEvaluationError, match="strictly increasing"):
        evaluate_economic_path(
            (decision("t1", 2, 0.1), decision("t2", 2, 0.1)),
            {"t1": realized("t1", 2, 0.1), "t2": realized("t2", 2, 0.1)},
            {"t1": cost("t1", 2), "t2": cost("t2", 2)},
            config=EconomicEvaluationConfig(cost_model_id="rates-v1"),
        )


def test_capacity_evidence_is_optional_by_default_but_strict_when_required() -> None:
    result = evaluate_economic_path(
        (decision("t1", 1, 0.5),),
        {"t1": realized("t1", 1, 0.1)},
        {"t1": capacity_cost("t1", 1, 0.75)},
        config=EconomicEvaluationConfig(
            cost_model_id="rates-v1",
            require_capacity_evidence=True,
        ),
    )
    assert result.metrics["capacity_evidence_count"] == 1.0
    assert result.metrics["capacity_utilization_max"] == pytest.approx(2.0 / 3.0)

    with pytest.raises(EconomicEvaluationError, match="capacity evidence is missing"):
        evaluate_economic_path(
            (decision("t1", 1, 0.5),),
            {"t1": realized("t1", 1, 0.1)},
            {"t1": cost("t1", 1)},
            config=EconomicEvaluationConfig(
                cost_model_id="rates-v1",
                require_capacity_evidence=True,
            ),
        )
    with pytest.raises(EconomicEvaluationError, match="exceeds supplied capacity"):
        evaluate_economic_path(
            (decision("t1", 1, 0.5),),
            {"t1": realized("t1", 1, 0.1)},
            {"t1": capacity_cost("t1", 1, 0.25)},
            config=EconomicEvaluationConfig(
                cost_model_id="rates-v1",
                require_capacity_evidence=True,
            ),
        )


def test_economic_evidence_bundle_is_digest_bound_and_selects_visible_revisions() -> None:
    early_return = RealizedReturn(
        target_id="t1",
        decision_time=1,
        realization_time=2,
        available_time=3,
        value=0.1,
        source_revision=0,
    )
    corrected_return = RealizedReturn(
        target_id="t1",
        decision_time=1,
        realization_time=2,
        available_time=5,
        value=0.2,
        source_revision=1,
    )
    early_cost = capacity_cost("t1", 1, 1.0)
    corrected_cost = ExecutionCost(
        target_id="t1",
        decision_time=1,
        execution_time=5,
        available_time=5,
        cost_model_id="rates-v1",
        fee_rate=0.02,
        source_revision=1,
        capacity_limit=1.0,
    )
    bundle = EconomicEvidence(
        "vendor-vintage:v1",
        realized_returns=(corrected_return, early_return),
        execution_costs=(corrected_cost, early_cost),
    )

    before_correction = bundle.select_at(4)
    after_correction = bundle.select_at(5)
    assert before_correction[0]["t1"].value == 0.1
    assert after_correction[0]["t1"].value == 0.2
    assert before_correction[1]["t1"].fee_rate == 0.01
    assert after_correction[1]["t1"].fee_rate == 0.02
    assert EconomicEvidence.from_json(bundle.to_json()) == bundle
    reordered = EconomicEvidence(
        "vendor-vintage:v1",
        realized_returns=(early_return, corrected_return),
        execution_costs=(early_cost, corrected_cost),
    )
    assert reordered.digest == bundle.digest

    result = evaluate_economic_evidence_path(
        (decision("t1", 1, 0.5),),
        bundle,
        4,
        config=EconomicEvaluationConfig(cost_model_id="rates-v1"),
    )
    assert result.evidence_digest == bundle.digest
    assert result.evidence_cutoff == 4
    assert result.to_dict()["evidence_cutoff"] == {"kind": "int", "value": 4}
    assert result.rows[0].realized_return == pytest.approx(0.1)

    tampered = dict(bundle.to_dict())
    tampered["digest"] = "0" * 64
    with pytest.raises(EconomicEvaluationError, match="digest"):
        EconomicEvidence.from_dict(tampered)

    unknown = dict(bundle.to_dict())
    unknown["unmodeled_vendor_field"] = "reject"
    with pytest.raises(EconomicEvaluationError, match="unknown"):
        EconomicEvidence.from_dict(unknown)

    with pytest.raises(EconomicEvaluationError, match="duplicate"):
        EconomicEvidence(
            "duplicate",
            realized_returns=(early_return, early_return),
        )


def test_economic_evidence_rejects_mixed_time_domains_before_selection() -> None:
    with pytest.raises(EconomicEvaluationError, match="one comparable time domain"):
        EconomicEvidence(
            "mixed-domains",
            realized_returns=(
                realized("numeric", 1, 0.1),
                RealizedReturn(
                    target_id="string",
                    decision_time="2026-01-01T00:00:00Z",
                    realization_time="2026-01-01T00:01:00Z",
                    available_time="2026-01-01T00:02:00Z",
                    value=0.1,
                ),
            ),
        )

def test_non_observed_return_cannot_be_disguised_as_zero() -> None:
    with pytest.raises(EconomicEvaluationError, match="non-observed"):
        RealizedReturn(
            target_id="t1",
            decision_time=1,
            realization_time=2,
            available_time=3,
            value=0.0,
            status="missing",
        )

    missing = RealizedReturn(
        target_id="t1",
        decision_time=1,
        realization_time=2,
        available_time=3,
        value=None,
        status="missing",
    )
    assert missing.value is None
    with pytest.raises(EconomicEvaluationError, match="explicitly missing"):
        evaluate_economic_path(
            (decision("t1", 1, 0.5),),
            {"t1": missing},
            {"t1": cost("t1", 1)},
            config=EconomicEvaluationConfig(cost_model_id="rates-v1"),
        )
