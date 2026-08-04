"""Adversarial tests for the anytime-valid promotion slice."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# The workspace intentionally owns only promotion.py, so keep the test import
# path local instead of adding a package/configuration file elsewhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from topology_gate.promotion import (  # noqa: E402
    AlphaSpender,
    EProcess,
    GateStatus,
    InvalidEtaError,
    PromotionClosedError,
    PromotionError,
    PromotionGate,
    PromotionState,
    PromotionStateMachine,
    bounded_utility_difference,
    clip_utility_difference,
    geometric_alpha_allocation,
    optional_stopping_threshold,
    optional_stopping_threshold_reached,
    predictable_betting_fraction,
    validate_eta,
)


def test_utility_difference_is_clipped_and_normalized_before_betting() -> None:
    assert clip_utility_difference(10.0, 0.0) == 1.0
    assert clip_utility_difference(-10.0, 0.0) == -1.0
    assert bounded_utility_difference(10.0, 0.0) == 1.0
    assert bounded_utility_difference(-10.0, 0.0) == -1.0
    assert bounded_utility_difference(0.25, 0.0, bound=0.5) == 0.5

    process = EProcess(alpha=0.1, eta=0.5)
    update = process.update_utilities(10.0, 0.0)
    assert update.score == 1.0
    assert update.factor == 1.5
    assert update.e_value_after == 1.5
    # The raw difference is audit context only; it cannot inflate the factor.
    assert update.audit_record.unclipped_difference == 10.0


def test_eta_must_be_finite_predictable_and_in_the_safe_interval() -> None:
    for invalid in (-0.01, 1.00001, float("nan"), float("inf"), -float("inf")):
        with pytest.raises(InvalidEtaError):
            validate_eta(invalid)
        with pytest.raises(InvalidEtaError):
            EProcess(eta=invalid)

    process = EProcess(eta=lambda _history: 1.1)
    with pytest.raises(InvalidEtaError):
        process.update(0.0)


def test_predictable_fraction_uses_only_prior_scores() -> None:
    assert predictable_betting_fraction(()) == 0.0
    assert predictable_betting_fraction((1.0,), max_eta=0.8) == 0.8
    assert predictable_betting_fraction((-1.0,), max_eta=0.8) == 0.0
    assert predictable_betting_fraction((0.25, 0.25), max_eta=1.0) == 0.25

    histories: list[tuple[float, ...]] = []

    def rule(history: tuple[float, ...]) -> float:
        histories.append(history)
        return 0.5

    process = EProcess(alpha=0.1, eta=rule)
    process.update(1.0)
    process.update(-1.0)
    assert histories == [(), (1.0,)]


def test_factors_and_products_are_nonnegative_even_at_worst_score() -> None:
    process = EProcess(alpha=0.1, eta=1.0)
    first = process.update(-1.0)
    assert first.factor == 0.0
    assert first.e_value_after == 0.0
    assert process.e_value >= 0.0

    for _ in range(10):
        update = process.update(1.0)
        assert update.factor >= 0.0
        assert update.e_value_after >= 0.0


def test_optional_stopping_threshold_is_checked_at_a_crossing_not_a_fixed_horizon() -> None:
    process = EProcess(alpha=0.1, eta=0.5)
    stopped_at = None
    for index in range(1, 20):
        update = process.update(1.0)
        if update.threshold_crossed:
            stopped_at = index
            break

    assert stopped_at == 6
    assert update.first_crossing is True
    assert process.first_crossing_observation == stopped_at
    assert process.ever_crossed is True
    assert update.e_value_after >= optional_stopping_threshold(0.1)

    # Continuing the stream can lower the current e-value, but cannot erase
    # the valid first-crossing event or the state-machine promotion.
    later = process.update(-1.0)
    assert later.e_value_after < later.threshold
    assert process.ever_crossed is True
    assert optional_stopping_threshold_reached(10.0, 0.1) is True
    assert optional_stopping_threshold_reached(9.999, 0.1) is False


def test_negative_utility_does_not_promote_and_reduces_wealth() -> None:
    machine = PromotionStateMachine("negative", alpha=0.1, eta=0.5)
    for _ in range(8):
        decision = machine.observe_utilities(-2.0, 0.0)
        assert decision.score == -1.0
        assert decision.factor == 0.5
        assert decision.promoted is False

    assert machine.state is PromotionState.ACTIVE
    assert machine.e_value == 0.5**8
    assert machine.ever_crossed is False


def test_positive_bounded_utility_can_promote_at_anytime() -> None:
    machine = PromotionStateMachine("positive", alpha=0.1, eta=0.5)
    decisions = [machine.observe_utilities(3.0, 0.0) for _ in range(6)]

    assert all(decision.score == 1.0 for decision in decisions)
    assert decisions[-1].threshold_crossed is True
    assert decisions[-1].promoted is True
    assert machine.state is PromotionState.PROMOTED
    assert machine.first_crossing_observation == 6
    assert machine.e_value == pytest.approx(1.5**6)

    # A very large raw utility difference does not create a raw-return claim:
    # the evidence remains based on the bounded score only.
    assert decisions[-1].audit_record.unclipped_difference == 3.0
    assert decisions[-1].audit_record.score == 1.0


def test_reset_starts_a_new_epoch_and_preserves_audit_history() -> None:
    machine = PromotionStateMachine("resettable", alpha=0.1, eta=0.5)
    machine.observe_score(1.0)
    old_records = machine.audit_records
    reset_record = machine.reset(reason="new data collection window")

    assert machine.epoch == 1
    assert machine.state is PromotionState.ACTIVE
    assert machine.e_value == 1.0
    assert machine.observations == 0
    assert machine.ever_crossed is False
    assert reset_record.event == "reset"
    assert reset_record.epoch == 1
    assert len(machine.audit_records) == len(old_records) + 1


def test_multiple_challengers_use_explicit_geometric_alpha_control() -> None:
    alpha = 0.2
    gate = PromotionGate("incumbent", alpha=alpha, eta=1.0)
    first = gate.register_challenger("first")
    second = gate.register_challenger("second")
    third = gate.register_challenger("third")

    assert first.alpha == geometric_alpha_allocation(alpha, 1, epoch=0)
    assert second.alpha == geometric_alpha_allocation(alpha, 2, epoch=0)
    assert third.alpha == geometric_alpha_allocation(alpha, 3, epoch=0)
    assert first.alpha + second.alpha + third.alpha < alpha
    assert gate.alpha_spent == pytest.approx(first.alpha + second.alpha + third.alpha)
    assert gate.alpha_budget_remaining > 0.0

    # The first candidate's threshold is its allocated threshold, not the
    # global alpha threshold.  With eta=1, five clipped positive scores cross it.
    decision = None
    for _ in range(5):
        decision = gate.observe_score("first", 1.0)
    assert decision is not None
    assert decision.threshold == first.threshold
    assert decision.promoted is True
    assert gate.status is GateStatus.PROMOTED
    assert gate.promoted_challenger_id == "first"

    with pytest.raises(PromotionClosedError):
        gate.observe_score("second", 1.0)


def test_registration_seal_freezes_family_and_survives_checkpoint_restore() -> None:
    gate = PromotionGate("incumbent", alpha=0.2, eta=0.5)
    gate.register_challenger("first")
    gate.register_challenger("second")

    record = gate.seal_registration()

    assert gate.registration_sealed is True
    assert record.event == "registration_sealed"
    assert gate.audit_records[-1] is record
    with pytest.raises(PromotionError, match="registration is sealed"):
        gate.register_challenger("late")

    gate.observe_score("first", 0.0)
    restored = PromotionGate.from_state_dict(gate.state_dict())
    assert restored.registration_sealed is True
    assert restored.state_dict() == gate.state_dict()


def test_registration_cannot_be_sealed_after_observation() -> None:
    gate = PromotionGate("incumbent", alpha=0.2, eta=0.5)
    gate.register_challenger("first")
    gate.observe_score("first", 0.0)

    with pytest.raises(PromotionError, match="before observations"):
        gate.seal_registration()


def test_gate_reset_funds_a_new_epoch_without_reusing_old_alpha() -> None:
    alpha = 0.2
    gate = PromotionGate("incumbent", alpha=alpha, eta=1.0)
    gate.register_challenger("first")
    gate.register_challenger("second")
    for _ in range(5):
        gate.observe_score("first", 1.0)

    old_spent = gate.alpha_spent
    records = gate.reset_epoch()
    assert gate.epoch == 1
    assert gate.status is GateStatus.OPEN
    assert gate.incumbent_id == "first"
    assert gate.promoted_challenger_id is None
    assert gate.challenger_state("first").epoch == 1
    assert gate.challenger_state("first").state is PromotionState.ACTIVE
    assert gate.challenger_state("first").e_value == 1.0
    assert gate.challenger_state("first").alpha == geometric_alpha_allocation(
        alpha, 1, epoch=1
    )
    assert gate.alpha_spent > old_spent
    assert gate.alpha_spent < alpha
    assert records[-1].event == "gate_reset"
    assert gate.audit_records[-1] is records[-1]
    assert sum(record.event == "gate_reset" for record in gate.audit_records) == 1
    assert any(record.event == "reset" for record in gate.audit_records)


def test_per_challenger_reset_can_be_followed_by_a_gate_epoch_reset() -> None:
    gate = PromotionGate("incumbent", alpha=0.2, eta=0.5)
    gate.register_challenger("first")
    gate.register_challenger("second")
    gate.reset_challenger("first")
    records = gate.reset_epoch()

    assert records[-1].event == "gate_reset"
    assert gate.epoch == 2
    assert all(snapshot.epoch == 2 for snapshot in gate.snapshots())


def test_geometric_alpha_control_is_bounded_across_slots_and_epochs() -> None:
    alpha = 0.05
    spender = AlphaSpender(alpha)
    finite_sum = sum(
        spender.allocation(index, epoch=epoch)
        for index in range(1, 30)
        for epoch in range(30)
    )
    assert finite_sum < alpha
    assert spender.total_possible_allocation() == alpha


def test_module_states_the_conditional_bounded_null_explicitly() -> None:
    from topology_gate import promotion

    module_text = promotion.__doc__ or ""
    assert "conditional" in module_text.lower()
    assert "clipped" in module_text.lower()
    assert "raw-return" in module_text.lower()
