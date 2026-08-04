"""Tests for heavy-tail-aware full-information expert allocation."""

from __future__ import annotations

import pytest

from topology_gate.experts import (
    HeavyTailExpertAllocator,
    HeavyTailExpertConfig,
    catoni_mean,
)


def test_catoni_mean_limits_one_large_outlier() -> None:
    estimate = catoni_mean([0.0, 0.0, 0.0, 1_000.0], scale=1.0)
    assert abs(estimate) < 2.0
    assert estimate < 250.0


def test_allocator_applies_switching_penalty_and_selects_next_expert() -> None:
    allocator = HeavyTailExpertAllocator(
        HeavyTailExpertConfig(
            expert_ids=("slow", "fast"),
            catoni_scale=0.5,
            switching_cost=0.2,
        )
    )
    first = allocator.observe([0.0, 1.0])
    second = allocator.observe([1.0, 0.0])
    third = allocator.observe([3.0, 0.0])
    assert first.selected_expert == "fast"
    assert second.selected_expert == "fast"
    assert not second.switched
    assert third.selected_expert == "slow"
    assert third.switched


def test_change_point_resets_history_before_new_observation() -> None:
    allocator = HeavyTailExpertAllocator(
        HeavyTailExpertConfig(expert_ids=("a", "b"), catoni_scale=1.0)
    )
    allocator.observe([10.0, 0.0])
    decision = allocator.observe([0.0, 2.0], change_point=True)
    assert decision.reset_applied
    assert allocator.histories == ((0.0,), (2.0,))


def test_allocator_state_round_trip_and_tamper_rejection() -> None:
    config = HeavyTailExpertConfig(expert_ids=("a", "b", "c"), max_history=4)
    first = HeavyTailExpertAllocator(config)
    first.observe([0.0, 1.0, -1.0])
    first.observe([1.0, 0.0, -1.0])
    state = first.state_dict()
    second = HeavyTailExpertAllocator.from_state_dict(state, config)
    assert second.state_dict() == state
    tampered = dict(state)
    tampered["step"] = 999
    with pytest.raises(ValueError, match="identity"):
        HeavyTailExpertAllocator.from_state_dict(tampered, config)


def test_allocator_rejects_invalid_inputs_and_limits() -> None:
    with pytest.raises(ValueError, match="unique"):
        HeavyTailExpertConfig(expert_ids=("a", "a"))
    allocator = HeavyTailExpertAllocator(
        HeavyTailExpertConfig(expert_ids=("a", "b"))
    )
    with pytest.raises(ValueError, match="expert count"):
        allocator.observe([1.0])
    with pytest.raises(ValueError, match="finite"):
        allocator.observe([float("nan"), 0.0])
