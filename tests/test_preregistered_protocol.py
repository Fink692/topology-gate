"""Tests for the frozen first-study protocol declaration."""

from __future__ import annotations

from examples.preregistered_pl_ridge_study import (
    SELECTION_BUDGET,
    STUDY_SPEC,
    build_manifest,
    protocol_state,
)


def test_protocol_has_sealed_holdout_and_role_bound_selection_budget() -> None:
    manifest = build_manifest()
    state = protocol_state()

    assert manifest.holdout_is_sealed
    assert STUDY_SPEC.embargo_steps == 5
    assert SELECTION_BUDGET.total_slots == 48
    assert state["selected_gate_alpha"] == SELECTION_BUDGET.allocated_alpha
    assert state["study_manifest_digest"] == manifest.digest
    assert state["selection_budget"]["identity"] == SELECTION_BUDGET.identity
