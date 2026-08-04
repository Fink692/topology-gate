"""Tests for pre-registered selection-family alpha allocation."""

from __future__ import annotations

import numpy as np
import pytest

from topology_gate.calibration import (
    SelectionCalibrationConfig,
    calibrate_selection_null,
)
from topology_gate.selection import SelectionBudget


def test_selection_budget_allocates_and_round_trips_selected_cell() -> None:
    budget = SelectionBudget(
        "study-selection:v1",
        0.12,
        model_slots=3,
        feature_slots=4,
        eta_slots=2,
        model_index=2,
        feature_index=3,
        eta_index=1,
    )

    assert budget.total_slots == 24
    assert budget.allocated_alpha == pytest.approx(0.005)
    restored = SelectionBudget.from_state_dict(budget.state_dict())
    assert restored == budget
    assert restored.identity == budget.identity


@pytest.mark.parametrize(
    "changes",
    [
        {"model_index": 0},
        {"feature_index": 5},
        {"eta_index": 3},
        {"model_slots": 0},
        {"global_alpha": 1.0},
    ],
)
def test_selection_budget_rejects_invalid_family_or_index(changes: dict[str, object]) -> None:
    values: dict[str, object] = {
        "budget_id": "budget",
        "global_alpha": 0.1,
        "model_slots": 1,
        "feature_slots": 1,
        "eta_slots": 1,
        "model_index": 1,
        "feature_index": 1,
        "eta_index": 1,
    }
    values.update(changes)
    with pytest.raises(ValueError):
        SelectionBudget(**values)


def test_selection_budget_rejects_tampering_and_oversized_family() -> None:
    budget = SelectionBudget("budget", 0.1, model_slots=2)
    tampered = budget.state_dict()
    tampered["model_index"] = 2
    with pytest.raises(ValueError, match="identity"):
        SelectionBudget.from_state_dict(tampered)

    with pytest.raises(ValueError, match="slot limit"):
        SelectionBudget("too-large", 0.1, model_slots=1001, feature_slots=1001)


def test_selection_null_calibration_covers_the_complete_cartesian_family() -> None:
    budget = SelectionBudget(
        "null-selection",
        0.05,
        model_slots=2,
        feature_slots=2,
        eta_slots=1,
    )

    def score_factory(rng: np.random.Generator, horizon: int, cells: int) -> np.ndarray:
        return rng.choice(np.array([-1.0, 1.0]), size=(horizon, cells))

    config = SelectionCalibrationConfig(
        budget=budget,
        trials=80,
        horizon=64,
        eta=0.5,
        seed=31,
    )
    first = calibrate_selection_null(score_factory, config=config)
    second = calibrate_selection_null(score_factory, config=config)

    assert first == second
    assert first.cell_count == 4
    assert first.cell_alpha == pytest.approx(0.0125)
    assert first.family_crossing_count == sum(
        step < first.horizon for step in first.first_crossing_steps
    )
    assert first.selection_budget_identity == budget.identity
    assert first.to_dict()["config_identity"] == first.config_identity

    with pytest.raises(ValueError, match="scores shaped"):
        calibrate_selection_null(
            lambda rng, horizon, cells: np.zeros((horizon, cells - 1)),
            config=config,
        )
