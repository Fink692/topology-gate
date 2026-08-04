"""Run the finite optional-stopping null across the preregistered family."""

from __future__ import annotations

import json

import numpy as np

try:
    from examples.preregistered_pl_ridge_study import SELECTION_BUDGET
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from preregistered_pl_ridge_study import SELECTION_BUDGET
from topology_gate.calibration import (
    SelectionCalibrationConfig,
    calibrate_selection_null,
)


class RademacherSelectionFamily:
    """Stable-identity bounded null factory for the declared family."""

    identity = "rademacher-selection-null:v1"

    def __call__(
        self, rng: np.random.Generator, horizon: int, cells: int
    ) -> np.ndarray:
        return rng.choice(np.array([-1.0, 1.0]), size=(horizon, cells))


def main() -> None:
    result = calibrate_selection_null(
        RademacherSelectionFamily(),
        config=SelectionCalibrationConfig(
            budget=SELECTION_BUDGET,
            trials=1_000,
            horizon=500,
            eta=0.5,
            seed=31,
        ),
    )
    payload = result.to_dict()
    print(
        json.dumps(
            {
                key: payload[key]
                for key in (
                    "kind",
                    "score_factory_identity",
                    "selection_budget_identity",
                    "trials",
                    "horizon",
                    "cell_count",
                    "parent_alpha",
                    "cell_alpha",
                    "eta",
                    "seed",
                    "family_crossing_count",
                    "family_crossing_rate",
                    "family_crossing_ci_95",
                    "config_identity",
                )
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
