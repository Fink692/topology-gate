"""Run a bounded heavy-tail expert-allocation diagnostic.

Utilities are shadow returns observed for every expert after the current
decision. The selected expert therefore applies on the next step. This is a
full-information control-layer experiment, not market or economic evidence.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from topology_gate.experts import HeavyTailExpertAllocator, HeavyTailExpertConfig

N_STEPS = 256
CHANGE_POINTS = (128, 192)
SEEDS = (11, 17, 23, 29)
SWITCHING_COST = 0.01


def _means(step: int) -> np.ndarray:
    if step < CHANGE_POINTS[0]:
        return np.asarray((0.12, 0.07, 0.04), dtype=float)
    if step < CHANGE_POINTS[1]:
        return np.asarray((0.05, 0.12, 0.04), dtype=float)
    return np.asarray((0.04, 0.06, 0.13), dtype=float)


def _mean_choice(
    histories: list[list[float]],
    current: int | None,
) -> tuple[int, bool]:
    estimates = [float(np.mean(history)) for history in histories]
    adjusted = [
        estimate
        - (
            SWITCHING_COST
            if current is not None and index != current
            else 0.0
        )
        for index, estimate in enumerate(estimates)
    ]
    selected = max(range(len(adjusted)), key=lambda index: adjusted[index])
    return selected, current is not None and selected != current


def _run_one(seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    robust = HeavyTailExpertAllocator(
        HeavyTailExpertConfig(
            expert_ids=("expert-a", "expert-b", "expert-c"),
            catoni_scale=0.08,
            switching_cost=SWITCHING_COST,
        )
    )
    mean_histories = [[], [], []]
    robust_current: int | None = None
    mean_current: int | None = None
    robust_utility = 0.0
    mean_utility = 0.0
    robust_switches = 0
    mean_switches = 0
    for step in range(N_STEPS):
        utilities = _means(step) + 0.08 * rng.standard_t(df=2, size=3)
        robust_current = 0 if robust_current is None else robust_current
        mean_current = 0 if mean_current is None else mean_current
        robust_utility += float(utilities[robust_current])
        mean_utility += float(utilities[mean_current])
        robust_decision = robust.observe(
            utilities.tolist(), change_point=step in CHANGE_POINTS
        )
        robust_switches += int(robust_decision.switched)
        robust_current = robust_decision.selected_index
        if step in CHANGE_POINTS:
            mean_histories = [[], [], []]
        for index, utility in enumerate(utilities):
            mean_histories[index].append(float(utility))
        mean_current, switched = _mean_choice(mean_histories, mean_current)
        mean_switches += int(switched)
        if switched:
            mean_utility -= SWITCHING_COST
        if robust_decision.switched:
            robust_utility -= SWITCHING_COST
    return {
        "seed": seed,
        "robust_net_utility": robust_utility,
        "mean_net_utility": mean_utility,
        "robust_switches": robust_switches,
        "mean_switches": mean_switches,
        "robust_minus_mean": robust_utility - mean_utility,
    }


def main() -> None:
    rows = [_run_one(seed) for seed in SEEDS]
    print(
        json.dumps(
            {
                "kind": "heavy_tail_expert_allocation_synthetic",
                "n_steps": N_STEPS,
                "change_points": list(CHANGE_POINTS),
                "seeds": list(SEEDS),
                "student_t_df": 2,
                "switching_cost": SWITCHING_COST,
                "rows": rows,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
