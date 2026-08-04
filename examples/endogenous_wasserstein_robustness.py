"""Synthetic diagnostic for endogenous Wasserstein robustness."""

from __future__ import annotations

import json

import numpy as np

from topology_gate.wasserstein import (
    EndogenousWassersteinLinearLearner,
    WassersteinRobustConfig,
)


def main() -> None:
    rng = np.random.default_rng(43)
    learner = EndogenousWassersteinLinearLearner(
        WassersteinRobustConfig(
            n_features=2,
            learning_rate=0.03,
            radius_floor=0.02,
            radius_sensitivity=0.04,
            radius_max=0.5,
            gradient_clip=3.0,
        )
    )
    radii = []
    absolute_losses = []
    for step in range(160):
        x = (1.0, float(rng.normal()))
        target = 0.5 + (0.8 if step < 100 else 1.8) * x[1]
        target += float(rng.normal(scale=0.05))
        score = 0.2 if step < 100 else 5.0
        update = learner.observe(x, target, score)
        radii.append(update.radius)
        absolute_losses.append(update.empirical_absolute_loss)
    print(
        json.dumps(
            {
                "kind": "endogenous_wasserstein_synthetic",
                "steps": 160,
                "shift": 100,
                "stable_radius": radii[0],
                "shift_radius": radii[-1],
                "mean_absolute_loss": float(np.mean(absolute_losses)),
                "coefficients": list(learner.coefficients),
                "note": "bounded-loss control diagnostic; not market evidence",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

