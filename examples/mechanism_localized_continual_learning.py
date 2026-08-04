"""Synthetic diagnostic for mechanism-localized continual learning.

The second mechanism changes after a known point.  The diagnostic checks that
the detector flags that module and freezes the stable module during the
localized shift.  This is a control-layer experiment, not causal evidence.
"""

from __future__ import annotations

import json

import numpy as np

from topology_gate.mechanisms import (
    MechanismLocalizedConfig,
    MechanismLocalizedRLS,
    MechanismSpec,
)

N_STEPS = 180
SHIFT = 110


def main() -> None:
    rng = np.random.default_rng(19)
    config = MechanismLocalizedConfig(
        mechanisms=(
            MechanismSpec("volatility", (0, 1)),
            MechanismSpec("transmission", (0, 2)),
        ),
        ridge=1.0,
        stable_forgetting_factor=0.995,
        shift_forgetting_factor=0.75,
        residual_history=48,
        minimum_history=24,
        residual_scale_floor=0.01,
        drift_threshold=4.0,
    )
    model = MechanismLocalizedRLS(config)
    before = None
    first_shift_frozen = None
    localized_steps = 0
    rows = []
    for step in range(N_STEPS):
        x1 = float(rng.normal())
        x2 = float(rng.normal())
        features = (1.0, x1, x2)
        target_a = 0.4 + 0.8 * x1 + float(rng.normal(scale=0.02))
        coefficient_b = 0.6 if step < SHIFT else 2.0
        target_b = 0.2 + coefficient_b * x2 + float(rng.normal(scale=0.02))
        update = model.observe(
            features,
            {"volatility": target_a, "transmission": target_b},
        )
        if step == SHIFT - 1:
            before = model.learners["volatility"].state_dict()
        if step == SHIFT:
            first_shift_frozen = before == model.learners["volatility"].state_dict()
        if step >= SHIFT and update.shifted_mechanisms == ("transmission",):
            localized_steps += 1
        if update.shifted_mechanisms:
            rows.append(
                {
                    "step": step,
                    "shifted_mechanisms": list(update.shifted_mechanisms),
                    "updated_mechanisms": list(update.updated_mechanisms),
                }
            )
    if before is None:
        raise RuntimeError("pre-shift checkpoint was not captured")
    print(
        json.dumps(
            {
                "kind": "mechanism_localized_continual_learning_synthetic",
                "steps": N_STEPS,
                "shift": SHIFT,
                "localized_steps": localized_steps,
                "first_shift": rows[0] if rows else None,
                "stable_module_digest_after": model.learners["volatility"].state_dict()["coefficients"],
                "stable_module_frozen_during_first_localized_shift": first_shift_frozen,
                "note": "synthetic control diagnostic; not causal market evidence",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
