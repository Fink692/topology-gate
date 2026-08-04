"""Run a bounded causal transport-replay synthetic experiment.

The experiment uses delayed labels and prefix-only state estimates.  The
transport path is a location-shift plus linear-parameter correction; it is a
prototype for the proposed replay idea, not an adapted-Wasserstein result.
"""

from __future__ import annotations

import json
from collections import deque
from typing import Any

import numpy as np

from topology_gate import RLS, RLSConfig
from topology_gate.transport import CausalTransportReplay, TransportReplayConfig

N_STEPS = 256
CHANGE_POINTS = (64, 128, 192)
SEEDS = (11, 17, 23, 29)
LABEL_DELAY = 2


def _weighted_ridge(
    features: np.ndarray[Any, Any],
    labels: np.ndarray[Any, Any],
    weights: np.ndarray[Any, Any],
    ridge: float = 1.0,
) -> np.ndarray[Any, Any] | None:
    if features.shape[0] == 0:
        return None
    diagonal = np.asarray(weights, dtype=float)[:, None]
    gram = features.T @ (diagonal * features) + ridge * np.eye(features.shape[1])
    rhs = features.T @ (weights * labels)
    return np.linalg.solve(gram, rhs)


def _run(seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    theta_by_regime = np.array(
        [[1.0, -0.5], [-0.6, 1.1], [0.8, 0.7], [-1.0, -0.4]], dtype=float
    )
    location_by_regime = np.array(
        [[0.0, 0.0], [1.0, -0.5], [-0.75, 0.8], [0.5, 1.0]], dtype=float
    )
    features = np.empty((N_STEPS, 2), dtype=float)
    labels = np.empty(N_STEPS, dtype=float)
    true_theta = np.empty((N_STEPS, 2), dtype=float)
    boundaries = (0, *CHANGE_POINTS, N_STEPS)
    for regime, (start, stop) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        features[start:stop] = rng.normal(
            location_by_regime[regime], 0.25, size=(stop - start, 2)
        )
        true_theta[start:stop] = theta_by_regime[regime]
        labels[start:stop] = np.sum(
            features[start:stop] * true_theta[start:stop], axis=1
        ) + rng.normal(0.0, 0.08, size=stop - start)

    raw = RLS(
        RLSConfig(
            n_features=2,
            ridge=1.0,
            forgetting_factor=0.98,
            lambda_min=0.8,
            lambda_max=1.0,
        )
    )
    replay = CausalTransportReplay(
        TransportReplayConfig(
            n_features=2,
            drift_sensitivity=0.75,
            location_sensitivity=0.50,
            minimum_weight=0.02,
        )
    )
    pending: deque[tuple[int, np.ndarray[Any, Any], float]] = deque()
    raw_predictions = np.empty(N_STEPS, dtype=float)
    transported_predictions = np.empty(N_STEPS, dtype=float)
    feature_history: list[np.ndarray[Any, Any]] = []

    for step in range(N_STEPS):
        while pending and pending[0][0] <= step:
            _, past_features, past_label = pending.popleft()
            raw.update(past_features, past_label, forgetting_factor=0.98)
        current_theta = np.asarray(raw.theta, dtype=float)
        location = (
            np.mean(np.asarray(feature_history), axis=0)
            if feature_history
            else np.zeros(2, dtype=float)
        )
        replay.observe_state(step, current_theta, feature_location=location)
        current_features = features[step]
        raw_predictions[step] = float(raw.predict(current_features))
        batch = replay.batch(step)
        transported_theta = _weighted_ridge(
            batch.features, batch.labels, batch.weights
        )
        transported_predictions[step] = (
            raw_predictions[step]
            if transported_theta is None
            else float(current_features @ transported_theta)
        )

        available = step + LABEL_DELAY
        pending.append((available, current_features.copy(), float(labels[step])))
        replay.append(
            step,
            available,
            current_features,
            labels[step],
            current_theta,
            feature_location=location,
        )
        feature_history.append(current_features.copy())

    raw_loss = (raw_predictions - labels) ** 2
    transported_loss = (transported_predictions - labels) ** 2
    holdout = slice(CHANGE_POINTS[-1], N_STEPS)
    return {
        "seed": seed,
        "raw_mse": float(np.mean(raw_loss)),
        "transported_mse": float(np.mean(transported_loss)),
        "raw_final_regime_mse": float(np.mean(raw_loss[holdout])),
        "transported_final_regime_mse": float(np.mean(transported_loss[holdout])),
        "transported_improvement_final_regime": float(
            np.mean(raw_loss[holdout]) - np.mean(transported_loss[holdout])
        ),
        "replay_records": replay.record_count,
        "replay_states": replay.state_count,
        "replay_identity": replay.identity,
    }


def main() -> None:
    rows = [_run(seed) for seed in SEEDS]
    print(
        json.dumps(
            {
                "kind": "causal_transport_replay_synthetic",
                "n_steps": N_STEPS,
                "change_points": list(CHANGE_POINTS),
                "label_delay": LABEL_DELAY,
                "holdout": "final regime [192, 256)",
                "rows": rows,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
