"""Run the strict synthetic walk-forward control-layer comparison.

This is a bounded methodological experiment, not market or economic evidence.
The data-generating process exposes the true binary regime direction, the
detector is calibrated on a separate AR(1) source, labels arrive one step late,
and the final regime is reported as a held-out post-shift diagnostic.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from topology_gate import (
    RLS,
    MeanCovarianceCUSUM,
    MeanCovarianceCUSUMConfig,
    OnlineRunConfig,
    RLSConfig,
    generate_synthetic_regimes,
    run_recursive_rls,
)
from topology_gate.calibration import (
    CalibrationConfig,
    StationaryBlockBootstrap,
    calibrate_threshold,
)
from topology_gate.persistent import (
    PersistentLaplacianBackend,
    PersistentLaplacianConfig,
)
from topology_gate.pl_cusum import PersistentCUSUMConfig, PersistentLaplacianCUSUM

N_STEPS = 256
CHANGE_POINTS = (64, 128, 192)
SEEDS = (11, 17, 23, 29)


class RollingWindowRLS:
    """Small fixed-window learner used only for the declared benchmark."""

    lambda_min = 1.0
    lambda_max = 1.0
    n_features = 2

    def __init__(self, window: int = 32, ridge: float = 1.0) -> None:
        self.window = window
        self.ridge = ridge
        self._rows: list[np.ndarray[Any, Any]] = []
        self._labels: list[float] = []
        self._theta = np.zeros(self.n_features, dtype=float)

    def reset(self) -> None:
        self._rows = []
        self._labels = []
        self._theta = np.zeros(self.n_features, dtype=float)

    def predict(self, features: Any) -> float:
        return float(np.asarray(features, dtype=float) @ self._theta)

    def update(self, features: Any, target: Any, forgetting_factor: Any = None) -> None:
        del forgetting_factor
        row = np.asarray(features, dtype=float).reshape(-1)
        self._rows.append(np.array(row, copy=True))
        self._labels.append(float(target))
        self._rows = self._rows[-self.window :]
        self._labels = self._labels[-self.window :]
        matrix = np.vstack(self._rows)
        gram = matrix.T @ matrix + self.ridge * np.eye(self.n_features)
        self._theta = np.linalg.solve(gram, matrix.T @ np.asarray(self._labels))


def _calibration_source() -> np.ndarray:
    rng = np.random.default_rng(20260804)
    values = np.empty((512, 2), dtype=float)
    innovations = rng.normal(0.0, 0.25, size=values.shape)
    values[0] = innovations[0]
    for index in range(1, len(values)):
        values[index] = 0.85 * values[index - 1] + innovations[index]
    return values


def _detector_factory(threshold: float) -> PersistentLaplacianCUSUM:
    backend = PersistentLaplacianBackend(
        PersistentLaplacianConfig(
            max_vertices=4,
            max_simplices=100,
            q=0,
            n_eigenvalues=4,
        )
    )
    return PersistentLaplacianCUSUM(
        backend,
        PersistentCUSUMConfig(
            cloud_window=4,
            min_points=4,
            backend_eigenvalues=4,
            positive_spectrum_width=2,
            betti_dimensions=(0, 1),
            calibration_window=8,
            calibration_min_periods=8,
            drift=0.5,
            forgetting_lambda_min=0.8,
            forgetting_lambda_max=0.99,
            threshold=threshold,
        ),
    )


def _cpd_factory(threshold: float) -> MeanCovarianceCUSUM:
    return MeanCovarianceCUSUM(
        MeanCovarianceCUSUMConfig(
            n_features=2,
            block_window=8,
            threshold=threshold,
            forgetting_lambda_min=0.8,
            forgetting_lambda_max=0.99,
            forgetting_sensitivity=1.0,
        )
    )


def _calibrate_detector() -> Any:
    source = _calibration_source()
    factory = StationaryBlockBootstrap(
        source,
        block_length=16,
        source_id="ar1-surrogate:v1",
    )
    result = calibrate_threshold(
        _detector_factory,
        factory,
        StationaryBlockBootstrap(
            source,
            block_length=16,
            source_id="ar1-surrogate:v1",
        ),
        detector_family_identity="persistent-laplacian-cusum-family:v1",
        candidate_thresholds=(2.0, 8.0, 32.0, 128.0, 1024.0),
        calibration_config=CalibrationConfig(
            trials=64,
            horizon=64,
            n_features=2,
            seed=11,
        ),
        evaluation_config=CalibrationConfig(
            trials=64,
            horizon=64,
            n_features=2,
            seed=29,
        ),
        max_false_alarm_rate=0.15,
    )
    if not result.approved:
        raise RuntimeError("separate synthetic detector calibration was not approved")
    return result


def _calibrate_cpd() -> Any:
    source = _calibration_source()
    result = calibrate_threshold(
        _cpd_factory,
        StationaryBlockBootstrap(
            source,
            block_length=16,
            source_id="ar1-surrogate:v1",
        ),
        StationaryBlockBootstrap(
            source,
            block_length=16,
            source_id="ar1-surrogate:v1",
        ),
        detector_family_identity="mean-covariance-cusum-family:v1",
        candidate_thresholds=(2.0, 4.0, 8.0, 16.0, 32.0),
        calibration_config=CalibrationConfig(
            trials=64,
            horizon=64,
            n_features=2,
            seed=13,
        ),
        evaluation_config=CalibrationConfig(
            trials=64,
            horizon=64,
            n_features=2,
            seed=37,
        ),
        max_false_alarm_rate=0.15,
    )
    if not result.approved:
        raise RuntimeError("separate synthetic mean/covariance calibration was not approved")
    return result


def _run_one(
    name: str,
    seed: int,
    *,
    pl_threshold: float | None = None,
    pl_certificate: Any | None = None,
    cpd_threshold: float | None = None,
    cpd_certificate: Any | None = None,
) -> dict[str, Any]:
    dataset = generate_synthetic_regimes(
        n_steps=N_STEPS,
        n_features=2,
        change_points=CHANGE_POINTS,
        seed=seed,
        label_delay=1,
        feature_noise=0.20,
        return_noise=0.0,
        signal_strength=1.0,
        return_magnitude=0.0,
    )
    if name == "static-rls":
        learner = RLS(
            RLSConfig(
                n_features=2,
                ridge=1.0,
                forgetting_factor=1.0,
                lambda_min=0.8,
                lambda_max=1.0,
            )
        )
        detector = None
    elif name == "exponential-rls":
        learner = RLS(
            RLSConfig(
                n_features=2,
                ridge=1.0,
                forgetting_factor=0.97,
                lambda_min=0.8,
                lambda_max=0.97,
            )
        )
        detector = None
    elif name == "rolling-rls":
        learner = RollingWindowRLS(window=32)
        detector = None
    elif name == "standard-cpd-rls":
        if cpd_threshold is None or cpd_certificate is None:
            raise ValueError("standard CPD run requires threshold and certificate")
        learner = RLS(
            RLSConfig(
                n_features=2,
                ridge=1.0,
                forgetting_factor=0.99,
                lambda_min=0.8,
                lambda_max=0.99,
            )
        )
        detector = _cpd_factory(cpd_threshold)
    elif name == "certified-pl-rls":
        if pl_threshold is None or pl_certificate is None:
            raise ValueError("certified PL run requires threshold and certificate")
        learner = RLS(
            RLSConfig(
                n_features=2,
                ridge=1.0,
                forgetting_factor=0.99,
                lambda_min=0.8,
                lambda_max=0.99,
            )
        )
        detector = _detector_factory(pl_threshold)
    else:
        raise ValueError(f"unknown system: {name}")

    result = run_recursive_rls(
        dataset.features.values,
        dataset.labels.values,
        realized_returns=dataset.realized_returns,
        learner=learner,
        detector=detector,
        config=OnlineRunConfig(label_delay=1, transaction_cost_bps=0.0),
        shift_points=CHANGE_POINTS,
        calibration=(
            cpd_certificate
            if name == "standard-cpd-rls"
            else pl_certificate
            if name == "certified-pl-rls"
            else None
        ),
    )
    loss = (result.predictions - dataset.labels.values) ** 2
    holdout = loss[CHANGE_POINTS[-1] :]
    return {
        "system": name,
        "seed": seed,
        "mse": float(np.mean(loss)),
        "post_shift_mse": tuple(
            float(np.mean(loss[start:stop]))
            for start, stop in zip((64, 128, 192), (128, 192, N_STEPS))
        ),
        "held_out_final_regime_mse": float(np.mean(holdout)),
        "mean_detection_delay": float(result.metrics["mean_detection_delay"]),
        "false_alarm_count": float(result.metrics["false_alarm_count"]),
        "accelerated_forgetting_count": float(
            result.metrics["accelerated_forgetting_count"]
        ),
    }


def main() -> None:
    calibration = _calibrate_detector()
    cpd_calibration = _calibrate_cpd()
    threshold = float(calibration.selected_threshold)
    certificate = calibration.to_certificate()
    cpd_threshold = float(cpd_calibration.selected_threshold)
    cpd_certificate = cpd_calibration.to_certificate()
    rows = [
        _run_one(
            system,
            seed,
            pl_threshold=threshold,
            pl_certificate=certificate,
            cpd_threshold=cpd_threshold,
            cpd_certificate=cpd_certificate,
        )
        for system in (
            "static-rls",
            "rolling-rls",
            "exponential-rls",
            "standard-cpd-rls",
            "certified-pl-rls",
        )
        for seed in SEEDS
    ]
    summary: dict[str, Any] = {
        "kind": "synthetic_strict_walk_forward",
        "n_steps": N_STEPS,
        "change_points": list(CHANGE_POINTS),
        "seeds": list(SEEDS),
        "label_delay": 1,
        "holdout": "final regime [192, 256)",
        "calibration_identity": calibration.identity,
        "certificate_identity": certificate.identity,
        "selected_threshold": threshold,
        "cpd_calibration_identity": cpd_calibration.identity,
        "cpd_certificate_identity": cpd_certificate.identity,
        "cpd_selected_threshold": cpd_threshold,
        "systems": rows,
    }
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
