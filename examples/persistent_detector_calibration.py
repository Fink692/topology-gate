"""Reproduce the finite synthetic persistent-CUSUM calibration record.

This example is deliberately a bounded synthetic surrogate.  Its approved
certificate is not market calibration, a level-alpha theorem for live data, or
evidence of economic value.
"""

from __future__ import annotations

import numpy as np

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


def _source() -> np.ndarray:
    rng = np.random.default_rng(20260804)
    values = np.empty((512, 2), dtype=float)
    innovations = rng.normal(0.0, 0.25, size=values.shape)
    values[0] = innovations[0]
    for index in range(1, values.shape[0]):
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


def main() -> None:
    source = _source()
    calibration_factory = StationaryBlockBootstrap(
        source,
        block_length=16,
        source_id="ar1-surrogate:v1",
    )
    evaluation_factory = StationaryBlockBootstrap(
        source,
        block_length=16,
        source_id="ar1-surrogate:v1",
    )
    result = calibrate_threshold(
        _detector_factory,
        calibration_factory,
        evaluation_factory,
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
        raise RuntimeError("synthetic split evaluation did not pass its declared budget")
    certificate = result.to_certificate()
    print(
        {
            "selected_threshold": result.selected_threshold,
            "calibration_false_alarm_counts": [
                item.false_alarm_count for item in result.calibration_results
            ],
            "evaluation_false_alarm_count": result.evaluation_result.false_alarm_count,
            "evaluation_false_alarm_ci_high": result.evaluation_result.false_alarm_ci_high,
            "result_identity": result.identity,
            "certificate_identity": certificate.identity,
        }
    )


if __name__ == "__main__":
    main()
