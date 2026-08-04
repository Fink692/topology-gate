"""Synthetic finite martingale stress-bridge diagnostic."""

from __future__ import annotations

import json

from topology_gate.bridge import MartingaleStressBridge, StressPath


def main() -> None:
    paths = (
        StressPath("down", 0.0, -1.0),
        StressPath("up", 0.0, 1.0),
        StressPath("stay", 1.0, 0.0),
        StressPath("rise", 1.0, 2.0),
    )
    result = MartingaleStressBridge().fit(
        paths,
        ((-1.0, 0.35), (0.0, 0.15), (1.0, 0.35), (2.0, 0.15)),
    )
    print(
        json.dumps(
            {
                "kind": "finite_martingale_stress_bridge_synthetic",
                "converged": result.converged,
                "iterations": result.iterations,
                "entropy": result.entropy,
                "weights": [[path_id, weight] for path_id, weight in result.weights],
                "terminal_masses": [list(row) for row in result.terminal_masses],
                "drift_residuals": [list(row) for row in result.drift_residuals],
                "digest": result.digest,
                "note": "finite stress-training diagnostic; not a continuous-time bridge or market evidence",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

