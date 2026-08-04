"""Tests for the finite martingale stress bridge."""

from __future__ import annotations

import pytest

from topology_gate import MartingaleStressBridge as PublicMartingaleStressBridge
from topology_gate.bridge import (
    BRIDGE_SCHEMA,
    BRIDGE_VERSION,
    MartingaleStressBridge,
    StressBridgeConfig,
    StressPath,
)


def _paths() -> tuple[StressPath, ...]:
    return (
        StressPath("down", 0.0, -1.0),
        StressPath("up", 0.0, 1.0),
        StressPath("stay", 1.0, 0.0),
        StressPath("rise", 1.0, 2.0),
    )


def test_bridge_matches_terminal_marginal_and_martingale_drift() -> None:
    bridge = MartingaleStressBridge(StressBridgeConfig(tolerance=1.0e-7))
    result = bridge.fit(
        _paths(),
        ((-1.0, 0.35), (0.0, 0.15), (1.0, 0.35), (2.0, 0.15)),
    )
    assert result.converged
    assert result.digest
    assert all(abs(actual - target) < 1.0e-6 for _, actual, target in result.terminal_masses)
    assert all(abs(residual) < 1.0e-6 for _, residual in result.drift_residuals)
    assert result.entropy >= 0.0


def test_nonzero_physical_drift_is_supported() -> None:
    bridge = MartingaleStressBridge()
    result = bridge.fit(
        _paths(),
        ((-1.0, 0.2), (0.0, 0.2), (1.0, 0.2), (2.0, 0.4)),
        drift_targets=((0.0, 0.0), (1.0, 1.0 / 3.0)),
    )
    assert result.converged
    assert abs(result.drift_residuals[1][1]) < 1.0e-6


def test_infeasible_or_malformed_stress_fails_closed() -> None:
    bridge = MartingaleStressBridge()
    with pytest.raises(ValueError, match="infeasible"):
        bridge.fit(
            _paths(),
            ((-1.0, 0.4), (0.0, 0.2), (1.0, 0.2), (2.0, 0.2)),
        )
    with pytest.raises(ValueError, match="sum to one"):
        bridge.fit(_paths(), ((-1.0, 0.5), (0.0, 0.5), (1.0, 0.5), (2.0, 0.5)))
    with pytest.raises(ValueError, match="unique"):
        MartingaleStressBridge().fit(
            (StressPath("x", 0.0, 0.0), StressPath("x", 0.0, 1.0)),
            ((0.0, 1.0),),
        )


def test_result_schema_constants_are_stable() -> None:
    assert BRIDGE_SCHEMA == "topology_gate.finite_martingale_stress_bridge"
    assert BRIDGE_VERSION == 1
    assert PublicMartingaleStressBridge is MartingaleStressBridge
