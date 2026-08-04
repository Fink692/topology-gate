"""Focused release-contract tests for the package root."""

from __future__ import annotations

from importlib import metadata
from pathlib import Path

import pytest

import topology_gate
from topology_gate import types


def test_root_all_is_unique_and_resolves_every_public_name() -> None:
    assert len(topology_gate.__all__) == len(set(topology_gate.__all__))

    numeric_names = {
        "BacktestEngine",
        "CalibrationCertificate",
        "CausalFeaturePlan",
        "CausalNumericError",
        "CausalPromotionConfig",
        "CausalPromotionError",
        "CausalPromotionModel",
        "CausalPromotionReplayResult",
        "CausalPromotionStep",
        "CausalRLSConfig",
        "CausalRLSModel",
        "CausalRLSReplayResult",
        "CausalStep",
        "FeatureBinding",
        "OnlineRunConfig",
        "OnlineRunResult",
        "PersistentLaplacianConfig",
        "PersistentLaplacianResult",
        "PersistentSpectrum",
        "PersistentStatus",
        "SyntheticDataset",
        "SyntheticRegimeProcess",
        "TimeIndexedFeatures",
        "TimeIndexedLabels",
        "WalkForwardBacktest",
        "WalkForwardBacktester",
        "WalkForwardConfig",
        "WalkForwardResult",
        "WorkerBacktestConfig",
        "generate_regime_switching",
        "generate_synthetic_regimes",
        "run_recursive_rls",
        "run_causal_rls_replay",
        "run_causal_promotion_replay",
        "compute_persistent_laplacian",
        "persistent_laplacian_backend",
    }

    for name in topology_gate.__all__:
        try:
            value = getattr(topology_gate, name)
        except ImportError as exc:
            if name not in numeric_names:
                raise
            assert "topology-gate[numeric]" in str(exc)
            continue
        assert value is not None
        assert name in dir(topology_gate)


def test_canonical_root_contract_names_are_dependency_light() -> None:
    assert topology_gate.GateStatus is types.GateStatus
    assert topology_gate.BacktestConfig is types.BacktestConfig
    assert topology_gate.BacktestResult is types.BacktestResult
    assert topology_gate.TopologyResult is types.TopologyResult


def test_worker_collisions_have_explicit_names() -> None:
    from topology_gate.backtest import BacktestConfig as worker_config
    from topology_gate.backtest import BacktestResult as worker_result
    from topology_gate.promotion import GateStatus as promotion_status
    from topology_gate.topology import TopologyResult as detected_result

    assert topology_gate.WorkerBacktestConfig is worker_config
    assert topology_gate.WalkForwardResult is worker_result
    assert topology_gate.PromotionGateStatus is promotion_status
    assert topology_gate.DetectedTopologyResult is detected_result
    assert topology_gate.GateStatus is not promotion_status


def test_distribution_version_matches_the_single_source() -> None:
    try:
        installed_version = metadata.version("topology-gate")
    except metadata.PackageNotFoundError as exc:  # pragma: no cover - setup error
        pytest.fail(
            "release-contract tests require an installed package; run "
            "`python -m pip install -e .` first"
        )
        raise AssertionError from exc
    assert installed_version == topology_gate.__version__


def test_py_typed_marker_is_present() -> None:
    marker = Path(topology_gate.__path__[0]) / "py.typed"
    assert marker.is_file()
