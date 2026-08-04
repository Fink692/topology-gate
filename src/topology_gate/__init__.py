"""Public import boundary for topology-gated recursive-model research.

The names defined directly here are dependency-light contracts from
``topology_gate.types``. Numerical worker implementations remain lazy exports:
requesting one imports only that worker, and a missing optional numerical stack
produces an actionable installation error.

``GateStatus``, ``BacktestConfig``, ``BacktestResult``, and ``TopologyResult``
are the canonical root contract names. Worker-specific objects that used to
collide with those names are available with explicit names such as
``PromotionGateStatus``, ``WalkForwardResult``, and ``DetectedTopologyResult``.
"""

from importlib import import_module
from typing import Any

from .checkpoint import (
    CheckpointCompatibilityError,
    CheckpointEnvelope,
    CheckpointError,
    CheckpointIntegrityError,
    checkpoint_from_components,
    load_checkpoint,
    restore_component_states,
    save_checkpoint,
)
from .types import (
    ArrayLike,
    BacktestConfig,
    BacktestDatasetProtocol,
    BacktesterProtocol,
    BacktestResult,
    BacktestResultProtocol,
    EProcessGateProtocol,
    EProcessState,
    GateDecision,
    GateStatus,
    Matrix,
    Metadata,
    Observation,
    PromotionDecisionProtocol,
    RLSLearnerProtocol,
    RLSState,
    RLSUpdate,
    Scalar,
    SyntheticConfig,
    SyntheticDataConfig,
    SyntheticDataFactoryProtocol,
    SyntheticDataGeneratorProtocol,
    TargetLike,
    Timestamp,
    TopologyDetectorProtocol,
    TopologyEstimate,
    TopologyResult,
    TopologyResultProtocol,
    TopologySignal,
    Vector,
)

# This is the single package-version source. setuptools reads the same
# attribute through ``tool.setuptools.dynamic`` in pyproject.toml.
__version__ = "0.1.0"

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "AdaptiveRLS": (".rls", "AdaptiveRLS"),
    "RLSConfig": (".rls", "RLSConfig"),
    "RLS": (".rls", "RLS"),
    "BacktestEngine": (".backtest", "BacktestEngine"),
    "WalkForwardResult": (".backtest", "BacktestResult"),
    "WorkerBacktestConfig": (".backtest", "BacktestConfig"),
    "EProcess": (".promotion", "EProcess"),
    "PromotionGateStatus": (".promotion", "GateStatus"),
    "PromotionDecision": (".promotion", "PromotionDecision"),
    "PromotionGate": (".promotion", "PromotionGate"),
    "PromotionStatus": (".promotion", "PromotionStatus"),
    "RollingTopologyDetector": (".topology", "RollingTopologyDetector"),
    "StreamingTopologyResult": (".topology", "StreamingTopologyResult"),
    "DetectedTopologyResult": (".topology", "TopologyResult"),
    "SyntheticDataset": (".synthetic", "SyntheticDataset"),
    "SyntheticRegimeProcess": (".synthetic", "SyntheticRegimeProcess"),
    "TimeIndexedFeatures": (".synthetic", "TimeIndexedFeatures"),
    "TimeIndexedLabels": (".synthetic", "TimeIndexedLabels"),
    "TopologyConfig": (".topology", "TopologyConfig"),
    "WalkForwardBacktest": (".backtest", "WalkForwardBacktest"),
    "WalkForwardBacktester": (".backtest", "WalkForwardBacktester"),
    "WalkForwardConfig": (".backtest", "WalkForwardConfig"),
    "ModelConfig": (".config", "ModelConfig"),
    "OnlineRunConfig": (".online", "OnlineRunConfig"),
    "OnlineRunResult": (".online", "OnlineRunResult"),
    "OnlineStreamState": (".online", "OnlineStreamState"),
    "PendingLabelRecord": (".online", "PendingLabelRecord"),
    "run_recursive_rls": (".online", "run_recursive_rls"),
    "generate_regime_switching": (".synthetic", "generate_regime_switching"),
    "generate_synthetic_regimes": (".synthetic", "generate_synthetic_regimes"),
}

# These workers import NumPy eagerly. Keeping this set explicit lets the root
# boundary preserve a useful standard-library-only import while giving a
# precise hint when a caller asks for a numerical worker without its extra.
_NUMERIC_WORKER_MODULES = frozenset({".backtest", ".online", ".synthetic"})


def _optional_dependency_error(
    name: str,
    error: ModuleNotFoundError,
) -> ImportError:
    missing = error.name or "an optional dependency"
    return ImportError(
        f"topology_gate.{name} requires the optional numerical worker "
        f"dependencies (missing {missing!r}). Install them with "
        "`python -m pip install 'topology-gate[numeric]'` or use the "
        "`test`/`dev` extra for the release toolchain."
    )


def __getattr__(name: str) -> Any:
    """Resolve worker implementations without importing optional dependencies."""

    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    try:
        value = getattr(import_module(module_name, __name__), attribute_name)
    except ModuleNotFoundError as exc:
        if module_name in _NUMERIC_WORKER_MODULES:
            raise _optional_dependency_error(name, exc) from exc
        raise
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


__all__ = [
    "AdaptiveRLS",
    "ArrayLike",
    "BacktestConfig",
    "BacktestDatasetProtocol",
    "BacktestEngine",
    "BacktestResult",
    "BacktestResultProtocol",
    "BacktesterProtocol",
    "CheckpointCompatibilityError",
    "CheckpointEnvelope",
    "CheckpointError",
    "CheckpointIntegrityError",
    "DetectedTopologyResult",
    "EProcess",
    "EProcessGateProtocol",
    "EProcessState",
    "GateDecision",
    "GateStatus",
    "Matrix",
    "Metadata",
    "ModelConfig",
    "Observation",
    "OnlineRunConfig",
    "OnlineRunResult",
    "OnlineStreamState",
    "PendingLabelRecord",
    "PromotionDecision",
    "PromotionDecisionProtocol",
    "PromotionGate",
    "PromotionGateStatus",
    "PromotionStatus",
    "RLS",
    "RLSConfig",
    "RLSLearnerProtocol",
    "RLSState",
    "RLSUpdate",
    "RollingTopologyDetector",
    "Scalar",
    "StreamingTopologyResult",
    "SyntheticConfig",
    "SyntheticDataConfig",
    "SyntheticDataFactoryProtocol",
    "SyntheticDataGeneratorProtocol",
    "SyntheticDataset",
    "SyntheticRegimeProcess",
    "TargetLike",
    "TimeIndexedFeatures",
    "TimeIndexedLabels",
    "Timestamp",
    "TopologyConfig",
    "TopologyDetectorProtocol",
    "TopologyEstimate",
    "TopologyResult",
    "TopologyResultProtocol",
    "TopologySignal",
    "Vector",
    "WalkForwardBacktest",
    "WalkForwardBacktester",
    "WalkForwardConfig",
    "WalkForwardResult",
    "WorkerBacktestConfig",
    "__version__",
    "generate_regime_switching",
    "generate_synthetic_regimes",
    "checkpoint_from_components",
    "load_checkpoint",
    "restore_component_states",
    "run_recursive_rls",
    "save_checkpoint",
]
