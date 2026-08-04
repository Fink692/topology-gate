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

from .asof import (
    LABEL_PRECEDENCE,
    MARKET_PRECEDENCE,
    UNIVERSE_PRECEDENCE,
    AmbiguousEventError,
    AsOfBook,
    AsOfError,
    AsOfSnapshot,
    DuplicateEventError,
    LabelObservation,
    MarketObservation,
    MissingLabelError,
    PointInTimePanel,
    UnavailableEventError,
    UniverseMembership,
    canonical_event_order,
)
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
from .economic import (
    ECONOMIC_SCHEMA,
    ECONOMIC_VERSION,
    EconomicDecision,
    EconomicEvaluationConfig,
    EconomicEvaluationError,
    EconomicEvaluationResult,
    EconomicPathRow,
    ExecutionCost,
    RealizedReturn,
    evaluate_economic_path,
)
from .evidence import (
    EvidenceLedger,
    EvidenceResolution,
    FrozenPrediction,
    LabelReceipt,
    PromotionEvidenceConfig,
)
from .manifest import (
    MANIFEST_SCHEMA,
    MANIFEST_VERSION,
    ManifestValidationError,
    RunManifest,
    RunSpec,
)
from .replay import (
    MAX_REPLAY_DECISIONS,
    MAX_REPLAY_RECORDS,
    REPLAY_SCHEMA,
    REPLAY_VERSION,
    CausalReplay,
    CausalReplayResult,
    ReplayConfig,
    ReplayError,
    ReplayPrediction,
    ReplayRecord,
    ReplayResolution,
    ReplayState,
    ReplayStateError,
    ReplayStatus,
    run_causal_replay,
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
    "CausalFeaturePlan": (".causal_numeric", "CausalFeaturePlan"),
    "CausalNumericError": (".causal_numeric", "CausalNumericError"),
    "CausalRLSConfig": (".causal_numeric", "CausalRLSConfig"),
    "CausalRLSModel": (".causal_numeric", "CausalRLSModel"),
    "CausalRLSReplayResult": (".causal_numeric", "CausalRLSReplayResult"),
    "CausalStep": (".causal_numeric", "CausalStep"),
    "FeatureBinding": (".causal_numeric", "FeatureBinding"),
    "run_causal_rls_replay": (".causal_numeric", "run_causal_rls_replay"),
    "CausalPromotionConfig": (".causal_promotion", "CausalPromotionConfig"),
    "CausalPromotionError": (".causal_promotion", "CausalPromotionError"),
    "CausalPromotionModel": (".causal_promotion", "CausalPromotionModel"),
    "CausalPromotionReplayResult": (
        ".causal_promotion",
        "CausalPromotionReplayResult",
    ),
    "CausalPromotionStep": (".causal_promotion", "CausalPromotionStep"),
    "run_causal_promotion_replay": (
        ".causal_promotion",
        "run_causal_promotion_replay",
    ),
    "CalibrationConfig": (".calibration", "CalibrationConfig"),
    "CalibrationCertificate": (".calibration", "CalibrationCertificate"),
    "EProcessCalibrationConfig": (".calibration", "EProcessCalibrationConfig"),
    "EProcessNullCalibrationResult": (
        ".calibration",
        "EProcessNullCalibrationResult",
    ),
    "StationaryBlockBootstrap": (".calibration", "StationaryBlockBootstrap"),
    "NullCalibrationResult": (".calibration", "NullCalibrationResult"),
    "ShiftCalibrationResult": (".calibration", "ShiftCalibrationResult"),
    "calibrate_null": (".calibration", "calibrate_null"),
    "calibrate_eprocess_null": (".calibration", "calibrate_eprocess_null"),
    "calibrate_shift": (".calibration", "calibrate_shift"),
    "PersistentLaplacianConfig": (".persistent", "PersistentLaplacianConfig"),
    "PersistentLaplacianBackend": (".persistent", "PersistentLaplacianBackend"),
    "PersistentLaplacianResult": (".persistent", "PersistentLaplacianResult"),
    "PERSISTENT_BACKEND_SCHEMA": (".persistent", "PERSISTENT_BACKEND_SCHEMA"),
    "PersistentSpectrum": (".persistent", "PersistentSpectrum"),
    "PersistentStatus": (".persistent", "PersistentStatus"),
    "compute_persistent_laplacian": (".persistent", "compute_persistent_laplacian"),
    "persistent_laplacian_backend": (".persistent", "persistent_laplacian_backend"),
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
_NUMERIC_WORKER_MODULES = frozenset(
    {
        ".backtest",
        ".calibration",
        ".causal_numeric",
        ".causal_promotion",
        ".online",
        ".persistent",
        ".synthetic",
    }
)


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
    "AmbiguousEventError",
    "ArrayLike",
    "AsOfBook",
    "AsOfError",
    "AsOfSnapshot",
    "BacktestConfig",
    "BacktestDatasetProtocol",
    "BacktestEngine",
    "BacktestResult",
    "BacktestResultProtocol",
    "BacktesterProtocol",
    "CalibrationConfig",
    "CalibrationCertificate",
    "StationaryBlockBootstrap",
    "EProcessCalibrationConfig",
    "EProcessNullCalibrationResult",
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
    "CausalReplay",
    "CausalReplayResult",
    "CausalStep",
    "CheckpointCompatibilityError",
    "CheckpointEnvelope",
    "CheckpointError",
    "CheckpointIntegrityError",
    "DetectedTopologyResult",
    "DuplicateEventError",
    "EvidenceLedger",
    "EvidenceResolution",
    "EProcess",
    "EProcessGateProtocol",
    "EProcessState",
    "ECONOMIC_SCHEMA",
    "ECONOMIC_VERSION",
    "EconomicDecision",
    "EconomicEvaluationConfig",
    "EconomicEvaluationError",
    "EconomicEvaluationResult",
    "EconomicPathRow",
    "ExecutionCost",
    "GateDecision",
    "GateStatus",
    "FrozenPrediction",
    "FeatureBinding",
    "LabelReceipt",
    "LABEL_PRECEDENCE",
    "LabelObservation",
    "MARKET_PRECEDENCE",
    "MarketObservation",
    "PointInTimePanel",
    "MANIFEST_SCHEMA",
    "MANIFEST_VERSION",
    "ManifestValidationError",
    "Matrix",
    "Metadata",
    "MissingLabelError",
    "ModelConfig",
    "Observation",
    "NullCalibrationResult",
    "OnlineRunConfig",
    "OnlineRunResult",
    "OnlineStreamState",
    "PendingLabelRecord",
    "PersistentLaplacianConfig",
    "PersistentLaplacianBackend",
    "PersistentLaplacianResult",
    "PersistentSpectrum",
    "PersistentStatus",
    "PERSISTENT_BACKEND_SCHEMA",
    "PromotionDecision",
    "PromotionEvidenceConfig",
    "PromotionDecisionProtocol",
    "PromotionGate",
    "PromotionGateStatus",
    "PromotionStatus",
    "ReplayConfig",
    "ReplayError",
    "ReplayPrediction",
    "ReplayRecord",
    "ReplayResolution",
    "ReplayState",
    "ReplayStateError",
    "ReplayStatus",
    "REPLAY_SCHEMA",
    "REPLAY_VERSION",
    "RLS",
    "RLSConfig",
    "RLSLearnerProtocol",
    "RLSState",
    "RLSUpdate",
    "RealizedReturn",
    "RunManifest",
    "RunSpec",
    "RollingTopologyDetector",
    "Scalar",
    "ShiftCalibrationResult",
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
    "UNIVERSE_PRECEDENCE",
    "UnavailableEventError",
    "UniverseMembership",
    "Vector",
    "WalkForwardBacktest",
    "WalkForwardBacktester",
    "WalkForwardConfig",
    "WalkForwardResult",
    "WorkerBacktestConfig",
    "MAX_REPLAY_DECISIONS",
    "MAX_REPLAY_RECORDS",
    "__version__",
    "generate_regime_switching",
    "generate_synthetic_regimes",
    "checkpoint_from_components",
    "canonical_event_order",
    "calibrate_null",
    "calibrate_eprocess_null",
    "calibrate_shift",
    "compute_persistent_laplacian",
    "load_checkpoint",
    "restore_component_states",
    "run_recursive_rls",
    "run_causal_replay",
    "run_causal_rls_replay",
    "run_causal_promotion_replay",
    "evaluate_economic_path",
    "save_checkpoint",
    "persistent_laplacian_backend",
]
