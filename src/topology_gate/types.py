"""Dependency-light types shared by the topology_gate components.

The numerical worker modules may use NumPy or other array libraries internally,
but these boundary types intentionally use standard-library typing.  This keeps
``import topology_gate`` useful in documentation, static analysis, and small
tests without forcing a numerical stack on every consumer.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Protocol, TypeAlias, runtime_checkable

Scalar: TypeAlias = int | float
Vector: TypeAlias = Sequence[Scalar]
Matrix: TypeAlias = Sequence[Sequence[Scalar]]
ArrayLike: TypeAlias = Vector | Matrix
TargetLike: TypeAlias = Scalar | Vector
Timestamp: TypeAlias = int | float | str | datetime
Metadata: TypeAlias = Mapping[str, Any]


def _finite_number(value: Any, name: str) -> float:
    """Validate a scalar boundary without importing a numerical stack."""

    if isinstance(value, bool):
        raise TypeError(f"{name} must be a finite real number")
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be a finite real number") from exc
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


class GateStatus(str, Enum):
    """Reason category reported by an e-process gate."""

    OPEN = "open"
    CLOSED = "closed"
    TOPOLOGY_REJECTED = "topology_rejected"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    INVALID_EVIDENCE = "invalid_evidence"


@dataclass(frozen=True, slots=True)
class Observation:
    """One supervised observation supplied to a recursive model."""

    features: ArrayLike
    target: TargetLike
    timestamp: Timestamp | None = None
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TopologySignal:
    """A detector's topology-related signal for one observation.

    ``score`` and ``dimension`` are deliberately not assigned a universal
    interpretation here: the detector owns their meaning and scale.  If the
    detector applies its own threshold, it can put that result in ``passed``.
    ``confidence`` is conventionally in ``[0, 1]`` when supplied.
    """

    score: float
    dimension: float | None = None
    confidence: float | None = None
    regime: str | None = None
    passed: bool | None = None
    index: int | None = None
    metadata: Metadata = field(default_factory=dict)


# These names make the intent clear to callers that prefer detector terminology.
TopologyEstimate: TypeAlias = TopologySignal
TopologyResult: TypeAlias = TopologySignal


@dataclass(frozen=True, slots=True)
class RLSState:
    """State snapshot exposed by a recursive least-squares learner."""

    coefficients: ArrayLike
    covariance: Matrix
    sample_count: int = 0
    forgetting_factor: float = 1.0

    def __post_init__(self) -> None:
        if self.sample_count < 0:
            raise ValueError("sample_count must be non-negative")
        factor = _finite_number(self.forgetting_factor, "forgetting_factor")
        if not 0.0 < factor <= 1.0:
            raise ValueError("forgetting_factor must be in (0, 1]")

    @property
    def observations(self) -> int:
        """Alias used by callers that name the count ``observations``."""

        return self.sample_count


@dataclass(frozen=True, slots=True)
class RLSUpdate:
    """Result of incorporating one target into an RLS learner."""

    prediction: TargetLike
    target: TargetLike
    residual: TargetLike
    state: RLSState


@dataclass(frozen=True, slots=True)
class EProcessState:
    """Current state of an e-process or its nonnegative capital process."""

    e_value: float
    observation_count: int = 0
    log_e_value: float | None = None
    wealth: float | None = None

    def __post_init__(self) -> None:
        e_value = _finite_number(self.e_value, "e_value")
        if e_value < 0.0:
            raise ValueError("e_value must be non-negative")
        if self.observation_count < 0:
            raise ValueError("observation_count must be non-negative")
        if self.log_e_value is not None:
            _finite_number(self.log_e_value, "log_e_value")
        if self.wealth is not None:
            _finite_number(self.wealth, "wealth")

    @property
    def observations(self) -> int:
        """Alias for the number of evidence updates incorporated."""

        return self.observation_count


@dataclass(frozen=True, slots=True)
class GateDecision:
    """A point-in-time decision made by the topology/e-process gate."""

    allowed: bool
    status: GateStatus
    e_value: float = 1.0
    threshold: float = 1.0
    topology: TopologySignal | None = None
    reason: str | None = None
    index: int | None = None

    def __post_init__(self) -> None:
        e_value = _finite_number(self.e_value, "e_value")
        threshold = _finite_number(self.threshold, "threshold")
        if e_value < 0.0:
            raise ValueError("e_value must be non-negative")
        if threshold <= 0.0:
            raise ValueError("threshold must be positive")

    @property
    def is_open(self) -> bool:
        """Readable alias for ``allowed``."""

        return self.allowed


@dataclass(frozen=True, slots=True)
class SyntheticDataConfig:
    """Configuration for a seeded covariance/regime-shift data factory."""

    n_samples: int = 512
    n_features: int = 4
    shift_points: Sequence[int] = (160, 320)
    seed: int = 7
    noise_scale: float = 0.25

    def __post_init__(self) -> None:
        if self.n_samples < 32:
            raise ValueError("n_samples must be at least 32")
        if self.n_features < 1:
            raise ValueError("n_features must be positive")
        if not math.isfinite(float(self.noise_scale)) or self.noise_scale <= 0.0:
            raise ValueError("noise_scale must be positive")


# A shorter spelling is useful for implementation modules and remains explicit.
SyntheticConfig: TypeAlias = SyntheticDataConfig


@runtime_checkable
class BacktestDatasetProtocol(Protocol):
    """Minimal view required by an offline backtest orchestrator.

    Concrete synthetic datasets may expose richer delayed labels, realized
    returns, regimes, and ground truth.  Keeping this boundary structural avoids
    duplicating a worker-owned dataset class in the shared types module.
    """

    features: Any


@runtime_checkable
class TopologyResultProtocol(Protocol):
    """Minimum detector-result view consumed by an orchestration layer."""

    features: Any
    scores: Any
    alarms: Any
    valid: Any
    calibrated: Any


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    """Controls chronological training and offline evaluation conventions."""

    initial_train_size: int = 20
    label_delay: int = 0
    retrain_every: int = 1
    training_window: int | None = None
    min_train_size: int = 1
    transaction_cost: float = 0.0
    slippage: float = 0.0
    transaction_cost_bps: float = 0.0
    max_position: float = 1.0
    allow_short: bool = True
    prediction_mode: str = "auto"
    hold_last_position: bool = True
    periods_per_year: float = 252.0
    detection_persistence: int = 1
    detection_position_threshold: float = 1.0e-12
    promotion_window: int | None = None

    def __post_init__(self) -> None:
        for name, value in {
            "initial_train_size": self.initial_train_size,
            "label_delay": self.label_delay,
            "retrain_every": self.retrain_every,
            "min_train_size": self.min_train_size,
            "detection_persistence": self.detection_persistence,
        }.items():
            if not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            minimum = 1 if name in {"retrain_every", "detection_persistence"} else 0
            if value < minimum:
                raise ValueError(f"{name} must be at least {minimum}")
        if self.training_window is not None and self.training_window < 1:
            raise ValueError("training_window must be positive when supplied")
        numeric_values: dict[str, float] = {
            "transaction_cost": self.transaction_cost,
            "slippage": self.slippage,
            "transaction_cost_bps": self.transaction_cost_bps,
            "max_position": self.max_position,
            "periods_per_year": self.periods_per_year,
            "detection_position_threshold": self.detection_position_threshold,
        }
        for numeric_name, numeric_value in numeric_values.items():
            if not math.isfinite(float(numeric_value)):
                raise ValueError(f"{numeric_name} must be finite")
            if numeric_value < 0.0:
                raise ValueError(f"{numeric_name} must be non-negative")
        if self.max_position == 0.0:
            raise ValueError("max_position must be positive")
        if self.periods_per_year == 0.0:
            raise ValueError("periods_per_year must be positive")

    @property
    def turnover_cost_rate(self) -> float:
        """Combined offline transaction-cost rate."""

        return self.transaction_cost + self.slippage + self.transaction_cost_bps / 10_000.0


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Small dependency-free result container for adapter-level evaluations."""

    predictions: Vector
    targets: Vector
    decisions: Sequence[GateDecision] = ()
    metrics: Mapping[str, float] = field(default_factory=dict)
    indices: Sequence[int] = ()

    def __len__(self) -> int:
        return len(self.targets)


@runtime_checkable
class BacktestResultProtocol(Protocol):
    """Minimum result view shared with richer worker-owned reports."""

    predictions: Any
    positions: Any
    net_returns: Any
    metrics: Any


@runtime_checkable
class TopologyDetectorProtocol(Protocol):
    """Minimal detector boundary consumed by orchestration code."""

    def detect(
        self,
        features: ArrayLike,
    ) -> TopologyResultProtocol:
        """Return a causal rolling detector result."""


@runtime_checkable
class RLSLearnerProtocol(Protocol):
    """Minimal stateful learner boundary for recursive quant prediction."""

    def predict(self, features: ArrayLike) -> TargetLike:
        """Predict the next target without mutating learner state."""

    def update(self, features: ArrayLike, target: TargetLike) -> RLSUpdate:
        """Incorporate one labelled observation and return an update record."""


@runtime_checkable
class EProcessGateProtocol(Protocol):
    """Minimal gate boundary for sequential evidence updates."""

    def update(
        self,
        evidence: float,
        *,
        eta: Any = None,
        metadata: Metadata | None = None,
    ) -> "PromotionDecisionProtocol":
        """Consume one bounded score and return a worker-owned decision."""


@runtime_checkable
class PromotionDecisionProtocol(Protocol):
    """Minimum promotion decision view exposed by an e-process worker."""

    e_value: float
    threshold: float
    threshold_crossed: bool
    promoted: bool


@runtime_checkable
class SyntheticDataFactoryProtocol(Protocol):
    """Callable boundary for a seeded synthetic-data factory."""

    def __call__(
        self,
        *,
        n_samples: int = 512,
        n_features: int = 4,
        shift_points: Sequence[int] = (160, 320),
        seed: int = 7,
        noise_scale: float = 0.25,
    ) -> BacktestDatasetProtocol:
        """Return an in-memory dataset with known synthetic shifts."""


@runtime_checkable
class SyntheticDataGeneratorProtocol(Protocol):
    """Boundary for deterministic synthetic data generation."""

    def generate(
        self,
        config: SyntheticDataConfig | None = None,
    ) -> BacktestDatasetProtocol:
        """Generate a dataset from worker-owned configuration."""


@runtime_checkable
class BacktesterProtocol(Protocol):
    """Boundary for offline sequential model evaluation."""

    def run(
        self,
        features: BacktestDatasetProtocol | ArrayLike,
        labels: ArrayLike | None = None,
        realized_returns: ArrayLike | None = None,
        **kwargs: Any,
    ) -> BacktestResultProtocol:
        """Evaluate chronologically without placing orders or trades."""


__all__ = [
    "ArrayLike",
    "BacktestConfig",
    "BacktestDatasetProtocol",
    "BacktestResult",
    "BacktestResultProtocol",
    "BacktesterProtocol",
    "EProcessGateProtocol",
    "EProcessState",
    "GateDecision",
    "GateStatus",
    "Matrix",
    "Metadata",
    "Observation",
    "PromotionDecisionProtocol",
    "RLSLearnerProtocol",
    "RLSState",
    "RLSUpdate",
    "Scalar",
    "SyntheticConfig",
    "SyntheticDataConfig",
    "SyntheticDataFactoryProtocol",
    "SyntheticDataGeneratorProtocol",
    "Timestamp",
    "TargetLike",
    "TopologyDetectorProtocol",
    "TopologyEstimate",
    "TopologyResultProtocol",
    "TopologyResult",
    "TopologySignal",
    "Vector",
]
