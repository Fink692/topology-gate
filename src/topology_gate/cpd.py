"""A causal mean/covariance change detector for non-topological baselines."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from numbers import Integral
from typing import Any, cast

try:
    import numpy as np
except ModuleNotFoundError as exc:  # pragma: no cover - import boundary
    raise ImportError(
        "topology_gate.cpd requires NumPy; install `topology-gate[numeric]`"
    ) from exc


MEAN_COV_CUSUM_SCHEMA = "topology_gate.mean_covariance_cusum"
MEAN_COV_CUSUM_VERSION = 1
MAX_MEAN_COV_HISTORY = 4_096


def _integer(name: str, value: Any, minimum: int, maximum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum or result > maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return result


def _finite(name: str, value: Any, *, minimum: float = -math.inf) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{name} must be finite and at least {minimum}")
    return result


def _vector(value: Any, name: str, width: int) -> np.ndarray[Any, Any]:
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite vector") from exc
    if result.ndim != 1 or result.size != width or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite vector of length {width}")
    return np.array(result, copy=True)


def _digest(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class MeanCovarianceCUSUMConfig:
    """Configuration for a diagonal mean/covariance two-block detector."""

    n_features: int
    block_window: int = 16
    threshold: float = 5.0
    forgetting_lambda_min: float = 0.8
    forgetting_lambda_max: float = 0.99
    forgetting_sensitivity: float = 1.0
    scale_floor: float = 1.0e-6
    max_history: int = MAX_MEAN_COV_HISTORY

    def __post_init__(self) -> None:
        features = _integer("n_features", self.n_features, 1, 256)
        window = _integer("block_window", self.block_window, 2, 512)
        threshold = _finite("threshold", self.threshold, minimum=0.0)
        lower = _finite("forgetting_lambda_min", self.forgetting_lambda_min)
        upper = _finite("forgetting_lambda_max", self.forgetting_lambda_max)
        if not 0.0 < lower <= upper <= 1.0:
            raise ValueError("forgetting lambda bounds are invalid")
        sensitivity = _finite(
            "forgetting_sensitivity", self.forgetting_sensitivity, minimum=0.0
        )
        floor = _finite("scale_floor", self.scale_floor, minimum=0.0)
        if floor <= 0.0:
            raise ValueError("scale_floor must be positive")
        history = _integer("max_history", self.max_history, 2 * window, MAX_MEAN_COV_HISTORY)
        object.__setattr__(self, "n_features", features)
        object.__setattr__(self, "block_window", window)
        object.__setattr__(self, "threshold", threshold)
        object.__setattr__(self, "forgetting_lambda_min", lower)
        object.__setattr__(self, "forgetting_lambda_max", upper)
        object.__setattr__(self, "forgetting_sensitivity", sensitivity)
        object.__setattr__(self, "scale_floor", floor)
        object.__setattr__(self, "max_history", history)

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": MEAN_COV_CUSUM_VERSION,
            "schema": MEAN_COV_CUSUM_SCHEMA,
            "n_features": self.n_features,
            "block_window": self.block_window,
            "threshold": self.threshold,
            "forgetting_lambda_min": self.forgetting_lambda_min,
            "forgetting_lambda_max": self.forgetting_lambda_max,
            "forgetting_sensitivity": self.forgetting_sensitivity,
            "scale_floor": self.scale_floor,
            "max_history": self.max_history,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "MeanCovarianceCUSUMConfig":
        if not isinstance(state, dict):
            raise ValueError("mean/covariance config must be a mapping")
        if state.get("version") != MEAN_COV_CUSUM_VERSION or state.get("schema") != MEAN_COV_CUSUM_SCHEMA:
            raise ValueError("unsupported mean/covariance config")
        return cls(
            n_features=cast(int, state["n_features"]),
            block_window=cast(int, state["block_window"]),
            threshold=cast(float, state["threshold"]),
            forgetting_lambda_min=cast(float, state["forgetting_lambda_min"]),
            forgetting_lambda_max=cast(float, state["forgetting_lambda_max"]),
            forgetting_sensitivity=cast(float, state["forgetting_sensitivity"]),
            scale_floor=cast(float, state["scale_floor"]),
            max_history=cast(int, state["max_history"]),
        )

    @property
    def identity(self) -> str:
        return _digest(self.state_dict())


@dataclass(frozen=True, slots=True)
class MeanCovarianceCUSUMObservation:
    score: float
    alarm: bool
    ready: bool
    forgetting_factor: float
    method: str = "mean-covariance-cusum"


@dataclass(frozen=True, slots=True)
class MeanCovarianceCUSUMBatchResult:
    alarms: np.ndarray[Any, Any]
    scores: np.ndarray[Any, Any]


class MeanCovarianceCUSUM:
    """Compare consecutive feature blocks using diagonal standardization."""

    method = "mean-covariance-cusum"

    def __init__(self, config: MeanCovarianceCUSUMConfig) -> None:
        if not isinstance(config, MeanCovarianceCUSUMConfig):
            raise TypeError("config must be a MeanCovarianceCUSUMConfig")
        self.config = config
        self.reset_stream()

    @property
    def config_identity(self) -> str:
        return self.config.identity

    def reset_stream(self) -> None:
        self._history: list[tuple[float, ...]] = []

    def _observe_array(self, value: Any) -> MeanCovarianceCUSUMObservation:
        row = _vector(value, "observation", self.config.n_features)
        self._history.append(tuple(float(item) for item in row))
        if len(self._history) > self.config.max_history:
            self._history = self._history[-self.config.max_history :]
        window = self.config.block_window
        if len(self._history) < 2 * window:
            return MeanCovarianceCUSUMObservation(
                score=0.0,
                alarm=False,
                ready=False,
                forgetting_factor=self.config.forgetting_lambda_max,
            )
        values = np.asarray(self._history, dtype=float)
        prior = values[-2 * window : -window]
        current = values[-window:]
        center = np.mean(prior, axis=0)
        scale = np.maximum(np.std(prior, axis=0, ddof=1), self.config.scale_floor)
        standardized = (np.mean(current, axis=0) - center) / scale
        score = float(np.linalg.norm(standardized))
        excess = max(0.0, score - self.config.threshold)
        factor = self.config.forgetting_lambda_min + (
            self.config.forgetting_lambda_max - self.config.forgetting_lambda_min
        ) * math.exp(-self.config.forgetting_sensitivity * excess)
        return MeanCovarianceCUSUMObservation(
            score=score,
            alarm=score >= self.config.threshold,
            ready=True,
            forgetting_factor=factor,
        )

    def observe(self, value: Any) -> MeanCovarianceCUSUMObservation:
        return self._observe_array(value)

    def detect(self, observations: Any) -> MeanCovarianceCUSUMBatchResult:
        values = np.asarray(observations, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.config.n_features:
            raise ValueError("observations must have the configured feature width")
        self.reset_stream()
        rows = [self._observe_array(row) for row in values]
        return MeanCovarianceCUSUMBatchResult(
            alarms=np.asarray([row.alarm for row in rows], dtype=bool),
            scores=np.asarray([row.score for row in rows], dtype=float),
        )

    def stream_state_dict(self) -> dict[str, Any]:
        return {
            "version": MEAN_COV_CUSUM_VERSION,
            "schema": MEAN_COV_CUSUM_SCHEMA,
            "config_identity": self.config_identity,
            "history": [list(row) for row in self._history],
        }

    def load_stream_state_dict(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            raise ValueError("mean/covariance stream state must be a mapping")
        if state.get("version") != MEAN_COV_CUSUM_VERSION or state.get("schema") != MEAN_COV_CUSUM_SCHEMA:
            raise ValueError("unsupported mean/covariance stream state")
        if state.get("config_identity") != self.config_identity:
            raise ValueError("mean/covariance detector identity mismatch")
        raw_history = state.get("history")
        if not isinstance(raw_history, list) or len(raw_history) > self.config.max_history:
            raise ValueError("mean/covariance history is invalid")
        candidate = []
        for row in raw_history:
            values = _vector(row, "history row", self.config.n_features)
            candidate.append(tuple(float(item) for item in values))
        self._history = candidate


__all__ = [
    "MAX_MEAN_COV_HISTORY",
    "MEAN_COV_CUSUM_SCHEMA",
    "MEAN_COV_CUSUM_VERSION",
    "MeanCovarianceCUSUM",
    "MeanCovarianceCUSUMBatchResult",
    "MeanCovarianceCUSUMConfig",
    "MeanCovarianceCUSUMObservation",
]
