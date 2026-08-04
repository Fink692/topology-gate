"""Endogenous Wasserstein-robust online linear learning.

For the absolute residual loss

    l(theta; x, y) = |y - x^T theta|,

the loss is Lipschitz in ``(x, y)`` under the Euclidean ground metric with
constant ``sqrt(1 + ||theta||^2)``.  The one-Wasserstein distributionally
robust objective is therefore bounded by the empirical loss plus
``rho * sqrt(1 + ||theta||^2)``.  This module performs a deterministic
projected subgradient step on that bounded surrogate.

The radius is endogenous only through a prediction-time instability score:
``rho_t = clip(rho_0 + c * G_t, rho_0, rho_max)``.  The score must be supplied
before the target is available.  This is a finite online robust-learning
primitive, not a proof of market distributional robustness or an adapted-
Wasserstein path solver.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Mapping, Sequence

WASSERSTEIN_SCHEMA = "topology_gate.endogenous_wasserstein_linear"
WASSERSTEIN_VERSION = 1
MAX_WASSERSTEIN_FEATURES = 512


def _finite(name: str, value: Any, *, minimum: float | None = None) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _integer(name: str, value: Any, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum or result > maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return result


def _digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _norm(values: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


@dataclass(frozen=True, slots=True)
class WassersteinRobustConfig:
    """Immutable configuration for the bounded robust online update."""

    n_features: int
    learning_rate: float = 0.01
    ridge: float = 1.0e-4
    radius_floor: float = 0.01
    radius_sensitivity: float = 0.05
    radius_max: float = 1.0
    gradient_clip: float = 10.0
    feature_abs_bound: float = 100.0

    def __post_init__(self) -> None:
        _integer(
            "n_features",
            self.n_features,
            minimum=1,
            maximum=MAX_WASSERSTEIN_FEATURES,
        )
        _finite("learning_rate", self.learning_rate, minimum=1.0e-15)
        _finite("ridge", self.ridge, minimum=0.0)
        floor = _finite("radius_floor", self.radius_floor, minimum=0.0)
        maximum = _finite("radius_max", self.radius_max, minimum=0.0)
        if floor > maximum:
            raise ValueError("radius_floor must not exceed radius_max")
        _finite("radius_sensitivity", self.radius_sensitivity, minimum=0.0)
        _finite("gradient_clip", self.gradient_clip, minimum=1.0e-15)
        _finite("feature_abs_bound", self.feature_abs_bound, minimum=1.0e-15)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_features": self.n_features,
            "learning_rate": self.learning_rate,
            "ridge": self.ridge,
            "radius_floor": self.radius_floor,
            "radius_sensitivity": self.radius_sensitivity,
            "radius_max": self.radius_max,
            "gradient_clip": self.gradient_clip,
            "feature_abs_bound": self.feature_abs_bound,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "WassersteinRobustConfig":
        expected = {
            "n_features",
            "learning_rate",
            "ridge",
            "radius_floor",
            "radius_sensitivity",
            "radius_max",
            "gradient_clip",
            "feature_abs_bound",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("Wasserstein config has unknown or missing fields")
        return cls(**{key: value[key] for key in expected})


@dataclass(frozen=True, slots=True)
class WassersteinRobustUpdate:
    """Auditable result of one prediction/update transition."""

    step: int
    prediction: float
    target: float
    residual: float
    instability_score: float
    radius: float
    empirical_absolute_loss: float
    robust_penalty: float
    robust_objective: float
    gradient_norm: float
    coefficients: tuple[float, ...]


class EndogenousWassersteinLinearLearner:
    """Online linear learner for a bounded 1-Wasserstein robust surrogate."""

    def __init__(self, config: WassersteinRobustConfig) -> None:
        if not isinstance(config, WassersteinRobustConfig):
            raise TypeError("config must be WassersteinRobustConfig")
        self.config = config
        self._coefficients = [0.0] * config.n_features
        self._step = 0

    @property
    def step(self) -> int:
        return self._step

    @property
    def coefficients(self) -> tuple[float, ...]:
        return tuple(self._coefficients)

    def _features(self, features: Sequence[float]) -> list[float]:
        try:
            values = list(features)
        except TypeError as exc:
            raise TypeError("features must be a sequence") from exc
        if len(values) != self.config.n_features:
            raise ValueError(
                f"features must contain exactly {self.config.n_features} values"
            )
        result = []
        for index, value in enumerate(values):
            numeric = _finite(f"features[{index}]", value)
            if abs(numeric) > self.config.feature_abs_bound:
                raise ValueError(f"features[{index}] exceeds feature_abs_bound")
            result.append(numeric)
        return result

    def _score(self, instability_score: Any) -> float:
        return _finite("instability_score", instability_score, minimum=0.0)

    def radius(self, instability_score: Any) -> float:
        score = self._score(instability_score)
        raw = self.config.radius_floor + self.config.radius_sensitivity * score
        return min(self.config.radius_max, max(self.config.radius_floor, raw))

    def predict(self, features: Sequence[float]) -> float:
        values = self._features(features)
        prediction = sum(coefficient * value for coefficient, value in zip(self._coefficients, values))
        if not math.isfinite(prediction):
            raise FloatingPointError("prediction is not finite")
        return prediction

    def robust_objective(
        self,
        features: Sequence[float],
        target: Any,
        instability_score: Any,
    ) -> tuple[float, float, float, float]:
        values = self._features(features)
        target_value = _finite("target", target)
        score = self._score(instability_score)
        prediction = self.predict(values)
        residual = target_value - prediction
        empirical = abs(residual)
        coefficient_norm = _norm(self._coefficients)
        penalty = self.radius(score) * math.sqrt(1.0 + coefficient_norm * coefficient_norm)
        ridge_penalty = 0.5 * self.config.ridge * coefficient_norm * coefficient_norm
        objective = empirical + penalty + ridge_penalty
        if not math.isfinite(objective):
            raise FloatingPointError("robust objective is not finite")
        return prediction, empirical, penalty, objective

    def observe(
        self,
        features: Sequence[float],
        target: Any,
        instability_score: Any,
    ) -> WassersteinRobustUpdate:
        values = self._features(features)
        target_value = _finite("target", target)
        score = self._score(instability_score)
        prediction = self.predict(values)
        residual = target_value - prediction
        radius = self.radius(score)
        coefficient_norm = _norm(self._coefficients)
        denominator = math.sqrt(1.0 + coefficient_norm * coefficient_norm)
        sign = 1.0 if residual > 0.0 else -1.0 if residual < 0.0 else 0.0
        gradient = [
            -sign * value
            + radius * coefficient / denominator
            + self.config.ridge * coefficient
            for value, coefficient in zip(values, self._coefficients)
        ]
        raw_gradient_norm = _norm(gradient)
        if raw_gradient_norm > self.config.gradient_clip:
            scale = self.config.gradient_clip / raw_gradient_norm
            gradient = [value * scale for value in gradient]
        gradient_norm = _norm(gradient)
        candidate = [
            coefficient - self.config.learning_rate * value
            for coefficient, value in zip(self._coefficients, gradient)
        ]
        if any(not math.isfinite(value) for value in candidate):
            raise FloatingPointError("non-finite robust coefficient update")
        empirical = abs(residual)
        penalty = radius * denominator
        objective = empirical + penalty + 0.5 * self.config.ridge * coefficient_norm**2
        self._coefficients = candidate
        self._step += 1
        return WassersteinRobustUpdate(
            step=self._step,
            prediction=prediction,
            target=target_value,
            residual=residual,
            instability_score=score,
            radius=radius,
            empirical_absolute_loss=empirical,
            robust_penalty=penalty,
            robust_objective=objective,
            gradient_norm=gradient_norm,
            coefficients=tuple(candidate),
        )

    def state_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": WASSERSTEIN_SCHEMA,
            "version": WASSERSTEIN_VERSION,
            "config": self.config.to_dict(),
            "step": self._step,
            "coefficients": list(self._coefficients),
        }
        payload["digest"] = _digest(payload)
        return payload

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "EndogenousWassersteinLinearLearner":
        if not isinstance(state, Mapping):
            raise TypeError("state must be a mapping")
        expected = {"schema", "version", "config", "step", "coefficients", "digest"}
        if set(state) != expected:
            raise ValueError("Wasserstein state has unknown or missing fields")
        unsigned = {key: state[key] for key in expected if key != "digest"}
        if state["digest"] != _digest(unsigned):
            raise ValueError("Wasserstein state digest mismatch")
        if state["schema"] != WASSERSTEIN_SCHEMA or state["version"] != WASSERSTEIN_VERSION:
            raise ValueError("unsupported Wasserstein state")
        config = WassersteinRobustConfig.from_dict(state["config"])
        step = _integer("state.step", state["step"], minimum=0, maximum=10**12)
        coefficients = state["coefficients"]
        if not isinstance(coefficients, Sequence) or isinstance(coefficients, (str, bytes, bytearray)):
            raise ValueError("coefficients must be a sequence")
        if len(coefficients) != config.n_features:
            raise ValueError("coefficient dimension does not match config")
        result = cls(config)
        result._step = step
        result._coefficients = [
            _finite(f"coefficients[{index}]", value)
            for index, value in enumerate(coefficients)
        ]
        return result

    def digest(self) -> str:
        return self.state_dict()["digest"]

