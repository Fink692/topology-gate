"""Mechanism-localized continual learning primitives.

This module treats a decomposed predictor as a collection of causal-ish
mechanisms.  Each mechanism owns a feature slice and a scalar target.  A
robust, prefix-only residual monitor identifies which mechanisms have lost
invariance.  During a localized shift, only the flagged learners are updated;
unchanged learners are frozen.  When no mechanism is flagged, all learners
receive the ordinary stable update.

The component is a research control primitive, not a causal-identification
claim.  The mechanism partition and target assignment remain part of the
pre-registered study design.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import deque
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Mapping, Sequence

from .rls import RLS

MECHANISM_SCHEMA = "topology_gate.mechanism_localized_rls"
MECHANISM_VERSION = 1
MAX_MECHANISMS = 128
MAX_FEATURE_INDICES = 256
MAX_HISTORY = 4_096


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


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return 0.5 * (ordered[midpoint - 1] + ordered[midpoint])


@dataclass(frozen=True, slots=True)
class MechanismSpec:
    """Feature ownership and identity for one independently updated module."""

    mechanism_id: str
    feature_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.mechanism_id, str) or not self.mechanism_id.strip():
            raise ValueError("mechanism_id must be a non-empty string")
        try:
            indices = tuple(self.feature_indices)
        except TypeError as exc:
            raise ValueError("feature_indices must be a sequence") from exc
        if not indices or len(indices) > MAX_FEATURE_INDICES:
            raise ValueError(
                f"feature_indices must contain 1..{MAX_FEATURE_INDICES} items"
            )
        if any(
            isinstance(index, bool) or not isinstance(index, Integral) or int(index) < 0
            for index in indices
        ):
            raise ValueError("feature_indices must contain non-negative integers")
        if len(set(int(index) for index in indices)) != len(indices):
            raise ValueError("feature_indices must not contain duplicates")
        object.__setattr__(self, "feature_indices", tuple(int(index) for index in indices))

    def to_dict(self) -> dict[str, Any]:
        return {
            "mechanism_id": self.mechanism_id,
            "feature_indices": list(self.feature_indices),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MechanismSpec":
        if not isinstance(value, Mapping):
            raise ValueError("mechanism spec must be a mapping")
        if set(value) != {"mechanism_id", "feature_indices"}:
            raise ValueError("mechanism spec has unknown or missing fields")
        return cls(
            mechanism_id=value["mechanism_id"],
            feature_indices=tuple(value["feature_indices"]),
        )


@dataclass(frozen=True, slots=True)
class MechanismLocalizedConfig:
    """Predeclared update and residual-monitor policy."""

    mechanisms: tuple[MechanismSpec, ...]
    ridge: float = 1.0
    stable_forgetting_factor: float = 0.995
    shift_forgetting_factor: float = 0.80
    residual_history: int = 64
    minimum_history: int = 16
    residual_scale_floor: float = 1.0e-6
    drift_threshold: float = 4.0

    def __post_init__(self) -> None:
        try:
            specs = tuple(self.mechanisms)
        except TypeError as exc:
            raise ValueError("mechanisms must be a sequence") from exc
        if not specs or len(specs) > MAX_MECHANISMS:
            raise ValueError(f"mechanisms must contain 1..{MAX_MECHANISMS} items")
        if any(not isinstance(spec, MechanismSpec) for spec in specs):
            raise ValueError("mechanisms must contain MechanismSpec values")
        identifiers = [spec.mechanism_id for spec in specs]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("mechanism_id values must be unique")
        object.__setattr__(self, "mechanisms", specs)
        _finite("ridge", self.ridge, minimum=1.0e-15)
        stable = _finite("stable_forgetting_factor", self.stable_forgetting_factor)
        shift = _finite("shift_forgetting_factor", self.shift_forgetting_factor)
        if not 0.0 < shift <= stable <= 1.0:
            raise ValueError(
                "forgetting factors must satisfy 0 < shift <= stable <= 1"
            )
        _integer("residual_history", self.residual_history, minimum=2, maximum=MAX_HISTORY)
        _integer(
            "minimum_history",
            self.minimum_history,
            minimum=2,
            maximum=self.residual_history,
        )
        _finite("residual_scale_floor", self.residual_scale_floor, minimum=1.0e-15)
        _finite("drift_threshold", self.drift_threshold, minimum=0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mechanisms": [spec.to_dict() for spec in self.mechanisms],
            "ridge": self.ridge,
            "stable_forgetting_factor": self.stable_forgetting_factor,
            "shift_forgetting_factor": self.shift_forgetting_factor,
            "residual_history": self.residual_history,
            "minimum_history": self.minimum_history,
            "residual_scale_floor": self.residual_scale_floor,
            "drift_threshold": self.drift_threshold,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MechanismLocalizedConfig":
        expected = {
            "mechanisms",
            "ridge",
            "stable_forgetting_factor",
            "shift_forgetting_factor",
            "residual_history",
            "minimum_history",
            "residual_scale_floor",
            "drift_threshold",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("mechanism config has unknown or missing fields")
        return cls(
            mechanisms=tuple(MechanismSpec.from_dict(item) for item in value["mechanisms"]),
            ridge=value["ridge"],
            stable_forgetting_factor=value["stable_forgetting_factor"],
            shift_forgetting_factor=value["shift_forgetting_factor"],
            residual_history=value["residual_history"],
            minimum_history=value["minimum_history"],
            residual_scale_floor=value["residual_scale_floor"],
            drift_threshold=value["drift_threshold"],
        )


@dataclass(frozen=True, slots=True)
class MechanismObservation:
    """One mechanism's prediction-time monitoring and update decision."""

    mechanism_id: str
    prediction: float
    target: float
    residual: float
    baseline: float
    scale: float
    score: float
    shifted: bool
    updated: bool
    forgetting_factor: float | None


@dataclass(frozen=True, slots=True)
class MechanismUpdate:
    """Receipt for one complete modular transition."""

    step: int
    prediction: float
    target: float
    residual: float
    shifted_mechanisms: tuple[str, ...]
    updated_mechanisms: tuple[str, ...]
    observations: tuple[MechanismObservation, ...]


class MechanismLocalizedRLS:
    """RLS modules with prefix-only, localized residual-triggered forgetting."""

    def __init__(self, config: MechanismLocalizedConfig) -> None:
        if not isinstance(config, MechanismLocalizedConfig):
            raise TypeError("config must be MechanismLocalizedConfig")
        self.config = config
        self._learners = {
            spec.mechanism_id: RLS(
                len(spec.feature_indices),
                ridge=config.ridge,
                forgetting_factor=config.stable_forgetting_factor,
                lambda_min=config.shift_forgetting_factor,
                lambda_max=config.stable_forgetting_factor,
            )
            for spec in config.mechanisms
        }
        self._histories: dict[str, deque[float]] = {
            spec.mechanism_id: deque(maxlen=config.residual_history)
            for spec in config.mechanisms
        }
        self._specs = {spec.mechanism_id: spec for spec in config.mechanisms}
        self._step = 0

    @property
    def step(self) -> int:
        return self._step

    @property
    def mechanism_ids(self) -> tuple[str, ...]:
        return tuple(spec.mechanism_id for spec in self.config.mechanisms)

    @property
    def learners(self) -> Mapping[str, RLS]:
        return self._learners

    def _features(self, features: Sequence[float], spec: MechanismSpec) -> list[float]:
        try:
            values = list(features)
        except TypeError as exc:
            raise TypeError("features must be a sequence") from exc
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in values
        ):
            raise ValueError("features must contain finite numeric values")
        if not values:
            raise ValueError("features must not be empty")
        if max(spec.feature_indices) >= len(values):
            raise ValueError(
                f"features has length {len(values)} but {spec.mechanism_id} "
                f"requires index {max(spec.feature_indices)}"
            )
        return [float(values[index]) for index in spec.feature_indices]

    def _targets(self, targets: Mapping[str, float]) -> dict[str, float]:
        if not isinstance(targets, Mapping):
            raise TypeError("targets must be a mapping keyed by mechanism_id")
        expected = set(self.mechanism_ids)
        if set(targets) != expected:
            raise ValueError("targets must contain exactly the configured mechanisms")
        result = {}
        for mechanism_id, target in targets.items():
            value = _finite(f"targets[{mechanism_id!r}]", target)
            result[mechanism_id] = value
        return result

    def predict(self, features: Sequence[float]) -> dict[str, float]:
        predictions: dict[str, float] = {}
        for mechanism_id in self.mechanism_ids:
            value = self._learners[mechanism_id].predict(
                self._features(features, self._specs[mechanism_id])
            )
            if isinstance(value, list):
                raise RuntimeError("mechanism learner must be scalar-output")
            predictions[mechanism_id] = float(value)
        return predictions

    def _diagnose(self, mechanism_id: str, residual: float) -> tuple[float, float, float, bool]:
        history = tuple(self._histories[mechanism_id])
        if not history:
            baseline = 0.0
            scale = self.config.residual_scale_floor
        else:
            baseline = _median(history)
            deviations = [abs(value - baseline) for value in history]
            scale = max(
                self.config.residual_scale_floor,
                1.4826 * _median(deviations),
            )
        score = abs(residual - baseline) / scale
        shifted = (
            len(history) >= self.config.minimum_history
            and score >= self.config.drift_threshold
        )
        return baseline, scale, score, shifted

    def observe(
        self,
        features: Sequence[float],
        targets: Mapping[str, float],
    ) -> MechanismUpdate:
        target_values = self._targets(targets)
        predictions = self.predict(features)
        residuals = {
            mechanism_id: target_values[mechanism_id] - predictions[mechanism_id]
            for mechanism_id in self.mechanism_ids
        }
        diagnostics = {
            mechanism_id: self._diagnose(mechanism_id, residuals[mechanism_id])
            for mechanism_id in self.mechanism_ids
        }
        shifted = tuple(
            mechanism_id
            for mechanism_id in self.mechanism_ids
            if diagnostics[mechanism_id][3]
        )
        update_ids = shifted if shifted else self.mechanism_ids
        update_set = set(update_ids)
        records: list[MechanismObservation] = []
        for mechanism_id in self.mechanism_ids:
            baseline, scale, score, shifted_flag = diagnostics[mechanism_id]
            updated = mechanism_id in update_set
            factor: float | None = None
            if updated:
                factor = (
                    self.config.shift_forgetting_factor
                    if shifted
                    else self.config.stable_forgetting_factor
                )
                local_features = self._features(features, self._specs[mechanism_id])
                self._learners[mechanism_id].update(
                    local_features,
                    target_values[mechanism_id],
                    forgetting_factor=factor,
                )
            self._histories[mechanism_id].append(residuals[mechanism_id])
            records.append(
                MechanismObservation(
                    mechanism_id=mechanism_id,
                    prediction=predictions[mechanism_id],
                    target=target_values[mechanism_id],
                    residual=residuals[mechanism_id],
                    baseline=baseline,
                    scale=scale,
                    score=score,
                    shifted=shifted_flag,
                    updated=updated,
                    forgetting_factor=factor,
                )
            )
        self._step += 1
        return MechanismUpdate(
            step=self._step,
            prediction=sum(predictions.values()),
            target=sum(target_values.values()),
            residual=sum(residuals.values()),
            shifted_mechanisms=shifted,
            updated_mechanisms=tuple(update_ids),
            observations=tuple(records),
        )

    def state_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": MECHANISM_SCHEMA,
            "version": MECHANISM_VERSION,
            "config": self.config.to_dict(),
            "step": self._step,
            "learners": {
                mechanism_id: self._learners[mechanism_id].state_dict()
                for mechanism_id in self.mechanism_ids
            },
            "residual_histories": {
                mechanism_id: list(self._histories[mechanism_id])
                for mechanism_id in self.mechanism_ids
            },
        }
        payload["digest"] = _digest({key: value for key, value in payload.items()})
        return payload

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "MechanismLocalizedRLS":
        if not isinstance(state, Mapping):
            raise TypeError("state must be a mapping")
        expected = {"schema", "version", "config", "step", "learners", "residual_histories", "digest"}
        if set(state) != expected:
            raise ValueError("mechanism state has unknown or missing fields")
        if state["schema"] != MECHANISM_SCHEMA or state["version"] != MECHANISM_VERSION:
            raise ValueError("unsupported mechanism state")
        unsigned = {key: state[key] for key in expected if key != "digest"}
        if state["digest"] != _digest(unsigned):
            raise ValueError("mechanism state digest mismatch")
        step = _integer("state.step", state["step"], minimum=0, maximum=10**12)
        config = MechanismLocalizedConfig.from_dict(state["config"])
        learners = state["learners"]
        histories = state["residual_histories"]
        if not isinstance(learners, Mapping) or not isinstance(histories, Mapping):
            raise ValueError("learners and residual_histories must be mappings")
        if set(learners) != set(config.mechanisms[i].mechanism_id for i in range(len(config.mechanisms))):
            raise ValueError("learner identities do not match config")
        if set(histories) != set(learners):
            raise ValueError("history identities do not match learners")
        result = cls(config)
        result._step = step
        result._learners = {
            mechanism_id: RLS.from_state_dict(learners[mechanism_id])
            for mechanism_id in result.mechanism_ids
        }
        for mechanism_id in result.mechanism_ids:
            values = histories[mechanism_id]
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
                raise ValueError("residual history must be a sequence")
            if len(values) > config.residual_history:
                raise ValueError("residual history exceeds configured bound")
            result._histories[mechanism_id].extend(
                _finite(f"residual_histories[{mechanism_id!r}]", value)
                for value in values
            )
        return result

    def digest(self) -> str:
        return self.state_dict()["digest"]
