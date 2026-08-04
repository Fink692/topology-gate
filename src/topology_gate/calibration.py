"""Bounded empirical calibration for causal detector alarms.

This module does not turn a heuristic detector into a likelihood-ratio test.
It provides a reproducible null/power experiment with explicit finite-horizon
uncertainty, censoring, and detector identity so that thresholds and
score-to-memory policies can be evaluated honestly.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Callable

try:
    import numpy as np
except ModuleNotFoundError as exc:  # pragma: no cover - exercised by import boundary.
    raise ImportError(
        "topology_gate.calibration requires NumPy; install "
        "`topology-gate[numeric]`"
    ) from exc


MAX_CALIBRATION_TRIALS = 4_096
MAX_CALIBRATION_HORIZON = 100_000
MAX_CALIBRATION_FEATURES = 256
_WILSON_Z_95 = 1.959963984540054

ObservationFactory = Callable[[np.random.Generator, int, int], Any]
DetectorFactory = Callable[[], Any]


def _bounded_int(name: str, value: Any, minimum: int, maximum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum or result > maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return result


def _finite_probability(name: str, value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite probability") from exc
    if not math.isfinite(result) or not 0.0 < result < 1.0:
        raise ValueError(f"{name} must lie strictly between zero and one")
    return result


@dataclass(frozen=True, slots=True)
class CalibrationConfig:
    """Finite experiment limits and reproducibility settings."""

    trials: int = 1_000
    horizon: int = 512
    n_features: int = 1
    alpha: float = 0.05
    seed: int = 7

    def __post_init__(self) -> None:
        trials = _bounded_int("trials", self.trials, 1, MAX_CALIBRATION_TRIALS)
        horizon = _bounded_int("horizon", self.horizon, 2, MAX_CALIBRATION_HORIZON)
        features = _bounded_int("n_features", self.n_features, 1, MAX_CALIBRATION_FEATURES)
        alpha = _finite_probability("alpha", self.alpha)
        if isinstance(self.seed, (bool, np.bool_)) or not isinstance(
            self.seed, (int, np.integer)
        ):
            raise ValueError("seed must be an integer")
        seed = int(self.seed)
        if seed < 0:
            raise ValueError("seed must be non-negative")
        object.__setattr__(self, "trials", trials)
        object.__setattr__(self, "horizon", horizon)
        object.__setattr__(self, "n_features", features)
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "seed", seed)


def _wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    if not 0 <= successes <= trials or trials < 1:
        raise ValueError("successes must be between zero and trials")
    proportion = successes / trials
    z2 = _WILSON_Z_95 * _WILSON_Z_95
    denominator = 1.0 + z2 / trials
    center = (proportion + z2 / (2.0 * trials)) / denominator
    radius = (
        _WILSON_Z_95
        * math.sqrt(
            proportion * (1.0 - proportion) / trials
            + z2 / (4.0 * trials * trials)
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def _identity(detector: Any) -> str:
    value = getattr(detector, "config_identity", None)
    if callable(value):
        value = value()
    if value is None:
        value = f"{type(detector).__module__}:{type(detector).__qualname__}"
    if not isinstance(value, str) or not value:
        raise ValueError("detector identity must be a non-empty string")
    return value


def _observations(
    factory: ObservationFactory,
    rng: np.random.Generator,
    config: CalibrationConfig,
) -> np.ndarray[Any, Any]:
    if not callable(factory):
        raise ValueError("observation_factory must be callable")
    values = np.asarray(factory(rng, config.horizon, config.n_features), dtype=float)
    if values.ndim == 1:
        if config.n_features != 1:
            raise ValueError("one-dimensional observations require n_features=1")
        values = values.reshape(-1, 1)
    if values.shape != (config.horizon, config.n_features):
        raise ValueError(
            "observation_factory must return an array shaped "
            f"({config.horizon}, {config.n_features})"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("observation_factory returned non-finite values")
    return np.array(values, copy=True)


def _alarms(detector: Any, values: np.ndarray[Any, Any], horizon: int) -> np.ndarray[Any, Any]:
    detect = getattr(detector, "detect", None)
    if not callable(detect):
        raise ValueError("detector_factory must return an object with detect")
    result = detect(values)
    alarms = np.asarray(getattr(result, "alarms", None), dtype=bool).reshape(-1)
    if alarms.size != horizon:
        raise ValueError("detector alarms must align with the calibration horizon")
    return alarms


@dataclass(frozen=True, slots=True)
class NullCalibrationResult:
    """Finite-horizon null alarm evidence with a 95% Wilson interval."""

    detector_identity: str
    trials: int
    horizon: int
    n_features: int
    alpha: float
    seed: int
    false_alarm_count: int
    false_alarm_rate: float
    false_alarm_ci_low: float
    false_alarm_ci_high: float
    average_run_length: float
    censored_run_fraction: float
    first_alarm_steps: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.first_alarm_steps) != self.trials:
            raise ValueError("first_alarm_steps must have one value per trial")
        if not 0 <= self.false_alarm_count <= self.trials:
            raise ValueError("false_alarm_count must be between zero and trials")
        fields = (
            self.false_alarm_rate,
            self.false_alarm_ci_low,
            self.false_alarm_ci_high,
            self.average_run_length,
            self.censored_run_fraction,
        )
        if not all(math.isfinite(float(value)) for value in fields):
            raise ValueError("calibration result contains a non-finite value")

    @property
    def config_identity(self) -> str:
        payload = {
            "detector_identity": self.detector_identity,
            "trials": self.trials,
            "horizon": self.horizon,
            "n_features": self.n_features,
            "alpha": self.alpha,
            "seed": self.seed,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "null_calibration",
            "detector_identity": self.detector_identity,
            "trials": self.trials,
            "horizon": self.horizon,
            "n_features": self.n_features,
            "alpha": self.alpha,
            "seed": self.seed,
            "false_alarm_count": self.false_alarm_count,
            "false_alarm_rate": self.false_alarm_rate,
            "false_alarm_ci_95": [self.false_alarm_ci_low, self.false_alarm_ci_high],
            "average_run_length_censored": self.average_run_length,
            "censored_run_fraction": self.censored_run_fraction,
            "first_alarm_steps": list(self.first_alarm_steps),
            "config_identity": self.config_identity,
        }


@dataclass(frozen=True, slots=True)
class ShiftCalibrationResult:
    """Finite-horizon detection power and delay evidence."""

    detector_identity: str
    trials: int
    horizon: int
    shift_index: int
    detection_count: int
    detection_rate: float
    detection_ci_low: float
    detection_ci_high: float
    mean_delay_with_censoring: float
    censored_fraction: float
    detection_delays: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "shift_calibration",
            "detector_identity": self.detector_identity,
            "trials": self.trials,
            "horizon": self.horizon,
            "shift_index": self.shift_index,
            "detection_count": self.detection_count,
            "detection_rate": self.detection_rate,
            "detection_ci_95": [self.detection_ci_low, self.detection_ci_high],
            "mean_delay_with_censoring": self.mean_delay_with_censoring,
            "censored_fraction": self.censored_fraction,
            "detection_delays": list(self.detection_delays),
        }


def calibrate_null(
    detector_factory: DetectorFactory,
    observation_factory: ObservationFactory,
    *,
    config: CalibrationConfig | None = None,
) -> NullCalibrationResult:
    """Estimate finite-horizon false alarms under a declared null factory."""

    settings = config or CalibrationConfig()
    if not callable(detector_factory):
        raise ValueError("detector_factory must be callable")
    probe = detector_factory()
    identity = _identity(probe)
    first_alarm_steps: list[int] = []
    rng = np.random.default_rng(settings.seed)
    for _ in range(settings.trials):
        detector = detector_factory()
        if _identity(detector) != identity:
            raise ValueError("detector_factory returned inconsistent identities")
        values = _observations(observation_factory, rng, settings)
        alarms = _alarms(detector, values, settings.horizon)
        hits = np.flatnonzero(alarms)
        first_alarm_steps.append(int(hits[0]) if hits.size else settings.horizon)
    false_count = sum(step < settings.horizon for step in first_alarm_steps)
    rate = false_count / settings.trials
    low, high = _wilson_interval(false_count, settings.trials)
    run_lengths = [step + 1 for step in first_alarm_steps]
    return NullCalibrationResult(
        detector_identity=identity,
        trials=settings.trials,
        horizon=settings.horizon,
        n_features=settings.n_features,
        alpha=settings.alpha,
        seed=settings.seed,
        false_alarm_count=false_count,
        false_alarm_rate=rate,
        false_alarm_ci_low=low,
        false_alarm_ci_high=high,
        average_run_length=float(np.mean(run_lengths)),
        censored_run_fraction=float(
            sum(step == settings.horizon for step in first_alarm_steps)
            / settings.trials
        ),
        first_alarm_steps=tuple(first_alarm_steps),
    )


def calibrate_shift(
    detector_factory: DetectorFactory,
    observation_factory: ObservationFactory,
    *,
    shift_index: int,
    config: CalibrationConfig | None = None,
) -> ShiftCalibrationResult:
    """Estimate detection power and delay after a declared synthetic shift."""

    settings = config or CalibrationConfig()
    shift = _bounded_int("shift_index", shift_index, 0, settings.horizon - 1)
    if not callable(detector_factory):
        raise ValueError("detector_factory must be callable")
    probe = detector_factory()
    identity = _identity(probe)
    delays: list[int] = []
    rng = np.random.default_rng(settings.seed)
    for _ in range(settings.trials):
        detector = detector_factory()
        if _identity(detector) != identity:
            raise ValueError("detector_factory returned inconsistent identities")
        values = _observations(observation_factory, rng, settings)
        alarms = _alarms(detector, values, settings.horizon)
        hits = np.flatnonzero(alarms[shift:])
        delays.append(int(hits[0]) if hits.size else settings.horizon - shift)
    detected = sum(delay < settings.horizon - shift for delay in delays)
    rate = detected / settings.trials
    low, high = _wilson_interval(detected, settings.trials)
    return ShiftCalibrationResult(
        detector_identity=identity,
        trials=settings.trials,
        horizon=settings.horizon,
        shift_index=shift,
        detection_count=detected,
        detection_rate=rate,
        detection_ci_low=low,
        detection_ci_high=high,
        mean_delay_with_censoring=float(np.mean(delays)),
        censored_fraction=float(
            sum(delay == settings.horizon - shift for delay in delays)
            / settings.trials
        ),
        detection_delays=tuple(delays),
    )


__all__ = [
    "CalibrationConfig",
    "MAX_CALIBRATION_FEATURES",
    "MAX_CALIBRATION_HORIZON",
    "MAX_CALIBRATION_TRIALS",
    "NullCalibrationResult",
    "ShiftCalibrationResult",
    "calibrate_null",
    "calibrate_shift",
]
