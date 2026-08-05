"""Bounded empirical calibration for causal detector alarms and e-processes.

This module does not turn a heuristic detector into a likelihood-ratio test.
It provides a reproducible null/power experiment with explicit finite-horizon
uncertainty, censoring, and detector identity so that thresholds and
score-to-memory policies can be evaluated honestly.  The e-process harness
below is an optional-stopping simulation of declared bounded score streams; it
does not establish the conditional-mean null for a market comparison.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

from .promotion import EProcess, PromotionGate, geometric_alpha_allocation, validate_eta
from .selection import SelectionBudget

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
MAX_CALIBRATION_SOURCE_ROWS = 100_000
MAX_CALIBRATION_CHALLENGERS = 64
MAX_CALIBRATION_EPOCHS = 64
MAX_CALIBRATION_THRESHOLDS = 64
CALIBRATION_CERTIFICATE_VERSION = 2
_WILSON_Z_95 = 1.959963984540054
_WILSON_CONFIDENCE_95 = 0.95

ObservationFactory = Callable[[np.random.Generator, int, int], Any]
DetectorFactory = Callable[[], Any]
EProcessScoreFactory = Callable[[np.random.Generator, int], Any]
PromotionScoreFactory = Callable[[np.random.Generator, int, int], Any]
SelectionScoreFactory = Callable[[np.random.Generator, int, int], Any]


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


def _unit_probability(name: str, value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite probability") from exc
    if not math.isfinite(result) or not 0.0 < result <= 1.0:
        raise ValueError(f"{name} must lie in (0, 1]")
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


@dataclass(frozen=True, slots=True)
class EProcessCalibrationConfig:
    """Finite optional-stopping simulation limits for bounded e-processes."""

    trials: int = 1_000
    horizon: int = 512
    alpha: float = 0.05
    eta: float = 0.5
    initial_wealth: float = 1.0
    seed: int = 7

    def __post_init__(self) -> None:
        trials = _bounded_int("trials", self.trials, 1, MAX_CALIBRATION_TRIALS)
        horizon = _bounded_int("horizon", self.horizon, 2, MAX_CALIBRATION_HORIZON)
        alpha = _finite_probability("alpha", self.alpha)
        eta = validate_eta(self.eta)
        try:
            initial_wealth = float(self.initial_wealth)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("initial_wealth must be finite and positive") from exc
        if not math.isfinite(initial_wealth) or initial_wealth <= 0.0:
            raise ValueError("initial_wealth must be finite and positive")
        if isinstance(self.seed, (bool, np.bool_)) or not isinstance(
            self.seed, (int, np.integer)
        ):
            raise ValueError("seed must be an integer")
        seed = int(self.seed)
        if seed < 0:
            raise ValueError("seed must be non-negative")
        object.__setattr__(self, "trials", trials)
        object.__setattr__(self, "horizon", horizon)
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "eta", eta)
        object.__setattr__(self, "initial_wealth", initial_wealth)
        object.__setattr__(self, "seed", seed)


@dataclass(frozen=True, slots=True)
class PromotionCalibrationConfig:
    """Finite null-simulation settings for a complete promotion gate.

    ``challengers`` are registered before any scores are observed, so the
    geometric alpha allocation and the candidate-selection rule are part of
    the experiment.  The harness simulates one gate epoch and stops a path at
    its first promotion.  It is evidence about this finite declared gate,
    not a market-validity or conditional-mean certificate.
    """

    trials: int = 1_000
    horizon: int = 512
    challengers: int = 1
    epochs: int = 1
    alpha: float = 0.05
    eta: float = 0.5
    initial_wealth: float = 1.0
    seed: int = 7

    def __post_init__(self) -> None:
        trials = _bounded_int("trials", self.trials, 1, MAX_CALIBRATION_TRIALS)
        horizon = _bounded_int("horizon", self.horizon, 2, MAX_CALIBRATION_HORIZON)
        challengers = _bounded_int(
            "challengers", self.challengers, 1, MAX_CALIBRATION_CHALLENGERS
        )
        epochs = _bounded_int("epochs", self.epochs, 1, MAX_CALIBRATION_EPOCHS)
        alpha = _finite_probability("alpha", self.alpha)
        eta = validate_eta(self.eta)
        try:
            initial_wealth = float(self.initial_wealth)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("initial_wealth must be finite and positive") from exc
        if not math.isfinite(initial_wealth) or initial_wealth <= 0.0:
            raise ValueError("initial_wealth must be finite and positive")
        if isinstance(self.seed, (bool, np.bool_)) or not isinstance(
            self.seed, (int, np.integer)
        ):
            raise ValueError("seed must be an integer")
        seed = int(self.seed)
        if seed < 0:
            raise ValueError("seed must be non-negative")
        object.__setattr__(self, "trials", trials)
        object.__setattr__(self, "horizon", horizon)
        object.__setattr__(self, "challengers", challengers)
        object.__setattr__(self, "epochs", epochs)
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "eta", eta)
        object.__setattr__(self, "initial_wealth", initial_wealth)
        object.__setattr__(self, "seed", seed)


@dataclass(frozen=True, slots=True)
class SelectionCalibrationConfig:
    """Finite null-simulation settings for a pre-registered selection family."""

    budget: SelectionBudget
    trials: int = 1_000
    horizon: int = 512
    eta: float = 0.5
    initial_wealth: float = 1.0
    seed: int = 7

    def __post_init__(self) -> None:
        if not isinstance(self.budget, SelectionBudget):
            raise TypeError("budget must be a SelectionBudget")
        if self.budget.total_slots > MAX_CALIBRATION_CHALLENGERS:
            raise ValueError(
                "selection calibration exceeds the configured cell limit"
            )
        trials = _bounded_int("trials", self.trials, 1, MAX_CALIBRATION_TRIALS)
        horizon = _bounded_int("horizon", self.horizon, 2, MAX_CALIBRATION_HORIZON)
        eta = validate_eta(self.eta)
        try:
            initial_wealth = float(self.initial_wealth)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("initial_wealth must be finite and positive") from exc
        if not math.isfinite(initial_wealth) or initial_wealth <= 0.0:
            raise ValueError("initial_wealth must be finite and positive")
        if isinstance(self.seed, (bool, np.bool_)) or not isinstance(
            self.seed, (int, np.integer)
        ):
            raise ValueError("seed must be an integer")
        seed = int(self.seed)
        if seed < 0:
            raise ValueError("seed must be non-negative")
        object.__setattr__(self, "trials", trials)
        object.__setattr__(self, "horizon", horizon)
        object.__setattr__(self, "eta", eta)
        object.__setattr__(self, "initial_wealth", initial_wealth)
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


def _factory_identity(factory: Any) -> str:
    if not callable(factory):
        raise ValueError("observation_factory must be callable")
    value = getattr(factory, "identity", None)
    if callable(value):
        value = value()
    if value is None:
        code = getattr(factory, "__code__", None)
        location = ""
        if code is not None:
            # Keep receipts portable: absolute checkout paths are neither
            # useful for reproducibility nor appropriate in public artifacts.
            location = f":{Path(code.co_filename).name}:{code.co_firstlineno}"
        value = (
            f"{getattr(factory, '__module__', type(factory).__module__)}:"
            f"{getattr(factory, '__qualname__', type(factory).__qualname__)}"
            f"{location}"
        )
    if not isinstance(value, str) or not value:
        raise ValueError("observation factory identity must be a non-empty string")
    return value


def _score_factory_identity(factory: Any) -> str:
    if not callable(factory):
        raise ValueError("score_factory must be callable")
    value = getattr(factory, "identity", None)
    if callable(value):
        value = value()
    if value is None:
        code = getattr(factory, "__code__", None)
        location = ""
        if code is not None:
            location = f":{Path(code.co_filename).name}:{code.co_firstlineno}"
        value = (
            f"{getattr(factory, '__module__', type(factory).__module__)}:"
            f"{getattr(factory, '__qualname__', type(factory).__qualname__)}"
            f"{location}"
        )
    if not isinstance(value, str) or not value:
        raise ValueError("score factory identity must be a non-empty string")
    return value


class _ThresholdSplitObservationFactory:
    """Bind an observation factory to one threshold-calibration split role."""

    __slots__ = ("_factory", "identity")

    def __init__(self, factory: ObservationFactory, role: str) -> None:
        self._factory = factory
        self.identity = f"{_factory_identity(factory)}:threshold-split:{role}:v1"

    def __call__(
        self, rng: np.random.Generator, horizon: int, n_features: int
    ) -> Any:
        return self._factory(rng, horizon, n_features)


class StationaryBlockBootstrap:
    """Seeded stationary block bootstrap observation factory.

    The factory samples a circular source sequence and starts a new block
    with probability ``restart_probability`` on each step.  It preserves
    local serial dependence in the declared source but does not establish
    exchangeability or a market-valid null by itself.
    """

    def __init__(
        self,
        source: Any,
        *,
        block_length: int = 16,
        source_id: str = "source",
        restart_probability: float | None = None,
    ) -> None:
        values = np.asarray(source, dtype=float)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
            raise ValueError("bootstrap source must be a non-empty 2-D sequence")
        if values.shape[0] > MAX_CALIBRATION_SOURCE_ROWS:
            raise ValueError("bootstrap source exceeds its row limit")
        if values.shape[1] > MAX_CALIBRATION_FEATURES:
            raise ValueError("bootstrap source exceeds its feature limit")
        if not np.all(np.isfinite(values)):
            raise ValueError("bootstrap source contains non-finite values")
        if not isinstance(source_id, str) or not source_id:
            raise ValueError("source_id must be a non-empty string")
        block = _bounded_int(
            "block_length", block_length, 1, int(values.shape[0])
        )
        restart = (
            1.0 / block
            if restart_probability is None
            else _unit_probability("restart_probability", restart_probability)
        )
        self._source = np.array(values, dtype=float, copy=True, order="C")
        self._source.setflags(write=False)
        self._source_id = source_id
        self._block_length = block
        self._restart_probability = restart
        self._source_digest = hashlib.sha256(self._source.tobytes()).hexdigest()

    @property
    def source_id(self) -> str:
        return self._source_id

    @property
    def source_shape(self) -> tuple[int, int]:
        return (int(self._source.shape[0]), int(self._source.shape[1]))

    @property
    def block_length(self) -> int:
        return self._block_length

    @property
    def restart_probability(self) -> float:
        return self._restart_probability

    @property
    def identity(self) -> str:
        payload = {
            "schema": "topology_gate.stationary_block_bootstrap",
            "version": 1,
            "source_id": self._source_id,
            "source_digest": self._source_digest,
            "source_shape": list(self.source_shape),
            "block_length": self._block_length,
            "restart_probability": self._restart_probability,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def __call__(
        self, rng: np.random.Generator, horizon: int, n_features: int
    ) -> np.ndarray[Any, Any]:
        if not isinstance(rng, np.random.Generator):
            raise ValueError("bootstrap requires a NumPy random generator")
        length = _bounded_int("horizon", horizon, 1, MAX_CALIBRATION_HORIZON)
        features = _bounded_int(
            "n_features", n_features, 1, MAX_CALIBRATION_FEATURES
        )
        if features != self._source.shape[1]:
            raise ValueError("bootstrap source feature dimension does not match calibration")
        output = np.empty((length, features), dtype=float)
        index = int(rng.integers(0, self._source.shape[0]))
        for row in range(length):
            output[row] = self._source[index]
            if row + 1 == length:
                break
            if float(rng.random()) < self._restart_probability:
                index = int(rng.integers(0, self._source.shape[0]))
            else:
                index = (index + 1) % self._source.shape[0]
        return output

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "topology_gate.stationary_block_bootstrap",
            "version": 1,
            "source_id": self._source_id,
            "source_digest": self._source_digest,
            "source_shape": list(self.source_shape),
            "block_length": self._block_length,
            "restart_probability": self._restart_probability,
            "identity": self.identity,
        }


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
    observation_identity: str = "unbound"

    def __post_init__(self) -> None:
        if len(self.first_alarm_steps) != self.trials:
            raise ValueError("first_alarm_steps must have one value per trial")
        if not 0 <= self.false_alarm_count <= self.trials:
            raise ValueError("false_alarm_count must be between zero and trials")
        if not isinstance(self.observation_identity, str) or not self.observation_identity:
            raise ValueError("observation_identity must be a non-empty string")
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
            "observation_identity": self.observation_identity,
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
            "observation_identity": self.observation_identity,
            "false_alarm_count": self.false_alarm_count,
            "false_alarm_rate": self.false_alarm_rate,
            "false_alarm_ci_95": [self.false_alarm_ci_low, self.false_alarm_ci_high],
            "average_run_length_censored": self.average_run_length,
            "censored_run_fraction": self.censored_run_fraction,
            "first_alarm_steps": list(self.first_alarm_steps),
            "config_identity": self.config_identity,
        }

    def to_certificate(self, *, max_false_alarm_rate: float) -> "CalibrationCertificate":
        """Create a finite-null certificate for a declared alarm budget.

        The certificate authorizes only the detector identity and null
        experiment supplied here.  It is not a market-calibration claim.
        Wilson bounds in this result are fixed at 95%, so the resulting
        certificate records that confidence level explicitly.
        """

        return CalibrationCertificate(
            detector_identity=self.detector_identity,
            null_config_identity=self.config_identity,
            trials=self.trials,
            horizon=self.horizon,
            false_alarm_count=self.false_alarm_count,
            false_alarm_rate=self.false_alarm_rate,
            false_alarm_ci_high=self.false_alarm_ci_high,
            max_false_alarm_rate=max_false_alarm_rate,
        )


@dataclass(frozen=True, slots=True)
class CalibrationCertificate:
    """Explicit finite-null authorization for detector-driven acceleration."""

    detector_identity: str
    null_config_identity: str
    trials: int
    horizon: int
    false_alarm_count: int
    false_alarm_rate: float
    false_alarm_ci_high: float
    max_false_alarm_rate: float
    confidence: float = _WILSON_CONFIDENCE_95
    selection_identity: str | None = None

    def __post_init__(self) -> None:
        for name in ("detector_identity", "null_config_identity"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        trials = _bounded_int("trials", self.trials, 1, MAX_CALIBRATION_TRIALS)
        horizon = _bounded_int("horizon", self.horizon, 2, MAX_CALIBRATION_HORIZON)
        if isinstance(self.false_alarm_count, (bool, np.bool_)) or not isinstance(
            self.false_alarm_count, (int, np.integer)
        ):
            raise ValueError("false_alarm_count must be an integer")
        false_count = int(self.false_alarm_count)
        if not 0 <= false_count <= trials:
            raise ValueError("false_alarm_count must be between zero and trials")
        rate = float(self.false_alarm_rate)
        upper = float(self.false_alarm_ci_high)
        if not all(math.isfinite(value) for value in (rate, upper)):
            raise ValueError("certificate alarm rates must be finite")
        if not 0.0 <= rate <= 1.0 or not 0.0 <= upper <= 1.0 or rate > upper:
            raise ValueError("certificate alarm rates are invalid")
        expected_rate = false_count / trials
        _, expected_upper = _wilson_interval(false_count, trials)
        if not math.isclose(rate, expected_rate, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError(
                "certificate false_alarm_rate is inconsistent with false_alarm_count"
            )
        if not math.isclose(upper, expected_upper, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError(
                "certificate false_alarm_ci_high is inconsistent with false_alarm_count"
            )
        max_rate = _finite_probability(
            "max_false_alarm_rate", self.max_false_alarm_rate
        )
        confidence = _finite_probability("confidence", self.confidence)
        if not math.isclose(
            confidence, _WILSON_CONFIDENCE_95, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise ValueError("certificate confidence must match the Wilson 95% interval")
        if self.selection_identity is not None and (
            not isinstance(self.selection_identity, str)
            or not self.selection_identity.strip()
        ):
            raise ValueError("selection_identity must be a non-empty string when set")
        object.__setattr__(self, "trials", trials)
        object.__setattr__(self, "horizon", horizon)
        object.__setattr__(self, "false_alarm_count", false_count)
        object.__setattr__(self, "false_alarm_rate", rate)
        object.__setattr__(self, "false_alarm_ci_high", upper)
        object.__setattr__(self, "max_false_alarm_rate", max_rate)
        object.__setattr__(self, "confidence", confidence)

    @property
    def approved(self) -> bool:
        """Whether the conservative finite-null bound passes the declared budget."""

        return self.false_alarm_ci_high <= self.max_false_alarm_rate

    @property
    def identity(self) -> str:
        payload = {
            "schema": "topology_gate.calibration_certificate",
            "version": CALIBRATION_CERTIFICATE_VERSION,
            "detector_identity": self.detector_identity,
            "null_config_identity": self.null_config_identity,
            "trials": self.trials,
            "horizon": self.horizon,
            "false_alarm_count": self.false_alarm_count,
            "false_alarm_rate": self.false_alarm_rate,
            "false_alarm_ci_high": self.false_alarm_ci_high,
            "max_false_alarm_rate": self.max_false_alarm_rate,
            "confidence": self.confidence,
            "selection_identity": self.selection_identity,
            "approved": self.approved,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "topology_gate.calibration_certificate",
            "version": CALIBRATION_CERTIFICATE_VERSION,
            "detector_identity": self.detector_identity,
            "null_config_identity": self.null_config_identity,
            "trials": self.trials,
            "horizon": self.horizon,
            "false_alarm_count": self.false_alarm_count,
            "false_alarm_rate": self.false_alarm_rate,
            "false_alarm_ci_high": self.false_alarm_ci_high,
            "max_false_alarm_rate": self.max_false_alarm_rate,
            "confidence": self.confidence,
            "selection_identity": self.selection_identity,
            "approved": self.approved,
            "identity": self.identity,
        }


@dataclass(frozen=True, slots=True)
class ThresholdCalibrationResult:
    """Independent threshold-selection and evaluation evidence.

    Candidate thresholds are selected only from the calibration split.  The
    evaluation result is generated by a distinct predeclared observation
    factory and seed, and is the only result eligible to authorize
    detector-driven acceleration.
    """

    detector_family_identity: str
    detector_factory_identity: str
    calibration_observation_identity: str
    evaluation_observation_identity: str
    candidate_thresholds: tuple[float, ...]
    selected_threshold: float
    max_false_alarm_rate: float
    calibration_results: tuple[NullCalibrationResult, ...]
    evaluation_result: NullCalibrationResult

    def __post_init__(self) -> None:
        for name in (
            "detector_family_identity",
            "detector_factory_identity",
            "calibration_observation_identity",
            "evaluation_observation_identity",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if isinstance(self.candidate_thresholds, (str, bytes, bytearray)):
            raise ValueError("candidate_thresholds must be a numeric sequence")
        raw_thresholds = tuple(self.candidate_thresholds)
        if not raw_thresholds or len(raw_thresholds) > MAX_CALIBRATION_THRESHOLDS:
            raise ValueError(
                "candidate_thresholds must contain between one and "
                f"{MAX_CALIBRATION_THRESHOLDS} values"
            )
        thresholds = tuple(float(value) for value in raw_thresholds)
        if not all(math.isfinite(value) and value > 0.0 for value in thresholds):
            raise ValueError("candidate_thresholds must be finite and positive")
        if tuple(sorted(thresholds)) != thresholds:
            raise ValueError("candidate_thresholds must be sorted")
        if len(set(thresholds)) != len(thresholds):
            raise ValueError("candidate_thresholds must be unique")
        selected = float(self.selected_threshold)
        if not math.isfinite(selected) or selected not in thresholds:
            raise ValueError("selected_threshold must be one of candidate_thresholds")
        max_rate = _finite_probability(
            "max_false_alarm_rate", self.max_false_alarm_rate
        )
        results = tuple(self.calibration_results)
        if len(results) != len(thresholds) or not all(
            isinstance(item, NullCalibrationResult) for item in results
        ):
            raise ValueError(
                "calibration_results must align with candidate_thresholds"
            )
        if not isinstance(self.evaluation_result, NullCalibrationResult):
            raise ValueError("evaluation_result must be NullCalibrationResult")
        if any(
            item.observation_identity != self.calibration_observation_identity
            for item in results
        ):
            raise ValueError("calibration observation identity does not match")
        if (
            self.evaluation_result.observation_identity
            != self.evaluation_observation_identity
        ):
            raise ValueError("evaluation observation identity does not match")
        reference = results[0]
        reference_shape = (
            reference.trials,
            reference.horizon,
            reference.n_features,
            reference.alpha,
            reference.seed,
        )
        if any(
            (
                item.trials,
                item.horizon,
                item.n_features,
                item.alpha,
                item.seed,
            )
            != reference_shape
            for item in results
        ):
            raise ValueError(
                "calibration results must share one independent split configuration"
            )
        evaluation_shape = (
            self.evaluation_result.trials,
            self.evaluation_result.horizon,
            self.evaluation_result.n_features,
            self.evaluation_result.alpha,
        )
        if evaluation_shape != reference_shape[:4]:
            raise ValueError(
                "evaluation result must share horizon, feature count, and alpha"
            )
        if self.evaluation_result.seed == reference.seed:
            raise ValueError("evaluation result must use a distinct split seed")
        detector_identities = tuple(item.detector_identity for item in results)
        if len(set(detector_identities)) != len(detector_identities):
            raise ValueError(
                "candidate detector identities must be unique and threshold-bound"
            )
        selected_index = thresholds.index(selected)
        approved_indices = tuple(
            index
            for index, item in enumerate(results)
            if item.false_alarm_ci_high <= max_rate
        )
        if not approved_indices:
            raise ValueError(
                "no candidate threshold satisfies the calibration false-alarm budget"
            )
        if selected_index != approved_indices[0]:
            raise ValueError(
                "selected_threshold must be the smallest calibration-approved candidate"
            )
        if (
            self.evaluation_result.detector_identity
            != results[selected_index].detector_identity
        ):
            raise ValueError(
                "evaluation detector identity does not match selected threshold"
            )
        object.__setattr__(self, "candidate_thresholds", thresholds)
        object.__setattr__(self, "selected_threshold", selected)
        object.__setattr__(self, "max_false_alarm_rate", max_rate)
        object.__setattr__(self, "calibration_results", results)

    @property
    def selected_index(self) -> int:
        return self.candidate_thresholds.index(self.selected_threshold)

    @property
    def approved(self) -> bool:
        """Whether the untouched evaluation split passes the declared budget."""

        return self.evaluation_result.false_alarm_ci_high <= self.max_false_alarm_rate

    @property
    def identity(self) -> str:
        payload = {
            "schema": "topology_gate.threshold_calibration",
            "version": 2,
            "detector_family_identity": self.detector_family_identity,
            "detector_factory_identity": self.detector_factory_identity,
            "calibration_observation_identity": self.calibration_observation_identity,
            "evaluation_observation_identity": self.evaluation_observation_identity,
            "candidate_thresholds": list(self.candidate_thresholds),
            "selected_threshold": self.selected_threshold,
            "max_false_alarm_rate": self.max_false_alarm_rate,
            "calibration_config_identities": [
                item.config_identity for item in self.calibration_results
            ],
            "evaluation_config_identity": self.evaluation_result.config_identity,
            "approved": self.approved,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def to_certificate(self) -> CalibrationCertificate:
        """Authorize only the selected detector after evaluation approval."""

        if not self.approved:
            raise ValueError(
                "evaluation false-alarm bound does not satisfy the declared budget"
            )
        certificate = self.evaluation_result.to_certificate(
            max_false_alarm_rate=self.max_false_alarm_rate
        )
        return replace(certificate, selection_identity=self.identity)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "threshold_calibration",
            "schema": "topology_gate.threshold_calibration",
            "version": 2,
            "detector_family_identity": self.detector_family_identity,
            "detector_factory_identity": self.detector_factory_identity,
            "calibration_observation_identity": self.calibration_observation_identity,
            "evaluation_observation_identity": self.evaluation_observation_identity,
            "candidate_thresholds": list(self.candidate_thresholds),
            "selected_threshold": self.selected_threshold,
            "selected_index": self.selected_index,
            "max_false_alarm_rate": self.max_false_alarm_rate,
            "calibration_results": [item.to_dict() for item in self.calibration_results],
            "evaluation_result": self.evaluation_result.to_dict(),
            "approved": self.approved,
            "identity": self.identity,
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
    observation_identity: str = "unbound"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "shift_calibration",
            "detector_identity": self.detector_identity,
            "trials": self.trials,
            "horizon": self.horizon,
            "shift_index": self.shift_index,
            "observation_identity": self.observation_identity,
            "detection_count": self.detection_count,
            "detection_rate": self.detection_rate,
            "detection_ci_95": [self.detection_ci_low, self.detection_ci_high],
            "mean_delay_with_censoring": self.mean_delay_with_censoring,
            "censored_fraction": self.censored_fraction,
            "detection_delays": list(self.detection_delays),
        }


@dataclass(frozen=True, slots=True)
class EProcessNullCalibrationResult:
    """Finite optional-stopping crossing evidence for a bounded score stream.

    A first-crossing step equal to ``horizon`` is the explicit no-crossing
    sentinel for that finite path.
    """

    score_factory_identity: str
    trials: int
    horizon: int
    alpha: float
    eta: float
    initial_wealth: float
    seed: int
    threshold_crossing_count: int
    threshold_crossing_rate: float
    threshold_crossing_ci_low: float
    threshold_crossing_ci_high: float
    first_crossing_steps: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.score_factory_identity, str) or not self.score_factory_identity:
            raise ValueError("score_factory_identity must be a non-empty string")
        _bounded_int("trials", self.trials, 1, MAX_CALIBRATION_TRIALS)
        _bounded_int("horizon", self.horizon, 2, MAX_CALIBRATION_HORIZON)
        _finite_probability("alpha", self.alpha)
        validate_eta(self.eta)
        if not math.isfinite(float(self.initial_wealth)) or self.initial_wealth <= 0.0:
            raise ValueError("initial_wealth must be finite and positive")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if len(self.first_crossing_steps) != self.trials:
            raise ValueError("first_crossing_steps must have one value per trial")
        if not 0 <= self.threshold_crossing_count <= self.trials:
            raise ValueError("threshold_crossing_count must be between zero and trials")
        if not all(
            math.isfinite(float(value))
            for value in (
                self.threshold_crossing_rate,
                self.threshold_crossing_ci_low,
                self.threshold_crossing_ci_high,
            )
        ):
            raise ValueError("e-process calibration contains a non-finite value")
        if not 0.0 <= self.threshold_crossing_rate <= 1.0:
            raise ValueError("threshold_crossing_rate must be in [0, 1]")
        if not 0.0 <= self.threshold_crossing_ci_low <= self.threshold_crossing_ci_high <= 1.0:
            raise ValueError("threshold crossing interval is invalid")
        if any(step < 0 or step > self.horizon for step in self.first_crossing_steps):
            raise ValueError("first_crossing_steps must be valid zero-based steps")
        expected_rate = self.threshold_crossing_count / self.trials
        if not math.isclose(
            self.threshold_crossing_rate, expected_rate, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise ValueError(
                "threshold_crossing_rate is inconsistent with threshold_crossing_count"
            )
        expected_count = sum(step < self.horizon for step in self.first_crossing_steps)
        if expected_count != self.threshold_crossing_count:
            raise ValueError(
                "threshold_crossing_count is inconsistent with first_crossing_steps"
            )

    @property
    def threshold(self) -> float:
        """The optional-stopping e-value threshold used by the simulation."""

        return self.initial_wealth / self.alpha

    @property
    def config_identity(self) -> str:
        payload = {
            "score_factory_identity": self.score_factory_identity,
            "trials": self.trials,
            "horizon": self.horizon,
            "alpha": self.alpha,
            "eta": self.eta,
            "initial_wealth": self.initial_wealth,
            "seed": self.seed,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "eprocess_null_calibration",
            "score_factory_identity": self.score_factory_identity,
            "trials": self.trials,
            "horizon": self.horizon,
            "alpha": self.alpha,
            "eta": self.eta,
            "initial_wealth": self.initial_wealth,
            "threshold": self.threshold,
            "seed": self.seed,
            "threshold_crossing_count": self.threshold_crossing_count,
            "threshold_crossing_rate": self.threshold_crossing_rate,
            "threshold_crossing_ci_95": [
                self.threshold_crossing_ci_low,
                self.threshold_crossing_ci_high,
            ],
            "first_crossing_steps": list(self.first_crossing_steps),
            "config_identity": self.config_identity,
        }


@dataclass(frozen=True, slots=True)
class PromotionNullCalibrationResult:
    """Finite optional-stopping evidence for a complete promotion gate.

    ``first_promotion_steps`` use zero-based steps and ``horizon`` is the
    explicit no-promotion sentinel.  ``first_promoted_challenger_indices``
    uses ``-1`` for that sentinel.  Candidate streams are observed in their
    registered order at each step; once one candidate promotes, the gate is
    closed and no later candidate is observed on that path.
    """

    score_factory_identity: str
    trials: int
    horizon: int
    challenger_count: int
    epochs: int
    alpha: float
    eta: float
    initial_wealth: float
    seed: int
    challenger_alpha_allocations: tuple[float, ...]
    challenger_alpha_schedule: tuple[tuple[float, ...], ...]
    threshold_crossing_count: int
    threshold_crossing_rate: float
    threshold_crossing_ci_low: float
    threshold_crossing_ci_high: float
    first_promotion_epochs: tuple[int, ...]
    first_promotion_steps: tuple[int, ...]
    first_promoted_challenger_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.score_factory_identity, str) or not self.score_factory_identity:
            raise ValueError("score_factory_identity must be a non-empty string")
        trials = _bounded_int("trials", self.trials, 1, MAX_CALIBRATION_TRIALS)
        horizon = _bounded_int("horizon", self.horizon, 2, MAX_CALIBRATION_HORIZON)
        challengers = _bounded_int(
            "challenger_count", self.challenger_count, 1, MAX_CALIBRATION_CHALLENGERS
        )
        epochs = _bounded_int("epochs", self.epochs, 1, MAX_CALIBRATION_EPOCHS)
        alpha = _finite_probability("alpha", self.alpha)
        validate_eta(self.eta)
        if not math.isfinite(float(self.initial_wealth)) or self.initial_wealth <= 0.0:
            raise ValueError("initial_wealth must be finite and positive")
        if isinstance(self.seed, (bool, np.bool_)) or not isinstance(
            self.seed, (int, np.integer)
        ) or int(self.seed) < 0:
            raise ValueError("seed must be a non-negative integer")
        if len(self.challenger_alpha_allocations) != challengers:
            raise ValueError(
                "challenger_alpha_allocations must have one value per challenger"
            )
        normalized_allocations = tuple(
            float(value) for value in self.challenger_alpha_allocations
        )
        normalized_schedule = tuple(
            tuple(float(value) for value in allocation)
            for allocation in self.challenger_alpha_schedule
        )
        if len(normalized_schedule) != epochs:
            raise ValueError(
                "challenger_alpha_schedule must have one allocation per epoch"
            )
        expected_schedule = tuple(
            tuple(
                geometric_alpha_allocation(alpha, index + 1, epoch=epoch)
                for index in range(challengers)
            )
            for epoch in range(epochs)
        )
        for actual_epoch, expected_epoch in zip(
            normalized_schedule, expected_schedule
        ):
            if len(actual_epoch) != challengers:
                raise ValueError(
                    "challenger_alpha_schedule must have one value per challenger"
                )
            for actual, expected in zip(actual_epoch, expected_epoch):
                if not math.isfinite(actual) or actual <= 0.0:
                    raise ValueError("challenger alpha allocations must be positive")
                if not math.isclose(
                    actual, expected, rel_tol=0.0, abs_tol=1.0e-15
                ):
                    raise ValueError(
                        "challenger alpha allocations do not match the gate rule"
                    )
        if any(
            not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-15)
            for actual, expected in zip(normalized_allocations, expected_schedule[0])
        ):
            raise ValueError(
                "challenger_alpha_allocations must match the first epoch schedule"
            )
        crossing_count = _bounded_int(
            "threshold_crossing_count",
            self.threshold_crossing_count,
            0,
            trials,
        )
        if not all(
            math.isfinite(float(value))
            for value in (
                self.threshold_crossing_rate,
                self.threshold_crossing_ci_low,
                self.threshold_crossing_ci_high,
            )
        ):
            raise ValueError("promotion calibration contains a non-finite value")
        if not 0.0 <= self.threshold_crossing_rate <= 1.0:
            raise ValueError("threshold_crossing_rate must be in [0, 1]")
        if not (
            0.0
            <= self.threshold_crossing_ci_low
            <= self.threshold_crossing_ci_high
            <= 1.0
        ):
            raise ValueError("threshold crossing interval is invalid")
        if len(self.first_promotion_epochs) != trials:
            raise ValueError("first_promotion_epochs must have one value per trial")
        if len(self.first_promotion_steps) != trials:
            raise ValueError("first_promotion_steps must have one value per trial")
        if len(self.first_promoted_challenger_indices) != trials:
            raise ValueError(
                "first_promoted_challenger_indices must have one value per trial"
            )
        normalized_epochs: list[int] = []
        normalized_steps: list[int] = []
        normalized_indices: list[int] = []
        for epoch, step, challenger in zip(
            self.first_promotion_epochs,
            self.first_promotion_steps,
            self.first_promoted_challenger_indices,
        ):
            if isinstance(epoch, (bool, np.bool_)) or not isinstance(
                epoch, (int, np.integer)
            ):
                raise ValueError("first_promotion_epochs must be integers")
            normalized_epoch = int(epoch)
            if normalized_epoch < 0 or normalized_epoch > epochs:
                raise ValueError("first_promotion_epochs must be valid epochs")
            if isinstance(step, (bool, np.bool_)) or not isinstance(
                step, (int, np.integer)
            ):
                raise ValueError("first_promotion_steps must be valid zero-based steps")
            normalized_step = int(step)
            if normalized_step < 0 or normalized_step > horizon:
                raise ValueError("first_promotion_steps must be valid zero-based steps")
            if isinstance(challenger, (bool, np.bool_)) or not isinstance(
                challenger, (int, np.integer)
            ):
                raise ValueError("promoted challenger indices must be integers")
            normalized_challenger = int(challenger)
            if normalized_epoch == epochs:
                if normalized_step != horizon:
                    raise ValueError(
                        "no-promotion paths must use the horizon step sentinel"
                    )
                if normalized_challenger != -1:
                    raise ValueError(
                        "no-promotion paths must use challenger index -1"
                    )
            else:
                if normalized_step >= horizon:
                    raise ValueError(
                        "promoted paths must use a step before the horizon"
                    )
                if normalized_challenger < 0 or normalized_challenger >= challengers:
                    raise ValueError("promoted challenger index is out of range")
            normalized_epochs.append(normalized_epoch)
            normalized_steps.append(normalized_step)
            normalized_indices.append(normalized_challenger)
        expected_rate = crossing_count / trials
        if not math.isclose(
            self.threshold_crossing_rate, expected_rate, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise ValueError(
                "threshold_crossing_rate is inconsistent with threshold_crossing_count"
            )
        expected_count = sum(epoch < epochs for epoch in normalized_epochs)
        if expected_count != crossing_count:
            raise ValueError(
                "threshold_crossing_count is inconsistent with first_promotion_steps"
            )
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "trials", trials)
        object.__setattr__(self, "horizon", horizon)
        object.__setattr__(self, "challenger_count", challengers)
        object.__setattr__(self, "epochs", epochs)
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "eta", validate_eta(self.eta))
        object.__setattr__(self, "initial_wealth", float(self.initial_wealth))
        object.__setattr__(
            self,
            "challenger_alpha_allocations",
            normalized_allocations,
        )
        object.__setattr__(self, "challenger_alpha_schedule", normalized_schedule)
        object.__setattr__(self, "threshold_crossing_count", crossing_count)
        object.__setattr__(self, "first_promotion_epochs", tuple(normalized_epochs))
        object.__setattr__(self, "first_promotion_steps", tuple(normalized_steps))
        object.__setattr__(
            self,
            "first_promoted_challenger_indices",
            tuple(normalized_indices),
        )

    @property
    def challenger_thresholds(self) -> tuple[float, ...]:
        """Per-slot e-value thresholds after geometric alpha allocation."""

        return tuple(
            self.initial_wealth / allocation
            for allocation in self.challenger_alpha_allocations
        )

    @property
    def config_identity(self) -> str:
        payload = {
            "score_factory_identity": self.score_factory_identity,
            "trials": self.trials,
            "horizon": self.horizon,
            "challenger_count": self.challenger_count,
            "epochs": self.epochs,
            "alpha": self.alpha,
            "eta": self.eta,
            "initial_wealth": self.initial_wealth,
            "seed": self.seed,
            "challenger_alpha_allocations": list(
                self.challenger_alpha_allocations
            ),
            "challenger_alpha_schedule": [
                list(allocation) for allocation in self.challenger_alpha_schedule
            ],
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "promotion_null_calibration",
            "score_factory_identity": self.score_factory_identity,
            "trials": self.trials,
            "horizon": self.horizon,
            "challenger_count": self.challenger_count,
            "alpha": self.alpha,
            "eta": self.eta,
            "initial_wealth": self.initial_wealth,
            "challenger_alpha_allocations": list(
                self.challenger_alpha_allocations
            ),
            "challenger_thresholds": list(self.challenger_thresholds),
            "seed": self.seed,
            "threshold_crossing_count": self.threshold_crossing_count,
            "threshold_crossing_rate": self.threshold_crossing_rate,
            "threshold_crossing_ci_95": [
                self.threshold_crossing_ci_low,
                self.threshold_crossing_ci_high,
            ],
            "first_promotion_epochs": list(self.first_promotion_epochs),
            "first_promotion_steps": list(self.first_promotion_steps),
            "first_promoted_challenger_indices": list(
                self.first_promoted_challenger_indices
            ),
            "config_identity": self.config_identity,
        }


@dataclass(frozen=True, slots=True)
class SelectionNullCalibrationResult:
    """Finite optional-stopping evidence across every selected family cell."""

    score_factory_identity: str
    selection_budget_identity: str
    trials: int
    horizon: int
    cell_count: int
    parent_alpha: float
    cell_alpha: float
    eta: float
    initial_wealth: float
    seed: int
    family_crossing_count: int
    family_crossing_rate: float
    family_crossing_ci_low: float
    family_crossing_ci_high: float
    first_crossing_steps: tuple[int, ...]
    first_crossing_cell_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.score_factory_identity, str) or not self.score_factory_identity:
            raise ValueError("score_factory_identity must be a non-empty string")
        if not isinstance(self.selection_budget_identity, str) or not self.selection_budget_identity:
            raise ValueError("selection_budget_identity must be a non-empty string")
        trials = _bounded_int("trials", self.trials, 1, MAX_CALIBRATION_TRIALS)
        horizon = _bounded_int("horizon", self.horizon, 2, MAX_CALIBRATION_HORIZON)
        cells = _bounded_int("cell_count", self.cell_count, 1, MAX_CALIBRATION_CHALLENGERS)
        parent_alpha = _finite_probability("parent_alpha", self.parent_alpha)
        cell_alpha = _finite_probability("cell_alpha", self.cell_alpha)
        eta = validate_eta(self.eta)
        try:
            initial_wealth = float(self.initial_wealth)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("initial_wealth must be finite and positive") from exc
        if not math.isfinite(initial_wealth) or initial_wealth <= 0.0:
            raise ValueError("initial_wealth must be finite and positive")
        if isinstance(self.seed, (bool, np.bool_)) or not isinstance(
            self.seed, (int, np.integer)
        ) or int(self.seed) < 0:
            raise ValueError("seed must be a non-negative integer")
        if len(self.first_crossing_steps) != trials or len(self.first_crossing_cell_indices) != trials:
            raise ValueError("selection crossing paths must have one entry per trial")
        crossing_count = _bounded_int(
            "family_crossing_count", self.family_crossing_count, 0, trials
        )
        if not math.isclose(
            self.cell_alpha * cells,
            parent_alpha,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ):
            raise ValueError("cell alpha does not exhaust the parent selection budget")
        for step, cell in zip(self.first_crossing_steps, self.first_crossing_cell_indices):
            if isinstance(step, (bool, np.bool_)) or not isinstance(step, (int, np.integer)):
                raise ValueError("first crossing steps must be integers")
            if isinstance(cell, (bool, np.bool_)) or not isinstance(cell, (int, np.integer)):
                raise ValueError("first crossing cell indices must be integers")
            normalized_step = int(step)
            normalized_cell = int(cell)
            if normalized_step == horizon:
                if normalized_cell != -1:
                    raise ValueError("no-crossing paths must use cell index -1")
            elif not 0 <= normalized_step < horizon:
                raise ValueError("first crossing step is outside the horizon")
            elif not 0 <= normalized_cell < cells:
                raise ValueError("first crossing cell index is out of range")
        expected_rate = crossing_count / trials
        if not math.isclose(
            self.family_crossing_rate,
            expected_rate,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("family crossing rate is inconsistent with its count")
        if not 0.0 <= self.family_crossing_ci_low <= self.family_crossing_ci_high <= 1.0:
            raise ValueError("family crossing interval is invalid")
        if sum(step < horizon for step in self.first_crossing_steps) != crossing_count:
            raise ValueError("family crossing count is inconsistent with crossing paths")
        object.__setattr__(self, "trials", trials)
        object.__setattr__(self, "horizon", horizon)
        object.__setattr__(self, "cell_count", cells)
        object.__setattr__(self, "parent_alpha", parent_alpha)
        object.__setattr__(self, "cell_alpha", cell_alpha)
        object.__setattr__(self, "eta", eta)
        object.__setattr__(self, "initial_wealth", initial_wealth)
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "family_crossing_count", crossing_count)

    @property
    def config_identity(self) -> str:
        payload = {
            "score_factory_identity": self.score_factory_identity,
            "selection_budget_identity": self.selection_budget_identity,
            "trials": self.trials,
            "horizon": self.horizon,
            "cell_count": self.cell_count,
            "parent_alpha": self.parent_alpha,
            "cell_alpha": self.cell_alpha,
            "eta": self.eta,
            "initial_wealth": self.initial_wealth,
            "seed": self.seed,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "selection_null_calibration",
            "score_factory_identity": self.score_factory_identity,
            "selection_budget_identity": self.selection_budget_identity,
            "trials": self.trials,
            "horizon": self.horizon,
            "cell_count": self.cell_count,
            "parent_alpha": self.parent_alpha,
            "cell_alpha": self.cell_alpha,
            "eta": self.eta,
            "initial_wealth": self.initial_wealth,
            "seed": self.seed,
            "family_crossing_count": self.family_crossing_count,
            "family_crossing_rate": self.family_crossing_rate,
            "family_crossing_ci_95": [
                self.family_crossing_ci_low,
                self.family_crossing_ci_high,
            ],
            "first_crossing_steps": list(self.first_crossing_steps),
            "first_crossing_cell_indices": list(self.first_crossing_cell_indices),
            "config_identity": self.config_identity,
        }


def calibrate_selection_null(
    score_factory: SelectionScoreFactory,
    *,
    config: SelectionCalibrationConfig,
) -> SelectionNullCalibrationResult:
    """Simulate optional stopping across a pre-registered selection family.

    The factory returns a ``(horizon, budget.total_slots)`` matrix.  Every
    cell receives the equal parent-alpha share declared by ``SelectionBudget``
    and the family stops at its first cell crossing.  This is a finite null
    simulation of the complete selection boundary; it does not establish the
    conditional-mean assumption for a market score stream.
    """

    if not callable(score_factory):
        raise ValueError("score_factory must be callable")
    settings = config
    identity = _score_factory_identity(score_factory)
    budget = settings.budget
    first_steps: list[int] = []
    first_cells: list[int] = []
    rng = np.random.default_rng(settings.seed)
    for _ in range(settings.trials):
        try:
            raw_scores = np.asarray(
                score_factory(rng, settings.horizon, budget.total_slots),
                dtype=float,
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "selection score_factory must return finite bounded scores"
            ) from exc
        if raw_scores.shape != (settings.horizon, budget.total_slots):
            raise ValueError(
                "selection score_factory must return scores shaped "
                f"({settings.horizon}, {budget.total_slots})"
            )
        if not np.all(np.isfinite(raw_scores)) or np.any(raw_scores < -1.0) or np.any(
            raw_scores > 1.0
        ):
            raise ValueError("selection score_factory returned scores outside [-1, 1]")
        processes = [
            EProcess(
                alpha=budget.allocated_alpha,
                eta=settings.eta,
                initial_wealth=settings.initial_wealth,
            )
            for _ in range(budget.total_slots)
        ]
        first_step = settings.horizon
        first_cell = -1
        for step in range(settings.horizon):
            for cell, process in enumerate(processes):
                update = process.update(
                    float(raw_scores[step, cell]), eta=settings.eta
                )
                if update.first_crossing:
                    first_step = step
                    first_cell = cell
                    break
            if first_cell >= 0:
                break
        first_steps.append(first_step)
        first_cells.append(first_cell)

    crossing_count = sum(step < settings.horizon for step in first_steps)
    rate = crossing_count / settings.trials
    low, high = _wilson_interval(crossing_count, settings.trials)
    return SelectionNullCalibrationResult(
        score_factory_identity=identity,
        selection_budget_identity=budget.identity,
        trials=settings.trials,
        horizon=settings.horizon,
        cell_count=budget.total_slots,
        parent_alpha=budget.global_alpha,
        cell_alpha=budget.allocated_alpha,
        eta=settings.eta,
        initial_wealth=settings.initial_wealth,
        seed=settings.seed,
        family_crossing_count=crossing_count,
        family_crossing_rate=rate,
        family_crossing_ci_low=low,
        family_crossing_ci_high=high,
        first_crossing_steps=tuple(first_steps),
        first_crossing_cell_indices=tuple(first_cells),
    )


def calibrate_eprocess_null(
    score_factory: EProcessScoreFactory,
    *,
    config: EProcessCalibrationConfig | None = None,
) -> EProcessNullCalibrationResult:
    """Simulate optional stopping for a declared bounded score null.

    The factory must return one-dimensional scores in ``[-1, 1]``.  The
    harness stops each path at its first threshold crossing and records the
    finite-horizon crossing rate.  It checks boundedness and predictably uses
    the predeclared constant ``eta``; it cannot prove that the supplied score
    factory satisfies the conditional-mean null required by the e-process
    theorem.
    """

    settings = config or EProcessCalibrationConfig()
    identity = _score_factory_identity(score_factory)
    first_crossing_steps: list[int] = []
    rng = np.random.default_rng(settings.seed)
    for _ in range(settings.trials):
        try:
            values = np.asarray(score_factory(rng, settings.horizon), dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("score_factory must return finite bounded scores") from exc
        if values.ndim != 1 or values.size != settings.horizon:
            raise ValueError(
                "score_factory must return a one-dimensional array shaped "
                f"({settings.horizon},)"
            )
        if not np.all(np.isfinite(values)) or np.any(values < -1.0) or np.any(values > 1.0):
            raise ValueError("score_factory returned scores outside [-1, 1]")
        process = EProcess(
            alpha=settings.alpha,
            eta=settings.eta,
            initial_wealth=settings.initial_wealth,
        )
        first = settings.horizon
        for step, value in enumerate(values):
            update = process.update(float(value), eta=settings.eta)
            if update.first_crossing:
                first = step
                break
        first_crossing_steps.append(first)
    crossing_count = sum(step < settings.horizon for step in first_crossing_steps)
    rate = crossing_count / settings.trials
    low, high = _wilson_interval(crossing_count, settings.trials)
    return EProcessNullCalibrationResult(
        score_factory_identity=identity,
        trials=settings.trials,
        horizon=settings.horizon,
        alpha=settings.alpha,
        eta=settings.eta,
        initial_wealth=settings.initial_wealth,
        seed=settings.seed,
        threshold_crossing_count=crossing_count,
        threshold_crossing_rate=rate,
        threshold_crossing_ci_low=low,
        threshold_crossing_ci_high=high,
        first_crossing_steps=tuple(first_crossing_steps),
    )


def calibrate_promotion_null(
    score_factory: PromotionScoreFactory,
    *,
    config: PromotionCalibrationConfig | None = None,
) -> PromotionNullCalibrationResult:
    """Simulate optional stopping through the complete multi-challenger gate.

    ``score_factory`` must return bounded scores shaped ``(horizon,)`` for
    one challenger or ``(horizon, challengers)`` for several challengers when
    ``epochs=1``. For repeated epochs it must return
    ``(epochs, horizon, challengers)``. At every step the gate observes
    candidates in registration order and stops the path immediately on the
    first promotion; a path with no promotion resets the gate between declared
    epochs. The factory is responsible for generating a declared null stream;
    this harness checks shape, finiteness, and score bounds but cannot
    establish the conditional-mean assumptions required for an anytime-valid
    market claim.
    """

    settings = config or PromotionCalibrationConfig()
    identity = _score_factory_identity(score_factory)
    allocations = tuple(
        geometric_alpha_allocation(settings.alpha, index + 1, epoch=0)
        for index in range(settings.challengers)
    )
    allocation_schedule = tuple(
        tuple(
            geometric_alpha_allocation(settings.alpha, index + 1, epoch=epoch)
            for index in range(settings.challengers)
        )
        for epoch in range(settings.epochs)
    )
    first_promotion_epochs: list[int] = []
    first_promotion_steps: list[int] = []
    first_promoted_indices: list[int] = []
    rng = np.random.default_rng(settings.seed)
    for _ in range(settings.trials):
        try:
            raw_scores = np.asarray(
                score_factory(rng, settings.horizon, settings.challengers),
                dtype=float,
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "score_factory must return finite bounded promotion scores"
            ) from exc
        scores: np.ndarray[Any, Any]
        if settings.epochs == 1 and raw_scores.ndim == 1:
            if settings.challengers != 1:
                raise ValueError(
                    "multi-challenger score_factory output must be two-dimensional"
                )
            scores = raw_scores.reshape(1, -1, 1)
        elif settings.epochs == 1 and raw_scores.ndim == 2:
            scores = raw_scores.reshape(1, raw_scores.shape[0], raw_scores.shape[1])
        elif settings.epochs > 1 and raw_scores.ndim == 3:
            scores = raw_scores
        else:
            expected_dimension = (
                "three-dimensional"
                if settings.epochs > 1
                else "one- or two-dimensional"
            )
            raise ValueError(f"score_factory must return a {expected_dimension} score array")
        if scores.shape != (
            settings.epochs,
            settings.horizon,
            settings.challengers,
        ):
            raise ValueError(
                "score_factory must return scores shaped "
                f"({settings.epochs}, {settings.horizon}, {settings.challengers})"
            )
        if not np.all(np.isfinite(scores)) or np.any(scores < -1.0) or np.any(
            scores > 1.0
        ):
            raise ValueError("score_factory returned scores outside [-1, 1]")

        gate = PromotionGate(
            "incumbent",
            alpha=settings.alpha,
            eta=settings.eta,
            initial_wealth=settings.initial_wealth,
        )
        challenger_ids = tuple(
            f"challenger-{index}" for index in range(settings.challengers)
        )
        for index, challenger_id in enumerate(challenger_ids):
            registered = gate.register_challenger(challenger_id)
            if not math.isclose(
                registered.alpha,
                allocations[index],
                rel_tol=0.0,
                abs_tol=1.0e-15,
            ):
                raise ValueError("promotion gate alpha allocation is inconsistent")
        # The challenger family is fixed before any simulated score is
        # observed.  This is the same registration boundary required by the
        # certified causal adapter.
        gate.seal_registration()

        first_epoch = settings.epochs
        first_step = settings.horizon
        promoted_index = -1
        for epoch in range(settings.epochs):
            for index, challenger_id in enumerate(challenger_ids):
                state = gate.challenger_state(challenger_id)
                if not math.isclose(
                    state.alpha,
                    allocation_schedule[epoch][index],
                    rel_tol=0.0,
                    abs_tol=1.0e-15,
                ):
                    raise ValueError("promotion gate epoch allocation is inconsistent")
            for step in range(settings.horizon):
                for index, challenger_id in enumerate(challenger_ids):
                    decision = gate.observe_score(
                        challenger_id,
                        float(scores[epoch, step, index]),
                    )
                    if decision.promoted:
                        first_epoch = epoch
                        first_step = step
                        promoted_index = index
                        break
                if first_epoch != settings.epochs:
                    break
            if first_epoch != settings.epochs:
                break
            if epoch + 1 < settings.epochs:
                gate.reset_epoch(reason=f"calibration epoch {epoch + 1}")
        first_promotion_epochs.append(first_epoch)
        first_promotion_steps.append(first_step)
        first_promoted_indices.append(promoted_index)

    crossing_count = sum(
        step < settings.horizon for step in first_promotion_steps
    )
    rate = crossing_count / settings.trials
    low, high = _wilson_interval(crossing_count, settings.trials)
    return PromotionNullCalibrationResult(
        score_factory_identity=identity,
        trials=settings.trials,
        horizon=settings.horizon,
        challenger_count=settings.challengers,
        epochs=settings.epochs,
        alpha=settings.alpha,
        eta=settings.eta,
        initial_wealth=settings.initial_wealth,
        seed=settings.seed,
        challenger_alpha_allocations=allocations,
        challenger_alpha_schedule=allocation_schedule,
        threshold_crossing_count=crossing_count,
        threshold_crossing_rate=rate,
        threshold_crossing_ci_low=low,
        threshold_crossing_ci_high=high,
        first_promotion_epochs=tuple(first_promotion_epochs),
        first_promotion_steps=tuple(first_promotion_steps),
        first_promoted_challenger_indices=tuple(first_promoted_indices),
    )


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
    observation_identity = _factory_identity(observation_factory)
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
        observation_identity=observation_identity,
    )


def calibrate_threshold(
    detector_factory: Callable[[float], Any],
    calibration_observation_factory: ObservationFactory,
    evaluation_observation_factory: ObservationFactory,
    *,
    detector_family_identity: str,
    candidate_thresholds: Sequence[float],
    calibration_config: CalibrationConfig,
    evaluation_config: CalibrationConfig,
    max_false_alarm_rate: float,
) -> ThresholdCalibrationResult:
    """Select a threshold on one null split and evaluate it on another.

    The candidate list and selection rule are fixed before either split is
    consumed.  The two observation factories are required to be distinct so
    an evaluation source or time range cannot be represented only by a new
    random seed.  The smallest candidate whose calibration Wilson upper bound
    is within ``max_false_alarm_rate`` is selected.  Only the independent
    evaluation result can produce an acceleration certificate.
    """

    if not callable(detector_factory):
        raise ValueError("detector_factory must be callable")
    if not callable(calibration_observation_factory):
        raise ValueError("calibration_observation_factory must be callable")
    if not callable(evaluation_observation_factory):
        raise ValueError("evaluation_observation_factory must be callable")
    if calibration_observation_factory is evaluation_observation_factory:
        raise ValueError(
            "calibration and evaluation require distinct observation factories"
        )
    if not isinstance(detector_family_identity, str) or not detector_family_identity.strip():
        raise ValueError("detector_family_identity must be a non-empty string")
    if not isinstance(calibration_config, CalibrationConfig):
        raise TypeError("calibration_config must be CalibrationConfig")
    if not isinstance(evaluation_config, CalibrationConfig):
        raise TypeError("evaluation_config must be CalibrationConfig")
    if (
        calibration_config.horizon != evaluation_config.horizon
        or calibration_config.n_features != evaluation_config.n_features
        or not math.isclose(
            calibration_config.alpha,
            evaluation_config.alpha,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    ):
        raise ValueError(
            "calibration and evaluation configs must share horizon, feature count, and alpha"
        )
    if calibration_config.seed == evaluation_config.seed:
        raise ValueError("calibration and evaluation configs require distinct seeds")
    max_rate = _finite_probability("max_false_alarm_rate", max_false_alarm_rate)
    try:
        raw_thresholds = tuple(candidate_thresholds)
    except TypeError as exc:
        raise ValueError("candidate_thresholds must be a sequence") from exc
    if isinstance(candidate_thresholds, (str, bytes, bytearray)):
        raise ValueError("candidate_thresholds must be a numeric sequence")
    thresholds = tuple(float(value) for value in raw_thresholds)
    if not thresholds or len(thresholds) > MAX_CALIBRATION_THRESHOLDS:
        raise ValueError(
            "candidate_thresholds must contain between one and "
            f"{MAX_CALIBRATION_THRESHOLDS} values"
        )
    if not all(math.isfinite(value) and value > 0.0 for value in thresholds):
        raise ValueError("candidate_thresholds must be finite and positive")
    if tuple(sorted(thresholds)) != thresholds:
        raise ValueError("candidate_thresholds must be sorted")
    if len(set(thresholds)) != len(thresholds):
        raise ValueError("candidate_thresholds must be unique")

    detector_factory_identity = _factory_identity(detector_factory)
    calibration_split_factory = _ThresholdSplitObservationFactory(
        calibration_observation_factory, "calibration"
    )
    evaluation_split_factory = _ThresholdSplitObservationFactory(
        evaluation_observation_factory, "evaluation"
    )
    calibration_observation_identity = calibration_split_factory.identity
    evaluation_observation_identity = evaluation_split_factory.identity
    calibration_results: list[NullCalibrationResult] = []
    for threshold in thresholds:
        def candidate_factory(value: float = threshold) -> Any:
            return detector_factory(value)

        result = calibrate_null(
            candidate_factory,
            calibration_split_factory,
            config=calibration_config,
        )
        calibration_results.append(result)

    passing = tuple(
        index
        for index, result in enumerate(calibration_results)
        if result.false_alarm_ci_high <= max_rate
    )
    if not passing:
        raise ValueError(
            "no candidate threshold satisfies the calibration false-alarm budget"
        )
    selected_index = passing[0]
    selected_threshold = thresholds[selected_index]

    def selected_factory(value: float = selected_threshold) -> Any:
        return detector_factory(value)

    evaluation_result = calibrate_null(
        selected_factory,
        evaluation_split_factory,
        config=evaluation_config,
    )
    if any(
        result.observation_identity != calibration_observation_identity
        for result in calibration_results
    ):
        raise ValueError("calibration factory identity changed across the split")
    if evaluation_result.observation_identity != evaluation_observation_identity:
        raise ValueError("evaluation factory identity changed across the split")

    return ThresholdCalibrationResult(
        detector_family_identity=detector_family_identity,
        detector_factory_identity=detector_factory_identity,
        calibration_observation_identity=calibration_observation_identity,
        evaluation_observation_identity=evaluation_observation_identity,
        candidate_thresholds=thresholds,
        selected_threshold=selected_threshold,
        max_false_alarm_rate=max_rate,
        calibration_results=tuple(calibration_results),
        evaluation_result=evaluation_result,
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
    observation_identity = _factory_identity(observation_factory)
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
        observation_identity=observation_identity,
    )


__all__ = [
    "CalibrationCertificate",
    "CalibrationConfig",
    "EProcessCalibrationConfig",
    "EProcessNullCalibrationResult",
    "MAX_CALIBRATION_CHALLENGERS",
    "MAX_CALIBRATION_EPOCHS",
    "MAX_CALIBRATION_FEATURES",
    "MAX_CALIBRATION_HORIZON",
    "MAX_CALIBRATION_SOURCE_ROWS",
    "MAX_CALIBRATION_THRESHOLDS",
    "MAX_CALIBRATION_TRIALS",
    "NullCalibrationResult",
    "PromotionCalibrationConfig",
    "PromotionNullCalibrationResult",
    "PromotionScoreFactory",
    "SelectionCalibrationConfig",
    "SelectionNullCalibrationResult",
    "SelectionScoreFactory",
    "StationaryBlockBootstrap",
    "ShiftCalibrationResult",
    "ThresholdCalibrationResult",
    "calibrate_null",
    "calibrate_eprocess_null",
    "calibrate_promotion_null",
    "calibrate_selection_null",
    "calibrate_threshold",
    "calibrate_shift",
]
