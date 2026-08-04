"""Exploratory persistent-Laplacian spectral CUSUM controller.

The controller is the research seam for the proposed topology-gated memory
policy.  It consumes a rolling point cloud, computes a bounded exact finite
persistent-Laplacian artifact, and compares the current Betti/spectral state
only with earlier valid states.  The resulting score and forgetting factor are
diagnostic suggestions: accelerated forgetting still requires an independent
calibration certificate in the causal RLS adapter.

This module intentionally uses a simple robust marginal standardization and a
non-negative innovation norm.  It is a reproducible PL-CUSUM-shaped baseline,
not a theorem that its score has a calibrated false-alarm level.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any

from .persistent import (
    MAX_PERSISTENT_VERTICES,
    PersistentLaplacianResult,
    PersistentStatus,
)

PERSISTENT_CUSUM_SCHEMA = "topology_gate.persistent_cusum"
PERSISTENT_CUSUM_VERSION = 1
MAX_PERSISTENT_CUSUM_HISTORY = 4_096
MAX_PERSISTENT_CUSUM_BETTI_DIMENSIONS = 8
_MAD_NORMALIZATION = 1.4826


class PersistentCUSUMError(ValueError):
    """Base error for persistent spectral CUSUM validation failures."""


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise PersistentCUSUMError(f"{name} must be finite")
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PersistentCUSUMError(f"{name} must be finite") from exc
    if not math.isfinite(converted):
        raise PersistentCUSUMError(f"{name} must be finite")
    return converted


def _integer(value: Any, name: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise PersistentCUSUMError(f"{name} must be an integer")
    converted = int(value)
    if not minimum <= converted <= maximum:
        raise PersistentCUSUMError(
            f"{name} must be in [{minimum}, {maximum}]"
        )
    return converted


def _digest(value: Any, name: str) -> str:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PersistentCUSUMError(f"{name} is not JSON-safe") from exc
    return hashlib.sha256(encoded).hexdigest()


def _text_digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in value)
    ):
        raise PersistentCUSUMError(f"{name} must be a 64-character hexadecimal digest")
    return value.lower()


def _median(values: Sequence[float]) -> float:
    if not values:
        raise PersistentCUSUMError("cannot take the median of an empty sequence")
    ordered = sorted(float(value) for value in values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _row(value: Any, name: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes, bytearray)):
        raise PersistentCUSUMError(f"{name} must be a finite one-dimensional row")
    try:
        values = tuple(_finite(item, f"{name} value") for item in value)
    except TypeError as exc:
        raise PersistentCUSUMError(
            f"{name} must be a finite one-dimensional row"
        ) from exc
    if not values:
        raise PersistentCUSUMError(f"{name} must not be empty")
    return values


def _backend_identity(backend: Callable[..., Any]) -> str:
    identity = getattr(backend, "identity", None)
    if callable(identity):
        identity = identity()
    if identity is None:
        identity = getattr(backend, "config_identity", None)
        if callable(identity):
            identity = identity()
    if not isinstance(identity, str) or not identity.strip():
        raise PersistentCUSUMError(
            "persistent CUSUM backend must expose a stable identity"
        )
    return identity


@dataclass(frozen=True, slots=True)
class PersistentCUSUMConfig:
    """Resource, feature, calibration, and forgetting policy for the controller."""

    cloud_window: int = 8
    min_points: int = 4
    backend_eigenvalues: int = 8
    positive_spectrum_width: int = 2
    betti_dimensions: tuple[int, ...] = (0, 1)
    calibration_window: int = 32
    calibration_min_periods: int = 8
    drift: float = 0.5
    threshold: float = 5.0
    decay: float = 1.0
    scale_floor: float = 1.0e-8
    z_clip: float = 12.0
    positive_eigenvalue_floor: float = 1.0e-10
    log_positive_spectrum: bool = True
    forgetting_lambda_min: float = 0.8
    forgetting_lambda_max: float = 0.99
    forgetting_sensitivity: float = 1.0
    max_stream_observations: int = 100_000
    max_history: int = 4_096

    def __post_init__(self) -> None:
        cloud_window = _integer(
            self.cloud_window,
            "cloud_window",
            minimum=2,
            maximum=MAX_PERSISTENT_VERTICES,
        )
        min_points = _integer(
            self.min_points,
            "min_points",
            minimum=2,
            maximum=cloud_window,
        )
        backend_eigenvalues = _integer(
            self.backend_eigenvalues,
            "backend_eigenvalues",
            minimum=1,
            maximum=64,
        )
        positive_width = _integer(
            self.positive_spectrum_width,
            "positive_spectrum_width",
            minimum=1,
            maximum=backend_eigenvalues,
        )
        raw_dimensions = self.betti_dimensions
        if isinstance(raw_dimensions, (str, bytes, bytearray)) or not isinstance(
            raw_dimensions, Sequence
        ):
            raise PersistentCUSUMError("betti_dimensions must be an integer sequence")
        dimensions = tuple(
            _integer(
                value,
                "betti dimension",
                minimum=0,
                maximum=MAX_PERSISTENT_CUSUM_BETTI_DIMENSIONS - 1,
            )
            for value in raw_dimensions
        )
        if not dimensions or len(set(dimensions)) != len(dimensions):
            raise PersistentCUSUMError(
                "betti_dimensions must contain unique dimensions"
            )
        if tuple(sorted(dimensions)) != dimensions:
            raise PersistentCUSUMError("betti_dimensions must be sorted")
        calibration_window = _integer(
            self.calibration_window,
            "calibration_window",
            minimum=1,
            maximum=MAX_PERSISTENT_CUSUM_HISTORY,
        )
        calibration_min_periods = _integer(
            self.calibration_min_periods,
            "calibration_min_periods",
            minimum=1,
            maximum=calibration_window,
        )
        drift = _finite(self.drift, "drift")
        if drift < 0.0:
            raise PersistentCUSUMError("drift must be non-negative")
        threshold = _finite(self.threshold, "threshold")
        if threshold <= 0.0:
            raise PersistentCUSUMError("threshold must be positive")
        decay = _finite(self.decay, "decay")
        if not 0.0 < decay <= 1.0:
            raise PersistentCUSUMError("decay must be in (0, 1]")
        scale_floor = _finite(self.scale_floor, "scale_floor")
        if scale_floor <= 0.0:
            raise PersistentCUSUMError("scale_floor must be positive")
        z_clip = _finite(self.z_clip, "z_clip")
        if z_clip <= 0.0:
            raise PersistentCUSUMError("z_clip must be positive")
        positive_floor = _finite(
            self.positive_eigenvalue_floor, "positive_eigenvalue_floor"
        )
        if positive_floor <= 0.0:
            raise PersistentCUSUMError(
                "positive_eigenvalue_floor must be positive"
            )
        if not isinstance(self.log_positive_spectrum, bool):
            raise PersistentCUSUMError("log_positive_spectrum must be boolean")
        lambda_min = _finite(self.forgetting_lambda_min, "forgetting_lambda_min")
        lambda_max = _finite(self.forgetting_lambda_max, "forgetting_lambda_max")
        sensitivity = _finite(self.forgetting_sensitivity, "forgetting_sensitivity")
        if not 0.0 < lambda_min <= lambda_max <= 1.0:
            raise PersistentCUSUMError(
                "forgetting bounds must satisfy 0 < min <= max <= 1"
            )
        if sensitivity < 0.0:
            raise PersistentCUSUMError("forgetting_sensitivity must be non-negative")
        max_observations = _integer(
            self.max_stream_observations,
            "max_stream_observations",
            minimum=cloud_window,
            maximum=1_000_000,
        )
        max_history = _integer(
            self.max_history,
            "max_history",
            minimum=calibration_window,
            maximum=MAX_PERSISTENT_CUSUM_HISTORY,
        )
        object.__setattr__(self, "cloud_window", cloud_window)
        object.__setattr__(self, "min_points", min_points)
        object.__setattr__(self, "backend_eigenvalues", backend_eigenvalues)
        object.__setattr__(self, "positive_spectrum_width", positive_width)
        object.__setattr__(self, "betti_dimensions", dimensions)
        object.__setattr__(self, "calibration_window", calibration_window)
        object.__setattr__(self, "calibration_min_periods", calibration_min_periods)
        object.__setattr__(self, "drift", drift)
        object.__setattr__(self, "threshold", threshold)
        object.__setattr__(self, "decay", decay)
        object.__setattr__(self, "scale_floor", scale_floor)
        object.__setattr__(self, "z_clip", z_clip)
        object.__setattr__(self, "positive_eigenvalue_floor", positive_floor)
        object.__setattr__(self, "forgetting_lambda_min", lambda_min)
        object.__setattr__(self, "forgetting_lambda_max", lambda_max)
        object.__setattr__(self, "forgetting_sensitivity", sensitivity)
        object.__setattr__(self, "max_stream_observations", max_observations)
        object.__setattr__(self, "max_history", max_history)

    @property
    def state_dimension(self) -> int:
        return len(self.betti_dimensions) + self.positive_spectrum_width

    def to_dict(self) -> dict[str, Any]:
        return {
            "cloud_window": self.cloud_window,
            "min_points": self.min_points,
            "backend_eigenvalues": self.backend_eigenvalues,
            "positive_spectrum_width": self.positive_spectrum_width,
            "betti_dimensions": list(self.betti_dimensions),
            "calibration_window": self.calibration_window,
            "calibration_min_periods": self.calibration_min_periods,
            "drift": self.drift,
            "threshold": self.threshold,
            "decay": self.decay,
            "scale_floor": self.scale_floor,
            "z_clip": self.z_clip,
            "positive_eigenvalue_floor": self.positive_eigenvalue_floor,
            "log_positive_spectrum": self.log_positive_spectrum,
            "forgetting_lambda_min": self.forgetting_lambda_min,
            "forgetting_lambda_max": self.forgetting_lambda_max,
            "forgetting_sensitivity": self.forgetting_sensitivity,
            "max_stream_observations": self.max_stream_observations,
            "max_history": self.max_history,
        }

    @property
    def identity(self) -> str:
        return _digest(self.to_dict(), "persistent CUSUM configuration")


@dataclass(frozen=True, slots=True)
class PersistentCUSUMObservation:
    """One causal persistent-spectrum control observation."""

    step: int
    ready: bool
    score: float
    innovation: float
    alarm: bool
    forgetting_factor: float
    method: str
    state: tuple[float, ...]
    standardized_state: tuple[float, ...]
    reference_location: tuple[float, ...]
    reference_scale: tuple[float, ...]
    betti_numbers: tuple[int, ...]
    positive_eigenvalues: tuple[float, ...]
    backend_evidence_digest: str | None
    reason: str

    @property
    def raw_features(self) -> tuple[float, ...]:
        """Compatibility alias for detector telemetry consumers."""

        return self.state

    @property
    def whitened_features(self) -> tuple[float, ...]:
        """Compatibility alias for standardized detector telemetry."""

        return self.standardized_state

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "ready": self.ready,
            "score": self.score,
            "innovation": self.innovation,
            "alarm": self.alarm,
            "forgetting_factor": self.forgetting_factor,
            "method": self.method,
            "state": list(self.state),
            "standardized_state": list(self.standardized_state),
            "reference_location": list(self.reference_location),
            "reference_scale": list(self.reference_scale),
            "betti_numbers": list(self.betti_numbers),
            "positive_eigenvalues": list(self.positive_eigenvalues),
            "backend_evidence_digest": self.backend_evidence_digest,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class _HistoryEntry:
    state: tuple[float, ...]
    evidence_digest: str


class PersistentLaplacianCUSUM:
    """Stateful rolling persistent-spectrum CUSUM controller.

    The callable backend must accept ``(point_cloud, n_eigenvalues)`` and
    return a :class:`PersistentLaplacianResult`.  Every reference statistic is
    computed from states strictly preceding the current point cloud.
    """

    method = "persistent_laplacian_positive_spectrum_cusum_v1"

    def __init__(
        self,
        backend: Callable[..., Any],
        config: PersistentCUSUMConfig | None = None,
    ) -> None:
        if not callable(backend):
            raise TypeError("backend must be callable")
        self.backend = backend
        self.config = config or PersistentCUSUMConfig()
        self.backend_identity = _backend_identity(backend)
        backend_limit = getattr(backend, "max_vertices", None)
        if backend_limit is not None and self.config.cloud_window > int(backend_limit):
            raise PersistentCUSUMError(
                "cloud_window exceeds the persistent backend vertex limit"
            )
        backend_width = getattr(backend, "n_eigenvalues", None)
        if backend_width is not None and int(backend_width) != self.config.backend_eigenvalues:
            raise PersistentCUSUMError(
                "backend_eigenvalues must match the configured backend width"
            )
        self._config_identity = _digest(
            {"config": self.config.to_dict(), "backend": self.backend_identity},
            "persistent CUSUM identity",
        )
        self._step = 0
        self._score = 0.0
        self._observations: list[tuple[float, ...]] = []
        self._history: list[_HistoryEntry] = []
        self._last_result: PersistentCUSUMObservation | None = None

    @property
    def config_identity(self) -> str:
        return self._config_identity

    @property
    def last_result(self) -> PersistentCUSUMObservation | None:
        return self._last_result

    @property
    def history_length(self) -> int:
        return len(self._history)

    def _forgetting_factor(self, score: float, *, ready: bool) -> float:
        if not ready:
            return self.config.forgetting_lambda_max
        span = self.config.forgetting_lambda_max - self.config.forgetting_lambda_min
        return self.config.forgetting_lambda_min + span * math.exp(
            -self.config.forgetting_sensitivity * score
        )

    def _not_ready(
        self,
        *,
        reason: str,
        state: tuple[float, ...] = (),
        betti_numbers: tuple[int, ...] = (),
        positive_eigenvalues: tuple[float, ...] = (),
        evidence_digest: str | None = None,
    ) -> PersistentCUSUMObservation:
        return PersistentCUSUMObservation(
            step=self._step,
            ready=False,
            score=0.0,
            innovation=0.0,
            alarm=False,
            forgetting_factor=self.config.forgetting_lambda_max,
            method=self.method,
            state=state,
            standardized_state=(),
            reference_location=(),
            reference_scale=(),
            betti_numbers=betti_numbers,
            positive_eigenvalues=positive_eigenvalues,
            backend_evidence_digest=evidence_digest,
            reason=reason,
        )

    def _extract_state(
        self, result: PersistentLaplacianResult
    ) -> tuple[tuple[float, ...], tuple[int, ...], tuple[float, ...], str]:
        if not isinstance(result, PersistentLaplacianResult):
            raise PersistentCUSUMError(
                "persistent CUSUM backend must return PersistentLaplacianResult"
            )
        if result.status is not PersistentStatus.VALID:
            raise PersistentCUSUMError(
                "persistent CUSUM requires a valid topology artifact"
            )
        spectrum = result.spectrum
        positive = []
        for value in spectrum.eigenvalues:
            converted = _finite(value, "persistent eigenvalue")
            if converted > self.config.positive_eigenvalue_floor:
                positive.append(converted)
        if len(positive) < self.config.positive_spectrum_width:
            raise PersistentCUSUMError(
                "persistent artifact does not contain the configured positive "
                "spectrum width"
            )
        positive = positive[: self.config.positive_spectrum_width]
        scale = _finite(spectrum.scale_s, "persistent scale_s")
        betti: list[int] = []
        for dimension in self.config.betti_dimensions:
            count = 0
            for interval in result.intervals:
                if interval.homology_dimension != dimension:
                    continue
                if interval.birth > scale:
                    continue
                if interval.death is None or scale < interval.death:
                    count += 1
            betti.append(count)
        transformed_positive = tuple(
            math.log1p(value) if self.config.log_positive_spectrum else value
            for value in positive
        )
        state = tuple(float(value) for value in (*betti, *transformed_positive))
        if len(state) != self.config.state_dimension or not all(
            math.isfinite(value) for value in state
        ):
            raise PersistentCUSUMError("persistent state is not finite")
        evidence_digest = _text_digest(
            result.evidence_digest, "persistent evidence digest"
        )
        return state, tuple(betti), tuple(positive), evidence_digest

    def _append_history(self, entry: _HistoryEntry) -> None:
        self._history.append(entry)
        if len(self._history) > self.config.max_history:
            del self._history[: len(self._history) - self.config.max_history]

    def _standardize(
        self, state: tuple[float, ...], reference: Sequence[_HistoryEntry]
    ) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...], float]:
        columns = list(zip(*(entry.state for entry in reference)))
        location = tuple(_median(column) for column in columns)
        scales = []
        for column, center in zip(columns, location):
            deviations = [abs(value - center) for value in column]
            scales.append(max(_MAD_NORMALIZATION * _median(deviations), self.config.scale_floor))
        standardized = tuple(
            max(
                -self.config.z_clip,
                min(self.config.z_clip, (value - center) / scale),
            )
            for value, center, scale in zip(state, location, scales)
        )
        innovation = math.sqrt(
            sum(value * value for value in standardized) / len(standardized)
        )
        if not math.isfinite(innovation):
            raise PersistentCUSUMError("persistent CUSUM innovation is not finite")
        return standardized, location, tuple(scales), innovation

    def observe(self, observation: Any) -> PersistentCUSUMObservation:
        """Consume one market-state row and return its causal control signal."""

        if self._step >= self.config.max_stream_observations:
            raise PersistentCUSUMError("persistent CUSUM stream exceeds its resource limit")
        row = _row(observation, "observation")
        before = self.stream_state_dict()
        try:
            self._step += 1
            self._observations.append(row)
            if len(self._observations) > self.config.cloud_window:
                del self._observations[: len(self._observations) - self.config.cloud_window]
            if len(self._observations) < self.config.min_points:
                result = self._not_ready(reason="insufficient_point_cloud")
                self._last_result = result
                return result

            artifact = self.backend(
                tuple(self._observations), self.config.backend_eigenvalues
            )
            state, betti, positive, evidence_digest = self._extract_state(artifact)
            reference = self._history[-self.config.calibration_window :]
            if len(reference) < self.config.calibration_min_periods:
                self._append_history(_HistoryEntry(state, evidence_digest))
                result = self._not_ready(
                    reason="calibration_warmup",
                    state=state,
                    betti_numbers=betti,
                    positive_eigenvalues=positive,
                    evidence_digest=evidence_digest,
                )
                self._last_result = result
                return result

            standardized, location, scales, innovation = self._standardize(
                state, reference
            )
            self._score = max(
                0.0,
                self.config.decay * self._score + innovation - self.config.drift,
            )
            self._append_history(_HistoryEntry(state, evidence_digest))
            ready = True
            result = PersistentCUSUMObservation(
                step=self._step,
                ready=ready,
                score=self._score,
                innovation=innovation,
                alarm=self._score >= self.config.threshold,
                forgetting_factor=self._forgetting_factor(self._score, ready=ready),
                method=self.method,
                state=state,
                standardized_state=standardized,
                reference_location=location,
                reference_scale=scales,
                betti_numbers=betti,
                positive_eigenvalues=positive,
                backend_evidence_digest=evidence_digest,
                reason="ok",
            )
            self._last_result = result
            return result
        except Exception:
            self.load_stream_state_dict(before)
            raise

    def reset_stream(self) -> None:
        """Clear all rolling observations and causal reference state."""

        self._step = 0
        self._score = 0.0
        self._observations = []
        self._history = []
        self._last_result = None

    def stream_state_dict(self) -> dict[str, Any]:
        """Return strict JSON-safe state needed for exact continuation."""

        dimension = len(self._observations[0]) if self._observations else None
        return {
            "schema": PERSISTENT_CUSUM_SCHEMA,
            "version": PERSISTENT_CUSUM_VERSION,
            "config_identity": self.config_identity,
            "backend_identity": self.backend_identity,
            "step": self._step,
            "score": self._score,
            "observation_dimension": dimension,
            "state_dimension": self.config.state_dimension,
            "observations": [list(row) for row in self._observations],
            "history": [
                {
                    "state": list(entry.state),
                    "evidence_digest": entry.evidence_digest,
                }
                for entry in self._history
            ],
        }

    def validate_stream_state_dict(self, state: Mapping[str, Any]) -> dict[str, Any]:
        """Validate and normalize state without changing this controller."""

        if not isinstance(state, Mapping):
            raise PersistentCUSUMError("persistent CUSUM state must be a mapping")
        expected = {
            "schema",
            "version",
            "config_identity",
            "backend_identity",
            "step",
            "score",
            "observation_dimension",
            "state_dimension",
            "observations",
            "history",
        }
        if set(state) != expected:
            raise PersistentCUSUMError("persistent CUSUM state fields are invalid")
        if (
            state.get("schema") != PERSISTENT_CUSUM_SCHEMA
            or state.get("version") != PERSISTENT_CUSUM_VERSION
            or state.get("config_identity") != self.config_identity
            or state.get("backend_identity") != self.backend_identity
        ):
            raise PersistentCUSUMError("persistent CUSUM state identity mismatch")
        step = _integer(
            state.get("step"),
            "state.step",
            minimum=0,
            maximum=self.config.max_stream_observations,
        )
        score = _finite(state.get("score"), "state.score")
        if score < 0.0:
            raise PersistentCUSUMError("state.score must be non-negative")
        if state.get("state_dimension") != self.config.state_dimension:
            raise PersistentCUSUMError("state dimension does not match configuration")
        raw_observations = state.get("observations")
        if isinstance(raw_observations, (str, bytes, bytearray)) or not isinstance(
            raw_observations, Sequence
        ):
            raise PersistentCUSUMError("state.observations must be a sequence")
        if len(raw_observations) > self.config.cloud_window:
            raise PersistentCUSUMError("state observations exceed the cloud window")
        observations = tuple(
            _row(value, "state observation") for value in raw_observations
        )
        dimension = len(observations[0]) if observations else None
        if any(len(value) != dimension for value in observations):
            raise PersistentCUSUMError("state observations have inconsistent dimensions")
        if state.get("observation_dimension") != dimension:
            raise PersistentCUSUMError("state observation dimension is invalid")
        if step < len(observations):
            raise PersistentCUSUMError("state step precedes retained observations")
        raw_history = state.get("history")
        if isinstance(raw_history, (str, bytes, bytearray)) or not isinstance(
            raw_history, Sequence
        ):
            raise PersistentCUSUMError("state.history must be a sequence")
        if len(raw_history) > self.config.max_history:
            raise PersistentCUSUMError("state history exceeds its resource limit")
        if len(raw_history) > step:
            raise PersistentCUSUMError("state history exceeds the retained step count")
        history: list[dict[str, Any]] = []
        for item in raw_history:
            if not isinstance(item, Mapping) or set(item) != {
                "state",
                "evidence_digest",
            }:
                raise PersistentCUSUMError("state history fields are invalid")
            state_row = _row(item.get("state"), "state history row")
            if len(state_row) != self.config.state_dimension:
                raise PersistentCUSUMError("state history dimension is invalid")
            history.append(
                {
                    "state": list(state_row),
                    "evidence_digest": _text_digest(
                        item.get("evidence_digest"), "state history digest"
                    ),
                }
            )
        return {
            "schema": PERSISTENT_CUSUM_SCHEMA,
            "version": PERSISTENT_CUSUM_VERSION,
            "config_identity": self.config_identity,
            "backend_identity": self.backend_identity,
            "step": step,
            "score": score,
            "observation_dimension": dimension,
            "state_dimension": self.config.state_dimension,
            "observations": [list(value) for value in observations],
            "history": history,
        }

    def load_stream_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore a validated state atomically."""

        candidate = self.validate_stream_state_dict(state)
        self._step = candidate["step"]
        self._score = candidate["score"]
        self._observations = [tuple(value) for value in candidate["observations"]]
        self._history = [
            _HistoryEntry(tuple(item["state"]), item["evidence_digest"])
            for item in candidate["history"]
        ]
        self._last_result = None


__all__ = [
    "MAX_PERSISTENT_CUSUM_BETTI_DIMENSIONS",
    "MAX_PERSISTENT_CUSUM_HISTORY",
    "PERSISTENT_CUSUM_SCHEMA",
    "PERSISTENT_CUSUM_VERSION",
    "PersistentCUSUMConfig",
    "PersistentCUSUMError",
    "PersistentCUSUMObservation",
    "PersistentLaplacianCUSUM",
]
