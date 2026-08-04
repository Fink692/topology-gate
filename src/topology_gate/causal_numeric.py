"""Point-in-time numerical adapter built on the shared causal replay loop.

This module is the migration seam between the dependency-light causal
contracts and the existing NumPy detector/RLS workers. It deliberately does
not compute portfolio returns: labels and tradable realized returns are
different data products, and the latter belong to the economic evaluator.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, cast

import numpy as np

from .asof import AsOfBook, AsOfSnapshot
from .replay import (
    CausalReplayResult,
    ReplayConfig,
    ReplayPrediction,
    ReplayState,
    ReplayStatus,
    run_causal_replay,
)

CAUSAL_NUMERIC_SCHEMA = "topology_gate.causal_numeric"
CAUSAL_NUMERIC_VERSION = 1
MAX_CAUSAL_FEATURE_BINDINGS = 256
MAX_CAUSAL_PENDING_CONTEXTS = 8_192


class CausalNumericError(ValueError):
    """Base error for the point-in-time numerical adapter."""


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CausalNumericError(f"{name} must be a non-empty string")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise CausalNumericError(f"{name} must be finite")
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CausalNumericError(f"{name} must be finite") from exc
    if not math.isfinite(converted):
        raise CausalNumericError(f"{name} must be finite")
    return converted


@dataclass(frozen=True, slots=True)
class FeatureBinding:
    """One immutable observation/field binding in a point-in-time feature row."""

    record_id: str
    fields: tuple[str, ...]
    instrument_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "record_id", _text(self.record_id, "record_id"))
        if isinstance(self.fields, (str, bytes, bytearray)):
            raise CausalNumericError("fields must be a sequence of field names")
        fields = tuple(_text(value, "field name") for value in self.fields)
        if not fields:
            raise CausalNumericError("a feature binding must contain at least one field")
        if len(set(fields)) != len(fields):
            raise CausalNumericError("feature binding fields must be unique")
        object.__setattr__(self, "fields", fields)
        if self.instrument_id is not None:
            object.__setattr__(
                self, "instrument_id", _text(self.instrument_id, "instrument_id")
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "fields": list(self.fields),
            "instrument_id": self.instrument_id,
        }


def _normalize_binding_map(
    value: Mapping[str, Sequence[FeatureBinding]], name: str
) -> Mapping[str, tuple[FeatureBinding, ...]]:
    if not isinstance(value, Mapping) or not value:
        raise CausalNumericError(f"{name} must be a non-empty mapping")
    normalized: dict[str, tuple[FeatureBinding, ...]] = {}
    total = 0
    for target, bindings in value.items():
        target_id = _text(target, f"{name} target")
        if isinstance(bindings, (str, bytes, bytearray)):
            raise CausalNumericError(f"{name}[{target_id!r}] must be a sequence")
        entries = tuple(bindings)
        if not entries or not all(isinstance(item, FeatureBinding) for item in entries):
            raise CausalNumericError(
                f"{name}[{target_id!r}] must contain FeatureBinding values"
            )
        record_ids = [item.record_id for item in entries]
        if len(set(record_ids)) != len(record_ids):
            raise CausalNumericError(
                f"{name}[{target_id!r}] must not repeat observation records"
            )
        total += len(entries)
        if total > MAX_CAUSAL_FEATURE_BINDINGS:
            raise CausalNumericError("feature binding count exceeds the resource limit")
        normalized[target_id] = entries
    return MappingProxyType(normalized)


@dataclass(frozen=True, slots=True)
class CausalFeaturePlan:
    """Versioned feature/state bindings for each predicted target."""

    bindings_by_target: Mapping[str, Sequence[FeatureBinding]]
    state_bindings_by_target: Mapping[str, Sequence[FeatureBinding]] | None = None
    require_membership: bool = False

    def __post_init__(self) -> None:
        bindings = _normalize_binding_map(self.bindings_by_target, "bindings_by_target")
        if self.state_bindings_by_target is None:
            state_bindings = bindings
        else:
            state_bindings = _normalize_binding_map(
                self.state_bindings_by_target, "state_bindings_by_target"
            )
            if set(state_bindings) != set(bindings):
                raise CausalNumericError(
                    "state bindings must cover exactly the feature targets"
                )
        if not isinstance(self.require_membership, bool):
            raise CausalNumericError("require_membership must be boolean")
        if self.require_membership:
            for group in (*bindings.values(), *state_bindings.values()):
                if any(binding.instrument_id is None for binding in group):
                    raise CausalNumericError(
                        "strict membership plans require instrument_id on every binding"
                    )
        object.__setattr__(self, "bindings_by_target", bindings)
        object.__setattr__(self, "state_bindings_by_target", state_bindings)

    @property
    def identity(self) -> str:
        payload = {
            "schema": CAUSAL_NUMERIC_SCHEMA,
            "version": CAUSAL_NUMERIC_VERSION,
            "require_membership": self.require_membership,
            "bindings": {
                target: [binding.to_dict() for binding in bindings]
                for target, bindings in sorted(self.bindings_by_target.items())
            },
            "state_bindings": {
                target: [binding.to_dict() for binding in bindings]
                for target, bindings in sorted(
                    cast(Mapping[str, tuple[FeatureBinding, ...]], self.state_bindings_by_target).items()
                )
            },
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def _bindings(self, target_id: str, *, state: bool) -> tuple[FeatureBinding, ...]:
        target = _text(target_id, "target_id")
        source = self.state_bindings_by_target if state else self.bindings_by_target
        assert source is not None
        try:
            return cast(Mapping[str, tuple[FeatureBinding, ...]], source)[target]
        except KeyError as exc:
            kind = "state" if state else "feature"
            raise CausalNumericError(
                f"no {kind} bindings are registered for target {target!r}"
            ) from exc

    @staticmethod
    def _before_or_equal(left: Any, right: Any) -> bool:
        try:
            return bool(left <= right)
        except TypeError as exc:
            raise CausalNumericError(
                "event and decision times must use one comparable domain"
            ) from exc

    def extract(
        self, snapshot: AsOfSnapshot, target_id: str, *, state: bool = False
    ) -> tuple[float, ...]:
        """Extract one row using only records visible in ``snapshot``."""

        values: list[float] = []
        for binding in self._bindings(target_id, state=state):
            observation = snapshot.observation(binding.record_id)
            if not self._before_or_equal(observation.event_time, snapshot.decision_time):
                raise CausalNumericError(
                    f"observation {binding.record_id!r} has a future event time"
                )
            if (
                binding.instrument_id is not None
                and observation.instrument_id != binding.instrument_id
            ):
                raise CausalNumericError(
                    f"observation {binding.record_id!r} has the wrong instrument"
                )
            if self.require_membership:
                if binding.instrument_id is None:
                    raise CausalNumericError(
                        "strict membership extraction requires instrument_id"
                    )
                if not snapshot.is_member(binding.instrument_id, at=observation.event_time):
                    raise CausalNumericError(
                        f"instrument {binding.instrument_id!r} was not in the point-in-time universe"
                    )
            for field in binding.fields:
                try:
                    values.append(_finite(observation.fields[field], f"field {field!r}"))
                except KeyError as exc:
                    raise CausalNumericError(
                        f"observation {binding.record_id!r} is missing field {field!r}"
                    ) from exc
        if not values:
            raise CausalNumericError("extracted feature row must not be empty")
        return tuple(values)


@dataclass(frozen=True, slots=True)
class CausalRLSConfig:
    """State-affecting policy for a causal detector/RLS adapter."""

    model_id: str = "causal-rls"
    position_scale: float = 1.0
    position_limit: float = 1.0
    default_forgetting_factor: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_id", _text(self.model_id, "model_id"))
        scale = _finite(self.position_scale, "position_scale")
        limit = _finite(self.position_limit, "position_limit")
        if scale <= 0.0 or limit <= 0.0:
            raise CausalNumericError("position scale and limit must be positive")
        object.__setattr__(self, "position_scale", scale)
        object.__setattr__(self, "position_limit", limit)
        if self.default_forgetting_factor is not None:
            factor = _finite(
                self.default_forgetting_factor, "default_forgetting_factor"
            )
            if not 0.0 < factor <= 1.0:
                raise CausalNumericError(
                    "default_forgetting_factor must be in (0, 1]"
                )
            object.__setattr__(self, "default_forgetting_factor", factor)

    @property
    def identity(self) -> str:
        payload = {
            "schema": CAUSAL_NUMERIC_SCHEMA,
            "version": CAUSAL_NUMERIC_VERSION,
            "model_id": self.model_id,
            "position_scale": self.position_scale,
            "position_limit": self.position_limit,
            "default_forgetting_factor": self.default_forgetting_factor,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class CausalStep:
    """Numerical telemetry emitted for one prediction boundary."""

    target_id: str
    feature_digest: str
    state_digest: str
    score: float
    alarm: bool
    ready: bool
    acceleration_authorized: bool
    forgetting_factor: float
    position: float
    method: str
    topology_evidence_digest: str | None = None


def _digest_row(values: Sequence[float]) -> str:
    payload = json.dumps([_finite(value, "row value") for value in values], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _optional_digest(value: Any, name: str) -> str | None:
    if value is None:
        return None
    digest = _text(value, name)
    if len(digest) != 64 or any(item not in "0123456789abcdefABCDEF" for item in digest):
        raise CausalNumericError(f"{name} must be a 64-character hexadecimal digest")
    return digest.lower()


class CausalRLSModel:
    """Adapter that runs the real detector and RLS behind ``CausalReplay``."""

    def __init__(
        self,
        learner: Any,
        plan: CausalFeaturePlan,
        *,
        detector: Any | None = None,
        config: CausalRLSConfig | None = None,
        calibration: Any | None = None,
    ) -> None:
        if not callable(getattr(learner, "predict", None)) or not callable(
            getattr(learner, "update", None)
        ):
            raise TypeError("learner must expose predict and update")
        if not callable(getattr(learner, "state_dict", None)) or not callable(
            getattr(learner, "load_state_dict", None)
        ):
            raise TypeError("learner must expose state_dict and load_state_dict")
        if detector is not None and (
            not callable(getattr(detector, "observe", None))
            or not callable(getattr(detector, "stream_state_dict", None))
            or not callable(getattr(detector, "load_stream_state_dict", None))
        ):
            raise TypeError(
                "detector must expose observe, stream_state_dict, and load_stream_state_dict"
            )
        self.learner = learner
        self.plan = plan
        self.detector = detector
        self.config = config or CausalRLSConfig()
        if detector is None and calibration is not None:
            raise CausalNumericError(
                "a calibration certificate requires a topology detector"
            )
        self.calibration = calibration
        self._calibration_identity = self._validate_calibration()
        self._pending: dict[str, tuple[tuple[float, ...], float]] = {}
        self._steps: list[CausalStep] = []

    @property
    def steps(self) -> tuple[CausalStep, ...]:
        return tuple(self._steps)

    def _default_factor(self) -> float:
        configured = self.config.default_forgetting_factor
        if configured is not None:
            return self._validate_forgetting_factor(configured, "default forgetting factor")
        detector_config = getattr(self.detector, "config", None)
        maximum = getattr(detector_config, "forgetting_lambda_max", None)
        if maximum is None:
            maximum = getattr(self.learner, "lambda_max", 1.0)
        return self._validate_forgetting_factor(maximum, "default forgetting factor")

    def _validate_calibration(self) -> str | None:
        if self.detector is None:
            return None
        detector_identity = getattr(self.detector, "config_identity", None)
        if callable(detector_identity):
            detector_identity = detector_identity()
        if not isinstance(detector_identity, str) or not detector_identity:
            raise CausalNumericError(
                "a topology detector must expose config_identity for calibration"
            )
        if self.calibration is None:
            return None
        certificate_identity = getattr(self.calibration, "detector_identity", None)
        if certificate_identity != detector_identity:
            raise CausalNumericError(
                "calibration certificate does not match detector identity"
            )
        identity = _text(
            getattr(self.calibration, "identity", None),
            "calibration certificate identity",
        )
        approved = getattr(self.calibration, "approved", False)
        if not isinstance(approved, bool):
            raise CausalNumericError("calibration certificate approval is invalid")
        return identity

    @property
    def calibration_authorized(self) -> bool:
        """Whether the supplied finite-null certificate permits acceleration."""

        return self.detector is None or bool(
            self.calibration is not None
            and self._calibration_identity is not None
            and getattr(self.calibration, "approved", False)
        )

    def _validate_forgetting_factor(self, value: Any, name: str) -> float:
        factor = _finite(value, name)
        if not 0.0 < factor <= 1.0:
            raise CausalNumericError(f"{name} must be in (0, 1]")
        lower = getattr(self.learner, "lambda_min", None)
        upper = getattr(self.learner, "lambda_max", None)
        if lower is not None and upper is not None:
            lower_value = _finite(lower, "learner lambda_min")
            upper_value = _finite(upper, "learner lambda_max")
            if not lower_value <= factor <= upper_value:
                raise CausalNumericError(
                    f"{name} must lie within the learner lambda bounds"
                )
        return factor

    def predict(self, snapshot: AsOfSnapshot, target_id: str) -> float:
        features = self.plan.extract(snapshot, target_id)
        state_features = self.plan.extract(snapshot, target_id, state=True)
        learner_before = self.learner.state_dict()
        detector_before = (
            None
            if self.detector is None
            else self.detector.stream_state_dict()
        )
        try:
            return self._predict_transaction(snapshot, target_id, features, state_features)
        except Exception:
            # The numerical workers expose validated loaders. Restore both
            # components if an adapter-side failure occurs after one of them
            # has advanced, preserving the all-or-nothing transition rule.
            self.learner.load_state_dict(learner_before)
            if self.detector is not None:
                assert detector_before is not None
                self.detector.load_stream_state_dict(detector_before)
            raise

    def _predict_transaction(
        self,
        snapshot: AsOfSnapshot,
        target_id: str,
        features: tuple[float, ...],
        state_features: tuple[float, ...],
    ) -> float:
        score = 0.0
        alarm = False
        ready = True
        method = "none"
        topology_evidence_digest: str | None = None
        if self.detector is not None:
            detection = self.detector.observe(np.asarray(state_features, dtype=float))
            score = _finite(detection.score, "detector score")
            alarm = bool(detection.alarm)
            ready = bool(detection.ready)
            reported_factor = _finite(
                detection.forgetting_factor, "forgetting factor"
            )
            # Insufficient or uncertified topology is never allowed to
            # accelerate memory.  The detector's neutral factor is the
            # learner/configured maximum, captured before the label.
            acceleration_authorized = ready and self.calibration_authorized
            if not acceleration_authorized:
                factor = self._default_factor()
            else:
                factor = self._validate_forgetting_factor(
                    reported_factor, "forgetting factor"
            )
            method = _text(detection.method, "detector method")
            topology_evidence_digest = _optional_digest(
                getattr(detection, "backend_evidence_digest", None),
                "topology evidence digest",
            )
        else:
            factor = self._default_factor()
            acceleration_authorized = True
        raw = self.learner.predict(np.asarray(features, dtype=float))
        values = np.asarray(raw, dtype=float).reshape(-1)
        if values.size != 1:
            raise CausalNumericError("causal RLS adapter requires a scalar learner output")
        prediction = float(values[0])
        position = (
            float(
                np.clip(
                    prediction / self.config.position_scale,
                    -self.config.position_limit,
                    self.config.position_limit,
                )
            )
            if math.isfinite(prediction)
            else 0.0
        )
        if math.isfinite(prediction) and len(self._pending) >= MAX_CAUSAL_PENDING_CONTEXTS:
            raise CausalNumericError("pending causal contexts exceed the resource limit")
        self._steps.append(
            CausalStep(
                target_id=_text(target_id, "target_id"),
                feature_digest=_digest_row(features),
                state_digest=_digest_row(state_features),
                score=score,
                alarm=alarm,
                ready=ready,
                acceleration_authorized=acceleration_authorized,
                forgetting_factor=factor,
                position=position,
                method=method,
                topology_evidence_digest=topology_evidence_digest,
            )
        )
        if math.isfinite(prediction):
            self._pending[target_id] = (features, factor)
        return prediction

    def on_label(
        self, prediction: ReplayPrediction, label: Any, score: float | None
    ) -> None:
        del score
        if prediction.status is not ReplayStatus.PREDICTED:
            return
        if getattr(label, "status", None) != "observed":
            return
        context = self._pending.get(prediction.target_id)
        if context is None:
            raise CausalNumericError(
                f"no frozen update context exists for {prediction.target_id!r}"
            )
        features, factor = context
        target = _finite(getattr(label, "value", None), "label value")
        self.learner.update(
            np.asarray(features, dtype=float), target, forgetting_factor=factor
        )

    def on_resolution(
        self, prediction: ReplayPrediction, label: Any, status: ReplayStatus
    ) -> None:
        del label, status
        self._pending.pop(prediction.target_id, None)

    def state_dict(self) -> dict[str, Any]:
        detector_state = None
        detector_identity = None
        if self.detector is not None:
            detector_state = self.detector.stream_state_dict()
            detector_identity = getattr(self.detector, "config_identity", None)
            if callable(detector_identity):
                detector_identity = detector_identity()
        return {
            "schema": CAUSAL_NUMERIC_SCHEMA,
            "version": CAUSAL_NUMERIC_VERSION,
            "model_id": self.config.model_id,
            "config_identity": self.config.identity,
            "plan_identity": self.plan.identity,
            "calibration_identity": self._calibration_identity,
            "calibration_authorized": self.calibration_authorized,
            "learner": self.learner.state_dict(),
            "detector_identity": detector_identity,
            "detector": detector_state,
            "pending": {
                target: {"features": list(features), "forgetting_factor": factor}
                for target, (features, factor) in sorted(self._pending.items())
            },
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping):
            raise CausalNumericError("causal model state must be a mapping")
        if state.get("schema") != CAUSAL_NUMERIC_SCHEMA or state.get("version") != CAUSAL_NUMERIC_VERSION:
            raise CausalNumericError("unsupported causal numerical model state")
        if state.get("model_id") != self.config.model_id:
            raise CausalNumericError("causal model identity mismatch")
        if state.get("config_identity") != self.config.identity:
            raise CausalNumericError("causal model configuration identity mismatch")
        if state.get("plan_identity") != self.plan.identity:
            raise CausalNumericError("causal feature plan identity mismatch")
        if state.get("calibration_identity") != self._calibration_identity:
            raise CausalNumericError("causal calibration identity mismatch")
        if state.get("calibration_authorized", self.calibration_authorized) != self.calibration_authorized:
            raise CausalNumericError("causal calibration authorization mismatch")
        learner_state = state.get("learner")
        if not isinstance(learner_state, Mapping):
            raise CausalNumericError("causal model state is missing learner state")
        detector_state = state.get("detector")
        if self.detector is None:
            if detector_state is not None:
                raise CausalNumericError("checkpoint contains detector state but no detector is configured")
            normalized_detector = None
        else:
            if not isinstance(detector_state, Mapping):
                raise CausalNumericError("checkpoint is missing detector state")
            validate = getattr(self.detector, "validate_stream_state_dict", None)
            if not callable(validate):
                raise CausalNumericError("detector cannot validate state before restore")
            normalized_detector = validate(detector_state)
            if state.get("detector_identity") != getattr(
                self.detector, "config_identity", None
            ):
                raise CausalNumericError("causal detector identity mismatch")
        raw_pending = state.get("pending", {})
        if not isinstance(raw_pending, Mapping):
            raise CausalNumericError("causal pending state must be a mapping")
        pending: dict[str, tuple[tuple[float, ...], float]] = {}
        if len(raw_pending) > MAX_CAUSAL_PENDING_CONTEXTS:
            raise CausalNumericError("causal pending state exceeds the resource limit")
        expected_features = getattr(self.learner, "n_features", None)
        for target, raw in raw_pending.items():
            target_id = _text(target, "pending target")
            if not isinstance(raw, Mapping):
                raise CausalNumericError("pending context must be a mapping")
            features_raw = raw.get("features")
            if not isinstance(features_raw, Sequence) or isinstance(
                features_raw, (str, bytes, bytearray)
            ):
                raise CausalNumericError("pending features must be numeric")
            features = tuple(_finite(value, "pending feature") for value in features_raw)
            if not features or (
                expected_features is not None and len(features) != int(expected_features)
            ):
                raise CausalNumericError("pending feature dimension does not match learner")
            factor = _finite(raw.get("forgetting_factor"), "pending forgetting_factor")
            if not 0.0 < factor <= 1.0:
                raise CausalNumericError("pending forgetting_factor must be in (0, 1]")
            pending[target_id] = (features, factor)

        # The component loaders validate their own candidate state before
        # committing. Detector validation above occurs before the learner load,
        # so malformed detector state cannot be hidden by a partial restore.
        self.learner.load_state_dict(learner_state)
        if self.detector is not None:
            assert normalized_detector is not None
            self.detector.load_stream_state_dict(normalized_detector)
        self._pending = pending
        self._steps = []


@dataclass(frozen=True, slots=True)
class CausalRLSReplayResult:
    """Numerical outputs plus the canonical causal replay result."""

    replay: CausalReplayResult
    steps: tuple[CausalStep, ...]
    prediction_start: int = 0

    @property
    def all_predictions(self) -> tuple[ReplayPrediction, ...]:
        """Return the complete prediction history in the replay checkpoint."""

        return self.replay.predictions

    @property
    def predictions(self) -> np.ndarray[Any, Any]:
        current = self.replay.predictions[self.prediction_start :]
        return np.asarray(
            [
                item.value if item.value is not None else math.nan
                for item in current
            ],
            dtype=float,
        )

    @property
    def positions(self) -> np.ndarray[Any, Any]:
        return np.asarray([item.position for item in self.steps], dtype=float)

    @property
    def detector_scores(self) -> np.ndarray[Any, Any]:
        return np.asarray([item.score for item in self.steps], dtype=float)

    @property
    def alarms(self) -> np.ndarray[Any, Any]:
        return np.asarray([item.alarm for item in self.steps], dtype=bool)

    @property
    def forgetting_factors(self) -> np.ndarray[Any, Any]:
        return np.asarray([item.forgetting_factor for item in self.steps], dtype=float)

    @property
    def topology_evidence_digests(self) -> tuple[str | None, ...]:
        """Return the exact-backend artifact digest captured per step."""

        return tuple(item.topology_evidence_digest for item in self.steps)

    @property
    def pending_target_ids(self) -> tuple[str, ...]:
        return self.replay.pending_target_ids

    @property
    def state(self) -> ReplayState:
        return self.replay.state

    def state_dict(self) -> dict[str, Any]:
        return self.replay.state_dict()


def run_causal_rls_replay(
    book: AsOfBook,
    decision_times: Sequence[Any],
    target_ids: Sequence[str],
    *,
    plan: CausalFeaturePlan,
    learner: Any,
    detector: Any | None = None,
    calibration: Any | None = None,
    model_config: CausalRLSConfig | None = None,
    replay_config: ReplayConfig | None = None,
    model_state: Mapping[str, Any] | None = None,
    initial_state: ReplayState | None = None,
) -> CausalRLSReplayResult:
    """Run detector-gated RLS through the shared point-in-time transition."""

    numeric_config = model_config or CausalRLSConfig()
    model = CausalRLSModel(
        learner,
        plan,
        detector=detector,
        config=numeric_config,
        calibration=calibration,
    )
    if model_state is not None:
        model.load_state_dict(model_state)
    elif initial_state is not None:
        raise CausalNumericError(
            "model_state is required when resuming a causal numerical replay"
        )
    settings = replay_config or ReplayConfig(
        model_id=numeric_config.model_id,
        score_id="none",
        require_model_state=True,
    )
    if settings.model_id != numeric_config.model_id:
        raise CausalNumericError("replay and numerical model identities must match")
    result = run_causal_replay(
        book,
        decision_times,
        target_ids,
        model.predict,
        model=model,
        on_label=model.on_label,
        on_resolution=model.on_resolution,
        config=settings,
        initial_state=initial_state,
    )
    prediction_start = 0 if initial_state is None else len(initial_state.predictions)
    return CausalRLSReplayResult(result, model.steps, prediction_start)


__all__ = [
    "CAUSAL_NUMERIC_SCHEMA",
    "CAUSAL_NUMERIC_VERSION",
    "CausalFeaturePlan",
    "CausalNumericError",
    "CausalRLSConfig",
    "CausalRLSModel",
    "CausalRLSReplayResult",
    "CausalStep",
    "FeatureBinding",
    "MAX_CAUSAL_FEATURE_BINDINGS",
    "MAX_CAUSAL_PENDING_CONTEXTS",
    "run_causal_rls_replay",
]
