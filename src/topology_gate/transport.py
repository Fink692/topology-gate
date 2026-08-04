"""Prefix-causal transport replay for drifting linear targets.

This module implements the first, deliberately narrow version of causal
transport replay proposed in the study plan.  A historical observation is
eligible only after its declared availability boundary.  The replay then
transports its feature location and linear target through the latest parameter
and location state that existed at the requested decision step:

    x_tilde = x_i + (mu_t - mu_i)
    y_tilde = y_i + x_tilde @ (theta_t - theta_i)

The location shift is a causal translation, not a claim to solve adapted
Wasserstein optimal transport.  Reliability is an explicit bounded weight
that decays with the observed parameter and location displacement.  Callers
must provide prefix-only state estimates to ``observe_state``.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any, cast

try:
    import numpy as np
except ModuleNotFoundError as exc:  # pragma: no cover - import boundary
    raise ImportError(
        "topology_gate.transport requires NumPy; install "
        "`topology-gate[numeric]`"
    ) from exc


TRANSPORT_REPLAY_SCHEMA = "topology_gate.transport_replay"
TRANSPORT_REPLAY_VERSION = 1
MAX_TRANSPORT_RECORDS = 100_000
MAX_TRANSPORT_STATES = 100_000


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
    return np.array(result, dtype=float, copy=True)


def _digest(value: Mapping[str, Any]) -> str:
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("transport replay state is not JSON-safe") from exc
    return hashlib.sha256(payload).hexdigest()


def _tuple_vector(value: Any, name: str, width: int) -> tuple[float, ...]:
    return tuple(float(item) for item in _vector(value, name, width))


@dataclass(frozen=True, slots=True)
class TransportReplayConfig:
    """Resource and reliability policy for one transport replay family."""

    n_features: int
    drift_sensitivity: float = 1.0
    location_sensitivity: float = 1.0
    minimum_weight: float = 1.0e-6
    max_records: int = MAX_TRANSPORT_RECORDS
    max_states: int = MAX_TRANSPORT_STATES

    def __post_init__(self) -> None:
        features = _integer("n_features", self.n_features, 1, 256)
        drift = _finite("drift_sensitivity", self.drift_sensitivity, minimum=0.0)
        location = _finite(
            "location_sensitivity", self.location_sensitivity, minimum=0.0
        )
        minimum = _finite("minimum_weight", self.minimum_weight, minimum=0.0)
        if minimum <= 0.0 or minimum > 1.0:
            raise ValueError("minimum_weight must lie in (0, 1]")
        records = _integer("max_records", self.max_records, 1, MAX_TRANSPORT_RECORDS)
        states = _integer("max_states", self.max_states, 1, MAX_TRANSPORT_STATES)
        object.__setattr__(self, "n_features", features)
        object.__setattr__(self, "drift_sensitivity", drift)
        object.__setattr__(self, "location_sensitivity", location)
        object.__setattr__(self, "minimum_weight", minimum)
        object.__setattr__(self, "max_records", records)
        object.__setattr__(self, "max_states", states)

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": TRANSPORT_REPLAY_VERSION,
            "schema": TRANSPORT_REPLAY_SCHEMA,
            "n_features": self.n_features,
            "drift_sensitivity": self.drift_sensitivity,
            "location_sensitivity": self.location_sensitivity,
            "minimum_weight": self.minimum_weight,
            "max_records": self.max_records,
            "max_states": self.max_states,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "TransportReplayConfig":
        if not isinstance(state, Mapping):
            raise ValueError("transport replay config must be a mapping")
        expected = {
            "version",
            "schema",
            "n_features",
            "drift_sensitivity",
            "location_sensitivity",
            "minimum_weight",
            "max_records",
            "max_states",
        }
        if set(state) != expected:
            raise ValueError("transport replay config fields are invalid")
        if (
            state.get("version") != TRANSPORT_REPLAY_VERSION
            or state.get("schema") != TRANSPORT_REPLAY_SCHEMA
        ):
            raise ValueError("unsupported transport replay config")
        return cls(
            n_features=cast(int, state["n_features"]),
            drift_sensitivity=cast(float, state["drift_sensitivity"]),
            location_sensitivity=cast(float, state["location_sensitivity"]),
            minimum_weight=cast(float, state["minimum_weight"]),
            max_records=cast(int, state["max_records"]),
            max_states=cast(int, state["max_states"]),
        )

    @property
    def identity(self) -> str:
        return _digest(self.state_dict())


@dataclass(frozen=True, slots=True)
class TransportReplayBatch:
    """Causally eligible, transported observations at one decision step."""

    current_step: int
    snapshot_step: int
    source_steps: tuple[int, ...]
    available_steps: tuple[int, ...]
    features: np.ndarray[Any, Any]
    labels: np.ndarray[Any, Any]
    weights: np.ndarray[Any, Any]

    def __post_init__(self) -> None:
        current = _integer("current_step", self.current_step, 0, MAX_TRANSPORT_STATES)
        snapshot = _integer(
            "snapshot_step", self.snapshot_step, 0, MAX_TRANSPORT_STATES
        )
        if snapshot > current:
            raise ValueError("snapshot_step cannot be after current_step")
        if len(self.source_steps) != len(self.available_steps):
            raise ValueError("transport batch step arrays must align")
        features = np.asarray(self.features, dtype=float)
        labels = np.asarray(self.labels, dtype=float)
        weights = np.asarray(self.weights, dtype=float)
        if features.ndim != 2 or labels.ndim != 1 or weights.ndim != 1:
            raise ValueError("transport batch arrays have invalid dimensions")
        if features.shape[0] != len(self.source_steps):
            raise ValueError("transport batch rows do not match source steps")
        if labels.size != features.shape[0] or weights.size != features.shape[0]:
            raise ValueError("transport batch arrays do not align")
        if not np.all(np.isfinite(features)) or not np.all(np.isfinite(labels)):
            raise ValueError("transport batch values must be finite")
        if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0) or np.any(weights > 1.0):
            raise ValueError("transport batch weights must lie in (0, 1]")
        if any(source >= current for source in self.source_steps):
            raise ValueError("transport batch contains a future source step")
        if any(available >= current for available in self.available_steps):
            raise ValueError("transport batch contains an unavailable label")
        object.__setattr__(self, "current_step", current)
        object.__setattr__(self, "snapshot_step", snapshot)
        object.__setattr__(self, "features", np.array(features, copy=True))
        object.__setattr__(self, "labels", np.array(labels, copy=True))
        object.__setattr__(self, "weights", np.array(weights, copy=True))

    @property
    def n_rows(self) -> int:
        return len(self.source_steps)


class CausalTransportReplay:
    """Maintain a prefix-only replay memory with causal linear transport."""

    def __init__(self, config: TransportReplayConfig) -> None:
        if not isinstance(config, TransportReplayConfig):
            raise TypeError("config must be a TransportReplayConfig")
        self.config = config
        self._records: list[dict[str, Any]] = []
        self._states: list[dict[str, Any]] = []

    @property
    def config_identity(self) -> str:
        return self.config.identity

    @property
    def record_count(self) -> int:
        return len(self._records)

    @property
    def state_count(self) -> int:
        return len(self._states)

    def append(
        self,
        source_step: int,
        available_step: int,
        features: Any,
        label: Any,
        theta: Any,
        *,
        feature_location: Any | None = None,
    ) -> None:
        source = _integer("source_step", source_step, 0, MAX_TRANSPORT_STATES - 1)
        available = _integer(
            "available_step", available_step, 1, MAX_TRANSPORT_STATES
        )
        if available <= source:
            raise ValueError("available_step must be strictly after source_step")
        if len(self._records) >= self.config.max_records:
            raise ValueError("transport replay record limit exceeded")
        record: dict[str, Any] = {
            "source_step": source,
            "available_step": available,
            "features": _tuple_vector(features, "features", self.config.n_features),
            "label": _finite("label", label),
            "theta": _tuple_vector(theta, "theta", self.config.n_features),
            "feature_location": None,
        }
        if feature_location is not None:
            record["feature_location"] = _tuple_vector(
                feature_location, "feature_location", self.config.n_features
            )
        self._records.append(record)

    def observe_state(
        self,
        step: int,
        theta: Any,
        *,
        feature_location: Any | None = None,
    ) -> None:
        """Record a state estimate that must have been computed from a prefix."""

        position = _integer("step", step, 0, MAX_TRANSPORT_STATES - 1)
        if self._states and position <= int(self._states[-1]["step"]):
            raise ValueError("transport state steps must be strictly increasing")
        if len(self._states) >= self.config.max_states:
            raise ValueError("transport replay state limit exceeded")
        state: dict[str, Any] = {
            "step": position,
            "theta": _tuple_vector(theta, "theta", self.config.n_features),
            "feature_location": None,
        }
        if feature_location is not None:
            state["feature_location"] = _tuple_vector(
                feature_location, "feature_location", self.config.n_features
            )
        self._states.append(state)

    def _state_at(self, current_step: int) -> dict[str, Any]:
        candidates = [
            state for state in self._states if int(state["step"]) <= current_step
        ]
        if not candidates:
            raise ValueError("no prefix state is available at current_step")
        return candidates[-1]

    def batch(self, current_step: int) -> TransportReplayBatch:
        """Return only records observable strictly before ``current_step``."""

        current = _integer("current_step", current_step, 0, MAX_TRANSPORT_STATES)
        state = self._state_at(current)
        snapshot_step = int(state["step"])
        current_theta = np.asarray(state["theta"], dtype=float)
        current_location = state["feature_location"]
        source_steps: list[int] = []
        available_steps: list[int] = []
        features: list[np.ndarray[Any, Any]] = []
        labels: list[float] = []
        weights: list[float] = []
        for record in self._records:
            source = int(record["source_step"])
            available = int(record["available_step"])
            if source >= current or available >= current:
                continue
            source_theta = np.asarray(record["theta"], dtype=float)
            transported_features = np.asarray(record["features"], dtype=float)
            source_location = record["feature_location"]
            location_delta = 0.0
            if current_location is not None and source_location is not None:
                location_delta_vector = np.asarray(current_location) - np.asarray(
                    source_location
                )
                transported_features = transported_features + location_delta_vector
                location_delta = float(np.linalg.norm(location_delta_vector))
            parameter_delta = current_theta - source_theta
            transported_label = float(
                record["label"] + transported_features @ parameter_delta
            )
            displacement = (
                self.config.drift_sensitivity * float(np.linalg.norm(parameter_delta))
                + self.config.location_sensitivity * location_delta
            )
            weight = max(
                self.config.minimum_weight,
                min(1.0, math.exp(-displacement)),
            )
            source_steps.append(source)
            available_steps.append(available)
            features.append(transported_features)
            labels.append(transported_label)
            weights.append(weight)
        if features:
            feature_array = np.vstack(features)
            label_array = np.asarray(labels, dtype=float)
            weight_array = np.asarray(weights, dtype=float)
        else:
            feature_array = np.empty((0, self.config.n_features), dtype=float)
            label_array = np.empty(0, dtype=float)
            weight_array = np.empty(0, dtype=float)
        return TransportReplayBatch(
            current_step=current,
            snapshot_step=snapshot_step,
            source_steps=tuple(source_steps),
            available_steps=tuple(available_steps),
            features=feature_array,
            labels=label_array,
            weights=weight_array,
        )

    def _state_without_identity(self) -> dict[str, Any]:
        return {
            "version": TRANSPORT_REPLAY_VERSION,
            "schema": TRANSPORT_REPLAY_SCHEMA,
            "config": self.config.state_dict(),
            "records": [dict(record) for record in self._records],
            "states": [dict(state) for state in self._states],
        }

    @property
    def identity(self) -> str:
        return _digest(self._state_without_identity())

    def state_dict(self) -> dict[str, Any]:
        state = self._state_without_identity()
        state["identity"] = self.identity
        return state

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping):
            raise ValueError("transport replay state must be a mapping")
        expected = {"version", "schema", "config", "records", "states", "identity"}
        if set(state) != expected:
            raise ValueError("transport replay state fields are invalid")
        if (
            state.get("version") != TRANSPORT_REPLAY_VERSION
            or state.get("schema") != TRANSPORT_REPLAY_SCHEMA
        ):
            raise ValueError("unsupported transport replay state")
        config = TransportReplayConfig.from_state_dict(cast(Mapping[str, Any], state["config"]))
        if config.identity != self.config.identity:
            raise ValueError("transport replay configuration identity mismatch")
        records_raw = state["records"]
        states_raw = state["states"]
        if not isinstance(records_raw, Sequence) or isinstance(records_raw, (str, bytes, bytearray)):
            raise ValueError("transport replay records must be a sequence")
        if not isinstance(states_raw, Sequence) or isinstance(states_raw, (str, bytes, bytearray)):
            raise ValueError("transport replay states must be a sequence")
        candidate = CausalTransportReplay(config)
        for raw in records_raw:
            if not isinstance(raw, Mapping):
                raise ValueError("transport replay record must be a mapping")
            expected_record = {
                "source_step",
                "available_step",
                "features",
                "label",
                "theta",
                "feature_location",
            }
            if set(raw) != expected_record:
                raise ValueError("transport replay record fields are invalid")
            candidate.append(
                cast(int, raw["source_step"]),
                cast(int, raw["available_step"]),
                raw["features"],
                raw["label"],
                raw["theta"],
                feature_location=raw["feature_location"],
            )
        for raw in states_raw:
            if not isinstance(raw, Mapping):
                raise ValueError("transport replay state entry must be a mapping")
            expected_entry = {"step", "theta", "feature_location"}
            if set(raw) != expected_entry:
                raise ValueError("transport replay state entry fields are invalid")
            candidate.observe_state(
                cast(int, raw["step"]),
                raw["theta"],
                feature_location=raw["feature_location"],
            )
        if state.get("identity") != candidate.identity:
            raise ValueError("transport replay identity mismatch")
        self._records = candidate._records
        self._states = candidate._states

    @classmethod
    def from_state_dict(
        cls, state: Mapping[str, Any]
    ) -> "CausalTransportReplay":
        if not isinstance(state, Mapping) or not isinstance(state.get("config"), Mapping):
            raise ValueError("transport replay state must contain a config mapping")
        replay = cls(TransportReplayConfig.from_state_dict(state["config"]))
        replay.load_state_dict(state)
        return replay


__all__ = [
    "CausalTransportReplay",
    "MAX_TRANSPORT_RECORDS",
    "MAX_TRANSPORT_STATES",
    "TRANSPORT_REPLAY_SCHEMA",
    "TRANSPORT_REPLAY_VERSION",
    "TransportReplayBatch",
    "TransportReplayConfig",
]
