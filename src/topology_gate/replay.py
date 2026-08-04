"""Dependency-light causal replay state machine.

The numerical workers in :mod:`topology_gate.online` and
:mod:`topology_gate.backtest` intentionally remain compatibility adapters. A
research run that needs point-in-time semantics can use this module as the
orchestration boundary instead. It materializes an :class:`AsOfSnapshot`,
settles labels that became available before the current prediction, predicts
the next target, and appends a hash-chained immutable record.

The callbacks receive only the current snapshot and frozen receipts. They are
still responsible for being pure/transactional; a Python callback can always
close over unsafe state. ``require_model_state`` and the manifest/checkpoint
contracts make that limitation explicit rather than silently presenting an
untracked callback as reproducible.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Protocol, TypeAlias, cast

from .asof import AsOfBook, AsOfSnapshot, LabelObservation, TimePoint

REPLAY_SCHEMA = "topology_gate.causal_replay"
REPLAY_VERSION = 1
MAX_REPLAY_DECISIONS = 100_000
MAX_REPLAY_RECORDS = 500_000


class ReplayError(ValueError):
    """Base class for causal replay contract failures."""


class ReplayStateError(ReplayError):
    """A checkpoint or callback state is incompatible with the replay."""


class ReplayStatus(str, Enum):
    """Statuses emitted by the replay boundary."""

    PREDICTED = "predicted"
    ABSTAINED = "abstained"
    INVALID = "invalid"
    OBSERVED = "observed"
    MISSING = "missing"
    CENSORED = "censored"
    INVALID_LABEL = "invalid_label"
    UNRESOLVED = "unresolved"


class ReplayModel(Protocol):
    """Minimum state contract for a checkpointable replay callback."""

    def state_dict(self) -> Any:
        """Return JSON-safe state that determines future model output."""


PredictionFn: TypeAlias = Callable[[AsOfSnapshot, str], Any]
ScoreFn: TypeAlias = Callable[["ReplayPrediction", LabelObservation], Any]
LabelFn: TypeAlias = Callable[["ReplayPrediction", LabelObservation, float | None], None]


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReplayError(f"{name} must be a non-empty string")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ReplayError(f"{name} must be finite")
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ReplayError(f"{name} must be finite") from exc
    if not math.isfinite(converted):
        raise ReplayError(f"{name} must be finite")
    return converted


def _time_key(value: TimePoint) -> tuple[str, Any]:
    if isinstance(value, datetime):
        return ("datetime", value.isoformat())
    if isinstance(value, bool):
        raise ReplayError("time points must not be booleans")
    if isinstance(value, (int, float, str)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ReplayError("time points must be finite")
        return (type(value).__name__, value)
    raise ReplayError("unsupported time point")


def _encode_time(value: TimePoint) -> dict[str, Any]:
    kind, encoded = _time_key(value)
    return {"kind": kind, "value": encoded}


def _decode_time(value: Any) -> TimePoint:
    if not isinstance(value, Mapping):
        raise ReplayStateError("encoded time point must be a mapping")
    kind = value.get("kind")
    raw = value.get("value")
    if kind == "datetime" and isinstance(raw, str):
        try:
            return datetime.fromisoformat(raw)
        except ValueError as exc:
            raise ReplayStateError("invalid encoded datetime") from exc
    if kind == "int" and isinstance(raw, int) and not isinstance(raw, bool):
        return raw
    if kind == "float" and isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return _finite(raw, "encoded time point")
    if kind == "str" and isinstance(raw, str):
        return raw
    raise ReplayStateError("invalid encoded time point")


def _json_safe(value: Any, *, path: str = "state") -> Any:
    """Copy a JSON-safe value while rejecting hidden non-determinism."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ReplayStateError(f"{path} contains a non-finite number")
        return value
    if isinstance(value, datetime):
        return {"__datetime__": value.isoformat()}
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            key_text = _text(key, f"{path} key")
            result[key_text] = _json_safe(item, path=f"{path}.{key_text}")
        return result
    if isinstance(value, (tuple, list)):
        return [
            _json_safe(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise ReplayStateError(f"{path} contains unsupported type {type(value).__name__}")


def _canonical(value: Any) -> bytes:
    return json.dumps(
        _json_safe(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _less(left: TimePoint, right: TimePoint) -> bool:
    try:
        return bool(left < right)  # type: ignore[operator]
    except TypeError as exc:
        raise ReplayError("decision times must use one comparable time domain") from exc


def _state_snapshot(model: Any, *, required: bool) -> tuple[str, Any]:
    state_fn = getattr(model, "state_dict", None)
    if not callable(state_fn):
        if required:
            raise ReplayStateError(
                "replay model must expose state_dict when require_model_state=True"
            )
        return "untracked", None
    state = _json_safe(state_fn())
    return _digest(state), state


def _validate_status(value: Any, name: str) -> ReplayStatus:
    try:
        return value if isinstance(value, ReplayStatus) else ReplayStatus(str(value))
    except ValueError as exc:
        raise ReplayStateError(f"{name} has an unsupported status") from exc


@dataclass(frozen=True, slots=True)
class ReplayConfig:
    """Versioned limits and identity for one causal replay."""

    model_id: str = "model"
    score_id: str = "none"
    require_model_state: bool = True
    finalize_unresolved: bool = False
    max_decisions: int = MAX_REPLAY_DECISIONS
    max_records: int = MAX_REPLAY_RECORDS

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_id", _text(self.model_id, "model_id"))
        object.__setattr__(self, "score_id", _text(self.score_id, "score_id"))
        if not isinstance(self.require_model_state, bool):
            raise ReplayError("require_model_state must be boolean")
        if not isinstance(self.finalize_unresolved, bool):
            raise ReplayError("finalize_unresolved must be boolean")
        for name, value, maximum in (
            ("max_decisions", self.max_decisions, MAX_REPLAY_DECISIONS),
            ("max_records", self.max_records, MAX_REPLAY_RECORDS),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ReplayError(f"{name} must be a positive integer")
            if value > maximum:
                raise ReplayError(f"{name} exceeds the replay resource limit")

    @property
    def identity(self) -> str:
        return _digest(
            {
                "schema": REPLAY_SCHEMA,
                "version": REPLAY_VERSION,
                "model_id": self.model_id,
                "score_id": self.score_id,
                "require_model_state": self.require_model_state,
                "finalize_unresolved": self.finalize_unresolved,
                "max_decisions": self.max_decisions,
                "max_records": self.max_records,
            }
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": REPLAY_SCHEMA,
            "version": REPLAY_VERSION,
            "model_id": self.model_id,
            "score_id": self.score_id,
            "require_model_state": self.require_model_state,
            "finalize_unresolved": self.finalize_unresolved,
            "max_decisions": self.max_decisions,
            "max_records": self.max_records,
            "identity": self.identity,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "ReplayConfig":
        if not isinstance(state, Mapping):
            raise ReplayStateError("replay config must be a mapping")
        if state.get("schema") != REPLAY_SCHEMA or state.get("version") != REPLAY_VERSION:
            raise ReplayStateError("unsupported replay config")
        candidate = cls(
            model_id=cast(str, state.get("model_id")),
            score_id=cast(str, state.get("score_id")),
            require_model_state=cast(bool, state.get("require_model_state")),
            finalize_unresolved=cast(bool, state.get("finalize_unresolved")),
            max_decisions=cast(int, state.get("max_decisions")),
            max_records=cast(int, state.get("max_records")),
        )
        if state.get("identity") != candidate.identity:
            raise ReplayStateError("replay config identity mismatch")
        return candidate


@dataclass(frozen=True, slots=True)
class ReplayPrediction:
    """Prediction frozen before any later label can update the model."""

    prediction_id: str
    target_id: str
    decision_time: TimePoint
    sequence: int
    snapshot_digest: str
    value: float | None
    status: ReplayStatus
    model_state_before: str
    model_state_after: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "prediction_id", _text(self.prediction_id, "prediction_id"))
        object.__setattr__(self, "target_id", _text(self.target_id, "target_id"))
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int) or self.sequence < 0:
            raise ReplayError("prediction sequence must be a non-negative integer")
        object.__setattr__(self, "snapshot_digest", _text(self.snapshot_digest, "snapshot_digest"))
        status = _validate_status(self.status, "prediction status")
        if status not in {
            ReplayStatus.PREDICTED,
            ReplayStatus.ABSTAINED,
            ReplayStatus.INVALID,
        }:
            raise ReplayError("prediction status is not a prediction status")
        if status is ReplayStatus.PREDICTED:
            if self.value is None:
                raise ReplayError("predicted records require a value")
            object.__setattr__(self, "value", _finite(self.value, "prediction value"))
        elif self.value is not None:
            raise ReplayError("abstained/invalid records must not carry a value")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "model_state_before", _text(self.model_state_before, "model_state_before"))
        object.__setattr__(self, "model_state_after", _text(self.model_state_after, "model_state_after"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "prediction_id": self.prediction_id,
            "target_id": self.target_id,
            "decision_time": _encode_time(self.decision_time),
            "sequence": self.sequence,
            "snapshot_digest": self.snapshot_digest,
            "value": self.value,
            "status": self.status.value,
            "model_state_before": self.model_state_before,
            "model_state_after": self.model_state_after,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "ReplayPrediction":
        return cls(
            prediction_id=cast(str, state.get("prediction_id")),
            target_id=cast(str, state.get("target_id")),
            decision_time=_decode_time(state.get("decision_time")),
            sequence=cast(int, state.get("sequence")),
            snapshot_digest=cast(str, state.get("snapshot_digest")),
            value=state.get("value"),
            status=cast(ReplayStatus, state.get("status")),
            model_state_before=cast(str, state.get("model_state_before")),
            model_state_after=cast(str, state.get("model_state_after")),
        )


@dataclass(frozen=True, slots=True)
class ReplayResolution:
    """One immutable settlement or explicit missing-label outcome."""

    resolution_id: str
    prediction_id: str | None
    target_id: str
    label_id: str | None
    label_revision: int | None
    settlement_time: TimePoint
    sequence: int
    status: ReplayStatus
    score: float | None
    reason: str | None
    model_state_before: str
    model_state_after: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "resolution_id", _text(self.resolution_id, "resolution_id"))
        if self.prediction_id is not None:
            object.__setattr__(self, "prediction_id", _text(self.prediction_id, "prediction_id"))
        object.__setattr__(self, "target_id", _text(self.target_id, "target_id"))
        if self.label_id is not None:
            object.__setattr__(self, "label_id", _text(self.label_id, "label_id"))
        if self.label_revision is not None:
            if isinstance(self.label_revision, bool) or not isinstance(self.label_revision, int) or self.label_revision < 0:
                raise ReplayError("label_revision must be a non-negative integer")
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int) or self.sequence < 0:
            raise ReplayError("resolution sequence must be a non-negative integer")
        status = _validate_status(self.status, "resolution status")
        if status not in {
            ReplayStatus.OBSERVED,
            ReplayStatus.MISSING,
            ReplayStatus.CENSORED,
            ReplayStatus.INVALID_LABEL,
            ReplayStatus.UNRESOLVED,
        }:
            raise ReplayError("resolution status is not a label status")
        if self.score is not None:
            object.__setattr__(self, "score", _finite(self.score, "label score"))
        if self.reason is not None:
            object.__setattr__(self, "reason", _text(self.reason, "reason"))
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "model_state_before", _text(self.model_state_before, "model_state_before"))
        object.__setattr__(self, "model_state_after", _text(self.model_state_after, "model_state_after"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "resolution_id": self.resolution_id,
            "prediction_id": self.prediction_id,
            "target_id": self.target_id,
            "label_id": self.label_id,
            "label_revision": self.label_revision,
            "settlement_time": _encode_time(self.settlement_time),
            "sequence": self.sequence,
            "status": self.status.value,
            "score": self.score,
            "reason": self.reason,
            "model_state_before": self.model_state_before,
            "model_state_after": self.model_state_after,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "ReplayResolution":
        return cls(
            resolution_id=cast(str, state.get("resolution_id")),
            prediction_id=state.get("prediction_id"),
            target_id=cast(str, state.get("target_id")),
            label_id=state.get("label_id"),
            label_revision=state.get("label_revision"),
            settlement_time=_decode_time(state.get("settlement_time")),
            sequence=cast(int, state.get("sequence")),
            status=cast(ReplayStatus, state.get("status")),
            score=state.get("score"),
            reason=state.get("reason"),
            model_state_before=cast(str, state.get("model_state_before")),
            model_state_after=cast(str, state.get("model_state_after")),
        )


@dataclass(frozen=True, slots=True)
class ReplayRecord:
    """Hash-chained canonical audit record."""

    sequence: int
    record_type: str
    payload: Mapping[str, Any]
    previous_digest: str
    digest: str

    def __post_init__(self) -> None:
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int) or self.sequence < 0:
            raise ReplayError("record sequence must be a non-negative integer")
        object.__setattr__(self, "record_type", _text(self.record_type, "record_type"))
        payload = _json_safe(self.payload)
        if not isinstance(payload, dict):
            raise ReplayError("record payload must be a mapping")
        object.__setattr__(self, "payload", payload)
        object.__setattr__(self, "previous_digest", _text(self.previous_digest, "previous_digest"))
        expected = _digest(
            {
                "schema": REPLAY_SCHEMA,
                "version": REPLAY_VERSION,
                "sequence": self.sequence,
                "record_type": self.record_type,
                "payload": payload,
                "previous_digest": self.previous_digest,
            }
        )
        if self.digest != expected:
            raise ReplayStateError("replay record digest mismatch")
        object.__setattr__(self, "digest", _text(self.digest, "digest"))

    @classmethod
    def create(
        cls, sequence: int, record_type: str, payload: Mapping[str, Any], previous_digest: str
    ) -> "ReplayRecord":
        digest = _digest(
            {
                "schema": REPLAY_SCHEMA,
                "version": REPLAY_VERSION,
                "sequence": sequence,
                "record_type": record_type,
                "payload": payload,
                "previous_digest": previous_digest,
            }
        )
        return cls(sequence, record_type, payload, previous_digest, digest)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "record_type": self.record_type,
            "payload": dict(self.payload),
            "previous_digest": self.previous_digest,
            "digest": self.digest,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "ReplayRecord":
        return cls(
            sequence=cast(int, state.get("sequence")),
            record_type=cast(str, state.get("record_type")),
            payload=cast(Mapping[str, Any], state.get("payload")),
            previous_digest=cast(str, state.get("previous_digest")),
            digest=cast(str, state.get("digest")),
        )


@dataclass(frozen=True, slots=True)
class ReplayState:
    """Serializable state at a decision boundary."""

    config_identity: str
    book_digest: str
    next_sequence: int
    last_decision_time: TimePoint | None
    predictions: tuple[ReplayPrediction, ...]
    resolutions: tuple[ReplayResolution, ...]
    records: tuple[ReplayRecord, ...]
    chain_digest: str
    model_state_digest: str
    model_state: Any

    def __post_init__(self) -> None:
        object.__setattr__(self, "config_identity", _text(self.config_identity, "config_identity"))
        object.__setattr__(self, "book_digest", _text(self.book_digest, "book_digest"))
        if isinstance(self.next_sequence, bool) or not isinstance(self.next_sequence, int) or self.next_sequence < 0:
            raise ReplayError("next_sequence must be a non-negative integer")
        object.__setattr__(self, "chain_digest", _text(self.chain_digest, "chain_digest"))
        object.__setattr__(self, "model_state_digest", _text(self.model_state_digest, "model_state_digest"))
        if self.last_decision_time is not None:
            _time_key(self.last_decision_time)
        object.__setattr__(self, "predictions", tuple(self.predictions))
        object.__setattr__(self, "resolutions", tuple(self.resolutions))
        object.__setattr__(self, "records", tuple(self.records))
        if self.next_sequence != len(self.records):
            raise ReplayStateError("next_sequence does not match the record chain length")
        object.__setattr__(self, "model_state", _json_safe(self.model_state))

    @property
    def pending_target_ids(self) -> tuple[str, ...]:
        settled = {
            item.target_id for item in self.resolutions if item.prediction_id is not None
        }
        return tuple(
            item.target_id for item in self.predictions if item.target_id not in settled
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": REPLAY_SCHEMA,
            "version": REPLAY_VERSION,
            "config_identity": self.config_identity,
            "book_digest": self.book_digest,
            "next_sequence": self.next_sequence,
            "last_decision_time": None
            if self.last_decision_time is None
            else _encode_time(self.last_decision_time),
            "predictions": [item.to_dict() for item in self.predictions],
            "resolutions": [item.to_dict() for item in self.resolutions],
            "records": [item.to_dict() for item in self.records],
            "chain_digest": self.chain_digest,
            "model_state_digest": self.model_state_digest,
            "model_state": self.model_state,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "ReplayState":
        if not isinstance(state, Mapping):
            raise ReplayStateError("replay state must be a mapping")
        if state.get("schema") != REPLAY_SCHEMA or state.get("version") != REPLAY_VERSION:
            raise ReplayStateError("unsupported replay state")
        predictions = tuple(
            ReplayPrediction.from_state_dict(item)
            for item in state.get("predictions", ())
        )
        resolutions = tuple(
            ReplayResolution.from_state_dict(item)
            for item in state.get("resolutions", ())
        )
        records = tuple(
            ReplayRecord.from_state_dict(item) for item in state.get("records", ())
        )
        candidate = cls(
            config_identity=cast(str, state.get("config_identity")),
            book_digest=cast(str, state.get("book_digest")),
            next_sequence=cast(int, state.get("next_sequence")),
            last_decision_time=None
            if state.get("last_decision_time") is None
            else _decode_time(state.get("last_decision_time")),
            predictions=predictions,
            resolutions=resolutions,
            records=records,
            chain_digest=cast(str, state.get("chain_digest")),
            model_state_digest=cast(str, state.get("model_state_digest")),
            model_state=state.get("model_state"),
        )
        _validate_record_chain(candidate.records, candidate.chain_digest)
        return candidate


@dataclass(frozen=True, slots=True)
class CausalReplayResult:
    """Terminal output of one fresh or resumed replay segment."""

    predictions: tuple[ReplayPrediction, ...]
    resolutions: tuple[ReplayResolution, ...]
    records: tuple[ReplayRecord, ...]
    state: ReplayState

    @property
    def pending_target_ids(self) -> tuple[str, ...]:
        return self.state.pending_target_ids

    @property
    def chain_digest(self) -> str:
        return self.state.chain_digest

    def state_dict(self) -> dict[str, Any]:
        return self.state.state_dict()


def _validate_record_chain(records: Sequence[ReplayRecord], terminal_digest: str) -> None:
    previous = "0" * 64
    for expected_sequence, record in enumerate(records):
        if record.sequence != expected_sequence:
            raise ReplayStateError("replay record sequence is not contiguous")
        if record.previous_digest != previous:
            raise ReplayStateError("replay record chain is broken")
        previous = record.digest
    if terminal_digest != previous:
        raise ReplayStateError("replay terminal digest does not match record chain")


class CausalReplay:
    """Run one deterministic prediction/label transition per decision time."""

    def __init__(
        self,
        book: AsOfBook,
        predictor: PredictionFn,
        *,
        score: ScoreFn | None = None,
        on_label: LabelFn | None = None,
        config: ReplayConfig | None = None,
    ) -> None:
        if not callable(predictor):
            raise TypeError("predictor must be callable")
        if score is not None and not callable(score):
            raise TypeError("score must be callable")
        if on_label is not None and not callable(on_label):
            raise TypeError("on_label must be callable")
        self.book = book
        self.predictor = predictor
        self.score = score
        self.on_label = on_label
        self.config = config or ReplayConfig(
            score_id="none" if score is None else "callable"
        )

    def _empty_state(self, model: Any) -> ReplayState:
        model_digest, model_state = _state_snapshot(
            model, required=self.config.require_model_state
        )
        return ReplayState(
            config_identity=self.config.identity,
            book_digest=self.book.digest,
            next_sequence=0,
            last_decision_time=None,
            predictions=(),
            resolutions=(),
            records=(),
            chain_digest="0" * 64,
            model_state_digest=model_digest,
            model_state=model_state,
        )

    def _check_initial_state(self, state: ReplayState, model: Any) -> None:
        if state.config_identity != self.config.identity:
            raise ReplayStateError("initial replay state uses a different configuration")
        if state.book_digest != self.book.digest:
            raise ReplayStateError("initial replay state uses a different input book")
        if state.next_sequence != len(state.records):
            raise ReplayStateError("initial replay state has an invalid record count")
        _validate_record_chain(state.records, state.chain_digest)
        model_digest, _ = _state_snapshot(model, required=self.config.require_model_state)
        if model_digest != state.model_state_digest:
            raise ReplayStateError("model state does not match the replay checkpoint")

    @staticmethod
    def _validate_times(times: tuple[TimePoint, ...], limit: int) -> None:
        if len(times) > limit:
            raise ReplayError("decision times exceed the replay resource limit")
        for left, right in zip(times, times[1:]):
            if not _less(left, right):
                raise ReplayError("decision times must be strictly increasing")

    @staticmethod
    def _validate_targets(targets: tuple[str, ...], expected: int) -> None:
        if len(targets) != expected:
            raise ReplayError("target_ids must align with decision_times")
        normalized = tuple(_text(value, "target_id") for value in targets)
        if len(set(normalized)) != len(normalized):
            raise ReplayError("target_ids must be unique within a replay")

    def _append(
        self,
        records: list[ReplayRecord],
        sequence: int,
        record_type: str,
        payload: Mapping[str, Any],
        previous_digest: str,
    ) -> tuple[int, str]:
        if len(records) >= self.config.max_records:
            raise ReplayError("replay record limit exceeded")
        record = ReplayRecord.create(sequence, record_type, payload, previous_digest)
        records.append(record)
        return sequence + 1, record.digest

    def run(
        self,
        decision_times: Iterable[TimePoint],
        target_ids: Iterable[str],
        *,
        model: Any,
        initial_state: ReplayState | None = None,
    ) -> CausalReplayResult:
        """Replay strictly increasing decisions and return checkpointable state.

        Labels visible at a decision boundary are settled *before* that
        boundary's new prediction. A label that is already visible for the
        target being predicted is rejected: it would be an impossible
        supervised-prediction request and accepting it would leak the target.
        """

        times = tuple(decision_times)
        targets = tuple(target_ids)
        self._validate_times(times, self.config.max_decisions)
        self._validate_targets(targets, len(times))
        if initial_state is None:
            state = self._empty_state(model)
        else:
            self._check_initial_state(initial_state, model)
            state = initial_state
        if state.last_decision_time is not None and times:
            if not _less(state.last_decision_time, times[0]):
                raise ReplayError("resumed decisions must follow the checkpoint boundary")

        predictions = list(state.predictions)
        resolutions = list(state.resolutions)
        records = list(state.records)
        prediction_by_target = {item.target_id: item for item in predictions}
        settled_targets = {
            item.target_id for item in resolutions if item.prediction_id is not None
        }
        sequence = state.next_sequence
        chain_digest = state.chain_digest
        last_time = state.last_decision_time

        for decision_time, target_id in zip(times, targets):
            snapshot = self.book.materialize(decision_time)

            # Settlement is deterministic by target ID and cannot see the
            # prediction created at this same boundary.
            visible_labels = {item.target_id: item for item in snapshot.labels}
            for pending_target in sorted(prediction_by_target):
                if pending_target in settled_targets:
                    continue
                label = visible_labels.get(pending_target)
                if label is None:
                    continue
                prediction = prediction_by_target[pending_target]
                before_digest, _ = _state_snapshot(
                    model, required=self.config.require_model_state
                )
                score_value: float | None = None
                resolution_status = {
                    "observed": ReplayStatus.OBSERVED,
                    "missing": ReplayStatus.MISSING,
                    "censored": ReplayStatus.CENSORED,
                    "invalid": ReplayStatus.INVALID_LABEL,
                }[label.status]
                reason: str | None = None
                if label.status == "observed":
                    if prediction.status is not ReplayStatus.PREDICTED:
                        reason = "prediction was not valid"
                    else:
                        if self.score is not None:
                            score_value = _finite(self.score(prediction, label), "label score")
                        if self.on_label is not None:
                            self.on_label(prediction, label, score_value)
                else:
                    reason = f"label status is {label.status}"
                after_digest, _ = _state_snapshot(
                    model, required=self.config.require_model_state
                )
                resolution = ReplayResolution(
                    resolution_id=f"{self.config.model_id}:resolution:{sequence}",
                    prediction_id=prediction.prediction_id,
                    target_id=pending_target,
                    label_id=label.label_id,
                    label_revision=label.source_revision,
                    settlement_time=decision_time,
                    sequence=sequence,
                    status=resolution_status,
                    score=score_value,
                    reason=reason,
                    model_state_before=before_digest,
                    model_state_after=after_digest,
                )
                sequence, chain_digest = self._append(
                    records,
                    sequence,
                    "label_resolution",
                    resolution.to_dict(),
                    chain_digest,
                )
                resolutions.append(resolution)
                settled_targets.add(pending_target)

            if target_id in visible_labels:
                raise ReplayError(
                    f"target label {target_id!r} is already available at decision time"
                )
            if target_id in prediction_by_target:
                raise ReplayError(f"target {target_id!r} was already predicted")

            before_digest, _ = _state_snapshot(model, required=self.config.require_model_state)
            raw_value = self.predictor(snapshot, target_id)
            if raw_value is None:
                prediction_status = ReplayStatus.ABSTAINED
                prediction_value = None
            else:
                try:
                    prediction_value = _finite(raw_value, "prediction")
                except ReplayError:
                    prediction_status = ReplayStatus.INVALID
                    prediction_value = None
                else:
                    prediction_status = ReplayStatus.PREDICTED
            after_digest, _ = _state_snapshot(model, required=self.config.require_model_state)
            prediction = ReplayPrediction(
                prediction_id=f"{self.config.model_id}:prediction:{sequence}",
                target_id=target_id,
                decision_time=decision_time,
                sequence=sequence,
                snapshot_digest=snapshot.digest,
                value=prediction_value,
                status=prediction_status,
                model_state_before=before_digest,
                model_state_after=after_digest,
            )
            sequence, chain_digest = self._append(
                records, sequence, "prediction", prediction.to_dict(), chain_digest
            )
            predictions.append(prediction)
            prediction_by_target[target_id] = prediction
            last_time = decision_time

        if self.config.finalize_unresolved:
            for pending_target in sorted(prediction_by_target):
                if pending_target in settled_targets:
                    continue
                before_digest, _ = _state_snapshot(
                    model, required=self.config.require_model_state
                )
                resolution = ReplayResolution(
                    resolution_id=f"{self.config.model_id}:resolution:{sequence}",
                    prediction_id=prediction_by_target[pending_target].prediction_id,
                    target_id=pending_target,
                    label_id=None,
                    label_revision=None,
                    settlement_time=last_time
                    if last_time is not None
                    else 0,
                    sequence=sequence,
                    status=ReplayStatus.UNRESOLVED,
                    score=None,
                    reason="no label was available at the terminal boundary",
                    model_state_before=before_digest,
                    model_state_after=before_digest,
                )
                sequence, chain_digest = self._append(
                    records,
                    sequence,
                    "unresolved_label",
                    resolution.to_dict(),
                    chain_digest,
                )
                resolutions.append(resolution)
                settled_targets.add(pending_target)

        model_digest, model_state = _state_snapshot(
            model, required=self.config.require_model_state
        )
        final_state = ReplayState(
            config_identity=self.config.identity,
            book_digest=self.book.digest,
            next_sequence=sequence,
            last_decision_time=last_time,
            predictions=tuple(predictions),
            resolutions=tuple(resolutions),
            records=tuple(records),
            chain_digest=chain_digest,
            model_state_digest=model_digest,
            model_state=model_state,
        )
        return CausalReplayResult(
            predictions=final_state.predictions,
            resolutions=final_state.resolutions,
            records=final_state.records,
            state=final_state,
        )


def run_causal_replay(
    book: AsOfBook,
    decision_times: Iterable[TimePoint],
    target_ids: Iterable[str],
    predictor: PredictionFn,
    *,
    model: Any,
    score: ScoreFn | None = None,
    on_label: LabelFn | None = None,
    config: ReplayConfig | None = None,
    initial_state: ReplayState | None = None,
) -> CausalReplayResult:
    """Functional wrapper for :class:`CausalReplay`."""

    return CausalReplay(
        book,
        predictor,
        score=score,
        on_label=on_label,
        config=config,
    ).run(
        decision_times,
        target_ids,
        model=model,
        initial_state=initial_state,
    )


__all__ = [
    "CausalReplay",
    "CausalReplayResult",
    "LabelFn",
    "MAX_REPLAY_DECISIONS",
    "MAX_REPLAY_RECORDS",
    "PredictionFn",
    "ReplayConfig",
    "ReplayError",
    "ReplayModel",
    "ReplayPrediction",
    "ReplayRecord",
    "ReplayResolution",
    "ReplayState",
    "ReplayStateError",
    "ReplayStatus",
    "REPLAY_SCHEMA",
    "REPLAY_VERSION",
    "run_causal_replay",
]
