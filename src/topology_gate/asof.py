"""Immutable point-in-time event contracts for causal replay.

This module intentionally has no third-party dependencies.  Events are
append-only facts and :class:`AsOfBook` never mutates an existing snapshot.
For a decision time ``t``, only records whose availability time is at most
``t`` are visible.  If several revisions of the same logical record are
visible, the greatest source revision wins; a later, lower revision can never
overwrite it.

The contract is deliberately strict at the data boundary.  Numeric fields
must be finite, event times must be ordered causally, revision keys must be
unique, and missing labels must be represented by an explicit status rather
than an absent or ``None`` value disguised as a valid observation.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Any, TypeAlias

TimePoint: TypeAlias = int | float | str | datetime
Revision: TypeAlias = int

MARKET_PRECEDENCE = 20
UNIVERSE_PRECEDENCE = 10
LABEL_PRECEDENCE = 30


class AsOfError(ValueError):
    """Base class for invalid point-in-time data or queries."""


class DuplicateEventError(AsOfError):
    """A logical event ID and source revision were repeated."""


class AmbiguousEventError(AsOfError):
    """The available event stream does not have a unique interpretation."""


class UnavailableEventError(AsOfError):
    """A requested record or label is not available at the decision time."""


class MissingLabelError(AsOfError):
    """A requested label is absent or explicitly marked as missing."""


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{name} must be a non-empty string")
    return value


def _revision(value: Any, name: str = "source_revision") -> Revision:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a non-negative integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _sequence(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("ingest_sequence must be a non-negative integer")
    if value < 0:
        raise ValueError("ingest_sequence must be non-negative")
    return value


def _time(value: Any, name: str) -> TimePoint:
    if isinstance(value, bool) or value is None:
        raise TypeError(f"{name} must be a finite time point")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if not isinstance(value, (int, float, str, datetime)):
        raise TypeError(f"{name} must be an int, float, str, or datetime")
    if isinstance(value, str) and not value:
        raise ValueError(f"{name} must not be empty")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a finite real number")
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be a finite real number") from exc
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _fields(value: Mapping[str, Any]) -> Mapping[str, float]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("fields must be a non-empty mapping")
    normalized: dict[str, float] = {}
    for key, item in value.items():
        normalized[_text(key, "field name")] = _finite(item, f"field {key!r}")
    return MappingProxyType(normalized)


def _le(left: TimePoint, right: TimePoint, name: str = "time") -> bool:
    """Compare times while turning heterogeneous comparisons into a rejection."""

    try:
        result = bool(left <= right)  # type: ignore[operator]
    except TypeError as exc:
        raise TypeError(f"{name} values must use one comparable time domain") from exc
    if not isinstance(result, bool):
        raise TypeError(f"{name} comparison did not produce a boolean")
    return result


def _time_domain(value: TimePoint) -> type[Any]:
    if isinstance(value, datetime):
        return datetime
    if isinstance(value, str):
        return str
    return float


def _time_json(value: TimePoint) -> str | float | int:
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _check_time_domains(events: Sequence[Any]) -> None:
    values = [event.available_time for event in events]
    domains = {_time_domain(value) for value in values}
    if len(domains) > 1:
        raise TypeError("all available_time values must use one time domain")
    for event in events:
        # Datetime values may be mixed only when Python can compare them; this
        # catches naive/aware datetime mixtures before sorting.
        _le(event.available_time, event.available_time, "available_time")


def _record_id(event: Any) -> str:
    if isinstance(event, MarketObservation):
        return event.record_id
    if isinstance(event, LabelObservation):
        return event.label_id
    return event.record_id


def _precedence(event: Any) -> int:
    if isinstance(event, UniverseMembership):
        return UNIVERSE_PRECEDENCE
    if isinstance(event, MarketObservation):
        return MARKET_PRECEDENCE
    if isinstance(event, LabelObservation):
        return LABEL_PRECEDENCE
    raise TypeError(f"unsupported event type: {type(event).__name__}")


def _ingest_sequence(event: Any) -> int:
    return event.ingest_sequence


def _sort_key(event: Any) -> tuple[TimePoint, int, int, str]:
    return (
        event.available_time,
        _precedence(event),
        _ingest_sequence(event),
        _record_id(event),
    )


def canonical_event_order(
    events: Iterable[MarketObservation | LabelObservation | UniverseMembership],
) -> tuple[MarketObservation | LabelObservation | UniverseMembership, ...]:
    """Return events in the deterministic causal order.

    The order is exactly ``(available_time, precedence, ingest_sequence,
    record_id)``.  Equal keys are rejected instead of relying on Python's
    stable-sort accident to choose an interpretation.
    """

    ordered_input = tuple(events)
    _check_time_domains(ordered_input)
    keyed: list[tuple[tuple[TimePoint, int, int, str], Any]] = []
    seen: set[tuple[Any, int, int, str]] = set()
    for event in ordered_input:
        key = _sort_key(event)
        if key in seen:
            raise AmbiguousEventError(f"ambiguous canonical event key: {key!r}")
        seen.add(key)
        keyed.append((key, event))
    keyed.sort(key=lambda item: item[0])
    return tuple(event for _, event in keyed)


@dataclass(frozen=True, slots=True)
class MarketObservation:
    """A feature record with an event-time and point-in-time availability."""

    record_id: str
    instrument_id: str
    event_time: TimePoint
    available_time: TimePoint
    source_revision: Revision
    ingest_sequence: int
    fields: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "record_id", _text(self.record_id, "record_id"))
        object.__setattr__(
            self, "instrument_id", _text(self.instrument_id, "instrument_id")
        )
        event_time = _time(self.event_time, "event_time")
        available_time = _time(self.available_time, "available_time")
        if not _le(event_time, available_time, "event/available time"):
            raise ValueError("event_time cannot be after available_time")
        object.__setattr__(self, "event_time", event_time)
        object.__setattr__(self, "available_time", available_time)
        object.__setattr__(self, "source_revision", _revision(self.source_revision))
        object.__setattr__(self, "ingest_sequence", _sequence(self.ingest_sequence))
        object.__setattr__(self, "fields", _fields(self.fields))

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "instrument_id": self.instrument_id,
            "event_time": _time_json(self.event_time),
            "available_time": _time_json(self.available_time),
            "source_revision": self.source_revision,
            "ingest_sequence": self.ingest_sequence,
            "fields": dict(self.fields),
        }


@dataclass(frozen=True, slots=True)
class LabelObservation:
    """A label whose availability and receipt are both explicit."""

    label_id: str
    target_id: str
    event_time: TimePoint
    available_time: TimePoint
    received_time: TimePoint
    status: str
    value: Any
    source_revision: Revision
    ingest_sequence: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "label_id", _text(self.label_id, "label_id"))
        object.__setattr__(self, "target_id", _text(self.target_id, "target_id"))
        event_time = _time(self.event_time, "event_time")
        available_time = _time(self.available_time, "available_time")
        received_time = _time(self.received_time, "received_time")
        if not _le(event_time, available_time, "event/available time"):
            raise ValueError("event_time cannot be after available_time")
        if not _le(available_time, received_time, "available/received time"):
            raise ValueError("available_time cannot be after received_time")
        status = _text(self.status, "status").lower()
        if status not in {"observed", "missing", "censored", "invalid"}:
            raise ValueError("status must be observed, missing, censored, or invalid")
        if status == "observed":
            if self.value is None:
                raise ValueError("observed labels require an explicit value")
            value = _finite(self.value, "label value")
        else:
            if self.value is not None:
                raise ValueError(f"{status} labels must not carry a value")
            value = None
        object.__setattr__(self, "event_time", event_time)
        object.__setattr__(self, "available_time", available_time)
        object.__setattr__(self, "received_time", received_time)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "source_revision", _revision(self.source_revision))
        object.__setattr__(self, "ingest_sequence", _sequence(self.ingest_sequence))

    def to_dict(self) -> dict[str, Any]:
        return {
            "label_id": self.label_id,
            "target_id": self.target_id,
            "event_time": _time_json(self.event_time),
            "available_time": _time_json(self.available_time),
            "received_time": _time_json(self.received_time),
            "status": self.status,
            "value": self.value,
            "source_revision": self.source_revision,
            "ingest_sequence": self.ingest_sequence,
        }


@dataclass(frozen=True, slots=True)
class UniverseMembership:
    """A versioned half-open membership interval ``[start, end)``."""

    instrument_id: str
    start: TimePoint
    end: TimePoint | None
    event_time: TimePoint
    available_time: TimePoint
    source_revision: Revision
    ingest_sequence: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "instrument_id", _text(self.instrument_id, "instrument_id")
        )
        start = _time(self.start, "start")
        event_time = _time(self.event_time, "event_time")
        available_time = _time(self.available_time, "available_time")
        end = None if self.end is None else _time(self.end, "end")
        if end is not None and not _le(start, end, "membership interval"):
            raise ValueError("start must be before end")
        if end is not None and start == end:
            raise ValueError("membership interval must not be empty")
        if not _le(event_time, available_time, "event/available time"):
            raise ValueError("event_time cannot be after available_time")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        object.__setattr__(self, "event_time", event_time)
        object.__setattr__(self, "available_time", available_time)
        object.__setattr__(self, "source_revision", _revision(self.source_revision))
        object.__setattr__(self, "ingest_sequence", _sequence(self.ingest_sequence))

    @property
    def record_id(self) -> str:
        """Stable logical record ID used by canonical event ordering."""

        return f"{self.instrument_id}|{self.start!r}"

    def contains(self, event_time: TimePoint) -> bool:
        """Return whether ``event_time`` is inside this membership interval."""

        point = _time(event_time, "event_time")
        if not _le(self.start, point, "membership interval"):
            return False
        return self.end is None or _le(point, self.end, "membership interval") and point != self.end

    def to_dict(self) -> dict[str, Any]:
        return {
            "instrument_id": self.instrument_id,
            "start": _time_json(self.start),
            "end": None if self.end is None else _time_json(self.end),
            "event_time": _time_json(self.event_time),
            "available_time": _time_json(self.available_time),
            "source_revision": self.source_revision,
            "ingest_sequence": self.ingest_sequence,
        }


@dataclass(frozen=True, slots=True)
class AsOfSnapshot:
    """Immutable materialization of one point-in-time view."""

    decision_time: TimePoint
    observations: tuple[MarketObservation, ...]
    universe: tuple[UniverseMembership, ...]
    labels: tuple[LabelObservation, ...]

    @property
    def digest(self) -> str:
        payload = {
            "decision_time": _time_json(self.decision_time),
            "observations": [value.to_dict() for value in self.observations],
            "universe": [value.to_dict() for value in self.universe],
            "labels": [value.to_dict() for value in self.labels],
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    @property
    def universe_digest(self) -> str:
        """Digest only the membership view used at this decision boundary."""

        payload = {
            "decision_time": _time_json(self.decision_time),
            "universe": [value.to_dict() for value in self.universe],
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    @property
    def memberships(self) -> tuple[UniverseMembership, ...]:
        return self.universe

    def observation(self, record_id: str) -> MarketObservation:
        for observation in self.observations:
            if observation.record_id == record_id:
                return observation
        raise UnavailableEventError(f"observation {record_id!r} is not available")

    def label_for(self, target_id: str) -> LabelObservation:
        for label in self.labels:
            if label.target_id == target_id:
                if label.status != "observed":
                    raise MissingLabelError(
                        f"label for target {target_id!r} is explicitly {label.status}"
                    )
                return label
        raise MissingLabelError(f"label for target {target_id!r} is unavailable")

    def is_member(self, instrument_id: str, at: TimePoint | None = None) -> bool:
        point = self.decision_time if at is None else _time(at, "membership time")
        return any(
            membership.instrument_id == instrument_id and membership.contains(point)
            for membership in self.universe
        )


@dataclass(frozen=True, slots=True)
class PointInTimePanel:
    """Canonical cross-asset rows selected from one as-of snapshot.

    A panel is intentionally constructed from explicit record IDs.  The
    snapshot has already applied availability and source-revision rules; this
    layer adds deterministic instrument ordering, one record per instrument,
    fixed field ordering, and a membership digest.  It is a data-boundary
    artifact, not a vendor adapter or a claim that the selected universe is
    economically complete.
    """

    decision_time: TimePoint
    instrument_ids: tuple[str, ...]
    record_ids: tuple[str, ...]
    field_names: tuple[str, ...]
    values: tuple[tuple[float, ...], ...]
    universe_digest: str

    def __post_init__(self) -> None:
        decision = _time(self.decision_time, "decision_time")
        instruments = tuple(_text(value, "instrument_id") for value in self.instrument_ids)
        records = tuple(_text(value, "record_id") for value in self.record_ids)
        fields = tuple(_text(value, "field name") for value in self.field_names)
        if not instruments or len(set(instruments)) != len(instruments):
            raise ValueError("instrument_ids must be non-empty and unique")
        if tuple(sorted(instruments)) != instruments:
            raise ValueError("instrument_ids must be in canonical sorted order")
        if len(records) != len(instruments) or len(set(records)) != len(records):
            raise ValueError("record_ids must align one-to-one with instruments")
        if not fields or len(set(fields)) != len(fields):
            raise ValueError("field_names must be non-empty and unique")
        raw_rows = tuple(tuple(_finite(item, "panel value") for item in row) for row in self.values)
        if len(raw_rows) != len(instruments) or any(len(row) != len(fields) for row in raw_rows):
            raise ValueError("panel values do not match instrument and field dimensions")
        universe_digest = _text(self.universe_digest, "universe_digest")
        if len(universe_digest) != 64 or any(
            item not in "0123456789abcdefABCDEF" for item in universe_digest
        ):
            raise ValueError("universe_digest must be a 64-character hexadecimal digest")
        object.__setattr__(self, "decision_time", decision)
        object.__setattr__(self, "instrument_ids", instruments)
        object.__setattr__(self, "record_ids", records)
        object.__setattr__(self, "field_names", fields)
        object.__setattr__(self, "values", raw_rows)
        object.__setattr__(self, "universe_digest", universe_digest.lower())

    @classmethod
    def from_snapshot(
        cls,
        snapshot: AsOfSnapshot,
        record_ids: Sequence[str],
        field_names: Sequence[str],
        *,
        require_membership: bool = True,
    ) -> "PointInTimePanel":
        """Select and canonicalize one row per instrument from a snapshot."""

        if not isinstance(snapshot, AsOfSnapshot):
            raise TypeError("snapshot must be an AsOfSnapshot")
        if isinstance(record_ids, (str, bytes, bytearray)):
            raise TypeError("record_ids must be a sequence of record IDs")
        selected_ids = tuple(_text(value, "record_id") for value in record_ids)
        if not selected_ids or len(set(selected_ids)) != len(selected_ids):
            raise ValueError("record_ids must be non-empty and unique")
        if isinstance(field_names, (str, bytes, bytearray)):
            raise TypeError("field_names must be a sequence of field names")
        fields = tuple(_text(value, "field name") for value in field_names)
        if not fields or len(set(fields)) != len(fields):
            raise ValueError("field_names must be non-empty and unique")
        if not isinstance(require_membership, bool):
            raise TypeError("require_membership must be boolean")

        rows: list[tuple[str, str, tuple[float, ...]]] = []
        seen_instruments: set[str] = set()
        for record_id in selected_ids:
            observation = snapshot.observation(record_id)
            if not _le(observation.event_time, snapshot.decision_time, "event/decision time"):
                raise UnavailableEventError(
                    f"observation {record_id!r} has a future event time"
                )
            if require_membership and not snapshot.is_member(
                observation.instrument_id, at=observation.event_time
            ):
                raise UnavailableEventError(
                    f"instrument {observation.instrument_id!r} is not in the point-in-time universe"
                )
            if observation.instrument_id in seen_instruments:
                raise AmbiguousEventError(
                    f"multiple panel records selected for instrument {observation.instrument_id!r}"
                )
            seen_instruments.add(observation.instrument_id)
            values: list[float] = []
            for field in fields:
                try:
                    values.append(_finite(observation.fields[field], f"field {field!r}"))
                except KeyError as exc:
                    raise UnavailableEventError(
                        f"observation {record_id!r} is missing field {field!r}"
                    ) from exc
            rows.append((observation.instrument_id, record_id, tuple(values)))

        rows.sort(key=lambda item: item[0])
        return cls(
            decision_time=snapshot.decision_time,
            instrument_ids=tuple(item[0] for item in rows),
            record_ids=tuple(item[1] for item in rows),
            field_names=fields,
            values=tuple(item[2] for item in rows),
            universe_digest=snapshot.universe_digest,
        )

    @property
    def digest(self) -> str:
        """Content identity of the canonical panel artifact."""

        payload = {
            "decision_time": _time_json(self.decision_time),
            "instrument_ids": list(self.instrument_ids),
            "record_ids": list(self.record_ids),
            "field_names": list(self.field_names),
            "values": [list(row) for row in self.values],
            "universe_digest": self.universe_digest,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "point_in_time_panel",
            "decision_time": _time_json(self.decision_time),
            "instrument_ids": list(self.instrument_ids),
            "record_ids": list(self.record_ids),
            "field_names": list(self.field_names),
            "values": [list(row) for row in self.values],
            "universe_digest": self.universe_digest,
            "digest": self.digest,
        }


def _latest_by(
    records: Iterable[Any],
    logical_key: Any,
) -> dict[Any, Any]:
    """Select the highest revision per logical key without silent ties."""

    selected: dict[Any, Any] = {}
    for record in records:
        key = logical_key(record)
        previous = selected.get(key)
        if previous is None or record.source_revision > previous.source_revision:
            selected[key] = record
        elif record.source_revision == previous.source_revision and record != previous:
            raise AmbiguousEventError(f"ambiguous revision for logical key {key!r}")
    return selected


@dataclass(frozen=True, slots=True, init=False)
class AsOfBook:
    """Append-only event book that produces immutable causal snapshots."""

    _observations: tuple[MarketObservation, ...]
    _universe: tuple[UniverseMembership, ...]
    _labels: tuple[LabelObservation, ...]

    def __init__(
        self,
        observations: Iterable[MarketObservation] = (),
        universe: Iterable[UniverseMembership] = (),
        labels: Iterable[LabelObservation] = (),
    ) -> None:
        observation_values = tuple(observations)
        universe_values = tuple(universe)
        label_values = tuple(labels)
        if not all(isinstance(item, MarketObservation) for item in observation_values):
            raise TypeError("observations must contain MarketObservation values")
        if not all(isinstance(item, UniverseMembership) for item in universe_values):
            raise TypeError("universe must contain UniverseMembership values")
        if not all(isinstance(item, LabelObservation) for item in label_values):
            raise TypeError("labels must contain LabelObservation values")
        _check_unique_revisions(observation_values, lambda item: (item.record_id, item.source_revision))
        _check_unique_revisions(label_values, lambda item: (item.label_id, item.source_revision))
        _check_unique_revisions(
            universe_values,
            lambda item: (item.instrument_id, item.start, item.source_revision),
        )
        canonical_event_order((*observation_values, *universe_values, *label_values))
        object.__setattr__(self, "_observations", observation_values)
        object.__setattr__(self, "_universe", universe_values)
        object.__setattr__(self, "_labels", label_values)

    @property
    def observations(self) -> tuple[MarketObservation, ...]:
        return self._observations

    @property
    def universe(self) -> tuple[UniverseMembership, ...]:
        return self._universe

    @property
    def labels(self) -> tuple[LabelObservation, ...]:
        return self._labels

    @property
    def digest(self) -> str:
        payload = {
            "observations": [value.to_dict() for value in self._observations],
            "universe": [value.to_dict() for value in self._universe],
            "labels": [value.to_dict() for value in self._labels],
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    @property
    def ordered_events(
        self,
    ) -> tuple[MarketObservation | LabelObservation | UniverseMembership, ...]:
        return canonical_event_order((*self._observations, *self._universe, *self._labels))

    def with_observation(self, observation: MarketObservation) -> AsOfBook:
        return AsOfBook((*self._observations, observation), self._universe, self._labels)

    def with_universe(self, membership: UniverseMembership) -> AsOfBook:
        return AsOfBook(self._observations, (*self._universe, membership), self._labels)

    def with_label(self, label: LabelObservation) -> AsOfBook:
        return AsOfBook(self._observations, self._universe, (*self._labels, label))

    def materialize(
        self,
        decision_time: TimePoint,
        *,
        required_record_ids: Iterable[str] = (),
        required_target_ids: Iterable[str] = (),
    ) -> AsOfSnapshot:
        """Materialize only facts available at ``decision_time``.

        ``required_record_ids`` and ``required_target_ids`` turn an absent or
        future fact into an explicit error.  This is useful at a model boundary
        where silently dropping a missing feature or label would corrupt a
        backtest.
        """

        decision = _time(decision_time, "decision_time")
        available = tuple(
            event
            for event in self.ordered_events
            if _le(event.available_time, decision, "availability/decision time")
        )
        available_observations = _latest_by(
            (event for event in available if isinstance(event, MarketObservation)),
            lambda event: event.record_id,
        )
        available_labels_by_id = _latest_by(
            (event for event in available if isinstance(event, LabelObservation)),
            lambda event: event.label_id,
        )
        labels_by_target = _latest_by(
            available_labels_by_id.values(), lambda event: event.target_id
        )
        available_memberships = _latest_by(
            (event for event in available if isinstance(event, UniverseMembership)),
            lambda event: (event.instrument_id, event.start),
        )
        active_memberships = tuple(
            membership
            for membership in available_memberships.values()
            if membership.contains(decision)
        )
        _reject_overlapping_memberships(active_memberships, decision)
        snapshot = AsOfSnapshot(
            decision,
            tuple(sorted(available_observations.values(), key=lambda item: item.record_id)),
            tuple(sorted(active_memberships, key=lambda item: item.record_id)),
            tuple(sorted(labels_by_target.values(), key=lambda item: item.target_id)),
        )
        required_records = tuple(_text(item, "required record ID") for item in required_record_ids)
        for record_id in required_records:
            snapshot.observation(record_id)
        required_targets = tuple(_text(item, "required target ID") for item in required_target_ids)
        for target_id in required_targets:
            snapshot.label_for(target_id)
        return snapshot


def _check_unique_revisions(records: Iterable[Any], key_function: Any) -> None:
    seen: set[Any] = set()
    for record in records:
        key = key_function(record)
        if key in seen:
            raise DuplicateEventError(f"duplicate event ID/revision: {key!r}")
        seen.add(key)


def _reject_overlapping_memberships(
    memberships: Sequence[UniverseMembership], decision_time: TimePoint
) -> None:
    by_instrument: dict[str, list[UniverseMembership]] = {}
    for membership in memberships:
        by_instrument.setdefault(membership.instrument_id, []).append(membership)
    for instrument_id, values in by_instrument.items():
        for left_index, left in enumerate(values):
            for right in values[left_index + 1 :]:
                if left.contains(decision_time) and right.contains(decision_time):
                    raise AmbiguousEventError(
                        f"overlapping active universe memberships for {instrument_id!r}"
                    )


__all__ = [
    "AmbiguousEventError",
    "AsOfBook",
    "AsOfError",
    "AsOfSnapshot",
    "DuplicateEventError",
    "LABEL_PRECEDENCE",
    "MARKET_PRECEDENCE",
    "MissingLabelError",
    "LabelObservation",
    "MarketObservation",
    "PointInTimePanel",
    "UNIVERSE_PRECEDENCE",
    "UnavailableEventError",
    "UniverseMembership",
    "canonical_event_order",
]
