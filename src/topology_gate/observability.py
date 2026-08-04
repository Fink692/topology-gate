"""Secret-free structured audit logging and in-process metrics."""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

MAX_AUDIT_EVENT_BYTES = 64 * 1024
MAX_AUDIT_PAYLOAD_DEPTH = 16
_SECRET_KEY = re.compile(
    r"(?:pass(word|wd)?|secret|token|api[_-]?key|credential|authorization|"
    r"private[_-]?key|access[_-]?token|refresh[_-]?token)",
    re.IGNORECASE,
)


def _safe_payload(value: Any, *, key: str | None = None, depth: int = 0) -> Any:
    """Copy an audit payload while recursively redacting sensitive keys."""

    if depth > MAX_AUDIT_PAYLOAD_DEPTH:
        raise ValueError("audit payload exceeds the nesting limit")
    if key is not None and _SECRET_KEY.search(key):
        return "[REDACTED]"
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("audit payload contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        if len(value) > 4096:
            raise ValueError("audit payload mapping exceeds 4096 items")
        return {
            str(item_key): _safe_payload(item, key=str(item_key), depth=depth + 1)
            for item_key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        if len(value) > 4096:
            raise ValueError("audit payload sequence exceeds 4096 items")
        return [_safe_payload(item, depth=depth + 1) for item in value]
    raise TypeError(f"audit payload value {type(value).__name__} is not JSON-safe")


@dataclass(frozen=True, slots=True)
class AuditEvent:
    event_type: str
    step: int
    payload: Mapping[str, Any]
    timestamp: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = _safe_payload(self.payload)
        if not isinstance(payload, dict):  # pragma: no cover - Mapping invariant
            raise TypeError("audit payload must be a mapping")
        return {
            "event_type": self.event_type,
            "step": self.step,
            "payload": payload,
            "timestamp": self.timestamp,
        }


class AuditLog:
    """Bounded in-memory event log with explicit JSONL export."""

    def __init__(
        self,
        *,
        max_events: int = 10_000,
        max_event_bytes: int = MAX_AUDIT_EVENT_BYTES,
        approved_root: str | Path | None = None,
    ) -> None:
        if max_events < 1:
            raise ValueError("max_events must be positive")
        if isinstance(max_event_bytes, bool) or max_event_bytes < 256:
            raise ValueError("max_event_bytes must be at least 256")
        self.max_events = max_events
        self.max_event_bytes = max_event_bytes
        self.approved_root = None if approved_root is None else Path(approved_root).resolve(strict=True)
        self._events: list[AuditEvent] = []
        self._lock = threading.RLock()

    def append(self, event: AuditEvent) -> None:
        if event.step < 0 or not event.event_type:
            raise ValueError("audit event must have a non-negative step and event type")
        sanitized = AuditEvent(
            event_type=event.event_type,
            step=event.step,
            payload=event.to_dict()["payload"],
            timestamp=event.timestamp,
        )
        encoded = json.dumps(sanitized.to_dict(), sort_keys=True, allow_nan=False).encode("utf-8")
        if len(encoded) > self.max_event_bytes:
            raise ValueError("audit event exceeds the configured size limit")
        with self._lock:
            self._events.append(sanitized)
            if len(self._events) > self.max_events:
                del self._events[: len(self._events) - self.max_events]

    def events(self) -> tuple[AuditEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def to_jsonl(self, path: str | Path, *, approved_root: str | Path | None = None) -> None:
        destination = Path(path)
        if destination.exists() and (destination.is_dir() or destination.is_symlink()):
            raise IsADirectoryError(destination)
        root_value = self.approved_root if approved_root is None else Path(approved_root).resolve(strict=True)
        resolved = destination.resolve(strict=False)
        if root_value is not None:
            try:
                resolved.relative_to(root_value)
            except ValueError as exc:
                raise ValueError("audit destination escapes the approved root") from exc
        destination.parent.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps(event.to_dict(), sort_keys=True, allow_nan=False) for event in self.events()]
        payload = ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")
        if len(payload) > self.max_events * self.max_event_bytes:
            raise ValueError("audit export exceeds the configured size limit")
        temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", prefix=f".{destination.name}.", suffix=".tmp",
                dir=destination.parent, delete=False,
            ) as handle:
                temporary = handle.name
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.chmod(temporary, 0o600)
            except OSError:
                pass
            os.replace(temporary, destination)
        finally:
            if temporary is not None and os.path.exists(temporary):
                os.unlink(temporary)


class MetricsRegistry:
    """Thread-safe counters and elapsed-time samples."""

    def __init__(self) -> None:
        self._counters: dict[str, int] = {}
        self._timings: dict[str, list[float]] = {}
        self._lock = threading.RLock()

    def increment(self, name: str, amount: int = 1) -> None:
        if not name or not isinstance(amount, int):
            raise ValueError("metric name must be non-empty and amount must be an integer")
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + amount

    @contextmanager
    def timer(self, name: str) -> Iterator[None]:
        if not name:
            raise ValueError("timer name must be non-empty")
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            with self._lock:
                self._timings.setdefault(name, []).append(elapsed)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "counters": dict(self._counters),
                "timings": {name: list(values) for name, values in self._timings.items()},
            }
