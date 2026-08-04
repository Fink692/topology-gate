"""Immutable, reproducible identity contracts for research runs.

The module intentionally uses only the Python standard library.  A manifest is
not a configuration loader: callers must provide explicit identities for every
input that can change the meaning of a run.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from hashlib import sha256
from typing import Any, Final, TypeAlias

MANIFEST_SCHEMA: Final[str] = "topology-gate.run-manifest"
MANIFEST_VERSION: Final[int] = 1
_MISSING = object()

JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


@dataclass(frozen=True, slots=True)
class _FrozenArray:
    values: tuple["_FrozenJsonValue", ...]


@dataclass(frozen=True, slots=True)
class _FrozenObject:
    items: tuple[tuple[str, "_FrozenJsonValue"], ...]


_FrozenJsonValue: TypeAlias = JsonScalar | _FrozenArray | _FrozenObject


class ManifestValidationError(ValueError):
    """Raised when a run identity cannot be represented safely as JSON."""


def _path_text(path: tuple[str, ...]) -> str:
    return ".".join(path) if path else "value"


def _freeze_json(value: Any, path: tuple[str, ...]) -> _FrozenJsonValue:
    """Validate and recursively freeze a JSON-shaped value."""

    location = _path_text(path)
    if value is None or type(value) in (bool, int, str):
        if isinstance(value, str) and not value.strip():
            raise ManifestValidationError(f"{location} must not be blank")
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ManifestValidationError(f"{location} must be finite")
        return value
    if isinstance(value, Mapping):
        frozen_items: list[tuple[str, _FrozenJsonValue]] = []
        for key, child in value.items():
            if type(key) is not str or not key.strip():
                raise ManifestValidationError(
                    f"{location} keys must be non-blank strings"
                )
            frozen_items.append((key, _freeze_json(child, (*path, key))))
        frozen_items.sort(key=lambda item: item[0])
        return _FrozenObject(tuple(frozen_items))
    if isinstance(value, list):
        return _FrozenArray(
            tuple(
                _freeze_json(child, (*path, str(index)))
                for index, child in enumerate(value)
            )
        )
    raise ManifestValidationError(
        f"{location} contains unsupported JSON type {type(value).__name__}"
    )


def _thaw_json(value: _FrozenJsonValue) -> JsonValue:
    """Return a fresh JSON-compatible value from an internal frozen value."""

    if isinstance(value, _FrozenObject):
        return {key: _thaw_json(child) for key, child in value.items}
    if isinstance(value, _FrozenArray):
        return [_thaw_json(child) for child in value.values]
    return value


def _required_identity(value: Any, name: str) -> _FrozenJsonValue:
    """Validate a required identity, including empty-container rejection."""

    if value is _MISSING:
        raise ManifestValidationError(f"{name} is required")
    frozen = _freeze_json(value, (name,))
    if value is None or (isinstance(value, (str, list, Mapping)) and not value):
        raise ManifestValidationError(f"{name} is required and must not be empty")
    return frozen


@dataclass(frozen=True, slots=True)
class RunSpec:
    """Required point-in-time identities that define one research run.

    Identity values may be strings, numbers, booleans, or nested JSON objects
    and arrays.  They are recursively validated and frozen at construction;
    use :meth:`to_dict` when a JSON-shaped view is needed.
    """

    run_id: Any = _MISSING
    input_vintage_id: Any = _MISSING
    universe_id: Any = _MISSING
    config_id: Any = _MISSING
    backend_id: Any = _MISSING
    dependency_id: Any = _MISSING
    seed_id: Any = _MISSING
    thread_id: Any = _MISSING

    def __post_init__(self) -> None:
        for name in (
            "run_id",
            "input_vintage_id",
            "universe_id",
            "config_id",
            "backend_id",
            "dependency_id",
            "seed_id",
            "thread_id",
        ):
            object.__setattr__(
                self,
                name,
                _required_identity(getattr(self, name), name),
            )

    def to_dict(self) -> dict[str, JsonValue]:
        """Return a fresh JSON-compatible mapping of the required identities."""

        return {
            "run_id": _thaw_json(self.run_id),
            "input_vintage_id": _thaw_json(self.input_vintage_id),
            "universe_id": _thaw_json(self.universe_id),
            "config_id": _thaw_json(self.config_id),
            "backend_id": _thaw_json(self.backend_id),
            "dependency_id": _thaw_json(self.dependency_id),
            "seed_id": _thaw_json(self.seed_id),
            "thread_id": _thaw_json(self.thread_id),
        }

    def to_json(self) -> str:
        """Serialize the specification using deterministic canonical JSON."""

        return _canonical_json(self.to_dict())

    @property
    def digest(self) -> str:
        """Return the lowercase SHA-256 digest of the canonical specification."""

        return sha256(self.to_json().encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class StudyWindow:
    """Half-open, non-empty index window in a predeclared study timeline."""

    name: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name.strip():
            raise ManifestValidationError("study window name must be non-blank")
        if isinstance(self.start, bool) or not isinstance(self.start, int) or self.start < 0:
            raise ManifestValidationError("study window start must be a non-negative integer")
        if isinstance(self.end, bool) or not isinstance(self.end, int) or self.end <= self.start:
            raise ManifestValidationError("study window end must exceed start")

    def to_dict(self) -> dict[str, JsonValue]:
        return {"name": self.name, "start": self.start, "end": self.end}


@dataclass(frozen=True, slots=True)
class StudySpec:
    """Pre-registered data and split identity for a walk-forward study.

    Window boundaries are timeline indices supplied by the caller's
    point-in-time event source.  The manifest enforces ordering and an optional
    purge/embargo gap, but it does not claim that those indices are market
    timestamps or that the source is survivorship-free.
    """

    run_spec: RunSpec
    feature_schema_id: str
    label_spec_id: str
    economic_spec_id: str
    calibration_window: StudyWindow
    tuning_window: StudyWindow
    validation_window: StudyWindow
    holdout_window: StudyWindow
    embargo_steps: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.run_spec, RunSpec):
            raise ManifestValidationError("study run_spec must be a RunSpec")
        for name in ("feature_schema_id", "label_spec_id", "economic_spec_id"):
            value = getattr(self, name)
            if type(value) is not str or not value.strip():
                raise ManifestValidationError(f"{name} must be a non-blank string")
        windows = (
            self.calibration_window,
            self.tuning_window,
            self.validation_window,
            self.holdout_window,
        )
        if not all(isinstance(window, StudyWindow) for window in windows):
            raise ManifestValidationError("study windows must be StudyWindow values")
        names = [window.name for window in windows]
        if len(set(names)) != len(names):
            raise ManifestValidationError("study window names must be unique")
        if (
            isinstance(self.embargo_steps, bool)
            or not isinstance(self.embargo_steps, int)
            or self.embargo_steps < 0
        ):
            raise ManifestValidationError("embargo_steps must be a non-negative integer")
        for previous, current in zip(windows, windows[1:]):
            if previous.end + self.embargo_steps > current.start:
                raise ManifestValidationError(
                    "study windows overlap or violate the embargo boundary"
                )

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "run_spec": self.run_spec.to_dict(),
            "feature_schema_id": self.feature_schema_id,
            "label_spec_id": self.label_spec_id,
            "economic_spec_id": self.economic_spec_id,
            "calibration_window": self.calibration_window.to_dict(),
            "tuning_window": self.tuning_window.to_dict(),
            "validation_window": self.validation_window.to_dict(),
            "holdout_window": self.holdout_window.to_dict(),
            "embargo_steps": self.embargo_steps,
        }

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    @property
    def digest(self) -> str:
        return sha256(self.to_json().encode("utf-8")).hexdigest()


STUDY_SCHEMA: Final[str] = "topology-gate.study-manifest"
STUDY_VERSION: Final[int] = 1


@dataclass(frozen=True, slots=True)
class StudyManifest:
    """Immutable study identity with an auditable sealed-holdout transition."""

    spec: StudySpec
    metadata: Any = field(default_factory=dict)
    holdout_status: str = "sealed"
    holdout_release_id: str | None = None
    schema: str = STUDY_SCHEMA
    version: int = STUDY_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.spec, StudySpec):
            raise ManifestValidationError("study manifest spec must be a StudySpec")
        if type(self.schema) is not str or self.schema != STUDY_SCHEMA:
            raise ManifestValidationError(f"schema must be exactly {STUDY_SCHEMA!r}")
        if type(self.version) is not int or self.version != STUDY_VERSION:
            raise ManifestValidationError(f"version must be exactly {STUDY_VERSION}")
        if not isinstance(self.metadata, Mapping):
            raise ManifestValidationError("metadata must be a JSON object")
        status = self.holdout_status
        if status not in {"sealed", "opened"}:
            raise ManifestValidationError("holdout_status must be sealed or opened")
        if status == "sealed" and self.holdout_release_id is not None:
            raise ManifestValidationError("sealed holdouts cannot have a release ID")
        if status == "opened":
            if type(self.holdout_release_id) is not str or not self.holdout_release_id.strip():
                raise ManifestValidationError(
                    "opened holdouts require a non-blank release ID"
                )
        object.__setattr__(self, "metadata", _freeze_json(self.metadata, ("metadata",)))

    @property
    def digest(self) -> str:
        return sha256(self.to_json().encode("utf-8")).hexdigest()

    @property
    def holdout_is_sealed(self) -> bool:
        return self.holdout_status == "sealed"

    def require_holdout_sealed(self) -> None:
        """Raise if a caller attempts a pre-release read after opening holdout."""

        if not self.holdout_is_sealed:
            raise ManifestValidationError("study holdout is already opened")

    def open_holdout(self, release_id: str) -> "StudyManifest":
        """Return a new manifest recording the explicit holdout release event."""

        self.require_holdout_sealed()
        if type(release_id) is not str or not release_id.strip():
            raise ManifestValidationError("release ID must be a non-blank string")
        return replace(
            self,
            metadata=_thaw_json(self.metadata),
            holdout_status="opened",
            holdout_release_id=release_id,
        )

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "schema": self.schema,
            "version": self.version,
            "spec": self.spec.to_dict(),
            "metadata": _thaw_json(self.metadata),
            "holdout_status": self.holdout_status,
            "holdout_release_id": self.holdout_release_id,
        }

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())


@dataclass(frozen=True, slots=True)
class RunManifest:
    """Immutable, canonically serializable identity for a complete run."""

    spec: Any = _MISSING
    metadata: Any = field(default_factory=dict)
    schema: str = MANIFEST_SCHEMA
    version: int = MANIFEST_VERSION

    def __post_init__(self) -> None:
        if self.spec is _MISSING or not isinstance(self.spec, RunSpec):
            raise ManifestValidationError("spec must be a RunSpec")
        if type(self.schema) is not str or self.schema != MANIFEST_SCHEMA:
            raise ManifestValidationError(
                f"schema must be exactly {MANIFEST_SCHEMA!r}"
            )
        if type(self.version) is not int or self.version != MANIFEST_VERSION:
            raise ManifestValidationError(
                f"version must be exactly {MANIFEST_VERSION}"
            )
        if not isinstance(self.metadata, Mapping):
            raise ManifestValidationError("metadata must be a JSON object")
        object.__setattr__(self, "metadata", _freeze_json(self.metadata, ("metadata",)))

    def to_dict(self) -> dict[str, JsonValue]:
        """Return a fresh JSON-compatible manifest mapping."""

        return {
            "schema": self.schema,
            "version": self.version,
            "spec": self.spec.to_dict(),
            "metadata": _thaw_json(self.metadata),
        }

    def to_json(self) -> str:
        """Serialize the manifest using the digest's canonical JSON encoding."""

        return _canonical_json(self.to_dict())

    @property
    def digest(self) -> str:
        """Return the lowercase SHA-256 digest of the canonical manifest JSON."""

        return sha256(self.to_json().encode("utf-8")).hexdigest()


def _canonical_json(value: JsonValue) -> str:
    """Encode a validated JSON value deterministically."""

    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


__all__ = [
    "MANIFEST_SCHEMA",
    "MANIFEST_VERSION",
    "JsonValue",
    "ManifestValidationError",
    "RunManifest",
    "RunSpec",
    "STUDY_SCHEMA",
    "STUDY_VERSION",
    "StudyManifest",
    "StudySpec",
    "StudyWindow",
]
