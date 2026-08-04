"""Authenticated, versioned checkpoint envelopes for causal replay.

The envelope is deliberately independent of NumPy and of any particular
worker implementation. Components contribute JSON-safe state dictionaries;
the envelope records the configuration/backend identities needed to reject a
checkpoint produced by a different run contract. The HMAC key is supplied by
the caller and is never serialized.

This module does not execute arbitrary serialized Python objects. Callable
forgetting-factor, eta, and backend policies must be supplied again by the
trusted caller when restoring their component state.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

CHECKPOINT_VERSION = 1
CHECKPOINT_SCHEMA = "topology_gate.checkpoint"
MAX_CHECKPOINT_BYTES = 50 * 1024 * 1024
MAX_CHECKPOINT_ITEMS = 1_000_000
MAX_CHECKPOINT_DEPTH = 32


class CheckpointError(ValueError):
    """Base class for checkpoint validation failures."""


class CheckpointIntegrityError(CheckpointError):
    """The envelope was truncated or modified after it was sealed."""


class CheckpointCompatibilityError(CheckpointError):
    """The envelope does not match the requested run contract."""


def _canonical(value: Any, *, path: str = "$", depth: int = 0) -> Any:
    if depth > MAX_CHECKPOINT_DEPTH:
        raise CheckpointError(f"checkpoint state is nested beyond {MAX_CHECKPOINT_DEPTH} levels")
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CheckpointError(f"checkpoint value at {path} is non-finite")
        return value
    if isinstance(value, Mapping):
        if len(value) > MAX_CHECKPOINT_ITEMS:
            raise CheckpointError(f"checkpoint mapping at {path} exceeds the item limit")
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise CheckpointError(f"checkpoint key at {path} must be a string")
            result[key] = _canonical(item, path=f"{path}.{key}", depth=depth + 1)
        return result
    if isinstance(value, (tuple, list)):
        if len(value) > MAX_CHECKPOINT_ITEMS:
            raise CheckpointError(f"checkpoint sequence at {path} exceeds the item limit")
        return [
            _canonical(item, path=f"{path}[{index}]", depth=depth + 1)
            for index, item in enumerate(value)
        ]
    raise CheckpointError(
        f"checkpoint value at {path} has unsupported type {type(value).__name__}"
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _canonical(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _digest(payload: Mapping[str, Any], key: bytes | None) -> tuple[str, str]:
    encoded = _canonical_json(payload).encode("utf-8")
    if key is None:
        return "sha256", hashlib.sha256(encoded).hexdigest()
    if not isinstance(key, bytes) or len(key) < 16:
        raise CheckpointError("hmac_key must be at least 16 bytes")
    return "hmac-sha256", hmac.new(key, encoded, hashlib.sha256).hexdigest()


def _require_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise CheckpointError(f"{name} must be a non-empty string")
    return value


@dataclass(frozen=True, slots=True)
class CheckpointEnvelope:
    """Immutable checkpoint payload plus its integrity proof."""

    package_version: str
    config_fingerprint: str
    backend_identity: str
    dependency_fingerprint: str
    learner_state: Mapping[str, Any] | None = None
    detector_state: Mapping[str, Any] | None = None
    online_state: Mapping[str, Any] | None = None
    promotion_state: Mapping[str, Any] | None = None
    rng_state: Mapping[str, Any] | None = None
    metadata: Mapping[str, Any] | None = None
    integrity_algorithm: str = "sha256"
    integrity: str = ""
    version: int = CHECKPOINT_VERSION
    schema: str = CHECKPOINT_SCHEMA

    def payload_dict(self) -> dict[str, Any]:
        """Return the data covered by the integrity proof."""

        return _canonical(
            {
                "version": self.version,
                "schema": self.schema,
                "package_version": self.package_version,
                "config_fingerprint": self.config_fingerprint,
                "backend_identity": self.backend_identity,
                "dependency_fingerprint": self.dependency_fingerprint,
                "learner_state": self.learner_state,
                "detector_state": self.detector_state,
                "online_state": self.online_state,
                "promotion_state": self.promotion_state,
                "rng_state": self.rng_state,
                "metadata": self.metadata,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        payload = self.payload_dict()
        payload["integrity_algorithm"] = self.integrity_algorithm
        payload["integrity"] = self.integrity
        return payload

    def to_json(self) -> str:
        encoded = _canonical_json(self.to_dict())
        if len(encoded.encode("utf-8")) > MAX_CHECKPOINT_BYTES:
            raise CheckpointError("checkpoint exceeds the serialized size limit")
        return encoded

    @classmethod
    def create(
        cls,
        *,
        package_version: str,
        config_fingerprint: str,
        backend_identity: str,
        dependency_fingerprint: str,
        learner_state: Mapping[str, Any] | None = None,
        detector_state: Mapping[str, Any] | None = None,
        online_state: Mapping[str, Any] | None = None,
        promotion_state: Mapping[str, Any] | None = None,
        rng_state: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        hmac_key: bytes | None = None,
        allow_untrusted: bool = False,
    ) -> "CheckpointEnvelope":
        if hmac_key is None and not allow_untrusted:
            raise CheckpointIntegrityError(
                "hmac_key is required for checkpoint creation; set "
                "allow_untrusted=True only for a trusted local artifact"
            )
        package = _require_text(package_version, "package_version")
        config = _require_text(config_fingerprint, "config_fingerprint")
        backend = _require_text(backend_identity, "backend_identity")
        dependencies = _require_text(dependency_fingerprint, "dependency_fingerprint")
        candidate = cls(
            package_version=package,
            config_fingerprint=config,
            backend_identity=backend,
            dependency_fingerprint=dependencies,
            learner_state=learner_state,
            detector_state=detector_state,
            online_state=online_state,
            promotion_state=promotion_state,
            rng_state=rng_state,
            metadata=metadata,
        )
        algorithm, proof = _digest(candidate.payload_dict(), hmac_key)
        return cls(
            package_version=candidate.package_version,
            config_fingerprint=candidate.config_fingerprint,
            backend_identity=candidate.backend_identity,
            dependency_fingerprint=candidate.dependency_fingerprint,
            learner_state=candidate.learner_state,
            detector_state=candidate.detector_state,
            online_state=candidate.online_state,
            promotion_state=candidate.promotion_state,
            rng_state=candidate.rng_state,
            metadata=candidate.metadata,
            integrity_algorithm=algorithm,
            integrity=proof,
        )

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
        *,
        hmac_key: bytes | None = None,
        expected_package_version: str | None = None,
        expected_config_fingerprint: str | None = None,
        expected_backend_identity: str | None = None,
        expected_dependency_fingerprint: str | None = None,
        allow_untrusted: bool = False,
    ) -> "CheckpointEnvelope":
        if not isinstance(value, Mapping):
            raise CheckpointError("checkpoint envelope must be a mapping")
        if value.get("version") != CHECKPOINT_VERSION or value.get("schema") != CHECKPOINT_SCHEMA:
            raise CheckpointCompatibilityError("unsupported checkpoint version or schema")
        algorithm = value.get("integrity_algorithm")
        proof = value.get("integrity")
        if algorithm not in {"sha256", "hmac-sha256"} or not isinstance(proof, str):
            raise CheckpointIntegrityError("checkpoint integrity fields are invalid")
        if algorithm == "sha256" and not allow_untrusted:
            raise CheckpointIntegrityError(
                "checkpoint requires HMAC authentication; set allow_untrusted=True "
                "only for a trusted local artifact"
            )
        if algorithm == "hmac-sha256" and hmac_key is None:
            raise CheckpointIntegrityError("checkpoint requires an HMAC key")
        payload = {key: value.get(key) for key in (
            "version", "schema", "package_version", "config_fingerprint",
            "backend_identity", "dependency_fingerprint", "learner_state",
            "detector_state", "online_state", "promotion_state", "rng_state", "metadata",
        )}
        expected_algorithm, expected_proof = _digest(payload, hmac_key if algorithm == "hmac-sha256" else None)
        if algorithm != expected_algorithm or not hmac.compare_digest(proof, expected_proof):
            raise CheckpointIntegrityError("checkpoint integrity verification failed")
        package = _require_text(payload.get("package_version"), "package_version")
        config = _require_text(payload.get("config_fingerprint"), "config_fingerprint")
        backend = _require_text(payload.get("backend_identity"), "backend_identity")
        dependencies = _require_text(payload.get("dependency_fingerprint"), "dependency_fingerprint")
        for expected, actual, name in (
            (expected_package_version, package, "package_version"),
            (expected_config_fingerprint, config, "config_fingerprint"),
            (expected_backend_identity, backend, "backend_identity"),
            (expected_dependency_fingerprint, dependencies, "dependency_fingerprint"),
        ):
            if expected is not None and expected != actual:
                raise CheckpointCompatibilityError(f"checkpoint {name} does not match the active run")
        return cls(
            package_version=package,
            config_fingerprint=config,
            backend_identity=backend,
            dependency_fingerprint=dependencies,
            learner_state=_canonical(payload.get("learner_state")),
            detector_state=_canonical(payload.get("detector_state")),
            online_state=_canonical(payload.get("online_state")),
            promotion_state=_canonical(payload.get("promotion_state")),
            rng_state=_canonical(payload.get("rng_state")),
            metadata=_canonical(payload.get("metadata")),
            integrity_algorithm=algorithm,
            integrity=proof,
        )

    @classmethod
    def from_json(cls, payload: str, **kwargs: Any) -> "CheckpointEnvelope":
        if not isinstance(payload, str) or len(payload.encode("utf-8")) > MAX_CHECKPOINT_BYTES:
            raise CheckpointError("checkpoint JSON is invalid or too large")
        try:
            value = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise CheckpointError("checkpoint JSON is invalid") from exc
        return cls.from_dict(value, **kwargs)

    def verify(self, hmac_key: bytes | None = None) -> None:
        expected_algorithm, expected_proof = _digest(self.payload_dict(), hmac_key if self.integrity_algorithm == "hmac-sha256" else None)
        if self.integrity_algorithm != expected_algorithm or not hmac.compare_digest(self.integrity, expected_proof):
            raise CheckpointIntegrityError("checkpoint integrity verification failed")


def checkpoint_from_components(
    *,
    package_version: str,
    config_fingerprint: str,
    backend_identity: str,
    dependency_fingerprint: str,
    learner: Any | None = None,
    detector: Any | None = None,
    online_state: Mapping[str, Any] | Any | None = None,
    promotion: Any | None = None,
    rng_state: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    hmac_key: bytes | None = None,
    allow_untrusted: bool = False,
) -> CheckpointEnvelope:
    """Capture component state through explicit state hooks only."""

    def state_of(component: Any, *methods: str) -> Mapping[str, Any] | None:
        if component is None:
            return None
        for method in methods:
            callback = getattr(component, method, None)
            if callable(callback):
                state = callback()
                if not isinstance(state, Mapping):
                    raise CheckpointError(f"{method} must return a mapping")
                return state
        raise CheckpointError(f"component {type(component).__name__} has no supported state hook")

    if online_state is not None and not isinstance(online_state, Mapping):
        callback = getattr(online_state, "state_dict", None)
        if not callable(callback):
            raise CheckpointError("online_state must be a mapping or expose state_dict")
        online_state = callback()
    return CheckpointEnvelope.create(
        package_version=package_version,
        config_fingerprint=config_fingerprint,
        backend_identity=backend_identity,
        dependency_fingerprint=dependency_fingerprint,
        learner_state=state_of(learner, "state_dict", "get_state"),
        detector_state=state_of(detector, "stream_state_dict", "state_dict"),
        online_state=online_state,
        promotion_state=state_of(promotion, "state_dict"),
        rng_state=rng_state,
        metadata=metadata,
        hmac_key=hmac_key,
        allow_untrusted=allow_untrusted,
    )


def restore_component_states(
    envelope: CheckpointEnvelope,
    *,
    learner: Any | None = None,
    detector: Any | None = None,
    promotion: Any | None = None,
    forgetting_factor: Any | None = None,
    eta: Any | None = None,
    hmac_key: bytes | None = None,
    expected_package_version: str | None = None,
    expected_config_fingerprint: str | None = None,
    expected_backend_identity: str | None = None,
    expected_dependency_fingerprint: str | None = None,
    allow_untrusted: bool = False,
) -> dict[str, Any]:
    """Validate component restores before mutating any supplied object.

    The returned candidates are detached objects where the implementation
    exposes a class-level ``from_state_dict``. Callers may use the mapping for
    a coordinated swap, or pass the original objects again after validation to
    invoke their transactional ``load_*`` hooks.
    """

    if not allow_untrusted and envelope.integrity_algorithm != "hmac-sha256":
        raise CheckpointIntegrityError(
            "component restore requires an HMAC-authenticated checkpoint; "
            "set allow_untrusted=True only for a trusted local artifact"
        )
    if not all(
        value is not None
        for value in (
            expected_package_version,
            expected_config_fingerprint,
            expected_backend_identity,
            expected_dependency_fingerprint,
        )
    ):
        raise CheckpointCompatibilityError(
            "component restore requires package, configuration, backend, and "
            "dependency identities"
        )
    for expected, actual, name in (
        (expected_package_version, envelope.package_version, "package_version"),
        (expected_config_fingerprint, envelope.config_fingerprint, "config_fingerprint"),
        (expected_backend_identity, envelope.backend_identity, "backend_identity"),
        (
            expected_dependency_fingerprint,
            envelope.dependency_fingerprint,
            "dependency_fingerprint",
        ),
    ):
        if expected != actual:
            raise CheckpointCompatibilityError(
                f"checkpoint {name} does not match the active run"
            )
    envelope.verify(hmac_key)
    restored: dict[str, Any] = {
        "online_state": envelope.online_state,
        "rng_state": envelope.rng_state,
    }
    if learner is not None and envelope.learner_state is not None:
        factory = getattr(type(learner), "from_state_dict", None)
        if not callable(factory):
            raise CheckpointError("learner does not support detached state restore")
        restored["learner"] = factory(envelope.learner_state, forgetting_factor=forgetting_factor)
    if detector is not None and envelope.detector_state is not None:
        validator = getattr(detector, "validate_stream_state_dict", None)
        if callable(validator):
            restored["detector_state"] = validator(envelope.detector_state)
        else:
            restored["detector_state"] = envelope.detector_state
    if promotion is not None and envelope.promotion_state is not None:
        factory = getattr(type(promotion), "from_state_dict", None)
        if not callable(factory):
            raise CheckpointError("promotion component does not support detached state restore")
        restored["promotion"] = factory(envelope.promotion_state, eta=eta)
    return restored


def save_checkpoint(
    envelope: CheckpointEnvelope,
    path: str | Path,
    *,
    approved_root: str | Path | None = None,
) -> Path:
    """Atomically write a checkpoint, optionally constrained to a directory."""

    destination = Path(path)
    if destination.exists() and destination.is_symlink():
        raise CheckpointError("checkpoint destination must not be a symlink")
    resolved = destination.resolve(strict=False)
    if approved_root is not None:
        root = Path(approved_root).resolve(strict=True)
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise CheckpointError("checkpoint path escapes the approved root") from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = envelope.to_json().encode("utf-8")
    if len(payload) > MAX_CHECKPOINT_BYTES:
        raise CheckpointError("checkpoint exceeds the serialized size limit")
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
        os.replace(temporary, destination)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)
    return destination


def load_checkpoint(path: str | Path, **kwargs: Any) -> CheckpointEnvelope:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise CheckpointError("checkpoint path must be a regular file")
    if source.stat().st_size > MAX_CHECKPOINT_BYTES:
        raise CheckpointError("checkpoint exceeds the serialized size limit")
    return CheckpointEnvelope.from_json(source.read_text(encoding="utf-8"), **kwargs)


__all__ = [
    "CHECKPOINT_SCHEMA",
    "CHECKPOINT_VERSION",
    "CheckpointCompatibilityError",
    "CheckpointEnvelope",
    "CheckpointError",
    "CheckpointIntegrityError",
    "MAX_CHECKPOINT_BYTES",
    "checkpoint_from_components",
    "load_checkpoint",
    "restore_component_states",
    "save_checkpoint",
]
