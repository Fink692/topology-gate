"""Immutable configuration and safety-boundary validation.

The component constructors predate this integration boundary and retain their
own validation.  ``ModelConfig`` is the boundary used for a reproducible run:
it applies conservative size and magnitude limits before a caller hands the
configuration to an estimator or detector, and it emits a JSON-safe identity.

The limits here are intentionally practical guardrails, not a proof that an
arbitrary deployment is safe for every workload.  A service should still set
process-level CPU, memory, and wall-clock limits around untrusted extensions.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from numbers import Integral
from typing import Any

from .rls import MAX_RLS_FEATURES as DENSE_MAX_RLS_FEATURES
from .rls import RLSConfig
from .topology import TopologyConfig


class SafetyBoundaryError(ValueError):
    """Base class for input failures at a safety boundary."""


class ConfigurationError(SafetyBoundaryError):
    """The supplied configuration is invalid, non-finite, or too large."""


class DataValidationError(SafetyBoundaryError):
    """Numeric data failed a finite-value or shape/resource check."""


class ResourceLimitError(SafetyBoundaryError):
    """A caller-controlled size exceeds a documented practical bound."""


class NumericalSafetyError(SafetyBoundaryError):
    """A numeric value is finite in isolation but unsafe at this boundary."""


# These limits are deliberately finite and public so a service can align its
# request validation and capacity planning with the library's defaults.
MAX_RLS_FEATURES = DENSE_MAX_RLS_FEATURES
MAX_TOPOLOGY_EMBEDDING_DIM = 64
MAX_TOPOLOGY_CLOUD_WINDOW = 1_024
MAX_TOPOLOGY_POINT_STRIDE = 1_024
MAX_TOPOLOGY_GRAPH_NEIGHBORS = 512
MAX_TOPOLOGY_EIGENVALUES = 512
MAX_TOPOLOGY_CALIBRATION_WINDOW = 16_384
MAX_TOPOLOGY_MATRIX_ELEMENTS = 1_000_000
MAX_TOPOLOGY_STREAM_OBSERVATIONS = 2_048
MAX_CONFIG_ABS_FLOAT = 1.0e100


def _finite_float(value: Any, field_name: str) -> float:
    if isinstance(value, (bool, str, bytes, bytearray)):
        raise ConfigurationError(f"{field_name} must be a finite real number")
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ConfigurationError(f"{field_name} must be a finite real number") from exc
    if not math.isfinite(converted):
        raise ConfigurationError(f"{field_name} must be finite")
    if abs(converted) > MAX_CONFIG_ABS_FLOAT:
        raise ResourceLimitError(
            f"{field_name} magnitude exceeds the configured limit "
            f"({MAX_CONFIG_ABS_FLOAT:g})"
        )
    return converted


def _bounded_int(value: Any, field_name: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ConfigurationError(f"{field_name} must be an integer")
    converted = int(value)
    if converted < 1:
        raise ConfigurationError(f"{field_name} must be positive")
    if converted > maximum:
        raise ResourceLimitError(
            f"{field_name}={converted} exceeds the practical limit {maximum}"
        )
    return converted


def _checked_product(left: int, right: int, field_name: str, maximum: int) -> int:
    # The comparison happens before any downstream allocation or array copy.
    if left > maximum // max(right, 1):
        raise ResourceLimitError(
            f"{field_name}={left}x{right} exceeds the practical element limit {maximum}"
        )
    return left * right


def _callable_identity(value: Any) -> str:
    """Return a stable, non-secret identity without serializing callable state."""

    target = value if inspect.isfunction(value) or inspect.ismethod(value) else type(value)
    module = getattr(target, "__module__", "unknown")
    qualname = getattr(target, "__qualname__", getattr(target, "__name__", type(value).__name__))
    return f"{module}:{qualname}"


def _json_safe_config(value: Any) -> Any:
    """Convert known config values to JSON-safe primitives without ``repr`` leaks."""

    if callable(value):
        return {"callable": _callable_identity(value)}
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ConfigurationError("configuration contains a non-finite float")
        if abs(value) > MAX_CONFIG_ABS_FLOAT:
            raise ResourceLimitError("configuration float exceeds the practical magnitude limit")
        return value
    if is_dataclass(value):
        return {
            field.name: _json_safe_config(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, (tuple, list)):
        if len(value) > 4_096:
            raise ResourceLimitError("configuration sequence exceeds the practical limit 4096")
        return [_json_safe_config(item) for item in value]
    if isinstance(value, Mapping):
        if len(value) > 4_096:
            raise ResourceLimitError("configuration mapping exceeds the practical limit 4096")
        return {
            str(key): _json_safe_config(item)
            for key, item in value.items()
        }
    raise ConfigurationError(
        f"configuration field of type {type(value).__name__} is not JSON-safe"
    )


def _validate_rls_config(config: RLSConfig) -> None:
    _bounded_int(config.n_features, "rls.n_features", MAX_RLS_FEATURES)
    ridge = _finite_float(config.ridge, "rls.ridge")
    if ridge <= 0.0:
        raise ConfigurationError("rls.ridge must be positive")
    if not math.isfinite(1.0 / ridge):
        raise NumericalSafetyError("rls.ridge is too small for a finite initial covariance")
    lambda_min = _finite_float(config.lambda_min, "rls.lambda_min")
    lambda_max = _finite_float(config.lambda_max, "rls.lambda_max")
    forgetting = _finite_float(config.forgetting_factor, "rls.forgetting_factor")
    if not 0.0 < lambda_min <= lambda_max <= 1.0:
        raise ConfigurationError("rls lambda bounds must satisfy 0 < min <= max <= 1")
    if not lambda_min <= forgetting <= lambda_max:
        raise ConfigurationError("rls.forgetting_factor must lie within lambda bounds")


def _validate_topology_config(config: TopologyConfig) -> None:
    embedding = _bounded_int(
        config.embedding_dim, "topology.embedding_dim", MAX_TOPOLOGY_EMBEDDING_DIM
    )
    cloud = _bounded_int(
        config.cloud_window, "topology.cloud_window", MAX_TOPOLOGY_CLOUD_WINDOW
    )
    _bounded_int(config.point_stride, "topology.point_stride", MAX_TOPOLOGY_POINT_STRIDE)
    neighbors = _bounded_int(
        config.graph_neighbors, "topology.graph_neighbors", MAX_TOPOLOGY_GRAPH_NEIGHBORS
    )
    eigenvalues = _bounded_int(
        config.n_eigenvalues, "topology.n_eigenvalues", MAX_TOPOLOGY_EIGENVALUES
    )
    calibration = _bounded_int(
        config.calibration_window,
        "topology.calibration_window",
        MAX_TOPOLOGY_CALIBRATION_WINDOW,
    )
    _bounded_int(
        config.max_stream_observations,
        "topology.max_stream_observations",
        MAX_TOPOLOGY_STREAM_OBSERVATIONS,
    )
    if neighbors >= cloud:
        raise ConfigurationError("topology.graph_neighbors must be smaller than cloud_window")
    if eigenvalues > cloud:
        raise ConfigurationError("topology.n_eigenvalues cannot exceed cloud_window")
    if config.min_points is not None:
        min_points = _bounded_int(config.min_points, "topology.min_points", cloud)
        if min_points > cloud:
            raise ConfigurationError("topology.min_points cannot exceed cloud_window")
    calibration_min = _bounded_int(
        config.calibration_min_periods,
        "topology.calibration_min_periods",
        calibration,
    )
    if calibration_min > calibration:
        raise ConfigurationError(
            "topology.calibration_min_periods cannot exceed calibration_window"
        )
    _checked_product(embedding, cloud, "topology embedding/cloud", MAX_TOPOLOGY_MATRIX_ELEMENTS)

    float_fields = (
        "threshold",
        "drift",
        "decay",
        "scale_floor",
        "z_clip",
        "whitening_ridge",
        "forgetting_lambda_min",
        "forgetting_lambda_max",
        "forgetting_sensitivity",
    )
    for field_name in float_fields:
        _finite_float(getattr(config, field_name), f"topology.{field_name}")
    if not 0.0 < config.decay <= 1.0:
        raise ConfigurationError("topology.decay must satisfy 0 < decay <= 1")
    if not 0.0 < config.forgetting_lambda_min <= config.forgetting_lambda_max <= 1.0:
        raise ConfigurationError("topology forgetting bounds must satisfy 0 < min <= max <= 1")
    if not isinstance(config.input_kind, str) or config.input_kind.lower() not in {"returns", "prices"}:
        raise ConfigurationError("topology.input_kind must be 'returns' or 'prices'")
    backend = config.persistent_laplacian_backend
    if backend is not None and not callable(backend):
        raise ConfigurationError("topology.persistent_laplacian_backend must be callable")


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Top-level configuration fingerprint for reproducible runs.

    This validation is the package integration boundary.  Direct use of the
    lower-level component constructors remains a trusted-code boundary and
    should be protected by the caller's own request/resource limits.
    """

    topology: TopologyConfig = TopologyConfig()
    rls: RLSConfig = RLSConfig(n_features=1)

    def __post_init__(self) -> None:
        if not isinstance(self.topology, TopologyConfig):
            raise ConfigurationError("topology must be a TopologyConfig instance")
        if not isinstance(self.rls, RLSConfig):
            raise ConfigurationError("rls must be an RLSConfig instance")
        _validate_topology_config(self.topology)
        _validate_rls_config(self.rls)
        if self.rls.lambda_min > self.topology.forgetting_lambda_min:
            raise ConfigurationError("RLS lambda_min must cover detector output range")
        if self.rls.lambda_max < self.topology.forgetting_lambda_max:
            raise ConfigurationError("RLS lambda_max must cover detector output range")

    def to_dict(self) -> dict[str, Any]:
        """Return canonical, JSON-safe configuration metadata.

        Callable backends are represented by module/qualified-name identity;
        callable closure state is intentionally never persisted or fingerprinted.
        Such a backend remains a trusted extension and is not fully restorable
        from this metadata alone.
        """

        result = _json_safe_config({"topology": self.topology, "rls": self.rls})
        if not isinstance(result, dict):  # pragma: no cover - defensive invariant
            raise ConfigurationError("configuration canonicalization produced a non-mapping")
        return result

    def fingerprint(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


__all__ = [
    "ConfigurationError",
    "DataValidationError",
    "MAX_CONFIG_ABS_FLOAT",
    "MAX_RLS_FEATURES",
    "MAX_TOPOLOGY_CALIBRATION_WINDOW",
    "MAX_TOPOLOGY_CLOUD_WINDOW",
    "MAX_TOPOLOGY_EMBEDDING_DIM",
    "MAX_TOPOLOGY_EIGENVALUES",
    "MAX_TOPOLOGY_GRAPH_NEIGHBORS",
    "MAX_TOPOLOGY_MATRIX_ELEMENTS",
    "MAX_TOPOLOGY_POINT_STRIDE",
    "MAX_TOPOLOGY_STREAM_OBSERVATIONS",
    "ModelConfig",
    "NumericalSafetyError",
    "ResourceLimitError",
    "SafetyBoundaryError",
]
