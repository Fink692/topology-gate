"""Rolling spectral market-geometry change detection.

The default implementation in this module is deliberately dependency-light.  A
series is converted into a causal delay-embedded point cloud, a deterministic
union k-nearest-neighbour graph is built on the robustly scaled cloud, and the
smallest eigenvalues of its normalized graph Laplacian are summarized together
with simple geometric features.

This is an approximation to topology, not persistent homology.  In particular,
the default ``knn_normalized_laplacian_approximation`` does *not* compute a
persistent Laplacian, Betti numbers, or a persistence diagram.  Code that has an
exact persistent-Laplacian implementation may be supplied through
``persistent_laplacian_backend``.  The recommended
``PersistentLaplacianBackend`` receives only the current, already robustly
scaled point cloud and the requested number of eigenvalues, returns a complete
finite evidence object, and carries its filtration/solver identity and vertex
budget.  The rolling compatibility seam consumes the selected spectrum.  This
keeps the approximation explicit without making a heavy dependency a
requirement for the base package.

All rolling operations are causal: the feature at index ``t`` uses observations
through ``t`` and calibration for ``t`` uses valid feature rows strictly before
``t``.  Non-finite input is rejected and all warm-up outputs are finite neutral
values accompanied by ``valid``/``calibrated`` masks; NaNs are never used as a
missing-data sentinel in the result.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass
from math import copysign, exp, isfinite, log, sqrt
from numbers import Integral, Real
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence, Tuple

_np: Any
try:  # NumPy is an optional accelerator, never a hard dependency.
    import numpy as _np
except ModuleNotFoundError as exc:  # pragma: no cover - explicit fallback test.
    if exc.name not in {"numpy", None}:
        raise
    _np = None


__all__ = [
    "CusumResult",
    "PointCloudFeatures",
    "RollingTopologyDetector",
    "SpectralSummary",
    "StreamingTopologyResult",
    "TopologyConfig",
    "TopologyResult",
    "WhiteningResult",
    "cusum_scores",
    "detect_topology_changes",
    "point_cloud_features",
    "robust_whiten",
    "rolling_point_cloud",
    "rolling_point_clouds",
    "spectral_summary",
]


_METHOD = "knn_normalized_laplacian_approximation"
_STREAM_SCHEMA = "topology_gate.topology.stream"
_STREAM_STATE_VERSION = 2
_DEFAULT_MAX_STREAM_OBSERVATIONS = 2048
_MAX_BATCH_OBSERVATIONS = 8192
_MAX_STREAM_OBSERVATIONS = _DEFAULT_MAX_STREAM_OBSERVATIONS
_MAX_EMBEDDING_DIM = 64
_MAX_CLOUD_WINDOW = 1024
_MAX_POINT_STRIDE = 1024
_MAX_GRAPH_NEIGHBORS = 512
_MAX_EIGENVALUES = 512
_MAX_CALIBRATION_WINDOW = 16_384
_MAX_BACKEND_EIGENVALUES = 4096
_MAX_OBSERVATION_DIMENSION = 256
_MAX_POINT_DIMENSION = 4096
_MAX_NORMALIZED_CLOUD_COORDINATE = 1_000_000.0
_TINY = 1.0e-12


def _validate_int(name: str, value: Any, minimum: int, maximum: int | None = None) -> int:
    """Validate an integer parameter without accepting booleans."""

    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be an integer <= {maximum}")
    return value


def _validate_finite(name: str, value: Any, *, minimum: Optional[float] = None,
                     maximum: Optional[float] = None,
                     strict_minimum: bool = False,
                     strict_maximum: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    if minimum is not None:
        bad = result <= minimum if strict_minimum else result < minimum
        if bad:
            relation = ">" if strict_minimum else ">="
            raise ValueError(f"{name} must be {relation} {minimum}")
    if maximum is not None:
        bad = result >= maximum if strict_maximum else result > maximum
        if bad:
            relation = "<" if strict_maximum else "<="
            raise ValueError(f"{name} must be {relation} {maximum}")
    return result


def _validate_backend(backend: Optional[Callable[..., Any]]) -> Optional[Callable[..., Any]]:
    if backend is not None and not callable(backend):
        raise ValueError("persistent_laplacian_backend must be callable or None")
    return backend


@dataclass(frozen=True)
class TopologyConfig:
    """Validated parameters for :class:`RollingTopologyDetector`.

    ``embedding_dim`` is the number of consecutive observations in each delay
    vector.  ``cloud_window`` is the number of recent delay vectors retained in
    a cloud, while ``point_stride`` subsamples their end times.  The CUSUM
    recursion is

    ``G_t = max(0, decay * G_(t-1) + innovation_t - drift)``.

    ``input_kind="prices"`` causally differences prices before embedding.  The
    first difference is zero and is naturally excluded by the warm-up mask.
    """

    embedding_dim: int = 3
    cloud_window: int = 64
    point_stride: int = 1
    graph_neighbors: int = 8
    n_eigenvalues: int = 4
    min_points: Optional[int] = None
    calibration_window: int = 128
    calibration_min_periods: int = 24
    threshold: float = 5.0
    drift: float = 2.0
    decay: float = 1.0
    scale_floor: float = 1.0e-8
    z_clip: float = 12.0
    whitening_ridge: float = 1.0e-6
    forgetting_lambda_min: float = 0.90
    forgetting_lambda_max: float = 0.995
    forgetting_sensitivity: float = 1.0
    input_kind: str = "returns"
    persistent_laplacian_backend: Optional[Callable[..., Any]] = None
    max_stream_observations: int = _DEFAULT_MAX_STREAM_OBSERVATIONS

    def __post_init__(self) -> None:
        embedding_dim = _validate_int("embedding_dim", self.embedding_dim, 1, _MAX_EMBEDDING_DIM)
        cloud_window = _validate_int("cloud_window", self.cloud_window, 2, _MAX_CLOUD_WINDOW)
        point_stride = _validate_int("point_stride", self.point_stride, 1, _MAX_POINT_STRIDE)
        graph_neighbors = _validate_int("graph_neighbors", self.graph_neighbors, 1, _MAX_GRAPH_NEIGHBORS)
        n_eigenvalues = _validate_int("n_eigenvalues", self.n_eigenvalues, 1, _MAX_EIGENVALUES)
        calibration_window = _validate_int(
            "calibration_window", self.calibration_window, 1, _MAX_CALIBRATION_WINDOW
        )
        calibration_min_periods = _validate_int(
            "calibration_min_periods", self.calibration_min_periods, 1
        )
        max_stream_observations = _validate_int(
            "max_stream_observations", self.max_stream_observations, 1, _MAX_STREAM_OBSERVATIONS
        )

        if graph_neighbors >= cloud_window:
            raise ValueError("graph_neighbors must be smaller than cloud_window")
        if n_eigenvalues > cloud_window:
            raise ValueError("n_eigenvalues cannot exceed cloud_window")
        if calibration_min_periods > calibration_window:
            raise ValueError("calibration_min_periods cannot exceed calibration_window")

        if self.min_points is None:
            # A small cloud is still useful in tests and in short live streams;
            # use eight points when possible, but never make the detector
            # permanently invalid for a deliberately small cloud_window.
            min_points = min(
                cloud_window,
                max(8, n_eigenvalues, graph_neighbors + 1),
            )
        else:
            min_points = _validate_int("min_points", self.min_points, 2)
            if min_points > cloud_window:
                raise ValueError("min_points cannot exceed cloud_window")
        required_points = max(2, n_eigenvalues, graph_neighbors + 1)
        if min_points < required_points:
            raise ValueError(
                "min_points must be at least max(n_eigenvalues, graph_neighbors + 1, 2)"
            )

        threshold = _validate_finite("threshold", self.threshold, minimum=0.0, strict_minimum=True)
        drift = _validate_finite("drift", self.drift, minimum=0.0)
        decay = _validate_finite("decay", self.decay, minimum=0.0, maximum=1.0,
                                 strict_minimum=True)
        scale_floor = _validate_finite("scale_floor", self.scale_floor,
                                       minimum=0.0, strict_minimum=True)
        z_clip = _validate_finite("z_clip", self.z_clip, minimum=0.0, strict_minimum=True)
        whitening_ridge = _validate_finite(
            "whitening_ridge", self.whitening_ridge, minimum=0.0
        )
        forgetting_lambda_min = _validate_finite(
            "forgetting_lambda_min", self.forgetting_lambda_min,
            minimum=0.0, maximum=1.0, strict_minimum=True,
        )
        forgetting_lambda_max = _validate_finite(
            "forgetting_lambda_max", self.forgetting_lambda_max,
            minimum=0.0, maximum=1.0,
        )
        forgetting_sensitivity = _validate_finite(
            "forgetting_sensitivity", self.forgetting_sensitivity, minimum=0.0
        )
        if forgetting_lambda_min > forgetting_lambda_max:
            raise ValueError("forgetting_lambda_min must not exceed forgetting_lambda_max")

        if not isinstance(self.input_kind, str):
            raise ValueError("input_kind must be 'returns' or 'prices'")
        input_kind = self.input_kind.lower()
        if input_kind not in {"returns", "prices"}:
            raise ValueError("input_kind must be 'returns' or 'prices'")

        backend = _validate_backend(self.persistent_laplacian_backend)

        # A configured exact backend exposes its finite vertex and spectrum
        # budgets.  Reject incompatible rolling settings at construction time;
        # otherwise the first large cloud would fail only after a stream had
        # already started consuming observations.
        backend_max_vertices = getattr(backend, "max_vertices", None)
        if backend_max_vertices is not None:
            if isinstance(backend_max_vertices, bool) or not isinstance(
                backend_max_vertices, Integral
            ):
                raise ValueError("persistent backend max_vertices must be an integer")
            if int(backend_max_vertices) < cloud_window:
                raise ValueError(
                    "cloud_window cannot exceed the persistent backend max_vertices"
                )
        backend_n_eigenvalues = getattr(backend, "n_eigenvalues", None)
        if backend_n_eigenvalues is not None:
            if isinstance(backend_n_eigenvalues, bool) or not isinstance(
                backend_n_eigenvalues, Integral
            ):
                raise ValueError("persistent backend n_eigenvalues must be an integer")
            if int(backend_n_eigenvalues) != n_eigenvalues:
                raise ValueError(
                    "persistent backend n_eigenvalues must match the topology configuration"
                )

        object.__setattr__(self, "embedding_dim", embedding_dim)
        object.__setattr__(self, "cloud_window", cloud_window)
        object.__setattr__(self, "point_stride", point_stride)
        object.__setattr__(self, "graph_neighbors", graph_neighbors)
        object.__setattr__(self, "n_eigenvalues", n_eigenvalues)
        object.__setattr__(self, "min_points", min_points)
        object.__setattr__(self, "calibration_window", calibration_window)
        object.__setattr__(self, "calibration_min_periods", calibration_min_periods)
        object.__setattr__(self, "threshold", threshold)
        object.__setattr__(self, "drift", drift)
        object.__setattr__(self, "decay", decay)
        object.__setattr__(self, "scale_floor", scale_floor)
        object.__setattr__(self, "z_clip", z_clip)
        object.__setattr__(self, "whitening_ridge", whitening_ridge)
        object.__setattr__(self, "forgetting_lambda_min", forgetting_lambda_min)
        object.__setattr__(self, "forgetting_lambda_max", forgetting_lambda_max)
        object.__setattr__(self, "forgetting_sensitivity", forgetting_sensitivity)
        object.__setattr__(self, "input_kind", input_kind)
        object.__setattr__(self, "persistent_laplacian_backend", backend)
        object.__setattr__(self, "max_stream_observations", max_stream_observations)

    @property
    def feature_count(self) -> int:
        """Number of raw rolling features emitted by the detector."""

        return 11 + self.n_eigenvalues

    @property
    def cusum_threshold(self) -> float:
        """Alias that makes the CUSUM parameter explicit at call sites."""

        return self.threshold

    @property
    def cusum_drift(self) -> float:
        return self.drift

    @property
    def cusum_decay(self) -> float:
        return self.decay

    def forgetting_factor(self, score: float) -> float:
        """Map a non-negative CUSUM score to the learner memory factor."""

        score = _validate_finite("score", score, minimum=0.0)
        value = self.forgetting_lambda_min + (
            self.forgetting_lambda_max - self.forgetting_lambda_min
        ) * exp(-self.forgetting_sensitivity * score)
        return min(self.forgetting_lambda_max, max(self.forgetting_lambda_min, value))


@dataclass(frozen=True)
class SpectralSummary:
    """Stable spectral descriptors of one point cloud.

    The eigenvalues are those of the normalized k-NN graph Laplacian for the
    default approximation, sorted ascending and padded with ``2.0`` only when
    a caller asks for more values than the cloud contains.  ``trace`` is the
    mean of the available spectrum, and ``spectral_entropy`` is normalized to
    ``[0, 1]``.
    """

    eigenvalues: Tuple[float, ...]
    spectral_gap: float
    algebraic_connectivity: float
    spectral_entropy: float
    trace: float
    distance_scale: float
    method: str = _METHOD
    backend_name: Optional[str] = None
    backend_evidence_digest: Optional[str] = None

    @property
    def is_approximation(self) -> bool:
        return self.method == _METHOD


@dataclass(frozen=True)
class PointCloudFeatures:
    """Raw geometry and spectral features for one rolling cloud."""

    values: Tuple[float, ...]
    names: Tuple[str, ...]
    spectral: SpectralSummary


@dataclass(frozen=True)
class WhiteningResult:
    """Result of robust marginal scaling plus covariance whitening."""

    values: Any
    location: Tuple[float, ...]
    scale: Tuple[float, ...]
    whitening_matrix: Tuple[Tuple[float, ...], ...]

    def __iter__(self):
        # Convenient unpacking while retaining named fields for callers who
        # prefer an explicit result object.
        yield self.values
        yield self.location
        yield self.scale


@dataclass(frozen=True)
class CusumResult:
    """CUSUM score and alarm arrays."""

    scores: Any
    alarms: Any

    def __iter__(self):
        yield self.scores
        yield self.alarms


@dataclass(frozen=True)
class TopologyResult:
    """Output of a causal rolling topology detector run.

    Arrays are NumPy arrays when NumPy is installed and ordinary Python lists
    otherwise.  Regardless of backend, every numeric output is finite.  The
    ``valid`` mask identifies rows with enough delay vectors for geometry, and
    ``calibrated`` identifies rows whose innovation was whitened against at
    least ``calibration_min_periods`` strictly earlier valid rows.
    """

    features: Any
    whitened_features: Any
    innovation: Any
    scores: Any
    alarms: Any
    valid: Any
    calibrated: Any
    point_counts: Any
    feature_names: Tuple[str, ...]
    method: str
    calibration_location: Any
    calibration_scale: Any
    backend_evidence_digests: Any = None

    @property
    def cusum_scores(self) -> Any:
        return self.scores

    @property
    def change_points(self) -> Any:
        if _np is not None and isinstance(self.alarms, _np.ndarray):
            return _np.flatnonzero(self.alarms)
        return [index for index, alarm in enumerate(self.alarms) if alarm]


@dataclass(frozen=True)
class StreamingTopologyResult:
    """One causal detector step suitable for an adaptive learner update."""

    step: int
    ready: bool
    score: float
    innovation: float
    alarm: bool
    forgetting_factor: float
    method: str
    raw_features: Tuple[float, ...]
    whitened_features: Tuple[float, ...]
    backend_evidence_digest: Optional[str] = None


def _feature_names(n_eigenvalues: int) -> Tuple[str, ...]:
    return (
        "point_count",
        "cloud_dimension",
        "cloud_centroid_norm",
        "cloud_dispersion",
        "cloud_pairwise_distance",
        "cloud_nearest_neighbor_distance",
        "cloud_anisotropy",
        *tuple(f"spectral_eigen_{index}" for index in range(n_eigenvalues)),
        "spectral_gap",
        "spectral_algebraic_connectivity",
        "spectral_entropy",
        "spectral_trace",
    )


def _as_finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} contains a non-numeric value")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} contains a non-numeric value") from exc
    if not isfinite(result):
        raise ValueError(f"{name} contains NaN or infinity")
    return result


def _rows_from_observations(observations: Any, name: str = "observations") -> list[list[float]]:
    """Convert a 1-D or 2-D numeric input to finite Python rows."""

    shape = getattr(observations, "shape", None)
    if shape is not None:
        try:
            dimensions = tuple(int(value) for value in shape)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} has an invalid shape") from exc
        if len(dimensions) not in {1, 2}:
            raise ValueError(f"{name} must be a one- or two-dimensional numeric sequence")
        if dimensions[0] > _MAX_BATCH_OBSERVATIONS:
            raise ValueError(f"{name} exceeds the topology row limit")
        if len(dimensions) == 2 and dimensions[1] > _MAX_OBSERVATION_DIMENSION:
            raise ValueError(f"{name} exceeds the observation dimension limit")
    else:
        try:
            observed_length = len(observations)
        except TypeError:
            observed_length = None
        if observed_length is not None and observed_length > _MAX_BATCH_OBSERVATIONS:
            raise ValueError(f"{name} exceeds the topology row limit")
        if observed_length:
            try:
                first_value = observations[0]
                first_length = len(first_value)
            except (TypeError, IndexError, KeyError):
                first_length = None
            if first_length is not None and first_length > _MAX_OBSERVATION_DIMENSION:
                raise ValueError(f"{name} exceeds the observation dimension limit")

    if _np is not None:
        try:
            array = _np.asarray(observations, dtype=float)
        except (TypeError, ValueError, OverflowError):
            array = None
        if array is not None:
            if array.ndim == 1:
                scalar_rows = [[_as_finite_float(value, name)] for value in array.tolist()]
                return scalar_rows
            if array.ndim == 2:
                if array.shape[1] == 0:
                    raise ValueError(f"{name} must have at least one column")
                if array.shape[1] > _MAX_OBSERVATION_DIMENSION:
                    raise ValueError(f"{name} exceeds the observation dimension limit")
                if not bool(_np.all(_np.isfinite(array))):
                    raise ValueError(f"{name} contains NaN or infinity")
                return [
                    [_as_finite_float(value, name) for value in row]
                    for row in array.tolist()
                ]
            raise ValueError(f"{name} must be a one- or two-dimensional numeric sequence")

    if isinstance(observations, (str, bytes)):
        raise ValueError(f"{name} must be a one- or two-dimensional numeric sequence")
    try:
        raw = list(itertools.islice(observations, _MAX_BATCH_OBSERVATIONS + 1))
    except TypeError as exc:
        raise ValueError(f"{name} must be a one- or two-dimensional numeric sequence") from exc
    if len(raw) > _MAX_BATCH_OBSERVATIONS:
        raise ValueError(f"{name} exceeds the topology row limit")
    if not raw:
        return []

    first = raw[0]
    is_row = not isinstance(first, Real) and not isinstance(first, (str, bytes))
    if not is_row:
        return [[_as_finite_float(value, name)] for value in raw]

    rows: list[list[float]] = []
    dimension: Optional[int] = None
    for row in raw:
        if isinstance(row, (str, bytes)):
            raise ValueError(f"{name} has inconsistent row dimensions")
        try:
            values = list(row)
        except TypeError as exc:
            raise ValueError(f"{name} has inconsistent row dimensions") from exc
        if not values:
            raise ValueError(f"{name} must have at least one column")
        if dimension is None:
            dimension = len(values)
            if dimension > _MAX_OBSERVATION_DIMENSION:
                raise ValueError(f"{name} exceeds the observation dimension limit")
        elif len(values) != dimension:
            raise ValueError(f"{name} has inconsistent row dimensions")
        rows.append([_as_finite_float(value, name) for value in values])
    return rows


def _prepare_observations(rows: list[list[float]], input_kind: str) -> list[list[float]]:
    if input_kind == "returns":
        return [row[:] for row in rows]
    if input_kind != "prices":  # Config validates this; keep the helper safe standalone.
        raise ValueError("input_kind must be 'returns' or 'prices'")
    if not rows:
        return []
    dimension = len(rows[0])
    result = [[0.0] * dimension]
    for index in range(1, len(rows)):
        row: list[float] = []
        for current, previous in zip(rows[index], rows[index - 1]):
            difference = current - previous
            if not isfinite(difference):
                raise ValueError("price differencing produced a non-finite value")
            row.append(difference)
        result.append(row)
    return result


def _coerce_point_cloud(point_cloud: Any, name: str = "point_cloud") -> list[list[float]]:
    if point_cloud is None:
        raise ValueError(f"{name} cannot be None")
    rows = _rows_from_observations(point_cloud, name=name)
    if rows and len(rows[0]) > _MAX_POINT_DIMENSION:
        raise ValueError(f"{name} exceeds the point dimension limit")
    return rows


def _validate_index(name: str, value: int, upper: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer in [0, {upper}]")
    value = int(value)
    if value < 0 or value > upper:
        raise ValueError(f"{name} must be an integer in [0, {upper}]")
    return value


def _point_cloud_from_rows(rows: Sequence[Sequence[float]], end_index: int,
                           embedding_dim: int, cloud_window: int,
                           point_stride: int) -> Tuple[Tuple[float, ...], ...]:
    if end_index < embedding_dim - 1:
        return ()
    endpoints = [
        end_index - offset * point_stride
        for offset in range(cloud_window - 1, -1, -1)
        if end_index - offset * point_stride >= embedding_dim - 1
    ]
    points: list[Tuple[float, ...]] = []
    for endpoint in endpoints:
        vector: list[float] = []
        for lag in range(embedding_dim - 1, -1, -1):
            vector.extend(float(value) for value in rows[endpoint - lag])
        point = tuple(vector)
        if not all(isfinite(value) for value in point):
            raise ValueError("point-cloud construction produced a non-finite value")
        points.append(point)
    return tuple(points)


def rolling_point_cloud(observations: Any, end_index: Optional[int] = None, *,
                        embedding_dim: int = 3, cloud_window: int = 64,
                        point_stride: int = 1) -> Tuple[Tuple[float, ...], ...]:
    """Return the causal delay-embedded cloud ending at ``end_index``.

    The current observation is included, but no observation after
    ``end_index`` can affect the result.  An insufficient warm-up returns an
    empty tuple rather than a partially shaped cloud.
    """

    embedding_dim = _validate_int("embedding_dim", embedding_dim, 1)
    cloud_window = _validate_int("cloud_window", cloud_window, 1)
    point_stride = _validate_int("point_stride", point_stride, 1)
    rows = _rows_from_observations(observations)
    if not rows:
        if end_index not in (None, -1):
            raise ValueError("end_index is invalid for an empty observation sequence")
        return ()
    if end_index is None:
        end_index = len(rows) - 1
    end_index = _validate_index("end_index", end_index, len(rows) - 1)
    return _point_cloud_from_rows(rows, end_index, embedding_dim, cloud_window, point_stride)


def rolling_point_clouds(observations: Any, *, embedding_dim: int = 3,
                         cloud_window: int = 64,
                         point_stride: int = 1) -> Iterator[Tuple[Tuple[float, ...], ...]]:
    """Yield each causal rolling delay cloud in time order."""

    embedding_dim = _validate_int("embedding_dim", embedding_dim, 1)
    cloud_window = _validate_int("cloud_window", cloud_window, 1)
    point_stride = _validate_int("point_stride", point_stride, 1)
    rows = _rows_from_observations(observations)
    for end_index in range(len(rows)):
        yield _point_cloud_from_rows(
            rows, end_index, embedding_dim, cloud_window, point_stride
        )


def _median(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("median requires at least one value")
    ordered = sorted(float(value) for value in values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        result = ordered[middle]
    else:
        result = 0.5 * (ordered[middle - 1] + ordered[middle])
    if not isfinite(result):
        raise ValueError("median calculation produced a non-finite value")
    return result


def _quantile(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise ValueError("quantile requires at least one value")
    if fraction <= 0.0:
        return min(values)
    if fraction >= 1.0:
        return max(values)
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    result = ordered[lower] * (1.0 - weight) + ordered[upper] * weight
    if not isfinite(result):
        raise ValueError("quantile calculation produced a non-finite value")
    return result


def _validate_matrix(matrix: Sequence[Sequence[float]], name: str) -> Tuple[list[list[float]], int]:
    rows = [list(row) for row in matrix]
    if not rows:
        return [], 0
    dimension = len(rows[0])
    if dimension == 0:
        raise ValueError(f"{name} must have at least one column")
    result: list[list[float]] = []
    for row in rows:
        if len(row) != dimension:
            raise ValueError(f"{name} has inconsistent row dimensions")
        clean = [_as_finite_float(value, name) for value in row]
        result.append(clean)
    return result, dimension


def _robust_location_scale(rows: Sequence[Sequence[float]], dimension: int,
                           scale_floor: float) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    if dimension < 0:
        raise ValueError("dimension cannot be negative")
    if not rows:
        return (tuple(0.0 for _ in range(dimension)),
                tuple(1.0 for _ in range(dimension)))
    location = []
    scale = []
    for column in range(dimension):
        values = [float(row[column]) for row in rows]
        center = _median(values)
        deviations = []
        for value in values:
            deviation = abs(value - center)
            if not isfinite(deviation):
                raise ValueError("robust calibration overflowed; rescale the input")
            deviations.append(deviation)
        mad_scale = 1.4826 * _median(deviations)
        q25 = _quantile(values, 0.25)
        q75 = _quantile(values, 0.75)
        iqr_scale = (q75 - q25) / 1.3489795
        candidate = max(mad_scale, iqr_scale, scale_floor)
        if not isfinite(candidate):
            raise ValueError("robust calibration produced a non-finite scale")
        location.append(center)
        scale.append(candidate)
    return tuple(location), tuple(scale)


def _normalize_cloud(cloud: Sequence[Sequence[float]], scale_floor: float
                     ) -> Tuple[list[list[float]], Tuple[float, ...], Tuple[float, ...]]:
    clean, dimension = _validate_matrix(cloud, "point_cloud")
    if not clean:
        return [], (), ()
    location, scale = _robust_location_scale(clean, dimension, scale_floor)
    normalized: list[list[float]] = []
    for row in clean:
        current = []
        for value, center, spread in zip(row, location, scale):
            difference = value - center
            if not isfinite(difference):
                raise ValueError("point-cloud normalization overflowed; rescale the input")
            scaled = difference / spread
            if not isfinite(scaled):
                raise ValueError("point-cloud normalization produced a non-finite value")
            current.append(max(-_MAX_NORMALIZED_CLOUD_COORDINATE,
                               min(_MAX_NORMALIZED_CLOUD_COORDINATE, scaled)))
        normalized.append(current)
    return normalized, location, scale


def _pairwise_distances(points: Sequence[Sequence[float]]) -> list[list[float]]:
    count = len(points)
    distances = [[0.0] * count for _ in range(count)]
    for left in range(count):
        for right in range(left + 1, count):
            squared = 0.0
            for a, b in zip(points[left], points[right]):
                difference = a - b
                squared += difference * difference
            distance = sqrt(max(0.0, squared))
            if not isfinite(distance):
                raise ValueError("point-cloud distance calculation produced a non-finite value")
            distances[left][right] = distance
            distances[right][left] = distance
    return distances


def _jacobi_eigh(matrix: Sequence[Sequence[float]], *, max_sweeps: int = 32,
                 tolerance: float = 1.0e-12) -> Tuple[list[float], list[list[float]]]:
    """Small deterministic symmetric eigensolver used when NumPy is absent."""

    size = len(matrix)
    if size == 0:
        return [], []
    work = [list(map(float, row)) for row in matrix]
    vectors = [[1.0 if row == column else 0.0 for column in range(size)]
               for row in range(size)]
    for _ in range(max_sweeps):
        pivot_p = 0
        pivot_q = 0
        largest = 0.0
        for row in range(size):
            for column in range(row + 1, size):
                magnitude = abs(work[row][column])
                if magnitude > largest:
                    largest = magnitude
                    pivot_p, pivot_q = row, column
        if largest <= tolerance:
            break

        p, q = pivot_p, pivot_q
        apq = work[p][q]
        if apq == 0.0:
            continue
        tau = (work[q][q] - work[p][p]) / (2.0 * apq)
        t = copysign(1.0 / (abs(tau) + sqrt(1.0 + tau * tau)), tau)
        c = 1.0 / sqrt(1.0 + t * t)
        s = t * c
        app = work[p][p]
        aqq = work[q][q]
        work[p][p] = app - t * apq
        work[q][q] = aqq + t * apq
        work[p][q] = 0.0
        work[q][p] = 0.0
        for index in range(size):
            if index in (p, q):
                continue
            aip = work[index][p]
            aiq = work[index][q]
            work[index][p] = c * aip - s * aiq
            work[p][index] = work[index][p]
            work[index][q] = s * aip + c * aiq
            work[q][index] = work[index][q]
        for index in range(size):
            vip = vectors[index][p]
            viq = vectors[index][q]
            vectors[index][p] = c * vip - s * viq
            vectors[index][q] = s * vip + c * viq

    pairs = [(work[index][index], index) for index in range(size)]
    pairs.sort(key=lambda pair: (pair[0], pair[1]))
    values = [pair[0] for pair in pairs]
    # Keep eigenvectors as columns in sorted order.  The deliberately explicit
    # loop avoids relying on a matrix package in the fallback path.
    result_vectors = [
        [vectors[row][pair[1]] for pair in pairs] for row in range(size)
    ]
    return values, result_vectors


def _symmetric_eigenvalues(matrix: Sequence[Sequence[float]]) -> list[float]:
    if not matrix:
        return []
    if _np is not None:
        try:
            values = _np.linalg.eigvalsh(_np.asarray(matrix, dtype=float)).tolist()
        except (_np.linalg.LinAlgError, ValueError, FloatingPointError) as exc:
            raise ValueError("the NumPy symmetric eigensolver failed") from exc
        if all(isfinite(float(value)) for value in values):
            return [float(value) for value in values]
        raise ValueError("the NumPy symmetric eigensolver returned non-finite values")
    values, _ = _jacobi_eigh(matrix)
    return [float(value) for value in values]


def _normalized_graph_laplacian(points: Sequence[Sequence[float]], graph_neighbors: int,
                                scale_floor: float) -> Tuple[list[list[float]], float]:
    count = len(points)
    if count == 0:
        return [], scale_floor
    if count == 1:
        return [[0.0]], scale_floor
    distances = _pairwise_distances(points)
    positive = [
        distances[row][column]
        for row in range(count)
        for column in range(row + 1, count)
        if distances[row][column] > scale_floor
    ]
    distance_scale = max(_median(positive) if positive else 0.0, scale_floor)
    weights = [[0.0] * count for _ in range(count)]
    neighbours = min(graph_neighbors, count - 1)
    denominator = 2.0 * distance_scale * distance_scale
    for row in range(count):
        ordered = sorted(
            ((distances[row][column], column) for column in range(count) if column != row),
            key=lambda item: (item[0], tuple(points[item[1]]), item[1]),
        )
        for distance, column in ordered[:neighbours]:
            exponent = min(700.0, (distance * distance) / denominator)
            weight = exp(-exponent)
            # Union graph: one-sided kNN choices are enough to connect a point,
            # while max() keeps the symmetric weight independent of traversal.
            if weight > weights[row][column]:
                weights[row][column] = weight
                weights[column][row] = weight

    degrees = [sum(row) for row in weights]
    laplacian = [[0.0] * count for _ in range(count)]
    for row in range(count):
        if degrees[row] > _TINY:
            laplacian[row][row] = 1.0
        for column in range(row):
            if weights[row][column] <= 0.0 or degrees[row] <= _TINY or degrees[column] <= _TINY:
                continue
            normalized_weight = weights[row][column] / sqrt(degrees[row] * degrees[column])
            laplacian[row][column] = -normalized_weight
            laplacian[column][row] = -normalized_weight
    return laplacian, distance_scale


def _spectral_descriptors(eigenvalues: Sequence[float], requested: int,
                          distance_scale: float, method: str,
                          backend_name: Optional[str] = None,
                          backend_evidence_digest: Optional[str] = None,
                          full_count: Optional[int] = None,
                          normalized_spectrum: bool = True) -> SpectralSummary:
    clean = []
    for value in sorted(float(value) for value in eigenvalues):
        if not isfinite(value):
            raise ValueError("spectral backend returned a non-finite value")
        if normalized_spectrum:
            # The default normalized Laplacian has spectrum in [0, 2].  Clipping
            # numerical round-off here keeps summaries stable across eigensolvers.
            clean.append(max(0.0, min(2.0, value)))
        else:
            if value < 0.0:
                raise ValueError("persistent_laplacian_backend returned a negative eigenvalue")
            clean.append(value)
    if not clean:
        clean = [0.0]
    if not normalized_spectrum and len(clean) < requested:
        raise ValueError(
            "persistent_laplacian_backend returned fewer eigenvalues than requested"
        )
    padded = clean[:requested]
    if len(padded) < requested:
        padded.extend([2.0] * (requested - len(padded)))
    if len(clean) > 1:
        algebraic_connectivity = clean[1]
        spectral_gap = max(0.0, clean[1] - clean[0])
    else:
        algebraic_connectivity = 0.0
        spectral_gap = 0.0
    positive = [value for value in clean if value > _TINY]
    if len(positive) <= 1:
        entropy = 0.0
    else:
        total = sum(positive)
        entropy = -sum((value / total) * log(value / total) for value in positive)
        entropy /= log(len(positive))
        entropy = max(0.0, min(1.0, entropy))
    divisor = float(full_count if full_count is not None else len(clean))
    trace = sum(clean) / max(1.0, divisor)
    if backend_evidence_digest is not None:
        if (
            not isinstance(backend_evidence_digest, str)
            or len(backend_evidence_digest) != 64
            or any(value not in "0123456789abcdefABCDEF" for value in backend_evidence_digest)
        ):
            raise ValueError("persistent backend evidence digest must be 64 hex characters")
        backend_evidence_digest = backend_evidence_digest.lower()
    return SpectralSummary(
        eigenvalues=tuple(float(value) for value in padded),
        spectral_gap=float(spectral_gap),
        algebraic_connectivity=float(algebraic_connectivity),
        spectral_entropy=float(entropy),
        trace=float(trace),
        distance_scale=float(max(0.0, distance_scale)),
        method=method,
        backend_name=backend_name,
        backend_evidence_digest=backend_evidence_digest,
    )


def _backend_name(backend: Callable[..., Any]) -> str:
    return getattr(backend, "__qualname__", getattr(backend, "__name__", type(backend).__name__))


def _backend_evidence_digest(result: Any) -> Optional[str]:
    digest = getattr(result, "evidence_digest", None)
    if callable(digest):
        digest = digest()
    if digest is None:
        return None
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(value not in "0123456789abcdefABCDEF" for value in digest)
    ):
        raise ValueError(
            "persistent backend evidence digest must be 64 hex characters"
        )
    return digest.lower()


def _backend_eigenvalues(result: Any) -> Sequence[float]:
    if isinstance(result, SpectralSummary):
        return result.eigenvalues
    if isinstance(result, Mapping):
        if "eigenvalues" not in result:
            raise ValueError("persistent_laplacian_backend mappings need an 'eigenvalues' key")
        result = result["eigenvalues"]
    elif hasattr(result, "eigenvalues"):
        result = getattr(result, "eigenvalues")
    if isinstance(result, (str, bytes)):
        raise ValueError("persistent_laplacian_backend must return numeric eigenvalues")
    try:
        iterator = iter(result)
    except TypeError as exc:
        raise ValueError("persistent_laplacian_backend must return numeric eigenvalues") from exc
    values = list(itertools.islice(iterator, _MAX_BACKEND_EIGENVALUES + 1))
    if len(values) > _MAX_BACKEND_EIGENVALUES:
        raise ValueError("persistent_laplacian_backend returned too many eigenvalues")
    if not values:
        raise ValueError("persistent_laplacian_backend returned no eigenvalues")
    return values


def spectral_summary(point_cloud: Any, *, n_eigenvalues: int = 4,
                     graph_neighbors: int = 8,
                     scale_floor: float = 1.0e-8,
                     persistent_laplacian_backend: Optional[Callable[..., Any]] = None,
                     assume_normalized: bool = False,
                     ) -> SpectralSummary:
    """Compute a deterministic spectral summary for one point cloud.

    By default this constructs a symmetric union kNN graph and diagonalizes its
    normalized graph Laplacian.  If ``persistent_laplacian_backend`` is given,
    it is called as ``backend(normalized_point_cloud, n_eigenvalues)`` and its
    returned eigenvalues are summarized instead.  The backend seam is the only
    place where an exact persistent-Laplacian implementation can be connected;
    the built-in path remains an explicitly labelled approximation.
    """

    n_eigenvalues = _validate_int("n_eigenvalues", n_eigenvalues, 1, _MAX_EIGENVALUES)
    graph_neighbors = _validate_int("graph_neighbors", graph_neighbors, 1, _MAX_GRAPH_NEIGHBORS)
    scale_floor = _validate_finite("scale_floor", scale_floor,
                                   minimum=0.0, strict_minimum=True)
    backend = _validate_backend(persistent_laplacian_backend)
    cloud = _coerce_point_cloud(point_cloud)
    if len(cloud) > _MAX_CLOUD_WINDOW:
        raise ValueError("point cloud exceeds the configured size limit")
    if assume_normalized:
        normalized = [list(row) for row in cloud]
        if any(
            not isfinite(value)
            for row in normalized
            for value in row
        ):
            raise ValueError("normalized point cloud contains a non-finite value")
    else:
        normalized, _, _ = _normalize_cloud(cloud, scale_floor)
    if not normalized:
        return _spectral_descriptors(
            [0.0], n_eigenvalues, scale_floor, _METHOD, full_count=1
        )

    if backend is not None:
        canonical_cloud = tuple(sorted(tuple(row) for row in normalized))
        result = backend(canonical_cloud, n_eigenvalues)
        values = _backend_eigenvalues(result)
        evidence_digest = _backend_evidence_digest(result)
        backend_label = _backend_name(backend)
        return _spectral_descriptors(
            values,
            n_eigenvalues,
            scale_floor,
            "persistent_laplacian_backend",
            backend_name=backend_label,
            backend_evidence_digest=evidence_digest,
            full_count=len(values),
            normalized_spectrum=False,
        )

    # Canonicalizing point order makes tie handling and the pure-Python Jacobi
    # fallback invariant to an otherwise irrelevant permutation of a cloud.
    normalized = sorted(normalized, key=tuple)
    laplacian, distance_scale = _normalized_graph_laplacian(
        normalized, graph_neighbors, scale_floor
    )
    values = _symmetric_eigenvalues(laplacian)
    return _spectral_descriptors(
        values,
        n_eigenvalues,
        distance_scale,
        _METHOD,
        full_count=len(values),
    )


def point_cloud_features(point_cloud: Any, *, n_eigenvalues: int = 4,
                         graph_neighbors: int = 8,
                         scale_floor: float = 1.0e-8,
                         persistent_laplacian_backend: Optional[Callable[..., Any]] = None
                         ) -> PointCloudFeatures:
    """Return robust geometric and spectral features for one point cloud."""

    n_eigenvalues = _validate_int("n_eigenvalues", n_eigenvalues, 1, _MAX_EIGENVALUES)
    graph_neighbors = _validate_int("graph_neighbors", graph_neighbors, 1, _MAX_GRAPH_NEIGHBORS)
    scale_floor = _validate_finite("scale_floor", scale_floor,
                                   minimum=0.0, strict_minimum=True)
    cloud = _coerce_point_cloud(point_cloud)
    if len(cloud) > _MAX_CLOUD_WINDOW:
        raise ValueError("point cloud exceeds the configured size limit")
    names = _feature_names(n_eigenvalues)
    if not cloud:
        spectral = spectral_summary(
            (),
            n_eigenvalues=n_eigenvalues,
            graph_neighbors=graph_neighbors,
            scale_floor=scale_floor,
            persistent_laplacian_backend=persistent_laplacian_backend,
        )
        return PointCloudFeatures(
            values=tuple(0.0 for _ in names), names=names, spectral=spectral
        )

    normalized, _, _ = _normalize_cloud(cloud, scale_floor)
    count = len(normalized)
    dimension = len(normalized[0])
    distances = _pairwise_distances(normalized)
    pairwise = [
        distances[row][column]
        for row in range(count)
        for column in range(row + 1, count)
    ]
    nearest = [
        min(
            (distances[row][column] for column in range(count) if column != row),
            default=0.0,
        )
        for row in range(count)
    ]
    centroid = [sum(row[column] for row in normalized) / count for column in range(dimension)]
    centroid_norm = sqrt(sum(value * value for value in centroid))
    dispersion = sqrt(
        sum(sum(value * value for value in row) for row in normalized) / count
    )
    variances = [
        sum((row[column] - centroid[column]) ** 2 for row in normalized) / count
        for column in range(dimension)
    ]
    variance_total = sum(variances)
    anisotropy = max(variances) / variance_total if variance_total > scale_floor else 0.0

    spectral = spectral_summary(
        normalized,
        n_eigenvalues=n_eigenvalues,
        graph_neighbors=graph_neighbors,
        scale_floor=scale_floor,
        persistent_laplacian_backend=persistent_laplacian_backend,
        assume_normalized=True,
    )
    values = [
        float(count),
        float(dimension),
        float(centroid_norm),
        float(dispersion),
        float(_median(pairwise) if pairwise else 0.0),
        float(_median(nearest) if nearest else 0.0),
        float(max(0.0, min(1.0, anisotropy))),
        *spectral.eigenvalues,
        float(spectral.spectral_gap),
        float(spectral.algebraic_connectivity),
        float(spectral.spectral_entropy),
        float(spectral.trace),
    ]
    if len(values) != len(names) or not all(isfinite(value) for value in values):
        raise ValueError("point-cloud feature calculation produced non-finite values")
    return PointCloudFeatures(values=tuple(values), names=names, spectral=spectral)


def _matrix_from_features(features: Any, name: str) -> Tuple[list[list[float]], int]:
    rows = _rows_from_observations(features, name=name)
    return rows, (len(rows[0]) if rows else 0)


def _identity(size: int) -> list[list[float]]:
    return [[1.0 if row == column else 0.0 for column in range(size)]
            for row in range(size)]


def _matrix_vector(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> list[float]:
    return [sum(value * component for value, component in zip(row, vector))
            for row in matrix]


def robust_whiten(features: Any, reference: Any = None, *,
                  scale_floor: float = 1.0e-8,
                  z_clip: float = 12.0,
                  ridge: float = 1.0e-6) -> WhiteningResult:
    """Robustly whiten feature rows against a reference sample.

    The location is the coordinate-wise median and the marginal scale is the
    larger of MAD and IQR scales, floored at ``scale_floor``.  A covariance
    whitening step then removes residual feature correlation; its ridge keeps
    constant or short reference windows finite.  ``reference`` may contain
    only rows strictly preceding the target when used in a causal detector.
    """

    scale_floor = _validate_finite("scale_floor", scale_floor,
                                   minimum=0.0, strict_minimum=True)
    z_clip = _validate_finite("z_clip", z_clip, minimum=0.0, strict_minimum=True)
    ridge = _validate_finite("ridge", ridge, minimum=0.0)
    target, dimension = _matrix_from_features(features, "features")
    if reference is None:
        calibration = target
    else:
        calibration, reference_dimension = _matrix_from_features(reference, "reference")
        if target and reference_dimension != dimension:
            raise ValueError("features and reference have different column counts")
        if not target and dimension == 0:
            dimension = reference_dimension
    if calibration and len(calibration[0]) != dimension:
        raise ValueError("reference has inconsistent column counts")

    location, scale = _robust_location_scale(calibration, dimension, scale_floor)
    if not calibration:
        whitening = _identity(dimension)
    else:
        standardized_reference = []
        for row in calibration:
            standardized_reference.append([
                max(-z_clip, min(z_clip, (value - center) / spread))
                for value, center, spread in zip(row, location, scale)
            ])
        if len(standardized_reference) < 2 or dimension == 0:
            whitening = _identity(dimension)
        else:
            means = [
                sum(row[column] for row in standardized_reference) /
                len(standardized_reference)
                for column in range(dimension)
            ]
            denominator = float(max(1, len(standardized_reference) - 1))
            covariance = []
            for row_index in range(dimension):
                covariance.append([
                    sum(
                        (sample[row_index] - means[row_index])
                        * (sample[column_index] - means[column_index])
                        for sample in standardized_reference
                    ) / denominator
                    for column_index in range(dimension)
                ])
            eigenvalues, eigenvectors = _jacobi_eigh(covariance, max_sweeps=48)
            eigenvalues = [max(0.0, float(value)) for value in eigenvalues]
            eigen_floor = max(ridge, _TINY)
            # V diag(lambda^-1/2) V^T.  The Jacobi vectors are columns.
            whitening = [[0.0] * dimension for _ in range(dimension)]
            for row_index in range(dimension):
                for column_index in range(dimension):
                    whitening[row_index][column_index] = sum(
                        eigenvectors[row_index][eigen_index]
                        * (1.0 / sqrt(max(eigenvalues[eigen_index], eigen_floor)))
                        * eigenvectors[column_index][eigen_index]
                        for eigen_index in range(dimension)
                    )

    whitened: list[list[float]] = []
    for row in target:
        standardized = [
            max(-z_clip, min(z_clip, (value - center) / spread))
            for value, center, spread in zip(row, location, scale)
        ]
        current = _matrix_vector(whitening, standardized)
        current = [max(-z_clip, min(z_clip, value)) for value in current]
        if not all(isfinite(value) for value in current):
            raise ValueError("whitening produced a non-finite value")
        whitened.append(current)
    return WhiteningResult(
        values=_to_matrix_output(whitened, dimension),
        location=tuple(location),
        scale=tuple(scale),
        whitening_matrix=tuple(tuple(float(value) for value in row) for row in whitening),
    )


def _to_matrix_output(rows: Sequence[Sequence[float]], dimension: int) -> Any:
    if _np is not None:
        return _np.asarray(rows, dtype=float).reshape((len(rows), dimension))
    return [list(row) for row in rows]


def _to_vector_output(values: Sequence[float]) -> Any:
    if _np is not None:
        return _np.asarray(list(values), dtype=float)
    return [float(value) for value in values]


def _to_bool_output(values: Sequence[bool]) -> Any:
    if _np is not None:
        return _np.asarray(list(values), dtype=bool)
    return [bool(value) for value in values]


def _to_int_output(values: Sequence[int]) -> Any:
    if _np is not None:
        return _np.asarray(list(values), dtype=int)
    return [int(value) for value in values]


def cusum_scores(innovations: Any, *, drift: float = 0.5,
                 threshold: float = 5.0, decay: float = 1.0,
                 valid: Optional[Sequence[bool]] = None) -> CusumResult:
    """Compute causal non-negative CUSUM scores ``G_t``.

    Invalid/warm-up rows reset the accumulator and emit a finite zero score.
    ``innovations`` must be a finite one-dimensional non-negative sequence.
    """

    drift = _validate_finite("drift", drift, minimum=0.0)
    threshold = _validate_finite("threshold", threshold, minimum=0.0, strict_minimum=True)
    decay = _validate_finite("decay", decay, minimum=0.0, maximum=1.0,
                             strict_minimum=True)
    if _np is not None:
        try:
            array = _np.asarray(innovations, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("innovations must be a finite one-dimensional sequence") from exc
        if array.ndim != 1:
            raise ValueError("innovations must be a finite one-dimensional sequence")
        values = [_as_finite_float(value, "innovations") for value in array.tolist()]
    else:
        if isinstance(innovations, (str, bytes)):
            raise ValueError("innovations must be a finite one-dimensional sequence")
        try:
            values = [_as_finite_float(value, "innovations") for value in innovations]
        except TypeError as exc:
            raise ValueError("innovations must be a finite one-dimensional sequence") from exc
    if any(value < 0.0 for value in values):
        raise ValueError("innovations must be non-negative")
    if valid is None:
        valid_values = [True] * len(values)
    else:
        valid_values = [bool(value) for value in valid]
        if len(valid_values) != len(values):
            raise ValueError("valid must have the same length as innovations")
    scores = []
    alarms = []
    accumulator = 0.0
    for value, is_valid in zip(values, valid_values):
        if not is_valid:
            accumulator = 0.0
        else:
            accumulator = max(0.0, decay * accumulator + value - drift)
            if not isfinite(accumulator):
                raise ValueError("CUSUM calculation produced a non-finite value")
        scores.append(accumulator)
        alarms.append(bool(is_valid and accumulator >= threshold))
    return CusumResult(scores=_to_vector_output(scores), alarms=_to_bool_output(alarms))


class RollingTopologyDetector:
    """Causal rolling market-geometry detector.

    The detector is stateless across ``detect`` calls.  This is intentional: a
    later call cannot contaminate an earlier calibration window, and a caller
    can safely compare prefixes of runs with different future observations.
    ``fit``/``transform`` aliases are supplied for pipeline-style callers but
    perform the same causal calculation.
    """

    def __init__(self, config: Optional[Any] = None, **kwargs: Any) -> None:
        if config is not None and kwargs:
            raise ValueError("pass either config or keyword parameters, not both")
        if isinstance(config, TopologyConfig):
            self.config = config
        else:
            if config is None:
                values = dict(kwargs)
            elif isinstance(config, Mapping):
                values = dict(config)
            else:
                raise ValueError("config must be TopologyConfig, a mapping, or None")
            aliases = {
                "window": "cloud_window",
                "embedding": "embedding_dim",
                "neighbors": "graph_neighbors",
                "eigenvalues": "n_eigenvalues",
                "cusum_threshold": "threshold",
                "cusum_drift": "drift",
                "cusum_decay": "decay",
                "backend": "persistent_laplacian_backend",
                "persistent_backend": "persistent_laplacian_backend",
            }
            for old_name, new_name in aliases.items():
                if old_name in values:
                    if new_name in values:
                        raise ValueError(f"specify only one of {old_name} and {new_name}")
                    values[new_name] = values.pop(old_name)
            self.config = TopologyConfig(**values)
        self._last_result: Optional[TopologyResult] = None
        self._stream_observations: list[list[float]] = []

    @property
    def backend_identity(self) -> str:
        """Stable human-readable identity for the selected spectral backend."""

        backend = self.config.persistent_laplacian_backend
        if backend is None:
            solver = "numpy-eigvalsh" if _np is not None else "python-jacobi"
            return f"{_METHOD}:{solver}"
        identity = getattr(backend, "identity", None)
        if callable(identity):
            identity = identity()
        if isinstance(identity, str) and identity:
            return identity
        module = getattr(backend, "__module__", type(backend).__module__)
        name = getattr(backend, "__qualname__", type(backend).__qualname__)
        return f"{module}:{name}"

    @property
    def config_identity(self) -> str:
        """Digest of state-affecting detector configuration and backend."""

        values = {
            "embedding_dim": self.config.embedding_dim,
            "cloud_window": self.config.cloud_window,
            "point_stride": self.config.point_stride,
            "graph_neighbors": self.config.graph_neighbors,
            "n_eigenvalues": self.config.n_eigenvalues,
            "min_points": self.config.min_points,
            "calibration_window": self.config.calibration_window,
            "calibration_min_periods": self.config.calibration_min_periods,
            "threshold": self.config.threshold,
            "drift": self.config.drift,
            "decay": self.config.decay,
            "scale_floor": self.config.scale_floor,
            "z_clip": self.config.z_clip,
            "whitening_ridge": self.config.whitening_ridge,
            "forgetting_lambda_min": self.config.forgetting_lambda_min,
            "forgetting_lambda_max": self.config.forgetting_lambda_max,
            "forgetting_sensitivity": self.config.forgetting_sensitivity,
            "input_kind": self.config.input_kind,
            "max_stream_observations": self.config.max_stream_observations,
            "backend_identity": self.backend_identity,
        }
        payload = json.dumps(values, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def detect(self, observations: Any) -> TopologyResult:
        cfg = self.config
        raw_rows = _rows_from_observations(observations)
        if len(raw_rows) > _MAX_BATCH_OBSERVATIONS:
            raise ValueError(
                "observations exceed the configured topology batch limit "
                f"({_MAX_BATCH_OBSERVATIONS})"
            )
        rows = _prepare_observations(raw_rows, cfg.input_kind)
        count = len(rows)
        feature_names = _feature_names(cfg.n_eigenvalues)
        feature_count = len(feature_names)
        raw_features: list[list[float]] = []
        backend_evidence_digests: list[Optional[str]] = [None] * count
        point_counts: list[int] = []
        valid: list[bool] = []
        methods: set[str] = set()

        for end_index in range(count):
            cloud = _point_cloud_from_rows(
                rows,
                end_index,
                cfg.embedding_dim,
                cfg.cloud_window,
                cfg.point_stride,
            )
            cloud_count = len(cloud)
            point_counts.append(cloud_count)
            if cfg.min_points is None:  # Defensive; __post_init__ always resolves it.
                raise RuntimeError("validated topology configuration lost min_points")
            is_valid = cloud_count >= cfg.min_points
            valid.append(is_valid)
            if not is_valid:
                raw_features.append([0.0] * feature_count)
                continue
            extracted = point_cloud_features(
                cloud,
                n_eigenvalues=cfg.n_eigenvalues,
                graph_neighbors=cfg.graph_neighbors,
                scale_floor=cfg.scale_floor,
                persistent_laplacian_backend=cfg.persistent_laplacian_backend,
            )
            raw_features.append(list(extracted.values))
            backend_evidence_digests[end_index] = extracted.spectral.backend_evidence_digest
            methods.add(extracted.spectral.method)

        if not methods:
            method = (
                "persistent_laplacian_backend"
                if cfg.persistent_laplacian_backend is not None
                else _METHOD
            )
        elif len(methods) == 1:
            method = next(iter(methods))
        else:
            method = "+".join(sorted(methods))

        whitened = [[0.0] * feature_count for _ in range(count)]
        innovations = [0.0] * count
        calibrated = [False] * count
        calibration_location = [[0.0] * feature_count for _ in range(count)]
        calibration_scale = [[1.0] * feature_count for _ in range(count)]

        for index in range(count):
            if not valid[index]:
                continue
            start = max(0, index - cfg.calibration_window)
            reference = [
                raw_features[prior]
                for prior in range(start, index)
                if valid[prior]
            ]
            if len(reference) < cfg.calibration_min_periods:
                continue
            whitened_result = robust_whiten(
                [raw_features[index]],
                reference=reference,
                scale_floor=cfg.scale_floor,
                z_clip=cfg.z_clip,
                ridge=cfg.whitening_ridge,
            )
            current = [float(value) for value in whitened_result.values[0]]
            if not all(isfinite(value) for value in current):
                raise ValueError("detector calibration produced a non-finite feature row")
            whitened[index] = current
            calibration_location[index] = list(whitened_result.location)
            calibration_scale[index] = list(whitened_result.scale)
            innovations[index] = sqrt(
                sum(value * value for value in current) / max(1, feature_count)
            )
            if not isfinite(innovations[index]):
                raise ValueError("detector innovation produced a non-finite value")
            calibrated[index] = True

        cusum = cusum_scores(
            innovations,
            drift=cfg.drift,
            threshold=cfg.threshold,
            decay=cfg.decay,
            valid=calibrated,
        )
        result = TopologyResult(
            features=_to_matrix_output(raw_features, feature_count),
            whitened_features=_to_matrix_output(whitened, feature_count),
            innovation=_to_vector_output(innovations),
            scores=cusum.scores,
            alarms=cusum.alarms,
            valid=_to_bool_output(valid),
            calibrated=_to_bool_output(calibrated),
            point_counts=_to_int_output(point_counts),
            feature_names=feature_names,
            method=method,
            calibration_location=_to_matrix_output(calibration_location, feature_count),
            calibration_scale=_to_matrix_output(calibration_scale, feature_count),
            backend_evidence_digests=tuple(backend_evidence_digests),
        )
        self._last_result = result
        return result

    def fit(self, observations: Any, y: Any = None) -> "RollingTopologyDetector":
        del y  # This detector is unsupervised; labels never enter calibration.
        self.detect(observations)
        return self

    def transform(self, observations: Any) -> TopologyResult:
        return self.detect(observations)

    def fit_transform(self, observations: Any, y: Any = None) -> TopologyResult:
        del y
        return self.detect(observations)

    def predict(self, observations: Any) -> Any:
        return self.detect(observations).alarms

    def score_samples(self, observations: Any) -> Any:
        return self.detect(observations).innovation

    def observe(self, observation: Any) -> StreamingTopologyResult:
        """Consume one point-in-time state and return its causal control signal.

        The implementation recomputes the bounded rolling prefix so the
        streaming path is exactly equivalent to the corresponding prefix of
        :meth:`detect`. This reference behavior is intentionally simple and
        deterministic; an accelerated backend can replace it later without
        changing the returned contract.  Backend/resource errors are
        transactional: the rejected observation is not retained.
        """

        rows = _rows_from_observations([observation], name="observation")
        if len(rows) != 1:
            raise ValueError("observation must contain exactly one feature row")
        if len(self._stream_observations) >= self.config.max_stream_observations:
            raise ValueError("stream exceeds max_stream_observations")
        if self._stream_observations and len(rows[0]) != len(self._stream_observations[0]):
            raise ValueError("observation feature dimension does not match the stream")
        previous_observations = self._stream_observations
        previous_result = self._last_result
        self._stream_observations = previous_observations + [list(rows[0])]
        try:
            result = self.detect(self._stream_observations)
        except Exception:
            # Exact backends can reject a cloud on a resource or numerical
            # boundary.  A failed step must not consume its observation or
            # leave a partially updated detector reference behind.
            self._stream_observations = previous_observations
            self._last_result = previous_result
            raise
        index = len(self._stream_observations) - 1
        raw = result.features[index]
        whitened = result.whitened_features[index]
        score = float(result.scores[index])
        innovation = float(result.innovation[index])
        ready = bool(result.calibrated[index])
        return StreamingTopologyResult(
            step=index + 1,
            ready=ready,
            score=score,
            innovation=innovation,
            alarm=bool(result.alarms[index]),
            forgetting_factor=self.config.forgetting_factor(score),
            method=result.method,
            raw_features=tuple(float(value) for value in raw),
            whitened_features=tuple(float(value) for value in whitened),
            backend_evidence_digest=(
                None
                if result.backend_evidence_digests is None
                else result.backend_evidence_digests[index]
            ),
        )

    def reset_stream(self) -> None:
        """Clear only the state accumulated by :meth:`observe`."""

        self._stream_observations = []
        self._last_result = None

    def stream_state_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible causal streaming snapshot."""

        dimension = (
            len(self._stream_observations[0]) if self._stream_observations else None
        )
        return {
            "version": _STREAM_STATE_VERSION,
            "schema": _STREAM_SCHEMA,
            "config_identity": self.config_identity,
            "backend_identity": self.backend_identity,
            "feature_dimension": dimension,
            "observations": [list(row) for row in self._stream_observations],
        }

    def validate_stream_state_dict(self, state: Mapping[str, Any]) -> dict[str, Any]:
        """Validate and normalize a stream state without mutating the detector."""

        if not isinstance(state, Mapping):
            raise ValueError("topology stream state must be a mapping")
        version = state.get("version")
        if version not in {1, _STREAM_STATE_VERSION}:
            raise ValueError("unsupported topology stream state version")
        observations = state.get("observations", [])
        rows = (
            _rows_from_observations(observations, name="state.observations")
            if observations
            else []
        )
        if len(rows) > self.config.max_stream_observations:
            raise ValueError("state exceeds max_stream_observations")
        dimension = len(rows[0]) if rows else None
        if rows and any(len(row) != dimension for row in rows):
            raise ValueError("state.observations have inconsistent dimensions")
        declared_dimension = state.get("feature_dimension", dimension)
        if declared_dimension is not None and declared_dimension != dimension:
            raise ValueError("state feature_dimension does not match observations")
        if version == _STREAM_STATE_VERSION:
            if state.get("schema") != _STREAM_SCHEMA:
                raise ValueError("unsupported topology stream schema")
            if state.get("config_identity") != self.config_identity:
                raise ValueError("topology stream configuration identity mismatch")
            if state.get("backend_identity") != self.backend_identity:
                raise ValueError("topology stream backend identity mismatch")
        return {
            "version": _STREAM_STATE_VERSION,
            "schema": _STREAM_SCHEMA,
            "config_identity": self.config_identity,
            "backend_identity": self.backend_identity,
            "feature_dimension": dimension,
            "observations": [list(row) for row in rows],
        }

    def load_stream_state_dict(self, state: Mapping[str, Any]) -> None:
        candidate = self.validate_stream_state_dict(state)
        # All parsing and compatibility checks happen before either field changes.
        self._stream_observations = candidate["observations"]
        self._last_result = None

    @property
    def last_result(self) -> Optional[TopologyResult]:
        return self._last_result


def detect_topology_changes(observations: Any, config: Optional[Any] = None,
                            **kwargs: Any) -> TopologyResult:
    """Convenience wrapper returning a complete rolling detector result."""

    return RollingTopologyDetector(config, **kwargs).detect(observations)
