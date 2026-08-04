"""Bounded deterministic Vietoris--Rips and persistent-Laplacian reference.

This is a small exact reference backend for finite point clouds.  Persistence
pairing is over the declared coefficient field ``F2``; the persistent
Laplacian uses oriented real boundary matrices and a pinned float64 eigensolve.
It is intentionally capped for research goldens and small rolling clouds.  It
must not silently truncate a complex or be confused with the package's faster
graph-spectrum approximation.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import dataclass
from enum import Enum
from numbers import Integral
from typing import Any, Iterable, Sequence

_NUMPY_IMPORT_ERROR: ModuleNotFoundError | None
try:
    import numpy as np
except ModuleNotFoundError as exc:  # pragma: no cover - import-boundary behavior.
    np = None  # type: ignore[assignment]
    _NUMPY_IMPORT_ERROR = exc
else:
    _NUMPY_IMPORT_ERROR = None


MAX_PERSISTENT_VERTICES = 32
MAX_PERSISTENT_SIMPLICES = 4_096
MAX_PERSISTENT_BOUNDARY_NONZEROS = 200_000
MAX_PERSISTENT_EIGENVALUES = 64
MAX_PERSISTENT_DIMENSION = 1
_TOLERANCE_FLOOR = 1.0e-12


class PersistentStatus(str, Enum):
    VALID = "valid"
    RIGHT_CENSORED = "right_censored"
    INSUFFICIENT_HISTORY = "insufficient_history"
    DEGRADED = "degraded"


class PersistentTopologyError(ValueError):
    """Base error for invalid finite-filtration inputs."""


class PersistentResourceError(PersistentTopologyError):
    """The exact bounded reference would exceed its declared work budget."""


class PersistentNumericalError(PersistentTopologyError):
    """The real persistent-Laplacian computation failed its numerical policy."""


@dataclass(frozen=True, slots=True)
class PersistentLaplacianConfig:
    """Versioned finite-complex and solver policy."""

    max_vertices: int = 20
    max_homology_dimension: int = 1
    max_radius: float | None = None
    max_simplices: int = MAX_PERSISTENT_SIMPLICES
    max_boundary_nonzeros: int = MAX_PERSISTENT_BOUNDARY_NONZEROS
    q: int = 1
    scale_s: float | None = None
    scale_t: float | None = None
    n_eigenvalues: int = 8
    rank_tolerance: float = 1.0e-10
    eigen_residual_tolerance: float = 1.0e-8
    negative_eigenvalue_tolerance: float = 1.0e-8
    nullity_tolerance: float = 1.0e-8
    algorithm_version: str = "vr-f2-persistent-laplacian-v1"

    def __post_init__(self) -> None:
        def integer(name: str, value: Any, low: int, high: int) -> int:
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"{name} must be an integer")
            result = int(value)
            if not low <= result <= high:
                raise ValueError(f"{name} must be in [{low}, {high}]")
            return result

        max_vertices = integer("max_vertices", self.max_vertices, 2, MAX_PERSISTENT_VERTICES)
        dimension = integer(
            "max_homology_dimension",
            self.max_homology_dimension,
            0,
            MAX_PERSISTENT_DIMENSION,
        )
        max_simplices = integer(
            "max_simplices", self.max_simplices, 1, MAX_PERSISTENT_SIMPLICES
        )
        max_nonzeros = integer(
            "max_boundary_nonzeros",
            self.max_boundary_nonzeros,
            1,
            MAX_PERSISTENT_BOUNDARY_NONZEROS,
        )
        q = integer("q", self.q, 0, dimension)
        eigenvalues = integer(
            "n_eigenvalues", self.n_eigenvalues, 1, MAX_PERSISTENT_EIGENVALUES
        )
        max_radius = None
        if self.max_radius is not None:
            radius = float(self.max_radius)
            if not math.isfinite(radius) or radius <= 0.0:
                raise ValueError("max_radius must be finite and positive")
            max_radius = radius
        scales: dict[str, float | None] = {"scale_s": None, "scale_t": None}
        for name in scales:
            raw = getattr(self, name)
            if raw is not None:
                value = float(raw)
                if not math.isfinite(value) or value < 0.0:
                    raise ValueError(f"{name} must be finite and non-negative")
                scales[name] = value
        if scales["scale_s"] is not None and scales["scale_t"] is not None:
            if scales["scale_s"] > scales["scale_t"]:
                raise ValueError("scale_s must not exceed scale_t")
        tolerances: dict[str, float] = {}
        for name in (
            "rank_tolerance",
            "eigen_residual_tolerance",
            "negative_eigenvalue_tolerance",
            "nullity_tolerance",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            tolerances[name] = value
        if not isinstance(self.algorithm_version, str) or not self.algorithm_version:
            raise ValueError("algorithm_version must be a non-empty string")
        object.__setattr__(self, "max_vertices", max_vertices)
        object.__setattr__(self, "max_homology_dimension", dimension)
        object.__setattr__(self, "max_simplices", max_simplices)
        object.__setattr__(self, "max_boundary_nonzeros", max_nonzeros)
        object.__setattr__(self, "q", q)
        object.__setattr__(self, "n_eigenvalues", eigenvalues)
        object.__setattr__(self, "max_radius", max_radius)
        object.__setattr__(self, "scale_s", scales["scale_s"])
        object.__setattr__(self, "scale_t", scales["scale_t"])
        for name, value in tolerances.items():
            object.__setattr__(self, name, value)

    @property
    def identity(self) -> str:
        values = {
            "max_vertices": self.max_vertices,
            "max_homology_dimension": self.max_homology_dimension,
            "max_radius": self.max_radius,
            "max_simplices": self.max_simplices,
            "max_boundary_nonzeros": self.max_boundary_nonzeros,
            "q": self.q,
            "scale_s": self.scale_s,
            "scale_t": self.scale_t,
            "n_eigenvalues": self.n_eigenvalues,
            "rank_tolerance": self.rank_tolerance,
            "eigen_residual_tolerance": self.eigen_residual_tolerance,
            "negative_eigenvalue_tolerance": self.negative_eigenvalue_tolerance,
            "nullity_tolerance": self.nullity_tolerance,
            "algorithm_version": self.algorithm_version,
            "coefficient_field": "F2",
            "metric": "euclidean",
            "simplex_order": "birth,dimension,vertices",
        }
        return hashlib.sha256(
            json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class Simplex:
    vertices: tuple[int, ...]
    dimension: int
    birth: float


@dataclass(frozen=True, slots=True)
class PersistenceInterval:
    homology_dimension: int
    birth: float
    death: float | None
    birth_simplex: int
    death_simplex: int | None

    @property
    def persistence(self) -> float:
        if self.death is None:
            return math.inf
        return max(0.0, self.death - self.birth)

    def to_dict(self) -> dict[str, Any]:
        return {
            "homology_dimension": self.homology_dimension,
            "birth": self.birth,
            "death": self.death,
            "birth_simplex": self.birth_simplex,
            "death_simplex": self.death_simplex,
        }


@dataclass(frozen=True, slots=True)
class Filtration:
    points: tuple[tuple[float, ...], ...]
    simplices: tuple[Simplex, ...]
    distances: tuple[tuple[float, ...], ...]
    max_radius: float
    right_censored: bool
    digest: str
    boundary_nonzeros: int

    @property
    def status(self) -> PersistentStatus:
        return (
            PersistentStatus.RIGHT_CENSORED
            if self.right_censored
            else PersistentStatus.VALID
        )


@dataclass(frozen=True, slots=True)
class PersistentSpectrum:
    homology_dimension: int
    scale_s: float
    scale_t: float
    eigenvalues: tuple[float, ...]
    zero_multiplicity: int
    first_positive_eigenvalue: float
    trace: float
    positive_spectrum_entropy: float
    residual: float
    status: PersistentStatus

    def to_dict(self) -> dict[str, Any]:
        return {
            "homology_dimension": self.homology_dimension,
            "scale_s": self.scale_s,
            "scale_t": self.scale_t,
            "eigenvalues": list(self.eigenvalues),
            "zero_multiplicity": self.zero_multiplicity,
            "first_positive_eigenvalue": self.first_positive_eigenvalue,
            "trace": self.trace,
            "positive_spectrum_entropy": self.positive_spectrum_entropy,
            "residual": self.residual,
            "status": self.status.value,
        }


@dataclass(frozen=True, slots=True)
class PersistentLaplacianResult:
    """Complete bounded persistence and selected spectrum evidence."""

    config_identity: str
    filtration: Filtration
    intervals: tuple[PersistenceInterval, ...]
    spectrum: PersistentSpectrum
    status: PersistentStatus

    @property
    def filtration_digest(self) -> str:
        return self.filtration.digest

    @property
    def interval_digest(self) -> str:
        payload = [interval.to_dict() for interval in self.intervals]
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "finite_vr_persistent_laplacian",
            "config_identity": self.config_identity,
            "filtration_digest": self.filtration_digest,
            "interval_digest": self.interval_digest,
            "status": self.status.value,
            "right_censored": self.filtration.right_censored,
            "boundary_nonzeros": self.filtration.boundary_nonzeros,
            "intervals": [value.to_dict() for value in self.intervals],
            "spectrum": self.spectrum.to_dict(),
        }


def _points(value: Any, limit: int) -> tuple[tuple[float, ...], ...]:
    if isinstance(value, (str, bytes)):
        raise PersistentTopologyError("point cloud must be a numeric sequence")
    try:
        rows = list(value)
    except TypeError as exc:
        raise PersistentTopologyError("point cloud must be a numeric sequence") from exc
    if len(rows) < 2:
        raise PersistentTopologyError("point cloud needs at least two points")
    if len(rows) > limit:
        raise PersistentResourceError(f"point cloud exceeds max_vertices={limit}")
    result: list[tuple[float, ...]] = []
    dimension: int | None = None
    for row in rows:
        if isinstance(row, (str, bytes)):
            raise PersistentTopologyError("point rows must be numeric sequences")
        try:
            values = tuple(float(item) for item in row)
        except (TypeError, ValueError, OverflowError) as exc:
            raise PersistentTopologyError("point rows must be numeric sequences") from exc
        if not values or not all(math.isfinite(item) for item in values):
            raise PersistentTopologyError("point coordinates must be finite")
        if dimension is None:
            dimension = len(values)
        elif len(values) != dimension:
            raise PersistentTopologyError("point rows must have equal dimensions")
        result.append(values)
    # Coordinate sorting is the explicit test-adapter identity policy.  Stable
    # occurrence IDs remain distinct when coordinates are duplicated.
    return tuple(sorted(result))


def _distance_matrix(points: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
    matrix: list[list[float]] = [[0.0] * len(points) for _ in points]
    for left in range(len(points)):
        for right in range(left + 1, len(points)):
            distance = math.sqrt(
                sum(
                    (points[left][column] - points[right][column]) ** 2
                    for column in range(len(points[left]))
                )
            )
            if not math.isfinite(distance):
                raise PersistentTopologyError("distance computation is non-finite")
            matrix[left][right] = distance
            matrix[right][left] = distance
    return tuple(tuple(row) for row in matrix)


def _simplex_count(points: int, max_dimension: int) -> int:
    return sum(math.comb(points, dimension + 1) for dimension in range(max_dimension + 2))


def build_filtration(
    point_cloud: Any,
    *,
    config: PersistentLaplacianConfig | None = None,
) -> Filtration:
    """Build the exact bounded Euclidean Vietoris--Rips event complex."""

    settings = config or PersistentLaplacianConfig()
    points = _points(point_cloud, settings.max_vertices)
    distances = _distance_matrix(points)
    diameter = max((value for row in distances for value in row), default=0.0)
    max_radius = diameter if settings.max_radius is None else settings.max_radius
    right_censored = max_radius < diameter
    expected = _simplex_count(len(points), settings.max_homology_dimension)
    if expected > settings.max_simplices:
        raise PersistentResourceError(
            f"full bounded complex requires {expected} simplices, "
            f"exceeding max_simplices={settings.max_simplices}"
        )
    simplices: list[Simplex] = [
        Simplex((index,), 0, 0.0) for index in range(len(points))
    ]
    for dimension in range(1, settings.max_homology_dimension + 2):
        for vertices in itertools.combinations(range(len(points)), dimension + 1):
            birth = max(
                distances[left][right]
                for left, right in itertools.combinations(vertices, 2)
            )
            if birth <= max_radius:
                simplices.append(Simplex(vertices, dimension, float(birth)))
    simplices.sort(key=lambda item: (item.birth, item.dimension, item.vertices))
    index = {simplex.vertices: position for position, simplex in enumerate(simplices)}
    boundary_nonzeros = 0
    for simplex in simplices:
        if simplex.dimension == 0:
            continue
        for omitted in range(len(simplex.vertices)):
            face = simplex.vertices[:omitted] + simplex.vertices[omitted + 1 :]
            if face not in index or index[face] >= index[simplex.vertices]:
                raise PersistentTopologyError("filtration face order is invalid")
            boundary_nonzeros += 1
    if boundary_nonzeros > settings.max_boundary_nonzeros:
        raise PersistentResourceError(
            "filtration boundary exceeds max_boundary_nonzeros"
        )
    payload = {
        "points": points,
        "simplices": [
            (simplex.vertices, simplex.dimension, simplex.birth) for simplex in simplices
        ],
        "max_radius": max_radius,
        "right_censored": right_censored,
        "config_identity": settings.identity,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return Filtration(
        points=points,
        simplices=tuple(simplices),
        distances=distances,
        max_radius=float(max_radius),
        right_censored=right_censored,
        digest=digest,
        boundary_nonzeros=boundary_nonzeros,
    )


def _boundary_bitsets(filtration: Filtration) -> list[int]:
    indices = {simplex.vertices: position for position, simplex in enumerate(filtration.simplices)}
    result: list[int] = []
    for simplex in filtration.simplices:
        if simplex.dimension == 0:
            result.append(0)
            continue
        column = 0
        for omitted in range(len(simplex.vertices)):
            face = simplex.vertices[:omitted] + simplex.vertices[omitted + 1 :]
            column ^= 1 << indices[face]
        result.append(column)
    return result


def compute_persistence(
    filtration: Filtration,
    *,
    dimensions: Iterable[int] = (0, 1),
) -> tuple[PersistenceInterval, ...]:
    """Compute exact F2 intervals for the finite filtration."""

    requested = {int(value) for value in dimensions}
    if any(value < 0 for value in requested):
        raise ValueError("homology dimensions must be non-negative")
    boundaries = _boundary_bitsets(filtration)
    reduced: dict[int, int] = {}
    paired_death: dict[int, int] = {}
    zero_columns: set[int] = set()
    for column_index, boundary in enumerate(boundaries):
        column = boundary
        while column:
            pivot = column.bit_length() - 1
            previous = reduced.get(pivot)
            if previous is None:
                break
            column ^= previous
        if column == 0:
            zero_columns.add(column_index)
        else:
            pivot = column.bit_length() - 1
            reduced[pivot] = column
            paired_death[pivot] = column_index
    intervals: list[PersistenceInterval] = []
    simplices = filtration.simplices
    for birth_index in sorted(zero_columns):
        dimension = simplices[birth_index].dimension
        if dimension not in requested:
            continue
        death_index = paired_death.get(birth_index)
        intervals.append(
            PersistenceInterval(
                homology_dimension=dimension,
                birth=simplices[birth_index].birth,
                death=None if death_index is None else simplices[death_index].birth,
                birth_simplex=birth_index,
                death_simplex=death_index,
            )
        )
    intervals.sort(
        key=lambda item: (
            item.homology_dimension,
            item.birth,
            math.inf if item.death is None else item.death,
            item.birth_simplex,
        )
    )
    return tuple(intervals)


def _oriented_boundary(
    simplices: Sequence[Simplex],
    *,
    source_dimension: int,
    target_dimension: int,
) -> Any:
    if np is None:
        raise ImportError("persistent Laplacian spectra require NumPy") from _NUMPY_IMPORT_ERROR
    source = [simplex for simplex in simplices if simplex.dimension == source_dimension]
    target_simplices = [
        simplex for simplex in simplices if simplex.dimension == target_dimension
    ]
    target = {
        simplex.vertices: index for index, simplex in enumerate(target_simplices)
    }
    matrix = np.zeros((len(target), len(source)), dtype=float)
    for column, simplex in enumerate(source):
        for omitted in range(len(simplex.vertices)):
            face = simplex.vertices[:omitted] + simplex.vertices[omitted + 1 :]
            row = target.get(face)
            if row is None:
                raise PersistentNumericalError("boundary face is missing")
            matrix[row, column] = -1.0 if omitted % 2 else 1.0
    return matrix


def _nullspace(matrix: Any, tolerance: float) -> Any:
    if matrix.shape[1] == 0:
        return np.zeros((0, 0), dtype=float)
    if matrix.shape[0] == 0:
        return np.eye(matrix.shape[1], dtype=float)
    _, singular, vh = np.linalg.svd(matrix, full_matrices=True)
    scale = float(singular[0]) if singular.size else 0.0
    threshold = max(tolerance, tolerance * scale)
    rank = int(np.count_nonzero(singular > threshold))
    return vh[rank:, :].T.copy()


def persistent_laplacian_spectrum(
    filtration: Filtration,
    *,
    config: PersistentLaplacianConfig | None = None,
) -> PersistentSpectrum:
    """Compute the nullspace-restricted persistent q-Laplacian spectrum."""

    if np is None:
        raise ImportError("persistent Laplacian spectra require NumPy") from _NUMPY_IMPORT_ERROR
    settings = config or PersistentLaplacianConfig()
    q = settings.q
    births = sorted(
        {simplex.birth for simplex in filtration.simplices if simplex.dimension == 1}
    )
    default_s = births[0] if births else 0.0
    scale_s = default_s if settings.scale_s is None else float(settings.scale_s)
    scale_t = filtration.max_radius if settings.scale_t is None else float(settings.scale_t)
    if not math.isfinite(scale_s) or not math.isfinite(scale_t) or scale_s > scale_t:
        raise ValueError("scale_s and scale_t must be finite with scale_s <= scale_t")
    Ks = [simplex for simplex in filtration.simplices if simplex.birth <= scale_s]
    Kt = [simplex for simplex in filtration.simplices if simplex.birth <= scale_t]
    if not Ks or not Kt:
        raise PersistentTopologyError("selected filtration pair is empty")
    q_s = [simplex for simplex in Ks if simplex.dimension == q]
    q_t = [simplex for simplex in Kt if simplex.dimension == q]
    if not q_s:
        # A zero-dimensional chain space has an empty spectrum, not fake 2.0
        # padding.  The result is explicitly insufficient for this pair.
        return PersistentSpectrum(
            homology_dimension=q,
            scale_s=scale_s,
            scale_t=scale_t,
            eigenvalues=(),
            zero_multiplicity=0,
            first_positive_eigenvalue=0.0,
            trace=0.0,
            positive_spectrum_entropy=0.0,
            residual=0.0,
            status=PersistentStatus.INSUFFICIENT_HISTORY,
        )
    Bq = _oriented_boundary(Ks, source_dimension=q, target_dimension=q - 1) if q else np.zeros((0, len(q_s)))
    q_t_index = {simplex.vertices: index for index, simplex in enumerate(q_t)}
    Bt = _oriented_boundary(Kt, source_dimension=q + 1, target_dimension=q)
    in_rows = [q_t_index[simplex.vertices] for simplex in q_s]
    out_rows = [index for index, simplex in enumerate(q_t) if simplex.birth > scale_s]
    E = Bt[out_rows, :] if out_rows else np.zeros((0, Bt.shape[1]), dtype=float)
    Z = _nullspace(E, settings.rank_tolerance)
    if E.size and not np.allclose(E @ Z, 0.0, atol=settings.eigen_residual_tolerance):
        raise PersistentNumericalError("persistent boundary nullspace residual is too large")
    A = Bt[in_rows, :] @ Z
    if Z.shape[1] == 0:
        up = np.zeros((len(q_s), len(q_s)), dtype=float)
    else:
        gram = Z.T @ Z
        try:
            up = A @ np.linalg.solve(gram, A.T)
        except np.linalg.LinAlgError as exc:
            raise PersistentNumericalError("persistent nullspace Gram matrix is singular") from exc
    down = Bq.T @ Bq
    operator = (down + up + (down + up).T) / 2.0
    try:
        values, vectors = np.linalg.eigh(operator)
    except np.linalg.LinAlgError as exc:
        raise PersistentNumericalError("persistent Laplacian eigensolver failed") from exc
    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values)):
        raise PersistentNumericalError("persistent Laplacian spectrum is non-finite")
    if values.size and float(values.min()) < -settings.negative_eigenvalue_tolerance:
        raise PersistentNumericalError("persistent Laplacian has a negative eigenvalue")
    values[values < 0.0] = 0.0
    residual = float(np.linalg.norm(operator @ vectors - vectors * values, ord="fro"))
    if not math.isfinite(residual) or residual > settings.eigen_residual_tolerance * max(1.0, float(np.linalg.norm(operator))):
        raise PersistentNumericalError("persistent Laplacian eigen residual is too large")
    zero_count = int(np.count_nonzero(values <= settings.nullity_tolerance))
    positive = [float(value) for value in values if value > settings.nullity_tolerance]
    first_positive = positive[0] if positive else 0.0
    total = sum(positive)
    entropy = 0.0
    if len(positive) > 1 and total > 0.0:
        entropy = -sum((value / total) * math.log(value / total) for value in positive)
        entropy /= math.log(len(positive))
    return PersistentSpectrum(
        homology_dimension=q,
        scale_s=scale_s,
        scale_t=scale_t,
        eigenvalues=tuple(float(value) for value in values[: settings.n_eigenvalues]),
        zero_multiplicity=zero_count,
        first_positive_eigenvalue=first_positive,
        trace=float(np.trace(operator)),
        positive_spectrum_entropy=float(max(0.0, min(1.0, entropy))),
        residual=residual,
        status=filtration.status,
    )


def compute_persistent_laplacian(
    point_cloud: Any,
    *,
    config: PersistentLaplacianConfig | None = None,
) -> PersistentLaplacianResult:
    """Build the finite filtration, intervals, and persistent spectrum."""

    settings = config or PersistentLaplacianConfig()
    filtration = build_filtration(point_cloud, config=settings)
    intervals = compute_persistence(filtration, dimensions=range(settings.max_homology_dimension + 1))
    spectrum = persistent_laplacian_spectrum(filtration, config=settings)
    status = filtration.status
    if spectrum.status is PersistentStatus.INSUFFICIENT_HISTORY:
        status = spectrum.status
    return PersistentLaplacianResult(
        config_identity=settings.identity,
        filtration=filtration,
        intervals=intervals,
        spectrum=spectrum,
        status=status,
    )


def persistent_laplacian_backend(
    point_cloud: Any,
    n_eigenvalues: int = 8,
) -> tuple[float, ...]:
    """Compatibility adapter for ``TopologyConfig``'s callable seam.

    The complete evidence is available through :func:`compute_persistent_laplacian`;
    this adapter returns only the selected q-spectrum required by the legacy
    graph-summary interface and therefore cannot carry the full result.
    """

    if isinstance(n_eigenvalues, bool) or not isinstance(n_eigenvalues, Integral):
        raise ValueError("n_eigenvalues must be an integer")
    settings = PersistentLaplacianConfig(n_eigenvalues=int(n_eigenvalues))
    result = compute_persistent_laplacian(point_cloud, config=settings)
    if result.status is not PersistentStatus.VALID:
        raise PersistentResourceError(f"persistent backend status is {result.status.value}")
    return result.spectrum.eigenvalues


__all__ = [
    "Filtration",
    "MAX_PERSISTENT_BOUNDARY_NONZEROS",
    "MAX_PERSISTENT_DIMENSION",
    "MAX_PERSISTENT_EIGENVALUES",
    "MAX_PERSISTENT_SIMPLICES",
    "MAX_PERSISTENT_VERTICES",
    "PersistentLaplacianConfig",
    "PersistentLaplacianResult",
    "PersistentNumericalError",
    "PersistentResourceError",
    "PersistentSpectrum",
    "PersistentStatus",
    "PersistentTopologyError",
    "PersistenceInterval",
    "Simplex",
    "build_filtration",
    "compute_persistence",
    "compute_persistent_laplacian",
    "persistent_laplacian_backend",
    "persistent_laplacian_spectrum",
]
