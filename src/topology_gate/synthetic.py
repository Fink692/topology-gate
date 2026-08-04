"""Deterministic synthetic data for offline topology-gate experiments.

The module has no data-provider or network dependencies.  A generated data set
contains time-indexed observations, delayed labels, realized returns, the
latent regime, and an evaluation-only oracle position.  The legacy
``generate_regime_switching`` entry point is retained as a compatibility
fixture for callers that use continuous synthetic targets.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Iterable, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from .config import (
    MAX_CONFIG_ABS_FLOAT,
    ConfigurationError,
    DataValidationError,
    ResourceLimitError,
)

MAX_SYNTHETIC_SAMPLES = 1_000_000
MAX_SYNTHETIC_FEATURES = 512
MAX_MATRIX_ELEMENTS = 8_000_000
MAX_INDEX_ITEMS = 1_000_000
MAX_CHANGE_POINTS = 100_000
MAX_SEED = (1 << 63) - 1


def _bounded_items(values: Iterable[Any], name: str, maximum: int) -> Tuple[Any, ...]:
    """Copy an iterable while stopping before an oversized materialization."""

    if isinstance(values, (str, bytes, bytearray)):
        raise DataValidationError(f"{name} must be a finite sequence, not text")
    try:
        expected = len(values)  # type: ignore[arg-type]
    except TypeError:
        expected = None
    if expected is not None and expected > maximum:
        raise ResourceLimitError(f"{name} has {expected} items; limit is {maximum}")
    result: list[Any] = []
    try:
        for item in values:
            if len(result) >= maximum:
                raise ResourceLimitError(f"{name} exceeds the practical limit {maximum}")
            result.append(item)
    except TypeError as exc:
        raise DataValidationError(f"{name} must be an iterable sequence") from exc
    return tuple(result)


def _bounded_int(value: Any, name: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ConfigurationError(f"{name} must be an integer")
    converted = int(value)
    if converted < minimum:
        raise ConfigurationError(f"{name} must be at least {minimum}")
    if converted > maximum:
        raise ResourceLimitError(f"{name}={converted} exceeds the practical limit {maximum}")
    return converted


def _finite_float(value: Any, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, (bool, str, bytes, bytearray)):
        raise ConfigurationError(f"{name} must be a finite real number")
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ConfigurationError(f"{name} must be a finite real number") from exc
    if not math.isfinite(converted):
        raise ConfigurationError(f"{name} must be finite")
    if abs(converted) > MAX_CONFIG_ABS_FLOAT:
        raise ResourceLimitError(
            f"{name} magnitude exceeds the configured limit ({MAX_CONFIG_ABS_FLOAT:g})"
        )
    if minimum is not None and converted < minimum:
        raise ConfigurationError(f"{name} must be at least {minimum}")
    return converted


def _checked_elements(rows: int, columns: int, name: str) -> None:
    if rows > MAX_SYNTHETIC_SAMPLES:
        raise ResourceLimitError(
            f"{name} has {rows} rows; limit is {MAX_SYNTHETIC_SAMPLES}"
        )
    if columns > MAX_SYNTHETIC_FEATURES:
        raise ResourceLimitError(
            f"{name} has {columns} columns; limit is {MAX_SYNTHETIC_FEATURES}"
        )
    if rows > MAX_MATRIX_ELEMENTS // max(columns, 1):
        raise ResourceLimitError(
            f"{name} has {rows * columns} elements; limit is {MAX_MATRIX_ELEMENTS}"
        )


def _preflight_matrix(values: Any, name: str) -> None:
    """Check observable shape before calling ``numpy.asarray``."""

    shape = getattr(values, "shape", None)
    if shape is not None:
        try:
            dimensions = tuple(int(dimension) for dimension in shape)
        except (TypeError, ValueError, OverflowError) as exc:
            raise DataValidationError(f"{name} has an invalid shape") from exc
        if len(dimensions) not in {1, 2}:
            raise DataValidationError(f"{name} must be one- or two-dimensional")
        rows = dimensions[0]
        columns = 1 if len(dimensions) == 1 else dimensions[1]
        if rows < 0 or columns < 0:
            raise DataValidationError(f"{name} has a negative dimension")
        _checked_elements(rows, columns, name)
        return

    try:
        rows = len(values)
    except TypeError:
        return
    if rows > MAX_SYNTHETIC_SAMPLES:
        raise ResourceLimitError(
            f"{name} has {rows} rows; limit is {MAX_SYNTHETIC_SAMPLES}"
        )
    if rows == 0:
        return
    try:
        first = values[0]
    except (IndexError, KeyError, TypeError):
        return
    if isinstance(first, (str, bytes, bytearray)):
        _checked_elements(rows, 1, name)
        return
    try:
        columns = len(first)
    except TypeError:
        _checked_elements(rows, 1, name)
        return
    if columns > MAX_SYNTHETIC_FEATURES:
        raise ResourceLimitError(
            f"{name} has {columns} columns; limit is {MAX_SYNTHETIC_FEATURES}"
        )
    _checked_elements(rows, columns, name)
    for row_number, row in enumerate(values):
        if row_number >= MAX_SYNTHETIC_SAMPLES:
            raise ResourceLimitError(f"{name} exceeds the row limit")
        try:
            row_length = len(row)
        except TypeError as exc:
            raise DataValidationError(f"{name}[{row_number}] must be a sequence") from exc
        if row_length != columns:
            raise DataValidationError(
                f"{name} rows must have equal length; row {row_number} has {row_length}, expected {columns}"
            )


def _preflight_vector(values: Any, name: str) -> None:
    shape = getattr(values, "shape", None)
    if shape is not None:
        try:
            dimensions = tuple(int(dimension) for dimension in shape)
        except (TypeError, ValueError, OverflowError) as exc:
            raise DataValidationError(f"{name} has an invalid shape") from exc
        if len(dimensions) == 2 and 1 in dimensions:
            length = max(dimensions)
        elif len(dimensions) == 1:
            length = dimensions[0]
        else:
            raise DataValidationError(f"{name} must be one-dimensional")
        if length > MAX_SYNTHETIC_SAMPLES:
            raise ResourceLimitError(
                f"{name} has {length} entries; limit is {MAX_SYNTHETIC_SAMPLES}"
            )
        return
    try:
        length = len(values)
    except TypeError:
        return
    if length > MAX_SYNTHETIC_SAMPLES:
        raise ResourceLimitError(
            f"{name} has {length} entries; limit is {MAX_SYNTHETIC_SAMPLES}"
        )


def _finite_array(values: Any, name: str, *, ndim: int | None = None) -> NDArray[Any]:
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DataValidationError(f"{name} must contain numeric values") from exc
    if ndim is not None and array.ndim != ndim:
        raise DataValidationError(f"{name} must be {ndim}-dimensional")
    if not np.all(np.isfinite(array)):
        raise DataValidationError(f"{name} must contain only finite values")
    if array.size and float(np.max(np.abs(array))) > MAX_CONFIG_ABS_FLOAT:
        raise ResourceLimitError(
            f"{name} contains a magnitude above {MAX_CONFIG_ABS_FLOAT:g}"
        )
    return np.array(array, dtype=float, copy=True)

def _as_tuple(values: Iterable[Any]) -> Tuple[Any, ...]:
    return _bounded_items(values, "sequence", MAX_INDEX_ITEMS)


def _validate_index(index: Sequence[Any], expected_length: int, name: str) -> Tuple[Any, ...]:
    result = _bounded_items(index, name, MAX_INDEX_ITEMS)
    if len(result) != expected_length:
        raise ValueError(f"{name} must have {expected_length} entries, got {len(result)}")
    try:
        if len(set(result)) != len(result):
            raise ValueError(f"{name} must contain unique time values")
    except TypeError as exc:
        raise TypeError(f"{name} values must be hashable") from exc
    try:
        if any(not (left < right) for left, right in zip(result, result[1:])):
            raise ValueError(f"{name} must be strictly increasing")
    except TypeError as exc:
        raise TypeError(f"{name} values must be orderable") from exc
    for position, value in enumerate(result):
        if isinstance(value, Real) and not math.isfinite(float(value)):
            raise DataValidationError(f"{name}[{position}] must be finite")
    return result


@dataclass(frozen=True)
class TimeIndexedFeatures:
    """A two-dimensional feature matrix with an explicit chronological index."""

    index: Tuple[Any, ...]
    values: NDArray[Any]
    columns: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _preflight_matrix(self.values, "features")
        array = _finite_array(self.values, "features")
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        if array.ndim != 2:
            raise DataValidationError("features must be a one- or two-dimensional array")
        index = _validate_index(self.index, array.shape[0], "feature index")
        columns = _bounded_items(self.columns, "feature columns", MAX_SYNTHETIC_FEATURES)
        if not columns:
            columns = tuple(f"feature_{i}" for i in range(array.shape[1]))
        if len(columns) != array.shape[1]:
            raise DataValidationError(
                f"columns must have {array.shape[1]} entries, got {len(columns)}"
            )
        if any(not isinstance(column, str) or not column for column in columns):
            raise DataValidationError("feature columns must be non-empty strings")
        if len(set(columns)) != len(columns):
            raise DataValidationError("columns must be unique")
        object.__setattr__(self, "index", index)
        object.__setattr__(self, "values", np.array(array, dtype=float, copy=True))
        object.__setattr__(self, "columns", columns)

    @classmethod
    def from_array(
        cls,
        values: Sequence[Sequence[float]] | Sequence[float] | NDArray[Any],
        index: Optional[Sequence[Any]] = None,
        columns: Optional[Sequence[str]] = None,
    ) -> "TimeIndexedFeatures":
        _preflight_matrix(values, "features")
        try:
            array = np.asarray(values)
        except (TypeError, ValueError, OverflowError) as exc:
            raise DataValidationError("features must be array-like") from exc
        if array.ndim == 0:
            array = array.reshape(1, 1)
        if array.ndim not in {1, 2}:
            raise DataValidationError("features must be one- or two-dimensional")
        n_rows = array.shape[0]
        if index is None:
            index = tuple(range(n_rows))
        return cls(tuple(index), array, tuple(columns or ()))

    @property
    def n_samples(self) -> int:
        return self.values.shape[0]

    @property
    def n_features(self) -> int:
        return self.values.shape[1]

    @property
    def shape(self) -> Tuple[int, int]:
        """Array-compatible shape for legacy callers."""

        return (int(self.values.shape[0]), int(self.values.shape[1]))

    @property
    def times(self) -> Tuple[Any, ...]:
        return self.index

    @property
    def timestamps(self) -> Tuple[Any, ...]:
        return self.index

    @property
    def data(self) -> NDArray[Any]:
        return np.array(self.values, copy=True)

    def __len__(self) -> int:
        return self.n_samples

    def __array__(self, dtype: Any = None) -> NDArray[Any]:
        return np.asarray(self.values, dtype=dtype)

    def __getitem__(self, item: Any) -> Any:
        return self.values[item]

    def row(self, position: int) -> NDArray[Any]:
        return np.array(self.values[position], dtype=float, copy=True)

    def slice(self, start: Optional[int] = None, stop: Optional[int] = None) -> "TimeIndexedFeatures":
        return TimeIndexedFeatures(
            self.index[slice(start, stop)],
            self.values[slice(start, stop)],
            self.columns,
        )


@dataclass(frozen=True)
class TimeIndexedLabels:
    """Labels and the time at which each label becomes available.

    ``available_at`` can contain actual values from the feature index or
    integer row positions.  Integer positions beyond the sample end represent
    labels that never become available during the finite replay.
    """

    index: Tuple[Any, ...]
    values: NDArray[Any]
    available_at: Optional[Tuple[Any, ...]] = None
    name: str = "label"

    def __post_init__(self) -> None:
        _preflight_vector(self.values, "labels")
        array = _finite_array(self.values, "labels")
        if array.ndim == 2 and 1 in array.shape:
            array = array.reshape(-1)
        if array.ndim != 1:
            raise DataValidationError("labels must be a one-dimensional array")
        index = _validate_index(self.index, array.shape[0], "label index")
        availability = self.available_at
        if availability is not None:
            availability = _bounded_items(
                availability, "label availability", MAX_INDEX_ITEMS
            )
            if len(availability) != len(index):
                raise DataValidationError(
                    f"available_at must have {len(index)} entries, got {len(availability)}"
                )
        if not self.name:
            raise DataValidationError("label name must not be empty")
        object.__setattr__(self, "index", index)
        object.__setattr__(self, "values", np.array(array, copy=True))
        object.__setattr__(self, "available_at", availability)

    @classmethod
    def from_array(
        cls,
        values: Sequence[Any] | NDArray[Any],
        index: Optional[Sequence[Any]] = None,
        available_at: Optional[Sequence[Any]] = None,
        name: str = "label",
    ) -> "TimeIndexedLabels":
        _preflight_vector(values, "labels")
        try:
            array = np.asarray(values)
        except (TypeError, ValueError, OverflowError) as exc:
            raise DataValidationError("labels must be array-like") from exc
        if array.ndim == 0:
            array = array.reshape(1)
        if array.ndim == 2 and 1 in array.shape:
            array = array.reshape(-1)
        if array.ndim != 1:
            raise DataValidationError("labels must be a one-dimensional array")
        n_rows = array.shape[0]
        if index is None:
            index = tuple(range(n_rows))
        return cls(tuple(index), array, None if available_at is None else tuple(available_at), name)

    @property
    def n_samples(self) -> int:
        return self.values.shape[0]

    @property
    def times(self) -> Tuple[Any, ...]:
        return self.index

    @property
    def timestamps(self) -> Tuple[Any, ...]:
        return self.index

    @property
    def label_available_at(self) -> Optional[Tuple[Any, ...]]:
        return self.available_at

    @property
    def availability(self) -> Optional[Tuple[Any, ...]]:
        return self.available_at

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, item: Any) -> Any:
        return self.values[item]

    def slice(self, start: Optional[int] = None, stop: Optional[int] = None) -> "TimeIndexedLabels":
        selected = slice(start, stop)
        availability = None if self.available_at is None else self.available_at[selected]
        return TimeIndexedLabels(self.index[selected], self.values[selected], availability, self.name)


@dataclass(frozen=True)
class SyntheticDataset:
    """A finite synthetic regime process and its evaluation-only ground truth."""

    features: TimeIndexedFeatures
    labels: TimeIndexedLabels
    realized_returns: NDArray[Any]
    optimal_position: NDArray[Any]
    regime_ids: NDArray[Any]
    change_points: Tuple[int, ...]
    expected_returns: NDArray[Any]
    seed: int
    label_delay: int = 0

    def __post_init__(self) -> None:
        n = self.features.n_samples
        if self.labels.index != self.features.index:
            raise DataValidationError("features and labels must use the same time index")
        if n > MAX_SYNTHETIC_SAMPLES:
            raise ResourceLimitError(
                f"dataset has {n} samples; limit is {MAX_SYNTHETIC_SAMPLES}"
            )
        for name in ("realized_returns", "optimal_position", "regime_ids", "expected_returns"):
            _preflight_vector(getattr(self, name), name)
            value = _finite_array(getattr(self, name), name)
            if value.ndim != 1 or len(value) != n:
                raise DataValidationError(f"{name} must be a one-dimensional array of length {n}")
            object.__setattr__(self, name, np.array(value, copy=True))
        raw_points = _bounded_items(
            self.change_points, "change_points", MAX_CHANGE_POINTS
        )
        points: Tuple[int, ...] = tuple(
            _bounded_int(point, "change point", minimum=1, maximum=max(n - 1, 1))
            for point in raw_points
        )
        if any(point <= 0 or point >= n for point in points):
            raise DataValidationError("change points must be strictly inside the sample")
        if tuple(sorted(set(points))) != points:
            raise DataValidationError("change points must be sorted and unique")
        object.__setattr__(self, "change_points", points)
        seed = _bounded_int(self.seed, "seed", minimum=0, maximum=MAX_SEED)
        label_delay = _bounded_int(
            self.label_delay,
            "label_delay",
            minimum=0,
            maximum=MAX_SYNTHETIC_SAMPLES,
        )
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "label_delay", label_delay)

    @property
    def n_samples(self) -> int:
        return self.features.n_samples

    @property
    def returns(self) -> NDArray[Any]:
        return np.array(self.realized_returns, copy=True)

    @property
    def targets(self) -> NDArray[Any]:
        """Legacy alias for the label stream."""

        return np.array(self.labels.values, copy=True)

    @property
    def market_states(self) -> NDArray[Any]:
        """Legacy alias for the feature matrix."""

        return np.array(self.features.values, copy=True)

    @property
    def regimes(self) -> NDArray[Any]:
        return np.array(self.regime_ids, copy=True)

    @property
    def shifts(self) -> Tuple[int, ...]:
        return self.change_points

    @property
    def shift_points(self) -> Tuple[int, ...]:
        """Legacy alias for known regime shifts."""

        return self.change_points


@dataclass(frozen=True)
class SyntheticRegimeConfig:
    """Configuration for :class:`SyntheticRegimeProcess`."""

    n_steps: int = 240
    n_features: int = 3
    change_points: Optional[Tuple[int, ...]] = None
    seed: int = 7
    label_delay: int = 1
    feature_noise: float = 0.20
    return_noise: float = 0.05
    signal_strength: float = 1.0
    return_magnitude: float = 0.02
    regime_signs: Optional[Tuple[float, ...]] = None

    def __post_init__(self) -> None:
        n_steps = _bounded_int(
            self.n_steps, "n_steps", minimum=2, maximum=MAX_SYNTHETIC_SAMPLES
        )
        n_features = _bounded_int(
            self.n_features, "n_features", minimum=1, maximum=MAX_SYNTHETIC_FEATURES
        )
        if n_steps > MAX_MATRIX_ELEMENTS // n_features:
            raise ResourceLimitError(
                f"n_steps*n_features exceeds the practical element limit {MAX_MATRIX_ELEMENTS}"
            )
        seed = _bounded_int(self.seed, "seed", minimum=0, maximum=MAX_SEED)
        label_delay = _bounded_int(
            self.label_delay, "label_delay", minimum=0, maximum=MAX_SYNTHETIC_SAMPLES
        )
        feature_noise = _finite_float(self.feature_noise, "feature_noise", minimum=0.0)
        return_noise = _finite_float(self.return_noise, "return_noise", minimum=0.0)
        signal_strength = _finite_float(self.signal_strength, "signal_strength")
        return_magnitude = _finite_float(
            self.return_magnitude, "return_magnitude", minimum=0.0
        )
        points = None
        if self.change_points is not None:
            points = _bounded_items(
                self.change_points, "change_points", MAX_CHANGE_POINTS
            )
        signs = None
        if self.regime_signs is not None:
            signs = _bounded_items(self.regime_signs, "regime_signs", MAX_CHANGE_POINTS + 1)
        object.__setattr__(self, "n_steps", n_steps)
        object.__setattr__(self, "n_features", n_features)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "label_delay", label_delay)
        object.__setattr__(self, "feature_noise", feature_noise)
        object.__setattr__(self, "return_noise", return_noise)
        object.__setattr__(self, "signal_strength", signal_strength)
        object.__setattr__(self, "return_magnitude", return_magnitude)
        object.__setattr__(self, "change_points", None if points is None else tuple(points))
        object.__setattr__(self, "regime_signs", None if signs is None else tuple(signs))


@dataclass(frozen=True)
class _LegacyRegimeConfig:
    n_samples: int = 512
    n_features: int = 4
    shift_points: Tuple[int, ...] = (160, 320)
    seed: int = 7
    noise_scale: float = 0.25

    def __post_init__(self) -> None:
        _bounded_int(
            self.n_samples, "n_samples", minimum=32, maximum=MAX_SYNTHETIC_SAMPLES
        )
        _bounded_int(
            self.n_features, "n_features", minimum=1, maximum=MAX_SYNTHETIC_FEATURES
        )
        if self.n_samples > MAX_MATRIX_ELEMENTS // max(self.n_features, 1):
            raise ResourceLimitError(
                f"n_samples*n_features exceeds the practical element limit {MAX_MATRIX_ELEMENTS}"
            )
        points = _bounded_items(self.shift_points, "shift_points", MAX_CHANGE_POINTS)
        _bounded_int(self.seed, "seed", minimum=0, maximum=MAX_SEED)
        _finite_float(self.noise_scale, "noise_scale", minimum=0.0)
        object.__setattr__(self, "n_samples", int(self.n_samples))
        object.__setattr__(self, "n_features", int(self.n_features))
        object.__setattr__(self, "shift_points", tuple(points))
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "noise_scale", float(self.noise_scale))


def _normalise_change_points(n_steps: int, change_points: Optional[Sequence[int]]) -> Tuple[int, ...]:
    if n_steps < 2:
        raise ConfigurationError("n_steps must be at least 2")
    if change_points is None:
        candidates = (n_steps // 3, (2 * n_steps) // 3)
        points = tuple(point for point in candidates if 0 < point < n_steps)
    else:
        raw_points = _bounded_items(change_points, "change_points", MAX_CHANGE_POINTS)
        points = tuple(
            _bounded_int(point, "change point", minimum=1, maximum=n_steps - 1)
            for point in raw_points
        )
    if tuple(sorted(set(points))) != points:
        raise DataValidationError("change_points must be sorted and unique")
    if any(point <= 0 or point >= n_steps for point in points):
        raise DataValidationError("change points must lie in (0, n_steps)")
    return points


def generate_synthetic_regimes(
    n_steps: int = 240,
    n_features: int = 3,
    change_points: Optional[Sequence[int]] = None,
    seed: int = 7,
    label_delay: int = 1,
    feature_noise: float = 0.20,
    return_noise: float = 0.05,
    signal_strength: float = 1.0,
    return_magnitude: float = 0.02,
    regime_signs: Optional[Sequence[float]] = None,
    index: Optional[Sequence[Any]] = None,
) -> SyntheticDataset:
    """Generate a deterministic piecewise-stationary binary regime process."""

    n_steps = _bounded_int(
        n_steps, "n_steps", minimum=2, maximum=MAX_SYNTHETIC_SAMPLES
    )
    n_features = _bounded_int(
        n_features, "n_features", minimum=1, maximum=MAX_SYNTHETIC_FEATURES
    )
    if n_steps > MAX_MATRIX_ELEMENTS // n_features:
        raise ResourceLimitError(
            f"n_steps*n_features exceeds the practical element limit {MAX_MATRIX_ELEMENTS}"
        )
    label_delay = _bounded_int(
        label_delay, "label_delay", minimum=0, maximum=MAX_SYNTHETIC_SAMPLES
    )
    seed = _bounded_int(seed, "seed", minimum=0, maximum=MAX_SEED)
    feature_noise = _finite_float(feature_noise, "feature_noise", minimum=0.0)
    return_noise = _finite_float(return_noise, "return_noise", minimum=0.0)
    signal_strength = _finite_float(signal_strength, "signal_strength")
    return_magnitude = _finite_float(
        return_magnitude, "return_magnitude", minimum=0.0
    )

    points = _normalise_change_points(n_steps, change_points)
    n_regimes = len(points) + 1
    if regime_signs is None:
        signs = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(n_regimes)])
    else:
        raw_signs = _bounded_items(regime_signs, "regime_signs", MAX_CHANGE_POINTS + 1)
        try:
            signs = np.asarray(raw_signs, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise DataValidationError("regime_signs must contain numeric values") from exc
        if signs.ndim != 1 or len(signs) != n_regimes:
            raise DataValidationError(f"regime_signs must have {n_regimes} entries")
        if np.any(~np.isfinite(signs)) or np.any(signs == 0):
            raise DataValidationError("regime_signs must contain finite non-zero values")
        signs = np.sign(signs)

    boundaries = (0,) + points + (n_steps,)
    regime_ids = np.empty(n_steps, dtype=int)
    optimal_position = np.empty(n_steps, dtype=float)
    for regime_id, (start, stop) in enumerate(zip(boundaries, boundaries[1:])):
        regime_ids[start:stop] = regime_id
        optimal_position[start:stop] = signs[regime_id]

    rng = np.random.default_rng(seed)
    values = rng.normal(0.0, feature_noise, size=(n_steps, n_features))
    values[:, 0] += signal_strength * optimal_position
    if n_features > 1:
        values[:, 1] += 0.35 * optimal_position

    # Keep the realized-return sign fixed in this control-layer fixture.  The
    # latent optimal position still changes at each regime boundary, which
    # allows detection-delay tests to remain independent from the deliberately
    # adversarial early-promotion/false-promotion path.
    expected_returns = np.full(n_steps, float(return_magnitude), dtype=float)
    realized_returns = expected_returns + rng.normal(0.0, return_noise, size=n_steps)
    availability = tuple(int(position + label_delay) for position in range(n_steps))
    if index is None:
        time_index: Tuple[Any, ...] = tuple(range(n_steps))
    else:
        time_index = _bounded_items(index, "index", MAX_INDEX_ITEMS)
        if len(time_index) != n_steps:
            raise DataValidationError(f"index must have {n_steps} entries")

    features = TimeIndexedFeatures(time_index, values)
    labels = TimeIndexedLabels(
        time_index,
        np.array(optimal_position, copy=True),
        availability,
        name="regime_direction",
    )
    return SyntheticDataset(
        features=features,
        labels=labels,
        realized_returns=realized_returns,
        optimal_position=optimal_position,
        regime_ids=regime_ids,
        change_points=points,
        expected_returns=expected_returns,
        seed=int(seed),
        label_delay=label_delay,
    )


class SyntheticRegimeProcess:
    """Reusable deterministic process object."""

    def __init__(self, config: Optional[SyntheticRegimeConfig] = None, **kwargs: Any) -> None:
        if config is not None and kwargs:
            raise TypeError("pass either config or keyword configuration, not both")
        self.config: Any
        if config is not None:
            self.config = config
        elif any(name in kwargs for name in ("n_samples", "shift_points", "noise_scale")):
            self.config = _LegacyRegimeConfig(**kwargs)
        else:
            self.config = SyntheticRegimeConfig(**kwargs)

    def generate(self, index: Optional[Sequence[Any]] = None) -> SyntheticDataset:
        if hasattr(self.config, "n_samples"):
            legacy = generate_regime_switching(
                n_samples=int(self.config.n_samples),
                n_features=int(self.config.n_features),
                shift_points=tuple(self.config.shift_points),
                seed=int(self.config.seed),
                noise_scale=float(self.config.noise_scale),
            )
            if index is None:
                return legacy
            time_index = _bounded_items(index, "index", MAX_INDEX_ITEMS)
            if len(time_index) != legacy.n_samples:
                raise DataValidationError(f"index must have {legacy.n_samples} entries")
            return SyntheticDataset(
                features=TimeIndexedFeatures(time_index, legacy.features.values),
                labels=TimeIndexedLabels(
                    time_index,
                    legacy.labels.values,
                    legacy.labels.available_at,
                    legacy.labels.name,
                ),
                realized_returns=legacy.realized_returns,
                optimal_position=legacy.optimal_position,
                regime_ids=legacy.regime_ids,
                change_points=legacy.change_points,
                expected_returns=legacy.expected_returns,
                seed=legacy.seed,
                label_delay=legacy.label_delay,
            )
        return generate_synthetic_regimes(
            n_steps=self.config.n_steps,
            n_features=self.config.n_features,
            change_points=self.config.change_points,
            seed=self.config.seed,
            label_delay=self.config.label_delay,
            feature_noise=self.config.feature_noise,
            return_noise=self.config.return_noise,
            signal_strength=self.config.signal_strength,
            return_magnitude=self.config.return_magnitude,
            regime_signs=self.config.regime_signs,
            index=index,
        )

    __call__ = generate


def _validate_shifts(n_samples: int, shift_points: Sequence[int]) -> Tuple[int, ...]:
    shifts = tuple(
        _bounded_int(point, "shift point", minimum=1, maximum=n_samples - 1)
        for point in _bounded_items(shift_points, "shift_points", MAX_CHANGE_POINTS)
    )
    if any(point <= 0 or point >= n_samples for point in shifts):
        raise DataValidationError("shift_points must lie strictly inside the sample range")
    if tuple(sorted(set(shifts))) != shifts:
        raise DataValidationError("shift_points must be strictly increasing")
    return shifts


def _regime_covariance(n_features: int, regime: int) -> NDArray[Any]:
    covariance = np.eye(n_features, dtype=float)
    if regime % 2 == 1 and n_features >= 2:
        covariance[0, 1] = covariance[1, 0] = 0.75
    if regime % 3 == 2:
        covariance *= 1.75
        np.fill_diagonal(covariance, 1.75)
    return covariance


def generate_regime_switching(
    *,
    n_samples: int = 512,
    n_features: int = 4,
    shift_points: Sequence[int] = (160, 320),
    seed: int = 7,
    noise_scale: float = 0.25,
) -> SyntheticDataset:
    """Generate the original continuous-target covariance-shift fixture."""

    n_samples = _bounded_int(
        n_samples, "n_samples", minimum=32, maximum=MAX_SYNTHETIC_SAMPLES
    )
    n_features = _bounded_int(
        n_features, "n_features", minimum=1, maximum=MAX_SYNTHETIC_FEATURES
    )
    if n_samples > MAX_MATRIX_ELEMENTS // n_features:
        raise ResourceLimitError(
            f"n_samples*n_features exceeds the practical element limit {MAX_MATRIX_ELEMENTS}"
        )
    seed = _bounded_int(seed, "seed", minimum=0, maximum=MAX_SEED)
    noise_scale = _finite_float(noise_scale, "noise_scale", minimum=0.0)
    if noise_scale <= 0.0:
        raise ConfigurationError("noise_scale must be positive")
    shifts = _validate_shifts(n_samples, shift_points)
    rng = np.random.default_rng(seed)
    boundaries = (0, *shifts, n_samples)
    n_regimes = len(boundaries) - 1
    features = np.empty((n_samples, n_features), dtype=float)
    targets = np.empty(n_samples, dtype=float)
    regimes = np.empty(n_samples, dtype=int)
    coefficients = []
    for regime in range(n_regimes):
        coefficient = rng.normal(0.0, 0.35, size=n_features)
        if regime % 2 == 1:
            coefficient[0] *= -1.0
        coefficients.append(coefficient)
    for regime, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        covariance = _regime_covariance(n_features, regime)
        block = rng.multivariate_normal(np.zeros(n_features), covariance, size=end - start)
        features[start:end] = block
        targets[start:end] = block @ coefficients[regime] + rng.normal(
            0.0, noise_scale, size=end - start
        )
        regimes[start:end] = regime
    labels = TimeIndexedLabels.from_array(
        targets,
        index=tuple(range(n_samples)),
        available_at=tuple(range(n_samples)),
        name="continuous_target",
    )
    target_sign = np.sign(targets)
    return SyntheticDataset(
        features=TimeIndexedFeatures.from_array(features),
        labels=labels,
        realized_returns=targets,
        optimal_position=target_sign,
        regime_ids=regimes,
        change_points=shifts,
        expected_returns=targets,
        seed=int(seed),
        label_delay=0,
    )


# Friendly aliases used by small experiments and older callers.
generate_synthetic_data = generate_synthetic_regimes
make_synthetic_regimes = generate_synthetic_regimes
SyntheticRegime = SyntheticRegimeProcess
SyntheticRegimeGenerator = SyntheticRegimeProcess
generate_regime_data = generate_synthetic_regimes


__all__ = [
    "TimeIndexedFeatures",
    "TimeIndexedLabels",
    "SyntheticDataset",
    "SyntheticRegimeConfig",
    "SyntheticRegimeProcess",
    "SyntheticRegime",
    "SyntheticRegimeGenerator",
    "generate_synthetic_regimes",
    "generate_synthetic_data",
    "make_synthetic_regimes",
    "generate_regime_data",
    "generate_regime_switching",
]
