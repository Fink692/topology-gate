"""Adaptive recursive ridge least squares.

The estimator in this module uses the covariance form of recursive least
 squares.  A positive ridge value initializes the information matrix with
 ``ridge * I`` (and therefore initializes the covariance with
 ``I / ridge``).  The forgetting factor may be constant, a finite sequence,
 or a callable schedule.

The implementation intentionally has no third-party dependency.  Inputs and
serialized state are kept as ordinary Python values, which also makes the
state hooks deterministic and JSON friendly.
"""

from __future__ import annotations

import inspect
import itertools
import json
import math
import operator
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Optional, Union, cast

from .types import RLSState, RLSUpdate

__all__ = [
    "AdaptiveRLS",
    "AdaptiveRidgeRLS",
    "AdaptiveRecursiveLeastSquares",
    "AdaptiveRecursiveRidgeLeastSquares",
    "RLSConfig",
    "RLS",
    "RecursiveRLS",
    "RecursiveLeastSquares",
    "RecursiveRidgeLeastSquares",
    "MAX_RLS_FEATURES",
    "MAX_RLS_SCHEDULE_LENGTH",
]


NumberLike = Union[int, float]
ForgettingFactor = Union[
    NumberLike,
    Iterable[NumberLike],
    Callable[..., NumberLike],
]
CoefficientState = Union[list[float], list[list[float]]]
PredictionValue = Union[float, list[float]]
PredictionResult = Union[PredictionValue, list[PredictionValue]]

_STATE_VERSION = 1
_SYMMETRY_TOLERANCE = 1.0e-10
_PSD_TOLERANCE = 1.0e-10
_JACOBI_EPSILON = 1.0e-14
# RLS stores a dense covariance matrix.  These limits are deliberately
# conservative so a caller cannot accidentally request an unbounded quadratic
# allocation through a public constructor or state restore.
MAX_RLS_FEATURES = 512
MAX_RLS_SCHEDULE_LENGTH = 100_000


def _finite_float(value: Any, name: str) -> float:
    """Convert a scalar to a finite float without accepting text/bools."""

    if isinstance(value, (bool, str, bytes, bytearray)):
        raise TypeError(f"{name} must be a finite real scalar")
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be a finite real scalar") from exc
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a positive integer")
    try:
        converted = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be a positive integer") from exc
    if converted <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(converted)


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a non-negative integer")
    try:
        converted = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be a non-negative integer") from exc
    if converted < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(converted)


def _validate_lambda(value: Any, name: str = "forgetting_factor") -> float:
    factor = _finite_float(value, name)
    if factor <= 0.0 or factor > 1.0:
        raise ValueError(f"{name} must satisfy 0 < {name} <= 1")
    return factor


@dataclass(frozen=True, slots=True)
class RLSConfig:
    """Validated configuration shared with the package integration layer."""

    n_features: int
    ridge: float = 1.0
    lambda_min: float = 1.0e-6
    lambda_max: float = 1.0
    forgetting_factor: float = 1.0

    def __post_init__(self) -> None:
        n_features = _positive_int(self.n_features, "n_features")
        if n_features > MAX_RLS_FEATURES:
            raise ValueError(
                f"n_features exceeds the dense-RLS limit ({MAX_RLS_FEATURES})"
            )
        ridge = _finite_float(self.ridge, "ridge")
        if ridge <= 0.0:
            raise ValueError("ridge must be positive")
        _initial_variance(ridge)
        lambda_min = _finite_float(self.lambda_min, "lambda_min")
        lambda_max = _finite_float(self.lambda_max, "lambda_max")
        if not 0.0 < lambda_min <= lambda_max <= 1.0:
            raise ValueError("lambda bounds must satisfy 0 < lambda_min <= lambda_max <= 1")
        forgetting_factor = _validate_lambda(self.forgetting_factor)
        if not lambda_min <= forgetting_factor <= lambda_max:
            raise ValueError("forgetting_factor must lie within lambda bounds")
        object.__setattr__(self, "n_features", n_features)
        object.__setattr__(self, "ridge", ridge)
        object.__setattr__(self, "lambda_min", lambda_min)
        object.__setattr__(self, "lambda_max", lambda_max)
        object.__setattr__(self, "forgetting_factor", forgetting_factor)


def _initial_variance(ridge: float) -> float:
    try:
        variance = 1.0 / ridge
    except (OverflowError, ZeroDivisionError) as exc:
        raise ValueError("ridge is too small for a finite initial covariance") from exc
    if not math.isfinite(variance):
        raise ValueError("ridge is too small for a finite initial covariance")
    return variance


def _dot(left: list[float], right: list[float]) -> float:
    try:
        result = math.fsum(a * b for a, b in zip(left, right))
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError("non-finite dot product") from exc
    if not math.isfinite(result):
        raise FloatingPointError("non-finite dot product")
    return result


def _vector(value: Any, size: int, name: str) -> list[float]:
    """Validate and copy a one-dimensional, finite vector."""

    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be a one-dimensional vector")
    ndim = getattr(value, "ndim", None)
    if ndim is not None and ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    try:
        values = list(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be a one-dimensional vector") from exc
    if len(values) != size:
        raise ValueError(f"{name} must have length {size}; got {len(values)}")
    converted: list[float] = []
    for index, item in enumerate(values):
        try:
            converted.append(_finite_float(item, f"{name}[{index}]"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name}[{index}] must be finite and scalar") from exc
    return converted


def _target(value: Any, name: str = "target") -> Union[float, list[float]]:
    """Validate a scalar or fixed-width target vector."""

    if _is_scalar_candidate(value):
        return _finite_float(value, name)
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be a finite scalar or vector")
    ndim = getattr(value, "ndim", None)
    if ndim is not None and ndim != 1:
        raise ValueError(f"{name} must be a scalar or one-dimensional vector")
    try:
        values = list(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be a finite scalar or vector") from exc
    if not values:
        raise ValueError(f"{name} vector must not be empty")
    return _vector(values, len(values), name)


def _coefficients(
    value: Any,
    n_features: int,
    name: str = "coefficients",
) -> tuple[Union[list[float], list[list[float]]], Optional[int]]:
    """Parse scalar-output or multi-output coefficient state."""

    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be a coefficient vector or matrix")
    ndim = getattr(value, "ndim", None)
    if ndim == 1:
        return _vector(value, n_features, name), 1
    if ndim is not None and ndim != 2:
        raise ValueError(f"{name} must be one- or two-dimensional")
    try:
        rows = list(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be a coefficient vector or matrix") from exc
    if not rows:
        raise ValueError(f"{name} must not be empty")
    if all(_is_scalar_candidate(item) for item in rows):
        return _vector(rows, n_features, name), 1
    matrix = _matrix(value, n_features, name)
    n_outputs = len(matrix[0]) if matrix else 0
    if n_outputs <= 0:
        raise ValueError(f"{name} must have at least one output column")
    return matrix, n_outputs


def _copy_coefficients(
    coefficients: CoefficientState,
) -> CoefficientState:
    if coefficients and isinstance(coefficients[0], list):
        matrix = cast(list[list[float]], coefficients)
        return [row[:] for row in matrix]
    vector = cast(list[float], coefficients)
    return vector[:]


def _matrix(value: Any, size: int, name: str) -> list[list[float]]:
    """Validate and copy a square finite matrix."""

    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be a {size}x{size} matrix")
    ndim = getattr(value, "ndim", None)
    if ndim is not None and ndim != 2:
        raise ValueError(f"{name} must be two-dimensional")
    try:
        rows = list(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be a {size}x{size} matrix") from exc
    if len(rows) != size:
        raise ValueError(f"{name} must have {size} rows; got {len(rows)}")
    converted: list[list[float]] = []
    for row_index, row in enumerate(rows):
        if isinstance(row, (str, bytes, bytearray)):
            raise ValueError(f"{name}[{row_index}] must be a row vector")
        try:
            row_values = list(row)
        except TypeError as exc:
            raise ValueError(f"{name}[{row_index}] must be a row vector") from exc
        if len(row_values) != size:
            raise ValueError(
                f"{name}[{row_index}] must have length {size}; got {len(row_values)}"
            )
        converted_row: list[float] = []
        for column_index, item in enumerate(row_values):
            try:
                converted_row.append(
                    _finite_float(item, f"{name}[{row_index}][{column_index}]")
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{name}[{row_index}][{column_index}] must be finite and scalar"
                ) from exc
        converted.append(converted_row)
    return converted


def _identity(size: int) -> list[list[float]]:
    return [[1.0 if row == column else 0.0 for column in range(size)] for row in range(size)]


def _mat_vec(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [_dot(row, vector) for row in matrix]


def _mat_mul(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    right_columns = [list(column) for column in zip(*right)]
    return [[_dot(row, column) for column in right_columns] for row in left]


def _transpose(matrix: list[list[float]]) -> list[list[float]]:
    return [list(column) for column in zip(*matrix)]


def _outer(left: list[float], right: list[float]) -> list[list[float]]:
    return [[a * b for b in right] for a in left]


def _symmetrize(matrix: list[list[float]]) -> list[list[float]]:
    size = len(matrix)
    return [
        [0.5 * (matrix[row][column] + matrix[column][row]) for column in range(size)]
        for row in range(size)
    ]


def _matrix_scale(matrix: list[list[float]]) -> float:
    return max(
        1.0,
        max((abs(value) for row in matrix for value in row), default=0.0),
    )


def _symmetric_eigenvalues(matrix: list[list[float]]) -> list[float]:
    """Return eigenvalue estimates for a real symmetric matrix.

    A small Jacobi eigensolver is sufficient here: RLS state dimensions are
    normally small, and using it avoids depending on NumPy just to validate
    the covariance's PSD invariant.
    """

    size = len(matrix)
    if size == 0:
        return []
    if size == 1:
        return [matrix[0][0]]

    work = [row[:] for row in matrix]
    scale = _matrix_scale(work)
    convergence = _JACOBI_EPSILON * scale
    max_sweeps = max(20, 50 * size * size)

    for _ in range(max_sweeps):
        pivot_row = 0
        pivot_column = 1
        largest = 0.0
        for row in range(size):
            for column in range(row + 1, size):
                magnitude = abs(work[row][column])
                if magnitude > largest:
                    largest = magnitude
                    pivot_row = row
                    pivot_column = column
        if largest <= convergence:
            break

        p = pivot_row
        q = pivot_column
        app = work[p][p]
        aqq = work[q][q]
        apq = work[p][q]
        if apq == 0.0:
            continue
        tau = (aqq - app) / (2.0 * apq)
        if tau >= 0.0:
            t = 1.0 / (tau + math.sqrt(1.0 + tau * tau))
        else:
            t = -1.0 / (-tau + math.sqrt(1.0 + tau * tau))
        cosine = 1.0 / math.sqrt(1.0 + t * t)
        sine = t * cosine

        for index in range(size):
            if index == p or index == q:
                continue
            aip = work[index][p]
            aiq = work[index][q]
            work[index][p] = cosine * aip - sine * aiq
            work[p][index] = work[index][p]
            work[index][q] = sine * aip + cosine * aiq
            work[q][index] = work[index][q]

        work[p][p] = (
            cosine * cosine * app
            - 2.0 * sine * cosine * apq
            + sine * sine * aqq
        )
        work[q][q] = (
            sine * sine * app
            + 2.0 * sine * cosine * apq
            + cosine * cosine * aqq
        )
        work[p][q] = 0.0
        work[q][p] = 0.0

    return sorted(work[index][index] for index in range(size))


def _stabilize_covariance(matrix: list[list[float]], name: str = "covariance") -> list[list[float]]:
    """Symmetrize a covariance and remove only round-off-level negativity."""

    size = len(matrix)
    for row in matrix:
        for value in row:
            if not math.isfinite(value):
                raise FloatingPointError(f"{name} contains a non-finite value")
    symmetric = _symmetrize(matrix)
    scale = _matrix_scale(symmetric)
    minimum = min(_symmetric_eigenvalues(symmetric))
    allowed_negative = _PSD_TOLERANCE * scale
    if minimum < -allowed_negative:
        raise ValueError(f"{name} must be positive semidefinite")
    if minimum < 0.0:
        # The Joseph-form update can leave a negative eigenvalue at a few ulps
        # on ill-conditioned problems.  A diagonal jitter at that scale keeps
        # the operation deterministic while preserving the estimate.
        jitter = -minimum + 16.0 * math.ulp(scale)
        for index in range(size):
            symmetric[index][index] += jitter
        symmetric = _symmetrize(symmetric)
    return symmetric


def _validate_symmetric(matrix: list[list[float]], name: str = "covariance") -> None:
    scale = _matrix_scale(matrix)
    tolerance = _SYMMETRY_TOLERANCE * scale
    size = len(matrix)
    for row in range(size):
        for column in range(row):
            if abs(matrix[row][column] - matrix[column][row]) > tolerance:
                raise ValueError(f"{name} must be symmetric")


def _is_scalar_candidate(value: Any) -> bool:
    if isinstance(value, (bool, str, bytes, bytearray)):
        return False
    try:
        float(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return True


def _normalise_forgetting_factor(forgetting_factor: ForgettingFactor) -> tuple[str, Any]:
    """Validate a schedule without mutating an estimator."""

    if callable(forgetting_factor):
        return "callable", forgetting_factor

    if _is_scalar_candidate(forgetting_factor):
        return "constant", _validate_lambda(forgetting_factor)

    if isinstance(forgetting_factor, (str, bytes, bytearray, Mapping)):
        raise TypeError(
            "forgetting_factor must be a scalar, iterable, or callable"
        )
    try:
        iterator = iter(cast(Iterable[NumberLike], forgetting_factor))
    except TypeError as exc:
        raise TypeError(
            "forgetting_factor must be a scalar, iterable, or callable"
        ) from exc
    values = tuple(itertools.islice(iterator, MAX_RLS_SCHEDULE_LENGTH + 1))
    if not values:
        raise ValueError("forgetting_factor schedule must not be empty")
    if len(values) > MAX_RLS_SCHEDULE_LENGTH:
        raise ValueError(
            "forgetting_factor schedule exceeds the configured resource limit "
            f"({MAX_RLS_SCHEDULE_LENGTH})"
        )
    return "sequence", tuple(
        _validate_lambda(value, f"forgetting_factor[{index}]")
        for index, value in enumerate(values)
    )


def _callable_identity(value: Any) -> str:
    return (
        f"{getattr(value, '__module__', type(value).__module__)}:"
        f"{getattr(value, '__qualname__', type(value).__qualname__)}"
    )


class AdaptiveRLS:
    """Recursive ridge least-squares estimator with adaptive forgetting.

    Parameters
    ----------
    n_features:
        Number of coefficients in each feature vector.
    ridge:
        Positive ridge precision used for initialization.  The initial
        covariance is ``I / ridge`` and the initial coefficient vector is 0.
    forgetting_factor:
        A scalar in ``(0, 1]``, a finite iterable of such factors (one per
        update), or a callable.  Callables receive ``step`` by default, where
        ``step`` is one-based.  Additional positional arguments receive the
        current features, pre-update prediction, and coefficients in that
        order.  The target is never supplied to a schedule.

    Notes
    -----
    For an update with factor ``lambda_t``, the prior covariance is
    ``P / lambda_t``.  The gain and covariance are computed using the Joseph
    form, which preserves symmetry and positive semidefiniteness more reliably
    than a subtractive covariance update.
    """

    def __init__(
        self,
        n_features: int | RLSConfig,
        ridge: NumberLike = 1.0,
        forgetting_factor: ForgettingFactor = 1.0,
        *,
        lambda_: Optional[ForgettingFactor] = None,
        lambda_t: Optional[ForgettingFactor] = None,
        lambda_min: Optional[NumberLike] = None,
        lambda_max: Optional[NumberLike] = None,
        n_outputs: Optional[int] = None,
        n_targets: Optional[int] = None,
    ) -> None:
        if n_outputs is not None and n_targets is not None and n_outputs != n_targets:
            raise ValueError("n_outputs and n_targets must agree")
        if n_outputs is None:
            n_outputs = n_targets
        if n_outputs is not None:
            n_outputs = _positive_int(n_outputs, "n_outputs")

        if isinstance(n_features, RLSConfig):
            config = n_features
            if lambda_ is not None or lambda_t is not None:
                raise ValueError(
                    "pass either RLSConfig or explicit estimator parameters, not both"
                )
            n_features = config.n_features
            ridge = config.ridge
            forgetting_factor = config.forgetting_factor
            if lambda_min is not None or lambda_max is not None:
                raise ValueError("lambda bounds are already defined by RLSConfig")
            lambda_min = config.lambda_min
            lambda_max = config.lambda_max

        if lambda_ is not None and lambda_t is not None:
            raise ValueError("provide only one of lambda_ and lambda_t")
        if lambda_t is not None:
            lambda_ = lambda_t
        if lambda_ is not None:
            if forgetting_factor != 1.0:
                raise ValueError("provide only one of forgetting_factor and lambda_")
            forgetting_factor = lambda_

        self._n_features = _positive_int(n_features, "n_features")
        if self._n_features > MAX_RLS_FEATURES:
            raise ValueError(
                f"n_features exceeds the dense-RLS limit ({MAX_RLS_FEATURES})"
            )
        self._n_outputs = n_outputs
        self._ridge = _finite_float(ridge, "ridge")
        if self._ridge <= 0.0:
            raise ValueError("ridge must be positive")
        self._lambda_min = (
            math.nextafter(0.0, 1.0)
            if lambda_min is None
            else _finite_float(lambda_min, "lambda_min")
        )
        self._lambda_max = (
            1.0 if lambda_max is None else _finite_float(lambda_max, "lambda_max")
        )
        if not 0.0 < self._lambda_min <= self._lambda_max <= 1.0:
            raise ValueError("lambda bounds must satisfy 0 < lambda_min <= lambda_max <= 1")

        self._forgetting_factor_kind: str
        self._forgetting_factor: Any
        self._configure_forgetting_factor(forgetting_factor)
        self._validate_configured_forgetting_factors()

        self._theta: CoefficientState
        if self._n_outputs is None or self._n_outputs == 1:
            self._theta = [0.0] * self._n_features
            if self._n_outputs == 1:
                self._n_outputs = 1
        else:
            self._theta = [
                [0.0] * self._n_outputs for _ in range(self._n_features)
            ]
        initial_variance = _initial_variance(self._ridge)
        self._covariance = [
            [initial_variance if row == column else 0.0 for column in range(self._n_features)]
            for row in range(self._n_features)
        ]
        self._n_updates = 0
        self._schedule_position = 0
        self._lambda_history: list[float] = []
        self._last_prediction: Optional[Union[float, list[float]]] = None
        self._last_residual: Optional[Union[float, list[float]]] = None
        self._last_forgetting_factor: Optional[float] = None

    def _validate_bounded_lambda(self, value: Any, name: str) -> float:
        factor = _validate_lambda(value, name)
        if not self._lambda_min <= factor <= self._lambda_max:
            raise ValueError(
                f"{name} must lie within [{self._lambda_min}, {self._lambda_max}]"
            )
        return factor

    def _validate_configured_forgetting_factors(self) -> None:
        if self._forgetting_factor_kind == "constant":
            self._forgetting_factor = self._validate_bounded_lambda(
                self._forgetting_factor, "forgetting_factor"
            )
        elif self._forgetting_factor_kind == "sequence":
            self._forgetting_factor = tuple(
                self._validate_bounded_lambda(
                    value, f"forgetting_factor[{index}]"
                )
                for index, value in enumerate(self._forgetting_factor)
            )

    # ------------------------------------------------------------------
    # Public state and configuration properties
    # ------------------------------------------------------------------
    @property
    def n_features(self) -> int:
        return self._n_features

    @property
    def n_outputs(self) -> Optional[int]:
        return self._n_outputs

    @property
    def n_targets(self) -> Optional[int]:
        return self._n_outputs

    @property
    def ridge(self) -> float:
        return self._ridge

    @property
    def regularization(self) -> float:
        """Alias for ``ridge`` used by some callers."""

        return self._ridge

    @property
    def lambda_min(self) -> float:
        return self._lambda_min

    @property
    def lambda_max(self) -> float:
        return self._lambda_max

    @property
    def coefficients(self) -> CoefficientState:
        return _copy_coefficients(self._theta)

    @property
    def theta(self) -> CoefficientState:
        return _copy_coefficients(self._theta)

    @property
    def weights(self) -> CoefficientState:
        return _copy_coefficients(self._theta)

    @property
    def covariance(self) -> list[list[float]]:
        return [row[:] for row in self._covariance]

    @property
    def covariance_matrix(self) -> list[list[float]]:
        return self.covariance

    @property
    def P(self) -> list[list[float]]:
        return self.covariance

    @property
    def n_updates(self) -> int:
        return self._n_updates

    @property
    def step(self) -> int:
        return self._n_updates

    @property
    def last_prediction(self) -> Optional[Union[float, list[float]]]:
        return self._last_prediction

    @property
    def last_residual(self) -> Optional[Union[float, list[float]]]:
        return self._last_residual

    @property
    def last_forgetting_factor(self) -> Optional[float]:
        return self._last_forgetting_factor

    @property
    def forgetting_factors(self) -> tuple[float, ...]:
        return tuple(self._lambda_history)

    def _state_snapshot(self, forgetting_factor: Optional[float] = None) -> RLSState:
        factor = forgetting_factor
        if factor is None:
            factor = self._last_forgetting_factor
        if factor is None:
            if self._forgetting_factor_kind == "constant":
                factor = self._forgetting_factor
            elif self._forgetting_factor_kind == "sequence":
                factor = self._forgetting_factor[0]
            else:
                factor = self._lambda_max
        return RLSState(
            coefficients=_copy_coefficients(self._theta),
            covariance=[row[:] for row in self._covariance],
            sample_count=self._n_updates,
            forgetting_factor=float(factor),
        )

    @property
    def state(self) -> RLSState:
        """Current shared-contract state snapshot."""

        return self._state_snapshot()

    def current_state(self) -> RLSState:
        """Explicit method form of the :attr:`state` snapshot hook."""

        return self._state_snapshot()

    # ------------------------------------------------------------------
    # Forgetting-factor handling
    # ------------------------------------------------------------------
    def _configure_forgetting_factor(self, forgetting_factor: ForgettingFactor) -> None:
        kind, value = _normalise_forgetting_factor(forgetting_factor)
        self._forgetting_factor_kind = kind
        self._forgetting_factor = value

    def _call_forgetting_factor(
        self,
        step: int,
        prediction: PredictionValue,
        features: list[float],
    ) -> Any:
        callback = self._forgetting_factor
        context = {
            "step": step,
            "t": step,
            "prediction": prediction,
            "predicted": prediction,
            "y_hat": prediction,
            "features": tuple(features),
            "x": tuple(features),
            "theta": _copy_coefficients(self._theta),
            "coefficients": _copy_coefficients(self._theta),
            "weights": _copy_coefficients(self._theta),
        }
        try:
            signature = inspect.signature(callback)
        except (TypeError, ValueError):
            result = callback(step)
        else:
            parameters = list(signature.parameters.values())
            positional = [
                parameter
                for parameter in parameters
                if parameter.kind
                in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
            ]
            has_varargs = any(
                parameter.kind == parameter.VAR_POSITIONAL for parameter in parameters
            )
            keyword_only = [
                parameter
                for parameter in parameters
                if parameter.kind == parameter.KEYWORD_ONLY
            ]

            if keyword_only and not positional and not has_varargs:
                keyword_values = {
                    parameter.name: context[parameter.name]
                    for parameter in keyword_only
                    if parameter.name in context
                }
                if len(keyword_values) != len(
                    [parameter for parameter in keyword_only if parameter.default is parameter.empty]
                ):
                    raise TypeError(
                        "forgetting-factor callable has unsupported required arguments"
                    )
                result = callback(**keyword_values)
            elif has_varargs:
                result = callback(
                    step,
                    tuple(features),
                    prediction,
                    _copy_coefficients(self._theta),
                )
            elif not positional:
                result = callback()
            elif len(positional) == 1:
                parameter_name = positional[0].name.lower()
                if parameter_name in {"prediction", "predicted", "y_hat"}:
                    result = callback(prediction)
                elif parameter_name in {"features", "x", "feature_vector"}:
                    result = callback(tuple(features))
                elif parameter_name in {"theta", "coefficients", "weights"}:
                    result = callback(_copy_coefficients(self._theta))
                elif parameter_name in {"residual", "error", "innovation"}:
                    raise ValueError(
                        "forgetting-factor callable cannot depend on the target"
                    )
                else:
                    result = callback(step)
            else:
                positional_values: tuple[Any, ...] = (
                    step,
                    tuple(features),
                    prediction,
                    _copy_coefficients(self._theta),
                )
                result = callback(*positional_values[: len(positional)])
        return result

    def _resolve_forgetting_factor(
        self,
        step: int,
        prediction: PredictionValue,
        features: list[float],
        override: Any,
    ) -> float:
        if override is not None:
            return self._validate_bounded_lambda(override, "forgetting_factor")
        if self._forgetting_factor_kind == "constant":
            return self._forgetting_factor
        if self._forgetting_factor_kind == "sequence":
            if self._schedule_position >= len(self._forgetting_factor):
                raise ValueError("forgetting_factor schedule is exhausted")
            return self._validate_bounded_lambda(
                self._forgetting_factor[self._schedule_position],
                "forgetting_factor",
            )
        return self._validate_bounded_lambda(
            self._call_forgetting_factor(step, prediction, features),
            "forgetting_factor callable result",
        )

    # ------------------------------------------------------------------
    # Estimator operations
    # ------------------------------------------------------------------
    def _predict_one(self, features: list[float]) -> PredictionValue:
        if not isinstance(self._theta[0], list):
            vector = cast(list[float], self._theta)
            return _dot(features, vector)
        matrix = cast(list[list[float]], self._theta)
        n_outputs = self._n_outputs
        if n_outputs is None:
            raise RuntimeError("matrix coefficients require a known output count")
        columns = [
            [matrix[row][column] for row in range(self._n_features)]
            for column in range(n_outputs)
        ]
        return [_dot(features, column) for column in columns]

    def predict(self, features: Any) -> PredictionResult:
        """Predict one target, or a list of targets for a 2-D feature input."""

        ndim = getattr(features, "ndim", None)
        if ndim is not None:
            if ndim == 1:
                return self._predict_one(
                    _vector(features, self._n_features, "features")
                )
            if ndim != 2:
                raise ValueError("features must be one- or two-dimensional")
            try:
                rows = list(features)
            except TypeError as exc:
                raise TypeError("features must be a vector or matrix") from exc
            if not rows:
                raise ValueError("features matrix must not be empty")
            return [
                self._predict_one(_vector(row, self._n_features, "features row"))
                for row in rows
            ]

        if isinstance(features, (str, bytes, bytearray)):
            raise TypeError("features must be a vector or matrix")
        try:
            values = list(features)
        except TypeError as exc:
            raise TypeError("features must be a vector or matrix") from exc
        if not values:
            raise ValueError("features must not be empty")
        if all(_is_scalar_candidate(item) for item in values):
            return self._predict_one(_vector(values, self._n_features, "features"))
        return [
            self._predict_one(_vector(row, self._n_features, "features row"))
            for row in values
        ]

    def update(
        self,
        features: Any,
        target: Any,
        forgetting_factor: Any = None,
        *,
        lambda_: Any = None,
        lambda_t: Any = None,
    ) -> RLSUpdate:
        """Assimilate one observation and return the shared update receipt.

        ``forgetting_factor``, ``lambda_``, or ``lambda_t`` can override the
        configured schedule for this observation; at most one override may be
        supplied.  The configured schedule advances once an update succeeds.
        """

        overrides = [
            value
            for value in (forgetting_factor, lambda_, lambda_t)
            if value is not None
        ]
        if len(overrides) > 1:
            raise ValueError(
                "provide at most one of forgetting_factor, lambda_, and lambda_t"
            )
        override = overrides[0] if overrides else None

        x = _vector(features, self._n_features, "features")
        prediction_before = self._predict_one(x)
        step = self._n_updates + 1
        factor = self._resolve_forgetting_factor(
            step,
            prediction_before,
            x,
            override,
        )
        y: PredictionValue = _target(target)
        prediction: PredictionValue
        residual: PredictionValue
        candidate_n_outputs = self._n_outputs
        if isinstance(y, list):
            if candidate_n_outputs is not None and candidate_n_outputs != len(y):
                raise ValueError(
                    f"target has {len(y)} outputs; expected {candidate_n_outputs}"
                )
            candidate_n_outputs = len(y)
            if isinstance(prediction_before, list):
                prediction = prediction_before[:]
            else:
                prediction = [prediction_before] * candidate_n_outputs
            residual = [
                target_value - predicted_value
                for target_value, predicted_value in zip(y, prediction)
            ]
            if any(not math.isfinite(value) for value in residual):
                raise FloatingPointError("non-finite residual")
        else:
            if candidate_n_outputs is not None and (
                candidate_n_outputs > 1 or isinstance(self._theta[0], list)
            ):
                raise ValueError(
                    f"scalar target supplied to a {candidate_n_outputs}-output estimator"
                )
            candidate_n_outputs = 1
            if isinstance(prediction_before, list):
                scalar_prediction = prediction_before[0]
            else:
                scalar_prediction = cast(float, prediction_before)
            prediction = scalar_prediction
            residual = y - scalar_prediction
            if not math.isfinite(residual):
                raise FloatingPointError("non-finite residual")

        # Discount the old information before incorporating the new unit
        # observation.  Keep every intermediate local so invalid arithmetic
        # cannot partially mutate estimator state.
        try:
            prior_covariance = [
                [value / factor for value in row] for row in self._covariance
            ]
        except (OverflowError, ZeroDivisionError) as exc:
            raise FloatingPointError("non-finite discounted covariance") from exc
        for row in prior_covariance:
            if any(not math.isfinite(value) for value in row):
                raise FloatingPointError("non-finite discounted covariance")

        prior_times_x = _mat_vec(prior_covariance, x)
        quadratic = _dot(x, prior_times_x)
        scale = max(1.0, abs(quadratic))
        if quadratic < -_PSD_TOLERANCE * scale:
            raise FloatingPointError("covariance is not positive semidefinite")
        if quadratic < 0.0:
            quadratic = 0.0
        denominator = 1.0 + quadratic
        if not math.isfinite(denominator) or denominator <= 0.0:
            raise FloatingPointError("invalid RLS gain denominator")

        gain = [value / denominator for value in prior_times_x]
        updated_theta: CoefficientState
        if isinstance(residual, list):
            if isinstance(self._theta[0], list):
                prior_theta = cast(list[list[float]], self._theta)
            else:
                prior_theta = [
                    [0.0] * len(residual) for _ in range(self._n_features)
                ]
            updated_theta = [
                [
                    prior_theta[row_index][output]
                    + gain[row_index] * residual[output]
                    for output in range(len(residual))
                ]
                for row_index in range(self._n_features)
            ]
            if any(
                not math.isfinite(value)
                for row in updated_theta
                for value in row
            ):
                raise FloatingPointError("non-finite coefficient update")
        else:
            prior_theta_vector = cast(list[float], self._theta)
            residual_value = cast(float, residual)
            updated_vector: list[float] = [
                coefficient + gain_value * residual_value
                for coefficient, gain_value in zip(prior_theta_vector, gain)
            ]
            if any(not math.isfinite(value) for value in updated_vector):
                raise FloatingPointError("non-finite coefficient update")
            updated_theta = updated_vector

        # Joseph form for measurement noise variance 1.  In exact arithmetic
        # this equals P_prior - P_prior*x*x'*P_prior/(1+x'*P_prior*x), while
        # retaining PSD under floating-point cancellation.
        identity = _identity(self._n_features)
        gain_outer_x = _outer(gain, x)
        transition = [
            [identity[row][column] - gain_outer_x[row][column]
             for column in range(self._n_features)]
            for row in range(self._n_features)
        ]
        joseph_left = _mat_mul(transition, prior_covariance)
        joseph_covariance = _mat_mul(joseph_left, _transpose(transition))
        gain_outer_gain = _outer(gain, gain)
        for row_index in range(self._n_features):
            for column_index in range(self._n_features):
                joseph_covariance[row_index][column_index] += gain_outer_gain[
                    row_index
                ][column_index]
        updated_covariance = _stabilize_covariance(joseph_covariance)

        # Commit only after all validation and numerical checks succeed.
        self._theta = updated_theta
        self._n_outputs = candidate_n_outputs
        self._covariance = updated_covariance
        self._n_updates = step
        self._schedule_position += 1
        self._lambda_history.append(factor)
        self._last_prediction = prediction
        self._last_residual = residual
        self._last_forgetting_factor = factor
        return RLSUpdate(
            prediction=prediction,
            target=y,
            residual=residual,
            state=self._state_snapshot(factor),
        )

    def reset(self) -> "AdaptiveRLS":
        """Restore the ridge prior and rewind any finite factor schedule."""

        if self._n_outputs is None or (
            self._n_outputs == 1 and not isinstance(self._theta[0], list)
        ):
            self._theta = [0.0] * self._n_features
        else:
            self._theta = [
                [0.0] * self._n_outputs for _ in range(self._n_features)
            ]
        initial_variance = _initial_variance(self._ridge)
        self._covariance = [
            [initial_variance if row == column else 0.0 for column in range(self._n_features)]
            for row in range(self._n_features)
        ]
        self._n_updates = 0
        self._schedule_position = 0
        self._lambda_history = []
        self._last_prediction = None
        self._last_residual = None
        self._last_forgetting_factor = None
        return self

    # ------------------------------------------------------------------
    # Serialization hooks
    # ------------------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        """Return a deterministic, JSON-serializable snapshot of the state."""

        if self._forgetting_factor_kind == "constant":
            serialized_factor: Any = self._forgetting_factor
        elif self._forgetting_factor_kind == "sequence":
            serialized_factor = list(self._forgetting_factor)
        else:
            serialized_factor = None
        factor_identity = (
            _callable_identity(self._forgetting_factor)
            if self._forgetting_factor_kind == "callable"
            else None
        )
        return {
            "version": _STATE_VERSION,
            "n_features": self._n_features,
            "n_outputs": self._n_outputs,
            "ridge": self._ridge,
            "lambda_min": self._lambda_min,
            "lambda_max": self._lambda_max,
            "forgetting_factor": serialized_factor,
            "forgetting_factor_kind": self._forgetting_factor_kind,
            "forgetting_factor_identity": factor_identity,
            "theta": _copy_coefficients(self._theta),
            "coefficients": _copy_coefficients(self._theta),
            "covariance": [row[:] for row in self._covariance],
            "n_updates": self._n_updates,
            "sample_count": self._n_updates,
            "schedule_position": self._schedule_position,
            "forgetting_factors": list(self._lambda_history),
            "last_prediction": self._last_prediction,
            "last_residual": self._last_residual,
            "last_forgetting_factor": self._last_forgetting_factor,
        }

    def get_state(self) -> dict[str, Any]:
        """Alias for :meth:`state_dict`."""

        return self.state_dict()

    def to_state(self) -> RLSState:
        """Return the shared typed state snapshot."""

        return self._state_snapshot()

    to_state_dict = state_dict
    serialize_state = state_dict

    def _load_state(
        self,
        state: Mapping[str, Any] | RLSState,
        forgetting_factor: Optional[ForgettingFactor] = None,
    ) -> None:
        if isinstance(state, RLSState):
            state = {
                "n_features": self._n_features,
                "theta": state.coefficients,
                "covariance": state.covariance,
                "n_updates": state.sample_count,
                "last_forgetting_factor": state.forgetting_factor,
            }
        if not isinstance(state, Mapping):
            raise TypeError("state must be a mapping")
        version = state.get("version", _STATE_VERSION)
        if version != _STATE_VERSION:
            raise ValueError(f"unsupported RLS state version: {version!r}")

        state_n_features = state.get("n_features", self._n_features)
        state_n_features = _positive_int(state_n_features, "state.n_features")
        if state_n_features > MAX_RLS_FEATURES:
            raise ValueError(
                "state.n_features exceeds the dense-RLS limit "
                f"({MAX_RLS_FEATURES})"
            )
        if state_n_features != self._n_features:
            raise ValueError(
                f"state has {state_n_features} features; expected {self._n_features}"
            )

        state_n_outputs = state.get("n_outputs")
        if state_n_outputs is not None:
            state_n_outputs = _positive_int(state_n_outputs, "state.n_outputs")

        state_ridge = state.get("ridge", self._ridge)
        state_ridge = _finite_float(state_ridge, "state.ridge")
        if state_ridge <= 0.0:
            raise ValueError("state.ridge must be positive")
        _initial_variance(state_ridge)

        candidate_lambda_min = self._lambda_min
        candidate_lambda_max = self._lambda_max
        if "lambda_min" in state:
            candidate_lambda_min = _finite_float(state["lambda_min"], "state.lambda_min")
        if "lambda_max" in state:
            candidate_lambda_max = _finite_float(state["lambda_max"], "state.lambda_max")
        if not 0.0 < candidate_lambda_min <= candidate_lambda_max <= 1.0:
            raise ValueError(
                "state lambda bounds must satisfy 0 < lambda_min <= lambda_max <= 1"
            )

        state_kind = state.get("forgetting_factor_kind")
        state_factor = state.get("forgetting_factor")
        configured_factor = forgetting_factor
        if configured_factor is None:
            if state_kind in {"constant", "sequence"} and state_factor is not None:
                configured_factor = state_factor
            elif state_kind == "callable":
                configured_factor = self._forgetting_factor
            elif state_factor is not None:
                configured_factor = state_factor
        candidate_kind = self._forgetting_factor_kind
        candidate_factor = self._forgetting_factor
        if configured_factor is not None:
            candidate_kind, candidate_factor = _normalise_forgetting_factor(
                configured_factor
            )
        state_identity = state.get("forgetting_factor_identity")
        if state_kind == "callable" and state_identity is not None:
            if not callable(configured_factor) or _callable_identity(configured_factor) != state_identity:
                raise ValueError("callable forgetting_factor identity does not match state")

        theta_value = state.get("theta", state.get("coefficients"))
        if theta_value is None:
            raise ValueError("state is missing theta/coefficients")
        covariance_value = state.get("covariance", state.get("P"))
        if covariance_value is None:
            raise ValueError("state is missing covariance/P")

        theta, parsed_n_outputs = _coefficients(
            theta_value, self._n_features, "state.theta"
        )
        if state_n_outputs is not None and state_n_outputs != parsed_n_outputs:
            raise ValueError("state.n_outputs disagrees with state.theta")
        candidate_n_outputs = parsed_n_outputs
        covariance = _matrix(covariance_value, self._n_features, "state.covariance")
        _validate_symmetric(covariance, "state.covariance")
        covariance = _stabilize_covariance(covariance, "state.covariance")

        n_updates_value = state.get("n_updates", state.get("step", 0))
        n_updates = _nonnegative_int(n_updates_value, "state.n_updates")
        history_value = state.get("forgetting_factors", [])
        if isinstance(history_value, (str, bytes, bytearray)):
            raise TypeError("state.forgetting_factors must be iterable")
        try:
            history_iterator = iter(history_value)
        except TypeError as exc:
            raise TypeError("state.forgetting_factors must be iterable") from exc
        history = []
        for index, value in enumerate(
            itertools.islice(history_iterator, MAX_RLS_SCHEDULE_LENGTH + 1)
        ):
            history.append(
                _validate_lambda(value, f"state.forgetting_factors[{index}]")
            )
        if len(history) > MAX_RLS_SCHEDULE_LENGTH:
            raise ValueError(
                "state.forgetting_factors exceeds the resource limit "
                f"({MAX_RLS_SCHEDULE_LENGTH})"
            )
        if history and len(history) != n_updates:
            raise ValueError("state.forgetting_factors length must equal n_updates")
        if not history and n_updates:
            # Minimal externally supplied states need not carry diagnostics.
            history = []
        if any(
            not candidate_lambda_min <= value <= candidate_lambda_max
            for value in history
        ):
            raise ValueError("state.forgetting_factors exceed lambda bounds")

        position_value = state.get("schedule_position", n_updates)
        if isinstance(position_value, bool):
            raise TypeError("state.schedule_position must be a non-negative integer")
        try:
            schedule_position = operator.index(position_value)
        except TypeError as exc:
            raise TypeError("state.schedule_position must be a non-negative integer") from exc
        if schedule_position < 0:
            raise ValueError("state.schedule_position must be non-negative")
        if schedule_position != n_updates:
            raise ValueError("state.schedule_position must equal n_updates")

        def optional_finite(name: str) -> Optional[float]:
            value = state.get(name)
            if value is None:
                return None
            return _finite_float(value, f"state.{name}")

        def optional_target(name: str) -> Optional[Union[float, list[float]]]:
            value = state.get(name)
            if value is None:
                return None
            return _target(value, f"state.{name}")

        last_prediction = optional_target("last_prediction")
        last_residual = optional_target("last_residual")
        last_factor = state.get("last_forgetting_factor")
        if last_factor is not None:
            last_factor = _validate_lambda(
                last_factor, "state.last_forgetting_factor"
            )
            if not candidate_lambda_min <= last_factor <= candidate_lambda_max:
                raise ValueError("state.last_forgetting_factor exceeds lambda bounds")
        if history and last_factor is not None and not math.isclose(
            history[-1], last_factor, rel_tol=1.0e-12, abs_tol=1.0e-15
        ):
            raise ValueError("state.last_forgetting_factor disagrees with history")

        if candidate_kind == "constant":
            candidate_factor = _validate_lambda(
                candidate_factor, "state.forgetting_factor"
            )
            if not candidate_lambda_min <= candidate_factor <= candidate_lambda_max:
                raise ValueError("state forgetting factor exceeds lambda bounds")
        elif candidate_kind == "sequence":
            candidate_factor = tuple(
                _validate_lambda(value, f"state.forgetting_factor[{index}]")
                for index, value in enumerate(candidate_factor)
            )
            if any(
                not candidate_lambda_min <= value <= candidate_lambda_max
                for value in candidate_factor
            ):
                raise ValueError("state forgetting factors exceed lambda bounds")
        elif candidate_kind == "callable":
            # Callable results are validated at update time; the configured
            # bounds still need to be retained in the candidate state.
            pass
        if candidate_kind == "sequence" and schedule_position > len(candidate_factor):
            raise ValueError("state exceeds the configured forgetting_factor schedule")

        # Commit only after the whole candidate state has passed validation.
        self._ridge = state_ridge
        self._n_outputs = candidate_n_outputs
        self._lambda_min = candidate_lambda_min
        self._lambda_max = candidate_lambda_max
        self._forgetting_factor_kind = candidate_kind
        self._forgetting_factor = candidate_factor
        self._theta = theta
        self._covariance = covariance
        self._n_updates = n_updates
        self._schedule_position = schedule_position
        self._lambda_history = history
        self._last_prediction = last_prediction
        self._last_residual = last_residual
        self._last_forgetting_factor = last_factor

    def set_state(
        self,
        state: Mapping[str, Any] | RLSState,
        *,
        forgetting_factor: Optional[ForgettingFactor] = None,
    ) -> "AdaptiveRLS":
        """Restore a validated state snapshot and return ``self``."""

        self._load_state(state, forgetting_factor)
        return self

    def load_state_dict(
        self,
        state: Mapping[str, Any] | RLSState,
        *,
        forgetting_factor: Optional[ForgettingFactor] = None,
    ) -> "AdaptiveRLS":
        """Alias for :meth:`set_state`."""

        return self.set_state(state, forgetting_factor=forgetting_factor)

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, Any],
        *,
        forgetting_factor: Optional[ForgettingFactor] = None,
    ) -> "AdaptiveRLS":
        if not isinstance(state, Mapping):
            raise TypeError("state must be a mapping")
        n_features = _positive_int(state.get("n_features"), "state.n_features")
        ridge = _finite_float(state.get("ridge", 1.0), "state.ridge")
        if ridge <= 0.0:
            raise ValueError("state.ridge must be positive")
        configured_factor = forgetting_factor
        if configured_factor is None:
            state_factor = state.get("forgetting_factor")
            state_kind = state.get("forgetting_factor_kind")
            if state_factor is not None and state_kind in {None, "constant", "sequence"}:
                configured_factor = state_factor
            elif state_kind == "callable":
                raise ValueError(
                    "a callable forgetting_factor must be supplied when restoring state"
                )
            else:
                configured_factor = 1.0
        estimator = cls(
            n_features=n_features,
            ridge=ridge,
            forgetting_factor=configured_factor,
            lambda_min=state.get("lambda_min"),
            lambda_max=state.get("lambda_max"),
            n_outputs=state.get("n_outputs"),
        )
        estimator.set_state(state, forgetting_factor=configured_factor)
        return estimator

    from_state = from_state_dict

    def to_json(self) -> str:
        """Serialize :meth:`state_dict` with stable key ordering."""

        return json.dumps(
            self.state_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    serialize = to_json

    @classmethod
    def from_json(
        cls,
        payload: str,
        *,
        forgetting_factor: Optional[ForgettingFactor] = None,
    ) -> "AdaptiveRLS":
        if not isinstance(payload, str):
            raise TypeError("payload must be a JSON string")
        try:
            state = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError("payload is not valid JSON") from exc
        return cls.from_state_dict(state, forgetting_factor=forgetting_factor)

    deserialize = from_json


# Descriptive aliases keep the module usable with either the short estimator
# name or the full name used in type contracts.
AdaptiveRecursiveLeastSquares = AdaptiveRLS
AdaptiveRecursiveRidgeLeastSquares = AdaptiveRLS
AdaptiveRidgeRLS = AdaptiveRLS
RecursiveLeastSquares = AdaptiveRLS
RecursiveRidgeLeastSquares = AdaptiveRLS
RecursiveRLS = AdaptiveRLS
RLS = AdaptiveRLS
