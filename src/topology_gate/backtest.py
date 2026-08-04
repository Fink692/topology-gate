"""Offline, chronological walk-forward backtesting.

This module intentionally operates on in-memory arrays only.  It does not
fetch data, place orders, or expose labels from the future to a model.  The
``training_positions`` audit trail in :class:`BacktestResult` is designed to
make the no-look-ahead rule easy to test.
"""

from __future__ import annotations

import copy
import inspect
from dataclasses import dataclass, field, replace
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    cast,
)

import numpy as np
from numpy.typing import NDArray

from .synthetic import (
    SyntheticDataset,
    TimeIndexedFeatures,
    TimeIndexedLabels,
)

ArrayLike = Sequence[float] | NDArray[Any]
Predictor = Callable[[NDArray[Any], NDArray[Any], NDArray[Any]], Any]
ActionMapper = Callable[[Any], float]
BaselineHook = Callable[..., Any]


def _as_1d_float(values: Any, n: int, name: str) -> NDArray[Any]:
    array = np.asarray(values)
    if array.ndim == 2 and 1 in array.shape:
        array = array.reshape(-1)
    if array.ndim != 1 or len(array) != n:
        raise ValueError(f"{name} must be one-dimensional with length {n}")
    try:
        result = np.asarray(array, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc
    if np.any(~np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return np.array(result, dtype=float, copy=True)


def _finite_correlation(left: NDArray[Any], right: NDArray[Any]) -> float:
    mask = np.isfinite(left) & np.isfinite(right)
    if int(mask.sum()) < 2:
        return 0.0
    x = left[mask]
    y = right[mask]
    x_centered = x - float(np.mean(x))
    y_centered = y - float(np.mean(y))
    denominator = float(np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2)))
    if denominator <= np.finfo(float).eps:
        return 0.0
    return float(np.sum(x_centered * y_centered) / denominator)


def _safe_mean(values: Iterable[float]) -> float:
    array = np.asarray(tuple(values), dtype=float)
    return 0.0 if len(array) == 0 else float(np.mean(array))


def _clone_model(model: Any) -> Any:
    try:
        return copy.deepcopy(model)
    except Exception:
        try:
            return copy.copy(model)
        except Exception:
            return model


def _callable_required_positional_count(function: Callable[..., Any]) -> Optional[int]:
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return None
    count = 0
    for parameter in signature.parameters.values():
        if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD):
            if parameter.default is parameter.empty:
                count += 1
    return count


def _invoke_callback(function: Callable[..., Any], x_train: NDArray[Any], y_train: NDArray[Any], x_test: NDArray[Any]) -> Any:
    """Call a predictor while supporting the small callback forms people use.

    The documented form is ``(x_train, y_train, x_test)``.  One- and two-
    argument callbacks are accepted as conveniences for stateless predictors;
    none receives future labels.
    """

    try:
        signature = inspect.signature(function)
        positional = [
            parameter
            for parameter in signature.parameters.values()
            if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
        ]
        has_varargs = any(
            parameter.kind == parameter.VAR_POSITIONAL
            for parameter in signature.parameters.values()
        )
        if has_varargs or len(positional) >= 3:
            return function(x_train, y_train, x_test)
        if len(positional) == 2:
            return function(x_train, y_train)
        if len(positional) == 1:
            return function(x_test)
        return function()
    except (TypeError, ValueError):
        # Some extension callables do not expose a signature.  The documented
        # form is the only fallback; an exception from the callback itself is
        # intentionally allowed to propagate.
        return function(x_train, y_train, x_test)


def _scalar_prediction(value: Any) -> Any:
    array = np.asarray(value)
    if array.size == 0:
        return np.nan
    if array.ndim == 0:
        return array.item()
    flattened = array.reshape(-1)
    if len(flattened) == 1:
        return flattened[0].item()
    # A two-column probability or score output is converted to a signed
    # margin.  For any other vector output, using the final component is a
    # deterministic and useful convention for a scalar action.
    if len(flattened) == 2:
        try:
            return float(flattened[1]) - float(flattened[0])
        except (TypeError, ValueError):
            return flattened[-1].item()
    return flattened[-1].item()


def _numeric_classes(values: NDArray[Any]) -> Optional[NDArray[Any]]:
    try:
        numeric = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        return None
    if len(numeric) == 0:
        return numeric
    if np.any(~np.isfinite(numeric)):
        return None
    return np.unique(numeric)


def _map_prediction_to_position(
    raw_prediction: Any,
    y_train: NDArray[Any],
    mode: str,
    max_position: float,
    allow_short: bool,
    action_mapper: Optional[ActionMapper],
) -> float:
    if action_mapper is not None:
        value = action_mapper(raw_prediction)
    else:
        value = raw_prediction
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        classes = np.unique(y_train)
        if len(classes) == 2:
            numeric = -1.0 if value == classes[0] else 1.0
        else:
            numeric = 0.0

    mode = mode.lower()
    if mode not in {"auto", "position", "signal", "label", "raw"}:
        raise ValueError("prediction_mode must be auto, position, signal, label, or raw")
    classes = _numeric_classes(y_train)
    if mode in {"auto", "label"} and classes is not None:
        if len(classes) == 2 and np.allclose(classes, np.array([0.0, 1.0])):
            if mode == "label" or numeric in (0.0, 1.0):
                numeric = -1.0 if numeric < 0.5 else 1.0
            elif 0.0 <= numeric <= 1.0:
                numeric = 2.0 * numeric - 1.0
        elif len(classes) == 2 and np.allclose(classes, np.array([-1.0, 1.0])):
            numeric = float(np.sign(numeric))
        elif len(classes) == 2 and mode == "label":
            numeric = -1.0 if numeric <= classes[0] else 1.0
    if mode == "label" and classes is None:
        numeric = float(np.sign(numeric))
    if not np.isfinite(numeric):
        return 0.0
    numeric = float(np.clip(numeric, -abs(max_position), abs(max_position)))
    if not allow_short:
        numeric = float(np.clip(numeric, 0.0, abs(max_position)))
    return numeric


@dataclass(frozen=True)
class WalkForwardConfig:
    """Controls chronological training, execution, and metric conventions."""

    initial_train_size: int = 20
    label_delay: int = 0
    retrain_every: int = 1
    training_window: Optional[int] = None
    min_train_size: int = 1
    transaction_cost: float = 0.0
    slippage: float = 0.0
    transaction_cost_bps: float = 0.0
    max_position: float = 1.0
    allow_short: bool = True
    prediction_mode: str = "auto"
    hold_last_position: bool = True
    periods_per_year: float = 252.0
    detection_persistence: int = 1
    detection_position_threshold: float = 1e-12
    promotion_window: Optional[int] = None
    # Compatibility names from topology_gate.types.BacktestConfig.  They are
    # optional so the richer worker config can keep its explicit semantics.
    warmup: Optional[int] = None
    horizon: Optional[int] = None
    refit_interval: Optional[int] = None
    record_predictions: bool = True
    require_realized_returns: bool = False

    def __post_init__(self) -> None:
        if self.warmup is not None:
            if int(self.warmup) < 0:
                raise ValueError("warmup must be non-negative")
            object.__setattr__(self, "initial_train_size", int(self.warmup))
        if self.horizon is not None:
            if int(self.horizon) <= 0:
                raise ValueError("horizon must be positive")
            object.__setattr__(self, "label_delay", max(1, int(self.horizon)))
        if self.refit_interval is not None:
            if int(self.refit_interval) <= 0:
                raise ValueError("refit_interval must be positive")
            object.__setattr__(self, "retrain_every", int(self.refit_interval))
        integer_fields = {
            "initial_train_size": self.initial_train_size,
            "label_delay": self.label_delay,
            "retrain_every": self.retrain_every,
            "min_train_size": self.min_train_size,
            "detection_persistence": self.detection_persistence,
        }
        for name, value in integer_fields.items():
            if not isinstance(value, (int, np.integer)):
                raise TypeError(f"{name} must be an integer")
            if int(value) < (1 if name in {"retrain_every", "detection_persistence"} else 0):
                raise ValueError(f"{name} must be non-negative")
        if self.training_window is not None and int(self.training_window) < 1:
            raise ValueError("training_window must be positive when supplied")
        numeric_fields: dict[str, float] = {
            "transaction_cost": self.transaction_cost,
            "slippage": self.slippage,
            "transaction_cost_bps": self.transaction_cost_bps,
            "max_position": self.max_position,
            "periods_per_year": self.periods_per_year,
            "detection_position_threshold": self.detection_position_threshold,
        }
        for numeric_name, numeric_value in numeric_fields.items():
            converted = float(numeric_value)
            if not np.isfinite(converted):
                raise ValueError(f"{numeric_name} must be finite")
            if converted < 0:
                raise ValueError(f"{numeric_name} must be non-negative")
        if self.max_position == 0:
            raise ValueError("max_position must be positive")
        if self.periods_per_year == 0:
            raise ValueError("periods_per_year must be positive")
        if not isinstance(self.require_realized_returns, bool):
            raise TypeError("require_realized_returns must be boolean")

    @property
    def turnover_cost_rate(self) -> float:
        return float(
            self.transaction_cost
            + self.slippage
            + self.transaction_cost_bps / 10_000.0
        )


@dataclass(frozen=True)
class BacktestMetrics:
    """Summary statistics computed from one realized walk-forward path.

    ``absolute_comparator_discrepancy`` and ``one_sided_utility_regret`` are
    the supported comparator metrics.  ``dynamic_regret`` and
    ``mean_step_regret`` are retained as deprecated compatibility fields for
    older callers; they expose the former gross-comparator/net-strategy gap
    and must not be interpreted as conventional dynamic regret.
    """

    n_observations: int
    n_evaluated: int
    n_predictions: int
    gross_return: float
    net_return: float
    total_transaction_cost: float
    turnover: float
    average_turnover: float
    sharpe: float
    max_drawdown: float
    information_coefficient: float
    hit_rate: float
    dynamic_regret: Optional[float]
    mean_step_regret: Optional[float]
    detection_delays: Tuple[Optional[int], ...] = ()
    recovery_delays: Tuple[Optional[int], ...] = ()
    false_promotions: int = 0
    baseline_comparisons: int = 0
    absolute_comparator_discrepancy: Optional[float] = None
    mean_step_absolute_comparator_discrepancy: Optional[float] = None
    one_sided_utility_regret: Optional[float] = None
    mean_step_utility_regret: Optional[float] = None

    @property
    def total_return(self) -> float:
        return self.net_return

    @property
    def net_pnl(self) -> float:
        return self.net_return

    @property
    def gross_pnl(self) -> float:
        return self.gross_return

    @property
    def ic(self) -> float:
        return self.information_coefficient

    @property
    def sharpe_ratio(self) -> float:
        return self.sharpe

    @property
    def drawdown(self) -> float:
        return self.max_drawdown

    @property
    def detection_delay(self) -> Optional[float]:
        delays = [delay for delay in self.detection_delays if delay is not None]
        return None if not delays else float(np.mean(delays))

    @property
    def recovery(self) -> Optional[float]:
        delays = [delay for delay in self.recovery_delays if delay is not None]
        return None if not delays else float(np.mean(delays))

    @property
    def mean_detection_delay(self) -> Optional[float]:
        return self.detection_delay

    @property
    def mean_recovery_delay(self) -> Optional[float]:
        return self.recovery

    @property
    def false_promotion_rate(self) -> float:
        if self.baseline_comparisons == 0:
            return 0.0
        return float(self.false_promotions / self.baseline_comparisons)

    @property
    def false_promotion(self) -> bool:
        return self.false_promotions > 0

    @property
    def comparator_discrepancy(self) -> Optional[float]:
        """Short alias for the supported absolute discrepancy metric."""

        return self.absolute_comparator_discrepancy

    @property
    def mean_step_comparator_discrepancy(self) -> Optional[float]:
        """Mean per-evaluated-row absolute comparator discrepancy."""

        return self.mean_step_absolute_comparator_discrepancy

    @property
    def utility_regret(self) -> Optional[float]:
        """Short alias for one-sided utility regret."""

        return self.one_sided_utility_regret

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)

    def keys(self) -> Tuple[str, ...]:
        return tuple(self.to_dict().keys())

    def items(self) -> Tuple[Tuple[str, Any], ...]:
        return tuple(self.to_dict().items())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_observations": self.n_observations,
            "n_evaluated": self.n_evaluated,
            "n_predictions": self.n_predictions,
            "gross_return": self.gross_return,
            "net_return": self.net_return,
            "total_return": self.total_return,
            "total_transaction_cost": self.total_transaction_cost,
            "turnover": self.turnover,
            "average_turnover": self.average_turnover,
            "sharpe": self.sharpe,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "drawdown": self.drawdown,
            "information_coefficient": self.information_coefficient,
            "ic": self.ic,
            "hit_rate": self.hit_rate,
            "absolute_comparator_discrepancy": self.absolute_comparator_discrepancy,
            "comparator_discrepancy": self.comparator_discrepancy,
            "mean_step_absolute_comparator_discrepancy": self.mean_step_absolute_comparator_discrepancy,
            "mean_step_comparator_discrepancy": self.mean_step_comparator_discrepancy,
            "one_sided_utility_regret": self.one_sided_utility_regret,
            "utility_regret": self.utility_regret,
            "mean_step_utility_regret": self.mean_step_utility_regret,
            # Kept solely for source compatibility.  See the class docstring.
            "dynamic_regret": self.dynamic_regret,
            "mean_step_regret": self.mean_step_regret,
            "dynamic_regret_status": "deprecated_legacy_comparator_gap",
            "detection_delays": self.detection_delays,
            "detection_delay": self.detection_delay,
            "recovery_delays": self.recovery_delays,
            "recovery": self.recovery,
            "false_promotions": self.false_promotions,
            "false_promotion": self.false_promotion,
            "false_promotion_rate": self.false_promotion_rate,
        }


@dataclass(frozen=True)
class BaselineComparison:
    """Early-promotion versus later out-of-sample comparison."""

    name: str
    candidate_return: float
    baseline_return: float
    excess_return: float
    early_candidate_return: float
    early_baseline_return: float
    later_candidate_return: float
    later_baseline_return: float
    promoted: bool
    false_promotion: bool

    @property
    def candidate_net_return(self) -> float:
        return self.candidate_return

    @property
    def baseline_net_return(self) -> float:
        return self.baseline_return

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "candidate_return": self.candidate_return,
            "baseline_return": self.baseline_return,
            "excess_return": self.excess_return,
            "early_candidate_return": self.early_candidate_return,
            "early_baseline_return": self.early_baseline_return,
            "later_candidate_return": self.later_candidate_return,
            "later_baseline_return": self.later_baseline_return,
            "promoted": self.promoted,
            "false_promotion": self.false_promotion,
        }


@dataclass(frozen=True)
class BacktestResult:
    """Full path plus an audit trail and optional baseline comparisons."""

    index: Tuple[Any, ...]
    predictions: NDArray[Any]
    positions: NDArray[Any]
    gross_returns: NDArray[Any]
    transaction_costs: NDArray[Any]
    net_returns: NDArray[Any]
    equity_curve: NDArray[Any]
    evaluated: NDArray[Any]
    trained: NDArray[Any]
    training_positions: Tuple[Tuple[int, ...], ...]
    metrics: BacktestMetrics
    dynamic_regret_series: Optional[NDArray[Any]] = None
    oracle_returns: Optional[NDArray[Any]] = None
    baseline_results: Mapping[str, "BacktestResult"] = field(default_factory=dict)
    baseline_comparisons: Mapping[str, BaselineComparison] = field(default_factory=dict)
    targets: Optional[NDArray[Any]] = None
    absolute_comparator_discrepancy_series: Optional[NDArray[Any]] = None
    one_sided_utility_regret_series: Optional[NDArray[Any]] = None
    oracle_utility_returns: Optional[NDArray[Any]] = None

    @property
    def times(self) -> Tuple[Any, ...]:
        return self.index

    @property
    def indices(self) -> Tuple[Any, ...]:
        """Compatibility alias for callers using the shared result vocabulary."""

        return self.index

    @property
    def returns(self) -> NDArray[Any]:
        return np.array(self.net_returns, copy=True)

    @property
    def signal(self) -> NDArray[Any]:
        return np.array(self.predictions, copy=True)

    @property
    def turnover(self) -> float:
        return self.metrics.turnover

    @property
    def sharpe(self) -> float:
        return self.metrics.sharpe

    @property
    def max_drawdown(self) -> float:
        return self.metrics.max_drawdown

    @property
    def final_equity(self) -> float:
        return float(self.equity_curve[-1]) if len(self.equity_curve) else 1.0

    def metric(self, name: str) -> Any:
        return self.metrics[name]


class _MeanLabelModel:
    """Dependency-free fallback model used when no predictor is supplied."""

    def __init__(self) -> None:
        self.value: Any = 0.0

    def fit(self, x_train: NDArray[Any], y_train: NDArray[Any]) -> "_MeanLabelModel":
        del x_train
        if len(y_train) == 0:
            self.value = 0.0
            return self
        try:
            self.value = float(np.mean(np.asarray(y_train, dtype=float)))
        except (TypeError, ValueError):
            values, counts = np.unique(y_train, return_counts=True)
            self.value = values[int(np.argmax(counts))]
        return self

    def predict(self, x_test: NDArray[Any]) -> NDArray[Any]:
        return np.full(len(x_test), self.value)


def _coerce_features(features: Any) -> TimeIndexedFeatures:
    if isinstance(features, TimeIndexedFeatures):
        return features
    return TimeIndexedFeatures.from_array(features)


def _coerce_labels(labels: Any, index: Sequence[Any]) -> Optional[TimeIndexedLabels]:
    if labels is None:
        return None
    if isinstance(labels, TimeIndexedLabels):
        return labels
    return TimeIndexedLabels.from_array(labels, index=index)


def _coerce_returns(
    returns: Any,
    n: int,
    *,
    required: bool = False,
) -> NDArray[Any]:
    if returns is None:
        if required:
            raise ValueError(
                "realized_returns are required for this walk-forward configuration"
            )
        return np.zeros(n, dtype=float)
    if isinstance(returns, TimeIndexedLabels):
        returns = returns.values
    return _as_1d_float(returns, n, "realized_returns")


def _coerce_walk_forward_config(config: Any) -> WalkForwardConfig:
    """Normalize the worker config and the package-level warmup contract."""

    if config is None:
        return WalkForwardConfig()
    if isinstance(config, WalkForwardConfig):
        return config
    if all(hasattr(config, field_name) for field_name in ("initial_train_size", "label_delay")):
        worker_fields = (
            "initial_train_size",
            "label_delay",
            "retrain_every",
            "training_window",
            "min_train_size",
            "transaction_cost",
            "slippage",
            "transaction_cost_bps",
            "max_position",
            "allow_short",
            "prediction_mode",
            "hold_last_position",
            "periods_per_year",
            "detection_persistence",
            "detection_position_threshold",
            "promotion_window",
            "require_realized_returns",
        )
        values = {
            field_name: getattr(config, field_name)
            for field_name in worker_fields
            if hasattr(config, field_name)
        }
        return WalkForwardConfig(**values)
    # ``topology_gate.types.BacktestConfig`` intentionally stays dependency
    # free and only exposes warmup/horizon/refit_interval.  Map it at the
    # worker boundary without importing the shared module's NumPy-free types.
    if all(hasattr(config, field_name) for field_name in ("warmup", "horizon")):
        horizon = int(config.horizon)
        return WalkForwardConfig(
            initial_train_size=int(config.warmup),
            # A horizon-one target is available after the next boundary; the
            # strict event loop below also requires availability < decision.
            label_delay=max(1, horizon),
            retrain_every=int(getattr(config, "refit_interval", None) or 1),
        )
    raise TypeError("config must be WalkForwardConfig or shared BacktestConfig")


def _label_feature_positions(labels: TimeIndexedLabels, feature_index: Tuple[Any, ...]) -> NDArray[Any]:
    feature_lookup = {value: position for position, value in enumerate(feature_index)}
    positions = []
    for value in labels.index:
        if value not in feature_lookup:
            raise ValueError(f"label time {value!r} is not present in the feature index")
        positions.append(feature_lookup[value])
    return np.asarray(positions, dtype=int)


def _availability_positions(
    labels: TimeIndexedLabels,
    target_positions: NDArray[Any],
    feature_index: Tuple[Any, ...],
    label_delay: int,
) -> NDArray[Any]:
    n = len(feature_index)
    if label_delay < 0:
        raise ValueError("label_delay must be non-negative")
    if labels.available_at is None:
        return target_positions + int(label_delay)
    lookup = {value: position for position, value in enumerate(feature_index)}
    result = np.empty(len(labels.index), dtype=int)
    for row, (target_position, available) in enumerate(
        zip(target_positions, labels.available_at)
    ):
        if available is None:
            available_position = int(target_position)
        elif available in lookup:
            available_position = int(lookup[available])
        elif isinstance(available, (int, np.integer)):
            # Integer positions also permit availability after the finite
            # sample, which is important for testing terminal label delay.
            available_position = int(available)
        else:
            try:
                available_position = next(
                    position
                    for position, time_value in enumerate(feature_index)
                    if time_value >= available
                )
            except StopIteration:
                available_position = n
            except TypeError as exc:
                raise TypeError("available_at values must be index values or integer positions") from exc
        result[row] = max(available_position, int(target_position) + int(label_delay))
    return result


def _normalise_change_points(
    change_points: Optional[Sequence[Any]], index: Tuple[Any, ...]
) -> Tuple[int, ...]:
    if change_points is None:
        return ()
    lookup = {value: position for position, value in enumerate(index)}
    positions = []
    for point in change_points:
        if point in lookup:
            positions.append(int(lookup[point]))
        else:
            positions.append(int(point))
    result = tuple(positions)
    if tuple(sorted(set(result))) != result:
        raise ValueError("change_points must be sorted and unique")
    if any(point <= 0 or point >= len(index) for point in result):
        raise ValueError("change_points must lie inside the feature sample")
    return result


def _infer_optimal_position(labels: Optional[TimeIndexedLabels], n: int) -> Optional[NDArray[Any]]:
    if labels is None:
        return None
    try:
        values = np.asarray(labels.values, dtype=float)
    except (TypeError, ValueError):
        return None
    if len(values) != n or np.any(~np.isfinite(values)):
        return None
    unique = np.unique(values)
    if len(unique) <= 2 and np.all(np.isin(unique, (-1.0, 0.0, 1.0))):
        return np.sign(values)
    return None


def _infer_change_points_from_regimes(regime_ids: Optional[NDArray[Any]]) -> Tuple[int, ...]:
    if regime_ids is None or len(regime_ids) < 2:
        return ()
    return tuple((np.flatnonzero(regime_ids[1:] != regime_ids[:-1]) + 1).tolist())


def _compute_detection_delays(
    positions: NDArray[Any],
    evaluated: NDArray[Any],
    optimal_position: Optional[NDArray[Any]],
    change_points: Tuple[int, ...],
    persistence: int,
    threshold: float,
) -> Tuple[Optional[int], ...]:
    if optimal_position is None:
        return ()
    delays: list[Optional[int]] = []
    n = len(positions)
    for change_point in change_points:
        detected: Optional[int] = None
        last_start = n - persistence
        for start in range(change_point, max(change_point, last_start) + 1):
            stop = start + persistence
            if stop > n:
                break
            aligned = (
                evaluated[start:stop]
                & np.isfinite(positions[start:stop])
                & (positions[start:stop] * optimal_position[start:stop] > threshold)
            )
            if bool(np.all(aligned)):
                detected = int(start - change_point)
                break
        delays.append(detected)
    return tuple(delays)


def _compute_recovery_delays(
    equity_curve: NDArray[Any],
    change_points: Tuple[int, ...],
) -> Tuple[Optional[int], ...]:
    delays: list[Optional[int]] = []
    n = len(equity_curve)
    for change_point in change_points:
        if change_point <= 0 or change_point >= n:
            delays.append(None)
            continue
        pre_shift_peak = float(np.max(equity_curve[:change_point]))
        recovered: Optional[int] = None
        for position in range(change_point, n):
            if equity_curve[position] >= pre_shift_peak:
                recovered = position - change_point
                break
        delays.append(recovered)
    return tuple(delays)


def _validate_comparator_position(
    optimal_position: NDArray[Any],
    *,
    max_position: float,
) -> None:
    """Validate the offline comparator against the action feasibility bound."""

    limit = float(max_position)
    if not np.isfinite(limit) or limit <= 0:
        raise ValueError("comparator max_position must be finite and positive")
    if np.any(np.abs(optimal_position) > limit + 1e-12):
        raise ValueError(
            "optimal_position contains an infeasible action outside "
            f"[-{limit}, {limit}]"
        )


def _costs_for_evaluated_positions(
    positions: NDArray[Any],
    evaluated: NDArray[Any],
    cost_rate: float,
) -> NDArray[Any]:
    """Charge turnover only at evaluated rows, carrying the last live action."""

    rate = float(cost_rate)
    if not np.isfinite(rate) or rate < 0:
        raise ValueError("cost_rate must be finite and non-negative")
    costs = np.zeros(len(positions), dtype=float)
    previous_position = 0.0
    for row, is_evaluated in enumerate(evaluated):
        if bool(is_evaluated):
            costs[row] = abs(float(positions[row]) - previous_position) * rate
            previous_position = float(positions[row])
    if np.any(~np.isfinite(costs)):
        raise ValueError("transaction costs are not finite")
    return costs


def _checked_product(
    left: NDArray[Any],
    right: NDArray[Any],
    name: str,
) -> NDArray[Any]:
    """Multiply metric inputs and fail rather than returning misleading inf."""

    with np.errstate(over="ignore", invalid="ignore"):
        result = np.asarray(left, dtype=float) * np.asarray(right, dtype=float)
    if np.any(~np.isfinite(result)):
        raise ValueError(f"{name} are not finite after multiplication")
    return result


def _comparator_metric_series(
    *,
    positions: NDArray[Any],
    net_returns: NDArray[Any],
    transaction_costs: NDArray[Any],
    evaluated: NDArray[Any],
    optimal_position: NDArray[Any],
    realized_returns: NDArray[Any],
    expected_returns: Optional[NDArray[Any]],
    comparator_transaction_costs: NDArray[Any],
) -> Tuple[
    NDArray[Any],
    NDArray[Any],
    NDArray[Any],
    NDArray[Any],
    NDArray[Any],
]:
    """Build legacy and cost-matched comparator series.

    The utility basis is deliberately common to both sides: point-in-time
    ``expected_returns`` when supplied, otherwise realized returns.  The
    realized path metrics remain based on ``realized_returns``; this function
    only constructs the offline comparator diagnostics.
    """

    basis = realized_returns if expected_returns is None else expected_returns
    masked_positions = np.where(evaluated, positions, 0.0)
    masked_basis = np.where(evaluated, basis, 0.0)
    masked_transaction_costs = np.where(evaluated, transaction_costs, 0.0)
    masked_comparator_costs = np.where(
        evaluated, comparator_transaction_costs, 0.0
    )
    strategy_utility = _checked_product(
        masked_positions, masked_basis, "strategy utility"
    )
    comparator_gross = _checked_product(
        np.where(evaluated, optimal_position, 0.0),
        masked_basis,
        "comparator utility",
    )
    strategy_utility = strategy_utility - masked_transaction_costs
    comparator_utility = comparator_gross - masked_comparator_costs
    if np.any(~np.isfinite(strategy_utility)) or np.any(~np.isfinite(comparator_utility)):
        raise ValueError("comparator utility is not finite after costs")

    # The old statistic is returned only for compatibility.  It intentionally
    # preserves the former gross-comparator versus realized-net convention.
    legacy_gap = comparator_gross - np.where(evaluated, net_returns, 0.0)
    utility_gap = comparator_utility - strategy_utility
    absolute_discrepancy = np.where(evaluated, np.abs(utility_gap), 0.0)
    one_sided_regret = np.where(evaluated, np.maximum(utility_gap, 0.0), 0.0)
    comparator_gross = np.where(evaluated, comparator_gross, 0.0)
    comparator_utility = np.where(evaluated, comparator_utility, 0.0)
    legacy_gap = np.where(evaluated, legacy_gap, 0.0)
    return (
        legacy_gap,
        comparator_gross,
        comparator_utility,
        absolute_discrepancy,
        one_sided_regret,
    )


def _build_metrics(
    *,
    predictions: NDArray[Any],
    positions: NDArray[Any],
    gross_returns: NDArray[Any],
    transaction_costs: NDArray[Any],
    net_returns: NDArray[Any],
    equity_curve: NDArray[Any],
    evaluated: NDArray[Any],
    labels_for_evaluation: Optional[NDArray[Any]],
    optimal_position: Optional[NDArray[Any]],
    change_points: Tuple[int, ...],
    periods_per_year: float,
    detection_persistence: int,
    detection_threshold: float,
    false_promotions: int = 0,
    baseline_comparisons: int = 0,
) -> Tuple[BacktestMetrics, Optional[NDArray[Any]], Optional[NDArray[Any]]]:
    evaluated = np.asarray(evaluated, dtype=bool)
    if evaluated.ndim != 1 or len(evaluated) != len(positions):
        raise ValueError("evaluated must align with positions")
    for values, name in (
        (positions, "positions"),
        (gross_returns, "gross_returns"),
        (transaction_costs, "transaction_costs"),
        (net_returns, "net_returns"),
    ):
        if np.any(~np.isfinite(values)):
            raise ValueError(f"{name} must contain only finite values")

    # Treat the evaluation mask as a hard boundary.  External callers may
    # provide non-zero arrays before evaluation, but those rows cannot affect
    # any aggregate or the equity curve.
    masked_gross = np.where(evaluated, gross_returns, 0.0)
    masked_costs = np.where(evaluated, transaction_costs, 0.0)
    masked_net = np.where(evaluated, net_returns, 0.0)
    metric_equity = 1.0 + np.cumsum(masked_net)
    valid_predictions = evaluated & np.isfinite(predictions)
    n_predictions = int(np.sum(valid_predictions))
    eval_returns = masked_net[evaluated]
    gross_return = float(np.sum(masked_gross))
    net_return = float(np.sum(masked_net))
    total_cost = float(np.sum(masked_costs))
    position_changes = _costs_for_evaluated_positions(positions, evaluated, 1.0)
    turnover = float(np.sum(position_changes))
    average_turnover = turnover / max(1, int(np.sum(evaluated)))
    if len(eval_returns) >= 2:
        standard_deviation = float(np.std(eval_returns, ddof=1))
    else:
        standard_deviation = 0.0
    if standard_deviation <= np.finfo(float).eps:
        sharpe = 0.0
    else:
        sharpe = float(np.mean(eval_returns) / standard_deviation * np.sqrt(periods_per_year))
    if len(metric_equity):
        peaks = np.maximum.accumulate(metric_equity)
        with np.errstate(divide="ignore", invalid="ignore"):
            drawdowns = np.where(
                peaks > 0, (peaks - metric_equity) / peaks, 0.0
            )
        max_drawdown = float(np.max(drawdowns)) if len(drawdowns) else 0.0
    else:
        max_drawdown = 0.0

    valid_labels = valid_predictions
    if labels_for_evaluation is not None:
        valid_labels = valid_predictions & np.isfinite(labels_for_evaluation)
    if labels_for_evaluation is None:
        information_coefficient = 0.0
        hit_rate = 0.0
    else:
        information_coefficient = _finite_correlation(
            predictions[valid_labels], labels_for_evaluation[valid_labels]
        )
        if int(np.sum(valid_labels)):
            hit_rate = float(
                np.mean(
                    np.sign(predictions[valid_labels])
                    == np.sign(labels_for_evaluation[valid_labels])
                )
            )
        else:
            hit_rate = 0.0

    dynamic_regret_series: Optional[NDArray[Any]]
    oracle_returns: Optional[NDArray[Any]]
    # Comparator diagnostics are attached by the caller after the common path
    # statistics have been built.  Keeping this helper independent prevents an
    # unevaluated comparator row from entering the realized metrics.
    dynamic_regret_series = None
    oracle_returns = None
    dynamic_regret = None
    mean_step_regret = None

    detection_delays = _compute_detection_delays(
        positions,
        evaluated,
        optimal_position,
        change_points,
        detection_persistence,
        detection_threshold,
    )
    recovery_delays = _compute_recovery_delays(metric_equity, change_points)
    metrics = BacktestMetrics(
        n_observations=len(positions),
        n_evaluated=int(np.sum(evaluated)),
        n_predictions=n_predictions,
        gross_return=gross_return,
        net_return=net_return,
        total_transaction_cost=total_cost,
        turnover=turnover,
        average_turnover=average_turnover,
        sharpe=sharpe,
        max_drawdown=max_drawdown,
        information_coefficient=information_coefficient,
        hit_rate=hit_rate,
        dynamic_regret=dynamic_regret,
        mean_step_regret=mean_step_regret,
        detection_delays=detection_delays,
        recovery_delays=recovery_delays,
        false_promotions=int(false_promotions),
        baseline_comparisons=int(baseline_comparisons),
    )
    return metrics, dynamic_regret_series, oracle_returns


def calculate_metrics(
    positions: ArrayLike,
    realized_returns: ArrayLike,
    *,
    predictions: Optional[ArrayLike] = None,
    labels: Optional[ArrayLike] = None,
    expected_returns: Optional[ArrayLike] = None,
    transaction_costs: Optional[ArrayLike] = None,
    comparator_transaction_costs: Optional[ArrayLike] = None,
    comparator_transaction_cost_rate: float = 0.0,
    comparator_max_position: float = 1.0,
    evaluated: Optional[Sequence[bool]] = None,
    optimal_position: Optional[ArrayLike] = None,
    change_points: Optional[Sequence[int]] = None,
    periods_per_year: float = 252.0,
    detection_persistence: int = 1,
    detection_position_threshold: float = 1e-12,
) -> BacktestMetrics:
    """Calculate all public metrics for an already-generated path.

    This hook is useful for comparing an external offline baseline.  Returns
    and positions are interpreted row-for-row; no data is fetched or inferred
    from outside the supplied arrays.
    """

    position_input = np.asarray(positions)
    if position_input.ndim == 0:
        position_input = position_input.reshape(1)
    position_array = _as_1d_float(positions, len(position_input.reshape(-1)), "positions")
    n = len(position_array)
    return_array = _as_1d_float(realized_returns, n, "realized_returns")
    prediction_array = (
        np.array(position_array, copy=True)
        if predictions is None
        else _as_1d_float(predictions, n, "predictions")
    )
    evaluation_mask = (
        np.ones(n, dtype=bool)
        if evaluated is None
        else np.asarray(evaluated, dtype=bool)
    )
    if evaluation_mask.ndim != 1 or len(evaluation_mask) != n:
        raise ValueError(f"evaluated must have length {n}")
    periods = float(periods_per_year)
    if not np.isfinite(periods) or periods <= 0.0:
        raise ValueError("periods_per_year must be finite and positive")
    if (
        isinstance(detection_persistence, bool)
        or not isinstance(detection_persistence, (int, np.integer))
        or int(detection_persistence) < 1
    ):
        raise ValueError("detection_persistence must be a positive integer")
    costs = (
        np.zeros(n, dtype=float)
        if transaction_costs is None
        else _as_1d_float(transaction_costs, n, "transaction_costs")
    )
    gross = _checked_product(position_array, return_array, "gross returns")
    net = gross - costs
    if np.any(~np.isfinite(net)):
        raise ValueError("net returns are not finite after transaction costs")
    equity = 1.0 + np.cumsum(np.where(evaluation_mask, net, 0.0))
    if np.any(~np.isfinite(equity)):
        raise ValueError("equity curve is not finite")
    optimal = (
        None
        if optimal_position is None
        else _as_1d_float(optimal_position, n, "optimal_position")
    )
    if optimal is not None:
        _validate_comparator_position(
            optimal, max_position=float(comparator_max_position)
        )
    expected_array = (
        None
        if expected_returns is None
        else _as_1d_float(expected_returns, n, "expected_returns")
    )
    if (
        optimal is not None
        and comparator_transaction_costs is not None
        and float(comparator_transaction_cost_rate) != 0.0
    ):
        raise ValueError(
            "pass either comparator_transaction_costs or "
            "comparator_transaction_cost_rate, not both"
        )
    if optimal is None and comparator_transaction_costs is not None:
        raise ValueError("comparator costs require optimal_position")
    if optimal is not None:
        if comparator_transaction_costs is not None:
            comparator_costs = _as_1d_float(
                comparator_transaction_costs, n, "comparator_transaction_costs"
            )
        else:
            rate = float(comparator_transaction_cost_rate)
            if not np.isfinite(rate) or rate < 0:
                raise ValueError(
                    "comparator_transaction_cost_rate must be finite and non-negative"
                )
            if rate == 0.0 and np.any(costs[evaluation_mask] != 0.0):
                raise ValueError(
                    "comparator transaction costs are required when evaluated "
                    "strategy costs are non-zero"
                )
            comparator_costs = _costs_for_evaluated_positions(
                optimal, evaluation_mask, rate
            )
    else:
        comparator_costs = np.zeros(n, dtype=float)
    labels_array = None if labels is None else _as_1d_float(labels, n, "labels")
    metrics, _, _ = _build_metrics(
        predictions=prediction_array,
        positions=position_array,
        gross_returns=gross,
        transaction_costs=costs,
        net_returns=net,
        equity_curve=equity,
        evaluated=evaluation_mask,
        labels_for_evaluation=labels_array,
        optimal_position=optimal,
        change_points=tuple(change_points or ()),
        periods_per_year=periods,
        detection_persistence=int(detection_persistence),
        detection_threshold=float(detection_position_threshold),
    )
    if optimal is not None:
        (
            legacy_gap,
            _comparator_gross,
            _comparator_utility,
            absolute_discrepancy,
            one_sided_regret,
        ) = _comparator_metric_series(
            positions=position_array,
            net_returns=net,
            transaction_costs=costs,
            evaluated=evaluation_mask,
            optimal_position=optimal,
            realized_returns=return_array,
            expected_returns=expected_array,
            comparator_transaction_costs=comparator_costs,
        )
        metrics = replace(
            metrics,
            # Deprecated compatibility result; use the explicit fields below.
            dynamic_regret=float(np.round(np.sum(np.abs(legacy_gap[evaluation_mask])), 12)),
            mean_step_regret=(
                float(np.round(np.mean(np.abs(legacy_gap[evaluation_mask])), 12))
                if np.any(evaluation_mask)
                else 0.0
            ),
            absolute_comparator_discrepancy=float(
                np.round(np.sum(absolute_discrepancy[evaluation_mask]), 12)
            ),
            mean_step_absolute_comparator_discrepancy=(
                float(np.round(np.mean(absolute_discrepancy[evaluation_mask]), 12))
                if np.any(evaluation_mask)
                else 0.0
            ),
            one_sided_utility_regret=float(
                np.round(np.sum(one_sided_regret[evaluation_mask]), 12)
            ),
            mean_step_utility_regret=(
                float(np.round(np.mean(one_sided_regret[evaluation_mask]), 12))
                if np.any(evaluation_mask)
                else 0.0
            ),
        )
    return metrics


class WalkForwardBacktest:
    """Strict chronological offline walk-forward engine."""

    def __init__(
        self,
        config: Optional[WalkForwardConfig] = None,
        model_factory: Any = None,
        *,
        predictor: Optional[Predictor] = None,
        baselines: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if model_factory is not None and predictor is not None:
            raise TypeError("pass either model_factory or predictor, not both")
        self.config = _coerce_walk_forward_config(config)
        self.model_factory = model_factory
        self.predictor = predictor
        self.baselines = dict(baselines or {})

    def run(
        self,
        features: TimeIndexedFeatures | SyntheticDataset | ArrayLike,
        labels: Optional[TimeIndexedLabels | ArrayLike] = None,
        realized_returns: Optional[ArrayLike] = None,
        *,
        config: Optional[Any] = None,
        model_factory: Any = None,
        predictor: Optional[Predictor] = None,
        model: Any = None,
        action_mapper: Optional[ActionMapper] = None,
        label_delay: Optional[int] = None,
        regime_ids: Optional[ArrayLike] = None,
        change_points: Optional[Sequence[Any]] = None,
        optimal_position: Optional[ArrayLike] = None,
        expected_returns: Optional[ArrayLike] = None,
        baseline_hooks: Optional[Mapping[str, Any]] = None,
        baselines: Optional[Mapping[str, Any]] = None,
        promotion_rule: Optional[Callable[["BacktestResult", "BacktestResult"], bool]] = None,
    ) -> BacktestResult:
        """Run one walk-forward path.

        At decision row ``t``, only labels whose target row is strictly less
        than ``t`` and whose availability position is strictly less than ``t`` enter the
        training set.  Features at row ``t`` may be used for the prediction at
        row ``t``.  The resulting training row positions are retained in
        ``BacktestResult.training_positions`` for audit and tests.
        """

        dataset: Optional[SyntheticDataset] = features if isinstance(features, SyntheticDataset) else None
        source_dataset: Any = None
        if dataset is not None:
            feature_frame = dataset.features
            if labels is None:
                labels = dataset.labels
            if realized_returns is None:
                realized_returns = dataset.realized_returns
            if regime_ids is None:
                regime_ids = dataset.regime_ids
            if change_points is None:
                change_points = dataset.change_points
            if optimal_position is None:
                optimal_position = dataset.optimal_position
            if expected_returns is None:
                expected_returns = dataset.expected_returns
        elif hasattr(features, "features"):
            # Structural compatibility for a worker-owned dataset protocol.
            # The engine still normalizes the resulting feature/label records
            # before entering the causal loop.
            source_dataset = features
            feature_frame = _coerce_features(getattr(source_dataset, "features"))
            if labels is None:
                labels = getattr(source_dataset, "labels", None)
                if labels is None:
                    labels = getattr(source_dataset, "targets", None)
            if realized_returns is None:
                realized_returns = getattr(source_dataset, "realized_returns", None)
                if realized_returns is None:
                    realized_returns = getattr(source_dataset, "returns", None)
            if regime_ids is None:
                regime_ids = getattr(source_dataset, "regime_ids", None)
                if regime_ids is None:
                    regime_ids = getattr(source_dataset, "regimes", None)
            if change_points is None:
                change_points = getattr(source_dataset, "change_points", None)
                if change_points is None:
                    change_points = getattr(source_dataset, "shift_points", None)
            if optimal_position is None:
                optimal_position = getattr(source_dataset, "optimal_position", None)
            if expected_returns is None:
                expected_returns = getattr(source_dataset, "expected_returns", None)
        else:
            feature_frame = _coerce_features(features)

        n = feature_frame.n_samples
        label_frame = _coerce_labels(labels, feature_frame.index)
        cfg = self.config if config is None else _coerce_walk_forward_config(config)
        returns = _coerce_returns(
            realized_returns,
            n,
            required=cfg.require_realized_returns,
        )
        expected: Optional[NDArray[Any]]
        optimal: Optional[NDArray[Any]]
        if expected_returns is not None:
            expected = _as_1d_float(expected_returns, n, "expected_returns")
        else:
            expected = None
        if optimal_position is not None:
            optimal = _as_1d_float(optimal_position, n, "optimal_position")
        else:
            optimal = _infer_optimal_position(label_frame, n)
        if regime_ids is not None:
            regimes = np.asarray(regime_ids)
            if regimes.ndim != 1 or len(regimes) != n:
                raise ValueError(f"regime_ids must have length {n}")
            regimes = np.array(regimes, copy=True)
        else:
            regimes = None
        points = _normalise_change_points(change_points, feature_frame.index)
        if not points:
            points = _infer_change_points_from_regimes(regimes)

        source_delay = getattr(source_dataset, "label_delay", None)
        if label_delay is None:
            delay = cfg.label_delay if source_delay is None else int(cast(int, source_delay))
        else:
            delay = int(label_delay)
        if delay < 0:
            raise ValueError("label_delay must be non-negative")
        if model is not None and (model_factory is not None or predictor is not None):
            raise TypeError("pass only one of model, model_factory, or predictor")
        predictor_spec = predictor if predictor is not None else self.predictor
        model_spec = model_factory if model_factory is not None else self.model_factory
        if model is not None:
            model_spec = model
        if predictor_spec is not None and model_spec is not None:
            raise TypeError("pass only one of model, model_factory, or predictor")
        # A three-argument callable supplied positionally as ``model_factory``
        # is almost certainly the documented predictor callback.  Accept it
        # as a convenience while retaining zero-argument model factories.
        if predictor_spec is None and model_spec is not None and not hasattr(model_spec, "fit"):
            required = _callable_required_positional_count(model_spec) if callable(model_spec) else None
            if required is not None and required > 0:
                predictor_spec = model_spec
                model_spec = None

        target_positions = (
            _label_feature_positions(label_frame, feature_frame.index)
            if label_frame is not None
            else np.empty(0, dtype=int)
        )
        availability = (
            _availability_positions(label_frame, target_positions, feature_frame.index, delay)
            if label_frame is not None
            else np.empty(0, dtype=int)
        )

        evaluation_start = min(max(int(cfg.initial_train_size), 0), n)
        evaluated = np.zeros(n, dtype=bool)
        evaluated[evaluation_start:] = True
        predictions = np.full(n, np.nan, dtype=float)
        positions = np.zeros(n, dtype=float)
        trained = np.zeros(n, dtype=bool)
        audit: list[Tuple[int, ...]] = [() for _ in range(n)]
        current_model: Any = None
        last_position = 0.0
        model_is_callback = predictor_spec is not None

        for decision_position in range(evaluation_start, n):
            eligible = np.flatnonzero(
                (target_positions < decision_position) & (availability < decision_position)
            )
            if cfg.training_window is not None:
                eligible = eligible[-int(cfg.training_window) :]
            if label_frame is not None:
                eligible = eligible[
                    :
                ]  # make the ndarray copy explicit for audit-safe slicing
            training_positions = tuple(int(target_positions[row]) for row in eligible)
            audit[decision_position] = training_positions
            enough_labels = len(training_positions) >= int(cfg.min_train_size)
            should_retrain = (
                decision_position == evaluation_start
                or (decision_position - evaluation_start) % int(cfg.retrain_every) == 0
            )

            raw_prediction: Any = np.nan
            if model_is_callback:
                # A callback is intentionally evaluated at each row so it can
                # implement a stateful, but still auditable, baseline/model.
                if enough_labels or label_frame is None:
                    x_train, y_train = self._training_arrays(
                        feature_frame, label_frame, target_positions, eligible
                    )
                    assert predictor_spec is not None
                    raw_prediction = _invoke_callback(
                        predictor_spec,
                        x_train,
                        y_train,
                        feature_frame.values[decision_position : decision_position + 1],
                    )
                    trained[decision_position] = True
            elif should_retrain and enough_labels:
                x_train, y_train = self._training_arrays(
                    feature_frame, label_frame, target_positions, eligible
                )
                current_model = self._fit_model(model_spec, x_train, y_train)
                trained[decision_position] = True
                raw_prediction = self._predict_model(
                    current_model, feature_frame.values[decision_position : decision_position + 1]
                )
            elif current_model is not None:
                raw_prediction = self._predict_model(
                    current_model, feature_frame.values[decision_position : decision_position + 1]
                )

            raw_prediction = _scalar_prediction(raw_prediction)
            try:
                prediction_value = float(raw_prediction)
            except (TypeError, ValueError):
                prediction_value = np.nan
            if not np.isfinite(prediction_value):
                predictions[decision_position] = np.nan
                if cfg.hold_last_position:
                    positions[decision_position] = last_position
                else:
                    positions[decision_position] = 0.0
            else:
                predictions[decision_position] = prediction_value
                positions[decision_position] = _map_prediction_to_position(
                    prediction_value,
                    label_frame.values[eligible] if label_frame is not None and len(eligible) else np.array([]),
                    cfg.prediction_mode,
                    float(cfg.max_position),
                    bool(cfg.allow_short),
                    action_mapper,
                )
            last_position = positions[decision_position]

        costs, gross, net, equity = self._returns_from_positions(positions, evaluated, returns, cfg)
        labels_for_evaluation = self._aligned_label_values(
            label_frame, target_positions, n
        )
        (
            metrics,
            regret_series,
            oracle_returns,
            absolute_discrepancy_series,
            utility_regret_series,
            oracle_utility_returns,
        ) = self._metrics_for_path(
            predictions=predictions,
            positions=positions,
            gross=gross,
            costs=costs,
            net=net,
            equity=equity,
            evaluated=evaluated,
            labels=labels_for_evaluation,
            optimal=optimal,
            realized_returns=returns,
            expected=expected,
            change_points=points,
            config=cfg,
        )
        result = BacktestResult(
            index=feature_frame.index,
            predictions=predictions,
            positions=positions,
            gross_returns=gross,
            transaction_costs=costs,
            net_returns=net,
            equity_curve=equity,
            evaluated=evaluated,
            trained=trained,
            training_positions=tuple(audit),
            metrics=metrics,
            dynamic_regret_series=regret_series,
            oracle_returns=oracle_returns,
            targets=labels_for_evaluation,
            absolute_comparator_discrepancy_series=absolute_discrepancy_series,
            one_sided_utility_regret_series=utility_regret_series,
            oracle_utility_returns=oracle_utility_returns,
        )

        hooks: Dict[str, Any] = {}
        hooks.update(self.baselines)
        if baselines:
            hooks.update(baselines)
        if baseline_hooks:
            hooks.update(baseline_hooks)
        if hooks:
            baseline_results: Dict[str, BacktestResult] = {}
            comparisons: Dict[str, BaselineComparison] = {}
            false_promotions = 0
            for name, hook in hooks.items():
                baseline_result = self._run_baseline(
                    name=str(name),
                    hook=hook,
                    feature_frame=feature_frame,
                    evaluated=evaluated,
                    returns=returns,
                    optimal=optimal,
                    expected=expected,
                    points=points,
                    config=cfg,
                )
                comparison = compare_to_baseline(
                    result,
                    baseline_result,
                    name=str(name),
                    promotion_window=cfg.promotion_window,
                    promotion_rule=promotion_rule,
                )
                baseline_results[str(name)] = baseline_result
                comparisons[str(name)] = comparison
                false_promotions += int(comparison.false_promotion)
            result = replace(
                result,
                baseline_results=baseline_results,
                baseline_comparisons=comparisons,
                metrics=replace(
                    result.metrics,
                    false_promotions=false_promotions,
                    baseline_comparisons=len(comparisons),
                ),
            )
        return result

    @staticmethod
    def _training_arrays(
        feature_frame: TimeIndexedFeatures,
        label_frame: Optional[TimeIndexedLabels],
        target_positions: NDArray[Any],
        eligible: NDArray[Any],
    ) -> Tuple[NDArray[Any], NDArray[Any]]:
        if label_frame is None:
            return np.empty((0, feature_frame.n_features), dtype=float), np.empty(0)
        rows = target_positions[eligible]
        return (
            np.array(feature_frame.values[rows], dtype=float, copy=True),
            np.array(label_frame.values[eligible], copy=True),
        )

    @staticmethod
    def _fit_model(model_spec: Any, x_train: NDArray[Any], y_train: NDArray[Any]) -> Any:
        if model_spec is None:
            model = _MeanLabelModel()
        elif isinstance(model_spec, type):
            model = model_spec()
        elif hasattr(model_spec, "fit"):
            model = _clone_model(model_spec)
        elif callable(model_spec):
            required = _callable_required_positional_count(model_spec)
            if required is not None and required > 0:
                raise TypeError(
                    "a callable requiring arguments is a predictor; pass it as predictor=..."
                )
            model = model_spec()
        else:
            raise TypeError("model_factory must be a fit/predict object or zero-argument factory")
        if not hasattr(model, "fit") or not hasattr(model, "predict"):
            raise TypeError("model factory must produce an object with fit and predict")
        fitted = model.fit(x_train, y_train)
        return model if fitted is None else fitted

    @staticmethod
    def _predict_model(model: Any, x_test: NDArray[Any]) -> Any:
        if model is None or not hasattr(model, "predict"):
            return np.nan
        return model.predict(x_test)

    @staticmethod
    def _aligned_label_values(
        label_frame: Optional[TimeIndexedLabels], target_positions: NDArray[Any], n: int
    ) -> Optional[NDArray[Any]]:
        if label_frame is None:
            return None
        # Evaluation labels are used only after the path has been generated.
        # This array is metric ground truth, never input to the model loop.
        result = np.full(n, np.nan, dtype=float)
        try:
            values = np.asarray(label_frame.values, dtype=float)
        except (TypeError, ValueError):
            return None
        result[target_positions] = values
        return result

    @staticmethod
    def _returns_from_positions(
        positions: NDArray[Any],
        evaluated: NDArray[Any],
        realized_returns: NDArray[Any],
        config: WalkForwardConfig,
    ) -> Tuple[NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]]:
        costs = _costs_for_evaluated_positions(
            positions, evaluated, config.turnover_cost_rate
        )
        masked_positions = np.where(evaluated, positions, 0.0)
        masked_returns = np.where(evaluated, realized_returns, 0.0)
        gross = _checked_product(masked_positions, masked_returns, "gross returns")
        net = gross - costs
        if np.any(~np.isfinite(net)):
            raise ValueError("net returns are not finite after transaction costs")
        equity = 1.0 + np.cumsum(net)
        if np.any(~np.isfinite(equity)):
            raise ValueError("equity curve is not finite")
        return costs, gross, net, equity

    def _metrics_for_path(
        self,
        *,
        predictions: NDArray[Any],
        positions: NDArray[Any],
        gross: NDArray[Any],
        costs: NDArray[Any],
        net: NDArray[Any],
        equity: NDArray[Any],
        evaluated: NDArray[Any],
        labels: Optional[NDArray[Any]],
        optimal: Optional[NDArray[Any]],
        realized_returns: NDArray[Any],
        expected: Optional[NDArray[Any]],
        change_points: Tuple[int, ...],
        config: WalkForwardConfig,
    ) -> Tuple[
        BacktestMetrics,
        Optional[NDArray[Any]],
        Optional[NDArray[Any]],
        Optional[NDArray[Any]],
        Optional[NDArray[Any]],
        Optional[NDArray[Any]],
    ]:
        metrics, _, _ = _build_metrics(
            predictions=predictions,
            positions=positions,
            gross_returns=gross,
            transaction_costs=costs,
            net_returns=net,
            equity_curve=equity,
            evaluated=evaluated,
            labels_for_evaluation=labels,
            optimal_position=optimal,
            change_points=change_points,
            periods_per_year=config.periods_per_year,
            detection_persistence=config.detection_persistence,
            detection_threshold=config.detection_position_threshold,
        )
        if optimal is None:
            return metrics, None, None, None, None, None
        _validate_comparator_position(optimal, max_position=config.max_position)
        oracle_base = realized_returns if expected is None else expected
        comparator_costs, oracle_returns, oracle_utility, _ = self._returns_from_positions(
            optimal, evaluated, oracle_base, config
        )
        (
            legacy_gap,
            comparator_gross,
            comparator_utility,
            absolute_discrepancy,
            one_sided_regret,
        ) = _comparator_metric_series(
            positions=positions,
            net_returns=net,
            transaction_costs=costs,
            evaluated=evaluated,
            optimal_position=optimal,
            realized_returns=realized_returns,
            expected_returns=expected,
            comparator_transaction_costs=comparator_costs,
        )
        del comparator_gross, comparator_utility
        valid = np.asarray(evaluated, dtype=bool)
        metrics = replace(
            metrics,
            # Deprecated compatibility result; use the explicit fields below.
            dynamic_regret=float(np.round(np.sum(np.abs(legacy_gap[valid])), 12)),
            mean_step_regret=(
                float(np.round(np.mean(np.abs(legacy_gap[valid])), 12))
                if np.any(valid)
                else 0.0
            ),
            absolute_comparator_discrepancy=float(
                np.round(np.sum(absolute_discrepancy[valid]), 12)
            ),
            mean_step_absolute_comparator_discrepancy=(
                float(np.round(np.mean(absolute_discrepancy[valid]), 12))
                if np.any(valid)
                else 0.0
            ),
            one_sided_utility_regret=float(
                np.round(np.sum(one_sided_regret[valid]), 12)
            ),
            mean_step_utility_regret=(
                float(np.round(np.mean(one_sided_regret[valid]), 12))
                if np.any(valid)
                else 0.0
            ),
        )
        # ``dynamic_regret_series`` remains the old signed gap for callers
        # that have not migrated.  New consumers should use the two explicit
        # series returned on BacktestResult.
        return (
            metrics,
            legacy_gap,
            oracle_returns,
            absolute_discrepancy,
            one_sided_regret,
            oracle_utility,
        )

    def _run_baseline(
        self,
        *,
        name: str,
        hook: Any,
        feature_frame: TimeIndexedFeatures,
        evaluated: NDArray[Any],
        returns: NDArray[Any],
        optimal: Optional[NDArray[Any]],
        expected: Optional[NDArray[Any]],
        points: Tuple[int, ...],
        config: WalkForwardConfig,
    ) -> BacktestResult:
        if isinstance(hook, BacktestResult):
            if hook.index != feature_frame.index:
                raise ValueError(f"baseline {name!r} uses a different time index")
            return hook
        if not callable(hook):
            array = _as_1d_float(hook, len(feature_frame.index), f"baseline {name}")
            raw_positions = array
        else:
            raw_positions = self._baseline_positions(hook, feature_frame, evaluated)
        raw_positions = np.asarray(raw_positions, dtype=float)
        raw_positions = np.where(np.isfinite(raw_positions), raw_positions, 0.0)
        raw_positions = np.clip(raw_positions, -config.max_position, config.max_position)
        if not config.allow_short:
            raw_positions = np.clip(raw_positions, 0.0, config.max_position)
        raw_positions = np.where(evaluated, raw_positions, 0.0)
        costs, gross, net, equity = self._returns_from_positions(
            raw_positions, evaluated, returns, config
        )
        predictions = np.array(raw_positions, copy=True)
        labels = None
        (
            metrics,
            regret_series,
            oracle_returns,
            absolute_discrepancy_series,
            utility_regret_series,
            oracle_utility_returns,
        ) = self._metrics_for_path(
            predictions=predictions,
            positions=raw_positions,
            gross=gross,
            costs=costs,
            net=net,
            equity=equity,
            evaluated=evaluated,
            labels=labels,
            optimal=optimal,
            realized_returns=returns,
            expected=expected,
            change_points=points,
            config=config,
        )
        return BacktestResult(
            index=feature_frame.index,
            predictions=predictions,
            positions=raw_positions,
            gross_returns=gross,
            transaction_costs=costs,
            net_returns=net,
            equity_curve=equity,
            evaluated=np.array(evaluated, copy=True),
            trained=np.zeros(len(evaluated), dtype=bool),
            training_positions=tuple(() for _ in evaluated),
            metrics=metrics,
            dynamic_regret_series=regret_series,
            oracle_returns=oracle_returns,
            targets=None,
            absolute_comparator_discrepancy_series=absolute_discrepancy_series,
            one_sided_utility_regret_series=utility_regret_series,
            oracle_utility_returns=oracle_utility_returns,
        )

    @staticmethod
    def _baseline_positions(
        hook: BaselineHook,
        feature_frame: TimeIndexedFeatures,
        evaluated: NDArray[Any],
    ) -> NDArray[Any]:
        n = feature_frame.n_samples
        positions = np.zeros(n, dtype=float)
        required = _callable_required_positional_count(hook)
        if required == 0:
            value = hook()
            array = np.asarray(value)
            if array.ndim == 1 and len(array) == n:
                return np.asarray(array, dtype=float)
            raise ValueError("a zero-argument baseline hook must return one value per row")
        for position in np.flatnonzero(evaluated):
            row = feature_frame.values[position : position + 1]
            if required is None or required >= 2:
                value = hook(int(position), row)
            elif required == 1:
                value = hook(row)
            else:
                value = hook()
            positions[position] = float(_scalar_prediction(value))
        return positions

def compare_to_baseline(
    candidate: BacktestResult,
    baseline: BacktestResult,
    *,
    name: str = "baseline",
    promotion_window: Optional[int] = None,
    min_excess: float = 0.0,
    promotion_rule: Optional[Callable[[BacktestResult, BacktestResult], bool]] = None,
) -> BaselineComparison:
    """Compare a candidate with a baseline without mixing their time paths.

    The default promotion policy uses the first ``promotion_window`` evaluated
    observations.  A false promotion occurs when the candidate wins that
    early promotion period but loses on the remaining observations.  A custom
    ``promotion_rule`` can provide a different offline selection policy.
    """

    if candidate.index != baseline.index:
        raise ValueError("candidate and baseline must use the same time index")
    mask = candidate.evaluated & baseline.evaluated
    rows = np.flatnonzero(mask)
    if len(rows) == 0:
        return BaselineComparison(
            name=name,
            candidate_return=0.0,
            baseline_return=0.0,
            excess_return=0.0,
            early_candidate_return=0.0,
            early_baseline_return=0.0,
            later_candidate_return=0.0,
            later_baseline_return=0.0,
            promoted=False,
            false_promotion=False,
        )
    if promotion_window is None:
        window = max(1, len(rows) // 2)
    else:
        window = max(1, min(int(promotion_window), len(rows)))
    early_rows = rows[:window]
    later_rows = rows[window:]
    early_candidate = float(np.sum(candidate.net_returns[early_rows]))
    early_baseline = float(np.sum(baseline.net_returns[early_rows]))
    later_candidate = float(np.sum(candidate.net_returns[later_rows]))
    later_baseline = float(np.sum(baseline.net_returns[later_rows]))
    candidate_return = float(np.sum(candidate.net_returns[rows]))
    baseline_return = float(np.sum(baseline.net_returns[rows]))
    if promotion_rule is None:
        promoted = early_candidate > early_baseline + float(min_excess)
    else:
        promoted = bool(promotion_rule(candidate, baseline))
    false_promotion = bool(
        promoted and len(later_rows) > 0 and later_candidate < later_baseline - float(min_excess)
    )
    return BaselineComparison(
        name=name,
        candidate_return=candidate_return,
        baseline_return=baseline_return,
        excess_return=candidate_return - baseline_return,
        early_candidate_return=early_candidate,
        early_baseline_return=early_baseline,
        later_candidate_return=later_candidate,
        later_baseline_return=later_baseline,
        promoted=promoted,
        false_promotion=false_promotion,
    )


def walk_forward_backtest(
    features: TimeIndexedFeatures | SyntheticDataset | ArrayLike,
    labels: Optional[TimeIndexedLabels | ArrayLike] = None,
    model_factory: Any = None,
    realized_returns: Optional[ArrayLike] = None,
    *,
    config: Optional[WalkForwardConfig] = None,
    predictor: Optional[Predictor] = None,
    model: Any = None,
    action_mapper: Optional[ActionMapper] = None,
    label_delay: Optional[int] = None,
    regime_ids: Optional[ArrayLike] = None,
    change_points: Optional[Sequence[Any]] = None,
    optimal_position: Optional[ArrayLike] = None,
    expected_returns: Optional[ArrayLike] = None,
    baseline_hooks: Optional[Mapping[str, Any]] = None,
    baselines: Optional[Mapping[str, Any]] = None,
    promotion_rule: Optional[Callable[[BacktestResult, BacktestResult], bool]] = None,
) -> BacktestResult:
    """Functional wrapper around :class:`WalkForwardBacktest`."""

    engine = WalkForwardBacktest(config=config, predictor=predictor, baselines=baselines)
    result = engine.run(
        features,
        labels,
        realized_returns,
        model_factory=model_factory,
        model=model,
        action_mapper=action_mapper,
        label_delay=label_delay,
        regime_ids=regime_ids,
        change_points=change_points,
        optimal_position=optimal_position,
        expected_returns=expected_returns,
        baseline_hooks=baseline_hooks,
        promotion_rule=promotion_rule,
    )
    return result


# Names commonly used by callers.
WalkForwardBacktester = WalkForwardBacktest
BacktestEngine = WalkForwardBacktest
BacktestConfig = WalkForwardConfig
run_walk_forward_backtest = walk_forward_backtest
OfflineWalkForwardEngine = WalkForwardBacktest
compute_metrics = calculate_metrics
compare_baselines = compare_to_baseline


__all__ = [
    "WalkForwardConfig",
    "BacktestConfig",
    "BacktestMetrics",
    "BaselineComparison",
    "BacktestResult",
    "WalkForwardBacktest",
    "WalkForwardBacktester",
    "BacktestEngine",
    "OfflineWalkForwardEngine",
    "calculate_metrics",
    "compute_metrics",
    "compare_to_baseline",
    "compare_baselines",
    "walk_forward_backtest",
    "run_walk_forward_backtest",
]
