"""End-to-end causal composition of the topology detector and recursive RLS."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, cast

import numpy as np

MAX_ONLINE_ROWS = 100_000
MAX_PENDING_LABELS = 8_192


def _finite_matrix(value: Any, name: str) -> np.ndarray[Any, Any]:
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite two-dimensional array")
    return np.array(array, copy=True)


def _finite_vector(value: Any, name: str, n: int | None = None) -> np.ndarray[Any, Any]:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1 or (n is not None and array.size != n) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite one-dimensional array")
    return np.array(array, copy=True)


def _state_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return result


@dataclass(frozen=True, slots=True)
class PendingLabelRecord:
    """One delayed label retained at the terminal boundary of a replay."""

    source_step: int
    available_step: int
    features: tuple[float, ...]
    target: float
    forgetting_factor: float

    def __post_init__(self) -> None:
        source = _state_int(self.source_step, "source_step")
        available = _state_int(self.available_step, "available_step")
        if available <= source:
            raise ValueError("available_step must be greater than source_step")
        values = tuple(float(value) for value in self.features)
        if not values or not all(math.isfinite(value) for value in values):
            raise ValueError("features must be a non-empty finite tuple")
        target = float(self.target)
        factor = float(self.forgetting_factor)
        if not math.isfinite(target):
            raise ValueError("target must be finite")
        if not math.isfinite(factor) or not 0.0 < factor <= 1.0:
            raise ValueError("forgetting_factor must be finite and in (0, 1]")
        object.__setattr__(self, "source_step", source)
        object.__setattr__(self, "available_step", available)
        object.__setattr__(self, "features", values)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "forgetting_factor", factor)

    def state_dict(self) -> dict[str, Any]:
        return {
            "source_step": self.source_step,
            "available_step": self.available_step,
            "features": list(self.features),
            "target": self.target,
            "forgetting_factor": self.forgetting_factor,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "PendingLabelRecord":
        if not isinstance(state, Mapping):
            raise ValueError("pending label state must be a mapping")
        return cls(
            source_step=cast(int, state.get("source_step")),
            available_step=cast(int, state.get("available_step")),
            features=tuple(state.get("features", ())),
            target=cast(float, state.get("target")),
            forgetting_factor=cast(float, state.get("forgetting_factor")),
        )


@dataclass(frozen=True, slots=True)
class OnlineStreamState:
    """Serializable state needed to continue an online replay."""

    next_step: int
    previous_position: float
    pending_labels: tuple[PendingLabelRecord, ...]
    feature_count: int
    max_pending_labels: int = 8192

    def __post_init__(self) -> None:
        next_step = _state_int(self.next_step, "next_step")
        feature_count = _state_int(self.feature_count, "feature_count", minimum=1)
        max_pending = _state_int(
            self.max_pending_labels, "max_pending_labels", minimum=1
        )
        previous = float(self.previous_position)
        if not math.isfinite(previous):
            raise ValueError("previous_position must be finite")
        pending = tuple(self.pending_labels)
        if len(pending) > max_pending:
            raise ValueError("pending_labels exceeds max_pending_labels")
        if any(not isinstance(item, PendingLabelRecord) for item in pending):
            raise ValueError("pending_labels must contain PendingLabelRecord values")
        if any(item.source_step >= next_step for item in pending):
            raise ValueError("pending label source_step must precede next_step")
        sources = [item.source_step for item in pending]
        if len(set(sources)) != len(sources):
            raise ValueError("pending_labels must not contain duplicate source steps")
        if any(len(item.features) != feature_count for item in pending):
            raise ValueError("pending label feature dimension does not match state")
        if tuple(sorted(pending, key=lambda item: item.source_step)) != pending:
            raise ValueError("pending_labels must be ordered by source_step")
        object.__setattr__(self, "next_step", next_step)
        object.__setattr__(self, "previous_position", previous)
        object.__setattr__(self, "pending_labels", pending)
        object.__setattr__(self, "feature_count", feature_count)
        object.__setattr__(self, "max_pending_labels", max_pending)

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "schema": "topology_gate.online.stream",
            "next_step": self.next_step,
            "previous_position": self.previous_position,
            "feature_count": self.feature_count,
            "max_pending_labels": self.max_pending_labels,
            "pending_labels": [item.state_dict() for item in self.pending_labels],
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, Any],
        *,
        max_pending_labels: int | None = None,
    ) -> "OnlineStreamState":
        if not isinstance(state, Mapping) or state.get("version") != 1:
            raise ValueError("unsupported online stream state")
        raw_pending = state.get("pending_labels", ())
        if isinstance(raw_pending, (str, bytes, bytearray)):
            raise ValueError("pending_labels must be a sequence")
        pending = tuple(PendingLabelRecord.from_state_dict(item) for item in raw_pending)
        configured_limit = state.get(
            "max_pending_labels", 8192
        ) if max_pending_labels is None else max_pending_labels
        return cls(
            next_step=cast(int, state.get("next_step")),
            previous_position=cast(float, state.get("previous_position")),
            pending_labels=pending,
            feature_count=cast(int, state.get("feature_count")),
            max_pending_labels=configured_limit,
        )


@dataclass(frozen=True, slots=True)
class OnlineRunConfig:
    label_delay: int = 0
    transaction_cost_bps: float = 0.0
    position_limit: float = 1.0
    position_scale: float = 1.0
    annualization: float = 252.0
    reset_state: bool = True
    max_pending_labels: int = 8192

    def __post_init__(self) -> None:
        if isinstance(self.label_delay, bool) or not isinstance(
            self.label_delay, (int, np.integer)
        ):
            raise ValueError("label_delay must be an integer")
        label_delay = int(self.label_delay)
        if label_delay < 0 or label_delay > MAX_ONLINE_ROWS:
            raise ValueError("label_delay must be non-negative")
        if self.transaction_cost_bps < 0 or not math.isfinite(self.transaction_cost_bps):
            raise ValueError("transaction_cost_bps must be finite and non-negative")
        if self.position_limit <= 0 or not math.isfinite(self.position_limit):
            raise ValueError("position_limit must be finite and positive")
        if self.position_scale <= 0 or not math.isfinite(self.position_scale):
            raise ValueError("position_scale must be finite and positive")
        if self.annualization <= 0 or not math.isfinite(self.annualization):
            raise ValueError("annualization must be finite and positive")
        max_pending = _state_int(self.max_pending_labels, "max_pending_labels", minimum=1)
        if max_pending > MAX_PENDING_LABELS:
            raise ValueError("max_pending_labels exceeds the configured resource limit")
        object.__setattr__(self, "label_delay", label_delay)
        object.__setattr__(self, "max_pending_labels", max_pending)


@dataclass(frozen=True, slots=True)
class OnlineRunResult:
    predictions: np.ndarray[Any, Any]
    positions: np.ndarray[Any, Any]
    outcomes: np.ndarray[Any, Any]
    realized_returns: np.ndarray[Any, Any]
    gross_returns: np.ndarray[Any, Any]
    transaction_costs: np.ndarray[Any, Any]
    net_returns: np.ndarray[Any, Any]
    detector_scores: np.ndarray[Any, Any]
    alarms: np.ndarray[Any, Any]
    forgetting_factors: np.ndarray[Any, Any]
    update_steps: np.ndarray[Any, Any]
    metrics: dict[str, float]
    pending_labels: tuple[PendingLabelRecord, ...] = ()
    stream_state: OnlineStreamState | None = None
    learner_state: Mapping[str, Any] | None = None
    detector_state: Mapping[str, Any] | None = None

    def state_dict(self) -> dict[str, Any]:
        """Return the terminal online state, including unresolved labels."""

        if self.stream_state is None:
            raise ValueError("online result has no stream state")
        return self.stream_state.state_dict()


def _correlation(left: np.ndarray[Any, Any], right: np.ndarray[Any, Any]) -> float:
    if (
        left.size < 2
        or not np.all(np.isfinite(left))
        or not np.all(np.isfinite(right))
        or np.std(left) <= 1e-15
        or np.std(right) <= 1e-15
    ):
        return 0.0
    result = float(np.corrcoef(left, right)[0, 1])
    return result if math.isfinite(result) else 0.0


def _max_drawdown(returns: np.ndarray[Any, Any]) -> float:
    equity = 1.0 + np.cumsum(returns)
    if not np.all(np.isfinite(equity)):
        raise ValueError("online equity curve is not finite")
    peak = np.maximum.accumulate(equity)
    drawdown = np.where(peak > 0, (peak - equity) / peak, 0.0)
    return float(np.max(drawdown)) if drawdown.size else 0.0


def _checked_product(left: float, right: float, name: str) -> float:
    with np.errstate(over="ignore", invalid="ignore"):
        result = float(left * right)
    if not math.isfinite(result):
        raise ValueError(f"{name} is not finite after multiplication")
    return result


def _detection_delays(
    alarms: np.ndarray[Any, Any], shift_points: tuple[int, ...] | None
) -> float:
    if not shift_points:
        return math.nan
    delays: list[int] = []
    for index, shift in enumerate(shift_points):
        stop = shift_points[index + 1] if index + 1 < len(shift_points) else alarms.size
        candidates = np.flatnonzero(alarms[shift:stop])
        delays.append(int(candidates[0]) if candidates.size else int(stop - shift))
    return float(np.mean(delays)) if delays else math.nan


def run_recursive_rls(
    features: Any,
    outcomes: Any,
    *,
    realized_returns: Any | None = None,
    learner: Any,
    detector: Any | None = None,
    market_states: Any | None = None,
    config: OnlineRunConfig | None = None,
    shift_points: tuple[int, ...] | None = None,
    label_available_at: Any | None = None,
    initial_state: OnlineStreamState | None = None,
) -> OnlineRunResult:
    """Run a causal RLS stream with optional topology-gated forgetting.

    At row ``t`` the learner predicts first. The current label is incorporated
    immediately only when ``label_delay == 0``; otherwise it is queued until
    its availability boundary. The forgetting factor is captured at prediction
    time and cannot be changed by a future detector observation. ``initial_state``
    continues a prior chunk when ``config.reset_state`` is false; availability
    positions are absolute stream positions in that mode.
    """

    settings = config or OnlineRunConfig()
    x = _finite_matrix(features, "features")
    if x.shape[0] > MAX_ONLINE_ROWS:
        raise ValueError(f"features exceed the online row limit ({MAX_ONLINE_ROWS})")
    y = _finite_vector(outcomes, "outcomes", x.shape[0])
    realized = y if realized_returns is None else _finite_vector(
        realized_returns, "realized_returns", x.shape[0]
    )
    states = x if market_states is None else _finite_matrix(market_states, "market_states")
    if states.shape[0] != x.shape[0]:
        raise ValueError("market_states and features must have the same number of rows")
    if initial_state is not None:
        if settings.reset_state:
            raise ValueError("initial_state requires config.reset_state=False")
        if initial_state.feature_count != x.shape[1]:
            raise ValueError("initial_state feature dimension does not match features")
        if len(initial_state.pending_labels) > settings.max_pending_labels:
            raise ValueError("initial_state exceeds max_pending_labels")
        start_step = initial_state.next_step
        pending: list[PendingLabelRecord] = list(initial_state.pending_labels)
        previous_position = initial_state.previous_position
    else:
        start_step = 0
        pending = []
        previous_position = 0.0
    availability: Any
    if label_available_at is None:
        availability = (
            np.arange(x.shape[0], dtype=int)
            + start_step
            + settings.label_delay
        )
    else:
        if settings.label_delay != 0:
            raise ValueError("set either label_delay or label_available_at, not both")
        availability_array = np.asarray(label_available_at)
        if availability_array.ndim == 2 and 1 in availability_array.shape:
            availability_array = availability_array.reshape(-1)
        if availability_array.ndim != 1 or availability_array.size != x.shape[0]:
            raise ValueError("label_available_at must align with the feature rows")
        try:
            availability_float = np.asarray(availability_array, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("label_available_at must contain finite integer steps") from exc
        if not np.all(np.isfinite(availability_float)) or not np.all(
            availability_float == np.floor(availability_float)
        ):
            raise ValueError("label_available_at must contain finite integer steps")
        availability = availability_float.astype(int)
        source_steps = np.arange(x.shape[0], dtype=int) + start_step
        if np.any(availability <= source_steps):
            raise ValueError("each label must become available strictly after its source step")
    if getattr(learner, "n_features", x.shape[1]) != x.shape[1]:
        raise ValueError("learner feature dimension does not match features")
    if settings.reset_state:
        reset_learner = getattr(learner, "reset", None)
        if callable(reset_learner):
            reset_learner()
        reset_detector = getattr(detector, "reset_stream", None)
        if callable(reset_detector):
            reset_detector()

    n = x.shape[0]
    predictions = np.zeros(n, dtype=float)
    positions = np.zeros(n, dtype=float)
    gross = np.zeros(n, dtype=float)
    costs = np.zeros(n, dtype=float)
    scores = np.zeros(n, dtype=float)
    alarms = np.zeros(n, dtype=bool)
    factors = np.ones(n, dtype=float)
    update_steps = np.zeros(n, dtype=bool)
    for t in range(n):
        global_step = start_step + t
        if pending:
            due = [entry for entry in pending if entry.available_step <= global_step]
            pending = [entry for entry in pending if entry.available_step > global_step]
            for entry in due:
                learner.update(
                    np.asarray(entry.features, dtype=float),
                    entry.target,
                    forgetting_factor=entry.forgetting_factor,
                )
                local_source = entry.source_step - start_step
                if 0 <= local_source < n:
                    update_steps[local_source] = True

        if detector is None:
            factor = 1.0
        else:
            detection = detector.observe(states[t])
            scores[t] = float(detection.score)
            alarms[t] = bool(detection.alarm)
            factor = float(detection.forgetting_factor)
        factors[t] = factor

        raw_prediction = learner.predict(x[t])
        prediction = float(np.asarray(raw_prediction, dtype=float).reshape(-1)[0])
        predictions[t] = prediction
        position = float(
            np.clip(
                prediction / settings.position_scale,
                -settings.position_limit,
                settings.position_limit,
            )
        )
        positions[t] = position
        gross[t] = _checked_product(position, float(realized[t]), "gross return")
        costs[t] = abs(position - previous_position) * settings.transaction_cost_bps / 10_000.0
        if not math.isfinite(float(costs[t])):
            raise ValueError("transaction cost is not finite")
        previous_position = position

        if label_available_at is None and settings.label_delay == 0:
            learner.update(x[t], y[t], forgetting_factor=factor)
            update_steps[t] = True
        else:
            if len(pending) >= settings.max_pending_labels:
                raise ValueError("pending label queue exceeds max_pending_labels")
            pending.append(
                PendingLabelRecord(
                    source_step=global_step,
                    available_step=int(availability[t]),
                    features=tuple(float(value) for value in x[t]),
                    target=float(y[t]),
                    forgetting_factor=factor,
                )
            )

    net = gross - costs
    if not np.all(np.isfinite(net)):
        raise ValueError("online net returns are not finite")
    volatility = float(np.std(net, ddof=1)) if n > 1 else 0.0
    metrics = {
        "mse": float(np.mean((predictions - y) ** 2)),
        "information_coefficient": _correlation(predictions, y),
        "mean_net_return": float(np.mean(net)),
        "net_sharpe": 0.0
        if volatility <= 1e-15
        else float(np.mean(net) / volatility * math.sqrt(settings.annualization)),
        "max_drawdown": _max_drawdown(net),
        "turnover": float(np.mean(np.abs(np.diff(np.r_[0.0, positions])))),
        "mean_detection_delay": _detection_delays(alarms, shift_points),
        "false_alarm_count": float(np.count_nonzero(alarms[: shift_points[0]]))
        if shift_points
        else math.nan,
    }
    pending_records = tuple(pending)
    stream_state = OnlineStreamState(
        next_step=start_step + n,
        previous_position=previous_position,
        pending_labels=pending_records,
        feature_count=x.shape[1],
        max_pending_labels=settings.max_pending_labels,
    )
    learner_state_fn = getattr(learner, "state_dict", None)
    detector_state_fn = getattr(detector, "stream_state_dict", None)
    learner_state = learner_state_fn() if callable(learner_state_fn) else None
    detector_state = detector_state_fn() if callable(detector_state_fn) else None
    return OnlineRunResult(
        predictions=predictions,
        positions=positions,
        outcomes=y,
        realized_returns=realized,
        gross_returns=gross,
        transaction_costs=costs,
        net_returns=net,
        detector_scores=scores,
        alarms=alarms,
        forgetting_factors=factors,
        update_steps=update_steps,
        metrics=metrics,
        pending_labels=pending_records,
        stream_state=stream_state,
        learner_state=learner_state,
        detector_state=detector_state,
    )


__all__ = [
    "OnlineRunConfig",
    "OnlineRunResult",
    "OnlineStreamState",
    "MAX_ONLINE_ROWS",
    "MAX_PENDING_LABELS",
    "PendingLabelRecord",
    "run_recursive_rls",
]
