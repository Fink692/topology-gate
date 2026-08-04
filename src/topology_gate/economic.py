"""Strict economic-evaluation contracts.

The worker backtest retains a compatibility mode for old synthetic examples.
This module is the fail-closed boundary for any result that might be described
as a tradable economic path: it requires separately sourced realized returns,
timestamped execution-cost rates, explicit positions, and visible abstentions.
It does not provide a market-data vendor, slippage estimator, capacity model,
or a claim that supplied rates are realistic.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Any

from .asof import TimePoint

ECONOMIC_SCHEMA = "topology_gate.economic"
ECONOMIC_VERSION = 1
MAX_ECONOMIC_STEPS = 100_000


class EconomicEvaluationError(ValueError):
    """Base error for missing, unavailable, or invalid economic evidence."""


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EconomicEvaluationError(f"{name} must be a non-empty string")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise EconomicEvaluationError(f"{name} must be finite")
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise EconomicEvaluationError(f"{name} must be finite") from exc
    if not math.isfinite(converted):
        raise EconomicEvaluationError(f"{name} must be finite")
    return converted


def _nonnegative(value: Any, name: str) -> float:
    converted = _finite(value, name)
    if converted < 0.0:
        raise EconomicEvaluationError(f"{name} must be non-negative")
    return converted


def _time_key(value: TimePoint) -> tuple[str, Any]:
    if isinstance(value, datetime):
        return ("datetime", value.isoformat())
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise EconomicEvaluationError("time points must be int, float, str, or datetime")
    if isinstance(value, float) and not math.isfinite(value):
        raise EconomicEvaluationError("time points must be finite")
    return (type(value).__name__, value)


def _encode_time(value: TimePoint) -> dict[str, Any]:
    kind, raw = _time_key(value)
    return {"kind": kind, "value": raw}


def _le(left: TimePoint, right: TimePoint, name: str) -> bool:
    try:
        return bool(left <= right)  # type: ignore[operator]
    except TypeError as exc:
        raise EconomicEvaluationError(f"{name} values use different time domains") from exc


def _lt(left: TimePoint, right: TimePoint, name: str) -> bool:
    try:
        return bool(left < right)  # type: ignore[operator]
    except TypeError as exc:
        raise EconomicEvaluationError(f"{name} values use different time domains") from exc


@dataclass(frozen=True, slots=True)
class EconomicDecision:
    """One position emitted at a decision boundary."""

    decision_id: str
    target_id: str
    decision_time: TimePoint
    position: float
    evaluated: bool = True
    reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "decision_id", _text(self.decision_id, "decision_id"))
        object.__setattr__(self, "target_id", _text(self.target_id, "target_id"))
        _time_key(self.decision_time)
        position = _finite(self.position, "position")
        object.__setattr__(self, "position", position)
        if not isinstance(self.evaluated, bool):
            raise EconomicEvaluationError("evaluated must be boolean")
        if self.reason is not None:
            object.__setattr__(self, "reason", _text(self.reason, "reason"))
        if self.evaluated and self.reason is not None:
            raise EconomicEvaluationError("evaluated decisions must not carry an abstention reason")
        if not self.evaluated and self.reason is None:
            raise EconomicEvaluationError("abstentions require an explicit reason")

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "target_id": self.target_id,
            "decision_time": _encode_time(self.decision_time),
            "position": self.position,
            "evaluated": self.evaluated,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class RealizedReturn:
    """A separately sourced, point-in-time realized return observation."""

    target_id: str
    decision_time: TimePoint
    realization_time: TimePoint
    available_time: TimePoint
    value: float
    source_revision: int = 0
    status: str = "observed"

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_id", _text(self.target_id, "target_id"))
        _time_key(self.decision_time)
        _time_key(self.realization_time)
        _time_key(self.available_time)
        if not _le(self.decision_time, self.realization_time, "decision/realization"):
            raise EconomicEvaluationError("realization_time cannot precede decision_time")
        if not _le(self.realization_time, self.available_time, "realization/availability"):
            raise EconomicEvaluationError("available_time cannot precede realization_time")
        object.__setattr__(self, "value", _finite(self.value, "realized return"))
        if isinstance(self.source_revision, bool) or not isinstance(self.source_revision, int):
            raise EconomicEvaluationError("source_revision must be a non-negative integer")
        if self.source_revision < 0:
            raise EconomicEvaluationError("source_revision must be non-negative")
        status = _text(self.status, "return status").lower()
        if status not in {"observed", "missing", "censored", "invalid"}:
            raise EconomicEvaluationError("return status is unsupported")
        object.__setattr__(self, "status", status)
        if status != "observed":
            raise EconomicEvaluationError("non-observed returns must use a missing-value record")

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "decision_time": _encode_time(self.decision_time),
            "realization_time": _encode_time(self.realization_time),
            "available_time": _encode_time(self.available_time),
            "value": self.value,
            "source_revision": self.source_revision,
            "status": self.status,
        }


@dataclass(frozen=True, slots=True)
class ExecutionCost:
    """Timestamped per-unit-turnover execution-cost rates."""

    target_id: str
    decision_time: TimePoint
    execution_time: TimePoint
    available_time: TimePoint
    cost_model_id: str
    fee_rate: float = 0.0
    spread_rate: float = 0.0
    slippage_rate: float = 0.0
    impact_rate: float = 0.0
    other_rate: float = 0.0
    source_revision: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_id", _text(self.target_id, "target_id"))
        _time_key(self.decision_time)
        _time_key(self.execution_time)
        _time_key(self.available_time)
        if not _le(self.decision_time, self.execution_time, "decision/execution"):
            raise EconomicEvaluationError("execution_time cannot precede decision_time")
        if not _le(self.available_time, self.execution_time, "cost availability/execution"):
            raise EconomicEvaluationError(
                "available_time must not be after execution_time"
            )
        object.__setattr__(self, "cost_model_id", _text(self.cost_model_id, "cost_model_id"))
        for name in (
            "fee_rate",
            "spread_rate",
            "slippage_rate",
            "impact_rate",
            "other_rate",
        ):
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name))
        if isinstance(self.source_revision, bool) or not isinstance(self.source_revision, int):
            raise EconomicEvaluationError("source_revision must be a non-negative integer")
        if self.source_revision < 0:
            raise EconomicEvaluationError("source_revision must be non-negative")

    @property
    def total_rate(self) -> float:
        return _finite(
            self.fee_rate
            + self.spread_rate
            + self.slippage_rate
            + self.impact_rate
            + self.other_rate,
            "total execution cost rate",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "decision_time": _encode_time(self.decision_time),
            "execution_time": _encode_time(self.execution_time),
            "available_time": _encode_time(self.available_time),
            "cost_model_id": self.cost_model_id,
            "fee_rate": self.fee_rate,
            "spread_rate": self.spread_rate,
            "slippage_rate": self.slippage_rate,
            "impact_rate": self.impact_rate,
            "other_rate": self.other_rate,
            "source_revision": self.source_revision,
        }


@dataclass(frozen=True, slots=True)
class EconomicEvaluationConfig:
    """Fail-closed policy for one economic-path evaluation."""

    evaluation_id: str = "economic-evaluation"
    cost_model_id: str = "cost-model"
    initial_position: float = 0.0
    max_position: float = 1.0
    annualization: float = 252.0
    max_steps: int = MAX_ECONOMIC_STEPS

    def __post_init__(self) -> None:
        object.__setattr__(self, "evaluation_id", _text(self.evaluation_id, "evaluation_id"))
        object.__setattr__(self, "cost_model_id", _text(self.cost_model_id, "cost_model_id"))
        initial = _finite(self.initial_position, "initial_position")
        maximum = _finite(self.max_position, "max_position")
        annualization = _finite(self.annualization, "annualization")
        if maximum <= 0.0 or abs(initial) > maximum or annualization <= 0.0:
            raise EconomicEvaluationError("invalid economic evaluation bounds")
        object.__setattr__(self, "initial_position", initial)
        object.__setattr__(self, "max_position", maximum)
        object.__setattr__(self, "annualization", annualization)
        if isinstance(self.max_steps, bool) or not isinstance(self.max_steps, int):
            raise EconomicEvaluationError("max_steps must be an integer")
        if self.max_steps < 1 or self.max_steps > MAX_ECONOMIC_STEPS:
            raise EconomicEvaluationError("max_steps exceeds the resource limit")

    @property
    def identity(self) -> str:
        payload = {
            "schema": ECONOMIC_SCHEMA,
            "version": ECONOMIC_VERSION,
            "evaluation_id": self.evaluation_id,
            "cost_model_id": self.cost_model_id,
            "initial_position": self.initial_position,
            "max_position": self.max_position,
            "annualization": self.annualization,
            "max_steps": self.max_steps,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class EconomicPathRow:
    """One evaluated or explicitly abstained economic row."""

    decision: EconomicDecision
    realized_return: float | None
    turnover: float
    gross_return: float
    fee_cost: float
    spread_cost: float
    slippage_cost: float
    impact_cost: float
    other_cost: float
    net_return: float

    @property
    def total_cost(self) -> float:
        return self.fee_cost + self.spread_cost + self.slippage_cost + self.impact_cost + self.other_cost


@dataclass(frozen=True, slots=True)
class EconomicEvaluationResult:
    """Costed path and summary metrics; no comparator or alpha claim."""

    config_identity: str
    rows: tuple[EconomicPathRow, ...]
    metrics: Mapping[str, float]
    digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.config_identity, str) or not self.config_identity:
            raise EconomicEvaluationError("config_identity must be non-empty")
        object.__setattr__(self, "rows", tuple(self.rows))
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))
        if not isinstance(self.digest, str) or len(self.digest) != 64:
            raise EconomicEvaluationError("economic result digest must be SHA-256")

    @property
    def net_returns(self) -> tuple[float, ...]:
        return tuple(row.net_return for row in self.rows)

    @property
    def positions(self) -> tuple[float, ...]:
        return tuple(row.decision.position for row in self.rows)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": ECONOMIC_SCHEMA,
            "version": ECONOMIC_VERSION,
            "config_identity": self.config_identity,
            "rows": [
                {
                    "decision": row.decision.to_dict(),
                    "realized_return": row.realized_return,
                    "turnover": row.turnover,
                    "gross_return": row.gross_return,
                    "fee_cost": row.fee_cost,
                    "spread_cost": row.spread_cost,
                    "slippage_cost": row.slippage_cost,
                    "impact_cost": row.impact_cost,
                    "other_cost": row.other_cost,
                    "net_return": row.net_return,
                }
                for row in self.rows
            ],
            "metrics": dict(self.metrics),
            "digest": self.digest,
        }


def _max_drawdown(values: list[float]) -> float:
    equity = 1.0
    peak = 1.0
    drawdown = 0.0
    for value in values:
        equity += value
        if not math.isfinite(equity):
            raise EconomicEvaluationError("economic equity curve is not finite")
        peak = max(peak, equity)
        if peak > 0.0:
            drawdown = max(drawdown, (peak - equity) / peak)
    return drawdown


def _correlation_free_sharpe(values: list[float], annualization: float) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    volatility = math.sqrt(max(0.0, variance))
    return 0.0 if volatility <= 1.0e-15 else mean / volatility * math.sqrt(annualization)


def evaluate_economic_path(
    decisions: Iterable[EconomicDecision],
    realized_returns: Mapping[str, RealizedReturn],
    execution_costs: Mapping[str, ExecutionCost],
    *,
    config: EconomicEvaluationConfig | None = None,
) -> EconomicEvaluationResult:
    """Evaluate an explicit position path using separately sourced evidence.

    Every evaluated decision must have a matching observed realized return and
    cost record. Abstentions are retained in the output but contribute neither
    return nor cost. The evaluator never substitutes a supervised target,
    zero return, or zero cost for missing evidence.
    """

    settings = config or EconomicEvaluationConfig()
    decision_values = tuple(decisions)
    if len(decision_values) > settings.max_steps:
        raise EconomicEvaluationError("decisions exceed the configured resource limit")
    if not all(isinstance(item, EconomicDecision) for item in decision_values):
        raise EconomicEvaluationError("decisions must contain EconomicDecision values")
    previous_time: TimePoint | None = None
    seen_decisions: set[str] = set()
    seen_targets: set[str] = set()
    rows: list[EconomicPathRow] = []
    previous_position = settings.initial_position
    component_totals = {name: 0.0 for name in ("fee", "spread", "slippage", "impact", "other")}
    evaluated_count = 0
    abstained_count = 0

    for decision in decision_values:
        if decision.decision_id in seen_decisions or decision.target_id in seen_targets:
            raise EconomicEvaluationError("decision and target IDs must be unique")
        seen_decisions.add(decision.decision_id)
        seen_targets.add(decision.target_id)
        if previous_time is not None and not _lt(previous_time, decision.decision_time, "decision ordering"):
            raise EconomicEvaluationError("decisions must be in strictly increasing time order")
        previous_time = decision.decision_time
        if abs(decision.position) > settings.max_position:
            raise EconomicEvaluationError("position exceeds max_position")
        if not decision.evaluated:
            if abs(decision.position) > 1.0e-15:
                raise EconomicEvaluationError("abstentions must have zero position")
            if abs(previous_position) > 1.0e-15:
                raise EconomicEvaluationError(
                    "an abstention after a non-zero position requires an explicit execution row"
                )
            abstained_count += 1
            rows.append(
                EconomicPathRow(
                    decision=decision,
                    realized_return=None,
                    turnover=0.0,
                    gross_return=0.0,
                    fee_cost=0.0,
                    spread_cost=0.0,
                    slippage_cost=0.0,
                    impact_cost=0.0,
                    other_cost=0.0,
                    net_return=0.0,
                )
            )
            previous_position = 0.0
            continue

        try:
            realized = realized_returns[decision.target_id]
        except KeyError as exc:
            raise EconomicEvaluationError(
                f"realized return is missing for evaluated target {decision.target_id!r}"
            ) from exc
        if not isinstance(realized, RealizedReturn) or realized.target_id != decision.target_id:
            raise EconomicEvaluationError("realized return identity does not match decision")
        if realized.decision_time != decision.decision_time:
            raise EconomicEvaluationError("realized return decision time does not match decision")
        try:
            cost = execution_costs[decision.target_id]
        except KeyError as exc:
            raise EconomicEvaluationError(
                f"execution cost is missing for evaluated target {decision.target_id!r}"
            ) from exc
        if not isinstance(cost, ExecutionCost) or cost.target_id != decision.target_id:
            raise EconomicEvaluationError("execution cost identity does not match decision")
        if cost.decision_time != decision.decision_time:
            raise EconomicEvaluationError("execution cost decision time does not match decision")
        if cost.cost_model_id != settings.cost_model_id:
            raise EconomicEvaluationError("execution cost model identity does not match config")
        turnover = abs(decision.position - previous_position)
        fee = turnover * cost.fee_rate
        spread = turnover * cost.spread_rate
        slippage = turnover * cost.slippage_rate
        impact = turnover * cost.impact_rate
        other = turnover * cost.other_rate
        gross = decision.position * realized.value
        net = gross - fee - spread - slippage - impact - other
        for name, value in (
            ("fee", fee),
            ("spread", spread),
            ("slippage", slippage),
            ("impact", impact),
            ("other", other),
            ("gross", gross),
            ("net", net),
        ):
            _finite(value, f"{name} return")
        for name, value in (
            ("fee", fee),
            ("spread", spread),
            ("slippage", slippage),
            ("impact", impact),
            ("other", other),
        ):
            component_totals[name] += value
        evaluated_count += 1
        rows.append(
            EconomicPathRow(
                decision=decision,
                realized_return=realized.value,
                turnover=turnover,
                gross_return=gross,
                fee_cost=fee,
                spread_cost=spread,
                slippage_cost=slippage,
                impact_cost=impact,
                other_cost=other,
                net_return=net,
            )
        )
        previous_position = decision.position

    net_values = [row.net_return for row in rows]
    evaluated_values = [row.net_return for row in rows if row.decision.evaluated]
    metrics = {
        "evaluated_count": float(evaluated_count),
        "abstained_count": float(abstained_count),
        "mean_net_return": 0.0
        if not evaluated_values
        else sum(evaluated_values) / len(evaluated_values),
        "net_sharpe": _correlation_free_sharpe(evaluated_values, settings.annualization),
        "max_drawdown": _max_drawdown(net_values),
        "turnover": sum(row.turnover for row in rows),
        "fee_cost": component_totals["fee"],
        "spread_cost": component_totals["spread"],
        "slippage_cost": component_totals["slippage"],
        "impact_cost": component_totals["impact"],
        "other_cost": component_totals["other"],
        "total_net_return": sum(net_values),
    }
    payload = {
        "schema": ECONOMIC_SCHEMA,
        "version": ECONOMIC_VERSION,
        "config_identity": settings.identity,
        "rows": [
            {
                "decision": row.decision.to_dict(),
                "realized_return": row.realized_return,
                "turnover": row.turnover,
                "gross_return": row.gross_return,
                "fee_cost": row.fee_cost,
                "spread_cost": row.spread_cost,
                "slippage_cost": row.slippage_cost,
                "impact_cost": row.impact_cost,
                "other_cost": row.other_cost,
                "net_return": row.net_return,
            }
            for row in rows
        ],
        "metrics": metrics,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return EconomicEvaluationResult(settings.identity, tuple(rows), metrics, digest)


__all__ = [
    "ECONOMIC_SCHEMA",
    "ECONOMIC_VERSION",
    "EconomicDecision",
    "EconomicEvaluationConfig",
    "EconomicEvaluationError",
    "EconomicEvaluationResult",
    "EconomicPathRow",
    "ExecutionCost",
    "MAX_ECONOMIC_STEPS",
    "RealizedReturn",
    "evaluate_economic_path",
]
