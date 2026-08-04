"""Strict economic-evaluation contracts.

The worker backtest retains a compatibility mode for old synthetic examples.
This module is the fail-closed boundary for any result that might be described
as a tradable economic path: it requires separately sourced realized returns,
timestamped execution-cost rates, explicit positions, and visible abstentions.
It does not provide a market-data vendor, slippage estimator, or liquidity
model; optional capacity limits are caller-supplied evidence and are never
estimated here.
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
ECONOMIC_VERSION = 2
ECONOMIC_EVIDENCE_SCHEMA = "topology_gate.economic_evidence"
ECONOMIC_EVIDENCE_VERSION = 1
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


def _decode_time(value: Any, name: str) -> TimePoint:
    if not isinstance(value, Mapping) or set(value) != {"kind", "value"}:
        raise EconomicEvaluationError(f"{name} must be an encoded time point")
    kind = value.get("kind")
    raw = value.get("value")
    if kind == "datetime":
        if not isinstance(raw, str) or not raw:
            raise EconomicEvaluationError(f"{name} datetime value is invalid")
        try:
            return datetime.fromisoformat(raw)
        except ValueError as exc:
            raise EconomicEvaluationError(f"{name} datetime value is invalid") from exc
    if kind == "int":
        if type(raw) is not int:
            raise EconomicEvaluationError(f"{name} integer value is invalid")
        return raw
    if kind == "float":
        if type(raw) is not float or not math.isfinite(raw):
            raise EconomicEvaluationError(f"{name} float value is invalid")
        return raw
    if kind == "str":
        if not isinstance(raw, str) or not raw:
            raise EconomicEvaluationError(f"{name} string value is invalid")
        return raw
    raise EconomicEvaluationError(f"{name} time kind is unsupported")


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
    value: float | None
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
        if isinstance(self.source_revision, bool) or not isinstance(self.source_revision, int):
            raise EconomicEvaluationError("source_revision must be a non-negative integer")
        if self.source_revision < 0:
            raise EconomicEvaluationError("source_revision must be non-negative")
        status = _text(self.status, "return status").lower()
        if status not in {"observed", "missing", "censored", "invalid"}:
            raise EconomicEvaluationError("return status is unsupported")
        object.__setattr__(self, "status", status)
        if status == "observed":
            if self.value is None:
                raise EconomicEvaluationError("observed returns require a finite value")
            object.__setattr__(self, "value", _finite(self.value, "realized return"))
        elif self.value is not None:
            raise EconomicEvaluationError(
                "non-observed returns must use value=None, never a zero placeholder"
            )

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
    """Timestamped cost rates and optional turnover-capacity evidence.

    ``capacity_limit`` is expressed in the same normalized position-turnover
    units as the evaluator.  It is evidence supplied by the caller's
    liquidity/capacity model, not an estimate manufactured by this module.
    """

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
    capacity_limit: float | None = None

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
        if self.capacity_limit is not None:
            capacity = _finite(self.capacity_limit, "capacity_limit")
            if capacity <= 0.0:
                raise EconomicEvaluationError("capacity_limit must be positive")
            object.__setattr__(self, "capacity_limit", capacity)

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
            "capacity_limit": self.capacity_limit,
        }


@dataclass(frozen=True, slots=True)
class EconomicEvidence:
    """Versioned realized-return and execution-cost source artifact.

    A bundle may contain source revisions for the same target.  Call
    :meth:`select_at` with the declared evidence cutoff to obtain the two
    mappings consumed by :func:`evaluate_economic_path`; selecting a final
    table without an explicit cutoff is intentionally not implicit.
    """

    source_id: str
    realized_returns: tuple[RealizedReturn, ...] = ()
    execution_costs: tuple[ExecutionCost, ...] = ()
    schema: str = ECONOMIC_EVIDENCE_SCHEMA
    version: int = ECONOMIC_EVIDENCE_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", _text(self.source_id, "source_id"))
        if self.schema != ECONOMIC_EVIDENCE_SCHEMA:
            raise EconomicEvaluationError(
                f"schema must be exactly {ECONOMIC_EVIDENCE_SCHEMA!r}"
            )
        if type(self.version) is not int or self.version != ECONOMIC_EVIDENCE_VERSION:
            raise EconomicEvaluationError(
                f"version must be exactly {ECONOMIC_EVIDENCE_VERSION}"
            )
        returns = tuple(self.realized_returns)
        costs = tuple(self.execution_costs)
        if not all(isinstance(item, RealizedReturn) for item in returns):
            raise EconomicEvaluationError(
                "realized_returns must contain RealizedReturn values"
            )
        if not all(isinstance(item, ExecutionCost) for item in costs):
            raise EconomicEvaluationError(
                "execution_costs must contain ExecutionCost values"
            )
        return_keys: set[tuple[str, TimePoint, int]] = set()
        for return_item in returns:
            key = (
                return_item.target_id,
                return_item.decision_time,
                return_item.source_revision,
            )
            if key in return_keys:
                raise EconomicEvaluationError(
                    "economic evidence contains duplicate realized-return revisions"
                )
            return_keys.add(key)
        cost_keys: set[tuple[str, TimePoint, int]] = set()
        for cost_item in costs:
            key = (cost_item.target_id, cost_item.decision_time, cost_item.source_revision)
            if key in cost_keys:
                raise EconomicEvaluationError(
                    "economic evidence contains duplicate execution-cost revisions"
                )
            cost_keys.add(key)
        object.__setattr__(self, "realized_returns", returns)
        object.__setattr__(self, "execution_costs", costs)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "version": self.version,
            "source_id": self.source_id,
            "realized_returns": [item.to_dict() for item in self.realized_returns],
            "execution_costs": [item.to_dict() for item in self.execution_costs],
        }

    @property
    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(
                self._payload(),
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "digest": self.digest}

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> "EconomicEvidence":
        if not isinstance(state, Mapping):
            raise EconomicEvaluationError("economic evidence state must be a mapping")
        expected = {
            "schema",
            "version",
            "source_id",
            "realized_returns",
            "execution_costs",
            "digest",
        }
        if set(state) != expected:
            raise EconomicEvaluationError(
                "economic evidence contains unknown or missing fields"
            )
        if state.get("schema") != ECONOMIC_EVIDENCE_SCHEMA:
            raise EconomicEvaluationError("unsupported economic evidence schema")
        if type(state.get("version")) is not int or state.get("version") != ECONOMIC_EVIDENCE_VERSION:
            raise EconomicEvaluationError("unsupported economic evidence version")

        def rows(value: Any, name: str) -> tuple[Mapping[str, Any], ...]:
            if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, (list, tuple)):
                raise EconomicEvaluationError(f"economic evidence {name} must be a sequence")
            result: list[Mapping[str, Any]] = []
            for item in value:
                if not isinstance(item, Mapping):
                    raise EconomicEvaluationError(
                        f"economic evidence {name} entries must be mappings"
                    )
                result.append(item)
            return tuple(result)

        def row(value: Mapping[str, Any], expected_keys: set[str], name: str) -> Mapping[str, Any]:
            if set(value) != expected_keys:
                raise EconomicEvaluationError(
                    f"economic evidence {name} contains unknown or missing fields"
                )
            return value

        return_rows = rows(state["realized_returns"], "realized_returns")
        cost_rows = rows(state["execution_costs"], "execution_costs")
        returns: list[RealizedReturn] = []
        return_keys = {
            "target_id",
            "decision_time",
            "realization_time",
            "available_time",
            "value",
            "source_revision",
            "status",
        }
        for index, raw in enumerate(return_rows):
            value = row(raw, return_keys, f"realized_returns[{index}]")
            try:
                returns.append(
                    RealizedReturn(
                        target_id=value["target_id"],
                        decision_time=_decode_time(
                            value["decision_time"],
                            f"realized_returns[{index}].decision_time",
                        ),
                        realization_time=_decode_time(
                            value["realization_time"],
                            f"realized_returns[{index}].realization_time",
                        ),
                        available_time=_decode_time(
                            value["available_time"],
                            f"realized_returns[{index}].available_time",
                        ),
                        value=value["value"],
                        source_revision=value["source_revision"],
                        status=value["status"],
                    )
                )
            except (TypeError, ValueError) as exc:
                raise EconomicEvaluationError(
                    f"invalid economic evidence realized_returns[{index}]"
                ) from exc
        costs: list[ExecutionCost] = []
        cost_keys = {
            "target_id",
            "decision_time",
            "execution_time",
            "available_time",
            "cost_model_id",
            "fee_rate",
            "spread_rate",
            "slippage_rate",
            "impact_rate",
            "other_rate",
            "source_revision",
            "capacity_limit",
        }
        for index, raw in enumerate(cost_rows):
            value = row(raw, cost_keys, f"execution_costs[{index}]")
            try:
                costs.append(
                    ExecutionCost(
                        target_id=value["target_id"],
                        decision_time=_decode_time(
                            value["decision_time"],
                            f"execution_costs[{index}].decision_time",
                        ),
                        execution_time=_decode_time(
                            value["execution_time"],
                            f"execution_costs[{index}].execution_time",
                        ),
                        available_time=_decode_time(
                            value["available_time"],
                            f"execution_costs[{index}].available_time",
                        ),
                        cost_model_id=value["cost_model_id"],
                        fee_rate=value["fee_rate"],
                        spread_rate=value["spread_rate"],
                        slippage_rate=value["slippage_rate"],
                        impact_rate=value["impact_rate"],
                        other_rate=value["other_rate"],
                        source_revision=value["source_revision"],
                        capacity_limit=value["capacity_limit"],
                    )
                )
            except (TypeError, ValueError) as exc:
                raise EconomicEvaluationError(
                    f"invalid economic evidence execution_costs[{index}]"
                ) from exc
        try:
            candidate = cls(
                source_id=state["source_id"],
                realized_returns=tuple(returns),
                execution_costs=tuple(costs),
            )
        except (TypeError, ValueError) as exc:
            raise EconomicEvaluationError("invalid economic evidence artifact") from exc
        digest = state["digest"]
        if not isinstance(digest, str) or len(digest) != 64 or any(
            character not in "0123456789abcdefABCDEF" for character in digest
        ):
            raise EconomicEvaluationError("economic evidence digest must be hexadecimal")
        if digest.lower() != candidate.digest:
            raise EconomicEvaluationError("economic evidence digest does not match content")
        return candidate

    @classmethod
    def from_json(cls, payload: str) -> "EconomicEvidence":
        if not isinstance(payload, str) or not payload.strip():
            raise TypeError("economic evidence JSON must be a non-empty string")
        try:
            state = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise EconomicEvaluationError("economic evidence JSON is invalid") from exc
        return cls.from_dict(state)

    @staticmethod
    def _select_latest(
        records: tuple[RealizedReturn, ...] | tuple[ExecutionCost, ...],
        availability_time: TimePoint,
        name: str,
    ) -> dict[str, RealizedReturn] | dict[str, ExecutionCost]:
        selected: dict[str, RealizedReturn | ExecutionCost] = {}
        decision_times: dict[str, TimePoint] = {}
        for record in records:
            if not _le(record.available_time, availability_time, f"{name} availability"):
                continue
            previous_time = decision_times.get(record.target_id)
            if previous_time is not None and previous_time != record.decision_time:
                raise EconomicEvaluationError(
                    f"{name} has multiple decision times for target {record.target_id!r}"
                )
            decision_times[record.target_id] = record.decision_time
            previous = selected.get(record.target_id)
            if previous is None or record.source_revision > previous.source_revision:
                selected[record.target_id] = record
        return selected  # type: ignore[return-value]

    def select_at(
        self,
        availability_time: TimePoint,
    ) -> tuple[dict[str, RealizedReturn], dict[str, ExecutionCost]]:
        """Select the latest visible revision at one declared cutoff."""

        _time_key(availability_time)
        returns = self._select_latest(
            self.realized_returns, availability_time, "realized return"
        )
        costs = self._select_latest(
            self.execution_costs, availability_time, "execution cost"
        )
        return (
            {key: value for key, value in returns.items() if isinstance(value, RealizedReturn)},
            {key: value for key, value in costs.items() if isinstance(value, ExecutionCost)},
        )


@dataclass(frozen=True, slots=True)
class EconomicEvaluationConfig:
    """Fail-closed policy for one economic-path evaluation."""

    evaluation_id: str = "economic-evaluation"
    cost_model_id: str = "cost-model"
    initial_position: float = 0.0
    max_position: float = 1.0
    annualization: float = 252.0
    max_steps: int = MAX_ECONOMIC_STEPS
    require_capacity_evidence: bool = False

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
        if not isinstance(self.require_capacity_evidence, bool):
            raise EconomicEvaluationError("require_capacity_evidence must be boolean")

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
            "require_capacity_evidence": self.require_capacity_evidence,
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
    evidence_digest: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.config_identity, str) or not self.config_identity:
            raise EconomicEvaluationError("config_identity must be non-empty")
        object.__setattr__(self, "rows", tuple(self.rows))
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))
        if not isinstance(self.digest, str) or len(self.digest) != 64:
            raise EconomicEvaluationError("economic result digest must be SHA-256")
        if self.evidence_digest is not None:
            if (
                not isinstance(self.evidence_digest, str)
                or len(self.evidence_digest) != 64
                or any(
                    character not in "0123456789abcdefABCDEF"
                    for character in self.evidence_digest
                )
            ):
                raise EconomicEvaluationError(
                    "economic evidence digest must be a 64-character hexadecimal value"
                )
            object.__setattr__(self, "evidence_digest", self.evidence_digest.lower())

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
            "evidence_digest": self.evidence_digest,
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
    evidence_digest: str | None = None,
) -> EconomicEvaluationResult:
    """Evaluate an explicit position path using separately sourced evidence.

    Every evaluated decision must have a matching observed realized return and
    cost record. Abstentions are retained in the output but contribute neither
    return nor cost. The evaluator never substitutes a supervised target,
    zero return, or zero cost for missing evidence.
    """

    settings = config or EconomicEvaluationConfig()
    if evidence_digest is not None:
        if (
            not isinstance(evidence_digest, str)
            or len(evidence_digest) != 64
            or any(
                character not in "0123456789abcdefABCDEF"
                for character in evidence_digest
            )
        ):
            raise EconomicEvaluationError(
                "evidence_digest must be a 64-character hexadecimal value"
            )
        evidence_digest = evidence_digest.lower()
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
    capacity_evidence_count = 0
    capacity_utilization_max = 0.0

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
        if realized.status != "observed" or realized.value is None:
            raise EconomicEvaluationError(
                f"realized return is explicitly {realized.status} for evaluated target "
                f"{decision.target_id!r}"
            )
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
        if settings.require_capacity_evidence and cost.capacity_limit is None:
            raise EconomicEvaluationError(
                f"capacity evidence is missing for evaluated target {decision.target_id!r}"
            )
        if cost.capacity_limit is not None:
            capacity_evidence_count += 1
            utilization = turnover / cost.capacity_limit
            if not math.isfinite(utilization):
                raise EconomicEvaluationError("capacity utilization is not finite")
            capacity_utilization_max = max(capacity_utilization_max, utilization)
            if utilization > 1.0 + 1.0e-12:
                raise EconomicEvaluationError(
                    f"turnover exceeds supplied capacity for evaluated target "
                    f"{decision.target_id!r}"
                )
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
        "capacity_evidence_count": float(capacity_evidence_count),
        "capacity_utilization_max": capacity_utilization_max,
        "total_net_return": sum(net_values),
    }
    payload = {
        "schema": ECONOMIC_SCHEMA,
        "version": ECONOMIC_VERSION,
        "config_identity": settings.identity,
        "evidence_digest": evidence_digest,
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
    return EconomicEvaluationResult(
        settings.identity,
        tuple(rows),
        metrics,
        digest,
        evidence_digest,
    )


def evaluate_economic_evidence_path(
    decisions: Iterable[EconomicDecision],
    evidence: EconomicEvidence,
    availability_time: TimePoint,
    *,
    config: EconomicEvaluationConfig | None = None,
) -> EconomicEvaluationResult:
    """Evaluate a path from one digest-bound evidence cutoff."""

    if not isinstance(evidence, EconomicEvidence):
        raise EconomicEvaluationError("evidence must be an EconomicEvidence bundle")
    realized_returns, execution_costs = evidence.select_at(availability_time)
    return evaluate_economic_path(
        decisions,
        realized_returns,
        execution_costs,
        config=config,
        evidence_digest=evidence.digest,
    )


__all__ = [
    "ECONOMIC_EVIDENCE_SCHEMA",
    "ECONOMIC_EVIDENCE_VERSION",
    "ECONOMIC_SCHEMA",
    "ECONOMIC_VERSION",
    "EconomicDecision",
    "EconomicEvidence",
    "EconomicEvaluationConfig",
    "EconomicEvaluationError",
    "EconomicEvaluationResult",
    "EconomicPathRow",
    "ExecutionCost",
    "MAX_ECONOMIC_STEPS",
    "RealizedReturn",
    "evaluate_economic_evidence_path",
    "evaluate_economic_path",
]
