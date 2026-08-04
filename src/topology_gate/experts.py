"""Heavy-tail-aware full-information expert allocation primitives."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Mapping, Sequence, cast

HEAVY_TAIL_EXPERT_SCHEMA = "topology_gate.heavy_tail_expert_allocator"
HEAVY_TAIL_EXPERT_VERSION = 1
MAX_EXPERTS = 256
MAX_EXPERT_HISTORY = 4_096


def _integer(name: str, value: Any, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum or result > maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return result


def _finite(name: str, value: Any, *, minimum: float | None = None) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _catoni_psi(value: float) -> float:
    """Use Catoni's odd influence function without quadratic overflow."""

    magnitude = abs(value)
    if magnitude <= 1.0:
        return math.log1p(value + 0.5 * value * value) - math.log1p(
            -value + 0.5 * value * value
        )
    inverse = 1.0 / magnitude
    inverse_square = inverse * inverse
    numerator = 2.0 * value * inverse_square + 2.0 * inverse_square
    denominator = -2.0 * value * inverse_square + 2.0 * inverse_square
    return math.log1p(numerator) - math.log1p(denominator)


def catoni_mean(
    values: Sequence[float],
    *,
    scale: float,
    iterations: int = 80,
) -> float:
    """Estimate a location by solving Catoni's bounded-influence equation.

    ``scale`` is a predeclared robustness scale. The estimator is intentionally
    a location estimate only; it is not a confidence bound and does not make a
    downstream trading decision anytime-valid by itself.
    """

    scale_value = _finite("scale", scale, minimum=1.0e-15)
    if not values:
        raise ValueError("values must not be empty")
    iteration_count = _integer("iterations", iterations, 8, 512)
    observations = tuple(
        _finite("value", value)
        for value in values
    )
    lower = min(observations)
    upper = max(observations)
    for _ in range(iteration_count):
        midpoint = lower + (upper - lower) * 0.5
        score = sum(
            _catoni_psi((value - midpoint) / scale_value)
            for value in observations
        )
        if score > 0.0:
            lower = midpoint
        else:
            upper = midpoint
    result = lower + (upper - lower) * 0.5
    if not math.isfinite(result):
        raise FloatingPointError("Catoni estimate is not finite")
    return result


@dataclass(frozen=True, slots=True)
class HeavyTailExpertConfig:
    """Immutable allocation policy for full-information expert streams."""

    expert_ids: tuple[str, ...]
    catoni_scale: float = 1.0
    switching_cost: float = 0.0
    max_history: int = MAX_EXPERT_HISTORY
    reset_on_change: bool = True

    def __post_init__(self) -> None:
        try:
            identifiers = tuple(self.expert_ids)
        except TypeError as exc:
            raise ValueError("expert_ids must be a sequence") from exc
        if not identifiers or len(identifiers) > MAX_EXPERTS:
            raise ValueError(f"expert_ids must contain 1..{MAX_EXPERTS} items")
        if any(not isinstance(item, str) or not item.strip() for item in identifiers):
            raise ValueError("expert_ids must contain non-empty strings")
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("expert_ids must be unique")
        scale = _finite("catoni_scale", self.catoni_scale, minimum=1.0e-15)
        switching_cost = _finite(
            "switching_cost", self.switching_cost, minimum=0.0
        )
        history = _integer("max_history", self.max_history, 1, MAX_EXPERT_HISTORY)
        if not isinstance(self.reset_on_change, bool):
            raise ValueError("reset_on_change must be boolean")
        object.__setattr__(self, "expert_ids", identifiers)
        object.__setattr__(self, "catoni_scale", scale)
        object.__setattr__(self, "switching_cost", switching_cost)
        object.__setattr__(self, "max_history", history)

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": HEAVY_TAIL_EXPERT_SCHEMA,
            "version": HEAVY_TAIL_EXPERT_VERSION,
            "expert_ids": list(self.expert_ids),
            "catoni_scale": self.catoni_scale,
            "switching_cost": self.switching_cost,
            "max_history": self.max_history,
            "reset_on_change": self.reset_on_change,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "HeavyTailExpertConfig":
        if not isinstance(state, Mapping):
            raise ValueError("expert config must be a mapping")
        expected = {
            "schema",
            "version",
            "expert_ids",
            "catoni_scale",
            "switching_cost",
            "max_history",
            "reset_on_change",
        }
        if set(state) != expected:
            raise ValueError("expert config contains unknown or missing fields")
        if (
            state.get("schema") != HEAVY_TAIL_EXPERT_SCHEMA
            or state.get("version") != HEAVY_TAIL_EXPERT_VERSION
        ):
            raise ValueError("unsupported expert config")
        return cls(
            expert_ids=tuple(cast(Sequence[str], state["expert_ids"])),
            catoni_scale=cast(float, state["catoni_scale"]),
            switching_cost=cast(float, state["switching_cost"]),
            max_history=cast(int, state["max_history"]),
            reset_on_change=cast(bool, state["reset_on_change"]),
        )

    @property
    def identity(self) -> str:
        return _digest(self.state_dict())


@dataclass(frozen=True, slots=True)
class ExpertDecision:
    """Allocation selected for the next decision boundary."""

    step: int
    selected_index: int
    selected_expert: str
    robust_estimates: tuple[float, ...]
    switching_adjusted_scores: tuple[float, ...]
    switched: bool
    reset_applied: bool


class HeavyTailExpertAllocator:
    """Allocate among shadow-return experts using robust location estimates.

    Utilities are full-information observations: every expert's score must be
    available at the same settlement boundary. The selected expert applies to
    the next boundary, so the current observation cannot affect its own action.
    """

    method = "catoni-full-information-experts"

    def __init__(self, config: HeavyTailExpertConfig) -> None:
        if not isinstance(config, HeavyTailExpertConfig):
            raise TypeError("config must be a HeavyTailExpertConfig")
        self.config = config
        self.reset()

    @property
    def config_identity(self) -> str:
        return self.config.identity

    @property
    def step(self) -> int:
        return self._step

    @property
    def current_index(self) -> int | None:
        return self._current_index

    @property
    def current_expert(self) -> str | None:
        if self._current_index is None:
            return None
        return self.config.expert_ids[self._current_index]

    @property
    def histories(self) -> tuple[tuple[float, ...], ...]:
        return tuple(tuple(history) for history in self._histories)

    def reset(self) -> None:
        self._histories: list[list[float]] = [
            [] for _ in self.config.expert_ids
        ]
        self._current_index: int | None = None
        self._step = 0

    def _validate_utilities(self, utilities: Sequence[float]) -> tuple[float, ...]:
        try:
            values = tuple(utilities)
        except TypeError as exc:
            raise ValueError("utilities must be a sequence") from exc
        if len(values) != len(self.config.expert_ids):
            raise ValueError("utilities must match the configured expert count")
        return tuple(_finite("utility", value) for value in values)

    def observe(
        self,
        utilities: Sequence[float],
        *,
        change_point: bool = False,
    ) -> ExpertDecision:
        """Settle one full-information row and choose the next expert."""

        if not isinstance(change_point, bool):
            raise ValueError("change_point must be boolean")
        values = self._validate_utilities(utilities)
        reset_applied = change_point and self.config.reset_on_change
        if reset_applied:
            self._histories = [[] for _ in self.config.expert_ids]
        for index, value in enumerate(values):
            history = self._histories[index]
            history.append(value)
            if len(history) > self.config.max_history:
                del history[: len(history) - self.config.max_history]
        estimates = tuple(
            catoni_mean(history, scale=self.config.catoni_scale)
            for history in self._histories
        )
        adjusted = tuple(
            estimate
            - (
                self.config.switching_cost
                if self._current_index is not None and index != self._current_index
                else 0.0
            )
            for index, estimate in enumerate(estimates)
        )
        selected = max(range(len(adjusted)), key=lambda index: adjusted[index])
        switched = (
            self._current_index is not None and selected != self._current_index
        )
        self._current_index = selected
        self._step += 1
        return ExpertDecision(
            step=self._step,
            selected_index=selected,
            selected_expert=self.config.expert_ids[selected],
            robust_estimates=estimates,
            switching_adjusted_scores=adjusted,
            switched=switched,
            reset_applied=reset_applied,
        )

    def state_dict(self) -> dict[str, Any]:
        state = {
            "schema": HEAVY_TAIL_EXPERT_SCHEMA,
            "version": HEAVY_TAIL_EXPERT_VERSION,
            "config_identity": self.config_identity,
            "step": self._step,
            "current_index": self._current_index,
            "histories": [list(history) for history in self._histories],
        }
        return {**state, "state_identity": _digest(state)}

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, Any],
        config: HeavyTailExpertConfig,
    ) -> "HeavyTailExpertAllocator":
        if not isinstance(state, Mapping):
            raise ValueError("expert allocator state must be a mapping")
        expected = {
            "schema",
            "version",
            "config_identity",
            "step",
            "current_index",
            "histories",
            "state_identity",
        }
        if set(state) != expected:
            raise ValueError("expert allocator state contains unknown or missing fields")
        payload = {key: state[key] for key in expected if key != "state_identity"}
        if state.get("state_identity") != _digest(payload):
            raise ValueError("expert allocator state identity mismatch")
        if (
            state.get("schema") != HEAVY_TAIL_EXPERT_SCHEMA
            or state.get("version") != HEAVY_TAIL_EXPERT_VERSION
        ):
            raise ValueError("unsupported expert allocator state")
        if state.get("config_identity") != config.identity:
            raise ValueError("expert allocator config identity mismatch")
        step = _integer("step", state.get("step"), 0, MAX_EXPERT_HISTORY)
        raw_current = state.get("current_index")
        if raw_current is not None:
            current = _integer(
                "current_index", raw_current, 0, len(config.expert_ids) - 1
            )
        else:
            current = None
        raw_histories = state.get("histories")
        if not isinstance(raw_histories, Sequence) or isinstance(
            raw_histories, (str, bytes, bytearray)
        ):
            raise ValueError("histories must be a sequence")
        if len(raw_histories) != len(config.expert_ids):
            raise ValueError("history count does not match the configured experts")
        histories: list[list[float]] = []
        for raw_history in raw_histories:
            if not isinstance(raw_history, Sequence) or isinstance(
                raw_history, (str, bytes, bytearray)
            ):
                raise ValueError("each expert history must be a sequence")
            if len(raw_history) > config.max_history:
                raise ValueError("expert history exceeds its resource limit")
            histories.append([_finite("history value", value) for value in raw_history])
        candidate = cls(config)
        candidate._step = step
        candidate._current_index = current
        candidate._histories = histories
        return candidate

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        candidate = type(self).from_state_dict(state, self.config)
        self._step = candidate._step
        self._current_index = candidate._current_index
        self._histories = candidate._histories


__all__ = [
    "HEAVY_TAIL_EXPERT_SCHEMA",
    "HEAVY_TAIL_EXPERT_VERSION",
    "MAX_EXPERTS",
    "MAX_EXPERT_HISTORY",
    "ExpertDecision",
    "HeavyTailExpertAllocator",
    "HeavyTailExpertConfig",
    "catoni_mean",
]
