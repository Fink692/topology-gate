"""Adaptive truncated path-signature memory.

The path signature of a piecewise-linear increment stream is a structured,
ordered summary of history.  This module computes truncated signatures without
external dependencies, runs one recursive ridge learner per predeclared depth,
and chooses the next depth using only losses settled at the previous boundary.

It is a finite rough-path memory prototype.  Signature depth, window length,
loss clipping, and switching cost are study choices; the component does not
claim that a selected signature is a causal market mechanism.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import deque
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Mapping, Sequence

from .rls import RLS

SIGNATURE_SCHEMA = "topology_gate.adaptive_signature_memory"
SIGNATURE_VERSION = 1
MAX_SIGNATURE_DEPTH = 5
MAX_SIGNATURE_FEATURES = 2_048
MAX_SIGNATURE_HISTORY = 4_096


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


def _integer(name: str, value: Any, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum or result > maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
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


def signature_dimension(input_dim: int, depth: int) -> int:
    """Return the flattened dimension including the level-zero coordinate."""

    dimension = _integer("input_dim", input_dim, minimum=1, maximum=64)
    maximum_depth = _integer(
        "depth", depth, minimum=0, maximum=MAX_SIGNATURE_DEPTH
    )
    return sum(dimension**level for level in range(maximum_depth + 1))


def _validate_increments(
    increments: Sequence[Sequence[float]], input_dim: int
) -> list[list[float]]:
    try:
        rows = list(increments)
    except TypeError as exc:
        raise TypeError("increments must be a sequence of vectors") from exc
    validated: list[list[float]] = []
    for row_index, row in enumerate(rows):
        try:
            values = list(row)
        except TypeError as exc:
            raise TypeError(f"increments[{row_index}] must be a vector") from exc
        if len(values) != input_dim:
            raise ValueError(
                f"increments[{row_index}] must contain exactly {input_dim} values"
            )
        validated.append(
            [_finite(f"increments[{row_index}][{column}]", value) for column, value in enumerate(values)]
        )
    return validated


def _tensor_product(left: Sequence[float], right: Sequence[float]) -> list[float]:
    return [left_value * right_value for left_value in left for right_value in right]


def path_signature(
    increments: Sequence[Sequence[float]],
    *,
    input_dim: int,
    depth: int,
) -> tuple[float, ...]:
    """Compute the truncated Chen signature of a piecewise-linear path."""

    dimension = _integer("input_dim", input_dim, minimum=1, maximum=64)
    maximum_depth = _integer(
        "depth", depth, minimum=0, maximum=MAX_SIGNATURE_DEPTH
    )
    if signature_dimension(dimension, maximum_depth) > MAX_SIGNATURE_FEATURES:
        raise ValueError("signature dimension exceeds the configured bound")
    rows = _validate_increments(increments, dimension)
    levels: list[list[float]] = [[1.0]] + [
        [0.0] * dimension**level for level in range(1, maximum_depth + 1)
    ]
    for increment in rows:
        segment: list[list[float]] = [[1.0]]
        power = [1.0]
        for level in range(1, maximum_depth + 1):
            power = _tensor_product(power, increment)
            factorial = math.factorial(level)
            segment.append([value / factorial for value in power])
        concatenated: list[list[float]] = []
        for level in range(maximum_depth + 1):
            result = [0.0] * dimension**level
            for left_level in range(level + 1):
                right_level = level - left_level
                product = _tensor_product(levels[left_level], segment[right_level])
                result = [old + new for old, new in zip(result, product)]
            concatenated.append(result)
        levels = concatenated
    flattened = tuple(value for level in levels for value in level)
    if any(not math.isfinite(value) for value in flattened):
        raise FloatingPointError("signature contains a non-finite value")
    return flattened


@dataclass(frozen=True, slots=True)
class SignatureMemoryConfig:
    """Predeclared candidate depths and causal selection policy."""

    input_dim: int
    candidate_depths: tuple[int, ...] = (1, 2, 3)
    window: int = 32
    ridge: float = 1.0
    forgetting_factor: float = 0.99
    switching_cost: float = 0.0
    loss_clip: float = 100.0
    history_size: int = 256

    def __post_init__(self) -> None:
        _integer("input_dim", self.input_dim, minimum=1, maximum=16)
        try:
            depths = tuple(self.candidate_depths)
        except TypeError as exc:
            raise ValueError("candidate_depths must be a sequence") from exc
        if not depths:
            raise ValueError("candidate_depths must not be empty")
        if any(
            isinstance(depth, bool)
            or not isinstance(depth, Integral)
            or not 1 <= int(depth) <= MAX_SIGNATURE_DEPTH
            for depth in depths
        ):
            raise ValueError("candidate_depths must contain valid positive depths")
        if len(set(int(depth) for depth in depths)) != len(depths):
            raise ValueError("candidate_depths must be unique")
        normalized = tuple(sorted(int(depth) for depth in depths))
        if any(signature_dimension(self.input_dim, depth) > MAX_SIGNATURE_FEATURES for depth in normalized):
            raise ValueError("candidate signature dimension exceeds the configured bound")
        object.__setattr__(self, "candidate_depths", normalized)
        _integer("window", self.window, minimum=1, maximum=MAX_SIGNATURE_HISTORY)
        _finite("ridge", self.ridge, minimum=1.0e-15)
        factor = _finite("forgetting_factor", self.forgetting_factor)
        if not 0.0 < factor <= 1.0:
            raise ValueError("forgetting_factor must be in (0, 1]")
        _finite("switching_cost", self.switching_cost, minimum=0.0)
        _finite("loss_clip", self.loss_clip, minimum=1.0e-15)
        _integer("history_size", self.history_size, minimum=1, maximum=MAX_SIGNATURE_HISTORY)

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "candidate_depths": list(self.candidate_depths),
            "window": self.window,
            "ridge": self.ridge,
            "forgetting_factor": self.forgetting_factor,
            "switching_cost": self.switching_cost,
            "loss_clip": self.loss_clip,
            "history_size": self.history_size,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SignatureMemoryConfig":
        expected = {
            "input_dim",
            "candidate_depths",
            "window",
            "ridge",
            "forgetting_factor",
            "switching_cost",
            "loss_clip",
            "history_size",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("signature config has unknown or missing fields")
        return cls(
            input_dim=value["input_dim"],
            candidate_depths=tuple(value["candidate_depths"]),
            window=value["window"],
            ridge=value["ridge"],
            forgetting_factor=value["forgetting_factor"],
            switching_cost=value["switching_cost"],
            loss_clip=value["loss_clip"],
            history_size=value["history_size"],
        )


@dataclass(frozen=True, slots=True)
class SignaturePrediction:
    """Prediction and all shadow candidates at one decision boundary."""

    selected_depth: int
    prediction: float
    candidate_predictions: tuple[tuple[int, float], ...]


@dataclass(frozen=True, slots=True)
class SignatureUpdate:
    """One settled target transition and the next selected memory depth."""

    step: int
    selected_depth: int
    next_depth: int
    prediction: float
    target: float
    loss: float
    candidate_losses: tuple[tuple[int, float], ...]


class AdaptiveSignatureMemory:
    """Full-information recursive learners with adaptive signature depth."""

    def __init__(self, config: SignatureMemoryConfig) -> None:
        if not isinstance(config, SignatureMemoryConfig):
            raise TypeError("config must be SignatureMemoryConfig")
        self.config = config
        self._learners = {
            depth: RLS(
                signature_dimension(config.input_dim, depth),
                ridge=config.ridge,
                forgetting_factor=config.forgetting_factor,
                lambda_min=config.forgetting_factor,
                lambda_max=1.0,
            )
            for depth in config.candidate_depths
        }
        self._losses: dict[int, deque[float]] = {
            depth: deque(maxlen=config.history_size)
            for depth in config.candidate_depths
        }
        self._selected_depth = config.candidate_depths[0]
        self._step = 0

    @property
    def step(self) -> int:
        return self._step

    @property
    def selected_depth(self) -> int:
        return self._selected_depth

    @property
    def candidate_depths(self) -> tuple[int, ...]:
        return self.config.candidate_depths

    def _path(self, increments: Sequence[Sequence[float]]) -> list[list[float]]:
        rows = _validate_increments(increments, self.config.input_dim)
        return rows[-self.config.window :]

    def _signatures(self, increments: Sequence[Sequence[float]]) -> dict[int, tuple[float, ...]]:
        rows = self._path(increments)
        return {
            depth: path_signature(rows, input_dim=self.config.input_dim, depth=depth)
            for depth in self.config.candidate_depths
        }

    def _score(self, depth: int) -> float:
        history = self._losses[depth]
        if not history:
            return 0.0 if depth == self._selected_depth else self.config.switching_cost
        penalty = 0.0 if depth == self._selected_depth else self.config.switching_cost
        return sum(history) / len(history) + penalty

    def _choose_next_depth(self) -> int:
        return min(
            self.config.candidate_depths,
            key=lambda depth: (self._score(depth), self.config.candidate_depths.index(depth)),
        )

    def predict(self, increments: Sequence[Sequence[float]]) -> SignaturePrediction:
        signatures = self._signatures(increments)
        predictions: list[tuple[int, float]] = []
        for depth in self.config.candidate_depths:
            value = self._learners[depth].predict(signatures[depth])
            if isinstance(value, list):
                raise RuntimeError("signature learner must be scalar-output")
            predictions.append((depth, float(value)))
        selected_prediction = dict(predictions)[self._selected_depth]
        return SignaturePrediction(
            selected_depth=self._selected_depth,
            prediction=selected_prediction,
            candidate_predictions=tuple(predictions),
        )

    def observe(
        self,
        increments: Sequence[Sequence[float]],
        target: Any,
    ) -> SignatureUpdate:
        target_value = _finite("target", target)
        signatures = self._signatures(increments)
        before = self.predict(increments)
        candidate_losses: list[tuple[int, float]] = []
        for depth, prediction in before.candidate_predictions:
            raw_loss = (target_value - prediction) ** 2
            candidate_losses.append((depth, min(self.config.loss_clip, raw_loss)))
        for depth in self.config.candidate_depths:
            self._learners[depth].update(
                signatures[depth],
                target_value,
                forgetting_factor=self.config.forgetting_factor,
            )
            self._losses[depth].append(dict(candidate_losses)[depth])
        loss = dict(candidate_losses)[before.selected_depth]
        next_depth = self._choose_next_depth()
        self._selected_depth = next_depth
        self._step += 1
        return SignatureUpdate(
            step=self._step,
            selected_depth=before.selected_depth,
            next_depth=next_depth,
            prediction=before.prediction,
            target=target_value,
            loss=loss,
            candidate_losses=tuple(candidate_losses),
        )

    def state_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": SIGNATURE_SCHEMA,
            "version": SIGNATURE_VERSION,
            "config": self.config.to_dict(),
            "step": self._step,
            "selected_depth": self._selected_depth,
            "learners": {str(depth): self._learners[depth].state_dict() for depth in self.candidate_depths},
            "losses": {str(depth): list(self._losses[depth]) for depth in self.candidate_depths},
        }
        payload["digest"] = _digest(payload)
        return payload

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "AdaptiveSignatureMemory":
        if not isinstance(state, Mapping):
            raise TypeError("state must be a mapping")
        expected = {"schema", "version", "config", "step", "selected_depth", "learners", "losses", "digest"}
        if set(state) != expected:
            raise ValueError("signature state has unknown or missing fields")
        unsigned = {key: state[key] for key in expected if key != "digest"}
        if state["digest"] != _digest(unsigned):
            raise ValueError("signature state digest mismatch")
        if state["schema"] != SIGNATURE_SCHEMA or state["version"] != SIGNATURE_VERSION:
            raise ValueError("unsupported signature state")
        config = SignatureMemoryConfig.from_dict(state["config"])
        step = _integer("state.step", state["step"], minimum=0, maximum=10**12)
        selected_depth = _integer("state.selected_depth", state["selected_depth"], minimum=1, maximum=MAX_SIGNATURE_DEPTH)
        if selected_depth not in config.candidate_depths:
            raise ValueError("selected depth is not configured")
        learners = state["learners"]
        losses = state["losses"]
        if not isinstance(learners, Mapping) or not isinstance(losses, Mapping):
            raise ValueError("learners and losses must be mappings")
        expected_keys = {str(depth) for depth in config.candidate_depths}
        if set(learners) != expected_keys or set(losses) != expected_keys:
            raise ValueError("signature candidate identities do not match config")
        result = cls(config)
        result._step = step
        result._selected_depth = selected_depth
        result._learners = {
            depth: RLS.from_state_dict(learners[str(depth)])
            for depth in config.candidate_depths
        }
        for depth in config.candidate_depths:
            values = losses[str(depth)]
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
                raise ValueError("signature losses must be sequences")
            if len(values) > config.history_size:
                raise ValueError("signature loss history exceeds configured bound")
            result._losses[depth].extend(
                _finite(f"losses[{depth}]", value, minimum=0.0) for value in values
            )
        return result

    def digest(self) -> str:
        return self.state_dict()["digest"]

