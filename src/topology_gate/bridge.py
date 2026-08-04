"""Finite martingale Schrödinger-bridge-style stress projection.

This module solves a small discrete projection problem.  Starting from
reference path weights, it alternates two entropic projections:

1. match a declared terminal-state marginal;
2. within each initial-state group, match a declared conditional terminal
   drift (zero for a martingale, or a supplied physical-measure drift).

The result is the closest feasible finite scenario law in relative entropy
within the numerical tolerance.  It is a stress-training primitive, not a
continuous-time martingale Schrödinger bridge, and it does not infer a pricing
measure from physical returns.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Mapping, Sequence

BRIDGE_SCHEMA = "topology_gate.finite_martingale_stress_bridge"
BRIDGE_VERSION = 1
MAX_BRIDGE_PATHS = 2_048


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


@dataclass(frozen=True, slots=True)
class StressPath:
    """One finite reference path endpoint pair."""

    path_id: str
    initial_state: float
    terminal_state: float
    reference_weight: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.path_id, str) or not self.path_id.strip():
            raise ValueError("path_id must be a non-empty string")
        _finite("initial_state", self.initial_state)
        _finite("terminal_state", self.terminal_state)
        _finite("reference_weight", self.reference_weight, minimum=0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path_id": self.path_id,
            "initial_state": self.initial_state,
            "terminal_state": self.terminal_state,
            "reference_weight": self.reference_weight,
        }


@dataclass(frozen=True, slots=True)
class StressBridgeConfig:
    """Numerical policy for finite alternating entropic projections."""

    max_iterations: int = 256
    tolerance: float = 1.0e-8
    bisection_iterations: int = 80
    gamma_bound: float = 64.0

    def __post_init__(self) -> None:
        _integer("max_iterations", self.max_iterations, minimum=1, maximum=10_000)
        _finite("tolerance", self.tolerance, minimum=1.0e-15)
        _integer("bisection_iterations", self.bisection_iterations, minimum=8, maximum=512)
        _finite("gamma_bound", self.gamma_bound, minimum=1.0e-12)

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_iterations": self.max_iterations,
            "tolerance": self.tolerance,
            "bisection_iterations": self.bisection_iterations,
            "gamma_bound": self.gamma_bound,
        }


@dataclass(frozen=True, slots=True)
class StressBridgeResult:
    """Digestable finite stress law and constraint diagnostics."""

    weights: tuple[tuple[str, float], ...]
    terminal_masses: tuple[tuple[float, float, float], ...]
    drift_residuals: tuple[tuple[float, float], ...]
    entropy: float
    iterations: int
    converged: bool
    digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": BRIDGE_SCHEMA,
            "version": BRIDGE_VERSION,
            "weights": [[path_id, weight] for path_id, weight in self.weights],
            "terminal_masses": [list(row) for row in self.terminal_masses],
            "drift_residuals": [list(row) for row in self.drift_residuals],
            "entropy": self.entropy,
            "iterations": self.iterations,
            "converged": self.converged,
            "digest": self.digest,
        }


class MartingaleStressBridge:
    """Project reference endpoint scenarios onto finite stress constraints."""

    def __init__(self, config: StressBridgeConfig | None = None) -> None:
        self.config = StressBridgeConfig() if config is None else config
        if not isinstance(self.config, StressBridgeConfig):
            raise TypeError("config must be StressBridgeConfig")

    def _paths(self, paths: Sequence[StressPath]) -> tuple[StressPath, ...]:
        try:
            normalized = tuple(paths)
        except TypeError as exc:
            raise TypeError("paths must be a sequence") from exc
        if not normalized or len(normalized) > MAX_BRIDGE_PATHS:
            raise ValueError(f"paths must contain 1..{MAX_BRIDGE_PATHS} items")
        if any(not isinstance(path, StressPath) for path in normalized):
            raise ValueError("paths must contain StressPath values")
        if len({path.path_id for path in normalized}) != len(normalized):
            raise ValueError("path_id values must be unique")
        total = sum(path.reference_weight for path in normalized)
        if total <= 0.0:
            raise ValueError("reference weights must have positive total")
        return normalized

    def _terminal_targets(
        self,
        values: Sequence[tuple[float, float]],
    ) -> dict[float, float]:
        try:
            pairs = tuple(values)
        except TypeError as exc:
            raise TypeError("terminal_masses must be a sequence") from exc
        if not pairs:
            raise ValueError("terminal_masses must not be empty")
        targets: dict[float, float] = {}
        for index, pair in enumerate(pairs):
            if len(pair) != 2:
                raise ValueError("each terminal mass must contain state and mass")
            state = _finite(f"terminal_masses[{index}].state", pair[0])
            mass = _finite(f"terminal_masses[{index}].mass", pair[1], minimum=0.0)
            if state in targets:
                raise ValueError("terminal states must be unique")
            targets[state] = mass
        total = sum(targets.values())
        if total <= 0.0 or abs(total - 1.0) > 1.0e-8:
            raise ValueError("terminal masses must sum to one")
        return {state: mass / total for state, mass in targets.items()}

    def _drift_targets(
        self,
        values: Sequence[tuple[float, float]] | None,
    ) -> dict[float, float]:
        if values is None:
            return {}
        targets: dict[float, float] = {}
        for index, pair in enumerate(values):
            if len(pair) != 2:
                raise ValueError("each drift target must contain state and drift")
            state = _finite(f"drift_targets[{index}].state", pair[0])
            drift = _finite(f"drift_targets[{index}].drift", pair[1])
            if state in targets:
                raise ValueError("initial states must be unique")
            targets[state] = drift
        return targets

    def _group_indices(
        self,
        paths: tuple[StressPath, ...],
    ) -> tuple[dict[float, list[int]], dict[float, list[int]]]:
        by_terminal: dict[float, list[int]] = {}
        by_initial: dict[float, list[int]] = {}
        for index, path in enumerate(paths):
            by_terminal.setdefault(path.terminal_state, []).append(index)
            by_initial.setdefault(path.initial_state, []).append(index)
        return by_terminal, by_initial

    def _conditional_drift(
        self,
        weights: Sequence[float],
        paths: Sequence[StressPath],
        indices: Sequence[int],
    ) -> float:
        total = sum(weights[index] for index in indices)
        if total <= 0.0:
            raise ValueError("stress projection produced an empty initial-state group")
        return sum(
            weights[index] * (paths[index].terminal_state - paths[index].initial_state)
            for index in indices
        ) / total

    def _tilt_to_drift(
        self,
        weights: list[float],
        paths: Sequence[StressPath],
        indices: Sequence[int],
        target: float,
    ) -> None:
        deltas = [paths[index].terminal_state - paths[index].initial_state for index in indices]
        lower = min(deltas)
        upper = max(deltas)
        if target < lower - self.config.tolerance or target > upper + self.config.tolerance:
            raise ValueError("requested drift is infeasible for an initial-state group")
        group_total = sum(weights[index] for index in indices)
        if group_total <= 0.0:
            raise ValueError("infeasible stress constraints: empty initial-state group")

        def mean(gamma: float) -> float:
            tilted = [weights[index] * math.exp(gamma * delta) for index, delta in zip(indices, deltas)]
            denominator = sum(tilted)
            return sum(value * delta for value, delta in zip(tilted, deltas)) / denominator

        left = -self.config.gamma_bound
        right = self.config.gamma_bound
        if mean(left) > target + self.config.tolerance or mean(right) < target - self.config.tolerance:
            raise ValueError("gamma_bound cannot reach requested drift")
        for _ in range(self.config.bisection_iterations):
            midpoint = 0.5 * (left + right)
            if mean(midpoint) < target:
                left = midpoint
            else:
                right = midpoint
        gamma = 0.5 * (left + right)
        tilted = [weights[index] * math.exp(gamma * delta) for index, delta in zip(indices, deltas)]
        tilted_total = sum(tilted)
        for index, value in zip(indices, tilted):
            weights[index] = group_total * value / tilted_total

    def fit(
        self,
        paths: Sequence[StressPath],
        terminal_masses: Sequence[tuple[float, float]],
        *,
        drift_targets: Sequence[tuple[float, float]] | None = None,
    ) -> StressBridgeResult:
        normalized_paths = self._paths(paths)
        targets = self._terminal_targets(terminal_masses)
        drifts = self._drift_targets(drift_targets)
        by_terminal, by_initial = self._group_indices(normalized_paths)
        if set(by_terminal) != set(targets):
            raise ValueError("terminal target states must match path terminal states")
        if drifts and set(drifts) != set(by_initial):
            raise ValueError("drift target states must match path initial states")
        normalized_reference = sum(path.reference_weight for path in normalized_paths)
        weights = [path.reference_weight / normalized_reference for path in normalized_paths]
        target_initial_drift = {
            initial: drifts.get(initial, 0.0) for initial in by_initial
        }
        converged = False
        iterations = 0
        for iterations in range(1, self.config.max_iterations + 1):
            for terminal_state, indices in by_terminal.items():
                current = sum(weights[index] for index in indices)
                if current <= 0.0 and targets[terminal_state] > 0.0:
                    raise ValueError("stress projection produced an empty terminal-state group")
                scale = 0.0 if current == 0.0 else targets[terminal_state] / current
                for index in indices:
                    weights[index] *= scale
            for initial_state, indices in by_initial.items():
                self._tilt_to_drift(
                    weights,
                    normalized_paths,
                    indices,
                    target_initial_drift[initial_state],
                )
            terminal_error = max(
                abs(sum(weights[index] for index in indices) - targets[state])
                for state, indices in by_terminal.items()
            )
            drift_error = max(
                abs(self._conditional_drift(weights, normalized_paths, indices) - target_initial_drift[state])
                for state, indices in by_initial.items()
            )
            if max(terminal_error, drift_error) <= self.config.tolerance:
                converged = True
                break
        terminal_rows = tuple(
            (
                state,
                sum(weights[index] for index in indices),
                targets[state],
            )
            for state, indices in sorted(by_terminal.items())
        )
        drift_rows = tuple(
            (
                state,
                self._conditional_drift(weights, normalized_paths, indices)
                - target_initial_drift[state],
            )
            for state, indices in sorted(by_initial.items())
        )
        if not converged:
            raise ValueError(
                "infeasible stress constraints or no convergence within max_iterations"
            )
        entropy = sum(
            weight * math.log(weight / (path.reference_weight / normalized_reference))
            for weight, path in zip(weights, normalized_paths)
            if weight > 0.0 and path.reference_weight > 0.0
        )
        payload: dict[str, Any] = {
            "schema": BRIDGE_SCHEMA,
            "version": BRIDGE_VERSION,
            "config": self.config.to_dict(),
            "weights": [[path.path_id, weight] for path, weight in zip(normalized_paths, weights)],
            "terminal_masses": [list(row) for row in terminal_rows],
            "drift_residuals": [list(row) for row in drift_rows],
            "entropy": entropy,
            "iterations": iterations,
            "converged": converged,
        }
        return StressBridgeResult(
            weights=tuple((path.path_id, weight) for path, weight in zip(normalized_paths, weights)),
            terminal_masses=terminal_rows,
            drift_residuals=drift_rows,
            entropy=entropy,
            iterations=iterations,
            converged=converged,
            digest=_digest(payload),
        )
