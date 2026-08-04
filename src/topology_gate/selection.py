"""Pre-registered selection-family alpha allocation.

An e-process only pays for choices that are included in its predeclared
comparison family.  This module makes the common model/feature/eta selection
boundary explicit: a finite Cartesian family receives equal parent-alpha
shares, and the selected cell is bound into the evidence identity.

The resulting ``allocated_alpha`` is the alpha available to the downstream
challenger gate.  The gate may spend that share again across challenger slots
and epochs; it may not silently borrow alpha from unregistered selection
cells.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, cast

SELECTION_BUDGET_SCHEMA = "topology_gate.selection_budget"
SELECTION_BUDGET_VERSION = 1
MAX_SELECTION_SLOTS = 1_000_000


def _text(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    if len(value) > 512:
        raise ValueError(f"{name} exceeds 512 characters")
    return value


def _alpha(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("global_alpha must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("global_alpha must be finite") from exc
    if not math.isfinite(result) or not 0.0 < result < 1.0:
        raise ValueError("global_alpha must be in (0, 1)")
    return result


def _slot(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a positive integer")
    if not 1 <= value <= MAX_SELECTION_SLOTS:
        raise ValueError(f"{name} is outside the configured limit")
    return value


def _digest(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class SelectionBudget:
    """A finite pre-registered model/feature/eta selection family.

    ``global_alpha`` is the parent family budget.  The selected Cartesian
    cell receives ``global_alpha / (model_slots * feature_slots * eta_slots)``.
    All indices are one-based and must be fixed before evidence is observed.
    """

    budget_id: str
    global_alpha: float
    model_slots: int = 1
    feature_slots: int = 1
    eta_slots: int = 1
    model_index: int = 1
    feature_index: int = 1
    eta_index: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "budget_id", _text("budget_id", self.budget_id))
        object.__setattr__(self, "global_alpha", _alpha(self.global_alpha))
        for name in ("model_slots", "feature_slots", "eta_slots"):
            object.__setattr__(self, name, _slot(name, getattr(self, name)))
        total = self.model_slots * self.feature_slots * self.eta_slots
        if total > MAX_SELECTION_SLOTS:
            raise ValueError("selection family exceeds the configured slot limit")
        for name, slots in (
            ("model_index", self.model_slots),
            ("feature_index", self.feature_slots),
            ("eta_index", self.eta_slots),
        ):
            index = _slot(name, getattr(self, name))
            if index > slots:
                raise ValueError(f"{name} must not exceed its slot count")
            object.__setattr__(self, name, index)

    @property
    def total_slots(self) -> int:
        return self.model_slots * self.feature_slots * self.eta_slots

    @property
    def allocated_alpha(self) -> float:
        return self.global_alpha / self.total_slots

    @property
    def identity(self) -> str:
        return _digest(self.state_dict(include_identity=False))

    def state_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        state: dict[str, Any] = {
            "version": SELECTION_BUDGET_VERSION,
            "schema": SELECTION_BUDGET_SCHEMA,
            "budget_id": self.budget_id,
            "global_alpha": self.global_alpha,
            "model_slots": self.model_slots,
            "feature_slots": self.feature_slots,
            "eta_slots": self.eta_slots,
            "model_index": self.model_index,
            "feature_index": self.feature_index,
            "eta_index": self.eta_index,
        }
        if include_identity:
            state["identity"] = self.identity
        return state

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "SelectionBudget":
        if not isinstance(state, Mapping):
            raise ValueError("selection budget must be a mapping")
        expected = {
            "version",
            "schema",
            "budget_id",
            "global_alpha",
            "model_slots",
            "feature_slots",
            "eta_slots",
            "model_index",
            "feature_index",
            "eta_index",
            "identity",
        }
        if set(state) != expected:
            raise ValueError("selection budget fields are invalid")
        if state.get("version") != SELECTION_BUDGET_VERSION or state.get("schema") != SELECTION_BUDGET_SCHEMA:
            raise ValueError("unsupported selection budget version or schema")
        candidate = cls(
            budget_id=cast(str, state.get("budget_id")),
            global_alpha=cast(float, state.get("global_alpha")),
            model_slots=cast(int, state.get("model_slots")),
            feature_slots=cast(int, state.get("feature_slots")),
            eta_slots=cast(int, state.get("eta_slots")),
            model_index=cast(int, state.get("model_index")),
            feature_index=cast(int, state.get("feature_index")),
            eta_index=cast(int, state.get("eta_index")),
        )
        if state.get("identity") != candidate.identity:
            raise ValueError("selection budget identity mismatch")
        return candidate


__all__ = [
    "MAX_SELECTION_SLOTS",
    "SELECTION_BUDGET_SCHEMA",
    "SELECTION_BUDGET_VERSION",
    "SelectionBudget",
]
