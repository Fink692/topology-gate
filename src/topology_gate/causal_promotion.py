"""Causal challenger promotion adapter.

This module composes the dependency-light replay boundary with the existing
anytime-valid promotion gate.  Challenger and incumbent predictions are
frozen at the decision boundary; the gate sees a bounded paired utility only
when the corresponding label settles.  The two learner states, gate state,
and unresolved comparisons share one checkpointable model state.

The adapter is an evidence-control contract, not a market study.  Its
absolute-error utility is deliberately explicit and bounded so callers must
choose a utility scale before looking at labels.

Prediction workers are required to be pure by default: a successful
``predict`` call may not change checkpointed learner state.  The paired gate's
registration, alpha/score scales, eta rules, and epochs are also fingerprinted
at construction and checked at every replay boundary, so an external reset or
late family mutation fails closed.

``CausalPromotionConfig.minimum_labels`` provides an explicit burn-in boundary:
observed labels can update both learners before they become eligible to
advance the promotion e-process.  The count is checkpointed and is part of the
configuration identity.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, cast

from .asof import AsOfBook, AsOfSnapshot, TimePoint
from .causal_numeric import CausalFeaturePlan
from .manifest import ManifestValidationError, StudyManifest
from .promotion import GateStatus, PromotionGate, validate_eta
from .replay import (
    CausalReplayResult,
    ReplayConfig,
    ReplayPrediction,
    ReplayState,
    ReplayStatus,
    run_causal_replay,
)

CAUSAL_PROMOTION_SCHEMA = "topology_gate.causal_promotion"
# Pending comparisons now carry the canonical panel identity used to produce
# their paired predictions.  Older promotion checkpoints must not resume
# without that provenance.
# The registration-seal requirement, pure-prediction/gate-binding checks,
# operational missingness budget, learner binding, and next-boundary
# activation receipt are
# part of the certified controller contract, so older checkpoints cannot
# silently resume under the stronger interpretation.
CAUSAL_PROMOTION_VERSION = 9
MAX_CAUSAL_PROMOTION_PENDING = 8_192


class CausalPromotionError(ValueError):
    """Base error for the causal promotion adapter."""


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CausalPromotionError(f"{name} must be a non-empty string")
    return value


def _optional_text(value: Any, name: str) -> str | None:
    return None if value is None else _text(value, name)


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise CausalPromotionError(f"{name} must be finite")
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CausalPromotionError(f"{name} must be finite") from exc
    if not math.isfinite(converted):
        raise CausalPromotionError(f"{name} must be finite")
    return converted


def _nonnegative_limit(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CausalPromotionError(f"{name} must be a non-negative integer")
    if value > MAX_CAUSAL_PROMOTION_PENDING:
        raise CausalPromotionError(f"{name} exceeds the resource limit")
    return value


def _canonical(value: Any, name: str) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise CausalPromotionError(f"{name} must be JSON-safe") from exc


def _digest(value: Any, name: str) -> str:
    return hashlib.sha256(_canonical(value, name).encode("utf-8")).hexdigest()


def _clone_json(value: Any, name: str) -> Any:
    return json.loads(_canonical(value, name))


def _component_state(component: Any, name: str) -> dict[str, Any]:
    state_fn = getattr(component, "state_dict", None)
    if not callable(state_fn):
        raise CausalPromotionError(f"{name} must expose state_dict")
    state = state_fn()
    if not isinstance(state, Mapping):
        raise CausalPromotionError(f"{name} state must be a mapping")
    return cast(dict[str, Any], _clone_json(dict(state), f"{name} state"))


def _component_binding_digest(component: Any, name: str) -> str:
    """Bind stable learner identity without hashing mutable training state."""

    state = _component_state(component, name)
    declared = getattr(component, "config_identity", None)
    if callable(declared):
        declared = declared()
    if declared is None:
        declared = getattr(component, "identity", None)
        if callable(declared):
            declared = declared()
    if declared is not None:
        declared = _text(declared, f"{name} config identity")
    component_type = type(component)
    return _digest(
        {
            "module": component_type.__module__,
            "qualname": component_type.__qualname__,
            "declared_identity": declared,
            "state_schema": state.get("schema"),
            "state_version": state.get("version"),
            "state_model_id": state.get("model_id"),
            "state_config_identity": state.get("config_identity"),
        },
        f"{name} binding",
    )


def _restore_component(component: Any, state: Mapping[str, Any], name: str) -> None:
    load_fn = getattr(component, "load_state_dict", None)
    if not callable(load_fn):
        raise CausalPromotionError(f"{name} must expose load_state_dict")
    load_fn(state)


def _scalar_prediction(value: Any, name: str) -> float | None:
    """Normalize scalar workers while rejecting accidental multi-output use."""

    if value is None:
        return None
    candidate = value
    to_list = getattr(candidate, "tolist", None)
    if callable(to_list):
        candidate = to_list()
    if isinstance(candidate, (str, bytes, bytearray, Mapping)):
        raise CausalPromotionError(f"{name} must be a scalar prediction")
    if isinstance(candidate, Sequence):
        if len(candidate) != 1:
            raise CausalPromotionError(f"{name} must be a scalar prediction")
        candidate = candidate[0]
    if isinstance(candidate, bool):
        raise CausalPromotionError(f"{name} must be a scalar prediction")
    try:
        return float(candidate)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CausalPromotionError(f"{name} must be a scalar prediction") from exc


def _digest_features(features: Sequence[float]) -> str:
    return _digest(list(features), "feature row")


def _optional_panel_digest(value: Any, name: str) -> str | None:
    if value is None:
        return None
    digest = _text(value, name)
    if len(digest) != 64 or any(
        item not in "0123456789abcdefABCDEF" for item in digest
    ):
        raise CausalPromotionError(
            f"{name} must be a 64-character hexadecimal digest"
        )
    return digest.lower()


def _gate_binding_state(gate_state: Mapping[str, Any]) -> dict[str, Any]:
    """Extract gate choices that must not change during one replay family.

    Wealth, score history, status, and audit records are intentionally not in
    this binding: those are the evidence produced by the family.  Registration
    order, alpha/score scales, eta rules, and epochs are control choices and
    therefore are bound separately from the mutable evidence state.
    """

    raw_challengers = gate_state.get("challengers")
    if isinstance(raw_challengers, (str, bytes, bytearray)) or not isinstance(
        raw_challengers, Sequence
    ):
        raise CausalPromotionError("promotion gate challenger binding is invalid")
    challengers: list[dict[str, Any]] = []
    for entry in raw_challengers:
        if not isinstance(entry, Mapping):
            raise CausalPromotionError("promotion gate challenger binding is invalid")
        machine = entry.get("state")
        if not isinstance(machine, Mapping):
            raise CausalPromotionError("promotion gate challenger binding is invalid")
        process = machine.get("process")
        if not isinstance(process, Mapping):
            raise CausalPromotionError("promotion gate process binding is invalid")
        challengers.append(
            {
                "index": entry.get("index"),
                "challenger_id": machine.get("challenger_id"),
                "incumbent_id": machine.get("incumbent_id"),
                "process": {
                    "alpha": process.get("alpha"),
                    "score_bound": process.get("score_bound"),
                    "initial_wealth": process.get("initial_wealth"),
                    "epoch": process.get("epoch"),
                    "challenger_id": process.get("challenger_id"),
                    "eta": process.get("eta"),
                },
            }
        )
    return {
        "incumbent_id": gate_state.get("incumbent_id"),
        "global_alpha": gate_state.get("global_alpha"),
        "score_bound": gate_state.get("score_bound"),
        "initial_wealth": gate_state.get("initial_wealth"),
        "epoch": gate_state.get("epoch"),
        "registration_sealed": gate_state.get("registration_sealed"),
        "eta": gate_state.get("eta"),
        "challengers": challengers,
    }


def _gate_binding_digest(gate_state: Mapping[str, Any]) -> str:
    return _digest(_gate_binding_state(gate_state), "promotion gate binding")


def _gate_evidence_digest(gate_state: Mapping[str, Any]) -> str:
    """Fingerprint mutable evidence so outside observations fail closed."""

    return _digest(gate_state, "promotion gate evidence")


class CausalPromotionStatus(str, Enum):
    """Operational state of one paired promotion family."""

    OPEN = "open"
    BLOCKED = "blocked"
    PROMOTED = "promoted"


@dataclass(frozen=True, slots=True)
class CausalPromotionConfig:
    """Immutable control choices for one paired promotion family.

    The four ``max_*`` fields are predeclared missingness/quality budgets.
    Their strict defaults are zero: a certified family stops supplying new
    evidence after the first unusable prediction or label.  A non-zero budget
    is an explicit diagnostic choice and does not by itself prove that the
    corresponding missingness mechanism preserves the e-process null.
    """

    promotion_id: str = "causal-promotion"
    challenger_id: str = "challenger"
    incumbent_id: str = "incumbent"
    eta: float = 0.5
    utility_cap: float = 1.0
    minimum_labels: int = 1
    max_pending: int = MAX_CAUSAL_PROMOTION_PENDING
    max_non_observed_labels: int = 0
    max_unresolved_labels: int = 0
    max_abstained_predictions: int = 0
    max_invalid_predictions: int = 0
    require_sealed_registration: bool = True
    require_pure_predictions: bool = True

    def __post_init__(self) -> None:
        for name in ("promotion_id", "challenger_id", "incumbent_id"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        try:
            normalized_eta = validate_eta(self.eta)
        except (TypeError, ValueError) as exc:
            raise CausalPromotionError("eta must be in [0, 1]") from exc
        object.__setattr__(self, "eta", normalized_eta)
        cap = _finite(self.utility_cap, "utility_cap")
        if cap <= 0.0:
            raise CausalPromotionError("utility_cap must be positive")
        object.__setattr__(self, "utility_cap", cap)
        if (
            isinstance(self.minimum_labels, bool)
            or not isinstance(self.minimum_labels, int)
            or not 1 <= self.minimum_labels <= MAX_CAUSAL_PROMOTION_PENDING
        ):
            raise CausalPromotionError("minimum_labels exceeds the resource limit")
        if (
            isinstance(self.max_pending, bool)
            or not isinstance(self.max_pending, int)
            or not 1 <= self.max_pending <= MAX_CAUSAL_PROMOTION_PENDING
        ):
            raise CausalPromotionError("max_pending exceeds the resource limit")
        for name in (
            "max_non_observed_labels",
            "max_unresolved_labels",
            "max_abstained_predictions",
            "max_invalid_predictions",
        ):
            object.__setattr__(
                self, name, _nonnegative_limit(getattr(self, name), name)
            )
        if not isinstance(self.require_sealed_registration, bool):
            raise CausalPromotionError("require_sealed_registration must be boolean")
        if not isinstance(self.require_pure_predictions, bool):
            raise CausalPromotionError("require_pure_predictions must be boolean")
        if self.require_sealed_registration and not self.require_pure_predictions:
            raise CausalPromotionError(
                "certified causal promotion requires pure predictions"
            )

    @property
    def identity(self) -> str:
        payload = {
            "schema": CAUSAL_PROMOTION_SCHEMA,
            "version": CAUSAL_PROMOTION_VERSION,
            "promotion_id": self.promotion_id,
            "challenger_id": self.challenger_id,
            "incumbent_id": self.incumbent_id,
            "eta": self.eta,
            "utility_cap": self.utility_cap,
            "minimum_labels": self.minimum_labels,
            "max_pending": self.max_pending,
            "max_non_observed_labels": self.max_non_observed_labels,
            "max_unresolved_labels": self.max_unresolved_labels,
            "max_abstained_predictions": self.max_abstained_predictions,
            "max_invalid_predictions": self.max_invalid_predictions,
            "require_sealed_registration": self.require_sealed_registration,
            "require_pure_predictions": self.require_pure_predictions,
        }
        return _digest(payload, "promotion configuration")


@dataclass(frozen=True, slots=True)
class CausalPromotionStep:
    """Telemetry frozen at one prediction boundary."""

    prediction_id: str
    target_id: str
    challenger_prediction: float
    incumbent_prediction: float
    feature_digest: str
    gate_status: str
    feature_panel_digest: str | None = None


@dataclass(frozen=True, slots=True)
class CausalPromotionActivation:
    """Promotion crossing plus the first later replay decision boundary.

    ``effective_*`` is ``None`` when the replay segment ends before another
    prediction boundary.  This makes a threshold crossing distinguishable
    from an actually scheduled activation in a finite or checkpointed run.
    """

    promotion_id: str
    challenger_id: str
    prediction_id: str
    target_id: str
    label_id: str
    settlement_time: TimePoint
    settlement_sequence: int
    effective_prediction_id: str | None
    effective_decision_time: TimePoint | None
    effective_prediction_sequence: int | None


@dataclass(frozen=True, slots=True)
class _PendingComparison:
    prediction_id: str
    target_id: str
    features: tuple[float, ...]
    challenger_prediction: float
    incumbent_prediction: float
    feature_digest: str
    feature_panel_digest: str | None
    challenger_state_digest: str
    incumbent_state_digest: str


@dataclass(frozen=True, slots=True)
class _ControlState:
    observed_label_count: int
    non_observed_label_count: int
    unresolved_label_count: int
    abstained_prediction_count: int
    invalid_prediction_count: int
    promotion_blocked: bool
    promotion_block_reason: str | None
    promotion_prediction_id: str | None
    promotion_target_id: str | None
    promotion_label_id: str | None


class CausalPromotionModel:
    """Run paired learners and promotion evidence inside one replay model."""

    def __init__(
        self,
        challenger: Any,
        incumbent: Any,
        plan: CausalFeaturePlan,
        gate: PromotionGate,
        *,
        config: CausalPromotionConfig | None = None,
        study_manifest_digest: str | None = None,
    ) -> None:
        self._validate_learner(challenger, "challenger")
        self._validate_learner(incumbent, "incumbent")
        if not isinstance(plan, CausalFeaturePlan):
            raise TypeError("plan must be a CausalFeaturePlan")
        if not isinstance(gate, PromotionGate):
            raise TypeError("gate must be a PromotionGate")
        self.challenger = challenger
        self.incumbent = incumbent
        self.plan = plan
        self.gate = gate
        self.config = config or CausalPromotionConfig()
        self._study_manifest_digest = _optional_panel_digest(
            study_manifest_digest, "study manifest digest"
        )
        dimensions = {
            int(value)
            for value in (
                getattr(challenger, "n_features", None),
                getattr(incumbent, "n_features", None),
            )
            if value is not None
        }
        if len(dimensions) > 1:
            raise CausalPromotionError("paired learners must use one feature dimension")
        self._challenger_binding_identity = _component_binding_digest(
            challenger, "challenger"
        )
        self._incumbent_binding_identity = _component_binding_digest(
            incumbent, "incumbent"
        )
        self._validate_gate(gate)
        self._gate_binding_identity = _gate_binding_digest(gate.state_dict())
        self._gate_evidence_identity = _gate_evidence_digest(gate.state_dict())
        self._pending: dict[str, _PendingComparison] = {}
        self._prediction_count = 0
        self._observed_label_count = 0
        self._non_observed_label_count = 0
        self._unresolved_label_count = 0
        self._abstained_prediction_count = 0
        self._invalid_prediction_count = 0
        self._promotion_blocked = False
        self._promotion_block_reason: str | None = None
        self._promotion_prediction_id: str | None = None
        self._promotion_target_id: str | None = None
        self._promotion_label_id: str | None = None
        self._steps: list[CausalPromotionStep] = []

    @staticmethod
    def _validate_learner(learner: Any, name: str) -> None:
        for method in ("predict", "update", "state_dict", "load_state_dict"):
            if not callable(getattr(learner, method, None)):
                raise TypeError(f"{name} must expose {method}")

    def _validate_gate(self, gate: PromotionGate) -> None:
        if gate.incumbent_id != self.config.incumbent_id:
            raise CausalPromotionError("promotion gate incumbent does not match config")
        if self.config.require_sealed_registration and not gate.registration_sealed:
            raise CausalPromotionError(
                "certified causal promotion requires a pre-registered, sealed "
                "challenger family"
            )
        if self.config.challenger_id not in gate.challenger_ids:
            raise CausalPromotionError("promotion challenger is not registered with gate")
        if (
            gate.status is GateStatus.PROMOTED
            and gate.promoted_challenger_id != self.config.challenger_id
        ):
            raise CausalPromotionError(
                "promotion gate was promoted by a different challenger"
            )
        state = gate.state_dict()
        score_bound = _finite(state.get("score_bound"), "gate score bound")
        if not math.isclose(
            score_bound,
            self.config.utility_cap,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ):
            raise CausalPromotionError(
                "promotion utility_cap must match the gate score bound"
            )
        eta_state = state.get("eta")
        if not isinstance(eta_state, Mapping) or eta_state.get("kind") != "constant":
            raise CausalPromotionError(
                "causal promotion requires a constant predeclared gate eta"
            )
        gate_eta = _finite(eta_state.get("value"), "gate eta")
        if not math.isclose(gate_eta, self.config.eta, rel_tol=0.0, abs_tol=1.0e-15):
            raise CausalPromotionError("promotion config eta does not match gate eta")
        selected_process: Mapping[str, Any] | None = None
        challengers = state.get("challengers", ())
        if isinstance(challengers, (str, bytes, bytearray)) or not isinstance(
            challengers, Sequence
        ):
            raise CausalPromotionError("promotion gate challenger state is invalid")
        for entry in challengers:
            if not isinstance(entry, Mapping):
                raise CausalPromotionError("promotion gate challenger state is invalid")
            machine_state = entry.get("state")
            if not isinstance(machine_state, Mapping):
                raise CausalPromotionError("promotion gate challenger state is invalid")
            process_state = machine_state.get("process")
            if not isinstance(process_state, Mapping):
                raise CausalPromotionError("promotion gate process state is missing")
            candidate_eta_state = process_state.get("eta")
            if (
                not isinstance(candidate_eta_state, Mapping)
                or candidate_eta_state.get("kind") != "constant"
            ):
                raise CausalPromotionError(
                    "causal promotion requires a constant eta for every "
                    "registered challenger; selected challenger eta must be "
                    "constant"
                )
            if machine_state.get("challenger_id") == self.config.challenger_id:
                selected_process = process_state
        if selected_process is None:
            raise CausalPromotionError("promotion challenger state is missing")
        selected_eta_state = selected_process.get("eta")
        if (
            not isinstance(selected_eta_state, Mapping)
            or selected_eta_state.get("kind") != "constant"
        ):
            raise CausalPromotionError(
                "causal promotion requires a constant eta for the selected challenger"
            )
        selected_eta = _finite(selected_eta_state.get("value"), "challenger eta")
        if not math.isclose(
            selected_eta,
            self.config.eta,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ):
            raise CausalPromotionError(
                "promotion config eta does not match selected challenger eta"
            )

    def _assert_gate_binding(self) -> None:
        current = _gate_binding_digest(self.gate.state_dict())
        if current != self._gate_binding_identity:
            raise CausalPromotionError(
                "promotion gate registration, scale, eta, or epoch changed "
                "during the replay family"
            )

    def _assert_gate_evidence(self) -> None:
        current = _gate_evidence_digest(self.gate.state_dict())
        if current != self._gate_evidence_identity:
            raise CausalPromotionError(
                "promotion gate evidence changed outside the replay family"
            )

    def _refresh_gate_evidence_binding(self) -> None:
        self._gate_evidence_identity = _gate_evidence_digest(
            self.gate.state_dict()
        )

    def _assert_learner_bindings(self) -> None:
        if (
            _component_binding_digest(self.challenger, "challenger")
            != self._challenger_binding_identity
            or _component_binding_digest(self.incumbent, "incumbent")
            != self._incumbent_binding_identity
        ):
            raise CausalPromotionError(
                "challenger or incumbent learner identity changed during the replay family"
            )

    def _assert_bindings(self) -> None:
        self._assert_gate_binding()
        self._assert_gate_evidence()
        self._assert_learner_bindings()

    @property
    def steps(self) -> tuple[CausalPromotionStep, ...]:
        return tuple(self._steps)

    @property
    def pending_target_ids(self) -> tuple[str, ...]:
        return tuple(self._pending)

    @property
    def promoted(self) -> bool:
        return self.gate.status is GateStatus.PROMOTED

    @property
    def operational_status(self) -> CausalPromotionStatus:
        if self.promoted:
            return CausalPromotionStatus.PROMOTED
        if self._promotion_blocked:
            return CausalPromotionStatus.BLOCKED
        return CausalPromotionStatus.OPEN

    @property
    def promotion_block_reason(self) -> str | None:
        return self._promotion_block_reason

    @property
    def operational_counts(self) -> Mapping[str, int]:
        return {
            "observed_labels": self._observed_label_count,
            "non_observed_labels": self._non_observed_label_count,
            "unresolved_labels": self._unresolved_label_count,
            "abstained_predictions": self._abstained_prediction_count,
            "invalid_predictions": self._invalid_prediction_count,
        }

    @property
    def gate_state(self) -> Mapping[str, Any]:
        return self.gate.state_dict()

    def _utility(self, prediction: float, label: float) -> float:
        error = abs(prediction - label)
        if not math.isfinite(error):
            return -self.config.utility_cap
        return -min(error, self.config.utility_cap)

    def _capture_control_state(self) -> _ControlState:
        return _ControlState(
            observed_label_count=self._observed_label_count,
            non_observed_label_count=self._non_observed_label_count,
            unresolved_label_count=self._unresolved_label_count,
            abstained_prediction_count=self._abstained_prediction_count,
            invalid_prediction_count=self._invalid_prediction_count,
            promotion_blocked=self._promotion_blocked,
            promotion_block_reason=self._promotion_block_reason,
            promotion_prediction_id=self._promotion_prediction_id,
            promotion_target_id=self._promotion_target_id,
            promotion_label_id=self._promotion_label_id,
        )

    def _restore_control_state(self, state: _ControlState) -> None:
        self._observed_label_count = state.observed_label_count
        self._non_observed_label_count = state.non_observed_label_count
        self._unresolved_label_count = state.unresolved_label_count
        self._abstained_prediction_count = state.abstained_prediction_count
        self._invalid_prediction_count = state.invalid_prediction_count
        self._promotion_blocked = state.promotion_blocked
        self._promotion_block_reason = state.promotion_block_reason
        self._promotion_prediction_id = state.promotion_prediction_id
        self._promotion_target_id = state.promotion_target_id
        self._promotion_label_id = state.promotion_label_id

    def _block_if_over_budget(
        self, *, kind: str, count: int, limit: int, target_id: str
    ) -> None:
        if self.gate.status is not GateStatus.OPEN or count <= limit:
            return
        if not self._promotion_blocked:
            self._promotion_blocked = True
            self._promotion_block_reason = (
                f"{kind} budget exceeded for target {target_id!r}: "
                f"observed {count}, allowed {limit}"
            )

    def _record_prediction_issue(self, *, invalid: bool, target_id: str) -> None:
        if invalid:
            self._invalid_prediction_count += 1
            self._block_if_over_budget(
                kind="invalid prediction",
                count=self._invalid_prediction_count,
                limit=self.config.max_invalid_predictions,
                target_id=target_id,
            )
        else:
            self._abstained_prediction_count += 1
            self._block_if_over_budget(
                kind="abstained prediction",
                count=self._abstained_prediction_count,
                limit=self.config.max_abstained_predictions,
                target_id=target_id,
            )

    def _record_label_issue(self, *, status: ReplayStatus, target_id: str) -> None:
        if status is ReplayStatus.UNRESOLVED:
            self._unresolved_label_count += 1
            self._block_if_over_budget(
                kind="unresolved label",
                count=self._unresolved_label_count,
                limit=self.config.max_unresolved_labels,
                target_id=target_id,
            )
        elif status is not ReplayStatus.OBSERVED:
            self._non_observed_label_count += 1
            self._block_if_over_budget(
                kind="non-observed label",
                count=self._non_observed_label_count,
                limit=self.config.max_non_observed_labels,
                target_id=target_id,
            )

    def _rollback(
        self,
        challenger_state: Mapping[str, Any],
        incumbent_state: Mapping[str, Any],
        gate_state: Mapping[str, Any],
        pending: Mapping[str, _PendingComparison],
        prediction_count: int,
        steps: Sequence[CausalPromotionStep],
        control_state: _ControlState | None = None,
    ) -> None:
        _restore_component(self.challenger, challenger_state, "challenger")
        _restore_component(self.incumbent, incumbent_state, "incumbent")
        self.gate.load_state_dict(gate_state, eta=self.config.eta)
        self._refresh_gate_evidence_binding()
        self._pending = dict(pending)
        self._prediction_count = prediction_count
        if control_state is not None:
            self._restore_control_state(control_state)
        self._steps = list(steps)

    def predict(self, snapshot: AsOfSnapshot, target_id: str) -> float | None:
        self._assert_bindings()
        target = _text(target_id, "target_id")
        if self.promoted:
            # Once the gate has promoted a challenger, no new comparison is
            # silently fed to the old evidence family.  The replay records an
            # explicit abstention until the caller starts a new gate epoch.
            return None
        features, feature_panel = self.plan.extract_with_panel(snapshot, target)
        feature_panel_digest = _optional_panel_digest(
            None if feature_panel is None else feature_panel.digest,
            "feature panel digest",
        )
        if target in self._pending:
            raise CausalPromotionError(f"target {target!r} already has pending evidence")
        if len(self._pending) >= self.config.max_pending:
            raise CausalPromotionError("pending promotion contexts exceed the resource limit")
        challenger_before = _component_state(self.challenger, "challenger")
        incumbent_before = _component_state(self.incumbent, "incumbent")
        gate_before = _clone_json(self.gate.state_dict(), "promotion gate state")
        pending_before = dict(self._pending)
        steps_before = tuple(self._steps)
        count_before = self._prediction_count
        control_before = self._capture_control_state()
        try:
            challenger_raw = self.challenger.predict(features)
            challenger_prediction = _scalar_prediction(
                challenger_raw, "challenger prediction"
            )
            if self.config.require_pure_predictions:
                if _component_state(self.challenger, "challenger") != challenger_before:
                    raise CausalPromotionError(
                        "challenger predict mutated checkpointed state"
                    )
            incumbent_raw = self.incumbent.predict(features)
            incumbent_prediction = _scalar_prediction(incumbent_raw, "incumbent prediction")
            if self.config.require_pure_predictions:
                if _component_state(self.incumbent, "incumbent") != incumbent_before:
                    raise CausalPromotionError(
                        "incumbent predict mutated checkpointed state"
                    )
            self._assert_bindings()
            invalid_prediction = (
                challenger_prediction is not None
                and not math.isfinite(challenger_prediction)
            ) or (
                incumbent_prediction is not None
                and not math.isfinite(incumbent_prediction)
            )
            if (
                challenger_prediction is None
                or incumbent_prediction is None
                or not math.isfinite(challenger_prediction)
                or not math.isfinite(incumbent_prediction)
            ):
                self._rollback(
                    challenger_before,
                    incumbent_before,
                    gate_before,
                    pending_before,
                    count_before,
                    steps_before,
                    control_before,
                )
                self._record_prediction_issue(
                    invalid=invalid_prediction,
                    target_id=target,
                )
                return None
            feature_digest = _digest_features(features)
            prediction_id = (
                f"{self.config.promotion_id}:prediction:{self._prediction_count}"
            )
            pending = _PendingComparison(
                prediction_id=prediction_id,
                target_id=target,
                features=tuple(features),
                challenger_prediction=challenger_prediction,
                incumbent_prediction=incumbent_prediction,
                feature_digest=feature_digest,
                feature_panel_digest=feature_panel_digest,
                challenger_state_digest=_digest(challenger_before, "challenger state"),
                incumbent_state_digest=_digest(incumbent_before, "incumbent state"),
            )
            self._pending[target] = pending
            self._steps.append(
                CausalPromotionStep(
                    prediction_id=prediction_id,
                    target_id=target,
                    challenger_prediction=challenger_prediction,
                    incumbent_prediction=incumbent_prediction,
                    feature_digest=feature_digest,
                    gate_status=self.gate.status.value,
                    feature_panel_digest=feature_panel_digest,
                )
            )
            self._prediction_count += 1
            return challenger_prediction
        except Exception:
            self._rollback(
                challenger_before,
                incumbent_before,
                gate_before,
                pending_before,
                count_before,
                steps_before,
                control_before,
            )
            raise

    def on_label(
        self, prediction: ReplayPrediction, label: Any, score: float | None
    ) -> None:
        del score
        self._assert_bindings()
        if prediction.status is not ReplayStatus.PREDICTED:
            return
        if getattr(label, "status", None) != "observed":
            return
        pending = self._pending.get(prediction.target_id)
        if pending is None:
            raise CausalPromotionError(
                f"no frozen promotion context exists for {prediction.target_id!r}"
            )
        if prediction.value != pending.challenger_prediction:
            raise CausalPromotionError("promotion prediction value was not frozen")
        label_target = getattr(label, "target_id", None)
        if label_target is not None and label_target != pending.target_id:
            raise CausalPromotionError("promotion label target does not match prediction")
        target = _finite(getattr(label, "value", None), "label value")
        challenger_before = _component_state(self.challenger, "challenger")
        incumbent_before = _component_state(self.incumbent, "incumbent")
        gate_before = _clone_json(self.gate.state_dict(), "promotion gate state")
        pending_before = dict(self._pending)
        steps_before = tuple(self._steps)
        count_before = self._prediction_count
        control_before = self._capture_control_state()
        try:
            challenger_utility = self._utility(pending.challenger_prediction, target)
            incumbent_utility = self._utility(pending.incumbent_prediction, target)
            self.challenger.update(pending.features, target)
            self.incumbent.update(pending.features, target)
            self._observed_label_count += 1
            if (
                self.gate.status is GateStatus.OPEN
                and not self._promotion_blocked
                and self._observed_label_count >= self.config.minimum_labels
            ):
                decision = self.gate.observe_utilities(
                    self.config.challenger_id,
                    challenger_utility,
                    incumbent_utility,
                    eta=self.config.eta,
                    metadata={
                        "prediction_id": pending.prediction_id,
                        "replay_prediction_id": prediction.prediction_id,
                        "label_id": _text(
                            getattr(label, "label_id", None), "label_id"
                        ),
                        "target_id": pending.target_id,
                        "feature_digest": pending.feature_digest,
                        "feature_panel_digest": pending.feature_panel_digest,
                        "utility_spec_id": "bounded_absolute_error.v1",
                        "challenger_state_digest": pending.challenger_state_digest,
                        "incumbent_state_digest": pending.incumbent_state_digest,
                    },
                )
                if decision.promoted:
                    self._promotion_prediction_id = prediction.prediction_id
                    self._promotion_target_id = pending.target_id
                    self._promotion_label_id = _text(
                        getattr(label, "label_id", None), "label_id"
                    )
                self._refresh_gate_evidence_binding()
            self._assert_bindings()
        except Exception:
            self._rollback(
                challenger_before,
                incumbent_before,
                gate_before,
                pending_before,
                count_before,
                steps_before,
                control_before,
            )
            raise

    def on_resolution(
        self, prediction: ReplayPrediction, label: Any | None, status: ReplayStatus
    ) -> None:
        self._assert_bindings()
        if prediction.status is not ReplayStatus.PREDICTED:
            return
        pending = self._pending.get(prediction.target_id)
        if pending is None:
            raise CausalPromotionError(
                f"no unresolved promotion context exists for {prediction.target_id!r}"
            )
        if status is ReplayStatus.OBSERVED:
            if label is None or getattr(label, "status", None) != "observed":
                raise CausalPromotionError("observed promotion resolution is missing its label")
        elif label is not None and getattr(label, "target_id", pending.target_id) != pending.target_id:
            raise CausalPromotionError("promotion resolution label target does not match")
        if status is not ReplayStatus.OBSERVED:
            self._record_label_issue(status=status, target_id=pending.target_id)
        self._pending.pop(prediction.target_id, None)

    def state_dict(self) -> dict[str, Any]:
        self._assert_bindings()
        return {
            "schema": CAUSAL_PROMOTION_SCHEMA,
            "version": CAUSAL_PROMOTION_VERSION,
            "promotion_id": self.config.promotion_id,
            "config_identity": self.config.identity,
            "plan_identity": self.plan.identity,
            "study_manifest_digest": self._study_manifest_digest,
            "gate_binding_identity": self._gate_binding_identity,
            "gate_evidence_identity": self._gate_evidence_identity,
            "challenger_binding_identity": self._challenger_binding_identity,
            "incumbent_binding_identity": self._incumbent_binding_identity,
            "challenger": _component_state(self.challenger, "challenger"),
            "incumbent": _component_state(self.incumbent, "incumbent"),
            "gate": self.gate.state_dict(),
            "pending": {
                target: {
                    "prediction_id": value.prediction_id,
                    "target_id": value.target_id,
                    "features": list(value.features),
                    "challenger_prediction": value.challenger_prediction,
                    "incumbent_prediction": value.incumbent_prediction,
                    "feature_digest": value.feature_digest,
                    "feature_panel_digest": value.feature_panel_digest,
                    "challenger_state_digest": value.challenger_state_digest,
                    "incumbent_state_digest": value.incumbent_state_digest,
                }
                for target, value in sorted(self._pending.items())
            },
            "prediction_count": self._prediction_count,
            "observed_label_count": self._observed_label_count,
            "non_observed_label_count": self._non_observed_label_count,
            "unresolved_label_count": self._unresolved_label_count,
            "abstained_prediction_count": self._abstained_prediction_count,
            "invalid_prediction_count": self._invalid_prediction_count,
            "promotion_blocked": self._promotion_blocked,
            "promotion_block_reason": self._promotion_block_reason,
            "promotion_prediction_id": self._promotion_prediction_id,
            "promotion_target_id": self._promotion_target_id,
            "promotion_label_id": self._promotion_label_id,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping):
            raise CausalPromotionError("causal promotion state must be a mapping")
        self._assert_bindings()
        expected_state_fields = {
            "schema",
            "version",
            "promotion_id",
            "config_identity",
            "plan_identity",
            "study_manifest_digest",
            "gate_binding_identity",
            "gate_evidence_identity",
            "challenger_binding_identity",
            "incumbent_binding_identity",
            "challenger",
            "incumbent",
            "gate",
            "pending",
            "prediction_count",
            "observed_label_count",
            "non_observed_label_count",
            "unresolved_label_count",
            "abstained_prediction_count",
            "invalid_prediction_count",
            "promotion_blocked",
            "promotion_block_reason",
            "promotion_prediction_id",
            "promotion_target_id",
            "promotion_label_id",
        }
        if set(state) != expected_state_fields:
            raise CausalPromotionError("causal promotion state fields are invalid")
        if (
            state.get("schema") != CAUSAL_PROMOTION_SCHEMA
            or state.get("version") != CAUSAL_PROMOTION_VERSION
        ):
            raise CausalPromotionError("unsupported causal promotion state")
        if state.get("promotion_id") != self.config.promotion_id:
            raise CausalPromotionError("causal promotion identity mismatch")
        if state.get("config_identity") != self.config.identity:
            raise CausalPromotionError("causal promotion configuration mismatch")
        if state.get("plan_identity") != self.plan.identity:
            raise CausalPromotionError("causal promotion feature plan mismatch")
        if state.get("study_manifest_digest") != self._study_manifest_digest:
            raise CausalPromotionError("causal promotion study manifest mismatch")
        if state.get("gate_binding_identity") != self._gate_binding_identity:
            raise CausalPromotionError("causal promotion gate binding mismatch")
        if state.get("gate_evidence_identity") != _gate_evidence_digest(
            cast(Mapping[str, Any], state.get("gate"))
        ):
            raise CausalPromotionError("causal promotion gate evidence mismatch")
        if state.get("challenger_binding_identity") != self._challenger_binding_identity:
            raise CausalPromotionError("causal promotion challenger binding mismatch")
        if state.get("incumbent_binding_identity") != self._incumbent_binding_identity:
            raise CausalPromotionError("causal promotion incumbent binding mismatch")
        challenger_state = state.get("challenger")
        incumbent_state = state.get("incumbent")
        gate_state = state.get("gate")
        if not isinstance(challenger_state, Mapping) or not isinstance(
            incumbent_state, Mapping
        ):
            raise CausalPromotionError("causal promotion learner state is missing")
        if not isinstance(gate_state, Mapping):
            raise CausalPromotionError("causal promotion gate state is missing")
        try:
            candidate_gate = PromotionGate.from_state_dict(
                gate_state, eta=self.config.eta
            )
        except (TypeError, ValueError) as exc:
            raise CausalPromotionError("causal promotion gate state is invalid") from exc
        self._validate_gate(candidate_gate)
        if _gate_binding_digest(gate_state) != self._gate_binding_identity:
            raise CausalPromotionError("causal promotion gate binding mismatch")
        raw_pending = state.get("pending", {})
        if not isinstance(raw_pending, Mapping):
            raise CausalPromotionError("causal promotion pending state must be a mapping")
        if len(raw_pending) > self.config.max_pending:
            raise CausalPromotionError("causal promotion pending state exceeds its limit")
        pending: dict[str, _PendingComparison] = {}
        expected_dimensions = {
            int(value)
            for value in (
                getattr(self.challenger, "n_features", None),
                getattr(self.incumbent, "n_features", None),
            )
            if value is not None
        }
        if len(expected_dimensions) > 1:
            raise CausalPromotionError("paired learners use different feature dimensions")
        seen_prediction_ids: set[str] = set()
        for target, raw in raw_pending.items():
            target_id = _text(target, "pending target")
            if not isinstance(raw, Mapping):
                raise CausalPromotionError("pending comparison must be a mapping")
            if set(raw) != {
                "prediction_id",
                "target_id",
                "features",
                "challenger_prediction",
                "incumbent_prediction",
                "feature_digest",
                "feature_panel_digest",
                "challenger_state_digest",
                "incumbent_state_digest",
            }:
                raise CausalPromotionError("pending comparison fields are invalid")
            features_raw = raw.get("features")
            if isinstance(features_raw, (str, bytes, bytearray)) or not isinstance(
                features_raw, Sequence
            ):
                raise CausalPromotionError("pending features must be a sequence")
            features = tuple(_finite(value, "pending feature") for value in features_raw)
            if not features or expected_dimensions and len(features) not in expected_dimensions:
                raise CausalPromotionError("pending feature dimension does not match learner")
            prediction_id = _text(raw.get("prediction_id"), "pending prediction_id")
            if prediction_id in seen_prediction_ids:
                raise CausalPromotionError("pending prediction IDs must be unique")
            seen_prediction_ids.add(prediction_id)
            if _text(raw.get("target_id"), "pending target_id") != target_id:
                raise CausalPromotionError("pending target identity does not match its key")
            challenger_prediction = _finite(
                raw.get("challenger_prediction"), "pending challenger prediction"
            )
            incumbent_prediction = _finite(
                raw.get("incumbent_prediction"), "pending incumbent prediction"
            )
            feature_digest = _text(raw.get("feature_digest"), "pending feature digest")
            if feature_digest != _digest_features(features):
                raise CausalPromotionError("pending feature digest does not match features")
            feature_panel_digest = _optional_panel_digest(
                raw.get("feature_panel_digest"), "pending feature panel digest"
            )
            challenger_digest = _text(
                raw.get("challenger_state_digest"), "pending challenger state digest"
            )
            incumbent_digest = _text(
                raw.get("incumbent_state_digest"), "pending incumbent state digest"
            )
            pending[target_id] = _PendingComparison(
                prediction_id=prediction_id,
                target_id=target_id,
                features=features,
                challenger_prediction=challenger_prediction,
                incumbent_prediction=incumbent_prediction,
                feature_digest=feature_digest,
                feature_panel_digest=feature_panel_digest,
                challenger_state_digest=challenger_digest,
                incumbent_state_digest=incumbent_digest,
            )
        prediction_count = state.get("prediction_count")
        if (
            isinstance(prediction_count, bool)
            or not isinstance(prediction_count, int)
            or prediction_count < len(pending)
        ):
            raise CausalPromotionError("prediction_count is invalid")
        observed_label_count = state.get("observed_label_count")
        if (
            isinstance(observed_label_count, bool)
            or not isinstance(observed_label_count, int)
            or observed_label_count < 0
        ):
            raise CausalPromotionError("observed_label_count is invalid")
        non_observed_label_count = _nonnegative_limit(
            state.get("non_observed_label_count"), "non_observed_label_count"
        )
        unresolved_label_count = _nonnegative_limit(
            state.get("unresolved_label_count"), "unresolved_label_count"
        )
        abstained_prediction_count = _nonnegative_limit(
            state.get("abstained_prediction_count"), "abstained_prediction_count"
        )
        invalid_prediction_count = _nonnegative_limit(
            state.get("invalid_prediction_count"), "invalid_prediction_count"
        )
        promotion_blocked = state.get("promotion_blocked")
        if not isinstance(promotion_blocked, bool):
            raise CausalPromotionError("promotion_blocked must be boolean")
        promotion_block_reason = _optional_text(
            state.get("promotion_block_reason"), "promotion_block_reason"
        )
        if promotion_blocked != (promotion_block_reason is not None):
            raise CausalPromotionError(
                "promotion_blocked and promotion_block_reason disagree"
            )
        promotion_prediction_id = _optional_text(
            state.get("promotion_prediction_id"), "promotion_prediction_id"
        )
        promotion_target_id = _optional_text(
            state.get("promotion_target_id"), "promotion_target_id"
        )
        promotion_label_id = _optional_text(
            state.get("promotion_label_id"), "promotion_label_id"
        )
        promotion_event_present = all(
            value is not None
            for value in (
                promotion_prediction_id,
                promotion_target_id,
                promotion_label_id,
            )
        )
        if any(
            value is not None
            for value in (
                promotion_prediction_id,
                promotion_target_id,
                promotion_label_id,
            )
        ) and not promotion_event_present:
            raise CausalPromotionError("promotion activation fields are incomplete")
        gate_observations = candidate_gate.challenger_state(
            self.config.challenger_id
        ).observations
        if observed_label_count < gate_observations:
            raise CausalPromotionError(
                "observed_label_count cannot be below promotion observations"
            )
        if candidate_gate.status is GateStatus.PROMOTED and not promotion_event_present:
            raise CausalPromotionError(
                "promoted causal promotion state is missing its activation event"
            )
        if candidate_gate.status is not GateStatus.PROMOTED and promotion_event_present:
            raise CausalPromotionError(
                "open causal promotion state contains an activation event"
            )
        over_budget = (
            non_observed_label_count > self.config.max_non_observed_labels
            or unresolved_label_count > self.config.max_unresolved_labels
            or abstained_prediction_count > self.config.max_abstained_predictions
            or invalid_prediction_count > self.config.max_invalid_predictions
        )
        if candidate_gate.status is GateStatus.OPEN and promotion_blocked != over_budget:
            raise CausalPromotionError(
                "promotion operational budget state is inconsistent"
            )
        if candidate_gate.status is GateStatus.PROMOTED and promotion_blocked:
            raise CausalPromotionError(
                "promoted causal promotion state cannot be operationally blocked"
            )
        old_challenger = _component_state(self.challenger, "challenger")
        old_incumbent = _component_state(self.incumbent, "incumbent")
        old_gate = _clone_json(self.gate.state_dict(), "promotion gate state")
        try:
            _restore_component(self.challenger, challenger_state, "challenger")
            _restore_component(self.incumbent, incumbent_state, "incumbent")
            self.gate.load_state_dict(gate_state, eta=self.config.eta)
            self._refresh_gate_evidence_binding()
        except Exception:
            self._rollback(
                old_challenger,
                old_incumbent,
                old_gate,
                self._pending,
                self._prediction_count,
                self._steps,
            )
            raise
        self._pending = pending
        self._prediction_count = prediction_count
        self._observed_label_count = observed_label_count
        self._non_observed_label_count = non_observed_label_count
        self._unresolved_label_count = unresolved_label_count
        self._abstained_prediction_count = abstained_prediction_count
        self._invalid_prediction_count = invalid_prediction_count
        self._promotion_blocked = promotion_blocked
        self._promotion_block_reason = promotion_block_reason
        self._promotion_prediction_id = promotion_prediction_id
        self._promotion_target_id = promotion_target_id
        self._promotion_label_id = promotion_label_id
        self._steps = []
        self._assert_bindings()


@dataclass(frozen=True, slots=True)
class CausalPromotionReplayResult:
    """Causal replay output plus paired-promotion telemetry."""

    replay: CausalReplayResult
    steps: tuple[CausalPromotionStep, ...]
    prediction_start: int = 0
    study_manifest_digest: str | None = None

    @property
    def all_predictions(self) -> tuple[ReplayPrediction, ...]:
        return self.replay.predictions

    @property
    def predictions(self) -> tuple[ReplayPrediction, ...]:
        return self.replay.predictions[self.prediction_start :]

    @property
    def pending_target_ids(self) -> tuple[str, ...]:
        return self.replay.pending_target_ids

    @property
    def promoted(self) -> bool:
        model_state = self.replay.state.model_state
        if not isinstance(model_state, Mapping):
            return False
        gate_state = model_state.get("gate")
        return isinstance(gate_state, Mapping) and gate_state.get(
            "promoted_challenger_id"
        ) is not None

    @property
    def operational_status(self) -> CausalPromotionStatus:
        model_state = self.replay.state.model_state
        if not isinstance(model_state, Mapping):
            return CausalPromotionStatus.BLOCKED
        gate_state = model_state.get("gate")
        if isinstance(gate_state, Mapping) and gate_state.get(
            "promoted_challenger_id"
        ) is not None:
            return CausalPromotionStatus.PROMOTED
        if model_state.get("promotion_blocked") is True:
            return CausalPromotionStatus.BLOCKED
        return CausalPromotionStatus.OPEN

    @property
    def promotion_block_reason(self) -> str | None:
        model_state = self.replay.state.model_state
        if not isinstance(model_state, Mapping):
            return "replay model state is not a mapping"
        return _optional_text(
            model_state.get("promotion_block_reason"), "promotion_block_reason"
        )

    @property
    def operational_counts(self) -> Mapping[str, int]:
        model_state = self.replay.state.model_state
        if not isinstance(model_state, Mapping):
            return {}
        values = {
            "observed_labels": model_state.get("observed_label_count"),
            "non_observed_labels": model_state.get("non_observed_label_count"),
            "unresolved_labels": model_state.get("unresolved_label_count"),
            "abstained_predictions": model_state.get(
                "abstained_prediction_count"
            ),
            "invalid_predictions": model_state.get("invalid_prediction_count"),
        }
        return {
            key: value
            for key, value in values.items()
            if isinstance(value, int) and not isinstance(value, bool)
        }

    @property
    def promotion_activation(self) -> CausalPromotionActivation | None:
        """Return the crossing and its first later replay boundary, if any."""

        model_state = self.replay.state.model_state
        if not isinstance(model_state, Mapping):
            return None
        prediction_id = _optional_text(
            model_state.get("promotion_prediction_id"), "promotion_prediction_id"
        )
        target_id = _optional_text(
            model_state.get("promotion_target_id"), "promotion_target_id"
        )
        label_id = _optional_text(
            model_state.get("promotion_label_id"), "promotion_label_id"
        )
        if prediction_id is None or target_id is None or label_id is None:
            return None
        resolution = next(
            (
                item
                for item in self.replay.resolutions
                if item.prediction_id == prediction_id
                and item.status is ReplayStatus.OBSERVED
            ),
            None,
        )
        if resolution is None:
            return None
        if resolution.target_id != target_id or resolution.label_id != label_id:
            raise CausalPromotionError("promotion activation does not match resolution")
        effective = next(
            (
                item
                for item in self.replay.predictions
                if item.sequence > resolution.sequence
            ),
            None,
        )
        gate_state = model_state.get("gate")
        if not isinstance(gate_state, Mapping):
            raise CausalPromotionError("promotion activation is missing gate state")
        challenger_id = _text(
            gate_state.get("promoted_challenger_id"),
            "promoted challenger id",
        )
        return CausalPromotionActivation(
            promotion_id=_text(model_state.get("promotion_id"), "promotion_id"),
            challenger_id=challenger_id,
            prediction_id=prediction_id,
            target_id=target_id,
            label_id=label_id,
            settlement_time=resolution.settlement_time,
            settlement_sequence=resolution.sequence,
            effective_prediction_id=None if effective is None else effective.prediction_id,
            effective_decision_time=None
            if effective is None
            else effective.decision_time,
            effective_prediction_sequence=None
            if effective is None
            else effective.sequence,
        )

    @property
    def feature_panel_digests(self) -> tuple[str | None, ...]:
        """Return canonical feature-panel identities captured per step."""

        return tuple(item.feature_panel_digest for item in self.steps)

    @property
    def state(self) -> ReplayState:
        return self.replay.state

    def state_dict(self) -> dict[str, Any]:
        return self.replay.state_dict()


def run_causal_promotion_replay(
    book: AsOfBook,
    decision_times: Sequence[Any],
    target_ids: Sequence[str],
    *,
    plan: CausalFeaturePlan,
    challenger: Any,
    incumbent: Any,
    gate: PromotionGate,
    config: CausalPromotionConfig | None = None,
    replay_config: ReplayConfig | None = None,
    model_state: Mapping[str, Any] | None = None,
    initial_state: ReplayState | None = None,
    study_manifest: StudyManifest | None = None,
    study_phase: str | None = None,
    decision_indices: Sequence[int] | None = None,
) -> CausalPromotionReplayResult:
    """Run paired challenger promotion behind the causal replay transition."""

    if study_manifest is None:
        if study_phase is not None or decision_indices is not None:
            raise CausalPromotionError(
                "study_phase and decision_indices require a study_manifest"
            )
        study_manifest_digest = None
    else:
        if not isinstance(study_manifest, StudyManifest):
            raise CausalPromotionError("study_manifest must be a StudyManifest")
        if study_phase is None or decision_indices is None:
            raise CausalPromotionError(
                "study_phase and decision_indices are required with a study_manifest"
            )
        indices = tuple(decision_indices)
        if len(indices) != len(decision_times):
            raise CausalPromotionError(
                "decision_indices must align with decision_times"
            )
        try:
            study_manifest.assert_indices_allowed(indices, study_phase)
        except ManifestValidationError as exc:
            raise CausalPromotionError(str(exc)) from exc
        study_manifest_digest = study_manifest.digest

    settings_config = config or CausalPromotionConfig()
    model = CausalPromotionModel(
        challenger,
        incumbent,
        plan,
        gate,
        config=settings_config,
        study_manifest_digest=study_manifest_digest,
    )
    if model_state is not None:
        model.load_state_dict(model_state)
    elif initial_state is not None:
        raise CausalPromotionError(
            "model_state is required when resuming a causal promotion replay"
        )
    settings = replay_config or ReplayConfig(
        model_id=settings_config.promotion_id,
        score_id="none",
        require_model_state=True,
    )
    if settings.model_id != settings_config.promotion_id:
        raise CausalPromotionError("replay and promotion model identities must match")
    result = run_causal_replay(
        book,
        decision_times,
        target_ids,
        model.predict,
        model=model,
        on_label=model.on_label,
        on_resolution=model.on_resolution,
        config=settings,
        initial_state=initial_state,
    )
    prediction_start = 0 if initial_state is None else len(initial_state.predictions)
    return CausalPromotionReplayResult(
        result,
        model.steps,
        prediction_start,
        study_manifest_digest,
    )


__all__ = [
    "CAUSAL_PROMOTION_SCHEMA",
    "CAUSAL_PROMOTION_VERSION",
    "CausalPromotionActivation",
    "CausalPromotionConfig",
    "CausalPromotionError",
    "CausalPromotionModel",
    "CausalPromotionReplayResult",
    "CausalPromotionStatus",
    "CausalPromotionStep",
    "MAX_CAUSAL_PROMOTION_PENDING",
    "run_causal_promotion_replay",
]
