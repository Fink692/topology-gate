"""Causal challenger promotion adapter.

This module composes the dependency-light replay boundary with the existing
anytime-valid promotion gate.  Challenger and incumbent predictions are
frozen at the decision boundary; the gate sees a bounded paired utility only
when the corresponding label settles.  The two learner states, gate state,
and unresolved comparisons share one checkpointable model state.

The adapter is an evidence-control contract, not a market study.  Its
absolute-error utility is deliberately explicit and bounded so callers must
choose a utility scale before looking at labels.

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
from typing import Any, cast

from .asof import AsOfBook, AsOfSnapshot
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
# The registration-seal requirement is part of the certified controller
# contract, so older checkpoints cannot silently resume under the stronger
# pre-registration interpretation.
CAUSAL_PROMOTION_VERSION = 5
MAX_CAUSAL_PROMOTION_PENDING = 8_192


class CausalPromotionError(ValueError):
    """Base error for the causal promotion adapter."""


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CausalPromotionError(f"{name} must be a non-empty string")
    return value


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


@dataclass(frozen=True, slots=True)
class CausalPromotionConfig:
    """Immutable control choices for one paired promotion family."""

    promotion_id: str = "causal-promotion"
    challenger_id: str = "challenger"
    incumbent_id: str = "incumbent"
    eta: float = 0.5
    utility_cap: float = 1.0
    minimum_labels: int = 1
    max_pending: int = MAX_CAUSAL_PROMOTION_PENDING
    require_sealed_registration: bool = True

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
        if not isinstance(self.require_sealed_registration, bool):
            raise CausalPromotionError("require_sealed_registration must be boolean")

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
            "require_sealed_registration": self.require_sealed_registration,
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
        self._validate_gate(gate)
        self._pending: dict[str, _PendingComparison] = {}
        self._prediction_count = 0
        self._observed_label_count = 0
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
        state = gate.state_dict()
        eta_state = state.get("eta")
        if not isinstance(eta_state, Mapping) or eta_state.get("kind") != "constant":
            raise CausalPromotionError(
                "causal promotion requires a constant predeclared gate eta"
            )
        gate_eta = _finite(eta_state.get("value"), "gate eta")
        if not math.isclose(gate_eta, self.config.eta, rel_tol=0.0, abs_tol=1.0e-15):
            raise CausalPromotionError("promotion config eta does not match gate eta")
        selected_process: Mapping[str, Any] | None = None
        for entry in state.get("challengers", ()):
            if not isinstance(entry, Mapping):
                continue
            machine_state = entry.get("state")
            if not isinstance(machine_state, Mapping):
                continue
            if machine_state.get("challenger_id") != self.config.challenger_id:
                continue
            process_state = machine_state.get("process")
            if isinstance(process_state, Mapping):
                selected_process = process_state
            break
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
    def gate_state(self) -> Mapping[str, Any]:
        return self.gate.state_dict()

    def _utility(self, prediction: float, label: float) -> float:
        error = abs(prediction - label)
        if not math.isfinite(error):
            return -self.config.utility_cap
        return -min(error, self.config.utility_cap)

    def _rollback(
        self,
        challenger_state: Mapping[str, Any],
        incumbent_state: Mapping[str, Any],
        gate_state: Mapping[str, Any],
        pending: Mapping[str, _PendingComparison],
        prediction_count: int,
        steps: Sequence[CausalPromotionStep],
        observed_label_count: int | None = None,
    ) -> None:
        _restore_component(self.challenger, challenger_state, "challenger")
        _restore_component(self.incumbent, incumbent_state, "incumbent")
        self.gate.load_state_dict(gate_state, eta=self.config.eta)
        self._pending = dict(pending)
        self._prediction_count = prediction_count
        if observed_label_count is not None:
            self._observed_label_count = observed_label_count
        self._steps = list(steps)

    def predict(self, snapshot: AsOfSnapshot, target_id: str) -> float | None:
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
        labels_before = self._observed_label_count
        try:
            challenger_prediction = _scalar_prediction(
                self.challenger.predict(features), "challenger prediction"
            )
            incumbent_prediction = _scalar_prediction(
                self.incumbent.predict(features), "incumbent prediction"
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
                labels_before,
            )
            raise

    def on_label(
        self, prediction: ReplayPrediction, label: Any, score: float | None
    ) -> None:
        del score
        if prediction.status is not ReplayStatus.PREDICTED:
            return
        if getattr(label, "status", None) != "observed":
            return
        pending = self._pending.get(prediction.target_id)
        if pending is None:
            raise CausalPromotionError(
                f"no frozen promotion context exists for {prediction.target_id!r}"
            )
        target = _finite(getattr(label, "value", None), "label value")
        challenger_before = _component_state(self.challenger, "challenger")
        incumbent_before = _component_state(self.incumbent, "incumbent")
        gate_before = _clone_json(self.gate.state_dict(), "promotion gate state")
        pending_before = dict(self._pending)
        steps_before = tuple(self._steps)
        count_before = self._prediction_count
        labels_before = self._observed_label_count
        try:
            challenger_utility = self._utility(pending.challenger_prediction, target)
            incumbent_utility = self._utility(pending.incumbent_prediction, target)
            self.challenger.update(pending.features, target)
            self.incumbent.update(pending.features, target)
            self._observed_label_count += 1
            if (
                self.gate.status is GateStatus.OPEN
                and self._observed_label_count >= self.config.minimum_labels
            ):
                self.gate.observe_utilities(
                    self.config.challenger_id,
                    challenger_utility,
                    incumbent_utility,
                    eta=self.config.eta,
                    metadata={
                        "prediction_id": pending.prediction_id,
                        "target_id": pending.target_id,
                        "feature_digest": pending.feature_digest,
                        "feature_panel_digest": pending.feature_panel_digest,
                        "utility_spec_id": "bounded_absolute_error.v1",
                        "challenger_state_digest": pending.challenger_state_digest,
                        "incumbent_state_digest": pending.incumbent_state_digest,
                    },
                )
        except Exception:
            self._rollback(
                challenger_before,
                incumbent_before,
                gate_before,
                pending_before,
                count_before,
                steps_before,
                labels_before,
            )
            raise

    def on_resolution(
        self, prediction: ReplayPrediction, label: Any | None, status: ReplayStatus
    ) -> None:
        del label, status
        self._pending.pop(prediction.target_id, None)

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": CAUSAL_PROMOTION_SCHEMA,
            "version": CAUSAL_PROMOTION_VERSION,
            "promotion_id": self.config.promotion_id,
            "config_identity": self.config.identity,
            "plan_identity": self.plan.identity,
            "study_manifest_digest": self._study_manifest_digest,
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
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping):
            raise CausalPromotionError("causal promotion state must be a mapping")
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
        gate_observations = candidate_gate.challenger_state(
            self.config.challenger_id
        ).observations
        if observed_label_count < gate_observations:
            raise CausalPromotionError(
                "observed_label_count cannot be below promotion observations"
            )
        old_challenger = _component_state(self.challenger, "challenger")
        old_incumbent = _component_state(self.incumbent, "incumbent")
        old_gate = _clone_json(self.gate.state_dict(), "promotion gate state")
        try:
            _restore_component(self.challenger, challenger_state, "challenger")
            _restore_component(self.incumbent, incumbent_state, "incumbent")
            self.gate.load_state_dict(gate_state, eta=self.config.eta)
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
        self._steps = []


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
    "CausalPromotionConfig",
    "CausalPromotionError",
    "CausalPromotionModel",
    "CausalPromotionReplayResult",
    "CausalPromotionStep",
    "MAX_CAUSAL_PROMOTION_PENDING",
    "run_causal_promotion_replay",
]
