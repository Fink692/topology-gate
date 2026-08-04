"""Point-in-time evidence ledger for composed challenger promotion.

The lower-level e-process remains usable for trusted callers that already own
an evidence ledger.  This module provides the missing composition boundary:
predictions are frozen before labels arrive, labels resolve exactly once, the
arrival order is explicit, burn-in is visible, and the gate's predeclared eta
policy is used without accepting a post-label eta override.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Optional, cast

from .promotion import (
    AuditRecord,
    GateStatus,
    PromotionDecision,
    PromotionGate,
    PromotionStatus,
    validate_eta,
)

MAX_EVIDENCE_PENDING = 8_192
MAX_EVIDENCE_RECORDS = 200_000
EVIDENCE_SCHEMA_VERSION = 4
EVIDENCE_SCHEMA = "topology_gate.evidence_ledger"


def _text(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    if len(value) > 512:
        raise ValueError(f"{name} exceeds 512 characters")
    return value


def _step(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _finite(name: str, value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _optional_step(name: str, value: Any) -> Optional[int]:
    if value is None:
        return None
    return _step(name, value)


def _callable_identity(value: Any) -> str:
    if not callable(value):
        return "constant"
    return (
        f"{getattr(value, '__module__', type(value).__module__)}:"
        f"{getattr(value, '__qualname__', type(value).__qualname__)}"
    )


def _stable_digest(value: Any) -> str:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("evidence state must be JSON-safe") from exc
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class PromotionEvidenceConfig:
    """Immutable contract that pays for one challenger evidence family.

    ``certified=False`` is intentional for compatibility callers that only
    want diagnostic e-values.  A certified claim requires non-placeholder run,
    source, code, backend, dependency, score, eta, and missingness identities.
    ``missingness_predictable`` is an explicit declaration that terminal
    missingness is determined without the unobserved label; it is a contract
    boundary, not an empirical proof of that assumption.
    Changing any field changes :attr:`identity` and therefore starts a new
    evidence family rather than mutating an old one.
    """

    run_id: str
    family_id: str
    incumbent_id: str
    score_spec_id: str = "bounded_absolute_error.v1"
    null_hypothesis_id: str = "conditional_mean_nonpositive.v1"
    eta_policy_id: str = "gate-default"
    missing_label_policy_id: str = "diagnostic-missing.v1"
    allocation_rule_id: str = "geometric_alpha.v1"
    package_version: str = "unbound"
    config_fingerprint: str = "unbound"
    backend_identity: str = "unbound"
    dependency_fingerprint: str = "unbound"
    manifest_digest: str = "unbound"
    global_alpha: float = 0.05
    initial_wealth: float = 1.0
    score_bound: float = 1.0
    certified: bool = False
    missingness_predictable: bool = False

    def __post_init__(self) -> None:
        for name in (
            "run_id",
            "family_id",
            "incumbent_id",
            "score_spec_id",
            "null_hypothesis_id",
            "eta_policy_id",
            "missing_label_policy_id",
            "allocation_rule_id",
            "package_version",
            "config_fingerprint",
            "backend_identity",
            "dependency_fingerprint",
            "manifest_digest",
        ):
            object.__setattr__(self, name, _text(name, getattr(self, name)))
        alpha = _finite("global_alpha", self.global_alpha)
        wealth = _finite("initial_wealth", self.initial_wealth)
        bound = _finite("score_bound", self.score_bound)
        if not 0.0 < alpha < 1.0:
            raise ValueError("global_alpha must be in (0, 1)")
        if wealth <= 0.0:
            raise ValueError("initial_wealth must be greater than zero")
        if bound <= 0.0:
            raise ValueError("score_bound must be greater than zero")
        if not isinstance(self.certified, bool):
            raise ValueError("certified must be boolean")
        if not isinstance(self.missingness_predictable, bool):
            raise ValueError("missingness_predictable must be boolean")
        if self.certified:
            non_placeholder = (
                self.run_id,
                self.family_id,
                self.incumbent_id,
                self.score_spec_id,
                self.null_hypothesis_id,
                self.eta_policy_id,
                self.missing_label_policy_id,
                self.allocation_rule_id,
                self.package_version,
                self.config_fingerprint,
                self.backend_identity,
                self.dependency_fingerprint,
                self.manifest_digest,
            )
            if any(value in {"unbound", "gate-default", "diagnostic-missing.v1"} for value in non_placeholder):
                raise ValueError("certified evidence requires non-placeholder identities")
            if not self.missingness_predictable:
                raise ValueError(
                    "certified evidence requires an explicit predictable-missingness declaration"
                )
        object.__setattr__(self, "global_alpha", alpha)
        object.__setattr__(self, "initial_wealth", wealth)
        object.__setattr__(self, "score_bound", bound)

    @property
    def identity(self) -> str:
        return _stable_digest(self.state_dict(include_identity=False))

    @property
    def promotion_claim(self) -> str:
        return "certified" if self.certified else "diagnostic"

    def state_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "version": 2,
            "schema": "topology_gate.promotion_evidence_config",
            "run_id": self.run_id,
            "family_id": self.family_id,
            "incumbent_id": self.incumbent_id,
            "score_spec_id": self.score_spec_id,
            "null_hypothesis_id": self.null_hypothesis_id,
            "eta_policy_id": self.eta_policy_id,
            "missing_label_policy_id": self.missing_label_policy_id,
            "allocation_rule_id": self.allocation_rule_id,
            "package_version": self.package_version,
            "config_fingerprint": self.config_fingerprint,
            "backend_identity": self.backend_identity,
            "dependency_fingerprint": self.dependency_fingerprint,
            "manifest_digest": self.manifest_digest,
            "global_alpha": self.global_alpha,
            "initial_wealth": self.initial_wealth,
            "score_bound": self.score_bound,
            "certified": self.certified,
            "missingness_predictable": self.missingness_predictable,
        }
        if include_identity:
            result["identity"] = self.identity
        return result

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "PromotionEvidenceConfig":
        if not isinstance(state, Mapping):
            raise ValueError("evidence config must be a mapping")
        if state.get("version") not in {1, 2} or state.get("schema") != "topology_gate.promotion_evidence_config":
            raise ValueError("unsupported evidence config version or schema")
        candidate = cls(
            run_id=cast(str, state.get("run_id")),
            family_id=cast(str, state.get("family_id")),
            incumbent_id=cast(str, state.get("incumbent_id")),
            score_spec_id=cast(str, state.get("score_spec_id", "bounded_absolute_error.v1")),
            null_hypothesis_id=cast(str, state.get("null_hypothesis_id", "conditional_mean_nonpositive.v1")),
            eta_policy_id=cast(str, state.get("eta_policy_id", "gate-default")),
            missing_label_policy_id=cast(str, state.get("missing_label_policy_id", "diagnostic-missing.v1")),
            allocation_rule_id=cast(str, state.get("allocation_rule_id", "geometric_alpha.v1")),
            package_version=cast(str, state.get("package_version", "unbound")),
            config_fingerprint=cast(str, state.get("config_fingerprint", "unbound")),
            backend_identity=cast(str, state.get("backend_identity", "unbound")),
            dependency_fingerprint=cast(str, state.get("dependency_fingerprint", "unbound")),
            manifest_digest=cast(str, state.get("manifest_digest", "unbound")),
            global_alpha=cast(float, state.get("global_alpha", 0.05)),
            initial_wealth=cast(float, state.get("initial_wealth", 1.0)),
            score_bound=cast(float, state.get("score_bound", 1.0)),
            certified=state.get("certified", False),
            missingness_predictable=state.get("missingness_predictable", False),
        )
        identity = state.get("identity")
        if identity is not None and identity != candidate.identity:
            raise ValueError("evidence config identity mismatch")
        return candidate


def _metadata(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping) or len(value) > 256:
        raise ValueError("evidence metadata must be a mapping with at most 256 items")
    # PromotionGate performs the recursive secret/non-finite sanitization.  A
    # shallow JSON check here prevents arbitrary objects entering the ledger.
    try:
        encoded = json.dumps(value, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("evidence metadata must be JSON-safe") from exc
    if len(encoded.encode("utf-8")) > 32 * 1024:
        raise ValueError("evidence metadata exceeds 32 KiB")
    return dict(value)


def _bounded_absolute_error_utilities(
    prediction: "FrozenPrediction", label: float
) -> tuple[float, float]:
    """Return the declared bounded utilities for the built-in score spec."""

    target = _finite("label_value", label)
    challenger_error = abs(prediction.challenger_prediction - target)
    incumbent_error = abs(prediction.incumbent_prediction - target)
    # A bounded utility is required before the gate sees the pair.  Capping
    # each absolute error also keeps the raw utility values finite when two
    # otherwise finite IEEE-754 inputs are very far apart.
    return -min(challenger_error, 1.0), -min(incumbent_error, 1.0)


@dataclass(frozen=True, slots=True)
class FrozenPrediction:
    """Prediction pair committed before the associated label is available."""

    prediction_id: str
    challenger_id: str
    decision_step: int
    label_available_step: int
    challenger_prediction: float
    incumbent_prediction: float
    model_fingerprint: str
    feature_fingerprint: str
    gate_epoch: int
    family_id: str = "unbound"
    incumbent_id: str = "incumbent"
    target_id: str = ""
    target_event_step: Optional[int] = None
    eta: Optional[float] = None
    eta_policy_id: str = "gate-default"
    score_spec_id: str = "bounded_absolute_error.v1"
    allocation_id: str = ""
    prior_state_fingerprint: str = ""

    def __post_init__(self) -> None:
        prediction_id = _text("prediction_id", self.prediction_id)
        challenger_id = _text("challenger_id", self.challenger_id)
        decision_step = _step("decision_step", self.decision_step)
        available = _step("label_available_step", self.label_available_step)
        if available <= decision_step:
            raise ValueError("label_available_step must be after decision_step")
        challenger = _finite("challenger_prediction", self.challenger_prediction)
        incumbent = _finite("incumbent_prediction", self.incumbent_prediction)
        model = _text("model_fingerprint", self.model_fingerprint)
        features = _text("feature_fingerprint", self.feature_fingerprint)
        epoch = _step("gate_epoch", self.gate_epoch)
        family = _text("family_id", self.family_id)
        incumbent_id = _text("incumbent_id", self.incumbent_id)
        target_id = self.prediction_id if self.target_id == "" else _text("target_id", self.target_id)
        target_event = self.decision_step if self.target_event_step is None else _step(
            "target_event_step", self.target_event_step
        )
        if target_event >= available:
            raise ValueError("target_event_step must precede label_available_step")
        eta = None if self.eta is None else validate_eta(self.eta)
        eta_policy = _text("eta_policy_id", self.eta_policy_id)
        score_spec = _text("score_spec_id", self.score_spec_id)
        allocation = self.allocation_id
        if allocation == "":
            allocation = f"{challenger_id}:epoch:{epoch}"
        allocation = _text("allocation_id", allocation)
        prior_fingerprint = self.prior_state_fingerprint
        if prior_fingerprint:
            prior_fingerprint = _text("prior_state_fingerprint", prior_fingerprint)
        object.__setattr__(self, "prediction_id", prediction_id)
        object.__setattr__(self, "challenger_id", challenger_id)
        object.__setattr__(self, "decision_step", decision_step)
        object.__setattr__(self, "label_available_step", available)
        object.__setattr__(self, "challenger_prediction", challenger)
        object.__setattr__(self, "incumbent_prediction", incumbent)
        object.__setattr__(self, "model_fingerprint", model)
        object.__setattr__(self, "feature_fingerprint", features)
        object.__setattr__(self, "gate_epoch", epoch)
        object.__setattr__(self, "family_id", family)
        object.__setattr__(self, "incumbent_id", incumbent_id)
        object.__setattr__(self, "target_id", target_id)
        object.__setattr__(self, "target_event_step", target_event)
        object.__setattr__(self, "eta", eta)
        object.__setattr__(self, "eta_policy_id", eta_policy)
        object.__setattr__(self, "score_spec_id", score_spec)
        object.__setattr__(self, "allocation_id", allocation)
        object.__setattr__(self, "prior_state_fingerprint", prior_fingerprint)

    def state_dict(self) -> dict[str, Any]:
        return {
            "prediction_id": self.prediction_id,
            "challenger_id": self.challenger_id,
            "decision_step": self.decision_step,
            "label_available_step": self.label_available_step,
            "challenger_prediction": self.challenger_prediction,
            "incumbent_prediction": self.incumbent_prediction,
            "model_fingerprint": self.model_fingerprint,
            "feature_fingerprint": self.feature_fingerprint,
            "gate_epoch": self.gate_epoch,
            "family_id": self.family_id,
            "incumbent_id": self.incumbent_id,
            "target_id": self.target_id,
            "target_event_step": self.target_event_step,
            "eta": self.eta,
            "eta_policy_id": self.eta_policy_id,
            "score_spec_id": self.score_spec_id,
            "allocation_id": self.allocation_id,
            "prior_state_fingerprint": self.prior_state_fingerprint,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "FrozenPrediction":
        if not isinstance(state, Mapping):
            raise ValueError("frozen prediction must be a mapping")
        return cls(
            prediction_id=cast(str, state.get("prediction_id")),
            challenger_id=cast(str, state.get("challenger_id")),
            decision_step=cast(int, state.get("decision_step")),
            label_available_step=cast(int, state.get("label_available_step")),
            challenger_prediction=cast(float, state.get("challenger_prediction")),
            incumbent_prediction=cast(float, state.get("incumbent_prediction")),
            model_fingerprint=cast(str, state.get("model_fingerprint")),
            feature_fingerprint=cast(str, state.get("feature_fingerprint")),
            gate_epoch=cast(int, state.get("gate_epoch")),
            family_id=cast(str, state.get("family_id", "unbound")),
            incumbent_id=cast(str, state.get("incumbent_id", "incumbent")),
            target_id=cast(str, state.get("target_id", "")),
            target_event_step=cast(Optional[int], state.get("target_event_step")),
            eta=cast(Optional[float], state.get("eta")),
            eta_policy_id=cast(str, state.get("eta_policy_id", "gate-default")),
            score_spec_id=cast(str, state.get("score_spec_id", "bounded_absolute_error.v1")),
            allocation_id=cast(str, state.get("allocation_id", "")),
            prior_state_fingerprint=cast(str, state.get("prior_state_fingerprint", "")),
        )


@dataclass(frozen=True, slots=True)
class LabelReceipt:
    """Point-in-time label receipt buffered before evidence settlement."""

    label_id: str
    prediction_id: str
    target_id: str
    label_available_step: int
    received_step: int
    challenger_utility: Optional[float] = None
    incumbent_utility: Optional[float] = None
    status: str = "observed"
    source_id: str = "unbound"
    source_revision: str = "unbound"
    metadata: Mapping[str, Any] | None = None
    label_value: Optional[float] = None

    def __post_init__(self) -> None:
        label_id = _text("label_id", self.label_id)
        prediction_id = _text("prediction_id", self.prediction_id)
        target_id = _text("target_id", self.target_id)
        available = _step("label_available_step", self.label_available_step)
        received = _step("received_step", self.received_step)
        if received < available:
            raise ValueError("received_step cannot precede label availability")
        if self.status not in {"observed", "missing", "expired"}:
            raise ValueError("unsupported label status")
        challenger: Optional[float]
        incumbent: Optional[float]
        label_value: Optional[float]
        if self.status == "observed":
            if self.label_value is not None:
                if self.challenger_utility is not None or self.incumbent_utility is not None:
                    raise ValueError(
                        "raw labels cannot carry caller-supplied utilities"
                    )
                label_value = _finite("label_value", self.label_value)
                challenger = None
                incumbent = None
            else:
                if self.challenger_utility is None or self.incumbent_utility is None:
                    raise ValueError(
                        "observed labels require a raw value or both utilities"
                    )
                label_value = None
                challenger = _finite("challenger_utility", self.challenger_utility)
                incumbent = _finite("incumbent_utility", self.incumbent_utility)
        else:
            if (
                self.label_value is not None
                or self.challenger_utility is not None
                or self.incumbent_utility is not None
            ):
                raise ValueError("missing labels cannot carry values or utilities")
            label_value = None
            challenger = None
            incumbent = None
        object.__setattr__(self, "label_id", label_id)
        object.__setattr__(self, "prediction_id", prediction_id)
        object.__setattr__(self, "target_id", target_id)
        object.__setattr__(self, "label_available_step", available)
        object.__setattr__(self, "received_step", received)
        object.__setattr__(self, "label_value", label_value)
        object.__setattr__(self, "challenger_utility", challenger)
        object.__setattr__(self, "incumbent_utility", incumbent)
        object.__setattr__(self, "source_id", _text("source_id", self.source_id))
        object.__setattr__(self, "source_revision", _text("source_revision", self.source_revision))
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    def state_dict(self) -> dict[str, Any]:
        return {
            "label_id": self.label_id,
            "prediction_id": self.prediction_id,
            "target_id": self.target_id,
            "label_available_step": self.label_available_step,
            "received_step": self.received_step,
            "label_value": self.label_value,
            "challenger_utility": self.challenger_utility,
            "incumbent_utility": self.incumbent_utility,
            "status": self.status,
            "source_id": self.source_id,
            "source_revision": self.source_revision,
            "metadata": dict(self.metadata or {}),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "LabelReceipt":
        if not isinstance(state, Mapping):
            raise ValueError("label receipt must be a mapping")
        return cls(
            label_id=cast(str, state.get("label_id")),
            prediction_id=cast(str, state.get("prediction_id")),
            target_id=cast(str, state.get("target_id")),
            label_available_step=cast(int, state.get("label_available_step")),
            received_step=cast(int, state.get("received_step")),
            label_value=cast(Optional[float], state.get("label_value")),
            challenger_utility=cast(Optional[float], state.get("challenger_utility")),
            incumbent_utility=cast(Optional[float], state.get("incumbent_utility")),
            status=cast(str, state.get("status", "observed")),
            source_id=cast(str, state.get("source_id", "unbound")),
            source_revision=cast(str, state.get("source_revision", "unbound")),
            metadata=state.get("metadata"),
        )


@dataclass(frozen=True, slots=True)
class EvidenceResolution:
    """Immutable terminal state for one prediction receipt."""

    prediction_id: str
    label_id: str
    label_available_step: int
    accepted: bool
    burn_in: bool
    reason: str
    decision: Optional[PromotionDecision] = None
    status: str = "settled"
    evidence_index: int = 0
    settlement_step: Optional[int] = None
    promotion_effective_step: Optional[int] = None

    def __post_init__(self) -> None:
        if self.status not in {"settled", "burn_in", "missing", "expired"}:
            raise ValueError("unsupported evidence resolution status")
        if not isinstance(self.accepted, bool) or not isinstance(self.burn_in, bool):
            raise ValueError("evidence resolution flags must be boolean")
        _text("prediction_id", self.prediction_id)
        _text("label_id", self.label_id)
        _step("label_available_step", self.label_available_step)
        _step("evidence_index", self.evidence_index)
        _text("reason", self.reason)
        if self.settlement_step is not None:
            _step("settlement_step", self.settlement_step)
        if self.promotion_effective_step is not None:
            _step("promotion_effective_step", self.promotion_effective_step)
        if self.status in {"missing", "expired"}:
            if self.accepted or self.burn_in or self.decision is not None:
                raise ValueError("missing or expired evidence cannot carry a decision")
        elif self.status == "burn_in":
            if not self.accepted or not self.burn_in or self.decision is not None:
                raise ValueError("burn-in evidence has an invalid decision state")
        elif not self.accepted or self.burn_in or self.decision is None:
            raise ValueError("settled evidence must carry an accepted decision")
        if self.decision is not None:
            if self.settlement_step is None:
                raise ValueError("decision evidence requires settlement_step")
            if self.decision.promoted:
                if self.promotion_effective_step != self.settlement_step + 1:
                    raise ValueError("promoted evidence has an invalid activation step")
            elif self.promotion_effective_step is not None:
                raise ValueError("unpromoted evidence cannot carry activation")

    @property
    def promoted(self) -> bool:
        return bool(self.decision is not None and self.decision.promoted)

    def to_dict(self) -> dict[str, Any]:
        decision = None
        if self.decision is not None:
            decision = {
                "challenger_id": self.decision.challenger_id,
                "epoch": self.decision.epoch,
                "observation": self.decision.observation,
                "score": self.decision.score,
                "eta": self.decision.eta,
                "factor": self.decision.factor,
                "e_value": self.decision.e_value,
                "alpha": self.decision.alpha,
                "threshold": self.decision.threshold,
                "threshold_crossed": self.decision.threshold_crossed,
                "promoted": self.decision.promoted,
                "state": self.decision.state.value,
                "audit_record": self.decision.audit_record.to_dict(),
            }
        return {
            "prediction_id": self.prediction_id,
            "label_id": self.label_id,
            "label_available_step": self.label_available_step,
            "accepted": self.accepted,
            "burn_in": self.burn_in,
            "reason": self.reason,
            "status": self.status,
            "evidence_index": self.evidence_index,
            "settlement_step": self.settlement_step,
            "promotion_effective_step": self.promotion_effective_step,
            "promoted": self.promoted,
            "decision": decision,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "EvidenceResolution":
        if not isinstance(state, Mapping):
            raise ValueError("evidence resolution must be a mapping")
        decision_raw = state.get("decision")
        decision: Optional[PromotionDecision] = None
        if decision_raw is not None:
            if not isinstance(decision_raw, Mapping):
                raise ValueError("evidence decision must be a mapping")
            audit_raw = decision_raw.get("audit_record")
            if not isinstance(audit_raw, Mapping):
                raise ValueError("evidence decision is missing its audit record")
            try:
                promotion_state = PromotionStatus(str(decision_raw.get("state")))
            except ValueError as exc:
                raise ValueError("evidence decision has an invalid state") from exc
            decision = PromotionDecision(
                challenger_id=_text("decision.challenger_id", decision_raw.get("challenger_id")),
                epoch=_step("decision.epoch", decision_raw.get("epoch")),
                observation=_step("decision.observation", decision_raw.get("observation")),
                score=_finite("decision.score", decision_raw.get("score")),
                eta=validate_eta(cast(float, decision_raw.get("eta"))),
                factor=_finite("decision.factor", decision_raw.get("factor")),
                e_value=_finite("decision.e_value", decision_raw.get("e_value")),
                alpha=_finite("decision.alpha", decision_raw.get("alpha")),
                threshold=_finite("decision.threshold", decision_raw.get("threshold")),
                threshold_crossed=bool(decision_raw.get("threshold_crossed", False)),
                promoted=bool(decision_raw.get("promoted", False)),
                state=promotion_state,
                audit_record=AuditRecord.from_dict(audit_raw),
            )
        return cls(
            prediction_id=cast(str, state.get("prediction_id")),
            label_id=cast(str, state.get("label_id")),
            label_available_step=cast(int, state.get("label_available_step")),
            accepted=state.get("accepted", False),
            burn_in=state.get("burn_in", False),
            reason=cast(str, state.get("reason")),
            decision=decision,
            status=cast(str, state.get("status", "settled")),
            evidence_index=state.get("evidence_index", 0),
            settlement_step=cast(Optional[int], state.get("settlement_step")),
            promotion_effective_step=cast(
                Optional[int], state.get("promotion_effective_step")
            ),
        )


class EvidenceLedger:
    """Frozen point-in-time comparison ledger feeding one ``PromotionGate`` slot.

    Prediction receipts are committed in decision order. Labels may be
    ingested out of order, but the e-process is advanced only by
    :meth:`settle_ready` in prediction order at or after each source
    availability boundary. The optional ``eta_policy`` is evaluated during
    prediction freezing, before a label can be read; settlement has no public
    eta override. Certified ledgers also require a sealed challenger family
    and derive utilities from a raw label and the frozen prediction through
    the declared score specification. Caller-supplied utility pairs remain a
    diagnostic-only compatibility path.
    """

    def __init__(
        self,
        gate: PromotionGate,
        challenger_id: str,
        *,
        burn_in: int = 0,
        max_pending: int = MAX_EVIDENCE_PENDING,
        max_records: int = MAX_EVIDENCE_RECORDS,
        config: PromotionEvidenceConfig | None = None,
        eta_policy: Callable[[Mapping[str, Any]], Any] | float | None = None,
        score_spec: Callable[[FrozenPrediction, float], Any] | None = None,
        _restoring: bool = False,
    ) -> None:
        if not isinstance(gate, PromotionGate):
            raise ValueError("gate must be a PromotionGate")
        challenger_id = _text("challenger_id", challenger_id)
        if challenger_id not in gate.challenger_ids:
            raise ValueError("challenger must be registered with the gate first")
        burn_in = _step("burn_in", burn_in)
        max_pending = _step("max_pending", max_pending)
        max_records = _step("max_records", max_records)
        if max_pending < 1 or max_pending > MAX_EVIDENCE_PENDING:
            raise ValueError("max_pending is outside the configured limit")
        if max_records < 1 or max_records > MAX_EVIDENCE_RECORDS:
            raise ValueError("max_records is outside the configured limit")
        gate_state = gate.state_dict()
        if config is None:
            config = PromotionEvidenceConfig(
                run_id="unbound",
                family_id="unbound",
                incumbent_id=gate.incumbent_id,
                global_alpha=float(gate_state["global_alpha"]),
                initial_wealth=float(gate_state["initial_wealth"]),
                score_bound=float(gate_state["score_bound"]),
                certified=False,
            )
        if config.incumbent_id != gate.incumbent_id:
            raise ValueError("evidence config incumbent does not match the gate")
        if not math.isclose(config.global_alpha, float(gate_state["global_alpha"]), rel_tol=0.0, abs_tol=1e-15):
            raise ValueError("evidence config alpha does not match the gate")
        if not math.isclose(config.initial_wealth, float(gate_state["initial_wealth"]), rel_tol=0.0, abs_tol=1e-15):
            raise ValueError("evidence config initial wealth does not match the gate")
        if not math.isclose(config.score_bound, float(gate_state["score_bound"]), rel_tol=0.0, abs_tol=1e-15):
            raise ValueError("evidence config score bound does not match the gate")
        if config.certified and eta_policy is None:
            raise ValueError("certified evidence requires an explicit eta_policy")
        if config.certified and not gate.registration_sealed:
            raise ValueError(
                "certified evidence requires a sealed challenger registration"
            )
        if config.certified and not _restoring:
            if gate.status is not GateStatus.OPEN or gate.epoch != 0:
                raise ValueError("certified evidence requires a fresh gate epoch")
            if any(
                state.observations != 0
                for state in gate.snapshots()
            ):
                raise ValueError(
                    "certified evidence cannot adopt prior direct gate evidence"
                )
            if any(
                record.event not in {"register", "registration_sealed"}
                for record in gate.audit_records
            ):
                raise ValueError(
                    "certified evidence cannot adopt prior direct gate evidence"
                )
        if eta_policy is not None and not callable(eta_policy):
            validate_eta(eta_policy)
        if score_spec is not None and not callable(score_spec):
            raise ValueError("score_spec must be callable")
        self._gate = gate
        self._challenger_id = challenger_id
        self._burn_in = burn_in
        self._max_pending = max_pending
        self._max_records = max_records
        self._config = config
        self._eta_policy = eta_policy
        self._eta_policy_identity = _callable_identity(eta_policy)
        self._score_spec = score_spec or _bounded_absolute_error_utilities
        self._score_spec_identity = (
            _callable_identity(score_spec)
            if score_spec is not None
            else "builtin:bounded_absolute_error.v1"
        )
        if config.certified and score_spec is None and config.score_spec_id != "bounded_absolute_error.v1":
            raise ValueError(
                "custom score_spec is required for the declared score_spec_id"
            )
        if config.certified and score_spec is not None and config.score_spec_id == "bounded_absolute_error.v1":
            raise ValueError(
                "custom score_spec requires an explicit score_spec_id"
            )
        self._initial_gate_epoch = gate.epoch
        self._allocation_id = self._allocation_identity(gate_state, challenger_id)
        self._expected_gate_evidence_digest = self._gate_evidence_digest(gate_state)
        self._pending: dict[str, FrozenPrediction] = {}
        self._prediction_history: dict[str, FrozenPrediction] = {}
        self._labels: dict[str, LabelReceipt] = {}
        self._resolved_prediction_ids: set[str] = set()
        self._resolved_label_ids: set[str] = set()
        self._resolutions: list[EvidenceResolution] = []
        self._last_decision_step: Optional[int] = None
        self._last_label_available_step: Optional[int] = None
        self._last_settlement_step: Optional[int] = None
        self._last_received_step: Optional[int] = None
        self._evidence_index = 0
        self._observed_label_count = 0

    @staticmethod
    def _gate_evidence_digest(state: Mapping[str, Any]) -> str:
        """Fingerprint all mutable gate evidence, including its audit log."""

        return _stable_digest(state)

    def _current_gate_evidence_digest(self) -> str:
        return self._gate_evidence_digest(self._gate.state_dict())

    def _assert_gate_binding(self) -> None:
        """Reject gate mutations that did not pass through this ledger."""

        current = self._current_gate_evidence_digest()
        if current == self._expected_gate_evidence_digest:
            return
        if self._gate.epoch != self._initial_gate_epoch:
            raise ValueError("promotion gate epoch changed outside the evidence ledger")
        raise ValueError("promotion gate evidence changed outside the evidence ledger")

    def _refresh_gate_binding(self) -> None:
        self._expected_gate_evidence_digest = self._current_gate_evidence_digest()

    def _rollback_gate(self, internal_state: Mapping[str, Any], digest: str) -> None:
        """Restore the live gate after a failed settlement transaction.

        The gate contains executable eta policies, so restoring from its JSON
        state can require caller-supplied callables and is not guaranteed to
        reconstruct per-challenger policies.  A detached in-memory snapshot is
        therefore used for the local transaction rollback; authenticated
        checkpoints still use the canonical ``state_dict`` representation.
        """

        self._gate.__dict__.clear()
        self._gate.__dict__.update(copy.deepcopy(dict(internal_state)))
        self._expected_gate_evidence_digest = digest

    @property
    def config(self) -> PromotionEvidenceConfig:
        return self._config

    @property
    def certified(self) -> bool:
        return self._config.certified and self._eta_policy is not None

    @property
    def promotion_claim(self) -> str:
        return "certified" if self.certified else "diagnostic"

    @property
    def allocation_id(self) -> str:
        return self._allocation_id

    @property
    def pending_labels(self) -> tuple[str, ...]:
        return tuple(self._labels)

    @staticmethod
    def _allocation_identity(gate_state: Mapping[str, Any], challenger_id: str | None = None) -> str:
        challengers = gate_state.get("challengers", ())
        if not isinstance(challengers, (list, tuple)):
            raise ValueError("promotion gate challenger state is invalid")
        selected: Mapping[str, Any] | None = None
        for entry in challengers:
            if not isinstance(entry, Mapping):
                continue
            state = entry.get("state")
            if not isinstance(state, Mapping):
                continue
            process = state.get("process")
            if not isinstance(process, Mapping):
                continue
            candidate_id = state.get("challenger_id")
            if challenger_id is None or candidate_id == challenger_id:
                selected = {"index": entry.get("index"), "challenger_id": candidate_id}
                if challenger_id is not None:
                    break
        if selected is None:
            raise ValueError("challenger allocation is missing from gate state")
        return _stable_digest(
            {
                "rule": "geometric_alpha.v1",
                "global_alpha": gate_state.get("global_alpha"),
                "epoch": gate_state.get("epoch"),
                **selected,
            }
        )

    def _allocation_identity_for_self(self, gate_state: Mapping[str, Any]) -> str:
        return self._allocation_identity(gate_state, self._challenger_id)

    @property
    def challenger_id(self) -> str:
        return self._challenger_id

    @property
    def burn_in(self) -> int:
        return self._burn_in

    @property
    def pending_prediction_ids(self) -> tuple[str, ...]:
        return tuple(self._pending)

    @property
    def resolved_count(self) -> int:
        return len(self._resolved_prediction_ids)

    @property
    def observed_count(self) -> int:
        """Number of finite observed labels, excluding missing terminal states."""

        return self._observed_label_count

    @property
    def resolutions(self) -> tuple[EvidenceResolution, ...]:
        return tuple(self._resolutions)

    @property
    def evidence_identity(self) -> str:
        payload = {
            "config_identity": self._config.identity,
            "challenger_id": self._challenger_id,
            "burn_in": self._burn_in,
            "max_pending": self._max_pending,
            "max_records": self._max_records,
            "gate_epoch": self._initial_gate_epoch,
            "allocation_id": self._allocation_id,
            "score_spec_id": self._config.score_spec_id,
            "score_spec_identity": self._score_spec_identity,
        }
        return _stable_digest(payload)

    def _prior_state_fingerprint(self) -> str:
        state = self._gate.challenger_state(self._challenger_id)
        return _stable_digest(
            {
                "challenger_id": state.challenger_id,
                "epoch": state.epoch,
                "e_value": state.e_value,
                "observations": state.observations,
                "score_history": list(state.score_history),
                "state": state.state.value,
            }
        )

    def _resolve_eta(
        self,
        *,
        prediction_id: str,
        decision_step: int,
        model_fingerprint: str,
        feature_fingerprint: str,
        prior_state_fingerprint: str,
    ) -> Optional[float]:
        if self._eta_policy is None:
            return None
        if callable(self._eta_policy):
            state = self._gate.challenger_state(self._challenger_id)
            context: Mapping[str, Any] = {
                "prediction_id": prediction_id,
                "challenger_id": self._challenger_id,
                "incumbent_id": self._config.incumbent_id,
                "decision_step": decision_step,
                "model_fingerprint": model_fingerprint,
                "feature_fingerprint": feature_fingerprint,
                "gate_epoch": state.epoch,
                "prior_observations": state.observations,
                "prior_e_value": state.e_value,
                "prior_score_history": tuple(state.score_history),
                "prior_state_fingerprint": prior_state_fingerprint,
            }
            return validate_eta(self._eta_policy(context))
        return validate_eta(self._eta_policy)

    def record_prediction(
        self,
        *,
        prediction_id: str,
        decision_step: int,
        label_available_step: int,
        challenger_prediction: float,
        incumbent_prediction: float,
        model_fingerprint: str,
        feature_fingerprint: str,
        target_id: str = "",
        target_event_step: int | None = None,
    ) -> FrozenPrediction:
        """Commit a frozen paired prediction before label resolution."""

        self._assert_gate_binding()
        if self._gate.epoch != self._initial_gate_epoch:
            raise ValueError("gate epoch changed; start a new evidence family")
        if self._gate.promoted_challenger_id is not None:
            raise ValueError("cannot record predictions after challenger promotion")
        prediction_id = _text("prediction_id", prediction_id)
        if prediction_id in self._prediction_history:
            raise ValueError("prediction_id has already been recorded")
        if len(self._pending) >= self._max_pending:
            raise ValueError("pending prediction ledger exceeds max_pending")
        decision_value = _step("decision_step", decision_step)
        prior_fingerprint = self._prior_state_fingerprint()
        eta = self._resolve_eta(
            prediction_id=prediction_id,
            decision_step=decision_value,
            model_fingerprint=model_fingerprint,
            feature_fingerprint=feature_fingerprint,
            prior_state_fingerprint=prior_fingerprint,
        )
        prediction = FrozenPrediction(
            prediction_id=prediction_id,
            challenger_id=self._challenger_id,
            decision_step=decision_value,
            label_available_step=label_available_step,
            challenger_prediction=challenger_prediction,
            incumbent_prediction=incumbent_prediction,
            model_fingerprint=model_fingerprint,
            feature_fingerprint=feature_fingerprint,
            gate_epoch=self._gate.epoch,
            family_id=self._config.family_id,
            incumbent_id=self._config.incumbent_id,
            target_id=target_id,
            target_event_step=target_event_step,
            eta=eta,
            eta_policy_id=self._config.eta_policy_id,
            score_spec_id=self._config.score_spec_id,
            allocation_id=self._allocation_id,
            prior_state_fingerprint=prior_fingerprint,
        )
        if self._last_decision_step is not None and prediction.decision_step <= self._last_decision_step:
            raise ValueError("decision_step must be strictly increasing")
        self._pending[prediction.prediction_id] = prediction
        self._prediction_history[prediction.prediction_id] = prediction
        self._last_decision_step = prediction.decision_step
        return prediction

    def ingest_label(
        self,
        *,
        prediction_id: str,
        label_id: str,
        label_available_step: int,
        received_step: int | None = None,
        label_value: float | None = None,
        challenger_utility: float | None = None,
        incumbent_utility: float | None = None,
        status: str = "observed",
        source_id: str = "unbound",
        source_revision: str = "unbound",
        metadata: Mapping[str, Any] | None = None,
    ) -> LabelReceipt:
        """Record a label arrival without advancing the e-process."""

        self._assert_gate_binding()
        prediction_id = _text("prediction_id", prediction_id)
        label_id = _text("label_id", label_id)
        if prediction_id in self._resolved_prediction_ids:
            raise ValueError("prediction_id has already been resolved")
        if label_id in self._resolved_label_ids or any(
            value.label_id == label_id for value in self._labels.values()
        ):
            raise ValueError("label_id has already been recorded")
        prediction = self._prediction_history.get(prediction_id)
        if prediction is None:
            raise KeyError("unknown prediction_id")
        available = _step("label_available_step", label_available_step)
        if available < prediction.label_available_step:
            raise ValueError("label arrived before its declared availability step")
        if available <= prediction.decision_step:
            raise ValueError("label availability must follow the decision step")
        received = available if received_step is None else _step("received_step", received_step)
        if self._last_received_step is not None and received < self._last_received_step:
            raise ValueError("received labels must have non-decreasing arrival steps")
        target_id = prediction.target_id
        safe_metadata = _metadata(metadata)
        if self.certified and status == "observed":
            if label_value is None:
                raise ValueError(
                    "certified evidence requires a raw label for the declared score spec"
                )
            if challenger_utility is not None or incumbent_utility is not None:
                raise ValueError(
                    "certified evidence derives utilities from the frozen prediction"
                )
        receipt = LabelReceipt(
            label_id=label_id,
            prediction_id=prediction_id,
            target_id=target_id,
            label_available_step=available,
            received_step=received,
            label_value=label_value,
            challenger_utility=challenger_utility,
            incumbent_utility=incumbent_utility,
            status=status,
            source_id=source_id,
            source_revision=source_revision,
            metadata=safe_metadata,
        )
        self._labels[prediction_id] = receipt
        self._last_received_step = received
        return receipt

    def _settle_receipt(
        self,
        receipt: LabelReceipt,
        *,
        settlement_step: int,
        missing_reason: str | None = None,
    ) -> EvidenceResolution:
        if len(self._resolutions) >= self._max_records:
            raise ValueError("evidence resolution history exceeds max_records")
        prediction = self._pending.get(receipt.prediction_id)
        if prediction is None:
            raise ValueError("label has no unresolved prediction")
        evidence_index = self._evidence_index
        if receipt.status in {"missing", "expired"}:
            resolution = EvidenceResolution(
                prediction_id=receipt.prediction_id,
                label_id=receipt.label_id,
                label_available_step=receipt.label_available_step,
                accepted=False,
                burn_in=False,
                reason=missing_reason or f"label terminalized as {receipt.status}",
                decision=None,
                status=receipt.status,
                evidence_index=evidence_index,
                settlement_step=settlement_step,
            )
        elif self._observed_label_count < self._burn_in:
            resolution = EvidenceResolution(
                prediction_id=receipt.prediction_id,
                label_id=receipt.label_id,
                label_available_step=receipt.label_available_step,
                accepted=True,
                burn_in=True,
                reason="burn-in evidence retained; not submitted to e-process",
                status="burn_in",
                evidence_index=evidence_index,
                settlement_step=settlement_step,
            )
        else:
            if receipt.label_value is not None:
                try:
                    utilities = self._score_spec(prediction, receipt.label_value)
                except Exception as exc:
                    raise ValueError("declared score_spec failed on the label") from exc
                if isinstance(utilities, (str, bytes, bytearray)):
                    raise ValueError("declared score_spec must return two utilities")
                try:
                    challenger_utility, incumbent_utility = tuple(utilities)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "declared score_spec must return two utilities"
                    ) from exc
                if not isinstance(utilities, (tuple, list)) or len(utilities) != 2:
                    raise ValueError("declared score_spec must return two utilities")
                challenger_utility = _finite(
                    "score_spec challenger utility", challenger_utility
                )
                incumbent_utility = _finite(
                    "score_spec incumbent utility", incumbent_utility
                )
            else:
                if self.certified:
                    raise ValueError(
                        "certified evidence is missing the raw label value"
                    )
                if (
                    receipt.challenger_utility is None
                    or receipt.incumbent_utility is None
                ):
                    raise ValueError("observed evidence is missing utilities")
                challenger_utility = receipt.challenger_utility
                incumbent_utility = receipt.incumbent_utility
            state_before = self._gate.challenger_state(self._challenger_id)
            safe_metadata = dict(receipt.metadata or {})
            safe_metadata.update(
                {
                    "prediction_id": receipt.prediction_id,
                    "label_id": receipt.label_id,
                    "target_id": receipt.target_id,
                    "decision_step": prediction.decision_step,
                    "label_available_step": receipt.label_available_step,
                    "settlement_step": settlement_step,
                    "model_fingerprint": prediction.model_fingerprint,
                    "feature_fingerprint": prediction.feature_fingerprint,
                    "family_id": self._config.family_id,
                    "config_identity": self._config.identity,
                    "eta_policy_id": prediction.eta_policy_id,
                    "eta": prediction.eta,
                    "label_value": receipt.label_value,
                    "score_spec_id": self._config.score_spec_id,
                    "score_spec_identity": self._score_spec_identity,
                    "allocation_id": prediction.allocation_id,
                    "prior_observation_count": state_before.observations,
                    "evidence_identity": self.evidence_identity,
                }
            )
            gate_internal_state = copy.deepcopy(self._gate.__dict__)
            gate_digest_before = self._expected_gate_evidence_digest
            try:
                decision = self._gate.observe_utilities(
                    self._challenger_id,
                    challenger_utility,
                    incumbent_utility,
                    eta=prediction.eta,
                    metadata=safe_metadata,
                )
                self._refresh_gate_binding()
                resolution = EvidenceResolution(
                    prediction_id=receipt.prediction_id,
                    label_id=receipt.label_id,
                    label_available_step=receipt.label_available_step,
                    accepted=True,
                    burn_in=False,
                    reason="frozen paired evidence submitted to e-process",
                    decision=decision,
                    status="settled",
                    evidence_index=evidence_index,
                    settlement_step=settlement_step,
                    promotion_effective_step=(
                        settlement_step + 1 if decision.promoted else None
                    ),
                )
            except Exception:
                self._rollback_gate(gate_internal_state, gate_digest_before)
                raise
        del self._pending[receipt.prediction_id]
        del self._labels[receipt.prediction_id]
        self._resolved_prediction_ids.add(receipt.prediction_id)
        self._resolved_label_ids.add(receipt.label_id)
        self._last_label_available_step = receipt.label_available_step
        self._last_settlement_step = settlement_step
        self._resolutions.append(resolution)
        if receipt.status == "observed":
            self._observed_label_count += 1
        self._evidence_index += 1
        return resolution

    def settle_ready(self, *, at_step: int) -> tuple[EvidenceResolution, ...]:
        """Settle the earliest contiguous prediction prefix available at ``at_step``."""

        self._assert_gate_binding()
        at_step = _step("at_step", at_step)
        if self._last_settlement_step is not None and at_step < self._last_settlement_step:
            raise ValueError("settlement steps must be non-decreasing")
        settled: list[EvidenceResolution] = []
        while self._pending:
            prediction = next(iter(self._pending.values()))
            receipt = self._labels.get(prediction.prediction_id)
            if receipt is None or receipt.label_available_step > at_step:
                break
            settled.append(self._settle_receipt(receipt, settlement_step=at_step))
        return tuple(settled)

    def resolve_label(
        self,
        *,
        prediction_id: str,
        label_id: str,
        label_available_step: int,
        label_value: float | None = None,
        challenger_utility: float | None = None,
        incumbent_utility: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> EvidenceResolution:
        """Resolve one frozen prediction and, after burn-in, feed the gate.

        Eta is intentionally not an argument. The registered gate policy is
        resolved from prior evidence only; a caller cannot tune it after seeing
        the current label. Certified ledgers require ``label_value`` and
        recompute both utilities from the frozen paired predictions; explicit
        utility arguments are retained only for diagnostic ledgers.
        """

        self.ingest_label(
            prediction_id=prediction_id,
            label_id=label_id,
            label_available_step=label_available_step,
            label_value=label_value,
            challenger_utility=challenger_utility,
            incumbent_utility=incumbent_utility,
            metadata=metadata,
        )
        settled = self.settle_ready(at_step=label_available_step)
        for resolution in settled:
            if resolution.prediction_id == prediction_id:
                return resolution
        raise ValueError("label buffered until earlier prediction evidence is available")

    def mark_missing(
        self,
        *,
        prediction_id: str,
        label_id: str,
        label_available_step: int,
        reason: str = "label missing at terminalization",
        expired: bool = False,
        received_step: int | None = None,
    ) -> EvidenceResolution:
        """Terminalize a label without advancing the e-process."""
        available = _step("label_available_step", label_available_step)
        self.ingest_label(
            prediction_id=prediction_id,
            label_id=label_id,
            label_available_step=available,
            received_step=(
                max(available, self._last_received_step)
                if received_step is None and self._last_received_step is not None
                else received_step
            ),
            status="expired" if expired else "missing",
        )
        settled = self.settle_ready(at_step=available)
        for resolution in settled:
            if resolution.prediction_id == prediction_id:
                replacement = replace(resolution, reason=_text("reason", reason))
                for index, existing in enumerate(self._resolutions):
                    if existing is resolution:
                        self._resolutions[index] = replacement
                        break
                return replacement
        raise ValueError("missing label buffered until earlier prediction evidence is terminalized")

    def _validate_restored_order(self) -> None:
        """Validate the append-only prediction/resolution prefix invariant."""

        predictions = tuple(self._prediction_history.values())
        decision_steps = tuple(value.decision_step for value in predictions)
        if any(right <= left for left, right in zip(decision_steps, decision_steps[1:])):
            raise ValueError("evidence prediction steps are not strictly increasing")
        if self._last_decision_step != (decision_steps[-1] if decision_steps else None):
            raise ValueError("evidence last decision step is inconsistent")

        resolutions = tuple(self._resolutions)
        resolution_ids = tuple(value.prediction_id for value in resolutions)
        expected_resolved = tuple(value.prediction_id for value in predictions[: len(resolutions)])
        if resolution_ids != expected_resolved:
            raise ValueError("evidence resolutions must consume a prediction prefix")
        pending_ids = tuple(self._pending)
        expected_pending = tuple(value.prediction_id for value in predictions[len(resolutions) :])
        if pending_ids != expected_pending:
            raise ValueError("evidence pending predictions must be the unresolved suffix")
        if self._evidence_index != len(resolutions):
            raise ValueError("evidence index is inconsistent with resolutions")

        settlement_steps: list[int] = []
        observed_count = 0
        for index, resolution in enumerate(resolutions):
            if resolution.evidence_index != index:
                raise ValueError("evidence resolution indices must be contiguous")
            if resolution.status in {"settled", "burn_in"}:
                observed_count += 1
            if resolution.status in {"missing", "expired"} and resolution.accepted:
                raise ValueError("missing evidence cannot be accepted")
            if resolution.status == "burn_in" and not resolution.burn_in:
                raise ValueError("burn-in resolution is missing its burn-in marker")
            if resolution.status != "burn_in" and resolution.burn_in:
                raise ValueError("non-burn-in resolution has a burn-in marker")
            if resolution.decision is not None:
                if resolution.status != "settled" or not resolution.accepted:
                    raise ValueError("promotion decision has an invalid resolution status")
                if resolution.decision.challenger_id != self._challenger_id:
                    raise ValueError("promotion decision challenger does not match the ledger")
                if resolution.decision.epoch != self._initial_gate_epoch:
                    raise ValueError("promotion decision epoch does not match the ledger")
            if resolution.settlement_step is None:
                raise ValueError("evidence resolution is missing settlement_step")
            settlement_steps.append(resolution.settlement_step)

        if any(right < left for left, right in zip(settlement_steps, settlement_steps[1:])):
            raise ValueError("evidence settlement steps must be non-decreasing")
        if self._observed_label_count != observed_count:
            raise ValueError("evidence observed-label count is inconsistent")
        last_resolution = resolutions[-1] if resolutions else None
        if self._last_label_available_step != (
            last_resolution.label_available_step if last_resolution is not None else None
        ):
            raise ValueError("evidence last label-availability step is inconsistent")
        if self._last_settlement_step != (
            last_resolution.settlement_step if last_resolution is not None else None
        ):
            raise ValueError("evidence last settlement step is inconsistent")
        if self._last_received_step is not None:
            _step("last_received_step", self._last_received_step)
        gate_state = self._gate.challenger_state(self._challenger_id)
        decision_count = sum(value.decision is not None for value in resolutions)
        if gate_state.observations != decision_count:
            raise ValueError("evidence decision count disagrees with gate observations")
        promoted = tuple(value for value in resolutions if value.promoted)
        if bool(promoted) != (self._gate.status is GateStatus.PROMOTED):
            raise ValueError("evidence promotion state disagrees with gate status")
        if promoted and self._gate.promoted_challenger_id != self._challenger_id:
            raise ValueError("evidence promotion challenger disagrees with gate")

    def state_dict(self) -> dict[str, Any]:
        self._assert_gate_binding()
        if len(self._resolutions) > self._max_records:
            raise ValueError("evidence resolution history exceeds max_records")
        return {
            "version": EVIDENCE_SCHEMA_VERSION,
            "schema": EVIDENCE_SCHEMA,
            "challenger_id": self._challenger_id,
            "burn_in": self._burn_in,
            "max_pending": self._max_pending,
            "max_records": self._max_records,
            "config": self._config.state_dict(),
            "config_identity": self._config.identity,
            "eta_policy_identity": self._eta_policy_identity,
            "score_spec_identity": self._score_spec_identity,
            "claim": self.promotion_claim,
            "initial_gate_epoch": self._initial_gate_epoch,
            "allocation_id": self._allocation_id,
            "predictions": [value.state_dict() for value in self._prediction_history.values()],
            "pending": [value.state_dict() for value in self._pending.values()],
            "labels": [value.state_dict() for value in self._labels.values()],
            "resolved_prediction_ids": sorted(self._resolved_prediction_ids),
            "resolved_label_ids": sorted(self._resolved_label_ids),
            "last_decision_step": self._last_decision_step,
            "last_label_available_step": self._last_label_available_step,
            "last_settlement_step": self._last_settlement_step,
            "last_received_step": self._last_received_step,
            "evidence_index": self._evidence_index,
            "observed_label_count": self._observed_label_count,
            "resolution_count": len(self._resolutions),
            "resolutions": [value.to_dict() for value in self._resolutions],
            "gate_evidence_digest": self._expected_gate_evidence_digest,
            "evidence_identity": self.evidence_identity,
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, Any],
        *,
        gate: PromotionGate,
        eta_policy: Callable[[Mapping[str, Any]], Any] | float | None = None,
        score_spec: Callable[[FrozenPrediction, float], Any] | None = None,
    ) -> "EvidenceLedger":
        if not isinstance(state, Mapping):
            raise ValueError("evidence ledger state must be a mapping")
        version = state.get("version")
        if state.get("schema") != EVIDENCE_SCHEMA or version not in {1, 2, EVIDENCE_SCHEMA_VERSION}:
            raise ValueError("unsupported evidence ledger version or schema")
        config_raw = state.get("config")
        config = (
            None
            if not isinstance(config_raw, Mapping)
            else PromotionEvidenceConfig.from_state_dict(config_raw)
        )
        candidate = cls(
            gate,
            cast(str, state.get("challenger_id")),
            burn_in=state.get("burn_in", 0),
            max_pending=state.get("max_pending", MAX_EVIDENCE_PENDING),
            max_records=state.get("max_records", MAX_EVIDENCE_RECORDS),
            config=config,
            eta_policy=eta_policy,
            score_spec=score_spec,
            _restoring=True,
        )
        if version == EVIDENCE_SCHEMA_VERSION and state.get("evidence_identity") not in {None, candidate.evidence_identity}:
            raise ValueError("evidence ledger identity mismatch")
        if version == EVIDENCE_SCHEMA_VERSION and state.get("allocation_id") not in {None, candidate.allocation_id}:
            raise ValueError("evidence allocation identity mismatch")
        if version == EVIDENCE_SCHEMA_VERSION:
            if state.get("config_identity") != candidate._config.identity:
                raise ValueError("evidence config identity mismatch")
            if state.get("claim") != candidate.promotion_claim:
                raise ValueError("evidence promotion claim mismatch")
            if state.get("initial_gate_epoch") != candidate._initial_gate_epoch:
                raise ValueError("evidence gate epoch mismatch")
            if state.get("eta_policy_identity") != candidate._eta_policy_identity:
                raise ValueError("evidence eta policy identity mismatch")
            if state.get("score_spec_identity") != candidate._score_spec_identity:
                raise ValueError("evidence score spec identity mismatch")
            if state.get("gate_evidence_digest") != candidate._expected_gate_evidence_digest:
                raise ValueError("evidence gate fingerprint mismatch")
        predictions_raw = state.get("predictions", state.get("pending", ()))
        if not isinstance(predictions_raw, (list, tuple)):
            raise ValueError("evidence predictions must be a sequence")
        for raw in predictions_raw:
            prediction = FrozenPrediction.from_state_dict(raw)
            if prediction.prediction_id in candidate._prediction_history:
                raise ValueError("duplicate prediction in evidence state")
            if prediction.challenger_id != candidate._challenger_id:
                raise ValueError("prediction challenger does not match the ledger")
            if prediction.family_id != candidate._config.family_id:
                raise ValueError("prediction family does not match the ledger")
            if prediction.incumbent_id != candidate._config.incumbent_id:
                raise ValueError("prediction incumbent does not match the ledger")
            if prediction.gate_epoch != candidate._initial_gate_epoch:
                raise ValueError("prediction gate epoch does not match the ledger")
            if prediction.allocation_id != candidate._allocation_id:
                raise ValueError("prediction allocation does not match the ledger")
            candidate._prediction_history[prediction.prediction_id] = prediction
        pending_raw = state.get("pending", ())
        if not isinstance(pending_raw, (list, tuple)):
            raise ValueError("evidence pending state must be a sequence")
        for raw in pending_raw:
            prediction = FrozenPrediction.from_state_dict(raw)
            original = candidate._prediction_history.get(prediction.prediction_id)
            if original is None or original != prediction:
                raise ValueError("pending prediction is not present in prediction history")
            candidate._pending[prediction.prediction_id] = original
        labels_raw = state.get("labels", ())
        if not isinstance(labels_raw, (list, tuple)):
            raise ValueError("evidence labels must be a sequence")
        for raw in labels_raw:
            label = LabelReceipt.from_state_dict(raw)
            if label.prediction_id not in candidate._pending:
                raise ValueError("buffered label has no pending prediction")
            if label.prediction_id in candidate._labels:
                raise ValueError("duplicate buffered label")
            if label.target_id != candidate._pending[label.prediction_id].target_id:
                raise ValueError("buffered label target does not match prediction")
            candidate._labels[label.prediction_id] = label
        resolution_raw = state.get("resolutions", ())
        if not isinstance(resolution_raw, (list, tuple)):
            raise ValueError("evidence resolutions must be a sequence")
        candidate._resolutions = [EvidenceResolution.from_state_dict(raw) for raw in resolution_raw]
        candidate._resolved_prediction_ids = set(str(value) for value in state.get("resolved_prediction_ids", ()))
        candidate._resolved_label_ids = set(str(value) for value in state.get("resolved_label_ids", ()))
        resolution_predictions = {value.prediction_id for value in candidate._resolutions}
        resolution_labels = {value.label_id for value in candidate._resolutions}
        if resolution_predictions != candidate._resolved_prediction_ids or resolution_labels != candidate._resolved_label_ids:
            raise ValueError("evidence resolution IDs do not match the ledger")
        if candidate._resolved_prediction_ids & set(candidate._pending):
            raise ValueError("resolved prediction remains pending")
        candidate._last_decision_step = _optional_step(
            "last_decision_step", state.get("last_decision_step")
        )
        candidate._last_label_available_step = _optional_step(
            "last_label_available_step", state.get("last_label_available_step")
        )
        candidate._last_settlement_step = _optional_step(
            "last_settlement_step", state.get("last_settlement_step")
        )
        candidate._last_received_step = _optional_step(
            "last_received_step", state.get("last_received_step")
        )
        candidate._evidence_index = _step("evidence_index", state.get("evidence_index", len(candidate._resolutions)))
        candidate._observed_label_count = _step(
            "observed_label_count",
            state.get(
                "observed_label_count",
                sum(value.status in {"settled", "burn_in"} for value in candidate._resolutions),
            ),
        )
        if state.get("resolution_count", len(candidate._resolutions)) != len(candidate._resolutions):
            raise ValueError("evidence resolution count mismatch")
        candidate._validate_restored_order()
        candidate._assert_gate_binding()
        return candidate


__all__ = [
    "EvidenceLedger",
    "EvidenceResolution",
    "FrozenPrediction",
    "LabelReceipt",
    "MAX_EVIDENCE_PENDING",
    "MAX_EVIDENCE_RECORDS",
    "PromotionEvidenceConfig",
]
