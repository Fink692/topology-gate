"""Strict source-bundle and phase-boundary contracts for causal studies.

The numerical replay adapters are intentionally usable with a canonical
``AsOfBook`` directly.  A real study needs one more boundary around them:
the source artifacts, ordered timeline, expected point-in-time universe, and
economic evidence must be bound together before a model is allowed to run.

This module does not parse vendor-native files or manufacture market data.  It
accepts the normalized artifacts produced by a vendor adapter and rejects an
incomplete or phase-incompatible bundle before entering causal replay.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .asof import AsOfBook, TimePoint
from .causal_numeric import (
    CausalFeaturePlan,
    CausalRLSConfig,
    CausalRLSReplayResult,
    run_causal_rls_replay,
)
from .causal_promotion import (
    CausalPromotionActivation,
    CausalPromotionConfig,
    CausalPromotionReplayResult,
    CausalPromotionStatus,
    run_causal_promotion_replay,
)
from .economic import EconomicDecision, EconomicEvidence
from .manifest import ManifestValidationError, RunManifest, StudyManifest
from .promotion import PromotionGate
from .replay import ReplayConfig, ReplayState, ReplayStatus

STUDY_INPUT_SCHEMA = "topology_gate.study_input_bundle"
STUDY_INPUT_VERSION = 1
MAX_STUDY_TIMELINE = 100_000


class StudyInputError(ValueError):
    """Raised when a normalized study source bundle is not run-ready."""


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StudyInputError(f"{name} must be a non-empty string")
    return value


def _time_domain(value: Any, name: str) -> str:
    if isinstance(value, bool) or value is None:
        raise StudyInputError(f"{name} must be a supported time point")
    if isinstance(value, datetime):
        return "datetime"
    if isinstance(value, str):
        if not value:
            raise StudyInputError(f"{name} must not be empty")
        return "str"
    if isinstance(value, int):
        return "numeric"
    if isinstance(value, float):
        if not math.isfinite(value):
            raise StudyInputError(f"{name} must be finite")
        return "numeric"
    raise StudyInputError(
        f"{name} must be an int, float, str, or datetime time point"
    )


def _time_less(left: TimePoint, right: TimePoint, name: str) -> bool:
    _time_domain(left, f"{name} left")
    _time_domain(right, f"{name} right")
    try:
        return bool(left < right)  # type: ignore[operator]
    except TypeError as exc:
        raise StudyInputError(f"{name} values use different time domains") from exc


def _encode_time(value: TimePoint, name: str) -> dict[str, Any]:
    domain = _time_domain(value, name)
    if isinstance(value, datetime):
        return {"kind": domain, "value": value.isoformat()}
    return {"kind": type(value).__name__, "value": value}


def _decode_time(value: Any, name: str) -> TimePoint:
    if not isinstance(value, Mapping) or set(value) != {"kind", "value"}:
        raise StudyInputError(f"{name} must be a tagged time point")
    kind = value["kind"]
    raw = value["value"]
    if kind == "datetime":
        if not isinstance(raw, str) or not raw:
            raise StudyInputError(f"{name} datetime value must be a non-empty string")
        try:
            return datetime.fromisoformat(raw)
        except ValueError as exc:
            raise StudyInputError(f"{name} datetime value is invalid") from exc
    if kind == "str":
        if not isinstance(raw, str) or not raw:
            raise StudyInputError(f"{name} string value must be non-empty")
        return raw
    if kind == "int":
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise StudyInputError(f"{name} integer value is invalid")
        return raw
    if kind == "float":
        if isinstance(raw, bool) or not isinstance(raw, float) or not math.isfinite(raw):
            raise StudyInputError(f"{name} float value is invalid")
        return raw
    raise StudyInputError(f"{name} has an unsupported time-point kind")


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise StudyInputError("study identity is not JSON-safe") from exc


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _stored_digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise StudyInputError(f"{name} must be a 64-character hexadecimal value")
    if any(character not in "0123456789abcdefABCDEF" for character in value):
        raise StudyInputError(f"{name} must be hexadecimal")
    return value.lower()


def _sequence(value: Any, name: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise StudyInputError(f"{name} must be a sequence")
    return tuple(value)


def _instrument_ids(value: Any, name: str) -> tuple[str, ...]:
    values = _sequence(value, name)
    if not values:
        raise StudyInputError(f"{name} must not be empty")
    normalized = tuple(_text(item, f"{name} instrument") for item in values)
    if len(set(normalized)) != len(normalized):
        raise StudyInputError(f"{name} must not contain duplicate instruments")
    return tuple(sorted(normalized))


@dataclass(frozen=True, slots=True)
class StudyTimeline:
    """Ordered decision anchors and their pre-registered timeline indices."""

    decision_times: Sequence[TimePoint]
    target_ids: Sequence[str]
    decision_indices: Sequence[int]
    expected_instrument_ids: Sequence[Sequence[str]] | None = None

    def __post_init__(self) -> None:
        times = _sequence(self.decision_times, "decision_times")
        targets = tuple(_text(value, "target_id") for value in _sequence(self.target_ids, "target_ids"))
        indices = _sequence(self.decision_indices, "decision_indices")
        if not times:
            raise StudyInputError("study timeline must not be empty")
        if len(times) > MAX_STUDY_TIMELINE:
            raise StudyInputError("study timeline exceeds the resource limit")
        if len(targets) != len(times) or len(indices) != len(times):
            raise StudyInputError(
                "decision_times, target_ids, and decision_indices must align"
            )
        if len(set(targets)) != len(targets):
            raise StudyInputError("target_ids must be unique within a study timeline")

        for index, value in enumerate(times):
            _time_domain(value, f"decision_times[{index}]")
            if index and not _time_less(times[index - 1], value, "decision_times"):
                raise StudyInputError("decision_times must be strictly increasing")
        normalized_indices: list[int] = []
        for index, value in enumerate(indices):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise StudyInputError(
                    f"decision_indices[{index}] must be a non-negative integer"
                )
            if normalized_indices and value <= normalized_indices[-1]:
                raise StudyInputError("decision_indices must be strictly increasing")
            normalized_indices.append(value)

        expected = self.expected_instrument_ids
        normalized_expected: tuple[tuple[str, ...], ...] | None = None
        if expected is not None:
            rows = _sequence(expected, "expected_instrument_ids")
            if len(rows) != len(times):
                raise StudyInputError(
                    "expected_instrument_ids must align with decision_times"
                )
            normalized_expected = tuple(
                _instrument_ids(row, f"expected_instrument_ids[{index}]")
                for index, row in enumerate(rows)
            )

        object.__setattr__(self, "decision_times", times)
        object.__setattr__(self, "target_ids", targets)
        object.__setattr__(self, "decision_indices", tuple(normalized_indices))
        object.__setattr__(self, "expected_instrument_ids", normalized_expected)

    def to_dict(self) -> dict[str, Any]:
        expected = self.expected_instrument_ids
        return {
            "decision_times": [
                _encode_time(value, f"decision_times[{index}]")
                for index, value in enumerate(self.decision_times)
            ],
            "target_ids": list(self.target_ids),
            "decision_indices": list(self.decision_indices),
            "expected_instrument_ids": (
                None if expected is None else [list(row) for row in expected]
            ),
        }

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> "StudyTimeline":
        if not isinstance(state, Mapping):
            raise StudyInputError("study timeline state must be a mapping")
        expected_keys = {
            "decision_times",
            "target_ids",
            "decision_indices",
            "expected_instrument_ids",
        }
        if set(state) != expected_keys:
            raise StudyInputError(
                "study timeline contains unknown or missing fields"
            )
        raw_times = _sequence(state["decision_times"], "decision_times")
        times = tuple(
            _decode_time(value, f"decision_times[{index}]")
            for index, value in enumerate(raw_times)
        )
        expected = state["expected_instrument_ids"]
        try:
            return cls(
                decision_times=times,
                target_ids=state["target_ids"],
                decision_indices=state["decision_indices"],
                expected_instrument_ids=(
                    None if expected is None else expected
                ),
            )
        except (TypeError, ValueError) as exc:
            raise StudyInputError("study timeline is invalid") from exc

    @classmethod
    def from_json(cls, payload: str) -> "StudyTimeline":
        if not isinstance(payload, str) or not payload.strip():
            raise TypeError("study timeline JSON must be a non-empty string")
        try:
            state = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise StudyInputError("study timeline JSON is invalid") from exc
        return cls.from_dict(state)

    @property
    def digest(self) -> str:
        return _digest(self.to_dict())


@dataclass(frozen=True, slots=True)
class StudyInputAudit:
    """Auditable result of validating a source bundle for one study phase."""

    bundle_digest: str
    phase: str
    decision_count: int
    timeline_digest: str
    as_of_book_digest: str
    economic_evidence_digest: str | None
    economic_cutoff: TimePoint | None
    expected_universe_complete: bool
    economic_records_complete: bool
    capacity_evidence_complete: bool
    holdout_status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": STUDY_INPUT_SCHEMA,
            "version": STUDY_INPUT_VERSION,
            "bundle_digest": self.bundle_digest,
            "phase": self.phase,
            "decision_count": self.decision_count,
            "timeline_digest": self.timeline_digest,
            "as_of_book_digest": self.as_of_book_digest,
            "economic_evidence_digest": self.economic_evidence_digest,
            "economic_cutoff": (
                None
                if self.economic_cutoff is None
                else _encode_time(self.economic_cutoff, "economic_cutoff")
            ),
            "expected_universe_complete": self.expected_universe_complete,
            "economic_records_complete": self.economic_records_complete,
            "capacity_evidence_complete": self.capacity_evidence_complete,
            "holdout_status": self.holdout_status,
        }

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> "StudyInputAudit":
        if not isinstance(state, Mapping):
            raise StudyInputError("study input audit state must be a mapping")
        expected = {
            "schema",
            "version",
            "bundle_digest",
            "phase",
            "decision_count",
            "timeline_digest",
            "as_of_book_digest",
            "economic_evidence_digest",
            "economic_cutoff",
            "expected_universe_complete",
            "economic_records_complete",
            "capacity_evidence_complete",
            "holdout_status",
        }
        if set(state) != expected:
            raise StudyInputError(
                "study input audit contains unknown or missing fields"
            )
        if state["schema"] != STUDY_INPUT_SCHEMA:
            raise StudyInputError("unsupported study input audit schema")
        if type(state["version"]) is not int or state["version"] != STUDY_INPUT_VERSION:
            raise StudyInputError("unsupported study input audit version")
        count = state["decision_count"]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise StudyInputError("decision_count must be a non-negative integer")
        for name in ("bundle_digest", "timeline_digest", "as_of_book_digest"):
            _stored_digest(state[name], name)
        economic_digest = state["economic_evidence_digest"]
        if economic_digest is not None:
            _stored_digest(economic_digest, "economic_evidence_digest")
        flags = (
            "expected_universe_complete",
            "economic_records_complete",
            "capacity_evidence_complete",
        )
        if any(type(state[name]) is not bool for name in flags):
            raise StudyInputError("study input audit completeness fields must be boolean")
        try:
            cutoff = (
                None
                if state["economic_cutoff"] is None
                else _decode_time(state["economic_cutoff"], "economic_cutoff")
            )
            return cls(
                bundle_digest=_stored_digest(state["bundle_digest"], "bundle_digest"),
                phase=_text(state["phase"], "phase"),
                decision_count=count,
                timeline_digest=_stored_digest(
                    state["timeline_digest"], "timeline_digest"
                ),
                as_of_book_digest=_stored_digest(
                    state["as_of_book_digest"], "as_of_book_digest"
                ),
                economic_evidence_digest=(
                    None
                    if economic_digest is None
                    else _stored_digest(
                        economic_digest, "economic_evidence_digest"
                    )
                ),
                economic_cutoff=cutoff,
                expected_universe_complete=state["expected_universe_complete"],
                economic_records_complete=state["economic_records_complete"],
                capacity_evidence_complete=state["capacity_evidence_complete"],
                holdout_status=_text(state["holdout_status"], "holdout_status"),
            )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, StudyInputError):
                raise
            raise StudyInputError("study input audit is invalid") from exc


@dataclass(frozen=True, slots=True)
class StudyInputBundle:
    """Normalized source artifacts bound to one ordered causal study."""

    run_manifest: RunManifest
    study_manifest: StudyManifest
    timeline: StudyTimeline
    as_of_book: AsOfBook
    economic_evidence: EconomicEvidence | None = None
    economic_cutoff: TimePoint | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.run_manifest, RunManifest):
            raise StudyInputError("run_manifest must be a RunManifest")
        if not isinstance(self.study_manifest, StudyManifest):
            raise StudyInputError("study_manifest must be a StudyManifest")
        if (
            self.study_manifest.spec.run_spec.digest
            != self.run_manifest.spec.digest
        ):
            raise StudyInputError(
                "study manifest run specification does not match run manifest"
            )
        if not isinstance(self.timeline, StudyTimeline):
            raise StudyInputError("timeline must be a StudyTimeline")
        if not isinstance(self.as_of_book, AsOfBook):
            raise StudyInputError("as_of_book must be an AsOfBook")
        if self.economic_evidence is not None and not isinstance(
            self.economic_evidence, EconomicEvidence
        ):
            raise StudyInputError("economic_evidence must be EconomicEvidence")
        if self.economic_cutoff is not None:
            _time_domain(self.economic_cutoff, "economic_cutoff")
            if self.economic_evidence is None:
                raise StudyInputError(
                    "economic_cutoff requires economic_evidence"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": STUDY_INPUT_SCHEMA,
            "version": STUDY_INPUT_VERSION,
            "run_manifest_digest": self.run_manifest.digest,
            "study_manifest_digest": self.study_manifest.digest,
            "timeline": self.timeline.to_dict(),
            "as_of_book_digest": self.as_of_book.digest,
            "economic_evidence_digest": (
                None
                if self.economic_evidence is None
                else self.economic_evidence.digest
            ),
            "economic_cutoff": (
                None
                if self.economic_cutoff is None
                else _encode_time(self.economic_cutoff, "economic_cutoff")
            ),
        }

    def _timeline_for_phase(self, phase: str) -> StudyTimeline:
        """Return the declared timeline rows belonging to ``phase``.

        A source package normally carries one complete pre-registered timeline.
        Audits and replays, however, operate one phase at a time.  Selecting
        the rows here lets the same sealed package pass calibration, tuning,
        and validation in order while preserving the full bundle digest.
        """

        phase_name = _text(phase, "study phase")
        try:
            window = self.study_manifest.spec.window_for_phase(phase_name)
        except ManifestValidationError as exc:
            raise StudyInputError(str(exc)) from exc
        if phase_name == "holdout" and self.study_manifest.holdout_is_sealed:
            raise StudyInputError("sealed study holdout cannot be read")

        positions = [
            position
            for position, decision_index in enumerate(self.timeline.decision_indices)
            if window.start <= decision_index < window.end
        ]
        if not positions:
            raise StudyInputError(
                f"study timeline has no decisions in the {phase_name} window"
            )
        selected_indices = tuple(self.timeline.decision_indices[position] for position in positions)
        try:
            self.study_manifest.assert_indices_allowed(selected_indices, phase_name)
        except ManifestValidationError as exc:
            raise StudyInputError(str(exc)) from exc
        expected = self.timeline.expected_instrument_ids
        return StudyTimeline(
            decision_times=tuple(self.timeline.decision_times[position] for position in positions),
            target_ids=tuple(self.timeline.target_ids[position] for position in positions),
            decision_indices=selected_indices,
            expected_instrument_ids=(
                None
                if expected is None
                else tuple(expected[position] for position in positions)
            ),
        )

    @property
    def digest(self) -> str:
        return _digest(self.to_dict())

    def audit(
        self,
        phase: str,
        *,
        require_complete_universe: bool = False,
        require_economic_evidence: bool = False,
        require_observed_economic_evidence: bool = False,
        require_capacity_evidence: bool = False,
    ) -> StudyInputAudit:
        """Validate source completeness before entering a causal replay."""

        phase_name = _text(phase, "study phase")
        if not isinstance(require_complete_universe, bool):
            raise StudyInputError("require_complete_universe must be boolean")
        if not isinstance(require_economic_evidence, bool):
            raise StudyInputError("require_economic_evidence must be boolean")
        if not isinstance(require_observed_economic_evidence, bool):
            raise StudyInputError(
                "require_observed_economic_evidence must be boolean"
            )
        if not isinstance(require_capacity_evidence, bool):
            raise StudyInputError("require_capacity_evidence must be boolean")
        if require_observed_economic_evidence:
            require_economic_evidence = True
        if require_capacity_evidence:
            require_economic_evidence = True

        phase_timeline = self._timeline_for_phase(phase_name)

        expected_rows = phase_timeline.expected_instrument_ids
        if require_complete_universe and expected_rows is None:
            raise StudyInputError(
                "complete-universe validation requires expected_instrument_ids"
            )
        expected_complete = expected_rows is not None
        for position, (decision_time, target_id) in enumerate(
            zip(phase_timeline.decision_times, phase_timeline.target_ids)
        ):
            try:
                snapshot = self.as_of_book.materialize(decision_time)
            except Exception as exc:
                raise StudyInputError(
                    f"cannot materialize decision {position} at {decision_time!r}"
                ) from exc
            if target_id in {label.target_id for label in snapshot.labels}:
                raise StudyInputError(
                    f"target label {target_id!r} is already visible at decision "
                    f"{decision_time!r}"
                )
            if expected_rows is not None:
                actual = tuple(
                    sorted({item.instrument_id for item in snapshot.universe})
                )
                if actual != expected_rows[position]:
                    raise StudyInputError(
                        "point-in-time universe mismatch at decision "
                        f"{position}: expected {expected_rows[position]!r}, "
                        f"got {actual!r}"
                    )

        economic_complete = False
        capacity_complete = False
        if require_economic_evidence:
            if self.economic_evidence is None:
                raise StudyInputError(
                    "economic evidence is required for this study phase"
                )
            if self.economic_cutoff is None:
                raise StudyInputError(
                    "economic evidence validation requires an explicit cutoff"
                )
            returns, costs = self.economic_evidence.select_at(self.economic_cutoff)
            missing_returns = []
            missing_costs = []
            non_observed = []
            missing_capacity = []
            for target_id, decision_time in zip(
                phase_timeline.target_ids, phase_timeline.decision_times
            ):
                return_item = returns.get(target_id)
                cost_item = costs.get(target_id)
                if return_item is None or return_item.decision_time != decision_time:
                    missing_returns.append(target_id)
                elif return_item.status != "observed":
                    non_observed.append(target_id)
                if cost_item is None or cost_item.decision_time != decision_time:
                    missing_costs.append(target_id)
                elif cost_item.capacity_limit is None:
                    missing_capacity.append(target_id)
            if missing_returns or missing_costs:
                raise StudyInputError(
                    "economic evidence is incomplete: "
                    f"missing returns={missing_returns!r}, "
                    f"missing costs={missing_costs!r}"
                )
            capacity_complete = not missing_capacity
            if require_capacity_evidence and missing_capacity:
                raise StudyInputError(
                    "capacity evidence is incomplete for "
                    f"{missing_capacity!r}"
                )
            if require_observed_economic_evidence and non_observed:
                raise StudyInputError(
                    "economic evidence contains non-observed returns for "
                    f"{non_observed!r}"
                )
            economic_complete = True

        return StudyInputAudit(
            bundle_digest=self.digest,
            phase=phase_name,
            decision_count=len(phase_timeline.decision_times),
            timeline_digest=phase_timeline.digest,
            as_of_book_digest=self.as_of_book.digest,
            economic_evidence_digest=(
                None
                if self.economic_evidence is None
                else self.economic_evidence.digest
            ),
            economic_cutoff=self.economic_cutoff,
            expected_universe_complete=expected_complete,
            economic_records_complete=economic_complete,
            capacity_evidence_complete=capacity_complete,
            holdout_status=self.study_manifest.holdout_status,
        )


@dataclass(frozen=True, slots=True)
class StudyRLSRunResult:
    """Causal RLS output paired with the preflight receipt."""

    audit: StudyInputAudit
    replay: CausalRLSReplayResult

    @property
    def economic_decisions(self) -> tuple[EconomicDecision, ...]:
        """Convert emitted positions into explicit economic decision records."""

        predictions = self.replay.replay.predictions[self.replay.prediction_start :]
        if len(predictions) != len(self.replay.steps):
            raise StudyInputError(
                "causal replay predictions and step telemetry are misaligned"
            )
        decisions: list[EconomicDecision] = []
        for prediction, step in zip(predictions, self.replay.steps):
            evaluated = prediction.status is ReplayStatus.PREDICTED
            decisions.append(
                EconomicDecision(
                    decision_id=prediction.prediction_id,
                    target_id=prediction.target_id,
                    decision_time=prediction.decision_time,
                    position=step.position if evaluated else 0.0,
                    evaluated=evaluated,
                    reason=None if evaluated else prediction.status.value,
                )
            )
        return tuple(decisions)


@dataclass(frozen=True, slots=True)
class StudyPromotionRunResult:
    """Paired-promotion output paired with the same preflight receipt."""

    audit: StudyInputAudit
    replay: CausalPromotionReplayResult

    @property
    def promoted(self) -> bool:
        return self.replay.promoted

    @property
    def operational_status(self) -> CausalPromotionStatus:
        return self.replay.operational_status

    @property
    def promotion_activation(self) -> CausalPromotionActivation | None:
        return self.replay.promotion_activation

    @property
    def operational_counts(self) -> Mapping[str, int]:
        return self.replay.operational_counts

    @property
    def promotion_block_reason(self) -> str | None:
        return self.replay.promotion_block_reason


def run_causal_rls_study(
    bundle: StudyInputBundle,
    phase: str,
    *,
    plan: CausalFeaturePlan,
    learner: Any,
    detector: Any | None = None,
    calibration: Any | None = None,
    model_config: CausalRLSConfig | None = None,
    replay_config: ReplayConfig | None = None,
    model_state: Mapping[str, Any] | None = None,
    initial_state: ReplayState | None = None,
    require_complete_universe: bool = False,
    require_economic_evidence: bool = False,
    require_observed_economic_evidence: bool = False,
    require_capacity_evidence: bool = False,
) -> StudyRLSRunResult:
    """Preflight a bundle, then run the shared causal RLS transition."""

    if not isinstance(bundle, StudyInputBundle):
        raise StudyInputError("bundle must be a StudyInputBundle")
    audit = bundle.audit(
        phase,
        require_complete_universe=require_complete_universe,
        require_economic_evidence=require_economic_evidence,
        require_observed_economic_evidence=require_observed_economic_evidence,
        require_capacity_evidence=require_capacity_evidence,
    )
    phase_timeline = bundle._timeline_for_phase(phase)
    result = run_causal_rls_replay(
        bundle.as_of_book,
        phase_timeline.decision_times,
        phase_timeline.target_ids,
        plan=plan,
        learner=learner,
        detector=detector,
        calibration=calibration,
        model_config=model_config,
        replay_config=replay_config,
        model_state=model_state,
        initial_state=initial_state,
        study_manifest=bundle.study_manifest,
        study_phase=phase,
        decision_indices=phase_timeline.decision_indices,
    )
    return StudyRLSRunResult(audit=audit, replay=result)


def run_causal_promotion_study(
    bundle: StudyInputBundle,
    phase: str,
    *,
    plan: CausalFeaturePlan,
    challenger: Any,
    incumbent: Any,
    gate: PromotionGate,
    config: CausalPromotionConfig | None = None,
    replay_config: ReplayConfig | None = None,
    model_state: Mapping[str, Any] | None = None,
    initial_state: ReplayState | None = None,
    require_complete_universe: bool = False,
    require_economic_evidence: bool = False,
    require_observed_economic_evidence: bool = False,
    require_capacity_evidence: bool = False,
) -> StudyPromotionRunResult:
    """Preflight a bundle, then run the shared paired-promotion transition."""

    if not isinstance(bundle, StudyInputBundle):
        raise StudyInputError("bundle must be a StudyInputBundle")
    audit = bundle.audit(
        phase,
        require_complete_universe=require_complete_universe,
        require_economic_evidence=require_economic_evidence,
        require_observed_economic_evidence=require_observed_economic_evidence,
        require_capacity_evidence=require_capacity_evidence,
    )
    phase_timeline = bundle._timeline_for_phase(phase)
    result = run_causal_promotion_replay(
        bundle.as_of_book,
        phase_timeline.decision_times,
        phase_timeline.target_ids,
        plan=plan,
        challenger=challenger,
        incumbent=incumbent,
        gate=gate,
        config=config,
        replay_config=replay_config,
        model_state=model_state,
        initial_state=initial_state,
        study_manifest=bundle.study_manifest,
        study_phase=phase,
        decision_indices=phase_timeline.decision_indices,
    )
    return StudyPromotionRunResult(audit=audit, replay=result)


__all__ = [
    "MAX_STUDY_TIMELINE",
    "STUDY_INPUT_SCHEMA",
    "STUDY_INPUT_VERSION",
    "StudyInputAudit",
    "StudyInputBundle",
    "StudyInputError",
    "StudyPromotionRunResult",
    "StudyRLSRunResult",
    "StudyTimeline",
    "run_causal_promotion_study",
    "run_causal_rls_study",
]
