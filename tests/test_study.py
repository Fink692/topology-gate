"""Tests for strict point-in-time study source preflight."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from topology_gate.asof import (
    AsOfBook,
    LabelObservation,
    MarketObservation,
    UniverseMembership,
)
from topology_gate.causal_numeric import CausalFeaturePlan, FeatureBinding
from topology_gate.causal_promotion import CausalPromotionConfig
from topology_gate.economic import EconomicEvidence, ExecutionCost, RealizedReturn
from topology_gate.manifest import (
    RunManifest,
    RunSpec,
    StudyManifest,
    StudySpec,
    StudyWindow,
)
from topology_gate.promotion import PromotionGate
from topology_gate.replay import ReplayConfig
from topology_gate.rls import RLS, RLSConfig
from topology_gate.study import (
    StudyInputBundle,
    StudyInputError,
    StudyPromotionRunResult,
    StudyTimeline,
    run_causal_promotion_study,
    run_causal_rls_study,
)
from topology_gate.study_package import (
    StudySourceArtifact,
    StudySourcePackage,
    StudySourcePackageError,
    StudySourceProvenance,
)


class _FixedLearner:
    n_features = 1

    def __init__(self, prediction: float) -> None:
        self.prediction = prediction
        self.updates = 0

    def predict(self, features: tuple[float, ...]) -> float:
        assert len(features) == 1
        return self.prediction

    def update(self, features: tuple[float, ...], target: float) -> None:
        assert len(features) == 1
        del target
        self.updates += 1

    def state_dict(self) -> dict[str, object]:
        return {"prediction": self.prediction, "updates": self.updates}

    def load_state_dict(self, state: dict[str, object]) -> None:
        if set(state) != {"prediction", "updates"}:
            raise ValueError("invalid fixed learner state")
        self.prediction = float(state["prediction"])
        self.updates = int(state["updates"])


def _run_manifest() -> RunManifest:
    return RunManifest(
        RunSpec(
            run_id="study-run",
            input_vintage_id="vendor-vintage:v1",
            universe_id="universe:v1",
            config_id="config:v1",
            backend_id="backend:v1",
            dependency_id="deps:v1",
            seed_id="seed:v1",
            thread_id="thread:v1",
        )
    )


def _study_manifest() -> StudyManifest:
    return StudyManifest(
        StudySpec(
            run_spec=_run_manifest().spec,
            feature_schema_id="features:v1",
            label_spec_id="labels:v1",
            economic_spec_id="economic:v1",
            calibration_window=StudyWindow("calibration", 0, 2),
            tuning_window=StudyWindow("tuning", 2, 4),
            validation_window=StudyWindow("validation", 4, 6),
            holdout_window=StudyWindow("holdout", 6, 8),
        )
    )


def _book(*, first_label_available: int = 2) -> AsOfBook:
    observations = tuple(
        MarketObservation(
            record_id=f"m{index}",
            instrument_id="ES",
            event_time=index,
            available_time=index,
            source_revision=0,
            ingest_sequence=0,
            fields={"x": float(index), "state": float(index)},
        )
        for index in range(1, 9)
    )
    labels = tuple(
        LabelObservation(
            label_id=f"label-t{index}",
            target_id=f"t{index}",
            event_time=(first_label_available - 1 if index == 1 else index),
            available_time=(first_label_available if index == 1 else index + 1),
            received_time=(first_label_available if index == 1 else index + 1),
            status="observed",
            value=float(index) / 10.0,
            source_revision=0,
        )
        for index in range(1, 9)
    )
    return AsOfBook(
        observations=observations,
        universe=(UniverseMembership("ES", 0, 100, 0, 0, 0),),
        labels=labels,
    )


def _economic_evidence(*, omit_target: str | None = None) -> EconomicEvidence:
    returns = tuple(
        RealizedReturn(
            target_id=f"t{index}",
            decision_time=index,
            realization_time=index + 1,
            available_time=index + 2,
            value=0.01 * index,
        )
        for index in range(1, 9)
        if f"t{index}" != omit_target
    )
    costs = tuple(
        ExecutionCost(
            target_id=f"t{index}",
            decision_time=index,
            execution_time=index,
            available_time=index,
            cost_model_id="rates:v1",
        )
        for index in range(1, 9)
        if f"t{index}" != omit_target
    )
    return EconomicEvidence(
        source_id="economic-vintage:v1",
        realized_returns=returns,
        execution_costs=costs,
    )


def _bundle(
    timeline: StudyTimeline,
    *,
    book: AsOfBook | None = None,
    manifest: StudyManifest | None = None,
    economic_evidence: EconomicEvidence | None = None,
) -> StudyInputBundle:
    return StudyInputBundle(
        run_manifest=_run_manifest(),
        study_manifest=manifest or _study_manifest(),
        timeline=timeline,
        as_of_book=book or _book(),
        economic_evidence=economic_evidence,
        economic_cutoff=10 if economic_evidence is not None else None,
    )


def _source_provenance() -> StudySourceProvenance:
    return StudySourceProvenance(
        provider_id="vendor-adapter:v1",
        dataset_id="cross-asset:v1",
        vintage_id="vendor-vintage:v1",
        license_id="test-license:v1",
        release_id="source-release:2026-08-04",
        adapter_revision="adapter-commit:test",
        as_of_rule="available_time <= decision_time",
        revision_rule="latest visible source_revision at cutoff",
        universe_rule="visible membership interval at decision time",
        delisting_rule="retain delisted instruments through final visible interval",
        source_artifacts=(
            StudySourceArtifact.from_bytes(
                "market-data.csv",
                "market observations and universe",
                b"record_id,instrument_id\nm1,ES\n",
                2,
            ),
        ),
        retrieved_at=2026_08_04,
    )


def test_timeline_normalizes_dynamic_universe_rows_and_binds_identity() -> None:
    timeline = StudyTimeline(
        decision_times=(1, 2),
        target_ids=("t1", "t2"),
        decision_indices=(0, 1),
        expected_instrument_ids=(("NQ", "ES"), ("ES", "NQ")),
    )

    assert timeline.expected_instrument_ids == (("ES", "NQ"), ("ES", "NQ"))
    assert len(timeline.digest) == 64
    assert timeline.digest == StudyTimeline(
        (1, 2),
        ("t1", "t2"),
        (0, 1),
        (("ES", "NQ"), ("ES", "NQ")),
    ).digest


@pytest.mark.parametrize(
    "kwargs",
    [
        {"decision_times": (1, 1), "target_ids": ("t1", "t2"), "decision_indices": (0, 1)},
        {"decision_times": (1, 2), "target_ids": ("t1", "t1"), "decision_indices": (0, 1)},
        {"decision_times": (1, 2), "target_ids": ("t1", "t2"), "decision_indices": (1, 0)},
        {"decision_times": (1,), "target_ids": ("t1", "t2"), "decision_indices": (0,)},
    ],
)
def test_timeline_rejects_ambiguous_or_misaligned_order(kwargs: dict[str, object]) -> None:
    with pytest.raises(StudyInputError):
        StudyTimeline(**kwargs)


def test_bundle_rejects_a_study_manifest_from_a_different_run() -> None:
    mismatched = RunManifest(
        RunSpec(
            run_id="different-study-run",
            input_vintage_id="vendor-vintage:v1",
            universe_id="universe:v1",
            config_id="config:v1",
            backend_id="backend:v1",
            dependency_id="deps:v1",
            seed_id="seed:v1",
            thread_id="thread:v1",
        )
    )

    with pytest.raises(StudyInputError, match="run specification"):
        StudyInputBundle(
            run_manifest=mismatched,
            study_manifest=_study_manifest(),
            timeline=StudyTimeline((1,), ("t1",), (0,), (("ES",),)),
            as_of_book=_book(),
        )


def test_bundle_audit_requires_complete_universe_and_economic_records() -> None:
    timeline = StudyTimeline(
        decision_times=(1, 2),
        target_ids=("t1", "t2"),
        decision_indices=(0, 1),
        expected_instrument_ids=(("ES",), ("ES",)),
    )
    bundle = _bundle(timeline, economic_evidence=_economic_evidence())

    audit = bundle.audit(
        "calibration",
        require_complete_universe=True,
        require_observed_economic_evidence=True,
    )

    assert audit.decision_count == 2
    assert audit.expected_universe_complete is True
    assert audit.economic_records_complete is True
    assert audit.economic_evidence_digest == bundle.economic_evidence.digest
    assert audit.bundle_digest == bundle.digest


def test_bundle_rejects_universe_mismatch_before_model_execution() -> None:
    timeline = StudyTimeline(
        decision_times=(1,),
        target_ids=("t1",),
        decision_indices=(0,),
        expected_instrument_ids=(("NQ",),),
    )

    with pytest.raises(StudyInputError, match="universe mismatch"):
        _bundle(timeline).audit("calibration", require_complete_universe=True)


def test_bundle_rejects_target_label_visible_at_decision_boundary() -> None:
    timeline = StudyTimeline((1,), ("t1",), (0,), (("ES",),))

    with pytest.raises(StudyInputError, match="already visible"):
        _bundle(timeline, book=_book(first_label_available=1)).audit("calibration")


def test_sealed_holdout_cannot_be_read_through_the_study_bundle() -> None:
    timeline = StudyTimeline((7,), ("t7",), (6,), (("ES",),))
    bundle = _bundle(timeline)

    with pytest.raises(StudyInputError, match="sealed study holdout"):
        bundle.audit("holdout", require_complete_universe=True)

    opened = replace(bundle, study_manifest=bundle.study_manifest.open_holdout("release-1"))
    audit = opened.audit("holdout", require_complete_universe=True)
    assert audit.holdout_status == "opened"


def test_bundle_rejects_incomplete_economic_evidence() -> None:
    timeline = StudyTimeline(
        decision_times=(1, 2),
        target_ids=("t1", "t2"),
        decision_indices=(0, 1),
        expected_instrument_ids=(("ES",), ("ES",)),
    )
    bundle = _bundle(
        timeline,
        economic_evidence=_economic_evidence(omit_target="t2"),
    )

    with pytest.raises(StudyInputError, match="economic evidence is incomplete"):
        bundle.audit("calibration", require_economic_evidence=True)


def test_causal_study_wrapper_returns_preflight_receipt_and_economic_decisions() -> None:
    timeline = StudyTimeline(
        decision_times=(1, 2),
        target_ids=("t1", "t2"),
        decision_indices=(0, 1),
        expected_instrument_ids=(("ES",), ("ES",)),
    )
    bundle = _bundle(timeline, economic_evidence=_economic_evidence())
    plan = CausalFeaturePlan(
        {
            "t1": (FeatureBinding("m1", ("x",), "ES"),),
            "t2": (FeatureBinding("m2", ("x",), "ES"),),
        },
        state_bindings_by_target={
            "t1": (FeatureBinding("m1", ("state",), "ES"),),
            "t2": (FeatureBinding("m2", ("state",), "ES"),),
        },
        require_membership=True,
    )

    result = run_causal_rls_study(
        bundle,
        "calibration",
        plan=plan,
        learner=RLS(RLSConfig(n_features=1, lambda_min=0.8, lambda_max=1.0)),
        require_complete_universe=True,
        require_observed_economic_evidence=True,
    )

    assert result.audit.phase == "calibration"
    assert result.replay.study_manifest_digest == bundle.study_manifest.digest
    decisions = result.economic_decisions
    assert tuple(item.target_id for item in decisions) == ("t1", "t2")
    assert all(item.evaluated for item in decisions)


def test_promotion_study_wrapper_reuses_the_same_source_preflight() -> None:
    timeline = StudyTimeline(
        decision_times=(1, 2),
        target_ids=("t1", "t2"),
        decision_indices=(0, 1),
        expected_instrument_ids=(("ES",), ("ES",)),
    )
    bundle = _bundle(timeline)
    plan = CausalFeaturePlan(
        {
            "t1": (FeatureBinding("m1", ("x",), "ES"),),
            "t2": (FeatureBinding("m2", ("x",), "ES"),),
        },
        require_membership=True,
    )
    gate = PromotionGate("incumbent", alpha=0.9, eta=0.5)
    gate.register_challenger("challenger")
    gate.seal_registration()

    result = run_causal_promotion_study(
        bundle,
        "calibration",
        plan=plan,
        challenger=_FixedLearner(0.0),
        incumbent=_FixedLearner(1.0),
        gate=gate,
        config=CausalPromotionConfig(
            promotion_id="study-paired",
            challenger_id="challenger",
            incumbent_id="incumbent",
            eta=0.5,
            utility_cap=1.0,
        ),
        replay_config=ReplayConfig(
            model_id="study-paired",
            score_id="none",
            require_model_state=True,
        ),
        require_complete_universe=True,
    )

    assert isinstance(result, StudyPromotionRunResult)
    assert result.audit.expected_universe_complete is True
    assert result.replay.study_manifest_digest == bundle.study_manifest.digest
    assert len(result.replay.steps) == 2


def test_study_bundle_digest_binds_source_artifact_revisions() -> None:
    timeline = StudyTimeline((1,), ("t1",), (0,), (("ES",),))
    first = _bundle(timeline, economic_evidence=_economic_evidence())
    changed = replace(first, as_of_book=_book(first_label_available=3))

    assert first.digest != changed.digest


def test_timeline_json_round_trip_preserves_tagged_time_identity() -> None:
    timeline = StudyTimeline(
        decision_times=(1.0, 2.5, 3.0),
        target_ids=("t1", "t2", "t3"),
        decision_indices=(0, 1, 2),
        expected_instrument_ids=(("ES",), ("ES",), ("ES",)),
    )

    restored = StudyTimeline.from_json(json.dumps(timeline.to_dict()))

    assert restored.digest == timeline.digest
    assert restored.decision_times == timeline.decision_times


def test_source_package_round_trip_binds_provenance_and_all_artifacts() -> None:
    timeline = StudyTimeline((1, 2), ("t1", "t2"), (0, 1), (("ES",), ("ES",)))
    bundle = _bundle(timeline, economic_evidence=_economic_evidence())
    package = StudySourcePackage(_source_provenance(), bundle)

    restored = StudySourcePackage.from_json(package.to_json())

    assert restored.digest == package.digest
    assert restored.bundle.digest == bundle.digest
    assert restored.provenance.digest == package.provenance.digest
    restored.verify_source_artifact(
        "market-data.csv",
        b"record_id,instrument_id\nm1,ES\n",
    )
    restored.verify_source_artifacts(
        {"market-data.csv": b"record_id,instrument_id\nm1,ES\n"}
    )
    with pytest.raises(StudySourcePackageError, match="byte size|sha256"):
        restored.verify_source_artifact("market-data.csv", b"tampered")
    with pytest.raises(StudySourcePackageError, match="missing"):
        restored.verify_source_artifacts({})
    with pytest.raises(StudySourcePackageError, match="unexpected"):
        restored.verify_source_artifacts(
            {
                "market-data.csv": b"record_id,instrument_id\nm1,ES\n",
                "extra.csv": b"unexpected",
            }
        )
    assert restored.audit("calibration").decision_count == 2


def test_source_package_rejects_tampered_artifacts_and_unknown_fields() -> None:
    package = StudySourcePackage(
        _source_provenance(),
        _bundle(StudyTimeline((1,), ("t1",), (0,), (("ES",),))),
    )
    state = json.loads(package.to_json())
    state["as_of_book"]["observations"][0]["fields"]["x"] = 999.0

    with pytest.raises(StudySourcePackageError, match="as-of book"):
        StudySourcePackage.from_dict(state)

    unknown = json.loads(package.to_json())
    unknown["unexpected"] = True
    with pytest.raises(StudySourcePackageError, match="unknown or missing"):
        StudySourcePackage.from_dict(unknown)


def test_source_package_rejects_schema_and_provenance_digest_tampering() -> None:
    package = StudySourcePackage(
        _source_provenance(),
        _bundle(StudyTimeline((1,), ("t1",), (0,), (("ES",),))),
    )

    wrong_schema = json.loads(package.to_json())
    wrong_schema["schema"] = "other"
    with pytest.raises(StudySourcePackageError, match="schema"):
        StudySourcePackage.from_dict(wrong_schema)

    wrong_provenance = json.loads(package.to_json())
    wrong_provenance["provenance"]["provider_id"] = "tampered"
    with pytest.raises(StudySourcePackageError, match="provenance digest"):
        StudySourcePackage.from_dict(wrong_provenance)

    wrong_artifact = json.loads(package.to_json())
    wrong_artifact["provenance"]["source_artifacts"][0]["sha256"] = "0" * 64
    with pytest.raises(StudySourcePackageError, match="source artifact digest"):
        StudySourcePackage.from_dict(wrong_artifact)
