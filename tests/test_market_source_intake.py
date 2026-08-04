"""Tests for the filesystem market-source intake bridge."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from examples.build_study_source_package import build_package
from examples.market_source_intake import (
    _artifact_path,
    collect_raw_payloads,
    run_all_pre_holdout_intake,
)
from topology_gate import (
    AsOfBook,
    EconomicEvidence,
    ExecutionCost,
    LabelObservation,
    MarketObservation,
    RealizedReturn,
    RunManifest,
    RunSpec,
    StudyInputBundle,
    StudyManifest,
    StudySourceArtifact,
    StudySourcePackage,
    StudySourceProvenance,
    StudySpec,
    StudyTimeline,
    StudyWindow,
    UniverseMembership,
)


def _minimal_source_package() -> StudySourcePackage:
    run = RunManifest(
        RunSpec(
            run_id="builder-test",
            input_vintage_id="test-vintage:v1",
            universe_id="test-universe:v1",
            config_id="test-config:v1",
            backend_id="test-backend:v1",
            dependency_id="test-deps:v1",
            seed_id="test-seed:v1",
            thread_id="test-thread:v1",
        )
    )
    study = StudyManifest(
        StudySpec(
            run_spec=run.spec,
            feature_schema_id="features:v1",
            label_spec_id="labels:v1",
            economic_spec_id="economic:v1",
            calibration_window=StudyWindow("calibration", 0, 1),
            tuning_window=StudyWindow("tuning", 1, 2),
            validation_window=StudyWindow("validation", 2, 3),
            holdout_window=StudyWindow("holdout", 3, 4),
        )
    )
    timeline = StudyTimeline((1,), ("t1",), (0,), (("ES",),))
    book = AsOfBook(
        observations=(
            MarketObservation(
                record_id="m1",
                instrument_id="ES",
                event_time=1,
                available_time=1,
                source_revision=0,
                ingest_sequence=0,
                fields={"x": 1.0},
            ),
        ),
        universe=(UniverseMembership("ES", 0, 10, 0, 0, 0),),
        labels=(
            LabelObservation(
                label_id="l1",
                target_id="t1",
                event_time=2,
                available_time=2,
                received_time=2,
                status="observed",
                value=0.01,
                source_revision=0,
                ingest_sequence=0,
            ),
        ),
    )
    evidence = EconomicEvidence(
        source_id="economic:v1",
        realized_returns=(
            RealizedReturn("t1", 1, 2, 2, 0.01),
        ),
        execution_costs=(
            ExecutionCost("t1", 1, 1, 1, "cost:v1", capacity_limit=1.0),
        ),
    )
    payload = b"market\n"
    provenance = StudySourceProvenance(
        provider_id="test-provider:v1",
        dataset_id="test-dataset:v1",
        vintage_id="test-vintage:v1",
        license_id="test-license:v1",
        release_id="test-release:v1",
        adapter_revision="test-adapter:v1",
        as_of_rule="available_time <= decision_time",
        revision_rule="latest visible revision",
        universe_rule="visible membership interval",
        delisting_rule="retain through final visible interval",
        source_artifacts=(
            StudySourceArtifact.from_bytes("market.csv", "market-observations", payload, 1),
        ),
        retrieved_at=3,
    )
    return StudySourcePackage(
        provenance,
        StudyInputBundle(
            run,
            study,
            timeline,
            book,
            evidence,
            economic_cutoff=1,
        ),
    )


def test_builder_binds_canonical_artifacts_and_verifies_raw_bytes(tmp_path) -> None:
    package = _minimal_source_package()
    run_manifest_path = tmp_path / "run.json"
    study_manifest_path = tmp_path / "study.json"
    timeline_path = tmp_path / "timeline.json"
    book_path = tmp_path / "book.json"
    evidence_path = tmp_path / "evidence.json"
    provenance_path = tmp_path / "provenance.json"
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "market.csv").write_bytes(b"market\n")
    run_manifest_path.write_text(package.bundle.run_manifest.to_json(), encoding="utf-8")
    study_manifest_path.write_text(package.bundle.study_manifest.to_json(), encoding="utf-8")
    timeline_path.write_text(
        json.dumps(package.bundle.timeline.to_dict()), encoding="utf-8"
    )
    book_path.write_text(package.bundle.as_of_book.to_json(), encoding="utf-8")
    evidence_path.write_text(package.bundle.economic_evidence.to_json(), encoding="utf-8")
    provenance_path.write_text(json.dumps(package.provenance.to_dict()), encoding="utf-8")
    output_path = tmp_path / "out" / "package.json"

    summary = build_package(
        run_manifest_path=run_manifest_path,
        study_manifest_path=study_manifest_path,
        timeline_path=timeline_path,
        as_of_book_path=book_path,
        economic_evidence_path=evidence_path,
        provenance_path=provenance_path,
        raw_dir=raw_dir,
        output_path=output_path,
        economic_cutoff=1,
    )

    restored = StudySourcePackage.from_json(output_path.read_text(encoding="utf-8"))
    assert summary["package_digest"] == package.digest == restored.digest
    assert summary["verified_artifact_ids"] == ["market.csv"]


def test_builder_rejects_tampered_declared_raw_bytes(tmp_path) -> None:
    package = _minimal_source_package()
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "market.csv").write_bytes(b"tampered\n")
    paths = {
        "run_manifest_path": tmp_path / "run.json",
        "study_manifest_path": tmp_path / "study.json",
        "timeline_path": tmp_path / "timeline.json",
        "as_of_book_path": tmp_path / "book.json",
        "economic_evidence_path": tmp_path / "evidence.json",
        "provenance_path": tmp_path / "provenance.json",
        "raw_dir": raw_dir,
        "output_path": tmp_path / "package.json",
        "economic_cutoff": 1,
    }
    paths["run_manifest_path"].write_text(package.bundle.run_manifest.to_json(), encoding="utf-8")
    paths["study_manifest_path"].write_text(package.bundle.study_manifest.to_json(), encoding="utf-8")
    paths["timeline_path"].write_text(
        json.dumps(package.bundle.timeline.to_dict()), encoding="utf-8"
    )
    paths["as_of_book_path"].write_text(package.bundle.as_of_book.to_json(), encoding="utf-8")
    paths["economic_evidence_path"].write_text(package.bundle.economic_evidence.to_json(), encoding="utf-8")
    paths["provenance_path"].write_text(json.dumps(package.provenance.to_dict()), encoding="utf-8")

    with pytest.raises(ValueError, match="sha256|byte size"):
        build_package(**paths)


def test_artifact_path_rejects_traversal_and_absolute_names(tmp_path) -> None:
    with pytest.raises(ValueError, match="safe filename"):
        _artifact_path(tmp_path, "../escape.csv")
    with pytest.raises(ValueError, match="safe filename"):
        _artifact_path(tmp_path, str(tmp_path / "absolute.csv"))


def test_collect_raw_payloads_reads_exact_declared_artifacts(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "market.csv").write_bytes(b"market")
    package = SimpleNamespace(
        provenance=SimpleNamespace(
            source_artifacts=(SimpleNamespace(artifact_id="market.csv"),)
        )
    )
    assert collect_raw_payloads(package, raw_dir) == {"market.csv": b"market"}


def test_collect_raw_payloads_fails_closed_on_missing_directory(tmp_path) -> None:
    package = SimpleNamespace(provenance=SimpleNamespace(source_artifacts=()))
    with pytest.raises(FileNotFoundError, match="does not exist"):
        collect_raw_payloads(package, tmp_path / "missing")


def test_all_pre_holdout_intake_audits_three_phases_and_never_opens_holdout(
    tmp_path, monkeypatch
) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "market.csv").write_bytes(b"market")
    observed_phases: list[str] = []

    class FakePackage:
        provenance = SimpleNamespace(
            source_artifacts=(SimpleNamespace(artifact_id="market.csv"),)
        )

        @classmethod
        def from_json(cls, payload: str) -> "FakePackage":
            assert payload == "{}"
            return cls()

        def audit_market(self, phase: str, payloads: dict[str, bytes]) -> SimpleNamespace:
            observed_phases.append(phase)
            assert payloads == {"market.csv": b"market"}
            return SimpleNamespace(
                phase=phase,
                package_digest="p" * 64,
                provenance_digest="v" * 64,
                digest=f"{phase[0]}" * 64,
                verified_artifact_ids=("market.csv",),
                to_json=lambda: "{}",
            )

    monkeypatch.setattr("examples.market_source_intake.StudySourcePackage", FakePackage)
    package_path = tmp_path / "package.json"
    package_path.write_text("{}", encoding="utf-8")
    receipt_dir = tmp_path / "receipts"

    result = run_all_pre_holdout_intake(package_path, raw_dir, receipt_dir)

    assert observed_phases == ["calibration", "tuning", "validation"]
    assert result["audited_phases"] == observed_phases
    assert result["holdout_opened"] is False
    assert sorted(path.name for path in receipt_dir.iterdir()) == [
        "calibration-source-audit.json",
        "tuning-source-audit.json",
        "validation-source-audit.json",
    ]
