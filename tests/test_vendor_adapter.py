"""Tests for the strict canonical JSONL vendor handoff adapter."""

from __future__ import annotations

import json

import pytest

from examples.normalize_vendor_handoff import (
    ROLE_FILES,
    VendorHandoffError,
    normalize_handoff,
)
from examples.synthetic_market_handoff import build_synthetic_package
from topology_gate import StudySourcePackage


def _write_canonical_handoff(tmp_path):
    package, _ = build_synthetic_package()
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    book = package.bundle.as_of_book
    evidence = package.bundle.economic_evidence
    assert evidence is not None
    rows = {
        "market-observations": [item.to_dict() for item in book.observations],
        "universe-membership": [item.to_dict() for item in book.universe],
        "labels": [item.to_dict() for item in book.labels],
        "realized-returns": [item.to_dict() for item in evidence.realized_returns],
        "execution-costs": [item.to_dict() for item in evidence.execution_costs],
        "delistings": [
            {
                "instrument_id": "SYNTH",
                "event_time": 7,
                "available_time": 7,
                "source_revision": 0,
                "ingest_sequence": 7,
                "delisting_time": 7,
                "delisting_return": -1.0,
                "delisting_reason": "synthetic_end_of_sample",
            }
        ],
    }
    for role, filename in ROLE_FILES:
        payload = "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows[role]
        )
        (raw_dir / filename).write_text(payload, encoding="utf-8")

    run_path = tmp_path / "run.json"
    study_path = tmp_path / "study.json"
    timeline_path = tmp_path / "timeline.json"
    provenance_path = tmp_path / "provenance.json"
    run_path.write_text(package.bundle.run_manifest.to_json(), encoding="utf-8")
    study_path.write_text(package.bundle.study_manifest.to_json(), encoding="utf-8")
    timeline_path.write_text(
        json.dumps(package.bundle.timeline.to_dict()), encoding="utf-8"
    )
    metadata = {
        name: getattr(package.provenance, name)
        for name in (
            "provider_id",
            "dataset_id",
            "vintage_id",
            "license_id",
            "release_id",
            "adapter_revision",
            "as_of_rule",
            "revision_rule",
            "universe_rule",
            "delisting_rule",
            "retrieved_at",
        )
    }
    provenance_path.write_text(json.dumps(metadata), encoding="utf-8")
    return raw_dir, run_path, study_path, timeline_path, provenance_path


def test_normalize_handoff_builds_auditable_package_from_canonical_jsonl(tmp_path) -> None:
    paths = _write_canonical_handoff(tmp_path)
    output_path = tmp_path / "out" / "study-source-package.json"

    result = normalize_handoff(
        raw_dir=paths[0],
        run_manifest_path=paths[1],
        study_manifest_path=paths[2],
        timeline_path=paths[3],
        provenance_path=paths[4],
        economic_cutoff=10,
        output_path=output_path,
    )

    package = StudySourcePackage.from_json(output_path.read_text(encoding="utf-8"))
    payloads = {
        filename: (paths[0] / filename).read_bytes()
        for _, filename in ROLE_FILES
    }
    audit = package.audit_market("validation", payloads)
    assert result["package_digest"] == package.digest
    assert audit.input_audit.decision_count == 2
    assert audit.input_audit.capacity_evidence_complete is True


def test_normalize_handoff_rejects_unknown_canonical_fields(tmp_path) -> None:
    paths = _write_canonical_handoff(tmp_path)
    path = paths[0] / "market-observations.jsonl"
    row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    row["vendor_private_field"] = "must be mapped explicitly"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(VendorHandoffError, match="invalid fields"):
        normalize_handoff(
            raw_dir=paths[0],
            run_manifest_path=paths[1],
            study_manifest_path=paths[2],
            timeline_path=paths[3],
            provenance_path=paths[4],
            economic_cutoff=10,
            output_path=tmp_path / "package.json",
        )


def test_normalize_handoff_rejects_extra_raw_files(tmp_path) -> None:
    paths = _write_canonical_handoff(tmp_path)
    (paths[0] / "unmapped-vendor-export.csv").write_text("x\n", encoding="utf-8")

    with pytest.raises(VendorHandoffError, match="exactly the six role files"):
        normalize_handoff(
            raw_dir=paths[0],
            run_manifest_path=paths[1],
            study_manifest_path=paths[2],
            timeline_path=paths[3],
            provenance_path=paths[4],
            economic_cutoff=10,
            output_path=tmp_path / "package.json",
        )
