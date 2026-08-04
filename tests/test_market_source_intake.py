"""Tests for the filesystem market-source intake bridge."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from examples.market_source_intake import (
    _artifact_path,
    collect_raw_payloads,
    run_all_pre_holdout_intake,
)


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
