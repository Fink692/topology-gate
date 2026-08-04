"""Tests for the filesystem market-source intake bridge."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from examples.market_source_intake import _artifact_path, collect_raw_payloads


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
