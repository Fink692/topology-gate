"""Normalize the strict six-role JSONL handoff into a source package.

This is a canonical boundary adapter, not a parser for CRSP, Bloomberg, or
any other vendor-native format. A vendor-specific mapper must first emit the
exact JSONL fields documented in ``docs/vendor-handoff-template.md``. Unknown
fields, missing fields, blank records, duplicate revisions, and invalid causal
timestamps fail closed before a package is written.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from topology_gate import (
    AsOfBook,
    EconomicEvidence,
    ExecutionCost,
    LabelObservation,
    MarketObservation,
    RealizedReturn,
    RunManifest,
    StudyInputBundle,
    StudyManifest,
    StudySourceArtifact,
    StudySourcePackage,
    StudySourceProvenance,
    StudyTimeline,
    UniverseMembership,
)


class VendorHandoffError(ValueError):
    """Raised when canonical role files cannot be normalized safely."""


ROLE_FILES: tuple[tuple[str, str], ...] = (
    ("delistings", "delistings.jsonl"),
    ("execution-costs", "execution-costs.jsonl"),
    ("labels", "labels.jsonl"),
    ("market-observations", "market-observations.jsonl"),
    ("realized-returns", "realized-returns.jsonl"),
    ("universe-membership", "universe-membership.jsonl"),
)

PROVENANCE_FIELDS = {
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
}

ROLE_FIELDS: dict[str, set[str]] = {
    "market-observations": {
        "record_id",
        "instrument_id",
        "event_time",
        "available_time",
        "source_revision",
        "ingest_sequence",
        "fields",
    },
    "universe-membership": {
        "instrument_id",
        "start",
        "end",
        "event_time",
        "available_time",
        "source_revision",
        "ingest_sequence",
    },
    "labels": {
        "label_id",
        "target_id",
        "event_time",
        "available_time",
        "received_time",
        "status",
        "value",
        "source_revision",
        "ingest_sequence",
    },
    "realized-returns": {
        "target_id",
        "decision_time",
        "realization_time",
        "available_time",
        "value",
        "source_revision",
        "status",
    },
    "execution-costs": {
        "target_id",
        "decision_time",
        "execution_time",
        "available_time",
        "cost_model_id",
        "fee_rate",
        "spread_rate",
        "slippage_rate",
        "impact_rate",
        "other_rate",
        "source_revision",
        "capacity_limit",
    },
    "delistings": {
        "instrument_id",
        "event_time",
        "available_time",
        "source_revision",
        "ingest_sequence",
        "delisting_time",
        "delisting_return",
        "delisting_reason",
    },
}


def _exact_row(row: Mapping[str, Any], role: str, line_number: int) -> dict[str, Any]:
    expected = ROLE_FIELDS[role]
    if set(row) != expected:
        missing = sorted(expected - set(row))
        extra = sorted(set(row) - expected)
        raise VendorHandoffError(
            f"{role} line {line_number} has invalid fields "
            f"(missing={missing!r}, extra={extra!r})"
        )
    return dict(row)


def _read_jsonl(raw_dir: Path, role: str, filename: str) -> tuple[bytes, tuple[dict[str, Any], ...]]:
    path = raw_dir / filename
    if not path.is_file():
        raise VendorHandoffError(f"missing required {role} artifact: {filename!r}")
    payload = path.read_bytes()
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise VendorHandoffError(f"{filename!r} is not UTF-8 JSONL") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            raise VendorHandoffError(f"{filename!r} contains a blank line at {line_number}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise VendorHandoffError(
                f"{filename!r} line {line_number} is not valid JSON"
            ) from exc
        if not isinstance(value, Mapping):
            raise VendorHandoffError(
                f"{filename!r} line {line_number} must be a JSON object"
            )
        rows.append(_exact_row(value, role, line_number))
    if not rows:
        raise VendorHandoffError(f"{filename!r} must contain at least one record")
    return payload, tuple(rows)


def _finite_or_none(value: Any, name: str) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        raise VendorHandoffError(f"{name} must be finite or null")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise VendorHandoffError(f"{name} must be finite or null") from exc
    if not math.isfinite(numeric):
        raise VendorHandoffError(f"{name} must be finite or null")


def _decode_time(value: Any, name: str) -> Any:
    """Decode canonical primitive or explicitly tagged time values."""

    if isinstance(value, Mapping):
        if set(value) != {"kind", "value"}:
            raise VendorHandoffError(f"{name} has an invalid tagged time")
        kind = value["kind"]
        raw = value["value"]
        if kind == "datetime":
            if not isinstance(raw, str) or not raw:
                raise VendorHandoffError(f"{name} datetime value is invalid")
            try:
                return datetime.fromisoformat(raw)
            except ValueError as exc:
                raise VendorHandoffError(f"{name} datetime value is invalid") from exc
        if kind == "int":
            if isinstance(raw, bool) or not isinstance(raw, int):
                raise VendorHandoffError(f"{name} integer value is invalid")
            return raw
        if kind == "float":
            if isinstance(raw, bool) or not isinstance(raw, float) or not math.isfinite(raw):
                raise VendorHandoffError(f"{name} float value is invalid")
            return raw
        if kind == "str":
            if not isinstance(raw, str) or not raw:
                raise VendorHandoffError(f"{name} string value is invalid")
            return raw
        raise VendorHandoffError(f"{name} has an unsupported tagged time kind")
    if isinstance(value, bool) or value is None:
        raise VendorHandoffError(f"{name} must be a time point")
    if isinstance(value, float) and not math.isfinite(value):
        raise VendorHandoffError(f"{name} must be finite")
    if not isinstance(value, (int, float, str)) or value == "":
        raise VendorHandoffError(f"{name} must be a time point")
    return value


def _decode_record_times(
    row: dict[str, Any], role: str, line_number: int, names: tuple[str, ...]
) -> dict[str, Any]:
    decoded = dict(row)
    for name in names:
        if decoded[name] is not None:
            decoded[name] = _decode_time(decoded[name], f"{role} line {line_number} {name}")
    return decoded


def _validate_delistings(rows: tuple[dict[str, Any], ...]) -> None:
    seen: set[tuple[str, Any, int]] = set()
    for index, row in enumerate(rows, 1):
        decoded = _decode_record_times(
            row,
            "delistings",
            index,
            ("event_time", "available_time", "delisting_time"),
        )
        instrument_id = row["instrument_id"]
        reason = row["delisting_reason"]
        if not isinstance(instrument_id, str) or not instrument_id.strip():
            raise VendorHandoffError(
                f"delistings line {index} instrument_id must be non-empty"
            )
        if not isinstance(reason, str) or not reason.strip():
            raise VendorHandoffError(
                f"delistings line {index} delisting_reason must be non-empty"
            )
        for name in ("source_revision", "ingest_sequence"):
            value = row[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise VendorHandoffError(
                    f"delistings line {index} {name} must be a non-negative integer"
                )
        try:
            if decoded["event_time"] > decoded["available_time"]:
                raise VendorHandoffError(
                    f"delistings line {index} event_time cannot follow available_time"
                )
            if (
                decoded["delisting_time"] is not None
                and decoded["delisting_time"] > decoded["available_time"]
            ):
                raise VendorHandoffError(
                    f"delistings line {index} delisting_time cannot follow available_time"
                )
        except TypeError as exc:
            raise VendorHandoffError(
                f"delistings line {index} times use different domains"
            ) from exc
        key = (instrument_id, decoded["event_time"], row["source_revision"])
        if key in seen:
            raise VendorHandoffError(
                f"delistings line {index} duplicates a source revision"
            )
        seen.add(key)
        _finite_or_none(row["delisting_return"], f"delistings line {index} delisting_return")


def _load_provenance(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise VendorHandoffError(f"provenance file does not exist: {path}")
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise VendorHandoffError("provenance file is not valid JSON") from exc
    if not isinstance(state, Mapping) or set(state) != PROVENANCE_FIELDS:
        raise VendorHandoffError(
            "provenance metadata must contain exactly "
            f"{sorted(PROVENANCE_FIELDS)!r}"
        )
    return dict(state)


def _parse_cutoff(value: str) -> int | float | str:
    try:
        candidate = json.loads(value)
    except json.JSONDecodeError:
        candidate = value
    if isinstance(candidate, bool) or not isinstance(candidate, (int, float, str)):
        raise VendorHandoffError("economic cutoff must be a JSON integer, number, or string")
    return candidate


def normalize_handoff(
    *,
    raw_dir: Path,
    run_manifest_path: Path,
    study_manifest_path: Path,
    timeline_path: Path,
    provenance_path: Path,
    economic_cutoff: int | float | str,
    output_path: Path,
) -> dict[str, Any]:
    """Parse canonical role JSONL and write one digest-bound source package."""

    if not raw_dir.is_dir():
        raise VendorHandoffError(f"raw handoff directory does not exist: {raw_dir}")
    expected_files = {filename for _, filename in ROLE_FILES}
    actual_files = {path.name for path in raw_dir.iterdir() if path.is_file()}
    if actual_files != expected_files:
        raise VendorHandoffError(
            "raw handoff directory must contain exactly the six role files "
            f"(missing={sorted(expected_files - actual_files)!r}, "
            f"extra={sorted(actual_files - expected_files)!r})"
        )

    payloads: dict[str, bytes] = {}
    rows_by_role: dict[str, tuple[dict[str, Any], ...]] = {}
    for role, filename in ROLE_FILES:
        payload, rows = _read_jsonl(raw_dir, role, filename)
        payloads[filename] = payload
        rows_by_role[role] = rows

    _validate_delistings(rows_by_role["delistings"])
    metadata = _load_provenance(provenance_path)
    try:
        as_of_book = AsOfBook(
            observations=tuple(
                MarketObservation(
                    **_decode_record_times(
                        row,
                        "market-observations",
                        index,
                        ("event_time", "available_time"),
                    )
                )
                for index, row in enumerate(rows_by_role["market-observations"], 1)
            ),
            universe=tuple(
                UniverseMembership(
                    **_decode_record_times(
                        row,
                        "universe-membership",
                        index,
                        ("start", "end", "event_time", "available_time"),
                    )
                )
                for index, row in enumerate(rows_by_role["universe-membership"], 1)
            ),
            labels=tuple(
                LabelObservation(
                    **_decode_record_times(
                        row,
                        "labels",
                        index,
                        ("event_time", "available_time", "received_time"),
                    )
                )
                for index, row in enumerate(rows_by_role["labels"], 1)
            ),
        )
        economic_evidence = EconomicEvidence(
            source_id=f"{metadata['dataset_id']}:{metadata['vintage_id']}",
            realized_returns=tuple(
                RealizedReturn(
                    **_decode_record_times(
                        row,
                        "realized-returns",
                        index,
                        ("decision_time", "realization_time", "available_time"),
                    )
                )
                for index, row in enumerate(rows_by_role["realized-returns"], 1)
            ),
            execution_costs=tuple(
                ExecutionCost(
                    **_decode_record_times(
                        row,
                        "execution-costs",
                        index,
                        ("decision_time", "execution_time", "available_time"),
                    )
                )
                for index, row in enumerate(rows_by_role["execution-costs"], 1)
            ),
        )
        run_manifest = RunManifest.from_json(
            run_manifest_path.read_text(encoding="utf-8")
        )
        study_manifest = StudyManifest.from_json(
            study_manifest_path.read_text(encoding="utf-8")
        )
        timeline = StudyTimeline.from_json(timeline_path.read_text(encoding="utf-8"))
        provenance = StudySourceProvenance(
            **{
                **metadata,
                "retrieved_at": _decode_time(
                    metadata["retrieved_at"], "provenance retrieved_at"
                ),
            },
            source_artifacts=tuple(
                StudySourceArtifact.from_bytes(
                    artifact_id=filename,
                    role=role,
                    payload=payloads[filename],
                    record_count=len(rows_by_role[role]),
                )
                for role, filename in ROLE_FILES
            ),
        )
        package = StudySourcePackage(
            provenance=provenance,
            bundle=StudyInputBundle(
                run_manifest=run_manifest,
                study_manifest=study_manifest,
                timeline=timeline,
                as_of_book=as_of_book,
                economic_evidence=economic_evidence,
                economic_cutoff=economic_cutoff,
            ),
        )
        package.verify_source_artifacts(payloads)
    except (OSError, TypeError, ValueError) as exc:
        if isinstance(exc, VendorHandoffError):
            raise
        raise VendorHandoffError(f"canonical handoff is invalid: {exc}") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(package.to_json() + "\n", encoding="utf-8")
    return {
        "package_digest": package.digest,
        "bundle_digest": package.bundle.digest,
        "provenance_digest": package.provenance.digest,
        "artifact_ids": sorted(payloads),
        "artifact_roles": [role for role, _ in ROLE_FILES],
        "timeline_decisions": len(package.bundle.timeline.decision_times),
        "holdout_status": package.bundle.study_manifest.holdout_status,
        "output_path": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--run-manifest", required=True, type=Path)
    parser.add_argument("--study-manifest", required=True, type=Path)
    parser.add_argument("--timeline", required=True, type=Path)
    parser.add_argument("--provenance", required=True, type=Path)
    parser.add_argument("--economic-cutoff", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.economic_cutoff = _parse_cutoff(args.economic_cutoff)
    print(json.dumps(normalize_handoff(**vars(args)), sort_keys=True))


if __name__ == "__main__":
    main()
