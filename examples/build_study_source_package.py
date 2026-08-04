"""Assemble and verify a canonical StudySourcePackage from vendor outputs.

The vendor adapter remains responsible for producing the normalized JSON
artifacts. This command binds those artifacts to one package, verifies every
declared raw byte, and writes a digest-stable package for the strict intake
command. It does not parse vendor-native files or certify the vendor's claims.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from topology_gate import (
    AsOfBook,
    EconomicEvidence,
    RunManifest,
    StudyInputBundle,
    StudyManifest,
    StudySourcePackage,
    StudySourceProvenance,
    StudyTimeline,
)


def _read(path: Path, name: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    return path.read_text(encoding="utf-8")


def _read_provenance(path: Path) -> StudySourceProvenance:
    try:
        state = json.loads(_read(path, "provenance file"))
    except json.JSONDecodeError as exc:
        raise ValueError("provenance file is not valid JSON") from exc
    return StudySourceProvenance.from_dict(state)


def _parse_cutoff(value: str) -> int | float | str:
    """Parse an explicit JSON scalar cutoff without guessing its time domain."""

    try:
        candidate = json.loads(value)
    except json.JSONDecodeError:
        candidate = value
    if isinstance(candidate, bool) or not isinstance(candidate, (int, float, str)):
        raise ValueError("economic cutoff must be a JSON integer, number, or string")
    return candidate


def _safe_artifact_path(raw_dir: Path, artifact_id: str) -> Path:
    relative = Path(artifact_id)
    if (
        not artifact_id
        or relative.is_absolute()
        or relative.name != artifact_id
        or artifact_id in {".", ".."}
    ):
        raise ValueError(f"artifact ID must be a single safe filename: {artifact_id!r}")
    return raw_dir / relative


def _raw_payloads(package: StudySourcePackage, raw_dir: Path) -> dict[str, bytes]:
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"raw artifact directory does not exist: {raw_dir}")
    payloads: dict[str, bytes] = {}
    for artifact in package.provenance.source_artifacts:
        path = _safe_artifact_path(raw_dir, artifact.artifact_id)
        if not path.is_file():
            raise FileNotFoundError(
                f"declared source artifact is missing: {artifact.artifact_id!r}"
            )
        payloads[artifact.artifact_id] = path.read_bytes()
    return payloads


def build_package(
    *,
    run_manifest_path: Path,
    study_manifest_path: Path,
    timeline_path: Path,
    as_of_book_path: Path,
    economic_evidence_path: Path,
    provenance_path: Path,
    raw_dir: Path,
    output_path: Path,
    economic_cutoff: int | float | str,
) -> dict[str, Any]:
    """Build a package and verify its exact declared raw artifacts."""

    run_manifest = RunManifest.from_json(_read(run_manifest_path, "run manifest"))
    study_manifest = StudyManifest.from_json(
        _read(study_manifest_path, "study manifest")
    )
    timeline = StudyTimeline.from_json(_read(timeline_path, "study timeline"))
    as_of_book = AsOfBook.from_json(
        _read(as_of_book_path, "as-of book"), require_digest=True
    )
    economic_evidence = EconomicEvidence.from_json(
        _read(economic_evidence_path, "economic evidence")
    )
    provenance = _read_provenance(provenance_path)
    bundle = StudyInputBundle(
        run_manifest=run_manifest,
        study_manifest=study_manifest,
        timeline=timeline,
        as_of_book=as_of_book,
        economic_evidence=economic_evidence,
        economic_cutoff=economic_cutoff,
    )
    package = StudySourcePackage(provenance=provenance, bundle=bundle)
    payloads = _raw_payloads(package, raw_dir)
    package.verify_source_artifacts(payloads)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(package.to_json() + "\n", encoding="utf-8")
    return {
        "package_digest": package.digest,
        "bundle_digest": package.bundle.digest,
        "provenance_digest": package.provenance.digest,
        "verified_artifact_ids": sorted(payloads),
        "timeline_decisions": len(package.bundle.timeline.decision_times),
        "holdout_status": package.bundle.study_manifest.holdout_status,
        "output_path": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-manifest", required=True, type=Path)
    parser.add_argument("--study-manifest", required=True, type=Path)
    parser.add_argument("--timeline", required=True, type=Path)
    parser.add_argument("--as-of-book", required=True, type=Path)
    parser.add_argument("--economic-evidence", required=True, type=Path)
    parser.add_argument("--provenance", required=True, type=Path)
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--economic-cutoff",
        required=True,
        help="explicit cutoff as a JSON scalar, e.g. 2026-08-04 or 1704067200",
    )
    args = parser.parse_args()
    args.economic_cutoff = _parse_cutoff(args.economic_cutoff)
    print(json.dumps(build_package(**vars(args)), sort_keys=True))


if __name__ == "__main__":
    main()
