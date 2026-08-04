"""Run the fail-closed market-source handoff audit.

The vendor adapter remains responsible for producing the canonical
``StudySourcePackage``. This command only verifies that the package and its
declared raw artifact bytes agree, then runs the strict market audit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from topology_gate import StudySourcePackage


def _artifact_path(raw_dir: Path, artifact_id: str) -> Path:
    """Resolve one declared artifact without allowing path traversal."""

    relative = Path(artifact_id)
    if (
        not artifact_id
        or relative.is_absolute()
        or relative.name != artifact_id
        or artifact_id in {".", ".."}
    ):
        raise ValueError(f"artifact ID must be a single safe filename: {artifact_id!r}")
    return raw_dir / relative


def collect_raw_payloads(package: StudySourcePackage, raw_dir: Path) -> dict[str, bytes]:
    """Read exactly the artifact IDs declared by a restored source package."""

    if not raw_dir.is_dir():
        raise FileNotFoundError(f"raw artifact directory does not exist: {raw_dir}")
    payloads: dict[str, bytes] = {}
    for artifact in package.provenance.source_artifacts:
        path = _artifact_path(raw_dir, artifact.artifact_id)
        if not path.is_file():
            raise FileNotFoundError(
                f"declared source artifact is missing: {artifact.artifact_id!r}"
            )
        payloads[artifact.artifact_id] = path.read_bytes()
    return payloads


def run_intake(
    package_path: Path,
    raw_dir: Path,
    phase: str,
    receipt_path: Path | None = None,
) -> dict[str, Any]:
    """Audit a package and return a compact, reproducible receipt summary."""

    package = StudySourcePackage.from_json(package_path.read_text(encoding="utf-8"))
    payloads = collect_raw_payloads(package, raw_dir)
    audit = package.audit_market(phase, payloads)
    if receipt_path is not None:
        receipt_path.write_text(audit.to_json() + "\n", encoding="utf-8")
    return {
        "phase": audit.phase,
        "package_digest": audit.package_digest,
        "provenance_digest": audit.provenance_digest,
        "source_audit_digest": audit.digest,
        "verified_artifact_ids": list(audit.verified_artifact_ids),
        "receipt_path": None if receipt_path is None else str(receipt_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True, type=Path, dest="package_path")
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument(
        "--phase",
        required=True,
        choices=("calibration", "tuning", "validation", "holdout"),
    )
    parser.add_argument("--receipt", type=Path, default=None, dest="receipt_path")
    args = parser.parse_args()
    print(
        json.dumps(
            run_intake(
                args.package_path,
                args.raw_dir,
                args.phase,
                args.receipt_path,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
