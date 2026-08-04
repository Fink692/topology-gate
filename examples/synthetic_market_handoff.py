"""Exercise the complete source-handoff gate with deterministic synthetic data.

This is a protocol/integration diagnostic only.  It deliberately does not
pretend that synthetic records establish point-in-time market coverage,
delisting treatment, or economic performance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

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

try:
    from .market_source_intake import run_all_pre_holdout_intake
except ImportError:  # pragma: no cover - direct script execution
    from market_source_intake import run_all_pre_holdout_intake


ROLES = (
    "delistings",
    "execution-costs",
    "labels",
    "market-observations",
    "realized-returns",
    "universe-membership",
)


def build_synthetic_package() -> tuple[StudySourcePackage, dict[str, bytes]]:
    """Build a complete, but explicitly synthetic, six-role source package."""

    run = RunManifest(
        RunSpec(
            run_id="synthetic-handoff-run:v1",
            input_vintage_id="synthetic-vintage:2026-08-04",
            universe_id="synthetic-universe:v1",
            config_id="synthetic-config:v1",
            backend_id="synthetic-backend:v1",
            dependency_id="synthetic-dependencies:v1",
            seed_id="synthetic-seed:v1",
            thread_id="synthetic-thread:v1",
        )
    )
    study = StudyManifest(
        StudySpec(
            run_spec=run.spec,
            feature_schema_id="synthetic-features:v1",
            label_spec_id="synthetic-labels:v1",
            economic_spec_id="synthetic-economic:v1",
            calibration_window=StudyWindow("calibration", 0, 2),
            tuning_window=StudyWindow("tuning", 2, 4),
            validation_window=StudyWindow("validation", 4, 6),
            holdout_window=StudyWindow("holdout", 6, 8),
        ),
        metadata={"evidence_class": "synthetic_protocol_only"},
    )
    timeline = StudyTimeline(
        decision_times=tuple(range(1, 7)),
        target_ids=tuple(f"target-{index}" for index in range(1, 7)),
        decision_indices=tuple(range(6)),
        expected_instrument_ids=(("SYNTH",),) * 6,
    )
    observations = tuple(
        MarketObservation(
            record_id=f"observation-{index}",
            instrument_id="SYNTH",
            event_time=index,
            available_time=index,
            source_revision=0,
            ingest_sequence=index,
            fields={"return_feature": float(index) / 100.0},
        )
        for index in range(1, 7)
    )
    labels = tuple(
        LabelObservation(
            label_id=f"label-{index}",
            target_id=f"target-{index}",
            event_time=index + 1,
            available_time=index + 1,
            received_time=index + 1,
            status="observed",
            value=float(index) / 1000.0,
            source_revision=0,
            ingest_sequence=index,
        )
        for index in range(1, 7)
    )
    book = AsOfBook(
        observations=observations,
        universe=(
            UniverseMembership(
                instrument_id="SYNTH",
                start=0,
                end=100,
                event_time=0,
                available_time=0,
                source_revision=0,
                ingest_sequence=0,
            ),
        ),
        labels=labels,
    )
    evidence = EconomicEvidence(
        source_id="synthetic-economic:v1",
        realized_returns=tuple(
            RealizedReturn(
                target_id=f"target-{index}",
                decision_time=index,
                realization_time=index + 1,
                available_time=index + 2,
                value=float(index) / 1000.0,
            )
            for index in range(1, 7)
        ),
        execution_costs=tuple(
            ExecutionCost(
                target_id=f"target-{index}",
                decision_time=index,
                execution_time=index,
                available_time=index,
                cost_model_id="synthetic-costs:v1",
                fee_rate=0.0001,
                spread_rate=0.0001,
                capacity_limit=1.0,
            )
            for index in range(1, 7)
        ),
    )
    raw_payloads = {
        f"synthetic-{role}.jsonl": f"{{\"role\":\"{role}\",\"synthetic\":true}}\n".encode(
            "ascii"
        )
        for role in ROLES
    }
    artifacts = tuple(
        StudySourceArtifact.from_bytes(
            artifact_id=artifact_id,
            role=role,
            payload=raw_payloads[artifact_id],
            record_count=1,
        )
        for role, artifact_id in zip(ROLES, raw_payloads)
    )
    provenance = StudySourceProvenance(
        provider_id="synthetic-provider:v1",
        dataset_id="synthetic-dataset:v1",
        vintage_id=run.spec.input_vintage_id,
        license_id="synthetic-license:none",
        release_id="synthetic-release:2026-08-04",
        adapter_revision="synthetic-adapter:v1",
        as_of_rule="available_time <= decision_time",
        revision_rule="latest visible source revision",
        universe_rule="visible membership interval",
        delisting_rule="synthetic delisting role retained as an explicit artifact",
        source_artifacts=artifacts,
        retrieved_at=2026_08_04,
    )
    return (
        StudySourcePackage(
            provenance=provenance,
            bundle=StudyInputBundle(
                run_manifest=run,
                study_manifest=study,
                timeline=timeline,
                as_of_book=book,
                economic_evidence=evidence,
                economic_cutoff=10,
            ),
        ),
        raw_payloads,
    )


def run(output_path: Path) -> dict[str, Any]:
    """Run the real filesystem intake and write a deterministic receipt."""

    package, raw_payloads = build_synthetic_package()
    with TemporaryDirectory(prefix="topology-gate-synthetic-handoff-") as directory:
        root = Path(directory)
        package_path = root / "study-source-package.json"
        raw_dir = root / "raw"
        raw_dir.mkdir()
        package_path.write_text(package.to_json() + "\n", encoding="utf-8")
        for artifact_id, payload in raw_payloads.items():
            (raw_dir / artifact_id).write_bytes(payload)
        intake = run_all_pre_holdout_intake(package_path, raw_dir)

    receipt = {
        "evidence_class": "synthetic_protocol_only",
        "market_claim_authorized": False,
        "holdout_opened": False,
        "package_digest": package.digest,
        "provenance_digest": package.provenance.digest,
        "artifact_roles": list(ROLES),
        "artifact_ids": sorted(raw_payloads),
        "audited_phases": intake["audited_phases"],
        "source_audit_digests": {
            phase: audit["source_audit_digest"]
            for phase, audit in intake["audits"].items()
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(receipt, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/synthetic-market-handoff.json"),
    )
    args = parser.parse_args()
    print(json.dumps(run(args.output), sort_keys=True))


if __name__ == "__main__":
    main()
