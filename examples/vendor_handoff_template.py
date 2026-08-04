"""Generate the exact point-in-time market-data handoff template.

The template deliberately contains no synthetic or placeholder observations.
It is a machine-readable request for the six raw roles required by
``StudySourcePackage.audit_market``. The generated status remains blocked until
the vendor supplies the artifacts and their retrieval/license metadata.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REQUIRED_ROLES = (
    "delistings",
    "execution-costs",
    "labels",
    "market-observations",
    "realized-returns",
    "universe-membership",
)

COMMON_FIELDS = (
    "instrument_id_or_target_id",
    "event_time",
    "available_time",
    "source_revision",
    "ingest_sequence",
)

ROLE_SPECIFICATIONS: dict[str, dict[str, Any]] = {
    "market-observations": {
        "filename": "market-observations.jsonl",
        "required_fields": [*COMMON_FIELDS, "open", "high", "low", "close", "volume"],
        "notes": "Use permanent instrument IDs and preserve raw vendor prices plus corporate-action fields.",
    },
    "universe-membership": {
        "filename": "universe-membership.jsonl",
        "required_fields": [*COMMON_FIELDS, "membership_start", "membership_end", "is_member"],
        "notes": "Membership must be point-in-time visible at each decision boundary.",
    },
    "delistings": {
        "filename": "delistings.jsonl",
        "required_fields": [*COMMON_FIELDS, "delisting_time", "delisting_return", "delisting_reason"],
        "notes": "Retain failed instruments through the final visible interval; explain missing returns.",
    },
    "labels": {
        "filename": "labels.jsonl",
        "required_fields": [*COMMON_FIELDS, "target_id", "label_value", "label_available_time"],
        "notes": "Label availability must be later than the decision event and must not use future revisions.",
    },
    "realized-returns": {
        "filename": "realized-returns.jsonl",
        "required_fields": [*COMMON_FIELDS, "target_id", "return_value", "return_available_time"],
        "notes": "Keep realized returns separate from labels and retain observed/censored status.",
    },
    "execution-costs": {
        "filename": "execution-costs.jsonl",
        "required_fields": [*COMMON_FIELDS, "target_id", "cost_rate", "turnover", "capacity_limit"],
        "notes": "Costs and capacity must be sourced per decision/target when economic claims are made.",
    },
}


def build_template() -> dict[str, Any]:
    return {
        "kind": "topology_gate_vendor_handoff_template",
        "version": 1,
        "status": "awaiting_vendor_handoff",
        "market_claim_authorized": False,
        "holdout_opened": False,
        "protocol": "pl-ridge-certified-promotion:v1",
        "source_vintage_id": "point-in-time-source:required",
        "required_artifacts": [
            {"role": role, "status": "missing", **ROLE_SPECIFICATIONS[role]}
            for role in REQUIRED_ROLES
        ],
        "required_provenance": {
            "provider_id": "vendor-provided",
            "dataset_id": "vendor-provided",
            "vintage_id": "must match the run manifest",
            "license_id": "vendor-provided",
            "release_id": "vendor-provided",
            "adapter_revision": "pinned repository adapter revision",
            "retrieved_at": "vendor retrieval timestamp",
            "raw_sha256_and_byte_size": "required for every artifact",
        },
        "next_command_after_handoff": (
            "python examples/market_source_intake.py "
            "--package handoff/study-source-package.json "
            "--raw-dir handoff/raw --all-pre-holdout "
            "--receipt-dir handoff/source-audits"
        ),
        "limitations": [
            "This template contains no observations and cannot authorize a study.",
            "A final adjusted-price history cannot satisfy the point-in-time gate.",
            "Holdout remains sealed until validation audit and explicit release.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/vendor-handoff-status.json"),
    )
    args = parser.parse_args()
    payload = build_template()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()

