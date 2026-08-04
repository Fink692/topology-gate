"""Frozen protocol declaration for the first PL-RLS promotion study.

This module declares identities and split boundaries only.  It intentionally
does not manufacture a market source or open the final holdout.  A vendor
adapter must replace the pending source identity with a verified
``StudySourcePackage`` before the study can produce market evidence.
"""

from __future__ import annotations

import json
from typing import Any

from topology_gate.manifest import RunSpec, StudyManifest, StudySpec, StudyWindow
from topology_gate.selection import SelectionBudget

PROTOCOL_ID = "pl-ridge-certified-promotion:v1"
SOURCE_VINTAGE_ID = "point-in-time-source:required"

# Four model choices x four feature sets x three eta choices.  The selected
# cell is the first pre-registered cell; changing it starts a new protocol.
SELECTION_BUDGET = SelectionBudget(
    budget_id="pl-ridge-selection-family:v1",
    global_alpha=0.05,
    model_slots=4,
    feature_slots=4,
    eta_slots=3,
    model_index=1,
    feature_index=1,
    eta_index=1,
)

STUDY_SPEC = StudySpec(
    run_spec=RunSpec(
        run_id=PROTOCOL_ID,
        input_vintage_id=SOURCE_VINTAGE_ID,
        universe_id="liquid-cross-asset-universe:required",
        config_id="pl-ridge-control-config:v1",
        backend_id="persistent-laplacian-f2-reference:v1",
        dependency_id="topology-gate-release-environment:py312-numpy2.2.6",
        seed_id="pl-ridge-seeds:v1",
        thread_id="single-thread-deterministic:v1",
    ),
    feature_schema_id="cross-asset-state-vector:v1",
    label_spec_id="five-day-volatility-normalized-forward-return:v1",
    economic_spec_id="net-return-cost-capacity:v1",
    calibration_window=StudyWindow("calibration", 0, 252),
    tuning_window=StudyWindow("tuning", 257, 509),
    validation_window=StudyWindow("validation", 514, 766),
    holdout_window=StudyWindow("holdout", 771, 1023),
    embargo_steps=5,
)


def build_manifest() -> StudyManifest:
    """Return the sealed, immutable protocol manifest."""

    return StudyManifest(
        STUDY_SPEC,
        metadata={
            "protocol_id": PROTOCOL_ID,
            "status": "protocol-only-source-pending",
            "selection_budget_id": SELECTION_BUDGET.budget_id,
            "detector_threshold_candidates": [2.0, 4.0, 8.0, 16.0],
            "block_length": 16,
            "prediction_horizon_steps": 5,
            "missingness_policy": "zero-certified-budget",
            "final_holdout": "sealed-until-explicit-release",
        },
    )


def protocol_state() -> dict[str, Any]:
    """Return the canonical protocol declaration for review and storage."""

    manifest = build_manifest()
    return {
        "protocol_id": PROTOCOL_ID,
        "selection_budget": SELECTION_BUDGET.state_dict(),
        "selected_gate_alpha": SELECTION_BUDGET.allocated_alpha,
        "study_manifest": manifest.to_dict(),
        "study_manifest_digest": manifest.digest,
    }


if __name__ == "__main__":
    print(json.dumps(protocol_state(), sort_keys=True, indent=2))
