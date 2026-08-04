# Point-in-time study intake and execution

`StudyInputBundle` is the normalized handoff between a vendor adapter and the
causal model. It does not parse vendor-native files and it does not claim that
an input source is survivorship-free. The adapter must first produce:

See [`docs/vendor-data-gate.md`](vendor-data-gate.md) for the required source,
license, delisting, vintage, capacity, and holdout evidence before calling a
run economic.

- an `AsOfBook` containing event time, availability time, source revision, and
  point-in-time universe membership;
- an `EconomicEvidence` bundle containing separately sourced realized returns
  and execution costs when economic evaluation is required; and
- a `StudyTimeline` whose decision times, target IDs, and manifest indices are
  aligned and strictly ordered.

The bundle digest binds the run manifest, study manifest, timeline, as-of
source digest, and economic-evidence digest. A phase audit then materializes
every decision snapshot before model execution. It rejects a visible target
label, a mismatched expected universe, a sealed holdout read, or incomplete
economic evidence.

The bundle also rejects a `StudyManifest` whose embedded `RunSpec` does not
match the supplied `RunManifest`; source, split, and execution identities must
refer to the same registered run.

For a reproducible handoff, wrap the bundle in `StudySourcePackage`:

```python
from topology_gate import (
    StudySourceArtifact,
    StudySourcePackage,
    StudySourceProvenance,
)

package = StudySourcePackage(
    provenance=StudySourceProvenance(
        provider_id="vendor-adapter:v1",
        dataset_id="cross-asset:v1",
        vintage_id=run_manifest.spec.input_vintage_id,
        license_id=license_id,
        release_id=source_release_id,
        adapter_revision=adapter_revision,
        as_of_rule="available_time <= decision_time",
        revision_rule="latest visible source_revision at cutoff",
        universe_rule="visible membership interval at decision time",
        delisting_rule="retain delisted instruments through final visible interval",
        source_artifacts=(
            StudySourceArtifact.from_bytes(
                "market-data.csv",
                "market observations and universe",
                raw_market_bytes,
                market_record_count,
            ),
        ),
        retrieved_at=retrieved_at,
    ),
    bundle=bundle,
)
payload = package.to_json()
restored = StudySourcePackage.from_json(payload)
audit = restored.audit("validation", require_complete_universe=True)
```

The package carries the full canonical manifests, timeline, as-of book, and
optional economic evidence. Restoration verifies exact schema fields, tagged
time domains, every nested artifact digest, the bundle digest, and the package
digest. `verify_source_artifact(...)` checks one raw payload against its
declared SHA-256 and byte size; `verify_source_artifacts(...)` additionally
requires the supplied payload mapping to contain exactly every declared
artifact ID. Provenance describes the adapter's source policy; it is not
independent proof that a vendor source is survivorship-free or
revision-complete.

For the market-evidence gate, use the stricter source audit after restoring
the raw payloads. It requires the exact artifact roles `delistings`,
`execution-costs`, `labels`, `market-observations`, `realized-returns`, and
`universe-membership`; verifies every declared byte; binds the provenance
vintage to `RunSpec.input_vintage_id`; requires complete point-in-time
universe rows, observed returns/costs, and per-target capacity limits; and
returns a digest-bound `StudySourceAudit` receipt:

```python
source_audit = restored.audit_market(
    "validation",
    {artifact_id: raw_bytes[artifact_id] for artifact_id in artifact_ids},
)
```

For a filesystem handoff, the checked-in intake command performs the same
operation while requiring one safe filename for every declared artifact:

```powershell
$env:PYTHONPATH = 'src'
py -3.12 examples\market_source_intake.py `
  --package handoff\study-source-package.json `
  --raw-dir handoff\raw `
  --phase validation `
  --receipt handoff\validation-source-audit.json
```

It verifies the package digest, exact raw bytes, required market roles,
complete point-in-time universe, observed economic records, and capacity
evidence. Missing, extra, unsafe, or tampered artifacts fail before a study
run starts. The command is an intake verifier, not a vendor-native parser.

To audit every pre-holdout phase and write one receipt per phase, use:

```powershell
$env:PYTHONPATH = 'src'
py -3.12 examples\market_source_intake.py `
  --package handoff\study-source-package.json `
  --raw-dir handoff\raw `
  --all-pre-holdout `
  --receipt-dir handoff\source-audits
```

This audits `calibration`, `tuning`, and `validation` in that order. It never
calls the holdout path; opening holdout still requires a new manifest and an
explicit release event.

The ordinary `audit(...)` path remains appropriate for synthetic or partial
engineering runs. It does not authorize a market-performance claim.

```python
from topology_gate import (
    StudyInputBundle,
    StudyTimeline,
    run_causal_rls_study,
)

timeline = StudyTimeline(
    decision_times=decision_times,
    target_ids=target_ids,
    decision_indices=manifest_indices,
    expected_instrument_ids=expected_instruments_by_decision,
)
bundle = StudyInputBundle(
    run_manifest=run_manifest,
    study_manifest=study_manifest,
    timeline=timeline,
    as_of_book=as_of_book,
    economic_evidence=economic_evidence,
    economic_cutoff=economic_cutoff,
)

result = run_causal_rls_study(
    bundle,
    "validation",
    plan=feature_plan,
    learner=learner,
    detector=detector,
    require_complete_universe=True,
)

economic_decisions = result.economic_decisions
```

Set `require_observed_economic_evidence=True` only when the selected evidence
cutoff is expected to contain an observed return and cost for every timeline
target. Missing or censored records must remain explicit in exploratory or
partial runs; they must not be replaced by zeroes.

The holdout remains sealed for calibration, tuning, and validation. To run the
pre-registered holdout, create a new manifest with
`study_manifest.open_holdout(release_id)` and retain that opened manifest and
the resulting audit receipt with the final report. The wrapper uses the same
`run_causal_rls_replay` transition as the lower-level adapter, so predictions,
delayed labels, topology evidence, and checkpoint state retain their existing
causal semantics.

Paired challenger studies use the analogous
`run_causal_promotion_study(...)` wrapper. It performs the same source and
phase audit before invoking `run_causal_promotion_replay`, so registration,
label settlement, e-process updates, and checkpoint identities remain bound to
the audited source bundle.

This boundary is a source-integrity and execution protocol. It is not a market
calibration certificate, a capacity model, or proof of economic performance.
