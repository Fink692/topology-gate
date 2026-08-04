# Production/research runbook

This package is an offline research/alpha control component. It is not a
production-certified broker, data vendor, or live-trading service. The caller
must provide point-in-time data and own portfolio/risk/execution controls.

## Install and verify

```powershell
python -m pip install -e ".[dev]"
python -m pytest -q -p no:cacheprovider
ruff check src tests
mypy src
```

The core package can be imported without NumPy because numerical workers are
lazy exports. Install the `numeric`, `statistics`, or `data` extras only when
those adapters are used.

## Causal online composition

```python
from topology_gate import (
    OnlineRunConfig,
    RLS,
    RLSConfig,
    RollingTopologyDetector,
    TopologyConfig,
    generate_synthetic_regimes,
    run_recursive_rls,
)

dataset = generate_synthetic_regimes(
    n_steps=512,
    n_features=4,
    change_points=(160, 320),
    seed=7,
    label_delay=1,
)
detector = RollingTopologyDetector(
    TopologyConfig(
        embedding_dim=1,
        cloud_window=32,
        calibration_window=64,
        calibration_min_periods=16,
        forgetting_lambda_min=0.90,
        forgetting_lambda_max=0.995,
    )
)
learner = RLS(
    RLSConfig(
        n_features=dataset.features.n_features,
        lambda_min=0.90,
        lambda_max=1.0,
    )
)
result = run_recursive_rls(
    dataset.features.values,
    dataset.labels.values,
    realized_returns=dataset.realized_returns,
    market_states=dataset.features.values,
    detector=detector,
    learner=learner,
    config=OnlineRunConfig(label_delay=1, transaction_cost_bps=2.0),
    shift_points=dataset.change_points,
)
print(result.metrics)
```

The ordering is fixed: observe current state, predict, compute cost-adjusted
outcome, then update immediately or queue the label until its availability
boundary. A detector score never uses a future row, and the factor used for a
queued update is captured at the original decision time.

The compatibility runner keeps topology-driven forgetting at the learner's
neutral maximum unless `run_recursive_rls(..., calibration=certificate)`
receives an approved certificate bound to the detector's `config_identity`.
The returned `acceleration_authorized` mask and calibration identity belong in
the run evidence; a detector's warm-up or `ready` flag alone is not a
calibration authorization.

For restartable replay, capture the learner, detector, online terminal state,
and promotion gate through `checkpoint_from_components`. Validate the envelope
with its expected package/configuration/backend/dependency identities and an
HMAC key before swapping any component state. Restore the learner and detector
into detached candidates, then continue the next data chunk with
`run_recursive_rls(..., initial_state=restored_online_state,
config=OnlineRunConfig(reset_state=False, ...))`; availability positions are
absolute stream positions. Callable policies must be supplied again and must
match the recorded callable identity. Plain SHA-256 checkpoints are for
trusted local artifacts only and are rejected by component restore unless the
caller opts into `allow_untrusted=True`.

## Challenger promotion

Use `PromotionGate` on aligned, cost-adjusted challenger and incumbent utility
streams. The gate clips the utility difference, uses a predictable e-factor,
and requires an explicit alpha allocation for each challenger. Register the
complete challenger family and call `seal_registration()` before the first
score in a certified run; the gate rejects post-seal candidates and persists
the boundary in its checkpoint. Do not reset an
active e-process after looking at an unfavorable segment. A promotion becomes
an operational decision only after the caller's schema, data-quality, risk, and
deployment checks pass.

Set `CausalPromotionConfig.minimum_labels` to the pre-registered burn-in
count when a challenger must accumulate learner evidence before promotion
testing. Those observed labels may update the paired learners, but they do not
advance the gate; the count is checkpointed as part of the promotion identity.

Before a multi-challenger study, run `calibrate_promotion_null` with the exact
bounded score factory, constant eta, challenger count, and global alpha used
by the gate. If the gate can reset, declare the epoch count and retain the
entire per-epoch alpha schedule. Keep the allocation and first-promotion
evidence beside the study manifest; it is a finite selection-budget diagnostic,
not a market promotion certificate.

## Metric and claim conventions

Use `absolute_comparator_discrepancy` for the symmetric comparator diagnostic
and `one_sided_utility_regret` for the cost-matched comparator advantage. The
legacy `dynamic_regret` field is retained only for compatibility and is not
conventional dynamic regret. When `expected_returns` is supplied, both
comparator utilities use that point-in-time series; otherwise they use realized
returns. Ordinary PnL metrics always use realized returns.

The default detector is a causal **kNN normalized-Laplacian spectral
approximation**, not a persistent Laplacian. Its CUSUM alarms, score-to-
forgetting decisions, and promotion statistics are exploratory until a
separate calibration study establishes their null behavior and selection
budget.

For a small exact reference run, configure the finite backend explicitly and
match both budgets to the rolling detector:

```python
from topology_gate import (
    PersistentLaplacianBackend,
    PersistentLaplacianConfig,
)

exact_backend = PersistentLaplacianBackend(
    PersistentLaplacianConfig(
        max_vertices=16,
        max_simplices=696,
        q=0,
        n_eigenvalues=2,
    )
)
exact_detector = RollingTopologyDetector(
    TopologyConfig(
        embedding_dim=1,
        cloud_window=16,
        min_points=8,
        n_eigenvalues=2,
        graph_neighbors=4,
        persistent_laplacian_backend=exact_backend,
    )
)
```

This is a bounded numerical reference path, not a market-calibrated detector.
The backend identity is checkpointed; an incompatible cloud or failed exact
calculation must be handled as an abstention/error by the surrounding replay.

For a cross-asset causal run, construct `PointInTimePanel` from explicit
as-of-visible record IDs and one fixed field schema. The adapter sorts
instrument rows and emits a panel digest plus the snapshot universe digest;
record those identities with the numerical replay. This is a deterministic
selection contract only: the run still needs a vendor manifest, delisting and
membership policy, cost model, and sealed holdout before any market claim.
When the vendor adapter produces a canonical `AsOfBook` export, retain its
versioned JSON and digest beside the `StudyManifest`; restore it with
`AsOfBook.from_json()` before replay so field, revision, and source-integrity
failures stop the run at the data boundary.
Restore the run/study manifest with `RunManifest.from_json()` or
`StudyManifest.from_json()` and compare the resulting digest with the recorded
run artifact before consuming any source rows; unknown or missing manifest
fields are rejected.

## Release gates

Before using a result outside research:

- record the source vintage/availability cutoff and configuration fingerprint;
- run a strict chronological replay with delayed-label audit enabled;
- verify detector, learner, and gate state snapshots restore identically;
- review `reports/statistical-review.md`, `reports/security-repro-review.md`,
  and `reports/integration-review.md` when present;
- keep the default kNN normalized-Laplacian detector labeled as an exploratory
  approximation; an exact persistent-Laplacian backend must be separately
  identified and independently verified before stronger claims are made;
- attach the metric and limitations contract in `docs/statistical-validity.md`.
