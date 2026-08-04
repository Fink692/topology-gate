# topology-gate

`topology-gate` is a small, typed Python package boundary for research on a
topology-gated recursive quant model. It defines the objects and protocols that
connect five independently implemented pieces:

- a topology detector;
- a recursive least-squares (RLS) learner;
- an e-process gate;
- a synthetic-data generator; and
- an offline backtest/evaluation runner.

This repository does not claim that a topology signal predicts returns or that a
gate is statistically valid for a particular data-generating process. The
algorithms, assumptions, and calibration procedures belong in their respective
worker modules and must be evaluated empirically.

The default detector is a causal k-nearest-neighbour normalized-Laplacian
spectral approximation. It is explicitly not a persistent-Laplacian or
persistent-homology implementation. The e-process primitive is valid only for
its documented bounded conditional-mean score and predictable betting rule;
the surrounding caller must enforce data availability, selection, and risk
controls.

For small research goldens, the optional `persistent` worker exposes a bounded
Euclidean Vietoris–Rips filtration over `F2`, persistence intervals, and a
nullspace-restricted persistent-Laplacian spectrum via
`compute_persistent_laplacian`. `PersistentLaplacianBackend` is the configured
callable adapter for using that exact finite construction in
`RollingTopologyDetector`; its filtration/solver identity and vertex budget
are included in the detector identity, and incompatible cloud/spectrum sizes
are rejected before a stream starts. It is a finite reference backend with
explicit resource caps; it is not silently substituted into the rolling
default and it does not calibrate the downstream CUSUM. A backend or solver
failure during streaming rolls back the rejected observation.

`PersistentLaplacianCUSUM` is the exploratory control-layer prototype for the
research proposal. It keeps a bounded rolling point cloud, extracts declared
Betti counts and positive persistent-Laplacian eigenvalues, standardizes the
current state only against earlier valid states, and emits a CUSUM score plus a
suggested forgetting factor. It implements the same `observe`/stream-state
protocol as the causal numerical adapter, but its score is not calibrated by
construction: accelerated forgetting still requires a matching approved
finite-null certificate and a market study still requires point-in-time data.
Its stateless `detect(...)` facade also plugs into the finite
`calibrate_null`/`calibrate_shift` harness while leaving streaming checkpoint
state available through `observe(...)`.

## Install

The core package has no runtime dependencies:

```bash
python -m pip install -e .
```

Install only the optional numerical stack needed by a worker module:

```bash
python -m pip install -e ".[numeric]"      # NumPy-based computation
python -m pip install -e ".[statistics]"   # NumPy plus SciPy routines
python -m pip install -e ".[data]"         # NumPy plus pandas adapters
python -m pip install -e ".[test]"
```

NumPy, SciPy, and pandas are optional because the shared types use only the
standard library. The extras make the dependency boundary explicit: numerical
workers can require NumPy, statistical helpers can require SciPy, and tabular
adapters can require pandas without imposing those packages on every caller.
The release-gate direct pins are recorded in
[`requirements-release-py312.txt`](requirements-release-py312.txt); numerical
results still depend on the platform's BLAS/runtime implementation.

## Stable package boundary

The package root exports the shared types from `topology_gate.types`:

```python
from topology_gate import (
    BacktestDatasetProtocol,
    BacktesterProtocol,
    TopologySignal,
)


def evaluate_offline(
    backtester: BacktesterProtocol,
    dataset: BacktestDatasetProtocol,
):
    return backtester.run(dataset)


signal = TopologySignal(score=0.0, confidence=None)
```

`ArrayLike` intentionally describes one- or two-dimensional numeric sequences;
worker implementations may accept NumPy arrays when the `numeric` extra is
installed. The richer `SyntheticDataset` implementation is owned by
`topology_gate.synthetic`; the shared API uses `BacktestDatasetProtocol` so it
does not duplicate that worker-owned class.

The protocols are intentionally narrow integration contracts:

| Component | Required boundary | Shared result |
| --- | --- | --- |
| Topology detector | `detect(features)` | worker-owned topology result |
| RLS learner | `predict(features)` and `update(features, target)` | updated coefficients/state |
| E-process gate | `update(score, *, eta=None, metadata=None)` | worker-owned promotion decision |
| Synthetic data | seeded factory callable | worker-owned synthetic dataset |
| Backtest | `run(features, labels=None, realized_returns=None, ...)` | worker-owned backtest report |

`TopologySignal.score` and `dimension` have no package-wide scale; the detector
must document their meaning. A detector may put its own threshold result in
`passed`. `GateDecision` records whether the gate allowed the step, the evidence
value and threshold used, and an optional topology signal/reason.

## Offline evaluation only

`BacktestConfig` and `BacktestResult` describe sequential model evaluation.
There are no broker, order, position, account, or live-data interfaces in this
package. A backtest should make its data split, warmup, update order, gate
semantics, and reported metrics explicit; it should not be interpreted as a
live-trading integration.

## Causal replay and checkpoints

`run_recursive_rls` separates training outcomes from realized returns, supports
fixed delays and explicit `label_available_at` positions, and returns terminal
pending labels instead of silently discarding them. Pass the returned
`OnlineStreamState` as `initial_state` with `reset_state=False` to continue a
chunked replay at its absolute stream position. `CheckpointEnvelope` and
`checkpoint_from_components` provide canonical JSON state with configuration,
backend, dependency, manifest, learner, detector, online, promotion, evidence,
and RNG fields. Use
an HMAC key from a secret manager; authenticated checkpoints are required by
default and the key is never written into the checkpoint. Plain SHA-256 is
available only with an explicit trusted-local opt-in.

When a topology detector is supplied to `run_recursive_rls`, pass an approved
calibration certificate through `calibration` before allowing its factor to
accelerate forgetting. A missing, unapproved, or mismatched certificate keeps
the detector diagnostic and applies the neutral learner maximum; the result
exposes `acceleration_authorized` and the calibration identity.

Set `require_realized_returns=True` in the worker backtest or online config for
economic evaluation. The compatibility default still permits target outcomes
to stand in for realized returns, but that path is diagnostic and must not be
reported as a tradable return result.

This is an offline control component. It has no broker, network, order, account,
or live-data side effect.

For point-in-time research identities, `RunSpec`/`RunManifest` freeze the input
vintage, universe, configuration, backend, dependency, seed, and thread policy
into canonical JSON and a SHA-256 digest. `PromotionEvidenceConfig` then binds a
challenger evidence family to its score, eta, missing-label, allocation, and
manifest identities. A ledger without those identities remains explicitly
diagnostic.

`StudySpec` adds the pre-registered calibration, tuning, validation, and
holdout windows plus an explicit purge/embargo gap. `StudyManifest` starts with
the holdout sealed and records a new digest and release ID only when
`open_holdout(...)` is called. This makes the study boundary auditable; it is a
protocol for a real source manifest, not proof that a supplied dataset is
survivorship-free or economically complete. `RunManifest.from_json()` and
`StudyManifest.from_json()` strictly restore these identities and reject
unknown or missing fields.

The causal numerical and promotion replay adapters can bind a study phase and
strictly increasing timeline indices to that manifest. The manifest digest is
stored in model state, so a resumed replay cannot silently switch source
identity or open a sealed holdout under a different study context.

For cross-asset panels, `PointInTimePanel.from_snapshot` can also receive an
explicit `expected_instrument_ids` set. A missing or unexpected instrument
then fails closed, and the coverage assertion is included in the panel digest;
omitting that argument remains an explicit partial-panel choice.

The dependency-light `AsOfBook` is the corresponding data-boundary contract:
observations, labels, and universe memberships carry event time, availability,
source revision, and deterministic ingest order. Its snapshots exclude future
revisions and expose missing labels explicitly. It is a causal contract and
test fixture, not a market-data vendor or a survivorship-bias guarantee. For a
canonical vendor export, `AsOfBook.to_json()` emits a versioned source artifact
with a content digest, and `AsOfBook.from_json()` rejects unknown fields,
malformed revisions, and digest mismatches.

`PointInTimePanel` is the explicit cross-asset selection contract layered on an
as-of snapshot. It requires one visible record per instrument, a fixed field
schema, deterministic instrument ordering, and carries both a panel digest and
the snapshot's universe digest. Pass `expected_instrument_ids` when the study
requires complete coverage of the pre-registered point-in-time universe; the
coverage assertion is bound into the panel digest. Instrument-labelled
`causal_numeric` and `causal_promotion` plans use that canonical ordering;
unlabelled plans remain
explicit legacy row adapters. This does not supply a vendor universe, dynamic
membership history, or evidence of market completeness.

`CausalReplay` is the dependency-light transition boundary for a timestamped
study. It materializes one `AsOfSnapshot`, settles only labels visible before
the next prediction, rejects pre-available targets, and emits a hash-chained
prediction/label ledger. A checkpointable model must expose a JSON-safe
`state_dict`; callers that disable that requirement are explicitly using an
untracked diagnostic replay.

`run_causal_rls_replay` is the numerical migration adapter: immutable feature
bindings are resolved from the snapshot, the real topology detector and RLS
learner run behind the shared transition, and the prediction-time forgetting
factor is retained until label settlement. It returns no tradable PnL by
design. Use `evaluate_economic_path` only with separately sourced
`RealizedReturn` and `ExecutionCost` records; missing, censored, or invalid
returns remain explicit and fail closed rather than becoming zero, and
abstentions remain visible. A strict economic configuration can also require
sourced per-decision capacity limits and reject turnover breaches. When the
configured topology backend
produces an exact finite artifact, its 64-character content digest is carried
into the per-step causal telemetry; canonical feature/state panel digests are
carried alongside it; malformed digests fail closed.
`EconomicEvidence` provides the corresponding digest-bound revision bundle for
realized returns and execution costs; call `select_at(...)` before evaluation
to make the evidence cutoff explicit, or use the wrapper that records both the
bundle digest and cutoff in the economic result.

For the vendor-to-model handoff, [`docs/study-runbook.md`](docs/study-runbook.md)
documents `StudyInputBundle`, `StudyTimeline`, and `run_causal_rls_study`. This
preflights manifest phase indices, target-label visibility, expected
point-in-time universe coverage, and optional economic evidence before entering
the shared causal replay. `run_causal_promotion_study` applies the same
preflight to paired challenger/e-process runs. `StudySourcePackage` adds a
canonical JSON envelope for the normalized artifacts and records the adapter's
as-of, revision, universe, and delisting policies with digest verification. Its
required raw-artifact fingerprints can be checked byte-for-byte through
`verify_source_artifact(...)`; `verify_source_artifacts(...)` additionally
rejects omitted or unexpected raw files. It does not parse vendor-native files
or certify a vendor's point-in-time claim. `audit_market(...)` is the stricter
handoff: it verifies every raw artifact, binds the provenance vintage to the
run manifest, requires the declared market/universe/delisting/label/return/
cost roles, and fails closed unless observed economic and per-target capacity
evidence are complete. It returns a digest-bound `StudySourceAudit` receipt.

`run_causal_promotion_replay` composes the same transition with paired
challenger/incumbent learners and the existing alpha-spending promotion gate.
Predictions are frozen at decision time; only an observed settled label can
advance the gate. Missing, invalid, censored, or terminally unresolved labels
clear pending context without creating evidence. The adapter requires a
predeclared and sealed challenger family, a constant eta, and bounded
absolute-error utility scale. Register every candidate and call
`PromotionGate.seal_registration()` before the first observation; a later
registration is rejected and the seal is checkpointed. Its
checkpointed `minimum_labels` policy can burn in learner updates without
advancing the e-process; it does not turn a synthetic or unbound run into a
certified promotion claim. The adapter requires prediction workers to leave
their checkpointed state unchanged during `predict` and fingerprints the
sealed gate family, alpha/score scales, eta rules, and epoch; an external gate
reset or family mutation therefore fails closed. Set
`require_pure_predictions=False` only for explicitly diagnostic, non-certified
experiments. Certified runs also default to zero budgets for non-observed,
unresolved, abstained, and invalid paired records. A breached budget is
checkpointed as `CausalPromotionStatus.BLOCKED` and prevents later clean labels
from advancing the same e-process; non-zero budgets are explicit diagnostic
choices whose missingness assumptions still require separate validation. A
threshold crossing is exposed as `promotion_activation`; its effective
prediction boundary is resolved from the replay ledger and remains unset when
the segment ends before a later decision.
The checkpoint also binds each learner's stable class/config identity and the
full mutable gate-evidence fingerprint; an outside learner reconfiguration or
gate observation is rejected at the next boundary.

The standalone `EvidenceLedger` applies the same strict boundary for callers
that already own a promotion gate: certified use requires a sealed challenger
family and a declared score specification. Certified labels carry the raw
target value; utilities are recomputed from the frozen challenger/incumbent
actions, while caller-supplied utility pairs remain diagnostic-only. Its
checkpoint records the gate evidence fingerprint and rejects direct gate
observations, resets, or score-spec substitution outside the ledger.

The external data acceptance checklist is in
[`docs/vendor-data-gate.md`](docs/vendor-data-gate.md). It explicitly requires
permanent identifiers, delisting/universe history, dated source cuts, separate
cost/capacity evidence, and a sealed holdout.

## Calibration evidence

Use [`docs/calibration.md`](docs/calibration.md) and the bounded
`calibrate_null`/`calibrate_shift` helpers before interpreting detector alarms,
forgetting changes, or shift delays. Use `calibrate_threshold` when a threshold
must be selected: it requires distinct declared calibration and evaluation
observation factories, separates predeclared candidate selection from an
independent evaluation split, and only the latter can produce a certificate.
These helpers produce finite-horizon Wilson intervals and censored run-length
evidence; they do not establish universal market or optional-stopping
guarantees. `NullCalibrationResult.to_certificate` and
`ThresholdCalibrationResult.to_certificate` are the explicit authorization
boundaries for the numerical adapter: without a matching approved finite-null
certificate, topology-driven forgetting remains at its neutral maximum. Use
`StationaryBlockBootstrap` when the declared null must preserve local serial
dependence; its source and bootstrap specification are included in the
calibration identity.

Use `calibrate_eprocess_null` for the paired-promotion control layer. It
simulates bounded score paths under optional stopping with a predeclared
constant betting fraction and records the finite crossing interval. It is an
empirical diagnostic of the declared stream, not a replacement for the
conditional-mean assumption or a market-selection audit.

Use `calibrate_promotion_null` when several challengers can compete. It runs
the complete `PromotionGate`, records the geometric per-slot alpha allocation,
seals the pre-registered challenger family before scores are observed, and
stops each path at the first selected promotion. It can also exercise a
predeclared sequence of gate epochs, including the alpha spent after resets.
This makes the finite selection boundary observable; it remains a
declared-score simulation rather than a market-calibrated promotion
certificate.

## Tests and typing

Run the configured test suite with:

```bash
python -m pip install -e ".[test]"
python -m pytest
ruff check src tests examples
mypy src
```

The reusable fixtures in `tests/conftest.py` are deterministic and use only the
standard library plus pytest. The package metadata also contains a small mypy
configuration for the typed public boundary.
