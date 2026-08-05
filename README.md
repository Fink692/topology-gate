# Topology Gate

[![CI](https://github.com/Fink692/topology-gate/actions/workflows/ci.yml/badge.svg)](https://github.com/Fink692/topology-gate/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Typed, dependency-light research infrastructure for recursive quant models that
adapt their memory and promotion decisions under nonstationarity.

> Research status: alpha. This is an offline research and paper-trading
> repository, not an investment product or a live trading system.

## Why this repository exists

The hard question in a recursive model is often not how to add another
predictor. It is when the model should learn, what it should forget, and when a
challenger has earned the right to replace an incumbent. This repository makes
those control decisions explicit and testable:

- topology and change-detection adapters for adaptive forgetting;
- recursive least-squares learning with causal delayed-label replay;
- bounded e-process and robust expert promotion primitives;
- source manifests, calibration certificates, checkpoints, and audit receipts;
- synthetic experiments and free public-history diagnostics.

The core package does not claim that topology predicts returns, that a detector
is statistically calibrated for markets, or that a historical paper model will
be profitable in the future. Every result is labeled by its data boundary and
evidence status.

## Current evidence at a glance

The repository includes an intentionally simple SPY/GLD trend-filter paper
model as an economic sanity check. On the checked-in final adjusted-history
receipt, its 2023 onward holdout at a 5 bps turnover assumption reports 23.05%
annualized return, 1.51 net Sharpe, and 18.28% maximum drawdown across 899 rows.
The stricter annual walk-forward receipt reports 12.67% annualized return, 0.95
net Sharpe, and 24.61% maximum drawdown at 5 bps; it contains losing years,
including 2022.

These are historical diagnostics—not verified live performance. The source is
not point-in-time, does not establish delisting or capacity coverage, and the
code never places an order. See the [paper-model report](reports/trend-filter-paper-model.json),
[walk-forward report](reports/trend-filter-walkforward.json), and
[research status](docs/research-status.md) for the full limitations.

## Quick start

```bash
python -m pip install -e ".[dev]"
python -m pytest
ruff check src tests examples
mypy src
```

Run the reproducible synthetic study:

```bash
PYTHONPATH=src python examples/integrated_synthetic_study.py
```

Run the no-cost public ETF diagnostic (it downloads final adjusted history and
stores it outside the repository):

```powershell
$env:PYTHONPATH = 'src'
python examples\trend_filter_paper_model.py `
  --cache-dir "$env:TEMP\topology-gate-public-market-diagnostic" `
  --output reports\trend-filter-paper-model.json
```

For a guarded paper-only signal, use
[`examples/paper_signal_guard.py`](examples/paper_signal_guard.py). It checks
data freshness, caps exposure at 1.0, requires manual confirmation, and has no
broker or order-submission code.

## Project map

| Area | Entry points |
| --- | --- |
| Shared contracts | `src/topology_gate/types.py`, `src/topology_gate/config.py` |
| Recursive learning | `src/topology_gate/rls.py`, `src/topology_gate/online.py` |
| Topology and forgetting | `src/topology_gate/topology.py`, `src/topology_gate/pl_cusum.py`, `src/topology_gate/persistent.py` |
| Promotion and evidence | `src/topology_gate/promotion.py`, `src/topology_gate/evidence.py`, `src/topology_gate/calibration.py` |
| Causal replay and source boundaries | `src/topology_gate/replay.py`, `src/topology_gate/asof.py`, `src/topology_gate/study_package.py` |
| Research receipts | `reports/`, `docs/research-status.md` |
| Free paper-model track | `examples/trend_filter_*.py`, `examples/paper_*.py` |

## Scope and limitations

This repository does not provide a broker, live-data feed, account handling,
order management, or portfolio-risk service. Final adjusted public history is
useful for a low-cost diagnostic but is not point-in-time evidence. A licensed
source package with historical universe membership, delistings, revisions,
execution costs, and capacity evidence is required before making a market
claim; the intake checklist is in [`docs/vendor-data-gate.md`](docs/vendor-data-gate.md).

The default detector is a causal k-nearest-neighbour normalized-Laplacian
spectral approximation. It is explicitly not a persistent-Laplacian or
persistent-homology implementation. The e-process primitive is valid only for
its documented bounded conditional-mean score and predictable betting rule;
the caller must enforce data availability, selection, and risk controls.

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
suggested forgetting factor. The forgetting map includes an explicit,
identity-bound `forgetting_score_scale`; its default is `1.0`, while a
calibration-only score scale can be used when the accumulated CUSUM score is
not on a unit scale. It implements the same `observe`/stream-state
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

The experimental `CausalTransportReplay` adapter provides a prefix-only
location and linear-parameter transport for delayed historical observations.
It is explicitly not an adapted-Wasserstein solver; its availability rules,
reliability weights, and state identity are documented in
[`docs/causal-transport-replay.md`](docs/causal-transport-replay.md).

The optional `topology_gate.experts` module provides a Catoni robust,
full-information expert allocator with switching-cost penalties and explicit
change-point resets. It is a control-layer diagnostic, not a market or
promotion certificate; see [`docs/heavy-tail-expert-allocation.md`](docs/heavy-tail-expert-allocation.md).

The experimental `topology_gate.mechanisms` module provides a modular RLS
control with prefix-only residual diagnostics. During a localized shift it
updates only the declared shifted mechanisms and freezes the others; it does
not infer causal structure from observational data. See
[`docs/mechanism-localized-continual-learning.md`](docs/mechanism-localized-continual-learning.md).

The remaining proposal directions are also available as bounded research
prototypes: endogenous Wasserstein robustness, adaptive path-signature memory,
and finite martingale stress bridging. Their scope and evidence status are
listed in [`docs/research-status.md`](docs/research-status.md).

Set `require_realized_returns=True` in the worker backtest or online config for
economic evaluation. The compatibility default still permits target outcomes
to stand in for realized returns, but that path is diagnostic and must not be
reported as a tradable return result.

This is an offline control component. It has no broker, network, order, account,
or live-data side effect.

## Free reproducible research track

The full control-layer prototype and all proposed secondary directions have
synthetic receipts in `reports/`. A public, no-cost adjusted-ETF diagnostic is
available without a data subscription:

```powershell
$env:PYTHONPATH = 'src'
py -3.12 examples\public_market_diagnostic.py `
  --cache-dir "$env:TEMP\topology-gate-public-market-diagnostic" `
  --output reports\public-market-diagnostic.json
```

The follow-up filtration sensitivity and calibration-only score-normalization
experiments are documented in
[`docs/public-market-sensitivity.md`](docs/public-market-sensitivity.md) and
[`docs/public-market-score-normalization.md`](docs/public-market-score-normalization.md).
They are research diagnostics on final adjusted history, not point-in-time or
live-trading evidence.

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

For the vendor-to-model handoff, the exact field-level request is in
[`docs/vendor-handoff-request.md`](docs/vendor-handoff-request.md), and the
execution instructions are in [`docs/study-runbook.md`](docs/study-runbook.md),
which documents `StudyInputBundle`, `StudyTimeline`, and `run_causal_rls_study`. This
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

Use `calibrate_selection_null` for the broader pre-registered model/feature/eta
family. It charges every Cartesian selection cell an equal parent-alpha share,
simulates optional stopping across all cells, and records the first family
crossing. This is finite null evidence for the declared selection boundary; it
does not establish the conditional-mean assumption for market scores.

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
