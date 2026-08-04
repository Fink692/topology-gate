# Architecture: Topology-Gated Recursive Quant Model

Status: Accepted baseline for implementation
Date: 2026-08-04
Scope: Python research/production library; no order execution, broker integration, or portfolio construction

This document is the architecture contract for a topology-gated recursive quantitative model. The target model consumes an as-of event stream, constructs rolling market-state point clouds, detects structural change with persistent-Laplacian spectra, selects an adaptive RLS forgetting policy, and evaluates challenger models with an anytime-valid e-process before promotion.

The requested production architecture is not implemented end to end. The
workspace contains an alpha topology_gate scaffold plus bounded reference
contracts for exact finite persistence, as-of events, causal replay, numerical
detector/RLS migration, paired challenger promotion, run manifests, evidence
ledger composition, strict economic evaluation, and authenticated state.
These additions narrow and test
important boundaries; they do not yet constitute a live market-data engine,
full legacy-worker migration, cross-asset portfolio system, or
statistical/economic certification. The package names and interfaces below
define the target boundary.

**Current implementation claim boundary:** the default detector in the alpha
scaffold is a causal kNN normalized-Laplacian spectral approximation. It is not
a persistent Laplacian and does not compute persistence pairs or homology
groups. The bounded exact worker is now available through an explicitly
configured `PersistentLaplacianBackend` adapter, with finite filtration/solver
identity, spectrum-width checks, and transactional stream failure behavior.
Its CUSUM alarms and score-to-forgetting map, together with promotion
statistics, are exploratory until independently calibrated; cross-asset
point-in-time data and the full persistent-Laplacian/e-process architecture
remain release gates.

## 1. Requirements, assumptions, and hard invariants

### 1.1 Requirements

The library must:

- use the same causal state-transition logic for offline replay and online operation;
- represent event time and data-availability time separately;
- make every prediction, gate decision, model update, and promotion auditable;
- support delayed labels without retroactively changing an emitted prediction;
- keep topology, RLS, and challenger monitoring replaceable behind stable interfaces;
- be deterministic for a fixed data snapshot, configuration, dependency lock, and seed;
- fail closed when a signal, numerical result, or statistical comparison is not valid;
- make research results reproducible from a run manifest and immutable input snapshot.

### 1.2 Assumptions made for this greenfield baseline

These assumptions are explicit because the workspace contains no existing code or requirements:

1. The existing distribution/project identity is topology-gate and the import package is topology_gate. Retain that identity unless a separately approved migration changes the public import path.
2. The current project metadata supports Python 3.10+. The reference numerical path is CPU, float64, and uses a pinned NumPy/SciPy-compatible backend when the optional numerical extras are installed; the standard-library contract layer remains dependency-light.
3. A market state is represented by a fixed-schema numeric vector. A point cloud contains a bounded number of instrument/state points at an anchor time.
4. Targets are numeric scalar or fixed-width vector labels, commonly returns or forward returns, with an explicit horizon and label-availability timestamp.
5. The primary execution model is a single process with durable checkpoints. The design permits later parallel numerical backends without making distributed execution a prerequisite.
6. The scale, latency budget, data vendor, storage engine, and deployment platform are not specified. They are therefore configuration and release-gate inputs, not hidden architectural assumptions.
7. A source that cannot provide a trustworthy availability/as-of timestamp is not production-causal. It may be used only in an explicitly marked exploratory run.

### 1.3 Non-negotiable invariants

1. **Causality:** a decision at time t may depend only on records whose event and availability semantics make them knowable at t.
2. **Prediction-before-label:** a target cannot influence the prediction, topology gate, feature transform, e-factor, or challenger selection that precedes that target's availability.
3. **Immutable evidence:** emitted records are append-only. A correction is a new revision, never an in-place mutation of historical evidence.
4. **Transactional state:** an event either advances all required state and its audit record, or advances none of them.
5. **Deterministic ordering:** equal-time records have a canonical tie-breaker; no component may depend on unordered container iteration.
6. **Explicit abstention:** insufficient history, missing data, solver failure, invalid e-factors, and numerical instability produce a typed status or error. They never silently become a normal signal.
7. **Promotion is statistical and operational:** crossing an e-process threshold is necessary but not sufficient for production promotion.
8. **Versioned semantics:** changing a schema, numerical convention, feature definition, topology configuration, RLS update rule, e-factor, or promotion rule creates a new algorithm/configuration identity.

## 2. System shape

The library is a modular monolith with pure numerical components and an explicit stateful orchestration boundary. There is one causal event loop; storage, data vendors, and parallel numerical backends are adapters.

The high-level flow is:

    source vintages
        -> canonical as-of events
        -> feature/state rows
        -> rolling point cloud
        -> topology observation (default: causal kNN normalized-Laplacian approximation)
        -> topology gate and forgetting decision
        -> incumbent/challenger predictions
        -> label arrival
        -> paired score and e-process update
        -> RLS updates and promotion decision
        -> append-only ledger and checkpoint

At a decision time t, the sequence is strictly:

1. Materialize the information set available at t.
2. Materialize any cross-asset feature/state panel with one visible record per
   instrument, fixed field order, canonical instrument order, and an auditable
   universe digest; then build the feature row and point cloud using only that
   information set.
3. Compare the current topology to a reference state made from prior committed observations.
4. Convert the topology result into a gate decision and a bounded forgetting factor.
5. Predict with each active model using its state before the target is known.
6. Persist the prediction, topology evidence, gate decision, feature/panel
   fingerprints, and update policy as pending evidence.

When a label becomes available, the sequence is:

1. Match it to an immutable prediction by prediction_id.
2. Compute incumbent and challenger scores from the frozen predictions.
3. Update the relevant e-process using a pre-registered, predictable e-factor.
4. Apply the stored RLS update policy to the corresponding model states.
5. Evaluate promotion at the label boundary and schedule a promoted model to become effective on the next decision boundary.

The e-process and score therefore never observe an outcome before comparing the predictions that were made for it. RLS uses the topology policy selected at prediction time; it does not recompute a policy after the label arrives.

### 2.1 Boundary responsibilities

| Boundary | Owns | Must not own |
|---|---|---|
| Data adapters | Vendor parsing, source revisions, raw-to-canonical conversion | Feature definitions or model state |
| Causal data layer | As-of joins, time ordering, window membership, validation | Trading decisions or statistical promotion |
| Feature/state layer | Fixed-schema transforms and rolling state | Access to future labels |
| Topology layer | Point-cloud geometry, filtration, persistent Laplacians (target), spectral change | RLS covariance or challenger selection |
| Model layer | RLS prediction/update and forgetting policy | Data loading or e-process thresholds |
| Monitoring layer | Prequential scores, e-processes, promotion policy | Recomputing predictions |
| Pipeline/orchestrator | Event order, transactions, pending labels, active model set | Changing algorithm mathematics |
| Persistence/observability | Checkpoints, hashes, audit records, metrics | Mutating scientific state outside a transaction |

## 3. Target repository and package layout

The following is the intended repository shape. It is a design target, not an instruction to create these files during the architecture phase.

    repo/
    ├── pyproject.toml
    ├── src/
    │   └── topology_gate/
    │       ├── __init__.py                 # small, stable re-exports only
    │       ├── api.py                      # public facade and lifecycle
    │       ├── config.py                   # frozen, versioned configuration
    │       ├── errors.py                   # public exception hierarchy
    │       ├── version.py
    │       ├── py.typed
    │       ├── contracts/
    │       │   ├── __init__.py
    │       │   ├── ids.py                  # typed identifiers and fingerprints
    │       │   ├── time.py                 # timestamps and as-of context
    │       │   ├── market.py               # observations, revisions, labels
    │       │   ├── features.py             # feature schema and feature rows
    │       │   ├── topology.py             # clouds, spectra, gates
    │       │   ├── model.py                # predictions, RLS state, updates
    │       │   ├── monitoring.py           # scores, e-process, promotion
    │       │   └── run.py                  # run manifest and provenance
    │       ├── protocols/
    │       │   ├── __init__.py
    │       │   ├── data.py                 # source and sink protocols
    │       │   ├── features.py
    │       │   ├── topology.py
    │       │   ├── models.py
    │       │   ├── monitoring.py
    │       │   └── persistence.py
    │       ├── data/
    │       │   ├── canonicalize.py         # raw adapter boundary
    │       │   ├── validate.py
    │       │   ├── asof.py                 # causal joins and vintages
    │       │   ├── windows.py               # rolling membership
    │       │   └── point_clouds.py
    │       ├── topology/
    │       │   ├── filtration.py
    │       │   ├── complexes.py
    │       │   ├── persistent_laplacian.py
    │       │   ├── spectra.py
    │       │   ├── change_detection.py
    │       │   └── gate.py
    │       ├── models/
    │       │   ├── rls.py
    │       │   ├── forgetting.py
    │       │   └── state.py
    │       ├── monitoring/
    │       │   ├── scoring.py
    │       │   ├── e_process.py
    │       │   ├── challenger.py
    │       │   └── promotion.py
    │       ├── pipeline/
    │       │   ├── engine.py               # one causal transition function
    │       │   ├── replay.py
    │       │   ├── pending.py
    │       │   └── transactions.py
    │       ├── persistence/
    │       │   ├── checkpoint.py
    │       │   ├── serialization.py
    │       │   └── migrations.py
    │       ├── evaluation/
    │       │   ├── prequential.py
    │       │   ├── metrics.py
    │       │   └── reports.py
    │       └── observability/
    │           ├── audit.py
    │           ├── diagnostics.py
    │           └── metrics.py
    ├── tests/
    │   ├── unit/
    │   ├── property/
    │   ├── integration/
    │   ├── golden/
    │   ├── leakage/
    │   └── fixtures/
    ├── benchmarks/
    └── docs/

### 3.1 Dependency direction

The dependency graph is intentionally one-way:

    contracts/errors
        <- protocols/config
        <- data/topology/models/monitoring/persistence
        <- pipeline/evaluation
        <- api

Numerical packages may depend on contracts, config, and errors, but not on adapters, the public facade, wall-clock time, or global process state. Evaluation consumes records produced by the pipeline; it does not modify the model. The public facade is the only supported composition entry point for application code.

contracts, protocols, config, and errors are public namespaces. Algorithm modules are public only through explicitly documented protocol/factory symbols. Underscored modules and implementation classes are private and may change in a minor release.

## 4. Stable interfaces

The exact Python spelling may change before the first release, but the semantics and import stability below are binding. Public objects are typed, immutable at the boundary, and carry a contract version.

### 4.1 Public facade

TopologyGatedRecursiveModel is the stable lifecycle facade:

    create(config: ModelConfig, components: Components) -> TopologyGatedRecursiveModel
    predict(request: PredictionRequest) -> PredictionRecord
    update(label: LabelObservation) -> UpdateReceipt
    replay(events: Iterable[CausalEvent], run: RunSpec) -> ReplayReport
    snapshot() -> ModelSnapshot
    restore(snapshot: ModelSnapshot, *, expected_config: ModelConfig | None = None)
        -> TopologyGatedRecursiveModel

predict is pure with respect to scientific state until its prediction record is committed by the pipeline. update is transactional and idempotent by label_id. replay uses the same transition function as production processing and returns records plus a manifest, not a silently different research result.

The facade must not accept arbitrary keyword arguments, untyped dictionaries, or mutable global configuration. Convenience adapters may accept vendor-native objects outside the core boundary and must convert them into the contracts below.

### 4.2 Component protocols

The stable protocols are:

    AsOfDataSource.iter_events(request: DataRequest) -> Iterator[CausalEvent]
    FeatureBuilder.build(context: DecisionContext) -> FeatureRow
    PointCloudBuilder.build(context: DecisionContext) -> PointCloudWindow
    PersistentLaplacianDetector.detect(
        cloud: PointCloudWindow,
        reference: TopologyReferenceState,
    ) -> TopologyObservation
    ForgettingPolicy.choose(
        topology: TopologyObservation,
        *,
        policy: ForgettingConfig,
    ) -> GateDecision
    AdaptiveRLS.predict(
        row: FeatureRow,
        *,
        decision: DecisionContext,
    ) -> PredictionCore
    AdaptiveRLS.update(
        pending: PendingPrediction,
        label: LabelObservation,
    ) -> RLSUpdateReceipt
    PrequentialScorer.score(
        prediction: PredictionRecord,
        label: LabelObservation,
    ) -> ScoreRecord
    EProcess.update(
        state: EProcessState,
        score: PairedScore,
    ) -> EProcessUpdate
    PromotionController.evaluate(
        comparison: ComparisonState,
        *,
        operational_checks: OperationalChecks,
    ) -> PromotionDecision
    CheckpointStore.save(snapshot: ModelSnapshot) -> CheckpointReceipt
    CheckpointStore.load(checkpoint_id: str) -> ModelSnapshot

Protocols return contract objects rather than bare arrays, mappings, or booleans. A component may add private diagnostics, but it may not omit required fields or change the time semantics.

### 4.3 Configuration boundary

ModelConfig is a frozen, normalized object containing:

- feature schema and target specification;
- point-cloud window, metric, filtration, homology dimensions, complex limits, and spectral solver policy;
- topology-reference update policy and gate thresholds;
- RLS priors, covariance regularization, target scaling, forgetting-factor bounds, and update policy;
- score/loss definitions and e-factor parameters;
- challenger alpha allocation, burn-in, minimum labels, promotion and rollback checks;
- resource limits, numerical tolerances, missing-data policy, and seed;
- contract version, algorithm version, and configuration digest.

The digest is computed from canonical serialization after defaults are resolved. A component receives its relevant frozen subconfiguration; it cannot mutate or read process environment variables to change behavior.

### 4.4 Existing alpha surface and migration boundary

The current topology_gate 0.1.x scaffold exposes shared objects such as Observation, TopologySignal, RLSState, RLSUpdate, EProcessState, GateDecision, and narrow backtest/component protocols. It also contains worker-oriented flat modules for topology, RLS, promotion, synthetic data, backtesting, and observability.

Those symbols are treated as an alpha compatibility surface, not as the production contract defined here. In particular, an object without an explicit availability timestamp, source revision, schema identity, and state/version fingerprint cannot be used as production evidence merely because it has a compatible shape. Migration must:

- preserve the topology_gate import identity;
- add adapters or versioned replacements rather than silently changing the meaning of existing fields;
- keep old symbols working or explicitly deprecate them with a documented removal version;
- route new production code through the contracts and protocols in this document;
- avoid maintaining two independent implementations of topology, RLS, or promotion semantics.

The existing tests and modules are evidence of current behavior only. They do not waive any acceptance gate below.

## 5. Time model and data contracts

### 5.1 Time vocabulary

All core timestamps are signed 64-bit integer nanoseconds since UTC epoch. Naive datetimes are rejected at the adapter boundary. Wall-clock timestamps may be recorded for operations, but never influence scientific decisions.

| Name | Meaning | Used for |
|---|---|---|
| event_time_ns | When the market fact or target economically occurred | Window membership and target horizon |
| available_time_ns | Earliest time this exact revision could be known to the engine | As-of eligibility |
| decision_time_ns | Time at which a prediction/gate is emitted | Prediction cutoff |
| label_end_time_ns | End of the target interval | Target definition |
| label_available_time_ns | Time at which the target is safe to use | Model/e-process update |
| ingest_sequence | Canonical tie-breaker among equal availability times | Deterministic replay |

For every prediction:

    event_time_ns <= decision_time_ns
    available_time_ns <= decision_time_ns
    decision_time_ns < label_available_time_ns

The final relation is required for a label that will update that prediction. If a vendor supplies no trustworthy label availability time, the label is not eligible for production learning or promotion.

### 5.2 Common record requirements

Every persisted scientific record includes:

- record_type and contract_version;
- a globally unique record_id and, where applicable, run_id, model_id, and instrument_id;
- causal timestamps relevant to the record;
- source_revision or a source snapshot reference;
- config_digest and algorithm_version;
- a deterministic input_fingerprint and output_fingerprint;
- quality/status codes and diagnostic reason codes;
- a monotonic state_version when the record changes model state.

Arrays are validated at the boundary, copied or made read-only, and carry shape, dtype, feature-schema ID, and ordering metadata. NaN and infinite values are not implicit missing values. Missingness is represented by an explicit mask/status and is handled by a configured policy.

### 5.3 Canonical contracts

#### Market observation

MarketObservation contains observation_id, instrument_id, event_time_ns, available_time_ns, source_revision, instrument metadata, and the observed fields needed by the feature schema. Prices and volumes must have declared units/currency; non-positive or otherwise invalid values are rejected according to the field policy. A revision is a new observation identity or revision identity, not an overwrite.

#### Feature row

FeatureRow contains feature_schema_id, instrument_id or cross-sectional scope, state_time_ns, as_of_time_ns, a fixed-width vector x, an explicit missing/quality mask, and the fingerprints of every fitted transform used. A transform's fit boundary is recorded. A row is eligible only if each contributing input revision was available no later than as_of_time_ns.

Feature scalers, imputers, winsorizers, and encoders are stateful artifacts. They are fitted on a past-only prefix and updated only after the row's decision boundary. A feature schema change creates a new schema ID; positional reinterpretation is forbidden.

#### Point-in-time panel

PointInTimePanel is the bounded cross-asset adapter used when a feature or
state row is assembled from multiple instruments. It is selected from one
AsOfSnapshot by explicit record IDs and a fixed field schema. It requires one
record per instrument, sorts instruments by canonical identifier, and carries a
content digest together with the snapshot's universe digest. A plan with mixed
instrument-labelled and unlabelled bindings is invalid; a plan without
instrument labels remains an explicit legacy row path. The panel contract does
not claim vendor coverage, dynamic-universe completeness, or economic validity.

#### Point-cloud window

PointCloudWindow contains:

- anchor_time_ns and as_of_time_ns;
- the half-open event-time interval [window_start_ns, window_end_ns] or an explicitly configured closed interval;
- a read-only matrix of shape (n_points, dimension);
- canonical point_ids and each point's state_time_ns and available_time_ns;
- feature_schema_id, metric specification, filtration specification, and window-policy IDs;
- completeness, duplicate, late-data, and resource-quality flags.

The default membership rule is state_time_ns <= anchor_time_ns, available_time_ns <= as_of_time_ns, and state_time_ns >= window_start_ns. Points are sorted by canonical point ID and timestamp before any distance, complex, or eigensolver operation. n_points, dimension, and graph/complex size are bounded by configuration.

#### Topology observation and gate

TopologyObservation contains the current cloud/reference IDs, homology dimensions, coefficient field, filtration/grid digest, persistent summaries, sorted spectral values, spectral-change score, solver convergence diagnostics, and a status such as VALID, INSUFFICIENT_HISTORY, DEGRADED, or INVALID.

The detector's reference state is made only from topology observations committed before the current anchor. The current observation is scored before it is eligible to update that reference. The baseline cannot learn from the point being tested.

GateDecision contains gate_id, topology observation ID, mode (COLD_START, STABLE, SHIFT, or ABSTAIN), selected forgetting factor, bounds used, prediction/update permissions, threshold/config digest, reason codes, and a validity flag. SHIFT is the only mode allowed to accelerate forgetting. ABSTAIN may use a configured neutral fallback factor, but can never authorize accelerated forgetting or promotion.

The persistent-Laplacian backend must make deterministic choices for point ordering, duplicate points, distance ties, simplex ties, filtration ties, zero-eigenvalue ordering, eigenpair sign conventions when eigenvectors are exposed, regularization, and non-convergence. These choices are part of the algorithm version.

#### Prediction and pending update

PredictionRecord contains prediction_id, model ID and state version before prediction, decision/as-of time, feature schema and input fingerprint, prediction vector, topology observation ID, gate ID, selected forgetting factor, target specification, active comparison IDs, and an eligible_for_update status.

The complete feature vector or a content-addressed immutable reference to it must remain available until the label update is resolved. A prediction is frozen once committed. Re-running a topology detector after the label arrives cannot change it.

#### Label and score

LabelObservation contains label_id, prediction_id or an explicit target key, target value, target interval, label_available_time_ns, source revision, and validity status. It must not be accepted for update if its availability is at or before the associated decision without a documented target definition that makes that ordering valid.

ScoreRecord contains the frozen prediction ID, label ID, loss/reward definition ID, score, missingness/status, and the exact values or fingerprints used. Scores for challenger comparison are paired on the same prediction target and computed before either model is updated with that target.

#### RLS state and update

RLSState contains model ID, feature/target schema IDs, coefficient matrix, covariance or square-root covariance, prior/regularization policy, current effective sample size, last update availability time, update count, current condition diagnostics, state version, and fingerprint.

The update contract requires the pending prediction's feature vector, prediction-time gate, and forgetting factor. It must reject a mismatched schema, duplicate label, out-of-order state transition, non-finite input, invalid factor, or unstable covariance. A rejected update leaves the prior state byte-for-byte/logically unchanged.

The adaptive forgetting policy is bounded by configuration. It cannot select a factor outside [lambda_min, lambda_max], where 0 < lambda_min <= lambda_max <= 1. The policy must be predictable at decision time and recorded with the prediction. The policy is not permitted to inspect the future target or a post-label metric.

#### E-process and promotion

EProcessState contains comparison_id, incumbent/challenger IDs and config digests, alpha allocation, e-factor version, label count, current e-value in stable/log form, last label ID, status, and state version. Each comparison has an immutable start boundary and an append-only sequence of paired scores.

An e-factor must be non-negative and satisfy the declared conditional-validity contract under its null. Its tuning/betting choice must be measurable from information available before the current label. The implementation may store log e-values for numerical stability, but must preserve the semantics of a non-negative wealth process.

Promotion requires all of:

1. the e-value crosses the pre-registered threshold 1 / alpha_allocated;
2. the comparison has met minimum label and burn-in requirements;
3. no invalid, contaminated, or unresolved scoring condition is present;
4. operational checks pass, including schema/config compatibility, numerical health, missingness limits, and resource health;
5. concurrent challenger alpha allocation and selection policy permit the promotion.

Crossing the threshold is never a license to reset the e-process, alter the e-factor, select a more favorable start point, or retroactively include excluded labels. A promotion is effective at the next decision boundary and is recorded exactly once for that comparison.

## 6. Leakage and causal-safety rules

The default mode is strict as-of mode. Any violation below is a hard error for a production run and a visible LEAKAGE status for an exploratory run.

### 6.1 Information-set rules

- As-of joins use available_time_ns <= decision_time_ns, never merely event_time_ns <= decision_time_ns.
- A later vendor correction is not available at an earlier replay time. Backtests must use point-in-time vintages or explicitly label themselves as non-causal.
- A rolling window may include a past event whose value arrived late only if its revision was available by the decision cutoff. Future event times are excluded even if a data file already contains them.
- A point cloud's reference state, feature-transform state, imputation state, thresholds, and hyperparameters are updated only after the current decision has been emitted.
- Current and future labels, realized returns, order outcomes, and post-decision diagnostics cannot enter a prediction or gate.
- The same information set and decision timestamp feed the incumbent and challenger. A challenger cannot receive a richer feature set.
- Any post-label model update happens at label_available_time_ns, not at the target's economic end time.

### 6.2 Splitting and evaluation rules

- Evaluation is walk-forward/prequential, never a random split for time-dependent claims.
- Overlapping target horizons use a configured purge and embargo policy. The policy is part of the run manifest.
- Hyperparameter, feature, topology-threshold, and e-factor tuning occurs on a prior training/tuning region. The holdout/test region is not used for selection.
- If a model is refit or a feature state is recalibrated, the new artifact gets a new model/config identity and a clear activation boundary.
- Screening many challengers and promoting the best one requires pre-allocated alpha or a documented multiple-comparison procedure. Individual anytime validity does not justify ungoverned winner selection.
- Missing labels remain unresolved records; they are not silently dropped from denominators or converted to neutral scores.

### 6.3 Leakage audit evidence

Every production/research run produces a causal audit containing the maximum source availability time used by each decision, the decision cutoff, target availability, feature-fit boundary, topology-reference boundary, pending-label status, and any rejected late/future records. The audit is checked before a result is marked eligible for promotion.

## 7. Determinism and reproducibility

Reproducibility is a contract, not a best-effort notebook property.

### 7.1 Run identity

RunSpec and RunManifest record:

- immutable input snapshot/vintage IDs and source checksums;
- canonical event ordering rule and ingest-sequence source;
- normalized configuration and config_digest;
- package version, dependency lock, Python version, numerical backend/version, platform, and reference precision;
- root seed and component seed derivation scheme;
- algorithm/backend versions, resource limits, numerical tolerances, and feature/topology schemas;
- code revision and checkpoint lineage.

The root seed is never consumed directly by multiple components. A deterministic SHA-256-derived child seed is assigned by stable component name and run ID; Python's process-salted hash() and ambient random state are forbidden.

### 7.2 Canonical computation

- Sort all events by (available_time_ns, event precedence, ingest_sequence, record_id).
- Sort instruments, point IDs, simplices, filtration values, and eigenvalues with documented tie rules.
- Do not rely on unordered mapping/set iteration for numerical reductions.
- Use fixed float64 reference arithmetic and a configured thread count for acceptance tests.
- If an accelerator or approximate solver is used, it is opt-in, versioned, and must emit backend/tolerance metadata. It cannot silently replace the reference result.
- Fix eigenvector sign/phase conventions when vectors are part of a public artifact. If only eigenvalues are used, do not expose unstable eigenvectors as if they were reproducible.
- Use compensated or otherwise documented reductions where accumulation order affects acceptance metrics.

### 7.3 Checkpoint/replay equivalence

A checkpoint contains all scientific state: RLS coefficients and covariance, pending predictions, topology reference and baseline history, feature-transform state, active challenger/e-process states, sequence counters, configuration/version identities, and any RNG/backend state used by a component.

Restoring a checkpoint and replaying the same remaining canonical events must produce the same event ledger, state-transition fingerprints, promotion decisions, and predictions within the declared numerical policy. The pinned reference environment must pass bitwise/canonical-hash equivalence where the backend supports it; cross-platform tolerance must be explicit in the manifest.

No scientific decision may use wall-clock time, filesystem enumeration order, process ID, thread scheduling, or an unseeded random source.

## 8. Error handling, degradation, and recovery

The public hierarchy is rooted at TopologyGateError:

    TopologyGateError
    ├── ConfigurationError
    ├── ContractViolation
    │   ├── SchemaMismatchError
    │   ├── TimestampError
    │   ├── OrderingError
    │   └── DuplicateRecordError
    ├── LeakageError
    ├── DataQualityError
    │   ├── MissingDataError
    │   └── UnsupportedRevisionError
    ├── NumericalStabilityError
    ├── InsufficientHistoryError
    ├── EProcessValidityError
    ├── CheckpointError
    ├── StateTransitionError
    └── ResourceLimitError

Error messages include record IDs, model/comparison IDs, time boundaries, config digest, and a remediation reason. Secrets and raw credentials are never included.

### 8.1 Fail-fast versus typed degradation

| Condition | Behavior | State mutation |
|---|---|---|
| Invalid schema, naive timestamp, future/as-of violation, duplicate identity | Raise typed error; quarantine the record | None |
| Missing required feature or point-cloud history | Return ABSTAIN/INSUFFICIENT_HISTORY record | No accelerated forgetting or promotion |
| Persistent-Laplacian solver non-convergence or resource cap | Return DEGRADED topology result | Preserve last valid reference; no SHIFT gate |
| Non-finite RLS input or unstable covariance | Raise numerical/state error | None; retain last good RLS state |
| Invalid/non-predictable e-factor | Quarantine comparison | E-process unchanged; no promotion |
| Duplicate already-committed event/label | Return idempotent receipt | None |
| Corrupt/incompatible checkpoint | Raise checkpoint/version error | Do not start with a reset state |
| Optional telemetry failure | Emit an operational diagnostic | Scientific state may commit only if the audit sink's durability policy allows it |

Degradation is never silent. A fallback prediction, if enabled by configuration, carries a degraded status and cannot be used for promotion unless the promotion policy explicitly allows that status.

### 8.2 Transaction and recovery rules

1. Validate the complete event and all required cross-record references before computing a state transition.
2. Compute into a new immutable candidate state.
3. Validate invariants and append the audit record.
4. Atomically commit the state and ledger position.
5. Only then acknowledge the source event.

Checkpoint writes are atomic (temporary object plus integrity check followed by commit), include a monotonic state version, and are content-addressed. No code uses pickle for long-lived or untrusted checkpoint data. Schema migrations are explicit, one-way, tested transformations; an unsupported migration blocks startup.

## 9. Resource and numerical policy

The configuration must bound point count, feature dimension, filtration thresholds, simplicial-complex size, eigenpairs, pending labels, checkpoint size, and per-event compute time. Exceeding a bound is a typed resource outcome, not an implicit truncation.

The numerical policy must define:

- finite-value and range checks;
- covariance positive-definiteness/condition thresholds;
- regularization and reset policy;
- eigensolver convergence and residual thresholds;
- spectral-distance normalization and zero-spectrum handling;
- overflow/underflow handling for e-process values;
- comparison tolerances for replay and acceptance.

Numerical repairs such as clipping, symmetrization, or covariance reset are allowed only when explicitly configured, recorded as a diagnostic, and covered by a test. A repair cannot silently convert an invalid scientific result into a valid promotion result.

## 10. Observability and audit ledger

The library emits structured records for ingestion, as-of selection, feature construction, point-cloud construction, topology detection, gate choice, prediction, label resolution, score, RLS update, e-process update, promotion, checkpoint, and error/degradation.

Required audit dimensions include:

- causal cutoff and maximum availability time;
- source revision and input fingerprints;
- model state version before/after;
- topology/reference/gate IDs and status;
- selected forgetting factor and reason;
- prediction/label/score IDs;
- e-process value, alpha allocation, and promotion state;
- resource and numerical diagnostics.

Metrics are derived from the ledger and cannot alter model state. Logs must distinguish “no evidence,” “invalid evidence,” “abstained,” and “normal/stable”; these states are not interchangeable.

## 11. Acceptance gates

The implementation is not production-ready until all gates pass in the pinned reference environment and the evidence is attached to a run manifest.

1. **Contract gate:** public objects validate required fields, versions, shapes, units, finite values, and immutable boundaries. Backward/forward compatibility behavior is tested.
2. **Causal gate:** adversarial tests prove that changing any record with available_time_ns > decision_time_ns cannot change a prior output. As-of corrections, late arrivals, overlapping labels, and embargoes are covered.
3. **Topology gate:** filtration/complex tie rules, persistence summaries, Laplacian construction, eigenvalue ordering, convergence diagnostics, and reference-update boundaries have deterministic golden tests.
4. **RLS gate:** scalar and multi-output updates match an independently written recursive oracle under fixed lambda; adaptive factors are bounded, recorded, and never label-dependent; invalid covariance leaves state unchanged.
5. **E-process gate:** every e-factor has a written null/validity contract, predictable tuning proof, stable log implementation, no-reset test, and optional-stopping simulation with the pre-registered alpha policy.
6. **Challenger gate:** incumbent/challenger predictions are paired and frozen before labels; promotion is single-shot, thresholded, alpha-governed, operationally gated, and effective only at a future decision boundary.
7. **Replay gate:** a fresh run and checkpoint-restore run produce identical ledger/state fingerprints under the reference environment and declared tolerances.
8. **Fault/recovery gate:** malformed records, duplicate events, missing labels, solver failures, resource exhaustion, numerical instability, and corrupt checkpoints produce the documented typed outcomes with no partial state commit.
9. **Performance gate:** a named representative workload, resource budget, and concurrency setting are committed before release; the system stays within them without changing causal semantics.
10. **Release gate:** type checking, unit/property/integration/leakage tests, dependency-lock verification, public API documentation, and a reproducible example all pass. Any waiver is a time-bounded, signed architecture decision.

## 12. Architecture decisions and trade-offs

### ADR-001: Use a modular monolith with pure numerical components

**Context:** The domain has complex coupled state but no stated requirement for independent service scaling or multi-team deployment.

**Options considered:** distributed microservices; a notebook-first collection of functions; a modular monolith with explicit protocols.

**Decision:** Choose the modular monolith. Keep data adapters, topology, RLS, monitoring, and orchestration behind contracts, with pure numerical code wherever state is not required.

**Rationale:** It preserves one ordering and transaction model, makes replay practical, and keeps deployment simple while allowing later extraction of a proven bottleneck. A notebook-first design would make causal state and versioning implicit; microservices would add network, consistency, and deployment failure modes before they are justified.

**Trade-offs accepted:** One process is a scaling and fault-isolation limit; topology computation may require an opt-in backend later.

**Revisit trigger:** independently scaling workloads, team ownership boundaries, or operational availability requirements that cannot be met by a single process.

### ADR-002: Make availability time and event sourcing first-class

**Context:** Market data is revised and labels are delayed; final historical tables can leak future knowledge.

**Options considered:** final cleaned tables; event time only; immutable revisions ordered by availability with as-of materialization.

**Decision:** Use immutable source revisions and canonical event ordering by availability time with deterministic tie-breaks.

**Rationale:** This is the minimum design that can reproduce the information set actually available at each decision. Event time remains necessary for window and target semantics, but it is not a substitute for availability.

**Trade-offs accepted:** More storage, more complex joins, and more explicit missing/late-data handling.

**Revisit trigger:** only if a source contract can prove point-in-time correctness and the provenance remains reconstructible; even then, the core timestamps stay in the contract.

### ADR-003: Use typed immutable records instead of DataFrames as core interfaces

**Context:** Research workflows benefit from tabular tools, but DataFrame schemas, mutability, index semantics, and implicit dtypes are unstable at a production boundary.

**Options considered:** DataFrames everywhere; untyped dictionaries; frozen typed records with validated arrays and tabular adapters.

**Decision:** Use typed contract records in the core and provide adapters at the edges.

**Rationale:** Shapes, units, versions, timestamps, and ownership are explicit, and the same contracts work in replay and online paths. Researchers can still use DataFrames through adapters.

**Trade-offs accepted:** Conversion overhead and more verbose boundary code.

**Revisit trigger:** a measured, material conversion bottleneck with an equivalent immutable columnar contract.

### ADR-004: Use prequential anytime-valid e-processes for challenger promotion

**Context:** Repeatedly checking a live challenger and promoting after a favorable sample is vulnerable to optional-stopping and selection errors.

**Options considered:** fixed-horizon batch tests; repeated p-values; anytime-valid e-processes with alpha allocation.

**Decision:** Use an immutable comparison stream with pre-registered predictable e-factors and a promotion threshold, supplemented by operational checks.

**Rationale:** It matches sequential monitoring, preserves validity under continuous looks when its assumptions are met, and gives a durable audit trail. Alpha allocation remains necessary when many challengers are screened.

**Trade-offs accepted:** E-process design and calibration are more demanding than a single batch metric; it may promote slowly or never.

**Revisit trigger:** a formally reviewed alternative with equal or stronger sequential validity and better operating characteristics for the target domain.

### ADR-005: Maintain a deterministic CPU reference path

**Context:** Spectral and linear-algebra computations can vary with backend, thread scheduling, and approximate solvers.

**Options considered:** fastest available backend; hardware-specific reproducibility; pinned CPU reference plus explicitly versioned accelerators.

**Decision:** Make the pinned CPU float64 path the acceptance oracle. Accelerators are optional and must declare tolerances and backend identity.

**Rationale:** Scientific reproducibility and debugging require an authoritative result. Performance optimization can be added without changing the contract.

**Trade-offs accepted:** The reference path may be slower and may require bounded workloads.

**Revisit trigger:** a benchmark-backed need for another reference backend with a documented equivalence policy.

## 13. Definition of done for the architecture

This architecture is complete when implementation agents can identify, without interpretation, the data they may read, the state they may mutate, the public records they must emit, the errors they must raise, and the evidence required for acceptance. Any unresolved choice that changes causal semantics, public contracts, numerical validity, or promotion validity must be raised as an architecture decision before implementation.
