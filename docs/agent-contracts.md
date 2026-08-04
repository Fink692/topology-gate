# Agent Contracts

Status: Companion contract to docs/architecture.md
Audience: implementation, research, verification, and operations agents
Purpose: define ownership, handoffs, invariants, and acceptance evidence for the topology-gated recursive quant library

This document governs future work on the package. It is not permission to change scientific semantics informally. When this document and docs/architecture.md disagree, implementation stops and the lead architecture owner resolves the conflict through a versioned decision.

## 1. Shared operating contract

### 1.1 Authority and change control

- The architecture owner owns causal semantics, public record shapes, state-transition order, versioning rules, and promotion validity.
- A component owner may change private implementation details inside its assigned boundary.
- A component owner may not change a public field, timestamp meaning, default, error behavior, state-transition order, or statistical validity assumption without an architecture decision.
- A breaking public change requires a major contract version, migration plan, compatibility tests, and an explicit release note.
- A numerical optimization must first establish equivalence to the pinned reference path. Faster output with unmeasured semantic drift is not an acceptable optimization.
- If a requirement is ambiguous in a way that could alter leakage, e-process validity, model state, or public API behavior, the agent must stop at the boundary and escalate. It must not choose a convenient interpretation silently.

### 1.2 Rules every agent must follow

1. Read the relevant sections of docs/architecture.md before modifying code.
2. Work only inside the assigned scope unless the handoff explicitly grants a cross-boundary change.
3. Use typed contract records at boundaries; do not pass mutable dictionaries, DataFrames, hidden globals, or positional arrays without schema metadata.
4. Preserve event time, availability time, decision time, and label-availability time as separate values.
5. Treat input records and committed state as immutable. Produce a candidate state and commit transactionally.
6. Make ordering, tie handling, seeds, tolerances, and backend choices explicit and deterministic.
7. Never fill, drop, clip, reset, or impute silently. Emit the configured status/reason code and preserve the audit trail.
8. Do not use a target, realized return, future revision, post-decision metric, or future feature state to construct an earlier output.
9. Add tests for the negative case: demonstrate that invalid or future information is rejected or cannot affect an earlier result.
10. A handoff is incomplete until it includes changed paths, public symbols, invariants, tests, evidence, and known limitations.

### 1.3 Dependency and ownership map

| Workstream | Owns | May depend on | May not directly mutate |
|---|---|---|---|
| Contract custodian | contracts, config, errors, public protocols | None below the standard library/type layer | Algorithm state |
| Data/time | canonical records, revisions, as-of joins, windows | contracts, config, errors | Feature/topology/RLS state |
| Feature and point cloud | feature transforms, rolling state, point-cloud construction | contracts, data protocols, numerical utilities | Raw source revisions or model state |
| Topology | filtration, complexes, persistent Laplacians, spectra, topology gate | contracts, config, errors, deterministic numerical utilities | RLS covariance, labels, e-process |
| Adaptive RLS | RLS state, prediction/update, forgetting policy | contracts, config, errors, linear algebra | Topology reference or challenger state |
| Monitoring | scoring, e-processes, challenger lifecycle, promotion | contracts, config, errors | Prediction inputs or RLS state |
| Orchestration | event loop, ordering, pending labels, transactions, active model set | all approved component protocols | Component internals |
| Persistence/replay | checkpoint format, migrations, fingerprints, replay equivalence | contracts, errors, serialization backend | Scientific state outside a transaction |
| Evaluation/release | tests, leakage probes, benchmarks, reports, release evidence | public API and audit records | Production state |
| Observability | audit records, diagnostics, operational metrics | contracts, errors | Scientific decisions |

No state is “shared by convenience.” The owner exposes a protocol operation; another agent calls that operation and records the resulting ID/version.

## 2. Canonical event and state-transition contract

The orchestrator is the only component allowed to define event sequencing. All agents must preserve this order.

### 2.1 Decision transition

For a decision boundary t:

1. Canonicalize and validate all source revisions eligible by availability time.
2. Materialize the as-of information set and record its fingerprint.
3. Build the feature row using transform state committed before t.
4. Build the point cloud using the declared event-time window and as-of cutoff.
5. Run topology detection against the reference committed before this anchor.
6. Convert the topology result into a GateDecision and bounded forgetting policy.
7. Ask every active model for a prediction using its pre-label state.
8. Freeze predictions, topology evidence, gate policy, and pending-update references.
9. Only after the decision record is durable may the current topology/reference or feature state become eligible for the next boundary.

No label is read in this transition.

### 2.2 Label transition

For a label at its availability boundary:

1. Resolve a unique pending prediction and validate the target horizon and source revision.
2. Compute the frozen incumbent/challenger scores.
3. Update each eligible e-process from the paired score and its predictable e-factor.
4. Apply each model's stored prediction-time update policy to its own RLS state.
5. Evaluate promotion and schedule activation, if all statistical and operational gates pass.
6. Append the audit/ledger records and commit the new state atomically.

The model update never changes the score already used by the e-process. A promotion becomes active only at a subsequent decision boundary.

### 2.3 Idempotency and ordering

- The canonical event key is the source/revision identity plus event type. The canonical label key is label_id.
- Reprocessing an already committed event returns an idempotent receipt and does not increment state version.
- An event with an unknown predecessor, wrong schema, duplicate identity, or invalid time relation is rejected before numerical work.
- Events are processed in ascending available time, then event precedence, ingest sequence, and record ID. The precedence rule is versioned; it must specify how same-time market, decision, and label events interact.
- A label that arrives late is applied when it becomes available, not when its economic interval ended. Predictions already emitted remain unchanged.
- If an implementation requires labels to be strictly ordered for a mathematical update, it must return a typed ordering error and preserve the last good state; it may not silently reorder by target event time.

## 3. Component contracts

Each agent below owns an outcome, a protocol boundary, and a proof obligation. The named paths are targets from the architecture document; no agent should create a duplicate parallel implementation.

### 3.1 Contract custodian

**Objective:** establish the stable vocabulary used by every other component.

**Owns:** contracts, protocols, configuration normalization, public error hierarchy, version IDs, fingerprints, and schema validation.

**Must provide:**

- immutable records for all records listed in the architecture document;
- explicit shapes, dtypes, units, timestamp semantics, status enums, and version fields;
- canonical serialization for config and record fingerprints;
- validation errors that identify the failing field, record, and boundary;
- compatibility policy for each public contract;
- type-checking fixtures and representative valid/invalid records.

**Must not provide:** topology mathematics, RLS formulas, e-factor choices, adapter-specific parsing, or orchestration shortcuts.

**Acceptance proof:** a contract test can construct every public record, reject malformed variants, round-trip its canonical form, and demonstrate that a caller cannot mutate committed arrays or state through an alias.

### 3.2 Data and causal-time agent

**Objective:** turn source vintages into a trustworthy, deterministic information set.

**Owns:** source adapters, canonicalization, source revisions, as-of joins, event ordering, rolling window membership, late-data statuses, and causal audit inputs.

**Consumes:** vendor records and a DataRequest.

**Produces:** CausalEvent, MarketObservation, LabelObservation, DecisionContext, source fingerprints, and as-of audit evidence.

**Hard rules:**

- retain source availability/revision metadata;
- never substitute final revised data for a historical vintage;
- reject naive timestamps and ambiguous timezone conversions;
- never use event time alone to determine eligibility;
- make duplicate and same-time tie behavior deterministic;
- represent a missing value or unavailable revision explicitly.

**Must not:** fit model parameters, update feature transforms after the wrong boundary, compute a topology score, or discard a late record without a reason code.

**Acceptance proof:** adversarial as-of tests change future revisions, delivery delays, and same-time ordering. Prior decisions remain unchanged, and every rejected/late record appears in the causal audit.

### 3.3 Feature and point-cloud agent

**Objective:** construct fixed-schema market states and rolling point clouds without forward contamination.

**Owns:** feature transformations, fit/update boundaries, feature-state checkpoints, point identity, point-cloud windows, metric metadata, and resource estimates.

**Consumes:** DecisionContext and past-only feature state.

**Produces:** FeatureRow and PointCloudWindow with schema IDs, masks, as-of cutoff, window bounds, ordering metadata, and fingerprints.

**Hard rules:**

- every transform has a fit boundary and state version;
- the current row is scored using state committed before its decision;
- point membership obeys both event-time and availability-time constraints;
- points are sorted before all geometry and linear algebra;
- missingness is explicit; no accidental forward fill or global normalization;
- point and complex resource limits are checked before expensive work.

**Must not:** inspect labels, tune thresholds on the evaluation region, mutate topology reference state, or change feature meaning without a new feature-schema ID.

**Acceptance proof:** a synthetic future row, future correction, or post-decision transform update cannot change an earlier FeatureRow or PointCloudWindow. Fit/transform/replay state restores identically.

### 3.4 Topology and gate agent

**Objective:** produce a deterministic, auditable persistent-Laplacian spectral change observation and a conservative gate.

**Owns:** filtration construction, complex construction, persistent-Laplacian matrices, eigensolver policy, spectral summaries/distances, topology reference state, and GateDecision.

**Consumes:** PointCloudWindow and a reference state committed before the current anchor.

**Produces:** TopologyObservation and GateDecision.

**Hard rules:**

- document the metric, filtration, coefficient field, homology dimensions, threshold grid, matrix convention, regularization, and solver;
- define duplicate, distance-tie, simplex-tie, zero-eigenvalue, eigenpair-sign, and non-convergence behavior;
- score the current observation before adding it to the reference;
- return a typed insufficient/degraded/invalid status rather than inventing a stable baseline;
- only a valid SHIFT decision may select accelerated forgetting;
- no topology result can authorize challenger promotion directly.

**Must not:** read labels, modify RLS covariance, select a challenger, or hide a solver/resource failure behind a numeric zero.

**Acceptance proof:** golden point clouds with known symmetries and perturbations produce stable ordered spectra and status codes. Reference self-contamination and failed solver tests prove the current point cannot rewrite the baseline used to score itself.

### 3.5 Adaptive RLS and forgetting agent

**Objective:** implement a numerically stable recursive estimator whose forgetting policy is selected before the target arrives.

**Owns:** RLS coefficients, covariance/square-root state, prediction core, effective sample size, forgetting policy, update receipt, and numerical diagnostics.

**Consumes:** FeatureRow, DecisionContext, GateDecision at prediction time, PendingPrediction, and a validated LabelObservation at update time.

**Produces:** PredictionCore, RLSState, RLSUpdateReceipt, and numerical status.

**Hard rules:**

- prediction reads the state before the matching label;
- the pending record stores the exact feature vector/reference, gate ID, factor, and pre-update state version;
- update uses the stored factor; it never recomputes topology or policy after observing y;
- factors remain within configured bounds and are finite;
- invalid inputs or unstable covariance leave state unchanged;
- duplicate label updates are idempotent and cannot double-count an observation;
- any reset/regularization is configured, versioned, and auditable.

**Must not:** load data directly, change e-process values, use future performance to select lambda, or silently reset to a prior on numerical failure.

**Acceptance proof:** an independent scalar/multi-output recursive oracle agrees under fixed factors; adaptive-factor tests show label invariance; transaction tests show rejected updates preserve state fingerprints.

### 3.6 Scoring, e-process, and challenger agent

**Objective:** evaluate frozen prequential predictions and govern challenger promotion with an anytime-valid evidence process.

**Owns:** score/loss definitions, paired comparison records, e-process state, alpha allocation, challenger lifecycle, operational promotion checks, and activation scheduling.

**Consumes:** frozen PredictionRecords, LabelObservations, active comparison configuration, and operational health checks.

**Produces:** ScoreRecord, PairedScore, EProcessUpdate, ComparisonState, and PromotionDecision.

**Hard rules:**

- predictions are made and frozen before their labels are eligible;
- incumbent and challenger use the same target and information boundary;
- the e-factor is non-negative, versioned, and predictable from pre-label information;
- a comparison has an immutable start boundary and may not be reset after monitoring begins;
- many concurrent challengers require pre-allocated alpha or an approved multiple-comparison policy;
- the complete challenger family is registered and sealed before the first observation;
- invalid/missing/contaminated scores remain visible and cannot silently improve a denominator;
- promotion is single-shot, operationally gated, and effective at the next decision boundary.

**Must not:** retrain a challenger based on the label it is currently scoring, cherry-pick a favorable start, tune the e-factor after observing scores, or replace e-process evidence with an informal metric threshold.

**Acceptance proof:** null simulations and optional-stopping tests respect the declared alpha policy; threshold crossing is reproduced exactly; reset, missing-label, alpha-overallocation, and post-label tuning tests fail closed.

### 3.7 Pipeline and orchestration agent

**Objective:** compose components into one causal, idempotent, transactional state machine.

**Owns:** the canonical transition function, active model registry, pending-label registry, event precedence, transaction boundaries, lifecycle status, and public facade composition.

**Consumes:** CausalEvents and approved component protocols.

**Produces:** ordered audit records, PredictionRecords, UpdateReceipts, ReplayReports, and committed ModelSnapshots.

**Hard rules:**

- enforce the decision and label sequence in section 2;
- validate before numerical work and commit only once all required records are ready;
- never reach into a component's private state;
- preserve component statuses and reason codes;
- make fresh replay and online processing call the same transition logic;
- schedule, rather than retroactively apply, a promotion;
- provide idempotent receipts for replays and duplicate deliveries.

**Must not:** introduce a second research-only processing path, reorder by target event time, mutate feature/topology/model state directly, or swallow typed errors.

**Acceptance proof:** integration tests trace one event from source through ledger, inject failures at each commit stage, restart from checkpoints, and verify no partial transition or causal reorder.

### 3.8 Persistence, checkpoint, and determinism agent

**Objective:** preserve enough state and provenance to reproduce every scientific decision.

**Owns:** checkpoint envelope, content fingerprints, serialization, atomic writes, schema migrations, restore validation, and replay comparison.

**Consumes:** ModelSnapshot and RunManifest.

**Produces:** durable checkpoint receipts, migration reports, and equivalence reports.

**Hard rules:**

- serialize all state named in the architecture document, including pending labels and topology/feature references;
- include config, algorithm, contract, dependency, backend, seed, and data-snapshot identities;
- reject incompatible/corrupt snapshots; never start by silently resetting state;
- do not use pickle for long-lived or untrusted state;
- make checksums and state versions monotonic;
- distinguish reference bitwise equality from declared cross-platform tolerance.

**Must not:** omit “temporary” state that can affect a future decision, use filesystem order as a sequence, or hide a migration in a loader.

**Acceptance proof:** save/restore at every event boundary, corrupted-checkpoint tests, migration tests, and fresh-versus-restored replay produce equal ledger/state fingerprints.

### 3.9 Evaluation, verification, and release agent

**Objective:** turn the architecture invariants into independent evidence.

**Owns:** unit/property/integration/leakage/golden tests, synthetic fixtures, benchmarks, test data provenance, reproducibility reports, and release checklist.

**Consumes:** public API, audit ledger, run manifests, and component outputs.

**Produces:** pass/fail evidence linked to acceptance gates, not just aggregate metrics.

**Hard rules:**

- include negative/adversarial tests for every leakage and failure rule;
- compare to independent oracles where formulas are used;
- keep test fixtures point-in-time and versioned;
- report abstentions, missing labels, degraded topology, and rejected updates separately from normal observations;
- record workload, hardware, thread count, dependency lock, and tolerance for benchmarks;
- never modify production configuration or state in order to make a test pass.

**Must not:** certify a component from only a happy-path test, tune implementation to a single favorable dataset, or treat statistical significance as an operational readiness check.

**Acceptance proof:** all release gates in section 6 have named evidence files and a reproducible command/run identity.

### 3.10 Observability agent

**Objective:** expose the ledger and diagnostics needed to audit causality, numerical health, and operations.

**Owns:** structured audit schemas, metrics, reason-code taxonomy, redaction, and sink adapters.

**Consumes:** immutable contract records and transaction results.

**Produces:** append-only audit events and derived operational metrics.

**Hard rules:**

- telemetry cannot change scientific state;
- every diagnostic links to a record, component, state version, or run;
- distinguish warning, abstention, degradation, error, and promotion;
- redact credentials and sensitive payloads;
- preserve causal timestamps and use wall-clock time only for operations.

**Acceptance proof:** forced failures produce the expected reason codes and correlation IDs, while telemetry sink failure obeys the configured durability policy without inventing scientific success.

## 4. Cross-agent handoff contract

Every completed workstream submits a handoff with exactly these fields:

1. **Objective and scope:** what was implemented and what was deliberately excluded.
2. **Changed paths:** exact files and public symbols touched.
3. **Contract impact:** schemas, defaults, errors, versions, serialization, or state transitions changed.
4. **Inputs/outputs:** record types, required fields, units, shapes, time boundaries, and fingerprints.
5. **Invariants proved:** causality, determinism, idempotency, numerical, or statistical properties.
6. **Failure behavior:** typed errors, degraded statuses, state-mutation behavior, and recovery.
7. **Tests and commands:** exact commands, fixture/run IDs, environment, and result.
8. **Performance evidence:** workload, resource limits, backend, thread count, and measured result.
9. **Known limitations:** unresolved questions, assumptions, and safe operating bounds.
10. **Follow-up or decision request:** any issue that cannot be resolved within the owned boundary.

A downstream agent may consume a handoff only if the contract-impact section is explicit. “Works locally” is not evidence.

## 5. Shared failure and degradation matrix

All agents use the same semantic outcomes:

| Outcome | Meaning | Allowed next action |
|---|---|---|
| VALID | Evidence satisfies the component contract | Continue normal transition |
| INSUFFICIENT_HISTORY | Required causal history is not available yet | Abstain/cold-start behavior; no accelerated forgetting or promotion |
| DEGRADED | A bounded fallback is available but the preferred computation is not valid | Continue only under configured fallback; emit diagnostic; no promotion by default |
| INVALID | Evidence violates the component's mathematical/data contract | Quarantine or raise; do not update state |
| ERROR | The transition cannot be completed safely | Abort transaction; retain last committed state |
| PROMOTED | A challenger passed statistical and operational gates | Activate only at the scheduled future boundary |

An agent must not reinterpret another agent's status. For example, DEGRADED topology is not equivalent to STABLE topology, and an unresolved label is not an average score of zero.

## 6. Acceptance gates and ownership

These gates are release blockers. A gate may be waived only by the lead architecture owner with a written reason, scope, expiry date, and follow-up owner.

| Gate | Pass criteria | Required evidence | Primary owner |
|---|---|---|---|
| G0 Architecture/API freeze | Public names, versions, defaults, and transition order are reviewed | Approved architecture decision and API inventory | Contract custodian + architecture owner |
| G1 Data contract | All records validate timestamps, schemas, units, provenance, shapes, and immutability | Contract tests and invalid-fixture report | Contract custodian |
| G2 Causal safety | Future revisions/labels cannot affect earlier outputs; purging/embargo is enforced | Adversarial leakage suite and causal audit sample | Data/time + evaluation |
| G3 Feature/cloud safety | Fit boundaries, window membership, point ordering, missingness, and resource bounds are reproducible | Transform replay report and point-cloud goldens | Feature/cloud |
| G4 Topology correctness | Deterministic filtration/Laplacian/spectrum behavior and conservative failure statuses | Mathematical invariants, golden outputs, failure tests | Topology |
| G5 RLS correctness | Recursive oracle agreement, bounded adaptive forgetting, safe numerical failure | Oracle comparison, state-fingerprint transaction tests | Adaptive RLS |
| G6 Sequential validity | E-factor contract, predictable tuning, alpha allocation, no reset, threshold semantics | Reviewable validity note and null/optional-stopping evidence | Monitoring |
| G7 Promotion safety | Paired frozen predictions, one-time promotion, future activation, operational checks | Promotion event ledger and adversarial challenger tests | Monitoring + orchestration |
| G8 Replay/recovery | Fresh and restored runs match; corrupt/incompatible state is rejected | Checkpoint matrix, migration report, replay equivalence | Persistence |
| G9 Operations/performance | Named workload stays within declared resource/latency/error budgets | Benchmark manifest and diagnostics dashboard/report | Evaluation + observability |
| G10 Release | Full test/type/dependency/documentation checks pass | Reproducible release report tied to code/data/config hashes | Evaluation/release |

No single aggregate score can substitute for a failed hard gate. In particular, good backtest performance cannot waive causal, numerical, or e-process validity.

## 7. Required test shapes

Every implementation agent contributes at least one test from each applicable category:

- **Contract:** valid/invalid construction, schema mismatch, dtype/shape/units, immutable ownership.
- **Causality:** future record perturbation, late revision, label timing, fit/reference boundary.
- **Determinism:** repeated execution, tie ordering, fixed seed, checkpoint restore.
- **Failure:** malformed input, non-finite value, resource cap, numerical failure, duplicate delivery.
- **Oracle/golden:** independent formula or fixed expected artifact where the component has mathematical output.
- **Integration:** interaction with the next boundary, including status and reason-code propagation.

Property tests must include minimal and adversarial cases, not only random large cases. Golden tests must record the algorithm/backend/config identity so an intentional change cannot look like an unexplained fixture drift.

## 8. Escalation rules

An agent escalates to the architecture owner when:

- a source lacks a trustworthy availability timestamp;
- a desired feature requires future data or a target before its label boundary;
- a topology fallback would change a gate from SHIFT to STABLE or vice versa;
- an RLS numerical repair would change historical state;
- an e-factor needs post-label tuning, a reset, or unplanned multiple-challenger selection;
- a checkpoint cannot represent all state affecting future output;
- a performance optimization changes reference outputs beyond the declared policy;
- a requested dependency, backend, or public symbol is not in the approved configuration;
- a bug fix would alter a public contract or causal transition.

The escalation includes the smallest reproducible record sequence, configuration digest, observed status/error, expected invariant, and at least one safe alternative. The agent continues only on work that is independent of the unresolved decision.

## 9. Agent definition of done

An agent is done only when:

1. its owned protocol and state boundaries are implemented or explicitly marked unimplemented;
2. its public outputs contain required provenance, status, and version data;
3. its state transitions are transactional and idempotent where required;
4. its tests include the relevant negative and leakage cases;
5. its deterministic/replay behavior is demonstrated;
6. its failure and degradation behavior matches the shared matrix;
7. its handoff is complete and contains reproducible evidence;
8. no unowned or undocumented behavior was introduced.

The architecture owner closes the workstream only after the handoff can be reviewed without reading private implementation details. A passing metric or a successful notebook run alone is not completion.
