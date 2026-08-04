# Jane Street-quality acceptance gates

**Date:** 2026-08-04
**Scope:** research-architecture upgrade for the topology-gated recursive quant proposal
**Reviewed:** `docs/architecture.md`, `docs/agent-contracts.md`, current `src/topology_gate`, current tests, and all prior reports
**Status:** initial gate-review snapshot. Follow-up implementation evidence is
maintained in [`reference-validation.md`](reference-validation.md); this
report's findings and counts should be read as the review baseline, not as a
replacement for that current evidence record.

## Executive decision

The repository is a useful, increasingly disciplined alpha scaffold. It is not
yet evidence for the five strong claims implied by the proposal:

1. exact persistent-Laplacian computation;
2. calibrated structural-change detection and justified adaptive forgetting;
3. end-to-end anytime-valid challenger promotion;
4. causal, restartable replay over real information vintages; or
5. economic value, dynamic regret, or profitability.

The current implementation and tests support narrower engineering claims. The
release posture must remain **research/alpha, offline, and exploratory** until
G0--G6 below pass. A green unit-test run, a favorable synthetic path, or a
single e-process crossing cannot waive a failed hard gate. This follows the
architecture's own release rule (`docs/architecture.md:547-560` and
`docs/agent-contracts.md:354-372`).

The smallest credible path is not a rewrite of the whole target repository
layout. Keep the current flat modules as compatibility adapters, add one
versioned reference persistent-Laplacian backend, one shared causal transition
function, and one contract-bound promotion controller. Defer distributed
execution, live order routing, portfolio construction, high-dimensional
topological searches, and broad hyperparameter sweeps.

## 1. Claim boundary: what is engineering versus evidence

| Area | Demonstrable engineering claim | Claim that still requires statistical, market, or external evidence |
|---|---|---|
| Current topology | `topology.py` implements a causal delay-embedded, robustly scaled kNN normalized-graph-Laplacian approximation. Its method name and documentation say so (`topology.py:1-23`, `:811-1001`). | It is not evidence of a persistent Laplacian, persistent homology, Betti change, or topology-based alpha. |
| Exact backend seam | A callable backend seam exists and is explicitly labeled (`topology.py:934-1001`). | A seam is not an implementation. Exactness requires a concrete algorithm, independent oracle, golden artifacts, and numerical residual evidence. |
| Detector causality | Prefix perturbation, current-before-reference calibration, finite-input checks, deterministic point ordering, and bounded stream state are testable engineering properties (`tests/test_topology.py`, `topology.py:1411-1464`, `:1527-1584`). | Causality does not imply stationarity, independent errors, a false-alarm level, average run length, power, or useful regime forecasts. |
| Calibration harness | `calibration.py` runs bounded seeded null/shift experiments and records Wilson intervals and censoring; `tests/test_calibration.py` tests determinism and shape/limit failures. | The current harness does not itself establish a market-relevant null, dependence validity, threshold selection validity, or a universal alarm guarantee. A small deterministic test is not a calibration study. |
| RLS | The RLS path has an independent weighted-batch oracle, bounded factors, finite-input checks, PSD/symmetry checks, and candidate-state commit discipline (`tests/test_rls.py`, `rls.py:478-1060`, `:1138-1392`). | A bounded heuristic forgetting map is not an estimated optimal adaptation policy, and no market result follows from RLS correctness. |
| E-process primitive | For a declared bounded score and a predictable `eta`, the product `∏(1 + eta_t X_t)` is implemented with non-negative factors, threshold crossing, alpha allocation helpers, and overflow rejection (`promotion.py:520-757`; `tests/test_promotion.py`, `tests/test_eprocess_null.py`). | The end-to-end promotion path does not yet bind prediction IDs, label availability, frozen paired scores, missing-label semantics, operational checks, or all selection/tuning choices to that process. The conditional theorem is not a certificate for arbitrary caller-supplied streams. |
| Positional replay | `run_recursive_rls` captures the factor before a delayed label, accepts explicit integer availability, returns terminal pending labels, and supports chunk continuation (`online.py:268-466`; `tests/test_online.py`). | Real causal replay needs event time, availability time, immutable source revisions, IDs, deterministic event precedence, late corrections, and one transition function shared by online and offline paths. A positional fixture is not vendor-grade point-in-time evidence. |
| Comparator metrics | Cost-matched comparator utilities, evaluated-row masking, feasibility checks, absolute discrepancy, and one-sided utility regret are implemented/tested (`backtest.py:829-888`, `:1018-1187`; `tests/test_metrics.py`). | The supplied comparator is not proved optimal or available at decision time. Synthetic returns are intentionally not an economic validation set (`synthetic.py:671-677`; `tests/test_backtest.py`). |
| Checkpoints | Authenticated envelopes are required by default, compatibility identities are checked by the explicit restore helper, and component states can be restored detached (`checkpoint.py:151-409`; `tests/test_checkpoint.py`). | Exact replay still requires a complete ledger/state snapshot, declarative and fingerprinted policies, dependency/backend identity, and fresh-versus-restored equivalence at every event boundary. |

The captured prior reports record green local engineering checks, but the active
workspace interpreter does not have `pytest`; this review did not rerun the
suite. The prior green result is therefore treated as historical evidence of
the tested snapshot, not as a new release certification.

## 2. Non-negotiable common contract

Every gate consumes and emits immutable, versioned records. The minimum fields
are:

- `record_id`, `run_id`, `model_id`/`comparison_id` where applicable;
- `event_time`, `available_time`, `decision_time`, and
  `label_available_time` as separate values;
- source revision or immutable input-snapshot ID;
- feature/topology/model/e-factor/config/backend/dependency identities;
- input and output fingerprints;
- status (`VALID`, `INSUFFICIENT_HISTORY`, `DEGRADED`, `INVALID`, `ERROR`, or
  `PROMOTED`) and a reason code;
- state version before and after the transition.

The canonical event order is `(available_time, event precedence, ingest
sequence, record_id)`. At a decision boundary the engine must materialize the
as-of set, build the feature/cloud state, score topology against the prior
reference, select the factor, predict, and freeze the pending record. A label
may score/update only at its availability boundary. A promotion is scheduled
for the next decision boundary. This is the ordering in
`docs/architecture.md:74-91` and `docs/agent-contracts.md:50-90`.

No component may silently turn an invalid or unavailable result into a stable
signal, neutral score, dropped denominator, or reset state.

## 3. Smallest credible reference implementation

### 3.1 Exact persistent-Laplacian MVP

The exact claim must be narrow and fully specified. The proposed MVP is:

- Input: the existing causal rolling delay-embedded cloud, after a
  past-only, versioned feature transform. The exact backend receives an already
  normalized cloud and must not normalize it a second time.
- Geometry: Euclidean distance; duplicate coordinates retain distinct
  canonical point IDs; point order is `(state_time, point_id)` and all simplex
  order is lexicographic by vertex IDs.
- Filtration: a Vietoris--Rips filtration on a fixed, pre-registered grid
  `epsilon_0 < ... < epsilon_m`, with inclusion rule `max pairwise distance <=
  epsilon_i`. The MVP hard-caps the cloud, number of grid values, and simplices
  before allocation; it does not truncate silently.
- Complex: simplices through dimension 2 for the first release. The first
  discriminating operator is `q=1`; q=0 may be emitted as a cross-check, not as
  the claimed higher-order signal. Coefficients are real numbers with the
  oriented boundary convention below. The coefficient convention, grid,
  dimension, pair list, and cap are part of the algorithm identity.
- Persistent pair: one fixed pair `(s, t)` with `s <= t`, or a fixed finite
  list of pairs declared before calibration. No data-dependent pair selection
  is allowed in the test region. A one-pair MVP is preferable to an
  undocumented all-pairs search.
- Boundary convention: for an ordered simplex
  `[v_0,...,v_q]`,
  `∂_q[v_0,...,v_q] = Σ_i (-1)^i[v_0,...,v_hat_i,...,v_q]`.
- For `K=K_s` and `L=K_t`, define
  `C_q^{L,K} = {c in C_q(L): ∂_q^L c is in C_{q-1}(K)}` and let
  `∂_q^{L,K}` be the restricted map into `C_{q-1}(K)`. The operator acts on
  `C_q(K)`. With canonical simplex bases, compute exactly the declared operator

  ```text
  Δ_q^(K,L) = ∂_(q+1)^(L,K) (∂_(q+1)^(L,K))*
              + (∂_q^K)* ∂_q^K
  ```

  This is the persistent-Laplacian definition for a simplicial pair; see
  [Mémoli, Wan & Wang, *Persistent Laplacians: properties, algorithms and implications*](https://arxiv.org/abs/2012.02808), especially its definition of the restricted chain space and operator. The implementation must construct a deterministic orthonormal basis for each restricted domain `C_(q+1)^(L,K)` (for example by a QR/SVD null-space calculation), express the restricted boundary in the canonical `C_q(K)` basis, and then form the two terms above. Taking an arbitrary boundary submatrix and multiplying it by its transpose is not an accepted substitute.
- Output: the full or explicitly top-`r` sorted eigenvalue vector for each
  declared `(q,s,t)` pair; persistent nullity; simplex counts; filtration and
  matrix digests; residuals; solver/backend identity; and status. Padding with
  an arbitrary number such as `2.0` is forbidden for a persistent spectrum.
- Numerical policy: pinned CPU float64 reference path, symmetric matrix check,
  finite/non-negative eigenvalue check within a declared tolerance, eigenpair
  residual check, deterministic tie rules, and fail-closed non-convergence or
  resource status. Any negative eigenvalue beyond tolerance is `INVALID`; a
  repair is not silently promoted to `VALID`.
- Reference update: score the current persistent spectrum against a reference
  made only from earlier committed spectra; only then make the current result
  eligible for the next reference. The current observation cannot calibrate
  itself.

This is enough to make an exact mathematical claim about one declared operator.
It is not a claim that every filtration, homology dimension, or market regime
has been covered.

### 3.2 Calibrated change and forgetting MVP

The detector output is a score, not a probability, until a study establishes a
finite-horizon operating point.

1. Freeze the exact backend/configuration, feature fit boundary, grid, pair/q
   set, CUSUM or alternative sequential statistic, horizon `H`, target
   false-alarm budget `alpha_det`, and selection budget before looking at the
   holdout result.
2. Define a null family that preserves the intended serial and cross-sectional
   dependence: at minimum a stationary/block bootstrap or a generative null
   fitted only on a pre-test region. IID noise is not an adequate proxy unless
   IID is the declared null.
3. Use an independent calibration split to select the threshold. Evaluate the
   final threshold on a separate null split. Report
   `P_0(any alarm in H)` with a confidence interval, alarm-time distribution,
   censoring convention, and resource/backend identity. Do not call the
   current `average_run_length` field a theoretical ARL; the current harness
   reports a censored descriptive mean.
4. Use a separate alternative family for detection power and delay. Include
   shifts not used for threshold selection and report detection probability,
   delay, censoring, and uncertainty.
5. Permit accelerated forgetting only when the topology result is `VALID`, the
   detector is in its calibrated operating region, and the pre-registered gate
   rule says `SHIFT`. `INSUFFICIENT_HISTORY`, `DEGRADED`, `INVALID`, and
   `ERROR` cannot authorize accelerated forgetting or promotion.
6. Record the exact factor selected at prediction time. Enforce one shared
   overlap invariant:

   ```text
   0 < lambda_min <= lambda_t <= lambda_max <= 1
   detector output bounds are a subset of RLS accepted bounds
   ```

   The mapping from score to factor is a policy, not evidence that the factor
   is statistically optimal.

### 3.3 Anytime-valid promotion MVP

The low-level `EProcess` is not the release promotion API. A release controller
must wrap it in a typed comparison stream:

- Create immutable incumbent/challenger `PredictionRecord`s before the matching
  label is available. Both use the same as-of cutoff, target definition,
  costs, and evaluation mask.
- At label availability, resolve exactly one pending prediction; reject
  duplicates, unknown IDs, future labels, missing/contaminated labels, and
  target revisions that do not match the declared source revision.
- Compute a paired score `X_t` from the frozen predictions and label. The
  accepted MVP score is a cost-matched utility difference clipped and
  normalized into `[-1,1]`.
- Use an e-factor `1 + eta_t X_t` only when `eta_t` is measurable from the
  pre-label filtration, lies in `[0,1]`, and has a written conditional-null
  argument. A caller-supplied numeric `eta` cannot be accepted as proof of
  predictability; the controller must resolve a registered declarative rule
  before the label is read.
- Give each comparison an immutable start boundary and centrally allocated
  alpha. Challenger slots, model/feature/eta searches, epochs, and reset
  actions must all consume the declared family budget. A reset with reused
  alpha is a new unaccounted test and is prohibited in the release path.
- Require minimum labels, burn-in, no unresolved/contaminated rows, data-quality
  limits, schema/config compatibility, numerical health, and operational checks
  in addition to `e_value >= initial_wealth / alpha_allocated`.
- Emit exactly one promotion event per comparison and make it effective only on
  the next decision boundary. A threshold crossing never retroactively changes
  earlier predictions, scores, or model states.

The current `PromotionGate` has useful geometric alpha allocation and state
serialization, but it accepts arbitrary scores/utilities and promotes on a
threshold crossing without the required frozen-prediction, label, burn-in,
operational, or next-boundary contract (`promotion.py:1431-1708`). Its reset
API also makes the alpha obligation a caller discipline (`promotion.py:794-810`).

### 3.4 Causal replay and recovery MVP

Implement one transition function and call it from both the online and offline
entry points. It must process a canonical event stream containing:

- immutable source revisions and source IDs;
- event and availability timestamps;
- decision events and label-availability events;
- deterministic same-time precedence and ingest sequence;
- pending prediction/label records with stored factor, feature fingerprint,
  topology/gate IDs, and model state version.

Required replay properties:

1. Appending or changing any record with availability after a decision cannot
   change that decision's ledger record, state fingerprint, factor, score, or
   promotion status.
2. A late label updates state only at its availability boundary; an emitted
   prediction is immutable. Duplicate delivery is idempotent; duplicate label
   identity cannot double-count.
3. One-shot replay, arbitrary chunked replay, and restore-and-continue at every
   event boundary produce the same canonical ledger and state fingerprints.
4. Invalid records, failed solvers, numerical failures, resource exhaustion,
   and checkpoint corruption leave the last committed state and ledger position
   unchanged.
5. A terminal pending-label ledger is returned and checkpointed. It is not
   silently dropped and it is not scored as zero.

The current `OnlineStreamState` is a good positional building block, but it is
not yet the full event-sourced contract. `backtest.py` and `online.py` also do
not yet prove that they call one transition function, and online summary
metrics currently aggregate rows without an availability-aware evaluation
denominator (`online.py:419-436`).

### 3.5 Economic evaluation MVP

The only supported economic target is an explicitly declared utility path:

```text
strategy_utility_t   = position_t * basis_t - strategy_cost_t
comparator_utility_t = comparator_position_t * basis_t - comparator_cost_t
basis_t              = realized_return_t
                         or point-in-time expected_return_t by declaration
```

The comparator must satisfy action, shorting, turnover, cost, and availability
constraints. If it is an offline oracle, label it as an evaluation reference;
do not imply it was available to the strategy. Report standard realized-path
metrics separately from comparator diagnostics. Use one-sided utility regret or
the explicitly named absolute comparator discrepancy; never report the legacy
`dynamic_regret` field as conventional dynamic regret.

Before an economic claim:

- use point-in-time, survivorship-bias-controlled market data with corporate
  actions, delistings, timestamps, spreads, fees, slippage, borrow/short,
  capacity, and market-impact assumptions;
- use chronological nested train/validation/test regions, with purge and
  embargo for overlapping labels;
- tune feature, topology, threshold, RLS, e-factor, and challenger choices only
  on prior regions; keep the final test untouched;
- compare against simple fixed and adaptive baselines under the same costs and
  evaluated mask;
- report uncertainty, turnover, drawdown, capacity sensitivity, regime
  sensitivity, missing-label counts, abstentions, and all selection attempts.

The synthetic regime fixture is suitable for control-layer and known-shift
tests, not economic promotion: its expected return is constant while the
latent optimal direction changes (`synthetic.py:671-677`). The current
`false_promotion` baseline comparison is descriptive selection diagnostics, not
an alpha-controlled false-discovery rate.

## 4. Prioritized acceptance gates

These are release blockers, not a scorecard. Each gate has to pass on the
pinned reference environment with named artifacts.

### G0 — Claim, contract, and identity freeze (P0)

**Pass criteria**

- Freeze the exact MVP conventions in one versioned configuration: timestamps,
  event precedence, feature schema, filtration, complex dimension, q/pair set,
  solver/tolerances, score/loss, factor policy, alpha allocation, purge/
  embargo, resource caps, dependency/backend identity, and seed derivation.
- Every public scientific record has the common provenance/status/fingerprint
  fields in section 2. No untyped mapping is the production boundary.
- The approximation path is named and isolated from the exact path. No default
  kNN result can satisfy an exact persistent-topology gate by naming alone.
- The run manifest records immutable input snapshot/vintage IDs, code revision,
  config digest, dependency lock, numerical backend, thread policy, and
  checkpoint lineage.

**Invariant tests**

- canonical config serialization is stable and changes digest when any
  state-affecting field changes;
- schema/version mismatch, non-finite values, units/shape mismatch, duplicate
  IDs, and unavailable timestamps fail before numerical work;
- public import aliases resolve to one meaning and no legacy metric name is
  silently reinterpreted.

**Current disposition:** partial. `config.py`, `py.typed`, root API tests, and
checkpoint identity fields exist; the target records and full run manifest do
not.

### G1 — Exact persistent-Laplacian correctness (P0)

**Pass criteria**

- The concrete reference backend implements section 3.1, not a callable seam
  around a hidden approximation.
- Boundary matrices and restricted chain spaces are independently checked;
  `B_q B_(q+1) = 0` where applicable; matrices are symmetric PSD within the
  declared tolerance; eigenpair residuals and nullity are recorded.
- `K=L` reduces to the ordinary combinatorial Laplacian. Hand-built pairs
  recover persistent Betti nullities from an independent homology/rank oracle.
- Filtration inclusion, duplicate/tie ordering, simplex closure, pair ordering,
  and resource-limit behavior are deterministic. Solver failure is explicit.
- Fresh and repeated reference runs produce the same canonical matrix/spectrum
  digest in the pinned environment.

**Invariant tests**

- a hand-computed edge, triangle boundary, filled/unfilled triangle, and a
  disconnected-complex fixture;
- pair tests for `K=L`, `K` strictly contained in `L`, and zero-dimensional
  restricted chain space;
- point permutation and duplicate-coordinate invariance;
- filtration monotonicity and `B_q B_(q+1)=0`;
- eigenvalue non-negativity/residual oracle and backend identity golden;
- current-point self-contamination and future-prefix perturbation tests;
- cap tests that prove oversized complexes abort without partial state.

**Current disposition:** FAIL. `topology.py`'s default is explicitly the kNN
normalized graph approximation; no concrete persistent-Laplacian implementation
or independent persistent-homology oracle is present.

### G2 — Calibrated change detection and forgetting (P0)

**Pass criteria**

- A pre-registered null and alternative family matches the intended data
  dependence and includes the exact detector, rolling transform, and reference
  update policy.
- Threshold selection and evaluation use independent data. The final report
  states finite-horizon false-alarm probability, confidence interval, censoring
  convention, power, delay, and selection budget. If the criterion is an
  upper-confidence bound, the criterion and confidence level are explicit.
- Calibration results are tied to detector/config/data/backend/seed digests and
  cannot be reused after any semantic change.
- Only calibrated `VALID` SHIFT results can accelerate forgetting. All degraded
  and insufficient states abstain. The factor is recorded before the label and
  accepted by RLS without coercion.

**Invariant tests**

- deterministic null/shift harness and stable result digest;
- iid-versus-block null negative control demonstrating that the declared
  dependence matters;
- future perturbation cannot change a calibration result before its availability
  boundary;
- threshold selection is not allowed to read the evaluation split;
- censoring and no-alarm cases retain their denominators;
- every emitted factor lies in both detector and learner bounds;
- invalid/degraded topology never emits accelerated forgetting or promotion.

**Current disposition:** FAIL for a calibrated claim, partial for engineering.
`calibrate_null`/`calibrate_shift` and their tests are a good harness, but the
repository contains no independent, dependence-aware, pre-registered market or
surrogate calibration result. The existing CUSUM recursion remains a heuristic.

### G3 — Recursive learner and transactional state (P0)

**Pass criteria**

- Fixed-factor scalar and multi-output RLS agree with an independently written
  batch oracle within the declared tolerance; adaptive factors are predictable,
  bounded, and frozen in the pending prediction record.
- Every update is candidate-state/validate/commit. Rejected input, duplicate
  label, invalid factor, non-finite residual, or unstable covariance leaves the
  prior state fingerprint unchanged.
- RLS, detector, pending-label, promotion, sequence, RNG/backend, and ledger
  state are all represented in a versioned checkpoint. Callable policies are
  either declarative/fingerprintable or the run is explicitly non-restorable.

**Invariant tests**

- independent scalar/multi-output recursive oracle;
- change the current label after prediction and verify prediction/factor are
  unchanged;
- rejected-update byte/logical fingerprint equality;
- lambda-bound overlap and schedule exhaustion;
- save/restore at every event boundary, corrupt/incompatible state rejection,
  and no-reset-on-failure.

**Current disposition:** partial/pass for the isolated RLS path; FAIL for the
full proposal. RLS tests are substantive, but the target pending prediction and
ledger contract are absent and callable policy identity is not a complete code
digest.

### G4 — Causal replay and recovery (P0)

**Pass criteria**

- Online and offline replay invoke the same transition function and produce the
  same decision/update/promotion ledger for the same canonical events.
- Event and availability time are distinct; source revisions are immutable;
  same-time ties are deterministic; future revisions/labels cannot alter an
  earlier output.
- Delayed, irregular, late, duplicate, out-of-order, and terminal-pending label
  cases have explicit records/statuses. No unresolved label is silently dropped
  or converted to a neutral score.
- Checkpoint restore produces identical ledger/state fingerprints and promotion
  boundaries. Integrity and compatibility are checked before state mutation.

**Invariant tests**

- prefix metamorphic test: append or mutate only future records;
- delayed-label test: label arrival changes only state after availability;
- equal-availability event precedence and deterministic replay;
- duplicate event/label idempotency and out-of-order rejection;
- one-shot versus arbitrary chunks, with restore at every boundary;
- injected failure at each commit stage leaves no partial state;
- terminal pending ledger is present in output and checkpoint;
- online/offline ledger equivalence on the same fixture.

**Current disposition:** partial. `OnlineStreamState`, explicit integer
availability, terminal pending labels, chunk continuation, and detector/RLS
state hooks are present and tested. Full event-time/source-revision semantics,
shared transition, and ledger equivalence are not.

### G5 — Anytime-valid, operationally safe promotion (P0)

**Pass criteria**

- A typed controller enforces the section 3.3 comparison contract; direct
  low-level `EProcess` use is not accepted as promotion evidence.
- The e-factor validity note states the null, filtration, bounded score,
  predictable tuning, missing-label policy, and all model/feature/eta selection
  budgets. The controller cannot observe current-label information when choosing
  eta or selecting a challenger.
- Alpha is preallocated over challengers, epochs, and any approved selection
  family. Resetting/restarting an active comparison without new alpha is
  rejected. There is one immutable comparison start boundary.
- Promotion requires threshold, burn-in/minimum labels, clean paired scores,
  operational/data-quality/numerical checks, and future activation. It is
  recorded once and cannot rewrite history.

**Invariant tests**

- frozen paired predictions and common target/availability ID;
- current-label mutation cannot change score, eta, e-value, or prior ledger;
- post-label eta injection is rejected;
- invalid, missing, duplicate, and contaminated score leaves e-process state
  unchanged and remains visible;
- threshold crossing is exact at any stopping time; first promotion is one-shot;
- reset consumes a new centrally allocated alpha share;
- geometric allocation plus adaptive registration never exceeds the global
  budget;
- null simulation under the declared conditional-mean process with optional
  stopping, challenger selection, and epoch behavior;
- promotion is scheduled for the next decision boundary, not the label event.

**Current disposition:** FAIL for end-to-end anytime-valid promotion. The
primitive's conditional algebra is coherent, but the current gate accepts
arbitrary score/utility calls and lacks the required data, lifecycle, and
operational binding.

### G6 — Economic evaluation and release evidence (P1)

**Pass criteria**

- The economic data contract proves point-in-time availability, universe and
  corporate-action treatment, tradeability, costs, slippage, shorting/borrow,
  capacity, and target horizon.
- Walk-forward evaluation uses purging/embargo where needed, nested selection,
  untouched final test, common costs, and a feasible comparator. All
  abstentions, unresolved labels, and selection attempts are reported.
- Report realized net path metrics separately from comparator diagnostics and
  include uncertainty/sensitivity. No report calls the legacy absolute gap
  `dynamic_regret`.
- Release artifact includes the full test/type/dependency/API/reproducibility
  evidence and a named benchmark/resource budget.

**Invariant tests**

- utility sign and cost symmetry; comparator feasibility and availability
  checks;
- evaluated-mask exclusion from return, cost, turnover, drawdown, and comparator
  sums;
- expected-return basis requires a point-in-time provenance flag;
- purge/embargo excludes overlapping future labels;
- model cannot inspect the oracle/comparator or future realized return;
- placebo and simple-baseline comparisons use the same event mask and costs;
- economic report refuses to call synthetic control results market evidence.

**Current disposition:** FAIL for economic evidence; partial for arithmetic.
The metric contract and tests are useful, but the repository has no market-data
study, point-in-time expected-return proof, capacity/cost evidence, or
statistical uncertainty for a trading claim.

## 5. Smallest implementation sequence

The order below minimizes rework and creates a usable research artifact after
each stage.

### Stage 0 — Freeze the vocabulary and stop unsafe claims

Deliver G0. Keep the current approximation available under its explicit name.
Publish a single status enum and the minimum immutable records. No exact or
economic claim is allowed to proceed on an unversioned interface.

### Stage 1 — Build the reference exact backend

Deliver G1 with tiny bounded clouds and golden hand complexes. Do not optimize
or add an accelerator until the reference matrices, nullities, spectra, and
fingerprints are independently correct. Integrate it as an explicit backend;
do not silently replace the current alpha default.

### Stage 2 — Make the causal transition real

Deliver G3 and G4 together. Add the minimal event ledger and use it from both
the online and offline wrappers. Checkpoint and restore at every boundary.
This stage is a prerequisite for trusting any calibration or promotion result.

### Stage 3 — Calibrate the detector and freeze the forgetting policy

Deliver G2 using the exact backend and the exact transition. First use a
declared synthetic/block null to validate the experiment, then a point-in-time
historical data study. Keep threshold selection separate from final evaluation.

### Stage 4 — Bind sequential promotion to frozen evidence

Deliver G5 only after the event ledger exists. The promotion controller owns
comparison IDs, alpha spending, score construction, missing labels, operational
checks, and activation scheduling. The low-level e-process remains a tested
mathematical primitive, not a free-form promotion API.

### Stage 5 — Run the economic study

Deliver G6 on an untouched, point-in-time market-data evaluation. The output is
an evidence report with costs, uncertainty, capacity, sensitivity, and all
selection/abstention counts. If it fails, retain the engineering result and
drop the economic claim; do not tune on the final test.

## 6. File-level work packages

These packages are deliberately scoped to the current flat repository. A full
move to the larger directory layout in `docs/architecture.md:110-185` is not a
prerequisite for the first credible reference release.

| WP | Priority / owner | Files | Deliverable and done test |
|---|---|---|---|
| WP-00 Claim and contract freeze | P0 / architecture + contract custodian | `docs/architecture.md`, `docs/agent-contracts.md`, `docs/statistical-validity.md`, `docs/production-runbook.md`, `README.md`, `src/topology_gate/types.py`, `src/topology_gate/config.py` | Versioned time/status/identity vocabulary, exact MVP convention, no contradictory release language, config digest tests, invalid-record tests, API inventory. |
| WP-01 Exact persistent-Laplacian reference | P0 / topology | **New:** `src/topology_gate/persistent_laplacian.py`; **adapter:** `src/topology_gate/topology.py`; `src/topology_gate/config.py` | VR filtration, bounded complexes, restricted chain spaces, exact `Δ_q^(K,L)`, eigen/residual/status record, backend identity. Add `tests/test_persistent_laplacian.py` and small golden fixtures under `tests/fixtures/`; attach an independent hand/rank oracle report. |
| WP-02 Detector calibration and forgetting | P0 / statistics + topology | `src/topology_gate/calibration.py`, `docs/calibration.md`, `src/topology_gate/topology.py`; **new evidence:** `reports/detector-calibration.md` | Pre-registered null/alternative manifest, block dependence, independent threshold evaluation, finite-horizon alarm CI, delay/power CI, censoring definition, factor-bound report. Add `tests/test_calibration.py` cases for split isolation, dependence, and future-prefix invariance. |
| WP-03 Causal contracts and one transition | P0 / data-time + orchestration | **New:** `src/topology_gate/replay.py` (or `pipeline.py`); `src/topology_gate/types.py`; `src/topology_gate/online.py`; `src/topology_gate/backtest.py` | Canonical event/label/prediction ledger, immutable revisions, deterministic precedence, shared transition, explicit unresolved statuses, online/offline equivalence. Add `tests/test_causal_replay.py`, `tests/test_leakage.py`, and `tests/test_transaction_boundaries.py`. |
| WP-04 Complete checkpoint/recovery | P0 / persistence | `src/topology_gate/checkpoint.py`, `src/topology_gate/online.py`, `src/topology_gate/topology.py`, `src/topology_gate/rls.py`, `src/topology_gate/promotion.py` | One envelope contains all state affecting future output, including ledger/pending labels, policies, sequence/RNG/backend identities. Restore is detached, authenticated, compatibility-checked, and equivalent. Extend `tests/test_checkpoint.py` with every-boundary replay, corrupt/mismatch, callable-policy, and no-partial-commit cases. |
| WP-05 Frozen paired promotion controller | P0 / monitoring | `src/topology_gate/promotion.py`; **new if kept separate:** `src/topology_gate/scoring.py` | Typed paired score, immutable prediction/label IDs, pre-label eta registry, alpha allocator for challengers/epochs/search, minimum labels/burn-in, operational checks, next-boundary activation. Add `tests/test_promotion_pipeline.py`, `tests/test_selection_budget.py`, and a conditional-null optional-stopping report. |
| WP-06 Economic evaluation contract | P1 / evaluation | `src/topology_gate/backtest.py`, `src/topology_gate/synthetic.py`, `docs/statistical-validity.md`; **new:** `src/topology_gate/evaluation.py` only if it prevents backtest overload | Common utility/cost basis, point-in-time comparator provenance, purge/embargo, explicit unresolved denominators, no legacy-regret reporting. Add `tests/test_economic_evaluation.py`; produce `reports/economic-evaluation.md` only from an external point-in-time market-data manifest. |
| WP-07 Release reproducibility | P1 / release + observability | `pyproject.toml`, `requirements-release-py312.txt`, `.github/`, `src/topology_gate/observability.py`, `docs/production-runbook.md` | Hash-locked environment, CI commands, backend/thread manifest, bounded/redacted audit export, benchmark workload, and a reproducible release report. No live-trading or broker scope is added. |

## 7. Explicit no-go conditions

The following are automatic stop-ship conditions:

- the default kNN normalized-Laplacian result is described as a persistent
  Laplacian or persistent homology result;
- a detector threshold is called `alpha`, ARL, or a controlled false-alarm rate
  without an independent dependence-aware calibration artifact;
- accelerated forgetting is enabled by `DEGRADED`/insufficient topology or by a
  factor not captured before the label;
- a caller can pass a current-label-dependent score, eta, challenger choice, or
  reset and still obtain promotion evidence;
- a promotion becomes active at the label event or rewrites a prior prediction;
- unresolved labels are silently dropped or scored as zero;
- a checkpoint restores under changed configuration/backend/dependencies or
  omits pending/ledger state;
- synthetic control paths, offline oracle positions, or expected returns without
  availability provenance are reported as market performance;
- the legacy `dynamic_regret` field is reported as conventional dynamic regret;
- any favorable backtest metric is used to waive G1--G5.

## Final disposition

The current repository can support a carefully labeled engineering alpha:
causal prefix computation, deterministic graph-spectrum approximation, bounded
RLS, a conditionally valid e-process primitive, positional delayed-label
replay, authenticated component checkpoints, and explicit comparator arithmetic.
It cannot yet support the original topology-gated recursive quant proposal as a
validated statistical or economic system. G0 is the immediate control; G1,
G3, and G4 are the critical engineering blockers; G2 and G5 are the inferential
blockers; G6 is the market-evidence blocker.

No implementation agent should claim completion from a passing local metric or
notebook. Completion means the named gate, invariant tests, immutable artifacts,
and evidence report all exist under the recorded code/config/data identities.
