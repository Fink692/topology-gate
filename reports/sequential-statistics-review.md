# Sequential-statistics review

Review date: 2026-08-04
Scope: `promotion.py`, `online.py`, `backtest.py`, the current tests, and
`docs/statistical-validity.md`.

This review specifies the minimum evidence ledger required for a promotion
claim. No source or test file was changed for this review.

## Decision

The product in `promotion.py` is a valid sequential primitive under its stated
conditions. The repository does not yet expose a valid composed promotion
pipeline. A passing unit test, a walk-forward replay, or a `PromotionGate`
threshold crossing must therefore remain an exploratory result, not an
alpha-controlled promotion decision.

The distinction is:

* **Valid primitive:** for a fixed score stream (X_t\in[-1,1]), a fixed
  initial wealth, a predictable η_t in `[0, 1]`, and the conditional null
  `E[X_t | F_(t-1)] <= 0`, `E_t = product(1 + eta_t * X_t)` is a nonnegative
  supermartingale. Ville's inequality gives a crossing probability at most
  α for the one allocated stream.
* **Valid composed pipeline:** the system must prove that every score was
  generated from a prediction frozen before its target label was available,
  that labels were settled by a predeclared point-in-time/missingness policy,
  that eta was fixed from pre-score information, and that every adaptive
  challenger/model/eta/epoch choice was paid for by an allocation whose total
  is at most the global alpha.

`EProcess` supplies the first item only. `PromotionStateMachine` adds a first
crossing state transition. `PromotionGate` adds a conservative geometric
allocation for registered challenger slots and epochs. None of those objects
can, by themselves, establish the second item.

## What the current code establishes

### Promotion primitive

The module docstring correctly states the bounded score and conditional null
(`src/topology_gate/promotion.py:1-18`). `_validate_score` enforces `[-1, 1]`,
`validate_eta` enforces `[0, 1]`, and `EProcess.update` resolves a callable eta
from prior score history before appending the current score
(`promotion.py:611-757`). The nonnegative factor and optional-stopping
threshold are consequently sound for the stated primitive.

The important limitation is that `EProcess.update(..., eta=...)` accepts a
number supplied at update time. The object cannot tell whether that number was
computed from the current label, a future feature, or a post-outcome model
selection. A callable is constrained to receive prior scores, but a callable
can still close over mutable external state. Predictability is currently a
caller obligation, not an auditable pipeline invariant.

`PromotionStateMachine.observe_utilities` correctly clips and normalizes the
utility difference (`promotion.py:1202-1254`), but it accepts arbitrary utility
pairs. It does not identify the prediction that produced either utility, the
target label, or the availability boundary.

### Alpha allocation

`geometric_alpha_allocation` uses

```text
alpha(slot, epoch) = global_alpha * 2^(-slot) * 2^(-(epoch + 1))
```

for one-based slots and zero-based epochs (`promotion.py:344-370`). The
infinite family sum is at most `global_alpha`, and the first slot in epoch zero
gets `global_alpha / 4`. `PromotionGate` reserves these shares when a
challenger is registered or reset (`promotion.py:1431-1601`,
`1739-1837`). This is useful accounting, but `alpha_spent` is reserved alpha,
not evidence consumed, and the allocation does not pay for arbitrary model,
feature, score, eta, or start-time search. Direct `EProcess` and
`PromotionStateMachine` resets can also repeat testing with the same alpha
(`promotion.py:794-810`).

### Online and walk-forward paths

`run_recursive_rls` predicts before applying a label, captures the forgetting
factor at prediction time, and returns terminal pending label records
(`online.py:268-289`, `362-444`). Its `OnlineStreamState` is a useful causal
replay state. It is not a promotion evidence ledger: outcomes must be a finite
array, missing values have no typed status, pending records identify source
steps rather than stable label/prediction IDs, and the state does not include
promotion evidence or alpha allocation.

`WalkForwardBacktest.run` has the strongest existing point-in-time rule:
training requires both `target_position < decision_position` and
`availability < decision_position` (`backtest.py:1228-1235`,
`1353-1364`). `training_positions` makes that rule testable. However, the
result does not freeze prediction receipts, attach model/config provenance, or
produce an evidence record that can later be joined to a label revision. The
baseline comparison in `compare_to_baseline` is a descriptive early-window
selection diagnostic, not an e-process test.

The current documentation is appropriately narrow: it calls this a research/
alpha contract and explicitly says that an isolated e-process does not certify
an arbitrary promotion stream (`docs/statistical-validity.md:3-9`, `67-94`).

## Minimum evidence ledger

The minimum addition should be one append-only `PromotionEvidenceLedger`
around the existing `PromotionGate`, with two typed inputs and one canonical
ordered record stream. It does not need to replace the detector, RLS, or
backtester.

### 1. Immutable run contract

Add a frozen `PromotionEvidenceConfig` (or equivalent manifest) containing:

* `run_id`, `family_id`, incumbent identity, global alpha, initial wealth, and
  score-bound/version;
* a stable `score_spec_id`, utility/cost convention, and null-hypothesis ID;
* an `eta_policy_id` plus a canonical policy/config fingerprint;
* a `missing_label_policy_id`, terminalization deadline, and the rule stating
  whether missingness is assumed predictable;
* the candidate/model/feature selection budget and alpha-allocation rule
  version; and
* package/config/backend/dependency identities needed for checkpoint restore.

Every ledger record carries the relevant IDs or a digest of this manifest.
Changing any of these values starts a new family/epoch and cannot mutate an
existing ledger.

### 2. Frozen prediction receipt

Add an immutable `FrozenPrediction`/`PredictionReceipt` with at least:

```text
prediction_id       stable unique ID, never a recycled row number
family_id, challenger_id, incumbent_id, slot, epoch
decision_step, target_id, target_time
challenger_action, incumbent_action   frozen numerical inputs to the utility
model_fingerprint, feature_fingerprint, training-set/provenance digest
score_spec_id, eta_policy_id, eta       resolved before the target label
alpha, allocation_id, prior-state digest
```

The receipt is created at the decision boundary. It stores the actual
prediction/action values, not merely a pointer to a mutable model. The utility
function used at settlement must be the pre-registered `score_spec_id`; the
settlement API must not accept a fresh arbitrary score or utility pair. If a
domain needs a richer action, the canonical serialized action and its digest
must be stored.

The receipt must be created before the target label is available. A target
already known at the decision boundary is either excluded from the evidence
family or explicitly handled by a separately justified design; it must not be
silently treated as future evidence.

### 3. Point-in-time label record

Add a typed `PointInTimeLabel`/`LabelReceipt` rather than using `NaN` as a
missing-label sentinel:

```text
label_id, target_id, value | None
available_at, received_at
status = observed | missing | expired
source_id/source_revision
```

Required rules:

1. `available_at` is source truth, not the time the worker happened to ingest
   the record. A settled label must satisfy
   `decision_step < available_at <= settlement_step`.
2. `observed` values are finite. `None` is only legal for a terminal missing
   status.
3. `label_id`/`target_id` uniqueness is enforced. A duplicate is rejected.
   A late correction never rewrites a sealed record; it is a separate audit
   event and, if corrections are to be used, requires a new data revision and
   a separately funded family/epoch.
4. Arrival order is not evidence order. Labels may be queued out of order,
   but e-process factors are applied in the predeclared prediction order.
   A missing/expired record must be terminalized before a later sequence can
   release, or the later evidence remains pending.

### 4. Canonical evidence record

Add an immutable `EvidenceRecord` (or extend the promotion audit record) with
one row for every prediction terminal state and the factor details for every
settled label. The minimum fields are:

```text
evidence_index, prediction_id, target_id, decision_step, settlement_step
family_id, challenger_id, slot, epoch, allocation_id, alpha, threshold
prediction/model/feature/label revision digests
status = settled | missing | expired | rejected | pending
eta, eta_policy_id, prior_observation_count
challenger_utility, incumbent_utility, raw_difference, bounded_score
factor, wealth_before, wealth_after, threshold_crossed, first_crossing
reason/metadata
```

For `missing`, `pending`, or `rejected`, score, factor, and wealth-change
fields are null and the e-process observation count does not increase. The
record remains visible in counts and exports. For `settled`, the ledger
recomputes the bounded score from the frozen receipt and the label; it then
calls the existing primitive and records the returned factor and wealth.

The record must contain both decision time and settlement time. This makes it
possible to audit that a later label was not used to change an earlier
prediction and that a threshold crossing was based only on eligible labels.

### 5. Ledger API

The concrete minimum API should be equivalent to:

```python
ledger = PromotionEvidenceLedger(
    config=evidence_config,
    gate=promotion_gate,
    score_spec=bounded_utility_spec,
    eta_policy=predictable_eta_policy,
)

receipt = ledger.freeze_prediction(
    prediction_id=..., target_id=..., decision_step=...,
    challenger_action=..., incumbent_action=...,
    model_fingerprint=..., feature_fingerprint=...,
)

ledger.ingest_label(PointInTimeLabel(...))
records = ledger.settle_ready(at_step=...)
ledger.mark_missing(prediction_id=..., at_step=..., reason=...)

ledger.pending_predictions()
ledger.evidence_records()
ledger.allocation_records()
ledger.state_dict()
PromotionEvidenceLedger.from_state_dict(...)
```

The names may vary, but the following behavior is mandatory:

* `freeze_prediction` resolves and stores eta before the label can be read.
  An eta policy receives only an immutable pre-score state/context. A
  settlement call has no `eta=` override. The policy identity and resolved
  numeric eta are part of the receipt.
* `ingest_label` records receipt even when the label is early, late, out of
  order, or missing. It never directly updates the e-process.
* `settle_ready` uses only labels whose source availability boundary has
  passed and consumes records in prediction order. Missingness is a policy
  event, not a silent filter.
* `mark_missing` requires the predeclared terminal policy. If missingness may
  depend on the unobserved value, the ledger must refuse an alpha-controlled
  promotion claim; merely dropping those rows is not valid selection.
* only this ledger may call the gate for certified evidence. Raw
  `PromotionGate.observe_score`, `observe_utilities`, or a direct state-machine
  reset remains a low-level diagnostic API unless supplied with a valid
  receipt/allocation token.
* `state_dict` includes receipts, pending labels, terminal missing records,
  ordered evidence records, eta-policy identity, and allocation records. An
  HMAC checkpoint must restore the exact next evidence index and pending set.

## Alpha and selection accounting requirements

The ledger must make the family being paid for explicit. At minimum:

1. Reserve an immutable allocation event before evidence for a challenger,
   eta policy, model variant, or epoch is observed. Recompute
   `alpha(slot, epoch)` from the manifest and reject a fabricated or reused
   allocation ID.
2. Sum all reserved allocations, including abandoned candidates and reset
   epochs, and require the sum to be at most the global alpha. Do not treat
   unused alpha from one candidate as borrowable by another.
3. If candidates or eta policies are chosen after seeing labels, either put
   every tried candidate/policy in the predeclared allocated family or use a
   fresh holdout/selection boundary. The winner alone cannot receive an
   allocation retroactively.
4. Do not reuse a settled observation in a new epoch. A reset starts a new
   e-process with a fresh allocated alpha and records the previous epoch,
   incumbent, selection boundary, and reason.
5. Record the selection step and selection-basis digest. A `false_promotion`
   from `compare_to_baseline` is not an alpha event and cannot be counted as
   an estimated false-discovery rate.

The geometric rule in the current gate is acceptable as the default
allocation, provided the ledger pays for all choices. It is not a substitute
for selection accounting.

## Required integration changes

These are API requirements, not source changes made in this review.

### `backtest.py`

Add an optional `evidence_ledger=`/`promotion_ledger=` parameter to
`WalkForwardBacktest.run` and the functional wrapper. At each decision row it
must freeze a receipt containing the prediction, incumbent action, training
positions, decision index, model/config fingerprint, and target ID. Later
label settlement must use `TimeIndexedLabels.available_at` and the ledger's
missing/revision policy. Extend `BacktestResult` with the receipt/evidence
ledger or a typed reference to it, plus settled, missing, pending, and rejected
counts. Without that argument, the current backtest remains a path/metrics
report only.

`expected_returns` and `optimal_position` remain offline comparator inputs;
they cannot be silently promoted to point-in-time evidence without an
availability record and a declared utility convention.

### `online.py`

Allow the online runner to consume typed label records with IDs, availability,
and missing status (or add a typed label-stream adapter). Keep `initial_state`
continuation, but include the pending prediction/label ledger and its next
evidence index, not only the RLS pending update queue. Return the evidence
ledger and its terminal pending/missing counts in `OnlineRunResult`.

The current forgetting factor is captured at prediction time; retain that
property and test it separately from promotion eta. A detector score or a
future label must not alter either the already-frozen forgetting factor or the
already-frozen promotion eta.

### `promotion.py` and checkpointing

Keep `EProcess` as the small primitive. Add receipt-oriented methods or a
separate ledger instead of making arbitrary raw calls look certified. Add
allocation IDs and prediction/label IDs to promotion audit output, and make
callable eta restoration require a policy/config digest, not only a Python
qualified name. `checkpoint_from_components` must be able to include and
restore the ledger state; tampering with any record or allocation must fail
before component state is mutated.

## Tests required before an end-to-end claim

The following test names are intentional acceptance criteria. They can be
implemented in a new `tests/test_sequential_ledger.py` and in the existing
focused files.

### Primitive tests

* `test_eprocess_conditional_null_optional_stopping_crossing_rate` — use a
  predeclared null, optional stopping rule, independent seeded paths, and a
  one-sided binomial confidence bound. The current
  `test_rademacher_null_has_no_obvious_optional_stopping_explosion` threshold
  of `< 0.15` is a smoke regression, not an alpha-level validation.
* `test_eta_rule_receives_only_prior_state` — assert the current label and
  current score are unavailable to the policy; reject a settlement eta
  override.
* `test_bounded_score_and_factor_are_recomputed_from_the_declared_spec` —
  changing an unclipped difference cannot change the bounded factor except
  through the declared clipping rule.
* `test_eprocess_reset_requires_new_allocated_alpha_for_repeated_testing` —
  distinguish a state reset from a new globally funded test.

### Ledger and label tests

* `test_frozen_prediction_receipt_is_immutable` — mutate future features,
  labels, or the model after prediction; the stored action, eta, hashes, and
  eventual evidence record remain unchanged.
* `test_label_cannot_settle_before_point_in_time_availability` — an early
  arrival is pending and cannot change wealth or promotion state.
* `test_missing_label_is_visible_and_does_not_update_wealth` — terminal
  missing/expired records appear in the ledger, do not increment e-process
  observations, and do not disappear from denominators.
* `test_out_of_order_labels_are_buffered_and_settled_in_prediction_order` —
  arrival order changes no factor sequence or final state.
* `test_duplicate_and_late_revision_cannot_rewrite_sealed_evidence` — reject
  duplicates and prove that a correction cannot alter a prior score, factor,
  crossing, or promotion.
* `test_nonpredictable_missingness_policy_blocks_certified_promotion` — make
  outcome-dependent missingness a hard failure unless a separate valid
  missing-data method is selected.

### Alpha and composition tests

* `test_all_challenger_eta_model_and_epoch_allocations_sum_to_global_alpha` —
  include abandoned candidates, adaptive registration, and resets; verify
  unique allocation IDs and the exact geometric formula.
* `test_selected_winner_cannot_borrow_unallocated_alpha` — select the apparent
  winner after a search and assert that the ledger either pays for every tried
  hypothesis or refuses promotion.
* `test_end_to_end_frozen_null_pipeline_respects_global_alpha` — generate
  frozen predictions independently of null labels, use a predeclared
  predictable eta, include optional stopping and missing labels under the
  declared predictable policy, and check the empirical crossing rate with a
  predeclared confidence bound.
* `test_direct_gate_calls_and_unfunded_epoch_resets_are_not_certified` — raw
  gate/state-machine calls may still work as diagnostics but cannot produce a
  certified ledger decision.

### Backtest, online, and checkpoint tests

* `test_backtest_future_mutation_cannot_change_prior_prediction_receipts` —
  extend or mutate future data and compare every earlier receipt byte-for-byte.
* `test_walk_forward_evidence_uses_strict_target_and_availability_boundaries`
  — retain the existing training-position assertion and add the prediction,
  label, and settlement IDs.
* `test_online_irregular_missing_and_terminal_labels_round_trip_exactly` —
  compare one-shot and chunked/restored runs, including pending and missing
  records and their statuses.
* `test_online_label_arrival_order_does_not_change_evidence_order` — separate
  learner update order, label ingestion order, and e-process settlement order.
* `test_checkpoint_round_trip_restores_ledger_and_allocation_state` — replay
  after an HMAC-authenticated checkpoint and compare evidence records, wealth,
  first crossing, pending labels, and next allocation ID.
* `test_checkpoint_tampering_with_label_or_allocation_fails_before_restore` —
  cover integrity and compatibility for the new ledger state.
* `test_public_api_exports_ledger_receipt_and_label_types` — add the new types
  to the explicit numeric/public API contract.

## Release criterion

The package may continue to describe the current behavior as a research/alpha
scaffold. It may claim an anytime-valid result only for a ledger run whose
manifest, immutable prediction receipts, point-in-time label statuses,
predictable eta receipts, ordered evidence records, missingness assumption,
selection family, and alpha allocations are all present and internally
consistent. Otherwise report the e-value and crossing as diagnostic output and
do not call it a valid promotion, false-discovery estimate, or raw-return
claim.
