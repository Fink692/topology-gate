# Research-engineering review: point-in-time cross-asset walk-forward research

**Review date:** 2026-08-04
**Scope:** `backtest.py`, `synthetic.py`, `online.py`, `observability.py`, `checkpoint.py`, all tests, and `docs/architecture.md`.

## Verdict

The current worker layer is a useful row-indexed, single-series experiment
scaffold, but it is not yet a defensible point-in-time cross-asset research
engine. The architecture contract correctly states that the production
architecture is not implemented and that the existing modules are alpha
compatibility behavior only (`docs/architecture.md:7-16, 279-291`). Results
should remain explicitly exploratory until the P0 items below are addressed.

The existing tests were read but could not be executed in this environment:
`python -m pytest -q` failed because `pytest` is not installed. This review did
not edit source or test files; pre-existing worktree modifications remain
untouched.

## Priority triage

| Priority | Area | Research risk | Required disposition |
|---|---|---|---|
| P0 | As-of feature/label contract | Historical revisions and feature availability can be absent or misinterpreted | Build a timestamped causal data layer before using real data |
| P0 | Cross-asset universe | The current shape cannot represent point-in-time membership, delistings, or asset-level availability | Add a panel/universe contract; do not flatten assets into anonymous columns |
| P0 | Challenger promotion | Baseline comparison is an early-window heuristic, not frozen paired sequential evaluation | Integrate the existing e-process behind a label-arrival lifecycle |
| P1 | Missing data and abstention | Missing values either abort a run or can silently become zero/hold-last behavior | Add typed masks/statuses and unresolved-label accounting |
| P1 | Execution costs/slippage | One scalar turnover rate is not a market-microstructure model | Add an explicit, timestamped, asset-level execution-cost protocol |
| P1 | Run manifest and ledger | Checkpoints are authenticated state blobs, not reproducible research manifests | Add immutable input/provenance identity and replay evidence |

## Findings and concrete changes

### P0 — Feature and label timestamps are not a production as-of model

The architecture requires separate `event_time_ns`, `available_time_ns`,
`decision_time_ns`, `label_end_time_ns`, `label_available_time_ns`, and a
deterministic ingest tie-breaker (`docs/architecture.md:293-314`). The worker
contract has only one ordered feature index (`synthetic.py:226-255`) and labels
with an optional `available_at` value (`synthetic.py:324-359`). There is no
feature-level source revision or availability timestamp, so a feature matrix
can contain a final revised value without any way for the engine to detect it.

`backtest.py` converts availability into row positions
(`backtest.py:646-682`); an integer that is not found in the index is explicitly
treated as a position (`backtest.py:664-670`). This is unsafe for integer
nanosecond timestamps and is not equivalent to `available_time_ns <=
decision_time_ns`. The audit trail records only target row positions
(`backtest.py:1353-1364`), not the source availability boundary. The online
runner has the same limitation: `label_available_at` is restricted to integer
steps (`online.py:316-342`).

The backtest also defaults omitted realized returns to zeros
(`backtest.py:581-586`), which can produce a clean-looking zero-PnL run instead
of a missing-input failure.

Change required:

- Introduce canonical immutable observations with `record_id`,
  `instrument_id`, event time, availability time, source revision, and ingest
  sequence. Feature rows must carry the maximum availability time and an
  immutable transform-fit boundary.
- Represent targets with an explicit interval and label availability timestamp;
  use timestamp comparisons in the causal loop, not positional inference.
- Implement canonical ordering by `(available_time_ns, precedence,
  ingest_sequence, record_id)`, plus purge/embargo for overlapping horizons.
- Make missing `realized_returns` a typed failure or explicit unresolved status;
  never silently synthesize zeros in a research run.

Tests to add:

- Mutating any record whose availability is after a decision must not change the
  prior feature, prediction, gate, or update fingerprint.
- A later vendor correction for an old event must be excluded from an earlier
  replay and included only after its revision availability time.
- Cover irregular timestamps, equal-time tie ordering, late labels, terminal
  labels, target intervals, and purge/embargo boundaries.
- Assert that an omitted return stream fails closed and that the audit contains
  decision cutoff, maximum source availability, label availability, and rejected
  late/future records.

### P0 — The current data shape cannot support point-in-time cross-asset research

`TimeIndexedFeatures` is an `(n_samples, n_features)` matrix with column names,
not `(time, instrument, field)` data (`synthetic.py:226-255`). The synthetic
dataset's returns, oracle positions, regimes, and expected returns are all
one-dimensional (`synthetic.py:417-465`). The backtest's position, cost, and
comparator interfaces are likewise one-dimensional (`backtest.py:794-812,
1573-1592`). There is no instrument identity, membership interval, listing or
delisting event, corporate-action revision, or point-in-time universe snapshot.

Flattening assets into columns would still leave survivorship and cross-sectional
availability unrepresented, and would make asset ordering part of an implicit
array convention rather than an audited contract.

Change required:

- Add an instrument-level panel contract and a universe snapshot with
  membership start/end and membership-availability timestamps. Selection must
  use the snapshot available at the decision cutoff.
- Carry deterministic asset IDs through features, labels, predictions,
  positions, costs, and challenger scores. Define behavior for new listings,
  delistings, suspensions, and missing asset rows.
- Make point-cloud and cross-sectional ordering explicit and include the
  universe snapshot digest in the run identity.

Tests to add:

- A future membership addition/removal must not alter an earlier universe.
- A delisted asset remains in historical evaluation only through its valid
  interval; a survivor-only universe must fail a leakage test.
- Permuting input asset order must leave canonical outputs unchanged.
- Verify that asset-specific missingness, labels, positions, and costs do not
  bleed into another asset or change the denominator silently.

### P0 — Baselines are not shadow challengers and can be selected with future data

The backtest's `baseline_hooks` produce a second path and then compare early
and later cumulative returns (`backtest.py:1470-1511, 1696-1769,
1797-1861`). The default `promoted` flag is simply an early-window return
comparison (`backtest.py:1831-1849`). A custom `promotion_rule` receives the
complete candidate and baseline results, including later returns, before it
returns its decision (`backtest.py:1843-1848`). An array baseline is accepted
without any provenance or causal-generation proof (`backtest.py:1709-1723`).

This is not the architecture's required lifecycle: freeze incumbent and
challenger predictions before the label, score the pair at label availability,
update a predictable e-process, apply operational checks, and activate a
promotion at the next decision boundary (`docs/architecture.md:74-91,
386-400`). The standalone promotion module has useful e-process and alpha
allocation tests (`tests/test_promotion.py:162-189`), but the walk-forward
engine does not connect them to frozen, delayed paired predictions.

Change required:

- Reclassify the current baseline facility as diagnostic-only, or replace it
  with a challenger registry containing challenger/model IDs, immutable start
  boundaries, paired prediction IDs, label-arrival sequence, and model state
  versions.
- Score incumbent and challenger from the same frozen feature snapshot before
  either model updates. Feed only the paired score into a per-challenger
  e-process with pre-allocated alpha, burn-in, minimum labels, invalid/unresolved
  blocking, and a single future activation boundary.
- Remove the unconstrained callback that can inspect the full future result;
  expose only a prefix-safe comparison context.

Tests to add:

- Mutating post-promotion-window returns or labels cannot change an earlier
  promotion decision.
- Incumbent and challenger predictions are frozen before a delayed label and
  remain unchanged when the label arrives.
- Two or more challengers consume the declared alpha allocation; resetting or
  changing a start boundary cannot regain alpha.
- A threshold crossing is recorded once and takes effect only on the next
  decision boundary; unresolved labels, invalid scores, and missing costs block
  promotion.

### P1 — Missing data has no explicit research status

Synthetic features and labels reject all non-finite values
(`synthetic.py:187-200, 324-359`), while the online runner rejects non-finite
features, outcomes, and returns (`online.py:15-26, 291-301`). This is a useful
numeric guard, but it is not a missing-data policy. There is no per-asset mask,
source-quality status, late/missing-label record, or unresolved denominator.

In the backtest, a non-finite prediction becomes `NaN` and then either silently
holds the last position or goes flat (`backtest.py:1401-1422`). In the online
path, the result has no typed abstention/degraded status. These behaviors do not
meet the contract's explicit-abstention and visible-degradation rules
(`docs/architecture.md:45-54, 489-502`).

Change required:

- Add explicit missing masks and quality/status codes to feature, label,
  prediction, return, and score records. Distinguish `ABSTAIN`, `INVALID`,
  `DEGRADED`, `UNRESOLVED_LABEL`, and normal observations.
- Make hold-last/flat fallback a named, configured policy with an audit reason;
  it must not count as a normal prediction or promotion-eligible observation.
- Report counts and denominators for missing, late, rejected, and abstained
  records rather than dropping them or converting them to neutral values.

Tests to add:

- One missing asset row causes only the configured asset-level abstention.
- A missing label remains unresolved and cannot update RLS or an e-process.
- Invalid model output produces a visible status and no silent promotion.
- Rejected updates leave learner, detector, promotion, and ledger state
  unchanged.

### P1 — Cost and slippage semantics are too weak for microstructure claims

Offline cost is a single additive rate applied to absolute position changes
(`backtest.py:794-812, 1573-1592`); online cost is only a scalar bps rate
(`online.py:166-170, 397-400`). There are no execution timestamps, bid/ask
quotes, spread, fees, price/volume/ADV, participation, impact, latency, or
asset-specific notional units. `calculate_metrics` accepts any finite supplied
cost array without rejecting negative costs (`backtest.py:1070-1078`). The
online default also aliases `realized_returns` to outcomes
(`online.py:295-298`), which makes target/return semantics implicit.

Change required:

- Define an execution/cost model protocol that consumes the decision-time order,
  execution boundary, price/quote/liquidity snapshot, asset, and turnover.
  Record fee, spread, slippage, impact, and rejection components separately.
- Make one-way versus round-trip and initial-position conventions explicit.
  Require non-negative realized costs and a declared behavior when cost inputs
  are missing. Keep target labels separate from tradable return streams.
- Include cost-model identity and all relevant parameters in the run digest.

Tests to add:

- Entry, exit, flip, no-trade, partial-fill, and missing-liquidity cases have
  exact expected costs; negative cost inputs are rejected.
- Cost cannot use a quote or volume that was unavailable at the execution time.
- Costs are calculated per asset and are monotone in spread/impact/turnover.
- Candidate and incumbent use the same execution assumptions while retaining
  separate position histories.

### P1 — Checkpoints and audit logs do not constitute a reproducible run manifest

`CheckpointEnvelope` provides good HMAC/canonical-JSON mechanics and component
state slots (`checkpoint.py:100-117, 151-203`), but its identities are caller
supplied and its metadata is unstructured. It does not require an input
snapshot/vintage, universe digest, cost/challenger configuration, code revision,
seed derivation, or canonical event-ordering rule. `OnlineStreamState` stores
row counters and pending features/targets, not prediction IDs, timestamps,
source revisions, or a ledger position (`online.py:90-138`). `BacktestResult`
has no manifest or causal audit object (`backtest.py:483-505`).

`AuditEvent` contains only `event_type`, `step`, payload, and an optional
timestamp (`observability.py:53-69`), and `AuditLog.append` silently discards
old events at its `max_events` bound (`observability.py:104-107`). That is
incompatible with an append-only scientific ledger unless truncation is
explicitly marked and the durable source remains complete. The configuration
fingerprint covers topology/RLS only (`config.py:228-268`), while callable
identity is only module and qualified name (`config.py:102-108`), so different
closure/code state can share an identity.

Change required:

- Define a canonical `RunSpec`/`RunManifest` containing run ID, immutable input
  snapshot and source checksums, universe snapshot, feature/target/cost/
  challenger configs, purge/embargo, normalized config digest, package/code
  revision, dependency lock, Python/backend/platform/precision/thread policy,
  root seed and child-seed derivation, and checkpoint lineage.
- Extend every ledger record with contract/version IDs, causal timestamps,
  source revisions, input/output fingerprints, state versions, status/reason,
  and prediction/label/score/comparison IDs. Use a durable append-only ledger
  with a chain or content hash; never silently drop scientific evidence.
- Put manifest digest, ledger position/fingerprint, pending records, and all
  model/detector/promotion/RNG state into the checkpoint. Restore must verify
  the complete identity before state mutation.
- Require registered/versioned callable backends or hash their source/config;
  module/qualified-name alone is not a reproducibility proof.

Tests to add:

- Canonical manifest serialization is byte-stable and changes when any input,
  universe, cost, dependency, code, or configuration identity changes.
- Fresh replay and checkpoint-restore replay produce identical predictions,
  state-transition fingerprints, ledger records, and promotion decisions.
- Equal-time events have deterministic results independent of container order.
- Ledger truncation, missing manifest fields, stale checkpoint lineage, and
  callable identity mismatch fail closed.

## What the current tests do and do not prove

Covered today:

- Strictly increasing synthetic indexes and finite numeric inputs.
- Row-position label delay and terminal pending labels
  (`tests/test_backtest.py:50-68`, `tests/test_online.py:117-186`).
- Basic scalar turnover/cost arithmetic and comparator diagnostics
  (`tests/test_backtest.py:71-106`, `tests/test_metrics.py:14-86`).
- Deterministic topology prefixes, RLS state, HMAC checkpoint integrity, and
  standalone e-process/alpha behavior.

Not covered:

- As-of feature vintages, source revisions, event/availability/decision-time
  inequalities, overlapping-horizon purge/embargo, or causal audit evidence.
- Cross-asset panel alignment, point-in-time universe membership, delistings,
  asset-level missingness, or survivor-bias adversaries.
- Execution-price/quote/liquidity costs, negative-cost rejection, or
  cost-model provenance.
- Frozen paired challenger predictions, delayed score arrival, operational
  promotion gates, next-boundary activation, and multiple-challenger selection
  in the backtest.
- Reproducible run manifests and fresh-versus-restore ledger equivalence.

## Recommended implementation order

1. Build and test the canonical as-of event/panel/universe contracts, including
   missingness, target intervals, purge/embargo, and immutable fingerprints.
2. Move the walk-forward engine to timestamped prediction/label records and
   require explicit realized-return and execution-cost inputs.
3. Integrate challenger scoring and the existing e-process through the same
   pending-label/next-boundary state machine.
4. Add the run manifest, complete ledger, checkpoint lineage, and replay
   equivalence gate.
5. Only then use the engine for cross-asset claims; retain the current
   row-indexed worker APIs as explicitly exploratory adapters.
