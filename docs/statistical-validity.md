# Statistical validity and metric contract

Status: research/alpha contract, not a calibration certificate
Date: 2026-08-04

This document narrows the claims made by the current implementation. Passing
unit tests, a walk-forward replay, or a wheel smoke test does not establish a
false-alarm rate, an e-process guarantee for an end-to-end promotion pipeline,
or profitable trading performance.

## Comparator metrics

The supported names are:

- **Absolute comparator discrepancy:** the evaluated-row sum of
  `abs(comparator_utility - strategy_utility)`. It is a symmetric diagnostic;
  it is not conventional dynamic regret.
- **One-sided utility regret:** the evaluated-row sum of
  `max(comparator_utility - strategy_utility, 0)`. It counts only rows on
  which the comparator has higher declared utility.

For both metrics, the declared utility basis is common to both sides:

```text
basis_t = expected_returns_t, when supplied and point-in-time valid
          realized_returns_t, otherwise
strategy_utility_t   = position_t * basis_t - strategy_cost_t
comparator_utility_t = optimal_position_t * basis_t - comparator_cost_t
```

Comparator costs must be supplied explicitly to `calculate_metrics`, or
derived there from a declared comparator turnover-cost rate. The walk-forward
engine derives them from the same configured turnover-cost rate and the same
evaluated mask as the strategy. Ordinary `gross_return`, `net_return`,
drawdown, turnover, and Sharpe remain realized-path metrics.

The legacy `dynamic_regret` field and `dynamic_regret_series` are retained for
source compatibility only. They expose the former absolute gross-comparator /
realized-net gap and are marked deprecated in the Python API. They must not be
used or reported as conventional dynamic regret. The explicit fields are
`absolute_comparator_discrepancy` and `one_sided_utility_regret`.

The comparator is an offline reference. It is not evidence that the model had
access to `optimal_position` at decision time. The caller must prove that any
expected-return series used for comparison was available at the declared
decision boundary; otherwise the result is exploratory and potentially
look-ahead contaminated.

For economic metrics, the worker configs expose an explicit
`require_realized_returns=True` mode that rejects the legacy compatibility
behavior of substituting targets or zeros for a tradable return stream. A run
that leaves this mode off is a control-path diagnostic, not a net-return claim.

The online runner returns a typed terminal pending-label ledger and supports
chunked continuation through `initial_state` with absolute positional
availability. This proves only the package's fixed positional replay contract;
it does not identify vendor label IDs, late corrections, source revisions, or
real-world availability timestamps.

## Evaluation masking and validation

`evaluated=False` is a hard aggregation boundary. Unevaluated rows contribute
zero to gross/net return, transaction cost, turnover, equity, drawdown, and
comparator metrics. The arrays may still be retained for audit purposes, but
they cannot change reported path totals.

Comparator values must be finite, aligned to the path, and within the declared
action limit. Non-finite or infeasible comparator inputs are rejected rather
than silently converted into an apparently favorable benchmark. Results that
overflow during return or utility arithmetic are rejected as well.

## Detector and promotion claims

The default detector is a **causal kNN normalized-Laplacian spectral
approximation** over a rolling, delayed-embedding point cloud. It does not
construct a filtration, persistence pairs, homology groups, or a persistent
Laplacian. An exact persistent-Laplacian implementation may be supplied only
through an explicitly identified backend; the default output must not be
described as persistent topology.

The package now includes an optional bounded finite reference backend in
`topology_gate.persistent`. It constructs a Euclidean Vietoris–Rips complex,
computes `F2` persistence intervals, and evaluates a declared
nullspace-restricted real q-Laplacian for small clouds. Its finite-complex and
float64 solver evidence is exact only for the declared construction and
resource limits; it does not establish rolling-market calibration. The
`PersistentLaplacianBackend` adapter makes the configuration, spectrum width,
and vertex budget part of the detector identity. `TopologyConfig` rejects a
rolling cloud larger than that exact budget, and a runtime backend failure
rolls back the stream step instead of emitting a shift signal. Valid exact
artifacts also emit a content digest through `StreamingTopologyResult` and
`CausalStep`, so a prediction can be linked to the finite topology evidence
without treating the derived feature vector as the artifact itself.

`PersistentLaplacianCUSUM` is an explicit exploratory controller built on that
backend. It extracts configured Betti counts and positive spectrum values,
uses only earlier valid artifacts for robust marginal standardization, and
applies a bounded non-negative CUSUM update. Its strict stream state includes
the rolling cloud, spectral reference history, backend identity, and evidence
digests. This makes the proposed control experiment reproducible, but it does
not supply a null distribution, dependence correction, or level-α guarantee.

Its stateless batch facade is compatible with the finite null/shift calibration
harness; the stateful `observe` path remains the one used for causal replay.

The default reflected CUSUM-like score, topology alarm, score-to-forgetting
map, and challenger promotion statistics are exploratory unless an independent
calibration study establishes their assumptions and operating characteristics.
The numerical causal adapter now requires a matching approved finite-null
certificate before it can use a topology factor below the neutral maximum;
the detector's warm-up `calibrated` mask alone cannot authorize acceleration.
In particular, a threshold is not an average-run-length or level-α guarantee,
and an isolated e-process primitive does not certify arbitrary caller-supplied
promotion streams. Calibration must specify the null, dependence structure,
rolling re-estimation policy, missing-label policy, stopping rule, and model/
feature/eta selection budget.

`EvidenceLedger` provides the composed promotion boundary for one registered
challenger: it freezes paired predictions and eta before labels, buffers label
arrivals until their source-availability boundary, settles in prediction order,
exposes burn-in and missing/expired records, and never accepts a settlement-time
eta override. `PromotionEvidenceConfig` makes the run/family, score, null,
missingness, allocation, dependency, backend, and manifest identities explicit;
without a certified configuration the ledger reports a diagnostic claim only.
This materially strengthens the audit contract, but the conditional null and
the model/feature selection budget still require an independent pre-registered
study.

`RunSpec`/`RunManifest` and authenticated checkpoint envelopes provide the
reproducibility identity layer. They do not manufacture point-in-time market
data: the input-vintage and universe digests must still come from a source that
records immutable availability and revision semantics.

The legacy row-oriented `run_recursive_rls` path now follows the same
authorization boundary as the timestamped adapter: a ready detector factor is
applied only with an approved certificate matching the detector identity.
Without that certificate, the runner records neutral forgetting and exposes a
all-false `acceleration_authorized` mask rather than silently treating warm-up as
calibration.

`AsOfBook` supplies a dependency-light contract and adversarial fixture for
those semantics. It keeps event and availability times separate, selects only
visible revisions, orders equal-availability events deterministically, tracks
point-in-time universe membership, and rejects implicit missing labels. Its
versioned `to_json()`/`from_json()` source boundary binds all revision-bearing
records to a digest and rejects unknown fields or tampering; it still requires
the caller's vendor adapter to establish survivorship and economic completeness.
The existing row-indexed backtest remains an exploratory adapter until it
consumes this event contract end to end.

`CausalReplay` provides the corresponding transition contract for a small
timestamped study: it materializes an as-of snapshot, settles prior pending
targets before the next prediction, rejects labels already visible for the
target being predicted, and records a hash chain. This closes the causal
ordering boundary for its callbacks, but it does not make arbitrary callback
code causal and it is not yet wired into the legacy NumPy row adapters.

The `causal_numeric` adapter now runs the existing detector/RLS workers through
that transition using immutable record/field bindings and point-in-time
universe checks. Instrument-labelled plans pass through `PointInTimePanel`,
which fixes field order, sorts instruments canonically, and records a content
digest plus the as-of universe digest; the causal step carries those identities
and resumed paths are tested for equivalence. This establishes a deterministic
cross-asset data contract, not a cross-asset market study. The separate
`economic` contract requires timestamped realized returns and execution-cost
components, rejects target/zero substitution and unavailable costs, and reports
abstentions explicitly; it is an evaluation boundary, not market evidence.

`PointInTimePanel.from_snapshot` can also receive an explicit
`expected_instrument_ids` set. A missing or unexpected instrument then fails
closed, and the coverage assertion is included in the panel digest; omitting
that argument remains an explicit partial-panel choice.

The `causal_promotion` adapter extends the same state boundary to paired
challenger/incumbent learners and a registered `PromotionGate`. Certified use
requires registering the complete challenger family and calling
`seal_registration()` before the first observation; the gate records the seal,
rejects later registration, and restores the frozen family from checkpoints.
It freezes both predictions before labels arrive, advances bounded paired
utility only at observed settlement, and rolls back both learners and gate state
on a failed transition. Its instrument-labelled plans use the same canonical
panel selection and carry the panel identity into pending promotion evidence.
Missing or terminally unresolved labels never advance the e-process. The
economic boundary likewise represents missing/censored realized returns
explicitly with `value=None` and refuses to score them as zero. Its strict
capacity mode requires sourced per-decision turnover limits and rejects
breaches; it does not estimate liquidity. This is a
tested composition boundary, not a calibrated market promotion result; the
utility scale, eta policy, minimum-label burn-in, source manifest, and holdout
evidence still require a pre-registered study. When configured, burn-in labels
update the learners but remain outside the e-process until the declared count
is met; that count is checkpointed and identity-bound.

`calibrate_eprocess_null` provides a finite optional-stopping simulation for
that bounded score primitive. It uses a predeclared constant eta, stops each
path at the first e-value threshold crossing, and reports a Wilson interval for
the crossing frequency. A Rademacher score stream is a useful algebraic null
check; it does not verify the conditional-mean assumption for paired market
utility or pay for data-dependent model selection.

`calibrate_promotion_null` extends that check through the complete
multi-challenger `PromotionGate`: challenger slots are registered and sealed
before the stream is observed, geometric alpha allocations are recorded, and
each path stops at its first selected promotion. This makes the selection
boundary testable in a finite simulation. It remains empirical evidence for
the declared score factory and predeclared epoch schedule, not a
conditional-mean theorem or market-calibrated promotion certificate.

When a `StudyManifest` is supplied, both causal replay adapters validate the
phase and timeline index before prediction and carry the manifest digest in
model state. A sealed holdout therefore fails closed at the adapter boundary;
opening it is a separate, auditable manifest transition. This still cannot
make an untrusted vendor vintage or an incomplete membership history valid.

## Minimum evidence before a stronger claim

Any paper or deployment claiming calibrated change detection, anytime-valid
promotion, or dynamic-regret guarantees should attach:

1. a point-in-time data and expected-return definition plus a `StudyManifest`
   whose calibration/tuning/validation/holdout windows and embargo are frozen;
2. a feasible comparator and explicit cost convention;
3. independent null simulations or block-bootstrap calibration with uncertainty;
4. optional-stopping and selection-budget tests for promotion;
5. strict walk-forward replay with unresolved and delayed labels visible; and
6. the exact numerical backend, dependency environment, and configuration
   identity.
