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

The default reflected CUSUM-like score, topology alarm, score-to-forgetting
map, and challenger promotion statistics are exploratory unless an independent
calibration study establishes their assumptions and operating characteristics.
In particular, a threshold is not an average-run-length or level-α guarantee,
and an isolated e-process primitive does not certify arbitrary caller-supplied
promotion streams. Calibration must specify the null, dependence structure,
rolling re-estimation policy, missing-label policy, stopping rule, and model/
feature/eta selection budget.

## Minimum evidence before a stronger claim

Any paper or deployment claiming calibrated change detection, anytime-valid
promotion, or dynamic-regret guarantees should attach:

1. a point-in-time data and expected-return definition;
2. a feasible comparator and explicit cost convention;
3. independent null simulations or block-bootstrap calibration with uncertainty;
4. optional-stopping and selection-budget tests for promotion;
5. strict walk-forward replay with unresolved and delayed labels visible; and
6. the exact numerical backend, dependency environment, and configuration
   identity.
