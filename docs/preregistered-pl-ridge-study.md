# Preregistered PL-RLS study

Status: protocol frozen; source bundle pending. This document defines the
first study before any market result is inspected. It is not a result and does
not authorize a market claim.

The machine-readable declaration is
[`examples/preregistered_pl_ridge_study.py`](../examples/preregistered_pl_ridge_study.py).
It creates a sealed `StudyManifest` and a `SelectionBudget`; the final holdout
can only be opened by an explicit release event.

The finite family-null execution record is maintained in
[`reports/selection-null-calibration.md`](../reports/selection-null-calibration.md).

## Research question

Does a persistent-Laplacian structural-change controller improve post-shift
adaptation of a recursive ridge learner relative to fixed-memory and ordinary
change-detection baselines, while the paired challenger is promoted only when
its bounded, cost-adjusted score crosses a preallocated e-process threshold?

The primary control-layer outcome is post-shift comparator-based loss over a
declared 20-decision recovery window. Any regret calculation must use an
explicit, feasible local comparator and the supported utility fields; the
legacy absolute-gap `dynamic_regret` field is not a scientific outcome.
Secondary outcomes are detection delay, false reset frequency, recovery time,
information coefficient, net Sharpe, drawdown, turnover, and cost-adjusted
challenger promotions. No result is called economic evidence unless the source package passes
[`docs/vendor-data-gate.md`](vendor-data-gate.md).

## Frozen source requirements

Before execution, the adapter must bind all of these roles to one immutable,
point-in-time `StudySourcePackage`:

- permanent instrument identifiers and membership history;
- dated source vintages and revision/availability timestamps;
- corporate-action and delisting treatment;
- daily features, five-decision forward labels, and realized returns kept as
  separate fields;
- transaction costs, turnover, borrow/slippage, and per-target capacity;
- a sealed final holdout and raw-artifact byte digests.

The current protocol deliberately uses the identity
`point-in-time-source:required`; it cannot be replaced with a final revised
table or a survivorship-biased public price download.

## Splits and selection

The timeline uses half-open indices with a five-decision embargo:

| Phase | Indices | Purpose |
|---|---:|---|
| Calibration | `[0, 252)` | detector null/threshold calibration |
| Tuning | `[257, 509)` | predeclared model and feature choices |
| Validation | `[514, 766)` | one untouched operating evaluation |
| Holdout | `[771, 1023)` | sealed until explicit release |

The selection family is fixed at four model choices, four feature sets, and
three eta choices: 48 Cartesian cells. The parent alpha is `0.05`; the
selected cell receives `0.05 / 48 = 0.0010416666666667`. The downstream gate
must use that allocated alpha, then applies its own challenger-slot and epoch
allocation. No unregistered model, feature, eta, threshold, or favorable-start
choice may be promoted as if it shared the same alpha.

Certified promotion uses zero budgets for invalid, abstained, unresolved, and
non-observed records; missingness is not silently discarded. The score is the
declared bounded absolute-error utility from frozen predictions and the raw
label. Eta is constant and fixed before each evidence stream.

## Detector protocol

Use the exact finite Vietoris--Rips/F2 persistent backend with the declared
cloud, Betti, eigenvalue, and resource limits from the source configuration.
Candidate CUSUM thresholds are `(2.0, 4.0, 8.0, 16.0)`. Selection occurs only
on the calibration split. The selected candidate is evaluated once on the
independent validation split; the holdout is not read.

The null must preserve serial and cross-sectional dependence using the frozen
stationary/block-bootstrap specification (initial block length 16, source
digest, restart rule, and seed schedule). Report finite-horizon false-alarm
probability, Wilson confidence bounds, censored run length, detection delay,
and recovery time. A failed or insufficient certificate leaves forgetting at
the neutral maximum.

## Promotion protocol

Run each recursive ridge challenger beside the incumbent with shadow
predictions frozen at decision time. Labels settle only at their declared
availability boundary. The gate is inspected continuously, but promotion is
effective only at the next decision boundary. The complete null harness must
include optional stopping, all 48 selection cells or their preallocated
equivalent, challenger slots, and repeated epochs.

The null is:

\[
H_0: \mathbb{E}[d_t\mid\mathcal F_{t-1}]\leq 0,
\]

where `d_t` is the bounded, transaction-cost-adjusted challenger-minus-
incumbent utility difference. A threshold crossing is evidence only for this
declared score and filtration; it is not a raw-return or Sharpe theorem.

## Release and stopping rules

1. Stop and retain diagnostic status if the source audit is incomplete.
2. Stop accelerated forgetting if the detector certificate is absent,
   mismatched, or exceeds its false-alarm budget.
3. Stop promotion after any certified missingness/quality budget breach.
4. Do not tune on validation or open the holdout after seeing intermediate
   results.
5. Release the holdout only with a new manifest, release ID, source digest,
   code revision, and independent review receipt.

The first report must include every attempted selection, abstention, unresolved
label, denominator, cost, capacity breach, threshold crossing, and failed
negative control. A clean engineering run without the source audit remains a
research artifact, not a market result.
