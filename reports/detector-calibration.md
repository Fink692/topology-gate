# Split detector calibration record

Date: 2026-08-04  
Scope: finite synthetic surrogate only; no market-calibration or economic claim

This record exercises `calibrate_threshold` with the exact finite persistent
Laplacian CUSUM backend. Threshold candidates were fixed before either split;
the smallest candidate whose calibration-split Wilson upper bound met the
declared budget was selected, then evaluated once through a separately
declared observation-factory instance and split seed.

## Declared experiment

- Observation null: `StationaryBlockBootstrap` over a deterministic 512-by-2
  AR(1) source generated with `default_rng(20260804)`, innovation scale
  `0.25`, initial state equal to the first innovation, recurrence
  `x[t] = 0.85*x[t-1] + innovation[t]`, circular block length `16`, and
  source ID `ar1-surrogate:v1`. Calibration and evaluation use separate
  factory instances over this declared surrogate source; the independent
  seeds are `11` and `29`.
- Persistent backend: `max_vertices=4`, `max_simplices=100`, `q=0`,
  `n_eigenvalues=4`.
- CUSUM: cloud/minimum points `4/4`, positive spectrum width `2`, Betti
  dimensions `(0, 1)`, calibration window/minimum periods `8/8`, drift `0.5`,
  forgetting bounds `[0.8, 0.99]`.
- Candidate thresholds: `(2.0, 8.0, 32.0, 128.0, 1024.0)`.
- Calibration split: `64` trials, `64` steps, two features, seed `11`.
- Evaluation split: `64` trials, `64` steps, two features, seed `29`.
- Declared maximum finite-horizon false-alarm rate: `0.15`.

## Results

Every candidate had `0/64` calibration false alarms and a 95% Wilson upper
bound of `0.056624`. The predeclared rule therefore selected threshold `2.0`.
The untouched evaluation split produced `2/64` alarms, an empirical rate of
`0.03125`, and a 95% Wilson upper bound of `0.106973`. The result is approved
for the declared `0.15` surrogate budget.

The same evaluation result does **not** pass a stricter `0.10` budget. The
protocol refused to produce a certificate at that budget. This is retained as
a negative control against treating calibration-split success as evidence of
generalization.

## Identities

- Calibration observation-factory identity:
  `b90af2b644c99a7578dbf79004cc87557471137b802d965d12ad3c976e50fca9:threshold-split:calibration:v1`
- Evaluation observation-factory identity:
  `b90af2b644c99a7578dbf79004cc87557471137b802d965d12ad3c976e50fca9:threshold-split:evaluation:v1`
- Selected detector identity:
  `856c87c650a17122b64d223527a63fbe0643f57c94bbd16802a7917e486926d7`
- Split-calibration result identity:
  `a28b32f1251cc1d1f7f604a107efda228387ff719415e413297bf34cd5d03fc1`
- Selected evaluation null-configuration identity:
  `c19ab87e1f5bf3e645e84188ba67e9ed3e922a1c46876c327f05c244d4c8179e`
- Certificate identity:
  `62043f59890bac87cc9873cc53a606f016a84cfd440513eeb947b55e7ce7fe7e`

The certificate carries the split-calibration result identity and is valid
only for this detector identity, source identity, horizon, seed schedule, and
surrogate null. It does not authorize a market run or establish that the
forgetting policy is economically useful. A point-in-time market calibration
must repeat the protocol on an independent, dependence-appropriate source
with a pre-registered operating budget.
