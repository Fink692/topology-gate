# Split detector calibration record

Date: 2026-08-04  
Scope: finite synthetic surrogate only; no market-calibration or economic claim

This record exercises `calibrate_threshold` with the exact finite persistent
Laplacian CUSUM backend. Threshold candidates were fixed before either split;
the smallest candidate whose calibration-split Wilson upper bound met the
declared budget was selected, then evaluated once on a distinct-seed split.

## Declared experiment

- Observation null: `StationaryBlockBootstrap` over a deterministic 512-by-2
  AR(1) source, `phi=0.85`, innovation scale `0.25`, circular block length
  `16`, source ID `ar1-surrogate:v1`.
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

- Observation identity:
  `c8eab54762b0238cb0afba2ed4ad6f407cebcf2a7590b30c2ad6e65a074d819b`
- Selected detector identity:
  `856c87c650a17122b64d223527a63fbe0643f57c94bbd16802a7917e486926d7`
- Split-calibration result identity:
  `b48d9b738a283fa1e2b0e6054a0149194e57bb0ea5e73accb659ae89a65fac7c`
- Certificate identity:
  `fe1cad60fbe9daea9685892c990534cd0c52666bd3b33e40f14601ac2c5a2b56`

The certificate carries the split-calibration result identity and is valid
only for this detector identity, source identity, horizon, seed schedule, and
surrogate null. It does not authorize a market run or establish that the
forgetting policy is economically useful. A point-in-time market calibration
must repeat the protocol on an independent, dependence-appropriate source
with a pre-registered operating budget.
