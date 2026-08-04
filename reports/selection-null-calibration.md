# Selection-family null calibration

Date: 2026-08-04  
Status: finite synthetic diagnostic; not market evidence

The preregistered family contains 4 model slots × 4 feature slots × 3 eta
slots = 48 selection cells. The parent alpha is `0.05`, so each cell receives
`0.05 / 48 = 0.0010416666666666667` before the downstream promotion-gate
allocation.

The run used the stable bounded Rademacher factory
`rademacher-selection-null:v1`, 1,000 trials, 500 steps, constant `eta=0.5`,
seed `31`, and selection-budget identity
`84de04371da2096518decdd5c961872cbc3bd7a2251fc9b66def022540f5ac5c`.

Result:

| quantity | value |
|---|---:|
| first family crossings | 36 / 1,000 |
| crossing rate | 0.036 |
| 95% Wilson interval | [0.0261156, 0.0494357] |
| parent alpha | 0.05 |
| per-cell alpha | 0.00104167 |

The observed upper bound is below the declared parent level in this finite
null simulation. This validates the implemented finite alpha-spending and
optional-stopping simulation for the declared bounded factory. It does not
establish the conditional-mean null for market utility differences, and it is
not a certificate for model promotion on external data.

Configuration identity: `8c7b894b0e518646df3c0c7e721866b42546642df69faaa13c8a008a0a21ced1`.
Reproduce with:

```powershell
$env:PYTHONPATH = 'src'
py -3.12 examples\selection_family_calibration.py
```
