# Synthetic strict walk-forward control-layer run

Date: 2026-08-04  
Status: bounded methodological diagnostic; not market or economic evidence

The run used 256 observations, known shifts at 64, 128, and 192, four seeds
(`11, 17, 23, 29`), a one-step label delay, and the final regime `[192, 256)`
as the held-out post-shift diagnostic. The persistent-Laplacian threshold was
selected and certified on a separate AR(1) block-bootstrap surrogate, with
selected threshold `2.0` and calibration identity
`c023bfca58faa28eecee57f7020bade870852d1e8f93ca676add07318fab6015`.

Mean squared loss against the synthetic regime-direction label was:

| system | mean full-stream MSE | mean final-regime MSE | accelerated updates |
|---|---:|---:|---:|
| Static RLS | 0.0404266 | 0.0304297 | 0 |
| Exponential RLS (`lambda=0.97`) | 0.0408137 | 0.0310597 | 0 |
| Certified PL-RLS | 0.0404269 | 0.0304119 | 245 |

The certified PL controller had zero pre-shift false alarms in all four paths.
The detector's reported alarm-delay diagnostic was censored at 64 steps in
this fixture; it should not be read as evidence of successful change-point
detection. The PL controller accelerated forgetting on 245 of 256 decisions,
but produced no material aggregate loss improvement over static RLS in this
small fixture.

This result is useful because it demonstrates the complete causal wiring—
separate calibration data, delayed labels, certificate-bound forgetting, and a
held-out regime—while also giving a negative scientific signal. It does not
validate financial prediction, dynamic regret, Sharpe, turnover, capacity, or
point-in-time market behavior. The data-generating process is intentionally a
control-layer surrogate.

Reproduce with:

```powershell
$env:PYTHONPATH = 'src'
py -3.12 examples\synthetic_walk_forward_suite.py
```
