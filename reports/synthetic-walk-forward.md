# Synthetic strict walk-forward control-layer run

Date: 2026-08-04  
Status: bounded methodological diagnostic; not market or economic evidence

The run used 256 observations, known shifts at 64, 128, and 192, four seeds
(`11, 17, 23, 29`), a one-step label delay, and the final regime `[192, 256)`
as the held-out post-shift diagnostic. The comparison now includes the full
declared memory-control matrix: static RLS, a 32-observation rolling window,
constant exponential forgetting, standard mean/covariance CUSUM forgetting,
and persistent-Laplacian CUSUM forgetting.

The persistent-Laplacian threshold was selected and certified on a separate
AR(1) block-bootstrap surrogate, with selected threshold `2.0`, calibration
identity
`77a24b6eefbbc550badd0664f99bedb65d42c37091e2ff4925c801d169301797`, and
certificate identity
`efc83def9795f69c8cdf716758a9bc9f177dbf28d2da5461d92ec1d93167b1d2`.
The standard mean/covariance CUSUM used selected threshold `16.0`, calibration
identity
`d5da248ee3387824ee1035a09d7a8b3e6612eac0779f2b754100ab9d25b5581d`, and
certificate identity
`c27ea04eddf49fa5ff1ab551e146ea62e7d75f35355b59740fbf34e70ca2415c`.

Mean squared loss against the synthetic regime-direction label was:

| system | mean full-stream MSE | mean final-regime MSE | mean detector delay | false alarms | accelerated updates |
|---|---:|---:|---:|---:|---:|
| Static RLS | 0.0404266 | 0.0304297 | 64.00 | 0 | 0 |
| Rolling RLS (window 32) | 0.0420181 | 0.0327197 | 64.00 | 0 | 0 |
| Exponential RLS (`lambda=0.97`) | 0.0408137 | 0.0310597 | 64.00 | 0 | 0 |
| Standard mean/covariance CPD-RLS | 0.0404326 | 0.0304350 | 54.42 | 0 | 241 |
| Certified PL-RLS | 0.0404269 | 0.0304119 | 64.00 | 0 | 245 |

Both calibrated detector paths had zero pre-shift false alarms in all four
paths. The standard mean/covariance detector had a lower average delay in this
fixture, while the PL detector's reported alarm-delay diagnostic remained
censored at 64 steps. Neither detector materially improved aggregate loss over
static RLS. The PL controller accelerated forgetting on 245 of 256 decisions;
the standard CPD controller did so on 241 of 256 decisions.

This result demonstrates the complete causal wiring—separate calibration data,
delayed labels, certificate-bound forgetting, explicit non-topological and
topological baselines, and a held-out regime—while retaining a negative
scientific signal. It does not validate financial prediction, dynamic regret,
Sharpe, turnover, capacity, or point-in-time market behavior. The
data-generating process is intentionally a control-layer surrogate.

Reproduce with:

```powershell
$env:PYTHONPATH = 'src'
py -3.12 examples\synthetic_walk_forward_suite.py
```
