# Calibration-only PL score normalization

The sensitivity study showed that the raw persistent-CUSUM score was much
larger than the unit-scale forgetting map assumed. This experiment adds the
explicit `forgetting_score_scale` configuration parameter and fits it only on
the calibration prefix.

The fixed exploratory filtration uses `scale_s=2.0` and `scale_t=20.0`. The
score scale is the median positive PL-CUSUM score on the calibration prefix;
the holdout is not used to derive it. The detector configuration and resulting
certificate include this scale in their identities.

Run it with:

```powershell
$env:PYTHONPATH = 'src;examples'
py -3.12 examples\public_market_score_normalization.py `
  --cache-dir "$env:TEMP\topology-gate-public-market-diagnostic" `
  --output reports\public-market-score-normalization.json
```

The run derived score scale `936.886` and selected threshold `128.0`. On the
final public-history diagnostic holdout, normalized PL-RLS reached net Sharpe
`-0.1679` versus `-0.2094` for the unnormalized fixed-scale cell and `0.0943`
for static RLS. The normalized factor had median about `0.812`, turnover was
about `0.962`, and its e-process ended at `5.16e-32` against threshold `80`.
The challenger did not promote.

This is evidence that calibration-only score normalization improves the
control-layer behavior in this proxy, not evidence of economic alpha. The
fixed filtration remains exploratory, and the source remains final adjusted
history rather than point-in-time market evidence.

Receipt: [`public-market-score-normalization.json`](../reports/public-market-score-normalization.json).
