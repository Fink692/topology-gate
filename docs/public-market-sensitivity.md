# Public PL-RLS sensitivity diagnostic

This is the next zero-cost experiment after the basic public ETF run. It uses
the same six fixed adjusted-ETF histories, one-step delayed labels, causal
calibration split, 5 bps transaction-cost diagnostic, and e-process promotion
gate. Every cell calibrates its own detector identity before the learner is
allowed to use a non-neutral forgetting factor.

Run it with:

```powershell
$env:PYTHONPATH = 'src;examples'
py -3.12 examples\public_market_sensitivity.py `
  --cache-dir "$env:TEMP\topology-gate-public-market-diagnostic" `
  --output reports\public-market-sensitivity.json
```

## Result

The default filtration configuration produced one constant persistent state on
the public feature stream. Its PL-CUSUM score stayed at zero for all three
drift settings, the forgetting factor stayed at `0.99`, and all three cells
ended with the same holdout net Sharpe, `-1.6412`. This means the basic public
run did not test adaptive forgetting in practice; it tested an exponential
`0.99` learner beside the static baseline.

A fixed filtration interval in calibration-normalized coordinates,
`scale_s=2.0` and `scale_t=20.0`, produced nonconstant topology states. Its
calibration selected threshold `128.0`; the median CUSUM score was about
`2572`, and the factor hit the `0.8` floor for the ordinary sensitivities.
Holdout net Sharpe improved to `-0.2094`, but static RLS remained better at
`0.0943`, turnover increased to about `0.962`, and the e-process ended at
`1.98e-31` against threshold `80`. No challenger promoted.

Lowering the forgetting sensitivity to `0.0001` and `0.00001` avoided complete
floor saturation, with median factors about `0.947` and `0.985`, but holdout
net Sharpe fell to `-0.9728` and `-1.7570`. These cells do not establish an
optimal setting; they show that the raw unbounded CUSUM score is on a scale
that makes the proposed exponential memory map numerically and economically
sensitive.

The correct next research change is therefore a predeclared, calibration-only
score-to-memory normalization or bounded score map, followed by fresh
independent calibration. It must not be selected by looking at this holdout.

The machine-readable receipt is
[`public-market-sensitivity.json`](../reports/public-market-sensitivity.json).
This remains public-final-history diagnostic evidence, not point-in-time
market evidence.
