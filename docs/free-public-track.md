# Zero-cost public-data track

You do not need to buy data to run the control-layer experiment. The checked-in
public track downloads adjusted daily ETF histories from Yahoo Finance and
runs the same recursive-model comparison, persistent-Laplacian calibration,
transaction-cost diagnostic, and e-process promotion gate.

Run it with:

```powershell
$env:PYTHONPATH = 'src'
py -3.12 examples\public_market_diagnostic.py `
  --cache-dir "$env:TEMP\topology-gate-public-market-diagnostic" `
  --output reports\public-market-diagnostic.json `
  --refresh
```

The latest no-cost run used six fixed ETFs from 2007-01-03 through 2026-08-04,
producing 4,927 common price rows and 4,905 feature rows. Persistent-Laplacian
calibration approved threshold `2.0`; the ordinary mean/covariance detector did
not receive an approved threshold. Static RLS ended with net Sharpe about
`0.0943`, while certified PL-RLS ended around `-1.6412`; the challenger
e-process ended near `3.62e-23` against threshold `80`, so it did not promote.

These are useful engineering and hypothesis-screening results, and they cost
nothing to reproduce. They are not a survivorship-free market claim: the
source is final adjusted ETF history, not a point-in-time security master, and
there is no sourced delisting, historical-membership, capacity, or execution-
cost evidence. The full receipt is
[`public-market-diagnostic.json`](../reports/public-market-diagnostic.json).
