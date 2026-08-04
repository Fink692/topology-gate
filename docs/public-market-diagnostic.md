# Public market diagnostic

The checked-in receipt at [`reports/public-market-diagnostic.json`](../reports/public-market-diagnostic.json)
records a reproducible diagnostic on the public Yahoo Finance chart endpoint.
It uses a fixed six-ETF proxy universe (`SPY`, `TLT`, `EFA`, `EEM`, `IWM`,
`GLD`) from 2007-01-03 through 2026-08-04, adjusted closes, one-step delayed
labels, causal normalization fit on the first 40% of rows, 5 bps turnover cost,
and a final 15% diagnostic holdout.

Run it with the numeric project environment:

```powershell
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe examples\public_market_diagnostic.py `
  --cache-dir "$env:TEMP\topology-gate-public-market-diagnostic" `
  --output reports\public-market-diagnostic.json
```

The run produced 4,927 common price rows and 4,905 feature rows. The persistent
Laplacian threshold screen passed its declared finite bootstrap budget at the
smallest candidate, `2.0`. The mean/covariance CPD screen failed to approve any
candidate; its `32.0` endpoint is retained only as an unapproved diagnostic.
The five matrix rows are therefore: static RLS, rolling-window RLS, exponential
RLS, fail-closed unapproved CPD diagnostic, and certified PL-RLS.

On the public final-history holdout, static RLS had MSE `0.6302`, mean net
return `0.0000472`, and net Sharpe `0.0943`. Certified PL-RLS had MSE `0.7226`,
mean net return `-0.0008367`, and net Sharpe `-1.6411`. The challenger e-process
ended at `3.63e-23` against its preallocated threshold `80`; it did not promote.
These numbers are negative diagnostic evidence for this public proxy, not an
economic-performance claim.

## Source boundary

The payload hashes and URLs are in the JSON receipt. The source is final
adjusted ETF history: it does not prove point-in-time revisions, historical
membership, delistings, capacity, or execution-cost completeness. Accordingly,
the receipt remains `public-final-history diagnostic only` and
`vendor_gate_status: not_evaluated`. A licensed point-in-time source package is
still required before calling the result market evidence; see
[`docs/vendor-data-gate.md`](vendor-data-gate.md).
