# SPY/GLD trend-filter paper model

The repository now contains a deliberately small paper model:

- At each close, compare SPY with its trailing 100-session average.
- Hold SPY for the next session when the signal is positive.
- Otherwise hold GLD.
- Use no leverage and charge turnover costs.

The 100-session lookback and GLD defensive asset were selected on data ending
2015-12-31. Results from 2016-2022 are tuning evidence; 2023 onward is the
reported public-history holdout. The runner is:

```powershell
python examples\trend_filter_paper_model.py `
  --cache-dir "$env:TEMP\topology-gate-public-market-diagnostic" `
  --output "$env:TEMP\topology-gate-trend-filter-paper-model.json"
```

The model is not connected to a broker. Do not treat the historical result as
a promise of profit or as financial advice. The source is final adjusted
history, so the result remains `public-final-history paper-model diagnostic
only` and does not open the market-evidence gate.
