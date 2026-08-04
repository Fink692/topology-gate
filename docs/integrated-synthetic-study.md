# Integrated synthetic control-layer study

`examples/integrated_synthetic_study.py` runs the complete declared memory
matrix—static RLS, rolling RLS, exponential RLS, standard mean/covariance CPD,
and certified PL-RLS—through the shared delayed-label online runner.

For each system and seed it records full-stream and final-regime metrics:

- post-shift one-sided utility regret against the declared synthetic oracle;
- detection delay and recovery;
- MSE, information coefficient, and hit rate;
- net return, net Sharpe, drawdown, turnover, and transaction cost;
- false alarms and authorized accelerated-forgetting updates.

It also runs a paired synthetic PL-versus-static promotion diagnostic and a
two-challenger finite promotion null. The receipt explicitly labels both as
synthetic control-layer evidence. It does not establish conventional dynamic
regret, a conditional-mean market theorem, or economic performance.

Reproduce it with:

```powershell
$env:PYTHONPATH = 'src;examples'
.venv\Scripts\python.exe examples\integrated_synthetic_study.py `
  --output reports\integrated-synthetic-study.json
```

