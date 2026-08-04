# Synthetic handoff receipt

[`synthetic_market_handoff.py`](../examples/synthetic_market_handoff.py) builds
a deterministic six-role source package and sends it through the same
filesystem intake used for a vendor handoff. It audits calibration, tuning,
and validation from one full timeline while leaving holdout sealed.

The resulting receipt is
[`synthetic-market-handoff.json`](../reports/synthetic-market-handoff.json).
It is a protocol diagnostic only: the records are synthetic, so it does not
authorize a market claim or establish survivorship-free coverage, delisting
returns, execution realism, or trading performance.

Run it with:

```powershell
$env:PYTHONPATH = 'src;examples'
py -3.12 examples\synthetic_market_handoff.py
```
