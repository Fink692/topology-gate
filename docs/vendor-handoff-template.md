# Vendor handoff template

Run `examples/vendor_handoff_template.py` to create
`reports/vendor-handoff-status.json`. It contains no observations. It is the
machine-readable request for the six raw roles required by the strict market
audit:

```powershell
$env:PYTHONPATH = 'src'
.venv\Scripts\python.exe examples\vendor_handoff_template.py
```

When the vendor supplies the files, the adapter must normalize them into the
canonical `AsOfBook`, `StudyTimeline`, and `EconomicEvidence` artifacts,
fingerprint the exact raw bytes, and build a `StudySourcePackage`. Then run the
`--all-pre-holdout` intake command documented in
[`study-runbook.md`](study-runbook.md). The template never opens the holdout
and never authorizes a market claim by itself.

