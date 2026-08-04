# Canonical JSONL handoff adapter

[`normalize_vendor_handoff.py`](../examples/normalize_vendor_handoff.py) is the
strict boundary between a vendor-specific mapper and the research package. It
accepts exactly these six files in one directory:

| Role | Filename |
|---|---|
| delistings | `delistings.jsonl` |
| execution costs | `execution-costs.jsonl` |
| labels | `labels.jsonl` |
| market observations | `market-observations.jsonl` |
| realized returns | `realized-returns.jsonl` |
| universe membership | `universe-membership.jsonl` |

Each line must use the exact canonical fields defined in the adapter. Unknown
fields are rejected rather than silently discarded. The typed `AsOfBook` and
`EconomicEvidence` constructors then enforce causal ordering, finite values,
revision uniqueness, and explicit missingness. The adapter fingerprints the
original bytes and writes a `StudySourcePackage`.

This adapter intentionally does not guess how a vendor-native export maps to
the canonical schema. A CRSP/Bloomberg/etc. mapper must be separately pinned,
reviewed, and tested before emitting these files.

Example:

```powershell
$env:PYTHONPATH = 'src;examples'
py -3.12 examples\normalize_vendor_handoff.py `
  --raw-dir handoff\canonical-jsonl `
  --run-manifest handoff\run-manifest.json `
  --study-manifest handoff\study-manifest.json `
  --timeline handoff\timeline.json `
  --provenance handoff\provenance-metadata.json `
  --economic-cutoff '"2026-08-04"' `
  --output handoff\study-source-package.json
```

The generated package must still pass
`examples/market_source_intake.py --all-pre-holdout`; successful parsing does
not authorize a market claim or open the holdout.
