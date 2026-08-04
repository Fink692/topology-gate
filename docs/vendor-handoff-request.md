# Vendor handoff request

This is the remaining external input required before the study can produce
market evidence. The package must be point-in-time and must preserve raw source
bytes. A final adjusted-price table or current-membership list is insufficient.

## Recommended source combination

For a U.S.-equity pilot, request CRSP US Stock through WRDS or the equivalent
licensed institutional route, plus [CRSPMI historical index membership](https://www.crsp.org/research/crspmi-historical-database/). CRSP
describes [PERMNO](https://www.crsp.org/research/permno/) as a permanent
security identifier and documents delisted history; preserve the exact dataset
release and retrieval metadata. Use a vintage source such as
[ALFRED](https://alfred.stlouisfed.org/help/downloaddata)
for revised macro variables. If CRSP is unavailable, a vendor may substitute a
delisted-inclusive source only if it can also provide dated historical
membership, revisions/availability, and the required economic evidence.

## Required raw artifacts

Deliver one immutable file for each role below. The filenames become the
`StudySourceArtifact.artifact_id` values and must not change after packaging.

| Role | Minimum fields |
| --- | --- |
| `market-observations` | `record_id`, permanent `instrument_id`, event time, available time, source revision, ingest sequence, feature fields, quality status |
| `universe-membership` | instrument ID, membership start/end, membership-available time, source revision, index/universe ID |
| `delistings` | instrument ID, delisting event time, delisting type/reason, delisting return or explicit unavailable status, available time |
| `labels` | target ID, event interval, label value/status, label available time, source revision |
| `realized-returns` | target ID, decision time, realization time, available time, return value/status, source revision |
| `execution-costs` | target ID, decision/execution/available times, cost-model ID, fee, spread, slippage, impact, other cost, capacity limit, source revision |

Also provide the source license ID, provider/dataset IDs, dataset release/cut
ID, retrieval timestamp, adapter revision, as-of rule, revision rule, universe
rule, delisting rule, time zone, and the exact raw-file SHA-256/byte-size
manifest. Retain the vendor's original files; normalized JSON is not a
substitute for raw artifacts.

## Acceptance conditions

The adapter must construct a `StudySourcePackage` whose provenance vintage
matches `RunSpec.input_vintage_id`, whose `AsOfBook` contains complete visible
membership at every decision boundary, and whose economic evidence contains
observed realized returns, non-negative execution costs, and capacity limits
for every required target. Missing, late, revised, or censored records must
remain explicit; they must not become zeroes.

After the vendor outputs are placed in `handoff\`, assemble the canonical
package first. The cutoff is explicit and must use the same time domain as the
timeline (a string cutoff should be quoted; a numeric cutoff may be passed as
JSON):

```powershell
$env:PYTHONPATH = 'src'
py -3.12 examples\build_study_source_package.py `
  --run-manifest handoff\run-manifest.json `
  --study-manifest handoff\study-manifest.json `
  --timeline handoff\timeline.json `
  --as-of-book handoff\as-of-book.json `
  --economic-evidence handoff\economic-evidence.json `
  --provenance handoff\provenance.json `
  --raw-dir handoff\raw `
  --economic-cutoff '"2026-08-04"' `
  --output handoff\study-source-package.json
```

Then audit the package through all pre-holdout phases:

```powershell
$env:PYTHONPATH = 'src'
py -3.12 examples\market_source_intake.py `
  --package handoff\study-source-package.json `
  --raw-dir handoff\raw `
  --all-pre-holdout `
  --receipt-dir handoff\source-audits
```

The command audits `calibration`, `tuning`, and `validation` in order and
writes one digest-bound receipt per phase. It does not open holdout. Holdout
requires a new manifest, an explicit release ID, and a separate final receipt.

## Current status

The repository contains the package schema, strict market audit, filesystem
intake, and this all-phase command. No licensed point-in-time source bundle is
present in the workspace, so the public ETF diagnostic remains explicitly
`public-final-history diagnostic only` and cannot be promoted to market
evidence.
