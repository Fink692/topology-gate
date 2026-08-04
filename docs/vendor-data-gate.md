# Vendor data gate

The research control layer is ready to consume a normalized
`StudySourcePackage`. It is not yet a market-data result because this workspace
does not contain a licensed, point-in-time vendor source. The next handoff must
close each row below before a walk-forward result is called economic evidence.

| Requirement | Minimum acceptable input | Why it matters |
| --- | --- | --- |
| Stable security identity | Permanent instrument identifier plus name/ticker history | Tickers are not stable through corporate actions, relistings, or delistings. |
| Delisting treatment | Daily returns/prices, delisting flags, delisting returns, and missing-return reasons | Dropping failed securities creates survivorship and delisting bias. |
| Historical universe | Opening/closing membership intervals or an equivalent dated constituent source | Current membership cannot reconstruct the investable universe in the past. |
| Revision/vintage history | Raw source cuts with retrieval/release dates and source revisions | A final revised value must not be visible before its availability time. |
| Execution realism | Separate realized returns, fees, spread/slippage/impact, turnover, and capacity limits | Gross forecasts do not establish tradable net performance. |
| Holdout governance | A sealed, pre-registered final window and an explicit release event | The final test must not influence model selection or promotion. |

## Recommended source combination

For a U.S.-equity pilot, request the CRSP US Stock database through the
licensed institutional access path. CRSP documents permanent PERMNO identifiers,
daily prices/returns, security history, and delisting fields in its [US Stock
database guide](https://www.crsp.org/wp-content/uploads/guides/CRSP10_Year_US_Stock_Database_Guide.pdf).
For historical index membership, request the dated files from the [CRSPMI
Historical database](https://www.crsp.org/research/crspmi-historical-database/),
which documents daily opening/closing constituent data. These sources address
identity, delisting, and membership coverage; they do not by themselves prove
that every historical database revision was available to an investor on the
original date. Preserve each dated vendor cut and its retrieval metadata.

For revised macro variables, use a vintage source such as the Federal Reserve
Bank of St. Louis [ALFRED download contract](https://alfred.stlouisfed.org/help/downloaddata),
which defines a vintage date as the historical date used to retrieve what was
available then. Do not merge ALFRED vintages with final market histories without
recording the distinct availability rules.

## Required handoff

The vendor adapter should deliver:

1. One `AsOfBook`-compatible market/universe/label source with permanent IDs,
   event time, availability time, source revision, and ingest sequence.
2. One ordered `StudyTimeline` with expected instrument membership at every
   decision boundary.
3. One `EconomicEvidence` bundle with separately sourced returns and costs,
   including capacity limits when capacity is claimed.
4. One raw-file fingerprint for every source input using
   `StudySourceArtifact.from_bytes(...)`, including role and record count.
   A market audit requires the exact roles `delistings`, `execution-costs`,
   `labels`, `market-observations`, `realized-returns`, and
   `universe-membership`.
5. The vendor license, dataset version, retrieval time, release/cut date, and
   the exact adapter revision in `StudySourceProvenance`.

Before phase auditing, pass the raw payloads back to the restored package as
`audit_market(phase, {artifact_id: raw_bytes, ...})`. That call requires
the supplied ID set to equal the declared artifact set, verifies every byte
size and SHA-256 fingerprint, binds the provenance vintage to the run
manifest, and requires complete universe, observed economic, and capacity
evidence. The adapter must fail closed on missing
availability timestamps, duplicate revisions, unknown fields, unstable
identifiers, unexplained delistings, or unverifiable raw-file hashes. An
adjusted-price download from a current public history is not sufficient
evidence for this gate.

## Acceptance sequence

Run the source package through `audit_market("calibration", payloads)`,
`audit_market("tuning", payloads)`, and `audit_market("validation", payloads)`
while the holdout remains sealed. Freeze the challenger
family and e-process budget before calibration. Only after the validation audit
and all economic records pass should the registered release event open the
holdout. The final report must retain the package digest, every raw-artifact
fingerprint, the audit receipts, and the exact source cut identifiers.

Until that handoff exists, report results as engineering or synthetic evidence,
not as validated market performance.
