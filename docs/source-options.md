# Point-in-time source options

This is a procurement decision record, not evidence that any source has been
received or passed the market gate.

## Recommended primary route

Request CRSP US Stock together with CRSP Historical Indexes through the
institutional access route. Morningstar's current CRSP research catalog lists
both products among its historical research data products:

- [CRSP research data products](https://indexes.morningstar.com/research-data-products)

This is the strongest fit for permanent security identity, historical prices,
returns, delistings, and dated index membership. The delivered extracts must
still include the source vintage/retrieval metadata required by this project;
the vendor product description alone is not an as-of proof. Execution costs,
capacity limits, and any investor-availability assumptions remain separate
study inputs.

## Viable alternative

Norgate's Platinum/Diamond US Stocks tiers advertise delisted securities and
historical index constituents, and its content tables describe those datasets
as suitable for backtesting:

- [Norgate data content tables](https://norgatedata.com/data-content-tables.php)
- [Norgate subscription tiers](https://norgatedata.com/subscribe/subscribe.php)

This may be practical for a pilot, but the export must preserve permanent
identifiers, raw source bytes, retrieval metadata, and the exact adapter
revision. Separate cost/capacity evidence is still required. The repository
must not infer point-in-time revision availability merely from a final
historical series.

## Zero-cost private diagnostic candidate

Quantiacs documents free strategy development and data access, and its
NASDAQ-100 data exposes daily OHLCV/dividend fields together with an
`is_liquid` field for historical index membership. Its documentation also
states that the historical universe includes companies that were NASDAQ-100
constituents since 2001 and retains delisted names:

- [Quantiacs data loading reference](https://quantiacs.com/documentation/en/reference/data_load_functions.html)
- [Quantiacs data documentation](https://quantiacs.com/documentation/en/user_guide/data.html)
- [Quantiacs FAQ](https://quantiacs.com/faq)

This is the best no-cost source for a private engineering diagnostic, but it
does not close the market gate. The public documentation does not provide the
complete six-role raw handoff required here: explicit delisting returns and
reasons, point-in-time source revisions/availability timestamps, separately
observed realized returns, execution-cost/capacity records, and a frozen
redistributable source cut. Quantiacs' terms also restrict making its market
data or analyses available to third parties:

- [Quantiacs terms of use](https://quantiacs.com/termsofuse)

Use it only in a private local run. Do not commit or publish its raw data,
derived tables, or a Quantiacs-specific performance synopsis. Record the
retrieval time and local environment, derive labels only after the decision
boundary, and keep the result classified as `private-final-history diagnostic
only`. The six-role gate must remain closed.

## Decision

No no-cost source inspected in this pass was both openly downloadable and
documented as satisfying all six required roles plus execution-cost/capacity
evidence. Quantiacs improves the private diagnostic path, but its documented
availability and terms do not make it a substitute for the requested vendor
handoff. Therefore the project keeps the market gate closed and does not
substitute a final-history download for market evidence.
