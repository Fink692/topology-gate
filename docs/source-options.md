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

## Decision

No public source inspected in this pass was both openly downloadable and
documented as satisfying all six required roles plus execution-cost/capacity
evidence. Therefore the project keeps the market gate closed and does not
substitute a final-history download for the requested vendor handoff.
