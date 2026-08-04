# Zero-cost Quantiacs private track

This is an optional local diagnostic path for the control-layer experiment.
It does not authorize a market claim, and it must not be used to publish raw
Quantiacs data or a Quantiacs-specific performance analysis.

## What it can test

The official Quantiacs documentation exposes a Python loader for historical
NASDAQ-100 data with daily open, high, low, close, volume, and dividend fields.
The same data includes an `is_liquid` field that represents historical
membership/liquidity at each date. The documented universe also retains names
that were NASDAQ-100 constituents since 2001 even when they later delisted.

That is enough to test, privately:

- persistent-Laplacian state construction;
- calibrated score-to-forgetting normalization;
- standard-CPD versus PL-CUSUM reset behavior;
- recursive ridge/RLS recovery after detected changes; and
- the challenger e-process as an engineering diagnostic.

It is not enough to establish survivorship-free, point-in-time, tradable
performance. In particular, the documented interface does not by itself
provide the required delisting-return ledger, source revision/availability
history, separately observed realized returns, or execution-cost/capacity
evidence.

## Local setup

Quantiacs recommends a separate Conda environment. Its documentation also
gives a pip/GitHub route. Follow the current instructions in the official
documentation rather than adding `qnt` to this dependency-light repository:

- [Local development instructions](https://quantiacs.com/documentation/en/user_guide/local_development.html)
- [Data loading reference](https://quantiacs.com/documentation/en/reference/data_load_functions.html)

The core loader shape is:

```python
import qnt.data as qndata

data = qndata.stocks.load_ndx_data(
    min_date="2005-01-01",
    max_date="2026-08-04",
    forward_order=True,
)
close = data.sel(field="close")
volume = data.sel(field="vol")
membership = data.sel(field="is_liquid")
```

Keep this environment and its downloaded cache local. Do not copy the raw
arrays into this repository. If you create a local diagnostic receipt, store
only the experiment configuration, code revision, source retrieval timestamp,
and aggregate results that you are permitted to retain under the current
terms.

## Required local controls

1. Freeze the date range and loader configuration before inspecting holdout
   results.
2. Use one-step-delayed labels and a separate calibration prefix, as in the
   existing public diagnostic.
3. Apply the repository's five-basis-point cost convention only as an
   engineering assumption; it is not Quantiacs execution evidence.
4. Do not use `is_liquid` as a complete historical-universe proof. Treat it as
   a source-provided membership proxy and retain that limitation in the local
   receipt.
5. Keep the claim status exactly `private-final-history diagnostic only`.
6. Do not pass the result to `audit_market(...)`; that audit is reserved for a
   complete six-role raw source package.

The zero-cost path can answer whether the control layer is worth pursuing on
your machine. It cannot answer whether a production strategy has validated
market performance.
