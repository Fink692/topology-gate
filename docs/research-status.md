# Research-program status

This workspace now contains bounded, executable prototypes for every
mathematical direction in the proposal. “Implemented” means the control layer
is represented in code, tested, and run on a declared synthetic diagnostic. It
does not mean the method has been validated on point-in-time market data.

| Direction | Current artifact | Evidence status |
|---|---|---|
| Persistent-Laplacian change detection and topology-gated forgetting | `persistent.py`, `pl_cusum.py`, causal RLS study path | Synthetic calibration only; public-final-history diagnostic is explicitly non-point-in-time |
| Anytime-valid challenger promotion | `promotion.py`, `causal_promotion.py`, e-process calibration | Finite null simulations and causal replay tests; no market promotion claim |
| Causal transport replay | `transport.py` | Four-seed synthetic diagnostic was negative; not adapted-Wasserstein |
| Mechanism-localized continual learning | `mechanisms.py` | Synthetic localized-shift diagnostic froze the unchanged module |
| Endogenous Wasserstein robustness | `wasserstein.py` | Bounded absolute-loss surrogate and endogenous-radius synthetic diagnostic |
| Heavy-tail-aware expert allocation | `experts.py` | Student-t full-information diagnostic; mixed result |
| Adaptive rough-path memory | `signatures.py` | Causal signature-depth selector and synthetic diagnostic |
| Martingale Schrödinger-bridge stress training | `bridge.py` | Finite discrete KL/martingale projection diagnostic; not continuous-time |

The complete experiment matrix and requested metric bundle are materialized by
[`integrated_synthetic_study.py`](../examples/integrated_synthetic_study.py)
and its receipt
[`integrated-synthetic-study.json`](../reports/integrated-synthetic-study.json).
It is still synthetic evidence and does not satisfy the external market-data
gate below.

The source-boundary workflow is exercised end-to-end by
[`synthetic_market_handoff.py`](../examples/synthetic_market_handoff.py) and
[`synthetic-market-handoff.json`](../reports/synthetic-market-handoff.json).
That receipt proves the six-role byte verification and phase selection path,
not the validity of a market vendor's data.

The canonical six-role JSONL normalizer is documented in
[`canonical-jsonl-adapter.md`](canonical-jsonl-adapter.md). It is ready for a
vendor-specific mapper to feed, but no vendor-native mapper or licensed source
files are present in this workspace.

## What still requires external data

The repository cannot create the missing licensed source evidence. Before a
market claim, supply the vendor package requested in
[`vendor-handoff-request.md`](vendor-handoff-request.md), including permanent
instrument identity, historical membership, delistings, revision/as-of
metadata, realized returns, execution costs, and capacity evidence. Then run
the ordered calibration, tuning, validation, and sealed holdout workflow in
[`study-runbook.md`](study-runbook.md). The intake and package audit fail closed
when those artifacts are absent or incomplete.

The exact machine-readable request can be regenerated with
[`vendor_handoff_template.py`](../examples/vendor_handoff_template.py); the
current workspace status is recorded in
[`vendor-handoff-status.json`](../reports/vendor-handoff-status.json).

The source procurement decision and alternatives are recorded in
[`source-options.md`](source-options.md). No source has yet been received or
authorized for a market claim.

For a zero-cost path, run the public ETF diagnostic described in
[`free-public-track.md`](free-public-track.md). It answers the control-layer
hypothesis but deliberately does not upgrade the market-data gate.

The follow-up public sensitivity receipt is documented in
[`public-market-sensitivity.md`](public-market-sensitivity.md). It found that
the default public filtration produced a constant persistent state, while a
fixed-scale exploratory filtration activated topology but saturated the
forgetting map and still failed the e-process promotion gate. The next
mathematical task is score-to-memory normalization on a calibration split,
with fresh holdout evaluation.

The calibration-only score-normalization experiment is recorded in
[`public-market-score-normalization.md`](public-market-score-normalization.md).
It improved the fixed-scale proxy to holdout net Sharpe `-0.1679`, but static
RLS remained better and the challenger was not promoted.

The requirement-by-requirement completion audit is in
[`reports/completion-audit.md`](../reports/completion-audit.md). It records
the exact evidence for each research direction and the six missing artifacts
that keep the point-in-time market claim closed.
