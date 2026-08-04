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

## What still requires external data

The repository cannot create the missing licensed source evidence. Before a
market claim, supply the vendor package requested in
[`vendor-handoff-request.md`](vendor-handoff-request.md), including permanent
instrument identity, historical membership, delistings, revision/as-of
metadata, realized returns, execution costs, and capacity evidence. Then run
the ordered calibration, tuning, validation, and sealed holdout workflow in
[`study-runbook.md`](study-runbook.md). The intake and package audit fail closed
when those artifacts are absent or incomplete.

