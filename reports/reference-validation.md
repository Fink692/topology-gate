# Reference validation record

Date: 2026-08-04
Interpreter: CPython 3.12.10
Scope: bounded research/control layer only; no live-data or execution claim

## Reproducible engineering checks

- Full suite: **153 passed**.
- Configured coverage run: **79.10%** total coverage.
- Ruff: passed on `src` and `tests`.
- Mypy: passed on all 16 source modules.
- Dependency-light root import: passed without NumPy site packages.
- Authenticated checkpoint, manifest digest, promotion/evidence state, and
  detached restore tests: passed.
- Causal replay prediction-before-label, future-prefix invariance, explicit
  missing/invalid status, chained-record tamper detection, and model-state
  restore tests: passed.
- Exact finite persistence algebra, PSD/eigen residual, permutation/duplicate,
  digest, and resource-cap invariants: passed.

## Exact reference backend timing

The pinned float64 finite Vietoris–Rips reference path was measured on the
workspace machine with regular polygon clouds and `q=1`:

| points | simplices | elapsed |
|---:|---:|---:|
| 4 | 14 | 0.000784 s |
| 8 | 92 | 0.001413 s |
| 12 | 298 | 0.014986 s |
| 16 | 696 | 0.043426 s |

These are reference-path timings, not a production latency budget. The rolling
adapter rejects an undersized exact spectrum instead of padding it with fake
values; a cloud/configuration pair that cannot produce the declared spectrum is
therefore fail-closed.

## Statistical control checks

- A 1,000-path, 500-step Rademacher null with `alpha=0.05`, optional stopping,
  and `eta=0.5` produced 43 threshold crossings. The 95% Wilson upper bound was
  0.05742. This is a finite simulation of the primitive, not a proof for a
  market score stream.
- The detector calibration harness was run on a declared AR(1) null and a
  volatility-shift alternative. Under the tested configuration the null
  alarmed on 32/32 paths at step 9. That result is intentionally retained as a
  failure signal: the detector is **not** treated as calibrated, and its
  forgetting map remains exploratory until threshold selection and independent
  dependence-aware calibration are completed.

## Acceptance disposition

| Gate | Disposition | Evidence |
|---|---|---|
| G0 identity/claim freeze | Partial | `RunSpec`, `RunManifest`, evidence config, as-of records, and authenticated checkpoint identities exist; no full vendor input manifest or shared event engine yet. |
| G1 exact persistent MVP | Pass for bounded reference scope | `persistent.py` plus hand-built and algebraic invariants; only the declared finite VR/F2/q/pair construction is covered. |
| G2 calibrated forgetting | Fail for a calibrated claim | Harness exists and caught a false-alarm failure; no independent market/dependence calibration artifact authorizes accelerated forgetting. |
| G3 recursive transactional state | Partial/pass for components | RLS and e-process state tests plus frozen eta/evidence/checkpoint replay; the dependency-light `CausalReplay` transition is hash-chained and model-state checked, but the numerical workers are not all routed through it. |
| G4 causal replay/recovery | Partial/pass for the new boundary | `AsOfBook`, `CausalReplay`, delayed-label ledger, chunk state, HMAC restore, and prefix tests exist; legacy online/offline row adapters still need migration to the shared transition. |
| G5 economic validation | Not evaluated | No point-in-time vendor data, costs, capacity, delistings, or final holdout are present. Strict configs now reject implicit target/zero return substitution. |

The package remains research/alpha. The evidence needed to upgrade the two
remaining scientific claims is explicit: a pre-registered dependence-aware
calibration split and a point-in-time cross-asset walk-forward run with a
sealed final test.
