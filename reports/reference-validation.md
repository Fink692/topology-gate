# Reference validation record

Date: 2026-08-04
Interpreter: CPython 3.12.10
Scope: bounded research/control layer only; no live-data or execution claim

## Reproducible engineering checks

- Full suite: **190 passed**.
- Configured coverage run: **79.99%** total coverage.
- Ruff: passed on `src` and `tests`.
- Mypy: passed on all 20 source modules.
- Dependency-light root import: passed without NumPy site packages.
- Wheel build and isolated target-directory smoke: passed; SHA-256
  `433881c82b73eadaa66484f0de15fc9d8a6ae1f02f54e9a162406e20dfcad81f`;
  the smoke also exercised the public point-in-time panel contract from the
  wheel.
- Authenticated checkpoint, manifest digest, promotion/evidence state, and
  detached restore tests: passed.
- Causal replay prediction-before-label, future-prefix invariance, explicit
  missing/invalid status, chained-record tamper detection, and model-state
  restore tests: passed.
- Timestamped detector/RLS migration tests: passed for feature-ID extraction,
  strict point-in-time universe membership, prediction-time factor freezing,
  one-shot versus resumed replay equivalence, missing-label context cleanup,
  neutral forgetting without an approved detector certificate, canonical
  cross-asset panel ordering, panel/universe digest telemetry, and panel
  identity equivalence under reversed binding input.
- Finite-null certificate tests: passed for conservative Wilson-bound approval,
  detector-identity binding, explicit rejection of an unapproved budget, and
  rejection of count/rate/interval inconsistencies.
- E-process optional-stopping harness tests: passed for reproducible bounded
  score paths, predeclared constant eta, first-crossing termination, score
  bounds, malformed factory output, and identity recording.
- Stationary block-bootstrap tests: passed for seeded reproducibility, source
  identity binding, feature-dimension rejection, and result-level factory
  identity recording.
- Strict economic contract tests: passed for separate realized returns,
  component execution costs, turnover/flip accounting, explicit abstentions,
  identity mismatches, unavailable costs, and non-observed return rejection.
- Paired causal-promotion tests: passed for prediction-time freezing,
  settlement-only gate advancement, missing/unresolved cleanup, one-shot versus
  resumed state equivalence, constant-eta enforcement, and learner/gate
  rollback on failed updates; instrument-labelled promotion evidence carries the
  canonical panel digest.
- Exact finite persistence algebra, PSD/eigen residual, permutation/duplicate,
  digest, and resource-cap invariants: passed.
- Configured exact-backend tests: passed for complete evidence return, stable
  filtration/solver identity, cloud/spectrum-width compatibility, causal
  prefix equivalence, checkpoint replay, transactional stream rollback on
  backend failure, content-digest propagation, and malformed-digest rejection.

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

- A 1,000-path, 500-step Rademacher null through `calibrate_eprocess_null` with
  `alpha=0.05`, optional stopping, and `eta=0.5` produced 40 threshold
  crossings. The 95% Wilson upper bound was 0.05401. This is a finite
  simulation of the bounded primitive, not a proof for a market score stream
  or its conditional-mean null.
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
| G3 recursive transactional state | Partial/pass for the migrated path | `CausalReplay` drives numerical detector/RLS plus paired challenger/`PromotionGate` state with prediction-time factor/utility capture, model-state rollback, and detached restore; legacy workers and the generic evidence-ledger path are not all one transaction. |
| G4 causal replay/recovery | Partial/pass for the migrated path | `AsOfBook`, `PointInTimePanel`, `CausalReplay`, `CausalRLSModel`, paired promotion, delayed-label ledger, chunk state, HMAC restore, prefix invariance, future append acceptance, consumed-prefix revision rejection, canonical panel identity, and terminal pending cleanup are tested; legacy row adapters remain compatibility paths. |
| G5 economic validation | Contract pass; market evidence not evaluated | `economic.py` is fail-closed for separate returns/costs and explicit abstention accounting, but no point-in-time vendor data, capacity, delistings, or sealed final holdout are present. |

The package remains research/alpha. The evidence needed to upgrade the two
remaining scientific claims is explicit: a pre-registered dependence-aware
calibration split and a point-in-time cross-asset walk-forward run with a
sealed final test.
