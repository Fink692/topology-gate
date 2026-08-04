# Reference validation record

Date: 2026-08-04
Interpreter: CPython 3.12.10
Scope: bounded research/control layer only; no live-data or execution claim

Supported-interpreter matrix: CPython 3.10.11 and 3.11.15 also passed the full
229-test suite, Ruff, mypy, and compileall in isolated environments. Coverage
is reported from CPython 3.12 only.

## Reproducible engineering checks

- Full suite: **229 passed**.
- Configured coverage run: **80.58%** total coverage.
- Ruff: passed on `src` and `tests`.
- Mypy: passed on all 21 source modules.
- Dependency-light root import: passed without NumPy site packages.
- Wheel build and isolated target-directory smoke: passed; SHA-256
  `4490501499bd31cb6ce11d1b0e1148bebda6f8288252905332bdea0dd3b3b83f`;
  the smoke exercised strict online-state/checkpoint restore, strict
  `RunManifest`/`StudyManifest` restore, canonical `AsOfBook` source restore,
  digest-bound `EconomicEvidence` selection/evaluation with cutoff provenance,
  and sealed challenger registration from the wheel.
- Authenticated checkpoint, manifest digest, promotion/evidence state, and
  detached restore tests: passed; checkpoint and online-state top-level schema
  drift is rejected before state mutation.
- Study-manifest tests: passed for canonical split identity, ordered
  calibration/tuning/validation/holdout windows, embargo enforcement, and
  explicit immutable holdout release state, strict JSON round-trip, and
  unknown-field rejection; causal numerical and paired
  promotion adapters reject sealed holdout reads and checkpoint identity
  changes.
- Canonical as-of source tests: passed for deterministic event-order identity,
  versioned JSON round-trip including typed datetimes, digest verification,
  unknown-field rejection, malformed source rejection, and explicit
  digest-free diagnostic restore.
- Causal replay prediction-before-label, future-prefix invariance, explicit
  missing/invalid status, chained-record tamper detection, and model-state
  restore tests: passed; composite causal numerical and paired-promotion
  checkpoints reject unknown top-level and pending-context fields.
- Timestamped detector/RLS migration tests: passed for feature-ID extraction,
  strict point-in-time universe membership, prediction-time factor freezing,
  one-shot versus resumed replay equivalence, missing-label context cleanup,
  neutral forgetting without an approved detector certificate, canonical
  cross-asset panel ordering, expected-universe coverage rejection,
  panel/universe digest telemetry, and panel identity equivalence under
  reversed binding input.
- Finite-null certificate tests: passed for conservative Wilson-bound approval,
  detector-identity binding, explicit rejection of an unapproved budget, and
  rejection of count/rate/interval inconsistencies.
- E-process optional-stopping harness tests: passed for reproducible bounded
  score paths, predeclared constant eta, first-crossing termination, score
  bounds, malformed factory output, and identity recording.
- Complete promotion-gate null harness tests: passed for pre-registered
  multi-challenger streams, geometric per-slot alpha allocation, fixed
  registration-order tie handling, sealed registration, checkpointed seal
  restore, post-seal registration rejection, selected-eta consistency,
  first-promotion stopping, repeated-epoch alpha spending, reproducibility,
  negative-null closure, and malformed score dimensions.
- Stationary block-bootstrap tests: passed for seeded reproducibility, source
  identity binding, feature-dimension rejection, and result-level factory
  identity recording.
- Strict economic contract tests: passed for separate realized returns,
  component execution costs, turnover/flip accounting, explicit abstentions,
  identity mismatches, unavailable costs, explicit missing/censored return
  records, zero-placeholder rejection, strict capacity-evidence/turnover
  breach handling, digest-bound revision bundles, and explicit evidence-cutoff
  evaluation with recorded cutoff provenance.
- Paired causal-promotion tests: passed for prediction-time freezing,
  settlement-only gate advancement, missing/unresolved cleanup, one-shot versus
  resumed state equivalence, constant-eta enforcement, checkpointed
  minimum-label burn-in across chunked restore, sealed challenger-family and
  selected-eta enforcement, internal state consistency, and learner/gate
  rollback on failed updates;
  instrument-labelled promotion evidence carries the canonical panel digest.
- Legacy online authorization tests: passed for neutral forgetting without a
  certificate, approved identity-matched acceleration, authorization telemetry,
  and detector-certificate mismatch rejection.
- Exact finite persistence algebra, PSD/eigen residual, permutation/duplicate,
  digest, and resource-cap invariants: passed.
- Exploratory persistent-spectrum CUSUM tests: passed for prior-only
  standardization, Betti/positive-spectrum extraction, persistent artifact
  provenance, checkpointed one-shot/resumed equivalence, strict state-schema
  rejection, transactional rollback on backend contract failure, integration
  with the finite null/shift calibration harness, and causal RLS replay
  composition with checkpointed continuation. This is an
  engineering/calibration-harness integration test, not market calibration
  evidence.
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
- A 1,000-path, 500-step, three-challenger Rademacher null through
  `calibrate_promotion_null` with global `alpha=0.05`, constant `eta=0.5`, and
  seed 31 produced 16 first promotions (rate 0.016; 95% Wilson interval
  [0.00987, 0.02583]). The predeclared challenger-slot alphas were
  [0.0125, 0.00625, 0.003125], with corresponding e-value thresholds
  [80, 160, 320]. This exercises the complete gate and selection boundary,
  but is still a finite simulation of a declared score factory, not a market
  conditional-mean or promotion certificate.
- The detector calibration harness was run on a declared AR(1) null and a
  volatility-shift alternative. Under the tested configuration the null
  alarmed on 32/32 paths at step 9. That result is intentionally retained as a
  failure signal: the detector is **not** treated as calibrated, and its
  forgetting map remains exploratory until threshold selection and independent
  dependence-aware calibration are completed.

## Acceptance disposition

| Gate | Disposition | Evidence |
|---|---|---|
| G0 identity/claim freeze | Partial/pass for the declared protocol | `RunSpec`, `RunManifest`, `StudySpec`, sealed/opened `StudyManifest`, evidence config, as-of records, and authenticated checkpoint identities exist; no full vendor input manifest or shared event engine yet. |
| G1 exact persistent MVP | Pass for bounded reference scope | `persistent.py` plus hand-built and algebraic invariants, and the exploratory `PersistentLaplacianCUSUM` controller; only the declared finite VR/F2/q/pair construction is covered. |
| G2 calibrated forgetting | Fail for a calibrated claim | The persistent-spectrum controller and calibration harness are exercised, and the harness caught a false-alarm failure; no independent market/dependence calibration artifact authorizes accelerated forgetting. |
| G3 recursive transactional state | Partial/pass for the migrated path | `CausalReplay` drives numerical detector/RLS plus paired challenger/`PromotionGate` state with prediction-time factor/utility capture, model-state rollback, and detached restore; legacy workers and the generic evidence-ledger path are not all one transaction. |
| G4 causal replay/recovery | Partial/pass for the migrated path | `AsOfBook`, `PointInTimePanel`, `StudyManifest` phase checks, `CausalReplay`, `CausalRLSModel`, paired promotion, delayed-label ledger, chunk state, HMAC restore, prefix invariance, future append acceptance, consumed-prefix revision rejection, canonical panel identity, and terminal pending cleanup are tested; legacy row adapters remain compatibility paths. |
| G5 economic validation | Contract pass; market evidence not evaluated | `economic.py` is fail-closed for separate returns/costs and explicit abstention accounting, and `StudyManifest` records a sealed holdout boundary, but no point-in-time vendor data, capacity, delistings, or opened final holdout evidence are present. |

The package remains research/alpha. The evidence needed to upgrade the two
remaining scientific claims is explicit: a pre-registered dependence-aware
calibration split and a point-in-time cross-asset walk-forward run with a
sealed final test.
