# Reference validation record

Date: 2026-08-04
Interpreter: CPython 3.12.10
Scope: bounded research/control layer only; no live-data or execution claim

Supported-interpreter matrix: CPython 3.10.11, 3.11.15, and 3.12.10 each
passed the current 279-test suite; the 3.10 and 3.11 runs used isolated `uv`
environment. The CPython 3.12 release gate also passed Ruff, mypy, and
compileall. Coverage is reported from CPython 3.12 only.

## Reproducible engineering checks

- Full suite: **279 passed**.
- Configured coverage run: **80.76%** total coverage.
- Coverage enforcement: configured floor **80%**; the current CPython 3.10,
  3.11, and 3.12 runs each clear it at **80.76%** total.
- Ruff: passed on `src` and `tests`.
- Mypy: passed on all 24 source modules.
- Dependency-light root import: passed without NumPy site packages.
- Wheel build and isolated target-directory smoke: passed; SHA-256
  `86DFD50C6A5677D6AA2B043213AC4912DC239EC3AD6E11011D3C2781FD5CB09A`;
  source commit `db60416` supplied `SOURCE_DATE_EPOCH=1785871618`; two
  consecutive builds under the same source date produced the same digest.
  The smoke verified a dependency-free root import under `python -S` and
  optional worker exports in the numeric environment. CI now repeats the
  deterministic-build and root-import checks in the CPython 3.12 release lane.
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
- Strict study-input bundle tests: passed for ordered timeline identity,
  dynamic expected-universe preflight, target-label visibility rejection,
  sealed-holdout enforcement, economic evidence completeness, causal RLS
  wrapper receipts/economic-decision conversion, and paired promotion wrapper
  receipts, and mismatched run/study-manifest identity rejection. This binds
  normalized source artifacts before replay but does not validate a vendor
  source itself.
- Canonical study-source package tests: passed for tagged timeline round-trip,
  nested manifest/as-of/economic artifact restoration, provenance and package
  digest binding, raw-artifact byte/size fingerprints, exact complete
  payload-ID verification, exact schema fields, tamper rejection, and strict
  market-source auditing of required roles, vintage binding, observed economic
  records, and capacity evidence. The provenance envelope and receipt record
  adapter policy and verified bytes but do not independently certify vendor
  point-in-time claims.
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
- Selection-budget tests: passed for exact model/feature/eta Cartesian-family
  allocation, selected-cell binding, identity/tamper rejection, and finite
  family-level optional-stopping calibration.
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
  rollback on failed updates, state-pure prediction enforcement, utility-scale
  binding, external gate-reset/evidence injection detection, stable learner
  identity binding, checkpointed missingness/quality budgets, explicit
  predictable-missingness declaration, raw-label declared score-spec settlement,
  fail-closed
  blocked-state continuation, and next-boundary activation receipts;
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
- The split threshold protocol was then exercised on the exact finite
  persistent-Laplacian CUSUM with a declared stationary block-bootstrap AR(1)
  surrogate. Five predeclared candidates had 0/64 calibration alarms; the
  smallest candidate, `2.0`, was evaluated on a distinct 64-trial split and
  produced 2/64 alarms with a 95% Wilson upper bound of `0.106973`. It passed
  only the declared `0.15` finite surrogate budget; the same evaluation was
  rejected at `0.10`. The serialized result role-binds the calibration and
  evaluation observation identities. The full identities and negative-control
  boundary are in [`reports/detector-calibration.md`](detector-calibration.md).
  This is still not market or dependence-validity evidence.
- The pre-registered 48-cell selection-family Rademacher null was run for
  1,000 trials × 500 steps at parent alpha `0.05`. It produced 36 first family
  crossings (rate `0.036`, Wilson 95% interval `[0.026116, 0.049436]`), below
  the parent level in this finite simulation. The selection budget identity is
  `84de04371da2096518decdd5c961872cbc3bd7a2251fc9b66def022540f5ac5c`; the
  full record is in [`reports/selection-null-calibration.md`](selection-null-calibration.md).
- The strict synthetic walk-forward suite used four seeds, three known shifts,
  one-step delayed labels, separate detector calibration data, and a sealed
  final regime. Static RLS mean MSE was `0.0404266`, exponential RLS at
  `lambda=0.97` was `0.0408137`, and certified PL-RLS was `0.0404269`; the
  certified path accelerated 245 updates but did not materially improve this
  fixture. The result is a control-layer negative diagnostic, not market
  evidence; see [`reports/synthetic-walk-forward.md`](synthetic-walk-forward.md).

## Acceptance disposition

| Gate | Disposition | Evidence |
|---|---|---|
| G0 identity/claim freeze | Partial/pass for the declared protocol | `RunSpec`, `RunManifest`, `StudySpec`, sealed/opened `StudyManifest`, strict `StudyInputBundle`/`StudyTimeline` preflight, digest-verified `StudySourcePackage`/provenance/raw-artifact fingerprints, strict `StudySourceAudit` market gate, `run_causal_rls_study`/`run_causal_promotion_study`, evidence config, as-of records, and authenticated checkpoint identities exist; no vendor-native adapter or source dataset is present. |
| G1 exact persistent MVP | Pass for bounded reference scope | `persistent.py` plus hand-built and algebraic invariants, and the exploratory `PersistentLaplacianCUSUM` controller; only the declared finite VR/F2/q/pair construction is covered. |
| G2 calibrated forgetting | Synthetic protocol pass; market claim fail | `calibrate_threshold` binds predeclared candidate selection to distinct role-bound observation factories and an independent evaluation split; `SelectionBudget` binds model/feature/eta family allocation; and the finite family null is recorded. The surrogate record includes a stricter-budget rejection; no independent market/dependence calibration artifact authorizes accelerated forgetting. |
| G3 recursive transactional state | Partial/pass for the migrated path | `CausalReplay` drives numerical detector/RLS plus paired challenger/`PromotionGate` state with prediction-time factor/utility capture, model-state rollback, and detached restore; legacy workers and the generic evidence-ledger path are not all one transaction. |
| G4 causal replay/recovery | Partial/pass for the migrated path | `AsOfBook`, `PointInTimePanel`, `StudyInputBundle`, `StudyTimeline`, digest-verified `StudySourcePackage`, `StudyManifest` phase checks, the RLS and paired-promotion study wrappers, `CausalReplay`, `CausalRLSModel`, paired promotion, delayed-label ledger, chunk state, HMAC restore, prefix invariance, future append acceptance, consumed-prefix revision rejection, canonical panel identity, and terminal pending cleanup are tested; legacy row adapters remain compatibility paths. |
| G5 economic validation | Contract pass; market evidence not evaluated | `economic.py` and `StudySourcePackage.audit_market` are fail-closed for separate returns/costs, explicit abstentions, required capacity evidence, and sealed holdout handling, but no point-in-time vendor data, capacity, delistings, or opened final holdout evidence are present. |

The package remains research/alpha. The local protocol, selection controls,
finite null calibration, and synthetic strict walk-forward are complete. The
remaining upgrade path is external: a point-in-time cross-asset source bundle,
an audited dependence-aware market calibration split, and a sealed final test.
