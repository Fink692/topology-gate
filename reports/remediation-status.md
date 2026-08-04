# Remediation status

Date: 2026-08-04

The historical review counts below are superseded by the current reproducible
record in [`reference-validation.md`](reference-validation.md), which includes
the later causal-checkpoint hardening, 222-test release gate, and Python
3.10/3.11 compatibility matrix.

## Release posture

The repository is release-ready as an offline, research/alpha control
component. It is not a live-trading system and it does not support claims of
persistent-Laplacian detection, calibrated CUSUM alarms, end-to-end anytime-
valid promotion, or validated economic performance.

## Review-driven fixes completed

- Canonicalized the root API and shared protocols; added `py.typed`, lazy
  optional-worker imports, pinned release tooling, CI, and a local Git
  revision boundary.
- Added bounded finite-input validation across topology, RLS, online,
  synthetic, metrics, promotion, checkpoint, and audit paths.
- Added atomic, size-bounded, versioned checkpoint envelopes with HMAC
  authentication required by default, explicit compatibility identities, and
  detached restore validation.
- Added terminal pending-label state and `initial_state` continuation for
  chunked causal replay; delayed-label factors remain fixed at prediction time.
- Added explicit graph-spectrum backend identity, including the NumPy versus
  pure-Python eigensolver, and rejected silent numerical fallback on solver
  failure.
- Narrowed the statistical and topology language in the README/runbook/docs,
  and separated comparator discrepancy from the deprecated legacy
  `dynamic_regret` field.

## Independent review record

- `final-integration-review.md`: FAIL for live-trading release; PASS only for
  the documented alpha boundary.
- `final-security-review.md`: FAIL for adversarial/exact-replay release; the
  report was used to harden HMAC and compatibility enforcement.
- `final-statistical-review.md`: FAIL for statistically validated claims;
  confirms the remaining calibration and inference limitations.

## Evidence after remediation

- 77 tests pass with coverage enforcement at 75%; final measured coverage is
  75.36%.
- Ruff, mypy, and compileall pass in the clean CPython 3.12 release venv.
- The wheel builds, installs into a dependency-free target, imports with
  `python -S`, performs an RLS update, and round-trips an HMAC checkpoint.
- The deterministic online walk-forward example completes with finite output.

The remaining research work is substantive methodology, not a hidden software
failure: implement and independently validate an exact persistent-Laplacian
backend, calibrate the change detector under declared nulls, and bind the
promotion pipeline to frozen, point-in-time paired evidence before making
stronger scientific or financial claims.
