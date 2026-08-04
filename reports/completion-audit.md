# Completion audit

**Audit date:** 2026-08-04
**Scope:** topology-gated recursive quant-model research program
**Disposition:** research implementation complete; market-evidence gate pending

## Requirement evidence

| Requirement | Current evidence | Disposition |
|---|---|---|
| Persistent-Laplacian detector and topology-gated forgetting | `src/topology_gate/persistent.py`, `src/topology_gate/pl_cusum.py`, `tests/test_pl_cusum.py`, public diagnostics, and calibrated score-normalization receipt | Complete for bounded research scope |
| Anytime-valid challenger promotion | `src/topology_gate/promotion.py`, `src/topology_gate/causal_promotion.py`, promotion tests, synthetic null receipt, and public e-process diagnostics | Complete for declared bounded score assumptions |
| Causal transport replay | `src/topology_gate/transport.py`, `tests/test_transport.py`, documentation, and four-seed receipt | Complete; result was negative in the declared fixture |
| Mechanism-localized continual learning | `src/topology_gate/mechanisms.py`, `tests/test_mechanisms.py`, documentation, and synthetic receipt | Complete for synthetic localized-shift fixture |
| Endogenous Wasserstein robustness | `src/topology_gate/wasserstein.py`, `tests/test_wasserstein.py`, documentation, and synthetic receipt | Complete for bounded surrogate |
| Heavy-tail-aware expert allocation | `src/topology_gate/experts.py`, `tests/test_experts.py`, documentation, and Student-t receipt | Complete; result was mixed |
| Adaptive rough-path memory | `src/topology_gate/signatures.py`, `tests/test_signatures.py`, documentation, and synthetic receipt | Complete for causal signature-depth fixture |
| Martingale stress training | `src/topology_gate/bridge.py`, `tests/test_bridge.py`, documentation, and finite stress receipt | Complete for finite discrete projection, not continuous-time pricing |
| Integrated experiment matrix | `examples/integrated_synthetic_study.py` and `reports/integrated-synthetic-study.json` | Complete as synthetic evidence |
| Free public-data path | `examples/public_market_diagnostic.py`, sensitivity and score-normalization scripts, three public receipts, and the Quantiacs private-track/source audit | Complete as private/final-history diagnostics; not market evidence |
| Canonical six-role source boundary | `examples/normalize_vendor_handoff.py`, tests, synthetic handoff, and fail-closed audits | Complete as an intake protocol |
| Point-in-time market evidence | `reports/vendor-handoff-status.json` | Not complete: all six required artifacts are missing |

## Public result boundary

The calibration-only score-normalization experiment fit its score scale on the
calibration prefix and produced holdout net Sharpe `-0.1679`, versus `0.0943`
for static RLS. Its e-process ended at `5.16e-32` against threshold `80`; the
challenger did not promote. This is explicitly public-final-history diagnostic
evidence, not a market-performance claim.

## Verification evidence

- Full Python 3.12 suite: `325 passed`.
- Ruff: passed for `src`, `tests`, and `examples`.
- mypy: passed for 31 source files.
- `compileall`: passed for `src`, `examples`, and `tests`.
- All JSON receipts parsed successfully.
- Working tree is clean at the audit commit.

## Exact remaining handoff

The market claim remains closed until the source package supplies all of:

1. `market-observations.jsonl`
2. `universe-membership.jsonl`
3. `delistings.jsonl`
4. `labels.jsonl`
5. `realized-returns.jsonl`
6. `execution-costs.jsonl`

Each must carry the required point-in-time fields, provenance, raw-byte
fingerprint, and source vintage described in
[`docs/vendor-handoff-request.md`](../docs/vendor-handoff-request.md). No
amount of additional modeling can manufacture those observations without
changing the claim being made.
