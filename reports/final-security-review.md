# Final security/reproducibility review

> Historical review snapshot: subsequent remediation is recorded in
> [`reference-validation.md`](reference-validation.md). Retain this report for
> the original findings; use the current record for release status.

## Verdict

**FAIL** for an adversarial or exact-replay release. The implementation has useful validation and atomic-write controls, but checkpoint authenticity and replay compatibility are optional/advisory rather than enforced.

## Findings

1. **BLOCKER — checkpoints are unauthenticated by default.** `checkpoint.py:85-91` selects plain SHA-256 when `hmac_key` is omitted; `checkpoint.py:152-183` and `checkpoint.py:265-268` allow create/verify without a key. A party able to replace a checkpoint can edit state and recompute the digest. The default path is exercised by `tests/test_checkpoint.py:57-66`.

2. **BLOCKER — compatibility identity is caller-supplied and weakly enforced.** Callable/backend identities contain only module and qualified name (`config.py:102-108`, `topology.py:1332-1369`, `promotion.py:1021-1052`), so code or closure changes can retain the same identity. Checkpoint compatibility checks are optional (`checkpoint.py:204-239`), while `restore_component_states` verifies integrity only and does not require expected package/config/backend/dependency identities (`checkpoint.py:319-358`). This permits valid-but-wrong run state to be restored unless every caller supplies and validates expectations.

3. **WARNING — public topology resource limits are bypassable.** The bounded `TopologyConfig` path is not the whole public surface: `rolling_point_cloud` and `rolling_point_clouds` validate only minimum parameter values (`topology.py:577-613`), and `_matrix_from_features`/`robust_whiten` impose no row/column cap (`topology.py:1086-1109`, `topology.py:1123-1205`). Untrusted callers can therefore request excessive work or memory outside detector-config limits.

4. **WARNING — secret redaction is key-name-only.** Observability redacts only when a mapping key matches `_SECRET_KEY` (`observability.py:26-50`); event fields are emitted directly (`observability.py:60-69`). Promotion metadata has the same limitation (`promotion.py:102-125`), and identifiers/reasons are retained in audit records (`promotion.py:424-445`). Secrets in generic string values, event types, or identifiers can be exported. Existing tests cover nested sensitive keys but not these paths (`tests/test_safety.py:23-44`, `tests/test_safety.py:58-67`).

5. **WARNING — the release environment is not bitwise reproducible.** NumPy is optional with a separate pure-Python eigensolver (`topology.py:37-43`, `topology.py:796-808`), yet the default backend identity remains the same (`topology.py:1332-1340`). The requirements file pins versions but explicitly disclaims cross-platform BLAS bit identity and contains no artifact hashes (`requirements-release-py312.txt:1-20`).

## Release recommendation

**Do not release with security or exact-replay claims.** Make HMAC mandatory, enforce expected run identities during restore, and release only from a hash-locked, fixed Python/NumPy/BLAS environment with process-level resource limits. A trusted internal deployment may proceed only as a documented conditional release after supplying those controls operationally.
