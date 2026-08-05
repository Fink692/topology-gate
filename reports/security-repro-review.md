# Security and reproducibility review

Date: 2026-08-04
Reviewer: independent third-party review
Scope: the repository tree at the project root.

## Disposition

The package is a useful alpha scaffold and its ordinary happy-path tests are green, but it is not ready to make production checkpoint/replay claims or to expose the numerical and observability boundaries to untrusted callers. The blockers are incomplete state restoration, missing dependency/provenance locking, unbounded work and memory at several public inputs, non-finite/overflow leakage into results, and an audit sink that accepts both secrets and arbitrary filesystem destinations.

No implementation or test file was modified. The only intended write from this review is this report.

## Verification evidence

- Python 3.12.10 with NumPy 2.2.6: `python -B -m pytest -p no:cacheprovider -q` — **50 passed** in 5.42 s.
- Python 3.13.5 with NumPy 2.2.6: the same command — **50 passed** in 5.01 s.
- Ruff: `ruff check src tests --output-format concise` — **All checks passed**.
- mypy: `python -B -m mypy --no-incremental src` — **Success: no issues found in 10 source files**. This is not a complete static assurance: `pyproject.toml:58`, `pyproject.toml:59`, and `pyproject.toml:64` set `ignore_errors = true` for `topology_gate.rls` and `topology_gate.backtest`; `pyproject.toml:55` also disables `disallow_any_generics`.
- Existing verification evidence supplied with this review also reports `compileall` and wheel/install smoke tests passing. Those checks do not address the findings below.
- The active agent Python 3.11.15 environment has no `pip` and no NumPy. A separate installed Python 3.12/3.13 environment was required for the numerical tests. `pip check` in the Python 3.12 shared environment reports unrelated conflicts for `mcp`/`pydantic`, `sse-starlette`/`starlette`, and `typer`/`click`; this is evidence that the environment is not an isolated release environment.

## Blockers

### B0 — Repository provenance is not reviewable from the target path

Evidence:

- `git rev-parse --show-toplevel` returned `C:/Users/Kristjan Backman`, not the target directory.
- The historical review was run before this project had its own repository metadata.
- `git status --short` from the target reported the whole user home, including unrelated personal files, as untracked. `git ls-files -- pyproject.toml src tests docs` returned no tracked paths.

Impact: the reviewed source cannot be tied to a code revision, review diff, or trusted dependency/provenance record. Any release or reproducibility result based on this tree is therefore not auditable.

Fix: review from a self-contained Git worktree, record the exact commit and source-tree hash, and exclude the home-directory Git root from the evidence chain.

### B1 — Dependency and build inputs are not pinned or attestable

Evidence:

- `pyproject.toml:2` uses `setuptools>=68` for the build backend.
- `pyproject.toml:24` declares no runtime dependencies; optional extras at `pyproject.toml:29`, `pyproject.toml:30`, `pyproject.toml:31`, `pyproject.toml:32`, and `pyproject.toml:33` use lower bounds only (`numpy>=1.26`, `scipy>=1.11`, `pandas>=2.1`, and unpinned test/dev tools).
- No lockfile, constraints file, requirements file, dependency hash manifest, SBOM, or build provenance file was present in the target tree.
- `src/topology_gate/online.py:9`, `src/topology_gate/synthetic.py:15`, and `src/topology_gate/backtest.py:16` import NumPy eagerly, while `src/topology_gate/__init__.py:44` through `src/topology_gate/__init__.py:73` only defer the failure until those lazy exports are accessed.

The no-NumPy probe showed `import topology_gate`, `RLS`, `TopologyConfig`, and `RollingTopologyDetector` working, but `SyntheticDataset`, `BacktestEngine`, and `OnlineRunConfig` raised a raw `ModuleNotFoundError`. This is a documented optional-dependency boundary, but it is not a reproducible environment contract.

Impact: two installs satisfying the same lower bounds can run different NumPy/SciPy/BLAS code paths and produce different numerical results. A clean rebuild cannot be reproduced from the repository alone.

Fix: commit a lock/constraints manifest with exact versions and hashes, pin the build backend, record Python/OS/BLAS/thread settings, generate an SBOM, and run the wheel smoke test in a clean isolated environment. Give optional-import failures an explicit install hint.

### B2 — Checkpoint/replay state is incomplete and can restore under incompatible configuration

Evidence:

- RLS serialization records a callable schedule as `forgetting_factor_kind="callable"` with `forgetting_factor=None` at `src/topology_gate/rls.py:1062`, `src/topology_gate/rls.py:1065`, and `src/topology_gate/rls.py:1067`. `src/topology_gate/rls.py:1317` through `src/topology_gate/rls.py:1328` then requires the caller to provide the callable. Probe result: `RLS.from_state_dict(state)` raised `ValueError: a callable forgetting_factor must be supplied when restoring state`.
- The topology stream snapshot contains only `version` and `observations` at `src/topology_gate/topology.py:1414` through `src/topology_gate/topology.py:1420`; loading validates only the version and rows at `src/topology_gate/topology.py:1422` through `src/topology_gate/topology.py:1427`. It contains no configuration, backend, schema, or code identity.
- A snapshot produced with two spectral eigenvalues was accepted by a detector configured for three. The restored step produced raw-feature tuple lengths **13** and **14**, and the two `StreamingTopologyResult` values were unequal. No incompatibility error was raised.
- `run_recursive_rls` keeps pending delayed-label updates and `previous_position` in local variables at `src/topology_gate/online.py:135` and `src/topology_gate/online.py:136`; those variables are consumed at `src/topology_gate/online.py:138` through `src/topology_gate/online.py:175` and are not checkpointable.
- Promotion exposes read-only snapshots at `src/topology_gate/promotion.py:731` and `src/topology_gate/promotion.py:1036`, but probes found no `load_state_dict` or `from_state_dict` on `EProcess` or `PromotionGate`.
- The architecture requires pending predictions, topology/reference state, active e-process state, configuration/version identities, and RNG/backend state in a checkpoint at `docs/architecture.md:450`, `docs/architecture.md:452`, and `docs/architecture.md:454`.

Impact: a restart can silently use a different detector schema/backend, cannot resume delayed updates or promotion state, and cannot restore a callable factor policy. Passing the existing happy-path snapshot test is insufficient for the documented checkpoint contract.

Fix: define a versioned checkpoint envelope containing every state-affecting field, config/schema/backend/code/dependency digests, pending labels, position state, promotion/e-process state, and RNG state. Reject mismatches before mutation; either make factor/backend policies declarative and fingerprintable or explicitly mark such runs non-restorable. Add save/restore tests at every event boundary and corrupted/incompatible-state tests.

### B3 — Public inputs have no effective resource budget

Evidence:

- Topology integer validation enforces only lower bounds at `src/topology_gate/topology.py:64` through `src/topology_gate/topology.py:72`; `TopologyConfig` accepts caller-sized embedding, cloud, eigenvalue, and calibration parameters at `src/topology_gate/topology.py:139` through `src/topology_gate/topology.py:168` with no upper caps.
- `src/topology_gate/topology.py:651` through `src/topology_gate/topology.py:665` allocates a dense pairwise distance matrix. `src/topology_gate/topology.py:745` through `src/topology_gate/topology.py:788` builds another dense graph/Laplacian structure. Input rows are fully materialized by `src/topology_gate/topology.py:402` through `src/topology_gate/topology.py:452`.
- `RollingTopologyDetector.observe` appends every row at `src/topology_gate/topology.py:1386` through `src/topology_gate/topology.py:1390` and recomputes the complete prefix on every call via `detect`; the stream and its snapshot have no length limit.
- RLS accepts any positive `n_features` at `src/topology_gate/rls.py:105` through `src/topology_gate/rls.py:120` and immediately allocates an `n_features` by `n_features` covariance at `src/topology_gate/rls.py:522` through `src/topology_gate/rls.py:556`. Forgetting-factor iterables are fully materialized at `src/topology_gate/rls.py:436` through `src/topology_gate/rls.py:451`.
- Promotion history/audit lists grow on every update at `src/topology_gate/promotion.py:428`, `src/topology_gate/promotion.py:589`, and `src/topology_gate/promotion.py:615`. History-based eta resolution revalidates the whole history at `src/topology_gate/promotion.py:493` through `src/topology_gate/promotion.py:499`.
- A supplied spectral backend result is converted to an unbounded list at `src/topology_gate/topology.py:845` through `src/topology_gate/topology.py:862`.

Impact: a caller-controlled row count, feature dimension, schedule, backend output, or long stream can cause excessive allocation or superlinear CPU use. `observe` is particularly unsuitable for an untrusted or long-lived service without a hard event budget.

Fix: validate upper bounds before materialization; cap rows, dimensions, cloud/eigenvalue sizes, pending labels, history, audit payloads, backend output, and per-event CPU/time. Return a typed resource-limit outcome. Replace full-prefix recomputation with bounded incremental state or impose an explicit stream horizon.

### B4 — Non-finite inputs and floating-point overflow can reach public results

Evidence:

- `TimeIndexedFeatures` copies and shape-checks values but does not check finiteness at `src/topology_gate/synthetic.py:48` through `src/topology_gate/synthetic.py:66`; `TimeIndexedLabels` has the same gap at `src/topology_gate/synthetic.py:143` through `src/topology_gate/synthetic.py:161`.
- `generate_synthetic_regimes` checks noise only with `< 0` at `src/topology_gate/synthetic.py:339` through `src/topology_gate/synthetic.py:342`; `NaN` passes. Probe result: `feature_noise=NaN` was accepted and the returned feature matrix contained non-finite values.
- `WalkForwardConfig` checks numeric fields only for negative/zero values at `src/topology_gate/backtest.py:253` through `src/topology_gate/backtest.py:267`; `WalkForwardConfig(transaction_cost=NaN)` was accepted.
- `calculate_metrics` multiplies finite arrays without a post-operation finite check at `src/topology_gate/backtest.py:879` through `src/topology_gate/backtest.py:881`. With `positions=[1e308]` and `realized_returns=[1e308]`, the result was `gross_return=inf`, `net_return=inf`, and `max_drawdown=nan`.
- E-process wealth uses direct multiplication at `src/topology_gate/promotion.py:570` through `src/topology_gate/promotion.py:583`. With `initial_wealth=1e308`, `eta=1`, and score `1`, the update and snapshot contained `inf`.

Impact: downstream decisions, metrics, JSON/export, and replay hashes can become non-finite even though individual input arrays were finite. Direct e-value overflow is not a stable/log-domain representation.

Fix: apply `isfinite` and range checks to every configuration and data boundary, including container classes. Detect overflow after every multiply/reduction and return a typed numerical failure without committing state. Store e-process wealth in a checked log representation with explicit overflow/underflow policy. Add extreme-value tests for all public numerical entry points.

### B5 — Audit/telemetry accepts secrets and writes to arbitrary paths

Evidence:

- `AuditEvent.to_dict` copies the complete caller payload at `src/topology_gate/observability.py:21` through `src/topology_gate/observability.py:27`.
- `AuditLog.to_jsonl` serializes the complete payload at `src/topology_gate/observability.py:52` through `src/topology_gate/observability.py:58`; there is no recursive redaction, allowlist, size limit, atomic replacement, or symlink/path-root check.
- Promotion metadata is stored unchanged in audit records at `src/topology_gate/promotion.py:597` through `src/topology_gate/promotion.py:614`. Probes showed `{'api_key': 'SECRET', 'nested': {'password': 'SECRET2'}}` and `{'token': 'SECRET3'}` returned unchanged.
- The destination is caller-controlled at `src/topology_gate/observability.py:52`, parent directories are created at `src/topology_gate/observability.py:56`, and an existing file is overwritten by `src/topology_gate/observability.py:58`.
- The agent-contract requirement is explicit redaction at `docs/agent-contracts.md:302` through `docs/agent-contracts.md:318`.

Impact: if payloads or destinations are influenced by an untrusted caller, credentials can be persisted and an arbitrary file can be created or truncated. This is an explicit library sink, so the severity depends on whether its caller boundary is trusted; it must not be treated as safe by default.

Fix: define a typed, recursively redacted audit schema; reject or hash secret-bearing fields; cap serialized bytes; restrict destinations to an approved directory using canonical/symlink-safe checks; use atomic temp-file-plus-replace with explicit permissions; and document the sink as trusted-only until those controls exist.

### B6 — Optional numerical/backend choices are not stable replay identities

Evidence:

- `topology.py:33` through `topology.py:37` catches **all** import exceptions, not only missing NumPy. A probe that raised a simulated `RuntimeError` while importing NumPy silently set `_np = None` and selected the fallback.
- `_symmetric_eigenvalues` catches all NumPy eigensolver exceptions and silently falls back at `src/topology_gate/topology.py:731` through `src/topology_gate/topology.py:742`. The public method remains the same approximation label, with no backend/version/tolerance identity.
- An optional callable backend is accepted at `src/topology_gate/topology.py:98` through `src/topology_gate/topology.py:101` and executed at `src/topology_gate/topology.py:892` through `src/topology_gate/topology.py:905`. No code/version/digest is recorded.
- `ModelConfig.fingerprint` uses `asdict` and JSON at `src/topology_gate/config.py:27` through `src/topology_gate/config.py:35`. With a configured backend function, the probe raised `TypeError: Object of type function is not JSON serializable`.
- The architecture requires an explicit backend identity and tolerance policy at `docs/architecture.md:430` through `docs/architecture.md:446`.

Impact: a broken optional install can silently change the numerical algorithm, and a backend-enabled configuration cannot be fingerprinted. The two passing interpreter runs do not prove equivalence across NumPy/fallback/BLAS/backend variants.

Fix: catch only the intended missing-dependency exception; fail closed or emit an explicit degraded/backend identity. Require a declarative backend ID/version/digest, include it in canonical config, pin thread/BLAS policy, and test reference-versus-fallback equivalence within stated tolerances.

## Warnings and residual trust boundaries

1. Model, predictor, action-mapper, baseline, eta, and topology-backend callables are arbitrary Python code. They can perform side effects, consume ambient randomness, block indefinitely, or mutate shared objects. `_clone_model` falls back to the original object after copy failures at `src/topology_gate/backtest.py:66` through `src/topology_gate/backtest.py:73`, so a failed clone can also make a run mutate caller-owned model state. Treat these extension points as trusted code or isolate them in a process with time and resource limits.
2. Seeded synthetic generation is a positive control: `np.random.default_rng(int(seed))` is used at `src/topology_gate/synthetic.py:363` and `src/topology_gate/synthetic.py:463`, and the repeatability/causality tests pass. This is not a run-wide seed manifest: no component-seed derivation, callback RNG state, or backend RNG state is persisted.
3. RLS validates candidate state before committing it at `src/topology_gate/rls.py:1267` through `src/topology_gate/rls.py:1281`, rejects non-finite update inputs, and has passing deterministic round-trip tests. That good behavior does not cover callable schedules or the missing promotion/online checkpoint state.
4. Topology has useful finite-input and permutation-stability tests, and its fallback can run without NumPy. There is no test that compares fresh versus restored runs under changed configuration, corrupted full checkpoints, backend identity changes, overflow, or resource limits.
5. A targeted repository search found no hard-coded credential/private-key literals and no `pickle`, `eval`, `exec`, network, subprocess, or filesystem-write call outside `observability.py`. This is a narrow source scan, not a secret scanner or dependency vulnerability assessment.
6. `AuditLog` does bound the number of retained events with `max_events` at `src/topology_gate/observability.py:33` through `src/topology_gate/observability.py:46`, but payload size, JSON serialization cost, and destination safety remain unbounded.

## Prioritized release conditions

Before claiming reproducible production results:

1. Establish a real repository revision and a clean, locked, hash-pinned build environment.
2. Implement a versioned, integrity-protected checkpoint envelope with config/backend/dependency identity and complete online, topology, RLS, and promotion state; reject incompatible restores.
3. Add pre-allocation resource limits and typed resource failures to every public input path.
4. Make all data/config/result boundaries finite-safe and define overflow/underflow behavior.
5. Redesign audit export for redaction, bounded payloads, trusted path handling, and atomic writes.
6. Add adversarial acceptance tests for the above, including fresh-versus-restored replay at every event boundary and cross-backend determinism under the declared tolerance.
