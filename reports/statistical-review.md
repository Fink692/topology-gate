# Independent Statistical Review

**Review date:** 2026-08-04
**Scope:** `topology.py`, `promotion.py`, `rls.py`, `online.py`, `backtest.py`, `synthetic.py`, supporting modules, tests, `README.md`, `docs/architecture.md`, and `docs/agent-contracts.md`.

## Executive disposition

**Not statistically release-ready for claims of persistent-topology detection, calibrated change detection, anytime-valid promotion, causal online replay, or dynamic regret.** The repository is a useful research scaffold, and several isolated numerical contracts are implemented coherently, but the missing assumptions are not cosmetic. A passing test suite, static checks, and a wheel smoke test do not establish inferential validity.

Severity used below:

- **BLOCKER:** the associated claim must not be made or used for release/promotion until remediated.
- **WARNING:** behavior is usable only with an explicit limitation, pre-registration, or narrower claim.

The architecture document itself says the requested production architecture is not implemented (`docs/architecture.md:7-9`) and the README disclaims predictive/statistical validity (`README.md:13-16`). This review agrees with those disclaimers; it does not treat them as evidence that a caller will avoid the unsafe interpretations.

## Evidence log

| Check | Observed result | What it establishes—and what it does not |
|---|---:|---|
| `C:\Users\Kristjan Backman\AppData\Local\Programs\Python\Python312\python.exe -m pytest -q` | **50 passed** | Current examples and unit/integration assertions pass. No null calibration, optional-stopping error-rate, selection, or economic-regret proof. |
| Ruff check | **Pass** (existing evidence) | Lint/style. No statistical meaning. |
| mypy | **Pass** (existing evidence) | Static typing. No statistical meaning. |
| `compileall` | **Pass** (existing evidence) | Syntax/bytecode compilation. No statistical meaning. |
| Wheel/install smoke test | **Pass** (existing evidence) | Packaging and importability. No causal or inferential guarantee. |
| Direct bare `pytest -q` in the active mixed PATH | Collection failed with `ModuleNotFoundError: src` in `test_rls.py` and `test_topology.py` | Interpreter/path reproducibility issue; the canonical Python 3.12 module invocation passed. Pin the interpreter and invocation in CI. |

There is no `.git` directory at the repository root, so a VCS diff could not be used as a cleanliness check. The only intended write in this review is this report.

## Blockers

### B1 — The implemented topology statistic is not a persistent Laplacian or a topology estimator

`topology.py:1-22` and `:865-920` are unusually candid: the default method is explicitly named `knn_normalized_laplacian_approximation`. The implementation is:

1. causal delay embedding (`:490-548`);
2. coordinate-wise median/MAD-or-IQR scaling with a floor and clipping (`:599-648`);
3. a symmetric union k-nearest-neighbor graph with Gaussian weights using the median positive pairwise distance (`:745-788`);
4. the normalized graph Laplacian and its smallest eigenvalues (`:791-838`, `:907-920`).

It does not construct a filtration, simplicial complex, homology groups, persistence pairs, or a persistent Laplacian. The optional backend is only a callable seam; no exact backend is present in the repository. Therefore the architecture’s persistent-Laplacian requirement (`docs/architecture.md:349-357`) is unmet. An output called `spectral_*` is defensible; an output called persistent topology, Betti change, or persistent-Laplacian evidence is not.

The default spectral summaries also have degenerate semantics. For a non-isolated normalized graph Laplacian, `trace / n` is effectively 1; the code reports precisely that mean (`:827-835`). A probe on a six-point cloud returned eigenvalues approximately `(0, 0.5671, 0.8068, 1.4329)` and `trace=1.0`. That trace is not an independent geometric signal in the normal default path.

There is also a duplicated normalization path: `point_cloud_features` normalizes at `:948`, then passes the normalized cloud to `spectral_summary` at `:976`, which normalizes again at `:886`. A tiny-scale probe produced the same eigenvalues but different reported distance scales (`0.2` direct versus `0.8993` through `point_cloud_features`). This is at least a semantic/metadata inconsistency and can become material with floors or clipping.

**Required remediation:** either narrow all public/documentation claims to this exact graph-spectrum heuristic, or implement and golden-test the specified persistent-Laplacian pipeline. Add an independently computed hand-graph reference, duplicate/tie tests, and a backend identity/version test before using “topology” as an inferential term.

### B2 — The CUSUM threshold has no calibrated false-alarm meaning

The exact recursion is documented and implemented as

```text
G_t = max(0, decay * G_(t-1) + innovation_t - drift)
```

(`topology.py:105-115`, `:1154-1202`). `innovation_t` is the RMS norm of the current robust-whitened feature row (`:1327-1329`), hence non-negative. This is a reflected discounted accumulation heuristic, not a likelihood-ratio CUSUM: there is no specified null density, alternative density, log-likelihood increment, or mapping from `drift`/`threshold` to average run length or family-wise false-alarm probability.

The detector also re-estimates location, scale, covariance, and a ridge-whitening matrix at every time point from a rolling, overlapping history (`:1303-1332`). The current feature is scored after being transformed by that past estimate, but the resulting innovation sequence is dependent, adaptive, clipped, and not demonstrably standardized under any null. A threshold crossing therefore cannot be called a statistically controlled change declaration.

The existing volatility-regime test only shows that one chosen synthetic path crosses one chosen threshold (`tests/test_topology.py:test_detector_reacts_to_a_large_volatility_regime_after_calibration`). It does not estimate null alarm probability, detection power, or delay distributions.

**Required remediation:** pre-register a null (including serial dependence and calibration-window behavior), estimate finite-horizon crossing probability/ARL by independent stationary block simulation or bootstrap, and report uncertainty. If the threshold is only heuristic, rename the output/status and prohibit it from being interpreted as a level-α detector or as a justified accelerated-forgetting trigger.

### B3 — The e-process theorem is conditional, but the composition does not enforce its conditions

The isolated `EProcess` algebra is correct under its stated null. The score is

```text
X_t = clip(u_challenger - u_incumbent, -B, B) / B in [-1, 1]
```

(`promotion.py:1-18`, `:124-169`). If, under the null, `E[X_t | F_(t-1)] <= 0`, and `eta_t` is measurable before the current score with `0 <= eta_t <= 1`, then `1 + eta_t X_t` is nonnegative with conditional expectation at most 1. The product and threshold `initial_wealth / alpha` (`:252-282`, `:543-628`) are therefore valid for Ville-style optional stopping **for that bounded score and that conditional null**.

The repository does not establish that the system being promoted supplies those objects:

- `PromotionGate` accepts arbitrary scores or utility pairs; it has no prediction ID, label ID, availability time, frozen-prediction check, paired-target check, or missing-label state (`promotion.py:1300-1405`).
- A callable eta is restricted by the process to prior score history (`:493-500` and `:207-249`), which is good, but an externally supplied numeric eta is accepted at update time. A caller can compute that number from the current outcome and the process cannot detect the violation. A probe using `process.update(1.0, eta=1.0)` was accepted; predictability is a caller obligation, not an enforced property.
- The null concerns the bounded paired score, not raw returns, Sharpe, cumulative P&L, or an unclipped utility claim. The module says this explicitly, but no surrounding pipeline binds promotion to that score definition.
- The architecture requires minimum labels, burn-in, no unresolved/contaminated scores, operational checks, and next-boundary activation (`docs/architecture.md:385-393`). `PromotionGate` promotes on the first allocated threshold crossing and implements none of those gates (`promotion.py:1308-1335`).
- `EProcess.reset` can retain the same alpha (`promotion.py:665-729`). Its docstring warns about this, but the public state machine permits repeated fresh testing unless callers use the gate correctly. A reset is a new test, not a continuation of the old e-process.

Thus the primitive is conditionally valid; the repository does not justify “anytime-valid challenger promotion” as an end-to-end claim. This is the most important distinction in the review.

**Required remediation:** make the comparison record carry frozen predictions, target/availability IDs, exact bounded-score version, and a pre-registered eta rule; reject post-label or current-score eta; represent missing/unresolved labels explicitly; enforce burn-in and operational gates; make resets consume centrally allocated alpha; and add an optional-stopping null simulation.

### B4 — Online delayed-label replay silently loses terminal pending updates

`online.py:100-105` says the factor is captured at prediction time, which is the right policy. The queue stores `(due_time, x, y, factor, source)` and applies due labels before the next prediction (`:135-175`). For a fixed integer delay this is causally sensible.

However, the function exits without flushing or returning `pending` (`:176-204`). A direct five-row run with delay 2 left only three learner updates and returned `update_steps=[True, True, True, False, False]`. The two labels are not yet available within the finite replay, so not applying them is defensible; silently discarding the pending records and returning no pending-state/checkpoint is not. This prevents exact online/replay state equivalence and makes a final learner state depend on arbitrary truncation.

The online API also accepts only a fixed integer delay (`OnlineRunConfig:27-47`). It cannot consume the synthetic/contract-level `TimeIndexedLabels.available_at` semantics, irregular availability, label IDs, duplicates, out-of-order arrivals, or late corrections. By contrast, the backtest has a reasonably clear strict rule: `target_position < decision_position` and `availability < decision_position` (`backtest.py:1078-1089`), and the delayed-label test covers that simple case.

**Required remediation:** return/persist a typed pending-label ledger at end-of-stream; accept actual availability timestamps/positions in online replay; test irregular, late, duplicate, out-of-order, and terminal labels; and prove fresh-versus-restored replay equality.

### B5 — `dynamic_regret` is an absolute comparator gap, not conventional dynamic regret

The implementation defines, for evaluated rows,

```text
oracle_returns_t = optimal_position_t * (expected_returns_t or realized_returns_t)
regret_t = oracle_returns_t - strategy_net_return_t
dynamic_regret = sum(abs(regret_t))
```

(`backtest.py:1319-1353`; the direct helper repeats it at `:879-910`). This has several statistical problems:

- A strategy that beats the supplied comparator on a row receives positive “regret.” The existing direct test intentionally demonstrates this: positions `[1,-1,1]`, returns `[.2,-.2,.2]`, comparator all `+1`, yields `dynamic_regret=0.4`; the middle strategy earns `+.2` while the comparator earns `-.2`.
- The comparator is not validated as feasible, bounded, or actually optimal. The synthetic generator deliberately switches `optimal_position` while keeping expected returns positive and realized-return sign fixed (`synthetic.py:378-384`), so its “oracle” is not an economic optimum for those returns.
- The comparator may use expected returns while the strategy uses realized net returns, and the oracle is not charged comparator turnover/cost. Noise and asymmetric costs are consequently folded into the reported gap.
- `calculate_metrics` does not mask `gross`, `net`, or equity before summing/drawdown when an external caller supplies `evaluated`; a probe with one unevaluated and one evaluated long row returned `n_evaluated=1` but `net_return=2.0`.

**Required remediation:** name this metric “absolute comparator discrepancy” or define a proper reward/utility and one-sided pseudo-regret. Use a feasible comparator, the same information boundary and cost model, and a predeclared expected-versus-realized convention. Add sign, cost, masking, and infeasible-oracle tests before reporting dynamic regret.

## Warnings

### W1 — Causal is not independent, stationary, or calibrated

The detector’s causal boundary is a real strength: `reference` contains only strictly earlier valid rows (`topology.py:1303-1313`), and the prefix perturbation test confirms future rows do not alter earlier outputs (`tests/test_topology.py:test_prefix_is_causal_and_future_changes_cannot_leak_backwards`). But rolling clouds overlap heavily, calibration windows contain past regime mixtures, robust scales are re-fit adaptively, and no time/as-of metadata exists in the detector API. The claim supported is “prefix-causal array computation,” not “valid out-of-sample calibration.”

### W2 — The forgetting map is bounded but heuristic and uncalibrated

The exact mapping is

```text
lambda(s) = lambda_min + (lambda_max - lambda_min) * exp(-sensitivity * s)
```

(`topology.py:247-254`). It is monotone and bounded, which tests indirectly exercise. With a representative `[0.80, 0.99]` range and sensitivity 1, a probe returned `lambda(0)=.99`, `lambda(2)=.8257`, `lambda(5)=.8013`, asymptoting to `.80`; the rough memory `1/(1-lambda)` falls from 100 to about 5 observations. There is no derivation connecting detector score, regime duration, prediction loss, or effective sample size to those values. “Adaptive” here means a deterministic score-to-lambda rule, not an estimated optimal policy.

The RLS update correctly stores the factor selected before a delayed label (`rls.py:866-1030`; `online.py:142-175`), but RLS callables can depend on current features/prediction/theta (`rls.py:703-810`). Predictability still depends on the caller not putting post-outcome information into those features. Also, `run_recursive_rls` does not cross-validate detector lambda bounds against learner bounds; a learner with a narrower interval can fail when the detector emits a valid-but-out-of-range factor.

### W3 — Alpha allocation is algebraically conservative, not a complete selection correction

`geometric_alpha_allocation` uses

```text
alpha_(i,e) = alpha_global * 2^(-i) * 2^(-(e+1)),  i >= 1, e >= 0
```

(`promotion.py:300-326`). The infinite sum over challenger slots and epochs is at most `alpha_global`; the first challenger in epoch 0 receives `alpha/4`, not the global alpha. Gate registration and resets add those allocations (`:1234-1261`, `:1407-1501`), and the finite-sum test is consistent with the formula.

This pays for registered slots/epochs only. It does not automatically pay for arbitrary feature/model/eta tuning, favorable start selection, or a data-dependent hypothesis family unless those choices are pre-registered and mapped to slots. Direct `EProcess`/`PromotionStateMachine` resets can also bypass central accounting. `alpha_spent` is allocated alpha, not evidence consumed or a proof that all adaptive searches were covered.

### W4 — Missing labels are not represented with the architecture’s required status semantics

Backtest training excludes unavailable labels correctly for the fixed-delay case, but evaluation labels are aligned after the path is generated (`backtest.py:1273-1286`) and non-finite labels are silently removed from IC/hit-rate denominators (`:767-785`). The architecture explicitly requires unresolved labels to remain visible rather than silently dropped (`docs/architecture.md:409-416`). `TimeIndexedLabels` accepts arbitrary `available_at` values without a full validation/status model (`synthetic.py:143-177`).

### W5 — The synthetic fixture cannot validate economic promotion or regret

`generate_synthetic_regimes` provides deterministic, sign-changing latent positions, but expected returns are constant positive and realized returns are independent of the latent position (`synthetic.py:365-408`). This is useful for detection-delay/control-layer tests and explicitly described as such, but it cannot establish predictive utility, challenger superiority, or economic regret. The legacy covariance-shift generator has a different target construction and should not be mixed into one claim.

### W6 — Baseline “false promotion” is descriptive, not e-process evidence

`compare_to_baseline` chooses on the first promotion window and calls a reversal on later rows a false promotion (`backtest.py:1444-1510`). That is a useful leakage/selection diagnostic, but it is not an alpha-controlled test and does not use the e-process. The name must not be read as a measured false-discovery rate.

## Test-support audit

| Area | What tests establish | Material missing evidence |
|---|---|---|
| Topology/CUSUM | Determinism, prefix causality, finite outputs, fallback path, one hand-checked recursion, one volatility fixture | Independent graph/persistent reference, null ARL/FPR, power/delay confidence intervals, calibration coverage, duplicate/tie/solver-failure semantics |
| RLS | Independent weighted batch agreement for scalar updates, multi-output shape/state round-trip, covariance symmetry/PSD checks | Delayed pending-state oracle, label-invariance for feature-dependent policies, rejected-update state fingerprint, cross-component lambda contract |
| E-process/promotion | Bounded clipping, nonnegative factors, prior-history eta rule, fixed positive crossing, finite geometric alpha sum, one reset | Conditional-null simulation with optional stopping, data-dependent challenger/model/eta selection, missing/late labels, no-reset alpha accounting, operational promotion gates |
| Backtest | Simple fixed-delay exclusion, deterministic synthetic data, path costs, a descriptive baseline reversal | Irregular availability, overlap purge/embargo, future revisions, unresolved-label denominator, proper regret definition, external `evaluated` masking |
| Online | Determinism, fixed-delay queue behavior, detector snapshot replay | Terminal pending labels, actual availability timestamps, duplicate/out-of-order labels, fresh/restored learner state equivalence |

The 50 passing tests therefore support implementation examples and several causal prefix invariants. They do **not** support the statistical claims required by the architecture’s own acceptance gates (`docs/architecture.md:540-553`, `docs/agent-contracts.md:208-230`).

## Exact remediation commands and tests

Run these after the corresponding changes; all are bounded, reproducible commands.

```powershell
# Existing engineering gates
ruff check .
mypy
& 'C:\Users\Kristjan Backman\AppData\Local\Programs\Python\Python312\python.exe' -m compileall -q src tests
& 'C:\Users\Kristjan Backman\AppData\Local\Programs\Python\Python312\python.exe' -m pytest -q
```

Add and run the following focused tests (names are intentional acceptance criteria):

```powershell
& 'C:\Users\Kristjan Backman\AppData\Local\Programs\Python\Python312\python.exe' -m pytest -q `
  tests/test_topology.py tests/test_topology_statistics.py `
  tests/test_promotion.py tests/test_eprocess_null.py `
  tests/test_online.py tests/test_backtest.py tests/test_causality.py
```

The new tests must, at minimum:

1. compare the default graph spectrum and every summary to an independent hand/scipy reference, and fail if the method is presented as persistent topology without an exact backend;
2. estimate `P_0(max_{t<=T} G_t >= threshold)` and detection-delay distributions under pre-registered stationary and block-dependent nulls, with confidence intervals;
3. reject or formally bind eta to pre-score information, simulate conditional-mean-null e-processes under optional stopping, and test the complete allocated alpha family under adaptive registration and epochs;
4. replay irregular and terminal label availability, retain pending labels, reject duplicates/out-of-order updates, and compare fresh/restored state fingerprints;
5. test regret sign, comparator feasibility/costs, expected-versus-realized convention, and `evaluated` masking; and
6. test detector/RLS lambda compatibility and prove changing any future feature/label/revision cannot change earlier prediction, factor, score, or promotion records.

For a wheel smoke test after changes, build into a temporary directory outside the repository, install into a fresh temporary environment, and run an import plus one minimal detector/RLS/e-process/backtest call. Do not use a build command that leaves `build/`, `dist/`, or regenerated package metadata in this review workspace.

## Final assessment

The repository passes its current engineering checks, and the isolated e-process product has a sound conditional-mean argument. The decisive gaps are claim-to-implementation mismatch: the topology statistic is explicitly an approximation, the CUSUM threshold is uncalibrated, the e-process assumptions are not enforced by the data/promotion path, delayed online labels are not fully stateful, and the reported dynamic regret is mathematically a sum of absolute comparator discrepancies. Until B1–B5 and the missing statistical tests are resolved, treat all detector alarms, accelerated forgetting, promotions, false-promotion rates, and dynamic-regret values as exploratory diagnostics—not validated statistical or economic evidence.
