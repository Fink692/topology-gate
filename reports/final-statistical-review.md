# Final Statistical Review

## Verdict

**FAIL for statistically validated release.** The implementation is suitable only as a research/alpha-contract scaffold. The documentation correctly limits the claims to exploratory diagnostics (`docs/statistical-validity.md:3-9`, `:70-77`).

## Findings

1. **BLOCKER — persistent-topology claim is not supported.** The default statistic is explicitly a causal delay-embedded kNN normalized-Laplacian approximation, not persistent homology or a persistent Laplacian (`src/topology_gate/topology.py:1-16`, `:934-947`). This is a **PASS** only when described with that narrower name; it is a **FAIL** for persistent-topology, Betti, or persistent-Laplacian evidence (`docs/statistical-validity.md:63-68`).

2. **BLOCKER — detector alarms and forgetting factors are uncalibrated.** The alarm is a reflected accumulation of nonnegative RMS-whitened innovations (`src/topology_gate/topology.py:1238-1285`, `:1432-1469`), with rolling adaptive reference estimation and no null distribution, false-alarm rate, ARL, or uncertainty calculation. The `calibrated` mask establishes only a minimum count of earlier rows, not statistical calibration (`src/topology_gate/topology.py:1435-1442`). Thus thresholds and score-to-forgetting behavior must not be treated as level-α detection or validated adaptive learning (`docs/statistical-validity.md:70-77`).

3. **BLOCKER — end-to-end anytime-valid promotion is not established.** The e-process theorem is conditional on the bounded score null and predictable eta (`src/topology_gate/promotion.py:3-18`). However, the public gate accepts caller-supplied scores/utilities and eta values without enforcing label availability, frozen predictions, or the conditional null (`src/topology_gate/promotion.py:611-617`, `:1628-1708`); direct resets may also retain alpha (`src/topology_gate/promotion.py:794-810`). The isolated algebra can pass, but the composed promotion claim fails without optional-stopping and selection-budget evidence (`docs/statistical-validity.md:70-77`, `:79-90`).

4. **BLOCKER — economic comparator/regret claims remain caller-dependent.** The code validates comparator magnitude, not that `optimal_position` is actually optimal or point-in-time feasible (`src/topology_gate/backtest.py:777-791`). Supplied `expected_returns` are accepted as finite arrays and used as the comparator basis without an availability proof (`src/topology_gate/backtest.py:1091-1125`, `:1635-1654`). The explicit discrepancy and one-sided utility fields are defensible only under the documented common-basis/cost contract; the retained `dynamic_regret` field is expressly a deprecated legacy gap (`src/topology_gate/backtest.py:1161-1186`, `:1658-1686`; `docs/statistical-validity.md:15-47`).

5. **WARNING — delayed online replay is observable but not resumable through the runner.** The implementation correctly captures the forgetting factor at prediction time and returns terminal pending labels (`src/topology_gate/online.py:274-280`, `:340-391`, `:412-439`). But `run_recursive_rls` has no input for the returned stream/pending state and reinitializes its queue and position on each run (`src/topology_gate/online.py:262-280`, `:337-339`); its summary metrics also aggregate every row rather than exposing an availability-aware evaluation denominator (`src/topology_gate/online.py:394-410`). Exact chunked replay and delayed-label statistical reporting therefore remain unproven.

## Release recommendation

**Do not release for claims of calibrated change detection, persistent topology, anytime-valid promotion, dynamic regret, or validated economic performance.** A research-only release is acceptable if all such outputs are labeled exploratory. Before a statistical release, require independent null/block-bootstrap calibration with uncertainty, optional-stopping and adaptive-selection tests, a point-in-time comparator/cost audit, and resumable delayed-label replay with explicit denominators.
