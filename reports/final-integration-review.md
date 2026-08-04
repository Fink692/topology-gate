# Final integration review

## Verdict

**FAIL for production/live-trading release. PASS only for the explicitly documented research/alpha boundary.**

## Findings

1. **Blocker — restartable online replay is incomplete.** `run_recursive_rls` has no input for `OnlineStreamState`, always starts with an empty pending-label queue and zero prior position, while checkpoint restore only returns `online_state`; the runbook nevertheless describes this as restartable replay. Evidence: `src/topology_gate/online.py:262-280`, `src/topology_gate/online.py:328-338`, `src/topology_gate/online.py:412-419`, `src/topology_gate/checkpoint.py:337-358`, `docs/production-runbook.md:75-80`.

2. **Blocker — delayed-label timing is not type-safe.** `OnlineRunConfig` checks only ordering/finite behavior, so a non-integer `label_delay` can pass; the resulting availability is later truncated with `int(...)`, changing the causal update boundary. Evidence: `src/topology_gate/online.py:166-189`, `src/topology_gate/online.py:294-317`, `src/topology_gate/online.py:384-390`.

3. **Warning — checkpoint compatibility is caller-only at the restore helper.** `restore_component_states` verifies integrity but accepts no expected package, configuration, backend, or dependency identities; an incompatible but validly signed envelope can therefore be restored unless callers separately use the expectation checks. Evidence: `src/topology_gate/checkpoint.py:199-239`, `src/topology_gate/checkpoint.py:319-358`, `docs/production-runbook.md:75-80`.

4. **Warning — optional-dependency error handling is incomplete.** The lazy export table includes `.rls` and `.topology`, but the actionable missing-dependency wrapper only recognizes `.backtest`, `.online`, and `.synthetic`; failures from the other worker modules can escape as raw `ModuleNotFoundError`. Evidence: `src/topology_gate/__init__.py:64-100`, `src/topology_gate/__init__.py:116-128`, `docs/production-runbook.md:16-18`.

5. **Warning — declared interpreter support exceeds the exercised release gate.** Metadata advertises Python `>=3.10` and classifies 3.10–3.12, but the release configuration records only Python 3.12 as tested. Evidence: `pyproject.toml:10`, `pyproject.toml:20-22`, `pyproject.toml:85-90`.

## Release recommendation

Do not promote this boundary to production or live trading. It may ship as a research-only 0.1.0 alpha if the runbook limitations remain prominent; before any stronger release, add true stream-state continuation, reject non-integer delays, enforce compatibility at restore, complete lazy-import handling, and either test or narrow the supported Python range.

Verification note: the two requested test modules were not executable in the active interpreter because `pytest` is not installed.
