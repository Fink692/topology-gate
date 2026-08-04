# Integration / Release Review

**Review date:** 2026-08-04
**Decision:** **NO-GO for a production release; acceptable only as an explicitly alpha/research offline component.**

This review uses the current worktree and the captured validation evidence. No source or test files were modified, and no checks were rerun during finalization.

## Executive summary

The numerical/test path is green in the captured Python 3.12 environment: 50 tests passed, Ruff passed, mypy reported no issues, compileall passed, and the wheel/install smoke evidence passed. The wheel was buildable as `topology_gate-0.1.0-py3-none-any.whl` and included `online.py`.

The release is still blocked by public-contract inconsistencies, a duplicate import namespace in the tests, incomplete optional-dependency behavior at the package root, weakened type-check coverage, and missing release controls. The architecture document itself says the production architecture is not implemented yet (`docs/architecture.md:9`) and calls the current code an alpha compatibility surface (`docs/architecture.md:274`). That is incompatible with the runbook’s unqualified “production-ready” claim (`docs/production-runbook.md:3`).

## Validation evidence

| Gate | Exact evidence | Result |
|---|---|---|
| Tests | `C:\Users\Kristjan Backman\AppData\Local\Programs\Python\Python312\python.exe -m pytest -p no:cacheprovider` | **PASS** — 50 collected, 50 passed in 5.78s |
| Type check | `...Python312\python.exe -m mypy --cache-dir <external-temp>` | **PASS, qualified** — no issues in 10 files; `rls` and `backtest` are configured with `ignore_errors = true` |
| Lint | `ruff check --no-cache src tests` | **PASS** — supplemental only; Ruff is not declared/pinned by the project |
| Coverage | pytest-cov run with `--cov=topology_gate --cov-report=term-missing` | **PASS, no threshold** — 50 tests passed, 78% total coverage |
| Compile/import | Existing compileall and import-smoke evidence | **PASS** — core import works without site packages; all package modules import with NumPy |
| Packaging | Existing wheel/install smoke evidence; isolated `pip wheel` produced `topology_gate-0.1.0-py3-none-any.whl` | **PASS** — wheel included all package modules, including `online.py` |

The current active `python` is Python 3.11.15 but has no pip, pytest, mypy, Ruff, or build module. These exact commands fail in that interpreter:

```text
python -m pip --version       -> No module named pip
python -m pytest --version    -> No module named pytest
python -m mypy --version      -> No module named mypy
python -m ruff --version      -> No module named ruff
python -m build --version     -> No module named build
```

Python 3.12 tooling exists locally, but the repository has no pinned environment, lock file, CI matrix, or release workflow proving that the declared Python `>=3.10` range is reproducible.

## Blockers

### 1. Public API names do not describe one stable contract

- `src/topology_gate/__init__.py:52` lazily maps root `GateStatus` to `promotion.GateStatus`.
- `src/topology_gate/types.py:26` defines a different shared `GateStatus`; `GateDecision.status` is typed against that shared enum.
- Runtime evidence: `root_gate_status_is_types=False`; root values are `open/promoted`, while shared values are `open/closed/topology_rejected/insufficient_evidence/invalid_evidence`.
- `RollingTopologyDetector` is in `_LAZY_EXPORTS` (`__init__.py:55`) but absent from root `__all__` (`__init__.py:91` onward), so direct import and star import expose different APIs.
- Root `BacktestConfig` and `TopologyResult` resolve to worker objects rather than the dependency-light/shared objects described in `types.py`.

This is a breaking-consumer risk: callers can import two different meanings for the same public name, and `from topology_gate import *` is not equivalent to the documented root API.

### 2. The RLS implementation does not match its public protocol

`RLSLearnerProtocol.update` promises `Vector` (`src/topology_gate/types.py:290-298`), while the public `RLS.update` returns `RLSUpdate` (`src/topology_gate/rls.py:866` and the captured signature evidence). The README describes the boundary as returning coefficients/state, but no single return contract is enforced. This must be resolved before claiming protocol compatibility.

### 3. “Dependency-free core” is not true for all root exports

`import topology_gate` succeeds under `python -S`, but these exact no-NumPy checks fail with `ModuleNotFoundError: No module named 'numpy'` from `src/topology_gate/backtest.py:16`:

```text
from topology_gate import BacktestConfig
from topology_gate import *
```

`topology_gate.types.BacktestConfig` does import without NumPy. The root lazy-export map therefore makes a supposedly dependency-light public name depend on a numerical worker. `backtest.py`, `online.py`, and `synthetic.py` import NumPy eagerly (`backtest.py:16`, `online.py:9`, `synthetic.py:15`). The behavior must be made intentional and documented, or the shared/root boundary must be separated cleanly.

### 4. The type gate is materially bypassed for two major modules

Current `pyproject.toml:50-64` sets `disallow_any_generics = false`, `warn_unused_ignores = false`, and `ignore_errors = true` for `topology_gate.rls` and `topology_gate.backtest`. The green mypy result is therefore not evidence that those public adapters type-check. This is especially material because the RLS protocol mismatch is in one of the ignored modules.

### 5. The repository cannot provide a reproducible release boundary

- `pyproject.toml` has only minimum dependency versions and no lock or constraints file.
- No CI/workflow or release automation is present.
- Git’s root is `C:/Users/Kristjan Backman`, not this project directory; the root reports “No commits yet on master” and the project is uncommitted/untracked within a broad user-home working tree.
- The declared `python -m pip install -e "[test]"` workflow cannot even start in the active interpreter because pip is absent.

The architecture release gate explicitly requires dependency-lock verification, API documentation, a reproducible example, and full test/type evidence (`docs/architecture.md:553`). Only a local Python 3.12 snapshot has been evidenced.

### 6. Tests exercise two different module identities

`tests/test_topology.py:10-17` and `tests/test_rls.py:5-6` import `src.topology_gate.*`; the other tests import `topology_gate.*`. Captured runtime evidence showed:

```text
types_same=False
topology_same=False
observation_class_same=False
```

The same source file is consequently loaded as two module namespaces. This can hide packaging/import defects and makes class identity, `isinstance`, exception, and protocol behavior environment-dependent.

### 7. Typed-package distribution is incomplete

The project calls itself typed and runs mypy against local source, but `src/topology_gate/py.typed` is absent. Downstream type checkers are not guaranteed to treat the installed package as typed. Add this to the release contract or explicitly stop claiming downstream typing support.

## Warnings and missing gates

1. **Coverage is measured but unenforced.** The captured total is 78%; `__init__.py` is 50%, `backtest.py` 71%, `synthetic.py` 65%, and `rls.py` 77%. There is no `fail_under` or coverage add-on in pytest configuration. Root API, import-without-NumPy, packaging, and namespace-collision tests are absent.
2. **Ruff is not a declared gate.** `ruff check` passes, but Ruff is not in either optional dependency set and no Ruff configuration/version is in `pyproject.toml`. The runbook command (`docs/production-runbook.md:11-13`) will not be reproducible from `.[dev]` alone.
3. **Optional extras do not match current code boundaries.** NumPy is used by current workers, but no source module imports SciPy or pandas. The `statistics` and `data` extras (`pyproject.toml:26-33`) look forward-looking; either provide those adapters or describe the extras as reserved/future.
4. **Documentation is internally inconsistent.** The architecture document says production is not implemented and the scaffold is alpha; the production runbook says “production-ready.” README text says the core has no runtime dependencies, while root star/direct imports can require NumPy. These claims need one release posture.
5. **The release test taxonomy is too narrow.** Only five test modules exist, and the declared `integration`/`slow` markers are not used. No captured evidence covers the architecture’s required contract, causal/leakage, golden topology, replay/recovery, fault, performance, dependency-lock, statistical-validity, or security/reproducibility gates.
6. **Version identity can drift.** `pyproject.toml:7` and `src/topology_gate/__init__.py:42` each hard-code `0.1.0`; there is no single source of truth or release check.
7. **Supported-version evidence is incomplete.** The metadata advertises Python 3.10+, but the captured functional run is Python 3.12 only. There is no matrix evidence for 3.10, 3.11, or 3.13 and no dependency upper-bound policy.

## What is good and proven

- The package builds into a wheel and the wheel contains the current `online.py` module.
- The captured 50-test suite covers topology, RLS, promotion, backtest, online composition, delayed labels, deterministic behavior, and several invalid-input cases.
- Ruff and compile/import smoke checks are clean in the available toolchain.
- `import topology_gate` remains usable without NumPy, and the topology module has an explicit standard-library fallback path.
- `.gitignore` excludes common generated caches/build artifacts, and `LICENSE` is present.

## Release disposition

**NO-GO.** The green local test result is necessary but not sufficient. Before release, resolve the root API/enum/protocol contracts, eliminate the `src.topology_gate` namespace split, define the optional-dependency contract, restore meaningful type checking for `rls`/`backtest`, add `py.typed`, pin and automate the environment, and attach the missing causal/statistical/replay/fault/performance/security evidence required by the architecture. Until then, label the artifact as alpha/research-only and do not use the runbook’s “production-ready” claim.
