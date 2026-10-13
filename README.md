# Topology Gate

Typed Python research infrastructure for adaptive forgetting, recursive learning, causal replay, and model-promotion decisions under changing data distributions.

**Status: research alpha.** Synthetic studies and historical paper diagnostics are available. There is no broker integration or live order submission.

## What it contains

- Recursive least-squares learning with delayed-label replay
- Topology/change-detection adapters and explicit forgetting controls
- Bounded e-process and robust expert-promotion primitives
- Source manifests, calibration certificates, checkpoints, and audit receipts
- Resource-bounded persistent-Laplacian reference experiments

The default rolling detector is a normalized-Laplacian spectral approximation. It is not a persistent-homology implementation. Statistical guarantees apply only under the documented score, data-availability, and predictable-betting assumptions.

## Run locally

Python 3.10, 3.11, or 3.12:

```sh
python -m pip install -e ".[dev]"
python -m pytest
ruff check src tests examples
mypy src
PYTHONPATH=src python examples/integrated_synthetic_study.py
```

[Installation and architecture](docs/architecture.md) · [Study runbook](docs/study-runbook.md)

## Research evidence

| Track | Evidence |
| --- | --- |
| Integrated synthetic study | [Protocol](docs/integrated-synthetic-study.md) |
| Public-history diagnostics | [Guide](docs/free-public-track.md) and [limitations](docs/public-market-diagnostic.md) |
| Trend-filter paper model | [Report](reports/trend-filter-paper-model.json) and [walk-forward report](reports/trend-filter-walkforward.json) |
| Calibration | [Methods](docs/calibration.md) and [statistical validity](docs/statistical-validity.md) |
| Source readiness | [Vendor-data gate](docs/vendor-data-gate.md) and [current research status](docs/research-status.md) |

Final adjusted public history is not point-in-time evidence. Historical paper results do not establish profitability, capacity, delisting coverage, or future performance. Promotion requires the relevant source and calibration evidence; a successful synthetic run is insufficient.

## Repository guide

- `src/topology_gate/`: contracts, learning, replay, detectors, promotion, and checkpoints
- `examples/`: runnable studies and paper-only diagnostics
- `reports/`: recorded results and review evidence
- `docs/`: statistical assumptions, architecture, source gates, and runbooks
- `tests/`: correctness and regression checks

MIT licensed. See [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md).
