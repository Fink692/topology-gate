"""Focused subprocess checks for dependency-light and lazy root imports."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run_without_site_packages(script: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    source_path = str(ROOT / "src")
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_path + os.pathsep + existing if existing else source_path
    )
    return subprocess.run(
        [sys.executable, "-S", "-c", script],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def test_core_root_import_does_not_require_numpy() -> None:
    result = _run_without_site_packages(
        """
import topology_gate
from topology_gate import BacktestConfig, BacktestResult, GateStatus, TopologyResult
assert topology_gate.__version__ == '0.1.0'
assert BacktestConfig.__module__ == 'topology_gate.types'
assert BacktestResult.__module__ == 'topology_gate.types'
assert GateStatus.__module__ == 'topology_gate.types'
assert TopologyResult is topology_gate.TopologyResult
"""
    )
    assert result.returncode == 0, result.stderr


def test_lazy_numeric_import_has_an_explicit_install_hint() -> None:
    result = _run_without_site_packages(
        """
try:
    from topology_gate import OnlineRunConfig
except ImportError as exc:
    assert 'topology-gate[numeric]' in str(exc)
else:
    raise AssertionError('OnlineRunConfig unexpectedly imported without site packages')
"""
    )
    assert result.returncode == 0, result.stderr


def test_root_all_is_present_as_an_import_contract() -> None:
    result = _run_without_site_packages(
        """
import topology_gate
core = {'ArrayLike', 'BacktestConfig', 'BacktestResult', 'GateStatus', 'TopologyResult'}
assert core.issubset(set(topology_gate.__all__))
for name in core:
    assert hasattr(topology_gate, name)
"""
    )
    assert result.returncode == 0, result.stderr
