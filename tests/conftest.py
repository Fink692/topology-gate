"""Small deterministic fixtures shared by package tests."""

from __future__ import annotations

import random

import pytest

from topology_gate import BacktestDatasetProtocol, Observation
from topology_gate.types import BacktestConfig


@pytest.fixture
def random_seed() -> int:
    """Seed value for reproducible worker-level tests."""

    return 17


@pytest.fixture
def rng(random_seed: int) -> random.Random:
    """A standard-library RNG so fixtures do not require NumPy."""

    return random.Random(random_seed)


@pytest.fixture
def sample_observations() -> tuple[Observation, ...]:
    """A short scalar-target sequence suitable for detector/learner tests."""

    return (
        Observation(features=(0.0, 0.25), target=0.10, timestamp=0),
        Observation(features=(0.25, 0.50), target=0.20, timestamp=1),
        Observation(features=(0.50, 0.75), target=0.35, timestamp=2),
        Observation(features=(0.75, 1.00), target=0.45, timestamp=3),
    )


@pytest.fixture
def sample_dataset() -> BacktestDatasetProtocol:
    """A deterministic worker-owned synthetic regime dataset.

    The import stays inside the fixture so importing the test configuration does
    not make NumPy a dependency of the core package.  The backtest worker owns
    the richer dataset shape and its delayed-label semantics.
    """

    synthetic = pytest.importorskip("topology_gate.synthetic")
    if hasattr(synthetic, "generate_regime_switching"):
        return synthetic.generate_regime_switching(
            n_samples=32,
            n_features=2,
            shift_points=(10, 20),
            seed=17,
            noise_scale=0.25,
        )
    if hasattr(synthetic, "generate_synthetic_regimes"):
        return synthetic.generate_synthetic_regimes(
            n_steps=32,
            n_features=2,
            seed=17,
            label_delay=1,
        )
    pytest.skip("synthetic worker does not expose a supported factory")


@pytest.fixture
def backtest_config() -> BacktestConfig:
    """Conservative defaults for a small offline evaluation."""

    return BacktestConfig(initial_train_size=1, label_delay=0)
