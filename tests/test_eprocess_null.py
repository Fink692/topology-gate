"""Bounded optional-stopping evidence checks for the promotion primitive."""

from __future__ import annotations

import random

import pytest

from topology_gate.promotion import EProcess


def test_rademacher_null_has_no_obvious_optional_stopping_explosion() -> None:
    rng = random.Random(20260804)
    crossings = 0
    paths = 200
    horizon = 250
    for _ in range(paths):
        process = EProcess(alpha=0.05, eta=0.5)
        for _ in range(horizon):
            update = process.update(1.0 if rng.randrange(2) else -1.0)
            if update.threshold_crossed:
                crossings += 1
                break
    # This is a regression simulation, not a theorem or a replacement for a
    # pre-registered dependence-aware calibration study.
    assert crossings / paths < 0.15


def test_eprocess_overflow_is_rejected_without_state_commit() -> None:
    process = EProcess(alpha=0.1, eta=1.0, initial_wealth=1.0e308)
    with pytest.raises(FloatingPointError, match="overflow"):
        process.update(1.0)
    assert process.e_value == 1.0e308
    assert process.observations == 0
