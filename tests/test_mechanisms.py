"""Tests for mechanism-localized continual learning."""

from __future__ import annotations

import pytest

from topology_gate import MechanismLocalizedRLS as PublicMechanismLocalizedRLS
from topology_gate import MechanismSpec as PublicMechanismSpec
from topology_gate.mechanisms import (
    MECHANISM_SCHEMA,
    MECHANISM_VERSION,
    MechanismLocalizedConfig,
    MechanismLocalizedRLS,
    MechanismSpec,
)


def _config() -> MechanismLocalizedConfig:
    return MechanismLocalizedConfig(
        mechanisms=(
            MechanismSpec("a", (0, 1)),
            MechanismSpec("b", (0, 2)),
        ),
        stable_forgetting_factor=0.99,
        shift_forgetting_factor=0.7,
        residual_history=8,
        minimum_history=4,
        residual_scale_floor=0.01,
        drift_threshold=3.0,
    )


def test_localized_shift_freezes_unchanged_module() -> None:
    model = MechanismLocalizedRLS(_config())
    for index in range(8):
        x = float(index + 1)
        model.observe(
            (1.0, x, x),
            {"a": 0.5 + 0.8 * x, "b": 0.2 + 0.4 * x},
        )
    stable_before = model.learners["a"].state_dict()
    update = model.observe(
        (1.0, 1.0, 1.0),
        {"a": 1.3, "b": 8.0},
    )
    assert update.shifted_mechanisms == ("b",)
    assert update.updated_mechanisms == ("b",)
    assert model.learners["a"].state_dict() == stable_before
    assert update.observations[1].forgetting_factor == 0.7


def test_stable_transition_updates_all_modules() -> None:
    model = MechanismLocalizedRLS(_config())
    first = model.observe((1.0, 1.0, 2.0), {"a": 1.0, "b": 2.0})
    assert first.shifted_mechanisms == ()
    assert first.updated_mechanisms == ("a", "b")
    assert all(
        observation.forgetting_factor == 0.99
        for observation in first.observations
    )


def test_state_round_trip_and_digest_tamper_rejection() -> None:
    model = MechanismLocalizedRLS(_config())
    model.observe((1.0, 2.0, 3.0), {"a": 2.0, "b": 3.0})
    state = model.state_dict()
    assert state["schema"] == MECHANISM_SCHEMA
    assert state["version"] == MECHANISM_VERSION
    restored = MechanismLocalizedRLS.from_state_dict(state)
    assert restored.state_dict() == state
    tampered = dict(state)
    tampered["step"] = 999
    with pytest.raises(ValueError, match="digest"):
        MechanismLocalizedRLS.from_state_dict(tampered)


def test_config_and_input_validation() -> None:
    with pytest.raises(ValueError, match="unique"):
        MechanismLocalizedConfig(
            mechanisms=(MechanismSpec("a", (0,)), MechanismSpec("a", (1,)))
        )
    model = MechanismLocalizedRLS(_config())
    with pytest.raises(ValueError, match="exactly"):
        model.observe((1.0, 2.0, 3.0), {"a": 1.0})
    with pytest.raises(ValueError, match="length"):
        model.predict((1.0,))
    with pytest.raises(ValueError, match="finite"):
        model.predict((1.0, float("nan"), 2.0))


def test_spec_and_config_json_round_trip() -> None:
    config = _config()
    assert MechanismLocalizedConfig.from_dict(config.to_dict()) == config
    assert MechanismSpec.from_dict(config.mechanisms[0].to_dict()) == config.mechanisms[0]
    assert PublicMechanismLocalizedRLS is MechanismLocalizedRLS
    assert PublicMechanismSpec is MechanismSpec
