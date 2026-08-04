"""Tests for causal adaptive path-signature memory."""

from __future__ import annotations

import pytest

from topology_gate import AdaptiveSignatureMemory as PublicAdaptiveSignatureMemory
from topology_gate import SignatureMemoryConfig as PublicSignatureMemoryConfig
from topology_gate.signatures import (
    SIGNATURE_SCHEMA,
    SIGNATURE_VERSION,
    AdaptiveSignatureMemory,
    SignatureMemoryConfig,
    path_signature,
    signature_dimension,
)


def _config() -> SignatureMemoryConfig:
    return SignatureMemoryConfig(
        input_dim=2,
        candidate_depths=(1, 2),
        window=4,
        ridge=1.0,
        forgetting_factor=0.98,
        switching_cost=0.01,
        loss_clip=10.0,
    )


def test_signature_dimension_and_empty_path() -> None:
    assert signature_dimension(2, 2) == 7
    assert path_signature([], input_dim=2, depth=2) == (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def test_signature_is_ordered_and_prefix_causal() -> None:
    first = path_signature([(1.0, 0.0), (0.0, 1.0)], input_dim=2, depth=2)
    reversed_path = path_signature([(0.0, 1.0), (1.0, 0.0)], input_dim=2, depth=2)
    prefix = path_signature([(1.0, 0.0)], input_dim=2, depth=2)
    assert first != reversed_path
    assert prefix == path_signature([(1.0, 0.0)], input_dim=2, depth=2)


def test_adaptive_memory_updates_candidates_and_round_trips() -> None:
    memory = AdaptiveSignatureMemory(_config())
    memory.observe([(1.0, 0.0)], 0.5)
    update = memory.observe([(1.0, 1.0)], 1.0)
    assert update.step == 2
    assert update.selected_depth in (1, 2)
    assert len(update.candidate_losses) == 2
    state = memory.state_dict()
    assert state["schema"] == SIGNATURE_SCHEMA
    assert state["version"] == SIGNATURE_VERSION
    restored = AdaptiveSignatureMemory.from_state_dict(state)
    assert restored.state_dict() == state
    tampered = dict(state)
    tampered["step"] = 99
    with pytest.raises(ValueError, match="digest"):
        AdaptiveSignatureMemory.from_state_dict(tampered)


def test_signature_inputs_fail_closed() -> None:
    memory = AdaptiveSignatureMemory(_config())
    with pytest.raises(ValueError, match="exactly"):
        memory.predict([(1.0, 2.0, 3.0)])
    with pytest.raises(ValueError, match="finite"):
        memory.observe([(float("nan"), 0.0)], 1.0)
    with pytest.raises(ValueError, match="finite"):
        memory.observe([(1.0, 0.0)], float("nan"))


def test_config_round_trip() -> None:
    config = _config()
    assert SignatureMemoryConfig.from_dict(config.to_dict()) == config
    assert PublicAdaptiveSignatureMemory is AdaptiveSignatureMemory
    assert PublicSignatureMemoryConfig is SignatureMemoryConfig
