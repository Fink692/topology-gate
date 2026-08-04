"""Tests for prefix-causal transport replay."""

from __future__ import annotations

import numpy as np
import pytest

from topology_gate.transport import (
    CausalTransportReplay,
    TransportReplayConfig,
)


def _replay() -> CausalTransportReplay:
    return CausalTransportReplay(
        TransportReplayConfig(
            n_features=2,
            drift_sensitivity=1.0,
            location_sensitivity=1.0,
            minimum_weight=0.01,
        )
    )


def test_transport_replay_enforces_availability_and_prefix_source() -> None:
    replay = _replay()
    replay.observe_state(0, [1.0, 0.0], feature_location=[0.0, 0.0])
    replay.observe_state(2, [2.0, 0.0], feature_location=[1.0, 0.0])
    replay.append(
        0,
        2,
        [1.0, 2.0],
        3.0,
        [1.0, 0.0],
        feature_location=[0.0, 0.0],
    )
    replay.append(
        1,
        3,
        [4.0, 5.0],
        7.0,
        [1.0, 0.0],
        feature_location=[0.0, 0.0],
    )

    before = replay.batch(2)
    assert before.n_rows == 0
    after = replay.batch(3)
    assert after.source_steps == (0,)
    assert after.available_steps == (2,)
    np.testing.assert_allclose(after.features, [[2.0, 2.0]])
    # 3 + [2, 2] @ ([2, 0] - [1, 0]) = 5.
    np.testing.assert_allclose(after.labels, [5.0])
    assert 0.0 < after.weights[0] <= 1.0


def test_later_state_and_future_record_do_not_change_an_earlier_batch() -> None:
    replay = _replay()
    replay.observe_state(0, [0.0, 0.0])
    replay.observe_state(1, [1.0, 0.0])
    replay.append(0, 1, [1.0, 0.0], 1.0, [0.0, 0.0])
    first = replay.batch(2)

    replay.append(5, 6, [100.0, 100.0], 100.0, [99.0, 99.0])
    replay.observe_state(3, [99.0, 99.0])
    second = replay.batch(2)

    assert second.snapshot_step == first.snapshot_step
    assert second.source_steps == first.source_steps
    np.testing.assert_array_equal(second.features, first.features)
    np.testing.assert_array_equal(second.labels, first.labels)
    np.testing.assert_array_equal(second.weights, first.weights)


def test_transport_state_round_trip_and_tamper_detection() -> None:
    replay = _replay()
    replay.observe_state(0, [0.0, 0.0], feature_location=[0.0, 0.0])
    replay.append(0, 1, [1.0, 2.0], 3.0, [0.0, 0.0], feature_location=[0.0, 0.0])
    restored = CausalTransportReplay.from_state_dict(replay.state_dict())
    assert restored.identity == replay.identity
    np.testing.assert_array_equal(restored.batch(2).features, replay.batch(2).features)

    tampered = replay.state_dict()
    tampered["records"][0]["label"] = 999.0
    with pytest.raises(ValueError, match="identity"):
        CausalTransportReplay.from_state_dict(tampered)


def test_transport_rejects_future_states_invalid_records_and_limits() -> None:
    replay = _replay()
    replay.observe_state(2, [0.0, 0.0])
    with pytest.raises(ValueError, match="strictly increasing"):
        replay.observe_state(2, [0.0, 0.0])
    with pytest.raises(ValueError, match="strictly after"):
        replay.append(2, 2, [0.0, 0.0], 0.0, [0.0, 0.0])

    limited = CausalTransportReplay(
        TransportReplayConfig(n_features=1, max_records=1, max_states=1)
    )
    limited.observe_state(0, [0.0])
    with pytest.raises(ValueError, match="state limit"):
        limited.observe_state(1, [0.0])
    limited.append(0, 1, [0.0], 0.0, [0.0])
    with pytest.raises(ValueError, match="record limit"):
        limited.append(1, 2, [0.0], 0.0, [0.0])


def test_transport_config_rejects_invalid_weight_policy() -> None:
    with pytest.raises(ValueError, match="minimum_weight"):
        TransportReplayConfig(n_features=1, minimum_weight=0.0)
