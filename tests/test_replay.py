"""Adversarial tests for the shared causal replay transition."""

from __future__ import annotations

import copy

import pytest

from topology_gate.asof import AsOfBook, LabelObservation, MarketObservation
from topology_gate.replay import (
    ReplayConfig,
    ReplayState,
    ReplayStateError,
    ReplayStatus,
    run_causal_replay,
)


class CounterModel:
    def __init__(self) -> None:
        self.updates = 0
        self.seen_scores: list[float] = []

    def state_dict(self) -> dict[str, object]:
        return {"updates": self.updates, "seen_scores": list(self.seen_scores)}


def market(record_id: str, available: int, value: float) -> MarketObservation:
    return MarketObservation(
        record_id=record_id,
        instrument_id="ES",
        event_time=available,
        available_time=available,
        source_revision=0,
        ingest_sequence=0,
        fields={"value": value},
    )


def observed_label(target_id: str, available: int, value: float) -> LabelObservation:
    return LabelObservation(
        label_id=f"label-{target_id}",
        target_id=target_id,
        event_time=available - 1,
        available_time=available,
        received_time=available,
        status="observed",
        value=value,
        source_revision=0,
    )


def test_prediction_is_frozen_before_delayed_label_and_updates_at_next_boundary() -> None:
    book = AsOfBook(
        observations=(market("m1", 1, 10.0), market("m2", 10, 99.0)),
        labels=(observed_label("t1", 3, 2.0),),
    )
    model = CounterModel()
    seen: list[tuple[str, int]] = []

    def predict(snapshot, target_id):
        seen.append((target_id, len(snapshot.labels)))
        return snapshot.observation("m1").fields["value"]

    def score(prediction, label):
        return float(label.value) - float(prediction.value)

    def update(prediction, label, score_value):
        assert prediction.target_id == "t1"
        assert label.target_id == "t1"
        assert score_value == -8.0
        model.updates += 1
        model.seen_scores.append(score_value)

    result = run_causal_replay(
        book,
        (1, 2, 4),
        ("t1", "t2", "t3"),
        predict,
        model=model,
        score=score,
        on_label=update,
        config=ReplayConfig(model_id="counter", score_id="difference"),
    )

    assert seen == [("t1", 0), ("t2", 0), ("t3", 1)]
    assert result.resolutions[0].settlement_time == 4
    assert result.resolutions[0].status is ReplayStatus.OBSERVED
    assert result.resolutions[0].score == -8.0
    assert result.pending_target_ids == ("t2", "t3")
    assert model.updates == 1
    assert result.state.next_sequence == 4
    assert result.chain_digest != "0" * 64


def test_future_event_cannot_change_prefix_records() -> None:
    base = AsOfBook(observations=(market("m1", 1, 5.0),))
    extended = base.with_observation(market("future", 20, 500.0))

    def predict(snapshot, target_id):
        return snapshot.observation("m1").fields["value"]

    left = run_causal_replay(
        base, (1, 2), ("t1", "t2"), predict, model=CounterModel()
    )
    right = run_causal_replay(
        extended, (1, 2), ("t1", "t2"), predict, model=CounterModel()
    )
    assert [record.to_dict() for record in left.records] == [
        record.to_dict() for record in right.records
    ]


def test_missing_label_is_explicit_and_does_not_call_update() -> None:
    missing = LabelObservation(
        label_id="label-t1",
        target_id="t1",
        event_time=1,
        available_time=2,
        received_time=2,
        status="missing",
        value=None,
        source_revision=0,
    )
    calls: list[str] = []
    result = run_causal_replay(
        AsOfBook(observations=(market("m1", 1, 1.0),), labels=(missing,)),
        (1, 3),
        ("t1", "t2"),
        lambda snapshot, target: 1.0,
        model=CounterModel(),
        on_label=lambda prediction, label, score: calls.append(label.status),
    )
    assert calls == []
    assert result.resolutions[0].status is ReplayStatus.MISSING
    assert result.resolutions[0].score is None
    assert result.resolutions[0].reason == "label status is missing"


def test_preavailable_target_label_is_rejected() -> None:
    with pytest.raises(ValueError, match="already available"):
        run_causal_replay(
            AsOfBook(
                observations=(market("m1", 1, 1.0),),
                labels=(observed_label("t1", 1, 2.0),),
            ),
            (1,),
            ("t1",),
            lambda snapshot, target: 1.0,
            model=CounterModel(),
        )


def test_invalid_and_abstained_predictions_are_typed() -> None:
    values = [None, float("nan"), 2.0]

    def predict(snapshot, target):
        return values.pop(0)

    result = run_causal_replay(
        AsOfBook(), (1, 2, 3), ("t1", "t2", "t3"), predict, model=CounterModel()
    )
    assert [item.status for item in result.predictions] == [
        ReplayStatus.ABSTAINED,
        ReplayStatus.INVALID,
        ReplayStatus.PREDICTED,
    ]


def test_state_roundtrip_and_model_identity_are_fail_closed() -> None:
    book = AsOfBook(observations=(market("m1", 1, 1.0),))
    model = CounterModel()
    first = run_causal_replay(
        book,
        (1,),
        ("t1",),
        lambda snapshot, target: 1.0,
        model=model,
    )
    restored_state = ReplayState.from_state_dict(copy.deepcopy(first.state_dict()))
    restored_model = CounterModel()
    restored_model.seen_scores = [1.0]
    with pytest.raises(ReplayStateError, match="does not match"):
        run_causal_replay(
            book,
            (2,),
            ("t2",),
            lambda snapshot, target: 2.0,
            model=restored_model,
            initial_state=restored_state,
        )
    restored_model.updates = model.updates
    restored_model.seen_scores = list(model.seen_scores)
    resumed = run_causal_replay(
        book,
        (2,),
        ("t2",),
        lambda snapshot, target: 2.0,
        model=restored_model,
        initial_state=restored_state,
    )
    assert resumed.predictions[-1].sequence == 1
    assert len(resumed.records) == 2


def test_tampered_chain_state_is_rejected() -> None:
    result = run_causal_replay(
        AsOfBook(), (1,), ("t1",), lambda snapshot, target: 1.0, model=CounterModel()
    )
    state = result.state_dict()
    state["records"][0]["payload"]["target_id"] = "tampered"
    with pytest.raises(ReplayStateError):
        ReplayState.from_state_dict(state)
