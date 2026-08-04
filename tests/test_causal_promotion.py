"""Integration tests for causal paired challenger promotion."""

from __future__ import annotations

import copy

import pytest

from topology_gate.asof import (
    AsOfBook,
    LabelObservation,
    MarketObservation,
    UniverseMembership,
)
from topology_gate.causal_numeric import CausalFeaturePlan, FeatureBinding
from topology_gate.causal_promotion import (
    CausalPromotionConfig,
    CausalPromotionError,
    run_causal_promotion_replay,
)
from topology_gate.promotion import PromotionGate
from topology_gate.replay import ReplayConfig, ReplayStatus


class FixedLearner:
    n_features = 1

    def __init__(self, prediction: float, *, fail_update: bool = False) -> None:
        self.prediction = prediction
        self.fail_update = fail_update
        self.updates = 0

    def predict(self, features: tuple[float, ...]) -> float:
        assert len(features) == 1
        return self.prediction

    def update(self, features: tuple[float, ...], target: float) -> None:
        assert len(features) == 1
        if self.fail_update:
            raise RuntimeError("intentional update failure")
        assert target == 0.0
        self.updates += 1

    def state_dict(self) -> dict[str, object]:
        return {"prediction": self.prediction, "updates": self.updates}

    def load_state_dict(self, state: dict[str, object]) -> None:
        if set(state) != {"prediction", "updates"}:
            raise ValueError("invalid fixed learner state")
        self.prediction = float(state["prediction"])
        self.updates = int(state["updates"])


def binding_plan() -> CausalFeaturePlan:
    return CausalFeaturePlan(
        {
            target: (FeatureBinding("m1", ("x",), "ES"),)
            for target in ("t1", "t2", "t3", "t4", "t5")
        },
        require_membership=True,
    )


def observation_book(*, missing: bool = False) -> AsOfBook:
    labels = tuple(
        LabelObservation(
            label_id=f"label-{target}",
            target_id=target,
            event_time=available - 1,
            available_time=available,
            received_time=available,
            status="missing" if missing and target == "t1" else "observed",
            value=None if missing and target == "t1" else 0.0,
            source_revision=0,
        )
        for target, available in zip(("t1", "t2", "t3", "t4"), (2, 3, 4, 5))
    )
    return AsOfBook(
        observations=(
            MarketObservation(
                record_id="m1",
                instrument_id="ES",
                event_time=1,
                available_time=1,
                source_revision=0,
                ingest_sequence=0,
                fields={"x": 1.0},
            ),
        ),
        universe=(UniverseMembership("ES", 0, 100, 0, 0, 0),),
        labels=labels,
    )


def make_gate() -> PromotionGate:
    gate = PromotionGate("incumbent", alpha=0.9, eta=0.5)
    gate.register_challenger("challenger")
    return gate


def config() -> CausalPromotionConfig:
    return CausalPromotionConfig(
        promotion_id="paired",
        challenger_id="challenger",
        incumbent_id="incumbent",
        eta=0.5,
        utility_cap=1.0,
    )


def replay_settings(*, finalize_unresolved: bool) -> ReplayConfig:
    return ReplayConfig(
        model_id="paired",
        score_id="none",
        require_model_state=True,
        finalize_unresolved=finalize_unresolved,
    )


def run(
    decisions: tuple[int, ...],
    targets: tuple[str, ...],
    *,
    challenger: FixedLearner | None = None,
    incumbent: FixedLearner | None = None,
    gate: PromotionGate | None = None,
    replay_config: ReplayConfig | None = None,
    model_state: dict[str, object] | None = None,
    initial_state=None,
    missing: bool = False,
):
    return run_causal_promotion_replay(
        observation_book(missing=missing),
        decisions,
        targets,
        plan=binding_plan(),
        challenger=challenger or FixedLearner(0.0),
        incumbent=incumbent or FixedLearner(1.0),
        gate=gate or make_gate(),
        config=config(),
        replay_config=replay_config,
        model_state=model_state,
        initial_state=initial_state,
    )


def test_paired_predictions_only_advance_the_gate_at_label_settlement() -> None:
    result = run(
        (1, 2, 3, 4, 5),
        ("t1", "t2", "t3", "t4", "t5"),
        replay_config=replay_settings(finalize_unresolved=True),
    )

    assert result.promoted
    assert [item.status for item in result.replay.predictions] == [
        ReplayStatus.PREDICTED,
        ReplayStatus.PREDICTED,
        ReplayStatus.PREDICTED,
        ReplayStatus.PREDICTED,
        ReplayStatus.ABSTAINED,
    ]
    assert result.replay.resolutions[-1].status is ReplayStatus.UNRESOLVED
    assert result.pending_target_ids == ()
    assert result.state.model_state["pending"] == {}
    assert result.state.model_state["gate"]["promoted_challenger_id"] == "challenger"


def test_chunked_promotion_replay_matches_one_shot_state() -> None:
    settings = replay_settings(finalize_unresolved=False)
    one_shot = run(
        (1, 2, 3, 4, 5),
        ("t1", "t2", "t3", "t4", "t5"),
        replay_config=settings,
    )
    first = run(
        (1, 2, 3),
        ("t1", "t2", "t3"),
        replay_config=settings,
    )
    second = run(
        (4, 5),
        ("t4", "t5"),
        replay_config=settings,
        model_state=copy.deepcopy(first.state.model_state),
        initial_state=first.state,
    )

    assert second.all_predictions == one_shot.all_predictions
    assert first.steps + second.steps == one_shot.steps
    assert second.state.model_state == one_shot.state.model_state
    assert second.pending_target_ids == ("t5",)


def test_missing_and_unresolved_labels_do_not_feed_promotion_or_leak_context() -> None:
    result = run(
        (1, 2),
        ("t1", "t2"),
        replay_config=replay_settings(finalize_unresolved=True),
        missing=True,
    )
    # The helper's first label is missing, and the second prediction is
    # terminalized unresolved.  Neither is evidence for the gate.
    assert result.state.model_state["pending"] == {}
    assert result.state.model_state["gate"]["challengers"][0]["state"]["process"][
        "observation_count"
    ] == 0


def test_paired_update_and_gate_transition_roll_back_together() -> None:
    challenger = FixedLearner(0.0)
    incumbent = FixedLearner(1.0, fail_update=True)
    gate = make_gate()
    before_challenger = challenger.state_dict()
    before_incumbent = incumbent.state_dict()
    before_gate = gate.state_dict()

    with pytest.raises(RuntimeError, match="intentional update failure"):
        run(
            (1, 2),
            ("t1", "t2"),
            challenger=challenger,
            incumbent=incumbent,
            gate=gate,
            replay_config=replay_settings(finalize_unresolved=False),
        )

    assert challenger.state_dict() == before_challenger
    assert incumbent.state_dict() == before_incumbent
    assert gate.state_dict() == before_gate


def test_state_identity_and_constant_eta_contract_are_fail_closed() -> None:
    with pytest.raises(CausalPromotionError, match="constant"):
        dynamic_gate = PromotionGate("incumbent", alpha=0.9, eta=lambda history: 0.5)
        dynamic_gate.register_challenger("challenger")
        run(
            (1,),
            ("t1",),
            gate=dynamic_gate,
            replay_config=replay_settings(finalize_unresolved=False),
        )
