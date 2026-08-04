"""Focused tests for the immutable point-in-time event contract."""

from __future__ import annotations

import math

import pytest

from topology_gate.asof import (
    UNIVERSE_PRECEDENCE,
    AmbiguousEventError,
    AsOfBook,
    DuplicateEventError,
    LabelObservation,
    MarketObservation,
    MissingLabelError,
    PointInTimePanel,
    UnavailableEventError,
    UniverseMembership,
    canonical_event_order,
)


def market(
    record_id: str,
    available: int,
    *,
    revision: int = 0,
    ingest: int = 0,
    value: float = 1.0,
) -> MarketObservation:
    return MarketObservation(
        record_id=record_id,
        instrument_id="ES",
        event_time=available - 1,
        available_time=available,
        source_revision=revision,
        ingest_sequence=ingest,
        fields={"value": value},
    )


def label(
    label_id: str,
    target_id: str,
    available: int,
    *,
    status: str = "observed",
    value: float | None = 2.0,
    revision: int = 0,
    ingest: int = 0,
) -> LabelObservation:
    return LabelObservation(
        label_id=label_id,
        target_id=target_id,
        event_time=available - 2,
        available_time=available,
        received_time=available + 1,
        status=status,
        value=value,
        source_revision=revision,
        ingest_sequence=ingest,
    )


def test_events_are_immutable_and_fields_are_copy_safe() -> None:
    fields = {"value": 1.0}
    observation = market("m1", 2)
    fields["value"] = 99.0
    assert observation.fields["value"] == 1.0
    with pytest.raises(TypeError):
        observation.fields["value"] = 3.0  # type: ignore[index]
    with pytest.raises((AttributeError, TypeError)):
        observation.record_id = "other"  # type: ignore[misc]


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_feature_and_label_values_are_rejected(bad: float) -> None:
    with pytest.raises(ValueError):
        market("bad", 1, value=bad)
    with pytest.raises(ValueError):
        label("bad", "t", 2, value=bad)


def test_duplicate_revision_and_ambiguous_canonical_tie_are_rejected() -> None:
    with pytest.raises(DuplicateEventError):
        AsOfBook(observations=(market("m1", 2), market("m1", 3)))

    first = market("m1", 2, revision=0, ingest=4)
    second = market("m1", 2, revision=1, ingest=4)
    with pytest.raises(AmbiguousEventError):
        AsOfBook(observations=(first, second))


def test_revision_selection_is_point_in_time_and_lower_revision_cannot_overwrite() -> None:
    book = AsOfBook(observations=(market("m1", 5, revision=1, value=9.0),))
    revised = book.with_observation(market("m1", 2, value=1.0))

    early = revised.materialize(4)
    late = revised.materialize(6)
    assert early.observation("m1").fields["value"] == 1.0
    assert late.observation("m1").fields["value"] == 9.0

    assert revised.materialize(8).observation("m1").fields["value"] == 9.0


def test_prefix_snapshot_does_not_mutate_when_future_event_is_appended() -> None:
    book = AsOfBook(observations=(market("m1", 2),))
    prefix = book.materialize(4)
    extended = book.with_observation(market("m2", 8, value=7.0))
    assert prefix == extended.materialize(4)
    with pytest.raises(UnavailableEventError):
        extended.materialize(4, required_record_ids=("m2",))


def test_equal_time_order_uses_precedence_then_ingest_then_record_id() -> None:
    events = (
        label("l1", "t", 5, ingest=99),
        market("m1", 5, ingest=99),
        UniverseMembership("ES", 0, 10, 0, 5, 0, ingest_sequence=99),
    )
    ordered = canonical_event_order(events)
    assert isinstance(ordered[0], UniverseMembership)
    assert ordered[0].record_id
    assert UNIVERSE_PRECEDENCE < 20
    assert [type(event).__name__ for event in ordered] == [
        "UniverseMembership",
        "MarketObservation",
        "LabelObservation",
    ]


def test_universe_membership_is_available_revision_safe_and_half_open() -> None:
    first = UniverseMembership("ES", 0, 10, 0, 2, 0)
    correction = UniverseMembership("ES", 0, 20, 0, 5, 1)
    book = AsOfBook(universe=(first, correction))
    assert book.materialize(4).is_member("ES", at=9)
    assert book.materialize(6).is_member("ES", at=19)
    assert book.materialize(6).is_member("ES", at=10)
    assert not book.materialize(6).is_member("ES", at=20)

    overlapping = UniverseMembership("ES", 8, 12, 1, 2, 0, ingest_sequence=1)
    with pytest.raises(AmbiguousEventError):
        AsOfBook(universe=(first, overlapping)).materialize(9)


def test_missing_labels_are_explicit_and_required_labels_cannot_be_silent() -> None:
    book = AsOfBook(labels=(label("l1", "t1", 5, status="missing", value=None),))
    snapshot = book.materialize(6)
    assert snapshot.labels[0].status == "missing"
    with pytest.raises(MissingLabelError):
        snapshot.label_for("t1")
    with pytest.raises(MissingLabelError):
        AsOfBook().materialize(6, required_target_ids=("t1",))


def test_label_availability_receipt_and_status_contract_is_causal() -> None:
    with pytest.raises(ValueError):
        LabelObservation("l", "t", 3, 5, 4, "observed", 1.0, 0)
    with pytest.raises(ValueError):
        LabelObservation("l", "t", 3, 5, 6, "observed", None, 0)
    with pytest.raises(ValueError):
        LabelObservation("l", "t", 3, 5, 6, "missing", 1.0, 0)


def test_future_records_are_excluded_but_explicitly_required_records_fail_closed() -> None:
    book = AsOfBook(observations=(market("future", 10),))
    assert book.materialize(9).observations == ()
    with pytest.raises(UnavailableEventError):
        book.materialize(9, required_record_ids=("future",))


def _panel_market(
    record_id: str,
    instrument_id: str,
    available: int,
    value: float,
    *,
    revision: int = 0,
) -> MarketObservation:
    return MarketObservation(
        record_id=record_id,
        instrument_id=instrument_id,
        event_time=available - 1,
        available_time=available,
        source_revision=revision,
        ingest_sequence=0,
        fields={"value": value, "other": value + 10.0},
    )


def _panel_book() -> AsOfBook:
    return AsOfBook(
        observations=(
            _panel_market("m-b", "B", 2, 2.0),
            _panel_market("m-a", "A", 2, 1.0),
        ),
        universe=(
            UniverseMembership("A", 0, 10, 0, 1, 0),
            UniverseMembership("B", 0, 10, 0, 1, 0),
        ),
    )


def test_point_in_time_panel_canonicalizes_asset_order_and_binds_universe() -> None:
    snapshot = _panel_book().materialize(3)
    first = PointInTimePanel.from_snapshot(
        snapshot,
        ("m-b", "m-a"),
        ("value",),
    )
    second = PointInTimePanel.from_snapshot(
        snapshot,
        ("m-a", "m-b"),
        ("value",),
    )

    assert first == second
    assert first.instrument_ids == ("A", "B")
    assert first.record_ids == ("m-a", "m-b")
    assert first.values == ((1.0,), (2.0,))
    assert len(first.universe_digest) == 64
    assert len(first.digest) == 64
    assert first.to_dict()["digest"] == first.digest


def test_point_in_time_panel_rejects_duplicate_assets_missing_membership_and_fields() -> None:
    duplicate_asset_book = AsOfBook(
        observations=(
            _panel_market("m-a1", "A", 2, 1.0),
            _panel_market("m-a2", "A", 2, 2.0),
        ),
        universe=(UniverseMembership("A", 0, 10, 0, 1, 0),),
    )
    with pytest.raises(AmbiguousEventError, match="multiple panel records"):
        PointInTimePanel.from_snapshot(
            duplicate_asset_book.materialize(3),
            ("m-a1", "m-a2"),
            ("value",),
        )

    nonmember = AsOfBook(observations=(_panel_market("m-c", "C", 2, 3.0),))
    with pytest.raises(UnavailableEventError, match="not in the point-in-time universe"):
        PointInTimePanel.from_snapshot(
            nonmember.materialize(3),
            ("m-c",),
            ("value",),
        )

    with pytest.raises(UnavailableEventError, match="missing field"):
        PointInTimePanel.from_snapshot(
            _panel_book().materialize(3),
            ("m-a",),
            ("missing",),
        )


def test_future_panel_events_and_membership_revisions_cannot_change_prefix_panel() -> None:
    book = _panel_book()
    prefix = PointInTimePanel.from_snapshot(
        book.materialize(3),
        ("m-a",),
        ("value",),
    )
    extended = book.with_observation(_panel_market("m-c", "C", 8, 3.0)).with_universe(
        UniverseMembership("C", 7, 12, 7, 8, 0)
    )
    assert prefix == PointInTimePanel.from_snapshot(
        extended.materialize(3),
        ("m-a",),
        ("value",),
    )
    with pytest.raises(UnavailableEventError):
        PointInTimePanel.from_snapshot(
            extended.materialize(3),
            ("m-c",),
            ("value",),
        )
