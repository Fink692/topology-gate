"""Focused contract tests for the immutable run manifest."""

from __future__ import annotations

import dataclasses
import json
import math

import pytest

from topology_gate.manifest import (
    MANIFEST_SCHEMA,
    MANIFEST_VERSION,
    ManifestValidationError,
    RunManifest,
    RunSpec,
    StudyManifest,
    StudySpec,
    StudyWindow,
)


def _spec(**overrides: object) -> RunSpec:
    values: dict[str, object] = {
        "run_id": "run-001",
        "input_vintage_id": {"source": "bars", "as_of": "2026-08-04T12:00:00Z"},
        "universe_id": {"name": "liquid-cross-asset", "revision": 3},
        "config_id": {"model": "ridge", "digest": "cfg-abc"},
        "backend_id": {"topology": "persistent-v1", "numeric": "numpy-2.2.6"},
        "dependency_id": {"lock": "deps-xyz", "python": "3.12"},
        "seed_id": {"algorithm": "pcg64", "value": 17},
        "thread_id": {"worker": "research-0", "count": 1},
    }
    values.update(overrides)
    return RunSpec(**values)


def _study(**overrides: object) -> StudySpec:
    values: dict[str, object] = {
        "run_spec": _spec(),
        "feature_schema_id": "features:v1",
        "label_spec_id": "labels:h1",
        "economic_spec_id": "costs:v1",
        "calibration_window": StudyWindow("calibration", 0, 100),
        "tuning_window": StudyWindow("tuning", 105, 200),
        "validation_window": StudyWindow("validation", 205, 300),
        "holdout_window": StudyWindow("holdout", 305, 400),
        "embargo_steps": 5,
    }
    values.update(overrides)
    return StudySpec(**values)


def test_manifest_is_immutable_and_has_explicit_contract() -> None:
    manifest = RunManifest(_spec(), metadata={"purpose": "walk-forward", "tags": ["golden"]})

    assert dataclasses.is_dataclass(manifest)
    assert manifest.to_dict()["schema"] == MANIFEST_SCHEMA
    assert manifest.to_dict()["version"] == MANIFEST_VERSION
    assert manifest.to_dict()["spec"]["run_id"] == "run-001"
    with pytest.raises(dataclasses.FrozenInstanceError):
        manifest.schema = "other"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        manifest.spec.run_id = "other"  # type: ignore[misc]


def test_nested_inputs_are_frozen_and_serialized_as_fresh_json_values() -> None:
    metadata = {"nested": {"values": [1, 2]}}
    manifest = RunManifest(_spec(), metadata=metadata)
    metadata["nested"]["values"].append(3)  # type: ignore[index]

    first = manifest.to_dict()
    first["metadata"]["nested"]["values"].append(99)  # type: ignore[index]

    assert manifest.to_dict()["metadata"] == {"nested": {"values": [1, 2]}}
    assert json.loads(manifest.to_json()) == manifest.to_dict()


def test_canonical_json_and_digest_are_order_independent() -> None:
    left = RunManifest(
        _spec(universe_id={"b": 2, "a": 1}),
        metadata={"z": "last", "a": ["first", {"b": True, "a": 0}]},
    )
    right = RunManifest(
        _spec(universe_id={"a": 1, "b": 2}),
        metadata={"a": ["first", {"a": 0, "b": True}], "z": "last"},
    )

    assert left.to_json() == right.to_json()
    assert left.digest == right.digest
    assert left.spec.digest == right.spec.digest
    assert len(left.digest) == 64
    assert left.to_json() == (
        '{"metadata":{"a":["first",{"a":0,"b":true}],"z":"last"},'
        '"schema":"topology-gate.run-manifest","spec":{'
        '"backend_id":{"numeric":"numpy-2.2.6","topology":"persistent-v1"},'
        '"config_id":{"digest":"cfg-abc","model":"ridge"},'
        '"dependency_id":{"lock":"deps-xyz","python":"3.12"},'
        '"input_vintage_id":{"as_of":"2026-08-04T12:00:00Z","source":"bars"},'
        '"run_id":"run-001","seed_id":{"algorithm":"pcg64","value":17},'
        '"thread_id":{"count":1,"worker":"research-0"},'
        '"universe_id":{"a":1,"b":2}},"version":1}'
    )


def test_manifests_round_trip_through_strict_json_restore() -> None:
    run = RunManifest(_spec(), metadata={"purpose": "walk-forward"})
    restored_run = RunManifest.from_json(run.to_json())
    assert restored_run == run
    assert restored_run.digest == run.digest

    sealed = StudyManifest(_study(), metadata={"purpose": "pre-registered"})
    restored_sealed = StudyManifest.from_json(sealed.to_json())
    assert restored_sealed == sealed
    assert restored_sealed.digest == sealed.digest

    opened = sealed.open_holdout("release-2026-08-04")
    assert StudyManifest.from_dict(opened.to_dict()) == opened


def test_manifest_restore_rejects_unknown_fields_and_invalid_json() -> None:
    run_payload = RunManifest(_spec()).to_dict()
    run_payload["unmodeled_field"] = "reject"
    with pytest.raises(ManifestValidationError, match="unknown or missing"):
        RunManifest.from_dict(run_payload)

    study_payload = StudyManifest(_study()).to_dict()
    study_payload["unmodeled_field"] = "reject"
    with pytest.raises(ManifestValidationError, match="unknown or missing"):
        StudyManifest.from_dict(study_payload)

    with pytest.raises(ManifestValidationError, match="JSON is invalid"):
        RunManifest.from_json("{not-json}")
    with pytest.raises(ManifestValidationError, match="must contain an object"):
        StudyManifest.from_json("[]")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("run_id", None),
        ("run_id", "   "),
        ("input_vintage_id", {}),
        ("universe_id", []),
        ("config_id", {"digest": ""}),
        ("backend_id", {"version": math.nan}),
        ("dependency_id", {"version": math.inf}),
        ("seed_id", {"value": object()}),
        ("thread_id", {"worker": {"bad": object()}}),
        ("backend_id", {1: "not-a-string-key"}),
    ],
)
def test_invalid_or_missing_identity_fails_closed(field: str, value: object) -> None:
    with pytest.raises(ManifestValidationError):
        _spec(**{field: value})


@pytest.mark.parametrize(
    ("schema", "version"),
    [("wrong.schema", MANIFEST_VERSION), (MANIFEST_SCHEMA, 2)],
)
def test_manifest_rejects_unknown_schema_or_version(schema: str, version: int) -> None:
    with pytest.raises(ManifestValidationError):
        RunManifest(_spec(), schema=schema, version=version)


def test_manifest_rejects_boolean_version() -> None:
    with pytest.raises(ManifestValidationError):
        RunManifest(_spec(), version=True)  # type: ignore[arg-type]


def test_manifest_rejects_missing_required_values() -> None:
    with pytest.raises(ManifestValidationError):
        RunSpec()
    with pytest.raises(ManifestValidationError):
        RunManifest()


@pytest.mark.parametrize(
    "metadata",
    [
        None,
        " ",
        {"nested": float("-inf")},
        {"nested": ("tuples are not JSON fields",)},
        {"nested": {"": "blank key"}},
        {"nested": {"bad": object()}},
    ],
)
def test_metadata_rejects_missing_blank_nonfinite_and_non_json_values(
    metadata: object,
) -> None:
    with pytest.raises(ManifestValidationError):
        RunManifest(_spec(), metadata=metadata)


def test_study_manifest_freezes_ordered_splits_and_embargo_identity() -> None:
    study = _study()
    manifest = StudyManifest(study, metadata={"purpose": "pre-registered"})

    assert study.digest
    assert manifest.holdout_is_sealed
    manifest.require_holdout_sealed()
    assert manifest.to_dict()["spec"]["holdout_window"] == {
        "name": "holdout",
        "start": 305,
        "end": 400,
    }
    assert manifest.digest == StudyManifest(
        _study(), metadata={"purpose": "pre-registered"}
    ).digest


def test_study_manifest_rejects_overlap_or_insufficient_embargo() -> None:
    with pytest.raises(ManifestValidationError, match="embargo"):
        _study(
            tuning_window=StudyWindow("tuning", 103, 200),
        )
    with pytest.raises(ManifestValidationError, match="unique"):
        _study(
            holdout_window=StudyWindow("validation", 305, 400),
        )
    with pytest.raises(ManifestValidationError, match="end"):
        StudyWindow("bad", 3, 3)


def test_study_manifest_holdout_release_is_explicit_and_irreversible() -> None:
    sealed = StudyManifest(_study())
    with pytest.raises(ManifestValidationError, match="release ID"):
        sealed.open_holdout(" ")

    opened = sealed.open_holdout("release-2026-08-04")
    assert not opened.holdout_is_sealed
    assert opened.holdout_release_id == "release-2026-08-04"
    assert opened.digest != sealed.digest
    with pytest.raises(ManifestValidationError, match="already opened"):
        opened.require_holdout_sealed()
    with pytest.raises(ManifestValidationError, match="already opened"):
        opened.open_holdout("second-release")


def test_study_manifest_phase_checks_reject_sealed_holdout_and_out_of_window_indices() -> None:
    sealed = StudyManifest(_study())
    assert sealed.spec.window_for_phase("tuning").name == "tuning"
    sealed.assert_indices_allowed((105, 150, 199), "tuning")
    with pytest.raises(ManifestValidationError, match="sealed study holdout"):
        sealed.assert_index_allowed(305, "holdout")
    with pytest.raises(ManifestValidationError, match="outside"):
        sealed.assert_index_allowed(100, "tuning")
    with pytest.raises(ManifestValidationError, match="strictly increasing"):
        sealed.assert_indices_allowed((105, 105), "tuning")
    with pytest.raises(ManifestValidationError, match="non-negative integer"):
        sealed.assert_indices_allowed(("105",), "tuning")  # type: ignore[arg-type]
    with pytest.raises(ManifestValidationError, match="non-negative integer"):
        sealed.assert_indices_allowed((True,), "tuning")  # type: ignore[arg-type]
    with pytest.raises(ManifestValidationError, match="calibration, tuning"):
        sealed.spec.window_for_phase("test")

    opened = sealed.open_holdout("release")
    opened.assert_indices_allowed((305, 350, 399), "holdout")


def test_study_manifest_rejects_unknown_status_or_schema_version() -> None:
    with pytest.raises(ManifestValidationError, match="holdout_status"):
        StudyManifest(_study(), holdout_status="peeked")
    with pytest.raises(ManifestValidationError, match="schema"):
        StudyManifest(_study(), schema="wrong.study")
    with pytest.raises(ManifestValidationError, match="version"):
        StudyManifest(_study(), version=2)
