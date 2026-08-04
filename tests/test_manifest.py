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
