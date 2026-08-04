"""Canonical, digest-verified handoff for point-in-time study sources.

Vendor adapters should translate native records into the strict artifacts used
by :mod:`topology_gate.study`.  This module provides the next boundary: one
canonical JSON envelope that carries those artifacts and the source-policy
metadata needed to audit them later.  It does not certify a vendor's claims;
it makes the supplied claims and bytes reproducible and tamper-evident.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .asof import AsOfBook, TimePoint
from .economic import EconomicEvidence
from .manifest import RunManifest, StudyManifest
from .study import (
    StudyInputAudit,
    StudyInputBundle,
    StudyTimeline,
    _decode_time,
    _encode_time,
)

SOURCE_PROVENANCE_SCHEMA = "topology_gate.study_source_provenance"
SOURCE_PROVENANCE_VERSION = 1
SOURCE_PACKAGE_SCHEMA = "topology_gate.study_source_package"
SOURCE_PACKAGE_VERSION = 1


class StudySourcePackageError(ValueError):
    """Raised when a canonical study source package is invalid or tampered."""


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise StudySourcePackageError("source package identity is not JSON-safe") from exc


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StudySourcePackageError(f"{name} must be a non-empty string")
    return value


def _stored_digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise StudySourcePackageError(f"{name} must be a 64-character hexadecimal value")
    if any(character not in "0123456789abcdefABCDEF" for character in value):
        raise StudySourcePackageError(f"{name} must be hexadecimal")
    return value.lower()


@dataclass(frozen=True, slots=True)
class StudySourceProvenance:
    """Source-policy metadata carried beside normalized study artifacts.

    These fields describe the adapter's source contract.  They are not an
    independent certification that the vendor actually supplied point-in-time
    data; the final report must still state what was and was not audited.
    """

    provider_id: str
    dataset_id: str
    vintage_id: str
    as_of_rule: str
    revision_rule: str
    universe_rule: str
    delisting_rule: str
    retrieved_at: TimePoint
    schema: str = SOURCE_PROVENANCE_SCHEMA
    version: int = SOURCE_PROVENANCE_VERSION

    def __post_init__(self) -> None:
        for name in (
            "provider_id",
            "dataset_id",
            "vintage_id",
            "as_of_rule",
            "revision_rule",
            "universe_rule",
            "delisting_rule",
        ):
            _text(getattr(self, name), name)
        if self.schema != SOURCE_PROVENANCE_SCHEMA:
            raise StudySourcePackageError(
                f"schema must be exactly {SOURCE_PROVENANCE_SCHEMA!r}"
            )
        if type(self.version) is not int or self.version != SOURCE_PROVENANCE_VERSION:
            raise StudySourcePackageError(
                f"version must be exactly {SOURCE_PROVENANCE_VERSION}"
            )
        _encode_time(self.retrieved_at, "retrieved_at")

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "version": self.version,
            "provider_id": self.provider_id,
            "dataset_id": self.dataset_id,
            "vintage_id": self.vintage_id,
            "as_of_rule": self.as_of_rule,
            "revision_rule": self.revision_rule,
            "universe_rule": self.universe_rule,
            "delisting_rule": self.delisting_rule,
            "retrieved_at": _encode_time(self.retrieved_at, "retrieved_at"),
        }

    @property
    def digest(self) -> str:
        return _digest(self._payload())

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "digest": self.digest}

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> "StudySourceProvenance":
        if not isinstance(state, Mapping):
            raise StudySourcePackageError("source provenance must be a mapping")
        expected = {
            "schema",
            "version",
            "provider_id",
            "dataset_id",
            "vintage_id",
            "as_of_rule",
            "revision_rule",
            "universe_rule",
            "delisting_rule",
            "retrieved_at",
            "digest",
        }
        if set(state) != expected:
            raise StudySourcePackageError(
                "source provenance contains unknown or missing fields"
            )
        try:
            candidate = cls(
                provider_id=state["provider_id"],
                dataset_id=state["dataset_id"],
                vintage_id=state["vintage_id"],
                as_of_rule=state["as_of_rule"],
                revision_rule=state["revision_rule"],
                universe_rule=state["universe_rule"],
                delisting_rule=state["delisting_rule"],
                retrieved_at=_decode_time(state["retrieved_at"], "retrieved_at"),
                schema=state["schema"],
                version=state["version"],
            )
        except (TypeError, ValueError) as exc:
            raise StudySourcePackageError("source provenance is invalid") from exc
        stored = _stored_digest(state["digest"], "source provenance digest")
        if stored != candidate.digest:
            raise StudySourcePackageError(
                "source provenance digest does not match its content"
            )
        return candidate


@dataclass(frozen=True, slots=True)
class StudySourcePackage:
    """A canonical source envelope ready for strict study preflight."""

    provenance: StudySourceProvenance
    bundle: StudyInputBundle
    schema: str = SOURCE_PACKAGE_SCHEMA
    version: int = SOURCE_PACKAGE_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.provenance, StudySourceProvenance):
            raise StudySourcePackageError(
                "provenance must be a StudySourceProvenance"
            )
        if not isinstance(self.bundle, StudyInputBundle):
            raise StudySourcePackageError("bundle must be a StudyInputBundle")
        if self.schema != SOURCE_PACKAGE_SCHEMA:
            raise StudySourcePackageError(
                f"schema must be exactly {SOURCE_PACKAGE_SCHEMA!r}"
            )
        if type(self.version) is not int or self.version != SOURCE_PACKAGE_VERSION:
            raise StudySourcePackageError(
                f"version must be exactly {SOURCE_PACKAGE_VERSION}"
            )

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "version": self.version,
            "provenance": self.provenance.to_dict(),
            "run_manifest": self.bundle.run_manifest.to_dict(),
            "study_manifest": self.bundle.study_manifest.to_dict(),
            "timeline": self.bundle.timeline.to_dict(),
            "as_of_book": self.bundle.as_of_book.to_dict(),
            "economic_evidence": (
                None
                if self.bundle.economic_evidence is None
                else self.bundle.economic_evidence.to_dict()
            ),
            "economic_cutoff": (
                None
                if self.bundle.economic_cutoff is None
                else _encode_time(self.bundle.economic_cutoff, "economic_cutoff")
            ),
            "bundle_digest": self.bundle.digest,
        }

    @property
    def digest(self) -> str:
        return _digest(self._payload())

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "digest": self.digest}

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> "StudySourcePackage":
        if not isinstance(state, Mapping):
            raise StudySourcePackageError("study source package must be a mapping")
        expected = {
            "schema",
            "version",
            "provenance",
            "run_manifest",
            "study_manifest",
            "timeline",
            "as_of_book",
            "economic_evidence",
            "economic_cutoff",
            "bundle_digest",
            "digest",
        }
        if set(state) != expected:
            raise StudySourcePackageError(
                "study source package contains unknown or missing fields"
            )
        if state.get("schema") != SOURCE_PACKAGE_SCHEMA:
            raise StudySourcePackageError("unsupported study source package schema")
        if (
            type(state.get("version")) is not int
            or state.get("version") != SOURCE_PACKAGE_VERSION
        ):
            raise StudySourcePackageError("unsupported study source package version")
        try:
            provenance = StudySourceProvenance.from_dict(state["provenance"])
            run_manifest = RunManifest.from_dict(state["run_manifest"])
            study_manifest = StudyManifest.from_dict(state["study_manifest"])
            timeline = StudyTimeline.from_dict(state["timeline"])
            as_of_book = AsOfBook.from_dict(state["as_of_book"], require_digest=True)
            economic_state = state["economic_evidence"]
            economic_evidence = (
                None
                if economic_state is None
                else EconomicEvidence.from_dict(economic_state)
            )
            cutoff_state = state["economic_cutoff"]
            economic_cutoff = (
                None
                if cutoff_state is None
                else _decode_time(cutoff_state, "economic_cutoff")
            )
            bundle = StudyInputBundle(
                run_manifest=run_manifest,
                study_manifest=study_manifest,
                timeline=timeline,
                as_of_book=as_of_book,
                economic_evidence=economic_evidence,
                economic_cutoff=economic_cutoff,
            )
            candidate = cls(provenance=provenance, bundle=bundle)
        except (TypeError, ValueError) as exc:
            if isinstance(exc, StudySourcePackageError):
                raise
            raise StudySourcePackageError(
                f"study source package is invalid: {exc}"
            ) from exc

        stored_bundle = _stored_digest(state["bundle_digest"], "bundle digest")
        if stored_bundle != candidate.bundle.digest:
            raise StudySourcePackageError(
                "study source package bundle digest does not match its content"
            )
        stored_package = _stored_digest(state["digest"], "study source package digest")
        if stored_package != candidate.digest:
            raise StudySourcePackageError(
                "study source package digest does not match its content"
            )
        return candidate

    @classmethod
    def from_json(cls, payload: str) -> "StudySourcePackage":
        if not isinstance(payload, str) or not payload.strip():
            raise TypeError("study source package JSON must be a non-empty string")
        try:
            state = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise StudySourcePackageError("study source package JSON is invalid") from exc
        return cls.from_dict(state)

    def audit(self, phase: str, **kwargs: Any) -> StudyInputAudit:
        """Run the existing strict preflight against the restored bundle."""

        return self.bundle.audit(phase, **kwargs)


__all__ = [
    "SOURCE_PACKAGE_SCHEMA",
    "SOURCE_PACKAGE_VERSION",
    "SOURCE_PROVENANCE_SCHEMA",
    "SOURCE_PROVENANCE_VERSION",
    "StudySourcePackage",
    "StudySourcePackageError",
    "StudySourceProvenance",
]
