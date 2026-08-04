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
SOURCE_ARTIFACT_SCHEMA = "topology_gate.study_source_artifact"
SOURCE_ARTIFACT_VERSION = 1
SOURCE_AUDIT_SCHEMA = "topology_gate.study_source_audit"
SOURCE_AUDIT_VERSION = 1
REQUIRED_MARKET_ARTIFACT_ROLES = (
    "delistings",
    "execution-costs",
    "labels",
    "market-observations",
    "realized-returns",
    "universe-membership",
)


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
class StudySourceArtifact:
    """Fingerprint for one immutable raw input artifact."""

    artifact_id: str
    role: str
    sha256: str
    byte_size: int
    record_count: int
    schema: str = SOURCE_ARTIFACT_SCHEMA
    version: int = SOURCE_ARTIFACT_VERSION

    def __post_init__(self) -> None:
        self_id = _text(self.artifact_id, "artifact_id")
        role = _text(self.role, "role")
        object.__setattr__(self, "artifact_id", self_id)
        object.__setattr__(self, "role", role)
        object.__setattr__(
            self,
            "sha256",
            _stored_digest(self.sha256, "artifact sha256"),
        )
        for name in ("byte_size", "record_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise StudySourcePackageError(
                    f"{name} must be a non-negative integer"
                )
        if self.schema != SOURCE_ARTIFACT_SCHEMA:
            raise StudySourcePackageError(
                f"schema must be exactly {SOURCE_ARTIFACT_SCHEMA!r}"
            )
        if type(self.version) is not int or self.version != SOURCE_ARTIFACT_VERSION:
            raise StudySourcePackageError(
                f"version must be exactly {SOURCE_ARTIFACT_VERSION}"
            )

    @classmethod
    def from_bytes(
        cls,
        artifact_id: str,
        role: str,
        payload: bytes,
        record_count: int,
    ) -> "StudySourceArtifact":
        if not isinstance(payload, bytes):
            raise TypeError("source artifact payload must be bytes")
        return cls(
            artifact_id=artifact_id,
            role=role,
            sha256=hashlib.sha256(payload).hexdigest(),
            byte_size=len(payload),
            record_count=record_count,
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "version": self.version,
            "artifact_id": self.artifact_id,
            "role": self.role,
            "sha256": self.sha256,
            "byte_size": self.byte_size,
            "record_count": self.record_count,
        }

    @property
    def digest(self) -> str:
        return _digest(self._payload())

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "digest": self.digest}

    def verify_bytes(self, payload: bytes) -> None:
        if not isinstance(payload, bytes):
            raise TypeError("source artifact payload must be bytes")
        if len(payload) != self.byte_size:
            raise StudySourcePackageError(
                f"source artifact {self.artifact_id!r} byte size does not match"
            )
        actual = hashlib.sha256(payload).hexdigest()
        if actual != self.sha256:
            raise StudySourcePackageError(
                f"source artifact {self.artifact_id!r} sha256 does not match"
            )

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> "StudySourceArtifact":
        if not isinstance(state, Mapping):
            raise StudySourcePackageError("source artifact must be a mapping")
        expected = {
            "schema",
            "version",
            "artifact_id",
            "role",
            "sha256",
            "byte_size",
            "record_count",
            "digest",
        }
        if set(state) != expected:
            raise StudySourcePackageError(
                "source artifact contains unknown or missing fields"
            )
        try:
            candidate = cls(
                artifact_id=state["artifact_id"],
                role=state["role"],
                sha256=state["sha256"],
                byte_size=state["byte_size"],
                record_count=state["record_count"],
                schema=state["schema"],
                version=state["version"],
            )
        except (TypeError, ValueError) as exc:
            raise StudySourcePackageError("source artifact is invalid") from exc
        stored = _stored_digest(state["digest"], "source artifact digest")
        if stored != candidate.digest:
            raise StudySourcePackageError(
                "source artifact digest does not match its content"
            )
        return candidate


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
    license_id: str
    release_id: str
    adapter_revision: str
    as_of_rule: str
    revision_rule: str
    universe_rule: str
    delisting_rule: str
    source_artifacts: tuple[StudySourceArtifact, ...]
    retrieved_at: TimePoint
    schema: str = SOURCE_PROVENANCE_SCHEMA
    version: int = SOURCE_PROVENANCE_VERSION

    def __post_init__(self) -> None:
        for name in (
            "provider_id",
            "dataset_id",
            "vintage_id",
            "license_id",
            "release_id",
            "adapter_revision",
            "as_of_rule",
            "revision_rule",
            "universe_rule",
            "delisting_rule",
        ):
            _text(getattr(self, name), name)
        try:
            artifacts = tuple(self.source_artifacts)
        except TypeError as exc:
            raise StudySourcePackageError(
                "source_artifacts must be a sequence"
            ) from exc
        if not artifacts or not all(
            isinstance(item, StudySourceArtifact) for item in artifacts
        ):
            raise StudySourcePackageError(
                "source_artifacts must contain StudySourceArtifact values"
            )
        if len({item.artifact_id for item in artifacts}) != len(artifacts):
            raise StudySourcePackageError("source_artifacts must have unique IDs")
        object.__setattr__(self, "source_artifacts", artifacts)
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
            "license_id": self.license_id,
            "release_id": self.release_id,
            "adapter_revision": self.adapter_revision,
            "as_of_rule": self.as_of_rule,
            "revision_rule": self.revision_rule,
            "universe_rule": self.universe_rule,
            "delisting_rule": self.delisting_rule,
            "source_artifacts": [
                item.to_dict() for item in self.source_artifacts
            ],
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
            "license_id",
            "release_id",
            "adapter_revision",
            "as_of_rule",
            "revision_rule",
            "universe_rule",
            "delisting_rule",
            "source_artifacts",
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
                license_id=state["license_id"],
                release_id=state["release_id"],
                adapter_revision=state["adapter_revision"],
                as_of_rule=state["as_of_rule"],
                revision_rule=state["revision_rule"],
                universe_rule=state["universe_rule"],
                delisting_rule=state["delisting_rule"],
                source_artifacts=tuple(
                    StudySourceArtifact.from_dict(item)
                    for item in state["source_artifacts"]
                ),
                retrieved_at=_decode_time(state["retrieved_at"], "retrieved_at"),
                schema=state["schema"],
                version=state["version"],
            )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, StudySourcePackageError):
                raise
            raise StudySourcePackageError(
                f"source provenance is invalid: {exc}"
            ) from exc
        stored = _stored_digest(state["digest"], "source provenance digest")
        if stored != candidate.digest:
            raise StudySourcePackageError(
                "source provenance digest does not match its content"
            )
        return candidate


@dataclass(frozen=True, slots=True)
class StudySourceAudit:
    """Receipt proving a strict market-source audit was actually performed.

    The receipt is only constructible after the package has verified every
    declared raw artifact and the normalized study bundle has passed the
    complete-universe, observed-economic-record, and capacity checks.
    """

    package_digest: str
    provenance_digest: str
    phase: str
    input_audit: StudyInputAudit
    verified_artifact_ids: tuple[str, ...]
    required_artifact_roles: tuple[str, ...]
    capacity_evidence_required: bool
    source_artifacts_verified: bool = True
    schema: str = SOURCE_AUDIT_SCHEMA
    version: int = SOURCE_AUDIT_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "package_digest",
            _stored_digest(self.package_digest, "source package digest"),
        )
        object.__setattr__(
            self,
            "provenance_digest",
            _stored_digest(self.provenance_digest, "source provenance digest"),
        )
        _text(self.phase, "source audit phase")
        if not isinstance(self.input_audit, StudyInputAudit):
            raise StudySourcePackageError("input_audit must be StudyInputAudit")
        for name in ("verified_artifact_ids", "required_artifact_roles"):
            value = getattr(self, name)
            if isinstance(value, (str, bytes, bytearray)):
                raise StudySourcePackageError(f"{name} must be a sequence")
            try:
                normalized = tuple(value)
            except TypeError as exc:
                raise StudySourcePackageError(f"{name} must be a sequence") from exc
            if not normalized or not all(
                isinstance(item, str) and item.strip() for item in normalized
            ):
                raise StudySourcePackageError(
                    f"{name} must contain non-empty strings"
                )
            if tuple(sorted(normalized)) != normalized:
                raise StudySourcePackageError(f"{name} must be sorted")
            if len(set(normalized)) != len(normalized):
                raise StudySourcePackageError(f"{name} must be unique")
            object.__setattr__(self, name, normalized)
        if not isinstance(self.capacity_evidence_required, bool):
            raise StudySourcePackageError(
                "capacity_evidence_required must be boolean"
            )
        if not isinstance(self.source_artifacts_verified, bool):
            raise StudySourcePackageError(
                "source_artifacts_verified must be boolean"
            )
        if not self.source_artifacts_verified:
            raise StudySourcePackageError(
                "source audit receipts require verified source artifacts"
            )
        if self.capacity_evidence_required and not self.input_audit.capacity_evidence_complete:
            raise StudySourcePackageError(
                "source audit receipt is missing capacity evidence"
            )
        if not set(REQUIRED_MARKET_ARTIFACT_ROLES).issubset(
            set(self.required_artifact_roles)
        ):
            raise StudySourcePackageError(
                "source audit receipt is missing required market artifact roles"
            )
        if self.schema != SOURCE_AUDIT_SCHEMA:
            raise StudySourcePackageError(
                f"schema must be exactly {SOURCE_AUDIT_SCHEMA!r}"
            )
        if type(self.version) is not int or self.version != SOURCE_AUDIT_VERSION:
            raise StudySourcePackageError(
                f"version must be exactly {SOURCE_AUDIT_VERSION}"
            )

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "version": self.version,
            "package_digest": self.package_digest.lower(),
            "provenance_digest": self.provenance_digest.lower(),
            "phase": self.phase,
            "input_audit": self.input_audit.to_dict(),
            "verified_artifact_ids": list(self.verified_artifact_ids),
            "required_artifact_roles": list(self.required_artifact_roles),
            "capacity_evidence_required": self.capacity_evidence_required,
            "source_artifacts_verified": self.source_artifacts_verified,
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
    def from_dict(cls, state: Mapping[str, Any]) -> "StudySourceAudit":
        if not isinstance(state, Mapping):
            raise StudySourcePackageError("source audit must be a mapping")
        expected = {
            "schema",
            "version",
            "package_digest",
            "provenance_digest",
            "phase",
            "input_audit",
            "verified_artifact_ids",
            "required_artifact_roles",
            "capacity_evidence_required",
            "source_artifacts_verified",
            "digest",
        }
        if set(state) != expected:
            raise StudySourcePackageError(
                "source audit contains unknown or missing fields"
            )
        try:
            for name in ("verified_artifact_ids", "required_artifact_roles"):
                if isinstance(state[name], (str, bytes, bytearray)):
                    raise StudySourcePackageError(f"{name} must be a sequence")
            candidate = cls(
                package_digest=state["package_digest"],
                provenance_digest=state["provenance_digest"],
                phase=state["phase"],
                input_audit=StudyInputAudit.from_dict(state["input_audit"]),
                verified_artifact_ids=tuple(state["verified_artifact_ids"]),
                required_artifact_roles=tuple(state["required_artifact_roles"]),
                capacity_evidence_required=state["capacity_evidence_required"],
                source_artifacts_verified=state["source_artifacts_verified"],
                schema=state["schema"],
                version=state["version"],
            )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, StudySourcePackageError):
                raise
            raise StudySourcePackageError("source audit is invalid") from exc
        stored = _stored_digest(state["digest"], "source audit digest")
        if stored != candidate.digest:
            raise StudySourcePackageError(
                "source audit digest does not match its content"
            )
        return candidate

    @classmethod
    def from_json(cls, payload: str) -> "StudySourceAudit":
        if not isinstance(payload, str) or not payload.strip():
            raise TypeError("source audit JSON must be a non-empty string")
        try:
            state = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise StudySourcePackageError("source audit JSON is invalid") from exc
        return cls.from_dict(state)


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

    def audit_market(
        self,
        phase: str,
        raw_payloads: Mapping[str, bytes],
    ) -> StudySourceAudit:
        """Verify the complete source handoff before a market-data study.

        This stricter boundary binds raw-byte verification, provenance vintage,
        required source roles, complete point-in-time universe coverage,
        observed economic records, and per-target capacity evidence into one
        immutable receipt.  It intentionally rejects the minimal synthetic
        fixtures used by ordinary package round-trip tests.
        """

        self.verify_source_artifacts(raw_payloads)
        if (
            self.provenance.vintage_id
            != self.bundle.run_manifest.spec.input_vintage_id
        ):
            raise StudySourcePackageError(
                "source provenance vintage does not match the run manifest"
            )
        declared_roles = {artifact.role for artifact in self.provenance.source_artifacts}
        missing_roles = sorted(
            set(REQUIRED_MARKET_ARTIFACT_ROLES) - declared_roles
        )
        if missing_roles:
            raise StudySourcePackageError(
                "market source artifacts are missing required roles "
                f"{missing_roles!r}"
            )
        input_audit = self.bundle.audit(
            phase,
            require_complete_universe=True,
            require_economic_evidence=True,
            require_observed_economic_evidence=True,
            require_capacity_evidence=True,
        )
        return StudySourceAudit(
            package_digest=self.digest,
            provenance_digest=self.provenance.digest,
            phase=input_audit.phase,
            input_audit=input_audit,
            verified_artifact_ids=tuple(sorted(raw_payloads)),
            required_artifact_roles=tuple(REQUIRED_MARKET_ARTIFACT_ROLES),
            capacity_evidence_required=True,
        )

    def verify_source_artifact(self, artifact_id: str, payload: bytes) -> None:
        """Verify raw bytes against the package's declared artifact fingerprint."""

        artifact_name = _text(artifact_id, "artifact_id")
        for artifact in self.provenance.source_artifacts:
            if artifact.artifact_id == artifact_name:
                artifact.verify_bytes(payload)
                return
        raise StudySourcePackageError(
            f"source artifact {artifact_name!r} is not declared"
        )

    def verify_source_artifacts(self, payloads: Mapping[str, bytes]) -> None:
        """Verify every declared raw artifact and reject omissions or extras."""

        if not isinstance(payloads, Mapping):
            raise TypeError("source artifact payloads must be a mapping")
        expected = {artifact.artifact_id for artifact in self.provenance.source_artifacts}
        provided = set(payloads)
        if provided != expected:
            missing = sorted(expected - provided)
            unexpected = sorted(provided - expected, key=repr)
            raise StudySourcePackageError(
                "source artifact payload IDs do not match the package "
                f"(missing={missing!r}, unexpected={unexpected!r})"
            )
        for artifact in self.provenance.source_artifacts:
            artifact.verify_bytes(payloads[artifact.artifact_id])


__all__ = [
    "REQUIRED_MARKET_ARTIFACT_ROLES",
    "SOURCE_AUDIT_SCHEMA",
    "SOURCE_AUDIT_VERSION",
    "SOURCE_PACKAGE_SCHEMA",
    "SOURCE_PACKAGE_VERSION",
    "SOURCE_ARTIFACT_SCHEMA",
    "SOURCE_ARTIFACT_VERSION",
    "SOURCE_PROVENANCE_SCHEMA",
    "SOURCE_PROVENANCE_VERSION",
    "StudySourcePackage",
    "StudySourcePackageError",
    "StudySourceArtifact",
    "StudySourceAudit",
    "StudySourceProvenance",
]
