"""Independent finite Vietoris--Rips/persistent-Laplacian goldens."""

from __future__ import annotations

import math

import numpy as np
import pytest

from topology_gate.persistent import (
    PersistentLaplacianBackend,
    PersistentLaplacianConfig,
    PersistentResourceError,
    PersistentStatus,
    build_filtration,
    compute_persistence,
    compute_persistent_laplacian,
    persistent_laplacian_backend,
)


def _intervals(result, dimension: int) -> list[tuple[float, float | None]]:
    return [
        (interval.birth, interval.death)
        for interval in result.intervals
        if interval.homology_dimension == dimension
    ]


def test_collinear_h0_intervals_match_the_vr_golden() -> None:
    config = PersistentLaplacianConfig(max_vertices=8, max_simplices=100, q=0)
    filtration = build_filtration([[0.0], [1.0], [3.0]], config=config)
    intervals = compute_persistence(filtration, dimensions=(0, 1))
    h0 = [
        (interval.birth, interval.death)
        for interval in intervals
        if interval.homology_dimension == 0
    ]
    assert h0 == [(0.0, 1.0), (0.0, 2.0), (0.0, None)]


def test_equilateral_triangle_laplacian_is_hand_checkable() -> None:
    height = math.sqrt(3.0) / 2.0
    result = compute_persistent_laplacian(
        [[0.0, 0.0], [1.0, 0.0], [0.5, height]],
        config=PersistentLaplacianConfig(
            max_vertices=8,
            max_simplices=100,
            q=1,
            scale_s=1.0,
            scale_t=1.0,
            n_eigenvalues=8,
        ),
    )
    assert result.status is PersistentStatus.VALID
    assert _intervals(result, 1) == [(1.0, 1.0)]
    np.testing.assert_allclose(result.spectrum.eigenvalues, [3.0, 3.0, 3.0])
    assert result.spectrum.zero_multiplicity == 0
    assert result.spectrum.residual < 1.0e-10


def test_square_has_one_h1_bar_until_diagonal_triangles_arrive() -> None:
    result = compute_persistent_laplacian(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        config=PersistentLaplacianConfig(
            max_vertices=8,
            max_simplices=500,
            q=1,
            n_eigenvalues=8,
        ),
    )
    h1 = [interval for interval in result.intervals if interval.homology_dimension == 1]
    assert h1[0].birth == 1.0
    assert math.isclose(h1[0].death or 0.0, math.sqrt(2.0))
    assert result.spectrum.zero_multiplicity == 0


def test_permutation_digest_and_adapter_are_deterministic() -> None:
    points = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    config = PersistentLaplacianConfig(max_vertices=8, max_simplices=500, q=1)
    first = compute_persistent_laplacian(points, config=config)
    second = compute_persistent_laplacian([points[2], points[0], points[3], points[1]], config=config)
    assert first.to_dict() == second.to_dict()
    assert persistent_laplacian_backend(points, 3) == first.spectrum.eigenvalues[:3]


def test_configured_backend_returns_complete_evidence_with_stable_identity() -> None:
    config = PersistentLaplacianConfig(
        max_vertices=8,
        max_simplices=500,
        q=0,
        n_eigenvalues=3,
    )
    backend = PersistentLaplacianBackend(config)
    result = backend(
        [[0.0], [0.4], [1.0], [1.8]],
        n_eigenvalues=3,
    )

    assert result.status is PersistentStatus.VALID
    assert result.config_identity == config.identity
    assert len(result.spectrum.eigenvalues) == 3
    assert len(result.evidence_digest) == 64
    assert result.evidence_digest == result.evidence_digest
    assert backend.identity.startswith("topology_gate.persistent_backend:v1:")
    with pytest.raises(ValueError, match="requires n_eigenvalues=3"):
        backend([[0.0], [1.0], [2.0]], n_eigenvalues=2)


def test_exact_backend_fails_closed_on_resource_caps() -> None:
    with pytest.raises(PersistentResourceError, match="max_vertices"):
        build_filtration(
            [[float(index)] for index in range(5)],
            config=PersistentLaplacianConfig(max_vertices=4),
        )
    with pytest.raises(PersistentResourceError, match="max_simplices"):
        build_filtration(
            [[0.0], [1.0], [2.0], [3.0]],
            config=PersistentLaplacianConfig(max_vertices=8, max_simplices=3),
        )
