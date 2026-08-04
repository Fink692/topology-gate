"""Algebraic and determinism invariants for the exact persistent backend."""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import pytest

from topology_gate.persistent import (
    PersistentLaplacianConfig,
    PersistentResourceError,
    _oriented_boundary,
    build_filtration,
    compute_persistent_laplacian,
)
from topology_gate.topology import spectral_summary


def _square_points() -> list[list[float]]:
    return [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]


def test_oriented_boundary_composition_is_zero() -> None:
    filtration = build_filtration(
        _square_points(),
        config=PersistentLaplacianConfig(
            max_vertices=8,
            max_simplices=500,
            max_homology_dimension=1,
        ),
    )

    boundary_one = _oriented_boundary(
        filtration.simplices,
        source_dimension=1,
        target_dimension=0,
    )
    boundary_two = _oriented_boundary(
        filtration.simplices,
        source_dimension=2,
        target_dimension=1,
    )

    composition = boundary_one @ boundary_two
    np.testing.assert_array_equal(composition, np.zeros_like(composition))


def test_equal_scales_reduce_to_the_ordinary_q_laplacian_and_are_psd() -> None:
    scale = math.sqrt(2.0)
    config = PersistentLaplacianConfig(
        max_vertices=8,
        max_simplices=500,
        q=1,
        scale_s=scale,
        scale_t=scale,
        n_eigenvalues=8,
    )
    filtration = build_filtration(_square_points(), config=config)
    result = compute_persistent_laplacian(_square_points(), config=config)
    selected = [simplex for simplex in filtration.simplices if simplex.birth <= scale]

    boundary_one = _oriented_boundary(
        selected,
        source_dimension=1,
        target_dimension=0,
    )
    boundary_two = _oriented_boundary(
        selected,
        source_dimension=2,
        target_dimension=1,
    )
    ordinary = boundary_one.T @ boundary_one + boundary_two @ boundary_two.T
    expected = np.linalg.eigvalsh(ordinary)

    np.testing.assert_allclose(result.spectrum.eigenvalues, expected, atol=1.0e-10)
    assert result.spectrum.trace == pytest.approx(float(np.trace(ordinary)), abs=1.0e-10)
    assert np.all(expected >= -1.0e-12)
    assert result.spectrum.residual < 1.0e-10


def test_permutation_and_duplicate_coordinates_have_identical_results_and_digests() -> None:
    points = [[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    permuted = [points[2], points[0], points[3], points[1]]
    config = PersistentLaplacianConfig(
        max_vertices=8,
        max_simplices=500,
        q=1,
        scale_s=1.0,
        scale_t=math.sqrt(2.0),
        n_eigenvalues=8,
    )

    first = compute_persistent_laplacian(points, config=config)
    second = compute_persistent_laplacian(permuted, config=config)

    assert first.to_dict() == second.to_dict()
    assert first.filtration_digest == second.filtration_digest
    assert first.interval_digest == second.interval_digest
    digest = lambda result: hashlib.sha256(  # noqa: E731
        json.dumps(result.to_dict(), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert digest(first) == digest(second)


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (
            PersistentLaplacianConfig(
                max_vertices=8,
                max_homology_dimension=1,
                max_simplices=24,
            ),
            "max_simplices",
        ),
        (
            PersistentLaplacianConfig(
                max_vertices=8,
                max_homology_dimension=1,
                max_boundary_nonzeros=49,
            ),
            "max_boundary_nonzeros",
        ),
    ],
)
def test_resource_caps_fail_closed_without_partial_result(
    config: PersistentLaplacianConfig,
    message: str,
) -> None:
    points = [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.5, 0.5],
    ]
    sentinel = object()
    result: object = sentinel

    with pytest.raises(PersistentResourceError, match=message):
        result = compute_persistent_laplacian(points, config=config)

    assert result is sentinel


def test_exact_adapter_does_not_fake_missing_spectrum_entries() -> None:
    def undersized_backend(_cloud, _count):
        return [0.0]

    with pytest.raises(ValueError, match="fewer eigenvalues"):
        spectral_summary(
            _square_points(),
            n_eigenvalues=3,
            graph_neighbors=2,
            persistent_laplacian_backend=undersized_backend,
        )


def test_persistent_pair_scales_are_validated_before_building() -> None:
    with pytest.raises(ValueError, match="scale_s"):
        PersistentLaplacianConfig(scale_s=2.0, scale_t=1.0)
