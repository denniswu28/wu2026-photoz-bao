"""Distinct covariance entries identify their multipole and radial slots after projection."""

import numpy as np
import pytest
from pzbao.covariance import (
    legendre_ordering_permutation,
    project_raw_components,
    raw_component_projection,
)


def test_linear_permutation_is_a_bijection_and_identity_for_one_multipole():
    np.testing.assert_array_equal(
        legendre_ordering_permutation(radial_bin_count=7, multipole_count=1), np.arange(7)
    )
    np.testing.assert_array_equal(
        legendre_ordering_permutation(radial_bin_count=4, multipole_count=2),
        [0, 2, 4, 6, 1, 3, 5, 7],
    )
    with pytest.raises(ValueError, match="at least one radial bin"):
        legendre_ordering_permutation(radial_bin_count=0, multipole_count=2)


@pytest.mark.parametrize("multipoles", [[0], [0, 2], [0, 2, 4]])
def test_linear_every_projected_element_names_its_multipole_and_separation(multipoles):
    edges = np.arange(0.0, 204.0, 4.0)
    fit_edges = np.arange(48.0, 156.0, 4.0)
    amplitudes = np.array([1 + 1000 * pole + radial for radial in range(50) for pole in range(3)])
    native = 0.25 * np.outer(amplitudes, amplitudes)
    np.fill_diagonal(native, amplitudes**2)
    operator = raw_component_projection(
        raw_separation_edges_mpc_h=edges,
        raw_multipoles=[0, 2, 4],
        fit_separation_edges_mpc_h=fit_edges,
        fit_multipoles=multipoles,
    )
    result = project_raw_components(
        {key: native * scale for key, scale in (("c2", 1.0), ("c3", 2.0), ("c4", 3.0))}, operator
    )
    expected = np.array(
        [1 + 1000 * pole + radial for pole in range(len(multipoles)) for radial in range(12, 38)]
    )
    matrix = 0.25 * np.outer(expected, expected)
    np.fill_diagonal(matrix, expected**2)
    for key, scale in (("c2", 1.0), ("c3", 2.0), ("c4", 3.0)):
        np.testing.assert_allclose(result[key], matrix * scale, rtol=1e-14)
    assert result["c4"].shape == (26 * len(multipoles),) * 2
    assert not np.allclose(
        np.linalg.inv(result["c2"]), operator @ np.linalg.inv(native) @ operator.T
    )


def test_linear_full_permutation_preserves_eigenvalues_and_conditioning():
    rng = np.random.RandomState(7)
    basis = rng.normal(size=(8, 8))
    native = basis @ basis.T + np.eye(8)
    permutation = legendre_ordering_permutation(radial_bin_count=4, multipole_count=2)
    fitted = native[np.ix_(permutation, permutation)]
    np.testing.assert_allclose(np.linalg.eigvalsh(fitted), np.linalg.eigvalsh(native), atol=1e-12)
    assert np.linalg.cond(fitted) == pytest.approx(np.linalg.cond(native), rel=1e-12)
    np.linalg.cholesky(fitted)


@pytest.mark.parametrize(
    "fit_edges",
    [
        np.arange(50.0, 150.0, 4.0),
        np.arange(48.0, 156.0, 8.0),
        [196.0, 200.0, 204.0],
        [48.0, np.nan, 56.0],
    ],
)
def test_linear_covariance_window_refuses_off_grid_noncontiguous_or_nonfinite_edges(fit_edges):
    with pytest.raises(ValueError):
        raw_component_projection(
            raw_separation_edges_mpc_h=np.arange(0.0, 204.0, 4.0),
            raw_multipoles=[0, 2, 4],
            fit_separation_edges_mpc_h=fit_edges,
            fit_multipoles=[0],
        )


def test_skip_l_drops_the_highest_multipoles_not_the_lowest():
    """Phase B's ``skip_l = len(model) - len(observed)`` depends on which end is dropped.

    If RascalC dropped the *leading* multipoles instead, an LRG3 run would return the ell=2,4
    block labelled ell=0,2 at the correct dimension 44 -- and the dimension check, the ordering
    permutation and every consumer check would pass on a wrongly-labelled matrix. That is the
    same class of defect Phase A exists to prevent, one level up, so the direction is asserted
    against the library rather than taken from a docstring.
    """
    rascalc_utils = pytest.importorskip("RascalC.post_process.utils")
    n_radial, max_l, n_l = 10, 4, 3

    def indices(kept_positions):
        return np.array(
            [n_l * r + p for r in range(n_radial) for p in sorted(kept_positions)], dtype=int
        )

    for skip_l, keeps_lowest, keeps_highest in ((1, (0, 1), (1, 2)), (2, (0,), (2,))):
        library = rascalc_utils.cov_filter_legendre_1d(n_radial, max_l, 0, skip_l)
        np.testing.assert_array_equal(
            library, indices(keeps_lowest), err_msg=f"skip_l={skip_l} did not keep the lowest"
        )
        assert not np.array_equal(library, indices(keeps_highest))
