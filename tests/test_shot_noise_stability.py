"""Conditional scalar sensitivity preserves the configured objective and input identities."""

import pzbao.diagnostics.covariance as diag_covariance
import copy
import numpy as np
import pytest


def _components(dimension):
    return {
        "c2": np.eye(dimension),
        "c3": np.zeros((dimension, dimension)),
        "c4": np.eye(dimension) * 0.1,
    }


def test_omission_curves_match_independent_isotropic_objective_and_keep_ids():
    rng = np.random.default_rng(86)
    vectors = rng.normal(size=(25, 26))
    vectors[-1] *= 4.0
    identities = np.arange(100, 125)
    components = _components(26)
    before = copy.deepcopy((components, vectors))
    result = diag_covariance.leave_one_out_calibration(
        components, vectors, identities, search_limits=[0.1, 4.0], points=40
    )
    assert result["full_ensemble"]["sample_covariance_rank"] == 24
    assert [r["omitted_realization_id"] for r in result["leave_one_out"]] == identities.tolist()
    grid = np.linspace(0.1, 4.0, 40)
    for index, row in enumerate(result["leave_one_out"]):
        retained = np.delete(vectors, index, axis=0)
        variance = np.var(retained, axis=0, ddof=1).sum()
        objective = 26 * np.log(0.1 + grid**2) + variance / (0.1 + grid**2)
        np.testing.assert_allclose(row["objective_curve"], objective, rtol=1e-13, atol=1e-13)
        assert row["shot_noise_rescaling"] == grid[np.argmin(objective)]
        assert row["calibration_realization_ids"] == np.delete(identities, index).tolist()
        assert row["sample_covariance_rank"] == 23
    assert result["leave_one_out"][-1]["delta_from_full"] < 0
    np.testing.assert_array_equal(vectors, before[1])
    for key in components:
        np.testing.assert_array_equal(components[key], before[0][key])


def test_omission_endpoints_are_reported_without_a_stability_acceptance_flag():
    result = diag_covariance.leave_one_out_calibration(
        _components(2), np.zeros((4, 2)), [8, 2, 7, 4], search_limits=[0.1, 4.0], points=10
    )
    assert result["full_ensemble"]["solution_on_search_boundary"]
    assert all((row["shot_noise_rescaling"] == 0.1 for row in result["leave_one_out"]))
    assert all((row["solution_on_search_boundary"] for row in result["leave_one_out"]))
    assert "accepted" not in result and "stable" not in result


@pytest.mark.parametrize(
    "vectors,ids",
    [
        (np.ones((2, 3)), [0, 1]),
        (np.ones((3, 2)), [0, 0, 1]),
        (np.ones((3, 2)), [0.0, 1.0, 2.0]),
        (np.ones((3, 2)), [0, -1, 2]),
        (np.ones((3, 2)), [0, 1]),
        (np.full((3, 2), np.nan), [0, 1, 2]),
    ],
)
def test_omission_refuses_invalid_vectors_and_id_axes(vectors, ids):
    with pytest.raises(ValueError, match="finite vectors"):
        diag_covariance.leave_one_out_calibration(
            _components(2), vectors, ids, search_limits=[0.1, 4.0], points=10
        )
