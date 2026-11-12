"""Value-level contracts for density and covariance isolation."""

import pzbao.diagnostics.covariance as diag_covariance
import copy
import numpy as np
import pytest


def test_historical_count_recovery_and_failures():
    north, south, weight = (120, 83, 0.25)
    dd = {
        "autocorr": True,
        "ncounts": np.array([0, 10, 100]),
        "wcounts": np.array([0, 10, 100]) * weight**2,
        "wnorm": (north * (north - 1) + south * (south - 1)) * weight**2,
    }
    power = {"data_size1": north, "sum_data_weights1": north * weight}
    counts, error, integer = diag_covariance.infer_counts(dd, power)
    assert counts == {"NGC": north, "SGC": south}
    assert error == 0 and integer == 0
    broken = copy.deepcopy(dd)
    broken["wcounts"][1] *= 1.1
    with pytest.raises(diag_covariance.Unavailable, match="weights"):
        diag_covariance.infer_counts(broken, power)
    broken = {**dd, "wnorm": dd["wnorm"] + 2 * weight**2}
    with pytest.raises(diag_covariance.Unavailable, match="integer"):
        diag_covariance.infer_counts(broken, power)
    with pytest.raises(diag_covariance.Unavailable, match="metadata"):
        diag_covariance.infer_counts(dd, {**power, "data_size1": 0})


def test_old_covariance_needs_recorded_normalization_not_matching_amplitude():
    matrix = np.array([[4.0, 0.5], [0.5, 2.0]])
    matrices = np.repeat(matrix[None], 25, axis=0)
    with pytest.raises(diag_covariance.Unavailable, match="not directly recorded"):
        diag_covariance.common_historical_matrix(matrices, [{} for _ in range(25)], [0], [0])
    attrs = [{"observable": {"cov_scale": 1.0}} for _ in range(25)]
    np.testing.assert_array_equal(
        diag_covariance.common_historical_matrix(matrices, attrs, [0], [0]), matrix
    )
    attrs[-1]["observable"]["cov_scale"] = 25
    with pytest.raises(diag_covariance.Unavailable, match="single-realization"):
        diag_covariance.common_historical_matrix(matrices, attrs, [0], [0])
    attrs[-1]["observable"]["cov_scale"] = 1
    matrices[-1] *= 2
    with pytest.raises(diag_covariance.Unavailable, match="different matrices"):
        diag_covariance.common_historical_matrix(matrices, attrs, [0], [0])
    with pytest.raises(diag_covariance.Unavailable, match="multipoles"):
        diag_covariance.common_historical_matrix(matrices, attrs, [0, 2], [0])


def test_covariance_swap_preserves_all_vectors_scatter_and_window():
    vectors = np.random.default_rng(7).normal(size=(25, 52))
    current = {
        "arrays": {
            "vectors": vectors,
            "separations": np.tile(np.arange(26), (25, 1)),
            "fit_edges": np.arange(27),
            "multipoles": np.array([0, 2]),
            "realization_ids": np.arange(25),
            "covariance": 4 * np.eye(52),
            "mean_covariance": 4 * np.eye(52) / 25,
        },
        "window": {"sedges": np.arange(27), "muedges": [-1, 0, 1], "wcounts": np.ones((26, 2))},
    }
    changed = diag_covariance.swap_covariance(current, np.eye(52))
    np.testing.assert_array_equal(changed["arrays"]["vectors"], vectors)
    np.testing.assert_array_equal(
        changed["arrays"]["vectors"].std(axis=0, ddof=1), vectors.std(axis=0, ddof=1)
    )
    np.testing.assert_array_equal(changed["arrays"]["mean_covariance"], np.eye(52) / 25)
    np.testing.assert_array_equal(current["arrays"]["covariance"], 4 * np.eye(52))
    for key in current["window"]:
        np.testing.assert_array_equal(changed["window"][key], current["window"][key])
    with pytest.raises(diag_covariance.Unavailable, match="dimension"):
        diag_covariance.swap_covariance(current, np.eye(26))
    broken = np.eye(52)
    broken[0, 0] = -1
    with pytest.raises(np.linalg.LinAlgError):
        diag_covariance.swap_covariance(current, broken)


def test_unattainable_target_is_unavailable_not_capped():
    counts = {"NGC": {"LRG2": 101}, "SGC": {"LRG2": 90}}
    preparation = {
        "realization_ids": list(range(25)),
        "conditions": {
            "sigma_0.001": {
                cap: {"attainable_counts_by_mock": [[100]] * 25} for cap in ("NGC", "SGC")
            }
        },
    }
    with pytest.raises(diag_covariance.Unavailable, match="exceeds common availability"):
        diag_covariance.require_attainable(
            counts, preparation, {"condition": "sigma_0.001"}, ["LRG2"], list(range(25))
        )
    assert counts["NGC"]["LRG2"] == 101
    counts["NGC"]["LRG2"] = 100
    result = diag_covariance.require_attainable(
        counts, preparation, {"condition": "sigma_0.001"}, ["LRG2"], list(range(25))
    )
    assert result == {"NGC": {"LRG2": 100}, "SGC": {"LRG2": 100}}
