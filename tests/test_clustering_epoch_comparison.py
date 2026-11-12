"""Scientific array/identity contracts for the read-only epoch diagnostic."""

import pzbao.diagnostics.clustering as diag_clustering
import pzbao.diagnostics.covariance as diag_covariance
import numpy as np
import pytest
from tests.test_covariance import native_estimator


def test_pairing_uses_ids_and_rejects_duplicates_and_tiny_cohorts():
    assert diag_clustering.common_ids([3, 0, 2, 1], [2, 0, 4], [0, 2]) == [0, 2]
    with pytest.raises(ValueError, match="duplicate"):
        diag_clustering.common_ids([0, 0, 1], [0, 1])
    with pytest.raises(ValueError, match="two common"):
        diag_clustering.common_ids([0, 1], [1, 2])


def test_sample_sd_nominal_sem_and_paired_difference_are_distinct():
    before = np.asarray([[[1.0, 2.0]], [[3.0, 6.0]], [[5.0, 10.0]]])
    after = before + 2.0
    statistics = diag_clustering.ensemble_statistics(before)
    np.testing.assert_allclose(statistics["mean"], [[3.0, 6.0]])
    np.testing.assert_allclose(statistics["sd"], [[2.0, 4.0]])
    np.testing.assert_allclose(statistics["nominal_sem"], [[2.0, 4.0]] / np.sqrt(3))
    paired = diag_clustering.ensemble_statistics(after - before)
    np.testing.assert_array_equal(paired["mean"], [[2.0, 2.0]])
    np.testing.assert_array_equal(paired["sd"], [[0.0, 0.0]])
    with pytest.raises(ValueError):
        diag_clustering.ensemble_statistics(np.full((2, 1, 1), np.nan))


def test_real_pair_count_serialization_rebinning_and_multipole_order(tmp_path):
    pytest.importorskip("pycorr")
    from pycorr import TwoPointCorrelationFunction

    native = native_estimator(radial_edges=np.arange(48.0, 57.0))
    path = tmp_path / "counts.npy"
    native.save(path)
    loaded = TwoPointCorrelationFunction.load(path)
    edges = np.array([48.0, 52.0, 56.0])
    _, actual = diag_clustering.count_vector(loaded, edges, ells=(2, 0))
    _, expected = native[::4](ells=(2, 0), return_sep=True, return_std=False)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    with pytest.raises(ValueError):
        diag_clustering.count_vector(loaded, np.array([48.0, 51.5, 55.0]))


def test_pooled_counts_preserved_separately_from_arithmetic_mean():
    pytest.importorskip("pycorr")
    edges = np.array([48.0, 52.0, 56.0])
    estimators = {7: native_estimator(cap="NGC"), 2: native_estimator(cap="SGC")}
    result = diag_clustering.summarize_estimators(estimators, [2, 7], edges)
    _, direct = diag_clustering.count_vector(estimators[2] + estimators[7], edges)
    np.testing.assert_allclose(result["pooled"], direct)
    np.testing.assert_allclose(result["mean"], result["vectors"].mean(axis=0))
    assert not np.allclose(result["pooled"], result["mean"])
    assert result["ids"] == [2, 7]


def historical_observable():
    return {
        "ells": [2, 0],
        "sedges": [[48.0, 52.0, 56.0]] * 2,
        "s": [[50.1, 54.1]] * 2,
        "data": [[3.0, 4.0], [1.0, 2.0]],
        "covariance": np.diag([9.0, 16.0, 1.0, 4.0]),
        "std": [[3.0, 4.0], [1.0, 2.0]],
    }


def test_historical_observable_keeps_actual_pole_order_and_covariance():
    poles, data, matrix = diag_clustering.decode_historical_observable(
        historical_observable(), np.array([48.0, 52.0, 56.0])
    )
    assert poles == (2, 0)
    np.testing.assert_array_equal(data[0], [3.0, 4.0])
    np.testing.assert_array_equal(np.diag(matrix), [9.0, 16.0, 1.0, 4.0])


@pytest.mark.parametrize(
    "key,value",
    [
        ("ells", [0, 0]),
        ("sedges", [[49.0, 53.0, 57.0]] * 2),
        ("s", [[47.0, 54.0]] * 2),
        ("covariance", np.eye(3)),
        ("std", np.ones((2, 2))),
    ],
)
def test_historical_observable_rejects_wrong_axes_and_error_scaling(key, value):
    original = historical_observable()
    original[key] = value
    with pytest.raises((ValueError, AssertionError)):
        diag_clustering.decode_historical_observable(original, np.array([48.0, 52.0, 56.0]))


def test_zero_noise_ratio_is_unavailable_and_singular_covariance_is_displayable():
    np.testing.assert_allclose(
        diag_clustering.safe_ratio([4, 2], [2, 0]), [2, np.nan], equal_nan=True
    )
    np.testing.assert_allclose(
        diag_covariance.covariance_correlation(np.ones((4, 4))), np.ones((4, 4))
    )
    with pytest.raises(ValueError):
        diag_covariance.covariance_correlation(np.array([[1.0, 2.0], [0.0, 1.0]]))


def make_route():
    vectors = np.arange(12.0).reshape(3, 2, 2)
    epoch = {
        "ids": [0, 1, 2],
        "vectors": vectors,
        "separations": np.tile([50, 54], (3, 1)),
        "pooled": vectors.mean(axis=0),
        "pooled_separation": np.array([50, 54]),
        **diag_clustering.ensemble_statistics(vectors),
    }
    newer = {**epoch, "vectors": vectors * 2, **diag_clustering.ensemble_statistics(vectors * 2)}
    return {
        "measurement_bin": "LRG2",
        "condition": "sigma_0.001",
        "reconstruction_state": "prereconstruction",
        "edges": np.array([48, 52, 56]),
        "centers": np.array([50, 54]),
        "historical": epoch,
        "september": newer,
        "historical_fit": {
            "poles": [2, 0],
            "ids": [0, 1, 2],
            "records": [],
            "vectors": vectors[:, ::-1],
            "covariances": np.tile(np.diag([4, 4, 1, 1]), (3, 1, 1)),
            "all_vectors_match": False,
        },
        "september_fit": {"poles": [0, 2], "covariance": np.diag([9, 9, 16, 16])},
    }
