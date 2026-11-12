"""Contracts for the exploratory fixed-covariance mesh fitting adapter."""

import pzbao.diagnostics.fitting as diag_fitting
import copy
import numpy as np
import pytest


def fixture():
    window = {"sedges": np.arange(3), "muedges": np.arange(3), "wcounts": np.ones((2, 2))}
    return {
        "arrays": {
            "realization_ids": np.arange(25),
            "vectors": np.zeros((25, 4)),
            "separations": np.ones((25, 2)),
            "covariance": np.eye(4) * 5,
            "mean_covariance": np.eye(4) / 5,
        },
        "window": window,
        "metadata": {},
    }


def test_full_ensemble_keeps_covariance_scaling_and_own_window():
    source = fixture()
    vectors = np.arange(100).reshape(25, 4)
    window = copy.deepcopy(source["window"])
    window["wcounts"] *= 2
    saved = diag_fitting.replace_vectors(
        source, vectors, source["arrays"]["separations"], list(range(25)), window
    )
    np.testing.assert_array_equal(saved["arrays"]["vectors"].mean(axis=0), [48, 49, 50, 51])
    np.testing.assert_array_equal(saved["arrays"]["mean_covariance"], np.eye(4) / 5)
    np.testing.assert_array_equal(saved["window"]["wcounts"], np.ones((2, 2)) * 2)
    np.testing.assert_array_equal(source["arrays"]["vectors"], np.zeros((25, 4)))


@pytest.mark.parametrize(
    "bad", ["missing", "reordered", "nonfinite", "double_scaled", "window_axes"]
)
def test_rejects_incompatible_inputs(bad):
    source = fixture()
    vectors = source["arrays"]["vectors"].copy()
    ids = list(range(25))
    window = copy.deepcopy(source["window"])
    if bad == "missing":
        vectors = vectors[:-1]
    elif bad == "reordered":
        ids.reverse()
    elif bad == "nonfinite":
        vectors[2, 1] = np.nan
    elif bad == "double_scaled":
        source["arrays"]["mean_covariance"] /= 25
    else:
        window["sedges"] = np.arange(3) + 1
    with pytest.raises((AssertionError, ValueError)):
        diag_fitting.replace_vectors(source, vectors, source["arrays"]["separations"], ids, window)
