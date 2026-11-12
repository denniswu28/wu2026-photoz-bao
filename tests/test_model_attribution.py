"""Controlled scientific differences, empirical pairing and native likelihood invariants."""

import pzbao.diagnostics.fitting as diag_fitting
import numpy as np
import pytest


def test_correlated_residual_contributions_are_signed_and_sum_to_chi2():
    matrix = np.array([[1.0, 0.8], [0.8, 1.0]])
    residual = np.array([1.0, 0.2])
    result = diag_fitting.residual_summary(residual, np.zeros(2), matrix, [50.0, 54.0], [0])
    expected = residual @ np.linalg.solve(matrix, residual)
    assert result["chi2"] == pytest.approx(expected)
    assert sum(result["signed_chi2_contribution"]) == pytest.approx(expected)
    assert result["signed_chi2_contribution"][1] < 0
    assert sum(np.square(result["diagonal_standardized_residual"])) != pytest.approx(expected)
