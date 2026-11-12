"""Preserve the field figure's joint-posterior and complete-ensemble statistics."""

import pzbao.diagnostics.sampling as diag_sampling
import numpy as np
import pytest


def test_joint_fom_keeps_correlation_and_sample_covariance():
    draws = np.array([[0.98, 0.95], [1.01, 1.03], [1.04, 1.06], [0.99, 1.01]])
    result = diag_sampling.posterior_statistics(draws)
    expected = np.cov(draws.T, ddof=1)
    np.testing.assert_allclose(result["covariance_ddof1"], expected)
    assert result["two_dimensional_fom"] == pytest.approx(1 / np.sqrt(np.linalg.det(expected)))
    assert result["two_dimensional_fom"] != pytest.approx(1 / np.prod(draws.std(axis=0, ddof=1)))
    with pytest.raises(ValueError, match="singular"):
        diag_sampling.posterior_statistics(np.ones((5, 2)))
