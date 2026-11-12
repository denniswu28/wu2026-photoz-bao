"""Value-level contracts; run these in the workbook's Slurm validation card."""

import pzbao.diagnostics.clustering as diag_clustering
import pzbao.diagnostics.sampling as diag_sampling
import numpy as np
import pytest


def test_three_arm_pairing_and_zero_denominator_remain_explicit():
    current = np.zeros((25, 2, 26))
    dense = current.copy()
    dense[:, 0, :] = np.arange(25)[:, None]
    result = diag_clustering.compare_arrays(current, dense, np.arange(25), np.arange(48, 156, 4))
    assert all((value is None for row in result["sd_ratio"] for value in row))
    np.testing.assert_array_equal(result["paired_differences"], dense)
    np.testing.assert_allclose(result["paired_mean"][0], 12)
    np.testing.assert_allclose(result["paired_sd"][0], np.std(np.arange(25), ddof=1))
    np.testing.assert_allclose(result["paired_nominal_sem"], result["paired_sd"] / 5)
    with pytest.raises(AssertionError):
        diag_clustering.compare_arrays(current, dense, np.arange(25)[::-1], np.arange(27))
    with pytest.raises(ValueError):
        diag_clustering.compare_arrays(current[:-1], dense[:-1], np.arange(24), np.arange(27))


def test_sampling_moments_do_not_treat_time_or_walker_axes_as_parameters():
    values = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    result = diag_sampling.moment_statistics(values)
    np.testing.assert_array_equal(result["mean"], [3.0, 4.0])
    np.testing.assert_array_equal(result["sd"], [2.0, 2.0])
    assert result["correlation"] == pytest.approx(1.0)
    assert diag_sampling.moment_statistics(np.ones((10, 2)))["correlation"] is None
    with pytest.raises(ValueError):
        diag_sampling.moment_statistics(np.ones((2, 3, 2)))


def test_mc_error_batches_ensemble_slices_and_rejects_short_autocorrelation(monkeypatch):
    import emcee.autocorr

    seen = []

    def tau(values, **kwargs):
        seen.append(values.shape)
        return np.array([2.0])

    monkeypatch.setattr(emcee.autocorr, "integrated_time", tau)
    rng = np.random.RandomState(17)
    result = diag_sampling.batch_precision(rng.normal(size=(40, 128, 2)))
    assert result["available"] and result["batches"] == 32
    assert seen == [(128,)] * 5
    assert len(result["mcse"]) == 5
    assert not diag_sampling.batch_precision(rng.normal(size=(40, 20, 2)))["available"]
