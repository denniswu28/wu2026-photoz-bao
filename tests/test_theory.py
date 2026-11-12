"""Independent unit and limiting-behavior tests for the photo-z relations."""

import numpy as np
import pytest

from pzbao.models.photoz_bao import gaussian_photoz_response
from pzbao.theory import (
    sigma_parallel_photoz_mpc_h,
    sigma_r_mpc_h,
)


def test_sigma_r_has_reviewed_units_and_sqrt_two_mapping():
    radial = sigma_r_mpc_h(0.7, 0.003, 1.47, 299792.458)
    parallel = sigma_parallel_photoz_mpc_h(0.7, 0.003, 1.47, 299792.458)
    assert radial == pytest.approx(10.40, rel=0.02)
    assert parallel == pytest.approx(np.sqrt(2) * radial)


def test_fixed_response_uses_one_object_observed_coordinate_scale():
    k = np.array([[0.1], [0.2]])
    mu = np.array([0.0, 0.5, 1.0])
    response = gaussian_photoz_response(k, mu, 10.0)
    np.testing.assert_allclose(response, np.exp(-((k * mu * 10.0) ** 2)))
    np.testing.assert_array_equal(response[:, 0], np.ones(2))


@pytest.mark.parametrize(
    "call",
    [
        lambda: sigma_r_mpc_h(0.7, -0.1, 1.4, 299792.458),
        lambda: sigma_r_mpc_h(0.7, 0.003, np.nan, 299792.458),
        lambda: gaussian_photoz_response(0.1, 1.0, -1.0),
    ],
)
def test_nonphysical_inputs_fail_closed(call):
    with pytest.raises(ValueError):
        call()


def test_damping_formula_inputs_do_not_broadcast_silently():
    with pytest.raises(ValueError, match="share a shape"):
        sigma_r_mpc_h([0.6, 0.7], [0.003], [1.4, 1.5], 299792.458)


def test_gaussian_damping_wedge_average_matches_numerical_integration_and_limits():
    from pzbao.theory import gaussian_damping_wedge_average

    k = np.linspace(0.0, 0.3, 61)
    sigma_r = 10.3
    for lower, upper in ((0.0, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1.0), (0.0, 1.0)):
        mu = np.linspace(lower, upper, 20001)
        numerical = np.trapz(np.exp(-((np.outer(k, mu) * sigma_r) ** 2) / 2), mu, axis=1) / (
            upper - lower
        )
        analytic = gaussian_damping_wedge_average(k, sigma_r, lower, upper)
        np.testing.assert_allclose(analytic, numerical, rtol=1e-6, atol=1e-9)
        assert analytic[0] == 1.0 and np.all(np.diff(analytic) <= 0)
    # The highest wedge is damped most at every nonzero wavenumber.
    low = gaussian_damping_wedge_average(k[1:], sigma_r, 0.0, 1 / 3)
    high = gaussian_damping_wedge_average(k[1:], sigma_r, 2 / 3, 1.0)
    assert np.all(high < low)
    for bad in ((0.5, 0.5), (-0.1, 0.5), (0.2, 1.2)):
        with pytest.raises(ValueError, match="wedge edges"):
            gaussian_damping_wedge_average(k, sigma_r, *bad)
    with pytest.raises(ValueError):
        gaussian_damping_wedge_average(k, 0.0, 0.0, 1.0)
