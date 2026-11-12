"""Fixed quadrature mathematics and explicit offline real-model comparisons."""

from __future__ import annotations
import pzbao.diagnostics.fitting as diag_fitting
import numpy as np
import pytest
from pzbao.models.mu_quadrature import (
    compare_multipole_orders,
    photoz_suppression,
    project_multipoles,
    symmetric_leggauss,
)

K_GRID = np.geomspace(0.0001, 0.6, 300)
LARGEST_PAIRWISE_SIGMA = float(np.sqrt(2.0) * 102.0)
GENTLE_PAIRWISE_SIGMA = float(np.sqrt(2.0) * 4.0)


def _template():
    """A smooth spectrum of the right shape with a realistic oscillation on top.

    The check is about how sharply the integrand varies in angle, which is set by the
    suppression and the damping rather than by the fine detail of the spectrum, so a smooth
    broadband times a decaying sinusoid is a faithful stand-in and keeps the test independent
    of a Boltzmann solver.
    """
    smooth = 20000.0 * (K_GRID / 0.05) ** (-1.5) / (1.0 + (K_GRID / 0.2) ** 2)
    wiggle = smooth * 0.05 * np.sin(K_GRID * 105.0) * np.exp(-((K_GRID / 0.35) ** 2))
    return (smooth, wiggle)


def test_the_rule_matches_the_library_convention():
    nodes, weights = symmetric_leggauss(4)
    assert nodes.shape == (4,) and weights.shape == (4,)
    assert np.all(nodes > 0.0) and np.all(np.diff(nodes) > 0.0)
    assert weights.sum() == pytest.approx(1.0)
    expected_nodes = np.array([0.18343464, 0.52553241, 0.79666648, 0.96028986])
    expected_weights = np.array([0.36268378, 0.31370665, 0.22238103, 0.10122854])
    assert np.allclose(nodes, expected_nodes)
    assert np.allclose(weights, expected_weights)


def test_a_rule_of_that_order_integrates_a_polynomial_exactly():
    nodes, weights = symmetric_leggauss(8)
    assert float((weights * nodes**2).sum()) == pytest.approx(1.0 / 3.0)
    assert float((weights * nodes**4).sum()) == pytest.approx(1.0 / 5.0)


def test_an_invalid_order_is_refused():
    with pytest.raises(ValueError, match="positive integer"):
        symmetric_leggauss(0)


def test_the_suppression_narrows_as_the_blur_grows():
    mu, _ = symmetric_leggauss(64)
    narrow = photoz_suppression(np.array([0.6]), mu, LARGEST_PAIRWISE_SIGMA)
    wide = photoz_suppression(np.array([0.6]), mu, 3.0)
    assert narrow[0, 0] == pytest.approx(0.571, abs=0.01)
    assert narrow[0, 1] < 0.01
    assert narrow[0, 2] < 1e-05
    assert int((narrow[0] > 0.5).sum()) == 1
    assert int((wide[0] > 0.5).sum()) >= 25


@pytest.mark.parametrize("order,within", [(64, False), (128, True)])
def test_full_signal_order_against_explicit_finer_integral(order, within):
    smooth, wiggle = _template()
    kwargs = dict(
        pairwise_sigma_mpc_h=LARGEST_PAIRWISE_SIGMA,
        multipoles=(0, 2, 4),
        growth_rate=0.8,
        linear_bias=2.0,
        sigma_par_mpc_h=9.0,
        sigma_perp_mpc_h=4.5,
    )
    low = project_multipoles(K_GRID, smooth, wiggle, order=order, **kwargs)
    high = project_multipoles(K_GRID, smooth, wiggle, order=256, **kwargs)
    comparison = compare_multipole_orders(
        low, high, multipoles=(0, 2, 4), maximum_fractional_difference=0.0001
    )
    assert comparison["within_tolerance"] is within


@pytest.mark.parametrize("order", [True, 3.5, -1])
def test_order_does_not_coerce_invalid_types(order):
    with pytest.raises(ValueError, match="positive integer"):
        symmetric_leggauss(order)


def test_comparison_uses_reference_monopole_at_each_wavenumber():
    reference = np.array([[1.0, 100.0], [0.1, 0.1]])
    changed = reference + np.array([[0.01, 0.0], [0.2, 0.2]])
    result = compare_multipole_orders(
        changed, reference, multipoles=(0, 2), maximum_fractional_difference=0.1
    )
    assert result["fractional_difference_by_multipole"] == pytest.approx([0.01, 0.2])
    assert result["within_tolerance"] is False


def test_real_model_study_uses_production_grid_without_promoting_order():
    pytest.importorskip("desilike")
    pytest.importorskip("cosmoprimo")
    import pzbao.theory as fitting_theory
    import pzbao.likelihood as fitting_likelihood
    from desilike.theories.galaxy_clustering import BAOPowerSpectrumTemplate
    from pzbao.catalog import build_mock_cosmology
    from pzbao.config import load_analysis_config

    config = load_analysis_config()
    cosmology = build_mock_cosmology(config)
    z_eff = 0.8
    prior = fitting_theory.resolve_damping_prior(
        "LRG",
        "prereconstruction",
        sigma_z_over_one_plus_z=0.03,
        z_eff=z_eff,
        e_z_eff=float(cosmology.efunc(z_eff)),
        config=config,
    )
    assert float(prior["sigma_parallel_photoz_mpc_h"]) == pytest.approx(144.0, abs=1.0)
    template = BAOPowerSpectrumTemplate(
        z=z_eff,
        fiducial=cosmology,
        apmode="qisoqap",
        only_now=False,
        with_now=str(config["fitting"]["no_wiggle_method"]),
    )
    settings = diag_fitting.quadrature_parameter_settings(prior, ap_mode="qisoqap", config=config)
    result = diag_fitting.real_model_quadrature_study(
        template=template,
        mode="",
        smoothing_radius=0.0,
        pairwise_sigma_mpc_h=prior["sigma_parallel_photoz_mpc_h"],
        broadband=config["fitting"]["broadband_model"],
        model_multipoles=(0, 2, 4),
        parameter_settings=settings,
        orders=[64, 128],
        reference_order=256,
        maximum_fractional_difference=0.0001,
    )
    assert result["production_order_changed"] is False
    assert config["fitting"]["photoz_response"]["mu_quadrature_order"] == 128
    assert (
        max((item["worst_fractional_difference"] for item in result["comparisons"][64].values()))
        > 0.01
    )
    assert all((item["within_tolerance"] for item in result["comparisons"][128].values()))
    np.testing.assert_allclose(result["k_h_mpc"], K_GRID, rtol=1e-12)
    assert not hasattr(fitting_likelihood, "_certify_production_mu_quadrature")


def test_criterion_requires_finite_arrays_and_reference_monopole():
    poles = np.ones((2, 5))
    with pytest.raises(ValueError, match="monopole"):
        compare_multipole_orders(
            poles, poles, multipoles=(2, 4), maximum_fractional_difference=0.0001
        )
    poles[0, 2] = np.nan
    with pytest.raises(ValueError, match="finite"):
        compare_multipole_orders(
            poles, poles, multipoles=(0, 2), maximum_fractional_difference=0.0001
        )
