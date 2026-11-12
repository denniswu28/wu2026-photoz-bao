"""Joint photo-z-tier constraint: non-negative BLUE combination of tier posteriors."""

import numpy as np
import pytest

import pzbao.diagnostics.fitting as diag_fitting
import pzbao.fitting as fitting

TIERS = ["sigma_0.001", "sigma_0.003", "sigma_0.01", "sigma_0.03"]


def _identity_correlation(tiers, parameters=2):
    return np.eye(tiers * parameters)


def _posterior(sd_iso, sd_ap, rho=0.0):
    return np.array([[sd_iso**2, rho * sd_iso * sd_ap], [rho * sd_iso * sd_ap, sd_ap**2]])


def test_four_identical_uncorrelated_tiers_halve_the_width_with_equal_weights():
    means = np.array([[1.0, 1.0]] * 4)
    covariances = np.array([_posterior(0.01, 0.03)] * 4)
    record = fitting.combine_tier_posteriors(TIERS, means, covariances, _identity_correlation(4))
    np.testing.assert_allclose(record["standard_deviations_ddof1"], [0.005, 0.015])
    for parameter in fitting.JOINT_PARAMETERS:
        np.testing.assert_allclose(record["weights"][parameter], [0.25] * 4)
        assert record["removed_tiers"][parameter] == []
    np.testing.assert_allclose(record["mean"], [1.0, 1.0])
    single = 1 / np.sqrt(np.linalg.det(covariances[0]))
    np.testing.assert_allclose(record["two_dimensional_fom"], 4 * single)
    # The joint record is a posterior in the sense every other route uses: one FoM definition.
    metrics = diag_fitting.precision_metrics(record)
    np.testing.assert_allclose(metrics["joint_fom"], record["two_dimensional_fom"])


def test_perfectly_correlated_duplicate_tier_gets_zero_weight_and_leaves_the_width_unchanged():
    labels = ["sigma_0.001", "sigma_0.003", "sigma_0.01"]
    means = np.array([[0.99, 1.01], [0.99, 1.01], [1.02, 0.98]])
    covariances = np.array([_posterior(1.0, 1.0), _posterior(1.01, 1.01), _posterior(2.0, 2.0)])
    correlation = np.eye(6)
    for index in range(2):  # the second tier duplicates the first, estimate by estimate
        correlation[index, 2 + index] = correlation[2 + index, index] = 1.0
    with_duplicate = fitting.combine_tier_posteriors(labels, means, covariances, correlation)
    without = fitting.combine_tier_posteriors(
        [labels[0], labels[2]], means[[0, 2]], covariances[[0, 2]], np.eye(4)
    )
    for parameter in fitting.JOINT_PARAMETERS:
        assert with_duplicate["weights"][parameter][1] == 0.0
        assert with_duplicate["removed_tiers"][parameter] == ["sigma_0.003"]
        np.testing.assert_allclose(
            with_duplicate["weights"][parameter][[0, 2][0]], without["weights"][parameter][0]
        )
    np.testing.assert_allclose(
        with_duplicate["standard_deviations_ddof1"], without["standard_deviations_ddof1"], rtol=1e-6
    )
    np.testing.assert_allclose(without["standard_deviations_ddof1"], [np.sqrt(0.8)] * 2)


def test_negative_unconstrained_weights_are_removed_and_never_beat_the_unconstrained_width():
    widths = np.array([1.0, 1.2, 3.0])
    correlation = np.array([[1.0, 0.9, 0.0], [0.9, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assert np.linalg.eigvalsh(correlation).min() > 0
    covariance = correlation * np.outer(widths, widths)
    precision = np.linalg.inv(covariance)
    unconstrained = precision.sum(axis=1) / precision.sum()
    assert (unconstrained < 0).any(), "the fixture must produce a negative unconstrained weight"
    weights, removed = fitting.nonnegative_blue_weights(covariance)
    assert removed and all(weights[index] == 0 for index in removed)
    assert np.all(weights >= 0)
    np.testing.assert_allclose(weights.sum(), 1.0)
    constrained_width = np.sqrt(weights @ covariance @ weights)
    unconstrained_width = np.sqrt(unconstrained @ covariance @ unconstrained)
    assert unconstrained_width <= constrained_width <= widths.min()


def test_native_tier_is_refused_everywhere():
    means = np.array([[1.0, 1.0]] * 2)
    covariances = np.array([_posterior(0.01, 0.03)] * 2)
    with pytest.raises(ValueError, match="native tier"):
        fitting.combine_tier_posteriors(["native", "sigma_0.003"], means, covariances, np.eye(4))
    estimates = np.ones((6, 2, 2)) + np.arange(6)[:, None, None] * 0.01
    with pytest.raises(ValueError, match="native tier"):
        fitting.leave_one_out_combination(["sigma_0.003", "native"], estimates)


def test_nearest_positive_definite_leaves_a_definite_matrix_alone_and_repairs_an_indefinite_one():
    definite = np.array([[2.0, 0.5], [0.5, 1.0]])
    np.testing.assert_allclose(fitting.nearest_positive_definite(definite), definite)
    indefinite = np.array([[1.0, 0.95, 0.0], [0.95, 1.0, 0.95], [0.0, 0.95, 1.0]])
    assert np.linalg.eigvalsh(indefinite).min() < 0
    repaired = fitting.nearest_positive_definite(indefinite)
    np.linalg.cholesky(repaired)
    np.testing.assert_allclose(repaired, repaired.T)
    with pytest.raises(ValueError, match="positive eigenvalue"):
        fitting.nearest_positive_definite(-definite)


def test_leave_one_out_and_bootstrap_on_a_known_correlated_ensemble():
    rng = np.random.default_rng(2026)
    widths = np.array([[0.03, 0.09], [0.025, 0.08], [0.06, 0.15], [0.09, 0.2]])
    rho = 0.5
    draws = []
    for parameter in range(2):
        shared = rng.standard_normal(400)
        own = rng.standard_normal((400, 4))
        draws.append(
            widths[:, parameter] * (np.sqrt(rho) * shared[:, None] + np.sqrt(1 - rho) * own)
        )
    estimates = 1.0 + np.stack(draws, axis=-1)  # (realization, tier, parameter)
    loo = fitting.leave_one_out_combination(TIERS, estimates)
    assert loo["best_tier"] == ["sigma_0.003", "sigma_0.003"]
    assert all(loo["improves"])
    assert np.asarray(loo["estimates"]).shape == (400, 2)
    means = np.array([[1.0, 1.0]] * 4)
    covariances = np.array([_posterior(w[0], w[1]) for w in widths]) / 25
    record = fitting.joint_tier_constraint(
        TIERS, means, covariances, estimates[:25], resamples=50, seed=7
    )
    nominal = record["combination"]["standard_deviations_ddof1"]
    best = np.sqrt(covariances[1, 0, 0])
    assert nominal[0] < best
    low, high = np.asarray(record["bootstrap"]["width_percentiles_16_84"])[0]
    assert low <= high and 0.5 * nominal[0] < low and high < 1.5 * nominal[0]
    assert record["bootstrap"]["resamples_used"] + record["bootstrap"]["resamples_skipped"] == 50
    for parameter in fitting.JOINT_PARAMETERS:
        assert all(w >= 0 for w in record["combination"]["weights"][parameter])
        np.testing.assert_allclose(sum(record["combination"]["weights"][parameter]), 1.0)


def test_realization_combination_weights_sum_to_one_and_width_matches_the_blue_identity():
    rng = np.random.default_rng(5)
    widths = np.array([[0.03, 0.09], [0.025, 0.08], [0.06, 0.15], [0.09, 0.2]])
    shared = rng.standard_normal((60, 1, 2))
    estimates = 1.0 + widths * (0.7 * shared + 0.7 * rng.standard_normal((60, 4, 2)))
    record = fitting.realization_combination(TIERS, estimates)
    for index, parameter in enumerate(fitting.JOINT_PARAMETERS):
        weights = np.array(record["weights"][parameter])
        assert np.all(weights >= 0)
        np.testing.assert_allclose(weights.sum(), 1.0)
        scatter = np.array([record["tier_scatter"][tier][index] for tier in TIERS])
        # A non-negative BLUE never does worse than the best single estimate it combines.
        assert record["joint_width"][index] <= scatter.min() * (1 + 1e-9)
    with pytest.raises(ValueError, match="native tier"):
        fitting.realization_combination(["native", *TIERS[1:]], estimates)


def test_joint_variance_never_exceeds_the_best_tier_even_when_blocks_disagree():
    # Within-tier posterior correlations that disagree with the measured cross correlations make
    # the stacked matrix indefinite; the parameter variances must still be the BLUE variances.
    means = np.array([[1.0, 1.0]] * 4)
    widths = np.array([[0.0066, 0.023], [0.0051, 0.017], [0.03, 0.095], [0.039, 0.115]])
    covariances = np.array([_posterior(w[0], w[1], rho=-0.5) for w in widths])
    rng = np.random.default_rng(3)
    draws = rng.standard_normal((25, 8))
    draws[:, 1::2] += 2.0 * draws[:, 0::2]  # strong positive iso-ap correlation in the estimates
    correlation = np.corrcoef(draws.T)
    record = fitting.combine_tier_posteriors(TIERS, means, covariances, correlation)
    assert record["stacked_minimum_eigenvalue"] < 0
    for index in range(2):
        assert record["standard_deviations_ddof1"][index] <= widths[:, index].min() * (1 + 1e-9)
    np.linalg.cholesky(np.array(record["covariance"]))
