"""Diagnostic-only science helpers remain deterministic and non-controlling."""

import pzbao.diagnostics.reconstruction as diag_reconstruction
import pzbao.diagnostics.selection as diag_selection
import copy
import numpy as np
import pytest
from pzbao.config import load_analysis_config
from pzbao.selection import attach_stable_randomness


def _random_catalog():
    return {
        "TARGETID": np.array([10, 20, 30, 40]),
        "RA": np.array([1.0, 2.0, 3.0, 4.0]),
        "DEC": np.array([5.0, 6.0, 7.0, 8.0]),
        "Z": np.array([0.11, 0.12, 0.13, 0.14]),
        "SOURCE_ID": np.array([10, 20, 30, 40], dtype=np.uint64),
    }


def test_shuffled_redshifts_are_deterministic_and_non_mutating():
    config = load_analysis_config()
    randoms = attach_stable_randomness(_random_catalog(), role="random", config=config)
    before = copy.deepcopy(randoms)
    data_redshifts = np.asarray([0.31, 0.21, 0.41])
    result = diag_selection.shuffled_redshifts_diagnostic(randoms, data_redshifts, config=config)
    quantile_column = config["selection"]["stable_randomness"]["random_thinning_rank_column"]
    indices = np.minimum(
        (randoms[quantile_column] * len(data_redshifts)).astype(int), len(data_redshifts) - 1
    )
    np.testing.assert_allclose(result["Z"], np.sort(data_redshifts)[indices])
    np.testing.assert_array_equal(result["RA"], randoms["RA"])
    np.testing.assert_array_equal(result["DEC"], randoms["DEC"])
    for key in before:
        np.testing.assert_array_equal(randoms[key], before[key])
    invalid = copy.deepcopy(randoms)
    invalid[quantile_column][0] = np.nan
    with pytest.raises(ValueError, match="quantiles must be finite"):
        diag_selection.shuffled_redshifts_diagnostic(invalid, data_redshifts, config=config)


def test_analytic_diagnostics_have_physical_limits_and_guards():
    assert diag_reconstruction.pre_to_post_radial_damping_ratio(9.0, 6.0, 0.0) == pytest.approx(1.5)
    assert diag_reconstruction.pre_to_post_radial_damping_ratio(9.0, 6.0, 1000.0) == pytest.approx(
        1.0, rel=0.0001
    )
    assert diag_reconstruction.photoz_power_damping(0.0, 1.0, 10.0) == pytest.approx(1.0)
    assert diag_reconstruction.photoz_power_damping(0.2, 0.0, 10.0) == pytest.approx(1.0)
    values = diag_reconstruction.photoz_power_damping(np.array([0.01, 0.1]), 1.0, 10.0)
    assert 0 < values[1] < values[0] <= 1
    with pytest.raises(ValueError):
        diag_reconstruction.photoz_power_damping(0.2, np.nan, 10.0)


def _skinned_threshold_field(interior=80, skin=20, skin_weight=0.3, tail=0.0, tail_cells=0):
    total = 2 * tail_cells + 2 * skin + interior
    unsmoothed = np.zeros(total)
    smoothed = np.full(total, float(tail))
    start = tail_cells + skin
    unsmoothed[start : start + interior] = 1.0
    smoothed[start : start + interior] = 1.0
    smoothed[tail_cells:start] = skin_weight
    smoothed[start + interior : start + interior + skin] = skin_weight
    return (unsmoothed, smoothed)


@pytest.mark.parametrize("skin_weight", [0.05, 0.2, 0.3, 0.6])
def test_support_and_whole_mesh_weight_losses_have_the_same_discarded_numerator(skin_weight):
    unsmoothed, smoothed = _skinned_threshold_field(skin_weight=skin_weight)
    smoothed[45:50] = 0.001
    result = diag_reconstruction.random_threshold_diagnostics(
        unsmoothed, smoothed, random_object_count=80, total_random_weight=80.0
    )
    assert result["support_restricted_cell_loss_fraction"] == pytest.approx(5 / 80)
    assert result["total_smoothed_weight"] > result["support_smoothed_weight"]
    assert result["threshold_weight_loss_fraction"] == pytest.approx(
        result["support_restricted_weight_loss_fraction"]
        * result["support_smoothed_weight"]
        / result["total_smoothed_weight"]
    )
    assert skin_weight > result["applied_absolute_threshold"]


def test_threshold_fft_roundoff_is_distinct_from_a_negative_random_mesh():
    unsmoothed, smoothed = _skinned_threshold_field()
    arguments = dict(random_object_count=80, total_random_weight=80.0)
    smoothed[0] = -np.finfo(float).eps
    result = diag_reconstruction.random_threshold_diagnostics(unsmoothed, smoothed, **arguments)
    assert result["most_negative_smoothed_weight"] == smoothed[0]
    assert result["smoothed_transform_roundoff_bound"] > -smoothed[0]
    smoothed[0] = -1e-06
    with pytest.raises(ValueError, match="non-negative"):
        diag_reconstruction.random_threshold_diagnostics(unsmoothed, smoothed, **arguments)
    with pytest.raises(ValueError, match="non-negative"):
        diag_reconstruction.random_threshold_diagnostics(-unsmoothed, abs(smoothed), **arguments)
    with pytest.raises(ValueError, match="integer"):
        diag_reconstruction.random_threshold_diagnostics(
            unsmoothed, abs(smoothed), random_object_count=True, total_random_weight=80.0
        )


def test_threshold_cell_accounting_excludes_arbitrarily_large_far_kernel_tail():
    arguments = dict(random_object_count=80, total_random_weight=80.0)
    compact = diag_reconstruction.random_threshold_diagnostics(
        *_skinned_threshold_field(), **arguments
    )
    padded = diag_reconstruction.random_threshold_diagnostics(
        *_skinned_threshold_field(tail=1e-09, tail_cells=2000), **arguments
    )
    assert padded["mesh_shape"] == [4120] and compact["mesh_shape"] == [120]
    assert padded["accounted_cell_count"] == compact["accounted_cell_count"] == 120
    assert padded["threshold_cell_loss_fraction"] == compact["threshold_cell_loss_fraction"]
    assert compact["threshold_weight_loss_fraction"] == pytest.approx(0.0, abs=1e-12)
    assert padded["threshold_weight_loss_fraction"] == pytest.approx(4e-06 / 92.0, rel=0.001)
    assert padded["unaccounted_smoothed_weight_fraction"] == pytest.approx(4e-06 / 92.0, rel=0.001)


def test_ordered_edge_diagnostics_preserve_reference_bands_without_overall_acceptance():
    config = load_analysis_config()
    steps = [
        {
            "order": 0,
            "lost_fraction": 0.0,
            "alpha_shift_sigma_1": 0.0,
            "s_xi": 0.0,
            "interval_width_fractional_change": 0.0,
        },
        {
            "order": 1,
            "lost_fraction": 0.2,
            "alpha_shift_sigma_1": -0.2,
            "s_xi": 2.0,
            "interval_width_fractional_change": 0.04,
        },
    ]
    result = diag_reconstruction.ordered_edge_diagnostics(
        steps, realization_ids=config["analysis"]["mock_ids"], config=config
    )
    assert result[1]["reference_bands"] == {
        "alpha_shift_sigma_1": "amber",
        "s_xi": "red",
        "interval_width_fractional_change": "green",
    }
    assert "passes" not in str(result)
    steps[1]["lost_fraction"] = -0.1
    with pytest.raises(ValueError, match="lost fractions"):
        diag_reconstruction.ordered_edge_diagnostics(
            steps, realization_ids=config["analysis"]["mock_ids"], config=config
        )
