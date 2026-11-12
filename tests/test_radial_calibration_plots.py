"""The radial-calibration evidence pages must measure the noise floor, not decorate it.

These tests pin the arithmetic the bandwidth criterion rests on: the standardised
residual is the model minus the measurement in units of the standard error on that
measurement, and its root-mean-square is one exactly when the model sits at the noise
floor. Every acceptance number for the random catalogues is read off that scale, so an
error here would silently move both tolerances.
"""

from __future__ import annotations
import pzbao.diagnostics.selection as diag_selection
import json
import math
import numpy as np
import pytest


def _candidate(mass, mean, sigma, *, method="smoothing_spline", bandwidth=1.0, tvd=0.0):
    return {
        "sample_identity": "TEST|tier=native|cap=NGC",
        "smoothing_method": method,
        "bandwidth": float(bandwidth),
        "bandwidth_semantics": "smoothing_factor_per_center",
        "z_edges": np.linspace(0.4, 1.0, len(mean) + 1),
        "mean_counts": np.asarray(mean, dtype=float),
        "target_probability_mass": np.asarray(mass, dtype=float),
        "standard_error": np.asarray(sigma, dtype=float),
        "fractional_variance": np.full(len(mean), 0.001),
        "mean_fractional_variance": 0.001,
        "total_variation_distance": float(tvd),
    }


def test_a_model_equal_to_the_measurement_has_no_residual():
    mean = np.array([100.0, 200.0, 300.0, 400.0])
    mass = mean / mean.sum()
    residual = diag_selection.standardised_residual(mean, mass, np.array([10.0, 10.0, 10.0, 10.0]))
    assert np.allclose(residual, 0.0)


def test_the_residual_is_expressed_in_units_of_the_standard_error():
    mean = np.array([100.0, 100.0])
    sigma = np.array([5.0, 20.0])
    model_counts = np.array([105.0, 95.0])
    mass = model_counts / model_counts.sum()
    residual = diag_selection.standardised_residual(mean, mass, sigma)
    assert residual[0] == pytest.approx(1.0)
    assert residual[1] == pytest.approx(-0.25)


def test_shells_without_a_measured_standard_error_cannot_poison_the_summary():
    mean = np.array([100.0, 100.0, 100.0])
    mass = np.array([0.5, 0.25, 0.25])
    residual = diag_selection.standardised_residual(mean, mass, np.array([10.0, 0.0, np.nan]))
    assert np.all(np.isfinite(residual))
    assert residual[1] == 0.0 and residual[2] == 0.0


def test_a_residual_at_the_noise_floor_has_unit_root_mean_square():
    summary = diag_selection.residual_summary(np.array([1.0, -1.0, 1.0, -1.0]))
    assert summary["root_mean_square"] == pytest.approx(1.0)
    assert summary["fraction_beyond_band"] == 0.0
    assert summary["shell_count"] == 4


def test_an_overfitted_residual_is_visibly_below_the_floor():
    summary = diag_selection.residual_summary(np.full(64, 0.1))
    assert summary["root_mean_square"] == pytest.approx(0.1)
    assert summary["root_mean_square"] < diag_selection._radial_calibration_plots_NOISE_FLOOR_BAND


def test_the_poisson_expectation_is_the_reciprocal_mean_count():
    expected = diag_selection.poisson_fractional_variance(np.array([4.0, 100.0, 0.0]))
    assert expected[0] == pytest.approx(0.25)
    assert expected[1] == pytest.approx(0.01)
    assert math.isnan(expected[2])


def test_candidates_measured_on_different_grids_are_refused():
    mean = np.array([100.0, 100.0, 100.0, 100.0])
    sigma = np.full(4, 10.0)
    first = _candidate(mean / mean.sum(), mean, sigma)
    second = _candidate(mean / mean.sum(), mean, sigma)
    second["z_edges"] = np.linspace(0.1, 0.4, 5)
    with pytest.raises(ValueError, match="shell grid"):
        diag_selection.evaluate_candidates([first, second])


def test_candidates_measured_on_different_samples_are_refused():
    mean = np.array([100.0, 100.0, 100.0, 100.0])
    sigma = np.full(4, 10.0)
    first = _candidate(mean / mean.sum(), mean, sigma)
    second = _candidate(mean / mean.sum(), mean + 1.0, sigma)
    with pytest.raises(ValueError, match="one calibration measurement"):
        diag_selection.evaluate_candidates([first, second])


def test_the_selected_bandwidth_is_the_largest_that_still_reaches_the_floor():
    mean = np.full(64, 100.0)
    sigma = np.full(64, 10.0)
    candidates = []
    for bandwidth, offset in ((1.0, 2.0), (2.0, 9.0), (4.0, 16.0)):
        displaced = mean.copy()
        displaced[::2] += offset
        displaced[1::2] -= offset
        candidates.append(_candidate(displaced / displaced.sum(), mean, sigma, bandwidth=bandwidth))
    scored = diag_selection.evaluate_candidates(candidates)
    chosen = diag_selection.select_bandwidth_at_noise_floor(scored)
    assert chosen["bandwidth"] == 2.0
    assert chosen["root_mean_square"] == pytest.approx(0.9)


def test_when_every_candidate_overshoots_the_least_overshooting_one_is_returned():
    mean = np.full(32, 100.0)
    sigma = np.full(32, 10.0)
    candidates = []
    for bandwidth, offset in ((1.0, 15.0), (2.0, 25.0)):
        displaced = mean.copy()
        displaced[::2] += offset
        displaced[1::2] -= offset
        candidates.append(_candidate(displaced / displaced.sum(), mean, sigma, bandwidth=bandwidth))
    scored = diag_selection.evaluate_candidates(candidates)
    chosen = diag_selection.select_bandwidth_at_noise_floor(scored)
    assert chosen["bandwidth"] == 1.0
    assert chosen["root_mean_square"] > diag_selection._radial_calibration_plots_NOISE_FLOOR_BAND


def test_a_stored_candidate_round_trips_through_the_loader(tmp_path):
    mean = [100.0, 120.0, 90.0]
    payload = {
        "kind": "smooth_radial_candidate",
        "sample_identity": "LRG|tier=native|cap=NGC",
        "smoothing_method": "smoothing_spline",
        "bandwidth": 2.0,
        "bandwidth_semantics": "smoothing_factor_per_center",
        "z_edges": [0.4, 0.6, 0.8, 1.0],
        "mean_counts": mean,
        "target_probability_mass": [1 / 3, 1 / 3, 1 / 3],
        "fidelity_evidence": {"total_variation_distance": 0.05},
        "variance_evidence": {
            "standard_error_of_mean_count_by_bin": [5.0, 5.0, 5.0],
            "fractional_variance_by_bin": [0.001, None, 0.002],
            "mean_fractional_variance_over_occupied_bins": 0.0015,
        },
    }
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(payload))
    loaded = diag_selection.load_candidate(path)
    assert loaded["sample_identity"] == "LRG|tier=native|cap=NGC"
    assert loaded["bandwidth"] == 2.0
    assert math.isnan(loaded["fractional_variance"][1])
    assert np.allclose(loaded["mean_counts"], mean)


def test_a_foreign_artifact_is_refused_by_the_loader(tmp_path):
    path = tmp_path / "other.json"
    path.write_text(json.dumps({"kind": "attainable_count_record"}))
    with pytest.raises(ValueError, match="smooth_radial_candidate"):
        diag_selection.load_candidate(path)


def test_the_page_renders_from_scored_candidates(tmp_path):
    pytest.importorskip("matplotlib")
    mean = np.full(48, 250.0) + np.arange(48)
    sigma = np.sqrt(mean)
    candidates = [
        _candidate(mean / mean.sum(), mean, sigma, bandwidth=bandwidth) for bandwidth in (1.0, 4.0)
    ]
    scored = diag_selection.evaluate_candidates(candidates)
    written = diag_selection.render_page(scored, tmp_path / "pages" / "sample.png")
    assert written.exists() and written.stat().st_size > 0
