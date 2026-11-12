"""Regression tests for the DESI-calibrated dilation-precision forecast."""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.colors import to_hex

from pzbao.diagnostics import covariance as forecast
from pzbao.figures.data import (
    DISPLAY_LABELS,
    SCENARIO_LINESTYLES,
    SCENARIO_MARKERS,
    SCENARIO_ORDER,
    build_r_precision_projection,
    calibration_coefficient_at,
    calibration_summary,
    density_only_r_over_r_cv,
    r_precision_percent,
)
from pzbao.figures.style import categorical_color


def test_visible_scenario_styles_are_blue_green_orange_red():
    assert SCENARIO_ORDER == (
        "desi_dr1",
        "desi_dr2_projection",
        "spherex_sample_1",
        "project_mock_achieved_sigma_0p003",
    )
    assert [to_hex(categorical_color(index)) for index in range(len(SCENARIO_ORDER))] == [
        "#0584c6",
        "#058d50",
        "#ed9900",
        "#ff4501",
    ]
    assert len(SCENARIO_MARKERS) == len(SCENARIO_ORDER)
    assert len(SCENARIO_LINESTYLES) == len(SCENARIO_ORDER)
    assert DISPLAY_LABELS["project_mock_achieved_sigma_0p003"] == "DESI mocks"


def test_table7_calibration_is_stable():
    calibration = calibration_summary()
    assert calibration["coefficient_pct_gpc_h3_sqrt"] == pytest.approx(0.8891647766302937)
    assert calibration["fractional_residual_rms"] == pytest.approx(0.04430166253506886)
    assert len(calibration["rows"]) == 11


def test_r_precision_has_inverse_square_root_scaling():
    result = r_precision_percent(np.array([1.0e9, 4.0e9]), coefficient=0.9)
    np.testing.assert_allclose(result, [0.9, 0.45])


def test_density_only_ratio_has_expected_fkp_limits():
    result = density_only_r_over_r_cv(np.array([1.0e-4, 1.0e-3]), p0_h3_mpc3=1.0e4)
    np.testing.assert_allclose(result, [2.0, 1.1])


@pytest.mark.parametrize("density", [0.0, -1.0, np.nan, np.inf])
def test_density_only_ratio_rejects_invalid_density(density):
    with pytest.raises(ValueError, match="number densities"):
        density_only_r_over_r_cv(density, p0_h3_mpc3=1.0e4)


@pytest.mark.parametrize("volume", [0.0, -1.0, np.nan, np.inf])
def test_r_precision_rejects_invalid_volume(volume):
    with pytest.raises(ValueError, match="effective volumes"):
        r_precision_percent(volume)


def test_projection_preserves_plotted_scenarios_and_spherex_cv_ordering():
    bins = [{"bin": f"z{index}", "redshift_center": 0.1 + 0.2 * index} for index in range(5)]
    scenarios = []
    for index, key in enumerate(
        (
            "desi_dr1",
            "desi_dr2_projection",
            "spherex_sample_1",
            "project_mock_achieved_sigma_0p003",
        )
    ):
        scenarios.append(
            {
                "key": key,
                "source_policy": f"policy-{index}",
                "nbar_h3_mpc3": [float(index + 1) * 1.0e-4] * 5,
                "shell_volume_mpc_h3": [10.0e9] * 5,
                "veff_mpc_h3": [float(index + 1) * 1.0e9] * 5,
            }
        )
    source = {
        "bins": bins,
        "geometry_policy": {"cosmic_variance_reference": "DESI footprint"},
        "scenarios": scenarios,
        "cosmic_variance": {
            "source_policy": "infinite density",
            "veff_mpc_h3": [6.0e9] * 5,
        },
    }
    projection = build_r_precision_projection(source, p0_h3_mpc3=10_000.0)
    assert [row["key"] for row in projection["scenarios"]] == [row["key"] for row in scenarios]
    cosmic_variance = np.asarray(projection["spherex_cosmic_variance"]["sigma_r_over_r_pct"])
    for scenario in projection["scenarios"]:
        assert np.all(cosmic_variance < np.asarray(scenario["sigma_r_over_r_pct"]))
    achieved = projection["scenarios"][-1]
    np.testing.assert_allclose(achieved["veff_mpc_h3"], [10.0e9 / 1.25**2] * 5)
    np.testing.assert_allclose(achieved["density_only_r_over_r_cv"], [1.25] * 5)
    assert "cosmic_variance" not in projection
    assert "density_only_panel" not in projection
    configured = forecast.configured_target_fsky()
    assert projection["absolute_panel"]["target_fsky"] == configured
    footprint = f"SPHEREx f_sky={configured:.4f}"
    assert footprint != "SPHEREx f_sky=0.6500"
    assert footprint in projection["absolute_panel"]["finite_curve_area_policy"]
    assert footprint in projection["absolute_panel"]["cosmic_variance_curve"]
    assert footprint in projection["spherex_cosmic_variance"]["source_policy"]
    assert projection["density_scaling"]["p0_h3_mpc3"] == 10_000.0


def test_coefficient_of_z_reproduces_table7_rows_exactly():
    """A Table 7 bin center must return that row's own coefficient."""
    from pzbao.figures.data import DESI_TABLE7_BGS_LRG

    for row in DESI_TABLE7_BGS_LRG:
        center = 0.5 * (row["zmin"] + row["zmax"])
        expected = row["sigma_r_pct"] * np.sqrt(row["veff_gpc_h3"])
        got = calibration_coefficient_at([center], [row["tracer"]])[0]
        assert got == pytest.approx(expected)


def test_coefficient_of_z_tracks_the_table7_trend():
    """A(z) falls from the BGS bins to LRG z~0.9; a median would hide that."""
    centers = [0.15, 0.30, 0.50, 0.70, 0.90]
    tracers = ["BGS", "BGS", "LRG", "LRG", "LRG"]
    a_of_z = calibration_coefficient_at(centers, tracers)
    median = calibration_summary()["coefficient_pct_gpc_h3_sqrt"]
    assert a_of_z[0] == pytest.approx(0.8959, abs=1e-3)
    assert a_of_z[-1] == pytest.approx(0.8453, abs=1e-3)
    # The high-z end is the part a single coefficient gets wrong.
    assert a_of_z[-1] / median < 0.96
    assert a_of_z[:3].min() / median > 0.99


def test_coefficient_interpolation_never_crosses_the_tracer_boundary():
    """BGS and LRG are different populations; mixing them would be a fiction."""
    bgs = calibration_coefficient_at([0.30], ["BGS"])[0]
    lrg = calibration_coefficient_at([0.30], ["LRG"])[0]
    assert bgs != lrg
    with pytest.raises(ValueError, match="no Table 7 rows"):
        calibration_coefficient_at([0.30], ["ELG"])
    with pytest.raises(ValueError, match="must match"):
        calibration_coefficient_at([0.3, 0.5], ["BGS"])


def test_r_precision_accepts_per_bin_coefficients():
    volumes = np.array([1.0e9, 4.0e9])
    per_bin = r_precision_percent(volumes, np.array([0.9, 0.8]))
    assert per_bin[0] == pytest.approx(0.9)
    assert per_bin[1] == pytest.approx(0.4)
    with pytest.raises(ValueError, match="must match"):
        r_precision_percent(volumes, np.array([0.9, 0.8, 0.7]))


def test_varying_a_leaves_every_density_ratio_unchanged():
    """A(z) cancels at fixed z, so no density comparison can shift."""
    import json
    from pathlib import Path

    data = json.loads(
        (
            Path(__file__).resolve().parent.parent
            / "paper/figures/source/fig_survey_r_precision_data.json"
        ).read_text()
    )
    scenarios = {s["key"]: s for s in data["scenarios"]}
    mock = scenarios["project_mock_achieved_sigma_0p003"]
    target = scenarios["spherex_sample_1"]
    for key in ("sigma_r_over_r_pct", "sigma_r_over_r_pct_single_a"):
        ratio = np.asarray(mock[key]) / np.asarray(target[key])
        np.testing.assert_allclose(
            ratio,
            np.asarray(mock["sigma_r_over_r_pct"]) / np.asarray(target["sigma_r_over_r_pct"]),
            rtol=1e-12,
        )
