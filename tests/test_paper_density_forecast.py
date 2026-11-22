"""Value checks for the conditional forecast; native checks run with saved inputs."""

import numpy as np
import pytest

from pzbao.diagnostics import covariance as forecast

# The configured SPHEREx sky fraction (revision 2.15); synthetic rows must carry it to validate.
FSKY = forecast.configured_target_fsky()


def test_uniform_radial_moments_and_area_scaling():
    volume = np.array([2.0, 3.0, 5.0])
    density = np.full(3, 0.2)
    expected = np.array([1 / 0.2**2, 1 / 0.2, 1.0]) / volume.sum()
    np.testing.assert_allclose(forecast.noise_moments(volume, density, 7.0), expected)
    np.testing.assert_allclose(forecast.noise_moments(4 * volume, density, 7.0), expected / 4)
    # Splitting cells cannot change the physical forecast.
    np.testing.assert_allclose(
        forecast.noise_moments(np.repeat(volume / 2, 2), np.repeat(density, 2), 7.0), expected
    )


def test_loaded_route_forecast_keeps_mean_normalization_source_and_fixed_priors(monkeypatch):
    from types import SimpleNamespace
    from pzbao import catalog, covariance, likelihood, selection
    from pzbao.config import load_analysis_config

    config = load_analysis_config()
    route = dict(
        tracer="LRG",
        measurement_bin="LRG2",
        condition="native",
        reconstruction_state="prereconstruction",
    )
    measured = {"root": "separate measurement source"}
    provider = {"root": "separate covariance source"}
    run = dict(
        config=config,
        source_run=measured,
        metadata=dict(
            run_name="forecast-test", code_commit="a" * 40, analysis_config_sha256="b" * 64
        ),
    )
    edges = np.array([48.0, 52.0, 56.0])
    centers = 0.75 * np.diff(edges**4) / np.diff(edges**3)
    shells = 4 * np.pi / 3 * np.diff(edges**3)
    window = np.eye(6)[:3]
    observable = SimpleNamespace(
        wmatrix=SimpleNamespace(sin=centers, matrix_full=window, ellsin=(0, 2, 4))
    )
    native = SimpleNamespace(varied_params=[])
    saved = dict(
        arrays={name: np.eye(3) for name in ("c2", "c3", "c4")},
        metadata={"shot_noise": {"shot_noise_rescaling": 1.0}},
        window={"sedges": edges},
    )

    def covariance_reader(selected, selected_route):
        assert selected is provider and selected_route == route
        return saved

    def preparation_reader(selected, tracer):
        assert selected is measured and tracer == "LRG"
        return {"selected source": True}

    monkeypatch.setattr(covariance, "likelihood_covariance_run", lambda r, s: provider)
    monkeypatch.setattr(covariance, "read_route_covariance", covariance_reader)
    monkeypatch.setattr(selection, "read_preparation", preparation_reader)
    power = lambda k, z: 2.0e4 * np.asarray(k) / (1 + (np.asarray(k) / 0.02) ** 2.5)
    cosmology = SimpleNamespace(
        rs_drag=100.0,
        get_fourier=lambda: SimpleNamespace(pk_interpolator=lambda of: power),
    )
    monkeypatch.setattr(catalog, "build_mock_cosmology", lambda c: cosmology)
    monkeypatch.setattr(
        likelihood,
        "_route_likelihood",
        lambda *a, **k: (
            None,
            observable,
            native,
            dict(
                ap_mode="qisoqap",
                attained_density_h3_mpc3=0.1,
                z_eff=0.5,
                fitting_growth_rate=0.8,
            ),
            {"realization_ids": np.arange(25)},
            None,
        ),
    )
    monkeypatch.setattr(
        forecast, "radial_cells", lambda *a: (np.array([4.0, 6.0]), np.array([0.1, 0.1]))
    )
    monkeypatch.setattr(
        forecast,
        "parameter_point",
        lambda *a: (
            ["qiso", "qap", "sigmaper"],
            {"qiso": 1.0, "qap": 1.0, "sigmaper": 5.0},
            np.full(3, 0.001),
            np.diag([0.0, 0.0, 1.0]),
        ),
    )
    monkeypatch.setattr(forecast, "derivative_matrix", lambda *a: np.eye(3))
    nodes, weights = np.polynomial.legendre.leggauss(16)
    monkeypatch.setattr(
        forecast,
        "native_power",
        lambda *a: (np.linspace(0.001, 0.5, 8), nodes[8:], weights[8:], np.zeros((8, 8))),
    )
    areas = config["tracers"]["LRG"]["footprint"]["area_deg2"]
    fsky = 2 * sum(areas.values()) / (4 * np.pi * (180 / np.pi) ** 2)
    result = forecast.forecast_route(run, route, target_fsky=fsky)
    # Pure Poisson covariance has an independent analytic answer; the route must
    # retain C/25 and scale only the data covariance when increasing sky area.
    expected = np.diag(2 / (0.1**2 * 10 * shells * 25))
    np.testing.assert_allclose(result["results"]["current"]["covariance"], expected)
    np.testing.assert_allclose(result["results"]["target_area"]["covariance"], expected / 2)
    assert result["density_ratio"] == pytest.approx(1)
    assert result["area_factor"] == pytest.approx(2)
    assert result["n_mocks_normalization"] == 25
    assert result["identity"]["code_commit"] == "a" * 40
    assert result["scientific_acceptance"] is False
    # One survey keeps the same prior curvature; here the dilations carry none, so its
    # covariance is exactly 25 times the mean's.
    single = forecast.forecast_route(run, route, target_fsky=fsky, n_realizations=1)
    np.testing.assert_allclose(single["results"]["current"]["covariance"], 25 * expected)
    np.testing.assert_allclose(single["results"]["target_area"]["covariance"], 25 * expected / 2)
    assert single["n_mocks_normalization"] == 1
    with pytest.raises(ValueError, match="normalization"):
        forecast.forecast_route(run, route, target_fsky=fsky, n_realizations=26)
    # The damping-prior centre is the default evaluation point. On request a pre-reconstruction
    # route is evaluated at the unreconstructed damping instead, parallel = (1 + f) transverse.
    assert result["current_sigma_perp"] == 5.0
    points = []
    monkeypatch.setattr(
        forecast, "derivative_matrix", lambda *a: points.append(dict(a[3])) or np.eye(3)
    )
    moved = forecast.forecast_route(
        run, route, target_fsky=fsky, unreconstructed_prereconstruction=True
    )
    perp, parallel = forecast.unreconstructed_damping(lambda k: power(k, 0.5), 100.0, 0.8)
    assert moved["current_sigma_perp"] == moved["target_sigma_perp"] == perp
    assert moved["current_sigma_parallel"] == parallel == pytest.approx(1.8 * perp)
    assert [(p["sigmaper"], p["sigmapar"]) for p in points] == [(perp, parallel)] * 2
    points.clear()
    post = dict(route, reconstruction_state="postreconstruction")
    monkeypatch.setattr(covariance, "read_route_covariance", lambda *a: saved)
    unmoved = forecast.forecast_route(
        run, post, target_fsky=fsky, unreconstructed_prereconstruction=True
    )
    assert unmoved["current_sigma_perp"] == 5.0 and "sigmapar" not in points[0]
    saved["arrays"]["c4"] *= -1
    with pytest.raises(ValueError, match="Invalid source covariance"):
        forecast.forecast_route(run, route, target_fsky=fsky)


def test_matched_density_run_has_no_forecast_target():
    # Its tier targets are the primary's, so the "target" arm would not describe this run.
    run = dict(config=dict(selection=dict(density_run=dict(name="matched"))))
    with pytest.raises(ValueError, match="no forecast target"):
        forecast.forecast_route(run, dict(tracer="LRG"), target_fsky=FSKY)


def test_nonuniform_density_uses_radial_moments():
    volumes, densities = np.array([2.0, 8.0]), np.array([0.1, 0.7])
    actual = forecast.noise_moments(volumes, densities, 3.0)
    mean = np.full(2, np.average(densities, weights=volumes))
    assert not np.allclose(actual, forecast.noise_moments(volumes, mean, 3.0))
    assert actual[1] ** 2 <= actual[0] * actual[2] * (1 + 1e-14)


@pytest.mark.parametrize("density", ([0.0, 0.0], [1.0, -1.0], [1.0, np.nan]))
def test_invalid_density_is_unavailable(density):
    with pytest.raises(ValueError):
        forecast.noise_moments([1.0, 1.0], density, 1.0)


def response():
    return dict(
        kind="analytic_postreconstruction_sigma_perp",
        formula={"version": 2},
        p_shot=1.0,
        attained_density_h3_mpc3=0.1,
        reconstruction_variance_mpc_h2=9.0,
        shot_noise_variance_mpc_h2=16.0,
        analytic_center_mpc_h=5.0,
    )


def test_density_response_has_identity_and_physical_saturation_limits():
    source = response()
    assert forecast.forecast_sigma_perp(source, 0.1) == 5.0
    assert forecast.forecast_sigma_perp(source, 0.2) == pytest.approx(np.sqrt(17))
    assert forecast.forecast_sigma_perp(source, 1e12) == pytest.approx(3.0)
    assert source == response()


def test_incompatible_response_is_rejected():
    source = response()
    source["analytic_center_mpc_h"] = 3.0
    with pytest.raises(AssertionError):
        forecast.forecast_sigma_perp(source, 0.2)
    source = response()
    source["p_shot"] = 1.8
    with pytest.raises(ValueError):
        forecast.forecast_sigma_perp(source, 0.2)


def test_joint_fom_retains_correlation_and_marginalizes_nuisance():
    j = np.array([[1.0, 0.0, 0.8], [0.0, 1.0, 0.5], [0.2, 0.3, 1.0], [0.1, 0.2, 0.4]])
    covariance = np.diag([2.0, 3.0, 4.0, 5.0])
    prior = np.diag([0.0, 0.0, 0.3])
    expected = np.linalg.inv(j.T @ np.linalg.solve(covariance, j) + prior)[:2, :2]
    result = forecast.joint_information(j, covariance, prior, ["qiso", "qap", "nuisance"])
    np.testing.assert_allclose(result["covariance"], expected)
    assert result["fom"] == pytest.approx(1 / np.sqrt(np.linalg.det(expected)))
    assert result["fom"] != pytest.approx(1 / np.sqrt(np.prod(np.diag(expected))))
    conditioned = np.linalg.inv((j.T @ np.linalg.solve(covariance, j) + prior)[:2, :2])
    assert np.linalg.det(expected) > np.linalg.det(conditioned)


def test_no_prior_fom_scales_with_area_and_prior_curvature_does_not():
    j, cov = np.array([[1.0, 0.4], [0.2, 1.0]]), np.array([[2.0, 0.3], [0.3, 3.0]])
    baseline = forecast.joint_information(j, cov, np.zeros((2, 2)), ["qiso", "qap"])
    larger = forecast.joint_information(j, cov / 4, np.zeros((2, 2)), ["qiso", "qap"])
    assert larger["fom"] / baseline["fom"] == pytest.approx(4.0)
    first = forecast.joint_information(j, cov, np.eye(2), ["qiso", "qap"])
    second = forecast.joint_information(j, cov / 4, np.eye(2), ["qiso", "qap"])
    assert second["fom"] / first["fom"] != pytest.approx(4.0)


def test_unconstrained_direction_is_not_regularized_into_a_gain():
    with pytest.raises((ValueError, np.linalg.LinAlgError)):
        forecast.joint_information(np.zeros((3, 2)), np.eye(3), np.zeros((2, 2)), ["qiso", "qap"])


def test_pure_poisson_gaussian_covariance_has_exact_bin_and_multipole_normalization():
    k = np.geomspace(0.001, 1.0, 40)
    mu, weights = np.polynomial.legendre.leggauss(32)
    mu, weights = mu[16:], weights[16:]
    s, shells = np.array([50.0, 54.0]), np.array([10.0, 20.0])
    w = np.eye(6)
    moments = forecast.noise_moments([10.0], [0.2], 7.0)
    actual = forecast.gaussian_xi_covariance(
        k, mu, weights, np.zeros((40, 16)), s, shells, w, moments
    )
    expected = np.diag(
        np.concatenate([2 * (2 * ell + 1) * moments[0] / shells for ell in (0, 2, 4)])
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=0)
    enlarged = forecast.gaussian_xi_covariance(
        k, mu, weights, np.zeros((40, 16)), s, shells, w, moments / 4
    )
    np.testing.assert_allclose(enlarged, actual / 4)


def test_isotropic_signal_has_no_cross_multipole_covariance_and_window_projects():
    k = np.geomspace(0.001, 0.5, 80)
    mu, weights = np.polynomial.legendre.leggauss(64)
    mu, weights = mu[32:], weights[32:]
    s, shells = np.array([50.0, 54.0]), np.array([10.0, 20.0])
    power = np.repeat((10 * np.exp(-k))[:, None], len(mu), axis=1)
    moments = forecast.noise_moments([10.0], [0.2], 7.0)
    full = forecast.gaussian_xi_covariance(k, mu, weights, power, s, shells, np.eye(6), moments)
    np.testing.assert_allclose(full[:2, 2:], 0.0, atol=1e-12)
    window = np.array([[0.8, 0.2, 0, 0, 0, 0], [0, 0, 0.1, 0.9, 0, 0]])
    observed = forecast.gaussian_xi_covariance(k, mu, weights, power, s, shells, window, moments)
    np.testing.assert_allclose(observed, window @ full @ window.T)


def test_ordered_contributions_keep_negative_density_effects():
    row = forecast.ratio_terms(10.0, 8.0, 32.0)
    assert row["density_contribution"] == pytest.approx(-0.2)
    assert row["area_contribution"] == pytest.approx(2.4)
    assert 1 + row["density_contribution"] + row["area_contribution"] == pytest.approx(3.2)


@pytest.mark.parametrize("mode", ["", "recsym"])
@pytest.mark.parametrize("pairwise_sigma", [0.0, 144.0])
def test_native_angular_capture_preserves_parent_model_and_matches_its_multipoles(
    mode, pairwise_sigma
):
    """Use the real dependency graph that lost the capture hook in the native pilot."""
    pytest.importorskip("desilike")
    pytest.importorskip("cosmoprimo")
    from desilike.theories.galaxy_clustering import (
        BAOPowerSpectrumTemplate,
        DampedBAOWigglesTracerCorrelationFunctionMultipoles,
    )
    from pzbao.catalog import build_mock_cosmology
    from pzbao.config import load_analysis_config
    from pzbao.models.photoz_bao import build_observed_photoz_power_model
    from scipy.special import eval_legendre

    config = load_analysis_config()
    template = BAOPowerSpectrumTemplate(
        z=0.8,
        fiducial=build_mock_cosmology(config),
        apmode="qisoqap",
        only_now=False,
        with_now=config["fitting"]["no_wiggle_method"],
    )
    power = build_observed_photoz_power_model(
        template=template,
        mode=mode,
        smoothing_radius=15.0,
        pairwise_sigma_mpc_h=pairwise_sigma,
        mu_quadrature_order=128,
    )
    theory = DampedBAOWigglesTracerCorrelationFunctionMultipoles(
        pt=power, broadband="power3", s=np.array([50.0, 100.0]), ells=(0, 2, 4)
    )
    point = dict(qiso=1.01, qap=0.99, b1=2.0, dbeta=1.0, sigmas=2.0, sigmaper=4.0, sigmapar=8.0)
    before = np.array(theory(**point), copy=True)
    saved_poles = np.array(theory.pt.power, copy=True)
    projection = theory.pt.to_poles
    k, mu, weights, pkmu = forecast.native_power(theory, point)
    assert pkmu.shape == (len(k), len(mu))
    projected = [(2 * ell + 1) * (pkmu @ (weights * eval_legendre(ell, mu))) for ell in (0, 2, 4)]
    np.testing.assert_allclose(projected, saved_poles, rtol=1e-10, atol=1e-10)
    assert theory.pt.to_poles == projection
    np.testing.assert_allclose(theory(**point, force=True), before, rtol=1e-12, atol=1e-12)
    with pytest.raises(AssertionError):
        forecast.native_power(theory, dict(point, sigmaper=5.0))


def example_rows():
    rows = []
    for name in forecast.BINS:
        for tier in forecast.TIERS:
            for state, base, ratio in zip(forecast.STATES, (10.0, 20.0), (1.2, 1.5)):
                ratio = 1.0 if tier == "native" else ratio
                values = dict(
                    current=base,
                    target=base * ratio,
                    noise_only=base * (1.0 if tier == "native" else 1.1),
                    target_area=4 * base * ratio,
                )
                flags = dict(
                    is_valid=True,
                    lower_valid=True,
                    upper_valid=True,
                    at_lower_limit=False,
                    at_upper_limit=False,
                )
                rows.append(
                    dict(
                        route=dict(
                            measurement_bin=name, condition=tier, reconstruction_state=state
                        ),
                        status="available",
                        scientific_acceptance=False,
                        results={k: dict(fom=v) for k, v in values.items()},
                        numerics={
                            arm: dict(
                                derivative_fractional_change=0.0, integration_fractional_change=0.0
                            )
                            for arm in ("current", "target")
                        },
                        target_fsky=FSKY,
                        target_nbar=None if tier == "native" else 0.2,
                        cap_area_deg2=dict(NGC=5000.0, SGC=2000.0),
                        area_factor=forecast.area_factor(dict(NGC=5000.0, SGC=2000.0), FSKY),
                        density_target_available=tier != "native",
                        source_profile=dict(
                            parameters=[
                                dict(basename=p, interval_flags=dict(flags))
                                for p in ("qiso", "qap")
                            ]
                        ),
                        **forecast.ratio_terms(base, base * ratio, 4 * base * ratio),
                    )
                )
    return rows


def test_paired_density_response_is_not_assumed_area_scaling():
    rows = example_rows()
    gains = forecast.paired_gains(rows)
    assert len(gains) == 25
    assert gains[0]["density_reconstruction_gain_ratio"] == pytest.approx(1.0)
    assert gains[1]["density_reconstruction_gain_ratio"] == pytest.approx(1.25)
    assert gains[1]["density_reconstruction_gain_percent"] == pytest.approx(25.0)
    assert gains[1]["reconstruction_gain"]["current"] == pytest.approx(2.0)
    assert gains[1]["reconstruction_gain"]["target"] == pytest.approx(2.5)


def test_refresh_rejects_missing_routes_or_changed_contributions():
    from pzbao.figures import data as refresh

    rows = example_rows()
    data = dict(
        rows=rows,
        all_available=True,
        specification_revision=forecast.REVISION,
        target_fsky=FSKY,
        paired_gains=forecast.paired_gains(rows),
    )
    assert refresh.validate_forecast(data) == rows
    data["rows"] = rows[:-1]
    with pytest.raises(ValueError, match="50 primary routes"):
        refresh.validate_forecast(data)
    data["rows"] = rows
    rows[-1]["density_contribution"] = 0.0
    with pytest.raises(AssertionError):
        refresh.validate_forecast(data)


def test_missing_source_profile_requires_reason_and_remains_flagged():
    from pzbao.figures import data as refresh

    row = example_rows()[0]
    assert not refresh.flagged(row)
    row["source_profile"] = dict(
        parameters=[], unavailable_reason="No primary power3 BGS mean-profile export is registered."
    )
    assert refresh.flagged(row)
    row["source_profile"]["unavailable_reason"] = None
    with pytest.raises(ValueError, match="without reason"):
        refresh.flagged(row)
    row["source_profile"] = dict(
        parameters=[dict(basename="qiso")], unavailable_reason="incomplete"
    )
    with pytest.raises(ValueError, match="Missing source dilation diagnostics"):
        refresh.flagged(row)


def test_forecast_sky_fraction_comes_from_validated_configuration():
    from pzbao.catalog import FULL_SKY_AREA_DEG2
    from pzbao.config import load_analysis_config

    # Revision 2.15: the SPHEREx effective area of 30,000 deg^2 (Feder et al. 2024) over the sky.
    configured = forecast.configured_target_fsky()
    assert configured == load_analysis_config()["reporting"]["forecast_target_fsky"]
    assert configured == 30000.0 / FULL_SKY_AREA_DEG2
    caps = dict(NGC=4530.177644444443, SGC=2552.158311111111)
    assert forecast.area_factor(caps, 0.65) == pytest.approx(3.786098962318873)
    assert forecast.area_factor(caps, 0.65) / forecast.area_factor(caps, 0.75) == pytest.approx(
        13 / 15
    )


@pytest.mark.parametrize("target", [0.0, -1.0, 1.1, float("nan")])
def test_forecast_rejects_invalid_sky_fraction(target):
    with pytest.raises(ValueError, match="footprint or sky fraction"):
        forecast.area_factor(dict(NGC=5000.0, SGC=2000.0), target)


def test_refresh_rejects_old_sky_fraction_and_wrong_area_factor():
    from pzbao.figures import data as refresh

    rows = example_rows()
    data = dict(
        rows=rows,
        all_available=True,
        specification_revision=forecast.REVISION,
        target_fsky=0.75,
        paired_gains=forecast.paired_gains(rows),
    )
    with pytest.raises(AssertionError):
        refresh.validate_forecast(data)
    data["target_fsky"] = FSKY
    rows[0]["area_factor"] *= 0.75 / FSKY
    with pytest.raises(AssertionError):
        refresh.validate_forecast(data)


def test_forecast_bar_endpoints_and_widths(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from IPython.display import Markdown

    from tests.notebook_helpers import notebook_namespace

    namespace = notebook_namespace(
        "forecasts", cells={"setup", "presentation-style", "density_caption", "density-forecast"}
    )
    displayed = []
    namespace["display"] = displayed.append
    namespace["density_plot_forecast"](example_rows())
    captured = {"fig": namespace["figures"]["forecast-density-area"]}
    captions = [item.data for item in displayed if isinstance(item, Markdown)]
    assert len(captions) == 1 and f"f_sky = {FSKY:g}" in captions[0]
    try:
        labels = [t.get_text() for t in captured["fig"].axes[-1].get_legend().get_texts()]
        assert any(f"{FSKY:g}" in label for label in labels)
        assert all("0.75" not in label for label in labels)
        for ax in captured["fig"].axes[:5]:
            assert len(ax.patches) == 30
            for segment in ax.patches:
                assert segment.get_width() == pytest.approx(0.325)
            for i, tier in enumerate(forecast.TIERS):
                for j, ratio in enumerate((1.2, 1.5)):
                    ratio = 1.0 if tier == "native" else ratio
                    segment = ax.patches[6 * i + 3 * j + 2]
                    assert segment.get_y() == pytest.approx(ratio)
                    assert segment.get_y() + segment.get_height() == pytest.approx(4 * ratio)
    finally:
        plt.close(captured["fig"])


def test_changed_input_is_rejected(tmp_path):
    path = tmp_path / "input.json"
    path.write_text("{}")
    inputs = forecast.Inputs()
    inputs.read(path)
    path.write_text('{"changed": true}')
    with pytest.raises(ValueError, match="Changed input"):
        inputs.verify()


def test_unreconstructed_damping_is_the_formula_without_smoothing():
    from pzbao.theory import reconstruction_transverse_damping_squared

    power = lambda k: 2.0e4 * np.asarray(k) / (1 + (np.asarray(k) / 0.02) ** 2.5)
    perp, parallel = forecast.unreconstructed_damping(power, 100.0, 0.8)
    # S(k) = exp(-k^2 R^2 / 2) vanishes on the whole quadrature grid at this radius.
    limit = reconstruction_transverse_damping_squared(
        linear_power=power, smoothing_radius_mpc_h=1.0e8, drag_scale_mpc_h=100.0
    )
    assert perp == pytest.approx(np.sqrt(limit), rel=1e-12)
    smoothed = reconstruction_transverse_damping_squared(
        linear_power=power, smoothing_radius_mpc_h=15.0, drag_scale_mpc_h=100.0
    )
    assert perp > np.sqrt(smoothed)
    assert parallel == pytest.approx(1.8 * perp)
    with pytest.raises(ValueError, match="unreconstructed damping"):
        forecast.unreconstructed_damping(power, 0.0, 0.8)


def test_saved_selection_returns_the_saved_counts_volumes_and_redshifts():
    # The saved volumes are those of the configured cosmology, so the check needs its engine.
    pytest.importorskip("cosmoprimo")
    import json
    from pathlib import Path

    from pzbao import catalog
    from pzbao.config import load_analysis_config
    from pzbao.figures.render_spherex_forecast import saved_selection

    config = load_analysis_config()
    cosmology = catalog.build_mock_cosmology(config)
    source = Path(__file__).resolve().parents[1] / "paper/figures/source/baseline"
    record = saved_selection(source, config, cosmology)(None, "LRG")
    row = next(
        r
        for r in json.loads((source / "density.json").read_text())
        if (r["measurement_bin"], r["condition"]) == ("LRG2", "sigma_0.003")
    )
    redshift = next(
        r["z_eff"]
        for r in json.loads((source / "effective-redshifts.json").read_text())
        if (r["measurement_bin"], r["condition"]) == ("LRG2", "sigma_0.003")
    )
    route = dict(tracer="LRG", measurement_bin="LRG2", condition="sigma_0.003")
    # The forecast's own reader accepts the stand-in and recovers the saved totals; the two
    # caps carry the one saved radial density.
    volume, density = forecast.radial_cells(record, dict(config=config), route, cosmology)
    assert volume.sum() == pytest.approx(row["volume_mpc_h3"], rel=1e-9)
    assert (volume * density).sum() == pytest.approx(row["mean_count"], rel=1e-9)
    np.testing.assert_allclose(density.reshape(2, -1), [row["density"]] * 2, rtol=1e-9)
    assert record["effective_redshifts"]["sigma_0.003"]["LRG2"]["z_eff"] == redshift
    groups = record["conditions"]["sigma_0.003"]
    assert set(groups) == {"NGC", "SGC"} and set(groups["NGC"]["counts"]) == {
        "LRG1",
        "LRG2",
        "LRG3",
    }
