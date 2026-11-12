"""Source-target damping prediction, its compression and the candidate centre artifact."""

import os

import numpy as np
import pytest
import yaml
from click.testing import CliRunner
from scipy.special import spherical_jn

from pzbao.config import load_analysis_config
from pzbao.theory import TRANSFER_RESPONSE_FLOOR, transfer_damping_factor


def power(p):
    return 2.0e4 * (p / 0.02) / (1.0 + (p / 0.02) ** 2.4)


BASE = dict(
    linear_power=power,
    drag_scale_mpc_h=100.0,
    smoothing_mpc_h=15.0,
    bias=2.0,
    source_growth=0.0,
    reconstruction_growth=0.0,
    target_growth=0.0,
    source_sigma_r_mpc_h=0.0,
    target_sigma_r_mpc_h=0.0,
    source_noise_power=0.0,
)


def test_self_noiseless_limit_is_the_zeldovich_pair_dispersion_of_the_residual_field():
    # Without growth, noise or errors the residual displacement is (1 - S) Psi, so the pair
    # tensor is X delta_ij + Y q_i q_j with P -> P (1 - S)^2 and the kernel averages
    # exp[-k^2 (X + Y t^2) / 2] over t = cos(k, q).
    p = np.geomspace(1e-4, 2.0, 40001)
    x = p * BASE["drag_scale_mpc_h"]
    filtered = power(p) * (1 - np.exp(-0.5 * (p * BASE["smoothing_mpc_h"]) ** 2)) ** 2
    X = np.trapz(filtered * 2 * (1 / 3 - spherical_jn(1, x) / x), p) / (2 * np.pi**2)
    Y = np.trapz(filtered * 2 * spherical_jn(2, x), p) / (2 * np.pi**2)
    k = np.array([0.02, 0.1, 0.2, 0.3])
    t = np.linspace(0.0, 1.0, 2001)
    expected = np.trapz(np.exp(-0.5 * k[:, None] ** 2 * (X + Y * t**2)), t, axis=1)
    mu = np.array([0.1, 0.6, 0.95])
    D = transfer_damping_factor(k, mu, **BASE, pair_directions=128)
    np.testing.assert_allclose(D, np.repeat(expected[:, None], mu.size, axis=1), atol=5e-3)


PHYSICAL = {
    **BASE,
    "source_growth": 0.8,
    "reconstruction_growth": 0.7,
    "target_growth": 0.85,
    "source_sigma_r_mpc_h": 4.0,
    "target_sigma_r_mpc_h": 60.0,
    "source_noise_power": 3.0e3,
}


def test_error_tables_reproduce_the_pair_tensor_at_any_two_errors():
    from pzbao.theory import (
        _SYMMETRIC_PAIRS,
        _displacement_modes,
        independent_source_pair_covariance,
        pair_tensor_tables,
    )

    modes, weights = _displacement_modes(1e-3, 1.0, 12, 6)
    p = np.sqrt(np.einsum("ij,ij->i", modes, modes))
    rng = np.random.default_rng(3)
    inputs = dict(
        source_response=rng.uniform(0.1, 1.0, p.size),
        source_noise_filter=rng.uniform(0.1, 1.0, p.size),
        pair_separation=np.array([30.0, -60.0, 70.0]),
        line_of_sight=np.array([0.0, 0.0, 1.0]),
        target_growth=0.8,
        reconstruction_growth=0.6,
    )
    errors = rng.normal(0.0, 40.0, (4, 2))
    constant, first, second, difference = pair_tensor_tables(
        modes,
        weights,
        power(p),
        np.full(p.size, 2e3),
        **inputs,
        offsets=errors.ravel(),
        separations=errors[:, 0] - errors[:, 1],
    )
    for row, (left, right) in enumerate(errors):
        expected = independent_source_pair_covariance(
            modes, weights, power(p), np.full(p.size, 2e3), **inputs, target_errors=[left, right]
        )
        six = constant + first[2 * row] + second[2 * row + 1] + difference[row]
        np.testing.assert_allclose(
            six, [expected[i, j] for i, j in _SYMMETRIC_PAIRS], rtol=1e-10, atol=1e-10
        )


def test_error_average_is_the_direct_double_sum_when_target_errors_exceed_the_radius():
    from pzbao.theory import (
        _displacement_modes,
        _fibonacci_directions,
        independent_source_pair_covariance,
    )

    options = dict(pair_directions=2, radial_nodes=6, angular_nodes=4)
    k, mu = np.array([0.02, 0.06]), np.array([0.1, 0.5])
    D = transfer_damping_factor(k, mu, **PHYSICAL, **options)
    # Direct double sum over the same discrete Gaussian error grid.
    sigma_t, radius = PHYSICAL["target_sigma_r_mpc_h"], PHYSICAL["smoothing_mpc_h"]
    step = 0.25 * min(radius, sigma_t)
    offsets = step * np.arange(-np.ceil(6 * sigma_t / step), np.ceil(6 * sigma_t / step) + 1)
    w = np.exp(-0.5 * (offsets / sigma_t) ** 2)
    w /= w.sum()
    modes, weights = _displacement_modes(1e-4, 2.0, 6, 4)
    p = np.sqrt(np.einsum("ij,ij->i", modes, modes))
    nu = modes[:, 2] / p
    smoothing = np.exp(-0.5 * (p * radius) ** 2)
    divisor = 2.0 + 0.7 * nu**2
    response = smoothing * np.exp(-0.5 * (modes[:, 2] * 4.0) ** 2) * (2.0 + 0.8 * nu**2) / divisor
    kvec = np.array([[kk * np.sqrt(1 - m**2), 0.0, kk * m] for kk in k for m in mu])
    kernel, photoz = np.zeros(len(kvec)), np.zeros(len(kvec))
    keep = np.arange(offsets.size)
    for direction in _fibonacci_directions(2):
        for i in keep:
            for j in keep:
                tensor = independent_source_pair_covariance(
                    modes,
                    weights,
                    power(p),
                    np.full(p.size, 3.0e3),
                    source_response=response,
                    source_noise_filter=smoothing / divisor,
                    pair_separation=100.0 * direction,
                    target_errors=[offsets[i], offsets[j]],
                    line_of_sight=np.array([0.0, 0.0, 1.0]),
                    target_growth=0.85,
                    reconstruction_growth=0.7,
                )
                variance = np.einsum("ni,ij,nj->n", kvec, tensor, kvec)
                phase = np.cos(kvec[:, 2] * (offsets[i] - offsets[j]))
                kernel += w[i] * w[j] * phase * np.exp(-0.5 * variance) / 2
    for i in keep:
        for j in keep:
            photoz += w[i] * w[j] * np.cos(kvec[:, 2] * (offsets[i] - offsets[j]))
    np.testing.assert_allclose(D.ravel(), kernel / photoz, rtol=2e-4, atol=1e-6)


def test_large_scales_are_undamped_and_a_sparser_source_damps_more():
    physical = {
        **BASE,
        "source_growth": 0.8,
        "reconstruction_growth": 0.8,
        "target_growth": 0.8,
        "source_sigma_r_mpc_h": 4.0,
        "target_sigma_r_mpc_h": 12.0,
        "source_noise_power": 1.0e4,
    }
    k, mu = np.array([0.005, 0.1, 0.2]), np.array([0.1, 0.5])
    options = dict(pair_directions=8, radial_nodes=48, angular_nodes=12)
    D = transfer_damping_factor(k, mu, **physical, **options)
    sparse = transfer_damping_factor(k, mu, **{**physical, "source_noise_power": 1e5}, **options)
    np.testing.assert_allclose(D[0], 1.0, atol=5e-3)
    assert np.all(sparse[1:] < D[1:])
    assert np.all((D >= 0) & (D <= 1.0 + 1e-6))


def test_modes_below_the_response_floor_carry_no_predicted_wiggle():
    k, mu = np.array([0.3]), np.array([0.05, 0.95])
    D = transfer_damping_factor(
        k,
        mu,
        **{**BASE, "target_sigma_r_mpc_h": 40.0},
        pair_directions=4,
        radial_nodes=32,
        angular_nodes=8,
    )
    # exp[-(0.3 * 0.95 * 40)^2] is far below the floor; the nearly transverse mode keeps a
    # prediction.
    assert np.exp(-((0.3 * 0.95 * 40.0) ** 2)) < TRANSFER_RESPONSE_FLOOR
    assert D[0, 1] == 0.0 and D[0, 0] > 0.1


def test_inputs_follow_the_route_context_and_snapshot():
    from pzbao.theory import sigma_r_mpc_h, transfer_damping_inputs

    config = load_analysis_config()
    context = {
        "smoothing_mpc_h": 25.0,
        "bias": 2.0,
        "source_redshift": {"z_eff": 0.7, "growth_rate": 0.83},
        "target_redshift": {"z_eff": 0.8, "growth_rate": 0.85},
        "sigma0": {"source": 0.003, "target": 0.03},
        "density_h3_mpc3": {"source": {"NGC": 2e-4, "SGC": 4e-4}},
    }

    class Cosmology:
        rs_drag = 99.0

        def efunc(self, z):
            return 1.0 + z

        def get_fourier(self):
            class Fourier:
                def pk_interpolator(self, of):
                    assert of == "delta_m"
                    return lambda p, z: p * z

            return Fourier()

    inputs = transfer_damping_inputs(context, config, cosmology=Cosmology())
    ratio = config["selection"]["radial_randoms"]["exact_random_data_ratio"]
    speed = config["physics"]["speed_of_light_km_s"]
    assert inputs["source_noise_power"] == pytest.approx((1 + 1 / ratio) / 3e-4)
    assert inputs["linear_power"](2.0) == pytest.approx(1.6)
    assert inputs["drag_scale_mpc_h"] == 99.0 and inputs["smoothing_mpc_h"] == 25.0
    assert inputs["source_growth"] == inputs["reconstruction_growth"] == 0.83
    assert inputs["target_growth"] == 0.85
    assert inputs["target_sigma_r_mpc_h"] == pytest.approx(sigma_r_mpc_h(0.8, 0.03, 1.8, speed))
    assert inputs["source_sigma_r_mpc_h"] == pytest.approx(sigma_r_mpc_h(0.7, 0.003, 1.7, speed))


def candidate(context):
    from pzbao.likelihood import TRANSFER_CENTRE_DECISION

    return {
        "units": "Mpc/h",
        "review_reference": None,
        "records": [
            {
                "context": context,
                "centers_mpc_h": {"sigma_perp": 3.4, "sigma_parallel": 6.1, "sigma_streaming": 2.0},
                "evidence": {
                    "kind": "source_target_theory",
                    "reference": TRANSFER_CENTRE_DECISION,
                    "approximations": ["fixture"],
                    "diagnostics": {},
                },
            }
        ],
    }


def test_candidate_centres_are_refused_until_reviewed_then_resolve_exactly():
    from pzbao.config.transfer import transfer_run_configs
    from pzbao.theory import resolve_transfer_damping_prior

    config = transfer_run_configs(load_analysis_config())["sigma_0.003--R25"]
    context = {"target_class": "sigma_0.03", "measurement_bin": "LRG2"}
    artifact = candidate(context)
    config["measurement"]["reconstruction"]["transfer_run"]["damping_centres"] = artifact
    kwargs = dict(
        context=context, sigma_z_over_one_plus_z=0.03, z_eff=0.9, e_z_eff=1.6, config=config
    )
    with pytest.raises(ValueError, match="review reference"):
        resolve_transfer_damping_prior("LRG", "postreconstruction", **kwargs)
    artifact["review_reference"] = "fixture-only-reference"
    prior = resolve_transfer_damping_prior("LRG", "postreconstruction", **kwargs)
    assert prior["centers_mpc_h"] == artifact["records"][0]["centers_mpc_h"]
    assert prior["transverse_damping_source"] == "source_target_theory"


def test_command_writes_one_create_only_candidate(tmp_path, monkeypatch):
    from pzbao import likelihood, run as driver
    from pzbao.config.transfer import transfer_run_configs

    config = transfer_run_configs(load_analysis_config(), pilot=True)["self--R25"]
    metadata = {"workload": "transfer"}
    runs = {str(tmp_path / "run"): {"config": config, "name": "run", "metadata": metadata}}
    (tmp_path / "run").mkdir()
    (tmp_path / "proxy").mkdir()
    runs[str(tmp_path / "proxy")] = {"config": config, "name": "proxy"}
    monkeypatch.setattr(driver, "load_run", lambda path: runs[str(path)])
    calls = []

    def artifact(run, routes, *, covariance_run, tracer):
        assert routes == driver.routes_for(config, "transfer")
        calls.append((run["name"], covariance_run and covariance_run["name"], tracer))
        return candidate({"measurement_bin": "LRG2"})

    monkeypatch.setattr(likelihood, "transfer_centre_artifact", artifact)
    output = tmp_path / "centres" / "candidate.yaml"
    arguments = ["transfer-centres", "--run", str(tmp_path / "run"), "--output", str(output)]
    proxy = ["--covariance-from", str(tmp_path / "proxy"), "--tracer", "LRG"]
    result = CliRunner().invoke(driver.main, arguments + proxy)
    assert result.exit_code == 0, (result.output, result.exception)
    assert calls == [("run", "proxy", "LRG")]
    assert yaml.safe_load(output.read_text())["review_reference"] is None
    result = CliRunner().invoke(driver.main, arguments)
    assert result.exit_code != 0 and "already exists" in result.output
    assert len(calls) == 1


def test_compression_recovers_a_known_gaussian_damping(tmp_path, monkeypatch):
    if os.environ.get("BAO_TRANSFER_CENTRES_NATIVE") != "1":
        pytest.skip("native compression runs only inside a Slurm compute job")
    assert os.environ.get("SLURM_JOB_ID"), "native validation requires an allocated compute job"
    from pzbao.likelihood import _predict_and_compress, build_likelihood

    monkeypatch.setenv("DESILIKE_CONFIG_DIR", str(tmp_path))
    config = load_analysis_config()
    edges = np.arange(0.0, 201.0)
    muedges = np.linspace(-1.0, 1.0, 101)
    mu = 0.5 * (muedges[1:] + muedges[:-1])
    radius = 0.5 * (edges[1:] + edges[:-1])
    counts = 1 + 0.35 * (radius[:, None] / 200) * mu[None, :] ** 2
    identity = dict(
        sample_identity="transfer-centre-test",
        tracer="LRG",
        measurement_bin="LRG2",
        reconstruction_state="postreconstruction",
    )
    names = ("sigma_streaming", "sigma_parallel", "sigma_perp")
    prior = {
        "sigma_parallel_photoz_mpc_h": 20.0,
        "centers_mpc_h": {"sigma_streaming": 2.0, "sigma_parallel": 6.0, "sigma_perp": 3.0},
        "widths_mpc_h": {"sigma_streaming": 2.0, "sigma_parallel": 2.0, "sigma_perp": 1.0},
        "bounds_mpc_h": {name: [0.0, 30.0] for name in names},
        "unpriored_parameters": list(names),
    }
    separations = np.arange(50.0, 151.0, 4.0)
    variance = (2e-3 * (50.0 / separations) ** 2) ** 2
    common = dict(
        covariance=np.diag(np.tile(variance, 2)) / 25,
        separations_mpc_h=separations,
        fit_sedges=np.arange(48.0, 153.0, 4.0),
        z_eff=0.8,
        multipoles=[0, 2],
        reconstruction_convention="recsym",
        smoothing_mpc_h=15.0,
        ap_mode="qisoqap",
        method="profiling",
        damping_prior=prior,
        geometry_artifact=dict(identity, sedges=edges, muedges=muedges, wcounts=counts),
        config=config,
        **identity,
    )
    truth = {"sigma_perp": 4.0, "sigma_parallel": 7.5}

    def gaussian(k, mu):
        return np.exp(
            -0.5
            * k[:, None] ** 2
            * (truth["sigma_parallel"] ** 2 * mu**2 + truth["sigma_perp"] ** 2 * (1 - mu**2))
        )

    values = {"qiso": 1.0, "qap": 1.0, "b1": 2.0, "sigmas": 2.0}
    prediction, model, summary = _predict_and_compress(
        common, values, damping=gaussian, streaming=2.0, config=config
    )
    assert summary["fitted"]["sigmaper"] == pytest.approx(truth["sigma_perp"], abs=0.05)
    assert summary["fitted"]["sigmapar"] == pytest.approx(truth["sigma_parallel"], abs=0.1)
    residual = prediction - model
    assert residual @ np.linalg.solve(common["covariance"], residual) < 1e-3
    # A non-Gaussian damping leaves a residual the broadband partly absorbs; the residual of
    # the pinned best-fit model must then be exactly the profiled chi-square.
    distorted, model, summary = _predict_and_compress(
        common,
        values,
        damping=lambda k, mu: gaussian(k, mu) * (1 + 0.3 * k[:, None] ** 2 * mu**2) * 1.2,
        streaming=2.0,
        config=config,
    )
    residual = distorted - model
    chi2 = residual @ np.linalg.solve(common["covariance"], residual)
    assert chi2 > 1e-3
    assert chi2 == pytest.approx(-2 * summary["loglikelihood"], rel=1e-6, abs=1e-8)
    assert np.max(np.abs(distorted - prediction)) > 1e-4
    theory, *_ = build_likelihood(np.zeros(prediction.size), **common)
    assert theory.pt.damping_override is None
