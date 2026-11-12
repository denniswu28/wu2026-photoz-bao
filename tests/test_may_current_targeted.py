"""Value, pairing and native-consumer contracts for the targeted diagnostic."""

import pzbao.diagnostics.covariance as diag_covariance
import pzbao.diagnostics.fitting as diag_fitting
import pzbao.diagnostics.reconstruction as diag_reconstruction
import pzbao.diagnostics.selection as diag_selection
import copy
from types import SimpleNamespace
import numpy as np
import pytest
from pzbao.config import load_analysis_config


def test_scalar_covariance_preserves_data_likelihood_optimum():
    matrix = np.array([[4.0, 1.0], [1.0, 3.0]])
    data = np.array([2.0, -1.0])
    models = np.array([[x, 2 * x] for x in np.linspace(-2, 2, 301)])
    residual = data - models
    baseline = np.einsum("ni,ij,nj->n", residual, np.linalg.inv(matrix), residual)
    for scale in (0.25, 0.5, 2.0, 4.0):
        changed = diag_covariance.covariance_intervention(matrix, scale=scale)
        statistic = np.einsum("ni,ij,nj->n", residual, np.linalg.inv(changed), residual)
        np.testing.assert_allclose(statistic, baseline / scale)
        assert statistic.argmin() == baseline.argmin()


def test_covariance_decomposition_separates_diagonal_and_correlations():
    current = np.array([[4.0, 1.0], [1.0, 9.0]])
    historical = np.array([[1.0, 0.75], [0.75, 2.25]])
    amplitude = diag_covariance.covariance_intervention(
        current, historical=historical, mode="amplitude"
    )
    correlation = diag_covariance.covariance_intervention(
        current, historical=historical, mode="correlation"
    )
    np.testing.assert_allclose(amplitude, current / 4)
    np.testing.assert_allclose(np.diag(correlation), np.diag(current))
    np.testing.assert_allclose(correlation, [[4, 3], [3, 9]])
    with pytest.raises(ValueError, match="identity"):
        diag_covariance.covariance_intervention(current, mode="correlation")
    with pytest.raises(ValueError, match="positive"):
        diag_covariance.covariance_intervention(current, scale=0)
    with pytest.raises(np.linalg.LinAlgError):
        diag_covariance.covariance_intervention([[1.0, 2.0], [2.0, 1.0]])


def test_no_window_projection_preserves_poles_without_angular_mixing():
    matrix = diag_fitting.unmixed_projection([50.0, 54.0], [0, 2], [0, 2, 4])
    raw = np.array([1.0, 2.0, 3.0, 4.0, 100.0, 200.0])
    np.testing.assert_array_equal(raw @ matrix, [1.0, 2.0, 3.0, 4.0])
    assert matrix.shape == (6, 4)


def test_scatter_is_projected_without_sample_precision():
    vectors = np.random.default_rng(42).normal(size=(25, 52))
    result = diag_covariance.covariance_metrics(np.eye(52), vectors)
    assert len(result["eigenmode_scatter_over_prediction"]) == 52
    assert len(result["whitened_residual_norm_squared"]) == 25
    with pytest.raises(ValueError, match="25"):
        diag_covariance.covariance_metrics(np.eye(52), vectors[:24])


@pytest.fixture
def damping():
    return {
        "centers_mpc_h": {"sigma_perp": 7.48, "sigma_parallel": 6.0, "sigma_streaming": 2.0},
        "widths_mpc_h": {"sigma_perp": 1.0, "sigma_parallel": 2.0, "sigma_streaming": 2.0},
        "bounds_mpc_h": {"sigma_perp": [0.0, None]},
        "sigma_parallel_photoz_mpc_h": 4.78,
    }


def test_prior_interventions_do_not_bundle_other_changes(damping):
    original = copy.deepcopy(damping)
    reference = diag_fitting.change_damping(
        damping, {"transverse_center_mpc_h": 3.0}, state="postreconstruction"
    )
    assert reference["centers_mpc_h"] == {
        "sigma_perp": 3.0,
        "sigma_parallel": 6.0,
        "sigma_streaming": 2.0,
    }
    assert reference["widths_mpc_h"] == damping["widths_mpc_h"]
    wide = diag_fitting.change_damping(
        damping, {"damping_width_factor": 3}, state="postreconstruction"
    )
    assert wide["centers_mpc_h"] == damping["centers_mpc_h"]
    assert wide["widths_mpc_h"] == {
        "sigma_perp": 3.0,
        "sigma_parallel": 6.0,
        "sigma_streaming": 6.0,
    }
    assert (
        diag_fitting.change_damping(
            damping, {"transverse_center_mpc_h": 3.0}, state="prereconstruction"
        )
        == damping
    )
    assert damping == original
    assert (
        diag_fitting.change_damping(
            reference, {"baseline_damping": True}, state="postreconstruction", baseline=damping
        )
        == damping
    )
    with pytest.raises(ValueError, match="baseline"):
        diag_fitting.change_damping(damping, {"baseline_damping": True}, state="postreconstruction")


def test_dilation_bounds_preserve_bias_growth_and_damping():
    config = load_analysis_config()
    changed = diag_fitting.configure_inference(config, {"dilation_limits": [0.6, 1.4]})
    assert changed["tracers"] == config["tracers"]
    assert changed["fitting"]["damping"] == config["fitting"]["damping"]
    assert changed["fitting"]["nonlinear_parameters"]["qap"]["prior"]["limits"] == [0.6, 1.4]
    assert config["fitting"]["nonlinear_parameters"]["qap"]["prior"]["limits"] == [0.8, 1.2]


def test_rank_change_is_only_allowed_for_selection_control():
    left = {
        "source_ids": np.arange(3),
        "noise": np.arange(3.0),
        "rsd_redshift": np.ones(3),
        "ranks": np.arange(3.0),
        "exterior": np.zeros(3, bool),
    }
    right = {**left, "ranks": left["ranks"] + 1}
    with pytest.raises(AssertionError):
        diag_selection.selection_pair(left, right)
    assert diag_selection.selection_pair(left, right, allow_rank_change=True)[
        "paired_photoz_deviates"
    ]
    with pytest.raises(AssertionError):
        diag_selection.selection_pair(
            left, {**right, "noise": right["noise"] + 1}, allow_rank_change=True
        )


def test_cell_exposure_distinguishes_counts_from_weights():
    mask = np.zeros((2, 2, 2), bool)
    mask[0, 0, 0] = True
    result = diag_reconstruction.weighted_cell_exposure(
        [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        [3.0, 1.0],
        mask,
        box=[2.0, 2.0, 2.0],
        center=[0.0, 0.0, 0.0],
    )
    assert result["nearest_cell_count_fraction"] == 0.5
    assert result["nearest_cell_weight_fraction"] == 0.75


def test_small_fixed_mesh_matches_native_iterative_fft():
    pyrecon = pytest.importorskip("pyrecon")
    rng = np.random.default_rng(73)
    center = np.array([300.0, 200.0, 500.0])
    points = [center + rng.uniform(-30, 30, size=(n, 3)) for n in (32, 1600)]
    data, randoms = [
        {
            **dict(zip(("X", "Y", "Z_CART"), p.T)),
            "Z": np.full(len(p), 0.7),
            "WEIGHT": np.ones(len(p)),
            "WEIGHT_FKP": np.ones(len(p)),
            "SOURCE_ID": np.arange(len(p)),
        }
        for p in points
    ]
    pipeline = SimpleNamespace(
        SHIFTED_COLUMNS=("X_SHIFTED", "Y_SHIFTED", "Z_CART_SHIFTED"),
        reconstruction_weight=lambda c, expression: c["WEIGHT"],
        _mesh_array=lambda mesh: np.asarray(mesh.value),
        reconstruction_shell_diagnostics=lambda *a, **k: {"synthetic_fixture": True},
    )
    kwargs = dict(
        common_geometry={"boxsize_mpc_h": [100.0] * 3, "boxcenter_mpc_h": center},
        source_visible_real_mesh_count=12,
        single_rank_peak_rss_ceiling_gib=1,
        weight_expression="WEIGHT",
        growth_rate=0.7,
        bias=2.0,
        resampler="tsc",
        fft_plan="estimate",
        mesh_dtype="f8",
        threshold_randoms=0.01,
        smoothing_radius_mpc_h=15.0,
        iterations=3,
        config={},
    )
    changed, changed_randoms, record = diag_reconstruction.fixed_mesh_reconstruction(
        data, randoms, nmesh=[8, 8, 8], pipeline=pipeline, **kwargs
    )
    native = pyrecon.IterativeFFTReconstruction(
        f=0.7,
        bias=2.0,
        los=None,
        positions=np.vstack(points),
        nmesh=[8, 8, 8],
        boxsize=[100.0] * 3,
        boxcenter=center,
        resampler="tsc",
        fft_plan="estimate",
        dtype="f8",
        threshold_randoms=0.01,
        wrap=False,
        position_type="pos",
    )
    native.assign_data(points[0], data["WEIGHT"], position_type="pos")
    native.assign_randoms(points[1], randoms["WEIGHT"], position_type="pos")
    native.set_density_contrast(
        threshold_randoms=0.01, smoothing_radius=15.0, check=False, kw_weights=None
    )
    native.run(niterations=3)
    for original, candidate, positions in (
        (data, changed, points[0]),
        (randoms, changed_randoms, points[1]),
    ):
        np.testing.assert_allclose(
            np.column_stack([candidate[k] for k in pipeline.SHIFTED_COLUMNS]),
            native.read_shifted_positions(positions, field="disp+rsd"),
            rtol=1e-10,
            atol=1e-10,
        )
        for key in original:
            np.testing.assert_array_equal(original[key], candidate[key])
    assert record["algorithm"] == "iterative_fft" and record["realized_nmesh"] == [8, 8, 8]


def test_log_gap_and_flags_do_not_turn_stopping_into_acceptance():
    result = diag_fitting.log_gap_response(100, 1000, 10000, stable=True, fraction=0.5)
    assert result["fraction_of_log_gap"] == pytest.approx(0.5)
    assert result["classification"] == "major conditional contributor"
    assert (
        diag_fitting.log_gap_response(100, 1000, 10000, stable=False, fraction=0.5)[
            "classification"
        ]
        == "inconclusive"
    )
    assert diag_fitting.log_gap_response(100, 110, 10000, stable=True, fraction=0.5)[
        "classification"
    ].startswith("insufficient")
