"""Native measured counts supply the fitter's exact subwindow, vector order and RR support."""

import numpy as np
import pytest

from pzbao.config import load_analysis_config
from pzbao.likelihood import xi_fit_data, likelihood_window_inputs
from tests.test_covariance import native_estimator


@pytest.mark.parametrize("tracer", ["BGS_BRIGHT-21.5", "LRG"])
def test_linear_native_counts_roundtrip_yields_configured_fit_vector_and_window(tmp_path, tracer):
    pytest.importorskip("pycorr")
    from pycorr import TwoPointCorrelationFunction

    config = load_analysis_config()
    measurement, fitting = config["measurement"]["xi"], config["fitting"]
    edges = np.arange(
        measurement["s_min_mpc_h"],
        measurement["s_max_mpc_h"] + 0.5 * measurement["bin_width_mpc_h"],
        measurement["bin_width_mpc_h"],
    )
    native = native_estimator(radial_edges=edges, post=True)
    path = tmp_path / "counts.npy"
    native.save(str(path))
    loaded = TwoPointCorrelationFunction.load(str(path))
    poles = fitting["fit_surfaces"][tracer]["observed_multipoles"]
    factor = int(fitting["input_bin_width_mpc_h"] / measurement["bin_width_mpc_h"])
    separation, vector, fit_edges = xi_fit_data(
        loaded,
        multipoles=poles,
        fit_edge_limits_mpc_h=fitting["fit_edge_limits_mpc_h"],
        rebin_factor=factor,
        expected_bin_width_mpc_h=fitting["input_bin_width_mpc_h"],
    )
    np.testing.assert_array_equal(fit_edges, np.arange(48.0, 156.0, 4.0))
    assert vector.shape == (26 * len(poles),)
    expected_s, expected_poles = native[::factor](ells=poles, return_sep=True, return_std=False)
    for i, pole in enumerate(expected_poles):
        np.testing.assert_array_equal(vector[i * 26 : (i + 1) * 26], pole[12:38])
    np.testing.assert_array_equal(separation, expected_s[12:38])
    identity = {
        "sample_identity": "test",
        "tracer": tracer,
        "measurement_bin": "LRG3" if tracer == "LRG" else "BGS1",
        "reconstruction_state": "postreconstruction",
    }
    rr = loaded.R1R2
    record = {**identity, "sedges": rr.edges[0], "muedges": rr.edges[1], "wcounts": rr.wcounts}
    result = likelihood_window_inputs(
        record,
        **identity,
        observed_multipoles=poles,
        model_multipoles=[0, 2, 4],
        fit_sedges=fit_edges,
    )
    np.testing.assert_array_equal(result["wmatrix"]["wcounts"], rr.wcounts)
    assert result["ellsin"] == (0, 2, 4)
    for key, value in (
        ("tracer", "wrong"),
        ("measurement_bin", "wrong"),
        ("sample_identity", "wrong"),
        ("reconstruction_state", "prereconstruction"),
    ):
        with pytest.raises(ValueError, match="another fitted sample"):
            likelihood_window_inputs(
                {**record, key: value},
                **identity,
                observed_multipoles=poles,
                model_multipoles=[0, 2, 4],
                fit_sedges=fit_edges,
            )


@pytest.mark.parametrize(
    "fit_edges",
    [
        np.arange(50.0, 154.0, 4.0),
        np.arange(48.0, 155.0, 3.0),
        [196.0, 200.0, 204.0],
        [48.0, 52.0, 60.0],
    ],
)
def test_linear_fit_window_refuses_a_nonuniform_or_noncoarsening_grid(fit_edges):
    identity = {
        "sample_identity": "test",
        "tracer": "LRG",
        "measurement_bin": "LRG3",
        "reconstruction_state": "postreconstruction",
    }
    record = {
        **identity,
        "sedges": np.arange(0.0, 204.0, 4.0),
        "muedges": np.linspace(-1, 1, 9),
        "wcounts": np.ones((50, 8)),
    }
    with pytest.raises(ValueError, match="coarsen"):
        likelihood_window_inputs(
            record,
            **identity,
            observed_multipoles=[0, 2],
            model_multipoles=[0, 2, 4],
            fit_sedges=fit_edges,
        )


def test_linear_cap_counter_identity_survives_native_save_and_detects_disagreement(tmp_path):
    from pzbao.clustering import bind_pair_count_identity, pair_count_identity

    pytest.importorskip("pycorr")
    from pycorr import TwoPointCorrelationFunction

    identity = {"tracer": "LRG", "realization_id": 1, "cap": "SGC"}
    estimator = native_estimator(cap="SGC")
    bind_pair_count_identity(estimator, identity)
    path = tmp_path / "counts.npy"
    estimator.save(str(path))
    loaded = TwoPointCorrelationFunction.load(str(path))
    assert pair_count_identity(loaded) == identity
    # Type drift is an identity change even though Python says True == 1.
    loaded.D1D2.attrs["identity"]["realization_id"] = True
    with pytest.raises(ValueError, match="identit"):
        pair_count_identity(loaded)
