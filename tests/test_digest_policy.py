"""One snapshot digest and direct saved-array identities replace output hash chains."""

import json

import numpy as np
import pytest

from pzbao import run as driver
from pzbao.covariance import read_route_covariance, tune_shot_noise_rescaling
from tests import test_run

linear_run = test_run.linear_run


@pytest.fixture
def saved_covariance(linear_run):
    run = linear_run
    route = driver.routes_for(run["config"], "pilot")[0]
    root = run["root"] / "covariance" / "routes" / driver.route_name(route)
    root.mkdir(parents=True)
    config = run["config"]
    lo, hi = config["fitting"]["fit_edge_limits_mpc_h"]
    width = config["fitting"]["input_bin_width_mpc_h"]
    edges = np.arange(lo, hi + width / 2, width)
    poles = config["fitting"]["fit_surfaces"][route["tracer"]]["observed_multipoles"]
    count = len(config["analysis"]["mock_ids"])
    dimension = (len(edges) - 1) * len(poles)
    matrix = 6 * np.eye(dimension)
    arrays = {
        "covariance": matrix,
        "mean_covariance": matrix / count,
        "c4": matrix,
        "c3": np.zeros_like(matrix),
        "c2": np.zeros_like(matrix),
        "vectors": np.zeros((count, dimension)),
        "separations": np.tile((edges[:-1] + edges[1:]) / 2, (count, 1)),
        "fit_edges": edges,
        "multipoles": np.array(poles),
        "realization_ids": np.array(config["analysis"]["mock_ids"]),
    }
    metadata = {
        key: run["metadata"][key] for key in ("run_name", "code_commit", "analysis_config_sha256")
    }
    metadata.update(
        route=route,
        realization_ids=config["analysis"]["mock_ids"],
        units="Mpc/h",
        ordering="multipole_major",
        cap_integrations={"NGC": "north", "SGC": "south"},
        shot_noise={
            **tune_shot_noise_rescaling(
                {key: arrays[key] for key in ("c2", "c3", "c4")},
                np.cov(arrays["vectors"], rowvar=False, ddof=1),
                search_limits=config["fitting"]["covariance"]["shot_noise_rescaling"]["tuning"][
                    "search_limits"
                ],
                points=config["fitting"]["covariance"]["shot_noise_rescaling"]["tuning"][
                    "search_points"
                ],
            ),
            "realization_ids": config["analysis"]["mock_ids"],
        },
    )
    (root / "covariance.json").write_text(json.dumps(metadata))
    driver._write_yaml(root / "result.yaml", metadata)
    np.savez(root / "covariance.npz", **arrays)
    xi = config["measurement"]["xi"]
    native = np.arange(
        xi["s_min_mpc_h"], xi["s_max_mpc_h"] + xi["bin_width_mpc_h"] / 2, xi["bin_width_mpc_h"]
    )
    mu = np.linspace(-1, 1, xi["mu_bins"] + 1)
    np.savez(
        root / "window.npz",
        sedges=native,
        muedges=mu,
        wcounts=np.ones((len(native) - 1, len(mu) - 1)),
    )
    return run, route, root, arrays


def test_native_covariance_reads_without_any_output_hash_or_acceptance_flag(saved_covariance):
    run, route, root, arrays = saved_covariance
    result = read_route_covariance(run, route)
    np.testing.assert_array_equal(result["arrays"]["covariance"], arrays["covariance"])
    np.testing.assert_array_equal(result["arrays"]["mean_covariance"], arrays["covariance"] / 25)
    assert "sha256" not in str(
        {k: v for k, v in result["metadata"].items() if k != "analysis_config_sha256"}
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("source", "frozen_registry"),
        ("search_limits", [0.2, 4.0]),
        ("search_points", 41),
        ("realization_ids", list(range(24))),
        ("objective_curve", []),
        ("solution_on_search_boundary", False),
    ],
)
def test_reader_rejects_another_calibration_source_or_search(saved_covariance, key, value):
    run, route, root, _ = saved_covariance
    record = json.loads((root / "covariance.json").read_text())
    record["shot_noise"][key] = value
    (root / "covariance.json").write_text(json.dumps(record))
    with pytest.raises(ValueError):
        read_route_covariance(run, route)


@pytest.mark.parametrize(
    "key,value",
    [
        ("units", "Mpc"),
        ("ordering", "radial_major"),
        ("run_name", "another"),
        ("code_commit", "b" * 40),
        ("analysis_config_sha256", "c" * 64),
        ("realization_ids", [False, *range(1, 25)]),
        ("cap_integrations", {"NGC": "north"}),
        ("shot_noise", {"shot_noise_rescaling": 1.8}),
    ],
)
def test_wrong_direct_identity_refuses_even_a_positive_definite_matrix(
    saved_covariance, key, value
):
    run, route, root, _ = saved_covariance
    record = json.loads((root / "covariance.json").read_text())
    record[key] = value
    (root / "covariance.json").write_text(json.dumps(record))
    with pytest.raises(ValueError):
        read_route_covariance(run, route)


@pytest.mark.parametrize(
    "fault", ["missing", "nan", "asymmetric", "negative", "dimensions", "scaling"]
)
def test_invalid_saved_arrays_never_reach_a_likelihood(saved_covariance, fault):
    run, route, root, arrays = saved_covariance
    if fault == "missing":
        (root / "covariance.npz").unlink()
    else:
        if fault == "nan":
            arrays["covariance"][0, 0] = np.nan
        elif fault == "asymmetric":
            arrays["covariance"][0, 1] = 1.0
        elif fault == "negative":
            arrays["covariance"] *= -1
        elif fault == "dimensions":
            arrays["vectors"] = arrays["vectors"][:-1]
        elif fault == "scaling":
            arrays["mean_covariance"] *= 25
        np.savez(root / "covariance.npz", **arrays)
    with pytest.raises((ValueError, FileNotFoundError, np.linalg.LinAlgError)):
        read_route_covariance(run, route)


def test_empty_native_window_support_is_invalid(saved_covariance):
    run, route, root, _ = saved_covariance
    with np.load(root / "window.npz") as archive:
        values = dict(archive)
    values["wcounts"][48:52] = 0
    np.savez(root / "window.npz", **values)
    with pytest.raises(ValueError, match="support"):
        read_route_covariance(run, route)
