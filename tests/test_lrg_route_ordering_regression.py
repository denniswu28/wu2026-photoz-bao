"""Native measurement, cap integration and route publication preserve each covariance slot."""

import json
import numpy as np
import pytest

from pzbao import covariance, run as driver
from pzbao.clustering import bind_pair_count_identity, combine_cap_pair_counts
from tests.test_covariance import native_estimator
from tests.test_run import synthetic_scalar_grid


def publish_measurement_route(run, route, *, referenced_randoms=False):
    """Write real native counts for every configured realization, with direct unit metadata."""
    config = run["config"]
    units = [
        unit
        for unit in driver.stage_units("cluster", config, run["metadata"]["workload"])
        if unit["tracer"] == route["tracer"] and unit["condition"] == route["condition"]
    ]
    spec = config["measurement"]["xi"]
    edges = np.arange(
        spec["s_min_mpc_h"],
        spec["s_max_mpc_h"] + 0.5 * spec["bin_width_mpc_h"],
        spec["bin_width_mpc_h"],
    )
    name, state = route["measurement_bin"], route["reconstruction_state"]
    run_identity = {
        key: run["metadata"][key] for key in ("run_name", "code_commit", "analysis_config_sha256")
    }
    for unit in units:
        directory = run["root"] / "cluster" / "units" / unit["unit_id"]
        directory.mkdir(parents=True, exist_ok=True)
        entries, products, caps = [], [], {}
        for cap in ("NGC", "SGC"):
            identity = {
                **route,
                "realization_id": unit["realization_id"],
                "cap": cap,
                "analysis_config_sha256": run_identity["analysis_config_sha256"],
            }
            estimator = native_estimator(
                cap=cap,
                post=state == "postreconstruction",
                identifier=unit["realization_id"],
                radial_edges=edges,
                mu_bins=spec["mu_bins"],
            ).normalize(wnorm="D1D2")
            bind_pair_count_identity(estimator, identity)
            caps[cap] = estimator
            filename = f"{name}--{state}--{cap}--counts.npy"
            estimator.save(str(directory / filename))
            random_filename = f"{name}--{cap}--randoms.npz"
            positions = np.arange(90, dtype=float).reshape(30, 3) + (200 if cap == "SGC" else 100)
            np.savez(
                directory / random_filename,
                positions=positions,
                shifted_positions=positions + 1,
                weights=np.linspace(1.0, 2.0, 30),
                source_ids=np.arange(30, dtype="u8"),
            )
            if referenced_randoms:
                from pzbao.catalog import (
                    catalogue_identity,
                    catalogue_partition,
                    save_catalogue,
                    save_catalogue_reference,
                )

                selected_identity = catalogue_identity(
                    run,
                    tracer=route["tracer"],
                    condition=route["condition"],
                    mock_id=unit["realization_id"],
                    cap=cap,
                    role="random",
                    stage="select",
                )
                partition = catalogue_partition(route["tracer"], name, "random", config)
                catalogue = {
                    "SOURCE_ID": np.arange(30, dtype="u8"),
                    "Z": np.full(30, (partition["zmin"] + partition["zmax"]) / 2),
                    "X": positions[:, 0],
                    "Y": positions[:, 1],
                    "Z_CART": positions[:, 2],
                    "WEIGHT": np.linspace(1.0, 2.0, 30),
                    "WEIGHT_FKP": np.ones(30),
                }
                selected_name = f"{name}--{cap}--selected.npz"
                save_catalogue(directory, selected_name, catalogue, identity=selected_identity)
                shifted_identity = {**selected_identity, "stage": "reconstruct"}
                shifted_name = f"{name}--{cap}--shifted.npz"
                shifted = {
                    **catalogue,
                    "X_SHIFTED": positions[:, 0] + 1,
                    "Y_SHIFTED": positions[:, 1] + 1,
                    "Z_CART_SHIFTED": positions[:, 2] + 1,
                }
                save_catalogue_reference(
                    directory,
                    shifted_name,
                    identity=shifted_identity,
                    base={"path": str(directory / selected_name), "identity": selected_identity},
                    partition=partition,
                    shifted=shifted,
                )
                random_filename = {
                    "representation": "measurement_random_reference_v1",
                    "path": str(directory / shifted_name),
                    "identity": shifted_identity,
                    "partition": partition,
                    "weight_expression": config["measurement"]["weights"]["correlation_function"],
                }
            entries.append(
                {
                    "identity": identity,
                    "counts": filename,
                    "randoms": random_filename,
                    "units": "Mpc/h",
                    "data_count": 50 if cap == "NGC" else 20,
                    "random_count": 30,
                    "coordinate_state": "shifted" if state == "postreconstruction" else "unshifted",
                }
            )
            products.append({"path": filename, "format": "pycorr", "identity": identity})
        combined = combine_cap_pair_counts(caps)
        identity = {**entries[0]["identity"], "cap": "NGC+SGC"}
        bind_pair_count_identity(combined, identity)
        filename = f"{name}--{state}--combined-counts.npy"
        combined.save(str(directory / filename))
        entries.append({"identity": identity, "counts": filename, "units": "Mpc/h"})
        products.append({"path": filename, "format": "pycorr", "identity": identity})
        (directory / "measurements.json").write_text(
            json.dumps({"measurements": entries, "diagnostics": {}})
        )
        driver._write_yaml(
            directory / "result.yaml",
            {**run_identity, "stage": "cluster", "unit": unit, "products": products},
        )
    return edges


@pytest.mark.parametrize(
    "tracer,bin_name,condition",
    [
        ("BGS_BRIGHT-21.5", "BGS1", "sigma_0.003"),
        ("LRG", "LRG3", "sigma_0.003"),
        ("BGS_BRIGHT-21.5", "BGS2", "native"),
        ("LRG", "LRG1", "native"),
    ],
)
@pytest.mark.parametrize("state", ["prereconstruction", "postreconstruction"])
@pytest.mark.parametrize("referenced_randoms", [False, True])
def test_linear_native_route_roundtrip_keeps_both_caps_all_mocks_and_exact_matrix_order(
    tmp_path, monkeypatch, tracer, bin_name, condition, state, referenced_randoms
):
    pytest.importorskip("pycorr")
    synthetic_scalar_grid(monkeypatch)
    monkeypatch.setattr(
        driver, "_code_identity", lambda: {"code_commit": "b" * 40, "repository": str(tmp_path)}
    )
    run = driver.create_run(tmp_path, "roundtrip", workload="full_grid", catalog_root=tmp_path)
    route = {
        "tracer": tracer,
        "measurement_bin": bin_name,
        "condition": condition,
        "reconstruction_state": state,
    }
    publish_measurement_route(run, route, referenced_randoms=referenced_randoms)
    seen, raw_matrices = [], {}

    def integrate(**kwargs):
        destination = kwargs["destination"]
        destination.mkdir()
        cap = "NGC" if kwargs["data_count"] == 50 else "SGC"
        seen.append(cap)
        expected_coordinates = np.arange(90).reshape(30, 3) + (100 if cap == "NGC" else 200)
        if state == "postreconstruction":
            expected_coordinates += 1
        np.testing.assert_array_equal(kwargs["random_positions"], expected_coordinates)
        np.testing.assert_array_equal(kwargs["random_weights"], np.linspace(1.0, 2.0, 30))
        assert (
            kwargs["alpha"]
            == covariance.fresh_shot_noise_controls(run["config"])["search_limits"][0]
        )
        factor = int(
            run["config"]["fitting"]["input_bin_width_mpc_h"]
            / run["config"]["measurement"]["xi"]["bin_width_mpc_h"]
        )
        tables = [
            covariance.rascalc_native_table(
                native_estimator(
                    cap=cap,
                    post=state == "postreconstruction",
                    radial_edges=np.arange(0.0, 201.0, 1.0),
                    mu_bins=run["config"]["measurement"]["xi"]["mu_bins"],
                    identifier=i,
                )[::factor]
            )
            for i in run["config"]["analysis"]["mock_ids"]
        ]
        np.testing.assert_allclose(
            kwargs["xi_table"][2], np.mean([table[2] for table in tables], axis=0)
        )
        raw_edges = kwargs["xi_table"][3]
        amplitudes = np.array(
            [1 + 100 * pole + radial for radial in range(len(raw_edges) - 1) for pole in range(3)],
            dtype=float,
        ) * (1 if cap == "NGC" else 2)
        c4 = 0.25 * np.outer(amplitudes, amplitudes)
        np.fill_diagonal(c4, amplitudes**2)
        components = {
            "c2": np.eye(len(amplitudes)) * 0.1,
            "c3": np.eye(len(amplitudes)) * 0.2,
            "c4": c4,
        }
        raw_matrices[cap] = components
        subsamples = {
            key: np.array([value * scale for scale in np.linspace(0.95, 1.05, 16)])
            for key, value in components.items()
        }
        archive = destination / "raw.npz"
        np.savez(archive, **components)
        return {
            "components": components,
            "subsamples": subsamples,
            "archive": archive,
            "raw_integration_completed": True,
            "ordering": "radial_major",
            "multipoles": [0, 2, 4],
            "separation_edges_mpc_h": raw_edges.tolist(),
            "n_loops": kwargs["n_loops"],
        }

    monkeypatch.setattr(covariance, "integrate_cap", integrate)
    directories = {}
    for cap in ("NGC", "SGC"):
        directory = run["root"] / "covariance" / "units" / (driver.route_name(route) + "--" + cap)
        directory.mkdir(parents=True)
        covariance.covariance_cap_stage(run, {**route, "cap": cap}, directory, nthread=1)
        directories[cap] = directory
    directory = run["root"] / "covariance" / "routes" / driver.route_name(route)
    directory.mkdir(parents=True)
    result = covariance.assemble_route(
        run,
        route,
        directories,
        directory,
        integrate_next=lambda *args: pytest.fail("valid required matrix must not escalate"),
    )
    driver._write_yaml(
        directory / "result.yaml",
        {
            **result,
            "route": route,
            **{
                key: run["metadata"][key]
                for key in ("run_name", "code_commit", "analysis_config_sha256")
            },
        },
    )
    saved = covariance.read_route_covariance(driver.load_run(run["root"]), route)
    arrays = saved["arrays"]
    assert seen == ["NGC", "SGC"]
    assert run["config"]["fitting"]["fit_surfaces"][tracer]["observed_multipoles"] == [0, 2]
    dimension = 52
    assert arrays["vectors"].shape == (25, dimension)
    alpha = saved["metadata"]["shot_noise"]["shot_noise_rescaling"]
    calibration = saved["metadata"]["shot_noise"]
    assert calibration["source"] == "route_mock_scatter"
    assert calibration["realization_ids"] == list(range(25))
    old = covariance.fixed_shot_noise_entry(run["registry"], route)
    if old is not None:
        assert alpha != old["shot_noise_rescaling"]
    controls = covariance.fresh_shot_noise_controls(run["config"])
    grid = np.linspace(*controls["search_limits"], controls["search_points"])
    empirical = np.cov(arrays["vectors"], rowvar=False, ddof=1)
    # Independently evaluate saved candidates against the actual serialized measurement scatter.
    for index in (0, len(grid) // 2, len(grid) - 1):
        model = arrays["c4"] + grid[index] * arrays["c3"] + grid[index] ** 2 * arrays["c2"]
        expected_objective = np.linalg.slogdet(model)[1] + np.trace(
            np.linalg.solve(model, empirical)
        )
        assert calibration["objective_curve"][index] == pytest.approx(expected_objective)
    # Build the expected raw-to-fitted operators independently with explicit Legendre
    # integration, preserving all three incoming multipoles and the angular cap weights.
    from numpy.polynomial.legendre import Legendre, leggauss

    nodes, node_weights = leggauss(12)
    muedges = np.linspace(0.0, 1.0, run["config"]["measurement"]["xi"]["mu_bins"] // 2 + 1)
    integral = np.array(
        [
            [
                (hi - lo)
                / 2
                * np.dot(node_weights, Legendre.basis(ell)((nodes + 1) * (hi - lo) / 2 + lo))
                for lo, hi in zip(muedges[:-1], muedges[1:])
            ]
            for ell in (0, 2, 4)
        ]
    )
    expected = np.zeros((dimension, dimension))
    for cap in ("NGC", "SGC"):
        weight = arrays[f"weights_{cap}"]
        operator = np.zeros((dimension, 150))
        for row in range(dimension):
            ell_index, radial = divmod(row, 26)
            for incoming in range(3):
                operator[row, 3 * (radial + 12) + incoming] = np.sum(
                    (4 * ell_index + 1)
                    * integral[ell_index]
                    * weight[radial]
                    * integral[incoming]
                    / np.diff(muedges)
                )
        np.testing.assert_allclose(arrays[f"W_{cap}"], operator, atol=1e-13)
        raw = raw_matrices[cap]
        expected += operator @ (raw["c4"] + alpha * raw["c3"] + alpha**2 * raw["c2"]) @ operator.T
    np.testing.assert_allclose(arrays["covariance"], expected, rtol=1e-12, atol=1e-9)
    np.testing.assert_array_equal(arrays["mean_covariance"], arrays["covariance"] / 25)
    assert saved["metadata"]["cross_cap_covariance"] == "neglected"
    metadata_path = directory / "covariance.json"
    changed = json.loads(metadata_path.read_text())
    changed["realization_ids"][1] = True
    metadata_path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="complete ensemble"):
        covariance.read_route_covariance(run, route)
