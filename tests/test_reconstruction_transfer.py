"""Value, identity and serialization contracts for exclusive reconstruction transfer."""

import copy
import json

import numpy as np
import pytest
import yaml
from click.testing import CliRunner

from pzbao.config import load_analysis_config
from tests.legacy_config import load_analysis_config as load_legacy_config
from pzbao.config.transfer import transfer_run_configs
from pzbao.config.schema_validation import validate_analysis_config
from pzbao.config.transfer import transfer_matrix
from pzbao.selection import (
    assign_exclusive_classes,
    forward_photoz_counts,
    forward_radial_model,
    joint_class_budget,
    matching_exclusive_randoms,
    select_exclusive_targets,
)
from pzbao.reconstruction import (
    apply_displacement_field,
    configured_reconstruction_settings,
    displacement_field_identity,
    reconstruction_mask_family,
)


def parent():
    z = np.repeat([0.35, 0.5, 0.7, 0.9, 1.05], 10)
    return {
        "SOURCE_ID": np.arange(len(z), dtype=np.uint64),
        "RA": np.arange(len(z), dtype=float),
        "DEC": np.zeros(len(z)),
        "Z": z,
    }


def test_joint_scaling_floors_each_class_and_leaves_remainders():
    budget = joint_class_budget([[6, 0], [4, 0]], [[7, 0], [9, 0]])
    assert budget["scale_by_shell"] == [0.7, 1.0]
    assert budget["allocated_parent_counts"] == [[4, 0], [2, 0]]
    assert budget["unused_parent_counts"] == [1, 0]
    assert joint_class_budget([[3], [2]], [[10], [7]])["allocated_parent_counts"] == [[3], [2]]
    with pytest.raises(ValueError):
        joint_class_budget([[1]], [[1.5]])
    with pytest.raises(ValueError):
        joint_class_budget([[np.nan]], [[10]])


def assignments(catalogue, config):
    return assign_exclusive_classes(
        catalogue,
        class_labels=["sigma_0.003", "sigma_0.01", "sigma_0.03"],
        shell_edges=[0.4, 0.6, 0.8, 1.0],
        allocated_counts=[[2, 2, 2], [3, 3, 3], [4, 4, 4]],
        seed=config["selection"]["stable_randomness"]["seed"],
        config=config,
    )


def test_partition_is_disjoint_reproducible_and_independent_of_photoz_draws():
    config, catalogue = load_analysis_config(), parent()
    assignment, record = assignments(catalogue, config)
    order = np.arange(len(assignment))[::-1]
    reordered = {name: values[order] for name, values in catalogue.items()}
    reordered["PHOTOZ_DEVIATE"] = np.full(len(order), 999.0)
    shuffled, _ = assignments(reordered, config)
    np.testing.assert_array_equal(shuffled[::-1], assignment)
    np.testing.assert_array_equal(record["class_probabilities"], [[0.2] * 3, [0.3] * 3, [0.4] * 3])
    for label, expected in enumerate([2, 3, 4]):
        for z in (0.5, 0.7, 0.9):
            assert np.count_nonzero((assignment == label) & (catalogue["Z"] == z)) == expected
    assert "PHOTOZ_DEVIATE" not in catalogue
    duplicate = {key: np.append(values, values[0]) for key, values in catalogue.items()}
    with pytest.raises(ValueError, match="duplicate parent"):
        assignments(duplicate, config)
    duplicate["SOURCE_ID"][-1] = 99999
    with pytest.raises(ValueError, match="physical row"):
        assignments(duplicate, config)


def test_exclusive_targets_save_fixed_identities_errors_and_complete_migration(tmp_path):
    config, catalogue = load_analysis_config(), parent()
    assigned, _ = assignments(catalogue, config)
    labels = ["sigma_0.003", "sigma_0.01", "sigma_0.03"]
    targets, diagnostics = select_exclusive_targets(
        catalogue,
        assigned,
        class_labels=labels,
        tracer="LRG",
        config=config,
    )
    identities = []
    for index, label in enumerate(labels):
        target = targets[label]
        path = tmp_path / f"{label}.npz"
        np.savez(path, **target)
        with np.load(path, allow_pickle=False) as stored:
            np.testing.assert_array_equal(stored["SOURCE_ID"], target["SOURCE_ID"])
            np.testing.assert_array_equal(stored["PHOTOZ_DEVIATE"], target["PHOTOZ_DEVIATE"])
            np.testing.assert_array_equal(stored["Z_OBS"], target["Z_OBS"])
        identities.extend(target["SOURCE_ID"].tolist())
        migration = np.asarray(diagnostics[label]["parent_to_observed_counts_with_exterior"])
        assert migration.sum() == np.count_nonzero(assigned == index)
        assert migration[:, 1:-1].sum() == len(target["Z_OBS"])
    assert len(set(identities)) == len(identities)
    assert "Z_OBS" not in catalogue


def test_forward_kernel_counts_conserve_mass_and_zero_error_half_open_edges():
    result = forward_photoz_counts([0.4, 0.6, 0.8], [2, 3, 5], [0.4, 0.6, 0.8], 0)
    assert result["predicted_final_counts"] == [2, 3]
    assert result["predicted_above_count"] == 5
    result = forward_photoz_counts([0.5], [100], [0.35, 0.5, 0.65], 0.1)
    np.testing.assert_allclose(result["predicted_final_counts"], [34.13447460685429] * 2)
    assert sum(result["predicted_final_counts"]) + result["predicted_below_count"] + result[
        "predicted_above_count"
    ] == pytest.approx(100)
    json.dumps(result, allow_nan=False)


def test_photoz_response_is_not_nonlinear_damping_and_transfer_centres_fail_closed():
    from pzbao.theory import (
        selection_weighted_photoz_response,
        sigma_r_mpc_h,
        resolve_transfer_damping_prior,
    )

    config = load_analysis_config()
    z, expansion, weights = np.array([0.6, 0.8]), np.array([1.4, 1.6]), np.array([1.0, 3.0])
    k_mu = np.array([0.0, 0.01, 0.1])
    radii = sigma_r_mpc_h(
        z, np.full_like(z, 0.01), expansion, config["physics"]["speed_of_light_km_s"]
    )
    expected = np.exp(-((k_mu[:, None] * radii) ** 2)) @ (weights / 4)
    result = selection_weighted_photoz_response(k_mu, z, weights, 0.01, expansion, config=config)
    np.testing.assert_allclose(result, expected)
    np.testing.assert_array_equal(
        selection_weighted_photoz_response(k_mu, z, weights, 0, expansion, config=config),
        np.ones(3),
    )
    with pytest.raises(ValueError, match="reviewed source-target"):
        resolve_transfer_damping_prior(
            "LRG",
            "postreconstruction",
            postreconstruction_sigma_perp_artifact={"analytic_center_mpc_h": 4.0},
            config=config,
        )


@pytest.mark.parametrize("reuse_inputs", [False, True])
def test_forward_random_selection_preserves_rows_and_rejects_altered_prediction(
    tmp_path, reuse_inputs
):
    from pzbao.selection import _InputPreparationCache

    config = load_analysis_config()
    input_cache = _InputPreparationCache(config) if reuse_inputs else None
    parent_model = {
        "calibration_mock_ids": [0, 1],
        "sample_identity": "fixture-parent-class",
        "convergence_evidence": {"converged": True},
        "z_edges": [0.4, 0.6, 0.8, 1.0],
        "mean_counts": [2.0, 3.0, 5.0],
        "target_probability_mass": [0.2, 0.3, 0.5],
    }
    radial = forward_radial_model(
        parent_model,
        observed_edges=[0.4, 0.6, 0.8, 1.0],
        sigma0=0,
        sample_identity="fixture-final-class",
    )
    path = tmp_path / "forward.json"
    path.write_text(json.dumps(radial, allow_nan=False))
    radial = json.loads(path.read_text())
    np.testing.assert_array_equal(radial["predicted_final_counts"], [2.0, 3.0, 5.0])
    randoms = {
        "SOURCE_ID": np.arange(600, dtype=np.uint64),
        "RA": np.arange(600) / 10,
        "DEC": np.zeros(600),
        "Z": np.repeat([0.5, 0.7, 0.9], 200),
    }
    before = {name: values.copy() for name, values in randoms.items()}
    for values in randoms.values():
        values.flags.writeable = False
    attained = {name: {"attained_nbar_h3_mpc3": 1e-4} for name in ("LRG1", "LRG2", "LRG3")}
    selected, diagnostics = matching_exclusive_randoms(
        randoms,
        radial,
        data_count=2,
        expected_sample_identity="fixture-final-class",
        tracer="LRG",
        attained_by_bin=attained,
        config=config,
        input_cache=input_cache,
    )
    assert len(selected["Z"]) == 100
    assert diagnostics["target_counts_by_bin"] == [20, 30, 50]
    np.testing.assert_array_equal(selected["Z"], randoms["Z"][selected["SOURCE_ID"]])
    np.testing.assert_allclose(selected["WEIGHT_FKP"], 0.5)
    assert "PHOTOZ_DEVIATE" not in selected and "Z_OBS" not in selected
    for name in before:
        np.testing.assert_array_equal(randoms[name], before[name])
        assert not np.shares_memory(selected[name], randoms[name])
        assert selected[name].flags.writeable
        if input_cache is not None:
            cached = input_cache.normalized(randoms, role="random", config=config)
            assert not np.shares_memory(selected[name], cached[name])
    selected["Z"][0] = 0.0
    np.testing.assert_array_equal(randoms["Z"], before["Z"])
    if input_cache is not None:
        np.testing.assert_array_equal(cached["Z"], before["Z"])
    altered = copy.deepcopy(radial)
    altered["target_probability_mass"] = [0.3, 0.3, 0.4]
    with pytest.raises(ValueError, match="differs from its parent prediction"):
        matching_exclusive_randoms(
            randoms,
            altered,
            data_count=2,
            expected_sample_identity="fixture-final-class",
            tracer="LRG",
            attained_by_bin=attained,
            config=config,
            input_cache=input_cache,
        )


def test_matrix_has_unique_fields_and_separate_fit_checks():
    config = load_analysis_config()
    matrix = transfer_matrix(config)
    assert (len(matrix["fields"]), len(matrix["measurements"]), len(matrix["fits"])) == (
        35,
        80,
        120,
    )
    assert len({row["measurement_id"] for row in matrix["measurements"]}) == 80
    for row in matrix["fits"]:
        assert not (row["broadband"] == "pcs2" and row["damping"] == "common_flat_sensitivity")
    for field in matrix["fields"]:
        targets = [row for row in matrix["measurements"] if row["field_id"] == field["field_id"]]
        assert len(targets) == (3 if field["source_class"] == "sigma_0.003" else 1)
    pilot = transfer_matrix(config, pilot=True)
    assert len(pilot["fields"]) == 9 and len(pilot["measurements"]) == 24
    assert pilot["mock_ids"] == [0, 1, 2]
    assert len(pilot["allocation_mock_ids"]) == 25
    from pzbao.run import main

    result = CliRunner().invoke(main, ["transfer-matrix", "--pilot"])
    assert result.exit_code == 0, result.output
    assert yaml.safe_load(result.output) == pilot
    old = copy.deepcopy(config)
    del old["measurement"]["reconstruction"]["exploratory_transfer"]
    validate_analysis_config(old)
    wrong = copy.deepcopy(config)
    wrong["measurement"]["reconstruction"]["exploratory_transfer"]["source_class"] = "sigma_0.03"
    with pytest.raises(ValueError, match="lowest redshift uncertainty"):
        validate_analysis_config(wrong)
    # Frozen 2.11-T1 snapshots have no self radius checks; 2.11-T2 requires them.
    frozen = copy.deepcopy(config)
    design = frozen["measurement"]["reconstruction"]["exploratory_transfer"]
    del design["self_smoothing_checks_mpc_h"]
    design["specification_revision"] = "2.11-T1"
    validate_analysis_config(frozen)
    assert transfer_matrix(frozen)["execution_counts"]["target_reconstruction_units"] == 550
    for revision, drop in (("2.11-T1", False), ("2.11-T2", True)):
        mixed = copy.deepcopy(config)
        design = mixed["measurement"]["reconstruction"]["exploratory_transfer"]
        design["specification_revision"] = revision
        if drop:
            del design["self_smoothing_checks_mpc_h"]
        with pytest.raises(ValueError, match="transfer specification_revision"):
            validate_analysis_config(mixed)
    duplicate = copy.deepcopy(config)
    duplicate["measurement"]["reconstruction"]["exploratory_transfer"][
        "self_smoothing_checks_mpc_h"
    ] = [25.0]
    with pytest.raises(ValueError, match="self radii must not duplicate"):
        validate_analysis_config(duplicate)


def test_executable_transfer_yaml_roundtrip_and_fit_only_reuse(tmp_path):
    from pzbao.config.transfer import transfer_run_configs
    from pzbao.config import derive_inference_config, validate_inference_config
    from pzbao.run import main, route_name, routes_for, stage_units
    from pzbao.config.stages import experiment_document

    configs = transfer_run_configs(load_analysis_config())
    assert len(configs) == 5
    assert sum(len(routes_for(config, "transfer")) for config in configs.values()) == 80
    checks = []
    for name, config in configs.items():
        if name.endswith("R25"):
            for definition in ("transfer_pcs2", "transfer_flat"):
                original = copy.deepcopy(config)
                variant = derive_inference_config(config, definition)
                validate_inference_config(config, variant, definition)
                assert config == original
                path = tmp_path / f"{name}--{definition}.yaml"
                path.write_text(yaml.safe_dump(experiment_document(variant)))
                assert load_analysis_config(path) == variant
                if definition == "transfer_pcs2":
                    expected = ["bl0_0", "bl0_2", "al2_0", "al2_1", "bl2_0", "bl2_2"]
                    nuisance = variant["fitting"]["linear_nuisance"]
                    assert nuisance["allowed_basenames_by_model"] == {"pcs2": expected}
                    assert all(
                        names == expected
                        for names in nuisance["varied_basenames_by_tracer"].values()
                    )
                checks.extend(
                    stage_units("sample", variant, "transfer", inference_definition=definition)
                )
    # PCS2 on the 20 low-precision routes; flat on every route of both R25 runs (revision 2.19).
    assert len(checks) == 60
    destination = tmp_path / "yaml"
    result = CliRunner().invoke(main, ["transfer-yamls", "--output-dir", str(destination)])
    assert result.exit_code == 0, (result.output, result.exception)
    for name, expected in configs.items():
        assert load_analysis_config(destination / (name + ".yaml")) == expected
        assert (
            yaml.safe_load((destination / (name + ".yaml")).read_text())["cluster"]["statistics"]
            == "xi"
        )
    pilot = transfer_run_configs(load_analysis_config(), pilot=True)
    assert sum(len(routes_for(config, "transfer")) for config in pilot.values()) == 24
    assert all(
        config["analysis"]["workloads"]["transfer"]["mock_ids"] == [0, 1, 2]
        for config in pilot.values()
    )
    original = configs["self--R25"]
    altered = derive_inference_config(original, "transfer_pcs2")
    altered["physics"]["speed_of_light_km_s"] += 1
    with pytest.raises(ValueError, match="more than"):
        validate_inference_config(original, altered, "transfer_pcs2")
    altered = derive_inference_config(original, "transfer_pcs2")
    altered["fitting"]["linear_nuisance"]["finite_difference_step"] *= 2
    with pytest.raises(ValueError, match="more than"):
        validate_inference_config(original, altered, "transfer_pcs2")
    for radius in (15, 20):
        restricted = configs[f"sigma_0.003--R{radius}"]
        with pytest.raises(ValueError, match="common-radius"):
            derive_inference_config(restricted, "transfer_pcs2")
        altered = derive_inference_config(configs["sigma_0.003--R25"], "transfer_pcs2")
        altered["measurement"]["reconstruction"]["smoothing_values_mpc_h"] = [radius]
        altered["analysis"]["workloads"]["transfer"] = restricted["analysis"]["workloads"][
            "transfer"
        ]
        with pytest.raises(ValueError, match="common-radius"):
            validate_analysis_config(altered)
    combined = derive_inference_config(original, "transfer_pcs2")
    combined["measurement"]["reconstruction"]["transfer_run"]["prior"] = "common_flat_sensitivity"
    with pytest.raises(ValueError, match="power3 fit, separate from the PCS2 check"):
        validate_analysis_config(combined)
    # The flat prior is the headline at every configured radius; PCS2 stays a common-radius check.
    for name in ("self--R15", "sigma_0.003--R15", "sigma_0.003--R20"):
        flat = derive_inference_config(configs[name], "transfer_flat")
        validate_inference_config(configs[name], flat, "transfer_flat")
        transfer = flat["measurement"]["reconstruction"]["transfer_run"]
        assert transfer["prior"] == "common_flat_sensitivity"

    # The flat headline and the Gaussian check fit the same routes, pre-reconstruction included.
    def fitted_routes(name, definition):
        units = stage_units("fit", configs[name], "transfer", inference_definition=definition)
        return {route_name(unit) for unit in units}

    for name, count in (("self--R25", 30), ("self--R15", 15), ("sigma_0.003--R25", 10)):
        assert len(fitted_routes(name, "transfer_flat")) == count
        assert fitted_routes(name, "transfer_flat") == fitted_routes(name, "transfer_reference")
    assert len(fitted_routes("self--R25", "transfer_pcs2")) == 10
    independent = load_analysis_config()
    independent["fitting"]["broadband_model"] = "pcs2"
    with pytest.raises(ValueError, match="unsupported broadband"):
        validate_analysis_config(independent)


def test_native_exclusive_selection_saves_fixed_targets_and_four_density_definitions(
    tmp_path, monkeypatch
):
    from pzbao import selection
    from pzbao.config.transfer import transfer_run_configs
    from pzbao.catalog import read_catalogue

    config = transfer_run_configs(load_analysis_config(), pilot=True)["self--R25"]
    # Small identified allocation ensemble for the serialization fixture only.
    config["analysis"]["mock_ids"] = [0, 1, 2]
    config["selection"]["radial_randoms"]["support_bin_width_redshift"] = 0.01
    tracer = "BGS_BRIGHT-21.5"

    class Cosmology:
        def comoving_radial_distance(self, z):
            return 3000 * np.asarray(z)

        def growth_rate(self, z):
            return 0.8

    monkeypatch.setattr(selection, "build_mock_cosmology", lambda config: Cosmology())
    sources = [
        {"role": "reservoir", "mock_id": mock, "cap": cap, "tracer": tracer}
        for mock in range(3)
        for cap in ("NGC", "SGC")
    ]
    monkeypatch.setattr(selection, "read_catalog_registration", lambda *a: {"sources": sources})

    pools = []

    def raw_parent(*args, mock_id, cap, include_randoms=True, parallel=None, **kwargs):
        pools.append(parallel)
        rng = np.random.default_rng(200 + mock_id + (cap == "SGC"))
        z = np.concatenate([rng.uniform(0.08, 0.42, 800), [0.101, 0.399]])

        def catalog(values, offset):
            ids = np.arange(len(values), dtype="u8") + offset + mock_id * 100000
            return {
                "SOURCE_ID": ids,
                "TARGETID": ids.copy(),
                "Z": values,
                "RA": np.linspace(20, 100, len(values)),
                "DEC": np.linspace(-20, 20, len(values)),
                "WEIGHT": 1.0 + 0.1 * (ids % 5),
            }

        data = catalog(z, 0)
        return {
            "data": data,
            "reservoir": {key: value[:0] for key, value in data.items()},
            "random": (
                catalog(np.linspace(0.1000001, 0.3999999, 60000), 1000000)
                if include_randoms
                else None
            ),
        }

    monkeypatch.setattr(selection, "_read_parent", raw_parent)
    unit = {"tracer": tracer, "realization_ids": [0, 1, 2]}
    saved = []
    for source in ("self", "sigma_0.003"):
        variant = copy.deepcopy(config)
        variant["measurement"]["reconstruction"]["transfer_run"]["source_class"] = source
        destination = tmp_path / source
        destination.mkdir()
        run = {
            "config": variant,
            "metadata": {
                "run_name": source,
                "code_commit": "fixture",
                "analysis_config_sha256": source,
                "workload": "transfer",
            },
        }
        pools.clear()
        selection.select_stage(run, unit, destination, cpus=1 if source == "self" else 2)
        assert len(pools) == 18  # Three passes over three mocks and two caps.
        assert all(pool is pools[0] for pool in pools)
        assert (pools[0] is None) == (source == "self")
        preparation = json.loads((destination / "preparation.json").read_text())
        row = preparation["conditions"]["sigma_0.01"]["NGC"]
        assert set(row["densities"]["BGS1"]) == {
            "forecast_h3_mpc3",
            "attainable_parent_h3_mpc3",
            "predicted_final_h3_mpc3",
            "measured_final_h3_mpc3",
        }
        assert set(preparation["geometry"]["NGC"]) == {"BGS1", "BGS2"}
        for cap_geometry in preparation["geometry"].values():
            for geometry in cap_geometry.values():
                assert np.all(np.asarray(geometry["boxsize_mpc_h"]) > 0)
        ids = []
        catalogs = {}
        for entry in preparation["catalogues"]:
            catalogue = read_catalogue(destination / entry["path"], identity=entry["identity"])
            catalogs[entry["path"]] = catalogue
            column = "Z_OBS" if entry["identity"]["role"] == "data" else "Z"
            distance = 3000 * catalogue[column]
            ra, dec = np.deg2rad(catalogue["RA"]), np.deg2rad(catalogue["DEC"])
            for name, expected in zip(
                ("X", "Y", "Z_CART"),
                (
                    distance * np.cos(dec) * np.cos(ra),
                    distance * np.cos(dec) * np.sin(ra),
                    distance * np.sin(dec),
                ),
            ):
                np.testing.assert_allclose(catalogue[name], expected, rtol=1e-12, atol=1e-12)
            if (
                entry["identity"]["role"] == "data"
                and entry["identity"]["cap"] == "NGC"
                and entry["identity"]["realization_id"] == 0
            ):
                ids.extend(catalogue["SOURCE_ID"].tolist())
                group = preparation["conditions"][entry["identity"]["condition"]]["NGC"]
                moments = group["selection_diagnostics_by_mock"][0]["reconstruction_weight_moments"]
                for item in variant["tracers"][tracer]["measurement_bins"]:
                    mask = (catalogue["Z_OBS"] >= item["zmin"]) & (
                        catalogue["Z_OBS"] < item["zmax"]
                    )
                    weights = catalogue["WEIGHT"][mask]
                    actual = moments[item["name"]]
                    assert actual["count"] == len(weights)
                    assert actual["sum_w"] == pytest.approx(weights.sum())
                    assert actual["sum_w2"] == pytest.approx(np.square(weights).sum())
                    assert not np.isclose(
                        actual["sum_w"], (weights * catalogue["WEIGHT_FKP"][mask]).sum()
                    )
        assert len(ids) == len(set(ids))
        saved.append(catalogs)
    assert saved[0].keys() == saved[1].keys()
    for name in saved[0]:
        for key in saved[0][name]:
            np.testing.assert_array_equal(saved[0][name][key], saved[1][name][key])


def test_transfer_centre_contexts_can_read_one_completed_tracer(tmp_path, monkeypatch):
    from pzbao import run as driver, selection, theory
    from pzbao.config.transfer import transfer_run_configs

    config = transfer_run_configs(load_analysis_config(), pilot=True)["self--R25"]
    run = {"config": config}
    monkeypatch.setattr(driver, "load_run", lambda path: run)
    reads = []

    def preparation(actual_run, tracer):
        assert actual_run is run
        if tracer != "LRG":
            raise FileNotFoundError("BGS selection is not complete")
        reads.append(tracer)
        return {"tracer": tracer}

    def context(actual_config, prepared, **route):
        assert actual_config is config
        assert prepared == {"tracer": route["tracer"]}
        return route

    monkeypatch.setattr(selection, "read_preparation", preparation)
    monkeypatch.setattr(theory, "transfer_damping_context", context)
    result = CliRunner().invoke(
        driver.main, ["transfer-centre-contexts", "--run", str(tmp_path), "--tracer", "LRG"]
    )
    assert result.exit_code == 0, (result.output, result.exception)
    artifact = yaml.safe_load(result.output)
    assert reads == ["LRG"]
    assert artifact["review_reference"] is None
    assert {
        (row["context"]["measurement_bin"], row["context"]["target_class"])
        for row in artifact["records"]
    } == {
        (name, label)
        for name in ("LRG2", "LRG3")
        for label in ("sigma_0.003", "sigma_0.01", "sigma_0.03")
    }
    assert all(row["centers_mpc_h"] is None for row in artifact["records"])
    # Omitting the operational selector still requires every configured tracer.
    result = CliRunner().invoke(driver.main, ["transfer-centre-contexts", "--run", str(tmp_path)])
    assert isinstance(result.exception, FileNotFoundError)


def test_reviewed_transfer_centres_bind_context_and_exclude_recovery_calibration():
    from pzbao.theory import resolve_transfer_damping_prior
    from pzbao.config.transfer import transfer_run_configs

    config = transfer_run_configs(load_analysis_config())["sigma_0.003--R25"]
    context = {"source_density_h3_mpc3": 1e-4, "target_class": "sigma_0.03"}
    record = {
        "context": context,
        "centers_mpc_h": {"sigma_perp": 3.2, "sigma_parallel": 5.5, "sigma_streaming": 2.0},
        "evidence": {
            "kind": "independent_calibration",
            "reference": "fixture-only",
            "mock_ids": [100, 101],
        },
    }
    transfer = config["measurement"]["reconstruction"]["transfer_run"]
    transfer["damping_centres"] = {
        "units": "Mpc/h",
        "review_reference": "fixture-only-reference",
        "records": [record],
    }
    kwargs = dict(
        context=context, sigma_z_over_one_plus_z=0.03, z_eff=0.9, e_z_eff=1.6, config=config
    )
    prior = resolve_transfer_damping_prior("LRG", "postreconstruction", **kwargs)
    assert prior["centers_mpc_h"] == record["centers_mpc_h"]
    assert prior["widths_mpc_h"] == {"sigma_perp": 1, "sigma_parallel": 2, "sigma_streaming": 2}
    assert prior["sigma_parallel_photoz_mpc_h"] > prior["centers_mpc_h"]["sigma_parallel"]
    with pytest.raises(ValueError, match="match this source-target"):
        resolve_transfer_damping_prior(
            "LRG",
            "postreconstruction",
            **{**kwargs, "context": {**context, "source_density_h3_mpc3": 2e-4}},
        )
    transfer["prior"] = "common_flat_sensitivity"
    flat = resolve_transfer_damping_prior("LRG", "postreconstruction", **kwargs)
    assert all(value == [0, 50] for value in flat["bounds_mpc_h"].values())
    assert len(flat["unpriored_parameters"]) == 3
    # The flat headline flattens every route, before reconstruction too (revision 2.19); the
    # Gaussian check keeps the baseline there.
    from pzbao.theory import TRANSFER_FLAT_DAMPING_SCOPE

    assert flat["scope"] == TRANSFER_FLAT_DAMPING_SCOPE
    pre = resolve_transfer_damping_prior("LRG", "prereconstruction", **kwargs)
    assert pre["scope"] == TRANSFER_FLAT_DAMPING_SCOPE and len(pre["unpriored_parameters"]) == 3
    assert all(value == [0, 50] for value in pre["bounds_mpc_h"].values())
    gaussian = copy.deepcopy(config)
    del gaussian["measurement"]["reconstruction"]["transfer_run"]["prior"]
    baseline = resolve_transfer_damping_prior(
        "LRG", "prereconstruction", **{**kwargs, "config": gaussian}
    )
    assert baseline["unpriored_parameters"] == []
    for malformed in (["0", "1"], [0.0, 1.0], [True], [-1], "100,101", [{"id": 100}]):
        record["evidence"]["mock_ids"] = malformed
        with pytest.raises(ValueError, match="integer mock IDs"):
            resolve_transfer_damping_prior("LRG", "postreconstruction", **kwargs)
    record["evidence"]["mock_ids"] = [0, 1]
    with pytest.raises(ValueError, match="independent"):
        resolve_transfer_damping_prior("LRG", "postreconstruction", **kwargs)


def field_inputs():
    config = load_analysis_config()
    data = {
        "SOURCE_ID": np.array([1, 2], dtype=np.uint64),
        "X": np.array([100.0, 101.0]),
        "Y": np.array([1.0, 2.0]),
        "Z_CART": np.array([1.0, 2.0]),
        "WEIGHT": np.ones(2),
    }
    randoms = {**data, "SOURCE_ID": np.array([3, 4], dtype=np.uint64)}
    family = reconstruction_mask_family(tracer="LRG", cap="NGC", smoothing_radius_mpc_h=25)
    settings = {
        **configured_reconstruction_settings(config),
        "bias": 2.0,
        "growth_rate": 0.8,
        "smoothing_radius_mpc_h": 25.0,
        "convention": "recsym",
        "mask_family": family,
        "common_geometry": {
            "nmesh": [4, 4, 4],
            "boxsize_mpc_h": [40.0] * 3,
            "boxcenter_mpc_h": [100.0, 0.0, 0.0],
            "family": family,
        },
        "identity_column": "SOURCE_ID",
        "weight_expression": "WEIGHT",
        "pyrecon_kw_weights": None,
    }
    return config, data, randoms, settings


def test_field_provenance_keys_actual_sources_weights_settings_and_code():
    config, data, randoms, settings = field_inputs()

    def identity(data=data, settings=settings, code="fixture-source-v1"):
        return displacement_field_identity(
            data,
            randoms,
            settings=settings,
            source_identity={"tracer": "LRG", "bin": "LRG2", "mock": 0, "cap": "NGC"},
            code_identity=code,
            config=config,
        )

    original = identity()
    assert identity() == original
    assert identity(code="fixture-source-v2")["key"] != original["key"]
    for column in ("X", "WEIGHT", "SOURCE_ID"):
        altered = copy.deepcopy(data)
        altered[column][0] += 1
        if column == "SOURCE_ID":
            altered[column][0] += 10
        assert identity(data=altered)["key"] != original["key"]
    assert identity(settings={**settings, "smoothing_radius_mpc_h": 20.0})["key"] != original["key"]
    with pytest.raises(ValueError, match="complete solve"):
        identity(settings={key: value for key, value in settings.items() if key != "iterations"})


def test_native_runtime_field_identical_galaxy_random_shifts(tmp_path, monkeypatch):
    pyrecon = pytest.importorskip("pyrecon")
    _, data, randoms, _ = field_inputs()
    recon = pyrecon.IterativeFFTReconstruction(
        f=0.8,
        bias=2.0,
        nmesh=[4] * 3,
        boxsize=[40.0] * 3,
        boxcenter=[100.0, 0.0, 0.0],
        los=None,
        wrap=False,
        resampler="tsc",
        dtype="f8",
        fft_plan="estimate",
    )
    recon.mesh_psi = [recon.pm.create(type="real", value=float(i + 1)) for i in range(3)]
    shifted = apply_displacement_field(recon, data, randoms, identity_column="SOURCE_ID")
    for name in ("X_SHIFTED", "Y_SHIFTED", "Z_CART_SHIFTED"):
        np.testing.assert_array_equal(shifted[0][name], shifted[1][name])
    np.testing.assert_array_equal(shifted[0]["SOURCE_ID"], data["SOURCE_ID"])
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(ValueError, match="geometry"):
        apply_displacement_field(
            recon, {**data, "X": data["X"] + 100}, randoms, identity_column="SOURCE_ID"
        )
    # Invalid cells away from the queried targets must still reject the field.
    for invalid in (
        np.zeros((2, 2, 2)),
        np.zeros((4, 4, 4), dtype="f4"),
        np.full((4, 4, 4), np.nan),
    ):
        with monkeypatch.context() as patch:
            patch.setattr(recon, "mesh_psi", [invalid] * 3)
            with pytest.raises(ValueError, match="invalid shape, dtype or values"):
                apply_displacement_field(recon, data, randoms, identity_column="SOURCE_ID")
    with monkeypatch.context() as patch:
        patch.setattr(recon, "mesh_psi", recon.mesh_psi[:2])
        with pytest.raises(ValueError, match="three displacement components"):
            apply_displacement_field(recon, data, randoms, identity_column="SOURCE_ID")


def test_native_runtime_transfer_catalogues_match_direct_reconstruction(tmp_path, monkeypatch):
    """Native solve, fixed-target shifts and saved catalogue readout without mesh files."""
    pytest.importorskip("pyrecon")
    from pzbao.catalog import size_box_from_extent
    from pzbao.reconstruction import run_reconstruction, SHIFTED_COLUMNS

    config, _, _, settings = field_inputs()
    # Fixed-target shifts are the transfer experiment's; its snapshot keeps the broad-field
    # scope, so the core per-bin partition guard does not apply to these catalogues.
    config = transfer_run_configs(config)["self--R25"]

    def catalogue(axis, first_id):
        xyz = np.array(np.meshgrid(axis, axis, axis, indexing="ij")).reshape(3, -1).T
        xyz[:, 0] += 1000.0
        return {
            "SOURCE_ID": np.arange(first_id, first_id + len(xyz), dtype=np.uint64),
            "X": xyz[:, 0],
            "Y": xyz[:, 1],
            "Z_CART": xyz[:, 2],
            "WEIGHT": np.ones(len(xyz)),
            "WEIGHT_FKP": np.ones(len(xyz)),
        }

    data = catalogue(np.linspace(-48.0, 48.0, 8), 0)
    randoms = catalogue(np.linspace(-54.0, 54.0, 16), 10000)
    data["WEIGHT"] += 0.2 * np.sin((data["X"] - 1000.0) / 30.0)
    family = reconstruction_mask_family(
        tracer="LRG", cap="NGC", smoothing_radius_mpc_h=25, measurement_bin="LRG2"
    )
    settings.update(
        mask_family=family,
        common_geometry={
            **size_box_from_extent(
                [946.0, -54.0, -54.0],
                [1054.0, 54.0, 54.0],
                cell_size_mpc_h=settings["cell_size_mpc_h"],
                box_padding=settings["box_padding"],
            ),
            "family": family,
        },
    )
    from pzbao.catalog import save_catalogue, read_catalogue

    def reject_mesh_save(*args, **kwargs):
        raise AssertionError("displacement arrays must remain in memory")

    monkeypatch.setattr(np, "save", reject_mesh_save)
    originals = [copy.deepcopy(data), copy.deepcopy(randoms)]
    direct_data, direct_randoms, summary = run_reconstruction(
        data, randoms, **settings, config=config
    )
    assert summary["field_storage"] == "runtime_only"
    assert summary["data_shift_rms_mpc_h"] > 0
    target = {name: values[:32].copy() for name, values in data.items()}
    target["SOURCE_ID"] += 100000
    target["PHOTOZ_ERROR"] = np.linspace(-0.01, 0.01, 32, dtype=np.float32)
    target["WEIGHT_FKP"] = np.full(32, 0.7)
    target_original = copy.deepcopy(target)
    shifted_data, shifted_randoms, transfer_summary = run_reconstruction(
        data, randoms, **settings, config=config, target_catalogues=(target, target)
    )
    grouped = []

    def consume(field, diagnostics):
        for offset in (100000, 200000, 300000):
            member = {**target, "SOURCE_ID": target["SOURCE_ID"] + offset}
            grouped.append(
                apply_displacement_field(field, member, member, identity_column="SOURCE_ID")
            )
        return diagnostics

    grouped_summary = run_reconstruction(
        data, randoms, **settings, config=config, _field_consumer=consume
    )
    assert grouped_summary["field_storage"] == "runtime_only" and len(grouped) == 3
    for members in grouped:
        for member in members:
            for name in SHIFTED_COLUMNS:
                np.testing.assert_allclose(member[name], shifted_data[name], rtol=0, atol=1e-11)
    assert transfer_summary["field_storage"] == "runtime_only"
    for original, unchanged in zip(originals, (data, randoms)):
        for name in original:
            np.testing.assert_array_equal(unchanged[name], original[name])
    for role, shifted in (("data", shifted_data), ("random", shifted_randoms)):
        for name, original in target_original.items():
            np.testing.assert_array_equal(target[name], original)
            np.testing.assert_array_equal(shifted[name], original)
            assert shifted[name].dtype == original.dtype
        for name in SHIFTED_COLUMNS:
            np.testing.assert_array_equal(shifted_data[name], shifted_randoms[name])
            np.testing.assert_allclose(shifted[name], direct_data[name][:32], rtol=0, atol=1e-11)
        identity = {"stage": "reconstruct", "role": role, "target": "fixture"}
        save_catalogue(tmp_path, role + ".npz", shifted, identity=identity)
        reloaded = read_catalogue(tmp_path / (role + ".npz"), identity=identity)
        for name in shifted:
            np.testing.assert_array_equal(reloaded[name], shifted[name])
            assert reloaded[name].dtype == shifted[name].dtype
        with pytest.raises(ValueError, match="different stage"):
            read_catalogue(tmp_path / (role + ".npz"), identity={**identity, "target": "other"})
    assert {path.name for path in tmp_path.iterdir()} == {
        "data.npz",
        "data.json",
        "random.npz",
        "random.json",
    }


@pytest.mark.parametrize("stage", ["covariance", "fit", "sample"])
def test_transfer_pilot_rejects_inference_before_creating_stage(tmp_path, monkeypatch, stage):
    from pzbao import run as runner
    from pzbao.config.transfer import transfer_run_configs

    monkeypatch.setattr(runner, "_code_identity", lambda **kwargs: {"code_commit": "fixture"})
    run = {
        "root": tmp_path / "pilot",
        "config": transfer_run_configs(load_analysis_config(), pilot=True)["self--R25"],
        "metadata": {"mode": "scratch", "code_commit": "fixture"},
    }
    with pytest.raises(ValueError, match="engineering-only"):
        runner.execute_stage(run, stage)
    assert not run["root"].exists()


def test_native_transfer_stage_uses_runtime_fields_for_fixed_targets(tmp_path, monkeypatch):
    pyrecon = pytest.importorskip("pyrecon")
    from pzbao import reconstruction, selection
    from pzbao.config.transfer import transfer_run_configs

    base, data, randoms, settings = field_inputs()
    config = transfer_run_configs(base)["sigma_0.003--R25"]
    data = {**data, "Z_OBS": np.full(2, 0.7)}
    randoms = {**randoms, "Z": np.full(2, 0.7)}
    family = reconstruction_mask_family(
        tracer="LRG", cap="NGC", smoothing_radius_mpc_h=25, measurement_bin="LRG2"
    )
    geometry = {**settings["common_geometry"], "family": family}
    preparation = {
        "population": "exclusive_parent_classes",
        "geometry": {"NGC": {"LRG2": geometry}},
        "effective_redshifts": {"sigma_0.003": {"LRG2": {"growth_rate": 0.8}}},
    }
    monkeypatch.setattr(selection, "read_selected_parent", lambda *a: (data, randoms, preparation))
    calls = []

    def solve(galaxies, randoms, *, target_catalogues, config, **settings):
        calls.append(settings)
        recon = pyrecon.IterativeFFTReconstruction(
            f=0.8,
            bias=2,
            nmesh=[4] * 3,
            boxsize=[40.0] * 3,
            boxcenter=[100.0, 0.0, 0.0],
            los=None,
            wrap=False,
            resampler="tsc",
            dtype="f8",
            fft_plan="estimate",
        )
        recon.mesh_psi = [recon.pm.create(type="real", value=float(i + 1)) for i in range(3)]
        shifted = apply_displacement_field(
            recon, *target_catalogues, identity_column=settings["identity_column"]
        )
        return (*shifted, {"fixture": True, "field_storage": "runtime_only"})

    monkeypatch.setattr(reconstruction, "run_reconstruction", solve)
    run = {
        "root": tmp_path / "run",
        "config": config,
        "execution_code": {"code_commit": "fixture", "code_state_sha256": "fixture-dirty"},
    }
    outputs, identities = [], []
    for index, condition in enumerate(("sigma_0.01", "sigma_0.03")):
        target = {**data, "SOURCE_ID": data["SOURCE_ID"] + 100 * (index + 1)}
        unit = {
            "tracer": "LRG",
            "condition": condition,
            "realization_id": 0,
            "measurement_bins": ["LRG2"],
        }
        shifted_data, _, diagnostics = reconstruction._reconstruct_transfer_cap(
            run, unit, "NGC", target, randoms, preparation
        )
        np.testing.assert_array_equal(shifted_data["SOURCE_ID"], target["SOURCE_ID"])
        np.testing.assert_array_equal(shifted_data["Z_OBS"], target["Z_OBS"])
        field = diagnostics["bins"]["LRG2"]
        assert field["field_storage"] == "runtime_only"
        assert "field_path" not in field and "field_reused" not in field
        identities.append(field["field_identity"])
        outputs.append(shifted_data["X_SHIFTED"])
    assert len(calls) == 2  # Separate stage invocations recompute; they never load a disk mesh.
    assert identities[0] == identities[1]
    np.testing.assert_array_equal(outputs[0], outputs[1])
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(ValueError, match="actual execution source identity"):
        reconstruction._reconstruct_transfer_cap(
            {key: value for key, value in run.items() if key != "execution_code"},
            unit,
            "NGC",
            target,
            randoms,
            preparation,
        )


def test_legacy_catalogue_handoff_is_sufficient_for_clustering(tmp_path, monkeypatch):
    from pzbao import catalog, reconstruction, selection

    _, data, randoms, _ = field_inputs()
    # Historical full-catalogue products were published under the broad-field scope; the
    # per-bin reader's rejection of this handoff is tested with the per-bin contracts.
    config = load_legacy_config()
    unit = {
        "unit_id": "LRG--sigma_0.01--mock-0",
        "tracer": "LRG",
        "condition": "sigma_0.01",
        "realization_id": 0,
        "measurement_bins": ["LRG2"],
    }
    run = {
        "root": tmp_path,
        "config": config,
        "metadata": {
            "run_name": "fixture",
            "code_commit": "fixture",
            "analysis_config_sha256": "fixture",
        },
    }
    preparation = {
        "geometry": {cap: {"LRG2": {}} for cap in ("NGC", "SGC")},
        "conditions": {
            unit["condition"]: {
                cap: {
                    "selection_diagnostics_by_mock": [{"realization_id": 0}],
                    "densities": {"fixture": True},
                }
                for cap in ("NGC", "SGC")
            }
        },
    }
    monkeypatch.setattr(
        selection, "read_selected_parent", lambda *args: (data, randoms, preparation)
    )

    def shift(*args):
        outputs = []
        for catalogue in (data, randoms):
            result = copy.deepcopy(catalogue)
            for original, shifted in zip(("X", "Y", "Z_CART"), reconstruction.SHIFTED_COLUMNS):
                result[shifted] = result[original] + 0.125
            outputs.append(result)
        return (*outputs, {"field_storage": "runtime_only"})

    directory = tmp_path / "reconstruct" / "units" / unit["unit_id"]
    directory.mkdir(parents=True)
    # Historical producers saved every column beside the shift; readers must still accept it.
    shifted = shift()
    catalogues, products = [], []
    for cap in ("NGC", "SGC"):
        for role, catalogue in zip(("data", "random"), shifted[:2]):
            identity = catalog.catalogue_identity(
                run,
                tracer=unit["tracer"],
                condition=unit["condition"],
                mock_id=unit["realization_id"],
                cap=cap,
                role=role,
                stage="reconstruct",
            )
            name = f"{cap}--{role}.npz"
            products.extend(catalog.save_catalogue(directory, name, catalogue, identity=identity))
            catalogues.append({"path": name, "identity": identity})
    payload = {
        "diagnostics": {cap: {"reconstruction": shifted[2]} for cap in ("NGC", "SGC")},
        "catalogues": catalogues,
        "catalogues_saved": True,
    }
    (directory / "reconstruction.json").write_text(json.dumps(payload))
    result = {
        "products": [*products, {"path": "reconstruction.json", "format": "json"}],
        "catalogues_saved": True,
    }
    (directory / "result.yaml").write_text(
        yaml.safe_dump(
            {
                **run["metadata"],
                "stage": "reconstruct",
                "unit": unit,
                **result,
            }
        )
    )
    for cap in ("NGC", "SGC"):
        restored = reconstruction.read_reconstructed_parent(run, unit, cap)
        for original, catalogue in zip((data, randoms), restored[:2]):
            for name in original:
                np.testing.assert_array_equal(catalogue[name], original[name])
                assert catalogue[name].dtype == original[name].dtype
            for name, shifted in zip(("X", "Y", "Z_CART"), reconstruction.SHIFTED_COLUMNS):
                np.testing.assert_array_equal(catalogue[shifted], original[name] + 0.125)
    assert {path.name for path in directory.iterdir()} == {
        "NGC--data.npz",
        "NGC--data.json",
        "NGC--random.npz",
        "NGC--random.json",
        "SGC--data.npz",
        "SGC--data.json",
        "SGC--random.npz",
        "SGC--random.json",
        "reconstruction.json",
        "result.yaml",
    }
    assert not (tmp_path.parent / "displacement-fields").exists()
    with pytest.raises(ValueError, match="another unit"):
        reconstruction.read_reconstructed_parent(run, {**unit, "condition": "sigma_0.03"}, "NGC")


@pytest.mark.parametrize("state", ["prereconstruction", "postreconstruction"])
@pytest.mark.parametrize("smooth", [False, True])
def test_transfer_flat_smooth_fit_fixes_the_unresponsive_damping_scales(
    tmp_path, monkeypatch, smooth, state
):
    # Under the flat prior the smooth template gives the two acoustic damping scales no
    # response; the smooth comparison fixes them as the other flat scopes do.
    pytest.importorskip("desilike")
    from pzbao.likelihood import build_likelihood
    from pzbao.theory import TRANSFER_FLAT_DAMPING_SCOPE

    monkeypatch.setenv("DESILIKE_CONFIG_DIR", str(tmp_path))
    config = load_analysis_config()
    edges, muedges = np.arange(0.0, 201.0), np.linspace(-1.0, 1.0, 101)
    identity = dict(
        sample_identity="transfer-flat-smooth-test",
        tracer="LRG",
        measurement_bin="LRG2",
        reconstruction_state=state,
    )
    names = ("sigma_streaming", "sigma_parallel", "sigma_perp")
    prior = {
        "sigma_parallel_photoz_mpc_h": 20.0,
        "centers_mpc_h": {"sigma_streaming": 2.0, "sigma_parallel": 6.0, "sigma_perp": 3.0},
        "widths_mpc_h": {"sigma_streaming": 2.0, "sigma_parallel": 2.0, "sigma_perp": 1.0},
        "bounds_mpc_h": {name: [0.0, 50.0] for name in names},
        "unpriored_parameters": list(names),
        "scope": TRANSFER_FLAT_DAMPING_SCOPE,
    }
    separations = np.arange(50.0, 151.0, 4.0)
    mu = 0.5 * (muedges[1:] + muedges[:-1])
    radius = 0.5 * (edges[1:] + edges[:-1])
    counts = 1 + 0.35 * (radius[:, None] / 200) * mu[None, :] ** 2
    variance = (2e-3 * (50.0 / separations) ** 2) ** 2
    _, _, likelihood, _ = build_likelihood(
        np.zeros(2 * separations.size),
        np.diag(np.tile(variance, 2)) / 25,
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
        only_now=smooth,
        **identity,
    )
    params = likelihood.all_params
    assert params["sigmaper"].fixed is smooth and params["sigmapar"].fixed is smooth
    assert params["sigmas"].fixed is False
