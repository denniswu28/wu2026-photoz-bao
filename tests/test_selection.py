"""Behavior tests for deterministic, non-mutating catalogue modification."""

import copy

import numpy as np
import pytest

import pzbao.selection as modifier
from pzbao.config import load_analysis_config
from pzbao.selection import (
    EDGE_RESIDUAL_SHELLS,
    apply_attained_fkp,
    disjoint_random_views,
    spline_residual_converged,
    attach_stable_randomness,
    fit_smooth_radial_model,
    merge_same_tracer_reservoir,
    normalize_weights,
    prepare_tracer_field,
    select_exact_attainable,
    smear_data_redshifts,
    thin_randoms_to_smooth_target,
)


def _catalog(ids, redshifts):
    size = len(ids)
    return {
        "SOURCE_ID": np.asarray(ids, dtype=np.uint64),
        "TARGETID": np.asarray(ids, dtype=np.int64),
        "RA": np.linspace(1.0, 2.0, size),
        "DEC": np.linspace(3.0, 4.0, size),
        "Z": np.asarray(redshifts, dtype=float),
        "WEIGHT": np.ones(size),
    }


def _observed_data(ids, redshifts, config):
    source = attach_stable_randomness(_catalog(ids, redshifts), role="data", config=config)
    return smear_data_redshifts(source, 0.0, config=config)


def test_linear_boundary_counts_are_computed_before_the_parent_cut_without_mutation():
    config = load_analysis_config()
    original = _catalog([1, 2, 3, 4], [0.099, 0.101, 0.201, 0.401])
    before = copy.deepcopy(original)
    selected, record = modifier.prepare_data_for_selection(
        original, tracer="BGS_BRIGHT-21.5", tier_label="native", cap="NGC", config=config
    )
    np.testing.assert_array_equal(selected["SOURCE_ID"], [2, 3])
    boundary = record["boundary_counts"]
    assert sum(boundary["original_clustering_counts"]) == 2
    assert boundary["original_clustering_counts"] == boundary["available_observed_counts"]
    assert sum(boundary["reservoir_observed_counts"]) == 0
    assert boundary["original_scattered_out_count"] == 0
    assert boundary["original_outside_after_smearing_count"] == 2
    assert boundary["reservoir_scattered_in_count"] == 0
    for key in original:
        np.testing.assert_array_equal(original[key], before[key])


def test_stable_deviates_and_ranks_follow_identity_across_row_order():
    config = load_analysis_config()
    first = attach_stable_randomness(
        _catalog([1, 2, 3], [0.11, 0.21, 0.31]), role="data", config=config
    )
    second = attach_stable_randomness(
        _catalog([3, 1, 2], [0.31, 0.11, 0.21]), role="data", config=config
    )
    columns = config["selection"]["stable_randomness"]
    assert columns["random_thinning_rank_column"] not in first
    for column in (columns["photoz_deviate_column"], columns["selection_rank_column"]):
        left = dict(zip(first["SOURCE_ID"], first[column]))
        right = dict(zip(second["SOURCE_ID"], second[column]))
        assert left == right


def test_measurement_bins_must_contiguously_cover_the_reconstruction_field():
    field = {"zmin": 0.1, "zmax": 0.4}
    bins = [{"zmin": 0.1, "zmax": 0.2}, {"zmin": 0.2, "zmax": 0.4}]
    assert modifier._measurement_edges_for_field(field, bins) == [0.1, 0.2, 0.4]
    with pytest.raises(ValueError, match="contiguously and exactly"):
        modifier._measurement_edges_for_field(
            field, [{"zmin": 0.1, "zmax": 0.2}, {"zmin": 0.3, "zmax": 0.4}]
        )


def test_smearing_is_data_only_formula_and_non_mutating():
    config = load_analysis_config()
    source = attach_stable_randomness(_catalog([1, 2], [0.2, 0.3]), role="data", config=config)
    before = copy.deepcopy(source)
    smeared = smear_data_redshifts(source, 0.003, config=config)
    expected = source["Z"] + 0.003 * (1.0 + source["Z"]) * source["PHOTOZ_DEVIATE"]
    np.testing.assert_array_equal(smeared["Z"], source["Z"])
    np.testing.assert_allclose(smeared["Z_OBS"], expected)
    np.testing.assert_array_equal(smeared["Z_RSD"], source["Z"])
    for key in before:
        np.testing.assert_array_equal(source[key], before[key])


def test_smearing_reserves_z_rsd_and_writes_observed_z_from_named_input():
    config = load_analysis_config()
    source = attach_stable_randomness(_catalog([1, 2], [0.2, 0.3]), role="data", config=config)
    source["INPUT_Z"] = np.asarray([0.25, 0.35])
    smeared = smear_data_redshifts(source, 0.0, input_redshift="INPUT_Z", config=config)
    np.testing.assert_array_equal(smeared["Z_RSD"], source["INPUT_Z"])
    np.testing.assert_array_equal(smeared["Z"], source["Z"])
    np.testing.assert_array_equal(smeared["Z_OBS"], source["INPUT_Z"])

    reserved = copy.deepcopy(source)
    reserved["Z_RSD"] = source["Z"].copy()
    with pytest.raises(ValueError, match="source or observed redshift already exists"):
        smear_data_redshifts(reserved, 0.0, config=config)


def test_randoms_are_never_photoz_smeared():
    config = load_analysis_config()
    rank_column = str(config["selection"]["stable_randomness"]["random_thinning_rank_column"])
    deviate_column = str(config["selection"]["stable_randomness"]["photoz_deviate_column"])

    randoms = attach_stable_randomness(_catalog([1, 2], [0.2, 0.3]), role="random", config=config)
    assert rank_column in randoms
    assert deviate_column not in randoms
    with pytest.raises(ValueError, match="randoms are never photo-z smeared"):
        smear_data_redshifts(randoms, 0.003, config=config)

    # The absence of a deviate column is the only thing that stopped this before,
    # so the positive check must also refuse randoms that carry one.
    disguised = copy.deepcopy(randoms)
    disguised[deviate_column] = np.zeros(2)
    with pytest.raises(ValueError, match="randoms are never photo-z smeared"):
        smear_data_redshifts(disguised, 0.003, config=config)


def test_smearing_uses_rsd_bearing_input_not_real_space_truez():
    config = load_analysis_config()
    source = attach_stable_randomness(_catalog([1, 2], [0.15, 0.25]), role="data", config=config)
    source["TRUEZ"] = np.array([0.05, 0.06])

    smeared = smear_data_redshifts(source, 0.01, config=config)

    np.testing.assert_array_equal(smeared["Z_RSD"], source["Z"])
    np.testing.assert_array_equal(smeared["TRUEZ"], source["TRUEZ"])
    expected = source["Z"] + 0.01 * (1.0 + source["Z"]) * source["PHOTOZ_DEVIATE"]
    np.testing.assert_allclose(smeared["Z_OBS"], expected)
    assert not np.allclose(smeared["Z_RSD"], source["TRUEZ"])


def test_weight_normalization_removes_observational_weights_without_mutation():
    source = _catalog([1, 2], [0.11, 0.21])
    source["WEIGHT_ZFAIL"] = np.asarray([0.5, 0.8])
    before = copy.deepcopy(source)
    normalized = normalize_weights(source)
    assert "WEIGHT_ZFAIL" not in normalized
    np.testing.assert_array_equal(normalized["WEIGHT"], np.ones(2))
    for key in before:
        np.testing.assert_array_equal(source[key], before[key])


def test_composed_tracer_preparation_is_non_mutating_and_fkp_complete(monkeypatch):
    config = copy.deepcopy(load_analysis_config())
    identity = "BGS|native|density=1|cap=NGC"
    radial = fit_smooth_radial_model(
        [[2, 2, 2], [2, 2, 2]],
        [0.1, 0.2, 0.3, 0.4],
        calibration_mock_ids=[0, 1],
        smoothing_method="gaussian_kernel",
        bandwidth=1.0,
        sample_identity=identity,
    )
    data = _catalog([1, 2, 3, 4], [0.11, 0.12, 0.21, 0.22])
    randoms = _catalog(range(100, 300), np.linspace(0.1, 0.4, 200, endpoint=False))
    data_before, randoms_before = copy.deepcopy(data), copy.deepcopy(randoms)
    monkeypatch.setattr(modifier, "comoving_shell_volume", lambda *args, **kwargs: [1e6, 1e6])
    selected, assigned_randoms, record = prepare_tracer_field(
        data,
        randoms,
        tracer="BGS_BRIGHT-21.5",
        tier_label="native",
        cap="NGC",
        cosmology=object(),
        radial_artifact=radial,
        sample_identity=identity,
        frozen_counts_by_bin={"BGS1": 2, "BGS2": 2},
        config=config,
    )
    assert len(selected["SOURCE_ID"]) == 4
    assert len(assigned_randoms["SOURCE_ID"]) == 200
    assert np.all(np.isfinite(selected["WEIGHT_FKP"]))
    assert np.all(np.isfinite(assigned_randoms["WEIGHT_FKP"]))
    assert record["fkp_p0_h3_mpc3"] == 7000.0
    assert record["reservoir"]["injected"] is False
    boundary = record["reservoir"]["boundary_counts"]
    assert sum(boundary["original_clustering_counts"]) == 4
    assert sum(boundary["available_observed_counts"]) == 4
    assert sum(boundary["reservoir_observed_counts"]) == 0
    for key in data_before:
        np.testing.assert_array_equal(data[key], data_before[key])
    for key in randoms_before:
        np.testing.assert_array_equal(randoms[key], randoms_before[key])

    mismatched_radial = copy.deepcopy(radial)
    mismatched_radial["z_edges"][0] = 0.0
    with pytest.raises(ValueError, match="support differs"):
        prepare_tracer_field(
            data,
            randoms,
            tracer="BGS_BRIGHT-21.5",
            tier_label="native",
            cap="NGC",
            cosmology=object(),
            radial_artifact=mismatched_radial,
            sample_identity=identity,
            frozen_counts_by_bin={"BGS1": 2, "BGS2": 2},
            config=config,
        )


def test_radial_assignment_outside_exact_tracer_field_fails(monkeypatch):
    config = load_analysis_config()
    data = _catalog([1, 2, 3, 4], [0.11, 0.12, 0.21, 0.22])
    randoms = _catalog(range(100, 200), np.linspace(0.1, 0.4, 100, endpoint=False))
    monkeypatch.setattr(modifier, "comoving_shell_volume", lambda *args, **kwargs: [1e6, 1e6])

    def mostly_outside(catalog, *args, **kwargs):
        result = copy.deepcopy(catalog)
        result["Z"] = np.full(len(result["SOURCE_ID"]), 0.5)
        result["Z"][0] = 0.15
        return result, {"random_data_ratio": 25.0}

    monkeypatch.setattr(modifier, "thin_randoms_to_smooth_target", mostly_outside)
    with pytest.raises(RuntimeError, match="outside its exact tracer field"):
        prepare_tracer_field(
            data,
            randoms,
            tracer="BGS_BRIGHT-21.5",
            tier_label="native",
            cap="NGC",
            cosmology=object(),
            radial_artifact={"z_edges": [0.1, 0.2, 0.3, 0.4]},
            sample_identity="BGS|native|density=1|cap=NGC",
            frozen_counts_by_bin={"BGS1": 2, "BGS2": 2},
            config=config,
        )


def test_exact_attainable_selection_is_nested_and_never_clones():
    config = load_analysis_config()
    source = _observed_data(
        range(10),
        [0.11, 0.12, 0.13, 0.14, 0.15, 0.21, 0.22, 0.23, 0.24, 0.25],
        config,
    )
    volumes = {"BGS1": 500.0, "BGS2": 1000.0}
    full, full_record = select_exact_attainable(
        source,
        tracer="BGS_BRIGHT-21.5",
        tier_label="sigma_0.003",
        shell_volumes_mpc_h3=volumes,
        frozen_counts_by_bin={"BGS1": 4, "BGS2": 4},
        config=config,
    )
    half, half_record = select_exact_attainable(
        source,
        tracer="BGS_BRIGHT-21.5",
        tier_label="sigma_0.003",
        shell_volumes_mpc_h3=volumes,
        frozen_counts_by_bin={"BGS1": 2, "BGS2": 2},
        config=config,
    )
    assert set(half["SOURCE_ID"]).issubset(set(full["SOURCE_ID"]))
    assert len(full["SOURCE_ID"]) <= len(source["SOURCE_ID"])
    assert all(
        row["attained_count"] <= row["maximum_attainable_count"]
        for row in full_record["bins"].values()
    )
    assert (
        half_record["bins"]["BGS1"]["attained_count"]
        <= full_record["bins"]["BGS1"]["attained_count"]
    )


def test_native_floor_records_configured_feasible_selected_and_attained():
    config = load_analysis_config()
    source = _observed_data([1, 2], [0.11, 0.21], config)
    selected, record = select_exact_attainable(
        source,
        tracer="BGS_BRIGHT-21.5",
        tier_label="sigma_0.03",
        shell_volumes_mpc_h3={"BGS1": 1e6, "BGS2": 1e6},
        frozen_counts_by_bin={"BGS1": 1, "BGS2": 1},
        config=config,
    )
    assert len(selected["Z"]) == 2
    for row in record["bins"].values():
        assert row["requested_count"] > row["maximum_attainable_count"]
        assert row["attained_count"] == row["maximum_attainable_count"]
        assert row["bounded_by_attainable"]


def test_exact_selection_rejects_duplicate_stable_ranks():
    config = load_analysis_config()
    source = _observed_data([1, 2], [0.11, 0.21], config)
    rank_column = config["selection"]["stable_randomness"]["selection_rank_column"]
    source[rank_column][1] = source[rank_column][0]
    with pytest.raises(ValueError, match="stable selection ranks collide"):
        select_exact_attainable(
            source,
            tracer="BGS_BRIGHT-21.5",
            tier_label="native",
            shell_volumes_mpc_h3={"BGS1": 1e6, "BGS2": 1e6},
            frozen_counts_by_bin={"BGS1": 1, "BGS2": 1},
            config=config,
        )


def test_fkp_uses_attained_density_and_tracer_specific_p0():
    config = load_analysis_config()
    source = _catalog([1, 2], [0.11, 0.21])
    source["WEIGHT_FKP"] = np.array([9.0, 8.0])
    before = {name: values.copy() for name, values in source.items()}
    for values in source.values():
        values.flags.writeable = False
    attained = {
        "BGS1": {"attained_nbar_h3_mpc3": 1e-4},
        "BGS2": {"attained_nbar_h3_mpc3": 2e-4},
    }
    result = apply_attained_fkp(
        source,
        tracer="BGS_BRIGHT-21.5",
        attained_by_bin=attained,
        redshift_column="Z",
        config=config,
    )
    np.testing.assert_allclose(result["WEIGHT_FKP"], [1 / 1.7, 1 / 2.4])
    for name in source:
        np.testing.assert_array_equal(source[name], before[name])
        assert not np.shares_memory(result[name], source[name])
        assert result[name].flags.writeable
    result["Z"][0] = 0.0
    np.testing.assert_array_equal(source["Z"], before["Z"])


def test_smooth_radial_model_preserves_ensemble_ids_and_radial_support():
    identity = "BGS|sigma_0.003|density=1|cap=NGC"
    artifact = fit_smooth_radial_model(
        [[2, 4, 6], [3, 5, 7]],
        [0.1, 0.2, 0.3, 0.4],
        calibration_mock_ids=[0, 1],
        smoothing_method="gaussian_kernel",
        bandwidth=1.0,
        sample_identity=identity,
    )
    assert "artifact_sha256" not in artifact
    assert artifact["calibration_mock_ids"] == [0, 1]
    assert artifact["bandwidth_semantics"] == "gaussian_sigma_bins"
    assert artifact["support_evidence"]["occupied_bins_have_positive_model_mass_by_construction"]
    assert sum(artifact["target_probability_mass"]) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="one row"):
        fit_smooth_radial_model(
            [[2, 4, 6]],
            [0.1, 0.2, 0.3, 0.4],
            calibration_mock_ids=[0, 1],
            smoothing_method="gaussian_kernel",
            bandwidth=1.0,
            sample_identity=identity,
        )
    with pytest.raises(ValueError, match="at least two mocks"):
        fit_smooth_radial_model(
            [[2, 4, 6]],
            [0.1, 0.2, 0.3, 0.4],
            calibration_mock_ids=[0],
            smoothing_method="gaussian_kernel",
            bandwidth=1.0,
            sample_identity=identity,
        )
    with pytest.raises(ValueError, match="must be unique"):
        fit_smooth_radial_model(
            [[2, 4, 6], [3, 5, 7]],
            [0.1, 0.2, 0.3, 0.4],
            calibration_mock_ids=[0, 0],
            smoothing_method="gaussian_kernel",
            bandwidth=1.0,
            sample_identity=identity,
        )
    with pytest.raises(ValueError, match="uniform redshift bins"):
        fit_smooth_radial_model(
            [[2, 4, 6], [3, 5, 7]],
            [0.1, 0.2, 0.35, 0.4],
            calibration_mock_ids=[0, 1],
            smoothing_method="gaussian_kernel",
            bandwidth=1.0,
            sample_identity=identity,
        )


def test_radial_assignment_preserves_angular_rows_and_leaves_finite_support():
    config = load_analysis_config()
    identity = "BGS|sigma_0.003|density=1|cap=NGC"
    artifact = fit_smooth_radial_model(
        [[2, 4, 6], [3, 5, 7]],
        [0.1, 0.2, 0.3, 0.4],
        calibration_mock_ids=[0, 1],
        smoothing_method="gaussian_kernel",
        bandwidth=1.0,
        sample_identity=identity,
    )
    randoms = attach_stable_randomness(
        _catalog(range(750), np.linspace(0.1, 0.4, 750, endpoint=False)),
        role="random",
        config=config,
    )
    controls = config["selection"]["stable_randomness"]
    assert controls["photoz_deviate_column"] not in randoms
    assert controls["selection_rank_column"] not in randoms
    config = copy.deepcopy(config)
    result, metrics = thin_randoms_to_smooth_target(
        randoms,
        artifact,
        data_count=5,
        exact_random_data_ratio=50,
        expected_sample_identity=identity,
        config=config,
    )
    assert len(result["RA"]) == 250
    assert np.all((result["Z"] >= 0.1) & (result["Z"] < 0.4))
    assert metrics["retained_rows_unchanged"]
    assert metrics["available_random_data_ratio_before_thinning"] == 150.0
    assert metrics["attained_random_data_ratio"] == 50.0
    invalid_angles = copy.deepcopy(randoms)
    invalid_angles["RA"][0] = np.nan
    with pytest.raises(ValueError, match="must be finite"):
        thin_randoms_to_smooth_target(
            invalid_angles,
            artifact,
            data_count=5,
            exact_random_data_ratio=50,
            expected_sample_identity=identity,
            config=config,
        )


def test_reservoir_must_identify_the_same_tracer():
    data = _catalog([1], [0.11])
    reservoir = _catalog([2], [0.09])
    record = {
        "tracer": "LRG",
        "sample_identity": "sample-a",
    }
    with pytest.raises(ValueError, match="cross-tracer"):
        merge_same_tracer_reservoir(
            data,
            reservoir,
            tracer="BGS_BRIGHT-21.5",
            expected_sample_identity="sample-a",
            reservoir_record=record,
            sigma_z_over_one_plus_z=0.0,
            config=load_analysis_config(),
        )


def test_reservoir_injects_only_outside_edge_rows():
    data = _catalog([1, 4], [0.11, 0.39])
    reservoir = _catalog([2, 3, 5], [0.04, 0.11, 0.46])
    record = {
        "tracer": "BGS_BRIGHT-21.5",
        "sample_identity": "sample-a",
    }
    merged, merge_record = merge_same_tracer_reservoir(
        data,
        reservoir,
        tracer="BGS_BRIGHT-21.5",
        expected_sample_identity="sample-a",
        reservoir_record=record,
        sigma_z_over_one_plus_z=0.0,
        config=load_analysis_config(),
    )
    np.testing.assert_array_equal(merged["SOURCE_ID"], [1, 4, 2, 5])
    np.testing.assert_array_equal(merged["IS_RESERVOIR"], [False, False, True, True])
    assert merge_record["overlap_removed_row_count"] == 0
    assert all(edge["guard_reached"] for edge in merge_record["edge_support"])

    mismatched_edges = copy.deepcopy(load_analysis_config())
    mismatched_edges["tracers"]["BGS_BRIGHT-21.5"]["reservoir"]["edges"][0]["redshift"] = 0.09
    with pytest.raises(ValueError, match="does not coincide with its tracer-field boundary"):
        merge_same_tracer_reservoir(
            data,
            reservoir,
            tracer="BGS_BRIGHT-21.5",
            expected_sample_identity="sample-a",
            reservoir_record=record,
            sigma_z_over_one_plus_z=0.0,
            config=mismatched_edges,
        )

    reserved = _catalog([1, 4], [0.11, 0.39])
    reserved["IS_RESERVOIR"] = np.zeros(2, dtype=bool)
    with pytest.raises(ValueError, match="IS_RESERVOIR is reserved"):
        merge_same_tracer_reservoir(
            reserved,
            reservoir,
            tracer="BGS_BRIGHT-21.5",
            expected_sample_identity="sample-a",
            reservoir_record=record,
            sigma_z_over_one_plus_z=0.0,
            config=load_analysis_config(),
        )


def test_reservoir_requires_clustering_support_inside_every_edge():
    data = _catalog([1], [0.11])
    reservoir = _catalog([2, 3], [0.04, 0.46])
    record = {
        "tracer": "BGS_BRIGHT-21.5",
        "sample_identity": "sample-a",
    }
    with pytest.raises(ValueError, match="every inside reservoir-edge band"):
        merge_same_tracer_reservoir(
            data,
            reservoir,
            tracer="BGS_BRIGHT-21.5",
            expected_sample_identity="sample-a",
            reservoir_record=record,
            sigma_z_over_one_plus_z=0.0,
            config=load_analysis_config(),
        )


@pytest.mark.parametrize("reuse_inputs", [False, True])
def test_complete_clustering_wins_overlap_and_exterior_galaxies_scatter_in(reuse_inputs):
    config = load_analysis_config()
    input_cache = modifier._InputPreparationCache(config) if reuse_inputs else None
    data = _catalog([1, 4, 8, 11], [0.415, 0.99, 1.02, 1.0])
    reservoir = {
        name: np.concatenate([data[name][2:], value])
        for name, value in _catalog([2, 5], [0.14, 1.29]).items()
    }
    # The two physical copies have independent source-local IDs and random draws.
    reservoir["TARGETID"][:2] = [900, 1000]
    reservoir["SOURCE_ID"][:2] = [9, 10]
    before = copy.deepcopy(data)
    record = {"tracer": "LRG", "sample_identity": "sample-a"}
    selected, diagnostic = modifier.prepare_data_for_selection(
        data,
        tracer="LRG",
        tier_label="sigma_0.03",
        cap="NGC",
        reservoir=reservoir,
        reservoir_record=record,
        config=config,
        input_cache=input_cache,
    )
    # Existing clustering IDs/draws survive, including two source-z >= 1 galaxies.
    np.testing.assert_array_equal(selected["SOURCE_ID"], [1, 8, 11])
    np.testing.assert_array_equal(selected["Z_RSD"], [0.415, 1.02, 1.0])
    np.testing.assert_allclose(
        selected["PHOTOZ_DEVIATE"],
        [-0.24269938060291443, -1.0094705968499427, -0.12139025684671917],
        rtol=0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        selected["Z_OBS"],
        [0.4046974112934063, 0.9588260818308935, 0.9927165845891969],
        rtol=0,
        atol=1e-15,
    )
    boundary = diagnostic["boundary_counts"]
    assert boundary["overlap_removed_row_count"] == 2
    assert boundary["reservoir_candidate_row_count"] == 4
    assert boundary["original_scattered_in_count"] == 2
    assert boundary["original_scattered_out_count"] == 1
    assert sum(boundary["original_clustering_counts"]) == 2  # source z=1 is excluded
    assert diagnostic["injected_row_count"] == 2  # short upper guard still contributes
    assert boundary["edge_support"][1]["shortfall_redshift"] == pytest.approx(0.07)
    for name in before:
        np.testing.assert_array_equal(data[name], before[name])
    native, native_diagnostic = modifier.prepare_data_for_selection(
        data,
        tracer="LRG",
        tier_label="native",
        cap="NGC",
        reservoir=reservoir,
        reservoir_record=record,
        config=config,
        input_cache=input_cache,
    )
    np.testing.assert_array_equal(native["SOURCE_ID"], [1, 4])
    assert native_diagnostic["injected_row_count"] == diagnostic["injected_row_count"]
    assert native_diagnostic["boundary_counts"]["overlap_removed_row_count"] == 2


def test_reservoir_cannot_silently_drop_data_columns():
    data = _catalog([1, 4], [0.11, 0.39])
    data["DATA_ONLY_CONTROL"] = np.array([7, 8])
    reservoir = _catalog([2, 3], [0.04, 0.46])
    record = {
        "tracer": "BGS_BRIGHT-21.5",
        "sample_identity": "sample-a",
    }
    with pytest.raises(KeyError, match="did not preserve data columns"):
        merge_same_tracer_reservoir(
            data,
            reservoir,
            tracer="BGS_BRIGHT-21.5",
            expected_sample_identity="sample-a",
            reservoir_record=record,
            sigma_z_over_one_plus_z=0.0,
            config=load_analysis_config(),
        )


def test_reservoir_allows_source_local_targetid_numbers():
    config = load_analysis_config()
    data = _catalog([1, 4], [0.11, 0.39])
    reservoir = _catalog([2, 5], [0.04, 0.46])
    reservoir["TARGETID"][0] = data["TARGETID"][0]
    merged, record = merge_same_tracer_reservoir(
        data,
        reservoir,
        tracer="BGS_BRIGHT-21.5",
        expected_sample_identity="sample-a",
        reservoir_record={"tracer": "BGS_BRIGHT-21.5", "sample_identity": "sample-a"},
        sigma_z_over_one_plus_z=0.0,
        config=config,
    )
    np.testing.assert_array_equal(merged["TARGETID"], [1, 4, 1, 5])
    np.testing.assert_array_equal(merged["SOURCE_ID"], [1, 4, 2, 5])
    np.testing.assert_array_equal(merged["Z"], [0.11, 0.39, 0.04, 0.46])
    assert record["overlap_removed_row_count"] == 0


@pytest.mark.parametrize(
    "tracer,data_z,reservoir_z,sigma,shortfalls",
    [
        ("LRG", [0.41, 0.99], [0.14, 1.29], 0.03, [0.0, 0.07]),
        ("BGS_BRIGHT-21.5", [0.11, 0.39], [0.0, 0.66], 0.03, [0.098, 0.0]),
        ("BGS_BRIGHT-21.5", [0.11, 0.39], [0.0, 0.6], 0.03, [0.098, 0.052]),
        ("BGS_BRIGHT-21.5", [0.11, 0.39], [0.0, 0.69], 1.0, [6.5, 8.11]),
        ("LRG", [0.41, 0.99], [0.18, 1.29], 0.025, [0.0, 0.01]),
    ],
)
def test_short_guards_retain_every_available_exterior_galaxy(
    tracer, data_z, reservoir_z, sigma, shortfalls
):
    merged, record = merge_same_tracer_reservoir(
        _catalog([1, 4], data_z),
        _catalog([2, 5], reservoir_z),
        tracer=tracer,
        expected_sample_identity="sample-a",
        reservoir_record={"tracer": tracer, "sample_identity": "sample-a"},
        sigma_z_over_one_plus_z=sigma,
        config=load_analysis_config(),
    )
    np.testing.assert_array_equal(merged["SOURCE_ID"], [1, 4, 2, 5])
    np.testing.assert_array_equal(merged["IS_RESERVOIR"], [False, False, True, True])
    np.testing.assert_allclose(
        [edge["shortfall_redshift"] for edge in record["edge_support"]],
        shortfalls,
        rtol=0,
        atol=1e-14,
    )
    assert [edge["guard_reached"] for edge in record["edge_support"]] == [
        value == 0 for value in shortfalls
    ]
    assert all(edge["added_reservoir_row_count"] == 1 for edge in record["edge_support"])


def test_empty_exterior_support_is_reported_without_removing_clustering():
    data = _catalog([1, 4], [0.11, 0.39])
    merged, record = merge_same_tracer_reservoir(
        data,
        _catalog([2, 5], [0.11, 0.39]),
        tracer="BGS_BRIGHT-21.5",
        expected_sample_identity="sample-a",
        reservoir_record={"tracer": "BGS_BRIGHT-21.5", "sample_identity": "sample-a"},
        sigma_z_over_one_plus_z=0.0,
        config=load_analysis_config(),
    )
    np.testing.assert_array_equal(merged["SOURCE_ID"], data["SOURCE_ID"])
    assert not merged["IS_RESERVOIR"].any()
    assert record["reservoir_candidate_row_count"] == 0
    assert all(edge["available_depth_redshift"] == 0 for edge in record["edge_support"])
    assert all(edge["shortfall_redshift"] == 0.05 for edge in record["edge_support"])


@pytest.mark.parametrize("role", ["data", "reservoir"])
def test_duplicate_physical_rows_within_one_source_are_refused(role):
    inputs = {"data": _catalog([1, 4], [0.11, 0.39]), "reservoir": _catalog([2, 5], [0.04, 0.46])}
    source = inputs[role]
    inputs[role] = {name: np.concatenate([value, value[:1]]) for name, value in source.items()}
    inputs[role]["TARGETID"][-1] = 999
    inputs[role]["SOURCE_ID"][-1] = 999
    with pytest.raises(ValueError, match="duplicate physical galaxies"):
        merge_same_tracer_reservoir(
            **inputs,
            tracer="BGS_BRIGHT-21.5",
            expected_sample_identity="sample-a",
            reservoir_record={"tracer": "BGS_BRIGHT-21.5", "sample_identity": "sample-a"},
            sigma_z_over_one_plus_z=0.0,
            config=load_analysis_config(),
        )


@pytest.mark.parametrize("column", ["RA", "DEC", "Z"])
def test_nonfinite_reservoir_coordinates_are_not_treated_as_missing_support(column):
    reservoir = _catalog([2, 5], [0.04, 0.46])
    reservoir[column][0] = np.nan
    with pytest.raises(ValueError, match="coordinates must be finite"):
        merge_same_tracer_reservoir(
            _catalog([1, 4], [0.11, 0.39]),
            reservoir,
            tracer="BGS_BRIGHT-21.5",
            expected_sample_identity="sample-a",
            reservoir_record={"tracer": "BGS_BRIGHT-21.5", "sample_identity": "sample-a"},
            sigma_z_over_one_plus_z=0.0,
            config=load_analysis_config(),
        )


def _staircase_counts(n_mocks=13, n_bins=60, step_at=30, high=4000.0, low=560.0, seed=7):
    """Thirteen Poisson realizations of a radial distribution with one density step.

    This reproduces, in miniature, the shape the calibration data actually has: the target
    density is one constant per measurement bin, so the selected n(z) jumps at the internal
    bin boundary. The step is what no smooth family can represent, so it is the feature the
    repaired diagnostics have to see.
    """
    rng = np.random.default_rng(seed)
    truth = np.where(np.arange(n_bins) < step_at, high, low)
    return rng.poisson(truth, size=(n_mocks, n_bins)).astype(float), truth


def test_the_unweighted_bandwidth_is_far_below_the_noise_it_is_meant_to_match():
    counts, _ = _staircase_counts()
    edges = np.linspace(0.4, 1.0, counts.shape[1] + 1)
    artifact = fit_smooth_radial_model(
        counts,
        edges,
        calibration_mock_ids=list(range(counts.shape[0])),
        smoothing_method="smoothing_spline",
        bandwidth=1.0,
        sample_identity="LRG|tier=native|cap=NGC",
    )
    assert artifact["bandwidth_semantics"] == "smoothing_factor_per_center"
    requested = artifact["convergence_evidence"]["requested_smoothing"]
    standard_error = np.asarray(
        artifact["variance_evidence"]["standard_error_of_mean_count_by_bin"], dtype=float
    )
    assert requested < 0.05 * float(np.sum(standard_error**2))


def test_weighting_makes_the_bandwidth_the_noise_floor_itself():
    counts, _ = _staircase_counts()
    edges = np.linspace(0.4, 1.0, counts.shape[1] + 1)
    for bandwidth in (0.5, 1.0, 2.0):
        artifact = fit_smooth_radial_model(
            counts,
            edges,
            calibration_mock_ids=list(range(counts.shape[0])),
            smoothing_method="smoothing_spline",
            bandwidth=bandwidth,
            sample_identity="LRG|tier=native|cap=NGC",
            residual_weighting="inverse_standard_error",
        )
        assert artifact["bandwidth_semantics"] == "chi_square_per_center"
        mean = np.asarray(artifact["mean_counts"], dtype=float)
        mass = np.asarray(artifact["target_probability_mass"], dtype=float)
        sigma = np.asarray(
            artifact["variance_evidence"]["standard_error_of_mean_count_by_bin"], dtype=float
        )
        usable = sigma > 0
        residual = (mass[usable] * mean.sum() - mean[usable]) / sigma[usable]
        assert np.sqrt(np.mean(residual**2)) == pytest.approx(np.sqrt(bandwidth), rel=0.02)


def test_the_weighted_bandwidth_means_the_same_thing_for_a_much_larger_sample():
    small, _ = _staircase_counts(high=400.0, low=56.0, seed=11)
    large, _ = _staircase_counts(high=400000.0, low=56000.0, seed=11)
    edges = np.linspace(0.4, 1.0, small.shape[1] + 1)
    roots = []
    for counts in (small, large):
        artifact = fit_smooth_radial_model(
            counts,
            edges,
            calibration_mock_ids=list(range(counts.shape[0])),
            smoothing_method="smoothing_spline",
            bandwidth=1.0,
            sample_identity="LRG|tier=native|cap=NGC",
            residual_weighting="inverse_standard_error",
        )
        mean = np.asarray(artifact["mean_counts"], dtype=float)
        mass = np.asarray(artifact["target_probability_mass"], dtype=float)
        sigma = np.asarray(
            artifact["variance_evidence"]["standard_error_of_mean_count_by_bin"], dtype=float
        )
        usable = sigma > 0
        residual = (mass[usable] * mean.sum() - mean[usable]) / sigma[usable]
        roots.append(float(np.sqrt(np.mean(residual**2))))
    assert roots[0] == pytest.approx(roots[1], rel=0.05)


def test_the_convergence_predicate_refuses_only_a_residual_above_the_request():
    # The predicate is the whole of defect C4's protection, so it is pinned directly
    # rather than only through a fit that happens to converge on this machine.
    assert spline_residual_converged(100.0, 100.0) is True
    assert spline_residual_converged(12.0, 100.0) is True
    # Every converged sample across the twenty calibration samples lands within 0.1 per cent
    # of its request, so this must stay accepted with room to spare.
    assert spline_residual_converged(100.1, 100.0) is True
    # The one sample that genuinely fails misses by a factor of twenty-one.
    assert spline_residual_converged(2140.0, 100.0) is False
    assert spline_residual_converged(200.0, 100.0) is False
    assert spline_residual_converged(float("nan"), 100.0) is False
    assert spline_residual_converged(float("inf"), 100.0) is False
    assert spline_residual_converged(-1.0, 100.0) is False


def test_nonconverged_spline_reports_actual_residual_without_changing_requested_smoothing(
    monkeypatch,
):
    from scipy import interpolate

    original = interpolate.UnivariateSpline

    class Unconverged:
        def __init__(self, *args, **kwargs):
            self.spline = original(*args, **kwargs)
            self.requested = kwargs["s"]

        def __call__(self, x):
            return self.spline(x)

        def get_residual(self):
            return self.requested * 20.0

    monkeypatch.setattr(interpolate, "UnivariateSpline", Unconverged)
    counts, _ = _staircase_counts()
    record = fit_smooth_radial_model(
        counts,
        np.linspace(0.4, 1.0, counts.shape[1] + 1),
        calibration_mock_ids=list(range(len(counts))),
        smoothing_method="smoothing_spline",
        bandwidth=1.0,
        residual_weighting="inverse_standard_error",
        sample_identity="synthetic",
    )
    evidence = record["convergence_evidence"]
    assert evidence["converged"] is False
    assert evidence["achieved_residual"] == evidence["requested_smoothing"] * 20.0


def test_a_spline_that_beats_its_requested_smoothing_still_counts_as_converged():
    # Landing below the requested residual is not a failure. A featureless sample is
    # tracked more closely than asked by the smoothest curve available, and refusing that
    # would reject a perfectly good model for being too easy to fit.
    counts = np.tile(np.linspace(1000.0, 1100.0, 40), (13, 1))
    edges = np.linspace(0.4, 1.0, counts.shape[1] + 1)
    artifact = fit_smooth_radial_model(
        counts,
        edges,
        calibration_mock_ids=list(range(counts.shape[0])),
        smoothing_method="smoothing_spline",
        bandwidth=1.0,
        sample_identity="LRG|tier=native|cap=NGC",
    )
    evidence = artifact["convergence_evidence"]
    assert evidence["achieved_residual"] < evidence["requested_smoothing"]
    assert evidence["converged"] is True


def test_a_converged_spline_is_recorded_as_converged():
    counts, _ = _staircase_counts()
    edges = np.linspace(0.4, 1.0, counts.shape[1] + 1)
    artifact = fit_smooth_radial_model(
        counts,
        edges,
        calibration_mock_ids=list(range(counts.shape[0])),
        smoothing_method="smoothing_spline",
        bandwidth=1.0,
        sample_identity="LRG|tier=native|cap=NGC",
        residual_weighting="inverse_standard_error",
    )
    evidence = artifact["convergence_evidence"]
    assert evidence["converged"] is True
    assert evidence["achieved_residual"] == pytest.approx(evidence["requested_smoothing"], rel=1e-3)


def test_no_model_mass_reaches_a_shell_with_no_galaxies():
    # A Gaussian kernel necessarily spreads mass into empty shells beside occupied ones.
    # Randoms at a redshift the survey does not reach bias the estimator directly, so the
    # invariant is enforced by construction and the size of the correction is recorded.
    counts = np.array([[0.0, 0.0, 100.0, 120.0, 0.0, 0.0]] * 3)
    counts[1] = counts[1] + 1.0 * (counts[1] > 0)
    edges = np.linspace(0.4, 1.0, counts.shape[1] + 1)
    artifact = fit_smooth_radial_model(
        counts,
        edges,
        calibration_mock_ids=[0, 1, 2],
        smoothing_method="gaussian_kernel",
        bandwidth=1.0,
        sample_identity="LRG|tier=native|cap=NGC",
    )
    mass = np.asarray(artifact["target_probability_mass"], dtype=float)
    mean = np.asarray(artifact["mean_counts"], dtype=float)
    support = artifact["support_evidence"]
    assert np.all(mass[mean == 0.0] == 0.0)
    assert support["unoccupied_bins_have_zero_model_mass"] is True
    assert support["unoccupied_bin_count"] == 4
    # The kernel really did leak, so the recorded correction has to be visibly non-zero;
    # otherwise the evidence would suggest a problem that had been silently absorbed. More than a
    # third of the smoothed mass landed in shells this sample does not reach, which is the size of
    # the bias the guard removes rather than merely the fact that it removed something.
    assert support["unoccupied_bin_mass_removed_fraction"] == pytest.approx(0.35605887, abs=1e-8)
    np.testing.assert_allclose(mass, [0.0, 0.0, 0.48890097, 0.51109903, 0.0, 0.0], atol=1e-8)
    assert sum(artifact["target_probability_mass"]) == pytest.approx(1.0)


def test_a_sample_with_no_empty_shells_records_no_correction():
    counts, _ = _staircase_counts()
    edges = np.linspace(0.4, 1.0, counts.shape[1] + 1)
    artifact = fit_smooth_radial_model(
        counts,
        edges,
        calibration_mock_ids=list(range(counts.shape[0])),
        smoothing_method="smoothing_spline",
        bandwidth=1.0,
        sample_identity="LRG|tier=native|cap=NGC",
        residual_weighting="inverse_standard_error",
    )
    support = artifact["support_evidence"]
    assert support["unoccupied_bin_count"] == 0
    assert support["unoccupied_bin_mass_removed_fraction"] == 0.0


def test_the_edge_residual_the_acceptance_record_requires_is_emitted_in_noise_units():
    counts, _ = _staircase_counts()
    edges = np.linspace(0.4, 1.0, counts.shape[1] + 1)
    artifact = fit_smooth_radial_model(
        counts,
        edges,
        calibration_mock_ids=list(range(counts.shape[0])),
        smoothing_method="smoothing_spline",
        bandwidth=1.0,
        sample_identity="LRG|tier=native|cap=NGC",
        residual_weighting="inverse_standard_error",
    )
    fidelity = artifact["fidelity_evidence"]
    edge = fidelity["edge_residual"]
    assert fidelity["edge_residual_shell_count"] == EDGE_RESIDUAL_SHELLS
    assert np.isfinite(edge) and edge >= 0.0
    mean = np.asarray(artifact["mean_counts"], dtype=float)
    mass = np.asarray(artifact["target_probability_mass"], dtype=float)
    sigma = np.asarray(
        artifact["variance_evidence"]["standard_error_of_mean_count_by_bin"], dtype=float
    )
    selected = np.zeros(mean.size, dtype=bool)
    selected[:EDGE_RESIDUAL_SHELLS] = True
    selected[-EDGE_RESIDUAL_SHELLS:] = True
    selected = selected & (sigma > 0)
    expected = np.max(np.abs(mass[selected] * mean.sum() - mean[selected]) / sigma[selected])
    assert edge == pytest.approx(float(expected))


def test_an_unknown_residual_weighting_is_refused():
    counts, _ = _staircase_counts()
    edges = np.linspace(0.4, 1.0, counts.shape[1] + 1)
    with pytest.raises(ValueError, match="residual weighting"):
        fit_smooth_radial_model(
            counts,
            edges,
            calibration_mock_ids=list(range(counts.shape[0])),
            smoothing_method="smoothing_spline",
            bandwidth=1.0,
            sample_identity="LRG|tier=native|cap=NGC",
            residual_weighting="poisson",
        )


def _radial_target(counts, edges, identity, config):
    """One frozen smooth-radial artifact, built the way production builds one."""
    candidate = fit_smooth_radial_model(
        counts,
        edges,
        calibration_mock_ids=[0, 1],
        smoothing_method="gaussian_kernel",
        bandwidth=1.0,
        sample_identity=identity,
    )
    return candidate


def test_equal_random_budgets_with_different_radial_targets_select_different_rows():
    """Equal total random counts do not imply equal -- or nested -- selected random sets.

    Thinning allocates the exact budget bin by bin from the sample's own frozen radial target
    and then keeps the lowest thinning ranks *within each bin*. Two samples therefore agree on
    which rows they take only if they agree bin by bin, which is a property of their target
    mass and not of their totals. A D-005 measurement that emulated the tiers by taking a
    prefix of one shared ordering sized to each tier's total assumed the opposite, and that
    assumption made every tier's support automatically nested inside every larger tier's.

    The two samples here differ only in their radial target: same randoms, same budget, same
    ratio, same edges. They select different rows, and neither selection contains the other,
    so a support difference between two tiers cannot be read as truncation of a nested set.
    """
    config = copy.deepcopy(load_analysis_config())
    edges = [0.1, 0.2, 0.3, 0.4]
    # One target leans low in redshift, the other leans high. Nothing else differs.
    low = "LRG|sigma_0.003|density=1|cap=NGC"
    high = "LRG|sigma_0.03|density=1|cap=NGC"
    low_target = _radial_target([[40, 4, 1], [40, 4, 1]], edges, low, config)
    high_target = _radial_target([[1, 4, 40], [1, 4, 40]], edges, high, config)
    assert low_target["target_probability_mass"] != high_target["target_probability_mass"]

    randoms = attach_stable_randomness(
        _catalog(range(900), np.linspace(0.1, 0.4, 900, endpoint=False)),
        role="random",
        config=config,
    )
    selections = {}
    for identity, artifact in ((low, low_target), (high, high_target)):
        retained, _ = thin_randoms_to_smooth_target(
            randoms,
            artifact,
            data_count=4,
            exact_random_data_ratio=50,
            expected_sample_identity=identity,
            config=config,
        )
        selections[identity] = set(int(value) for value in retained["TARGETID"])

    # The budgets are identical by construction, so any difference below is the target's doing.
    assert len(selections[low]) == len(selections[high]) == 200
    assert selections[low] != selections[high]
    # Neither contains the other: each sample holds rows the other left behind. This is the
    # claim that matters, because containment is what would let a shared mask be described as
    # merely truncating a nested support rather than resolving a real disagreement.
    assert selections[low] - selections[high]
    assert selections[high] - selections[low]


def _retained_randoms_for_views(config):
    """One sample's retained randoms, produced by the production thinning route."""
    edges = [0.1, 0.2, 0.3, 0.4]
    identity = "LRG|sigma_0.003|density=1|cap=NGC"
    target = _radial_target([[40, 30, 30], [40, 30, 30]], edges, identity, config)
    randoms = attach_stable_randomness(
        _catalog(range(3000), np.linspace(0.1, 0.4, 3000, endpoint=False)),
        role="random",
        config=config,
    )
    retained, _ = thin_randoms_to_smooth_target(
        randoms,
        target,
        data_count=20,
        exact_random_data_ratio=50,
        expected_sample_identity=identity,
        config=config,
    )
    return retained, edges


def _view_test_config():
    config = copy.deepcopy(load_analysis_config())
    return config


@pytest.mark.parametrize("view_count", [3, 10])
def test_random_views_partition_rows_and_survive_reordering_and_native_byte_order(view_count):
    config = _view_test_config()
    retained, edges = _retained_randoms_for_views(config)
    views = disjoint_random_views(retained, edges, view_count=view_count, config=config)
    indices = views["view_indices"]
    np.testing.assert_array_equal(np.sort(np.concatenate(indices)), np.arange(len(retained["Z"])))
    assert max(views["view_row_counts"]) - min(views["view_row_counts"]) <= len(edges) - 1
    expected = [retained["SOURCE_ID"][index] for index in indices]
    order = np.argsort(-retained["TARGETID"])
    shuffled = {name: np.asarray(value)[order] for name, value in retained.items()}
    reordered = disjoint_random_views(shuffled, edges, view_count=view_count, config=config)
    for before, index in zip(expected, reordered["view_indices"]):
        np.testing.assert_array_equal(before, shuffled["SOURCE_ID"][index])
    big_endian = {
        name: np.asarray(value).astype(np.asarray(value).dtype.newbyteorder(">"))
        for name, value in retained.items()
    }
    reread = disjoint_random_views(big_endian, edges, view_count=view_count, config=config)
    for before, after in zip(indices, reread["view_indices"]):
        np.testing.assert_array_equal(before, after)
