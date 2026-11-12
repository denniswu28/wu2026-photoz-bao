"""Scientific values, compatible dimensions, and the complete immutable snapshot digest."""

from pathlib import Path

import pytest
import yaml

from pzbao.config import (
    _UniqueKeyLoader,
    analysis_config_sha256,
    load_analysis_config,
    same_identity,
    validate_analysis_config,
)


def test_exact_scientific_values_are_centralized():
    config = load_analysis_config()
    assert config["tracers"]["LRG"]["fkp_p0_h3_mpc3"] == 10000.0
    assert config["tracers"]["BGS_BRIGHT-21.5"]["fkp_p0_h3_mpc3"] == 7000.0
    assert config["tracers"]["BGS_BRIGHT-21.5"]["field"] == {"zmin": 0.1, "zmax": 0.4}
    assert config["tracers"]["LRG"]["field"] == {"zmin": 0.4, "zmax": 1.0}
    damping = config["fitting"]["damping"]
    assert damping["registered_sigma_z_over_one_plus_z"] == [0.0, 0.001, 0.003, 0.01, 0.03]
    assert damping["nonlinear_minimum_mpc_h"] == 0.0
    assert damping["postreconstruction_sigma_perp_policy"] == ("analytic_producer_is_authoritative")
    assert damping["assessment"] == "post_fit_prior_posterior_overlay_only"
    assert damping["gaussian_widths_mpc_h"] == {
        "sigma_perp": 1.0,
        "sigma_parallel": 2.0,
        "sigma_streaming": 2.0,
    }
    assert damping["baseline_centers_mpc_h"]["LRG"]["prereconstruction"] == {
        "sigma_perp": 4.5,
        "sigma_parallel": 9.0,
        "sigma_streaming": 2.0,
    }
    cosmology = config["physics"]["mock_cosmology"]
    assert cosmology["fundamental_inputs"]["omega_b_h2"] == 0.02237
    assert cosmology["fundamental_inputs"]["N_ncdm"] == 1
    assert cosmology["official_derived_reference"] == {
        "sigma8_m": 0.807952,
        "sigma8_cb": 0.811355,
    }
    assert cosmology["realized_engine_fingerprint"] == {
        "cosmoprimo_version": "1.0.0",
        "sigma8_m": 0.80763539901556,
        "sigma8_cb": 0.8110979270214209,
    }
    assert cosmology["verification_absolute_tolerances"]["A_s"] == 1e-20
    assert cosmology["verification_absolute_tolerances"]["sigma8_cb"] == 1e-10
    assert config["measurement"]["reconstruction"]["line_of_sight"] == "local"
    assert config["measurement"]["reconstruction"]["algorithm"] == "iterative_fft"
    assert config["measurement"]["reconstruction"]["cell_size_mpc_h"] == 4.0
    assert config["measurement"]["reconstruction"]["box_padding"] == 1.4
    assert config["measurement"]["reconstruction"]["resampler"] == "tsc"
    # One frozen radius. The 20-60 Mpc/h sensitivity family is deleted rather than registered as
    # non-promotable: a radius nothing may promote is a radius that only invites a run.
    assert config["measurement"]["reconstruction"]["smoothing_values_mpc_h"] == [15.0]
    assert config["measurement"]["reconstruction"]["threshold_randoms"]["value"] == 0.01
    assert config["measurement"]["reconstruction"]["field_validity"] == {
        "reject_empty_field": True,
        "reject_non_finite": True,
        "require_catalogue_containment": True,
        "require_wrapping_disabled": True,
        "random_support_loss_disposition": "reported_never_rejecting",
    }
    # Every displacement threshold is gone with the paired comparison that applied them.
    assert "paired_recovery_limits" not in config["measurement"]["reconstruction"]
    assert "sensitivity_padding_policy" not in config["measurement"]["reconstruction"]
    # Revision 2.13: every measurement bin is reconstructed on its own selected galaxies and
    # randoms, cut by strict half-open redshift intervals with no radial padding; the broad
    # parent solve remains readable as history only.
    assert config["measurement"]["reconstruction"]["scope"] == "separate_measurement_bins"
    assert "bin_window_padding_smoothing_multiple" not in config["measurement"]["reconstruction"]
    assert config["measurement"]["weights"] == {
        "reconstruction": "WEIGHT",
        "correlation_function": "WEIGHT*WEIGHT_FKP",
        "power_spectrum_secondary": "WEIGHT*WEIGHT_FKP",
        "pyrecon_kw_weights": None,
    }
    power = config["measurement"]["power_spectrum_secondary"]
    # A nonbinding pre/post diagnostic on the two pilot anchors, not a full-grid requirement.
    assert power["required"] is False
    assert power["diagnostic_measurement_bins"] == ["BGS1", "LRG3"]
    assert power["diagnostic_scope"] == "ensemble_mean_pre_post"
    assert "minimum_lattice_modes_per_comparison_bin" not in power
    assert power["multipoles"] == [0, 2, 4]
    assert power["line_of_sight"] == "firstpoint"
    assert power["cell_size_mpc_h"] == 4.0
    assert power["interlacing"] == 3
    assert config["measurement"]["reconstruction"]["allowed_conventions"] == ["recsym"]
    assert config["fitting"]["covariance"]["primary_source"] == "route_level_rascalc"
    covariance = config["fitting"]["covariance"]
    assert covariance["primary_source"] == "route_level_rascalc"
    assert covariance["route_key"] == [
        "measurement_bin",
        "sigma_z_tier",
        "reconstruction_state",
    ]
    assert covariance["realization_scaling"] == "shared_route_covariance"
    assert covariance["mean_vector_scaling"] == "route_covariance_over_realization_count"
    assert config["fitting"]["broadband_model"] == "power3"
    # The alternative broadband arm is absent, not merely unused.
    assert "broadband_sensitivity_models" not in config["fitting"]
    assert config["fitting"]["fit_surfaces"]["BGS_BRIGHT-21.5"] == {
        "observed_multipoles": [0, 2],
        "model_multipoles": [0, 2, 4],
        "ap_mode": "qisoqap",
        "fixed_parameters": {"dbeta": 1.0},
    }
    assert config["fitting"]["no_wiggle_method"] == "peakaverage"
    assert config["fitting"]["nonlinear_parameters"]["qiso"]["prior"] == {
        "dist": "uniform",
        "limits": [0.8, 1.2],
    }
    assert config["fitting"]["nonlinear_parameters"]["b1"]["reference"]["limits"] == [
        1.5,
        2.5,
    ]
    assert config["fitting"]["linear_nuisance"]["treatment"] == {
        "profiling": ".auto",
        "mcmc": ".prec",
        "detection": ".auto",
    }
    assert config["fitting"]["linear_nuisance"]["parameter_patterns"] == [
        "al*_*",
        "bl*_*",
    ]
    assert config["fitting"]["linear_nuisance"]["varied_basenames_by_tracer"] == {
        tracer: [f"alobs_{index}" for index in range(8)] for tracer in ("BGS_BRIGHT-21.5", "LRG")
    }
    assert config["selection"]["stable_randomness"]["random_coordinate_significant_digits"] == 12
    assert config["selection"]["stable_randomness"]["source_redshift_column"] == "Z_RSD"
    assert config["selection"]["stable_randomness"]["observed_redshift_column"] == "Z_OBS"
    assert config["selection"]["primary_density_policy"] == (
        "spherex_motivated_bounded_by_attainable"
    )
    assert config["selection"]["density_diagnostic"]["density_factors"] is None
    assert config["selection"]["radial_randoms"]["exact_random_data_ratio"] == 50
    assert config["fitting"]["fit_edge_limits_mpc_h"] == [48.0, 152.0]
    assert config["fitting"]["sampling"]["seed"] == 4201
    sampling = config["fitting"]["sampling"]
    assert sampling["ensembles"] == sampling["nsplits"] == 4
    assert sampling["walkers"] == 40
    assert sampling["minimum_iterations"] == 200
    assert sampling["check_every"] == 300
    assert sampling["absolute_iteration_ceiling"] == 10000
    assert sampling["burnin_fraction"] == 0.5
    assert sampling["max_eigen_gr"] == 0.01
    assert sampling["stable_over"] == 2
    assert config["fitting"]["photoz_response"]["mu_quadrature_order"] == 128
    assert (
        config["measurement"]["xi"]["cap_combination"] == "normalize_to_cap_dd_then_sum_pair_counts"
    )
    assert config["reporting"]["review_bands"]["edge_rehearsal"] == {
        "alpha_shift_sigma_1_bands": [0.10, 0.33],
        "s_xi_bands": [0.50, 1.00],
        "interval_width_fractional_change_bands": [0.05, 0.10],
        "lost_fraction_visibility_bands": [0.01, 0.05],
    }
    assert config["fitting"]["ap_interval_parameters"] == {
        "qiso": ["qiso"],
        "qisoqap": ["qiso", "qap"],
    }


@pytest.mark.parametrize("version", [13, 14])
def test_bgs_versioned_definition_roundtrip_and_hybrid_rejection(version):
    config = load_analysis_config(Path(__file__).with_name("fixtures") / "analysis_schema14.yaml")
    bgs = "BGS_BRIGHT-21.5"
    if version == 13:
        config["schema_version"] = 13
        config["fitting"]["fit_surfaces"][bgs] = {
            "observed_multipoles": [0],
            "model_multipoles": [0, 2, 4],
            "ap_mode": "qiso",
            "fixed_parameters": {"qap": 1.0, "dbeta": 1.0},
        }
        config["fitting"]["linear_nuisance"]["varied_basenames_by_tracer"][bgs] = [
            "bl0_0",
            "bl0_2",
        ]
    before = analysis_config_sha256(config)
    restored = yaml.safe_load(yaml.safe_dump(config))
    validate_analysis_config(restored)
    assert analysis_config_sha256(restored) == before
    surface = restored["fitting"]["fit_surfaces"][bgs]
    assert 26 * len(surface["observed_multipoles"]) == (26 if version == 13 else 52)
    assert ("qap" in surface["fixed_parameters"]) is (version == 13)
    assert restored["fitting"]["nonlinear_parameters"]["qap"]["prior"] == {
        "dist": "uniform",
        "limits": [0.8, 1.2],
    }
    restored["schema_version"] = 14 if version == 13 else 13
    with pytest.raises(ValueError, match="tracer fit surface"):
        validate_analysis_config(restored)


def test_configuration_has_one_workload_ensemble_and_configured_scalar_search():
    config = load_analysis_config()
    assert config["schema_version"] == 17
    assert "acceptance" not in config
    assert config["analysis"]["mock_ids"] == list(range(25))
    for workload in config["analysis"]["workloads"].values():
        assert workload["mock_ids"] == config["analysis"]["mock_ids"]
    gain = config["reporting"]["gain"]
    assert gain == {
        "ratio": "post_over_pre",
        "one_dimensional_fom": "uncertainty",
        "posterior_uncertainty": "posterior_standard_deviation",
        "profile_uncertainty": "minos_half_width",
    }
    tuning = config["fitting"]["covariance"]["shot_noise_rescaling"]["tuning"]
    assert config["fitting"]["covariance"]["shot_noise_rescaling"]["source"] == "route_mock_scatter"
    assert tuning == {
        "search_limits": [0.1, 10.0],
        "search_points": 4096,
        "status": "APPROVED 2026-09-05",
    }
    assert config["analysis"]["source_redshift_comparison"]["absolute_tolerance"] == 1e-12
    for tracer in config["tracers"].values():
        assert "accepted_artifact" not in tracer["reservoir"]
        assert "equivalence_limits" not in tracer["reservoir"]
    with pytest.raises(TypeError, match="overrides"):
        load_analysis_config(overrides={"measurement.xi.mu_bins": 20})


@pytest.mark.parametrize(
    "key,bad_value",
    [
        ("clustering", "field_cut_before_smearing"),
        ("overlap_precedence", "reservoir"),
        ("physical_identity_columns", ["TARGETID"]),
        ("insufficient_support", "disable_edge"),
    ],
)
def test_boundary_rule_is_explicit_and_rejects_other_prescriptions(key, bad_value):
    from pzbao.config.schema_validation import validate_analysis_config

    config = load_analysis_config()
    support = config["selection"]["boundary_support"]
    assert "experiment" not in support
    assert {name: value for name, value in support.items() if name != "experiment_design"} == {
        "clustering": "complete",
        "overlap_precedence": "clustering",
        "physical_identity_columns": ["RA", "DEC", "Z"],
        "insufficient_support": "retain_and_report",
    }
    config["selection"]["boundary_support"][key] = bad_value
    with pytest.raises(ValueError, match="boundary support"):
        validate_analysis_config(config)


def test_load_returns_independent_copies_and_digest_tracks_every_scalar():
    first, second = load_analysis_config(), load_analysis_config()
    original = analysis_config_sha256(second)
    first["measurement"]["xi"]["mu_bins"] += 1
    assert analysis_config_sha256(first) != original
    assert analysis_config_sha256(load_analysis_config()) == original
    assert analysis_config_sha256(dict(reversed(list(second.items())))) == original
    assert analysis_config_sha256({"identity": True}) != analysis_config_sha256({"identity": 1})
    assert analysis_config_sha256({"identity": 1.0}) != analysis_config_sha256({"identity": 1})
    with pytest.raises(ValueError):
        analysis_config_sha256({"value": float("nan")})


def test_yaml_roundtrip_and_duplicate_keys(tmp_path):
    config = load_analysis_config()
    path = tmp_path / "analysis.yaml"
    path.write_text(yaml.safe_dump(config))
    assert same_identity(load_analysis_config(path), config)
    with pytest.raises(ValueError, match="duplicate YAML key.*lines 1 and 2"):
        yaml.load("schema_version: 9\nschema_version: 9\n", Loader=_UniqueKeyLoader)


@pytest.mark.parametrize(
    "path,value",
    [
        (("analysis", "mock_ids"), [True, *range(1, 25)]),
        (("analysis", "workloads", "pilot", "mock_ids"), list(range(24))),
        (("analysis", "workloads", "pilot", "measurement_bins"), ["unknown"]),
        (("analysis", "workloads", "pilot", "sigma_z_tiers"), ["sigma_0.2"]),
        (("analysis", "workloads", "full_grid", "reconstruction_states"), ["prereconstruction"]),
        (("tracers", "LRG", "reconstruction", "fiducial_bias"), True),
        (("tracers", "LRG", "reconstruction", "fiducial_bias"), -1),
        (("tracers", "LRG", "catalogue", "random_splits"), 18.0),
        (("tracers", "LRG", "catalogue", "random_template"), "random.fits"),
        (("tracers", "LRG", "measurement_bins"), [{"name": "LRG1", "zmin": 0.4, "zmax": 0.5}]),
        (("selection", "stable_randomness", "observed_redshift_column"), "Z_RSD"),
        (("selection", "radial_randoms", "exact_random_data_ratio"), 0),
        (("measurement", "reconstruction", "scope"), "one_field_per_measurement_bin"),
        (("measurement", "reconstruction", "wrap_positions"), True),
        (("measurement", "reconstruction", "field_validity", "reject_non_finite"), False),
        (("measurement", "weights", "reconstruction"), "WEIGHT*WEIGHT_FKP"),
        (("measurement", "xi", "bin_width_mpc_h"), float("nan")),
        (("measurement", "xi", "mu_bins"), True),
        (("measurement", "xi", "rebin_parity_tolerances", "relative"), None),
        (("fitting", "fit_edge_limits_mpc_h"), [49.0, 152.0]),
        (("fitting", "input_bin_width_mpc_h"), 3.5),
        (("fitting", "fit_surfaces", "BGS_BRIGHT-21.5", "fixed_parameters", "qap"), True),
        (("fitting", "photoz_response", "mu_quadrature_order"), 128.0),
        (
            ("fitting", "nonlinear_parameters", "dbeta", "prior"),
            {"dist": "uniform", "limits": [0.7, 1.3]},
        ),
        (("fitting", "damping", "gaussian_widths_mpc_h", "sigma_perp"), 0),
        (("fitting", "sampling", "ensembles"), 1),
        (("fitting", "sampling", "nsplits"), True),
        (("fitting", "sampling", "burnin_fraction"), 1.0),
        (("fitting", "sampling", "absolute_iteration_ceiling"), 100),
        (("fitting", "covariance", "shot_noise_rescaling", "tuning", "search_points"), 1),
        (("fitting", "covariance", "shot_noise_rescaling", "tuning", "search_limits"), None),
        (
            ("fitting", "covariance", "shot_noise_rescaling", "tuning", "status"),
            "MANUAL DECISION REQUIRED",
        ),
        (("fitting", "covariance", "shot_noise_rescaling", "source"), "frozen_registry"),
        (("fitting", "covariance", "rascalc", "route_geometry", "explicit_realization_id"), 0),
        (("reporting", "gain", "ratio"), "pre_over_post"),
        (("reporting", "gain", "one_dimensional_fom"), "inverse_uncertainty"),
        (("reporting", "gain", "posterior_uncertainty"), "quantile_half_width"),
        (("reporting", "gain", "profile_uncertainty"), "parabolic_error"),
        (("reporting", "interval_coverage", "interval_kind"), "posterior"),
        (("reporting", "interval_coverage", "uncertainty_method"), "wald"),
        (("reporting", "interval_coverage", "uncertainty_confidence_level"), 0.95),
    ],
)
def test_invalid_domains_axes_and_supported_methods_fail(path, value):
    config = load_analysis_config()
    parent = config
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = value
    with pytest.raises(ValueError):
        validate_analysis_config(config)


def test_interval_coverage_is_explicit_and_precoverage_snapshots_remain_readable():
    config = load_analysis_config()
    assert config["reporting"].pop("interval_coverage") == {
        "interval_kind": "native_minos",
        "uncertainty_method": "wilson",
        "uncertainty_confidence_level": 0.68,
    }
    validate_analysis_config(config)


@pytest.mark.parametrize(
    "key",
    [
        "accepted_artifact",
        "accepted_bias",
        "required_artifacts",
        "holdout_mock_ids",
        "soft_gate_freeze_policy",
        "scheduler",
    ],
)
def test_retired_credentials_cannot_reappear_in_a_nested_block(key):
    config = load_analysis_config()
    config["measurement"]["reconstruction"][key] = None
    with pytest.raises(ValueError, match="retired"):
        validate_analysis_config(config)


def test_growth_comparison_requires_the_full_table_and_redshift_definition():
    for key in (
        "growth_rate_by_tracer",
        "maximum_fractional_difference_by_tracer",
        "table_redshift_range_by_tracer",
    ):
        config = load_analysis_config()
        del config["measurement"]["reconstruction"]["growth"]["fiducial_reference"][key]["LRG"]
        with pytest.raises(ValueError):
            validate_analysis_config(config)
    config = load_analysis_config()
    config["measurement"]["reconstruction"]["growth"]["fiducial_reference"][
        "reference_redshift_estimator"
    ] = "bin_midpoint"
    with pytest.raises(ValueError):
        validate_analysis_config(config)


def test_schema_checks_domains_without_duplicating_the_scientific_table():
    config = load_analysis_config()
    # Preparing a new configuration file is distinct from overriding an existing run.
    # A valid new value is validated here; the immutable-run consumer still rejects any mutation.
    config["fitting"]["damping"]["gaussian_widths_mpc_h"]["sigma_perp"] = 1.25
    config["tracers"]["LRG"]["reconstruction"]["fiducial_bias"] = 2.05
    validate_analysis_config(config)


def test_native_parent_configuration_encodes_the_configured_hardware_recipe():
    config = load_analysis_config()
    hardware = config["selection"]["reservoir_hardware"]
    assert hardware["veto_zwarn_mtl_bits"] == [9, 18, 19]
    assert hardware["missing_zwarn"] == 999999
    assert hardware["tile_location_multiplier"] == 10000
    assert hardware["maximum_target_id"] == 10**13
    assert hardware["north_declination_min_deg"] == 32.375
    assert hardware["north_galactic_latitude_min_deg"] == 0.0
    for tracer, program, column, bit in [
        ("LRG", "dark", "DESI_TARGET", 0),
        ("BGS_BRIGHT-21.5", "bright", "BGS_TARGET", 1),
    ]:
        spec = config["tracers"][tracer]
        assert spec["reservoir"]["filename"] == spec["source_reference"]["source_basename"]
        assert spec["reservoir"]["hardware_program"] == program
        assert spec["reservoir"]["target_flag"] == {"column": column, "bit": bit}
