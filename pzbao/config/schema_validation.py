"""Type and scientific-domain validation for the resolved analysis configuration."""

from __future__ import annotations

import math
from string import Formatter
from typing import Any, Mapping

import yaml

_TRACERS = {"LRG", "BGS_BRIGHT-21.5"}
#: Schemas whose core reconstruction solves each measurement bin separately (revision 2.13).
#: Schema 17 adds the V_eff-matched density block (revision 2.15) and changes nothing else.
PER_BIN_SCHEMA_VERSIONS = (16, 17)
#: The full sky in square degrees (4 pi sr); shell volumes and sky fractions share it.
FULL_SKY_AREA_DEG2 = 4.0 * math.pi * (180.0 / math.pi) ** 2
#: The V_eff-matched density experiment's count authority and tiers (revision 2.15). This
#: module is also loaded standalone by the forecast readers, so it holds the constants itself.
MATCHED_DENSITY_NAME = "veff_matched_lrg23_exact_counts"
MATCHED_DENSITY_CONDITIONS = ("sigma_0.001", "sigma_0.003")
PCS2_NUISANCE_AXES = ("bl0_0", "bl0_2", "al2_0", "al2_1", "bl2_0", "bl2_2")
POWER3_NUISANCE_AXES = tuple(f"alobs_{index}" for index in range(8))
_RASCALC_ARGUMENTS = [
    "mode",
    "nthread",
    "N2",
    "N3",
    "N4",
    "n_loops",
    "loops_per_sample",
    "out_dir",
    "tmp_dir",
    "randoms_positions1",
    "randoms_weights1",
    "pycorr_allcounts_11",
    "xi_table_11",
    "position_type",
    "xi_table_12",
    "xi_table_22",
    "xi_cut_s",
    "xi_refinement_iterations",
    "pycorr_allcounts_12",
    "pycorr_allcounts_22",
    "normalize_wcounts",
    "no_data_galaxies1",
    "no_data_galaxies2",
    "randoms_samples1",
    "randoms_positions2",
    "randoms_weights2",
    "randoms_samples2",
    "xi_11_samples",
    "xi_12_samples",
    "xi_22_samples",
    "xi_sample_cov",
    "max_l",
    "boxsize",
    "skip_s_bins",
    "skip_l",
    "shot_noise_rescaling1",
    "shot_noise_rescaling2",
    "sampling_grid_size",
    "coordinate_scaling",
    "seed",
    "start_integral_index",
    "last_integral_index",
    "verbose",
]


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects repeated mapping keys."""


def _construct_unique_mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode, deep=False):
    mapping = {}
    first_lines = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            first = first_lines[key]
            repeated = key_node.start_mark.line + 1
            raise ValueError(f"duplicate YAML key {key!r} at lines {first} and {repeated}")
        first_lines[key] = key_node.start_mark.line + 1
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def _mapping(parent, key, keys=None):
    value = parent.get(key) if isinstance(parent, Mapping) else None
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping")
    if keys is not None and set(value) != set(keys.split() if isinstance(keys, str) else keys):
        raise ValueError(f"{key} has missing or unknown fields")
    return value


def _number(value, name, *, minimum=None, positive=False, integer=False):
    if (
        type(value) not in (int, float)
        or (integer and type(value) is not int)
        or not math.isfinite(value)
        or (positive and value <= 0)
        or (minimum is not None and value < minimum)
    ):
        raise ValueError(
            f"{name} must be a finite {'integer' if integer else 'number'} in its domain"
        )
    return value


def _sequence(value, name, *, nonempty=True, unique=False):
    if not isinstance(value, list) or (nonempty and not value):
        raise ValueError(f"{name} must be a {'nonempty ' if nonempty else ''}list")
    if unique and len(value) != len(set(value)):
        raise ValueError(f"{name} must contain unique values")
    return value


def _bounds(value, name, *, minimum=None):
    values = _sequence(value, name)
    if len(values) != 2:
        raise ValueError(f"{name} must have two endpoints")
    for item in values:
        _number(item, name, minimum=minimum)
    if values[0] >= values[1]:
        raise ValueError(f"{name} endpoints must increase")
    return values


def same_identity(actual, expected) -> bool:
    """Compare direct metadata recursively without treating True, 1 and 1.0 as one identity."""
    if isinstance(expected, Mapping):
        return (
            isinstance(actual, Mapping)
            and set(actual) == set(expected)
            and all(same_identity(actual[key], value) for key, value in expected.items())
        )
    if isinstance(expected, (list, tuple)):
        return (
            type(actual) is type(expected)
            and len(actual) == len(expected)
            and all(same_identity(left, right) for left, right in zip(actual, expected))
        )
    if type(expected) not in (str, int, float, bool, type(None)):
        raise TypeError(
            "scientific identity must contain only mappings, sequences and scalar coordinates"
        )
    return type(actual) is type(expected) and actual == expected


def _require(value, expected, name):
    if not same_identity(value, expected):
        raise ValueError(f"unsupported {name}: {value!r}")


def _plain(value, path="analysis configuration"):
    """Reject nonfinite numbers, non-string keys and retired execution credentials."""
    retired = {
        "accepted_artifact",
        "accepted_bias",
        "accepted_calibration_artifact",
        "accepted_holdout_artifact",
        "required_artifacts",
        "calibration_order",
        "holdout_mock_ids",
        "soft_gate_freeze_policy",
        "acceptance",
        "equivalence_limits",
        "manifest_owned_arguments",
        "source_identity_policy",
        "version_policy",
        "output_root",
        "scheduler",
        "campaigns",
        "continuation_maximum_iterations",
        "maximum_iterations",
        "sampler_comparison",
        "power_spectrum_diagnostic",
        "broadband_sensitivity_models",
        "bin_window_padding_smoothing_multiple",
    }
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str) or key in retired:
                raise ValueError(f"unknown or retired configuration key {path}.{key}")
            _plain(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _plain(item, f"{path}[{index}]")
    elif type(value) is float and not math.isfinite(value):
        raise ValueError(f"{path} must be finite")
    elif type(value) not in (str, int, float, bool, type(None)):
        raise ValueError(f"{path} has an unsupported value type")


def _validate_reconstruction_pilot(payload, recon):
    pilot = _mapping(
        recon,
        "exploratory_pilot",
        "experiment specification_revision tracer condition mock_ids cases "
        "diagnostic_shell_width_mpc_h reproduction_rtol reproduction_atol",
    )
    for key in ("experiment", "specification_revision"):
        if not isinstance(pilot[key], str) or not pilot[key].strip():
            raise ValueError(f"pilot {key} must be a nonempty string")
    if pilot["tracer"] not in payload["tracers"] or pilot["condition"] not in {
        tier["label"] for tier in payload["selection"]["tiers"]
    }:
        raise ValueError("unknown pilot tracer or condition")
    for mock in _sequence(pilot["mock_ids"], "pilot mocks", unique=True):
        _number(mock, "pilot mock", integer=True, minimum=0)
        if mock not in payload["analysis"]["mock_ids"]:
            raise ValueError("pilot mock absent from the analysis")
    cases = _sequence(pilot["cases"], "pilot cases")
    for case in cases:
        _mapping({"case": case}, "case", "name smoothing_mpc_h threshold_randoms")
        for key in ("smoothing_mpc_h", "threshold_randoms"):
            _number(case[key], f"pilot {key}", positive=True)
    _require(
        [case["name"] for case in cases],
        ["baseline", "smoothing", "floor", "combined"],
        "pilot case order",
    )
    base, smooth, floor, _ = cases
    points = [(case["smoothing_mpc_h"], case["threshold_randoms"]) for case in cases]
    if len(set(points)) != 4 or points != [
        (base["smoothing_mpc_h"], base["threshold_randoms"]),
        (smooth["smoothing_mpc_h"], base["threshold_randoms"]),
        (base["smoothing_mpc_h"], floor["threshold_randoms"]),
        (smooth["smoothing_mpc_h"], floor["threshold_randoms"]),
    ]:
        raise ValueError("pilot cases must form the labelled two by two experiment")
    for key in ("diagnostic_shell_width_mpc_h", "reproduction_rtol", "reproduction_atol"):
        _number(pilot[key], f"pilot {key}", positive=True)


def _validate_mesh_comparison(payload, recon):
    experiment = _mapping(
        recon,
        "exploratory_mesh_comparison",
        "experiment specification_revision tracer condition measurement_bins mock_ids "
        "scope geometry algorithm smoothing_mpc_h threshold_randoms solver cases "
        "reproduction_rtol reproduction_atol",
    )
    for key in ("experiment", "specification_revision"):
        if not isinstance(experiment[key], str) or not experiment[key].strip():
            raise ValueError(f"mesh comparison {key} must be a nonempty string")
    if experiment["tracer"] not in payload["tracers"] or experiment["condition"] not in {
        tier["label"] for tier in payload["selection"]["tiers"]
    }:
        raise ValueError("unknown mesh comparison tracer or condition")
    bins = _sequence(experiment["measurement_bins"], "mesh comparison bins", unique=True)
    known_bins = {
        item["name"] for item in payload["tracers"][experiment["tracer"]]["measurement_bins"]
    }
    if not set(bins).issubset(known_bins):
        raise ValueError("unknown mesh comparison measurement bin")
    for mock in _sequence(experiment["mock_ids"], "mesh comparison mocks", unique=True):
        _number(mock, "mesh comparison mock", integer=True, minimum=0)
        if mock not in payload["analysis"]["mock_ids"]:
            raise ValueError("mesh comparison mock absent from analysis")
    for key, value in {
        "scope": "separate_measurement_bins",
        "geometry": "fixed_source_parent_box",
        "algorithm": "multigrid",
    }.items():
        _require(experiment[key], value, key)
    for key in ("smoothing_mpc_h", "threshold_randoms", "reproduction_rtol", "reproduction_atol"):
        _number(experiment[key], key, positive=True)
    solver = _mapping(
        experiment, "solver", "jacobi_damping_factor jacobi_niterations vcycle_niterations"
    )
    for key, value in solver.items():
        _number(value, key, positive=True, integer=key != "jacobi_damping_factor")
    cases = _sequence(experiment["cases"], "mesh comparison cases")
    names = []
    for case in cases:
        if not isinstance(case, Mapping) or set(case) not in (
            {"name", "maximum_cell_mpc_h"},
            {"name", "nmesh"},
        ):
            raise ValueError("mesh case needs exactly one mesh prescription")
        if not isinstance(case["name"], str) or not case["name"].isalnum():
            raise ValueError("mesh case name must be alphanumeric")
        names.append(case["name"])
        if "maximum_cell_mpc_h" in case:
            _number(case["maximum_cell_mpc_h"], "cell width", positive=True)
        else:
            if len(case["nmesh"]) != 3:
                raise ValueError("mesh needs three axes")
            for n in case["nmesh"]:
                _number(n, "mesh dimension", positive=True, integer=True)
    if len(set(names)) != len(names):
        raise ValueError("mesh comparison case names must be unique")


SENSITIVITY_KINDS = ("smoothing_scan", "reciso")


def _validate_smoothing_scan(payload, recon):
    design = _mapping(
        recon,
        "exploratory_smoothing_scan",
        "specification_revision radii_mpc_h measurement_bins sigma_z_tiers reconstruction_states "
        "reference_smoothing_mpc_h",
    )
    _require(
        design["specification_revision"], "smoothing-scan-20261013-v2", "smoothing scan revision"
    )
    radii = _sequence(design["radii_mpc_h"], "smoothing scan radii", unique=True)
    for value in radii:
        _number(value, "smoothing scan radius", positive=True)
    if radii != sorted(radii) or design["reference_smoothing_mpc_h"] not in radii:
        raise ValueError("smoothing scan radii must be ordered and include the reference radius")
    bins = {
        item["name"]
        for tracer in payload["tracers"].values()
        for item in tracer["measurement_bins"]
    }
    tiers = {item["label"] for item in payload["selection"]["tiers"]} - {"native"}
    if (
        not set(_sequence(design["measurement_bins"], "scan bins", unique=True)) <= bins
        or not set(_sequence(design["sigma_z_tiers"], "scan tiers", unique=True)) <= tiers
    ):
        raise ValueError("smoothing scan names an unregistered bin or photo-z tier")
    # Pre-reconstruction routes do not depend on the radius and stay with the core run.
    _require(design["reconstruction_states"], ["postreconstruction"], "scan states")


def _validate_sensitivity_run(payload, recon):
    """One radius of the core smoothing scan, or the RecIso robustness run."""
    run = _mapping(recon, "sensitivity_run", "kind")
    if run["kind"] not in SENSITIVITY_KINDS:
        raise ValueError("unknown sensitivity run")
    if "transfer_run" in recon or "exploratory_smoothing_scan" not in recon:
        raise ValueError("sensitivity runs are core runs defined by the smoothing scan design")
    design = recon["exploratory_smoothing_scan"]
    radius = recon["smoothing_values_mpc_h"][0]
    damping = payload["fitting"]["damping"]
    # Revision 2.18: the scan centre refit evaluates the transverse centre at its own radius.
    if payload["fitting"].get("core_fit_check") == "core_scan_radius_centre_v1":
        if run["kind"] != "smoothing_scan" or damping.get("reference_smoothing_mpc_h") != radius:
            raise ValueError("the scan centre refit evaluates the centre at its own scan radius")
    elif damping.get("reference_smoothing_mpc_h") != design["reference_smoothing_mpc_h"]:
        raise ValueError("sensitivity runs keep the reference-radius damping centres")
    mocks = payload["analysis"]["mock_ids"]
    if run["kind"] == "smoothing_scan":
        if radius not in design["radii_mpc_h"]:
            raise ValueError("smoothing scan radius is outside the configured design")
        expected = {
            "measurement_bins": design["measurement_bins"],
            "sigma_z_tiers": design["sigma_z_tiers"],
            "reconstruction_states": design["reconstruction_states"],
            "mock_ids": mocks,
        }
    else:
        scope = payload["fitting"].get("paper_comparisons")
        if scope is None or radius != design["reference_smoothing_mpc_h"]:
            raise ValueError("RecIso is the reference-radius paper comparison")
        expected = {
            "measurement_bins": scope["measurement_bins"],
            "sigma_z_tiers": [scope["condition"]],
            "reconstruction_states": [scope["reconstruction_state"]],
            "mock_ids": mocks,
        }
    _require(payload["analysis"]["workloads"][run["kind"]], expected, "sensitivity workload")


def _validate_transfer_design(payload, recon):
    required = set(
        "specification_revision population class_labels source_class assignment_namespace "
        "allocation exterior_assignment radial_selection scope source_bias common_smoothing_mpc_h "
        "source_smoothing_checks_mpc_h measurement_bins pilot_measurement_bins pilot_mock_ids "
        "primary_broadband broadband_check damping_check postreconstruction_centres".split()
    )
    design = _mapping(recon, "exploratory_transfer")
    # 2.11-T2 adds self-reconstruction radius checks; frozen 2.11-T1 snapshots omit them.
    self_checks = "self_smoothing_checks_mpc_h" in design
    if set(design) != required | ({"self_smoothing_checks_mpc_h"} if self_checks else set()):
        raise ValueError("exploratory_transfer has missing or unknown fields")
    for key, value in {
        "specification_revision": "2.11-T2" if self_checks else "2.11-T1",
        "population": "exclusive_parent_classes",
        "assignment_namespace": "exclusive-class-v1",
        "allocation": "joint_minimum_parent_counts",
        "exterior_assignment": "adjacent_shell_fractions",
        "radial_selection": "smooth_parent_forward_prediction",
        "scope": "separate_measurement_bins",
        "source_bias": "parent_tracer",
        "primary_broadband": "power3",
        "broadband_check": "pcs2",
        "damping_check": "common_flat_sensitivity",
        "postreconstruction_centres": "requires_reviewed_source_target_model",
    }.items():
        _require(design[key], value, f"transfer {key}")
    labels = _sequence(design["class_labels"], "exclusive class labels", unique=True)
    tiers = {item["label"]: item for item in payload["selection"]["tiers"]}
    if len(labels) < 2 or any(label not in tiers for label in labels):
        raise ValueError("exclusive classes must name at least two registered tiers")
    if any(tiers[label]["target_nbar_h3_mpc3"] is None for label in labels):
        raise ValueError("exclusive classes require parent abundance targets")
    if set(labels) != {"sigma_0.003", "sigma_0.01", "sigma_0.03"}:
        raise ValueError("transfer design excludes the separate proxy and native experiments")
    source = min(labels, key=lambda label: tiers[label]["sigma_z_over_one_plus_z"])
    if design["source_class"] != source:
        raise ValueError("transfer source must have the lowest redshift uncertainty")
    radii = [
        design["common_smoothing_mpc_h"],
        *_sequence(design["source_smoothing_checks_mpc_h"], "transfer radii", unique=True),
    ]
    for value in radii:
        _number(value, "transfer radius", positive=True)
    if len(radii) != len(set(radii)):
        raise ValueError("transfer radii must not duplicate the common-radius run")
    self_radii = [design["common_smoothing_mpc_h"]]
    if self_checks:
        self_radii += _sequence(design["self_smoothing_checks_mpc_h"], "self radii", unique=True)
    for value in self_radii:
        _number(value, "self transfer radius", positive=True)
    if len(self_radii) != len(set(self_radii)):
        raise ValueError("self radii must not duplicate the common-radius run")
    all_bins = {
        item["name"]
        for tracer in payload["tracers"].values()
        for item in tracer["measurement_bins"]
    }
    bins = _sequence(design["measurement_bins"], "transfer bins", unique=True)
    pilot_bins = _sequence(design["pilot_measurement_bins"], "transfer pilot bins", unique=True)
    if not set(bins) <= all_bins or not set(pilot_bins) <= set(bins):
        raise ValueError("transfer bins must belong to the parent tracers")
    mocks = _sequence(design["pilot_mock_ids"], "transfer pilot mocks", unique=True)
    if not set(mocks) <= set(payload["analysis"]["mock_ids"]):
        raise ValueError("transfer pilot mocks must belong to the matched ensemble")


def _validate_matched_density(selection, tier_labels, tracers):
    """The V_eff-matched density block (schema 17) names its reference sky, tiers and bins.

    The rule itself lives with the density successor; this checks only that the block is
    complete, that its sky fraction is the recorded reference area over the full sky, and that
    it names registered red-galaxy tiers and bins, so that every schema-17 configuration carries
    one consistent definition of the experiment it may later bind.
    """
    matched = _mapping(
        selection,
        "veff_matched_density",
        "count_authority reference_area_deg2 reference_f_sky tiers measurement_bins bounded_allowed",
    )
    _require(matched["count_authority"], MATCHED_DENSITY_NAME, "matched density count authority")
    area = _number(matched["reference_area_deg2"], "reference sky area", positive=True)
    fsky = _number(matched["reference_f_sky"], "reference sky fraction", positive=True)
    if area > FULL_SKY_AREA_DEG2 or abs(fsky - area / FULL_SKY_AREA_DEG2) > 1e-12:
        raise ValueError("reference sky fraction must equal the reference area over the full sky")
    tiers = _sequence(matched["tiers"], "matched tiers", unique=True)
    if tiers != list(MATCHED_DENSITY_CONDITIONS) or not set(tiers) <= set(tier_labels):
        raise ValueError("matched tiers must be the implemented registered red-galaxy tiers")
    red_bins = {item["name"] for item in tracers["LRG"]["measurement_bins"]}
    bins = _sequence(matched["measurement_bins"], "matched bins", unique=True)
    if not set(bins) <= red_bins:
        raise ValueError("matched bins must be luminous-red measurement bins")
    bounded = _sequence(matched["bounded_allowed"], "bounded bins", nonempty=False)
    pairs = [tuple(item) if isinstance(item, list) else None for item in bounded]
    if len(set(pairs)) != len(pairs) or any(
        pair is None or len(pair) != 2 or pair[0] not in tiers or pair[1] not in bins
        for pair in pairs
    ):
        raise ValueError("bounded bins must be distinct [tier, bin] pairs of the experiment")


def _validate_config(payload: Mapping[str, Any]) -> None:
    """Check types, supported methods and mutually consistent scientific domains.

    Values come from YAML. This schema does not duplicate its cosmology, density table, mesh
    sizes, prior widths or integration budgets as a second numerical authority.
    """
    _plain(payload)
    required = {
        "schema_version",
        "configuration_name",
        "choice_provenance",
        "physics",
        "analysis",
        "tracers",
        "selection",
        "measurement",
        "fitting",
        "reporting",
    }
    if not isinstance(payload, Mapping) or set(payload) != required:
        raise ValueError("analysis configuration has missing or unknown sections")
    version = payload["schema_version"]
    if type(version) is not int or version not in (13, 14, 15, 16, 17):
        raise ValueError("schema_version must be 13, 14, 15, 16 or 17")
    _require(payload["configuration_name"], "bao_analysis", "configuration_name")
    origins = _mapping(payload, "choice_provenance")
    for name in origins:
        origin = _mapping(origins, name, "primary_origin_class evidence_tags")
        if (
            not isinstance(origin["primary_origin_class"], str)
            or not origin["primary_origin_class"]
        ):
            raise ValueError("choice provenance needs a named origin")
        _sequence(origin["evidence_tags"], "evidence_tags", unique=True)

    physics = _mapping(payload, "physics", "speed_of_light_km_s mock_cosmology")
    _number(physics["speed_of_light_km_s"], "speed of light", positive=True)
    cosmology = _mapping(
        physics,
        "mock_cosmology",
        "provider engine fundamental_inputs official_derived_reference realized_engine_fingerprint "
        "verification_absolute_tolerances provenance",
    )
    _require(cosmology["provider"], "cosmoprimo.fiducial.DESI", "cosmology provider")
    _require(cosmology["engine"], "class", "cosmology engine")
    parameters = _mapping(
        cosmology,
        "fundamental_inputs",
        "omega_b_h2 omega_cdm_h2 h A_s n_s alpha_s N_ur N_ncdm omega_ncdm_h2 tau_reio w0 wa",
    )
    for name, value in parameters.items():
        _number(
            value,
            name,
            integer=name == "N_ncdm",
            minimum=None if name in {"alpha_s", "w0", "wa"} else 0,
        )
    for key in ("h", "A_s", "omega_b_h2", "omega_cdm_h2"):
        _number(parameters[key], key, positive=True)
    for group in ("official_derived_reference", "realized_engine_fingerprint"):
        record = _mapping(cosmology, group)
        for key in ("sigma8_m", "sigma8_cb"):
            _number(record[key], key, positive=True)
    tolerances = _mapping(
        cosmology,
        "verification_absolute_tolerances",
        (set(parameters) - {"N_ncdm"}) | {"sigma8_m", "sigma8_cb"},
    )
    for key, value in tolerances.items():
        _number(value, key, positive=True)

    tracers = _mapping(payload, "tracers", _TRACERS)
    bin_names = set()
    for tracer in tracers:
        spec = _mapping(
            tracers,
            tracer,
            "catalogue footprint field measurement_bins source_reference reservoir reconstruction fkp_p0_h3_mpc3",
        )
        field = _mapping(spec, "field", "zmin zmax")
        zmin, zmax = _bounds([field["zmin"], field["zmax"]], "parent redshift", minimum=0)
        cursor = zmin
        for item in _sequence(spec["measurement_bins"], "measurement_bins"):
            if not isinstance(item, Mapping) or set(item) != {"name", "zmin", "zmax"}:
                raise ValueError("measurement bin fields are not exact")
            lo, hi = _bounds([item["zmin"], item["zmax"]], "bin redshift", minimum=0)
            if lo != cursor or not isinstance(item["name"], str) or item["name"] in bin_names:
                raise ValueError("named measurement bins must contiguously partition their parent")
            bin_names.add(item["name"])
            cursor = hi
        if cursor != zmax:
            raise ValueError("measurement bins do not cover their parent")
        _number(spec["fkp_p0_h3_mpc3"], "FKP P0", positive=True)
        catalogue = _mapping(
            spec,
            "catalogue",
            "base_url cap_strategy data_template random_template random_splits source_redshift_column "
            "source_redshift_semantics data_required_columns random_required_columns",
        )
        _require(catalogue["cap_strategy"], "concat_ngc_sgc", "catalogue cap layout")
        _require(catalogue["source_redshift_column"], "Z", "source column")
        _require(catalogue["source_redshift_semantics"], "rsd_pre_photoz", "source redshift")
        _require(catalogue["data_required_columns"], ["TARGETID", "RA", "DEC", "Z"], "data columns")
        _require(catalogue["random_required_columns"], ["RA", "DEC", "Z"], "random columns")
        _number(catalogue["random_splits"], "random_splits", integer=True, positive=True)
        for key, fields in (
            ("data_template", {"mock_id", "cap"}),
            ("random_template", {"mock_id", "cap", "split"}),
        ):
            names = [field for _, field, _, _ in Formatter().parse(catalogue[key]) if field]
            if len(names) != len(fields) or set(names) != fields:
                raise ValueError(f"{key} must bind exactly its source coordinates")
        footprint = _mapping(spec, "footprint", "provenance area_deg2 estimator")
        areas = _mapping(footprint, "area_deg2", "NGC SGC")
        for value in areas.values():
            _number(value, "area", positive=True)
        estimator = _mapping(
            footprint,
            "estimator",
            "table_selector split_ids split_aggregation cross_mock_rule rows_per_square_degree normalization_provenance",
        )
        _require(
            estimator["split_ids"], list(range(catalogue["random_splits"])), "footprint splits"
        )
        if any(type(value) is not int for value in estimator["split_ids"]):
            raise ValueError("footprint split identifiers must be integers")
        _require(estimator["split_aggregation"], "arithmetic_mean", "footprint mean")
        _require(
            estimator["cross_mock_rule"], "exact_integer_split_sum_equality", "footprint comparison"
        )
        _number(estimator["rows_per_square_degree"], "random surface density", positive=True)
        source = _mapping(
            spec,
            "source_reference",
            "source_template source_basename checksum_manifest_template source_required_columns data_required_columns",
        )
        _require(source["source_required_columns"], ["TARGETID", "RSDZ"], "reference columns")
        _require(source["data_required_columns"], ["TARGETID", "Z"], "comparison columns")
        reservoir = _mapping(spec, "reservoir")
        _require(reservoir["source_population"], "same_Abacus_parent", "reservoir source")
        _require(reservoir["source_redshift_column"], "RSDZ", "reservoir source redshift")
        _require(reservoir["active_redshift_column"], "Z", "reservoir active redshift")
        _require(reservoir["filename"], source["source_basename"], "native reservoir parent")
        _require(
            reservoir["hardware_program"],
            "dark" if tracer == "LRG" else "bright",
            "hardware program",
        )
        flag = _mapping(reservoir, "target_flag", "column bit")
        _require(
            flag["column"], "DESI_TARGET" if tracer == "LRG" else "BGS_TARGET", "target flag column"
        )
        _require(flag["bit"], 0 if tracer == "LRG" else 1, "target flag bit")
        required = {
            "TARGETID",
            "RA",
            "DEC",
            "RSDZ",
            "MASKBITS",
            "COLLISION",
            "TILEID",
            "LOCATION",
            flag["column"],
        }
        if tracer != "LRG":
            required.add("R_MAG_ABS")
        if (
            set(
                _sequence(
                    reservoir["parent_required_columns"], "native parent columns", unique=True
                )
            )
            != required
        ):
            raise ValueError("native parent columns differ from the configured tracer")
        expected = {(zmin, "below"), (zmax, "above")}
        edges = _sequence(reservoir["edges"], "reservoir edges")
        if len(edges) != 2 or {(item["redshift"], item["side"]) for item in edges} != expected:
            raise ValueError("reservoir must cover both parent boundaries")
        for edge in edges:
            if set(edge) - {"redshift", "side", "slice_half_width_redshift"}:
                raise ValueError("unknown reservoir edge fields")
            if "slice_half_width_redshift" in edge:
                _number(edge["slice_half_width_redshift"], "edge extraction width", positive=True)
        for key in (
            "slice_half_width_redshift",
            "guard_n_sigma",
            "minimum_guard_width_redshift",
            "calibration_band_width_redshift",
        ):
            _number(reservoir[key], key, positive=True)
        for bit in _sequence(reservoir["veto_maskbits"], "veto bits", unique=True):
            _number(bit, "veto bit", integer=True, minimum=0)
        _require(
            reservoir["imaging_map_tracer"],
            "QSO" if tracer == "LRG" else "BGS_BRIGHT",
            "publisher imaging map tracer",
        )
        if tracer == "LRG":
            mask = _mapping(
                reservoir,
                "target_mask",
                "filename_template checksum_manifest column accepted_value",
            )
            _require(mask["column"], "lrg_mask", "target mask column")
            _require(mask["accepted_value"], 0, "target mask accepted value")
            if "{mock_id}" not in mask["filename_template"]:
                raise ValueError("target mask must identify its realization")
        else:
            _number(reservoir["absolute_magnitude_offset"], "magnitude offset")
            _number(reservoir["absolute_magnitude_cut"], "magnitude cut")
        recon = _mapping(spec, "reconstruction", "bias_source fiducial_bias")
        _require(recon["bias_source"], "fiducial_sourced_table", "bias source")
        _number(recon["fiducial_bias"], "fiducial bias", positive=True)

    analysis = _mapping(
        payload, "analysis", "source_redshift_comparison execution_environment mock_ids workloads"
    )
    comparison = _mapping(
        analysis, "source_redshift_comparison", "absolute_tolerance duplicate_policy"
    )
    _number(comparison["absolute_tolerance"], "redshift comparison tolerance", minimum=0)
    _require(comparison["duplicate_policy"], "finite_exactly_identical", "source duplicates")
    environment = _mapping(
        analysis,
        "execution_environment",
        "packages thread_environment_variables record_threadpoolctl",
    )
    for entry in _sequence(environment["packages"], "packages"):
        if set(entry) != {"distribution", "import_module"} or any(
            not isinstance(value, str) or not value for value in entry.values()
        ):
            raise ValueError("package metadata must identify a distribution and import module")
    _sequence(environment["thread_environment_variables"], "thread variables", unique=True)
    _require(environment["record_threadpoolctl"], True, "thread pool reporting")
    identifiers = _sequence(analysis["mock_ids"], "mock IDs", unique=True)
    for identifier in identifiers:
        _number(identifier, "mock ID", integer=True, minimum=0)

    selection = _mapping(
        payload,
        "selection",
        "reservoir_hardware reservoir_imaging boundary_support stable_randomness catastrophic_outlier_fraction primary_density_policy density_diagnostic tiers radial_randoms"
        + (" density_run" if "density_run" in payload["selection"] else "")
        + (" veff_matched_density" if version >= 17 else ""),
    )
    imaging = _mapping(
        selection, "reservoir_imaging", "base_url filename_template nside ordering cuts sha256"
    )
    hardware = _mapping(
        selection,
        "reservoir_hardware",
        "base_url references tile_location_multiplier missing_zwarn veto_zwarn_mtl_bits maximum_target_id north_declination_min_deg north_galactic_latitude_min_deg",
    )
    for key in ("tile_location_multiplier", "missing_zwarn", "maximum_target_id"):
        _number(hardware[key], key, integer=True, positive=True)
    for key in ("north_declination_min_deg", "north_galactic_latitude_min_deg"):
        if abs(_number(hardware[key], key)) > 90:
            raise ValueError("hardware angular boundary must be within +/-90 degrees")
    for bit in _sequence(hardware["veto_zwarn_mtl_bits"], "hardware warning bits", unique=True):
        _number(bit, "hardware warning bit", integer=True, minimum=0)
    for reference in _mapping(hardware, "references", "dark bright bad_fibers").values():
        if set(reference) != {"filename", "sha256"}:
            raise ValueError("hardware reference has missing or unknown fields")
        if (
            not isinstance(reference["filename"], str)
            or not reference["filename"]
            or "/" in reference["filename"]
        ):
            raise ValueError("hardware reference requires a basename")
        digest = reference["sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise ValueError("invalid hardware reference checksum")
    nside = _number(imaging["nside"], "imaging nside", integer=True, positive=True)
    if nside & (nside - 1):
        raise ValueError("imaging nside must be a power of two")
    _require(imaging["ordering"], "nested", "imaging map ordering")
    cuts = _mapping(
        imaging,
        "cuts",
        "EBV STARDENS PSFSIZE_G PSFSIZE_R PSFSIZE_Z GALDEPTH_G GALDEPTH_R GALDEPTH_Z PSFDEPTH_W1",
    )
    for name, cut in cuts.items():
        operator = (
            "dust_corrected_gt" if "DEPTH" in name else ("log10_lt" if name == "STARDENS" else "lt")
        )
        _require(cut["operator"], operator, "imaging cut operator")
        _mapping(
            cuts,
            name,
            (
                "operator value extinction_coefficient"
                if operator == "dust_corrected_gt"
                else "operator value"
            ),
        )
        _number(cut["value"], "imaging cut", positive=True)
        if cut["operator"] == "dust_corrected_gt":
            _number(cut["extinction_coefficient"], "extinction coefficient", positive=True)
    for maps in _mapping(imaging, "sha256", "BGS_BRIGHT QSO").values():
        if set(maps) != {"N", "S"}:
            raise ValueError("imaging maps must cover N and S")
        for digest in maps.values():
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)
            ):
                raise ValueError("invalid imaging map checksum")
    boundary = _mapping(selection, "boundary_support")
    optional = {"experiment_design", "experiment"}
    _require(
        {key: value for key, value in boundary.items() if key not in optional},
        {
            "clustering": "complete",
            "overlap_precedence": "clustering",
            "physical_identity_columns": ["RA", "DEC", "Z"],
            "insufficient_support": "retain_and_report",
        },
        "boundary support",
    )
    if "experiment_design" in boundary:
        design = _mapping(
            boundary,
            "experiment_design",
            "reference_condition rehearsal_condition targets extension_tracer",
        )
        scales = {tier["label"]: tier["sigma_z_over_one_plus_z"] for tier in selection["tiers"]}
        labels = set(scales)
        if any(design[key] not in labels for key in ("reference_condition", "rehearsal_condition")):
            raise ValueError("boundary design condition is absent")
        if not 0 < scales[design["rehearsal_condition"]] < scales[design["reference_condition"]]:
            raise ValueError(
                "boundary rehearsal requires smaller positive scatter than its reference"
            )
        targets = _sequence(design["targets"], "boundary targets")
        if len({(item["tracer"], item["side"]) for item in targets}) != len(targets):
            raise ValueError("boundary targets must be distinct")
        for target in targets:
            _mapping({"target": target}, "target", "tracer side measurement_bin")
            if target["tracer"] not in _TRACERS or target["side"] not in {"below", "above"}:
                raise ValueError("invalid boundary target")
            bins = payload["tracers"][target["tracer"]]["measurement_bins"]
            item = bins[0] if target["side"] == "below" else bins[-1]
            if target["measurement_bin"] != item["name"]:
                raise ValueError("boundary target must be the affected edge bin")
        if design["extension_tracer"] not in {target["tracer"] for target in targets}:
            raise ValueError("extension tracer is not a boundary target")
    if "experiment" in boundary:
        experiment = _mapping(
            boundary,
            "experiment",
            "mode tracer condition side depth_redshift reference_depth_redshift reference",
        )
        design = _mapping(boundary, "experiment_design")
        mode = experiment["mode"]
        if mode not in {"truncate", "extend"}:
            raise ValueError("invalid boundary experiment mode")
        if not any(
            target["tracer"] == experiment["tracer"] and target["side"] == experiment["side"]
            for target in design["targets"]
        ):
            raise ValueError("experiment does not match a boundary target")
        condition = design["rehearsal_condition" if mode == "truncate" else "reference_condition"]
        if experiment["condition"] != condition:
            raise ValueError("experiment condition differs from its design")
        depth = _number(experiment["depth_redshift"], "experiment depth", positive=True)
        previous = _number(experiment["reference_depth_redshift"], "reference depth", positive=True)
        if (mode == "truncate" and depth >= previous) or (mode == "extend" and depth <= previous):
            raise ValueError("experiment depth has the wrong direction")
        if mode == "extend" and experiment["tracer"] != design["extension_tracer"]:
            raise ValueError("extension uses the wrong tracer")
        reference = _mapping(
            experiment, "reference", "root run_name code_commit analysis_config_sha256"
        )
        if not all(isinstance(value, str) and value for value in reference.values()):
            raise ValueError("experiment requires its reference identity")
        for key, length in (("code_commit", 40), ("analysis_config_sha256", 64)):
            if len(reference[key]) != length or any(
                c not in "0123456789abcdef" for c in reference[key]
            ):
                raise ValueError("invalid boundary reference identity")
    stable = _mapping(
        selection,
        "stable_randomness",
        "identity_column random_coordinate_significant_digits photoz_deviate_column selection_rank_column "
        "random_thinning_rank_column source_redshift_column observed_redshift_column seed",
    )
    _require(stable["identity_column"], "SOURCE_ID", "catalogue identity")
    _number(stable["seed"], "selection seed", integer=True, minimum=0)
    _number(
        stable["random_coordinate_significant_digits"],
        "coordinate precision",
        integer=True,
        positive=True,
    )
    names = [value for key, value in stable.items() if key.endswith("_column")]
    if len(names) != len(set(names)) or any(
        not isinstance(name, str) or not name for name in names
    ):
        raise ValueError("scientific columns must be distinct names")
    _require(
        selection["primary_density_policy"],
        "spherex_motivated_bounded_by_attainable",
        "density policy",
    )
    # The only implemented injection has no catastrophic population.
    _number(selection["catastrophic_outlier_fraction"], "outlier fraction", minimum=0)
    if selection["catastrophic_outlier_fraction"] != 0:
        raise ValueError("catastrophic outlier injection is not implemented")
    tiers = _sequence(selection["tiers"], "tiers")
    labels, sigmas = [], []
    for index, tier in enumerate(tiers):
        label = tier["label"]
        if not isinstance(label, str) or not label or label in labels:
            raise ValueError("tier names must be unique nonempty strings")
        labels.append(label)
        sigmas.append(_number(tier["sigma_z_over_one_plus_z"], "photo-z sigma", minimum=0))
        target = tier["target_nbar_h3_mpc3"]
        if index == 0:
            if sigmas[-1] != 0 or target is not None:
                raise ValueError("first tier must be native without a density target")
        else:
            if (
                sigmas[-1] <= sigmas[-2]
                or not isinstance(target, Mapping)
                or set(target) != bin_names
            ):
                raise ValueError("ordered non-native tiers must target every measurement bin")
            for value in target.values():
                _number(value, "target density", positive=True)
    if version >= 17:
        _validate_matched_density(selection, labels, tracers)
    workload_names = "pilot full_grid boundary" if "experiment" in boundary else "pilot full_grid"
    transfer = payload["measurement"]["reconstruction"].get("transfer_run")
    if transfer is not None:
        workload_names += " transfer"
    sensitivity = payload["measurement"]["reconstruction"].get("sensitivity_run")
    if isinstance(sensitivity, Mapping) and sensitivity.get("kind") in SENSITIVITY_KINDS:
        workload_names += " " + sensitivity["kind"]
    if "density_run" in selection:
        workload_names += " density"
    for name, workload in _mapping(analysis, "workloads", workload_names).items():
        if set(workload) != {
            "measurement_bins",
            "sigma_z_tiers",
            "reconstruction_states",
            "mock_ids",
        }:
            raise ValueError(f"{name} workload has unknown fields")
        for key, choices in (
            ("measurement_bins", bin_names),
            ("sigma_z_tiers", labels),
            ("reconstruction_states", {"prereconstruction", "postreconstruction"}),
        ):
            values = _sequence(workload[key], key, unique=True)
            if not set(values) <= set(choices):
                raise ValueError(f"{name} workload contains an unknown {key}")
        expected_mocks = (
            payload["measurement"]["reconstruction"]["exploratory_transfer"]["pilot_mock_ids"]
            if name == "transfer" and transfer["pilot"]
            else identifiers
        )
        if workload["mock_ids"] != expected_mocks or any(
            type(value) is not int for value in workload["mock_ids"]
        ):
            raise ValueError("each workload must use the complete configured mock ensemble")
    full = analysis["workloads"]["full_grid"]
    if (
        set(full["measurement_bins"]) != bin_names
        or full["sigma_z_tiers"] != labels
        or set(full["reconstruction_states"]) != {"prereconstruction", "postreconstruction"}
    ):
        raise ValueError("full_grid must cover every bin, tier and reconstruction state")
    radial = _mapping(
        selection,
        "radial_randoms",
        "production_method diagnostic_method support_bin_width_redshift frozen_method frozen_residual_weighting "
        "frozen_bandwidth exact_random_data_ratio stable_thinning_seed review_references",
    )
    _require(
        radial["production_method"], "calibration_smooth_target_exact_thinning", "radial method"
    )
    _require(radial["diagnostic_method"], "shuffled_data_redshifts", "radial diagnostic")
    _require(radial["frozen_method"], "smoothing_spline", "radial fit")
    _require(radial["frozen_residual_weighting"], "inverse_standard_error", "radial weighting")
    for key in ("support_bin_width_redshift", "frozen_bandwidth", "exact_random_data_ratio"):
        _number(radial[key], key, positive=True, integer=key == "exact_random_data_ratio")
    _number(radial["stable_thinning_seed"], "thinning seed", integer=True, minimum=0)
    for value in _mapping(radial, "review_references").values():
        _number(value, "radial review reference", minimum=0)

    measurement = _mapping(
        payload, "measurement", "weights reconstruction xi power_spectrum_secondary"
    )
    _require(
        dict(_mapping(measurement, "weights")),
        {
            "reconstruction": "WEIGHT",
            "pyrecon_kw_weights": None,
            "correlation_function": "WEIGHT*WEIGHT_FKP",
            "power_spectrum_secondary": "WEIGHT*WEIGHT_FKP",
        },
        "measurement weights",
    )
    recon = _mapping(measurement, "reconstruction")
    for key, value in {
        "algorithm": "iterative_fft",
        "line_of_sight": "local",
        # RecSym is the sole primary convention; RecIso exists only as its robustness run.
        "allowed_conventions": (
            ["reciso"]
            if isinstance(sensitivity, Mapping) and sensitivity.get("kind") == "reciso"
            else ["recsym"]
        ),
        "scope": (
            "separate_measurement_bins"
            if version in PER_BIN_SCHEMA_VERSIONS
            else "one_field_per_tracer"
        ),
        "position_type": "pos",
        "wrap_positions": False,
        "resampler": "tsc",
        "fft_plan": "estimate",
        "mesh_dtype": "f8",
    }.items():
        _require(recon[key], value, key)
    for key in ("iterations", "cell_size_mpc_h", "box_padding"):
        _number(recon[key], key, positive=True, integer=key == "iterations")
    if recon["box_padding"] <= 1:
        raise ValueError("box padding must exceed unity")
    smooth = _sequence(recon["smoothing_values_mpc_h"], "smoothing")
    if len(smooth) != 1:
        raise ValueError("one reconstruction smoothing radius is implemented")
    _number(smooth[0], "smoothing radius", positive=True)
    threshold = _mapping(recon, "threshold_randoms", "prescription value support_predicate")
    _require(threshold["prescription"], "pyrecon_native_mean_random_weight", "random threshold")
    _require(
        threshold["support_predicate"], "positive_unsmoothed_tsc_random_weight", "support predicate"
    )
    _number(threshold["value"], "random threshold", positive=True)
    # Historical frozen snapshots predate this inactive exploratory definition.
    if "exploratory_pilot" in recon:
        _validate_reconstruction_pilot(payload, recon)
    if "exploratory_mesh_comparison" in recon:
        _validate_mesh_comparison(payload, recon)
    if "exploratory_transfer" in recon:
        _validate_transfer_design(payload, recon)
    if "exploratory_smoothing_scan" in recon:
        _validate_smoothing_scan(payload, recon)
    if "sensitivity_run" in recon:
        _validate_sensitivity_run(payload, recon)
    if transfer is not None:
        _mapping(
            recon,
            "transfer_run",
            "source_class pilot damping_centres" + (" prior" if "prior" in transfer else ""),
        )
        if "prior" in transfer:
            _require(transfer["prior"], "common_flat_sensitivity", "transfer prior check")
        design = recon["exploratory_transfer"]
        if transfer["source_class"] not in ("self", design["source_class"]):
            raise ValueError("transfer run requires self or the configured high-precision source")
        if type(transfer["pilot"]) is not bool:
            raise ValueError("transfer pilot selector must be boolean")
        radii = [design["common_smoothing_mpc_h"], *design["source_smoothing_checks_mpc_h"]]
        self_radii = [radii[0], *design.get("self_smoothing_checks_mpc_h", [])]
        if smooth[0] not in (self_radii if transfer["source_class"] == "self" else radii):
            raise ValueError("transfer run smoothing is outside the configured matrix")
        # The flat prior is the transfer headline at every configured radius (revision 2.19).
        if (
            "prior" in transfer
            and payload["fitting"]["broadband_model"] != design["primary_broadband"]
        ):
            raise ValueError(
                "the flat transfer prior is a power3 fit, separate from the PCS2 check"
            )
        if "experiment" in boundary:
            raise ValueError("transfer and boundary experiments cannot be combined")
        workload = analysis["workloads"]["transfer"]
        expected_bins = (
            design["pilot_measurement_bins"] if transfer["pilot"] else design["measurement_bins"]
        )
        if workload["measurement_bins"] != expected_bins or not set(
            workload["sigma_z_tiers"]
        ) <= set(design["class_labels"]):
            raise ValueError("transfer workload differs from the configured bins or targets")
        expected_targets = list(design["class_labels"])
        # Pre-reconstruction routes are radius-independent; only common-radius self owns them.
        expected_states = (
            ["prereconstruction", "postreconstruction"]
            if transfer["source_class"] == "self" and smooth[0] == radii[0]
            else ["postreconstruction"]
        )
        # A self run at the same radius already measures the source class on its own field.
        if transfer["source_class"] != "self" and smooth[0] in self_radii:
            expected_targets.remove(design["source_class"])
        if (
            workload["sigma_z_tiers"] != expected_targets
            or workload["reconstruction_states"] != expected_states
        ):
            raise ValueError("transfer workload duplicates or omits a configured measurement route")
        if transfer["damping_centres"] is not None and not isinstance(
            transfer["damping_centres"], Mapping
        ):
            raise ValueError("transfer damping centres must be a frozen artifact or null")
    resources = _mapping(
        recon,
        "resource_policy",
        "single_rank_peak_rss_ceiling_gib source_visible_real_mesh_count required_high_water_measurements mpi_policy",
    )
    for key in (
        "single_rank_peak_rss_ceiling_gib",
        "source_visible_real_mesh_count",
        "required_high_water_measurements",
    ):
        _number(
            resources[key], key, positive=True, integer=key != "single_rank_peak_rss_ceiling_gib"
        )
    _require(resources["mpi_policy"], "prohibit_without_separate_review", "reconstruction MPI")
    validity = _mapping(
        recon,
        "field_validity",
        "reject_empty_field reject_non_finite require_catalogue_containment require_wrapping_disabled "
        "random_support_loss_disposition",
    )
    for key in (
        "reject_empty_field",
        "reject_non_finite",
        "require_catalogue_containment",
        "require_wrapping_disabled",
    ):
        _require(validity[key], True, key)
    _require(
        validity["random_support_loss_disposition"], "reported_never_rejecting", "support reporting"
    )
    for key in ("bias_estimation", "effective_redshift"):
        controls = _mapping(recon, key)
        if controls["calibration_mock_ids"] != identifiers or any(
            type(value) is not int for value in controls["calibration_mock_ids"]
        ):
            raise ValueError(f"{key} requires the complete ensemble")
    zeff = recon["effective_redshift"]
    bias = recon["bias_estimation"]
    _require(bias["role"], "non_primary_calibration_sensitivity", "optional bias role")
    _require(bias["method"], "weighted_large_scale_kaiser_monopole", "optional bias estimator")
    _bounds(bias["scale_range_mpc_h"], "bias scales", minimum=0)
    _number(bias["minimum_scale_points"], "bias scale count", integer=True, minimum=2)
    for key, value in {
        "estimator": "weighted_random_density",
        "positions": "selected_random_redshifts",
        "caps": "pooled_numerator_denominator",
    }.items():
        _require(zeff[key], value, key)
    growth = _mapping(recon, "growth", "method consistency_tolerances fiducial_reference")
    _require(growth["method"], "exact_mock_cosmology_at_frozen_effective_redshift", "growth method")
    for value in _mapping(growth, "consistency_tolerances", "relative absolute").values():
        _number(value, "growth tolerance", positive=True)
    reference = _mapping(
        growth,
        "fiducial_reference",
        "citation reference_redshift_estimator growth_rate_by_tracer table_redshift_range_by_tracer "
        "maximum_fractional_difference_by_tracer",
    )
    _require(
        reference["reference_redshift_estimator"],
        "field_volume_weighted_centroid",
        "reference redshift",
    )
    if not isinstance(reference["citation"], str) or not reference["citation"]:
        raise ValueError("growth reference needs its source citation")
    for key in ("growth_rate_by_tracer", "maximum_fractional_difference_by_tracer"):
        for value in _mapping(reference, key, _TRACERS).values():
            _number(value, key, positive=True)
    for value in _mapping(reference, "table_redshift_range_by_tracer", _TRACERS).values():
        _bounds(value, "reference redshift bounds", minimum=0)
    xi = _mapping(measurement, "xi")
    for key, value in {
        "estimator_prereconstruction": "landy_szalay",
        "estimator_postreconstruction": "shifted_random",
        "line_of_sight": "midpoint",
        "multipoles": [0, 2, 4],
        "cap_combination": "normalize_to_cap_dd_then_sum_pair_counts",
    }.items():
        _require(xi[key], value, key)
    _bounds([xi["s_min_mpc_h"], xi["s_max_mpc_h"]], "native xi edges", minimum=0)
    _number(xi["bin_width_mpc_h"], "xi bin width", positive=True)
    _number(xi["mu_bins"], "mu bins", positive=True, integer=True)
    for value in _mapping(xi, "rebin_parity_tolerances", "relative absolute").values():
        _number(value, "rebin parity tolerance", positive=True)
    power = _mapping(measurement, "power_spectrum_secondary")
    for key, value in {
        "line_of_sight": "firstpoint",
        "multipoles": [0, 2, 4],
        "resampler": "tsc",
        "diagnostic_scope": "ensemble_mean_pre_post",
    }.items():
        _require(power[key], value, key)
    if (
        type(power["required"]) is not bool
        or not set(power["diagnostic_measurement_bins"]) <= bin_names
    ):
        raise ValueError("invalid secondary P(k) scope")
    for key in ("cell_size_mpc_h", "box_padding", "native_k_max_h_mpc", "interlacing"):
        _number(power[key], key, positive=True, integer=key == "interlacing")
    comparison = _mapping(power, "comparison_binning", "k_min_h_mpc k_max_h_mpc bin_width_h_mpc")
    _bounds([comparison["k_min_h_mpc"], comparison["k_max_h_mpc"]], "comparison k", minimum=0)
    _number(comparison["bin_width_h_mpc"], "k bin width", positive=True)
    if comparison["k_max_h_mpc"] > power["native_k_max_h_mpc"]:
        raise ValueError("comparison k range exceeds native support")

    fitting = _mapping(payload, "fitting")
    fit_edges = _bounds(fitting["fit_edge_limits_mpc_h"], "fit edges", minimum=0)
    width = _number(fitting["input_bin_width_mpc_h"], "fit bin width", positive=True)
    ratios = [
        width / xi["bin_width_mpc_h"],
        (fit_edges[1] - fit_edges[0]) / width,
        (xi["s_max_mpc_h"] - xi["s_min_mpc_h"]) / xi["bin_width_mpc_h"],
    ]
    ratios.extend((edge - xi["s_min_mpc_h"]) / xi["bin_width_mpc_h"] for edge in fit_edges)
    if (
        fit_edges[0] < xi["s_min_mpc_h"]
        or fit_edges[1] > xi["s_max_mpc_h"]
        or any(not math.isclose(value, round(value), rel_tol=0, abs_tol=1e-10) for value in ratios)
    ):
        raise ValueError("fit edges and width must exactly coarsen the native xi grid")
    _require(fitting["ap_modes"], ["qiso", "qisoqap"], "AP modes")
    _require(fitting["methods"], ["profiling", "mcmc", "detection"], "fit methods")
    for tracer, surface in _mapping(fitting, "fit_surfaces", _TRACERS).items():
        legacy_bgs = version == 13 and tracer == "BGS_BRIGHT-21.5"
        expected = {
            "observed_multipoles": [0] if legacy_bgs else [0, 2],
            "model_multipoles": [0, 2, 4],
            "ap_mode": "qiso" if legacy_bgs else "qisoqap",
            "fixed_parameters": {"qap": 1.0, "dbeta": 1.0} if legacy_bgs else {"dbeta": 1.0},
        }
        _require(surface, expected, "tracer fit surface")
    broadband = "power3" if version >= 15 else "pcs2"
    check = fitting.get("core_fit_check")
    if check is not None:
        if (
            version not in PER_BIN_SCHEMA_VERSIONS
            or transfer is not None
            or check
            not in (
                "core_bin_pcs2_v1",
                "core_bin_flat_v1",
                "core_bin_common_flat_v1",
                "core_scan_radius_centre_v1",
                "unified_prior_pcs2_v1",
                "unified_prior_flat_v1",
            )
        ):
            raise ValueError("unknown per-bin core fit check")
        if check == "core_scan_radius_centre_v1" and recon.get("sensitivity_run") != {
            "kind": "smoothing_scan"
        }:
            raise ValueError("the scan centre refit belongs to smoothing-scan runs only")
        applied = fitting["damping"].get("applied_prior")
        if check in ("core_bin_pcs2_v1", "unified_prior_pcs2_v1"):
            broadband = "pcs2"
            if (check == "unified_prior_pcs2_v1") != (applied == "unified_bin_prior_v1"):
                raise ValueError("the revision 2.20 PCS2 check keeps the configured damping priors")
        elif (
            check in ("core_bin_flat_v1", "unified_prior_flat_v1")
            and applied != "paper_finite_flat"
        ):
            raise ValueError("per-bin flat check requires the finite flat prior")
        elif check == "core_bin_common_flat_v1" and applied != "common_flat_sensitivity":
            raise ValueError("per-bin common flat check requires the common flat prior")
    if version == 15 and transfer is not None and fitting["broadband_model"] == "pcs2":
        if smooth[0] != recon["exploratory_transfer"]["common_smoothing_mpc_h"]:
            raise ValueError("transfer PCS2 is a common-radius fit check")
        broadband = recon["exploratory_transfer"]["broadband_check"]
    _require(fitting["broadband_model"], broadband, "broadband")
    if "paper_comparisons" in fitting:
        comparison = _mapping(
            fitting,
            "paper_comparisons",
            "condition measurement_bins reconstruction_state flat_damping broadband_alternative",
        )
        _require(comparison["condition"], "sigma_0.003", "paper comparison tier")
        _require(
            comparison["measurement_bins"],
            ["BGS1", "BGS2", "LRG1", "LRG2", "LRG3"],
            "paper comparison bins",
        )
        _require(comparison["reconstruction_state"], "postreconstruction", "comparison state")
        _require(comparison["broadband_alternative"], "pcs2", "comparison broadband")
        flat = _mapping(
            comparison,
            "flat_damping",
            "distribution lower_mpc_h nonlinear_upper_mpc_h parallel_upper",
        )
        _require(flat["distribution"], "uniform", "comparison prior family")
        _require(flat["parallel_upper"], "quadrature_with_pairwise_photoz", "parallel support")
        _number(flat["lower_mpc_h"], "flat lower support", minimum=0)
        _number(flat["nonlinear_upper_mpc_h"], "flat upper support", positive=True)
        if flat["lower_mpc_h"] >= flat["nonlinear_upper_mpc_h"]:
            raise ValueError("flat damping support is empty")
    response = _mapping(fitting, "photoz_response", "primary primary_scope mu_quadrature_order")
    _require(
        response["primary"],
        "realized_full_signal_fixed_observed_fiducial_coordinates",
        "photo-z response",
    )
    _require(response["primary_scope"], "full_signal", "photo-z scope")
    _number(response["mu_quadrature_order"], "mu quadrature", integer=True, positive=True)
    nonlinear = _mapping(fitting, "nonlinear_parameters", "qiso qap b1 dbeta")
    for name, parameter in nonlinear.items():
        _number(parameter["value"], name)
        for key in ("prior", "reference"):
            if name == "dbeta":
                _require(parameter[key], None, "fixed dbeta prior")
            else:
                prior = _mapping(parameter, key, "dist limits")
                _require(prior["dist"], "uniform", "nonlinear prior")
                limits = _bounds(prior["limits"], name + " " + key)
                if not limits[0] <= parameter["value"] <= limits[1]:
                    raise ValueError("initial parameter must lie inside its prior and reference")
        for key in ("proposal", "finite_difference_step"):
            if parameter[key] is not None:
                _number(parameter[key], key, positive=True)
    _require(nonlinear["dbeta"]["value"], 1.0, "fixed dbeta")
    _require(fitting["no_wiggle_method"], "peakaverage", "smooth template")
    _require(
        fitting["fiducial_policy"], "exact_mock_cosmology_expected_alpha_unity", "fit fiducial"
    )
    _require(
        dict(_mapping(fitting, "ap_interval_parameters")),
        {"qiso": ["qiso"], "qisoqap": ["qiso", "qap"]},
        "AP intervals",
    )
    _require(
        dict(_mapping(fitting, "nonlinear_parameter_names_by_ap_mode")),
        {"qiso": ["qiso", "b1"], "qisoqap": ["qiso", "qap", "b1"]},
        "nonlinear parameter axes",
    )
    nuisance = _mapping(
        fitting,
        "linear_nuisance",
        "parameter_patterns varied_basenames_by_tracer allowed_basenames_by_model value prior reference "
        "finite_difference_step treatment",
    )
    _require(nuisance["parameter_patterns"], ["al*_*", "bl*_*"], "nuisance patterns")
    axes = list(POWER3_NUISANCE_AXES) if broadband == "power3" else list(PCS2_NUISANCE_AXES)
    _require(
        dict(_mapping(nuisance, "varied_basenames_by_tracer")),
        {
            "BGS_BRIGHT-21.5": ["bl0_0", "bl0_2"] if version == 13 else axes,
            "LRG": axes,
        },
        "nuisance axes",
    )
    allowed = _mapping(nuisance, "allowed_basenames_by_model", broadband)
    _require(allowed[broadband], axes, "allowed nuisance axes")
    _require(nuisance["prior"], None, "linear nuisance prior")
    _number(nuisance["value"], "linear nuisance value")
    _number(nuisance["finite_difference_step"], "nuisance finite difference", positive=True)
    _require(
        dict(_mapping(nuisance, "treatment")),
        {"profiling": ".auto", "mcmc": ".prec", "detection": ".auto"},
        "nuisance treatment",
    )
    damping = _mapping(fitting, "damping")
    if damping["registered_sigma_z_over_one_plus_z"] != sigmas:
        raise ValueError("damping and injected photo-z tiers differ")
    for key, value in {
        "postreconstruction_sigma_perp_policy": "analytic_producer_is_authoritative",
        "width_authority": "fixed_gaussian_widths_mpc_h",
        "assessment": "post_fit_prior_posterior_overlay_only",
    }.items():
        _require(damping[key], value, key)
    names = {"sigma_perp", "sigma_parallel", "sigma_streaming"}
    for value in _mapping(damping, "gaussian_widths_mpc_h", names).values():
        _number(value, "damping prior width", positive=True)
    for tracer in _TRACERS:
        centers = _mapping(
            damping["baseline_centers_mpc_h"], tracer, "prereconstruction postreconstruction"
        )
        for state in centers:
            for value in _mapping(centers, state, names).values():
                _number(value, "damping center", minimum=0)
    _number(damping["nonlinear_minimum_mpc_h"], "damping minimum", minimum=0)
    # Optional inference choices: the finite or common flat robustness prior and a pinned radius
    # for the analytic post-reconstruction transverse centre in radius comparisons.
    # The configured per-bin prior (revision 2.20) applies to every run, route and state.
    if "applied_prior" in damping and damping["applied_prior"] != "unified_bin_prior_v1":
        if damping["applied_prior"] not in ("paper_finite_flat", "common_flat_sensitivity"):
            raise ValueError(f"unknown applied damping prior: {damping['applied_prior']!r}")
        if "paper_comparisons" not in fitting:
            raise ValueError("a flat damping prior requires its paper comparison scope")
        if damping["applied_prior"] == "common_flat_sensitivity":
            if fitting.get("core_fit_check") != "core_bin_common_flat_v1":
                raise ValueError("the common flat damping prior belongs to the per-bin check only")
            if "common_flat_sensitivity" not in damping:
                raise ValueError("the common flat damping prior requires its configured support")
    if "reference_smoothing_mpc_h" in damping:
        _number(damping["reference_smoothing_mpc_h"], "damping reference radius", positive=True)
    if "common_flat_sensitivity" in damping:
        flat = _mapping(
            damping,
            "common_flat_sensitivity",
            "specification_revision distribution bounds_mpc_h parameters photoz_response",
        )
        _require(flat["specification_revision"], "2.11-P", "flat damping revision")
        _require(flat["distribution"], "uniform", "flat damping distribution")
        _bounds(flat["bounds_mpc_h"], "common flat damping bounds", minimum=0)
        _require(flat["bounds_mpc_h"], [0.0, 50.0], "common flat damping bounds")
        _require(
            flat["parameters"],
            ["sigma_perp", "sigma_parallel", "sigma_streaming"],
            "joint flat damping coordinates",
        )
        _require(flat["photoz_response"], "unchanged_separate", "flat damping photo-z response")

    covariance = _mapping(
        fitting,
        "covariance",
        "primary_source route_key prohibit_cross_route_transfer realization_scaling mean_vector_scaling "
        "prohibit_study_mock_covariance permit_study_mock_matrix_for_scalar_calibration shot_noise_rescaling rascalc",
    )
    for key, value in {
        "primary_source": "route_level_rascalc",
        "route_key": ["measurement_bin", "sigma_z_tier", "reconstruction_state"],
        "realization_scaling": "shared_route_covariance",
        "mean_vector_scaling": "route_covariance_over_realization_count",
        "prohibit_cross_route_transfer": True,
        "prohibit_study_mock_covariance": True,
        "permit_study_mock_matrix_for_scalar_calibration": True,
    }.items():
        _require(covariance[key], value, key)
    shot_noise = _mapping(
        covariance,
        "shot_noise_rescaling",
        "source model objective objective_realization_count frozen_registry tuning",
    )
    _require(shot_noise["source"], "route_mock_scatter", "shot-noise scalar source")
    if type(shot_noise["objective_realization_count"]) is not int or shot_noise[
        "objective_realization_count"
    ] != len(identifiers):
        raise ValueError("scalar objective must use the complete ensemble")
    tuning = _mapping(shot_noise, "tuning", "search_limits search_points status")
    if tuning["search_limits"] is None or tuning["search_points"] is None:
        if tuning["search_limits"] is not None or tuning["search_points"] is not None:
            raise ValueError("both scalar tuning controls must be supplied together")
        _require(tuning["status"], "MANUAL DECISION REQUIRED", "unresolved scalar tuning")
    else:
        limits = _bounds(tuning["search_limits"], "scalar search", minimum=0)
        if limits[0] == 0:
            raise ValueError("scalar search must remain positive")
        _number(tuning["search_points"], "scalar search points", integer=True, minimum=2)
        if tuning["status"] == "MANUAL DECISION REQUIRED":
            raise ValueError("scalar tuning status disagrees with its supplied controls")
    rascalc = _mapping(
        covariance,
        "rascalc",
        "interface public_argument_count public_arguments run_arguments frozen_integration "
        "runtime_arguments explicit_null_arguments escalation_ladder_n_loops escalation_eligible_failures route_geometry",
    )
    _require(rascalc["interface"], "RascalC.interface:run_cov", "RascalC interface")
    _require(rascalc["public_arguments"], _RASCALC_ARGUMENTS, "RascalC arguments")
    _require(rascalc["public_argument_count"], len(_RASCALC_ARGUMENTS), "RascalC argument count")
    arguments = _mapping(rascalc, "run_arguments", _RASCALC_ARGUMENTS)
    _require(arguments["mode"], "legendre_projected", "RascalC mode")
    _require(arguments["position_type"], "pos", "RascalC position type")
    _require(arguments["normalize_wcounts"], True, "RascalC normalization")
    _require(arguments["max_l"], max(xi["multipoles"]), "RascalC raw multipoles")
    for key in ("N2", "N3", "N4", "sampling_grid_size"):
        _number(arguments[key], key, integer=True, positive=True)
    for key in ("xi_cut_s", "coordinate_scaling"):
        _number(arguments[key], key, positive=True)
    _number(arguments["xi_refinement_iterations"], "xi refinement", integer=True, minimum=0)
    runtime = _sequence(rascalc["runtime_arguments"], "runtime arguments", unique=True)
    nulls = _sequence(rascalc["explicit_null_arguments"], "null arguments", unique=True)
    if set(runtime) & set(nulls) or not set(runtime + nulls) <= set(arguments):
        raise ValueError("RascalC runtime and null arguments must be disjoint known arguments")
    if any(arguments[key] is not None for key in runtime + nulls):
        raise ValueError("runtime and unused RascalC arguments cannot carry scientific overrides")
    integration = _mapping(rascalc, "frozen_integration", "n_loops loops_per_sample seed")
    for key, value in integration.items():
        _number(value, key, integer=True, minimum=0 if key == "seed" else 1)
    ladder = _sequence(rascalc["escalation_ladder_n_loops"], "numerical ladder", unique=True)
    if ladder != sorted(ladder) or ladder[0] != integration["n_loops"]:
        raise ValueError("ladder must start at the configured integration and increase")
    for depth in ladder:
        _number(depth, "integration depth", integer=True, positive=True)
        if depth % integration["loops_per_sample"]:
            raise ValueError("integration depth must contain complete subsamples")
    geometry = _mapping(
        rascalc,
        "route_geometry",
        "selection_rule registered_selection_rules explicit_realization_id",
    )
    _require(
        geometry["registered_selection_rules"],
        ["lowest_realization_identifier_in_the_route", "explicit_realization_identifier"],
        "geometry rules",
    )
    if geometry["selection_rule"] not in geometry["registered_selection_rules"]:
        raise ValueError("geometry selection rule must be supplied")
    explicit = geometry["explicit_realization_id"]
    if geometry["selection_rule"] == "explicit_realization_identifier":
        if type(explicit) is not int or explicit not in identifiers:
            raise ValueError("explicit geometry must name a configured integer realization")
    elif explicit is not None:
        raise ValueError("lowest-ID geometry cannot specify another realization")

    optimization = _mapping(fitting, "optimization", "random_seed profiler_iterations")
    _number(optimization["random_seed"], "optimizer seed", integer=True, minimum=0)
    _number(optimization["profiler_iterations"], "optimizer starts", integer=True, positive=True)
    sampling = _mapping(
        fitting,
        "sampling",
        "ensembles walkers seed random_stream_derivation minimum_iterations absolute_iteration_ceiling "
        "check_every burnin_fraction nsplits max_eigen_gr stable_over autocorrelation",
    )
    _require(sampling["ensembles"], 4, "independent ensembles")
    _require(sampling["nsplits"], sampling["ensembles"], "GR ensembles")
    _require(
        sampling["random_stream_derivation"],
        "SeedSequence_route_coordinates_ensemble_index",
        "sampler streams",
    )
    for key in (
        "walkers",
        "minimum_iterations",
        "absolute_iteration_ceiling",
        "check_every",
        "stable_over",
    ):
        _number(sampling[key], key, integer=True, positive=True)
    _number(sampling["seed"], "sampler seed", integer=True, minimum=0)
    _number(sampling["burnin_fraction"], "burnin", positive=True)
    _number(sampling["max_eigen_gr"], "GR tolerance", positive=True)
    if (
        sampling["burnin_fraction"] >= 1
        or sampling["minimum_iterations"] > sampling["absolute_iteration_ceiling"]
    ):
        raise ValueError("sampler budget or burnin leaves no retained samples")
    autocorrelation = _mapping(sampling, "autocorrelation", "method c tol ess_label")
    _require(autocorrelation["method"], "emcee_sokal", "autocorrelation estimator")
    _require(autocorrelation["ess_label"], "nominal_autocorrelation_based_ess", "ESS label")
    for key in ("c", "tol"):
        _number(autocorrelation[key], key, positive=True)

    reporting_keys = set(
        "gain review_bands alpha_rail_tolerance damping_rail_fraction truth_coverage_policy "
        "detection_sigma_bands".split()
    )
    # Frozen releases predate the optional forecast-only reporting value.
    if "forecast_target_fsky" in _mapping(payload, "reporting"):
        reporting_keys.add("forecast_target_fsky")
        fsky = _number(
            payload["reporting"]["forecast_target_fsky"], "forecast sky fraction", positive=True
        )
        if fsky > 1:
            raise ValueError("forecast sky fraction must not exceed one")
    # Schema 17 writes the reference sky fraction twice, for the matched-density rule and for the
    # reporting forecasts; the two are one value.
    if version >= 17 and payload["reporting"].get("forecast_target_fsky") != (
        payload["selection"]["veff_matched_density"]["reference_f_sky"]
    ):
        raise ValueError(
            "forecast sky fraction must equal the matched-density reference sky fraction"
        )
    # Frozen snapshots without the configured coverage report remain unchanged.
    if "interval_coverage" in _mapping(payload, "reporting"):
        reporting_keys.add("interval_coverage")
        coverage = _mapping(
            payload["reporting"],
            "interval_coverage",
            "interval_kind uncertainty_method uncertainty_confidence_level",
        )
        _require(coverage["interval_kind"], "native_minos", "coverage interval kind")
        _require(coverage["uncertainty_method"], "wilson", "coverage uncertainty method")
        _require(coverage["uncertainty_confidence_level"], 0.68, "coverage confidence level")
    reporting = _mapping(
        payload,
        "reporting",
        reporting_keys,
    )
    gain = _mapping(
        reporting, "gain", "ratio one_dimensional_fom posterior_uncertainty profile_uncertainty"
    )
    _require(gain["ratio"], "post_over_pre", "gain direction")
    _require(gain["one_dimensional_fom"], "uncertainty", "1D FoM")
    if gain["posterior_uncertainty"] not in {None, "posterior_standard_deviation"} or gain[
        "profile_uncertainty"
    ] not in {None, "minos_half_width"}:
        raise ValueError("gain uncertainty statistic is not registered")
    _number(reporting["alpha_rail_tolerance"], "alpha rail reference", positive=True)
    _number(reporting["damping_rail_fraction"], "damping rail reference", positive=True)
    _require(
        reporting["truth_coverage_policy"],
        "descriptive_only_for_25_study_mocks",
        "coverage reporting",
    )
    _bounds(reporting["detection_sigma_bands"], "detection reference bands", minimum=0)
    for group in _mapping(reporting, "review_bands").values():
        for key, values in group.items():
            _bounds(values, key, minimum=0)


def validate_analysis_config(payload: Mapping[str, Any]) -> None:
    """Reject incomplete or incompatible settings with one clear configuration exception."""
    try:
        _validate_config(payload)
        if "density_run" in payload["selection"]:
            from .density import validate_density_run

            validate_density_run(payload)
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"incomplete or malformed analysis configuration: {exc}") from exc


__all__ = ["_UniqueKeyLoader", "validate_analysis_config"]
