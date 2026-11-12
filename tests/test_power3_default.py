"""Power3 production values, frozen-run compatibility and native likelihood equivalence."""

import pzbao.diagnostics.fitting as diag_fitting
from copy import deepcopy
import os
from pathlib import Path
import numpy as np
import pytest
import pzbao.likelihood as fitting_likelihood
from pzbao.config import (
    BGS_ANISOTROPIC,
    derive_inference_config,
    load_analysis_config,
    validate_analysis_config,
)

LEGACY = Path(__file__).with_name("fixtures") / "analysis_schema14.yaml"


def test_primary_changes_only_version_broadband_and_its_coordinates():
    previous = load_analysis_config(LEGACY)
    current = load_analysis_config()
    assert current["schema_version"] == 17
    assert current["fitting"]["broadband_model"] == "power3"
    axes = [f"alobs_{index}" for index in range(8)]
    nuisance = current["fitting"]["linear_nuisance"]
    assert nuisance["allowed_basenames_by_model"] == {"power3": axes}
    assert nuisance["varied_basenames_by_tracer"] == {"BGS_BRIGHT-21.5": axes, "LRG": axes}
    comparable = deepcopy(current)
    assert comparable["measurement"]["reconstruction"]["scope"] == "separate_measurement_bins"
    comparable["measurement"]["reconstruction"]["scope"] = "one_field_per_tracer"
    # This inactive extension must not alter the independent benchmark.
    assert "transfer_run" not in comparable["measurement"]["reconstruction"]
    assert comparable["measurement"]["reconstruction"].pop("exploratory_smoothing_scan") == {
        "specification_revision": "smoothing-scan-20261013-v2",
        "radii_mpc_h": [15.0, 25.0, 35.0, 45.0],
        "measurement_bins": ["BGS1", "BGS2", "LRG1", "LRG2", "LRG3"],
        "sigma_z_tiers": ["sigma_0.003", "sigma_0.01", "sigma_0.03"],
        "reconstruction_states": ["postreconstruction"],
        "reference_smoothing_mpc_h": 15.0,
    }
    assert comparable["measurement"]["reconstruction"].pop("exploratory_transfer") == {
        "specification_revision": "2.11-T2",
        "population": "exclusive_parent_classes",
        "class_labels": ["sigma_0.003", "sigma_0.01", "sigma_0.03"],
        "source_class": "sigma_0.003",
        "assignment_namespace": "exclusive-class-v1",
        "allocation": "joint_minimum_parent_counts",
        "exterior_assignment": "adjacent_shell_fractions",
        "radial_selection": "smooth_parent_forward_prediction",
        "scope": "separate_measurement_bins",
        "source_bias": "parent_tracer",
        "common_smoothing_mpc_h": 25.0,
        "source_smoothing_checks_mpc_h": [15.0, 20.0],
        "self_smoothing_checks_mpc_h": [15.0],
        "measurement_bins": ["BGS1", "BGS2", "LRG1", "LRG2", "LRG3"],
        "pilot_measurement_bins": ["BGS1", "LRG2", "LRG3"],
        "pilot_mock_ids": [0, 1, 2],
        "primary_broadband": "power3",
        "broadband_check": "pcs2",
        "damping_check": "common_flat_sensitivity",
        "postreconstruction_centres": "requires_reviewed_source_target_model",
    }
    # Revision 2.15: the SPHEREx effective area (Feder et al. 2024) over the full sky, and the
    # V_eff-matched density block; the benchmark's own scientific settings are untouched.
    full_sky = 4 * np.pi * (180 / np.pi) ** 2
    assert comparable["reporting"].pop("forecast_target_fsky") == 30000.0 / full_sky
    assert comparable["selection"].pop("veff_matched_density") == {
        "count_authority": "veff_matched_lrg23_exact_counts",
        "reference_area_deg2": 30000.0,
        "reference_f_sky": 30000.0 / full_sky,
        "tiers": ["sigma_0.001", "sigma_0.003"],
        "measurement_bins": ["LRG2", "LRG3"],
        "bounded_allowed": [["sigma_0.003", "LRG2"]],
    }
    # Reporting revision 2.11-T1-C leaves the benchmark's scientific settings intact.
    assert comparable["reporting"].pop("interval_coverage") == {
        "interval_kind": "native_minos",
        "uncertainty_method": "wilson",
        "uncertainty_confidence_level": 0.68,
    }
    assert comparable["fitting"]["damping"].pop("common_flat_sensitivity") == {
        "specification_revision": "2.11-P",
        "distribution": "uniform",
        "bounds_mpc_h": [0.0, 50.0],
        "parameters": ["sigma_perp", "sigma_parallel", "sigma_streaming"],
        "photoz_response": "unchanged_separate",
    }
    assert comparable["fitting"].pop("paper_comparisons") == {
        "condition": "sigma_0.003",
        "measurement_bins": ["BGS1", "BGS2", "LRG1", "LRG2", "LRG3"],
        "reconstruction_state": "postreconstruction",
        "flat_damping": {
            "distribution": "uniform",
            "lower_mpc_h": 0.0,
            "nonlinear_upper_mpc_h": 10.0,
            "parallel_upper": "quadrature_with_pairwise_photoz",
        },
        "broadband_alternative": "pcs2",
    }
    comparable["schema_version"] = previous["schema_version"]
    comparable["fitting"]["broadband_model"] = previous["fitting"]["broadband_model"]
    for key in ("varied_basenames_by_tracer", "allowed_basenames_by_model"):
        comparable["fitting"]["linear_nuisance"][key] = deepcopy(
            previous["fitting"]["linear_nuisance"][key]
        )
    assert comparable == previous


@pytest.mark.parametrize("legacy_version", [13, 14])
def test_old_definitions_remain_readable_and_hybrids_fail(legacy_version):
    previous = load_analysis_config(LEGACY)
    previous["schema_version"] = legacy_version
    if legacy_version == 13:
        previous["fitting"]["fit_surfaces"]["BGS_BRIGHT-21.5"].update(
            observed_multipoles=[0], ap_mode="qiso", fixed_parameters={"qap": 1.0, "dbeta": 1.0}
        )
        previous["fitting"]["linear_nuisance"]["varied_basenames_by_tracer"]["BGS_BRIGHT-21.5"] = [
            "bl0_0",
            "bl0_2",
        ]
    validate_analysis_config(previous)
    original = deepcopy(previous)
    if legacy_version == 13:
        derived = derive_inference_config(previous, BGS_ANISOTROPIC)
        assert derived == load_analysis_config(LEGACY)
        assert previous == original
    previous["fitting"]["broadband_model"] = "power3"
    with pytest.raises(ValueError, match="broadband"):
        validate_analysis_config(previous)


@pytest.mark.parametrize("change", ["duplicate", "missing", "wrong_family"])
def test_new_definition_rejects_changed_nuisance_surface(change):
    current = load_analysis_config()
    nuisance = current["fitting"]["linear_nuisance"]
    if change == "wrong_family":
        current["fitting"]["broadband_model"] = "pcs2"
    elif change == "duplicate":
        nuisance["allowed_basenames_by_model"]["power3"][-1] = "alobs_0"
    else:
        nuisance["varied_basenames_by_tracer"]["LRG"].pop()
    with pytest.raises(ValueError):
        validate_analysis_config(current)


@pytest.mark.parametrize("tracer,redshift", [("LRG", 0.8), ("BGS_BRIGHT-21.5", 0.2)])
@pytest.mark.parametrize("state", ["prereconstruction", "postreconstruction"])
def test_native_production_matches_existing_family_and_linear_solution(
    tracer, redshift, state, tmp_path, monkeypatch
):
    if os.environ.get("BAO_POWER3_NATIVE") != "1":
        pytest.skip("native power3 validation runs only inside a Slurm compute job")
    assert os.environ.get("SLURM_JOB_ID"), "native validation requires an allocated compute job"
    monkeypatch.setenv("DESILIKE_CONFIG_DIR", str(tmp_path))
    current = load_analysis_config()
    old = load_analysis_config(LEGACY)
    edges = np.arange(0.0, 201.0)
    muedges = np.linspace(-1.0, 1.0, 101)
    mu = 0.5 * (muedges[1:] + muedges[:-1])
    radius = 0.5 * (edges[1:] + edges[:-1])
    counts = (
        1
        + 0.35 * (radius[:, None] / 200) * mu[None, :] ** 2
        + 0.15 * (radius[:, None] / 200) ** 2 * mu[None, :] ** 4
    )
    name = "LRG2" if tracer == "LRG" else "BGS1"
    identity = dict(
        sample_identity="power3-native-test",
        tracer=tracer,
        measurement_bin=name,
        reconstruction_state=state,
    )
    geometry = dict(identity, sedges=edges, muedges=muedges, wcounts=counts)
    prior = {
        "sigma_parallel_photoz_mpc_h": 20.0,
        "centers_mpc_h": {"sigma_streaming": 2.0, "sigma_parallel": 8.0, "sigma_perp": 3.0},
        "widths_mpc_h": {"sigma_streaming": 2.0, "sigma_parallel": 2.0, "sigma_perp": 1.0},
        "bounds_mpc_h": {
            name: [0.0, None] for name in ("sigma_streaming", "sigma_parallel", "sigma_perp")
        },
    }
    covariance = np.diag(np.linspace(0.8, 1.2, 52)) * 1e-06
    data = np.linspace(-0.002, 0.002, 52)

    def build(config):
        return fitting_likelihood.build_likelihood(
            data,
            covariance,
            separations_mpc_h=np.arange(50.0, 151.0, 4.0),
            fit_sedges=np.arange(48.0, 153.0, 4.0),
            z_eff=redshift,
            multipoles=[0, 2],
            reconstruction_convention="recsym",
            smoothing_mpc_h=15.0,
            ap_mode="qisoqap",
            method="profiling",
            damping_prior=prior,
            geometry_artifact=geometry,
            config=config,
            **identity,
        )

    _, observable, likelihood, _ = build(current)
    record = observable.power_broadband_repair
    expected = {f"al{ell}_{power}" for ell in (0, 2, 4) for power in (-2, -1, 0)}
    assert set(record["raw_names"]) == expected - {"al4_0"}
    assert record["null_names"] == ["al4_0"]
    assert record["effective_rank"] == 8 and record["full_mu_support"]
    assert len(likelihood.all_params.select(basename="alobs_*")) == 8
    for parameter in likelihood.all_params.select(basename=["al0_*", "al2_*", "al4_*"]):
        assert parameter.fixed and parameter.value == 0
    control = {"broadband": "power3", "nuisance_solver": fitting_likelihood.FAMILY_SOLVER}
    old = diag_fitting.configure_inference(old, control)
    with diag_fitting.likelihood_adapters(fitting_likelihood, control):
        _, reference_observable, reference, _ = build(old)
    np.testing.assert_allclose(
        record["observed_basis"], reference_observable.power_broadband_repair["observed_basis"]
    )
    for treatment in (".best", ".prec"):
        for model in (likelihood, reference):
            for parameter in model.all_params.select(basename="alobs_*"):
                parameter.update(derived=treatment)
        for qiso in (1.0, 1.013):
            point = dict(qiso=qiso, qap=1.0, b1=2.0, sigmas=2.0, sigmapar=8.0, sigmaper=3.0)
            np.testing.assert_allclose(
                likelihood(**point), reference(**point), rtol=1e-09, atol=1e-08
            )
            np.testing.assert_allclose(
                likelihood.flattheory, reference.flattheory, rtol=1e-09, atol=1e-10
            )
            assert likelihood.nvaried == reference.nvaried == 14
    basis = np.asarray(record["observed_basis"])
    point = dict(qiso=1.013, qap=1.0, b1=2.0, sigmas=2.0, sigmapar=8.0, sigmaper=3.0)
    for parameter in likelihood.all_params.select(basename="alobs_*"):
        parameter.update(value=0.0, fixed=True, derived=False)
    likelihood(**point)
    physical = np.asarray(likelihood.flattheory).copy()
    precision_basis = np.linalg.solve(covariance, basis)
    coefficients = np.linalg.solve(
        basis.T @ precision_basis, basis.T @ np.linalg.solve(covariance, data - physical)
    )
    expected_model = physical + basis @ coefficients
    for parameter in likelihood.all_params.select(basename="alobs_*"):
        parameter.update(value=0.0, fixed=False, derived=".best")
    _, derived = likelihood(**point, return_derived=True)
    names = record["coordinate_names"]
    solved = np.array([float(np.asarray(derived[name]).item()) for name in names])
    np.testing.assert_allclose(solved, coefficients, rtol=1e-08, atol=1e-10)
    residual = data - expected_model
    expected_chi2 = residual @ np.linalg.solve(covariance, residual)
    np.testing.assert_allclose(
        np.asarray(likelihood.loglikelihood).ravel()[0],
        -0.5 * expected_chi2,
        rtol=1e-08,
        atol=1e-10,
    )
    likelihood(**point, **dict(zip(names, solved)))
    np.testing.assert_allclose(likelihood.flattheory, expected_model, rtol=1e-08, atol=1e-10)
