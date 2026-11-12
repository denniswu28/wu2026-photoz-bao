"""The configured per-bin damping priors of revision 2.20 (2026-10-20)."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from pzbao.config import (
    derive_inference_config,
    load_analysis_config,
    validate_analysis_config,
    validate_inference_config,
)
from pzbao.config.sensitivity import sensitivity_run_configs
from pzbao.config.transfer import transfer_run_configs
from pzbao.theory import (
    UNIFIED_BIN_DAMPING_MPC_H,
    unified_damping_prior,
)

DEFINITION = "unified_prior_v1"
CONFIGURED = {
    "BGS1": ((6, 3), (13, 9)),
    "BGS2": ((5, 3), (9, 6)),
    "LRG1": ((5, 4), (8, 7)),
    "LRG2": ((5, 3), (8, 6)),
    "LRG3": ((6, 5), (9, 7)),
}


def prior(state, measurement_bin="LRG2", radius=15.0, sigma0=0.03, config=None):
    return unified_damping_prior(
        "LRG",
        state,
        measurement_bin=measurement_bin,
        smoothing_mpc_h=radius,
        sigma_z_over_one_plus_z=sigma0,
        z_eff=0.7,
        e_z_eff=1.4,
        config=config or load_analysis_config(),
    )


def test_the_table_is_the_configured_one_and_the_same_before_and_after_reconstruction():
    assert {
        name: (row["sigma_perp"], row["sigma_parallel"])
        for name, row in UNIFIED_BIN_DAMPING_MPC_H.items()
    } == CONFIGURED
    for name, ((perp, perp_width), (parallel, parallel_width)) in CONFIGURED.items():
        pre, post = (prior(state, name) for state in ("prereconstruction", "postreconstruction"))
        for record in (pre, post):
            assert record["centers_mpc_h"] == {
                "sigma_perp": perp,
                "sigma_parallel": parallel,
                "sigma_streaming": 2.0,
            }
            assert record["widths_mpc_h"] == {
                "sigma_perp": perp_width,
                "sigma_parallel": parallel_width,
                "sigma_streaming": 4.0,
            }
            assert record["unpriored_parameters"] == [] and record["scope"] == "final_science_fit"
        assert {k: v for k, v in pre.items() if k != "reconstruction_state"} == {
            k: v for k, v in post.items() if k != "reconstruction_state"
        }
    # Every redshift-error condition shares the prior; only the photo-z response differs.
    low, high = prior("postreconstruction", sigma0=0.003), prior("postreconstruction")
    assert low["centers_mpc_h"] == high["centers_mpc_h"]
    assert low["sigma_parallel_photoz_mpc_h"] < high["sigma_parallel_photoz_mpc_h"]


def test_the_smoothing_radius_adds_a_constant_to_the_centres_only():
    base = prior("postreconstruction", radius=15.0)
    assert prior("postreconstruction", radius=25.0)["centers_mpc_h"] == base["centers_mpc_h"]
    for radius in (35.0, 45.0):
        for state in ("prereconstruction", "postreconstruction"):
            moved = prior(state, radius=radius)
            assert moved["centers_mpc_h"]["sigma_perp"] == base["centers_mpc_h"]["sigma_perp"] + 1
            assert (
                moved["centers_mpc_h"]["sigma_parallel"]
                == base["centers_mpc_h"]["sigma_parallel"] + 2
            )
            assert moved["widths_mpc_h"] == base["widths_mpc_h"]
    with pytest.raises(ValueError, match="smoothing radius 20"):
        prior("postreconstruction", radius=20.0)
    with pytest.raises(ValueError, match="measurement bin"):
        prior("postreconstruction", measurement_bin="QSO1")
    with pytest.raises(ValueError, match="photo-z scale"):
        prior("postreconstruction", sigma0=0.02)


def test_every_measurement_run_takes_the_definition_and_changes_only_its_prior():
    from pzbao.run import stage_units

    config = load_analysis_config()
    sources = {
        "primary": (config, "full_grid"),
        "scan--R35": (sensitivity_run_configs(config)["scan--R35"], "smoothing_scan"),
        "self--R25": (transfer_run_configs(config)["self--R25"], "transfer"),
        "sigma_0.003--R15": (transfer_run_configs(config)["sigma_0.003--R15"], "transfer"),
    }
    for name, (source, workload) in sources.items():
        derived = derive_inference_config(source, DEFINITION)
        assert derived["fitting"]["damping"]["applied_prior"] == "unified_bin_prior_v1"
        validate_analysis_config(derived)
        restored = deepcopy(derived)
        del restored["fitting"]["damping"]["applied_prior"]
        assert restored == source, name
        # Every route of the source, before and after reconstruction.
        fits = stage_units("fit", derived, workload, inference_definition=DEFINITION)
        assert fits == stage_units("fit", source, workload)
        with pytest.raises(ValueError, match="reuses measured vectors"):
            stage_units("covariance", derived, workload, inference_definition=DEFINITION)
        altered = deepcopy(derived)
        altered["fitting"]["broadband_model"] = "pcs2"
        with pytest.raises(ValueError):
            validate_inference_config(source, altered, DEFINITION)
    # Fit checks and other applied priors are not primary measurement definitions.
    flat = derive_inference_config(config, "core_bin_common_flat_v1")
    with pytest.raises(ValueError, match="primary measurement definition"):
        derive_inference_config(flat, DEFINITION)


def test_the_two_checks_change_one_more_choice_on_the_primary_comparison_routes():
    from pzbao.run import comparison_routes, routes_for, stage_units

    config = load_analysis_config()
    comparison = comparison_routes(routes_for(config, "full_grid"), config)
    assert {route["condition"] for route in comparison} == {"sigma_0.003"}
    assert {route["reconstruction_state"] for route in comparison} == {"postreconstruction"}
    assert len(comparison) == 5
    pcs2 = derive_inference_config(config, "unified_prior_pcs2_v1")
    flat = derive_inference_config(config, "unified_prior_flat_v1")
    for name, derived in (("pcs2", pcs2), ("flat", flat)):
        fitting = derived["fitting"]
        assert fitting["core_fit_check"] == f"unified_prior_{name}_v1"
        units = stage_units(
            "fit", derived, "full_grid", inference_definition=f"unified_prior_{name}_v1"
        )
        assert {route_key(unit) for unit in units} == {route_key(r) for r in comparison}
        assert len(units) == 5 * 26
    # PCS2 keeps the configured priors under the alternative broadband.
    assert pcs2["fitting"]["damping"]["applied_prior"] == "unified_bin_prior_v1"
    assert pcs2["fitting"]["broadband_model"] == "pcs2"
    assert pcs2["fitting"]["paper_comparisons"] == config["fitting"]["paper_comparisons"]
    # The flat check keeps power3 and widens only the uniform support's upper edge, 10 to 20.
    assert flat["fitting"]["damping"]["applied_prior"] == "paper_finite_flat"
    assert flat["fitting"]["broadband_model"] == config["fitting"]["broadband_model"]
    widened = deepcopy(config["fitting"]["paper_comparisons"])
    assert widened["flat_damping"]["nonlinear_upper_mpc_h"] == 10.0
    widened["flat_damping"]["nonlinear_upper_mpc_h"] = 20.0
    assert flat["fitting"]["paper_comparisons"] == widened
    restored = deepcopy(flat)
    del restored["fitting"]["core_fit_check"]
    del restored["fitting"]["damping"]["applied_prior"]
    restored["fitting"]["paper_comparisons"] = config["fitting"]["paper_comparisons"]
    assert restored == config
    for name, derived in (("pcs2", pcs2), ("flat", flat)):
        altered = deepcopy(derived)
        altered["fitting"]["paper_comparisons"]["flat_damping"]["nonlinear_upper_mpc_h"] = 30.0
        with pytest.raises(ValueError):
            validate_inference_config(config, altered, f"unified_prior_{name}_v1")
    # Only the per-bin primary carries the checks.
    for source in (
        sensitivity_run_configs(config)["scan--R25"],
        transfer_run_configs(config)["self--R25"],
    ):
        for name in ("pcs2", "flat"):
            with pytest.raises(ValueError, match="per-bin fit checks require"):
                derive_inference_config(source, f"unified_prior_{name}_v1")


def route_key(route):
    return tuple(route[key] for key in ("measurement_bin", "condition", "reconstruction_state"))


def test_the_route_likelihood_applies_the_table_and_keeps_transfer_provenance(monkeypatch):
    import pzbao.covariance as covariance
    import pzbao.likelihood as likelihood
    import pzbao.selection as selection
    import pzbao.theory as theory

    config = load_analysis_config()
    arrays = {
        "vectors": np.ones((25, 4)),
        "separations": np.ones((25, 4)),
        "mean_covariance": np.eye(4),
        "fit_edges": np.arange(5.0),
        "multipoles": np.array([0, 2]),
    }
    preparation = {
        "effective_redshifts": {
            condition: {"LRG2": {"z_eff": 0.7}, "parent": {"z_eff": 0.7}}
            for condition in ("sigma_0.003", "sigma_0.03")
        },
        "conditions": {
            condition: {
                cap: {"counts": {"LRG2": 1000}, "volumes_mpc_h3": {"LRG2": 1e7}}
                for cap in ("NGC", "SGC")
            }
            for condition in ("sigma_0.003", "sigma_0.03")
        },
    }
    context = {"fixture": "source-target context"}
    captured = []

    def build(data, matrix, **kwargs):
        captured.append(kwargs["damping_prior"])
        return None, SimpleNamespace(power_broadband_repair={}), None, {}

    identity = {"run_name": "fixture", "code_commit": "d" * 40, "analysis_config_sha256": "e"}
    monkeypatch.setattr(covariance, "likelihood_covariance_run", lambda run, route: run)
    monkeypatch.setattr(
        covariance,
        "read_route_covariance",
        lambda run, route: {
            "arrays": arrays,
            "window": {},
            "metadata": {"mean_covariance_interpretation": "fixture"},
        },
    )
    monkeypatch.setattr(selection, "read_preparation", lambda run, tracer: preparation)
    monkeypatch.setattr(theory, "transfer_damping_context", lambda *a, **k: context)
    fourier = SimpleNamespace(pk_interpolator=lambda of: (lambda k, z: 1.0))
    monkeypatch.setattr(
        likelihood,
        "build_mock_cosmology",
        lambda config: SimpleNamespace(
            efunc=lambda z: 1.4,
            growth_rate=lambda z: 0.8,
            get_fourier=lambda: fourier,
            rs_drag=100.0,
        ),
    )
    # The analytic transverse centre is recorded for core post-reconstruction routes only.
    analytic = {"fixture": "analytic"}
    monkeypatch.setattr(likelihood, "analytic_postreconstruction_sigma_perp", lambda **k: analytic)
    monkeypatch.setattr(likelihood, "build_likelihood", build)
    runs = {
        "self--R25": transfer_run_configs(config)["self--R25"],
        "primary": config,
    }
    for name, source in runs.items():
        derived = derive_inference_config(source, DEFINITION)
        states = (
            ("prereconstruction",)
            if name == "primary"
            else (
                "prereconstruction",
                "postreconstruction",
            )
        )
        for state in states:
            unit = {
                "tracer": "LRG",
                "measurement_bin": "LRG2",
                "condition": "sigma_0.03",
                "reconstruction_state": state,
                "realization_id": None,
            }
            run = {"config": derived, "metadata": identity}
            metadata = likelihood._route_likelihood(run, unit, method="profiling")[3]
            assert captured[-1] is metadata["damping_prior"]
            assert metadata["damping_prior"]["prior_table"] == "unified_bin_prior_v1"
            assert metadata["damping_prior"]["centers_mpc_h"]["sigma_perp"] == 5.0
            assert metadata["damping_prior"]["widths_mpc_h"]["sigma_parallel"] == 6.0
            if name == "self--R25":
                inputs = metadata["reconstruction_inputs"]
                assert inputs["source_target_context"] is context
                assert inputs["target_class"] == "sigma_0.03"
    # The two checks on a primary comparison route, after reconstruction: PCS2 applies the
    # table, the flat check replaces it by uniform support to 20 Mpc/h (sigma_parallel widened
    # by the pairwise photo-z scale).
    unit = {
        "tracer": "LRG",
        "measurement_bin": "LRG2",
        "condition": "sigma_0.003",
        "reconstruction_state": "postreconstruction",
        "realization_id": None,
    }
    for definition in ("unified_prior_v1", "unified_prior_pcs2_v1", "unified_prior_flat_v1"):
        run = {"config": derive_inference_config(config, definition), "metadata": identity}
        metadata = likelihood._route_likelihood(run, unit, method="profiling")[3]
        damping = metadata["damping_prior"]
        assert metadata["analytic_transverse_damping"] is analytic
        assert damping["prior_table"] == "unified_bin_prior_v1"
        assert damping["centers_mpc_h"] == {
            "sigma_perp": 5.0,
            "sigma_parallel": 8.0,
            "sigma_streaming": 2.0,
        }
        if definition != "unified_prior_flat_v1":
            assert damping["unpriored_parameters"] == []
            continue
        photoz = damping["sigma_parallel_photoz_mpc_h"]
        assert damping["unpriored_parameters"] == [
            "sigma_streaming",
            "sigma_parallel",
            "sigma_perp",
        ]
        assert damping["bounds_mpc_h"] == {
            "sigma_streaming": [0.0, 20.0],
            "sigma_parallel": [0.0, float(np.hypot(20.0, photoz))],
            "sigma_perp": [0.0, 20.0],
        }
