"""Optional common-flat priors preserve the Gaussian baseline and photo-z response."""

import copy
import json
import numpy as np
import pytest

from pzbao.config import load_analysis_config, validate_analysis_config
from pzbao.diagnostics import fitting as comparison
from pzbao import likelihood as likelihood_module, theory

DAMPING = ("sigma_perp", "sigma_parallel", "sigma_streaming")
STATES = ("prereconstruction", "postreconstruction")


def damping():
    return {
        "centers_mpc_h": dict(zip(DAMPING, [3.0, 6.0, 2.0])),
        "widths_mpc_h": dict(zip(DAMPING, [1.0, 2.0, 2.0])),
        "bounds_mpc_h": {key: [0.0, 20.0] for key in DAMPING},
    }


def flat_rule():
    return load_analysis_config()["fitting"]["damping"]["common_flat_sensitivity"]


@pytest.mark.parametrize("photoz", [0.0, 5.0, 200.0])
def test_common_flat_serialization_preserves_response_and_uses_identical_support(photoz):
    record = dict(damping(), sigma_parallel_photoz_mpc_h=photoz, scope="final_science_fit")
    before = copy.deepcopy(record)
    control = {"common_flat_damping": flat_rule()}
    actual = comparison.adjust_damping(record, control)
    expected = copy.deepcopy(record)
    expected.update(
        bounds_mpc_h={key: [0.0, 50.0] for key in DAMPING},
        unpriored_parameters=list(DAMPING),
        scope="common_flat_damping_sensitivity_2.11-P",
    )
    assert actual == expected and record == before
    restored = json.loads(json.dumps(actual, allow_nan=False))
    assert restored == comparison.adjust_damping(restored, control)
    restored["bounds_mpc_h"]["sigma_parallel"][1] = np.hypot(50, photoz + 1)
    assert restored != comparison.adjust_damping(restored, control)


@pytest.mark.parametrize(
    "extra",
    [
        {"single_damping_width": "sigma_perp"},
        {"centre_smoothing_mpc_h": 25},
        {"dilation_limits": [0.5, 1.5]},
        {"damping_width_factor": 2},
    ],
)
def test_common_flat_rejects_unrelated_interventions(extra):
    with pytest.raises(ValueError, match="another intervention"):
        comparison.adjust_damping(damping(), dict(extra, common_flat_damping=flat_rule()))


@pytest.mark.parametrize(
    "key,value",
    [
        ("bounds_mpc_h", [0.0, 200.0]),
        ("bounds_mpc_h", [False, 50.0]),
        ("distribution", "norm"),
        ("parameters", ["sigma_perp"]),
        ("photoz_response", "quadrature_upper_bound"),
    ],
)
def test_common_flat_config_rejects_another_scientific_prescription(key, value):
    config = load_analysis_config()
    config["fitting"]["damping"]["common_flat_sensitivity"][key] = value
    with pytest.raises(ValueError):
        validate_analysis_config(config)


@pytest.mark.parametrize("state", STATES)
@pytest.mark.parametrize("sigma", [0.0, 0.001, 0.003, 0.01, 0.03])
def test_registering_flat_arm_does_not_change_baseline_gaussian_or_photoz(state, sigma):
    config = load_analysis_config()
    previous = copy.deepcopy(config)
    previous["fitting"]["damping"].pop("common_flat_sensitivity")
    args = dict(
        tracer="LRG",
        reconstruction_state=state,
        sigma_z_over_one_plus_z=sigma,
        z_eff=0.7,
        e_z_eff=1.5,
        postreconstruction_sigma_perp_artifact={
            "analytic_center_mpc_h": 7.2,
            "attained_density_h3_mpc3": 0.0001,
            "tracer": "LRG",
        },
    )
    actual = theory.resolve_damping_prior(**args, config=config)
    assert actual == theory.resolve_damping_prior(**args, config=previous)
    assert actual["unpriored_parameters"] == []
    assert actual["widths_mpc_h"] == dict(zip(DAMPING, [1.0, 2.0, 2.0]))


@pytest.mark.parametrize("smooth", [False, True])
def test_flat_adapter_checks_real_parameter_priors_and_redundant_smooth_axes(monkeypatch, smooth):
    pytest.importorskip("desilike")
    from desilike.parameter import Parameter, ParameterCollection
    from types import SimpleNamespace

    record = comparison.adjust_damping(damping(), {"common_flat_damping": flat_rule()})
    names = ("sigmaper", "sigmapar", "sigmas")
    parameters = ParameterCollection(
        [
            Parameter(name, value=3.0, fixed=False, prior={"dist": "uniform", "limits": [0, 50]})
            for name in names
        ]
    )
    likelihood = SimpleNamespace(all_params=parameters)
    monkeypatch.setattr(likelihood_module, "_configure_likelihood_parameters", lambda *a, **k: None)
    with comparison.common_flat_parameters():
        likelihood_module._configure_likelihood_parameters(likelihood, record, only_now=smooth)
    assert parameters["sigmas"].fixed is False
    assert parameters["sigmaper"].fixed is smooth
    assert parameters["sigmapar"].fixed is smooth
    assert all(p.prior.dist == "uniform" and list(p.prior.limits) == [0, 50] for p in parameters)
    for parameter in parameters:
        parameter.update(fixed=False)
    parameters["sigmas"].update(prior={"dist": "norm", "loc": 2, "scale": 2, "limits": [0, 50]})
    with comparison.common_flat_parameters(), pytest.raises(ValueError, match="uniform"):
        likelihood_module._configure_likelihood_parameters(likelihood, record, only_now=smooth)
