"""Scientific intervention and provenance contracts for the bounded comparisons."""

from copy import deepcopy

import numpy as np
import pytest

from pzbao.config import load_analysis_config, validate_analysis_config
from pzbao.diagnostics import fitting as robustness
from tests.legacy_config import load_analysis_config as load_legacy_config


def settings():
    return load_analysis_config()["fitting"]["paper_comparisons"]


def test_flat_support_uses_pairwise_scale_without_changing_the_response():
    from pzbao.theory import resolve_damping_prior
    from pzbao.theory import sigma_parallel_photoz_mpc_h

    config = load_analysis_config()
    baseline = resolve_damping_prior(
        "LRG",
        "prereconstruction",
        sigma_z_over_one_plus_z=0.003,
        z_eff=0.8,
        e_z_eff=1.6,
        config=config,
    )
    original = deepcopy(baseline)
    result = robustness.paper_flat_prior(baseline, settings())
    pairwise = sigma_parallel_photoz_mpc_h(
        0.8, 0.003, 1.6, config["physics"]["speed_of_light_km_s"]
    )
    assert result["bounds_mpc_h"] == {
        "sigma_streaming": [0, 10],
        "sigma_perp": [0, 10],
        "sigma_parallel": [0, pytest.approx(np.sqrt(100 + pairwise**2))],
    }
    assert set(result["unpriored_parameters"]) == set(robustness.DAMPING.values())
    for field in (
        "centers_mpc_h",
        "widths_mpc_h",
        "sigma_parallel_photoz_mpc_h",
        "photoz_response_policy",
    ):
        assert result[field] == baseline[field]
    assert baseline == original


@pytest.mark.parametrize("change", ["tier", "bins", "support", "unknown"])
def test_comparison_schema_rejects_unsupported_scope(change):
    config = load_analysis_config()
    target = config["fitting"]["paper_comparisons"]
    if change == "tier":
        target["condition"] = "sigma_0.001"
    elif change == "bins":
        target["measurement_bins"].pop()
    elif change == "support":
        target["flat_damping"]["lower_mpc_h"] = 11
    else:
        target["flat_damping"]["invented"] = True
    with pytest.raises(ValueError):
        validate_analysis_config(config)


def test_pcs2_changes_only_registered_broadband_representation():
    # The PCS2 robustness row pairs historical broad-field fits; per-bin runs use the
    # core_bin_pcs2_v1 fit check instead (tests/test_per_bin_reconstruction.py).
    reference = load_legacy_config()
    pcs2 = deepcopy(reference)
    pcs2["schema_version"] = 14
    pcs2["fitting"]["broadband_model"] = "pcs2"
    axes = ["bl0_0", "bl0_2", "al2_0", "al2_1", "bl2_0", "bl2_2"]
    pcs2["fitting"]["linear_nuisance"]["varied_basenames_by_tracer"] = {
        k: axes.copy() for k in ("LRG", "BGS_BRIGHT-21.5")
    }
    pcs2["fitting"]["linear_nuisance"]["allowed_basenames_by_model"] = {"pcs2": axes.copy()}
    candidate = robustness.comparison_config(reference, pcs2, "pcs2")
    validate_analysis_config(candidate)
    assert candidate["fitting"]["damping"] == reference["fitting"]["damping"]
    assert candidate["measurement"] == reference["measurement"]
    candidate["fitting"]["nonlinear_parameters"]["qap"]["prior"]["limits"] = [0.9, 1.1]
    with pytest.raises(ValueError, match="unrelated science"):
        robustness.comparison_config(reference, candidate, "pcs2")


def test_routes_reject_other_tiers_states_and_unimplemented_variants():
    route = dict(
        measurement_bin="BGS2", condition="sigma_0.003", reconstruction_state="postreconstruction"
    )
    robustness.validate_comparison_route(route, settings(), "flat")
    for changed, variant in (
        ({**route, "condition": "sigma_0.001"}, "flat"),
        ({**route, "reconstruction_state": "prereconstruction"}, "pcs2"),
        (route, "ngc"),
    ):
        with pytest.raises(ValueError, match="outside"):
            robustness.validate_comparison_route(changed, settings(), variant)


