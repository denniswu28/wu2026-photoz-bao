"""Read the single scientific configuration and identify complete run snapshots."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import yaml

from .schema_validation import (
    PCS2_NUISANCE_AXES,
    PER_BIN_SCHEMA_VERSIONS,
    POWER3_NUISANCE_AXES,
    _UniqueKeyLoader,
    same_identity,
    validate_analysis_config,
)

__all__ = [
    "DEFAULT_ANALYSIS_CONFIG_PATH",
    "load_analysis_config",
    "analysis_config_sha256",
    "same_identity",
]


DEFAULT_ANALYSIS_CONFIG_PATH = Path(__file__).with_name("analysis.yaml")


def per_bin_reconstruction(config):
    """Resolve field topology without conflating core bins with exclusive transfer."""
    recon = config["measurement"]["reconstruction"]
    return "transfer_run" in recon or recon["scope"] == "separate_measurement_bins"


BGS_ANISOTROPIC = "bgs_anisotropic_v1"
TRANSFER_INFERENCE = ("transfer_reference", "transfer_pcs2", "transfer_flat")
PRIMARY_POWER3 = "primary_power3_v1"
POWER3_FLAT = "power3_flat_finite_v1"
CORE_POWER3_INFERENCE = (PRIMARY_POWER3, POWER3_FLAT)
CORE_BIN_INFERENCE = ("core_bin_pcs2_v1", "core_bin_flat_v1", "core_bin_common_flat_v1")
# Revision 2.18: every smoothing-scan radius refitted with its own analytic transverse centre.
CORE_SCAN_INFERENCE = ("core_scan_radius_centre_v1",)
CORE_INFERENCE = (*CORE_POWER3_INFERENCE, *CORE_BIN_INFERENCE, *CORE_SCAN_INFERENCE)
# Revision 2.20: every run refitted with the configured per-bin damping priors, and two fit checks
# of them on the per-bin primary's comparison routes (2026-10-20).
UNIFIED_PRIOR = "unified_prior_v1"
UNIFIED_PCS2, UNIFIED_FLAT = "unified_prior_pcs2_v1", "unified_prior_flat_v1"
UNIFIED_PRIOR_INFERENCE = (UNIFIED_PRIOR, UNIFIED_PCS2, UNIFIED_FLAT)
#: Upper edge of the flat check's uniform support, twice the 2.11 support
#: (2026-10-20); the parallel edge is still widened by the pairwise photo-z scale.
UNIFIED_FLAT_UPPER_MPC_H = 20.0
BGS_ANISOTROPIC_SURFACE = {
    "observed_multipoles": [0, 2],
    "model_multipoles": [0, 2, 4],
    "ap_mode": "qisoqap",
    "fixed_parameters": {"dbeta": 1.0},
}


def derive_core_power3(source, definition, flat_scope=None):
    """Refit a schema-13 PCS2 core measurement with the 2.11 power3 fitting definition.

    Measurement, covariance, priors and photo-z response stay those of the frozen source;
    only the broadband, its nuisance directions and the anisotropic BGS surface change. The
    flat check additionally adopts the configured finite flat damping support.
    """
    if definition not in CORE_POWER3_INFERENCE:
        raise ValueError("unknown core power3 inference definition")
    if source["schema_version"] != 13 or source["fitting"]["broadband_model"] != "pcs2":
        raise ValueError("core power3 refits derive from a schema-13 PCS2 measurement source")
    if "transfer_run" in source["measurement"]["reconstruction"]:
        raise ValueError("transfer runs use their own inference definitions")
    result = deepcopy(source)
    result["schema_version"] = 15
    fitting = result["fitting"]
    fitting["broadband_model"] = "power3"
    fitting["fit_surfaces"]["BGS_BRIGHT-21.5"] = deepcopy(BGS_ANISOTROPIC_SURFACE)
    nuisance = fitting["linear_nuisance"]
    nuisance["varied_basenames_by_tracer"] = {
        tracer: list(POWER3_NUISANCE_AXES) for tracer in nuisance["varied_basenames_by_tracer"]
    }
    nuisance["allowed_basenames_by_model"] = {"power3": list(POWER3_NUISANCE_AXES)}
    if definition == POWER3_FLAT:
        if flat_scope is None:
            flat_scope = load_analysis_config()["fitting"]["paper_comparisons"]
        fitting["paper_comparisons"] = deepcopy(flat_scope)
        fitting["damping"]["applied_prior"] = "paper_finite_flat"
    return result


def validate_inference_config(source, derived, definition=None):
    """Allow only the named observable or transfer fit check across immutable snapshots."""
    validate_analysis_config(source)
    validate_analysis_config(derived)
    if definition is None:
        if not same_identity(source, derived):
            raise ValueError("inference requires its source configuration")
        return
    if definition in TRANSFER_INFERENCE:
        expected = derive_transfer_inference(source, definition)
        # A new inference snapshot may adopt only the reviewed source-target artifact
        # in addition to its named fitting definition. Native context validation is
        # performed by the run producer before publication and by the resolver at fit.
        expected["measurement"]["reconstruction"]["transfer_run"]["damping_centres"] = deepcopy(
            derived["measurement"]["reconstruction"]["transfer_run"]["damping_centres"]
        )
        comparable = deepcopy(derived)
        comparable["fitting"]["broadband_model"] = source["fitting"]["broadband_model"]
        for key in ("varied_basenames_by_tracer", "allowed_basenames_by_model"):
            comparable["fitting"]["linear_nuisance"][key] = deepcopy(
                source["fitting"]["linear_nuisance"][key]
            )
        comparable["measurement"]["reconstruction"]["transfer_run"] = deepcopy(
            source["measurement"]["reconstruction"]["transfer_run"]
        )
        if not same_identity(source, comparable) or not same_identity(derived, expected):
            raise ValueError("transfer inference changed more than its single configured fit check")
        return
    if definition in CORE_BIN_INFERENCE:
        if not same_identity(derived, derive_core_bin_inference(source, definition)):
            raise ValueError("per-bin inference changes more than its configured fit definition")
        return
    if definition in UNIFIED_PRIOR_INFERENCE:
        if not same_identity(derived, derive_unified_prior_inference(source, definition)):
            raise ValueError("the unified prior refit changes more than its damping prior")
        return
    if definition in CORE_SCAN_INFERENCE:
        if not same_identity(derived, derive_core_scan_inference(source, definition)):
            raise ValueError("scan centre refit changes more than its configured fit definition")
        return
    if definition in CORE_POWER3_INFERENCE:
        # The flat check freezes the configured comparison scope it adopted at creation.
        scope = derived["fitting"].get("paper_comparisons")
        expected = derive_core_power3(source, definition, flat_scope=scope)
        if not same_identity(derived, expected):
            raise ValueError("core power3 inference changes more than its configured fit definition")
        return
    if definition != BGS_ANISOTROPIC:
        raise ValueError("unknown inference definition")
    if source["schema_version"] != 13 or derived["schema_version"] != 14:
        raise ValueError(
            "BGS anisotropic inference requires a schema-13 source and schema-14 target"
        )
    comparable = deepcopy(derived)
    comparable["schema_version"] = source["schema_version"]
    tracer = "BGS_BRIGHT-21.5"
    comparable["fitting"]["fit_surfaces"][tracer] = source["fitting"]["fit_surfaces"][tracer]
    comparable["fitting"]["linear_nuisance"]["varied_basenames_by_tracer"][tracer] = source[
        "fitting"
    ]["linear_nuisance"]["varied_basenames_by_tracer"][tracer]
    if not same_identity(source, comparable):
        raise ValueError(
            "inference changes more than the configured BGS multipoles and AP definition"
        )


def derive_inference_config(source, definition):
    """Derive a versioned inference change from its frozen source, never today's defaults."""
    result = deepcopy(source)
    if definition is None:
        return result
    if definition in TRANSFER_INFERENCE:
        result = derive_transfer_inference(source, definition)
        validate_inference_config(source, result, definition)
        return result
    if definition in CORE_BIN_INFERENCE:
        result = derive_core_bin_inference(source, definition)
        validate_inference_config(source, result, definition)
        return result
    if definition in UNIFIED_PRIOR_INFERENCE:
        result = derive_unified_prior_inference(source, definition)
        validate_inference_config(source, result, definition)
        return result
    if definition in CORE_SCAN_INFERENCE:
        result = derive_core_scan_inference(source, definition)
        validate_inference_config(source, result, definition)
        return result
    if definition in CORE_POWER3_INFERENCE:
        result = derive_core_power3(source, definition)
        validate_inference_config(source, result, definition)
        return result
    if definition != BGS_ANISOTROPIC:
        raise ValueError("unknown inference definition")
    result["schema_version"] = 14
    tracer = "BGS_BRIGHT-21.5"
    result["fitting"]["fit_surfaces"][tracer] = deepcopy(BGS_ANISOTROPIC_SURFACE)
    result["fitting"]["linear_nuisance"]["varied_basenames_by_tracer"][tracer] = list(
        result["fitting"]["linear_nuisance"]["varied_basenames_by_tracer"]["LRG"]
    )
    validate_inference_config(source, result, definition)
    return result


def derive_unified_prior_inference(source, definition):
    """Refit a measurement run with the configured per-bin damping priors (revision 2.20).

    Only the damping prior changes: measurement, covariance, calibration, broadband and every
    other setting stay those of the frozen source, which must be a primary measurement
    definition (no fit check, no other applied prior). Each check changes one more choice on
    the per-bin primary's comparison routes: PCS2 keeps the configured priors under the
    alternative broadband; the flat check replaces them by the enlarged uniform support.
    """
    if definition not in UNIFIED_PRIOR_INFERENCE:
        raise ValueError("unknown unified prior inference definition")
    fitting = source["fitting"]
    transfer = source["measurement"]["reconstruction"].get("transfer_run") or {}
    if (
        fitting["damping"].get("applied_prior") is not None
        or "core_fit_check" in fitting
        or "prior" in transfer
    ):
        raise ValueError("the unified prior refits a primary measurement definition")
    if definition == UNIFIED_PRIOR:
        result = deepcopy(source)
        result["fitting"]["damping"]["applied_prior"] = "unified_bin_prior_v1"
        return result
    check = "core_bin_pcs2_v1" if definition == UNIFIED_PCS2 else "core_bin_flat_v1"
    result = derive_core_bin_inference(source, check)
    result["fitting"]["core_fit_check"] = definition
    if definition == UNIFIED_PCS2:
        result["fitting"]["damping"]["applied_prior"] = "unified_bin_prior_v1"
    else:
        flat = result["fitting"]["paper_comparisons"]["flat_damping"]
        flat["nonlinear_upper_mpc_h"] = UNIFIED_FLAT_UPPER_MPC_H
    return result


def derive_core_bin_inference(source, definition):
    """Fit-only checks bound to a new primary measurement and its own covariance."""
    recon = source["measurement"]["reconstruction"]
    if (
        source["schema_version"] not in PER_BIN_SCHEMA_VERSIONS
        or "transfer_run" in recon
        or "sensitivity_run" in recon
        or "core_fit_check" in source["fitting"]
        or source["fitting"]["broadband_model"] != "power3"
        or recon["smoothing_values_mpc_h"]
        != [recon["exploratory_smoothing_scan"]["reference_smoothing_mpc_h"]]
    ):
        raise ValueError(
            "per-bin fit checks require the per-bin (schema 16/17) primary measurement"
        )
    result = deepcopy(source)
    fitting = result["fitting"]
    fitting["core_fit_check"] = definition
    if definition == "core_bin_pcs2_v1":
        fitting["broadband_model"] = fitting["paper_comparisons"]["broadband_alternative"]
        nuisance = fitting["linear_nuisance"]
        nuisance["varied_basenames_by_tracer"] = {
            tracer: list(PCS2_NUISANCE_AXES) for tracer in nuisance["varied_basenames_by_tracer"]
        }
        nuisance["allowed_basenames_by_model"] = {"pcs2": list(PCS2_NUISANCE_AXES)}
    elif definition == "core_bin_flat_v1":
        fitting["damping"]["applied_prior"] = "paper_finite_flat"
    elif definition == "core_bin_common_flat_v1":
        fitting["damping"]["applied_prior"] = "common_flat_sensitivity"
    else:
        raise ValueError("unknown per-bin fit check")
    return result


def derive_core_scan_inference(source, definition):
    """Refit one published smoothing-scan radius with the transverse centre at that radius.

    The scan runs pin the analytic post-reconstruction centre to the reference radius in their
    frozen definitions; revision 2.18 evaluates it at each run's own radius. Measurement,
    covariance, calibration, widths, bounds and every other centre stay the source's.
    """
    if definition not in CORE_SCAN_INFERENCE:
        raise ValueError("unknown scan centre inference definition")
    recon = source["measurement"]["reconstruction"]
    damping = source["fitting"]["damping"]
    if (
        source["schema_version"] not in PER_BIN_SCHEMA_VERSIONS
        or "transfer_run" in recon
        or recon.get("sensitivity_run") != {"kind": "smoothing_scan"}
        or "core_fit_check" in source["fitting"]
        or source["fitting"]["broadband_model"] != "power3"
        or damping.get("reference_smoothing_mpc_h")
        != recon["exploratory_smoothing_scan"]["reference_smoothing_mpc_h"]
    ):
        raise ValueError(
            "scan centre refits require a per-bin smoothing-scan measurement pinned to the "
            "reference radius"
        )
    result = deepcopy(source)
    result["fitting"]["core_fit_check"] = definition
    result["fitting"]["damping"]["reference_smoothing_mpc_h"] = recon["smoothing_values_mpc_h"][0]
    return result


def derive_transfer_inference(source, definition):
    result = deepcopy(source)
    recon = result["measurement"]["reconstruction"]
    if "transfer_run" not in recon:
        raise ValueError("transfer inference requires an exclusive source run")
    if definition == "transfer_reference":
        return result
    if definition == "transfer_flat":
        # The uniform prior is the transfer headline at every radius (revision 2.19).
        recon["transfer_run"]["prior"] = recon["exploratory_transfer"]["damping_check"]
        return result
    if recon["smoothing_values_mpc_h"] != [recon["exploratory_transfer"]["common_smoothing_mpc_h"]]:
        raise ValueError("the PCS2 transfer check requires a common-radius exclusive source run")
    if definition == "transfer_pcs2":
        result["fitting"]["broadband_model"] = recon["exploratory_transfer"]["broadband_check"]
        nuisance = result["fitting"]["linear_nuisance"]
        nuisance["varied_basenames_by_tracer"] = {
            tracer: list(PCS2_NUISANCE_AXES) for tracer in nuisance["varied_basenames_by_tracer"]
        }
        nuisance["allowed_basenames_by_model"] = {"pcs2": list(PCS2_NUISANCE_AXES)}
    else:
        raise ValueError("unknown transfer inference definition")
    return result


def load_analysis_config(path: Optional[Union[str, Path]] = None) -> dict:
    """Read a complete validated scientific definition, including explicit experiment snapshots."""
    source = Path(path or DEFAULT_ANALYSIS_CONFIG_PATH).resolve()
    with source.open(encoding="utf-8") as stream:
        payload = yaml.load(stream, Loader=_UniqueKeyLoader)
    if isinstance(payload, dict) and "shared" in payload:
        from .stages import expand_document

        with DEFAULT_ANALYSIS_CONFIG_PATH.open(encoding="utf-8") as stream:
            baseline = yaml.load(stream, Loader=_UniqueKeyLoader)
        payload = expand_document(payload, baseline)
    validate_analysis_config(payload)
    return payload


def analysis_config_sha256(payload: Mapping[str, Any]) -> str:
    """Identify the complete resolved settings and materialized scientific references."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_boundary_variant(reference, variant):
    """Allow only the declared support change and affected workload relative to the reference."""
    validate_analysis_config(variant)
    before, after = deepcopy(reference), deepcopy(variant)
    boundary = after["selection"]["boundary_support"]
    experiment = boundary.pop("experiment")
    design = boundary.pop("experiment_design")
    if "experiment" in before["selection"]["boundary_support"]:
        raise ValueError("a boundary reference must be an unmodified analysis")
    approved_design = before["selection"]["boundary_support"].pop("experiment_design", None)
    if approved_design is None:
        approved_design = load_analysis_config()["selection"]["boundary_support"][
            "experiment_design"
        ]
    if design != approved_design:
        raise ValueError("boundary variant differs from the configured design")
    target = next(
        item
        for item in design["targets"]
        if item["tracer"] == experiment["tracer"] and item["side"] == experiment["side"]
    )
    workload = after["analysis"]["workloads"].pop("boundary")
    expected = {
        "measurement_bins": [target["measurement_bin"]],
        "sigma_z_tiers": [experiment["condition"]],
        "reconstruction_states": ["prereconstruction", "postreconstruction"],
        "mock_ids": before["analysis"]["mock_ids"],
    }
    if workload != expected:
        raise ValueError("boundary workload must retain the full paired ensemble")
    if experiment["mode"] == "extend":
        tracer = experiment["tracer"]
        old = before["tracers"][tracer]["reservoir"]
        new = after["tracers"][tracer]["reservoir"]
        edge = next(item for item in new["edges"] if item["side"] == experiment["side"])
        old_edge = next(item for item in old["edges"] if item["side"] == experiment["side"])
        sigma = next(
            item["sigma_z_over_one_plus_z"]
            for item in before["selection"]["tiers"]
            if item["label"] == experiment["condition"]
        )
        depth = old["guard_n_sigma"] * sigma * (1 + edge["redshift"])
        if (
            experiment["side"] != "above"
            or experiment["depth_redshift"] != depth
            or edge.get("slice_half_width_redshift") != depth
            or experiment["reference_depth_redshift"]
            != old_edge.get("slice_half_width_redshift", old["slice_half_width_redshift"])
        ):
            raise ValueError("extension must change only the upper edge to six-scatter support")
        if new["slice_filename"] == old["slice_filename"]:
            raise ValueError("extension must publish a separately named reservoir")
        new["slice_filename"] = old["slice_filename"]
        edge.clear()
        edge.update(old_edge)
    if before != after:
        raise ValueError("boundary experiment changed science beyond its declared support")
