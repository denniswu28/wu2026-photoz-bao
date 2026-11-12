"""Run definitions for the core smoothing scan and the RecIso robustness comparison.

Each radius, and RecIso at the reference radius, is a separate named run. Selection does not
depend on either choice, so all of them reuse one selected cohort through the select stage.
"""

from copy import deepcopy

from .schema_validation import validate_analysis_config


def sensitivity_run_configs(config):
    """Complete core snapshots; the primary configuration itself is left unchanged."""
    validate_analysis_config(config)
    recon = config["measurement"]["reconstruction"]
    if "transfer_run" in recon or "sensitivity_run" in recon:
        raise ValueError("sensitivity runs derive from the primary core configuration")
    design = recon["exploratory_smoothing_scan"]
    reference = design["reference_smoothing_mpc_h"]
    scope = config["fitting"]["paper_comparisons"]
    result = {}

    def make(kind, radius, workload, convention):
        snapshot = deepcopy(config)
        snapshot_recon = snapshot["measurement"]["reconstruction"]
        snapshot_recon["smoothing_values_mpc_h"] = [radius]
        snapshot_recon["allowed_conventions"] = [convention]
        snapshot_recon["sensitivity_run"] = {"kind": kind}
        snapshot["fitting"]["damping"]["reference_smoothing_mpc_h"] = reference
        snapshot["analysis"]["workloads"][kind] = {
            **workload,
            "mock_ids": list(config["analysis"]["mock_ids"]),
        }
        validate_analysis_config(snapshot)
        return snapshot

    for radius in design["radii_mpc_h"]:
        result[f"scan--R{radius:g}"] = make(
            "smoothing_scan",
            radius,
            {
                "measurement_bins": list(design["measurement_bins"]),
                "sigma_z_tiers": list(design["sigma_z_tiers"]),
                "reconstruction_states": list(design["reconstruction_states"]),
            },
            "recsym",
        )
    result[f"reciso--R{reference:g}"] = make(
        "reciso",
        reference,
        {
            "measurement_bins": list(scope["measurement_bins"]),
            "sigma_z_tiers": [scope["condition"]],
            "reconstruction_states": [scope["reconstruction_state"]],
        },
        "reciso",
    )
    return result
