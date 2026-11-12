"""Dependency inventory for the opt-in exclusive reconstruction-transfer design.

This is run preparation, not a replacement execution or publication pipeline. Field labels
below are dependency selectors; only actual input/settings digests may key field caches.
"""

from copy import deepcopy

from . import analysis_config_sha256
from .schema_validation import PER_BIN_SCHEMA_VERSIONS, validate_analysis_config


def transfer_run_configs(config, *, pilot=False):
    """Complete native-stage configurations; no duplicate high-source R25 measurement.

    Damping centres are deliberately unresolved. Catalogues through covariance can run;
    reference-prior post-reconstruction fits require independently reviewed centres.
    """
    design = config["measurement"]["reconstruction"]["exploratory_transfer"]
    common, high = design["common_smoothing_mpc_h"], design["source_class"]
    self_radii = [common] + ([] if pilot else design.get("self_smoothing_checks_mpc_h", []))
    variants = [("self", common), (high, common)]
    if not pilot:
        variants += [(high, radius) for radius in design["source_smoothing_checks_mpc_h"]]
        variants += [("self", radius) for radius in self_radii[1:]]
    result = {}
    for source, radius in variants:
        snapshot = deepcopy(config)
        recon = snapshot["measurement"]["reconstruction"]
        # Revisions 2.13 and 2.15 change core reconstruction and the matched density tiers
        # only. Transfer snapshots derived from the new default keep the schema-15 design (one
        # field per tracer, no matched-density block) but carry the current reporting forecast
        # sky fraction, so they are not byte-identical to the frozen schema-15 definition.
        if snapshot["schema_version"] in PER_BIN_SCHEMA_VERSIONS:
            snapshot["schema_version"] = 15
            recon["scope"] = "one_field_per_tracer"
            snapshot["selection"].pop("veff_matched_density", None)
        recon["smoothing_values_mpc_h"] = [radius]
        recon["transfer_run"] = {"source_class": source, "pilot": pilot, "damping_centres": None}
        targets = list(design["class_labels"])
        # The same-radius self run already owns the source class on its own field.
        if source == high and radius in self_radii:
            targets.remove(high)
        snapshot["analysis"]["workloads"]["transfer"] = {
            "measurement_bins": list(
                design["pilot_measurement_bins"] if pilot else design["measurement_bins"]
            ),
            "sigma_z_tiers": targets,
            # Pre-reconstruction routes are radius-independent and owned by common-radius self.
            "reconstruction_states": (
                ["prereconstruction", "postreconstruction"]
                if source == "self" and radius == common
                else ["postreconstruction"]
            ),
            "mock_ids": list(design["pilot_mock_ids"] if pilot else config["analysis"]["mock_ids"]),
        }
        snapshot["fitting"]["broadband_model"] = design["primary_broadband"]
        validate_analysis_config(snapshot)
        result[f"{source}--R{radius:g}"] = snapshot
    return result


def transfer_matrix(config, *, pilot=False):
    """Enumerate unique measurement/field definitions and separate fit-only checks."""
    validate_analysis_config(config)
    design = config["measurement"]["reconstruction"]["exploratory_transfer"]
    bins = design["pilot_measurement_bins"] if pilot else design["measurement_bins"]
    mocks = design["pilot_mock_ids"] if pilot else config["analysis"]["mock_ids"]
    tracers = {
        item["name"]: tracer
        for tracer, spec in config["tracers"].items()
        for item in spec["measurement_bins"]
    }
    source = design["source_class"]
    common = design["common_smoothing_mpc_h"]
    fields, measurements, fits = {}, [], []

    def add(name, target, field_source=None, radius=None):
        state = "prereconstruction" if field_source is None else "postreconstruction"
        field_id = None
        if field_source is not None:
            field_id = f"{name}--source-{field_source}--R{radius:g}"
            fields[field_id] = {
                "field_id": field_id,
                "tracer": tracers[name],
                "measurement_bin": name,
                "source_class": field_source,
                "smoothing_mpc_h": radius,
            }
        identity = f"{name}--target-{target}--" + (
            "pre" if field_id is None else f"source-{field_source}--R{radius:g}"
        )
        row = {
            "measurement_id": identity,
            "tracer": tracers[name],
            "measurement_bin": name,
            "target_class": target,
            "source_class": field_source,
            "reconstruction_state": state,
            "smoothing_mpc_h": radius,
            "field_id": field_id,
        }
        measurements.append(row)
        fits.append(
            {
                "measurement_id": identity,
                "broadband": design["primary_broadband"],
                "damping": "reference",
                "prerequisite": "reviewed_source_target_centres" if field_id else None,
            }
        )
        if not pilot and radius == common and target != source:
            fits.extend(
                [
                    {**fits[-1], "broadband": design["broadband_check"]},
                    {**fits[-1], "damping": design["damping_check"]},
                ]
            )

    for name in bins:
        for target in design["class_labels"]:
            add(name, target)
            add(name, target, target, common)
            if target != source:
                add(name, target, source, common)
        if not pilot:
            self_checks = design.get("self_smoothing_checks_mpc_h", [])
            for radius in self_checks:
                for target in design["class_labels"]:
                    add(name, target, target, radius)
            for radius in design["source_smoothing_checks_mpc_h"]:
                for target in design["class_labels"]:
                    if target != source or radius not in self_checks:
                        add(name, target, source, radius)
    groups = {}
    for field in fields.values():
        key = (field["tracer"], field["source_class"], field["smoothing_mpc_h"])
        groups.setdefault(key, []).append(field["measurement_bin"])
    source_groups = [
        {
            "unit_id": f"{tracer}--source-{source}--R{radius:g}--mock-{mock}",
            "tracer": tracer,
            "source_class": source,
            "smoothing_mpc_h": radius,
            "realization_id": mock,
            "measurement_bins": names,
        }
        for (tracer, source, radius), names in groups.items()
        for mock in mocks
    ]
    target_units = {
        (row["tracer"], row["target_class"], row["source_class"], row["smoothing_mpc_h"], mock)
        for row in measurements
        if row["field_id"] is not None
        for mock in mocks
    }
    return {
        "kind": "transfer_preparation_inventory",
        "specification_revision": design["specification_revision"],
        "analysis_config_sha256": analysis_config_sha256(config),
        "population": design["population"],
        "mock_ids": list(mocks),
        "allocation_mock_ids": list(config["analysis"]["mock_ids"]),
        "caps": ["NGC", "SGC"],
        "statistic": "xi",
        "fields": list(fields.values()),
        "source_groups": source_groups,
        "execution_counts": {
            "source_groups": len(source_groups),
            "bin_footprint_solves": len(fields) * len(mocks) * 2,
            "target_reconstruction_units": len(target_units),
            "measurement_definitions": len(measurements),
            "fitting_definitions": len(fits),
            "separate_fitting_checks": len(fits) - len(measurements),
        },
        "measurements": measurements,
        "fits": fits,
        "covariance": "new_route_specific_RascalC_and_shot_noise_calibration",
        "pilot_limitation": "engineering_only_no_bias_or_covariance_calibration" if pilot else None,
    }
