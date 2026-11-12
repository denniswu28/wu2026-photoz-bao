"""V_eff-matched LRG2/LRG3 density tiers: matching rule, handoff production and validation.

Density-matching revision 2.15. The handoff producer
reads a per-bin primary's published LRG selection, derives the matched exact counts from the
rule, proves paired membership through the native selector for both tiers, fits the matched
radial templates and writes ``prepared-inputs.json``; the density successor binds that handoff
and ``validate_density_run`` recomputes every count from the recorded inputs.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np
import yaml

from .schema_validation import MATCHED_DENSITY_CONDITIONS, MATCHED_DENSITY_NAME

NAME = MATCHED_DENSITY_NAME
CONDITIONS = MATCHED_DENSITY_CONDITIONS
TRACER = "LRG"
CAPS = ("NGC", "SGC")
STATES = ("prereconstruction", "postreconstruction")
GEOMETRY_AND_GROWTH = "recomputed_from_matched_selected_randoms"
RULE = (
    "g(n) = n P0/(1 + n P0); R = f_ref A_sky/(A_NGC + A_SGC); g_M = sqrt(R) g(n_S); "
    "n_M = g_M/((1 - g_M) P0) where g_M < 1, else unattainable and the primary selection is kept; "
    "N_req = floor(n_M V_cap); N_sel = min(N_req, min over realizations of the attainable count)"
)


def radial_sample_identity(condition, cap):
    return f"{TRACER}|tier={condition}|density={NAME}|cap={cap}"


def handoff_scope(mock_ids):
    return {
        "tracer": TRACER,
        "conditions": list(CONDITIONS),
        "mock_ids": list(mock_ids),
        "caps": list(CAPS),
        "geometry_and_growth": GEOMETRY_AND_GROWTH,
    }


def _settings(config):
    settings = config["selection"]["veff_matched_density"]
    if settings["count_authority"] != NAME or list(settings["tiers"]) != list(CONDITIONS):
        raise ValueError("configured matched-density authority differs from this implementation")
    return settings


def _source_group(preparation, condition, cap, names, ids):
    """Baseline counts, shell volumes and the attainable matrix of one source tier and cap."""
    group = preparation["conditions"][condition][cap]
    available = np.asarray(group["attainable_counts_by_mock"])
    if (
        available.shape != (len(ids), len(names))
        or available.dtype.kind not in "iu"
        or np.any(available < 0)
    ):
        raise ValueError(
            "attainable counts must be a complete non-negative integer mock/bin matrix"
        )
    if set(group["counts"]) != set(names) or set(group["volumes_mpc_h3"]) != set(names):
        raise ValueError("invalid or incomplete source counts or shell volumes")
    for name in names:
        baseline, volume = group["counts"][name], group["volumes_mpc_h3"][name]
        if type(baseline) is not int or baseline <= 0:
            raise ValueError("source selected counts must be positive integers")
        if type(volume) not in (int, float) or not math.isfinite(volume) or volume <= 0:
            raise ValueError("source shell volumes must be finite and positive")
    rows = group.get("selection_diagnostics_by_mock")
    if rows is not None:
        if [row["realization_id"] for row in rows] != ids:
            raise ValueError("saved count rows do not identify the configured mock order")
        for mock, row in enumerate(rows):
            for column, name in enumerate(names):
                saved = row["selection"]["bins"][name]
                if saved["attained_count"] != group["counts"][name] or saved[
                    "maximum_attainable_count"
                ] != int(available[mock, column]):
                    raise ValueError("native selection evidence and count matrix disagree")
    return {
        "counts": {name: int(group["counts"][name]) for name in names},
        "volumes_mpc_h3": {name: float(group["volumes_mpc_h3"][name]) for name in names},
        "attainable_counts_by_mock": available.tolist(),
    }


def matched_density_plan(preparation, config):
    """Matched exact counts per tier, bin and cap from the frozen rule and the source evidence.

    g(n) = nP0/(1+nP0); R = f_ref A_sky/(A_NGC + A_SGC); g_M = sqrt(R) g(n_S);
    n_M = g_M/[(1-g_M) P0] where g_M < 1; N_req,cap = floor(n_M V_cap);
    N_sel,cap = min(N_req,cap, min over the 25 realizations of the attainable count).
    A bin with g_M >= 1 is unattainable on this footprint and keeps the primary selection, which
    the configuration records by leaving it out of the experiment's bins; a bounded bin is
    refused unless the configuration names it; no bin ever selects fewer objects than the primary.
    Every input (reference sky, P0, cap areas, targets, volumes, ceilings) is recorded so that the
    plan can be recomputed from the record alone.
    """
    from ..catalog import FULL_SKY_AREA_DEG2, measurement_bins

    settings = _settings(config)
    names = [item["name"] for item in measurement_bins(TRACER, config)]
    ids = list(config["analysis"]["mock_ids"])
    if "experiment" in config["selection"]["boundary_support"]:
        raise ValueError("matched density cannot add a boundary intervention")
    if names != ["LRG1", "LRG2", "LRG3"] or ids != list(range(25)):
        raise ValueError("matched density requires the original three bins and 25 paired mocks")
    if preparation["tracer"] != TRACER or list(preparation["realization_ids"]) != ids:
        raise ValueError("preparation tracer or paired mock identities differ")
    cap_areas = {
        cap: float(config["tracers"][TRACER]["footprint"]["area_deg2"][cap]) for cap in CAPS
    }
    p0 = float(config["tracers"][TRACER]["fkp_p0_h3_mpc3"])
    f_ref = float(settings["reference_f_sky"])
    ratio = f_ref * FULL_SKY_AREA_DEG2 / sum(cap_areas.values())
    targets = {tier["label"]: tier["target_nbar_h3_mpc3"] for tier in config["selection"]["tiers"]}
    matched_bins = list(settings["measurement_bins"])
    bounded_allowed = {tuple(item) for item in settings["bounded_allowed"]}

    def g(density):
        return density * p0 / (1.0 + density * p0)

    inputs = {
        "count_authority": NAME,
        "reference_area_deg2": float(settings["reference_area_deg2"]),
        "reference_f_sky": f_ref,
        "full_sky_area_deg2": float(FULL_SKY_AREA_DEG2),
        "cap_areas_deg2": cap_areas,
        "volume_ratio": ratio,
        "p0_h3_mpc3": p0,
        "spherex_nbar_h3_mpc3": {
            condition: {name: float(targets[condition][name]) for name in names}
            for condition in CONDITIONS
        },
        "measurement_bins": matched_bins,
        "bounded_allowed": sorted(list(pair) for pair in bounded_allowed),
        "conditions": {},
    }
    plan = {}
    for condition in CONDITIONS:
        groups = {cap: _source_group(preparation, condition, cap, names, ids) for cap in CAPS}
        inputs["conditions"][condition] = groups
        plan[condition] = {}
        for column, name in enumerate(names):
            n_spherex = float(targets[condition][name])
            volumes = {cap: groups[cap]["volumes_mpc_h3"][name] for cap in CAPS}
            baseline = {cap: groups[cap]["counts"][name] for cap in CAPS}
            ceiling = {
                cap: int(min(row[column] for row in groups[cap]["attainable_counts_by_mock"]))
                for cap in CAPS
            }
            g_matched = math.sqrt(ratio) * g(n_spherex)
            unattainable = g_matched >= 1.0
            if unattainable == (name in matched_bins):
                raise ValueError(
                    f"configured matched bins must be exactly the attainable bins: {condition} {name}"
                )
            if unattainable:
                n_matched, requested = None, {cap: None for cap in CAPS}
                selected, bounded = dict(baseline), False
            else:
                n_matched = g_matched / ((1.0 - g_matched) * p0)
                requested = {cap: int(math.floor(n_matched * volumes[cap])) for cap in CAPS}
                selected = {cap: min(requested[cap], ceiling[cap]) for cap in CAPS}
                bounded = any(selected[cap] < requested[cap] for cap in CAPS)
                if bounded and (condition, name) not in bounded_allowed:
                    raise ValueError(
                        f"bounded matched sample is not authorized: {condition} {name}"
                    )
                if any(selected[cap] < baseline[cap] for cap in CAPS):
                    raise ValueError(
                        f"matched density would select fewer objects than the primary: {condition} {name}"
                    )
            total = sum(volumes.values())
            veff_target = ratio * total * g(n_spherex) ** 2
            veff_attained = sum(volumes[cap] * g(selected[cap] / volumes[cap]) ** 2 for cap in CAPS)
            veff_primary = sum(volumes[cap] * g(baseline[cap] / volumes[cap]) ** 2 for cap in CAPS)
            plan[condition][name] = {
                "in_experiment": name in matched_bins,
                "unattainable": unattainable,
                "bounded": bounded,
                "n_spherex_h3_mpc3": n_spherex,
                "g_spherex": g(n_spherex),
                "g_matched": g_matched,
                "n_matched_h3_mpc3": n_matched,
                "n_primary_h3_mpc3": sum(baseline.values()) / total,
                "n_attained_h3_mpc3": sum(selected.values()) / total,
                "factor_matched_over_spherex": None if n_matched is None else n_matched / n_spherex,
                "factor_selected_over_primary": sum(selected.values()) / sum(baseline.values()),
                "veff_target_mpc_h3": veff_target,
                "veff_attained_mpc_h3": veff_attained,
                "veff_primary_mpc_h3": veff_primary,
                "veff_fraction": veff_attained / veff_target,
                "shortfall": {
                    cap: 0 if requested[cap] is None else requested[cap] - selected[cap]
                    for cap in CAPS
                },
                "caps": {
                    cap: {
                        "volume_mpc_h3": volumes[cap],
                        "baseline": baseline[cap],
                        "requested": requested[cap],
                        "ceiling": ceiling[cap],
                        "selected": selected[cap],
                    }
                    for cap in CAPS
                },
            }
    return {"rule": RULE, "inputs": inputs, "conditions": plan}


def plan_counts(plan, condition, cap, key="selected"):
    """Per-bin counts of one tier and cap from a matched plan (``selected`` or ``baseline``)."""
    return {name: row["caps"][cap][key] for name, row in plan["conditions"][condition].items()}


def selection_config(config):
    """The matched tiers take exact counts: their density targets are withdrawn, nothing else."""
    result = deepcopy(config)
    for tier in result["selection"]["tiers"]:
        if tier["label"] in CONDITIONS:
            tier["target_nbar_h3_mpc3"] = None
    return result


def matched_selection_record(record, config, tracer, condition, cap):
    """Relabel one selector record from the bound handoff; every other record passes through.

    The selector takes exact counts and no density target, so for a matched tier it reports a
    native attainable sample. Its counts are right; the handoff's matching plan is the authority
    for the target and the bounded flag. Saved records keep their bytes and readers call this
    too, so published density runs report the same provenance as new ones.
    """
    density = config["selection"].get("density_run")
    if density is None or tracer != TRACER or condition not in CONDITIONS:
        return record
    plan = density["handoff"]["definition"]["matching"]["conditions"][condition]
    bins = dict(record["bins"])
    for name, row in plan.items():
        bins[name] = {
            **bins[name],
            "target_basis": (
                "veff_matched_handoff" if row["in_experiment"] else "primary_selection_retained"
            ),
            "requested_nbar_h3_mpc3": row["n_matched_h3_mpc3"],
            "requested_count": row["caps"][cap]["requested"],
            "bounded_by_attainable": row["shortfall"][cap] > 0,
        }
    return {**record, "bins": bins}


# ------------------------------------------------------------------ run configuration


def density_run_config(config, handoff, expected_snapshot):
    """Keep the identified primary selection; replace only its downstream prescription."""
    from . import validate_analysis_config

    validate_analysis_config(config)
    recon = config["measurement"]["reconstruction"]
    if (
        config["schema_version"] != 17
        or any(key in recon for key in ("transfer_run", "sensitivity_run"))
        or "density_run" in config["selection"]
    ):
        raise ValueError(
            "density successors derive from the schema-17 per-bin primary configuration"
        )
    result = deepcopy(config)
    result["selection"]["density_run"] = {
        "name": NAME,
        "handoff_sha256": expected_snapshot,
        "handoff": deepcopy(handoff),
    }
    result["analysis"]["workloads"]["density"] = {
        "measurement_bins": list(config["selection"]["veff_matched_density"]["measurement_bins"]),
        "sigma_z_tiers": list(CONDITIONS),
        "reconstruction_states": list(STATES),
        "mock_ids": list(config["analysis"]["mock_ids"]),
    }
    validate_analysis_config(result)
    return result


def validate_density_run(config):
    """Reject changed streams, weights, cohorts, counts or an unidentified density definition.

    Every count is recomputed from the recorded matching inputs under the configuration's own
    reference sky fraction, P0, cap areas and bounded set; a handoff derived under other values
    does not validate, however self-consistent its digest.
    """
    from . import analysis_config_sha256, same_identity

    record = config["selection"]["density_run"]
    recon = config["measurement"]["reconstruction"]
    saved = record["handoff"]
    definition = saved["definition"]
    digest = record["handoff_sha256"]
    ids = list(config["analysis"]["mock_ids"])
    settings = _settings(config)
    if (
        config["schema_version"] != 17
        or record["name"] != NAME
        or definition["count_authority"] != NAME
        or digest != saved["resolved_snapshot_sha256"]
        or digest != analysis_config_sha256(definition)
        or recon["scope"] != "separate_measurement_bins"
        or recon["allowed_conventions"] != ["recsym"]
        or recon["smoothing_values_mpc_h"] != [15.0]
        or "transfer_run" in recon
        or "sensitivity_run" in recon
        or saved["scope"] != handoff_scope(ids)
    ):
        raise ValueError(
            "density successor differs from the identified V_eff-matched R15 experiment"
        )
    previous = definition["configuration"]
    selection = deepcopy(config["selection"])
    selection.pop("density_run")
    for tier in selection["tiers"]:
        if tier["label"] in CONDITIONS:
            tier["target_nbar_h3_mpc3"] = None
    for old, new in (
        (previous["selection"], selection),
        (previous["physics"], config["physics"]),
        (previous["tracers"], config["tracers"]),
        (previous["measurement"]["weights"], config["measurement"]["weights"]),
        (previous["analysis"]["mock_ids"], config["analysis"]["mock_ids"]),
    ):
        if not same_identity(old, new):
            raise ValueError("density successor changes inputs, weights, streams or cohort")
    preparation = definition["preparation"]
    if (
        ids != list(range(25))
        or preparation["realization_ids"] != ids
        or preparation["tracer"] != TRACER
    ):
        raise ValueError("density handoff requires the original paired 25-mock LRG cohort")
    workload = config["analysis"]["workloads"].get("density")
    if workload != {
        "measurement_bins": list(settings["measurement_bins"]),
        "sigma_z_tiers": list(CONDITIONS),
        "reconstruction_states": list(STATES),
        "mock_ids": ids,
    }:
        raise ValueError("density successor changes the comparison workload")
    matching = definition["matching"]
    recomputed = matched_density_plan(
        {"tracer": TRACER, "realization_ids": ids, "conditions": matching["inputs"]["conditions"]},
        config,
    )
    if not same_identity(matching, recomputed):
        raise ValueError("density handoff counts do not follow the configured matching rule")
    for condition in CONDITIONS:
        for cap in CAPS:
            group = preparation["conditions"][condition][cap]
            radial = definition["cap_products"][condition][cap]["radial"]
            rows = definition["selection_records"][condition][cap]
            counts = plan_counts(matching, condition, cap)
            baseline = plan_counts(matching, condition, cap, "baseline")
            if (
                group["radial"] != radial
                or group["counts"] != counts
                or radial["sample_identity"] != radial_sample_identity(condition, cap)
                or radial["calibration_mock_ids"] != ids
                or not radial["convergence_evidence"]["converged"]
                or [row["mock"] for row in rows] != ids
            ):
                raise ValueError("density radial identity, calibration evidence or counts differ")
            expected_bins = {
                name: {"baseline": baseline[name], "selected": counts[name]} for name in counts
            }
            for row in rows:
                if (
                    row["cap"] != cap
                    or row["condition"] != condition
                    or row["count_authority"] != NAME
                    or not row["pairing"]["object_noise_and_ranks_identical"]
                ):
                    raise ValueError("density selection lost paired noise or identities")
                if row["pairing"]["bins"] != expected_bins:
                    raise ValueError("density handoff changes the exact matched counts")
            if not np.array_equal(
                radial["mean_counts"], np.mean([row["histogram"] for row in rows], axis=0)
            ):
                raise ValueError("density radial target differs from its selected ensemble")


def density_preparation(config, tracer):
    """The bound handoff's preparation; geometry and growth are recomputed from its randoms."""
    validate_density_run(config)
    if tracer != TRACER:
        raise ValueError("the matched density experiment supplies only LRG")
    record = config["selection"]["density_run"]
    result = deepcopy(record["handoff"]["definition"]["preparation"])
    result["effective_redshifts"] = {}
    result["geometry"] = {}
    result["selection_source"] = {
        **record["handoff"]["definition"]["source"],
        "density_handoff_sha256": record["handoff_sha256"],
    }
    return result


# ------------------------------------------------------------------ handoff production


def source_identity(run):
    """The per-bin primary's identity including its resolved directory."""
    return {
        "root": str(run["root"]),
        **{k: run["metadata"][k] for k in ("run_name", "code_commit", "analysis_config_sha256")},
    }


def matched_source(run):
    """The published per-bin primary LRG selection supplying the counts; refuse anything else."""
    from ..selection import read_preparation
    from .stages import handoff_path

    config, metadata = run["config"], run["metadata"]
    recon = config["measurement"]["reconstruction"]
    if config["schema_version"] != 17:
        # The density successor pairs route by route with its source and shares its whole
        # scientific definition; a primary prepared under the earlier per-bin schema (16, no
        # matched-density block) cannot be that partner. Such a campaign is re-prepared at the
        # current configuration before the handoff is produced; nothing derives from it.
        raise ValueError(
            "the matched-density handoff derives from a schema-17 per-bin primary; a campaign "
            "prepared at schema 16 is re-prepared at the current configuration first"
        )
    if (
        recon["scope"] != "separate_measurement_bins"
        or "transfer_run" in recon
        or "sensitivity_run" in recon
        or "density_run" in config["selection"]
        or metadata.get("workload") != "full_grid"
        or metadata.get("mode") != "production"
        or "source_run" in metadata
    ):
        raise ValueError("the matched-density handoff derives from a published per-bin primary run")
    published = handoff_path(run["root"], "select")
    if not published.is_file():
        raise ValueError(f"the per-bin primary has not published its selection: {published}")
    handoff = yaml.safe_load(published.read_text())
    if (
        handoff.get("stage") != "select"
        or handoff.get("run_name") != metadata["run_name"]
        or handoff.get("analysis_config_sha256") != metadata["analysis_config_sha256"]
        or not any(item["unit"]["unit_id"] == TRACER for item in handoff.get("units", []))
    ):
        raise ValueError("the published selection does not identify this run's LRG unit")
    preparation = read_preparation(run, TRACER)
    return preparation, matched_density_plan(preparation, config)


def paired_membership(baseline, candidate, config, expected):
    """Prove exact counts, nested membership, unchanged bins and common object noise and ranks."""
    from ..catalog import measurement_bins

    stable = config["selection"]["stable_randomness"]
    key, observed = stable["identity_column"], stable["observed_redshift_column"]
    for catalog in (baseline, candidate):
        if len(np.unique(catalog[key])) != len(catalog[key]):
            raise ValueError("duplicate identities invalidate paired density selection")
    common, left, right = np.intersect1d(
        baseline[key], candidate[key], assume_unique=True, return_indices=True
    )
    if len(common) != len(baseline[key]):
        raise ValueError("matched sample lost primary members")
    for column in (
        stable["source_redshift_column"],
        stable["photoz_deviate_column"],
        stable["selection_rank_column"],
        observed,
        "RA",
        "DEC",
    ):
        if not np.array_equal(baseline[column][left], candidate[column][right]):
            raise ValueError(f"matched sample changes {column} of a common object")
    record = {}
    for item in measurement_bins(TRACER, config):
        masks = [
            (catalog[observed] >= item["zmin"]) & (catalog[observed] < item["zmax"])
            for catalog in (baseline, candidate)
        ]
        before, after = [catalog[key][mask] for catalog, mask in zip((baseline, candidate), masks)]
        target = expected[item["name"]]
        if len(before) != target["baseline"] or len(after) != target["selected"]:
            raise ValueError("selected count differs from the exact density authority")
        if len(np.intersect1d(before, after, assume_unique=True)) != len(before):
            raise ValueError("density membership is not nested within its redshift bin")
        if target["selected"] == target["baseline"] and not np.array_equal(
            np.sort(before), np.sort(after)
        ):
            raise ValueError(f"an unchanged bin changed its members: {item['name']}")
        record[item["name"]] = {"baseline": int(len(before)), "selected": int(len(after))}
    return {
        "bins": record,
        "common_objects": int(len(common)),
        "object_noise_and_ranks_identical": True,
    }


def select_pair(raw, preparation, config, plan, condition, mock, cap):
    """Call the native selector twice (primary, matched) and keep the matched radial histogram."""
    from ..diagnostics.selection import selected_data

    conditions = preparation["conditions"]
    counts = plan_counts(plan, condition, cap)
    baseline_counts = plan_counts(plan, condition, cap, "baseline")
    candidate_preparation = {
        **preparation,
        "conditions": {
            **conditions,
            condition: {
                **conditions[condition],
                cap: {**conditions[condition][cap], "counts": counts},
            },
        },
    }
    baseline, _ = selected_data(raw, preparation, config, TRACER, condition, mock, cap)
    candidate, native = selected_data(
        raw, candidate_preparation, selection_config(config), TRACER, condition, mock, cap
    )
    expected = {
        name: {"baseline": baseline_counts[name], "selected": counts[name]} for name in counts
    }
    proof = paired_membership(baseline, candidate, config, expected)
    observed = config["selection"]["stable_randomness"]["observed_redshift_column"]
    histogram = np.histogram(
        candidate[observed], bins=conditions[condition][cap]["radial"]["z_edges"]
    )[0]
    if histogram.sum() != len(candidate[observed]):
        raise ValueError("radial support loses selected galaxies")
    return histogram, {
        "mock": mock,
        "cap": cap,
        "condition": condition,
        "count_authority": NAME,
        "pairing": proof,
        "native_selector_record": native["selection"],
        "interpretation": (
            "The native selector uses externally supplied exact counts; its native_attainable "
            "label is not the authority for this density variant."
        ),
    }


def random_targets(mass, required):
    """Match the native floor plus stable-largest-remainder allocation exactly."""
    mass = np.asarray(mass, dtype=float)
    if (
        mass.ndim != 1
        or not len(mass)
        or not np.isfinite(mass).all()
        or np.any(mass < 0)
        or not np.isclose(mass.sum(), 1.0, rtol=0, atol=1e-12)
        or type(required) is not int
        or required <= 0
    ):
        raise ValueError("invalid smooth random target")
    raw = mass * required
    counts = np.floor(raw).astype(int)
    remainder = required - int(counts.sum())
    if not 0 <= remainder <= len(counts):
        raise ValueError("random mass cannot produce the exact integer budget")
    if remainder:
        counts[np.argsort(-(raw - counts), kind="stable")[:remainder]] += 1
    return counts


def random_support(radial, source_group, data_count, ratio):
    """Check the exact new random targets against the saved per-bin random availability."""
    if not np.array_equal(radial["z_edges"], source_group["radial"]["z_edges"]):
        raise ValueError("matched radial support must keep the source redshift bins")
    if (
        radial["calibration_mock_ids"] != list(range(25))
        or [row["realization_id"] for row in source_group["selection_diagnostics_by_mock"]]
        != radial["calibration_mock_ids"]
    ):
        raise ValueError("random support must identify every paired mock exactly once")
    targets = random_targets(radial["target_probability_mass"], int(ratio) * int(data_count))
    checks = []
    for row in source_group["selection_diagnostics_by_mock"]:
        saved = row["randoms"]
        available = np.asarray(saved["available_counts_by_bin"])
        if (
            available.shape != targets.shape
            or available.dtype.kind not in "iu"
            or np.any(available < 0)
            or saved["radial_sample_identity"] != source_group["radial"]["sample_identity"]
            or saved["exact_random_data_ratio"] != ratio
            or saved["retained_rows_unchanged"] is not True
        ):
            raise ValueError("saved random support is incompatible with the density target")
        if np.any(targets > available):
            bad = np.flatnonzero(targets > available).tolist()
            raise ValueError(f"random target unavailable: mock {row['realization_id']} bins {bad}")
        checks.append(
            {
                "mock": row["realization_id"],
                "minimum_bin_surplus": int(np.min(available - targets)),
                "available_total_in_support": int(available.sum()),
            }
        )
    return {
        "target_counts_by_bin": targets.tolist(),
        "randoms_per_mock": int(targets.sum()),
        "exact_random_data_ratio": ratio,
        "checks": checks,
        "scope": "Saved support only; the native select stage validates actual random rows.",
    }


def input_preparation(original, config, plan, cap_products, selection_records):
    """Build the native-shaped matched preparation; every other tier stays the primary's."""
    ids = list(range(25))
    if (
        set(cap_products) != set(CONDITIONS)
        or set(selection_records) != set(CONDITIONS)
        or any(
            set(cap_products[c]) != set(CAPS) or set(selection_records[c]) != set(CAPS)
            for c in CONDITIONS
        )
    ):
        raise ValueError("density handoff requires both matched tiers and both caps")
    settings = config["selection"]["radial_randoms"]
    result = deepcopy(original)
    for condition in CONDITIONS:
        for cap in CAPS:
            source_group = original["conditions"][condition][cap]
            product, records = cap_products[condition][cap], selection_records[condition][cap]
            radial = product["radial"]
            counts = plan_counts(plan, condition, cap)
            baseline = plan_counts(plan, condition, cap, "baseline")
            if (
                radial["sample_identity"] != radial_sample_identity(condition, cap)
                or radial["calibration_mock_ids"] != ids
                or radial["convergence_evidence"]["converged"] is not True
                or [row["mock"] for row in records] != ids
            ):
                raise ValueError("density handoff has a wrong radial identity or mock sequence")
            if (
                radial["smoothing_method"] != settings["frozen_method"]
                or radial["bandwidth"] != settings["frozen_bandwidth"]
                or radial["convergence_evidence"]["residual_weighting"]
                != settings["frozen_residual_weighting"]
            ):
                raise ValueError("density handoff changed the frozen radial prescription")
            support = random_support(
                radial, source_group, sum(counts.values()), settings["exact_random_data_ratio"]
            )
            if support != product["random_support"]:
                raise ValueError("density handoff random targets differ from native allocation")
            expected_bins = {
                name: {"baseline": baseline[name], "selected": counts[name]} for name in counts
            }
            for row in records:
                if (
                    row["cap"] != cap
                    or row["condition"] != condition
                    or row["count_authority"] != NAME
                    or row["pairing"]["object_noise_and_ranks_identical"] is not True
                    or row["pairing"]["common_objects"] != sum(baseline.values())
                    or row["pairing"]["bins"] != expected_bins
                    or any(
                        row["native_selector_record"]["bins"][name]["attained_count"]
                        != counts[name]
                        for name in counts
                    )
                ):
                    raise ValueError("density handoff lost paired membership or selected counts")
                histogram = np.asarray(row["histogram"])
                if (
                    histogram.shape != (len(radial["z_edges"]) - 1,)
                    or histogram.dtype.kind not in "iu"
                    or np.any(histogram < 0)
                    or histogram.sum() != sum(counts.values())
                ):
                    raise ValueError("density handoff has incomplete radial selection evidence")
            if not np.array_equal(
                radial["mean_counts"], np.mean([row["histogram"] for row in records], axis=0)
            ):
                raise ValueError("density radial target differs from its selected ensemble")
            group = result["conditions"][condition][cap]
            group["counts"] = dict(counts)
            group["radial"] = deepcopy(radial)
            # Primary random-row diagnostics cannot be attributed to the matched sample.
            group.pop("selection_diagnostics_by_mock")
            group["density_selection_by_mock"] = deepcopy(records)
            group["density_random_support"] = support
            group["actual_random_rows_validated"] = False
    return result


def make_handoff(run, original, plan, cap_products, selection_records):
    """Bind the matched products to their source, matching inputs and one resolved definition."""
    from . import analysis_config_sha256

    definition = {
        "source": source_identity(run),
        "source_preparation_sha256": analysis_config_sha256(original),
        "count_authority": NAME,
        "configuration": selection_config(run["config"]),
        "matching": plan,
        "cap_products": cap_products,
        "selection_records": selection_records,
        "preparation": input_preparation(
            original, run["config"], plan, cap_products, selection_records
        ),
    }
    return {
        "definition": definition,
        "resolved_snapshot_sha256": analysis_config_sha256(definition),
        "scope": handoff_scope(run["config"]["analysis"]["mock_ids"]),
        "native_inputs_published": False,
        "actual_random_rows_validated": False,
        "scientific_acceptance": False,
    }


def read_handoff(path, run, original, expected_snapshot):
    """Check the declared source, rule and snapshot before exposing the matched preparation."""
    from . import analysis_config_sha256

    saved = json.loads(Path(path).read_text())
    definition = saved["definition"]
    if (
        not isinstance(expected_snapshot, str)
        or len(expected_snapshot) != 64
        or saved["resolved_snapshot_sha256"] != expected_snapshot
        or analysis_config_sha256(definition) != expected_snapshot
    ):
        raise ValueError("density handoff snapshot differs from the explicit consumer selector")
    expected = make_handoff(
        run,
        original,
        matched_density_plan(original, run["config"]),
        definition["cap_products"],
        definition["selection_records"],
    )
    if saved != expected:
        raise ValueError("density handoff changed source, configuration, evidence or preparation")
    return saved["definition"]["preparation"], saved["definition"]["configuration"]


def _write_json(path, payload):
    """Keep every handoff product create-only."""
    with Path(path).open("x") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)


def produce_handoff(run, output, action, *, parallel=None):
    """Write the matched count plan, and for ``prepare-radial`` the complete handoff.

    ``counts`` records the plan for review. ``prepare-radial`` reads each raw parent once per
    realization and cap, selects the primary and matched samples of both tiers through the
    native selector, proves paired membership, fits the matched radial templates under the
    frozen prescription, checks the saved random support at the configured ratio and writes
    ``prepared-inputs.json`` with its resolved snapshot digest, re-reading it before completion.
    """
    from ..catalog import _read_parent
    from ..selection import fit_smooth_radial_model

    if action not in ("counts", "prepare-radial"):
        raise ValueError("density handoff actions are counts and prepare-radial")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    preparation, plan = matched_source(run)
    config = run["config"]
    ids = list(config["analysis"]["mock_ids"])
    _write_json(
        output / "attempt.json",
        {
            "source": source_identity(run),
            "count_authority": NAME,
            "action": action,
            "matching": plan,
            "scientific_acceptance": False,
        },
    )
    handoff = None
    try:
        if action == "prepare-radial":
            settings = config["selection"]["radial_randoms"]
            cap_products = {condition: {} for condition in CONDITIONS}
            selection_records = {condition: {cap: [] for cap in CAPS} for condition in CONDITIONS}
            histograms = {condition: {cap: [] for cap in CAPS} for condition in CONDITIONS}
            for cap in CAPS:
                for mock in ids:
                    raw = _read_parent(
                        preparation["sources"],
                        tracer=TRACER,
                        mock_id=mock,
                        cap=cap,
                        config=config,
                        include_randoms=False,
                        parallel=parallel,
                    )
                    for condition in CONDITIONS:
                        histogram, proof = select_pair(
                            raw, preparation, config, plan, condition, mock, cap
                        )
                        histograms[condition][cap].append(histogram)
                        record = {**proof, "histogram": histogram.tolist()}
                        selection_records[condition][cap].append(record)
                        _write_json(output / f"{condition}--{cap}--mock-{mock}.json", record)
                        print(
                            f"Paired selection checked: {condition} {cap} mock {mock}", flush=True
                        )
                    del raw
                for condition in CONDITIONS:
                    old = preparation["conditions"][condition][cap]
                    radial = fit_smooth_radial_model(
                        histograms[condition][cap],
                        old["radial"]["z_edges"],
                        calibration_mock_ids=ids,
                        smoothing_method=settings["frozen_method"],
                        bandwidth=settings["frozen_bandwidth"],
                        residual_weighting=settings["frozen_residual_weighting"],
                        sample_identity=radial_sample_identity(condition, cap),
                    )
                    if not radial["convergence_evidence"]["converged"]:
                        raise ValueError("matched ensemble radial target did not converge")
                    cap_products[condition][cap] = {
                        "radial": radial,
                        "random_support": random_support(
                            radial,
                            old,
                            sum(plan_counts(plan, condition, cap).values()),
                            settings["exact_random_data_ratio"],
                        ),
                    }
                    _write_json(
                        output / f"{condition}--{cap}--radial.json", cap_products[condition][cap]
                    )
            handoff = make_handoff(run, preparation, plan, cap_products, selection_records)
            _write_json(output / "prepared-inputs.json", handoff)
            read_handoff(
                output / "prepared-inputs.json",
                run,
                preparation,
                handoff["resolved_snapshot_sha256"],
            )
        _write_json(
            output / "complete.json",
            {
                "action": action,
                "count_authority": NAME,
                "input_handoff": "prepared-inputs.json" if handoff else None,
                "resolved_snapshot_sha256": (
                    handoff["resolved_snapshot_sha256"] if handoff else None
                ),
                "native_inputs_published": False,
                "actual_random_rows_validated": False,
                "scientific_acceptance": False,
            },
        )
    except Exception as exc:
        _write_json(output / "failure.json", {"error": str(exc), "scientific_acceptance": False})
        raise
    return plan, handoff
