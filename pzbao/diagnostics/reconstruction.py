"""Reusable reconstruction diagnostics. Inputs and experiment choices are explicit arguments."""

from __future__ import annotations

import math

import numpy as np

from pzbao.config import load_analysis_config

from typing import Any, Dict, List, Mapping, Tuple

from pzbao.catalog import measurement_bins

from pzbao.selection import fit_smooth_radial_model, select_exact_attainable

import copy

import hashlib

from typing import Sequence, Union

from unittest.mock import patch

_threshold_loss_anatomy_DEEP_INTERIOR_SMOOTHING_MULTIPLE = 1.0


def classify_loss(
    unsmoothed: np.ndarray,
    smoothed: np.ndarray,
    *,
    absolute_threshold: float,
    cellsize_mpc_h: float,
    smoothing_radius_mpc_h: float,
) -> dict:
    """Say where the thresholded cells sit relative to the footprint boundary.

    ``depth`` is measured on the painted support, because that is the set the project's own
    mask is built on and the set the loss fractions are quoted over.
    """
    from scipy import ndimage

    support = unsmoothed > 0
    n_support = int(np.count_nonzero(support))
    if n_support == 0:
        raise ValueError("threshold support is empty")
    retained = smoothed >= float(absolute_threshold)
    lost = support & ~retained
    n_lost = int(np.count_nonzero(lost))
    depth_cells = ndimage.distance_transform_edt(support).astype(np.float32)
    depth_mpc_h = depth_cells * float(cellsize_mpc_h)
    edges = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, np.inf]) * float(
        smoothing_radius_mpc_h
    )
    profile = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        band = support & (depth_mpc_h > lo) & (depth_mpc_h <= hi)
        n_band = int(np.count_nonzero(band))
        if n_band == 0:
            continue
        n_band_lost = int(np.count_nonzero(band & lost))
        profile.append(
            {
                "depth_low_smoothing_radii": float(lo / smoothing_radius_mpc_h),
                "depth_high_smoothing_radii": (
                    float(hi / smoothing_radius_mpc_h) if np.isfinite(hi) else None
                ),
                "supported_cells": n_band,
                "lost_cells": n_band_lost,
                "lost_fraction_of_band": n_band_lost / n_band,
                "share_of_all_loss": n_band_lost / n_lost if n_lost else 0.0,
            }
        )
    deep = depth_mpc_h > _threshold_loss_anatomy_DEEP_INTERIOR_SMOOTHING_MULTIPLE * float(
        smoothing_radius_mpc_h
    )
    n_deep = int(np.count_nonzero(support & deep))
    n_deep_lost = int(np.count_nonzero(lost & deep))
    labels, n_components = ndimage.label(lost)
    interior_component_sizes: List[int] = []
    if n_components:
        touches_edge = np.zeros(n_components + 1, dtype=bool)
        shallow = lost & (depth_cells <= 1.0)
        for value in np.unique(labels[shallow]):
            if value:
                touches_edge[value] = True
        sizes = np.bincount(labels.ravel(), minlength=n_components + 1)
        for value in range(1, n_components + 1):
            if not touches_edge[value]:
                interior_component_sizes.append(int(sizes[value]))
    interior_component_sizes.sort(reverse=True)
    n_interior_component_cells = int(sum(interior_component_sizes))
    singletons = sum((1 for size in interior_component_sizes if size == 1))
    comparison = {}
    for name, cut in (
        ("project_derived", float(absolute_threshold)),
        ("desi_dr1_iterative_fft_0.01_per_object", 0.01),
        ("bare_literal_0.75_per_object", 0.75),
    ):
        removed = support & (smoothed < cut)
        n_removed = int(np.count_nonzero(removed))
        comparison[name] = {
            "absolute_cut": cut,
            "lost_cells": n_removed,
            "lost_fraction_of_support": n_removed / n_support,
        }
    return {
        "reference_cut_comparison": comparison,
        "support_cell_count": n_support,
        "lost_cell_count": n_lost,
        "lost_fraction_of_support": n_lost / n_support,
        "mean_randoms_per_supported_cell": float(np.sum(unsmoothed[support]) / n_support),
        "depth_profile": profile,
        "deep_interior_smoothing_multiple": _threshold_loss_anatomy_DEEP_INTERIOR_SMOOTHING_MULTIPLE,
        "deep_interior_supported_cells": n_deep,
        "deep_interior_lost_cells": n_deep_lost,
        "deep_interior_lost_fraction": n_deep_lost / n_deep if n_deep else 0.0,
        "deep_interior_share_of_all_loss": n_deep_lost / n_lost if n_lost else 0.0,
        "lost_components_total": int(n_components),
        "interior_component_count": len(interior_component_sizes),
        "interior_component_cells": n_interior_component_cells,
        "interior_component_share_of_all_loss": (
            n_interior_component_cells / n_lost if n_lost else 0.0
        ),
        "interior_singleton_components": int(singletons),
        "interior_component_sizes_largest": interior_component_sizes[:25],
    }


_threshold_loss_anatomy_SMOOTHED_EDGE_REACH_AT_ONE_RADIUS = 0.5 * math.erfc(1.0 / math.sqrt(2.0))


def random_threshold_diagnostics(
    unsmoothed_random_weight,
    smoothed_random_weight,
    *,
    random_object_count,
    total_random_weight,
    config=None,
):
    """Describe PyRecon's native floor with whole-mesh and support-restricted accounting.

    Permit only FFT-scale negative roundoff. The accounting floor is the Gaussian tail at
    one smoothing radius; no diagnostic fraction changes the reconstruction threshold.
    """
    resolved = config or load_analysis_config()
    policy = resolved["measurement"]["reconstruction"]["threshold_randoms"]
    unsmoothed = np.asarray(unsmoothed_random_weight, dtype=float)
    smoothed = np.asarray(smoothed_random_weight, dtype=float)
    if unsmoothed.shape != smoothed.shape or unsmoothed.size == 0:
        raise ValueError("threshold meshes must be nonempty and shape-identical")
    if np.any(~np.isfinite(unsmoothed)) or np.any(~np.isfinite(smoothed)):
        raise ValueError("threshold meshes must be finite")
    if np.any(unsmoothed < 0):
        raise ValueError("threshold mesh weights must be non-negative")
    peak = float(np.max(np.abs(smoothed)))
    roundoff_bound = float(np.finfo(float).eps * math.log2(max(smoothed.size, 2)) * peak)
    most_negative = float(np.min(smoothed))
    if most_negative < -roundoff_bound:
        raise ValueError("threshold mesh weights must be non-negative")
    if type(random_object_count) is not int:
        raise ValueError("random-object count must be an integer")
    count = random_object_count
    total = float(total_random_weight)
    if count <= 0 or not np.isfinite(total) or total <= 0:
        raise ValueError("threshold random count and total weight must be positive")
    support = unsmoothed > 0
    n_support = int(np.count_nonzero(support))
    if n_support == 0:
        raise ValueError("threshold support is empty")
    support_smoothed_weight = float(np.sum(smoothed[support]))
    mean_support_smoothed_weight = support_smoothed_weight / n_support
    threshold_value = float(policy["value"])
    absolute = threshold_value * total / count
    applied_mask = support & (smoothed >= absolute)
    rejected = support & ~applied_mask
    support_cell_loss = float(np.count_nonzero(rejected) / n_support)
    support_weight_loss = (
        float(np.sum(smoothed[rejected]) / support_smoothed_weight)
        if support_smoothed_weight > 0
        else 0.0
    )
    retained_everywhere = smoothed >= absolute
    total_smoothed_weight = float(np.sum(smoothed))
    if not np.isfinite(total_smoothed_weight) or total_smoothed_weight <= 0:
        raise ValueError("threshold mesh carries no smoothed random weight")
    weight_loss = float(
        (total_smoothed_weight - np.sum(smoothed[retained_everywhere])) / total_smoothed_weight
    )
    accounting_floor = (
        _threshold_loss_anatomy_SMOOTHED_EDGE_REACH_AT_ONE_RADIUS * mean_support_smoothed_weight
    )
    accounted = support | (smoothed >= accounting_floor)
    n_accounted = int(np.count_nonzero(accounted))
    accounted_rejected = accounted & ~retained_everywhere
    cell_loss = float(np.count_nonzero(accounted_rejected) / n_accounted)
    unaccounted_weight = float(
        (total_smoothed_weight - np.sum(smoothed[accounted])) / total_smoothed_weight
    )
    return {
        "support_predicate": str(policy["support_predicate"]),
        "loss_accounting_cell_set": "painted_support_union_smoothed_reach_at_one_smoothing_radius",
        "loss_accounting_weight_set": "complete_mesh",
        "mesh_shape": list(unsmoothed.shape),
        "random_object_count": count,
        "total_random_weight": total,
        "support_cell_count": n_support,
        "support_smoothed_weight": support_smoothed_weight,
        "mean_support_smoothed_weight": mean_support_smoothed_weight,
        "total_smoothed_weight": total_smoothed_weight,
        "accounted_cell_count": n_accounted,
        "accounting_floor_weight": accounting_floor,
        "accounting_floor_fraction_of_support_mean": _threshold_loss_anatomy_SMOOTHED_EDGE_REACH_AT_ONE_RADIUS,
        "unaccounted_smoothed_weight_fraction": unaccounted_weight,
        "most_negative_smoothed_weight": most_negative,
        "smoothed_transform_roundoff_bound": roundoff_bound,
        "threshold_value": threshold_value,
        "applied_absolute_threshold": absolute,
        "threshold_cell_loss_fraction": cell_loss,
        "threshold_weight_loss_fraction": weight_loss,
        "support_restricted_cell_loss_fraction": support_cell_loss,
        "support_restricted_weight_loss_fraction": support_weight_loss,
    }


_threshold_loss_anatomy_THRESHOLD_SUMMARY_KEYS = (
    "support_cell_count",
    "accounted_cell_count",
    "mean_support_smoothed_weight",
    "threshold_value",
    "applied_absolute_threshold",
    "threshold_cell_loss_fraction",
    "threshold_weight_loss_fraction",
    "support_restricted_cell_loss_fraction",
    "support_restricted_weight_loss_fraction",
    "unaccounted_smoothed_weight_fraction",
)


def threshold_summary(evidence: Mapping[str, Any]) -> dict:
    """Read the published quantities out of one field-evidence record."""
    return {key: evidence[key] for key in _threshold_loss_anatomy_THRESHOLD_SUMMARY_KEYS}


def ordered_edge_diagnostics(steps, *, realization_ids, config):
    """Label an optional ordered truncation study using the saved descriptive reference bands."""
    from pzbao.config import same_identity

    if not same_identity(realization_ids, config["analysis"]["mock_ids"]):
        raise ValueError("edge comparison requires the complete configured ensemble")
    if len(steps) < 2:
        raise ValueError("edge comparison requires at least two ordered support levels")
    bands = config["reporting"]["review_bands"]["edge_rehearsal"]
    keys = {
        "alpha_shift_sigma_1": "alpha_shift_sigma_1_bands",
        "s_xi": "s_xi_bands",
        "interval_width_fractional_change": "interval_width_fractional_change_bands",
    }
    result, previous = ([], -1.0)
    for index, step in enumerate(steps):
        if type(step["order"]) is not int or step["order"] != index:
            raise ValueError("edge comparison step order must be contiguous")
        fraction = float(step["lost_fraction"])
        if not np.isfinite(fraction) or not previous <= fraction <= 1 or fraction < 0:
            raise ValueError("edge comparison lost fractions must increase within [0, 1]")
        previous = fraction
        labels = {}
        for name, band_key in keys.items():
            value = abs(float(step[name]))
            if not np.isfinite(value):
                raise ValueError("edge comparison quantities must be finite")
            green, amber = bands[band_key]
            labels[name] = "green" if value <= green else "amber" if value <= amber else "red"
        result.append({**step, "reference_bands": labels})
    return result


def tier_support_overlap_sample_identity(
    tracer: str, bin_name: str, tier_label: str, cap: str
) -> str:
    """The sample this window's radial target belongs to, bin included.

    Production identities name tracer, tier, density and cap. The measurement bin is added here
    because this module scopes the target to one window, and an identity that omitted the window
    would claim the artifact is the sample-wide one.
    """
    return f"{tracer}|{bin_name}|{tier_label}|density=1|cap={cap}"


def tier_support_overlap_frozen_counts(
    smeared_by_mock: Mapping[int, Mapping[str, np.ndarray]],
    tracer: str,
    tier_label: str,
    volumes: Mapping[str, float],
    config: Mapping[str, Any],
) -> Dict[str, int]:
    """The common attainable count per bin: the across-mock minimum, bounded by the tier target.

    This is the count artifact's job in production. It is recomputed here rather than read,
    because no accepted count artifact exists yet, and it is the same rule: a cell may never
    request more objects than the sparsest mock can supply, nor more than its density target.
    """
    from pzbao.selection import tier_spec

    targets = tier_spec(tier_label, config)["target_nbar_h3_mpc3"]
    observed_column = str(config["selection"]["stable_randomness"]["observed_redshift_column"])
    frozen: Dict[str, int] = {}
    for item in measurement_bins(tracer, config):
        name = str(item["name"])
        attainable = min(
            (
                int(
                    np.count_nonzero(
                        (np.asarray(smeared[observed_column]) >= float(item["zmin"]))
                        & (np.asarray(smeared[observed_column]) < float(item["zmax"]))
                    )
                )
                for smeared in smeared_by_mock.values()
            )
        )
        if targets is None:
            frozen[name] = attainable
            continue
        requested = int(np.floor(float(targets[name]) * float(volumes[name])))
        frozen[name] = min(attainable, requested)
    return frozen


def tier_radial_target(
    smeared_by_mock: Mapping[int, Mapping[str, np.ndarray]],
    *,
    tracer: str,
    bin_name: str,
    cap: str,
    tier_label: str,
    volumes: Mapping[str, float],
    window: Tuple[float, float],
    config: Mapping[str, Any],
) -> Tuple[dict, Dict[str, Any]]:
    """Fit each tier's complete-ensemble radial target using the scientific selection kernels.

    Preserve counts, support, spline convergence and budget diagnostics without selecting a
    different prescription. This optional comparison does not change production settings.
    """
    radial = config["selection"]["radial_randoms"]
    observed_column = str(config["selection"]["stable_randomness"]["observed_redshift_column"])
    width = float(radial["support_bin_width_redshift"])
    zlo, zhi = window
    edge_count = int(round((zhi - zlo) / width))
    edges = zlo + width * np.arange(edge_count + 1, dtype=float)
    frozen = tier_support_overlap_frozen_counts(
        smeared_by_mock, tracer, tier_label, volumes, config
    )
    counts_by_mock: List[np.ndarray] = []
    window_counts: List[int] = []
    for mock_id in sorted(smeared_by_mock):
        selected, _ = select_exact_attainable(
            smeared_by_mock[mock_id],
            tracer=tracer,
            tier_label=tier_label,
            shell_volumes_mpc_h3=volumes,
            frozen_counts_by_bin=frozen,
            config=config,
        )
        observed = np.asarray(selected[observed_column], dtype=float)
        inside = observed[(observed >= edges[0]) & (observed < edges[-1])]
        counts_by_mock.append(np.histogram(inside, bins=edges)[0].astype(float))
        window_counts.append(int(inside.size))
    candidate = fit_smooth_radial_model(
        np.asarray(counts_by_mock),
        edges.tolist(),
        calibration_mock_ids=sorted(smeared_by_mock),
        smoothing_method=radial["frozen_method"],
        bandwidth=radial["frozen_bandwidth"],
        sample_identity=tier_support_overlap_sample_identity(tracer, bin_name, tier_label, cap),
        residual_weighting=str(radial["frozen_residual_weighting"]),
    )
    artifact = candidate
    summary = {
        "tier": tier_label,
        "frozen_counts_by_bin": frozen,
        "window_selected_count_by_mock": window_counts,
        "total_variation_distance": float(
            candidate["fidelity_evidence"]["total_variation_distance"]
        ),
        "mean_fractional_variance": float(
            candidate["variance_evidence"]["mean_fractional_variance_over_occupied_bins"]
        ),
        "converged": bool(candidate["convergence_evidence"].get("converged")),
    }
    return (artifact, summary)


def target_counts_by_bin(artifact: Mapping[str, Any], required_count: int) -> np.ndarray:
    """The exact per-bin allocation, by the arithmetic ``thin_randoms_to_smooth_target`` uses.

    Reproduced here so containment between two tiers can be decided from the allocation alone,
    without painting: tier A's selection is contained in tier B's exactly when A's allocation is
    no larger than B's in every bin, because both keep the lowest ranks of the same candidate
    pool within each bin.
    """
    mass = np.asarray(artifact["target_probability_mass"], dtype=float)
    raw = mass * int(required_count)
    counts = np.floor(raw).astype(int)
    remainder = int(required_count) - int(counts.sum())
    if remainder:
        order = np.argsort(-(raw - counts), kind="stable")
        counts[order[:remainder]] += 1
    return counts


def tier_support_overlap_overlap(first: np.ndarray, second: np.ndarray) -> dict:
    intersection = int(np.count_nonzero(first & second))
    union = int(np.count_nonzero(first | second))
    return {
        "intersection": intersection,
        "union": union,
        "intersection_over_union": intersection / union if union else 0.0,
        "first_only": int(np.count_nonzero(first & ~second)),
        "second_only": int(np.count_nonzero(second & ~first)),
        "first_contained_in_second": bool(not np.any(first & ~second)),
        "second_contained_in_first": bool(not np.any(second & ~first)),
    }


def tier_support_overlap_shell_profile(
    disagreement: np.ndarray, support: np.ndarray, cell: float, radius: float
):
    """Where the tiers disagree, in smoothing radii from the edge of the shared support."""
    from scipy import ndimage

    if not np.any(disagreement):
        return []
    depth = ndimage.distance_transform_edt(support, sampling=(cell, cell, cell))
    values = depth[disagreement] / float(radius)
    bands = [(0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, np.inf)]
    total = int(values.size)
    return [
        {
            "band_smoothing_radii": [low, None if np.isinf(high) else high],
            "cells": int(np.count_nonzero((values >= low) & (values < high))),
            "fraction": float(np.count_nonzero((values >= low) & (values < high)) / total),
        }
        for low, high in bands
    ]


def reconstruct_bins(data, randoms, *, solver, kwargs, bins, observed_column, weight):
    """Solve each bin on the same parent mesh; scatter shifts back by original row index."""
    from pzbao.reconstruction import SHIFTED_COLUMNS

    output = [dict(data), dict(randoms)]
    covered = [np.zeros(len(c["Z"]), dtype=bool) for c in (data, randoms)]
    for target, catalog in zip(output, (data, randoms)):
        for column in SHIFTED_COLUMNS:
            target[column] = np.full(len(catalog["Z"]), np.nan)
    parent_mean_weight = float(np.mean(weight(randoms)))
    records = {}
    for item in bins:
        masks = [
            (c[z] >= item["zmin"]) & (c[z] < item["zmax"])
            for c, z in ((data, observed_column), (randoms, "Z"))
        ]
        subsets = [
            {key: np.asarray(value)[mask] for key, value in c.items()}
            for c, mask in zip((data, randoms), masks)
        ]
        if any((not np.any(mask) for mask in masks)):
            raise ValueError("empty data/random bin cannot define a displacement field")
        local = copy.deepcopy(kwargs)
        local["threshold_randoms"] *= parent_mean_weight / np.mean(weight(subsets[1]))
        shifted_data, shifted_randoms, record = solver(*subsets, **local)
        for target, shifted, mask, seen in zip(
            output, (shifted_data, shifted_randoms), masks, covered
        ):
            if np.any(seen & mask):
                raise ValueError("measurement bins overlap")
            for column in SHIFTED_COLUMNS:
                target[column][mask] = shifted[column]
            seen |= mask
        records[item["name"]] = record
    if any((not np.all(mask) for mask in covered)):
        raise ValueError("bin reconstruction omitted parent rows")
    return (
        output[0],
        output[1],
        {
            "scope": "analysis_bins",
            "bins": records,
            "absolute_floor": kwargs["threshold_randoms"] * parent_mean_weight,
        },
    )


def apply_saved_shifts(catalog, files, identity_column, *, allow_subset=False):
    """Reuse shifts only after exact ID and unshifted-position joins; row order is irrelevant."""
    from pzbao.reconstruction import SHIFTED_COLUMNS

    saved = []
    for path in files:
        with np.load(path, allow_pickle=False) as archive:
            saved.append(
                {key: archive[key] for key in ("source_ids", "positions", "shifted_positions")}
            )
    ids = np.concatenate([record["source_ids"] for record in saved])
    order = np.argsort(ids)
    ids = ids[order]
    wanted = np.asarray(catalog[identity_column])
    if len(ids) != len(wanted) and (not allow_subset) or len(np.unique(ids)) != len(ids):
        raise ValueError("saved shifts have duplicate or incomplete identities")
    indices = np.searchsorted(ids, wanted)
    if np.any(indices == len(ids)) or not np.array_equal(ids[indices], wanted):
        raise ValueError("saved shifts belong to different objects")
    positions = np.concatenate([r["positions"] for r in saved])[order][indices]
    np.testing.assert_array_equal(
        positions, np.column_stack([catalog[k] for k in ("X", "Y", "Z_CART")])
    )
    shifts = np.concatenate([r["shifted_positions"] for r in saved])[order][indices]
    if not np.all(np.isfinite(shifts)):
        raise ValueError("saved shifts are not finite")
    result = dict(catalog)
    for i, key in enumerate(SHIFTED_COLUMNS):
        result[key] = shifts[:, i]
    return result


def scientific_columns(config, role):
    controls = config["selection"]["stable_randomness"]
    return (
        controls["identity_column"],
        controls["observed_redshift_column"] if role == "data" else "Z",
        "X",
        "Y",
        "Z_CART",
        "WEIGHT",
        "WEIGHT_FKP",
    )


def catalog_fingerprint(catalog, config, role):
    result = {}
    lengths = set()
    for name in scientific_columns(config, role):
        array = np.asarray(catalog[name])
        if array.ndim != 1 or not len(array) or (not np.all(np.isfinite(array))):
            raise ValueError(f"invalid {role} column {name}")
        lengths.add(len(array))
        result[name] = {
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "sha256": hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest(),
        }
    if len(lengths) != 1:
        raise ValueError("catalogue column lengths differ")
    return result


def case_kwargs(original, case, width, family_builder):
    """Change solver inputs only; keep the numerical mesh and the original selection."""
    kwargs = dict(original)
    config = copy.deepcopy(original["config"])
    reconstruction = config["measurement"]["reconstruction"]
    reconstruction["smoothing_values_mpc_h"] = [float(case["smoothing_mpc_h"])]
    reconstruction["threshold_randoms"]["value"] = float(case["threshold_randoms"])
    reconstruction["shell_diagnostics"]["shell_width_smoothing_multiple"] = float(width) / float(
        case["smoothing_mpc_h"]
    )
    family = family_builder(
        tracer=original["mask_family"]["tracer"],
        cap=original["mask_family"]["cap"],
        smoothing_radius_mpc_h=case["smoothing_mpc_h"],
    )
    geometry = copy.deepcopy(original["common_geometry"])
    geometry["family"] = family
    kwargs.update(
        config=config,
        mask_family=family,
        common_geometry=geometry,
        smoothing_radius_mpc_h=float(case["smoothing_mpc_h"]),
        threshold_randoms=float(case["threshold_randoms"]),
    )
    return kwargs


def check_reproduction(candidate, reference, design):
    if candidate.count_names != reference.count_names:
        raise ValueError("reference count types differ")
    for a, b in zip(candidate.edges, reference.edges):
        np.testing.assert_array_equal(a, b)
    for name in candidate.count_names:
        a, b = (getattr(candidate, name), getattr(reference, name))
        for value in ("wcounts", "wnorm"):
            np.testing.assert_allclose(
                getattr(a, value),
                getattr(b, value),
                rtol=design["reproduction_rtol"],
                atol=design["reproduction_atol"],
                equal_nan=False,
                err_msg=f"reference mismatch in {name}.{value}",
            )


def mesh_geometry(geometry, case, select_nmesh):
    """Keep physical geometry fixed, even when multigrid rounds mesh dimensions."""
    box = np.asarray(geometry["boxsize_mpc_h"], dtype=float)
    center = np.asarray(geometry["boxcenter_mpc_h"], dtype=float)
    if box.shape != (3,) or center.shape != (3,) or np.any(box <= 0):
        raise ValueError("invalid frozen mesh geometry")
    if not np.all(np.isfinite(box)) or not np.all(np.isfinite(center)):
        raise ValueError("nonfinite mesh geometry")
    if "nmesh" in case:
        nmesh = np.asarray(case["nmesh"], dtype=int)
        if not np.array_equal(select_nmesh(nmesh), nmesh):
            raise ValueError("fixed mesh is incompatible with multigrid")
    else:
        requested = np.ceil(box / case["maximum_cell_mpc_h"]).astype(int)
        requested += requested % 2
        nmesh = select_nmesh(requested)
        if np.any(nmesh < requested):
            raise ValueError("mesh selector coarsened the requested resolution")
    return {
        "nmesh": nmesh.tolist(),
        "boxsize_mpc_h": box.tolist(),
        "boxcenter_mpc_h": center.tolist(),
        "cellsize_mpc_h": (box / nmesh).tolist(),
    }


def weighted_cell_exposure(positions, weights, mask, *, box, center):
    """Nearest-cell exposure, explicitly distinct from TSC paint-volume fractions."""
    positions, weights = (np.asarray(positions), np.asarray(weights))
    box, center = (np.asarray(box), np.asarray(center))
    index = np.floor((positions - center + box / 2) / box * np.asarray(mask.shape)).astype(int)
    if np.any(index < 0) or np.any(index >= np.asarray(mask.shape)) or weights.sum() <= 0:
        raise ValueError("cell exposure needs contained positions and positive total weight")
    exposed = mask[tuple(index.T)]
    return {
        "nearest_cell_count_fraction": float(np.mean(exposed)),
        "nearest_cell_weight_fraction": float(np.sum(weights[exposed]) / np.sum(weights)),
    }


def instrument_reconstruction(solver, data, randoms, *, pipeline, **kwargs):
    import pyrecon

    original = pyrecon.IterativeFFTReconstruction
    captured = {}

    def build(*args, **options):
        recon = original(*args, **options)
        smooth_original = recon._smooth_gaussian

        def smooth(mesh):
            result = smooth_original(mesh)
            if mesh is recon.mesh_randoms:
                support = pipeline._mesh_array(mesh) > 0
                threshold = kwargs["threshold_randoms"] * result.csum() / len(randoms["Z"])
                mask = pipeline._mesh_array(result) <= threshold
                captured.update(
                    absolute_cutoff=float(threshold),
                    supported_cells=int(support.sum()),
                    zeroed_supported_cells=int((support & mask).sum()),
                    criterion="smoothed random weight <= native absolute cutoff",
                )
                for name, catalog in (("data", data), ("randoms", randoms)):
                    positions = np.column_stack([catalog[k] for k in ("X", "Y", "Z_CART")])
                    captured[name] = weighted_cell_exposure(
                        positions,
                        pipeline.reconstruction_weight(catalog, kwargs["weight_expression"]),
                        mask,
                        box=recon.boxsize,
                        center=recon.boxcenter,
                    )
            return result

        recon._smooth_gaussian = smooth
        return recon

    with patch.object(pyrecon, "IterativeFFTReconstruction", build):
        shifted_data, shifted_randoms, record = solver(data, randoms, **kwargs)
    if "absolute_cutoff" not in captured:
        raise ValueError("native smoothed-random cutoff was not observed")
    record["targeted_floor_diagnostics"] = captured
    return (shifted_data, shifted_randoms, record)


def fixed_mesh_reconstruction(data, randoms, *, nmesh, pipeline, **kwargs):
    """Same iterative FFT solver and physical box; only the mesh dimensions change."""
    from pyrecon import IterativeFFTReconstruction

    geometry = kwargs["common_geometry"]
    box = np.asarray(geometry["boxsize_mpc_h"], dtype=float)
    center = np.asarray(geometry["boxcenter_mpc_h"], dtype=float)
    mesh = np.asarray(nmesh, dtype=int)
    if mesh.shape != (3,) or np.any(mesh <= 0):
        raise ValueError("mesh must contain three positive dimensions")
    positions = [np.column_stack([c[k] for k in ("X", "Y", "Z_CART")]) for c in (data, randoms)]
    for values in positions:
        if np.any(~np.isfinite(values)) or np.any(np.abs(values - center) >= box / 2):
            raise ValueError("fixed-mesh inputs leave the frozen physical box")
    floor_gib = (
        float(np.prod(mesh)) * 8 * (kwargs["source_visible_real_mesh_count"] + 0.25)
        + sum((p.nbytes * 3 for p in positions))
    ) / 1024**3
    if floor_gib > kwargs["single_rank_peak_rss_ceiling_gib"]:
        raise MemoryError("fixed-mesh source-visible memory floor exceeds ceiling")
    weights = [
        pipeline.reconstruction_weight(c, kwargs["weight_expression"]) for c in (data, randoms)
    ]
    reconstruction = IterativeFFTReconstruction(
        f=kwargs["growth_rate"],
        bias=kwargs["bias"],
        los=None,
        positions=np.vstack(positions),
        nmesh=mesh,
        boxsize=box,
        boxcenter=center,
        resampler=kwargs["resampler"],
        fft_plan=kwargs["fft_plan"],
        dtype=kwargs["mesh_dtype"],
        threshold_randoms=kwargs["threshold_randoms"],
        wrap=False,
        position_type="pos",
    )
    if reconstruction.mpicomm.size != 1:
        raise ValueError("fixed-mesh diagnostic requires one MPI rank")
    np.testing.assert_array_equal(reconstruction.nmesh, mesh)
    np.testing.assert_allclose(reconstruction.boxsize, box, rtol=0, atol=1e-12)
    np.testing.assert_allclose(reconstruction.boxcenter, center, rtol=0, atol=1e-12)
    reconstruction.assign_data(positions[0], weights[0], position_type="pos")
    reconstruction.assign_randoms(positions[1], weights[1], position_type="pos")
    support = pipeline._mesh_array(reconstruction.mesh_randoms) > 0
    if np.count_nonzero(support) < 2:
        raise ValueError("very few randoms in the fixed-mesh diagnostic")
    masking = {}
    original_smooth = reconstruction._smooth_gaussian

    def smooth(mesh):
        result = original_smooth(mesh)
        if mesh is reconstruction.mesh_randoms:
            masking["absolute_cutoff"] = (
                kwargs["threshold_randoms"] * result.csum() / len(randoms["Z"])
            )
            masking["zeroed"] = pipeline._mesh_array(result) <= masking["absolute_cutoff"]
        return result

    with patch.object(reconstruction, "_smooth_gaussian", smooth):
        reconstruction.set_density_contrast(
            threshold_randoms=kwargs["threshold_randoms"],
            smoothing_radius=kwargs["smoothing_radius_mpc_h"],
            check=False,
            kw_weights=None,
        )
    if "zeroed" not in masking:
        raise ValueError("fixed-mesh native cutoff was not observed")
    zeroed = masking["zeroed"]
    exposure = {
        name: weighted_cell_exposure(p, w, zeroed, box=box, center=center)
        for name, p, w in zip(("data", "randoms"), positions, weights)
    }
    reconstruction.run(niterations=kwargs["iterations"])
    shifted = [
        np.asarray(reconstruction.read_shifted_positions(p, field="disp+rsd")) for p in positions
    ]
    result = []
    for catalog, before, after in zip((data, randoms), positions, shifted):
        if (
            after.shape != before.shape
            or not np.all(np.isfinite(after))
            or (not np.any(after != before))
        ):
            raise ValueError("fixed-mesh displacement is invalid or null")
        result.append({**catalog, **dict(zip(pipeline.SHIFTED_COLUMNS, after.T))})
    shells = pipeline.reconstruction_shell_diagnostics(
        positions[0],
        shifted[0],
        positions[1],
        random_support=support,
        threshold_zeroed=zeroed,
        cellsize=box / mesh,
        boxcenter=center,
        boxsize=box,
        smoothing_radius_mpc_h=kwargs["smoothing_radius_mpc_h"],
        config=kwargs["config"],
    )
    return (
        *result,
        {
            "algorithm": "iterative_fft",
            "convention": "recsym",
            "bias": kwargs["bias"],
            "growth_rate": kwargs["growth_rate"],
            "smoothing_radius_mpc_h": kwargs["smoothing_radius_mpc_h"],
            "iterations": kwargs["iterations"],
            "threshold_randoms": kwargs["threshold_randoms"],
            "realized_nmesh": mesh.tolist(),
            "realized_boxsize_mpc_h": box.tolist(),
            "realized_boxcenter_mpc_h": center.tolist(),
            "realized_cellsize_mpc_h": (box / mesh).tolist(),
            "supported_cells": int(support.sum()),
            "zeroed_supported_cells": int((support & zeroed).sum()),
            "weighted_cell_exposure": exposure,
            "shell_diagnostics": shells,
            "absolute_cutoff": float(masking["absolute_cutoff"]),
            "pyrecon_internal_check": False,
            "source_visible_memory_floor_gib": floor_gib,
        },
    )


def factorial(values):
    """Smoothing and floor main effects averaged over the other setting; interaction."""
    baseline, smoothing, floor, combined = values
    return np.asarray(
        [
            (smoothing - baseline + combined - floor) / 2,
            (floor - baseline + combined - smoothing) / 2,
            combined - smoothing - floor + baseline,
        ]
    )


def verify_pairs(records, design):
    mocks, cases = (design["mock_ids"], design["cases"])
    if set(records) != {(mock, case["name"]) for mock in mocks for case in cases}:
        raise ValueError("completion records differ from the declared pilot units")
    for mock in mocks:
        base = records[mock, "baseline"]
        for case in cases:
            record = records[mock, case["name"]]
            if record["mock"] != mock or record["case"] != case:
                raise ValueError(
                    f"completion record identity differs for mock {mock}, {case['name']}"
                )
            for key in ("experiment", "source_commit", "source_snapshot_sha256"):
                if record[key] != design[key]:
                    raise ValueError(f"completion record differs from design: {key}")
            if record["experimental_only"] is not True:
                raise ValueError("pilot completion record must be exploratory")
            if record["fingerprints"] != base["fingerprints"]:
                raise ValueError(f"unpaired input catalogue for mock {mock}, {case['name']}")
            for key in (
                "driver_sha256",
                "design_sha256",
                "source_commit",
                "source_snapshot_sha256",
            ):
                if record[key] != base[key]:
                    raise ValueError(f"mixed provenance: {key}")
    for record in records.values():
        for key in ("driver_sha256", "design_sha256", "source_commit", "source_snapshot_sha256"):
            if record[key] != next(iter(records.values()))[key]:
                raise ValueError("different experiment versions across mocks")


_photoz_theory_ArrayLike = Union[float, Sequence[float], np.ndarray]


def photoz_theory_finite_array(value: _photoz_theory_ArrayLike, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if np.any(~np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def pre_to_post_radial_damping_ratio(
    nonlinear_pre_mpc_h: _photoz_theory_ArrayLike,
    nonlinear_post_mpc_h: _photoz_theory_ArrayLike,
    sigma_parallel_photoz: _photoz_theory_ArrayLike,
) -> np.ndarray:
    """Predict the pre/post damping-scale ratio when reconstruction changes only gravity.

    This ratio of physical damping lengths is not the post/pre uncertainty FoM gain.

    As photo-z smearing dominates, the pre/post ratio tends to one even if the
    nonlinear pre- and post-reconstruction damping scales differ.  Inputs
    deliberately follow NumPy broadcasting so a diagnostic caller can evaluate
    compatible parameter grids.
    """
    pre = photoz_theory_finite_array(nonlinear_pre_mpc_h, "nonlinear_pre_mpc_h")
    post = photoz_theory_finite_array(nonlinear_post_mpc_h, "nonlinear_post_mpc_h")
    photoz = photoz_theory_finite_array(sigma_parallel_photoz, "sigma_parallel_photoz")
    if np.any(pre <= 0) or np.any(post <= 0) or np.any(photoz < 0):
        raise ValueError("radial damping-ratio inputs are invalid")
    return np.hypot(pre, photoz) / np.hypot(post, photoz)


def photoz_power_damping(
    k_h_mpc: _photoz_theory_ArrayLike,
    mu: _photoz_theory_ArrayLike,
    sigma_r: _photoz_theory_ArrayLike,
) -> np.ndarray:
    """Return the Gaussian power-level window ``exp[-(k mu sigma_r)^2]``.

    ``sigma_r`` is the one-object value from
    :func:`pzbao.theory.sigma_r_mpc_h`, not its pairwise
    square-root-of-two counterpart.  The window is unity for transverse modes
    or exact redshifts, and inputs deliberately support NumPy broadcasting.
    """
    k = photoz_theory_finite_array(k_h_mpc, "k_h_mpc")
    angle = photoz_theory_finite_array(mu, "mu")
    radial = photoz_theory_finite_array(sigma_r, "sigma_r")
    if np.any(k < 0) or np.any(np.abs(angle) > 1) or np.any(radial < 0):
        raise ValueError("damping-window inputs are invalid")
    return np.exp(-((k * angle * radial) ** 2))


def displacement_summary(catalogue):
    """Describe saved shifts without replacing a reconstruction solve."""
    from ..catalog import validate_catalog
    from ..reconstruction import SHIFTED_COLUMNS

    validate_catalog(catalogue, required=("SOURCE_ID", "X", "Y", "Z_CART", *SHIFTED_COLUMNS))
    original = np.column_stack([catalogue[key] for key in ("X", "Y", "Z_CART")])
    shifted = np.column_stack([catalogue[key] for key in SHIFTED_COLUMNS])
    displacement = shifted - original
    if not np.all(np.isfinite(displacement)):
        raise ValueError("non-finite saved displacement")
    return {
        "source_ids": catalogue["SOURCE_ID"],
        "displacement_mpc_h": displacement,
        "mean_mpc_h": displacement.mean(axis=0),
        "rms_mpc_h": np.sqrt(np.mean(displacement**2, axis=0)),
    }


def plot_displacements(catalogue, *, bins=40, ax=None):
    from ..figures.style import axes

    ax = axes(ax)
    summary = displacement_summary(catalogue)
    for index, label in enumerate(("x", "y", "z")):
        counts, edges = np.histogram(summary["displacement_mpc_h"][:, index], bins=bins)
        ax.stairs(counts, edges, label=label)
    ax.set(xlabel="Shift [Mpc/h]", ylabel="Objects")
    return ax, summary


def subset(catalog, name, role, config, pipeline=None):
    if pipeline is None:
        from .. import selection as pipeline
    column = (
        config["selection"]["stable_randomness"]["observed_redshift_column"]
        if role == "data"
        else "Z"
    )
    result = pipeline.stable_measurement_subset(
        catalog, tracer="LRG", bin_name=name, redshift_column=column, config=config
    )
    if not len(result[column]):
        raise ValueError("empty reconstruction bin")
    return result


def reconstruct_with_mesh(
    data, randoms, *, design, case, geometry, settings, config, pipeline=None
):
    if pipeline is None:
        from .. import reconstruction as pipeline
    from pyrecon import MultiGridReconstruction

    realized = mesh_geometry(geometry, case, MultiGridReconstruction._select_nmesh)
    nmesh = np.asarray(realized["nmesh"])
    box, center = [np.asarray(realized[k]) for k in ("boxsize_mpc_h", "boxcenter_mpc_h")]
    screen = float(np.prod(nmesh)) * 8 * design["resources"]["screen_real_mesh_count"] / 1024**3
    if screen > design["resources"]["mesh_memory_screen_gib"]:
        raise MemoryError("mesh storage screen exceeded; review resources before allocation")
    positions = [np.column_stack([c[k] for k in ("X", "Y", "Z_CART")]) for c in (data, randoms)]
    weights = [
        pipeline.reconstruction_weight(c, settings["weight_expression"]) for c in (data, randoms)
    ]
    for pos in positions:
        if (
            not np.all(np.isfinite(pos))
            or np.any(pos < center - box / 2)
            or np.any(pos >= center + box / 2)
        ):
            raise ValueError("bin catalogue outside the frozen physical box")
    recon = MultiGridReconstruction(
        f=settings["growth_rate"],
        bias=settings["bias"],
        los=None,
        nmesh=nmesh,
        boxsize=box,
        boxcenter=center,
        resampler=settings["resampler"],
        fft_plan=settings["fft_plan"],
        dtype=settings["mesh_dtype"],
        wrap=False,
        position_type="pos",
        threshold_randoms=design["threshold_randoms"],
    )
    if recon.mpicomm.size != 1 or recon.mpicomm.rank != 0:
        raise ValueError("this experiment requires one MPI rank")
    for actual, wanted in ((recon.nmesh, nmesh), (recon.boxsize, box), (recon.boxcenter, center)):
        np.testing.assert_array_equal(actual, wanted)
    recon.assign_data(positions[0], weights[0])
    recon.assign_randoms(positions[1], weights[1])
    support = pipeline._mesh_array(recon.mesh_randoms) > 0
    if np.count_nonzero(support) < 2:
        raise ValueError("insufficient random support")
    masking = {}
    original_smooth = recon._smooth_gaussian

    def smooth(mesh):
        result = original_smooth(mesh)
        if mesh is recon.mesh_randoms:
            masking["absolute_floor"] = (
                design["threshold_randoms"] * result.csum() / len(weights[1])
            )
            masking["zeroed"] = pipeline._mesh_array(result) <= masking["absolute_floor"]
        return result

    with patch.object(recon, "_smooth_gaussian", smooth):
        recon.set_density_contrast(smoothing_radius=design["smoothing_mpc_h"], check=False)
    if "zeroed" not in masking:
        raise ValueError("failed to capture the actual smoothed-random cutoff")
    absolute_floor = float(masking["absolute_floor"])
    recon.run(**design["solver"])
    shifted = [np.asarray(recon.read_shifted_positions(pos, field="disp+rsd")) for pos in positions]
    for new, old in zip(shifted, positions):
        if new.shape != old.shape or not np.all(np.isfinite(new)) or not np.any(new != old):
            raise ValueError("invalid reconstructed positions")
    shell = pipeline.reconstruction_shell_diagnostics(
        positions[0],
        shifted[0],
        positions[1],
        random_support=support,
        threshold_zeroed=masking["zeroed"],
        cellsize=box / nmesh,
        boxcenter=center,
        boxsize=box,
        smoothing_radius_mpc_h=design["smoothing_mpc_h"],
        config=config,
    )
    outputs = [dict(data), dict(randoms)]
    for output, new in zip(outputs, shifted):
        for axis, name in enumerate(pipeline.SHIFTED_COLUMNS):
            output[name] = new[:, axis]
    record = {
        "algorithm": "MultiGridReconstruction",
        "geometry": realized,
        "solver": design["solver"],
        "smoothing_mpc_h": design["smoothing_mpc_h"],
        "threshold_randoms": design["threshold_randoms"],
        "absolute_floor": absolute_floor,
        "random_density_cutoff": absolute_floor / float(np.prod(box / nmesh)),
        "mesh_storage_screen_gib": screen,
        "shell_diagnostics": shell,
        "supported_cells": int(support.sum()),
        "zeroed_supported_cells": int(np.count_nonzero(support & masking["zeroed"])),
        "bias": settings["bias"],
        "growth_rate": settings["growth_rate"],
        "convention": "recsym",
        "resampler": settings["resampler"],
    }
    return outputs[0], outputs[1], record


DISPLACEMENT_AXES = ("Delta X", "Delta Y", "Delta Z", "Delta radial", "Transverse magnitude")


def read_shifts(path, zmin, zmax):
    """Post minus pre, preserving observed-bin membership and source identity."""
    with np.load(path, allow_pickle=False) as archive:
        ids, redshift, before, after = (
            archive[k].copy() for k in ("source_ids", "redshift", "positions", "shifted_positions")
        )
    if (
        ids.ndim != 1
        or ids.dtype.kind not in "iu"
        or len(np.unique(ids)) != len(ids)
        or redshift.shape != ids.shape
        or before.shape != (len(ids), 3)
        or after.shape != before.shape
        or not np.isfinite(before).all()
        or not np.isfinite(after).all()
        or not np.isfinite(redshift).all()
    ):
        raise ValueError("invalid saved coordinate identities, shapes or values")
    keep = (redshift >= zmin) & (redshift < zmax)
    if not keep.any():
        raise ValueError("empty observed-redshift bin")
    order = np.argsort(ids[keep])
    ids, redshift, before, after = [a[keep][order] for a in (ids, redshift, before, after)]
    radius = np.linalg.norm(before, axis=1)
    if np.any(radius <= 0):
        raise ValueError("local line of sight is undefined")
    delta = after - before
    radial = np.einsum("ij,ij->i", delta, before / radius[:, None])
    transverse = np.linalg.norm(delta - radial[:, None] * before / radius[:, None], axis=1)
    identity = hashlib.sha256()
    for values in (ids, redshift, before):
        identity.update(np.ascontiguousarray(values).tobytes())
    return np.column_stack((delta, radial, transverse)), identity.hexdigest()


def moments(values):
    if (
        values.ndim != 2
        or values.shape[1] != len(DISPLACEMENT_AXES)
        or not len(values)
        or not np.isfinite(values).all()
    ):
        raise ValueError("invalid displacement sample")
    return {
        "count": len(values),
        "mean": values.mean(axis=0).tolist(),
        "sd_ddof0": values.std(axis=0).tolist(),
        "rms": np.sqrt(np.mean(values**2, axis=0)).tolist(),
        "quantile_probabilities": [0.001, 0.01, 0.16, 0.5, 0.84, 0.99, 0.999],
        "quantiles": np.quantile(
            values, [0.001, 0.01, 0.16, 0.5, 0.84, 0.99, 0.999], axis=0
        ).tolist(),
        "minimum": values.min(axis=0).tolist(),
        "maximum": values.max(axis=0).tolist(),
    }


def histogram(values, edges):
    counts = np.histogram(values, bins=edges)[0]
    underflow, overflow = int((values < edges[0]).sum()), int((values > edges[-1]).sum())
    if counts.sum() + underflow + overflow != len(values):
        raise ValueError("histogram lost coordinate shifts")
    return {
        "counts": counts.tolist(),
        "density": (counts / (len(values) * np.diff(edges))).tolist(),
        "underflow": underflow,
        "overflow": overflow,
    }


def equal_mock_distribution(rows, component):
    if not rows:
        raise ValueError("no complete mocks")
    density = np.mean([r["histograms"][component]["density"] for r in rows], axis=0)
    means = np.asarray([r["moments"]["mean"][component] for r in rows])
    variance = np.asarray([r["moments"]["sd_ddof0"][component] ** 2 for r in rows])
    mean = float(means.mean())
    return density, mean, float(np.sqrt(max(0.0, np.mean(variance + means**2) - mean**2)))


def gaussian(grid, mean, sd):
    if sd <= 0:
        return None
    return np.exp(-0.5 * ((grid - mean) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
