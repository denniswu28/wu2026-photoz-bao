"""Select analysis-ready catalogues and freeze complete-ensemble selection targets.

Transformations return new arrays, retaining object identities, weights and random streams.
The selection stage saves catalogues for independent reconstruction and clustering.
"""

from __future__ import annotations

import hashlib
import json
import time
import copy
import math
from copy import deepcopy
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from .catalog import (
    Catalog,
    _map_row_chunks,
    add_cartesian_positions,
    build_mock_cosmology,
    copy_catalog,
    final_sample_identity,
    filter_redshift,
    measurement_bin,
    measurement_bins,
    validate_catalog,
    validate_boundary_input_sources,
    footprint_area_deg2,
    comoving_shell_volume,
    reconstruction_field,
    reconstruction_weight,
    measurement_weight,
    ra_dec_z_to_xyz,
    size_box_from_extent,
)
from .catalog import (
    _read_parent,
    catalogue_identity,
    save_catalogue,
    read_catalogue,
    read_catalog_registration,
)
from .config import (
    PER_BIN_SCHEMA_VERSIONS,
    load_analysis_config,
    same_identity,
    per_bin_reconstruction,
)
from .reconstruction import (
    exact_growth_rate,
    weighted_random_zeff_accumulator,
    reconstruction_mask_family,
    reconstruction_window_bounds,
)

# Stable object-bound randomness


def _stable_uniform(
    source_ids: Sequence[int], *, seed: int, namespace: str, parallel=None
) -> np.ndarray:
    """Map object identities to reproducible independent positive values.

    Adding one half before division keeps every value above zero, so Box--Muller
    never sees ``log(0)``. Float64 rounding can map the largest digests to exactly
    one; inverse-CDF consumers therefore clip to ``nextafter(1.0, 0.0)`` before
    indexing. That consumer-side clip, not this helper alone, excludes the upper
    endpoint. The scalar per-row BLAKE2b route is linear in catalogue size and
    intentionally preserves exact object-bound digest semantics; callers must
    budget that cost rather than substitute a different vectorized random stream.
    The UTF-8 token layout ``source_id|seed|namespace``, little-endian digest
    conversion, and the in-file namespace strings are frozen result-affecting
    contracts: changing any of them rerolls scientific sample membership.
    """
    return _map_row_chunks(
        _stable_uniform_values,
        {"source_ids": np.asarray(source_ids, dtype=np.uint64)},
        dtype=float,
        parallel=parallel,
        seed=seed,
        namespace=namespace,
    )


def _stable_uniform_values(columns, *, seed, namespace):
    values = columns["source_ids"]
    scale = float(2**64)

    def uniform_value(source_id: np.uint64) -> float:
        token = f"{int(source_id)}|{int(seed)}|{namespace}".encode()
        integer = int.from_bytes(hashlib.blake2b(token, digest_size=8).digest(), "little")
        return (integer + 0.5) / scale

    return np.fromiter(
        (uniform_value(source_id) for source_id in values), dtype=float, count=len(values)
    )


def attach_stable_randomness(
    catalog: Mapping[str, Any],
    *,
    role: str,
    config: Optional[Mapping[str, Any]] = None,
    parallel=None,
) -> Catalog:
    """Attach only the object-bound random streams consumed by this role.

    Namespaces make the photo-z and selection streams independent while one
    shared seed keeps their pairing fixed. The separately configured radial
    seed lets random-row thinning ranks change without re-rolling galaxy
    smearing or singular primary-density selection. Rows of data provenance, including
    reservoir rows passed with ``role='data'``, receive the photo-z deviate and
    selection rank; random rows receive only their radial thinning rank. The photo-z
    deviate uses the standard-normal Box--Muller cosine branch; the paired sine
    branch is deliberately unused. Selection ranks and radial thinning ranks are
    deterministic hash-uniform float values in ``(0, 1]``.
    """
    return _attach_stable_randomness_to_owned(
        copy_catalog(catalog), role=role, config=config, parallel=parallel
    )


def _attach_stable_randomness_to_owned(result, *, role, config=None, parallel=None):
    """Attach streams to privately owned arrays; callers must not pass shared inputs."""
    if role not in {"data", "random"}:
        raise ValueError("stable-randomness role must be data or random")
    resolved = config or load_analysis_config()
    controls = resolved["selection"]["stable_randomness"]
    identity = str(controls["identity_column"])
    validate_catalog(result, (identity,))
    source_ids = np.asarray(result[identity], dtype=np.uint64)
    seed = int(controls["seed"])
    if role == "data":
        # _stable_uniform is strictly positive, so log(u1) is always finite. Its
        # possible upper endpoint is harmless here; inverse-CDF consumers clip it.
        u1 = _stable_uniform(source_ids, seed=seed, namespace="photoz-u1", parallel=parallel)
        u2 = _stable_uniform(source_ids, seed=seed, namespace="photoz-u2", parallel=parallel)
        result[str(controls["photoz_deviate_column"])] = np.sqrt(-2.0 * np.log(u1)) * np.cos(
            2.0 * np.pi * u2
        )
        result[str(controls["selection_rank_column"])] = _stable_uniform(
            source_ids, seed=seed, namespace="selection", parallel=parallel
        )
    else:
        result[str(controls["random_thinning_rank_column"])] = _stable_uniform(
            source_ids,
            seed=int(resolved["selection"]["radial_randoms"]["stable_thinning_seed"]),
            namespace="radial-thinning",
            parallel=parallel,
        )
    return result


# Weight normalization and same-tracer boundary support


class _InputPreparationCache:
    """Reuse invariant transforms within one immutable mock/cap input pass only.

    Own the normalized arrays, make them read-only, and discard this cache before
    reading another cap. No cached scientific quantity crosses a run or file boundary.
    """

    def __init__(self, config, *, parallel=None):
        self.selection = self._settings(copy.deepcopy(config["selection"]))
        self.parallel = parallel
        self.catalogues = {}
        self.orders = {}

    @staticmethod
    def _settings(selection):
        # Tier target densities only set counts, after these transforms. The matched-density
        # run clears its matched tier's target on a configuration copy (revision 2.15) and must
        # reuse the pass's arrays; every other selection setting still has to agree. A shallow
        # view: the density run's selection embeds its whole handoff.
        tiers = [
            {key: value for key, value in tier.items() if key != "target_nbar_h3_mpc3"}
            for tier in selection["tiers"]
        ]
        return {**selection, "tiers": tiers}

    def normalized(self, catalog, *, role, config):
        if self._settings(config["selection"]) != self.selection:
            raise ValueError("input cache belongs to different selection settings")
        key = (id(catalog), role)
        if key not in self.catalogues:
            # normalize_weights already owns a deep copy. Reuse it rather than
            # hold a second full copy while generating object-bound streams.
            prepared = _attach_stable_randomness_to_owned(
                normalize_weights(catalog), role=role, config=config, parallel=self.parallel
            )
            for values in prepared.values():
                values.flags.writeable = False
            # Keep the source reference alive so its identity cannot be recycled.
            self.catalogues[key] = (catalog, prepared)
        return self.catalogues[key][1]

    def random_order(self, randoms, edges, rank_column):
        if not any(randoms is pair[1] for pair in self.catalogues.values()):
            raise ValueError("random ordering requires this input cache's immutable catalogue")
        key = (id(randoms), rank_column, tuple(edges))
        if key not in self.orders:
            redshift = np.asarray(randoms["Z"], dtype=float)
            # Zero and len(edges) label rows outside the half-open support. Integer
            # grouping is stable, preserving original row order when ranks tie.
            labels = np.searchsorted(edges, redshift, side="right").astype(
                np.min_scalar_type(len(edges))
            )
            labels[redshift == edges[-1]] = 0
            grouped = np.argsort(labels, kind="stable")
            offsets = np.concatenate(
                ([0], np.cumsum(np.bincount(labels, minlength=len(edges) + 1)))
            )
            del labels
            ranks = np.asarray(randoms[rank_column], dtype=float)
            for index in range(1, len(edges)):
                start, stop = offsets[index : index + 2]
                members = grouped[start:stop]
                grouped[start:stop] = members[np.argsort(ranks[members], kind="stable")]
            self.orders[key] = (grouped, offsets)
        return self.orders[key]


def _normalized_randomness(catalog, *, role, config, parallel, input_cache):
    if input_cache is not None:
        return input_cache.normalized(catalog, role=role, config=config)
    return attach_stable_randomness(
        normalize_weights(catalog), role=role, config=config, parallel=parallel
    )


def normalize_weights(catalog: Mapping[str, Any]) -> Catalog:
    """Drop the seven named DESI correction weights and set ``WEIGHT=1``.

    Any upstream ``WEIGHT_FKP`` remains temporarily present but is never used;
    :func:`apply_attained_fkp` replaces it after exact density selection.
    """
    result = copy_catalog(catalog)
    for name in (
        "WEIGHT_IMLIN",
        "WEIGHT_SYSTOT",
        "WEIGHT_ZFAIL",
        "WEIGHT_NOFIB",
        "WEIGHT_TILELOCID",
        "WEIGHT_SYS",
        "WEIGHT_COMP",
    ):
        result.pop(name, None)
    result["WEIGHT"] = np.ones(validate_catalog(result), dtype=float)
    return result


def merge_same_tracer_reservoir(
    data: Mapping[str, Any],
    reservoir: Mapping[str, Any],
    *,
    tracer: str,
    expected_sample_identity: str,
    reservoir_record: Mapping[str, Any],
    sigma_z_over_one_plus_z: float,
    config: Mapping[str, Any],
) -> Tuple[Catalog, dict]:
    """Keep complete clustering and add distinct exterior galaxies before smearing.

    Inputs are cap-selected and weight-normalized, with their unchanged role-specific
    stable identities and random streams already attached. This merge requires upstream
    reservoir eligibility and footprint consistency. Exact native coordinates identify cross-source
    duplicates independently of source-local TARGETID numbering; the clustering copy
    always wins. Duplicate physical rows within one input are invalid, not reidentified.

    All configured edges contribute available exterior support at every condition.
    The minimum/N-sigma depth is diagnostic only: insufficient depth never removes
    galaxies. Inner reservoir bands are comparison samples and are never injected.
    The generic merge preserves every supplied data column and ignores reservoir-only
    metadata. The observed-redshift field cut remains after photo-z smearing.
    """
    if reservoir_record.get("tracer") != tracer:
        raise ValueError("cross-tracer reservoir injection is forbidden")
    if reservoir_record.get("sample_identity") != str(expected_sample_identity):
        raise ValueError("reservoir equivalence belongs to another analysis sample")
    identity = str(config["selection"]["stable_randomness"]["identity_column"])
    coordinate_columns = config["selection"]["boundary_support"]["physical_identity_columns"]
    required = {"TARGETID", identity, *coordinate_columns}
    for catalog in (data, reservoir):
        validate_catalog(catalog, required)
        if "IS_RESERVOIR" in catalog:
            raise ValueError("IS_RESERVOIR is reserved for the merged provenance flag")
        if any(np.any(~np.isfinite(catalog[name])) for name in coordinate_columns):
            raise ValueError("reservoir and clustering native coordinates must be finite")
    missing = sorted(set(data) - set(reservoir))
    if missing:
        raise KeyError(f"reservoir extractor did not preserve data columns {missing}")

    controls = config["tracers"][tracer]["reservoir"]
    field = reconstruction_field(tracer, config)
    expected_boundary = {"below": float(field["zmin"]), "above": float(field["zmax"])}
    band = float(controls["calibration_band_width_redshift"])
    sigma_z = float(sigma_z_over_one_plus_z)
    guard_n_sigma = float(controls["guard_n_sigma"])
    minimum_guard = float(controls["minimum_guard_width_redshift"])
    if not np.isfinite(band) or band <= 0:
        raise ValueError("reservoir calibration-band width must be finite and positive")
    if not np.isfinite(sigma_z) or sigma_z < 0:
        raise ValueError("reservoir photo-z uncertainty must be finite and non-negative")
    if not all(np.isfinite(value) and value > 0 for value in (guard_n_sigma, minimum_guard)):
        raise ValueError("reservoir guard controls must be finite and positive")

    redshift = np.asarray(reservoir["Z"], dtype=float)
    data_redshift = np.asarray(data["Z"], dtype=float)
    outside = np.zeros(len(redshift), dtype=bool)
    observed_sides = []
    for edge in controls["edges"]:
        side, boundary = str(edge["side"]), float(edge["redshift"])
        if side not in expected_boundary:
            raise ValueError("reservoir edge side is invalid")
        if not np.isclose(boundary, expected_boundary[side], rtol=0.0, atol=1e-12):
            raise ValueError("reservoir edge does not coincide with its tracer-field boundary")
        if side in observed_sides:
            raise ValueError("reservoir declares a duplicate tracer-field boundary side")
        observed_sides.append(side)
        if side == "below":
            outside |= redshift < boundary
            inside = (data_redshift >= boundary) & (data_redshift < boundary + band)
        else:
            outside |= redshift >= boundary
            inside = (data_redshift >= boundary - band) & (data_redshift < boundary)
        if not np.any(inside):
            raise ValueError("clustering data do not reach every inside reservoir-edge band")
    reservoir = _slice(reservoir, outside)
    candidates = validate_catalog(reservoir)
    coordinates = [
        np.rec.fromarrays([np.asarray(catalog[name], dtype=float) for name in coordinate_columns])
        for catalog in (data, reservoir)
    ]
    for role, values in zip(("clustering", "exterior reservoir"), coordinates):
        if len(np.unique(values)) != len(values):
            raise ValueError(f"{role} contains duplicate physical galaxies")
    _, _, duplicate_indices = np.intersect1d(*coordinates, assume_unique=True, return_indices=True)
    keep = np.ones(candidates, dtype=bool)
    keep[duplicate_indices] = False
    reservoir = _slice(reservoir, keep)
    merged = {
        name: np.concatenate([np.asarray(data[name]), np.asarray(reservoir[name])]) for name in data
    }
    if len(np.unique(merged[identity])) != len(merged[identity]):
        raise ValueError("data and reservoir stable identities overlap")
    merged["IS_RESERVOIR"] = np.concatenate(
        [np.zeros(len(data_redshift), dtype=bool), np.ones(validate_catalog(reservoir), dtype=bool)]
    )
    if "IS_BOUNDARY_EXTENSION" in reservoir:
        merged["IS_BOUNDARY_EXTENSION"] = np.concatenate(
            [np.zeros(len(data_redshift), dtype=bool), reservoir["IS_BOUNDARY_EXTENSION"]]
        )
    edge_support = []
    for edge in controls["edges"]:
        side, boundary = str(edge["side"]), float(edge["redshift"])
        guard = max(minimum_guard, guard_n_sigma * sigma_z * (1.0 + boundary))
        depths = []
        for values in (data_redshift, np.asarray(reservoir["Z"], dtype=float)):
            depths.append(
                boundary - values[values < boundary]
                if side == "below"
                else values[values >= boundary] - boundary
            )
        available_depth = max((float(values.max()) if len(values) else 0.0) for values in depths)
        edge_support.append(
            {
                "side": side,
                "redshift": boundary,
                "clustering_row_count": len(depths[0]),
                "added_reservoir_row_count": len(depths[1]),
                "required_guard_width_redshift": guard,
                "available_depth_redshift": available_depth,
                "shortfall_redshift": max(0.0, guard - available_depth),
                "guard_reached": bool(available_depth >= guard),
            }
        )
    return merged, {
        "reservoir_candidate_row_count": candidates,
        "overlap_removed_row_count": len(duplicate_indices),
        "edge_support": edge_support,
    }


# Data-only photo-z smearing, tier lookup, and shared row-slicing


def smear_data_redshifts(
    catalog: Mapping[str, Any],
    sigma_z_over_one_plus_z: float,
    *,
    config: Optional[Mapping[str, Any]] = None,
    input_redshift: str = "Z",
) -> Catalog:
    """Apply deterministic Gaussian smearing to data only.

    Only data/reservoir rows have the required photo-z deviate column; randoms
    therefore fail validation before smearing. That is enforcement by absence, so
    it is backed here by a positive check: random rows are the only rows that
    carry the radial thinning rank, and a catalogue carrying it is refused even
    if it also carries a deviate column. The selected ``input_redshift``
    must represent the incoming clustering redshift after peculiar-velocity
    redshift-space distortions and before observational photo-z noise. It is
    retained as ``Z_RSD`` and the observed value is written to configured ``Z_OBS``;
    the incoming ``Z`` remains unchanged.
    Real-space ``TRUEZ`` may coexist for provenance, but it is never used by
    this function unless a caller explicitly and incorrectly names it; the
    production caller always supplies the active RSD-bearing ``Z``. ``Z_RSD`` and
    ``Z_OBS`` are reserved: their presence means this module already
    smeared the catalogue or an upstream input used the reserved name. The configured
    catastrophic-outlier fraction must remain zero because this implementation
    contains no configured outlier model.
    """
    resolved = config or load_analysis_config()
    if float(sigma_z_over_one_plus_z) < 0:
        raise ValueError("photo-z scale must be non-negative")
    if float(resolved["selection"]["catastrophic_outlier_fraction"]) != 0.0:
        raise ValueError("catastrophic outliers are not a configured analysis component")
    controls = resolved["selection"]["stable_randomness"]
    deviate_column = str(controls["photoz_deviate_column"])
    source_column = str(controls["source_redshift_column"])
    observed_column = str(controls["observed_redshift_column"])
    if str(controls["random_thinning_rank_column"]) in catalog:
        raise ValueError(
            "randoms are never photo-z smeared: this catalogue carries the random "
            "thinning rank column"
        )
    validate_catalog(catalog, (input_redshift, deviate_column))
    if source_column in catalog or observed_column in catalog:
        raise ValueError(
            "source or observed redshift already exists: catalogue was already smeared "
            "or uses a reserved input name"
        )
    result = copy_catalog(catalog)
    rsd_redshift = np.asarray(catalog[input_redshift], dtype=float)
    result[source_column] = rsd_redshift.copy()
    result[observed_column] = rsd_redshift + float(sigma_z_over_one_plus_z) * (
        1.0 + rsd_redshift
    ) * np.asarray(catalog[deviate_column], dtype=float)
    return result


def joint_class_budget(desired_counts, available_counts_by_mock):
    """Allocate exclusive parent classes jointly; arrays are (class,shell)/(mock,shell).

    These are parent counts, before photo-z migration. The common minimum is taken
    over mocks, never over independently smeared copies. Rounding remainders stay unused.
    """
    desired = np.asarray(desired_counts, dtype=float)
    available = np.asarray(available_counts_by_mock)
    if (
        desired.ndim != 2
        or available.ndim != 2
        or 0 in desired.shape
        or 0 in available.shape
        or desired.shape[1] != available.shape[1]
        or not np.isfinite(desired).all()
        or not np.isfinite(available).all()
        or np.any(desired < 0)
        or np.any(available < 0)
        or np.any(available != np.floor(available))
    ):
        raise ValueError(
            "joint allocation requires nonnegative class counts and integer availability"
        )
    total = desired.sum(axis=0)
    if not np.isfinite(total).all() or np.any(available >= np.iinfo(np.int64).max):
        raise ValueError("joint allocation count overflow")
    minimum = available.min(axis=0).astype(np.int64)
    scale = np.divide(minimum, total, out=np.ones_like(total), where=total > 0)
    scale = np.minimum(1.0, scale)
    allocated = np.floor(desired * scale).astype(np.int64)
    if np.any(allocated.sum(axis=0) > minimum):
        raise ValueError("rounded joint allocation exceeds the parent population")
    return {
        "forecast_parent_counts": desired.tolist(),
        "minimum_available_parent_counts": minimum.tolist(),
        "scale_by_shell": scale.tolist(),
        "allocated_parent_counts": allocated.tolist(),
        "unused_parent_counts": (minimum - allocated.sum(axis=0)).tolist(),
    }


def assign_exclusive_classes(parent, *, class_labels, shell_edges, allocated_counts, seed, config):
    """Partition one eligible, unsmeared tracer/cap parent without replacement.

    Upstream must merge eligible exterior support using clustering precedence first.
    Sorted object-keyed hashes make interior quotas exact and row-order independent.
    Exterior rows use the adjacent shell's class probabilities, with no exact final-count
    retargeting. A separate hash namespace keeps assignments independent of photo-z draws.
    Returns integer class indices (-1 means unused) and diagnostics; inputs are unchanged.
    """
    controls = config["selection"]["stable_randomness"]
    identity = controls["identity_column"]
    validate_catalog(parent, (identity, "RA", "DEC", "Z"))
    if any(
        name in parent
        for name in (controls["observed_redshift_column"], controls["source_redshift_column"])
    ):
        raise ValueError("exclusive assignment must precede photo-z smearing")
    ids = np.asarray(parent[identity])
    z = np.asarray(parent["Z"], dtype=float)
    coordinates = np.column_stack([parent[name] for name in ("RA", "DEC", "Z")])
    if not np.isfinite(coordinates).all() or np.any(z <= -1):
        raise ValueError("nonfinite or unphysical eligible parent coordinates")
    if len(np.unique(ids)) != len(ids) or len(np.unique(coordinates, axis=0)) != len(ids):
        raise ValueError("duplicate parent identity or physical row before class assignment")
    edges = np.asarray(shell_edges, dtype=float)
    counts = np.asarray(allocated_counts)
    if (
        edges.ndim != 1
        or len(edges) < 2
        or not np.isfinite(edges).all()
        or np.any(np.diff(edges) <= 0)
        or not class_labels
        or len(set(class_labels)) != len(class_labels)
        or counts.shape != (len(class_labels), len(edges) - 1)
        or not np.isfinite(counts).all()
        or np.any(counts < 0)
        or np.any(counts != np.floor(counts))
        or np.any(counts >= np.iinfo(np.int64).max)
    ):
        raise ValueError("invalid exclusive classes, shells or integer allocation")
    counts = counts.astype(np.int64)
    shells = np.searchsorted(edges, z, side="right") - 1
    ranks = _stable_uniform(ids, seed=seed, namespace="exclusive-class-v1")
    assignment = np.full(len(ids), -1, dtype=np.int64)
    probabilities = np.zeros_like(counts, dtype=float)
    available = []
    for shell in range(len(edges) - 1):
        members = np.flatnonzero(shells == shell)
        available.append(len(members))
        if sum(int(n) for n in counts[:, shell]) > len(members):
            raise ValueError("exclusive class allocation exceeds this mock's shell availability")
        if len(members):
            probabilities[:, shell] = counts[:, shell] / len(members)
        ordered = members[np.lexsort((ids[members], ranks[members]))]
        start = 0
        for index, count in enumerate(counts[:, shell]):
            assignment[ordered[start : start + count]] = index
            start += count
    exterior_counts = {}
    for side, mask, shell in (("below", shells < 0, 0), ("above", shells >= len(edges) - 1, -1)):
        # No invented allocation when adjacent eligible support is empty.
        cumulative = np.cumsum(probabilities[:, shell])
        labels = np.searchsorted(cumulative, ranks[mask], side="right")
        labels[labels == len(class_labels)] = -1
        assignment[mask] = labels
        exterior_counts[side] = [
            int(np.count_nonzero(labels == i)) for i in range(len(class_labels))
        ]
    return assignment, {
        "class_labels": list(class_labels),
        "parent_shell_edges": edges.tolist(),
        "available_parent_counts": available,
        "allocated_parent_counts": counts.tolist(),
        "class_probabilities": probabilities.tolist(),
        "exterior_assigned_counts": exterior_counts,
        "assignment_namespace": "exclusive-class-v1",
        "assignment_seed": int(seed),
    }


def select_exclusive_targets(parent, assignment, *, class_labels, tracer, config):
    """Smear each assigned galaxy once, then apply fixed observed target selections.

    This consumes assignments, not independent tier catalogues. There is no observed-count
    clipping. Saved targets can subsequently serve every reconstruction strategy unchanged.
    """
    assignment = np.asarray(assignment)
    if (
        assignment.shape != (validate_catalog(parent),)
        or not np.issubdtype(assignment.dtype, np.integer)
        or np.any(assignment < -1)
        or np.any(assignment >= len(class_labels))
        or len(set(class_labels)) != len(class_labels)
    ):
        raise ValueError("invalid exclusive assignment")
    prepared = attach_stable_randomness(normalize_weights(parent), role="data", config=config)
    observed = config["selection"]["stable_randomness"]["observed_redshift_column"]
    targets, diagnostics = {}, {}
    bins = measurement_bins(tracer, config)
    edges = [bins[0]["zmin"], *[item["zmax"] for item in bins]]
    for index, label in enumerate(class_labels):
        selected = _slice(prepared, assignment == index)
        smeared = smear_data_redshifts(
            selected, tier_spec(label, config)["sigma_z_over_one_plus_z"], config=config
        )
        # Rows/columns include outside-below and outside-above, so migration conserves counts.
        parent_shell = np.searchsorted(edges, selected["Z"], side="right")
        observed_shell = np.searchsorted(edges, smeared[observed], side="right")
        migration = np.zeros((len(edges) + 1, len(edges) + 1), dtype=np.int64)
        np.add.at(migration, (parent_shell, observed_shell), 1)
        inside = (smeared[observed] >= edges[0]) & (smeared[observed] < edges[-1])
        targets[label] = _slice(smeared, inside)
        diagnostics[label] = {
            "parent_assigned_count": len(parent_shell),
            "measured_final_counts": migration[:, 1:-1].sum(axis=0).tolist(),
            "parent_to_observed_counts_with_exterior": migration.tolist(),
        }
    return targets, diagnostics


def forward_photoz_counts(parent_redshifts, parent_count_weights, observed_edges, sigma0):
    """Integrate the Gaussian kernel over observed bins, including migration tails.

    Count weights are a quadrature of smooth (dV/dz)*n_parent(z)*p_class(z)*dz,
    not n(z) or dN/dz alone and never a single realization's histogram. Parent grid
    refinement is an external validation requirement. Zero error has exact half-open cuts.
    """
    from scipy.special import ndtr

    z, counts, edges = (
        np.asarray(v, dtype=float) for v in (parent_redshifts, parent_count_weights, observed_edges)
    )
    if (
        z.ndim != 1
        or not len(z)
        or counts.shape != z.shape
        or edges.ndim != 1
        or len(edges) < 2
        or np.any(np.diff(edges) <= 0)
        or any(not np.isfinite(v).all() for v in (z, counts, edges))
        or np.any(z <= -1)
        or np.any(counts < 0)
        or not np.isfinite(sigma0)
        or sigma0 < 0
    ):
        raise ValueError("invalid photo-z count quadrature")
    predicted = np.zeros(len(edges) - 1)
    below = above = 0.0
    # Keep memory bounded for fine support grids. Far-tail subtraction uses survival CDFs.
    for start in range(0, len(z), 256):
        zz, ww = z[start : start + 256], counts[start : start + 256]
        if sigma0 == 0:
            index = np.searchsorted(edges, zz, side="right") - 1
            inside = (index >= 0) & (index < len(predicted))
            np.add.at(predicted, index[inside], ww[inside])
            below += ww[index < 0].sum()
            above += ww[index >= len(predicted)].sum()
        else:
            x = (edges[:, None] - zz) / (sigma0 * (1 + zz))
            lo, hi = x[:-1], x[1:]
            mass = np.where(lo > 0, ndtr(-lo) - ndtr(-hi), ndtr(hi) - ndtr(lo))
            predicted += mass @ ww
            below += float(ndtr(x[0]) @ ww)
            above += float(ndtr(-x[-1]) @ ww)
    return {
        "predicted_final_counts": predicted.tolist(),
        "predicted_below_count": float(below),
        "predicted_above_count": float(above),
        "parent_count": float(counts.sum()),
    }


def tier_spec(label: str, config: Optional[Mapping[str, Any]] = None) -> dict:
    """Return one registered photo-z/density tier."""
    resolved = config or load_analysis_config()
    for tier in resolved["selection"]["tiers"]:
        if tier["label"] == label:
            return deepcopy(tier)
    raise KeyError(f"unregistered selection tier {label!r}")


def _slice(catalog: Mapping[str, Any], mask: np.ndarray) -> Catalog:
    """Copy exactly the rows selected by a shape-checked Boolean mask."""
    validate_catalog(catalog)
    if mask.dtype != bool or mask.shape != (validate_catalog(catalog),):
        raise ValueError("catalogue mask has an invalid shape")
    return {str(key): np.asarray(value)[mask].copy() for key, value in catalog.items()}


# Measurement-bin coverage, exact attainable selection, and attained FKP weights


def _measurement_edges_for_field(
    field: Mapping[str, Any], bins: Sequence[Mapping[str, Any]]
) -> list:
    """Return shared bin edges after proving they exactly cover the broad field.

    :func:`prepare_tracer_field` passes the returned edge array to
    :func:`comoving_shell_volume` for its per-bin attainable-count calculation.
    """
    if not bins:
        raise ValueError("a reconstruction field requires measurement bins")
    # Redshift edges are configured decimal identities. Absolute 1e-12 admits
    # only their binary float representation, not a physically open gap/overlap.
    contiguous = all(
        np.isclose(float(left["zmax"]), float(right["zmin"]), rtol=0.0, atol=1e-12)
        for left, right in zip(bins[:-1], bins[1:])
    )
    covers_field = np.isclose(
        float(bins[0]["zmin"]), float(field["zmin"]), rtol=0.0, atol=1e-12
    ) and np.isclose(float(bins[-1]["zmax"]), float(field["zmax"]), rtol=0.0, atol=1e-12)
    if not contiguous or not covers_field:
        raise ValueError("measurement bins must contiguously and exactly cover the tracer field")
    return [float(bins[0]["zmin"])] + [float(item["zmax"]) for item in bins]


def select_exact_attainable(
    catalog: Mapping[str, Any],
    *,
    tracer: str,
    tier_label: str,
    shell_volumes_mpc_h3: Mapping[str, float],
    frozen_counts_by_bin: Mapping[str, int],
    config: Optional[Mapping[str, Any]] = None,
) -> Tuple[Catalog, dict]:
    """Select exact counts with stable ranks, never cloning unavailable data.

    The caller supplies one frozen count per bin. This function verifies that
    count is attainable in this mock and no larger than the configured target;
    the upstream count artifact, not this function, establishes whether it is
    the all-mock common minimum. At fixed photo-z tier, lowest stable ranks make
    the singular selected sample deterministic and independent of row order.
    Different photo-z tiers can move rows across
    bin edges and are not claimed to nest. The returned record distinguishes
    the request from what the mock can supply and what was actually selected.
    Bin membership uses the observed/post-smearing ``Z``. If a tier has no target n(z), the frozen
    common native count is used without imposing a density target. When a
    target n(z) exists, target density times shell volume is floored to an
    integer count before the frozen count is compared with it.
    """
    resolved = config or load_analysis_config()
    if resolved["selection"]["primary_density_policy"] != (
        "spherex_motivated_bounded_by_attainable"
    ):
        raise ValueError("primary density policy is not the reviewed singular selection")
    controls = resolved["selection"]["stable_randomness"]
    rank_column = str(controls["selection_rank_column"])
    observed_column = str(controls["observed_redshift_column"])
    validate_catalog(catalog, (observed_column, rank_column))
    ranks = np.asarray(catalog[rank_column], dtype=float)
    if np.any(~np.isfinite(ranks)):
        raise ValueError("stable selection ranks must be finite")
    if len(np.unique(ranks)) != len(ranks):
        raise ValueError("stable selection ranks collide; row-order invariance is undefined")
    tier = tier_spec(tier_label, resolved)
    targets = tier["target_nbar_h3_mpc3"]
    selected = np.zeros(validate_catalog(catalog), dtype=bool)
    covered = np.zeros(validate_catalog(catalog), dtype=bool)
    records = {}
    for item in measurement_bins(tracer, resolved):
        name = str(item["name"])
        bin_mask = (catalog[observed_column] >= float(item["zmin"])) & (
            catalog[observed_column] < float(item["zmax"])
        )
        covered |= bin_mask
        indices = np.flatnonzero(bin_mask)
        feasible = int(len(indices))
        volume = float(shell_volumes_mpc_h3[name])
        if not np.isfinite(volume) or volume <= 0:
            raise ValueError(f"invalid shell volume for {name}")
        if name not in frozen_counts_by_bin:
            raise ValueError(f"frozen attainable counts omit {name}")
        chosen_count = int(frozen_counts_by_bin[name])
        requested_nbar = None if targets is None else float(targets[name])
        requested_count = None if requested_nbar is None else int(np.floor(requested_nbar * volume))
        if chosen_count < 0:
            raise ValueError(f"frozen attainable count for {name} is negative")
        if chosen_count > feasible:
            raise ValueError(f"frozen attainable count for {name} exceeds this mock's support")
        if requested_count is not None and chosen_count > requested_count:
            raise ValueError(f"frozen attainable count for {name} exceeds its configured target")
        order = indices[np.argsort(ranks[indices], kind="stable")]
        selected[order[:chosen_count]] = True
        attained = chosen_count / volume
        records[name] = {
            "requested_nbar_h3_mpc3": requested_nbar,
            "requested_count": requested_count,
            "maximum_attainable_count": feasible,
            "maximum_attainable_nbar_h3_mpc3": feasible / volume,
            "frozen_selected_count": chosen_count,
            "attained_count": chosen_count,
            "attained_nbar_h3_mpc3": attained,
            "shell_volume_mpc_h3": volume,
            "bounded_by_attainable": requested_count is not None and chosen_count < requested_count,
            "frozen_count_applied": True,
            "target_basis": "native_attainable" if targets is None else "spherex_motivated",
        }
    if not np.all(covered):
        raise ValueError("density-selection input contains rows outside the measurement bins")
    return _slice(catalog, selected), {
        "method": "stable_exact_rank",
        "tier": tier_label,
        "density_policy": "spherex_motivated_bounded_by_attainable",
        "bins": records,
    }


def apply_attained_fkp(
    catalog: Mapping[str, Any],
    *,
    tracer: str,
    attained_by_bin: Mapping[str, Mapping[str, Any]],
    redshift_column: str,
    config: Optional[Mapping[str, Any]] = None,
) -> Catalog:
    """Compute FKP weights from achieved, not requested, number density.

    This function consumes the attained n(z) already computed from count and
    shell volume during exact selection, then inserts it into
    ``1 / (1 + n P0)``. ``P0`` is tracer-specific but fixed across tiers.
    Membership is evaluated using the same observed/post-smearing ``Z`` as
    density selection. The attained density, and therefore the weight, is
    piecewise constant within each measurement bin rather than a smooth n(z).
    The same attained *data* densities are deliberately applied to both data
    and random catalogues; random-catalogue row density is not a second FKP n(z).
    """
    weights = _attained_fkp_weights(
        catalog,
        tracer=tracer,
        attained_by_bin=attained_by_bin,
        redshift_column=redshift_column,
        config=config,
    )
    result = copy_catalog(catalog)
    result["WEIGHT_FKP"] = weights
    return result


def _attained_fkp_weights(catalog, *, tracer, attained_by_bin, redshift_column, config=None):
    """Compute a new weight column without copying unrelated catalogue arrays."""
    resolved = config or load_analysis_config()
    p0 = float(resolved["tracers"][tracer]["fkp_p0_h3_mpc3"])
    validate_catalog(catalog, (redshift_column,))
    weights = np.empty(validate_catalog(catalog), dtype=float)
    covered = np.zeros(validate_catalog(catalog), dtype=bool)
    for item in measurement_bins(tracer, resolved):
        name = str(item["name"])
        nbar = float(attained_by_bin[name]["attained_nbar_h3_mpc3"])
        mask = (catalog[redshift_column] >= float(item["zmin"])) & (
            catalog[redshift_column] < float(item["zmax"])
        )
        weights[mask] = 1.0 / (1.0 + nbar * p0)
        covered |= mask
    if not np.all(covered):
        raise ValueError("FKP input includes rows outside the tracer measurement bins")
    return weights


# Calibration-derived radial selection for the existing angular randoms


#: How far above the requested smoothing the achieved residual may sit before the curve stops
#: counting as the one that was asked for.
#:
#: The value is set from the measured separation rather than copied from the library's own
#: internal tolerance, because the two populations are far apart and the library's tolerance
#: sits inside the wrong one. Across the twenty calibration samples at the production setting
#: every converged fit lands within 0.1 per cent of its request -- the worst is +9.4e-4 -- while
#: the one sample that genuinely fails misses by a factor of 21. Using the library's 1e-3 would
#: put the threshold at 94 per cent of the worst converged sample's overshoot, so an ordinary
#: change in the solver's path would start refusing a production sample that is perfectly good.
#: Five per cent sits about fifty times above the converged cluster and twenty times below the
#: real failure, which separates them with room on both sides.
_SPLINE_RESIDUAL_TOLERANCE = 5e-2

#: How many shells at each end of the field the edge residual inspects. A smoothing
#: spline extrapolates as a constant beyond its knots and a Gaussian kernel pads with the
#: nearest value, so both misbehave in the outermost shells while looking well-behaved on
#: any whole-field summary. Five shells is narrow enough that the interior cannot mask a
#: bad edge and wide enough that one noisy shell cannot dominate the number.
EDGE_RESIDUAL_SHELLS = 5


def spline_residual_converged(achieved: float, requested: float) -> bool:
    """Say whether a fitted spline actually reached the smoothness it was asked for.

    The library searches for a knot set whose residual sum equals the requested smoothing,
    but it abandons the search after a fixed number of iterations and returns the best
    approximation it has, raising a warning rather than an error. Where that iteration
    stopped depends on the numerical path rather than on any input, so a curve that missed
    its target would change the artifact's content hash without any input changing.

    The comparison is one-sided. Landing below the requested residual means the smoothest
    curve the knot budget allows already tracks the calibration mean more closely than was
    asked, which is what a nearly featureless sample gives and is not a defect. Landing
    above it means the search ran out, and that is the failure worth refusing.
    """
    if not (math.isfinite(achieved) and math.isfinite(requested)):
        return False
    if achieved < 0.0 or requested < 0.0:
        return False
    return achieved <= requested * (1.0 + _SPLINE_RESIDUAL_TOLERANCE)


def _bandwidth_semantics(method: str, weighting: str) -> str:
    """Name the scale a stored bandwidth was expressed on, so two cannot be confused."""
    if method != "smoothing_spline":
        return "gaussian_sigma_bins"
    if weighting == "inverse_standard_error":
        return "chi_square_per_center"
    return "smoothing_factor_per_center"


def _edge_residual_in_noise_units(
    mean_counts: np.ndarray,
    mass: np.ndarray,
    standard_error: np.ndarray,
    measurable: np.ndarray,
) -> float:
    """Return the worst model-minus-mean disagreement at the field edges, in noise units.

    This descriptive quantity is the largest absolute difference between the
    model and the calibration mean over the outermost shells at each end, divided by the
    standard error on that mean, so it is on the same scale as every other residual the
    calibration pages show and a value near one means the edges are as well described as
    the interior. The edges are singled out because that is where a spline's constant
    extrapolation and a kernel's nearest-value padding go wrong, and where model mass would
    leak into empty shells if any existed.
    """
    total = float(np.sum(mean_counts))
    if not np.isfinite(total) or total <= 0:
        raise ValueError("edge residual requires positive total calibration counts")
    span = min(EDGE_RESIDUAL_SHELLS, mean_counts.size)
    edge = np.zeros(mean_counts.size, dtype=bool)
    edge[:span] = True
    edge[-span:] = True
    usable = edge & measurable
    if not np.any(usable):
        return 0.0
    residual = np.abs(mass[usable] * total - mean_counts[usable]) / standard_error[usable]
    return float(np.max(residual))


def fit_smooth_radial_model(
    counts_by_mock: Sequence[Sequence[float]],
    z_edges: Sequence[float],
    *,
    calibration_mock_ids: Sequence[int],
    smoothing_method: str,
    bandwidth: float,
    sample_identity: str,
    residual_weighting: str = "unweighted",
) -> dict:
    """Estimate a smooth target mass for deterministic thinning of existing randoms.

    A random catalogue must sample the survey selection function without the
    realization's cosmological clustering. Its angular coordinates already
    encode the DESI footprint and veto masks, but it also needs a redshift
    probability distribution: the expected ``p(z)`` (equivalently ``n(z)`` up
    to normalization) of the selected tracer. Copying one mock's observed
    redshift histogram would imprint that realization's line-of-sight large-
    scale structure into the randoms, so each row here is one calibration
    mock's selected-sample counts and the model begins with their across-mock
    mean.

    The counts use uniform redshift bins for the single Galactic cap encoded in
    ``sample_identity``. The caller must choose either a smoothing spline, whose
    bandwidth is the per-centre smoothing factor multiplied by the centre count,
    or a Gaussian kernel, whose bandwidth is sigma in fine-bin units. Normalizing
    the nonnegative smoothed counts produces target probability mass by bin.
    :func:`thin_randoms_to_smooth_target` uses that mass only to allocate an
    exact row budget; it never changes any retained random value.

    ``residual_weighting`` selects what the spline's smoothing parameter is measured
    against, and the distinction is not cosmetic. Left ``unweighted`` the library compares
    the requested smoothing against a sum of squared residuals in raw counts squared, so
    the same bandwidth means different things for samples of different size: measured over
    the twenty calibration samples the requested residual sits between one hundred and two
    thousand six hundred times below the noise level of the very mean being fitted, and
    that factor itself varies by a factor sixty-five between samples. Any bandwidth chosen
    on that scale is therefore fitting noise, and no single frozen value can be right for
    every sample. Weighting by ``inverse_standard_error`` divides each residual by the
    standard error on the calibration mean in that shell, which turns the smoothing
    parameter into a chi-square: the standardised residual then has root-mean-square
    exactly the square root of the bandwidth, so a bandwidth of one means the model sits
    precisely at the noise floor, independent of sample size. The two settings are recorded
    under different ``bandwidth_semantics`` so a stored candidate always declares which
    scale its bandwidth was expressed on, and one can never be mistaken for the other.

    The resulting randoms are used by reconstruction and xi estimators to
    represent the unclustered survey selection. This constructor proves only
    internal geometry and occupied-bin support. It cannot prove that smoothing
    preserves BAO results, reduces variance, or has enough finite random rows;
    those require scientific review. The caller supplies the complete configured ensemble,
    and failure never selects the shuffled diagnostic or any alternate method.
    """
    if not str(sample_identity).strip():
        raise ValueError("radial model requires an exact sample identity")
    counts = np.asarray(counts_by_mock, dtype=float)
    edges = np.asarray(z_edges, dtype=float)
    if counts.ndim != 2 or counts.shape[0] != len(calibration_mock_ids):
        raise ValueError("radial counts must have one row per calibration mock")
    mock_ids = [int(value) for value in calibration_mock_ids]
    if len(mock_ids) != len(set(mock_ids)):
        raise ValueError("calibration mock ids must be unique")
    if counts.shape[0] < 2:
        raise ValueError("radial variance calibration requires at least two mocks")
    if counts.shape[1] != len(edges) - 1 or np.any(counts < 0) or np.any(~np.isfinite(counts)):
        raise ValueError("radial counts and edges are inconsistent")
    if np.any(np.diff(edges) <= 0):
        raise ValueError("radial-model edges must be strictly increasing")
    widths = np.diff(edges)
    # Radial-model redshift edges use the same absolute configured-edge tolerance.
    if not np.allclose(widths, widths[0], rtol=0.0, atol=1e-12):
        raise ValueError("radial-model smoothing requires uniform redshift bins")
    method = str(smoothing_method)
    if method not in {"smoothing_spline", "gaussian_kernel"}:
        raise ValueError("smoothing method must be smoothing_spline or gaussian_kernel")
    width = float(bandwidth)
    if not np.isfinite(width) or width <= 0:
        raise ValueError("radial smoothing bandwidth must be finite and positive")
    weighting = str(residual_weighting)
    if weighting not in {"unweighted", "inverse_standard_error"}:
        raise ValueError("residual weighting must be unweighted or inverse_standard_error")
    mean_counts = counts.mean(axis=0)
    if not np.any(mean_counts > 0):
        raise ValueError("calibration radial counts contain no support")
    variance = counts.var(axis=0, ddof=1)
    standard_error = np.sqrt(variance / counts.shape[0])
    # A shell whose mocks all agree exactly carries no measurable noise, so it cannot
    # weight a residual and cannot be judged against a noise floor. Those shells are left
    # out of the weighted fit rather than given an infinite weight, and the model is still
    # evaluated on the full grid so the returned mass covers every shell.
    measurable = np.isfinite(standard_error) & (standard_error > 0)
    convergence: dict = {
        "residual_weighting": weighting,
        "requested_smoothing": None,
        "achieved_residual": None,
        "converged": True,
    }
    if method == "smoothing_spline":
        try:
            from scipy.interpolate import UnivariateSpline
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError("scipy is required for radial smoothing splines") from exc
        centers = 0.5 * (edges[:-1] + edges[1:])
        degree = min(3, len(centers) - 1)
        if weighting == "inverse_standard_error":
            if int(measurable.sum()) <= degree:
                raise ValueError("weighted radial smoothing needs more measurable shells")
            requested = width * float(measurable.sum())
            spline = UnivariateSpline(
                centers[measurable],
                mean_counts[measurable],
                w=1.0 / standard_error[measurable],
                k=degree,
                s=requested,
                ext=3,
            )
        else:
            requested = width * len(centers)
            spline = UnivariateSpline(centers, mean_counts, k=degree, s=requested, ext=3)
        smoothed = spline(centers)
        # The library solves for a curve whose residual sum equals the requested smoothing,
        # but it gives up after a fixed iteration count and returns whatever it reached,
        # warning rather than raising. A curve that missed its target is not the curve that
        # was asked for, and because where the iteration stopped depends on the numerical
        # path it would also change the resulting target without any input changing.
        # The achieved value is therefore recorded and compared here rather than discarded.
        achieved = float(spline.get_residual())
        convergence["requested_smoothing"] = float(requested)
        convergence["achieved_residual"] = achieved
        # The test is one-sided on purpose. Landing *below* the requested residual is not a
        # failure: it means the smoothest curve the knot budget allows already tracks the
        # mean more closely than was asked, which happens for a nearly featureless sample.
        # Landing *above* it is the failure, because the iteration ran out before reaching
        # the requested smoothness and returned a curve nobody chose.
        convergence["converged"] = spline_residual_converged(achieved, requested)
    else:
        try:
            from scipy.ndimage import gaussian_filter1d
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError("scipy is required for Gaussian radial smoothing") from exc
        smoothed = gaussian_filter1d(mean_counts, sigma=width, mode="nearest")
    smoothed = np.clip(np.asarray(smoothed, dtype=float), 0.0, None)
    occupied = mean_counts > 0
    occupied_support = bool(np.all(smoothed[occupied] > 0))
    if not occupied_support:
        raise RuntimeError("positive smoothing kernel lost an occupied calibration bin")
    # A shell where the calibration mean is exactly zero holds no galaxies, so a random placed
    # there sits at a redshift the survey does not reach. That is a direct bias in the
    # Landy-Szalay ratio rather than an inefficiency, and neither smoothing family prevents it:
    # a Gaussian kernel necessarily spreads mass into empty shells beside occupied ones, and a
    # ringing spline can too. The invariant is therefore enforced by construction rather than
    # merely asserted, because enforcing it makes the leak impossible while asserting it only
    # notices afterwards. How much mass was removed is recorded so the correction is visible
    # and can be judged; this constructor makes no judgement of its own.
    total_before = float(np.sum(smoothed))
    if total_before <= 0.0:
        raise RuntimeError("smoothed radial model has no positive mass")
    unoccupied_mass = float(np.sum(smoothed[~occupied]))
    smoothed = np.where(occupied, smoothed, 0.0)
    removed_fraction = unoccupied_mass / total_before
    mass = smoothed / smoothed.sum()
    empirical_mass = mean_counts / mean_counts.sum()
    fractional_variance = [
        None if mean <= 0 else float(value / mean**2) for mean, value in zip(mean_counts, variance)
    ]
    finite_fractional_variance = [value for value in fractional_variance if value is not None]
    edge_residual = _edge_residual_in_noise_units(mean_counts, mass, standard_error, measurable)
    payload = {
        "schema_version": 1,
        "calibration_mock_ids": mock_ids,
        "z_edges": edges.tolist(),
        "mean_counts": mean_counts.tolist(),
        "target_probability_mass": mass.tolist(),
        "smoothing_method": method,
        "bandwidth": width,
        "bandwidth_semantics": _bandwidth_semantics(method, weighting),
        "sample_identity": str(sample_identity),
        "convergence_evidence": convergence,
        "support_evidence": {
            "occupied_bins": np.flatnonzero(mean_counts > 0).tolist(),
            "occupied_bins_have_positive_model_mass_by_construction": occupied_support,
            "unoccupied_bins_have_zero_model_mass": bool(np.all(mass[~occupied] == 0.0)),
            "unoccupied_bin_count": int(np.count_nonzero(~occupied)),
            "unoccupied_bin_mass_removed_fraction": removed_fraction,
            "construction": method,
        },
        "fidelity_evidence": {
            "total_variation_distance": float(0.5 * np.sum(np.abs(mass - empirical_mass))),
            "maximum_absolute_probability_difference": float(np.max(np.abs(mass - empirical_mass))),
            "edge_residual": edge_residual,
            "edge_residual_shell_count": EDGE_RESIDUAL_SHELLS,
        },
        "variance_evidence": {
            "meaning": (
                "unbiased_across_mock_raw_count_variance_including_shot_noise; "
                "summary_mean_excludes_zero_mean_bins"
            ),
            "count_variance_by_bin": variance.tolist(),
            "fractional_variance_by_bin": fractional_variance,
            "occupied_bin_count": len(finite_fractional_variance),
            "mean_fractional_variance_over_occupied_bins": float(
                np.mean(finite_fractional_variance)
            ),
            "standard_error_of_mean_count_by_bin": np.sqrt(variance / counts.shape[0]).tolist(),
        },
    }
    return payload


def forward_radial_model(parent_model, *, observed_edges, sigma0, sample_identity):
    """Forward a smooth ensemble parent-class count model into a random-selection artifact.

    The input is the output of ``fit_smooth_radial_model`` applied to parent-class
    expected counts, before error draws. Its probability mass supplies the smooth
    radial shape and its mean counts supply the normalization; these are not n(z).
    """
    ids = parent_model.get("calibration_mock_ids", [])
    if len(ids) < 2 or len(set(ids)) != len(ids):
        raise ValueError("forward radial selection requires an identified parent ensemble")
    if not parent_model.get("convergence_evidence", {}).get("converged"):
        raise ValueError("forward radial selection requires a converged smooth parent model")
    if not sample_identity or not parent_model.get("sample_identity"):
        raise ValueError("forward radial selection requires parent and target identities")
    edges = np.asarray(parent_model["z_edges"], dtype=float)
    mass = np.asarray(parent_model["target_probability_mass"], dtype=float)
    counts = np.asarray(parent_model["mean_counts"], dtype=float)
    if (
        edges.ndim != 1
        or mass.shape != (len(edges) - 1,)
        or counts.shape != mass.shape
        or not np.isfinite(mass).all()
        or not np.isfinite(counts).all()
        or not np.isfinite(edges).all()
        or np.any(np.diff(edges) <= 0)
        or np.any(mass < 0)
        or np.any(counts < 0)
        or counts.sum() <= 0
        or not np.isclose(mass.sum(), 1.0, rtol=0, atol=1e-12)
    ):
        raise ValueError("invalid smooth parent radial model")
    prediction = forward_photoz_counts(
        (edges[:-1] + edges[1:]) / 2,
        mass * counts.sum(),
        observed_edges,
        sigma0,
    )
    final_counts = np.asarray(prediction["predicted_final_counts"])
    if final_counts.sum() <= 0:
        raise ValueError("forward radial selection has no final support")
    return {
        "method": "smooth_parent_forward_prediction",
        "sample_identity": str(sample_identity),
        "parent_model": copy.deepcopy(parent_model),
        "sigma_z_over_one_plus_z": float(sigma0),
        "z_edges": np.asarray(observed_edges, dtype=float).tolist(),
        "target_probability_mass": (final_counts / final_counts.sum()).tolist(),
        **prediction,
    }


def matching_exclusive_randoms(
    randoms,
    radial_artifact,
    *,
    data_count,
    expected_sample_identity,
    tracer,
    attained_by_bin,
    config,
    parallel=None,
    input_cache=None,
):
    """Match existing angular randoms to the forward prediction without resmearing rows."""
    if radial_artifact.get("method") != "smooth_parent_forward_prediction":
        raise ValueError("exclusive randoms require the forward-predicted radial selection")
    # Recompute from the recorded parent model, so an altered target mass is not accepted.
    expected = forward_radial_model(
        radial_artifact["parent_model"],
        observed_edges=radial_artifact["z_edges"],
        sigma0=radial_artifact["sigma_z_over_one_plus_z"],
        sample_identity=expected_sample_identity,
    )
    if not same_identity(expected, radial_artifact):
        raise ValueError(
            "exclusive random selection differs from its parent prediction or identity"
        )
    prepared = _normalized_randomness(
        randoms, role="random", config=config, parallel=parallel, input_cache=input_cache
    )
    selected, diagnostics = thin_randoms_to_smooth_target(
        prepared,
        radial_artifact,
        data_count=data_count,
        exact_random_data_ratio=config["selection"]["radial_randoms"]["exact_random_data_ratio"],
        expected_sample_identity=expected_sample_identity,
        config=config,
        input_cache=input_cache,
    )
    # Thinning already owns independent arrays. Attach the new column without
    # making another full selected-catalogue copy beside the raw input and cache.
    selected["WEIGHT_FKP"] = _attained_fkp_weights(
        selected,
        tracer=tracer,
        attained_by_bin=attained_by_bin,
        redshift_column="Z",
        config=config,
    )
    return selected, {**diagnostics, "radial_selection": radial_artifact["method"]}


def thin_randoms_to_smooth_target(
    randoms: Mapping[str, Any],
    radial_artifact: Mapping[str, Any],
    *,
    data_count: int,
    exact_random_data_ratio: int,
    expected_sample_identity: str,
    config: Optional[Mapping[str, Any]] = None,
    input_cache=None,
) -> Tuple[Catalog, dict]:
    """Thin unchanged random rows to an exact frozen smooth radial target.

    The frozen target mass allocates exactly ``50 * data_count`` rows over the
    declared fine redshift bins by floor plus stable largest remainder. Within
    each bin, the lowest object-bound thinning ranks are retained. Insufficient
    support fails closed. Every retained RA, DEC, Z, identity, weight, and
    metadata value is checked elementwise against its source row; there is no
    cloning, inverse-CDF reassignment, or fallback. FKP is assigned later from
    the attained data density. The shared target comes from the run's complete-ensemble
    preparation and must carry the requested sample identity and finite probability mass.
    """
    resolved = config or load_analysis_config()
    controls = resolved["selection"]["stable_randomness"]
    rank_column = str(controls["random_thinning_rank_column"])
    validate_catalog(randoms, ("RA", "DEC", "Z", rank_column))
    if any(
        np.any(~np.isfinite(np.asarray(randoms[name], dtype=float)))
        for name in ("RA", "DEC", "Z", rank_column)
    ):
        raise ValueError("random angles, redshifts, and thinning ranks must be finite")
    if radial_artifact.get("sample_identity") != str(expected_sample_identity):
        raise ValueError("radial artifact belongs to another tracer/tier/density/cap sample")
    ratio = int(exact_random_data_ratio)
    if int(data_count) <= 0 or ratio != int(
        resolved["selection"]["radial_randoms"]["exact_random_data_ratio"]
    ):
        raise ValueError("random/data ratio differs from the resolved configuration")
    required_count = ratio * int(data_count)
    available_count = validate_catalog(randoms)
    if available_count < required_count:
        raise ValueError("finite random budget cannot supply the exact random/data ratio")
    edges = np.asarray(radial_artifact["z_edges"], dtype=float)
    mass = np.asarray(radial_artifact["target_probability_mass"], dtype=float)
    if (
        edges.ndim != 1
        or mass.ndim != 1
        or len(edges) != len(mass) + 1
        or len(edges) < 2
        or np.any(~np.isfinite(edges))
        or np.any(~np.isfinite(mass))
        or np.any(np.diff(edges) <= 0)
        or np.any(mass < 0)
        or not np.isclose(mass.sum(), 1.0, rtol=0.0, atol=1e-12)
    ):
        raise ValueError("radial artifact edges and target mass are invalid")
    raw_targets = mass * required_count
    target_counts = np.floor(raw_targets).astype(int)
    remainder = required_count - int(target_counts.sum())
    if remainder:
        order = np.argsort(-(raw_targets - target_counts), kind="stable")
        target_counts[order[:remainder]] += 1
    redshift = np.asarray(randoms["Z"], dtype=float)
    ranks = np.asarray(randoms[rank_column], dtype=float)
    if input_cache is None:
        bins = np.searchsorted(edges, redshift, side="right") - 1
        bins[redshift == edges[-1]] = -1
    else:
        grouped, offsets = input_cache.random_order(randoms, edges, rank_column)
    selected = np.zeros(validate_catalog(randoms), dtype=bool)
    available_counts = []
    for index, target_count in enumerate(target_counts):
        if input_cache is None:
            candidates = np.flatnonzero(bins == index)
        else:
            start, stop = offsets[index + 1 : index + 3]
            candidates = grouped[start:stop]
        available_counts.append(int(len(candidates)))
        if target_count > len(candidates):
            raise ValueError(f"random support bin {index} cannot supply its smooth target")
        order = (
            candidates[np.argsort(ranks[candidates], kind="stable")]
            if input_cache is None
            else candidates
        )
        selected[order[:target_count]] = True
    result = _slice(randoms, selected)
    if validate_catalog(result) != required_count:
        raise RuntimeError("smooth-target thinning did not produce the exact random budget")
    for name in randoms:
        if not np.array_equal(result[name], np.asarray(randoms[name])[selected]):
            raise RuntimeError("random thinning changed retained row values")
    return result, {
        "method": "calibration_smooth_target_exact_thinning",
        "retained_rows_unchanged": True,
        "support_bin_width_redshift": float(np.diff(edges)[0]),
        "target_counts_by_bin": target_counts.tolist(),
        "available_counts_by_bin": available_counts,
        "exact_random_data_ratio": ratio,
        "available_random_data_ratio_before_thinning": float(available_count / data_count),
        "attained_random_data_ratio": float(validate_catalog(result) / data_count),
        "radial_sample_identity": radial_artifact["sample_identity"],
    }


def disjoint_random_views(
    randoms: Mapping[str, Any],
    z_edges: Sequence[float],
    *,
    view_count: int,
    config: Optional[Mapping[str, Any]] = None,
) -> dict:
    """Partition retained randoms by stable thinning rank within each radial support bin.

    Return row indices for explicit finite-random diagnostics. Counts are requested by the
    diagnostic caller; no digest inventory, acceptance flag or new random draw is produced.
    """
    resolved = config or load_analysis_config()
    controls = resolved["selection"]["stable_randomness"]
    rank_column = str(controls["random_thinning_rank_column"])
    identity_column = str(controls["identity_column"])
    validate_catalog(randoms, ("RA", "DEC", "Z", rank_column, identity_column))
    count = int(view_count)
    if count < 2:
        raise ValueError("disjoint random views require at least two views")
    edges = np.asarray(z_edges, dtype=float)
    if edges.ndim != 1 or len(edges) < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError("random-view support edges must be strictly increasing")
    redshift = np.asarray(randoms["Z"], dtype=float)
    if len(redshift) < count:
        raise ValueError("retained randoms cannot supply one row per disjoint view")
    bins = np.searchsorted(edges, redshift, side="right") - 1
    bins[redshift == edges[-1]] = len(edges) - 2
    if np.any(bins < 0) or np.any(bins >= len(edges) - 1):
        raise ValueError("retained randoms fall outside the radial model's support")
    ranks = np.asarray(randoms[rank_column], dtype=float)
    identities = np.asarray(randoms[identity_column])
    rows_by_view: list = [[] for _ in range(count)]
    for index in range(len(edges) - 1):
        members = np.flatnonzero(bins == index)
        # The identity breaks a rank tie, so both the membership of each view and the row order
        # inside it depend only on object-bound values. Digesting rows in the order they arrive
        # in the array instead would make the digests a property of how the catalogue happened
        # to be sorted.
        ordered = members[np.lexsort((identities[members], ranks[members]))]
        for position, row in enumerate(ordered):
            rows_by_view[position % count].append(int(row))
    views = [np.asarray(rows, dtype=np.int64) for rows in rows_by_view]
    if any(not len(rows) for rows in views):
        raise ValueError("a random view holds no row")
    joined = np.concatenate(views)
    if len(joined) != len(redshift) or len(np.unique(joined)) != len(redshift):
        raise RuntimeError("random views do not partition retained rows")
    return {"view_indices": views, "view_row_counts": [len(rows) for rows in views]}


def apply_boundary_experiment(data, *, tracer, tier_label, config):
    """Apply the frozen native-redshift cut after merging, without changing random streams."""
    experiment = config["selection"]["boundary_support"].get("experiment")
    if experiment is None or experiment["tracer"] != tracer:
        return data, None
    active = experiment["condition"] == tier_label
    if experiment["mode"] == "truncate" and not active:
        return data, None
    z = np.asarray(data["Z"], dtype=float)
    field = reconstruction_field(tracer, config)
    boundary = field["zmin" if experiment["side"] == "below" else "zmax"]
    depth = experiment["depth_redshift"]
    keep = np.ones(len(z), dtype=bool)
    if experiment["mode"] == "truncate":
        # Lower endpoint included; upper endpoint excluded, as for extraction windows.
        keep = (
            z >= max(0.0, boundary - depth)
            if experiment["side"] == "below"
            else z < boundary + depth
        )
    elif "IS_BOUNDARY_EXTENSION" not in data:
        raise ValueError("extension requires identified original and added populations")
    elif not active:
        # The wider raw slice must not inject its new population into other conditions.
        depth = experiment["reference_depth_redshift"]
        keep &= ~np.asarray(data["IS_BOUNDARY_EXTENSION"], dtype=bool)
    injected = np.asarray(data.get("IS_RESERVOIR", np.zeros(len(z), dtype=bool)), dtype=bool)
    record = {
        "mode": experiment["mode"],
        "active_condition": active,
        "side": experiment["side"],
        "depth_redshift": depth,
        "removed_clustering_count": int(np.count_nonzero(~keep & ~injected)),
        "removed_reservoir_count": int(np.count_nonzero(~keep & injected)),
    }
    return _slice(data, keep), record


def prepare_data_for_selection(
    data,
    *,
    tracer,
    tier_label,
    cap,
    reservoir=None,
    reservoir_record=None,
    config=None,
    parallel=None,
    input_cache=None,
):
    """Normalize, inject boundary support, smear and cut the parent before count selection."""
    resolved = config or load_analysis_config()
    tier = tier_spec(tier_label, resolved)
    prepared = _normalized_randomness(
        data, role="data", config=resolved, parallel=parallel, input_cache=input_cache
    )
    record = {"injected": False}
    merge_record = {
        "reservoir_candidate_row_count": 0,
        "overlap_removed_row_count": 0,
        "edge_support": [],
    }
    if reservoir is not None:
        if reservoir_record is None:
            raise ValueError("reservoir data require their identified source record")
        prepared_reservoir = _normalized_randomness(
            reservoir, role="data", config=resolved, parallel=parallel, input_cache=input_cache
        )
        prepared, merge_record = merge_same_tracer_reservoir(
            prepared,
            prepared_reservoir,
            tracer=tracer,
            expected_sample_identity=reservoir_record["sample_identity"],
            reservoir_record=reservoir_record,
            sigma_z_over_one_plus_z=float(tier["sigma_z_over_one_plus_z"]),
            config=resolved,
        )
        record = {
            "injected": bool(prepared["IS_RESERVOIR"].any()),
            "injected_row_count": int(prepared["IS_RESERVOIR"].sum()),
            "sample_identity": reservoir_record["sample_identity"],
            "source": reservoir_record.get("path"),
        }
    prepared, experiment_record = apply_boundary_experiment(
        prepared, tracer=tracer, tier_label=tier_label, config=resolved
    )
    if experiment_record is not None:
        record["boundary_experiment"] = experiment_record
        merge_record["boundary_experiment"] = experiment_record
        merge_record["uncut_edge_support"] = merge_record["edge_support"]
        merge_record["edge_support"] = []
        injected = np.asarray(
            prepared.get("IS_RESERVOIR", np.zeros(len(prepared["Z"]), dtype=bool))
        )
        record["injected"] = bool(injected.any())
        record["injected_row_count"] = int(injected.sum())
        z = np.asarray(prepared["Z"])
        for edge in merge_record["uncut_edge_support"]:
            depth = edge["redshift"] - z if edge["side"] == "below" else z - edge["redshift"]
            exterior = depth > 0 if edge["side"] == "below" else depth >= 0
            available = float(depth[exterior].max()) if np.any(exterior) else 0.0
            merge_record["edge_support"].append(
                {
                    **edge,
                    "clustering_row_count": int(np.count_nonzero(exterior & ~injected)),
                    "added_reservoir_row_count": int(np.count_nonzero(exterior & injected)),
                    "available_depth_redshift": available,
                    "shortfall_redshift": max(
                        0.0, edge["required_guard_width_redshift"] - available
                    ),
                    "guard_reached": available >= edge["required_guard_width_redshift"],
                }
            )
    prepared = smear_data_redshifts(
        prepared, float(tier["sigma_z_over_one_plus_z"]), config=resolved
    )
    column = resolved["selection"]["stable_randomness"]["observed_redshift_column"]
    field = reconstruction_field(tracer, resolved)
    inside = (prepared[column] >= field["zmin"]) & (prepared[column] < field["zmax"])
    injected = np.asarray(
        prepared.get("IS_RESERVOIR", np.zeros(len(inside), dtype=bool)), dtype=bool
    )
    source_column = resolved["selection"]["stable_randomness"]["source_redshift_column"]
    originally_inside = (prepared[source_column] >= field["zmin"]) & (
        prepared[source_column] < field["zmax"]
    )
    width = resolved["selection"]["radial_randoms"]["support_bin_width_redshift"]
    edges = np.arange(field["zmin"], field["zmax"] + 0.5 * width, width)
    edges[-1] = field["zmax"]
    record["boundary_counts"] = {
        "z_edges": edges.tolist(),
        "original_clustering_counts": np.histogram(
            prepared[source_column][originally_inside & ~injected], bins=edges
        )[0].tolist(),
        "available_observed_counts": np.histogram(prepared[column][inside], bins=edges)[0].tolist(),
        "reservoir_observed_counts": np.histogram(prepared[column][inside & injected], bins=edges)[
            0
        ].tolist(),
        "original_scattered_out_count": int(
            np.count_nonzero(originally_inside & ~inside & ~injected)
        ),
        "original_outside_after_smearing_count": int(np.count_nonzero(~inside & ~injected)),
        "original_scattered_in_count": int(
            np.count_nonzero(~originally_inside & inside & ~injected)
        ),
        "reservoir_scattered_in_count": int(np.count_nonzero(inside & injected)),
        **merge_record,
    }
    return _slice(prepared, inside), record


def prepare_tracer_field(
    data: Mapping[str, Any],
    randoms: Mapping[str, Any],
    *,
    tracer: str,
    tier_label: str,
    cap: str,
    cosmology: Any,
    radial_artifact: Mapping[str, Any],
    sample_identity: str,
    frozen_counts_by_bin: Mapping[str, int],
    reservoir: Optional[Mapping[str, Any]] = None,
    reservoir_record: Optional[Mapping[str, Any]] = None,
    reservoir_sample_identity: Optional[str] = None,
    config: Optional[Mapping[str, Any]] = None,
    parallel=None,
    input_cache=None,
) -> Tuple[Catalog, Catalog, dict]:
    """Apply the complete configured catalogue transformation for one cap.

    This is the module's causal entry point. It consumes the pipeline-derived
    sample identity plus frozen reservoir and radial-model evidence and returns
    selected data, rematched randoms, and provenance. Upstream byte and
    cosmology evidence remains the pipeline's responsibility. Cartesian
    conversion and reconstruction also occur there. The tracer field must
    equal the contiguous union of its measurement bins. Radial evidence is
    cap-specific because each cap has its own selected n(z). Reservoir
    provenance identifies the same parent population; no downstream BAO outcome
    selects or authenticates a reservoir.

    Reservoir merging receives cap-selected, weight-normalized, but
    redshift-unfiltered data before smearing and the broad-field cut. That
    ordering admits exterior clustering and reservoir galaxies that scatter inward.
    Clustering copies retain precedence when the two sources overlap; available
    exterior support is retained even when a diagnostic guard is not reached.
    """
    resolved = config or load_analysis_config()
    if reservoir is not None and (
        not isinstance(reservoir_record, Mapping)
        or reservoir_record.get("sample_identity") != reservoir_sample_identity
    ):
        raise ValueError("reservoir identity differs from the selected parent")
    prepared_data, reservoir_provenance = prepare_data_for_selection(
        data,
        tracer=tracer,
        tier_label=tier_label,
        cap=cap,
        reservoir=reservoir,
        reservoir_record=reservoir_record,
        config=resolved,
        parallel=parallel,
        input_cache=input_cache,
    )
    observed_column = str(resolved["selection"]["stable_randomness"]["observed_redshift_column"])
    field = reconstruction_field(tracer, resolved)
    bins = measurement_bins(tracer, resolved)
    edges = _measurement_edges_for_field(field, bins)
    field_mask = (prepared_data[observed_column] >= float(field["zmin"])) & (
        prepared_data[observed_column] < float(field["zmax"])
    )
    prepared_data = _slice(prepared_data, field_mask)

    area = footprint_area_deg2(tracer, cap, resolved)
    volumes = comoving_shell_volume(cosmology, edges, area)
    volume_by_bin = {str(item["name"]): float(volume) for item, volume in zip(bins, volumes)}
    selected_data, selection_record = select_exact_attainable(
        prepared_data,
        tracer=tracer,
        tier_label=tier_label,
        shell_volumes_mpc_h3=volume_by_bin,
        frozen_counts_by_bin=frozen_counts_by_bin,
        config=resolved,
    )
    from .config.density import matched_selection_record

    selection_record = matched_selection_record(selection_record, resolved, tracer, tier_label, cap)
    selected_data = apply_attained_fkp(
        selected_data,
        tracer=tracer,
        attained_by_bin=selection_record["bins"],
        redshift_column=observed_column,
        config=resolved,
    )
    if reservoir_provenance["injected"]:
        reservoir_provenance["selected_reservoir_count"] = int(
            np.count_nonzero(selected_data["IS_RESERVOIR"])
        )

    prepared_randoms = _normalized_randomness(
        randoms, role="random", config=resolved, parallel=parallel, input_cache=input_cache
    )
    radial_controls = resolved["selection"]["radial_randoms"]
    artifact_edges = np.asarray(radial_artifact.get("z_edges", []), dtype=float)
    if artifact_edges.ndim != 1 or artifact_edges.size < 2:
        raise ValueError("radial artifact has no usable redshift support")
    # Artifact/config edges share the absolute configured-decimal tolerance.
    if not np.isclose(
        artifact_edges[0], float(field["zmin"]), rtol=0.0, atol=1e-12
    ) or not np.isclose(artifact_edges[-1], float(field["zmax"]), rtol=0.0, atol=1e-12):
        raise ValueError("radial artifact support differs from the reconstructed tracer field")
    exact_random_data_ratio = int(radial_controls["exact_random_data_ratio"])
    prepared_randoms, random_record = thin_randoms_to_smooth_target(
        prepared_randoms,
        radial_artifact,
        data_count=validate_catalog(selected_data),
        exact_random_data_ratio=exact_random_data_ratio,
        expected_sample_identity=sample_identity,
        config=resolved,
        input_cache=input_cache,
    )
    random_field_mask = (prepared_randoms["Z"] >= float(field["zmin"])) & (
        prepared_randoms["Z"] < float(field["zmax"])
    )
    if not np.all(random_field_mask):
        raise RuntimeError("radial artifact generated a redshift outside its exact tracer field")
    prepared_randoms = apply_attained_fkp(
        prepared_randoms,
        tracer=tracer,
        attained_by_bin=selection_record["bins"],
        redshift_column="Z",
        config=resolved,
    )
    return (
        selected_data,
        prepared_randoms,
        {
            "tracer": tracer,
            "cap": cap.upper(),
            "field": dict(field),
            "selection": selection_record,
            "reservoir": reservoir_provenance,
            "randoms": random_record,
            "fkp_p0_h3_mpc3": float(resolved["tracers"][tracer]["fkp_p0_h3_mpc3"]),
        },
    )


def stable_measurement_subset(
    catalog: Mapping[str, Any],
    *,
    tracer: str,
    bin_name: str,
    redshift_column: str,
    config: Mapping[str, Any],
) -> Catalog:
    """Subset by the explicit role-specific redshift while preserving row identity.

    Membership is deliberately fixed by the catalogue's unshifted observed
    ``Z`` values. Reconstruction changes only Cartesian positions, so the same
    source identities are measured before and after shifting even when a shifted
    Cartesian position would imply a radial coordinate beyond the original bin.
    Configuration is required because it supplies both the registered bin and
    the stable integer identity-column name. The function fails closed if that
    column is not a non-negative integer type, if the bin subset is empty, or if
    identities repeat; an empty bin aborts the cell rather than publishing a
    degenerate product.
    """
    record = measurement_bin(tracer, bin_name, config)
    if redshift_column not in catalog:
        raise ValueError(f"measurement catalogue omits required {redshift_column} coordinate")
    result = filter_redshift(
        catalog,
        float(record["zmin"]),
        float(record["zmax"]),
        z_col=redshift_column,
    )
    identity_column = str(config["selection"]["stable_randomness"]["identity_column"])
    validate_catalog(result, (identity_column,))
    raw_identities = np.asarray(result[identity_column])
    if raw_identities.dtype.kind not in "iu" or (
        raw_identities.dtype.kind == "i" and np.any(raw_identities < 0)
    ):
        raise ValueError("measurement source identities must be non-negative integers")
    identities = raw_identities.astype(np.uint64, copy=False)
    if identities.size == 0 or len(np.unique(identities)) != len(identities):
        raise ValueError("measurement subset must be non-empty with unique source identities")
    return result


# Xi is authoritative; P(k) is a required secondary verification vector.


def _parent_sample_name(tracer, condition, config):
    reconstruction = config["measurement"]["reconstruction"]
    return final_sample_identity(
        tracer,
        condition,
        reconstruction["allowed_conventions"][0],
        reconstruction["smoothing_values_mpc_h"][0],
    )


def _reservoir_record(records, tracer, condition, mock_id, cap, config):
    sources = [
        record
        for record in records
        if record["role"] == "reservoir" and record["mock_id"] == mock_id and record["cap"] == cap
    ]
    if len(sources) != 1:
        raise ValueError("missing or duplicate same-tracer reservoir")
    return {**sources[0], "sample_identity": _parent_sample_name(tracer, condition, config)}


def random_shell_accumulator(randoms, *, tracer, cap, cosmology, zmin, zmax, config):
    """Pool the squared weighted density with the cap's solid angle included once."""
    width = config["selection"]["radial_randoms"]["support_bin_width_redshift"]
    edges = np.arange(zmin, zmax + 0.5 * width, width)
    edges[-1] = zmax
    z = np.asarray(randoms["Z"], dtype=float)
    inside = (z >= zmin) & (z < zmax)
    weights = measurement_weight(randoms, config["measurement"]["weights"]["correlation_function"])
    counts = np.histogram(z[inside], bins=edges)[0]
    weight_sum = np.histogram(z[inside], bins=edges, weights=weights[inside])[0]
    area = footprint_area_deg2(tracer, cap, config)
    volumes = comoving_shell_volume(cosmology, edges, area)
    distance = np.asarray(cosmology.comoving_radial_distance(edges))
    centres = 0.5 * (edges[:-1] + edges[1:])
    density = np.divide(weight_sum, volumes, out=np.zeros_like(volumes), where=volumes > 0)
    return weighted_random_zeff_accumulator(
        centres,
        cosmology.comoving_radial_distance(centres),
        density,
        np.diff(distance),
        np.where(counts > 0, area, 0.0),
        positions=config["measurement"]["reconstruction"]["effective_redshift"]["positions"],
    )


def _prepared_parent(
    raw,
    preparation,
    *,
    tracer,
    condition,
    mock_id,
    cap,
    cosmology,
    config,
    parallel=None,
    input_cache=None,
):
    density = config["selection"].get("density_run")
    if (
        density is not None
        and tracer == "LRG"
        and condition in config["selection"]["veff_matched_density"]["tiers"]
    ):
        # The bound density handoff supplies exact matched counts for its tiers, not a new
        # target nbar; every other tier keeps the primary's targets and frozen counts.
        config = deepcopy(config)
        for tier in config["selection"]["tiers"]:
            if tier["label"] == condition:
                tier["target_nbar_h3_mpc3"] = None
    group = preparation["conditions"][condition][cap]
    reservoir_record = _reservoir_record(
        preparation["sources"], tracer, condition, mock_id, cap, config
    )
    data, randoms, diagnostics = prepare_tracer_field(
        raw["data"],
        raw["random"],
        tracer=tracer,
        tier_label=condition,
        cap=cap,
        cosmology=cosmology,
        radial_artifact=group["radial"],
        sample_identity=group["radial"]["sample_identity"],
        frozen_counts_by_bin=group["counts"],
        reservoir=raw["reservoir"],
        reservoir_record=reservoir_record,
        reservoir_sample_identity=reservoir_record["sample_identity"],
        config=config,
        parallel=parallel,
        input_cache=input_cache,
    )
    data = add_cartesian_positions(
        data, cosmology, z_col=config["selection"]["stable_randomness"]["observed_redshift_column"]
    )
    randoms = add_cartesian_positions(randoms, cosmology, z_col="Z")
    return data, randoms, diagnostics


def _input_progress(tracer, phase, **details):
    print(
        json.dumps(
            {
                "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "stage": "inputs",
                "tracer": tracer,
                "phase": phase,
                **details,
            }
        ),
        flush=True,
    )


def select_stage(run, unit, destination, *, cpus=1):
    """Use one bounded pool for input row kernels; retain serial catalogue reductions."""
    if type(cpus) is not int or cpus < 1:
        raise ValueError("input preparation requires a positive CPU count")
    stage = (
        _select_transfer_stage
        if "transfer_run" in run["config"]["measurement"]["reconstruction"]
        else _select_stage
    )
    if cpus == 1:
        return stage(run, unit, destination)
    from joblib import Parallel, parallel_config

    # Kernels hash short Python tokens and need separate interpreters. Only small
    # required-column chunks cross process boundaries. No worker loads a full cap.
    with parallel_config(backend="loky", inner_max_num_threads=1):
        with Parallel(
            n_jobs=cpus,
            return_as="generator",
            batch_size=1,
            pre_dispatch="2*n_jobs",
            max_nbytes=None,
        ) as parallel:
            return stage(run, unit, destination, parallel=parallel)


def _exclusive_parent(raw, sources, *, tracer, mock_id, cap, config):
    """Merge physical parent identities before assignment or any added redshift error."""
    data = normalize_weights(raw["data"])
    if raw["reservoir"] is None:
        raise ValueError("exclusive selection requires registered exterior support")
    design = config["measurement"]["reconstruction"]["exploratory_transfer"]
    label = max(
        design["class_labels"], key=lambda key: tier_spec(key, config)["sigma_z_over_one_plus_z"]
    )
    record = _reservoir_record(sources, tracer, label, mock_id, cap, config)
    return merge_same_tracer_reservoir(
        data,
        normalize_weights(raw["reservoir"]),
        tracer=tracer,
        expected_sample_identity=record["sample_identity"],
        reservoir_record=record,
        sigma_z_over_one_plus_z=tier_spec(label, config)["sigma_z_over_one_plus_z"],
        config=config,
    )


def _select_transfer_stage(run, unit, destination, *, parallel=None):
    """Freeze exclusive parents, migration, smooth forward randoms and bin-local boxes.

    Three bounded catalogue passes share the allocation ensemble with every strategy.
    Class membership and radial prediction depend on the population, never field source/R.
    """
    config, tracer = run["config"], unit["tracer"]
    identifiers = list(unit["realization_ids"])
    if identifiers != config["analysis"]["mock_ids"]:
        raise ValueError("exclusive allocation requires the complete matched ensemble")
    recon = config["measurement"]["reconstruction"]
    design, radial = recon["exploratory_transfer"], config["selection"]["radial_randoms"]
    labels, caps = design["class_labels"], ("NGC", "SGC")
    controls = config["selection"]["stable_randomness"]
    bins = measurement_bins(tracer, config)
    edges = np.array([bins[0]["zmin"], *[item["zmax"] for item in bins]])
    width = radial["support_bin_width_redshift"]
    observed_edges = np.arange(edges[0], edges[-1] + width / 2, width)
    observed_edges[-1] = edges[-1]
    cosmology = build_mock_cosmology(config)
    sources = read_catalog_registration(run, tracer)["sources"]
    availability, support = {cap: [] for cap in caps}, {cap: [edges[0], edges[-1]] for cap in caps}
    volumes = {
        cap: comoving_shell_volume(cosmology, edges, footprint_area_deg2(tracer, cap, config))
        for cap in caps
    }
    for mock in identifiers:
        for cap in caps:
            raw = _read_parent(
                sources,
                tracer=tracer,
                mock_id=mock,
                cap=cap,
                config=config,
                include_randoms=False,
                parallel=parallel,
            )
            parent, _ = _exclusive_parent(
                raw, sources, tracer=tracer, mock_id=mock, cap=cap, config=config
            )
            z = np.asarray(parent["Z"])
            inside = (z >= edges[0]) & (z < edges[-1])
            availability[cap].append(np.histogram(z[inside], edges)[0])
            support[cap] = [
                min(support[cap][0], float(z.min())),
                max(support[cap][1], float(z.max())),
            ]
    budgets = {}
    parent_edges = {}
    for cap in caps:
        desired = [
            [
                tier_spec(label, config)["target_nbar_h3_mpc3"][item["name"]] * volume
                for item, volume in zip(bins, volumes[cap])
            ]
            for label in labels
        ]
        budgets[cap] = joint_class_budget(desired, availability[cap])
        # Anchor to the configured observed grid; include every eligible exterior row.
        lo = edges[0] + np.floor((support[cap][0] - edges[0]) / width) * width
        hi = edges[-1] + (np.floor((support[cap][1] - edges[-1]) / width) + 1) * width
        parent_edges[cap] = np.arange(lo, hi + width / 2, width)
    histograms = {(label, cap): [] for label in labels for cap in caps}
    measured = {(label, cap): [] for label in labels for cap in caps}
    observed_histograms = {(label, cap): [] for label in labels for cap in caps}
    diagnostics = {(label, cap): [] for label in labels for cap in caps}

    def assign(raw, mock, cap):
        parent, support_record = _exclusive_parent(
            raw, sources, tracer=tracer, mock_id=mock, cap=cap, config=config
        )
        assignment, record = assign_exclusive_classes(
            parent,
            class_labels=labels,
            shell_edges=edges,
            allocated_counts=budgets[cap]["allocated_parent_counts"],
            seed=controls["seed"],
            config=config,
        )
        targets, migration = select_exclusive_targets(
            parent, assignment, class_labels=labels, tracer=tracer, config=config
        )
        return parent, targets, record, migration, support_record

    for mock in identifiers:
        for cap in caps:
            raw = _read_parent(
                sources,
                tracer=tracer,
                mock_id=mock,
                cap=cap,
                config=config,
                include_randoms=False,
                parallel=parallel,
            )
            parent, targets, assignment, migration, support_record = assign(raw, mock, cap)
            shell = np.clip(np.searchsorted(edges, parent["Z"], side="right") - 1, 0, len(bins) - 1)
            for i, label in enumerate(labels):
                probabilities = np.asarray(assignment["class_probabilities"])[i, shell]
                histograms[label, cap].append(
                    np.histogram(parent["Z"], parent_edges[cap], weights=probabilities)[0]
                )
                measured[label, cap].append(migration[label]["measured_final_counts"])
                observed_histograms[label, cap].append(
                    np.histogram(
                        targets[label][controls["observed_redshift_column"]], observed_edges
                    )[0].tolist()
                )
                diagnostics[label, cap].append(
                    {
                        "realization_id": mock,
                        "assignment": assignment,
                        "migration": migration[label],
                        "reservoir": support_record,
                    }
                )
    preparation = {
        "tracer": tracer,
        "realization_ids": identifiers,
        "sources": sources,
        "population": design["population"],
        "allocation": budgets,
        "conditions": {label: {} for label in labels},
        "geometry": {},
        "effective_redshifts": {},
        "catalogues": [],
    }
    for i, label in enumerate(labels):
        for cap in caps:
            identity = f"{tracer}|exclusive|{label}|cap={cap}"
            model = fit_smooth_radial_model(
                histograms[label, cap],
                parent_edges[cap],
                calibration_mock_ids=identifiers,
                smoothing_method=radial["frozen_method"],
                bandwidth=radial["frozen_bandwidth"],
                residual_weighting=radial["frozen_residual_weighting"],
                sample_identity=identity + "|parent",
            )
            predicted = forward_radial_model(
                model,
                observed_edges=observed_edges,
                sigma0=tier_spec(label, config)["sigma_z_over_one_plus_z"],
                sample_identity=identity,
            )
            final_counts = np.asarray(measured[label, cap])
            if np.any(final_counts <= 0):
                raise ValueError(
                    f"exclusive {tracer} {label} {cap} has an empty final bin; review attainable design"
                )
            mean_counts = final_counts.mean(axis=0)
            fine_centres = (observed_edges[:-1] + observed_edges[1:]) / 2
            prediction_by_bin = np.histogram(
                fine_centres, edges, weights=predicted["predicted_final_counts"]
            )[0]
            preparation["conditions"][label][cap] = {
                "counts": {item["name"]: float(n) for item, n in zip(bins, mean_counts)},
                "volumes_mpc_h3": {item["name"]: float(v) for item, v in zip(bins, volumes[cap])},
                "measured_final_counts_by_mock": final_counts.tolist(),
                "radial": predicted,
                "observed_counts_by_mock": observed_histograms[label, cap],
                "selection_diagnostics_by_mock": diagnostics[label, cap],
                "densities": {
                    item["name"]: {
                        "forecast_h3_mpc3": tier_spec(label, config)["target_nbar_h3_mpc3"][
                            item["name"]
                        ],
                        "attainable_parent_h3_mpc3": budgets[cap]["allocated_parent_counts"][i][j]
                        / float(volumes[cap][j]),
                        "predicted_final_h3_mpc3": float(prediction_by_bin[j] / volumes[cap][j]),
                        "measured_final_h3_mpc3": float(mean_counts[j] / volumes[cap][j]),
                    }
                    for j, item in enumerate(bins)
                },
            }
    envelopes = {(cap, item["name"]): [] for cap in caps for item in bins}
    zeff = {
        (label, name): []
        for label in labels
        for name in ["parent", *[item["name"] for item in bins]]
    }
    products = []
    for mock in identifiers:
        for cap in caps:
            raw = _read_parent(
                sources, tracer=tracer, mock_id=mock, cap=cap, config=config, parallel=parallel
            )
            _, targets, _, _, _ = assign(raw, mock, cap)
            input_cache = _InputPreparationCache(config, parallel=parallel)
            for label in labels:
                group = preparation["conditions"][label][cap]
                # Common ensemble radial density, shared by galaxies/randoms and strategies.
                attained = {
                    name: {
                        "attained_nbar_h3_mpc3": group["densities"][name]["measured_final_h3_mpc3"]
                    }
                    for name in group["counts"]
                }
                data = apply_attained_fkp(
                    targets[label],
                    tracer=tracer,
                    attained_by_bin=attained,
                    redshift_column=controls["observed_redshift_column"],
                    config=config,
                )
                randoms, random_record = matching_exclusive_randoms(
                    raw["random"],
                    group["radial"],
                    data_count=validate_catalog(data),
                    expected_sample_identity=group["radial"]["sample_identity"],
                    tracer=tracer,
                    attained_by_bin=attained,
                    config=config,
                    parallel=parallel,
                    input_cache=input_cache,
                )
                data = add_cartesian_positions(
                    data, cosmology, z_col=controls["observed_redshift_column"]
                )
                # Matching returns private selected arrays. Retain their ownership
                # while adding the same Cartesian columns used by the public helper.
                randoms.update(
                    zip(
                        ("X", "Y", "Z_CART"),
                        ra_dec_z_to_xyz(randoms["RA"], randoms["DEC"], randoms["Z"], cosmology),
                    )
                )
                group["selection_diagnostics_by_mock"][identifiers.index(mock)][
                    "randoms"
                ] = random_record
                moments = {}
                for item in bins:
                    for catalogue, column in (
                        (data, controls["observed_redshift_column"]),
                        (randoms, "Z"),
                    ):
                        subset = stable_measurement_subset(
                            catalogue,
                            tracer=tracer,
                            bin_name=item["name"],
                            redshift_column=column,
                            config=config,
                        )
                        xyz = np.column_stack([subset[key] for key in ("X", "Y", "Z_CART")])
                        envelopes[cap, item["name"]].extend([xyz.min(axis=0), xyz.max(axis=0)])
                        if column == controls["observed_redshift_column"]:
                            weights = reconstruction_weight(
                                subset, config["measurement"]["weights"]["reconstruction"]
                            )
                            moments[item["name"]] = {
                                "sum_w": float(weights.sum()),
                                "sum_w2": float(np.square(weights).sum()),
                                "count": len(weights),
                            }
                group["selection_diagnostics_by_mock"][identifiers.index(mock)][
                    "reconstruction_weight_moments"
                ] = moments
                for item in [{"name": "parent", "zmin": edges[0], "zmax": edges[-1]}, *bins]:
                    zeff[label, item["name"]].append(
                        random_shell_accumulator(
                            randoms,
                            tracer=tracer,
                            cap=cap,
                            cosmology=cosmology,
                            zmin=item["zmin"],
                            zmax=item["zmax"],
                            config=config,
                        )
                    )
                # Selection is the canonical full allocation cohort, also for a pilot.
                # Reconstruction/measurement workload selectors limit computation only.
                if mock in identifiers:
                    for role, catalogue in (("data", data), ("random", randoms)):
                        path = f"{label}--mock-{mock}--{cap}--{role}.npz"
                        identity = catalogue_identity(
                            run,
                            tracer=tracer,
                            condition=label,
                            mock_id=mock,
                            cap=cap,
                            role=role,
                            stage="select",
                        )
                        products.extend(
                            save_catalogue(destination, path, catalogue, identity=identity)
                        )
                        preparation["catalogues"].append({"path": path, "identity": identity})
                # Retained rows are already serialized or reduced to small diagnostics.
                del data, randoms, catalogue, subset, xyz
            # Avoid retaining the previous cap's large input/cache during the next read.
            del raw, targets, input_cache
    for (cap, name), corners in envelopes.items():
        corners = np.asarray(corners)
        preparation["geometry"].setdefault(cap, {})[name] = {
            **size_box_from_extent(
                corners.min(axis=0),
                corners.max(axis=0),
                cell_size_mpc_h=recon["cell_size_mpc_h"],
                box_padding=recon["box_padding"],
            ),
            "family": reconstruction_mask_family(
                tracer=tracer,
                cap=cap,
                smoothing_radius_mpc_h=recon["smoothing_values_mpc_h"][0],
                measurement_bin=name,
            ),
        }
    for (label, name), records in zeff.items():
        numerator, denominator = (
            sum(record[key] for record in records) for key in ("numerator", "denominator")
        )
        if denominator <= 0 or not np.isfinite(numerator + denominator):
            raise ValueError("exclusive selection has no finite effective redshift")
        value = numerator / denominator
        preparation["effective_redshifts"].setdefault(label, {})[name] = {
            "z_eff": value,
            "growth_rate": exact_growth_rate(value, cosmology),
            "numerator": numerator,
            "denominator": denominator,
        }
    (destination / "preparation.json").write_text(
        json.dumps(preparation, indent=2, allow_nan=False)
    )
    return {
        "products": [{"path": "preparation.json", "format": "json"}, *products],
        "tracer": tracer,
        "realization_ids": identifiers,
    }


def _select_stage(run, unit, destination, *, parallel=None):
    """Prepare complete-ensemble counts, radial targets, redshifts and common parent boxes.

    Three bounded passes retain only compact counts or one mock/cap's catalogues in memory.
    The final pass saves selected catalogues without repeating selection or changing reductions.
    """
    config, tracer = run["config"], unit["tracer"]
    if "density_run" in config["selection"]:
        from .config.density import density_preparation

        preparation = density_preparation(config, tracer)
        sources = read_catalog_registration(run, tracer)["sources"]
        if not same_identity(sources, preparation["sources"]):
            raise ValueError("density successor changed the registered catalogue inputs")
        return _materialize_core_selection(run, unit, destination, preparation, parallel=parallel)
    identifiers = list(unit["realization_ids"])
    if identifiers != config["analysis"]["mock_ids"]:
        raise ValueError("input preparation requires the complete configured mock ensemble")
    # The workload selects solves, not geometry: every family includes all configured tiers.
    conditions = [tier["label"] for tier in config["selection"]["tiers"]]
    cosmology = build_mock_cosmology(config)
    field = reconstruction_field(tracer, config)
    bins = measurement_bins(tracer, config)
    bin_edges = [bins[0]["zmin"], *[item["zmax"] for item in bins]]
    radial = config["selection"]["radial_randoms"]
    width = radial["support_bin_width_redshift"]
    z_edges = np.arange(field["zmin"], field["zmax"] + 0.5 * width, width)
    z_edges[-1] = field["zmax"]
    sources = read_catalog_registration(run, tracer)["sources"]
    if "experiment" in config["selection"]["boundary_support"]:
        if "boundary_reference" not in run:
            raise ValueError("boundary selection requires its resolved frozen reference")
        reference_sources = read_preparation(run["boundary_reference"], tracer)["sources"]
        validate_boundary_input_sources(reference_sources, sources, tracer=tracer, config=config)
    preparation = {
        "tracer": tracer,
        "realization_ids": identifiers,
        "sources": sources,
        "conditions": {},
        "geometry": {},
        "effective_redshifts": {},
    }
    feasible = {(condition, cap): [] for condition in conditions for cap in ("NGC", "SGC")}
    boundaries = {(condition, cap): [] for condition in conditions for cap in ("NGC", "SGC")}
    # First pass: attainability after exactly the same injection/smearing as measurement.
    for mock_id in identifiers:
        for cap in ("NGC", "SGC"):
            _input_progress(tracer, "counts", mock_id=mock_id, cap=cap, status="started")
            raw = _read_parent(
                sources,
                tracer=tracer,
                mock_id=mock_id,
                cap=cap,
                config=config,
                include_randoms=False,
                parallel=parallel,
            )
            input_cache = _InputPreparationCache(config, parallel=parallel)
            for condition in conditions:
                data, diagnostics = prepare_data_for_selection(
                    raw["data"],
                    tracer=tracer,
                    tier_label=condition,
                    cap=cap,
                    reservoir=raw["reservoir"],
                    config=config,
                    parallel=parallel,
                    input_cache=input_cache,
                    reservoir_record=_reservoir_record(
                        sources, tracer, condition, mock_id, cap, config
                    ),
                )
                column = config["selection"]["stable_randomness"]["observed_redshift_column"]
                feasible[condition, cap].append(np.histogram(data[column], bins=bin_edges)[0])
                boundaries[condition, cap].append(
                    {"realization_id": mock_id, **diagnostics["boundary_counts"]}
                )
            _input_progress(tracer, "counts", mock_id=mock_id, cap=cap, status="completed")
            del raw, data, input_cache
    for condition in conditions:
        preparation["conditions"][condition] = {}
        targets = tier_spec(condition, config)["target_nbar_h3_mpc3"]
        for cap in ("NGC", "SGC"):
            volume = comoving_shell_volume(
                cosmology, bin_edges, footprint_area_deg2(tracer, cap, config)
            )
            counts = np.min(feasible[condition, cap], axis=0)
            if targets is not None:
                requested = np.floor(
                    [targets[item["name"]] * value for item, value in zip(bins, volume)]
                )
                counts = np.minimum(counts, requested).astype(int)
            if np.any(counts <= 0):
                raise ValueError(f"{tracer} {condition} {cap} has an empty common sample")
            preparation["conditions"][condition][cap] = {
                "counts": {item["name"]: int(n) for item, n in zip(bins, counts)},
                "volumes_mpc_h3": {item["name"]: float(v) for item, v in zip(bins, volume)},
                "attainable_counts_by_mock": np.asarray(feasible[condition, cap]).tolist(),
                "boundary_counts_by_mock": boundaries[condition, cap],
            }
    histograms = {(condition, cap): [] for condition in conditions for cap in ("NGC", "SGC")}
    # Second pass: each shared smooth radial target uses all selected realizations.
    for mock_id in identifiers:
        for cap in ("NGC", "SGC"):
            _input_progress(tracer, "radial", mock_id=mock_id, cap=cap, status="started")
            raw = _read_parent(
                sources,
                tracer=tracer,
                mock_id=mock_id,
                cap=cap,
                config=config,
                include_randoms=False,
                parallel=parallel,
            )
            input_cache = _InputPreparationCache(config, parallel=parallel)
            for condition in conditions:
                data, _ = prepare_data_for_selection(
                    raw["data"],
                    tracer=tracer,
                    tier_label=condition,
                    cap=cap,
                    reservoir=raw["reservoir"],
                    config=config,
                    parallel=parallel,
                    input_cache=input_cache,
                    reservoir_record=_reservoir_record(
                        sources, tracer, condition, mock_id, cap, config
                    ),
                )
                group = preparation["conditions"][condition][cap]
                selected, _ = select_exact_attainable(
                    data,
                    tracer=tracer,
                    tier_label=condition,
                    shell_volumes_mpc_h3=group["volumes_mpc_h3"],
                    frozen_counts_by_bin=group["counts"],
                    config=config,
                )
                column = config["selection"]["stable_randomness"]["observed_redshift_column"]
                histograms[condition, cap].append(np.histogram(selected[column], bins=z_edges)[0])
            _input_progress(tracer, "radial", mock_id=mock_id, cap=cap, status="completed")
            del raw, data, selected, input_cache
    for (condition, cap), histogram in histograms.items():
        model = fit_smooth_radial_model(
            histogram,
            z_edges,
            calibration_mock_ids=identifiers,
            smoothing_method=radial["frozen_method"],
            bandwidth=radial["frozen_bandwidth"],
            residual_weighting=radial["frozen_residual_weighting"],
            sample_identity=_parent_sample_name(tracer, condition, config) + "|cap=" + cap,
        )
        if not model["convergence_evidence"]["converged"]:
            raise ValueError("radial spline did not realize the configured smoothing prescription")
        preparation["conditions"][condition][cap]["radial"] = model
    return _materialize_core_selection(run, unit, destination, preparation, parallel=parallel)


def _materialize_core_selection(run, unit, destination, preparation, *, parallel=None):
    """Materialize a frozen selection and derive geometry/growth from its selected rows."""
    config, tracer = run["config"], unit["tracer"]
    identifiers = list(unit["realization_ids"])
    if (
        identifiers != config["analysis"]["mock_ids"]
        or preparation["realization_ids"] != identifiers
    ):
        raise ValueError("selection materialization requires the complete calibration cohort")
    conditions = [tier["label"] for tier in config["selection"]["tiers"]]
    cosmology = build_mock_cosmology(config)
    field, bins = reconstruction_field(tracer, config), measurement_bins(tracer, config)
    sources = preparation["sources"]
    products, saved_catalogues = [], []
    for groups in preparation["conditions"].values():
        for group in groups.values():
            group["selection_diagnostics_by_mock"] = []
    envelopes = {cap: [] for cap in ("NGC", "SGC")}
    bin_envelopes = {cap: {item["name"]: [] for item in bins} for cap in envelopes}
    redshifts = {
        (condition, name): []
        for condition in conditions
        for name in ["parent", *[item["name"] for item in bins]]
    }
    # Third pass: geometry and growth use the actual selected/thinned parent catalogues.
    for mock_id in identifiers:
        for cap in ("NGC", "SGC"):
            _input_progress(tracer, "geometry_read", mock_id=mock_id, cap=cap, status="started")
            raw = _read_parent(
                sources, tracer=tracer, mock_id=mock_id, cap=cap, config=config, parallel=parallel
            )
            _input_progress(tracer, "geometry_read", mock_id=mock_id, cap=cap, status="completed")
            input_cache = _InputPreparationCache(config, parallel=parallel)
            for condition in conditions:
                data, randoms, diagnostics = _prepared_parent(
                    raw,
                    preparation,
                    tracer=tracer,
                    condition=condition,
                    mock_id=mock_id,
                    cap=cap,
                    cosmology=cosmology,
                    config=config,
                    parallel=parallel,
                    input_cache=input_cache,
                )
                preparation["conditions"][condition][cap].setdefault(
                    "selection_diagnostics_by_mock", []
                ).append({"realization_id": mock_id, **diagnostics})
                for catalogue in (data, randoms):
                    xyz = np.column_stack([catalogue[name] for name in ("X", "Y", "Z_CART")])
                    envelopes[cap].extend([xyz.min(axis=0), xyz.max(axis=0)])
                if per_bin_reconstruction(config):
                    _extend_bin_envelopes(bin_envelopes[cap], data, randoms, tracer, config)
                for item in [{"name": "parent", **field}, *bins]:
                    redshifts[condition, item["name"]].append(
                        random_shell_accumulator(
                            randoms,
                            tracer=tracer,
                            cap=cap,
                            cosmology=cosmology,
                            zmin=item["zmin"],
                            zmax=item["zmax"],
                            config=config,
                        )
                    )
                _input_progress(
                    tracer,
                    "geometry",
                    mock_id=mock_id,
                    cap=cap,
                    condition=condition,
                    status="completed",
                )
                workload = config["analysis"]["workloads"][run["metadata"]["workload"]]
                if condition in workload["sigma_z_tiers"] and mock_id in workload["mock_ids"]:
                    for role, catalogue in (("data", data), ("random", randoms)):
                        name = f"{condition}--mock-{mock_id}--{cap}--{role}.npz"
                        identity = catalogue_identity(
                            run,
                            tracer=tracer,
                            condition=condition,
                            mock_id=mock_id,
                            cap=cap,
                            role=role,
                            stage="select",
                        )
                        products.extend(
                            save_catalogue(destination, name, catalogue, identity=identity)
                        )
                        saved_catalogues.append({"path": name, "identity": identity})
                del data, randoms, catalogue, xyz
            del raw, input_cache
    reconstruction = config["measurement"]["reconstruction"]
    for cap, corners in envelopes.items():
        corners = np.asarray(corners)
        preparation["geometry"][cap] = {
            **size_box_from_extent(
                corners.min(axis=0),
                corners.max(axis=0),
                cell_size_mpc_h=reconstruction["cell_size_mpc_h"],
                box_padding=reconstruction["box_padding"],
            ),
            "family": reconstruction_mask_family(
                tracer=tracer,
                cap=cap,
                smoothing_radius_mpc_h=reconstruction["smoothing_values_mpc_h"][0],
            ),
        }
    if per_bin_reconstruction(config):
        _freeze_core_bin_geometry(preparation, bin_envelopes, tracer, cosmology, config)
    from .clustering import derive_power_spectrum_geometry

    preparation["power_geometry"] = derive_power_spectrum_geometry(
        np.concatenate([np.asarray(value) for value in envelopes.values()]),
        tracer=tracer,
        config=config,
    )
    for (condition, name), accumulators in redshifts.items():
        numerator = sum(record["numerator"] for record in accumulators)
        denominator = sum(record["denominator"] for record in accumulators)
        if not np.isfinite(numerator + denominator) or denominator <= 0:
            raise ValueError("selected randoms do not define a finite effective redshift")
        zeff = numerator / denominator
        preparation["effective_redshifts"].setdefault(condition, {})[name] = {
            "z_eff": zeff,
            "growth_rate": exact_growth_rate(zeff, cosmology),
            "numerator": numerator,
            "denominator": denominator,
        }
    if per_bin_reconstruction(config) and "transfer_run" not in reconstruction:
        _validate_bin_redshifts(preparation, tracer, config, cosmology)
    preparation["catalogues"] = saved_catalogues
    (destination / "preparation.json").write_text(
        json.dumps(preparation, indent=2, allow_nan=False)
    )
    return {
        "products": [{"path": "preparation.json", "format": "json"}, *products],
        "tracer": tracer,
        "realization_ids": identifiers,
    }


def _extend_bin_envelopes(envelopes, data, randoms, tracer, config):
    """Accumulate actual selected extents, independently of workload selectors."""
    observed = config["selection"]["stable_randomness"]["observed_redshift_column"]
    for name, corners in envelopes.items():
        for catalogue, column in ((data, observed), (randoms, "Z")):
            subset = stable_measurement_subset(
                catalogue, tracer=tracer, bin_name=name, redshift_column=column, config=config
            )
            xyz = np.column_stack([subset[key] for key in ("X", "Y", "Z_CART")])
            if np.any(~np.isfinite(xyz)):
                raise ValueError("reconstruction partition contains nonfinite positions")
            corners.extend([xyz.min(axis=0), xyz.max(axis=0)])


def _freeze_core_bin_geometry(preparation, envelopes, tracer, cosmology, config):
    recon = config["measurement"]["reconstruction"]
    preparation["reconstruction_scope"] = recon["scope"]
    preparation["windows"] = {
        item["name"]: reconstruction_window_bounds(
            tracer,
            measurement_bin=item["name"],
            cosmology=cosmology,
            smoothing_radius_mpc_h=recon["smoothing_values_mpc_h"][0],
            config=config,
        )
        for item in measurement_bins(tracer, config)
    }
    preparation["geometry"] = {}
    for cap, bins in envelopes.items():
        preparation["geometry"][cap] = {}
        for name, corners in bins.items():
            corners = np.asarray(corners)
            expected = 4 * len(config["selection"]["tiers"]) * len(preparation["realization_ids"])
            if corners.shape != (expected, 3):
                raise ValueError("bin geometry does not cover all conditions and calibration mocks")
            preparation["geometry"][cap][name] = {
                **size_box_from_extent(
                    corners.min(axis=0),
                    corners.max(axis=0),
                    cell_size_mpc_h=recon["cell_size_mpc_h"],
                    box_padding=recon["box_padding"],
                ),
                "family": reconstruction_mask_family(
                    tracer=tracer,
                    cap=cap,
                    measurement_bin=name,
                    smoothing_radius_mpc_h=recon["smoothing_values_mpc_h"][0],
                ),
                "conditions": [tier["label"] for tier in config["selection"]["tiers"]],
                "calibration_mock_ids": list(preparation["realization_ids"]),
            }


def _validate_bin_redshifts(preparation, tracer, config, cosmology):
    tolerances = config["measurement"]["reconstruction"]["growth"]["consistency_tolerances"]
    for tier in config["selection"]["tiers"]:
        for item in measurement_bins(tracer, config):
            value = preparation["effective_redshifts"][tier["label"]][item["name"]]
            z, growth, numerator, denominator = [
                float(value[key]) for key in ("z_eff", "growth_rate", "numerator", "denominator")
            ]
            if (
                not np.all(np.isfinite([z, growth, numerator, denominator]))
                or denominator <= 0
                or not item["zmin"] <= z < item["zmax"]
                or not np.isclose(
                    z,
                    numerator / denominator,
                    rtol=tolerances["relative"],
                    atol=tolerances["absolute"],
                )
                or not np.isclose(
                    growth,
                    exact_growth_rate(z, cosmology),
                    rtol=tolerances["relative"],
                    atol=tolerances["absolute"],
                )
            ):
                raise ValueError("invalid bin effective-redshift or growth record")


def read_preparation(run, tracer):
    # Preserve readers for immutable native products from the previous interface.
    stage = "select" if run["metadata"].get("interface_version") == 2 else "inputs"
    path = run["root"] / stage / "units" / tracer
    import yaml

    result = yaml.safe_load((path / "result.yaml").read_text())
    if (
        any(
            result.get(key) != run["metadata"][key]
            for key in ("run_name", "code_commit", "analysis_config_sha256")
        )
        or result.get("stage") != stage
        or not same_identity(
            result.get("unit"),
            {
                "unit_id": tracer,
                "tracer": tracer,
                "realization_ids": run["config"]["analysis"]["mock_ids"],
            },
        )
    ):
        raise ValueError("preparation belongs to a different snapshot or tracer")
    record = json.loads((path / "preparation.json").read_text())
    if record["tracer"] != tracer or not same_identity(
        record["realization_ids"], run["config"]["analysis"]["mock_ids"]
    ):
        raise ValueError("preparation does not cover this tracer's full ensemble")
    config = run["config"]
    recon = config["measurement"]["reconstruction"]
    if recon["scope"] == "separate_measurement_bins" and "transfer_run" not in recon:
        if record.get("reconstruction_scope") != recon["scope"]:
            raise ValueError("per-bin preparation cannot use a broad-field geometry")
        bins = {item["name"] for item in measurement_bins(tracer, config)}
        if set(record.get("windows", {})) != bins or set(record["geometry"]) != {"NGC", "SGC"}:
            raise ValueError("per-bin preparation lacks its bin windows or caps")
        for name, window in record["windows"].items():
            item = measurement_bin(tracer, name, config)
            if (
                window.get("measurement_bin") != name
                or window.get("reconstruction_scope") != recon["scope"]
                or window.get("window_redshift_range") != [item["zmin"], item["zmax"]]
                or window.get("radial_padding_mpc_h") != 0.0
            ):
                raise ValueError("reconstruction window names a missing or foreign bin")
        for cap, geometry in record["geometry"].items():
            if set(geometry) != bins:
                raise ValueError("per-bin preparation lacks the configured geometry family")
            for name, value in geometry.items():
                expected = reconstruction_mask_family(
                    tracer=tracer,
                    cap=cap,
                    measurement_bin=name,
                    smoothing_radius_mpc_h=recon["smoothing_values_mpc_h"][0],
                )
                if (
                    value["family"] != expected
                    or value.get("conditions") != [t["label"] for t in config["selection"]["tiers"]]
                    or value.get("calibration_mock_ids") != config["analysis"]["mock_ids"]
                ):
                    raise ValueError("bin geometry has a different family or calibration cohort")
    return record


def selected_catalogue_entry(run, unit, cap, role, *, preparation=None):
    """Return a validated local selected-product locator for reference consumers."""
    if preparation is None:
        preparation = read_preparation(run, unit["tracer"])
    directory = run["root"] / "select" / "units" / unit["tracer"]
    identity = catalogue_identity(
        run,
        tracer=unit["tracer"],
        condition=unit["condition"],
        mock_id=unit["realization_id"],
        cap=cap,
        role=role,
        stage="select",
    )
    entries = [
        item for item in preparation["catalogues"] if same_identity(item["identity"], identity)
    ]
    if len(entries) != 1:
        raise ValueError(f"missing or duplicate selected {role} catalogue")
    path = (directory / entries[0]["path"]).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError("selected catalogue path leaves its unit")
    return {"path": str(path), "identity": identity}


def read_selected_parent(run, unit, cap, *, bin_name=None):
    """Load this parent's saved selection; no resmearing, reselection or geometry refit."""
    preparation = read_preparation(run, unit["tracer"])
    catalogues = []
    for role in ("data", "random"):
        from .catalog import catalogue_partition

        entry = selected_catalogue_entry(run, unit, cap, role, preparation=preparation)
        partition = (
            None
            if bin_name is None
            else catalogue_partition(unit["tracer"], bin_name, role, run["config"])
        )
        catalogues.append(read_catalogue(**entry, partition=partition))
    return (*catalogues, preparation)


SENSITIVITY_WORKLOADS = ("smoothing_scan", "reciso")


def validate_selection_reuse(source, target, tracer):
    """Permit only reconstruction choices and downstream routing to differ among siblings.

    Transfer siblings may differ in field source and radius. Core siblings (the smoothing scan
    and the RecIso check) may differ in radius, convention, the reference-radius damping pin
    and their own workload; selection depends on none of these.
    """
    from copy import deepcopy

    target_recon = target["config"]["measurement"]["reconstruction"]
    if (
        target["config"]["schema_version"] in PER_BIN_SCHEMA_VERSIONS
        and "transfer_run" not in target_recon
    ):
        source_config, target_config = source["config"], target["config"]
        if "transfer_run" in source_config["measurement"]["reconstruction"]:
            raise ValueError("core selection reuse cannot import an exclusive transfer cohort")

        # These are the complete selection dependencies. Fitting choices and the solve
        # topology do not define selected rows; density targets and all streams do. The
        # matched-density block (schema 17) defines the density successor's handoff, not the
        # rows a primary or sensitivity selection retains.
        def selected_rows_definition(config):
            selection = dict(config["selection"])
            selection.pop("veff_matched_density", None)
            return {"physics": config["physics"], "tracers": config["tracers"], **selection}

        if not same_identity(
            selected_rows_definition(source_config), selected_rows_definition(target_config)
        ):
            raise ValueError("selection reuse changes inputs, weights, streams or density")
        for left, right in (
            (source_config["analysis"]["mock_ids"], target_config["analysis"]["mock_ids"]),
            (source_config["measurement"]["weights"], target_config["measurement"]["weights"]),
            (
                source_config["measurement"]["reconstruction"]["effective_redshift"],
                target_recon["effective_redshift"],
            ),
        ):
            if not same_identity(left, right):
                raise ValueError("selection reuse changes weights or calibration cohort")
        left = read_catalog_registration(source, tracer)["sources"]
        right = read_catalog_registration(target, tracer)["sources"]
        if not same_identity(left, right):
            raise ValueError("selection reuse has different registered parent inputs")
        return

    def selection_definition(config):
        value = deepcopy(config)
        reconstruction = value["measurement"]["reconstruction"]
        reconstruction.pop("smoothing_values_mpc_h")
        if "transfer_run" in reconstruction:
            reconstruction["transfer_run"].pop("source_class")
            workload = value["analysis"]["workloads"]["transfer"]
            for key in ("sigma_z_tiers", "reconstruction_states"):
                workload.pop(key)
            return value
        if "sensitivity_run" not in reconstruction:
            raise ValueError("selection reuse is defined for transfer or core sensitivity siblings")
        reconstruction.pop("sensitivity_run")
        reconstruction.pop("allowed_conventions")
        value["fitting"]["damping"].pop("reference_smoothing_mpc_h", None)
        for name in SENSITIVITY_WORKLOADS:
            value["analysis"]["workloads"].pop(name, None)
        return value

    if not same_identity(
        selection_definition(source["config"]), selection_definition(target["config"])
    ):
        raise ValueError("selection reuse changes parent assignment, marks, weights or cohort")
    left = read_catalog_registration(source, tracer)["sources"]
    right = read_catalog_registration(target, tracer)["sources"]
    if not same_identity(left, right):
        raise ValueError("selection reuse has different registered parent inputs")


def _reuse_core_bin_selection(run, unit, destination, *, source, parallel=None):
    """Reference saved partitions; generate missing tiers with their frozen selection."""
    from .catalog import save_catalogue_reference
    from .clustering import derive_power_spectrum_geometry

    config, tracer = run["config"], unit["tracer"]
    preparation = deepcopy(read_preparation(source, tracer))
    if preparation["realization_ids"] != unit["realization_ids"]:
        raise ValueError("selection reuse does not cover the allocation cohort")
    cosmology = build_mock_cosmology(config)
    _validate_bin_redshifts(preparation, tracer, config, cosmology)
    bins = measurement_bins(tracer, config)
    conditions = [tier["label"] for tier in config["selection"]["tiers"]]
    envelopes = {cap: {item["name"]: [] for item in bins} for cap in ("NGC", "SGC")}
    parent_envelopes = []
    accumulators = {(condition, item["name"]): [] for condition in conditions for item in bins}
    workload = config["analysis"]["workloads"][run["metadata"]["workload"]]
    products, entries = [], []
    for mock_id in unit["realization_ids"]:
        for cap in ("NGC", "SGC"):
            raw = input_cache = None
            for condition in conditions:
                selected_unit = {
                    "tracer": tracer,
                    "condition": condition,
                    "realization_id": mock_id,
                }
                bases = {}
                for role in ("data", "random"):
                    identity = catalogue_identity(
                        source,
                        tracer=tracer,
                        condition=condition,
                        mock_id=mock_id,
                        cap=cap,
                        role=role,
                        stage="select",
                    )
                    matches = [
                        entry
                        for entry in preparation["catalogues"]
                        if same_identity(entry["identity"], identity)
                    ]
                    if len(matches) > 1:
                        raise ValueError("selection reuse has duplicate catalogue identities")
                    if matches:
                        bases[role] = selected_catalogue_entry(
                            source, selected_unit, cap, role, preparation=preparation
                        )
                if len(bases) == 2:
                    data, randoms = (read_catalogue(**bases[role]) for role in ("data", "random"))
                else:
                    if raw is None:
                        raw = _read_parent(
                            preparation["sources"],
                            tracer=tracer,
                            mock_id=mock_id,
                            cap=cap,
                            config=source["config"],
                            parallel=parallel,
                        )
                        input_cache = _InputPreparationCache(source["config"], parallel=parallel)
                    data, randoms, _ = _prepared_parent(
                        raw,
                        preparation,
                        tracer=tracer,
                        condition=condition,
                        mock_id=mock_id,
                        cap=cap,
                        cosmology=cosmology,
                        config=source["config"],
                        parallel=parallel,
                        input_cache=input_cache,
                    )
                    # A partially saved pair must agree byte-for-byte in values and order
                    # with regeneration under its unchanged frozen selection.
                    for role, catalogue in (("data", data), ("random", randoms)):
                        if role in bases:
                            original = read_catalogue(**bases[role])
                            if set(original) != set(catalogue) or any(
                                original[key].dtype != catalogue[key].dtype
                                or not np.array_equal(original[key], catalogue[key])
                                for key in original
                            ):
                                raise ValueError(
                                    "reused catalogue differs from its frozen selection"
                                )
                            del original
                _extend_bin_envelopes(envelopes[cap], data, randoms, tracer, config)
                for catalogue in (data, randoms):
                    xyz = np.column_stack([catalogue[key] for key in ("X", "Y", "Z_CART")])
                    parent_envelopes.extend([xyz.min(axis=0), xyz.max(axis=0)])
                for item in bins:
                    accumulators[condition, item["name"]].append(
                        random_shell_accumulator(
                            randoms,
                            tracer=tracer,
                            cap=cap,
                            cosmology=cosmology,
                            zmin=item["zmin"],
                            zmax=item["zmax"],
                            config=config,
                        )
                    )
                # Release each role before the next is written: an alias re-reads its whole
                # base, so holding both roles alongside it would double the peak.
                pair = {"data": data, "random": randoms}
                del data, randoms, catalogue, xyz
                if condition in workload["sigma_z_tiers"] and mock_id in workload["mock_ids"]:
                    for role in ("data", "random"):
                        catalogue = pair.pop(role)
                        identity = catalogue_identity(
                            run,
                            tracer=tracer,
                            condition=condition,
                            mock_id=mock_id,
                            cap=cap,
                            role=role,
                            stage="select",
                        )
                        stem = f"{condition}--mock-{mock_id}--{cap}--{role}"
                        if role in bases:
                            path = stem + ".json"
                            del catalogue
                            products.extend(
                                save_catalogue_reference(
                                    destination, path, identity=identity, base=bases[role]
                                )
                            )
                        else:
                            path = stem + ".npz"
                            products.extend(
                                save_catalogue(destination, path, catalogue, identity=identity)
                            )
                            del catalogue
                        entries.append({"path": path, "identity": identity})
                del pair
            del raw, input_cache
    # Reuse the saved bin values only after validating their complete pooled moments
    # against the actual source arrays and regenerated missing coverage.
    tolerance = config["measurement"]["reconstruction"]["growth"]["consistency_tolerances"]
    for (condition, name), values in accumulators.items():
        saved = preparation["effective_redshifts"][condition][name]
        for key in ("numerator", "denominator"):
            if not np.isclose(
                saved[key],
                sum(value[key] for value in values),
                rtol=tolerance["relative"],
                atol=tolerance["absolute"],
            ):
                raise ValueError("reused bin effective redshift differs from selected randoms")
    _freeze_core_bin_geometry(preparation, envelopes, tracer, cosmology, config)
    preparation["power_geometry"] = derive_power_spectrum_geometry(
        np.asarray(parent_envelopes), tracer=tracer, config=config
    )
    preparation["catalogues"] = entries
    preparation["selection_source"] = {
        "root": str(source["root"]),
        **{
            key: source["metadata"][key]
            for key in ("run_name", "code_commit", "analysis_config_sha256")
        },
    }
    (destination / "preparation.json").write_text(
        json.dumps(preparation, indent=2, allow_nan=False)
    )
    return {
        "products": [{"path": "preparation.json", "format": "json"}, *products],
        "tracer": tracer,
        "realization_ids": unit["realization_ids"],
    }


def reuse_selection_stage(run, unit, destination, *, source, cpus=1):
    """Publish sibling provenance around unchanged canonical arrays; rebuild radius metadata.

    Core per-bin reuse regenerates the tiers the source did not save with the same bounded
    row-kernel pool as a full selection; alias-only reuse needs no pool.
    """
    from copy import deepcopy
    from .catalog import save_catalogue_reference

    if type(cpus) is not int or cpus < 1:
        raise ValueError("selection reuse requires a positive CPU count")
    tracer = unit["tracer"]
    validate_selection_reuse(source, run, tracer)
    recon = run["config"]["measurement"]["reconstruction"]
    if recon["scope"] == "separate_measurement_bins" and "transfer_run" not in recon:
        if cpus == 1:
            return _reuse_core_bin_selection(run, unit, destination, source=source)
        from joblib import Parallel, parallel_config

        with parallel_config(backend="loky", inner_max_num_threads=1):
            with Parallel(
                n_jobs=cpus,
                return_as="generator",
                batch_size=1,
                pre_dispatch="2*n_jobs",
                max_nbytes=None,
            ) as parallel:
                return _reuse_core_bin_selection(
                    run, unit, destination, source=source, parallel=parallel
                )
    preparation = deepcopy(read_preparation(source, tracer))
    if preparation["realization_ids"] != unit["realization_ids"]:
        raise ValueError("selection reuse does not cover the allocation cohort")
    config = run["config"]
    recon = config["measurement"]["reconstruction"]
    transfer = "transfer_run" in recon
    if transfer:
        labels = recon["exploratory_transfer"]["class_labels"]
    else:
        labels = config["analysis"]["workloads"][run["metadata"]["workload"]]["sigma_z_tiers"]
    products, entries = [], []
    for mock in unit["realization_ids"]:
        for label in labels:
            target = {"tracer": tracer, "condition": label, "realization_id": mock}
            for cap in ("NGC", "SGC"):
                for role in ("data", "random"):
                    base = selected_catalogue_entry(source, target, cap, role)
                    identity = catalogue_identity(
                        run,
                        tracer=tracer,
                        condition=label,
                        mock_id=mock,
                        cap=cap,
                        role=role,
                        stage="select",
                    )
                    name = f"{label}--mock-{mock}--{cap}--{role}.json"
                    products.extend(
                        save_catalogue_reference(destination, name, identity=identity, base=base)
                    )
                    entries.append({"path": name, "identity": identity})
    # Geometry extents and ensemble moments are selection invariants. Recreate the
    # radius-dependent family through the same builder as the original selection.
    radius = recon["smoothing_values_mpc_h"][0]
    for cap, geometries in preparation["geometry"].items():
        if not transfer:
            # Core runs solve one common parent mesh per cap.
            geometries["family"] = reconstruction_mask_family(
                tracer=tracer, cap=cap, smoothing_radius_mpc_h=radius
            )
            continue
        for name, geometry in geometries.items():
            geometry["family"] = reconstruction_mask_family(
                tracer=tracer, cap=cap, smoothing_radius_mpc_h=radius, measurement_bin=name
            )
    preparation["catalogues"] = entries
    preparation["selection_source"] = {
        "root": str(source["root"]),
        **{
            key: source["metadata"][key]
            for key in ("run_name", "code_commit", "analysis_config_sha256")
        },
    }
    (destination / "preparation.json").write_text(
        json.dumps(preparation, indent=2, allow_nan=False)
    )
    return {
        "products": [{"path": "preparation.json", "format": "json"}, *products],
        "tracer": tracer,
        "realization_ids": unit["realization_ids"],
    }
