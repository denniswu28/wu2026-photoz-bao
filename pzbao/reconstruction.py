"""Reconstruction geometry, displacement solving and reusable shifted catalogues."""

from __future__ import annotations

import json
import hashlib
import math
import time
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from .catalog import (
    Catalog,
    copy_catalog,
    measurement_bin as configured_measurement_bin,
    measurement_bins,
    validate_catalog,
    reconstruction_weight,
)
from .config import (
    PER_BIN_SCHEMA_VERSIONS,
    load_analysis_config,
    same_identity,
    per_bin_reconstruction,
)

SHIFTED_COLUMNS = ("X_SHIFTED", "Y_SHIFTED", "Z_CART_SHIFTED")


def displacement_field_identity(data, randoms, *, settings, source_identity, code_identity, config):
    """Key a solve by ordered source arrays and all actual solver settings, never its target.

    Mesh geometry must already cover every intended target. A changed target that requires a
    different box therefore changes this key. Code identity includes scratch source changes;
    callers must use the runner's execution identity, not just a dirty tree's base commit.
    """
    required = {
        "bias",
        "growth_rate",
        "smoothing_radius_mpc_h",
        "convention",
        "algorithm",
        "cell_size_mpc_h",
        "iterations",
        "box_padding",
        "resampler",
        "fft_plan",
        "mesh_dtype",
        "threshold_randoms",
        "mask_family",
        "common_geometry",
        "source_visible_real_mesh_count",
        "single_rank_peak_rss_ceiling_gib",
        "wrap_positions",
        "position_type",
        "line_of_sight",
        "identity_column",
        "weight_expression",
        "pyrecon_kw_weights",
    }
    if set(settings) != required or not source_identity or not code_identity:
        raise ValueError("field identity requires complete solve, source and execution identities")
    if settings["wrap_positions"] or settings["convention"] != "recsym":
        raise ValueError("transfer requires unwrapped RecSym")
    identity_column = settings["identity_column"]
    inputs = {}
    for role, catalogue in (("data", data), ("random", randoms)):
        size = validate_catalog(catalogue, (identity_column, "X", "Y", "Z_CART"))
        ids = np.asarray(catalogue[identity_column])
        if not size or not np.issubdtype(ids.dtype, np.integer) or len(np.unique(ids)) != size:
            raise ValueError("field source needs nonempty unique integer identities")
        arrays = {
            identity_column: ids,
            **{name: np.asarray(catalogue[name]) for name in ("X", "Y", "Z_CART")},
            "solve_weights": reconstruction_weight(catalogue, settings["weight_expression"]),
        }
        digest = hashlib.sha256()
        for name, values in arrays.items():
            array = np.ascontiguousarray(values)
            if array.dtype.hasobject or not np.isfinite(array).all():
                raise ValueError("field source arrays must be finite numeric values")
            digest.update(json.dumps([name, array.dtype.str, array.shape]).encode())
            digest.update(memoryview(array).cast("B"))
        inputs[role] = {"rows": size, "sha256": digest.hexdigest()}
    definition = {
        "version": 1,
        "units": "Mpc/h",
        "source_identity": source_identity,
        "source_inputs": inputs,
        "settings": settings,
        "physics": config["physics"],
        "execution_code": code_identity,
    }
    token = json.dumps(definition, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {"key": hashlib.sha256(token.encode()).hexdigest(), "definition": json.loads(token)}


def validate_displacement_field(reconstruction):
    """A source failure blocks every target depending on this live mesh."""
    # Keep the field-validity checks that previously accompanied grid serialization.
    if reconstruction.mpicomm.size != 1 or reconstruction.mpicomm.rank != 0:
        raise ValueError("transfer field application requires one MPI rank")
    if len(reconstruction.mesh_psi) != 3:
        raise ValueError("transfer requires three displacement components")
    shape = tuple(int(value) for value in reconstruction.nmesh)
    for mesh in reconstruction.mesh_psi:
        values = _mesh_array(mesh)
        if values.shape != shape or values.dtype != np.dtype("f8") or not np.isfinite(values).all():
            raise ValueError("displacement mesh has invalid shape, dtype or values")


def apply_displacement_field(reconstruction, data, randoms, *, identity_column):
    """Apply one solved field to fixed targets, using disp+rsd for galaxies and randoms."""
    validate_displacement_field(reconstruction)
    lower = np.asarray(reconstruction.boxcenter) - np.asarray(reconstruction.boxsize) / 2
    upper = lower + np.asarray(reconstruction.boxsize)
    outputs = []
    for catalogue in (data, randoms):
        validate_catalog(catalogue, (identity_column, "X", "Y", "Z_CART"))
        positions = np.column_stack([catalogue[name] for name in ("X", "Y", "Z_CART")])
        if not len(positions) or not np.isfinite(positions).all():
            raise ValueError("transfer target must have finite nonempty positions")
        if np.any(positions < lower) or np.any(positions >= upper):
            raise ValueError("transfer target leaves the solved source geometry")
        shifted = np.asarray(reconstruction.read_shifted_positions(positions, field="disp+rsd"))
        if shifted.shape != positions.shape or not np.isfinite(shifted).all():
            raise ValueError("transfer returned invalid shifted target positions")
        if np.any(shifted < lower) or np.any(shifted >= upper):
            raise ValueError("shifted transfer target leaves the source geometry")
        result = copy_catalog(catalogue)
        for axis, name in enumerate(SHIFTED_COLUMNS):
            result[name] = shifted[:, axis]
        outputs.append(result)
    return tuple(outputs)


def _uniform_edges(minimum: float, maximum: float, width: float, *, label: str) -> np.ndarray:
    """Build an exact uniform grid after proving the configured span is divisible."""
    values = np.asarray([minimum, maximum, width], dtype=float)
    if np.any(~np.isfinite(values)) or width <= 0 or maximum <= minimum:
        raise ValueError(f"{label} bin limits and width are invalid")
    quotient = (maximum - minimum) / width
    n_bins = int(round(quotient))
    # This dimensionless tolerance only absorbs binary representation of a
    # configured integer bin count; it cannot admit a scientifically partial bin.
    tolerance = 1e-10 * max(1.0, abs(quotient))
    if n_bins <= 0 or not np.isclose(quotient, n_bins, rtol=0.0, atol=tolerance):
        raise ValueError(f"{label} span is not an integer multiple of its bin width")
    return np.linspace(minimum, maximum, n_bins + 1)


def exact_growth_rate(z_eff: float, cosmology: Any) -> float:
    """Evaluate the exact mock-cosmology linear growth rate at frozen z_eff."""
    if not np.isfinite(z_eff) or float(z_eff) < 0:
        raise ValueError("z_eff must be finite and non-negative")
    value = np.asarray(cosmology.growth_rate(float(z_eff)), dtype=float)
    if value.size != 1 or not np.isfinite(value).all() or float(value) <= 0:
        raise ValueError("mock cosmology returned an invalid growth rate")
    return float(value)


def weighted_random_zeff_accumulator(
    redshift: Sequence[float],
    comoving_distance_mpc_h: Sequence[float],
    random_density: Sequence[float],
    radial_width_mpc_h: Sequence[float],
    weights: Sequence[float],
    *,
    positions: str,
) -> dict:
    """Accumulate one cap/mock weighted-random-density effective-redshift integral."""
    if positions != "selected_random_redshifts":
        raise ValueError("effective redshift must use selected random redshifts")
    arrays = [
        np.asarray(values, dtype=float)
        for values in (
            redshift,
            comoving_distance_mpc_h,
            random_density,
            radial_width_mpc_h,
            weights,
        )
    ]
    if len({array.shape for array in arrays}) != 1:
        raise ValueError("weighted-random effective-redshift arrays must have identical shapes")
    z, distance, density, widths, weight = arrays
    mask = (
        np.isfinite(z)
        & np.isfinite(distance)
        & np.isfinite(density)
        & np.isfinite(widths)
        & np.isfinite(weight)
        & (distance >= 0)
        & (density >= 0)
        & (widths > 0)
        & (weight > 0)
    )
    base = distance[mask] ** 2 * density[mask] ** 2 * widths[mask] * weight[mask]
    return {
        "row_count": int(np.count_nonzero(mask)),
        "numerator": float(np.sum(z[mask] * base)),
        "denominator": float(np.sum(base)),
        "positions": positions,
    }


def estimate_linear_bias(
    separations_mpc_h: Sequence[float],
    tracer_monopole: Sequence[float],
    matter_monopole: Sequence[float],
    weights: Sequence[float],
    *,
    tracer: str,
    field_identity: str,
    mock_ids: Sequence[int],
    growth_rate: float,
    effective_redshift: float,
    config: Optional[Mapping[str, Any]] = None,
) -> dict:
    """Diagnostic large-scale Kaiser amplitude fit; it never supplies production bias.

    Solve A=b²+(2/3)bf+(1/5)f² for positive b using declared diagonal weights on the
    configured unsmeared parent ensemble. The matter reference must share its redshift/grid.
    No bias uncertainty is estimated, and nonfinite or nonpositive-weight points are excluded.
    """
    resolved = config or load_analysis_config()
    controls = resolved["measurement"]["reconstruction"]["bias_estimation"]
    if sorted(int(value) for value in mock_ids) != sorted(
        int(value) for value in controls["calibration_mock_ids"]
    ):
        raise ValueError("bias estimate must cover the exact configured calibration mocks")
    if str(field_identity) != f"{tracer}|broad":
        raise ValueError("bias estimate requires the tracer's canonical broad-field identity")
    if not np.isfinite(effective_redshift) or effective_redshift <= 0:
        raise ValueError("bias matter reference requires a positive effective redshift")
    scale_range_mpc_h = controls["scale_range_mpc_h"]
    separation, tracer_xi, matter_xi, fit_weights = [
        np.asarray(values, dtype=float)
        for values in (separations_mpc_h, tracer_monopole, matter_monopole, weights)
    ]
    if len({array.shape for array in (separation, tracer_xi, matter_xi, fit_weights)}) != 1:
        raise ValueError("bias-estimator arrays must have identical shapes")
    lo, hi = [float(value) for value in scale_range_mpc_h]
    mask = (
        np.isfinite(separation)
        & np.isfinite(tracer_xi)
        & np.isfinite(matter_xi)
        & np.isfinite(fit_weights)
        & (fit_weights > 0)
        & (separation >= lo)
        & (separation <= hi)
    )
    denominator = float(np.sum(fit_weights[mask] * matter_xi[mask] ** 2))
    minimum_points = int(controls["minimum_scale_points"])
    if (
        np.count_nonzero(mask) < minimum_points
        or denominator <= 0
        or not np.isfinite(float(growth_rate))
        or float(growth_rate) <= 0
    ):
        raise ValueError("bias estimator has insufficient physical support")
    amplitude = float(np.sum(fit_weights[mask] * tracer_xi[mask] * matter_xi[mask]) / denominator)
    discriminant = amplitude - 4.0 * float(growth_rate) ** 2 / 45.0
    if discriminant <= 0:
        raise ValueError("Kaiser amplitude does not yield a positive bias")
    bias = -float(growth_rate) / 3.0 + math.sqrt(discriminant)
    if not np.isfinite(bias) or bias <= 0:
        raise ValueError("resolved tracer bias is not finite and positive")
    payload = {
        "schema_version": 2,
        "kind": "linear_bias_estimate",
        "role": str(controls["role"]),
        "tracer": str(tracer),
        "field_identity": str(field_identity),
        "field_scope": str(controls["field_scope"]),
        "tracer_positions": str(controls["tracer_positions"]),
        "mock_ids": sorted(int(value) for value in mock_ids),
        "bias": bias,
        "growth_rate": float(growth_rate),
        "z_eff": float(effective_redshift),
        "matter_reference": str(controls["matter_reference"]),
        "kaiser_monopole_amplitude": amplitude,
        "scale_range_mpc_h": [lo, hi],
        "point_count": int(np.count_nonzero(mask)),
        "estimator": str(controls["method"]),
    }
    return payload


def reconstruction_field_identity(
    tracer: str,
    measurement_bin: Optional[str],
    *,
    scope: str,
    config: Optional[Mapping[str, Any]] = None,
) -> Tuple[str, Optional[str]]:
    """Return the solve identity and its required effective-redshift scope.

    Schema 16 core fields require a tracer-owned measurement bin. Historical broad fields
    validate optional bin context but retain their parent identity and parent redshift record.
    """
    resolved = config or load_analysis_config()
    if scope not in ("one_field_per_tracer", "separate_measurement_bins"):
        raise ValueError("reconstruction scope is not a registered field division")
    configured = {str(spec["name"]) for spec in measurement_bins(tracer, resolved)}
    if measurement_bin is not None and str(measurement_bin) not in configured:
        raise ValueError("reconstruction bin is not a configured measurement bin")
    if scope == "separate_measurement_bins":
        if resolved["schema_version"] not in PER_BIN_SCHEMA_VERSIONS or measurement_bin is None:
            raise ValueError("per-bin reconstruction requires schema 16/17 and a measurement bin")
        return f"{tracer}|bin={measurement_bin}", str(measurement_bin)
    return f"{tracer}|broad", None


def reconstruction_window_bounds(
    tracer: str,
    *,
    cosmology: Any,
    smoothing_radius_mpc_h: float,
    measurement_bin: Optional[str] = None,
    config: Optional[Mapping[str, Any]] = None,
) -> dict:
    """Record strict bin or historical parent bounds, without radial padding.

    Smoothing radius identifies the mesh family but never changes catalogue membership.
    Cartesian mesh padding is applied separately to the actual selected position extents.
    """
    resolved = config or load_analysis_config()
    scope = str(resolved["measurement"]["reconstruction"]["scope"])
    if scope not in ("one_field_per_tracer", "separate_measurement_bins"):
        raise ValueError("reconstruction scope is not a registered field division")
    reconstruction_field_identity(tracer, measurement_bin, scope=scope, config=resolved)
    radius = float(smoothing_radius_mpc_h)
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("reconstruction smoothing radius must be finite and positive")
    field = resolved["tracers"][str(tracer)]["field"]
    field_zmin, field_zmax = float(field["zmin"]), float(field["zmax"])
    if not field_zmin < field_zmax:
        raise ValueError("tracer field must span a positive redshift interval")

    if scope == "separate_measurement_bins":
        bounds = configured_measurement_bin(tracer, measurement_bin, resolved)
        lo, hi = float(bounds["zmin"]), float(bounds["zmax"])
        distances = np.asarray(
            cosmology.comoving_radial_distance([field_zmin, lo, hi, field_zmax]), dtype=float
        )
        if np.any(~np.isfinite(distances)) or distances[2] <= distances[1]:
            raise ValueError("comoving distance must increase across the measurement bin")
        return {
            "tracer": str(tracer),
            "reconstruction_scope": scope,
            "measurement_bin": str(measurement_bin),
            "smoothing_radius_mpc_h": radius,
            "field_redshift_range": [field_zmin, field_zmax],
            "window_redshift_range": [lo, hi],
            "field_comoving_range_mpc_h": distances[[0, 3]].tolist(),
            "window_comoving_range_mpc_h": distances[[1, 2]].tolist(),
            "clipped_at_field_minimum": lo == field_zmin,
            "clipped_at_field_maximum": hi == field_zmax,
            "membership": "unshifted_observed_redshift_half_open",
            "radial_padding_mpc_h": 0.0,
        }

    grid = np.linspace(field_zmin, field_zmax, 200001)
    distance = np.asarray(cosmology.comoving_radial_distance(grid), dtype=float)
    if np.any(np.diff(distance) <= 0):
        raise ValueError("comoving distance must increase with redshift across the field")
    window_lo, window_hi = float(distance[0]), float(distance[-1])
    return {
        "tracer": str(tracer),
        # The measurement scope this window describes, which is the parent field and no bin.
        # Present so a stored window record cannot be mistaken for a bin's.
        "reconstruction_scope": scope,
        "measurement_bin": None,
        "smoothing_radius_mpc_h": radius,
        "field_redshift_range": [field_zmin, field_zmax],
        "window_redshift_range": [field_zmin, field_zmax],
        "field_comoving_range_mpc_h": [window_lo, window_hi],
        "window_comoving_range_mpc_h": [window_lo, window_hi],
        # Constant under this scope, and kept for exactly that reason: the parent field is
        # solved to its own edges, so both ends are clipped by construction rather than by
        # arithmetic that could come out either way.
        "clipped_at_field_minimum": True,
        "clipped_at_field_maximum": True,
    }


def _mesh_array(mesh: Any) -> np.ndarray:
    """Return the local array behind one PyRecon mesh field without copying it."""
    return np.asarray(getattr(mesh, "value", mesh))


def radial_shell_partition(
    minimum_mpc_h: float, maximum_mpc_h: float, *, width_mpc_h: float
) -> dict:
    """Tile one solved field's radial extent with equal shells of about ``width_mpc_h``.

    The realized width is the target width adjusted so that a whole number of shells covers
    the extent exactly. Trimming instead would leave a final sliver shell holding a handful of
    objects, whose displacement RMS and retained-cell fraction would be noise and would then
    be gated as though they were measurements.
    """
    low, high = float(minimum_mpc_h), float(maximum_mpc_h)
    width = float(width_mpc_h)
    if not np.isfinite([low, high, width]).all() or low <= 0 or high <= low or width <= 0:
        raise ValueError("radial shell partition requires a positive increasing radial extent")
    count = max(1, int(round((high - low) / width)))
    realized = (high - low) / count
    return {
        "count": count,
        "minimum_mpc_h": low,
        "maximum_mpc_h": high,
        "target_width_mpc_h": width,
        "width_mpc_h": realized,
        "edges_mpc_h": (low + realized * np.arange(count + 1)).tolist(),
    }


# Half-width of the painting stencil in cells. TSC is the sole configured resampler and
# ``run_reconstruction`` refuses any other, so a mesh cell can hold painted random weight at
# most one and a half cells beyond the object that painted it ON EACH AXIS. In radius that
# bound is the norm of the three offsets, not one of them: 1.5 cells on all three axes at once
# moves a node 1.5 * sqrt(3) cells away from the object, which on the reference field is 10.4
# Mpc/h rather than 6. The radial partition is widened by that whole amount so every supported
# cell has a shell; the first version of this used the per-axis figure and the real LRG NGC
# field rejected it, which is the check below doing its job.
RECONSTRUCTION_PAINTING_STENCIL_HALF_WIDTH_CELLS = 1.5


def _shell_index(radii: np.ndarray, partition: Mapping[str, Any]) -> np.ndarray:
    """Assign radii to shells, clipping the two ends into the first and last shells."""
    index = np.floor(
        (np.asarray(radii, dtype=float) - float(partition["minimum_mpc_h"]))
        / float(partition["width_mpc_h"])
    )
    return np.clip(index, 0, int(partition["count"]) - 1).astype(np.int64)


def _shell_cell_counts(
    support: np.ndarray,
    threshold_zeroed: np.ndarray,
    *,
    cellsize: np.ndarray,
    offset: np.ndarray,
    partition: Mapping[str, Any],
) -> dict:
    """Count supported and zeroed mesh cells per shell without a full radius mesh.

    The radii are formed one leading-axis slab at a time. A float64 radius array over the
    whole mesh would be another real-mesh equivalent on top of PyRecon's own peak, for a
    diagnostic; the two boolean masks this consumes are an eighth of one each.

    ``solved`` counts the cells the solver actually ran on. With the shared family mask retired
    the only thing that darkens a cell is the density-contrast floor, so this now equals the
    thresholded count by construction. It is still reported separately rather than collapsed,
    because the two answer different questions -- what the floor removed, and what the solver
    saw -- and a future device that darkens cells would separate them again.
    """
    count = int(partition["count"])
    shape = tuple(int(value) for value in support.shape)
    axes = [offset[axis] + np.arange(shape[axis]) * float(cellsize[axis]) for axis in range(3)]
    transverse = axes[1][:, None] ** 2 + axes[2][None, :] ** 2
    supported = np.zeros(count, dtype=np.int64)
    thresholded = np.zeros(count, dtype=np.int64)
    solved = np.zeros(count, dtype=np.int64)
    outside = 0
    for index in range(shape[0]):
        radii = np.sqrt(axes[0][index] ** 2 + transverse)
        shells = _shell_index(radii, partition).ravel()
        support_slab = support[index].ravel()
        zeroed_slab = threshold_zeroed[index].ravel()
        solved_slab = zeroed_slab
        supported += np.bincount(shells[support_slab], minlength=count).astype(np.int64)
        thresholded += np.bincount(shells[support_slab & zeroed_slab], minlength=count).astype(
            np.int64
        )
        solved += np.bincount(shells[support_slab & solved_slab], minlength=count).astype(np.int64)
        radial = radii.ravel()
        outside += int(
            np.count_nonzero(
                support_slab
                & (
                    (radial < float(partition["minimum_mpc_h"]))
                    | (radial > float(partition["maximum_mpc_h"]))
                )
            )
        )
    return {
        "supported": supported,
        "threshold_zeroed": thresholded,
        "solved_zeroed": solved,
        "supported_outside_partition": outside,
    }


def reconstruction_shell_diagnostics(
    data_positions: np.ndarray,
    data_shifted: np.ndarray,
    random_positions: np.ndarray,
    *,
    random_support: np.ndarray,
    threshold_zeroed: np.ndarray,
    cellsize: np.ndarray,
    boxcenter: np.ndarray,
    boxsize: np.ndarray,
    smoothing_radius_mpc_h: float,
    config: Optional[Mapping[str, Any]] = None,
) -> dict:
    """Report radial support and displacement through the solved parent field.

    Empty or nonfinite required fields fail. Retained fractions and displacement variations are
    descriptive; no historical-result bands or displacement-quality thresholds are applied.
    """
    data = np.asarray(data_positions, dtype=float)
    shifted = np.asarray(data_shifted, dtype=float)
    randoms = np.asarray(random_positions, dtype=float)
    if data.shape != shifted.shape or data.ndim != 2 or data.shape[1] != 3 or len(data) == 0:
        raise ValueError("shell diagnostics require matching non-empty (N, 3) data positions")
    radius = float(smoothing_radius_mpc_h)
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("shell diagnostics require a finite positive smoothing radius")
    shape = tuple(int(value) for value in np.shape(random_support))
    if np.shape(threshold_zeroed) != shape:
        raise ValueError("shell diagnostic meshes do not share the realized mesh shape")
    data_radii = np.sqrt((data**2).sum(axis=1))
    # Shell resolution comes from the run configuration; its measurements are descriptive.
    resolved = config or load_analysis_config()
    shell_controls = resolved["measurement"]["reconstruction"]["shell_diagnostics"]
    random_radii = np.sqrt((randoms**2).sum(axis=1))
    reach = RECONSTRUCTION_PAINTING_STENCIL_HALF_WIDTH_CELLS * float(
        np.sqrt(np.sum(np.asarray(cellsize, dtype=float) ** 2))
    )
    partition = radial_shell_partition(
        min(float(data_radii.min()), float(random_radii.min())) - reach,
        max(float(data_radii.max()), float(random_radii.max())) + reach,
        width_mpc_h=shell_controls["shell_width_smoothing_multiple"] * radius,
    )
    counts = _shell_cell_counts(
        np.asarray(random_support, dtype=bool),
        np.asarray(threshold_zeroed, dtype=bool),
        cellsize=np.asarray(cellsize, dtype=float),
        offset=np.asarray(boxcenter, dtype=float) - 0.5 * np.asarray(boxsize, dtype=float),
        partition=partition,
    )
    # The partition already reaches one painting stencil beyond the outermost object, so a
    # supported cell outside it is not a rounding question: it means random weight was painted
    # somewhere the catalogue positions cannot explain.
    if counts["supported_outside_partition"]:
        raise ValueError("supported reconstruction cells fall outside the radial shell partition")
    displacement = np.sqrt(((shifted - data) ** 2).sum(axis=1))
    field_rms = float(np.sqrt(np.mean(displacement**2)))
    if not np.isfinite(field_rms) or field_rms <= 0:
        raise ValueError("reconstruction produced no field-wide displacement")
    shells = _shell_index(data_radii, partition)
    supported = counts["supported"]
    solved_zeroed = counts["solved_zeroed"]
    retained = np.where(supported > 0, 1.0 - solved_zeroed / np.maximum(supported, 1), np.nan)
    table = []
    for shell in range(int(partition["count"])):
        selected = shells == shell
        objects = int(np.count_nonzero(selected))
        if objects == 0 or supported[shell] == 0:
            raise ValueError(
                f"reconstruction shell {shell} holds no {'objects' if objects == 0 else 'cells'}: "
                "the solved field has a radial gap"
            )
        shifts = displacement[selected]
        shell_rms = float(np.sqrt(np.mean(shifts**2)))
        table.append(
            {
                "shell": shell,
                "radial_range_mpc_h": [
                    float(partition["edges_mpc_h"][shell]),
                    float(partition["edges_mpc_h"][shell + 1]),
                ],
                "supported_cell_count": int(supported[shell]),
                "threshold_zeroed_cell_count": int(counts["threshold_zeroed"][shell]),
                "solved_zeroed_cell_count": int(solved_zeroed[shell]),
                "retained_supported_cell_fraction": float(retained[shell]),
                "object_count": objects,
                "displacement_rms_mpc_h": shell_rms,
                "displacement_rms_ratio": shell_rms / field_rms,
                "moved_object_fraction": float(
                    np.mean(np.any(shifted[selected] != data[selected], axis=1))
                ),
            }
        )
    median_rms = float(np.median([shell["displacement_rms_mpc_h"] for shell in table]))
    if not np.isfinite(median_rms) or median_rms <= 0:
        raise ValueError("more than half of the reconstruction shells have no displacement")
    for shell in table:
        shell["displacement_rms_ratio_to_median_shell"] = (
            shell["displacement_rms_mpc_h"] / median_rms
        )
    # Nothing scores these shells. The retained fraction, the displacement scale relative to the
    # median shell and the moved-object fraction were each compared against a configured limit
    # until 2026-09-02; every one of them screened a displacement, which is not the quantity this
    # experiment measures, so they are published instead (SPEC 12.2-12.3). A non-finite or absent
    # displacement is still refused, by the field-wide and median checks above, because that is
    # field invalidity rather than an anomalous measurement (SPEC 12.1).
    return {
        "partition": partition,
        "shell_width_smoothing_multiple": shell_controls["shell_width_smoothing_multiple"],
        "field_displacement_rms_mpc_h": field_rms,
        "field_supported_cell_count": int(supported.sum()),
        "field_threshold_zeroed_cell_count": int(counts["threshold_zeroed"].sum()),
        "field_solved_zeroed_cell_count": int(solved_zeroed.sum()),
        "field_retained_supported_cell_fraction": float(
            1.0 - solved_zeroed.sum() / max(int(supported.sum()), 1)
        ),
        "median_shell_displacement_rms_mpc_h": median_rms,
        "gated_quantities": ["displacement_rms_is_finite"],
        "evidence_only_quantities": [
            "retained_supported_cell_fraction",
            "displacement_rms_mpc_h",
            "displacement_rms_ratio_to_median_shell",
            "moved_object_fraction",
            "threshold_zeroed_cell_count",
        ],
        "shells": table,
        # Nothing in this record scores the reconstruction. A shell that lost most of its support,
        # or that moved unlike its neighbours, is worth seeing -- it is how a silently zeroed
        # redshift range shows up -- but it is published for a reader rather than compared against
        # a limit.
        "decides_nothing": True,
    }


# Broad-field reconstruction followed by identity-preserving bin subsets


def configured_reconstruction_settings(config: Mapping[str, Any]) -> dict:
    """Every :func:`run_reconstruction` setting the ``measurement.reconstruction`` block owns.

    A displacement solve is reproducible only if two callers pass the same thirteen settings, and
    the paired-recovery instrument depends on two arms differing in the box and in nothing else.
    Spelling them out at each call site made that a promise kept by hand; here it is one mapping
    read from one authority, so an arm cannot be solved at a different mesh resolution or
    iteration count than the arm it is compared against.

    The sample-shaped arguments -- catalogues, bias, growth, smoothing, convention, mask family,
    box, thresholds and weights -- are deliberately absent. They vary per field or per arm, which
    is exactly why they are not settings.
    """
    reconstruction = config["measurement"]["reconstruction"]
    policy = reconstruction["resource_policy"]
    return {
        "algorithm": str(reconstruction["algorithm"]),
        "cell_size_mpc_h": float(reconstruction["cell_size_mpc_h"]),
        "iterations": int(reconstruction["iterations"]),
        "box_padding": float(reconstruction["box_padding"]),
        "resampler": str(reconstruction["resampler"]),
        "fft_plan": str(reconstruction["fft_plan"]),
        "mesh_dtype": str(reconstruction["mesh_dtype"]),
        "source_visible_real_mesh_count": int(policy["source_visible_real_mesh_count"]),
        "single_rank_peak_rss_ceiling_gib": float(policy["single_rank_peak_rss_ceiling_gib"]),
        "wrap_positions": bool(reconstruction["wrap_positions"]),
        "position_type": str(reconstruction["position_type"]),
        "line_of_sight": str(reconstruction["line_of_sight"]),
        "threshold_randoms": float(reconstruction["threshold_randoms"]["value"]),
    }


def run_reconstruction(
    data: Mapping[str, Any],
    randoms: Mapping[str, Any],
    *,
    bias: float,
    growth_rate: float,
    smoothing_radius_mpc_h: float,
    convention: str,
    algorithm: str,
    cell_size_mpc_h: float,
    iterations: int,
    box_padding: float,
    resampler: str,
    fft_plan: str,
    mesh_dtype: str,
    threshold_randoms: float,
    mask_family: Mapping[str, Any],
    common_geometry: Mapping[str, Any],
    source_visible_real_mesh_count: int,
    single_rank_peak_rss_ceiling_gib: float,
    wrap_positions: bool,
    position_type: str,
    line_of_sight: str,
    identity_column: str,
    weight_expression: str,
    pyrecon_kw_weights: Any,
    config: Optional[Mapping[str, Any]] = None,
    target_catalogues: Optional[Tuple[Catalog, Catalog]] = None,
    _field_consumer=None,
) -> Tuple[Catalog, Catalog, dict]:
    """Solve one parent/cap with the configured local-LOS iterative FFT prescription.

    Adopt the common parent/cap mesh, checking actual catalogue containment and the memory floor
    before allocation. RecSym adds disp+rsd shifted columns to both data and random catalogues;
    the RecIso check shifts data by disp+rsd and randoms by disp alone.
    Original positions, weights and stable IDs remain available for the pre/post estimators.
    PyRecon's disabled ragged-slab check is replaced by its few-randoms guard and saved shell
    diagnostics. All configuration-dependent diagnostics receive the caller's run snapshot.
    When target_catalogues are supplied, apply the same live field to those fixed targets
    instead of returning shifted source catalogues. No displacement mesh is serialized.
    """
    if target_catalogues is not None and len(target_catalogues) != 2:
        raise ValueError("transfer requires target data and random catalogues")
    solve_started = time.monotonic()
    # RecSym is primary; RecIso is permitted only for its configured robustness run.
    if convention not in ("recsym", "reciso"):
        raise ValueError("reconstruction convention must be RecSym or the RecIso check")
    if algorithm != "iterative_fft":
        raise ValueError("analysis reconstruction requires iterative_fft")
    if line_of_sight != "local":
        raise ValueError("analysis reconstruction requires local line of sight")
    # One builder owns the family name, so the threshold field this cell reads and the mask
    # artifact it consumes are provably the same reconstruction field.
    family = reconstruction_mask_family(
        tracer=mask_family["tracer"],
        cap=mask_family["cap"],
        smoothing_radius_mpc_h=mask_family["smoothing_mpc_h"],
        measurement_bin=mask_family.get("measurement_bin"),
    )
    if dict(mask_family) != family:
        raise ValueError("reconstruction mask family identity is not derived from its coordinates")
    if config is not None and per_bin_reconstruction(config):
        name = family.get("measurement_bin")
        if name is None:
            raise ValueError("per-bin reconstruction requires a measurement bin")
        configured_measurement_bin(family["tracer"], name, config)
    if (
        config is not None
        and config["measurement"]["reconstruction"]["scope"] == "separate_measurement_bins"
        and "transfer_run" not in config["measurement"]["reconstruction"]
    ):
        # Core bins solve on exact observed-redshift partitions of the selected catalogues.
        # Exclusive transfer keeps its own subset construction and target validation.
        from .catalog import catalogue_partition, _partition_rows

        for role, catalogue in (("data", data), ("random", randoms)):
            partition = catalogue_partition(family["tracer"], name, role, config)
            if partition["redshift_column"] not in catalogue or identity_column not in catalogue:
                raise ValueError(
                    "per-bin solve requires the partition redshift and identity columns"
                )
            ids = np.asarray(catalogue[identity_column])
            if (
                not len(ids)
                or ids.dtype.kind not in "iu"
                or np.any(ids < 0)
                or len(np.unique(ids)) != len(ids)
                or not np.all(_partition_rows(catalogue, partition))
            ):
                raise ValueError(
                    "per-bin solve requires the exact nonempty observed-redshift partition"
                )
    if float(family["smoothing_mpc_h"]) != float(smoothing_radius_mpc_h):
        raise ValueError("reconstruction mask family names another smoothing radius")
    threshold_field_identity = family["identity"]
    # ``threshold_randoms`` arrives as an explicit input like every other scientific value this
    # function takes. It is the configured PyRecon-native floor, identical for every field and
    # every photo-z tier; PyRecon applies its own mean-random-weight scaling to it. The frozen
    # surface, when one is supplied, carries published diagnostics and no longer supplies the
    # number, so it is optional.
    if not np.isfinite(threshold_randoms) or threshold_randoms <= 0:
        raise ValueError("reconstruction threshold floor must be finite and positive")
    numeric_inputs = np.asarray(
        [
            bias,
            growth_rate,
            smoothing_radius_mpc_h,
            cell_size_mpc_h,
            iterations,
            threshold_randoms,
            box_padding,
        ],
        dtype=float,
    )
    if np.any(~np.isfinite(numeric_inputs)) or np.any(numeric_inputs <= 0):
        raise ValueError("reconstruction inputs must be finite and positive")
    if position_type != "pos" or resampler != "tsc" or mesh_dtype != "f8":
        raise ValueError("reconstruction position, resampler, or mesh dtype is unsupported")
    if fft_plan != "estimate" or bool(wrap_positions) or float(box_padding) <= 1:
        raise ValueError("reconstruction FFT, wrapping, or box-padding control is unsupported")
    if not isinstance(identity_column, str) or not identity_column.strip():
        raise ValueError("reconstruction requires the configured stable-identity column")
    # The identity column is not consumed numerically here; requiring it proves object
    # identity survived preparation into every shifted catalogue.
    required = ("X", "Y", "Z_CART", "WEIGHT", "WEIGHT_FKP", identity_column)
    validate_catalog(data, required)
    validate_catalog(randoms, required)
    if pyrecon_kw_weights is not None:
        raise ValueError("PyRecon kw_weights must remain absent")
    try:
        from pyrecon import IterativeFFTReconstruction
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("pyrecon is required for analysis reconstruction") from exc
    data_positions = np.column_stack([data["X"], data["Y"], data["Z_CART"]])
    random_positions = np.column_stack([randoms["X"], randoms["Y"], randoms["Z_CART"]])
    geometry = derive_reconstruction_geometry(
        data_positions,
        random_positions,
        cell_size_mpc_h=float(cell_size_mpc_h),
        box_padding=float(box_padding),
        mesh_dtype=mesh_dtype,
        source_visible_real_mesh_count=int(source_visible_real_mesh_count),
        single_rank_peak_rss_ceiling_gib=float(single_rank_peak_rss_ceiling_gib),
    )
    if not geometry["resource_estimate"]["source_visible_floor_within_single_rank_ceiling"]:
        raise MemoryError("reconstruction source-visible floor exceeds the one-rank ceiling")
    # The derived geometry above stays as evidence and as the independent prediction of what
    # this cell alone would need, but it is not the mesh that gets allocated. Cells that share
    # a mask must share a box: each cell's extrema differ slightly, so a per-cell box differs
    # between the tiers and mocks being compared, and a mask frozen on one of them could never
    # be applied to the others. The frozen artifact carries the family's box and this cell
    # adopts it, having proven its own catalogue fits inside.
    adopted_geometry = adopt_common_geometry(
        common_geometry,
        data_positions,
        random_positions,
        family=family,
        cell_size_mpc_h=float(cell_size_mpc_h),
        mesh_dtype=mesh_dtype,
        source_visible_real_mesh_count=int(source_visible_real_mesh_count),
        single_rank_peak_rss_ceiling_gib=float(single_rank_peak_rss_ceiling_gib),
        known_position_weight_shift_array_bytes=int(
            geometry["resource_estimate"]["known_position_weight_shift_array_bytes"]
        ),
    )
    # The adopted box is the union of the family's members, so it is at least as large as
    # this cell's own; the ceiling has to be checked against the mesh actually allocated.
    if not adopted_geometry["resource_estimate"]["source_visible_floor_within_single_rank_ceiling"]:
        raise MemoryError("adopted common reconstruction mesh exceeds the one-rank ceiling")
    data_weights = reconstruction_weight(data, weight_expression)
    random_weights = reconstruction_weight(randoms, weight_expression)
    kwargs = {
        "f": float(growth_rate),
        "bias": float(bias),
        "los": None,  # installed pyrecon spells its varying local LOS as None
        "positions": np.vstack([data_positions, random_positions]),
        "nmesh": np.asarray(adopted_geometry["adopted_nmesh"], dtype=int),
        "boxsize": np.asarray(adopted_geometry["adopted_boxsize_mpc_h"], dtype=float),
        "boxcenter": np.asarray(adopted_geometry["adopted_boxcenter_mpc_h"], dtype=float),
        "resampler": resampler,
        "fft_plan": fft_plan,
        "dtype": mesh_dtype,
        "threshold_randoms": float(threshold_randoms),
        "wrap": bool(wrap_positions),
        "position_type": position_type,
    }
    reconstruction = IterativeFFTReconstruction(**kwargs)
    communicator = getattr(reconstruction, "mpicomm", None)
    if (
        communicator is None
        or int(getattr(communicator, "size", -1)) != 1
        or int(getattr(communicator, "rank", -1)) != 0
    ):
        raise RuntimeError("analysis reconstruction requires pyrecon on one rank")
    nmesh = np.asarray(reconstruction.nmesh, dtype=int)
    boxsize = np.asarray(reconstruction.boxsize, dtype=float)
    cellsize = np.asarray(reconstruction.cellsize, dtype=float)
    boxcenter = np.asarray(reconstruction.boxcenter, dtype=float)
    if any(values.shape != (3,) for values in (nmesh, boxsize, cellsize, boxcenter)):
        raise ValueError("reconstruction mesh geometry must have three axes")
    # These comparisons are unchanged in strength and only change their reference: the mesh
    # PyRecon actually built must be the frozen family mesh, to 1e-12, so the mask bytes can
    # only ever be multiplied into the exact box they were derived on.
    if np.any(nmesh != np.asarray(adopted_geometry["adopted_nmesh"], dtype=int)):
        raise ValueError("pyrecon realized nmesh differs from the adopted common mesh")
    if not np.allclose(
        boxsize,
        np.asarray(adopted_geometry["adopted_boxsize_mpc_h"], dtype=float),
        rtol=0.0,
        atol=1e-12,
    ) or not np.allclose(cellsize, float(cell_size_mpc_h), rtol=0.0, atol=1e-12):
        raise ValueError("pyrecon realized box or cell size differs from the adopted geometry")
    if not np.allclose(
        boxcenter,
        np.asarray(adopted_geometry["adopted_boxcenter_mpc_h"], dtype=float),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("pyrecon realized box center differs from the adopted geometry")
    if np.any(nmesh <= 0) or np.any(~np.isfinite(boxsize)) or np.any(cellsize <= 0):
        raise ValueError("reconstruction mesh geometry is invalid")
    reconstruction.assign_data(data_positions, data_weights, position_type="pos")
    reconstruction.assign_randoms(random_positions, random_weights, position_type="pos")
    if not hasattr(reconstruction, "mesh_randoms"):
        raise RuntimeError("pyrecon did not expose the painted random field")
    # The painted random support, captured while it still exists: PyRecon deletes its random
    # mesh at the end of the density-contrast step. One boolean array is an eighth of a real
    # mesh, and it is the denominator of every per-shell fraction below.
    random_support = _mesh_array(reconstruction.mesh_randoms) > 0.0
    # PyRecon's own "Very few randoms!" guard lives inside the check block disabled below, so
    # it is carried across here rather than lost: same predicate, same field, same point in
    # the sequence -- before the density contrast is formed from it.
    if int(np.count_nonzero(random_support)) < 2:
        raise ValueError("very few randoms: fewer than two mesh cells hold painted random weight")
    # ``check=False`` is deliberate, and it is not a convenience. Installed pyrecon 1.0.0's
    # check block computes
    #     mean_nran_per_cell = self.mpicomm.allreduce(
    #         sum(randoms[randoms > 0] for randoms in self.mesh_randoms))
    # which sums ragged boolean-selected slabs, the inner ``np.sum`` having been dropped, and
    # so raises ValueError whenever two slabs of the mesh hold different numbers of occupied
    # cells. Every irregular survey footprint does. Verified directly on this project's own
    # catalogues -- mock 0, LRG, NGC, the padded LRG1 window -- where check=True raises
    # "operands could not be broadcast together with shapes (51380,) (79449,)" and check=False
    # completes. With the check enabled EVERY reconstruction in this pipeline aborts.
    # Upstream main has rewritten the block; environment.yml installs pyrecon from an unpinned
    # git URL: upgrading means adopting whatever main is on the day, with an unbounded diff
    # across the library that computes the displacement field. The check is disabled and both
    # of the integrity checks it supplied are performed directly here.
    # Its "Very few randoms!" guard is immediately above. Its field-wide masked-fraction log
    # is replaced by the descriptive per-shell table below -- which the library could not have
    # produced in any case: its own frac_nonzero_masked counts non-empty cells of the SMOOTHED
    # random field, which the threshold never zeroes, against the count of non-empty cells of
    # the unsmoothed one, so that diagnostic is wrong independently of the ragged sum.
    reconstruction.set_density_contrast(
        threshold_randoms=float(threshold_randoms),
        smoothing_radius=float(smoothing_radius_mpc_h),
        check=False,
        kw_weights=None,
    )
    if not hasattr(reconstruction, "mesh_delta"):
        raise RuntimeError("pyrecon did not expose the staged density field")
    # The cells the threshold zeroed. PyRecon sets the density contrast to exactly 0.0 below
    # the cutoff and to a ratio of two smoothed fields above it, so within the painted support
    # -- where both fields are bounded away from zero -- equality with zero identifies the
    # thresholded cells exactly. With the shared family mask retired the density-contrast floor
    # is the only thing that darkens a cell, so this is the whole of it.
    threshold_zeroed = _mesh_array(reconstruction.mesh_delta) == 0.0
    reconstruction.run(niterations=int(iterations))
    data_shifted = reconstruction.read_shifted_positions(data_positions, field="disp+rsd")
    # RecIso shifts randoms by the real-space displacement only; RecSym adds the RSD term.
    random_field = "disp" if convention == "reciso" else "disp+rsd"
    random_shifted = reconstruction.read_shifted_positions(random_positions, field=random_field)
    data_shifted = np.asarray(data_shifted, dtype=float)
    random_shifted = np.asarray(random_shifted, dtype=float)
    for label, shifted, source in (
        ("data", data_shifted, data_positions),
        ("randoms", random_shifted, random_positions),
    ):
        if shifted.shape != source.shape or np.any(~np.isfinite(shifted)):
            raise ValueError(f"reconstruction returned invalid shifted {label} positions")
        if not np.any(shifted != source):
            raise ValueError(f"reconstruction returned a null shifted {label} field")
    data_shift_rms = float(np.sqrt(np.mean((data_shifted - data_positions) ** 2)))
    random_shift_rms = float(np.sqrt(np.mean((random_shifted - random_positions) ** 2)))
    # The whole-field checks above cannot see a redshift range that went dark, so the field is
    # measured and gated shell by shell before anything is returned.
    shell_diagnostics = reconstruction_shell_diagnostics(
        data_positions,
        data_shifted,
        random_positions,
        random_support=random_support,
        threshold_zeroed=threshold_zeroed,
        cellsize=cellsize,
        boxcenter=boxcenter,
        boxsize=boxsize,
        smoothing_radius_mpc_h=float(smoothing_radius_mpc_h),
        config=config,
    )
    record = {
        "algorithm": algorithm,
        "bias": float(bias),
        "growth_rate": float(growth_rate),
        "smoothing_radius_mpc_h": float(smoothing_radius_mpc_h),
        "convention": convention,
        "line_of_sight": line_of_sight,
        "requested_cell_size_mpc_h": float(cell_size_mpc_h),
        "iterations": int(iterations),
        "box_padding": float(box_padding),
        "resampler": resampler,
        "fft_plan": fft_plan,
        "mesh_dtype": mesh_dtype,
        "threshold_randoms": float(threshold_randoms),
        "threshold_field_identity": str(threshold_field_identity),
        "mask_family": dict(family),
        "wrap_positions": bool(wrap_positions),
        "position_type": position_type,
        "weight_expression": weight_expression,
        "pyrecon_kw_weights": None,
        # Both geometries are kept: the preflight is what this cell alone would have derived
        # and the adopted block is the family mesh it actually ran on. Their difference is
        # the reviewable evidence that adoption, rather than per-cell derivation, happened.
        "geometry_preflight": geometry,
        "adopted_geometry": adopted_geometry,
        "realized_nmesh": nmesh.tolist(),
        "realized_boxsize_mpc_h": boxsize.tolist(),
        "realized_cellsize_mpc_h": cellsize.tolist(),
        "data_shift_rms_mpc_h": data_shift_rms,
        "random_shift_rms_mpc_h": random_shift_rms,
        # The library check that would have reported on the density field is disabled, so the
        # record carries what replaced it. ``pyrecon_internal_check`` is published rather than
        # implied: a reader comparing two runs must be able to see that the library's own
        # block was off and what stood in for it, without reading this module.
        "pyrecon_internal_check": False,
        "pyrecon_internal_check_reason": "pyrecon_1.0.0_ragged_slab_sum_in_density_check",
        "very_few_randoms_guard": "reimplemented_from_disabled_library_check",
        "shell_diagnostics": shell_diagnostics,
        # The two boolean meshes the diagnostics consume, in units of the float64 real-mesh
        # equivalents the resource floor is quoted in. They are alive across PyRecon's own
        # peak, so they belong in the same accounting even though they do not change it.
        "diagnostic_boolean_mesh_equivalents": 0.25,
    }
    record["field_storage"] = "runtime_only"
    record["source_solve_and_diagnostics_seconds"] = time.monotonic() - solve_started
    if _field_consumer is not None:
        if target_catalogues is not None:
            raise ValueError("a grouped solve cannot also specify a single target")
        # A callback bounds the native mesh lifetime. Only compact diagnostics escape;
        # the consumer loads, shifts and writes one target partition at a time.
        del data_positions, random_positions, data_shifted, random_shifted
        del random_support, threshold_zeroed, data_weights, random_weights, kwargs
        return _field_consumer(reconstruction, record)
    if target_catalogues is None:
        data_result, random_result = copy_catalog(data), copy_catalog(randoms)
        for axis, name in enumerate(SHIFTED_COLUMNS):
            data_result[name] = data_shifted[:, axis]
            random_result[name] = random_shifted[:, axis]
    else:
        data_result, random_result = apply_displacement_field(
            reconstruction, *target_catalogues, identity_column=identity_column
        )
    return data_result, random_result, record


def derive_reconstruction_geometry(
    data_positions: Sequence[Sequence[float]],
    random_positions: Sequence[Sequence[float]],
    *,
    cell_size_mpc_h: float,
    box_padding: float,
    mesh_dtype: str,
    source_visible_real_mesh_count: int = 6,
    single_rank_peak_rss_ceiling_gib: float = 329.0,
) -> dict:
    """Predict PyRecon geometry and a conservative pre-allocation memory floor.

    PyRecon pads the combined catalogue span, rounds ``ceil(span / cellsize)``
    to an even node count, and restores the requested cell size exactly. This
    independent calculation makes that geometry reviewable before allocation.
    The additive estimate includes six source-visible real meshes, both diagnostic boolean
    arrays, and known input-weight and output-shift arrays. It remains a lower bound: three
    representative canaries must measure FFT, allocator, and library overhead
    before production resources are requested.
    """
    data_array = np.asarray(data_positions, dtype=float)
    random_array = np.asarray(random_positions, dtype=float)
    for label, positions in (("data", data_array), ("randoms", random_array)):
        if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] == 0:
            raise ValueError(f"reconstruction {label} positions must have shape (N, 3)")
        if np.any(~np.isfinite(positions)):
            raise ValueError(f"reconstruction {label} positions must be finite")
    controls = np.asarray([cell_size_mpc_h, box_padding], dtype=float)
    if np.any(~np.isfinite(controls)) or float(cell_size_mpc_h) <= 0 or float(box_padding) <= 1:
        raise ValueError("reconstruction cell size and box padding are invalid")
    try:
        dtype = np.dtype(mesh_dtype)
    except TypeError as exc:
        raise ValueError("reconstruction mesh dtype is invalid") from exc
    if dtype != np.dtype("f8"):
        raise ValueError("primary reconstruction geometry requires float64 meshes")

    catalogue_min = np.minimum(data_array.min(axis=0), random_array.min(axis=0))
    catalogue_max = np.maximum(data_array.max(axis=0), random_array.max(axis=0))
    catalogue_extent = catalogue_max - catalogue_min
    if np.any(catalogue_extent <= 0):
        raise ValueError("reconstruction catalogue extent must be positive on every axis")
    padded_target_extent = catalogue_extent * float(box_padding)
    nmesh = np.ceil(padded_target_extent / float(cell_size_mpc_h)).astype(np.int64)
    nmesh += nmesh % 2
    realized_box_extent = nmesh.astype(float) * float(cell_size_mpc_h)
    quantization_expansion = realized_box_extent - padded_target_extent
    if np.any(quantization_expansion < 0) or np.any(
        quantization_expansion >= 2.0 * float(cell_size_mpc_h)
    ):
        raise ValueError("reconstruction even-mesh quantization is outside its derived bound")
    requested_boundary = 0.5 * (padded_target_extent - catalogue_extent)
    realized_boundary = 0.5 * (realized_box_extent - catalogue_extent)
    cell_count = int(np.prod(nmesh, dtype=np.int64))
    real_mesh_bytes = cell_count * dtype.itemsize
    mesh_equivalents = int(source_visible_real_mesh_count)
    ceiling_gib = float(single_rank_peak_rss_ceiling_gib)
    if mesh_equivalents < 6 or not np.isfinite(ceiling_gib) or ceiling_gib <= 0:
        raise ValueError("reconstruction memory policy is invalid")
    known_array_bytes = int((len(data_array) + len(random_array)) * (3 + 1 + 3) * dtype.itemsize)
    source_visible_floor_bytes = mesh_equivalents * real_mesh_bytes + known_array_bytes
    source_visible_floor_bytes += 2 * cell_count * np.dtype(bool).itemsize
    gib = float(1024**3)
    return {
        "catalogue_min_mpc_h": catalogue_min.tolist(),
        "catalogue_max_mpc_h": catalogue_max.tolist(),
        "catalogue_extent_mpc_h": catalogue_extent.tolist(),
        "box_center_mpc_h": (0.5 * (catalogue_min + catalogue_max)).tolist(),
        "requested_cell_size_mpc_h": float(cell_size_mpc_h),
        "box_padding": float(box_padding),
        "padded_target_extent_mpc_h": padded_target_extent.tolist(),
        "requested_boundary_padding_per_side_mpc_h": requested_boundary.tolist(),
        "derived_even_nmesh": nmesh.tolist(),
        "realized_box_extent_mpc_h": realized_box_extent.tolist(),
        "quantization_expansion_mpc_h": quantization_expansion.tolist(),
        "realized_boundary_padding_per_side_mpc_h": realized_boundary.tolist(),
        "mesh_cell_count": cell_count,
        "resource_estimate": {
            "mesh_dtype": dtype.name,
            "bytes_per_real_mesh": real_mesh_bytes,
            "gib_per_real_mesh": real_mesh_bytes / gib,
            "source_visible_peak_mesh_equivalents": mesh_equivalents,
            "known_position_weight_shift_array_bytes": known_array_bytes,
            "source_visible_peak_floor_bytes": source_visible_floor_bytes,
            "source_visible_peak_floor_gib": source_visible_floor_bytes / gib,
            "single_rank_peak_rss_ceiling_gib": ceiling_gib,
            "source_visible_floor_within_single_rank_ceiling": (
                source_visible_floor_bytes / gib <= ceiling_gib
            ),
            "required_high_water_measurements": 3,
            "mpi_policy": "prohibit_without_separate_review",
            "actual_peak_rss_requires_canary": True,
        },
    }


def reconstruction_mask_family(
    *,
    tracer: str,
    cap: str,
    smoothing_radius_mpc_h: float,
    measurement_bin: Optional[str] = None,
) -> dict:
    """Name the common mesh by parent tracer, physical cap and smoothing radius."""
    smoothing = float(smoothing_radius_mpc_h)
    if not np.isfinite(smoothing) or smoothing <= 0:
        raise ValueError("reconstruction mask family smoothing radius must be finite and positive")
    names = {"tracer": str(tracer), "cap": str(cap)}
    if not all(value.strip() for value in names.values()):
        raise ValueError("reconstruction mask family names an empty scientific coordinate")
    family = dict(names)
    family["smoothing_mpc_h"] = smoothing
    family["identity"] = (
        f"{names['tracer']}|cap={names['cap']}"
        f"|smoothing_mpc_h={np.format_float_positional(smoothing, trim='-')}"
    )
    if measurement_bin is not None:
        if not isinstance(measurement_bin, str) or not measurement_bin.strip():
            raise ValueError("reconstruction bin must be a nonempty name")
        family["measurement_bin"] = measurement_bin
        family["identity"] += f"|bin={measurement_bin}"
    return family


def adopt_common_geometry(
    geometry,
    data_positions,
    random_positions,
    *,
    family,
    cell_size_mpc_h,
    mesh_dtype,
    source_visible_real_mesh_count,
    single_rank_peak_rss_ceiling_gib,
    known_position_weight_shift_array_bytes,
):
    """Check the shared box directly against this family's actual catalogues."""
    if not same_identity(geometry["family"], family):
        raise ValueError("shared reconstruction box belongs to another parent or cap")
    mesh = np.asarray(geometry["nmesh"])
    box = np.asarray(geometry["boxsize_mpc_h"], dtype=float)
    centre = np.asarray(geometry["boxcenter_mpc_h"], dtype=float)
    if (
        mesh.shape != (3,)
        or mesh.dtype.kind not in "iu"
        or np.any(mesh <= 0)
        or np.any(mesh % 2)
        or box.shape != (3,)
        or centre.shape != (3,)
        or not np.all(np.isfinite([box, centre]))
        or not np.allclose(box, mesh * cell_size_mpc_h, rtol=0, atol=1e-12)
    ):
        raise ValueError("shared reconstruction mesh has incompatible dimensions or cell size")
    for positions in (data_positions, random_positions):
        array = np.asarray(positions, dtype=float)
        if (
            array.ndim != 2
            or array.shape[1] != 3
            or not len(array)
            or not np.all(np.isfinite(array))
            or np.any(array < centre - box / 2)
            or np.any(array >= centre + box / 2)
        ):
            raise ValueError(
                "shared reconstruction box does not contain finite catalogue positions"
            )
    floor = (
        int(np.prod(mesh))
        * (
            np.dtype(mesh_dtype).itemsize * source_visible_real_mesh_count
            + 2 * np.dtype(bool).itemsize
        )
        + known_position_weight_shift_array_bytes
    ) / 1024**3
    return {
        "adopted_nmesh": mesh.tolist(),
        "adopted_boxsize_mpc_h": box.tolist(),
        "adopted_boxcenter_mpc_h": centre.tolist(),
        "resource_estimate": {
            "source_visible_peak_floor_gib": floor,
            "source_visible_floor_within_single_rank_ceiling": floor
            <= single_rank_peak_rss_ceiling_gib,
        },
    }


def reconstruct_stage(run, unit, destination, *, save_catalogs=True):
    """Solve one parent per cap; save only the shift, referencing the selected catalogue."""
    from .catalog import catalogue_identity, save_catalogue_reference
    from .selection import read_selected_parent, selected_catalogue_entry

    config = run["config"]
    reconstruction = config["measurement"]["reconstruction"]
    if "transfer_run" in reconstruction and save_catalogs:
        targets = [{"run": run, "unit": unit, "destination": destination}]
        result = reconstruct_source_group(targets, identities=validate_source_group(targets))[0]
        if "error" in result:
            raise ValueError(result["error"]["error"])
        return result
    if (
        reconstruction["scope"] == "separate_measurement_bins"
        and "transfer_run" not in reconstruction
    ):
        return _reconstruct_core_bins(run, unit, destination, save_catalogs=save_catalogs)
    diagnostics, catalogues, products = {}, [], []
    for cap in ("NGC", "SGC"):
        data, randoms, preparation = read_selected_parent(run, unit, cap)
        if "transfer_run" in reconstruction:
            shifted_data, shifted_randoms, summary = _reconstruct_transfer_cap(
                run, unit, cap, data, randoms, preparation
            )
            geometry = {
                name: preparation["geometry"][cap][name] for name in unit["measurement_bins"]
            }
        else:
            geometry = preparation["geometry"][cap]
            shifted_data, shifted_randoms, summary = _reconstruct_parent_cap(
                run, unit, data, randoms, preparation, geometry
            )
        selection = [
            row
            for row in preparation["conditions"][unit["condition"]][cap][
                "selection_diagnostics_by_mock"
            ]
            if row["realization_id"] == unit["realization_id"]
        ]
        if len(selection) != 1:
            raise ValueError("missing selection diagnostics for this reconstructed parent")
        diagnostics[cap] = {
            "reconstruction": summary,
            "geometry": geometry,
            "selection": selection[0],
        }
        if "transfer_run" in reconstruction:
            group = preparation["conditions"][unit["condition"]][cap]
            diagnostics[cap]["exclusive_densities"] = group["densities"]
        if save_catalogs:
            for role, catalogue in (("data", shifted_data), ("random", shifted_randoms)):
                identity = catalogue_identity(
                    run,
                    tracer=unit["tracer"],
                    condition=unit["condition"],
                    mock_id=unit["realization_id"],
                    cap=cap,
                    role=role,
                    stage="reconstruct",
                )
                path = f"{cap}--{role}.npz"
                base = selected_catalogue_entry(run, unit, cap, role, preparation=preparation)
                products.extend(
                    save_catalogue_reference(
                        destination, path, identity=identity, base=base, shifted=catalogue
                    )
                )
                catalogues.append({"path": path, "identity": identity})
    payload = {
        "diagnostics": diagnostics,
        "catalogues": catalogues,
        "catalogues_saved": save_catalogs,
    }
    (destination / "reconstruction.json").write_text(json.dumps(payload, indent=2, allow_nan=False))
    return {
        "products": [*products, {"path": "reconstruction.json", "format": "json"}],
        "catalogues_saved": save_catalogs,
    }


def _reconstruct_core_bins(run, unit, destination, *, save_catalogs):
    """Solve, serialize and release one bin/cap at a time within the parent unit."""
    from .catalog import catalogue_identity, catalogue_partition, save_catalogue_reference
    from .selection import read_selected_parent, selected_catalogue_entry

    config, tracer = run["config"], unit["tracer"]
    recon = config["measurement"]["reconstruction"]
    if not unit["measurement_bins"] or len(set(unit["measurement_bins"])) != len(
        unit["measurement_bins"]
    ):
        raise ValueError("reconstruction requires distinct nonempty measurement bins")
    controls = config["selection"]["stable_randomness"]
    products, catalogues, diagnostics = [], [], {}
    for cap in ("NGC", "SGC"):
        for name in unit["measurement_bins"]:
            data, randoms, preparation = read_selected_parent(run, unit, cap, bin_name=name)
            geometry = preparation["geometry"][cap][name]
            field_id, _ = reconstruction_field_identity(
                tracer, name, scope=recon["scope"], config=config
            )
            inputs = preparation["effective_redshifts"][unit["condition"]][name]
            shifted_data, shifted_randoms, summary = run_reconstruction(
                data,
                randoms,
                bias=config["tracers"][tracer]["reconstruction"]["fiducial_bias"],
                growth_rate=inputs["growth_rate"],
                smoothing_radius_mpc_h=recon["smoothing_values_mpc_h"][0],
                convention=recon["allowed_conventions"][0],
                mask_family=geometry["family"],
                common_geometry=geometry,
                identity_column=controls["identity_column"],
                weight_expression=config["measurement"]["weights"]["reconstruction"],
                pyrecon_kw_weights=None,
                config=config,
                **configured_reconstruction_settings(config),
            )
            selection = [
                row
                for row in preparation["conditions"][unit["condition"]][cap][
                    "selection_diagnostics_by_mock"
                ]
                if row["realization_id"] == unit["realization_id"]
            ]
            if len(selection) != 1:
                raise ValueError("missing selection diagnostics for reconstructed bin")
            diagnostic = diagnostics.setdefault(
                cap,
                {
                    "reconstruction": {"scope": recon["scope"], "bins": {}},
                    "geometry": {},
                    "selection": selection[0],
                },
            )
            diagnostic["geometry"][name] = geometry
            diagnostic["reconstruction"]["bins"][name] = {
                "field_identity": field_id,
                "window": preparation["windows"][name],
                "reconstruction_inputs": inputs,
                "source_diagnostics": summary,
            }
            if save_catalogs:
                for role, catalogue in (("data", shifted_data), ("random", shifted_randoms)):
                    identity = catalogue_identity(
                        run,
                        tracer=tracer,
                        condition=unit["condition"],
                        mock_id=unit["realization_id"],
                        cap=cap,
                        role=role,
                        stage="reconstruct",
                    )
                    partition = catalogue_partition(tracer, name, role, config)
                    path = f"{cap}--{name}--{role}.npz"
                    base = selected_catalogue_entry(run, unit, cap, role, preparation=preparation)
                    products.extend(
                        save_catalogue_reference(
                            destination,
                            path,
                            identity=identity,
                            base=base,
                            partition=partition,
                            shifted=catalogue,
                        )
                    )
                    catalogues.append({"path": path, "identity": identity, "partition": partition})
                del catalogue
            del data, randoms, shifted_data, shifted_randoms
    payload = {
        "diagnostics": diagnostics,
        "catalogues": catalogues,
        "catalogues_saved": save_catalogs,
        "representation": "bin_shards_v1",
    }
    (destination / "reconstruction.json").write_text(json.dumps(payload, indent=2, allow_nan=False))
    return {
        "products": [*products, {"path": "reconstruction.json", "format": "json"}],
        "catalogues_saved": save_catalogs,
    }


def _reconstruct_parent_cap(run, unit, data, randoms, preparation, geometry):
    config = run["config"]
    reconstruction = config["measurement"]["reconstruction"]
    return run_reconstruction(
        data,
        randoms,
        bias=config["tracers"][unit["tracer"]]["reconstruction"]["fiducial_bias"],
        growth_rate=preparation["effective_redshifts"][unit["condition"]]["parent"]["growth_rate"],
        smoothing_radius_mpc_h=reconstruction["smoothing_values_mpc_h"][0],
        convention=reconstruction["allowed_conventions"][0],
        mask_family=geometry["family"],
        common_geometry=geometry,
        identity_column=config["selection"]["stable_randomness"]["identity_column"],
        weight_expression=config["measurement"]["weights"]["reconstruction"],
        pyrecon_kw_weights=None,
        config=config,
        **configured_reconstruction_settings(config),
    )


def _reconstruct_transfer_cap(run, unit, cap, data, randoms, preparation):
    """Solve and apply each bin field in memory; publish only target catalogues."""
    import time
    from .selection import read_selected_parent, stable_measurement_subset

    config, tracer = run["config"], unit["tracer"]
    recon = config["measurement"]["reconstruction"]
    controls = config["selection"]["stable_randomness"]
    source = recon["transfer_run"]["source_class"]
    source = unit["condition"] if source == "self" else source
    source_data, source_randoms, _ = read_selected_parent(run, {**unit, "condition": source}, cap)
    shifted = [[], []]
    diagnostics = {}
    for name in unit["measurement_bins"]:

        def subset(catalogue, role):
            return stable_measurement_subset(
                catalogue,
                tracer=tracer,
                bin_name=name,
                redshift_column=controls["observed_redshift_column"] if role == "data" else "Z",
                config=config,
            )

        galaxies, source_r = subset(source_data, "data"), subset(source_randoms, "random")
        geometry = preparation["geometry"][cap][name]
        settings = {
            **configured_reconstruction_settings(config),
            "bias": config["tracers"][tracer]["reconstruction"]["fiducial_bias"],
            "growth_rate": preparation["effective_redshifts"][source][name]["growth_rate"],
            "smoothing_radius_mpc_h": recon["smoothing_values_mpc_h"][0],
            "convention": recon["allowed_conventions"][0],
            "mask_family": geometry["family"],
            "common_geometry": geometry,
            "identity_column": controls["identity_column"],
            "weight_expression": config["measurement"]["weights"]["reconstruction"],
            "pyrecon_kw_weights": None,
        }
        source_identity = {
            "population": preparation["population"],
            "tracer": tracer,
            "measurement_bin": name,
            "condition": source,
            "mock_id": unit["realization_id"],
            "cap": cap,
        }
        execution = run.get("execution_code")
        if execution is None:
            raise ValueError(
                "transfer reconstruction requires the actual execution source identity"
            )
        code_identity = {
            key: execution[key] for key in ("code_commit", "code_state_sha256") if key in execution
        }
        identity = displacement_field_identity(
            galaxies,
            source_r,
            settings=settings,
            source_identity=source_identity,
            code_identity=code_identity,
            config=config,
        )
        started = time.monotonic()
        shifted_data, shifted_randoms, summary = run_reconstruction(
            galaxies,
            source_r,
            **settings,
            config=config,
            target_catalogues=(subset(data, "data"), subset(randoms, "random")),
        )
        target = (shifted_data, shifted_randoms)
        for pieces, catalogue in zip(shifted, target):
            pieces.append(catalogue)
        diagnostics[name] = {
            "source_class": source,
            "field_identity": identity,
            "field_storage": "runtime_only",
            "source_diagnostics": summary,
            "solve_and_shift_seconds": time.monotonic() - started,
        }
    catalogues = [
        {key: np.concatenate([piece[key] for piece in pieces]) for key in pieces[0]}
        for pieces in shifted
    ]
    return (*catalogues, {"scope": "separate_measurement_bins", "bins": diagnostics})


def _transfer_source_inputs(run, unit, cap, name):
    from .selection import read_selected_parent

    config = run["config"]
    recon = config["measurement"]["reconstruction"]
    source = recon["transfer_run"]["source_class"]
    source = unit["condition"] if source == "self" else source
    data, randoms, preparation = read_selected_parent(
        run, {**unit, "condition": source}, cap, bin_name=name
    )
    geometry = preparation["geometry"][cap][name]
    settings = {
        **configured_reconstruction_settings(config),
        "bias": config["tracers"][unit["tracer"]]["reconstruction"]["fiducial_bias"],
        "growth_rate": preparation["effective_redshifts"][source][name]["growth_rate"],
        "smoothing_radius_mpc_h": recon["smoothing_values_mpc_h"][0],
        "convention": recon["allowed_conventions"][0],
        "mask_family": geometry["family"],
        "common_geometry": geometry,
        "identity_column": config["selection"]["stable_randomness"]["identity_column"],
        "weight_expression": config["measurement"]["weights"]["reconstruction"],
        "pyrecon_kw_weights": None,
    }
    execution = run.get("execution_code")
    if not execution:
        raise ValueError("grouped reconstruction requires the actual execution source identity")
    identity = displacement_field_identity(
        data,
        randoms,
        settings=settings,
        source_identity={
            "population": preparation["population"],
            "tracer": unit["tracer"],
            "measurement_bin": name,
            "condition": source,
            "mock_id": unit["realization_id"],
            "cap": cap,
        },
        code_identity={
            key: execution[key] for key in ("code_commit", "code_state_sha256") if key in execution
        },
        config=config,
    )
    return data, randoms, settings, identity


def validate_source_group(targets):
    """Compare actual source arrays and solve settings before any group field is allocated."""
    if not targets:
        raise ValueError("source group has no targets")
    bins = targets[0]["unit"]["measurement_bins"]
    if any(item["unit"]["measurement_bins"] != bins for item in targets):
        raise ValueError("source group has inconsistent bin coverage or order")
    identities = {}
    for cap in ("NGC", "SGC"):
        for name in bins:
            for target in targets:
                data, randoms, settings, identity = _transfer_source_inputs(
                    target["run"], target["unit"], cap, name
                )
                key = (cap, name)
                if key in identities and not same_identity(identities[key], identity):
                    raise ValueError(
                        "intended source group has inconsistent source inputs or solver settings"
                    )
                identities[key] = identity
                del data, randoms, settings
    return identities


def reconstruct_source_group(targets, *, identities):
    """Solve one partition, shift/write each target, release it, then advance the field.

    The runner owns attempts and locks. Source failures propagate to every pending target;
    target read/shift/write failures leave other targets eligible for independent publication.
    """
    import traceback
    from .catalog import catalogue_identity, catalogue_partition, save_catalogue_reference
    from .selection import read_selected_parent, selected_catalogue_entry

    states = [{"products": [], "catalogues": [], "diagnostics": {}, "error": None} for _ in targets]
    for cap, name in identities:
        live = [i for i, state in enumerate(states) if state["error"] is None]
        if not live:
            break
        source = targets[live[0]]
        data, randoms, settings, identity = _transfer_source_inputs(
            source["run"], source["unit"], cap, name
        )
        if not same_identity(identity, identities[cap, name]):
            raise ValueError("source changed after group compatibility validation")

        def consume(field, summary):
            # Validate the mesh before target-specific exception handling. A corrupt
            # field is a source failure and must never be relabelled as a target failure.
            validate_displacement_field(field)
            for index in live:
                target, state = targets[index], states[index]
                run, unit, destination = target["run"], target["unit"], target["destination"]
                try:
                    application_started = time.monotonic()
                    galaxies, target_randoms, preparation = read_selected_parent(
                        run, unit, cap, bin_name=name
                    )
                    shifted = apply_displacement_field(
                        field, galaxies, target_randoms, identity_column=settings["identity_column"]
                    )
                    application_seconds = time.monotonic() - application_started
                    serialization_started = time.monotonic()
                    for role, catalogue in zip(("data", "random"), shifted):
                        entry_identity = catalogue_identity(
                            run,
                            tracer=unit["tracer"],
                            condition=unit["condition"],
                            mock_id=unit["realization_id"],
                            cap=cap,
                            role=role,
                            stage="reconstruct",
                        )
                        partition = catalogue_partition(unit["tracer"], name, role, run["config"])
                        path = f"{cap}--{name}--{role}.npz"
                        base = selected_catalogue_entry(
                            run, unit, cap, role, preparation=preparation
                        )
                        state["products"].extend(
                            save_catalogue_reference(
                                destination,
                                path,
                                identity=entry_identity,
                                base=base,
                                partition=partition,
                                shifted=catalogue,
                            )
                        )
                        state["catalogues"].append(
                            {"path": path, "identity": entry_identity, "partition": partition}
                        )
                    group = preparation["conditions"][unit["condition"]][cap]
                    selection = [
                        row
                        for row in group["selection_diagnostics_by_mock"]
                        if row["realization_id"] == unit["realization_id"]
                    ]
                    if len(selection) != 1:
                        raise ValueError("missing selection diagnostics for grouped target")
                    diagnostic = state["diagnostics"].setdefault(
                        cap,
                        {
                            "reconstruction": {"scope": "separate_measurement_bins", "bins": {}},
                            "geometry": {},
                            "selection": selection[0],
                            "exclusive_densities": group["densities"],
                        },
                    )
                    diagnostic["geometry"][name] = preparation["geometry"][cap][name]
                    diagnostic["reconstruction"]["bins"][name] = {
                        "source_class": identity["definition"]["source_identity"]["condition"],
                        "field_identity": identity,
                        "field_storage": "runtime_only",
                        "source_diagnostics": summary,
                        "target_load_and_application_seconds": application_seconds,
                        "target_serialization_seconds": time.monotonic() - serialization_started,
                    }
                except Exception as exc:
                    state["error"] = {"error": str(exc), "type": type(exc).__name__}
                    (destination / "failure.txt").write_text(traceback.format_exc())
                finally:
                    # Scope target arrays inside a helper invocation would also work;
                    # explicit clearing makes the one-target lifetime inspectable.
                    galaxies = target_randoms = preparation = shifted = catalogue = None

        run_reconstruction(
            data, randoms, **settings, config=source["run"]["config"], _field_consumer=consume
        )
        del data, randoms
    results = []
    for target, state in zip(targets, states):
        if state["error"] is not None:
            results.append({"error": state["error"]})
            continue
        expected = 2 * len(identities)
        if len(state["catalogues"]) != expected:
            raise ValueError("grouped target lacks required bin/footprint shards")
        payload = {
            "diagnostics": state["diagnostics"],
            "catalogues": state["catalogues"],
            "catalogues_saved": True,
            "representation": "bin_shards_v1",
        }
        (target["destination"] / "reconstruction.json").write_text(
            json.dumps(payload, indent=2, allow_nan=False)
        )
        results.append(
            {
                "products": [*state["products"], {"path": "reconstruction.json", "format": "json"}],
                "catalogues_saved": True,
            }
        )
    return results


def read_reconstructed_parent(run, unit, cap, *, bin_name=None):
    """Read a published solve, rejecting diagnostic-only reconstruction outputs."""
    import yaml
    from .catalog import catalogue_identity, read_catalogue

    directory = run["root"] / "reconstruct" / "units" / unit["unit_id"]
    result = yaml.safe_load((directory / "result.yaml").read_text())
    parent_keys = ("unit_id", "tracer", "condition", "realization_id")
    if (
        not same_identity(
            {key: result.get("unit", {}).get(key) for key in parent_keys},
            {key: unit[key] for key in parent_keys},
        )
        or not set(unit["measurement_bins"]).issubset(
            result.get("unit", {}).get("measurement_bins", [])
        )
        or result.get("stage") != "reconstruct"
        or any(
            result.get(key) != run["metadata"][key]
            for key in ("run_name", "code_commit", "analysis_config_sha256")
        )
    ):
        raise ValueError("reconstruction belongs to another unit or snapshot")
    payload = json.loads((directory / "reconstruction.json").read_text())
    if (
        run["config"]["measurement"]["reconstruction"]["scope"] == "separate_measurement_bins"
        and payload.get("representation") != "bin_shards_v1"
    ):
        raise ValueError("per-bin reconstruction requires published bin shards")
    if not payload.get("catalogues_saved"):
        raise ValueError("reconstruction has no reusable catalogues; rerun with catalogue saving")
    from .catalog import catalogue_partition, concatenate_catalogs

    catalogues = []
    for role in ("data", "random"):
        identity = catalogue_identity(
            run,
            tracer=unit["tracer"],
            condition=unit["condition"],
            mock_id=unit["realization_id"],
            cap=cap,
            role=role,
            stage="reconstruct",
        )
        entries = [
            item for item in payload["catalogues"] if same_identity(item["identity"], identity)
        ]
        if payload.get("representation") == "bin_shards_v1":
            names = unit["measurement_bins"] if bin_name is None else [bin_name]
            pieces = []
            for name in names:
                if name not in unit["measurement_bins"]:
                    raise ValueError("requested bin is outside reconstruction unit")
                partition = catalogue_partition(unit["tracer"], name, role, run["config"])
                matches = [
                    entry for entry in entries if same_identity(entry.get("partition"), partition)
                ]
                if len(matches) != 1:
                    raise ValueError("missing or duplicate reconstructed bin shard")
                path = (directory / matches[0]["path"]).resolve()
                if not path.is_relative_to(directory.resolve()):
                    raise ValueError("reconstruction catalogue path leaves its unit")
                pieces.append(read_catalogue(path, identity=identity, partition=partition))
            catalogue = pieces[0] if len(pieces) == 1 else concatenate_catalogs(pieces)
            validate_catalog(catalogue, required=SHIFTED_COLUMNS)
            if len(np.unique(catalogue["SOURCE_ID"])) != len(catalogue["SOURCE_ID"]):
                raise ValueError("reconstruction shards repeat target identities")
            catalogues.append(catalogue)
            continue
        if len(entries) != 1:
            raise ValueError(f"missing or duplicate reconstructed {role} catalogue")
        path = (directory / entries[0]["path"]).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError("reconstruction catalogue path leaves its unit")
        partition = (
            None
            if bin_name is None
            else catalogue_partition(unit["tracer"], bin_name, role, run["config"])
        )
        catalogue = read_catalogue(path, identity=identity, partition=partition)
        validate_catalog(catalogue, required=SHIFTED_COLUMNS)
        catalogues.append(catalogue)
    return (*catalogues, payload["diagnostics"][cap])
