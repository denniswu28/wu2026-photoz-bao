"""Photo-z response and analytic nonlinear damping, with explicit physical units."""

from __future__ import annotations

import itertools
import math
from typing import Any, Mapping, Optional, Sequence, Union

from .config import load_analysis_config

import numpy as np

ArrayLike = Union[float, Sequence[float], np.ndarray]


# Finite-input and physical-domain guards


def _finite_array(value: ArrayLike, name: str) -> np.ndarray:
    """Convert one formula input and reject NaN or infinity before arithmetic."""
    result = np.asarray(value, dtype=float)
    if np.any(~np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_scalar(value: float, name: str) -> float:
    """Return a finite positive scalar control."""
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _require_same_shape(label: str, *values: np.ndarray) -> None:
    """Reject accidental NumPy broadcasting among one physical formula's inputs."""
    shapes = {value.shape for value in values}
    if len(shapes) != 1:
        raise ValueError(f"{label} inputs must share a shape")


# Fixed photo-z response scales consumed by the shared likelihood


def sigma_r_mpc_h(
    redshift: ArrayLike,
    sigma_z_over_one_plus_z: ArrayLike,
    e_z: ArrayLike,
    speed_of_light_km_s: float,
) -> np.ndarray:
    """Convert ``sigma_z/(1+z)`` into one-object radial scatter in Mpc/h.

    ``H(z)=100 h E(z)`` supplies the distance conversion, while multiplication
    by ``(1+z)`` recovers the absolute redshift scatter from ``sigma0``. With
    ``c`` in km/s, ``c/100`` is already in Mpc/h: the explicit ``h`` in H(z)
    cancels the ``h`` carried by the requested Mpc/h distance unit.
    """
    z = _finite_array(redshift, "redshift")
    sigma0 = _finite_array(sigma_z_over_one_plus_z, "sigma_z_over_one_plus_z")
    expansion = _finite_array(e_z, "e_z")
    _require_same_shape("radial-scatter", z, sigma0, expansion)
    speed = _positive_scalar(speed_of_light_km_s, "speed_of_light_km_s")
    if np.any(z < 0) or np.any(sigma0 < 0) or np.any(expansion <= 0):
        raise ValueError("redshift/photo-z must be non-negative and E(z) positive")
    return (speed / 100.0) * sigma0 * (1.0 + z) / expansion


def gaussian_damping_wedge_average(
    k_h_mpc: ArrayLike, sigma_r_mpc_h: float, mu_lower: float, mu_upper: float
) -> np.ndarray:
    """Average of ``exp[-(k mu sigma_r)^2 / 2]`` over ``mu`` in ``[mu_lower, mu_upper]``.

    This is the photo-z propagator of a Gaussian radial error on the galaxy field (the
    characteristic function of the error), averaged over a line-of-sight wedge; the closed
    form uses the error function and tends to one as ``k sigma_r`` vanishes.
    """
    from scipy.special import erf

    k = _finite_array(k_h_mpc, "k_h_mpc")
    sigma = _positive_scalar(sigma_r_mpc_h, "sigma_r_mpc_h")
    lower, upper = float(mu_lower), float(mu_upper)
    if not (0.0 <= lower < upper <= 1.0):
        raise ValueError("mu wedge edges must satisfy 0 <= lower < upper <= 1")
    if np.any(k < 0):
        raise ValueError("wavenumbers must be non-negative")
    scale = k * sigma
    result = np.ones_like(scale, dtype=float)
    positive = scale > 0
    result[positive] = (
        np.sqrt(np.pi / 2.0)
        / (scale[positive] * (upper - lower))
        * (
            erf(scale[positive] * upper / np.sqrt(2.0))
            - erf(scale[positive] * lower / np.sqrt(2.0))
        )
    )
    return result


def sigma_parallel_photoz_mpc_h(
    redshift: ArrayLike,
    sigma_z_over_one_plus_z: ArrayLike,
    e_z: ArrayLike,
    speed_of_light_km_s: float,
) -> np.ndarray:
    """Convert one-object scatter to the pairwise template damping scale.

    Two independent galaxy redshift errors contribute variances that add, so
    the pair-separation scatter is ``sqrt(2)`` times the one-object scatter.
    This is intended to match the pinned desilike template's radial damping of
    the wiggle component, ``exp[-k^2 mu^2 Sigma_parallel^2 / 2]``; it does not
    separately suppress the smooth component. Pinned-version compatibility
    evidence, not this helper, must establish that external template convention.
    """
    return np.sqrt(2.0) * sigma_r_mpc_h(redshift, sigma_z_over_one_plus_z, e_z, speed_of_light_km_s)


def selection_weighted_photoz_response(k_mu, redshifts, pair_weights, sigma0, e_z, *, config):
    """Local-pair diagnostic of the known full-signal response, not a mixed-target fit.

    The caller supplies selection/estimator pair weights on the redshift quadrature.
    No number-density versus count conversion or pair weighting is silently assumed.
    Compare this response with the effective-redshift approximation before fitting;
    neither expression validates response factorization after reconstruction.
    """
    redshifts, weights, expansion = (
        np.asarray(value, dtype=float) for value in (redshifts, pair_weights, e_z)
    )
    modes = np.asarray(k_mu, dtype=float)
    if (
        redshifts.ndim != 1
        or not len(redshifts)
        or weights.shape != redshifts.shape
        or expansion.shape != redshifts.shape
        or any(not np.isfinite(value).all() for value in (redshifts, weights, expansion, modes))
        or np.any(weights < 0)
        or weights.sum() <= 0
        or not np.isfinite(weights.sum())
        or np.any(redshifts <= -1)
        or np.any(expansion <= 0)
        or not np.isfinite(sigma0)
        or sigma0 < 0
    ):
        raise ValueError("invalid selection-weighted photo-z response inputs")
    radii = sigma_r_mpc_h(
        redshifts,
        np.full_like(redshifts, sigma0),
        expansion,
        config["physics"]["speed_of_light_km_s"],
    )
    return np.exp(-((modes[..., None] * radii) ** 2)) @ (weights / weights.sum())


def fixed_count_source_moments(
    contribution_means,
    source_count,
    *,
    contribution_covariances=None,
    forced_indices=(),
    excluded_indices=(),
):
    """Conditional moments of a fixed-count source sum, not total damping centres.

    Implements the assignment and error-mark identities of the fixed-count source-noise
    model. Each row is a parent's real
    vector contribution to a specified pair's shift difference. The count includes
    forced members. Self-source targets are forced; distinct-class targets are
    excluded. Counts refer to the parent assignment before observed selection.

    Marks must be independent conditional on the parent and specified target pair.
    If target errors are already conditioned on, their contributions are fixed:
    supply those values as means and zero covariance, not their marginal errors.
    Observed selection may zero contributions inside the mark expectation. This
    computes a linear sum with fixed normalization, not a ratio with stochastic
    source/random normalization. It excludes parent cosmic/sampling variance and
    shared-random correlations; do not add unconditional source Poisson noise.
    """
    means = _finite_array(contribution_means, "contribution_means")
    if means.ndim != 2 or means.shape[1] == 0:
        raise ValueError("contribution_means must have shape (parents, dimensions)")
    parents, dimensions = means.shape
    if isinstance(source_count, (bool, np.bool_)) or not isinstance(
        source_count, (int, np.integer)
    ):
        raise ValueError("source_count must be an integer")
    memberships = []
    for name, value in (("forced_indices", forced_indices), ("excluded_indices", excluded_indices)):
        indices = np.asarray(value)
        if indices.ndim != 1 or (
            indices.size
            and (
                indices.dtype.kind not in "iu"
                or np.any(indices < 0)
                or np.any(indices >= parents)
                or len(np.unique(indices)) != len(indices)
            )
        ):
            raise ValueError(f"{name} must contain distinct valid parent indices")
        memberships.append(indices.astype(np.intp))
    forced, excluded = memberships
    if np.intersect1d(forced, excluded).size:
        raise ValueError("forced and excluded identities must be disjoint")
    if not len(forced) <= source_count <= parents - len(excluded):
        raise ValueError("source_count is infeasible for the conditioned identities")
    if contribution_covariances is None:
        mark_covariance = np.zeros((parents, dimensions, dimensions))
    else:
        mark_covariance = _finite_array(contribution_covariances, "contribution_covariances")
        if mark_covariance.shape != (parents, dimensions, dimensions):
            raise ValueError(
                "contribution_covariances must have shape (parents, dimensions, dimensions)"
            )
        scale = np.maximum(1.0, np.max(np.abs(mark_covariance), axis=(1, 2)))
        tolerance = 64 * np.finfo(float).eps * scale
        if np.any(
            np.max(np.abs(mark_covariance - mark_covariance.swapaxes(1, 2)), axis=(1, 2))
            > tolerance
        ) or np.any(np.linalg.eigvalsh(mark_covariance) < -tolerance[:, None]):
            raise ValueError("mark covariances must be symmetric positive semidefinite")
    available = np.ones(parents, dtype=bool)
    available[forced] = available[excluded] = False
    candidates = means[available]
    remaining, draw_count = len(candidates), source_count - len(forced)
    mean = means[forced].sum(axis=0)
    covariance = mark_covariance[forced].sum(axis=0)
    if draw_count:
        centre = candidates.mean(axis=0)
        mean += draw_count * centre
        covariance += (draw_count / remaining) * mark_covariance[available].sum(axis=0)
        if draw_count < remaining:
            deviations = candidates - centre
            covariance += (draw_count * (remaining - draw_count) / (remaining - 1)) * (
                deviations.T @ deviations / remaining
            )
    return mean, covariance


def fixed_count_normalized_source_kernel(
    contribution_numerators,
    contribution_masses,
    source_count,
    wavevectors,
    *,
    window_offset,
    max_assignments,
    forced_indices=(),
    excluded_indices=(),
):
    """Exact small-population diagnostic of E[exp(-i k . (G/A - c))].

    Each row supplies the native fixed-window pair numerator g_j and painted mass
    a_j for one fixed parent/error mark. Enumerate uniform fixed-count assignments,
    forcing self targets or excluding exclusive-transfer targets before averaging.
    Normalize each assignment before taking its exponential; neither the mean mass
    nor Gaussian moments may replace the realized denominator in this diagnostic.

    This is conditional on fixed randoms, weights and observed target positions.
    The uniform assignment law must be justified under that conditioning. It is
    not the joint native selection/random ensemble, a full clustering prediction,
    or an adopted damping-centre producer. No production likelihood calls it.
    The caller must provide an enumeration ceiling; no Monte Carlo fallback exists.

    If any assignment has zero mass, return its count/probability and kernel=None.
    Invalid source realizations are never silently discarded or renormalized away.
    Numerators have units of painted mass times Mpc/h, masses are painted weights,
    offsets are Mpc/h and wavevectors are h/Mpc.
    """
    numerators = _finite_array(contribution_numerators, "contribution_numerators")
    masses = _finite_array(contribution_masses, "contribution_masses")
    modes = _finite_array(wavevectors, "wavevectors")
    offset = _finite_array(window_offset, "window_offset")
    if numerators.ndim != 2 or numerators.shape[1] != 3:
        raise ValueError("contribution_numerators must have shape (parents, 3)")
    if masses.shape != (len(numerators),) or np.any(masses < 0):
        raise ValueError("contribution_masses must be nonnegative with one value per parent")
    if np.any(numerators[masses == 0] != 0):
        raise ValueError("zero painted mass must have zero numerator")
    if modes.ndim != 2 or modes.shape[1] != 3 or not len(modes) or offset.shape != (3,):
        raise ValueError("wavevectors must have shape (modes, 3) and window_offset shape (3,)")
    if (
        isinstance(max_assignments, (bool, np.bool_))
        or not isinstance(max_assignments, (int, np.integer))
        or max_assignments < 1
    ):
        raise ValueError("max_assignments must be a positive integer")
    # Reuse the established count/membership validation without a second contract.
    fixed_count_source_moments(
        np.column_stack([numerators, masses]),
        source_count,
        forced_indices=forced_indices,
        excluded_indices=excluded_indices,
    )
    forced = np.asarray(forced_indices, dtype=np.intp)
    excluded = np.asarray(excluded_indices, dtype=np.intp)
    available = np.ones(len(masses), dtype=bool)
    available[forced] = available[excluded] = False
    candidates = np.flatnonzero(available)
    draw_count = source_count - len(forced)
    total = math.comb(len(candidates), draw_count)
    if total > max_assignments:
        raise ValueError(f"exact enumeration needs {total} assignments, above {max_assignments}")
    kernel = np.zeros(len(modes), dtype=complex)
    empty_count = 0
    for drawn in itertools.combinations(candidates, draw_count):
        selected = np.concatenate([forced, np.asarray(drawn, dtype=np.intp)])
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            mass = masses[selected].sum()
            numerator = numerators[selected].sum(axis=0)
            if not np.isfinite(mass) or not np.isfinite(numerator).all():
                raise ValueError("assignment sums overflowed")
            if mass == 0:
                empty_count += 1
                continue
            phase = modes @ (numerator / mass - offset)
        if not np.isfinite(phase).all():
            raise ValueError("normalized assignment phase overflowed")
        kernel += np.exp(-1j * phase)
    return {
        "kernel": None if empty_count else kernel / total,
        "assignment_count": total,
        "empty_source_count": empty_count,
        "empty_source_probability": empty_count / total,
    }


def exclusive_class_source_cross_covariance(
    left_contributions,
    right_contributions,
    left_count,
    right_count,
    *,
    forced_left=(),
    forced_right=(),
    excluded_indices=(),
):
    """Conditional cross covariance of two disjoint fixed-count linear sums.

    Uniform exclusive assignments share one fixed parent population. Forced class
    members and parents excluded from both draws are removed before evaluating
    -n_left * n_right * C_left_right / (remaining - 1). Counts include forced
    members and refer to parent assignments before observed cuts. C uses the
    population denominator, not a sample-covariance denominator.

    Contributions may be conditional mark means only when marks on different
    parents are independent of one another and of assignment. Same-parent marks
    cannot co-occur across the exclusive classes; no extra independent mark-noise
    term belongs in this cross block. Forced target errors must already be fixed.

    This is a fixed-normalization linear-sum diagnostic. It excludes stochastic
    ratio effects, shared randoms and parent cosmic variance. Conditioning on a
    realized selection/random window requires its actual conditional assignment
    law; this helper does not assert that the uniform law survives conditioning.
    """
    left = _finite_array(left_contributions, "left_contributions")
    right = _finite_array(right_contributions, "right_contributions")
    if left.ndim != 2 or right.ndim != 2 or left.shape[0] != right.shape[0]:
        raise ValueError("class contributions must share a two-dimensional parent axis")
    # Reuse the single-class domain checks, including integer counts and distinct
    # disjoint memberships. Every forced opposite-class parent is unavailable.
    memberships = [np.asarray(value) for value in (forced_left, forced_right, excluded_indices)]
    left_forced, right_forced, excluded = [
        values if values.size else values.astype(np.intp) for values in memberships
    ]
    fixed_count_source_moments(
        left,
        left_count,
        forced_indices=forced_left,
        excluded_indices=np.concatenate((right_forced, excluded)),
    )
    fixed_count_source_moments(
        right,
        right_count,
        forced_indices=forced_right,
        excluded_indices=np.concatenate((left_forced, excluded)),
    )
    if left_count + right_count > len(left) - len(excluded):
        raise ValueError("joint class counts exceed the available exclusive population")
    available = np.ones(len(left), dtype=bool)
    unavailable = np.concatenate((left_forced, right_forced, excluded)).astype(np.intp)
    available[unavailable] = False
    draw_left, draw_right = left_count - len(left_forced), right_count - len(right_forced)
    if not draw_left or not draw_right:
        return np.zeros((left.shape[1], right.shape[1]))
    left, right = left[available], right[available]
    population_cross = (left - left.mean(axis=0)).T @ (right - right.mean(axis=0)) / len(left)
    return -draw_left * draw_right * population_cross / (len(left) - 1)


def independent_source_pair_covariance(
    wavevectors,
    integration_weights,
    linear_power,
    source_noise_power,
    *,
    source_response,
    source_noise_filter,
    pair_separation,
    target_errors,
    line_of_sight,
    target_growth,
    reconstruction_growth,
):
    """Exploratory conditional long-displacement tensor, not fitted damping centres.

    This local, homogeneous Gaussian approximation assumes source noise independent
    of the target marks and linear matter field. It does not model self-source
    stochastic correlations, survey windows, or the complete biased density signal.
    No native likelihood consumes this tensor.

    Wavevectors are (N, 3) in h/Mpc; weights include d^3p/(2pi)^3, and both spectra
    are in (Mpc/h)^3. ``source_response`` is S*T_source*B_source/B_rec and
    ``source_noise_filter`` is S/B_rec: Poisson self-pair noise is not multiplied
    by the source photo-z response. Separation and the two radial errors are Mpc/h.
    The caller supplies quadrature, physical settings and the unit line of sight.
    Integrating conjugate modes requires no extra factor of two in this expression.
    """
    modes = _finite_array(wavevectors, "wavevectors")
    if modes.ndim != 2 or modes.shape[1] != 3 or not len(modes):
        raise ValueError("wavevectors must have nonempty shape (N, 3)")
    size = len(modes)
    vectors = []
    for name, value in (
        ("integration_weights", integration_weights),
        ("linear_power", linear_power),
        ("source_noise_power", source_noise_power),
        ("source_response", source_response),
        ("source_noise_filter", source_noise_filter),
    ):
        array = _finite_array(value, name)
        if array.shape != (size,):
            raise ValueError(f"{name} must have shape (N,)")
        if name in {"integration_weights", "linear_power", "source_noise_power"} and np.any(
            array < 0
        ):
            raise ValueError(f"{name} must be non-negative")
        vectors.append(array)
    weights, power, noise, response, noise_filter = vectors
    if not np.any(weights > 0):
        raise ValueError("integration_weights must have positive total weight")
    separation = _finite_array(pair_separation, "pair_separation")
    errors = _finite_array(target_errors, "target_errors")
    los = _finite_array(line_of_sight, "line_of_sight")
    if separation.shape != (3,) or errors.shape != (2,) or los.shape != (3,):
        raise ValueError("separation/line of sight need three entries; target errors need two")
    if not np.isclose(los @ los, 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("line_of_sight must be a unit vector")
    growth = []
    for name, value in (
        ("target_growth", target_growth),
        ("reconstruction_growth", reconstruction_growth),
    ):
        array = _finite_array(value, name)
        if array.ndim != 0 or array < 0:
            raise ValueError(f"{name} must be a non-negative scalar")
        growth.append(float(array))
    squared_modes = np.einsum("ij,ij->i", modes, modes)
    if np.any(squared_modes <= 0):
        raise ValueError("wavevectors must exclude the zero mode")
    radial_modes = modes @ los
    target_vector = (modes + growth[0] * radial_modes[:, None] * los) / squared_modes[:, None]
    solver_vector = (modes + growth[1] * radial_modes[:, None] * los) / squared_modes[:, None]
    relative_phase = np.exp(1j * (modes @ separation))
    error_phase_difference = np.exp(1j * radial_modes * errors[0]) - relative_phase * np.exp(
        1j * radial_modes * errors[1]
    )
    signal_difference = (
        target_vector * (1 - relative_phase)[:, None]
        - solver_vector * (response * error_phase_difference)[:, None]
    )
    noise_difference = -solver_vector * (noise_filter * error_phase_difference)[:, None]
    covariance = np.zeros((3, 3))
    for spectrum, difference in ((power, signal_difference), (noise, noise_difference)):
        covariance += np.einsum(
            "n,ni,nj->ij", weights * spectrum, difference, difference.conj()
        ).real
    if not np.isfinite(covariance).all():
        raise ValueError("pair-displacement covariance overflowed")
    return covariance


def independent_source_pair_kernel(k_vector, error_nodes, error_weights, *, covariance_inputs):
    """Average photo-z phase and displacement damping jointly over two error draws.

    Nodes are radial errors in Mpc/h with normalized probability weights supplied
    by the caller. This fixed-redshift independent-error diagnostic is not valid
    for a selection-truncated pair distribution without changing the quadrature.
    The generally complex pair kernel is returned without discarding its phase;
    the subsequent separation integral is required for a real power spectrum.
    """
    modes = _finite_array(k_vector, "k_vector")
    nodes = _finite_array(error_nodes, "error_nodes")
    weights = _finite_array(error_weights, "error_weights")
    if (
        modes.shape != (3,)
        or nodes.ndim != 1
        or not len(nodes)
        or weights.shape != nodes.shape
        or np.any(weights < 0)
        or not np.isclose(weights.sum(), 1.0, rtol=0.0, atol=1e-12)
    ):
        raise ValueError("invalid wavevector or normalized error quadrature")
    if "target_errors" in covariance_inputs:
        raise ValueError("target_errors are set by the kernel quadrature")
    radial_mode = modes @ _finite_array(covariance_inputs["line_of_sight"], "line_of_sight")
    kernel = 0j
    for left, left_weight in zip(nodes, weights):
        for right, right_weight in zip(nodes, weights):
            if left_weight == 0 or right_weight == 0:
                continue
            covariance = independent_source_pair_covariance(
                **covariance_inputs, target_errors=[left, right]
            )
            variance = float(modes @ covariance @ modes)
            # The covariance is a sum of non-negative outer products. Permit only
            # rounding at zero when a nearly null direction is projected.
            tolerance = np.finfo(float).eps * float(modes @ modes) * float(np.trace(covariance))
            if variance < -8 * tolerance:
                raise ValueError("pair-displacement covariance is not positive semidefinite")
            kernel += (
                left_weight
                * right_weight
                * np.exp(-1j * radial_mode * (left - right) - max(variance, 0.0) / 2)
            )
    return complex(kernel)


def transfer_damping_context(config, preparation, *, tracer, measurement_bin, target_class):
    """Physical inputs a reviewed source-target calculation/calibration must identify."""
    from .config import analysis_config_sha256

    recon = config["measurement"]["reconstruction"]
    source = recon["transfer_run"]["source_class"]
    source = target_class if source == "self" else source
    tiers = {row["label"]: row for row in config["selection"]["tiers"]}
    return {
        "units": "Mpc/h",
        "population": preparation["population"],
        "tracer": tracer,
        "measurement_bin": measurement_bin,
        "source_class": source,
        "target_class": target_class,
        "smoothing_mpc_h": recon["smoothing_values_mpc_h"][0],
        "bias": config["tracers"][tracer]["reconstruction"]["fiducial_bias"],
        "source_redshift": preparation["effective_redshifts"][source][measurement_bin],
        "target_redshift": preparation["effective_redshifts"][target_class][measurement_bin],
        "sigma0": {
            role: tiers[label]["sigma_z_over_one_plus_z"]
            for role, label in (("source", source), ("target", target_class))
        },
        "density_h3_mpc3": {
            role: {
                cap: preparation["conditions"][label][cap]["densities"][measurement_bin][
                    "measured_final_h3_mpc3"
                ]
                for cap in ("NGC", "SGC")
            }
            for role, label in (("source", source), ("target", target_class))
        },
        "source_weight_moments": {
            cap: [
                {
                    "mock_id": row["realization_id"],
                    **row["reconstruction_weight_moments"][measurement_bin],
                }
                for row in preparation["conditions"][source][cap]["selection_diagnostics_by_mock"]
            ]
            for cap in ("NGC", "SGC")
        },
        "geometry": {cap: preparation["geometry"][cap][measurement_bin] for cap in ("NGC", "SGC")},
        "radial_selection_sha256": {
            role: {
                cap: analysis_config_sha256(preparation["conditions"][label][cap]["radial"])
                for cap in ("NGC", "SGC")
            }
            for role, label in (("source", source), ("target", target_class))
        },
        "source_inputs_sha256": analysis_config_sha256(preparation["sources"]),
        "physics": config["physics"],
        "weights": config["measurement"]["weights"],
        "solver": {
            key: recon[key]
            for key in (
                "algorithm",
                "allowed_conventions",
                "iterations",
                "cell_size_mpc_h",
                "threshold_randoms",
                "line_of_sight",
                "resampler",
            )
        },
    }


def resolve_transfer_damping_prior(tracer, reconstruction_state, *, context=None, **kwargs):
    """Do not reinterpret the historical self-field formula as source-target theory.

    The configured design supplies pre-reconstruction references. Post-reconstruction
    numerical centres require the separate theory/calibration review, including flat-prior
    sensitivity fits; a legacy transverse artifact cannot satisfy that prerequisite.
    """
    config = kwargs.get("config") or load_analysis_config()
    transfer = config["measurement"]["reconstruction"].get("transfer_run", {})
    if reconstruction_state != "postreconstruction":
        # The flat headline covers every route, before reconstruction too (revision 2.19).
        prior = resolve_damping_prior(tracer, reconstruction_state, **kwargs)
        return _transfer_flat_prior(prior, transfer, config)
    from .config import same_identity

    artifact = transfer.get("damping_centres")
    if not artifact or context is None:
        raise ValueError(
            "transfer post-reconstruction fits require reviewed source-target damping centres"
        )
    if artifact.get("units") != "Mpc/h" or not artifact.get("review_reference"):
        raise ValueError("damping artifact requires units and its scientific review reference")
    records = [
        row for row in artifact.get("records", []) if same_identity(row.get("context"), context)
    ]
    if len(records) != 1:
        raise ValueError("reviewed damping centres do not match this source-target design")
    record = records[0]
    evidence = record.get("evidence", {})
    if evidence.get("kind") not in {
        "source_target_theory",
        "independent_calibration",
    } or not evidence.get("reference"):
        raise ValueError(
            "damping centres lack source-target theory or independent calibration evidence"
        )
    if evidence["kind"] == "independent_calibration":
        ids = evidence.get("mock_ids", [])
        if not isinstance(ids, list) or any(type(value) is not int or value < 0 for value in ids):
            raise ValueError("damping calibration requires non-negative integer mock IDs")
        if not ids or len(ids) != len(set(ids)) or set(ids) & set(config["analysis"]["mock_ids"]):
            raise ValueError("damping calibration must be independent of recovery mocks")
    centres = record.get("centers_mpc_h", {})
    if set(centres) != {"sigma_perp", "sigma_parallel", "sigma_streaming"} or any(
        not np.isfinite(value) or value < 0 for value in centres.values()
    ):
        raise ValueError("invalid reviewed nonlinear damping centres")
    streaming = config["fitting"]["damping"]["baseline_centers_mpc_h"][tracer]["prereconstruction"][
        "sigma_streaming"
    ]
    if centres["sigma_streaming"] != streaming:
        raise ValueError(
            "transfer retains the configured streaming centre separately from BAO damping"
        )
    # Use only the configured pre-reference response/width/bound conventions as a template.
    kwargs.pop("postreconstruction_sigma_perp_artifact", None)
    prior = resolve_damping_prior(tracer, "prereconstruction", **kwargs)
    prior.update(
        reconstruction_state=reconstruction_state,
        centers_mpc_h=dict(centres),
        transverse_damping_source=evidence["kind"],
        source_target_evidence=record,
        review_reference=artifact["review_reference"],
    )
    return _transfer_flat_prior(prior, transfer, config)


def _transfer_flat_prior(prior, transfer, config):
    """The common flat prior of revision 2.11-P where the transfer run selects it."""
    if transfer.get("prior") != "common_flat_sensitivity":
        return prior
    flat = config["fitting"]["damping"]["common_flat_sensitivity"]
    return {
        **prior,
        "unpriored_parameters": list(flat["parameters"]),
        "bounds_mpc_h": {name: list(flat["bounds_mpc_h"]) for name in flat["parameters"]},
        "scope": TRANSFER_FLAT_DAMPING_SCOPE,
    }


TRANSFER_DAMPING_APPROXIMATIONS = (
    "leading-displacement (Zel'dovich) source-target kernel at the acoustic separation, "
    "averaged over pair orientation",
    "homogeneous local-line-of-sight Gaussian fields; no survey window or finite randoms",
    "source shot noise independent of target marks; self routes omit self-source correlation",
    "independent Gaussian target radial errors at fixed redshift, averaged with the "
    "displacement exponential per pair; no conditioning on the realized selection, randoms "
    "or weights",
    "the kernel divided by the template's photo-z response is the predicted damping; where "
    "target errors far exceed the radius it can exceed one and a Gaussian template cannot "
    "follow it",
    "source noise from the mean measured source density of the two caps and the exact "
    "random/data ratio",
    "modes with target photo-z power response below 1e-3 carry no predicted wiggle",
)
TRANSFER_RESPONSE_FLOOR = 1.0e-3


def _fibonacci_directions(count):
    index = np.arange(count) + 0.5
    polar = np.arccos(1.0 - 2.0 * index / count)
    azimuth = np.pi * (1.0 + np.sqrt(5.0)) * index
    return np.column_stack(
        [np.sin(polar) * np.cos(azimuth), np.sin(polar) * np.sin(azimuth), np.cos(polar)]
    )


def _displacement_modes(p_min, p_max, radial_nodes, angular_nodes):
    """(N, 3) wavevectors and weights including d^3p/(2 pi)^3 on a spherical grid."""
    x, w = np.polynomial.legendre.leggauss(radial_nodes)
    log_p = 0.5 * (np.log(p_max) - np.log(p_min)) * x + 0.5 * (np.log(p_max) + np.log(p_min))
    weight_log = 0.5 * (np.log(p_max) - np.log(p_min)) * w
    mu, weight_mu = np.polynomial.legendre.leggauss(angular_nodes)
    phi = 2 * np.pi * (np.arange(angular_nodes) + 0.5) / angular_nodes
    p = np.exp(log_p)
    P, M, F = np.meshgrid(p, mu, phi, indexing="ij")
    W = (
        (p**3 * weight_log)[:, None, None]
        * weight_mu[None, :, None]
        * np.full(angular_nodes, 2 * np.pi / angular_nodes)[None, None, :]
        / (2 * np.pi) ** 3
    )
    sine = np.sqrt(1.0 - M**2)
    modes = np.column_stack(
        [(P * sine * np.cos(F)).ravel(), (P * sine * np.sin(F)).ravel(), (P * M).ravel()]
    )
    return modes, W.ravel()


_SYMMETRIC_PAIRS = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))


def _outer6(left, right):
    return np.stack([left[:, i] * right[:, j] for i, j in _SYMMETRIC_PAIRS], axis=1)


def _contract6(vectors):
    """Coefficients c with k_i T_ij k_j = c . T6 for six symmetric components."""
    return np.stack(
        [(1.0 if i == j else 2.0) * vectors[:, i] * vectors[:, j] for i, j in _SYMMETRIC_PAIRS],
        axis=1,
    )


def pair_tensor_tables(
    wavevectors,
    integration_weights,
    linear_power,
    source_noise_power,
    *,
    source_response,
    source_noise_filter,
    pair_separation,
    line_of_sight,
    target_growth,
    reconstruction_growth,
    offsets,
    separations,
):
    """independent_source_pair_covariance split by its target-error dependence.

    A(e1, e2) = constant + G(e1) + H(e2) + F(e1 - e2): every error enters through a phase
    exp(i p_z e), so each term is a one-dimensional Fourier sum over the distinct radial
    modes and is tabulated here on ``offsets`` (G, H) and ``separations`` (F). Arrays hold
    the components (xx, yy, zz, xy, xz, yz); tables are (grid, 6).
    """
    modes = np.asarray(wavevectors, dtype=float)
    los = np.asarray(line_of_sight, dtype=float)
    squared = np.einsum("ij,ij->i", modes, modes)
    radial = modes @ los
    target = (modes + target_growth * radial[:, None] * los) / squared[:, None]
    solver = (modes + reconstruction_growth * radial[:, None] * los) / squared[:, None]
    phase = np.exp(1j * (modes @ np.asarray(pair_separation, dtype=float)))
    signal = np.asarray(integration_weights) * np.asarray(linear_power)
    noise = np.asarray(integration_weights) * np.asarray(source_noise_power)
    response, noise_filter = np.asarray(source_response), np.asarray(source_noise_filter)
    solver_weight = signal * response**2 + noise * noise_filter**2
    solver_outer = _outer6(solver, solver)
    cross = _outer6(target, solver) + _outer6(solver, target)
    constant = (signal * np.abs(1 - phase) ** 2) @ _outer6(target, target) + 2 * (
        solver_weight @ solver_outer
    )
    unique, inverse = np.unique(radial, return_inverse=True)

    def table(coefficient, grid):
        grouped = np.zeros((unique.size, 6), dtype=complex)
        np.add.at(grouped, inverse.ravel(), coefficient)
        return np.real(np.exp(-1j * np.outer(grid, unique)) @ grouped)

    return (
        constant,
        table(-(signal * response * (1 - phase))[:, None] * cross, offsets),
        table(-(signal * response * (1 - np.conj(phase)))[:, None] * cross, offsets),
        table(-2 * (solver_weight * phase)[:, None] * solver_outer, separations),
    )


def _correlate(left, right, size):
    """C[d] = sum_i left[i] right[i - d] for d = -(n - 1) .. n - 1, along the last axis."""
    count = left.shape[-1]
    spectrum = np.fft.rfft(left, size) * np.conj(np.fft.rfft(right, size))
    circular = np.fft.irfft(spectrum, size)
    return np.concatenate([circular[..., size - count + 1 :], circular[..., :count]], axis=-1)


def transfer_damping_factor(
    k,
    mu,
    *,
    linear_power,
    drag_scale_mpc_h,
    smoothing_mpc_h,
    bias,
    source_growth,
    reconstruction_growth,
    target_growth,
    source_sigma_r_mpc_h,
    target_sigma_r_mpc_h,
    source_noise_power,
    pair_directions=32,
    error_step_fraction=0.25,
    radial_nodes=96,
    angular_nodes=24,
    chunk=4096,
):
    """Predicted wiggle damping D(k, mu) for a fixed target shifted by a source field.

    The source-target pair tensor at |q| = r_d is split by target-error dependence
    (pair_tensor_tables), so the pair kernel average over two independent Gaussian errors
    is one FFT cross-correlation and one sum on a grid of step min(R, sigma_t) times
    ``error_step_fraction``, which resolves the tensor's smoothing-scale structure even when
    sigma_t >> R. Kernels are averaged over pair orientations and divided by the target's
    photo-z power response on the same grid, which the native template applies separately.
    Inputs are in Mpc/h and h/Mpc; ``linear_power`` maps p to P_L(p) at the target redshift.
    The approximations are listed in TRANSFER_DAMPING_APPROXIMATIONS.
    """
    k = _finite_array(k, "k")
    mu = _finite_array(mu, "mu")
    if k.ndim != 1 or mu.ndim != 1 or np.any(k <= 0) or np.any(np.abs(mu) > 1):
        raise ValueError("damping factor needs positive k and |mu| <= 1 grids")
    radius = _positive_scalar(smoothing_mpc_h, "smoothing_mpc_h")
    r_d = _positive_scalar(drag_scale_mpc_h, "drag_scale_mpc_h")
    sigma_s = float(source_sigma_r_mpc_h)
    sigma_t = float(target_sigma_r_mpc_h)
    if min(sigma_s, sigma_t, float(source_noise_power), float(bias)) < 0:
        raise ValueError("photo-z scales, source noise and bias must be non-negative")
    modes, weights = _displacement_modes(1e-4, 2.0, radial_nodes, angular_nodes)
    p = np.sqrt(np.einsum("ij,ij->i", modes, modes))
    nu = modes[:, 2] / p
    smoothing = np.exp(-0.5 * (p * radius) ** 2)
    divisor = bias + reconstruction_growth * nu**2
    response = (
        smoothing
        * np.exp(-0.5 * (modes[:, 2] * sigma_s) ** 2)
        * (bias + source_growth * nu**2)
        / divisor
    )
    power = np.asarray(linear_power(p), dtype=float)
    noise = np.full(p.shape, float(source_noise_power))
    if sigma_t > 0:
        step = float(error_step_fraction) * min(radius, sigma_t)
        half = int(np.ceil(6.0 * sigma_t / step))
        offsets = step * np.arange(-half, half + 1)
    else:
        step, offsets = 0.0, np.zeros(1)
    error_weights = np.exp(-0.5 * (offsets / sigma_t) ** 2) if sigma_t > 0 else np.ones(1)
    error_weights /= error_weights.sum()
    count = offsets.size
    separations = step * np.arange(-(count - 1), count)
    size = int(2 ** np.ceil(np.log2(max(2 * count - 1, 1))))
    kvec = np.stack(
        [
            np.outer(k, np.sqrt(1.0 - mu**2)).ravel(),
            np.zeros(k.size * mu.size),
            np.outer(k, mu).ravel(),
        ],
        axis=-1,
    )
    # Modes whose exact response exp[-(k mu sigma_t)^2] is below 1e-3 carry no prediction:
    # the template applies that response, so they hold < 0.1 per cent of the wiggle.
    resolved = np.flatnonzero(np.exp(-((kvec[:, 2] * sigma_t) ** 2)) > TRANSFER_RESPONSE_FLOOR)
    pair_weights = _correlate(error_weights, error_weights, size)
    tables = [
        pair_tensor_tables(
            modes,
            weights,
            power,
            noise,
            source_response=response,
            source_noise_filter=smoothing / divisor,
            pair_separation=r_d * direction,
            line_of_sight=np.array([0.0, 0.0, 1.0]),
            target_growth=target_growth,
            reconstruction_growth=reconstruction_growth,
            offsets=offsets,
            separations=separations,
        )
        for direction in _fibonacci_directions(pair_directions)
    ]
    damping = np.zeros(kvec.shape[0])
    for start in range(0, resolved.size, int(chunk)):
        index = resolved[start : start + int(chunk)]
        contraction = _contract6(kvec[index])
        phase = np.cos(np.outer(kvec[index, 2], separations))
        photoz = np.sum(phase * pair_weights, axis=1)
        kernel = np.zeros(index.size)
        for constant, first, second, difference in tables:
            left = error_weights * np.exp(-0.5 * contraction @ first.T)
            right = error_weights * np.exp(-0.5 * contraction @ second.T)
            exponent = -0.5 * (contraction @ constant[:, None] + contraction @ difference.T)
            kernel += np.sum(phase * np.exp(exponent) * _correlate(left, right, size), axis=1)
        kernel /= len(tables)
        with np.errstate(divide="ignore", invalid="ignore"):
            damping[index] = np.where(photoz > 0, kernel / photoz, 0.0)
    return np.clip(damping, 0.0, None).reshape(k.size, mu.size)


def transfer_damping_inputs(context, config, *, cosmology=None):
    """Physical inputs of transfer_damping_factor for one route context and snapshot."""
    from .catalog import build_mock_cosmology

    cosmology = cosmology or build_mock_cosmology(config)
    interpolator = cosmology.get_fourier().pk_interpolator(of="delta_m")
    z_target = float(context["target_redshift"]["z_eff"])
    z_source = float(context["source_redshift"]["z_eff"])
    speed = float(config["physics"]["speed_of_light_km_s"])
    density = np.mean([float(value) for value in context["density_h3_mpc3"]["source"].values()])
    ratio = float(config["selection"]["radial_randoms"]["exact_random_data_ratio"])
    return {
        "linear_power": lambda p: interpolator(p, z=z_target),
        "drag_scale_mpc_h": float(cosmology.rs_drag),
        "smoothing_mpc_h": float(context["smoothing_mpc_h"]),
        "bias": float(context["bias"]),
        "source_growth": float(context["source_redshift"]["growth_rate"]),
        "reconstruction_growth": float(context["source_redshift"]["growth_rate"]),
        "target_growth": float(context["target_redshift"]["growth_rate"]),
        "source_sigma_r_mpc_h": float(
            sigma_r_mpc_h(z_source, context["sigma0"]["source"], cosmology.efunc(z_source), speed)
        ),
        "target_sigma_r_mpc_h": float(
            sigma_r_mpc_h(z_target, context["sigma0"]["target"], cosmology.efunc(z_target), speed)
        ),
        "source_noise_power": (1.0 + 1.0 / ratio) / density,
    }


FINITE_FLAT_DAMPING_SCOPE = "paper_finite_flat_damping_comparison"
# The headline prior of the transfer fits (revision 2.19), not a robustness check.
TRANSFER_FLAT_DAMPING_SCOPE = "transfer_flat_damping_headline"
COMMON_FLAT_DAMPING_SCOPE = "paper_common_flat_damping_comparison"


def common_flat_damping_prior(record, flat):
    """Replace the three damping priors by the common uniform sensitivity (revision 2.11-P).

    Every parameter takes the same bounds; the photo-z response stays separate and does not
    widen the parallel support, unlike the finite flat prior below.
    """
    result = dict(record)
    result["bounds_mpc_h"] = {name: list(flat["bounds_mpc_h"]) for name in flat["parameters"]}
    result["unpriored_parameters"] = list(flat["parameters"])
    result["scope"] = COMMON_FLAT_DAMPING_SCOPE
    return result


def finite_flat_damping_prior(record, flat_damping):
    """Replace the Gaussian damping priors by the configured finite uniform support.

    The pairwise photo-z scale only widens the parallel support in quadrature; the explicit
    full-signal photo-z response is still applied exactly once by the model.
    """
    result = dict(record)
    lower, upper = flat_damping["lower_mpc_h"], flat_damping["nonlinear_upper_mpc_h"]
    photoz = record["sigma_parallel_photoz_mpc_h"]
    if not np.isfinite(photoz) or photoz < 0:
        raise ValueError("invalid pairwise photo-z scale")
    names = ("sigma_streaming", "sigma_parallel", "sigma_perp")
    result["bounds_mpc_h"] = {
        name: [lower, float(np.hypot(upper, photoz)) if name == "sigma_parallel" else upper]
        for name in names
    }
    result["unpriored_parameters"] = list(names)
    result["scope"] = FINITE_FLAT_DAMPING_SCOPE
    return result


#: Per-bin damping priors (2026-10-20, specification revision 2.20): one Gaussian
#: (mean, standard deviation) per redshift bin, used before and after reconstruction, for every
#: redshift-error condition and every run. Each spans the union of the per-condition theory
#: intervals.
UNIFIED_DAMPING_PRIOR = "unified_bin_prior_v1"
UNIFIED_BIN_DAMPING_MPC_H = {
    "BGS1": {"sigma_perp": (6.0, 3.0), "sigma_parallel": (13.0, 9.0)},
    "BGS2": {"sigma_perp": (5.0, 3.0), "sigma_parallel": (9.0, 6.0)},
    "LRG1": {"sigma_perp": (5.0, 4.0), "sigma_parallel": (8.0, 7.0)},
    "LRG2": {"sigma_perp": (5.0, 3.0), "sigma_parallel": (8.0, 6.0)},
    "LRG3": {"sigma_perp": (6.0, 5.0), "sigma_parallel": (9.0, 7.0)},
}
UNIFIED_STREAMING_MPC_H = (2.0, 4.0)
#: Added to the two acoustic centres at each smoothing radius, in both reconstruction states;
#: the widths do not change. Other radii have no configured prior.
UNIFIED_RADIUS_OFFSET_MPC_H = {
    15.0: (0.0, 0.0),
    25.0: (0.0, 0.0),
    35.0: (1.0, 2.0),
    45.0: (1.0, 2.0),
}


def unified_damping_prior(
    tracer,
    reconstruction_state,
    *,
    measurement_bin,
    smoothing_mpc_h,
    sigma_z_over_one_plus_z,
    z_eff,
    e_z_eff,
    config,
):
    """The configured per-bin Gaussian damping prior, identical before and after reconstruction.

    It replaces every damping centre and width of the run (the analytic transverse centre, the
    baselines and source-target centres alike); the photo-z response stays the separate,
    fixed response of every other fit.
    """
    if reconstruction_state not in {"prereconstruction", "postreconstruction"}:
        raise ValueError("reconstruction_state is not registered")
    if measurement_bin not in UNIFIED_BIN_DAMPING_MPC_H:
        raise ValueError(f"no configured damping prior for measurement bin {measurement_bin!r}")
    radius = float(smoothing_mpc_h)
    if radius not in UNIFIED_RADIUS_OFFSET_MPC_H:
        raise ValueError(f"no configured damping prior offset for smoothing radius {radius:g} Mpc/h")
    damping = config["fitting"]["damping"]
    registered = [float(value) for value in damping["registered_sigma_z_over_one_plus_z"]]
    sigma0 = float(sigma_z_over_one_plus_z)
    if not any(np.isclose(sigma0, value, rtol=0.0, atol=1e-12) for value in registered):
        raise ValueError("photo-z scale is not registered")
    table = UNIFIED_BIN_DAMPING_MPC_H[measurement_bin]
    perp_offset, parallel_offset = UNIFIED_RADIUS_OFFSET_MPC_H[radius]
    speed = float(config["physics"]["speed_of_light_km_s"])
    minimum = float(damping["nonlinear_minimum_mpc_h"])
    return {
        "tracer": tracer,
        "reconstruction_state": reconstruction_state,
        "sigma_z_over_one_plus_z": sigma0,
        "z_eff": float(z_eff),
        "e_z_eff": float(e_z_eff),
        "sigma_parallel_photoz_mpc_h": float(
            sigma_parallel_photoz_mpc_h(z_eff, sigma0, e_z_eff, speed)
        ),
        "centers_mpc_h": {
            "sigma_perp": table["sigma_perp"][0] + perp_offset,
            "sigma_parallel": table["sigma_parallel"][0] + parallel_offset,
            "sigma_streaming": UNIFIED_STREAMING_MPC_H[0],
        },
        "widths_mpc_h": {
            "sigma_perp": table["sigma_perp"][1],
            "sigma_parallel": table["sigma_parallel"][1],
            "sigma_streaming": UNIFIED_STREAMING_MPC_H[1],
        },
        "bounds_mpc_h": {
            "sigma_perp": [minimum, None],
            "sigma_parallel": [minimum, None],
            "sigma_streaming": [minimum, None],
        },
        "photoz_response_policy": config["fitting"]["photoz_response"]["primary"],
        "transverse_damping_source": UNIFIED_DAMPING_PRIOR,
        "prior_table": UNIFIED_DAMPING_PRIOR,
        "measurement_bin": measurement_bin,
        "smoothing_mpc_h": radius,
        "unpriored_parameters": [],
        "scope": "final_science_fit",
    }


def resolve_damping_prior(
    tracer: str,
    reconstruction_state: str,
    *,
    sigma_z_over_one_plus_z: float,
    z_eff: float,
    e_z_eff: float,
    postreconstruction_sigma_perp_artifact: Optional[Mapping[str, Any]] = None,
    config: Optional[Mapping[str, Any]] = None,
) -> dict:
    """Resolve nonlinear damping separately from the explicit photo-z response.

    Damping centres are the DESI-informed baselines for the tracer and state, except the
    post-reconstruction transverse centre, which is computed analytically for the sample from its
    own attained density, bias, effective redshift and the smoothing radius. Injected photo-z
    scatter is never added in quadrature to nonlinear damping; the physical photo-z scale is
    recorded as response provenance only.

    **Every width is the fixed triple, in both reconstruction states.** The prescription states
    the Gaussian prior widths once for the whole centre table and replaces only the
    post-reconstruction transverse centre analytically, so there is one width authority and it is
    configuration. A width taken instead from the scatter of the realizations the prior will later
    weight is a prior selected against the data it is supposed to be independent of -- the same
    objection that keeps a mock-fitted median out of the centre, and the reason the analytical
    prescription is authoritative for this experiment. Its applicability under photometric
    smearing still requires scientific interpretation of the prior and posterior diagnostics.

    The transverse artifact a post-reconstruction fit supplies therefore carries a centre and the
    sample coordinates it was computed for, and no width: there is no quantity for a calibration
    to contribute, and consequently no calibration.
    """
    resolved = config or load_analysis_config()
    damping = resolved["fitting"]["damping"]
    if reconstruction_state not in {"prereconstruction", "postreconstruction"}:
        raise ValueError("reconstruction_state is not registered")
    registered_sigma = [float(value) for value in damping["registered_sigma_z_over_one_plus_z"]]
    sigma0 = float(sigma_z_over_one_plus_z)
    if not any(np.isclose(sigma0, value, rtol=0.0, atol=1e-12) for value in registered_sigma):
        raise ValueError("photo-z scale is not registered")
    if tracer not in damping["baseline_centers_mpc_h"]:
        raise ValueError(f"no damping baseline for tracer {tracer!r}")
    tracer_baselines = damping["baseline_centers_mpc_h"][tracer]
    if reconstruction_state not in tracer_baselines:
        raise ValueError(
            f"no damping baseline for tracer {tracer!r} and state {reconstruction_state!r}"
        )
    baseline = tracer_baselines[reconstruction_state]
    widths = damping["gaussian_widths_mpc_h"]
    speed = float(resolved["physics"]["speed_of_light_km_s"])
    photoz = float(sigma_parallel_photoz_mpc_h(z_eff, sigma0, e_z_eff, speed))
    # One width authority, named in configuration and used in both states. Reading it once here
    # is what makes that single authority a property of the code rather than of two call sites
    # that happen to agree.
    if str(damping["width_authority"]) != "fixed_gaussian_widths_mpc_h":
        raise ValueError(
            "the damping width authority is not the fixed Gaussian widths, and no other width "
            "source is implemented: a width derived from the realizations the prior weights is "
            "the per-realization damping inference the prescription excludes"
        )
    transverse_source = "prereconstruction_desi_informed_baseline"
    transverse_center = float(baseline["sigma_perp"])
    transverse_width = float(widths["sigma_perp"])
    unpriored: list = []
    scope = "final_science_fit"
    if reconstruction_state == "postreconstruction":
        analytic = postreconstruction_sigma_perp_artifact
        # The centre and the sample it was computed for. No width: the width is the fixed one
        # above, so there is nothing here for a calibration to supply and no field through which
        # a mock-derived number could arrive.
        required = {"analytic_center_mpc_h", "attained_density_h3_mpc3", "tracer"}
        if not isinstance(analytic, Mapping) or not required.issubset(analytic):
            raise ValueError(
                "post-reconstruction damping requires density-aware transverse evidence"
            )
        transverse_center = float(analytic["analytic_center_mpc_h"])
        if (
            not np.all(
                np.isfinite([transverse_center, float(analytic["attained_density_h3_mpc3"])])
            )
            or transverse_center < 0
            or float(analytic["attained_density_h3_mpc3"]) <= 0
            or analytic["tracer"] != tracer
        ):
            raise ValueError("density-aware transverse damping evidence is invalid")
        transverse_source = "analytic_density_aware_centre"
    minimum = float(damping["nonlinear_minimum_mpc_h"])
    return {
        "tracer": tracer,
        "reconstruction_state": reconstruction_state,
        "sigma_z_over_one_plus_z": sigma0,
        "z_eff": float(z_eff),
        "e_z_eff": float(e_z_eff),
        "sigma_parallel_photoz_mpc_h": photoz,
        "centers_mpc_h": {
            "sigma_perp": transverse_center,
            "sigma_parallel": float(baseline["sigma_parallel"]),
            "sigma_streaming": float(baseline["sigma_streaming"]),
        },
        "widths_mpc_h": {
            "sigma_perp": transverse_width,
            "sigma_parallel": float(widths["sigma_parallel"]),
            "sigma_streaming": float(widths["sigma_streaming"]),
        },
        "bounds_mpc_h": {
            "sigma_perp": [minimum, None],
            "sigma_parallel": [minimum, None],
            "sigma_streaming": [minimum, None],
        },
        "photoz_response_policy": resolved["fitting"]["photoz_response"]["primary"],
        "transverse_damping_source": transverse_source,
        # Which damping coordinates are varied without a Gaussian, and what the resulting fit may
        # be used for. Both are on the record rather than implied by the caller, so a downstream
        # reader can see that a result came from the calibration route without reconstructing how
        # it was configured.
        "unpriored_parameters": list(unpriored),
        "scope": scope,
    }


# Calibration and no-alternate holdout evidence


#: Poisson stochasticity of the reconstruction displacement estimator, in the shot-noise term of
#: the RecSym transverse damping. It is the baseline value and deliberately not the covariance's
#: calibrated ``alpha_SN``: that scalar rescales the shot noise of the *covariance* estimator, and
#: no configured mapping exists between it and the stochasticity of the displacement estimator. The
#: two are different measurements of different things that happen to share a name.
POSTRECONSTRUCTION_SHOT_NOISE_P_SHOT = 1.0

#: Deterministic quadrature controls for the reconstruction term, whose integrand contains the
#: linear power spectrum and therefore has no closed form.
#:
#: These are implementation detail rather than scientific choices, and they are fixed rather than
#: exposed so that they cannot become a tuning parameter. They are recorded in the formula payload
#: only because another implementation needs them to reproduce the centre exactly. The grid spans
#: the range over which the integrand is non-negligible at either end -- at small ``k`` the
#: ``(1 - S)^2`` factor suppresses it as ``k^4``, at large ``k`` the power spectrum does -- and its
#: convergence is measured in the tests against the 1 Mpc/h comparison scale, not asserted.
SIGMA_REC_QUADRATURE = {
    "k_min_h_mpc": 1e-4,
    "k_max_h_mpc": 10.0,
    "points": 4096,
    "spacing": "log",
    "rule": "trapezoid",
}

#: Descriptive formula metadata; this is neither a result hash nor a runtime credential.
POSTRECONSTRUCTION_SIGMA_PERP_FORMULA = {
    "name": "recsym_analytic_transverse_damping",
    "version": 2,
    "smoothing_kernel": "S(k) = exp(-k^2 R^2 / 2)",
    "reconstruction_term": (
        "Sigma_rec^2 = (2/3) * int dk/(2 pi^2) [1 - S(k)]^2 P_lin(k, z_eff) [1 - j0(k r_d)]"
    ),
    "shot_noise_term": (
        "Sigma_shot^2 = (2/3) * P_shot / (b^2 nbar) * int dk/(2 pi^2) S(k)^2 [1 - j0(k r_d)]"
    ),
    "combination": "Sigma_perp_analytic = sqrt(Sigma_rec^2 + Sigma_shot^2)",
    # The configured 3 Mpc/h post-reconstruction baseline is DESI-informed and may already carry
    # typical survey shot-noise and nonlinear contributions, so adding the full shot term to it
    # would double count. The baseline is kept as start state and as a diagnostic comparison, and
    # is not a term of this prescription.
    "baseline_role": (
        "the configured post-reconstruction baseline is start state and diagnostic only; it is "
        "not a term of the analytic centre, because adding the full shot term to a DESI-informed "
        "baseline would double count"
    ),
    "shot_noise_closed_form": (
        "the shot-noise integral is evaluated exactly as "
        "(1/(2 pi^2)) (sqrt(pi)/(2 R) - (pi/(2 r_d)) erf(r_d/(2 R)))"
    ),
    "reconstruction_quadrature": dict(SIGMA_REC_QUADRATURE),
    "p_shot": POSTRECONSTRUCTION_SHOT_NOISE_P_SHOT,
    "p_shot_note": (
        "the Poisson baseline; the covariance's calibrated alpha_SN is a different estimator's "
        "rescaling and may not be substituted"
    ),
    "density_source": (
        "attained_density_h3_mpc3 of the sample being calibrated, never the requested target"
    ),
    "bias_source": "tracers.<tracer>.reconstruction.fiducial_bias, frozen",
    "smoothing_radius_source": "measurement.reconstruction.smoothing_values_mpc_h[0]",
    "linear_power_source": (
        "linear matter power of the configured c000 mock cosmology at the anchor's frozen z_eff"
    ),
    "drag_scale_source": "rs_drag of the same configured mock cosmology",
    "recsym_transverse_interpretation": (
        "the transverse direction under RecSym receives no additional line-of-sight growth-factor "
        "multiplier"
    ),
    "citation": "DESI BAO reconstruction damping model; White 2015 reconstruction propagator",
    "approximation_role": (
        "Sigma_shot^2 ~ P_shot / (6 pi^(3/2) b^2 nbar R) for R << r_d is a diagnostic regression "
        "and never replaces the exact expression"
    ),
    "units": "lengths in Mpc/h; k in h/Mpc; P_lin in (Mpc/h)^3; nbar in h^3 Mpc^-3",
}


def _damping_wavenumber_grid() -> np.ndarray:
    """The fixed logarithmic grid the reconstruction term is integrated on."""
    return np.logspace(
        math.log10(float(SIGMA_REC_QUADRATURE["k_min_h_mpc"])),
        math.log10(float(SIGMA_REC_QUADRATURE["k_max_h_mpc"])),
        int(SIGMA_REC_QUADRATURE["points"]),
    )


def _one_minus_spherical_j0(k: np.ndarray, drag_scale_mpc_h: float) -> np.ndarray:
    """``1 - j0(k r_d)``, with the removable singularity at ``k = 0`` handled exactly."""
    argument = np.asarray(k, dtype=float) * float(drag_scale_mpc_h)
    return 1.0 - np.sinc(argument / math.pi)


def reconstruction_transverse_damping_squared(
    *,
    linear_power,
    smoothing_radius_mpc_h: float,
    drag_scale_mpc_h: float,
) -> float:
    """The transverse damping the modes reconstruction does not restore.

    Reconstruction removes the large-scale displacement it can measure; what survives is weighted
    by ``[1 - S(k)]^2``, so the smoothing radius decides how much of the field is left undone. This
    is the dominant term at the densities of this study's bright sample and is comparable to the
    shot-noise term at its sparse one.

    ``linear_power`` is called on the fixed grid rather than sampled adaptively, because an
    adaptive rule would make the centre depend on a tolerance and turn an implementation detail
    into something a reader would have to be told.
    """
    radius = float(smoothing_radius_mpc_h)
    drag = float(drag_scale_mpc_h)
    if not np.all(np.isfinite([radius, drag])) or radius <= 0 or drag <= 0:
        raise ValueError("the reconstruction damping needs a positive finite radius and drag scale")
    k = _damping_wavenumber_grid()
    power = np.asarray(linear_power(k), dtype=float)
    if power.shape != k.shape or not np.all(np.isfinite(power)) or np.any(power < 0):
        raise ValueError("the linear power spectrum is not finite and non-negative on the grid")
    smoothing = np.exp(-0.5 * (k * radius) ** 2)
    integrand = (1.0 - smoothing) ** 2 * power * _one_minus_spherical_j0(k, drag)
    integral = float(np.trapz(integrand, k) / (2.0 * math.pi**2))
    if not math.isfinite(integral) or integral <= 0:
        raise ValueError("the reconstruction damping integral is not positive and finite")
    return (2.0 / 3.0) * integral


def shot_noise_transverse_damping_squared(
    *,
    bias: float,
    attained_density_h3_mpc3: float,
    smoothing_radius_mpc_h: float,
    drag_scale_mpc_h: float,
    p_shot: float = POSTRECONSTRUCTION_SHOT_NOISE_P_SHOT,
) -> float:
    """The additional transverse damping the finite sampling of the displacement field leaves.

    Reconstruction estimates that field from a finite sample, so the displacement it applies is
    itself noisy. How much depends on ``b^2 nbar``, which is why this study's sparse high-redshift
    bin is affected several times more than its dense low-redshift one.

    The integral is evaluated in closed form. Substituting ``j0(x) = sin(x)/x`` and using
    ``int_0^inf exp(-a^2 k^2) sin(bk)/k dk = (pi/2) erf(b/(2a))`` gives

        int_0^inf dk/(2 pi^2) exp(-k^2 R^2) (1 - j0(k r_d))
            = (1/(2 pi^2)) * (sqrt(pi)/(2 R) - (pi/(2 r_d)) erf(r_d/(2 R)))

    which is exact rather than approximate and reproducible without a quadrature tolerance.
    """
    values = [
        float(bias),
        float(attained_density_h3_mpc3),
        float(smoothing_radius_mpc_h),
        float(drag_scale_mpc_h),
        float(p_shot),
    ]
    if not np.all(np.isfinite(values)) or any(value <= 0 for value in values):
        raise ValueError(
            "shot-noise transverse damping needs a positive finite bias, attained density, "
            "smoothing radius, drag scale and stochasticity"
        )
    bias_value, density, radius, drag, stochasticity = values
    integral = (1.0 / (2.0 * math.pi**2)) * (
        math.sqrt(math.pi) / (2.0 * radius)
        - (math.pi / (2.0 * drag)) * math.erf(drag / (2.0 * radius))
    )
    if integral <= 0:
        raise ValueError("the shot-noise damping integral is not positive")
    return (2.0 / 3.0) * stochasticity / (bias_value**2 * density) * integral


def approximate_shot_noise_transverse_damping_squared(
    *,
    bias: float,
    attained_density_h3_mpc3: float,
    smoothing_radius_mpc_h: float,
    p_shot: float = POSTRECONSTRUCTION_SHOT_NOISE_P_SHOT,
) -> float:
    """The ``R << r_d`` limit of the shot-noise term, a diagnostic regression and never the centre.

    At the primary radius the two differ by about a third, because ``r_d / R`` is only ~6.6, which
    is a large enough gap that a prescription resting on this would be a different prescription.
    """
    values = [
        float(bias),
        float(attained_density_h3_mpc3),
        float(smoothing_radius_mpc_h),
        float(p_shot),
    ]
    if not np.all(np.isfinite(values)) or any(value <= 0 for value in values):
        raise ValueError("the approximate shot-noise damping needs positive finite inputs")
    bias_value, density, radius, stochasticity = values
    return stochasticity / (6.0 * math.pi**1.5 * bias_value**2 * density * radius)


def analytic_postreconstruction_sigma_perp(
    *,
    tracer: str,
    attained_density_h3_mpc3: float,
    linear_power,
    drag_scale_mpc_h: float,
    config: Optional[Mapping[str, Any]] = None,
) -> dict:
    """The density-aware analytic transverse damping centre, with everything it was computed from.

    The centre is the two theory terms in quadrature and **not** the configured baseline plus the
    shot term. That baseline is DESI-informed and may already carry typical survey shot-noise and
    nonlinear contributions, so adding the full shot term to it would double count. It is carried
    here as start state and as a diagnostic comparison, which is what it can honestly be.

    Every other input is read from its own authority rather than accepted: the smoothing radius
    from configuration, the bias from the tracer's frozen reconstruction block. The attained
    density, the linear power and the drag scale are supplied, because none is a property of the
    configuration alone -- the first is what the mocks delivered for this sample, and the other two
    belong to the cosmology the caller has already built at this anchor's frozen effective redshift.
    """
    resolved = config or load_analysis_config()
    tracers = resolved["tracers"]
    if str(tracer) not in tracers:
        raise ValueError(f"{tracer!r} is not a configured tracer")
    bias = float(tracers[str(tracer)]["reconstruction"]["fiducial_bias"])
    baselines = resolved["fitting"]["damping"]["baseline_centers_mpc_h"]
    if str(tracer) not in baselines:
        raise ValueError(f"{tracer!r} has no configured damping baseline")
    baseline = float(baselines[str(tracer)]["postreconstruction"]["sigma_perp"])
    # The frozen scan runs pin the centre to the reference radius; their refits (revision 2.18)
    # and every other run use their own radius.
    radius = float(
        resolved["fitting"]["damping"].get(
            "reference_smoothing_mpc_h",
            resolved["measurement"]["reconstruction"]["smoothing_values_mpc_h"][0],
        )
    )

    reconstruction_squared = reconstruction_transverse_damping_squared(
        linear_power=linear_power,
        smoothing_radius_mpc_h=radius,
        drag_scale_mpc_h=drag_scale_mpc_h,
    )
    shot_squared = shot_noise_transverse_damping_squared(
        bias=bias,
        attained_density_h3_mpc3=attained_density_h3_mpc3,
        smoothing_radius_mpc_h=radius,
        drag_scale_mpc_h=drag_scale_mpc_h,
    )
    approximate = approximate_shot_noise_transverse_damping_squared(
        bias=bias,
        attained_density_h3_mpc3=attained_density_h3_mpc3,
        smoothing_radius_mpc_h=radius,
    )
    centre = math.sqrt(reconstruction_squared + shot_squared)
    return {
        "kind": "analytic_postreconstruction_sigma_perp",
        "tracer": str(tracer),
        "analytic_center_mpc_h": centre,
        "formula": POSTRECONSTRUCTION_SIGMA_PERP_FORMULA,
        "reconstruction_variance_mpc_h2": reconstruction_squared,
        "reconstruction_sigma_mpc_h": math.sqrt(reconstruction_squared),
        "shot_noise_variance_mpc_h2": shot_squared,
        "shot_noise_sigma_mpc_h": math.sqrt(shot_squared),
        # Published beside the exact value, never in place of it, so the distance from the
        # R << r_d limit is visible rather than assumed.
        "approximate_shot_noise_variance_mpc_h2": approximate,
        "approximation_ratio": approximate / shot_squared,
        # Start state and diagnostic. Not a term: see ``baseline_role`` in the formula payload.
        "configured_baseline_mpc_h": baseline,
        "baseline_difference_mpc_h": centre - baseline,
        "bias": bias,
        "attained_density_h3_mpc3": float(attained_density_h3_mpc3),
        "smoothing_radius_mpc_h": radius,
        "drag_scale_mpc_h": float(drag_scale_mpc_h),
        "p_shot": POSTRECONSTRUCTION_SHOT_NOISE_P_SHOT,
    }
