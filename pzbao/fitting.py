"""Deterministic optimization, profiles and native profile readers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple

import numpy as np

from .config import same_identity


from .likelihood import _route_likelihood, _require_single_process

#: The Minuit fit-summary flags the pinned desilike profiler copies onto
#: ``Profiles.bestfit.attrs``. They are the only externally visible difference between a
#: maximization and a maximization that did not happen, because the library catches Minuit's
#: ``RuntimeError``, logs a warning and returns an empty ``Profiles`` for that iteration -- while
#: the caller has already seeded ``bestfit`` with the starting point, so a full-sized file is
#: still written and still loads.
MAXIMIZATION_CONVERGENCE_FLAGS = ("is_valid", "is_above_max_edm", "has_reached_call_limit")

#: The Minos per-parameter interval flags. ``at_lower_limit`` and ``at_upper_limit`` are the
#: rails Document A section 16.7 counts together with profiling failures.
INTERVAL_CONVERGENCE_FLAGS = (
    "is_valid",
    "lower_valid",
    "upper_valid",
    "at_lower_limit",
    "at_upper_limit",
)


def _per_start(array: Any) -> np.ndarray:
    """One row per start, with ``desilike``'s trailing derivative axis removed.

    On a surface with analytically solved parameters ``desilike`` stores each quantity with a
    trailing "derivs" axis, so ``bestfit["loglikelihood"]`` is ``(starts, derivatives)`` rather
    than ``(starts,)``. The value at the solution is the zeroth entry; the others are derivatives
    and are larger by orders of magnitude. Flattening such an array and taking ``argmax`` yields an
    index into the flattened space, which is not a start.
    """
    values = np.asarray(array, dtype=float)
    while values.ndim > 1:
        values = values[..., 0]
    return np.atleast_1d(values)


def _maximization_convergence(profiles: Any) -> dict:
    """Read the optimizer's own validity flags; absent flags mean no usable maximization."""
    attrs = dict(getattr(getattr(profiles, "bestfit", None), "attrs", None) or {})
    missing = [name for name in MAXIMIZATION_CONVERGENCE_FLAGS if name not in attrs]
    if missing:
        raise RuntimeError(
            f"profiler published no maximization summary {missing}; the saved best fit is the "
            f"starting point rather than a fit"
        )
    record = {name: bool(attrs[name]) for name in MAXIMIZATION_CONVERGENCE_FLAGS}
    record["converged"] = (
        record["is_valid"]
        and not record["is_above_max_edm"]
        and not record["has_reached_call_limit"]
    )
    return record


def _interval_convergence(
    profiles: Any, interval_names: Sequence[str], *, strict: bool = True
) -> dict:
    """Read native Minos validity and rail flags, retaining explicit missing intervals."""
    attrs = dict(getattr(getattr(profiles, "interval", None), "attrs", None) or {})
    missing = sorted(name for name in interval_names if not isinstance(attrs.get(name), Mapping))
    if missing and strict:
        raise RuntimeError(f"profiler published no confidence interval for {missing}")
    records = {}
    for name in interval_names:
        flags = attrs.get(name)
        if not isinstance(flags, Mapping):
            record = {key: False for key in INTERVAL_CONVERGENCE_FLAGS}
            record["railed"] = False
            record["published"] = False
            records[str(name)] = record
            continue
        record = {key: bool(flags.get(key)) for key in INTERVAL_CONVERGENCE_FLAGS}
        record["railed"] = record["at_lower_limit"] or record["at_upper_limit"]
        record["published"] = True
        records[str(name)] = record
    return records


def maximize_usable(profiler, config):
    """Keep native start generation and retain each start's own optimizer validity flags."""
    count = config["fitting"]["optimization"]["profiler_iterations"]
    # Generating the complete starting array first preserves native maximize(niterations=count)
    # random draws. Native Profiles.concatenate keeps only the last attrs, so decode each start
    # before concatenation can discard its Minuit summary.
    starts = profiler._get_start(niterations=count)
    candidates, records = [], []
    for index, start in enumerate(starts):
        profiler.profiles = None
        # The pinned profiler treats only a Python sequence as multiple starts; an ndarray
        # would be wrapped once more and acquire an incompatible third dimension.
        profiles = profiler.maximize(niterations=1, start=[start.copy()])
        try:
            flags = _maximization_convergence(profiles)
        except RuntimeError as exc:
            records.append({"start_index": index, "converged": False, "reason": str(exc)})
            continue
        objective = np.asarray(profiles.bestfit.logposterior, dtype=float)
        finite = objective.size == 1 and bool(np.all(np.isfinite(objective)))
        records.append(
            {
                "start_index": index,
                **flags,
                "finite_objective": finite,
                "logposterior": float(objective.item()) if finite else None,
            }
        )
        if flags["converged"] and finite:
            candidates.append((float(objective.item()), index, profiles.deepcopy()))
    if not candidates:
        raise ValueError("no optimizer start produced a usable successful result")
    _, selected, best = max(candidates, key=lambda item: item[0])
    best.attrs.update(optimizer_starts=records, selected_start_index=selected)
    profiler.profiles = best
    return best


def run_profiling(
    likelihood: Any,
    output: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    ap_mode: str,
    identity=None,
) -> dict:
    """Save a usable native profile and every available configured AP interval.

    Each optimizer start keeps its own native validity flags. Missing or invalid Minos intervals
    remain visible for review. These paths are inside the driver's unpublished unit attempt.
    """
    from desilike.profilers import MinuitProfiler

    interval_parameters = config["fitting"].get("ap_interval_parameters")
    if not isinstance(interval_parameters, Mapping) or ap_mode not in interval_parameters:
        raise ValueError(f"AP interval parameters are not configured for mode {ap_mode!r}")
    interval_names = list(interval_parameters[ap_mode])
    profiler = MinuitProfiler(
        likelihood, seed=int(config["fitting"]["optimization"]["random_seed"])
    )
    _require_single_process(profiler.mpicomm, "profiling")
    profile_path, stats_path = Path(output["profiles"]), Path(output["statistics"])
    if profile_path == stats_path or any(path.exists() for path in (profile_path, stats_path)):
        raise FileExistsError("profiling requires two distinct unused attempt paths")
    maximized = maximize_usable(profiler, config)
    maximization = _maximization_convergence(maximized)
    parameters = likelihood.varied_params.select(basename=interval_names)
    if {parameter.basename for parameter in parameters} != set(interval_names):
        raise ValueError("profiling likelihood is missing declared AP interval parameters")
    profiles = profiler.interval(params=parameters)
    intervals = _interval_convergence(profiles, interval_names, strict=False)
    if identity is not None:
        profiles.attrs["identity"] = dict(identity)
    profiles.save(str(profile_path))
    profiles.to_stats(tablefmt="pretty", fn=str(stats_path))
    for temporary in (profile_path, stats_path):
        if not temporary.is_file() or temporary.stat().st_size <= 0:
            raise RuntimeError(f"profiler did not write a non-empty claimed product: {temporary}")
    return {
        "method": "profiling",
        "outputs": dict(output),
        "maximization": maximization,
        "intervals": intervals,
        "profile_failures_or_rails": int(not maximization["converged"])
        + sum(1 for record in intervals.values() if not record["is_valid"] or record["railed"]),
    }


def fit_unit(run, unit, destination):
    """Save a real deterministic profile, its fitted curve and a matched smooth comparison."""
    from desilike.profilers import MinuitProfiler

    config = run["config"]
    theory, observable, likelihood, metadata, arrays, covariance = _route_likelihood(
        run, unit, method="profiling"
    )
    identity = {
        key: run["metadata"][key] for key in ("run_name", "code_commit", "analysis_config_sha256")
    }
    identity.update(**metadata["route"], realization_id=unit["realization_id"])
    paths = {
        "profiles": str(destination / "profiles.npy"),
        "statistics": str(destination / "statistics.txt"),
    }
    optimization = run_profiling(
        likelihood, paths, config, ap_mode=metadata["ap_mode"], identity=identity
    )
    decoded = read_profiles(paths["profiles"], identity=identity)
    likelihood(**decoded["profiles"].bestfit.choice(input=True))
    data, model = np.asarray(likelihood.flatdata), np.asarray(likelihood.flattheory)
    residual = data - model
    if data.shape != model.shape or not np.all(np.isfinite(model)):
        raise ValueError("best-fit curve does not match its finite data vector")
    chi2 = float(residual @ np.linalg.solve(covariance, residual))
    np.savez(
        destination / "fit_curve.npz",
        data=data,
        model=model,
        residual=residual,
        standard_deviation=np.sqrt(np.diag(covariance)),
        fit_edges=arrays["fit_edges"],
        multipoles=arrays["multipoles"],
        separation=(
            arrays["separations"].mean(axis=0)
            if unit["realization_id"] is None
            else arrays["separations"][
                np.flatnonzero(arrays["realization_ids"] == unit["realization_id"]).item()
            ]
        ),
    )
    # The smooth fit is built through the same constructor so its explicit photo-z response,
    # window, priors and nuisance prescription cannot diverge through template reinitialization.
    _, _, smooth_likelihood, _, _, _ = _route_likelihood(
        run, unit, method="detection", only_now=True
    )
    smooth_profiler = MinuitProfiler(
        smooth_likelihood, seed=config["fitting"]["optimization"]["random_seed"]
    )
    _require_single_process(smooth_profiler.mpicomm, "smooth comparison")
    smooth = maximize_usable(smooth_profiler, config)
    smooth_identity = {**identity, "model": "smooth"}
    smooth.attrs["identity"] = smooth_identity
    smooth.save(str(destination / "smooth_profiles.npy"))
    smooth_likelihood(**smooth.bestfit.choice(input=True))
    smooth_residual = np.asarray(smooth_likelihood.flatdata) - np.asarray(
        smooth_likelihood.flattheory
    )
    chi2_smooth = float(smooth_residual @ np.linalg.solve(covariance, smooth_residual))
    delta = chi2_smooth - chi2
    if not np.all(np.isfinite([chi2, chi2_smooth, delta])):
        raise ValueError("profile comparison has non-finite fit quality")
    parameter_definitions = []
    for parameter in likelihood.all_params:
        if parameter.basename in {"loglikelihood", "logprior", "logposterior"}:
            continue
        prior = parameter.prior
        parameter_definitions.append(
            {
                "name": str(parameter),
                "fixed": bool(parameter.fixed),
                "derived": parameter.derived,
                "value": float(parameter.value),
                "prior": {
                    "distribution": prior.dist,
                    "limits": [
                        float(value) if np.isfinite(value) else None for value in prior.limits
                    ],
                    "parameters": dict(prior.attrs),
                },
            }
        )
    metadata.update(
        identity=identity,
        parameters=decoded["parameters"],
        parameter_definitions=parameter_definitions,
        optimizer={key: value for key, value in optimization.items() if key != "outputs"},
        optimizer_starts=decoded["attrs"].get("optimizer_starts", []),
        chi2=chi2,
        degrees_of_freedom=int(len(data) - likelihood.nvaried),
        detection={
            "chi2_smooth": chi2_smooth,
            "delta_chi2": delta,
            "sqrt_delta_chi2_proxy": float(np.sqrt(max(delta, 0))),
            "interpretation": "descriptive proxy, not calibrated Gaussian significance",
        },
    )
    (destination / "fit.json").write_text(json.dumps(metadata, indent=2, allow_nan=False))
    return {
        "products": [
            {"path": "profiles.npy", "format": "profiles", "identity": identity},
            {"path": "smooth_profiles.npy", "format": "profiles", "identity": smooth_identity},
            {"path": "statistics.txt", "format": "txt"},
            {"path": "fit.json", "format": "json"},
            {
                "path": "fit_curve.npz",
                "format": "npz",
                "arrays": [
                    "data",
                    "model",
                    "residual",
                    "standard_deviation",
                    "fit_edges",
                    "multipoles",
                    "separation",
                ],
            },
        ]
    }


def read_profiles(path, *, identity=None):
    """Decode native best fits, Minos offsets, parabolic errors and their validity flags once."""
    from desilike.samples import Profiles

    from .artifacts import resolve_saved_path

    profiles = Profiles.load(str(resolve_saved_path(path)))
    if identity is not None and not same_identity(profiles.attrs.get("identity"), dict(identity)):
        raise ValueError("native profile identity differs from the requested fit")
    if profiles.bestfit._logposterior not in profiles.bestfit:
        raise ValueError("native profiles contain no saved objective")
    objective = _per_start(profiles.bestfit.logposterior)
    if not objective.size or not np.any(np.isfinite(objective)):
        raise ValueError("native profiles contain no finite objective")
    index = int(np.argmax(np.where(np.isfinite(objective), objective, -np.inf)))
    convergence = _maximization_convergence(profiles)
    if not convergence["converged"]:
        raise ValueError("native profile has no usable successful optimizer result")
    rows = []
    excluded = {"logposterior", "loglikelihood", "logprior", "aweight", "fweight"}
    for parameter in profiles.bestfit.params():
        name = str(parameter)
        if parameter.basename in excluded:
            continue
        values = _per_start(profiles.bestfit[name])
        if len(values) != len(objective) or not np.isfinite(values[index]):
            raise ValueError(f"native profile parameter {name} has invalid best-fit axes or value")
        row = {
            "name": name,
            "basename": parameter.basename,
            "value": float(values[index]),
            "fixed": bool(parameter.fixed),
            "derived": parameter.derived,
            "parabolic_error": None,
            "minos_offsets": None,
            "minos_bounds": None,
            "interval_flags": None,
            "interval_unavailable_reason": "no Minos interval saved",
        }
        error = getattr(profiles, "error", None)
        if error is not None and name in error.names():
            value = _per_start(error[name])[index]
            if np.isfinite(value) and value > 0:
                row["parabolic_error"] = float(value)
        interval = getattr(profiles, "interval", None)
        if interval is not None and name in interval.names():
            value = np.asarray(interval[name], dtype=float).reshape(-1)
            flags = dict(interval.attrs.get(name, {}))
            row["interval_flags"] = {
                key: bool(flags.get(key)) for key in INTERVAL_CONVERGENCE_FLAGS
            }
            if value.shape == (2,) and np.all(np.isfinite(value)) and value[0] <= 0 <= value[1]:
                row["minos_offsets"] = value.tolist()
                row["minos_bounds"] = (values[index] + value).tolist()
                row["interval_unavailable_reason"] = None
            else:
                row["interval_unavailable_reason"] = (
                    "native Minos interval has invalid shape, signs or values"
                )
        rows.append(row)
    try:
        chi2 = -2 * float(_per_start(profiles.bestfit[profiles.bestfit._loglikelihood])[index])
        chi2_reason = None if np.isfinite(chi2) else "non-finite native data log-likelihood"
        if chi2_reason is not None:
            chi2 = None
    except (AttributeError, KeyError, ValueError, IndexError):
        chi2, chi2_reason = None, "native data log-likelihood is unavailable"
    return {
        "profiles": profiles,
        "parameters": rows,
        "bestfit_index": index,
        "logposterior": float(objective[index]),
        "chi2": chi2,
        "chi2_reason": chi2_reason,
        "maximization": convergence,
        "attrs": dict(profiles.attrs),
    }


# Joint photo-z-tier constraint: a combination of the tier posteriors, not a joint likelihood
# (specification revision 2.14). Every function below is pure arithmetic on saved
# posterior summaries and per-realization estimates; nothing here reads products or runs a fit.

#: The dilation coordinates every tier posterior reports, in the stacked tier-major order.
JOINT_PARAMETERS = ("qiso", "qap")

#: The tier that may never enter the joint constraint: it is the reference, not a photo-z sample.
EXCLUDED_JOINT_TIER = "native"


def nearest_positive_definite(matrix: Any, *, floor_ratio: float = 1e-10) -> np.ndarray:
    """Symmetrize and raise every eigenvalue below ``floor_ratio`` of the largest to that floor.

    A stacked estimate covariance assembled from exact posterior blocks and measured cross
    correlations is not guaranteed positive definite; this is the smallest change that makes it
    usable, and a matrix that already is positive definite is returned unchanged.
    """
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1] or not np.isfinite(values).all():
        raise ValueError("a covariance must be a finite square matrix")
    if not 0 < float(floor_ratio) < 1:
        raise ValueError("the eigenvalue floor must lie strictly between zero and one")
    symmetric = 0.5 * (values + values.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    largest = float(eigenvalues.max())
    if largest <= 0:
        raise ValueError("a covariance needs at least one positive eigenvalue")
    floor = float(floor_ratio) * largest
    if eigenvalues.min() >= floor:
        return symmetric
    raised = np.maximum(eigenvalues, floor)
    return (eigenvectors * raised) @ eigenvectors.T


def nonnegative_blue_weights(covariance: Any) -> Tuple[np.ndarray, list]:
    """Best linear unbiased weights that are never negative.

    The unconstrained weights are ``C^-1 1 / (1^T C^-1 1)``. When the measured correlations make
    one of them negative the combination would extrapolate beyond the estimates that carry it, so
    that estimate is removed and the rest re-solved until every weight is non-negative. Returned
    weights sum to one and carry zeros for the removed estimates; the removed indices follow.
    """
    matrix = np.asarray(covariance, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not np.isfinite(matrix).all():
        raise ValueError("estimate covariance must be a finite square matrix")
    np.linalg.cholesky(matrix)
    active = list(range(matrix.shape[0]))
    removed: list = []
    while True:
        precision = np.linalg.inv(matrix[np.ix_(active, active)])
        weights = precision.sum(axis=1) / precision.sum()
        if np.all(weights >= 0) or len(active) == 1:
            break
        dropped = [index for index, weight in zip(active, weights) if weight < 0]
        removed.extend(dropped)
        active = [index for index in active if index not in dropped]
    full = np.zeros(matrix.shape[0])
    full[active] = weights
    return full, sorted(removed)


def estimate_correlation(estimates: Any) -> np.ndarray:
    """Correlation of per-realization estimates stacked tier-major: ``(n, M, P) -> (M P, M P)``."""
    values = np.asarray(estimates, dtype=float)
    if values.ndim != 3 or values.shape[0] < 3 or not np.isfinite(values).all():
        raise ValueError(
            "per-realization estimates must be a finite (realization, tier, parameter) array "
            "with at least three realizations"
        )
    stacked = values.reshape(values.shape[0], -1)
    if np.any(stacked.std(axis=0) <= 0):
        raise ValueError("every tier estimate needs a nonzero realization scatter")
    return np.corrcoef(stacked, rowvar=False)


def _validated_tiers(labels, means, covariances, parameters):
    labels = [str(label) for label in labels]
    if len(labels) < 2 or len(set(labels)) != len(labels):
        raise ValueError("a joint constraint needs at least two distinct tiers")
    if EXCLUDED_JOINT_TIER in labels:
        raise ValueError("the native tier is the reference and never enters the joint constraint")
    count = len(parameters)
    means = np.asarray(means, dtype=float)
    covariances = np.asarray(covariances, dtype=float)
    if means.shape != (len(labels), count) or covariances.shape != (len(labels), count, count):
        raise ValueError("tier means and covariances must match the tier and parameter counts")
    if not (np.isfinite(means).all() and np.isfinite(covariances).all()):
        raise ValueError("tier posteriors must be finite")
    for block in covariances:
        np.testing.assert_allclose(block, block.T, rtol=0, atol=1e-12)
        np.linalg.cholesky(block)
    return labels, means, covariances


def combine_tier_posteriors(
    labels: Sequence[str],
    means: Any,
    covariances: Any,
    correlation: Any,
    *,
    parameters: Sequence[str] = JOINT_PARAMETERS,
) -> dict:
    """Combine tier posteriors parameter by parameter with non-negative BLUE weights.

    ``correlation`` is the stacked tier-major correlation of the per-realization estimates. The
    estimate covariance keeps every tier's own posterior block exactly and uses the measured
    correlations only between tiers. Each parameter's weights and variance come from its own
    block, the cross term from both weight vectors on the stacked cross block, and the figure of
    merit is ``1 / sqrt(det)`` like every other route's.
    """
    labels, means, covariances = _validated_tiers(labels, means, covariances, parameters)
    tiers, count = means.shape
    correlation = np.asarray(correlation, dtype=float)
    if correlation.shape != (tiers * count, tiers * count) or not np.isfinite(correlation).all():
        raise ValueError("the estimate correlation must be square over tiers times parameters")
    widths = np.sqrt(np.einsum("tii->ti", covariances)).reshape(-1)
    stacked = correlation * np.outer(widths, widths)
    for tier in range(tiers):
        block = slice(tier * count, (tier + 1) * count)
        stacked[block, block] = covariances[tier]
    # Weights are solved parameter by parameter on the measured cross-tier correlations and the
    # tiers' own widths; only the stacked matrix that yields the joint covariance is projected,
    # because exact posterior blocks and measured cross blocks need not be jointly definite.
    weights = np.zeros((count, tiers))
    removed = {}
    for index, parameter in enumerate(parameters):
        coordinates = [tier * count + index for tier in range(tiers)]
        block = nearest_positive_definite(stacked[np.ix_(coordinates, coordinates)])
        weights[index], dropped = nonnegative_blue_weights(block)
        removed[parameter] = [labels[tier] for tier in dropped]
    smallest = float(np.linalg.eigvalsh(0.5 * (stacked + stacked.T)).min())
    # The joint covariance: each parameter's variance is its own BLUE variance on its block, so it
    # never exceeds the best tier's; the cross term applies both weight vectors to the stacked
    # cross block (exact within-tier posterior cross-covariances on its diagonal, measured
    # cross-tier correlations elsewhere) and is clipped only if it would break definiteness.
    joint_covariance = np.zeros((count, count))
    for row in range(count):
        for col in range(count):
            block = stacked[
                np.ix_(
                    [tier * count + row for tier in range(tiers)],
                    [tier * count + col for tier in range(tiers)],
                )
            ]
            joint_covariance[row, col] = weights[row] @ block @ weights[col]
    joint_covariance = 0.5 * (joint_covariance + joint_covariance.T)
    clipped = False
    if count == 2:
        bound = 0.99 * np.sqrt(joint_covariance[0, 0] * joint_covariance[1, 1])
        if abs(joint_covariance[0, 1]) >= bound:
            joint_covariance[0, 1] = joint_covariance[1, 0] = (
                np.sign(joint_covariance[0, 1]) * bound
            )
            clipped = True
    np.linalg.cholesky(joint_covariance)
    return {
        "parameters": list(parameters),
        "tiers": labels,
        "mean": (weights * means.T).sum(axis=1).tolist(),
        "covariance": joint_covariance.tolist(),
        "standard_deviations_ddof1": np.sqrt(np.diag(joint_covariance)).tolist(),
        "two_dimensional_fom": float(1 / np.sqrt(np.linalg.det(joint_covariance))),
        "weights": {
            parameter: weights[index].tolist() for index, parameter in enumerate(parameters)
        },
        "removed_tiers": removed,
        "stacked_minimum_eigenvalue": smallest,
        "cross_term_clipped": clipped,
    }


def leave_one_out_combination(
    labels: Sequence[str], estimates: Any, *, parameters: Sequence[str] = JOINT_PARAMETERS
) -> dict:
    """Coverage check: combine each realization with weights learned from the others.

    ``estimates`` holds one profile estimate per realization, tier and parameter. For every
    held-out realization the tier widths are the other realizations' scatters and the
    correlation is theirs too, so the held-out combination never sees its own estimate. The
    scatter of the held-out combinations is reported beside every tier's own scatter; a joint
    constraint whose scatter exceeds the best tier's does not improve on it.
    """
    labels = [str(label) for label in labels]
    if EXCLUDED_JOINT_TIER in labels:
        raise ValueError("the native tier is the reference and never enters the joint constraint")
    values = np.asarray(estimates, dtype=float)
    count = len(parameters)
    if values.ndim != 3 or values.shape[1:] != (len(labels), count) or values.shape[0] < 5:
        raise ValueError(
            "estimates must be (realization, tier, parameter) with at least five realizations"
        )
    if not np.isfinite(values).all():
        raise ValueError("per-realization estimates must be finite")
    realizations = values.shape[0]
    combined = np.zeros((realizations, count))
    for held in range(realizations):
        others = np.delete(values, held, axis=0)
        correlation = estimate_correlation(others)
        widths = others.std(axis=0, ddof=1).reshape(-1)
        stacked = nearest_positive_definite(correlation * np.outer(widths, widths))
        for index in range(count):
            coordinates = [tier * count + index for tier in range(len(labels))]
            weights, _ = nonnegative_blue_weights(stacked[np.ix_(coordinates, coordinates)])
            combined[held, index] = float(weights @ values[held, :, index])
    tier_scatter = values.std(axis=0, ddof=1)
    scatter = combined.std(axis=0, ddof=1)
    best = tier_scatter.argmin(axis=0)
    return {
        "parameters": list(parameters),
        "tiers": labels,
        "estimates": combined.tolist(),
        "mean": combined.mean(axis=0).tolist(),
        "scatter": scatter.tolist(),
        "tier_scatter": {label: tier_scatter[tier].tolist() for tier, label in enumerate(labels)},
        "best_tier": [labels[tier] for tier in best],
        "best_tier_scatter": tier_scatter[best, np.arange(count)].tolist(),
        "improves": (scatter < tier_scatter[best, np.arange(count)]).tolist(),
    }


def realization_combination(
    labels: Sequence[str], estimates: Any, *, parameters: Sequence[str] = JOINT_PARAMETERS
) -> dict:
    """Weights from the realization scatter and correlation, and the joint width they imply.

    This is the per-realization form of the combination: tier widths are the realization
    scatters of the profile estimates and the correlation is theirs, so the weights describe how
    single-realization estimates combine. The implied width ``sqrt(w^T Psi w)`` is the joint
    single-realization width on the same footing as the tier scatters it is built from.
    """
    labels = [str(label) for label in labels]
    if EXCLUDED_JOINT_TIER in labels:
        raise ValueError("the native tier is the reference and never enters the joint constraint")
    values = np.asarray(estimates, dtype=float)
    count = len(parameters)
    if values.ndim != 3 or values.shape[1:] != (len(labels), count) or values.shape[0] < 4:
        raise ValueError(
            "estimates must be (realization, tier, parameter) with at least four realizations"
        )
    correlation = estimate_correlation(values)
    widths = values.std(axis=0, ddof=1).reshape(-1)
    stacked = nearest_positive_definite(correlation * np.outer(widths, widths))
    weights, implied, removed = np.zeros((count, len(labels))), [], {}
    for index, parameter in enumerate(parameters):
        coordinates = [tier * count + index for tier in range(len(labels))]
        block = stacked[np.ix_(coordinates, coordinates)]
        weights[index], dropped = nonnegative_blue_weights(block)
        implied.append(float(np.sqrt(weights[index] @ block @ weights[index])))
        removed[parameter] = [labels[tier] for tier in dropped]
    return {
        "parameters": list(parameters),
        "tiers": labels,
        "weights": {
            parameter: weights[index].tolist() for index, parameter in enumerate(parameters)
        },
        "removed_tiers": removed,
        "tier_scatter": {
            label: values[:, tier, :].std(axis=0, ddof=1).tolist()
            for tier, label in enumerate(labels)
        },
        "joint_width": implied,
    }


def bootstrap_joint_width(
    labels: Sequence[str],
    means: Any,
    covariances: Any,
    estimates: Any,
    *,
    resamples: int,
    seed: int,
    parameters: Sequence[str] = JOINT_PARAMETERS,
) -> dict:
    """The joint width's dependence on the measured correlation, by resampling realizations.

    Each resample draws realizations with replacement, re-measures the estimate correlation and
    recombines the same tier posteriors. The 16th and 84th percentiles of the joint widths and
    figure of merit bound what the finite realization count leaves undetermined. Resamples in
    which a tier estimate has no scatter are skipped and counted.
    """
    labels, means, covariances = _validated_tiers(labels, means, covariances, parameters)
    values = np.asarray(estimates, dtype=float)
    if values.ndim != 3 or values.shape[1:] != means.shape or values.shape[0] < 5:
        raise ValueError(
            "estimates must be (realization, tier, parameter) with at least five realizations"
        )
    if int(resamples) < 2:
        raise ValueError("bootstrap needs at least two resamples")
    rng = np.random.default_rng(int(seed))
    widths, foms, skipped = [], [], 0
    for _ in range(int(resamples)):
        chosen = values[rng.integers(0, values.shape[0], values.shape[0])]
        try:
            correlation = estimate_correlation(chosen)
        except ValueError:
            skipped += 1
            continue
        record = combine_tier_posteriors(
            labels, means, covariances, correlation, parameters=parameters
        )
        widths.append(record["standard_deviations_ddof1"])
        foms.append(record["two_dimensional_fom"])
    if len(widths) < 2:
        raise ValueError("too few usable bootstrap resamples")
    widths = np.asarray(widths)
    return {
        "parameters": list(parameters),
        "resamples_used": int(len(widths)),
        "resamples_skipped": int(skipped),
        "seed": int(seed),
        "width_percentiles_16_84": np.percentile(widths, [16, 84], axis=0).T.tolist(),
        "fom_percentiles_16_84": np.percentile(foms, [16, 84]).tolist(),
    }


def joint_tier_constraint(
    labels: Sequence[str],
    means: Any,
    covariances: Any,
    estimates: Any,
    *,
    resamples: int,
    seed: int,
    parameters: Sequence[str] = JOINT_PARAMETERS,
) -> dict:
    """The complete joint-constraint record: combination, coverage check and bootstrap range."""
    correlation = estimate_correlation(estimates)
    return {
        "combination": combine_tier_posteriors(
            labels, means, covariances, correlation, parameters=parameters
        ),
        "estimate_correlation": correlation.tolist(),
        "leave_one_out": leave_one_out_combination(labels, estimates, parameters=parameters),
        "realization_combination": realization_combination(
            labels, estimates, parameters=parameters
        ),
        "bootstrap": bootstrap_joint_width(
            labels,
            means,
            covariances,
            estimates,
            resamples=resamples,
            seed=seed,
            parameters=parameters,
        ),
    }
