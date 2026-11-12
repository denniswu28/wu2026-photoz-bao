"""Reusable fitting diagnostics. Inputs and experiment choices are explicit arguments."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import copy

import pzbao.fitting as fitting_fitting

import pzbao.likelihood as fitting_likelihood

from contextlib import contextmanager, ExitStack

from unittest.mock import patch

from typing import Any, Mapping

import html

import json

import fnmatch

_broadband_audit_PARAMETERS = ("qiso", "qap")


def detection_statistics(chi2_bao, chi2_smooth):
    """Signed data-fit difference and historical nonnegative descriptive proxy."""
    if not np.all(np.isfinite([chi2_bao, chi2_smooth])):
        raise ValueError("Detection requires finite data chi-square values")
    delta = float(chi2_smooth - chi2_bao)
    return {"delta_chi2": delta, "sqrt_delta_chi2_proxy": float(np.sqrt(max(delta, 0.0)))}


def precision_metrics(posterior):
    if posterior is None:
        return {"uncertainty": None, "precision_1d": None, "joint_fom": None}
    if posterior["parameters"] != list(_broadband_audit_PARAMETERS):
        raise ValueError("joint FoM requires the declared qiso/qap coordinates")
    sd = np.asarray(posterior["standard_deviations_ddof1"], dtype=float)
    matrix = np.asarray(posterior["covariance"], dtype=float)
    if sd.shape != (2,) or matrix.shape != (2, 2) or (not np.isfinite(matrix).all()):
        raise ValueError("invalid posterior covariance axes or values")
    np.testing.assert_allclose(matrix, matrix.T, rtol=0, atol=1e-12)
    np.linalg.cholesky(matrix)
    np.testing.assert_allclose(sd, np.sqrt(np.diag(matrix)), rtol=1e-10, atol=1e-12)
    if not np.all(np.isfinite(sd)) or np.any(sd <= 0):
        raise ValueError("invalid marginal posterior uncertainties")
    joint = float(1 / np.sqrt(np.linalg.det(matrix)))
    np.testing.assert_allclose(joint, posterior["two_dimensional_fom"], rtol=1e-10)
    return {
        "uncertainty": dict(zip(_broadband_audit_PARAMETERS, sd.tolist())),
        "precision_1d": dict(zip(_broadband_audit_PARAMETERS, (1 / sd).tolist())),
        "joint_fom": joint,
    }


def gain_metrics(pre, post):
    first, second = (precision_metrics(pre), precision_metrics(post))
    if first["joint_fom"] is None or second["joint_fom"] is None:
        return {"available": False, "reason": "both paired posteriors are required"}
    ratios = {
        p: second["uncertainty"][p] / first["uncertainty"][p] for p in _broadband_audit_PARAMETERS
    }
    return {
        "available": True,
        "uncertainty_post_over_pre": ratios,
        "precision_post_over_pre": {p: 1 / ratios[p] for p in _broadband_audit_PARAMETERS},
        "joint_fom_post_over_pre": second["joint_fom"] / first["joint_fom"],
    }


def frozen_basis(evaluate, names, coefficients, covariance, expected, *, rtol, atol):
    """Evaluate additive basis directions without reoptimizing other parameters."""
    zero = {name: 0.0 for name in names}
    template = np.asarray(evaluate(zero), dtype=float).copy()
    coefficients = np.asarray([coefficients[name] for name in names], dtype=float)
    scales = np.maximum(1.0, np.abs(coefficients))
    columns = [
        (np.asarray(evaluate({**zero, name: scale})) - template) / scale
        for name, scale in zip(names, scales)
    ]
    basis = np.column_stack(columns)
    broadband = basis @ coefficients
    reconstructed = template + broadband
    if not all((np.isfinite(x).all() for x in (template, basis, coefficients))):
        raise ValueError("nonfinite broadband decomposition")
    explicit = np.asarray(evaluate(dict(zip(names, coefficients))))
    np.testing.assert_allclose(
        explicit, expected, rtol=rtol, atol=atol, err_msg="direct frozen-coefficient model replay"
    )
    np.testing.assert_allclose(
        reconstructed,
        expected,
        rtol=rtol,
        atol=atol,
        err_msg="scaled basis sum at saved coefficients",
    )
    whitened = np.linalg.solve(np.linalg.cholesky(covariance), basis)
    singular = np.linalg.svd(whitened, compute_uv=False)
    rank = int(np.linalg.matrix_rank(whitened))
    return {
        "template": template,
        "broadband": broadband,
        "model": reconstructed,
        "basis": basis,
        "coefficient_names": names,
        "coefficients": coefficients,
        "basis_evaluation_scales": scales,
        "whitened_basis_singular_values": singular,
        "basis_rank": rank,
        "basis_columns": len(names),
        "rank_deficient": rank < len(names),
        "basis_column_whitened_norms": np.linalg.norm(whitened, axis=0),
        "basis_condition_number": float(singular[0] / singular[-1]) if rank == len(names) else None,
    }


_broadband_audit_OBJECTIVE_CONVENTIONS = {
    "profile_logposterior": {
        "nuisance_treatment": ".auto resolved as .best, matching MinuitProfiler",
        "definition": "Data log likelihood at optimized linear nuisances plus physical log prior; no nuisance log-determinant term.",
    },
    "marginalized_logposterior": {
        "nuisance_treatment": ".prec, matching the saved MCMC likelihood",
        "definition": "Precision-projected linear-nuisance marginalization plus physical log prior, in the sampler's normalization convention.",
    },
    "comparison": "Compare changes within the same model and objective. Absolute levels across objectives or nuisance bases are not model evidence.",
}


def set_objective_conventions(likelihood, marginalized):
    """Match the producer explicitly, including before any profiler is attached."""
    likelihood.solved_default = ".best"
    marginalized.solved_default = ".marg"
    for calculator, prefix in ((likelihood, ".auto"), (marginalized, ".prec")):
        solved = [str(p.derived) for p in calculator.all_params if p.solved]
        if not solved or any((not value.startswith(prefix) for value in solved)):
            raise ValueError("unexpected solved nuisance convention for objective review")


def same_point_review(profile, posterior, likelihood, marginalized):
    """Keep independent ensembles and compare rankings within each objective."""
    import pzbao.sampling as fitting_sampling

    set_objective_conventions(likelihood, marginalized)
    fitted = profile["profiles"].bestfit.choice(input=True)
    points = [
        {
            "origin": "saved_profile_MAP",
            "point": {p.name: float(fitted[p.name]) for p in likelihood.varied_params},
        }
    ]
    ensembles = []
    if posterior is not None:
        summary = posterior["sampling_diagnostics"]
        for item in summary["ensembles"]:
            directory = Path(posterior["directory"]).resolve()
            path = (directory / item["path"]).resolve()
            if not path.is_relative_to(directory):
                raise ValueError("chain path escapes posterior directory")
            decoded = fitting_sampling.read_chain(
                path,
                parameter_names=summary["sampled_parameters"],
                identity={**summary["identity"], "ensemble_index": item["ensemble_index"]},
            )
            start = int(summary["burnin_fraction"] * decoded["draws"].shape[1] + 0.5)
            retained = decoded["draws"][:, start:, :]
            if (
                retained.shape[1] < 2
                or retained.shape[0] * retained.shape[1] != item["retained_draws"]
            ):
                raise ValueError("invalid retained ensemble axes")
            chain = decoded["chain"][start:]
            values = np.asarray(chain.logposterior).ravel()
            if values.size != np.prod(chain.shape) or not np.isfinite(values).all():
                raise ValueError("invalid saved posterior objective")
            peak = int(np.argmax(values))
            points.append(
                {
                    "origin": "ensemble_" + str(item["ensemble_index"]) + "_peak",
                    "saved_logposterior": float(values[peak]),
                    "point": {
                        p.name: float(np.asarray(chain[p.name]).ravel()[peak])
                        for p in marginalized.varied_params
                    },
                }
            )
            ensembles.append(
                {
                    "index": item["ensemble_index"],
                    "diagnostics": item,
                    "parameters": summary["sampled_parameters"],
                    "means": retained.mean(axis=(0, 1)).tolist(),
                    "half_means": [
                        part.mean(axis=(0, 1)).tolist()
                        for part in np.array_split(retained, 2, axis=1)
                    ],
                }
            )
    for point in points:
        point["profile_logposterior"] = float(likelihood(**point["point"]))
        point["marginalized_logposterior"] = float(marginalized(**point["point"]))
        if "saved_logposterior" in point:
            np.testing.assert_allclose(
                point["marginalized_logposterior"],
                point["saved_logposterior"],
                rtol=1e-08,
                atol=1e-09,
            )
    reference = points[0]["profile_logposterior"]
    np.testing.assert_allclose(reference, profile["logposterior"], rtol=1e-08, atol=1e-09)
    return {
        "same_point_objectives": points,
        "ensembles": ensembles,
        "objective_conventions": _broadband_audit_OBJECTIVE_CONVENTIONS,
        "saved_profile_objective_verified": True,
        "largest_profile_improvement_over_saved": max(
            (p["profile_logposterior"] - reference for p in points)
        ),
        "interpretation": "Compare rankings within each objective; no global optimum or convergence claim.",
    }


def saved_fit_components(likelihood, point, patterns, matrix, expected):
    """Decompose the saved input point, retaining a fresh analytic solve separately."""
    params = likelihood.all_params.deepcopy()
    names = [p.name for p in params.select(basename=patterns) if str(p.derived).startswith(".")]
    if not names:
        raise ValueError("no active broadband basis")
    coefficients = {}
    for name in names:
        if name not in point:
            raise ValueError("saved fitted point lacks broadband coefficient: " + name)
        value = np.asarray(point[name])
        if value.size != 1 or not np.isfinite(value).all():
            raise ValueError("invalid saved broadband coefficient: " + name)
        coefficients[name] = float(value.item())
        params[name].update(value=coefficients[name], fixed=True, derived=False)
    _, derived = likelihood(**point, return_derived=True)
    resolved = {}
    for name in names:
        value = np.asarray(derived[name])
        if value.size != 1 or not np.isfinite(value).all():
            raise ValueError("invalid analytically resolved coefficient: " + name)
        resolved[name] = float(value.item())
    np.testing.assert_allclose(likelihood.flattheory, expected, rtol=1e-08, atol=1e-10)
    likelihood.all_params = params
    physical = {name: value for name, value in point.items() if name not in names}

    def evaluate(nuisance):
        likelihood(**physical, **nuisance)
        return np.asarray(likelihood.flattheory).copy()

    components = frozen_basis(
        evaluate, names, coefficients, matrix, expected, rtol=1e-08, atol=1e-10
    )
    components.update(
        coefficient_source="saved_profile_input_point",
        analytically_resolved_coefficients=resolved,
        resolved_minus_saved={name: resolved[name] - coefficients[name] for name in names},
    )
    return components


def require_complete_posteriors(records):
    missing = [
        key
        for key, row in records.items()
        if row["posterior"] is None or row["sample_status"] != "complete"
    ]
    if missing:
        raise ValueError("all configured posteriors are required: " + ", ".join(missing))


def freeze_point(likelihood, point, patterns):
    """Disable every analytic solve before the first evaluation of the saved point."""
    params = likelihood.all_params.deepcopy()
    names = [p.name for p in params.select(basename=patterns) if p.solved]
    if not names:
        raise ValueError("no active saved nuisance coordinates")
    values = {}
    for parameter in params:
        name = parameter.name
        if parameter.derived and (not parameter.solved):
            continue
        if (not parameter.fixed or name in names) and name not in point:
            raise ValueError("saved point lacks fitted coefficient or physical parameter: " + name)
        value = np.asarray(point[name] if name in point else parameter.value)
        if value.size != 1 or not np.isfinite(value).all():
            raise ValueError("nonfinite or nonscalar saved parameter: " + name)
        values[name] = float(value.item())
        parameter.update(value=values[name], fixed=True, derived=False)
    likelihood.all_params = params
    if any((p.solved for p in likelihood.all_params)):
        raise ValueError("fixed-point evaluation retained an analytic solve")
    return (values, names)


def decompose(likelihood, smooth, point, patterns, matrix):
    values, names = freeze_point(likelihood, point, patterns)
    smooth_values, smooth_names = freeze_point(smooth, point, patterns)
    if names != smooth_names or values != smooth_values:
        raise ValueError("smooth toggle changed the saved parameter coordinates")

    def evaluate(nuisance):
        likelihood(**{**values, **nuisance})
        return np.asarray(likelihood.flattheory, dtype=float).copy()

    coefficients = {name: values[name] for name in names}
    full = evaluate(coefficients)
    components = frozen_basis(evaluate, names, coefficients, matrix, full, rtol=1e-08, atol=1e-10)
    smooth(**{**smooth_values, **dict.fromkeys(names, 0.0)})
    smooth_template = np.asarray(smooth.flattheory, dtype=float).copy()
    smooth(**smooth_values)
    np.testing.assert_allclose(
        smooth.flattheory,
        smooth_template + components["broadband"],
        rtol=1e-08,
        atol=1e-10,
        err_msg="smooth toggle changed the frozen nuisance contribution",
    )
    data = np.asarray(likelihood.flatdata, dtype=float).copy()
    np.testing.assert_array_equal(data, smooth.flatdata)
    components.update(
        smooth_template=smooth_template,
        smooth_total=smooth_template + components["broadband"],
        bao_template=components["template"] - smooth_template,
        bao_data=data - smooth_template - components["broadband"],
    )
    if any(
        (
            np.asarray(v).shape != full.shape or not np.isfinite(v).all()
            for v in (data, smooth_template, components["bao_template"], components["bao_data"])
        )
    ):
        raise ValueError("invalid fixed-point component vector")
    np.testing.assert_allclose(
        components["smooth_total"] + components["bao_template"], full, rtol=1e-08, atol=1e-10
    )
    np.testing.assert_allclose(
        components["bao_data"] - components["bao_template"], data - full, rtol=1e-08, atol=1e-10
    )
    return components


def validate_pair(first, second):
    for field in ("physical_priors",):
        if json_value(first[field]) != json_value(second[field]):
            raise ValueError("paired fits changed " + field)
    if first["metadata"]["damping_prior"] != second["metadata"]["damping_prior"]:
        raise ValueError("paired fits changed damping")
    for field in ("fit_edges", "multipoles", "separation", "covariance"):
        np.testing.assert_array_equal(first[field], second[field])
    np.testing.assert_array_equal(first["residuals"]["data"], second["residuals"]["data"])


def fit_table(rows):
    values = []
    for label, row in rows:
        item = {
            "fit": label,
            "solution": row["solution"],
            "chi2/dof": f"{row['residuals']['chi2']:.5f}/{row['dof']}",
            "nuisances": row["nuisance_count"],
        }
        for name in _broadband_audit_PARAMETERS:
            p = next((p for p in row["parameters"] if p["basename"] == name))
            item[name] = p["value"]
            item[name + " MINOS bounds"] = p["minos_bounds"]
            item[name + " flags"] = {
                "flags": p["interval_flags"],
                "unavailable": p["interval_unavailable_reason"],
            }
        values.append(item)
    return table(values, tuple(values[0]))


def fit_caption(label, row):
    values = []
    for name in _broadband_audit_PARAMETERS:
        parameter = next((p for p in row["parameters"] if p["basename"] == name))
        flags = parameter["interval_flags"] or {}
        notes = []
        if not flags.get("is_valid"):
            notes.append("invalid/unavailable interval")
        if flags.get("at_lower_limit") or flags.get("at_upper_limit"):
            notes.append("prior boundary")
        values.append(
            f"{name}={parameter['value']:.4f}" + (" (" + ", ".join(notes) + ")" if notes else "")
        )
    return (
        f"{label}: " + "; ".join(values) + f"; χ²/dof={row['residuals']['chi2']:.2f}/{row['dof']}"
    )


class ExplicitStartsProfiler:
    """Supply recorded starts to the existing validity-preserving native selector."""

    def __init__(self, profiler, starts):
        self.native = profiler
        self.starts = np.asarray(starts, dtype=float)

    def _get_start(self, *, niterations):
        if niterations != len(self.starts):
            raise ValueError("explicit optimizer start count differs")
        return self.starts.copy()

    @property
    def profiles(self):
        return self.native.profiles

    @profiles.setter
    def profiles(self, value):
        self.native.profiles = value

    def maximize(self, **kwargs):
        return self.native.maximize(**kwargs)


def fixed_prior_starts(profiler, config, recorded, *, smooth=False):
    """Keep reference starts; add stratified dilations and identified physical points."""
    count = config["fitting"]["optimization"]["profiler_iterations"]
    seed = config["fitting"]["optimization"]["random_seed"]
    parameters = list(profiler.varied_params)
    names = [p.name for p in parameters]
    native = profiler._get_start(niterations=count)
    if native.shape != (count, len(names)):
        raise ValueError("native start axes differ from physical parameter order")
    broad = native.copy()
    rng = np.random.RandomState(seed)
    dilation_names = config["fitting"]["ap_interval_parameters"]["qisoqap"]
    active = {p.basename for p in parameters}
    if smooth and set(dilation_names) & active:
        raise ValueError("smooth comparison must exclude inactive dilations")
    if not smooth and (not set(dilation_names) <= active):
        raise ValueError("repair requires both configured dilation parameters")
    for index, parameter in enumerate(parameters):
        if parameter.basename not in dilation_names:
            continue
        lower, upper = parameter.prior.limits
        if not np.isfinite([lower, upper]).all() or lower >= upper:
            raise ValueError("dilation starts require finite unchanged prior limits")
        broad[:, index] = lower + (upper - lower) * (rng.permutation(count) + 0.5) / count
    rows = [
        {"origin": f"reference_{i}", "point": dict(zip(names, values.tolist()))}
        for i, values in enumerate(native)
    ]
    if not smooth:
        rows += [
            {"origin": f"prior_dilation_stratum_{i}", "point": dict(zip(names, values.tolist()))}
            for i, values in enumerate(broad)
        ]
    rows += copy.deepcopy(recorded)
    for row in rows:
        point = row["point"]
        if set(point) != set(names):
            raise ValueError("recorded start physical parameter names differ")
        for parameter in parameters:
            value = point[parameter.name]
            lower, upper = parameter.prior.limits
            if not np.isfinite(value) or not lower <= value <= upper:
                raise ValueError("recorded start lies outside the unchanged prior")
    return rows


def maximize_recorded(profiler, config, starts, *, selector=None):
    """Reuse the original per-start validity checks and retain every starting point."""
    names = [p.name for p in profiler.varied_params]
    if not starts or any((set(row["point"]) != set(names) for row in starts)):
        raise ValueError("explicit optimizer start names differ")
    values = np.asarray([[row["point"][name] for name in names] for row in starts])
    if not np.isfinite(values).all():
        raise ValueError("nonfinite explicit optimizer start")
    for index, parameter in enumerate(profiler.varied_params):
        lower, upper = parameter.prior.limits
        if np.any(values[:, index] < lower) or np.any(values[:, index] > upper):
            raise ValueError("explicit optimizer start is outside the unchanged prior")
    execution = copy.deepcopy(config)
    execution["fitting"]["optimization"]["profiler_iterations"] = len(starts)
    selector = fitting_fitting.maximize_usable if selector is None else selector
    best = selector(ExplicitStartsProfiler(profiler, values), execution)
    for record, start in zip(best.attrs["optimizer_starts"], starts):
        record.update(copy.deepcopy(start))
    best.attrs["optimizer_search"] = {
        "method": "reference_plus_stratified_dilations_plus_recorded_points",
        "reference_seed": config["fitting"]["optimization"]["random_seed"],
        "priors": {p.name: p.prior.__getstate__() for p in profiler.varied_params},
        "global_optimum_established": False,
    }
    return best


def saved_two_basin_points(directory, producer, route, names):
    """Read both native solutions with direct producer, route and objective identities."""
    directory = Path(directory).resolve()
    record = _read_json(directory / "restarts.json")
    if (
        any((record.get(k) != v for k, v in producer["identity"].items()))
        or record["route"] != route
    ):
        raise ValueError("restart record producer or route differs")
    attempts = [
        (i, a)
        for i, a in enumerate(record["attempts"])
        if a["model"] == "current" and a["objective"] == "profiling"
    ]
    if not attempts or any((a["status"] != "complete" for _, a in attempts)):
        raise ValueError("complete baseline profile restarts are required")
    points, decoded = ([], [])
    for number, (_, attempt) in enumerate(attempts):
        path = (directory / attempt["profile"]).resolve()
        if not path.is_relative_to(directory):
            raise ValueError("restart profile escapes its directory")
        identity = {
            **producer["identity"],
            "route": route,
            "diagnostic": "matched_recorded_starts",
            "model": "current",
            "objective": "profiling",
            "start_index": number,
        }
        native = fitting_fitting.read_profiles(path, identity=identity)
        np.testing.assert_allclose(
            native["logposterior"], attempt["final_logposterior"], rtol=1e-10, atol=1e-10
        )
        point = attempt["start"]["point"]
        if set(point) != set(names):
            raise ValueError("restart physical parameter names differ")
        points.append(
            {
                "origin": "recorded_" + str(number),
                "point": point,
                "source": attempt["start"],
                "source_profile": str(path),
            }
        )
        decoded.append(native)
    best = max(decoded, key=lambda p: p["logposterior"])
    original = next(
        (
            p
            for p, (_, a) in zip(decoded, attempts)
            if a["start"]["origin"] == "saved_profile_MAP"
            and a["start"]["source_route"].endswith("--current")
        ),
        None,
    )
    if original is None or best["logposterior"] <= original["logposterior"]:
        raise ValueError("saved regression must contain two distinct objective basins")
    fitted = best["profiles"].bestfit.choice(input=True)
    points.append(
        {
            "origin": "controlled_best_solution",
            "point": {name: float(fitted[name]) for name in names},
        }
    )
    return (
        points,
        {
            "old_logposterior": original["logposterior"],
            "better_logposterior": best["logposterior"],
            "route": route,
            "producer": producer["identity"],
        },
    )


def repair_profile(likelihood, config, recorded, output, identity):
    """Write one separate profile and fresh intervals; never replace a source mean."""
    from desilike.profilers import MinuitProfiler

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    priors = {p.name: p.prior.__getstate__() for p in likelihood.varied_params}
    profiler = MinuitProfiler(likelihood, seed=config["fitting"]["optimization"]["random_seed"])
    fitting_likelihood._require_single_process(profiler.mpicomm, "optimizer repair")
    starts = fixed_prior_starts(profiler, config, recorded)
    best = maximize_recorded(profiler, config, starts)
    best.attrs["identity"] = identity
    best.save(str(output / "maximized.npy"))
    interval_names = config["fitting"]["ap_interval_parameters"]["qisoqap"]
    profiles = profiler.interval(params=likelihood.varied_params.select(basename=interval_names))
    flags = fitting_fitting._interval_convergence(profiles, interval_names, strict=False)
    profiles.attrs.update(
        identity=identity,
        optimizer_search=best.attrs["optimizer_search"],
        optimizer_starts=best.attrs["optimizer_starts"],
        selected_start_index=best.attrs["selected_start_index"],
    )
    profiles.save(str(output / "profiles.npy"))
    profiles.to_stats(tablefmt="pretty", fn=str(output / "statistics.txt"))
    decoded = fitting_fitting.read_profiles(output / "profiles.npy", identity=identity)
    if priors != {p.name: p.prior.__getstate__() for p in likelihood.varied_params}:
        raise ValueError("optimizer repair changed a physical prior")
    return {
        "identity": identity,
        "logposterior": decoded["logposterior"],
        "parameters": decoded["parameters"],
        "intervals": flags,
        "optimizer_starts": profiles.attrs["optimizer_starts"],
        "scientific_acceptance": False,
    }


def smooth_points(recorded, names, dilation_names):
    """Project physical starts only across the smooth model's inactive dilations."""
    rows = copy.deepcopy(recorded)
    for row in rows:
        if set(row["point"]) != set(names) | set(dilation_names):
            raise ValueError("smooth start projection differs beyond inactive dilations")
        row["point"] = {name: row["point"][name] for name in names}
    return rows


def json_value(value):
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and (not np.isfinite(value)):
        return {"nonfinite": str(value)}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return {"type": type(value).__name__, "repr": repr(value)}


def injected_vectors(truth, vectors, noise):
    vectors, truth = (np.asarray(vectors), np.asarray(truth))
    if vectors.ndim != 2 or truth.shape != vectors.shape[1:]:
        raise ValueError("injection axes differ")
    residuals = vectors - vectors.mean(axis=0)
    if noise == "none":
        residuals = np.zeros_like(vectors)
    elif noise != "paired_mock_residuals":
        raise ValueError("unknown injection noise definition")
    return truth[None, :] + residuals


def residual_summary(data, model, matrix, separation, multipoles):
    """Signed per-bin contributions sum to chi2; whitening mixes separation bins."""
    residual = np.asarray(data) - np.asarray(model)
    matrix = np.asarray(matrix)
    cholesky = np.linalg.cholesky(matrix)
    whitened = np.linalg.solve(cholesky, residual)
    contributions = residual * np.linalg.solve(matrix, residual)
    count = len(residual) // len(multipoles)
    if len(residual) != count * len(multipoles):
        raise ValueError("residual multipole axes differ")
    s = np.asarray(separation)
    if s.size == count:
        s = np.tile(s.ravel(), len(multipoles))
    else:
        s = s.ravel()
    if s.shape != residual.shape:
        raise ValueError("separation axis differs")
    np.testing.assert_allclose(contributions.sum(), whitened @ whitened)
    return {
        "data": np.asarray(data).tolist(),
        "model": np.asarray(model).tolist(),
        "residual": residual.tolist(),
        "separation": s.tolist(),
        "multipole": np.repeat(multipoles, count).tolist(),
        "diagonal_standardized_residual": (residual / np.sqrt(np.diag(matrix))).tolist(),
        "signed_chi2_contribution": contributions.tolist(),
        "cholesky_whitened_residual": whitened.tolist(),
        "chi2": float(whitened @ whitened),
        "interpretation": "Signed bin contributions include cross-bin covariance; whitened entries are mixed coordinates.",
    }


def parameter_definitions(parameters):
    return [
        {
            "name": p.name,
            "fixed": p.fixed,
            "derived": p.derived,
            "value": p.value,
            "prior": json_value(p.prior.__getstate__()) if p.prior is not None else None,
            "reference": json_value(p.ref.__getstate__()) if p.ref is not None else None,
        }
        for p in parameters
    ]


def noiseless_vector(source, inference, route, saved):
    """Generate one current-model vector at unity, with zero broadband and prior centres."""
    from pzbao import covariance
    import pzbao.likelihood as fitting_likelihood

    original = fitting_likelihood._configure_likelihood_parameters

    def configure(likelihood, damping_prior, **kwargs):
        original(likelihood, damping_prior, **kwargs)
        patterns = kwargs["config"]["fitting"]["linear_nuisance"]["parameter_patterns"]
        for parameter in likelihood.all_params.select(basename=patterns):
            parameter.update(value=0.0, fixed=True, derived=False)

    def read(run, requested):
        if requested != route:
            raise ValueError("noiseless generator requested another route")
        return saved

    with (
        patch.object(covariance, "read_route_covariance", read),
        patch.object(fitting_likelihood, "_configure_likelihood_parameters", configure),
    ):
        _, _, likelihood, _, _, _ = fitting_likelihood._route_likelihood(
            {**inference, "source_run": source},
            {**route, "realization_id": None},
            method="profiling",
        )
        likelihood(qiso=1.0, qap=1.0)
        vector = np.asarray(likelihood.flattheory, dtype=float).copy()
    if vector.shape != saved["arrays"]["vectors"].shape[1:] or not np.all(np.isfinite(vector)):
        raise ValueError("noiseless current-model vector is incompatible")
    return vector


def configure_inference(config, control):
    result = copy.deepcopy(config)
    if "dilation_limits" in control:
        low, high = control["dilation_limits"]
        if not 0 < low < 1 < high:
            raise ValueError("diagnostic dilation bounds must enclose unity")
        for name in ("qiso", "qap"):
            result["fitting"]["nonlinear_parameters"][name]["prior"]["limits"] = [low, high]
    if "broadband" in control:
        if control["broadband"] not in ("power", "pcs", "power3"):
            raise ValueError("unsupported exploratory broadband family")
        if (
            control["broadband"] != "power"
            and control.get("nuisance_solver") != "observed_svd_families_v1"
        ):
            raise ValueError("new broadband families require the declared observed solver")
        result["fitting"]["broadband_model"] = control["broadband"]
    return result


def change_damping(record, control, *, state, baseline=None):
    result = copy.deepcopy(baseline if control.get("baseline_damping") else record)
    if result is None:
        raise ValueError("fixed density prior requires the identified baseline prior")
    if "transverse_center_mpc_h" in control and state == "postreconstruction":
        result["centers_mpc_h"]["sigma_perp"] = control["transverse_center_mpc_h"]
        result["transverse_damping_source"] = "isolated_reference_center_diagnostic"
    if "damping_width_factor" in control:
        factor = float(control["damping_width_factor"])
        if not np.isfinite(factor) or factor <= 0:
            raise ValueError("damping-width factor must be positive")
        result["widths_mpc_h"] = {
            key: value * factor for key, value in result["widths_mpc_h"].items()
        }
    return result


def unmixed_projection(separations, observed_multipoles, model_multipoles):
    """Identity evaluation of observed poles while retaining the raw model's ell=4."""
    count = len(separations)
    model_multipoles = tuple(model_multipoles)
    matrix = np.zeros((count * len(model_multipoles), count * len(observed_multipoles)))
    for index, ell in enumerate(observed_multipoles):
        source = model_multipoles.index(ell)
        matrix[source * count : (source + 1) * count, index * count : (index + 1) * count] = np.eye(
            count
        )
    return matrix


@contextmanager
def likelihood_adapters(fitting, control):
    original_parameters = fitting._configure_likelihood_parameters
    original_window = fitting.likelihood_window_inputs
    original_route = fitting._route_likelihood
    solver = control.get("nuisance_solver")
    if solver is not None:
        from ..likelihood import SOLVER, FAMILY_SOLVER, repaired_observable

        supported = (
            solver == SOLVER
            and control.get("broadband") == "power"
            or (solver == FAMILY_SOLVER and control.get("broadband") in ("pcs", "power3"))
        )
        if not supported or control.get("window") == "none":
            raise ValueError("unsupported exploratory nuisance solver control")

    def route_likelihood(*args, **kwargs):
        result = original_route(*args, **kwargs)
        result[3]["power_broadband_repair"] = result[1].power_broadband_repair
        return result

    def parameters(likelihood, damping_prior, **kwargs):
        if control.get("broadband") in ("power", "pcs", "power3"):
            config = copy.deepcopy(kwargs["config"])
            nuisance = config["fitting"]["linear_nuisance"]
            names = [
                str(p.basename)
                for p in likelihood.all_params.select(basename=nuisance["parameter_patterns"])
            ]
            if not names or len(names) != len(set(names)):
                raise ValueError("historical broadband nuisance surface is unavailable")
            if solver is not None:
                names = [name for name in names if name.startswith("alobs_")]
                if not names:
                    raise ValueError("observed SVD nuisance coordinates are missing")
            elif control.get("window") == "none":
                observed = config["fitting"]["fit_surfaces"][kwargs["tracer"]][
                    "observed_multipoles"
                ]
                prefixes = {f"al{ell}" for ell in observed}
                names = [name for name in names if name.split("_", 1)[0] in prefixes]
                if not names:
                    raise ValueError("no observed historical broadband coordinates")
            nuisance["allowed_basenames_by_model"][control["broadband"]] = names
            nuisance["varied_basenames_by_tracer"][kwargs["tracer"]] = names
            kwargs["config"] = config
        original_parameters(likelihood, damping_prior, **kwargs)
        if control.get("likelihood_only"):
            for parameter in likelihood.all_params:
                if not parameter.fixed and (not str(parameter.derived).startswith(".")):
                    limits = list(parameter.prior.limits)
                    parameter.update(prior={"dist": "uniform", "limits": limits})
            if kwargs.get("only_now", False):
                for name in ("sigmapar", "sigmaper"):
                    likelihood.all_params[name].update(fixed=True, derived=False)

    def window(*args, **kwargs):
        result = original_window(*args, **kwargs)
        if control.get("window") == "none":
            result = {**result, "wmatrix": None, "systematic_templates": None}
        return result

    def response(**kwargs):
        from desilike.theories.galaxy_clustering.bao import DampedBAOWigglesPowerSpectrumMultipoles

        return DampedBAOWigglesPowerSpectrumMultipoles(
            template=kwargs["template"],
            mode=kwargs["mode"],
            smoothing_radius=kwargs["smoothing_radius"],
            mu=kwargs["mu_quadrature_order"],
            method="leggauss",
        )

    with ExitStack() as stack:
        stack.enter_context(patch.object(fitting, "_configure_likelihood_parameters", parameters))
        stack.enter_context(patch.object(fitting, "likelihood_window_inputs", window))
        if control.get("response") == "historical":
            stack.enter_context(
                patch.object(fitting, "_build_observed_photoz_power_model", response)
            )
        if control.get("window") == "none" or solver is not None:
            from desilike.observables import galaxy_clustering as observables

            original_observable = observables.TracerCorrelationFunctionMultipolesObservable

            def observable(**kwargs):
                if solver is not None:
                    return repaired_observable(
                        original_observable, kwargs, model=control["broadband"]
                    )
                if kwargs["wmatrix"] is not None:
                    raise ValueError("no-window adapter received a nonempty window")
                kwargs["wmatrix"] = unmixed_projection(
                    kwargs["s"], kwargs["ells"], kwargs["ellsin"]
                )
                kwargs["sin"] = np.asarray(kwargs["s"])
                return original_observable(**kwargs)

            stack.enter_context(
                patch.object(
                    observables, "TracerCorrelationFunctionMultipolesObservable", observable
                )
            )
        if solver is not None:
            stack.enter_context(patch.object(fitting, "_route_likelihood", route_likelihood))
        yield


_lrg_mesh_mean_fits_INTERPRETATION = "Exploratory fits to all 25 paired mock means. The source postreconstruction RascalC covariance is held fixed across meshes within each bin and divided by 25 exactly once. Native fitting settings and each arm's mock-0 RR window are retained. Profile intervals are conditional on this covariance, not posterior precision, ensemble scatter, coverage or a validation verdict. No power-spectrum fits."


def replace_vectors(saved, vectors, separations, mock_ids, window):
    """Keep the source covariance and its already-applied mean scaling unchanged."""
    result = copy.deepcopy(saved)
    arrays = result["arrays"]
    np.testing.assert_array_equal(arrays["realization_ids"], mock_ids)
    for key, values in (("vectors", vectors), ("separations", separations)):
        values = np.asarray(values)
        if values.shape != arrays[key].shape or not np.isfinite(values).all():
            raise ValueError(f"mesh {key} do not match the complete covariance ensemble")
        arrays[key] = values.copy()
    np.testing.assert_allclose(
        arrays["mean_covariance"] * len(mock_ids), arrays["covariance"], rtol=1e-12, atol=0
    )
    for key in ("sedges", "muedges"):
        np.testing.assert_array_equal(window[key], saved["window"][key])
    weights = np.asarray(window["wcounts"])
    if weights.shape != saved["window"]["wcounts"].shape or not np.isfinite(weights).all():
        raise ValueError("mesh RR window has incompatible weights")
    if np.any(weights < 0):
        raise ValueError("mesh RR window has negative weights")
    result["window"] = copy.deepcopy(window)
    result["metadata"]["mean_covariance_interpretation"] = _lrg_mesh_mean_fits_INTERPRETATION
    return result


def comparison_rows(records, baseline_name):
    """Retain failures and interval flags; never turn an invalid interval into precision."""
    baselines = {
        r["bin"]: r for r in records if r["case"] == baseline_name and r["status"] == "complete"
    }
    result = copy.deepcopy(records)
    for row in result:
        row["difference_from_reference"] = None
        if row["status"] != "complete" or row["bin"] not in baselines:
            continue
        base = baselines[row["bin"]]
        row["difference_from_reference"] = {
            parameter: row["parameters"][parameter]["value"]
            - base["parameters"][parameter]["value"]
            for parameter in ("qiso", "qap")
        }
    return result


def paired_summary(left, right):
    left, right = (np.asarray(left, dtype=float), np.asarray(right, dtype=float))
    if left.shape != right.shape or left.ndim != 1 or len(left) < 2:
        raise ValueError("paired scalar comparison needs equal nontrivial cohorts")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("nonfinite fits must be counted separately, not silently omitted")
    delta = right - left
    return {
        "n": len(delta),
        "mean_difference": float(delta.mean()),
        "paired_scatter_ddof1": float(delta.std(ddof=1)),
        "paired_mean_standard_error": float(delta.std(ddof=1) / np.sqrt(len(delta))),
    }


def factorial(base, first, second, both):
    return np.asarray(both) - np.asarray(first) - np.asarray(second) + np.asarray(base)


def fit_metrics(records, *, mock_ids=None):
    """Retain failures and rails beside shifts; do not silently select successful fits."""
    all_individual = [r for r in records if r["mock"] is not None]
    found_ids = [r["mock"] for r in all_individual]
    if len(set(found_ids)) != len(found_ids):
        raise ValueError("duplicate mock identity")
    if mock_ids is not None:
        mock_ids = list(mock_ids)
        if len(set(mock_ids)) != len(mock_ids):
            raise ValueError("duplicate requested mock identity")
        if set(found_ids) - set(mock_ids):
            raise ValueError("fit cohort contains unexpected mock identities")
        # Absent files count as missing; callers need not invent failed fit records.
        all_individual = all_individual + [
            {"mock": mock, "status": "missing"} for mock in mock_ids if mock not in found_ids
        ]
    individual = [r for r in all_individual if r["status"] == "complete"]
    result = {
        "n": len(individual),
        "required": len(all_individual),
        "failed_profiles": sum((r["status"] == "failed" for r in all_individual)),
        "missing_profiles": sum((r["status"] == "missing" for r in all_individual)),
        "optimizer_failures": sum(
            (not r["optimizer"]["maximization"]["converged"] for r in individual)
        ),
        "mean_fit_status": next((r["status"] for r in records if r["mock"] is None), "missing"),
        "mean_chi2": float(np.mean([r["chi2"] for r in individual])) if individual else None,
        "mean_delta_chi2": (
            float(np.mean([r["detection"]["delta_chi2"] for r in individual]))
            if individual
            else None
        ),
    }
    if len(individual) != len(all_individual):
        result["interpretation"] = (
            "Incomplete cohort: no ensemble dilation or paired gain is inferred."
        )
        return result
    if not individual:
        return result
    for name in ("qiso", "qap"):
        params = [
            next((p for p in r["parameters"] if p["basename"] == name and (not p["fixed"])), None)
            for r in individual
        ]
        if any((p is None for p in params)):
            continue
        values = np.asarray([p["value"] for p in params], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("nonfinite individual fit; cannot silently filter")
        scatter = float(values.std(ddof=1)) if len(values) > 1 else None
        flags = [p["interval_flags"] or {} for p in params]
        result[name] = {
            "mean_offset_from_truth": float(values.mean() - 1),
            "mock_scatter_ddof1": scatter,
            "mean_offset_standard_error": (
                float(scatter / np.sqrt(len(values))) if scatter is not None else None
            ),
            "invalid_intervals": sum((not f.get("is_valid", False) for f in flags)),
            "boundary_intervals": sum(
                (f.get("at_lower_limit", False) or f.get("at_upper_limit", False) for f in flags)
            ),
            "parabolic_errors": [p["parabolic_error"] for p in params],
            "minos_bounds": [p["minos_bounds"] for p in params],
        }
    return result


def log_gap_response(baseline, alternative, historical, *, stable, fraction):
    values = np.asarray([baseline, alternative, historical], dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values <= 0) or historical <= baseline:
        return {"classification": "inconclusive", "reason": "no positive finite reference gap"}
    response = float(np.log(alternative / baseline) / np.log(historical / baseline))
    return {
        "fraction_of_log_gap": response,
        "classification": (
            "inconclusive"
            if not stable
            else (
                "major conditional contributor"
                if response >= fraction
                else "insufficient alone at the measured point estimate"
            )
        ),
        "reason": "fixed-covariance sensitivity; not replacement precision or a validation verdict",
    }


def quadrature_parameter_settings(
    damping_prior: Mapping[str, Any], *, ap_mode: str, config: Mapping[str, Any]
) -> "dict[str, dict]":
    """Explicit study points: configured prior centres and sharp damping/dilation corners.

    These points describe a bounded diagnostic study, not a certificate for the full prior.
    Bias and growth stay at their configured values; changing the scientific model is separate.
    """
    nonlinear = config["fitting"]["nonlinear_parameters"]
    for name in ("centers_mpc_h", "widths_mpc_h", "bounds_mpc_h"):
        if not isinstance(damping_prior.get(name), Mapping):
            raise ValueError("quadrature study requires the resolved damping prior")
    centres = damping_prior["centers_mpc_h"]
    widths = damping_prior["widths_mpc_h"]
    bounds = damping_prior["bounds_mpc_h"]
    shared = {"b1": float(nonlinear["b1"]["value"]), "dbeta": float(nonlinear["dbeta"]["value"])}
    centre = dict(
        shared,
        sigmas=float(centres["sigma_streaming"]),
        sigmapar=float(centres["sigma_parallel"]),
        sigmaper=float(centres["sigma_perp"]),
    )

    def floor(name: str) -> float:
        limit = bounds[name][0]
        return 0.0 if limit is None else float(limit)

    sharp = dict(
        shared,
        sigmas=float(centres["sigma_streaming"]) + 3.0 * float(widths["sigma_streaming"]),
        sigmapar=floor("sigma_parallel"),
        sigmaper=floor("sigma_perp"),
    )
    dilation_lo, dilation_hi = [float(value) for value in nonlinear["qiso"]["prior"]["limits"]]
    settings = {
        "prior_centre": dict(centre, qiso=float(nonlinear["qiso"]["value"])),
        "sharpest_low_dilation": dict(sharp, qiso=dilation_lo),
        "sharpest_high_dilation": dict(sharp, qiso=dilation_hi),
    }
    if ap_mode == "qisoqap":
        anisotropy_lo, anisotropy_hi = [
            float(value) for value in nonlinear["qap"]["prior"]["limits"]
        ]
        settings["prior_centre"]["qap"] = float(nonlinear["qap"]["value"])
        settings["sharpest_low_dilation"]["qap"] = anisotropy_lo
        settings["sharpest_high_dilation"]["qap"] = anisotropy_hi
    return settings


def real_model_quadrature_study(
    *,
    template,
    mode,
    smoothing_radius,
    pairwise_sigma_mpc_h,
    broadband,
    model_multipoles,
    parameter_settings,
    orders,
    reference_order,
    maximum_fractional_difference,
):
    """Compare explicitly requested rules through the actual correlation-function model.

    The wrapper supplies the production Fourier grid, AP mapping, reconstruction and broadband
    context. Read the power multipoles from that initialized model. This callable is an offline
    diagnostic only: it never changes a production order or constructs an acceptance credential.
    """
    from desilike.theories.galaxy_clustering import (
        DampedBAOWigglesTracerCorrelationFunctionMultipoles,
    )
    from ..models.photoz_bao import build_observed_photoz_power_model
    from ..models.mu_quadrature import compare_multipole_orders, symmetric_leggauss

    orders = list(orders)
    if (
        not orders
        or len(set(orders)) != len(orders)
        or any((type(value) is not int or value < 1 for value in [*orders, reference_order]))
        or (max(orders) >= reference_order)
    ):
        raise ValueError(
            "offline study needs distinct positive orders below an explicit finer reference"
        )
    if not parameter_settings:
        raise ValueError("offline study requires explicit parameter settings")
    poles = tuple((int(value) for value in model_multipoles))
    evaluated, grids = ({}, {})
    for order in [*orders, reference_order]:
        power = build_observed_photoz_power_model(
            template=template,
            mode=mode,
            smoothing_radius=smoothing_radius,
            pairwise_sigma_mpc_h=pairwise_sigma_mpc_h,
            mu_quadrature_order=order,
        )
        theory = DampedBAOWigglesTracerCorrelationFunctionMultipoles(
            template=template,
            mode=mode,
            smoothing_radius=smoothing_radius,
            broadband=broadband,
            pt=power,
            ells=poles,
        )
        evaluated[order] = {}
        for label, values in parameter_settings.items():
            theory(**values)
            if not np.allclose(power.mu, symmetric_leggauss(order)[0], rtol=0, atol=1e-12):
                raise ValueError("offline model did not retain its requested quadrature")
            evaluated[order][label] = np.asarray(power.power, dtype=float).copy()
        grids[order] = np.asarray(power.k, dtype=float).copy()
    if any((not np.array_equal(grid, grids[reference_order]) for grid in grids.values())):
        raise ValueError("quadrature candidates use different production Fourier grids")
    comparisons = {
        order: {
            label: compare_multipole_orders(
                value,
                evaluated[reference_order][label],
                multipoles=poles,
                maximum_fractional_difference=maximum_fractional_difference,
            )
            for label, value in evaluated[order].items()
        }
        for order in orders
    }
    return {
        "orders": orders,
        "reference_order": reference_order,
        "multipoles": list(poles),
        "parameter_settings": parameter_settings,
        "k_h_mpc": grids[reference_order].tolist(),
        "comparisons": comparisons,
        "production_order_changed": False,
        "pairwise_sigma_mpc_h": pairwise_sigma_mpc_h,
        "reconstruction_mode": mode,
        "smoothing_radius_mpc_h": smoothing_radius,
        "broadband": broadband,
    }


def basin_references(likelihood, point, fraction):
    """Change initialization references only; keep every physical prior unchanged."""
    if not 0 < fraction < 1 or set(point) != set(likelihood.varied_params.names()):
        raise ValueError("invalid basin initialization definition")
    before = {p.name: p.prior.__getstate__() for p in likelihood.varied_params}
    references = {}
    for parameter in likelihood.varied_params:
        lower, upper = parameter.prior.limits
        center = point[parameter.name]
        if not np.isfinite(center) or not lower <= center <= upper:
            raise ValueError("saved basin point outside unchanged prior")
        ref_lower, ref_upper = parameter.ref.limits
        width = ref_upper - ref_lower
        if not np.isfinite(width):
            width = parameter.ref.scale
        if not np.isfinite(width) or width <= 0:
            raise ValueError("initialization reference has no finite positive width")
        left, right = (max(lower, center - fraction * width), min(upper, center + fraction * width))
        if not left < right:
            raise ValueError("empty initialization interval")
        reference = {"dist": "uniform", "limits": [left, right]}
        likelihood.all_params[parameter.name].update(ref=reference)
        references[parameter.name] = reference
    if before != {p.name: p.prior.__getstate__() for p in likelihood.varied_params}:
        raise ValueError("initialization changed physical priors")
    return references


_post_cluster_comparisons_DAMPING = ("sigma_perp", "sigma_parallel", "sigma_streaming")


def adjust_damping(record, control, reference=None):
    """Apply one declared damping intervention without altering the source record."""
    result = copy.deepcopy(record)
    if "common_flat_damping" in control:
        rule = control["common_flat_damping"]
        if set(control) - {"broadband", "nuisance_solver", "common_flat_damping"}:
            raise ValueError("common flat damping cannot be combined with another intervention")
        if rule != {
            "specification_revision": "2.11-P",
            "distribution": "uniform",
            "bounds_mpc_h": [0.0, 50.0],
            "parameters": list(_post_cluster_comparisons_DAMPING),
            "photoz_response": "unchanged_separate",
        }:
            raise ValueError("unregistered common flat damping prescription")
        for field in ("centers_mpc_h", "widths_mpc_h", "bounds_mpc_h"):
            if set(result[field]) != set(_post_cluster_comparisons_DAMPING):
                raise ValueError("incomplete baseline damping record")
        result["bounds_mpc_h"] = {
            name: list(rule["bounds_mpc_h"]) for name in _post_cluster_comparisons_DAMPING
        }
        result["unpriored_parameters"] = list(_post_cluster_comparisons_DAMPING)
        result["scope"] = "common_flat_damping_sensitivity_2.11-P"
        return result
    if "centre_smoothing_mpc_h" in control:
        if reference is None or set(reference["centers_mpc_h"]) != set(record["centers_mpc_h"]):
            raise ValueError("identified centre prescription is required")
        result["centers_mpc_h"] = copy.deepcopy(reference["centers_mpc_h"])
        result["transverse_damping_source"] = "crossed_saved_smoothing_centre_diagnostic"
        result["centre_reference"] = reference
    if "single_damping_width" in control:
        key = control["single_damping_width"]
        if key not in _post_cluster_comparisons_DAMPING or "centre_smoothing_mpc_h" in control:
            raise ValueError("vary exactly one defined damping width")
        result["widths_mpc_h"][key] *= 2.0
    if result["bounds_mpc_h"] != record["bounds_mpc_h"]:
        raise ValueError("diagnostic changed physical support")
    return result


def table(rows, columns):

    def display(value):
        if value is None:
            return "pending / unavailable"
        if isinstance(value, float):
            return f"{value:.5g}"
        if isinstance(value, (list, dict)):
            return json.dumps(value)
        return str(value)

    return (
        "<table><thead><tr>"
        + "".join((f"<th>{html.escape(k)}</th>" for k in columns))
        + "</tr></thead><tbody>"
        + "".join(
            (
                "<tr>"
                + "".join((f"<td>{html.escape(display(row.get(k)))}</td>" for k in columns))
                + "</tr>"
                for row in rows
            )
        )
        + "</tbody></table>"
    )


def _read_json(path):
    import json

    return json.loads(Path(path).read_text())


def plot_residuals(separation, data, model, covariance, *, ax=None):
    """Show signed residuals in marginal-error units and retain the correlated chi-square."""
    from ..figures.style import axes

    x, data, model, matrix = map(np.asarray, (separation, data, model, covariance))
    if data.shape != model.shape or data.shape != x.shape or matrix.shape != (data.size, data.size):
        raise ValueError("fit residual axes differ")
    np.linalg.cholesky(matrix)
    residual = data - model
    summary = {
        "residual": residual,
        "standardized_residual": residual / np.sqrt(np.diag(matrix)),
        "chi2": float(residual @ np.linalg.solve(matrix, residual)),
    }
    ax = axes(ax)
    ax.plot(x, summary["standardized_residual"])
    ax.axhline(0, color="0.5", linewidth=0.8)
    ax.set(xlabel="s [Mpc/h]", ylabel="(data - model) / marginal error")
    return ax, summary


def flags(record):
    if record is None or record["status"] != "complete":
        return {"missing": True, "invalid": False, "boundary": False}
    intervals = record["optimizer"]["intervals"]
    return {
        "missing": False,
        "invalid": not record["optimizer"]["maximization"]["converged"]
        or any(
            not intervals.get(p, {}).get("is_valid", False) for p in _broadband_audit_PARAMETERS
        ),
        "boundary": any(
            intervals.get(p, {}).get("at_lower_limit", False)
            or intervals.get(p, {}).get("at_upper_limit", False)
            for p in _broadband_audit_PARAMETERS
        ),
    }


def mean_record(record):
    return next((r for r in record["fits"] if r["mock"] is None), None)


def individual_values(record, parameter, mock_ids):
    mock_ids = list(mock_ids)
    if len(set(mock_ids)) != len(mock_ids):
        raise ValueError("duplicate requested mock identity")
    individuals = {r["mock"]: r for r in record["fits"] if r["mock"] is not None}
    if len(individuals) != len([r for r in record["fits"] if r["mock"] is not None]):
        raise ValueError("duplicate mock identity")
    if set(individuals) != set(mock_ids) or any(
        r["status"] != "complete" for r in individuals.values()
    ):
        return None
    values = []
    for mock in mock_ids:
        parameter_record = next(
            (
                p
                for p in individuals[mock]["parameters"]
                if p["basename"] == parameter and not p["fixed"]
            ),
            None,
        )
        if parameter_record is None:
            return None
        values.append(parameter_record["value"])
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("nonfinite individual fit; cannot silently filter")
    return values


def sampling_limits(posterior):
    if posterior is None:
        return ["posterior unavailable"]
    reasons = []
    if posterior["stop_reason"] != "stopping_rule_reached":
        reasons.append("numerical stopping rule not reached")
    ensembles = posterior["sampling_diagnostics"]["ensembles"]
    if any(
        e.get("diagnostics", {})
        .get("parameters", {})
        .get(p, {})
        .get("nominal_autocorrelation_based_ess")
        is None
        for e in ensembles
        for p in _broadband_audit_PARAMETERS
    ):
        reasons.append("autocorrelation-based ESS unavailable")
    return reasons


def summarize_route(record, mock_ids):
    task, posterior = record["task"], record["posterior"]
    mean = mean_record(record)
    good_mean = mean is not None and mean["status"] == "complete"
    individuals = [r for r in record["fits"] if r["mock"] is not None]
    summary = {
        "route": "--".join(
            [
                task["case"]["tracer"],
                task["case"]["condition"],
                task["case"]["name"],
                task["measurement_bin"],
                task["reconstruction_state"],
                task["inference"],
            ]
        ),
        "case": task["case"]["name"],
        "condition": task["case"]["condition"],
        "inference": task["inference"],
        "bin": task["measurement_bin"],
        "state": task["reconstruction_state"],
        "mean_status": mean["status"] if mean else "missing",
        "individual_complete": sum(r["status"] == "complete" for r in individuals),
        "individual_required": len(individuals),
        "individual_failed": sum(r["status"] == "failed" for r in individuals),
        "individual_flagged": sum(
            any(flags(r)[k] for k in ("invalid", "boundary")) for r in individuals
        ),
        "mean_flags": flags(mean),
        "sample_status": record["sample_status"],
        "sampling_limits": sampling_limits(posterior),
        "fom": posterior["two_dimensional_fom"] if posterior else None,
        "chi2": mean["chi2"] if good_mean else None,
        "dof": mean["dof"] if good_mean else None,
        "delta_chi2_bao": mean["detection"]["delta_chi2"] if good_mean else None,
    }
    for p in _broadband_audit_PARAMETERS:
        values = individual_values(record, p, mock_ids)
        fitted = (
            next((v for v in mean["parameters"] if v["basename"] == p), None) if good_mean else None
        )
        summary[p] = {
            "ensemble_bias": float(values.mean() - 1) if values is not None else None,
            "ensemble_sem": (
                float(values.std(ddof=1) / np.sqrt(len(values)))
                if values is not None and len(values) > 1
                else None
            ),
            "posterior_offset": None,
            "posterior_sd": None,
            "mean_profile_value": fitted["value"] if fitted else None,
            "mean_profile_minos_bounds": fitted["minos_bounds"] if fitted else None,
            "mean_profile_interval_flags": fitted["interval_flags"] if fitted else None,
        }
        if posterior is not None and p in posterior["parameters"]:
            index = posterior["parameters"].index(p)
            summary[p].update(
                posterior_offset=posterior["means"][index] - 1,
                posterior_sd=posterior["standard_deviations_ddof1"][index],
            )
    return summary


def measured_comparison(left, right, mock_ids):
    a, b = [summarize_route(r, mock_ids) for r in (left, right)]
    result = {
        "reference": a["route"],
        "alternative": b["route"],
        "parameters": {},
        "fom_ratio": b["fom"] / a["fom"] if a["fom"] and b["fom"] else None,
        "delta_chi2": (
            b["chi2"] - a["chi2"] if a["chi2"] is not None and b["chi2"] is not None else None
        ),
        "delta_dof": b["dof"] - a["dof"] if a["dof"] is not None and b["dof"] is not None else None,
        "limitations": sorted(set(a["sampling_limits"] + b["sampling_limits"])),
        "mean_flags": {"reference": a["mean_flags"], "alternative": b["mean_flags"]},
    }
    for p in _broadband_audit_PARAMETERS:
        av, bv = [individual_values(r, p, mock_ids) for r in (left, right)]
        stats = {
            "paired_individual": (
                paired_summary(av, bv) if av is not None and bv is not None else None
            ),
            "posterior_shift": (
                b[p]["posterior_offset"] - a[p]["posterior_offset"]
                if all(s[p]["posterior_offset"] is not None for s in (a, b))
                else None
            ),
            "posterior_sd_ratio": (
                b[p]["posterior_sd"] / a[p]["posterior_sd"]
                if a[p]["posterior_sd"] and b[p]["posterior_sd"]
                else None
            ),
        }
        result["parameters"][p] = stats
    if any(any(s["mean_flags"].values()) or s["individual_flagged"] for s in (a, b)):
        result["limitations"].append("missing or limited profiles; every flag retained")
    if any(
        result["parameters"][p]["paired_individual"] is None for p in _broadband_audit_PARAMETERS
    ):
        result["limitations"].append("full paired individual cohort unavailable")
    return result


def common_flat_settings(config=None):
    """Return the optional common prior definition without changing the baseline."""
    from ..config import load_analysis_config

    resolved = load_analysis_config() if config is None else config
    return copy.deepcopy(resolved["fitting"]["damping"]["common_flat_sensitivity"])


@contextmanager
def common_flat_parameters():
    """Assert realized uniform priors; omit only redundant smooth-model coordinates."""
    original = fitting_likelihood._configure_likelihood_parameters

    def configure(likelihood, damping_prior, **kwargs):
        original(likelihood, damping_prior, **kwargs)
        if damping_prior.get("scope") != "common_flat_damping_sensitivity_2.11-P":
            return
        for name, key in zip(("sigmaper", "sigmapar", "sigmas"), _post_cluster_comparisons_DAMPING):
            parameter = likelihood.all_params[name]
            if parameter.fixed or parameter.prior.dist != "uniform":
                raise ValueError("common flat likelihood did not realize a free uniform prior")
            np.testing.assert_array_equal(
                parameter.prior.limits, damping_prior["bounds_mpc_h"][key]
            )
        if kwargs.get("only_now", False):
            # The smooth template has no acoustic damping response. Streaming remains free.
            for name in ("sigmaper", "sigmapar"):
                likelihood.all_params[name].update(fixed=True, derived=False)

    with patch.object(fitting_likelihood, "_configure_likelihood_parameters", configure):
        yield


DAMPING = {"sigmas": "sigma_streaming", "sigmapar": "sigma_parallel", "sigmaper": "sigma_perp"}


def validate_comparison_route(route, settings, variant):
    if variant not in ("flat", "pcs2") or (
        route["measurement_bin"] not in settings["measurement_bins"]
        or route["condition"] != settings["condition"]
        or route["reconstruction_state"] != settings["reconstruction_state"]
    ):
        raise ValueError("outside the registered fitting-only paper comparison")


def comparison_config(reference, pcs2, variant):
    """Permit only the declared broadband representation change, or no config change."""
    if variant == "flat":
        return copy.deepcopy(reference)
    if variant != "pcs2":
        raise ValueError("unimplemented comparison")
    result = copy.deepcopy(reference)
    result["schema_version"] = pcs2["schema_version"]
    result["fitting"]["broadband_model"] = pcs2["fitting"]["broadband_model"]
    for key in ("varied_basenames_by_tracer", "allowed_basenames_by_model"):
        result["fitting"]["linear_nuisance"][key] = copy.deepcopy(
            pcs2["fitting"]["linear_nuisance"][key]
        )
    if (
        result != pcs2
        or result["schema_version"] != 14
        or result["fitting"]["broadband_model"] != "pcs2"
    ):
        raise ValueError("PCS2 reference changes unrelated science")
    return result


def paper_flat_prior(record, settings):
    """Use the existing pairwise photo-z scale only to set parallel prior support."""
    from ..theory import finite_flat_damping_prior

    return finite_flat_damping_prior(copy.deepcopy(record), settings["flat_damping"])


@contextmanager
def paper_comparison_adapters(fitting, variant, settings):
    if variant == "pcs2":
        yield
        return
    if variant != "flat":
        raise ValueError("unimplemented comparison")
    original_prior = fitting.resolve_damping_prior
    original_parameters = fitting._configure_likelihood_parameters

    def resolve(*args, **kwargs):
        return paper_flat_prior(original_prior(*args, **kwargs), settings)

    def parameters(likelihood, damping_prior, **kwargs):
        original_parameters(likelihood, damping_prior, **kwargs)
        for name, key in DAMPING.items():
            parameter = likelihood.all_params[name]
            if parameter.prior.dist != "uniform":
                raise ValueError("flat comparison did not install a uniform prior")
            np.testing.assert_array_equal(
                parameter.prior.limits, damping_prior["bounds_mpc_h"][key]
            )
        if kwargs.get("only_now", False):
            # No acoustic signal: these two damping coordinates have no response.
            # Fix only their redundant smooth-template coordinates, as in the
            # existing likelihood-only diagnostic; keep streaming damping free.
            for name in ("sigmapar", "sigmaper"):
                likelihood.all_params[name].update(fixed=True, derived=False)

    with (
        patch.object(fitting, "resolve_damping_prior", resolve),
        patch.object(fitting, "_configure_likelihood_parameters", parameters),
    ):
        yield


def interval_flagged(fits):
    mean = next((r for r in fits if r["mock"] is None and r["status"] == "complete"), None)
    if mean is None:
        return True
    parameters = [p for p in mean["parameters"] if p["basename"] in ("qiso", "qap")]
    return len(parameters) != 2 or any(
        not (p["interval_flags"] or {}).get("is_valid", False)
        or (p["interval_flags"] or {}).get("at_lower_limit", False)
        or (p["interval_flags"] or {}).get("at_upper_limit", False)
        for p in parameters
    )


def verify_components(row):
    c = {
        key: np.asarray(value)
        for key, value in row["components"].items()
        if key
        in (
            "model",
            "template",
            "broadband",
            "smooth_template",
            "smooth_total",
            "bao_template",
            "bao_data",
        )
    }
    data = np.asarray(row["residuals"]["data"])
    for actual, expected in (
        (c["model"], c["template"] + c["broadband"]),
        (c["smooth_total"], c["smooth_template"] + c["broadband"]),
        (c["bao_template"], c["template"] - c["smooth_template"]),
        (c["bao_data"], data - c["smooth_total"]),
    ):
        np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-10)
    residual = data - c["model"]
    chi2 = residual @ np.linalg.solve(np.asarray(row["covariance"]), residual)
    np.testing.assert_allclose(chi2, row["residuals"]["chi2"], rtol=1e-8, atol=1e-9)


def validate_native_comparison(
    fitting, primary, provider, reference, candidate, route, variant, settings
):
    """Cross real likelihoods, including unchanged predictions and physical priors."""
    unit = dict(route, realization_id=None)
    with primary.covariance_provider(provider, route):
        before = fitting._route_likelihood(reference, unit, method="profiling")
        with paper_comparison_adapters(fitting, variant, settings):
            after = fitting._route_likelihood(candidate, unit, method="profiling")
    old, new = before[2], after[2]
    for key in (
        "vectors",
        "separations",
        "fit_edges",
        "multipoles",
        "realization_ids",
        "covariance",
        "mean_covariance",
    ):
        np.testing.assert_array_equal(before[4][key], after[4][key])
    np.testing.assert_array_equal(after[4]["realization_ids"], np.arange(25))
    np.testing.assert_array_equal(before[5], after[5])
    np.testing.assert_allclose(after[5] * 25, after[4]["covariance"], rtol=1e-12, atol=1e-14)
    for key in (
        "z_eff",
        "attained_density_h3_mpc3",
        "reconstruction_inputs",
        "ap_mode",
        "covariance_identity",
        "quadrature",
    ):
        if before[3][key] != after[3][key]:
            raise ValueError("comparison changed physical metadata: " + key)
    expected_damping = (
        paper_flat_prior(before[3]["damping_prior"], settings)
        if variant == "flat"
        else before[3]["damping_prior"]
    )
    if after[3]["damping_prior"] != expected_damping:
        raise ValueError("comparison damping differs from its declared intervention")
    patterns = reference["config"]["fitting"]["linear_nuisance"]["parameter_patterns"]
    physical = lambda p: not any(fnmatch.fnmatch(p.basename, pattern) for pattern in patterns)
    for parameter in old.all_params:
        if physical(parameter) and not (variant == "flat" and parameter.basename in DAMPING):
            if parameter.__getstate__() != new.all_params[parameter.name].__getstate__():
                raise ValueError("comparison changed another physical parameter: " + parameter.name)
    priors = {p.name: p.prior.__getstate__() for p in new.all_params if physical(p)}
    # Pin all broadband coordinates to zero in these disposable likelihoods.
    for model in (old, new):
        for parameter in model.all_params.select(basename=patterns):
            parameter.update(value=0.0, fixed=True, derived=False)
    residuals = []
    for qiso in (1.0, 1.013):
        point = dict(qiso=qiso, qap=1.0)
        old(**point)
        new(**point)
        np.testing.assert_array_equal(old.flatdata, new.flatdata)
        np.testing.assert_allclose(old.flattheory, new.flattheory, rtol=1e-9, atol=1e-10)
        residuals.append(float(np.max(np.abs(old.flattheory - new.flattheory))))
    smooth_difference = None
    if variant == "flat":
        with (
            primary.covariance_provider(provider, route),
            paper_comparison_adapters(fitting, variant, settings),
        ):
            smooth = fitting._route_likelihood(candidate, unit, method="detection", only_now=True)[
                2
            ]
        if any(not smooth.all_params[name].fixed for name in ("sigmapar", "sigmaper")):
            raise ValueError("smooth comparison retained unidentifiable damping coordinates")
        for name in ("sigmapar", "sigmaper"):
            smooth.all_params[name].update(fixed=False)
        predictions = []
        for fraction in (0.2, 0.8):
            point = {
                name: fraction * smooth.all_params[name].prior.limits[1]
                for name in ("sigmapar", "sigmaper")
            }
            smooth(**point)
            predictions.append(np.asarray(smooth.flattheory).copy())
        np.testing.assert_allclose(*predictions, rtol=1e-12, atol=1e-14)
        smooth_difference = float(np.max(np.abs(predictions[0] - predictions[1])))
    return dict(
        physical_priors=priors,
        damping_prior=after[3]["damping_prior"],
        covariance_identity=after[3]["covariance_identity"],
        realization_ids=list(range(25)),
        physical_model_max_abs_differences=residuals,
        smooth_damping_max_abs_difference=smooth_difference,
        scientific_acceptance=False,
    )
