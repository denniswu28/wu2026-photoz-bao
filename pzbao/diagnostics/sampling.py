"""Reusable sampling diagnostics. Inputs and experiment choices are explicit arguments."""

from __future__ import annotations

import numpy as np

import pzbao.sampling as fitting_sampling

import pzbao.likelihood as fitting_likelihood

from pathlib import Path

import hashlib

import json

_longer_chain_QUANTILES = (0.16, 0.5, 0.84)

_longer_chain_STATISTICS = ("mean", "standard_deviation", "q16", "median", "q84")


def statistics(values):
    return np.array(
        [np.mean(values), np.std(values, ddof=1), *np.quantile(values, _longer_chain_QUANTILES)]
    )


def batch_errors(values, tau):
    """Approximate Monte Carlo errors using contiguous time blocks of whole ensembles.

    Walkers stay together in each block. Two block lengths expose sensitivity;
    eight blocks is a minimum to calculate this diagnostic, not an acceptance rule.
    """
    if tau is None or not np.isfinite(tau) or tau <= 0:
        return {"unavailable_reason": "no reliable autocorrelation time", "estimates": []}
    estimates = []
    for multiplier in (2, 4):
        length = max(2, int(np.ceil(multiplier * tau)))
        count = values.shape[0] // length
        record = {"tau_multiplier": multiplier, "block_steps": length, "blocks": count}
        if count < 8:
            record["unavailable_reason"] = "fewer than eight complete time blocks"
        else:
            blocks = values[: count * length].reshape(count, length, values.shape[1])
            estimates_array = np.array([statistics(block) for block in blocks])
            record.update(
                discarded_tail_steps=values.shape[0] - count * length,
                mcse=dict(
                    zip(
                        _longer_chain_STATISTICS,
                        (estimates_array.std(axis=0, ddof=1) / np.sqrt(count)).tolist(),
                    )
                ),
                unavailable_reason=None,
            )
        estimates.append(record)
    return {
        "method": "nonoverlapping whole-ensemble batch statistics; approximate for quantiles and standard deviation",
        "estimates": estimates,
    }


def summarize(chains, *, sampled_parameters, sampling):
    """Report ensemble agreement, tau, and uncertainty in the reported statistics."""
    records, columns = ([], {})
    for index, decoded in enumerate(chains):
        draws, names = (decoded["draws"], decoded["parameters"])
        burn = int(draws.shape[1] * sampling["burnin_fraction"] + 0.5)
        retained = draws[:, burn:, :]
        autocorrelation = fitting_sampling.ensemble_autocorrelation(draws, names, sampling=sampling)
        parameters = {}
        for column, name in enumerate(names):
            values = retained[:, :, column].T
            fixed = name not in sampled_parameters and np.ptp(values) == 0
            diagnostic = autocorrelation["parameters"][name]
            parameters[name] = {
                **dict(zip(_longer_chain_STATISTICS, statistics(values).tolist())),
                "autocorrelation": diagnostic,
                "sampling_error": (
                    {
                        "unavailable_reason": "not applicable: fixed derived quantity",
                        "estimates": [],
                    }
                    if fixed
                    else batch_errors(values, diagnostic["tau"])
                ),
            }
            if fixed:
                parameters[name]["autocorrelation"] = {
                    "tau": None,
                    "nominal_autocorrelation_based_ess": None,
                    "unavailable_reason": "not applicable: fixed derived quantity",
                }
            columns.setdefault(name, []).append(values)
        records.append(
            {
                "ensemble_index": index,
                "steps": draws.shape[1],
                "retained_steps": retained.shape[1],
                "parameters": parameters,
            }
        )
    pooled = {}
    for name, values in columns.items():
        pooled[name] = dict(
            zip(_longer_chain_STATISTICS, statistics(np.concatenate(values, axis=1)).tolist())
        )
        estimates = np.array([statistics(value) for value in values])
        pooled[name]["between_ensemble_mcse_approximation"] = dict(
            zip(
                _longer_chain_STATISTICS,
                (estimates.std(axis=0, ddof=1) / np.sqrt(len(values))).tolist(),
            )
        )
    return {
        "burnin_fraction": sampling["burnin_fraction"],
        "inference_thinning": 1,
        "gr": fitting_sampling.ensemble_gelman_rubin(
            [row["chain"] for row in chains], sampled_parameters, sampling=sampling
        ),
        "ensembles": records,
        "pooled": pooled,
        "interpretation": "descriptive precision diagnostics, not a validation verdict; four ensembles, not individual walkers, are independent replicates",
    }


def validate_checkpoints(decoded, streams, *, identity, sampling, sampled_parameters):
    if len(decoded) != sampling["ensembles"] or len(streams) != len(decoded):
        raise ValueError("incomplete independent ensemble collection")
    shapes = {row["chain"].shape for row in decoded}
    if len(shapes) != 1:
        raise ValueError("parent ensembles have unequal completed steps")
    shape = shapes.pop()
    if len(shape) != 2 or shape[1] != sampling["walkers"]:
        raise ValueError("parent ensemble walker count changed")
    if shape[0] not in fitting_sampling.sampling_check_steps(sampling):
        raise ValueError("parent checkpoint is outside its frozen schedule")
    for row, stream in zip(decoded, streams):
        if row["attrs"].get("identity") != {**identity, "ensemble_index": stream["ensemble_index"]}:
            raise ValueError("parent ensemble identity changed")
        if row["attrs"].get("random_stream") != stream:
            raise ValueError("parent ensemble random stream changed")
        state = row["attrs"].get("rng_state")
        if state is None:
            raise ValueError("parent checkpoint lacks RNG state")
        np.random.RandomState().set_state(state)
        if not set(sampled_parameters).issubset(row["parameters"]):
            raise ValueError("parent checkpoint lacks sampled parameters")
    return shape[0]


def continue_ensembles(
    likelihood, decoded, streams, *, identity, sampling, sampled_parameters, endpoints, destination
):
    """Extend native ensembles to explicit endpoints, irrespective of previous GR stopping."""
    from desilike.samplers import EmceeSampler

    start = decoded[0]["chain"].shape[0]
    if (
        tuple(sorted(set(endpoints))) != tuple(endpoints)
        or not endpoints
        or endpoints[0] <= start
        or (endpoints[-1] > sampling["absolute_iteration_ceiling"])
    ):
        raise ValueError("diagnostic endpoints must increase within the frozen ceiling")
    samplers = []
    for row, stream in zip(decoded, streams):
        seed = np.random.SeedSequence(stream["root_seed"], spawn_key=stream["spawn_key"])
        sampler = EmceeSampler(
            likelihood.deepcopy(),
            nwalkers=sampling["walkers"],
            rng=np.random.RandomState(seed.generate_state(4)),
            chains=[row["chain"].deepcopy()],
        )
        fitting_likelihood._require_single_process(sampler.mpicomm, "longer-chain diagnostic")
        if list(sampler.varied_params.names()) != list(sampled_parameters):
            raise ValueError("continuation changed the sampled coordinates")
        sampler._set_vlikelihood()
        sampler.rng.set_state(row["attrs"]["rng_state"])
        samplers.append(sampler)
    schedule = sorted(
        set(endpoints)
        | {
            step
            for step in fitting_sampling.sampling_check_steps(sampling)
            if start < step <= endpoints[-1]
        }
    )
    paths = [destination / f"ensemble-{index}.npy" for index in range(len(samplers))]
    for step in schedule:
        for index, (sampler, stream, path) in enumerate(zip(samplers, streams, paths)):
            length = step - sampler.chains[0].shape[0]
            sampler.run(
                min_iterations=length,
                max_iterations=length,
                check_every=length,
                check=False,
                thin_by=1,
            )
            if sampler.chains[0].shape != (step, sampling["walkers"]):
                raise ValueError("continuation produced unexpected axes")
            sampler.chains[0].attrs["parent_identity"] = decoded[index]["attrs"]["identity"]
            fitting_sampling._save_ensemble_checkpoint(
                sampler.chains[0], path, sampler, {**identity, "ensemble_index": index}, stream
            )
        print(f"Completed {step} steps in all {len(samplers)} ensembles", flush=True)
        if step in endpoints:
            current = [
                fitting_sampling.read_chain(path, identity={**identity, "ensemble_index": index})
                for index, path in enumerate(paths)
            ]
            yield (
                step,
                summarize(current, sampled_parameters=sampled_parameters, sampling=sampling),
            )


def moment_statistics(values):
    values = np.asarray(values)
    if (
        values.ndim != 2
        or values.shape[1] != 2
        or len(values) < 2
        or (not np.isfinite(values).all())
    ):
        raise ValueError("need finite paired dilation samples")
    sd = values.std(axis=0, ddof=1)
    correlation = float(np.corrcoef(values.T)[0, 1]) if np.all(sd > 0) else None
    return {"mean": values.mean(axis=0), "sd": sd, "correlation": correlation}


def batch_precision(draws):
    """Batch whole ensemble time slices, never pretend walkers are independent chains.

    Batch length must exceed twice the estimated autocorrelation time of every
    first/second/cross moment. Unreliable tau or fewer than eight batches gives
    unavailable MC errors. This diagnostic criterion does not modify sampling.
    """
    from emcee.autocorr import integrated_time, AutocorrError

    draws = np.asarray(draws)
    if draws.ndim != 3 or draws.shape[2] != 2 or (not np.isfinite(draws).all()):
        raise ValueError("expected finite walker,time,parameter axes")
    x, y = (draws[:, :, 0], draws[:, :, 1])
    moments = np.stack((x, y, x * x, y * y, x * y), axis=2).mean(axis=0)
    try:
        taus = np.array(
            [
                float(np.asarray(integrated_time(moments[:, i], quiet=False)).item())
                for i in range(moments.shape[1])
            ]
        )
    except AutocorrError as exc:
        return {"available": False, "reason": str(exc), "mcse": None}
    if not np.isfinite(taus).all() or np.any(taus <= 0):
        return {"available": False, "reason": "nonpositive/nonfinite moment tau", "mcse": None}
    length = max(2, int(np.ceil(2 * taus.max())))
    count = draws.shape[1] // length
    if count < 8:
        return {
            "available": False,
            "reason": "fewer than eight autocorrelation-sized batches",
            "mcse": None,
            "moment_tau": taus,
            "batch_length": length,
            "batches": count,
        }
    estimates = []
    for i in range(count):
        stats = moment_statistics(draws[:, i * length : (i + 1) * length].reshape(-1, 2))
        if stats["correlation"] is None:
            return {"available": False, "reason": "constant batch coordinate", "mcse": None}
        estimates.append([*stats["mean"], *stats["sd"], stats["correlation"]])
    estimates = np.asarray(estimates)
    return {
        "available": True,
        "moment_tau": taus,
        "batch_length": length,
        "batches": count,
        "discarded_tail_steps": draws.shape[1] - count * length,
        "mcse_order": ["mean_qiso", "mean_qap", "sd_qiso", "sd_qap", "correlation"],
        "mcse": estimates.std(axis=0, ddof=1) / np.sqrt(count),
        "interpretation": "Asymptotic temporal-batch MC error; stationarity/mixing still need independent-ensemble review.",
    }


def posterior_statistics(draws):
    draws = np.asarray(draws)
    if draws.ndim != 2 or draws.shape[1] != 2 or len(draws) < 2 or (not np.isfinite(draws).all()):
        raise ValueError("posterior needs finite paired dilation draws")
    covariance = np.cov(draws.T, ddof=1)
    determinant = np.linalg.det(covariance)
    if determinant <= 0:
        raise ValueError("joint posterior covariance is singular")
    return {
        "means": draws.mean(axis=0).tolist(),
        "standard_deviations_ddof1": draws.std(axis=0, ddof=1).tolist(),
        "covariance_ddof1": covariance.tolist(),
        "two_dimensional_fom": float(1 / np.sqrt(determinant)),
        "retained_draws": len(draws),
    }


def plot_trace(draws, parameters, parameter, *, ax=None):
    """Keep walker and iteration axes visible; do not flatten time for chain inspection."""
    from ..figures.style import axes

    values = np.asarray(draws)
    if values.ndim != 3 or values.shape[2] != len(parameters) or parameter not in parameters:
        raise ValueError("chain axes or parameter name differ")
    ax = axes(ax)
    selected = values[:, :, list(parameters).index(parameter)]
    ax.plot(selected.T, alpha=0.35, linewidth=0.7)
    ax.set(xlabel="Iteration", ylabel=parameter)
    return ax, {"draws": selected, "parameter": parameter}


PARAMETERS = ("qiso", "qap")


def read_json(path):
    from ..artifacts import resolve_saved_path

    return json.loads(resolve_saved_path(path).read_text())


def digest(path):
    from ..artifacts import resolve_saved_path

    with resolve_saved_path(path).open("rb") as stream:
        result = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def equal(actual, expected, label):
    if actual != expected:
        raise ValueError(label + " differs")


def verify_files(sources):
    for path, expected in sources.items():
        equal(digest(path), expected, "source bytes: " + str(path))


def joint_summary(draws):
    """The statistic uses the full two-dimensional covariance, including correlation."""
    draws = np.asarray(draws, dtype=float)
    if draws.ndim != 2 or draws.shape[1] != 2 or len(draws) < 3 or not np.isfinite(draws).all():
        raise ValueError("expected finite retained draws with columns qiso, qap")
    covariance = np.cov(draws, rowvar=False, ddof=1)
    try:
        factor = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as error:
        raise ValueError("singular or non-positive posterior covariance") from error
    fom = float(1 / np.prod(np.diag(factor)))
    if not np.isfinite(fom) or fom <= 0:
        raise ValueError("nonfinite joint FoM")
    return dict(fom=fom, covariance=covariance.tolist(), retained_draws=len(draws))


def posterior(root, expected, fitting):
    """Read native ensembles; never interpret fitted mock centres as posterior draws."""
    root = Path(root)
    attempt, summary, complete, model, history = (
        read_json(root / name)
        for name in (
            "attempt.json",
            "sampling.json",
            "complete.json",
            "model.json",
            "gr_history.json",
        )
    )
    equal(attempt["current_definition"], expected["current_definition"], "posterior definition")
    equal(complete["index"], expected["index"], "posterior selector")
    for record in (summary, complete, model, history):
        equal(record["identity"], expected["identity"], "posterior identity")
    equal(model["realization_id"], None, "mean posterior")
    equal(model["broadband_representation"]["broadband_model"], "power3", "posterior model")
    for key, value in expected["model"].items():
        equal(model[key], value, "posterior model field " + key)
    config = expected["current_definition"]["analysis"]
    equal(config["fitting"]["broadband_model"], "power3", "configured posterior model")
    sampling = config["fitting"]["sampling"]
    equal(attempt["sampling"], sampling, "saved sampler configuration")
    equal(history["sampling"], sampling, "GR sampler configuration")
    equal(summary["gr_history"], history["history"], "GR history")
    equal(summary["burnin_fraction"], sampling["burnin_fraction"], "burn-in")
    equal(summary["inference_thinning"], 1, "inference thinning")
    equal(summary["complete_ensemble_collection"], True, "complete ensembles")
    for key in ("stop_reason", "stopping_rule_reached", "budget_exhausted"):
        equal(summary[key], complete[key], "saved " + key)
    streams = fitting.ensemble_random_streams(config, expected["route"])
    rows = summary["ensembles"]
    equal(len(rows), sampling["ensembles"], "ensemble count")
    equal(len(complete["chain_readback"]), len(rows), "readback count")
    files = [
        root / name
        for name in (
            "attempt.json",
            "sampling.json",
            "complete.json",
            "model.json",
            "gr_history.json",
        )
    ]
    files.extend(root / f"ensemble-{i}.npy" for i in range(len(rows)))
    sources = {str(path): digest(path) for path in files}
    retained, limits, shapes = [], [], []
    for index, (row, proof, stream) in enumerate(zip(rows, complete["chain_readback"], streams)):
        equal(row["ensemble_index"], index, "ensemble index")
        equal(row["path"], f"ensemble-{index}.npy", "ensemble path")
        equal(proof["path"], row["path"], "readback path")
        equal(proof["ensemble_index"], index, "readback index")
        equal(row["random_stream"], stream, "ensemble stream")
        decoded = fitting.read_chain(
            root / row["path"],
            parameter_names=summary["sampled_parameters"],
            identity=dict(expected["identity"], ensemble_index=index),
        )
        equal(decoded["axes"], ["walker", "iteration", "parameter"], "native axes")
        names, values = decoded["parameters"], decoded["draws"]
        equal(names, summary["sampled_parameters"], "parameter order")
        equal(decoded["attrs"]["random_stream"], stream, "native stream")
        equal(decoded["attrs"]["acceptance_fraction"], row["acceptance_fraction"], "acceptance")
        shape = [sampling["walkers"], row["steps"], len(names)]
        equal(list(values.shape), shape, "native chain shape")
        equal(proof["shape"], shape, "readback shape")
        if not np.isfinite(values).all() or not set(PARAMETERS).issubset(names):
            raise ValueError("invalid native posterior values or missing dilation parameters")
        shapes.append(shape)
        burn = int(row["steps"] * sampling["burnin_fraction"] + 0.5)
        kept = values[:, burn:, :]
        equal(kept.shape[0] * kept.shape[1], row["retained_draws"], "retained draw count")
        if kept.shape[1] < 2:
            raise ValueError("too few retained iterations")
        retained.append(kept[:, :, [names.index(p) for p in PARAMETERS]].reshape(-1, 2))
        for name in PARAMETERS:
            diagnostic = row["diagnostics"]["parameters"].get(name)
            if diagnostic is None:
                raise ValueError("missing native sampling diagnostic: " + name)
            if diagnostic.get("nominal_autocorrelation_based_ess") is None:
                limits.append(f"ensemble {index} {name}: " + diagnostic["unavailable_reason"])
    if any(shape != shapes[0] for shape in shapes):
        raise ValueError("incomplete common sampling step")
    if summary["budget_exhausted"]:
        limits.append("Configured iteration budget exhausted before numerical stopping.")
    pooled = np.concatenate(retained, axis=0)
    flags = expected["profile"]["parameters"]
    flagged = bool(expected["profile"]["unavailable_reason"])
    for name in PARAMETERS:
        selected = [p for p in flags if p["basename"] == name and not p["fixed"]]
        interval = selected[0].get("interval_flags") if len(selected) == 1 else None
        flagged |= (
            interval is None
            or not interval["is_valid"]
            or not interval["lower_valid"]
            or not interval["upper_valid"]
            or interval["at_lower_limit"]
            or interval["at_upper_limit"]
        )
    # Descriptive contact uses the configured reporting tolerance; it is not an acceptance cut.
    tolerance = config["reporting"]["alpha_rail_tolerance"]
    contact = {}
    for column, name in enumerate(PARAMETERS):
        low, high = config["fitting"]["nonlinear_parameters"][name]["prior"]["limits"]
        contact[name] = dict(
            lower_fraction=float(np.mean(pooled[:, column] <= low + tolerance)),
            upper_fraction=float(np.mean(pooled[:, column] >= high - tolerance)),
        )
    verify_files(sources)
    return dict(
        index=expected["index"],
        route=expected["route"],
        identity=expected["identity"],
        status="available",
        reason=None,
        source=str(root),
        **joint_summary(pooled),
        profile=expected["profile"],
        profile_flagged=flagged,
        prior_contact=contact,
        sampling=summary,
        sampling_limits=limits,
        covariance_validity=expected["covariance_validity"],
        input_sha256=sources,
        covariance_identity=model["covariance_identity"],
        scientific_acceptance=False,
    )
