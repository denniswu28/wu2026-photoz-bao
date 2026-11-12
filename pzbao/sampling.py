"""Native independent ensembles, resumable checkpoints and descriptive chain diagnostics."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np

from .config import same_identity


from .likelihood import _route_likelihood, _require_single_process
from .fitting import read_profiles


def _chain_parameter_draws(path: Path, parameter_names: Sequence[str]) -> np.ndarray:
    """Notebook-facing array view of the shared native ensemble reader."""
    return read_chain(path, parameter_names=parameter_names)["draws"]


def sample_route(run, unit, destination):
    """Sample one route mean with the four independent native ensembles in the snapshot."""
    import yaml

    route = {
        key: unit[key] for key in ("tracer", "measurement_bin", "condition", "reconstruction_state")
    }
    name = "--".join(route[key] for key in ("measurement_bin", "condition", "reconstruction_state"))
    fit_directory = run["root"] / "fit" / "units" / (name + "--mean")
    fitted = yaml.safe_load((fit_directory / "result.yaml").read_text())
    if fitted["analysis_config_sha256"] != run["metadata"]["analysis_config_sha256"]:
        raise ValueError("mean fit belongs to another configuration")
    run_identity = {
        key: run["metadata"][key] for key in ("run_name", "code_commit", "analysis_config_sha256")
    }
    read_profiles(
        fit_directory / "profiles.npy", identity={**run_identity, **route, "realization_id": None}
    )
    _, _, likelihood, metadata, _, _ = _route_likelihood(run, unit, method="mcmc")
    paths = [
        destination / f"ensemble-{index}.npy"
        for index in range(run["config"]["fitting"]["sampling"]["ensembles"])
    ]
    summary = run_sampling(
        likelihood,
        {"chains": paths, "summary": destination / "sampling.json"},
        run["config"],
        ap_mode=metadata["ap_mode"],
        route=route,
        run_identity=run_identity,
    )
    metadata.update(identity=summary["identity"])
    (destination / "model.json").write_text(json.dumps(metadata, indent=2, allow_nan=False))
    return {
        "products": [
            {
                "path": path.name,
                "format": "chain",
                "identity": {**summary["identity"], "ensemble_index": index},
            }
            for index, path in enumerate(paths)
        ]
        + [
            {"path": "sampling.json", "format": "json"},
            {"path": "model.json", "format": "json"},
            {"path": "gr_history.json", "format": "json"},
        ],
        "stop_reason": summary["stop_reason"],
    }


def read_chain(path: Path, *, parameter_names=None, identity=None) -> dict:
    """Decode one native ensemble without flattening its walker or iteration axes.

    Sampling and report consumers share this reader. Identity is a direct run/route/ensemble
    mapping stored inside the native file, not a certificate or a check on output bytes.
    """
    from desilike.samples import Chain

    from .artifacts import resolve_saved_path

    chain = Chain.load(str(resolve_saved_path(path)))
    attrs = dict(chain.attrs)
    if identity is not None and not same_identity(attrs.get("identity"), dict(identity)):
        raise ValueError(f"chain identity differs from the requested ensemble: {path}")
    names = (
        list(parameter_names)
        if parameter_names is not None
        else [
            str(param)
            for param in chain.params()
            if param.basename
            not in {"logposterior", "loglikelihood", "logprior", "aweight", "fweight"}
            and (not getattr(param, "fixed", False) or bool(param.derived))
        ]
    )
    if not names or len(names) != len(set(names)):
        raise ValueError("chain reader requires distinct parameter names")
    columns, retained_names, unavailable = [], [], {}
    for name in names:
        if name not in chain.names():
            raise ValueError(f"chain is missing parameter {name!r}")
        values = np.asarray(chain[name], dtype=float)
        if parameter_names is None and str(chain[name].param.derived).startswith("."):
            unavailable[name] = (
                "analytically solved nuisance; native derivative coefficients are not posterior draws"
            )
            continue
        if values.ndim != 2 or not values.size or not np.all(np.isfinite(values)):
            raise ValueError(f"chain {name!r} must have finite iteration-by-walker values")
        columns.append(values.T)
        retained_names.append(name)
    if not columns:
        raise ValueError("native ensemble contains no scalar posterior draws")
    if len({column.shape for column in columns}) != 1:
        raise ValueError("chain parameters have incompatible axes")
    return {
        "chain": chain,
        "draws": np.stack(columns, axis=-1),
        "parameters": retained_names,
        "axes": ["walker", "iteration", "parameter"],
        "attrs": attrs,
        "unavailable_parameters": unavailable,
    }


def covariance_ratio_eigen_gr(total, within) -> float:
    """Largest symmetric generalized covariance-ratio eigenvalue minus one.

    Compatibility correction for desilike 41f082f: its covariance definitions are preserved,
    but eigvalsh(inv(W) @ V) incorrectly assumes a generally nonsymmetric product is Hermitian.
    No global dependency patch or change to the statistical estimator is involved.
    """
    from scipy.linalg import eigh

    total, within = np.asarray(total, dtype=float), np.asarray(within, dtype=float)
    if (
        total.ndim != 2
        or total.shape[0] == 0
        or total.shape[0] != total.shape[1]
        or within.shape != total.shape
    ):
        raise ValueError("GR requires matching nonempty square covariance matrices")
    for matrix in (total, within):
        if not np.all(np.isfinite(matrix)) or not np.allclose(matrix, matrix.T):
            raise ValueError("GR covariance is non-finite or nonsymmetric")
    scale = np.sqrt(np.diag(total))
    if not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("GR has a constant or invalid sampled coordinate")
    normalized_within = within / scale[:, None] / scale[None, :]
    # Cholesky and eigh both fail on singular W, including an unchanging sampled parameter.
    np.linalg.cholesky(normalized_within)
    values = eigh(
        total / scale[:, None] / scale[None, :],
        normalized_within,
        eigvals_only=True,
        check_finite=True,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("GR covariance ratios are non-finite")
    return float(np.max(values) - 1.0)


def ensemble_gelman_rubin(chains, parameters, *, sampling) -> dict:
    """Compare the native retained ensembles on actually sampled coordinates only."""
    from desilike.samples import diagnostics

    statistic = "max_generalized_covariance_ratio_minus_one"
    record = {"statistic": statistic, "value": None, "usable": False, "reason": None}
    if len(chains) != int(sampling["ensembles"]):
        return {**record, "reason": "incomplete ensemble collection"}
    shapes = [chain.shape for chain in chains]
    if len(set(shapes)) != 1 or len(shapes[0]) != 2:
        return {**record, "reason": "ensembles lack common iteration and walker axes"}
    if shapes[0][1] != int(sampling["walkers"]):
        return {**record, "reason": "ensemble walker count differs from the snapshot"}
    burn = int(shapes[0][0] * float(sampling["burnin_fraction"]) + 0.5)
    retained = [chain[burn:] for chain in chains]
    try:
        if not parameters or any(
            not np.all(np.isfinite(np.asarray(chain[param], dtype=float)))
            for chain in retained
            for param in parameters
        ):
            raise ValueError("GR requires finite values for every sampled coordinate")
        # In this pinned revision method='diag' returns the unscaled native V and W. Its
        # eigen branch rescales V only, so its returned matrices must not be used here.
        # nsplits is the library's single-chain fallback and is inert for these four ensembles.
        with np.errstate(divide="raise", invalid="raise", over="raise"):
            _, (total, within) = diagnostics.gelman_rubin(
                retained,
                params=list(parameters),
                nsplits=int(sampling["nsplits"]),
                method="diag",
                return_matrices=True,
            )
        value = covariance_ratio_eigen_gr(total, within)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        return {**record, "reason": str(exc)}
    return {**record, "value": value, "usable": True, "retained_steps": shapes[0][0] - burn}


def sampling_check_steps(sampling) -> list:
    """Common completed steps, including a ceiling that is off cadence."""
    ceiling, cadence = int(sampling["absolute_iteration_ceiling"]), int(sampling["check_every"])
    minimum = int(sampling["minimum_iterations"])
    if not 0 < minimum <= ceiling or cadence <= 0:
        raise ValueError("invalid sampling minimum, ceiling or check cadence")
    return [step for step in range(cadence, ceiling, cadence) if step >= minimum] + [ceiling]


def update_gr_history(history, check, *, step, sampling) -> list:
    """Append a new common check; replay of the same step cannot earn another pass."""
    rows = [dict(row) for row in history]
    if step not in sampling_check_steps(sampling):
        raise ValueError("GR check is outside the frozen common-step schedule")
    if rows and step <= rows[-1]["step"]:
        if step == rows[-1]["step"]:
            return rows
        raise ValueError("GR checks cannot run backwards")
    expected = sampling_check_steps(sampling)[len(rows)]
    if step != expected:
        raise ValueError("GR history is missing a configured common check")
    value = check.get("value")
    passed = (
        check.get("usable") is True
        and value is not None
        and np.isfinite(value)
        and value < float(sampling["max_eigen_gr"])
    )
    streak = (int(rows[-1]["streak"]) if rows else 0) + 1 if passed else 0
    rows.append(
        {
            **check,
            "step": int(step),
            "passed": bool(passed),
            "streak": streak,
            "stopping_rule_reached": streak >= int(sampling["stable_over"]),
        }
    )
    return rows


def ensemble_random_streams(config, route) -> list:
    """Derive independent streams from stable configuration indices, never Python hashes."""
    sampling = config["fitting"]["sampling"]
    bins = [
        str(item["name"])
        for spec in config["tracers"].values()
        for item in spec["measurement_bins"]
    ]
    tiers = [str(item["label"]) for item in config["selection"]["tiers"]]
    states = ["prereconstruction", "postreconstruction"]
    coordinates = (
        bins.index(str(route["measurement_bin"])),
        tiers.index(str(route["condition"])),
        states.index(str(route["reconstruction_state"])),
    )
    return [
        {
            "root_seed": int(sampling["seed"]),
            "spawn_key": [*coordinates, index],
            "ensemble_index": index,
        }
        for index in range(int(sampling["ensembles"]))
    ]


def ensemble_autocorrelation(draws, names, *, sampling) -> dict:
    """Report emcee/Sokal tau and nominal ESS from unthinned walker time histories."""
    from emcee.autocorr import AutocorrError, integrated_time

    draws = np.asarray(draws, dtype=float)
    if draws.ndim != 3 or draws.shape[2] != len(names):
        raise ValueError("diagnostics require walker-by-iteration-by-parameter draws")
    burn = int(draws.shape[1] * float(sampling["burnin_fraction"]) + 0.5)
    retained = draws[:, burn:, :]
    controls = sampling["autocorrelation"]
    rows = {}
    for index, name in enumerate(names):
        values = retained[:, :, index]
        tau, ess, reason = None, None, None
        if values.shape[1] < 2:
            reason = "fewer than two retained steps"
        elif not np.all(np.isfinite(values)):
            reason = "non-finite parameter history"
        elif np.any(np.ptp(values, axis=1) == 0):
            reason = "one or more walkers are constant on the retained time axis"
        else:
            try:
                with np.errstate(invalid="raise", divide="raise", over="raise"):
                    estimate = float(
                        integrated_time(
                            values.T, c=controls["c"], tol=controls["tol"], quiet=False
                        )[0]
                    )
                if not np.isfinite(estimate) or estimate <= 0:
                    raise ValueError("non-positive or non-finite autocorrelation estimate")
                tau, ess = estimate, float(values.size / estimate)
            except (AutocorrError, ValueError, FloatingPointError) as exc:
                reason = str(exc)
        rows[str(name)] = {
            "tau": tau,
            "nominal_autocorrelation_based_ess": ess,
            "unavailable_reason": reason,
        }
    return {
        "report_only": True,
        "retained_steps": retained.shape[1],
        "walkers": retained.shape[0],
        "parameters": rows,
    }


def _replace_sampler_json(path, payload):
    """Replace a checkpoint within an unpublished attempt; final publication belongs to driver."""
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _save_ensemble_checkpoint(chain, path, sampler, identity, stream):
    chain.attrs.update(
        identity=dict(identity),
        random_stream=dict(stream),
        rng_state=sampler.rng.get_state(),
        acceptance_fraction=np.asarray(sampler.sampler.acceptance_fraction).tolist(),
    )
    path = Path(path)
    temporary = path.with_name(path.stem + ".partial.npy")
    chain.save(str(temporary))
    # Native decoding, required axes and identity are checked before checkpoint replacement.
    read_chain(temporary, parameter_names=list(sampler.varied_params.names()), identity=identity)
    os.replace(temporary, path)


def run_sampling(likelihood, output, config, *, ap_mode, route, run_identity) -> dict:
    """Sample four independent native ensembles within one cumulative route budget.

    Science writes only into the supplied unpublished attempt. Each native checkpoint contains
    the ensemble's own RNG state. A process interruption can leave ensembles at different common
    steps; resume completes that same step before evaluating GR. Descriptive tau/ESS never chooses
    the budget or stop. This function does not submit jobs or publish a completed route.
    """
    from desilike.samplers import EmceeSampler

    sampling = config["fitting"]["sampling"]
    streams = ensemble_random_streams(config, route)
    paths = [Path(path) for path in output["chains"]]
    if len(paths) != len(streams) or len(set(paths)) != len(paths):
        raise ValueError("sampling requires one distinct native path per configured ensemble")
    summary_path = Path(output["summary"])
    history_path = summary_path.with_name("gr_history.json")
    identity = {**dict(run_identity), **dict(route)}
    history = []
    if history_path.exists():
        previous = json.loads(history_path.read_text())
        if not same_identity(previous.get("identity"), identity) or not same_identity(
            previous.get("sampling"), sampling
        ):
            raise ValueError("sampling resume settings or route identity changed")
        for row in previous["history"]:
            history = update_gr_history(history, row, step=row["step"], sampling=sampling)
    samplers = []
    parameters = None
    for path, stream in zip(paths, streams):
        path.parent.mkdir(parents=True, exist_ok=True)
        ensemble_identity = {**identity, "ensemble_index": stream["ensemble_index"]}
        previous = read_chain(path, identity=ensemble_identity) if path.exists() else None
        if previous is not None and not same_identity(
            previous["attrs"].get("random_stream"), stream
        ):
            raise ValueError("saved ensemble random-stream identity changed")
        seed_sequence = np.random.SeedSequence(stream["root_seed"], spawn_key=stream["spawn_key"])
        rng = np.random.RandomState(seed_sequence.generate_state(4))
        # Vectorization mutates calculator caches. Each independently initialized ensemble
        # needs its own likelihood state, including during resume warmup.
        sampler = EmceeSampler(
            likelihood.deepcopy(),
            nwalkers=int(sampling["walkers"]),
            rng=rng,
            chains=[previous["chain"]] if previous is not None else 1,
        )
        _require_single_process(sampler.mpicomm, "route sampling")
        names = list(sampler.varied_params.names())
        if parameters is not None and names != parameters:
            raise ValueError("ensembles sample different parameter coordinates")
        parameters = names
        if previous is not None:
            # JIT/vectorization warmup draws from the library RNG. Warm up first, then restore
            # the saved state, so restarting does not change the future scientific substream.
            sampler._set_vlikelihood()
            state = previous["attrs"].get("rng_state")
            if state is None:
                raise ValueError("ensemble checkpoint has no RNG state")
            sampler.rng.set_state(state)
            steps = previous["draws"].shape[1]
            if steps not in sampling_check_steps(sampling):
                raise ValueError("checkpoint steps do not lie on the frozen sampling schedule")
            if history and steps < history[-1]["step"]:
                raise ValueError("GR history is ahead of a required ensemble checkpoint")
        elif history:
            raise ValueError("GR history exists but an ensemble checkpoint is missing")
        samplers.append(sampler)
    for step in sampling_check_steps(sampling):
        if history and (step <= history[-1]["step"] or history[-1]["stopping_rule_reached"]):
            continue
        for sampler, path, stream in zip(samplers, paths, streams):
            chain = sampler.chains[0]
            completed = 0 if chain is None else chain.shape[0]
            if completed > step:
                raise ValueError("ensemble checkpoint is ahead of the next common check")
            if completed < step:
                length = step - completed
                sampler.run(
                    min_iterations=length,
                    max_iterations=length,
                    check_every=length,
                    check=False,
                    thin_by=1,
                )
                if sampler.chains[0].shape != (step, int(sampling["walkers"])):
                    raise ValueError("native sampler did not produce the requested ensemble axes")
                _save_ensemble_checkpoint(
                    sampler.chains[0],
                    path,
                    sampler,
                    {**identity, "ensemble_index": stream["ensemble_index"]},
                    stream,
                )
        check = ensemble_gelman_rubin(
            [sampler.chains[0] for sampler in samplers], parameters, sampling=sampling
        )
        history = update_gr_history(history, check, step=step, sampling=sampling)
        _replace_sampler_json(
            history_path, {"identity": identity, "sampling": sampling, "history": history}
        )
        if history[-1]["stopping_rule_reached"]:
            break
    records = []
    for path, stream in zip(paths, streams):
        decoded = read_chain(
            path, identity={**identity, "ensemble_index": stream["ensemble_index"]}
        )
        diagnostics = ensemble_autocorrelation(
            decoded["draws"], decoded["parameters"], sampling=sampling
        )
        records.append(
            {
                "ensemble_index": stream["ensemble_index"],
                "path": str(path.relative_to(summary_path.parent)),
                "random_stream": stream,
                "steps": decoded["draws"].shape[1],
                "retained_draws": diagnostics["retained_steps"] * diagnostics["walkers"],
                "acceptance_fraction": decoded["attrs"].get("acceptance_fraction"),
                "diagnostics": diagnostics,
            }
        )
    reached = bool(history and history[-1]["stopping_rule_reached"])
    summary = {
        "identity": identity,
        "ensembles": records,
        "gr_history": history,
        "sampled_parameters": parameters,
        "complete_ensemble_collection": True,
        "stopping_rule_reached": reached,
        "budget_exhausted": not reached,
        "stop_reason": "stopping_rule_reached" if reached else "budget_exhausted",
        "burnin_fraction": sampling["burnin_fraction"],
        "inference_thinning": 1,
    }
    _replace_sampler_json(summary_path, summary)
    return summary
