"""Render saved scientific products as local Markdown, HTML, CSV, PNG and PDF.

Reports describe coverage and numerical diagnostics; interpretation is recorded separately in
review.md. Reading never fits a model, extends a chain or changes the run's snapshot.
"""

from __future__ import annotations

import fcntl
import html
import json
import os
from copy import deepcopy
from itertools import combinations
from pathlib import Path
from statistics import NormalDist
from typing import Any, List, Mapping, Sequence

import numpy as np
import yaml

from ..config import same_identity
from ..config.density import matched_selection_record
from ..figures.data import write_csv as _write_csv


class ReviewPackageIncomplete(ValueError):
    pass


def _condition(record):
    return record["condition"]


def _figure():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save(fig, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Preserve an active constrained layout, including its colorbar geometry.
        if not fig.get_constrained_layout():
            fig.tight_layout()
        paths = {}
        for extension in ("png", "pdf"):
            path = destination.with_suffix("." + extension)
            partial = path.with_name(path.stem + ".partial." + extension)
            fig.savefig(partial, dpi=150, bbox_inches="tight")
            os.replace(partial, path)
            paths[extension] = str(path)
        return paths
    finally:
        _figure().close(fig)


def route_key(record):
    return "|".join(
        str(record[key]) for key in ("measurement_bin", "condition", "reconstruction_state")
    )


def _flatten_values(value, prefix=""):
    if isinstance(value, Mapping):
        return [
            row
            for key, item in value.items()
            for row in _flatten_values(item, prefix + "." + str(key) if prefix else str(key))
        ]
    return [{"parameter": prefix, "value": value}]


def uncertainty_gain(pre_fom, post_fom):
    """The post/pre FoM ratio; a one-dimensional FoM is an uncertainty."""
    values = np.asarray([pre_fom, post_fom], dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("gain requires positive finite pre and post FoMs")
    return float(post_fom / pre_fom)


def paired_realizations(pre, post, *, expected_ids=None):
    """Pair by explicit realization ID and name missing counterparts without changing N."""

    def index(rows):
        result = {}
        for row in rows:
            identifier = row["realization_id"]
            if type(identifier) is not int or identifier in result:
                raise ValueError("paired fit records need unique integer realization IDs")
            result[identifier] = row
        return result

    left, right = index(pre), index(post)
    return [
        {
            "realization_id": identifier,
            "pre": left.get(identifier),
            "post": right.get(identifier),
            "missing_pair": (
                None
                if identifier in left and identifier in right
                else (
                    "pre and post missing"
                    if identifier not in left and identifier not in right
                    else "pre missing" if identifier not in left else "post missing"
                )
            ),
        }
        for identifier in sorted(set(left) | set(right) if expected_ids is None else expected_ids)
    ]


def sampling_products(run, route):
    """Locate published sampling or the latest identified interrupted attempt, read-only."""
    import yaml

    name = "--".join(route[key] for key in ("measurement_bin", "condition", "reconstruction_state"))
    identity = {
        key: run["metadata"][key] for key in ("run_name", "code_commit", "analysis_config_sha256")
    }
    expected = run["config"]["fitting"]["sampling"]["ensembles"]
    directory = run["root"] / "sample" / "units" / name
    if directory.exists():
        summary = json.loads((directory / "sampling.json").read_text())
        if not same_identity(summary["identity"], {**identity, **route}):
            raise ValueError("sampling summary belongs to another route")
        if [item["ensemble_index"] for item in summary["ensembles"]] != list(
            range(expected)
        ) or summary["complete_ensemble_collection"] is not True:
            raise ValueError("published sampling lacks the complete identified ensemble collection")
        return directory, summary
    candidates = sorted(
        (run["root"] / "sample" / "attempts").glob(name + ".*"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    if not candidates:
        raise ValueError("no sampling unit or interrupted attempt exists")
    directory = candidates[0]
    attempt = yaml.safe_load((directory / "attempt.yaml").read_text())
    if attempt["analysis_config_sha256"] != identity["analysis_config_sha256"] or any(
        attempt["unit"].get(key) != value for key, value in route.items()
    ):
        raise ValueError("interrupted sampler attempt belongs to another snapshot or route")
    history = []
    if (directory / "gr_history.json").exists():
        record = json.loads((directory / "gr_history.json").read_text())
        if not same_identity(record["identity"], {**identity, **route}):
            raise ValueError("interrupted GR history belongs to another route")
        history = record["history"]
    ensembles = [
        {"ensemble_index": index, "path": f"ensemble-{index}.npy"}
        for index in range(expected)
        if (directory / f"ensemble-{index}.npy").exists()
    ]
    return directory, {
        "identity": {**identity, **route},
        "ensembles": ensembles,
        "gr_history": history,
        "complete_ensemble_collection": False,
        "stop_reason": "interrupted",
    }


def collect_route(run, route):
    """Read current native profiles/chains, never cached posterior tables or later repository YAML."""
    from ..covariance import read_route_covariance, route_vectors, likelihood_covariance_run
    from ..fitting import read_profiles
    from ..sampling import read_chain, ensemble_autocorrelation
    from ..clustering import read_measurements

    config = run["config"]
    name = "--".join(route[key] for key in ("measurement_bin", "condition", "reconstruction_state"))
    identity = {
        key: run["metadata"][key] for key in ("run_name", "code_commit", "analysis_config_sha256")
    }
    record = {
        **route,
        "coverage": {},
        "individual_fits": [],
        "posterior_draws": {},
        "ensembles": [],
    }
    rows = {
        key: []
        for key in (
            "profiles",
            "posterior",
            "sampler",
            "gr",
            "covariance",
            "parameters",
            "reconstruction",
            "density",
        )
    }
    try:
        saved = read_route_covariance(likelihood_covariance_run(run, route), route)
        arrays = saved["arrays"]
        record.update(
            covariance=arrays["covariance"].tolist(),
            mean_vector=arrays["vectors"].mean(axis=0).tolist(),
            realization_vectors=arrays["vectors"].tolist(),
            separations_mpc_h=arrays["separations"].mean(axis=0).tolist(),
            observed_multipoles=arrays["multipoles"].tolist(),
            covariance_realization_count=len(arrays["realization_ids"]),
        )
        record["covariance_arrays"] = arrays
        record["coverage"]["covariance"] = "complete"
        rows["covariance"].extend(
            {"route": route_key(route), **row} for row in _flatten_values(saved["metadata"])
        )
    except (ImportError, OSError, ValueError, KeyError, TypeError, np.linalg.LinAlgError) as exc:
        record["coverage"]["covariance"] = str(exc)
    measurements = []
    try:
        measurements = read_measurements(run.get("source_run", run), route)
        vectors = route_vectors(run, route, measurements)
        record.update(
            mean_vector=vectors["vectors"].mean(axis=0).tolist(),
            realization_vectors=vectors["vectors"].tolist(),
            separations_mpc_h=vectors["separations"].mean(axis=0).tolist(),
            observed_multipoles=vectors["multipoles"].tolist(),
        )
        record["coverage"]["measurements"] = "complete"
    except (ImportError, OSError, ValueError, KeyError, TypeError) as exc:
        record["coverage"]["measurements"] = str(exc)
    try:
        if (
            not measurements
            and route["measurement_bin"]
            in config["measurement"]["power_spectrum_secondary"]["diagnostic_measurement_bins"]
        ):
            measurements = read_measurements(run.get("source_run", run), route, statistic="pk")
        powers = []
        for measurement in measurements:
            if measurement["identity"]["cap"] == "NGC+SGC" and "power" in measurement:
                with np.load(
                    measurement["directory"] / measurement["power"], allow_pickle=False
                ) as archive:
                    power = {key: archive[key].copy() for key in archive.files}
                ells = config["measurement"]["power_spectrum_secondary"]["multipoles"]
                if (
                    set(power) != {"k", "power", "ells"}
                    or power["k"].ndim != 1
                    or not np.array_equal(power["ells"], ells)
                    or power["power"].shape != (len(ells), len(power["k"]))
                    or not np.all(np.isfinite(power["k"]))
                    or np.any(np.diff(power["k"]) <= 0)
                    or not np.all(np.isfinite(power["power"]))
                ):
                    raise ValueError("P(k) diagnostic has non-finite values or incompatible axes")
                powers.append(power)
        if powers:
            required_ids = config["analysis"]["workloads"][run["metadata"]["workload"]]["mock_ids"]
            if len(powers) != len(required_ids) or any(
                not np.array_equal(value["ells"], powers[0]["ells"])
                or value["power"].shape != powers[0]["power"].shape
                for value in powers
            ):
                raise ValueError("P(k) diagnostic lacks matching complete-ensemble axes")
            record["power"] = {
                "k": np.mean([value["k"] for value in powers], axis=0),
                "power": np.mean([value["power"] for value in powers], axis=0),
                "ells": powers[0]["ells"],
            }
            record["coverage"]["power"] = "complete"
        else:
            record["coverage"]["power"] = (
                "not produced"
                if route["measurement_bin"]
                in config["measurement"]["power_spectrum_secondary"]["diagnostic_measurement_bins"]
                else "not configured for this bin"
            )
    except (ImportError, OSError, ValueError, KeyError, TypeError) as exc:
        record["coverage"]["power"] = str(exc)
    identifiers = config["analysis"]["workloads"][run["metadata"]["workload"]]["mock_ids"]
    for identifier in [*identifiers, None]:
        suffix = "mean" if identifier is None else f"mock-{identifier}"
        directory = run["root"] / "fit" / "units" / (name + "--" + suffix)
        try:
            decoded = read_profiles(
                directory / "profiles.npy",
                identity={**identity, **route, "realization_id": identifier},
            )
            metadata = json.loads((directory / "fit.json").read_text())
            if not same_identity(
                metadata.get("identity"), {**identity, **route, "realization_id": identifier}
            ):
                raise ValueError("fit diagnostics belong to another profile or run")
            fitted = {
                "realization_id": identifier,
                "parameters": decoded["parameters"],
                "chi2": decoded["chi2"],
                "chi2_reason": decoded["chi2_reason"],
                "detection": metadata["detection"],
                "degrees_of_freedom": metadata["degrees_of_freedom"],
            }
            for parameter in decoded["parameters"]:
                rows["profiles"].append(
                    {
                        "route": route_key(route),
                        "realization_id": suffix,
                        "fit_scope": "mean" if identifier is None else "individual",
                        **parameter,
                        "chi2": decoded["chi2"],
                        "chi2_reason": decoded["chi2_reason"],
                        "degrees_of_freedom": metadata["degrees_of_freedom"],
                        "detection": metadata["detection"],
                    }
                )
            if identifier is None:
                record["mean_fit"] = fitted
                with np.load(directory / "fit_curve.npz", allow_pickle=False) as archive:
                    curve = {key: archive[key].copy() for key in archive.files}
                surface = config["fitting"]["fit_surfaces"][route["tracer"]]
                width = config["fitting"]["input_bin_width_mpc_h"]
                low, high = config["fitting"]["fit_edge_limits_mpc_h"]
                edges = np.arange(low, high + 0.5 * width, width)
                dimension = (len(edges) - 1) * len(surface["observed_multipoles"])
                if (
                    not np.array_equal(curve["fit_edges"], edges)
                    or not np.array_equal(curve["multipoles"], surface["observed_multipoles"])
                    or any(
                        curve[key].shape != (dimension,) or not np.all(np.isfinite(curve[key]))
                        for key in ("data", "model", "residual", "standard_deviation")
                    )
                    or np.any(curve["standard_deviation"] <= 0)
                    or not np.allclose(
                        curve["residual"],
                        curve["data"] - curve["model"],
                        rtol=1.0e-12,
                        atol=1.0e-14,
                    )
                ):
                    raise ValueError(
                        "mean fit curve has incompatible axes, residuals or required values"
                    )
                if "mean_vector" in record and not np.allclose(
                    curve["data"], record["mean_vector"], rtol=1.0e-12, atol=1.0e-14
                ):
                    raise ValueError("mean fit curve data differ from the saved ensemble mean")
                record["model_vector"] = curve["model"].tolist()
                prior = metadata["damping_prior"]
                record.update(
                    damping_centres_mpc_h=prior["centers_mpc_h"],
                    damping_widths_mpc_h=prior["widths_mpc_h"],
                    damping_bounds_mpc_h=prior["bounds_mpc_h"],
                )
                rows["parameters"].extend(
                    {"route": route_key(route), **row} for row in _flatten_values(metadata)
                )
            else:
                record["individual_fits"].append(fitted)
            record["coverage"]["fit:" + suffix] = "complete"
        except (ImportError, OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
            record["coverage"]["fit:" + suffix] = str(exc)
    record["chi2_values"] = [
        item["chi2"] for item in record["individual_fits"] if item["chi2"] is not None
    ]
    if record["individual_fits"]:
        record["degrees_of_freedom"] = record["individual_fits"][0]["degrees_of_freedom"]
    # Reconstruction is a parent product shared by pre/post routes. Read its saved shell values.
    record["reconstruction"] = []
    source = run.get("source_run", run)
    source_identity = {
        key: source["metadata"][key]
        for key in ("run_name", "code_commit", "analysis_config_sha256")
    }
    reconstruction_stage = (
        "reconstruct" if source["metadata"].get("interface_version") == 2 else "measure"
    )
    for identifier in identifiers:
        directory = (
            source["root"]
            / reconstruction_stage
            / "units"
            / f"{route['tracer']}--{route['condition']}--mock-{identifier}"
        )
        try:
            result = yaml.safe_load((directory / "result.yaml").read_text())
            expected = {
                "tracer": route["tracer"],
                "condition": route["condition"],
                "realization_id": identifier,
                "unit_id": directory.name,
            }
            if (
                result.get("stage") != reconstruction_stage
                or not same_identity({key: result["unit"].get(key) for key in expected}, expected)
                or any(result.get(key) != value for key, value in source_identity.items())
            ):
                raise ValueError("reconstruction diagnostics belong to another parent or run")
            filename = (
                "reconstruction.json"
                if reconstruction_stage == "reconstruct"
                else "measurements.json"
            )
            parent = json.loads((directory / filename).read_text())
            if set(parent["diagnostics"]) != {"NGC", "SGC"}:
                raise ValueError("reconstruction diagnostics do not cover both caps")
            for cap, diagnostic in parent["diagnostics"].items():
                reconstruction = diagnostic["reconstruction"]
                per_bin = reconstruction.get("scope") == "separate_measurement_bins"
                transfer = "transfer_run" in run["config"]["measurement"]["reconstruction"]
                if per_bin:
                    field = reconstruction["bins"][route["measurement_bin"]]
                    shell_record = field["source_diagnostics"]["shell_diagnostics"]
                else:
                    shell_record = reconstruction["shell_diagnostics"]
                for shell in shell_record["shells"]:
                    row = {
                        "route": route_key(route),
                        "realization_id": identifier,
                        "cap": cap,
                        **shell,
                    }
                    if transfer:
                        row["displacement_source_class"] = field["source_class"]
                    rows["reconstruction"].append(row)
                    record["reconstruction"].append(row)
                density = (
                    diagnostic["exclusive_densities"][route["measurement_bin"]]
                    if transfer
                    else matched_selection_record(
                        diagnostic["selection"]["selection"],
                        source["config"],
                        route["tracer"],
                        route["condition"],
                        cap,
                    )["bins"][route["measurement_bin"]]
                )
                rows["density"].append(
                    {"route": route_key(route), "realization_id": identifier, "cap": cap, **density}
                )
                record.setdefault("density", []).append(
                    {"realization_id": identifier, "cap": cap, **density}
                )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            record["coverage"][f"reconstruction:{identifier}"] = str(exc)
    try:
        directory, summary = sampling_products(run, route)
        record["gr_history"] = summary["gr_history"]
        rows["gr"].extend({"route": route_key(route), **row} for row in summary["gr_history"])
        for ensemble in summary["ensembles"]:
            try:
                path = (directory / ensemble["path"]).resolve()
                if not path.is_relative_to(directory.resolve()):
                    raise ValueError("native chain path leaves its sampling unit")
                decoded = read_chain(
                    path,
                    identity={**identity, **route, "ensemble_index": ensemble["ensemble_index"]},
                )
                diagnostics = ensemble_autocorrelation(
                    decoded["draws"], decoded["parameters"], sampling=config["fitting"]["sampling"]
                )
                ensemble = {
                    **ensemble,
                    "steps": decoded["draws"].shape[1],
                    "retained_draws": diagnostics["retained_steps"] * diagnostics["walkers"],
                    "acceptance_fraction": decoded["attrs"].get("acceptance_fraction"),
                    "random_stream": decoded["attrs"].get("random_stream"),
                }
                record["ensembles"].append(
                    {**ensemble, "decoded": decoded, "diagnostics": diagnostics}
                )
                for parameter, diagnostic in diagnostics["parameters"].items():
                    rows["sampler"].append(
                        {
                            "route": route_key(route),
                            "ensemble_index": ensemble["ensemble_index"],
                            "parameter": parameter,
                            **diagnostic,
                            "steps": ensemble["steps"],
                            "retained_draws": ensemble["retained_draws"],
                            "acceptance_fraction": ensemble["acceptance_fraction"],
                            "stop_reason": summary["stop_reason"],
                            "random_stream": ensemble["random_stream"],
                        }
                    )
                for parameter, reason in decoded.get("unavailable_parameters", {}).items():
                    rows["sampler"].append(
                        {
                            "route": route_key(route),
                            "ensemble_index": ensemble["ensemble_index"],
                            "parameter": parameter,
                            "unavailable_reason": reason,
                        }
                    )
            except (ImportError, OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
                record["coverage"][f"ensemble:{ensemble['ensemble_index']}"] = str(exc)
        expected_count = config["fitting"]["sampling"]["ensembles"]
        if (
            not summary["complete_ensemble_collection"]
            or len(record["ensembles"]) != expected_count
        ):
            raise ValueError(
                f"partial sampling: {len(record['ensembles'])}/{expected_count} readable native ensembles; no aggregate posterior"
            )
        if (
            len(
                {
                    (tuple(item["decoded"]["parameters"]), item["decoded"]["draws"].shape[:2])
                    for item in record["ensembles"]
                }
            )
            != 1
        ):
            raise ValueError(
                "native ensembles have different reported parameters or walker/iteration axes"
            )
        pooled = []
        for ensemble in record["ensembles"]:
            values = ensemble["decoded"]["draws"]
            burn = int(values.shape[1] * config["fitting"]["sampling"]["burnin_fraction"] + 0.5)
            pooled.append(values[:, burn:, :].reshape(-1, values.shape[-1]))
        pooled = np.concatenate(pooled, axis=0)
        for index, name_ in enumerate(record["ensembles"][0]["decoded"]["parameters"]):
            values = pooled[:, index]
            record["posterior_draws"][name_] = values
            quantiles = np.quantile(values, [0.16, 0.5, 0.84])
            rows["posterior"].append(
                {
                    "route": route_key(route),
                    "parameter": name_,
                    "mean": float(values.mean()),
                    "standard_deviation": float(values.std(ddof=1)),
                    "median": float(quantiles[1]),
                    "q16": float(quantiles[0]),
                    "q84": float(quantiles[2]),
                    "retained_draws": len(values),
                    "ensemble_contributions": [
                        ensemble["retained_draws"] for ensemble in record["ensembles"]
                    ],
                    "stop_reason": summary["stop_reason"],
                }
            )
        mapping = {
            "sigmaper": "sigma_perp",
            "sigmapar": "sigma_parallel",
            "sigmas": "sigma_streaming",
        }
        record["damping_posterior_mpc_h"] = {
            mapping[key.rsplit(".", 1)[-1]]: value
            for key, value in record["posterior_draws"].items()
            if key.rsplit(".", 1)[-1] in mapping
        }
        record["coverage"]["sampling"] = summary["stop_reason"]
    except (ImportError, OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        record["posterior_draws"] = {}
        record["coverage"]["sampling"] = str(exc)
    return record, rows


def _minos_half_width(parameter):
    if parameter is None or parameter.get("minos_offsets") is None:
        raise ValueError("native Minos interval is unavailable")
    offsets = np.asarray(parameter["minos_offsets"], dtype=float)
    if (
        offsets.shape != (2,)
        or not np.all(np.isfinite(offsets))
        or not offsets[0] <= 0 <= offsets[1]
        or offsets[1] <= offsets[0]
    ):
        raise ValueError("native Minos interval has invalid shape, signs or width")
    return float((offsets[1] - offsets[0]) / 2)


def paired_fit_rows(routes, *, config, workload):
    rows = []
    definitions = config["reporting"]["gain"]
    expected = config["analysis"]["workloads"][workload]["mock_ids"]
    for (name, condition), states in _paired_by_condition(routes):
        surface = config["fitting"]["fit_surfaces"][next(iter(states.values()))["tracer"]]
        parameters = config["fitting"]["ap_interval_parameters"][surface["ap_mode"]]
        pairs = paired_realizations(
            states.get("prereconstruction", {}).get("individual_fits", []),
            states.get("postreconstruction", {}).get("individual_fits", []),
            expected_ids=expected,
        )
        for pair in pairs:
            left = {item["basename"]: item for item in (pair["pre"] or {}).get("parameters", [])}
            right = {item["basename"]: item for item in (pair["post"] or {}).get("parameters", [])}
            for parameter in parameters:
                pre, post = left.get(parameter), right.get(parameter)
                row = {
                    "measurement_bin": name,
                    "condition": condition,
                    "parameter": parameter,
                    "fit_scope": "individual profile",
                    "fom_definition": definitions["profile_uncertainty"],
                    "realization_id": pair["realization_id"],
                    "missing_pair": pair["missing_pair"],
                    "pre_value": pre["value"] if pre else None,
                    "post_value": post["value"] if post else None,
                    "post_minus_pre": post["value"] - pre["value"] if pre and post else None,
                    "pre_minos_bounds": pre["minos_bounds"] if pre else None,
                    "post_minos_bounds": post["minos_bounds"] if post else None,
                    "pre_interval_flags": pre.get("interval_flags") if pre else None,
                    "post_interval_flags": post.get("interval_flags") if post else None,
                    "pre_detection": (pair["pre"] or {}).get("detection"),
                    "post_detection": (pair["post"] or {}).get("detection"),
                    "prior_limits": config["fitting"]["nonlinear_parameters"][parameter]["prior"][
                        "limits"
                    ],
                    "pre_fom": None,
                    "post_fom": None,
                    "gain": None,
                    "gain_reason": "profile FoM uncertainty statistic is not selected",
                }
                if definitions["profile_uncertainty"] == "minos_half_width":
                    reasons = []
                    for label, fitted in (("pre", pre), ("post", post)):
                        try:
                            row[label + "_fom"] = _minos_half_width(fitted)
                        except ValueError as exc:
                            reasons.append(f"{label}: {exc}")
                    if not reasons:
                        row["gain"] = uncertainty_gain(row["pre_fom"], row["post_fom"])
                        row["gain_reason"] = None
                    else:
                        row["gain_reason"] = "; ".join(reasons)
                rows.append(row)
    return rows


def paired_condition_rows(routes, *, config, workload):
    """Compare conditions at fixed bin/state and mock ID, retaining incomplete pairs.

    The conditions change density and redshift precision together. These descriptive
    differences neither isolate a photo-z effect nor establish a paired mean uncertainty.
    """
    groups = {}
    scope = config["analysis"]["workloads"][workload]
    for route in routes:
        key = tuple(route[name] for name in ("tracer", "measurement_bin", "reconstruction_state"))
        group = groups.setdefault(key, {})
        condition = route["condition"]
        if condition not in scope["sigma_z_tiers"] or condition in group:
            raise ValueError("condition comparison requires unique configured routes")
        group[condition] = route
    rows = []
    for (tracer, name, state), group in groups.items():
        surface = config["fitting"]["fit_surfaces"][tracer]
        parameters = config["fitting"]["ap_interval_parameters"][surface["ap_mode"]]
        for left_condition, right_condition in combinations(scope["sigma_z_tiers"], 2):
            left = group.get(left_condition, {})
            right = group.get(right_condition, {})
            pairs = paired_realizations(
                left.get("individual_fits", []),
                right.get("individual_fits", []),
                expected_ids=scope["mock_ids"],
            )
            for pair in pairs:
                for parameter in parameters:
                    row = {
                        "tracer": tracer,
                        "measurement_bin": name,
                        "reconstruction_state": state,
                        "left_condition": left_condition,
                        "right_condition": right_condition,
                        "realization_id": pair["realization_id"],
                        "parameter": parameter,
                        "fit_scope": "paired individual profiles across conditions",
                        "right_minus_left": None,
                        "right_over_left_interval_width": None,
                    }
                    reasons = []
                    for label, fit, source in (
                        ("left", pair["pre"], left),
                        ("right", pair["post"], right),
                    ):
                        fitted = next(
                            (
                                p
                                for p in (fit or {}).get("parameters", [])
                                if p["basename"] == parameter
                            ),
                            None,
                        )
                        row[label + "_value"] = fitted["value"] if fitted else None
                        row[label + "_minos_bounds"] = (
                            fitted.get("minos_bounds") if fitted else None
                        )
                        row[label + "_interval_flags"] = (
                            fitted.get("interval_flags") if fitted else None
                        )
                        row[label + "_density_by_cap"] = {
                            item["cap"]: {
                                key: value
                                for key, value in item.items()
                                if key not in {"realization_id", "cap"}
                            }
                            for item in source.get("density", [])
                            if item["realization_id"] == pair["realization_id"]
                        }
                        try:
                            row[label + "_minos_half_width"] = _minos_half_width(fitted)
                        except ValueError as exc:
                            row[label + "_minos_half_width"] = None
                            reasons.append(f"{label}: {exc}")
                    if row["left_value"] is not None and row["right_value"] is not None:
                        row["right_minus_left"] = row["right_value"] - row["left_value"]
                    if not reasons:
                        row["right_over_left_interval_width"] = (
                            row["right_minos_half_width"] / row["left_minos_half_width"]
                        )
                    row["interval_comparison_reason"] = "; ".join(reasons) or None
                    rows.append(row)
    return rows


def aggregate_gain_rows(routes, *, config):
    rows = []
    definition = config["reporting"]["gain"]["posterior_uncertainty"]
    for (name, condition), states in _paired_by_condition(routes):
        pre = {
            key.rsplit(".", 1)[-1]: value
            for key, value in states.get("prereconstruction", {}).get("posterior_draws", {}).items()
        }
        post = {
            key.rsplit(".", 1)[-1]: value
            for key, value in states.get("postreconstruction", {})
            .get("posterior_draws", {})
            .items()
        }
        for parameter in ("qiso", "qap", "qpar", "qper"):
            if parameter not in pre and parameter not in post:
                continue
            row = {
                "measurement_bin": name,
                "condition": condition,
                "parameter": parameter,
                "fit_scope": "aggregate posterior",
                "fom_definition": definition,
                "pre_fom": None,
                "post_fom": None,
                "gain": None,
                "prior_standard_deviation": None,
                "pre_to_prior_standard_deviation": None,
                "post_to_prior_standard_deviation": None,
                "prior_comparison_reason": "posterior FoM uncertainty statistic is not selected",
                "gain_reason": "posterior FoM uncertainty statistic is not selected",
            }
            if definition == "posterior_standard_deviation":
                prior = config["fitting"]["nonlinear_parameters"].get(parameter, {}).get("prior")
                prior_sd = None
                if prior and prior["dist"] == "uniform":
                    prior_sd = float(np.diff(prior["limits"]).item() / np.sqrt(12))
                    row.update(prior_standard_deviation=prior_sd, prior_comparison_reason=None)
                else:
                    row["prior_comparison_reason"] = (
                        "no independent uniform prior is defined for this derived coordinate"
                    )
                if parameter in pre and parameter in post:
                    row["pre_fom"], row["post_fom"] = [
                        float(values[parameter].std(ddof=1)) for values in (pre, post)
                    ]
                    try:
                        row["gain"] = uncertainty_gain(row["pre_fom"], row["post_fom"])
                        row["gain_reason"] = None
                        if prior_sd is not None:
                            row.update(
                                pre_to_prior_standard_deviation=row["pre_fom"] / prior_sd,
                                post_to_prior_standard_deviation=row["post_fom"] / prior_sd,
                            )
                    except ValueError as exc:
                        row["gain_reason"] = str(exc)
                else:
                    row["gain_reason"] = "complete pre/post posterior pair is unavailable"
            rows.append(row)
    return rows


def profile_summary_rows(routes, *, config):
    """Bias and explicitly defined interval pulls remain descriptive ensemble statistics."""
    rows = []
    for route in routes:
        for fitted in route["individual_fits"]:
            for parameter in fitted["parameters"]:
                if parameter["basename"] not in ("qiso", "qap") or parameter["fixed"]:
                    continue
                row = {
                    "route": route_key(route),
                    "realization_id": fitted["realization_id"],
                    "parameter": parameter["basename"],
                    "truth": 1.0,
                    "bias": parameter["value"] - 1,
                    "pull": None,
                    "pull_reason": "profile uncertainty statistic is not selected",
                }
                if config["reporting"]["gain"]["profile_uncertainty"] == "minos_half_width":
                    try:
                        row.update(
                            pull=row["bias"] / _minos_half_width(parameter),
                            pull_reason=None,
                            pull_definition="bias / native Minos half-width",
                        )
                    except ValueError as exc:
                        row["pull_reason"] = str(exc)
                rows.append(row)
    return rows


def ensemble_recovery_rows(routes, *, config, workload):
    """Report complete-cohort bias and its SEM without dropping flagged intervals."""
    expected = list(config["analysis"]["workloads"][workload]["mock_ids"])
    if len(set(expected)) != len(expected) or not expected:
        raise ValueError("recovery requires unique nonempty configured mock identities")
    rows = []
    for route in routes:
        fits = route["individual_fits"]
        ids = [fit["realization_id"] for fit in fits]
        if len(set(ids)) != len(ids) or set(ids) - set(expected):
            raise ValueError("recovery has duplicate or unexpected mock identities")
        by_id = {fit["realization_id"]: fit for fit in fits}
        surface = config["fitting"]["fit_surfaces"][route["tracer"]]
        for name in config["fitting"]["ap_interval_parameters"][surface["ap_mode"]]:
            parameters, missing = [], []
            for identifier in expected:
                matches = [
                    item
                    for item in by_id.get(identifier, {}).get("parameters", [])
                    if item["basename"] == name and not item["fixed"]
                ]
                if len(matches) > 1:
                    raise ValueError("recovery has duplicate fitted parameters")
                if matches:
                    parameters.append(matches[0])
                else:
                    missing.append(identifier)
            values = np.asarray([item["value"] for item in parameters], dtype=float)
            if not np.isfinite(values).all():
                raise ValueError("recovery has nonfinite fitted values; cannot silently filter")
            flags = [item.get("interval_flags") or {} for item in parameters]
            row = {
                "route": route_key(route),
                "parameter": name,
                "truth": 1.0,
                "required": len(expected),
                "available": len(parameters),
                "missing_mock_ids": missing,
                "invalid_intervals": sum(not flag.get("is_valid", False) for flag in flags),
                "boundary_intervals": sum(
                    flag.get("at_lower_limit", False) or flag.get("at_upper_limit", False)
                    for flag in flags
                ),
                "mean_offset_from_truth": None,
                "mock_scatter_ddof1": None,
                "mean_offset_standard_error": None,
                "interpretation": "Descriptive matched-mock recovery; not a validation verdict.",
            }
            if missing:
                row["interpretation"] = "Incomplete cohort: no ensemble bias is inferred."
            else:
                row["mean_offset_from_truth"] = float(values.mean() - 1)
                if len(values) > 1:
                    scatter = float(values.std(ddof=1))
                    row["mock_scatter_ddof1"] = scatter
                    row["mean_offset_standard_error"] = scatter / np.sqrt(len(values))
            rows.append(row)
    return rows


def interval_coverage_rows(routes, *, config, workload):
    """Descriptive native-Minos containment; independent of product availability.

    The Wilson interval quantifies binomial uncertainty on the measured fraction,
    not the nominal level of the saved Minos interval. It does not correct for
    dependence from calibrating covariance with the same mocks. Diagnostic failure
    counts overlap; missing or flagged fits are never imputed as ordinary intervals.
    """
    settings = config["reporting"].get("interval_coverage")
    if settings is None:
        return []  # Preserve frozen snapshots predating this reporting decision.
    expected = list(config["analysis"]["workloads"][workload]["mock_ids"])
    if (
        not expected
        or any(type(item) is not int for item in expected)
        or len(set(expected)) != len(expected)
    ):
        raise ValueError("coverage requires unique nonempty configured mock identities")
    confidence = settings["uncertainty_confidence_level"]
    z_squared = NormalDist().inv_cdf((1 + confidence) / 2) ** 2
    rows = []
    for route in routes:
        if {item["realization_id"] for item in route["individual_fits"]} - set(expected):
            raise ValueError("coverage has unexpected mock identities")
        pairs = paired_realizations(route["individual_fits"], [], expected_ids=expected)
        surface = config["fitting"]["fit_surfaces"][route["tracer"]]
        for name in config["fitting"]["ap_interval_parameters"][surface["ap_mode"]]:
            finite, valid = [], []
            missing_fits, missing_parameters, unavailable_intervals = [], [], []
            invalid_flags, boundary_contacts = [], []
            for pair in pairs:
                identifier, fit = pair["realization_id"], pair["pre"]
                if fit is None:
                    missing_fits.append(identifier)
                    continue
                matches = [
                    item
                    for item in fit["parameters"]
                    if item["basename"] == name and not item["fixed"]
                ]
                if len(matches) > 1:
                    raise ValueError("coverage has duplicate fitted parameters")
                if not matches:
                    missing_parameters.append(identifier)
                    continue
                parameter = matches[0]
                flags = parameter.get("interval_flags") or {}
                is_valid = all(
                    flags.get(key, False) for key in ("is_valid", "lower_valid", "upper_valid")
                )
                boundary = flags.get("at_lower_limit", False) or flags.get("at_upper_limit", False)
                if not is_valid:
                    invalid_flags.append(identifier)
                if boundary:
                    boundary_contacts.append(identifier)
                bounds = np.asarray(parameter.get("minos_bounds"), dtype=float)
                if bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
                    unavailable_intervals.append(identifier)
                    continue
                record = (identifier, bool(bounds[0] <= 1.0 <= bounds[1]))
                finite.append(record)
                if is_valid and not boundary:
                    valid.append(record)
            for subset, records in (("all_finite", finite), ("valid_non_boundary", valid)):
                count = len(records)
                successes = sum(contains for _, contains in records)
                row = {
                    "route": route_key(route),
                    "parameter": name,
                    "truth": 1.0,
                    "interval_kind": settings["interval_kind"],
                    "subset": subset,
                    "required": len(expected),
                    "available_fits": len(expected) - len(missing_fits),
                    "finite_intervals": len(finite),
                    "complete_finite_cohort": len(finite) == len(expected),
                    "missing_or_unreadable_fits": len(missing_fits),
                    "missing_or_unreadable_fit_ids": missing_fits,
                    "unavailable_fit_reasons": {
                        identifier: route.get("coverage", {}).get(
                            f"fit:mock-{identifier}", "fit unavailable"
                        )
                        for identifier in missing_fits
                    },
                    "missing_parameters": len(missing_parameters),
                    "missing_parameter_ids": missing_parameters,
                    "unavailable_intervals": len(unavailable_intervals),
                    "unavailable_interval_ids": unavailable_intervals,
                    "invalid_intervals": len(invalid_flags),
                    "invalid_interval_ids": invalid_flags,
                    "boundary_intervals": len(boundary_contacts),
                    "boundary_interval_ids": boundary_contacts,
                    "denominator": count,
                    "numerator": successes,
                    "mock_ids": [identifier for identifier, _ in records],
                    "contains_truth_ids": [
                        identifier for identifier, contains in records if contains
                    ],
                    "fraction": None,
                    "uncertainty_method": settings["uncertainty_method"],
                    "uncertainty_confidence_level": confidence,
                    "wilson_lower": None,
                    "wilson_upper": None,
                    "interpretation": (
                        "Descriptive subset coverage; unavailable/flagged intervals remain explicit. "
                        "Wilson uncertainty assumes binomial trials and does not correct for shared "
                        "covariance calibration or establish full-cohort or posterior coverage."
                    ),
                }
                if count:
                    fraction = successes / count
                    denominator = 1 + z_squared / count
                    centre = (fraction + z_squared / (2 * count)) / denominator
                    half_width = (
                        np.sqrt(
                            z_squared * fraction * (1 - fraction) / count
                            + z_squared**2 / (4 * count**2)
                        )
                        / denominator
                    )
                    row.update(
                        fraction=fraction,
                        wilson_lower=(
                            0.0 if successes == 0 else max(0.0, float(centre - half_width))
                        ),
                        wilson_upper=(
                            1.0 if successes == count else min(1.0, float(centre + half_width))
                        ),
                    )
                rows.append(row)
    return rows


def _differing_settings(left, right):
    flat = [
        {row["parameter"]: row["value"] for row in _flatten_values(item)} for item in (left, right)
    ]
    return sorted(
        key
        for key in set(flat[0]) | set(flat[1])
        if key not in flat[0] or key not in flat[1] or not same_identity(flat[0][key], flat[1][key])
    )


def _transfer_measurement_run(item, label):
    """The run holding this side's selected targets, randoms and reconstruction.

    A native transfer run holds its own. A transfer inference run holds only fits; its
    measurements belong to the source run it records, which must be the run loaded with it
    and must have derived the inference snapshot by its named definition.
    """
    from ..config import validate_inference_config
    from ..storage import same_run_identity

    name, workload = item["metadata"]["run_name"], item["metadata"]["workload"]
    if workload != "transfer":
        raise ValueError(
            f"transfer comparison requires transfer runs; {label} run {name} has workload {workload}"
        )
    source, recorded = item.get("source_run"), item["metadata"].get("source_run")
    if source is None:
        if recorded is not None:
            raise ValueError(f"{label} run {name} records a source run that was not loaded")
        return item
    if recorded is None:
        raise ValueError(f"{label} run {name} carries a source run its metadata does not record")
    loaded = {
        "root": str(source["root"]),
        **{
            key: source["metadata"][key]
            for key in ("run_name", "code_commit", "analysis_config_sha256")
        },
    }
    if not same_run_identity(recorded, loaded):
        differing = sorted(
            key for key in set(recorded) | set(loaded) if recorded.get(key) != loaded.get(key)
        )
        raise ValueError(
            f"{label} run {name} records a source run identity differing from its loaded "
            f"measurement run in {differing}: recorded {recorded}, loaded {loaded}"
        )
    validate_inference_config(
        source["config"], item["config"], item["metadata"].get("inference_definition")
    )
    return source


def _validate_transfer_reference(run, reference):
    """Only the displacement source, radius and reviewed centres may differ.

    Both sides may be native transfer runs or transfer inference runs created from them.
    Returns the two measurement runs whose selected targets and randoms the comparison reads.
    """
    measurement = _transfer_measurement_run(run, "comparison")
    reference_measurement = _transfer_measurement_run(reference, "reference")
    definitions = [item["metadata"].get("inference_definition") for item in (run, reference)]
    if definitions[0] != definitions[1]:
        raise ValueError(
            "transfer comparison requires the same inference definition on both sides; "
            f"comparison run has {definitions[0]!r}, reference has {definitions[1]!r}"
        )
    config, baseline = run["config"], reference["config"]
    recon = config["measurement"]["reconstruction"]
    reference_recon = baseline["measurement"]["reconstruction"]
    reference_source = reference_recon["transfer_run"]["source_class"]
    if reference_source != "self":
        raise ValueError(
            "transfer reference must use self-reconstruction; "
            f"{reference['metadata']['run_name']} uses source class {reference_source!r}"
        )
    high, source_class = (
        recon["exploratory_transfer"]["source_class"],
        recon["transfer_run"]["source_class"],
    )
    if source_class != high:
        raise ValueError(
            f"comparison run must use the configured high-precision source {high!r}; "
            f"{run['metadata']['run_name']} uses {source_class!r}"
        )
    for label, left, right in (
        ("fit", run, reference),
        ("measurement", measurement, reference_measurement),
    ):
        for key in ("code_commit", "code_state_sha256"):
            if left["metadata"].get(key) != right["metadata"].get(key):
                raise ValueError(
                    f"transfer comparison requires the same execution source; {label} {key} "
                    f"differs: {left['metadata'].get(key)!r} vs {right['metadata'].get(key)!r}"
                )
    current_scope = config["analysis"]["workloads"]["transfer"]
    reference_scope = baseline["analysis"]["workloads"]["transfer"]
    missing = set(current_scope["sigma_z_tiers"]) - set(reference_scope["sigma_z_tiers"])
    if missing:
        raise ValueError(
            f"self reference does not include every target class; missing {sorted(missing)}"
        )
    comparable = deepcopy(config)
    normalized_recon = comparable["measurement"]["reconstruction"]
    normalized_recon["smoothing_values_mpc_h"] = reference_recon["smoothing_values_mpc_h"]
    for key in ("source_class", "damping_centres"):
        normalized_recon["transfer_run"][key] = reference_recon["transfer_run"][key]
    for key in ("sigma_z_tiers", "reconstruction_states"):
        comparable["analysis"]["workloads"]["transfer"][key] = reference_scope[key]
    if not same_identity(comparable, baseline):
        raise ValueError(
            "transfer comparison changed settings beyond source, radius or centres: "
            + ", ".join(_differing_settings(comparable, baseline))
        )
    return measurement, reference_measurement


def transfer_comparison_rows(run, reference, routes):
    """Pair saved fits only after exact comparison of saved targets and randoms.

    Fits and routes come from the runs given (native transfer runs or the transfer inference
    runs that fit them); targets and randoms come from the measurement runs those resolve to.
    Full parent columns are compared once per target, mock and cap. This includes
    identities, injected errors, observed selection, weights and unshifted positions.
    Missing fits remain explicit; incomplete cohorts never yield a paired mean.
    """
    from ..selection import read_selected_parent

    measurement, reference_measurement = _validate_transfer_reference(run, reference)
    config = run["config"]
    expected = config["analysis"]["workloads"]["transfer"]["mock_ids"]
    if not expected or len(set(expected)) != len(expected):
        raise ValueError("transfer comparison requires a unique nonempty mock cohort")
    settings = {
        "reference_source_class": "self",
        "variant_source_class": config["measurement"]["reconstruction"]["transfer_run"][
            "source_class"
        ],
        "reference_smoothing_mpc_h": reference["config"]["measurement"]["reconstruction"][
            "smoothing_values_mpc_h"
        ][0],
        "variant_smoothing_mpc_h": config["measurement"]["reconstruction"][
            "smoothing_values_mpc_h"
        ][0],
        "reference_measurement_run": str(reference_measurement["root"]),
        "variant_measurement_run": str(measurement["root"]),
    }
    verified, seen, rows, summaries = set(), set(), [], []
    for route in routes:
        if route["reconstruction_state"] != "postreconstruction":
            continue
        key = (route["tracer"], route["measurement_bin"], route["condition"])
        if key in seen:
            raise ValueError("duplicate transfer comparison route")
        seen.add(key)
        for mock in expected:
            unit = {"tracer": key[0], "condition": key[2], "realization_id": mock}
            for cap in ("NGC", "SGC"):
                parent_key = (key[0], key[2], mock, cap)
                if parent_key in verified:
                    continue
                left = read_selected_parent(reference_measurement, unit, cap)
                right = read_selected_parent(measurement, unit, cap)
                for role, a, b in zip(("data", "random"), left[:2], right[:2]):
                    if a.keys() != b.keys() or any(
                        a[column].dtype != b[column].dtype
                        or not np.array_equal(a[column], b[column])
                        for column in a
                    ):
                        raise ValueError(f"transfer target {parent_key} {role} differs from self")
                verified.add(parent_key)
                del left, right
        coordinates = {
            name: route[name]
            for name in ("tracer", "measurement_bin", "condition", "reconstruction_state")
        }
        reference_route, _ = collect_route(reference, coordinates)
        for record in (reference_route, route):
            if {item["realization_id"] for item in record["individual_fits"]} - set(expected):
                raise ValueError("transfer comparison contains unexpected mock IDs")
        pairs = paired_realizations(
            reference_route["individual_fits"], route["individual_fits"], expected_ids=expected
        )
        surface = config["fitting"]["fit_surfaces"][key[0]]
        for parameter in config["fitting"]["ap_interval_parameters"][surface["ap_mode"]]:
            group = []
            for pair in pairs:
                row = {
                    **coordinates,
                    **settings,
                    "parameter": parameter,
                    "realization_id": pair["realization_id"],
                    "reference_run": str(reference["root"]),
                    "variant_run": str(run["root"]),
                    "target_catalogues_equal": True,
                    "variant_minus_self": None,
                    "variant_over_self_interval_width": None,
                }
                reasons = []
                for label, fit in (("self", pair["pre"]), ("variant", pair["post"])):
                    matches = [
                        item
                        for item in (fit or {}).get("parameters", [])
                        if item["basename"] == parameter and not item["fixed"]
                    ]
                    if len(matches) > 1:
                        raise ValueError("transfer comparison has duplicate fitted parameters")
                    fitted = matches[0] if matches else None
                    value = fitted["value"] if fitted else None
                    if value is not None and not np.isfinite(value):
                        raise ValueError("transfer comparison has a nonfinite fitted value")
                    row[label + "_value"] = value
                    row[label + "_minos_bounds"] = fitted.get("minos_bounds") if fitted else None
                    row[label + "_interval_flags"] = (
                        fitted.get("interval_flags") if fitted else None
                    )
                    try:
                        row[label + "_minos_half_width"] = _minos_half_width(fitted)
                    except ValueError as exc:
                        row[label + "_minos_half_width"] = None
                        reasons.append(f"{label}: {exc}")
                if row["self_value"] is not None and row["variant_value"] is not None:
                    row["variant_minus_self"] = row["variant_value"] - row["self_value"]
                if not reasons:
                    row["variant_over_self_interval_width"] = (
                        row["variant_minos_half_width"] / row["self_minos_half_width"]
                    )
                row["interval_comparison_reason"] = "; ".join(reasons) or None
                rows.append(row)
                group.append(row)
            missing = [
                item["realization_id"] for item in group if item["variant_minus_self"] is None
            ]
            summary = {
                **coordinates,
                **settings,
                "parameter": parameter,
                "reference_run": str(reference["root"]),
                "variant_run": str(run["root"]),
                "required": len(expected),
                "available_pairs": len(expected) - len(missing),
                "missing_pair_ids": missing,
                "mean_variant_minus_self": None,
                "paired_scatter_ddof1": None,
                "paired_mean_standard_error": None,
                "interpretation": "Descriptive paired shift; agreement does not establish unbiased recovery.",
            }
            if not missing:
                deltas = np.asarray([item["variant_minus_self"] for item in group])
                summary["mean_variant_minus_self"] = float(deltas.mean())
                if len(deltas) > 1:
                    scatter = float(deltas.std(ddof=1))
                    summary["paired_scatter_ddof1"] = scatter
                    summary["paired_mean_standard_error"] = scatter / np.sqrt(len(deltas))
            summaries.append(summary)
    return rows, summaries


def _plot_reconstruction(routes, destination):
    plt = _figure()
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), squeeze=False)
    # Both states describe the same parent solve, so plot it once per condition.
    route = next(item for item in routes if item.get("reconstruction"))
    for column, cap in enumerate(("NGC", "SGC")):
        records = [item for item in route["reconstruction"] if item["cap"] == cap]
        for identifier in sorted({item["realization_id"] for item in records}):
            shells = sorted(
                [item for item in records if item["realization_id"] == identifier],
                key=lambda item: item["shell"],
            )
            radial = [np.mean(item["radial_range_mpc_h"]) for item in shells]
            axes[0, column].plot(
                radial, [item["retained_supported_cell_fraction"] for item in shells], alpha=0.3
            )
            axes[1, column].plot(
                radial, [item["displacement_rms_mpc_h"] for item in shells], alpha=0.3
            )
        axes[0, column].set_title(cap)
        axes[0, column].set_ylabel("retained supported cell fraction")
        axes[1, column].set_ylabel("data displacement RMS [Mpc/h]")
        axes[1, column].set_xlabel("comoving radius [Mpc/h]")
    return _save(fig, destination)


def _plot_density(routes, destination):
    plt = _figure()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    route = next(item for item in routes if item.get("density"))
    for axis, cap in zip(axes, ("NGC", "SGC")):
        rows = sorted(
            [item for item in route["density"] if item["cap"] == cap],
            key=lambda item: item["realization_id"],
        )
        identifiers = [item["realization_id"] for item in rows]
        series = (
            ("attained_nbar_h3_mpc3", "attained"),
            ("maximum_attainable_nbar_h3_mpc3", "available"),
            ("requested_nbar_h3_mpc3", "requested"),
        )
        if rows and "forecast_h3_mpc3" in rows[0]:
            series = tuple(
                (key, label)
                for key, label in (
                    ("forecast_h3_mpc3", "forecast"),
                    ("attainable_parent_h3_mpc3", "allocated parent"),
                    ("predicted_final_h3_mpc3", "predicted final"),
                    ("measured_final_h3_mpc3", "measured final ensemble mean"),
                )
            )
        for key, label in series:
            axis.plot(identifiers, [item[key] for item in rows], marker=".", label=label)
        axis.set(xlabel="realization ID", ylabel="number density [h³/Mpc³]", title=cap)
        axis.legend()
    return _save(fig, destination)


def _plot_covariance_inputs(route, destination):
    arrays = route["covariance_arrays"]
    plt = _figure()
    fig, axes = plt.subplots(3, 2, figsize=(11, 12))
    for column, cap in enumerate(("NGC", "SGC")):
        weights = arrays["weights_" + cap]
        image = axes[0, column].imshow(weights, origin="lower", aspect="auto", vmin=0, vmax=1)
        fig.colorbar(image, ax=axes[0, column])
        axes[0, column].set_title(f"{cap}: angular RR weight")
        axes[0, column].set_xlabel("wrapped mu bin")
        axes[0, column].set_ylabel("fitted radial bin")
        image = axes[1, column].imshow(
            arrays["W_" + cap], aspect="auto", origin="lower", cmap="coolwarm"
        )
        fig.colorbar(image, ax=axes[1, column])
        axes[1, column].set_xlabel("raw radial-major ell=0,2,4 coordinate")
        axes[1, column].set_ylabel("fitted multipole-major coordinate")
        samples = arrays.get("covariance_subsamples_" + cap, [])
        matrix = arrays.get("covariance_" + cap)
        if len(samples) and matrix is not None:
            diagonal = np.diag(matrix)
            for sample in samples:
                ratio = np.divide(
                    np.diag(sample),
                    diagonal,
                    out=np.full_like(diagonal, np.nan),
                    where=diagonal != 0,
                )
                axes[2, column].plot(ratio, alpha=0.25)
            axes[2, column].axhline(1, color="black", linewidth=0.7)
        else:
            axes[2, column].text(
                0.1,
                0.5,
                "Cap subsample comparison unavailable",
                transform=axes[2, column].transAxes,
            )
        axes[2, column].set_xlabel("fitted vector coordinate")
        axes[2, column].set_ylabel("subsample diagonal / complete cap diagonal")
    return _save(fig, destination)


def _plot_joint_posteriors(routes, destination):
    plt = _figure()
    fig, axes = plt.subplots(1, len(routes), figsize=(5 * len(routes), 4), squeeze=False)
    for axis, route in zip(axes[0], routes):
        draws = {name.rsplit(".", 1)[-1]: value for name, value in route["posterior_draws"].items()}
        if "qiso" in draws and "qap" in draws:
            axis.hist2d(draws["qiso"], draws["qap"], bins=60)
            axis.axvline(1, color="white", linewidth=0.7)
            axis.axhline(1, color="white", linewidth=0.7)
        else:
            axis.text(0.1, 0.5, "Joint AP posterior unavailable", transform=axis.transAxes)
        axis.set(
            xlabel="isotropic dilation", ylabel="AP dilation", title=route["reconstruction_state"]
        )
    return _save(fig, destination)


def damping_posterior_diagnostics(routes: Sequence[Mapping[str, Any]], *, config) -> List[dict]:
    """The four numerical post-fit damping diagnostics, per route and per coordinate.

    All four are descriptive and none is a threshold. The recurring failure this guards against is
    a diagnostic hardening into a rule within one review cycle, so nothing here is compared with a
    limit, coloured by a band, or reported as a pass.

    * **displacement in prior standard deviations** -- how far the posterior median sits from the
      prior centre, in units of the prior width. It is the quantity a reader wants and the one a
      figure cannot be read off precisely.
    * **width ratio** -- the posterior standard deviation over the prior's. Below one the data
      constrain the coordinate further than the prior does; near one the prior is doing the work,
      which is information about the prior rather than a fault in the fit.
    * **bound proximity** -- the distance from the posterior median to the nearest parameter bound,
      again in prior widths. Damping scales are truncated below at zero and uncapped above, so the
      lower bound is the only one a posterior can reach.
    * **fraction of draws at the bound** -- how much of the posterior sits within one per cent of
      the support's lower edge, which is what "contact with a boundary" means for a sampled
      posterior rather than for a single best fit.
    """
    rows: List[dict] = []
    for route in routes:
        centres = route.get("damping_centres_mpc_h") or {}
        widths = route.get("damping_widths_mpc_h") or {}
        posterior = route.get("damping_posterior_mpc_h") or {}
        bounds = route.get("damping_bounds_mpc_h") or {}
        for name in ("sigma_perp", "sigma_parallel", "sigma_streaming"):
            draws = np.asarray(posterior.get(name, ()), dtype=float)
            draws = draws[np.isfinite(draws)]
            if name not in centres or not draws.size:
                continue
            centre = float(centres[name])
            width = float(widths.get(name, float("nan")))
            median = float(np.median(draws))
            spread = float(np.std(draws, ddof=1)) if draws.size > 1 else float("nan")
            lower = float(bounds[name][0])
            if not np.isfinite(width) or width <= 0 or draws.size < 2:
                rows.append(
                    {
                        "route": route_key(route),
                        "coordinate": name,
                        "unavailable_reason": "finite positive prior width and multiple draws are required",
                    }
                )
                continue
            rows.append(
                {
                    "route": route_key(route),
                    "coordinate": name,
                    "prior_centre_mpc_h": centre,
                    "prior_width_mpc_h": width,
                    "posterior_median_mpc_h": median,
                    "posterior_width_mpc_h": spread,
                    "displacement_in_prior_widths": (median - centre) / width if width else None,
                    "posterior_to_prior_width_ratio": spread / width if width else None,
                    "lower_bound_mpc_h": lower,
                    "bound_distance_in_prior_widths": ((median - lower) / width if width else None),
                    "bound_proximity_fraction": config["reporting"]["damping_rail_fraction"],
                    "posterior_fraction_near_bound": float(
                        np.mean(
                            draws
                            <= lower
                            + config["reporting"]["damping_rail_fraction"] * max(abs(centre), 1.0)
                        )
                    ),
                    "decides_nothing": True,
                }
            )
    return rows


def _plot_sampler(route, destination):
    ensembles = route["ensembles"]
    names = sorted({name for item in ensembles for name in item["decoded"]["parameters"]})
    plt = _figure()
    fig, axes = plt.subplots(
        max(len(names), 1) + 1,
        max(len(ensembles), 1),
        figsize=(4 * max(len(ensembles), 1), 2.3 * (max(len(names), 1) + 1)),
        squeeze=False,
    )
    for column, ensemble in enumerate(ensembles):
        decoded = ensemble["decoded"]
        for row, name in enumerate(names):
            axis = axes[row, column]
            if name in decoded["parameters"]:
                values = decoded["draws"][:, :, decoded["parameters"].index(name)]
                # Every walker's time history remains visible. Their mean can hide opposing drifts.
                axis.plot(values.T, linewidth=0.35, alpha=0.35)
                axis.axvline(
                    values.shape[1] - ensemble["diagnostics"]["retained_steps"],
                    color="black",
                    linewidth=0.6,
                    linestyle="--",
                )
                axis.set_ylabel(name)
                if row == 0:
                    axis.set_title(f"Ensemble {ensemble['ensemble_index']}")
            axis.set_xlabel("unthinned iteration")
        history = route.get("gr_history", [])
        valid = [row for row in history if row.get("value") is not None]
        axes[-1, column].plot(
            [row["step"] for row in valid], [row["value"] for row in valid], marker="."
        )
        axes[-1, column].set_ylabel("largest covariance ratio − 1")
        axes[-1, column].set_xlabel("common ensemble iteration")
    return _save(fig, destination)


def _plot_sampler_diagnostics(route, destination):
    """Display ensemble differences and Monte-Carlo diagnostics without another stop rule."""
    ensembles = route["ensembles"]
    names = sorted({name for item in ensembles for name in item["decoded"]["parameters"]})
    plt = _figure()
    fig, axes = plt.subplots(
        max(len(names), 1) + 1, 4, figsize=(15, 2.8 * (max(len(names), 1) + 1)), squeeze=False
    )
    for row, name in enumerate(names):
        for ensemble in ensembles:
            index = ensemble["ensemble_index"]
            decoded, diagnostics = ensemble["decoded"], ensemble["diagnostics"]
            if name not in decoded["parameters"]:
                continue
            retained_steps = diagnostics["retained_steps"]
            values = decoded["draws"][:, -retained_steps:, decoded["parameters"].index(name)]
            color = f"C{index}"
            if not retained_steps:
                continue
            axes[row, 0].hist(
                values.ravel(),
                bins=40,
                density=True,
                histtype="step",
                color=color,
                label=f"ensemble {index}",
            )
            for block_index, block in enumerate(np.array_split(values, 4, axis=1)):
                if not block.size:
                    continue
                low, median, high = np.quantile(block, [0.16, 0.5, 0.84])
                x = block_index + (index - 1.5) * 0.1
                axes[row, 1].errorbar(
                    x,
                    median,
                    yerr=[[median - low], [high - median]],
                    color=color,
                    marker=".",
                    capsize=2,
                )
            parameter = diagnostics["parameters"][name]
            for column, key in ((2, "tau"), (3, "nominal_autocorrelation_based_ess")):
                value = parameter[key]
                if value is None:
                    axes[row, column].text(
                        index,
                        0.5,
                        "unavailable",
                        rotation=90,
                        ha="center",
                        va="center",
                        fontsize=7,
                        transform=axes[row, column].get_xaxis_transform(),
                    )
                else:
                    axes[row, column].plot(index, value, ".", color=color)
                axes[row, column].set_xticks([item["ensemble_index"] for item in ensembles])
                axes[row, column].set_xlabel("ensemble")
        axes[row, 0].set_xlabel(name)
        axes[row, 0].set_ylabel("retained marginal density")
        axes[row, 0].legend(fontsize=7)
        axes[row, 1].set_xlabel("successive quarters of retained iterations")
        axes[row, 1].set_ylabel(name + ": median and 16–84% interval")
        axes[row, 2].set_ylabel("Sokal autocorrelation time [iterations]")
        axes[row, 3].set_ylabel("nominal autocorrelation ESS")
    for ensemble in ensembles:
        fractions = ensemble.get("acceptance_fraction")
        if fractions is not None:
            fractions = np.asarray(fractions, dtype=float)
            if fractions.shape != (ensemble["decoded"]["draws"].shape[0],) or not np.all(
                np.isfinite(fractions)
            ):
                raise ValueError("acceptance fractions have incompatible walker axes")
            axes[-1, 0].plot(
                np.arange(len(fractions)),
                fractions,
                ".",
                label=f"ensemble {ensemble['ensemble_index']}",
            )
    axes[-1, 0].set_xlabel("walker")
    axes[-1, 0].set_ylabel("native acceptance fraction")
    axes[-1, 0].set_ylim(0, 1)
    for axis in axes[-1, 1:]:
        axis.axis("off")
    axes[-1, 1].text(
        0,
        0.8,
        "Display diagnostics only.\nUnavailable reasons are in sampler.csv.\n"
        "Walkers are not independent chains.\nNo diagnostic extends the frozen budget.",
        fontsize=9,
        va="top",
    )
    return _save(fig, destination)


def _plot_power(routes, destination):
    plt = _figure()
    fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
    for route in routes:
        power = route.get("power")
        if power is None:
            continue
        for axis, ell, values in zip(axes, power["ells"], power["power"]):
            axis.plot(power["k"], power["k"] * values, label=route["reconstruction_state"])
            axis.set_ylabel(f"k P{ell}(k)")
            axis.legend(frameon=False)
    axes[-1].set_xlabel("k [h/Mpc]")
    return _save(fig, destination)


def _plot_individual_parameters(routes, destination):
    plt = _figure()
    fig, axes = plt.subplots(
        2, max(len(routes), 1), figsize=(5 * max(len(routes), 1), 6), squeeze=False
    )
    for column, route in enumerate(routes):
        for row, parameter in enumerate(("qiso", "qap")):
            axis = axes[row, column]
            for fit in route["individual_fits"]:
                values = [
                    item
                    for item in fit["parameters"]
                    if item["basename"] == parameter and not item["fixed"]
                ]
                if not values:
                    continue
                item = values[0]
                offsets = item["minos_offsets"]
                error = np.abs(np.asarray(offsets))[:, None] if offsets else None
                axis.errorbar(
                    fit["realization_id"], item["value"], yerr=error, fmt="o", markersize=3
                )
            axis.axhline(1.0, color=".6", linewidth=0.6)
            axis.set_ylabel(parameter + " (individual profile)")
            axis.set_xlabel("paired realization ID")
            axis.set_title(route["reconstruction_state"])
    return _save(fig, destination)


def _input_review(run, destination):
    from ..selection import read_preparation
    from ..run import stage_units

    figures, rows, support_rows, errors = [], [], [], []
    stage = "select" if run["metadata"].get("interface_version") == 2 else "inputs"
    units = stage_units("select", run["config"], run["metadata"]["workload"])
    for unit in units:
        directory = run["root"] / stage / "units" / unit["unit_id"]
        try:
            preparation = read_preparation(run, directory.name)
            rows.extend({"tracer": directory.name, **row} for row in _flatten_values(preparation))
            for condition, caps in preparation["conditions"].items():
                plt = _figure()
                fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                for axis, (cap, group) in zip(axes, caps.items()):
                    radial = group["radial"]
                    edges = np.asarray(radial["z_edges"])
                    z = (edges[:-1] + edges[1:]) / 2
                    if radial.get("method") == "smooth_parent_forward_prediction":
                        counts = np.asarray(group["observed_counts_by_mock"], dtype=float)
                        prediction = np.asarray(radial["predicted_final_counts"])
                        if counts.shape != (len(run["config"]["analysis"]["mock_ids"]), len(z)):
                            raise ValueError(
                                "exclusive observed histograms lack the allocation ensemble"
                            )
                        axis.plot(z, counts.mean(axis=0), label="measured final ensemble mean")
                        axis.plot(z, prediction, label="forward prediction")
                        axis.set(
                            xlabel="observed redshift",
                            ylabel="objects per redshift bin",
                            title=f"{directory.name} {condition} {cap}",
                        )
                        axis.legend(fontsize=7)
                        continue
                    counts = np.asarray(radial["mean_counts"])
                    mass = np.asarray(radial["target_probability_mass"])
                    boundary = group["boundary_counts_by_mock"]
                    expected_ids = run["config"]["analysis"]["mock_ids"]
                    if not same_identity(
                        [item["realization_id"] for item in boundary], expected_ids
                    ) or any(not np.array_equal(item["z_edges"], edges) for item in boundary):
                        raise ValueError(
                            "input boundary diagnostics have incompatible grids or realizations"
                        )
                    for item in boundary:
                        for edge in item.get("edge_support", []):
                            support_rows.append(
                                {
                                    "tracer": directory.name,
                                    "condition": condition,
                                    "cap": cap,
                                    **{
                                        key: value
                                        for key, value in item.items()
                                        if not isinstance(value, (list, dict))
                                    },
                                    **edge,
                                }
                            )
                    for key, label in (
                        ("original_clustering_counts", "original clustering"),
                        ("available_observed_counts", "after smearing and boundary injection"),
                        ("reservoir_observed_counts", "injected reservoir population"),
                    ):
                        values = np.asarray([item[key] for item in boundary], dtype=float)
                        if values.shape != (len(expected_ids), len(z)) or np.any(
                            ~np.isfinite(values)
                        ):
                            raise ValueError(
                                "input boundary count arrays are incomplete or non-finite"
                            )
                        axis.plot(z, values.mean(axis=0), label=label, alpha=0.65)
                    axis.plot(z, counts, label="ensemble mean selected count")
                    axis.plot(z, mass * counts.sum(), label="shared radial target")
                    axis.set_title(f"{directory.name} {condition} {cap}")
                    axis.set_xlabel("observed redshift")
                    axis.legend(fontsize=7)
                figures.append(
                    (
                        f"Radial selection: {directory.name} {condition}",
                        _save(fig, destination / f"radial--{directory.name}--{condition}.png"),
                    )
                )
        except (ImportError, OSError, ValueError, KeyError, TypeError) as exc:
            errors.append({"input_tracer": directory.name, "error": str(exc)})
    return figures, rows, support_rows, errors


def render_review(root, *, reference_root=None):
    """Regenerate a partial or complete static package while preserving the review notes."""
    from ..run import load_run

    run = load_run(root)
    reference = load_run(reference_root) if reference_root is not None else None
    if reference is not None:
        _validate_transfer_reference(run, reference)
    destination = run["root"] / "review"
    destination.mkdir(exist_ok=True)
    with (destination / ".render.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _render_review(run, reference=reference)


def _render_review(run, *, reference=None):
    from ..run import stage_units

    destination = run["root"] / "review"
    notes = destination / "review.md"
    try:
        with notes.open("x") as stream:
            stream.write(
                "# Review notes\n\nRecord observations, decisions, rationale and links here.\n"
            )
    except FileExistsError:
        pass
    figures, inputs, boundary_support, errors = _input_review(
        run.get("source_run", run), destination
    )
    tables = {
        key: []
        for key in (
            "profiles",
            "posterior",
            "sampler",
            "gr",
            "covariance",
            "parameters",
            "paired_fits",
            "condition_comparisons",
            "coverage",
            "reconstruction",
            "density",
            "aggregate_gains",
            "profile_recovery",
            "ensemble_recovery",
            "interval_coverage",
            "transfer_pairs",
            "transfer_summary",
            "damping",
        )
    }
    tables["coverage"].extend(
        {"route": error["input_tracer"], "product": "inputs", "coverage": error["error"]}
        for error in errors
    )
    tables["parameters"] = [{"source": "snapshot", **row} for row in _flatten_values(run["config"])]
    tables["boundary_support"] = boundary_support
    tables["parameters"].extend(
        {"source": "run", **row} for row in _flatten_values(run["metadata"])
    )
    tables["parameters"].extend({"source": "preparation", **row} for row in inputs)

    def record_figure(title, plotter, *args, **kwargs):
        try:
            figures.append((title, plotter(*args, **kwargs)))
        except (ImportError, OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
            errors.append({"figure": title, "error": str(exc)})
            tables["coverage"].append({"route": title, "product": "figure", "coverage": str(exc)})

    condition_records = []
    pairs = {}
    for unit in stage_units(
        "sample",
        run["config"],
        run["metadata"]["workload"],
        inference_definition=run["metadata"].get("inference_definition"),
    ):
        route = {
            key: unit[key]
            for key in ("tracer", "measurement_bin", "condition", "reconstruction_state")
        }
        pairs.setdefault((route["measurement_bin"], route["condition"]), []).append(route)
    for (name, condition), pair in pairs.items():
        records = []
        for route in pair:
            record, rows = collect_route(run, route)
            records.append(record)
            # Keep only the small profile/density records across groups, not covariance arrays
            # or posterior draws from every route in the campaign.
            condition_records.append(
                {**route, **{key: record.get(key, []) for key in ("individual_fits", "density")}}
            )
            for key, values in rows.items():
                tables[key].extend(values)
            tables["coverage"].extend(
                {"route": route_key(route), "product": key, "coverage": value}
                for key, value in record["coverage"].items()
            )
            if record["ensembles"]:
                record_figure(
                    "Unthinned walker traces and GR: " + route_key(route),
                    _plot_sampler,
                    record,
                    destination / ("sampler--" + "--".join(route.values()) + ".png"),
                )
                record_figure(
                    "Ensemble distributions, interval stability and sampling diagnostics: "
                    + route_key(route),
                    _plot_sampler_diagnostics,
                    record,
                    destination / ("sampler-diagnostics--" + "--".join(route.values()) + ".png"),
                )
            if "covariance_arrays" in record:
                record_figure(
                    "Cap weights and integration samples: " + route_key(route),
                    _plot_covariance_inputs,
                    record,
                    destination / ("cap-covariance--" + "--".join(route.values()) + ".png"),
                )
        tables["paired_fits"].extend(
            paired_fit_rows(records, config=run["config"], workload=run["metadata"]["workload"])
        )
        tables["aggregate_gains"].extend(aggregate_gain_rows(records, config=run["config"]))
        tables["profile_recovery"].extend(profile_summary_rows(records, config=run["config"]))
        tables["ensemble_recovery"].extend(
            ensemble_recovery_rows(
                records, config=run["config"], workload=run["metadata"]["workload"]
            )
        )
        tables["damping"].extend(damping_posterior_diagnostics(records, config=run["config"]))
        tables["interval_coverage"].extend(
            interval_coverage_rows(
                records, config=run["config"], workload=run["metadata"]["workload"]
            )
        )
        prefix = destination / (name + "--" + condition)
        plotters = [
            (
                "Attained densities",
                _plot_density,
                [item for item in records if item.get("density")],
            ),
            (
                "Reconstruction shells",
                _plot_reconstruction,
                [item for item in records if item.get("reconstruction")],
            ),
            (
                "Data vectors",
                _plot_data_vectors,
                [item for item in records if "mean_vector" in item],
            ),
            (
                "Change in acoustic feature",
                _plot_bao_sharpening,
                [item for item in records if "mean_vector" in item],
            ),
            (
                "Models and residuals",
                _plot_fit_quality,
                [item for item in records if "mean_fit" in item],
            ),
            (
                "Individual profiles",
                _plot_individual_parameters,
                [item for item in records if item["individual_fits"]],
            ),
            (
                "Aggregate posteriors",
                _plot_posteriors,
                [item for item in records if item["posterior_draws"]],
            ),
            (
                "Joint aggregate AP posteriors",
                _plot_joint_posteriors,
                [item for item in records if item["posterior_draws"]],
            ),
            (
                "Damping priors and posteriors",
                _plot_damping,
                [item for item in records if "damping_centres_mpc_h" in item],
            ),
            ("Power spectrum", _plot_power, [item for item in records if "power" in item]),
        ]
        for title, plotter, available in plotters:
            if available:
                record_figure(
                    f"{title}: {name} {condition}",
                    plotter,
                    available,
                    prefix.with_name(prefix.name + "--" + plotter.__name__ + ".png"),
                )
        available = [item for item in records if "covariance" in item]
        for mode in ("diagonal", "correlation", "conditioning"):
            if available:
                record_figure(
                    f"Covariance {mode}: {name} {condition}",
                    _plot_covariance,
                    available,
                    prefix.with_name(prefix.name + "--" + mode + ".png"),
                    mode=mode,
                )
    tables["condition_comparisons"] = paired_condition_rows(
        condition_records, config=run["config"], workload=run["metadata"]["workload"]
    )
    if reference is not None:
        tables["transfer_pairs"], tables["transfer_summary"] = transfer_comparison_rows(
            run, reference, condition_records
        )
    for name, rows in tables.items():
        _write_csv(destination / (name + ".csv"), rows)
    _write_csv(destination / "errors.csv", errors)
    complete_states = {
        "complete",
        "stopping_rule_reached",
        "budget_exhausted",
        "not configured for this bin",
    }
    missing = [row for row in tables["coverage"] if row["coverage"] not in complete_states]
    gain = run["config"]["reporting"]["gain"]
    posterior_fom = {
        None: "unavailable (no uncertainty statistic selected)",
        "posterior_standard_deviation": "posterior standard deviation",
    }[gain["posterior_uncertainty"]]
    profile_fom = {
        None: "unavailable (no uncertainty statistic selected)",
        "minos_half_width": "native Minos half-width",
    }[gain["profile_uncertainty"]]
    introduction = (
        f"Run {run['metadata']['run_name']}. {len(missing)} product entries are missing or unreadable. "
        "Numerical stopping and complete files do not establish a validation verdict. "
        "The mean-vector covariance uses C/N conditional on the shared radial template; its uncertainty and cross terms are not propagated. "
        "Nominal autocorrelation ESS is a manual diagnostic, not bulk or tail ESS. "
        "Gain is post FoM / pre FoM, with uncertainty as 1D FoM; below one denotes reduced uncertainty. "
        f"Aggregate FoM: {posterior_fom}. Individual profile FoM: {profile_fom}. "
        "Asymmetric profile bounds and validity flags accompany paired gains; realization scatter is a separate statistic."
    )
    markdown = [
        f"# Review: {run['metadata']['run_name']}",
        "",
        introduction,
        "",
        "[Notes](review.md)",
        "",
    ]
    page = [
        "<!doctype html><html><head><meta charset='utf-8'><title>BAO scientific review</title>",
        "<style>body{font:16px sans-serif;max-width:1100px;margin:2em auto;padding:1em}img{max-width:100%}table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:.4em;text-align:left}</style></head><body>",
        "<h1>"
        + html.escape(run["metadata"]["run_name"])
        + "</h1><p>"
        + html.escape(introduction)
        + "</p>",
        "<p><a href='review.md'>Notes</a></p><h2>Tables</h2><ul>",
    ]
    for name in [*tables, "errors"]:
        markdown.append(f"- [{name.replace('_', ' ')}]({name}.csv)")
        page.append(f"<li><a href='{name}.csv'>{html.escape(name.replace('_', ' '))}</a></li>")
    page.append(
        "</ul><h2>Missing or unreadable products</h2><table><tr><th>Route</th><th>Product</th><th>Reason</th></tr>"
    )
    for row in missing:
        page.append(
            "<tr>"
            + "".join(
                "<td>" + html.escape(str(row[key])) + "</td>"
                for key in ("route", "product", "coverage")
            )
            + "</tr>"
        )
    page.append("</table>")
    for title, paths in figures:
        png, pdf = Path(paths["png"]).name, Path(paths["pdf"]).name
        markdown.extend(["", f"## {title}", "", f"![{title}]({png})", "", f"[PDF]({pdf})"])
        page.append(
            f"<h2>{html.escape(title)}</h2><a href='{html.escape(pdf)}'>PDF</a><br><img src='{html.escape(png)}' alt='{html.escape(title)}'>"
        )
    page.append("</body></html>")
    for filename, content in (
        ("report.md", "\n".join(markdown) + "\n"),
        ("report.html", "\n".join(page)),
    ):
        partial = destination / (filename + ".partial")
        partial.write_text(content)
        os.replace(partial, destination / filename)
    return {
        "report": str(destination / "report.html"),
        "missing_product_count": len(missing),
        "figure_count": len(figures),
    }


def _paired_by_condition(routes):
    """Group produced routes by measurement bin and condition, keyed by reconstruction state.

    A ROUTE is keyed by reconstruction state, so one record carries one state's vector and a
    pre/post figure is a statement about two records. Pairing them here is what makes the
    comparison possible at all; reading both states out of a single record could only ever have
    drawn nothing, because no record has ever carried both.
    """
    grouped = {}
    for route in routes:
        key = (str(route["measurement_bin"]), _condition(route))
        grouped.setdefault(key, {})[str(route["reconstruction_state"])] = route
    return sorted(grouped.items())


def _multipole_blocks(route):
    """The route's vector, split into one block per observed multipole, with its separations.

    The fitted vector is multipole-major -- every retained separation of the monopole, then every
    retained separation of the quadrupole -- so the block for multipole index ``l`` is the slice
    ``[l * radial : (l + 1) * radial]``. Slicing the first ``radial`` elements and labelling the
    axis ``xi_0`` shows the monopole and silently discards the quadrupole, which for LRG3 is half
    the measurement.
    """
    separations = np.asarray(route.get("separations_mpc_h") or (), dtype=float)
    multipoles = [int(value) for value in (route.get("observed_multipoles") or [0])]
    radial = separations.size
    if radial == 0:
        return separations, multipoles, {}
    blocks = {}
    for name in ("mean_vector", "model_vector"):
        vector = route.get(name)
        if vector is None:
            continue
        flat = np.asarray(vector, dtype=float)
        if flat.size != radial * len(multipoles):
            # Reported by drawing nothing rather than by a slice that quietly fits: a vector of
            # the wrong length is a route whose geometry does not match its own declaration.
            continue
        blocks[name] = [
            flat[index * radial : (index + 1) * radial] for index in range(len(multipoles))
        ]
    ensemble = route.get("realization_vectors")
    if ensemble is not None:
        stack = np.asarray(ensemble, dtype=float)
        if stack.ndim == 2 and stack.shape[1] == radial * len(multipoles):
            blocks["realization_vectors"] = [
                stack[:, index * radial : (index + 1) * radial] for index in range(len(multipoles))
            ]
    return separations, multipoles, blocks


def _plot_data_vectors(routes, destination):
    """Pre and post data vectors, one panel per multipole, with the realization scatter behind.

    Every observed multipole gets its own row, so the fifty-two-element red vector is shown as the
    two twenty-six-element blocks it is rather than as its first half.
    """
    plt = _figure()
    paired = _paired_by_condition(routes)
    rows = max(
        (len(_multipole_blocks(next(iter(states.values())))[1]) for _, states in paired),
        default=1,
    )
    columns = max(len(paired), 1)
    fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 3.4 * rows), squeeze=False)
    for column, ((measurement_bin, condition), states) in enumerate(paired):
        for state, colour in (("prereconstruction", "0.55"), ("postreconstruction", "C0")):
            route = states.get(state)
            if route is None:
                continue
            separations, multipoles, blocks = _multipole_blocks(route)
            if "mean_vector" not in blocks:
                continue
            for index, multipole in enumerate(multipoles):
                axis = axes[min(index, rows - 1)][column]
                mean = blocks["mean_vector"][index]
                scatter = blocks.get("realization_vectors")
                if scatter is not None and scatter[index].shape[0] > 1:
                    spread = scatter[index].std(axis=0, ddof=1)
                    axis.fill_between(
                        separations,
                        (mean - spread) * separations**2,
                        (mean + spread) * separations**2,
                        color=colour,
                        alpha=0.25,
                        linewidth=0,
                    )
                axis.plot(separations, mean * separations**2, color=colour, label=state)
                axis.set_xlabel(r"$s\ [h^{-1}\mathrm{Mpc}]$")
                axis.set_ylabel(rf"$s^2\xi_{{{multipole}}}$")
                axis.set_title(rf"{measurement_bin}|{condition}  $\ell={multipole}$", fontsize=8)
                axis.legend(frameon=False, fontsize=7)
    return _save(fig, destination)


def _plot_bao_sharpening(routes, destination):
    """What reconstruction did to the acoustic feature, as the difference it made.

    Distinct from the data-vector figure, which shows the two vectors. This shows the monopole
    post minus pre over the fitted window, which is where a sharpened peak appears as structure
    and an unchanged one as a flat line. Drawing the same figure twice under two names -- which is
    what the package did -- tells a reviewer nothing the first one did not.
    """
    plt = _figure()
    paired = _paired_by_condition(routes)
    columns = max(len(paired), 1)
    fig, axes = plt.subplots(1, columns, figsize=(4.5 * columns, 3.6), squeeze=False)
    for column, ((measurement_bin, condition), states) in enumerate(paired):
        axis = axes[0][column]
        blocks = {}
        separations = None
        for state in ("prereconstruction", "postreconstruction"):
            route = states.get(state)
            if route is None:
                continue
            separations, _, split = _multipole_blocks(route)
            if "mean_vector" in split:
                blocks[state] = split["mean_vector"][0]
        axis.set_xlabel(r"$s\ [h^{-1}\mathrm{Mpc}]$")
        axis.set_title(f"{measurement_bin}|{condition}", fontsize=8)
        if len(blocks) == 2 and separations is not None:
            difference = blocks["postreconstruction"] - blocks["prereconstruction"]
            axis.plot(separations, difference * separations**2, color="C3")
            axis.axhline(0.0, color="0.7", linewidth=0.6)
            axis.set_ylabel(r"$s^2\,\Delta\xi_0$  (post $-$ pre)")
        else:
            # Named, not imputed: one of the two states was not produced, so there is no
            # comparison to draw and the panel says which.
            axis.set_ylabel(r"$s^2\,\Delta\xi_0$")
            axis.text(
                0.5,
                0.5,
                "one reconstruction state absent",
                ha="center",
                va="center",
                fontsize=8,
                transform=axis.transAxes,
            )
    return _save(fig, destination)


def route_mean_sigma(route: Mapping[str, Any]) -> np.ndarray:
    """The uncertainty a route-mean residual is divided by: ``sqrt(diag(C_route / N))``.

    A route carries one covariance. That covariance is ``C_route``, the covariance of **one**
    realization's fit vector, and it is what the route's twenty-five per-realization profile fits
    are normalized by. The vector drawn against the model in a review figure is not one
    realization: it is the mean of the route's ``N`` realizations, whose covariance is
    ``C_route / N``.

    Dividing that mean's residual by ``sqrt(diag(C_route))`` therefore understates every pull by
    exactly ``sqrt(N)`` -- a factor of five at the twenty-five realizations a route is -- so a
    five-sigma model failure draws at one sigma, which is the single thing a residual panel
    exists to make visible. A factor of twenty-five in a chi-square is the kind of error that
    looks entirely plausible on a figure, and this is where it would enter.

    ``N`` comes from the route's own record rather than from configuration, because the
    covariance being normalized and the count normalizing it must belong to the same route. A
    record that does not declare it is refused rather than given a divisor this package invented.
    """
    realizations = route.get("covariance_realization_count")
    if type(realizations) is not int or realizations <= 0:
        raise ReviewPackageIncomplete(
            f"{route_key(route)} does not declare how many realizations its mean averages, so "
            f"the covariance of that mean cannot be formed and no residual is drawn against a "
            f"guessed divisor"
        )
    covariance = np.asarray(route["covariance"], dtype=float)
    return np.sqrt(np.diag(covariance) / float(int(realizations)))


def _plot_fit_quality(routes, destination):
    """The fitted model over the data, its residual, and the reduced chi-square beside them.

    A histogram of raw chi-square says a fit was numerically fine; it does not say whether the
    model goes through the points. The three panels here are what a reviewer reads to answer that:
    the model over the mean vector, the residual in units of the covariance diagonal, and
    chi-square per degree of freedom over the realizations.
    """
    plt = _figure()
    columns = max(len(routes), 1)
    fig, axes = plt.subplots(3, columns, figsize=(4.6 * columns, 8.4), squeeze=False)
    for column, route in enumerate(routes):
        separations, multipoles, blocks = _multipole_blocks(route)
        top, middle, bottom = (axes[row][column] for row in range(3))
        top.set_title(route_key(route), fontsize=8)
        if "mean_vector" in blocks:
            for index, multipole in enumerate(multipoles):
                top.plot(
                    separations,
                    blocks["mean_vector"][index] * separations**2,
                    marker="o",
                    markersize=2.5,
                    linestyle="none",
                    color=f"C{index}",
                    label=rf"data $\ell={multipole}$",
                )
                if "model_vector" in blocks:
                    top.plot(
                        separations,
                        blocks["model_vector"][index] * separations**2,
                        color=f"C{index}",
                        linewidth=1.0,
                        label=rf"model $\ell={multipole}$",
                    )
            top.legend(frameon=False, fontsize=6)
        top.set_ylabel(r"$s^2\xi_\ell$")

        covariance = route.get("covariance")
        if (
            "mean_vector" in blocks
            and "model_vector" in blocks
            and covariance is not None
            and route.get("covariance_realization_count")
        ):
            # The mean of N realizations against the mean's own uncertainty, C_route / N. See
            # route_mean_sigma for why the single-realization diagonal is the wrong divisor.
            sigma = route_mean_sigma(route)
            radial = separations.size
            for index, multipole in enumerate(multipoles):
                block = slice(index * radial, (index + 1) * radial)
                residual = blocks["mean_vector"][index] - blocks["model_vector"][index]
                middle.plot(
                    separations,
                    residual / sigma[block],
                    marker="o",
                    markersize=2.5,
                    linestyle="none",
                    color=f"C{index}",
                )
            middle.axhline(0.0, color="0.7", linewidth=0.6)
        middle.set_ylabel(r"(mean $-$ model) / $\sqrt{\mathrm{diag}\,C_{\rm route}/N}$")
        middle.set_xlabel(r"$s\ [h^{-1}\mathrm{Mpc}]$")

        chi2 = np.asarray(route.get("chi2_values") or (), dtype=float)
        chi2 = chi2[np.isfinite(chi2)]
        dof = route.get("degrees_of_freedom")
        if chi2.size and dof:
            bottom.hist(
                chi2 / float(dof), bins=min(12, max(3, chi2.size // 2)), color="C0", alpha=0.8
            )
            bottom.set_xlabel(r"$\chi^2/\mathrm{dof}$")
        elif chi2.size:
            # The degrees of freedom are a property of the fit, not of this package, so a
            # normalization is not invented where the route did not state one.
            bottom.hist(chi2, bins=min(12, max(3, chi2.size // 2)), color="C0", alpha=0.8)
            bottom.set_xlabel(r"$\chi^2$  (degrees of freedom not declared)")
        else:
            bottom.set_xlabel(r"$\chi^2$")
    return _save(fig, destination)


def _plot_posteriors(routes, destination):
    """The aggregate posterior of each fitted parameter, by name.

    A flat unlabelled list of draws on an axis labelled "draw" cannot be read: a reviewer looking
    at a posterior needs to know which parameter it belongs to, and the acoustic dilations are the
    whole result. The draws are therefore consumed as a mapping from parameter name to samples.
    Per-ensemble autocorrelation diagnostics remain in the dedicated sampler figures and tables.
    """
    plt = _figure()
    names = sorted({str(name) for route in routes for name in (route.get("posterior_draws") or {})})
    rows = max(len(names), 1)
    columns = max(len(routes), 1)
    fig, axes = plt.subplots(rows, columns, figsize=(4.0 * columns, 2.8 * rows), squeeze=False)
    for column, route in enumerate(routes):
        draws = route.get("posterior_draws") or {}
        for row, name in enumerate(names):
            axis = axes[row][column]
            values = np.asarray(draws.get(name, ()), dtype=float)
            values = values[np.isfinite(values)]
            if values.size:
                axis.hist(values, bins=40, color=f"C{row % 10}", alpha=0.85, density=True)
            axis.set_xlabel(name)
            axis.set_ylabel("posterior density")
            axis.ticklabel_format(axis="x", style="sci", scilimits=(-3, 3), useOffset=True)
            if row == 0:
                axis.set_title(route_key(route), fontsize=8)
    return _save(fig, destination)


def _plot_covariance(routes, destination, *, mode):
    """Diagonal uncertainties, correlation matrices, or a conditioning summary."""
    plt = _figure()
    fig, axes = plt.subplots(1, max(len(routes), 1), figsize=(4.5 * max(len(routes), 1), 4))
    axes = np.atleast_1d(axes)
    for axis, route in zip(axes, routes):
        covariance = np.asarray(route["covariance"], dtype=float)
        if mode == "diagonal":
            axis.plot(np.sqrt(np.diag(covariance)))
            axis.set_ylabel(r"$\sqrt{\mathrm{diag}\,C}$")
            axis.set_xlabel("vector element")
        elif mode == "correlation":
            sigma = np.sqrt(np.diag(covariance))
            correlation = covariance / np.outer(sigma, sigma)
            image = axis.imshow(correlation, vmin=-1, vmax=1, cmap="RdBu_r")
            fig.colorbar(image, ax=axis, fraction=0.046)
        else:
            eigenvalues = np.linalg.eigvalsh(covariance)
            axis.semilogy(np.sort(eigenvalues)[::-1], marker="o", markersize=3)
            axis.set_ylabel("eigenvalue")
            axis.set_xlabel("index")
            # Reported, never scored: no band, no threshold, no colour rule.
            axis.set_title(
                f"{route_key(route)}\ncondition number "
                f"{float(eigenvalues.max() / eigenvalues.min()):.3g}",
                fontsize=8,
            )
        if mode != "conditioning":
            axis.set_title(route_key(route), fontsize=8)
    return _save(fig, destination)


def _plot_damping(routes, destination):
    """The prior and the posterior of each damping coordinate, overlaid, with the analytic centre."""
    plt = _figure()
    fig, axes = plt.subplots(1, max(len(routes), 1), figsize=(4 * max(len(routes), 1), 3.5))
    axes = np.atleast_1d(axes)
    for axis, route in zip(axes, routes):
        centre = route.get("damping_centres_mpc_h") or {}
        width = route.get("damping_widths_mpc_h") or {}
        posterior = route.get("damping_posterior_mpc_h") or {}
        for index, name in enumerate(("sigma_perp", "sigma_parallel", "sigma_streaming")):
            if name not in centre:
                continue
            loc, scale = float(centre[name]), float(width.get(name, 1.0))
            grid = np.linspace(max(loc - 4 * scale, 0.0), loc + 4 * scale, 200)
            from scipy.special import ndtr

            lower = (route.get("damping_bounds_mpc_h", {}).get(name) or [0.0, None])[0]
            grid = np.linspace(max(loc - 4 * scale, lower), loc + 4 * scale, 200)
            prior = np.exp(-0.5 * ((grid - loc) / scale) ** 2) / (
                scale * np.sqrt(2 * np.pi) * (1 - ndtr((lower - loc) / scale))
            )
            axis.plot(grid, prior + index, color="0.5", linestyle="--")
            draws = np.asarray(posterior.get(name, ()), dtype=float)
            if draws.size:
                counts, edges = np.histogram(draws, bins=30, density=True)
                axis.step(0.5 * (edges[:-1] + edges[1:]), counts + index, color=f"C{index}")
            axis.axvline(loc, color=f"C{index}", linewidth=0.6)
        axis.set_xlabel(r"damping scale $[h^{-1}\mathrm{Mpc}]$")
        axis.set_title(route_key(route), fontsize=8)
    return _save(fig, destination)
