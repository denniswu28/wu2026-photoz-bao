"""Recompute check numbers of the paper (Secs. IV-V, Appendices B and D) from archived run products.

Thirteen read-only checks, each written to paper/figures/source/appendix/:

- alpha-sn-leave-one-out.json: the leave-one-out spread of alpha_SN for the 50 measurements of
  the independent samples, from each route's saved covariance.npz (Appendix B).
- cross-cap.json: the NGC-SGC correlation of the fitted monopole across the 25 realizations and
  its share of the cap-combined variance, from the per-cap pair counts (Appendix B).
- cross-sigma-checks.json: self- and cross-sigma_z sample fits at 15 and 25 Mpc/h, their paired
  differences and the coverage of alpha = 1, from the review tables of the four transfer fit
  runs (Appendix D).
- cross-sigma-alpha-perp.json: the transverse dilation of the same fits, from the posterior draws
  of the fits to the mean and the best fits of individual realizations (Sec. IV C, Appendix D).
- alpha-perp-narrowing.json: the transverse dilation of the independent samples before and after
  reconstruction, from the draws and by linear propagation (Sec. IV B).
- random-floor.json: the share of supported mesh cells (positive unsmoothed random weight) that
  the random-weight floor zeroes in each displacement field, from the review reconstruction
  tables (Secs. IV C and V).
- prior-width-ratios.json: the posterior widths of the independent samples relative to the
  uniform prior, from the review gain table (Sec. IV B).
- template-b1.json: the template bias b1 of the pre-reconstruction fits against the fixed
  reconstruction bias (Sec. V).
- matched-density-posterior.json: the posterior means and standard deviations of the
  matched-density samples, from the retained draws of the density run (Sec. IV C).
- convergence.json: the last Gelman-Rubin value and the effective sample size of the 50 fits to
  the mean, from the review tables of the fit run (Sec. III E).
- interval-coverage.json: how often the profile-likelihood intervals of single realizations
  contain the true value, and their pulls, from the same tables (Sec. IV B).
- threshold-interval.json: the reconstruction gain of every main sample against its shot-noise
  level, with the interval of the paired gain over realizations (Sec. IV C).
- combined-gain-uncertainty.json: the gain in the combined figure of merit with the interval that
  the 25 realizations leave on it (Sec. IV D).

Restore the runs from the Tier 2 archive, to their original root or with BAO_STORAGE_ROOT and a
relocation map, as for the figure re-renders.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml

BINS = ("BGS1", "BGS2", "LRG1", "LRG2", "LRG3")
TRACERS = {
    "BGS1": "BGS_BRIGHT-21.5",
    "BGS2": "BGS_BRIGHT-21.5",
    "LRG1": "LRG",
    "LRG2": "LRG",
    "LRG3": "LRG",
}
CLASSES = ("native", "sigma_0.001", "sigma_0.003", "sigma_0.01", "sigma_0.03")
STATES = ("prereconstruction", "postreconstruction")
TRANSFER_TARGETS = ("sigma_0.01", "sigma_0.03")
TRANSFER_RUNS = {
    ("self", 15): "unified-fit-v3--self--R15",
    ("self", 25): "unified-fit-v3--self--R25",
    ("cross", 15): "unified-fit-v3--sigma_0.003--R15",
    ("cross", 25): "unified-fit-v3--sigma_0.003--R25",
}


def alpha_sn_leave_one_out(routes, *, search_limits, points):
    """Repeat each route's alpha_SN calibration with one realization left out at a time."""
    from pzbao.diagnostics.covariance import leave_one_out_calibration

    rows = []
    for directory in sorted(routes.iterdir()):
        saved = np.load(directory / "covariance.npz")
        published = json.loads((directory / "covariance.json").read_text())["shot_noise"]
        components = {key: saved[key] for key in ("c2", "c3", "c4")}
        result = leave_one_out_calibration(
            components,
            saved["vectors"],
            saved["realization_ids"],
            search_limits=search_limits,
            points=points,
        )
        full = result["full_ensemble"]["shot_noise_rescaling"]
        if not np.isclose(full, published["shot_noise_rescaling"], rtol=0, atol=1e-12):
            raise ValueError(
                f"{directory.name}: recomputed alpha_SN differs from the published one"
            )
        values = np.array([row["shot_noise_rescaling"] for row in result["leave_one_out"]])

        def sigma(alpha):
            matrix = components["c4"] + alpha * components["c3"] + alpha**2 * components["c2"]
            return np.sqrt(np.diag(matrix))

        reference = sigma(full)
        rows.append(
            {
                "route": directory.name,
                "alpha_sn_full": full,
                "alpha_sn_leave_one_out_min": float(values.min()),
                "alpha_sn_leave_one_out_max": float(values.max()),
                "max_relative_shift": float(np.max(np.abs(values - full)) / full),
                "full_on_search_boundary": bool(
                    result["full_ensemble"]["solution_on_search_boundary"]
                ),
                "leave_one_out_on_search_boundary": [
                    row["omitted_realization_id"]
                    for row in result["leave_one_out"]
                    if row["solution_on_search_boundary"]
                ],
                "max_relative_change_sqrt_diag_covariance": float(
                    max(
                        np.max(np.abs(sigma(a) / reference - 1))
                        for a in (values.min(), values.max())
                    )
                ),
            }
        )
    return {
        "method": "pzbao.diagnostics.covariance.leave_one_out_calibration on each saved covariance.npz",
        "search_limits": list(search_limits),
        "search_points": points,
        "routes": rows,
    }


def cross_cap_correlation(run_root, bins=BINS):
    """Correlate the NGC and SGC fitted monopoles across realizations for every route."""
    from pzbao.clustering import read_measurements
    from pzbao.likelihood import xi_fit_data
    from pzbao.run import load_run

    run = load_run(run_root)
    config = run["config"]
    fitting = config["fitting"]
    identifiers = config["analysis"]["workloads"][run["metadata"]["workload"]]["mock_ids"]
    factor = int(fitting["input_bin_width_mpc_h"] / config["measurement"]["xi"]["bin_width_mpc_h"])
    rows = []
    for measurement_bin in bins:
        tracer = TRACERS[measurement_bin]
        poles = fitting["fit_surfaces"][tracer]["observed_multipoles"]
        for condition in CLASSES:
            for state in STATES:
                route = dict(
                    tracer=tracer,
                    measurement_bin=measurement_bin,
                    condition=condition,
                    reconstruction_state=state,
                )
                name = f"{measurement_bin}--{condition}--{state}"
                records = read_measurements(run, route)
                vectors = {}
                for cap in ("NGC", "SGC", "NGC+SGC"):
                    by_id = {
                        r["identity"]["realization_id"]: r
                        for r in records
                        if r["identity"]["cap"] == cap
                    }
                    vectors[cap] = np.asarray(
                        [
                            xi_fit_data(
                                by_id[i]["estimator"],
                                multipoles=poles,
                                fit_edge_limits_mpc_h=fitting["fit_edge_limits_mpc_h"],
                                rebin_factor=factor,
                                expected_bin_width_mpc_h=fitting["input_bin_width_mpc_h"],
                            )[1]
                            for i in identifiers
                        ]
                    )
                directory = Path(run_root) / "covariance" / "routes" / name
                saved = np.load(directory / "covariance.npz")
                if not np.allclose(vectors["NGC+SGC"], saved["vectors"], rtol=0, atol=1e-12):
                    raise ValueError(f"{name}: combined vectors differ from the covariance input")
                nbin = len(saved["fit_edges"]) - 1
                north, south, combined = (vectors[c][:, :nbin] for c in ("NGC", "SGC", "NGC+SGC"))
                # Cap weight of each fitted bin: the mu-average of RR_cap / RR_total.
                w_north = saved["weights_NGC"].mean(axis=1)
                w_south = saved["weights_SGC"].mean(axis=1)
                linear = w_north * north + w_south * south
                residual = np.std(linear - combined, axis=0, ddof=1) / np.std(
                    combined, axis=0, ddof=1
                )
                rho = np.array([np.corrcoef(north[:, j], south[:, j])[0, 1] for j in range(nbin)])
                cov_ns = np.array([np.cov(north[:, j], south[:, j])[0, 1] for j in range(nbin)])
                independent = w_north**2 * north.var(axis=0, ddof=1) + w_south**2 * south.var(
                    axis=0, ddof=1
                )
                added = 2 * w_north * w_south * cov_ns / independent
                scatter = np.cov(saved["vectors"], rowvar=False, ddof=1)
                model_over_scatter = np.diag(saved["covariance"]) / np.diag(scatter)
                shot_noise = json.loads((directory / "covariance.json").read_text())["shot_noise"]
                rows.append(
                    {
                        "route": name,
                        "rho_monopole_per_bin": rho.tolist(),
                        "rho_mean": float(rho.mean()),
                        "rho_min": float(rho.min()),
                        "rho_max": float(rho.max()),
                        "cap_weight_ngc_mean": float(w_north.mean()),
                        "linearized_combination_residual_over_scatter_max": float(residual.max()),
                        "cross_term_added_variance_median": float(np.median(added)),
                        "sigma_overstatement_median": float(np.median(np.sqrt(1 + added)) - 1),
                        "model_over_scatter_variance_median": float(np.median(model_over_scatter)),
                        "model_over_scatter_variance_median_monopole": float(
                            np.median(model_over_scatter[:nbin])
                        ),
                        "alpha_sn": float(shot_noise["shot_noise_rescaling"]),
                    }
                )
    return {
        "run": Path(run_root).name,
        "statistic": "xi_0 on the fitted separation bins",
        "cap_weight": "mean over mu of RR_cap/RR_total in each fitted bin",
        "routes": rows,
    }


def _review_table(root, run, name):
    with open(Path(root) / run / "review" / f"{name}.csv") as handle:
        return list(csv.DictReader(handle))


def _mean_interval(rows, route, parameter):
    (row,) = [
        r
        for r in rows
        if r["route"] == route and r["fit_scope"] == "mean" and r["name"] == parameter
    ]
    bounds = json.loads(row["minos_bounds"]) if row["minos_bounds"] else None
    flags = json.loads(row["interval_flags"]) if row["interval_flags"] else {}
    usable = bool(
        bounds
        and flags.get("is_valid")
        and flags.get("lower_valid")
        and flags.get("upper_valid")
        and not flags.get("at_lower_limit")
        and not flags.get("at_upper_limit")
    )
    return {
        "value": float(row["value"]),
        "minos_bounds": bounds,
        "half_width": None if not bounds else (bounds[1] - bounds[0]) / 2,
        "usable": usable,
        "flags": flags,
    }


def cross_sigma_checks(transfer_root):
    """Collect the mean fits, paired differences and coverage of the four transfer fit runs."""
    result = {
        "source_runs": {f"{kind}-R{radius}": run for (kind, radius), run in TRANSFER_RUNS.items()},
        "mean_fits": [],
        "paired": [],
        "coverage": [],
    }
    for (kind, radius), run in TRANSFER_RUNS.items():
        profiles = _review_table(transfer_root, run, "profiles")
        for measurement_bin in BINS:
            for target in TRANSFER_TARGETS:
                route = f"{measurement_bin}|{target}|postreconstruction"
                for parameter in ("qiso", "qap"):
                    result["mean_fits"].append(
                        {
                            "source": kind,
                            "radius": radius,
                            "bin": measurement_bin,
                            "target": target,
                            "parameter": parameter,
                            **_mean_interval(profiles, route, parameter),
                        }
                    )
        for row in _review_table(transfer_root, run, "interval_coverage"):
            measurement_bin, target, state = row["route"].split("|")
            if (
                state != "postreconstruction"
                or target not in TRANSFER_TARGETS
                or row["interval_kind"] != "native_minos"
            ):
                continue
            result["coverage"].append(
                {
                    "source": kind,
                    "radius": radius,
                    "bin": measurement_bin,
                    "target": target,
                    "parameter": row["parameter"],
                    "subset": row["subset"],
                    "numerator": int(row["numerator"]),
                    "denominator": int(row["denominator"]),
                    "fraction": float(row["fraction"]) if row["fraction"] else None,
                    "wilson68": [
                        float(row["wilson_lower"]) if row["wilson_lower"] else None,
                        float(row["wilson_upper"]) if row["wilson_upper"] else None,
                    ],
                }
            )
        if kind == "cross":
            for row in _review_table(transfer_root, run, "transfer_summary"):
                result["paired"].append(
                    {
                        "radius": radius,
                        "bin": row["measurement_bin"],
                        "target": row["condition"],
                        "parameter": row["parameter"],
                        "pairs": int(row["available_pairs"]),
                        "mean_cross_minus_self": float(row["mean_variant_minus_self"]),
                        "standard_error": float(row["paired_mean_standard_error"]),
                        "scatter": float(row["paired_scatter_ddof1"]),
                    }
                )
    return result


def _read_yaml(path):
    with open(path) as handle:
        return yaml.safe_load(handle)


def _alpha_perp(qiso, qap):
    """Transverse dilation of Eq. (dilations): alpha_perp = alpha_iso * alpha_AP^(-1/3)."""
    return qiso * qap ** (-1.0 / 3.0)


def _posterior_draws(run_root, route, names=("qiso", "qap")):
    """Retained draws of the named parameters of one sampling unit, as load_chain_samples reads them.

    load_run cannot open the transfer runs while their predecessors are not restored, so the
    identity and burn-in come from the run's own run.yaml, analysis.yaml and sampling.json.
    """
    from pzbao.config import same_identity
    from pzbao.sampling import read_chain

    metadata = _read_yaml(run_root / "run.yaml")
    sampling = _read_yaml(run_root / "analysis.yaml")["fitting"]["sampling"]
    directory = (
        run_root
        / "sample"
        / "units"
        / "--".join(route[key] for key in ("measurement_bin", "condition", "reconstruction_state"))
    )
    summary = json.loads((directory / "sampling.json").read_text())
    identity = {
        **route,
        **{key: metadata[key] for key in ("run_name", "code_commit", "analysis_config_sha256")},
    }
    if (
        not same_identity(summary["identity"], identity)
        or summary.get("complete_ensemble_collection") is not True
        or [item["ensemble_index"] for item in summary["ensembles"]]
        != list(range(sampling["ensembles"]))
        or summary["burnin_fraction"] != sampling["burnin_fraction"]
    ):
        raise ValueError(f"{directory}: not a complete identified ensemble collection")
    draws = {name: [] for name in names}
    for record in summary["ensembles"]:
        decoded = read_chain(
            directory / record["path"],
            identity={**identity, "ensemble_index": record["ensemble_index"]},
        )
        columns = decoded["parameters"]
        burn = int(decoded["draws"].shape[1] * sampling["burnin_fraction"] + 0.5)
        retained = decoded["draws"][:, burn:, :]
        iso, ap = (retained[..., columns.index(name)].ravel() for name in ("qiso", "qap"))
        # D_M/r_d is proportional to alpha_perp in every draw; this fixes the AP orientation.
        ratio = retained[..., columns.index("DM_over_rd")].ravel() / _alpha_perp(iso, ap)
        if np.ptp(ratio) > 1e-9 * ratio.mean():
            raise ValueError(f"{directory}: alpha_perp is not proportional to D_M/r_d")
        for name in names:
            draws[name].append(retained[..., columns.index(name)].ravel())
    return tuple(np.concatenate(draws[name]) for name in names)


def _posterior_summary(qiso, qap, limits):
    perp = _alpha_perp(qiso, qap)
    summary = {"draws": int(perp.size)}
    for name, values in (("qiso", qiso), ("qap", qap), ("alpha_perp", perp)):
        lower, median, upper = np.percentile(values, [16, 50, 84])
        summary[name] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "median": float(median),
            "half_width_68": float((upper - lower) / 2),
        }
    summary["corr_alpha_perp_qap"] = float(np.corrcoef(perp, qap)[0, 1])
    summary["corr_qiso_qap"] = float(np.corrcoef(qiso, qap)[0, 1])
    summary["qap_draws_within_0.01_of_bound"] = float(
        np.mean((qap < limits[0] + 0.01) | (qap > limits[1] - 0.01))
    )
    return summary


def _individual_alpha_perp(rows, route, limits):
    """Best-fitting alpha_perp of each realization, and whether its qap lies at a prior bound."""
    values = {}
    for row in rows:
        if row["route"] == route and row["fit_scope"] == "individual":
            if row["name"] in ("qiso", "qap"):
                values.setdefault(row["realization_id"], {})[row["name"]] = float(row["value"])
    return {
        realization: (
            float(_alpha_perp(fit["qiso"], fit["qap"])),
            bool(fit["qap"] <= limits[0] + 1e-3 or fit["qap"] >= limits[1] - 1e-3),
        )
        for realization, fit in values.items()
        if len(fit) == 2
    }


def cross_sigma_alpha_perp(transfer_root, *, resamples=4000, seed=20261026):
    """Transverse dilation of the pre-, self- and cross-sigma_z sample fits of the four runs.

    Posterior moments come from the draws of the fits to the mean (C/25); the realization
    scatter comes from the best fits in each run's review profiles table, paired by realization.
    """
    transfer_root = Path(transfer_root)
    limits = _read_yaml(transfer_root / TRANSFER_RUNS[("self", 25)] / "analysis.yaml")["fitting"][
        "nonlinear_parameters"
    ]["qap"]["prior"]["limits"]
    rng = np.random.default_rng(seed)
    result = {
        "definition": "alpha_perp = qiso * qap^(-1/3), proportional to D_M/r_d in every draw",
        "qap_prior_limits": limits,
        "source_runs": {f"{kind}-R{radius}": run for (kind, radius), run in TRANSFER_RUNS.items()},
        "posteriors": [],
        "ratios": [],
        "individual": [],
    }
    for radius in (25, 15):
        arms = {
            "self": (TRANSFER_RUNS[("self", radius)], "postreconstruction"),
            "cross": (TRANSFER_RUNS[("cross", radius)], "postreconstruction"),
        }
        # Pre-reconstruction targets are measured once, in the self-reconstruction run at 25 Mpc/h.
        if radius == 25:
            arms = {"pre": (TRANSFER_RUNS[("self", 25)], "prereconstruction"), **arms}
        tables = {run: _review_table(transfer_root, run, "profiles") for run, _ in arms.values()}
        for measurement_bin in BINS:
            for target in TRANSFER_TARGETS:
                std, individual = {}, {}
                for arm, (run, state) in arms.items():
                    route = {
                        "tracer": TRACERS[measurement_bin],
                        "measurement_bin": measurement_bin,
                        "condition": target,
                        "reconstruction_state": state,
                    }
                    summary = _posterior_summary(
                        *_posterior_draws(transfer_root / run, route), limits
                    )
                    result["posteriors"].append(
                        {
                            "source": arm,
                            "radius": radius,
                            "bin": measurement_bin,
                            "target": target,
                            **summary,
                        }
                    )
                    std[arm] = summary["alpha_perp"]["std"]
                    individual[arm] = _individual_alpha_perp(
                        tables[run], f"{measurement_bin}|{target}|{state}", limits
                    )
                result["ratios"].append(
                    {
                        "radius": radius,
                        "bin": measurement_bin,
                        "target": target,
                        "std_cross_over_self": std["cross"] / std["self"],
                        **(
                            {
                                "std_self_over_pre": std["self"] / std["pre"],
                                "std_cross_over_pre": std["cross"] / std["pre"],
                            }
                            if "pre" in std
                            else {}
                        ),
                    }
                )
                common = sorted(
                    set.intersection(*(set(fits) for fits in individual.values())),
                    key=lambda realization: int(realization.split("-")[-1]),
                )
                values = {
                    arm: np.array([individual[arm][realization][0] for realization in common])
                    for arm in individual
                }
                difference = values["cross"] - values["self"]
                samples = rng.integers(0, len(common), size=(resamples, len(common)))
                bootstrap = values["cross"][samples].std(axis=1, ddof=1) / values["self"][
                    samples
                ].std(axis=1, ddof=1)
                result["individual"].append(
                    {
                        "radius": radius,
                        "bin": measurement_bin,
                        "target": target,
                        "pairs": len(common),
                        **{f"scatter_{arm}": float(values[arm].std(ddof=1)) for arm in individual},
                        **{
                            f"qap_at_bound_fraction_{arm}": float(
                                np.mean([individual[arm][realization][1] for realization in common])
                            )
                            for arm in individual
                        },
                        "mean_cross_minus_self": float(difference.mean()),
                        "standard_error": float(difference.std(ddof=1) / np.sqrt(len(common))),
                        "scatter_ratio_cross_over_self": float(
                            values["cross"].std(ddof=1) / values["self"].std(ddof=1)
                        ),
                        "scatter_ratio_bootstrap68": [
                            float(value) for value in np.percentile(bootstrap, [16, 84])
                        ],
                    }
                )
                print("alpha_perp", radius, measurement_bin, target, flush=True)
    return result


def _linear_alpha_perp_std(qiso, qap):
    """Width of alpha_perp from the posterior mean and covariance of (qiso, qap), to first order.

    The mean and covariance (ddof=1 over all retained draws) are those of posterior-records.json,
    so this is what a reader propagating the posterior tables would find.
    """
    mean = np.array([qiso.mean(), qap.mean()])
    covariance = np.cov(np.stack([qiso, qap]), ddof=1)
    gradient = np.array([mean[1] ** (-1.0 / 3.0), -mean[0] * mean[1] ** (-4.0 / 3.0) / 3.0])
    return float(np.sqrt(gradient @ covariance @ gradient))


def _prior_limits(run_root, name):
    config = _read_yaml(Path(run_root) / "analysis.yaml")
    return config["fitting"]["nonlinear_parameters"][name]["prior"]["limits"]


def alpha_perp_narrowing(core_run):
    """Transverse dilation of the independent samples before and after reconstruction.

    Each width is given twice: the standard deviation of the retained draws, and the linear
    propagation of their mean and covariance. Narrowing is 1 - width_post / width_pre.
    """
    core_run = Path(core_run)
    limits = _prior_limits(core_run, "qap")
    result = {
        "definition": (
            "alpha_perp = qiso * qap^(-1/3); narrowing = 1 - width_post / width_pre;"
            " gain = width_pre / width_post"
        ),
        "run": core_run.name,
        "qap_prior_limits": limits,
        "posteriors": [],
        "narrowing": [],
    }
    for measurement_bin in BINS:
        for condition in CLASSES:
            widths = {}
            for state in STATES:
                route = {
                    "tracer": TRACERS[measurement_bin],
                    "measurement_bin": measurement_bin,
                    "condition": condition,
                    "reconstruction_state": state,
                }
                qiso, qap = _posterior_draws(core_run, route)
                summary = _posterior_summary(qiso, qap, limits)
                linear = _linear_alpha_perp_std(qiso, qap)
                result["posteriors"].append(
                    {
                        "bin": measurement_bin,
                        "condition": condition,
                        "state": state,
                        **summary,
                        "alpha_perp_linear_std": linear,
                    }
                )
                widths[state] = {"draws": summary["alpha_perp"]["std"], "linear": linear}
            pre, post = widths["prereconstruction"], widths["postreconstruction"]
            result["narrowing"].append(
                {
                    "bin": measurement_bin,
                    "condition": condition,
                    **{f"width_pre_{key}": pre[key] for key in pre},
                    **{f"width_post_{key}": post[key] for key in post},
                    **{f"narrowing_{key}": 1 - post[key] / pre[key] for key in pre},
                    **{f"gain_{key}": pre[key] / post[key] for key in pre},
                }
            )
            print("alpha_perp narrowing", measurement_bin, condition, flush=True)
    return result


def random_floor(core_run, transfer_root):
    """Share of supported mesh cells that the random-weight floor zeroes, per displacement field.

    A field is one (run, redshift bin, class whose galaxies solve it). Routes that reuse a field
    (pre- and post-reconstruction, or both cross-sigma_z targets) are counted once, after checking
    that they carry the same cell counts. The pooled share sums cells over realizations, caps and
    shells; the realization share is the mean over (realization, cap) of each one's own share.
    """
    runs = {Path(core_run).name: Path(core_run)}
    runs.update({run: Path(transfer_root) / run for run in TRANSFER_RUNS.values()})
    result = {"definition": "threshold_zeroed_cell_count / supported_cell_count", "fields": []}
    for name, root in runs.items():
        reconstruction = _read_yaml(root / "analysis.yaml")["measurement"]["reconstruction"]
        (radius,) = reconstruction["smoothing_values_mpc_h"]
        fields = {}
        for row in _review_table(root.parent, name, "reconstruction"):
            measurement_bin, condition, _ = row["route"].split("|")
            source = row.get("displacement_source_class") or condition
            field = fields.setdefault((measurement_bin, source), {"cells": {}, "routes": set()})
            cell = (row["realization_id"], row["cap"], row["shell"])
            counts = (int(row["supported_cell_count"]), int(row["threshold_zeroed_cell_count"]))
            if field["cells"].setdefault(cell, counts) != counts:
                raise ValueError(f"{name}: routes of one field disagree at {cell}")
            field["routes"].add(row["route"])
        for (measurement_bin, source), field in sorted(fields.items()):
            supported = sum(counts[0] for counts in field["cells"].values())
            zeroed = sum(counts[1] for counts in field["cells"].values())
            by_realization = {}
            for (realization, cap, _), counts in field["cells"].items():
                totals = by_realization.setdefault((realization, cap), [0, 0])
                totals[0] += counts[0]
                totals[1] += counts[1]
            shares = [zero / support for support, zero in by_realization.values()]
            result["fields"].append(
                {
                    "run": name,
                    "smoothing_mpc_h": radius,
                    "floor": reconstruction["threshold_randoms"]["value"],
                    "floor_prescription": reconstruction["threshold_randoms"]["prescription"],
                    "bin": measurement_bin,
                    "field_class": source,
                    "routes": sorted(field["routes"]),
                    "realization_caps": len(by_realization),
                    "supported_cell_count": supported,
                    "threshold_zeroed_cell_count": zeroed,
                    "pooled_share": zeroed / supported,
                    "realization_share_mean": float(np.mean(shares)),
                    "realization_share_max": float(np.max(shares)),
                }
            )
    return result


def prior_width_ratios(core_run):
    """Posterior widths of the independent samples relative to the width of the uniform prior."""
    rows = _review_table(Path(core_run).parent, Path(core_run).name, "aggregate_gains")
    if any(row["fom_definition"] != "posterior_standard_deviation" for row in rows):
        raise ValueError(
            "aggregate_gains.csv must give pre_fom and post_fom as standard deviations"
        )
    if len(rows) != len(BINS) * len(CLASSES) * 2:
        raise ValueError(f"expected {len(BINS) * len(CLASSES) * 2} gain rows, found {len(rows)}")
    return {
        "definition": "posterior standard deviation / standard deviation of the uniform prior",
        "run": Path(core_run).name,
        "rows": [
            {
                "bin": row["measurement_bin"],
                "condition": row["condition"],
                "parameter": row["parameter"],
                "prior_std": float(row["prior_standard_deviation"]),
                "pre_std": float(row["pre_fom"]),
                "post_std": float(row["post_fom"]),
                "pre_over_prior": float(row["pre_to_prior_standard_deviation"]),
                "post_over_prior": float(row["post_to_prior_standard_deviation"]),
            }
            for row in rows
        ],
    }


def template_b1(core_run):
    """Template bias b1 of the pre-reconstruction fits against the fixed reconstruction bias.

    b1 is degenerate with the broadband, damping and redshift-error response of the template,
    so it measures the template amplitude, not the bias of the reconstructed tracer.
    """
    core_run = Path(core_run)
    config = _read_yaml(core_run / "analysis.yaml")
    rows = []
    for measurement_bin in BINS:
        tracer = TRACERS[measurement_bin]
        fixed = config["tracers"][tracer]["reconstruction"]["fiducial_bias"]
        for condition in CLASSES:
            route = {
                "tracer": tracer,
                "measurement_bin": measurement_bin,
                "condition": condition,
                "reconstruction_state": "prereconstruction",
            }
            (b1,) = _posterior_draws(core_run, route, names=("b1",))
            lower, median, upper = np.percentile(b1, [16, 50, 84])
            rows.append(
                {
                    "bin": measurement_bin,
                    "condition": condition,
                    "b1_mean": float(b1.mean()),
                    "b1_std": float(b1.std(ddof=1)),
                    "b1_median": float(median),
                    "b1_half_width_68": float((upper - lower) / 2),
                    "reconstruction_bias": fixed,
                    "offset_in_std": float((b1.mean() - fixed) / b1.std(ddof=1)),
                }
            )
            print("b1", measurement_bin, condition, flush=True)
    return {
        "definition": "posterior b1 of the pre-reconstruction fit to the mean, retained draws",
        "run": core_run.name,
        "b1_prior_limits": _prior_limits(core_run, "b1"),
        "rows": rows,
    }


def matched_density_posterior(density_run):
    """Posterior means and standard deviations of the matched-density samples.

    One row per matched route and dilation: the mean and standard deviation of the retained draws
    (ddof=1, as posterior-summary.json gives them for the main samples) and the prior contact of
    the mean-fit profile under the rule applied to the main samples.
    """
    from pzbao.figures.data import interval
    from pzbao.fitting import read_profiles

    density_run = Path(density_run)
    metadata = _read_yaml(density_run / "run.yaml")
    if metadata.get("workload") != "density":
        raise ValueError(f"{density_run}: not a matched-density run")
    workload = _read_yaml(density_run / "analysis.yaml")["analysis"]["workloads"]["density"]
    run = {key: metadata[key] for key in ("run_name", "code_commit", "analysis_config_sha256")}
    rows = []
    for measurement_bin in workload["measurement_bins"]:
        for condition in workload["sigma_z_tiers"]:
            for state in workload["reconstruction_states"]:
                route = {
                    "tracer": TRACERS[measurement_bin],
                    "measurement_bin": measurement_bin,
                    "condition": condition,
                    "reconstruction_state": state,
                }
                unit = "--".join([measurement_bin, condition, state, "mean"])
                profile = read_profiles(
                    density_run / "fit" / "units" / unit / "profiles.npy",
                    identity={**route, "realization_id": None, **run},
                )
                for name, values in zip(("qiso", "qap"), _posterior_draws(density_run, route)):
                    fitted = next(p for p in profile["parameters"] if p["basename"] == name)
                    rows.append(
                        {
                            **route,
                            "parameter": name,
                            "mean": float(values.mean()),
                            "sd": float(values.std(ddof=1)),
                            "draws": int(values.size),
                            "prior_contact": interval(fitted, profile["maximization"])["boundary"],
                        }
                    )
                print("matched posterior", measurement_bin, condition, state, flush=True)
    return {
        "definition": "mean and standard deviation (ddof=1) of the retained posterior draws",
        "run": run,
        "rows": rows,
    }


def _routes():
    for measurement_bin in BINS:
        for condition in CLASSES:
            for state in STATES:
                yield measurement_bin, condition, state


def _by_route(rows, expected):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["route"], []).append(row)
    if set(grouped) != set(expected):
        raise ValueError("review table does not cover the 50 fits of the main samples")
    return grouped


def _optional_float(text):
    return float(text) if text not in ("", None) else None


def convergence(core_run):
    """Sampler convergence of the 50 fits to the mean, from the review tables of the fit run.

    The Gelman-Rubin value is the last one of the saved history. The autocorrelation time of a
    dilation is the sampler's own estimate; where the sampler withheld it because the chain is
    shorter than 50 autocorrelation times, the value it printed with that warning is kept and
    flagged, and the effective sample size is the retained draws divided by it.
    """
    import re

    core_run = Path(core_run)
    expected = ["|".join(route) for route in _routes()]
    history = _by_route(_review_table(core_run.parent, core_run.name, "gr"), expected)
    sampler = _by_route(_review_table(core_run.parent, core_run.name, "sampler"), expected)
    rows = []
    for route in expected:
        last = max(history[route], key=lambda row: int(row["step"]))
        record = {
            "route": route,
            "gelman_rubin_statistic": last["statistic"],
            "gelman_rubin_last": float(last["value"]),
            "steps": int(last["step"]),
            "retained_steps": int(last["retained_steps"]),
            "stopping_rule_reached": last["stopping_rule_reached"] == "True",
        }
        for parameter in ("qiso", "qap"):
            members = [row for row in sampler[route] if row["parameter"] == parameter]
            taus, withheld, sizes = [], 0, []
            for row in members:
                tau = _optional_float(row["tau"])
                if tau is None:
                    printed = re.search(r"tau: \[([0-9.eE+-]+)\]", row["unavailable_reason"])
                    if printed is None:
                        raise ValueError(f"{route}: no autocorrelation time for {parameter}")
                    tau, withheld = float(printed.group(1)), withheld + 1
                taus.append(tau)
                sizes.append(int(row["retained_draws"]) / tau)
            record[parameter] = {
                "ensembles": len(members),
                "ensembles_shorter_than_50_tau": withheld,
                "tau_max": max(taus),
                "effective_sample_size": float(sum(sizes)),
                "effective_sample_size_min_ensemble": float(min(sizes)),
                "retained_draws": sum(int(row["retained_draws"]) for row in members),
            }
        rows.append(record)
    return {
        "definition": (
            "last saved Gelman-Rubin value; effective sample size = retained draws / integrated"
            " autocorrelation time, summed over ensembles"
        ),
        "run": core_run.name,
        "rows": rows,
    }


def interval_coverage(core_run):
    """Coverage and pulls of the single-realization profile-likelihood intervals.

    One row per fit and dilation: how many of the realizations' intervals contain the true
    value, over all finite intervals and over the valid intervals clear of the prior bounds, the
    scatter of the pulls of the latter, and the offset and scatter of the best fits.
    """
    core_run = Path(core_run)
    expected = ["|".join(route) for route in _routes()]
    coverage = _by_route(
        _review_table(core_run.parent, core_run.name, "interval_coverage"), expected
    )
    pulls = _by_route(_review_table(core_run.parent, core_run.name, "profile_recovery"), expected)
    recovery = _by_route(
        _review_table(core_run.parent, core_run.name, "ensemble_recovery"), expected
    )
    rows = []
    for route in expected:
        for parameter in ("qiso", "qap"):
            record, clear = {"route": route, "parameter": parameter}, set()
            for row in coverage[route]:
                if row["parameter"] != parameter or row["interval_kind"] != "native_minos":
                    continue
                record[row["subset"]] = {
                    "contains_truth": int(row["numerator"]),
                    "intervals": int(row["denominator"]),
                    "fraction": _optional_float(row["fraction"]),
                    "wilson68": [
                        _optional_float(row["wilson_lower"]),
                        _optional_float(row["wilson_upper"]),
                    ],
                }
                if row["subset"] == "valid_non_boundary":
                    clear = set(json.loads(row["mock_ids"]))
            values = [
                float(row["pull"])
                for row in pulls[route]
                if row["parameter"] == parameter
                and row["pull"] != ""
                and int(row["realization_id"]) in clear
            ]
            record["pull_definition"] = "best fit minus truth, over the interval half-width"
            record["pulls"] = len(values)
            record["pull_mean"] = float(np.mean(values)) if values else None
            record["pull_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
            (row,) = [row for row in recovery[route] if row["parameter"] == parameter]
            record["best_fit_mean_offset"] = float(row["mean_offset_from_truth"])
            record["best_fit_scatter"] = float(row["mock_scatter_ddof1"])
            record["best_fit_mean_offset_standard_error"] = float(row["mean_offset_standard_error"])
            rows.append(record)
    return {
        "definition": (
            "profile-likelihood intervals of the fits to single realizations; pulls use the"
            " valid intervals clear of the prior bounds"
        ),
        "run": core_run.name,
        "rows": rows,
    }


def _mock_fits(core_run, measurement_bin, condition, state):
    """Best fits and errors of the dilations in the fits to the single realizations of one route."""
    from pzbao.figures.data import interval

    core_run = Path(core_run)
    fits = {}
    for mock in _read_yaml(core_run / "analysis.yaml")["analysis"]["mock_ids"]:
        unit = "--".join([measurement_bin, condition, state, f"mock-{mock}"])
        saved = json.loads((core_run / "fit" / "units" / unit / "fit.json").read_text())
        record = {}
        for name in ("qiso", "qap"):
            fitted = next(p for p in saved["parameters"] if p["basename"] == name)
            flagged = interval(fitted, saved["optimizer"]["maximization"])
            record[name] = flagged["value"]
            record[name + "_err"] = fitted["parabolic_error"]
            record["optimizer_ok"] = flagged["optimizer_ok"]
        fits[mock] = record
    return fits


def threshold_interval(core_run, *, resamples=4000, seed=20260615):
    """Reconstruction gain of every main sample against its shot-noise level, with intervals.

    Three gains of the isotropic dilation per sample: the ratio of the posterior standard
    deviations of the fits to the mean (the estimator of the gain figures), the ratio of the
    profile-likelihood half-widths of the same fits where both intervals are valid and clear of
    the prior bounds, and the median over realizations of the paired ratio of the parabolic
    errors with the bootstrap 68% interval of that median.
    """
    from pzbao.figures.data import _gain_summary

    core_run = Path(core_run)
    config = _read_yaml(core_run / "analysis.yaml")
    gains = {
        (row["measurement_bin"], row["condition"]): row
        for row in _review_table(core_run.parent, core_run.name, "aggregate_gains")
        if row["parameter"] == "qiso"
    }
    profiles = _review_table(core_run.parent, core_run.name, "profiles")
    rows = []
    for measurement_bin in BINS:
        power = config["tracers"][TRACERS[measurement_bin]]["fkp_p0_h3_mpc3"]
        for condition in CLASSES:
            density, intervals, fits = set(), {}, {}
            for state in STATES:
                unit = "--".join([measurement_bin, condition, state, "mean"])
                saved = json.loads((core_run / "fit" / "units" / unit / "fit.json").read_text())
                density.add(saved["attained_density_h3_mpc3"])
                route = "|".join([measurement_bin, condition, state])
                intervals[state] = _mean_interval(profiles, route, "qiso")
                fits[state] = _mock_fits(core_run, measurement_bin, condition, state)
            if len(density) != 1:
                raise ValueError(f"{measurement_bin} {condition}: density differs between states")
            (density,) = density
            pre, post = (intervals[state] for state in STATES)
            paired, railed = [], 0
            for mock, before in fits[STATES[0]].items():
                after = fits[STATES[1]][mock]
                if not (before["optimizer_ok"] and after["optimizer_ok"]):
                    continue
                if any(x <= 0.805 or x >= 1.195 for x in (before["qiso"], after["qiso"])):
                    railed += 1
                errors = (before["qiso_err"], after["qiso_err"])
                if all(e is not None and np.isfinite(e) and e > 0 for e in errors):
                    paired.append(errors[0] / errors[1])
            gain = gains[measurement_bin, condition]
            rows.append(
                {
                    "bin": measurement_bin,
                    "condition": condition,
                    "density_h3_mpc3": density,
                    "fkp_p0_h3_mpc3": power,
                    "nbar_p0": density * power,
                    "gain_posterior_std": float(gain["pre_fom"]) / float(gain["post_fom"]),
                    "gain_profile_half_width": (
                        pre["half_width"] / post["half_width"]
                        if pre["usable"] and post["usable"]
                        else None
                    ),
                    "paired_realizations": (
                        {
                            **_gain_summary(np.asarray(paired), resamples, seed),
                            "n_railed": railed,
                        }
                        if len(paired) > 1
                        else None
                    ),
                }
            )
        print("threshold interval", measurement_bin, flush=True)
    return {
        "definition": (
            "gain = width before / width after reconstruction of the isotropic dilation;"
            " nbar_p0 = attained density times the FKP power of the tracer"
        ),
        "run": core_run.name,
        "paired_resamples": resamples,
        "paired_seed": seed,
        "rows": sorted(rows, key=lambda row: row["nbar_p0"]),
    }


def combined_gain_uncertainty(core_run, *, resamples=4000):
    """Uncertainty of the gain in the combined figure of merit from the 25 realizations.

    The combination of the four redshift precision classes with redshift errors is the one of
    the figures: the posteriors of the fits to the mean, weighted with the correlation between
    classes that the best fits of the realizations measure. That correlation is the only input
    from a finite ensemble. Each resample draws realizations with replacement, the same ones
    before and after reconstruction, re-measures both correlations and recombines; the jackknife
    leaves one realization out at a time. The class posteriors stay fixed, so the interval does
    not include their own uncertainty.
    """
    from pzbao.fitting import combine_tier_posteriors, estimate_correlation

    core_run = Path(core_run)
    classes = [condition for condition in CLASSES if condition != "native"]
    seed = int(_read_yaml(core_run / "analysis.yaml")["fitting"]["sampling"]["seed"])
    rng = np.random.default_rng(seed)

    def gain(means, covariances, estimates, chosen):
        foms = []
        for state in STATES:
            record = combine_tier_posteriors(
                classes, means[state], covariances[state], estimate_correlation(chosen[state])
            )
            foms.append(record["two_dimensional_fom"])
        return 100 * (foms[1] / foms[0] - 1), foms

    rows = []
    for measurement_bin in BINS:
        means, covariances, fits, combination = {}, {}, {}, {}
        for state in STATES:
            draws = []
            for condition in classes:
                route = {
                    "tracer": TRACERS[measurement_bin],
                    "measurement_bin": measurement_bin,
                    "condition": condition,
                    "reconstruction_state": state,
                }
                draws.append(np.stack(_posterior_draws(core_run, route), axis=1))
                fits[state, condition] = _mock_fits(core_run, measurement_bin, condition, state)
            means[state] = np.array([d.mean(axis=0) for d in draws])
            covariances[state] = np.array([np.cov(d, rowvar=False, ddof=1) for d in draws])
        # A realization enters when its optimizer succeeded in every class of both states, so
        # that the same realizations are resampled before and after reconstruction.
        cohort = sorted(set.intersection(*(set(members) for members in fits.values())))
        used = [m for m in cohort if all(members[m]["optimizer_ok"] for members in fits.values())]
        estimates = {
            state: np.array(
                [
                    [[fits[state, c][m][name] for name in ("qiso", "qap")] for c in classes]
                    for m in used
                ]
            )
            for state in STATES
        }
        point, foms = gain(means, covariances, estimates, estimates)
        for state in STATES:
            record = combine_tier_posteriors(
                classes, means[state], covariances[state], estimate_correlation(estimates[state])
            )
            combination[state] = {
                "removed_classes": record["removed_tiers"],
                "stacked_minimum_eigenvalue": record["stacked_minimum_eigenvalue"],
                "cross_term_clipped": record["cross_term_clipped"],
            }
        resampled, skipped = [], 0
        for _ in range(int(resamples)):
            index = rng.integers(0, len(used), len(used))
            try:
                value = gain(
                    means, covariances, estimates, {s: estimates[s][index] for s in STATES}
                )[0]
            except ValueError:
                value = np.nan
            # A resample whose correlation cannot be measured or combined is counted, not used.
            if np.isfinite(value):
                resampled.append(value)
            else:
                skipped += 1
        left_out = np.array(
            [
                gain(
                    means,
                    covariances,
                    estimates,
                    {s: np.delete(estimates[s], i, axis=0) for s in STATES},
                )[0]
                for i in range(len(used))
            ]
        )
        count = len(left_out)
        rows.append(
            {
                "bin": measurement_bin,
                "classes": classes,
                "realizations_used": len(used),
                "realizations_excluded": sorted(set(cohort) - set(used)),
                "fom_pre": foms[0],
                "fom_post": foms[1],
                "gain_percent": point,
                "bootstrap_gain_percent_16_84": np.percentile(resampled, [16, 84]).tolist(),
                "bootstrap_gain_percent_2p5_97p5": np.percentile(resampled, [2.5, 97.5]).tolist(),
                "bootstrap_resamples_used": len(resampled),
                "bootstrap_resamples_skipped": skipped,
                "jackknife_gain_percent_standard_error": float(
                    np.sqrt((count - 1) / count * np.sum((left_out - left_out.mean()) ** 2))
                ),
                "jackknife_gain_percent_min_max": [float(left_out.min()), float(left_out.max())],
                "combination": combination,
            }
        )
        print("combined gain", measurement_bin, flush=True)
    return {
        "definition": (
            "gain_percent = 100 (FoM after / FoM before - 1) of the combined classes;"
            " intervals resample the realizations that measure the correlation between classes,"
            " paired before and after reconstruction, with the class posteriors fixed"
        ),
        "run": core_run.name,
        "resamples": int(resamples),
        "seed": seed,
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--primary-run", type=Path, help="restored per-bin-v2--primary")
    parser.add_argument("--transfer-root", type=Path, help="restored runs/reconstruction-transfer")
    parser.add_argument(
        "--core-run", type=Path, help="restored unified-fit-v3--primary (the fits and draws)"
    )
    parser.add_argument(
        "--density-run", type=Path, help="restored unified-fit-v3--density (the matched samples)"
    )
    parser.add_argument("--paper-dir", type=Path, default=Path("paper"))
    products = {
        "cross-sigma-checks": lambda: cross_sigma_checks(args.transfer_root),
        "alpha-sn-leave-one-out": lambda: alpha_sn_leave_one_out(
            args.primary_run / "covariance" / "routes", search_limits=(0.1, 10.0), points=4096
        ),
        "cross-cap": lambda: cross_cap_correlation(args.primary_run),
        "cross-sigma-alpha-perp": lambda: cross_sigma_alpha_perp(args.transfer_root),
        "alpha-perp-narrowing": lambda: alpha_perp_narrowing(args.core_run),
        "random-floor": lambda: random_floor(args.core_run, args.transfer_root),
        "prior-width-ratios": lambda: prior_width_ratios(args.core_run),
        "template-b1": lambda: template_b1(args.core_run),
        "matched-density-posterior": lambda: matched_density_posterior(args.density_run),
        "convergence": lambda: convergence(args.core_run),
        "interval-coverage": lambda: interval_coverage(args.core_run),
        "threshold-interval": lambda: threshold_interval(args.core_run),
        "combined-gain-uncertainty": lambda: combined_gain_uncertainty(args.core_run),
    }
    # The runs each product reads.
    needs = {
        "cross-sigma-checks": ("transfer_root",),
        "alpha-sn-leave-one-out": ("primary_run",),
        "cross-cap": ("primary_run",),
        "cross-sigma-alpha-perp": ("transfer_root",),
        "alpha-perp-narrowing": ("core_run",),
        "random-floor": ("core_run", "transfer_root"),
        "prior-width-ratios": ("core_run",),
        "template-b1": ("core_run",),
        "matched-density-posterior": ("density_run",),
        "convergence": ("core_run",),
        "interval-coverage": ("core_run",),
        "threshold-interval": ("core_run",),
        "combined-gain-uncertainty": ("core_run",),
    }
    parser.add_argument(
        "--products",
        nargs="+",
        choices=sorted(products),
        help="default: every product whose runs are given",
    )
    args = parser.parse_args()
    absent = {name: [run for run in needs[name] if getattr(args, run) is None] for name in products}
    if args.products is None:
        args.products = [name for name in products if not absent[name]]
    for name in args.products:
        if absent[name]:
            flags = ", ".join("--" + run.replace("_", "-") for run in absent[name])
            parser.error(f"{name} requires {flags}")
    out = args.paper_dir / "figures" / "source" / "appendix"
    out.mkdir(parents=True, exist_ok=True)
    for name in args.products:
        produce = products[name]
        (out / f"{name}.json").write_text(json.dumps(produce(), indent=1))
        print("wrote", out / f"{name}.json", flush=True)


if __name__ == "__main__":
    main()
