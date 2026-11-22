"""Checks of the appendix number producers that need no restored runs."""

import csv
import json
import sys

import numpy as np
import pytest
import yaml

from pzbao.figures import appendix_checks

RECONSTRUCTION_COLUMNS = [
    "route",
    "realization_id",
    "cap",
    "shell",
    "supported_cell_count",
    "threshold_zeroed_cell_count",
]


def test_linear_alpha_perp_width_matches_narrow_draws():
    rng = np.random.default_rng(1)
    draws = rng.multivariate_normal([1.0, 1.0], [[1e-6, 5e-7], [5e-7, 4e-6]], size=200_000)
    qiso, qap = draws.T
    linear = appendix_checks._linear_alpha_perp_std(qiso, qap)
    assert linear == pytest.approx(appendix_checks._alpha_perp(qiso, qap).std(), rel=1e-2)


def write_run(root, name, radius, rows, *, source=None):
    run = root / name
    (run / "review").mkdir(parents=True)
    reconstruction = {
        "smoothing_values_mpc_h": [radius],
        "threshold_randoms": {"value": 0.01, "prescription": "pyrecon_native_mean_random_weight"},
    }
    (run / "analysis.yaml").write_text(
        yaml.safe_dump({"measurement": {"reconstruction": reconstruction}})
    )
    columns = RECONSTRUCTION_COLUMNS + (["displacement_source_class"] if source else [])
    with open(run / "review" / "reconstruction.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, **({"displacement_source_class": source} if source else {})})
    return run


def field_rows(route, zeroed):
    # Two realizations, one cap, two shells: realization 0 loses 10 of 100 cells, realization 1
    # loses `zeroed` of 300.
    return [
        dict(zip(RECONSTRUCTION_COLUMNS, [route, "0", "NGC", "0", 60, 6])),
        dict(zip(RECONSTRUCTION_COLUMNS, [route, "0", "NGC", "1", 40, 4])),
        dict(zip(RECONSTRUCTION_COLUMNS, [route, "1", "NGC", "0", 300, zeroed])),
    ]


def test_random_floor_counts_each_field_once(tmp_path):
    core = write_run(
        tmp_path / "core",
        "unified-fit-v3--primary",
        15.0,
        field_rows("LRG3|sigma_0.001|prereconstruction", 30)
        + field_rows("LRG3|sigma_0.001|postreconstruction", 30),
    )
    transfer = tmp_path / "transfer"
    for (kind, radius), name in appendix_checks.TRANSFER_RUNS.items():
        rows = field_rows("LRG3|sigma_0.01|postreconstruction", 30)
        if kind == "cross":
            rows += field_rows("LRG3|sigma_0.03|postreconstruction", 30)
        write_run(transfer, name, float(radius), rows, source="sigma_0.003")

    result = appendix_checks.random_floor(core, transfer)

    (field,) = [row for row in result["fields"] if row["run"] == "unified-fit-v3--primary"]
    assert field["field_class"] == "sigma_0.001"
    assert field["smoothing_mpc_h"] == 15.0
    assert field["supported_cell_count"] == 400
    assert field["threshold_zeroed_cell_count"] == 40
    assert field["pooled_share"] == pytest.approx(0.1)
    assert field["realization_share_mean"] == pytest.approx(0.1)
    assert len(field["routes"]) == 2
    cross = [row for row in result["fields"] if row["run"] == "unified-fit-v3--sigma_0.003--R25"]
    assert [row["field_class"] for row in cross] == ["sigma_0.003"]
    assert cross[0]["supported_cell_count"] == 400


def test_random_floor_refuses_routes_that_disagree(tmp_path):
    core = write_run(
        tmp_path / "core",
        "unified-fit-v3--primary",
        15.0,
        field_rows("LRG3|sigma_0.001|prereconstruction", 30)
        + field_rows("LRG3|sigma_0.001|postreconstruction", 31),
    )
    with pytest.raises(ValueError, match="disagree"):
        appendix_checks.random_floor(core, tmp_path / "transfer")


def run_main(monkeypatch, tmp_path, *arguments):
    names = {
        "cross_sigma_checks": "cross-sigma-checks",
        "alpha_sn_leave_one_out": "alpha-sn-leave-one-out",
        "cross_cap_correlation": "cross-cap",
        "cross_sigma_alpha_perp": "cross-sigma-alpha-perp",
        "alpha_perp_narrowing": "alpha-perp-narrowing",
        "random_floor": "random-floor",
        "prior_width_ratios": "prior-width-ratios",
        "template_b1": "template-b1",
        "matched_density_posterior": "matched-density-posterior",
        "convergence": "convergence",
        "interval_coverage": "interval-coverage",
        "threshold_interval": "threshold-interval",
        "combined_gain_uncertainty": "combined-gain-uncertainty",
    }
    for function, product in names.items():
        monkeypatch.setattr(
            appendix_checks, function, lambda *args, product=product, **kw: {"product": product}
        )
    monkeypatch.setattr(
        sys,
        "argv",
        ["appendix_checks", "--paper-dir", str(tmp_path)] + list(arguments),
    )
    appendix_checks.main()
    out = tmp_path / "figures" / "source" / "appendix"
    return sorted(json.loads(path.read_text())["product"] for path in out.glob("*.json"))


MEASURED = ("--primary-run", "p", "--transfer-root", "t")


def test_default_products_need_no_core_run(monkeypatch, tmp_path):
    written = run_main(monkeypatch, tmp_path, *MEASURED)
    assert written == [
        "alpha-sn-leave-one-out",
        "cross-cap",
        "cross-sigma-alpha-perp",
        "cross-sigma-checks",
    ]


def test_core_run_adds_its_products(monkeypatch, tmp_path):
    written = run_main(monkeypatch, tmp_path, *MEASURED, "--core-run", "c")
    assert len(written) == 12
    assert {
        "convergence",
        "interval-coverage",
        "threshold-interval",
        "combined-gain-uncertainty",
    } <= set(written)


def test_density_run_alone_writes_only_its_product(monkeypatch, tmp_path):
    assert run_main(monkeypatch, tmp_path, "--density-run", "d") == ["matched-density-posterior"]


@pytest.mark.parametrize("product", ["template-b1", "cross-cap", "matched-density-posterior"])
def test_product_without_its_run_is_refused(monkeypatch, tmp_path, product):
    with pytest.raises(SystemExit):
        run_main(monkeypatch, tmp_path, "--products", product)


def write_density_run(root, *, workload="density"):
    run = root / "unified-fit-v3--density"
    run.mkdir(parents=True)
    (run / "run.yaml").write_text(
        yaml.safe_dump(
            {
                "run_name": run.name,
                "code_commit": "c" * 40,
                "analysis_config_sha256": "a" * 64,
                "workload": workload,
            }
        )
    )
    density = {
        "measurement_bins": ["LRG2"],
        "sigma_z_tiers": ["sigma_0.001"],
        "reconstruction_states": list(appendix_checks.STATES),
    }
    (run / "analysis.yaml").write_text(
        yaml.safe_dump({"analysis": {"workloads": {"density": density}}})
    )
    return run


def test_matched_density_posterior_reports_the_width_of_the_draws(monkeypatch, tmp_path):
    from pzbao import fitting

    run = write_density_run(tmp_path)
    rng = np.random.default_rng(3)
    draws = {state: rng.normal(1.0, 0.02, size=(2, 500)) for state in appendix_checks.STATES}
    requested = []

    def read_profiles(path, *, identity):
        requested.append((path.relative_to(run).as_posix(), identity))
        at_bound = identity["reconstruction_state"] == "postreconstruction"
        parameters = [
            {"basename": name, "value": 1.0, "boundary": at_bound and name == "qap"}
            for name in ("qiso", "qap")
        ]
        return {"parameters": parameters, "maximization": {}}

    monkeypatch.setattr(fitting, "read_profiles", read_profiles)
    monkeypatch.setattr(
        appendix_checks,
        "_posterior_draws",
        lambda root, route: tuple(draws[route["reconstruction_state"]]),
    )

    result = appendix_checks.matched_density_posterior(run)

    assert result["run"]["run_name"] == "unified-fit-v3--density"
    assert [(row["reconstruction_state"], row["parameter"]) for row in result["rows"]] == [
        (state, name) for state in appendix_checks.STATES for name in ("qiso", "qap")
    ]
    for row in result["rows"]:
        values = draws[row["reconstruction_state"]][("qiso", "qap").index(row["parameter"])]
        assert row["tracer"] == "LRG" and row["measurement_bin"] == "LRG2"
        assert row["mean"] == pytest.approx(values.mean())
        assert row["sd"] == pytest.approx(values.std(ddof=1))
        assert row["draws"] == 500
        assert row["prior_contact"] == (
            (row["reconstruction_state"], row["parameter"]) == ("postreconstruction", "qap")
        )
    # The prior contact is read from the fit to the mean of the same route and run.
    path, identity = requested[0]
    assert path == "fit/units/LRG2--sigma_0.001--prereconstruction--mean/profiles.npy"
    assert identity["realization_id"] is None and identity["code_commit"] == "c" * 40


def test_matched_density_posterior_refuses_another_workload(tmp_path):
    run = write_density_run(tmp_path, workload="full_grid")
    with pytest.raises(ValueError, match="not a matched-density run"):
        appendix_checks.matched_density_posterior(run)


ROUTES = ["|".join(route) for route in appendix_checks._routes()]


def write_review_table(run, name, columns, rows):
    (run / "review").mkdir(parents=True, exist_ok=True)
    with open(run / "review" / f"{name}.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def test_convergence_keeps_the_last_check_and_flags_withheld_autocorrelation_times(tmp_path):
    run = tmp_path / "unified-fit-v3--primary"
    history, sampler = [], []
    for route in ROUTES:
        for step, value in ((300, 0.03), (600, 0.008)):
            history.append(
                {
                    "route": route,
                    "statistic": "max_generalized_covariance_ratio_minus_one",
                    "value": value,
                    "retained_steps": step // 2,
                    "step": step,
                    "stopping_rule_reached": step == 600,
                }
            )
        for parameter in ("qiso", "qap", "b1"):
            # One ensemble with an accepted estimate, one whose estimate the sampler withheld.
            sampler.append(
                {
                    "route": route,
                    "ensemble_index": 0,
                    "parameter": parameter,
                    "tau": 40.0,
                    "unavailable_reason": "",
                    "retained_draws": 12000,
                }
            )
            sampler.append(
                {
                    "route": route,
                    "ensemble_index": 1,
                    "parameter": parameter,
                    "tau": "",
                    "unavailable_reason": "The chain is shorter than 50 times\nN/50 = 6;\ntau: [60.0]",
                    "retained_draws": 12000,
                }
            )
    write_review_table(run, "gr", list(history[0]), history)
    write_review_table(run, "sampler", list(sampler[0]), sampler)

    result = appendix_checks.convergence(run)

    assert [row["route"] for row in result["rows"]] == ROUTES
    row = result["rows"][0]
    assert row["gelman_rubin_last"] == 0.008 and row["steps"] == 600
    assert row["retained_steps"] == 300 and row["stopping_rule_reached"] is True
    assert row["qap"] == {
        "ensembles": 2,
        "ensembles_shorter_than_50_tau": 1,
        "tau_max": 60.0,
        "effective_sample_size": pytest.approx(12000 / 40 + 12000 / 60),
        "effective_sample_size_min_ensemble": pytest.approx(200.0),
        "retained_draws": 24000,
    }


def test_convergence_refuses_a_table_that_misses_a_fit(tmp_path):
    run = tmp_path / "unified-fit-v3--primary"
    columns = ["route", "statistic", "value", "retained_steps", "step", "stopping_rule_reached"]
    write_review_table(run, "gr", columns, [dict(zip(columns, [ROUTES[0], "s", 0.1, 1, 2, True]))])
    write_review_table(run, "sampler", ["route"], [])
    with pytest.raises(ValueError, match="50 fits"):
        appendix_checks.convergence(run)


def test_interval_coverage_takes_pulls_from_the_intervals_clear_of_the_bounds(tmp_path):
    run = tmp_path / "unified-fit-v3--primary"
    coverage, pulls, recovery = [], [], []
    for route in ROUTES:
        for parameter in ("qiso", "qap"):
            for subset, ids, inside in (
                ("all_finite", [0, 1, 2, 3], 3),
                ("valid_non_boundary", [1, 2, 3], 2),
            ):
                coverage.append(
                    {
                        "route": route,
                        "parameter": parameter,
                        "interval_kind": "native_minos",
                        "subset": subset,
                        "denominator": len(ids),
                        "numerator": inside,
                        "mock_ids": json.dumps(ids),
                        "fraction": inside / len(ids),
                        "wilson_lower": 0.4,
                        "wilson_upper": 0.9,
                    }
                )
            # Realization 0 sits at a prior bound: its large pull must not enter.
            for mock, pull in enumerate((-7.0, -1.0, 0.0, 2.0)):
                pulls.append(
                    {"route": route, "realization_id": mock, "parameter": parameter, "pull": pull}
                )
            recovery.append(
                {
                    "route": route,
                    "parameter": parameter,
                    "mean_offset_from_truth": 0.002,
                    "mock_scatter_ddof1": 0.02,
                    "mean_offset_standard_error": 0.004,
                }
            )
    write_review_table(run, "interval_coverage", list(coverage[0]), coverage)
    write_review_table(run, "profile_recovery", list(pulls[0]), pulls)
    write_review_table(run, "ensemble_recovery", list(recovery[0]), recovery)

    result = appendix_checks.interval_coverage(run)

    assert len(result["rows"]) == 100
    row = result["rows"][1]
    assert (row["route"], row["parameter"]) == (ROUTES[0], "qap")
    assert row["all_finite"] == {
        "contains_truth": 3,
        "intervals": 4,
        "fraction": 0.75,
        "wilson68": [0.4, 0.9],
    }
    assert row["valid_non_boundary"]["intervals"] == 3
    assert row["pulls"] == 3
    assert row["pull_mean"] == pytest.approx(1 / 3)
    assert row["pull_std"] == pytest.approx(np.std([-1.0, 0.0, 2.0], ddof=1))
    assert row["best_fit_scatter"] == 0.02


def write_fit_run(root, *, mocks=6):
    run = root / "unified-fit-v3--primary"
    run.mkdir(parents=True)
    config = {
        "analysis": {"mock_ids": list(range(mocks))},
        "fitting": {"sampling": {"seed": 4201}},
        "tracers": {
            "LRG": {"fkp_p0_h3_mpc3": 10000.0},
            "BGS_BRIGHT-21.5": {"fkp_p0_h3_mpc3": 7000.0},
        },
    }
    (run / "analysis.yaml").write_text(yaml.safe_dump(config))
    return run


def test_threshold_interval_gives_three_gains_per_sample(monkeypatch, tmp_path):
    run = write_fit_run(tmp_path)
    flags = {"is_valid": True, "lower_valid": True, "upper_valid": True}
    clear = {**flags, "at_lower_limit": False, "at_upper_limit": False}
    profiles, gains = [], []
    for measurement_bin, condition, state in appendix_checks._routes():
        unit = run / "fit" / "units" / "--".join([measurement_bin, condition, state, "mean"])
        unit.mkdir(parents=True)
        density = 5e-4 if condition == "native" else 3e-5
        (unit / "fit.json").write_text(json.dumps({"attained_density_h3_mpc3": density}))
        pre = state == "prereconstruction"
        # LRG3 sigma_0.03 after reconstruction reaches a prior bound: no profile gain there.
        railed = (measurement_bin, condition, pre) == ("LRG3", "sigma_0.03", False)
        profiles.append(
            {
                "route": "|".join([measurement_bin, condition, state]),
                "fit_scope": "mean",
                "name": "qiso",
                "value": 1.0,
                "minos_bounds": json.dumps([0.97, 1.03] if pre else [0.98, 1.02]),
                "interval_flags": json.dumps({**clear, "at_upper_limit": railed}),
            }
        )
        if pre:
            gains.append(
                {
                    "measurement_bin": measurement_bin,
                    "condition": condition,
                    "parameter": "qiso",
                    "pre_fom": 0.03,
                    "post_fom": 0.02,
                }
            )
    write_review_table(run, "profiles", list(profiles[0]), profiles)
    write_review_table(run, "aggregate_gains", list(gains[0]), gains)

    def mock_fits(core_run, measurement_bin, condition, state):
        error = 0.04 if state == "prereconstruction" else 0.02
        fits = {
            mock: {"qiso": 1.0, "qap": 1.0, "qiso_err": error, "qap_err": 0.1, "optimizer_ok": True}
            for mock in range(6)
        }
        fits[5]["optimizer_ok"] = state == "prereconstruction"  # fails after reconstruction
        fits[4]["qiso"] = 0.8  # at the prior bound: counted, and still paired
        return fits

    monkeypatch.setattr(appendix_checks, "_mock_fits", mock_fits)

    result = appendix_checks.threshold_interval(run, resamples=50)

    assert len(result["rows"]) == 25
    products = [row["nbar_p0"] for row in result["rows"]]
    assert products == sorted(products)
    by_sample = {(row["bin"], row["condition"]): row for row in result["rows"]}
    row = by_sample["BGS1", "native"]
    assert row["nbar_p0"] == pytest.approx(5e-4 * 7000.0)
    assert row["gain_posterior_std"] == pytest.approx(1.5)
    assert row["gain_profile_half_width"] == pytest.approx(1.5)
    assert row["paired_realizations"]["n"] == 5
    assert row["paired_realizations"]["median"] == pytest.approx(2.0)
    assert row["paired_realizations"]["n_railed"] == 1
    assert by_sample["LRG3", "sigma_0.03"]["gain_profile_half_width"] is None
    assert by_sample["LRG3", "sigma_0.03"]["nbar_p0"] == pytest.approx(0.3)


def combined_inputs(monkeypatch, *, post_scale, mocks=30):
    """Class posteriors and realization best fits for one synthetic combination."""
    rng = np.random.default_rng(11)
    classes = [c for c in appendix_checks.CLASSES if c != "native"]
    draws = {
        c: rng.multivariate_normal([1, 1], [[4e-4, 1e-4], [1e-4, 9e-4]], 4000) for c in classes
    }
    best = {c: 1 + 0.02 * rng.standard_normal((mocks, 2)) for c in classes}

    def posterior_draws(run, route):
        scale = post_scale if route["reconstruction_state"] == "postreconstruction" else 1.0
        values = 1 + (draws[route["condition"]] - 1) * scale
        return values[:, 0], values[:, 1]

    def mock_fits(run, measurement_bin, condition, state):
        fits = {
            mock: {
                "qiso": best[condition][mock, 0],
                "qap": best[condition][mock, 1],
                "optimizer_ok": True,
            }
            for mock in range(mocks)
        }
        if (condition, state) == ("sigma_0.01", "postreconstruction"):
            fits[mocks - 1]["optimizer_ok"] = False
        return fits

    monkeypatch.setattr(appendix_checks, "_posterior_draws", posterior_draws)
    monkeypatch.setattr(appendix_checks, "_mock_fits", mock_fits)


def test_combined_gain_resamples_the_same_realizations_in_both_states(monkeypatch, tmp_path):
    from pzbao.fitting import combine_tier_posteriors, estimate_correlation

    run = write_fit_run(tmp_path, mocks=30)
    # Halving every posterior width quadruples the figure of merit whatever the correlation is,
    # because the same realizations measure it before and after reconstruction.
    combined_inputs(monkeypatch, post_scale=0.5)

    result = appendix_checks.combined_gain_uncertainty(run, resamples=40)

    assert result["seed"] == 4201 and len(result["rows"]) == 5
    for row in result["rows"]:
        assert row["realizations_used"] == 29 and row["realizations_excluded"] == [29]
        assert row["gain_percent"] == pytest.approx(300.0)
        assert row["bootstrap_gain_percent_16_84"] == pytest.approx([300.0, 300.0])
        assert row["bootstrap_gain_percent_2p5_97p5"] == pytest.approx([300.0, 300.0])
        assert row["jackknife_gain_percent_standard_error"] == pytest.approx(0.0, abs=1e-6)
        assert row["bootstrap_resamples_used"] + row["bootstrap_resamples_skipped"] == 40
    # The point value is the combination of the figures, built from the same inputs.
    row = result["rows"][0]
    classes = row["classes"]
    means, covariances, estimates = [], [], []
    for condition in classes:
        route = {"condition": condition, "reconstruction_state": "prereconstruction"}
        values = np.stack(appendix_checks._posterior_draws(run, route), axis=1)
        means.append(values.mean(axis=0))
        covariances.append(np.cov(values, rowvar=False, ddof=1))
        fits = appendix_checks._mock_fits(run, "BGS1", condition, "prereconstruction")
        estimates.append([[fits[m]["qiso"], fits[m]["qap"]] for m in range(29)])
    expected = combine_tier_posteriors(
        classes,
        np.array(means),
        np.array(covariances),
        estimate_correlation(np.transpose(np.array(estimates), (1, 0, 2))),
    )
    assert row["fom_pre"] == pytest.approx(expected["two_dimensional_fom"])
    assert row["fom_post"] == pytest.approx(4 * expected["two_dimensional_fom"])
