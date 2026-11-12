"""Value-level contracts for read-only aggregation of completed LRG exports."""

from copy import deepcopy
import itertools

import numpy as np
import pytest

from pzbao.figures import data as figures


def config():
    return {
        "analysis": {
            "workloads": {
                "full_grid": {"sigma_z_tiers": ["native", "sigma_0.001"], "mock_ids": [0, 1]}
            }
        },
        "reporting": {
            "gain": {
                "ratio": "post_over_pre",
                "one_dimensional_fom": "uncertainty",
                "profile_uncertainty": "minos_half_width",
            }
        },
        "tracers": {"LRG": {"fkp_p0_h3_mpc3": 10000.0}},
    }


def row(name="LRG1", condition="native", state="prereconstruction", mock=0, parameter="qiso"):
    width = 0.02 if state == "prereconstruction" else 0.01
    value = 1 + 0.01 * mock
    identity = {
        "run_name": "--".join((name, condition, state)),
        "code_commit": "producer",
        "analysis_config_sha256": "--".join((name, condition, state)),
    }
    return dict(
        measurement_bin=name,
        condition=condition,
        state=state,
        mock=mock,
        parameter=parameter,
        value=value,
        width=width,
        valid=True,
        boundary=False,
        minos_bounds=[value - width, value + width],
        interval_flags=dict(
            is_valid=True,
            lower_valid=True,
            upper_valid=True,
            at_lower_limit=False,
            at_upper_limit=False,
        ),
        optimizer={
            "maximization": dict(
                is_valid=True, is_above_max_edm=False, has_reached_call_limit=False
            )
        },
        identity=dict(
            identity,
            measurement_bin=name,
            condition=condition,
            reconstruction_state=state,
            realization_id=mock,
        ),
        covariance_identity={
            "run_name": "baseline",
            "code_commit": "covariance",
            "analysis_config_sha256": "cov-config",
        },
        detection={"delta_chi2": -2.0},
    )


def cohort(bins=figures.LRG_BINS):
    return [
        row(*args)
        for args in itertools.product(
            bins,
            config()["analysis"]["workloads"]["full_grid"]["sigma_z_tiers"],
            figures.STATES,
            [0, 1],
            figures.PARAMETERS,
        )
    ]


def provenance(rows):
    return dict(
        analysis_config=config(),
        release={"definition": {"code_commit": "producer"}, "resolved_snapshot_sha256": "snapshot"},
        route_identities=[{key: r["identity"][key] for key in figures.IDENTITY} for r in rows],
    )


def complete(rows):
    return {
        "measurement_bins": sorted({r["measurement_bin"] for r in rows}),
        "fit_count": len(rows) // 2,
        "completed_at": "2026-10-05T00:00:00+00:00",
    }


def preparation():
    groups = {
        name: {
            "counts": {b: count for b in figures.LRG_BINS},
            "volumes_mpc_h3": {b: volume for b in figures.LRG_BINS},
        }
        for name, count, volume in [("NGC", 100000, 1e9), ("SGC", 400000, 2e9)]
    }
    return dict(
        tracer="LRG",
        realization_ids=[0, 1],
        conditions={c: deepcopy(groups) for c in ("native", "sigma_0.001")},
        effective_redshifts={
            c: {b: {"z_eff": 0.7} for b in figures.LRG_BINS} for c in ("native", "sigma_0.001")
        },
    )


def test_gain_orientation_pairing_and_signed_diagnostics():
    rows = list(reversed(cohort()))
    pairs = figures.paired_lrg_rows(rows, config()["analysis"]["workloads"]["full_grid"])
    assert all(p["uncertainty_ratio"] == pytest.approx(0.5) for p in pairs)
    assert all(p["precision_ratio"] == pytest.approx(2) for p in pairs)
    assert all(p["post_minus_pre"] == 0 for p in pairs)
    assert all(r["detection"]["delta_chi2"] == -2 for r in rows)


@pytest.mark.parametrize("failure", ["boundary", "invalid", "unavailable", "optimizer"])
def test_flags_are_retained_without_ordinary_gain(failure):
    rows = cohort()
    target = rows[0]
    if failure == "boundary":
        target["boundary"] = True
        target["interval_flags"]["at_lower_limit"] = True
    elif failure == "invalid":
        target["interval_flags"]["lower_valid"] = False
        target["valid"] = False
    elif failure == "unavailable":
        target["minos_bounds"] = None
        target["width"] = None
        target["valid"] = False
    else:
        target["optimizer"]["maximization"]["has_reached_call_limit"] = True
    pair = figures.paired_lrg_rows(rows, config()["analysis"]["workloads"]["full_grid"])[0]
    assert pair["uncertainty_ratio"] is None and pair["precision_ratio"] is None
    assert not pair["usable"]
    assert pair["pre_interval_flags"] == target["interval_flags"]
    assert pair["raw_width_ratio"] == (None if failure == "unavailable" else pytest.approx(0.5))


@pytest.mark.parametrize("failure", ["missing", "duplicate", "identity", "producer", "flags"])
def test_report_rejects_invalid_cohort_or_provenance(failure):
    rows = cohort()
    source, done = provenance(rows), complete(rows)
    if failure == "missing":
        rows.pop()
    elif failure == "duplicate":
        rows[-1] = deepcopy(rows[0])
    elif failure == "identity":
        rows[0]["identity"]["realization_id"] = 8
    elif failure == "producer":
        rows[0]["identity"]["code_commit"] = "other"
    else:
        rows[0]["valid"] = False
    with pytest.raises(ValueError):
        figures.validate_lrg_rows(rows, source, done)


def test_forecast_uses_cap_sum_and_fractional_units():
    row = figures.lrg_forecast_rows(preparation(), config())[0]
    expected_veff = 1e9 / 4 + 2e9 * (2 / 3) ** 2
    assert row["veff_mpc_h3"] == pytest.approx(expected_veff)
    expected_fraction = (
        figures.r_precision_percent(expected_veff, row["calibration_coefficient_percent"]) / 100
    )
    assert row["sigma_r_over_r"] == pytest.approx(expected_fraction)
    assert row["cosmic_variance_sigma_r_over_r"] < row["sigma_r_over_r"]
    # A volume-averaged density gives a different result for unequal cap densities.
    nbar = 500000 / 3e9
    assert row["veff_mpc_h3"] != pytest.approx(3e9 * (nbar * 10000 / (1 + nbar * 10000)) ** 2)


@pytest.mark.parametrize("bad", [0, -1, float("nan")])
def test_forecast_rejects_invalid_volume(bad):
    prep = preparation()
    prep["conditions"]["native"]["NGC"]["volumes_mpc_h3"]["LRG1"] = bad
    with pytest.raises(ValueError, match="counts or volume"):
        figures.lrg_forecast_rows(prep, config())


def curve_archive(rows):
    curves = {}
    for r in rows:
        prefix = f"{r['measurement_bin']}--{r['condition']}__{r['state']}__mock{r['mock']}__"
        data = np.array([1.0, 2.0, 3.0, 4.0]) + r["mock"] * 2
        for key, value in dict(
            separation=np.array([50.0, 54.0]),
            fit_edges=np.array([48.0, 52.0, 56.0]),
            multipoles=np.array([0, 2]),
            data=data,
            model=data - 1,
            residual=np.ones(4),
            standard_deviation=np.ones(4) * 2,
        ).items():
            curves[prefix + key] = value
    return curves


def test_clustering_scatter_is_not_mean_error_and_rejects_bad_residual():
    curves = curve_archive(cohort())
    result = figures.lrg_clustering(curves, "LRG1", "native", figures.STATES[0], [0, 1])
    np.testing.assert_allclose(result["mean"], [2, 3, 4, 5])
    np.testing.assert_allclose(result["scatter"], np.sqrt(2))
    curves["LRG1--native__prereconstruction__mock0__residual"][0] = 100
    with pytest.raises(AssertionError):
        figures.lrg_clustering(curves, "LRG1", "native", figures.STATES[0], [0, 1])


def test_sources_detect_concurrent_change(tmp_path):
    source = tmp_path / "source.json"
    source.write_text('{"value": 1}')
    inputs = figures.Sources()
    inputs.json(source)
    source.write_text('{"value": 2}')
    with pytest.raises(ValueError, match="source bytes changed"):
        inputs.unchanged()


def test_saved_lrg_tables_load_without_historical_renderers(tmp_path):
    reports = []
    for index, bins in enumerate((("LRG1", "LRG2"), ("LRG3",))):
        directory = tmp_path / str(index)
        directory.mkdir()
        rows = cohort(bins)
        metadata = provenance(rows)
        metadata["source_bytes"] = {"/retired/renderer.py": "old-source"}
        for name, value in (
            ("fit-diagnostics", rows),
            ("provenance", metadata),
            ("complete", complete(rows)),
        ):
            figures.write_json(directory / (name + ".json"), value)
        np.savez(directory / "fit-curves.npz", **curve_archive(rows))
        reports.append(directory)
    loaded = figures.load_lrg_reports(reports)
    assert len(loaded["rows"]) == len(cohort())
    assert all(row["uncertainty_ratio"] == pytest.approx(0.5) for row in loaded["pairs"])
    assert all(row["detection"]["delta_chi2"] == -2.0 for row in loaded["rows"])
    with pytest.raises(ValueError, match="Duplicate LRG curve"):
        figures.load_lrg_reports([reports[0], reports[0], reports[1]])
