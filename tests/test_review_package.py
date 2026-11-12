"""Saved native science reaches static reports, including explicit partial coverage."""

import copy
import csv
import json
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pytest
import yaml

from pzbao import run as driver
from pzbao.diagnostics import review
from tests.test_boundary_support import per_bin_preparation_fields
from pzbao.fitting import INTERVAL_CONVERGENCE_FLAGS


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.setattr(
        driver, "_code_identity", lambda: {"code_commit": "a" * 40, "repository": str(tmp_path)}
    )
    return driver.create_run(tmp_path, "review-fixture", workload="pilot", catalog_root=tmp_path)


def _route(run):
    return driver.routes_for(run["config"], run["metadata"]["workload"])[0]


def _identity(run, route):
    return {
        **{
            key: run["metadata"][key]
            for key in ("run_name", "code_commit", "analysis_config_sha256")
        },
        **route,
    }


def _profile(directory, identity, *, value=1.02, offsets=(-0.03, 0.08), parameter_names=("qiso",)):
    pytest.importorskip("desilike")
    from desilike.samples import Profiles
    from desilike.samples.profiles import ParameterBestFit
    from desilike.parameter import Parameter, Samples

    parameters = [Parameter(name, fixed=False) for name in parameter_names]
    flags = {"is_valid": True, "is_above_max_edm": False, "has_reached_call_limit": False}
    bestfit = ParameterBestFit(
        [*[np.array([value]) for _ in parameters], np.array([-4.0]), np.array([-4.5])],
        params=[*parameters, "loglikelihood", "logposterior"],
        attrs=flags,
    )
    interval_flags = {name: False for name in INTERVAL_CONVERGENCE_FLAGS}
    interval_flags.update(is_valid=True, lower_valid=True, upper_valid=True, at_upper_limit=True)
    profiles = Profiles(
        bestfit=bestfit,
        error=Samples([np.array([0.04]) for _ in parameters], params=parameters),
        interval=Samples(
            [np.array(offsets) for _ in parameters],
            params=parameters,
            attrs={name: interval_flags.copy() for name in parameter_names},
        ),
        attrs={"identity": identity},
    )
    directory.mkdir(parents=True, exist_ok=True)
    profiles.save(str(directory / "profiles.npy"))
    (directory / "fit.json").write_text(
        json.dumps(
            {
                "identity": identity,
                "degrees_of_freedom": 19,
                "detection": {"delta_chi2": 2.0},
                "parameters": [{"value": -999.0}],
            }
        )
    )
    return profiles


def _chain(
    directory,
    identity,
    *,
    mean=1.0,
    spread=0.02,
    constant=False,
    parameter_names=("qiso",),
    steps=400,
):
    Chain = pytest.importorskip("desilike.samples").Chain
    Parameter = pytest.importorskip("desilike.parameter").Parameter
    rng = np.random.default_rng(234 + identity["ensemble_index"])
    data = [
        np.full((steps, 40), mean) if constant else rng.normal(mean, spread, size=(steps, 40))
        for _ in parameter_names
    ]
    values = data[0]
    chain = Chain(data=data, params=[Parameter(name, fixed=False) for name in parameter_names])
    chain.attrs.update(
        identity=identity,
        acceptance_fraction=[0.4] * 40,
        random_stream={"ensemble_index": identity["ensemble_index"]},
    )
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"ensemble-{identity['ensemble_index']}.npy"
    chain.save(str(path))
    return chain, values


def _table(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


@pytest.mark.parametrize("failure_stage", ["tight_layout", "savefig"])
def test_failed_figure_save_closes_only_its_own_figure(tmp_path, monkeypatch, failure_stage):
    plt = review._figure()
    existing, figure = plt.figure(layout="none"), plt.figure(layout="none")

    def fail(*args, **kwargs):
        raise ValueError("synthetic rendering failure")

    monkeypatch.setattr(figure, failure_stage, fail)
    try:
        with pytest.raises(ValueError, match="synthetic rendering failure"):
            review._save(figure, tmp_path / "failed.png")
        assert not plt.fignum_exists(figure.number)
        assert plt.fignum_exists(existing.number)
    finally:
        plt.close(existing)


def test_constrained_colorbar_saves_both_formats_and_preserves_other_figures(tmp_path):
    plt = review._figure()
    with plt.rc_context({"figure.constrained_layout.use": True}):
        existing = plt.figure()
        figure, axis = plt.subplots()
        data = np.arange(4).reshape(2, 2)
        artist = axis.imshow(data)
        figure.colorbar(artist, ax=axis)
        layout = figure.get_layout_engine()
        try:
            paths = review._save(figure, tmp_path / "colorbar")
            assert Path(paths["png"]).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
            assert Path(paths["pdf"]).read_bytes().startswith(b"%PDF")
            assert figure.get_layout_engine() is layout
            np.testing.assert_array_equal(artist.get_array(), data)
            assert not plt.fignum_exists(figure.number)
            assert plt.fignum_exists(existing.number)
        finally:
            plt.close(figure)
            plt.close(existing)


def test_saved_boundary_support_reaches_csv_with_clustering_inflow_and_shortfall(run):
    from pzbao.selection import prepare_data_for_selection

    data = {
        "TARGETID": np.array([1, 4, 8, 11]),
        "SOURCE_ID": np.array([1, 4, 8, 11], dtype=np.uint64),
        "RA": np.array([10.0, 20.0, 30.0, 40.0]),
        "DEC": np.zeros(4),
        "Z": np.array([0.415, 0.99, 1.02, 1.0]),
    }
    reservoir = {
        "TARGETID": np.array([900, 1000, 2, 5]),
        "SOURCE_ID": np.array([9, 10, 2, 5], dtype=np.uint64),
        "RA": np.array([30.0, 40.0, 50.0, 60.0]),
        "DEC": np.zeros(4),
        "Z": np.array([1.02, 1.0, 0.14, 1.29]),
    }
    unit = next(
        item
        for item in driver.stage_units("select", run["config"], "pilot")
        if item["tracer"] == "LRG"
    )
    caps = {}
    identifiers = run["config"]["analysis"]["mock_ids"]
    for cap in ("NGC", "SGC"):
        _, diagnostic = prepare_data_for_selection(
            data,
            tracer="LRG",
            tier_label="sigma_0.03",
            cap=cap,
            reservoir=reservoir,
            reservoir_record={"tracer": "LRG", "sample_identity": "review-test"},
            config=run["config"],
        )
        boundary = diagnostic["boundary_counts"]
        counts = np.asarray(boundary["available_observed_counts"])
        caps[cap] = {
            "radial": {
                "z_edges": boundary["z_edges"],
                "mean_counts": counts.tolist(),
                "target_probability_mass": (counts / counts.sum()).tolist(),
            },
            "boundary_counts_by_mock": [
                {"realization_id": identifier, **boundary} for identifier in identifiers
            ],
        }
    directory = run["root"] / "select" / "units" / unit["unit_id"]
    directory.mkdir(parents=True)
    driver._write_yaml(
        directory / "result.yaml",
        {
            **{
                key: run["metadata"][key]
                for key in ("run_name", "code_commit", "analysis_config_sha256")
            },
            "stage": "select",
            "unit": unit,
        },
    )
    (directory / "preparation.json").write_text(
        json.dumps(
            {
                "tracer": "LRG",
                "realization_ids": identifiers,
                "conditions": {"sigma_0.03": caps},
                **per_bin_preparation_fields(run["config"], "LRG"),
            }
        )
    )
    destination = run["root"] / "review"
    figures, _, support, errors = review._input_review(run, destination)
    assert len(figures) == 1
    assert [row["input_tracer"] for row in errors] == ["BGS_BRIGHT-21.5"]
    assert {(row["cap"], row["realization_id"], row["side"]) for row in support} == {
        (cap, identifier, side)
        for cap in caps
        for identifier in identifiers
        for side in ("below", "above")
    }
    path = destination / "boundary_support.csv"
    review._write_csv(path, support)
    rows = _table(path)
    row = next(item for item in rows if item["cap"] == "NGC" and item["side"] == "above")
    assert row["tracer"] == "LRG" and row["condition"] == "sigma_0.03"
    assert int(row["overlap_removed_row_count"]) == 2
    assert int(row["original_scattered_in_count"]) == 2
    assert int(row["original_scattered_out_count"]) == 1
    assert int(row["clustering_row_count"]) == 2
    assert int(row["added_reservoir_row_count"]) == 1
    assert float(row["shortfall_redshift"]) == pytest.approx(0.07)
    assert row["guard_reached"] == "False"
    # A moved publication must not be consumed as a matching preparation.
    result = yaml.safe_load((directory / "result.yaml").read_text())
    result["code_commit"] = "b" * 40
    driver._write_yaml(directory / "result.yaml", result)
    _, _, support, errors = review._input_review(run, destination)
    assert not support
    assert any("different snapshot" in row["error"] for row in errors)


def test_native_profile_values_and_asymmetric_bounds_reach_regenerated_report(run):
    route = _route(run)
    directory = run["root"] / "fit" / "units" / (driver.route_name(route) + "--mock-0")
    profiles = _profile(directory, {**_identity(run, route), "realization_id": 0})
    result = review.render_review(run["root"])
    destination = run["root"] / "review"
    rows = _table(destination / "profiles.csv")
    row = (
        next(item for item in rows if item["parameter"] == "qiso")
        if rows and "parameter" in rows[0]
        else rows[0]
    )
    assert float(row["value"]) == 1.02
    assert json.loads(row["minos_bounds"]) == pytest.approx([0.99, 1.10])
    assert json.loads(row["interval_flags"])["at_upper_limit"]
    assert result["missing_product_count"] > 0
    recovered = next(
        item
        for item in _table(destination / "ensemble_recovery.csv")
        if item["route"] == review.route_key(route) and item["parameter"] == "qiso"
    )
    assert int(recovered["available"]) == 1
    expected = run["config"]["analysis"]["workloads"][run["metadata"]["workload"]]["mock_ids"]
    assert json.loads(recovered["missing_mock_ids"]) == [mock for mock in expected if mock != 0]
    coverage = [
        item
        for item in _table(destination / "interval_coverage.csv")
        if item["route"] == review.route_key(route) and item["parameter"] == "qiso"
    ]
    assert [item["subset"] for item in coverage] == ["all_finite", "valid_non_boundary"]
    assert int(coverage[0]["numerator"]) == int(coverage[0]["denominator"]) == 1
    assert float(coverage[0]["wilson_lower"]) < 1
    assert float(coverage[0]["wilson_upper"]) == 1
    assert coverage[1]["fraction"] == "" and int(coverage[1]["denominator"]) == 0
    assert "interval_coverage.csv" in (destination / "report.html").read_text()
    assert recovered["mean_offset_from_truth"] == ""
    assert any(destination.glob("*.png")) and any(destination.glob("*.pdf"))
    assert "conditional on the shared radial template" in (destination / "report.html").read_text()
    (destination / "review.md").write_text("My scientific observations.\n")
    profiles.bestfit["qiso"][...] = 0.97
    profiles.save(str(directory / "profiles.npy"))
    review.render_review(run["root"])
    assert float(_table(destination / "profiles.csv")[0]["value"]) == 0.97
    assert (destination / "review.md").read_text() == "My scientific observations.\n"


def test_native_profiles_supply_complete_ensemble_bias_and_uncertainty(run):
    route = _route(run)
    for mock, value in enumerate([0.98, 1.0, 1.04]):
        directory = run["root"] / "fit" / "units" / (driver.route_name(route) + f"--mock-{mock}")
        _profile(directory, {**_identity(run, route), "realization_id": mock}, value=value)
    record, _ = review.collect_route(run, route)
    config = copy.deepcopy(run["config"])
    workload = run["metadata"]["workload"]
    config["analysis"]["workloads"][workload]["mock_ids"] = [0, 1, 2]
    kwargs = dict(config=config, workload=workload)
    result = review.ensemble_recovery_rows([record], **kwargs)[0]
    assert result["mean_offset_from_truth"] == pytest.approx(0.02 / 3)
    assert result["mean_offset_standard_error"] == pytest.approx(np.sqrt(0.0028) / 3)
    assert result["boundary_intervals"] == 3
    assert result["missing_mock_ids"] == []
    # Record ordering cannot alter the cohort or erase native interval flags.
    record["individual_fits"].reverse()
    assert review.ensemble_recovery_rows([record], **kwargs)[0] == result
    record["individual_fits"][0]["parameters"][0]["interval_flags"]["is_valid"] = False
    assert review.ensemble_recovery_rows([record], **kwargs)[0]["invalid_intervals"] == 1
    record["individual_fits"].pop()
    incomplete = review.ensemble_recovery_rows([record], **kwargs)[0]
    assert incomplete["mean_offset_from_truth"] is None
    record["individual_fits"].append(record["individual_fits"][0])
    with pytest.raises(ValueError, match="duplicate or unexpected"):
        review.ensemble_recovery_rows([record], **kwargs)


def test_native_coverage_counts_failures_and_matches_wilson_reference(run):
    from scipy.stats import binomtest

    config = copy.deepcopy(run["config"])
    workload = run["metadata"]["workload"]
    config["analysis"]["workloads"][workload]["mock_ids"] = list(range(6))
    scoped = {**run, "config": config}
    route = _route(run)
    cases = [(1.0, (0, 0.1)), (1.1, (-0.01, 0.02)), (1.0, (-0.02, 0.03))]
    cases.extend([(1.0, (-0.02, 0.03)), (1.0, (np.nan, np.nan))])
    for mock, (value, offsets) in enumerate(cases):
        directory = run["root"] / "fit" / "units" / (driver.route_name(route) + f"--mock-{mock}")
        profiles = _profile(
            directory,
            {**_identity(run, route), "realization_id": mock},
            value=value,
            offsets=offsets,
        )
        profiles.interval.attrs["qiso"]["at_upper_limit"] = mock == 2
        profiles.interval.attrs["qiso"]["is_valid"] = mock != 3
        profiles.save(str(directory / "profiles.npy"))
    record, _ = review.collect_route(scoped, route)
    kwargs = dict(config=config, workload=workload)
    rows = review.interval_coverage_rows([record], **kwargs)
    all_finite, valid = [item for item in rows if item["parameter"] == "qiso"]
    assert all_finite["required"] == 6 and not all_finite["complete_finite_cohort"]
    assert all_finite["missing_or_unreadable_fit_ids"] == [5]
    assert all_finite["unavailable_interval_ids"] == [4]
    assert all_finite["invalid_interval_ids"] == [3]
    assert all_finite["boundary_interval_ids"] == [2]
    assert all_finite["contains_truth_ids"] == [0, 2, 3]
    assert valid["mock_ids"] == [0, 1] and valid["contains_truth_ids"] == [0]
    for row, numerator, denominator in [(all_finite, 3, 4), (valid, 1, 2)]:
        expected = binomtest(numerator, denominator).proportion_ci(
            confidence_level=0.68, method="wilson"
        )
        assert row["numerator"] == numerator and row["denominator"] == denominator
        assert row["fraction"] == pytest.approx(numerator / denominator)
        assert row["wilson_lower"] == pytest.approx(expected.low)
        assert row["wilson_upper"] == pytest.approx(expected.high)
    missing_parameter = next(item for item in rows if item["parameter"] == "qap")
    assert missing_parameter["missing_parameters"] == 5
    assert missing_parameter["fraction"] is None
    assert missing_parameter["wilson_lower"] is None
    record["individual_fits"].reverse()
    assert review.interval_coverage_rows([record], **kwargs) == rows
    duplicate = copy.deepcopy(record)
    duplicate["individual_fits"].append(duplicate["individual_fits"][0])
    with pytest.raises(ValueError, match="unique integer"):
        review.interval_coverage_rows([duplicate], **kwargs)
    unexpected = copy.deepcopy(record)
    unexpected["individual_fits"][0]["realization_id"] = 99
    with pytest.raises(ValueError, match="unexpected"):
        review.interval_coverage_rows([unexpected], **kwargs)
    config["analysis"]["workloads"][workload]["mock_ids"] = [0, 0]
    with pytest.raises(ValueError, match="unique nonempty"):
        review.interval_coverage_rows([record], **kwargs)


@pytest.mark.parametrize("contains", [False, True])
def test_native_coverage_zero_and_all_success_keep_finite_uncertainty(run, contains):
    from scipy.stats import binomtest

    route = _route(run)
    config = copy.deepcopy(run["config"])
    workload = run["metadata"]["workload"]
    config["analysis"]["workloads"][workload]["mock_ids"] = [0, 1, 2]
    for mock in (0, 1, 2):
        directory = run["root"] / "fit" / "units" / (driver.route_name(route) + f"--mock-{mock}")
        _profile(
            directory,
            {**_identity(run, route), "realization_id": mock},
            value=1.0 if contains else 1.15,
            offsets=(-0.01, 0.01),
        )
    record, _ = review.collect_route({**run, "config": config}, route)
    rows = review.interval_coverage_rows([record], config=config, workload=workload)
    row = next(
        item for item in rows if item["parameter"] == "qiso" and item["subset"] == "all_finite"
    )
    expected = binomtest(3 if contains else 0, 3).proportion_ci(
        confidence_level=0.68, method="wilson"
    )
    assert row["complete_finite_cohort"]
    assert row["fraction"] == float(contains)
    assert row["wilson_lower"] == pytest.approx(expected.low, abs=1e-14)
    assert row["wilson_upper"] == pytest.approx(expected.high, abs=1e-14)
    assert row["wilson_upper"] > row["wilson_lower"]
    record["individual_fits"] = []
    empty = review.interval_coverage_rows([record], config=config, workload=workload)
    assert all(item["fraction"] is None and item["denominator"] == 0 for item in empty)
    del config["reporting"]["interval_coverage"]
    assert review.interval_coverage_rows([record], config=config, workload=workload) == []


TRANSFER_OFFSETS = {"self--R25": [0, 0, 0], "sigma_0.003--R25": [-0.02, 0, 0.04]}
TRANSFER_SOURCE_KEYS = ("run_name", "code_commit", "analysis_config_sha256")


def _transfer_fits(item, route, offsets):
    """Three mock profiles under this run's fit/units, identified by its own metadata."""
    for mock, offset in enumerate(offsets):
        path = item["root"] / "fit" / "units" / (driver.route_name(route) + f"--mock-{mock}")
        _profile(
            path,
            {**_identity(item, route), "realization_id": mock},
            value=1 + offset,
            parameter_names=("qiso", "qap"),
        )


def _transfer_measurement_runs(tmp_path, *, fits):
    """Two native pilot transfer runs (self reference, high-precision source) sharing targets.

    ``fits`` places the mock profiles in the measurement runs themselves; the campaign fits
    them into separate inference runs instead (``transfer_inference_pair``).
    """
    from pzbao.catalog import catalogue_identity, save_catalogue
    from pzbao.config import analysis_config_sha256, load_analysis_config
    from pzbao.config.transfer import transfer_run_configs

    configs = transfer_run_configs(load_analysis_config(), pilot=True)
    runs = []
    route = {
        "tracer": "BGS_BRIGHT-21.5",
        "measurement_bin": "BGS1",
        "condition": "sigma_0.01",
        "reconstruction_state": "postreconstruction",
    }
    for name, offsets in TRANSFER_OFFSETS.items():
        config = configs[name]
        item = {
            "root": tmp_path / name,
            "config": config,
            "metadata": {
                "run_name": name,
                "code_commit": "a" * 40,
                "interface_version": 2,
                "workload": "transfer",
                "analysis_config_sha256": analysis_config_sha256(config),
            },
        }
        directory = item["root"] / "select" / "units" / route["tracer"]
        directory.mkdir(parents=True)
        preparation = {
            "tracer": route["tracer"],
            "realization_ids": config["analysis"]["mock_ids"],
            "catalogues": [],
        }
        if fits:
            _transfer_fits(item, route, offsets)
        for mock in range(len(offsets)):
            for cap in ("NGC", "SGC"):
                for role in ("data", "random"):
                    identity = catalogue_identity(
                        item,
                        tracer=route["tracer"],
                        condition=route["condition"],
                        mock_id=mock,
                        cap=cap,
                        role=role,
                        stage="select",
                    )
                    filename = f"mock-{mock}--{cap}--{role}.npz"
                    catalogue = {
                        "SOURCE_ID": np.array([11, 23], dtype=np.uint64),
                        "X": np.array([10.0, 20.0]),
                        "Y": np.array([2.0, 3.0]),
                        "Z_CART": np.array([50.0, 60.0]),
                        "Z_OBS": np.array([0.15, 0.17]),
                        "Z_RSD": np.array([0.145, 0.174]),
                        "WEIGHT": np.array([1.0, 2.0]),
                    }
                    save_catalogue(directory, filename, catalogue, identity=identity)
                    preparation["catalogues"].append({"path": filename, "identity": identity})
        (directory / "preparation.json").write_text(json.dumps(preparation))
        driver._write_yaml(
            directory / "result.yaml",
            {
                **{
                    key: item["metadata"][key]
                    for key in ("run_name", "code_commit", "analysis_config_sha256")
                },
                "stage": "select",
                "unit": {
                    "unit_id": route["tracer"],
                    "tracer": route["tracer"],
                    "realization_ids": config["analysis"]["mock_ids"],
                },
            },
        )
        runs.append(item)
    return runs, route


@pytest.fixture
def transfer_pair(tmp_path):
    (reference, variant), route = _transfer_measurement_runs(tmp_path, fits=True)
    record, _ = review.collect_route(variant, route)
    return reference, variant, record


@pytest.fixture
def transfer_inference_pair(tmp_path):
    """The campaign layout: fit-only transfer inference runs over fit-less measurement runs.

    Each inference run adopts its own reviewed centres, records its measurement source and
    holds the three mock profiles; the measurement runs hold targets and randoms, no fits.
    """
    from pzbao.config import analysis_config_sha256

    sources, route = _transfer_measurement_runs(tmp_path, fits=False)
    runs = []
    for source, centre in zip(sources, (6.0, 5.5)):
        config = copy.deepcopy(source["config"])
        config["measurement"]["reconstruction"]["transfer_run"]["damping_centres"] = {
            "fixture_centre_mpc_h": centre
        }
        name = source["metadata"]["run_name"] + "--reference"
        item = {
            "root": tmp_path / name,
            "config": config,
            "metadata": {
                **source["metadata"],
                "run_name": name,
                "analysis_config_sha256": analysis_config_sha256(config),
                "inference_definition": "transfer_reference",
                "source_run": {
                    "root": str(source["root"]),
                    **{key: source["metadata"][key] for key in TRANSFER_SOURCE_KEYS},
                },
            },
            "source_run": source,
        }
        _transfer_fits(item, route, TRANSFER_OFFSETS[source["metadata"]["run_name"]])
        runs.append(item)
    reference, variant = runs
    record, _ = review.collect_route(variant, route)
    return reference, variant, record


def test_transfer_comparison_reads_native_targets_profiles_and_paired_uncertainty(transfer_pair):
    reference, variant, record = transfer_pair
    rows, summaries = review.transfer_comparison_rows(variant, reference, [record])
    assert len(rows) == 6 and len(summaries) == 2
    assert [
        row["variant_minus_self"] for row in rows if row["parameter"] == "qiso"
    ] == pytest.approx([-0.02, 0, 0.04])
    assert all(row["target_catalogues_equal"] for row in rows)
    assert all(row["variant_interval_flags"]["at_upper_limit"] for row in rows)
    assert summaries[0]["mean_variant_minus_self"] == pytest.approx(0.02 / 3)
    assert summaries[0]["paired_mean_standard_error"] == pytest.approx(np.sqrt(0.0028) / 3)
    record["individual_fits"].reverse()
    assert review.transfer_comparison_rows(variant, reference, [record]) == (rows, summaries)
    record["individual_fits"].pop()
    rows, summaries = review.transfer_comparison_rows(variant, reference, [record])
    assert summaries[0]["available_pairs"] == 2
    assert summaries[0]["missing_pair_ids"] == [0]
    assert summaries[0]["mean_variant_minus_self"] is None
    assert rows[0]["self_value"] == 1 and rows[0]["variant_value"] is None


@pytest.mark.parametrize("pair", ["transfer_pair", "transfer_inference_pair"])
def test_transfer_comparison_reaches_native_report_tables(request, monkeypatch, pair):
    reference, variant, record = request.getfixturevalue(pair)
    monkeypatch.setattr(
        driver,
        "load_run",
        lambda path: {item["root"]: item for item in (reference, variant)}[Path(path)],
    )
    native_stage_units = driver.stage_units

    def report_units(stage, *args, **kwargs):
        if stage == "sample":
            return [record]
        return native_stage_units(stage, *args, **kwargs)

    monkeypatch.setattr(driver, "stage_units", report_units)
    review.render_review(variant["root"], reference_root=reference["root"])
    destination = variant["root"] / "review"
    assert len(_table(destination / "transfer_pairs.csv")) == 6
    summary = _table(destination / "transfer_summary.csv")[0]
    assert float(summary["paired_mean_standard_error"]) == pytest.approx(np.sqrt(0.0028) / 3)
    assert summary["reference_run"] == str(reference["root"])
    assert "transfer_summary.csv" in (destination / "report.html").read_text()
    review.render_review(variant["root"])
    assert _table(destination / "transfer_summary.csv") == []


@pytest.mark.parametrize(
    "column,role", [("SOURCE_ID", "data"), ("Z_OBS", "data"), ("WEIGHT", "random")]
)
def test_transfer_comparison_rejects_changed_native_targets(transfer_pair, column, role):
    from pzbao.catalog import save_catalogue

    reference, variant, record = transfer_pair
    path = variant["root"] / "select" / "units" / record["tracer"] / f"mock-0--NGC--{role}.npz"
    identity = json.loads(path.with_suffix(".json").read_text())["identity"]
    with np.load(path, allow_pickle=False) as arrays:
        catalogue = {key: arrays[key].copy() for key in arrays.files}
    catalogue[column][0] += 1
    save_catalogue(path.parent, path.name, catalogue, identity=identity)
    with pytest.raises(ValueError, match="differs from self"):
        review.transfer_comparison_rows(variant, reference, [record])


def test_transfer_comparison_rejects_selection_or_source_provenance_changes(transfer_pair):
    reference, variant, record = transfer_pair
    changed = copy.deepcopy(variant)
    changed["config"]["selection"]["stable_randomness"]["seed"] += 1
    with pytest.raises(ValueError, match="changed settings"):
        review.transfer_comparison_rows(changed, reference, [record])
    changed = copy.deepcopy(variant)
    changed["metadata"]["code_state_sha256"] = "different"
    with pytest.raises(ValueError, match="same execution source"):
        review.transfer_comparison_rows(changed, reference, [record])
    path = variant["root"] / "select" / "units" / record["tracer"] / "mock-0--NGC--data.json"
    metadata = json.loads(path.read_text())
    metadata["identity"]["run_name"] = "another-run"
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="different stage, unit or snapshot"):
        review.transfer_comparison_rows(variant, reference, [record])


def _route_of(record):
    return {
        key: record[key]
        for key in ("tracer", "measurement_bin", "condition", "reconstruction_state")
    }


def test_transfer_comparison_pairs_fits_held_by_inference_runs(transfer_inference_pair):
    reference, variant, record = transfer_inference_pair
    route = _route_of(record)
    # The campaign's measurement runs hold no fits; only their inference runs do.
    for item in (reference, variant):
        assert review.collect_route(item["source_run"], route)[0]["individual_fits"] == []
    assert len(record["individual_fits"]) == 3
    rows, summaries = review.transfer_comparison_rows(variant, reference, [record])
    assert len(rows) == 6 and len(summaries) == 2
    assert {(row["tracer"], row["measurement_bin"], row["condition"]) for row in rows} == {
        ("BGS_BRIGHT-21.5", "BGS1", "sigma_0.01")
    }
    assert [item["parameter"] for item in summaries] == ["qiso", "qap"]
    assert [
        row["variant_minus_self"] for row in rows if row["parameter"] == "qiso"
    ] == pytest.approx([-0.02, 0, 0.04])
    assert all(row["target_catalogues_equal"] for row in rows)
    assert summaries[0]["mean_variant_minus_self"] == pytest.approx(0.02 / 3)
    assert summaries[0]["paired_mean_standard_error"] == pytest.approx(np.sqrt(0.0028) / 3)
    for item in rows + summaries:
        assert item["reference_run"] == str(reference["root"])
        assert item["variant_run"] == str(variant["root"])
        assert item["reference_measurement_run"] == str(reference["source_run"]["root"])
        assert item["variant_measurement_run"] == str(variant["source_run"]["root"])


def test_transfer_comparison_names_native_runs_as_their_own_measurement(transfer_pair):
    reference, variant, record = transfer_pair
    rows, summaries = review.transfer_comparison_rows(variant, reference, [record])
    for item in rows + summaries:
        assert item["reference_measurement_run"] == item["reference_run"] == str(reference["root"])
        assert item["variant_measurement_run"] == item["variant_run"] == str(variant["root"])


@pytest.mark.parametrize("key", [*TRANSFER_SOURCE_KEYS, "root"])
def test_transfer_comparison_refuses_an_inference_run_whose_source_identity_changed(
    transfer_inference_pair, key
):
    reference, variant, record = transfer_inference_pair
    changed = copy.deepcopy(variant)
    changed["metadata"]["source_run"][key] = (
        str(reference["source_run"]["root"]) if key == "root" else "other-" + key
    )
    with pytest.raises(ValueError, match=rf"comparison run .* source run identity .*\['{key}'\]"):
        review.transfer_comparison_rows(changed, reference, [record])
    changed = copy.deepcopy(variant)
    changed["config"]["selection"]["stable_randomness"]["seed"] += 1
    with pytest.raises(ValueError, match="transfer inference changed more than"):
        review.transfer_comparison_rows(changed, reference, [record])


def test_transfer_comparison_refuses_mixing_inference_and_measurement_runs(
    transfer_inference_pair,
):
    reference, variant, record = transfer_inference_pair
    with pytest.raises(ValueError, match="same inference definition.*'transfer_reference'.*None"):
        review.transfer_comparison_rows(variant, reference["source_run"], [record])
    with pytest.raises(ValueError, match="same inference definition.*None.*'transfer_reference'"):
        review.transfer_comparison_rows(variant["source_run"], reference, [record])


def test_transfer_comparison_reads_targets_from_the_measurement_runs(transfer_inference_pair):
    from pzbao.catalog import save_catalogue

    reference, variant, record = transfer_inference_pair
    directory = variant["source_run"]["root"] / "select" / "units" / record["tracer"]
    path = directory / "mock-1--SGC--random.npz"
    identity = json.loads(path.with_suffix(".json").read_text())["identity"]
    with np.load(path, allow_pickle=False) as arrays:
        catalogue = {key: arrays[key].copy() for key in arrays.files}
    catalogue["WEIGHT"][1] += 1
    save_catalogue(path.parent, path.name, catalogue, identity=identity)
    with pytest.raises(ValueError, match="random differs from self"):
        review.transfer_comparison_rows(variant, reference, [record])


def test_figure_transfer_comparison_resolves_fits_from_the_inference_runs(
    transfer_inference_pair, monkeypatch
):
    from pzbao.figures import data

    reference, variant, record = transfer_inference_pair
    route = _route_of(record)
    monkeypatch.setattr(
        driver,
        "load_run",
        lambda path: {item["root"]: item for item in (reference, variant)}[Path(path)],
    )
    monkeypatch.setattr(
        data,
        "routes_for_run",
        lambda run: [
            {**route, "reconstruction_state": state}
            for state in ("prereconstruction", "postreconstruction")
        ],
    )
    assert data.transfer_comparison(
        variant["root"], reference["root"]
    ) == review.transfer_comparison_rows(variant, reference, [record])


def test_native_ensemble_changes_reach_posterior_tables_and_all_contributions(run):
    route = _route(run)
    directory = run["root"] / "sample" / "units" / driver.route_name(route)
    values = []
    for index in range(4):
        _, draws = _chain(
            directory, {**_identity(run, route), "ensemble_index": index}, mean=1 + index * 0.001
        )
        values.append(draws[200:].ravel())
    summary = {
        "identity": _identity(run, route),
        "complete_ensemble_collection": True,
        "ensembles": [
            {"ensemble_index": index, "path": f"ensemble-{index}.npy"} for index in range(4)
        ],
        "gr_history": [],
        "stop_reason": "stopping_rule_reached",
    }
    (directory / "sampling.json").write_text(json.dumps(summary))
    record, rows = review.collect_route(run, route)
    expected = np.concatenate(values)
    assert rows["posterior"][0]["mean"] == pytest.approx(expected.mean())
    assert rows["posterior"][0]["standard_deviation"] == pytest.approx(expected.std(ddof=1))
    assert rows["posterior"][0]["ensemble_contributions"] == [8000] * 4
    assert len(record["ensembles"]) == 4
    _, changed = _chain(directory, {**_identity(run, route), "ensemble_index": 2}, mean=1.2)
    _, rows = review.collect_route(run, route)
    values[2] = changed[200:].ravel()
    assert rows["posterior"][0]["mean"] == pytest.approx(np.concatenate(values).mean())


def test_interrupted_sampler_keeps_walkers_visible_without_a_pooled_posterior(run):
    route = _route(run)
    unit = next(
        unit
        for unit in driver.stage_units("sample", run["config"], "pilot")
        if driver.route_name(unit) == driver.route_name(route)
    )
    directory = run["root"] / "sample" / "attempts" / (driver.route_name(route) + ".interrupted")
    _chain(directory, {**_identity(run, route), "ensemble_index": 0}, constant=True)
    (directory / "attempt.yaml").write_text(
        yaml.safe_dump(
            {"unit": unit, "analysis_config_sha256": run["metadata"]["analysis_config_sha256"]}
        )
    )
    record, rows = review.collect_route(run, route)
    assert len(record["ensembles"]) == 1
    assert "partial sampling: 1/4" in record["coverage"]["sampling"]
    assert not record["posterior_draws"] and not rows["posterior"]
    assert rows["sampler"][0]["tau"] is None
    assert "constant" in rows["sampler"][0]["unavailable_reason"]
    paths = review._plot_sampler(record, run["root"] / "traces.png")
    assert all(Path(path).is_file() for path in paths.values())
    paths = review._plot_sampler_diagnostics(record, run["root"] / "mixing.png")
    assert all(Path(path).is_file() for path in paths.values())


def test_paired_ids_and_literal_gain_never_invert_or_impute():
    assert review.uncertainty_gain(0.04, 0.02) == 0.5
    pairs = review.paired_realizations(
        [{"realization_id": 2}],
        [{"realization_id": 0}, {"realization_id": 2}],
        expected_ids=[0, 1, 2],
    )
    assert [row["realization_id"] for row in pairs] == [0, 1, 2]
    assert [row["missing_pair"] for row in pairs] == ["pre missing", "pre and post missing", None]
    with pytest.raises(ValueError, match="unique integer"):
        review.paired_realizations([{"realization_id": True}], [])


def test_gain_uses_standard_deviation_not_quantile_width_or_estimate_scatter(run):
    route = _route(run)
    pre, post = np.array([0.97, 0.98, 0.99, 1.06]), np.array([0.985, 0.995, 1.005, 1.015])
    routes = [
        {**route, "posterior_draws": {"qiso": pre}},
        {**route, "reconstruction_state": "postreconstruction", "posterior_draws": {"qiso": post}},
    ]
    row = review.aggregate_gain_rows(routes, config=run["config"])[0]
    assert row["fom_definition"] == "posterior_standard_deviation"
    assert row["pre_fom"] == pytest.approx(np.sqrt(0.005 / 3))
    assert row["post_fom"] == pytest.approx(np.sqrt(0.0005 / 3))
    assert row["gain"] == pytest.approx(np.sqrt(0.1))
    assert row["prior_standard_deviation"] == pytest.approx(0.4 / np.sqrt(12))
    assert row["pre_to_prior_standard_deviation"] == pytest.approx(
        np.sqrt(0.005 / 3) / (0.4 / np.sqrt(12))
    )
    quantile_ratio = (
        np.diff(np.quantile(post, [0.16, 0.84])).item()
        / np.diff(np.quantile(pre, [0.16, 0.84])).item()
    )
    assert not np.isclose(row["gain"], quantile_ratio)
    # Immutable snapshots without either choice remain explicitly unavailable.
    config = copy.deepcopy(run["config"])
    config["reporting"]["gain"]["posterior_uncertainty"] = None
    row = review.aggregate_gain_rows(routes, config=config)[0]
    assert row["gain"] is None and "not selected" in row["gain_reason"]


@pytest.mark.parametrize("offsets", [None, [0.0, 0.0], [0.01, 0.03], [-0.01, np.nan], [-0.01]])
def test_profile_gain_keeps_invalid_and_missing_pairs_with_reasons(run, offsets):
    route = _route(run)
    parameter = {
        "basename": "qiso",
        "fixed": False,
        "value": 1.0,
        "minos_bounds": None,
        "minos_offsets": offsets,
    }
    routes = [
        {**route, "individual_fits": [{"realization_id": 0, "parameters": [parameter]}]},
        {
            **route,
            "reconstruction_state": "postreconstruction",
            "individual_fits": [
                {"realization_id": 0, "parameters": [{**parameter, "minos_offsets": [-0.02, 0.02]}]}
            ],
        },
    ]
    rows = review.paired_fit_rows(routes, config=run["config"], workload="pilot")
    assert [(row["realization_id"], row["parameter"]) for row in rows] == [
        (identifier, parameter) for identifier in range(25) for parameter in ("qiso", "qap")
    ]
    # The fixture supplies only qiso. Missing AP intervals must remain explicit.
    ap_rows = [row for row in rows if row["parameter"] == "qap"]
    assert all(row["pre_fom"] is None and row["post_fom"] is None for row in ap_rows)
    assert all(row["gain"] is None and "unavailable" in row["gain_reason"] for row in ap_rows)
    rows = [row for row in rows if row["parameter"] == "qiso"]
    assert [row["realization_id"] for row in rows] == list(range(25))
    assert rows[0]["gain"] is None and rows[0]["gain_reason"].startswith("pre:")
    assert rows[0]["post_fom"] == 0.02
    assert rows[1]["missing_pair"] == "pre and post missing"
    assert rows[1]["gain"] is None and "unavailable" in rows[1]["gain_reason"]


def test_condition_comparison_joins_native_profiles_by_id_and_keeps_missing_intervals(run):
    config = copy.deepcopy(run["config"])
    config["analysis"]["workloads"]["pilot"]["sigma_z_tiers"] = ["sigma_0.001", "sigma_0.003"]
    route = _route(run)
    records = []
    for condition, identifier, value, offsets in (
        ("sigma_0.001", 3, 1.03, (-0.02, 0.06)),
        ("sigma_0.001", 0, 1.00, (-0.02, 0.06)),
        ("sigma_0.003", 0, 1.02, (-0.01, 0.03)),
        ("sigma_0.003", 3, 1.01, (-0.01, 0.03)),
    ):
        selected = {**route, "condition": condition}
        directory = (
            run["root"] / "fit" / "units" / (driver.route_name(selected) + f"--mock-{identifier}")
        )
        _profile(
            directory,
            {**_identity(run, selected), "realization_id": identifier},
            value=value,
            offsets=offsets,
        )
    for condition in config["analysis"]["workloads"]["pilot"]["sigma_z_tiers"]:
        selected = {**route, "condition": condition}
        record, _ = review.collect_route(run, selected)
        record["individual_fits"].reverse()  # Pairing must never depend on input order.
        record["density"] = [{"cap": "NGC", "realization_id": 0, "attained_density_h3_mpc3": 0.001}]
        records.append(record)
    all_rows = review.paired_condition_rows(records, config=config, workload="pilot")
    assert [(row["realization_id"], row["parameter"]) for row in all_rows] == [
        (identifier, parameter) for identifier in range(25) for parameter in ("qiso", "qap")
    ]
    ap_rows = [row for row in all_rows if row["parameter"] == "qap"]
    assert all(row["right_minus_left"] is None for row in ap_rows)
    assert all("unavailable" in row["interval_comparison_reason"] for row in ap_rows)
    rows = [row for row in all_rows if row["parameter"] == "qiso"]
    assert [row["realization_id"] for row in rows] == list(range(25))
    assert rows[0]["right_minus_left"] == pytest.approx(0.02)
    assert rows[3]["right_minus_left"] == pytest.approx(-0.02)
    assert rows[0]["right_over_left_interval_width"] == pytest.approx(0.5)
    assert rows[0]["left_minos_bounds"] == pytest.approx([0.98, 1.06])
    assert rows[0]["left_interval_flags"]["at_upper_limit"]
    assert rows[0]["left_density_by_cap"]["NGC"]["attained_density_h3_mpc3"] == 0.001
    assert rows[1]["right_minus_left"] is None
    assert "unavailable" in rows[1]["interval_comparison_reason"]
    destination = run["root"] / "review" / "condition_comparisons.csv"
    destination.parent.mkdir()
    review._write_csv(destination, all_rows)
    saved = _table(destination)
    assert len(saved) == 50
    assert sum(row["parameter"] == "qap" for row in saved) == 25
    saved = [row for row in saved if row["parameter"] == "qiso"]
    assert float(saved[3]["right_minus_left"]) == pytest.approx(-0.02)
    assert json.loads(saved[0]["right_minos_bounds"]) == pytest.approx([1.01, 1.05])
    # Native identity checks must refuse a profile rebound to the other condition.
    selected = {**route, "condition": "sigma_0.003"}
    directory = run["root"] / "fit" / "units" / (driver.route_name(selected) + "--mock-0")
    metadata = json.loads((directory / "fit.json").read_text())
    metadata["identity"]["condition"] = "sigma_0.001"
    (directory / "fit.json").write_text(json.dumps(metadata))
    rejected, _ = review.collect_route(run, selected)
    rows = review.paired_condition_rows([records[0], rejected], config=config, workload="pilot")
    assert rows[0]["right_minus_left"] is None
    assert "right: native Minos interval is unavailable" in rows[0]["interval_comparison_reason"]
    with pytest.raises(ValueError, match="unique configured routes"):
        review.paired_condition_rows([records[0], records[0]], config=config, workload="pilot")


def test_native_profiles_and_four_ensembles_reach_distinct_gain_csv_tables(run):
    route = _route(run)
    retained = []
    profiles = []
    for state, spread, offsets in (
        ("prereconstruction", 0.04, (-0.03, 0.09)),
        ("postreconstruction", 0.02, (-0.01, 0.03)),
    ):
        current = {**route, "reconstruction_state": state}
        identity = _identity(run, current)
        profile_dir = run["root"] / "fit" / "units" / (driver.route_name(current) + "--mock-0")
        profiles.append(_profile(profile_dir, {**identity, "realization_id": 0}, offsets=offsets))
        sample_dir = run["root"] / "sample" / "units" / driver.route_name(current)
        draws = []
        for index in range(4):
            _, values = _chain(sample_dir, {**identity, "ensemble_index": index}, spread=spread)
            draws.append(values[200:].ravel())
        retained.append(np.concatenate(draws))
        (sample_dir / "sampling.json").write_text(
            json.dumps(
                {
                    "identity": identity,
                    "complete_ensemble_collection": True,
                    "ensembles": [
                        {"ensemble_index": index, "path": f"ensemble-{index}.npy"}
                        for index in range(4)
                    ],
                    "gr_history": [],
                    "stop_reason": "stopping_rule_reached",
                }
            )
        )
    review.render_review(run["root"])
    destination = run["root"] / "review"
    aggregate = _table(destination / "aggregate_gains.csv")[0]
    assert aggregate["fom_definition"] == "posterior_standard_deviation"
    assert float(aggregate["pre_fom"]) == pytest.approx(retained[0].std(ddof=1))
    assert float(aggregate["post_fom"]) == pytest.approx(retained[1].std(ddof=1))
    assert float(aggregate["gain"]) == pytest.approx(0.5)
    paired = next(
        row
        for row in _table(destination / "paired_fits.csv")
        if row["measurement_bin"] == route["measurement_bin"] and row["realization_id"] == "0"
    )
    assert paired["fit_scope"] == "individual profile"
    assert paired["fom_definition"] == "minos_half_width"
    assert float(paired["pre_fom"]) == pytest.approx(0.06)
    assert float(paired["post_fom"]) == pytest.approx(0.02)
    assert float(paired["gain"]) == pytest.approx(1 / 3)
    assert json.loads(paired["pre_minos_bounds"]) == pytest.approx([0.99, 1.11])
    assert json.loads(paired["post_interval_flags"])["at_upper_limit"]
    assert json.loads(paired["pre_detection"])["delta_chi2"] == 2.0
    # Numeric widths remain reported with their native invalidity flags; they are not filtered.
    profiles[1].interval.attrs["qiso"]["is_valid"] = False
    profiles[1].save(str(profile_dir / "profiles.npy"))
    record, _ = review.collect_route(run, current)
    pre_record, _ = review.collect_route(run, route)
    paired = review.paired_fit_rows([pre_record, record], config=run["config"], workload="pilot")[0]
    assert paired["gain"] == pytest.approx(1 / 3)
    assert paired["post_interval_flags"]["is_valid"] is False


@pytest.mark.parametrize("invalid_count", [None, True, 25.5, 0])
def test_mean_residual_uses_mean_covariance_and_requires_declared_count(invalid_count):
    route = {
        "measurement_bin": "BGS1",
        "condition": "sigma_0.003",
        "reconstruction_state": "prereconstruction",
        "covariance": np.diag([4.0, 9.0]).tolist(),
        "covariance_realization_count": 25,
    }
    sigma = review.route_mean_sigma(route)
    np.testing.assert_allclose(sigma, [2 / 5, 3 / 5])
    route["covariance_realization_count"] = invalid_count
    with pytest.raises(review.ReviewPackageIncomplete, match="realizations"):
        review.route_mean_sigma(route)


def test_damping_diagnostics_keep_width_and_bound_meanings_separate_from_gain(run):
    route = {
        "measurement_bin": "BGS1",
        "condition": "sigma_0.003",
        "reconstruction_state": "postreconstruction",
        "damping_centres_mpc_h": {"sigma_perp": 4.0},
        "damping_widths_mpc_h": {"sigma_perp": 1.0},
        "damping_bounds_mpc_h": {"sigma_perp": [0.0, None]},
        "damping_posterior_mpc_h": {"sigma_perp": np.array([3.0, 4.0, 5.0])},
    }
    row = review.damping_posterior_diagnostics([route], config=run["config"])[0]
    assert row["displacement_in_prior_widths"] == 0
    assert row["posterior_to_prior_width_ratio"] == 1
    assert row["bound_distance_in_prior_widths"] == 4
    assert row["posterior_fraction_near_bound"] == 0
    assert row["bound_proximity_fraction"] == 0.01
    assert row["decides_nothing"] is True


def test_full_grid_saved_native_products_render_all_routes_and_local_links(tmp_path, monkeypatch):
    """Exercise full-grid reporting without running a full-grid scientific campaign.

    Upstream covariance/measurement reader fixtures supply small numerical records. Preparation,
    parent diagnostics, all native profiles/chains, P(k) arrays and mean-fit curves are saved and
    read normally. The native pilot separately tests the upstream producers and their readers.
    """
    pytest.importorskip("desilike")
    from pzbao import covariance
    import pzbao.clustering as pipeline_clustering

    monkeypatch.setattr(
        driver, "_code_identity", lambda: {"code_commit": "a" * 40, "repository": str(tmp_path)}
    )
    run = driver.create_run(
        tmp_path, "full-grid-report", workload="full_grid", catalog_root=tmp_path
    )
    config = run["config"]
    routes = driver.routes_for(config, "full_grid")
    identifiers = config["analysis"]["workloads"]["full_grid"]["mock_ids"]
    base_identity = {
        key: run["metadata"][key] for key in ("run_name", "code_commit", "analysis_config_sha256")
    }

    def parent_product(stage, unit, filename, record):
        directory = run["root"] / stage / "units" / unit["unit_id"]
        directory.mkdir(parents=True)
        driver._write_yaml(
            directory / "result.yaml", {**base_identity, "stage": stage, "unit": unit}
        )
        (directory / filename).write_text(json.dumps(record))

    for unit in driver.stage_units("select", config, "full_grid"):
        conditions = {}
        for condition in sorted({route["condition"] for route in routes}):
            conditions[condition] = {
                cap: {
                    "radial": {
                        "z_edges": [0.1, 0.2, 0.3],
                        "mean_counts": [2.0, 3.0],
                        "target_probability_mass": [0.4, 0.6],
                    },
                    "boundary_counts_by_mock": [
                        {
                            "realization_id": identifier,
                            "z_edges": [0.1, 0.2, 0.3],
                            "original_clustering_counts": [2.0, 3.0],
                            "available_observed_counts": [3.0, 4.0],
                            "reservoir_observed_counts": [1.0, 1.0],
                        }
                        for identifier in identifiers
                    ],
                }
                for cap in ("NGC", "SGC")
            }
        parent_product(
            "select",
            unit,
            "preparation.json",
            {
                "tracer": unit["tracer"],
                "realization_ids": identifiers,
                "conditions": conditions,
                **per_bin_preparation_fields(config, unit["tracer"]),
            },
        )
    for unit in driver.stage_units("reconstruct", config, "full_grid"):
        bins = {route["measurement_bin"] for route in routes if route["tracer"] == unit["tracer"]}
        diagnostic = {
            "reconstruction": {
                "shell_diagnostics": {
                    "shells": [
                        {
                            "shell": 0,
                            "radial_range_mpc_h": [300.0, 600.0],
                            "retained_supported_cell_fraction": 0.9,
                            "displacement_rms_mpc_h": 2.0,
                        }
                    ]
                }
            },
            "selection": {
                "selection": {
                    "bins": {
                        name: {
                            "attained_nbar_h3_mpc3": 0.001,
                            "maximum_attainable_nbar_h3_mpc3": 0.002,
                            "requested_nbar_h3_mpc3": 0.0015,
                        }
                        for name in bins
                    }
                }
            },
        }
        parent_product(
            "reconstruct",
            unit,
            "reconstruction.json",
            {"diagnostics": {cap: diagnostic for cap in ("NGC", "SGC")}},
        )

    sources = {}
    expected_parameters = {}
    low, high = config["fitting"]["fit_edge_limits_mpc_h"]
    width = config["fitting"]["input_bin_width_mpc_h"]
    edges = np.arange(low, high + 0.5 * width, width)
    centers = (edges[:-1] + edges[1:]) / 2
    for route in routes:
        name, key = driver.route_name(route), review.route_key(route)
        surface = config["fitting"]["fit_surfaces"][route["tracer"]]
        names = config["fitting"]["ap_interval_parameters"][surface["ap_mode"]]
        expected_parameters[key] = names
        multipoles = surface["observed_multipoles"]
        dimension = len(centers) * len(multipoles)
        model = np.tile(0.001 * np.cos(centers / 20), len(multipoles))
        vectors = model + np.arange(len(identifiers))[:, None] * 1.0e-6
        matrix = np.eye(dimension) * 1.0e-4
        arrays = {
            "covariance": matrix,
            "vectors": vectors,
            "separations": np.tile(centers, (len(identifiers), 1)),
            "multipoles": np.array(multipoles),
            "realization_ids": np.array(identifiers),
        }
        for cap, weight in (("NGC", 0.8), ("SGC", 0.2)):
            arrays.update(
                {
                    "weights_" + cap: np.full((len(centers), 4), weight),
                    "W_" + cap: np.eye(dimension, 3 * len(centers)) * weight,
                    "covariance_" + cap: matrix * weight,
                    "covariance_subsamples_"
                    + cap: np.array([matrix * weight * 0.9, matrix * weight * 1.1]),
                }
            )
        measurements = []
        if (
            route["measurement_bin"]
            in config["measurement"]["power_spectrum_secondary"]["diagnostic_measurement_bins"]
        ):
            directory = run["root"] / "reconstruct" / "synthetic-power" / name
            directory.mkdir(parents=True)
            k = np.linspace(0.02, 0.3, 8)
            ells = config["measurement"]["power_spectrum_secondary"]["multipoles"]
            for identifier in identifiers:
                filename = f"power-{identifier}.npz"
                np.savez(
                    directory / filename,
                    k=k,
                    power=np.tile(1000 * np.exp(-k) + identifier, (len(ells), 1)),
                    ells=ells,
                )
                measurements.append(
                    {
                        "identity": {"cap": "NGC+SGC", "realization_id": identifier},
                        "directory": directory,
                        "power": filename,
                    }
                )
        sources[key] = (arrays, measurements)
        post = route["reconstruction_state"] == "postreconstruction"
        for identifier in [*identifiers, None]:
            suffix = "mean" if identifier is None else f"mock-{identifier}"
            directory = run["root"] / "fit" / "units" / (name + "--" + suffix)
            _profile(
                directory,
                {**_identity(run, route), "realization_id": identifier},
                parameter_names=names,
                offsets=(-0.01, 0.03) if post else (-0.03, 0.09),
            )
            if identifier is None:
                metadata = json.loads((directory / "fit.json").read_text())
                metadata["damping_prior"] = {
                    "centers_mpc_h": {"sigma_perp": 4.0},
                    "widths_mpc_h": {"sigma_perp": 1.0},
                    "bounds_mpc_h": {"sigma_perp": [0.0, None]},
                }
                (directory / "fit.json").write_text(json.dumps(metadata))
                data = vectors.mean(axis=0)
                np.savez(
                    directory / "fit_curve.npz",
                    fit_edges=edges,
                    multipoles=multipoles,
                    data=data,
                    model=model,
                    residual=data - model,
                    standard_deviation=np.sqrt(np.diag(matrix) / len(identifiers)),
                )
        directory = run["root"] / "sample" / "units" / name
        for index in range(config["fitting"]["sampling"]["ensembles"]):
            _chain(
                directory,
                {**_identity(run, route), "ensemble_index": index},
                parameter_names=names,
                spread=0.02 if post else 0.04,
                steps=40,
            )
        (directory / "sampling.json").write_text(
            json.dumps(
                {
                    "identity": _identity(run, route),
                    "complete_ensemble_collection": True,
                    "ensembles": [
                        {"ensemble_index": index, "path": f"ensemble-{index}.npy"}
                        for index in range(config["fitting"]["sampling"]["ensembles"])
                    ],
                    "gr_history": [],
                    "stop_reason": "budget_exhausted",
                }
            )
        )

    monkeypatch.setattr(
        covariance,
        "read_route_covariance",
        lambda current, route: {
            "arrays": sources[review.route_key(route)][0],
            "metadata": {"synthetic_reader_fixture": True},
        },
    )
    monkeypatch.setattr(
        covariance,
        "route_vectors",
        lambda current, route, measurements: sources[review.route_key(route)][0],
    )
    monkeypatch.setattr(
        pipeline_clustering,
        "read_measurements",
        lambda current, route: sources[review.route_key(route)][1],
    )
    monkeypatch.setattr(
        driver, "_science", lambda *args, **kwargs: pytest.fail("review launched science")
    )
    destination = run["root"] / "review"
    destination.mkdir()
    notes = destination / "review.md"
    notes.write_text("Notes about synthetic report fixtures.\n")
    result = review.render_review(run["root"])
    # Keep actual errors visible rather than truncating a mostly complete coverage table.
    errors = _table(destination / "errors.csv")
    assert not errors, json.dumps(errors, indent=2)
    assert result["missing_product_count"] == 0, [
        row
        for row in _table(destination / "coverage.csv")
        if row["coverage"]
        not in {
            "complete",
            "stopping_rule_reached",
            "budget_exhausted",
            "not configured for this bin",
        }
    ]
    assert notes.read_text() == "Notes about synthetic report fixtures.\n"
    posterior = _table(destination / "posterior.csv")
    assert {(row["route"], row["parameter"]) for row in posterior} == {
        (key, name) for key, names in expected_parameters.items() for name in names
    }
    assert all(names == ["qiso", "qap"] for names in expected_parameters.values())
    assert len(posterior) == 100  # Both dilation parameters for all fifty routes.
    assert all(json.loads(row["ensemble_contributions"]) == [800] * 4 for row in posterior)
    profiles = _table(destination / "profiles.csv")
    assert len(profiles) == 100 * 26
    assert sum(row["fit_scope"] == "mean" for row in profiles) == 100
    comparisons = _table(destination / "condition_comparisons.csv")
    assert len(comparisons) == 5000  # Five bins, ten condition pairs, two states/axes, 25 mocks.
    assert {row["realization_id"] for row in comparisons} == {str(i) for i in range(25)}
    assert {row["reconstruction_state"] for row in comparisons} == {
        "prereconstruction",
        "postreconstruction",
    }
    assert all(row["interval_comparison_reason"] == "" for row in comparisons)
    sampler = _table(destination / "sampler.csv")
    assert len({(row["route"], row["ensemble_index"]) for row in sampler}) == 200
    for filename, count, gain in (
        ("paired_fits.csv", 1250, 1 / 3),
        ("aggregate_gains.csv", 50, 0.5),
    ):
        rows = _table(destination / filename)
        assert len(rows) == count
        assert all(float(row["gain"]) == pytest.approx(gain) for row in rows)
    parameters = {
        row["parameter"]: row["value"]
        for row in _table(destination / "parameters.csv")
        if row["source"] == "snapshot"
    }
    assert json.loads(parameters["fitting.nonlinear_parameters.qiso.prior.limits"]) == [0.8, 1.2]
    assert parameters["reporting.gain.ratio"] == "post_over_pre"
    power_coverage = [
        row for row in _table(destination / "coverage.csv") if row["product"] == "power"
    ]
    assert sum(row["coverage"] == "complete" for row in power_coverage) == 20
    assert sum(row["coverage"] == "not configured for this bin" for row in power_coverage) == 30
    assert len(list(destination.glob("*--_plot_power.png"))) == 10

    class LocalLinks(HTMLParser):
        def __init__(self):
            super().__init__()
            self.links = []

        def handle_starttag(self, tag, attrs):
            self.links.extend(value for name, value in attrs if name in {"href", "src"})

    parser = LocalLinks()
    parser.feed((destination / "report.html").read_text())
    assert len([link for link in parser.links if link.endswith(".png")]) == result["figure_count"]
    assert len([link for link in parser.links if link.endswith(".pdf")]) == result["figure_count"]
    for link in parser.links:
        assert not Path(link).is_absolute() and ":" not in link
        assert (destination / link).is_file(), link
