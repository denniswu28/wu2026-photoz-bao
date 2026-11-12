"""Notebook/table helpers consume the same actual native products as static review."""

import csv
import json
from pathlib import Path
import numpy as np
import pytest
from pzbao import run as driver
from pzbao.figures.data import (
    effective_volume,
    load_all_fits,
    load_all_per_mock_profiles,
    load_chain_samples,
    render_table,
    write_csv,
)
from tests import test_run
from tests.test_review_package import _profile, _chain, _identity, _route

linear_run = test_run.linear_run


def test_effective_volume_requires_explicit_p0():
    np.testing.assert_allclose(effective_volume([1e9], [1e-4], p0_h3_mpc3=10000.0), [2.5e8])
    with pytest.raises(TypeError):
        effective_volume([1e9], [1e-4])


def test_linear_native_profiles_reach_notebook_and_table_readers(linear_run):
    route = _route(linear_run)
    base = linear_run["root"] / "fit" / "units"
    for identifier in (0, None):
        suffix = "--mean" if identifier is None else "--mock-0"
        _profile(
            base / (driver.route_name(route) + suffix),
            {**_identity(linear_run, route), "realization_id": identifier},
        )
    rows = load_all_fits(linear_run["root"])
    assert len(rows) == 2
    assert len(load_all_per_mock_profiles(linear_run["root"])) == 1
    assert {row["realization_id"] for row in rows} == {0, None}
    for row in rows:
        parameter = next(item for item in row["parameters"] if item["basename"] == "qiso")
        assert parameter["value"] == 1.02
        assert parameter["minos_offsets"] == [-0.03, 0.08]
    table = render_table(rows)
    assert "1.02_{-0.03}^{+0.08}" in table and "mean" in table
    destination = linear_run["root"] / "inspection.csv"
    exact = np.nextafter(1.0, 2.0)
    write_csv(destination, [{"value": exact, "interval": [-0.03, 0.08]}])
    with destination.open() as stream:
        decoded = next(csv.DictReader(stream))
    assert float(decoded["value"]) == exact
    assert json.loads(decoded["interval"]) == [-0.03, 0.08]


def test_linear_native_chain_reader_preserves_ensemble_walker_and_retained_time(linear_run):
    route = _route(linear_run)
    directory = linear_run["root"] / "sample" / "units" / driver.route_name(route)
    expected = []
    for index in range(4):
        _, values = _chain(
            directory,
            {**_identity(linear_run, route), "ensemble_index": index},
            mean=1.0 + 0.001 * index,
        )
        expected.append(values[200:].T)
    summary = {
        "identity": _identity(linear_run, route),
        "complete_ensemble_collection": True,
        "ensembles": [
            {"ensemble_index": index, "path": f"ensemble-{index}.npy"} for index in range(4)
        ],
    }
    path = directory / "sampling.json"
    path.write_text(json.dumps(summary))
    values = load_chain_samples(linear_run["root"], selectors=route, parameter="qiso")
    assert values.shape == (4, 40, 200)
    np.testing.assert_array_equal(values, np.stack(expected))
    summary["ensembles"][0]["ensemble_index"] = False
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="identified ensemble"):
        load_chain_samples(linear_run["root"], selectors=route, parameter="qiso")


def test_linear_diagnostic_power_renders_supplied_values_and_preserves_manuscript_assets(tmp_path):
    k = np.array([0.1, 0.2, 0.3])
    pre = np.arange(9.0).reshape(3, 3)
    from tests.notebook_helpers import notebook_namespace

    namespace = notebook_namespace("baseline", cells={"setup", "baseline-style", "power"})
    render = namespace["render_diagnostic_power_demonstration"]
    fig = render(k, pre, pre * 1.4)
    for index, ax in enumerate(fig.axes):
        np.testing.assert_array_equal(ax.lines[0].get_ydata(), pre[index])
        np.testing.assert_array_equal(ax.lines[1].get_ydata(), pre[index] * 1.4)
    assert not list(tmp_path.iterdir())
    with pytest.raises(ValueError, match="same first dimension"):
        render(k, pre[:, :2], pre)


def _execute_paper_notebook(monkeypatch, run):
    pytest.importorskip("IPython.display")
    monkeypatch.setenv("BAO_RUN_ROOT", str(run["root"]))
    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(root)
    notebook = json.loads((root / "pzbao/figures/baseline.ipynb").read_text())
    namespace = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert not cell["outputs"] and cell["execution_count"] is None
            exec(compile("".join(cell["source"]), cell["id"], "exec"), namespace)
    return namespace


def test_paper_notebook_opens_partial_run_without_starting_science(linear_run, monkeypatch):
    monkeypatch.setattr(driver, "_science", lambda *a, **k: pytest.fail("notebook ran science"))
    result = _execute_paper_notebook(monkeypatch, linear_run)
    assert result["profiles"] == ()
    assert result["chain_samples"] is None
    assert result["figure_paths"] == []
    assert not (linear_run["root"] / "review").exists()


def test_paper_notebook_preserves_native_intervals_axes_and_post_pre_gain(linear_run, monkeypatch):
    route = _route(linear_run)
    base = linear_run["root"] / "fit" / "units"
    for identifier in (0, None):
        suffix = "--mean" if identifier is None else "--mock-0"
        _profile(
            base / (driver.route_name(route) + suffix),
            {**_identity(linear_run, route), "realization_id": identifier},
        )
    directory = linear_run["root"] / "sample" / "units" / driver.route_name(route)
    expected = []
    for index in range(4):
        _, values = _chain(
            directory,
            {**_identity(linear_run, route), "ensemble_index": index},
            mean=1.0 + 0.001 * index,
        )
        expected.append(values[200:].T)
    (directory / "sampling.json").write_text(
        json.dumps(
            {
                "identity": _identity(linear_run, route),
                "complete_ensemble_collection": True,
                "ensembles": [{"ensemble_index": i, "path": f"ensemble-{i}.npy"} for i in range(4)],
            }
        )
    )
    review = linear_run["root"] / "review"
    review.mkdir()
    write_csv(
        review / "aggregate_gains.csv",
        [{"parameter": "qiso", "sigma_pre": 0.04, "sigma_post": 0.02, "gain": 0.5}],
    )
    result = _execute_paper_notebook(monkeypatch, linear_run)
    assert len(result["individual_profiles"]) == len(result["mean_profiles"]) == 1
    assert "1.02_{-0.03}^{+0.08}" in result["mean_profile_table"]
    np.testing.assert_array_equal(result["chain_samples"], np.stack(expected))
    assert result["chain_samples"].shape == (4, 40, 200)
    assert float(result["mean_gain_rows"][0]["gain"]) == 0.5
