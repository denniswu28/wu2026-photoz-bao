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
    }
    for function, product in names.items():
        monkeypatch.setattr(
            appendix_checks, function, lambda *args, product=product, **kw: {"product": product}
        )
    monkeypatch.setattr(
        sys,
        "argv",
        ["appendix_checks", "--primary-run", "p", "--transfer-root", "t"]
        + ["--paper-dir", str(tmp_path)]
        + list(arguments),
    )
    appendix_checks.main()
    out = tmp_path / "figures" / "source" / "appendix"
    return sorted(json.loads(path.read_text())["product"] for path in out.glob("*.json"))


def test_default_products_need_no_core_run(monkeypatch, tmp_path):
    written = run_main(monkeypatch, tmp_path)
    assert written == [
        "alpha-sn-leave-one-out",
        "cross-cap",
        "cross-sigma-alpha-perp",
        "cross-sigma-checks",
    ]


def test_core_run_adds_its_products(monkeypatch, tmp_path):
    assert len(run_main(monkeypatch, tmp_path, "--core-run", "c")) == 8


def test_core_product_without_core_run_is_refused(monkeypatch, tmp_path):
    with pytest.raises(SystemExit):
        run_main(monkeypatch, tmp_path, "--products", "template-b1")
