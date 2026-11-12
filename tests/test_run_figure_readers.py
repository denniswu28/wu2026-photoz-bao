"""Figure readers for eight-command runs: robustness, smoothing scan, reproducibility, transfer."""

import copy

import pytest

from pzbao import fitting, run as driver
from pzbao.config import (
    POWER3_FLAT,
    PRIMARY_POWER3,
    UNIFIED_FLAT,
    UNIFIED_PCS2,
    UNIFIED_PRIOR,
    load_analysis_config,
)
from pzbao.config.sensitivity import sensitivity_run_configs
from pzbao.figures import data
from tests.test_core_sensitivity import schema13
from tests.legacy_config import load_analysis_config as load_legacy_config

CODE = {"code_commit": "a" * 40, "repository": "fixture", "dirty": False, "code_state_sha256": "x"}


def _profile(value=1.01):
    parameter = {
        "value": value,
        "minos_bounds": [value - 0.02, value + 0.02],
        "interval_flags": {"is_valid": True, "lower_valid": True, "upper_valid": True},
        "fixed": False,
    }
    return {
        "parameters": [{**parameter, "basename": "qiso"}, {**parameter, "basename": "qap"}],
        "maximization": {
            "is_valid": True,
            "is_above_max_edm": False,
            "has_reached_call_limit": False,
        },
    }


@pytest.fixture
def profiles(monkeypatch):
    values = {}
    monkeypatch.setattr(
        fitting, "read_profiles", lambda path, *, identity: _profile(values.get(str(path), 1.01))
    )
    return values


def _publish(run, *, state="postreconstruction", condition="sigma_0.003", mocks=(None, 0, 1)):
    for unit in driver.stage_units("fit", run["config"], run["metadata"]["workload"]):
        if unit["reconstruction_state"] != state or unit["condition"] != condition:
            continue
        if unit["realization_id"] not in mocks:
            continue
        directory = run["root"] / "fit" / "units" / unit["unit_id"]
        directory.mkdir(parents=True)
        (directory / "profiles.npy").write_bytes(b"fixture")


@pytest.fixture
def core(tmp_path, monkeypatch, profiles):
    monkeypatch.setattr(driver, "_code_identity", lambda **kwargs: dict(CODE))
    source_config = schema13()
    monkeypatch.setattr(driver, "load_analysis_config", lambda: copy.deepcopy(source_config))
    source = driver.create_run(tmp_path, "core", workload="pilot", catalog_root=tmp_path)
    runs = {}
    for name, definition in (("refit", PRIMARY_POWER3), ("flat", POWER3_FLAT)):
        runs[name] = driver.create_run(
            tmp_path,
            name,
            workload="pilot",
            catalog_root=tmp_path,
            source_run=source,
            inference_definition=definition,
        )
    _publish(runs["refit"])
    _publish(runs["refit"], state="prereconstruction")
    _publish(runs["flat"])
    return source, runs


def test_robustness_rows_pair_baseline_pre_and_flat_and_reject_drift(core, monkeypatch):
    source, runs = core
    means, rows = data.robustness_rows_from_runs(runs["refit"]["root"], flat=runs["flat"]["root"])
    assert {(r["variant"], r["measurement_bin"]) for r in means} == {
        (v, b) for v in ("baseline", "pre", "flat") for b in ("BGS1", "LRG3")
    }
    assert all(r["status"] == "available" and r["usable"] for r in means)
    individuals = [r for r in rows if r["realization_id"] is not None]
    assert len(individuals) == 3 * 2 * 2 * 25
    assert sum(r["status"] == "complete" for r in individuals) == 3 * 2 * 2 * 2
    grid = data.robustness_grid(means)
    assert {r["status"] for r in grid} == {"available", "missing"}
    original = data._comparable_core
    monkeypatch.setattr(
        data,
        "_comparable_core",
        lambda config: {
            **original(config),
            "drift": config["fitting"]["damping"].get("applied_prior"),
        },
    )
    with pytest.raises(ValueError, match="beyond its registered variation"):
        data.robustness_rows_from_runs(runs["refit"]["root"], flat=runs["flat"]["root"])
    with pytest.raises(ValueError, match="core power3 refit"):
        data.robustness_rows_from_runs(runs["flat"]["root"])


def _scan_runs(tmp_path, monkeypatch, radii=(15, 25), config=None):
    monkeypatch.setattr(driver, "_code_identity", lambda **kwargs: dict(CODE))
    # The reproducibility pairing compares the broad-field scan with the broad-field core
    # refit; per-bin scans belong to the per-bin primary measurement (revision 2.13).
    configs = sensitivity_run_configs(config or load_legacy_config())
    runs = {}
    for radius in radii:
        snapshot = configs[f"scan--R{radius}"]
        monkeypatch.setattr(driver, "load_analysis_config", lambda s=snapshot: copy.deepcopy(s))
        runs[radius] = driver.create_run(
            tmp_path, f"scan-{radius}", workload="smoothing_scan", catalog_root=tmp_path
        )
    return runs


def test_radius_scan_rows_compare_radii_only(tmp_path, monkeypatch, profiles):
    runs = _scan_runs(tmp_path, monkeypatch)
    for radius, run in runs.items():
        _publish(run)
    for path in (runs[25]["root"] / "fit" / "units").rglob("profiles.npy"):
        profiles[str(path)] = 1.02
    summary = data.radius_scan_rows([run["root"] for run in runs.values()])
    lrg = [r for r in summary if r["measurement_bin"] == "LRG3" and r["parameter"] == "qiso"]
    assert {r["smoothing_mpc_h"] for r in lrg} == {15.0, 25.0}
    wide = next(r for r in lrg if r["smoothing_mpc_h"] == 25.0)
    assert wide["bias"] == pytest.approx(0.02) and wide["width_ratio_pairs"] == 2
    assert wide["median_width_ratio_vs_reference"] == pytest.approx(1.0)
    assert wide["joint_fom"] is None
    assert wide["n_saved"] == 2 and wide["n_failed"] == 23
    assert wide["n_bestfit"] == wide["n_usable_interval"] == 2
    missing_route = next(
        r
        for r in summary
        if r["measurement_bin"] == "LRG3"
        and r["condition"] == "sigma_0.01"
        and r["parameter"] == "qiso"
        and r["smoothing_mpc_h"] == 25.0
    )
    assert missing_route["n_saved"] == 0 and missing_route["n_failed"] == 25
    assert missing_route["bias"] is None and missing_route["scatter"] is None
    assert missing_route["median_width_ratio_vs_reference"] is None
    with pytest.raises(ValueError, match="distinct radii"):
        data.radius_scan_rows([runs[15]["root"], runs[15]["root"]])


def test_scan_r15_reproduces_against_the_core_refit(tmp_path, monkeypatch, profiles, core):
    _, runs = core
    scan = _scan_runs(tmp_path / "scan", monkeypatch, radii=(15,))[15]
    _publish(scan)
    rows = data.scan_reproducibility_rows(scan["root"], runs["refit"]["root"])
    paired = [r for r in rows if not r["missing"]]
    assert paired and all(r["scan_minus_core"] == pytest.approx(0.0) for r in paired)
    assert all(r["width_ratio_scan_over_core"] == pytest.approx(1.0) for r in paired)
    both_missing = [r for r in rows if r["missing"] == ["scan", "core"]]
    assert both_missing
    assert all(r["scan_minus_core"] is None for r in both_missing)
    assert all(r["width_ratio_scan_over_core"] is None for r in both_missing)


def test_missing_fit_units_include_failed_attempts_and_missing_means(core):
    _, runs = core
    root = runs["refit"]["root"]
    unit = "LRG3--sigma_0.003--postreconstruction--mock-0"
    (root / "fit" / "units" / unit / "profiles.npy").unlink()
    attempt = root / "fit" / "attempts" / (unit + ".fixture")
    attempt.mkdir(parents=True)
    (attempt / "failure.txt").write_text("optimizer failed")
    (
        root / "fit" / "units" / "LRG3--sigma_0.003--postreconstruction--mean" / "profiles.npy"
    ).unlink()
    means, rows = data.robustness_rows_from_runs(root)
    group = [
        r
        for r in rows
        if r["measurement_bin"] == "LRG3"
        and r["variant"] == "baseline"
        and r["parameter"] == "qiso"
    ]
    assert len(group) == 26
    summary = data._route_summary(group)
    assert summary["n_saved"] == summary["n_bestfit"] == 1
    assert summary["n_failed"] == 24
    assert summary["bias"] == pytest.approx(0.01)
    assert summary["scatter"] is None
    assert summary["mean_fit_value"] is None and not summary["mean_fit_usable"]
    missing_mean = next(
        r
        for r in means
        if r["measurement_bin"] == "LRG3"
        and r["variant"] == "baseline"
        and r["parameter"] == "qiso"
    )
    assert missing_mean["status"] == "missing"


def test_published_profiles_with_invalid_provenance_still_fail(core, monkeypatch):
    _, runs = core

    def invalid(*args, **kwargs):
        raise ValueError("native profile identity differs")

    monkeypatch.setattr(fitting, "read_profiles", invalid)
    with pytest.raises(ValueError, match="identity differs"):
        data._fit_rows(runs["refit"]["root"], variant="baseline")


def test_radius_scan_requires_configured_reference_before_reading_fits(
    tmp_path, monkeypatch, profiles
):
    runs = _scan_runs(tmp_path, monkeypatch, radii=(25, 35))
    monkeypatch.setattr(
        data, "_fit_rows", lambda *a, **k: pytest.fail("read fits before validation")
    )
    with pytest.raises(ValueError, match="missing configured smoothing reference R15"):
        data.radius_scan_rows([r["root"] for r in runs.values()])


def test_radius_scan_uses_saved_reference_instead_of_smallest_radius(
    tmp_path, monkeypatch, profiles
):
    runs = _scan_runs(tmp_path, monkeypatch)
    for run in runs.values():
        _publish(run)
    loaded = {r["root"]: copy.deepcopy(r) for r in runs.values()}
    for run in loaded.values():
        run["config"]["measurement"]["reconstruction"]["exploratory_smoothing_scan"][
            "reference_smoothing_mpc_h"
        ] = 25.0
    monkeypatch.setattr(driver, "load_run", lambda root: loaded[root])
    summary = data.radius_scan_rows([runs[25]["root"], runs[15]["root"]])
    assert summary and {r["reference_smoothing_mpc_h"] for r in summary} == {25.0}
    loaded[runs[15]["root"]]["config"]["measurement"]["reconstruction"][
        "exploratory_smoothing_scan"
    ]["reference_smoothing_mpc_h"] = 15.0
    with pytest.raises(ValueError, match="different configured references"):
        data.radius_scan_rows([r["root"] for r in runs.values()])


def test_width_ratios_only_use_paired_available_mock_ids(tmp_path, monkeypatch, profiles, core):
    _, core_runs = core
    runs = _scan_runs(tmp_path / "scan", monkeypatch)
    for run in runs.values():
        _publish(run)
    # R15 has only mock 0; R25 has only mock 1. Neither can supply a paired ratio.
    for radius, mock in ((15, 1), (25, 0)):
        (
            runs[radius]["root"]
            / "fit"
            / "units"
            / f"LRG3--sigma_0.003--postreconstruction--mock-{mock}"
            / "profiles.npy"
        ).unlink()
    summary = data.radius_scan_rows([r["root"] for r in runs.values()])
    wide = next(
        r
        for r in summary
        if r["measurement_bin"] == "LRG3"
        and r["condition"] == "sigma_0.003"
        and r["parameter"] == "qiso"
        and r["smoothing_mpc_h"] == 25.0
    )
    assert wide["n_saved"] == 1 and wide["n_failed"] == 24
    assert wide["width_ratio_pairs"] == 0 and wide["median_width_ratio_vs_reference"] is None
    rows = data.scan_reproducibility_rows(runs[15]["root"], core_runs["refit"]["root"])
    missing = next(
        r
        for r in rows
        if r["measurement_bin"] == "LRG3"
        and r["condition"] == "sigma_0.003"
        and r["parameter"] == "qiso"
        and r["realization_id"] == 1
    )
    assert missing["missing"] == ["scan"]
    assert missing["scan_minus_core"] is None and missing["width_ratio_scan_over_core"] is None


def test_reproducibility_rejects_finite_flat_refit(tmp_path, monkeypatch, profiles, core):
    _, runs = core
    scan = _scan_runs(tmp_path / "scan", monkeypatch, radii=(15,))[15]
    monkeypatch.setattr(
        data, "_fit_rows", lambda *a, **k: pytest.fail("read fits before validation")
    )
    with pytest.raises(ValueError, match="core power3 refit"):
        data.scan_reproducibility_rows(scan["root"], runs["flat"]["root"])


@pytest.mark.parametrize(
    "target,path,value",
    [
        ("scan", ("metadata", "workload"), "pilot"),
        (
            "scan",
            ("config", "measurement", "reconstruction", "sensitivity_run"),
            {"kind": "reciso"},
        ),
        ("both", ("config", "measurement", "reconstruction", "smoothing_values_mpc_h"), [25.0]),
        ("scan", ("config", "measurement", "reconstruction", "allowed_conventions"), ["reciso"]),
        ("core", ("config", "fitting", "damping", "applied_prior"), "paper_finite_flat"),
        ("scan", ("config", "fitting", "damping", "reference_smoothing_mpc_h"), 25.0),
        ("scan", ("config", "fitting", "broadband_model"), "pcs2"),
        ("scan", ("config", "fitting", "fit_surfaces", "BGS_BRIGHT-21.5", "ap_mode"), "qiso"),
        ("scan", ("config", "fitting", "linear_nuisance", "varied_basenames_by_tracer", "LRG"), []),
    ],
)
def test_reproducibility_checks_run_and_inference_definitions_before_fits(
    tmp_path, monkeypatch, profiles, core, target, path, value
):
    _, runs = core
    scan = _scan_runs(tmp_path / "scan", monkeypatch, radii=(15,))[15]
    loaded = {
        scan["root"]: copy.deepcopy(scan),
        runs["refit"]["root"]: copy.deepcopy(runs["refit"]),
    }
    targets = ("scan", "core") if target == "both" else (target,)
    for member in targets:
        owner = loaded[scan["root"] if member == "scan" else runs["refit"]["root"]]
        for key in path[:-1]:
            owner = owner[key]
        owner[path[-1]] = value
    monkeypatch.setattr(driver, "load_run", lambda root: loaded[root])
    monkeypatch.setattr(
        data, "_fit_rows", lambda *a, **k: pytest.fail("read fits before validation")
    )
    with pytest.raises(
        ValueError, match="reproducibility requires|different inference definitions"
    ):
        data.scan_reproducibility_rows(scan["root"], runs["refit"]["root"])


def test_reproducibility_refuses_a_per_bin_scan_against_the_broad_field_refit(
    tmp_path, monkeypatch, profiles, core
):
    _, runs = core
    current = load_analysis_config()
    assert current["measurement"]["reconstruction"]["scope"] == "separate_measurement_bins"
    scan = _scan_runs(tmp_path / "scan", monkeypatch, radii=(15,), config=current)[15]
    _publish(scan)
    monkeypatch.setattr(
        data, "_fit_rows", lambda *a, **k: pytest.fail("read fits before validation")
    )
    with pytest.raises(ValueError, match="differ beyond their producer"):
        data.scan_reproducibility_rows(scan["root"], runs["refit"]["root"])


def test_transfer_comparison_wraps_the_verified_review_pairs(monkeypatch):
    from pzbao.diagnostics import review

    calls = {}
    run = {"config": {}, "metadata": {"workload": "transfer"}}
    monkeypatch.setattr(driver, "load_run", lambda root: {**run, "root": root})
    monkeypatch.setattr(
        data,
        "routes_for_run",
        lambda r: [
            {
                "tracer": "LRG",
                "measurement_bin": "LRG2",
                "condition": "sigma_0.01",
                "reconstruction_state": state,
            }
            for state in ("prereconstruction", "postreconstruction")
        ],
    )
    monkeypatch.setattr(review, "collect_route", lambda r, coords: ({**coords, "fits": True}, None))

    def rows(variant, reference, records):
        calls["records"] = records
        if reference["root"] == "bad":
            raise ValueError("transfer target differs from self")
        return ["pair"], ["summary"]

    monkeypatch.setattr(review, "transfer_comparison_rows", rows)
    assert data.transfer_comparison("source", "self") == (["pair"], ["summary"])
    assert [r["reconstruction_state"] for r in calls["records"]] == ["postreconstruction"]
    with pytest.raises(ValueError, match="differs from self"):
        data.transfer_comparison("source", "bad")


def test_per_bin_radius_curve_reads_only_the_own_radius_centre_refits(
    tmp_path, monkeypatch, profiles
):
    """Revision 2.18: frozen per-bin scan runs pin the centre to 15 Mpc/h and are not read."""
    current = load_analysis_config()
    scans = _scan_runs(tmp_path, monkeypatch, config=current)
    refits = {
        radius: driver.create_run(
            tmp_path,
            f"scan-{radius}-centre",
            workload="smoothing_scan",
            catalog_root=tmp_path,
            source_run=run,
            inference_definition="core_scan_radius_centre_v1",
        )
        for radius, run in scans.items()
    }
    for radius, run in refits.items():
        assert run["config"]["fitting"]["damping"]["reference_smoothing_mpc_h"] == radius
        _publish(run)
    summary = data.radius_scan_rows([run["root"] for run in refits.values()])
    assert {r["smoothing_mpc_h"] for r in summary} == {15.0, 25.0}
    assert {r["reference_smoothing_mpc_h"] for r in summary} == {15.0}
    with pytest.raises(ValueError, match="own-radius centre refits"):
        data.radius_scan_rows([run["root"] for run in scans.values()])
    with pytest.raises(ValueError, match="mix damping-centre definitions"):
        data.radius_scan_rows([scans[15]["root"], refits[25]["root"]])
    # Each refit reads its scan run's measured vectors, route covariance and calibration.
    from pzbao.covariance import likelihood_covariance_run

    loaded = driver.load_run(refits[25]["root"])
    route = driver.routes_for(loaded["config"], "smoothing_scan")[0]
    assert likelihood_covariance_run(loaded, route)["root"] == scans[25]["root"]


# Revision 2.20: the unified-prior refits replace the power3 refit and the scan centre refits as
# the readers' inputs; each curve or comparison stays inside one prior family.


@pytest.fixture
def unified_core(tmp_path, monkeypatch, profiles):
    """The per-bin primary refitted with the unified prior, plus its PCS2 and flat checks."""
    monkeypatch.setattr(driver, "_code_identity", lambda **kwargs: dict(CODE))
    current = load_analysis_config()
    monkeypatch.setattr(driver, "load_analysis_config", lambda: copy.deepcopy(current))
    root = tmp_path / "unified"  # beside, never inside, the power3 fixture's runs
    source = driver.create_run(root, "primary", workload="full_grid", catalog_root=root)
    runs = {}
    for name, definition in (
        ("refit", UNIFIED_PRIOR),
        ("pcs2", UNIFIED_PCS2),
        ("flat", UNIFIED_FLAT),
    ):
        runs[name] = driver.create_run(
            root,
            name,
            workload="full_grid",
            catalog_root=root,
            source_run=source,
            inference_definition=definition,
        )
    _publish(runs["refit"])
    _publish(runs["refit"], state="prereconstruction")
    _publish(runs["pcs2"])
    _publish(runs["flat"])
    return source, runs


def test_unified_robustness_rows_pair_the_primary_refit_with_its_checks(unified_core):
    _, runs = unified_core
    means, rows = data.robustness_rows_from_runs(
        runs["refit"]["root"],
        flat=runs["flat"]["root"],
        pcs2_lrg=runs["pcs2"]["root"],
        pcs2_bgs=runs["pcs2"]["root"],
    )
    assert {r["variant"] for r in rows} == {"baseline", "pre", "flat", "pcs2"}
    available = {(r["variant"], r["measurement_bin"]) for r in means if r["status"] == "available"}
    assert {("baseline", "LRG3"), ("pre", "LRG3"), ("flat", "LRG3"), ("pcs2", "LRG3")} <= available
    assert {("baseline", "BGS1"), ("pcs2", "BGS1")} <= available
    # A unified baseline pairs only with unified checks, and a check is never the baseline.
    with pytest.raises(ValueError, match="prior family"):
        data.robustness_rows_from_runs(runs["refit"]["root"], flat=runs["refit"]["root"])
    with pytest.raises(ValueError, match="core power3 refit or the unified refit"):
        data.robustness_rows_from_runs(runs["flat"]["root"])
    with pytest.raises(ValueError, match="core power3 refit or the unified refit"):
        data.robustness_rows_from_runs(runs["pcs2"]["root"])


def test_unified_baseline_refuses_power3_family_checks(unified_core, core):
    _, unified = unified_core
    _, old = core
    with pytest.raises(ValueError, match="registered variation|prior family"):
        data.robustness_rows_from_runs(unified["refit"]["root"], flat=old["flat"]["root"])
    with pytest.raises(ValueError, match="registered variation|prior family"):
        data.robustness_rows_from_runs(old["refit"]["root"], flat=unified["flat"]["root"])


def test_unified_scan_curve_reads_the_unified_refits_and_never_mixes_families(
    tmp_path, monkeypatch, profiles
):
    current = load_analysis_config()
    scans = _scan_runs(tmp_path, monkeypatch, config=current)
    refits = {
        radius: driver.create_run(
            tmp_path,
            f"scan-{radius}-unified",
            workload="smoothing_scan",
            catalog_root=tmp_path,
            source_run=run,
            inference_definition=UNIFIED_PRIOR,
        )
        for radius, run in scans.items()
    }
    for run in refits.values():
        _publish(run)
    summary = data.radius_scan_rows([run["root"] for run in refits.values()])
    assert {r["smoothing_mpc_h"] for r in summary} == {15.0, 25.0}
    assert {r["reference_smoothing_mpc_h"] for r in summary} == {15.0}
    centre = driver.create_run(
        tmp_path,
        "scan-15-centre",
        workload="smoothing_scan",
        catalog_root=tmp_path,
        source_run=scans[15],
        inference_definition="core_scan_radius_centre_v1",
    )
    with pytest.raises(ValueError, match="mix damping-centre definitions"):
        data.radius_scan_rows([centre["root"], refits[25]["root"]])
    with pytest.raises(ValueError, match="own-radius centre refits or unified refits"):
        data.radius_scan_rows([run["root"] for run in scans.values()])


def test_unified_reproducibility_pairs_the_scan_refit_with_the_primary_refit(
    tmp_path, monkeypatch, profiles, unified_core
):
    _, runs = unified_core
    scan = _scan_runs(tmp_path / "scan", monkeypatch, radii=(15,), config=load_analysis_config())
    refit = driver.create_run(
        tmp_path / "scan",
        "scan-15-unified",
        workload="smoothing_scan",
        catalog_root=tmp_path / "scan",
        source_run=scan[15],
        inference_definition=UNIFIED_PRIOR,
    )
    _publish(refit)
    rows = data.scan_reproducibility_rows(refit["root"], runs["refit"]["root"])
    paired = [r for r in rows if not r["missing"]]
    assert paired and all(r["scan_minus_core"] == pytest.approx(0.0) for r in paired)
    assert all(r["width_ratio_scan_over_core"] == pytest.approx(1.0) for r in paired)
    # The scan run's own fits (old family) never pair with the unified primary refit.
    with pytest.raises(ValueError, match="one prior family"):
        data.scan_reproducibility_rows(scan[15]["root"], runs["refit"]["root"])
