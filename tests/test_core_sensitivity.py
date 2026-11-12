"""Core power3 refit, finite flat check, smoothing scan, RecIso and compact core storage."""

import copy
import sys

import numpy as np
import pytest
import yaml
from click.testing import CliRunner

from pzbao import catalog, covariance, selection, storage, run as driver
from pzbao.config import (
    BGS_ANISOTROPIC,
    POWER3_FLAT,
    PRIMARY_POWER3,
    derive_inference_config,
    validate_inference_config,
)
from pzbao.config.schema_validation import (
    PCS2_NUISANCE_AXES,
    POWER3_NUISANCE_AXES,
    validate_analysis_config,
)
from pzbao.config.sensitivity import sensitivity_run_configs
from pzbao.theory import analytic_postreconstruction_sigma_perp, finite_flat_damping_prior
from tests.test_bgs_inference import legacy_bgs
from pzbao.config import load_analysis_config as load_current_config
from tests.legacy_config import load_analysis_config


def schema13():
    """A schema-13 PCS2 core source built from today's definition, as the core run was."""
    source = legacy_bgs(load_analysis_config())
    fitting = source["fitting"]
    fitting["broadband_model"] = "pcs2"
    fitting["linear_nuisance"]["varied_basenames_by_tracer"]["LRG"] = list(PCS2_NUISANCE_AXES)
    fitting["linear_nuisance"]["allowed_basenames_by_model"] = {"pcs2": list(PCS2_NUISANCE_AXES)}
    # The measured core snapshot predates the paper comparison scope.
    fitting.pop("paper_comparisons")
    validate_analysis_config(source)
    return source


def test_power3_refit_changes_only_the_configured_fit_definition():
    source = schema13()
    before = copy.deepcopy(source)
    derived = derive_inference_config(source, PRIMARY_POWER3)
    assert source == before
    assert derived["schema_version"] == 15 and derived["fitting"]["broadband_model"] == "power3"
    nuisance = derived["fitting"]["linear_nuisance"]
    assert nuisance["allowed_basenames_by_model"] == {"power3": list(POWER3_NUISANCE_AXES)}
    assert set(map(tuple, nuisance["varied_basenames_by_tracer"].values())) == {
        POWER3_NUISANCE_AXES
    }
    assert derived["fitting"]["fit_surfaces"]["BGS_BRIGHT-21.5"]["ap_mode"] == "qisoqap"
    # Measurement, covariance, damping priors and photo-z response are the source's.
    for key in ("measurement", "selection", "tracers", "analysis"):
        assert derived[key] == source[key]
    for key in ("damping", "covariance", "photoz_response", "nonlinear_parameters"):
        assert derived["fitting"][key] == source["fitting"][key]
    changed = copy.deepcopy(derived)
    changed["fitting"]["damping"]["gaussian_widths_mpc_h"]["sigma_perp"] = 2.0
    with pytest.raises(ValueError, match="changes more"):
        validate_inference_config(source, changed, PRIMARY_POWER3)
    with pytest.raises(ValueError, match="schema-13 PCS2"):
        derive_inference_config(load_analysis_config(), PRIMARY_POWER3)


def test_flat_refit_adopts_only_the_finite_flat_support_and_its_routes():
    source = schema13()
    flat = derive_inference_config(source, POWER3_FLAT)
    primary = derive_inference_config(source, PRIMARY_POWER3)
    assert flat["fitting"]["damping"].pop("applied_prior") == "paper_finite_flat"
    scope = flat["fitting"].pop("paper_comparisons")
    assert flat == primary
    full = driver.stage_units("fit", primary, "full_grid", inference_definition=PRIMARY_POWER3)
    assert len(full) == 50 * 26
    routes = driver.stage_units(
        "sample",
        derive_inference_config(source, POWER3_FLAT),
        "full_grid",
        inference_definition=POWER3_FLAT,
    )
    assert {(u["measurement_bin"], u["condition"], u["reconstruction_state"]) for u in routes} == {
        (name, scope["condition"], scope["reconstruction_state"])
        for name in scope["measurement_bins"]
    }
    with pytest.raises(ValueError, match="reuse measured vectors"):
        driver.stage_units("covariance", primary, "full_grid", inference_definition=PRIMARY_POWER3)


def test_finite_flat_bounds_widen_only_the_parallel_support():
    record = {"sigma_parallel_photoz_mpc_h": 7.5, "centers_mpc_h": {}, "scope": "final"}
    flat = finite_flat_damping_prior(record, {"lower_mpc_h": 0.0, "nonlinear_upper_mpc_h": 10.0})
    assert flat["bounds_mpc_h"] == {
        "sigma_streaming": [0.0, 10.0],
        "sigma_parallel": [0.0, 12.5],
        "sigma_perp": [0.0, 10.0],
    }
    assert set(flat["unpriored_parameters"]) == {"sigma_streaming", "sigma_parallel", "sigma_perp"}
    assert record["scope"] == "final"
    with pytest.raises(ValueError, match="photo-z"):
        finite_flat_damping_prior(
            {"sigma_parallel_photoz_mpc_h": -1.0},
            {"lower_mpc_h": 0.0, "nonlinear_upper_mpc_h": 10.0},
        )
    config = load_analysis_config()
    config["fitting"]["damping"]["applied_prior"] = "paper_finite_flat"
    validate_analysis_config(config)
    del config["fitting"]["paper_comparisons"]
    with pytest.raises(ValueError, match="comparison scope"):
        validate_analysis_config(config)


def test_reference_radius_pins_the_analytic_transverse_centre():
    config = load_analysis_config()
    arguments = dict(
        tracer="LRG",
        attained_density_h3_mpc3=2.0e-4,
        linear_power=lambda k: 1.0e4 / (1.0 + (k / 0.05) ** 2),
        drag_scale_mpc_h=100.0,
    )
    reference = analytic_postreconstruction_sigma_perp(**arguments, config=config)
    wide = copy.deepcopy(config)
    wide["measurement"]["reconstruction"]["smoothing_values_mpc_h"] = [35.0]
    assert analytic_postreconstruction_sigma_perp(**arguments, config=wide) != reference
    wide["fitting"]["damping"]["reference_smoothing_mpc_h"] = 15.0
    pinned = analytic_postreconstruction_sigma_perp(**arguments, config=wide)
    assert pinned["analytic_center_mpc_h"] == reference["analytic_center_mpc_h"]
    assert pinned["smoothing_radius_mpc_h"] == 15.0


def test_sensitivity_runs_cover_the_scan_and_recisO_and_reject_drift():
    config = load_analysis_config()
    runs = sensitivity_run_configs(config)
    assert list(runs) == ["scan--R15", "scan--R25", "scan--R35", "scan--R45", "reciso--R15"]
    for name, snapshot in runs.items():
        recon = snapshot["measurement"]["reconstruction"]
        kind = recon["sensitivity_run"]["kind"]
        workload = snapshot["analysis"]["workloads"][kind]
        assert workload["reconstruction_states"] == ["postreconstruction"]
        assert snapshot["fitting"]["damping"]["reference_smoothing_mpc_h"] == 15.0
        assert recon["allowed_conventions"] == (["reciso"] if kind == "reciso" else ["recsym"])
        assert len(driver.stage_units("reconstruct", snapshot, kind)) == (
            25 * (1 if kind == "reciso" else 3) * 2
        )
    assert runs["reciso--R15"]["analysis"]["workloads"]["reciso"]["sigma_z_tiers"] == [
        "sigma_0.003"
    ]
    assert config["measurement"]["reconstruction"]["allowed_conventions"] == ["recsym"]
    for mutate, message in (
        (
            lambda c: c["measurement"]["reconstruction"].update(smoothing_values_mpc_h=[30.0]),
            "outside the configured",
        ),
        (lambda c: c["fitting"]["damping"].pop("reference_smoothing_mpc_h"), "reference-radius"),
        (
            lambda c: c["measurement"]["reconstruction"].update(allowed_conventions=["reciso"]),
            "allowed_conventions",
        ),
    ):
        broken = copy.deepcopy(runs["scan--R35"])
        mutate(broken)
        with pytest.raises(ValueError, match=message):
            validate_analysis_config(broken)
    primary = copy.deepcopy(config)
    primary["measurement"]["reconstruction"]["allowed_conventions"] = ["reciso"]
    with pytest.raises(ValueError, match="allowed_conventions"):
        validate_analysis_config(primary)


def test_sensitivity_yamls_reload_to_the_generated_snapshots(tmp_path):
    destination = tmp_path / "yaml"
    result = CliRunner().invoke(
        driver.main, ["sensitivity-yamls", "--output-dir", str(destination)]
    )
    assert result.exit_code == 0, (result.output, result.exception)
    for name, expected in sensitivity_run_configs(load_current_config()).items():
        path = destination / (name + ".yaml")
        assert load_current_config(path) == expected
        assert expected["measurement"]["reconstruction"]["scope"] == "separate_measurement_bins"
        assert yaml.safe_load(path.read_text())["cluster"]["statistics"] == "xi"


def test_reciso_shifts_randoms_by_displacement_only(monkeypatch):
    from tests.test_reconstruction_clustering import (
        _radial_field,
        _reconstruction_controls,
        _shell_reconstruction,
    )
    from pzbao.reconstruction import run_reconstruction

    _shell_reconstruction(monkeypatch)
    stub = sys.modules["pyrecon"].IterativeFFTReconstruction
    fields = []

    def shifted(self, positions, field):
        fields.append(field)
        return np.asarray(positions, dtype=float) + (2.0 if field == "disp+rsd" else 1.0)

    monkeypatch.setattr(stub, "read_shifted_positions", shifted)
    for convention, random_shift in (("recsym", 2.0), ("reciso", 1.0)):
        fields.clear()
        data, randoms = _radial_field(), _radial_field()
        moved_data, moved_randoms, record = run_reconstruction(
            data,
            randoms,
            bias=2.0,
            growth_rate=0.8,
            smoothing_radius_mpc_h=15,
            **{**_reconstruction_controls(), "convention": convention},
        )
        assert fields == ["disp+rsd", "disp+rsd" if convention == "recsym" else "disp"]
        np.testing.assert_allclose(moved_data["X_SHIFTED"], data["X"] + 2.0)
        np.testing.assert_allclose(moved_randoms["X_SHIFTED"], randoms["X"] + random_shift)
        assert record["convention"] == convention


def _selected(tmp_path, rows=20):
    identity = {
        "stage": "select",
        "tracer": "LRG",
        "condition": "sigma_0.003",
        "realization_id": 0,
        "cap": "NGC",
        "role": "random",
        "units": "Mpc/h",
        "run_name": "fixture",
    }
    rng = np.random.default_rng(1)
    base = {
        "SOURCE_ID": np.arange(10, 10 + rows, dtype=np.uint64),
        "Z": np.linspace(0.41, 0.99, rows),
        "X": rng.uniform(1000.0, 3000.0, rows),
        "Y": rng.uniform(-500.0, 500.0, rows),
        "Z_CART": rng.uniform(-500.0, 500.0, rows),
        "WEIGHT": np.ones(rows),
    }
    catalog.save_catalogue(tmp_path, "selected.npz", base, identity=identity)
    return {"path": str(tmp_path / "selected.npz"), "identity": identity}, base


def test_shard_v2_stores_only_the_shift_and_verifies_ordered_identities(tmp_path):
    base, original = _selected(tmp_path)
    shifted = dict(original)
    rng = np.random.default_rng(2)
    for column, key in (("X", "X_SHIFTED"), ("Y", "Y_SHIFTED"), ("Z_CART", "Z_CART_SHIFTED")):
        shifted[key] = original[column] + rng.normal(0.0, 8.0, len(original["X"]))
    identity = {**base["identity"], "stage": "reconstruct"}
    catalog.save_catalogue_reference(
        tmp_path, "shard.npz", identity=identity, base=base, shifted=shifted
    )
    with np.load(tmp_path / "shard.npz") as archive:
        assert archive.files == ["SHIFT"] and archive["SHIFT"].dtype == np.float32
    restored = catalog.read_catalogue(tmp_path / "shard.npz", identity=identity)
    for key in original:
        np.testing.assert_array_equal(restored[key], original[key])
    for key in ("X_SHIFTED", "Y_SHIFTED", "Z_CART_SHIFTED"):
        np.testing.assert_allclose(restored[key], shifted[key], rtol=0, atol=1e-4)
    partition = {
        "tracer": "LRG",
        "measurement_bin": "LRG2",
        "redshift_column": "Z",
        "zmin": 0.6,
        "zmax": 0.8,
    }
    subset = catalog.read_catalogue(tmp_path / "shard.npz", identity=identity, partition=partition)
    rows = (original["Z"] >= 0.6) & (original["Z"] < 0.8)
    np.testing.assert_array_equal(subset["SOURCE_ID"], original["SOURCE_ID"][rows])
    np.testing.assert_allclose(subset["X_SHIFTED"], shifted["X_SHIFTED"][rows], rtol=0, atol=1e-4)
    # A reordered base no longer matches the recorded identity digest.
    reordered = {key: value[::-1] for key, value in original.items()}
    catalog.save_catalogue(tmp_path, "selected.npz", reordered, identity=base["identity"])
    with pytest.raises(ValueError, match="source identities"):
        catalog.read_catalogue(tmp_path / "shard.npz", identity=identity)


def test_core_siblings_reuse_one_selection_and_reject_selection_changes(monkeypatch):
    runs = sensitivity_run_configs(load_analysis_config())
    source = {"config": runs["scan--R15"]}
    monkeypatch.setattr(selection, "read_catalog_registration", lambda *args: {"sources": ["same"]})
    for name in ("scan--R25", "scan--R45", "reciso--R15"):
        selection.validate_selection_reuse(source, {"config": runs[name]}, "LRG")
    changed = copy.deepcopy(runs["scan--R25"])
    changed["selection"]["radial_randoms"]["stable_thinning_seed"] += 1
    with pytest.raises(ValueError, match="selection reuse changes"):
        selection.validate_selection_reuse(source, {"config": changed}, "LRG")
    with pytest.raises(ValueError, match="transfer or core sensitivity"):
        selection.validate_selection_reuse(
            {"config": load_analysis_config()}, {"config": runs["scan--R25"]}, "LRG"
        )


def test_reservations_fill_planned_components_so_large_writers_run_together():
    terabyte = 10**12
    quantities = dict(
        existing_allocated_bytes=terabyte // 20,
        remaining_components={
            **dict.fromkeys(storage.TRANSFER_STORAGE_COMPONENTS, 0),
            "selected_catalogues": 1440 * 10**9,
            "shifted_shards": 850 * 10**9,
        },
        bytes_per_writer=900 * 10**9,
        quota_headroom_bytes=3 * 1024**4,
        filesystem_headroom_bytes=3_400 * 10**9,
        requested_concurrency=2,
    )
    plan = storage.transfer_storage_plan(**quantities)
    assert plan["max_concurrent"] == 2
    assert plan["conservative_peak_bytes"] == terabyte // 20 + 2290 * 10**9
    with pytest.raises(ValueError, match="complete campaign peak exceeds the verified storage"):
        storage.transfer_storage_plan(**{**quantities, "quota_headroom_bytes": 2289 * 10**9})


def test_campaign_writers_may_use_any_accounting_root(tmp_path):
    from tests.test_grouped_transfer import resource_budget

    transfer, core = tmp_path / "reconstruction-transfer", tmp_path / "core-sensitivity"
    transfer.mkdir()
    core.mkdir()
    budget = resource_budget(transfer)
    budget["accounting_roots"] = [str(transfer), str(core)]
    storage.validate_transfer_storage_budget(
        budget, stage="reconstruct", outputs=[core / "scan--R25"]
    )
    with pytest.raises(ValueError, match="leaves the accounted campaign roots"):
        storage.validate_transfer_storage_budget(
            budget, stage="reconstruct", outputs=[tmp_path / "elsewhere"]
        )


@pytest.fixture
def core_runs(tmp_path, monkeypatch):
    source_config = schema13()
    monkeypatch.setattr(driver, "load_analysis_config", lambda: copy.deepcopy(source_config))
    monkeypatch.setattr(
        driver, "_code_identity", lambda: {"code_commit": "a" * 40, "repository": str(tmp_path)}
    )
    source = driver.create_run(tmp_path, "core", workload="pilot", catalog_root=tmp_path)
    provider = driver.create_run(
        tmp_path,
        "bgs",
        workload="pilot",
        catalog_root=tmp_path,
        source_run=source,
        inference_definition=BGS_ANISOTROPIC,
    )
    return source, provider


def _publish_covariance_route(run, route):
    (run["root"] / "covariance").mkdir()
    driver._write_yaml(
        run["root"] / "covariance" / "covariances.yaml",
        {"stage": "covariance", "units": [{"unit": route, "path": "routes/x/result.yaml"}]},
    )


def test_power3_refit_reads_lrg_from_source_and_bgs_from_its_provider(core_runs):
    source, provider = core_runs
    refit = driver.create_run(
        source["root"].parent,
        "power3",
        workload="pilot",
        catalog_root=source["root"].parent,
        source_run=source,
        inference_definition=PRIMARY_POWER3,
        covariance_runs=[provider["root"]],
    )
    reloaded = driver.load_run(refit["root"])
    assert reloaded["metadata"]["covariance_runs"] == [driver._source_identity(provider)]
    assert reloaded["config"]["fitting"]["broadband_model"] == "power3"
    lrg = {
        "tracer": "LRG",
        "measurement_bin": "LRG3",
        "condition": "sigma_0.003",
        "reconstruction_state": "postreconstruction",
    }
    bgs = {**lrg, "tracer": "BGS_BRIGHT-21.5", "measurement_bin": "BGS1"}
    assert covariance.likelihood_covariance_run(reloaded, lrg)["root"] == source["root"]
    with pytest.raises(ValueError, match="exactly one anisotropic covariance provider"):
        covariance.likelihood_covariance_run(reloaded, bgs)
    _publish_covariance_route(provider, bgs)
    assert covariance.likelihood_covariance_run(reloaded, bgs)["root"] == provider["root"]
    with pytest.raises(ValueError, match="BGS anisotropic run of this source"):
        driver.create_run(
            source["root"].parent,
            "wrong",
            workload="pilot",
            catalog_root=source["root"].parent,
            source_run=source,
            inference_definition=PRIMARY_POWER3,
            covariance_runs=[source["root"]],
        )
    with pytest.raises(ValueError, match="core power3 inference"):
        driver.create_run(
            source["root"].parent,
            "plain",
            workload="pilot",
            catalog_root=source["root"].parent,
            source_run=source,
            inference_definition=BGS_ANISOTROPIC,
            covariance_runs=[provider["root"]],
        )
