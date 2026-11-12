"""Stage boundaries, source provenance and independently selected clustering statistics."""

import copy
import fcntl
import hashlib
import io
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from click.testing import CliRunner

from pzbao import catalog, clustering, reconstruction, selection, run as driver
from pzbao.config import load_analysis_config as load_current_config
from tests.legacy_config import load_analysis_config
from pzbao.config.stages import experiment_document, stage_document, read_stage_config


def small_catalogue():
    z = np.array([0.82, 0.87, 0.93, 0.98])
    return {
        "SOURCE_ID": np.arange(2**63, 2**63 + 4, dtype=np.uint64),
        "Z": z,
        "Z_OBS": z.copy(),
        "X": 100 * z,
        "Y": 50 * z,
        "Z_CART": 70 * z,
        "WEIGHT": np.ones(4),
        "WEIGHT_FKP": np.array([0.7, 0.8, 0.8, 0.9]),
        "PHOTSYS": np.array(["N"] * 4),
    }


def stage_run(tmp_path):
    config = load_analysis_config()
    return {
        "root": tmp_path,
        "config": config,
        "metadata": {
            "run_name": "fixture",
            "code_commit": "abc",
            "interface_version": 2,
            "analysis_config_sha256": "fixed",
            "workload": "pilot",
        },
    }


def parent_unit():
    return {
        "unit_id": "LRG--sigma_0.003--mock-0",
        "tracer": "LRG",
        "condition": "sigma_0.003",
        "realization_id": 0,
        "measurement_bins": ["LRG3"],
    }


def test_catalogue_roundtrip_retains_every_column_and_rejects_changed_identity(tmp_path):
    data = small_catalogue()
    data["X_SHIFTED"] = data["X"] + 0.1
    identity = {"tracer": "LRG", "role": "data", "stage": "reconstruct"}
    catalog.save_catalogue(tmp_path, "data.npz", data, identity=identity)
    decoded = catalog.read_catalogue(tmp_path / "data.npz", identity=identity)
    for key in data:
        np.testing.assert_array_equal(decoded[key], data[key])
        assert decoded[key].dtype == data[key].dtype
    with pytest.raises(ValueError, match="different stage"):
        catalog.read_catalogue(tmp_path / "data.npz", identity={**identity, "role": "random"})
    data["SOURCE_ID"][1] = data["SOURCE_ID"][0]
    with pytest.raises(ValueError, match="unique"):
        catalog.save_catalogue(tmp_path, "duplicate.npz", data, identity=identity)


def test_download_validation_is_atomic_and_never_replaces_existing_source(tmp_path, monkeypatch):
    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"fixture-fits"))
    monkeypatch.setattr(catalog, "read_fits_catalog", lambda path: {"Z": np.array([0.2])})
    target = tmp_path / "catalog.fits"
    with pytest.raises(ValueError, match="checksum"):
        catalog.download_catalogue(
            "https://example.invalid/source", target, expected_sha256="wrong"
        )
    assert not target.exists()
    assert not list(tmp_path.glob("*.partial"))
    catalog.download_catalogue("https://example.invalid/source", target)
    assert target.read_bytes() == b"fixture-fits"
    with pytest.raises(FileExistsError):
        catalog.download_catalogue("https://example.invalid/source", target)
    assert target.read_bytes() == b"fixture-fits"


def test_catalog_validation_does_not_download(tmp_path, monkeypatch):
    run = stage_run(tmp_path)
    run["metadata"]["catalog_root"] = str(tmp_path)
    run["config"]["analysis"]["mock_ids"] = [0]
    monkeypatch.setattr(catalog, "reservoir_slice_path", lambda *a: tmp_path / "reservoir")
    (tmp_path / "reservoir").write_text("fixture")
    monkeypatch.setattr(catalog, "planned_catalogue_path", lambda *a, **k: tmp_path / "absent.fits")
    monkeypatch.setattr(
        catalog, "download_catalogue", lambda *a, **k: pytest.fail("implicit network access")
    )
    monkeypatch.setattr(
        catalog,
        "register_parent_inputs",
        lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("missing")),
    )
    with pytest.raises(FileNotFoundError):
        catalog.catalog_stage(run, {"tracer": "LRG", "realization_ids": [0]}, tmp_path)


@pytest.mark.parametrize("save", [True, False])
def test_reconstruction_handoff_keeps_originals_and_honors_opt_out(tmp_path, monkeypatch, save):
    run, unit = stage_run(tmp_path), parent_unit()
    data = small_catalogue()
    # Shards reference the immutable selected catalogues saved by the select stage.
    selected = tmp_path / "select" / "units" / unit["tracer"]
    selected.mkdir(parents=True)
    entries = []
    for cap in ("NGC", "SGC"):
        for role in ("data", "random"):
            identity = catalog.catalogue_identity(
                run,
                tracer=unit["tracer"],
                condition=unit["condition"],
                mock_id=unit["realization_id"],
                cap=cap,
                role=role,
                stage="select",
            )
            name = f"{cap}--{role}.npz"
            catalog.save_catalogue(selected, name, data, identity=identity)
            entries.append({"path": name, "identity": identity})
    preparation = {
        "catalogues": entries,
        "geometry": {cap: {"family": {"cap": cap}} for cap in ("NGC", "SGC")},
        "effective_redshifts": {unit["condition"]: {"parent": {"growth_rate": 0.8}}},
        "conditions": {
            unit["condition"]: {
                cap: {
                    "selection_diagnostics_by_mock": [
                        {"realization_id": 0, "selection": {"bins": {}}}
                    ]
                }
                for cap in ("NGC", "SGC")
            }
        },
    }
    monkeypatch.setattr(
        selection,
        "read_selected_parent",
        lambda *a: (copy.deepcopy(data), copy.deepcopy(data), preparation),
    )
    calls = []

    def solve(galaxies, randoms, **kwargs):
        calls.append(kwargs["mask_family"]["cap"])
        for catalogue in (galaxies, randoms):
            for original, shifted in zip(("X", "Y", "Z_CART"), reconstruction.SHIFTED_COLUMNS):
                catalogue[shifted] = catalogue[original] + 1
        return galaxies, randoms, {"shell_diagnostics": {"shells": []}}

    monkeypatch.setattr(reconstruction, "run_reconstruction", solve)
    destination = tmp_path / "reconstruct" / "units" / unit["unit_id"]
    destination.mkdir(parents=True)
    result = reconstruction.reconstruct_stage(run, unit, destination, save_catalogs=save)
    driver._write_yaml(
        destination / "result.yaml",
        {**run["metadata"], **result, "unit": unit, "stage": "reconstruct"},
    )
    assert calls == ["NGC", "SGC"]
    if save:
        decoded, _, _ = reconstruction.read_reconstructed_parent(run, unit, "NGC")
        for key in data:
            np.testing.assert_array_equal(decoded[key], data[key])
        np.testing.assert_allclose(decoded["X_SHIFTED"], data["X"] + 1, rtol=0, atol=1e-4)
        # Only the float32 shift is stored; the selected columns are referenced.
        with np.load(destination / "NGC--random.npz") as archive:
            assert archive.files == ["SHIFT"] and archive["SHIFT"].dtype == np.float32
    else:
        assert not list(destination.glob("*.npz"))
        with pytest.raises(ValueError, match="no reusable catalogues"):
            reconstruction.read_reconstructed_parent(run, unit, "NGC")


@pytest.mark.parametrize(
    "statistics,xi_calls,pk_calls", [("xi", 2, 0), ("pk", 0, 2), ("both", 2, 2)]
)
def test_clustering_selectors_consume_selection_without_an_implicit_solve(
    tmp_path, monkeypatch, statistics, xi_calls, pk_calls
):
    run, unit = stage_run(tmp_path), parent_unit()
    run["config"]["analysis"]["workloads"]["pilot"]["reconstruction_states"] = ["prereconstruction"]
    monkeypatch.setattr(clustering, "read_preparation", lambda *a: {"power_geometry": {}})
    monkeypatch.setattr(
        selection, "read_selected_parent", lambda *a: (small_catalogue(), small_catalogue(), {})
    )
    monkeypatch.setattr(
        reconstruction,
        "read_reconstructed_parent",
        lambda *a: pytest.fail("pre-only clustering must not reconstruct"),
    )
    calls = {"xi": 0, "pk": 0}

    class Counts:
        attrs = None
        count_names = ("R1R2",)

        def __init__(self):
            self.attrs = {}
            self.R1R2 = SimpleNamespace(attrs={})

        def normalize(self, **kwargs):
            return self

        def save(self, path):
            Path(path).write_text("identified fixture")

    class Power:
        ells = [0, 2, 4]

        def copy(self):
            return self

        def select(self, *args):
            return self

        def __call__(self, **kwargs):
            return np.arange(1.0, 5.0), np.ones((3, 4))

    def count(*args, **kwargs):
        calls["xi"] += 1
        return Counts()

    def power(*args, **kwargs):
        calls["pk"] += 1
        return Power()

    monkeypatch.setattr(clustering, "compute_xi", count)
    monkeypatch.setattr(clustering, "compute_power_spectrum_secondary", power)
    monkeypatch.setattr(
        clustering, "combine_cap_pair_counts", lambda counts: copy.deepcopy(counts["NGC"])
    )
    monkeypatch.setattr(clustering, "combine_cap_power", lambda powers: powers["NGC"])
    result = clustering.cluster_stage(run, unit, tmp_path, nthreads=1, statistics=statistics)
    assert calls == {"xi": xi_calls, "pk": pk_calls}
    assert result["statistics"] == statistics
    manifest = json.loads((tmp_path / "measurements.json").read_text())
    assert manifest["statistics"] == statistics
    assert any("counts" in row for row in manifest["measurements"]) == (statistics != "pk")
    assert any(p["path"].endswith("power.npz") for p in result["products"]) == (statistics != "xi")


def test_stage_yaml_and_combined_yaml_resolve_identically(tmp_path):
    # Stage files fill omitted sections from the current authority, so this round trip
    # exercises the schema-17 definition rather than the historical broad-field fixture.
    config = load_current_config()
    combined = experiment_document(config)
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(combined))
    assert load_current_config(path) == config
    for stage in driver.STAGES:
        document = stage_document(config, stage)
        stage_path = tmp_path / (stage + ".yaml")
        stage_path.write_text(yaml.safe_dump(document))
        assert load_current_config(stage_path) == config
        assert read_stage_config(stage_path, stage=stage, baseline=config) == (
            {"statistics": "both"} if stage == "cluster" else {}
        )
    document = stage_document(config, "cluster")
    document["cluster"]["statistics"] = "pk"
    path.write_text(yaml.safe_dump(document))
    assert read_stage_config(path, stage="cluster", baseline=config) == {"statistics": "pk"}
    document["cluster"]["measurement"].pop("xi")
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError, match="missing|complete"):
        load_current_config(path)


def test_statistics_override_is_recorded_and_preserved_in_scheduler_plan(tmp_path):
    run = stage_run(tmp_path)
    document = stage_document(run["config"], "cluster")
    document["cluster"]["statistics"] = "pk"
    path = tmp_path / "stage.yaml"
    path.write_text(yaml.safe_dump(document))
    assert driver._resolve_stage_options(run, "cluster", path, None, True) == {"statistics": "pk"}
    assert driver._resolve_stage_options(run, "cluster", path, "xi", True) == {"statistics": "xi"}
    plan = driver.slurm_plan(
        "cluster",
        [parent_unit()],
        input_path=tmp_path / "selection.yaml",
        cpus=1,
        memory="1G",
        walltime="00:05:00",
        account="fixture",
        config_path=path,
        statistics="pk",
    )
    assert "--statistics pk" in plan and "--config" in plan
    assert "--partition" not in plan and "--qos" not in plan


def test_scratch_creation_records_dirty_code_and_rejects_changed_resume(tmp_path, monkeypatch):
    state = {
        "code_commit": "abc",
        "repository": str(tmp_path),
        "dirty": True,
        "code_state_sha256": "one",
        "source_state": {"patch": "edited"},
    }
    monkeypatch.setattr(driver, "_code_identity", lambda **kwargs: dict(state))
    run = driver.create_run(
        tmp_path, "experiment", workload="pilot", catalog_root=tmp_path, scratch=True
    )
    assert run["root"] == tmp_path / "scratch" / "experiment"
    assert run["metadata"]["mode"] == "scratch"
    assert run["metadata"]["source_state"]["patch"] == "edited"

    def science(stage, run, unit, destination, cpus):
        (destination / "fixture.json").write_text("{}")
        return {"products": [{"path": "fixture.json", "format": "json"}]}

    monkeypatch.setattr(driver, "_science", science)
    assert not driver.execute_stage(run, "catalog", selection="0")["errors"]
    assert not driver.execute_stage(run, "catalog", selection="0", resume=True)["errors"]
    state["code_state_sha256"] = "two"
    result = driver.execute_stage(run, "catalog", selection="0", resume=True)
    assert "scratch source changed" in result["errors"][0]["error"]
    with pytest.raises(ValueError, match="production"):
        driver.create_run(
            tmp_path / "scratch", "production", workload="pilot", catalog_root=tmp_path
        )


def test_cli_exposes_eight_stages_and_named_transfer_preparation_commands():
    preparation = {
        "transfer-matrix",
        "transfer-yamls",
        "transfer-centre-contexts",
        "transfer-centres",
        "sensitivity-yamls",
        "per-bin-yamls",
        "density-handoff",
    }
    assert driver.STAGES == (
        "catalog",
        "select",
        "reconstruct",
        "cluster",
        "covariance",
        "fit",
        "sample",
        "review",
    )
    operations = {"adopt-predecessor"}  # moves a predecessor's published units into a successor
    # The centres review report reads an unfrozen candidate and renders it; it runs no stage.
    reports = {"transfer-centres-report"}
    assert set(driver.main.commands) == set(driver.STAGES) | preparation | operations | reports
    assert (
        tuple(
            name for name in driver.main.commands if name not in preparation | operations | reports
        )
        == driver.STAGES
    )
    cli = CliRunner()
    for command in preparation | operations | reports:
        assert cli.invoke(driver.main, [command, "--help"]).exit_code == 0
    for old in ("inputs", "measure"):
        assert cli.invoke(driver.main, [old]).exit_code != 0
    assert "--no-save-catalogs" in cli.invoke(driver.main, ["reconstruct", "--help"]).output


def test_download_checks_sidecar_and_empty_catalogue_before_publication(tmp_path, monkeypatch):
    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"fixture"))
    monkeypatch.setattr(catalog, "read_fits_catalog", lambda path: {"Z": np.array([0.2])})
    target = tmp_path / "catalog.fits"
    target.with_name(target.name + ".sha256").write_text("a" * 64)
    with pytest.raises(ValueError, match="checksum"):
        catalog.download_catalogue("https://example.invalid/source", target)
    assert not target.exists()
    monkeypatch.setattr(catalog, "read_fits_catalog", lambda path: {"Z": np.array([])})
    with pytest.raises(ValueError, match="empty"):
        catalog.download_catalogue("https://example.invalid/source", target)
    assert not target.exists()


def test_power_only_handoff_can_be_reviewed_without_xi(tmp_path, monkeypatch):
    from pzbao.diagnostics.review import collect_route

    monkeypatch.setitem(sys.modules, "pycorr", None)
    run, unit = stage_run(tmp_path), parent_unit()
    workload = run["config"]["analysis"]["workloads"]["pilot"]
    workload["mock_ids"] = [0]
    workload["reconstruction_states"] = ["prereconstruction"]
    monkeypatch.setattr(clustering, "read_preparation", lambda *a: {"power_geometry": {}})
    monkeypatch.setattr(
        selection, "read_selected_parent", lambda *a: (small_catalogue(), small_catalogue(), {})
    )

    class Power:
        ells = [0, 2, 4]

        def copy(self):
            return self

        def select(self, *a):
            return self

        def __call__(self, **k):
            return np.arange(1.0, 5.0), np.ones((3, 4))

    monkeypatch.setattr(clustering, "compute_power_spectrum_secondary", lambda *a, **k: Power())
    monkeypatch.setattr(clustering, "combine_cap_power", lambda powers: powers["NGC"])
    destination = tmp_path / "cluster" / "units" / unit["unit_id"]
    destination.mkdir(parents=True)
    result = clustering.cluster_stage(run, unit, destination, nthreads=1, statistics="pk")
    driver._write_yaml(
        destination / "result.yaml", {**run["metadata"], **result, "unit": unit, "stage": "cluster"}
    )
    route = {key: unit[key] for key in ("tracer", "condition")}
    route.update(measurement_bin="LRG3", reconstruction_state="prereconstruction")
    records = clustering.read_measurements(run, route, statistic="pk")
    assert len(records) == 3
    np.testing.assert_array_equal(records[-1]["power_arrays"]["power"], np.ones((3, 4)))
    with pytest.raises(ValueError, match="requires xi"):
        clustering.read_measurements(run, route)
    report, _ = collect_route(run, route)
    assert report["coverage"]["power"] == "complete"
    assert report["coverage"]["measurements"] != "complete"
    np.testing.assert_array_equal(report["power"]["power"], np.ones((3, 4)))
    path = destination / records[-1]["power"]
    np.savez(path, k=np.array([1.0, 1.0, 2.0, 3.0]), power=np.ones((3, 4)), ells=[0, 2, 4])
    with pytest.raises(ValueError, match="axes"):
        clustering.read_measurements(run, route, statistic="pk")


def test_review_does_not_require_power_for_an_unconfigured_bin(tmp_path, monkeypatch):
    from pzbao.diagnostics.review import collect_route

    run = stage_run(tmp_path)
    route = {
        "tracer": "LRG",
        "condition": "native",
        "measurement_bin": "LRG2",
        "reconstruction_state": "prereconstruction",
    }
    requested_statistics = []

    def measurements(current, selected, *, statistic="xi"):
        requested_statistics.append(statistic)
        return []

    monkeypatch.setattr(clustering, "read_measurements", measurements)
    report, _ = collect_route(run, route)
    assert requested_statistics == ["xi"]
    assert report["coverage"]["power"] == "not configured for this bin"
    assert "power" not in report


def test_scratch_stage_retains_failed_attempt_if_source_changes_mid_execution(
    tmp_path, monkeypatch
):
    state = {
        "code_commit": "abc",
        "repository": str(tmp_path),
        "dirty": True,
        "code_state_sha256": "before",
        "source_state": {"patch": "initial"},
    }
    monkeypatch.setattr(driver, "_code_identity", lambda **kw: copy.deepcopy(state))
    run = driver.create_run(
        tmp_path, "scratch-edit", workload="pilot", catalog_root=tmp_path, scratch=True
    )

    def edit_during_science(stage, run, unit, destination, cpus):
        state["code_state_sha256"] = "after"
        return {"products": []}

    monkeypatch.setattr(driver, "_science", edit_during_science)
    result = driver.execute_stage(run, "catalog", selection="0")
    assert "source changed during execution" in result["errors"][0]["error"]
    assert not list((run["root"] / "catalog" / "units").glob("*/result.yaml"))
    assert list((run["root"] / "catalog" / "attempts").glob("*/failure.txt"))


def test_inference_from_scratch_cannot_become_production_implicitly(tmp_path, monkeypatch):
    state = {
        "code_commit": "abc",
        "repository": str(tmp_path),
        "dirty": True,
        "code_state_sha256": "one",
        "source_state": {"patch": "edited"},
    }
    monkeypatch.setattr(driver, "_code_identity", lambda **kw: copy.deepcopy(state))
    source = driver.create_run(
        tmp_path, "original", workload="pilot", catalog_root=tmp_path, scratch=True
    )
    with pytest.raises(ValueError, match="remain in scratch"):
        driver.create_run(
            tmp_path, "production", workload="pilot", catalog_root=tmp_path, source_run=source
        )
    with pytest.raises(ValueError, match="scratch directory"):
        driver._inference_destination(
            source, tmp_path / "production", definition=None, resume=False, create=True
        )
    derived = driver._inference_destination(
        source, tmp_path / "scratch" / "inference", definition=None, resume=False, create=True
    )
    assert derived["metadata"]["mode"] == "scratch"
    assert derived["root"] == tmp_path / "scratch" / "inference"


CLEAN_CODE = {
    "code_commit": "abc",
    "repository": "fixture",
    "dirty": False,
    "code_state_sha256": "clean",
}
PLAN = ["--scheduler", "slurm", "--account", "fixture", "--memory", "1G", "--walltime", "1:00"]


def test_after_renders_a_dependent_array_and_rejects_non_job_ids(tmp_path):
    common = dict(input_path=tmp_path / "selection.yaml", cpus=1, memory="1G", walltime="1:00")
    plan = driver.slurm_plan(
        "cluster", [parent_unit()], account="x", after=("12", "34_1"), **common
    )
    assert "#SBATCH --dependency=afterok:12:34_1" in plan
    assert "#SBATCH --kill-on-invalid-dep=yes" in plan
    assert "--dependency" not in driver.slurm_plan(
        "cluster", [parent_unit()], account="x", **common
    )
    with pytest.raises(ValueError, match="numeric Slurm job IDs"):
        driver.slurm_plan("cluster", [parent_unit()], account="x", after=("12;rm",), **common)


def test_pending_handoff_is_planned_only_with_after_inside_an_existing_run(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "_code_identity", lambda **kwargs: dict(CLEAN_CODE))
    run = driver.create_run(tmp_path, "chain", workload="pilot", catalog_root=tmp_path)
    pending = run["root"] / "catalog" / "catalog.yaml"
    cli = CliRunner()
    planned = cli.invoke(driver.main, ["select", "--input", str(pending), *PLAN, "--after", "7"])
    assert planned.exit_code == 0, planned.output
    assert "afterok:7" in planned.output and f"--input {pending}" in planned.output
    assert not pending.exists()
    for arguments, message in (
        (["--input", str(pending), *PLAN], "does not exist"),
        (["--input", str(pending), "--after", "7"], "only plans dependent Slurm"),
        (
            ["--input", str(run["root"] / "select" / "catalog.yaml"), *PLAN, "--after", "7"],
            "upstream",
        ),
        (
            [
                "--input",
                str(tmp_path / "missing" / "catalog" / "catalog.yaml"),
                *PLAN,
                "--after",
                "7",
            ],
            "",
        ),
    ):
        result = cli.invoke(driver.main, ["select", *arguments])
        assert result.exit_code != 0 and message in str(result.output) + str(result.exception)


def test_catalog_plan_freezes_one_snapshot_for_dependent_planning(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "_code_identity", lambda **kwargs: dict(CLEAN_CODE))
    arguments = ["catalog", "--run-name", "chain", "--output-root", str(tmp_path)]
    arguments += ["--catalog-root", str(tmp_path), "--workload", "pilot", *PLAN]
    cli = CliRunner()
    assert not (tmp_path / "chain").exists()
    first = cli.invoke(driver.main, arguments)
    assert first.exit_code == 0 and not (tmp_path / "chain").exists()
    created = cli.invoke(driver.main, [*arguments, "--create-snapshot", "--after", "5"])
    assert created.exit_code == 0, created.output
    assert "--resume" in created.output and "afterok:5" in created.output
    snapshot = (tmp_path / "chain" / "run.yaml").read_bytes()
    again = cli.invoke(driver.main, [*arguments, "--create-snapshot"])
    assert again.exit_code == 0 and (tmp_path / "chain" / "run.yaml").read_bytes() == snapshot
    local = cli.invoke(driver.main, arguments[:9] + ["--create-snapshot"])
    assert local.exit_code != 0 and "only apply to Slurm plans" in local.output


def test_tracer_filter_applies_only_to_plans_and_all_group_plans():
    base = ["reconstruct", "--input", "x/select/selection.yaml", "--tracer", "LRG"]
    for arguments in (base, [*base, "--source-group", "LRG--source-x--R25--mock-0", *PLAN]):
        result = CliRunner().invoke(driver.main, arguments)
        assert result.exit_code != 0 and "--tracer filters" in result.output


def test_tracer_filtered_plans_keep_only_that_tracers_units(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "_code_identity", lambda **kwargs: dict(CLEAN_CODE))
    run = driver.create_run(tmp_path, "chain", workload="pilot", catalog_root=tmp_path)
    pending = run["root"] / "select" / "selection.yaml"
    cli = CliRunner()
    for tracer in ("LRG", "BGS_BRIGHT-21.5"):
        result = cli.invoke(
            driver.main,
            ["reconstruct", "--input", str(pending), "--tracer", tracer, *PLAN, "--after", "3"],
        )
        assert result.exit_code == 0, result.output
        line = next(row for row in result.output.splitlines() if row.startswith("unit_ids=("))
        units = line[len("unit_ids=(") : -1].split()
        assert units and all(name.startswith(tracer + "--") for name in units)


def test_stage_order_and_handoff_names_have_one_authority():
    """run.py reads stage order and handoff names from one place."""
    from pzbao.config import stages

    assert driver.STAGES is stages.STAGES
    assert driver.HANDOFFS is stages.HANDOFFS
    assert set(stages.HANDOFFS) == set(stages.STAGES) - {"review"}
    for stage, name in stages.HANDOFFS.items():
        assert stages.handoff_path(Path("/run"), stage) == Path("/run") / stage / name
    with pytest.raises(ValueError, match="publishes no handoff"):
        stages.handoff_path(Path("/run"), "review")


# ------------------------------------------------------------------ successor adoption


def _publish_predecessor(stage, run, unit, destination, cpus):
    """Write the real native products of one unit so the stage readers can cross the handoff."""
    from tests.test_boundary_support import per_bin_preparation_fields

    tracer, config = unit["tracer"], run["config"]
    if stage == "catalog":
        payload = {"tracer": tracer, "realization_ids": unit["realization_ids"]}
        payload["sources"] = [{"path": f"{tracer}.fits", "sha256": "0" * 64}]
        (destination / "catalog.json").write_text(json.dumps(payload))
        return {"products": [{"path": "catalog.json", "format": "json"}], **payload}
    if stage == "select":
        products, entries = [], []
        for cap in ("NGC", "SGC"):
            for role in ("data", "random"):
                identity = catalog.catalogue_identity(
                    run,
                    tracer=tracer,
                    condition="sigma_0.003",
                    mock_id=0,
                    cap=cap,
                    role=role,
                    stage="select",
                )
                name = f"sigma_0.003--mock-0--{cap}--{role}.npz"
                catalog.save_catalogue(destination, name, small_catalogue(), identity=identity)
                products.append({"path": name, "format": "catalogue", "identity": identity})
                entries.append({"path": name, "identity": identity})
        preparation = {
            "tracer": tracer,
            "realization_ids": unit["realization_ids"],
            "catalogues": entries,
            **per_bin_preparation_fields(config, tracer),
        }
        (destination / "preparation.json").write_text(json.dumps(preparation))
        return {
            "products": [{"path": "preparation.json", "format": "json"}, *products],
            "tracer": tracer,
            "realization_ids": unit["realization_ids"],
        }
    if stage == "reconstruct":
        name = unit["measurement_bins"][0]
        products, entries = [], []
        for cap in ("NGC", "SGC"):
            for role in ("data", "random"):
                base = selection.selected_catalogue_entry(run, unit, cap, role)
                partition = catalog.catalogue_partition(tracer, name, role, config)
                original = catalog.read_catalogue(
                    base["path"], identity=base["identity"], partition=partition
                )
                shifted = dict(original)
                for offset, (column, moved) in enumerate(
                    zip(("X", "Y", "Z_CART"), reconstruction.SHIFTED_COLUMNS), 1
                ):
                    shifted[moved] = original[column] + offset
                identity = catalog.catalogue_identity(
                    run,
                    tracer=tracer,
                    condition=unit["condition"],
                    mock_id=unit["realization_id"],
                    cap=cap,
                    role=role,
                    stage="reconstruct",
                )
                path = f"{cap}--{name}--{role}.npz"
                products.extend(
                    catalog.save_catalogue_reference(
                        destination,
                        path,
                        identity=identity,
                        base=base,
                        partition=partition,
                        shifted=shifted,
                    )
                )
                entries.append({"path": path, "identity": identity, "partition": partition})
        payload = {
            "diagnostics": {
                cap: {"solver": {"commit": run["metadata"]["code_commit"]}}
                for cap in ("NGC", "SGC")
            },
            "catalogues": entries,
            "catalogues_saved": True,
            "representation": "bin_shards_v1",
        }
        (destination / "reconstruction.json").write_text(json.dumps(payload))
        return {
            "products": [*products, {"path": "reconstruction.json", "format": "json"}],
            "catalogues_saved": True,
        }
    assert stage == "cluster"
    name = unit["measurement_bins"][0]
    power = f"{name}--postreconstruction--power.npz"
    ells = config["measurement"]["power_spectrum_secondary"]["multipoles"]
    np.savez(
        destination / power,
        k=np.arange(1.0, 5.0),
        power=np.full((len(ells), 4), 1.0 + unit["realization_id"]),
        ells=ells,
    )
    measurements = [
        {
            "identity": {
                "tracer": tracer,
                "condition": unit["condition"],
                "measurement_bin": name,
                "reconstruction_state": "postreconstruction",
                "realization_id": unit["realization_id"],
                "cap": cap,
                "analysis_config_sha256": run["metadata"]["analysis_config_sha256"],
            },
            "power": power,
            "units": "Mpc/h",
            "coordinate_state": "shifted",
        }
        for cap in ("NGC", "SGC", "NGC+SGC")
    ]
    solved = run["root"] / "reconstruct/units" / unit["unit_id"] / "reconstruction.json"
    if solved.is_file():
        # As the real stage does: the randoms of a measured cap are referenced by absolute
        # path and identity into this run's reconstruction unit, never copied.
        for record in measurements[:2]:
            randoms = clustering.measurement_random_reference(
                run, unit, record["identity"]["cap"], name, reconstructed=True
            )
            rows = catalog.read_catalogue(
                randoms["path"], identity=randoms["identity"], partition=randoms["partition"]
            )
            record.update(randoms=randoms, random_count=len(rows["SOURCE_ID"]))
    (destination / "measurements.json").write_text(
        json.dumps({"measurements": measurements, "statistics": "pk"})
    )
    return {
        "products": [
            {"path": power, "format": "npz", "arrays": ["k", "power"]},
            {"path": "measurements.json", "format": "json"},
        ],
        "statistics": "pk",
    }


@pytest.fixture
def successor_pair(tmp_path, monkeypatch):
    """A predecessor published at commit a and its empty successor created at commit b.

    The predecessor publishes LRG's catalogue registration, selection (mock 0), one bin-shard
    reconstruction unit and every LRG clustering unit; its BGS clustering units stay
    unpublished so the successor has something to compute.
    """
    from tests.test_run import synthetic_scalar_grid

    synthetic_scalar_grid(monkeypatch)
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    code = {"code_commit": "a" * 40, "repository": str(tmp_path / "producer-a")}
    monkeypatch.setattr(driver, "_code_identity", lambda **kwargs: dict(code))
    real_science = driver._science
    monkeypatch.setattr(driver, "_science", _publish_predecessor)
    predecessor = driver.create_run(
        tmp_path, "predecessor", workload="pilot", catalog_root=tmp_path
    )
    config = predecessor["config"]
    units = {stage: driver.stage_units(stage, config, "pilot") for stage in driver.STAGES[:4]}
    lrg = {
        stage: [unit for unit in items if unit["tracer"] == "LRG"] for stage, items in units.items()
    }
    for stage in ("catalog", "select"):
        assert not driver.execute_stage(predecessor, stage, selection="LRG")["errors"]
    recon = lrg["reconstruct"][0]
    assert not driver.execute_stage(predecessor, "reconstruct", selection=recon["unit_id"])[
        "errors"
    ]
    for unit in lrg["cluster"]:
        assert not driver.execute_stage(predecessor, "cluster", selection=unit["unit_id"])["errors"]
    monkeypatch.setattr(driver, "_science", real_science)
    code.update(code_commit="b" * 40, repository=str(tmp_path / "producer-b"))
    successor = driver.create_run(
        tmp_path,
        "successor",
        workload="pilot",
        catalog_root=tmp_path,
        predecessor=predecessor["root"],
    )
    return SimpleNamespace(
        predecessor=predecessor, successor=successor, code=code, units=units, lrg=lrg, root=tmp_path
    )


def _handoff(run, stage):
    return run["root"] / stage / driver.HANDOFFS[stage]


def _tree(root):
    """(path -> inode, mtime) of every file under a run; adoption must leave non-products as is."""
    return {p: (p.stat().st_ino, p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()}


def _resume(run, stage, selection=None):
    """Execute one stage with --resume under the options a plain CLI execution records."""
    run = dict(run)
    run["stage_options"] = driver._adopted_stage_options(run, stage)
    return driver.execute_stage(run, stage, selection=selection, resume=True)


def test_successor_records_its_predecessor_and_refuses_a_different_definition(successor_pair):
    pair = successor_pair
    predecessor, successor = pair.predecessor, pair.successor
    assert successor["metadata"]["predecessor"] == driver._source_identity(predecessor)
    assert (
        successor["metadata"]["code_commit"] == "b" * 40 != predecessor["metadata"]["code_commit"]
    )
    assert (
        successor["metadata"]["analysis_config_sha256"]
        == predecessor["metadata"]["analysis_config_sha256"]
    )
    loaded = driver.load_run(successor["root"])
    assert loaded["predecessor"]["root"] == predecessor["root"]
    common = dict(workload="pilot", catalog_root=pair.root, predecessor=predecessor["root"])
    changed = copy.deepcopy(predecessor["config"])
    changed["fitting"]["sampling"]["seed"] += 1
    path = pair.root / "changed.yaml"
    path.write_text(yaml.safe_dump(changed))
    with pytest.raises(ValueError, match="differs from its predecessor"):
        driver.create_run(pair.root, "changed", config_path=path, **common)
    with pytest.raises(ValueError, match="workload and catalog root"):
        driver.create_run(pair.root, "workload", **{**common, "workload": "full_grid"})
    (pair.root / "elsewhere").mkdir()
    with pytest.raises(ValueError, match="workload and catalog root"):
        driver.create_run(
            pair.root, "inputs", **{**common, "catalog_root": pair.root / "elsewhere"}
        )
    with pytest.raises(ValueError, match="outside its predecessor"):
        driver.create_run(predecessor["root"] / "nested", "inside", **common)
    with pytest.raises(ValueError, match="inference run registers nothing"):
        driver.create_run(pair.root, "inference", source_run=predecessor, **common)
    # A production successor never adopts scratch products.
    scratch = driver.create_run(
        pair.root, "exploratory", workload="pilot", catalog_root=pair.root, scratch=True
    )
    with pytest.raises(ValueError, match="scratch products"):
        driver.create_run(
            pair.root,
            "promoted",
            workload="pilot",
            catalog_root=pair.root,
            predecessor=scratch["root"],
        )
    # Every later load re-checks the predecessor: changed identity or inputs fail closed.
    metadata_path = predecessor["root"] / "run.yaml"
    original = metadata_path.read_text()
    for field, value in (("code_commit", "c" * 40), ("catalog_root", str(pair.root / "elsewhere"))):
        record = yaml.safe_load(original)
        record[field] = value
        metadata_path.write_text(yaml.safe_dump(record))
        with pytest.raises(ValueError, match="predecessor identity or snapshot changed"):
            driver.load_run(successor["root"])
    metadata_path.write_text(original)
    assert driver.load_run(successor["root"])["predecessor"]["root"] == predecessor["root"]


def test_adoption_moves_products_through_the_real_readers_and_resume_measures_only_the_gaps(
    successor_pair, monkeypatch
):
    from pzbao.catalog import read_catalog_registration
    from pzbao.clustering import read_measurements
    from pzbao.reconstruction import read_reconstructed_parent
    from pzbao.selection import read_preparation, read_selected_parent

    pair = successor_pair
    predecessor, successor = pair.predecessor, pair.successor
    identity = {key: successor["metadata"][key] for key in driver._IDENTITY_KEYS}
    predecessor_identity = driver._source_identity(predecessor)
    registration = read_catalog_registration(predecessor, "LRG")
    before = _tree(predecessor["root"])
    catalogue_bytes = (predecessor["root"] / "catalog/units/LRG/catalog.json").read_bytes()
    # Catalogue: the one published unit moves (same inode), the other stays for the stage.
    summary = driver.adopt_predecessor(successor, "catalog")
    assert summary["units_adopted"] == ["LRG"] and summary["units_computed"] == ["BGS_BRIGHT-21.5"]
    assert summary["files_moved"] == 1 and summary["bytes_moved"] == len(catalogue_bytes)
    receipt = json.loads((successor["root"] / "catalog/adopted.json").read_text())
    assert receipt["predecessor"] == predecessor_identity
    assert receipt["handoff"] == str(_handoff(predecessor, "catalog"))
    assert (
        receipt["handoff_sha256"]
        == hashlib.sha256(_handoff(predecessor, "catalog").read_bytes()).hexdigest()
    )
    assert receipt["units_adopted"] == ["LRG"] and receipt["files_moved"] == 1
    assert receipt["units_previously_adopted"] == [] and receipt["siblings"] == []
    assert receipt["execution_code"]["code_commit"] == "b" * 40
    assert receipt["adopted_at"][:4] == "2026" or "T" in receipt["adopted_at"]
    moved = successor["root"] / "catalog/units/LRG/catalog.json"
    origin = predecessor["root"] / "catalog/units/LRG/catalog.json"
    assert moved.read_bytes() == catalogue_bytes and not origin.exists()
    assert moved.stat().st_ino == before[origin][0]
    record = driver._unit_result(successor, "catalog", pair.lrg["catalog"][0])
    assert {key: record[key] for key in identity} == identity
    assert record["execution_code"]["code_commit"] == "b" * 40
    assert record["stage_options"] == {"download": False}
    registered = record["registered_from"]
    assert registered["run_name"] == "predecessor" and registered["code_commit"] == "a" * 40
    assert registered["root"] == str(predecessor["root"]) and registered["stage"] == "catalog"
    assert registered["execution_code"]["code_commit"] == "a" * 40
    assert registered["stage_options"] == {}  # the predecessor's own options, as it published them
    assert registered["handoff_sha256"] == receipt["handoff_sha256"]
    assert (
        registered["result_sha256"]
        == hashlib.sha256(
            (predecessor["root"] / "catalog/units/LRG/result.yaml").read_bytes()
        ).hexdigest()
    )
    assert registered["files"] == {"moved": 1}
    assert read_catalog_registration(successor, "LRG") == registration
    # The predecessor is retired: its handoff, result and run.yaml stay, its product is gone.
    assert (predecessor["root"] / "catalog/units/LRG/result.yaml").is_file()
    after = _tree(predecessor["root"])
    assert after == {path: value for path, value in before.items() if path != origin}
    with pytest.raises(ValueError, match="missing or invalid scientific product"):
        driver._unit_result(predecessor, "catalog", pair.lrg["catalog"][0])
    with pytest.raises(FileExistsError, match="already adopted"):
        driver.adopt_predecessor(successor, "catalog")
    # A --resume execution with the stage's plain options skips the unit and publishes.
    handoff = _resume(successor, "catalog", selection="LRG")
    assert not handoff["errors"] and len(handoff["units"]) == 1
    assert handoff["predecessor"] == predecessor_identity
    assert yaml.safe_load(_handoff(successor, "catalog").read_text())["predecessor"] == (
        predecessor_identity
    )
    # Selection: every product file moves; the sidecars are rewritten under the successor.
    select_dir = predecessor["root"] / "select/units/LRG"
    inodes = {p.name: p.stat().st_ino for p in select_dir.iterdir()}
    summary = driver.adopt_predecessor(successor, "select")
    assert summary["units_adopted"] == ["LRG"] and summary["files_moved"] == 9
    assert sorted(p.name for p in select_dir.iterdir()) == ["attempt.yaml", "result.yaml"]
    preparation = read_preparation(successor, "LRG")
    assert all(entry["identity"]["run_name"] == "successor" for entry in preparation["catalogues"])
    parent = {"tracer": "LRG", "condition": "sigma_0.003", "realization_id": 0}
    data, randoms, _ = read_selected_parent(successor, parent, "NGC")
    expected = small_catalogue()
    for key in expected:
        np.testing.assert_array_equal(data[key], expected[key])
        np.testing.assert_array_equal(randoms[key], expected[key])
    selected = successor["root"] / "select/units/LRG"
    sidecar = json.loads((selected / "sigma_0.003--mock-0--NGC--data.json").read_text())
    assert sidecar["identity"]["run_name"] == "successor"
    assert sidecar["identity"]["code_commit"] == "b" * 40
    assert sidecar["registered_from"] == predecessor_identity
    for name in ("sigma_0.003--mock-0--NGC--data.npz", "sigma_0.003--mock-0--NGC--data.json"):
        assert (selected / name).stat().st_ino == inodes[name]  # moved, not copied
    record = driver._unit_result(successor, "select", pair.lrg["select"][0])
    assert record["stage_options"] == {} and record["registered_from"]["files"] == {"moved": 9}
    handoff = _resume(successor, "select", selection="LRG")
    assert not handoff["errors"] and len(handoff["units"]) == 1
    # Reconstruction: shard bases are redirected into the successor's adopted selection.
    unit = pair.lrg["reconstruct"][0]
    summary = driver.adopt_predecessor(successor, "reconstruct")
    assert summary["units_adopted"] == [unit["unit_id"]]
    assert len(summary["units_computed"]) == len(pair.units["reconstruct"]) - 1
    shard_path = successor["root"] / "reconstruct/units" / unit["unit_id"] / "NGC--LRG3--data.json"
    shard = json.loads(shard_path.read_text())
    assert (
        shard["identity"]["run_name"] == "successor"
        and shard["base"]["identity"]["run_name"] == "successor"
    )
    assert shard["base"]["path"] == str(
        successor["root"] / "select/units/LRG/sigma_0.003--mock-0--NGC--data.npz"
    )
    assert shard["representation"] == "displaced_reference_v2"
    moved, _, _ = read_reconstructed_parent(successor, unit, "NGC", bin_name="LRG3")
    np.testing.assert_allclose(moved["X_SHIFTED"], expected["X"] + 1, atol=1e-4)
    np.testing.assert_allclose(moved["Z_CART_SHIFTED"], expected["Z_CART"] + 3, atol=1e-4)
    payload = json.loads(
        (
            successor["root"] / "reconstruct/units" / unit["unit_id"] / "reconstruction.json"
        ).read_text()
    )
    # The solve's own identity survives: diagnostics are the predecessor's, not re-identified.
    assert payload["diagnostics"]["NGC"] == {"solver": {"commit": "a" * 40}}
    assert all(item["identity"]["run_name"] == "successor" for item in payload["catalogues"])
    record = driver._unit_result(successor, "reconstruct", unit)
    assert record["stage_options"] == {"save_catalogs": True}
    assert record["registered_from"]["files"] == {"moved": 9}  # 4 shards, 4 sidecars, 1 document
    handoff = _resume(successor, "reconstruct", selection=unit["unit_id"])
    assert not handoff["errors"] and len(handoff["units"]) == 1
    # Clustering: every published LRG unit moves; the BGS units are measured by the array.
    computed = []

    def measure(run, unit, destination, *, nthreads, statistics):
        computed.append(unit["unit_id"])
        (destination / "measured.json").write_text(json.dumps({"unit": unit["unit_id"]}))
        return {"products": [{"path": "measured.json", "format": "json"}], "statistics": statistics}

    monkeypatch.setattr(clustering, "cluster_stage", measure)
    summary = driver.adopt_predecessor(successor, "cluster")
    assert sorted(summary["units_adopted"]) == sorted(u["unit_id"] for u in pair.lrg["cluster"])
    assert sorted(summary["units_computed"]) == sorted(
        u["unit_id"] for u in pair.units["cluster"] if u["tracer"] != "LRG"
    )
    cli = CliRunner()
    resumed = cli.invoke(
        driver.main, ["cluster", "--input", str(_handoff(successor, "reconstruct")), "--resume"]
    )
    assert resumed.exit_code == 0, (resumed.output, resumed.exception)
    handoff = yaml.safe_load(_handoff(successor, "cluster").read_text())
    assert not handoff["errors"] and len(handoff["units"]) == len(pair.units["cluster"])
    assert sorted(computed) == sorted(summary["units_computed"]) and len(computed) == 25
    route = {
        "tracer": "LRG",
        "condition": "sigma_0.003",
        "measurement_bin": "LRG3",
        "reconstruction_state": "postreconstruction",
    }
    records = read_measurements(successor, route, statistic="pk")
    assert len(records) == 3 * 25
    assert records[-1]["power_arrays"]["power"][0, 0] == 25.0
    first = driver._unit_result(successor, "cluster", pair.lrg["cluster"][0])
    assert first["registered_from"]["stage"] == "cluster" and first["statistics"] == "pk"
    assert first["stage_options"] == {"statistics": "both"}
    # The random references of the solved mock resolve inside the successor's adopted shards.
    solved = next(record for record in records if "randoms" in record)
    assert solved["randoms"]["path"] == str(
        successor["root"] / "reconstruct/units" / unit["unit_id"] / "NGC--LRG3--random.npz"
    )
    assert solved["randoms"]["identity"]["run_name"] == "successor"
    assert solved["randoms"]["identity"]["code_commit"] == "b" * 40
    arrays = clustering.read_measurement_randoms(solved, successor["config"])
    assert arrays["source_ids"].shape == (solved["random_count"],)
    np.testing.assert_allclose(
        arrays["shifted_positions"][:, 2], arrays["positions"][:, 2] + 3, atol=1e-4
    )
    assert sum("randoms" in record for record in records) == 2  # the solved mock's two caps
    text = (successor["root"] / "cluster/units" / unit["unit_id"] / "measurements.json").read_text()
    assert "predecessor" not in text and str(predecessor["root"]) not in text
    bgs = next(u for u in pair.units["cluster"] if u["tracer"] != "LRG")
    assert "registered_from" not in driver._unit_result(successor, "cluster", bgs)
    # Resuming again keeps the adopted units, measures nothing and leaves the predecessor alone.
    before = _tree(predecessor["root"])
    again = cli.invoke(
        driver.main, ["cluster", "--input", str(_handoff(successor, "reconstruct")), "--resume"]
    )
    assert again.exit_code == 0 and len(computed) == 25
    assert before == _tree(predecessor["root"])
    for stage in driver.REGISTERED_STAGES:
        assert (successor["root"] / stage / "adopted.json").is_file()
        assert _handoff(predecessor, stage).is_file()
    assert not any((predecessor["root"] / "reconstruct/attempts").iterdir())


def test_adoption_fails_closed_before_the_first_move(successor_pair, monkeypatch):
    pair = successor_pair
    predecessor, successor = pair.predecessor, pair.successor
    before = _tree(predecessor["root"])

    def refused(run, stage, exception, match):
        with pytest.raises(exception, match=match):
            driver.adopt_predecessor(run, stage)
        assert before == _tree(predecessor["root"]), "the predecessor changed"
        assert not (run["root"] / stage / "adopted.json").exists()
        units = run["root"] / stage / "units"
        assert not units.exists() or not any(units.iterdir())

    # Later stages are computed, never adopted; a run without a predecessor adopts nothing.
    refused(successor, "covariance", ValueError, "never adopted")
    plain = driver.create_run(pair.root, "plain", workload="pilot", catalog_root=pair.root)
    refused(plain, "catalog", ValueError, "created with --predecessor")
    # The adopting checkout must be the successor's recorded producer.
    pair.code.update(code_commit="c" * 40)
    refused(successor, "catalog", ValueError, "execution checkout differs")
    pair.code.update(code_commit="b" * 40)
    # The predecessor's handoff must be its own and must list only published units.
    handoff = _handoff(predecessor, "catalog")
    saved = handoff.read_text()
    document = yaml.safe_load(saved)
    document["run_name"] = "someone-else"
    handoff.write_text(yaml.safe_dump(document))
    before = _tree(predecessor["root"])
    refused(successor, "catalog", ValueError, "not the predecessor's published catalog handoff")
    handoff.write_text(saved)
    handoff.with_name("gone").write_text(saved)
    handoff.rename(handoff.with_name("catalog.moved"))
    before = _tree(predecessor["root"])
    refused(successor, "catalog", ValueError, "has not published catalog")
    handoff.with_name("catalog.moved").rename(handoff)
    handoff.with_name("gone").unlink()
    before = _tree(predecessor["root"])
    directory = predecessor["root"] / "catalog/units/LRG"
    result = directory / "result.yaml"
    original = result.read_text()
    record = yaml.safe_load(original)
    record["analysis_config_sha256"] = "0" * 64
    result.write_text(yaml.safe_dump(record))
    before = _tree(predecessor["root"])
    refused(successor, "catalog", ValueError, "identity or snapshot differs")
    result.write_text(original)
    directory.rename(directory.with_name("LRG.absent"))
    before = _tree(predecessor["root"])
    refused(successor, "catalog", ValueError, "has not published")
    directory.with_name("LRG.absent").rename(directory)
    before = _tree(predecessor["root"])
    # A missing or empty product refuses the whole stage; so does an existing destination.
    product = directory / "catalog.json"
    product.rename(product.with_name("catalog.hidden"))
    before = _tree(predecessor["root"])
    refused(successor, "catalog", ValueError, "missing or invalid predecessor product")
    product.with_name("catalog.hidden").rename(product)
    content = product.read_bytes()
    product.write_bytes(b"")
    before = _tree(predecessor["root"])
    refused(successor, "catalog", ValueError, "missing or invalid predecessor product")
    product.write_bytes(content)
    before = _tree(predecessor["root"])
    (successor["root"] / "catalog/units/LRG").mkdir(parents=True)
    with pytest.raises(FileExistsError, match="destination already exists"):
        driver.adopt_predecessor(successor, "catalog")
    assert before == _tree(predecessor["root"])
    (successor["root"] / "catalog/units/LRG").rmdir()
    # Products on another filesystem cannot be moved; nothing is copied instead.
    same_filesystem = driver._same_filesystem
    monkeypatch.setattr(driver, "_same_filesystem", lambda source, destination: False)
    refused(successor, "catalog", ValueError, "within one filesystem")
    monkeypatch.setattr(driver, "_same_filesystem", same_filesystem)
    # A catalogue product may embed only the predecessor's own identity, matching its sidecar.
    select_dir = predecessor["root"] / "select/units/LRG"
    sidecar = select_dir / "sigma_0.003--mock-0--NGC--data.json"
    saved = sidecar.read_text()
    metadata = json.loads(saved)
    metadata["identity"]["run_name"] = "someone-else"
    sidecar.write_text(json.dumps(metadata))
    before = _tree(predecessor["root"])
    refused(successor, "select", ValueError, "sidecar identity differs")
    saved_result = (select_dir / "result.yaml").read_text()
    record = yaml.safe_load(saved_result)
    for product in record["products"]:
        if product["path"] == "sigma_0.003--mock-0--NGC--data.npz":
            product["identity"]["run_name"] = "someone-else"
    (select_dir / "result.yaml").write_text(yaml.safe_dump(record))
    before = _tree(predecessor["root"])
    refused(successor, "select", ValueError, "not the predecessor's")
    (select_dir / "result.yaml").write_text(saved_result)
    sidecar.write_text(saved)
    before = _tree(predecessor["root"])
    # Shards cannot be adopted before the selection they reference has been adopted here.
    refused(successor, "reconstruct", ValueError, "has not adopted the base")
    # A dry run reports the coverage and creates nothing; it takes no lock either.
    successor_before = _tree(successor["root"])
    coverage = driver.adopt_predecessor(successor, "select", dry_run=True)
    assert coverage["units_adopted"] == ["LRG"] and coverage["units_computed"] == [
        "BGS_BRIGHT-21.5"
    ]
    assert coverage["files_moved"] == 9 and "adopted_at" not in coverage
    assert before == _tree(predecessor["root"]) and successor_before == _tree(successor["root"])
    assert not (successor["root"] / "select/units").exists()
    # A rename that fails mid-unit moves every file back; no partial unit is left anywhere.
    real_rename, calls = os.rename, []

    def failing_rename(source, destination):
        calls.append(Path(source).name)
        if len(calls) == 3:
            raise OSError("I/O error")
        real_rename(source, destination)

    monkeypatch.setattr(os, "rename", failing_rename)
    with pytest.raises(OSError, match="I/O error"):
        driver.adopt_predecessor(successor, "select")
    monkeypatch.setattr(os, "rename", real_rename)
    assert len(calls) == 5  # two moves, the failure, then the two moves back
    assert before == _tree(predecessor["root"])
    assert not (successor["root"] / "select/units/LRG").exists()
    assert not any((successor["root"] / "select/attempts").iterdir())
    assert not (successor["root"] / "select/adopted.json").exists()
    assert json.loads(sidecar.read_text()) == json.loads(saved)
    # The attempted unit of the predecessor is never resumed: it is absent from the handoff.
    attempted = pair.lrg["reconstruct"][1]
    stale = predecessor["root"] / "reconstruct/attempts" / f"{attempted['unit_id']}.deadbeef"
    stale.mkdir()
    (stale / "attempt.yaml").write_text("unit: stale\n")
    assert not driver.adopt_predecessor(successor, "select")["units_computed"] == []
    summary = driver.adopt_predecessor(successor, "reconstruct")
    assert attempted["unit_id"] in summary["units_computed"]
    assert (stale / "attempt.yaml").read_text() == "unit: stale\n"


def _publish_alias_predecessor(source):
    """A stage stand-in whose selection aliases ``source``'s cohort (select --reuse-selection).

    The non-primary transfer runs publish exactly this: selected_reference_v1 sidecars whose
    bases lie in the sibling run, a preparation naming that run as ``selection_source``, and
    shards and measurements chained on top of the aliases.
    """

    def science(stage, run, unit, destination, cpus):
        if stage != "select":
            return _publish_predecessor(stage, run, unit, destination, cpus)
        from tests.test_boundary_support import per_bin_preparation_fields

        tracer, config = unit["tracer"], run["config"]
        parent = {"tracer": tracer, "condition": "sigma_0.003", "realization_id": 0}
        products, entries = [], []
        for cap in ("NGC", "SGC"):
            for role in ("data", "random"):
                identity = catalog.catalogue_identity(
                    run,
                    tracer=tracer,
                    condition="sigma_0.003",
                    mock_id=0,
                    cap=cap,
                    role=role,
                    stage="select",
                )
                name = f"sigma_0.003--mock-0--{cap}--{role}.json"
                base = selection.selected_catalogue_entry(source, parent, cap, role)
                products += catalog.save_catalogue_reference(
                    destination, name, identity=identity, base=base
                )
                entries.append({"path": name, "identity": identity})
        preparation = {
            "tracer": tracer,
            "realization_ids": unit["realization_ids"],
            "catalogues": entries,
            "selection_source": driver._source_identity(source),
            **per_bin_preparation_fields(config, tracer),
        }
        (destination / "preparation.json").write_text(json.dumps(preparation))
        return {
            "products": [{"path": "preparation.json", "format": "json"}, *products],
            "tracer": tracer,
            "realization_ids": unit["realization_ids"],
        }

    return science


@pytest.fixture
def alias_chain(tmp_path, monkeypatch):
    """The production selection topology: external <- primary (aliases) <- sigma (aliases).

    ``external`` is an earlier campaign's immutable run holding the LRG cohort bytes;
    ``primary`` aliases it and ``sigma`` aliases the primary's aliases, then publishes LRG's
    first reconstruction unit and every LRG clustering unit on top of them. The primary and
    sigma each get an empty successor at commit b.
    """
    from tests.test_run import synthetic_scalar_grid

    synthetic_scalar_grid(monkeypatch)
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    code = {"code_commit": "a" * 40, "repository": str(tmp_path / "producer-a")}
    monkeypatch.setattr(driver, "_code_identity", lambda **kwargs: dict(code))
    real_science = driver._science
    runs = {}
    for name, source in (("external", None), ("primary", "external"), ("sigma", "primary")):
        science = (
            _publish_predecessor if source is None else _publish_alias_predecessor(runs[source])
        )
        monkeypatch.setattr(driver, "_science", science)
        runs[name] = driver.create_run(tmp_path, name, workload="pilot", catalog_root=tmp_path)
        for stage in ("catalog", "select"):
            assert not driver.execute_stage(runs[name], stage, selection="LRG")["errors"]
    config = runs["sigma"]["config"]
    units = {stage: driver.stage_units(stage, config, "pilot") for stage in driver.STAGES[:4]}
    lrg = {
        stage: [unit for unit in items if unit["tracer"] == "LRG"] for stage, items in units.items()
    }
    recon = lrg["reconstruct"][0]
    assert not driver.execute_stage(runs["sigma"], "reconstruct", selection=recon["unit_id"])[
        "errors"
    ]
    for unit in lrg["cluster"]:
        assert not driver.execute_stage(runs["sigma"], "cluster", selection=unit["unit_id"])[
            "errors"
        ]
    monkeypatch.setattr(driver, "_science", real_science)
    code.update(code_commit="b" * 40, repository=str(tmp_path / "producer-b"))
    successors = {
        name: driver.create_run(
            tmp_path,
            f"{name}-successor",
            workload="pilot",
            catalog_root=tmp_path,
            predecessor=runs[name]["root"],
        )
        for name in ("primary", "sigma")
    }
    return SimpleNamespace(
        external=runs["external"],
        primary=runs["primary"],
        sigma=runs["sigma"],
        primary_successor=successors["primary"],
        sigma_successor=successors["sigma"],
        lrg=lrg,
    )


def _names_any(text, *roots):
    """Whether a document names one of these run roots (exactly or as a path prefix)."""
    return any(f'"{root}"' in text or f"{root}/" in text for root in roots)


def test_adoption_redirects_aliases_into_the_adopted_sibling_and_refuses_dangling_bases(
    alias_chain,
):
    from pzbao.reconstruction import read_reconstructed_parent
    from pzbao.selection import read_preparation, read_selected_parent

    chain = alias_chain
    external, primary, sigma = chain.external, chain.primary, chain.sigma
    primary_successor, sigma_successor = chain.primary_successor, chain.sigma_successor
    select_unit = chain.lrg["select"][0]
    parent = {"tracer": "LRG", "condition": "sigma_0.003", "realization_id": 0}
    alias = "sigma_0.003--mock-0--NGC--data.json"
    bytes_path = external["root"] / "select/units/LRG/sigma_0.003--mock-0--NGC--data.npz"
    expected = small_catalogue()
    # The published topology: sigma -> the primary's alias -> the external run's bytes.
    recorded = json.loads((sigma["root"] / "select/units/LRG" / alias).read_text())
    assert recorded["base"]["path"] == str(primary["root"] / "select/units/LRG" / alias)
    assert json.loads(Path(recorded["base"]["path"]).read_text())["base"]["path"] == str(bytes_path)
    external_before, sigma_before = _tree(external["root"]), _tree(sigma["root"])
    # Siblings must be other successors with other predecessors.
    with pytest.raises(ValueError, match="not created with --predecessor"):
        driver.adopt_predecessor(sigma_successor, "select", siblings=[primary])
    with pytest.raises(ValueError, match="other successors"):
        driver.adopt_predecessor(sigma_successor, "select", siblings=[sigma_successor])
    # Before the primary is adopted, the sibling's aliases cannot be redirected into it.
    with pytest.raises(ValueError, match="primary-successor has not adopted the base"):
        driver.adopt_predecessor(sigma_successor, "select", siblings=[primary_successor])
    assert sigma_before == _tree(sigma["root"])
    assert not (sigma_successor["root"] / "select/units").exists()
    # The primary's aliases point outside every retired run: they stay as recorded.
    summary = driver.adopt_predecessor(primary_successor, "select", siblings=[sigma_successor])
    assert summary["units_adopted"] == ["LRG"] and summary["files_moved"] == 5
    assert summary["siblings"] == [driver._source_identity(sigma_successor)]
    moved = json.loads((primary_successor["root"] / "select/units/LRG" / alias).read_text())
    assert moved["identity"]["run_name"] == "primary-successor"
    assert moved["base"]["path"] == str(bytes_path)
    assert moved["base"]["identity"]["run_name"] == "external"
    assert moved["registered_from"] == driver._source_identity(primary)
    assert read_preparation(primary_successor, "LRG")["selection_source"] == (
        driver._source_identity(external)
    )
    data, randoms, _ = read_selected_parent(primary_successor, parent, "NGC")
    for key in expected:
        np.testing.assert_array_equal(data[key], expected[key])
        np.testing.assert_array_equal(randoms[key], expected[key])
    driver._unit_result(primary_successor, "select", select_unit)  # the array's validation
    assert external_before == _tree(external["root"])
    # Without the sibling declared, the moved bases are simply gone: refused, nothing moved.
    with pytest.raises(ValueError, match="missing or re-identified"):
        driver.adopt_predecessor(sigma_successor, "select")
    assert sigma_before == _tree(sigma["root"])
    # A reused selection naming the retired run under another identity is refused.
    preparation_path = sigma["root"] / "select/units/LRG/preparation.json"
    saved = preparation_path.read_text()
    document = json.loads(saved)
    document["selection_source"]["code_commit"] = "c" * 40
    preparation_path.write_text(json.dumps(document))
    tampered = _tree(sigma["root"])
    with pytest.raises(ValueError, match="retired run under another identity"):
        driver.adopt_predecessor(sigma_successor, "select", siblings=[primary_successor])
    assert tampered == _tree(sigma["root"])
    preparation_path.write_text(saved)
    # With the primary adopted and declared, the aliases are redirected into it.
    summary = driver.adopt_predecessor(sigma_successor, "select", siblings=[primary_successor])
    assert summary["units_adopted"] == ["LRG"] and summary["files_moved"] == 5
    redirected = json.loads((sigma_successor["root"] / "select/units/LRG" / alias).read_text())
    assert redirected["identity"]["run_name"] == "sigma-successor"
    assert redirected["base"]["path"] == str(primary_successor["root"] / "select/units/LRG" / alias)
    assert redirected["base"]["identity"]["run_name"] == "primary-successor"
    assert redirected["base"]["identity"]["code_commit"] == "b" * 40
    assert redirected["registered_from"] == driver._source_identity(sigma)
    preparation = read_preparation(sigma_successor, "LRG")
    assert preparation["selection_source"] == driver._source_identity(primary_successor)
    text = (sigma_successor["root"] / "select/units/LRG/preparation.json").read_text()
    assert not _names_any(text, primary["root"], sigma["root"])
    data, randoms, _ = read_selected_parent(sigma_successor, parent, "NGC")
    for key in expected:
        np.testing.assert_array_equal(data[key], expected[key])
        np.testing.assert_array_equal(randoms[key], expected[key])
    record = driver._unit_result(sigma_successor, "select", select_unit)
    assert record["registered_from"]["files"] == {"moved": 5}
    # Shards on top of the aliases redirect into this run's adopted aliases; the chain reads.
    unit = chain.lrg["reconstruct"][0]
    summary = driver.adopt_predecessor(sigma_successor, "reconstruct", siblings=[primary_successor])
    assert summary["units_adopted"] == [unit["unit_id"]]
    shard_dir = sigma_successor["root"] / "reconstruct/units" / unit["unit_id"]
    shard = json.loads((shard_dir / "NGC--LRG3--data.json").read_text())
    assert shard["base"]["path"] == str(sigma_successor["root"] / "select/units/LRG" / alias)
    assert shard["base"]["identity"]["run_name"] == "sigma-successor"
    moved_rows, _, _ = read_reconstructed_parent(sigma_successor, unit, "NGC", bin_name="LRG3")
    np.testing.assert_allclose(moved_rows["X_SHIFTED"], expected["X"] + 1, atol=1e-4)
    driver._unit_result(sigma_successor, "reconstruct", unit)
    # A clustering document that still named the predecessor by path or identity is refused
    # before anything moves; so is an unsupported random reference.
    measured = sigma["root"] / "cluster/units" / unit["unit_id"] / "measurements.json"
    saved = measured.read_text()
    for diagnostics, match in (
        ({"log": str(sigma["root"] / "cluster/attempts/stale.log")}, "still names a retired run"),
        (
            {"solver": {key: sigma["metadata"][key] for key in driver._IDENTITY_KEYS}},
            "still carries a retired identity",
        ),
    ):
        document = json.loads(saved)
        document["diagnostics"] = diagnostics
        measured.write_text(json.dumps(document))
        tampered = _tree(sigma["root"])
        with pytest.raises(ValueError, match=match):
            driver.adopt_predecessor(sigma_successor, "cluster", siblings=[primary_successor])
        assert tampered == _tree(sigma["root"])
        assert not (sigma_successor["root"] / "cluster/units").exists()
    document = json.loads(saved)
    document["measurements"][0]["randoms"]["representation"] = "copied_randoms_v0"
    measured.write_text(json.dumps(document))
    tampered = _tree(sigma["root"])
    with pytest.raises(ValueError, match="unsupported reference"):
        driver.adopt_predecessor(sigma_successor, "cluster", siblings=[primary_successor])
    assert tampered == _tree(sigma["root"])
    measured.write_text(saved)
    # Clustering: the random references chain through the adopted shards and aliases.
    summary = driver.adopt_predecessor(sigma_successor, "cluster", siblings=[primary_successor])
    assert sorted(summary["units_adopted"]) == sorted(u["unit_id"] for u in chain.lrg["cluster"])
    adopted = sigma_successor["root"] / "cluster/units" / unit["unit_id"] / "measurements.json"
    measurements = json.loads(adopted.read_text())["measurements"]
    solved = next(record for record in measurements if "randoms" in record)
    assert solved["randoms"]["path"] == str(shard_dir / "NGC--LRG3--random.npz")
    assert solved["randoms"]["identity"]["run_name"] == "sigma-successor"
    arrays = clustering.read_measurement_randoms(solved, sigma_successor["config"])
    assert arrays["source_ids"].shape == (solved["random_count"],)
    np.testing.assert_allclose(arrays["shifted_positions"][:, 0], expected["X"] + 1, atol=1e-4)
    assert not _names_any(adopted.read_text(), primary["root"], sigma["root"])
    for stage in ("select", "reconstruct", "cluster"):
        assert _handoff(sigma, stage).is_file()
        assert (sigma_successor["root"] / stage / "adopted.json").is_file()
    assert external_before == _tree(external["root"])


def test_interrupted_adoption_resumes_and_a_concurrent_adoption_is_refused(
    successor_pair, monkeypatch
):
    pair = successor_pair
    predecessor, successor = pair.predecessor, pair.successor
    for stage in ("catalog", "select", "reconstruct"):
        driver.adopt_predecessor(successor, stage)
    cluster_units = [unit["unit_id"] for unit in pair.lrg["cluster"]]
    real_adopt_unit, calls = driver._adopt_unit, []

    def interrupted(run, stage, plan, code):
        calls.append(plan["unit"]["unit_id"])
        if len(calls) == 3:
            raise OSError("I/O error")  # the export stalls, or the process is killed
        real_adopt_unit(run, stage, plan, code)

    monkeypatch.setattr(driver, "_adopt_unit", interrupted)
    with pytest.raises(OSError, match="I/O error"):
        driver.adopt_predecessor(successor, "cluster")
    monkeypatch.setattr(driver, "_adopt_unit", real_adopt_unit)
    moved, third = calls[:2], calls[2]
    assert not (successor["root"] / "cluster/adopted.json").exists()
    for unit_id in moved:
        assert (successor["root"] / "cluster/units" / unit_id / "result.yaml").is_file()
        assert not (predecessor["root"] / "cluster/units" / unit_id / "measurements.json").exists()
    assert not (successor["root"] / "cluster/units" / third).exists()
    # A dry run recognizes the units already adopted; the receipt is still absent.
    coverage = driver.adopt_predecessor(successor, "cluster", dry_run=True)
    assert coverage["units_previously_adopted"] == moved
    assert sorted(coverage["units_adopted"]) == sorted(cluster_units)
    assert coverage["files_moved"] == 2 * (len(cluster_units) - 2)
    # Anything else at a destination still refuses: a foreign directory, a tampered record.
    foreign = successor["root"] / "cluster/units" / third
    foreign.mkdir()
    with pytest.raises(FileExistsError, match="destination already exists"):
        driver.adopt_predecessor(successor, "cluster")
    foreign.rmdir()
    result = successor["root"] / "cluster/units" / moved[0] / "result.yaml"
    saved = result.read_text()
    record = yaml.safe_load(saved)
    record["registered_from"]["result_sha256"] = "0" * 64
    result.write_text(yaml.safe_dump(record))
    with pytest.raises(FileExistsError, match="not this unit's completed adoption"):
        driver.adopt_predecessor(successor, "cluster")
    result.write_text(saved)
    # The rerun adopts the remaining units and writes the receipt for the whole stage.
    summary = driver.adopt_predecessor(successor, "cluster")
    assert summary["units_previously_adopted"] == moved
    assert sorted(summary["units_adopted"]) == sorted(cluster_units)
    assert summary["files_moved"] == 2 * (len(cluster_units) - 2)
    receipt = json.loads((successor["root"] / "cluster/adopted.json").read_text())
    assert receipt["units_previously_adopted"] == moved
    assert receipt["units_adopted"] == summary["units_adopted"]
    for unit in pair.lrg["cluster"]:
        assert driver._unit_result(successor, "cluster", unit)["registered_from"]["run_name"] == (
            "predecessor"
        )
    with pytest.raises(FileExistsError, match="already adopted"):
        driver.adopt_predecessor(successor, "cluster")
    handoff = _resume(successor, "cluster", selection=moved[0])
    assert not handoff["errors"] and len(handoff["units"]) == len(cluster_units)
    # A second adoption of one stage never runs beside the first: the stage lock refuses it.
    with (successor["root"] / "reconstruct/.locks/adopt.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            with pytest.raises(
                RuntimeError, match="another writer owns an adoption of reconstruct"
            ):
                driver.adopt_predecessor(successor, "reconstruct")
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def test_adopt_predecessor_cli_reports_coverage_and_the_stage_clis_lost_reuse_stage(
    successor_pair,
):
    pair = successor_pair
    predecessor, successor = pair.predecessor, pair.successor
    cli = CliRunner()
    dry = cli.invoke(
        driver.main,
        ["adopt-predecessor", "--run", str(successor["root"]), "--stage", "catalog", "--dry-run"],
    )
    assert dry.exit_code == 0, (dry.output, dry.exception)
    assert "catalog: would adopt 1 units of predecessor (1 files" in dry.output
    assert "1 units left for the stage to compute" in dry.output
    assert "  compute BGS_BRIGHT-21.5" in dry.output and "receipt" not in dry.output
    assert not (successor["root"] / "catalog").exists()
    real = cli.invoke(
        driver.main, ["adopt-predecessor", "--run", str(successor["root"]), "--stage", "catalog"]
    )
    assert real.exit_code == 0, (real.output, real.exception)
    assert "catalog: adopted 1 units of predecessor" in real.output
    assert f"receipt {successor['root'] / 'catalog/adopted.json'}" in real.output
    again = cli.invoke(
        driver.main, ["adopt-predecessor", "--run", str(successor["root"]), "--stage", "catalog"]
    )
    assert again.exit_code != 0 and "already adopted" in str(again.exception)
    later = cli.invoke(
        driver.main, ["adopt-predecessor", "--run", str(successor["root"]), "--stage", "covariance"]
    )
    assert later.exit_code != 0 and "covariance" in later.output
    # Registration through the stage commands is gone; the stages are untouched.
    for stage in driver.STAGES[:-1]:
        text = cli.invoke(driver.main, [stage, "--help"]).output
        assert "--reuse-stage" not in text
        assert ("--predecessor" in text) == (stage == "catalog")
    # The successor's creation still records the predecessor in local and planned runs.
    base = ["catalog", "--run-name", "cli-successor", "--output-root", str(pair.root)]
    base += ["--catalog-root", str(pair.root), "--workload", "pilot"]
    arguments = [*base, "--predecessor", str(predecessor["root"])]
    plan = cli.invoke(driver.main, [*arguments, *PLAN, "--create-snapshot", "--resume"])
    assert plan.exit_code == 0, (plan.output, plan.exception)
    assert (
        f"--predecessor {predecessor['root']}" in plan.output and "--reuse-stage" not in plan.output
    )
    run = driver.load_run(pair.root / "cli-successor")
    assert run["metadata"]["predecessor"] == driver._source_identity(predecessor)
    other = cli.invoke(
        driver.main,
        [*base, "--predecessor", str(pair.successor["root"]), "--unit", "LRG", "--resume"],
    )
    assert other.exit_code != 0 and "recorded predecessor" in str(other.output) + str(
        other.exception
    )
