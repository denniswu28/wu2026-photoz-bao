"""Linear driver snapshots, unit isolation, publication and scheduler planning."""

import pzbao.diagnostics.selection as diag_selection
import copy
import json
import shlex
from pathlib import Path
import subprocess
import sys
from click.testing import CliRunner
import pytest
from pzbao import run as driver
from pzbao.config import load_analysis_config


def synthetic_scalar_grid(monkeypatch):
    """Small explicit fixture grid; never supplies or overrides production science controls."""
    config = load_analysis_config()
    config["fitting"]["covariance"]["shot_noise_rescaling"]["tuning"] = {
        "search_limits": [0.1, 4.0],
        "search_points": 40,
        "status": "synthetic test fixture",
    }
    monkeypatch.setattr(
        driver,
        "load_analysis_config",
        lambda path=None: load_analysis_config(path) if path else copy.deepcopy(config),
    )


@pytest.fixture
def linear_run(tmp_path, monkeypatch):
    synthetic_scalar_grid(monkeypatch)
    monkeypatch.setattr(
        driver, "_code_identity", lambda: {"code_commit": "a" * 40, "repository": str(tmp_path)}
    )
    return driver.create_run(tmp_path, "synthetic", workload="pilot", catalog_root=tmp_path)


def test_linear_snapshot_materializes_full_precision_registry_and_cannot_be_overwritten(linear_run):
    run = linear_run
    original = (
        driver.DEFAULT_ANALYSIS_CONFIG_PATH.parent
        / run["config"]["fitting"]["covariance"]["shot_noise_rescaling"]["frozen_registry"]
    )
    assert (run["root"] / "frozen_alpha_sn.json").read_bytes() == original.read_bytes()
    with pytest.raises(FileExistsError):
        driver.create_run(
            run["root"].parent, "synthetic", workload="pilot", catalog_root=run["root"].parent
        )
    loaded = driver.load_run(run["root"])
    assert loaded["metadata"]["analysis_config_sha256"] == run["metadata"]["analysis_config_sha256"]
    loaded["config"]["fitting"]["sampling"]["seed"] += 1
    assert (
        driver.load_run(run["root"])["config"]["fitting"]["sampling"]["seed"]
        == run["config"]["fitting"]["sampling"]["seed"]
    )
    (run["root"] / "frozen_alpha_sn.json").write_text('{"routes": []}')
    with pytest.raises(ValueError, match="changed"):
        driver.load_run(run["root"])


def test_explicit_boundary_config_freezes_and_resumes_without_science_reuse(
    linear_run, monkeypatch, tmp_path
):
    import yaml
    from tests.test_boundary_support import publish_reference_preparation

    source = linear_run
    design = source["config"]["selection"]["boundary_support"]["experiment_design"]
    prepared = publish_reference_preparation(source, design["targets"][0], 0.09)
    derived = diag_selection.fixed_depth(
        prepared, target=design["targets"][0], config=source["config"], design=design
    )
    config = diag_selection.variant_config(
        source,
        design["targets"][0],
        mode="truncate",
        depth=derived["depth_redshift"],
        previous_depth=derived["reference_depth_redshift"],
        design=design,
    )
    path = tmp_path / "variant config.yaml"
    path.write_text(yaml.safe_dump(config))
    run = driver.create_run(
        tmp_path, "boundary-variant", workload="boundary", catalog_root=tmp_path, config_path=path
    )
    assert run["config"] == config and "source_run" not in run
    assert run["registry"] == source["registry"]
    assert len(driver.stage_units("cluster", config, "boundary")) == 25
    assert len(driver.stage_units("covariance", config, "boundary")) == 4
    assert len(driver.stage_units("fit", config, "boundary")) == 52
    with pytest.raises(ValueError, match="cannot change"):
        driver.create_run(
            tmp_path,
            "bad",
            workload="boundary",
            catalog_root=tmp_path,
            config_path=path,
            source_run=run,
        )
    calls = []
    monkeypatch.setattr(
        driver, "execute_stage", lambda *args, **kwargs: calls.append(args) or {"errors": []}
    )
    args = [
        "catalog",
        "--output-root",
        str(tmp_path),
        "--run-name",
        run["root"].name,
        "--catalog-root",
        str(tmp_path),
        "--workload",
        "boundary",
        "--resume",
        "--config",
        str(path),
    ]
    result = CliRunner().invoke(driver.main, args)
    assert result.exit_code == 0, (result.output, result.exception)
    config["selection"]["boundary_support"]["experiment"]["depth_redshift"] = 0.008
    path.write_text(yaml.safe_dump(config))
    for extra in (
        [],
        ["--scheduler", "slurm", "--account", "test", "--memory", "4G", "--walltime", "00:05:00"],
    ):
        result = CliRunner().invoke(driver.main, args + extra)
        assert result.exit_code != 0 and "differs from" in result.output
    assert len(calls) == 1


@pytest.mark.parametrize("operation", ["create", "load"])
@pytest.mark.parametrize(
    "changed",
    ["depth", "reference_depth", "both_depths", "condition", "target", "reference_condition"],
)
def test_boundary_run_rejects_edited_design_and_derived_depth(
    linear_run, tmp_path, operation, changed
):
    from pzbao.config import analysis_config_sha256
    from tests.test_boundary_support import publish_reference_preparation

    source = linear_run
    design = source["config"]["selection"]["boundary_support"]["experiment_design"]
    target = design["targets"][0]
    prepared = publish_reference_preparation(source, target, 0.09)
    derived = diag_selection.fixed_depth(
        prepared, target=target, config=source["config"], design=design
    )
    config = diag_selection.variant_config(
        source,
        target,
        mode="truncate",
        depth=derived["depth_redshift"],
        previous_depth=derived["reference_depth_redshift"],
        design=design,
    )
    path = tmp_path / "variant.yaml"
    driver._write_yaml(path, config)
    if operation == "load":
        run = driver.create_run(
            tmp_path, "variant", workload="boundary", catalog_root=tmp_path, config_path=path
        )
    support = config["selection"]["boundary_support"]
    experiment = support["experiment"]
    if changed in {"depth", "both_depths"}:
        experiment["depth_redshift"] *= 0.5
    if changed in {"reference_depth", "both_depths"}:
        experiment["reference_depth_redshift"] *= 0.5
    if changed == "condition":
        support["experiment_design"]["rehearsal_condition"] = "sigma_0.001"
        experiment["condition"] = "sigma_0.001"
        config["analysis"]["workloads"]["boundary"]["sigma_z_tiers"] = ["sigma_0.001"]
    if changed == "target":
        support["experiment_design"]["targets"][0].update(side="above", measurement_bin="BGS2")
        experiment["side"] = "above"
        config["analysis"]["workloads"]["boundary"]["measurement_bins"] = ["BGS2"]
    if changed == "reference_condition":
        support["experiment_design"]["reference_condition"] = "sigma_0.01"
    driver._write_yaml(path, config)
    if operation == "load":
        driver._write_yaml(run["root"] / "analysis.yaml", config)
        metadata = dict(run["metadata"])
        metadata["analysis_config_sha256"] = analysis_config_sha256(
            {"analysis": config, "frozen_alpha_sn": run["registry"]}
        )
        driver._write_yaml(run["root"] / "run.yaml", metadata)
    with pytest.raises(ValueError, match="configured design|frozen reference preparation"):
        if operation == "create":
            driver.create_run(
                tmp_path, "variant", workload="boundary", catalog_root=tmp_path, config_path=path
            )
        else:
            driver.load_run(run["root"])
    if operation == "create":
        assert not (tmp_path / "variant").exists()


def test_explicit_config_is_carried_into_inputs_array(tmp_path):
    config = load_analysis_config()
    path = tmp_path / "variant config.yaml"
    script = driver.slurm_plan(
        "catalog",
        driver.stage_units("catalog", config, "pilot"),
        input_path=None,
        cpus=1,
        memory="4G",
        walltime="00:05:00",
        account="test",
        run={
            "run_name": "variant",
            "output_root": tmp_path,
            "catalog_root": tmp_path,
            "workload": "pilot",
            "config_path": path,
        },
    )
    command = shlex.split(script.rstrip().splitlines()[-1])
    assert command[command.index("--config") + 1] == str(path.resolve())
    with pytest.raises(ValueError, match="one explicit --unit"):
        driver.slurm_plan(
            "cluster",
            driver.stage_units("cluster", config, "pilot"),
            input_path=tmp_path / "catalog.yaml",
            cpus=1,
            memory="4G",
            walltime="00:05:00",
            account="test",
            boundary_experiment=True,
        )


def test_linear_existing_run_never_reloads_repository_settings(linear_run, monkeypatch):
    expected = copy.deepcopy(linear_run["config"])
    monkeypatch.setattr(
        driver, "load_analysis_config", lambda: pytest.fail("later stage reloaded editable YAML")
    )
    assert driver.load_run(linear_run["root"])["config"] == expected


def test_inference_snapshot_pins_source_and_rejects_changed_or_nested_sources(
    linear_run, monkeypatch
):
    source = linear_run
    original = (source["root"] / "run.yaml").read_bytes()
    monkeypatch.setattr(
        driver, "_code_identity", lambda: {"code_commit": "b" * 40, "repository": "/corrected"}
    )
    monkeypatch.setattr(
        driver, "load_analysis_config", lambda: pytest.fail("inference reloaded editable science")
    )
    run = driver.create_run(
        source["root"].parent,
        "inference",
        workload="pilot",
        catalog_root=source["metadata"]["catalog_root"],
        source_run=source,
    )
    assert run["metadata"]["code_commit"] == "b" * 40
    assert run["metadata"]["source_run"] == driver._source_identity(source)
    assert run["config"] == source["config"]
    assert (run["root"] / "frozen_alpha_sn.json").read_bytes() == (
        source["root"] / "frozen_alpha_sn.json"
    ).read_bytes()
    assert not (run["root"] / "catalog").exists()
    for stage in ("catalog", "cluster", "covariance"):
        with pytest.raises(ValueError, match="only publish fits and samples"):
            driver.execute_stage(run, stage)
        assert not (run["root"] / stage).exists()
    with pytest.raises(ValueError, match="original measurement run"):
        driver.create_run(
            source["root"].parent,
            "nested",
            workload="pilot",
            catalog_root=source["metadata"]["catalog_root"],
            source_run=run,
        )
    with pytest.raises(ValueError, match="outside the source"):
        driver.create_run(
            source["root"],
            "nested",
            workload="pilot",
            catalog_root=source["metadata"]["catalog_root"],
            source_run=source,
        )
    assert (source["root"] / "run.yaml").read_bytes() == original
    changed = {**source["metadata"], "code_commit": "e" * 40}
    driver._write_yaml(source["root"] / "run.yaml", changed)
    with pytest.raises(ValueError, match="source identity or snapshot changed"):
        driver.load_run(run["root"])


def test_inference_cli_dry_plan_publication_and_resume_keep_both_identities(
    linear_run, monkeypatch
):
    source = linear_run
    source_handoff = source["root"] / "covariance" / "covariances.yaml"
    source_handoff.parent.mkdir()
    driver._write_yaml(
        source_handoff,
        {
            "stage": "covariance",
            "run": "..",
            "run_name": source["metadata"]["run_name"],
            "analysis_config_sha256": source["metadata"]["analysis_config_sha256"],
            "units": [],
        },
    )
    monkeypatch.setattr(
        driver, "_code_identity", lambda: {"code_commit": "b" * 40, "repository": "/corrected"}
    )
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    calls = []

    def science(stage, run, unit, destination, cpus):
        calls.append(unit)
        assert run["source_run"]["metadata"]["code_commit"] == "a" * 40
        (destination / "dispatch.json").write_text(json.dumps(unit))
        return {"products": [{"path": "dispatch.json", "format": "json"}]}

    monkeypatch.setattr(driver, "_science", science)
    target = source["root"].parent / "corrected fits"
    target = target / "inference"
    unit = next(
        (
            u
            for u in driver.stage_units("fit", source["config"], "pilot")
            if u["realization_id"] is None
        )
    )
    args = [
        "fit",
        "--input",
        str(source_handoff),
        "--inference-run",
        str(target),
        "--unit",
        unit["unit_id"],
        "--resume",
    ]
    cli = CliRunner()
    plan = cli.invoke(
        driver.main,
        args
        + ["--scheduler", "slurm", "--account", "test", "--memory", "4G", "--walltime", "00:10:00"],
    )
    assert plan.exit_code == 0, plan.output
    assert not target.exists() and (not calls)
    command = shlex.split(plan.output.rstrip().splitlines()[-1])
    assert command[command.index("--inference-run") + 1] == str(target)
    for _ in range(2):
        result = cli.invoke(driver.main, args)
        assert result.exit_code == 0, (result.output, result.exception)
    assert calls == [unit]
    run = driver.load_run(target)
    record = driver._unit_result(run, "fit", unit)
    assert record["code_commit"] == "b" * 40
    assert record["source_run"] == driver._source_identity(source)
    assert not (source["root"] / "fit").exists()
    path = target / "fit" / "units" / unit["unit_id"] / "result.yaml"
    record["source_run"]["code_commit"] = "f" * 40
    driver._write_yaml(path, record)
    with pytest.raises(ValueError, match="identity or snapshot differs"):
        driver._unit_result(run, "fit", unit)


def test_linear_array_creation_race_publishes_one_snapshot(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    monkeypatch.setattr(
        driver, "_code_identity", lambda: {"code_commit": "d" * 40, "repository": str(tmp_path)}
    )

    def create():
        try:
            return driver.create_run(tmp_path, "same-run", workload="pilot", catalog_root=tmp_path)
        except FileExistsError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda _: create(), range(2)))
    assert sum((value is not None for value in outcomes)) == 1
    published = next((value for value in outcomes if value is not None))
    assert driver.load_run(tmp_path / "same-run")["metadata"] == published["metadata"]
    assert not list(tmp_path.glob("same-run.*.partial"))


@pytest.mark.parametrize(
    "workload,caps,routes,ensembles", [("pilot", 8, 4, 16), ("full_grid", 100, 50, 200)]
)
def test_linear_workload_counts_and_slurm_selection(workload, caps, routes, ensembles):
    config = load_analysis_config()
    integrations = driver.stage_units("covariance", config, workload)
    samplers = driver.stage_units("sample", config, workload)
    assert len(integrations) == caps
    assert len(samplers) == routes
    assert sum((unit["ensemble_count"] for unit in samplers)) == ensembles
    script = driver.slurm_plan(
        "sample",
        samplers,
        input_path=Path("/tmp/a space/fits.yaml"),
        cpus=3,
        memory="5G",
        walltime="00:20:00",
        account="test-account",
        partition="standard",
    )
    assert f"#SBATCH --array=0-{routes - 1}" in script
    assert "#SBATCH --cpus-per-task=3" in script
    assert "#SBATCH --mem=5G" in script and "#SBATCH --time=00:20:00" in script
    assert "#SBATCH --account=test-account" in script
    assert "#SBATCH --mail-type=END,FAIL" in script
    assert "--mail-user" not in script and "ARRAY_TASKS" not in script
    assert "#SBATCH --partition=standard" in script
    assert '"${unit_ids[$SLURM_ARRAY_TASK_ID]}"' in script
    assert "sbatch" not in script
    selected = driver.slurm_plan(
        "covariance",
        [integrations[3]],
        input_path=Path("clustering.yaml"),
        cpus=2,
        memory="4G",
        walltime="00:10:00",
        account="test-account",
    )
    assert "#SBATCH --array=0-0" in selected
    assert integrations[3]["unit_id"] in selected
    assert integrations[0]["unit_id"] not in selected
    assert str(Path("clustering.yaml").resolve()) in selected


def test_batched_slurm_plan_gives_each_task_a_contiguous_slice():
    """unified-fit-v3: a task runs several units in one process, so ~4,900 fits take ~380
    queue slots; the slices cover every unit once, in order, and a one-unit plan is unchanged."""
    samplers = driver.stage_units("sample", load_analysis_config(), "full_grid")
    fields = dict(
        input_path=Path("fits.yaml"), cpus=2, memory="10G", walltime="01:00:00", account="a"
    )
    script = driver.slurm_plan("sample", samplers, units_per_task=13, **fields)
    assert "#SBATCH --array=0-3%1" in script and "# 50 units in 4 tasks;" in script
    lines = script.splitlines()
    selection = next(line for line in lines if line.startswith("unit_ids=("))
    task_units = lines[-1].split(" --unit ", 1)[1]
    expanded = subprocess.run(
        [
            "bash",
            "-c",
            f"{selection}\nfor SLURM_ARRAY_TASK_ID in 0 1 2 3; do echo {task_units}; done",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    slices = [driver._selected_units(samplers, value) for value in expanded]
    assert [len(chosen) for chosen in slices] == [13, 13, 13, 11]
    assert [unit for chosen in slices for unit in chosen] == samplers
    assert driver.slurm_plan("sample", samplers, units_per_task=1, **fields) == (
        driver.slurm_plan("sample", samplers, **fields)
    )
    with pytest.raises(ValueError, match="units-per-task"):
        driver.slurm_plan("sample", samplers, units_per_task=0, **fields)
    with pytest.raises(ValueError, match="source groups"):
        driver.slurm_plan("reconstruct", samplers, units_per_task=2, source_group=True, **fields)


def test_comma_selection_names_units_in_order_once():
    units = [{"unit_id": name} for name in ("a--x", "b.0", "c_1")]
    assert driver._selected_units(units, "c_1,0") == [units[2], units[0]]
    with pytest.raises(ValueError, match="more than once"):
        driver._selected_units(units, "b.0,1")
    with pytest.raises(ValueError, match="absent"):
        driver._selected_units(units, "a--x,d")


def test_batched_task_publishes_every_unit_and_one_handoff(linear_run, monkeypatch):
    def science(stage, run, unit, attempt, cpus):
        (attempt / "input.json").write_text(json.dumps({"tracer": unit["tracer"]}))
        return {"products": [{"path": "input.json", "format": "json"}]}

    publications = []
    publish = driver._publish_handoff
    monkeypatch.setattr(driver, "_science", science)
    monkeypatch.setattr(
        driver,
        "_publish_handoff",
        lambda *args: publications.append(args[1]) or publish(*args),
    )
    units = driver.stage_units("catalog", linear_run["config"], "pilot")
    selection = ",".join(unit["unit_id"] for unit in units[:2])
    result = driver.execute_stage(linear_run, "catalog", selection=selection)
    assert not result["errors"] and publications == ["catalog"]
    assert [row["unit"] for row in result["units"]] == units[:2]
    assert (
        driver._read_yaml(linear_run["root"] / "catalog/catalog.yaml")["units"] == result["units"]
    )


def test_handoff_lock_polls_while_held_and_reports_the_wait(linear_run, monkeypatch, capsys):
    """2026-10-21: finished fits waited a median 636 s on a blocking NFS flock of this lock."""
    import fcntl
    import threading

    monkeypatch.setattr(driver, "HANDOFF_POLL_SECONDS", 0.01)
    monkeypatch.setattr(driver, "HANDOFF_PROGRESS_SECONDS", 0.05)
    units = driver.stage_units("catalog", linear_run["config"], "pilot")
    root = linear_run["root"] / "catalog"
    root.mkdir()
    holder = (root / ".handoff.lock").open("a")
    fcntl.flock(holder, fcntl.LOCK_EX)
    release = threading.Timer(0.3, holder.close)
    release.start()
    try:
        handoff = driver._publish_handoff(linear_run, "catalog", units, [])
    finally:
        release.join()
    assert handoff["units"] == [] and handoff["expected_unit_count"] == len(units)
    err = capsys.readouterr().err
    assert "handoff: " in err and " held; waited " in err and "acquired" in err


@pytest.mark.parametrize("stage", ["catalog", "cluster"])
def test_slurm_paths_survive_the_worker_changing_directory(tmp_path, monkeypatch, stage):
    monkeypatch.chdir(tmp_path)
    catalog = tmp_path / "catalog with spaces"
    catalog.mkdir()
    handoff = tmp_path / "cluster" / "clustering.yaml"
    handoff.parent.mkdir()
    handoff.write_text("stage: measure\n")
    script = driver.slurm_plan(
        stage,
        [{"unit_id": "unit"}],
        input_path=Path("cluster/clustering.yaml"),
        cpus=1,
        memory="1G",
        walltime="00:01:00",
        account="test-account",
        run={
            "run_name": "run",
            "output_root": Path("outputs with spaces"),
            "catalog_root": Path("catalog with spaces"),
            "workload": "pilot",
        },
    )
    args = shlex.split(script.splitlines()[-1])
    expected = (
        {"--output-root": tmp_path / "outputs with spaces", "--catalog-root": catalog}
        if stage == "catalog"
        else {"--input": handoff}
    )
    monkeypatch.chdir(Path(driver.__file__).resolve().parents[1])
    for flag, path in expected.items():
        emitted = Path(args[args.index(flag) + 1])
        assert emitted.is_absolute() and emitted.resolve() == path


@pytest.mark.parametrize(
    "invalid",
    [{"account": None}, {"account": "a\n#SBATCH --exclusive"}, {"partition": "two tokens"}],
)
def test_slurm_requires_account_and_single_token_resource_fields(invalid):
    fields = dict(cpus=1, memory="1G", walltime="00:01:00", account="test-account")
    fields.update(invalid)
    with pytest.raises(ValueError, match="account|single tokens"):
        driver.slurm_plan(
            "cluster", [{"unit_id": "unit"}], input_path=Path("catalog.yaml"), **fields
        )


def test_array_discovery_reads_identities_without_reopening_other_units_products(
    linear_run, monkeypatch
):
    import numpy as np

    validated = []
    original = driver._validate_products

    def validate(directory, result):
        validated.append(result["unit"]["unit_id"])
        return original(directory, result)

    def science(stage, run, unit, attempt, cpus):
        np.savez(attempt / "positions.npz", positions=np.ones((3, 3)))
        return {"products": [{"path": "positions.npz", "format": "npz", "arrays": ["positions"]}]}

    monkeypatch.setattr(driver, "_science", science)
    monkeypatch.setattr(driver, "_validate_products", validate)
    units = driver.stage_units("catalog", linear_run["config"], "pilot")
    for index in range(len(units)):
        result = driver.execute_stage(linear_run, "catalog", selection=str(index))
        assert not result["errors"]
    assert validated == [unit["unit_id"] for unit in units]
    first = linear_run["root"] / "catalog" / "units" / units[0]["unit_id"]
    np.savez(first / "positions.npz", positions=np.full((3, 3), np.nan))
    result = driver.execute_stage(linear_run, "catalog", selection="1", resume=True)
    assert not result["errors"] and len(result["units"]) == len(units)
    with pytest.raises(ValueError, match="non-finite"):
        driver._unit_result(linear_run, "catalog", units[0])
    record = driver._read_yaml(first / "result.yaml")
    record["code_commit"] = "wrong"
    driver._write_yaml(first / "result.yaml", record)
    result = driver.execute_stage(linear_run, "catalog", selection="1", resume=True)
    assert any((error["unit_id"] == units[0]["unit_id"] for error in result["errors"]))
    assert len(result["units"]) == len(units) - 1


def test_linear_independent_failures_preserve_failed_attempts_and_reuse_complete_units(
    linear_run, monkeypatch
):
    calls = []
    units = driver.stage_units("catalog", linear_run["config"], "pilot")
    broken = units[0]["unit_id"]

    def science(stage, run, unit, attempt, cpus):
        calls.append(unit["unit_id"])
        (attempt / "input.json").write_text(json.dumps({"value": 3.25, "tracer": unit["tracer"]}))
        if unit["unit_id"] == broken:
            raise OSError("synthetic missing source")
        return {"products": [{"path": "input.json", "format": "json"}]}

    monkeypatch.setattr(driver, "_science", science)
    first = driver.execute_stage(linear_run, "catalog")
    assert len(first["errors"]) == 1 and len(first["units"]) == len(units) - 1
    assert calls == [unit["unit_id"] for unit in units]
    assert not (linear_run["root"] / "catalog" / "units" / broken).exists()
    assert list((linear_run["root"] / "catalog" / "attempts").glob(broken + ".*/failure.txt"))
    broken = None
    driver.execute_stage(linear_run, "catalog", resume=True)
    assert calls.count(units[0]["unit_id"]) == 2
    assert calls.count(units[1]["unit_id"]) == 1


def test_linear_unreadable_products_never_publish_complete_units(linear_run, monkeypatch):

    def science(stage, run, unit, attempt, cpus):
        (attempt / "bad.json").write_text("not json")
        return {"products": [{"path": "bad.json", "format": "json"}]}

    monkeypatch.setattr(driver, "_science", science)
    result = driver.execute_stage(linear_run, "catalog", selection="0")
    assert result["errors"] and (not result["units"])
    assert not list((linear_run["root"] / "catalog" / "units").iterdir())


def test_linear_handoffs_cross_fresh_process_and_reject_wrong_stage(linear_run):
    directory = linear_run["root"] / "catalog"
    directory.mkdir()
    path = directory / "catalog.yaml"
    driver._write_yaml(
        path,
        {
            "stage": "catalog",
            "run": "..",
            "run_name": "synthetic",
            "analysis_config_sha256": linear_run["metadata"]["analysis_config_sha256"],
        },
    )
    command = [
        sys.executable,
        "-c",
        "from pzbao.run import _run_from_handoff; import sys; print(_run_from_handoff(sys.argv[1], 'select')['metadata']['analysis_config_sha256'])",
        str(path),
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=True)
    assert completed.stdout.strip() == linear_run["metadata"]["analysis_config_sha256"]
    with pytest.raises(ValueError, match="requires a"):
        driver._run_from_handoff(path, "fit")


def test_linear_help_has_eight_commands_and_stage_configs():
    runner = CliRunner()
    result = runner.invoke(driver.main, ["--help"])
    assert result.exit_code == 0
    for stage in driver.STAGES:
        assert stage in result.output
        help_ = runner.invoke(driver.main, [stage, "--help"])
        assert help_.exit_code == 0
        assert "--config" in help_.output
        assert not any((flag in help_.output for flag in ("--mu-order", "--sigma-z", "--nmock")))
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import pzbao.run,sys; print(','.join(sorted(set(sys.modules)&{'numpy','pycorr','desilike','RascalC'})))",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert completed.stdout.strip() == ""


@pytest.mark.parametrize("review", [True, False])
def test_sample_review_option_survives_planning_publication_and_resume(
    linear_run, monkeypatch, review
):
    from pzbao.diagnostics import review as reporting

    run = linear_run
    handoff = run["root"] / "fit/fits.yaml"
    handoff.parent.mkdir()
    driver._write_yaml(
        handoff,
        {
            "stage": "fit",
            "run": "..",
            "run_name": run["metadata"]["run_name"],
            "analysis_config_sha256": run["metadata"]["analysis_config_sha256"],
            "units": [],
        },
    )
    calls, reports = [], []

    def science(stage, run, unit, destination, cpus):
        calls.append(unit)
        (destination / "sample.json").write_text(json.dumps(unit))
        return {
            "products": [{"path": "sample.json", "format": "json"}],
            "stop_reason": "stopping_rule_reached",
        }

    monkeypatch.setattr(driver, "_science", science)
    monkeypatch.setattr(reporting, "render_review", lambda root: reports.append(root))
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    args = ["sample", "--input", str(handoff), "--unit", "0", "--resume"]
    if not review:
        args.append("--no-review")
    cli = CliRunner()
    plan = cli.invoke(
        driver.main,
        args
        + [
            "--scheduler",
            "slurm",
            "--account",
            "test",
            "--memory",
            "10G",
            "--walltime",
            "08:00:00",
        ],
    )
    assert plan.exit_code == 0, (plan.output, plan.exception)
    assert ("--no-review" in plan.output) is not review
    for _ in range(2):
        result = cli.invoke(driver.main, args)
        assert result.exit_code == 0, (result.output, result.exception)
    assert len(calls) == 1  # Resume validates and preserves the existing product.
    saved = driver._read_yaml(run["root"] / "sample/samples.yaml")
    record = driver._read_yaml(run["root"] / "sample" / saved["units"][0]["path"])
    assert record["stop_reason"] == "stopping_rule_reached"
    assert record["stage_options"] == {}  # Reporting is not a scientific identity.
    assert len(reports) == (2 if review else 0)


def test_linear_cap_infrastructure_retry_budget_renews_per_invocation(
    linear_run, tmp_path, monkeypatch
):
    from pzbao import covariance

    unit = driver.stage_units("covariance", linear_run["config"], "pilot")[0]
    calls = []
    failure = TimeoutError("synthetic preemption")

    def interrupted(run, selected, destination, *, nthread, n_loops):
        calls.append((selected["cap"], n_loops, destination))
        raise failure

    monkeypatch.setattr(covariance, "covariance_cap_stage", interrupted)
    for invocation in range(3):
        attempt = tmp_path / f"invocation-{invocation}"
        attempt.mkdir()
        with pytest.raises(TimeoutError) as caught:
            driver._run_cap_attempt(linear_run, unit, attempt, cpus=1, n_loops=3072)
        assert caught.value is failure
        assert len(list(attempt.glob("cap-attempt-*/failure.txt"))) == 2
        assert not list(attempt.rglob("numerical_exhaustion.yaml"))
    assert len(calls) == 6
    assert {depth for _, depth, _ in calls} == {3072}


def test_linear_resume_dry_plan_does_not_read_editable_yaml(linear_run, monkeypatch):
    monkeypatch.setattr(driver, "load_analysis_config", lambda: pytest.fail("reloaded live YAML"))
    result = CliRunner().invoke(
        driver.main,
        [
            "catalog",
            "--run-name",
            linear_run["root"].name,
            "--output-root",
            str(linear_run["root"].parent),
            "--catalog-root",
            linear_run["metadata"]["catalog_root"],
            "--resume",
            "--scheduler",
            "slurm",
            "--account",
            "test-account",
            "--partition",
            "standard",
            "--cpus",
            "1",
            "--memory",
            "4G",
            "--walltime",
            "00:10:00",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "#SBATCH --array=0-1" in result.output
    assert "#SBATCH --account=test-account" in result.output
    assert "#SBATCH --partition=standard" in result.output


def test_linear_covariance_child_receives_thread_limits_before_library_imports(
    linear_run, monkeypatch
):
    handoff = linear_run["root"] / "cluster" / "clustering.yaml"
    handoff.parent.mkdir()
    driver._write_yaml(
        handoff,
        {
            "stage": "cluster",
            "run": "..",
            "run_name": linear_run["metadata"]["run_name"],
            "analysis_config_sha256": linear_run["metadata"]["analysis_config_sha256"],
        },
    )
    monkeypatch.delenv("BAO_COVARIANCE_PROCESS", raising=False)
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    calls = []

    def child(args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(driver.subprocess, "run", child)
    result = CliRunner().invoke(
        driver.main,
        ["covariance", "--input", str(handoff), "--unit", "3", "--cpus", "2", "--resume"],
    )
    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[1:4] == ["-m", "pzbao.run", "covariance"]
    assert args[args.index("--unit") + 1] == "3"
    assert args[args.index("--cpus") + 1] == "2"
    environment = kwargs["env"]
    assert environment["BAO_COVARIANCE_PROCESS"] == "1"
    assert environment["NTHREADS"] == "2"
    assert all(
        (
            environment[key] == "1"
            for key in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        )
    )


def test_linear_completed_deeper_cap_is_reused_after_other_cap_failure(linear_run, monkeypatch):
    """Cross the driver's real cap publication, route lock, failure and resume boundaries."""
    from pzbao import covariance

    run = linear_run
    route = driver.routes_for(run["config"], "pilot")[0]
    name = driver.route_name(route)
    caps = [
        unit
        for unit in driver.stage_units("covariance", run["config"], "pilot")
        if driver.route_name(unit) == name
    ]
    depths = run["config"]["fitting"]["covariance"]["rascalc"]["escalation_ladder_n_loops"]
    interrupted = True
    calls = []

    def integrate(run, unit, destination, *, nthread, n_loops=None):
        depth = depths[0] if n_loops is None else n_loops
        calls.append((unit["cap"], depth))
        if interrupted and unit["cap"] == "SGC" and (depth == depths[1]):
            raise TimeoutError("SGC preempted at unchanged deeper depth")
        (destination / "value.json").write_text(json.dumps({"cap": unit["cap"], "depth": depth}))
        return {"n_loops": depth, "products": [{"path": "value.json", "format": "json"}]}

    def assemble(run, selected, directories, destination, *, integrate_next):
        deeper = {cap: integrate_next(cap, depths[1]) for cap in ("NGC", "SGC")}
        values = [json.loads((deeper[cap] / "value.json").read_text()) for cap in ("NGC", "SGC")]
        assert values == [{"cap": cap, "depth": depths[1]} for cap in ("NGC", "SGC")]
        (destination / "assembled.json").write_text(json.dumps(values))
        return {"products": [{"path": "assembled.json", "format": "json"}]}

    monkeypatch.setattr(covariance, "covariance_cap_stage", integrate)
    monkeypatch.setattr(covariance, "assemble_route", assemble)
    driver.execute_stage(run, "covariance", selection=caps[0]["unit_id"])
    failed = driver.execute_stage(run, "covariance", selection=caps[1]["unit_id"])
    assert any((row.get("type") == "TimeoutError" for row in failed["errors"]))
    assert not (run["root"] / "covariance" / "routes" / name).exists()
    complete = run["root"] / "covariance" / "depths" / f"{name}--NGC--{depths[1]}"
    original = (driver._cap_directory(complete) / "value.json").read_bytes()
    assert calls.count(("NGC", depths[1])) == 1
    assert calls.count(("SGC", depths[1])) == 2
    assert not list((run["root"] / "covariance").rglob("numerical_exhaustion.yaml"))
    interrupted = False
    resumed = driver.execute_stage(run, "covariance", selection=caps[1]["unit_id"], resume=True)
    assert resumed["errors"] == []
    assert calls.count(("NGC", depths[1])) == 1
    assert calls.count(("SGC", depths[1])) == 3
    assert all((calls.count((cap, depths[0])) == 1 for cap in ("NGC", "SGC")))
    assert (driver._cap_directory(complete) / "value.json").read_bytes() == original
    assert (run["root"] / "covariance" / "routes" / name / "assembled.json").is_file()
