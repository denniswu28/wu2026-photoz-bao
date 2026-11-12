"""Relocation crosses real snapshots and readers without rewriting scientific evidence."""

import copy
import json
import shlex

from click.testing import CliRunner
import numpy as np
import pytest
import yaml

from pzbao import catalog, run as driver
from pzbao.artifacts import resolve_saved_path
from pzbao.config import BGS_ANISOTROPIC, load_analysis_config
from pzbao.config.stages import experiment_document
from pzbao.storage import resolve_path, same_run_identity


def install_map(root, rows, monkeypatch):
    path = root / "archive/relocation/paths.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema_version": 1, "roots": rows}))
    monkeypatch.setenv("BAO_STORAGE_ROOT", str(root))
    return path


def test_relocated_plot_arrays_and_originals_fallback(tmp_path, monkeypatch):
    old = tmp_path / "old-products"
    root = tmp_path / "storage"
    old.mkdir()
    archived = old / "archive/originals/density"
    archived.mkdir(parents=True)
    expected = np.array([[1.01, -0.3], [0.99, 0.7]])
    np.savez(archived / "values.npz", values=expected)
    target = root / "products"
    root.mkdir()
    old.rename(target)
    install_map(root, [{"source": str(old), "destination": "products"}], monkeypatch)
    from pzbao.figures.data import load_npz

    np.testing.assert_array_equal(load_npz(old / "density/values.npz")["values"], expected)
    assert not old.exists()
    assert not resolve_path(old / "missing.npz").exists()


@pytest.mark.parametrize("definition", [None, BGS_ANISOTROPIC])
def test_source_and_inference_snapshots_survive_relocation(tmp_path, monkeypatch, definition):
    monkeypatch.setattr(
        driver,
        "_code_identity",
        lambda **kwargs: {"code_commit": "a" * 40, "repository": "fixture"},
    )
    if definition == BGS_ANISOTROPIC:
        from tests.test_bgs_inference import _schema14_config, legacy_bgs

        config = legacy_bgs(_schema14_config())
        monkeypatch.setattr(driver, "load_analysis_config", lambda: copy.deepcopy(config))
    old = tmp_path / "old-runs"
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    source = driver.create_run(old, "source", workload="pilot", catalog_root=inputs)
    derived = driver.create_run(
        old,
        "derived",
        workload="pilot",
        catalog_root=inputs,
        source_run=source,
        inference_definition=definition,
    )
    snapshots = {p.relative_to(old): p.read_bytes() for p in old.rglob("*") if p.is_file()}
    root = tmp_path / "storage"
    target = root / "runs/baseline"
    target.parent.mkdir(parents=True)
    old.rename(target)
    (root / "data").mkdir()
    inputs.rename(root / "data/prepared")
    install_map(
        root,
        [
            {"source": str(old), "destination": "runs/baseline"},
            {"source": str(inputs), "destination": "data/prepared"},
        ],
        monkeypatch,
    )
    loaded = driver.load_run(old / "derived")
    assert loaded["root"] == target / "derived"
    assert loaded["source_run"]["root"] == target / "source"
    assert loaded["metadata"] == derived["metadata"]
    assert loaded["config"] == derived["config"]
    for relative, content in snapshots.items():
        assert (target / relative).read_bytes() == content
    if definition is not None:
        from pzbao.covariance import _component_source_run

        assert _component_source_run(loaded)["root"] == target / "source"
    for create in (False, True):
        resumed = driver._inference_destination(
            loaded["source_run"],
            old / "derived",
            definition=definition,
            resume=True,
            create=create,
        )
        assert resumed["root"] == target / "derived"
        assert resumed["metadata"] == derived["metadata"]
        with pytest.raises(FileExistsError, match="already exists"):
            driver._inference_destination(
                loaded["source_run"],
                old / "derived",
                definition=definition,
                resume=False,
                create=create,
            )
    planned = driver._inference_destination(
        loaded["source_run"], old / "new", definition=definition, resume=False, create=False
    )
    assert planned["root"] == target / "new"
    assert not planned["root"].exists()
    created = driver._inference_destination(
        loaded["source_run"], old / "new", definition=definition, resume=False, create=True
    )
    assert driver.load_run(created["root"])["config"] == derived["config"]
    assert created["root"] == target / "new"
    assert not old.exists()
    for relative, content in snapshots.items():
        assert (target / relative).read_bytes() == content
    changed = copy.deepcopy(loaded["metadata"])
    changed["source_run"]["code_commit"] = "b" * 40
    (target / "derived/run.yaml").write_text(yaml.safe_dump(changed))
    with pytest.raises(ValueError, match="source identity"):
        driver.load_run(old / "derived")


def test_mapped_identity_does_not_relax_other_fields(tmp_path, monkeypatch):
    root = tmp_path / "storage"
    (root / "runs/source").mkdir(parents=True)
    old = tmp_path / "old-source"
    install_map(root, [{"source": str(old), "destination": "runs/source"}], monkeypatch)
    original = {"root": str(old), "run_name": "source", "code_commit": "a" * 40, "count": 1}
    relocated = dict(original, root=str(root / "runs/source"))
    assert same_run_identity(original, relocated)
    for changed in (
        dict(relocated, count=True),
        dict(relocated, count=1.0),
        dict(relocated, extra=1),
    ):
        assert not same_run_identity(original, changed)
    assert original["root"] == str(old)


@pytest.mark.parametrize(
    "fault", ["overlap", "destination_overlap", "escape", "missing", "collision"]
)
def test_invalid_or_unfinished_migration_is_rejected(tmp_path, monkeypatch, fault):
    root, old = tmp_path / "storage", tmp_path / "old"
    (root / "data/prepared").mkdir(parents=True)
    rows = [{"source": str(old), "destination": "data/prepared"}]
    if fault == "overlap":
        rows.append({"source": str(old / "child"), "destination": "other"})
    elif fault == "destination_overlap":
        rows.append({"source": str(tmp_path / "other"), "destination": "data/prepared/child"})
    elif fault == "escape":
        rows[0]["destination"] = "../escape"
    elif fault == "missing":
        rows[0]["destination"] = "data/missing"
    elif fault == "collision":
        old.mkdir()
        (old / "value").write_text("conflicting bytes")
    install_map(root, rows, monkeypatch)
    with pytest.raises((ValueError, FileNotFoundError)):
        resolve_path(old / "value")


def test_registered_fits_preserves_columns_and_detects_changed_bytes(tmp_path, monkeypatch):
    Table = pytest.importorskip("astropy.table").Table
    from tests.test_catalog import _registered_inputs

    config = load_analysis_config()
    old = tmp_path / "old-inputs"
    records = _registered_inputs(old, config)
    record = next(row for row in records if row["cap"] == "NGC" and row["role"] == "data")
    values = {"TARGETID": [100, 200], "RA": [1.0, 2.0], "DEC": [4.0, 5.0], "Z": [0.5, 0.6]}
    Table(values).write(record["path"], overwrite=True)
    records = catalog.register_parent_inputs(old, tracer="LRG", mock_id=2, config=config)
    record = next(row for row in records if row["cap"] == "NGC" and row["role"] == "data")
    original = copy.deepcopy(record)
    before = catalog.read_registered_catalogue(
        record, tracer="LRG", mock_id=2, cap="NGC", role="data", config=config
    )
    root = tmp_path / "storage"
    target = root / "data/prepared"
    target.parent.mkdir(parents=True)
    old.rename(target)
    install_map(root, [{"source": str(old), "destination": "data/prepared"}], monkeypatch)
    after = catalog.read_registered_catalogue(
        record, tracer="LRG", mock_id=2, cap="NGC", role="data", config=config
    )
    assert record == original
    for key in before:
        np.testing.assert_array_equal(after[key], before[key])
    path = resolve_saved_path(record["path"])
    values["RA"][0] = 9.0
    Table(values).write(path, overwrite=True)
    with pytest.raises(ValueError, match="changed since registration"):
        catalog.read_registered_catalogue(
            record, tracer="LRG", mock_id=2, cap="NGC", role="data", config=config
        )


def test_topic_scratch_destination_is_not_nested_twice(tmp_path, monkeypatch):
    monkeypatch.setattr(
        driver,
        "_code_identity",
        lambda **kwargs: {"code_commit": "a" * 40, "repository": "fixture"},
    )
    destination = tmp_path / "scratch/smoothing"
    run = driver.create_run(
        destination, "example", workload="pilot", catalog_root=tmp_path, scratch=True
    )
    assert run["root"] == destination / "example"
    assert run["metadata"]["mode"] == "scratch"
    with pytest.raises(ValueError, match="production output cannot"):
        driver.create_run(destination, "production", workload="pilot", catalog_root=tmp_path)


@pytest.fixture
def relocated_cli_run(tmp_path, monkeypatch):
    monkeypatch.setattr(
        driver,
        "_code_identity",
        lambda: {"code_commit": "a" * 40, "repository": "fixture"},
    )
    inputs = tmp_path / "old-inputs"
    inputs.mkdir()
    run = driver.create_run(tmp_path, "old-run", workload="pilot", catalog_root=inputs)
    driver._write_yaml(run["root"] / "experiment.yaml", experiment_document(run["config"]))
    root = tmp_path / "storage"
    destination = root / "runs" / run["root"].name
    destination.parent.mkdir(parents=True)
    run["root"].rename(destination)
    (root / "data").mkdir()
    inputs.rename(root / "data/prepared")
    install_map(
        root,
        [
            {"source": str(run["root"]), "destination": str(destination.relative_to(root))},
            {"source": str(inputs), "destination": "data/prepared"},
        ],
        monkeypatch,
    )
    return run, destination


@pytest.mark.parametrize("stage", driver.STAGES[1:-1])
def test_cli_plans_from_relocated_handoff_and_config(relocated_cli_run, stage):
    run, destination = relocated_cli_run
    previous = driver.STAGES[driver.STAGES.index(stage) - 1]
    relative = f"{previous}/{driver.HANDOFFS[previous]}"
    handoff = destination / relative
    handoff.parent.mkdir()
    driver._write_yaml(
        handoff,
        {
            "stage": previous,
            "run": "..",
            "run_name": run["metadata"]["run_name"],
            "analysis_config_sha256": run["metadata"]["analysis_config_sha256"],
        },
    )
    result = CliRunner().invoke(
        driver.main,
        [
            stage,
            "--input",
            str(run["root"] / relative),
            "--config",
            str(run["root"] / "experiment.yaml"),
            "--scheduler",
            "slurm",
            "--account",
            "test",
            "--memory",
            "1G",
            "--walltime",
            "00:01:00",
        ],
    )
    assert result.exit_code == 0, (result.output, result.exception)
    arguments = shlex.split(result.output.rstrip().splitlines()[-1])
    assert arguments[arguments.index("--input") + 1] == str(handoff)
    assert arguments[arguments.index("--config") + 1] == str(destination / "experiment.yaml")
    assert not run["root"].exists()
    assert not (destination / stage).exists()


def test_catalog_cli_accepts_relocated_catalog_root(relocated_cli_run, tmp_path):
    run, destination = relocated_cli_run
    result = CliRunner().invoke(
        driver.main,
        [
            "catalog",
            "--run-name",
            "new-run",
            "--output-root",
            str(tmp_path / "new-runs"),
            "--catalog-root",
            run["metadata"]["catalog_root"],
            "--config",
            str(run["root"] / "experiment.yaml"),
            "--scheduler",
            "slurm",
            "--account",
            "test",
            "--memory",
            "1G",
            "--walltime",
            "00:01:00",
        ],
    )
    assert result.exit_code == 0, (result.output, result.exception)
    arguments = shlex.split(result.output.rstrip().splitlines()[-1])
    assert arguments[arguments.index("--catalog-root") + 1] == str(
        resolve_path(run["metadata"]["catalog_root"])
    )
    assert arguments[arguments.index("--config") + 1] == str(destination / "experiment.yaml")
    assert not (tmp_path / "new-runs").exists()


def test_review_cli_renders_relocated_run_and_preserves_notes(relocated_cli_run):
    run, destination = relocated_cli_run
    notes = destination / "review/review.md"
    notes.parent.mkdir()
    notes.write_text("Notes to retain.\n")
    snapshot = (destination / "run.yaml").read_bytes()
    result = CliRunner().invoke(
        driver.main,
        [
            "review",
            "--run",
            str(run["root"]),
            "--config",
            str(run["root"] / "experiment.yaml"),
        ],
    )
    assert result.exit_code == 0, (result.output, result.exception)
    report = (destination / "review/report.html").read_text()
    assert run["metadata"]["run_name"] in report
    assert "Missing or unreadable products" in report
    assert notes.read_text() == "Notes to retain.\n"
    assert (destination / "run.yaml").read_bytes() == snapshot
    assert not run["root"].exists()


@pytest.mark.parametrize("stage,option", [("select", "--input"), ("review", "--run")])
def test_cli_rejects_missing_relocated_input(relocated_cli_run, stage, option):
    run, _ = relocated_cli_run
    result = CliRunner().invoke(driver.main, [stage, option, str(run["root"] / "missing")])
    assert result.exit_code == 2
    assert "does not exist" in result.output
    assert not run["root"].exists()
