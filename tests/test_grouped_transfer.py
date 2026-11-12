"""Source reuse, compact catalogue boundaries, independent publication and storage limits."""

import copy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from click.testing import CliRunner

from pzbao import catalog, clustering, reconstruction, run as runner, selection, storage
from pzbao.config import load_analysis_config, derive_inference_config, validate_inference_config
from pzbao.config.transfer import transfer_matrix, transfer_run_configs

CODE = {"code_commit": "fixture", "code_state_sha256": "grouped-v1"}


def resource_budget(root):
    """Small synthetic limits, never a campaign volume or quota measurement."""
    observed = datetime.now(timezone.utc)
    return {
        "schema_version": 1,
        "campaign_root": str(root.resolve()),
        "accounting_roots": [str(root.resolve())],
        "shared_source_roots": [],
        "observed_allocated_bytes": 0,
        "quota_headroom_bytes": 128 * 1024**2,
        "observed_at": (observed - timedelta(hours=1)).isoformat(),
        "valid_until": (observed + timedelta(hours=1)).isoformat(),
        "remaining_components": dict.fromkeys(storage.TRANSFER_STORAGE_COMPONENTS, 1024),
        "writer_bounds": dict.fromkeys(runner.STAGES, 8 * 1024**2),
        "max_concurrent": 2,
        "evidence": dict.fromkeys(("quota", "volume", "writers"), "synthetic fixture only"),
    }


def identity(run, condition, cap, role, stage="select", tracer="LRG"):
    return catalog.catalogue_identity(
        run, tracer=tracer, condition=condition, mock_id=0, cap=cap, role=role, stage=stage
    )


def fixture_run(root, config, *, tracer="LRG"):
    root.mkdir()
    run = {
        "root": root,
        "config": config,
        "metadata": {
            "run_name": root.name,
            "code_commit": "fixture",
            "analysis_config_sha256": root.name,
            "workload": "transfer",
            "mode": "scratch",
            "interface_version": 2,
        },
        "stage_options": {"save_catalogs": True},
    }
    directory = root / "select" / "units" / tracer
    directory.mkdir(parents=True)
    bins = catalog.measurement_bins(tracer, config)
    labels = config["measurement"]["reconstruction"]["exploratory_transfer"]["class_labels"]
    preparation = {
        "tracer": tracer,
        "realization_ids": config["analysis"]["mock_ids"],
        "population": "exclusive_parent_classes",
        "sources": [],
        "catalogues": [],
        "geometry": {},
        "conditions": {},
        "effective_redshifts": {},
    }
    for cap in ("NGC", "SGC"):
        preparation["geometry"][cap] = {}
        for item in bins:
            radius = config["measurement"]["reconstruction"]["smoothing_values_mpc_h"][0]
            family = reconstruction.reconstruction_mask_family(
                tracer=tracer, cap=cap, measurement_bin=item["name"], smoothing_radius_mpc_h=radius
            )
            preparation["geometry"][cap][item["name"]] = {
                "nmesh": [4, 4, 4],
                "boxsize_mpc_h": [16.0] * 3,
                "boxcenter_mpc_h": [100.0, 0.0, 0.0],
                "family": family,
            }
    for number, label in enumerate(labels):
        preparation["conditions"][label] = {}
        preparation["effective_redshifts"][label] = {
            item["name"]: {"growth_rate": 0.8} for item in bins
        }
        for cap in ("NGC", "SGC"):
            preparation["conditions"][label][cap] = {
                "selection_diagnostics_by_mock": [{"realization_id": 0}],
                "densities": {"fixture": 1.0},
            }
            for role in ("data", "random"):
                count = len(bins) * 2
                redshift = np.repeat([(item["zmin"] + item["zmax"]) / 2 for item in bins], 2)
                data = {
                    "SOURCE_ID": np.arange(number * 100, number * 100 + count, dtype=np.uint64),
                    "Z": redshift,
                    "Z_OBS": redshift.copy(),
                    "X": 100.0 + np.arange(count) / 10,
                    "Y": np.ones(count),
                    "Z_CART": np.ones(count),
                    "WEIGHT": np.ones(count),
                    "WEIGHT_FKP": np.full(count, 0.7),
                    "MARK": np.arange(count, dtype=np.float32),
                    "PHOTSYS": np.full(count, "N", dtype="U1"),
                }
                name = f"{label}--{cap}--{role}.npz"
                ident = identity(run, label, cap, role, tracer=tracer)
                catalog.save_catalogue(directory, name, data, identity=ident)
                preparation["catalogues"].append({"path": name, "identity": ident})
    (directory / "preparation.json").write_text(json.dumps(preparation))
    (directory / "result.yaml").write_text(
        yaml.safe_dump(
            {
                **run["metadata"],
                "stage": "select",
                "unit": {
                    "unit_id": tracer,
                    "tracer": tracer,
                    "realization_ids": config["analysis"]["mock_ids"],
                },
            }
        )
    )
    run["storage_budget"] = resource_budget(root.parent)
    return run


def with_shared_budget(runs, root):
    """Peer runs of one campaign must carry the identical budget."""
    budget = resource_budget(root)
    for run in runs:
        run["storage_budget"] = copy.deepcopy(budget)
    return runs


@pytest.fixture
def siblings(tmp_path, monkeypatch):
    configs = transfer_run_configs(load_analysis_config())
    runs = [
        fixture_run(tmp_path / name, configs[name]) for name in ("self--R25", "sigma_0.003--R25")
    ]
    with_shared_budget(runs, tmp_path)
    monkeypatch.setattr(runner, "_code_identity", lambda **kwargs: dict(CODE))
    return runs


@pytest.fixture
def native_stub(monkeypatch):
    calls = []

    def solve(data, randoms, *, _field_consumer, **kwargs):
        calls.append(kwargs)
        geometry = kwargs["common_geometry"]
        field = SimpleNamespace(
            mpicomm=SimpleNamespace(size=1, rank=0),
            nmesh=np.array(geometry["nmesh"]),
            boxcenter=np.array(geometry["boxcenter_mpc_h"]),
            boxsize=np.array(geometry["boxsize_mpc_h"]),
            mesh_psi=[np.ones(geometry["nmesh"], dtype="f8")] * 3,
            read_shifted_positions=lambda positions, field: positions + 0.125,
        )
        return _field_consumer(field, {"fixture": True, "field_storage": "runtime_only"})

    monkeypatch.setattr(reconstruction, "run_reconstruction", solve)
    return calls


def selector(tracer="LRG", radius=25):
    return f"{tracer}--source-sigma_0.003--R{radius}--mock-0"


def chosen_targets(runs):
    return next(
        group["targets"] for group in runner.source_groups(runs) if group["unit_id"] == selector()
    )


def test_matrix_counts_source_groups_separately_from_targets():
    config = load_analysis_config()
    matrix = transfer_matrix(config)
    configs = transfer_run_configs(config)
    assert len(configs) == 5
    assert len(matrix["fields"]) == 35
    assert matrix["execution_counts"] == {
        "source_groups": 350,
        "bin_footprint_solves": 1750,
        "target_reconstruction_units": 650,
        "measurement_definitions": 80,
        "fitting_definitions": 120,
        "separate_fitting_checks": 40,
    }
    scopes = {name: cfg["analysis"]["workloads"]["transfer"] for name, cfg in configs.items()}
    # Self R15 is post-only and owns the source class at R15; R20 keeps every target.
    assert scopes["self--R15"]["reconstruction_states"] == ["postreconstruction"]
    assert scopes["self--R15"]["sigma_z_tiers"] == ["sigma_0.003", "sigma_0.01", "sigma_0.03"]
    assert scopes["sigma_0.003--R15"]["sigma_z_tiers"] == ["sigma_0.01", "sigma_0.03"]
    assert scopes["sigma_0.003--R20"]["sigma_z_tiers"] == [
        "sigma_0.003",
        "sigma_0.01",
        "sigma_0.03",
    ]
    runs = [
        {"root": Path(name), "config": cfg, "metadata": {"workload": "transfer"}}
        for name, cfg in transfer_run_configs(config).items()
    ]
    groups = runner.source_groups(runs)
    assert len(groups) == 350
    assert sum(len(group["targets"]) for group in groups) == 650
    assert len(next(group for group in groups if group["unit_id"] == selector())["targets"]) == 3


def test_cross_run_one_solve_per_partition_and_native_readback(siblings, native_stub, monkeypatch):
    monkeypatch.setattr(np, "save", lambda *args, **kwargs: pytest.fail("no grid persistence"))
    result = runner.execute_source_group(siblings, selector())
    assert result["errors"] == []
    assert len(native_stub) == 6  # three LRG bins, two footprints, three targets
    targets = chosen_targets(siblings)
    assert len(targets) == 3
    for target in targets:
        run, unit = target["run"], target["unit"]
        for cap in ("NGC", "SGC"):
            decoded = reconstruction.read_reconstructed_parent(run, unit, cap)
            original = selection.read_selected_parent(run, unit, cap)
            for saved, expected in zip(decoded[:2], original[:2]):
                for key, values in expected.items():
                    np.testing.assert_array_equal(saved[key], values)
                    assert saved[key].dtype == values.dtype
                for original_key, shifted_key in zip(
                    ("X", "Y", "Z_CART"), reconstruction.SHIFTED_COLUMNS
                ):
                    np.testing.assert_array_equal(
                        saved[shifted_key], expected[original_key] + 0.125
                    )
            directory = run["root"] / "reconstruct" / "units" / unit["unit_id"]
            for path in directory.glob("*.npz"):
                with np.load(path) as archive:
                    # Only the float32 shift; identities live in the verified base digest.
                    assert archive.files == ["SHIFT"] and archive["SHIFT"].dtype == np.float32
            reference = clustering.measurement_random_reference(
                run, unit, cap, "LRG2", reconstructed=True
            )
            record = {
                "randoms": reference,
                "identity": {
                    "tracer": "LRG",
                    "condition": unit["condition"],
                    "realization_id": 0,
                    "cap": cap,
                    "measurement_bin": "LRG2",
                    "analysis_config_sha256": run["metadata"]["analysis_config_sha256"],
                },
                "random_count": 2,
                "coordinate_state": "shifted",
            }
            arrays = clustering.read_measurement_randoms(record, run["config"])
            np.testing.assert_allclose(arrays["shifted_positions"] - arrays["positions"], 0.125)
            np.testing.assert_array_equal(arrays["weights"], [0.7, 0.7])
    before = {
        path: path.read_bytes()
        for run in siblings
        for path in (run["root"] / "reconstruct" / "units").rglob("*")
        if path.is_file()
    }
    assert runner.execute_source_group(siblings, selector(), resume=True)["errors"] == []
    assert len(native_stub) == 6
    assert all(path.read_bytes() == value for path, value in before.items())


@pytest.mark.parametrize("column", ["X", "WEIGHT", "SOURCE_ID"])
def test_inconsistent_source_rejected_before_solve(siblings, native_stub, column):
    source = siblings[1]
    unit = {"tracer": "LRG", "condition": "sigma_0.003", "realization_id": 0}
    entry = selection.selected_catalogue_entry(source, unit, "NGC", "data")
    arrays = catalog.read_catalogue(**entry)
    arrays[column][0] += 10
    catalog.save_catalogue(
        Path(entry["path"]).parent, Path(entry["path"]).name, arrays, identity=entry["identity"]
    )
    with pytest.raises(ValueError, match="inconsistent source"):
        runner.execute_source_group(siblings, selector())
    assert native_stub == []
    assert not any((run["root"] / "reconstruct").exists() for run in siblings)


def test_target_failure_publication_and_explicit_resume(siblings, native_stub, monkeypatch):
    original = reconstruction.apply_displacement_field
    fail = {"enabled": True}

    def shift(field, data, randoms, **kwargs):
        if fail["enabled"] and data["SOURCE_ID"][0] == 100:
            raise ValueError("target fixture failure")
        return original(field, data, randoms, **kwargs)

    monkeypatch.setattr(reconstruction, "apply_displacement_field", shift)
    result = runner.execute_source_group(siblings, selector())
    assert len(result["errors"]) == 1
    targets = chosen_targets(siblings)
    published = [
        t for t in targets if runner._unit_result(t["run"], "reconstruct", t["unit"]) is not None
    ]
    assert len(published) == 2
    saved = [
        (t["run"]["root"] / "reconstruct" / "units" / t["unit"]["unit_id"] / "result.yaml")
        for t in published
    ]
    before = [p.read_bytes() for p in saved]
    fail["enabled"] = False
    assert runner.execute_source_group(siblings, selector(), resume=True)["errors"] == []
    assert len(native_stub) == 12
    assert [p.read_bytes() for p in saved] == before
    assert len(list(siblings[1]["root"].glob("reconstruct/attempts/*/failure.txt"))) == 1


def test_source_failure_blocks_all_targets(siblings, monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("invalid source field")

    monkeypatch.setattr(reconstruction, "run_reconstruction", fail)
    result = runner.execute_source_group(siblings, selector())
    assert len(result["errors"]) == 3
    assert all(
        runner._unit_result(t["run"], "reconstruct", t["unit"]) is None
        for t in chosen_targets(siblings)
    )
    assert (
        sum(len(list(run["root"].glob("reconstruct/attempts/*/failure.txt"))) for run in siblings)
        == 3
    )


def test_group_and_ordinary_execution_share_writer_lock(siblings, native_stub):
    target = chosen_targets(siblings)[0]
    with runner._unit_lock(target["run"], "reconstruct", target["unit"]):
        with pytest.raises(RuntimeError, match="another writer"):
            runner.execute_source_group(siblings, selector())
        ordinary = runner.execute_stage(
            target["run"], "reconstruct", selection=target["unit"]["unit_id"]
        )
        assert "another writer" in ordinary["errors"][0]["error"]
    assert native_stub == []


def test_interrupted_publication_leaves_attempt_and_other_valid_targets(
    siblings, native_stub, monkeypatch
):
    rename = runner.os.rename
    fail = {"once": True}

    def interrupt(source, target):
        if fail["once"]:
            fail["once"] = False
            raise OSError("publication interrupted")
        return rename(source, target)

    monkeypatch.setattr(runner.os, "rename", interrupt)
    assert len(runner.execute_source_group(siblings, selector())["errors"]) == 1
    assert (
        sum(
            runner._unit_result(t["run"], "reconstruct", t["unit"]) is not None
            for t in chosen_targets(siblings)
        )
        == 2
    )
    assert runner.execute_source_group(siblings, selector(), resume=True)["errors"] == []
    assert len(native_stub) == 12


@pytest.mark.parametrize("radius", [15, 20])
@pytest.mark.parametrize("tracer,fields", [("LRG", 6), ("BGS_BRIGHT-21.5", 4)])
def test_high_quality_radii_apply_one_source_to_three_targets(
    tmp_path, monkeypatch, native_stub, radius, tracer, fields
):
    configs = transfer_run_configs(load_analysis_config())
    runs = [fixture_run(tmp_path / "source", configs[f"sigma_0.003--R{radius}"], tracer=tracer)]
    # At a self-check radius the self run owns the source class; the group spans both runs.
    if f"self--R{radius}" in configs:
        runs.insert(0, fixture_run(tmp_path / "self", configs[f"self--R{radius}"], tracer=tracer))
    with_shared_budget(runs, tmp_path)
    monkeypatch.setattr(runner, "_code_identity", lambda **kwargs: dict(CODE))
    result = runner.execute_source_group(runs, selector(tracer, radius))
    assert result["errors"] == [] and len(native_stub) == fields
    published = [
        path for run in runs for path in run["root"].glob("reconstruct/units/*/result.yaml")
    ]
    assert len(published) == 3


def test_compact_reference_rejects_missing_reordered_and_changed_base(siblings, native_stub):
    runner.execute_source_group(siblings, selector())
    target = chosen_targets(siblings)[0]
    run, unit = target["run"], target["unit"]
    path = run["root"] / "reconstruct" / "units" / unit["unit_id"] / "NGC--LRG2--random.npz"
    meta = json.loads(path.with_suffix(".json").read_text())
    original = copy.deepcopy(meta)
    tampered = copy.deepcopy(meta)
    tampered["source_ids"]["sha256"] = "0" * 64
    path.with_suffix(".json").write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="source identities"):
        catalog.read_catalogue(path, identity=meta["identity"])
    with np.load(path) as archive:
        shift = archive["SHIFT"]
    path.with_suffix(".json").write_text(json.dumps(original))
    np.savez(path, SHIFT=shift[:-1])
    with pytest.raises(ValueError, match="schema changed|shape"):
        catalog.read_catalogue(path, identity=meta["identity"])
    np.savez(path, SHIFT=shift)
    meta["base"]["identity"]["analysis_config_sha256"] = "wrong"
    path.with_suffix(".json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="different stage"):
        catalog.read_catalogue(path, identity=meta["identity"])
    meta["base"]["path"] = str(path.parent / "missing.npz")
    path.with_suffix(".json").write_text(json.dumps(meta))
    with pytest.raises(FileNotFoundError):
        catalog.read_catalogue(path, identity=meta["identity"])


def test_selection_alias_preserves_every_column_and_rejects_incompatible_marks(
    siblings, tmp_path, monkeypatch
):
    source, target = siblings
    unit = {"tracer": "LRG", "condition": "sigma_0.01", "realization_id": 0}
    base = selection.selected_catalogue_entry(source, unit, "NGC", "data")
    ident = identity(target, unit["condition"], "NGC", "data")
    catalog.save_catalogue_reference(tmp_path, "selected.json", identity=ident, base=base)
    arrays = catalog.read_catalogue(tmp_path / "selected.json", identity=ident)
    original = catalog.read_catalogue(**base)
    for key in arrays:
        np.testing.assert_array_equal(arrays[key], original[key])
        assert arrays[key].dtype == original[key].dtype
    monkeypatch.setattr(selection, "read_catalog_registration", lambda *args: {"sources": ["same"]})
    selection.validate_selection_reuse(source, target, "LRG")
    changed = copy.deepcopy(target)
    changed["config"]["selection"]["stable_randomness"]["seed"] = 98765
    with pytest.raises(ValueError, match="assignment, marks"):
        selection.validate_selection_reuse(source, changed, "LRG")


def test_group_scheduler_preserves_peers_bound_and_explicit_resume(siblings):
    budget_path = siblings[0]["root"].parent / "storage.yaml"
    budget_path.write_text(yaml.safe_dump(siblings[0]["storage_budget"]))
    plan = runner.slurm_plan(
        "reconstruct",
        runner.source_groups(siblings),
        input_path="selection.yaml",
        cpus=2,
        memory="4G",
        walltime="00:05:00",
        account="fixture",
        source_group=True,
        peer_inputs=["peer selection.yaml"],
        max_concurrent=2,
        storage_budget=budget_path,
        storage_outputs=[run["root"] for run in siblings],
    )
    assert "--source-group" in plan and "--peer-input" in plan and "%2" in plan
    assert "--no-requeue" in plan and "--resume" not in plan
    result = CliRunner().invoke(
        runner.main,
        ["reconstruct", "--input", __file__, "--unit", "0", "--source-group", selector()],
    )
    assert result.exit_code != 0 and "excludes --unit" in result.output


def test_reuse_selection_producer_preserves_columns_and_rebuilds_radius(tmp_path, monkeypatch):
    configs = transfer_run_configs(load_analysis_config())
    source_config, target_config = (configs[key] for key in ("self--R25", "sigma_0.003--R15"))
    # A one-mock fixture exercises the entire selected producer boundary, not a
    # scientific ensemble or a change to the authoritative configuration.
    source_config["analysis"]["mock_ids"] = target_config["analysis"]["mock_ids"] = [0]
    source = fixture_run(tmp_path / "original", source_config)
    target = {
        **source,
        "root": tmp_path / "reused",
        "config": target_config,
        "metadata": {
            **source["metadata"],
            "run_name": "reused",
            "analysis_config_sha256": "reused",
        },
    }
    destination = target["root"] / "select" / "units" / "LRG"
    destination.mkdir(parents=True)
    monkeypatch.setattr(selection, "read_catalog_registration", lambda *args: {"sources": ["same"]})
    unit = {"unit_id": "LRG", "tracer": "LRG", "realization_ids": [0]}
    result = selection.reuse_selection_stage(target, unit, destination, source=source)
    runner._write_yaml(
        destination / "result.yaml",
        {**target["metadata"], **result, "unit": unit, "stage": "select"},
    )
    assert not list(destination.glob("*.npz"))
    for label in target_config["measurement"]["reconstruction"]["exploratory_transfer"][
        "class_labels"
    ]:
        for cap in ("NGC", "SGC"):
            parent = {"tracer": "LRG", "condition": label, "realization_id": 0}
            before = selection.read_selected_parent(source, parent, cap)
            after = selection.read_selected_parent(target, parent, cap)
            for left, right in zip(before[:2], after[:2]):
                assert set(left) == set(right)
                for key in left:
                    np.testing.assert_array_equal(left[key], right[key])
                    assert left[key].dtype == right[key].dtype
            assert all(
                geometry["family"]["smoothing_mpc_h"] == 15
                for geometry in after[2]["geometry"][cap].values()
            )


def test_inference_adoption_snapshot_and_native_resolver_are_separate_from_source(
    tmp_path, monkeypatch
):
    from pzbao import theory

    config = transfer_run_configs(load_analysis_config())["sigma_0.003--R25"]
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(runner, "_code_identity", lambda **kwargs: dict(CODE))
    source = runner.create_run(
        tmp_path,
        "source",
        workload="transfer",
        catalog_root=tmp_path,
        config_path=path,
        scratch=True,
    )
    monkeypatch.setattr(selection, "read_preparation", lambda run, tracer: {"tracer": tracer})

    def context(config, preparation, *, tracer, measurement_bin, target_class):
        return {
            "tracer": tracer,
            "measurement_bin": measurement_bin,
            "target_class": target_class,
            "target_redshift": {"z_eff": 0.9},
            "sigma0": {"target": 0.03},
        }

    monkeypatch.setattr(theory, "transfer_damping_context", context)
    monkeypatch.setattr(
        catalog, "build_mock_cosmology", lambda config: SimpleNamespace(efunc=lambda z: 1.6)
    )
    records = [
        {
            "context": context(
                config,
                {},
                tracer=row["tracer"],
                measurement_bin=row["measurement_bin"],
                target_class=row["condition"],
            ),
            "centers_mpc_h": {"sigma_perp": 3.2, "sigma_parallel": 5.5, "sigma_streaming": 2.0},
            "evidence": {"kind": "source_target_theory", "reference": "synthetic-interface-only"},
        }
        for row in runner.routes_for(config, "transfer")
    ]
    artifact = {
        "units": "Mpc/h",
        "review_reference": "synthetic-not-accepted-science",
        "records": records,
    }
    artifact_path = tmp_path / "centres.yaml"
    artifact_path.write_text(yaml.safe_dump(artifact))
    before = {p: p.read_bytes() for p in source["root"].iterdir() if p.is_file()}
    for definition in ("transfer_reference", "transfer_pcs2", "transfer_flat"):
        destination = source["root"].parent / definition
        derived = runner._inference_destination(
            source,
            destination,
            definition=definition,
            resume=False,
            create=True,
            damping_centres=artifact_path,
        )
        assert yaml.safe_load((destination / "damping_centres.yaml").read_text()) == artifact
        assert derived["source_run"]["root"] == source["root"]
        assert (
            runner.load_run(destination)["config"]["measurement"]["reconstruction"]["transfer_run"][
                "damping_centres"
            ]
            == artifact
        )
        with pytest.raises(ValueError, match="original measurement"):
            runner._inference_destination(
                derived,
                destination.parent / "nested",
                definition=definition,
                resume=False,
                create=True,
            )
    assert all(p.read_bytes() == value for p, value in before.items())
    artifact["records"] = records[:-1]
    artifact_path.write_text(yaml.safe_dump(artifact))
    with pytest.raises(ValueError, match="match this source-target"):
        runner._inference_destination(
            source,
            source["root"].parent / "incomplete",
            definition="transfer_reference",
            resume=False,
            create=True,
            damping_centres=artifact_path,
        )
    with pytest.raises(ValueError, match="cannot mutate"):
        runner._inference_destination(
            source,
            source["root"].parent / "transfer_reference",
            definition="transfer_reference",
            resume=True,
            create=True,
            damping_centres=artifact_path,
        )


def test_reference_inference_all_radii_preserves_measurement_and_checks():
    for config in transfer_run_configs(load_analysis_config()).values():
        derived = derive_inference_config(config, "transfer_reference")
        artifact = {"units": "Mpc/h", "review_reference": "fixture-only", "records": []}
        derived["measurement"]["reconstruction"]["transfer_run"]["damping_centres"] = artifact
        validate_inference_config(config, derived, "transfer_reference")
        assert len(
            runner.stage_units(
                "fit", derived, "transfer", inference_definition="transfer_reference"
            )
        ) == len(runner.stage_units("fit", config, "transfer"))
        derived["measurement"]["reconstruction"]["iterations"] += 1
        with pytest.raises(ValueError):
            validate_inference_config(config, derived, "transfer_reference")


def test_storage_counts_hardlinks_once_and_rejects_unbounded_peak(tmp_path):
    root = tmp_path / "campaign"
    root.mkdir()
    data = root / "data"
    data.write_bytes(b"x" * 100)
    (root / "copy").hardlink_to(data)
    (root / "loop").symlink_to(root, target_is_directory=True)
    result = storage.allocated_campaign_bytes([root, root])
    assert result["logical_bytes"] == 100
    quantities = dict(
        existing_allocated_bytes=1000,
        remaining_components=dict.fromkeys(storage.TRANSFER_STORAGE_COMPONENTS, 100),
        bytes_per_writer=100,
        quota_headroom_bytes=1000,
        filesystem_headroom_bytes=2000,
        requested_concurrency=8,
    )
    plan = storage.transfer_storage_plan(**quantities)
    # Writers fill the planned components: existing + max(remaining 700, 8 writers 800).
    assert plan["max_concurrent"] == 8 and plan["conservative_peak_bytes"] == 1800
    with pytest.raises(ValueError, match="complete campaign peak"):
        storage.transfer_storage_plan(**{**quantities, "quota_headroom_bytes": 699})
    with pytest.raises(ValueError, match="complete campaign components"):
        storage.transfer_storage_plan(**{**quantities, "remaining_components": {}})


def test_tracer_filtered_group_plans_partition_the_complete_groups(siblings):
    groups = runner.source_groups(siblings)
    split = {
        tracer: runner.planned_source_groups(siblings, "all", tracer)
        for tracer in ("LRG", "BGS_BRIGHT-21.5")
    }
    assert all(split.values())
    assert all(g["unit_id"].startswith(t + "--") for t, gs in split.items() for g in gs)
    assert sorted(g["unit_id"] for gs in split.values() for g in gs) == sorted(
        g["unit_id"] for g in groups
    )
    assert runner.planned_source_groups(siblings, "all") == groups
    assert [g["unit_id"] for g in runner.planned_source_groups(siblings, selector())] == [
        selector()
    ]
