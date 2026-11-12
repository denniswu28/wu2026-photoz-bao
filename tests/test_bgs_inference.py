"""BGS observable changes reuse identified counts/components without changing the experiment."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from click.testing import CliRunner

from pzbao import covariance, run as driver
from pzbao.config import BGS_ANISOTROPIC, load_analysis_config, validate_inference_config
from tests.test_covariance import native_estimator


def _schema14_config():
    """Keep the versioned BGS reprojection fixtures independent of new-run defaults."""
    return load_analysis_config(Path(__file__).with_name("fixtures") / "analysis_schema14.yaml")


def legacy_bgs(config):
    result = copy.deepcopy(config)
    result["schema_version"] = 13
    result["measurement"]["reconstruction"]["scope"] = "one_field_per_tracer"
    result["fitting"]["fit_surfaces"]["BGS_BRIGHT-21.5"] = {
        "observed_multipoles": [0],
        "model_multipoles": [0, 2, 4],
        "ap_mode": "qiso",
        "fixed_parameters": {"qap": 1.0, "dbeta": 1.0},
    }
    result["fitting"]["linear_nuisance"]["varied_basenames_by_tracer"]["BGS_BRIGHT-21.5"] = [
        "bl0_0",
        "bl0_2",
    ]
    return result


@pytest.mark.parametrize(
    "path,value",
    [
        (("tracers", "BGS_BRIGHT-21.5", "reconstruction", "fiducial_bias"), 1.6),
        (("tracers", "BGS_BRIGHT-21.5", "fkp_p0_h3_mpc3"), 7100.0),
        (("fitting", "fit_edge_limits_mpc_h"), [52.0, 152.0]),
        (("fitting", "sampling", "seed"), 4202),
        (("fitting", "nonlinear_parameters", "qap", "prior", "limits"), [0.7, 1.3]),
        (("fitting", "covariance", "shot_noise_rescaling", "tuning", "search_points"), 2048),
    ],
)
def test_bgs_source_compatibility_rejects_other_scientific_changes(path, value):
    derived = _schema14_config()
    source = legacy_bgs(derived)
    before = copy.deepcopy(source)
    validate_inference_config(source, derived, BGS_ANISOTROPIC)
    assert source == before
    with pytest.raises(ValueError, match="source configuration"):
        validate_inference_config(source, derived)
    owner = derived
    for key in path[:-1]:
        owner = owner[key]
    owner[path[-1]] = value
    with pytest.raises(ValueError, match="changes more"):
        validate_inference_config(source, derived, BGS_ANISOTROPIC)


@pytest.fixture
def bgs_runs(tmp_path, monkeypatch):
    config = _schema14_config()
    config["measurement"]["xi"]["mu_bins"] = 8
    config["fitting"]["covariance"]["shot_noise_rescaling"]["tuning"] = {
        "search_limits": [0.1, 4.0],
        "search_points": 40,
        "status": "synthetic fixture",
    }
    monkeypatch.setattr(driver, "load_analysis_config", lambda: legacy_bgs(config))
    monkeypatch.setattr(
        driver, "_code_identity", lambda: {"code_commit": "a" * 40, "repository": str(tmp_path)}
    )
    source = driver.create_run(tmp_path, "measurements", workload="pilot", catalog_root=tmp_path)
    monkeypatch.setattr(driver, "load_analysis_config", lambda: copy.deepcopy(config))
    monkeypatch.setattr(
        driver, "_code_identity", lambda: {"code_commit": "b" * 40, "repository": str(tmp_path)}
    )
    derived = driver.create_run(
        tmp_path,
        "joint",
        workload="pilot",
        catalog_root=tmp_path,
        source_run=source,
        inference_definition=BGS_ANISOTROPIC,
    )
    return source, derived


def test_bgs_run_snapshot_preserves_both_identities_and_limits_work(bgs_runs):
    source, derived = bgs_runs
    before = {p.name: p.read_bytes() for p in source["root"].iterdir() if p.is_file()}
    reloaded = driver.load_run(derived["root"])
    assert reloaded["metadata"]["source_run"] == driver._source_identity(source)
    assert reloaded["config"]["schema_version"] == 14
    assert reloaded["source_run"]["config"]["schema_version"] == 13
    assert (
        reloaded["metadata"]["analysis_config_sha256"]
        != source["metadata"]["analysis_config_sha256"]
    )
    for stage, expected in (("covariance", 20), ("fit", 520), ("sample", 20)):
        units = driver.stage_units(
            stage, derived["config"], "full_grid", inference_definition=BGS_ANISOTROPIC
        )
        assert len(units) == expected
        assert all(unit["tracer"] == "BGS_BRIGHT-21.5" for unit in units)
        if stage == "covariance":
            assert all("cap" not in unit for unit in units)
    route = driver.stage_units(
        "sample", derived["config"], "pilot", inference_definition=BGS_ANISOTROPIC
    )[0]
    assert covariance.likelihood_covariance_run(derived, route) is derived
    assert covariance.likelihood_covariance_run(source, route) is source
    with pytest.raises(ValueError, match="does not supply LRG"):
        covariance.likelihood_covariance_run(derived, {**route, "tracer": "LRG"})
    with pytest.raises(ValueError, match="only publish fits and samples"):
        driver.execute_stage(derived, "cluster")
    assert before == {p.name: p.read_bytes() for p in source["root"].iterdir() if p.is_file()}
    changed = {**source["metadata"], "code_commit": "c" * 40}
    driver._write_yaml(source["root"] / "run.yaml", changed)
    with pytest.raises(ValueError, match="source identity or snapshot changed"):
        driver.load_run(derived["root"])


def test_bgs_covariance_dry_plan_uses_routes_without_creating_or_integrating(bgs_runs, monkeypatch):
    source, derived = bgs_runs
    handoff = source["root"] / "measure/clustering.yaml"
    handoff.parent.mkdir()
    driver._write_yaml(
        handoff,
        {
            "stage": "cluster",
            "run": "..",
            "run_name": source["metadata"]["run_name"],
            "analysis_config_sha256": source["metadata"]["analysis_config_sha256"],
        },
    )
    target = source["root"].parent / "planned-only"
    monkeypatch.setattr(driver, "_science", lambda *a, **kw: pytest.fail("dry plan ran science"))
    result = CliRunner().invoke(
        driver.main,
        [
            "covariance",
            "--input",
            str(handoff),
            "--inference-run",
            str(target),
            "--inference-definition",
            BGS_ANISOTROPIC,
            "--scheduler",
            "slurm",
            "--account",
            "test",
            "--memory",
            "4G",
            "--walltime",
            "00:10:00",
        ],
    )
    assert result.exit_code == 0, (result.output, result.exception)
    assert "#SBATCH --array=0-1" in result.output  # BGS1 pre/post in the pilot workload.
    assert "--inference-definition bgs_anisotropic_v1" in result.output
    assert "--NGC" not in result.output and "--SGC" not in result.output
    assert not target.exists()


@pytest.fixture
def saved_bgs_components(bgs_runs, monkeypatch):
    pytest.importorskip("pycorr")
    from pycorr import TwoPointCorrelationFunction
    import pzbao.clustering as pipeline_clustering

    source, derived = bgs_runs
    unit = driver.stage_units(
        "covariance", derived["config"], "pilot", inference_definition=BGS_ANISOTROPIC
    )[0]
    route = {
        key: unit[key] for key in ("tracer", "measurement_bin", "condition", "reconstruction_state")
    }
    records = []
    count_root = source["root"] / "native-counts"
    count_root.mkdir()
    for mid in range(25):
        caps = {
            cap: native_estimator(cap=cap, radial_edges=np.arange(201.0), identifier=mid)
            for cap in ("NGC", "SGC")
        }
        all_counts = {**caps, "NGC+SGC": pipeline_clustering.combine_cap_pair_counts(caps)}
        for cap, counts in all_counts.items():
            identity = {
                **{
                    key: source["metadata"][key]
                    for key in ("run_name", "code_commit", "analysis_config_sha256")
                },
                **route,
                "realization_id": mid,
                "cap": cap,
            }
            pipeline_clustering.bind_pair_count_identity(counts, identity)
            path = count_root / f"{mid}-{cap}.npy"
            counts.save(str(path))
            loaded = TwoPointCorrelationFunction.load(str(path))
            assert pipeline_clustering.pair_count_identity(loaded) == identity
            records.append({"identity": identity, "estimator": loaded})

    def measurements(run, requested):
        assert run["metadata"] == source["metadata"]
        assert {key: requested[key] for key in route} == route
        return records

    monkeypatch.setattr(pipeline_clustering, "read_measurements", measurements)
    directories, raw = {}, {}
    for cap, scale in (("NGC", 1.0), ("SGC", 3.0)):
        directory = source["root"] / "caps" / cap
        directory.mkdir(parents=True)
        terms = {
            "c2": np.eye(150) * 0.001 * scale,
            "c3": np.zeros((150, 150)),
            "c4": np.eye(150) * 0.00001 * scale,
        }
        np.savez(
            directory / "components.npz",
            **terms,
            **{
                key + "_samples": np.repeat(value[None], 16, axis=0) for key, value in terms.items()
            },
        )
        record = {
            **{
                key: source["metadata"][key]
                for key in ("run_name", "code_commit", "analysis_config_sha256")
            },
            "route": route,
            "cap": cap,
            "realization_ids": list(range(25)),
            "geometry": {"realization_id": 0},
            "multipoles": [0, 2, 4],
            "separation_edges_mpc_h": np.arange(0.0, 201.0, 4.0).tolist(),
            "n_loops": 1024,
            "raw_integration_completed": True,
            "ordering": "radial_major",
        }
        (directory / "cap.json").write_text(json.dumps(record))
        directories[cap], raw[cap] = directory, terms
    original = source["root"] / "covariance/routes" / unit["unit_id"]
    original.mkdir(parents=True)
    result = covariance.assemble_route(
        source,
        route,
        directories,
        original,
        integrate_next=lambda *a: pytest.fail("fixture escalated"),
    )
    result.update(
        route=route,
        **{
            key: source["metadata"][key]
            for key in ("run_name", "code_commit", "analysis_config_sha256")
        },
    )
    driver._write_yaml(original / "result.yaml", result)
    monkeypatch.setattr(
        driver, "_run_cap_attempt", lambda *a, **kw: pytest.fail("integration launched")
    )
    return source, derived, unit, route, directories, raw, records


def test_bgs_joint_covariance_roundtrip_retunes_from_source_counts(
    saved_bgs_components, monkeypatch
):
    source, derived, unit, route, directories, raw, records = saved_bgs_components
    originals = {
        str(path): path.read_bytes() for path in source["root"].rglob("*") if path.is_file()
    }
    result = driver.execute_stage(derived, "covariance", selection=unit["unit_id"])
    assert result["errors"] == []
    assert result["units"][0]["path"] == f"routes/{unit['unit_id']}/result.yaml"
    published = covariance.read_route_covariance(driver.load_run(derived["root"]), route)
    arrays, metadata = published["arrays"], published["metadata"]
    assert arrays["vectors"].shape == (25, 52)
    assert arrays["covariance"].shape == (52, 52)
    np.testing.assert_array_equal(arrays["multipoles"], [0, 2])
    np.testing.assert_array_equal(arrays["mean_covariance"], arrays["covariance"] / 25)
    for index, record in enumerate(r for r in records if r["identity"]["cap"] == "NGC+SGC"):
        _, poles = record["estimator"][::4](ells=[0, 2], return_sep=True, return_std=False)
        np.testing.assert_allclose(
            arrays["vectors"][index], np.concatenate([p[12:38] for p in poles])
        )
    terms = {
        key: sum(
            arrays["W_" + cap] @ raw[cap][key] @ arrays["W_" + cap].T for cap in ("NGC", "SGC")
        )
        for key in ("c2", "c3", "c4")
    }
    grid = np.linspace(0.1, 4.0, 40)
    sample = np.cov(arrays["vectors"], rowvar=False, ddof=1)
    objectives = []
    for alpha in grid:
        model = terms["c4"] + alpha * terms["c3"] + alpha**2 * terms["c2"]
        objectives.append(np.linalg.slogdet(model)[1] + np.trace(np.linalg.solve(model, sample)))
    assert metadata["shot_noise"]["shot_noise_rescaling"] == grid[np.argmin(objectives)]
    np.testing.assert_allclose(metadata["shot_noise"]["objective_curve"], objectives)
    assert metadata["source_run"] == driver._source_identity(source)
    for section in (metadata["diagnostics"]["combined"], *metadata["diagnostics"]["caps"].values()):
        structural = section["structural"]
        assert structural["acceptance_policy"] == "weaker_eigenvalue_and_positive_definiteness"
        assert structural["structural_test_passes"] is True
        assert "stronger_eigenvalue_margin" in structural
        assert "structural_test_warnings" in structural
    assert originals == {
        str(path): path.read_bytes() for path in source["root"].rglob("*") if path.is_file()
    }
    again = driver.execute_stage(derived, "covariance", selection=unit["unit_id"], resume=True)
    assert not again["errors"]

    import pzbao.likelihood as fitting_likelihood
    import pzbao.selection as pipeline_selection

    name, condition = route["measurement_bin"], route["condition"]
    preparation = {
        "effective_redshifts": {condition: {name: {"z_eff": 0.15}, "parent": {"z_eff": 0.2}}},
        "conditions": {
            condition: {
                cap: {"counts": {name: 1000}, "volumes_mpc_h3": {name: 1e7}}
                for cap in ("NGC", "SGC")
            }
        },
    }

    def read_preparation(run, tracer):
        assert run["metadata"] == source["metadata"]
        return preparation

    captured = []

    def likelihood(data, matrix, **kwargs):
        captured.append((data.copy(), matrix.copy(), kwargs))
        return None, None, None, {"fixture": True}

    monkeypatch.setattr(pipeline_selection, "read_preparation", read_preparation)
    monkeypatch.setattr(
        fitting_likelihood,
        "build_mock_cosmology",
        lambda config: SimpleNamespace(efunc=lambda z: 1.0, growth_rate=lambda z: 0.8),
    )
    monkeypatch.setattr(fitting_likelihood, "build_likelihood", likelihood)
    for mid in (None, 7):
        result = fitting_likelihood._route_likelihood(
            derived, {**route, "realization_id": mid}, method="profiling"
        )
        data, matrix, arguments = captured[-1]
        assert data.shape == (52,) and matrix.shape == (52, 52)
        np.testing.assert_array_equal(
            matrix, arrays["mean_covariance"] if mid is None else arrays["covariance"]
        )
        np.testing.assert_array_equal(
            data, arrays["vectors"].mean(axis=0) if mid is None else arrays["vectors"][mid]
        )
        np.testing.assert_array_equal(arguments["multipoles"], [0, 2])
        assert arguments["ap_mode"] == "qisoqap"
        assert result[3]["covariance_identity"]["run_name"] == derived["metadata"]["run_name"]
        assert result[3]["source_run"] == driver._source_identity(source)

    record_path = published["directory"] / "covariance.json"
    invalid = json.loads(record_path.read_text())
    invalid["source_run"]["code_commit"] = "f" * 40
    record_path.write_text(json.dumps(invalid))
    with pytest.raises(ValueError, match="component-source identity"):
        covariance.read_route_covariance(derived, route)


def test_bgs_covariance_rejects_mismatched_component_and_never_escalates(saved_bgs_components):
    source, derived, unit, route, directories, raw, records = saved_bgs_components
    path = directories["NGC"] / "cap.json"
    record = json.loads(path.read_text())
    record["code_commit"] = "f" * 40
    path.write_text(json.dumps(record))
    result = driver.execute_stage(derived, "covariance", selection=unit["unit_id"])
    assert result["errors"] and "geometry disagree" in result["errors"][0]["error"]
    assert not (derived["root"] / "covariance/routes" / unit["unit_id"]).exists()
    record["code_commit"] = source["metadata"]["code_commit"]
    path.write_text(json.dumps(record))
    # The selected monopole remains usable; only the newly requested quadrupole is corrupted.
    terms = {key: value.copy() for key, value in raw["NGC"].items()}
    terms["c4"][1::3, 1::3] -= np.eye(50) * 1e6
    np.savez(
        directories["NGC"] / "components.npz",
        **terms,
        **{key + "_samples": np.repeat(value[None], 16, axis=0) for key, value in terms.items()},
    )
    attempt = derived["root"] / "reuse-only-failure"
    attempt.mkdir()
    with pytest.raises(ValueError, match="no deeper integration was launched"):
        covariance.assemble_route(
            derived,
            route,
            directories,
            attempt,
            integrate_next=lambda *a: pytest.fail("integration launched"),
        )
