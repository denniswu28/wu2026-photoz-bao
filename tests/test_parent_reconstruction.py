"""Parent reconstruction precedes workload bin cuts and preserves cap identities."""

import pzbao.clustering as diag_core_clustering
import json
import numpy as np
import pytest
import pzbao.selection as pipeline_selection
import pzbao.reconstruction as pipeline_reconstruction
import pzbao.clustering as pipeline_clustering
from pzbao.catalog import build_mock_cosmology, measurement_bins
from tests.legacy_config import load_analysis_config
from pzbao.reconstruction import reconstruction_field_identity, reconstruction_window_bounds

SCOPE = "one_field_per_tracer"


def test_pilot_and_full_grid_prepare_one_family_box_across_all_conditions(tmp_path, monkeypatch):
    config = load_analysis_config()
    tracer = "LRG"
    identifiers = config["analysis"]["mock_ids"]
    tiers = {tier["label"] for tier in config["selection"]["tiers"]}
    bins = measurement_bins(tracer, config)
    raw = {
        "data": {"Z_OBS": np.array([(item["zmin"] + item["zmax"]) / 2 for item in bins])},
        "reservoir": {},
    }
    monkeypatch.setattr(pipeline_selection, "build_mock_cosmology", lambda config: object())
    monkeypatch.setattr(
        pipeline_selection,
        "read_catalog_registration",
        lambda run, tracer: {
            "sources": [
                {"role": "reservoir", "mock_id": mock_id, "cap": cap}
                for mock_id in identifiers
                for cap in ("NGC", "SGC")
            ]
        },
    )
    monkeypatch.setattr(pipeline_selection, "_read_parent", lambda *args, **kwargs: raw)
    monkeypatch.setattr(
        pipeline_selection,
        "prepare_data_for_selection",
        lambda data, **kwargs: (data, {"boundary_counts": {}}),
    )
    monkeypatch.setattr(
        pipeline_selection,
        "comoving_shell_volume",
        lambda cosmology, edges, area: np.full(len(edges) - 1, 1000000000.0),
    )
    monkeypatch.setattr(
        pipeline_selection, "select_exact_attainable", lambda data, **kwargs: (data, {})
    )
    monkeypatch.setattr(
        pipeline_selection,
        "fit_smooth_radial_model",
        lambda *args, **kwargs: {"convergence_evidence": {"converged": True}},
    )
    monkeypatch.setattr(
        pipeline_selection,
        "random_shell_accumulator",
        lambda *args, **kwargs: {"numerator": 0.7, "denominator": 1.0},
    )
    monkeypatch.setattr(pipeline_selection, "exact_growth_rate", lambda zeff, cosmology: 0.8)
    monkeypatch.setattr(
        diag_core_clustering, "derive_power_spectrum_geometry", lambda *args, **kwargs: {}
    )
    seen = []

    def prepared(raw, preparation, *, condition, mock_id, cap, **kwargs):
        seen.append((condition, mock_id, cap))
        high = (1000 if condition == "sigma_0.03" else 100) + mock_id
        data = {
            axis: np.array([0.0, high / 2**index])
            for index, axis in enumerate(("X", "Y", "Z_CART"))
        }
        randoms = {axis: np.array([-10.0 - mock_id, 0.0]) for axis in data}
        data["SOURCE_ID"] = np.arange(2, dtype=np.uint64)
        randoms["SOURCE_ID"] = np.arange(2, dtype=np.uint64)
        return (data, randoms, {"randoms": {"attained_random_data_ratio": 2.5 + mock_id}})

    monkeypatch.setattr(pipeline_selection, "_prepared_parent", prepared)
    saved = []
    for workload, cpus in (("pilot", 1), ("full_grid", 1), ("pilot", 2)):
        destination = tmp_path / f"{workload}-{cpus}"
        destination.mkdir()
        run = {
            "config": config,
            "metadata": {
                "run_name": "fixture",
                "code_commit": "a",
                "analysis_config_sha256": "b",
                "workload": workload,
                "catalog_root": str(tmp_path),
            },
        }
        pipeline_selection.select_stage(
            run, {"tracer": tracer, "realization_ids": identifiers}, destination, cpus=cpus
        )
        document = json.loads((destination / "preparation.json").read_text())
        assert set(document["conditions"]) == tiers
        assert set(seen) == {
            (tier, mock_id, cap)
            for tier in tiers
            for mock_id in identifiers
            for cap in ("NGC", "SGC")
        }
        assert len(seen) == len(tiers) * len(identifiers) * 2
        for condition in document["conditions"].values():
            for cap in condition.values():
                diagnostics = cap["selection_diagnostics_by_mock"]
                assert [row["realization_id"] for row in diagnostics] == identifiers
                assert [row["randoms"]["attained_random_data_ratio"] for row in diagnostics] == [
                    2.5 + mock_id for mock_id in identifiers
                ]
        seen.clear()
        document.pop("catalogues")
        saved.append(document)
    assert saved[0] == saved[1] == saved[2]
    assert (tmp_path / "pilot-1/preparation.json").read_bytes() == (
        tmp_path / "pilot-2/preparation.json"
    ).read_bytes()
    for geometry in saved[0]["geometry"].values():
        centre = np.asarray(geometry["boxcenter_mpc_h"])
        np.testing.assert_allclose(centre, (np.array([1024, 512, 256]) - 34) / 2)


def test_the_configured_scope_solves_one_field_per_parent_tracer():
    config = load_analysis_config()
    reconstruction = config["measurement"]["reconstruction"]
    assert reconstruction["scope"] == SCOPE
    assert "bin_window_padding_smoothing_multiple" not in reconstruction


def test_every_bin_of_a_parent_resolves_to_that_parents_one_field():
    """LRG1/LRG2/LRG3 are three cuts of one solve; BGS1/BGS2 are two cuts of another."""
    config = load_analysis_config()
    identities_by_tracer = {}
    for tracer in config["tracers"]:
        names = [str(spec["name"]) for spec in measurement_bins(tracer, config)]
        resolved = set()
        for bin_name in names:
            identity, expected_bin = reconstruction_field_identity(
                tracer, bin_name, scope=SCOPE, config=config
            )
            assert identity == f"{tracer}|broad"
            assert expected_bin is None
            resolved.add(identity)
        assert len(resolved) == 1
        identities_by_tracer[tracer] = resolved.pop()
        assert reconstruction_field_identity(tracer, None, scope=SCOPE, config=config) == (
            identities_by_tracer[tracer],
            None,
        )
    assert identities_by_tracer["LRG"] == "LRG|broad"
    assert identities_by_tracer["BGS_BRIGHT-21.5"] == "BGS_BRIGHT-21.5|broad"
    assert sum((len(measurement_bins(t, config)) for t in config["tracers"])) == 5
    assert len(set(identities_by_tracer.values())) == len(config["tracers"]) == 2


def test_the_two_tracer_parents_are_never_combined():
    """Independent tracer families keep separate parent reconstructions."""
    config = load_analysis_config()
    lrg, _ = reconstruction_field_identity("LRG", "LRG3", scope=SCOPE, config=config)
    bgs, _ = reconstruction_field_identity("BGS_BRIGHT-21.5", "BGS1", scope=SCOPE, config=config)
    assert lrg != bgs
    for tracer, foreign in (("LRG", "BGS1"), ("BGS_BRIGHT-21.5", "LRG3")):
        with pytest.raises(ValueError, match="not a configured measurement bin"):
            reconstruction_field_identity(tracer, foreign, scope=SCOPE, config=config)
    lrg_field = config["tracers"]["LRG"]["field"]
    bgs_field = config["tracers"]["BGS_BRIGHT-21.5"]["field"]
    assert float(bgs_field["zmax"]) <= float(lrg_field["zmin"])


def test_an_unregistered_reconstruction_scope_is_refused():
    config = load_analysis_config()
    for scope in ("one_field_per_measurement_bin", "one_field_per_cap", ""):
        with pytest.raises(ValueError, match="not a registered field division"):
            reconstruction_field_identity("LRG", None, scope=scope, config=config)


def test_the_parent_solve_window_is_the_whole_parent_field_clipped_at_both_ends():
    """No padding anywhere, and the record still says which ends were clipped."""
    config = load_analysis_config()
    pytest.importorskip("cosmoprimo")
    cosmology = build_mock_cosmology(config)
    for tracer in config["tracers"]:
        field = config["tracers"][tracer]["field"]
        window = reconstruction_window_bounds(
            tracer, cosmology=cosmology, smoothing_radius_mpc_h=15.0, config=config
        )
        low, high = window["window_redshift_range"]
        assert low == pytest.approx(float(field["zmin"]), abs=1e-12)
        assert high == pytest.approx(float(field["zmax"]), abs=1e-12)
        assert window["clipped_at_field_minimum"] is True
        assert window["clipped_at_field_maximum"] is True
        assert window["reconstruction_scope"] == SCOPE
        assert window["measurement_bin"] is None
        assert "pad_mpc_h" not in window
        assert "padding_smoothing_multiple" not in window
        reservoir_edges = sorted(
            (float(edge["redshift"]) for edge in config["tracers"][tracer]["reservoir"]["edges"])
        )
        assert reservoir_edges == [pytest.approx(low), pytest.approx(high)]


def test_the_solve_window_does_not_move_with_the_smoothing_radius():
    config = load_analysis_config()
    pytest.importorskip("cosmoprimo")
    cosmology = build_mock_cosmology(config)
    windows = [
        reconstruction_window_bounds(
            "LRG", cosmology=cosmology, smoothing_radius_mpc_h=radius, config=config
        )
        for radius in (15.0, 60.0)
    ]
    assert windows[0]["window_redshift_range"] == windows[1]["window_redshift_range"]
    assert [window["smoothing_radius_mpc_h"] for window in windows] == [15.0, 60.0]
    with pytest.raises(ValueError, match="finite and positive"):
        reconstruction_window_bounds(
            "LRG", cosmology=cosmology, smoothing_radius_mpc_h=0.0, config=config
        )


@pytest.mark.parametrize("tracer", ["LRG", "BGS_BRIGHT-21.5"])
def test_linear_parent_stage_solves_once_per_cap_before_workload_cuts(
    tmp_path, monkeypatch, tracer
):
    from pzbao.run import stage_units
    from tests.test_covariance import native_estimator

    pytest.importorskip("pycorr")
    from pycorr import TwoPointCorrelationFunction

    config = load_analysis_config()
    config["measurement"]["power_spectrum_secondary"]["diagnostic_measurement_bins"] = []
    bins = measurement_bins(tracer, config)
    all_bins = [item["name"] for item in bins]
    redshifts = np.concatenate([np.linspace(item["zmin"], item["zmax"], 5)[1:-1] for item in bins])
    catalogue = {
        "SOURCE_ID": np.arange(len(redshifts), dtype="u8"),
        "Z": redshifts,
        "Z_OBS": redshifts.copy(),
        "X": 100 * redshifts,
        "Y": 50 * redshifts,
        "Z_CART": 70 * redshifts,
        "WEIGHT": np.ones(len(redshifts)),
        "WEIGHT_FKP": np.ones(len(redshifts)),
    }
    condition = next((unit["condition"] for unit in stage_units("cluster", config, "pilot")))
    preparation = {
        "tracer": tracer,
        "realization_ids": config["analysis"]["mock_ids"],
        "sources": [],
        "geometry": {cap: {"family": {"cap": cap}} for cap in ("NGC", "SGC")},
        "effective_redshifts": {condition: {"parent": {"growth_rate": 0.77}}},
    }
    events = []
    monkeypatch.setattr(pipeline_selection, "build_mock_cosmology", lambda config: object())
    monkeypatch.setattr(
        pipeline_selection, "_read_parent", lambda *args, **kwargs: {"cap": kwargs["cap"]}
    )

    def prepare(run, unit, cap):
        data = {key: value.copy() for key, value in catalogue.items()}
        if cap == "SGC":
            data["SOURCE_ID"] += 100
        return (data, {key: value.copy() for key, value in data.items()}, preparation)

    def reconstruct(data, randoms, **kwargs):
        cap = kwargs["mask_family"]["cap"]
        events.append(("solve", cap, data["SOURCE_ID"].tolist(), kwargs["growth_rate"]))
        assert kwargs["config"] is config
        assert len(data["SOURCE_ID"]) == len(redshifts)
        for cat in (data, randoms):
            for source, target in zip(
                ("X", "Y", "Z_CART"), pipeline_reconstruction.SHIFTED_COLUMNS
            ):
                cat[target] = cat[source] + 1
        return (data, randoms, {"realized_nmesh": [8, 8, 8]})

    def measure(data, randoms, **kwargs):
        cap = "SGC" if data["SOURCE_ID"].min() >= 100 else "NGC"
        events.append(("xi", cap, data["SOURCE_ID"].tolist(), kwargs["reconstructed"]))
        assert len(data["SOURCE_ID"]) == 3
        if kwargs["reconstructed"]:
            assert kwargs["precomputed_random_counts"] is not None
        return native_estimator(
            cap=cap,
            post=kwargs["reconstructed"],
            radial_edges=kwargs["s_edges_mpc_h"],
            mu_bins=kwargs["mu_bins"],
        )

    monkeypatch.setattr(pipeline_selection, "read_selected_parent", prepare)
    monkeypatch.setattr(pipeline_reconstruction, "run_reconstruction", reconstruct)
    monkeypatch.setattr(pipeline_clustering, "compute_xi", measure)
    run = {
        "root": tmp_path,
        "config": config,
        "metadata": {
            "interface_version": 2,
            "run_name": "one-ready-tracer",
            "code_commit": "b" * 40,
            "workload": "pilot",
            "analysis_config_sha256": "a" * 64,
        },
    }
    from pzbao.run import _write_yaml

    input_directory = tmp_path / "select" / "units" / tracer
    input_directory.mkdir(parents=True)
    (input_directory / "preparation.json").write_text(json.dumps(preparation))
    input_unit = next(
        (unit for unit in stage_units("select", config, "pilot") if unit["tracer"] == tracer)
    )
    _write_yaml(
        input_directory / "result.yaml", {**run["metadata"], "stage": "select", "unit": input_unit}
    )
    other_tracer = next((name for name in config["tracers"] if name != tracer))
    with pytest.raises(FileNotFoundError):
        pipeline_selection.read_preparation(run, other_tracer)
    common = {"unit_id": "parent", "tracer": tracer, "condition": condition, "realization_id": 0}
    preparation["conditions"] = {
        condition: {
            cap: {
                "selection_diagnostics_by_mock": [{"realization_id": 0, "selection": {"bins": {}}}]
            }
            for cap in ("NGC", "SGC")
        }
    }
    # Reconstruction stores shifts that reference the saved selected catalogues.
    from pzbao import catalog as pipeline_catalog

    preparation["catalogues"] = []
    for cap in ("NGC", "SGC"):
        data, randoms, _ = prepare(run, common, cap)
        for role, values in (("data", data), ("random", randoms)):
            identity = pipeline_catalog.catalogue_identity(
                run,
                tracer=tracer,
                condition=condition,
                mock_id=0,
                cap=cap,
                role=role,
                stage="select",
            )
            name = f"{cap}--{role}.npz"
            pipeline_catalog.save_catalogue(input_directory, name, values, identity=identity)
            preparation["catalogues"].append({"path": name, "identity": identity})
    solve_unit = {**common, "measurement_bins": all_bins}
    solve_directory = tmp_path / "reconstruct" / "units" / common["unit_id"]
    solve_directory.mkdir(parents=True)
    pipeline_reconstruction.reconstruct_stage(run, solve_unit, solve_directory)
    _write_yaml(
        solve_directory / "result.yaml",
        {**run["metadata"], "stage": "reconstruct", "unit": solve_unit},
    )
    assert len([event for event in events if event[0] == "solve"]) == 2
    snapshots = {}
    for label, names in (("whole", all_bins), ("narrow", [all_bins[-1]])):
        directory = tmp_path / label
        directory.mkdir()
        events.clear()
        pipeline_clustering.cluster_stage(
            run, {**common, "measurement_bins": names}, directory, nthreads=1
        )
        snapshots[label] = list(events)
        entries = json.loads((directory / "measurements.json").read_text())["measurements"]
        assert len(entries) == len(names) * 2 * 3
        assert {entry["identity"]["measurement_bin"] for entry in entries} == set(names)
        for cap in ("NGC", "SGC"):
            cap_events = [event for event in events if event[1] == cap]
            assert cap_events[0][0] == "xi"
            assert not any(event[0] == "solve" for event in cap_events)
        with np.load(directory / f"{names[-1]}--NGC--randoms.npz") as arrays:
            np.testing.assert_array_equal(arrays["shifted_positions"], arrays["positions"] + 1)
            assert len(arrays["source_ids"]) == 3
    assert [event for event in snapshots["whole"] if event[0] == "solve"] == [
        event for event in snapshots["narrow"] if event[0] == "solve"
    ]
    for state in ("prereconstruction", "postreconstruction"):
        filename = f"{all_bins[-1]}--{state}--combined-counts.npy"
        whole = TwoPointCorrelationFunction.load(str(tmp_path / "whole" / filename))
        narrow = TwoPointCorrelationFunction.load(str(tmp_path / "narrow" / filename))
        np.testing.assert_array_equal(whole.corr, narrow.corr)
        assert pipeline_clustering.pair_count_identity(
            whole
        ) == pipeline_clustering.pair_count_identity(narrow)
    _write_yaml(
        input_directory / "result.yaml",
        {**run["metadata"], "run_name": "foreign-run", "stage": "select", "unit": input_unit},
    )
    with pytest.raises(ValueError, match="different snapshot or tracer"):
        pipeline_selection.read_preparation(run, tracer)


def test_linear_workloads_change_bin_cuts_without_changing_parent_ensemble():
    from pzbao.run import stage_units

    config = load_analysis_config()
    pilot = stage_units("cluster", config, "pilot")
    full = stage_units("cluster", config, "full_grid")
    assert len(pilot) == 2 * 25
    for unit in pilot:
        matching = [other for other in full if other["unit_id"] == unit["unit_id"]]
        assert len(matching) == 1
        assert set(unit["measurement_bins"]) <= set(matching[0]["measurement_bins"])
        assert unit["realization_id"] in config["analysis"]["mock_ids"]
