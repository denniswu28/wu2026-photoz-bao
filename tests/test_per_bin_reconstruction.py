"""Per-bin core reconstruction: strict partitions, bin identities, sequential solves, readers.

Specification revision 2.13. Exclusive
transfer keeps its own design; these contracts cover the primary, scan, RecIso and density runs.
"""

import json
from copy import deepcopy

import numpy as np
import pytest

import pzbao.clustering as pipeline_clustering
import pzbao.reconstruction as pipeline_reconstruction
import pzbao.selection as pipeline_selection
from pzbao import catalog as pipeline_catalog
from pzbao.catalog import build_mock_cosmology, catalogue_partition, measurement_bins
from pzbao.config import (
    derive_inference_config,
    load_analysis_config,
    per_bin_reconstruction,
    validate_analysis_config,
    validate_inference_config,
)
from pzbao.config.sensitivity import sensitivity_run_configs
from pzbao.config.transfer import transfer_run_configs
from pzbao.reconstruction import (
    SHIFTED_COLUMNS,
    reconstruction_field_identity,
    reconstruction_mask_family,
    reconstruction_window_bounds,
)
from tests.legacy_config import load_analysis_config as load_legacy_config, load_schema16_config

SCOPE = "separate_measurement_bins"
CAPS = ("NGC", "SGC")


def observed_column(config):
    return config["selection"]["stable_randomness"]["observed_redshift_column"]


class LinearCosmology:
    """Monotone comoving distance; geometry tests need no Boltzmann code."""

    def comoving_radial_distance(self, redshift):
        return 3000.0 * np.asarray(redshift, dtype=float)


def per_bin_preparation_fields(config, tracer, *, growth=None, condition=None):
    """Frozen windows, families and bin growth records in the shape the producers publish."""
    recon = config["measurement"]["reconstruction"]
    radius = recon["smoothing_values_mpc_h"][0]
    bins = measurement_bins(tracer, config)
    tiers = [tier["label"] for tier in config["selection"]["tiers"]]
    fields = {
        "reconstruction_scope": SCOPE,
        "windows": {
            item["name"]: {
                "tracer": tracer,
                "reconstruction_scope": SCOPE,
                "measurement_bin": item["name"],
                "smoothing_radius_mpc_h": radius,
                "window_redshift_range": [item["zmin"], item["zmax"]],
                "radial_padding_mpc_h": 0.0,
            }
            for item in bins
        },
        "geometry": {
            cap: {
                item["name"]: {
                    "nmesh": [8, 8, 8],
                    "boxsize_mpc_h": [32.0] * 3,
                    "boxcenter_mpc_h": [0.0] * 3,
                    "family": reconstruction_mask_family(
                        tracer=tracer,
                        cap=cap,
                        measurement_bin=item["name"],
                        smoothing_radius_mpc_h=radius,
                    ),
                    "conditions": tiers,
                    "calibration_mock_ids": list(config["analysis"]["mock_ids"]),
                }
                for item in bins
            }
            for cap in CAPS
        },
    }
    if condition is not None:
        fields["effective_redshifts"] = {
            condition: {
                "parent": {"z_eff": 0.7, "growth_rate": 0.99, "numerator": 0.7, "denominator": 1},
                **{
                    item["name"]: {
                        "z_eff": (item["zmin"] + item["zmax"]) / 2,
                        "growth_rate": growth[item["name"]],
                        "numerator": (item["zmin"] + item["zmax"]) / 2,
                        "denominator": 1.0,
                    }
                    for item in bins
                },
            }
        }
    return fields


# ------------------------------------------------------------------ configuration and identities


def test_per_bin_schemas_bind_the_per_bin_scope_and_keep_transfer_broad_field():
    config = load_analysis_config()
    assert config["schema_version"] == 17
    assert config["measurement"]["reconstruction"]["scope"] == SCOPE
    assert per_bin_reconstruction(config)
    mixed = deepcopy(config)
    mixed["measurement"]["reconstruction"]["scope"] = "one_field_per_tracer"
    with pytest.raises(ValueError, match="unsupported scope"):
        validate_analysis_config(mixed)
    # Schema 16 (revision 2.13) keeps loading for historical readers; it carries no matched
    # density block, and a schema-17 definition cannot omit it.
    previous = load_schema16_config()
    assert previous["schema_version"] == 16 and per_bin_reconstruction(previous)
    assert previous["measurement"]["reconstruction"]["scope"] == SCOPE
    assert "veff_matched_density" not in previous["selection"]
    hybrid = deepcopy(previous)
    hybrid["selection"]["veff_matched_density"] = config["selection"]["veff_matched_density"]
    with pytest.raises(ValueError, match="missing or unknown fields"):
        validate_analysis_config(hybrid)
    hybrid = deepcopy(config)
    hybrid["selection"].pop("veff_matched_density")
    with pytest.raises(ValueError, match="missing or unknown fields"):
        validate_analysis_config(hybrid)
    legacy = load_legacy_config()
    assert legacy["schema_version"] == 15
    assert not per_bin_reconstruction(legacy)
    mixed = deepcopy(legacy)
    mixed["measurement"]["reconstruction"]["scope"] = SCOPE
    with pytest.raises(ValueError, match="unsupported scope"):
        validate_analysis_config(mixed)
    for name, snapshot in sensitivity_run_configs(config).items():
        assert snapshot["schema_version"] == 17, name
        assert snapshot["measurement"]["reconstruction"]["scope"] == SCOPE, name
    # Revisions 2.13 and 2.15 change core analyses only: transfer snapshots keep their design.
    for name, snapshot in transfer_run_configs(config).items():
        recon = snapshot["measurement"]["reconstruction"]
        assert snapshot["schema_version"] == 15, name
        assert recon["scope"] == "one_field_per_tracer", name
        assert "veff_matched_density" not in snapshot["selection"], name
        assert per_bin_reconstruction(snapshot), name


def test_core_field_identities_name_the_tracer_bin_and_refuse_foreign_or_missing_bins():
    config = load_analysis_config()
    for tracer in config["tracers"]:
        names = [item["name"] for item in measurement_bins(tracer, config)]
        identities = [
            reconstruction_field_identity(tracer, name, scope=SCOPE, config=config)
            for name in names
        ]
        assert identities == [(f"{tracer}|bin={name}", name) for name in names]
        with pytest.raises(ValueError, match="requires schema 16/17 and a measurement bin"):
            reconstruction_field_identity(tracer, None, scope=SCOPE, config=config)
    for tracer, foreign in (("LRG", "BGS1"), ("BGS_BRIGHT-21.5", "LRG3")):
        with pytest.raises(ValueError, match="not a configured measurement bin"):
            reconstruction_field_identity(tracer, foreign, scope=SCOPE, config=config)
    with pytest.raises(ValueError, match="requires schema 16"):
        reconstruction_field_identity("LRG", "LRG1", scope=SCOPE, config=load_legacy_config())
    with pytest.raises(ValueError, match="not a registered field division"):
        reconstruction_field_identity("LRG", "LRG1", scope="one_field_per_bin", config=config)


def test_bin_windows_are_the_strict_bin_edges_without_padding():
    config = load_analysis_config()
    pytest.importorskip("cosmoprimo")
    cosmology = build_mock_cosmology(config)
    for tracer in config["tracers"]:
        field = config["tracers"][tracer]["field"]
        bins = measurement_bins(tracer, config)
        for index, item in enumerate(bins):
            windows = [
                reconstruction_window_bounds(
                    tracer,
                    cosmology=cosmology,
                    smoothing_radius_mpc_h=radius,
                    measurement_bin=item["name"],
                    config=config,
                )
                for radius in (15.0, 45.0)
            ]
            window = windows[0]
            assert window["window_redshift_range"] == [item["zmin"], item["zmax"]]
            assert window["radial_padding_mpc_h"] == 0.0
            assert window["membership"] == "unshifted_observed_redshift_half_open"
            assert window["measurement_bin"] == item["name"]
            assert window["reconstruction_scope"] == SCOPE
            assert window["clipped_at_field_minimum"] is (index == 0)
            assert window["clipped_at_field_maximum"] is (index == len(bins) - 1)
            assert window["field_redshift_range"] == [float(field["zmin"]), float(field["zmax"])]
            low, high = window["window_comoving_range_mpc_h"]
            field_low, field_high = window["field_comoving_range_mpc_h"]
            assert field_low <= low < high <= field_high
            # The smoothing radius names the mesh family; it never moves the window.
            assert windows[1]["window_redshift_range"] == window["window_redshift_range"]
            assert [w["smoothing_radius_mpc_h"] for w in windows] == [15.0, 45.0]
        with pytest.raises(ValueError, match="requires schema 16/17 and a measurement bin"):
            reconstruction_window_bounds(
                tracer, cosmology=cosmology, smoothing_radius_mpc_h=15.0, config=config
            )


def test_catalogue_partitions_are_half_open_on_the_unshifted_observed_redshift():
    config = load_analysis_config()
    item = measurement_bins("LRG", config)[1]
    data = catalogue_partition("LRG", item["name"], "data", config)
    random = catalogue_partition("LRG", item["name"], "random", config)
    assert data["redshift_column"] == observed_column(config)
    assert random["redshift_column"] == "Z"
    assert (data["zmin"], data["zmax"]) == (item["zmin"], item["zmax"])
    epsilon = 1e-9
    redshift = np.array(
        [
            item["zmin"] - epsilon,
            item["zmin"],
            (item["zmin"] + item["zmax"]) / 2,
            item["zmax"] - epsilon,
            item["zmax"],
        ]
    )
    # Observed and true redshifts disagree so each role must read its own column.
    catalogue = {observed_column(config): redshift, "Z": redshift[::-1].copy()}
    expected = [False, True, True, True, False]
    np.testing.assert_array_equal(pipeline_catalog._partition_rows(catalogue, data), expected)
    np.testing.assert_array_equal(
        pipeline_catalog._partition_rows(catalogue, random), expected[::-1]
    )
    assert pipeline_catalog._partition_rows(catalogue, None) == slice(None)
    with pytest.raises(ValueError, match="invalid catalogue partition"):
        pipeline_catalog._partition_rows(catalogue, {**data, "pad_mpc_h": 10.0})
    for bad in ({**data, "zmin": data["zmax"]}, {**data, "zmax": float("nan")}):
        with pytest.raises(ValueError, match="invalid catalogue redshift bounds"):
            pipeline_catalog._partition_rows(catalogue, bad)


def test_per_bin_solve_refuses_rows_outside_the_partition_empty_bins_and_repeated_identities():
    from pzbao.reconstruction import configured_reconstruction_settings, run_reconstruction

    config = load_analysis_config()
    item = measurement_bins("LRG", config)[0]
    family = reconstruction_mask_family(
        tracer="LRG", cap="NGC", smoothing_radius_mpc_h=15.0, measurement_bin=item["name"]
    )
    settings = {
        **configured_reconstruction_settings(config),
        "bias": 2.0,
        "growth_rate": 0.8,
        "smoothing_radius_mpc_h": 15.0,
        "convention": "recsym",
        "mask_family": family,
        "common_geometry": {
            "nmesh": [4, 4, 4],
            "boxsize_mpc_h": [40.0] * 3,
            "boxcenter_mpc_h": [100.0, 0.0, 0.0],
            "family": family,
        },
        "identity_column": "SOURCE_ID",
        "weight_expression": "WEIGHT",
        "pyrecon_kw_weights": None,
    }

    def catalogue(redshifts, ids=(1, 2)):
        z = np.asarray(redshifts, dtype=float)
        return {
            "SOURCE_ID": np.asarray(ids, dtype=np.uint64),
            observed_column(config): z,
            "Z": z.copy(),
            "X": 100.0 + z,
            "Y": z,
            "Z_CART": z,
            "WEIGHT": np.ones(len(z)),
        }

    inside = [(item["zmin"] + item["zmax"]) / 2, item["zmin"]]
    cases = [
        (catalogue([inside[0], item["zmax"]]), catalogue(inside, ids=(3, 4))),
        (catalogue(inside), catalogue([inside[0], item["zmin"] - 1e-6], ids=(3, 4))),
        (catalogue([], ids=()), catalogue(inside, ids=(3, 4))),
        (catalogue(inside, ids=(1, 1)), catalogue(inside, ids=(3, 4))),
    ]
    for data, randoms in cases:
        with pytest.raises(ValueError, match="exact nonempty observed-redshift partition"):
            run_reconstruction(data, randoms, **settings, config=config)
    broad = reconstruction_mask_family(tracer="LRG", cap="NGC", smoothing_radius_mpc_h=15.0)
    with pytest.raises(ValueError, match="requires a measurement bin"):
        run_reconstruction(
            catalogue(inside),
            catalogue(inside, ids=(3, 4)),
            **{
                **settings,
                "mask_family": broad,
                "common_geometry": {**settings["common_geometry"], "family": broad},
            },
            config=config,
        )


# ------------------------------------------------------------------ reconstruct, shards, cluster


def _save_selected_catalogues(run, tracer, condition, directory, preparation, make):
    from pzbao.run import _write_yaml, stage_units

    preparation["catalogues"] = []
    for cap in CAPS:
        data, randoms = make(cap)
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
            pipeline_catalog.save_catalogue(directory, name, values, identity=identity)
            preparation["catalogues"].append({"path": name, "identity": identity})
    (directory / "preparation.json").write_text(json.dumps(preparation))
    unit = next(
        unit for unit in stage_units("select", run["config"], "pilot") if unit["tracer"] == tracer
    )
    _write_yaml(directory / "result.yaml", {**run["metadata"], "stage": "select", "unit": unit})


@pytest.mark.parametrize("tracer", ["LRG", "BGS_BRIGHT-21.5"])
def test_core_stage_solves_each_bin_and_cap_on_exact_partitions_and_publishes_bin_shards(
    tmp_path, monkeypatch, tracer
):
    from pzbao.run import _write_yaml, stage_units
    from tests.test_covariance import native_estimator

    pytest.importorskip("pycorr")
    from pycorr import TwoPointCorrelationFunction

    config = load_analysis_config()
    config["measurement"]["power_spectrum_secondary"]["diagnostic_measurement_bins"] = []
    observed = observed_column(config)
    bins = measurement_bins(tracer, config)
    names = [item["name"] for item in bins]
    per_bin = 3
    redshifts = np.concatenate(
        [np.linspace(item["zmin"], item["zmax"], per_bin + 2)[1:-1] for item in bins]
    )
    # One galaxy sits exactly on an interior edge: half-open bins place it in the upper bin.
    redshifts[per_bin] = bins[1]["zmin"]
    # Rows are stored in descending redshift so partitions must preserve row order, not sort.
    z = np.sort(redshifts)[::-1].copy()
    ids = np.arange(len(z), dtype="u8")

    def make(cap):
        offset = 0 if cap == "NGC" else 100
        data = {
            "SOURCE_ID": ids + offset,
            # The true redshift of every galaxy lies above its observed one; membership
            # follows the observed value only.
            "Z": z + 0.013,
            observed: z.copy(),
            "X": 100 * z,
            "Y": 50 * z,
            "Z_CART": 70 * z,
            "WEIGHT": np.ones(len(z)),
            "WEIGHT_FKP": np.ones(len(z)),
        }
        randoms = {key: value.copy() for key, value in data.items() if key != observed}
        randoms["Z"] = z.copy()
        randoms["SOURCE_ID"] = ids + offset + 1000
        return data, randoms

    def expected_ids(cap, name):
        item = next(item for item in bins if item["name"] == name)
        rows = (z >= item["zmin"]) & (z < item["zmax"])
        return (ids[rows] + (0 if cap == "NGC" else 100)).tolist()

    condition = next(unit["condition"] for unit in stage_units("cluster", config, "pilot"))
    growth = {name: 0.7 + 0.01 * index for index, name in enumerate(names)}
    preparation = {
        "tracer": tracer,
        "realization_ids": config["analysis"]["mock_ids"],
        "sources": [],
        "conditions": {
            condition: {
                cap: {
                    "selection_diagnostics_by_mock": [
                        {"realization_id": 0, "selection": {"bins": {}}}
                    ]
                }
                for cap in CAPS
            }
        },
        **per_bin_preparation_fields(config, tracer, growth=growth, condition=condition),
    }
    run = {
        "root": tmp_path,
        "config": config,
        "metadata": {
            "interface_version": 2,
            "run_name": "per-bin-fixture",
            "code_commit": "b" * 40,
            "workload": "pilot",
            "analysis_config_sha256": "a" * 64,
        },
    }
    input_directory = tmp_path / "select" / "units" / tracer
    input_directory.mkdir(parents=True)
    _save_selected_catalogues(run, tracer, condition, input_directory, preparation, make)

    solves, reads = [], []
    real_read = pipeline_selection.read_selected_parent

    def reading(run, unit, cap, *, bin_name=None):
        reads.append((cap, bin_name))
        return real_read(run, unit, cap, bin_name=bin_name)

    def reconstruct(data, randoms, **kwargs):
        family = kwargs["mask_family"]
        solves.append(
            (
                family["cap"],
                family["measurement_bin"],
                data["SOURCE_ID"].tolist(),
                kwargs["growth_rate"],
            )
        )
        assert kwargs["config"] is config
        assert kwargs["common_geometry"]["family"] == family
        assert kwargs["bias"] == config["tracers"][tracer]["reconstruction"]["fiducial_bias"]
        assert kwargs["convention"] == "recsym"
        assert len(randoms["SOURCE_ID"]) == len(data["SOURCE_ID"]) == per_bin
        for catalogue in (data, randoms):
            for source, target in zip(("X", "Y", "Z_CART"), SHIFTED_COLUMNS):
                catalogue[target] = catalogue[source] + 1.0
        # A displacement far along the line of sight must not change bin membership.
        data["Z_CART_SHIFTED"][0] += 500.0
        return data, randoms, {"realized_nmesh": [8, 8, 8], "shell_diagnostics": {"shells": []}}

    monkeypatch.setattr(pipeline_selection, "read_selected_parent", reading)
    monkeypatch.setattr(pipeline_reconstruction, "run_reconstruction", reconstruct)
    common = {"unit_id": "parent", "tracer": tracer, "condition": condition, "realization_id": 0}
    solve_unit = {**common, "measurement_bins": names}
    solve_directory = tmp_path / "reconstruct" / "units" / common["unit_id"]
    solve_directory.mkdir(parents=True)
    result = pipeline_reconstruction.reconstruct_stage(run, solve_unit, solve_directory)
    _write_yaml(
        solve_directory / "result.yaml",
        {**run["metadata"], "stage": "reconstruct", "unit": solve_unit},
    )
    # One solve per cap and bin, caps outermost, each on exactly its own ordered partition.
    expected_order = [(cap, name) for cap in CAPS for name in names]
    assert [(cap, name) for cap, name, _, _ in solves] == expected_order
    assert reads == expected_order  # one live field: every bin is read on its own
    for cap, name, solved_ids, growth_rate in solves:
        assert solved_ids == expected_ids(cap, name)
        assert growth_rate == growth[name]
    assert result["catalogues_saved"] is True
    payload = json.loads((solve_directory / "reconstruction.json").read_text())
    assert payload["representation"] == "bin_shards_v1"
    assert len(payload["catalogues"]) == 2 * 2 * len(names)
    for entry in payload["catalogues"]:
        partition = entry["partition"]
        assert partition["tracer"] == tracer and partition["measurement_bin"] in names
        assert partition["redshift_column"] == (
            observed if entry["identity"]["role"] == "data" else "Z"
        )
    for cap in CAPS:
        diagnostics = payload["diagnostics"][cap]["reconstruction"]
        assert diagnostics["scope"] == SCOPE and set(diagnostics["bins"]) == set(names)
        for name in names:
            record = diagnostics["bins"][name]
            assert record["field_identity"] == f"{tracer}|bin={name}"
            assert record["window"] == preparation["windows"][name]
            assert record["reconstruction_inputs"]["growth_rate"] == growth[name]
    # Readers rebuild each bin from its shard: same identities, same order, shifted columns.
    for cap in CAPS:
        data, randoms = make(cap)
        whole_data, whole_randoms, diagnostics = pipeline_reconstruction.read_reconstructed_parent(
            run, solve_unit, cap
        )
        assert diagnostics is not None
        assert whole_data["SOURCE_ID"].tolist() == [
            identifier for name in names for identifier in expected_ids(cap, name)
        ]
        for name in names:
            shard_data, shard_randoms, _ = pipeline_reconstruction.read_reconstructed_parent(
                run, solve_unit, cap, bin_name=name
            )
            rows = np.isin(data["SOURCE_ID"], expected_ids(cap, name))
            assert shard_data["SOURCE_ID"].tolist() == expected_ids(cap, name)
            for key in data:
                np.testing.assert_array_equal(shard_data[key], data[key][rows])
            np.testing.assert_allclose(shard_data["X_SHIFTED"], data["X"][rows] + 1.0, atol=1e-4)
            np.testing.assert_allclose(
                shard_data["Z_CART_SHIFTED"][1:], data["Z_CART"][rows][1:] + 1.0, atol=1e-4
            )
            # The boundary crosser kept its membership and its original observed redshift.
            np.testing.assert_allclose(
                shard_data["Z_CART_SHIFTED"][0], data["Z_CART"][rows][0] + 501.0, atol=1e-3
            )
            assert len(shard_randoms["SOURCE_ID"]) == per_bin
            chosen = np.isin(randoms["SOURCE_ID"], shard_randoms["SOURCE_ID"])
            np.testing.assert_allclose(
                shard_randoms["Y_SHIFTED"], randoms["Y"][chosen] + 1.0, atol=1e-4
            )
    with pytest.raises(ValueError, match="outside reconstruction unit"):
        pipeline_reconstruction.read_reconstructed_parent(
            run, {**solve_unit, "measurement_bins": names[:1]}, "NGC", bin_name=names[-1]
        )
    # Clustering reads one bin at a time and references the displaced randoms by partition.
    measured = []

    def measure(data, randoms, **kwargs):
        cap = "SGC" if data["SOURCE_ID"].min() >= 100 else "NGC"
        measured.append((cap, data["SOURCE_ID"].tolist(), kwargs["reconstructed"]))
        assert len(data["SOURCE_ID"]) == len(randoms["SOURCE_ID"]) == per_bin
        if kwargs["reconstructed"]:
            assert kwargs["precomputed_random_counts"] is not None
        # Counts depend on the rows handed to the estimator, so equal counts below mean
        # equal membership rather than a constant fixture.
        return native_estimator(
            cap=cap,
            post=kwargs["reconstructed"],
            radial_edges=kwargs["s_edges_mpc_h"],
            mu_bins=kwargs["mu_bins"],
            identifier=int(data["SOURCE_ID"].sum()),
        )

    monkeypatch.setattr(pipeline_clustering, "compute_xi", measure)
    snapshots = {}
    for label, chosen in (("whole", names), ("narrow", [names[-1]])):
        directory = tmp_path / label
        directory.mkdir()
        measured.clear()
        pipeline_clustering.cluster_stage(
            run, {**common, "measurement_bins": chosen}, directory, nthreads=1
        )
        snapshots[label] = list(measured)
        entries = json.loads((directory / "measurements.json").read_text())["measurements"]
        # Two caps and their combination, pre and post, per requested bin.
        assert len(entries) == len(chosen) * 2 * 3
        assert {entry["identity"]["measurement_bin"] for entry in entries} == set(chosen)
        for entry in entries:
            if entry["identity"].get("cap") not in CAPS:
                continue
            reference = entry["randoms"]
            assert reference["representation"] == "measurement_random_reference_v1"
            assert reference["partition"]["measurement_bin"] == entry["identity"]["measurement_bin"]
            assert reference["identity"]["stage"] == "reconstruct"
            arrays = pipeline_clustering.read_measurement_randoms(entry, config)
            assert len(arrays["source_ids"]) == per_bin
            assert "shifted_positions" in arrays
    assert len(snapshots["whole"]) == 2 * len(names) * 2  # caps x bins x (pre, post)
    assert {tuple(ids) for _, ids, _ in snapshots["whole"]} == {
        tuple(expected_ids(cap, name)) for cap in CAPS for name in names
    }
    assert snapshots["narrow"] == [
        event
        for event in snapshots["whole"]
        if set(event[1]) == set(expected_ids(event[0], names[-1]))
    ]
    for state in ("prereconstruction", "postreconstruction"):
        filename = f"{names[-1]}--{state}--combined-counts.npy"
        whole = TwoPointCorrelationFunction.load(str(tmp_path / "whole" / filename))
        narrow = TwoPointCorrelationFunction.load(str(tmp_path / "narrow" / filename))
        np.testing.assert_array_equal(whole.corr, narrow.corr)


def test_per_bin_runs_refuse_broad_field_products_that_historical_runs_still_read(tmp_path):
    import yaml

    current, legacy = load_analysis_config(), load_legacy_config()
    tracer, name = "LRG", "LRG2"
    item = next(item for item in measurement_bins(tracer, current) if item["name"] == name)
    observed = observed_column(current)
    unit = {
        "unit_id": "LRG--sigma_0.01--mock-0",
        "tracer": tracer,
        "condition": "sigma_0.01",
        "realization_id": 0,
        "measurement_bins": [name],
    }
    metadata = {"run_name": "fixture", "code_commit": "fixture", "analysis_config_sha256": "f"}
    z = np.array([item["zmin"] - 0.05, item["zmin"], item["zmax"] - 0.01, item["zmax"]])
    catalogue = {
        "SOURCE_ID": np.arange(4, dtype=np.uint64),
        "Z": z,
        observed: z.copy(),
        "X": 100 + z,
        "Y": z,
        "Z_CART": 2 * z,
        "WEIGHT": np.ones(4),
    }
    for source, target in zip(("X", "Y", "Z_CART"), SHIFTED_COLUMNS):
        catalogue[target] = catalogue[source] + 0.5
    directory = tmp_path / "reconstruct" / "units" / unit["unit_id"]
    directory.mkdir(parents=True)
    run = {"root": tmp_path, "config": legacy, "metadata": metadata}
    catalogues = []
    for cap in CAPS:
        for role in ("data", "random"):
            identity = pipeline_catalog.catalogue_identity(
                run,
                tracer=tracer,
                condition=unit["condition"],
                mock_id=0,
                cap=cap,
                role=role,
                stage="reconstruct",
            )
            path = f"{cap}--{role}.npz"
            pipeline_catalog.save_catalogue(directory, path, catalogue, identity=identity)
            catalogues.append({"path": path, "identity": identity})
    payload = {
        "diagnostics": {cap: {"reconstruction": {}} for cap in CAPS},
        "catalogues": catalogues,
        "catalogues_saved": True,
    }
    (directory / "reconstruction.json").write_text(json.dumps(payload))
    (directory / "result.yaml").write_text(
        yaml.safe_dump({**metadata, "stage": "reconstruct", "unit": unit})
    )
    # The historical broad-field reader still cuts the complete catalogue by observed redshift.
    data, randoms, _ = pipeline_reconstruction.read_reconstructed_parent(
        run, unit, "NGC", bin_name=name
    )
    assert data["SOURCE_ID"].tolist() == [1, 2]
    assert randoms["SOURCE_ID"].tolist() == [1, 2]
    np.testing.assert_array_equal(data["X_SHIFTED"], catalogue["X"][1:3] + 0.5)
    # A per-bin run never consumes a broad-field solve, even for one requested bin.
    per_bin_run = {"root": tmp_path, "config": current, "metadata": metadata}
    with pytest.raises(ValueError, match="requires published bin shards"):
        pipeline_reconstruction.read_reconstructed_parent(per_bin_run, unit, "NGC", bin_name=name)
    with pytest.raises(ValueError, match="requires published bin shards"):
        pipeline_reconstruction.read_reconstructed_parent(per_bin_run, unit, "NGC")


# ------------------------------------------------------------------ selection geometry and reuse


def _stub_selection_inputs(monkeypatch, config, tracer, mids):
    import pzbao.clustering as clustering

    observed = observed_column(config)
    identifiers = config["analysis"]["mock_ids"]
    raw = {"data": {observed: mids.copy()}, "reservoir": {}}
    monkeypatch.setattr(
        pipeline_selection, "build_mock_cosmology", lambda config: LinearCosmology()
    )
    monkeypatch.setattr(
        pipeline_selection,
        "read_catalog_registration",
        lambda run, tracer: {
            "sources": [
                {"role": "reservoir", "mock_id": mock_id, "cap": cap}
                for mock_id in identifiers
                for cap in CAPS
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
        lambda cosmology, edges, area: np.full(len(edges) - 1, 1.0e9),
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
        lambda randoms, *, zmin, zmax, **kwargs: {
            "numerator": (zmin + zmax) / 2,
            "denominator": 1.0,
        },
    )
    monkeypatch.setattr(pipeline_selection, "exact_growth_rate", lambda zeff, cosmology: 0.8)
    monkeypatch.setattr(clustering, "derive_power_spectrum_geometry", lambda *args, **kwargs: {})


def test_selection_freezes_bin_geometry_and_growth_from_every_condition_and_mock(
    tmp_path, monkeypatch
):
    from pzbao.run import _write_yaml

    config = load_analysis_config()
    tracer = "LRG"
    observed = observed_column(config)
    identifiers = config["analysis"]["mock_ids"]
    tiers = [tier["label"] for tier in config["selection"]["tiers"]]
    bins = measurement_bins(tracer, config)
    names = [item["name"] for item in bins]
    mids = np.array([(item["zmin"] + item["zmax"]) / 2 for item in bins])
    _stub_selection_inputs(monkeypatch, config, tracer, mids)
    seen = []

    def prepared(raw, preparation, *, condition, mock_id, cap, **kwargs):
        seen.append((condition, mock_id, cap))
        high = (1000 if condition == "sigma_0.03" else 100) + mock_id
        count = len(names)
        data = {
            "SOURCE_ID": np.arange(count, dtype=np.uint64),
            observed: mids.copy(),
            "Z": mids.copy(),
            "X": np.linspace(0.0, high, count),
            "Y": np.linspace(0.0, high / 2, count),
            "Z_CART": np.linspace(0.0, high / 4, count),
        }
        randoms = {
            "SOURCE_ID": np.arange(count, dtype=np.uint64),
            "Z": mids.copy(),
            "X": np.full(count, -10.0 - mock_id),
            "Y": -5.0 - np.arange(count, dtype=float),
            "Z_CART": -7.0 - np.arange(count, dtype=float),
        }
        return data, randoms, {"randoms": {"attained_random_data_ratio": 2.5 + mock_id}}

    monkeypatch.setattr(pipeline_selection, "_prepared_parent", prepared)
    saved, runs = [], {}
    for workload in ("pilot", "full_grid"):
        root = tmp_path / workload
        destination = root / "select" / "units" / tracer
        destination.mkdir(parents=True)
        unit = {"unit_id": tracer, "tracer": tracer, "realization_ids": identifiers}
        run = {
            "root": root,
            "config": config,
            "metadata": {
                "interface_version": 2,
                "run_name": workload,
                "code_commit": "a",
                "analysis_config_sha256": "b",
                "workload": workload,
                "catalog_root": str(tmp_path),
            },
        }
        pipeline_selection.select_stage(run, unit, destination, cpus=1)
        _write_yaml(
            destination / "result.yaml", {**run["metadata"], "stage": "select", "unit": unit}
        )
        assert set(seen) == {(t, m, c) for t in tiers for m in identifiers for c in CAPS}
        seen.clear()
        document = json.loads((destination / "preparation.json").read_text())
        document.pop("catalogues")
        saved.append(document)
        runs[workload] = run
    # A workload selects solves, never geometry: both preparations freeze the same families.
    assert saved[0] == saved[1]
    document = saved[0]
    assert document["reconstruction_scope"] == SCOPE
    assert set(document["windows"]) == set(names)
    for item in bins:
        window = document["windows"][item["name"]]
        assert window["window_redshift_range"] == [item["zmin"], item["zmax"]]
        assert window["radial_padding_mpc_h"] == 0.0
    for cap in CAPS:
        geometry = document["geometry"][cap]
        assert set(geometry) == set(names)
        centres = set()
        for name, value in geometry.items():
            assert value["family"]["identity"].endswith(f"|bin={name}")
            assert value["family"]["cap"] == cap
            assert value["conditions"] == tiers
            assert value["calibration_mock_ids"] == identifiers
            centres.add(tuple(value["boxcenter_mpc_h"]))
        assert len(centres) == len(names)  # bin-specific extents, not one parent box
    for condition in tiers:
        for item in bins:
            record = document["effective_redshifts"][condition][item["name"]]
            assert record["z_eff"] == pytest.approx((item["zmin"] + item["zmax"]) / 2)
            assert record["growth_rate"] == 0.8
            assert record["denominator"] == pytest.approx(2 * len(identifiers))
    # The producer satisfies its own reader, and the reader refuses broad-field preparations.
    run = runs["pilot"]
    record = pipeline_selection.read_preparation(run, tracer)
    assert record["reconstruction_scope"] == SCOPE
    path = run["root"] / "select" / "units" / tracer / "preparation.json"
    original = json.loads(path.read_text())
    for damage, message in (
        (lambda doc: doc.pop("reconstruction_scope"), "cannot use a broad-field geometry"),
        (lambda doc: doc["windows"].pop(names[0]), "lacks its bin windows"),
        (
            lambda doc: doc["geometry"]["NGC"][names[0]]["family"].update(identity="other"),
            "different family or calibration cohort",
        ),
        (
            lambda doc: doc["geometry"]["SGC"][names[1]].update(calibration_mock_ids=[0]),
            "different family or calibration cohort",
        ),
        (
            lambda doc: doc["windows"][names[0]].update(radial_padding_mpc_h=10.0),
            "missing or foreign bin",
        ),
    ):
        damaged = deepcopy(original)
        damage(damaged)
        path.write_text(json.dumps(damaged))
        with pytest.raises(ValueError, match=message):
            pipeline_selection.read_preparation(run, tracer)
    path.write_text(json.dumps(original))
    pipeline_selection.read_preparation(run, tracer)


def _reuse_fixture(tmp_path, monkeypatch):
    """A validated three-tier broad-field selection and a five-tier per-bin primary target."""
    from pzbao.run import _write_yaml

    target_config = load_analysis_config()
    source_config = sensitivity_run_configs(load_legacy_config())["scan--R15"]
    tracer = "LRG"
    observed = observed_column(target_config)
    bins = measurement_bins(tracer, target_config)
    names = [item["name"] for item in bins]
    identifiers = target_config["analysis"]["mock_ids"]
    tiers = [tier["label"] for tier in target_config["selection"]["tiers"]]
    saved_tiers = source_config["analysis"]["workloads"]["smoothing_scan"]["sigma_z_tiers"]
    assert len(saved_tiers) == 3 and set(saved_tiers) < set(tiers)
    mids = np.array([(item["zmin"] + item["zmax"]) / 2 for item in bins])

    def make(condition, mock_id, cap, role):
        tier_index, cap_index = tiers.index(condition), CAPS.index(cap)
        count = 2 * len(names)
        redshift = np.repeat(mids, 2) + np.tile([-0.01, 0.01], len(names))
        base = (
            10_000 * (tier_index + 1)
            + 100 * mock_id
            + 50 * cap_index
            + (25 if role == "random" else 0)
        )
        offset = 0.1 if role == "random" else 0.0
        catalogue = {
            "SOURCE_ID": base + np.arange(count, dtype=np.uint64),
            "Z": redshift,
            "X": 100.0 + 10.0 * mock_id + offset + np.arange(count, dtype=float),
            "Y": tier_index + offset + 0.5 * np.arange(count, dtype=float),
            "Z_CART": cap_index + offset + 0.25 * np.arange(count, dtype=float),
            "WEIGHT": np.ones(count),
            "WEIGHT_FKP": np.full(count, 0.7),
        }
        if role == "data":
            catalogue[observed] = redshift + 0.002
        return catalogue

    source = {
        "root": tmp_path / "source",
        "config": source_config,
        "metadata": {
            "interface_version": 2,
            "run_name": "full-v2--scan--R15",
            "code_commit": "c" * 40,
            "analysis_config_sha256": "s" * 64,
            "workload": "smoothing_scan",
        },
    }
    directory = source["root"] / "select" / "units" / tracer
    directory.mkdir(parents=True)
    preparation = {
        "tracer": tracer,
        "realization_ids": identifiers,
        "sources": ["registered"],
        "catalogues": [],
        "geometry": {cap: {"family": {"cap": cap}} for cap in CAPS},
        "conditions": {
            tier: {cap: {"selection_diagnostics_by_mock": []} for cap in CAPS} for tier in tiers
        },
        "effective_redshifts": {
            tier: {
                "parent": {
                    "z_eff": 0.7,
                    "growth_rate": 0.8,
                    "numerator": 35.0,
                    "denominator": 50.0,
                },
                **{
                    item["name"]: {
                        "z_eff": mid,
                        "growth_rate": 0.8,
                        "numerator": 50.0 * mid,
                        "denominator": 50.0,
                    }
                    for item, mid in zip(bins, mids)
                },
            }
            for tier in tiers
        },
    }
    for condition in saved_tiers:
        for mock_id in identifiers:
            for cap in CAPS:
                for role in ("data", "random"):
                    identity = pipeline_catalog.catalogue_identity(
                        source,
                        tracer=tracer,
                        condition=condition,
                        mock_id=mock_id,
                        cap=cap,
                        role=role,
                        stage="select",
                    )
                    name = f"{condition}--mock-{mock_id}--{cap}--{role}.npz"
                    pipeline_catalog.save_catalogue(
                        directory, name, make(condition, mock_id, cap, role), identity=identity
                    )
                    preparation["catalogues"].append({"path": name, "identity": identity})
    unit = {"unit_id": tracer, "tracer": tracer, "realization_ids": identifiers}
    (directory / "preparation.json").write_text(json.dumps(preparation))
    _write_yaml(directory / "result.yaml", {**source["metadata"], "stage": "select", "unit": unit})

    target = {
        "root": tmp_path / "target",
        "config": target_config,
        "metadata": {
            "interface_version": 2,
            "run_name": "per-bin-v1--primary",
            "code_commit": "d" * 40,
            "analysis_config_sha256": "t" * 64,
            "workload": "full_grid",
        },
    }
    registrations = {"full-v2--scan--R15": ["same"], "per-bin-v1--primary": ["same"]}
    monkeypatch.setattr(
        pipeline_selection,
        "read_catalog_registration",
        lambda run, tracer: {"sources": registrations[run["metadata"]["run_name"]]},
    )
    monkeypatch.setattr(
        pipeline_selection, "build_mock_cosmology", lambda config: LinearCosmology()
    )
    monkeypatch.setattr(pipeline_selection, "exact_growth_rate", lambda zeff, cosmology: 0.8)
    monkeypatch.setattr(
        pipeline_selection,
        "random_shell_accumulator",
        lambda randoms, *, zmin, zmax, **kwargs: {
            "numerator": (zmin + zmax) / 2,
            "denominator": 1.0,
        },
    )
    reads = []
    monkeypatch.setattr(
        pipeline_selection,
        "_read_parent",
        lambda sources, **kwargs: reads.append((kwargs["mock_id"], kwargs["cap"])) or {"raw": True},
    )
    regenerated = []

    def prepared(raw, preparation, *, condition, mock_id, cap, config, **kwargs):
        assert raw == {"raw": True} and config is source_config
        regenerated.append((condition, mock_id, cap))
        return (
            make(condition, mock_id, cap, "data"),
            make(condition, mock_id, cap, "random"),
            {"fixture": True},
        )

    monkeypatch.setattr(pipeline_selection, "_prepared_parent", prepared)
    import pzbao.clustering as clustering

    monkeypatch.setattr(clustering, "derive_power_spectrum_geometry", lambda *args, **kwargs: {})
    return {
        "source": source,
        "target": target,
        "unit": unit,
        "make": make,
        "names": names,
        "tiers": tiers,
        "saved_tiers": saved_tiers,
        "identifiers": identifiers,
        "preparation": preparation,
        "directory": directory,
        "registrations": registrations,
        "reads": reads,
        "regenerated": regenerated,
    }


def test_selection_reuse_references_saved_tiers_and_regenerates_the_missing_cohort(
    tmp_path, monkeypatch
):
    from pzbao.run import _write_yaml

    fixture = _reuse_fixture(tmp_path, monkeypatch)
    source, target, unit = fixture["source"], fixture["target"], fixture["unit"]
    tiers, saved_tiers, identifiers = (
        fixture["tiers"],
        fixture["saved_tiers"],
        fixture["identifiers"],
    )
    missing = [tier for tier in tiers if tier not in saved_tiers]
    tracer = unit["tracer"]
    destination = target["root"] / "select" / "units" / tracer
    destination.mkdir(parents=True)
    result = pipeline_selection.reuse_selection_stage(target, unit, destination, source=source)
    _write_yaml(
        destination / "result.yaml", {**target["metadata"], "stage": "select", "unit": unit}
    )
    document = json.loads((destination / "preparation.json").read_text())
    entries = document["catalogues"]
    assert len(entries) == len(identifiers) * 2 * len(tiers) * 2
    by_tier = {}
    for entry in entries:
        by_tier.setdefault(entry["identity"]["condition"], set()).add(
            entry["path"].rsplit(".", 1)[1]
        )
    assert all(by_tier[tier] == {"json"} for tier in saved_tiers)
    assert all(by_tier[tier] == {"npz"} for tier in missing)
    # Only the missing tiers were regenerated, once per mock and cap, from the frozen selection.
    assert sorted(set(fixture["regenerated"])) == sorted(
        (tier, mock_id, cap) for tier in missing for mock_id in identifiers for cap in CAPS
    )
    assert len(fixture["regenerated"]) == len(missing) * len(identifiers) * 2
    assert sorted(set(fixture["reads"])) == sorted((m, c) for m in identifiers for c in CAPS)
    assert document["selection_source"]["run_name"] == source["metadata"]["run_name"]
    assert document["selection_source"]["root"] == str(source["root"])
    assert document["reconstruction_scope"] == SCOPE
    assert {p["path"] for p in result["products"]} >= {"preparation.json"}
    for cap in CAPS:
        for name, geometry in document["geometry"][cap].items():
            assert geometry["family"]["identity"].endswith(f"|bin={name}")
            assert geometry["conditions"] == tiers
    # Saved bin growth records were validated against the regenerated randoms and retained.
    for tier in tiers:
        for name in fixture["names"]:
            assert document["effective_redshifts"][tier][name] == (
                fixture["preparation"]["effective_redshifts"][tier][name]
            )
    # Aliases and regenerated catalogues read back through the ordinary per-bin reader.
    record = pipeline_selection.read_preparation(target, tracer)
    assert record["reconstruction_scope"] == SCOPE
    for condition in (saved_tiers[0], missing[-1]):
        for cap in CAPS:
            parent = {"tracer": tracer, "condition": condition, "realization_id": 7}
            data, randoms, _ = pipeline_selection.read_selected_parent(
                target, parent, cap, bin_name=fixture["names"][1]
            )
            expected = fixture["make"](condition, 7, cap, "data")
            rows = slice(2, 4)  # the second bin's two rows, in catalogue order
            for key in expected:
                np.testing.assert_array_equal(data[key], expected[key][rows])
            assert randoms["SOURCE_ID"].tolist() == (
                fixture["make"](condition, 7, cap, "random")["SOURCE_ID"][rows].tolist()
            )


def test_selection_reuse_rejects_foreign_cohorts_changed_selections_and_altered_saved_rows(
    tmp_path, monkeypatch
):
    fixture = _reuse_fixture(tmp_path, monkeypatch)
    source, target, unit = fixture["source"], fixture["target"], fixture["unit"]
    tracer = unit["tracer"]
    destination = target["root"] / "select" / "units" / tracer
    destination.mkdir(parents=True)
    transfer = {"config": transfer_run_configs(load_analysis_config())["self--R25"]}
    with pytest.raises(ValueError, match="cannot import an exclusive transfer cohort"):
        pipeline_selection.validate_selection_reuse(transfer, target, tracer)
    changed = deepcopy(source)
    changed["config"] = deepcopy(source["config"])
    changed["config"]["selection"]["radial_randoms"]["stable_thinning_seed"] += 1
    with pytest.raises(ValueError, match="changes inputs, weights, streams or density"):
        pipeline_selection.validate_selection_reuse(changed, target, tracer)
    fixture["registrations"]["per-bin-v1--primary"] = ["other"]
    with pytest.raises(ValueError, match="different registered parent inputs"):
        pipeline_selection.validate_selection_reuse(source, target, tracer)
    fixture["registrations"]["per-bin-v1--primary"] = ["same"]
    with pytest.raises(ValueError, match="does not cover the allocation cohort"):
        pipeline_selection.reuse_selection_stage(
            target, {**unit, "realization_ids": [0]}, destination, source=source
        )
    # A saved bin growth record that disagrees with the selected randoms is refused.
    path = fixture["directory"] / "preparation.json"
    original = path.read_text()
    damaged = json.loads(original)
    record = damaged["effective_redshifts"][fixture["saved_tiers"][0]][fixture["names"][0]]
    record.update(z_eff=record["z_eff"] + 0.01, numerator=record["numerator"] + 0.5)
    path.write_text(json.dumps(damaged))
    with pytest.raises(ValueError, match="differs from selected randoms"):
        pipeline_selection.reuse_selection_stage(target, unit, destination, source=source)
    path.write_text(original)
    # A partially saved pair is completed from the frozen selection only if it agrees exactly.
    partial = json.loads(original)
    removed = next(
        entry
        for entry in partial["catalogues"]
        if entry["identity"]["condition"] == fixture["saved_tiers"][0]
        and entry["identity"]["realization_id"] == 0
        and entry["identity"]["cap"] == "NGC"
        and entry["identity"]["role"] == "random"
    )
    partial["catalogues"].remove(removed)
    path.write_text(json.dumps(partial))
    for child in list(destination.iterdir()):
        child.unlink()
    pipeline_selection.reuse_selection_stage(target, unit, destination, source=source)
    document = json.loads((destination / "preparation.json").read_text())
    completed = {
        entry["identity"]["role"]: entry["path"].rsplit(".", 1)[1]
        for entry in document["catalogues"]
        if entry["identity"]["condition"] == fixture["saved_tiers"][0]
        and entry["identity"]["realization_id"] == 0
        and entry["identity"]["cap"] == "NGC"
    }
    assert completed == {"data": "json", "random": "npz"}
    for child in list(destination.iterdir()):
        child.unlink()
    make = fixture["make"]

    def perturbed(raw, preparation, *, condition, mock_id, cap, config, **kwargs):
        data = make(condition, mock_id, cap, "data")
        if (condition, mock_id, cap) == (fixture["saved_tiers"][0], 0, "NGC"):
            data["WEIGHT_FKP"] = data["WEIGHT_FKP"] + 1e-6
        return data, make(condition, mock_id, cap, "random"), {}

    monkeypatch.setattr(pipeline_selection, "_prepared_parent", perturbed)
    with pytest.raises(ValueError, match="differs from its frozen selection"):
        pipeline_selection.reuse_selection_stage(target, unit, destination, source=source)


# ------------------------------------------------------------------ fit checks on the new measurement


def test_core_bin_fit_checks_change_only_their_registered_fit_definition():
    from pzbao.run import stage_units

    config = load_analysis_config()
    pcs2 = derive_inference_config(config, "core_bin_pcs2_v1")
    assert pcs2["fitting"]["core_fit_check"] == "core_bin_pcs2_v1"
    assert pcs2["fitting"]["broadband_model"] == "pcs2"
    flat = derive_inference_config(config, "core_bin_flat_v1")
    assert flat["fitting"]["core_fit_check"] == "core_bin_flat_v1"
    assert flat["fitting"]["damping"]["applied_prior"] == "paper_finite_flat"
    assert flat["fitting"]["broadband_model"] == "power3"
    common = derive_inference_config(config, "core_bin_common_flat_v1")
    assert common["fitting"]["core_fit_check"] == "core_bin_common_flat_v1"
    assert common["fitting"]["damping"]["applied_prior"] == "common_flat_sensitivity"
    assert common["fitting"]["broadband_model"] == "power3"
    for derived in (pcs2, flat, common):
        validate_analysis_config(derived)
        for key in ("measurement", "selection", "tracers", "physics", "analysis"):
            assert derived[key] == config[key]
    unchanged = deepcopy(common)
    del unchanged["fitting"]["core_fit_check"], unchanged["fitting"]["damping"]["applied_prior"]
    assert unchanged == config
    validate_inference_config(config, pcs2, "core_bin_pcs2_v1")
    validate_inference_config(config, flat, "core_bin_flat_v1")
    validate_inference_config(config, common, "core_bin_common_flat_v1")
    for derived, other in ((flat, "core_bin_pcs2_v1"), (common, "core_bin_flat_v1")):
        with pytest.raises(ValueError, match="more than its configured fit definition"):
            validate_inference_config(config, derived, other)
    for bad in (
        sensitivity_run_configs(config)["scan--R25"],
        transfer_run_configs(config)["self--R25"],
        load_legacy_config(),
        pcs2,
    ):
        with pytest.raises(ValueError, match="primary measurement"):
            derive_inference_config(bad, "core_bin_flat_v1")
    scope = config["fitting"]["paper_comparisons"]
    units = stage_units("fit", flat, "full_grid", inference_definition="core_bin_flat_v1")
    assert units
    assert all(
        unit["condition"] == scope["condition"]
        and unit["reconstruction_state"] == scope["reconstruction_state"]
        and unit["measurement_bin"] in scope["measurement_bins"]
        for unit in units
    )
    assert units == stage_units(
        "fit", common, "full_grid", inference_definition="core_bin_common_flat_v1"
    )
    with pytest.raises(ValueError, match="reuse measured vectors"):
        stage_units("covariance", flat, "full_grid", inference_definition="core_bin_flat_v1")


def test_scan_centre_refits_move_only_the_transverse_centre_radius():
    """Revision 2.18: each scan radius evaluates the analytic centre at its own radius."""
    from pzbao.run import stage_units
    from pzbao.theory import analytic_postreconstruction_sigma_perp

    definition = "core_scan_radius_centre_v1"
    config = load_analysis_config()
    runs = sensitivity_run_configs(config)
    arguments = dict(
        tracer="LRG",
        attained_density_h3_mpc3=3.0e-4,
        linear_power=lambda k: 1.0e4 / (1.0 + (k / 0.05) ** 2),
        drag_scale_mpc_h=100.0,
    )
    centres = {}
    for name in ("scan--R15", "scan--R25", "scan--R35", "scan--R45"):
        source = runs[name]
        radius = source["measurement"]["reconstruction"]["smoothing_values_mpc_h"][0]
        assert source["fitting"]["damping"]["reference_smoothing_mpc_h"] == 15.0
        derived = derive_inference_config(source, definition)
        assert derived["fitting"]["core_fit_check"] == definition
        assert derived["fitting"]["damping"]["reference_smoothing_mpc_h"] == radius
        validate_analysis_config(derived)
        validate_inference_config(source, derived, definition)
        restored = deepcopy(derived)
        del restored["fitting"]["core_fit_check"]
        restored["fitting"]["damping"]["reference_smoothing_mpc_h"] = 15.0
        assert restored == source
        record = analytic_postreconstruction_sigma_perp(**arguments, config=derived)
        assert record["smoothing_radius_mpc_h"] == radius
        centres[radius] = record["analytic_center_mpc_h"]
        # Every scan route is refitted; covariances stay the scan run's own.
        assert stage_units("fit", derived, "smoothing_scan", inference_definition=definition) == (
            stage_units("fit", source, "smoothing_scan")
        )
        with pytest.raises(ValueError, match="reuse measured vectors"):
            stage_units("covariance", derived, "smoothing_scan", inference_definition=definition)
        drifted = deepcopy(derived)
        drifted["fitting"]["damping"]["applied_prior"] = "paper_finite_flat"
        with pytest.raises(ValueError, match="more than its configured fit definition"):
            validate_inference_config(source, drifted, definition)
    # The reference radius keeps its centre; wider radii change it.
    pinned = analytic_postreconstruction_sigma_perp(**arguments, config=runs["scan--R25"])
    assert pinned["analytic_center_mpc_h"] == centres[15.0] != centres[25.0]
    assert len(set(centres.values())) == 4
    for bad in (
        config,
        runs["reciso--R15"],
        derive_inference_config(runs["scan--R25"], definition),
    ):
        with pytest.raises(ValueError, match="smoothing-scan measurement pinned"):
            derive_inference_config(bad, definition)
    # The schema admits the own radius only under the refit, and the refit only on scan runs.
    own = deepcopy(runs["scan--R25"])
    own["fitting"]["damping"]["reference_smoothing_mpc_h"] = 25.0
    with pytest.raises(ValueError, match="reference-radius"):
        validate_analysis_config(own)
    refit = derive_inference_config(runs["scan--R25"], definition)
    refit["fitting"]["damping"]["reference_smoothing_mpc_h"] = 15.0
    with pytest.raises(ValueError, match="own scan radius"):
        validate_analysis_config(refit)
    marked = deepcopy(runs["reciso--R15"])
    marked["fitting"]["core_fit_check"] = definition
    with pytest.raises(ValueError, match="smoothing-scan runs only|own scan radius"):
        validate_analysis_config(marked)


def test_flat_checks_apply_their_support_only_on_the_comparison_routes():
    from pzbao.likelihood import applied_damping_prior

    config = load_analysis_config()
    scope = config["fitting"]["paper_comparisons"]
    names = ("sigma_perp", "sigma_parallel", "sigma_streaming")
    record = {
        "centers_mpc_h": dict(zip(names, (4.6, 6.0, 2.0))),
        "widths_mpc_h": dict(zip(names, (1.0, 2.0, 2.0))),
        "bounds_mpc_h": {name: [0.0, 20.0] for name in names},
        "unpriored_parameters": [],
        "sigma_parallel_photoz_mpc_h": 14.35,
        "scope": "final_science_fit",
    }
    before = deepcopy(record)
    route = {
        "measurement_bin": scope["measurement_bins"][-1],
        "condition": scope["condition"],
        "reconstruction_state": scope["reconstruction_state"],
    }
    assert applied_damping_prior(record, route, config) is record
    common = applied_damping_prior(
        record, route, derive_inference_config(config, "core_bin_common_flat_v1")
    )
    # Revision 2.11-P: identical [0, 50] support; the photo-z scale does not widen the parallel.
    assert common["bounds_mpc_h"] == {name: [0.0, 50.0] for name in names}
    assert common["unpriored_parameters"] == list(names)
    assert common["scope"] == "paper_common_flat_damping_comparison"
    assert common["centers_mpc_h"] == record["centers_mpc_h"]
    assert common["sigma_parallel_photoz_mpc_h"] == 14.35
    finite = applied_damping_prior(
        record, route, derive_inference_config(config, "core_bin_flat_v1")
    )
    assert finite["bounds_mpc_h"]["sigma_parallel"] == [0.0, float(np.hypot(10.0, 14.35))]
    assert finite["bounds_mpc_h"]["sigma_perp"] == [0.0, 10.0]
    assert record == before
    derived = derive_inference_config(config, "core_bin_common_flat_v1")
    for change in (
        {"condition": "sigma_0.01"},
        {"reconstruction_state": "prereconstruction"},
        {"measurement_bin": "LRG9"},
    ):
        with pytest.raises(ValueError, match="only inside their comparison"):
            applied_damping_prior(record, {**route, **change}, derived)
    wrong = deepcopy(derived)
    wrong["fitting"]["damping"]["applied_prior"] = "paper_finite_flat"
    with pytest.raises(ValueError, match="requires the common flat prior"):
        validate_analysis_config(wrong)
    unknown = deepcopy(config)
    unknown["fitting"]["damping"]["applied_prior"] = "uniform_100"
    with pytest.raises(ValueError, match="unknown applied damping prior"):
        validate_analysis_config(unknown)
    # The common prior belongs to its fit check alone, never to a primary or other run.
    unchecked = deepcopy(config)
    unchecked["fitting"]["damping"]["applied_prior"] = "common_flat_sensitivity"
    with pytest.raises(ValueError, match="per-bin check only"):
        validate_analysis_config(unchecked)
    unsupported = deepcopy(derived)
    del unsupported["fitting"]["damping"]["common_flat_sensitivity"]
    with pytest.raises(ValueError, match="configured support"):
        validate_analysis_config(unsupported)
