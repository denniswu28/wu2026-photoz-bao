"""Value-level contracts for isolated support experiments and saved membership evidence."""

import pzbao.diagnostics.selection as diag_selection
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import pytest
import yaml
from pzbao import selection as modifier
from pzbao.config import load_analysis_config, validate_boundary_variant
from tests.test_selection import _catalog


def reference_run(config=None):
    return {
        "root": Path("/reference"),
        "config": config or load_analysis_config(),
        "metadata": {
            "run_name": "reference",
            "code_commit": "a" * 40,
            "analysis_config_sha256": "b" * 64,
        },
    }


def make_variant(mode="truncate", tracer="LRG", depth=None):
    reference = reference_run()
    design = reference["config"]["selection"]["boundary_support"]["experiment_design"]
    target = next((item for item in design["targets"] if item["tracer"] == tracer))
    return diag_selection.variant_config(
        reference,
        target,
        mode=mode,
        depth=depth if depth is not None else 0.36 if mode == "extend" else 0.03,
        previous_depth=0.3,
        design=design,
    )


def per_bin_preparation_fields(config, tracer):
    """Minimal frozen per-bin windows and geometry families the preparation reader requires."""
    from pzbao.catalog import measurement_bins
    from pzbao.reconstruction import reconstruction_mask_family

    recon = config["measurement"]["reconstruction"]
    if recon["scope"] != "separate_measurement_bins" or "transfer_run" in recon:
        return {}
    bins = measurement_bins(tracer, config)
    radius = recon["smoothing_values_mpc_h"][0]
    return {
        "reconstruction_scope": recon["scope"],
        "windows": {
            item["name"]: {
                "tracer": tracer,
                "reconstruction_scope": recon["scope"],
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
                    "family": reconstruction_mask_family(
                        tracer=tracer,
                        cap=cap,
                        measurement_bin=item["name"],
                        smoothing_radius_mpc_h=radius,
                    ),
                    "conditions": [tier["label"] for tier in config["selection"]["tiers"]],
                    "calibration_mock_ids": list(config["analysis"]["mock_ids"]),
                }
                for item in bins
            }
            for cap in ("NGC", "SGC")
        },
    }


def reference_preparation(config, target, depth):
    design = config["selection"]["boundary_support"]["experiment_design"]
    field = config["tracers"][target["tracer"]]["field"]
    edge = field["zmin" if target["side"] == "below" else "zmax"]
    return {
        "tracer": target["tracer"],
        "realization_ids": config["analysis"]["mock_ids"],
        **per_bin_preparation_fields(config, target["tracer"]),
        "conditions": {
            design["reference_condition"]: {
                cap: {
                    "boundary_counts_by_mock": [
                        {
                            "realization_id": mock,
                            "edge_support": [
                                {
                                    "side": target["side"],
                                    "redshift": edge,
                                    "available_depth_redshift": depth + mock * 0.0001,
                                }
                            ],
                        }
                        for mock in config["analysis"]["mock_ids"]
                    ]
                }
                for cap in ("NGC", "SGC")
            }
        },
    }


def publish_reference_preparation(reference, target, depth):
    """Serialize a minimal published fixture through the real preparation reader boundary."""
    prepared = reference_preparation(reference["config"], target, depth)
    tracer = target["tracer"]
    directory = reference["root"] / "select" / "units" / tracer
    directory.mkdir(parents=True)
    workload = reference["config"]["analysis"]["workloads"][reference["metadata"]["workload"]]
    unit = {"unit_id": tracer, "tracer": tracer, "realization_ids": workload["mock_ids"]}
    (directory / "result.yaml").write_text(
        yaml.safe_dump({**reference["metadata"], "stage": "select", "unit": unit})
    )
    (directory / "preparation.json").write_text(json.dumps(prepared))
    return prepared


def test_fixed_depth_uses_every_saved_mock_and_cap_and_refuses_missing_records():
    reference = reference_run()
    config = reference["config"]
    design = config["selection"]["boundary_support"]["experiment_design"]
    target = design["targets"][0]
    depths = [0.09 + mock * 0.0001 for mock in config["analysis"]["mock_ids"]]
    prepared = {
        "conditions": {
            design["reference_condition"]: {
                cap: {
                    "boundary_counts_by_mock": [
                        {
                            "realization_id": mock,
                            "edge_support": [
                                {
                                    "side": "below",
                                    "redshift": 0.1,
                                    "available_depth_redshift": value,
                                }
                            ],
                        }
                        for mock, value in enumerate(depths)
                    ]
                }
                for cap in ("NGC", "SGC")
            }
        }
    }
    result = diag_selection.fixed_depth(prepared, target=target, config=config, design=design)
    assert result["normalized_depth"] == pytest.approx(0.09 / (0.03 * 1.1))
    assert result["depth_redshift"] == pytest.approx(0.009)
    assert len(result["source_rows"]) == 50
    prepared["conditions"][design["reference_condition"]]["SGC"]["boundary_counts_by_mock"].pop()
    with pytest.raises(ValueError, match="each realization"):
        diag_selection.fixed_depth(prepared, target=target, config=config, design=design)


@pytest.mark.parametrize("legacy_reference", [False, True])
def test_variant_roundtrip_preserves_science_and_rejects_other_changes(tmp_path, legacy_reference):
    reference = reference_run()["config"]
    if legacy_reference:
        reference["selection"]["boundary_support"].pop("experiment_design")
    for mode in ("truncate", "extend"):
        config = make_variant(mode)
        path = tmp_path / (mode + ".yaml")
        path.write_text(yaml.safe_dump(config))
        saved = load_analysis_config(path)
        validate_boundary_variant(reference, saved)
        assert saved == config
        changed = deepcopy(saved)
        changed["selection"]["stable_randomness"]["seed"] += 1
        with pytest.raises(ValueError, match="beyond its declared support"):
            validate_boundary_variant(reference, changed)
        changed = deepcopy(saved)
        changed["analysis"]["workloads"]["boundary"]["mock_ids"] = [0]
        with pytest.raises(ValueError, match="complete configured mock ensemble"):
            validate_boundary_variant(reference, changed)
        changed = deepcopy(saved)
        changed["selection"]["boundary_support"]["experiment_design"]["targets"].pop(0)
        with pytest.raises(ValueError, match="configured design"):
            validate_boundary_variant(reference, changed)


def test_disabled_experiment_and_old_snapshot_produce_identical_values():
    config = load_analysis_config()
    old = deepcopy(config)
    old["selection"]["boundary_support"].pop("experiment_design")
    data = _catalog([1, 2, 3, 4], [0.099, 0.101, 0.201, 0.401])
    results = [
        modifier.prepare_data_for_selection(
            data, tracer="BGS_BRIGHT-21.5", tier_label="sigma_0.003", cap="NGC", config=item
        )
        for item in (config, old)
    ]
    assert results[0][1] == results[1][1]
    for key in results[0][0]:
        np.testing.assert_array_equal(results[0][0][key], results[1][0][key])


@pytest.mark.parametrize(
    "tracer,z,expected",
    [
        ("LRG", [0.39, 0.4, 0.99, 1.0, 1.029, 1.03, 1.04], [0, 1, 2, 3, 4]),
        ("BGS_BRIGHT-21.5", [0.06, 0.07, 0.099, 0.1, 0.2, 0.4, 0.7], [1, 2, 3, 4, 5, 6]),
    ],
)
def test_native_cut_endpoints_and_opposite_edge(tracer, z, expected):
    config = make_variant(tracer=tracer)
    data = modifier.attach_stable_randomness(
        _catalog(list(range(len(z))), z), role="data", config=config
    )
    before = deepcopy(data)
    selected, record = modifier.apply_boundary_experiment(
        data, tracer=tracer, tier_label="sigma_0.003", config=config
    )
    np.testing.assert_array_equal(selected["SOURCE_ID"], expected)
    for key in data:
        np.testing.assert_array_equal(data[key], before[key])
        np.testing.assert_array_equal(selected[key], data[key][expected])
    assert record["removed_clustering_count"] == len(z) - len(expected)
    unchanged, record = modifier.apply_boundary_experiment(
        data, tracer=tracer, tier_label="native", config=config
    )
    assert unchanged is data and record is None


def test_lower_cut_clips_at_zero_and_preserves_nonnegative_support():
    config = make_variant(tracer="BGS_BRIGHT-21.5", depth=0.2)
    data = _catalog([1, 2, 3], [-0.001, 0.0, 0.4])
    selected, _ = modifier.apply_boundary_experiment(
        data, tracer="BGS_BRIGHT-21.5", tier_label="sigma_0.003", config=config
    )
    np.testing.assert_array_equal(selected["SOURCE_ID"], [2, 3])


def test_cut_follows_physical_dedup_and_screen_survives_serialization(tmp_path):
    config = make_variant(depth=0.0005)
    reference = reference_run()["config"]
    data = _catalog([10, 11, 8, 13], [0.405, 0.99, 1.004, 1.0002])
    reservoir = _catalog([20, 1, 22], [1.004, 1.001, 1.0003])
    reservoir["RA"][0], reservoir["DEC"][0] = (data["RA"][2], data["DEC"][2])
    record = {"tracer": "LRG", "sample_identity": "paired"}
    pairs = [
        modifier.prepare_data_for_selection(
            data,
            tracer="LRG",
            tier_label="sigma_0.003",
            cap="NGC",
            reservoir=reservoir,
            reservoir_record=record,
            config=item,
        )
        for item in (reference, config)
    ]
    assert pairs[1][1]["boundary_counts"]["overlap_removed_row_count"] == 1
    assert pairs[1][1]["boundary_experiment"]["removed_clustering_count"] == 1
    assert pairs[1][1]["boundary_experiment"]["removed_reservoir_count"] == 1
    saved = []
    for index, (catalog, _) in enumerate(pairs):
        path = tmp_path / f"selected-{index}.npz"
        np.savez(path, **catalog)
        with np.load(path, allow_pickle=False) as archive:
            saved.append({key: archive[key] for key in archive.files})
    rows = diag_selection.compare_membership(*saved, config=config, tracer="LRG")
    assert rows[0]["lost_count"] == 2 and rows[0]["added_count"] == 0
    assert rows[-1]["lost_count"] == 2
    assert rows[0]["variant_count"] == len(pairs[1][0]["SOURCE_ID"])
    saved[1]["PHOTOZ_DEVIATE"][0] += 1
    with pytest.raises(ValueError, match="surviving values"):
        diag_selection.compare_membership(*saved, config=config, tracer="LRG")


def test_extension_does_not_change_other_conditions_or_exterior_clustering():
    config = make_variant("extend")
    data = _catalog([1, 2, 3, 4], [0.2, 1.29, 1.3, 1.34])
    data["IS_RESERVOIR"] = np.array([True, True, True, False])
    data["IS_BOUNDARY_EXTENSION"] = np.array([False, False, True, False])
    selected, _ = modifier.apply_boundary_experiment(
        data, tracer="LRG", tier_label="sigma_0.003", config=config
    )
    np.testing.assert_array_equal(selected["SOURCE_ID"], [1, 2, 4])
    selected, _ = modifier.apply_boundary_experiment(
        data, tracer="LRG", tier_label="sigma_0.03", config=config
    )
    np.testing.assert_array_equal(selected["SOURCE_ID"], data["SOURCE_ID"])


def test_paired_profiles_keep_asymmetry_missing_intervals_and_rails():
    parameter = {
        "basename": "qiso",
        "value": 1.0,
        "minos_offsets": [-0.01, 0.03],
        "interval_flags": {
            "is_valid": True,
            "lower_valid": True,
            "upper_valid": True,
            "at_upper_limit": True,
        },
    }
    left = {"individual_fits": [{"realization_id": 0, "parameters": [parameter]}]}
    right = deepcopy(left)
    right["individual_fits"][0]["parameters"][0].update(value=1.004, minos_offsets=[-0.02, 0.04])
    rows, summaries = diag_selection.paired_profile_rows(left, right, [0], ["qiso"])
    assert rows[0]["reference"]["minos_offsets"] == [-0.01, 0.03]
    assert rows[0]["reference"]["interval_flags"]["at_upper_limit"]
    assert summaries[0]["alpha_shift_sigma_1"] == pytest.approx(0.2)
    assert summaries[0]["interval_width_fractional_change"] == pytest.approx(0.5)
    rows, summaries = diag_selection.paired_profile_rows(left, right, [0, 1], ["qiso"])
    assert len(rows) == 3 and rows[1]["reference_half_width"] is None
    assert summaries[0]["alpha_shift_sigma_1"] is None
    right["individual_fits"][0]["parameters"][0]["interval_flags"]["is_valid"] = False
    rows, summaries = diag_selection.paired_profile_rows(left, right, [0], ["qiso"])
    assert summaries[0]["interval_width_fractional_change"] is None
    assert rows[0]["variant"]["minos_offsets"] == [-0.02, 0.04]


@pytest.mark.parametrize("role", ["data", "random", "reservoir"])
def test_preparation_rejects_equal_count_catalogue_changes_before_reading_fields(
    tmp_path, monkeypatch, role
):
    from astropy.table import Table
    from pzbao import catalog
    import pzbao.selection as pipeline_selection
    from tests.test_catalog import _registered_inputs

    config = make_variant()
    catalog_root = tmp_path / "catalogues"
    initial = _registered_inputs(catalog_root, config, mock_id=0)
    path = Path(next((record["path"] for record in initial if record["role"] == role)))
    original = {"TARGETID": [1, 2], "RA": [220.0, 221.0], "DEC": [50.0, 51.0], "Z": [0.5, 0.9]}
    Table(original).write(path, overwrite=True)
    reference = catalog.register_parent_inputs(catalog_root, tracer="LRG", mock_id=0, config=config)
    unchanged = deepcopy(reference)
    for record in unchanged:
        record["path"] = str(tmp_path / "identical-copy" / Path(record["path"]).name)
    catalog.validate_boundary_input_sources(reference, unchanged, tracer="LRG", config=config)
    Table({**original, "RA": [220.1, 221.0]}).write(path, overwrite=True)
    np.testing.assert_array_equal(Table.read(path)["Z"], original["Z"])
    assert len(Table.read(path)) == len(original["Z"])
    reference_run_ = {"root": tmp_path / "reference"}
    run = {
        "config": config,
        "metadata": {"catalog_root": str(catalog_root)},
        "boundary_reference": reference_run_,
    }
    monkeypatch.setattr(pipeline_selection, "build_mock_cosmology", lambda config: object())
    monkeypatch.setattr(
        pipeline_selection,
        "read_catalog_registration",
        lambda run, tracer: {
            "sources": catalog.register_parent_inputs(
                catalog_root, tracer=tracer, mock_id=0, config=config
            )
        },
    )
    monkeypatch.setattr(
        pipeline_selection, "read_preparation", lambda run, tracer: {"sources": reference}
    )
    monkeypatch.setattr(
        pipeline_selection,
        "_read_parent",
        lambda *args, **kwargs: pytest.fail("changed source reached preparation"),
    )
    unit = {"tracer": "LRG", "realization_ids": config["analysis"]["mock_ids"]}
    with pytest.raises(ValueError, match="input bytes differ"):
        pipeline_selection.select_stage(run, unit, tmp_path / "output", cpus=1)


def test_extension_sources_must_name_the_frozen_original_slice(tmp_path):
    from pzbao.catalog import validate_boundary_input_sources

    config = make_variant("extend")
    reference = [
        {
            "tracer": "LRG",
            "mock_id": 0,
            "cap": "NGC",
            "role": "reservoir",
            "random_split": None,
            "path": str(tmp_path / "original.fits"),
            "size_bytes": 1,
            "sha256": "e" * 64,
        }
    ]
    variant = deepcopy(reference)
    variant[0].update(path=str(tmp_path / "extended.fits"), sha256="f" * 64, size_bytes=2)
    variant[0]["extension_reference"] = {
        "path": reference[0]["path"],
        "sha256": reference[0]["sha256"],
        "redshift_dtype": "f8",
    }
    validate_boundary_input_sources(reference, variant, tracer="LRG", config=config)
    for key, wrong in (("sha256", "a" * 64), ("path", str(tmp_path / "another.fits"))):
        bad = deepcopy(variant)
        bad[0]["extension_reference"][key] = wrong
        with pytest.raises(ValueError, match="frozen original slice"):
            validate_boundary_input_sources(reference, bad, tracer="LRG", config=config)
