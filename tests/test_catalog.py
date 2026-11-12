"""Deterministic fixture tests for catalogue input and identity behavior."""

import numpy as np
import pytest
import hashlib
import json

import pzbao.catalog as catalog
from pzbao.catalog import (
    assign_source_identity,
    build_mock_cosmology,
    catalogue_url,
    concatenate_catalogs,
    filter_redshift,
    filter_galactic_cap,
    footprint_area_deg2,
    measurement_bins,
    reconstruction_field,
    validate_catalog,
)
from pzbao.config import load_analysis_config


def _catalog(order=(0, 1, 2)):
    values = np.asarray(order)
    return {
        "TARGETID": np.array([100, 200, 300])[values],
        "RA": np.array([1.0, 2.0, 3.0])[values],
        "DEC": np.array([4.0, 5.0, 6.0])[values],
        "Z": np.array([0.1, 0.2, 0.4])[values],
    }


def test_fields_and_bins_are_exact_for_both_tracers():
    config = load_analysis_config()
    assert reconstruction_field("BGS_BRIGHT-21.5", config) == {"zmin": 0.1, "zmax": 0.4}
    assert [item["name"] for item in measurement_bins("LRG", config)] == [
        "LRG1",
        "LRG2",
        "LRG3",
    ]
    assert measurement_bins("BGS_BRIGHT-21.5", config)[0] == {
        "name": "BGS1",
        "zmin": 0.1,
        "zmax": 0.2,
    }


def test_measured_cap_areas_sum_exactly():
    config = load_analysis_config()
    for tracer in config["tracers"]:
        expected = footprint_area_deg2(tracer, "NGC", config) + footprint_area_deg2(
            tracer, "SGC", config
        )
        assert footprint_area_deg2(tracer, "GCcomb", config) == expected


def test_configured_mock_cosmology_matches_installed_provider():
    pytest.importorskip("cosmoprimo")
    cosmology = build_mock_cosmology(load_analysis_config())
    assert cosmology.h == pytest.approx(0.6736)


def test_upstream_url_routing_is_tracer_and_role_specific():
    config = load_analysis_config()
    bgs = catalogue_url(0, "BGS_BRIGHT-21.5", "data", "NGC", config=config)
    lrg = catalogue_url(0, "LRG", "random", "SGC", random_split=3, config=config)
    assert "/bright/v1/mock0/" in bgs
    assert bgs.endswith("BGS_BRIGHT-21.5_complete_NGC_clustering.dat.fits")
    assert lrg.endswith("LRG_complete_SGC_3_clustering.ran.fits")
    for tracer in ("LRG", "BGS_BRIGHT-21.5"):
        with pytest.raises(ValueError, match=r"logical NGC\+SGC only"):
            catalogue_url(0, tracer, "data", "GCcomb", config=config)
    with pytest.raises(ValueError, match="random_split"):
        catalogue_url(0, "LRG", "random", "NGC", config=config)

    invalid = load_analysis_config()
    invalid["tracers"]["LRG"]["catalogue"]["cap_strategy"] = "implicit_fallback"
    with pytest.raises(ValueError, match="requires physical NGC and SGC"):
        catalogue_url(0, "LRG", "data", "NGC", config=invalid)

    invalid = load_analysis_config()
    invalid["tracers"]["LRG"]["catalogue"]["data_template"] += "?split={split}"
    with pytest.raises(ValueError, match="must not consume a random split"):
        catalogue_url(0, "LRG", "data", "NGC", config=invalid)

    invalid["tracers"]["LRG"]["catalogue"]["data_template"] = "data-{split:03d}.fits"
    with pytest.raises(ValueError, match="must not consume a random split"):
        catalogue_url(0, "LRG", "data", "NGC", config=invalid)

    invalid["tracers"]["LRG"]["catalogue"]["data_template"] = "data-{split.real}.fits"
    with pytest.raises(ValueError, match="must not consume a random split"):
        catalogue_url(0, "LRG", "data", "NGC", config=invalid)

    invalid = load_analysis_config()
    invalid["tracers"]["LRG"]["catalogue"]["random_template"] = "random.fits"
    with pytest.raises(ValueError, match="consume split exactly once"):
        catalogue_url(0, "LRG", "random", "NGC", random_split=0, config=invalid)


def test_source_identity_is_object_bound_not_row_bound():
    first = assign_source_identity(_catalog(), tracer="LRG", mock_id=2, cap="NGC", role="data")
    second = assign_source_identity(
        _catalog((2, 0, 1)), tracer="LRG", mock_id=2, cap="NGC", role="data"
    )
    by_target_first = dict(zip(first["TARGETID"], first["SOURCE_ID"]))
    by_target_second = dict(zip(second["TARGETID"], second["SOURCE_ID"]))
    assert by_target_first == by_target_second


@pytest.mark.parametrize("role", ["data", "reservoir"])
def test_data_derived_identity_requires_targetid(role):
    source = _catalog()
    source.pop("TARGETID")
    with pytest.raises(KeyError, match="requires TARGETID"):
        assign_source_identity(source, tracer="LRG", mock_id=2, cap="NGC", role=role)


def test_random_source_identity_is_split_bound():
    with pytest.raises(ValueError, match="requires a non-negative random_split"):
        assign_source_identity(_catalog(), tracer="LRG", mock_id=2, cap="NGC", role="random")
    split_zero = assign_source_identity(
        _catalog(), tracer="LRG", mock_id=2, cap="NGC", role="random", random_split=0
    )
    split_one = assign_source_identity(
        _catalog(), tracer="LRG", mock_id=2, cap="NGC", role="random", random_split=1
    )
    assert not np.array_equal(split_zero["SOURCE_ID"], split_one["SOURCE_ID"])

    altered = _catalog()
    altered["Z"] = np.array([9.0, 8.0, 7.0])
    altered["PHOTSYS"] = np.array(["S", "N", "S"])
    same_angular_identity = assign_source_identity(
        altered, tracer="LRG", mock_id=2, cap="NGC", role="random", random_split=0
    )
    np.testing.assert_array_equal(split_zero["SOURCE_ID"], same_angular_identity["SOURCE_ID"])

    lower_precision = load_analysis_config()
    lower_precision["selection"]["stable_randomness"]["random_coordinate_significant_digits"] = 2
    precise = _catalog()
    precise["RA"] = np.array([1.2345, 2.3456, 3.4567])
    exact = assign_source_identity(
        precise, tracer="LRG", mock_id=2, cap="NGC", role="random", random_split=0
    )
    rounded = assign_source_identity(
        precise,
        tracer="LRG",
        mock_id=2,
        cap="NGC",
        role="random",
        random_split=0,
        config=lower_precision,
    )
    assert not np.array_equal(exact["SOURCE_ID"], rounded["SOURCE_ID"])


def test_half_open_cuts_keep_bgs_floor_and_exclude_upper_edge():
    selected = filter_redshift(_catalog(), 0.1, 0.4)
    np.testing.assert_array_equal(selected["TARGETID"], [100, 200])


def test_combined_parent_is_split_by_galactic_latitude_without_mutation():
    pytest.importorskip("astropy")
    source = {
        "TARGETID": np.array([100, 200]),
        # ICRS positions of the north and south Galactic poles.
        "RA": np.array([192.85948, 12.85948]),
        "DEC": np.array([27.12825, -27.12825]),
        "Z": np.array([0.5, 0.5]),
        # Deliberately the same value: PHOTSYS is not the cap selector.
        "PHOTSYS": np.array(["N", "N"]),
    }
    north = filter_galactic_cap(source, "NGC")
    south = filter_galactic_cap(source, "SGC")
    np.testing.assert_array_equal(north["TARGETID"], [100])
    np.testing.assert_array_equal(south["TARGETID"], [200])
    np.testing.assert_array_equal(source["TARGETID"], [100, 200])

    invalid = {name: values.copy() for name, values in source.items()}
    invalid["RA"][0] = np.nan
    with pytest.raises(ValueError, match="finite RA and DEC"):
        filter_galactic_cap(invalid, "NGC")


def test_catalog_contract_and_concatenation_fail_closed():
    with pytest.raises(ValueError, match="inconsistent lengths"):
        validate_catalog({"A": np.ones(2), "B": np.ones(3)})
    left = {"A": np.array([1]), "B": np.array([2])}
    right = {"A": np.array([3]), "B": np.array([4])}
    combined = concatenate_catalogs([left, right])
    np.testing.assert_array_equal(combined["A"], [1, 3])
    with pytest.raises(KeyError, match="identical columns"):
        concatenate_catalogs([left, {"A": np.array([3])}])


def _registered_inputs(tmp_path, config, tracer="LRG", mock_id=2):
    for cap in ("NGC", "SGC"):
        for role, split in [("data", None)] + [
            ("random", value)
            for value in range(config["tracers"][tracer]["catalogue"]["random_splits"])
        ]:
            path = catalog.planned_catalogue_path(
                tmp_path, mock_id, tracer, role, cap, random_split=split, config=config
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"registered synthetic source")
    reservoir = catalog.reservoir_slice_path(tmp_path, mock_id, tracer, config)
    reservoir.parent.mkdir(parents=True, exist_ok=True)
    reservoir.write_bytes(b"registered synthetic reservoir")
    return catalog.register_parent_inputs(tmp_path, tracer=tracer, mock_id=mock_id, config=config)


def test_registration_requires_every_configured_split_and_both_caps(tmp_path):
    config = load_analysis_config()
    records = _registered_inputs(tmp_path, config)
    assert len(records) == 2 * (config["tracers"]["LRG"]["catalogue"]["random_splits"] + 2)
    randoms = [
        record for record in records if record["role"] == "random" and record["cap"] == "SGC"
    ]
    assert [record["random_split"] for record in randoms] == list(range(18))
    from pathlib import Path

    Path(randoms[3]["path"]).unlink()
    with pytest.raises(FileNotFoundError, match="SGC random"):
        catalog.register_parent_inputs(tmp_path, tracer="LRG", mock_id=2, config=config)


@pytest.mark.parametrize("role", ["data", "reservoir", "random"])
def test_parallel_ids_preserve_literal_tokens_and_global_duplicate_checks(monkeypatch, role):
    from joblib import Parallel, parallel_config

    source = _catalog()
    source["RA"] = np.array([0.0, -0.0, 1e-12])
    source["DEC"] = np.array([-12.5, 90.0, 0.123456789012345])
    config = load_analysis_config()
    config["selection"]["stable_randomness"]["random_coordinate_significant_digits"] = 12
    tokens = (
        [
            b"LRG|2|NGC|random|split=0|0|-12.5",
            b"LRG|2|NGC|random|split=0|-0|90",
            b"LRG|2|NGC|random|split=0|1e-12|0.123456789012",
        ]
        if role == "random"
        else [f"LRG|2|NGC|{role}|{value}".encode() for value in (100, 200, 300)]
    )
    expected = np.array(
        [
            int.from_bytes(hashlib.blake2b(token, digest_size=8).digest(), "little")
            for token in tokens
        ],
        dtype="u8",
    )
    kwargs = dict(
        tracer="LRG",
        mock_id=2,
        cap="NGC",
        role=role,
        random_split=0 if role == "random" else None,
        config=config,
    )
    monkeypatch.setattr(catalog, "_ROW_CHUNK_BYTES", 16)
    with parallel_config(backend="loky", inner_max_num_threads=1):
        with Parallel(n_jobs=2, return_as="generator", max_nbytes=None) as parallel:
            for order in ([0, 1, 2], [2, 0, 1]):
                reordered = {key: value[order] for key, value in source.items()}
                result = assign_source_identity(reordered, parallel=parallel, **kwargs)
                np.testing.assert_array_equal(result["SOURCE_ID"], expected[order])
                for key in reordered:
                    np.testing.assert_array_equal(result[key], reordered[key])
                assert "SOURCE_ID" not in reordered
            duplicate = {key: value[[0, 1, 0]] for key, value in source.items()}
            with pytest.raises(ValueError, match="duplicate source keys"):
                assign_source_identity(duplicate, parallel=parallel, **kwargs)
            if role == "random":
                invalid = {key: value.copy() for key, value in source.items()}
                invalid["RA"][-1] = np.nan
                with pytest.raises(ValueError, match="finite RA and DEC"):
                    assign_source_identity(invalid, parallel=parallel, **kwargs)


def test_catalog_driver_reuses_pool_and_preserves_saved_registration(tmp_path, monkeypatch, capsys):
    from pzbao import run as driver

    config = load_analysis_config()
    config["analysis"]["mock_ids"] = [2]
    records = _registered_inputs(tmp_path, config)
    monkeypatch.setattr(catalog, "_ROW_CHUNK_BYTES", 16)
    calls = []

    def read(record, *, parallel=None, **kwargs):
        calls.append(parallel)
        result = assign_source_identity(
            _catalog(), random_split=record["random_split"], parallel=parallel, **kwargs
        )
        serial = assign_source_identity(_catalog(), random_split=record["random_split"], **kwargs)
        np.testing.assert_array_equal(result["SOURCE_ID"], serial["SOURCE_ID"])
        return result

    monkeypatch.setattr(catalog, "read_registered_catalogue", read)
    run = {"config": config, "metadata": {"catalog_root": str(tmp_path)}}
    unit = {"tracer": "LRG", "realization_ids": [2]}
    driver._science("catalog", run, unit, tmp_path, cpus=2)
    assert len(calls) == len(records)
    assert calls[0] is not None and all(pool is calls[0] for pool in calls)
    saved = json.loads((tmp_path / "catalog.json").read_text())
    assert saved == {"tracer": "LRG", "realization_ids": [2], "sources": records}
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert sum(event["phase"] == "validation_complete" for event in events) == len(records)


def test_native_registered_reader_rejects_wrong_source_and_preserves_stable_ids(
    tmp_path, monkeypatch
):
    fits = pytest.importorskip("astropy.io.fits")

    config = load_analysis_config()
    records = _registered_inputs(tmp_path, config)
    record = next(
        record for record in records if record["role"] == "data" and record["cap"] == "NGC"
    )
    source = _catalog()
    columns = [
        fits.Column(name=name, array=value, format="K" if name == "TARGETID" else "D")
        for name, value in source.items()
    ]
    fits.BinTableHDU.from_columns(columns).writeto(record["path"], overwrite=True)
    record = next(
        item
        for item in catalog.register_parent_inputs(tmp_path, tracer="LRG", mock_id=2, config=config)
        if item["role"] == "data" and item["cap"] == "NGC"
    )
    loaded = catalog.read_registered_catalogue(
        record, tracer="LRG", mock_id=2, cap="NGC", role="data", config=config
    )
    expected = assign_source_identity(
        source, tracer="LRG", mock_id=2, cap="NGC", role="data", config=config
    )
    np.testing.assert_array_equal(loaded["SOURCE_ID"], expected["SOURCE_ID"])
    from joblib import Parallel, parallel_config

    monkeypatch.setattr(catalog, "_ROW_CHUNK_BYTES", 8)
    with parallel_config(backend="loky", inner_max_num_threads=1):
        with Parallel(n_jobs=2, return_as="generator", max_nbytes=None) as parallel:
            parallel_loaded = catalog.read_registered_catalogue(
                record,
                tracer="LRG",
                mock_id=2,
                cap="NGC",
                role="data",
                config=config,
                parallel=parallel,
            )
    for key in loaded:
        np.testing.assert_array_equal(parallel_loaded[key], loaded[key])
    for changed in (
        {"mock_id": True},
        {"tracer": "BGS_BRIGHT-21.5"},
        {"cap": "SGC"},
        {"size_bytes": 1},
        {"source_url": "wrong"},
        {"random_split": 3},
    ):
        with pytest.raises(ValueError):
            catalog.read_registered_catalogue(
                {**record, **changed},
                tracer="LRG",
                mock_id=2,
                cap="NGC",
                role="data",
                config=config,
            )


def test_reservoir_native_header_identifies_source_without_parent_output_credential(tmp_path):
    fits = pytest.importorskip("astropy.io.fits")

    config = load_analysis_config()
    records = _registered_inputs(tmp_path, config)
    record = next(
        record for record in records if record["role"] == "reservoir" and record["cap"] == "NGC"
    )
    catalogue = {
        "TARGETID": np.array([10, 20]),
        "RA": np.array([192.85948, 12.85948]),
        "DEC": np.array([27.12825, -27.12825]),
        "Z": np.array([0.3, 1.1]),
    }
    table = fits.BinTableHDU.from_columns(
        [
            fits.Column(name=key, array=value, format="K" if key == "TARGETID" else "D")
            for key, value in catalogue.items()
        ]
    )
    table.header.update(TRACER="LRG", MOCKID=2, CAP="GCCOMB", ZSOURCE="RSDZ", ZACTIVE="Z")
    import json
    from tests.test_extract_ffa_reservoir import selection_auxiliaries
    from pzbao.data.ffa_reservoir_extractor import publisher_imaging_selection

    auxiliary = selection_auxiliaries(tmp_path / "aux", config, "LRG", 2, [10, 20])
    _, inputs = publisher_imaging_selection(
        {**catalogue, "PHOTSYS": np.array(["N", "S"])}, "LRG", 2, auxiliary, config
    )

    from pzbao.data.ffa_reservoir_extractor import publisher_good_locations

    _, hardware_inputs = publisher_good_locations("LRG", auxiliary, config)
    inputs += hardware_inputs

    table.header["SELDEF"] = json.dumps(catalog.reservoir_selection_definition("LRG", config))
    table.header["SELINPUT"] = json.dumps(inputs)
    table.writeto(record["path"], overwrite=True)
    record = next(
        item
        for item in catalog.register_parent_inputs(tmp_path, tracer="LRG", mock_id=2, config=config)
        if item["role"] == "reservoir" and item["cap"] == "NGC"
    )
    decoded = catalog.read_reservoir_provenance(record["path"])
    assert decoded["parent_artifact_sha256"] is None
    data = catalog.read_registered_catalogue(
        record, tracer="LRG", mock_id=2, cap="NGC", role="reservoir", config=config
    )
    np.testing.assert_array_equal(data["TARGETID"], [10])
    table.header["MOCKID"] = 3
    table.writeto(record["path"], overwrite=True)
    record = next(
        item
        for item in catalog.register_parent_inputs(tmp_path, tracer="LRG", mock_id=2, config=config)
        if item["role"] == "reservoir" and item["cap"] == "NGC"
    )
    with pytest.raises(ValueError, match="reservoir native identity"):
        catalog.read_registered_catalogue(
            record, tracer="LRG", mock_id=2, cap="NGC", role="reservoir", config=config
        )


@pytest.mark.parametrize("tracer", ["BGS_BRIGHT-21.5", "LRG"])
def test_registered_survey_columns_merge_with_extracted_parent_through_preparation(
    tmp_path, tracer
):
    from astropy.table import Table
    from pzbao.data import ffa_reservoir_extractor as extractor
    from pzbao.selection import prepare_data_for_selection

    config = load_analysis_config()
    records = _registered_inputs(tmp_path, config, tracer=tracer)
    data_record = next(r for r in records if r["role"] == "data" and r["cap"] == "NGC")
    reservoir_record = next(r for r in records if r["role"] == "reservoir")
    source = {
        "TARGETID": np.array([100, 200, 300, 400]),
        "RA": np.full(4, 192.85948),
        "DEC": np.full(4, 27.12825),
        "Z": np.array(
            [0.105, 0.2, 0.3, 0.385] if tracer == "BGS_BRIGHT-21.5" else [0.405, 0.7, 0.985, 1.05]
        ),
    }
    # These real clustering columns are absent from the extracted parent schema.
    metadata = {
        name: np.full(4, 7.0)
        for name in (
            "BRICKID",
            "COLLISION",
            "DESI_TARGET",
            "FIBER",
            "LOCATION",
            "MASKBITS",
            "NOBS_G",
            "NOBS_R",
            "NOBS_Z",
            "NTILE",
            "NX",
            "PRIORITY",
            "PRIORITY_INIT",
            "TILEID",
            "WEIGHT_FKP",
            "ZWARN",
            "WEIGHT",
            "WEIGHT_SYS",
            "WEIGHT_COMP",
        )
    }
    Table({**source, **metadata}).write(data_record["path"], overwrite=True)
    source_digest = catalog.file_sha256(data_record["path"])
    parent_path = tmp_path / "parent.fits"
    parent_ids, parent_z = (
        ([200, 100, 400, 300], [0.005, 0.105, 0.385, 0.695])
        if tracer == "BGS_BRIGHT-21.5"
        else ([10, 100, 300, 400, 500], [0.105, 0.405, 0.985, 1.05, 1.295])
    )
    from tests.test_extract_ffa_reservoir import potential_assignments

    size = len(parent_ids)
    Table(
        {
            **potential_assignments(size),
            "TARGETID": parent_ids,
            "RA": np.full(size, 192.85948),
            "DEC": np.full(size, 27.12825),
            "RSDZ": parent_z,
            "PHOTSYS": ["N"] * size,
            "MASKBITS": np.zeros(size, dtype=int),
            "R_MAG_ABS": [-22.0] * size,
        }
    ).write(parent_path)
    from tests.test_extract_ffa_reservoir import selection_auxiliaries

    auxiliary = selection_auxiliaries(tmp_path / "aux", config, tracer, 2, parent_ids)
    extracted, selection_inputs = extractor.extract_slice_arrays(
        parent_path, tracer, config, auxiliary_root=auxiliary, mock_id=2
    )
    # Replace only the synthetic placeholder; the producer itself remains create-only.
    from pathlib import Path

    Path(reservoir_record["path"]).unlink()
    extractor.write_slice(
        Path(reservoir_record["path"]),
        extracted,
        extractor.calibration_ratios(extracted, [source], tracer, config),
        tracer=tracer,
        mock_id=2,
        parent_path=parent_path,
        phase="calibration",
        selection_inputs=selection_inputs,
        config=config,
    )
    records = catalog.register_parent_inputs(tmp_path, tracer=tracer, mock_id=2, config=config)
    loaded = {}
    for role in ("data", "reservoir"):
        record = next(r for r in records if r["role"] == role and r["cap"] == "NGC")
        loaded[role] = catalog.read_registered_catalogue(
            record, tracer=tracer, mock_id=2, cap="NGC", role=role, config=config
        )
    expected = assign_source_identity(source, tracer=tracer, mock_id=2, cap="NGC", role="data")
    assert set(loaded["data"]) == set(expected)
    for name in expected:
        np.testing.assert_array_equal(loaded["data"][name], expected[name])
    selected, diagnostic = prepare_data_for_selection(
        loaded["data"],
        tracer=tracer,
        tier_label="native",
        cap="NGC",
        reservoir=loaded["reservoir"],
        reservoir_record={"tracer": tracer, "sample_identity": "test-parent"},
        config=config,
    )
    field = config["tracers"][tracer]["field"]
    inside = (source["Z"] >= field["zmin"]) & (source["Z"] < field["zmax"])
    for name in expected:
        np.testing.assert_array_equal(selected[name], expected[name][inside])
    np.testing.assert_array_equal(selected["Z_RSD"], source["Z"][inside])
    np.testing.assert_array_equal(selected["Z_OBS"], source["Z"][inside])
    np.testing.assert_array_equal(selected["WEIGHT"], np.ones(inside.sum()))
    assert diagnostic["injected_row_count"] == 2
    assert diagnostic["boundary_counts"]["reservoir_scattered_in_count"] == 0
    assert diagnostic["boundary_counts"]["overlap_removed_row_count"] == (
        1 if tracer == "LRG" else 0
    )
    assert catalog.file_sha256(data_record["path"]) == source_digest
    assert set(catalog.read_fits_catalog(data_record["path"])) == set(source) | set(metadata)


@pytest.mark.parametrize("column", ["Z_RSD", "Z_OBS", "IS_RESERVOIR", "TARGETID"])
def test_registered_data_projection_cannot_hide_invalid_scientific_input(tmp_path, column):
    from astropy.table import Table

    config = load_analysis_config()
    record = _registered_inputs(tmp_path, config)[0]
    source = _catalog()
    if column == "TARGETID":
        source.pop(column)
    else:
        source[column] = np.zeros(3)
    Table(source).write(record["path"], overwrite=True)
    record = catalog.register_parent_inputs(tmp_path, tracer="LRG", mock_id=2, config=config)[0]
    with pytest.raises((KeyError, ValueError), match="TARGETID|reserved"):
        catalog.read_registered_catalogue(
            record, tracer="LRG", mock_id=2, cap="NGC", role="data", config=config
        )


def test_registered_random_reader_retains_survey_metadata(tmp_path):
    from astropy.table import Table

    config = load_analysis_config()
    records = _registered_inputs(tmp_path, config)
    record = next(r for r in records if r["role"] == "random")
    source = {**_catalog(), "PHOTSYS": np.array(["N", "S", "N"]), "NX": np.array([1, 2, 3])}
    Table(source).write(record["path"], overwrite=True)
    records = catalog.register_parent_inputs(tmp_path, tracer="LRG", mock_id=2, config=config)
    record = next(r for r in records if r["role"] == "random")
    loaded = catalog.read_registered_catalogue(
        record, tracer="LRG", mock_id=2, cap="NGC", role="random", config=config
    )
    assert set(loaded) == set(source) | {"SOURCE_ID"}
    for name in source:
        np.testing.assert_array_equal(loaded[name], source[name])


def test_publisher_checksum_is_checked_once_per_physical_file(tmp_path, monkeypatch):
    from pathlib import Path

    config = load_analysis_config()
    records = _registered_inputs(tmp_path, config)
    reservoir = Path(next(record["path"] for record in records if record["role"] == "reservoir"))
    checksum = catalog.file_sha256(reservoir)
    reservoir.with_name(reservoir.name + ".sha256").write_text(checksum + "  " + reservoir.name)
    called = []
    original = catalog.file_sha256

    def measured(path):
        called.append(path)
        return original(path)

    monkeypatch.setattr(catalog, "file_sha256", measured)
    records = catalog.register_parent_inputs(tmp_path, tracer="LRG", mock_id=2, config=config)
    assert set(called) == {Path(record["path"]) for record in records}
    assert len(called) == len(set(called)) and called.count(reservoir) == 1
    assert [
        record["publisher_checksum_verified"] for record in records if record["role"] == "reservoir"
    ] == [checksum, checksum]
    reservoir.write_bytes(b"changed source")
    with pytest.raises(ValueError, match="publisher checksum mismatch"):
        catalog.register_parent_inputs(tmp_path, tracer="LRG", mock_id=2, config=config)


@pytest.mark.parametrize("tracer", ["LRG", "BGS_BRIGHT-21.5"])
def test_configured_publisher_manifest_verifies_both_caps_and_every_split(tmp_path, tracer):
    from pathlib import Path

    config = load_analysis_config()
    records = _registered_inputs(tmp_path, config, tracer=tracer)
    manifest = tmp_path / config["tracers"][tracer]["source_reference"][
        "checksum_manifest_template"
    ].format(mock_id=2)
    sources = [record for record in records if record["role"] != "reservoir"]
    manifest.write_text(
        "\n".join(
            record["sha256"] + ("  ./" if index % 2 else " *") + Path(record["path"]).name
            for index, record in enumerate(sources)
        )
    )
    registered = catalog.register_parent_inputs(tmp_path, tracer=tracer, mock_id=2, config=config)
    for record in registered:
        if record["role"] != "reservoir":
            assert record["publisher_checksum_verified"] == record["sha256"]
            assert record["publisher_checksum_manifest"] == str(manifest.resolve())
    # An unchanged-size corrupt source fails against the publisher, before any FITS read.
    path = Path(sources[-1]["path"])
    path.write_bytes(b"x" * path.stat().st_size)
    with pytest.raises(ValueError, match="publisher checksum mismatch"):
        catalog.register_parent_inputs(tmp_path, tracer=tracer, mock_id=2, config=config)


@pytest.mark.parametrize("content", ["", "not-a-checksum  source.fits", "duplicate"])
def test_supplied_publisher_manifest_cannot_be_incomplete_or_ambiguous(tmp_path, content):
    from pathlib import Path

    config = load_analysis_config()
    records = _registered_inputs(tmp_path, config)
    manifest = tmp_path / config["tracers"]["LRG"]["source_reference"][
        "checksum_manifest_template"
    ].format(mock_id=2)
    if content == "duplicate":
        record = records[0]
        content = (record["sha256"] + "  " + Path(record["path"]).name + "\n") * 2
    manifest.write_text(content)
    with pytest.raises(ValueError, match="publisher checksum"):
        catalog.register_parent_inputs(tmp_path, tracer="LRG", mock_id=2, config=config)


def test_registered_input_avoids_rehashing_unchanged_metadata(tmp_path, monkeypatch):
    config = load_analysis_config()
    record = _registered_inputs(tmp_path, config)[0]
    assert len(record["sha256"]) == 64 and "publisher_checksum_unavailable_reason" in record
    monkeypatch.setattr(catalog, "file_sha256", lambda path: pytest.fail("consumer rehashed input"))
    monkeypatch.setattr(catalog, "read_fits_catalog", lambda path: _catalog())
    catalog.read_registered_catalogue(
        record, tracer="LRG", mock_id=2, cap="NGC", role="data", config=config
    )


@pytest.mark.parametrize("changed_bytes", [False, True])
def test_registered_input_rechecks_bytes_after_ctime_only_change(
    tmp_path, monkeypatch, changed_bytes
):
    from copy import deepcopy
    from pathlib import Path

    config = load_analysis_config()
    record = _registered_inputs(tmp_path, config)[0]
    registered = deepcopy(record)
    path = Path(record["path"])
    if changed_bytes:
        path.write_bytes(b"x" * record["size_bytes"])
    # Model chmod/chgrp, or an equal-size byte edit followed by restoring mtime.
    metadata = {key: record[key] for key in ("size_bytes", "mtime_ns", "ctime_ns")}
    metadata["ctime_ns"] += 1
    monkeypatch.setattr(catalog, "_source_file_metadata", lambda path: metadata.copy())
    hashes = []
    original_hash = catalog.file_sha256

    def rehash(path):
        hashes.append(path)
        return original_hash(path)

    monkeypatch.setattr(catalog, "file_sha256", rehash)
    monkeypatch.setattr(catalog, "read_fits_catalog", lambda path: _catalog())
    if changed_bytes:
        with pytest.raises(ValueError, match="changed since registration"):
            catalog.read_registered_catalogue(
                record, tracer="LRG", mock_id=2, cap="NGC", role="data", config=config
            )
    else:
        result = catalog.read_registered_catalogue(
            record, tracer="LRG", mock_id=2, cap="NGC", role="data", config=config
        )
        np.testing.assert_array_equal(result["Z"], _catalog()["Z"])
    assert hashes == [path]
    assert record == registered


def test_registered_input_refuses_changes_during_fallback_hash(tmp_path, monkeypatch):
    config = load_analysis_config()
    record = _registered_inputs(tmp_path, config)[0]
    metadata = {key: record[key] for key in ("size_bytes", "mtime_ns", "ctime_ns")}
    metadata["ctime_ns"] += 1
    monkeypatch.setattr(catalog, "_source_file_metadata", lambda path: metadata.copy())

    def changing_file(path):
        metadata["ctime_ns"] += 1
        return record["sha256"]

    monkeypatch.setattr(catalog, "file_sha256", changing_file)
    with pytest.raises(ValueError, match="changed during byte verification"):
        catalog.read_registered_catalogue(
            record, tracer="LRG", mock_id=2, cap="NGC", role="data", config=config
        )


def test_optional_source_comparison_preserves_duplicates_missing_and_redshift_differences():
    from pzbao.catalog import compare_source_redshifts

    config = load_analysis_config()
    tolerance = config["analysis"]["source_redshift_comparison"]["absolute_tolerance"]
    source = {
        "TARGETID": np.array([3, 1, 2, 1, 5, 5, 8]),
        "RSDZ": np.array([0.3, 0.1, 0.2, 0.1, 0.5, 0.6, np.nan]),
    }
    caps = {
        "NGC": {
            "TARGETID": np.array([1, 2, 3, 4, 5, 8]),
            "Z": np.array([0.1, 0.2 + tolerance / 2, 0.3 + 2 * tolerance, 0.4, 0.5, 0.8]),
        },
        "SGC": {"TARGETID": np.array([1, 1]), "Z": np.array([0.1, 0.1])},
    }
    before = source["RSDZ"].copy()
    result = compare_source_redshifts(source, caps, config=config)
    assert result["duplicate_group_count"] == 2
    assert result["duplicate_row_count"] == 4
    assert result["conflicting_duplicate_count"] == 1
    assert result["nonfinite_source_group_count"] == 1
    north = result["caps"]["NGC"]
    assert north["joined_row_count"] == 5
    assert north["coverage_fraction"] == pytest.approx(5 / 6)
    assert north["missing_identity_count"] == north["mismatching_identity_count"] == 1
    assert north["nonfinite_comparison_count"] == 2
    assert north["maximum_absolute_difference"] == pytest.approx(2 * tolerance, rel=1e-4)
    assert result["caps"]["SGC"]["duplicated_data_identity_count"] == 1
    np.testing.assert_array_equal(source["RSDZ"], before)
    assert "passes" not in result and "sha256" not in str(result)
    with pytest.raises(ValueError, match="both caps"):
        compare_source_redshifts(source, {"NGC": caps["NGC"]}, config=config)
    malformed = {**source, "TARGETID": source["TARGETID"].astype(float)}
    with pytest.raises(ValueError, match="integer"):
        compare_source_redshifts(malformed, caps, config=config)


def test_optional_footprint_count_mean_uses_each_cap_and_all_mock_splits():
    from pzbao.catalog import footprint_from_random_counts

    config = load_analysis_config()
    nmock = len(config["analysis"]["mock_ids"])
    splits = config["tracers"]["LRG"]["catalogue"]["random_splits"]
    counts = {
        "NGC": np.tile(2500 * np.arange(1, splits + 1), (nmock, 1)),
        "SGC": np.full((nmock, splits), 5000),
    }
    result = footprint_from_random_counts(counts, tracer="LRG", config=config)
    np.testing.assert_allclose(result["NGC"]["area_deg2_by_mock"], (splits + 1) / 2)
    np.testing.assert_allclose(result["SGC"]["area_deg2_by_mock"], 2)
    assert result["NGC"]["split_sum_identical_across_mocks"]
    counts["SGC"][2, 0] += 2500 * splits
    result = footprint_from_random_counts(counts, tracer="LRG", config=config)
    assert result["SGC"]["area_deg2_by_mock"][2] == 3
    assert not result["SGC"]["split_sum_identical_across_mocks"]
    with pytest.raises(ValueError, match="every integer"):
        footprint_from_random_counts(
            {cap: array[:-1] for cap, array in counts.items()}, tracer="LRG", config=config
        )
