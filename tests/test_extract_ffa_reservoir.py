"""Safety and YAML-ownership tests for reservoir extraction."""

import pzbao.diagnostics.selection as diag_selection
import json
import numpy as np
import pytest
from pzbao.config import load_analysis_config
from pzbao.data import ffa_reservoir_extractor as extractor


def potential_assignments(size):
    """Native potential-assignment fields with one good assignment per galaxy."""
    return {
        "COLLISION": np.zeros(size, dtype=bool),
        "TILEID": np.ones(size, dtype="i8"),
        "LOCATION": np.zeros(size, dtype="i8"),
        "DESI_TARGET": np.ones(size, dtype="i8"),
        "BGS_TARGET": np.full(size, 2, dtype="i8"),
    }


def selection_auxiliaries(root, config, tracer, mock_id, target_ids):
    """Small native maps and publisher manifest; explicit synthetic selection inputs."""
    from astropy.table import Table
    from pzbao.catalog import file_sha256

    root.mkdir(parents=True, exist_ok=True)
    imaging = config["selection"]["reservoir_imaging"]
    imaging["nside"] = 1
    spec = config["tracers"][tracer]["reservoir"]
    for cap in ("N", "S"):
        name = imaging["filename_template"].format(
            tracer=spec["imaging_map_tracer"], nside=1, cap=cap
        )
        table = {
            key: np.full(12, 1000000.0 if cut["operator"] == "dust_corrected_gt" else 0.01)
            for key, cut in imaging["cuts"].items()
        }
        Table(table).write(root / name)
        imaging["sha256"][spec["imaging_map_tracer"]][cap] = file_sha256(root / name)
    if "target_mask" in spec:
        mask = spec["target_mask"]
        name = mask["filename_template"].format(mock_id=mock_id)
        Table({"TARGETID": target_ids, "lrg_mask": np.zeros(len(target_ids), dtype="i1")}).write(
            root / name
        )
        (root / mask["checksum_manifest"]).write_text(f"{file_sha256(root / name)}  {name}\n")
    hardware = config["selection"]["reservoir_hardware"]
    for key in (spec["hardware_program"], "bad_fibers"):
        reference = hardware["references"][key]
        path = root / reference["filename"]
        if key == "bad_fibers":
            path.write_text("99\n")
        else:
            Table(
                {"TILEID": [1], "LOCATION": [0], "FIBER": [0], "ZWARN": [0], "ZWARN_MTL": [0]}
            ).write(path)
        reference["sha256"] = file_sha256(path)
    return root


def test_extraction_windows_follow_each_tracer_edges():
    config = load_analysis_config()
    assert extractor.extraction_windows("LRG", config) == (
        (0.1, 0.4),
        (0.4, 0.42),
        (1.0, 1.3),
        (0.98, 1.0),
    )
    assert extractor.extraction_windows("BGS_BRIGHT-21.5", config) == (
        (0.0, 0.1),
        (0.1, 0.12),
        (0.4, 0.7),
        (0.38, 0.4),
    )


@pytest.mark.parametrize("native_dtype,original_count", [("f4", 5), ("f8", 4)])
def test_extended_native_slice_preserves_old_values_through_real_reader(
    tmp_path, native_dtype, original_count
):
    from astropy.table import Table
    from pzbao.catalog import (
        compare_extended_reservoir,
        file_sha256,
        read_fits_catalog,
        read_registered_catalogue,
        read_reservoir_provenance,
        validate_reservoir_extension,
    )
    from pzbao.selection import attach_stable_randomness
    from tests.test_boundary_support import reference_run

    config = load_analysis_config()
    ids = np.arange(1, 10)
    aux = selection_auxiliaries(tmp_path / "aux", config, "LRG", 0, ids)
    source = {
        **potential_assignments(len(ids)),
        "TARGETID": ids,
        "RA": np.full(len(ids), 222.0),
        "DEC": np.full(len(ids), 50.0),
        "RSDZ": np.array(
            [
                0.2,
                0.41,
                0.99,
                1.29,
                1.3,
                1.35,
                1.36,
                np.nextafter(np.array(1.36, dtype=native_dtype), -np.inf, dtype=native_dtype),
                np.nextafter(np.array(1.36, dtype=native_dtype), np.inf, dtype=native_dtype),
            ],
            dtype=native_dtype,
        ),
        "MASKBITS": np.zeros(len(ids), dtype=int),
    }
    parent = tmp_path / "parent.fits"
    Table(source).write(parent)
    old, inputs = extractor.extract_slice_arrays(
        parent, "LRG", config, auxiliary_root=aux, mock_id=0
    )
    old_path = tmp_path / "original.fits"
    extractor.write_slice(
        old_path,
        old,
        [],
        tracer="LRG",
        mock_id=0,
        parent_path=parent,
        phase="calibration",
        selection_inputs=inputs,
        config=config,
    )
    reference = reference_run(config)
    design = config["selection"]["boundary_support"]["experiment_design"]
    variant = diag_selection.variant_config(
        reference,
        design["targets"][1],
        mode="extend",
        depth=0.36,
        previous_depth=0.3,
        design=design,
    )
    assert extractor.extraction_windows("LRG", variant) == (
        (0.1, 0.4),
        (0.4, 0.42),
        (1.0, 1.36),
        (0.98, 1.0),
    )
    new, inputs = extractor.extract_slice_arrays(
        parent, "LRG", variant, auxiliary_root=aux, mock_id=0
    )
    np.testing.assert_array_equal(old["TARGETID"], ids[:original_count])
    np.testing.assert_array_equal(new["TARGETID"], [1, 2, 3, 4, 5, 6, 8])
    path = tmp_path / "extended.fits"
    with pytest.raises(ValueError, match="direct original slice"):
        extractor.write_slice(
            path,
            new,
            [],
            tracer="LRG",
            mock_id=0,
            parent_path=parent,
            phase="calibration",
            selection_inputs=inputs,
            config=variant,
        )
    extractor.write_slice(
        path,
        new,
        [],
        tracer="LRG",
        mock_id=0,
        parent_path=parent,
        phase="calibration",
        selection_inputs=inputs,
        config=variant,
        reference_slice=old_path,
    )
    embedded = read_reservoir_provenance(path)
    counts = validate_reservoir_extension(read_fits_catalog(path), embedded, variant)
    assert counts == {"preserved_row_count": original_count, "added_row_count": 7 - original_count}
    catalogues = []
    for member, settings in ((old_path, config), (path, variant)):
        stat = member.stat()
        record = {
            "path": str(member),
            "sha256": file_sha256(member),
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "ctime_ns": stat.st_ctime_ns,
            "tracer": "LRG",
            "mock_id": 0,
            "cap": "NGC",
            "role": "reservoir",
            "random_split": None,
        }
        native = read_registered_catalogue(
            record, tracer="LRG", mock_id=0, cap="NGC", role="reservoir", config=settings
        )
        catalogues.append(attach_stable_randomness(native, role="data", config=settings))
    for column in ("SOURCE_ID", "PHOTOZ_DEVIATE", "SELECTION_RANK", "RA", "DEC", "Z"):
        np.testing.assert_array_equal(catalogues[0][column], catalogues[1][column][:original_count])
    np.testing.assert_array_equal(
        catalogues[1]["IS_BOUNDARY_EXTENSION"], np.arange(7) >= original_count
    )
    changed = {key: value.copy() for key, value in new.items()}
    changed["RA"][0] += 0.1
    with pytest.raises(ValueError, match="native values"):
        compare_extended_reservoir(old, changed, boundary=1.0, old_depth=0.3, new_depth=0.36)
    changed = {key: value.copy() for key, value in new.items()}
    changed["Z"][-1] = 0.3
    with pytest.raises(ValueError, match="new upper-edge band"):
        compare_extended_reservoir(old, changed, boundary=1.0, old_depth=0.3, new_depth=0.36)
    embedded["extension_reference"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="identity changed"):
        validate_reservoir_extension(new, embedded, variant)


@pytest.mark.parametrize(
    "native_dtype,redshift,accepted",
    [("f4", 1.3, False), ("f8", 1.3, True), ("f4", 1.36, False), ("f8", 1.36, False)],
)
def test_extension_validation_uses_decimal_edges_before_and_after_serialization(
    tmp_path, native_dtype, redshift, accepted
):
    from astropy.table import Table
    from pzbao.catalog import compare_extended_reservoir, read_fits_catalog

    original = {
        "TARGETID": np.array([1]),
        "RA": np.array([222.0]),
        "DEC": np.array([50.0]),
        "Z": np.array([1.29], dtype=native_dtype),
    }
    extended = {
        **{name: np.repeat(values, 2) for name, values in original.items()},
        "TARGETID": np.array([1, 2]),
        "Z": np.array([1.29, redshift], dtype=native_dtype),
    }
    path = tmp_path / "extended.fits"
    Table({**extended, "Z": extended["Z"].astype("f8")}).write(path)
    for candidate in (extended, read_fits_catalog(path)):
        kwargs = {
            "boundary": 1.0,
            "old_depth": 0.3,
            "new_depth": 0.36,
            "native_redshift_dtype": native_dtype,
        }
        if accepted:
            assert compare_extended_reservoir(original, candidate, **kwargs) == {
                "preserved_row_count": 1,
                "added_row_count": 1,
            }
        else:
            with pytest.raises(ValueError, match="new upper-edge band"):
                compare_extended_reservoir(original, candidate, **kwargs)


def test_plan_is_explicit_and_non_destructive(tmp_path):
    parent, output = (tmp_path / "parent.fits", tmp_path / "slice.fits")
    plan = extractor.extraction_plan(2, "LRG", parent, output, load_analysis_config())
    assert plan["parent_path"] == str(parent.resolve())
    assert plan["output_path"] == str(output.resolve())
    assert "delete_parent_after_success" not in plan
    assert not parent.exists() and (not output.exists())


def test_extractor_cli_plans_without_retired_execution_options(tmp_path, capsys):
    from pzbao.config import DEFAULT_ANALYSIS_CONFIG_PATH

    args = [
        "--mock-id",
        "2",
        "--tracer",
        "LRG",
        "--parent",
        str(tmp_path / "parent.fits"),
        "--output",
        str(tmp_path / "slice.fits"),
        "--analysis-config",
        str(DEFAULT_ANALYSIS_CONFIG_PATH),
    ]
    assert extractor.main(args) == 0
    assert json.loads(capsys.readouterr().out)["mock_id"] == 2
    assert not list(tmp_path.iterdir())
    with pytest.raises(SystemExit) as caught:
        extractor.main(["--help"])
    assert caught.value.code == 0
    help_text = capsys.readouterr().out
    for retired in ("--phase", "--execute", "--holdout-release-artifact-sha256", "artifact-chain"):
        assert retired not in help_text
    with pytest.raises(SystemExit) as caught:
        extractor.main(args + ["--execute"])
    assert caught.value.code == 2 and "unrecognized arguments" in capsys.readouterr().err
    assert not list(tmp_path.iterdir())


def test_parent_cuts_come_from_yaml(tmp_path):
    from astropy.table import Table

    source = {
        "RA": np.arange(4.0),
        "DEC": np.arange(4.0),
        "RSDZ": np.array([0.09, 0.11, 0.39, 0.41]),
        "PHOTSYS": np.array(["N", "N", "S", "S"]),
        "MASKBITS": np.array([0, 1 << 1, 0, 0]),
        "TARGETID": np.arange(4),
        "R_MAG_ABS": np.array([-22.0, -22.0, -20.0, -22.0]),
    }
    config = load_analysis_config()
    parent = tmp_path / "parent.fits"
    Table({**source, **potential_assignments(len(source["TARGETID"]))}).write(parent)
    auxiliary = selection_auxiliaries(
        tmp_path / "aux", config, "BGS_BRIGHT-21.5", 0, source["TARGETID"]
    )
    result, _ = extractor.extract_slice_arrays(
        parent, "BGS_BRIGHT-21.5", config, auxiliary_root=auxiliary, mock_id=0
    )
    np.testing.assert_allclose(result["Z"], [0.09, 0.41])
    np.testing.assert_array_equal(result["TARGETID"], [0, 3])


@pytest.mark.parametrize("download", [False, True])
def test_explicit_extraction_cli_writes_current_identity_and_redshift_values(
    tmp_path, capsys, monkeypatch, download
):
    from astropy.table import Table
    from pzbao.catalog import read_fits_catalog, read_reservoir_provenance
    import yaml

    parent, output = (tmp_path / "parent.fits", tmp_path / "slice.fits")
    Table(
        {
            **potential_assignments(5),
            "TARGETID": np.arange(5),
            "RA": np.arange(5.0),
            "DEC": np.arange(5.0),
            "RSDZ": [0.09, 0.11, 0.39, 0.41, 0.5],
            "PHOTSYS": ["N"] * 5,
            "MASKBITS": [0, 0, 0, 0, 1 << 1],
            "R_MAG_ABS": [-22.0] * 5,
        }
    ).write(parent)
    source_bytes = parent.read_bytes()
    (tmp_path / "mock13").mkdir()
    for cap in ("NGC", "SGC"):
        inside = {"RA": [1.0, 2.0], "DEC": [1.0, 2.0], "Z": [0.11, 0.39]}
        if cap == "SGC":
            inside = {key: np.array([], dtype=float) for key in inside}
        Table(inside).write(
            tmp_path / "mock13" / f"BGS_BRIGHT-21.5_complete_{cap}_clustering.dat.fits"
        )
    config = load_analysis_config()
    auxiliary = selection_auxiliaries(tmp_path / "aux", config, "BGS_BRIGHT-21.5", 13, np.arange(5))
    config_path = tmp_path / "analysis.yaml"
    config_path.write_text(yaml.safe_dump(config))
    args = [
        "--mock-id",
        "13",
        "--tracer",
        "BGS_BRIGHT-21.5",
        "--parent",
        str(parent),
        "--output",
        str(output),
        "--analysis-config",
        str(config_path),
        "--auxiliary-root",
        str(auxiliary),
        "--extract",
        "--catalog-root",
        str(tmp_path),
    ]
    if download:
        import io
        import urllib.request
        import hashlib

        parent.unlink()
        reference = config["tracers"]["BGS_BRIGHT-21.5"]["source_reference"]
        manifest = tmp_path / reference["checksum_manifest_template"].format(mock_id=13)
        manifest.write_text(
            f"{hashlib.sha256(source_bytes).hexdigest()}  {reference['source_basename']}\n"
        )
        monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: io.BytesIO(source_bytes))
        args.append("--download-parent")
    assert extractor.main(args) == 0
    assert json.loads(capsys.readouterr().out)["written_rows"] == 4
    saved = read_fits_catalog(output)
    np.testing.assert_array_equal(saved["TARGETID"], np.arange(4))
    np.testing.assert_array_equal(saved["Z"], [0.09, 0.11, 0.39, 0.41])
    provenance = read_reservoir_provenance(output)
    assert provenance["mock_id"] == 13
    assert provenance["source_redshift_column"] == "RSDZ"
    if download:
        assert not parent.exists()
    else:
        assert parent.read_bytes() == source_bytes
    with pytest.raises(FileExistsError, match="overwrite"):
        extractor.main(args)


def test_calibration_ratio_is_measured_not_invented():
    slice_catalog = {"Z": np.array([0.105, 0.385, 0.39, 0.395])}
    clustering = [{"Z": np.array([0.105, 0.385, 0.39])}]
    records = extractor.calibration_ratios(
        slice_catalog, clustering, "BGS_BRIGHT-21.5", load_analysis_config()
    )
    upper = next((record for record in records if record["edge"] == 0.4))
    assert upper["parent_to_clustering_ratio"] == 1.5


def test_written_slice_declares_combined_cap_provenance(tmp_path):
    fits = pytest.importorskip("astropy.io.fits")
    parent = tmp_path / "parent.fits"
    parent.write_bytes(b"parent")
    output = tmp_path / "slice.fits"
    config = load_analysis_config()
    auxiliary = selection_auxiliaries(tmp_path / "aux", config, "LRG", 2, [1])
    _, inputs = extractor.publisher_imaging_selection(
        {
            "TARGETID": np.array([1]),
            "RA": np.array([0.0]),
            "DEC": np.array([0.0]),
            "PHOTSYS": np.array(["S"]),
        },
        "LRG",
        2,
        auxiliary,
        config,
    )
    _, hardware_inputs = extractor.publisher_good_locations("LRG", auxiliary, config)
    inputs += hardware_inputs
    extractor.write_slice(
        output,
        {"TARGETID": np.array([1]), "Z": np.array([0.09])},
        [],
        tracer="LRG",
        mock_id=2,
        parent_path=parent,
        phase="calibration",
        selection_inputs=inputs,
        config=config,
    )
    with fits.open(output) as hdul:
        assert hdul[1].header["CAP"] == "GCCOMB"
        assert hdul[1].header["ZSOURCE"] == "RSDZ"
        assert hdul[1].header["ZACTIVE"] == "Z"
        assert hdul[1].header["PHASE"] == "calibration"


def test_the_holdout_phase_publishes_no_reservoir_slice(tmp_path):
    """The retired phase is refused outright rather than gated on a release that cannot exist.

    Mock 13 is an ordinary calibration realization now, so a slice for it publishes through the
    calibration phase with no release at all; declaring the holdout phase names a stage that no
    longer acquires anything.
    """
    parent = tmp_path / "parent.fits"
    parent.write_bytes(b"parent")
    with pytest.raises(ValueError, match="holdout phase is retired"):
        extractor.write_slice(
            tmp_path / "holdout.fits",
            {"TARGETID": np.array([1]), "Z": np.array([0.5])},
            [],
            tracer="LRG",
            mock_id=13,
            parent_path=parent,
            phase="holdout",
        )
    with pytest.raises(ValueError, match="with no release"):
        extractor.write_slice(
            tmp_path / "calibration.fits",
            {"TARGETID": np.array([1]), "Z": np.array([0.5])},
            [],
            tracer="LRG",
            mock_id=13,
            parent_path=parent,
            phase="calibration",
            holdout_release_artifact_sha256="e" * 64,
        )


def test_publisher_cuts_remove_star_mask_and_magnitude_boundary(tmp_path):
    from astropy.table import Table

    config = load_analysis_config()
    tracer = "BGS_BRIGHT-21.5"
    source = {
        "TARGETID": np.arange(5),
        "RA": np.arange(5.0),
        "DEC": np.arange(5.0),
        "RSDZ": np.full(5, 0.41),
        "PHOTSYS": ["N"] * 5,
        "MASKBITS": [0, 1 << 11, 0, 0, 0],
        "R_MAG_ABS": [-22.0, -22.0, -21.54, -21.55, -21.56],
    }
    parent = tmp_path / "parent.fits"
    Table({**source, **potential_assignments(len(source["TARGETID"]))}).write(parent)
    auxiliary = selection_auxiliaries(tmp_path / "aux", config, tracer, 0, np.arange(5))
    selected, inputs = extractor.extract_slice_arrays(
        parent, tracer, config, auxiliary_root=auxiliary, mock_id=0
    )
    np.testing.assert_array_equal(selected["TARGETID"], [0, 4])
    assert len(inputs) == 4
    assert {row["sha256"] for row in inputs[2:]} == set(
        config["selection"]["reservoir_imaging"]["sha256"]["BGS_BRIGHT"].values()
    )


@pytest.mark.parametrize("invalid", ["missing", "duplicate", "checksum"])
def test_target_mask_rejects_missing_ambiguous_or_changed_inputs(tmp_path, invalid):
    from astropy.table import Table
    from pzbao.catalog import file_sha256

    config = load_analysis_config()
    auxiliary = selection_auxiliaries(tmp_path / "aux", config, "LRG", 2, [10, 20])
    spec = config["tracers"]["LRG"]["reservoir"]["target_mask"]
    name = spec["filename_template"].format(mock_id=2)
    ids = [10] if invalid == "missing" else [10, 10, 20]
    Table({"TARGETID": ids, "lrg_mask": np.zeros(len(ids), dtype=int)}).write(
        auxiliary / name, overwrite=True
    )
    if invalid != "checksum":
        (auxiliary / spec["checksum_manifest"]).write_text(
            f"{file_sha256(auxiliary / name)}  {name}\n"
        )
    source = {
        "TARGETID": np.array([10, 20]),
        "RA": np.array([1.0, 2.0]),
        "DEC": np.array([1.0, 2.0]),
        "PHOTSYS": np.array(["N", "S"]),
    }
    with pytest.raises(ValueError, match="checksum mismatch|exactly once"):
        extractor.publisher_imaging_selection(source, "LRG", 2, auxiliary, config)


def test_target_mask_join_preserves_order_and_map_cuts_use_both_caps(tmp_path):
    from astropy.table import Table
    from pzbao.catalog import file_sha256

    config = load_analysis_config()
    auxiliary = selection_auxiliaries(tmp_path / "aux", config, "LRG", 2, [20, 10, 30])
    spec = config["tracers"]["LRG"]["reservoir"]["target_mask"]
    name = spec["filename_template"].format(mock_id=2)
    Table({"TARGETID": [20, 10, 30], "lrg_mask": [1, 0, 0]}).write(auxiliary / name, overwrite=True)
    (auxiliary / spec["checksum_manifest"]).write_text(f"{file_sha256(auxiliary / name)}  {name}\n")
    imaging = config["selection"]["reservoir_imaging"]
    name = imaging["filename_template"].format(tracer="QSO", nside=1, cap="S")
    table = Table.read(auxiliary / name)
    table["GALDEPTH_Z"] = np.full(12, 31.0)
    table["EBV"] = np.full(12, 0.1)
    table.write(auxiliary / name, overwrite=True)
    imaging["sha256"]["QSO"]["S"] = file_sha256(auxiliary / name)
    name = imaging["filename_template"].format(tracer="QSO", nside=1, cap="N")
    table = Table.read(auxiliary / name)
    table["PSFSIZE_R"] = np.full(12, 2.3, dtype="f4")
    table.write(auxiliary / name, overwrite=True)
    imaging["sha256"]["QSO"]["N"] = file_sha256(auxiliary / name)
    source = {
        "TARGETID": np.array([10, 20, 30]),
        "RA": np.array([1.0, 2.0, 3.0]),
        "DEC": np.array([1.0, 2.0, 3.0]),
        "PHOTSYS": np.array(["N", "N", "S"]),
    }
    selected, _ = extractor.publisher_imaging_selection(source, "LRG", 2, auxiliary, config)
    np.testing.assert_array_equal(selected, [True, False, False])


def test_interior_comparison_rejects_equal_counts_with_different_galaxies():
    source = {"RA": np.array([1.0, 2.0]), "DEC": np.array([1.0, 2.0]), "Z": np.array([0.105, 0.39])}
    complete = {**source, "RA": np.array([9.0, 2.0])}
    with pytest.raises(ValueError, match="1 extra, 1 missing"):
        extractor.verify_interior_membership(
            source, [complete], "BGS_BRIGHT-21.5", load_analysis_config()
        )


@pytest.mark.parametrize(
    "changed", ["definition", "map_digest", "mask_realization", "missing_input"]
)
def test_consumer_refuses_changed_publisher_selection_provenance(tmp_path, changed):
    from pzbao.catalog import reservoir_selection_definition, validate_reservoir_selection

    config = load_analysis_config()
    auxiliary = selection_auxiliaries(tmp_path / "aux", config, "LRG", 2, [10])
    _, inputs = extractor.publisher_imaging_selection(
        {
            "TARGETID": np.array([10]),
            "RA": np.array([1.0]),
            "DEC": np.array([1.0]),
            "PHOTSYS": np.array(["N"]),
        },
        "LRG",
        2,
        auxiliary,
        config,
    )
    _, hardware_inputs = extractor.publisher_good_locations("LRG", auxiliary, config)
    inputs += hardware_inputs
    embedded = {
        "selection": reservoir_selection_definition("LRG", config),
        "selection_inputs": inputs,
    }
    if changed == "definition":
        embedded["selection"]["imaging"]["cuts"]["EBV"]["value"] += 0.01
    elif changed == "map_digest":
        inputs[-1]["sha256"] = "e" * 64
    elif changed == "mask_realization":
        inputs[0]["name"] = inputs[0]["name"].replace("forFA2", "forFA3")
    else:
        inputs.pop()
    with pytest.raises(ValueError, match="selection"):
        validate_reservoir_selection(embedded, "LRG", 2, config)


def test_absent_parent_material_is_reported_but_wrong_selection_is_refused(tmp_path):
    from astropy.table import Table

    source = {"RA": np.array([1.0, 2.0]), "DEC": np.array([1.0, 2.0]), "Z": np.array([0.105, 0.39])}
    complete = {
        key: np.append(values, {"RA": 3.0, "DEC": 3.0, "Z": 0.395}[key])
        for key, values in source.items()
    }
    raw = tmp_path / "parent.fits"
    Table({"RSDZ" if key == "Z" else key: value for key, value in source.items()}).write(raw)
    records = extractor.verify_interior_membership(
        source, [complete], "BGS_BRIGHT-21.5", load_analysis_config(), parent_path=raw
    )
    assert [row["source_unavailable_count"] for row in records] == [0, 1]
    Table({"RSDZ" if key == "Z" else key: value for key, value in complete.items()}).write(
        raw, overwrite=True
    )
    with pytest.raises(ValueError, match="0 extra, 1 missing"):
        extractor.verify_interior_membership(
            source, [complete], "BGS_BRIGHT-21.5", load_analysis_config(), parent_path=raw
        )


def test_interior_membership_uses_the_same_decimal_edges_as_extraction():
    native = {
        "RA": np.array([1.0, 2.0]),
        "DEC": np.array([1.0, 2.0]),
        "Z": np.array([0.38, 0.39], dtype="f4"),
    }
    extracted = {name: values[1:] for name, values in native.items()}
    records = extractor.verify_interior_membership(
        extracted, [native], "BGS_BRIGHT-21.5", load_analysis_config()
    )
    assert [row["source_unavailable_count"] for row in records] == [0, 0]


@pytest.mark.parametrize("tracer", ["LRG", "BGS_BRIGHT-21.5"])
def test_publisher_lower_endpoint_is_excluded_from_comparison_after_serialization(tmp_path, tracer):
    """Mock 8's float32 endpoint is absent under DESI's strict native Z > zmin cut."""
    from astropy.table import Table
    from pzbao.catalog import read_fits_catalog

    config = load_analysis_config()
    field = config["tracers"][tracer]["field"]
    endpoint = np.float32(field["zmin"])
    adjacent = np.nextafter(endpoint, np.float32(np.inf))
    redshift = np.array([endpoint, adjacent, field["zmax"] - 0.01], dtype="f4")
    source = {"RA": np.arange(3.0), "DEC": np.arange(3.0), "Z": redshift}
    keep = redshift > field["zmin"]
    np.testing.assert_array_equal(keep, [False, True, True])
    complete = {name: values[keep] for name, values in source.items()}
    path = tmp_path / "slice.fits"
    Table({name: values.astype("f8") for name, values in source.items()}).write(path)
    for candidate in (source, read_fits_catalog(path)):
        records = extractor.verify_interior_membership(candidate, [complete], tracer, config)
        assert [row["source_unavailable_count"] for row in records] == [0, 0]
        counts = extractor.calibration_ratios(candidate, [complete], tracer, config)
        assert [row["parent_count"] for row in counts] == [1, 1]
        assert [row["clustering_count"] for row in counts] == [1, 1]
        assert [row["parent_to_clustering_ratio"] for row in counts] == [1.0, 1.0]
        missing_neighbor = {name: values[1:] for name, values in complete.items()}
        with pytest.raises(ValueError, match="1 extra, 0 missing"):
            extractor.verify_interior_membership(candidate, [missing_neighbor], tracer, config)


@pytest.mark.parametrize("tracer,redshift", [("BGS_BRIGHT-21.5", 0.41), ("LRG", 1.01)])
def test_native_potential_assignments_preserve_same_tracer_and_any_good_hardware(
    tmp_path, tracer, redshift
):
    from astropy.table import Table
    from pzbao.catalog import file_sha256, read_reservoir_provenance, validate_reservoir_selection

    config = load_analysis_config()
    ids = np.array([10, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 10000000000001])
    size = len(ids)
    aux = selection_auxiliaries(tmp_path / "aux", config, tracer, 0, np.unique(ids))
    hardware = config["selection"]["reservoir_hardware"]
    spec = config["tracers"][tracer]["reservoir"]
    reference = hardware["references"][spec["hardware_program"]]
    table = Table(
        {
            "TILEID": np.ones(size, dtype=int),
            "LOCATION": np.arange(size),
            "FIBER": np.zeros(size, dtype=int),
            "ZWARN": np.zeros(size),
            "ZWARN_MTL": np.zeros(size, dtype=int),
        }
    )
    table["ZWARN"][[0, 2]] = 999999
    table["ZWARN"][3] = np.nan
    table["ZWARN_MTL"][[4, 5, 6]] = [1 << 9, 1 << 18, 1 << 19]
    table["FIBER"][7] = 99
    table.write(aux / reference["filename"], overwrite=True)
    reference["sha256"] = file_sha256(aux / reference["filename"])
    source = {
        **potential_assignments(size),
        "TARGETID": ids,
        "RA": np.full(size, 222.0),
        "DEC": np.full(size, 50.0),
        "RSDZ": np.full(size, redshift, dtype="f4"),
        "MASKBITS": np.zeros(size, dtype=int),
        "R_MAG_ABS": np.full(size, -22.0),
    }
    source["LOCATION"] = np.arange(size)
    source["COLLISION"][8] = True
    source[spec["target_flag"]["column"]][9] = 1 if tracer != "LRG" else 2
    parent = tmp_path / "parent.fits"
    Table(source).write(parent)
    catalog, inputs = extractor.extract_slice_arrays(
        parent, tracer, config, auxiliary_root=aux, mock_id=0
    )
    np.testing.assert_array_equal(catalog["TARGETID"], [10, 100, 110])
    np.testing.assert_array_equal(catalog["Z"], np.full(3, redshift, dtype="f4"))
    np.testing.assert_array_equal(catalog["PHOTSYS"], ["N"] * 3)
    assert set(catalog) == {"TARGETID", "RA", "DEC", "Z", "PHOTSYS"} | (
        {"R_MAG_ABS"} if tracer != "LRG" else set()
    )
    output = tmp_path / "slice.fits"
    extractor.write_slice(
        output,
        catalog,
        [],
        tracer=tracer,
        mock_id=0,
        parent_path=parent,
        phase="calibration",
        selection_inputs=inputs,
        config=config,
    )
    embedded = read_reservoir_provenance(output)
    validate_reservoir_selection(embedded, tracer, 0, config)
    embedded["selection_inputs"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="selection input identity"):
        validate_reservoir_selection(embedded, tracer, 0, config)
    (aux / reference["filename"]).write_bytes(b"changed hardware")
    with pytest.raises(ValueError, match="hardware checksum"):
        extractor.extract_slice_arrays(parent, tracer, config, auxiliary_root=aux, mock_id=0)


def test_native_parent_rejects_conflicting_assignments(tmp_path):
    from astropy.table import Table

    config = load_analysis_config()
    aux = selection_auxiliaries(tmp_path / "aux", config, "LRG", 0, [10])
    source = {
        **potential_assignments(2),
        "TARGETID": [10, 10],
        "RA": [222.0, 223.0],
        "DEC": [50.0, 50.0],
        "RSDZ": [1.01, 1.01],
        "MASKBITS": [0, 0],
    }
    parent = tmp_path / "parent.fits"
    Table(source).write(parent)
    with pytest.raises(ValueError, match="conflicting native values.*RA"):
        extractor.extract_slice_arrays(parent, "LRG", config, auxiliary_root=aux, mock_id=0)


def test_publisher_photsys_requires_both_northern_conditions():
    result = extractor.publisher_photsys(
        np.array([222.0, 0.0, 222.0]), np.array([50.0, 50.0, 32.375]), load_analysis_config()
    )
    np.testing.assert_array_equal(result, ["N", "S", "S"])


@pytest.mark.parametrize("bad_checksum", [False, True])
def test_parent_download_is_create_only_and_verifies_the_publisher(
    tmp_path, monkeypatch, bad_checksum
):
    import io
    import hashlib
    import urllib.request

    config = load_analysis_config()
    reference = config["tracers"]["LRG"]["source_reference"]
    manifest = tmp_path / reference["checksum_manifest_template"].format(mock_id=0)
    manifest.parent.mkdir(parents=True)
    content = b"native parent fixture"
    expected = "0" * 64 if bad_checksum else hashlib.sha256(content).hexdigest()
    manifest.write_text(f"{expected}  {reference['source_basename']}\n")
    parent = tmp_path / "temporary" / "parent.fits"
    plan = extractor.extraction_plan(0, "LRG", parent, tmp_path / "slice.fits", config)
    requested = []

    def open_source(url, timeout):
        requested.append(url)
        return io.BytesIO(content)

    monkeypatch.setattr(urllib.request, "urlopen", open_source)
    if bad_checksum:
        with pytest.raises(ValueError, match="parent checksum mismatch"):
            extractor.download_parent(plan, tmp_path, config)
    else:
        extractor.download_parent(plan, tmp_path, config)
    assert parent.read_bytes() == content
    assert requested == [plan["parent_url"]]
    with pytest.raises(FileExistsError):
        extractor.download_parent(plan, tmp_path, config)
    assert requested == [plan["parent_url"]]
