"""Plan or extract a same-tracer parent-catalogue reservoir slice.

The utility is non-controlling: all cuts come from ``analysis.yaml`` and all
paths are explicit. The CLI prints an extraction plan by default; ``--extract`` performs one
explicitly requested extraction. Existing parents are preserved. The separate
``--download-parent`` option owns one new temporary download and removes it only after verified
slice publication.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pzbao.catalog import (  # noqa: E402
    file_sha256,
    planned_catalogue_path,
    read_fits_catalog,
    read_reservoir_provenance,
    reservoir_spec,
    reservoir_selection_definition,
    validate_reservoir_selection,
    validate_reservoir_extension,
    _publisher_checksums,
)
from pzbao.config import load_analysis_config  # noqa: E402


def extraction_windows(tracer: str, config: Mapping[str, Any]) -> Tuple[Tuple[float, float], ...]:
    """Return outside reservoir and inside calibration windows from YAML."""
    spec = reservoir_spec(tracer, config)
    calibration_width = float(spec["calibration_band_width_redshift"])
    windows = []
    for edge in spec["edges"]:
        half_width = float(edge.get("slice_half_width_redshift", spec["slice_half_width_redshift"]))
        redshift, side = float(edge["redshift"]), str(edge["side"])
        if side == "below":
            windows.extend(
                [
                    (max(0.0, redshift - half_width), redshift),
                    (redshift, redshift + calibration_width),
                ]
            )
        elif side == "above":
            windows.extend(
                [(redshift, redshift + half_width), (redshift - calibration_width, redshift)]
            )
        else:
            raise ValueError(f"invalid reservoir side {side!r}")
    # Keep reviewed decimal cut edges legible and byte-stable in plans rather
    # than exposing binary floating-point artifacts such as 0.10000000000000003.
    return tuple((round(lo, 12), round(hi, 12)) for lo, hi in windows)


def extract_slice_arrays(
    parent_path: Path,
    tracer: str,
    config: Optional[Mapping[str, Any]] = None,
    *,
    auxiliary_root: Path,
    mock_id: int,
) -> Tuple[Dict[str, np.ndarray], list]:
    """Apply publisher selection and return native arrays plus verified auxiliary identities."""
    resolved = config or load_analysis_config()
    spec = reservoir_spec(tracer, resolved)
    source = read_fits_catalog(parent_path, columns=spec["parent_required_columns"])
    hardware = resolved["selection"]["reservoir_hardware"]
    flag = spec["target_flag"]
    redshift = np.asarray(source["RSDZ"], dtype=float)
    selected = np.zeros(len(redshift), dtype=bool)
    for lo, hi in extraction_windows(tracer, resolved):
        selected |= (redshift >= lo) & (redshift < hi)
    selected &= ~np.asarray(source["COLLISION"], dtype=bool)
    selected &= (source[flag["column"]] & (1 << flag["bit"])) != 0
    selected &= source["TARGETID"] <= hardware["maximum_target_id"]
    veto = sum(1 << int(bit) for bit in spec["veto_maskbits"])
    selected &= (np.asarray(source["MASKBITS"], dtype=np.int64) & veto) == 0
    if "absolute_magnitude_cut" in spec:
        selected &= (source["R_MAG_ABS"] + spec["absolute_magnitude_offset"]) < spec[
            "absolute_magnitude_cut"
        ]
    # Restrict the large parent before reading masks; retain native coordinate precision
    # for the publisher's pixel lookup. RSDZ becomes Z only after selection.
    source = {name: np.asarray(values)[selected] for name, values in source.items()}
    good_locations, inputs = publisher_good_locations(tracer, Path(auxiliary_root), resolved)
    location = (
        source["TILEID"].astype(np.int64) * hardware["tile_location_multiplier"]
        + source["LOCATION"]
    )
    keep = np.isin(location, good_locations)
    source = {name: values[keep] for name, values in source.items()}
    # A target can have several potential assignments. Accept any good one, then
    # collapse only identical native galaxy rows; assignment-specific columns differ.
    order = np.argsort(source["TARGETID"], kind="stable")
    source = {name: values[order] for name, values in source.items()}
    duplicate = source["TARGETID"][1:] == source["TARGETID"][:-1]
    galaxy_columns = ["TARGETID", "RA", "DEC", "RSDZ", "MASKBITS"]
    if "absolute_magnitude_cut" in spec:
        galaxy_columns.append("R_MAG_ABS")
    for name in galaxy_columns:
        if np.any(duplicate & (source[name][1:] != source[name][:-1])):
            raise ValueError(f"conflicting native values for repeated TARGETID: {name}")
    indices = np.flatnonzero(np.r_[True, ~duplicate]) if len(source["TARGETID"]) else []
    source = {name: source[name][indices] for name in galaxy_columns}
    source["PHOTSYS"] = publisher_photsys(source["RA"], source["DEC"], resolved)
    selected, imaging_inputs = publisher_imaging_selection(
        source, tracer, mock_id, Path(auxiliary_root), resolved
    )
    return {
        ("Z" if name == "RSDZ" else name): np.asarray(values)[selected].copy()
        for name, values in source.items()
        if name != "MASKBITS"
    }, inputs + imaging_inputs


def publisher_photsys(ra, dec, config):
    """Reproduce the complete builder's imaging-system boundary."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    spec = config["selection"]["reservoir_hardware"]
    latitude = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs").galactic.b.deg
    north = (dec > spec["north_declination_min_deg"]) & (
        latitude > spec["north_galactic_latitude_min_deg"]
    )
    return np.where(north, "N", "S")


def publisher_good_locations(tracer, auxiliary_root, config):
    """Return good potential tile-locations and their verified direct input identities."""
    spec = config["selection"]["reservoir_hardware"]
    inputs, paths = [], {}
    for key in (reservoir_spec(tracer, config)["hardware_program"], "bad_fibers"):
        reference = spec["references"][key]
        name = reference["filename"]
        path = Path(auxiliary_root) / name
        digest = file_sha256(path)
        if digest != reference["sha256"]:
            raise ValueError(f"publisher hardware checksum mismatch: {path}")
        inputs.append(
            {
                "name": name,
                "source_url": spec["base_url"].rstrip("/") + "/" + name,
                "sha256": digest,
            }
        )
        paths[key] = path
    program = reservoir_spec(tracer, config)["hardware_program"]
    table = read_fits_catalog(
        paths[program], columns=["TILEID", "LOCATION", "FIBER", "ZWARN", "ZWARN_MTL"]
    )
    bad_fibers = np.loadtxt(paths["bad_fibers"], dtype=np.int64, ndmin=1)
    good = np.isfinite(table["ZWARN"]) & (table["ZWARN"] != spec["missing_zwarn"])
    good &= (table["ZWARN_MTL"] & sum(1 << bit for bit in spec["veto_zwarn_mtl_bits"])) == 0
    good &= ~np.isin(table["FIBER"], bad_fibers)
    locations = (
        table["TILEID"].astype(np.int64) * spec["tile_location_multiplier"] + table["LOCATION"]
    )
    return np.unique(locations[good]), inputs


def publisher_imaging_selection(source, tracer, mock_id, auxiliary_root, config):
    """Evaluate the release's maps without silently changing versions or target-ID joins."""
    import healpy as hp

    spec = reservoir_spec(tracer, config)
    imaging = config["selection"]["reservoir_imaging"]
    inputs = []

    def read_source(name, url, expected, columns):
        path = auxiliary_root / name
        digest = file_sha256(path)
        if digest != expected:
            raise ValueError(f"publisher selection checksum mismatch: {path}")
        inputs.append({"name": name, "source_url": url, "sha256": digest})
        return read_fits_catalog(path, columns=columns)

    selected = np.ones(len(source["RA"]), dtype=bool)
    if "target_mask" in spec:
        mask_spec = spec["target_mask"]
        name = mask_spec["filename_template"].format(mock_id=mock_id)
        publisher = _publisher_checksums(auxiliary_root / mask_spec["checksum_manifest"])
        if name not in publisher:
            raise ValueError("publisher manifest omits the target mask")
        mask = read_source(
            name,
            config["tracers"][tracer]["catalogue"]["base_url"].rstrip("/") + "/" + name,
            publisher[name],
            ["TARGETID", mask_spec["column"]],
        )
        # Sort only rows needed by this slice, but refuse ambiguous or missing joins.
        relevant = np.isin(mask["TARGETID"], source["TARGETID"])
        ids = mask["TARGETID"][relevant]
        values = mask[mask_spec["column"]][relevant]
        order = np.argsort(ids)
        ids, values = ids[order], values[order]
        indices = np.searchsorted(ids, source["TARGETID"])
        if (
            np.any(ids[1:] == ids[:-1])
            or np.any(indices >= len(ids))
            or not np.array_equal(ids[indices], source["TARGETID"])
        ):
            raise ValueError("target mask must cover each candidate TARGETID exactly once")
        selected &= values[indices] == mask_spec["accepted_value"]
    nside = imaging["nside"]
    # Same arithmetic as the release builder, including native float32 rounding.
    theta = (-source["DEC"] + 90.0) * np.pi / 180.0
    phi = source["RA"] * np.pi / 180.0
    pix = hp.ang2pix(nside, theta, phi, nest=True)
    if not np.all(np.isin(source["PHOTSYS"], ["N", "S"])):
        raise ValueError("publisher imaging selection requires PHOTSYS N or S")
    for cap in ("N", "S"):
        name = imaging["filename_template"].format(
            tracer=spec["imaging_map_tracer"], nside=nside, cap=cap
        )
        table = read_source(
            name,
            imaging["base_url"].rstrip("/") + "/" + name,
            imaging["sha256"][spec["imaging_map_tracer"]][cap],
            list(imaging["cuts"]),
        )
        if any(len(values) != hp.nside2npix(nside) for values in table.values()):
            raise ValueError("publisher imaging map has incompatible pixel coverage")
        in_cap = source["PHOTSYS"] == cap
        pixels = pix[in_cap]
        accepted = np.ones(len(pixels), dtype=bool)
        for column, cut in imaging["cuts"].items():
            values = table[column][pixels]
            if cut["operator"] == "dust_corrected_gt":
                values = values * 10 ** (
                    -0.4 * cut["extinction_coefficient"] * table["EBV"][pixels]
                )
                # DESI computes native map values, then stores them in a float64
                # comparison array. Preserve strict endpoints for float32 maps.
                accepted &= values.astype(float) > cut["value"]
            else:
                values = values.astype(float)
                if cut["operator"] == "log10_lt":
                    with np.errstate(divide="ignore", invalid="ignore"):
                        values = np.log10(values)
                accepted &= values < cut["value"]
        selected[in_cap] &= accepted
    return selected, inputs


def _interior_band_mask(redshift, edge, width, publisher_dtype):
    """Compare the interior overlap using the publisher's strict lower field cut."""
    z, side = float(edge["redshift"]), edge["side"]
    lo, hi = (z, z + width) if side == "below" else (z - width, z)
    values = np.asarray(redshift, dtype=float)
    inside = (values >= lo) & (values < hi)
    if side == "below":
        # DESI cuts Z > zmin in native precision. The float32 value labelled 0.4
        # is above decimal 0.4 after promotion, but was excluded by that cut.
        # Use the clustering column's dtype even after a slice is saved as float64.
        inside &= values > float(np.asarray(z, dtype=publisher_dtype))
    return inside


def calibration_ratios(
    slice_catalog: Mapping[str, Any],
    clustering_catalogues: Sequence[Mapping[str, Any]],
    tracer: str,
    config: Mapping[str, Any],
) -> Tuple[dict, ...]:
    """Measure parent/clustering ratios in the YAML-declared inside bands."""
    if not clustering_catalogues:
        raise ValueError("calibration requires explicit NGC/SGC clustering catalogues")
    clustering_z = np.concatenate([np.asarray(catalog["Z"]) for catalog in clustering_catalogues])
    spec = reservoir_spec(tracer, config)
    width = float(spec["calibration_band_width_redshift"])
    records = []
    for edge in spec["edges"]:
        redshift, side = float(edge["redshift"]), str(edge["side"])
        parent_count = int(
            np.count_nonzero(
                _interior_band_mask(slice_catalog["Z"], edge, width, clustering_z.dtype)
            )
        )
        clustering_count = int(
            np.count_nonzero(_interior_band_mask(clustering_z, edge, width, clustering_z.dtype))
        )
        if clustering_count <= 0:
            raise ValueError(f"clustering calibration band at z={redshift} is empty")
        records.append(
            {
                "edge": redshift,
                "side": side,
                "parent_count": parent_count,
                "clustering_count": clustering_count,
                "parent_to_clustering_ratio": parent_count / clustering_count,
            }
        )
    return tuple(records)


def verify_interior_membership(
    slice_catalog, clustering_catalogues, tracer, config, *, parent_path=None
):
    """Check eligibility on available source rows and report absent parent material."""
    spec = reservoir_spec(tracer, config)
    columns = ["RA", "DEC", "Z"]
    clustering = {
        col: np.concatenate([catalog[col] for catalog in clustering_catalogues]) for col in columns
    }
    records, raw_parent = [], None
    for edge in spec["edges"]:
        z, width = edge["redshift"], spec["calibration_band_width_redshift"]

        def identities(catalog):
            inside = _interior_band_mask(catalog["Z"], edge, width, clustering["Z"].dtype)
            return np.rec.fromarrays([np.asarray(catalog[c], dtype=float)[inside] for c in columns])

        parent, complete = identities(slice_catalog), identities(clustering)
        if len(np.unique(parent)) != len(parent) or len(np.unique(complete)) != len(complete):
            raise ValueError("interior comparison contains duplicate physical galaxies")
        extra = np.count_nonzero(~np.isin(parent, complete))
        absent = complete[~np.isin(complete, parent)]
        unavailable = 0
        if len(absent) and parent_path is not None:
            if raw_parent is None:
                raw_parent = read_fits_catalog(parent_path, columns=["RA", "DEC", "RSDZ"])
                raw_parent["Z"] = raw_parent.pop("RSDZ")
            unavailable = np.count_nonzero(~np.isin(absent, identities(raw_parent)))
        missing = len(absent) - unavailable
        if extra or missing:
            raise ValueError(
                f"reservoir interior membership differs at z={z}: {extra} extra, {missing} missing"
            )
        records.append({"edge": z, "source_unavailable_count": int(unavailable)})
    return records


def write_slice(
    output_path: Path,
    catalog: Mapping[str, Any],
    calibration: Sequence[Mapping[str, Any]],
    *,
    tracer: str,
    mock_id: int,
    parent_path: Path,
    phase: str,
    selection_inputs: Optional[list] = None,
    config: Optional[Mapping[str, Any]] = None,
    holdout_release_artifact_sha256: Optional[str] = None,
    reference_slice: Optional[Path] = None,
) -> Path:
    """Write one atomic FITS slice with input and calibration provenance."""
    mock_id = int(mock_id)
    if phase == "calibration":
        # Every realization is a calibration mock now, and none waits on a one-shot release.
        if mock_id not in range(25) or holdout_release_artifact_sha256 is not None:
            raise ValueError("calibration reservoir publication covers mocks 0--24 with no release")
    elif phase == "holdout":
        raise ValueError("the holdout phase is retired and publishes no reservoir")
    else:
        raise ValueError("reservoir publication phase must be calibration or holdout")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite reservoir slice: {output_path}")
    if not selection_inputs:
        raise ValueError("reservoir publication requires verified publisher selection inputs")
    definition = reservoir_selection_definition(tracer, config or load_analysis_config())
    parent_digest = file_sha256(parent_path)
    extension_reference = (
        {
            "path": str(reference_slice.resolve()),
            "sha256": file_sha256(reference_slice),
            "redshift_dtype": np.asarray(catalog["Z"]).dtype.str,
        }
        if reference_slice is not None
        else None
    )
    validate_reservoir_extension(
        catalog,
        {
            "tracer": tracer,
            "mock_id": mock_id,
            "cap": "GCCOMB",
            "parent_artifact_sha256": parent_digest,
            "selection": definition,
            "selection_inputs": selection_inputs,
            "extension_reference": extension_reference,
        },
        config or load_analysis_config(),
    )
    validate_reservoir_selection(
        {"selection": definition, "selection_inputs": selection_inputs},
        tracer,
        mock_id,
        config or load_analysis_config(),
    )
    try:
        from astropy.io import fits
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("astropy is required to write reservoir slices") from exc
    columns = []
    for name, values in catalog.items():
        array = np.asarray(values)
        format_code = "D" if np.issubdtype(array.dtype, np.floating) else "K"
        if array.dtype.kind in {"U", "S"}:
            width = max(int(array.dtype.itemsize), 1)
            format_code = f"{width}A"
        columns.append(fits.Column(name=name, array=array, format=format_code))
    table = fits.BinTableHDU.from_columns(columns)
    table.header["TRACER"] = tracer
    table.header["MOCKID"] = mock_id
    table.header["PHASE"] = phase
    table.header["RELEASE"] = holdout_release_artifact_sha256 or "NONE"
    # GCCOMB is a historical logical label for the NGC+SGC parent reservoir, not an input URL.
    table.header["CAP"] = "GCCOMB"
    table.header["PARENTSH"] = parent_digest
    table.header["ZSOURCE"] = "RSDZ"
    table.header["ZACTIVE"] = "Z"
    table.header["SELDEF"] = json.dumps(definition, sort_keys=True)
    table.header["SELINPUT"] = json.dumps(selection_inputs, sort_keys=True)
    if extension_reference is not None:
        table.header["EXTREF"] = json.dumps(extension_reference, sort_keys=True)
    for index, record in enumerate(calibration):
        table.header[f"CALZ{index}"] = float(record["edge"])
        table.header[f"CALS{index}"] = str(record["side"])
        table.header[f"CALR{index}"] = float(record["parent_to_clustering_ratio"])
        table.header[f"CALP{index}"] = int(record["parent_count"])
        table.header[f"CALC{index}"] = int(record["clustering_count"])
        table.header[f"CALU{index}"] = int(record.get("source_unavailable_count", 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.partial")
    if temporary.exists():
        raise FileExistsError(f"stale reservoir partial exists: {temporary}")
    fits.HDUList([fits.PrimaryHDU(), table]).writeto(temporary, overwrite=False)
    os.link(temporary, output_path)
    temporary.unlink()
    return output_path


def extraction_plan(
    mock_id: int,
    tracer: str,
    parent_path: Path,
    output_path: Path,
    config: Optional[Mapping[str, Any]] = None,
) -> dict:
    """Describe a parent-retaining extraction without downloading, reading, or writing."""
    resolved = config or load_analysis_config()
    spec = reservoir_spec(tracer, resolved)
    base = resolved["tracers"][tracer]["catalogue"]["base_url"].rstrip("/")
    return {
        "mock_id": int(mock_id),
        "tracer": tracer,
        "parent_url": f"{base}/mock{int(mock_id)}/{spec['filename']}",
        "parent_path": str(Path(parent_path).resolve()),
        "output_path": str(Path(output_path).resolve()),
        "windows": extraction_windows(tracer, resolved),
    }


def download_parent(plan, catalog_root, config):
    """Acquire one new temporary parent, leaving failed or existing files untouched."""
    from urllib.request import urlopen

    reference = config["tracers"][plan["tracer"]]["source_reference"]
    manifest = Path(catalog_root) / reference["checksum_manifest_template"].format(
        mock_id=plan["mock_id"]
    )
    expected = _publisher_checksums(manifest)[reference["source_basename"]]
    path = Path(plan["parent_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as destination:
        with urlopen(plan["parent_url"], timeout=120) as response:
            shutil.copyfileobj(response, destination, length=1024**2)
    if file_sha256(path) != expected:
        raise ValueError(f"publisher parent checksum mismatch; retained for inspection: {path}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mock-id", type=int, required=True)
    parser.add_argument("--tracer", choices=["LRG", "BGS_BRIGHT-21.5"], required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--analysis-config", type=Path, required=True)
    parser.add_argument(
        "--auxiliary-root", type=Path, help="Release-matched masks and imaging maps."
    )
    parser.add_argument(
        "--extract", action="store_true", help="Write one slice from existing local inputs."
    )
    parser.add_argument(
        "--download-parent",
        action="store_true",
        help="With --extract, download to a NEW --parent path; delete only that download after verified publication.",
    )
    parser.add_argument(
        "--catalog-root",
        type=Path,
        help="Complete clustering catalogues for the two calibration bands.",
    )
    parser.add_argument(
        "--reference-slice",
        type=Path,
        help="Original immutable slice; required for an upper-edge extension.",
    )
    args = parser.parse_args(argv)
    config = load_analysis_config(args.analysis_config)
    if args.mock_id not in config["analysis"]["mock_ids"]:
        parser.error("--mock-id must name a configured realization")
    if args.extract and args.catalog_root is None:
        parser.error("--extract requires --catalog-root")
    if args.extract and args.auxiliary_root is None:
        parser.error("--extract requires --auxiliary-root")
    plan = extraction_plan(args.mock_id, args.tracer, args.parent, args.output, config)
    plan["download_parent"] = args.download_parent
    if args.extract:
        # Refuse before reading large parents if this result is already present.
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite reservoir slice: {args.output}")
        if args.download_parent:
            download_parent(plan, args.catalog_root, config)
        clustering = [
            read_fits_catalog(
                planned_catalogue_path(
                    args.catalog_root, args.mock_id, args.tracer, "data", cap, config=config
                ),
                columns=["RA", "DEC", "Z"],
            )
            for cap in ("NGC", "SGC")
        ]
        catalogue, selection_inputs = extract_slice_arrays(
            args.parent,
            args.tracer,
            config,
            auxiliary_root=args.auxiliary_root,
            mock_id=args.mock_id,
        )
        membership = verify_interior_membership(
            catalogue, clustering, args.tracer, config, parent_path=args.parent
        )
        calibration = calibration_ratios(catalogue, clustering, args.tracer, config)
        calibration = tuple(
            {**counts, **identity} for counts, identity in zip(calibration, membership)
        )
        write_slice(
            args.output,
            catalogue,
            calibration,
            tracer=args.tracer,
            mock_id=args.mock_id,
            parent_path=args.parent,
            phase="calibration",
            selection_inputs=selection_inputs,
            config=config,
            reference_slice=args.reference_slice,
        )
        saved = read_fits_catalog(args.output)
        if any(not np.array_equal(saved[name], values) for name, values in catalogue.items()):
            raise ValueError("published reservoir changed native catalogue values; parent retained")
        validate_reservoir_selection(
            read_reservoir_provenance(args.output), args.tracer, args.mock_id, config
        )
        if args.download_parent:
            args.parent.unlink()
            plan["temporary_download_removed"] = True
        plan["written_rows"] = len(catalogue["Z"])
        plan["calibration"] = calibration
        plan["selection_inputs"] = selection_inputs
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())
