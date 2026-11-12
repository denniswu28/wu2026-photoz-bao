"""Read configured catalogues and preserve physical source identities.

Catalogues are mappings of equal-length one-dimensional arrays. Registration records existing
source paths, catalogue coordinates, byte digests and file metadata; supplied publisher checksums
are verified once. Consumers rehash only when file metadata changes. Reservoir FITS
headers identify their tracer, realization and redshift semantics.
Object-bound IDs and coordinate conversions are shared by all downstream science.
"""

from __future__ import annotations

import hashlib
import json
import time
from copy import deepcopy
from pathlib import Path
from string import Formatter
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    from .config import load_analysis_config
    from .config.schema_validation import FULL_SKY_AREA_DEG2
    from .artifacts import resolve_saved_path
    from .storage import same_location
except ImportError:  # direct execution from the package directory
    from config import load_analysis_config
    from config.schema_validation import FULL_SKY_AREA_DEG2
    from artifacts import resolve_saved_path
    from storage import same_location

Catalog = Dict[str, np.ndarray]
_ANGULAR_COLUMNS = ("RA", "DEC")
# Bound inter-process messages to roughly one MiB of required input columns.
_ROW_CHUNK_BYTES = 1 << 20


def _tracer_spec(tracer: str, config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return one registered tracer block without inventing aliases or defaults."""
    tracers = config["tracers"]
    if tracer not in tracers:
        raise ValueError(f"unsupported tracer {tracer!r}; choose from {sorted(tracers)}")
    return tracers[tracer]


def validate_catalog(catalog: Mapping[str, Any], required: Sequence[str] = ()) -> int:
    """Validate equal-length, strictly one-dimensional columns and return row count."""
    if not isinstance(catalog, Mapping) or not catalog:
        raise ValueError("catalogue must be a non-empty mapping")
    missing = sorted(set(required) - set(catalog))
    if missing:
        raise KeyError(f"catalogue is missing required columns {missing}")
    if any(np.asarray(value).ndim != 1 for value in catalog.values()):
        raise ValueError("catalogue columns must be one-dimensional")
    lengths = {len(np.asarray(value)) for value in catalog.values()}
    if len(lengths) != 1:
        raise ValueError(f"catalogue columns have inconsistent lengths {sorted(lengths)}")
    return lengths.pop()


def copy_catalog(catalog: Mapping[str, Any]) -> Catalog:
    """Return a deep array copy so public transformations never mutate inputs."""
    validate_catalog(catalog)
    return {str(key): np.asarray(value).copy() for key, value in catalog.items()}


def _map_row_chunks(function, columns, *, dtype, parallel=None, **kwargs):
    """Evaluate unchanged row kernels in order, sending only their required columns.

    The caller owns a reusable joblib pool. Small inputs retain the direct serial path;
    large inputs stream ordered results into one output array, without full-catalogue
    worker copies or temporary memory-mapped catalogue files.
    """
    size = validate_catalog(columns)
    row_bytes = sum(np.asarray(value).dtype.itemsize for value in columns.values())
    chunk_rows = max(1, _ROW_CHUNK_BYTES // row_bytes)
    if parallel is None or size <= chunk_rows:
        return function(columns, **kwargs)
    from joblib import delayed

    starts = range(0, size, chunk_rows)
    pieces = parallel(
        delayed(function)(
            {
                name: np.asarray(value)[start : start + chunk_rows]
                for name, value in columns.items()
            },
            **kwargs,
        )
        for start in starts
    )
    output = np.empty(size, dtype=dtype)
    for start, values in zip(starts, pieces):
        if values.shape != (min(chunk_rows, size - start),):
            raise RuntimeError("row kernel returned an incompatible chunk shape")
        output[start : start + len(values)] = values
    return output


def measurement_bins(tracer: str, config: Optional[Mapping[str, Any]] = None) -> Tuple[dict, ...]:
    """Return deep copies of the configured measurement-bin records."""
    resolved = config or load_analysis_config()
    return tuple(deepcopy(item) for item in _tracer_spec(tracer, resolved)["measurement_bins"])


def measurement_bin(tracer: str, name: str, config: Optional[Mapping[str, Any]] = None) -> dict:
    """Return one named measurement-bin record or fail explicitly."""
    for item in measurement_bins(tracer, config):
        if item["name"] == name:
            return item
    raise KeyError(f"no measurement bin {name!r} for tracer {tracer!r}")


def reconstruction_field(tracer: str, config: Optional[Mapping[str, Any]] = None) -> dict:
    """Return a deep copy of the broad redshift field reconstructed for a tracer."""
    resolved = config or load_analysis_config()
    return deepcopy(_tracer_spec(tracer, resolved)["field"])


# Configuration-backed sample identity; stable object identity is defined later
# beside the catalogue columns from which it is derived.


def final_sample_identity(
    tracer: str,
    tier: str,
    convention: str,
    smoothing_mpc_h: float,
) -> str:
    """Derive an identity from sample coordinates shared across mocks and caps.

    Reconstruction convention and smoothing remain part of the identity even
    for pre-reconstruction products. This deliberately prevents any artifact
    from one declared reconstruction cell being substituted into another. The
    pipeline appends a cap only when it needs cap-specific evidence. Density and
    smoothing use positional formatting with trimmed trailing zeros, so identity
    never depends on exponent notation or Python's float ``repr``. Every producer
    of a declared sample identity must call this helper rather than format anew.
    """
    if convention != "recsym":
        raise ValueError("RecSym is the sole primary reconstruction convention")
    smoothing = np.format_float_positional(float(smoothing_mpc_h), trim="-")
    return (
        f"{tracer}|tier={tier}|density=primary_spherex_motivated|"
        f"reconstruction={convention}|"
        f"smoothing_mpc_h={smoothing}"
    )


def footprint_area_deg2(tracer: str, cap: str, config: Optional[Mapping[str, Any]] = None) -> float:
    """Return the configured mock-footprint area; combined caps are an exact sum.

    This accessor cannot remeasure a mask or verify provenance. It returns the
    values declared in ``analysis.yaml``; the parameter ledger separately
    records their claimed origin and whether it has been checked.
    """
    resolved = config or load_analysis_config()
    areas = _tracer_spec(tracer, resolved)["footprint"]["area_deg2"]
    cap_name = cap.upper()
    if cap_name == "GCCOMB":
        return float(areas["NGC"]) + float(areas["SGC"])
    if cap_name not in areas:
        raise ValueError("cap must be NGC, SGC, or GCcomb")
    return float(areas[cap_name])


def sky_fraction(tracer: str, cap: str, config: Optional[Mapping[str, Any]] = None) -> float:
    """Convert the configured footprint area to a sky fraction without validating its origin."""
    return footprint_area_deg2(tracer, cap, config) / FULL_SKY_AREA_DEG2


def build_mock_cosmology(config: Optional[Mapping[str, Any]] = None) -> Any:
    """Instantiate and verify the exact configured mock cosmology.

    The provider is the cosmoprimo DESI preset whose inputs reproduce the
    AbacusSummit c000 Planck-2018 cosmology. Configuration separates three
    meanings that must not be conflated: fundamental inputs define the
    cosmology, official Abacus sigma8 values document published provenance, and
    the realized fingerprint freezes the output of the installed cosmoprimo
    version/CLASS engine. The official derived values are not used as a loose
    tolerance envelope for a different engine result.
    """
    resolved = config or load_analysis_config()
    spec = resolved["physics"]["mock_cosmology"]
    if spec["provider"] != "cosmoprimo.fiducial.DESI":
        raise ValueError("the configured mock-cosmology provider is not supported")
    try:
        from cosmoprimo.fiducial import DESI
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("cosmoprimo is required for analysis coordinates") from exc
    cosmology = DESI(engine=str(spec["engine"]))
    inputs = spec["fundamental_inputs"]
    checks = {
        "h": float(cosmology.h),
        "omega_b_h2": float(cosmology.get("omega_b")),
        "omega_cdm_h2": float(cosmology.get("omega_cdm")),
        "A_s": float(np.asarray(cosmology.get("A_s"))),
        "n_s": float(cosmology.get("n_s")),
        "alpha_s": float(cosmology.get("alpha_s")),
        "N_ur": float(np.asarray(cosmology.get("N_ur"))),
        "omega_ncdm_h2": float(np.asarray(cosmology.get("omega_ncdm")).sum()),
        "tau_reio": float(cosmology.get("tau_reio")),
        "w0": float(np.asarray(cosmology.get("w0_fld"))),
        "wa": float(np.asarray(cosmology.get("wa_fld"))),
    }
    tolerances = spec["verification_absolute_tolerances"]
    for name, actual in checks.items():
        expected = float(inputs[name])
        if not np.isclose(actual, expected, atol=tolerances[name], rtol=0.0):
            raise ValueError(f"cosmoprimo {name}={actual} does not match configured {expected}")
    if int(cosmology.get("N_ncdm")) != int(inputs["N_ncdm"]):
        raise ValueError("cosmoprimo N_ncdm does not match the exact configured integer")
    return cosmology


# Upstream catalogue resolution and explicit input I/O


def catalogue_url(
    mock_id: int,
    tracer: str,
    role: str,
    cap: str,
    *,
    random_split: Optional[int] = None,
    config: Optional[Mapping[str, Any]] = None,
) -> str:
    """Resolve an upstream catalogue URL without selecting a local path.

    The mock id must be registered. Active tracers require distinct physical NGC
    and SGC files; ``GCcomb`` remains only a repository-level logical aggregate
    label and can never identify an upstream URL. Data templates are forbidden
    from consuming any top-level replacement field rooted at ``split``.
    Random templates must consume that top-level root exactly once. This proves that split
    identity participates in formatting; injectivity of the rendered value also
    depends on the reviewed format specification. ``split=0`` is only a harmless
    formatter value for the data-template call and never identifies a data split.
    """
    resolved = config or load_analysis_config()
    analysis_ids = {int(value) for value in resolved["analysis"]["mock_ids"]}
    if int(mock_id) not in analysis_ids:
        raise ValueError(f"mock_id {mock_id} is not registered")
    spec = _tracer_spec(tracer, resolved)["catalogue"]
    strategy = str(spec["cap_strategy"])
    if strategy != "concat_ngc_sgc":
        raise ValueError("active catalogue loading requires physical NGC and SGC inputs")
    cap_name = cap.upper()
    if cap_name not in {"NGC", "SGC", "GCCOMB"}:
        raise ValueError("cap must be NGC, SGC, or GCcomb")
    if cap_name == "GCCOMB":
        raise ValueError("GCcomb is logical NGC+SGC only; declare physical NGC and SGC inputs")
    if role == "data":
        template = spec["data_template"]
        fields = [field for _, field, _, _ in Formatter().parse(str(template))]
        field_roots = [field.split(".", 1)[0].split("[", 1)[0] for field in fields if field]
        if "split" in field_roots:
            raise ValueError("data catalogue templates must not consume a random split")
        split = 0
    elif role == "random":
        if random_split is None:
            raise ValueError("random_split is required for a random catalogue")
        split = int(random_split)
        if split < 0 or split >= int(spec["random_splits"]):
            raise ValueError(f"random_split {split} is outside the registered range")
        template = spec["random_template"]
        fields = [field for _, field, _, _ in Formatter().parse(str(template))]
        field_roots = [field.split(".", 1)[0].split("[", 1)[0] for field in fields if field]
        if field_roots.count("split") != 1:
            raise ValueError("random catalogue templates must consume split exactly once")
    else:
        raise ValueError("role must be 'data' or 'random'")
    relative = str(template).format(mock_id=int(mock_id), cap=cap_name, split=split)
    return str(spec["base_url"]).rstrip("/") + "/" + relative


def planned_catalogue_path(
    cache_root: Path,
    mock_id: int,
    tracer: str,
    role: str,
    cap: str,
    *,
    random_split: Optional[int] = None,
    config: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Map a registered URL to an explicit caller-owned cache root."""
    url = catalogue_url(mock_id, tracer, role, cap, random_split=random_split, config=config)
    return resolve_saved_path(cache_root) / f"mock{int(mock_id)}" / url.rsplit("/", 1)[-1]


def read_fits_catalog(path: Path, columns: Optional[Sequence[str]] = None) -> Catalog:
    """Read selected scalar FITS columns into the catalogue contract.

    ``columns=None`` materializes the complete HDU-1 schema. Every selected
    column must be strictly one-dimensional; vector-valued FITS columns fail
    closed instead of being silently flattened. The measurement orchestrator
    deliberately requests the full schema because a reservoir merge must
    preserve every clustering-data column. That full-schema route reserves the
    names ``Z_RSD`` and ``Z_OBS`` for :func:`selection.smear_data_redshifts`;
    declared data and reservoir inputs must not natively ship either column. Rejection is
    enforced by ``read_registered_catalogue`` and by the
    modifier, not by this general-purpose reader.
    """
    path = resolve_saved_path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        from astropy.io import fits
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("astropy is required to read FITS catalogues") from exc
    with fits.open(path, memmap=True) as hdul:
        # Standard DESI catalogues store the binary table in extension HDU 1.
        table = hdul[1].data
        names = list(table.names)
        selected = names if columns is None else list(columns)
        missing = sorted(set(selected) - set(names))
        if missing:
            raise KeyError(f"{path} is missing requested FITS columns {missing}")
        result = {name: np.asarray(table[name]).copy() for name in selected}
    validate_catalog(result)
    return result


def read_reservoir_provenance(path: Path) -> dict:
    """Read a reservoir header and validate its locally decidable fields.

    Tracer, realization, cap and redshift semantics identify the source. An existing parent
    checksum remains descriptive upstream provenance; no parent-output hash is required.
    """
    path = resolve_saved_path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        from astropy.io import fits
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("astropy is required to read FITS reservoir provenance") from exc
    with fits.open(path, memmap=True) as hdul:
        # The derived reservoir preserves provenance in the table-extension header.
        header = hdul[1].header
        missing = [
            key for key in ("TRACER", "MOCKID", "CAP", "ZSOURCE", "ZACTIVE") if key not in header
        ]
        if missing:
            raise ValueError(f"reservoir FITS header is missing provenance {missing}")
        tracer = str(header["TRACER"]).strip()
        mock_id = header["MOCKID"]
        if type(mock_id) is not int or mock_id < 0:
            raise ValueError("reservoir realization ID must be a non-negative integer")
        cap = str(header["CAP"]).strip().upper()
        parent_sha256 = str(header["PARENTSH"]).strip().lower() if "PARENTSH" in header else None
        source_redshift_column = str(header["ZSOURCE"]).strip().upper()
        active_redshift_column = str(header["ZACTIVE"]).strip().upper()
        selection = json.loads(header["SELDEF"]) if "SELDEF" in header else None
        selection_inputs = json.loads(header["SELINPUT"]) if "SELINPUT" in header else None
        extension_reference = json.loads(header["EXTREF"]) if "EXTREF" in header else None
    if (
        not tracer
        or cap != "GCCOMB"
        or (
            parent_sha256 is not None
            and (
                len(parent_sha256) != 64
                or any(character not in "0123456789abcdef" for character in parent_sha256)
            )
        )
        or source_redshift_column != "RSDZ"
        or active_redshift_column != "Z"
    ):
        raise ValueError("reservoir FITS header provenance is invalid")
    return {
        "tracer": tracer,
        "mock_id": mock_id,
        "cap": cap,
        "parent_artifact_sha256": parent_sha256,
        "source_redshift_column": source_redshift_column,
        "active_redshift_column": active_redshift_column,
        "selection": selection,
        "selection_inputs": selection_inputs,
        **({"extension_reference": extension_reference} if extension_reference is not None else {}),
    }


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash an input artifact without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Pure catalogue cuts and stable object identity


def filter_redshift(
    catalog: Mapping[str, Any], zmin: float, zmax: float, z_col: str = "Z"
) -> Catalog:
    """Apply the caller-declared finite half-open cut and return new arrays.

    Non-finite redshifts fail the finite mask and are excluded rather than
    leaking into a measurement bin.
    """
    validate_catalog(catalog, (z_col,))
    if not float(zmin) < float(zmax):
        raise ValueError("redshift bounds must be increasing")
    redshift = np.asarray(catalog[z_col], dtype=float)
    mask = np.isfinite(redshift) & (redshift >= float(zmin)) & (redshift < float(zmax))
    return {str(key): np.asarray(value)[mask].copy() for key, value in catalog.items()}


def filter_galactic_cap(catalog: Mapping[str, Any], cap: str) -> Catalog:
    """Select NGC or SGC from a combined catalogue using Galactic latitude.

    RA and DEC are interpreted as ICRS coordinates before transformation to
    Galactic latitude. ``PHOTSYS`` describes the imaging system and is not the Galactic-cap
    label.  In particular, NGC contains both PHOTSYS values, so substituting
    that column would silently route objects to the wrong cap. The measure-zero
    boundary ``b == 0`` is assigned to SGC. An empty requested cap fails.
    """
    validate_catalog(catalog, _ANGULAR_COLUMNS)
    if any(
        np.any(~np.isfinite(np.asarray(catalog[name], dtype=float))) for name in _ANGULAR_COLUMNS
    ):
        raise ValueError("Galactic-cap selection requires finite RA and DEC")
    cap_name = cap.upper()
    if cap_name not in {"NGC", "SGC"}:
        raise ValueError("combined parent catalogues can be split only into NGC or SGC")
    try:
        import astropy.units as units
        from astropy.coordinates import SkyCoord
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "astropy is required to split a combined catalogue by Galactic cap"
        ) from exc
    galactic_latitude = SkyCoord(
        ra=np.asarray(catalog["RA"], dtype=float) * units.deg,
        dec=np.asarray(catalog["DEC"], dtype=float) * units.deg,
        frame="icrs",
    ).galactic.b.deg
    mask = galactic_latitude > 0.0 if cap_name == "NGC" else galactic_latitude <= 0.0
    if not np.any(mask):
        raise ValueError(f"combined parent catalogue has no {cap_name} rows")
    return {str(key): np.asarray(value)[mask].copy() for key, value in catalog.items()}


def concatenate_catalogs(catalogues: Iterable[Mapping[str, Any]]) -> Catalog:
    """Concatenate equal-schema catalogues without altering their inputs."""
    items = list(catalogues)
    if not items:
        raise ValueError("at least one catalogue is required")
    for item in items:
        validate_catalog(item)
    keys = set(items[0])
    if any(set(item) != keys for item in items[1:]):
        raise KeyError("catalogues must have identical columns before concatenation")
    return {key: np.concatenate([np.asarray(item[key]) for item in items]) for key in items[0]}


def _identity_tokens(
    catalog: Mapping[str, Any],
    tracer: str,
    mock_id: int,
    cap: str,
    role: str,
    random_split: Optional[int],
    coordinate_significant_digits: int,
) -> Iterable[bytes]:
    """Yield row-content tokens whose identities survive catalogue reordering.

    Data and reservoir identities require ``TARGETID``. Random identities use
    only immutable angular coordinates plus split and cap. In particular,
    upstream random ``Z`` is replaced by the accepted radial model and optional
    ``PHOTSYS`` is irrelevant to identity, so neither may perturb the token.
    The basis is role-defined rather than selected by optional input columns.
    Because role is part of every token, source-local ``TARGETID`` numbering in
    data and reservoir inputs cannot collide. Physical duplication is checked
    separately using native coordinates and source redshift before merging. The
    deliberately scalar, per-row token construction is linear in catalogue size;
    preserving exact cross-platform identity bytes takes priority over replacing
    it with a differently serialized vectorized route. The byte-exact UTF-8
    layouts are ``tracer|mock|cap|role|TARGETID`` for data/reservoir and
    ``tracer|mock|cap|random|split=<n>|<RA:.{digits}g>|<DEC:.{digits}g>`` for
    randoms. The significant-digit format, BLAKE2b 64-bit digest size, and
    little-endian integer conversion are frozen result-affecting contracts:
    changing any of them rerolls downstream scientific sample membership.
    """
    split_token = "" if random_split is None else f"|split={int(random_split)}"
    if role in {"data", "reservoir"}:
        if "TARGETID" not in catalog:
            raise KeyError(f"{role} identity requires TARGETID")
        for target_id in np.asarray(catalog["TARGETID"]):
            yield f"{tracer}|{mock_id}|{cap}|{role}{split_token}|{target_id}".encode()
        return
    validate_catalog(catalog, _ANGULAR_COLUMNS)
    if any(
        np.any(~np.isfinite(np.asarray(catalog[name], dtype=float))) for name in _ANGULAR_COLUMNS
    ):
        raise ValueError("random identity requires finite RA and DEC")
    for ra, dec in zip(catalog["RA"], catalog["DEC"]):
        ra_token = format(float(ra), f".{coordinate_significant_digits}g")
        dec_token = format(float(dec), f".{coordinate_significant_digits}g")
        yield (f"{tracer}|{mock_id}|{cap}|{role}{split_token}|{ra_token}|{dec_token}").encode()


def _source_identity_values(catalog, **kwargs):
    return np.fromiter(
        (
            int.from_bytes(hashlib.blake2b(token, digest_size=8).digest(), "little")
            for token in _identity_tokens(catalog, **kwargs)
        ),
        dtype=np.uint64,
        count=validate_catalog(catalog),
    )


def assign_source_identity(
    catalog: Mapping[str, Any],
    *,
    tracer: str,
    mock_id: int,
    cap: str,
    role: str,
    random_split: Optional[int] = None,
    column: str = "SOURCE_ID",
    config: Optional[Mapping[str, Any]] = None,
    parallel=None,
) -> Catalog:
    """Attach deterministic object-bound uint64 identities.

    Data/reservoir IDs bind ``TARGETID``; random IDs bind RA and DEC rounded to
    the configured significant-digit identity precision plus split. Both include
    tracer, mock, cap, and role and therefore survive row sorting, upstream-random
    redshift replacement, and optional-column changes. Role namespacing means
    these identities do not by themselves detect the same galaxy appearing in
    both data and reservoir inputs, whose TARGETID numbering can differ. The literal ``SOURCE_ID``
    default is not compared with the configured identity-column name; the
    controlling pipeline always passes that configured name explicitly through
    :func:`read_registered_catalogue`.
    """
    validate_catalog(catalog)
    if role not in {"data", "random", "reservoir"}:
        raise ValueError("role must be data, random, or reservoir")
    resolved = config or load_analysis_config()
    _tracer_spec(tracer, resolved)
    coordinate_digits = int(
        resolved["selection"]["stable_randomness"]["random_coordinate_significant_digits"]
    )
    if int(mock_id) not in {int(value) for value in resolved["analysis"]["mock_ids"]}:
        raise ValueError(f"mock_id {mock_id} is not registered")
    if role == "random":
        if random_split is None or int(random_split) < 0:
            raise ValueError("random source identity requires a non-negative random_split")
    elif random_split is not None:
        raise ValueError("only random source identity may declare random_split")
    identity_columns = _ANGULAR_COLUMNS if role == "random" else ("TARGETID",)
    if role in {"data", "reservoir"} and "TARGETID" not in catalog:
        raise KeyError(f"{role} identity requires TARGETID")
    validate_catalog(catalog, identity_columns)
    identities = _map_row_chunks(
        _source_identity_values,
        {name: catalog[name] for name in identity_columns},
        dtype=np.uint64,
        parallel=parallel,
        tracer=tracer,
        mock_id=int(mock_id),
        cap=cap.upper(),
        role=role,
        random_split=random_split,
        coordinate_significant_digits=coordinate_digits,
    )
    if len(np.unique(identities)) != len(identities):
        # Distinguishing a duplicate token from a digest collision requires a
        # wider second hash, which this fail-closed identity surface omits.
        raise ValueError("duplicate source keys or a 64-bit stable-identity digest collision")
    result = copy_catalog(catalog)
    result[column] = identities
    return result


# Coordinate and physical-volume conversion


def ra_dec_z_to_xyz(
    ra_deg: Sequence[float],
    dec_deg: Sequence[float],
    redshift: Sequence[float],
    cosmology: Any,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert sky coordinates using cosmoprimo's comoving-distance convention.

    The final pipeline supplies the configured mock cosmology. The pinned
    cosmoprimo version is expected to return comoving Mpc/h distances here,
    matching the units consumed by pyrecon, pycorr, and the shell-volume
    calculation below. Pinned-version compatibility evidence, not this function,
    establishes that external unit contract.
    """
    ra = np.deg2rad(np.asarray(ra_deg, dtype=float))
    dec = np.deg2rad(np.asarray(dec_deg, dtype=float))
    z = np.asarray(redshift, dtype=float)
    if not (ra.shape == dec.shape == z.shape):
        raise ValueError("RA, DEC, and redshift arrays must share a shape")
    distance = np.asarray(cosmology.comoving_radial_distance(z), dtype=float)
    cos_dec = np.cos(dec)
    return distance * cos_dec * np.cos(ra), distance * cos_dec * np.sin(ra), distance * np.sin(dec)


def add_cartesian_positions(
    catalog: Mapping[str, Any], cosmology: Any, *, z_col: str = "Z"
) -> Catalog:
    """Return X/Y/Z_CART in comoving Mpc/h from the explicitly named redshift.

    The measurement pipeline explicitly passes ``Z_OBS`` for data and ``Z`` for
    randoms. ``Z`` and ``Z_RSD`` preserve the incoming RSD-bearing source value;
    every retained random keeps its original ``Z``. The default remains available
    only to callers that explicitly operate on the source coordinate.
    """
    validate_catalog(catalog, ("RA", "DEC", z_col))
    x, y, z_cart = ra_dec_z_to_xyz(catalog["RA"], catalog["DEC"], catalog[z_col], cosmology)
    result = copy_catalog(catalog)
    result.update(X=x, Y=y, Z_CART=z_cart)
    return result


def comoving_shell_volume(cosmology: Any, z_edges: Sequence[float], area_deg2: float) -> np.ndarray:
    """Return shell volumes in (Mpc/h)^3 for a configured footprint area.

    This uses the same cosmoprimo comoving radial distance and Mpc/h convention
    as :func:`ra_dec_z_to_xyz`; the function does not independently audit the
    footprint provenance.
    """
    edges = np.asarray(z_edges, dtype=float)
    if edges.ndim != 1 or len(edges) < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError("z_edges must be a strictly increasing one-dimensional array")
    if not 0.0 < float(area_deg2) <= FULL_SKY_AREA_DEG2:
        raise ValueError("area_deg2 must be positive and no larger than the full sky")
    distance = np.asarray(cosmology.comoving_radial_distance(edges), dtype=float)
    full_sky_shell = 4.0 * np.pi / 3.0 * np.diff(distance**3)
    return full_sky_shell * float(area_deg2) / FULL_SKY_AREA_DEG2


# Same-tracer reservoir declarations and byte-bound input provenance


def reservoir_spec(tracer: str, config: Optional[Mapping[str, Any]] = None) -> dict:
    """Return the same-tracer reservoir declaration without selecting it."""
    resolved = config or load_analysis_config()
    return deepcopy(_tracer_spec(tracer, resolved)["reservoir"])


def reservoir_slice_path(
    cache_root: Path, mock_id: int, tracer: str, config: Optional[Mapping[str, Any]] = None
) -> Path:
    """Resolve the declared slice name under an explicit cache root."""
    return (
        resolve_saved_path(cache_root)
        / f"mock{int(mock_id)}"
        / reservoir_spec(tracer, config)["slice_filename"]
    )


def reservoir_selection_definition(tracer, config):
    """Readable source-selection values carried across extraction and consumption."""
    return {
        "reservoir": reservoir_spec(tracer, config),
        "imaging": deepcopy(config["selection"]["reservoir_imaging"]),
        "hardware": deepcopy(config["selection"]["reservoir_hardware"]),
    }


def compare_extended_reservoir(
    reference,
    extended,
    *,
    boundary,
    old_depth,
    new_depth,
    native_redshift_dtype=None,
    return_added_mask=False,
):
    """Require identical native old rows and additions only in the new upper-edge band."""
    for catalog in (reference, extended):
        validate_catalog(catalog, ("TARGETID", "RA", "DEC", "Z"))
        if len(np.unique(catalog["TARGETID"])) != len(catalog["TARGETID"]):
            raise ValueError("extension contains duplicate native target identities")
    if not set(reference).issubset(extended):
        raise ValueError("extension dropped native columns")
    _, old_rows, new_rows = np.intersect1d(
        reference["TARGETID"], extended["TARGETID"], assume_unique=True, return_indices=True
    )
    if len(old_rows) != len(reference["TARGETID"]):
        raise ValueError("extension dropped original native rows")
    for name in reference:
        if not np.array_equal(
            np.asarray(reference[name])[old_rows], np.asarray(extended[name])[new_rows]
        ):
            raise ValueError(f"extension changed original native values: {name}")
    added = np.ones(len(extended["TARGETID"]), dtype=bool)
    added[new_rows] = False
    z = np.asarray(extended["Z"])[added]
    dtype = np.dtype(native_redshift_dtype or z.dtype)
    if dtype.kind != "f" or dtype.itemsize not in (4, 8):
        raise ValueError("extension has invalid native redshift precision")
    native_z = z.astype(dtype)
    if not np.array_equal(native_z.astype(z.dtype), z):
        raise ValueError("extension redshifts differ from their native precision")
    # Match extraction's decimal cuts without rounding them to the native dtype.
    selection_z = native_z.astype(float)
    lower, upper = (round(boundary + depth, 12) for depth in (old_depth, new_depth))
    if not np.all((selection_z >= lower) & (selection_z < upper)):
        raise ValueError("extension added rows outside its new upper-edge band")
    counts = {"preserved_row_count": len(old_rows), "added_row_count": int(added.sum())}
    return (counts, added) if return_added_mask else counts


def validate_reservoir_extension(catalogue, embedded, config, *, return_added_mask=False):
    """Verify the direct original slice and native overlap at publication and consumption."""
    experiment = config["selection"]["boundary_support"].get("experiment")
    if (
        experiment is None
        or experiment["mode"] != "extend"
        or experiment["tracer"] != embedded["tracer"]
    ):
        return None
    reference = embedded.get("extension_reference")
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256", "redshift_dtype"}:
        raise ValueError("extension requires its direct original slice identity")
    path = resolve_saved_path(reference["path"])
    if file_sha256(path) != reference["sha256"]:
        raise ValueError("extension original slice identity changed")
    original = read_reservoir_provenance(path)
    for key in ("tracer", "mock_id", "cap", "parent_artifact_sha256", "selection_inputs"):
        if original[key] != embedded[key]:
            raise ValueError(f"extension original provenance differs: {key}")
    old_selection, new_selection = deepcopy(original["selection"]), deepcopy(embedded["selection"])
    old_spec = old_selection["reservoir"]
    old_edge = next(item for item in old_spec["edges"] if item["side"] == experiment["side"])
    if (
        old_edge.get("slice_half_width_redshift", old_spec["slice_half_width_redshift"])
        != experiment["reference_depth_redshift"]
    ):
        raise ValueError("extension original extraction depth differs")
    new_selection["reservoir"]["slice_filename"] = old_selection["reservoir"]["slice_filename"]
    for edge, old_edge in zip(
        new_selection["reservoir"]["edges"], old_selection["reservoir"]["edges"]
    ):
        if edge["side"] == experiment["side"]:
            edge.pop("slice_half_width_redshift", None)
            if "slice_half_width_redshift" in old_edge:
                edge["slice_half_width_redshift"] = old_edge["slice_half_width_redshift"]
    if old_selection != new_selection:
        raise ValueError("extension changed the original selection recipe")
    return compare_extended_reservoir(
        read_fits_catalog(path),
        catalogue,
        boundary=config["tracers"][experiment["tracer"]]["field"]["zmax"],
        old_depth=experiment["reference_depth_redshift"],
        new_depth=experiment["depth_redshift"],
        native_redshift_dtype=reference["redshift_dtype"],
        return_added_mask=return_added_mask,
    )


def validate_reservoir_selection(embedded, tracer, mock_id, config):
    """Compare saved selection values and each direct auxiliary source identity."""
    definition = reservoir_selection_definition(tracer, config)
    if embedded["selection"] != definition:
        raise ValueError("reservoir selection differs from the configured publisher recipe")
    spec, imaging = definition["reservoir"], definition["imaging"]
    expected = {}
    hardware = definition["hardware"]
    for key in (spec["hardware_program"], "bad_fibers"):
        reference = hardware["references"][key]
        name = reference["filename"]
        expected[name] = (hardware["base_url"].rstrip("/") + "/" + name, reference["sha256"])
    for cap in ("N", "S"):
        name = imaging["filename_template"].format(
            tracer=spec["imaging_map_tracer"], nside=imaging["nside"], cap=cap
        )
        expected[name] = (
            imaging["base_url"].rstrip("/") + "/" + name,
            imaging["sha256"][spec["imaging_map_tracer"]][cap],
        )
    if "target_mask" in spec:
        name = spec["target_mask"]["filename_template"].format(mock_id=mock_id)
        expected[name] = (
            config["tracers"][tracer]["catalogue"]["base_url"].rstrip("/") + "/" + name,
            None,
        )
    inputs = embedded["selection_inputs"]
    if not isinstance(inputs, list) or len(inputs) != len(expected):
        raise ValueError("reservoir lacks its publisher selection input identities")
    seen = set()
    for record in inputs:
        name = record.get("name") if isinstance(record, dict) else None
        digest = record.get("sha256") if isinstance(record, dict) else None
        if (
            name not in expected
            or name in seen
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
            or record.get("source_url") != expected[name][0]
            or (expected[name][1] is not None and digest != expected[name][1])
        ):
            raise ValueError("reservoir publisher selection input identity differs")
        seen.add(name)


def _source_file_metadata(path):
    stat = Path(path).stat()
    return {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "ctime_ns": stat.st_ctime_ns}


def _publisher_checksums(path):
    """Read the publisher's sha256sum text, including text/binary filename markers."""
    checksums = {}
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split(maxsplit=1)
        if (
            len(fields) != 2
            or len(fields[0]) != 64
            or any(character not in "0123456789abcdef" for character in fields[0].lower())
        ):
            raise ValueError(f"invalid publisher checksum manifest: {path}")
        name = Path(fields[1].lstrip("*")).name
        if not name or name in checksums:
            raise ValueError(f"ambiguous publisher checksum filename: {path}")
        checksums[name] = fields[0].lower()
    return checksums


def validate_boundary_input_sources(reference, variant, *, tracer, config):
    """Bind unchanged catalogue bytes and an extension's original slice to the reference."""
    experiment = config["selection"]["boundary_support"].get("experiment")
    if experiment is None:
        return
    keys = ("tracer", "mock_id", "cap", "role", "random_split")
    indexed = []
    for records in (reference, variant):
        entries = {
            tuple(record.get(key) if key == "random_split" else record[key] for key in keys): record
            for record in records
        }
        if not entries or len(entries) != len(records):
            raise ValueError("boundary inputs have missing or duplicated source identities")
        if any(key[0] != tracer for key in entries):
            raise ValueError("boundary inputs belong to another tracer")
        indexed.append(entries)
    original, changed = indexed
    if original.keys() != changed.keys():
        raise ValueError("boundary inputs do not cover the reference sources")
    for key, before in original.items():
        after = changed[key]
        if (
            experiment["mode"] == "extend"
            and tracer == experiment["tracer"]
            and key[3] == "reservoir"
        ):
            extension = after.get("extension_reference") or {}
            if extension.get("sha256") != before["sha256"] or not same_location(
                extension.get("path", ""), before["path"]
            ):
                raise ValueError("boundary extension does not reference its frozen original slice")
        elif any(after[field] != before[field] for field in ("sha256", "size_bytes")):
            raise ValueError(f"boundary input bytes differ from the frozen reference: {key}")


def register_parent_inputs(catalog_root, *, tracer, mock_id, config):
    """Register configured existing files without downloading or authenticating output chains."""
    if type(mock_id) is not int or mock_id not in config["analysis"]["mock_ids"]:
        raise ValueError("input registration requires a configured integer realization ID")
    root = resolve_saved_path(catalog_root)
    records = []
    for cap in ("NGC", "SGC"):
        members = [("data", None)] + [
            ("random", index)
            for index in range(int(config["tracers"][tracer]["catalogue"]["random_splits"]))
        ]
        for role, split in members:
            path = planned_catalogue_path(
                root, mock_id, tracer, role, cap, random_split=split, config=config
            )
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(f"required {tracer} mock {mock_id} {cap} {role}: {path}")
            records.append(
                {
                    "path": str(path.resolve()),
                    "tracer": tracer,
                    "mock_id": mock_id,
                    "cap": cap,
                    "role": role,
                    "random_split": split,
                    "size_bytes": path.stat().st_size,
                    "source_url": catalogue_url(
                        mock_id, tracer, role, cap, random_split=split, config=config
                    ),
                }
            )
        reservoir = reservoir_slice_path(root, mock_id, tracer, config)
        if not reservoir.is_file():
            raise FileNotFoundError(f"required same-tracer boundary reservoir: {reservoir}")
        records.append(
            {
                "path": str(reservoir.resolve()),
                "tracer": tracer,
                "mock_id": mock_id,
                "cap": cap,
                "role": "reservoir",
                "random_split": None,
                "size_bytes": reservoir.stat().st_size,
            }
        )
    reference = config["tracers"][tracer]["source_reference"]
    manifest = root / reference["checksum_manifest_template"].format(mock_id=mock_id)
    publisher = _publisher_checksums(manifest) if manifest.is_file() else None
    # Hash each raw input once, including the reservoir shared by both caps. Supplied publisher
    # manifests must cover every configured source; absent manifests are recorded explicitly.
    verified = {}
    for record in records:
        path = Path(record["path"])
        expected = []
        if record["role"] in {"data", "random"} and publisher is not None:
            if path.name not in publisher:
                raise ValueError(f"publisher checksum manifest omits required source: {path}")
            expected.append(publisher[path.name])
            record["publisher_checksum_manifest"] = str(manifest.resolve())
        checksum = path.with_name(path.name + ".sha256")
        if checksum.is_file():
            tokens = checksum.read_text().split()
            if (
                not tokens
                or len(tokens[0]) != 64
                or any(character not in "0123456789abcdef" for character in tokens[0].lower())
            ):
                raise ValueError(f"publisher checksum mismatch: {path}")
            expected.append(tokens[0].lower())
        if path not in verified:
            before = _source_file_metadata(path)
            digest = file_sha256(path)
            if before != _source_file_metadata(path):
                raise ValueError(f"input file changed during registration: {path}")
            verified[path] = {**before, "sha256": digest}
        identity = verified[path]
        if any(value != identity["sha256"] for value in expected):
            raise ValueError(f"publisher checksum mismatch: {path}")
        record.update(identity)
        if expected:
            record["publisher_checksum_verified"] = identity["sha256"]
        else:
            record["publisher_checksum_unavailable_reason"] = (
                "no local publisher checksum supplied; registered digest identifies local bytes"
            )
        experiment = config["selection"]["boundary_support"].get("experiment")
        if (
            experiment is not None
            and experiment["mode"] == "extend"
            and experiment["tracer"] == tracer
            and record["role"] == "reservoir"
        ):
            record["extension_reference"] = read_reservoir_provenance(path).get(
                "extension_reference"
            )
    return records


def read_registered_catalogue(record, *, tracer, mock_id, cap, role, config, parallel=None):
    """Read a directly identified input and attach the unchanged object-bound scientific IDs."""
    expected = {"tracer": tracer, "mock_id": mock_id, "cap": cap, "role": role}
    if any(
        record.get(key) != value or type(record.get(key)) is not type(value)
        for key, value in expected.items()
    ):
        raise ValueError("input catalogue belongs to a different tracer, realization, cap or role")
    path = resolve_saved_path(record["path"])
    if not path.is_file() or path.stat().st_size != record["size_bytes"]:
        raise ValueError(f"input file is missing or its registered size changed: {path}")
    digest = record.get("sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("input catalogue lacks its registered byte digest")
    before = _source_file_metadata(path)
    if before != {key: record.get(key) for key in ("size_bytes", "mtime_ns", "ctime_ns")}:
        # Permission changes update ctime without changing catalogue bytes. Keep ctime in
        # the fast path so same-size edits with a restored mtime still trigger verification.
        current_digest = file_sha256(path)
        if before != _source_file_metadata(path):
            raise ValueError(f"input file changed during byte verification: {path}")
        if current_digest != digest:
            raise ValueError(f"input file changed since registration: {path}")
    split = record.get("random_split")
    if (role == "random" and type(split) is not int) or (role != "random" and split is not None):
        raise ValueError("input catalogue has an invalid random_split identity")
    if role in {"data", "random"}:
        url = catalogue_url(mock_id, tracer, role, cap, random_split=split, config=config)
        if record.get("source_url") != url or path.name != url.rsplit("/", 1)[-1]:
            raise ValueError("input source differs from the configured catalogue")
    catalogue = read_fits_catalog(path)
    columns = config["tracers"][tracer]["catalogue"]
    required = columns["random_required_columns" if role == "random" else "data_required_columns"]
    validate_catalog(catalogue, required)
    if role in {"data", "reservoir"}:
        controls = config["selection"]["stable_randomness"]
        if {
            controls["source_redshift_column"],
            controls["observed_redshift_column"],
        } & catalogue.keys():
            raise ValueError("raw catalogue contains reserved derived-redshift columns")
        if {"IS_RESERVOIR", "IS_BOUNDARY_EXTENSION"} & catalogue.keys():
            raise ValueError("IS_RESERVOIR and IS_BOUNDARY_EXTENSION are reserved provenance flags")
    if role == "data":
        # Survey metadata (e.g. NX and WEIGHT_FKP) are not part of the declared
        # galaxy input schema and need not exist in an extracted parent reservoir.
        # Keep the raw file identity above; downstream weights are recomputed.
        catalogue = {name: catalogue[name] for name in required}
    if role == "reservoir":
        # Extraction provenance establishes same-tracer source semantics; no byte certificate
        # or downstream BAO outcome is needed to read this population.
        embedded = read_reservoir_provenance(path)
        if (
            embedded["tracer"] != tracer
            or embedded["mock_id"] != mock_id
            or embedded["cap"] != "GCCOMB"
        ):
            raise ValueError("reservoir native identity differs from its registered source")
        validate_reservoir_selection(embedded, tracer, mock_id, config)
        extension = validate_reservoir_extension(
            catalogue, embedded, config, return_added_mask=True
        )
        if extension is not None:
            catalogue["IS_BOUNDARY_EXTENSION"] = extension[1]
        catalogue = filter_galactic_cap(catalogue, cap)
    return assign_source_identity(
        catalogue,
        tracer=tracer,
        mock_id=mock_id,
        cap=cap,
        role=role,
        random_split=split,
        column=config["selection"]["stable_randomness"]["identity_column"],
        config=config,
        parallel=parallel,
    )


def compare_source_redshifts(source, data_caps, *, config):
    """Describe the optional TARGETID join of complete-data Z against parent RSDZ.

    Identical finite duplicate source rows collapse to one identity. Conflicting duplicates,
    missing identities and nonfinite values remain separately counted; no row is repaired and
    no acceptance flag is produced. Catalogue identities are not exported.
    """
    if set(source) != {"TARGETID", "RSDZ"} or set(data_caps) != {"NGC", "SGC"}:
        raise ValueError("source comparison requires exactly the redshift columns and both caps")
    tolerance = config["analysis"]["source_redshift_comparison"]["absolute_tolerance"]

    def columns(catalogue, redshift):
        ids, z = np.asarray(catalogue["TARGETID"]), np.asarray(catalogue[redshift], dtype=float)
        if ids.ndim != 1 or z.shape != ids.shape or ids.dtype.kind not in "iu":
            raise ValueError(
                "source comparison requires equal one-dimensional integer-ID/redshift arrays"
            )
        return ids, z

    source_ids, source_z = columns(source, "RSDZ")
    order = np.argsort(source_ids, kind="stable")
    unique, starts, counts = np.unique(source_ids[order], return_index=True, return_counts=True)
    sorted_z = source_z[order]
    if len(unique):
        finite = np.logical_and.reduceat(np.isfinite(sorted_z), starts)
        minimum = np.minimum.reduceat(sorted_z, starts)
        maximum = np.maximum.reduceat(sorted_z, starts)
        conflicting = finite & (minimum != maximum)
        usable_z = np.where(finite & ~conflicting, minimum, np.nan)
    else:
        finite, conflicting, usable_z = np.empty(0, bool), np.empty(0, bool), np.empty(0)
    result = {
        "source_row_count": len(source_ids),
        "source_unique_identity_count": len(unique),
        "duplicate_group_count": int(np.count_nonzero(counts > 1)),
        "duplicate_row_count": int(counts[counts > 1].sum()),
        "conflicting_duplicate_count": int(conflicting.sum()),
        "nonfinite_source_group_count": int((~finite).sum()),
        "caps": {},
    }
    for cap, catalogue in data_caps.items():
        if set(catalogue) != {"TARGETID", "Z"}:
            raise ValueError("source comparison reads only complete-data identities and Z")
        ids, z = columns(catalogue, "Z")
        locations = np.searchsorted(unique, ids)
        matched = locations < len(unique)
        matched[matched] &= unique[locations[matched]] == ids[matched]
        reference = np.full(len(ids), np.nan)
        reference[matched] = usable_z[locations[matched]]
        finite_pair = matched & np.isfinite(z) & np.isfinite(reference)
        differences = np.abs(z[finite_pair] - reference[finite_pair])
        _, multiplicity = np.unique(ids, return_counts=True)
        result["caps"][cap] = {
            "data_row_count": len(ids),
            "joined_row_count": int(matched.sum()),
            "coverage_fraction": float(matched.mean()) if len(ids) else None,
            "missing_identity_count": int((~matched).sum()),
            "duplicated_data_identity_count": int(np.count_nonzero(multiplicity > 1)),
            "nonfinite_comparison_count": int(np.count_nonzero(matched & ~finite_pair)),
            "mismatching_identity_count": int(np.count_nonzero(differences > tolerance)),
            "maximum_absolute_difference": float(differences.max()) if len(differences) else None,
            "absolute_tolerance": tolerance,
        }
    return result


def footprint_from_random_counts(counts_by_cap, *, tracer, config):
    """Report cap area from complete random split counts, before any scientific selection."""
    controls = config["tracers"][tracer]["footprint"]
    estimator = controls["estimator"]
    if set(counts_by_cap) != {"NGC", "SGC"}:
        raise ValueError("footprint comparison requires both caps")
    result = {}
    for cap, raw in counts_by_cap.items():
        counts = np.asarray(raw)
        if (
            counts.shape != (len(config["analysis"]["mock_ids"]), len(estimator["split_ids"]))
            or counts.dtype.kind not in "iu"
            or np.any(counts <= 0)
        ):
            raise ValueError("footprint counts require every integer mock/split count")
        sums = [sum(map(int, row)) for row in counts]
        areas = (
            np.asarray(sums, dtype=float) / counts.shape[1] / estimator["rows_per_square_degree"]
        )
        result[cap] = {
            "area_deg2_by_mock": areas.tolist(),
            "split_sum_identical_across_mocks": len(set(sums)) == 1,
            "configured_area_deg2": controls["area_deg2"][cap],
            "area_difference_deg2_by_mock": (areas - controls["area_deg2"][cap]).tolist(),
        }
    return result


def _catalog_weight(catalog: Mapping[str, Any], *, include_fkp: bool) -> np.ndarray:
    columns = ("WEIGHT", "WEIGHT_FKP") if include_fkp else ("WEIGHT",)
    validate_catalog(catalog, columns)
    result = np.asarray(catalog["WEIGHT"], dtype=float)
    if include_fkp:
        result = result * np.asarray(catalog["WEIGHT_FKP"], dtype=float)
    if np.any(~np.isfinite(result)) or np.any(result < 0):
        raise ValueError("catalogue weights must be finite and non-negative")
    return result


def reconstruction_weight(catalog: Mapping[str, Any], expression: str) -> np.ndarray:
    """Return selection/completeness weights for displacement estimation."""
    if expression != "WEIGHT":
        raise ValueError("reconstruction weight expression must be WEIGHT")
    return _catalog_weight(catalog, include_fkp=False)


def measurement_weight(catalog: Mapping[str, Any], expression: str) -> np.ndarray:
    """Return selection times FKP weights for a two-point estimator."""
    if expression != "WEIGHT*WEIGHT_FKP":
        raise ValueError("two-point weight expression must be WEIGHT*WEIGHT_FKP")
    return _catalog_weight(catalog, include_fkp=True)


def size_box_from_extent(
    minimum_corner: Sequence[float],
    maximum_corner: Sequence[float],
    *,
    cell_size_mpc_h: float,
    box_padding: float,
) -> dict:
    """Size one mesh around a position envelope: pad, round up to whole cells, centre.

    The arithmetic lives here once because two boxes need it -- the reconstruction family mesh and
    the secondary power-spectrum box -- and a second copy is how their rounding quietly diverges.
    The mesh is forced even on every axis, which is what the FFT the mesh feeds expects.

    The centre is the midpoint of the envelope rather than of the padded box, so the padding is
    symmetric about the sample and a box that is correctly sized cannot be wrongly centred.
    """
    low = np.asarray(minimum_corner, dtype=float)
    high = np.asarray(maximum_corner, dtype=float)
    if low.shape != (3,) or high.shape != (3,):
        raise ValueError("box extent requires one minimum and one maximum corner in three axes")
    if np.any(~np.isfinite(low)) or np.any(~np.isfinite(high)):
        raise ValueError("box extent corners must be finite")
    extent = high - low
    if np.any(extent <= 0):
        raise ValueError("box extent must be positive on every axis")
    cell = float(cell_size_mpc_h)
    padding = float(box_padding)
    if not np.isfinite(cell) or cell <= 0 or not np.isfinite(padding) or padding <= 1.0:
        raise ValueError("box sizing requires a positive cell size and a padding above one")
    nmesh = np.ceil(extent * padding / cell).astype(np.int64)
    nmesh += nmesh % 2
    return {
        "nmesh": nmesh.tolist(),
        "boxsize_mpc_h": (nmesh.astype(float) * cell).tolist(),
        "boxcenter_mpc_h": (0.5 * (low + high)).tolist(),
        "cell_size_mpc_h": cell,
        "box_padding": padding,
    }


def _read_parent(records, *, tracer, mock_id, cap, config, include_randoms=True, parallel=None):
    selected = [
        record for record in records if record["mock_id"] == mock_id and record["cap"] == cap
    ]
    result = {}
    for role in ("data", "reservoir", "random") if include_randoms else ("data", "reservoir"):
        sources = [record for record in selected if record["role"] == role]
        expected = (
            int(config["tracers"][tracer]["catalogue"]["random_splits"]) if role == "random" else 1
        )
        if len(sources) != expected:
            raise ValueError(f"incomplete {role} inputs for {tracer} {mock_id} {cap}")
        blocks = [
            read_registered_catalogue(
                record,
                tracer=tracer,
                mock_id=mock_id,
                cap=cap,
                role=role,
                config=config,
                parallel=parallel,
            )
            for record in sources
        ]
        result[role] = concatenate_catalogs(blocks) if len(blocks) > 1 else blocks[0]
    return result


def catalogue_identity(run, *, tracer, condition, mock_id, cap, role, stage):
    """Direct coordinates of a saved catalogue; no output checksum chains."""
    return {
        **{
            key: run["metadata"][key]
            for key in ("run_name", "code_commit", "analysis_config_sha256")
        },
        "tracer": tracer,
        "condition": condition,
        "realization_id": mock_id,
        "cap": cap,
        "role": role,
        "stage": stage,
        "units": "Mpc/h",
    }


def save_catalogue(destination, name, catalogue, *, identity):
    """Save all columns without pickle, including unshifted and shifted coordinates."""
    destination = Path(destination)
    validate_catalog(catalogue, required=("SOURCE_ID", "X", "Y", "Z_CART"))
    arrays = {key: np.asarray(value) for key, value in catalogue.items()}
    for key, values in arrays.items():
        if values.dtype.hasobject:
            raise ValueError(f"catalogue column {key} cannot require pickle")
        if values.dtype.kind in "fc" and not np.all(np.isfinite(values)):
            raise ValueError(f"catalogue column {key} is non-finite")
    ids = arrays["SOURCE_ID"]
    if ids.dtype.kind not in "iu" or len(np.unique(ids)) != len(ids):
        raise ValueError("catalogue SOURCE_ID must contain unique integer identities")
    path = destination / name
    if path.parent.resolve() != destination.resolve() or path.suffix != ".npz":
        raise ValueError("catalogue name must be a local .npz filename")
    np.savez(path, **arrays)
    metadata = {
        "identity": identity,
        "columns": {
            key: {"dtype": value.dtype.str, "shape": list(value.shape)}
            for key, value in arrays.items()
        },
    }
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2, allow_nan=False))
    return [
        {"path": name, "format": "catalogue", "identity": identity},
        {"path": path.with_suffix(".json").name, "format": "json"},
    ]


def catalogue_partition(tracer, bin_name, role, config):
    """The original-redshift partition shared by selection, shifts and measurement."""
    item = measurement_bin(tracer, bin_name, config)
    return {
        "tracer": tracer,
        "measurement_bin": bin_name,
        "redshift_column": (
            config["selection"]["stable_randomness"]["observed_redshift_column"]
            if role == "data"
            else "Z"
        ),
        "zmin": item["zmin"],
        "zmax": item["zmax"],
    }


def _partition_rows(catalogue, partition):
    if partition is None:
        return slice(None)
    if set(partition) != {"tracer", "measurement_bin", "redshift_column", "zmin", "zmax"}:
        raise ValueError("invalid catalogue partition")
    zmin, zmax = partition["zmin"], partition["zmax"]
    if not np.isfinite([zmin, zmax]).all() or zmin >= zmax:
        raise ValueError("invalid catalogue redshift bounds")
    redshift = np.asarray(catalogue[partition["redshift_column"]])
    return (redshift >= zmin) & (redshift < zmax)


SHIFT_ROUNDTRIP_ATOL_MPC_H = 1.0e-4


def _source_id_digest(ids):
    ids = np.ascontiguousarray(ids)
    return {
        "dtype": ids.dtype.str,
        "count": int(ids.size),
        "sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
    }


def save_catalogue_reference(destination, name, *, identity, base, partition=None, shifted=None):
    """Reference an immutable selection, optionally storing only the reconstruction shift.

    Base identity is checked on every read. The partition fixes the entire intended subset
    and order; accepting an arbitrary ID subset would hide missing target rows. A displaced
    shard stores the float32 shift beside a digest of the ordered base SOURCE_ID column.
    """
    from .config import same_identity

    original = read_catalogue(base["path"], identity=base["identity"], partition=partition)
    if partition is not None and partition["tracer"] != identity["tracer"]:
        raise ValueError("catalogue partition belongs to another tracer")
    if base["identity"]["stage"] != "select":
        raise ValueError("catalogue references require an immutable selected base")
    coordinates = ("tracer", "condition", "realization_id", "cap", "role", "units")
    if any(not same_identity(identity[k], base["identity"][k]) for k in coordinates):
        raise ValueError("catalogue reference changes the base population")
    path = Path(destination) / name
    if path.parent.resolve() != Path(destination).resolve():
        raise ValueError("catalogue reference must be local to its unit")
    metadata = {
        "representation": "selected_reference_v1" if shifted is None else "displaced_reference_v2",
        "identity": identity,
        "base": {
            "path": str(resolve_saved_path(base["path"]).resolve()),
            "identity": base["identity"],
        },
        "partition": partition,
    }
    if shifted is None:
        if path.suffix != ".json" or identity["stage"] != "select" or partition is not None:
            raise ValueError("selection aliases require a complete selected catalogue and .json")
        metadata["columns"] = {
            key: {"dtype": value.dtype.str, "shape": list(value.shape)}
            for key, value in original.items()
        }
        path.write_text(json.dumps(metadata, indent=2, allow_nan=False))
        return [{"path": name, "format": "catalogue", "identity": identity}]
    columns = ("X_SHIFTED", "Y_SHIFTED", "Z_CART_SHIFTED")
    if identity["stage"] != "reconstruct" or path.suffix != ".npz":
        raise ValueError("displaced catalogues require a reconstruction shard")
    if set(shifted) != set(original) | set(columns):
        raise ValueError("displaced catalogue changes logical columns")
    for key, values in original.items():
        if shifted[key].dtype != values.dtype or not np.array_equal(shifted[key], values):
            raise ValueError("displaced catalogue changes original values, dtype or order")
    moved = np.column_stack([np.asarray(shifted[key]) for key in columns])
    if (
        moved.shape != (len(original["SOURCE_ID"]), 3)
        or moved.dtype.kind != "f"
        or not np.isfinite(moved).all()
    ):
        raise ValueError("invalid displaced coordinate shape, dtype or values")
    start = np.column_stack([np.asarray(original[key]) for key in ("X", "Y", "Z_CART")])
    shift = (moved - start).astype(np.float32)
    if np.max(np.abs(start + shift.astype(np.float64) - moved), initial=0.0) > (
        SHIFT_ROUNDTRIP_ATOL_MPC_H
    ):
        raise ValueError("float32 shift does not reproduce the shifted coordinates")
    metadata["source_ids"] = _source_id_digest(original["SOURCE_ID"])
    metadata["shift_roundtrip_atol_mpc_h"] = SHIFT_ROUNDTRIP_ATOL_MPC_H
    metadata["columns"] = {"SHIFT": {"dtype": shift.dtype.str, "shape": list(shift.shape)}}
    np.savez(path, SHIFT=shift)
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2, allow_nan=False))
    return [
        {"path": name, "format": "catalogue", "identity": identity},
        {"path": path.with_suffix(".json").name, "format": "json"},
    ]


def read_catalogue(path, *, identity, partition=None, _visited=()):
    """Reject mismatched catalogue coordinates, schema or row identities at the handoff."""
    from .config import same_identity

    path = resolve_saved_path(path).resolve()
    if path in _visited:
        raise ValueError("cyclic catalogue reference")
    metadata = json.loads(path.with_suffix(".json").read_text())
    if not same_identity(metadata.get("identity"), identity):
        raise ValueError("saved catalogue belongs to a different stage, unit or snapshot")
    representation = metadata.get("representation")
    if representation is not None:
        if representation not in {
            "selected_reference_v1",
            "displaced_reference_v1",
            "displaced_reference_v2",
        }:
            raise ValueError("unsupported catalogue representation")
        base = metadata["base"]
        if base["identity"].get("stage") != "select" or any(
            not same_identity(identity.get(key), base["identity"].get(key))
            for key in ("tracer", "condition", "realization_id", "cap", "role", "units")
        ):
            raise ValueError("catalogue reference changes the base population")
        stored_partition = metadata.get("partition")
        if stored_partition is not None and stored_partition["tracer"] != identity["tracer"]:
            raise ValueError("catalogue partition belongs to another tracer")
        if representation == "selected_reference_v1":
            if stored_partition is not None or identity["stage"] != "select":
                raise ValueError("invalid selected catalogue reference")
            base_metadata = json.loads(
                resolve_saved_path(base["path"]).with_suffix(".json").read_text()
            )
            if base_metadata.get("columns") != metadata["columns"]:
                raise ValueError("selected base schema changed")
            return read_catalogue(
                base["path"],
                identity=base["identity"],
                partition=partition,
                _visited=(*_visited, path),
            )
        if identity["stage"] != "reconstruct":
            raise ValueError("displaced reference must belong to reconstruction")
        if representation == "displaced_reference_v2":
            return _read_displaced_v2(path, metadata, partition, _visited)
        if stored_partition is None:
            raise ValueError("displaced reference lacks a reconstruction partition")
        if partition is not None and not same_identity(stored_partition, partition):
            raise ValueError("displaced shard belongs to another partition")
        original = read_catalogue(
            base["path"],
            identity=base["identity"],
            partition=stored_partition,
            _visited=(*_visited, path),
        )
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        columns = {"SOURCE_ID", "X_SHIFTED", "Y_SHIFTED", "Z_CART_SHIFTED"}
        schema = {
            key: {"dtype": value.dtype.str, "shape": list(value.shape)}
            for key, value in arrays.items()
        }
        if set(arrays) != columns or schema != metadata["columns"]:
            raise ValueError("displaced shard schema changed")
        if arrays["SOURCE_ID"].dtype != original["SOURCE_ID"].dtype or not np.array_equal(
            arrays["SOURCE_ID"], original["SOURCE_ID"]
        ):
            raise ValueError(
                "displaced shard has missing, reordered or mismatched source identities"
            )
        if any(
            value.shape != original["SOURCE_ID"].shape or not np.isfinite(value).all()
            for value in arrays.values()
        ):
            raise ValueError("invalid displaced shard shape or values")
        if any(arrays[key].dtype.kind != "f" for key in columns - {"SOURCE_ID"}):
            raise ValueError("displaced coordinates must be floating point")
        return {**original, **arrays}
    with np.load(path, allow_pickle=False) as archive:
        # Read at most one full parent column while materializing a bin. NPZ itself does
        # not support row slicing, but no full parent catalogue is retained here.
        rows = _partition_rows(archive, partition)
        arrays, stored_schema = {}, {}
        for key in archive.files:
            value = archive[key]
            stored_schema[key] = {"dtype": value.dtype.str, "shape": list(value.shape)}
            arrays[key] = value[rows]
    schema = {
        key: {"dtype": value.dtype.str, "shape": list(value.shape)} for key, value in arrays.items()
    }
    if stored_schema != metadata.get("columns"):
        raise ValueError("saved catalogue schema changed")
    validate_catalog(arrays, required=("SOURCE_ID", "X", "Y", "Z_CART"))
    if any(
        value.dtype.kind in "fc" and not np.all(np.isfinite(value)) for value in arrays.values()
    ):
        raise ValueError("saved catalogue contains non-finite values")
    ids = arrays["SOURCE_ID"]
    if not len(ids) or ids.dtype.kind not in "iu" or len(np.unique(ids)) != len(ids):
        raise ValueError("saved catalogue has invalid or repeated source identities")
    return arrays


def _read_displaced_v2(path, metadata, partition, visited):
    """Rebuild shifted coordinates from the verified base rows and the stored float32 shift."""
    from .config import same_identity

    stored = metadata.get("partition")
    if stored is not None and partition is not None and not same_identity(stored, partition):
        raise ValueError("displaced shard belongs to another partition")
    original = read_catalogue(
        metadata["base"]["path"],
        identity=metadata["base"]["identity"],
        partition=stored,
        _visited=(*visited, path),
    )
    if _source_id_digest(original["SOURCE_ID"]) != metadata.get("source_ids"):
        raise ValueError("displaced shard has missing, reordered or mismatched source identities")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"SHIFT"}:
            raise ValueError("displaced shard schema changed")
        shift = archive["SHIFT"]
    if {"SHIFT": {"dtype": shift.dtype.str, "shape": list(shift.shape)}} != metadata["columns"]:
        raise ValueError("displaced shard schema changed")
    if shift.shape != (len(original["SOURCE_ID"]), 3) or not np.isfinite(shift).all():
        raise ValueError("invalid displaced shard shape or values")
    result = dict(original)
    for axis, (column, key) in enumerate(
        (("X", "X_SHIFTED"), ("Y", "Y_SHIFTED"), ("Z_CART", "Z_CART_SHIFTED"))
    ):
        result[key] = np.asarray(original[column], dtype=np.float64) + shift[:, axis].astype(
            np.float64
        )
    if stored is None and partition is not None:
        rows = _partition_rows(result, partition)
        result = {key: value[rows] for key, value in result.items()}
    return result


def download_catalogue(url, destination, *, expected_sha256=None):
    """Validate a temporary FITS file before create-only publication of an explicit download."""
    import os
    import tempfile
    from urllib.request import urlopen

    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"refusing to replace existing source: {destination}")
    checksums = [] if expected_sha256 is None else [expected_sha256.lower()]
    sidecar = destination.with_name(destination.name + ".sha256")
    if sidecar.is_file():
        tokens = sidecar.read_text().split()
        if (
            not tokens
            or len(tokens[0]) != 64
            or any(character not in "0123456789abcdef" for character in tokens[0].lower())
        ):
            raise ValueError(f"invalid publisher checksum: {sidecar}")
        checksums.append(tokens[0].lower())
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=destination.name + ".",
            suffix=".partial",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            with urlopen(url, timeout=60) as response:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    stream.write(block)
        if validate_catalog(read_fits_catalog(temporary)) == 0:
            raise ValueError("downloaded catalogue is empty")
        digest = file_sha256(temporary)
        if any(digest != expected for expected in checksums):
            raise ValueError("download differs from the supplied publisher checksum")
        os.link(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def catalog_stage(run, unit, destination, *, download=False, cpus=1):
    """Register an ensemble with one reusable pool for unchanged identity kernels."""
    if type(cpus) is not int or cpus < 1:
        raise ValueError("catalog registration requires a positive CPU count")
    if cpus == 1:
        return _catalog_stage(run, unit, destination, download=download)
    from joblib import Parallel, parallel_config

    with parallel_config(backend="loky", inner_max_num_threads=1):
        with Parallel(
            n_jobs=cpus,
            return_as="generator",
            batch_size=1,
            pre_dispatch="2*n_jobs",
            max_nbytes=None,
        ) as parallel:
            return _catalog_stage(run, unit, destination, download=download, parallel=parallel)


def _catalog_stage(run, unit, destination, *, download=False, parallel=None):
    """Register a complete tracer ensemble; network access is opt-in."""
    config, tracer = run["config"], unit["tracer"]
    identifiers = list(config["analysis"]["mock_ids"])
    if list(unit["realization_ids"]) != identifiers:
        raise ValueError("catalog registration requires the complete configured ensemble")
    root = resolve_saved_path(run["metadata"]["catalog_root"])
    sources = []

    def progress(phase, **details):
        print(
            json.dumps({"stage": "catalog", "tracer": tracer, "phase": phase, **details}),
            flush=True,
        )

    for mock_id in identifiers:
        started = time.monotonic()
        progress("registration_started", mock_id=mock_id)
        reservoir = reservoir_slice_path(root, mock_id, tracer, config)
        if not reservoir.is_file():
            raise FileNotFoundError(f"required prepared same-tracer reservoir: {reservoir}")
        reference = config["tracers"][tracer]["source_reference"]
        manifest = root / reference["checksum_manifest_template"].format(mock_id=mock_id)
        checksums = _publisher_checksums(manifest) if manifest.is_file() else None
        for cap in ("NGC", "SGC"):
            members = [("data", None)] + [
                ("random", index)
                for index in range(int(config["tracers"][tracer]["catalogue"]["random_splits"]))
            ]
            for role, split in members:
                path = planned_catalogue_path(
                    root,
                    mock_id,
                    tracer,
                    role,
                    cap,
                    random_split=split,
                    config=config,
                )
                if not path.exists() and download:
                    if checksums is not None and path.name not in checksums:
                        raise ValueError(f"publisher manifest omits required source: {path}")
                    download_catalogue(
                        catalogue_url(
                            mock_id, tracer, role, cap, random_split=split, config=config
                        ),
                        path,
                        expected_sha256=(checksums or {}).get(path.name),
                    )
        registered = register_parent_inputs(root, tracer=tracer, mock_id=mock_id, config=config)
        progress("registration_complete", mock_id=mock_id, seconds=time.monotonic() - started)
        for record in registered:
            details = {
                "mock_id": mock_id,
                "cap": record["cap"],
                "role": record["role"],
                "random_split": record["random_split"],
            }
            started = time.monotonic()
            progress("validation_started", **details)
            read_registered_catalogue(
                record,
                tracer=tracer,
                mock_id=mock_id,
                cap=record["cap"],
                role=record["role"],
                config=config,
                parallel=parallel,
            )
            progress("validation_complete", seconds=time.monotonic() - started, **details)
        sources.extend(registered)
    payload = {"tracer": tracer, "realization_ids": identifiers, "sources": sources}
    (Path(destination) / "catalog.json").write_text(json.dumps(payload, indent=2, allow_nan=False))
    return {"products": [{"path": "catalog.json", "format": "json"}], **payload}


def read_catalog_registration(run, tracer):
    """Read only a published registration with matching run and full-ensemble identity."""
    import yaml
    from .config import same_identity

    directory = run["root"] / "catalog" / "units" / tracer
    result = yaml.safe_load((directory / "result.yaml").read_text())
    expected = {
        "unit_id": tracer,
        "tracer": tracer,
        "realization_ids": list(run["config"]["analysis"]["mock_ids"]),
    }
    if (
        result.get("stage") != "catalog"
        or not same_identity(result.get("unit"), expected)
        or any(
            result.get(key) != run["metadata"][key]
            for key in ("run_name", "code_commit", "analysis_config_sha256")
        )
    ):
        raise ValueError("catalog registration belongs to another run or tracer")
    registration = json.loads((directory / "catalog.json").read_text())
    if (
        registration["tracer"] != tracer
        or registration["realization_ids"] != expected["realization_ids"]
    ):
        raise ValueError("catalog registration does not cover the configured ensemble")
    return registration
