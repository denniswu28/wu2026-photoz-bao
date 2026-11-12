"""Native correlation functions and power spectra on saved selected or reconstructed catalogues."""

from __future__ import annotations

import json
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .catalog import validate_catalog, measurement_weight, size_box_from_extent
from .config import load_analysis_config, same_identity

from .reconstruction import SHIFTED_COLUMNS, _uniform_edges
from .selection import stable_measurement_subset, read_preparation


def combine_cap_power(cap_results: Mapping[str, Any]) -> Any:
    """Use pypower's sum of numerators and normalizations on the same cap grids."""
    if set(cap_results) != {"NGC", "SGC"}:
        raise ValueError("power combination requires exactly NGC and SGC")
    first = cap_results["NGC"]
    if type(first) is not type(cap_results["SGC"]):
        raise TypeError("cap power estimators have different types")
    other = cap_results["SGC"]
    if (
        first.ells != other.ells
        or len(first.edges) != len(other.edges)
        or any(not np.array_equal(a, b) for a, b in zip(first.edges, other.edges))
    ):
        raise ValueError("cap power multipoles or radial grids differ")
    if any(not np.isfinite(value.wnorm) or value.wnorm <= 0 for value in (first, other)):
        raise ValueError("cap power normalization must be finite and positive")
    return type(first).average(first, cap_results["SGC"])


def compute_power_spectrum_secondary(
    data: Mapping[str, Any],
    randoms: Mapping[str, Any],
    *,
    reconstructed: bool,
    geometry_artifact: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Any:
    """Measure required secondary P(k) with firstpoint LOS and ell=(0,2,4).

    These hard requirements are checked at the library boundary. Native edges
    and the sample-containing box come from a frozen geometry artifact;
    physical cell size, resampler, and interlacing remain configuration-bound;
    ``position_type='pos'`` is fixed in-file as for xi and reconstruction. The
    returned pypower statistic is write-only within the five controlling modules
    and never controls authoritative xi acceptance.
    """
    spec = config["measurement"]["power_spectrum_secondary"]
    if str(spec["line_of_sight"]) != "firstpoint" or tuple(
        int(value) for value in spec["multipoles"]
    ) != (0, 2, 4):
        raise ValueError("secondary P(k) requires firstpoint LOS and multipoles 0,2,4")
    if float(geometry_artifact.get("cell_size_mpc_h", float("nan"))) != float(
        spec["cell_size_mpc_h"]
    ):
        raise ValueError("P(k) geometry cell size differs from configuration")
    try:
        from pypower import CatalogFFTPower
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("pypower is required for the secondary P(k) product") from exc
    coordinate_names = (
        ("X_SHIFTED", "Y_SHIFTED", "Z_CART_SHIFTED") if reconstructed else ("X", "Y", "Z_CART")
    )
    data_positions = np.column_stack([data[name] for name in coordinate_names])
    original_randoms = np.column_stack([randoms[name] for name in ("X", "Y", "Z_CART")])
    shifted_randoms = (
        np.column_stack([randoms[name] for name in coordinate_names]) if reconstructed else None
    )
    expression = str(config["measurement"]["weights"]["power_spectrum_secondary"])
    data_weights = measurement_weight(data, expression)
    random_weights = measurement_weight(randoms, expression)
    # The box comes from a reviewed position envelope this route never reads, so no
    # configuration digest -- scoped or complete -- can establish that it contains *this*
    # sample. Check the property directly instead of hashing a proxy for it: positions outside
    # the box are silently wrapped or dropped by the FFT and the spectrum would be wrong with
    # nothing downstream detecting it.
    boxsize = np.asarray(geometry_artifact["boxsize_mpc_h"], dtype=float)
    boxcenter = np.asarray(geometry_artifact["boxcenter_mpc_h"], dtype=float)
    nmesh = np.asarray(geometry_artifact["nmesh"])
    if (
        boxsize.shape != (3,)
        or boxcenter.shape != (3,)
        or nmesh.shape != (3,)
        or nmesh.dtype.kind not in "iu"
        or np.any(nmesh <= 0)
        or np.any(nmesh % 2)
        or not np.all(np.isfinite([boxsize, boxcenter]))
        or np.any(boxsize <= 0)
        or not np.allclose(boxsize, nmesh * float(spec["cell_size_mpc_h"]), rtol=0, atol=1e-10)
    ):
        raise ValueError("P(k) geometry has incompatible box, mesh or cell size")
    for array in (data_positions, original_randoms, shifted_randoms):
        if array is not None and (
            array.ndim != 2
            or array.shape[1] != 3
            or not len(array)
            or not np.all(np.isfinite(array))
        ):
            raise ValueError("P(k) positions must be nonempty finite three-dimensional arrays")
    low, high = boxcenter - 0.5 * boxsize, boxcenter + 0.5 * boxsize
    # The mesh spans [centre - size/2, centre + size/2), so a position exactly on the upper face
    # maps to index nmesh and wraps to zero. The upper bound is therefore exclusive: admitting
    # equality would admit the one position this guard exists to catch.
    for name, array in (("data", data_positions), ("randoms", original_randoms)):
        if len(array) and (np.any(array.min(axis=0) < low) or np.any(array.max(axis=0) >= high)):
            raise ValueError(f"P(k) geometry box does not contain the {name} positions")
    if reconstructed and shifted_randoms is not None and len(shifted_randoms):
        if np.any(shifted_randoms.min(axis=0) < low) or np.any(shifted_randoms.max(axis=0) >= high):
            raise ValueError("P(k) geometry box does not contain the shifted random positions")

    k_edges = np.asarray(geometry_artifact["native_k_edges_h_mpc"], dtype=float)
    if (
        k_edges.ndim != 1
        or len(k_edges) < 2
        or not np.all(np.isfinite(k_edges))
        or np.any(np.diff(k_edges) <= 0)
    ):
        raise ValueError("P(k) geometry native edges are invalid")
    kwargs = {
        "data_positions1": data_positions,
        "data_weights1": data_weights,
        "randoms_positions1": original_randoms,
        "randoms_weights1": random_weights,
        "edges": k_edges,
        "ells": tuple(int(value) for value in spec["multipoles"]),
        "los": str(spec["line_of_sight"]),
        "nmesh": tuple(int(value) for value in geometry_artifact["nmesh"]),
        "boxsize": tuple(float(value) for value in geometry_artifact["boxsize_mpc_h"]),
        "boxcenter": tuple(float(value) for value in geometry_artifact["boxcenter_mpc_h"]),
        "resampler": str(spec["resampler"]),
        "interlacing": int(spec["interlacing"]),
        "position_type": "pos",
    }
    if reconstructed:
        kwargs.update(shifted_positions1=shifted_randoms, shifted_weights1=random_weights)
    return CatalogFFTPower(**kwargs).poles


__all__ = ["combine_cap_power", "compute_power_spectrum_secondary"]


def _direct_lattice_mode_counts(
    nmesh: np.ndarray, boxsize: np.ndarray, edges: np.ndarray, *, chunk_size: int = 1024
) -> list:
    """Count discrete Fourier lattice points in half-open radial shells."""
    axes = [
        2.0 * np.pi * np.fft.fftfreq(int(n), d=float(length) / int(n))
        for n, length in zip(nmesh, boxsize)
    ]
    maximum = float(edges[-1])
    axes = [values[np.abs(values) < maximum] for values in axes]
    qxy = (axes[0][:, None] ** 2 + axes[1][None, :] ** 2).reshape(-1)
    kz2 = axes[2] ** 2
    counts = np.zeros(len(edges) - 1, dtype=np.int64)
    for start in range(0, len(qxy), int(chunk_size)):
        radii = np.sqrt(qxy[start : start + int(chunk_size), None] + kz2[None, :])
        counts += np.histogram(radii, bins=edges)[0]
    return counts.tolist()


def derive_power_spectrum_geometry(
    positions: Sequence[Sequence[float]],
    *,
    tracer: str,
    config: Optional[Mapping[str, Any]] = None,
) -> dict:
    """Freeze a common causal-tier P(k) box, native binning, and mode guard."""
    resolved = config or load_analysis_config()
    spec = resolved["measurement"]["power_spectrum_secondary"]
    array = np.asarray(positions, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3 or len(array) == 0 or np.any(~np.isfinite(array)):
        raise ValueError("P(k) geometry positions must have finite shape (N, 3)")
    if tracer not in resolved["tracers"]:
        raise ValueError("P(k) geometry requires a configured tracer")
    cellsize = float(spec["cell_size_mpc_h"])
    # The same sizing rule the reconstruction family mesh uses, called rather than repeated.
    sized = size_box_from_extent(
        array.min(axis=0),
        array.max(axis=0),
        cell_size_mpc_h=cellsize,
        box_padding=float(spec["box_padding"]),
    )
    nmesh = np.asarray(sized["nmesh"], dtype=np.int64)
    boxsize = np.asarray(sized["boxsize_mpc_h"], dtype=float)
    boxcenter = np.asarray(sized["boxcenter_mpc_h"], dtype=float)
    fundamental = 2.0 * np.pi / boxsize
    maximum_fundamental = float(np.max(fundamental))
    comparison = spec["comparison_binning"]
    comparison_width = float(comparison["bin_width_h_mpc"])
    divisor = int(np.floor(comparison_width / maximum_fundamental))
    if divisor < 1:
        raise ValueError("P(k) box cannot support an exact native divisor of comparison bins")
    native_width = comparison_width / divisor
    comparison_edges = np.arange(
        float(comparison["k_min_h_mpc"]),
        float(comparison["k_max_h_mpc"]) + 0.5 * comparison_width,
        comparison_width,
    )
    native_edges = np.arange(
        0.0,
        float(spec["native_k_max_h_mpc"]) + 0.5 * native_width,
        native_width,
    )
    if not np.isclose(native_edges[-1], float(spec["native_k_max_h_mpc"]), rtol=0, atol=1e-12):
        raise ValueError("geometry-derived native P(k) width does not exactly cover its range")
    if not np.allclose(
        comparison_edges / native_width,
        np.rint(comparison_edges / native_width),
        rtol=0.0,
        atol=1e-10,
    ):
        raise ValueError("P(k) comparison edges do not exactly coarsen native bins")
    nyquist = np.pi / np.full(3, cellsize)
    if float(comparison_edges[-1]) >= float(np.min(nyquist)):
        raise ValueError("P(k) comparison maximum reaches or exceeds a Nyquist frequency")
    # Reported, never scored. The mode counts say how well each comparison shell is sampled and
    # a reader can see a thin one; there is no minimum, because the product they describe is a
    # nonbinding diagnostic and a criterion on it would decide something it cannot.
    mode_counts = _direct_lattice_mode_counts(nmesh, boxsize, comparison_edges)
    return {
        "schema_version": 2,
        "kind": "power_spectrum_geometry",
        "tracer": str(tracer),
        "box_policy": "common_sample_containing_causal_tiers",
        "box_padding": float(spec["box_padding"]),
        "boxcenter_mpc_h": boxcenter.tolist(),
        "boxsize_mpc_h": boxsize.tolist(),
        "nmesh": nmesh.tolist(),
        "cell_size_mpc_h": cellsize,
        "fundamental_modes_h_mpc": fundamental.tolist(),
        "nyquist_h_mpc": nyquist.tolist(),
        "native_bin_width_h_mpc": native_width,
        "native_k_edges_h_mpc": native_edges.tolist(),
        "comparison_k_edges_h_mpc": comparison_edges.tolist(),
        "comparison_lattice_mode_counts": mode_counts,
        "minimum_realized_lattice_modes": int(min(mode_counts)) if mode_counts else 0,
        "diagnostic_only": True,
    }


# One smoothing radius outside a straight survey edge, the Gaussian-smoothed random field
# has fallen to this fraction of the interior mean: it is the standard normal CDF at -1, and
# it carries no tuning. It is the floor that separates "cell the survey's randoms still reach"
# from "cell that holds nothing but the kernel's far tail" in the loss accounting below.


def compute_xi(
    data: Mapping[str, Any],
    randoms: Mapping[str, Any],
    *,
    reconstructed: bool,
    s_edges_mpc_h: Sequence[float],
    mu_bins: int,
    line_of_sight: str,
    nthreads: int,
    declared_estimator: str,
    weight_expression: str,
    precomputed_random_counts: Any = None,
    jackknife_regions: Optional[Mapping[str, Any]] = None,
) -> Any:
    """Measure pre Landy--Szalay or the post shifted-random estimator.

    ``jackknife_regions`` turns the same measurement into its jackknife form. It carries one
    region label per data row, one per random row, and the total number of regions in the survey.
    pycorr then returns a jackknife estimator: the same correlation function, plus the
    region-omitted realizations of it. The estimator, its edges, its weights and its normalization
    are untouched, which is the point -- a jackknife measured with different settings would
    calibrate a shot-noise rescaling for a correlation function nobody fits.

    **The region count is the whole survey's, not this cap's.** Regions are angular and the caps
    are disjoint on the sky, so a cap holds only some of them. Left to itself pycorr would build
    realizations for the regions it actually sees, and the two caps' estimators could then not be
    added -- their realization keys would differ, and the addition raises. Declaring the full
    range gives every cap an entry for every region, zero-counted where it has no objects, and the
    combined realization that omits a region is then exactly that region's own cap with the region
    removed plus the other cap entire. That is the definition a survey-wide jackknife needs, and
    it comes out of pycorr's own arithmetic rather than being assembled here.

    Post-reconstruction uses shifted data as ``D``, original randoms as ``R``,
    and shifted randoms as ``S``; both random sets retain identical weights.
    Both declared modes bind pycorr's explicit ``landyszalay`` implementation:
    with ``S`` present that pinned implementation is expected to evaluate the shifted-random
    ``(DD - DS - SD + SS) / RR`` form rather than the pre-reconstruction
    ``(DD - DR - RD + RR) / RR`` form. Pinned-library compatibility evidence,
    not this module, establishes those library semantics.
    ``declared_estimator`` is the configuration label written into provenance:
    it must be ``landy_szalay`` before reconstruction or ``shifted_random``
    afterward, so label agreement cannot drift away from this implementation.

    ``precomputed_random_counts`` supplies ``R1R2`` instead of counting it again. Both
    reconstruction states normalize by the *original* randoms, and reconstruction returns those
    positions and weights untouched -- it adds shifted columns rather than overwriting anything --
    so the two states' ``R1R2`` is the same count over the same rows with the same weights. It is
    the most expensive term in either measurement, since the randoms outnumber the galaxies fifty
    to one and pair counting goes as their square, and counting it twice buys nothing.

    This is an implementation identity and not a modelling choice: the estimator, its inputs and
    its arithmetic are unchanged, and the reused counts are required to be the ones this call
    would have produced.
    The estimator uses Corrfunc and uniform full-range ``mu`` edges from -1 to
    1; configuration controls their count, not their fixed physical extent.
    """
    if int(mu_bins) < 1:
        raise ValueError("xi requires at least one mu bin")
    if line_of_sight != "midpoint":
        raise ValueError("analysis xi requires midpoint line of sight")
    expected_label = "shifted_random" if reconstructed else "landy_szalay"
    if str(declared_estimator) != expected_label:
        raise ValueError(
            f"configured xi estimator label must be {expected_label!r} for this measurement state"
        )
    try:
        from pycorr import TwoPointCorrelationFunction
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("pycorr is required for analysis xi") from exc
    original_randoms = np.column_stack([randoms["X"], randoms["Y"], randoms["Z_CART"]])
    data_weights = measurement_weight(data, weight_expression)
    random_weights = measurement_weight(randoms, weight_expression)
    if reconstructed:
        validate_catalog(data, SHIFTED_COLUMNS)
        validate_catalog(randoms, SHIFTED_COLUMNS)
        data_positions = np.column_stack(
            [data["X_SHIFTED"], data["Y_SHIFTED"], data["Z_CART_SHIFTED"]]
        )
        shifted_positions = np.column_stack(
            [randoms["X_SHIFTED"], randoms["Y_SHIFTED"], randoms["Z_CART_SHIFTED"]]
        )
    else:
        data_positions = np.column_stack([data["X"], data["Y"], data["Z_CART"]])
        shifted_positions = None
    kwargs = {
        "data_positions1": data_positions,
        "data_weights1": data_weights,
        "randoms_positions1": original_randoms,
        "randoms_weights1": random_weights,
        "position_type": "pos",
        "los": line_of_sight,
        "engine": "corrfunc",
        "estimator": "landyszalay",
        "nthreads": int(nthreads),
    }
    if shifted_positions is not None:
        kwargs.update(shifted_positions1=shifted_positions, shifted_weights1=random_weights)
    if jackknife_regions is not None:
        data_regions = np.asarray(jackknife_regions["data"], dtype=np.int64)
        random_regions = np.asarray(jackknife_regions["randoms"], dtype=np.int64)
        region_count = int(jackknife_regions["region_count"])
        if data_regions.shape != (len(data_positions),):
            raise ValueError("jackknife data regions do not label every measured galaxy")
        if random_regions.shape != (len(original_randoms),):
            raise ValueError("jackknife random regions do not label every measured random")
        if data_regions.min() < 0 or random_regions.min() < 0:
            raise ValueError("jackknife regions must be non-negative labels")
        if max(int(data_regions.max()), int(random_regions.max())) >= region_count:
            raise ValueError("a jackknife region label lies outside the declared region count")
        kwargs.update(
            data_samples1=data_regions,
            randoms_samples1=random_regions,
            samples=np.arange(region_count, dtype=np.int64),
        )
        if shifted_positions is not None:
            # The shifted randoms are the same rows displaced, so they keep their labels. Giving
            # them their own angular assignment would let reconstruction move a random between
            # regions and make the omitted set differ between the two counters of one estimator.
            kwargs.update(shifted_samples1=random_regions)
    if precomputed_random_counts is not None:
        # A copy, because the estimator takes ownership of what it is handed and provenance is
        # later bound onto the counter itself. Sharing one object between the two products would
        # let the second binding overwrite the first, and the pre-reconstruction product would
        # then carry the post-reconstruction record. The copy duplicates an array of counts, not
        # the counting.
        kwargs["R1R2"] = precomputed_random_counts.deepcopy()
    return TwoPointCorrelationFunction(
        "smu",
        edges=(np.asarray(s_edges_mpc_h, dtype=float), np.linspace(-1.0, 1.0, int(mu_bins) + 1)),
        **kwargs,
    )


def combine_cap_pair_counts(cap_results: Mapping[str, Any]) -> Any:
    """Normalize each cap to its own DD, then sum all count types before projection."""
    if set(cap_results) != {"NGC", "SGC"}:
        raise ValueError("pair-count combination requires exactly NGC and SGC")
    north, south = [cap_results[cap] for cap in ("NGC", "SGC")]
    if type(north) is not type(south) or north.count_names != south.count_names:
        raise ValueError("caps have incompatible estimator or count types")
    if len(north.edges) != len(south.edges) or any(
        not np.array_equal(a, b) for a, b in zip(north.edges, south.edges)
    ):
        raise ValueError("caps have incompatible pair-count edges")
    normalized = []
    for estimator in (north, south):
        for name in estimator.count_names:
            counts = getattr(estimator, name)
            if (
                not np.all(np.isfinite(counts.wnorm))
                or np.any(np.asarray(counts.wnorm) <= 0)
                or not np.all(np.isfinite(counts.wcounts))
            ):
                raise ValueError(f"cap {name} has invalid counts or normalization")
        normalized.append(estimator.normalize(wnorm="D1D2"))
    # Distinct caps must both contribute even if synthetic count arrays happen to be identical.
    return type(north).sum(*normalized, uniques=False)


def bind_pair_count_identity(estimator, identity):
    """Pycorr serializes each counter's attrs; estimator-level ad hoc attrs are not saved."""
    for name in estimator.count_names:
        counter = getattr(estimator, name)
        counter.attrs = {**dict(counter.attrs), "identity": dict(identity), "units": "Mpc/h"}


def pair_count_identity(estimator):
    identities = [
        dict(getattr(estimator, name).attrs.get("identity", {})) for name in estimator.count_names
    ]
    if (
        not identities
        or not identities[0]
        or any(not same_identity(identity, identities[0]) for identity in identities)
    ):
        raise ValueError("native counters have absent or inconsistent scientific identities")
    if any(
        getattr(estimator, name).attrs.get("units") != "Mpc/h" for name in estimator.count_names
    ):
        raise ValueError("native counters have incompatible distance units")
    return identities[0]


# Optional scalar-calibration geometry; no calibration subgraph runs in the linear driver.


def assign_jackknife_regions(
    data_by_cap: Mapping[str, Mapping[str, Any]],
    randoms_by_cap: Mapping[str, Mapping[str, Any]],
    *,
    region_count: int,
    position_columns: Sequence[str],
) -> dict:
    """Label every galaxy and random with one of ``region_count`` angular jackknife regions.

    The construction is RascalC's own ``get_subsampler_xirunpc`` rather than a local one: angular
    k-means at a fixed Healpix resolution and a fixed centroid initialization, which is what makes
    the assignment deterministic without this module owning a seed. Reproducing the reference
    implementation is deliberate -- the shot-noise rescaling this feeds is only comparable with
    published values if the regions are built the same way.

    **One subsampler spans both caps.** The measured correlation function is cap-combined, so its
    jackknife realizations must omit one region of the union. Labelling each cap separately would
    give both a region 0, and adding the two estimators would merge two unrelated patches of sky
    into one realization -- a wrong jackknife covariance, and therefore a wrong rescaling, with
    nothing raised. Because the caps are disjoint on the sky, a single angular partition assigns
    each of them its own regions and the union is exactly the sixty.

    Regions are angular, so they do not depend on radial position and are identical before and
    after reconstruction. That is what lets one assignment serve both states of one realization:
    the two differ through the displacement and through nothing else.
    """
    if int(region_count) < 2:
        raise ValueError("a jackknife needs at least two regions")
    caps = sorted(data_by_cap)
    if not caps or caps != sorted(randoms_by_cap):
        raise ValueError("jackknife regions need the same caps for data and randoms")

    def _positions(catalog: Mapping[str, Any]) -> np.ndarray:
        validate_catalog(catalog, tuple(position_columns))
        return np.column_stack(
            [np.asarray(catalog[name], dtype=float) for name in position_columns]
        )

    from RascalC.pre_process.create_jackknives_pycorr import get_subsampler_xirunpc

    data_positions = {cap: _positions(data_by_cap[cap]) for cap in caps}
    union = np.vstack([data_positions[cap] for cap in caps])
    if len(union) < int(region_count):
        raise ValueError("fewer galaxies than jackknife regions")
    subsampler = get_subsampler_xirunpc(union, int(region_count), position_type="xyz")

    assigned: dict = {}
    occupancy = np.zeros(int(region_count), dtype=np.int64)
    for cap in caps:
        labels = {
            "data": np.asarray(subsampler.label(data_positions[cap].T, position_type="xyz")),
            "randoms": np.asarray(
                subsampler.label(_positions(randoms_by_cap[cap]).T, position_type="xyz")
            ),
        }
        for role, values in labels.items():
            if values.size and (values.min() < 0 or values.max() >= int(region_count)):
                raise ValueError(f"{cap} {role} received a label outside the region range")
        occupancy += np.bincount(labels["data"], minlength=int(region_count))
        assigned[cap] = {role: values.astype(np.int64) for role, values in labels.items()}
        assigned[cap]["region_count"] = int(region_count)

    per_cap_regions = {
        cap: sorted(int(value) for value in np.unique(assigned[cap]["data"])) for cap in caps
    }

    shared = set.intersection(*(set(value) for value in per_cap_regions.values()))
    if len(caps) > 1 and shared:
        raise ValueError(
            f"jackknife regions {sorted(shared)} span more than one cap, so a region-omitted "
            f"realization would remove two unrelated patches of sky"
        )
    empty = [index for index, count in enumerate(occupancy) if count == 0]
    return {
        "regions_by_cap": assigned,
        "region_count": int(region_count),
        "occupancy": occupancy.tolist(),
        "empty_regions": empty,
        "regions_by_cap_index": {cap: per_cap_regions[cap] for cap in caps},
    }


def cluster_stage(run, unit, destination, *, nthreads, statistics="both"):
    """Measure xi and/or P(k) from saved catalogues without changing selection or solving again."""
    from .selection import read_selected_parent
    from .reconstruction import read_reconstructed_parent
    from .config import per_bin_reconstruction

    if statistics not in {"xi", "pk", "both"}:
        raise ValueError("statistics must be xi, pk or both")
    config = run["config"]
    per_bin = per_bin_reconstruction(config)
    tracer, condition, mock_id = unit["tracer"], unit["condition"], unit["realization_id"]
    if "transfer_run" in config["measurement"]["reconstruction"] and statistics != "xi":
        raise ValueError("exclusive transfer requires xi-only measurement")
    preparation = read_preparation(run, tracer)
    xi = config["measurement"]["xi"]
    edges = _uniform_edges(xi["s_min_mpc_h"], xi["s_max_mpc_h"], xi["bin_width_mpc_h"], label="xi")
    controls = config["selection"]["stable_randomness"]
    weight_expression = config["measurement"]["weights"]["correlation_function"]
    states = config["analysis"]["workloads"][run["metadata"]["workload"]]["reconstruction_states"]
    all_counts, all_power, measurements, products, reconstruction_diagnostics = {}, {}, [], [], {}
    power_spec = config["measurement"]["power_spectrum_secondary"]
    if statistics == "pk" and not any(
        name in power_spec["diagnostic_measurement_bins"] for name in unit["measurement_bins"]
    ):
        raise ValueError("this unit has no configured P(k) measurement bins")
    for cap in ("NGC", "SGC"):
        if per_bin:
            data = randoms = None
        elif "postreconstruction" in states:
            data, randoms, diagnostics = read_reconstructed_parent(run, unit, cap)
            reconstruction_diagnostics[cap] = diagnostics
        else:
            data, randoms, _ = read_selected_parent(run, unit, cap)
        for name in unit["measurement_bins"]:
            if per_bin:
                if "postreconstruction" in states:
                    data, randoms, diagnostics = read_reconstructed_parent(
                        run, unit, cap, bin_name=name
                    )
                    reconstruction_diagnostics[cap] = diagnostics
                else:
                    data, randoms, _ = read_selected_parent(run, unit, cap, bin_name=name)
            galaxies = stable_measurement_subset(
                data,
                tracer=tracer,
                bin_name=name,
                redshift_column=controls["observed_redshift_column"],
                config=config,
            )
            random_bin = stable_measurement_subset(
                randoms,
                tracer=tracer,
                bin_name=name,
                redshift_column="Z",
                config=config,
            )
            random_path = f"{name}--{cap}--randoms.npz"
            arrays = {
                "positions": np.column_stack([random_bin[key] for key in ("X", "Y", "Z_CART")]),
                "weights": measurement_weight(random_bin, weight_expression),
                "source_ids": np.asarray(random_bin[controls["identity_column"]], dtype=np.uint64),
            }
            if all(key in random_bin for key in SHIFTED_COLUMNS):
                arrays["shifted_positions"] = np.column_stack(
                    [random_bin[key] for key in SHIFTED_COLUMNS]
                )
            if per_bin:
                random_path = measurement_random_reference(
                    run, unit, cap, name, reconstructed="postreconstruction" in states
                )
            else:
                np.savez(destination / random_path, **arrays)
                products.append({"path": random_path, "format": "npz", "arrays": list(arrays)})
            rr = None
            count_states = (
                list(dict.fromkeys(["prereconstruction", *states]))
                if statistics != "pk"
                else states
            )
            for state in count_states:
                post = state == "postreconstruction"
                estimator = None
                if statistics != "pk":
                    estimator = compute_xi(
                        galaxies,
                        random_bin,
                        reconstructed=post,
                        s_edges_mpc_h=edges,
                        mu_bins=xi["mu_bins"],
                        line_of_sight=xi["line_of_sight"],
                        nthreads=nthreads,
                        declared_estimator=xi["estimator_" + state],
                        weight_expression=weight_expression,
                        precomputed_random_counts=rr,
                    )
                    if not post:
                        rr = estimator.R1R2
                if state not in states:
                    continue
                identity = {
                    "tracer": tracer,
                    "condition": condition,
                    "measurement_bin": name,
                    "reconstruction_state": state,
                    "realization_id": mock_id,
                    "cap": cap,
                    "analysis_config_sha256": run["metadata"]["analysis_config_sha256"],
                }
                record = {
                    "identity": identity,
                    "randoms": random_path,
                    "units": "Mpc/h",
                    "data_count": len(galaxies[controls["identity_column"]]),
                    "random_count": len(random_bin[controls["identity_column"]]),
                    "coordinate_state": "shifted" if post else "unshifted",
                }
                if estimator is not None:
                    estimator = estimator.normalize(wnorm="D1D2")
                    bind_pair_count_identity(estimator, identity)
                    path = f"{name}--{state}--{cap}--counts.npy"
                    estimator.save(str(destination / path))
                    products.append({"path": path, "format": "pycorr", "identity": identity})
                    record["counts"] = path
                    all_counts.setdefault((name, state), {})[cap] = estimator
                if statistics != "xi" and name in power_spec["diagnostic_measurement_bins"]:
                    power = compute_power_spectrum_secondary(
                        galaxies,
                        random_bin,
                        reconstructed=post,
                        geometry_artifact=preparation["power_geometry"],
                        config=config,
                    )
                    all_power.setdefault((name, state), {})[cap] = power
                    path = f"{name}--{state}--{cap}--power.npz"
                    k, values = power(ell=power_spec["multipoles"], return_k=True, complex=False)
                    np.savez(destination / path, k=k, power=values, ells=power.ells)
                    products.append(
                        {"path": path, "format": "npz", "arrays": ["k", "power", "ells"]}
                    )
                    record["power"] = path
                measurements.append(record)
    for (name, state), counts in all_counts.items():
        combined = combine_cap_pair_counts(counts)
        identity = {**pair_count_identity(counts["NGC"]), "cap": "NGC+SGC"}
        bind_pair_count_identity(combined, identity)
        path = f"{name}--{state}--combined-counts.npy"
        combined.save(str(destination / path))
        products.append({"path": path, "format": "pycorr", "identity": identity})
        measurements.append({"identity": identity, "counts": path, "units": "Mpc/h"})
    for (name, state), powers in all_power.items():
        power = combine_cap_power(powers).copy()
        comparison = power_spec["comparison_binning"]
        power.select(
            (comparison["k_min_h_mpc"], comparison["k_max_h_mpc"], comparison["bin_width_h_mpc"])
        )
        k, values = power(ell=power_spec["multipoles"], return_k=True, complex=False)
        path = f"{name}--{state}--power.npz"
        np.savez(destination / path, k=k, power=values, ells=power.ells)
        products.append({"path": path, "format": "npz", "arrays": ["k", "power", "ells"]})
        matching = [
            record
            for record in measurements
            if record["identity"]["measurement_bin"] == name
            and record["identity"]["reconstruction_state"] == state
        ]
        combined_record = next(
            (record for record in matching if record["identity"]["cap"] == "NGC+SGC"), None
        )
        if combined_record is None:
            combined_record = {
                "identity": {**matching[0]["identity"], "cap": "NGC+SGC"},
                "units": "Mpc/h",
            }
            measurements.append(combined_record)
        combined_record["power"] = path
    (destination / "measurements.json").write_text(
        json.dumps(
            {
                "measurements": measurements,
                "diagnostics": reconstruction_diagnostics,
                "statistics": statistics,
            },
            indent=2,
            allow_nan=False,
        )
    )
    products.append({"path": "measurements.json", "format": "json"})
    return {"products": products, "statistics": statistics}


def measurement_random_reference(run, unit, cap, bin_name, *, reconstructed):
    """Identify the already validated selected/displaced randoms without copying arrays."""
    from .catalog import catalogue_identity, catalogue_partition
    from .selection import selected_catalogue_entry

    partition = catalogue_partition(unit["tracer"], bin_name, "random", run["config"])
    if reconstructed:
        directory = run["root"] / "reconstruct" / "units" / unit["unit_id"]
        payload = json.loads((directory / "reconstruction.json").read_text())
        identity = catalogue_identity(
            run,
            tracer=unit["tracer"],
            condition=unit["condition"],
            mock_id=unit["realization_id"],
            cap=cap,
            role="random",
            stage="reconstruct",
        )
        matches = [
            row
            for row in payload["catalogues"]
            if same_identity(row["identity"], identity)
            and ("partition" not in row or same_identity(row["partition"], partition))
        ]
        if len(matches) != 1:
            raise ValueError("missing or duplicate random catalogue reference")
        path = (directory / matches[0]["path"]).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError("random catalogue path leaves reconstruction unit")
        base = {"path": str(path), "identity": identity}
    else:
        base = selected_catalogue_entry(run, unit, cap, "random")
    return {
        "representation": "measurement_random_reference_v1",
        **base,
        "partition": partition,
        "weight_expression": run["config"]["measurement"]["weights"]["correlation_function"],
    }


def read_measurement_randoms(record, config):
    """One reader for native clustering/RascalC, including historical standalone archives."""
    from .catalog import catalogue_partition, read_catalogue

    reference = record["randoms"]
    if isinstance(reference, str):
        path = (record["directory"] / reference).resolve()
        if not path.is_relative_to(record["directory"].resolve()):
            raise ValueError("random product path leaves its measurement unit")
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
    else:
        if reference.get("representation") != "measurement_random_reference_v1":
            raise ValueError("unsupported measurement random reference")
        expected = record["identity"]
        identity = reference["identity"]
        if (
            identity.get("role") != "random"
            or identity.get("stage") not in {"select", "reconstruct"}
            or any(
                not same_identity(identity.get(key), expected[key])
                for key in (
                    "tracer",
                    "condition",
                    "realization_id",
                    "cap",
                    "analysis_config_sha256",
                )
            )
        ):
            raise ValueError("measurement random reference changes its target identity")
        partition = catalogue_partition(
            expected["tracer"], expected["measurement_bin"], "random", config
        )
        weights = config["measurement"]["weights"]["correlation_function"]
        if (
            not same_identity(reference.get("partition"), partition)
            or reference.get("weight_expression") != weights
        ):
            raise ValueError("measurement random partition or weights changed")
        catalogue = read_catalogue(reference["path"], identity=identity, partition=partition)
        arrays = {
            "positions": np.column_stack([catalogue[key] for key in ("X", "Y", "Z_CART")]),
            "weights": measurement_weight(catalogue, weights),
            "source_ids": catalogue["SOURCE_ID"],
        }
        if all(key in catalogue for key in SHIFTED_COLUMNS):
            arrays["shifted_positions"] = np.column_stack(
                [catalogue[key] for key in SHIFTED_COLUMNS]
            )
    count = record["random_count"]
    ids = arrays["source_ids"]
    if ids.shape != (count,) or ids.dtype.kind not in "iu" or len(np.unique(ids)) != count:
        raise ValueError("measurement randoms have missing or duplicate identities")
    for key, shape in (("positions", (count, 3)), ("weights", (count,))):
        if arrays[key].shape != shape or not np.isfinite(arrays[key]).all():
            raise ValueError("invalid measurement random coordinates or weights")
    if "shifted_positions" in arrays and (
        arrays["shifted_positions"].shape != (count, 3)
        or not np.isfinite(arrays["shifted_positions"]).all()
    ):
        raise ValueError("invalid shifted random coordinates")
    if record["coordinate_state"] == "shifted" and "shifted_positions" not in arrays:
        raise ValueError("post-reconstruction covariance requires shifted randoms")
    return arrays


def read_measurements(run, route, *, statistic="xi"):
    """Read the complete identified ensemble for one route, including both physical caps."""
    import yaml

    if statistic not in {"xi", "pk"}:
        raise ValueError("measurement reader statistic must be xi or pk")
    config = run["config"]
    identifiers = config["analysis"]["workloads"][run["metadata"]["workload"]]["mock_ids"]
    xi = config["measurement"]["xi"]
    expected_edges = _uniform_edges(
        xi["s_min_mpc_h"], xi["s_max_mpc_h"], xi["bin_width_mpc_h"], label="xi"
    )
    expected_mu = np.linspace(-1, 1, xi["mu_bins"] + 1)
    records = []
    for mock_id in identifiers:
        unit_id = f"{route['tracer']}--{route['condition']}--mock-{mock_id}"
        stage = "cluster" if run["metadata"].get("interface_version") == 2 else "measure"
        directory = run["root"] / stage / "units" / unit_id
        result = yaml.safe_load((directory / "result.yaml").read_text())
        expected_unit = {
            "unit_id": unit_id,
            "tracer": route["tracer"],
            "condition": route["condition"],
            "realization_id": mock_id,
        }
        if (
            any(
                result.get(key) != run["metadata"][key]
                for key in ("run_name", "code_commit", "analysis_config_sha256")
            )
            or result.get("stage") != stage
            or not same_identity(
                {key: result["unit"].get(key) for key in expected_unit}, expected_unit
            )
            or route["measurement_bin"] not in result["unit"]["measurement_bins"]
        ):
            raise ValueError("measurement unit identity or configuration differs")
        entries = json.loads((directory / "measurements.json").read_text())["measurements"]
        for cap in ("NGC", "SGC", "NGC+SGC"):
            expected = {
                key: route[key]
                for key in ("tracer", "condition", "measurement_bin", "reconstruction_state")
            }
            expected.update(
                realization_id=mock_id,
                cap=cap,
                analysis_config_sha256=run["metadata"]["analysis_config_sha256"],
            )
            matching = [record for record in entries if same_identity(record["identity"], expected)]
            if len(matching) != 1:
                raise ValueError(f"missing or duplicate {mock_id} {cap} measurement")
            record = matching[0]
            if statistic == "pk":
                if "power" not in record:
                    raise ValueError("P(k) was not produced for this route")
                path = (directory / record["power"]).resolve()
                if not path.is_relative_to(directory.resolve()) or record["units"] != "Mpc/h":
                    raise ValueError("power measurement has invalid path or units")
                with np.load(path, allow_pickle=False) as archive:
                    arrays = {key: archive[key] for key in archive.files}
                ells = config["measurement"]["power_spectrum_secondary"]["multipoles"]
                if (
                    set(arrays) != {"k", "power", "ells"}
                    or arrays["k"].ndim != 1
                    or arrays["k"].size == 0
                    or not np.array_equal(arrays["ells"], ells)
                    or arrays["power"].shape != (len(ells), len(arrays["k"]))
                    or not np.all(np.isfinite(arrays["k"]))
                    or np.any(np.diff(arrays["k"]) <= 0)
                    or not np.all(np.isfinite(arrays["power"]))
                ):
                    raise ValueError("P(k) has non-finite values or incompatible axes")
                records.append({**record, "directory": directory, "power_arrays": arrays})
                continue
            if "counts" not in record:
                raise ValueError(
                    "xi covariance requires xi products; this clustering output contains only P(k)"
                )
            path = (directory / record["counts"]).resolve()
            if not path.is_relative_to(directory.resolve()) or record["units"] != "Mpc/h":
                raise ValueError("measurement has invalid path or units")
            from pycorr import TwoPointCorrelationFunction

            estimator = TwoPointCorrelationFunction.load(str(path))
            if not same_identity(pair_count_identity(estimator), expected):
                raise ValueError("native pair-count metadata differ from the requested route")
            for count_name in estimator.count_names:
                count = getattr(estimator, count_name)
                if (
                    not np.array_equal(count.edges[0], expected_edges)
                    or not np.array_equal(count.edges[1], expected_mu)
                    or count.wcounts.shape != (len(expected_edges) - 1, len(expected_mu) - 1)
                    or not np.all(np.isfinite(count.wcounts))
                    or not np.all(np.isfinite(count.wnorm))
                    or not np.all(np.asarray(count.wnorm) > 0)
                ):
                    raise ValueError(
                        "native pair counts have invalid grids, values or normalization"
                    )
            records.append({**record, "directory": directory, "estimator": estimator})
    return records
