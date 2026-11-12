"""Reusable selection diagnostics. Inputs and experiment choices are explicit arguments."""

from __future__ import annotations


import numpy as np


def summarize_radial_artifact(record: Mapping[str, Any]) -> dict:
    """Summarize support, normalization, and smoothness without disposition."""
    edges = np.asarray(record["z_edges"], dtype=float)
    mass = np.asarray(record["target_probability_mass"], dtype=float)
    mean_counts = np.asarray(
        (
            record["predicted_final_counts"]
            if record.get("method") == "smooth_parent_forward_prediction"
            else record["mean_counts"]
        ),
        dtype=float,
    )
    if len(edges) != len(mass) + 1 or mass.shape != mean_counts.shape:
        raise ValueError("radial artifact arrays are inconsistent")
    occupied = mean_counts > 0
    return {
        "sample_identity": record["sample_identity"],
        "z_range": [float(edges[0]), float(edges[-1])],
        "bin_count": int(len(mass)),
        "probability_sum": float(mass.sum()),
        "minimum_occupied_probability": float(mass[occupied].min()) if np.any(occupied) else None,
        "unsupported_occupied_bins": np.flatnonzero(occupied & (mass <= 0)).tolist(),
        "adjacent_probability_jump_max": (
            float(np.max(np.abs(np.diff(mass)))) if len(mass) > 1 else 0.0
        ),
    }


def build_report(run):
    """Read preparation through the same direct identity checks as measurement."""
    from ..selection import read_preparation
    from ..run import stage_units

    records = []
    for unit in stage_units("select", run["config"], run["metadata"]["workload"]):
        preparation = read_preparation(run, unit["tracer"])
        for condition, caps in preparation["conditions"].items():
            for cap, group in caps.items():
                records.append(
                    {
                        "tracer": unit["tracer"],
                        "condition": condition,
                        "cap": cap,
                        "radial": summarize_radial_artifact(group["radial"]),
                        "counts": group["counts"],
                        "volumes_mpc_h3": group["volumes_mpc_h3"],
                        "attained_density_h3_mpc3": {
                            name: count / group["volumes_mpc_h3"][name]
                            for name, count in group["counts"].items()
                        },
                    }
                )
    return {
        "run_name": run["metadata"]["run_name"],
        "analysis_config_sha256": run["metadata"]["analysis_config_sha256"],
        "radial_models": records,
        "diagnostic_only": True,
    }


from ..config import load_analysis_config

from ..selection import Catalog, copy_catalog, validate_catalog


def shuffled_redshifts_diagnostic(
    randoms: Mapping[str, Any],
    data_redshifts: Sequence[float],
    *,
    config: Optional[Mapping[str, Any]] = None,
) -> Catalog:
    """Map stable random quantiles through the empirical data-redshift CDF.

    This deterministic inverse-CDF assignment is not a literal draw-with-
    replacement shuffle.  Quantiles must be finite and are clipped to
    ``[0, nextafter(1, 0)]``; ``min(int(q * n), n - 1)`` therefore remains
    inside empirical data support.  The returned catalogue is an independent
    copy and is diagnostic evidence only.
    """
    resolved = config or load_analysis_config()
    rank_column = str(resolved["selection"]["stable_randomness"]["random_thinning_rank_column"])
    validate_catalog(randoms, (rank_column,))
    data_z = np.sort(np.asarray(data_redshifts, dtype=float))
    if data_z.size == 0 or np.any(~np.isfinite(data_z)):
        raise ValueError("diagnostic shuffled redshifts require finite data support")
    quantiles = np.asarray(randoms[rank_column], dtype=float)
    if np.any(~np.isfinite(quantiles)):
        raise ValueError("diagnostic random quantiles must be finite")
    quantiles = np.clip(quantiles, 0.0, np.nextafter(1.0, 0.0))
    indices = np.minimum((quantiles * len(data_z)).astype(int), len(data_z) - 1)
    result = copy_catalog(randoms)
    result["Z"] = data_z[indices]
    return result


import json

import math

from pathlib import Path

from typing import Any, Dict, Mapping, Optional, Sequence

_radial_calibration_plots_NOISE_FLOOR_BAND = 1.0


def standardised_residual(
    mean_counts: Sequence[float],
    model_probability_mass: Sequence[float],
    standard_error: Sequence[float],
) -> np.ndarray:
    """Return ``(model - mean) / standard-error`` per shell, in units of the noise floor.

    The stored model is a probability mass over shells while the measurement is a count,
    so the model is put on the count scale by the measured total before differencing.
    Shells with no measured standard error cannot be judged and return zero rather than a
    non-finite value, so a single empty shell cannot poison the summary statistics.
    """
    mean = np.asarray(mean_counts, dtype=float)
    mass = np.asarray(model_probability_mass, dtype=float)
    sigma = np.asarray(standard_error, dtype=float)
    if not mean.shape == mass.shape == sigma.shape:
        raise ValueError("radial residual inputs must share one shell grid")
    if mean.ndim != 1 or mean.size == 0:
        raise ValueError("radial residual inputs must be one non-empty shell axis")
    total = float(mean.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("radial residual requires positive total calibration counts")
    residual = np.zeros_like(mean)
    usable = np.isfinite(sigma) & (sigma > 0)
    residual[usable] = (mass[usable] * total - mean[usable]) / sigma[usable]
    return residual


def residual_summary(residual: Sequence[float]) -> Dict[str, float]:
    """Summarise a standardised residual against the noise floor.

    ``root_mean_square`` is the informative one. A model fitted exactly to the floor has
    RMS one; below one it is absorbing noise, above one it is failing to track the mean.
    ``fraction_beyond_band`` is reported alongside because for a few hundred shells a
    handful of excursions past one standard error is expected rather than alarming --
    roughly a third of shells should exceed it if the residual is genuinely noise.
    """
    values = np.asarray(residual, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("residual summary requires one non-empty shell axis")
    magnitude = np.abs(values)
    return {
        "root_mean_square": float(math.sqrt(float(np.mean(values**2)))),
        "maximum_absolute": float(np.max(magnitude)),
        "fraction_beyond_band": float(
            np.mean(magnitude > _radial_calibration_plots_NOISE_FLOOR_BAND)
        ),
        "shell_count": int(values.size),
    }


def poisson_fractional_variance(mean_counts: Sequence[float]) -> np.ndarray:
    """Return the fractional variance a pure counting process would show per shell.

    For Poisson counts the variance equals the mean, so the fractional variance is the
    reciprocal of the mean count. Panel D overlays this on the measured mock-to-mock
    fractional variance; the excess over this curve is the part of the shell-to-shell
    structure that is real rather than counting noise.
    """
    mean = np.asarray(mean_counts, dtype=float)
    expected = np.full(mean.shape, np.nan)
    positive = mean > 0
    expected[positive] = 1.0 / mean[positive]
    return expected


def load_candidate(path: Path) -> Dict[str, Any]:
    """Read one stored smooth-radial candidate and expose the arrays the panels need."""
    from ..artifacts import resolve_saved_path

    payload = json.loads(resolve_saved_path(path).read_text())
    kind = payload.get("kind")
    if kind not in (None, "smooth_radial_candidate"):
        raise ValueError("expected a smooth_radial_candidate artifact, found %r" % (kind,))
    variance = payload["variance_evidence"]
    return {
        "sample_identity": payload["sample_identity"],
        "smoothing_method": payload["smoothing_method"],
        "bandwidth": float(payload["bandwidth"]),
        "bandwidth_semantics": payload.get("bandwidth_semantics"),
        "z_edges": np.asarray(payload["z_edges"], dtype=float),
        "mean_counts": np.asarray(payload["mean_counts"], dtype=float),
        "target_probability_mass": np.asarray(payload["target_probability_mass"], dtype=float),
        "standard_error": np.asarray(variance["standard_error_of_mean_count_by_bin"], dtype=float),
        "fractional_variance": np.asarray(
            [
                np.nan if value is None else value
                for value in variance["fractional_variance_by_bin"]
            ],
            dtype=float,
        ),
        "mean_fractional_variance": float(variance["mean_fractional_variance_over_occupied_bins"]),
        "total_variation_distance": float(payload["fidelity_evidence"]["total_variation_distance"]),
    }


def evaluate_candidates(candidates: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Score one sample's candidates against the noise floor.

    The candidates must describe the same sample on the same shell grid; differing
    measurements cannot be compared on one page and that is refused rather than plotted.
    """
    if not candidates:
        raise ValueError("a calibration page needs at least one candidate")
    reference = candidates[0]
    for candidate in candidates[1:]:
        if not np.array_equal(candidate["z_edges"], reference["z_edges"]):
            raise ValueError("candidates for one page must share a shell grid")
        if not np.allclose(candidate["mean_counts"], reference["mean_counts"], rtol=0, atol=0):
            raise ValueError("candidates for one page must share one calibration measurement")
    scored = []
    for candidate in candidates:
        residual = standardised_residual(
            candidate["mean_counts"],
            candidate["target_probability_mass"],
            candidate["standard_error"],
        )
        summary = residual_summary(residual)
        scored.append(
            {
                "smoothing_method": candidate["smoothing_method"],
                "bandwidth": candidate["bandwidth"],
                "bandwidth_semantics": candidate.get("bandwidth_semantics"),
                "total_variation_distance": candidate["total_variation_distance"],
                "residual": residual,
                "model_counts": candidate["target_probability_mass"]
                * float(np.asarray(candidate["mean_counts"], dtype=float).sum()),
                **summary,
            }
        )
    scored.sort(key=lambda row: (row["smoothing_method"], row["bandwidth"]))
    return {
        "sample_identity": reference["sample_identity"],
        "z_edges": reference["z_edges"],
        "mean_counts": reference["mean_counts"],
        "standard_error": reference["standard_error"],
        "fractional_variance": reference["fractional_variance"],
        "mean_fractional_variance": reference["mean_fractional_variance"],
        "candidates": scored,
    }


def select_bandwidth_at_noise_floor(
    scored: Mapping[str, Any], *, smoothing_method: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Return the candidate whose residual sits closest to the noise floor from below.

    The calibration asks for the *largest* bandwidth whose residual still tracks the mean, on
    the reasoning that a curve fitting more tightly than the mean is determined is
    describing noise. Expressed on the standardised residual that is the candidate whose
    root-mean-square is nearest one without exceeding it; if every candidate overshoots,
    the least-overshooting one is returned so the caller sees the best available rather
    than nothing, and the caller can see from the summary that none reached the floor.
    """
    rows = [
        row
        for row in scored["candidates"]
        if smoothing_method is None or row["smoothing_method"] == smoothing_method
    ]
    if not rows:
        return None
    below = [
        row for row in rows if row["root_mean_square"] <= _radial_calibration_plots_NOISE_FLOOR_BAND
    ]
    if below:
        return max(below, key=lambda row: row["root_mean_square"])
    return min(rows, key=lambda row: row["root_mean_square"])


def render_page(scored: Mapping[str, Any], output_path: Path) -> Path:
    """Write the four-panel evidence page for one sample.

    Importing matplotlib is deferred to here so the scoring helpers above stay usable in a
    lightweight environment; the repository keeps heavy optional dependencies guarded.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    edges = np.asarray(scored["z_edges"], dtype=float)
    centres = 0.5 * (edges[:-1] + edges[1:])
    mean = np.asarray(scored["mean_counts"], dtype=float)
    sigma = np.asarray(scored["standard_error"], dtype=float)
    total = float(mean.sum())
    figure, axes = plt.subplots(4, 1, figsize=(7.2, 11.0), sharex=True)
    axis_a, axis_b, axis_c, axis_d = axes
    axis_a.fill_between(
        centres,
        mean - sigma,
        mean + sigma,
        alpha=0.35,
        linewidth=0,
        label="standard error on the calibration mean",
    )
    axis_a.plot(centres, mean, linewidth=0.9, color="0.15", label="across-mock mean")
    axis_a.set_ylabel("galaxies per shell")
    axis_a.legend(loc="best", fontsize=7, frameon=False)
    axis_a.set_title(str(scored["sample_identity"]), fontsize=8)
    axis_b.fill_between(centres, mean - sigma, mean + sigma, alpha=0.35, linewidth=0)
    for row in scored["candidates"]:
        axis_b.plot(
            centres,
            row["model_counts"],
            linewidth=0.8,
            label="%s bw=%g" % (row["smoothing_method"], row["bandwidth"]),
        )
    axis_b.set_ylabel("model, galaxies per shell")
    axis_b.legend(loc="best", fontsize=6, frameon=False, ncol=2)
    axis_c.axhspan(
        -_radial_calibration_plots_NOISE_FLOOR_BAND,
        _radial_calibration_plots_NOISE_FLOOR_BAND,
        alpha=0.2,
        linewidth=0,
    )
    for row in scored["candidates"]:
        axis_c.plot(
            centres,
            row["residual"],
            linewidth=0.7,
            label="%s bw=%g  RMS=%.2f"
            % (row["smoothing_method"][:8], row["bandwidth"], row["root_mean_square"]),
        )
    axis_c.axhline(0.0, linewidth=0.6, color="0.4")
    axis_c.set_yscale("symlog", linthresh=_radial_calibration_plots_NOISE_FLOOR_BAND, linscale=1.0)
    axis_c.set_ylabel("(model - mean) / standard error")
    axis_c.legend(loc="best", fontsize=6, frameon=False, ncol=2)
    measured = np.asarray(scored["fractional_variance"], dtype=float)
    axis_d.plot(
        centres, measured, linewidth=0.8, color="0.15", label="mock-to-mock fractional variance"
    )
    axis_d.plot(
        centres,
        poisson_fractional_variance(mean),
        linewidth=0.8,
        linestyle="--",
        label="Poisson expectation",
    )
    axis_d.set_yscale("log")
    axis_d.set_ylabel("fractional variance")
    axis_d.set_xlabel("redshift")
    axis_d.legend(loc="best", fontsize=7, frameon=False)
    figure.align_ylabels(axes)
    figure.tight_layout()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=200)
    plt.close(figure)
    if not np.isfinite(total):
        raise RuntimeError("calibration page written from a non-finite measurement")
    return destination


from contextlib import contextmanager

from unittest.mock import patch


@contextmanager
def selection_adapters(case, design, mock, cap, config):
    """Change selection draws/support while preserving object-bound photo-z deviates."""
    from pzbao import selection as modifier

    original_prepare = modifier.prepare_data_for_selection
    original_select = modifier.select_exact_attainable

    def prepare(data, **kwargs):
        if case.get("exterior") is False:
            from pzbao.catalog import reconstruction_field

            field = reconstruction_field(kwargs["tracer"], config)
            redshift = np.asarray(data["Z"])
            keep = (redshift >= field["zmin"]) & (redshift < field["zmax"])
            data = {key: np.asarray(value)[keep] for key, value in data.items()}
            kwargs.update(reservoir=None, reservoir_record=None)
        return original_prepare(data, **kwargs)

    def select(data, **kwargs):
        if case.get("selection") == "may_fsky":
            from .selection import select as may_select

            return may_select(original_select, data, design=design, cap=cap, **kwargs)
        if case.get("selection") == "sequential_bin_average":
            rng = np.random.default_rng(
                np.random.SeedSequence([design["selection_seed"], mock, ["NGC", "SGC"].index(cap)])
            )
            column = config["selection"]["stable_randomness"]["selection_rank_column"]
            from pzbao.catalog import measurement_bins

            observed = config["selection"]["stable_randomness"]["observed_redshift_column"]
            ranks = 1.0 + np.arange(len(data["Z"]), dtype=float) / max(len(data["Z"]), 1)
            for item in measurement_bins(kwargs["tracer"], config):
                indices = np.flatnonzero(
                    (data[observed] >= item["zmin"]) & (data[observed] < item["zmax"])
                )
                count = kwargs["frozen_counts_by_bin"][item["name"]]
                if count > len(indices):
                    raise ValueError("bin-averaged selection is not attainable")
                selected = indices[rng.choice(len(indices), size=count, replace=False)]
                ranks[selected] -= 1.0
            result, record = original_select({**data, column: ranks}, **kwargs)
            record.update(
                method="sequential_choice_at_fixed_bin_counts",
                historical_membership_replayed=False,
                five_percent_skip_disabled_to_hold_counts_fixed=True,
            )
            return (result, record)
        return original_select(data, **kwargs)

    with (
        patch.object(modifier, "prepare_data_for_selection", prepare),
        patch.object(modifier, "select_exact_attainable", select),
    ):
        yield


from copy import deepcopy

from pzbao.config import analysis_config_sha256, same_identity, validate_boundary_variant


def reference_identity(run):
    return {
        "root": str(run["root"]),
        **{
            key: run["metadata"][key]
            for key in ("run_name", "code_commit", "analysis_config_sha256")
        },
    }


def fixed_depth(preparation, *, target, config, design):
    """Derive the fixed cut from all saved edge records, never from fitted outcomes."""
    tracer, side = (target["tracer"], target["side"])
    boundary = config["tracers"][tracer]["field"]["zmin" if side == "below" else "zmax"]
    sigmas = {
        item["label"]: item["sigma_z_over_one_plus_z"] for item in config["selection"]["tiers"]
    }
    rows = []
    for cap in ("NGC", "SGC"):
        records = preparation["conditions"][design["reference_condition"]][cap][
            "boundary_counts_by_mock"
        ]
        if [item["realization_id"] for item in records] != config["analysis"]["mock_ids"]:
            raise ValueError("fixed-depth source must cover each realization exactly once")
        for record in records:
            edges = [item for item in record["edge_support"] if item["side"] == side]
            if len(edges) != 1 or edges[0]["redshift"] != boundary:
                raise ValueError("fixed-depth source has a different or duplicated edge")
            available = float(edges[0]["available_depth_redshift"])
            if not np.isfinite(available) or available <= 0:
                raise ValueError("fixed-depth source has invalid available support")
            rows.append(
                {
                    "cap": cap,
                    "realization_id": record["realization_id"],
                    "available_depth_redshift": available,
                }
            )
    minimum = min((row["available_depth_redshift"] for row in rows))
    k = minimum / (sigmas[design["reference_condition"]] * (1 + boundary))
    return {
        "normalized_depth": k,
        "reference_depth_redshift": minimum,
        "depth_redshift": k * sigmas[design["rehearsal_condition"]] * (1 + boundary),
        "source_rows": rows,
    }


def validate_reference_variant(reference, config, *, preparation=None):
    """Bind the configured design and fixed cuts to the published reference preparation."""
    from pzbao.selection import read_preparation

    validate_boundary_variant(reference["config"], config)
    support = config["selection"]["boundary_support"]
    experiment, design = (support["experiment"], support["experiment_design"])
    if not same_identity(experiment["reference"], reference_identity(reference)):
        raise ValueError("boundary reference identity changed")
    if experiment["mode"] == "truncate":
        if preparation is None:
            preparation = read_preparation(reference, experiment["tracer"])
        target = next(
            (
                item
                for item in design["targets"]
                if item["tracer"] == experiment["tracer"] and item["side"] == experiment["side"]
            )
        )
        derived = fixed_depth(preparation, target=target, config=reference["config"], design=design)
        if any(
            (
                experiment[key] != derived[key]
                for key in ("depth_redshift", "reference_depth_redshift")
            )
        ):
            raise ValueError("boundary truncation depths differ from frozen reference preparation")


def variant_config(reference, target, *, mode, depth, previous_depth, design):
    config = deepcopy(reference["config"])
    condition = design["rehearsal_condition" if mode == "truncate" else "reference_condition"]
    support = config["selection"]["boundary_support"]
    support["experiment_design"] = deepcopy(design)
    support["experiment"] = {
        "mode": mode,
        "tracer": target["tracer"],
        "side": target["side"],
        "condition": condition,
        "depth_redshift": depth,
        "reference_depth_redshift": previous_depth,
        "reference": reference_identity(reference),
    }
    config["analysis"]["workloads"]["boundary"] = {
        "measurement_bins": [target["measurement_bin"]],
        "sigma_z_tiers": [condition],
        "reconstruction_states": ["prereconstruction", "postreconstruction"],
        "mock_ids": list(config["analysis"]["mock_ids"]),
    }
    if mode == "extend":
        reservoir = config["tracers"][target["tracer"]]["reservoir"]
        reservoir["slice_filename"] = (
            str(Path(reservoir["slice_filename"]).stem) + "_boundary_upper.fits"
        )
        next((edge for edge in reservoir["edges"] if edge["side"] == target["side"]))[
            "slice_half_width_redshift"
        ] = depth
    validate_boundary_variant(reference["config"], config)
    return config


def compare_membership(reference, variant, *, config, tracer):
    """Compare actual stable-ID populations; verify every common draw and observed redshift."""
    stable = config["selection"]["stable_randomness"]
    identity = stable["identity_column"]
    for catalog in (reference, variant):
        if len(np.unique(catalog[identity])) != len(catalog[identity]):
            raise ValueError("membership screen contains duplicate identities")
    _, left, right = np.intersect1d(
        reference[identity], variant[identity], assume_unique=True, return_indices=True
    )
    for column in (
        "RA",
        "DEC",
        "Z",
        identity,
        stable["photoz_deviate_column"],
        stable["selection_rank_column"],
        stable["source_redshift_column"],
        stable["observed_redshift_column"],
    ):
        if not np.array_equal(reference[column][left], variant[column][right]):
            raise ValueError(f"membership screen changed surviving values: {column}")
    column = stable["observed_redshift_column"]
    rows = []
    field = config["tracers"][tracer]["field"]
    for item in [{"name": "field", **field}, *config["tracers"][tracer]["measurement_bins"]]:
        ids = [
            catalog[identity][(catalog[column] >= item["zmin"]) & (catalog[column] < item["zmax"])]
            for catalog in (reference, variant)
        ]
        rows.append(
            {
                "measurement_bin": item["name"],
                "reference_count": len(ids[0]),
                "variant_count": len(ids[1]),
                "lost_count": len(np.setdiff1d(ids[0], ids[1], assume_unique=True)),
                "added_count": len(np.setdiff1d(ids[1], ids[0], assume_unique=True)),
            }
        )
    return rows


def screen_unit(reference, config, *, mock_id, cap, catalog_root=None):
    from pzbao.selection import read_preparation, _reservoir_record
    from pzbao.catalog import _read_parent
    from pzbao.selection import prepare_data_for_selection
    from pzbao.catalog import file_sha256, reservoir_slice_path, read_registered_catalogue

    validate_boundary_variant(reference["config"], config)
    experiment = config["selection"]["boundary_support"]["experiment"]
    if not same_identity(experiment["reference"], reference_identity(reference)):
        raise ValueError("screen uses a different frozen reference")
    if mock_id not in config["analysis"]["mock_ids"] or cap not in {"NGC", "SGC"}:
        raise ValueError("screen unit is outside the ensemble")
    tracer, condition = (experiment["tracer"], experiment["condition"])
    preparation = read_preparation(reference, tracer)
    validate_reference_variant(reference, config, preparation=preparation)
    sources = preparation["sources"]
    raw = _read_parent(
        sources,
        tracer=tracer,
        mock_id=mock_id,
        cap=cap,
        config=reference["config"],
        include_randoms=False,
    )
    record = _reservoir_record(sources, tracer, condition, mock_id, cap, reference["config"])
    baseline, before = prepare_data_for_selection(
        raw["data"],
        tracer=tracer,
        tier_label=condition,
        cap=cap,
        reservoir=raw["reservoir"],
        reservoir_record=record,
        config=reference["config"],
    )
    if experiment["mode"] == "extend":
        if catalog_root is None:
            raise ValueError("extension screen requires a separately published catalog root")
        path = reservoir_slice_path(catalog_root, mock_id, tracer, config)
        stat = path.stat()
        record = {
            **record,
            "path": str(path.resolve()),
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "ctime_ns": stat.st_ctime_ns,
            "sha256": file_sha256(path),
        }
        raw["reservoir"] = read_registered_catalogue(
            record, tracer=tracer, mock_id=mock_id, cap=cap, role="reservoir", config=config
        )
    changed, after = prepare_data_for_selection(
        raw["data"],
        tracer=tracer,
        tier_label=condition,
        cap=cap,
        reservoir=raw["reservoir"],
        reservoir_record=record,
        config=config,
    )
    rows = compare_membership(baseline, changed, config=config, tracer=tracer)
    if any(
        (row["added_count" if experiment["mode"] == "truncate" else "lost_count"] for row in rows)
    ):
        raise ValueError("screen membership is not nested in the declared direction")
    return {
        "reference": reference_identity(reference),
        "variant_config_sha256": analysis_config_sha256(config),
        "mock_id": mock_id,
        "cap": cap,
        "rows": rows,
        "reference_diagnostics": before,
        "variant_diagnostics": after,
        "variant_reservoir": {key: record[key] for key in ("path", "sha256")},
    }


def summarize_screens(records, config):
    expected = {(mock, cap) for mock in config["analysis"]["mock_ids"] for cap in ("NGC", "SGC")}
    observed = [(record["mock_id"], record["cap"]) for record in records]
    if len(observed) != len(expected) or set(observed) != expected:
        raise ValueError("screen summary requires each mock and cap exactly once")
    experiment = config["selection"]["boundary_support"]["experiment"]
    names = [
        "field",
        *[item["name"] for item in config["tracers"][experiment["tracer"]]["measurement_bins"]],
    ]
    totals = {
        name: {key: 0 for key in ("reference_count", "variant_count", "lost_count", "added_count")}
        for name in names
    }
    for record in records:
        if record["variant_config_sha256"] != analysis_config_sha256(config) or not same_identity(
            record["reference"], experiment["reference"]
        ):
            raise ValueError("screen summary contains incompatible provenance")
        if [row["measurement_bin"] for row in record["rows"]] != names:
            raise ValueError("screen bins differ")
        for key in totals["field"]:
            if record["rows"][0][key] != sum((row[key] for row in record["rows"][1:])):
                raise ValueError("screen field counts differ from its disjoint bins")
        for row in record["rows"]:
            if (
                row["variant_count"]
                != row["reference_count"] - row["lost_count"] + row["added_count"]
            ):
                raise ValueError("screen unit count accounting differs")
            if row["added_count" if experiment["mode"] == "truncate" else "lost_count"]:
                raise ValueError("screen membership is not nested in the declared direction")
            for key in totals[row["measurement_bin"]]:
                value = row[key]
                if type(value) is not int or value < 0:
                    raise ValueError("screen counts must be nonnegative integers")
                totals[row["measurement_bin"]][key] += value
    for row in totals.values():
        if row["variant_count"] != row["reference_count"] - row["lost_count"] + row["added_count"]:
            raise ValueError("screen count accounting differs")
        for label in ("lost", "added"):
            row[label + "_fraction"] = (
                row[label + "_count"] / row["reference_count"] if row["reference_count"] else None
            )
    changed = totals["field"]["lost_count"] + totals["field"]["added_count"]
    return {
        "reference": experiment["reference"],
        "variant_config_sha256": analysis_config_sha256(config),
        "totals": totals,
        "disposition": (
            "zero_membership_change" if changed == 0 else "human_review_before_full_chain"
        ),
        "units": records,
    }


def paired_profile_rows(reference, variant, identifiers, parameters):
    """Keep failed/missing intervals in the paired table and withhold incomplete summaries."""
    from pzbao.diagnostics.review import _minos_half_width

    rows, summaries = ([], [])
    for parameter in parameters:
        values = []
        for identifier in [*identifiers, None]:
            row = {"realization_id": identifier, "parameter": parameter}
            for label, record in (("reference", reference), ("variant", variant)):
                fits = (
                    record.get("individual_fits", [])
                    if identifier is not None
                    else [record.get("mean_fit", {})]
                )
                fits = [fit for fit in fits if fit.get("realization_id") == identifier]
                matches = [
                    item
                    for fit in fits
                    for item in fit.get("parameters", [])
                    if item["basename"] == parameter
                ]
                fitted = matches[0] if len(matches) == 1 else None
                row[label] = fitted
                try:
                    width = _minos_half_width(fitted)
                    flags = fitted.get("interval_flags") or {}
                    if not all(
                        (flags.get(key) for key in ("is_valid", "lower_valid", "upper_valid"))
                    ):
                        raise ValueError("native interval is invalid")
                    row[label + "_half_width"] = width
                    row[label + "_reason"] = None
                except ValueError as exc:
                    row[label + "_half_width"] = None
                    row[label + "_reason"] = str(exc)
            row["dilation_shift"] = (
                row["variant"]["value"] - row["reference"]["value"]
                if row["variant"] and row["reference"]
                else None
            )
            left, right = (row["reference_half_width"], row["variant_half_width"])
            row["interval_width_fractional_change"] = (
                right / left - 1 if left is not None and right is not None else None
            )
            rows.append(row)
            if identifier is not None:
                values.append(row)
        complete = all(
            (
                item["dilation_shift"] is not None
                and item["interval_width_fractional_change"] is not None
                for item in values
            )
        )
        summaries.append(
            {
                "parameter": parameter,
                "alpha_shift_sigma_1": (
                    float(
                        np.mean([item["dilation_shift"] for item in values])
                        / np.median([item["reference_half_width"] for item in values])
                    )
                    if complete
                    else None
                ),
                "interval_width_fractional_change": (
                    float(np.mean([item["interval_width_fractional_change"] for item in values]))
                    if complete
                    else None
                ),
                "reason": (
                    None
                    if complete
                    else "Incomplete or invalid paired intervals; no realizations dropped"
                ),
            }
        )
    return (rows, summaries)


import copy


def selection_config(config, condition, case):
    result = copy.deepcopy(config)
    tier = next((t for t in result["selection"]["tiers"] if t["label"] == condition))
    if case["density"] == "historical" or case["zero_error"]:
        tier["target_nbar_h3_mpc3"] = None
    if case["zero_error"]:
        tier["sigma_z_over_one_plus_z"] = 0.0
    if case.get("random_seed_offset"):
        result["selection"]["radial_randoms"]["stable_thinning_seed"] += int(
            case["random_seed_offset"]
        )
    return result


def selected_data(raw, preparation, config, tracer, condition, mock, cap):
    from pzbao import selection as modifier
    import pzbao.selection as pipeline_selection

    reservoir_record = pipeline_selection._reservoir_record(
        preparation["sources"], tracer, condition, mock, cap, config
    )
    data, boundary = modifier.prepare_data_for_selection(
        raw["data"],
        tracer=tracer,
        tier_label=condition,
        cap=cap,
        reservoir=raw["reservoir"],
        reservoir_record=reservoir_record,
        config=config,
    )
    group = preparation["conditions"][condition][cap]
    data, selection = modifier.select_exact_attainable(
        data,
        tracer=tracer,
        tier_label=condition,
        shell_volumes_mpc_h3=group["volumes_mpc_h3"],
        frozen_counts_by_bin=group["counts"],
        config=config,
    )
    return (data, {"boundary": boundary, "selection": selection})


def empirical_radial(artifact, redshift):
    result = copy.deepcopy(artifact)
    histogram = np.histogram(redshift, bins=result["z_edges"])[0]
    if histogram.sum() != len(redshift) or not histogram.sum():
        raise ValueError("empirical radial target loses selected data")
    result["target_probability_mass"] = (histogram / histogram.sum()).tolist()
    result["assessment_target"] = "individual_mock_histogram"
    return result


def requested_fkp(catalog, *, tracer, condition, redshift_column, config, p0):
    from pzbao.catalog import measurement_bins
    from pzbao.selection import tier_spec

    targets = tier_spec(condition, config)["target_nbar_h3_mpc3"]
    if targets is None:
        raise ValueError("requested-density FKP needs a finite target table")
    result = dict(catalog)
    weights = np.full(len(catalog[redshift_column]), np.nan)
    for item in measurement_bins(tracer, config):
        mask = (catalog[redshift_column] >= item["zmin"]) & (
            catalog[redshift_column] < item["zmax"]
        )
        weights[mask] = 1.0 / (1.0 + targets[item["name"]] * p0)
    if not np.all(np.isfinite(weights)):
        raise ValueError("historical FKP omitted catalogue rows")
    result["WEIGHT_FKP"] = weights
    return result


def prescription(design):
    spec = design["density_selection"]
    if (
        spec["kind"] != "may_fsky"
        or spec["weights"] != "current_actual_volume"
        or spec["distance_relation"] != "current"
    ):
        raise ValueError("unsupported May sky-fraction intervention")
    shares = spec["cap_area_split_deg2"]
    if set(shares) != {"NGC", "SGC"}:
        raise ValueError("both physical cap shares are required")
    values = np.asarray([spec["combined_f_sky"], spec["skip_tolerance"], *shares.values()])
    if (
        not np.isfinite(values).all()
        or not 0 < values[0] <= 1
        or (not 0 <= values[1] < 1)
        or np.any(values[2:] <= 0)
    ):
        raise ValueError("invalid sky fraction, cap share or skip tolerance")
    return spec


def assumed_fraction(design, cap):
    spec = prescription(design)
    shares = spec["cap_area_split_deg2"]
    return spec["combined_f_sky"] * shares[cap] / sum(shares.values())


def retained_count(requested, available, tolerance):
    """May's integer threshold, including exact boundary and no upsampling."""
    if (
        int(requested) != requested
        or int(available) != available
        or requested < 0
        or (available < 0)
        or (not 0 <= tolerance < 1)
    ):
        raise ValueError("invalid count or tolerance")
    skipped = requested >= int((1.0 - tolerance) * available)
    return (int(available) if skipped else int(requested), bool(skipped))


def count_plan(design, config, preparation, case):
    from pzbao.catalog import measurement_bins, sky_fraction

    spec = prescription(design)
    if preparation["realization_ids"] != design["mock_ids"]:
        raise ValueError("sky-fraction preparation has another mock cohort")
    condition = case["condition"]
    target_condition = spec["density_target_conditions"].get(condition, condition)
    target = next((t for t in config["selection"]["tiers"] if t["label"] == target_condition))
    if target["target_nbar_h3_mpc3"] is None:
        raise ValueError("sky-fraction control requires a finite density target")
    bins = measurement_bins(case["tracer"], config)
    counts, caps = ({}, {})
    for cap in ("NGC", "SGC"):
        group = preparation["conditions"][condition][cap]
        actual_fraction = sky_fraction(case["tracer"], cap, config)
        assumed = assumed_fraction(design, cap)
        available = np.asarray(group["attainable_counts_by_mock"])
        if (
            available.shape != (len(design["mock_ids"]), len(bins))
            or available.dtype.kind not in "iu"
            or np.any(available < 0)
        ):
            raise ValueError("invalid per-mock availability")
        counts[cap], caps[cap] = ({}, {})
        for column, item in enumerate(bins):
            name = item["name"]
            volume = float(group["volumes_mpc_h3"][name])
            nbar = float(target["target_nbar_h3_mpc3"][name])
            if (
                not np.isfinite([volume, nbar, actual_fraction]).all()
                or min(volume, nbar, actual_fraction) <= 0
            ):
                raise ValueError("invalid target density or actual volume")
            wrong_volume = volume * (assumed / actual_fraction)
            requested = int(round(nbar * wrong_volume))
            outcomes = [
                retained_count(requested, int(n), spec["skip_tolerance"])
                for n in available[:, column]
            ]
            counts[cap][name] = requested
            caps[cap][name] = {
                "actual_f_sky": actual_fraction,
                "assumed_f_sky": assumed,
                "actual_volume_mpc_h3": volume,
                "assumed_volume_mpc_h3": wrong_volume,
                "target_nbar_h3_mpc3": nbar,
                "requested_count": requested,
                "available_by_mock": available[:, column].tolist(),
                "retained_by_mock": [r[0] for r in outcomes],
                "skip_thinning_by_mock": [r[1] for r in outcomes],
            }
    return (
        counts,
        {
            "policy": "may_fsky_per_mock_skip",
            "mock_ids": design["mock_ids"],
            "target_condition": target_condition,
            "caps": caps,
        },
    )


def select(original, data, *, design, cap, **kwargs):
    """Use native rank selection and actual-volume FKP after May's count rule."""
    from pzbao.catalog import measurement_bins, sky_fraction

    config = kwargs["config"]
    spec = prescription(design)
    observed = config["selection"]["stable_randomness"]["observed_redshift_column"]
    counts, evidence = ({}, {})
    for item in measurement_bins(kwargs["tracer"], config):
        name = item["name"]
        available = int(
            np.count_nonzero((data[observed] >= item["zmin"]) & (data[observed] < item["zmax"]))
        )
        requested = kwargs["frozen_counts_by_bin"][name]
        counts[name], skipped = retained_count(requested, available, spec["skip_tolerance"])
        evidence[name] = {
            "requested_count": int(requested),
            "available_count": available,
            "retained_count": counts[name],
            "skip_thinning": skipped,
        }
    selected, record = original(data, **{**kwargs, "frozen_counts_by_bin": counts})
    record["may_fsky_selection"] = {
        "assumed_f_sky": assumed_fraction(design, cap),
        "actual_f_sky": sky_fraction(kwargs["tracer"], cap, config),
        "skip_tolerance": spec["skip_tolerance"],
        "bins": evidence,
        "weights": spec["weights"],
        "historical_membership_replayed": False,
    }
    for name, count in counts.items():
        if record["bins"][name]["attained_count"] != count:
            raise ValueError("native selector did not retain the planned count")
    return (selected, record)


def verify_plan(plan, selection, mock, cap):
    """Compare the catalogue producer to the independently saved availability."""
    index = plan["mock_ids"].index(mock)
    actual = selection["may_fsky_selection"]["bins"]
    for name, expected in plan["caps"][cap].items():
        wanted = {
            "requested_count": expected["requested_count"],
            "available_count": expected["available_by_mock"][index],
            "retained_count": expected["retained_by_mock"][index],
            "skip_thinning": expected["skip_thinning_by_mock"][index],
        }
        if actual[name] != wanted:
            raise ValueError(f"May selection differs from count plan: {mock} {cap} {name}")
        native = selection["bins"][name]
        np.testing.assert_allclose(
            native["shell_volume_mpc_h3"], expected["actual_volume_mpc_h3"], rtol=1e-12, atol=0
        )
        np.testing.assert_allclose(
            native["attained_nbar_h3_mpc3"],
            wanted["retained_count"] / expected["actual_volume_mpc_h3"],
            rtol=1e-12,
            atol=0,
        )


def require_identity(record, expected):
    if any((record.get(key) != value for key, value in expected.items())):
        raise ValueError("reused measurement producer or design differs")


def check_pairing(saved, selected, baseline, bins):
    """Verify saved selection, stable common-object noise/ranks and bin nesting."""
    columns = ("source_ids", "redshift", "rsd_redshift", "noise", "ranks", "exterior")
    for label, data in (("saved", saved), ("selected", selected), ("baseline", baseline)):
        n = len(data["source_ids"])
        if len(np.unique(data["source_ids"])) != n:
            raise ValueError(f"duplicate {label} source IDs")
        if any(
            (
                np.asarray(data[key]).shape != (n,) or not np.isfinite(data[key]).all()
                for key in columns
            )
        ):
            raise ValueError(f"invalid {label} selection columns")
    for key in columns:
        np.testing.assert_array_equal(saved[key], selected[key])
    _, left, right = np.intersect1d(
        baseline["source_ids"], saved["source_ids"], return_indices=True
    )
    for key in columns[1:]:
        np.testing.assert_array_equal(baseline[key][left], saved[key][right])
    counts = {}
    for item in bins:
        ids = [
            data["source_ids"][
                (data["redshift"] >= item["zmin"]) & (data["redshift"] < item["zmax"])
            ]
            for data in (baseline, saved)
        ]
        common = len(np.intersect1d(*ids))
        if common != min(map(len, ids)):
            raise ValueError("density selection is not nested within redshift bin")
        counts[item["name"]] = {"baseline": len(ids[0]), "density": len(ids[1]), "common": common}
    return {
        "common_objects": len(left),
        "saved_selection_reproduced": True,
        "object_noise_and_ranks_identical": True,
        "nested_bin_counts": counts,
    }


def radial_summary(data, randoms, target):
    data, randoms, target = [np.asarray(x, dtype=float) for x in (data, randoms, target)]
    if data.shape != randoms.shape or data.shape != target.shape or data.sum() <= 0:
        raise ValueError("invalid radial vectors")
    if randoms.sum() != 50 * data.sum():
        raise ValueError("radial evidence violates 50:1")
    scaled = randoms / 50
    expected = target * data.sum()
    return {
        "data_count": int(data.sum()),
        "random_count": int(randoms.sum()),
        "total_variation": float(np.abs(data / data.sum() - scaled / scaled.sum()).sum() / 2),
        "largest_allocation_residual_data_objects": float(np.max(np.abs(scaled - expected))),
        "interpretation": "Shell allocation residual; does not measure angular random noise.",
    }


def selection_pair(left, right, *, allow_rank_change=False):
    ids, first, second = np.intersect1d(
        left["source_ids"], right["source_ids"], return_indices=True
    )
    for name in ("ranks", "noise", "rsd_redshift"):
        if name == "ranks" and allow_rank_change:
            continue
        np.testing.assert_array_equal(
            left[name][first], right[name][second], err_msg=f"object pairing changed {name}"
        )
    return {
        "common_objects": len(ids),
        "left_count": len(left["source_ids"]),
        "right_count": len(right["source_ids"]),
        "left_exterior": int(left["exterior"].sum()),
        "right_exterior": int(right["exterior"].sum()),
        "left_retained_fraction": len(ids) / len(left["source_ids"]),
        "right_retained_fraction": len(ids) / len(right["source_ids"]),
        "paired_deviates_and_ranks": bool(
            np.array_equal(left["ranks"][first], right["ranks"][second])
        ),
        "paired_photoz_deviates": True,
    }


def plot_radial_target(artifact, *, ax=None, label=None):
    """Plot the saved mean counts and smooth probability mass on their native grid."""
    from ..figures.style import axes

    ax = axes(ax)
    edges = np.asarray(artifact["z_edges"])
    counts = np.asarray(
        artifact["predicted_final_counts"]
        if artifact.get("method") == "smooth_parent_forward_prediction"
        else artifact["mean_counts"]
    )
    mass = np.asarray(artifact["target_probability_mass"])
    if counts.shape != mass.shape or len(edges) != len(counts) + 1:
        raise ValueError("radial target has incompatible bin axes")
    ax.stairs(counts, edges, label=label or "Mean selected counts")
    ax.stairs(mass * counts.sum(), edges, label="Smooth target")
    ax.set(xlabel="Redshift", ylabel="Objects per bin")
    return ax, {"edges": edges, "mean_counts": counts, "target_probability_mass": mass}
