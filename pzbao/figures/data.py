"""Read and validate saved scientific products for the paper notebooks.

Readers retain native identities, covariance normalization, interval flags and sampling limits.
No function runs a fit, changes a source product, selects a manuscript result or draws a figure.
The notebook cells own plotting and the explicit choice of input and output paths.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import yaml

from pzbao.artifacts import resolve_saved_path
from pzbao.diagnostics import covariance as forecast

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = _REPO_ROOT / "paper" / "figures" / "source" / "fig_survey_veff_comparison_data.json"
DEFAULT_OUTPUT_DATA = (
    _REPO_ROOT / "paper" / "figures" / "source" / "fig_survey_r_precision_data.json"
)
# BGS and LRG rows from Table 7 of DESI Collaboration (2024),
# arXiv:2306.06307. V_eff is in (h^-1 Gpc)^3 and sigma_R is in percent.
DESI_TABLE7_BGS_LRG: Tuple[Mapping[str, object], ...] = (
    {"tracer": "BGS", "zmin": 0.0, "zmax": 0.1, "veff_gpc_h3": 0.04, "sigma_r_pct": 4.95},
    {"tracer": "BGS", "zmin": 0.1, "zmax": 0.2, "veff_gpc_h3": 0.22, "sigma_r_pct": 1.91},
    {"tracer": "BGS", "zmin": 0.2, "zmax": 0.3, "veff_gpc_h3": 0.54, "sigma_r_pct": 1.21},
    {"tracer": "BGS", "zmin": 0.3, "zmax": 0.4, "veff_gpc_h3": 0.83, "sigma_r_pct": 0.99},
    {"tracer": "LRG", "zmin": 0.4, "zmax": 0.5, "veff_gpc_h3": 1.06, "sigma_r_pct": 0.88},
    {"tracer": "LRG", "zmin": 0.5, "zmax": 0.6, "veff_gpc_h3": 1.42, "sigma_r_pct": 0.74},
    {"tracer": "LRG", "zmin": 0.6, "zmax": 0.7, "veff_gpc_h3": 1.76, "sigma_r_pct": 0.65},
    {"tracer": "LRG", "zmin": 0.7, "zmax": 0.8, "veff_gpc_h3": 2.07, "sigma_r_pct": 0.59},
    {"tracer": "LRG", "zmin": 0.8, "zmax": 0.9, "veff_gpc_h3": 2.32, "sigma_r_pct": 0.55},
    {"tracer": "LRG", "zmin": 0.9, "zmax": 1.0, "veff_gpc_h3": 2.09, "sigma_r_pct": 0.59},
    {"tracer": "LRG", "zmin": 1.0, "zmax": 1.1, "veff_gpc_h3": 1.25, "sigma_r_pct": 0.83},
)

SCENARIO_ORDER = (
    "desi_dr1",
    "desi_dr2_projection",
    "spherex_sample_1",
    "project_mock_achieved_sigma_0p003",
)
DISPLAY_LABELS = {
    "desi_dr1": "DESI DR1",
    "desi_dr2_projection": "DESI DR2",
    "spherex_sample_1": "SPHEREx sample 1",
    "project_mock_achieved_sigma_0p003": "DESI mocks",
    "spherex_cosmic_variance": "cosmic-variance limit",
}
SCENARIO_MARKERS = ("o", "s", "^", "v")
SCENARIO_LINESTYLES = ("-", "--", "-.", ":")


BINS = ("BGS1", "BGS2", "LRG1", "LRG2", "LRG3")
PARAMETERS = ("qiso", "qap")
STATES = ("prereconstruction", "postreconstruction")
ROBUSTNESS = (
    ("baseline", "Baseline: power3, RecSym, R15"),
    ("pre", "Pre-reconstruction"),
    ("pk", "Power spectrum only"),
    ("ngc", "NGC only"),
    ("flat", "Flat damping priors"),
    ("pcs2", "PCS2 spline broadband"),
    ("reciso", "RecIso"),
    ("r25", r"$R = 25\,h^{-1}\mathrm{Mpc}$"),
)
DEPENDENCIES = {
    "baseline": "Compatible power3 mean profile is unavailable.",
    "pre": "Compatible power3 pre-reconstruction mean profile is unavailable.",
    "pk": "Requires matched P(k) measurements, window, covariance and native mean fits.",
    "ngc": "Requires NGC-only measurements, window, covariance and native mean fits.",
    "flat": "Requires the configured finite-prior definition and native mean fits.",
    "pcs2": "Requires a PCS2 fit with verified common data, covariance and physical priors.",
    "reciso": "Requires RecIso reconstruction, matching window/covariance and power3 fits.",
    "r25": "Requires compatible R25 products at sigma_0.003 with fixed reference priors.",
}
LABELS = {"qiso": r"$\alpha_{\rm iso}$", "qap": r"$\alpha_{\rm AP}$"}
IDENTITY = ("run_name", "code_commit", "analysis_config_sha256")


TIERS = ("native", "sigma_0.001", "sigma_0.003", "sigma_0.01", "sigma_0.03")
LRG_BINS = ("LRG1", "LRG2", "LRG3")
R25_VARIANTS = {
    "baseline": "Baseline (R15)",
    "pre": "Pre-recon.",
    "flat": "Flat priors",
    "pcs2": "PCS2",
    "r25": r"$R=25\,h^{-1}\mathrm{Mpc}$",
}
COMPARISON_XLIM = {"qiso": (0.96, 1.04), "qap": (0.85, 1.15)}


class Sources:
    """Bind all consumed bytes; refuse a changed source during rendering."""

    def __init__(self):
        self.digests = {}

    def read(self, path, expected=None):
        return self.read_current(resolve_saved_path(path), expected)

    def read_current(self, path, expected=None):
        """Bind executing source bytes without redirecting them through an evidence archive."""
        path = Path(path).resolve()
        value = path.read_bytes()
        digest = hashlib.sha256(value).hexdigest()
        previous = self.digests.setdefault(str(path), digest)
        if digest != previous or (expected is not None and digest != expected):
            raise ValueError(f"source bytes changed: {path}")
        return value

    def json(self, path):
        return json.loads(self.read(path))

    def unchanged(self):
        for path, digest in list(self.digests.items()):
            self.read_current(path, digest)


def write_json(path, value):
    """Serialize inspection values; the caller owns destination creation and overwrite policy."""
    with Path(path).open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False, default=lambda x: x.tolist())
        stream.write("\n")


# figure support: saved values and validation.


def _run_routes(root, selectors=None):
    from ..run import load_run, routes_for
    from ..config import same_identity

    run = load_run(root)
    selectors = dict(selectors or {})
    fields = {"tracer", "measurement_bin", "condition", "reconstruction_state"}
    if not set(selectors) <= fields:
        raise ValueError("route selectors contain undeclared coordinates")
    routes = [
        route
        for route in routes_for(run["config"], run["metadata"]["workload"])
        if all(same_identity(route[key], value) for key, value in selectors.items())
    ]
    if not routes:
        raise ValueError("no configured route matches the requested selectors")
    return run, routes


def load_xi_multipoles_with_std(root, *, selectors, multipoles):
    """Read a complete native realization ensemble and return its mean and empirical scatter."""
    from ..covariance import route_vectors

    run, routes = _run_routes(root, selectors)
    if len(routes) != 1:
        raise ValueError("xi requires exactly one configured route")
    vectors = route_vectors(run, routes[0])
    if not np.array_equal(vectors["multipoles"], multipoles):
        raise ValueError("requested multipoles differ from the fitted route")
    values = vectors["vectors"].reshape(len(vectors["realization_ids"]), len(multipoles), -1)
    return vectors["separations"].mean(axis=0), values.mean(axis=0), values.std(axis=0, ddof=1)


def load_xi_residual_with_std(root, *, numerator_selectors, denominator_selectors, multipoles):
    """Return paired realization differences on the denominator's mean separation coordinates."""
    from ..covariance import route_vectors

    run, numerator = _run_routes(root, numerator_selectors)
    _, denominator = _run_routes(root, denominator_selectors)
    if len(numerator) != 1 or len(denominator) != 1:
        raise ValueError("xi residual needs two uniquely selected configured routes")
    left, right = route_vectors(run, numerator[0]), route_vectors(run, denominator[0])
    for key in ("realization_ids", "fit_edges", "multipoles"):
        if not np.array_equal(left[key], right[key]):
            raise ValueError("xi residual requires identical IDs, radial edges and multipoles")
    if not np.array_equal(left["multipoles"], multipoles):
        raise ValueError("requested multipoles differ from the fitted routes")
    differences = (left["vectors"] - right["vectors"]).reshape(
        len(left["realization_ids"]), len(multipoles), -1
    )
    return (
        right["separations"].mean(axis=0),
        differences.mean(axis=0),
        differences.std(axis=0, ddof=1),
    )


def load_chain_samples(root, *, selectors, parameter):
    """Read retained native draws with explicit (ensemble, walker, iteration) axes."""
    from ..sampling import read_chain
    from ..config import same_identity

    run, routes = _run_routes(root, selectors)
    if len(routes) != 1:
        raise ValueError("chain reading requires exactly one configured route")
    route = routes[0]
    directory = (
        run["root"]
        / "sample"
        / "units"
        / "--".join(route[key] for key in ("measurement_bin", "condition", "reconstruction_state"))
    )
    summary = json.loads((directory / "sampling.json").read_text())
    identity = {
        **route,
        **{
            key: run["metadata"][key]
            for key in ("run_name", "code_commit", "analysis_config_sha256")
        },
    }
    count = run["config"]["fitting"]["sampling"]["ensembles"]
    if (
        not same_identity(summary["identity"], identity)
        or summary.get("complete_ensemble_collection") is not True
        or not same_identity(
            [item["ensemble_index"] for item in summary["ensembles"]], list(range(count))
        )
    ):
        raise ValueError("sampling is not a complete identified ensemble collection")
    values = []
    for record in summary["ensembles"]:
        path = (directory / record["path"]).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError("native chain path leaves its sampling unit")
        decoded = read_chain(
            path, identity={**identity, "ensemble_index": record["ensemble_index"]}
        )
        if parameter not in decoded["parameters"]:
            raise ValueError(f"no scalar posterior draws for parameter {parameter}")
        index = decoded["parameters"].index(parameter)
        burn = int(
            decoded["draws"].shape[1] * run["config"]["fitting"]["sampling"]["burnin_fraction"]
            + 0.5
        )
        values.append(decoded["draws"][:, burn:, index])
    if len({value.shape for value in values}) != 1:
        raise ValueError("native ensembles have different retained axes")
    return np.stack(values)


def load_all_fits(root):
    """Read available actual mean and individual profiles; malformed native results fail."""
    from ..run import stage_units
    from ..fitting import read_profiles

    run, _ = _run_routes(root)
    rows = []
    for unit in stage_units("fit", run["config"], run["metadata"]["workload"]):
        directory = run["root"] / "fit" / "units" / unit["unit_id"]
        if not directory.exists():
            continue
        identity = {
            key: unit[key]
            for key in (
                "tracer",
                "measurement_bin",
                "condition",
                "reconstruction_state",
                "realization_id",
            )
        }
        identity.update(
            {
                key: run["metadata"][key]
                for key in ("run_name", "code_commit", "analysis_config_sha256")
            }
        )
        profile = read_profiles(directory / "profiles.npy", identity=identity)
        rows.append(
            {
                **identity,
                **profile,
                "path": str(directory / "profiles.npy"),
                "sample_label": unit["measurement_bin"] + " " + unit["condition"],
            }
        )
    return tuple(rows)


def load_all_per_mock_profiles(root):
    """Read individual profiles separately from fits to ensemble means."""
    return tuple(record for record in load_all_fits(root) if record["realization_id"] is not None)


def render_table(records):
    """Format native estimates and asymmetric profile intervals for an optional LaTeX table."""

    def estimate(record, name):
        parameter = next((row for row in record["parameters"] if row["basename"] == name), None)
        if parameter is None:
            return "---"
        value = format(parameter["value"], ".6g")
        offsets = parameter.get("minos_offsets")
        if offsets is None:
            return value + " (interval unavailable)"
        return (
            "$"
            + value
            + "_{"
            + format(offsets[0], "+.4g")
            + "}^{"
            + format(offsets[1], "+.4g")
            + "}$"
        )

    line_end = chr(92) * 2
    rows = [
        "sample & state & realization & " + r"$\alpha_{\rm iso}$ & $\alpha_{\rm AP}$ " + line_end
    ]
    for record in records:
        label = str(record["sample_label"]).replace("_", chr(92) + "_")
        state = str(record["reconstruction_state"])
        realization = "mean" if record["realization_id"] is None else str(record["realization_id"])
        rows.append(
            f"{label} & {state} & {realization} & {estimate(record, 'qiso')} & "
            f"{estimate(record, 'qap')} " + line_end
        )
    return "\n".join(rows)


def write_csv(path, rows):
    """Persist full-precision inspection values; formatting never selects scientific validity."""
    import csv
    import os

    path = Path(path)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    partial = path.with_suffix(".partial.csv")
    with partial.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys or ["unavailable_reason"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, allow_nan=False)
                        if isinstance(value, (dict, list))
                        else value
                    )
                    for key, value in row.items()
                }
            )
    os.replace(partial, path)


def effective_volume(
    shell_volume_mpc_h3: Sequence[float],
    nbar_h3_mpc3: Sequence[float],
    *,
    p0_h3_mpc3: float,
) -> np.ndarray:
    """Compute V_eff with an explicit tracer P0 supplied by the caller."""
    volume = np.asarray(shell_volume_mpc_h3, dtype=float)
    density = np.asarray(nbar_h3_mpc3, dtype=float)
    if np.any(~np.isfinite(volume)) or np.any(volume <= 0):
        raise ValueError("shell volumes must be finite and positive")
    if np.any(np.isnan(density)) or np.any(density < 0):
        raise ValueError("number densities must be non-negative")
    if float(p0_h3_mpc3) <= 0:
        raise ValueError("p0_h3_mpc3 must be positive")
    np0 = density * float(p0_h3_mpc3)
    fraction = np.where(np.isinf(np0), 1.0, (np0 / (1.0 + np0)) ** 2)
    return volume * fraction


def calibration_summary() -> Dict[str, object]:
    """Return the explicit DESI Table 7 calibration and residual scatter."""
    veff = np.asarray([row["veff_gpc_h3"] for row in DESI_TABLE7_BGS_LRG], dtype=float)
    sigma_r = np.asarray([row["sigma_r_pct"] for row in DESI_TABLE7_BGS_LRG], dtype=float)
    coefficients = sigma_r * np.sqrt(veff)
    coefficient = float(np.median(coefficients))
    predicted = coefficient / np.sqrt(veff)
    fractional_residual = (predicted - sigma_r) / sigma_r
    rows = []
    for source, value, residual in zip(DESI_TABLE7_BGS_LRG, coefficients, fractional_residual):
        row = dict(source)
        row["coefficient_pct_gpc_h3_sqrt"] = float(value)
        row["fractional_residual"] = float(residual)
        rows.append(row)
    return {
        "source": "DESI Collaboration (2024), arXiv:2306.06307, Table 7",
        "source_url": "https://arxiv.org/pdf/2306.06307",
        "selection": "BGS and LRG rows spanning 0.0 <= z < 1.1",
        "formula": "sigma_R_over_R_pct = coefficient / sqrt(V_eff_gpc_h3)",
        "coefficient_pct_gpc_h3_sqrt": coefficient,
        "coefficient_mean_pct_gpc_h3_sqrt": float(np.mean(coefficients)),
        "coefficient_std_pct_gpc_h3_sqrt": float(np.std(coefficients, ddof=1)),
        "fractional_residual_rms": float(np.sqrt(np.mean(fractional_residual**2))),
        "rows": rows,
    }


def calibration_coefficient_at(
    redshift_centers: Union[float, Sequence[float], np.ndarray],
    tracers: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """Return Table 7 ``A(z)`` interpolated in redshift within tracer."""
    centers = np.atleast_1d(np.asarray(redshift_centers, dtype=float))
    if np.any(~np.isfinite(centers)):
        raise ValueError("redshift centers must be finite")
    row_z = np.asarray(
        [0.5 * (float(row["zmin"]) + float(row["zmax"])) for row in DESI_TABLE7_BGS_LRG]
    )
    row_a = np.asarray(
        [
            float(row["sigma_r_pct"]) * np.sqrt(float(row["veff_gpc_h3"]))
            for row in DESI_TABLE7_BGS_LRG
        ]
    )
    if tracers is None:
        return np.interp(centers, row_z, row_a)
    row_tracer = np.asarray([str(row["tracer"]) for row in DESI_TABLE7_BGS_LRG])
    tracer_list = [str(value) for value in tracers]
    if len(tracer_list) != centers.size:
        raise ValueError("tracers must match the number of redshift centers")
    out = np.empty(centers.size, dtype=float)
    for index, (center, tracer) in enumerate(zip(centers, tracer_list)):
        mask = row_tracer == tracer
        if not mask.any():
            raise ValueError(f"no Table 7 rows for tracer {tracer!r}")
        out[index] = np.interp(center, row_z[mask], row_a[mask])
    return out


def _bin_redshift_center(row: Mapping[str, Any]) -> float:
    if "redshift_center" in row:
        return float(row["redshift_center"])
    if "zmin" in row and "zmax" in row:
        return 0.5 * (float(row["zmin"]) + float(row["zmax"]))
    raise ValueError("projection bin needs redshift_center or zmin/zmax")


def _table7_tracer(row: Mapping[str, Any], center: float) -> str:
    tracer = str(row.get("tracer", "") or "")
    if tracer.startswith("BGS"):
        return "BGS"
    if tracer.startswith("LRG"):
        return "LRG"
    return "BGS" if center < 0.4 else "LRG"


def r_precision_percent(
    veff_mpc_h3: Union[float, Sequence[float], np.ndarray],
    coefficient: Optional[Union[float, Sequence[float], np.ndarray]] = None,
) -> np.ndarray:
    """Convert positive effective volumes to DESI-calibrated R precision."""
    volume = np.asarray(veff_mpc_h3, dtype=float)
    if np.any(~np.isfinite(volume)) or np.any(volume <= 0.0):
        raise ValueError("effective volumes must be positive and finite")
    if coefficient is None:
        coefficient = float(calibration_summary()["coefficient_pct_gpc_h3_sqrt"])
    coefficients = np.asarray(coefficient, dtype=float)
    if np.any(~np.isfinite(coefficients)) or np.any(coefficients <= 0.0):
        raise ValueError("calibration coefficient must be positive and finite")
    if coefficients.ndim and coefficients.shape != volume.shape:
        raise ValueError("per-bin coefficients must match the effective-volume shape")
    return coefficients / np.sqrt(volume / 1.0e9)


def density_only_r_over_r_cv(
    nbar_h3_mpc3: Union[float, Sequence[float], np.ndarray],
    *,
    p0_h3_mpc3: float,
) -> np.ndarray:
    """Return the fixed-area FKP density penalty relative to cosmic variance."""
    nbar = np.asarray(nbar_h3_mpc3, dtype=float)
    if np.any(~np.isfinite(nbar)) or np.any(nbar <= 0.0):
        raise ValueError("number densities must be positive and finite")
    if not np.isfinite(p0_h3_mpc3) or p0_h3_mpc3 <= 0.0:
        raise ValueError("p0_h3_mpc3 must be positive and finite")
    return 1.0 + 1.0 / (nbar * float(p0_h3_mpc3))


def build_r_precision_projection(
    veff_projection: Mapping[str, object],
    *,
    p0_h3_mpc3: float,
) -> Dict[str, object]:
    """Build common-SPHEREx-area precision curves with tracer-local A(z).

    The SPHEREx footprint of the absolute panel is the frozen configuration's forecast sky
    fraction (revision 2.15: the 30,000 deg^2 effective area over the sky), never a literal here.
    """
    calibration = calibration_summary()
    coefficient = float(calibration["coefficient_pct_gpc_h3_sqrt"])
    target_fsky = float(forecast.configured_target_fsky())
    footprint = f"SPHEREx f_sky={target_fsky:.4f}"
    if not np.isfinite(p0_h3_mpc3) or p0_h3_mpc3 <= 0.0:
        raise ValueError("p0_h3_mpc3 must be positive and finite")
    bin_centers = np.asarray(
        [_bin_redshift_center(row) for row in veff_projection["bins"]], dtype=float
    )
    bin_tracers = [
        _table7_tracer(row, center) for row, center in zip(veff_projection["bins"], bin_centers)
    ]
    coefficients = calibration_coefficient_at(bin_centers, bin_tracers)
    source_scenarios = {row["key"]: row for row in veff_projection["scenarios"]}
    missing = [key for key in SCENARIO_ORDER if key not in source_scenarios]
    if missing:
        raise ValueError(f"effective-volume projection is missing scenarios: {missing}")
    spherex_source = source_scenarios["spherex_sample_1"]
    spherex_shell_volume = np.asarray(spherex_source["shell_volume_mpc_h3"], dtype=float)
    if np.any(~np.isfinite(spherex_shell_volume)) or np.any(spherex_shell_volume <= 0.0):
        raise ValueError("SPHEREx shell volumes must be positive and finite")
    scenarios = []
    for key in SCENARIO_ORDER:
        source = source_scenarios[key]
        source_veff = np.asarray(source["veff_mpc_h3"], dtype=float)
        nbar = np.asarray(source["nbar_h3_mpc3"], dtype=float)
        if source_veff.shape != nbar.shape or nbar.shape != spherex_shell_volume.shape:
            raise ValueError(f"effective-volume and density shapes differ for {key}")
        density_penalty = density_only_r_over_r_cv(nbar, p0_h3_mpc3=p0_h3_mpc3)
        veff = spherex_shell_volume / density_penalty**2
        scenarios.append(
            {
                "key": key,
                "label": DISPLAY_LABELS[key],
                "source_policy": source["source_policy"],
                "nbar_h3_mpc3": nbar.tolist(),
                "n_p0": (nbar * float(p0_h3_mpc3)).tolist(),
                "source_veff_mpc_h3": source_veff.tolist(),
                "absolute_panel_shell_volume_mpc_h3": spherex_shell_volume.tolist(),
                "veff_mpc_h3": veff.tolist(),
                "veff_gpc_h3": (veff / 1.0e9).tolist(),
                "sigma_r_over_r_pct": r_precision_percent(veff, coefficients).tolist(),
                "sigma_r_over_r_pct_single_a": r_precision_percent(veff, coefficient).tolist(),
                "density_only_r_over_r_cv": density_penalty.tolist(),
            }
        )
    spherex_cosmic_variance = {
        "key": "spherex_cosmic_variance",
        "label": DISPLAY_LABELS["spherex_cosmic_variance"],
        "source_policy": f"infinite-density limit on the {footprint} footprint",
        "veff_mpc_h3": spherex_shell_volume.tolist(),
        "veff_gpc_h3": (spherex_shell_volume / 1.0e9).tolist(),
        "sigma_r_over_r_pct": r_precision_percent(spherex_shell_volume, coefficients).tolist(),
        "sigma_r_over_r_pct_single_a": r_precision_percent(
            spherex_shell_volume, coefficient
        ).tolist(),
    }
    return {
        "bins": [dict(row) for row in veff_projection["bins"]],
        "calibration": calibration,
        "coefficient_policy": {
            "used": "redshift-dependent A(z) interpolated within tracer",
            "basis": "DESI Collaboration (2024), arXiv:2306.06307, Table 7",
            "interpolated_in": "redshift, never effective volume",
            "tracer_policy": "interpolation is confined to rows of the same tracer",
            "bin_redshift_centers": bin_centers.tolist(),
            "bin_tracers": list(bin_tracers),
            "coefficient_of_z": coefficients.tolist(),
            "coefficient_single_median": coefficient,
            "coefficient_ratio_to_median": (coefficients / coefficient).tolist(),
            "density_ratio_invariance": "A(z) cancels in every fixed-redshift ratio",
        },
        "forecast_scope": (
            "single-panel effective-volume and density forecast; excludes anisotropic photo-z "
            "information loss and reconstruction response"
        ),
        "absolute_panel": {
            "finite_curve_area_policy": (
                f"all finite-density scenarios use the {footprint} shell volumes"
            ),
            "target_fsky": target_fsky,
            "spherex_shell_volume_mpc_h3": spherex_shell_volume.tolist(),
            "cosmic_variance_curve": f"infinite-density {footprint} limit",
        },
        "density_scaling": {
            "formula": "R_over_R_CV = 1 + 1 / (nbar * P0)",
            "p0_h3_mpc3": float(p0_h3_mpc3),
            "density_policy": "scenario shell-mean nbar on the common SPHEREx footprint",
        },
        "geometry_policy": dict(veff_projection["geometry_policy"]),
        "scenarios": scenarios,
        "spherex_cosmic_variance": spherex_cosmic_variance,
    }


# claim figures: saved values and validation.


def recon_gain_distribution(
    profiles_full: dict,
    tracer: str,
    zmin: float,
    zmax: float,
    sigma_label: str,
    smoothing: int,
    kind: str = "1d",
    n_boot: int = 4000,
    seed: int = 20260615,
):
    """Per-mock paired reconstruction-gain distribution and its uncertainty.

    Treats the 25 mocks as 25 independent DESI-volume surveys. For each mock
    present (and Minuit-valid) both pre- and post-reconstruction, forms the
    paired gain ``g_i = (precision_pre / precision_post)_i`` where the precision
    is ``σ(α_iso)`` for ``kind='1d'`` and the 2D FoM denominator
    ``σ(α_iso)·σ(α_AP)`` for ``kind='2d'`` (so >1 means reconstruction tightened
    that mock's constraint, matching the mean-of-25 ratios plotted as points).

    Returns ``None`` if fewer than two paired mocks exist, else a dict with the
    median gain, the bootstrap 68% CI on that median (the signal-vs-noise test:
    a CI spanning 1.0 means the gain is consistent with no improvement), the
    16-84% spread of the per-mock gains (the survey-to-survey scatter), the
    fraction of mocks with ``g_i>1``, and the railed/total mock counts.
    """
    kpre = (tracer, zmin, zmax, sigma_label, smoothing, "prerecon")
    kpost = (tracer, zmin, zmax, sigma_label, smoothing, "postrecon")
    pre = profiles_full.get(kpre, {})
    post = profiles_full.get(kpost, {})
    common = sorted(set(pre) & set(post))
    gains: list[float] = []
    n_railed = 0
    for m in common:
        a, b = pre[m], post[m]
        if not (a["valid"] and b["valid"]):
            continue
        if any(x <= 0.805 or x >= 1.195 for x in (a["qiso_bf"], b["qiso_bf"])):
            n_railed += 1
        if kind == "2d":
            num = a["qiso_err"] * a["qap_err"]
            den = b["qiso_err"] * b["qap_err"]
        else:
            num, den = a["qiso_err"], b["qiso_err"]
        if np.isfinite(num) and np.isfinite(den) and num > 0 and den > 0:
            gains.append(num / den)
    g = np.asarray(gains, dtype=float)
    if g.size < 2:
        return None
    return {**_gain_summary(g, n_boot, seed), "n_railed": int(n_railed)}


def _gain_summary(gains, n_boot, seed):
    """Median of paired gains with its bootstrap 68% interval and the per-mock spread."""
    rng = np.random.default_rng(seed)
    meds = np.array(
        [np.median(rng.choice(gains, size=gains.size, replace=True)) for _ in range(n_boot)]
    )
    return {
        "median": float(np.median(gains)),
        "boot_lo": float(np.percentile(meds, 16)),
        "boot_hi": float(np.percentile(meds, 84)),
        "p16": float(np.percentile(gains, 16)),
        "p84": float(np.percentile(gains, 84)),
        "frac_gt1": float(np.mean(gains > 1.0)),
        "n": int(gains.size),
    }


def consensus_width(widths, correlation):
    """Error of the non-negative best linear unbiased combination of correlated estimates.

    The estimate covariance is the correlation times the outer product of the widths (Sanchez et
    al. 2017, arXiv:1607.03146), as the literature comparison combines the tiers' median widths.
    """
    from pzbao.fitting import nonnegative_blue_weights

    widths = np.asarray(widths, dtype=float)
    covariance = np.asarray(correlation, dtype=float) * np.outer(widths, widths)
    weights, _ = nonnegative_blue_weights(covariance)
    return float(np.sqrt(weights @ covariance @ weights))


def joint_recon_gain_distribution(
    profiles_full: dict,
    joint_rows,
    measurement_bin: str,
    tracer: str,
    zmin: float,
    zmax: float,
    smoothing: int,
    kind: str = "1d",
    n_boot: int = 4000,
    seed: int = 20260615,
):
    """Per-mock gain of the four-condition combination, paired like recon_gain_distribution.

    For each mock valid in every combined condition before and after reconstruction, the
    conditions' widths are combined with that state's measured correlation of their best fits
    (the joint record's estimate correlation) by consensus_width. The gain is the pre/post ratio
    of the combined alpha_iso width, or for ``kind='2d'`` of the product of both dilations'
    combined widths. Returns None without both joint records or fewer than two paired mocks.
    """
    records = {
        r["route"]["reconstruction_state"]: r
        for r in joint_rows
        if r["route"]["measurement_bin"] == measurement_bin and r.get("status") == "available"
    }
    if set(records) != set(STATES):
        return None
    tiers = records["prereconstruction"]["tiers"]
    if records["postreconstruction"]["tiers"] != tiers:
        raise ValueError("pre- and post-reconstruction joint records combine different conditions")
    dilations = (0,) if kind == "1d" else (0, 1)
    widths = {}
    for state, short in zip(STATES, ("prerecon", "postrecon")):
        correlation = np.asarray(records[state]["estimate_correlation"], dtype=float)
        profiles = [
            profiles_full.get((tracer, zmin, zmax, tier, smoothing, short), {}) for tier in tiers
        ]
        widths[state] = {}
        for mock in set.intersection(*(set(p) for p in profiles)):
            fits = [p[mock] for p in profiles]
            if not all(f["valid"] for f in fits):
                continue
            width = 1.0
            for k in dilations:
                values = [f[("qiso_err", "qap_err")[k]] for f in fits]
                if not all(np.isfinite(v) and v > 0 for v in values):
                    width = np.nan
                    break
                index = [
                    2 * t + k for t in range(len(tiers))
                ]  # estimates ordered (iso, AP) per tier
                width *= consensus_width(values, correlation[np.ix_(index, index)])
            widths[state][mock] = width
    pre, post = widths["prereconstruction"], widths["postreconstruction"]
    gains = np.array(
        [pre[m] / post[m] for m in sorted(set(pre) & set(post)) if np.isfinite(pre[m] / post[m])]
    )
    if gains.size < 2:
        return None
    return {**_gain_summary(gains, n_boot, seed), "tiers": list(tiers)}


def require_identity(identity):
    if not isinstance(identity, dict) or any(not identity.get(k) for k in IDENTITY):
        raise ValueError("missing source identity")
    for key, size in (("code_commit", 40), ("analysis_config_sha256", 64)):
        if not re.fullmatch(r"[0-9a-f]{%d}" % size, identity[key]):
            raise ValueError(f"invalid identity {key}")


def interval(parameter, maximization):
    """Keep raw bounds and every flag, including otherwise finite invalid intervals."""
    value = float(parameter["value"])
    if not math.isfinite(value):
        raise ValueError("non-finite fit value")
    bounds = parameter.get("minos_bounds")
    ordered = (
        bounds is not None
        and len(bounds) == 2
        and all(math.isfinite(x) for x in bounds)
        and bounds[0] <= value <= bounds[1]
        and bounds[0] < bounds[1]
    )
    flags = parameter.get("interval_flags") or {}
    optimizer_ok = (
        maximization.get("is_valid") is True
        and maximization.get("is_above_max_edm") is False
        and maximization.get("has_reached_call_limit") is False
    )
    boundary = bool(parameter.get("boundary")) or any(
        flags.get(k) is True for k in ("at_lower_limit", "at_upper_limit")
    )
    valid = ordered and all(
        flags.get(k) is True for k in ("is_valid", "lower_valid", "upper_valid")
    )
    return {
        "value": value,
        "minos_bounds": bounds,
        "interval_flags": flags,
        "maximization": maximization,
        "ordered_bounds": bool(ordered),
        "interval_valid": bool(valid),
        "boundary": boundary,
        "optimizer_ok": optimizer_ok,
        "usable": bool(valid and not boundary and optimizer_ok),
        "half_width": (bounds[1] - bounds[0]) / 2 if ordered else None,
        "interval_unavailable_reason": parameter.get("interval_unavailable_reason"),
    }


def load_recovery(sources, directory, tiers):
    provenance = sources.json(directory / "provenance.json")
    complete = sources.json(directory / "complete.json")
    if complete.get("stage2_modified") is not False:
        raise ValueError("recovery source did not preserve Stage 2")
    allowed = {item["definition"]["code_commit"] for item in provenance["source_releases"]}
    for item in provenance["source_releases"]:
        if not re.fullmatch("[0-9a-f]{64}", item["resolved_snapshot_sha256"]):
            raise ValueError("missing resolved source snapshot")
    rows = sources.json(directory / "fit-diagnostics.json")
    seen = set()
    result = []
    for row in rows:
        identity = row["identity"]
        require_identity(identity)
        require_identity(row["covariance_identity"])
        if identity["code_commit"] not in allowed:
            raise ValueError("recovery fit does not belong to a declared source release")
        if row["covariance_identity"] != provenance["preparation_identity"]:
            raise ValueError("recovery covariance differs from declared baseline")
        coords = (row["measurement_bin"], row["condition"], row["state"], row["mock"])
        actual = tuple(
            identity[k]
            for k in ("measurement_bin", "condition", "reconstruction_state", "realization_id")
        )
        if coords != actual or not identity["run_name"].startswith("primary-power3--"):
            raise ValueError("recovery route identity mismatch or non-power3 source")
        key = coords + (row["parameter"],)
        if key in seen:
            raise ValueError("duplicate recovery row")
        seen.add(key)
        result.append({**row, **interval(row, row["optimizer"]["maximization"])})
    expected = set(
        itertools.product(("LRG1", "LRG2", "LRG3"), tiers, STATES, range(25), PARAMETERS)
    )
    if seen != expected:
        raise ValueError("incomplete or unexpected registered LRG recovery cohort")
    return result, provenance


def summarize_recovery(rows, tiers):
    lookup = {
        (r["measurement_bin"], r["condition"], r["state"], r["mock"], r["parameter"]): r
        for r in rows
    }
    summary, pairs = [], []
    for bin_name, tier, parameter in itertools.product(BINS, tiers, PARAMETERS):
        for state in STATES:
            group = [
                lookup[(bin_name, tier, state, m, parameter)]
                for m in range(25)
                if (bin_name, tier, state, m, parameter) in lookup
            ]
            values = [r["value"] - 1 for r in group if r["optimizer_ok"]]
            summary.append(
                {
                    "measurement_bin": bin_name,
                    "condition": tier,
                    "parameter": parameter,
                    "state": state,
                    "n_saved": len(group),
                    "n_bestfit": len(values),
                    "n_usable_interval": sum(r["usable"] for r in group),
                    "n_boundary": sum(r["boundary"] for r in group),
                    "bias": float(np.mean(values)) if values else None,
                    "bias_sem": (
                        float(np.std(values, ddof=1) / np.sqrt(len(values)))
                        if len(values) > 1
                        else None
                    ),
                    "scatter": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                }
            )
        for mock in range(25):
            pre, post = [lookup.get((bin_name, tier, state, mock, parameter)) for state in STATES]
            if pre is None or post is None:
                continue
            usable = pre["usable"] and post["usable"]
            pairs.append(
                {
                    "measurement_bin": bin_name,
                    "condition": tier,
                    "parameter": parameter,
                    "mock": mock,
                    "post_minus_pre": post["value"] - pre["value"],
                    "usable_intervals": usable,
                    "profile_width_gain_percent": (
                        100 * (pre["half_width"] / post["half_width"] - 1) if usable else None
                    ),
                    "pre_flags": pre["interval_flags"],
                    "post_flags": post["interval_flags"],
                    "pre_boundary": pre["boundary"],
                    "post_boundary": post["boundary"],
                }
            )
    return summary, pairs


def mean_rows(sources, ready):
    from pzbao.fitting import read_profiles

    rows, registry = [], {}
    for record in ready["records"]:
        route = record["route"]
        if route["condition"] != "sigma_0.003":
            continue
        key = (route["measurement_bin"], route["reconstruction_state"])
        if key in registry or record["status"] != "ready":
            raise ValueError("duplicate or unready baseline route")
        fitting = record["current_definition"]["analysis"]["fitting"]
        if fitting["broadband_model"] != "power3" or (
            fitting["covariance"]["mean_vector_scaling"]
            != "route_covariance_over_realization_count"
        ):
            raise ValueError("baseline must use power3 and the covariance of the mean")
        registry[key] = record
        profile = record["profile"]
        if not profile["parameters"]:
            continue
        require_identity(profile["identity"])
        if any(profile["identity"].get(k) != v for k, v in route.items()):
            raise ValueError("mean profile route differs from the registered route")
        if profile["identity"].get("realization_id") is not None:
            raise ValueError("mean profile is an individual-realization fit")
        for path, digest in record["input_sha256"].items():
            sources.read(path, digest)
        sources.read(profile["source"], profile["source_sha256"])
        native = read_profiles(profile["source"], identity=profile["identity"])
        for parameter in PARAMETERS:
            p = next(p for p in native["parameters"] if p["basename"] == parameter)
            saved = next(p for p in profile["parameters"] if p["basename"] == parameter)
            if p != saved:
                raise ValueError("native mean profile differs from registered replay")
            rows.append(
                {
                    "measurement_bin": key[0],
                    "parameter": parameter,
                    "variant": "baseline" if key[1] == STATES[1] else "pre",
                    "status": "available",
                    "source": profile["source"],
                    "identity": profile["identity"],
                    "n_mocks": 25,
                    "covariance_normalization": "C/25",
                    "ess": None,
                    "ess_reason": "MINOS profile interval; no posterior draws",
                    **interval(p, native["maximization"]),
                }
            )
    if set(registry) != set(itertools.product(BINS, STATES)):
        raise ValueError("ready file must register all five bins and both states")
    return rows, registry


def broadband_pairs(sources, path, registry):
    from pzbao.diagnostics.fitting import validate_pair
    from pzbao.diagnostics.fitting import verify_components
    from pzbao.fitting import read_profiles

    fits = sources.json(path)["fits"]
    rows, pairs = [], []
    for bin_name, state in itertools.product(("LRG2", "LRG3"), STATES):
        prefix = f"LRG--sigma_0.003--baseline--{bin_name}--{state}--"
        power, spline = fits[prefix + "power3"], fits[prefix + "current"]
        reference = registry[(bin_name, state)]["profile"]
        if power["profile_source"]["sha256"] != reference["source_sha256"]:
            raise ValueError("broadband pair is not the replayed baseline native fit")
        validate_pair(power, spline)
        if power["origins"] != spline["origins"]:
            raise ValueError("broadband pairs have different measurement origins")
        if sorted(o["identity"]["realization_id"] for o in power["origins"]) != list(range(25)):
            raise ValueError("broadband pair is not the registered 25-mock mean")
        for fit in (power, spline):
            require_identity(fit["identity"])
            if any(
                fit["identity"].get(k) != v
                for k, v in {
                    "measurement_bin": bin_name,
                    "reconstruction_state": state,
                    "condition": "sigma_0.003",
                    "tracer": "LRG",
                    "realization_id": None,
                }.items()
            ):
                raise ValueError("broadband route mismatch")
            sources.read(fit["profile_source"]["path"], fit["profile_source"]["sha256"])
            native = read_profiles(fit["profile_source"]["path"], identity=fit["identity"])
            if (
                native["parameters"] != fit["parameters"]
                or native["maximization"] != fit["maximization"]
            ):
                raise ValueError("saved broadband parameters or optimizer differ from native fit")
            verify_components(fit)
        pairs.append({"measurement_bin": bin_name, "state": state, "power3": power, "pcs2": spline})
        if state == STATES[1]:
            for parameter in PARAMETERS:
                p = next(p for p in spline["parameters"] if p["basename"] == parameter)
                rows.append(
                    {
                        "measurement_bin": bin_name,
                        "parameter": parameter,
                        "variant": "pcs2",
                        "status": "available",
                        "source": spline["profile_source"]["path"],
                        "identity": spline["identity"],
                        "n_mocks": 25,
                        "covariance_normalization": "C/25",
                        "ess": None,
                        "ess_reason": "MINOS profile interval; no posterior draws",
                        **interval(p, spline["maximization"]),
                    }
                )
    return rows, pairs


def robustness_grid(rows):
    lookup = {}
    expected = set(itertools.product(BINS, PARAMETERS, dict(ROBUSTNESS)))
    for row in rows:
        key = tuple(row[k] for k in ("measurement_bin", "parameter", "variant"))
        if key not in expected or key in lookup:
            raise ValueError("unexpected or duplicate robustness row")
        lookup[key] = row
    return [
        lookup.get(
            (b, p, v),
            {
                "measurement_bin": b,
                "parameter": p,
                "variant": v,
                "status": "missing",
                "reason": DEPENDENCIES[v],
            },
        )
        for b, p, v in itertools.product(BINS, PARAMETERS, dict(ROBUSTNESS))
    ]


def normalized_density(counts, edges):
    counts, edges = np.asarray(counts, dtype=float), np.asarray(edges, dtype=float)
    if len(edges) != len(counts) + 1 or np.any(np.diff(edges) <= 0):
        raise ValueError("invalid redshift cell axes")
    if np.any(~np.isfinite(counts)) or np.any(counts < 0) or counts.sum() <= 0:
        raise ValueError("invalid radial counts")
    return counts / counts.sum() / np.diff(edges)


def area_projection(area_deg2, target_fsky):
    full_sky = 4 * np.pi * (180 / np.pi) ** 2
    if not 0 < area_deg2 <= full_sky or not 0 < target_fsky <= 1:
        raise ValueError("invalid sky area")
    factor = target_fsky * full_sky / area_deg2
    return {
        "attained_fsky": area_deg2 / full_sky,
        "target_fsky": target_fsky,
        "area_factor": factor,
        "sigma_factor": factor**-0.5,
        "gaussian_likelihood_fom_factor": factor,
        "relative_reconstruction_gain_factor": 1.0,
        "interpretation": (
            "Conditional Gaussian likelihood scaling; fixed density, response and shape; "
            "ignores prior limitation and added survey systematics."
        ),
    }


def _preparation_stage(sources, root):
    """The stage holding a run's selection preparation: inputs (interface 1) or select (2)."""
    run = yaml.safe_load(sources.read(Path(root) / "run.yaml"))
    return "select" if run.get("interface_version") == 2 else "inputs"


def selection_rows(sources, root, config, expected_identity, target_fsky):
    from pzbao.catalog import build_mock_cosmology, comoving_shell_volume

    cosmology = build_mock_cosmology(config)
    density, radial, area = [], [], []
    stage = _preparation_stage(sources, root)
    for tracer, spec in config["tracers"].items():
        directory = root / stage / "units" / tracer
        identity = yaml.safe_load(sources.read(directory / "result.yaml"))
        if any(identity[k] != expected_identity[k] for k in IDENTITY):
            raise ValueError("preparation identity differs from the fitted baseline")
        prep = sources.json(directory / "preparation.json")
        if prep["tracer"] != tracer or prep["realization_ids"] != list(range(25)):
            raise ValueError("preparation tracer or realization mismatch")
        areas = spec["footprint"]["area_deg2"]
        area.append(
            {
                "tracer": tracer,
                "cap_area_deg2": areas,
                **area_projection(sum(areas.values()), target_fsky),
            }
        )
        for tier in config["selection"]["tiers"]:
            caps = prep["conditions"][tier["label"]]
            if set(caps) != set(areas):
                raise ValueError("preparation cap mismatch")
            for shell in spec["measurement_bins"]:
                name = shell["name"]
                count = sum(c["counts"][name] for c in caps.values())
                volume = sum(c["volumes_mpc_h3"][name] for c in caps.values())
                if not math.isfinite(volume) or volume <= 0 or count <= 0:
                    raise ValueError("invalid selected count or shell volume")
                target = (tier["target_nbar_h3_mpc3"] or {}).get(name)
                density.append(
                    {
                        "measurement_bin": name,
                        "condition": tier["label"],
                        "count": count,
                        "volume_mpc_h3": volume,
                        "attained_nbar_h3_mpc3": count / volume,
                        "target_nbar_h3_mpc3": target,
                        "attained_over_target": count / volume / target if target else None,
                        "target_caveat": (
                            "constructed quarter-density proxy"
                            if tier["label"] == "sigma_0.001"
                            else (
                                "lowest target shell 0-0.2 approximates BGS1 0.1-0.2"
                                if name == "BGS1"
                                else None
                            )
                        ),
                    }
                )
            if tier["label"] != "sigma_0.003":
                continue
            first = next(iter(caps.values()))["radial"]
            edges = np.asarray(first["z_edges"])
            counts = np.zeros(len(edges) - 1)
            for cap, content in caps.items():
                np.testing.assert_array_equal(content["radial"]["z_edges"], edges)
                if f"{tracer}|tier=sigma_0.003|" not in content["radial"]["sample_identity"]:
                    raise ValueError("radial selection identity mismatch")
                counts += np.asarray(content["radial"]["mean_counts"])
            for shell in spec["measurement_bins"]:
                mask = (edges[:-1] >= shell["zmin"] - 1e-10) & (edges[1:] <= shell["zmax"] + 1e-10)
                local_edges = np.r_[edges[:-1][mask], edges[1:][mask][-1]]
                if not np.allclose(local_edges[[0, -1]], [shell["zmin"], shell["zmax"]]):
                    raise ValueError("radial cells do not cover the configured bin")
                target_counts = tier["target_nbar_h3_mpc3"][shell["name"]] * comoving_shell_volume(
                    cosmology, local_edges, sum(areas.values())
                )
                radial.append(
                    {
                        "measurement_bin": shell["name"],
                        "condition": tier["label"],
                        "z_edges": local_edges.tolist(),
                        "attained_pdf": normalized_density(counts[mask], local_edges).tolist(),
                        "target_pdf": normalized_density(target_counts, local_edges).tolist(),
                        "target_model": (
                            "constant comoving number density within the published shell; "
                            "no inferred fine structure"
                        ),
                    }
                )
    return density, radial, area


# saved comparisons: saved values and validation.


def add_primary_means(sources, ready, directories):
    """Add explicitly supplied completed primary fits; never replace an existing fit."""
    from pzbao.fitting import read_profiles

    result = deepcopy(ready)
    for directory in directories:
        attempt = sources.json(directory / "attempt.json")
        complete = sources.json(directory / "complete.json")
        row = next(r for r in result["records"] if r["index"] == complete["index"])
        expected = {**row["identity"], "realization_id": None}
        if (
            complete["mode"] != "fit"
            or attempt["mode"] != "fit"
            or attempt["index"] != complete["index"]
            or row["profile"]["parameters"]
            or complete["identity"] != expected
            or attempt["current_definition"] != row["current_definition"]
            or attempt["input_sha256"] != row["input_sha256"]
        ):
            raise ValueError("incompatible or duplicate primary mean")
        for path, digest in attempt["input_sha256"].items():
            sources.read(path, digest)
        path = directory / "profiles.npy"
        sources.read(path)
        native = read_profiles(path, identity=expected)
        saved = sources.json(directory / "fit.json")
        if (
            saved["identity"] != expected
            or native["parameters"] != saved["parameters"]
            or native["maximization"] != saved["optimizer"]["maximization"]
            or saved["covariance_identity"] != row["model"]["covariance_identity"]
        ):
            raise ValueError("primary mean native readback mismatch")
        row["profile"] = {
            "parameters": native["parameters"],
            "identity": expected,
            "source": str(path),
            "source_sha256": sources.digests[str(resolve_saved_path(path).resolve())],
            "unavailable_reason": None,
        }
    return result


def load_fom(sources, directory, ready):
    payload = sources.json(directory / "figure-data.json")
    complete = sources.json(directory / "complete.json")
    expected = {r["index"]: r for r in ready["records"]}
    records = payload["records"]
    if len(records) != len(expected) or {r["index"] for r in records} != set(expected):
        raise ValueError("incomplete or duplicate FoM inventory")
    for row in records:
        reference = expected[row["index"]]
        if row["route"] != reference["route"]:
            raise ValueError("FoM route mismatch")
        if row["status"] != "available":
            continue
        if (
            row["identity"] != reference["identity"]
            or row["covariance_identity"] != reference["model"]["covariance_identity"]
        ):
            raise ValueError("FoM identity mismatch")
        for path, digest in row["input_sha256"].items():
            sources.read(path, digest)
        attempt = sources.json(Path(row["source"]) / "attempt.json")
        if attempt["current_definition"] != reference["current_definition"]:
            raise ValueError("FoM scientific definition mismatch")
        matrix = np.asarray(row["covariance"])
        if matrix.shape != (2, 2) or not np.isfinite(matrix).all():
            raise ValueError("invalid joint posterior covariance")
        np.testing.assert_allclose(matrix, matrix.T, rtol=0, atol=1e-15)
        value = 1 / np.prod(np.diag(np.linalg.cholesky(matrix)))
        np.testing.assert_allclose(row["fom"], value, rtol=1e-12)
    if complete["available_routes"] != sum(r["status"] == "available" for r in records):
        raise ValueError("FoM completion count mismatch")
    return records


def robustness_fits(sources, directories, registry):
    """Admit native fitting-only variants against the registered primary inputs."""
    from pzbao.config import analysis_config_sha256, validate_analysis_config
    from pzbao.diagnostics.fitting import (
        comparison_config,
        paper_flat_prior as flat_prior,
        validate_comparison_route as validate_route,
    )
    from pzbao.fitting import read_profiles

    config_path = Path(__file__).resolve().parents[1] / "config/analysis.yaml"
    defaults = yaml.safe_load(sources.read(config_path))
    validate_analysis_config(defaults)
    settings = defaults["fitting"]["paper_comparisons"]
    rows = []
    for directory in directories:
        attempt = sources.json(directory / "attempt.json")
        complete = sources.json(directory / "complete.json")
        definition, identity = attempt["definition"], attempt["identity"]
        variant = definition["variant"]
        route = {
            k: identity[k]
            for k in ("tracer", "measurement_bin", "condition", "reconstruction_state")
        }
        validate_route(route, settings, variant)
        reference = registry[(route["measurement_bin"], route["reconstruction_state"])]
        require_identity(identity)
        expected_name = (
            "paper-"
            + variant
            + "--"
            + "--".join(route[k] for k in ("measurement_bin", "condition", "reconstruction_state"))
        )
        if (
            attempt["mode"] != "fit"
            or complete["mode"] != "fit"
            or attempt["index"] != reference["index"]
            or complete["index"] != reference["index"]
            or complete["identity"] != identity
            or complete["variant"] != variant
            or identity["realization_id"] is not None
            or identity["run_name"] != expected_name
            or identity["code_commit"] != definition["producer_commit"]
            or identity["analysis_config_sha256"] != analysis_config_sha256(definition)
            or definition["baseline"] != reference["current_definition"]
            or definition["comparison"] != settings
            or attempt["input_sha256"] != reference["input_sha256"]
        ):
            raise ValueError("incompatible robustness fit definition or identity")
        candidate = definition["analysis"]
        validate_analysis_config(candidate)
        if candidate != comparison_config(
            reference["current_definition"]["analysis"], candidate, variant
        ):
            raise ValueError("robustness fit changed unrelated settings")
        for path, digest in attempt["input_sha256"].items():
            sources.read(path, digest)
        check = sources.json(directory / "check.json")
        saved = sources.json(directory / "fit.json")
        damping = reference["model"]["damping_prior"]
        if variant == "flat":
            damping = flat_prior(damping, settings)
        if (
            saved["identity"] != identity
            or saved["damping_prior"] != damping
            or check["damping_prior"] != damping
            or check["realization_ids"] != list(range(25))
            or check["covariance_identity"] != reference["model"]["covariance_identity"]
            or saved["covariance_identity"] != reference["model"]["covariance_identity"]
        ):
            raise ValueError("robustness fit changed covariance, priors or paired mock IDs")
        path = directory / "profiles.npy"
        sources.read(path)
        native = read_profiles(path, identity=identity)
        if (
            native["parameters"] != saved["parameters"]
            or native["maximization"] != saved["optimizer"]["maximization"]
        ):
            raise ValueError("robustness native profile differs from saved fit")
        for name in PARAMETERS:
            p = next(p for p in native["parameters"] if p["basename"] == name)
            rows.append(
                dict(
                    measurement_bin=route["measurement_bin"],
                    parameter=name,
                    variant=variant,
                    status="available",
                    source=str(path),
                    identity=identity,
                    n_mocks=25,
                    covariance_normalization="C/25",
                    ess=None,
                    ess_reason="MINOS profile interval; no posterior draws",
                    **interval(p, native["maximization"]),
                )
            )
    return rows


def fom_pairs(records, tiers):
    lookup = {}
    for row in records:
        key = tuple(
            row["route"][k] for k in ("measurement_bin", "condition", "reconstruction_state")
        )
        if key in lookup:
            raise ValueError("duplicate FoM route")
        lookup[key] = row
    pairs = []
    for name, tier in itertools.product(BINS, tiers):
        members = [lookup.get((name, tier, state)) for state in STATES]
        available = [r if r and r["status"] == "available" else None for r in members]
        values = [r["fom"] if r else None for r in available]
        if any(v is not None and (not np.isfinite(v) or v <= 0) for v in values):
            raise ValueError("invalid absolute FoM")
        pairs.append(
            {
                "measurement_bin": name,
                "condition": tier,
                "pre": values[0],
                "post": values[1],
                "gain_percent": 100 * (values[1] / values[0] - 1) if all(available) else None,
                "profile_flagged": any(r["profile_flagged"] for r in available if r),
                "sampling_limited": any(r["sampling_limits"] for r in available if r),
                "covariance_warning": any(
                    r["covariance_validity"].get("structural_test_warnings") for r in available if r
                ),
            }
        )
    return pairs


def density_means(sources, root, ready):
    from pzbao.config import analysis_config_sha256
    from pzbao.fitting import read_profiles

    root = resolve_saved_path(root)
    definition = sources.json(root / "definition.json")
    config = definition["analysis"]
    rows = []
    for reference in ready["records"]:
        route = reference["route"]
        if route["measurement_bin"] not in ("LRG2", "LRG3") or route["condition"] != "sigma_0.001":
            continue
        baseline = reference["current_definition"]["analysis"]
        for key in ("fitting", "measurement", "physics", "photoz_response"):
            if config.get(key) != baseline.get(key):
                raise ValueError("density comparison changes unrelated " + key)
        name = "--".join(route[k] for k in ("measurement_bin", "condition", "reconstruction_state"))
        directory = root / "fit/units" / (name + "--mean")
        if (directory / "failure.json").exists():
            raise ValueError("density fit retains failure")
        complete = sources.json(directory / "complete.json")
        saved = sources.json(directory / "fit.json")
        result = yaml.safe_load(sources.read(directory / "result.yaml"))
        identity = saved["identity"]
        require_identity(identity)
        if (
            complete["definition"] != definition
            or complete["identity"] != identity
            or identity["analysis_config_sha256"] != analysis_config_sha256(definition)
            or any(identity.get(k) != v for k, v in route.items())
            or identity["realization_id"] is not None
            or any(result[k] != identity[k] for k in IDENTITY)
        ):
            raise ValueError("density mean identity mismatch")
        if saved["covariance_identity"] != {
            k: definition["covariance_source"][k] for k in IDENTITY
        }:
            raise ValueError("density covariance identity mismatch")
        for key in (
            "damping_prior",
            "z_eff",
            "reconstruction_inputs",
            "fitting_growth_rate",
            "ap_mode",
        ):
            if saved[key] != reference["model"][key]:
                raise ValueError("density comparison changes reference " + key)
        covdir = (
            resolve_saved_path(definition["covariance_source"]["root"]) / "covariance/routes" / name
        )
        meta = sources.json(covdir / "covariance.json")
        if (
            any(meta[k] != saved["covariance_identity"][k] for k in IDENTITY)
            or meta["route"] != route
            or meta["realization_ids"] != list(range(25))
        ):
            raise ValueError("density covariance route mismatch")
        sources.read(covdir / "covariance.npz")
        with np.load(covdir / "covariance.npz", allow_pickle=False) as arrays:
            np.testing.assert_array_equal(arrays["realization_ids"], np.arange(25))
            np.testing.assert_allclose(
                arrays["mean_covariance"], arrays["covariance"] / 25, rtol=1e-14
            )
        for variant, profile in (
            ("baseline", reference["profile"]),
            ("doubled", {"source": str(directory / "profiles.npy"), "identity": identity}),
        ):
            if variant == "baseline":
                for path, digest in reference["input_sha256"].items():
                    sources.read(path, digest)
            sources.read(profile["source"], profile.get("source_sha256"))
            native = read_profiles(profile["source"], identity=profile["identity"])
            expected = (
                reference["profile"]["parameters"] if variant == "baseline" else saved["parameters"]
            )
            if native["parameters"] != expected:
                raise ValueError("density native parameters differ from saved fit")
            if (
                variant == "doubled"
                and native["maximization"] != saved["optimizer"]["maximization"]
            ):
                raise ValueError("density native optimizer differs")
            for parameter in PARAMETERS:
                p = next(p for p in native["parameters"] if p["basename"] == parameter)
                rows.append(
                    {
                        **route,
                        "parameter": parameter,
                        "variant": variant,
                        "identity": profile["identity"],
                        "source": profile["source"],
                        "covariance_identity": (
                            saved if variant == "doubled" else reference["model"]
                        )["covariance_identity"],
                        "attained_nbar_h3_mpc3": (
                            saved if variant == "doubled" else reference["model"]
                        )["attained_density_h3_mpc3"],
                        "covariance_diagnostics": (
                            meta["diagnostics"]
                            if variant == "doubled"
                            else reference["covariance_validity"]
                        ),
                        "covariance_normalization": "C/25",
                        "n_mocks": 25,
                        **interval(p, native["maximization"]),
                    }
                )
    if len(rows) != 16:
        raise ValueError("density comparison requires all four matched mean routes")
    return rows


# Optional inference choices that do not change the shared power3 fitting definition.
_OPTIONAL_FITTING = ("paper_comparisons",)
_OPTIONAL_DAMPING = ("common_flat_sensitivity", "applied_prior", "reference_smoothing_mpc_h")


def _comparable_fitting(config):
    fitting = deepcopy(config["fitting"])
    for key in _OPTIONAL_FITTING:
        fitting.pop(key, None)
    for key in _OPTIONAL_DAMPING:
        fitting["damping"].pop(key, None)
    return fitting


def matched_density_routes(config):
    """The density successor's routes: its matched bins and tiers in both reconstruction states."""
    workload = config["analysis"]["workloads"]["density"]
    return tuple(
        ("LRG", name, condition, state)
        for name in workload["measurement_bins"]
        for condition in workload["sigma_z_tiers"]
        for state in workload["reconstruction_states"]
    )


def matched_density_context(plan):
    """Densities, effective volumes and flags of one matched tier and bin, carried by every row."""
    return {
        "spherex_nbar_h3_mpc3": plan["n_spherex_h3_mpc3"],
        "matched_nbar_h3_mpc3": plan["n_matched_h3_mpc3"],
        "attained_nbar_h3_mpc3": plan["n_attained_h3_mpc3"],
        "primary_nbar_h3_mpc3": plan["n_primary_h3_mpc3"],
        "veff_target_mpc_h3": plan["veff_target_mpc_h3"],
        "veff_attained_mpc_h3": plan["veff_attained_mpc_h3"],
        "veff_primary_mpc_h3": plan["veff_primary_mpc_h3"],
        "veff_fraction": plan["veff_fraction"],
        "bounded": plan["bounded"],
        "selected_counts": {cap: row["selected"] for cap, row in plan["caps"].items()},
    }


def matched_density_fits(root):
    """Every configured matched-density fit (25 realizations and the mean per route).

    Rows come from the density successor's own eight-command layout, each carrying its route's
    matching record (SPHEREx, matched and attained densities, target and attained effective
    volume, bounded flag); failed or missing fits stay visible.
    """
    from pzbao.run import load_run

    run = load_run(root)
    record = run["config"]["selection"].get("density_run")
    if record is None or run["metadata"]["workload"] != "density":
        raise ValueError("matched density readers require a density successor run")
    matching = record["handoff"]["definition"]["matching"]
    rows = _fit_rows(root, variant="matched", routes=set(matched_density_routes(run["config"])))
    for row in rows:
        plan = matching["conditions"][row["condition"]][row["measurement_bin"]]
        row.update(matched_density_context(plan))
    return rows, matching


def _route_summary(rows):
    """Ensemble bias, scatter and interval counts with failures and contact kept visible."""
    individual = [r for r in rows if r["realization_id"] is not None]
    complete = [r for r in individual if r.get("status") == "complete"]
    values = [r["value"] - 1.0 for r in complete if r["optimizer_ok"]]
    mean = next((r for r in rows if r["realization_id"] is None), None)
    return {
        "n_saved": len(complete),
        "n_failed": len(individual) - len(complete),
        "n_bestfit": len(values),
        "n_usable_interval": sum(r["usable"] for r in complete),
        "n_boundary": sum(r["boundary"] for r in complete),
        "bias": float(np.mean(values)) if values else None,
        "bias_sem": (
            float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else None
        ),
        "scatter": float(np.std(values, ddof=1)) if len(values) > 1 else None,
        "mean_fit_value": (
            None if mean is None or mean.get("status") != "complete" else mean["value"]
        ),
        "mean_fit_half_width": (
            None if mean is None or mean.get("status") != "complete" else mean["half_width"]
        ),
        "mean_fit_usable": bool(mean and mean.get("status") == "complete" and mean["usable"]),
    }


def matched_density_comparison(density_root, primary_root):
    """Pair the matched LRG2/3 tiers with the per-bin primary's same routes.

    Both runs must share every scientific setting except the density definition, so paired
    differences reflect the raised density alone. Besides the realization-by-realization pairs
    and the mean-of-25 summaries, each single-realization matched fit is set against the
    mean-of-25 primary fit of the same route: both reach a large effective volume by different
    routes, and their width ratio against the square root of the effective-volume ratio measures
    how much of the precision the Gaussian two-point budget predicts.

    The supplied primary must be the handoff's recorded source (root, run name, code commit
    and configuration digest): another run with the same configuration does not share the
    handoff's source objects, so it is refused before any of its fits are read. From revision
    2.20 both sides may instead be unified-prior refits: the primary refit's measurement run is
    then the recorded source, and a refit never pairs with the other prior family.
    """
    from pzbao.config import UNIFIED_PRIOR
    from pzbao.config.density import source_identity
    from pzbao.run import load_run

    matched, matching = matched_density_fits(density_root)
    density, primary = load_run(density_root), load_run(primary_root)
    families = {run["metadata"].get("inference_definition") for run in (density, primary)}
    if families not in ({None}, {UNIFIED_PRIOR}):
        raise ValueError("the matched density and the primary use different prior families")
    measurement = (
        load_run(primary["metadata"]["source_run"]["root"])
        if families == {UNIFIED_PRIOR}
        else primary
    )
    if (
        "density_run" in primary["config"]["selection"]
        or primary["metadata"]["workload"] != "full_grid"
        or "source_run" in measurement["metadata"]
    ):
        raise ValueError("the matched density compares against the per-bin primary measurement")
    comparable = _comparable_core(density["config"])
    comparable["selection"].pop("density_run")
    if comparable != _comparable_core(primary["config"]):
        raise ValueError("matched density and per-bin primary differ beyond the density definition")
    recorded = density["config"]["selection"]["density_run"]["handoff"]["definition"]["source"]
    supplied = source_identity(measurement)
    if supplied != recorded:
        differing = sorted(
            key for key in set(recorded) | set(supplied) if recorded.get(key) != supplied.get(key)
        )
        raise ValueError(
            "the supplied per-bin primary is not the handoff's recorded source; differing fields: "
            + ", ".join(
                f"{key} {supplied.get(key)!r} != {recorded.get(key)!r}" for key in differing
            )
        )
    routes = matched_density_routes(density["config"])
    realizations = len(primary["config"]["analysis"]["mock_ids"])
    rows = matched + _fit_rows(primary_root, variant="primary", routes=set(routes))
    summary, pairs, single = [], [], []
    for (_, name, condition, state), parameter in itertools.product(routes, PARAMETERS):
        plan = matching["conditions"][condition][name]
        route = {
            "measurement_bin": name,
            "condition": condition,
            "reconstruction_state": state,
            "parameter": parameter,
        }
        selected = {
            variant: [
                r
                for r in rows
                if r["variant"] == variant and all(r[key] == value for key, value in route.items())
            ]
            for variant in ("primary", "matched")
        }
        for variant, group in selected.items():
            summary.append(
                {
                    **route,
                    "variant": variant,
                    **_route_summary(group),
                    **matched_density_context(plan),
                }
            )
        lookup = {
            variant: {r["realization_id"]: r for r in group if r.get("status") == "complete"}
            for variant, group in selected.items()
        }
        mean = lookup["primary"].get(None)
        veff_mean_primary = realizations * plan["veff_primary_mpc_h3"]
        for mock in range(realizations):
            base, dense = lookup["primary"].get(mock), lookup["matched"].get(mock)
            if dense is not None and base is not None:
                usable = base["usable"] and dense["usable"]
                pairs.append(
                    {
                        **route,
                        "mock": mock,
                        "matched_minus_primary": dense["value"] - base["value"],
                        "usable_intervals": usable,
                        "width_ratio_matched_over_primary": (
                            dense["half_width"] / base["half_width"] if usable else None
                        ),
                    }
                )
            if dense is not None and mean is not None:
                usable = mean["usable"] and dense["usable"]
                single.append(
                    {
                        **route,
                        "mock": mock,
                        "usable_intervals": usable,
                        "width_ratio_single_matched_over_mean_primary": (
                            dense["half_width"] / mean["half_width"] if usable else None
                        ),
                        "veff_single_matched_mpc_h3": plan["veff_attained_mpc_h3"],
                        "veff_mean_primary_mpc_h3": veff_mean_primary,
                        "expected_sqrt_veff_ratio": math.sqrt(
                            veff_mean_primary / plan["veff_attained_mpc_h3"]
                        ),
                    }
                )
    return {
        "rows": rows,
        "summary": summary,
        "pairs": pairs,
        "single_vs_mean": single,
        "matching": matching,
    }


# Readers for eight-command runs: robustness, smoothing scan and transfer comparisons.

_ROUTE_KEYS = ("tracer", "measurement_bin", "condition", "reconstruction_state")


def _comparable_core(config):
    """Scientific definition shared by every robustness variant and radius sibling.

    Removes only what a registered variant may change (broadband, nuisance directions,
    BGS surface, finite-flat scope, radius, convention, run markers) and the optional
    blocks added since the core measurement that do not alter these routes.
    """
    value = deepcopy(config)
    value.pop("schema_version", None)
    value.pop("reporting", None)
    value["analysis"].pop("workloads", None)
    fitting = value["fitting"]
    for key in ("broadband_model", "fit_surfaces", "paper_comparisons", "core_fit_check"):
        fitting.pop(key, None)
    for key in ("varied_basenames_by_tracer", "allowed_basenames_by_model"):
        fitting["linear_nuisance"].pop(key, None)
    for key in _OPTIONAL_DAMPING:
        fitting["damping"].pop(key, None)
    recon = value["measurement"]["reconstruction"]
    for key in list(recon):
        if key.startswith("exploratory_") or key in (
            "smoothing_values_mpc_h",
            "allowed_conventions",
            "sensitivity_run",
            "transfer_run",
        ):
            recon.pop(key)
    value["selection"]["boundary_support"].pop("experiment_design", None)
    return value


def _fit_rows(root, *, variant, routes=None):
    """Keep every configured fit unit, including unpublished or failed fits."""
    from pzbao.fitting import read_profiles
    from pzbao.run import load_run, stage_units

    run = load_run(root)
    rows = []
    for unit in stage_units(
        "fit",
        run["config"],
        run["metadata"]["workload"],
        inference_definition=run["metadata"].get("inference_definition"),
    ):
        key = tuple(unit[k] for k in _ROUTE_KEYS)
        if routes is not None and key not in routes:
            continue
        path = run["root"] / "fit" / "units" / unit["unit_id"] / "profiles.npy"
        identity = {
            **dict(zip(_ROUTE_KEYS, key)),
            "realization_id": unit["realization_id"],
            **{k: run["metadata"][k] for k in IDENTITY},
        }
        if not path.exists():
            rows.extend(
                {
                    **dict(zip(_ROUTE_KEYS, key)),
                    "realization_id": unit["realization_id"],
                    "parameter": parameter,
                    "variant": variant,
                    "status": "missing",
                    "source": str(path),
                    "value": None,
                    "half_width": None,
                    "optimizer_ok": False,
                    "usable": False,
                    "boundary": False,
                    "interval_unavailable_reason": "fit unit has no published native profile",
                }
                for parameter in PARAMETERS
            )
            continue
        # Malformed or incompatible published profiles still fail closed.
        record = read_profiles(path, identity=identity)
        for parameter in PARAMETERS:
            p = next((p for p in record["parameters"] if p["basename"] == parameter), None)
            if p is None or p.get("fixed"):
                continue
            rows.append(
                {
                    **dict(zip(_ROUTE_KEYS, key)),
                    "realization_id": unit["realization_id"],
                    "parameter": parameter,
                    "variant": variant,
                    "status": "complete",
                    "source": str(path),
                    **interval(p, record["maximization"]),
                }
            )
    return rows


def robustness_rows_from_runs(
    baseline, flat=None, pcs2_lrg=None, pcs2_bgs=None, reciso=None, r25=None
):
    """Mean-fit robustness rows for the sigma_z 0.003 post-reconstruction display.

    The baseline is the core power3 refit or, from revision 2.20, the unified-prior refit of
    the per-bin primary; it also supplies the pre-reconstruction row. Each variant must share
    the baseline's scientific core and its prior family, and differ only in its own registered
    change. Unavailable variants stay missing in robustness_grid.
    """
    from pzbao.config import (
        POWER3_FLAT,
        PRIMARY_POWER3,
        UNIFIED_FLAT,
        UNIFIED_PCS2,
        UNIFIED_PRIOR,
    )
    from pzbao.run import load_run

    reference = load_run(baseline)
    definition = reference["metadata"].get("inference_definition")
    if definition not in (PRIMARY_POWER3, UNIFIED_PRIOR):
        raise ValueError("the robustness baseline is the core power3 refit or the unified refit")
    unified = definition == UNIFIED_PRIOR
    scope = load_analysis_config_scope()
    core = _comparable_core(reference["config"])
    radius = reference["config"]["measurement"]["reconstruction"]["smoothing_values_mpc_h"]

    def routes(state, tracers=("BGS_BRIGHT-21.5", "LRG")):
        return {
            (tracer, name, scope["condition"], state)
            for name in scope["measurement_bins"]
            for tracer in tracers
        }

    def check(root, label, expectation):
        run = load_run(root)
        if _comparable_core(run["config"]) != core:
            raise ValueError(f"{label} changes settings beyond its registered variation")
        expectation(run)
        return root

    def family_check(run, label, expected):
        # A unified baseline pairs only with unified refits; the power3 baseline keeps the
        # measurement runs' own fits for RecIso and R25 (expected None).
        if run["metadata"].get("inference_definition") != expected:
            raise ValueError(f"the {label} row requires the baseline's prior family ({expected})")

    def flat_check(run):
        family_check(run, "flat", UNIFIED_FLAT if unified else POWER3_FLAT)

    def pcs2_check(run):
        if run["config"]["fitting"]["broadband_model"] != "pcs2":
            raise ValueError("the PCS2 row requires a PCS2 fit")
        if unified:
            family_check(run, "PCS2", UNIFIED_PCS2)

    def reciso_check(run):
        recon = run["config"]["measurement"]["reconstruction"]
        if recon["allowed_conventions"] != ["reciso"] or recon["smoothing_values_mpc_h"] != radius:
            raise ValueError("the RecIso row requires RecIso at the reference radius")
        family_check(run, "RecIso", UNIFIED_PRIOR if unified else None)

    def r25_check(run):
        recon = run["config"]["measurement"]["reconstruction"]
        # The unified prior is one table per bin with a radius offset, so the frozen scan
        # definition's reference-centre key plays no part in a unified R25 row.
        centre_ok = unified or (
            run["config"]["fitting"]["damping"].get("reference_smoothing_mpc_h") == radius[0]
        )
        if (
            recon["smoothing_values_mpc_h"] != [25.0]
            or recon["allowed_conventions"] != ["recsym"]
            or not centre_ok
        ):
            raise ValueError("the R25 row requires RecSym at 25 Mpc/h with reference centres")
        family_check(run, "R25", UNIFIED_PRIOR if unified else None)

    post, pre = routes("postreconstruction"), routes("prereconstruction")
    rows = _fit_rows(baseline, variant="baseline", routes=post)
    rows += _fit_rows(baseline, variant="pre", routes=pre)
    for label, root, expectation, selected in (
        ("flat", flat, flat_check, post),
        ("pcs2", pcs2_lrg, pcs2_check, routes("postreconstruction", ("LRG",))),
        ("pcs2", pcs2_bgs, pcs2_check, routes("postreconstruction", ("BGS_BRIGHT-21.5",))),
        ("reciso", reciso, reciso_check, post),
        ("r25", r25, r25_check, post),
    ):
        if root is not None:
            rows += _fit_rows(check(root, label, expectation), variant=label, routes=selected)
    means = []
    n_mocks = len(
        reference["config"]["analysis"]["workloads"][reference["metadata"]["workload"]]["mock_ids"]
    )
    for row in rows:
        if row["realization_id"] is not None:
            continue
        means.append(
            {
                **row,
                "status": "available" if row["status"] == "complete" else "missing",
                "n_mocks": n_mocks,
            }
        )
    return means, rows


def load_analysis_config_scope():
    from pzbao.config import load_analysis_config

    return load_analysis_config()["fitting"]["paper_comparisons"]


def radius_scan_rows(scan_roots):
    """Per radius, tier, bin and parameter: ensemble recovery, intervals and joint FoM."""
    from pzbao.config import CORE_SCAN_INFERENCE, UNIFIED_PRIOR
    from pzbao.config.schema_validation import PER_BIN_SCHEMA_VERSIONS
    from pzbao.diagnostics.sampling import joint_summary
    from pzbao.run import load_run

    if len({str(root) for root in scan_roots}) != len(scan_roots):
        raise ValueError("smoothing-scan runs must have distinct radii")
    runs = {root: load_run(root) for root in scan_roots}
    # Revision 2.18 read per-bin radii from their own-radius centre refits; revision 2.20 reads
    # them from the unified-prior refits. The frozen scan runs pin the centre to the reference
    # radius and are never read for a per-bin curve. One curve never mixes definitions.
    definitions = {run["metadata"].get("inference_definition") for run in runs.values()}
    per_bin = {run["config"]["schema_version"] in PER_BIN_SCHEMA_VERSIONS for run in runs.values()}
    if len(definitions) != 1 or not definitions <= {None, *CORE_SCAN_INFERENCE, UNIFIED_PRIOR}:
        raise ValueError("smoothing-scan runs mix damping-centre definitions")
    if True in per_bin and definitions not in (set(CORE_SCAN_INFERENCE), {UNIFIED_PRIOR}):
        raise ValueError(
            "per-bin scan radii are read from their own-radius centre refits or unified refits"
        )
    cores = {str(root): _comparable_core(run["config"]) for root, run in runs.items()}
    if len({json.dumps(core, sort_keys=True, default=str) for core in cores.values()}) != 1:
        raise ValueError("smoothing-scan runs differ beyond their radius")
    radii = {
        root: run["config"]["measurement"]["reconstruction"]["smoothing_values_mpc_h"][0]
        for root, run in runs.items()
    }
    if len(set(radii.values())) != len(radii):
        raise ValueError("smoothing-scan runs must have distinct radii")
    references = {
        run["config"]["measurement"]["reconstruction"]["exploratory_smoothing_scan"][
            "reference_smoothing_mpc_h"
        ]
        for run in runs.values()
    }
    if len(references) != 1:
        raise ValueError("smoothing-scan runs have different configured references")
    reference_radius = references.pop()
    reference = next((root for root in runs if radii[root] == reference_radius), None)
    if reference is None:
        raise ValueError(f"missing configured smoothing reference R{reference_radius:g}")
    fits = {root: _fit_rows(root, variant=f"R{radii[root]:g}") for root in runs}
    summary = []
    for root, rows in fits.items():
        groups = {}
        for row in rows:
            key = tuple(row[k] for k in _ROUTE_KEYS) + (row["parameter"],)
            groups.setdefault(key, []).append(row)
        for key, group in sorted(groups.items()):
            base = {
                r["realization_id"]: r
                for r in fits[reference]
                if tuple(r[k] for k in _ROUTE_KEYS) + (r["parameter"],) == key
            }
            ratios = [
                r["half_width"] / base[r["realization_id"]]["half_width"]
                for r in group
                if r["realization_id"] is not None
                and r["status"] == "complete"
                and r["usable"]
                and r["realization_id"] in base
                and base[r["realization_id"]]["status"] == "complete"
                and base[r["realization_id"]]["usable"]
            ]
            entry = {
                **dict(zip(_ROUTE_KEYS, key[:4])),
                "parameter": key[4],
                "smoothing_mpc_h": radii[root],
                **_route_summary(group),
                "median_width_ratio_vs_reference": float(np.median(ratios)) if ratios else None,
                "width_ratio_pairs": len(ratios),
                "reference_smoothing_mpc_h": radii[reference],
                "joint_fom": None,
            }
            summary.append(entry)
        for route in routes_for_run(runs[root]):
            try:
                draws = [
                    load_chain_samples(root, selectors=route, parameter=name).reshape(-1)
                    for name in PARAMETERS
                ]
            except (OSError, ValueError, KeyError):
                continue
            fom = joint_summary(np.column_stack(draws))["fom"]
            for entry in summary:
                if entry["smoothing_mpc_h"] == radii[root] and all(
                    entry[k] == route[k] for k in _ROUTE_KEYS
                ):
                    entry["joint_fom"] = fom
    return summary


def routes_for_run(run):
    from pzbao.run import routes_for

    return routes_for(run["config"], run["metadata"]["workload"])


def scan_reproducibility_rows(scan_r15, refit):
    """Paired scan-R15 minus core-R15 fits at shared post-reconstruction routes.

    Both sides belong to one prior family: the scan run's own fits against the core power3
    refit, or (revision 2.20) the unified-prior refit of the scan run against the unified
    refit of the primary.
    """
    from pzbao.config import PRIMARY_POWER3, UNIFIED_PRIOR
    from pzbao.run import load_run

    scan, core = load_run(scan_r15), load_run(refit)
    scan_recon = scan["config"]["measurement"]["reconstruction"]
    core_recon = core["config"]["measurement"]["reconstruction"]
    reference_radius = scan_recon["exploratory_smoothing_scan"]["reference_smoothing_mpc_h"]
    scan_definition = scan["metadata"].get("inference_definition")
    core_definition = core["metadata"].get("inference_definition")
    if (
        scan["metadata"]["workload"] != "smoothing_scan"
        or scan_recon.get("sensitivity_run") != {"kind": "smoothing_scan"}
        or scan_definition not in (None, UNIFIED_PRIOR)
        or "transfer_run" in scan_recon
    ):
        raise ValueError("reproducibility requires the reference smoothing_scan run")
    if (
        core_definition not in (PRIMARY_POWER3, UNIFIED_PRIOR)
        or "sensitivity_run" in core_recon
        or "transfer_run" in core_recon
    ):
        raise ValueError("reproducibility requires the core power3 refit")
    if (scan_definition == UNIFIED_PRIOR) != (core_definition == UNIFIED_PRIOR):
        raise ValueError("reproducibility requires both sides from one prior family")
    if any(
        recon["smoothing_values_mpc_h"] != [reference_radius]
        or recon["allowed_conventions"] != ["recsym"]
        for recon in (scan_recon, core_recon)
    ):
        raise ValueError("reproducibility requires RecSym at the configured reference radius")
    if _comparable_core(scan["config"]) != _comparable_core(core["config"]):
        raise ValueError("scan R15 and the core refit differ beyond their producer")

    def inference(config):
        fitting = deepcopy(config["fitting"])
        # These optional blocks register checks; they do not apply a different prior.
        fitting.pop("paper_comparisons", None)
        fitting["damping"].pop("common_flat_sensitivity", None)
        fitting["damping"].setdefault("reference_smoothing_mpc_h", reference_radius)
        return fitting

    if inference(scan["config"]) != inference(core["config"]):
        raise ValueError("scan R15 and the core refit use different inference definitions")
    shared = {tuple(r[k] for k in _ROUTE_KEYS) for r in routes_for_run(scan)}
    left = {
        tuple(r[k] for k in _ROUTE_KEYS) + (r["parameter"], r["realization_id"]): r
        for r in _fit_rows(scan_r15, variant="scan", routes=shared)
    }
    right = {
        tuple(r[k] for k in _ROUTE_KEYS) + (r["parameter"], r["realization_id"]): r
        for r in _fit_rows(refit, variant="core", routes=shared)
    }
    rows = []
    for key in sorted(set(left) | set(right), key=str):
        a, b = left.get(key), right.get(key)
        a = a if a and a["status"] == "complete" else None
        b = b if b and b["status"] == "complete" else None
        usable = bool(a and b and a["usable"] and b["usable"])
        rows.append(
            {
                **dict(zip(_ROUTE_KEYS, key[:4])),
                "parameter": key[4],
                "realization_id": key[5],
                "scan_minus_core": (a["value"] - b["value"]) if a and b else None,
                "width_ratio_scan_over_core": (
                    a["half_width"] / b["half_width"] if usable else None
                ),
                "missing": [label for label, x in (("scan", a), ("core", b)) if x is None],
            }
        )
    return rows


def transfer_comparison(run_root, reference_root):
    """Transfer versus self-reconstruction pairs, verified on identical targets and randoms."""
    from pzbao.diagnostics.review import collect_route, transfer_comparison_rows
    from pzbao.run import load_run

    run, reference = load_run(run_root), load_run(reference_root)
    records = []
    for route in routes_for_run(run):
        if route["reconstruction_state"] != "postreconstruction":
            continue
        record, _ = collect_route(run, {k: route[k] for k in _ROUTE_KEYS})
        records.append(record)
    return transfer_comparison_rows(run, reference, records)


def transfer_mean_fits(run_root, reference_root):
    """Mean-of-25 fits of a cross-sample run's targets beside the self run's pre/post fits.

    Rows carry ``source_kind`` (pre, self or cross); missing fits stay visible. Pair the two
    runs with ``transfer_comparison`` first, which verifies identical targets and randoms.
    """
    from pzbao.run import load_run

    targets = {
        tuple(route[k] for k in _ROUTE_KEYS[:3])
        for route in routes_for_run(load_run(run_root))
        if route["reconstruction_state"] == "postreconstruction"
    }

    def means(root, kind, state):
        rows = _fit_rows(root, variant=kind, routes={(*target, state) for target in targets})
        return [{**r, "source_kind": kind} for r in rows if r["realization_id"] is None]

    return (
        means(reference_root, "pre", "prereconstruction")
        + means(reference_root, "self", "postreconstruction")
        + means(run_root, "cross", "postreconstruction")
    )


# presentation: saved values and validation.


def posterior_rows(records, inputs):
    from pzbao.sampling import read_chain

    inputs.read_current(Path(__file__).resolve().parents[1] / "sampling.py")
    result = []
    for record in records:
        if record["status"] != "available":
            continue
        for filename, digest in record["input_sha256"].items():
            inputs.read(filename, digest)
        sampling = record["sampling"]
        names = sampling["sampled_parameters"]
        retained = []
        for row in sampling["ensembles"]:
            decoded = read_chain(
                Path(record["source"]) / row["path"],
                parameter_names=names,
                identity=dict(record["identity"], ensemble_index=row["ensemble_index"]),
            )
            if decoded["axes"] != ["walker", "iteration", "parameter"]:
                raise ValueError("Unexpected native chain axes")
            values = decoded["draws"]
            burn = int(row["steps"] * sampling["burnin_fraction"] + 0.5)
            draws = values[:, burn:, [names.index("qiso"), names.index("qap")]].reshape(-1, 2)
            if len(draws) != row["retained_draws"] or not np.isfinite(draws).all():
                raise ValueError("Invalid retained native draws")
            retained.append(draws)
        draws = np.concatenate(retained)
        covariance = np.cov(draws, rowvar=False, ddof=1)
        np.testing.assert_allclose(covariance, record["covariance"], rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(
            1 / np.sqrt(np.linalg.det(covariance)), record["fom"], rtol=1e-10
        )
        if len(draws) != record["retained_draws"]:
            raise ValueError("Retained draw count differs")
        for i, parameter in enumerate(("qiso", "qap")):
            result.append(
                dict(
                    record["route"],
                    parameter=parameter,
                    mean=float(draws[:, i].mean()),
                    sd=float(np.sqrt(covariance[i, i])),
                    identity=record["identity"],
                    profile_flagged=record["profile_flagged"],
                    prior_contact=record["prior_contact"][parameter],
                    sampling_limits=record["sampling_limits"],
                )
            )
        print(
            f"Read posterior: {record['route']['measurement_bin']} "
            f"{record['route']['condition']} {record['route']['reconstruction_state']}",
            flush=True,
        )
    return result


# r25 refresh: saved values and validation.


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_r25(fit, complete, replay, native, covariance, reference):
    """Compare direct scientific identities and the saved native validation record."""
    identity = fit["identity"]
    require_identity(identity)
    route = reference["route"]
    require(identity == complete["identity"] == replay["identity"], "R25 identity mismatch")
    require(all(identity[k] == v for k, v in route.items()), "R25 route mismatch")
    require(identity["realization_id"] is None, "R25 must be a mean fit")
    for record in (complete, replay, native):
        require(record["code_commit"] == identity["code_commit"], "R25 producer mismatch")
        require(record["design_sha256"] == native["design_sha256"], "R25 design mismatch")
        require(record["task"] == native["task"], "R25 task mismatch")
    require(
        complete["resolved_snapshot_sha256"] == identity["analysis_config_sha256"],
        "R25 snapshot mismatch",
    )
    require(replay["saved_curve_reproduced"] is True, "R25 native replay unavailable")
    require(replay["scientific_acceptance"] is False, "Unexpected R25 acceptance state")
    task, design = native["task"], native["design"]
    require(
        task["case"]["name"] == "R25" and task["case"]["smoothing"] == 25,
        "Expected R25 reconstruction",
    )
    require(
        all(
            task["case"][k] == v
            for k, v in {
                "tracer": route["tracer"],
                "density": "current",
                "zero_error": False,
                "field": "parent",
                "radial": "ensemble",
                "measurement_fkp": "current",
                "reconstruction_fkp": False,
            }.items()
        ),
        "R25 changed unrelated case settings",
    )
    require(
        task["measurement_bin"] == route["measurement_bin"]
        and task["reconstruction_state"] == route["reconstruction_state"]
        and task["case"]["condition"] == route["condition"],
        "R25 task route mismatch",
    )
    require(
        design["mock_ids"] == covariance["realization_ids"] == list(range(25)),
        "R25 paired mock IDs differ",
    )
    control = design["inference_controls"][task["inference"]]
    require(
        control["broadband"] == "power3" and control["centre_smoothing_mpc_h"] == 15,
        "R25 broadband or prior reference differs",
    )
    model = reference["model"]
    require(
        control["resolved_centre_reference"] == model["damping_prior"], "R25 baseline prior differs"
    )
    expected_prior = deepcopy(model["damping_prior"])
    expected_prior.update(
        transverse_damping_source="crossed_saved_smoothing_centre_diagnostic",
        centre_reference=model["damping_prior"],
    )
    require(fit["damping_prior"] == expected_prior, "R25 damping prior differs")
    for key in ("ordering", "units", "ap_mode", "quadrature", "z_eff", "fitting_growth_rate"):
        require(fit[key] == model[key], f"R25 {key} differs")
    reconstruction = dict(model["reconstruction_inputs"], smoothing_mpc_h=25)
    require(fit["reconstruction_inputs"] == reconstruction, "R25 reconstruction inputs differ")
    for key in ("root", "code_commit", "analysis_config_sha256"):
        require(fit["source_run"][key] == model["source_run"][key], "R25 source differs")
    require(
        native["source_commit"] == model["source_run"]["code_commit"]
        and native["source_snapshot_sha256"] == model["source_run"]["analysis_config_sha256"],
        "R25 measurement origin differs",
    )
    require(
        covariance["route"] == route
        and covariance["code_commit"] == identity["code_commit"]
        and covariance["mean_covariance_scaling"] == "C_route/N",
        "R25 covariance route or normalization differs",
    )
    require(
        covariance["diagnostics"]["combined"]["positive_definite"] is True,
        "R25 calibrated covariance is not positive definite",
    )
    require(
        all(c["positive_definite"] is True for c in covariance["diagnostics"]["caps"].values()),
        "R25 calibrated cap covariance is not positive definite",
    )
    require(
        {c["method"] for c in native["checks"]} == {"profiling", "detection", "mcmc"},
        "Missing native R25 checks",
    )
    for check in native["checks"]:
        require(np.isfinite(check["objective"]), "Nonfinite native R25 objective")
        for key in ("route", "damping_prior", "reconstruction_inputs"):
            require(check["metadata"][key] == fit[key], f"R25 native {key} differs")


def r25_tasks(manifest):
    selected = [
        (index, task)
        for index, task in enumerate(manifest["inference_tasks"])
        if task["case"]["name"] == "R25"
        and task["case"]["condition"] == "sigma_0.003"
        and task["reconstruction_state"] == "postreconstruction"
        and task["inference"] == "power3"
    ]
    require(
        len(selected) == len(BINS)
        and {task["measurement_bin"] for _, task in selected} == set(BINS),
        "Expected one R25 power3 task for each of the five bins",
    )
    return selected


def r25_rows(sources, root, ready):
    from pzbao.fitting import read_profiles

    rows, diagnostics = [], []
    root = resolve_saved_path(root)
    manifest = sources.json(root / "manifest.json")
    for index, task in r25_tasks(manifest):
        name = task["measurement_bin"]
        tracer = task["case"]["tracer"]
        directory = (
            root
            / "fit"
            / (
                f"{tracer}--sigma_0.003--R25--{name}--postreconstruction--"
                f"power3--{name}--sigma_0.003--postreconstruction"
            )
        )
        if not directory.exists():
            diagnostics.append(
                dict(
                    measurement_bin=name,
                    status="missing",
                    selector=index,
                    reason="R25 mean fit not yet produced",
                    scientific_acceptance=False,
                )
            )
            continue
        native = sources.json(root / "native" / f"{index}.json")
        resolved_task = dict(
            task,
            inference="--".join(
                (task["inference"], name, task["case"]["condition"], task["reconstruction_state"])
            ),
        )
        require(
            native["task"] == resolved_task
            and native["measurement_design_sha256"] == manifest["design_sha256"]
            and native["code_commit"] == manifest["code_commit"],
            "R25 native selector or design differs from manifest",
        )
        reference = next(
            r
            for r in ready["records"]
            if r["route"]
            == {
                "tracer": tracer,
                "measurement_bin": name,
                "condition": "sigma_0.003",
                "reconstruction_state": "postreconstruction",
            }
        )
        fit = sources.json(directory / "mean/fit.json")
        complete = sources.json(directory / "mean/complete.json")
        replay = sources.json(directory / "components.json")
        covroot = root / f"covariance-runs/{tracer}--sigma_0.003--R25/covariance"
        covariance = sources.json(
            covroot / "routes" / f"{name}--sigma_0.003--postreconstruction/covariance.json"
        )
        validate_r25(fit, complete, replay, native, covariance, reference)
        for product in complete["products"]["products"]:
            path = Path(product["path"])
            require(len(path.parts) == 1, "Unexpected R25 product path")
            require(bool(sources.read(directory / "mean" / path)), "Empty R25 product")
        path = directory / "mean/profiles.npy"
        decoded = read_profiles(path, identity=complete["identity"])
        require(
            decoded["parameters"] == fit["parameters"]
            and decoded["maximization"] == fit["optimizer"]["maximization"],
            "Native R25 profile differs from saved fit",
        )
        for parameter in PARAMETERS:
            p = next(p for p in decoded["parameters"] if p["basename"] == parameter)
            rows.append(
                dict(
                    measurement_bin=name,
                    parameter=parameter,
                    variant="r25",
                    status="available",
                    source=str(path),
                    identity=fit["identity"],
                    n_mocks=25,
                    covariance_normalization="C/25",
                    ess=None,
                    ess_reason="MINOS profile interval; no posterior draws",
                    **interval(p, decoded["maximization"]),
                )
            )
        caps = [
            sources.json(
                covroot / "caps" / f"{name}--sigma_0.003--postreconstruction--{cap}/cap.json"
            )
            for cap in ("NGC", "SGC")
        ]
        diagnostics.append(
            dict(
                measurement_bin=name,
                status="available",
                identity=fit["identity"],
                shot_noise_rescaling=covariance["shot_noise"]["shot_noise_rescaling"],
                shot_noise_boundary=covariance["shot_noise"]["solution_on_search_boundary"],
                combined_positive_definite=covariance["diagnostics"]["combined"][
                    "positive_definite"
                ],
                covariance_interpretation=covariance["mean_covariance_interpretation"],
                original_cap_diagnostics=[
                    {
                        k: cap[k]
                        for k in (
                            "cap",
                            "shot_noise_rescaling",
                            "warnings",
                            "postprocessing_failure",
                        )
                    }
                    for cap in caps
                ],
                saved_curve_reproduced=True,
                scientific_acceptance=False,
            )
        )
    return rows, diagnostics


def merge_r25_rows(rows, added):
    """Allow refreshes of unchanged results and new completed bins, preserving missing rows."""
    key = lambda r: (r["measurement_bin"], r["parameter"], r["variant"])
    combined = {key(r): r for r in rows}
    require(len(combined) == len(rows), "Duplicate source robustness row")
    require(len({key(r) for r in added}) == len(added), "Duplicate new R25 row")
    for row in added:
        previous = combined.get(key(row))
        require(
            previous is None or previous["status"] == "missing" or previous == row,
            "R25 would replace an existing result",
        )
        combined[key(row)] = row
    return list(combined.values())


def available_rows(rows, name, parameter):
    group = [
        r
        for r in rows
        if r["measurement_bin"] == name
        and r["parameter"] == parameter
        and r["variant"] in R25_VARIANTS
        and r["status"] == "available"
    ]
    require(len({r["variant"] for r in group}) == len(group), "Duplicate robustness row")
    return sorted(group, key=lambda r: list(R25_VARIANTS).index(r["variant"]))


# lrg figures: saved values and validation.


def lrg_row_key(row):
    return tuple(row[key] for key in ("measurement_bin", "condition", "state", "mock", "parameter"))


def lrg_interval(row):
    """Retain numerical widths separately from validity and prior contact."""
    bounds = row["minos_bounds"]
    width = None
    if bounds is not None and len(bounds) == 2:
        values = np.asarray(bounds, dtype=float)
        if np.isfinite(values).all() and values[1] > values[0]:
            width = float((values[1] - values[0]) / 2)
    flags = row["interval_flags"] or {}
    valid = width is not None and all(
        flags.get(k) is True for k in ("is_valid", "lower_valid", "upper_valid")
    )
    boundary = bool(row["boundary"]) or any(
        flags.get(k) is True for k in ("at_lower_limit", "at_upper_limit")
    )
    maximum = row["optimizer"]["maximization"]
    optimizer_ok = (
        maximum.get("is_valid") is True
        and maximum.get("is_above_max_edm") is False
        and maximum.get("has_reached_call_limit") is False
    )
    return {
        "width": width,
        "valid": valid,
        "boundary": boundary,
        "optimizer_ok": optimizer_ok,
        "usable": valid and not boundary and optimizer_ok,
    }


def validate_lrg_rows(rows, provenance, complete):
    """Check the serialized report's complete cohort and route-owned identities."""
    config = provenance["analysis_config"]
    grid = config["analysis"]["workloads"]["full_grid"]
    bins = complete["measurement_bins"]
    expected = set(
        itertools.product(bins, grid["sigma_z_tiers"], STATES, grid["mock_ids"], PARAMETERS)
    )
    keys = [lrg_row_key(row) for row in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("missing, duplicate or unexpected report rows")
    if complete["fit_count"] * len(PARAMETERS) != len(rows) or not complete["completed_at"]:
        raise ValueError("report completion does not match its cohort")
    routes = {tuple(row[k] for k in IDENTITY) for row in provenance["route_identities"]}
    for row in rows:
        ident = row["identity"]
        if tuple(ident[k] for k in IDENTITY) not in routes or any(
            ident[k] != row[r]
            for k, r in (
                ("measurement_bin", "measurement_bin"),
                ("condition", "condition"),
                ("reconstruction_state", "state"),
                ("realization_id", "mock"),
            )
        ):
            raise ValueError("fit identity differs from report route")
        if ident["code_commit"] != provenance["release"]["definition"]["code_commit"]:
            raise ValueError("fit producer differs from release")
        if not all(row["covariance_identity"].get(k) for k in IDENTITY):
            raise ValueError("missing covariance provider identity")
        if not np.isfinite(row["value"]):
            raise ValueError("nonfinite fitted centre")
        status = lrg_interval(row)
        if status["valid"] != row["valid"]:
            raise ValueError("exported interval validity differs from its flags")
        if status["width"] is not None and row["width"] is not None:
            np.testing.assert_allclose(status["width"], row["width"], rtol=1e-12)
    return grid


def paired_lrg_rows(rows, grid):
    from pzbao.diagnostics.review import uncertainty_gain

    lookup = {lrg_row_key(row): row for row in rows}
    pairs = []
    for bin_name, condition, mock, parameter in itertools.product(
        LRG_BINS, grid["sigma_z_tiers"], grid["mock_ids"], PARAMETERS
    ):
        pre, post = [lookup[(bin_name, condition, state, mock, parameter)] for state in STATES]
        statuses = [lrg_interval(row) for row in (pre, post)]
        widths = [status["width"] for status in statuses]
        raw_ratio = uncertainty_gain(*widths) if all(w is not None for w in widths) else None
        usable = all(status["usable"] for status in statuses)
        pairs.append(
            dict(
                measurement_bin=bin_name,
                condition=condition,
                mock=mock,
                parameter=parameter,
                post_minus_pre=post["value"] - pre["value"],
                raw_width_ratio=raw_ratio,
                uncertainty_ratio=raw_ratio if usable else None,
                precision_ratio=1 / raw_ratio if usable else None,
                usable=usable,
                pre_status=statuses[0],
                post_status=statuses[1],
                pre_interval_flags=pre["interval_flags"],
                post_interval_flags=post["interval_flags"],
            )
        )
    return pairs


def lrg_summaries(rows, pairs, grid):
    result = []
    for bin_name, condition, parameter in itertools.product(
        LRG_BINS, grid["sigma_z_tiers"], PARAMETERS
    ):
        matched = [
            p
            for p in pairs
            if (p["measurement_bin"], p["condition"], p["parameter"])
            == (bin_name, condition, parameter)
        ]
        usable = [p["uncertainty_ratio"] for p in matched if p["usable"]]
        for state in STATES:
            group = [
                r
                for r in rows
                if lrg_row_key(r)[:3] == (bin_name, condition, state)
                and r["parameter"] == parameter
            ]
            values = np.array([r["value"] for r in group])
            statuses = [lrg_interval(r) for r in group]
            widths = [s["width"] for s in statuses if s["usable"]]
            result.append(
                dict(
                    measurement_bin=bin_name,
                    condition=condition,
                    state=state,
                    parameter=parameter,
                    N=len(group),
                    mean=float(values.mean()),
                    bias=float(values.mean() - 1),
                    scatter=float(values.std(ddof=1)),
                    sem=float(values.std(ddof=1) / np.sqrt(len(group))),
                    valid_unconfined=sum(s["usable"] for s in statuses),
                    boundary=sum(s["boundary"] for s in statuses),
                    invalid=sum(not s["valid"] for s in statuses),
                    unavailable=sum(s["width"] is None for s in statuses),
                    median_width=float(np.median(widths)) if widths else None,
                    median_precision=float(np.median(1 / np.array(widths))) if widths else None,
                    paired_shift=float(np.mean([p["post_minus_pre"] for p in matched])),
                    paired_shift_sem=float(
                        np.std([p["post_minus_pre"] for p in matched], ddof=1)
                        / np.sqrt(len(matched))
                    ),
                    usable_pairs=len(usable),
                    median_uncertainty_ratio=float(np.median(usable)) if usable else None,
                )
            )
    return result


def lrg_forecast_rows(preparation, config):
    """Use attained per-cap counts and volumes in the existing DESI calibration."""
    p0 = config["tracers"]["LRG"]["fkp_p0_h3_mpc3"]
    if not np.isfinite(p0) or p0 <= 0:
        raise ValueError("invalid configured P0")
    output = []
    for condition in config["analysis"]["workloads"]["full_grid"]["sigma_z_tiers"]:
        for name in LRG_BINS:
            caps = []
            for cap in ("NGC", "SGC"):
                source = preparation["conditions"][condition][cap]
                count, volume = source["counts"][name], source["volumes_mpc_h3"][name]
                if type(count) is not int or count <= 0 or not np.isfinite(volume) or volume <= 0:
                    raise ValueError("invalid attained counts or volume")
                density = count / volume
                veff = volume * (density * p0 / (1 + density * p0)) ** 2
                caps.append(
                    dict(
                        cap=cap,
                        count=count,
                        volume_mpc_h3=volume,
                        nbar_h3_mpc3=density,
                        veff_mpc_h3=veff,
                    )
                )
            zeff = preparation["effective_redshifts"][condition][name]["z_eff"]
            coefficient = float(calibration_coefficient_at([zeff], ["LRG"])[0])
            volume = sum(c["volume_mpc_h3"] for c in caps)
            veff = sum(c["veff_mpc_h3"] for c in caps)
            output.append(
                dict(
                    measurement_bin=name,
                    condition=condition,
                    z_eff=zeff,
                    caps=caps,
                    count=sum(c["count"] for c in caps),
                    volume_mpc_h3=volume,
                    veff_mpc_h3=veff,
                    p0_mpc_h3=p0,
                    sigma_r_over_r=float(r_precision_percent(veff, coefficient) / 100),
                    cosmic_variance_sigma_r_over_r=float(
                        r_precision_percent(volume, coefficient) / 100
                    ),
                    calibration_coefficient_percent=coefficient,
                )
            )
    return output


def lrg_clustering(curves, name, condition, state, ids):
    """Return an ensemble mean and one-realization scatter; never silently use SEM."""
    records = []
    for mock in ids:
        prefix = f"{name}--{condition}__{state}__mock{mock}__"
        record = {
            key: curves[prefix + key]
            for key in (
                "separation",
                "multipoles",
                "fit_edges",
                "data",
                "model",
                "residual",
                "standard_deviation",
            )
        }
        n = len(record["separation"])
        dimension = n * len(record["multipoles"])
        for key in ("data", "model", "residual", "standard_deviation"):
            if record[key].shape != (dimension,) or not np.isfinite(record[key]).all():
                raise ValueError("invalid saved curve shape or value")
        if np.any(record["standard_deviation"] <= 0):
            raise ValueError("invalid saved one-realization uncertainty")
        np.testing.assert_allclose(record["residual"], record["data"] - record["model"], atol=1e-12)
        if records:
            for key in ("fit_edges", "multipoles"):
                np.testing.assert_array_equal(record[key], records[0][key])
        records.append(record)
    data = np.array([r["data"] for r in records])
    return dict(
        separation=np.mean([r["separation"] for r in records], axis=0),
        multipoles=records[0]["multipoles"],
        mean=data.mean(axis=0),
        scatter=data.std(axis=0, ddof=1),
        example=records[0],
    )


# precision refresh: saved values and validation.


def precision_row(name, condition, z, caps, p0, tracer):
    volume = np.array([c["volume_mpc_h3"] for c in caps], dtype=float)
    density = np.array([c["nbar_h3_mpc3"] for c in caps], dtype=float)
    require(len(caps) == 2 and {c["cap"] for c in caps} == {"NGC", "SGC"}, "Wrong cap coverage")
    require(
        np.isfinite(volume).all()
        and np.isfinite(density).all()
        and np.all(volume > 0)
        and np.all(density > 0)
        and np.isfinite(p0)
        and p0 > 0,
        "Invalid density, volume or weighting",
    )
    veff = float(np.sum(volume * (density * p0 / (1 + density * p0)) ** 2))
    coefficient = float(calibration_coefficient_at([z], [tracer])[0])
    return dict(
        measurement_bin=name,
        condition=condition,
        z=float(z),
        caps=caps,
        volume_mpc_h3=float(volume.sum()),
        veff_mpc_h3=veff,
        p0_mpc_h3=float(p0),
        coefficient_percent=coefficient,
        sigma_r_over_r=float(r_precision_percent(veff, coefficient) / 100),
        cosmic_variance_sigma_r_over_r=float(r_precision_percent(volume.sum(), coefficient) / 100),
    )


def mock_rows(preparations, config):
    rows = []
    for tracer, spec in config["tracers"].items():
        prep = preparations[tracer]
        require(
            prep["tracer"] == tracer and prep["realization_ids"] == list(range(25)),
            "Preparation tracer or paired mock IDs differ",
        )
        for condition in config["analysis"]["workloads"]["full_grid"]["sigma_z_tiers"]:
            for shell in spec["measurement_bins"]:
                name = shell["name"]
                caps = []
                for cap in ("NGC", "SGC"):
                    group = prep["conditions"][condition][cap]
                    volume, count = group["volumes_mpc_h3"][name], group["counts"][name]
                    require(
                        type(count) is int and count > 0 and volume > 0, "Invalid attained count"
                    )
                    caps.append(
                        dict(
                            cap=cap, count=count, volume_mpc_h3=volume, nbar_h3_mpc3=count / volume
                        )
                    )
                rows.append(
                    precision_row(
                        name,
                        condition,
                        prep["effective_redshifts"][condition][name]["z_eff"],
                        caps,
                        spec["fkp_p0_h3_mpc3"],
                        "LRG" if tracer == "LRG" else "BGS",
                    )
                )
    require(len(rows) == 25, "Expected all five bins and five mock tiers")
    return rows


def effective_redshift_rows(preparations, config):
    """One (bin, tier) effective redshift from the preparation records; state independent."""
    rows = []
    for tracer, spec in config["tracers"].items():
        prep = preparations[tracer]
        require(prep["tracer"] == tracer, "Preparation tracer differs")
        for condition in config["analysis"]["workloads"]["full_grid"]["sigma_z_tiers"]:
            for shell in spec["measurement_bins"]:
                value = float(prep["effective_redshifts"][condition][shell["name"]]["z_eff"])
                require(np.isfinite(value) and value > 0, "Invalid effective redshift")
                rows.append(dict(measurement_bin=shell["name"], condition=condition, z_eff=value))
    require(len(rows) == len(BINS) * len(TIERS), "Expected all five bins and five mock tiers")
    return rows


def dr1_reference(data):
    """Snapshot only the consumed DR1 scenario, excluding historical infinite-density cases."""
    panel = data["absolute_veff_plot"]
    selected = [r for r in panel["scenarios"] if r["key"] == "desi_dr1"]
    require(len(selected) == 1, "Expected exactly one DR1 reference")
    return dict(absolute_veff_plot=dict(bins=panel["bins"], scenarios=selected))


def dr1_rows(data, config):
    panel = data["absolute_veff_plot"]
    source = next(r for r in panel["scenarios"] if r["key"] == "desi_dr1")
    require([b["bin"] for b in panel["bins"]] == list(BINS), "DR1 bin order differs")
    require(len(source["components_by_bin"]) == 5, "Missing DR1 cap geometry")
    rows = []
    for shell, components in zip(panel["bins"], source["components_by_bin"]):
        spec = config["tracers"][shell["tracer"]]
        expected = next(b for b in spec["measurement_bins"] if b["name"] == shell["bin"])
        require(all(shell[k] == expected[k] for k in ("zmin", "zmax")), "DR1 bin edges differ")
        caps = [
            dict(
                cap=c["region"],
                volume_mpc_h3=c["shell_volume_mpc_h3"],
                nbar_h3_mpc3=c["nbar_h3_mpc3"],
            )
            for c in components
        ]
        rows.append(
            precision_row(
                shell["bin"],
                "desi_dr1",
                shell["redshift_center"],
                caps,
                spec["fkp_p0_h3_mpc3"],
                "LRG" if shell["tracer"] == "LRG" else "BGS",
            )
        )
    return rows


# density refresh: saved values and validation.


def flagged(row):
    profile = row["source_profile"]
    parameters = profile["parameters"]
    if not parameters:
        require(
            bool(profile.get("unavailable_reason")),
            "Missing source profile diagnostics without reason",
        )
        return True
    selected = [p for p in parameters if p["basename"] in ("qiso", "qap")]
    require(len(selected) == 2, "Missing source dilation diagnostics")
    for param in selected:
        flags = param["interval_flags"]
        if (
            flags is None
            or not all(flags[k] for k in ("is_valid", "lower_valid", "upper_valid"))
            or flags["at_lower_limit"]
            or flags["at_upper_limit"]
        ):
            return True
    return False


def validate_forecast(data):
    require(
        data["all_available"] and data["specification_revision"] == forecast.REVISION,
        "Incomplete or incompatible conditional forecast",
    )
    rows = data["rows"]
    np.testing.assert_allclose(
        data["target_fsky"], forecast.configured_target_fsky(), rtol=0, atol=0
    )
    expected = {
        (name, tier, state)
        for name in forecast.BINS
        for tier in forecast.TIERS
        for state in forecast.STATES
    }
    keys = {
        (r["route"]["measurement_bin"], r["route"]["condition"], r["route"]["reconstruction_state"])
        for r in rows
    }
    require(len(rows) == 50 and keys == expected, "Forecast does not cover all 50 primary routes")
    for row in rows:
        require(
            row["status"] == "available" and row["scientific_acceptance"] is False,
            "Unavailable or mislabeled forecast",
        )
        require(set(row["numerics"]) == {"current", "target"}, "Missing numerical checks")
        for arm in row["numerics"].values():
            require(all(np.isfinite(v) for v in arm.values()), "Nonfinite numerical checks")
        terms = forecast.ratio_terms(
            *(row["results"][arm]["fom"] for arm in ("current", "target", "target_area"))
        )
        for key, value in terms.items():
            np.testing.assert_allclose(row[key], value, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(row["target_fsky"], data["target_fsky"], rtol=0, atol=1e-14)
        np.testing.assert_allclose(
            row["area_factor"],
            forecast.area_factor(row["cap_area_deg2"], data["target_fsky"]),
            rtol=1e-12,
        )
        if row["route"]["condition"] == "native":
            require(
                row["target_nbar"] is None and not row["density_target_available"],
                "Native redshifts have no density target",
            )
            np.testing.assert_allclose(row["density_ratio"], 1.0, rtol=1e-10)
        flagged(row)
    require(data["paired_gains"] == forecast.paired_gains(rows), "Paired forecast summary differs")
    return rows


def flat_rows(rows):
    result = []
    for row in rows:
        result.append(
            dict(
                **row["route"],
                attained_nbar=row["attained_nbar"],
                target_nbar=row["target_nbar"],
                current_sigma_perp=row["current_sigma_perp"],
                target_sigma_perp=row["target_sigma_perp"],
                area_factor=row["area_factor"],
                target_fsky=row["target_fsky"],
                density_noise_only_ratio=row["density_noise_only_ratio"],
                density_ratio=row["density_ratio"],
                total_ratio=row["total_ratio"],
                density_contribution=row["density_contribution"],
                area_contribution=row["area_contribution"],
                source_profile_flagged=flagged(row),
                source_profile_unavailable_reason=row["source_profile"].get("unavailable_reason"),
                **{arm + "_fom": row["results"][arm]["fom"] for arm in row["results"]},
            )
        )
    return result


# baseline refresh: saved values and validation.


def validate_grid(records, measurement_identity, covariance_overlays=()):
    expected = {(name, tier, state) for name in BINS for tier in TIERS for state in STATES}
    overlay_routes = {}
    for overlay in covariance_overlays:
        if {k: overlay["source_run"][k] for k in IDENTITY} != measurement_identity:
            raise ValueError("Covariance overlay uses different measurements")
        for key in overlay["published_routes"]:
            if key in overlay_routes:
                raise ValueError("Duplicate covariance overlay route")
            overlay_routes[key] = {k: overlay[k] for k in IDENTITY}
    keys = []
    for record in records:
        route = record["route"]
        key = tuple(route[k] for k in ("measurement_bin", "condition", "reconstruction_state"))
        keys.append(key)
        if record["status"] != "available":
            raise ValueError(f"Missing baseline posterior: {key}")
        covariance_identity = overlay_routes.get("--".join(key), measurement_identity)
        if record["covariance_identity"] != covariance_identity:
            raise ValueError("Clustering and posterior measurement identities differ")
        if any(record["identity"].get(k) != v for k, v in route.items()):
            raise ValueError("Posterior route and native identity differ")
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("Incomplete or duplicate baseline posterior grid")


def retained_draws(record, read_chain, sources):
    """Use the native axes and burn-in exactly once; check the published covariance."""
    for path, digest in record["input_sha256"].items():
        sources.read(path, digest)
    sampling = record["sampling"]
    names = sampling["sampled_parameters"]
    ensembles = sampling["ensembles"]
    if [row["ensemble_index"] for row in ensembles] != list(range(4)):
        raise ValueError("Incomplete native ensemble collection")
    retained = []
    for row in ensembles:
        directory = resolve_saved_path(record["source"]).resolve()
        path = (directory / row["path"]).resolve()
        if directory not in path.parents:
            raise ValueError("Native chain path leaves the source directory")
        decoded = read_chain(
            path,
            parameter_names=names,
            identity=dict(record["identity"], ensemble_index=row["ensemble_index"]),
        )
        values = decoded["draws"]
        if decoded["axes"] != ["walker", "iteration", "parameter"]:
            raise ValueError("Unexpected native chain axes")
        if values.shape != (row["diagnostics"]["walkers"], row["steps"], len(names)):
            raise ValueError("Native chain shape differs")
        burn = int(row["steps"] * sampling["burnin_fraction"] + 0.5)
        if row["steps"] - burn != row["diagnostics"]["retained_steps"]:
            raise ValueError("Native retained step count differs")
        draws = values[:, burn:, [names.index("qiso"), names.index("qap")]].reshape(-1, 2)
        if len(draws) != row["retained_draws"] or not np.isfinite(draws).all():
            raise ValueError("Invalid retained native draws")
        retained.append(draws)
    draws = np.concatenate(retained)
    covariance = np.cov(draws, rowvar=False, ddof=1)
    if "mean" in record:
        np.testing.assert_allclose(draws.mean(axis=0), record["mean"], rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(covariance, record["covariance"], rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(
        1 / np.sqrt(np.linalg.det(covariance)),
        record["fom"],
        rtol=1e-10,
    )
    if len(draws) != record["retained_draws"]:
        raise ValueError("Retained draw count differs")
    return draws


def historical_key(route, style):
    index = BINS.index(route["measurement_bin"])
    tracer, zmin, zmax, _ = style.ZBIN_GRID_PANELS[index]
    tier = "desi_baseline" if route["condition"] == "native" else route["condition"]
    state = {"prereconstruction": "prerecon", "postreconstruction": "postrecon"}[
        route["reconstruction_state"]
    ]
    return tracer, zmin, zmax, tier, 15, state


def profile_rows(rows, style, measurement_identity):
    profiles = {}
    seen = set()
    for row in rows:
        if row["parameter"] not in ("qiso", "qap"):
            continue
        if row["covariance_identity"] != measurement_identity:
            raise ValueError("Individual profiles use a different measurement ensemble")
        route = dict(
            measurement_bin=row["measurement_bin"],
            condition=row["condition"],
            reconstruction_state=row["state"],
        )
        if row["identity"].get("run_name") != ("primary-power3--" + "--".join(route.values())):
            raise ValueError("Individual profile is not the primary power3 definition")
        key = historical_key(route, style)
        unique = (key, row["mock"], row["parameter"])
        if unique in seen:
            raise ValueError("Duplicate individual profile parameter")
        seen.add(unique)
        item = profiles.setdefault(key, {}).setdefault(row["mock"], {"valid": True})
        item["valid"] &= bool(row["optimizer_ok"])
        item[row["parameter"] + "_bf"] = row["value"]
        error = row["parabolic_error"]
        item[row["parameter"] + "_err"] = np.nan if error is None else error
        # Minos half-width only for a valid interval without prior-boundary contact.
        usable = row.get("category") == "valid" and row.get("width") is not None
        item[row["parameter"] + "_minos"] = float(row["width"]) if usable else np.nan
        chi2 = np.nan if row.get("chi2") is None else float(row["chi2"])
        if not np.isnan(item.setdefault("chi2", chi2)) and item["chi2"] != chi2:
            raise ValueError("Parameter rows of one profile disagree on chi2")
        item["dof"] = row.get("degrees_of_freedom")
    for members in profiles.values():
        if set(members) != set(range(25)):
            raise ValueError("Incomplete individual-realization cohort")
        if any(
            not all(p + suffix in row for p in ("qiso", "qap") for suffix in ("_bf", "_err"))
            for row in members.values()
        ):
            raise ValueError("Incomplete paired parameter export")
    return profiles


# joint tier constraint: combination of the published tier posteriors (specification 2.14).

#: The degraded tiers that enter the joint constraint; the native tier is the reference.
JOINT_TIERS = tuple(tier for tier in TIERS if tier != "native")
JOINT_CONDITION = "joint"
#: Presentation control for the bootstrap range of the joint width; recorded in every record.
JOINT_BOOTSTRAP_RESAMPLES = 400


def joint_bootstrap_seed():
    """Reuse the inference root seed of the analysis configuration for the bootstrap substream."""
    from pzbao.config import load_analysis_config

    return int(load_analysis_config()["fitting"]["sampling"]["seed"])


def joint_constraint_rows(draws, profiles, style, *, seed, resamples=JOINT_BOOTSTRAP_RESAMPLES):
    """One joint record per bin and state from the retained draws and per-realization profiles.

    Tier posteriors are the retained draws' means and covariances; the cross-tier correlation
    comes from the per-realization best fits. A realization whose optimizer failed in any tier
    is excluded from the correlation and the coverage check and is listed, so the record says
    which realizations it rests on.
    """
    from pzbao.fitting import JOINT_PARAMETERS, joint_tier_constraint

    if list(JOINT_PARAMETERS) != list(PARAMETERS):
        raise ValueError("joint constraint parameters differ from the retained draw columns")
    rows = []
    for name in BINS:
        for state in STATES:
            keys = [
                historical_key(
                    dict(measurement_bin=name, condition=tier, reconstruction_state=state), style
                )
                for tier in JOINT_TIERS
            ]
            route = dict(
                measurement_bin=name, condition=JOINT_CONDITION, reconstruction_state=state
            )
            if any(key not in profiles for key in keys):
                # The designated report carries per-realization profiles for some bins only; a
                # joint constraint without the cross-tier correlation is not formed.
                rows.append(
                    dict(
                        route=route,
                        tiers=list(JOINT_TIERS),
                        status="unavailable",
                        reason="no per-realization profiles for every tier in the designated report",
                    )
                )
                continue
            means = np.array([draws[key].mean(axis=0) for key in keys])
            covariances = np.array([np.cov(draws[key], rowvar=False, ddof=1) for key in keys])
            cohort = sorted(set.intersection(*(set(profiles[key]) for key in keys)))
            usable = [mock for mock in cohort if all(profiles[key][mock]["valid"] for key in keys)]
            estimates = np.array(
                [
                    [[profiles[key][mock][p + "_bf"] for p in PARAMETERS] for key in keys]
                    for mock in usable
                ]
            )
            record = joint_tier_constraint(
                list(JOINT_TIERS), means, covariances, estimates, resamples=resamples, seed=seed
            )
            combination = record["combination"]
            rows.append(
                dict(
                    route=route,
                    tiers=list(JOINT_TIERS),
                    status="available",
                    parameters=list(PARAMETERS),
                    realizations_used=len(usable),
                    realizations_excluded=sorted(set(cohort) - set(usable)),
                    mean=combination["mean"],
                    covariance=combination["covariance"],
                    sd=combination["standard_deviations_ddof1"],
                    fom=combination["two_dimensional_fom"],
                    weights=combination["weights"],
                    removed_tiers=combination["removed_tiers"],
                    stacked_minimum_eigenvalue=combination["stacked_minimum_eigenvalue"],
                    cross_term_clipped=combination["cross_term_clipped"],
                    leave_one_out={
                        field: record["leave_one_out"][field]
                        for field in (
                            "scatter",
                            "tier_scatter",
                            "best_tier",
                            "best_tier_scatter",
                            "improves",
                        )
                    },
                    realization_combination=record["realization_combination"],
                    bootstrap=record["bootstrap"],
                    estimate_correlation=record["estimate_correlation"],
                )
            )
    return rows


def validate_detection_fit(saved, record, measurement_identity):
    """Use the compatible profile identity recorded by the baseline collector."""
    profile = record["profile"]
    if saved["identity"] != profile["identity"] or saved["parameters"] != profile["parameters"]:
        raise ValueError("Detection fit differs from the registered mean profile")
    if any(saved["identity"].get(k) != v for k, v in record["route"].items()):
        raise ValueError("Detection fit route differs")
    source = saved["source_run"]
    identity = dict(
        run_name=Path(source["root"]).name,
        code_commit=source["code_commit"],
        analysis_config_sha256=source["analysis_config_sha256"],
    )
    if identity != measurement_identity:
        raise ValueError("Detection fit uses different measurements")


# joint fom: saved values and validation.


def route_key(route):
    return "--".join(
        route[name] for name in ("measurement_bin", "condition", "reconstruction_state")
    )


def paired_values(records):
    lookup = {}
    for row in records:
        key = route_key(row["route"])
        if key in lookup:
            raise ValueError("duplicate posterior route: " + key)
        lookup[key] = row
    pairs = []
    for name in BINS:
        for tier in TIERS:
            members = [lookup.get("--".join((name, tier, state))) for state in STATES]
            values = [r["fom"] if r and r["status"] == "available" else None for r in members]
            for value in values:
                if value is not None and (not np.isfinite(value) or value <= 0):
                    raise ValueError("invalid absolute FoM")
            pairs.append(
                dict(
                    measurement_bin=name,
                    condition=tier,
                    pre=values[0],
                    post=values[1],
                    gain=values[1] / values[0] if all(v is not None for v in values) else None,
                )
            )
    return pairs


def read_json(path):
    """Read a saved table, including products relocated to the selected archive."""
    return json.loads(resolve_saved_path(path).read_text())


def load_npz(path):
    """Load saved arrays into memory and close the archive."""
    with np.load(resolve_saved_path(path), allow_pickle=False) as arrays:
        return {name: arrays[name].copy() for name in arrays.files}


def read_tables(directory, filenames):
    """Load selected plot tables without requiring the historical renderer checkout."""
    directory = resolve_saved_path(directory)
    return {name: read_json(directory / name) for name in filenames}


def load_baseline(directory, *, detection_bin="LRG1"):
    """Load baseline plot arrays through native readers, retaining scientific flags."""
    from pzbao.figures import style

    if detection_bin not in LRG_BINS:
        raise ValueError("Detection comparison requires a registered LRG bin")
    inputs = Sources()
    draws, fits, mean_fits = {}, {}, {}
    directory = resolve_saved_path(directory)
    baseline = {
        name: inputs.json(directory / name)
        for name in [
            "clustering-observation.json",
            "profile-diagnostics.json",
            "fom-values.json",
        ]
    }
    provenance = inputs.json(directory / "provenance.json")
    paths = list(provenance["source_bytes"])
    posterior_paths = [
        p for p in paths if p.endswith("/figure-data.json") and "power3-joint-fom" in p
    ]
    if len(posterior_paths) != 1:
        raise ValueError("Ambiguous native posterior export")
    records = json.loads(
        inputs.read(posterior_paths[0], provenance["source_bytes"][posterior_paths[0]])
    )["records"]
    clustering_paths = [
        p for p in paths if p.endswith("/figure-data.json") and p != posterior_paths[0]
    ]
    if len(clustering_paths) != 1:
        raise ValueError("Ambiguous matched clustering export")
    clustering_data = json.loads(
        inputs.read(clustering_paths[0], provenance["source_bytes"][clustering_paths[0]])
    )
    observation = baseline["clustering-observation.json"]
    measurement_identity = {k: observation["measurements"][k] for k in IDENTITY}
    validate_grid(records, measurement_identity, observation["bgs_inference"])
    from pzbao.sampling import read_chain
    from pzbao.fitting import read_profiles

    for record in records:
        values = retained_draws(record, read_chain, inputs)
        key = historical_key(record["route"], style)
        draws[key] = values
        fits[key] = dict(
            qiso_std=float(values[:, 0].std(ddof=1)),
            qiso_mean=float(values[:, 0].mean()),
            qap_std=float(values[:, 1].std(ddof=1)),
        )
        if record["route"]["measurement_bin"] == detection_bin and key[-1] == "postrecon":
            profile = record["profile"]
            inputs.read(profile["source"], profile["source_sha256"])
            path = resolve_saved_path(profile["source"])
            saved = inputs.json(path.with_name("fit.json"))
            validate_detection_fit(saved, record, measurement_identity)
            native_profile = read_profiles(path, identity=profile["identity"])
            np.testing.assert_allclose(saved["chi2"], native_profile["chi2"], rtol=1e-12)
            mean_fits[key] = saved
    profiles_full = profile_rows(baseline["profile-diagnostics.json"], style, measurement_identity)
    joint = joint_constraint_rows(draws, profiles_full, style, seed=joint_bootstrap_seed())
    correlation_panels = saved_correlation_panels(clustering_data["figures"]["fig_cf"])
    fom_pairs = baseline["fom-values.json"]
    expected_pairs = {(name, tier) for name in BINS for tier in TIERS}
    if (
        len(fom_pairs) != len(expected_pairs)
        or {(r["measurement_bin"], r["condition"]) for r in fom_pairs} != expected_pairs
    ):
        raise ValueError("Incomplete or duplicate displayed FoM pairs")
    lookup = {
        (
            r["route"]["measurement_bin"],
            r["route"]["condition"],
            r["route"]["reconstruction_state"],
        ): r["fom"]
        for r in records
    }
    for pair in fom_pairs:
        for field, state in zip(("pre", "post"), STATES):
            np.testing.assert_allclose(
                pair[field], lookup[(pair["measurement_bin"], pair["condition"], state)], rtol=1e-12
            )
    inputs.unchanged()
    posterior_summary = []
    for record in records:
        values = draws[historical_key(record["route"], style)]
        for index, parameter in enumerate(PARAMETERS):
            posterior_summary.append(
                dict(
                    record["route"],
                    parameter=parameter,
                    mean=float(values[:, index].mean()),
                    sd=float(values[:, index].std(ddof=1)),
                    profile_flagged=record["profile_flagged"],
                    prior_contact=record["prior_contact"][parameter],
                    sampling_limits=record["sampling_limits"],
                )
            )
    return dict(
        draws=draws,
        fits=fits,
        mean_fits=mean_fits,
        profiles=profiles_full,
        joint=joint,
        correlation=correlation_panels,
        fom=fom_pairs,
        posterior_summary=posterior_summary,
        posterior_records=records,
        profile_diagnostics=baseline["profile-diagnostics.json"],
        measurement_identity=measurement_identity,
        density=saved_density_rows(clustering_data["density"]),
        density_axes=clustering_data["figures"]["fig_nz"]["updated"]["axes"][0],
        sources=inputs,
    )


def saved_density_rows(rows):
    """Check count/volume units and retain only each row's measurement shell."""
    from pzbao.figures.style import ZBIN_GRID_PANELS

    expected = set(itertools.product(BINS, TIERS))
    keys = [(r["bin"], r["condition"]) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("Incomplete or duplicate baseline density grid")
    result = []
    for row in rows:
        edges = np.asarray(row["edges"], dtype=float)
        counts = np.asarray(row["mean_counts"], dtype=float)
        volume = np.asarray(row["volume_mpc_h3"], dtype=float)
        density = np.asarray(row["nbar_h3_mpc3"], dtype=float)
        if (
            edges.ndim != 1
            or len(edges) != len(counts) + 1
            or counts.shape != volume.shape
            or counts.shape != density.shape
            or not all(np.isfinite(a).all() for a in (edges, counts, volume, density))
            or np.any(np.diff(edges) <= 0)
            or np.any(volume <= 0)
            or np.any(counts < 0)
        ):
            raise ValueError("Invalid baseline density axes or values")
        np.testing.assert_allclose(density, counts / volume, rtol=1e-10, atol=1e-15)
        _, zmin, zmax, _ = ZBIN_GRID_PANELS[BINS.index(row["bin"])]
        keep = (edges[:-1] >= zmin - 1e-10) & (edges[1:] <= zmax + 1e-10)
        if not np.any(keep):
            raise ValueError("Density grid omits its measurement shell")
        local_edges = np.r_[edges[:-1][keep], edges[1:][keep][-1]]
        np.testing.assert_allclose(local_edges[[0, -1]], [zmin, zmax], atol=1e-10)
        result.append(
            dict(
                measurement_bin=row["bin"],
                condition=row["condition"],
                edges=local_edges,
                density=density[keep],
                mean_count=float(counts[keep].sum()),
                volume_mpc_h3=float(volume[keep].sum()),
                attained_nbar_h3_mpc3=float(counts[keep].sum() / volume[keep].sum()),
            )
        )
    return result


def baseline_selection(root, measurement_identity, sources):
    """Read matched preparation diagnostics without rebuilding selected catalogs."""
    rows = []
    stage = _preparation_stage(sources, Path(root))
    for tracer in ("BGS_BRIGHT-21.5", "LRG"):
        directory = Path(root) / stage / "units" / tracer
        identity = yaml.safe_load(sources.read(directory / "result.yaml"))
        if any(identity.get(k) != measurement_identity[k] for k in IDENTITY):
            raise ValueError("Selection diagnostics differ from baseline measurements")
        prep = sources.json(directory / "preparation.json")
        if prep["tracer"] != tracer or prep["realization_ids"] != list(range(25)):
            raise ValueError("Selection tracer or realization grid differs")
        if set(prep["conditions"]) != set(TIERS):
            raise ValueError("Selection conditions differ")
        for tier in TIERS:
            if set(prep["conditions"][tier]) != {"NGC", "SGC"}:
                raise ValueError("Selection cap grid differs")
            for cap, content in prep["conditions"][tier].items():
                radial = content["radial"]
                if not all(
                    token in radial["sample_identity"]
                    for token in (tracer + "|", "|tier=" + tier + "|", "|cap=" + cap)
                ):
                    raise ValueError("Radial diagnostic sample differs")
                edges = np.asarray(radial["z_edges"], dtype=float)
                counts = np.asarray(radial["mean_counts"], dtype=float)
                target = np.asarray(radial["target_probability_mass"], dtype=float)
                empirical = normalized_density(counts, edges) * np.diff(edges)
                if (
                    target.shape != counts.shape
                    or not np.isfinite(target).all()
                    or np.any(target < 0)
                ):
                    raise ValueError("Invalid radial target")
                np.testing.assert_allclose(target.sum(), 1.0, atol=1e-10)
                diagnostics = content["selection_diagnostics_by_mock"]
                boundaries = content["boundary_counts_by_mock"]
                for collection in (diagnostics, boundaries):
                    if [r["realization_id"] for r in collection] != list(range(25)):
                        raise ValueError("Incomplete selection realization cohort")
                ratios = []
                for item in diagnostics:
                    randoms = item["randoms"]
                    if randoms["radial_sample_identity"] != radial["sample_identity"]:
                        raise ValueError("Random selection uses a different radial target")
                    ratio = randoms["attained_random_data_ratio"]
                    np.testing.assert_allclose(ratio, randoms["exact_random_data_ratio"])
                    ratios.append(ratio)
                for item in boundaries:
                    np.testing.assert_allclose(item["z_edges"], edges, atol=1e-10)
                rows.append(
                    dict(
                        tracer=tracer,
                        condition=tier,
                        cap=cap,
                        edges=edges,
                        empirical_mass=empirical,
                        target_mass=target,
                        random_data_ratios=ratios,
                        radial_evidence={
                            k: radial[k]
                            for k in ("fidelity_evidence", "variance_evidence", "support_evidence")
                        },
                        boundary_diagnostics=boundaries,
                        counts=content["counts"],
                        volumes_mpc_h3=content["volumes_mpc_h3"],
                    )
                )
    sources.unchanged()
    return rows


def baseline_interval_counts(rows):
    """Count matched usable native intervals; retain unavailable BGS cohorts."""
    lookup = {}
    for row in rows:
        key = lrg_row_key(row)
        if key in lookup:
            raise ValueError("Duplicate realization interval")
        lookup[key] = row
    result = []
    for name, tier, parameter in itertools.product(BINS, TIERS, PARAMETERS):
        paired = []
        for mock in range(25):
            members = [lookup.get((name, tier, state, mock, parameter)) for state in STATES]
            if any(member is None for member in members):
                if name.startswith("BGS") and all(member is None for member in members):
                    continue
                raise ValueError("Incomplete matched interval cohort")
            paired.append(all(lrg_interval(member)["usable"] for member in members))
        result.append(
            dict(
                measurement_bin=name,
                condition=tier,
                parameter=parameter,
                available_pairs=len(paired),
                usable_pairs=sum(paired) if paired else None,
            )
        )
    return result


def load_lrg_reports(reports):
    """Combine compatible LRG tables and curves for exploratory plotting."""
    rows, curves, config, grid = [], {}, None, None
    for directory in map(Path, reports):
        bundle = read_tables(directory, ["fit-diagnostics.json", "complete.json"])
        provenance = read_json(directory / "provenance.json")
        current_grid = validate_lrg_rows(
            bundle["fit-diagnostics.json"], provenance, bundle["complete.json"]
        )
        if grid is not None and current_grid != grid:
            raise ValueError("LRG ensemble grids differ")
        if config is not None and provenance["analysis_config"] != config:
            raise ValueError("LRG report configurations differ")
        grid, config = current_grid, provenance["analysis_config"]
        rows.extend(bundle["fit-diagnostics.json"])
        arrays = load_npz(directory / "fit-curves.npz")
        if set(arrays) & curves.keys():
            raise ValueError("Duplicate LRG curve cohort")
        curves.update(arrays)
    if {row["measurement_bin"] for row in rows} != set(LRG_BINS):
        raise ValueError("Primary LRG cohort is incomplete")
    pairs = paired_lrg_rows(rows, grid)
    return dict(
        rows=rows, curves=curves, pairs=pairs, summary=lrg_summaries(rows, pairs, grid), grid=grid
    )


def load_density_forecast(path):
    """Read the conditional density forecast with its signed contributions and flags."""
    return validate_forecast(read_json(path))


def load_lrg_volume(directory):
    """Read the saved volume forecast and its plotting grid."""
    tables = read_tables(directory, ["forecast.json", "summary.json"])
    config = read_json(Path(directory) / "provenance.json")["analysis_config"]
    return (
        tables["forecast.json"],
        tables["summary.json"],
        config["analysis"]["workloads"]["full_grid"],
    )


def saved_correlation_panels(record):
    """Decode the matched plot's saved pooled means and paired mock scatter.

    The export stores s**2 xi and the displayed error-bar endpoints. This preserves
    the complete displayed separation range without substituting fit-range vectors
    or averaging correlations instead of pooling counts.
    """
    artists = record["artists"]
    if {row["axis"] for row in artists} != set(range(10)):
        raise ValueError("Expected five main/residual correlation panels")
    panels = []
    for index in range(10):
        rows = [r for r in artists if r["axis"] == index]
        lines = [np.asarray(r["xy"], dtype=float) for r in rows if r["kind"] == "line"]
        errors = [np.asarray(r["xy"], dtype=float) for r in rows if r["kind"] == "errorbar"]
        tiers = TIERS if index % 2 == 0 else TIERS[1:]
        if len(lines) != 2 * len(tiers) or any(r["kind"] not in {"line", "errorbar"} for r in rows):
            raise ValueError("Unexpected saved correlation artist layout")
        panel, offset = {}, 0
        for i, tier in enumerate(tiers):
            curve, markers = lines[2 * i : 2 * i + 2]
            segments = np.asarray(errors[offset : offset + len(markers)])
            offset += len(markers)
            if (
                curve.ndim != 2
                or curve.shape[1] != 2
                or segments.shape != (len(markers), 2, 2)
                or not np.all(np.isfinite(curve))
                or not np.all(np.isfinite(segments))
            ):
                raise ValueError("Invalid saved correlation coordinates")
            np.testing.assert_allclose(segments[:, :, 0], np.repeat(markers[:, :1], 2, axis=1))
            np.testing.assert_allclose(segments[:, :, 1].mean(axis=1), markers[:, 1])
            error = (segments[:, 1, 1] - segments[:, 0, 1]) / 2
            if np.any(error < 0):
                raise ValueError("Negative saved mock scatter")
            panel[tier] = dict(curve=curve, markers=markers, error=error)
        if offset != len(errors):
            raise ValueError("Unassigned correlation error bars")
        panels.append(panel)
    return panels


def reconstruction_overlay_values(pooled, covariance, *, measurement_bin, condition):
    """Keep pooled-count curves and the common post-reconstruction mean error."""
    ids = np.asarray(covariance["realization_ids"])
    require(
        ids.shape == (25,) and np.array_equal(np.sort(ids), np.arange(25)),
        "Reconstruction overlay requires all 25 baseline realizations",
    )
    edges = np.asarray(covariance["fit_edges"])
    poles = np.asarray(covariance["multipoles"])
    require(np.count_nonzero(poles == 0) == 1, "Overlay requires one monopole")
    diagonal = np.diag(covariance["mean_covariance"])
    require(
        diagonal.size == len(poles) * (len(edges) - 1)
        and np.isfinite(diagonal).all()
        and np.all(diagonal > 0),
        "Invalid reconstruction mean errors",
    )
    np.testing.assert_allclose(
        covariance["mean_covariance"], covariance["covariance"] / len(ids), rtol=1e-12
    )
    sigma = np.sqrt(diagonal.reshape(len(poles), -1)[np.flatnonzero(poles == 0)[0]])
    result = dict(measurement_bin=measurement_bin, condition=condition, realization_ids=ids)
    monopoles = {}
    for state in STATES:
        row = pooled[state]
        np.testing.assert_array_equal(row["ids"], ids)
        separation = np.asarray(row["pooled_separation"])
        values = np.asarray(row["pooled"])[0]  # summarize_estimators uses (0, 2).
        require(
            separation.shape == sigma.shape
            and values.shape == sigma.shape
            and np.isfinite(separation).all()
            and np.isfinite(values).all()
            and np.all(separation >= edges[:-1])
            and np.all(separation <= edges[1:]),
            "Invalid pooled reconstruction coordinates",
        )
        curve = np.column_stack((separation, separation**2 * values))
        result[state] = dict(curve=curve, markers=curve.copy(), error=separation**2 * sigma)
        monopoles[state] = values
    result["residual"] = np.column_stack(
        (
            pooled[STATES[1]]["pooled_separation"],
            (monopoles[STATES[1]] - monopoles[STATES[0]]) / sigma,
        )
    )
    result["error_definition"] = "Common post-reconstruction RascalC error on the 25-mock mean"
    return result


def load_reconstruction_overlay(root, *, measurement_bin, condition, expected_identity, sources):
    """Read explicitly selected, matched native counts and an existing covariance."""
    from pzbao.clustering import read_measurements
    from pzbao.covariance import read_route_covariance
    from pzbao.diagnostics.clustering import summarize_estimators

    require(measurement_bin in LRG_BINS and condition in TIERS, "Invalid overlay route")
    root = resolve_saved_path(root)
    for name in ("run.yaml", "analysis.yaml", "frozen_alpha_sn.json"):
        sources.read(root / name)
    run, routes = _run_routes(root, dict(measurement_bin=measurement_bin, condition=condition))
    require(
        all(run["metadata"].get(k) == expected_identity[k] for k in IDENTITY),
        "Overlay belongs to another measurement release",
    )
    require(
        len(routes) == 2 and {r["reconstruction_state"] for r in routes} == set(STATES),
        "Overlay requires exactly one pre/post route pair",
    )
    routes = {r["reconstruction_state"]: r for r in routes}
    post = read_route_covariance(run, routes[STATES[1]])
    for name in ("result.yaml", "covariance.json", "covariance.npz", "window.npz"):
        sources.read(post["directory"] / name)
    arrays = post["arrays"]
    pooled = {}
    for state in STATES:
        records = read_measurements(run, routes[state])
        estimators = {}
        for record in records:
            for name in ("result.yaml", "measurements.json", record["counts"]):
                sources.read(record["directory"] / name)
            if record["identity"]["cap"] == "NGC+SGC":
                mid = record["identity"]["realization_id"]
                require(mid not in estimators, "Duplicate overlay realization")
                estimators[mid] = record["estimator"]
        np.testing.assert_array_equal(sorted(estimators), arrays["realization_ids"])
        pooled[state] = summarize_estimators(
            estimators, arrays["realization_ids"].tolist(), arrays["fit_edges"]
        )
    result = reconstruction_overlay_values(
        pooled, arrays, measurement_bin=measurement_bin, condition=condition
    )
    result["measurement_identity"] = dict(expected_identity)
    return result


def footprint_map(sources, preparations, *, nside=128, realization_id=0, random_split=0):
    """HEALPix occupancy of each tracer's complete random footprint, both caps.

    One recorded random split per cap suffices at this resolution (~500 randoms
    per pixel). Bytes are checked against the preparation's sha256 and parsed in
    memory, so each catalogue is read from disk once.
    """
    import io

    import healpy
    from astropy.io import fits

    footprint = {}
    for tracer, preparation in preparations.items():
        rows = [
            row
            for row in preparation["sources"]
            if row["role"] == "random"
            and row["mock_id"] == realization_id
            and row["random_split"] == random_split
        ]
        require(
            sorted(row["cap"] for row in rows) == ["NGC", "SGC"],
            "Footprint requires one recorded random split per cap",
        )
        occupied = np.zeros(healpy.nside2npix(nside), dtype=bool)
        for row in rows:
            with fits.open(io.BytesIO(sources.read(row["path"], row["sha256"]))) as hdus:
                table = hdus[1].data
                occupied[healpy.ang2pix(nside, table["RA"], table["DEC"], lonlat=True)] = True
        footprint[tracer] = occupied
    return footprint


def survey_effective_volume(cosmology, z_edges, nbar_h3_mpc3, *, f_sky, p0_h3_mpc3):
    """Total V_eff in (Gpc/h)^3 summed over redshift shells of one survey footprint."""
    z_edges = np.asarray(z_edges, dtype=float)
    require(0 < f_sky <= 1 and np.all(np.diff(z_edges) > 0), "Invalid survey geometry")
    chi = np.asarray(cosmology.comoving_radial_distance(z_edges), dtype=float)
    shells = f_sky * 4 * np.pi / 3 * np.diff(chi**3)
    return float(effective_volume(shells, nbar_h3_mpc3, p0_h3_mpc3=p0_h3_mpc3).sum() / 1e9)


def surface_density_nbar(cosmology, z_edges, *, arcmin2_density, dndz):
    """Shell densities for a surface density distributed as ``dndz`` (normalized over z > 0)."""
    from scipy.integrate import quad

    total = quad(dndz, 0, np.inf, limit=200)[0]
    fraction = (
        np.array([quad(dndz, lo, hi)[0] for lo, hi in zip(z_edges[:-1], z_edges[1:])]) / total
    )
    chi = np.asarray(cosmology.comoving_radial_distance(np.asarray(z_edges)), dtype=float)
    per_steradian = arcmin2_density * (180 * 60 / np.pi) ** 2 * fraction
    return per_steradian / (np.diff(chi**3) / 3)


def read_precision_sources(sources, root, expected_identity):
    """Read a frozen count/volume preparation after checking its complete snapshot."""
    from pzbao.config import analysis_config_sha256

    root = resolve_saved_path(root)
    run = yaml.safe_load(sources.read(root / "run.yaml"))
    require(
        all(run[k] == v for k, v in expected_identity.items()), "Wrong frozen measurement release"
    )
    config = yaml.safe_load(sources.read(root / "analysis.yaml"))
    registry = sources.json(root / "frozen_alpha_sn.json")
    require(
        analysis_config_sha256(dict(analysis=config, frozen_alpha_sn=registry))
        == expected_identity["analysis_config_sha256"],
        "Changed configuration or frozen registry",
    )
    preparations = {}
    stage = "select" if run.get("interface_version") == 2 else "inputs"
    for tracer in config["tracers"]:
        directory = root / stage / "units" / tracer
        identity = yaml.safe_load(sources.read(directory / "result.yaml"))
        require(
            all(identity[k] == v for k, v in expected_identity.items()),
            "Wrong preparation identity",
        )
        preparations[tracer] = sources.json(directory / "preparation.json")
    return preparations, config, registry


# per-bin baseline: the unified-prior refit of the per-bin primary, read from its run roots.

#: Every route's fit and sample units are named from these route coordinates.
_UNIT_KEYS = ("measurement_bin", "condition", "reconstruction_state")


def _perbin_unit(run, stage, route, realization=None):
    name = "--".join(route[key] for key in _UNIT_KEYS)
    if stage == "fit":
        name += "--" + ("mean" if realization is None else f"mock-{realization}")
    return run["root"] / stage / "units" / name


def _perbin_profile_item(parameters, maximization, chi2, dof):
    """One realization's entry of the profiles mapping the figure cells consume."""
    item = {"valid": True, "chi2": chi2, "dof": dof}
    for name in PARAMETERS:
        parameter = next((p for p in parameters if p["basename"] == name), None)
        if parameter is None or parameter.get("fixed"):
            raise ValueError(f"individual profile has no varied {name}")
        record = interval(parameter, maximization)
        item["valid"] &= bool(record["optimizer_ok"])
        item[name + "_bf"] = record["value"]
        error = parameter.get("parabolic_error")
        item[name + "_err"] = np.nan if error is None else float(error)
        item[name + "_minos"] = float(record["half_width"]) if record["usable"] else np.nan
    return item


def _perbin_diagnostic_rows(route, mock, saved, profile, identity):
    """Per-parameter rows in the shape of the saved profile diagnostics."""
    rows = []
    for parameter in profile["parameters"]:
        if parameter["basename"] not in PARAMETERS:
            continue
        record = interval(parameter, profile["maximization"])
        rows.append(
            dict(
                measurement_bin=route["measurement_bin"],
                condition=route["condition"],
                state=route["reconstruction_state"],
                mock=mock,
                parameter=parameter["basename"],
                name=parameter.get("name", parameter["basename"]),
                basename=parameter["basename"],
                value=record["value"],
                parabolic_error=parameter.get("parabolic_error"),
                minos_bounds=record["minos_bounds"],
                interval_flags=record["interval_flags"],
                boundary=record["boundary"],
                optimizer={"maximization": profile["maximization"]},
                optimizer_ok=record["optimizer_ok"],
                category="valid" if record["usable"] else "flagged",
                width=record["half_width"] if record["usable"] else None,
                chi2=saved["chi2"],
                degrees_of_freedom=saved["degrees_of_freedom"],
                identity=identity,
                covariance_identity=saved["covariance_identity"],
            )
        )
    return rows


def _perbin_density_rows(sources, measurement, measurement_identity):
    """Radial selection per tier from the per-bin primary's published preparation."""
    from pzbao.catalog import build_mock_cosmology, comoving_shell_volume

    config = measurement["config"]
    cosmology = build_mock_cosmology(config)
    root = measurement["root"]
    stage = _preparation_stage(sources, root)
    rows = []
    for tracer, spec in config["tracers"].items():
        directory = root / stage / "units" / tracer
        identity = yaml.safe_load(sources.read(directory / "result.yaml"))
        if any(identity.get(k) != measurement_identity[k] for k in IDENTITY):
            raise ValueError("preparation identity differs from the per-bin measurement")
        prep = sources.json(directory / "preparation.json")
        if prep["tracer"] != tracer or prep["realization_ids"] != list(range(25)):
            raise ValueError("preparation tracer or realization grid differs")
        area = sum(spec["footprint"]["area_deg2"].values())
        for tier in TIERS:
            caps = prep["conditions"][tier]
            if set(caps) != set(spec["footprint"]["area_deg2"]):
                raise ValueError("preparation cap grid differs")
            edges = None
            counts = None
            for content in caps.values():
                radial = content["radial"]
                if edges is None:
                    edges = np.asarray(radial["z_edges"], dtype=float)
                    counts = np.zeros(len(edges) - 1)
                np.testing.assert_allclose(radial["z_edges"], edges, atol=1e-12)
                counts += np.asarray(radial["mean_counts"], dtype=float)
            volume = np.asarray(comoving_shell_volume(cosmology, edges, area), dtype=float)
            for shell in spec["measurement_bins"]:
                rows.append(
                    dict(
                        bin=shell["name"],
                        condition=tier,
                        edges=edges,
                        mean_counts=counts,
                        volume_mpc_h3=volume,
                        nbar_h3_mpc3=counts / volume,
                    )
                )
    return saved_density_rows(rows)


def load_perbin_baseline(fit_root, measurement_root, *, detection_bin="LRG1"):
    """The per-bin counterpart of load_baseline, built from run roots instead of a bundle.

    ``fit_root`` is the unified-prior refit of the per-bin primary (its fits and samples);
    ``measurement_root`` is the per-bin primary measurement run that refit records as its
    source. Every route of the five-bin, five-tier, two-state grid must carry a published
    mean fit, twenty-five realization fits and a complete sampling collection; a missing
    unit fails closed, as the bundle loader does. The mapping returned has the keys the
    baseline notebook reads, so the figure cells are unchanged.
    """
    from pzbao.config import UNIFIED_PRIOR, same_identity
    from pzbao.figures import style
    from pzbao.fitting import read_profiles
    from pzbao.run import load_run, routes_for

    inputs = Sources()
    fit_run, measurement = load_run(fit_root), load_run(measurement_root)
    for run in (fit_run, measurement):
        inputs.read(run["root"] / "run.yaml")
    if fit_run["metadata"].get("inference_definition") != UNIFIED_PRIOR:
        raise ValueError("the per-bin baseline is the unified-prior refit of the primary")
    recon = measurement["config"]["measurement"]["reconstruction"]
    if (
        measurement["metadata"].get("inference_definition") is not None
        or measurement["metadata"]["workload"] != "full_grid"
        or recon.get("scope") != "separate_measurement_bins"
        or "sensitivity_run" in recon
        or "transfer_run" in recon
    ):
        raise ValueError("the measurement run must be the per-bin primary")
    measurement_identity = {k: measurement["metadata"][k] for k in IDENTITY}
    source = fit_run["metadata"].get("source_run") or {}
    if any(source.get(k) != v for k, v in measurement_identity.items()):
        raise ValueError("the refit does not record this measurement run as its source")
    if detection_bin not in LRG_BINS:
        raise ValueError("Detection comparison requires a registered LRG bin")

    routes = [
        {k: route[k] for k in _ROUTE_KEYS}
        for route in routes_for(fit_run["config"], fit_run["metadata"]["workload"])
    ]
    identity_base = {k: fit_run["metadata"][k] for k in IDENTITY}
    draws, fits, mean_fits, profiles, diagnostics, records = {}, {}, {}, {}, [], []
    for route in routes:
        key = historical_key(route, style)
        # The mean fit: the published profile and its saved record.
        identity = {**route, "realization_id": None, **identity_base}
        unit = _perbin_unit(fit_run, "fit", route)
        if not (unit / "profiles.npy").exists():
            raise ValueError(f"Missing per-bin mean fit: {unit.name}")
        saved = inputs.json(unit / "fit.json")
        inputs.read(unit / "profiles.npy")
        profile = read_profiles(unit / "profiles.npy", identity=identity)
        if not same_identity(saved["identity"], identity):
            raise ValueError("Saved fit identity differs from its native profile")
        if saved["covariance_identity"] != measurement_identity:
            raise ValueError("Clustering and posterior measurement identities differ")
        np.testing.assert_allclose(saved["chi2"], profile["chi2"], rtol=1e-12)
        mean_fits[key] = saved
        intervals = {
            name: interval(
                next(p for p in profile["parameters"] if p["basename"] == name),
                profile["maximization"],
            )
            for name in PARAMETERS
        }
        # The individual realizations: the per-mock profiles the gain and forest figures use.
        members = {}
        for mock in range(25):
            each = _perbin_unit(fit_run, "fit", route, mock)
            if not (each / "profiles.npy").exists():
                raise ValueError(f"Incomplete individual-realization cohort: {each.name}")
            mock_identity = {**route, "realization_id": mock, **identity_base}
            mock_saved = inputs.json(each / "fit.json")
            inputs.read(each / "profiles.npy")
            mock_profile = read_profiles(each / "profiles.npy", identity=mock_identity)
            if mock_saved["covariance_identity"] != measurement_identity:
                raise ValueError("Individual profiles use a different measurement ensemble")
            members[mock] = _perbin_profile_item(
                mock_profile["parameters"],
                mock_profile["maximization"],
                float(mock_saved["chi2"]),
                mock_saved["degrees_of_freedom"],
            )
            diagnostics.extend(
                _perbin_diagnostic_rows(route, mock, mock_saved, mock_profile, mock_identity)
            )
        profiles[key] = members
        # The posterior: retained draws of every ensemble, burnt in once by the native reader.
        sample = _perbin_unit(fit_run, "sample", route)
        if not (sample / "sampling.json").exists():
            raise ValueError(f"Missing per-bin posterior: {sample.name}")
        summary = inputs.json(sample / "sampling.json")
        for row in summary["ensembles"]:
            inputs.read(sample / row["path"])
        columns = [
            load_chain_samples(fit_run["root"], selectors=route, parameter=name).reshape(-1)
            for name in PARAMETERS
        ]
        values = np.column_stack(columns)
        if not np.isfinite(values).all():
            raise ValueError("Invalid retained native draws")
        covariance = np.cov(values, rowvar=False, ddof=1)
        draws[key] = values
        fits[key] = dict(
            qiso_std=float(values[:, 0].std(ddof=1)),
            qiso_mean=float(values[:, 0].mean()),
            qap_std=float(values[:, 1].std(ddof=1)),
        )
        limits = [
            f"ensemble {row['ensemble_index']} {name}: {flags['unavailable_reason']}"
            for row in summary["ensembles"]
            for name, flags in row["diagnostics"]["parameters"].items()
            if name in PARAMETERS and flags.get("unavailable_reason")
        ]
        records.append(
            dict(
                route=route,
                status="available",
                identity=identity,
                covariance_identity=saved["covariance_identity"],
                source=str(sample),
                mean=values.mean(axis=0).tolist(),
                covariance=covariance.tolist(),
                fom=float(1 / np.sqrt(np.linalg.det(covariance))),
                retained_draws=int(len(values)),
                profile=dict(source=str(unit / "profiles.npy"), identity=identity),
                profile_flagged=not all(item["usable"] for item in intervals.values()),
                prior_contact={name: bool(item["boundary"]) for name, item in intervals.items()},
                sampling_limits=limits,
            )
        )
    validate_grid(records, measurement_identity)
    joint = joint_constraint_rows(draws, profiles, style, seed=joint_bootstrap_seed())
    lookup = {
        (
            r["route"]["measurement_bin"],
            r["route"]["condition"],
            r["route"]["reconstruction_state"],
        ): r
        for r in records
    }
    fom_pairs = []
    for name in BINS:
        for tier in TIERS:
            pre, post = (lookup[(name, tier, state)] for state in STATES)
            fom_pairs.append(
                dict(
                    measurement_bin=name,
                    condition=tier,
                    pre=pre["fom"],
                    post=post["fom"],
                    gain_percent=100.0 * (post["fom"] / pre["fom"] - 1.0),
                    profile_flagged=bool(pre["profile_flagged"] or post["profile_flagged"]),
                    sampling_limited=bool(pre["sampling_limits"] or post["sampling_limits"]),
                    covariance_warning=False,
                )
            )
    posterior_summary = []
    for record in records:
        values = draws[historical_key(record["route"], style)]
        for index, parameter in enumerate(PARAMETERS):
            posterior_summary.append(
                dict(
                    record["route"],
                    parameter=parameter,
                    mean=float(values[:, index].mean()),
                    sd=float(values[:, index].std(ddof=1)),
                    profile_flagged=record["profile_flagged"],
                    prior_contact=record["prior_contact"][parameter],
                    sampling_limits=record["sampling_limits"],
                )
            )
    density = _perbin_density_rows(inputs, measurement, measurement_identity)
    inputs.unchanged()
    return dict(
        draws=draws,
        fits=fits,
        mean_fits=mean_fits,
        profiles=profiles,
        joint=joint,
        correlation=[],
        fom=fom_pairs,
        posterior_summary=posterior_summary,
        posterior_records=records,
        profile_diagnostics=diagnostics,
        measurement_identity=measurement_identity,
        density=density,
        density_axes=None,
        sources=inputs,
    )
