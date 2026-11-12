"""Reusable covariance diagnostics. Inputs and experiment choices are explicit arguments."""

from __future__ import annotations

import numpy as np

from pzbao.artifacts import resolve_saved_path

from ..covariance import tune_shot_noise_rescaling

import copy

from pathlib import Path

import hashlib

import importlib.util

import json

from scipy.linalg import cho_factor, cho_solve

from scipy.special import eval_legendre, spherical_jn


def leave_one_out_calibration(components, vectors, realization_ids, *, search_limits, points):
    """Repeat the configured scalar objective after omitting each realization in turn.

    The smooth radial target, RascalC components and estimator weights stay fixed. The
    resulting spread describes sensitivity to the calibration vectors, not independent
    validation or uncertainty propagated through the full shared-input calculation.
    """
    values = np.asarray(vectors, dtype=float)
    identifiers = np.asarray(realization_ids)
    if (
        values.ndim != 2
        or values.shape[0] < 3
        or values.shape[1] < 1
        or (not np.all(np.isfinite(values)))
        or (identifiers.shape != (values.shape[0],))
        or (identifiers.dtype.kind not in "iu")
        or np.any(identifiers < 0)
        or (len(np.unique(identifiers)) != len(identifiers))
    ):
        raise ValueError(
            "calibration needs finite vectors and at least three unique integer mock IDs"
        )

    def calibrate(sample):
        return tune_shot_noise_rescaling(
            components,
            np.atleast_2d(np.cov(sample, rowvar=False, ddof=1)),
            search_limits=search_limits,
            points=points,
        )

    full = calibrate(values)
    omitted = []
    for index, identifier in enumerate(identifiers):
        result = calibrate(np.delete(values, index, axis=0))
        omitted.append(
            {
                "omitted_realization_id": int(identifier),
                "calibration_realization_ids": np.delete(identifiers, index).tolist(),
                "delta_from_full": result["shot_noise_rescaling"] - full["shot_noise_rescaling"],
                **result,
            }
        )
    return {
        "method": "leave_one_realization_out",
        "fixed_inputs": ["radial_selection", "covariance_components", "estimator_weights"],
        "interpretation": "Sensitivity to calibration vectors conditional on the published shared inputs; not independent coverage validation or full-chain uncertainty propagation.",
        "realization_ids": identifiers.tolist(),
        "full_ensemble": full,
        "leave_one_out": omitted,
    }


def covariance_intervention(current, *, scale=1.0, historical=None, mode="full"):
    current = np.asarray(current, dtype=float)
    if current.ndim != 2 or current.shape[0] != current.shape[1]:
        raise ValueError("covariance must be square")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("covariance scale must be finite and positive")
    matrices = [current] if historical is None else [current, np.asarray(historical, dtype=float)]
    for matrix in matrices:
        if matrix.shape != current.shape or not np.all(np.isfinite(matrix)):
            raise ValueError("covariance matrices must have identical finite axes")
        np.testing.assert_allclose(matrix, matrix.T, rtol=0, atol=1e-12)
        np.linalg.cholesky(matrix)
    if historical is None:
        if mode != "full":
            raise ValueError("historical covariance identity is required for decomposition")
        return current * scale
    historical = matrices[1]
    if mode == "full":
        result = historical
    elif mode == "amplitude":
        result = current * np.exp(np.mean(np.log(np.diag(historical) / np.diag(current))))
    elif mode == "correlation":
        ratio = np.sqrt(np.diag(current) / np.diag(historical))
        result = historical * ratio[:, None] * ratio[None, :]
    else:
        raise ValueError("unknown covariance decomposition")
    np.linalg.cholesky(result)
    return result * scale


def covariance_metrics(matrix, vectors):
    """Project scatter without ever inverting the rank-deficient sample covariance."""
    matrix = np.asarray(matrix, dtype=float)
    vectors = np.asarray(vectors, dtype=float)
    if vectors.ndim != 2 or vectors.shape[1] != matrix.shape[0] or len(vectors) != 25:
        raise ValueError("covariance diagnostic requires all 25 compatible vectors")
    covariance_intervention(matrix)
    scatter = np.cov(vectors, rowvar=False, ddof=1)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    residual = vectors - vectors.mean(axis=0)
    whitened = np.linalg.solve(np.linalg.cholesky(matrix), residual.T).T
    return {
        "diagonal": np.diag(matrix).tolist(),
        "scatter_diagonal": np.diag(scatter).tolist(),
        "variance_over_mock_scatter": (np.diag(matrix) / np.diag(scatter)).tolist(),
        "eigenvalues": eigenvalues.tolist(),
        "condition_number": float(eigenvalues[-1] / eigenvalues[0]),
        "eigenmode_scatter_over_prediction": (
            np.diag(eigenvectors.T @ scatter @ eigenvectors) / eigenvalues
        ).tolist(),
        "whitened_residual_norm_squared": np.sum(whitened**2, axis=1).tolist(),
        "mean_scaling": "individual covariance divided by 25 exactly once",
    }


class Unavailable(ValueError):
    """A required scientific input is missing or incompatible; never impute it."""


def infer_counts(dd, power):
    """Use companion NGC header and verified uniform-weight cap-summed DD."""
    north = int(power["data_size1"])
    weight = float(power["sum_data_weights1"]) / north if north > 0 else np.nan
    occupied = np.asarray(dd["ncounts"]) > 0
    if (
        north <= 0
        or not np.isfinite(weight)
        or weight <= 0
        or (not dd["autocorr"])
        or (not occupied.any())
    ):
        raise Unavailable("invalid historical auto-count/header metadata")
    ratio = np.asarray(dd["wcounts"])[occupied] / np.asarray(dd["ncounts"])[occupied]
    error = float(np.max(np.abs(ratio / weight**2 - 1)))
    remainder = float(dd["wnorm"]) / weight**2 - north * (north - 1)
    if not np.isfinite(error) or error > 1e-08 or (not np.isfinite(remainder)) or (remainder < 0):
        raise Unavailable("historical DD and uniform header weights disagree")
    root = (1 + np.sqrt(1 + 4 * remainder)) / 2
    integer_error = float(abs(root - round(root)))
    if integer_error > 0.01 or round(root) <= 0:
        raise Unavailable("historical SGC count has no positive integer solution")
    return ({"NGC": north, "SGC": int(round(root))}, error, integer_error)


def common_historical_matrix(matrices, metadata, expected_poles, actual_poles):
    """Consume a documented per-realization matrix, never infer scale from amplitude.

    Even identical matrices in 25 individual profiles do not prove cov_scale=1.
    A directly saved normalization declaration is required for the mean crossover.
    """
    if list(actual_poles) != list(expected_poles):
        raise Unavailable("historical and current fitted multipoles differ")
    values = np.asarray(matrices, dtype=float)
    if len(values) != 25 or len(metadata) != 25:
        raise Unavailable("historical covariance requires all 25 individual products")
    for attrs in metadata:
        declarations = [
            container
            for container in (attrs, attrs.get("observable", {}))
            if isinstance(container, dict)
        ]
        scales = [c["cov_scale"] for c in declarations if "cov_scale" in c]
        labels = [
            c["covariance_normalization"] for c in declarations if "covariance_normalization" in c
        ]
        if not scales and (not labels):
            raise Unavailable("historical covariance normalization is not directly recorded")
        if any((float(s) != 1.0 for s in scales)) or any((s != "individual_mock" for s in labels)):
            raise Unavailable("historical covariance is not documented as single-realization C")
    matrix = covariance_intervention(values[0])
    for candidate in values[1:]:
        if not np.array_equal(candidate, matrix):
            raise Unavailable("historical mocks use different matrices; no common mean covariance")
    return matrix


def swap_covariance(saved, matrix):
    """Replace only the likelihood matrices, keeping vectors/window byte-identical."""
    changed = copy.deepcopy(saved)
    matrix = covariance_intervention(matrix)
    if matrix.shape != saved["arrays"]["covariance"].shape:
        raise Unavailable("historical covariance dimension differs")
    count = len(saved["arrays"]["realization_ids"])
    if count != 25:
        raise ValueError("the crossover requires 25 mocks")
    changed["arrays"].update(covariance=matrix, mean_covariance=matrix / count)
    for key in ("vectors", "separations", "fit_edges", "multipoles", "realization_ids"):
        np.testing.assert_array_equal(changed["arrays"][key], saved["arrays"][key])
    for key in saved["window"]:
        np.testing.assert_array_equal(changed["window"][key], saved["window"][key])
    return changed


def require_attainable(counts, preparation, case, bin_names, mock_ids):
    if preparation["realization_ids"] != mock_ids:
        raise Unavailable("availability evidence belongs to another mock cohort")
    evidence = {}
    for cap in ("NGC", "SGC"):
        available = np.asarray(
            preparation["conditions"][case["condition"]][cap]["attainable_counts_by_mock"]
        )
        if (
            available.shape != (len(mock_ids), len(bin_names))
            or available.dtype.kind not in "iu"
            or np.any(available < 0)
        ):
            raise Unavailable("invalid per-mock attainable-count evidence")
        evidence[cap] = dict(zip(bin_names, available.min(axis=0).tolist()))
        for name in bin_names:
            if counts[cap][name] > evidence[cap][name]:
                raise Unavailable(
                    f"{name} {cap}: historical target {counts[cap][name]} exceeds common availability {evidence[cap][name]}"
                )
    return evidence


def covariance_correlation(matrix):
    matrix = np.asarray(matrix, dtype=float)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or (not np.isfinite(matrix).all())
        or (not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-14))
        or np.any(np.diag(matrix) <= 0)
    ):
        raise ValueError("invalid covariance axes, symmetry or diagonal")
    sd = np.sqrt(np.diag(matrix))
    return matrix / np.outer(sd, sd)


_bgs_depth_robustness_MATCHED_KEYS = (
    "route",
    "cap",
    "geometry",
    "realization_ids",
    "run_name",
    "code_commit",
    "analysis_config_sha256",
    "ordering",
    "seed",
    "nthread",
    "random_coordinate_state",
    "data_count",
    "random_count",
    "multipoles",
    "separation_edges_mpc_h",
)


def check_matched_caps(previous, actual, high):
    if any((actual[key] != previous[key] for key in _bgs_depth_robustness_MATCHED_KEYS)):
        raise ValueError("deeper cap changes more than numerical integration depth")
    if actual["n_loops"] != high:
        raise ValueError("deeper cap did not execute the requested depth")


def compare_matrices(reference, candidate):
    """Retain signed differences and generalized eigenvalues for a matched matrix pair."""
    from scipy.linalg import eigvalsh

    reference, candidate = np.asarray(reference), np.asarray(candidate)
    if (
        reference.shape != candidate.shape
        or reference.ndim != 2
        or reference.shape[0] != reference.shape[1]
    ):
        raise ValueError("covariance axes differ")
    for matrix in (reference, candidate):
        if not np.isfinite(matrix).all() or not np.allclose(matrix, matrix.T):
            raise ValueError("covariance is non-finite or asymmetric")
        np.linalg.cholesky(matrix)
    return {
        "difference": candidate - reference,
        "diagonal_ratio": np.diag(candidate) / np.diag(reference),
        "generalized_eigenvalues": eigvalsh(candidate, reference),
    }


def plot_correlation(matrix, *, ax=None):
    from ..figures.style import axes

    ax = axes(ax)
    correlation = covariance_correlation(matrix)
    artist = ax.imshow(correlation, vmin=-1, vmax=1, cmap="RdBu_r", origin="lower")
    ax.set(xlabel="Vector index", ylabel="Vector index")
    return ax, {"correlation": correlation, "artist": artist}


CONFIG = Path(__file__).resolve().parents[1] / "config/analysis.yaml"
BINS = ("BGS1", "BGS2", "LRG1", "LRG2", "LRG3")
TIERS = ("native", "sigma_0.001", "sigma_0.003", "sigma_0.01", "sigma_0.03")
STATES = ("prereconstruction", "postreconstruction")
REVISION = "2.11-F3"


def require(value, message):
    if not value:
        raise ValueError(message)


class Inputs:
    def __init__(self):
        self.hashes = {}

    def read(self, path, expected=None):
        path = resolve_saved_path(path).resolve()
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        require(
            self.hashes.setdefault(str(path), digest) == digest
            and (expected is None or expected == digest),
            f"Changed input: {path}",
        )
        return data

    def json(self, path):
        return json.loads(self.read(path))

    def verify(self):
        for path, digest in list(self.hashes.items()):
            self.read(path, digest)


def noise_moments(volume, density, p0):
    """FKP radial moments; uniform density gives (1/n²V, 1/nV, 1/V)."""
    volume, density = np.asarray(volume, float), np.asarray(density, float)
    require(
        volume.shape == density.shape and volume.ndim == 1 and len(volume) > 0,
        "Radial volume/density axes differ",
    )
    require(
        np.isfinite(volume).all()
        and np.isfinite(density).all()
        and np.all(volume > 0)
        and np.all(density >= 0)
        and np.isfinite(p0)
        and p0 > 0,
        "Invalid radial density or volume",
    )
    weights = 1 / (1 + density * p0)
    normalization = np.sum(volume * density**2 * weights**2)
    require(normalization > 0, "Empty selection")
    return np.array(
        [np.sum(volume * density**m * weights**4) / normalization**2 for m in (2, 3, 4)]
    )


def forecast_sigma_perp(artifact, target_density):
    """Use the existing exact displacement-noise variance, without adding a baseline."""
    require(
        artifact["kind"] == "analytic_postreconstruction_sigma_perp"
        and artifact["formula"]["version"] == 2
        and artifact["p_shot"] == 1.0,
        "Unidentified reconstruction response",
    )
    n = float(artifact["attained_density_h3_mpc3"])
    rec = float(artifact["reconstruction_variance_mpc_h2"])
    shot = float(artifact["shot_noise_variance_mpc_h2"])
    require(
        np.isfinite([n, target_density, rec, shot]).all() and min(n, target_density, rec, shot) > 0,
        "Invalid reconstruction inputs",
    )
    np.testing.assert_allclose(
        np.sqrt(rec + shot), artifact["analytic_center_mpc_h"], rtol=1e-12, atol=1e-12
    )
    return float(np.sqrt(rec + shot * n / target_density))


def trapezoid_weights(x):
    x = np.asarray(x, float)
    require(
        x.ndim == 1 and len(x) > 1 and np.isfinite(x).all() and np.all(np.diff(x) > 0),
        "Invalid integration axis",
    )
    result = np.empty_like(x)
    result[0], result[-1] = (x[1] - x[0]) / 2, (x[-1] - x[-2]) / 2
    result[1:-1] = (x[2:] - x[:-2]) / 2
    return result


def gaussian_xi_covariance(
    k, mu, muweights, power, separations, shell_volumes, window, moments, multipoles=(0, 2, 4)
):
    """Full angular Gaussian signal covariance plus analytically integrated Poisson noise.

    The window projects native fine radial cells and physical multipoles into the
    fitted vector. This is a fixed-window approximation, not RascalC integration.
    """
    k, mu, muweights, power, s, shells, w = map(
        np.asarray, (k, mu, muweights, power, separations, shell_volumes, window)
    )
    a2, a3, a4 = np.asarray(moments, float)
    require(
        power.shape == (len(k), len(mu))
        and muweights.shape == mu.shape
        and s.shape == shells.shape
        and w.shape[1] == len(s) * len(multipoles),
        "Gaussian covariance axes differ",
    )
    require(
        all(np.isfinite(x).all() for x in (power, s, shells, w, moments))
        and np.all(power >= 0)
        and np.all(shells > 0)
        and min(a2, a3, a4) > 0,
        "Invalid Gaussian covariance inputs",
    )
    require(np.all((mu > 0) & (mu < 1)) and np.all(muweights > 0), "Invalid half-range rule")
    np.testing.assert_allclose(muweights.sum(), 1.0, atol=1e-12, rtol=0)
    signal = a4 * power**2 + 2 * a3 * power
    radial = [spherical_jn(ell, s[:, None] * k[None, :]) for ell in multipoles]
    angular = [eval_legendre(ell, mu) for ell in multipoles]
    measure = k**2 * trapezoid_weights(k) / np.pi**2
    blocks = []
    for i, ell in enumerate(multipoles):
        row = []
        for j, other in enumerate(multipoles):
            angular_signal = signal @ (muweights * angular[i] * angular[j])
            sign = (-1.0) ** ((ell + other) // 2)
            scale = sign * (2 * ell + 1) * (2 * other + 1)
            block = (radial[i] * (measure * angular_signal * scale)) @ radial[j].T
            if ell == other:
                block += np.diag(2 * (2 * ell + 1) * a2 / shells)
            row.append(block)
        blocks.append(row)
    covariance = w @ np.block(blocks) @ w.T
    covariance = (covariance + covariance.T) / 2
    np.linalg.cholesky(covariance)
    return covariance


def joint_information(jacobian, covariance, prior_precision, names):
    """Marginalize the complete parameter space before taking the AP determinant."""
    j, c, prior = map(lambda x: np.asarray(x, float), (jacobian, covariance, prior_precision))
    require(
        j.ndim == 2
        and c.shape == (j.shape[0],) * 2
        and prior.shape == (j.shape[1],) * 2
        and len(names) == j.shape[1]
        and len(set(names)) == len(names),
        "Fisher axes differ",
    )
    require(all(np.isfinite(x).all() for x in (j, c, prior)), "Nonfinite Fisher inputs")
    np.testing.assert_allclose(c, c.T, rtol=1e-12, atol=0)
    fisher = j.T @ cho_solve(cho_factor(c, lower=True), j) + prior
    scale = np.sqrt(np.diag(fisher))
    require(np.isfinite(scale).all() and np.all(scale > 0), "Unconstrained forecast coordinate")
    normalized = fisher / scale[:, None] / scale[None, :]
    inverse = cho_solve(cho_factor(normalized, lower=True), np.eye(len(names)))
    inverse /= scale[:, None] * scale[None, :]
    indices = [names.index(name) for name in ("qiso", "qap")]
    joint = inverse[np.ix_(indices, indices)]
    factor = np.linalg.cholesky(joint)
    return dict(
        fom=float(1 / np.prod(np.diag(factor))),
        covariance=joint.tolist(),
        parameter_names=list(names),
        condition_number=float(np.linalg.cond(normalized)),
    )


def ratio_terms(current, density, total):
    require(
        np.isfinite([current, density, total]).all() and min(current, density, total) > 0,
        "FoMs must be positive and finite",
    )
    rn, rt = density / current, total / current
    return dict(
        density_ratio=float(rn),
        total_ratio=float(rt),
        density_contribution=float(rn - 1),
        area_contribution=float(rt - rn),
    )


def paired_gains(rows):
    """Density changes reconstruction gain by R_density,post / R_density,pre."""
    result = []
    for name in BINS:
        for tier in TIERS:
            pair = {
                r["route"]["reconstruction_state"]: r
                for r in rows
                if r["route"]["measurement_bin"] == name and r["route"]["condition"] == tier
            }
            if len(pair) != 2 or any(r["status"] != "available" for r in pair.values()):
                continue
            pre, post = (pair[s] for s in STATES)
            gains = {
                arm: post["results"][arm]["fom"] / pre["results"][arm]["fom"]
                for arm in ("current", "target", "noise_only", "target_area")
            }
            ratio = post["density_ratio"] / pre["density_ratio"]
            np.testing.assert_allclose(gains["target"] / gains["current"], ratio, rtol=1e-12)
            result.append(
                dict(
                    measurement_bin=name,
                    condition=tier,
                    density_target_available=pre["density_target_available"],
                    reconstruction_gain=gains,
                    density_reconstruction_gain_ratio=ratio,
                    density_reconstruction_gain_percent=100 * (ratio - 1),
                )
            )
    return result


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configured_target_fsky(inputs=None):
    """Read the forecast setting without importing a frozen science checkout's defaults."""
    import yaml

    inputs = Inputs() if inputs is None else inputs
    schema_path = CONFIG.with_name("schema_validation.py")
    inputs.read(schema_path)
    schema = load_module("forecast_configuration_schema", schema_path)
    config = yaml.load(inputs.read(CONFIG), Loader=schema._UniqueKeyLoader)
    schema.validate_analysis_config(config)
    return config["reporting"]["forecast_target_fsky"]


def area_factor(cap_areas, target_fsky):
    full_sky = 4 * np.pi * (180 / np.pi) ** 2
    require(
        set(cap_areas) == {"NGC", "SGC"}
        and all(np.isfinite(v) and v > 0 for v in cap_areas.values())
        and 0 < sum(cap_areas.values()) <= full_sky
        and 0 < target_fsky <= 1,
        "Invalid forecast footprint or sky fraction",
    )
    return target_fsky * full_sky / sum(cap_areas.values())


def radial_cells(preparation, run, route, cosmology):
    """Normalize the frozen radial shape to the attained count in each bin/cap."""
    config = run["config"]
    spec = config["tracers"][route["tracer"]]
    shell = next(x for x in spec["measurement_bins"] if x["name"] == route["measurement_bin"])
    volumes, densities = [], []
    for cap in ("NGC", "SGC"):
        group = preparation["conditions"][route["condition"]][cap]
        edges = np.asarray(group["radial"]["z_edges"])
        counts = np.asarray(group["radial"]["mean_counts"])
        mask = (edges[:-1] >= shell["zmin"] - 1e-10) & (edges[1:] <= shell["zmax"] + 1e-10)
        local = np.r_[edges[:-1][mask], edges[1:][mask][-1]]
        np.testing.assert_allclose(local[[0, -1]], [shell["zmin"], shell["zmax"]], atol=1e-10)
        volume = np.diff(np.asarray(cosmology.comoving_radial_distance(local)) ** 3)
        volume *= spec["footprint"]["area_deg2"][cap] * (np.pi / 180) ** 2 / 3
        np.testing.assert_allclose(volume.sum(), group["volumes_mpc_h3"][shell["name"]], rtol=1e-9)
        count = counts[mask]
        require(
            np.isfinite(count).all() and np.all(count >= 0) and count.sum() > 0,
            "Invalid frozen radial shape",
        )
        count = count * group["counts"][shell["name"]] / count.sum()
        volumes.extend(volume)
        densities.extend(count / volume)
    return np.asarray(volumes), np.asarray(densities)


def parameter_point(likelihood, metadata, config, tracer):
    for param in list(likelihood.all_params.select(solved=True)):
        param.update(derived=False)
    parameters = list(likelihood.varied_params)
    names = [p.name for p in parameters]
    point = {p.name: float(p.value) for p in parameters}
    centers = metadata["damping_prior"]["centers_mpc_h"]
    point.update(
        qiso=1.0,
        qap=1.0,
        b1=config["tracers"][tracer]["reconstruction"]["fiducial_bias"],
        sigmaper=centers["sigma_perp"],
        sigmapar=centers["sigma_parallel"],
        sigmas=centers["sigma_streaming"],
    )
    require(set(point) == set(names), "Unexpected native forecast parameter namespace")
    priors, steps = [], []
    for p in parameters:
        require(p.prior.dist in ("norm", "uniform"), "Unsupported native prior curvature")
        priors.append(1 / p.prior.scale**2 if p.prior.dist == "norm" else 0.0)
        delta = p.delta
        steps.append(min(float(delta[1]), float(delta[2])) / 2)
    require(np.all(np.asarray(steps) > 0), "Missing native derivative steps")
    return names, point, np.asarray(steps), np.diag(priors)


def derivative_matrix(likelihood, observable, names, point, steps):
    columns = []
    for name, step in zip(names, steps):
        param = likelihood.all_params[name]
        lo, hi = param.prior.limits
        require(
            lo < point[name] - step and point[name] + step < hi,
            f"Forecast derivative touches support: {name}",
        )
        likelihood(**dict(point, **{name: point[name] + step}))
        plus = np.array(observable.flattheory, dtype=float)
        likelihood(**dict(point, **{name: point[name] - step}))
        minus = np.array(observable.flattheory, dtype=float)
        columns.append((plus - minus) / (2 * step))
    likelihood(**point)
    return np.column_stack(columns)


def native_power(theory, point):
    """Capture angular signal at the point already evaluated by the parent likelihood.

    Creating a separate child pipeline reinitializes the calculator and discards
    an instance-level projection hook. Replay only its native calculate method,
    with the existing dependency state and resolved input values instead.
    """
    power = theory.pt
    parameters = dict(power.runtime_info.input_values)
    for name, basename in power.runtime_info.input_names.items():
        if name in point:
            np.testing.assert_allclose(parameters[basename], point[name], rtol=0, atol=0)
    saved_poles = np.array(power.power, dtype=float, copy=True)
    capture = {}
    original = power.to_poles

    def record(pkmu):
        capture["pkmu"] = np.array(pkmu, dtype=float)
        return original(pkmu)

    power.to_poles = record
    try:
        power.calculate(**parameters)
    finally:
        power.to_poles = original
    require("pkmu" in capture, "Native angular power was not evaluated")
    np.testing.assert_allclose(power.power, saved_poles, rtol=1e-12, atol=1e-12)
    mu = np.asarray(power.mu).ravel()
    nodes, weights = np.polynomial.legendre.leggauss(2 * len(mu))
    np.testing.assert_allclose(mu, nodes[len(mu) :], atol=1e-12, rtol=0)
    weights = (weights[len(mu) :] + weights[len(mu) - 1 :: -1]) / 2
    return np.asarray(power.k), mu, weights, capture["pkmu"]


def forecast_route(
    run,
    route,
    *,
    target_fsky,
    check_numerics=False,
    index=None,
    profile=None,
    n_realizations=None,
):
    """Calculate the retained conditional forecast from a loaded native run and explicit route.

    This opt-in diagnostic keeps the frozen window, priors and mean normalization. By default the
    data vector is the mean of the source's realizations (C/25); ``n_realizations=1`` forecasts
    one survey, as a single-realization precision comparison needs. The prior curvature is the
    same in both. It writes no products and does not replace the run covariance or measured results.
    """
    from pzbao import catalog, covariance, likelihood as native_likelihood, selection

    config = run["config"]
    require(
        "density_run" not in config["selection"],
        "A matched-density run has no forecast target: its tier targets are the primary's",
    )
    provider = covariance.likelihood_covariance_run(run, route)
    saved = covariance.read_route_covariance(provider, route)
    validity = covariance.projected_structural_verdict(
        {key: saved["arrays"][key] for key in ("c2", "c3", "c4")},
        shot_noise_rescaling=saved["metadata"]["shot_noise"]["shot_noise_rescaling"],
    )
    require(validity["structural_test_passes"], "Invalid source covariance for forecast")
    theory, observable, likelihood, metadata, arrays, _ = native_likelihood._route_likelihood(
        run, dict(route, realization_id=None), method="mcmc"
    )
    require(
        metadata["ap_mode"] == "qisoqap" and config["fitting"]["broadband_model"] == "power3",
        "Expected joint power3 likelihood",
    )
    identity = {
        key: run["metadata"][key] for key in ("run_name", "code_commit", "analysis_config_sha256")
    }
    identity.update(route)
    if profile is None:
        profile = dict(parameters=[], unavailable_reason="No identified profile supplied.")
    nmock = len(arrays["realization_ids"])
    normalization = nmock if n_realizations is None else int(n_realizations)
    require(
        1 <= normalization <= nmock, "Forecast normalization must be 1 to the realization count"
    )
    np.testing.assert_array_equal(arrays["realization_ids"], np.arange(25))
    source = run.get("source_run", run)
    preparation = selection.read_preparation(source, route["tracer"])
    cosmology = catalog.build_mock_cosmology(config)
    volume, density = radial_cells(preparation, run, route, cosmology)
    attained = float(np.sum(volume * density) / volume.sum())
    np.testing.assert_allclose(attained, metadata["attained_density_h3_mpc3"], rtol=1e-12)
    tier = next(t for t in config["selection"]["tiers"] if t["label"] == route["condition"])
    target = (tier["target_nbar_h3_mpc3"] or {}).get(route["measurement_bin"])
    target_density = np.full_like(density, target) if target is not None else density.copy()
    p0 = config["tracers"][route["tracer"]]["fkp_p0_h3_mpc3"]
    moments = {
        "current": noise_moments(volume, density, p0),
        "target": noise_moments(volume, target_density, p0),
    }
    names, current_point, steps, prior = parameter_point(
        likelihood, metadata, config, route["tracer"]
    )
    target_point = dict(current_point)
    if target is not None and route["reconstruction_state"] == "postreconstruction":
        target_point["sigmaper"] = forecast_sigma_perp(
            metadata["analytic_transverse_damping"], target
        )
    native_edges = np.asarray(saved["window"]["sedges"])
    s = np.asarray(observable.wmatrix.sin)
    centers = 0.75 * np.diff(native_edges**4) / np.diff(native_edges**3)
    selection = np.array(
        [np.flatnonzero(np.isclose(centers, value, rtol=0, atol=1e-9)).item() for value in s]
    )
    shells = 4 * np.pi / 3 * np.diff(native_edges**3)[selection]
    window = np.asarray(observable.wmatrix.matrix_full)
    require(tuple(observable.wmatrix.ellsin) == (0, 2, 4), "Changed window multipoles")
    areas = config["tracers"][route["tracer"]]["footprint"]["area_deg2"]
    factor = area_factor(areas, target_fsky)
    results, debug = {}, {}
    for label, point in (("current", current_point), ("target", target_point)):
        jacobian = derivative_matrix(likelihood, observable, names, point, steps)
        k, mu, muweights, pkmu = native_power(theory, point)
        matrix = (
            gaussian_xi_covariance(k, mu, muweights, pkmu, s, shells, window, moments[label])
            / normalization
        )
        results[label] = joint_information(jacobian, matrix, prior, names)
        if label == "target":
            results["target_area"] = joint_information(jacobian, matrix / factor, prior, names)
        else:
            noise_only = (
                gaussian_xi_covariance(k, mu, muweights, pkmu, s, shells, window, moments["target"])
                / normalization
            )
            results["noise_only"] = joint_information(jacobian, noise_only, prior, names)
        if check_numerics:
            fine = derivative_matrix(likelihood, observable, names, point, steps / 2)
            finer = joint_information(fine, matrix, prior, names)
            keep = np.unique(np.r_[np.arange(0, len(k), 2), len(k) - 1])
            coarse = (
                gaussian_xi_covariance(
                    k[keep], mu, muweights, pkmu[keep], s, shells, window, moments[label]
                )
                / normalization
            )
            coarse_fom = joint_information(jacobian, coarse, prior, names)["fom"]
            debug[label] = dict(
                derivative_half_step_fom=finer["fom"],
                derivative_fractional_change=finer["fom"] / results[label]["fom"] - 1,
                integration_coarse_fom=coarse_fom,
                integration_fractional_change=coarse_fom / results[label]["fom"] - 1,
            )
    if target is None:
        np.testing.assert_allclose(results["target"]["fom"], results["current"]["fom"], rtol=1e-10)
    return dict(
        index=index,
        route=route,
        status="available",
        identity=identity,
        forecast_kind="conditional Gaussian Fisher; fixed RR window and z_eff",
        results=results,
        numerics=debug,
        n_mocks_normalization=normalization,
        attained_nbar=attained,
        target_nbar=target,
        density_target_available=target is not None,
        area_factor=factor,
        target_fsky=target_fsky,
        cap_area_deg2=areas,
        current_sigma_perp=current_point["sigmaper"],
        target_sigma_perp=target_point["sigmaper"],
        density_noise_only_ratio=results["noise_only"]["fom"] / results["current"]["fom"],
        **ratio_terms(
            results["current"]["fom"], results["target"]["fom"], results["target_area"]["fom"]
        ),
        source_profile=profile,
        source_covariance_validity=validity,
        baseline_priors={p.name: p.prior.__getstate__() for p in likelihood.varied_params},
        scientific_acceptance=False,
    )
