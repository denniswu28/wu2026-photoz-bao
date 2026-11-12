"""Separate-cap covariance calculations, native integration and descriptive diagnostics.

All arrays describe explicit separation/multipole axes. Execution, process isolation and final
publication belong to pzbao.run. Numerical depth selection never receives fit outcomes.
"""

from __future__ import annotations

import inspect
import json
import warnings
import traceback
from pathlib import Path
from typing import Any, List, Mapping, Sequence, Tuple

import numpy as np

from .config import (
    BGS_ANISOTROPIC,
    CORE_BIN_INFERENCE,
    CORE_SCAN_INFERENCE,
    CORE_POWER3_INFERENCE,
    TRANSFER_INFERENCE,
    UNIFIED_PRIOR_INFERENCE,
    same_identity,
    validate_inference_config,
)
from .storage import same_run_identity


def _component_source_run(run):
    """Keep reused measurements/components under their original immutable identity."""
    definition = run["metadata"].get("inference_definition")
    if "source_run" not in run:
        if definition is not None:
            raise ValueError("covariance reprojection lacks its measurement source")
        return run
    if definition != BGS_ANISOTROPIC:
        raise ValueError("only the BGS anisotropic definition permits covariance reprojection")
    source = run["source_run"]
    validate_inference_config(source["config"], run["config"], definition)
    identity = {
        "root": str(source["root"]),
        **{
            key: source["metadata"][key]
            for key in ("run_name", "code_commit", "analysis_config_sha256")
        },
    }
    if not same_run_identity(run["metadata"].get("source_run"), identity):
        raise ValueError("covariance measurement-source identity differs")
    return source


def likelihood_covariance_run(run, route):
    """Select the covariance with the likelihood's observed vector and calibration."""
    if run["metadata"].get("inference_definition") is None:
        return run.get("source_run", run)
    if run["metadata"]["inference_definition"] in (
        *TRANSFER_INFERENCE,
        *CORE_BIN_INFERENCE,
        *CORE_SCAN_INFERENCE,
        *UNIFIED_PRIOR_INFERENCE,
    ):
        source = run["source_run"]
        validate_inference_config(
            source["config"], run["config"], run["metadata"]["inference_definition"]
        )
        identity = {
            "root": str(source["root"]),
            **{
                key: source["metadata"][key]
                for key in ("run_name", "code_commit", "analysis_config_sha256")
            },
        }
        if not same_run_identity(run["metadata"].get("source_run"), identity):
            raise ValueError("inference covariance source identity differs")
        return source
    if run["metadata"]["inference_definition"] in CORE_POWER3_INFERENCE:
        # Power3 leaves the covariance unchanged: LRG reads its measurement source, BGS the
        # one published anisotropic route covariance recorded at run creation.
        if route["tracer"] != "BGS_BRIGHT-21.5":
            return run["source_run"]
        holders = [
            provider
            for provider in run.get("covariance_runs", [])
            if _publishes_route(provider, route)
        ]
        if len(holders) != 1:
            raise ValueError("BGS route needs exactly one anisotropic covariance provider")
        return likelihood_covariance_run(holders[0], route)
    _component_source_run(run)
    if route["tracer"] != "BGS_BRIGHT-21.5":
        raise ValueError("the BGS inference run does not supply LRG likelihoods")
    return run


def _publishes_route(run, route):
    import yaml

    handoff = run["root"] / "covariance" / "covariances.yaml"
    if not handoff.is_file():
        return False
    units = yaml.safe_load(handoff.read_text()).get("units", [])
    keys = ("tracer", "measurement_bin", "condition", "reconstruction_state")
    return any(all(item["unit"].get(key) == route[key] for key in keys) for item in units)


def _covariance_symmetry_atol(matrix):
    return np.finfo(float).eps * max(float(np.max(np.abs(matrix))), np.finfo(float).tiny) * 100


def complete_realizations(records, expected_ids):
    """Return one directly identified record per configured realization, in configuration order."""
    expected = list(expected_ids)
    ids = [record["realization_id"] for record in records]
    if (
        any(type(identifier) is not int for identifier in ids + expected)
        or len(set(ids)) != len(ids)
        or set(ids) != set(expected)
    ):
        raise ValueError(
            f"incomplete or duplicate realization IDs: expected {expected}, found {ids}"
        )
    lookup = dict(zip(ids, records))
    return [lookup[identifier] for identifier in expected]


def resolve_route_geometry(config, *, realization_ids):
    ids = list(realization_ids)
    if (
        not ids
        or any(type(identifier) is not int for identifier in ids)
        or len(set(ids)) != len(ids)
    ):
        raise ValueError("route geometry requires distinct integer realization identifiers")
    controls = config["fitting"]["covariance"]["rascalc"]["route_geometry"]
    rule = controls["selection_rule"]
    if rule == "lowest_realization_identifier_in_the_route":
        selected = min(ids)
    elif rule == "explicit_realization_identifier":
        selected = controls["explicit_realization_id"]
        if type(selected) is not int or selected not in ids:
            raise ValueError("explicit geometry realization is absent from this route")
    else:
        raise ValueError("unregistered route geometry selection rule")
    return {"selection_rule": rule, "realization_id": selected}


def cap_multipole_operators(rr_counts, mu_edges, *, multipoles):
    """Column-vector operators from finite-multipole angular reconstruction and RR weights.

    Matches RascalC's bin-integrated Legendre projection and bin-averaged inverse expansion.
    Input RR counts must already use each cap's own DD normalization. No scalar area weighting.
    Output uses multipole-major ordering, retains every supplied multipole, and is selected later.
    """
    from numpy.polynomial.legendre import Legendre

    if set(rr_counts) != {"NGC", "SGC"}:
        raise ValueError("covariance weights require both Galactic caps")
    mu = np.asarray(mu_edges, dtype=float)
    if (
        mu.ndim != 1
        or mu.size < 2
        or not np.all(np.isfinite(mu))
        or np.any(np.diff(mu) <= 0)
        or mu[0] != 0
        or mu[-1] != 1
    ):
        raise ValueError("cap operators require wrapped angular edges from zero to one")
    poles = list(multipoles)
    if poles != sorted(set(poles)) or any(
        type(ell) is not int or ell < 0 or ell % 2 for ell in poles
    ):
        raise ValueError("cap expansion multipoles must be distinct increasing even integers")
    counts = {cap: np.asarray(value, dtype=float) for cap, value in rr_counts.items()}
    if len({value.shape for value in counts.values()}) != 1:
        raise ValueError("cap RR axes disagree")
    for value in counts.values():
        if (
            value.ndim != 2
            or value.shape[1] != mu.size - 1
            or np.any(~np.isfinite(value))
            or np.any(value < 0)
        ):
            raise ValueError("invalid cap RR counts")
    total = counts["NGC"] + counts["SGC"]
    if np.any(total <= 0):
        raise ValueError("combined cap RR has an empty angular cell")
    integrals = np.stack([np.diff(Legendre.basis(ell).integ()(mu)) for ell in poles])
    inverse = integrals / np.diff(mu)
    projection = (2 * np.asarray(poles) + 1)[:, None] * integrals
    radial = total.shape[0]
    operators, weights = {}, {}
    for cap, value in counts.items():
        weight = value / total
        angular = np.einsum("om,rm,im->roi", projection, weight, inverse)
        matrix = np.zeros((len(poles) * radial, len(poles) * radial))
        for r in range(radial):
            indices = np.arange(len(poles)) * radial + r
            matrix[np.ix_(indices, indices)] = angular[r]
        operators[cap], weights[cap] = matrix, weight
    return operators, weights


def assemble_cap_components(cap_components, operators, *, selection=None):
    """Combine each component with W C W.T; neglect cross-cap terms explicitly."""
    if set(cap_components) != {"NGC", "SGC"} or set(operators) != set(cap_components):
        raise ValueError("final covariance assembly requires both caps")
    combined = None
    for cap in ("NGC", "SGC"):
        operator = np.asarray(operators[cap], dtype=float)
        if selection is not None:
            operator = np.asarray(selection, dtype=float) @ operator
        projected = project_raw_components(cap_components[cap], operator)
        combined = (
            projected
            if combined is None
            else {key: combined[key] + projected[key] for key in projected}
        )
    return combined


def combine_cap_subsamples(cap_samples, operators, *, alpha, selection=None):
    """Pair independent cap subsamples in index order, retaining noise from both integrals.

    Different integration depths are reduced to their greatest-common-divisor number of equal
    consecutive blocks. Block means preserve each cap mean and never duplicate one cap's samples.
    """
    if set(cap_samples) != {"NGC", "SGC"}:
        raise ValueError("subsample assembly requires both caps")
    sizes = []
    for parts in cap_samples.values():
        if (
            set(parts) != {"c2", "c3", "c4"}
            or len({np.asarray(v).shape for v in parts.values()}) != 1
        ):
            raise ValueError("cap subsample component axes disagree")
        shape = np.asarray(parts["c2"]).shape
        if len(shape) != 3 or shape[0] < 2 or shape[1] != shape[2]:
            raise ValueError("cap components require subsample-by-vector-by-vector axes")
        sizes.append(shape[0])
    count = int(np.gcd(*sizes))
    if count < 2:
        raise ValueError("cap depths have fewer than two matched equal-size blocks")
    grouped = {
        cap: {
            key: np.asarray(value).reshape(count, -1, *np.asarray(value).shape[1:]).mean(axis=1)
            for key, value in parts.items()
        }
        for cap, parts in cap_samples.items()
    }
    result = []
    for index in range(count):
        parts = {cap: {key: value[index] for key, value in grouped[cap].items()} for cap in grouped}
        result.append(
            combine_components(
                assemble_cap_components(parts, operators, selection=selection), alpha
            )
        )
    return np.asarray(result)


def covariance_diagnostics(matrix, samples=None):
    """Describe the published-alpha matrix; undefined optional diagnostics never veto it."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not np.all(np.isfinite(matrix)):
        raise ValueError("covariance is not a finite square matrix")
    record = {
        "eigenvalues": None,
        "condition_number": None,
        "positive_definite": None,
        "optional": {},
    }
    try:
        eigenvalues = np.linalg.eigvalsh(matrix)
        record.update(
            eigenvalues=eigenvalues.tolist(), positive_definite=bool(np.all(eigenvalues > 0))
        )
    except np.linalg.LinAlgError as exc:
        record["eigenvalues_unavailable_reason"] = str(exc)
    try:
        condition = float(np.linalg.cond(matrix))
        if not np.isfinite(condition):
            raise ValueError("condition number is non-finite")
        record["condition_number"] = condition
    except (ValueError, np.linalg.LinAlgError) as exc:
        record["condition_number_unavailable_reason"] = str(exc)
    if samples is not None:
        try:
            from RascalC.post_process.utils import compute_D_precision_matrix, compute_N_eff_D
            from RascalC.convergence_check_extra import convergence_check_extra_splittings
        except ImportError as exc:
            record["optional"] = {
                name: {"value": None, "reason": str(exc)}
                for name in ("half_splits", "inversion_bias", "effective_integration_count")
            }
            return record

        for name, calculate in (
            ("half_splits", lambda: convergence_check_extra_splittings(samples)),
            ("inversion_bias", lambda: compute_D_precision_matrix(samples, matrix)[0]),
        ):
            try:
                with np.errstate(invalid="raise", divide="raise", over="raise"):
                    value = calculate()
                record["optional"][name] = {"value": _diagnostic_values(value), "reason": None}
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                record["optional"][name] = {"value": None, "reason": str(exc)}
        bias = record["optional"]["inversion_bias"]["value"]
        try:
            if bias is None:
                raise ValueError("inversion-bias matrix is unavailable")
            effective = compute_N_eff_D(np.asarray(bias), print_function=lambda *args: None)
            record["optional"]["effective_integration_count"] = {
                "value": _diagnostic_values(effective),
                "reason": None,
            }
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            record["optional"]["effective_integration_count"] = {"value": None, "reason": str(exc)}
    return record


def _diagnostic_values(value):
    if isinstance(value, Mapping):
        return {str(key): _diagnostic_values(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_diagnostic_values(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return {"value": None, "reason": "diagnostic is non-finite"}
    return value.item() if isinstance(value, np.generic) else value


class NumericalExhaustion(RuntimeError):
    """All configured numerical depths failed; infrastructure failures never become this."""

    def __init__(self, history):
        self.history = history
        super().__init__("covariance remains numerically unusable at the configured depth ceiling")


class NonFiniteCovariance(ValueError):
    """Required covariance entries are non-finite; incompatible axes use ordinary ValueError."""


class NoPositiveDefiniteCandidate(ValueError):
    """No model covariance on the configured scalar grid admits a Cholesky factor."""


def is_infrastructure_failure(exc):
    if isinstance(exc, (TimeoutError, OSError)):
        return True
    # os.system can return a shell's signal-derived exit status after a killed compiled worker.
    # Other constructor/runtime errors remain errors; they are not evidence for deeper integrals.
    return isinstance(exc, RuntimeError) and any(
        f"exit code {code}" in str(exc) for code in (137, 143, -9, -15)
    )


def numerical_failure(components, alpha, config):
    """Assess only the required fitted covariance at its published shot-noise scalar."""
    allowed = config["fitting"]["covariance"]["rascalc"]["escalation_eligible_failures"]
    try:
        matrix = combine_components(components, alpha)
        np.linalg.cholesky(matrix)
        structural = projected_structural_verdict(components, shot_noise_rescaling=alpha)
        reason = None if structural["structural_test_passes"] else "rascalc_validity_failure"
    except np.linalg.LinAlgError:
        reason = "cholesky_failure"
    except NonFiniteCovariance:
        reason = "non_finite_covariance"
    if reason is not None and reason not in allowed:
        raise ValueError(f"unregistered numerical failure {reason}")
    return reason


def numerical_ladder(attempt, *, config, alpha, infrastructure_retries=1):
    """Run frozen numerical depths, with bounded same-depth retries per invocation.

    attempt(depth, retry_index) returns only combined fitted covariance components. It owns its
    output locations. Timeout/preemption/OSError re-raises the original error when the current
    invocation's retries are spent; old attempt directories cannot spend a new invocation's budget.
    """
    history = []
    for depth in config["fitting"]["covariance"]["rascalc"]["escalation_ladder_n_loops"]:
        failure = None
        for retry in range(infrastructure_retries + 1):
            try:
                components = attempt(depth, retry)
                break
            except NonFiniteCovariance:
                failure = "non_finite_covariance"
                break
            except Exception as exc:
                if not is_infrastructure_failure(exc):
                    raise
                if retry == infrastructure_retries:
                    raise
        if failure is None:
            try:
                failure = numerical_failure(
                    components, alpha(components) if callable(alpha) else alpha, config
                )
            except NoPositiveDefiniteCandidate:
                failure = "cholesky_failure"
        history.append({"n_loops": depth, "numerical_failure": failure})
        if failure is None:
            return components, history
    raise NumericalExhaustion(history)


def fixed_shot_noise_entry(registry, route):
    """Read historical scalar evidence; current production never selects this registry."""
    expected = {
        "measurement_bin": route["measurement_bin"],
        "sigma_z_tier": route["condition"],
        "reconstruction_state": route["reconstruction_state"],
    }
    entries = [entry for entry in registry["routes"] if entry["route"] == expected]
    if len(entries) > 1:
        raise ValueError("frozen shot-noise registry has duplicate route coordinates")
    if not entries:
        return None
    alpha = entries[0]["shot_noise_rescaling"]
    if type(alpha) not in (int, float) or not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("frozen route shot-noise scalar is not finite and positive")
    return dict(entries[0])


def fresh_shot_noise_controls(config):
    """Require the same explicit B source and numerical controls in producer and reader."""
    scalar = config["fitting"]["covariance"]["shot_noise_rescaling"]
    if scalar["source"] != "route_mock_scatter":
        raise ValueError("covariance requires fresh route-mock shot-noise calibration")
    tuning = scalar["tuning"]
    if tuning["search_limits"] is None or tuning["search_points"] is None:
        raise ValueError("fresh scalar search controls are not configured")
    return tuning


def read_raw_components(archive, *, dimension, subsamples):
    """Read all raw terms and subsamples; selection decides which blocks must be finite."""
    keys = {"c2": "c2_11", "c3": "c3_1,11", "c4": "c4_11,11"}
    components, samples = {}, {}
    with np.load(archive, allow_pickle=False) as raw:
        for name, key in keys.items():
            if key not in raw or key + "_full" not in raw:
                raise OSError(f"incomplete raw component archive: missing {key}")
            full, parts = np.asarray(raw[key + "_full"]), np.asarray(raw[key])
            if full.shape != (dimension, dimension) or parts.shape != (
                subsamples,
                dimension,
                dimension,
            ):
                raise OSError(f"incomplete raw component archive: incorrect {key} axes")
            components[name], samples[name] = full.copy(), parts.copy()
    return components, samples


def integrate_cap(
    *,
    estimator,
    xi_table,
    random_positions,
    random_weights,
    data_count,
    config,
    destination,
    nthread,
    n_loops,
    alpha,
    run_cov=None,
):
    """Integrate one cap using fixed-alpha post-processing and recover only complete raw output.

    Report references are deliberately absent from this interface, so a saved sample covariance
    cannot silently switch RascalC into calibration mode. A normally completed raw integration may
    survive optional post-processing diagnostics; killed or incomplete integrals cannot.
    """
    if run_cov is None:
        from RascalC.interface import run_cov
    controls = config["fitting"]["covariance"]["rascalc"]
    arguments = dict(controls["run_arguments"])
    if list(inspect.signature(run_cov).parameters) != list(controls["public_arguments"]):
        raise ValueError("RascalC public arguments differ from the resolved configuration")
    table = require_refinable_xi_table(xi_table)
    positions, weights = np.asarray(random_positions), np.asarray(random_weights)
    if (
        positions.shape != (len(weights), 3)
        or weights.ndim != 1
        or not len(weights)
        or not np.all(np.isfinite(positions))
        or not np.all(np.isfinite(weights))
        or np.any(weights < 0)
        or weights.sum() <= 0
        or type(data_count) is not int
        or data_count <= 0
    ):
        raise ValueError("cap random positions, weights or galaxy count are invalid")
    if int(nthread) < 1 or n_loops not in controls["escalation_ladder_n_loops"]:
        raise ValueError("invalid integration resource or depth")
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("fixed-alpha integration requires a finite positive scalar")
    root = Path(destination)
    if root.exists():
        raise FileExistsError(f"integration attempt already exists: {root}")
    raw_root = root / "raw"
    edges = np.asarray(estimator.edges[0], dtype=float)
    width = float(config["fitting"]["input_bin_width_mpc_h"])
    if not np.allclose(np.diff(edges), width, rtol=0, atol=1e-9):
        raise ValueError("RascalC counts must already be rebinned to the configured radial width")
    fit_edges = config["fitting"]["fit_edge_limits_mpc_h"]
    low = np.flatnonzero(np.isclose(edges, fit_edges[0], rtol=0, atol=1e-9))
    high = np.flatnonzero(np.isclose(edges, fit_edges[1], rtol=0, atol=1e-9))
    if len(low) != 1 or len(high) != 1:
        raise ValueError("RascalC input does not cover the fitted window")
    arguments.update(
        nthread=int(nthread),
        n_loops=int(n_loops),
        loops_per_sample=int(controls["frozen_integration"]["loops_per_sample"]),
        seed=int(controls["frozen_integration"]["seed"]),
        out_dir=str(raw_root),
        tmp_dir=str(root / "temporary"),
        randoms_positions1=positions,
        randoms_weights1=weights,
        pycorr_allcounts_11=estimator.normalize(wnorm="D1D2"),
        xi_table_11=table,
        no_data_galaxies1=data_count,
        shot_noise_rescaling1=float(alpha),
        skip_s_bins=(int(low[0]), int(len(edges) - 1 - high[0])),
        xi_11_samples=None,
        xi_sample_cov=None,
    )
    optional_failure = None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            run_cov(**arguments)
        except (ValueError, np.linalg.LinAlgError) as exc:
            # The pinned library emits this only after checking the compiled process exit code.
            # Missing/malformed archives below still fail; this is not a blanket success catch.
            logs = [raw_root / "log.txt"]
            frames = traceback.extract_tb(exc.__traceback__)
            postprocess = any(
                "RascalC/post_process/" in frame.filename.replace("\\", "/")
                or frame.filename.endswith("convergence_check_extra.py")
                for frame in frames
            )
            if not postprocess or not any(
                path.is_file()
                and "The C++ code finished succesfully" in path.read_text(errors="replace")
                for path in logs
            ):
                raise
            optional_failure = str(exc)
    archive = raw_root / f"Raw_Covariance_Matrices_n{len(edges) - 1}_l{arguments['max_l']}.npz"
    dimension = (len(edges) - 1) * (int(arguments["max_l"]) // 2 + 1)
    components, samples = read_raw_components(
        archive, dimension=dimension, subsamples=n_loops // arguments["loops_per_sample"]
    )
    return {
        "components": components,
        "subsamples": samples,
        "archive": str(archive),
        "raw_integration_completed": True,
        "postprocessing_failure": optional_failure,
        "warnings": [str(item.message) for item in caught],
        "n_loops": n_loops,
        "seed": arguments["seed"],
        "nthread": nthread,
        "shot_noise_rescaling": alpha,
        "separation_edges_mpc_h": edges,
        "multipoles": list(range(0, arguments["max_l"] + 1, 2)),
        "ordering": "radial_major",
    }


def legendre_ordering_permutation(*, radial_bin_count: int, multipole_count: int) -> np.ndarray:
    """Indices carrying a RascalC radial-major matrix into likelihood multipole-major order.

    Returns ``perm`` with ``perm[fit_index] = rascalc_index``, so the published covariance is
    ``C[np.ix_(perm, perm)]`` -- the permutation applied to **both** axes, which is what makes it
    a similarity transform and leaves every eigenvalue, the condition number, the Cholesky status
    and R_inv unchanged. Only the association between an element and its ``(multipole,
    separation)`` pair moves, which is precisely what was wrong.

    With one multipole the two conventions coincide and this is the identity, which is why the
    defect was invisible on BGS1 and would have corrupted only LRG3.
    """
    radial = int(radial_bin_count)
    multipoles = int(multipole_count)
    if radial < 1 or multipoles < 1:
        raise ValueError("covariance ordering needs at least one radial bin and one multipole")
    permutation = np.array(
        [multipoles * r + l for l in range(multipoles) for r in range(radial)], dtype=np.int64
    )
    # A permutation that is not a bijection would silently drop or duplicate elements, which is
    # the one failure this function must not have.
    if permutation.size != radial * multipoles or set(permutation.tolist()) != set(
        range(radial * multipoles)
    ):
        raise ValueError("derived covariance ordering permutation is not a bijection")
    return permutation


def raw_component_projection(
    *,
    raw_separation_edges_mpc_h: Sequence[float],
    raw_multipoles: Sequence[int],
    fit_separation_edges_mpc_h: Sequence[float],
    fit_multipoles: Sequence[int],
) -> np.ndarray:
    """The operator ``A`` that carries a raw RascalC component onto the fitted vector.

    RascalC integrates its two-, three- and four-point components on the whole rebinned radial
    grid and every multipole up to ``max_l``, in its own radial-major Legendre order: the element
    at ``multipole_count * r + l`` is multipole ``l`` at radial bin ``r``. The fitted vector is a
    contiguous window of that grid, a leading subset of those multipoles, in multipole-major
    order. ``A`` is therefore a selection-and-permutation matrix with one 1 per row, and

        C_k^fit = A C_k^raw A^T

    is an exact restriction rather than an approximation. Applying it to the *components* and
    rescaling afterwards is what lets a raw integration serve a fit vector it was not
    post-processed for, and it is why a raw product covering the required interval need not have
    its Monte-Carlo integration repeated.

    **Components, never a precision matrix.** The inverse of a restricted covariance is not the
    restriction of the inverse -- dropping rows and columns of ``C^-1`` conditions on the dropped
    bins instead of marginalizing over them -- so a rebinned precision matrix is a different
    statistical object wearing the same shape. The transform is defined on ``C_2``, ``C_3`` and
    ``C_4`` and the ``alpha_SN`` combination is formed afterwards.
    """
    raw_edges = np.asarray(raw_separation_edges_mpc_h, dtype=float)
    fit_edges = np.asarray(fit_separation_edges_mpc_h, dtype=float)
    raw_poles = [int(value) for value in raw_multipoles]
    fit_poles = [int(value) for value in fit_multipoles]
    if (
        raw_edges.ndim != 1
        or raw_edges.size < 2
        or not np.all(np.isfinite(raw_edges))
        or np.any(np.diff(raw_edges) <= 0)
    ):
        raise ValueError("raw separation edges must be increasing")
    if (
        fit_edges.ndim != 1
        or fit_edges.size < 2
        or not np.all(np.isfinite(fit_edges))
        or np.any(np.diff(fit_edges) <= 0)
    ):
        raise ValueError("fitted separation edges must be increasing")
    if raw_poles != sorted(raw_poles) or len(raw_poles) != len(set(raw_poles)):
        raise ValueError("raw multipoles must be sorted and unique")
    if fit_poles != raw_poles[: len(fit_poles)]:
        raise ValueError(
            f"fitted multipoles {fit_poles} are not a leading subset of the raw multipoles "
            f"{raw_poles}, so no restriction of the raw components reproduces them"
        )
    # Locate the fitted window on the raw grid. Every fitted edge has to *be* a raw edge: a
    # window that merely overlaps would need the counts re-summed, which is a different
    # operation from selecting bins and is not what this returns.
    matches = [
        np.flatnonzero(np.isclose(raw_edges, value, rtol=0.0, atol=1e-9)) for value in fit_edges
    ]
    if any(found.size != 1 for found in matches):
        raise ValueError("the fitted separation edges do not fall on the raw integration grid")
    indices = [int(found[0]) for found in matches]
    if indices != list(range(indices[0], indices[0] + len(indices))):
        raise ValueError("the fitted separation window is not contiguous on the raw grid")
    first_bin = indices[0]
    bin_count = len(indices) - 1
    raw_bin_count = raw_edges.size - 1
    raw_pole_count = len(raw_poles)
    rows = bin_count * len(fit_poles)
    columns = raw_bin_count * raw_pole_count
    operator = np.zeros((rows, columns), dtype="<f8")
    for pole_index in range(len(fit_poles)):
        for offset in range(bin_count):
            # Row: multipole-major, as the likelihood vector is built.
            row = pole_index * bin_count + offset
            # Column: RascalC radial-major Legendre, as the raw component is stored.
            column = raw_pole_count * (first_bin + offset) + pole_index
            operator[row, column] = 1.0
    return operator


def project_raw_components(
    components: Mapping[str, Sequence[Sequence[float]]],
    operator: np.ndarray,
) -> "dict[str, np.ndarray]":
    """Apply ``A C_k A^T`` to each of ``c2``, ``c3`` and ``c4``.

    The three are transformed independently and combined afterwards, because ``alpha_SN`` enters
    the combination with a different power on each and a matrix already combined at one value of
    it cannot be recombined at another.

    Each projected component is symmetrized, which is what RascalC's own ``load_matrices_single``
    does and in the same order: it applies its ``cov_filter`` and *then* symmetrizes. The raw
    three-point term is genuinely asymmetric as stored -- it accumulates one index order and the
    swapped one into different entries -- so requiring symmetry of the input would reject the
    library's own output. For a selection operator the two orders agree exactly, since
    ``A (M + M^T) A^T / 2`` is ``(A M A^T + (A M A^T)^T) / 2``.
    """
    matrix = np.asarray(operator, dtype="<f8")
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < matrix.shape[0]:
        raise ValueError("the projection operator must restrict onto a smaller vector")
    expected = {"c2", "c3", "c4"}
    if set(components) != expected:
        raise ValueError(f"raw component set {sorted(components)} is not {sorted(expected)}")
    projected = {}
    if not np.all(np.isfinite(matrix)):
        raise ValueError("the covariance projection operator must be finite")
    support = np.flatnonzero(np.any(matrix != 0, axis=0))
    active_operator = matrix[:, support]
    for name in sorted(expected):
        raw = np.ascontiguousarray(np.asarray(components[name], dtype="<f8"))
        if raw.shape != (matrix.shape[1], matrix.shape[1]):
            raise ValueError(f"raw component {name} has incompatible vector axes")
        active = raw[np.ix_(support, support)]
        if np.any(~np.isfinite(active)):
            raise NonFiniteCovariance(
                f"raw component {name} is not finite in the required vector subspace"
            )
        restricted = active_operator @ active @ active_operator.T
        projected[name] = np.ascontiguousarray(0.5 * (restricted + restricted.T), dtype="<f8")
    return projected


def combine_components(
    components: Mapping[str, Sequence[Sequence[float]]], shot_noise_rescaling: float
) -> np.ndarray:
    """Form ``C(alpha_SN) = C4 + alpha_SN * C3 + alpha_SN**2 * C2``.

    This is RascalC's own decomposition of the semi-analytic covariance by the power of the
    number density each term carries, so ``alpha_SN`` -- which rescales the sample's shot noise
    relative to the Poisson model -- enters linearly in the three-point term and quadratically in
    the two-point one. It is the whole role of the scalar.
    """
    alpha = float(shot_noise_rescaling)
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha_SN must be finite and positive")
    expected = {"c2", "c3", "c4"}
    if set(components) != expected:
        raise ValueError(f"component set {sorted(components)} is not {sorted(expected)}")
    parts = {}
    for name in sorted(expected):
        value = np.ascontiguousarray(np.asarray(components[name], dtype="<f8"))
        if value.ndim != 2 or value.shape[0] != value.shape[1] or value.shape[0] == 0:
            raise ValueError(f"component {name} has incompatible square vector axes")
        if np.any(~np.isfinite(value)):
            raise NonFiniteCovariance(f"component {name} is not a finite square matrix")
        parts[name] = value
    if len({value.shape for value in parts.values()}) != 1:
        raise ValueError("the three components describe vectors of different dimension")
    combined = parts["c4"] + alpha * parts["c3"] + alpha**2 * parts["c2"]
    if not np.all(np.isfinite(combined)):
        raise NonFiniteCovariance("combined fitted covariance is not finite")
    return np.ascontiguousarray(0.5 * (combined + combined.T), dtype="<f8")


#: The diagnostics field carrying the structural verdict on the vector a covariance actually
#: describes. RascalC's own test is taken on whatever block it integrated, and a route integrates
#: the untrimmed one deliberately -- every separation from zero to two hundred and every multipole
#: up to four -- while the likelihood consumes twenty-six or fifty-two of those elements. Naming
#: the two separately is what keeps an ell=4 block, or a pair-count-starved bin below the fitted
#: window, from vetoing a covariance no fit would ever ask it about.
CONSUMED_VECTOR_STRUCTURAL_TEST = "rascalc_structural_test_passes"

#: The same test taken on the whole block the integration produced, recorded beside the decisive
#: one as evidence and deciding nothing.
RAW_BLOCK_STRUCTURAL_TEST = "raw_block_structural_test_passes"


def projected_structural_verdict(
    components: Mapping[str, Sequence[Sequence[float]]], *, shot_noise_rescaling: float
) -> dict:
    """Required weaker test and descriptive stronger test on the consumed vector.

    ``check_eigval_convergence`` asks two questions of a semi-analytic covariance's terms, and
    ``check_positive_definiteness`` asks a third of the matrix they combine into. RascalC asks all
    three of the block it integrated. This asks the same three of the projection onto the fitted
    window, at the rescaling the covariance is published at, because that -- and not the untrimmed
    block -- is the matrix a fit inverts.

    Both eigenvalue inequalities retain RascalC's expressions. The stronger inequality compares
    extrema that may describe different modes, so its failure is a warning under the configured
    rule. Only the weaker condition and positive-definite construction determine this verdict.

    A projected two-point term with a non-positive eigenvalue leaves ``C2^{-1/2}`` undefined --
    the library's own matrix square root returns complex values there -- so that condition is
    reported as a failure rather than evaluated. It then reaches the escalation ladder under a
    name the configuration already registers, because inventing a fifth failure name would put a
    condition outside the set an escalation is authorized by.
    """
    alpha = float(shot_noise_rescaling)
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha_SN must be finite and positive")
    # Formed by the one function that defines the combination, so this cannot come to disagree
    # with the published matrix about what a component set is.
    combined = combine_components(components, alpha)
    c2 = np.ascontiguousarray(np.asarray(components["c2"], dtype="<f8"))
    c4 = np.ascontiguousarray(np.asarray(components["c4"], dtype="<f8"))
    failures: List[str] = []
    warnings: List[str] = []
    minimum_c2 = float(np.min(np.linalg.eigvalsh(c2)))
    minimum_c4 = float(np.min(np.linalg.eigvalsh(c4)))
    whitened = None
    if minimum_c2 <= 0.0:
        failures.append(
            f"the projected two-point term has minimum eigenvalue {minimum_c2:.3e}, so the "
            f"whitening the weaker eigenvalue test is taken in is undefined"
        )
    else:
        # ``min eigenvalue of C2^{-1/2} C4 C2^{-1/2}``, formed from the symmetric eigen-
        # decomposition of C2 rather than a general matrix square root: for a symmetric positive
        # definite C2 the principal square root is that decomposition, so the two agree exactly.
        values, vectors = np.linalg.eigh(c2)
        inverse_sqrt = (vectors / np.sqrt(values)) @ vectors.T
        whitened = float(np.min(np.linalg.eigvalsh(inverse_sqrt @ c4 @ inverse_sqrt)))
        # RascalC's weaker test: ``min_eig_comb <= -alpha**2``.
        if whitened <= -(alpha**2):
            failures.append(
                f"the weaker eigenvalue test fails on the consumed vector at alpha_SN "
                f"{alpha:.2f}: the whitened four-point minimum is {whitened:.2f} and must exceed "
                f"{-(alpha ** 2):.2f}"
            )
    # RascalC's stronger test: ``min_eig_c4 < -min_eig_c2 * alpha**2``, taken whatever the sign of
    # the two-point minimum, exactly as the library takes it.
    stronger_passes = bool(minimum_c4 >= -minimum_c2 * alpha**2)
    if not stronger_passes:
        warnings.append(
            f"the stronger eigenvalue test fails on the consumed vector at alpha_SN {alpha:.2f}: "
            f"the four-point minimum is {minimum_c4:.3e} and the two-point one {minimum_c2:.3e}"
        )
    try:
        np.linalg.cholesky(combined)
    except np.linalg.LinAlgError:
        failures.append(
            f"the combined covariance at alpha_SN {alpha:.2f} is not positive definite on the "
            f"consumed vector"
        )
    return {
        "structural_test_passes": not failures,
        "structural_test_failures": failures,
        "structural_test_warnings": warnings,
        "acceptance_policy": "weaker_eigenvalue_and_positive_definiteness",
        "stronger_eigenvalue_test_passes": stronger_passes,
        "stronger_eigenvalue_margin": minimum_c4 + minimum_c2 * alpha**2,
        "weaker_eigenvalue_margin": None if whitened is None else whitened + alpha**2,
        "shot_noise_rescaling": alpha,
        "dimension": int(combined.shape[0]),
        "minimum_eigenvalue_c2": minimum_c2,
        "minimum_eigenvalue_c4": minimum_c4,
        "minimum_eigenvalue_whitened_c4": whitened,
    }


def tune_shot_noise_rescaling(
    components: Mapping[str, Sequence[Sequence[float]]],
    route_sample_covariance: Sequence[Sequence[float]],
    *,
    search_limits: Tuple[float, float],
    points: int,
) -> dict:
    """Tune the single scalar ``alpha_SN`` against a route's own realizations.

    The objective is the cross-entropy between the model covariance and the empirical one,

        Q(alpha_SN) = logdet C(alpha_SN) + trace( C(alpha_SN)^-1 S_route )

    minimized over one scalar. ``S_route`` is the sample covariance of the route's twenty-five
    correlation-function realizations.

    **``S_route`` may be singular, and that is not a problem here.** With twenty-five
    realizations its rank is at most twenty-four against ``p = 26`` or ``52``, so it has no
    inverse and no usable determinant -- which is exactly why it may never be the likelihood's
    covariance, and why no Hartlap or Sellentin-Heavens correction repairs it (the Hartlap factor
    ``(N - p - 2)/(N - 1)`` is negative for both dimensions). But this objective never inverts it
    and never takes its determinant: it appears only inside a trace against the *model*
    precision, which is well conditioned. One scalar is being estimated from twenty-five
    realizations, not hundreds of covariance elements.

    **The search interval and the grid size are arguments with no defaults.** They are numerical
    controls on a measurement: a narrower interval that excludes the optimum returns its own
    endpoint, and a coarse grid returns the nearest grid point rather than the minimum. Carrying
    either as a function default made them look like implementation detail while they set what
    the estimator can return, so a caller that has not been given configured values cannot reach
    this function at all.

    A solution on either endpoint is reported for review: it may be constrained by the interval.
    Save every objective value, using null for undefined candidates. The caller publishes this
    fresh calibration once; resuming a complete route reads its saved result.
    """
    sample = np.ascontiguousarray(np.asarray(route_sample_covariance, dtype="<f8"))
    if sample.ndim != 2 or sample.shape[0] != sample.shape[1] or np.any(~np.isfinite(sample)):
        raise ValueError("the route sample covariance is not a finite square matrix")
    if not np.allclose(sample, sample.T, rtol=0.0, atol=_covariance_symmetry_atol(sample)):
        raise ValueError("the route sample covariance is not symmetric")
    lower, upper = (float(value) for value in search_limits)
    if (
        not np.isfinite([lower, upper]).all()
        or not 0 < lower < upper
        or type(points) is not int
        or points < 2
    ):
        raise ValueError("the alpha_SN search interval is invalid")
    # The model and the empirical matrix must describe the same vector. Raw RascalC components
    # are 150-dimensional and the route sample covariance is 26 or 52; combining them here would
    # be tuning a scalar of one quantity against the scatter of another, so the components are
    # required to have been projected onto the fitted vector before this is called.
    model_dimension = int(np.asarray(components["c4"], dtype="<f8").shape[0])
    if model_dimension != sample.shape[0]:
        raise ValueError(
            f"the components describe a {model_dimension}-element vector and the route sample "
            f"covariance a {sample.shape[0]}-element one; project the components onto the fitted "
            f"vector before tuning"
        )
    grid = np.linspace(lower, upper, int(points))
    best_alpha, best_objective, evaluated = None, np.inf, 0
    objectives = [None] * points
    for index, alpha in enumerate(grid):
        model = combine_components(components, float(alpha))
        try:
            cholesky = np.linalg.cholesky(model)
        except np.linalg.LinAlgError:
            # Not every alpha yields a positive-definite combination; those are simply not
            # candidates. Skipping them is not selection on an outcome -- the objective is
            # undefined there.
            continue
        evaluated += 1
        log_determinant = 2.0 * float(np.sum(np.log(np.diag(cholesky))))
        solved = np.linalg.solve(model, sample)
        objective = log_determinant + float(np.trace(solved))
        if np.isfinite(objective):
            objectives[index] = objective
        if np.isfinite(objective) and objective < best_objective:
            best_alpha, best_objective = float(alpha), objective
    if best_alpha is None:
        if evaluated == 0:
            raise NoPositiveDefiniteCandidate(
                "no alpha_SN in the search interval gives a positive-definite covariance"
            )
        raise ValueError("no alpha_SN in the search interval gives a finite objective")
    # An optimum on an endpoint is an optimum of the interval, not of the objective.
    on_boundary = bool(
        np.isclose(best_alpha, lower, rtol=0.0, atol=0.5 * float(grid[1] - grid[0]))
        or np.isclose(best_alpha, upper, rtol=0.0, atol=0.5 * float(grid[1] - grid[0]))
    )
    return {
        "source": "route_mock_scatter",
        "shot_noise_rescaling": best_alpha,
        "objective": best_objective,
        "objective_definition": (
            "Q(alpha_SN) = logdet C(alpha_SN) + trace(inv(C(alpha_SN)) @ S_route)"
        ),
        "model_definition": "C(alpha_SN) = C4 + alpha_SN * C3 + alpha_SN**2 * C2",
        "sample_covariance_rank": int(np.linalg.matrix_rank(sample)),
        "sample_covariance_dimension": int(sample.shape[0]),
        "model_dimension": int(np.asarray(components["c4"]).shape[0]),
        "search_limits": [lower, upper],
        "search_points": int(points),
        "objective_curve": objectives,
        "solution_on_search_boundary": on_boundary,
        "evaluated_points": int(evaluated),
    }


def rascalc_native_table(estimator: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(s_values, mu_values, xi_values, s_edges)`` exactly as RascalC would build them.

    RascalC's own estimator branch does three things before it writes its input table: it repairs
    bins whose weighted counts came out negative, it wraps a ``mu``-symmetric estimator onto
    non-negative ``mu``, and it divides the correlation function by the ratio of normalized
    ``RR`` to ``SS`` counts -- which matters post-reconstruction, where the shifted-random
    estimator's denominators differ. Its library functions are called here rather than
    reimplemented, so the route mean is by construction the same object RascalC would have
    derived from a single estimator, differing only in that its values are the route's mean.

    The returned ``s_edges`` is what makes this a bin-averaged table rather than a sampled model.
    """
    try:
        from RascalC.pycorr_utils.input_xi import get_input_xi_from_pycorr
        from RascalC.pycorr_utils.utils import fix_bad_bins_pycorr
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "RascalC is required to build its own correlation-function table"
        ) from exc

    wrapped = estimator
    if float(np.asarray(estimator.edges[1])[0]) < 0:
        wrapped = fix_bad_bins_pycorr(estimator).wrap()
    separations = np.ascontiguousarray(np.asarray(wrapped.sepavg(axis=0), dtype="<f8"))
    mu_values = np.ascontiguousarray(np.asarray(wrapped.sepavg(axis=1), dtype="<f8"))
    table = np.ascontiguousarray(np.asarray(get_input_xi_from_pycorr(wrapped), dtype="<f8"))
    edges = np.ascontiguousarray(np.asarray(wrapped.edges[0], dtype="<f8"))
    if table.shape != (separations.size, mu_values.size):
        raise ValueError("the RascalC correlation-function table does not match its own grid")
    if edges.size != separations.size + 1 or np.any(np.diff(edges) <= 0):
        raise ValueError("the RascalC separation edges do not bound its own separation values")
    return separations, mu_values, table, edges


def require_refinable_xi_table(table) -> tuple:
    """Refuse any correlation-function table RascalC would treat as a sampled model.

    RascalC accepts a table as three elements ``(s, mu, xi)`` or four ``(s, mu, xi, s_edges)``.
    The two are not variants of one argument: a three-element table declares the values to be a
    theoretical model evaluated at points, and the library then sets its refinement iterations to
    zero **regardless of what the configuration asked for**. The configuration asks for ten. So a
    three-element table produces a different covariance from the same twenty-five measurements,
    silently, with nothing in the result to say which form was passed.

    This refuses it at the interface rather than letting the difference appear as an unexplained
    change in a matrix nobody can reproduce.
    """
    if not isinstance(table, (tuple, list)):
        raise ValueError(
            "a RascalC correlation-function table is a sequence of (s, mu, xi, s_edges)"
        )
    if len(table) == 3:
        raise ValueError(
            "a three-element correlation-function table declares a theoretical model sampled at "
            "points, which disables RascalC's refinement iterations entirely and silently "
            "changes the integral. Supply the four-element form (s_values, mu_values, xi_values, "
            "s_edges): these are bin averages and their edges say so."
        )
    if len(table) != 4:
        raise ValueError(
            f"a RascalC correlation-function table has four elements, not {len(table)}"
        )
    separations, mu_values, values, edges = (np.asarray(item, dtype=float) for item in table)
    if values.shape != (separations.size, mu_values.size):
        raise ValueError("the correlation-function table does not match its own grid")
    if edges.size != separations.size + 1 or np.any(np.diff(edges) <= 0):
        raise ValueError("the correlation-function table's separation edges do not bound its grid")
    return separations, mu_values, values, edges


def route_vectors(run, route, records=None):
    """Read the same fitted vectors, ordering and edge selection used by every likelihood."""
    from .clustering import read_measurements
    from .likelihood import xi_fit_data

    config = run["config"]
    records = read_measurements(run, route) if records is None else records
    combined = [record for record in records if record["identity"]["cap"] == "NGC+SGC"]
    identifiers = config["analysis"]["workloads"][run["metadata"]["workload"]]["mock_ids"]
    ordered = complete_realizations(
        [{**record, "realization_id": record["identity"]["realization_id"]} for record in combined],
        identifiers,
    )
    poles = config["fitting"]["fit_surfaces"][route["tracer"]]["observed_multipoles"]
    factor = (
        config["fitting"]["input_bin_width_mpc_h"] / config["measurement"]["xi"]["bin_width_mpc_h"]
    )
    if not float(factor).is_integer():
        raise ValueError("fitted bin width does not exactly coarsen native counts")
    arrays = [
        xi_fit_data(
            record["estimator"],
            multipoles=poles,
            fit_edge_limits_mpc_h=config["fitting"]["fit_edge_limits_mpc_h"],
            rebin_factor=int(factor),
            expected_bin_width_mpc_h=config["fitting"]["input_bin_width_mpc_h"],
        )
        for record in ordered
    ]
    if any(not np.array_equal(item[2], arrays[0][2]) for item in arrays):
        raise ValueError("realizations do not share identical fitted radial edges")
    vectors = np.asarray([item[1] for item in arrays])
    separation = np.asarray([item[0] for item in arrays])
    if not np.all(np.isfinite(vectors)):
        raise ValueError("required ensemble vector contains non-finite values")
    return {
        "vectors": vectors,
        "separations": separation,
        "fit_edges": arrays[0][2],
        "multipoles": np.asarray(poles),
        "realization_ids": np.asarray(identifiers),
        "records": ordered,
        "rebin_factor": int(factor),
    }


def read_route_covariance(run, route):
    """Read one complete fitted route with direct coordinate, matrix and scaling checks."""
    import yaml

    coordinates = {
        key: route[key]
        for key in ("tracer", "measurement_bin", "condition", "reconstruction_state")
    }
    name = "--".join(
        coordinates[key] for key in ("measurement_bin", "condition", "reconstruction_state")
    )
    directory = run["root"] / "covariance" / "routes" / name
    result = yaml.safe_load((directory / "result.yaml").read_text())
    record = json.loads((directory / "covariance.json").read_text())
    for metadata in (result, record):
        if not same_identity(metadata["route"], coordinates) or any(
            metadata.get(key) != run["metadata"][key]
            for key in ("run_name", "code_commit", "analysis_config_sha256")
        ):
            raise ValueError("covariance belongs to another route or snapshot")
    if run["metadata"].get("inference_definition") is not None:
        source = _component_source_run(run)
        if route["tracer"] != "BGS_BRIGHT-21.5":
            raise ValueError("BGS covariance reprojection cannot publish an LRG route")
        for metadata in (result, record):
            if not same_identity(metadata.get("source_run"), run["metadata"]["source_run"]):
                raise ValueError("covariance component-source identity differs")
        for cap, relative in record["cap_integrations"].items():
            path = (source["root"] / relative).resolve()
            if not path.is_relative_to(source["root"]):
                raise ValueError("covariance component path leaves its measurement source")
            cap_record = json.loads((path / "cap.json").read_text())
            if (
                not same_identity(cap_record.get("route"), coordinates)
                or cap_record.get("cap") != cap
                or any(
                    cap_record.get(key) != source["metadata"][key]
                    for key in ("run_name", "code_commit", "analysis_config_sha256")
                )
            ):
                raise ValueError("reused cap component provenance differs")
    if record["units"] != "Mpc/h" or record["ordering"] != "multipole_major":
        raise ValueError("covariance units or vector ordering are incompatible")
    config = run["config"]
    expected_ids = config["analysis"]["workloads"][run["metadata"]["workload"]]["mock_ids"]
    if not same_identity(record["realization_ids"], expected_ids) or set(
        record["cap_integrations"]
    ) != {"NGC", "SGC"}:
        raise ValueError("covariance does not cover both caps and the complete ensemble")
    alpha = record["shot_noise"]["shot_noise_rescaling"]
    tuning = fresh_shot_noise_controls(config)
    if type(alpha) not in (int, float) or not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("covariance shot-noise scalar is not finite and positive")
    scalar = record["shot_noise"]
    if not same_identity(scalar.get("realization_ids"), expected_ids):
        raise ValueError("scalar calibration does not cover the complete route ensemble")
    if scalar.get("source") != config["fitting"]["covariance"]["shot_noise_rescaling"][
        "source"
    ] or any(
        not same_identity(scalar.get(key), tuning[key])
        for key in ("search_limits", "search_points")
    ):
        raise ValueError("covariance scalar source or search differs from the run configuration")
    grid = np.linspace(*tuning["search_limits"], tuning["search_points"])
    curve = np.asarray(scalar.get("objective_curve"), dtype=float)
    if curve.shape != grid.shape or not np.isfinite(curve).any() or np.isinf(curve).any():
        raise ValueError("covariance scalar lacks its finite calibration objective curve")
    best = int(np.nanargmin(curve))
    if (
        alpha != grid[best]
        or scalar.get("objective") != curve[best]
        or not same_identity(scalar.get("solution_on_search_boundary"), best in (0, len(grid) - 1))
    ):
        raise ValueError("covariance scalar differs from its saved calibration minimum")
    with np.load(directory / "covariance.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    with np.load(directory / "window.npz", allow_pickle=False) as archive:
        window = {key: archive[key].copy() for key in archive.files}
    poles = config["fitting"]["fit_surfaces"][coordinates["tracer"]]["observed_multipoles"]
    lo, hi = config["fitting"]["fit_edge_limits_mpc_h"]
    width = config["fitting"]["input_bin_width_mpc_h"]
    edges = np.arange(lo, hi + 0.5 * width, width)
    radial, count = len(edges) - 1, len(expected_ids)
    dimension = radial * len(poles)
    xi = config["measurement"]["xi"]
    native_edges = np.arange(
        xi["s_min_mpc_h"], xi["s_max_mpc_h"] + 0.5 * xi["bin_width_mpc_h"], xi["bin_width_mpc_h"]
    )
    mu_edges = np.linspace(-1, 1, xi["mu_bins"] + 1)
    if (
        set(window) != {"sedges", "muedges", "wcounts"}
        or not np.array_equal(window["sedges"], native_edges)
        or not np.array_equal(window["muedges"], mu_edges)
        or window["wcounts"].shape != (len(native_edges) - 1, len(mu_edges) - 1)
        or not np.all(np.isfinite(window["wcounts"]))
        or np.any(window["wcounts"] < 0)
    ):
        raise ValueError("covariance window has incompatible native axes or weights")
    native_centres = (native_edges[1:] + native_edges[:-1]) / 2
    for low, high in zip(edges[:-1], edges[1:]):
        if window["wcounts"][(native_centres >= low) & (native_centres < high)].sum() <= 0:
            raise ValueError("covariance window has no random pair support in a fitted bin")
    if (
        arrays["realization_ids"].dtype.kind not in "iu"
        or not np.array_equal(arrays["realization_ids"], expected_ids)
        or not np.array_equal(arrays["fit_edges"], edges)
        or not np.array_equal(arrays["multipoles"], poles)
    ):
        raise ValueError(
            "covariance fitted axes or realization identifiers differ from configuration"
        )
    for key, shape in (
        ("vectors", (count, dimension)),
        ("separations", (count, radial)),
        ("covariance", (dimension, dimension)),
        ("mean_covariance", (dimension, dimension)),
    ):
        if arrays[key].shape != shape or not np.all(np.isfinite(arrays[key])):
            raise ValueError(f"covariance product {key} has incompatible axes or values")
    if np.any(arrays["separations"] < edges[:-1]) or np.any(arrays["separations"] > edges[1:]):
        raise ValueError("fitted separation falls outside its radial bin")
    matrix = combine_components({key: arrays[key] for key in ("c2", "c3", "c4")}, alpha)
    if not np.array_equal(arrays["covariance"], matrix) or not np.array_equal(
        arrays["mean_covariance"], matrix / count
    ):
        raise ValueError(
            "covariance components or mean scaling differ from the saved likelihood matrix"
        )
    np.linalg.cholesky(matrix)
    return {"directory": directory, "metadata": record, "arrays": arrays, "window": window}


def covariance_cap_stage(run, unit, destination, *, nthread, n_loops=None):
    """Save one complete cap integration. Final fitted validity is assessed after cap assembly."""
    from .clustering import read_measurements

    config = run["config"]
    records = read_measurements(run, unit)
    ids = config["analysis"]["workloads"][run["metadata"]["workload"]]["mock_ids"]
    cap = unit["cap"]
    ordered = complete_realizations(
        [
            {**record, "realization_id": record["identity"]["realization_id"]}
            for record in records
            if record["identity"]["cap"] == cap
        ],
        ids,
    )
    geometry = resolve_route_geometry(config, realization_ids=ids)
    representative = next(
        record for record in ordered if record["realization_id"] == geometry["realization_id"]
    )
    factor = (
        config["fitting"]["input_bin_width_mpc_h"] / config["measurement"]["xi"]["bin_width_mpc_h"]
    )
    if not float(factor).is_integer():
        raise ValueError("covariance grid does not exactly coarsen measurement counts")
    tables = [rascalc_native_table(record["estimator"][:: int(factor)]) for record in ordered]
    if any(
        not np.array_equal(table[3], tables[0][3]) or not np.array_equal(table[1], tables[0][1])
        for table in tables
    ):
        raise ValueError("cap ensemble native xi tables have incompatible grids")
    mean_table = (
        np.mean([table[0] for table in tables], axis=0),
        tables[0][1],
        np.mean([table[2] for table in tables], axis=0),
        tables[0][3],
    )
    require_refinable_xi_table(mean_table)
    tuning = fresh_shot_noise_controls(config)
    # Provisional native post-processing only. Complete raw terms survive a post-processing
    # failure; the final scalar is calibrated after fitted-vector projection and cap assembly.
    provisional_alpha = float(tuning["search_limits"][0])
    from .clustering import read_measurement_randoms

    coordinate_key = (
        "positions" if unit["reconstruction_state"] == "prereconstruction" else "shifted_positions"
    )
    random_arrays = read_measurement_randoms(representative, config)
    positions, weights = random_arrays[coordinate_key], random_arrays["weights"]
    source_ids = random_arrays["source_ids"]
    if len(source_ids) != representative["random_count"] or len(np.unique(source_ids)) != len(
        source_ids
    ):
        raise ValueError("covariance geometry has incomplete or duplicate random identities")
    depth = (
        config["fitting"]["covariance"]["rascalc"]["escalation_ladder_n_loops"][0]
        if n_loops is None
        else n_loops
    )
    integrated = integrate_cap(
        estimator=representative["estimator"][:: int(factor)],
        xi_table=mean_table,
        random_positions=positions,
        random_weights=weights,
        data_count=representative["data_count"],
        config=config,
        destination=destination / "integration",
        nthread=nthread,
        n_loops=depth,
        alpha=provisional_alpha,
    )
    arrays = {
        **integrated["components"],
        **{key + "_samples": value for key, value in integrated["subsamples"].items()},
    }
    np.savez(destination / "components.npz", **arrays)
    np.savez(
        destination / "mean_xi.npz",
        s=mean_table[0],
        mu=mean_table[1],
        xi=mean_table[2],
        edges=mean_table[3],
    )
    metadata = {
        key: _diagnostic_values(value)
        for key, value in integrated.items()
        if key not in {"components", "subsamples", "archive"}
    }
    metadata.update(
        geometry=geometry,
        cap=cap,
        realization_ids=ids,
        route={
            key: unit[key]
            for key in ("tracer", "measurement_bin", "condition", "reconstruction_state")
        },
        **{
            key: run["metadata"][key]
            for key in ("run_name", "code_commit", "analysis_config_sha256")
        },
        raw_archive=str(Path(integrated["archive"]).relative_to(destination)),
        scalar_source="provisional_postprocessing_only",
        random_coordinate_state=coordinate_key,
        data_count=representative["data_count"],
        random_count=len(weights),
    )
    (destination / "cap.json").write_text(json.dumps(metadata, indent=2, allow_nan=False))
    return {
        "products": [
            {"path": "cap.json", "format": "json"},
            {
                "path": "components.npz",
                "format": "rascalc_components",
                "dimension": len(integrated["components"]["c2"]),
                "subsamples": len(integrated["subsamples"]["c2"]),
            },
            {"path": "mean_xi.npz", "format": "npz", "arrays": ["s", "mu", "xi", "edges"]},
        ],
        "n_loops": depth,
        "geometry_realization_id": geometry["realization_id"],
    }


def read_cap_components(directory):
    record = json.loads((directory / "cap.json").read_text())
    if record["raw_integration_completed"] is not True or record["ordering"] != "radial_major":
        raise ValueError("cap archive does not represent a complete native integration")
    with np.load(directory / "components.npz", allow_pickle=False) as archive:
        components = {key: archive[key].copy() for key in ("c2", "c3", "c4")}
        subsamples = {key: archive[key + "_samples"].copy() for key in components}
    dimension = (len(record["separation_edges_mpc_h"]) - 1) * len(record["multipoles"])
    if any(value.shape != (dimension, dimension) for value in components.values()):
        raise ValueError("cap component axes differ from their radial and multipole grid")
    if any(
        value.ndim != 3 or value.shape[1:] != (dimension, dimension)
        for value in subsamples.values()
    ):
        raise ValueError("cap subsample component axes are invalid")
    return components, subsamples, record


def assembled_components(run, route, cap_directories, vectors, records):
    """Compose angular mixing, multipole ordering and fitted selection before raw projection."""
    source = _component_source_run(run)
    representative_id = resolve_route_geometry(
        run["config"], realization_ids=vectors["realization_ids"].tolist()
    )["realization_id"]
    representative = {
        record["identity"]["cap"]: record["estimator"][:: vectors["rebin_factor"]].wrap()
        for record in records
        if record["identity"]["realization_id"] == representative_id
        and record["identity"]["cap"] in ("NGC", "SGC")
    }
    north = representative["NGC"]
    edges = np.asarray(north.edges[0])
    fit_edges = vectors["fit_edges"]
    mask = (edges[:-1] >= fit_edges[0] - 1e-9) & (edges[1:] <= fit_edges[-1] + 1e-9)
    for value in representative.values():
        if not all(np.array_equal(left, right) for left, right in zip(value.edges, north.edges)):
            raise ValueError("representative cap grids differ")
    raw_poles = run["config"]["fitting"]["fit_surfaces"][route["tracer"]]["model_multipoles"]
    operators, weights = cap_multipole_operators(
        {
            cap: value.normalize(wnorm="D1D2").R1R2.wcounts[mask]
            for cap, value in representative.items()
        },
        north.edges[1],
        multipoles=raw_poles,
    )
    reorder = raw_component_projection(
        raw_separation_edges_mpc_h=edges,
        raw_multipoles=raw_poles,
        fit_separation_edges_mpc_h=fit_edges,
        fit_multipoles=raw_poles,
    )
    dimension = (len(fit_edges) - 1) * len(vectors["multipoles"])
    select = np.eye(len(operators["NGC"]))[:dimension]
    composed = {cap: select @ operator @ reorder for cap, operator in operators.items()}
    raw, samples, cap_records = {}, {}, {}
    for cap, directory in cap_directories.items():
        raw[cap], samples[cap], cap_records[cap] = read_cap_components(directory)
        identity = {
            key: route[key]
            for key in ("tracer", "measurement_bin", "condition", "reconstruction_state")
        }
        if (
            not same_identity(cap_records[cap]["route"], identity)
            or any(
                cap_records[cap].get(key) != source["metadata"][key]
                for key in ("run_name", "code_commit", "analysis_config_sha256")
            )
            or not same_identity(
                cap_records[cap]["realization_ids"], vectors["realization_ids"].tolist()
            )
            or cap_records[cap]["cap"] != cap
            or not same_identity(cap_records[cap]["geometry"]["realization_id"], representative_id)
            or not np.array_equal(cap_records[cap]["separation_edges_mpc_h"], edges)
            or cap_records[cap]["multipoles"] != raw_poles
        ):
            raise ValueError("cap integration and weighting geometry disagree")
    if cap_records["NGC"]["n_loops"] != cap_records["SGC"]["n_loops"]:
        raise ValueError("cap integration depths differ")
    return assemble_cap_components(raw, composed), samples, composed, weights, cap_records


def assemble_route(run, route, cap_directories, destination, *, integrate_next):
    """Assess the required covariance; request deeper caps only after numerical failure."""
    from .clustering import read_measurements

    source = _component_source_run(run)
    reuse = source is not run
    if reuse and route["tracer"] != "BGS_BRIGHT-21.5":
        raise ValueError("only BGS routes may reuse components under the changed definition")
    records = read_measurements(source, route)
    vectors = route_vectors(run, route, records)
    config = run["config"]
    controls = fresh_shot_noise_controls(config)
    first_depth = config["fitting"]["covariance"]["rascalc"]["escalation_ladder_n_loops"][0]
    samples, operators, weights, cap_records = {}, {}, {}, {}
    scalar = None
    sample_covariance = np.cov(vectors["vectors"], rowvar=False, ddof=1)

    def attempt(depth, retry):
        nonlocal cap_directories, samples, operators, weights, cap_records
        if depth != first_depth:
            cap_directories = {cap: integrate_next(cap, depth) for cap in ("NGC", "SGC")}
        components, samples, operators, weights, cap_records = assembled_components(
            run, route, cap_directories, vectors, records
        )
        return components

    def select_alpha(components):
        nonlocal scalar
        scalar = tune_shot_noise_rescaling(
            components,
            sample_covariance,
            search_limits=controls["search_limits"],
            points=controls["search_points"],
        )
        scalar["realization_ids"] = vectors["realization_ids"].tolist()
        return scalar["shot_noise_rescaling"]

    # Worker publication already bounds infrastructure retries; this owner alone selects depth.
    if reuse:
        combined, samples, operators, weights, cap_records = assembled_components(
            run, route, cap_directories, vectors, records
        )
        try:
            failure = numerical_failure(combined, select_alpha(combined), config)
        except NoPositiveDefiniteCandidate:
            failure = "cholesky_failure"
        if failure is not None:
            raise ValueError(
                f"saved BGS components fail joint-vector validation ({failure}); "
                "no deeper integration was launched"
            )
        history = [{"n_loops": cap_records["NGC"]["n_loops"], "numerical_failure": None}]
    else:
        combined, history = numerical_ladder(
            attempt, config=config, alpha=select_alpha, infrastructure_retries=0
        )
    alpha = scalar["shot_noise_rescaling"]
    matrix = combine_components(combined, alpha)
    # Optional numerical diagnostics cannot withhold an otherwise usable fitted matrix.
    try:
        combined_samples = combine_cap_subsamples(samples, operators, alpha=alpha)
        diagnostics = {"combined": covariance_diagnostics(matrix, combined_samples), "caps": {}}
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        combined_samples = np.empty((0, *matrix.shape))
        diagnostics = {
            "combined": covariance_diagnostics(matrix),
            "caps": {},
            "subsample_diagnostic_unavailable": str(exc),
        }
    cap_arrays = {}
    diagnostics["combined"]["structural"] = projected_structural_verdict(
        combined, shot_noise_rescaling=alpha
    )
    for cap in ("NGC", "SGC"):
        try:
            raw, _, _ = read_cap_components(cap_directories[cap])
            projected = project_raw_components(raw, operators[cap])
            cap_matrix = combine_components(projected, alpha)
            cap_samples = np.asarray(
                [
                    combine_components(
                        project_raw_components(
                            {key: value[i] for key, value in samples[cap].items()}, operators[cap]
                        ),
                        alpha,
                    )
                    for i in range(len(samples[cap]["c2"]))
                ]
            )
            cap_arrays[f"covariance_{cap}"] = cap_matrix
            cap_arrays[f"covariance_subsamples_{cap}"] = cap_samples
            diagnostics["caps"][cap] = covariance_diagnostics(cap_matrix, cap_samples)
            diagnostics["caps"][cap]["structural"] = projected_structural_verdict(
                projected, shot_noise_rescaling=alpha
            )
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            diagnostics["caps"][cap] = {"value": None, "reason": str(exc)}
    identifiers = vectors["realization_ids"]
    np.savez(
        destination / "covariance.npz",
        covariance=matrix,
        mean_covariance=matrix / len(identifiers),
        vectors=vectors["vectors"],
        separations=vectors["separations"],
        fit_edges=vectors["fit_edges"],
        multipoles=vectors["multipoles"],
        realization_ids=identifiers,
        **combined,
        covariance_subsamples=combined_samples,
        **cap_arrays,
        **{f"W_{cap}": value for cap, value in operators.items()},
        **{f"weights_{cap}": value for cap, value in weights.items()},
    )
    geometry_id = cap_records["NGC"]["geometry"]["realization_id"]
    representative = next(
        record
        for record in records
        if record["identity"]["cap"] == "NGC+SGC"
        and record["identity"]["realization_id"] == geometry_id
    )["estimator"]
    rr = representative.R1R2
    np.savez(
        destination / "window.npz", sedges=rr.edges[0], muedges=rr.edges[1], wcounts=rr.wcounts
    )
    metadata = {
        "route": {
            key: route[key]
            for key in ("tracer", "measurement_bin", "condition", "reconstruction_state")
        },
        "run_name": run["metadata"]["run_name"],
        "code_commit": run["metadata"]["code_commit"],
        "analysis_config_sha256": run["metadata"]["analysis_config_sha256"],
        "units": "Mpc/h",
        "ordering": "multipole_major",
        "geometry_realization_id": geometry_id,
        "realization_ids": identifiers.tolist(),
        "shot_noise": scalar,
        "numerical_history": history,
        "cap_integrations": {
            cap: str(directory.relative_to(source["root"]))
            for cap, directory in cap_directories.items()
        },
        "cross_cap_covariance": "neglected",
        "mean_covariance_scaling": "C_route/N",
        "mean_covariance_interpretation": "conditional on the shared fitted radial template; its uncertainty and cross terms are not propagated",
        "diagnostics": diagnostics,
    }
    if reuse:
        metadata["source_run"] = run["metadata"]["source_run"]
    (destination / "covariance.json").write_text(json.dumps(metadata, indent=2, allow_nan=False))
    return {
        "products": [
            {
                "path": "covariance.npz",
                "format": "npz",
                "arrays": [
                    "covariance",
                    "mean_covariance",
                    "vectors",
                    "separations",
                    "fit_edges",
                    "multipoles",
                    "realization_ids",
                ],
            },
            {"path": "window.npz", "format": "npz", "arrays": ["sedges", "muedges", "wcounts"]},
            {"path": "covariance.json", "format": "json"},
        ]
    }
