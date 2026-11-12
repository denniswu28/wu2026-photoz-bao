"""Shared observables, priors and likelihood construction for profiling and sampling."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from .catalog import (
    build_mock_cosmology,
    measurement_bin as _measurement_bin,
)
from .models.mu_quadrature import symmetric_leggauss
from .config import UNIFIED_FLAT, per_bin_reconstruction


from .theory import (
    COMMON_FLAT_DAMPING_SCOPE,
    FINITE_FLAT_DAMPING_SCOPE,
    TRANSFER_FLAT_DAMPING_SCOPE,
    UNIFIED_DAMPING_PRIOR,
    analytic_postreconstruction_sigma_perp,
    resolve_damping_prior,
    unified_damping_prior,
)


def _build_observed_photoz_power_model(**kwargs):
    """Import the pinned-desilike response model only when a fit is constructed."""
    try:
        from .models.photoz_bao import build_observed_photoz_power_model
    except ImportError:
        from models.photoz_bao import build_observed_photoz_power_model
    return build_observed_photoz_power_model(**kwargs)


def likelihood_window_inputs(
    record,
    *,
    sample_identity,
    tracer,
    measurement_bin,
    reconstruction_state,
    observed_multipoles,
    model_multipoles,
    fit_sedges,
):
    """Check the actual RR geometry and pass it directly to the native likelihood window."""
    expected = {
        "sample_identity": sample_identity,
        "tracer": tracer,
        "measurement_bin": measurement_bin,
        "reconstruction_state": reconstruction_state,
    }
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError("RR window belongs to another fitted sample")
    sedges, muedges, counts = [
        np.asarray(record[key], dtype=float) for key in ("sedges", "muedges", "wcounts")
    ]
    fit = np.asarray(fit_sedges, dtype=float)
    if fit.ndim != 1 or fit.size < 2 or not np.all(np.isfinite(fit)):
        raise ValueError("fitted radial edges must be a finite one-dimensional grid")
    if (
        sedges.ndim != 1
        or muedges.ndim != 1
        or counts.shape != (len(sedges) - 1, len(muedges) - 1)
        or np.any(np.diff(sedges) <= 0)
        or np.any(np.diff(muedges) <= 0)
        or not np.all(np.isfinite(sedges))
        or not np.all(np.isfinite(muedges))
        or not np.all(np.isfinite(counts))
        or np.any(counts < 0)
        or muedges[0] != -1
        or muedges[-1] != 1
    ):
        raise ValueError("RR window has incompatible axes, units or finite angular support")
    matches = [np.flatnonzero(np.isclose(sedges, edge, rtol=0, atol=1e-9)) for edge in fit]
    if any(len(index) != 1 for index in matches):
        raise ValueError("fitted radial edges do not exactly coarsen the saved RR window")
    indices = np.asarray([index.item() for index in matches])
    if len(set(np.diff(indices))) != 1 or np.diff(indices)[0] < 1:
        raise ValueError("RR window does not coarsen uniformly onto the fitted vector")
    rebinned = np.array(
        [counts[left:right].sum(axis=0) for left, right in zip(indices[:-1], indices[1:])]
    )
    if np.any(rebinned.sum(axis=1) <= 0):
        raise ValueError("required RR window has an empty fitted radial bin")
    if list(model_multipoles) != [0, 2, 4] or not set(observed_multipoles).issubset(
        model_multipoles
    ):
        raise ValueError("likelihood window must retain all configured raw model multipoles")
    return {
        "wmatrix": {"sedges": sedges, "muedges": muedges, "wcounts": counts},
        "ellsin": tuple(model_multipoles),
        "systematic_templates": {},
    }


def xi_fit_data(
    xi: Any,
    *,
    multipoles: Sequence[int],
    fit_edge_limits_mpc_h: Sequence[float],
    rebin_factor: int = 1,
    expected_bin_width_mpc_h: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the exact edge-selected xi vector used by covariance and likelihood.

    ``xi[::rebin_factor]`` invokes pycorr's count-rebinning slice; it is not
    array subsampling and requires the factor to divide the native radial-bin
    count, a requirement the pinned pycorr version is expected to enforce;
    pinned-library compatibility evidence establishes that behavior. Locally,
    ragged rebinning is caught by the exact edge-width test below.
    ``return_sep=True`` then supplies pycorr's pair-count-weighted separation
    coordinates, which are the fit coordinates. Pycorr performs the mu
    integration into the requested multipoles. Concatenation below is locally
    multipole-major (all ell=0 bins, then ell=2, then ell=4). The pinned desilike
    observable is expected to use the same data/covariance layout; external
    compatibility evidence establishes that agreement. The core fit-cell route
    always supplies ``rebin_factor`` from its immutable configuration snapshot; the default value
    performs identity rebinning only for direct helper callers.

    **Membership is decided by the bin EDGES, never by the pair-weighted
    separations.** A pair-weighted separation is a pair-count-weighted mean
    inside a bin, so it sits slightly above the geometric centre and is a
    property of the measured counts. Masking on it against a centre limit made
    the *number of fitted bins* depend on the measurement: on the frozen grid it
    dropped the top bin, yielding twenty-two bins from a range that names
    twenty-three. Selecting the bins whose own edges lie inside the closed
    interval is deterministic, so ``[48, 152]`` on 4 Mpc/h bins retains exactly
    twenty-six bins for every realization and every route. The pair-weighted
    separations are still returned as the fit coordinates and are still required
    to lie inside their own edge pair; they simply no longer decide membership.
    """
    if (
        isinstance(rebin_factor, (bool, np.bool_))
        or not isinstance(rebin_factor, (int, np.integer))
        or rebin_factor < 1
    ):
        raise ValueError("rebin_factor must be positive")
    if not np.isfinite(expected_bin_width_mpc_h) or float(expected_bin_width_mpc_h) <= 0:
        raise ValueError("expected fit-bin width must be positive")
    rebinned = xi if int(rebin_factor) == 1 else xi[:: int(rebin_factor)]
    # ``return_std=False`` is stated rather than left to the default, and the default is the
    # reason. pycorr resolves ``return_std or return_std is None and hasattr(self, 'cov')``, so an
    # estimator that happens to carry a covariance -- a jackknife one does -- silently returns a
    # third value and this unpacking fails. The fitter derives a vector and never a standard
    # deviation, so saying so makes the return shape a property of what is asked for rather than
    # of which estimator arrived. Plain estimators have no ``cov`` and already resolved to False,
    # so nothing about the accepted route moves.
    separation, poles = rebinned(ells=list(multipoles), return_sep=True, return_std=False)
    separation = np.asarray(separation, dtype=float)
    radial_edges = np.asarray(rebinned.edges[0], dtype=float)
    if radial_edges.ndim != 1 or radial_edges.size != separation.size + 1:
        raise ValueError("rebinned xi radial edges do not match its separation coordinates")
    # Mpc/h fit edges are exact declared grid coordinates. Absolute 1e-10 only
    # absorbs library float serialization; relative slack would grow with scale.
    widths = np.diff(radial_edges)
    if np.any(widths <= 0) or not np.allclose(
        widths, float(expected_bin_width_mpc_h), rtol=0.0, atol=1e-10
    ):
        raise ValueError("rebinned xi edges differ from the declared fit-bin width")
    lo, hi = [float(value) for value in fit_edge_limits_mpc_h]
    if not lo < hi:
        raise ValueError("xi fit range must have increasing bounds")
    # A bin is fitted when BOTH of its edges lie inside the closed interval. The 1e-9 slack
    # absorbs library float serialization of exact grid coordinates and nothing else; it is
    # absolute because relative slack would grow with separation.
    mask = (radial_edges[:-1] >= lo - 1e-9) & (radial_edges[1:] <= hi + 1e-9)
    if not np.any(mask):
        raise ValueError("xi contains no bins in the fit range")
    fit_separation = separation[mask]
    selected_indices = np.flatnonzero(mask)
    if np.any(np.diff(selected_indices) != 1):
        raise ValueError("xi fit-range bins are not one contiguous edge interval")
    fit_edges = radial_edges[selected_indices[0] : selected_indices[-1] + 2]
    # The retained window must be the whole requested interval, not the part of it the
    # measurement happened to cover. A grid that starts above ``lo`` or stops below ``hi``
    # would silently fit a shorter vector of the right shape.
    if abs(float(fit_edges[0]) - lo) > 1e-9 or abs(float(fit_edges[-1]) - hi) > 1e-9:
        raise ValueError(
            f"xi grid does not span the fitted edge interval [{lo}, {hi}]: "
            f"retained [{float(fit_edges[0])}, {float(fit_edges[-1])}]"
        )
    if np.any(~np.isfinite(fit_separation)):
        raise ValueError("xi fit-range pair-weighted separations must be finite")
    if np.any(fit_separation < fit_edges[:-1]) or np.any(fit_separation > fit_edges[1:]):
        raise ValueError("xi pair-weighted separation lies outside its exact edge pair")
    vector = np.concatenate([np.asarray(pole)[mask] for pole in poles])
    if vector.size != fit_separation.size * len(tuple(multipoles)):
        raise ValueError("xi multipoles do not form the declared rectangular fit vector")
    if not np.all(np.isfinite(vector)):
        raise ValueError("required xi fit vector must be finite")
    return fit_separation, vector, fit_edges


SOLVER = "observed_svd_v1"
FAMILY_SOLVER = "observed_svd_families_v1"


def full_mu_support(rr, output_edges):
    """Prove the RR normalization covers [-1, 1] in every fitted radial bin."""
    s, mu, counts = (np.asarray(rr[key], dtype=float) for key in ("sedges", "muedges", "wcounts"))
    if (
        s.ndim != 1
        or mu.ndim != 1
        or counts.shape != (s.size - 1, mu.size - 1)
        or not all(np.isfinite(x).all() for x in (s, mu, counts))
        or np.any(np.diff(s) <= 0)
        or np.any(np.diff(mu) <= 0)
        or np.any(counts < 0)
    ):
        raise ValueError("invalid RR support for nuisance null-space proof")
    if not np.array_equal(mu[[0, -1]], [-1.0, 1.0]):
        return False
    for edges in output_edges:
        indices = np.searchsorted(s, edges)
        if np.any(indices >= s.size) or not np.array_equal(s[indices], edges):
            raise ValueError("fitted radial edges do not align with RR support")
        for left, right in zip(indices[:-1], indices[1:]):
            if right <= left or np.any(counts[left:right].sum(axis=0) <= 0):
                return False
    return True


def factor_basis(basis, covariance, names):
    """Factor L^-1 A D^-1 = U S V^T, retaining every supplied direction.

    Prediction = L U eta; eta = S V^T D a. Flat da differs from flat deta
    by exp(-log_abs_det_raw_to_orthonormal), a parameter-independent constant.
    Native likelihood conventions omit absolute flat-prior normalizations; the
    recorded Jacobian allows their explicit reconstruction across coordinates.
    """
    basis, covariance = np.asarray(basis, dtype=float), np.asarray(covariance, dtype=float)
    if (
        basis.ndim != 2
        or basis.shape[1] != len(names)
        or not names
        or len(set(names)) != len(names)
        or basis.shape[0] < basis.shape[1]
        or covariance.shape != (basis.shape[0], basis.shape[0])
        or not np.isfinite(basis).all()
        or not np.isfinite(covariance).all()
        or not np.allclose(covariance, covariance.T, rtol=1e-12, atol=0)
    ):
        raise ValueError("invalid observed nuisance basis or covariance")
    chol = np.linalg.cholesky(covariance)
    whitened = np.linalg.solve(chol, basis)
    scales = np.linalg.norm(whitened, axis=0)
    relative_cut = np.finfo(float).eps * max(basis.shape)
    if np.any(scales <= relative_cut * scales.max()):
        raise ValueError("unproven null or unresolved nuisance column; refusing to truncate")
    u, singular, vt = np.linalg.svd(whitened / scales, full_matrices=False)
    if singular[-1] <= relative_cut * singular[0]:
        raise ValueError("additional nuisance rank deficiency; refusing to truncate")
    forward = singular[:, None] * vt * scales[None, :]
    inverse = (vt.T / singular) / scales[:, None]
    observed = chol @ u
    white_observed = np.linalg.solve(chol, observed)
    if not np.allclose(
        white_observed.T @ white_observed, np.eye(len(names)), rtol=1e-11, atol=1e-11
    ):
        raise ValueError("orthonormal nuisance precision check failed")
    metadata = {
        "solver": SOLVER,
        "raw_names": list(names),
        "coordinate_names": [f"alobs_{i}" for i in range(len(names))],
        "effective_rank": len(names),
        "whitened_column_scales": scales.tolist(),
        "scaled_singular_values": singular.tolist(),
        "scaled_condition_number": float(singular[0] / singular[-1]),
        "raw_to_orthonormal": forward.tolist(),
        "orthonormal_to_raw": inverse.tolist(),
        "observed_basis": observed.tolist(),
        "log_abs_det_raw_to_orthonormal": float(np.log(singular).sum() + np.log(scales).sum()),
        "flat_measure": "native unnormalized flat orthonormal measure; raw measure subtracts log Jacobian",
        "covariance_sha256": hashlib.sha256(np.ascontiguousarray(covariance).tobytes()).hexdigest(),
    }
    return observed, metadata


def nuisance_columns(theory, model):
    """Exact native correlation columns, preserving the chosen family's fixed terms."""
    if model not in ("power", "power3", "pcs"):
        raise ValueError("unsupported broadband basis")
    active = {p.basename for p in theory.all_params if not p.fixed}
    for index, ell in enumerate(theory.ells):
        for row, (name, power) in enumerate(theory.broadband_orders[ell].items()):
            if model != "power" and name not in active:
                continue
            raw = np.zeros((len(theory.ells), len(theory.s)))
            raw[index] = np.asarray(theory.broadband_matrix[ell][row])
            yield str(name), ell, power == 0, raw
    if model == "pcs":
        # The installed transform is linear. Transform the spline kernel itself,
        # avoiding subtraction of two almost identical physical templates.
        for index, ell in enumerate(theory.ells):
            for row, name in enumerate(theory.power.broadband_orders[ell]):
                if name not in active:
                    continue
                raw_power = np.zeros((len(theory.ells), len(theory.power.k)))
                raw_power[index] = np.asarray(theory.power.broadband_matrix[ell][row])
                yield str(name), ell, False, np.asarray(theory.get_corr(raw_power))


def repaired_observable(factory, kwargs, *, model="power"):
    """Build a native observable with additive, observed-space SVD templates."""
    rr = kwargs.get("wmatrix")
    if not isinstance(rr, dict) or "wcounts" not in rr:
        raise ValueError("observed SVD repair requires the explicit RR window")
    if kwargs.get("systematic_templates"):
        raise ValueError("repair cannot replace existing systematic templates")
    initial = factory(**kwargs)
    window = initial.wmatrix
    theory = window.theory
    theory.runtime_info.initialize()
    expected = "pcs" if model == "pcs" else "power"
    if theory.broadband != expected or window.offset is not None or window.matrix_diag is not None:
        raise ValueError("unsupported physical theory or window for nuisance repair")
    complete = full_mu_support(rr, window.sedges)
    columns, names, nulls, null_residuals = [], [], [], {}
    for name, ell, constant, raw in nuisance_columns(theory, model):
        column = np.asarray(window._apply(raw))
        if complete and ell not in window.ells and constant:
            # Orthogonality applies only to the constant, with full angular
            # support. Verify the actual operator agrees to roundoff too.
            bound = (
                np.finfo(float).eps
                * max(window.matrix_full.shape)
                * np.max(np.sum(np.abs(window.matrix_full), axis=1))
            )
            index = tuple(theory.ells).index(ell)
            if not np.all(raw[index] == 1) or np.max(np.abs(column)) > bound:
                raise ValueError("RR projection contradicts the analytic constant null direction")
            nulls.append(name)
            null_residuals[name] = float(np.max(np.abs(column)))
        else:
            names.append(name)
            columns.append(column)
    observed, metadata = factor_basis(np.column_stack(columns), initial.covariance, names)
    metadata.update(
        null_names=nulls,
        null_projection_maxabs=null_residuals,
        full_mu_support=complete,
        physical_multipoles=list(theory.ells),
        observed_multipoles=list(window.ells),
    )
    if model != "power":
        metadata.update(solver=FAMILY_SOLVER, broadband_model=model)
        # Cross the native calculator boundary, not just the matrix factorization.
        # Scale each probe to a resolved observed signal before subtracting the
        # physical template, so tiny kernel amplitudes cannot pass by cancellation.
        zero = dict.fromkeys(names + nulls, 0.0)
        initial(**zero)
        physical = np.asarray(initial.flattheory).copy()
        errors = {}
        for name, column in zip(names, columns):
            amplitude = 1.0 / np.max(np.abs(column))
            initial(**{**zero, name: amplitude})
            delta = np.asarray(initial.flattheory) - physical
            np.testing.assert_allclose(
                delta,
                amplitude * column,
                rtol=1e-8,
                atol=1e-10,
                err_msg="native family column differs: " + name,
            )
            errors[name] = float(np.max(np.abs(delta - amplitude * column)))
        initial(**zero)
        metadata["native_column_replay_maxabs"] = errors
    result = factory(
        **{**kwargs, "systematic_templates": dict(zip(metadata["coordinate_names"], observed.T))}
    )
    # Force initialization before attaching metadata: BaseCalculator initializes lazily.
    result.runtime_info.initialize()
    result.power_broadband_repair = metadata
    return result


def build_likelihood(
    data_vector: Sequence[float],
    covariance: np.ndarray,
    *,
    separations_mpc_h: Sequence[float],
    fit_sedges: Sequence[float],
    z_eff: float,
    multipoles: Sequence[int],
    reconstruction_state: str,
    reconstruction_convention: str,
    smoothing_mpc_h: float,
    ap_mode: str,
    method: str,
    broadband_model: Optional[str] = None,
    damping_prior: Mapping[str, Any],
    geometry_artifact: Mapping[str, Any],
    sample_identity: str,
    tracer: str,
    measurement_bin: str,
    config: Mapping[str, Any],
    only_now: bool = False,
) -> Tuple[Any, Any, Any, dict]:
    """Construct the single configured theory, RR window and Gaussian likelihood.

    Photo-z suppression multiplies the complete smooth-plus-wiggle power in observed
    coordinates. The nonlinear damping priors remain separate. Pre uses no reconstruction;
    post uses RecSym. The smooth comparison uses this same factory with only_now=True.
    Each caller supplies the exact route's saved RR geometry and fitting effective redshift.
    The fourth return value records the fixed angular rule; no convergence study runs here.
    """
    if ap_mode not in config["fitting"]["ap_modes"] or method not in config["fitting"]["methods"]:
        raise ValueError("fit AP mode or method is not registered")
    if reconstruction_state == "prereconstruction":
        mode, smoothing = "", 0.0
    elif reconstruction_state == "postreconstruction":
        if reconstruction_convention not in ("recsym", "reciso"):
            raise ValueError("post-reconstruction fit requires RecSym or the RecIso check")
        mode, smoothing = reconstruction_convention, float(smoothing_mpc_h)
    else:
        raise ValueError("reconstruction_state is not registered")
    try:
        from desilike.likelihoods import ObservablesGaussianLikelihood
        from desilike.observables.galaxy_clustering import (
            TracerCorrelationFunctionMultipolesObservable,
        )
        from desilike.theories.galaxy_clustering import (
            BAOPowerSpectrumTemplate,
            DampedBAOWigglesTracerCorrelationFunctionMultipoles,
        )
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("desilike is required for analysis fitting") from exc
    template = BAOPowerSpectrumTemplate(
        z=float(z_eff),
        fiducial=build_mock_cosmology(config),
        apmode=ap_mode,
        # False selects the wiggle-plus-smooth BAO template for the primary fit.
        only_now=only_now,
        with_now=str(config["fitting"]["no_wiggle_method"]),
    )
    broadband = str(broadband_model or config["fitting"]["broadband_model"])
    if broadband != str(config["fitting"]["broadband_model"]):
        raise ValueError("likelihood broadband model is not the one frozen broadband model")
    surface = config["fitting"]["fit_surfaces"][tracer]
    order = int(config["fitting"]["photoz_response"]["mu_quadrature_order"])
    quadrature = {
        "mu_quadrature_order": order,
        "convention": "positive_half_of_symmetrized_2n_point_Gauss_Legendre",
    }
    photoz_power = _build_observed_photoz_power_model(
        template=template,
        mode=mode,
        smoothing_radius=smoothing,
        pairwise_sigma_mpc_h=float(damping_prior["sigma_parallel_photoz_mpc_h"]),
        mu_quadrature_order=order,
    )
    theory = DampedBAOWigglesTracerCorrelationFunctionMultipoles(
        template=template,
        mode=mode,
        smoothing_radius=smoothing,
        broadband=broadband,
        pt=photoz_power,
    )
    separation = np.asarray(separations_mpc_h, dtype=float)
    vector = np.asarray(data_vector, dtype=float)
    if vector.size != separation.size * len(tuple(multipoles)):
        raise ValueError("likelihood data vector does not match its separation grid and multipoles")
    fit_width = float(config["fitting"]["input_bin_width_mpc_h"])
    fit_sedges = np.asarray(fit_sedges, dtype=float)
    if (
        fit_sedges.ndim != 1
        or fit_sedges.size < 2
        or not np.allclose(np.diff(fit_sedges), fit_width, rtol=0.0, atol=1e-10)
    ):
        raise ValueError("likelihood geometry does not expose exact 4 Mpc/h fit edges")
    if separation.size != fit_sedges.size - 1:
        raise ValueError("likelihood separation coordinates do not cover every configured fit bin")
    fit_lo, fit_hi = [float(value) for value in config["fitting"]["fit_edge_limits_mpc_h"]]
    if (
        abs(float(fit_sedges[0]) - fit_lo) > 1e-9
        or abs(float(fit_sedges[-1]) - fit_hi) > 1e-9
        or np.any(separation < fit_sedges[:-1])
        or np.any(separation > fit_sedges[1:])
    ):
        raise ValueError("likelihood separations do not occupy the exact admitted fit edges")
    geometry = likelihood_window_inputs(
        geometry_artifact,
        sample_identity=sample_identity,
        tracer=tracer,
        measurement_bin=measurement_bin,
        reconstruction_state=reconstruction_state,
        observed_multipoles=multipoles,
        model_multipoles=surface["model_multipoles"],
        fit_sedges=fit_sedges,
    )
    observable_inputs = dict(
        data=vector,
        covariance=covariance,
        theory=theory,
        ells=tuple(int(value) for value in multipoles),
        s=separation,
        sedges=fit_sedges,
        wmatrix=geometry["wmatrix"],
        ellsin=geometry["ellsin"],
        systematic_templates=geometry["systematic_templates"],
    )
    transfer_pcs2 = (
        broadband == "pcs2"
        and "transfer_run" in config["measurement"]["reconstruction"]
        and "prior" not in config["measurement"]["reconstruction"]["transfer_run"]
        and config["measurement"]["reconstruction"]["smoothing_values_mpc_h"]
        == [
            config["measurement"]["reconstruction"]["exploratory_transfer"][
                "common_smoothing_mpc_h"
            ]
        ]
    )
    if config["schema_version"] == 15 and broadband != "power3" and not transfer_pcs2:
        raise ValueError("schema-15 requires power3 or the common-radius transfer PCS2 check")
    # Schema 15 introduced power3; the per-bin schemas keep its observed-space repair.
    if config["schema_version"] >= 15 and broadband == "power3":
        observable = repaired_observable(
            TracerCorrelationFunctionMultipolesObservable, observable_inputs, model=broadband
        )
        basis = observable.power_broadband_repair
        expected_raw = {
            f"al{ell}_{power}" for ell in surface["model_multipoles"] for power in (-2, -1, 0)
        } - {"al4_0"}
        expected_coordinates = config["fitting"]["linear_nuisance"]["allowed_basenames_by_model"][
            broadband
        ]
        if (
            not basis["full_mu_support"]
            or basis["null_names"] != ["al4_0"]
            or set(basis["raw_names"]) != expected_raw
            or basis["coordinate_names"] != expected_coordinates
            or basis["effective_rank"] != len(expected_coordinates)
        ):
            raise ValueError("power3 observable span differs from its frozen definition")
    else:
        observable = TracerCorrelationFunctionMultipolesObservable(**observable_inputs)
    if tuple(int(value) for value in observable.wmatrix.ellsin) != tuple(
        int(value) for value in surface["model_multipoles"]
    ):
        raise RuntimeError("initialized xi window changed the required model multipoles")
    # Window initialization re-enters the theory; verify it preserved the requested fixed rule.
    realized_nodes = np.asarray(theory.pt.mu, dtype=float)
    expected_nodes, _ = symmetric_leggauss(order)
    if realized_nodes.size != order or not np.allclose(
        realized_nodes, expected_nodes, rtol=0.0, atol=1e-12
    ):
        raise RuntimeError("initialized likelihood changed the fixed quadrature rule")
    likelihood = ObservablesGaussianLikelihood(observables=[observable])
    _configure_likelihood_parameters(
        likelihood,
        damping_prior,
        method=method,
        ap_mode=ap_mode,
        tracer=tracer,
        broadband_model=broadband,
        config=config,
        only_now=only_now,
    )
    return theory, observable, likelihood, quadrature


def _configure_likelihood_parameters(
    likelihood: Any,
    damping_prior: Mapping[str, Any],
    *,
    method: str,
    ap_mode: str,
    tracer: Optional[str] = None,
    broadband_model: Optional[str] = None,
    config: Mapping[str, Any],
    only_now: bool = False,
) -> None:
    """Install and verify the complete configured varied-parameter surface.

    The configured ``derived`` treatment is installed verbatim. When supplied,
    the pinned desilike version is expected to use ``.auto`` to profile eligible
    linear nuisances and ``.prec`` for fixed-linear-derivative analytic
    marginalization, so those coordinates are not sampled. Pinned-version
    compatibility evidence, not this function, establishes that behavior and
    the basename-pattern selection semantics. Locally, empty selection and the
    final varied-surface assertion fail closed.

    The contract has three parameter families and one final surface proof:

    1. every broadband nuisance matching the configured basename patterns gets
       the method-specific analytic/profile treatment;
    2. the tracer surface and AP-mode list jointly define all non-damping model
       coordinates. Active names receive YAML value/prior/reference/proposal/
       delta state. Every inactive configured name must appear in the tracer's
       explicit ``fixed_parameters`` mapping with the same value, is fixed when
       the library exposes it, and is re-read after the update;
    3. ``sigmas``, ``sigmapar``, and ``sigmaper`` are always varied under the
       accepted damping prior. Each Gaussian receives declared ``bounds_mpc_h``
       as desilike prior limits, which the pinned version is expected to apply as
       hard truncation; a posterior reaching a bound is prior-limited and must
       be judged by external acceptance. Only ``fixed`` and that prior are set
       and re-verified for damping coordinates, including the Gaussian
       distribution family; start value, reference, proposal,
       and finite-difference state deliberately remain the pinned library's
       defaults for the installed prior; and
    4. the realized varied set must contain all active nonlinear/damping names
       and nothing outside those names plus the recognized nuisances. This final
       comparison uses desilike basenames, so configured nonlinear names must be
       basenames rather than fully qualified parameter names.

    Nonlinear parameter numeric state uses one absolute ``1e-12`` comparison policy, including
    proposal and delta; paired nulls compare exactly. Every field declared by a
    nonlinear prior/reference mapping is re-read recursively; extra library
    metadata is ignored, but no configured distribution field may disappear or
    change. Desilike prior objects
    expose serialized distribution state through ``__getstate__`` while test or
    mapping adapters may already be mappings. A nested ``attrs`` mapping is
    promoted with ``setdefault`` so explicit top-level state wins while the
    pinned library's nested serialization remains comparable. Lacking either interface fails
    closed rather than being mistaken for a changed physical prior. Linear
    nuisances are re-read only for their method-specific treatment; their
    installed value/prior/reference/delta fields are not re-verified.
    """
    # 1. Linear broadband nuisances and their method-specific treatment.
    nuisance_config = config["fitting"]["linear_nuisance"]
    nuisance_treatments = nuisance_config.get("treatment")
    if not isinstance(nuisance_treatments, Mapping) or method not in nuisance_treatments:
        raise ValueError(f"linear nuisance treatment is not configured for method {method!r}")
    nuisance_mode = str(nuisance_treatments[method])
    nuisance_patterns = list(nuisance_config["parameter_patterns"])
    all_nuisances = list(likelihood.all_params.select(basename=nuisance_patterns))
    if not all_nuisances:
        raise ValueError("likelihood exposes no registered linear broadband nuisance parameters")
    if tracer is None:
        matching = [
            name
            for name, surface in config["fitting"]["fit_surfaces"].items()
            if surface["ap_mode"] == ap_mode
        ]
        if len(matching) != 1:
            raise ValueError("tracer cannot be inferred uniquely from the AP mode")
        tracer = matching[0]
    model = str(broadband_model or config["fitting"]["broadband_model"])
    allowed_by_model = set(nuisance_config["allowed_basenames_by_model"].get(model, []))
    if not allowed_by_model:
        raise ValueError("broadband model has no declared nuisance surface")
    configured_nuisances = (
        set(nuisance_config["varied_basenames_by_tracer"][str(tracer)]) & allowed_by_model
    )

    def nuisance_basename(parameter: Any) -> str:
        return str(getattr(parameter, "basename", str(parameter.name).rsplit(".", 1)[-1]))

    exposed_basenames = {nuisance_basename(parameter) for parameter in all_nuisances}
    missing_nuisances = configured_nuisances - exposed_basenames
    if missing_nuisances:
        raise ValueError(
            f"likelihood omits configured nuisance coordinates {sorted(missing_nuisances)}"
        )
    nuisances = [
        parameter
        for parameter in all_nuisances
        if nuisance_basename(parameter) in configured_nuisances
    ]
    fixed_nuisances = [parameter for parameter in all_nuisances if parameter not in nuisances]
    for parameter in fixed_nuisances:
        parameter.update(value=0.0, fixed=True, derived=False)
        if parameter.fixed is not True:
            raise RuntimeError(
                f"likelihood nuisance parameter {nuisance_basename(parameter)} was not fixed"
            )
    nuisance_names = [str(parameter.name) for parameter in nuisances]
    for name in nuisance_names:
        nuisance_prior = nuisance_config["prior"]
        likelihood.all_params[name].update(
            fixed=False,
            value=float(nuisance_config["value"]),
            # Null deliberately preserves desilike's unbounded linear prior;
            # a declared mapping is copied so fits cannot share mutable limits.
            prior=None if nuisance_prior is None else dict(nuisance_prior),
            ref=dict(nuisance_config["reference"]),
            delta=float(nuisance_config["finite_difference_step"]),
            derived=nuisance_mode,
        )
    # Read back out of the container the fit will actually use, not off the objects captured
    # before the write. ``select`` does not always hand back references into ``all_params``, so a
    # verification over the captured list can report a treatment that was never installed, and --
    # worse in the other direction -- can report success when the container did not take the
    # update. Asking the container is the only reading that describes the likelihood being built.
    realized_nuisances = {name: likelihood.all_params[name].derived for name in nuisance_names}
    if set(realized_nuisances) != set(nuisance_names) or any(
        value != nuisance_mode for value in realized_nuisances.values()
    ):
        raise RuntimeError("likelihood nuisance treatment did not remain configured")

    # 2. AP-mode nonlinear model coordinates and their complete realized state.
    parameter_config = config["fitting"]["nonlinear_parameters"]
    if ap_mode not in config["fitting"]["nonlinear_parameter_names_by_ap_mode"]:
        raise ValueError("likelihood AP mode is not registered")
    surface = config["fitting"]["fit_surfaces"].get(str(tracer))
    if not isinstance(surface, Mapping) or surface.get("ap_mode") != ap_mode:
        raise ValueError("tracer fit surface does not match the requested AP mode")
    active_names = list(config["fitting"]["nonlinear_parameter_names_by_ap_mode"][ap_mode])
    missing_parameter_configs = [name for name in active_names if name not in parameter_config]
    if missing_parameter_configs:
        raise ValueError(
            "nonlinear parameter configuration omits active names " f"{missing_parameter_configs}"
        )
    fixed_parameters = surface.get("fixed_parameters")
    if not isinstance(fixed_parameters, Mapping):
        raise ValueError("tracer fit surface has no explicit fixed-parameter mapping")
    inactive_names = set(parameter_config) - set(active_names)
    if set(fixed_parameters) != inactive_names:
        raise ValueError(
            "tracer fixed-parameter mapping differs from the inactive nonlinear surface"
        )
    for name, value in fixed_parameters.items():
        if not np.isclose(
            float(value), float(parameter_config[name]["value"]), rtol=0.0, atol=1e-12
        ):
            raise ValueError(
                f"tracer fixed parameter {name} differs from its configured nonlinear value"
            )

    def distribution_state(realized: Any) -> Optional[dict]:
        """Serialize a prior/reference distribution or a full parameter state."""
        if isinstance(realized, Mapping):
            state = dict(realized)
        elif hasattr(realized, "__getstate__"):
            serialized = realized.__getstate__()
            if not isinstance(serialized, Mapping):
                return None
            state = dict(serialized)
        else:
            return None
        attributes = state.get("attrs")
        if isinstance(attributes, Mapping):
            for name, value in attributes.items():
                state.setdefault(name, value)
        return state

    def optional_numeric_matches(realized: Any, expected: Any) -> bool:
        if realized is None or expected is None:
            return realized is None and expected is None
        observed = np.asarray(realized, dtype=float)
        wanted = np.asarray(expected, dtype=float)
        return observed.shape == wanted.shape and np.allclose(
            observed, wanted, rtol=0.0, atol=1e-12
        )

    def finite_difference_step_matches(realized: Any, expected: Any) -> bool:
        """Match YAML scalars to desilike's exact symmetric two-sided step."""
        if realized is None or expected is None:
            return realized is None and expected is None
        wanted = np.asarray(expected, dtype=float)
        observed = np.asarray(realized, dtype=float)
        if wanted.shape != () or observed.shape not in {(), (2,)}:
            return False
        return np.allclose(observed, wanted.item(), rtol=0.0, atol=1e-12)

    def configured_state_matches(realized: Any, expected: Any) -> bool:
        """Compare configured fields; test strings/bools before numeric coercion."""
        if isinstance(expected, Mapping):
            return isinstance(realized, Mapping) and all(
                name in realized and configured_state_matches(realized[name], value)
                for name, value in expected.items()
            )
        if expected is None or isinstance(expected, (str, bool)):
            return realized == expected
        try:
            return optional_numeric_matches(realized, expected)
        except (TypeError, ValueError):
            return realized == expected

    def distribution_matches(realized: Any, expected: Mapping[str, Any]) -> bool:
        state = distribution_state(realized)
        return state is not None and configured_state_matches(state, expected)

    for name in active_names:
        record = parameter_config[name]
        try:
            parameter = likelihood.all_params[name]
        except (KeyError, IndexError) as exc:
            raise ValueError(
                f"likelihood exposes no configured nonlinear parameter {name}"
            ) from exc
        parameter.update(
            value=float(record["value"]),
            prior=dict(record["prior"]),
            ref=dict(record["reference"]),
            proposal=record["proposal"],
            delta=record["finite_difference_step"],
        )
        realized = likelihood.all_params[name]
        state = distribution_state(realized)
        if state is None:
            raise RuntimeError(f"likelihood nonlinear parameter {name} exposes no state")
        if (
            not np.isclose(float(realized.value), float(record["value"]), rtol=0.0, atol=1e-12)
            or not distribution_matches(realized.prior, record["prior"])
            or not distribution_matches(realized.ref, record["reference"])
            or not optional_numeric_matches(state.get("proposal"), record["proposal"])
            or not finite_difference_step_matches(
                state.get("delta"), record["finite_difference_step"]
            )
        ):
            raise RuntimeError(
                f"likelihood nonlinear parameter {name} differs from its complete configured state"
            )

    for name in sorted(inactive_names):
        try:
            parameter = likelihood.all_params[name]
        except (KeyError, IndexError):
            continue
        fixed_value = float(fixed_parameters[name])
        parameter.update(value=fixed_value, fixed=True)
        if parameter.fixed is not True or not np.isclose(
            float(parameter.value), fixed_value, rtol=0.0, atol=1e-12
        ):
            raise RuntimeError(f"likelihood inactive nonlinear parameter {name} was not fixed")

    # 3. Accepted sample-specific damping coordinates and realized priors.
    centers = damping_prior["centers_mpc_h"]
    widths = damping_prior["widths_mpc_h"]
    bounds = damping_prior["bounds_mpc_h"]
    damping_names = {"sigmas", "sigmapar", "sigmaper"}
    for desilike_name, record_name in (
        ("sigmas", "sigma_streaming"),
        ("sigmapar", "sigma_parallel"),
        ("sigmaper", "sigma_perp"),
    ):
        try:
            damping_parameter = likelihood.all_params[desilike_name]
        except (KeyError, IndexError) as exc:
            raise ValueError(
                f"likelihood exposes no required damping parameter {desilike_name}"
            ) from exc
        configured_limits = [
            np.inf if value is None else float(value) for value in bounds[record_name]
        ]
        # The transverse calibration route varies one coordinate on its physical support with no
        # Gaussian, because the calibration that produces that Gaussian cannot presuppose it. The
        # support is unchanged; only the informative term is absent.
        unpriored = record_name in set(damping_prior.get("unpriored_parameters", ()))
        damping_parameter.update(
            fixed=False,
            prior=(
                {"limits": configured_limits}
                if unpriored
                else {
                    "dist": "norm",
                    "loc": float(centers[record_name]),
                    "scale": float(widths[record_name]),
                    "limits": configured_limits,
                }
            ),
        )
        realized = likelihood.all_params[desilike_name]
        if realized.fixed is not False or realized.prior is None:
            raise RuntimeError(f"likelihood damping parameter {desilike_name} was not configured")
        realized_prior = distribution_state(realized.prior)
        if realized_prior is None:
            raise RuntimeError(
                f"likelihood damping parameter {desilike_name} exposes no prior state"
            )

        expected_limits = np.asarray(configured_limits, dtype=float)
        if unpriored:
            # Only the support is asserted. A realized Gaussian here would mean the calibration
            # was run under the prior it exists to produce.
            observed_limits = np.asarray(realized_prior.get("limits"), dtype=float)
            if observed_limits.shape != (2,) or not np.allclose(
                observed_limits, expected_limits, rtol=0.0, atol=1e-12
            ):
                raise RuntimeError(
                    f"likelihood damping parameter {desilike_name} changed its physical support"
                )
            if realized_prior.get("dist") in {"norm"}:
                raise RuntimeError(
                    f"likelihood damping parameter {desilike_name} was given a Gaussian on the "
                    f"calibration route, which presupposes the prior being calibrated"
                )
            continue
        expected_loc = float(centers[record_name])
        expected_scale = float(widths[record_name])
        observed_loc = realized_prior.get("loc")
        observed_scale = realized_prior.get("scale")
        observed_limit_values = realized_prior.get("limits")
        observed_distribution = realized_prior.get("dist")
        if (
            observed_distribution is None
            or observed_loc is None
            or observed_scale is None
            or observed_limit_values is None
        ):
            raise RuntimeError(
                f"likelihood damping parameter {desilike_name} prior state is incomplete"
            )
        observed_limits = np.asarray(observed_limit_values, dtype=float)
        if (
            observed_distribution != "norm"
            or not np.isclose(float(observed_loc), expected_loc, rtol=0.0, atol=1e-12)
            or not np.isclose(float(observed_scale), expected_scale, rtol=0.0, atol=1e-12)
            or observed_limits.shape != (2,)
            or not np.allclose(observed_limits, expected_limits, rtol=0.0, atol=1e-12)
        ):
            raise RuntimeError(
                f"likelihood damping parameter {desilike_name} changed its configured prior"
            )

    # The smooth template removes the acoustic component, the only term dilated by
    # this model. Keep its configured AP values/priors, but exclude these flat
    # coordinates from optimization. Damping priors and broadband remain active.
    smooth_dilations = (
        set(config["fitting"]["ap_interval_parameters"][ap_mode]) if only_now else set()
    )
    # Under flat support the smooth template gives these two acoustic damping scales no
    # response; fix them as the comparison diagnostic does. Streaming damping stays free.
    if only_now and damping_prior.get("scope") in (
        FINITE_FLAT_DAMPING_SCOPE,
        COMMON_FLAT_DAMPING_SCOPE,
        TRANSFER_FLAT_DAMPING_SCOPE,
    ):
        smooth_dilations |= {"sigmapar", "sigmaper"}
    for name in sorted(smooth_dilations):
        likelihood.all_params[name].update(fixed=True)
        if likelihood.all_params[name].fixed is not True:
            raise RuntimeError(f"smooth likelihood inactive dilation {name} was not fixed")

    # 4. Exact final varied surface: required coordinates and no undeclared ones.
    def basename(parameter: Any) -> str:
        return str(getattr(parameter, "basename", str(parameter.name).rsplit(".", 1)[-1]))

    varied_names = {basename(parameter) for parameter in likelihood.varied_params}
    required_names = (set(active_names) | damping_names) - smooth_dilations
    allowed_names = required_names | {basename(parameter) for parameter in nuisances}
    if not required_names.issubset(varied_names) or not varied_names.issubset(allowed_names):
        raise RuntimeError(
            "likelihood varied-parameter surface is incomplete or contains an undeclared parameter"
        )


def _require_single_process(mpicomm: Any, operation: str) -> None:
    """Reject distributed fitting before inference or product writes begin.

    One cell owns one create-only output set. Each desilike inference route
    checks its communicator before work so MPI ranks cannot race on those paths.
    """
    if int(getattr(mpicomm, "size", -1)) != 1 or int(getattr(mpicomm, "rank", -1)) != 0:
        raise RuntimeError(f"{operation} requires exactly one process")


def applied_damping_prior(damping, route, config):
    """Install a configured flat robustness prior, which exists only on its comparison routes.

    The finite and the common flat checks fit the same routes.
    """
    from .theory import common_flat_damping_prior, finite_flat_damping_prior

    applied = config["fitting"]["damping"].get("applied_prior")
    if applied is None:
        return damping
    scope = config["fitting"]["paper_comparisons"]
    if (
        route["measurement_bin"] not in scope["measurement_bins"]
        or route["condition"] != scope["condition"]
        or route["reconstruction_state"] != scope["reconstruction_state"]
    ):
        raise ValueError("the flat damping priors apply only inside their comparison")
    if applied == "paper_finite_flat":
        return finite_flat_damping_prior(damping, scope["flat_damping"])
    return common_flat_damping_prior(
        damping, config["fitting"]["damping"]["common_flat_sensitivity"]
    )


def _route_likelihood(run, unit, *, method, only_now=False):
    from .covariance import read_route_covariance, likelihood_covariance_run
    from .selection import read_preparation
    from .selection import tier_spec

    config = run["config"]
    route = {
        key: unit[key] for key in ("tracer", "measurement_bin", "condition", "reconstruction_state")
    }
    source = run.get("source_run", run)
    covariance_run = likelihood_covariance_run(run, route)
    saved = read_route_covariance(covariance_run, route)
    arrays = saved["arrays"]
    identifier = unit.get("realization_id")
    if identifier is None:
        data = arrays["vectors"].mean(axis=0)
        separation = arrays["separations"].mean(axis=0)
        covariance = arrays["mean_covariance"]
    else:
        indices = np.flatnonzero(arrays["realization_ids"] == identifier)
        if type(identifier) is not int or len(indices) != 1:
            raise ValueError("individual fit realization is absent from its covariance ensemble")
        index = indices.item()
        data, separation, covariance = (
            arrays["vectors"][index],
            arrays["separations"][index],
            arrays["covariance"],
        )
    preparation = read_preparation(source, route["tracer"])
    tracer, condition, name = route["tracer"], route["condition"], route["measurement_bin"]
    zeff = preparation["effective_redshifts"][condition][name]["z_eff"]
    bin_spec = _measurement_bin(tracer, name, config)
    if not bin_spec["zmin"] <= zeff < bin_spec["zmax"]:
        raise ValueError("fitting effective redshift lies outside its own measurement bin")
    groups = preparation["conditions"][condition]
    density = sum(groups[cap]["counts"][name] for cap in ("NGC", "SGC")) / sum(
        groups[cap]["volumes_mpc_h3"][name] for cap in ("NGC", "SGC")
    )
    cosmology = build_mock_cosmology(config)
    analytic = None
    transfer = config["measurement"]["reconstruction"].get("transfer_run")
    if route["reconstruction_state"] == "postreconstruction" and transfer is None:
        interpolator = cosmology.get_fourier().pk_interpolator(of="delta_m")
        analytic = analytic_postreconstruction_sigma_perp(
            tracer=tracer,
            attained_density_h3_mpc3=density,
            linear_power=lambda k: interpolator(k, z=zeff),
            drag_scale_mpc_h=float(cosmology.rs_drag),
            config=config,
        )
    sigma0 = tier_spec(condition, config)["sigma_z_over_one_plus_z"]
    damping_options = {}
    if transfer is not None:
        from .theory import transfer_damping_context

        # Source-target provenance for the fit record, whichever prior is applied.
        damping_options["context"] = transfer_damping_context(
            config, preparation, tracer=tracer, measurement_bin=name, target_class=condition
        )
    flat_check = config["fitting"].get("core_fit_check") == UNIFIED_FLAT
    if config["fitting"]["damping"].get("applied_prior") == UNIFIED_DAMPING_PRIOR or flat_check:
        # Revision 2.20: one configured prior per redshift bin for every run, route and state,
        # so neither the analytic centre nor reviewed source-target centres enter. Its flat
        # check replaces those priors by the enlarged uniform support.
        damping = unified_damping_prior(
            tracer,
            route["reconstruction_state"],
            measurement_bin=name,
            smoothing_mpc_h=config["measurement"]["reconstruction"]["smoothing_values_mpc_h"][0],
            sigma_z_over_one_plus_z=sigma0,
            z_eff=zeff,
            e_z_eff=float(cosmology.efunc(zeff)),
            config=config,
        )
        if flat_check:
            damping = applied_damping_prior(damping, route, config)
    else:
        resolver = resolve_damping_prior
        if transfer is not None:
            from .theory import resolve_transfer_damping_prior

            resolver = resolve_transfer_damping_prior
        damping = resolver(
            tracer,
            route["reconstruction_state"],
            sigma_z_over_one_plus_z=sigma0,
            z_eff=zeff,
            e_z_eff=float(cosmology.efunc(zeff)),
            postreconstruction_sigma_perp_artifact=analytic,
            config=config,
            **damping_options,
        )
        damping = applied_damping_prior(damping, route, config)
    surface = config["fitting"]["fit_surfaces"][tracer]
    sample_identity = "|".join(
        route[key] for key in ("measurement_bin", "condition", "reconstruction_state")
    )
    geometry = {**saved["window"], **route, "sample_identity": sample_identity}
    theory, observable, likelihood, quadrature = build_likelihood(
        data,
        covariance,
        separations_mpc_h=separation,
        fit_sedges=arrays["fit_edges"],
        z_eff=zeff,
        multipoles=arrays["multipoles"],
        reconstruction_state=route["reconstruction_state"],
        reconstruction_convention=config["measurement"]["reconstruction"]["allowed_conventions"][0],
        smoothing_mpc_h=config["measurement"]["reconstruction"]["smoothing_values_mpc_h"][0],
        ap_mode=surface["ap_mode"],
        method=method,
        damping_prior=damping,
        geometry_artifact=geometry,
        sample_identity=sample_identity,
        tracer=tracer,
        measurement_bin=name,
        config=config,
        only_now=only_now,
    )
    source_class = (
        condition
        if transfer is None or transfer["source_class"] == "self"
        else transfer["source_class"]
    )
    metadata = {
        "route": route,
        "realization_id": identifier,
        "z_eff": zeff,
        "reconstruction_inputs": {
            **preparation["effective_redshifts"][source_class][
                name if per_bin_reconstruction(config) else "parent"
            ],
            "bias": config["tracers"][tracer]["reconstruction"]["fiducial_bias"],
            "smoothing_mpc_h": config["measurement"]["reconstruction"]["smoothing_values_mpc_h"][0],
        },
        "fitting_growth_rate": float(cosmology.growth_rate(zeff)),
        "attained_density_h3_mpc3": density,
        "damping_prior": damping,
        "analytic_transverse_damping": analytic,
        "quadrature": quadrature,
        "covariance_interpretation": saved["metadata"]["mean_covariance_interpretation"],
        "covariance_identity": {
            key: covariance_run["metadata"][key]
            for key in ("run_name", "code_commit", "analysis_config_sha256")
        },
        "ap_mode": surface["ap_mode"],
        "units": "Mpc/h",
        "ordering": "multipole_major",
    }
    if transfer is not None:
        metadata["reconstruction_inputs"].update(
            source_class=source_class,
            target_class=condition,
            source_target_context=damping_options["context"],
        )
    if config["schema_version"] >= 15 and config["fitting"]["broadband_model"] == "power3":
        metadata["broadband_representation"] = observable.power_broadband_repair
    if "source_run" in run:
        metadata["source_run"] = run["metadata"]["source_run"]
    return theory, observable, likelihood, metadata, arrays, covariance


TRANSFER_CENTRE_DECISION = (
    "2026-10-13 — Transfer damping centres by predicted-multipole compression "
    "(APPROVED DESIGN, IMPLEMENTED, VALIDATION PENDING)"
)
TRANSFER_CENTRE_BOUNDS_MPC_H = (0.0, 30.0)


def _fix_parameters(likelihood, values):
    for name, value in values.items():
        likelihood.all_params[name].update(value=float(value), fixed=True, derived=False)


def _predict_and_compress(common, prediction_values, *, damping, streaming, config):
    """Predict windowed multipoles under a supplied wiggle damping, then fit the native template.

    ``damping`` is an array on the template's (k, mu) grid, or a callable building it from
    that grid. The prediction uses fixed dilations, bias and streaming with zero broadband;
    the compression varies b1, sigmaper, sigmapar and the native broadband nuisances.
    """
    from desilike.profilers import MinuitProfiler

    from .fitting import maximize_usable

    size = len(common["separations_mpc_h"]) * len(common["multipoles"])
    theory, observable, likelihood, _ = build_likelihood(np.zeros(size), **common)
    pt = theory.pt
    pt.damping_override = (
        damping(np.asarray(pt.k, dtype=float), np.asarray(pt.mu, dtype=float))
        if callable(damping)
        else np.asarray(damping, dtype=float)
    )
    if pt.damping_override.shape != (np.size(pt.k), np.size(pt.mu)):
        raise ValueError("predicted damping does not cover the template (k, mu) grid")
    patterns = list(config["fitting"]["linear_nuisance"]["parameter_patterns"])
    nuisances = {str(p.name): 0.0 for p in likelihood.all_params.select(basename=patterns)}
    _fix_parameters(likelihood, {**prediction_values, **nuisances})
    pt.runtime_info.tocalculate = True
    likelihood()
    prediction = np.array(observable.flattheory, dtype=float)
    if prediction.shape != (size,) or not np.isfinite(prediction).all():
        raise ValueError("predicted transfer multipoles are not finite")
    dilations = {
        name: prediction_values[name]
        for name in config["fitting"]["ap_interval_parameters"][common["ap_mode"]]
    }
    _, fit_observable, fit_likelihood, _ = build_likelihood(prediction, **common)
    _fix_parameters(fit_likelihood, {**dilations, "sigmas": streaming})
    profiler = MinuitProfiler(
        fit_likelihood, seed=int(config["fitting"]["optimization"]["random_seed"])
    )
    best = maximize_usable(profiler, config)
    fitted = {name: float(best.bestfit[name][0]) for name in ("sigmaper", "sigmapar", "b1")}
    errors = {}
    for name in fitted:
        try:
            errors[name] = float(best.error[name][0])
        except (AttributeError, KeyError, IndexError, TypeError):
            errors[name] = None
    # Pin the profiled broadband at its best fit so the residual model includes it.
    free = [str(p.name) for p in fit_likelihood.all_params.select(basename=patterns) if not p.fixed]
    if not free or not set(free) <= set(map(str, best.bestfit.names())):
        raise RuntimeError("compression profile does not report every free broadband nuisance")
    _fix_parameters(fit_likelihood, {name: float(best.bestfit[name][0]) for name in free})
    fit_likelihood(**fitted)
    model = np.array(fit_observable.flattheory, dtype=float)
    summary = {
        "damping_grid": (np.asarray(pt.k, dtype=float), np.asarray(pt.mu, dtype=float)),
        "damping": np.asarray(pt.damping_override),
        "fitted": fitted,
        "curvature_errors": errors,
        # Solved nuisances add a trailing derivatives axis; column 0 is the solution's value.
        "loglikelihood": float(
            np.asarray(best.bestfit["loglikelihood"], dtype=float).reshape(1, -1)[0, 0]
        ),
        "optimizer_starts": best.attrs.get("optimizer_starts"),
    }
    return prediction, model, summary


def transfer_compression_setup(run, route, *, covariance_run=None, damping_options=None):
    """Inputs shared by a transfer route's prediction and compression, built in one place.

    Used by transfer_centre_record and by the centres review diagnostics, so both see the same
    covariance, window, kernel inputs and template prior.
    """
    from .covariance import likelihood_covariance_run, read_route_covariance
    from .selection import read_preparation, tier_spec
    from .theory import transfer_damping_context, transfer_damping_factor, transfer_damping_inputs

    config = run["config"]
    transfer = config["measurement"]["reconstruction"].get("transfer_run")
    if transfer is None or route["reconstruction_state"] != "postreconstruction":
        raise ValueError("damping centres are post-reconstruction transfer quantities")
    tracer, name, condition = route["tracer"], route["measurement_bin"], route["condition"]
    source = run.get("source_run", run)
    preparation = read_preparation(source, tracer)
    context = transfer_damping_context(
        config, preparation, tracer=tracer, measurement_bin=name, target_class=condition
    )
    cosmology = build_mock_cosmology(config)
    inputs = transfer_damping_inputs(context, config, cosmology=cosmology)
    holder = covariance_run if covariance_run is not None else likelihood_covariance_run(run, route)
    saved = read_route_covariance(holder, route)
    arrays = saved["arrays"]
    zeff = float(context["target_redshift"]["z_eff"])
    template = resolve_damping_prior(
        tracer,
        "prereconstruction",
        sigma_z_over_one_plus_z=tier_spec(condition, config)["sigma_z_over_one_plus_z"],
        z_eff=zeff,
        e_z_eff=float(cosmology.efunc(zeff)),
        config=config,
    )
    damping_names = ("sigma_perp", "sigma_parallel", "sigma_streaming")
    prior = {
        **template,
        "reconstruction_state": "postreconstruction",
        "unpriored_parameters": list(damping_names),
        "bounds_mpc_h": {key: list(TRANSFER_CENTRE_BOUNDS_MPC_H) for key in damping_names},
        "scope": "transfer_centre_compression",
    }
    streaming = float(
        config["fitting"]["damping"]["baseline_centers_mpc_h"][tracer]["prereconstruction"][
            "sigma_streaming"
        ]
    )
    surface = config["fitting"]["fit_surfaces"][tracer]
    sample_identity = "|".join((name, condition, "postreconstruction"))
    common = dict(
        covariance=arrays["mean_covariance"],
        separations_mpc_h=arrays["separations"].mean(axis=0),
        fit_sedges=arrays["fit_edges"],
        z_eff=zeff,
        multipoles=arrays["multipoles"],
        reconstruction_state="postreconstruction",
        reconstruction_convention=config["measurement"]["reconstruction"]["allowed_conventions"][0],
        smoothing_mpc_h=config["measurement"]["reconstruction"]["smoothing_values_mpc_h"][0],
        ap_mode=surface["ap_mode"],
        method="profiling",
        damping_prior=prior,
        geometry_artifact={**saved["window"], **route, "sample_identity": sample_identity},
        sample_identity=sample_identity,
        tracer=tracer,
        measurement_bin=name,
        config=config,
    )
    dilations = {
        key: 1.0 for key in config["fitting"]["ap_interval_parameters"][surface["ap_mode"]]
    }
    return {
        "config": config,
        "context": context,
        "cosmology": cosmology,
        "inputs": inputs,
        "holder": holder,
        "arrays": arrays,
        "common": common,
        "streaming": streaming,
        "prediction_values": {**dilations, "b1": context["bias"], "sigmas": streaming},
        "damping": lambda k, mu: transfer_damping_factor(
            k, mu, **inputs, **(damping_options or {})
        ),
    }


def transfer_centre_record(run, route, *, covariance_run=None, damping_options=None):
    """Predict one transfer route's windowed multipoles from the source-target kernel and fit
    the native template's two nonlinear damping scales to that prediction.

    Dilations stay at one, streaming at its configured centre and the photo-z response in the
    template, so the prediction differs from the template only in its wiggle damping
    (theory.transfer_damping_factor). No recovery mock enters. ``covariance_run`` supplies a
    labelled proxy covariance and window before the run's own covariance exists.
    """
    from .theory import TRANSFER_DAMPING_APPROXIMATIONS

    setup = transfer_compression_setup(
        run, route, covariance_run=covariance_run, damping_options=damping_options
    )
    config, context, cosmology = setup["config"], setup["context"], setup["cosmology"]
    inputs, holder, arrays, streaming = (
        setup["inputs"],
        setup["holder"],
        setup["arrays"],
        setup["streaming"],
    )
    tracer, condition = route["tracer"], route["condition"]
    zeff = float(context["target_redshift"]["z_eff"])
    prediction, model, summary = _predict_and_compress(
        setup["common"],
        setup["prediction_values"],
        damping=setup["damping"],
        streaming=streaming,
        config=config,
    )
    fitted, errors = summary["fitted"], summary["curvature_errors"]
    wave, angle = summary["damping_grid"]
    response = np.exp(-((np.outer(wave, angle) * inputs["target_sigma_r_mpc_h"]) ** 2))
    above = summary["damping"] > 1.0
    residual = prediction - model
    covariance = np.asarray(arrays["mean_covariance"], dtype=float)
    normalized = (residual / np.sqrt(np.diag(covariance))).reshape(len(arrays["multipoles"]), -1)
    centres = {
        "sigma_perp": fitted["sigmaper"],
        "sigma_parallel": fitted["sigmapar"],
        "sigma_streaming": streaming,
    }
    lower, upper = TRANSFER_CENTRE_BOUNDS_MPC_H
    diagnostics = {
        "compression_chi2": float(residual @ np.linalg.solve(covariance, residual)),
        "compression_minus_two_loglikelihood": -2.0 * summary["loglikelihood"],
        "data_points": int(residual.size),
        "rms_residual_over_sigma_by_multipole": {
            f"ell{int(ell)}": float(np.sqrt(np.mean(row**2)))
            for ell, row in zip(arrays["multipoles"], normalized)
        },
        "curvature_errors_mpc_h": {
            "sigma_perp": errors["sigmaper"],
            "sigma_parallel": errors["sigmapar"],
        },
        "fitted_b1": fitted["b1"],
        # Where target errors far exceed the radius, pairs with nearly equal errors keep
        # their source correlation and the wiggle decays more slowly than the response;
        # a Gaussian template cannot follow it, so it shows here and in the chi-square.
        "damping_above_one": {
            "fraction_of_template_modes": float(above.mean()),
            "max_damping": float(summary["damping"].max()),
            "max_response_times_damping": (
                float((response * summary["damping"])[above].max()) if above.any() else 0.0
            ),
        },
        "bounds_mpc_h": list(TRANSFER_CENTRE_BOUNDS_MPC_H),
        "railed": sorted(
            key
            for key in ("sigma_perp", "sigma_parallel")
            if not lower + 1e-3 < centres[key] < upper - 1e-3
        ),
        "covariance_source": {
            **{
                key: holder["metadata"][key]
                for key in ("run_name", "code_commit", "analysis_config_sha256")
            },
            "proxy": covariance_run is not None,
            "scaling": "mean covariance (single-realization covariance / 25)",
        },
        "damping_inputs": {key: value for key, value in inputs.items() if key != "linear_power"},
        "optimizer_starts": summary["optimizer_starts"],
    }
    if context["source_class"] == condition:
        density = np.mean([float(v) for v in context["density_h3_mpc3"]["source"].values()])
        interpolator = cosmology.get_fourier().pk_interpolator(of="delta_m")
        analytic = analytic_postreconstruction_sigma_perp(
            tracer=tracer,
            attained_density_h3_mpc3=density,
            linear_power=lambda k: interpolator(k, z=zeff),
            drag_scale_mpc_h=float(cosmology.rs_drag),
            config=config,
        )
        # Reported beside the compressed value, never compared for equality: the analytic
        # centre is the historical self-field formula, not this kernel.
        diagnostics["self_field_analytic_sigma_perp_mpc_h"] = float(
            analytic["analytic_center_mpc_h"]
        )
    return {
        "context": context,
        "centers_mpc_h": centres,
        "evidence": {
            "kind": "source_target_theory",
            "reference": TRANSFER_CENTRE_DECISION,
            "approximations": list(TRANSFER_DAMPING_APPROXIMATIONS),
            "diagnostics": _plain(diagnostics),
        },
    }


def _plain(value):
    """YAML-safe copy: numpy scalars to Python numbers, tuples to lists."""
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def transfer_centre_artifact(
    run, routes, *, covariance_run=None, tracer=None, damping_options=None
):
    """Candidate centre artifact for every post-reconstruction route of a transfer run.

    ``routes`` are the run's planned routes, supplied by the driver. ``review_reference``
    stays null: the transfer resolver refuses the artifact until someone reviews the
    records and adds the reference, which freezes the centres.
    """
    records = [
        transfer_centre_record(
            run, route, covariance_run=covariance_run, damping_options=damping_options
        )
        for route in routes
        if route["reconstruction_state"] == "postreconstruction"
        and (tracer is None or route["tracer"] == tracer)
    ]
    if not records:
        raise ValueError("no post-reconstruction route matches the requested transfer scope")
    return {
        "units": "Mpc/h",
        "review_reference": None,
        "producer": {
            key: run["metadata"][key]
            for key in ("run_name", "code_commit", "analysis_config_sha256")
        },
        "decision": TRANSFER_CENTRE_DECISION,
        "records": records,
    }
