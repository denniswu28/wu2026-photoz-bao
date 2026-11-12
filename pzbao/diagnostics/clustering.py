"""Reusable clustering diagnostics. Inputs and experiment choices are explicit arguments."""

from __future__ import annotations

import numpy as np

from pzbao.likelihood import xi_fit_data

_clustering_epoch_comparison_ELLS = (0, 2)


def common_ids(*groups):
    """Intersect explicitly identified ensembles, never align by row position."""
    for ids in groups:
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate realization identifier")
    result = sorted(set(groups[0]).intersection(*map(set, groups[1:])))
    if len(result) < 2:
        raise ValueError("at least two common realizations are required")
    return result


def ensemble_statistics(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 3 or len(values) < 2 or (not np.isfinite(values).all()):
        raise ValueError("expected finite mock x multipole x separation vectors")
    sd = values.std(axis=0, ddof=1)
    return {"mean": values.mean(axis=0), "sd": sd, "nominal_sem": sd / np.sqrt(len(values))}


def safe_ratio(numerator, denominator):
    a, b = np.broadcast_arrays(np.asarray(numerator, float), np.asarray(denominator, float))
    result = np.full(a.shape, np.nan)
    np.divide(a, b, out=result, where=np.isfinite(a) & np.isfinite(b) & (b > 0))
    return result


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


def count_vector(estimator, edges, ells=_clustering_epoch_comparison_ELLS):
    """Use the scientific owner's exact count rebinning and edge selection."""
    width = float(edges[1] - edges[0])
    source_widths = np.diff(estimator.edges[0])
    factor = width / source_widths[0]
    if (
        not np.allclose(source_widths, source_widths[0], rtol=0, atol=1e-10)
        or not np.isclose(factor, round(factor), rtol=0, atol=1e-10)
        or factor < 1
    ):
        raise ValueError("native counts do not exactly coarsen to the comparison grid")
    separation, vector, actual_edges = xi_fit_data(
        estimator,
        multipoles=ells,
        fit_edge_limits_mpc_h=[edges[0], edges[-1]],
        rebin_factor=int(round(factor)),
        expected_bin_width_mpc_h=width,
    )
    np.testing.assert_allclose(actual_edges, edges, rtol=0, atol=1e-10)
    return (separation, vector.reshape(len(ells), len(edges) - 1))


def summarize_estimators(estimators, ids, edges):
    total, vectors, separations = (None, [], [])
    for mid in ids:
        estimator = estimators[mid]
        sep, vector = count_vector(estimator, edges)
        vectors.append(vector)
        separations.append(sep)
        total = estimator.copy() if total is None else total + estimator
    pooled_sep, pooled = count_vector(total, edges)
    values = np.asarray(vectors)
    return {
        "ids": ids,
        "vectors": values,
        "separations": np.asarray(separations),
        "pooled": pooled,
        "pooled_separation": pooled_sep,
        **ensemble_statistics(values),
    }


def decode_historical_observable(observable, edges):
    """Decode saved arrays only. Never initialize or evaluate a likelihood."""
    poles = tuple((int(x) for x in observable["ells"]))
    if (
        not poles
        or len(set(poles)) != len(poles)
        or (not set(poles).issubset(_clustering_epoch_comparison_ELLS))
    ):
        raise ValueError("historical multipoles are not an identified comparison subset")
    saved_edges = np.asarray(observable["sedges"], dtype=float)
    if saved_edges.ndim == 1:
        saved_edges = np.tile(saved_edges, (len(poles), 1))
    if saved_edges.shape != (len(poles), len(edges)):
        raise ValueError("historical fitted edge dimensions differ")
    np.testing.assert_allclose(saved_edges, np.tile(edges, (len(poles), 1)), rtol=0, atol=1e-08)
    data = np.asarray(observable["data"], float).reshape(len(poles), len(edges) - 1)
    separation = np.asarray(observable["s"], float).reshape(len(poles), len(edges) - 1)
    if (
        not np.isfinite(data).all()
        or not np.isfinite(separation).all()
        or np.any(separation < edges[:-1])
        or np.any(separation > edges[1:])
    ):
        raise ValueError("historical data/separations are invalid")
    matrix = np.asarray(observable["covariance"], float)
    if matrix.shape != (data.size, data.size):
        raise ValueError("historical covariance dimension differs from its vector")
    covariance_correlation(matrix)
    np.testing.assert_allclose(
        np.asarray(observable["std"]).ravel(), np.sqrt(np.diag(matrix)), rtol=1e-09, atol=1e-12
    )
    return (poles, data, matrix)


def compare_arrays(current, dense, ids, edges):
    current, dense = (np.asarray(current), np.asarray(dense))
    if current.shape != dense.shape or current.shape != (25, 2, len(edges) - 1):
        raise ValueError("expected 25 paired mocks, two multipoles, common separation grid")
    np.testing.assert_array_equal(ids, np.arange(25))
    if not np.isfinite(current).all() or not np.isfinite(dense).all():
        raise ValueError("nonfinite measured vector")
    if np.any(np.diff(edges) <= 0):
        raise ValueError("unordered separation grid")
    delta = dense - current
    base_sd, sd = (current.std(axis=0, ddof=1), dense.std(axis=0, ddof=1))
    ratios = np.divide(sd, base_sd, out=np.full_like(sd, np.nan), where=base_sd > 0)
    return {
        "current_mean": current.mean(axis=0),
        "dense_mean": dense.mean(axis=0),
        "current_sd": base_sd,
        "dense_sd": sd,
        "sd_ratio": [[float(v) if np.isfinite(v) else None for v in pole] for pole in ratios],
        "zero_denominator": base_sd == 0,
        "paired_differences": delta,
        "paired_mean": delta.mean(axis=0),
        "paired_sd": delta.std(axis=0, ddof=1),
        "paired_nominal_sem": delta.std(axis=0, ddof=1) / np.sqrt(len(ids)),
    }


def plot_multipoles(coordinate, values, multipoles, *, ax=None, statistic="xi"):
    """Plot explicit multipole axes; no normalization or estimator is changed."""
    from ..figures.style import axes

    x, values = np.asarray(coordinate), np.asarray(values)
    if statistic not in {"xi", "pk"} or values.shape != (len(multipoles), len(x)):
        raise ValueError("statistic or multipole axes do not match")
    ax = axes(ax)
    for ell, vector in zip(multipoles, values):
        ax.plot(x, vector, label=f"ell={ell}")
    ax.set(
        xlabel="s [Mpc/h]" if statistic == "xi" else "k [h/Mpc]",
        ylabel="xi" if statistic == "xi" else "P(k) [(Mpc/h)^3]",
    )
    return ax, {"coordinate": x, "values": values, "multipoles": list(multipoles)}
