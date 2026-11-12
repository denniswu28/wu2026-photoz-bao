"""Numerical angular quadrature and multipole comparison primitives.

Production uses the configured fixed rule. Explicit real-model comparisons live in the photo-z
diagnostics module and do not promote orders or provide runtime certificates.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


# Adapted from desilike's utils.weights_leggauss(nx, sym=True) (commit 41f082f).
# BSD-3-Clause, Copyright (c) 2021 cosmodesi; see LICENSES/desilike-BSD-3-Clause.txt.
def symmetric_leggauss(order: int):
    """Return the half-range Gauss-Legendre rule the fitting library actually uses.

    An order of ``n`` means ``n`` strictly positive nodes taken from a ``2n``-point rule, with
    the weights symmetrised about zero and normalised so they sum to one. That is not the same
    as an ``n``-point rule on the half range, and reproducing the library's convention exactly
    matters: a check built on a different rule would certify something the model never
    evaluates. Verified node for node and weight for weight against the installed library.
    """
    if isinstance(order, (bool, np.bool_)) or not isinstance(order, (int, np.integer)):
        raise ValueError("quadrature order must be a positive integer")
    count = int(order)
    if count < 1:
        raise ValueError("quadrature order must be a positive integer")
    nodes, weights = np.polynomial.legendre.leggauss(2 * count)
    return nodes[count:], (weights[count:] + weights[count - 1 :: -1]) / 2.0


def _legendre(order: int, mu: np.ndarray) -> np.ndarray:
    coefficients = np.zeros(order + 1)
    coefficients[order] = 1.0
    return np.polynomial.legendre.legval(mu, coefficients)


def photoz_suppression(k: np.ndarray, mu: np.ndarray, pairwise_sigma_mpc_h: float) -> np.ndarray:
    """Return the observed-coordinate photometric suppression on a wavenumber-angle grid.

    The exponent carries a half because the scale supplied is the *pairwise* one, larger than
    the one-galaxy radial blur by the square root of two, which is the same convention the
    fitted model uses.
    """
    scale = float(pairwise_sigma_mpc_h)
    if not np.isfinite(scale) or scale < 0:
        raise ValueError("pairwise photometric scale must be finite and non-negative")
    return np.exp(-0.5 * (k[:, None] * mu[None, :] * scale) ** 2)


def project_multipoles(
    k: np.ndarray,
    pk_smooth: np.ndarray,
    pk_wiggle: np.ndarray,
    *,
    pairwise_sigma_mpc_h: float,
    multipoles: Sequence[int],
    order: int,
    growth_rate: float,
    linear_bias: float,
    sigma_par_mpc_h: float,
    sigma_perp_mpc_h: float,
) -> np.ndarray:
    """Project the observed model onto multipoles at one quadrature order.

    The angular structure reproduces the fitted model at unit dilation: a Kaiser amplitude on
    both components, anisotropic non-linear damping on the oscillatory one, and the
    photometric suppression applied to the complete smooth-plus-wiggle signal.
    """
    wavenumber = np.asarray(k, dtype=float)
    smooth = np.asarray(pk_smooth, dtype=float)
    wiggle = np.asarray(pk_wiggle, dtype=float)
    if not (wavenumber.shape == smooth.shape == wiggle.shape) or wavenumber.ndim != 1:
        raise ValueError("quadrature inputs must share one wavenumber axis")
    mu, weights = symmetric_leggauss(order)
    kaiser = (linear_bias + growth_rate * mu[None, :] ** 2) ** 2
    damping = np.exp(
        -0.5
        * wavenumber[:, None] ** 2
        * (sigma_par_mpc_h**2 * mu[None, :] ** 2 + sigma_perp_mpc_h**2 * (1.0 - mu[None, :] ** 2))
    )
    response = photoz_suppression(wavenumber, mu, pairwise_sigma_mpc_h)
    smooth_term = kaiser * smooth[:, None]
    wiggle_term = kaiser * damping * wiggle[:, None]
    pkmu = response * (smooth_term + wiggle_term)
    poles = []
    for degree in multipoles:
        legendre = _legendre(int(degree), mu)
        poles.append((2 * int(degree) + 1) * (pkmu * (weights * legendre)[None, :]).sum(axis=1))
    return np.asarray(poles)


def compare_multipole_orders(
    produced: np.ndarray,
    checked: np.ndarray,
    *,
    multipoles: Sequence[int],
    maximum_fractional_difference: float,
) -> dict:
    """Compare two orders' multipoles under the one declared convergence criterion.

    The comparison is per wavenumber, normalised by the monopole at that same wavenumber. The
    normalisation matters more than it looks. Normalising instead by each multipole's peak over
    the whole range would divide the error by the power at the largest scales, which is orders
    of magnitude larger than the power where the angular spike is narrow, and would report a
    converged result for a rule that is visibly wrong at the top of the range. Normalising by
    each multipole's own value at each wavenumber is unstable wherever a higher multipole
    crosses zero. The monopole is positive everywhere for this model and is the natural measure
    of total power at a wavenumber, so it is the stable denominator that still exposes the
    error where it actually lives.

    Offline studies share this comparison; production fitting never calls it.
    """
    poles = [int(value) for value in multipoles]
    if 0 not in poles:
        raise ValueError("the quadrature check needs the monopole as its normalisation")
    low = np.asarray(produced, dtype=float)
    high = np.asarray(checked, dtype=float)
    if low.shape != high.shape or low.ndim != 2 or low.shape[0] != len(poles):
        raise ValueError("quadrature comparison needs one multipole row per declared multipole")
    if (
        not np.all(np.isfinite(low))
        or not np.all(np.isfinite(high))
        or not low.size
        or not np.isfinite(maximum_fractional_difference)
        or maximum_fractional_difference <= 0
    ):
        raise ValueError(
            "quadrature comparison requires finite non-empty arrays and a positive tolerance"
        )
    monopole = np.abs(high[poles.index(0)])
    if not np.all(monopole > 0.0):
        raise ValueError("the quadrature check requires a positive monopole to normalise by")
    fractional = np.max(np.abs(low - high) / monopole[None, :], axis=1)
    return {
        "multipoles": poles,
        "fractional_difference_by_multipole": [float(value) for value in fractional],
        "worst_fractional_difference": float(np.max(fractional)),
        "maximum_fractional_difference": float(maximum_fractional_difference),
        "within_tolerance": bool(np.max(fractional) <= float(maximum_fractional_difference)),
    }
