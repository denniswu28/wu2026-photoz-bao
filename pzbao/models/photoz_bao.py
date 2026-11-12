"""Pinned-desilike BAO power model with a fixed observed-coordinate photo-z response.

This module changes one well-defined part of desilike's damped-wiggle model:
the known Gaussian redshift-error response is evaluated at observed/fiducial
``k, mu`` and applied outside the AP-transformed nonlinear damping.

**The response multiplies the complete smooth-plus-wiggle signal.** A redshift
error displaces a galaxy along the line of sight regardless of which Fourier
component put it where it was, so the observational kernel acts on the whole
clustering signal and not on the acoustic feature alone. The BAO dilation
parameters remain applied to the oscillatory component only; that is a separate
statement about the *model*, and it is unchanged. The wiggle-only alternative is
deleted rather than retained as a scope, so no fit can run it by accident.
"""

from __future__ import annotations

import numpy as np


def gaussian_photoz_response(k, mu, sigma_r_mpc_h):
    """Return ``exp[-(k mu sigma_r)^2]`` for one-object radial scatter."""
    wave = np.asarray(k, dtype=float)
    angle = np.asarray(mu, dtype=float)
    sigma = float(sigma_r_mpc_h)
    if sigma < 0 or not np.isfinite(sigma):
        raise ValueError("photo-z radial scatter must be finite and non-negative")
    return np.exp(-((wave * angle * sigma) ** 2))


def build_observed_photoz_power_model(
    *,
    template,
    mode: str,
    smoothing_radius: float,
    pairwise_sigma_mpc_h: float,
    mu_quadrature_order: int,
):
    """Build the desilike calculator lazily so light tests need no heavy imports."""
    from desilike.jax import numpy as jnp
    from desilike.theories.galaxy_clustering.bao import (
        DampedBAOWigglesPowerSpectrumMultipoles,
        _interp,
    )

    pairwise_sigma = float(pairwise_sigma_mpc_h)
    if pairwise_sigma < 0 or not np.isfinite(pairwise_sigma):
        raise ValueError("pairwise photo-z scale must be finite and non-negative")
    order = int(mu_quadrature_order)
    if order < 16:
        raise ValueError("photo-z mu quadrature requires at least sixteen points")

    class ObservedPhotoZPowerSpectrumMultipoles(DampedBAOWigglesPowerSpectrumMultipoles):
        """One pinned model specialization with no new varied parameter."""

        # Set only by the transfer damping-centre producer: an array on (k, mu) replacing the
        # Gaussian wiggle damping with a predicted source-target damping. Fits never set it.
        damping_override = None

        def initialize(self, *args, **kwargs):
            super().initialize(*args, mu=order, method="leggauss", **kwargs)
            self.photoz_pairwise_sigma_mpc_h = pairwise_sigma

        # Adapted from desilike's DampedBAOWigglesPowerSpectrumMultipoles.calculate, standard
        # branch (commit 41f082f); only the photo-z response is new. BSD-3-Clause, Copyright (c)
        # 2021 cosmodesi; see LICENSES/desilike-BSD-3-Clause.txt.
        def calculate(self, b1=1.0, dbeta=1.0, sigmas=0.0, sigmapar=9.0, sigmaper=6.0):
            # This follows the pinned desilike 1.0.0 standard model exactly,
            # except for the explicitly marked observed-coordinate response.
            super(DampedBAOWigglesPowerSpectrumMultipoles, self).calculate()
            f = dbeta * self.template.f
            _, kap, muap = self.template.ap_k_mu(self.k, self.mu)
            pknowap = _interp(self.template, "pknow_dd", kap)
            pkap = _interp(self.template, "pk_dd", kap)
            k, mu = self.k[:, None], self.mu
            pkwap = pkap - pknowap
            sigma_nl2ap = kap**2 * (sigmapar**2 * muap**2 + sigmaper**2 * (1.0 - muap**2))
            sk = 0.0
            if self.mode == "reciso":
                sk = jnp.exp(-0.5 * (k * self.smoothing_radius) ** 2)
            damping = (
                jnp.exp(-sigma_nl2ap / 2.0)
                if self.damping_override is None
                else jnp.asarray(self.damping_override)
            )
            cap = (b1 + f * muap**2 * (1.0 - sk)) ** 2 * damping
            fog = 1.0 / (1.0 + (sigmas * k * mu) ** 2 / 2.0) ** 2
            smooth_amplitude = (b1 + f * mu**2 * (1.0 - sk)) ** 2 * fog
            pknow = _interp(self.template, "pknow_dd", k)
            # pairwise_sigma = sqrt(2) * sigma_r, hence the 1/2 exponent.
            response = jnp.exp(-0.5 * (k * mu * self.photoz_pairwise_sigma_mpc_h) ** 2)
            smooth = smooth_amplitude * pknow
            wiggle = cap * pkwap
            # The kernel multiplies the full signal: smooth broadband and acoustic wiggle alike.
            pkmu = response * (smooth + wiggle)
            self.power = self.to_poles(pkmu)

    return ObservedPhotoZPowerSpectrumMultipoles(
        template=template,
        mode=mode,
        smoothing_radius=float(smoothing_radius),
    )


__all__ = ["build_observed_photoz_power_model", "gaussian_photoz_response"]
