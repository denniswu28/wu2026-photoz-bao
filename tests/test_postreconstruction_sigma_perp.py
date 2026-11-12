"""The analytic post-reconstruction transverse damping centre, and the direct prior a fit consumes.

Two effects damp the acoustic feature after reconstruction. The modes reconstruction cannot restore
survive weighted by ``[1 - S(k)]^2``, and the displacement field is itself estimated from a finite
sample, so the displacement applied is noisy. The first depends on the smoothing radius and the
linear power, the second on ``b^2 nbar`` -- so the centre is a property of each *sample* rather than
of each tracer, and this study's dense low-redshift bin and sparse high-redshift bin arrive at
nearly the same centre by opposite routes.

Only the centre is computed. The prescription states the Gaussian prior widths once for the whole
centre table and replaces only the post-reconstruction transverse centre analytically, so the fixed
triple is the width authority in both reconstruction states, the artifact carries no width, and no
quantity here is derived from the realizations the prior will later weight.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from pzbao.config import load_analysis_config
from pzbao.theory import (
    POSTRECONSTRUCTION_SHOT_NOISE_P_SHOT,
    POSTRECONSTRUCTION_SIGMA_PERP_FORMULA,
    SIGMA_REC_QUADRATURE,
    analytic_postreconstruction_sigma_perp,
    approximate_shot_noise_transverse_damping_squared,
    reconstruction_transverse_damping_squared,
    resolve_damping_prior,
    shot_noise_transverse_damping_squared,
)

DRAG = 99.084426793444880
BGS_DENSITY = 4.472717e-04
LRG_DENSITY = 3.159967e-05
BGS_Z_EFF = 0.1605703245287798
LRG_Z_EFF = 0.8718010000000000
SHOT = dict(
    bias=1.5,
    attained_density_h3_mpc3=BGS_DENSITY,
    smoothing_radius_mpc_h=15.0,
    drag_scale_mpc_h=DRAG,
)


def _power(amplitude=1.5e6, turnover=0.05):
    """A linear-power stand-in with the right shape: ``~k`` at low ``k``, ``~k^-3`` at high ``k``.

    Used for the tests that check the integral's *behaviour*, where only the shape matters. The
    tests that check numbers use the configured cosmology itself.
    """

    def evaluate(k):
        k = np.asarray(k, dtype=float)
        return amplitude * k / (1.0 + (k / turnover) ** 4)

    return evaluate


@pytest.fixture(scope="module")
def linear_power():
    """The configured mock cosmology's linear power, and its drag scale.

    Returns a function of redshift giving ``P_lin(k)``, so each anchor is evaluated at its own
    frozen effective redshift rather than at a shared one.
    """
    pytest.importorskip("cosmoprimo")
    from pzbao.catalog import build_mock_cosmology

    cosmology = build_mock_cosmology(load_analysis_config())
    interpolator = cosmology.get_fourier().pk_interpolator(of="delta_m")
    return lambda z: (lambda k: interpolator(k, z=z)), float(cosmology.rs_drag)


def _numerical_shot_integral(radius, drag, points=2_000_001):
    """The shot-noise integral by quadrature, checking the closed form the module uses."""
    k = np.linspace(1e-12, 40.0 / radius, points)
    j0 = np.sin(k * drag) / (k * drag)
    return float(np.trapz(np.exp(-(k**2) * radius**2) * (1.0 - j0), k) / (2.0 * math.pi**2))


# ------------------------------------------------------------------ the two terms of the centre


def test_the_shot_noise_closed_form_is_the_integral_the_prescription_writes():
    """Exactness is why a closed form is used at all, so it is checked against quadrature."""
    for radius, drag in ((15.0, DRAG), (10.0, DRAG), (25.0, 120.0)):
        closed = (1.0 / (2.0 * math.pi**2)) * (
            math.sqrt(math.pi) / (2.0 * radius)
            - (math.pi / (2.0 * drag)) * math.erf(drag / (2.0 * radius))
        )
        assert closed == pytest.approx(_numerical_shot_integral(radius, drag), rel=1e-6)

    # And it is reproducible: same inputs, same value, exactly.
    first = shot_noise_transverse_damping_squared(**SHOT)
    assert shot_noise_transverse_damping_squared(**SHOT) == first


def test_both_terms_are_positive_and_refuse_impossible_inputs():
    assert shot_noise_transverse_damping_squared(**SHOT) > 0
    assert (
        reconstruction_transverse_damping_squared(
            linear_power=_power(), smoothing_radius_mpc_h=15.0, drag_scale_mpc_h=DRAG
        )
        > 0
    )
    for field in ("bias", "attained_density_h3_mpc3", "smoothing_radius_mpc_h", "drag_scale_mpc_h"):
        for bad in (0.0, -1.0, float("nan")):
            with pytest.raises(ValueError, match="positive finite"):
                shot_noise_transverse_damping_squared(**{**SHOT, field: bad})
    with pytest.raises(ValueError, match="positive finite radius and drag"):
        reconstruction_transverse_damping_squared(
            linear_power=_power(), smoothing_radius_mpc_h=0.0, drag_scale_mpc_h=DRAG
        )
    with pytest.raises(ValueError, match="finite and non-negative"):
        reconstruction_transverse_damping_squared(
            linear_power=lambda k: -np.ones_like(np.asarray(k, dtype=float)),
            smoothing_radius_mpc_h=15.0,
            drag_scale_mpc_h=DRAG,
        )


@pytest.mark.parametrize(
    "field, larger",
    [
        ("attained_density_h3_mpc3", 10.0 * BGS_DENSITY),
        ("bias", 3.0),
        ("smoothing_radius_mpc_h", 30.0),
    ],
)
def test_a_better_measured_displacement_field_carries_less_shot_noise_damping(field, larger):
    """Denser samples, more biased tracers and coarser smoothing all measure the field better.

    A sign error in any of the three would otherwise sit undetected behind a plausible centre.
    """
    assert shot_noise_transverse_damping_squared(**{**SHOT, field: larger}) < (
        shot_noise_transverse_damping_squared(**SHOT)
    )


def test_the_two_terms_move_oppositely_in_the_smoothing_radius():
    """That opposition is what makes the sum a real trade-off rather than a single knob.

    A larger radius reconstructs less of the field, so the unrestored part grows -- while the
    displacement it does estimate is measured over more objects and is less noisy.
    """
    coarse = reconstruction_transverse_damping_squared(
        linear_power=_power(), smoothing_radius_mpc_h=30.0, drag_scale_mpc_h=DRAG
    )
    fine = reconstruction_transverse_damping_squared(
        linear_power=_power(), smoothing_radius_mpc_h=15.0, drag_scale_mpc_h=DRAG
    )
    assert coarse > fine
    assert shot_noise_transverse_damping_squared(
        **{**SHOT, "smoothing_radius_mpc_h": 30.0}
    ) < shot_noise_transverse_damping_squared(**SHOT)


def test_the_reconstruction_term_is_linear_in_the_linear_power():
    """It is an integral of ``P_lin`` against a fixed kernel, so growth enters through it alone."""
    doubled = reconstruction_transverse_damping_squared(
        linear_power=_power(amplitude=3.0e6), smoothing_radius_mpc_h=15.0, drag_scale_mpc_h=DRAG
    )
    single = reconstruction_transverse_damping_squared(
        linear_power=_power(amplitude=1.5e6), smoothing_radius_mpc_h=15.0, drag_scale_mpc_h=DRAG
    )
    assert doubled == pytest.approx(2.0 * single)


def test_the_quadrature_is_converged_far_below_the_comparison_scale(monkeypatch, linear_power):
    """The prior width is 1 Mpc/h; the grid must be nowhere near deciding anything.

    Measured on the real integrand rather than asserted, because "converged" is otherwise a claim
    about a choice nobody checked. Refining the grid eightfold or extending it fivefold each move
    the centre by less than a hundredth of a Mpc/h -- two orders below the scale that matters.
    """
    import pzbao.theory as fitting_theory

    power, drag = linear_power
    at_z = power(BGS_Z_EFF)

    def sigma(**overrides):
        monkeypatch.setattr(
            fitting_theory, "SIGMA_REC_QUADRATURE", dict(SIGMA_REC_QUADRATURE, **overrides)
        )
        return math.sqrt(
            fitting_theory.reconstruction_transverse_damping_squared(
                linear_power=at_z, smoothing_radius_mpc_h=15.0, drag_scale_mpc_h=drag
            )
        )

    reference = sigma()
    assert reference > 1.0  # the term is not incidentally zero, which would make this vacuous
    for points in (1024, 2048, 8192, 32768):
        assert abs(sigma(points=points) - reference) < 0.01
    for k_max in (5.0, 20.0, 50.0):
        assert abs(sigma(k_max_h_mpc=k_max) - reference) < 0.05
    for k_min in (1e-3, 1e-5):
        assert abs(sigma(k_min_h_mpc=k_min) - reference) < 0.01
    # Deterministic: the same controls give the same number, every time.
    assert sigma() == reference


# ------------------------------------------------------------------------------ the centre itself


def _analytic(tracer, density, power, drag, *, config=None):
    return analytic_postreconstruction_sigma_perp(
        tracer=tracer,
        attained_density_h3_mpc3=density,
        linear_power=power,
        drag_scale_mpc_h=drag,
        config=config or load_analysis_config(),
    )


def test_the_centre_is_the_two_theory_terms_and_not_the_configured_baseline(linear_power):
    """Adding the full shot term to the DESI-informed baseline would double count.

    That baseline may already carry typical survey shot-noise and nonlinear contributions, so here
    it is start state and a diagnostic rather than a term of the centre.
    """
    power, drag = linear_power
    record = _analytic("BGS_BRIGHT-21.5", BGS_DENSITY, power(BGS_Z_EFF), drag)
    assert record["analytic_center_mpc_h"] == pytest.approx(
        math.sqrt(record["reconstruction_variance_mpc_h2"] + record["shot_noise_variance_mpc_h2"])
    )
    # Explicitly not the superseded combination.
    baseline = record["configured_baseline_mpc_h"]
    assert baseline == 3.0
    superseded = math.sqrt(baseline**2 + record["shot_noise_variance_mpc_h2"])
    assert abs(record["analytic_center_mpc_h"] - superseded) > 1.0
    assert POSTRECONSTRUCTION_SIGMA_PERP_FORMULA["combination"] == (
        "Sigma_perp_analytic = sqrt(Sigma_rec^2 + Sigma_shot^2)"
    )
    assert "double count" in POSTRECONSTRUCTION_SIGMA_PERP_FORMULA["baseline_role"]


def test_the_shot_noise_terms_alone_are_not_interchangeable(linear_power):
    """One prior for both tracers is what this prescription replaces."""
    power, drag = linear_power
    config = load_analysis_config()
    bgs = _analytic("BGS_BRIGHT-21.5", BGS_DENSITY, power(BGS_Z_EFF), drag, config=config)
    lrg = _analytic("LRG", LRG_DENSITY, power(LRG_Z_EFF), drag, config=config)
    assert bgs["bias"] == 1.5 and lrg["bias"] == 2.0
    # The two shot terms differ by more than the prior's own width, so one shared centre could
    # not stand in for both without moving a posterior.
    assert lrg["shot_noise_sigma_mpc_h"] - bgs["shot_noise_sigma_mpc_h"] > (
        float(config["fitting"]["damping"]["gaussian_widths_mpc_h"]["sigma_perp"])
    )
    with pytest.raises(ValueError, match="not a configured tracer"):
        _analytic("ELG", 1e-4, power(0.5), drag, config=config)


def test_the_density_is_the_attained_one_and_not_the_requested_target(linear_power):
    """The mocks do not always reach the requested density, and the damping follows what they did.

    Using the target would understate the damping for exactly the samples where reconstruction is
    hardest, which is the case this prescription exists to get right.
    """
    power, drag = linear_power
    config = load_analysis_config()
    tier = next(t for t in config["selection"]["tiers"] if str(t["label"]) == "sigma_0.003")
    target = float(tier["target_nbar_h3_mpc3"]["LRG3"])
    assert LRG_DENSITY != target

    at_target = _analytic("LRG", target, power(LRG_Z_EFF), drag, config=config)
    at_attained = _analytic("LRG", LRG_DENSITY, power(LRG_Z_EFF), drag, config=config)
    assert at_attained["attained_density_h3_mpc3"] == LRG_DENSITY
    # The attained sample is sparser, so it damps more; the two are not interchangeable.
    assert at_attained["analytic_center_mpc_h"] > at_target["analytic_center_mpc_h"]
    # The reconstruction term is untouched by density -- only the shot term moved.
    assert at_attained["reconstruction_variance_mpc_h2"] == pytest.approx(
        at_target["reconstruction_variance_mpc_h2"]
    )


def test_the_approximation_is_a_diagnostic_and_is_not_the_centre(linear_power):
    """At the primary radius the ``R << r_d`` limit is wrong by a third, so it cannot be authority.

    It converges to the exact result as the radius shrinks against the drag scale, which is what
    makes it worth publishing beside the exact value rather than instead of it.
    """
    power, drag = linear_power
    exact = shot_noise_transverse_damping_squared(**SHOT)
    approximate = approximate_shot_noise_transverse_damping_squared(
        bias=SHOT["bias"], attained_density_h3_mpc3=BGS_DENSITY, smoothing_radius_mpc_h=15.0
    )
    assert approximate / exact == pytest.approx(1.3667, rel=1e-3)
    for radius, tolerance in ((2.0, 0.05), (0.5, 0.02)):
        near = shot_noise_transverse_damping_squared(**{**SHOT, "smoothing_radius_mpc_h": radius})
        near_approximate = approximate_shot_noise_transverse_damping_squared(
            bias=SHOT["bias"], attained_density_h3_mpc3=BGS_DENSITY, smoothing_radius_mpc_h=radius
        )
        assert near_approximate / near == pytest.approx(1.0, abs=tolerance)

    record = _analytic("BGS_BRIGHT-21.5", BGS_DENSITY, power(BGS_Z_EFF), drag)
    assert record["shot_noise_variance_mpc_h2"] == pytest.approx(exact)
    assert record["approximate_shot_noise_variance_mpc_h2"] != record["shot_noise_variance_mpc_h2"]


def test_the_covariance_shot_noise_rescaling_cannot_enter_the_formula():
    """``alpha_SN`` rescales the *covariance* estimator's shot noise, not the displacement's.

    The two are different measurements of different things that happen to share a name, and no
    configured mapping exists between them. The centre producer exposes no argument through which the
    calibrated scalar could be supplied.
    """
    import inspect

    assert POSTRECONSTRUCTION_SHOT_NOISE_P_SHOT == 1.0
    assert POSTRECONSTRUCTION_SIGMA_PERP_FORMULA["p_shot"] == 1.0
    assert "alpha_SN" in POSTRECONSTRUCTION_SIGMA_PERP_FORMULA["p_shot_note"]
    parameters = set(inspect.signature(analytic_postreconstruction_sigma_perp).parameters)
    assert not parameters & {"p_shot", "alpha_sn", "shot_noise_rescaling"}
    # The frozen BGS1 post value is ~1.807; using it would scale the variance by that outright.
    assert shot_noise_transverse_damping_squared(
        **SHOT, p_shot=1.806966145833335
    ) / shot_noise_transverse_damping_squared(**SHOT) == pytest.approx(1.806966145833335)


TRACER = "BGS_BRIGHT-21.5"


def test_the_transverse_width_is_the_configured_one_in_both_reconstruction_states(linear_power):
    """One width authority, and it is configuration.

    Reconstruction moves the transverse *centre* and nothing else about the prior. The width is
    the frozen Gaussian triple on both sides, so the post-reconstruction width is the same number
    as the pre-reconstruction one for the same tracer.
    """
    power, drag = linear_power
    config = load_analysis_config()
    widths = {
        name: float(value)
        for name, value in config["fitting"]["damping"]["gaussian_widths_mpc_h"].items()
    }
    analytic = _analytic(TRACER, BGS_DENSITY, power(BGS_Z_EFF), drag, config=config)
    common = dict(sigma_z_over_one_plus_z=0.003, z_eff=BGS_Z_EFF, e_z_eff=1.08, config=config)

    before = resolve_damping_prior(TRACER, "prereconstruction", **common)
    after = resolve_damping_prior(
        TRACER,
        "postreconstruction",
        postreconstruction_sigma_perp_artifact=analytic,
        **common,
    )
    assert after["widths_mpc_h"]["sigma_perp"] == widths["sigma_perp"]
    assert after["widths_mpc_h"]["sigma_perp"] == before["widths_mpc_h"]["sigma_perp"]
    assert after["centers_mpc_h"]["sigma_perp"] != before["centers_mpc_h"]["sigma_perp"]
    for prior in (before, after):
        assert prior["widths_mpc_h"] == widths
        # Neither state leaves a damping coordinate without its Gaussian: no route varies
        # sigma_perp on its bare physical support.
        assert prior["unpriored_parameters"] == []
        assert prior["scope"] == "final_science_fit"


def test_a_configuration_naming_another_width_authority_is_refused():
    """No second width source is implemented, so naming one may not pass silently.

    A configuration that named a mock-derived width while the code used the fixed triple anyway
    would leave the record and the behaviour disagreeing with nothing to notice.
    """
    altered = copy.deepcopy(load_analysis_config())
    altered["fitting"]["damping"]["width_authority"] = "mock_fitted_scatter"
    with pytest.raises(ValueError, match="width authority is not the fixed Gaussian widths"):
        resolve_damping_prior(
            TRACER,
            "prereconstruction",
            sigma_z_over_one_plus_z=0.003,
            z_eff=BGS_Z_EFF,
            e_z_eff=1.08,
            config=altered,
        )
