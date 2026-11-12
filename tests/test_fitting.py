"""Scientific priors, exact likelihood inputs, native profiles and independent ensembles."""

import pzbao.diagnostics.fitting as diag_fitting
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest
import pzbao.fitting as fitting_fitting
import pzbao.likelihood as fitting_likelihood
import pzbao.sampling as fitting_sampling
from pzbao.config import load_analysis_config
from pzbao.theory import resolve_damping_prior
from pzbao.models.mu_quadrature import project_multipoles, symmetric_leggauss


def test_exact_baseline_centers_for_each_tracer_and_state():
    config = load_analysis_config()
    lrg = resolve_damping_prior(
        "LRG",
        "prereconstruction",
        sigma_z_over_one_plus_z=0.0,
        z_eff=0.7,
        e_z_eff=1.47,
        config=config,
    )
    bgs = resolve_damping_prior(
        "BGS_BRIGHT-21.5",
        "postreconstruction",
        sigma_z_over_one_plus_z=0.0,
        z_eff=0.3,
        e_z_eff=1.17,
        postreconstruction_sigma_perp_artifact={
            "analytic_center_mpc_h": 3.0,
            "attained_density_h3_mpc3": 0.001,
            "tracer": "BGS_BRIGHT-21.5",
        },
        config=config,
    )
    assert lrg["centers_mpc_h"] == {
        "sigma_perp": 4.5,
        "sigma_parallel": 9.0,
        "sigma_streaming": 2.0,
    }
    assert bgs["centers_mpc_h"] == {
        "sigma_perp": 3.0,
        "sigma_parallel": 8.0,
        "sigma_streaming": 2.0,
    }


def test_postreconstruction_damping_requires_density_evidence_and_never_adds_photoz():
    with pytest.raises(ValueError, match="requires density-aware transverse evidence"):
        resolve_damping_prior(
            "LRG", "postreconstruction", sigma_z_over_one_plus_z=0.01, z_eff=0.7, e_z_eff=1.47
        )
    prior = resolve_damping_prior(
        "LRG",
        "postreconstruction",
        sigma_z_over_one_plus_z=0.01,
        z_eff=0.7,
        e_z_eff=1.47,
        postreconstruction_sigma_perp_artifact={
            "analytic_center_mpc_h": 6.5,
            "attained_density_h3_mpc3": 0.001,
            "tracer": "LRG",
        },
    )
    assert prior["centers_mpc_h"]["sigma_perp"] == 6.5
    assert prior["widths_mpc_h"]["sigma_perp"] == 1.0
    assert prior["centers_mpc_h"]["sigma_parallel"] == 6.0
    assert prior["bounds_mpc_h"]["sigma_parallel"] == [0.0, None]
    assert prior["sigma_parallel_photoz_mpc_h"] > 0


def test_unregistered_prior_inputs_are_rejected():
    with pytest.raises(ValueError, match="not registered"):
        resolve_damping_prior(
            "LRG", "postreconstruction", sigma_z_over_one_plus_z=0.02, z_eff=0.7, e_z_eff=1.47
        )
    missing_baseline = copy.deepcopy(load_analysis_config())
    del missing_baseline["fitting"]["damping"]["baseline_centers_mpc_h"]["LRG"][
        "postreconstruction"
    ]
    with pytest.raises(ValueError, match="no damping baseline.*postreconstruction"):
        resolve_damping_prior(
            "LRG",
            "postreconstruction",
            sigma_z_over_one_plus_z=0.003,
            z_eff=0.7,
            e_z_eff=1.47,
            postreconstruction_sigma_perp_artifact={
                "analytic_center_mpc_h": 6.5,
                "attained_density_h3_mpc3": 0.001,
                "tracer": "LRG",
            },
            config=missing_baseline,
        )


def _toy_qiso_likelihood(identified):
    """A real desilike likelihood over one parameter named like the declared AP interval.

    ``identified=False`` makes the model ignore ``qiso`` entirely, which is the converged-to-
    nothing case: Minuit runs, finds no minimum, and reports so through its own fit summary
    while still returning a full ``Profiles`` object.
    """
    from desilike.base import BaseCalculator
    from desilike.likelihoods import BaseGaussianLikelihood

    class Theory(BaseCalculator):
        _params = {
            "qiso": dict(
                value=0.5,
                ref=dict(dist="norm", loc=0.5, scale=0.1),
                prior=dict(limits=[-5.0, 5.0]),
                proposal=0.1,
            )
        }

        def initialize(self):
            pass

        def calculate(self, qiso=0.5):
            self.flatmodel = np.array([qiso, qiso**2]) if identified else np.array([1.0, 1.0])

    class Likelihood(BaseGaussianLikelihood):

        def initialize(self, theory=None):
            self.theory = theory
            super().initialize(data=np.array([1.0, 1.0]), covariance=np.eye(2) * 0.01)

        @property
        def flattheory(self):
            return self.theory.flatmodel

    return Likelihood(theory=Theory())


def test_a_profiler_that_did_not_converge_is_distinguishable_from_one_that_did(tmp_path):
    """Known-defect register item 6, measured against the real pinned profiler.

    The products were checked for existence and non-zero size only. Neither of the library's two
    failure modes changes either property: a minimization that finds no minimum still returns a
    full ``Profiles``, and a minimization that raises inside Minuit is caught, logged as a
    warning, and returns an empty one -- on top of which ``BaseProfiler.maximize`` has already
    seeded ``bestfit`` with the starting point, so the saved file is a fit-shaped object holding
    no fit. Every profiler here is the real ``MinuitProfiler``.
    """
    pytest.importorskip("desilike")
    from desilike.profilers import MinuitProfiler
    from desilike.samples.profiles import Profiles

    config = load_analysis_config()
    iterations = int(config["fitting"]["optimization"]["profiler_iterations"])
    converged = fitting_fitting.run_profiling(
        _toy_qiso_likelihood(True),
        {
            "profiles": str(tmp_path / "profiles.npy"),
            "statistics": str(tmp_path / "statistics.txt"),
        },
        config,
        ap_mode="qiso",
    )
    assert converged["maximization"] == {
        "is_valid": True,
        "is_above_max_edm": False,
        "has_reached_call_limit": False,
        "converged": True,
    }
    assert converged["intervals"]["qiso"]["is_valid"] is True
    assert converged["intervals"]["qiso"]["railed"] is False
    assert converged["profile_failures_or_rails"] == 0
    assert (tmp_path / "profiles.npy").stat().st_size > 0
    unidentified = MinuitProfiler(_toy_qiso_likelihood(False), seed=42)
    failed = fitting_fitting._maximization_convergence(
        unidentified.maximize(niterations=iterations)
    )
    assert failed["is_valid"] is False
    assert failed["is_above_max_edm"] is True
    assert failed["converged"] is False

    class Raised(MinuitProfiler):

        def _maximize_one(self, state, **kwargs):
            return Profiles()

    started = Raised(_toy_qiso_likelihood(True), seed=42).maximize(niterations=iterations)
    assert "bestfit" in started
    assert (
        dict(started.bestfit.attrs)
        .keys()
        .isdisjoint(fitting_fitting.MAXIMIZATION_CONVERGENCE_FLAGS)
    )
    started_path = tmp_path / "started.npy"
    started.save(str(started_path))
    assert started_path.stat().st_size > 0
    with pytest.raises(RuntimeError, match="no maximization summary"):
        fitting_fitting._maximization_convergence(started)
    with pytest.raises(RuntimeError, match="no confidence interval for \\['qiso'\\]"):
        fitting_fitting._interval_convergence(Profiles(), ["qiso"])


def test_fitting_rejects_multi_rank_communicators_before_inference():
    fitting_likelihood._require_single_process(SimpleNamespace(size=1, rank=0), "fit")
    with pytest.raises(RuntimeError, match="exactly one process"):
        fitting_likelihood._require_single_process(SimpleNamespace(size=2, rank=0), "fit")
    with pytest.raises(RuntimeError, match="exactly one process"):
        fitting_likelihood._require_single_process(SimpleNamespace(size=2, rank=1), "fit")


def test_likelihood_parameter_configuration_is_reapplied_as_one_contract():

    class Parameter:

        def __init__(self, name):
            self.name = name
            self.basename = name
            self.derived = False
            self.fixed = True
            self.prior = None

        def update(self, **kwargs):
            for key, value in kwargs.items():
                if key == "delta" and value is not None and (np.ndim(value) == 0):
                    value = (value, value)
                setattr(self, key, value)

    nuisances = {name: Parameter(name) for name in ("bl0_0", "bl0_2", "al2_0")}
    damping = {name: Parameter(name) for name in ("sigmas", "sigmapar", "sigmaper")}
    nonlinear = {name: Parameter(name) for name in ("qiso", "b1", "dbeta")}

    class Parameters(dict):

        def select(self, *, basename):
            assert basename == ["al*_*", "bl*_*"]
            return list(nuisances.values())

    parameters = Parameters({**damping, **nonlinear, **nuisances})
    likelihood = SimpleNamespace(
        all_params=parameters,
        varied_params=[
            *damping.values(),
            nonlinear["qiso"],
            nonlinear["b1"],
            nuisances["bl0_0"],
            nuisances["bl0_2"],
        ],
    )
    prior = {
        "centers_mpc_h": {"sigma_streaming": 2.0, "sigma_parallel": 8.0, "sigma_perp": 3.0},
        "widths_mpc_h": {"sigma_streaming": 2.0, "sigma_parallel": 2.0, "sigma_perp": 1.0},
        "bounds_mpc_h": {
            "sigma_streaming": [0.0, None],
            "sigma_parallel": [0.0, None],
            "sigma_perp": [0.0, None],
        },
    }
    config = load_analysis_config(Path(__file__).with_name("fixtures") / "analysis_schema14.yaml")
    config["schema_version"] = 13
    config["fitting"]["fit_surfaces"]["BGS_BRIGHT-21.5"] = {
        "observed_multipoles": [0],
        "model_multipoles": [0, 2, 4],
        "ap_mode": "qiso",
        "fixed_parameters": {"qap": 1.0, "dbeta": 1.0},
    }
    config["fitting"]["linear_nuisance"]["varied_basenames_by_tracer"]["BGS_BRIGHT-21.5"] = [
        "bl0_0",
        "bl0_2",
    ]
    fitting_likelihood._configure_likelihood_parameters(
        likelihood, prior, method="detection", ap_mode="qiso", config=config
    )
    assert nuisances["bl0_0"].derived == ".auto"
    assert nuisances["al2_0"].fixed is True
    assert all((parameter.fixed is False for parameter in damping.values()))
    assert damping["sigmapar"].prior["loc"] == 8.0
    assert nonlinear["qiso"].prior == config["fitting"]["nonlinear_parameters"]["qiso"]["prior"]
    assert nonlinear["b1"].value == 2.0
    assert nonlinear["dbeta"].fixed is True
    assert nonlinear["dbeta"].value == 1.0
    assert nuisances["bl0_0"].prior is None
    assert nuisances["bl0_0"].ref == config["fitting"]["linear_nuisance"]["reference"]
    missing_fixed = copy.deepcopy(config)
    del missing_fixed["fitting"]["fit_surfaces"]["BGS_BRIGHT-21.5"]["fixed_parameters"]["dbeta"]
    with pytest.raises(ValueError, match="fixed-parameter mapping differs"):
        fitting_likelihood._configure_likelihood_parameters(
            likelihood, prior, method="detection", ap_mode="qiso", config=missing_fixed
        )
    changed_fixed = copy.deepcopy(config)
    changed_fixed["fitting"]["fit_surfaces"]["BGS_BRIGHT-21.5"]["fixed_parameters"]["dbeta"] = 0.9
    with pytest.raises(ValueError, match="differs from its configured nonlinear value"):
        fitting_likelihood._configure_likelihood_parameters(
            likelihood, prior, method="detection", ap_mode="qiso", config=changed_fixed
        )
    missing_treatment = copy.deepcopy(config)
    del missing_treatment["fitting"]["linear_nuisance"]["treatment"]["detection"]
    with pytest.raises(ValueError, match="treatment is not configured"):
        fitting_likelihood._configure_likelihood_parameters(
            likelihood, prior, method="detection", ap_mode="qiso", config=missing_treatment
        )
    missing_nonlinear = copy.deepcopy(config)
    del missing_nonlinear["fitting"]["nonlinear_parameters"]["qiso"]
    with pytest.raises(ValueError, match="omits active names"):
        fitting_likelihood._configure_likelihood_parameters(
            likelihood, prior, method="detection", ap_mode="qiso", config=missing_nonlinear
        )
    exact_update = nonlinear["qiso"].update

    def drifting_update(**kwargs):
        exact_update(**kwargs)
        nonlinear["qiso"].value += 0.01

    nonlinear["qiso"].update = drifting_update
    with pytest.raises(RuntimeError, match="qiso differs from its complete configured state"):
        fitting_likelihood._configure_likelihood_parameters(
            likelihood, prior, method="detection", ap_mode="qiso", config=config
        )
    nonlinear["qiso"].update = exact_update

    def asymmetric_delta_update(**kwargs):
        exact_update(**kwargs)
        nonlinear["qiso"].delta = (0.008, 0.009)

    nonlinear["qiso"].update = asymmetric_delta_update
    with pytest.raises(RuntimeError, match="qiso differs from its complete configured state"):
        fitting_likelihood._configure_likelihood_parameters(
            likelihood, prior, method="detection", ap_mode="qiso", config=config
        )
    nonlinear["qiso"].update = exact_update
    configured_distribution = copy.deepcopy(config)
    configured_distribution["fitting"]["nonlinear_parameters"]["qiso"]["prior"] = {
        "dist": "norm",
        "loc": 1.0,
        "scale": 0.05,
        "limits": [0.8, 1.2],
    }

    def drifting_distribution_update(**kwargs):
        exact_update(**kwargs)
        nonlinear["qiso"].prior = dict(nonlinear["qiso"].prior, loc=1.01)

    nonlinear["qiso"].update = drifting_distribution_update
    with pytest.raises(RuntimeError, match="complete configured state"):
        fitting_likelihood._configure_likelihood_parameters(
            likelihood, prior, method="detection", ap_mode="qiso", config=configured_distribution
        )
    nonlinear["qiso"].update = exact_update
    exact_damping_update = damping["sigmapar"].update

    def drifting_damping_distribution(**kwargs):
        exact_damping_update(**kwargs)
        damping["sigmapar"].prior = dict(damping["sigmapar"].prior, dist="uniform")

    damping["sigmapar"].update = drifting_damping_distribution
    with pytest.raises(RuntimeError, match="sigmapar changed its configured prior"):
        fitting_likelihood._configure_likelihood_parameters(
            likelihood, prior, method="detection", ap_mode="qiso", config=config
        )
    damping["sigmapar"].update = exact_damping_update
    missing_damping = parameters.pop("sigmas")
    with pytest.raises(ValueError, match="no required damping parameter sigmas"):
        fitting_likelihood._configure_likelihood_parameters(
            likelihood, prior, method="detection", ap_mode="qiso", config=config
        )
    parameters["sigmas"] = missing_damping
    likelihood.varied_params.append(Parameter("undeclared"))
    with pytest.raises(RuntimeError, match="undeclared parameter"):
        fitting_likelihood._configure_likelihood_parameters(
            likelihood, prior, method="detection", ap_mode="qiso", config=config
        )


def test_pinned_desilike_parameter_state_and_total_fitted_counts(tmp_path, monkeypatch):
    monkeypatch.setenv("DESILIKE_CONFIG_DIR", str(tmp_path))
    parameter_module = pytest.importorskip("desilike.parameter")
    Parameter = parameter_module.Parameter
    ParameterCollection = parameter_module.ParameterCollection
    config = load_analysis_config(Path(__file__).with_name("fixtures") / "analysis_schema14.yaml")
    prior = {
        "centers_mpc_h": {"sigma_streaming": 2.0, "sigma_parallel": 8.0, "sigma_perp": 3.0},
        "widths_mpc_h": {"sigma_streaming": 2.0, "sigma_parallel": 2.0, "sigma_perp": 1.0},
        "bounds_mpc_h": {
            "sigma_streaming": [0.0, None],
            "sigma_parallel": [0.0, None],
            "sigma_perp": [0.0, None],
        },
    }
    names = [
        "qiso",
        "qap",
        "b1",
        "dbeta",
        "sigmas",
        "sigmapar",
        "sigmaper",
        "bl0_0",
        "bl0_2",
        "al2_0",
        "al2_1",
        "bl2_0",
        "bl2_2",
    ]

    class Likelihood:

        def __init__(self):
            self.all_params = ParameterCollection([Parameter(name, fixed=False) for name in names])

        @property
        def varied_params(self):
            return self.all_params.select(varied=True, derived=False)

        @property
        def nvaried(self):
            return len(self.varied_params) + len(self.all_params.select(solved=True))

    for tracer, ap_mode, expected in (("BGS_BRIGHT-21.5", "qisoqap", 12), ("LRG", "qisoqap", 12)):
        likelihood = Likelihood()
        fitting_likelihood._configure_likelihood_parameters(
            likelihood, prior, method="profiling", ap_mode=ap_mode, tracer=tracer, config=config
        )
        assert likelihood.nvaried == expected
        assert not likelihood.all_params["qap"].fixed
        np.testing.assert_array_equal(likelihood.all_params["qap"].prior.limits, [0.8, 1.2])
        assert likelihood.all_params["dbeta"].fixed
        qiso_delta = likelihood.all_params["qiso"].__getstate__()["delta"]
        assert qiso_delta == (0.008, 0.008)


QUADRATURE_K_GRID = np.geomspace(0.0001, 0.6, 300)
QUADRATURE_SMOOTH = (
    20000.0 * (QUADRATURE_K_GRID / 0.05) ** (-1.5) / (1.0 + (QUADRATURE_K_GRID / 0.2) ** 2)
)
QUADRATURE_WIGGLE = (
    QUADRATURE_SMOOTH
    * 0.05
    * np.sin(QUADRATURE_K_GRID * 105.0)
    * np.exp(-((QUADRATURE_K_GRID / 0.35) ** 2))
)
LARGEST_PAIRWISE_SIGMA = float(np.sqrt(2.0) * 102.0)


def _damping_prior(pairwise_sigma_mpc_h):
    """The fields of a resolved damping prior that the quadrature certificate reads."""
    return {
        "sigma_parallel_photoz_mpc_h": float(pairwise_sigma_mpc_h),
        "centers_mpc_h": {"sigma_perp": 4.5, "sigma_parallel": 9.0, "sigma_streaming": 2.0},
        "widths_mpc_h": {"sigma_perp": 1.0, "sigma_parallel": 2.0, "sigma_streaming": 2.0},
        "bounds_mpc_h": {
            "sigma_perp": [0.0, None],
            "sigma_parallel": [0.0, None],
            "sigma_streaming": [0.0, None],
        },
    }


@pytest.mark.parametrize("tracer", ["BGS_BRIGHT-21.5", "LRG"])
@pytest.mark.parametrize("state", ["prereconstruction", "postreconstruction"])
def test_native_smooth_likelihood_dilations_are_inactive_and_optimizer_is_usable(
    tmp_path, tracer, state
):
    pytest.importorskip("desilike")
    from desilike.profilers import MinuitProfiler

    config = load_analysis_config(Path(__file__).with_name("fixtures") / "analysis_schema14.yaml")
    surface = config["fitting"]["fit_surfaces"][tracer]
    name = config["tracers"][tracer]["measurement_bins"][0]["name"]
    multipoles = surface["observed_multipoles"]
    edges = np.arange(
        config["fitting"]["fit_edge_limits_mpc_h"][0],
        config["fitting"]["fit_edge_limits_mpc_h"][1] + 1,
        config["fitting"]["input_bin_width_mpc_h"],
    )
    size = (len(edges) - 1) * len(multipoles)
    geometry = {
        **_bgs_geometry_artifact(config),
        "tracer": tracer,
        "measurement_bin": name,
        "reconstruction_state": state,
    }

    def build(data, only_now):
        return fitting_likelihood.build_likelihood(
            data,
            np.eye(size) * 1e-08,
            separations_mpc_h=(edges[:-1] + edges[1:]) / 2,
            fit_sedges=edges,
            z_eff=0.15 if tracer.startswith("BGS") else 0.5,
            multipoles=multipoles,
            reconstruction_state=state,
            reconstruction_convention="recsym",
            smoothing_mpc_h=15.0,
            ap_mode=surface["ap_mode"],
            method="detection" if only_now else "profiling",
            damping_prior=_damping_prior(20.0),
            geometry_artifact=geometry,
            sample_identity="sample-a",
            tracer=tracer,
            measurement_bin=name,
            config=config,
            only_now=only_now,
        )[2]

    smooth = build(np.zeros(size), True)
    smooth()
    truth = np.array(smooth.flattheory, copy=True)
    smooth = build(truth, True)
    baseline = float(smooth())
    baseline_model = np.array(smooth.flattheory, copy=True)
    dilation_names = config["fitting"]["ap_interval_parameters"][surface["ap_mode"]]
    for parameter in dilation_names:
        assert smooth.all_params[parameter].fixed
        smooth.all_params[parameter].update(fixed=False)
        for value in (0.95, 1.0, 1.05):
            assert float(smooth(**{parameter: value})) == baseline
            np.testing.assert_array_equal(smooth.flattheory, baseline_model)
        smooth.all_params[parameter].update(fixed=True)
    varied = {parameter.basename for parameter in smooth.varied_params}
    assert not varied.intersection(dilation_names)
    assert {"b1", "sigmas", "sigmapar", "sigmaper"}.issubset(varied)
    profile = fitting_fitting.maximize_usable(
        MinuitProfiler(smooth, seed=config["fitting"]["optimization"]["random_seed"]), config
    )
    destination = tmp_path / "smooth_profiles.npy"
    profile.save(destination)
    decoded = fitting_fitting.read_profiles(destination)
    assert decoded["maximization"]["converged"]
    assert np.isfinite(decoded["logposterior"])
    bao = build(truth, False)
    bao()
    reference = np.array(bao.flattheory, copy=True)
    for parameter in dilation_names:
        assert not bao.all_params[parameter].fixed
        bao(**{parameter: 1.05})
        assert np.max(np.abs(bao.flattheory - reference)) > 0
        bao(**{parameter: config["fitting"]["nonlinear_parameters"][parameter]["value"]})


def _install_fitting_library_fakes(monkeypatch, *, realized_order=None):
    """Install desilike stand-ins whose angular integral is the real analytic one.

    The power-model stand-in projects the same multipoles the pinned model does, at whatever
    order it is built with, so the convergence behaviour these tests assert on is the physics
    rather than a scripted answer. ``realized_order`` forces the stand-in to build a rule of a
    different order from the one requested, which is how the guard against certifying one rule
    and running another is exercised.
    """
    built, observables, likelihoods = ([], [], [])

    class Template:

        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class PowerModel:

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.requested_order = int(kwargs["mu_quadrature_order"])
            self.mu, _ = symmetric_leggauss(int(realized_order or self.requested_order))
            self.k = QUADRATURE_K_GRID
            self.power = None
            built.append(self)

        def evaluate(self, ells, **parameters):
            self.power = project_multipoles(
                QUADRATURE_K_GRID,
                QUADRATURE_SMOOTH,
                QUADRATURE_WIGGLE,
                pairwise_sigma_mpc_h=float(self.kwargs["pairwise_sigma_mpc_h"]),
                multipoles=ells,
                order=int(self.mu.size),
                growth_rate=0.8 * float(parameters.get("dbeta", 1.0)),
                linear_bias=float(parameters.get("b1", 2.0)),
                sigma_par_mpc_h=float(parameters.get("sigmapar", 9.0)),
                sigma_perp_mpc_h=float(parameters.get("sigmaper", 4.5)),
            )

    class Theory:

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.pt = kwargs.get("pt")
            self.ells = tuple((int(value) for value in kwargs.get("ells", (0, 2))))

        def __call__(self, **parameters):
            self.pt.evaluate(self.ells, **parameters)
            return np.zeros((len(self.ells), 5))

    class Observable:

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.wmatrix = SimpleNamespace(ellsin=tuple(kwargs["ellsin"]))
            observables.append(self)

    class Likelihood:

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            likelihoods.append(self)

    monkeypatch.setitem(
        sys.modules,
        "desilike.theories.galaxy_clustering",
        SimpleNamespace(
            BAOPowerSpectrumTemplate=Template,
            DampedBAOWigglesTracerCorrelationFunctionMultipoles=Theory,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "desilike.observables.galaxy_clustering",
        SimpleNamespace(TracerCorrelationFunctionMultipolesObservable=Observable),
    )
    monkeypatch.setitem(
        sys.modules,
        "desilike.likelihoods",
        SimpleNamespace(ObservablesGaussianLikelihood=Likelihood),
    )
    monkeypatch.setattr(fitting_likelihood, "build_mock_cosmology", lambda config: object())
    monkeypatch.setattr(
        fitting_likelihood,
        "_build_observed_photoz_power_model",
        lambda **kwargs: PowerModel(**kwargs),
    )
    monkeypatch.setattr(
        fitting_likelihood, "_configure_likelihood_parameters", lambda *args, **kwargs: None
    )
    return SimpleNamespace(power_models=built, observables=observables, likelihoods=likelihoods)


def _bgs_geometry_artifact(config):
    return {
        "sample_identity": "sample-a",
        "tracer": "BGS_BRIGHT-21.5",
        "measurement_bin": "BGS1",
        "reconstruction_state": "postreconstruction",
        "sedges": np.arange(0.0, 201.0, 1.0),
        "muedges": np.array([-1.0, 0.0, 1.0]),
        "wcounts": np.ones((200, 2)),
    }


def _build_bgs_likelihood(
    config,
    geometry,
    *,
    pairwise_sigma_mpc_h,
    method="profiling",
    only_now=False,
    smoothing_mpc_h=15.0,
):
    surface = config["fitting"]["fit_surfaces"]["BGS_BRIGHT-21.5"]
    dimension = 26 * len(surface["observed_multipoles"])
    return fitting_likelihood.build_likelihood(
        np.zeros(dimension),
        np.eye(dimension),
        separations_mpc_h=np.arange(50.0, 151.0, 4.0),
        fit_sedges=np.arange(48.0, 153.0, 4.0),
        z_eff=0.15,
        multipoles=surface["observed_multipoles"],
        reconstruction_state="postreconstruction",
        reconstruction_convention="recsym",
        smoothing_mpc_h=smoothing_mpc_h,
        ap_mode=surface["ap_mode"],
        method="profiling",
        damping_prior=_damping_prior(pairwise_sigma_mpc_h),
        geometry_artifact=geometry,
        sample_identity="sample-a",
        tracer="BGS_BRIGHT-21.5",
        measurement_bin="BGS1",
        config=config,
        only_now=only_now,
    )


def test_native_transfer_pcs2_likelihood_uses_six_nuisances(tmp_path, monkeypatch):
    monkeypatch.setenv("DESILIKE_CONFIG_DIR", str(tmp_path))
    pytest.importorskip("desilike")
    from pzbao.config import derive_inference_config
    from pzbao.config.transfer import transfer_run_configs

    source = transfer_run_configs(load_analysis_config())["self--R25"]
    config = derive_inference_config(source, "transfer_pcs2")
    _, observable, likelihood, _ = _build_bgs_likelihood(
        config,
        _bgs_geometry_artifact(config),
        pairwise_sigma_mpc_h=20.0,
        smoothing_mpc_h=25.0,
    )
    assert np.isfinite(float(likelihood()))
    assert np.asarray(likelihood.flattheory).shape == (52,)
    assert not hasattr(observable, "power_broadband_repair")
    assert {parameter.basename for parameter in likelihood.all_params.select(solved=True)} == {
        "bl0_0",
        "bl0_2",
        "al2_0",
        "al2_1",
        "bl2_0",
        "bl2_2",
    }
    assert likelihood.nvaried == 12


class _GridXi:
    """The frozen production grid: 0 to 200 Mpc/h in 4 Mpc/h bins after rebinning by four.

    The pair-weighted separations sit deliberately *above* their geometric centres, as real
    pair counts do, because that offset is exactly what the retired centre-masking selection
    was sensitive to. Edge selection must be indifferent to it.
    """

    edges = (np.arange(0.0, 204.0, 4.0), np.array([-1.0, 1.0]))
    separations = 0.5 * (edges[0][:-1] + edges[0][1:]) + 0.2

    def __getitem__(self, item):
        assert item == slice(None, None, 4)
        return self

    def __call__(self, *, ells, return_sep, return_std):
        assert return_sep is True and return_std is False
        return (self.separations, [np.full(self.separations.size, float(ell)) for ell in ells])


def test_frozen_fit_grid_is_twenty_six_bins_on_edges_48_to_152():
    """Edges 48, 52, ... 152; centres 50, 54, ... 150; twenty-six radial bins.

    This is the whole fit-vector contract in one place. It is asserted against the
    configuration's own values, so a configuration edit that moved the interval fails here
    rather than silently changing every likelihood dimension in the campaign.
    """
    config = load_analysis_config()
    lo, hi = [float(value) for value in config["fitting"]["fit_edge_limits_mpc_h"]]
    width = float(config["fitting"]["input_bin_width_mpc_h"])
    assert (lo, hi, width) == (48.0, 152.0, 4.0)
    edges = np.arange(lo, hi + 0.5 * width, width)
    centres = 0.5 * (edges[:-1] + edges[1:])
    assert edges[0] == 48.0 and edges[-1] == 152.0
    assert centres[0] == 50.0 and centres[-1] == 150.0
    assert centres.size == 26


def test_xi_fit_data_selects_on_edges_and_not_on_pair_weighted_separations():
    """Membership follows the bin edges, so the retained count cannot depend on the counts.

    The fixture's pair-weighted separations sit 0.2 Mpc/h above their centres. Under the
    retired centre-masking rule the top bin's separation (150.2) fell outside a [50, 150]
    limit and the vector came back one bin short. Under edge selection it does not.
    """
    xi = _GridXi()
    separation, vector, edges = fitting_likelihood.xi_fit_data(
        xi,
        multipoles=[0, 2],
        fit_edge_limits_mpc_h=[48.0, 152.0],
        rebin_factor=4,
        expected_bin_width_mpc_h=4.0,
    )
    assert separation.size == 26
    assert vector.size == 52
    np.testing.assert_allclose(edges[0], 48.0)
    np.testing.assert_allclose(edges[-1], 152.0)
    np.testing.assert_allclose(separation[-1], 150.2)
    with pytest.raises(ValueError, match="edges differ"):
        fitting_likelihood.xi_fit_data(
            xi,
            multipoles=[0, 2],
            fit_edge_limits_mpc_h=[48.0, 152.0],
            rebin_factor=4,
            expected_bin_width_mpc_h=5.0,
        )

    class _ShortXi(_GridXi):
        edges = (np.arange(0.0, 104.0, 4.0), np.array([-1.0, 1.0]))
        separations = 0.5 * (edges[0][:-1] + edges[0][1:]) + 0.2

    with pytest.raises(ValueError, match="does not span the fitted edge interval"):
        fitting_likelihood.xi_fit_data(
            _ShortXi(),
            multipoles=[0],
            fit_edge_limits_mpc_h=[48.0, 152.0],
            rebin_factor=4,
            expected_bin_width_mpc_h=4.0,
        )


def test_linear_gr_uses_symmetric_generalized_eigenvalues():
    within = np.array([[4.0, 1.8], [1.8, 1.0]])
    total = np.array([[4.2, 1.7], [1.7, 1.3]])
    expected = np.linalg.eigvals(np.linalg.solve(within, total)).real.max() - 1
    observed = fitting_sampling.covariance_ratio_eigen_gr(total, within)
    assert observed == pytest.approx(expected, rel=1e-12)
    wrong = np.linalg.eigvalsh(np.linalg.solve(within, total)).max() - 1
    assert not np.isclose(observed, wrong, rtol=1e-12, atol=1e-12)
    transform = np.array([[1.0, 3.0], [-2.0, 0.5]])
    assert fitting_sampling.covariance_ratio_eigen_gr(
        transform @ total @ transform.T, transform @ within @ transform.T
    ) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize(
    "within", [np.zeros((2, 2)), np.ones((2, 2)), np.array([[1.0, 0.0], [0.0, np.nan]])]
)
def test_linear_gr_cannot_pass_singular_or_nonfinite_within_covariance(within):
    with pytest.raises((ValueError, np.linalg.LinAlgError)):
        fitting_sampling.covariance_ratio_eigen_gr(np.eye(2), within)


def test_linear_sampling_history_is_cumulative_and_unusable_checks_reset_it():
    sampling = load_analysis_config()["fitting"]["sampling"]
    steps = fitting_sampling.sampling_check_steps(sampling)
    assert steps == list(range(300, 10000, 300)) + [10000]
    good = {"usable": True, "value": 0.009}
    first = fitting_sampling.update_gr_history([], good, step=300, sampling=sampling)
    assert not first[-1]["stopping_rule_reached"]
    assert fitting_sampling.update_gr_history(first, good, step=300, sampling=sampling) == first
    second = fitting_sampling.update_gr_history(first, good, step=600, sampling=sampling)
    assert second[-1]["stopping_rule_reached"]
    failed = fitting_sampling.update_gr_history(
        first, {"usable": False, "value": None}, step=600, sampling=sampling
    )
    assert failed[-1]["streak"] == 0
    next_good = fitting_sampling.update_gr_history(failed, good, step=900, sampling=sampling)
    assert next_good[-1]["streak"] == 1
    with pytest.raises(ValueError, match="missing"):
        fitting_sampling.update_gr_history([], good, step=600, sampling=sampling)
    boundary = fitting_sampling.update_gr_history(
        [], {"usable": True, "value": 0.01}, step=300, sampling=sampling
    )
    assert boundary[-1]["streak"] == 0


def test_linear_sampler_stream_identities_derive_from_route_and_ensemble_indices():
    config = load_analysis_config()
    route = {
        "measurement_bin": "BGS1",
        "condition": "sigma_0.003",
        "reconstruction_state": "prereconstruction",
    }
    streams = fitting_sampling.ensemble_random_streams(config, route)
    assert len(streams) == 4
    assert [item["ensemble_index"] for item in streams] == list(range(4))
    assert len({tuple(item["spawn_key"]) for item in streams}) == 4
    assert all((item["root_seed"] == 4201 for item in streams))
    post = fitting_sampling.ensemble_random_streams(
        config, {**route, "reconstruction_state": "postreconstruction"}
    )
    assert not {tuple(item["spawn_key"]) for item in streams}.intersection(
        (tuple(item["spawn_key"]) for item in post)
    )
    assert fitting_sampling.ensemble_random_streams(config, dict(route)) == streams


def test_linear_native_chain_roundtrip_preserves_time_axes_and_identity(tmp_path):
    Chain = pytest.importorskip("desilike.samples").Chain
    values = np.arange(240, dtype=float).reshape(60, 4)
    identity = {"run_name": "tiny", "ensemble_index": 2}
    chain = Chain(data=[values, values * 2 + 1], params=["qiso", "b1"])
    chain.attrs["identity"] = identity
    path = tmp_path / "chain.npy"
    chain.save(path)
    decoded = fitting_sampling.read_chain(path, parameter_names=["qiso", "b1"], identity=identity)
    assert decoded["axes"] == ["walker", "iteration", "parameter"]
    np.testing.assert_array_equal(decoded["draws"][:, :, 0], values.T)
    np.testing.assert_array_equal(decoded["draws"][:, :, 1], (2 * values + 1).T)
    with pytest.raises(ValueError, match="identity"):
        fitting_sampling.read_chain(
            path, parameter_names=["qiso"], identity={**identity, "ensemble_index": 3}
        )
    with pytest.raises(ValueError, match="missing"):
        fitting_sampling.read_chain(path, parameter_names=["qap"])


def test_linear_native_gr_compares_four_retained_ensembles(tmp_path):
    Chain = pytest.importorskip("desilike.samples").Chain
    rng = np.random.RandomState(234)
    arrays = [
        rng.multivariate_normal([i * 0.01, -i * 0.02], [[2, 0.8], [0.8, 1]], size=(600, 40))
        for i in range(4)
    ]
    chains = []
    for i, values in enumerate(arrays):
        path = tmp_path / f"ensemble-{i}.npy"
        Chain(data=[values[..., 0], values[..., 1]], params=["qiso", "b1"]).save(path)
        chains.append(fitting_sampling.read_chain(path, parameter_names=["qiso", "b1"])["chain"])
    sampling = load_analysis_config()["fitting"]["sampling"]
    result = fitting_sampling.ensemble_gelman_rubin(chains, ["qiso", "b1"], sampling=sampling)
    retained = [array[300:].reshape(-1, 2) for array in arrays]
    within = np.mean([np.cov(array.T, ddof=1) for array in retained], axis=0)
    total = (1 - 1 / len(retained[0])) * within + 1.25 * np.cov(
        np.array([array.mean(axis=0) for array in retained]).T, ddof=1
    )
    expected = np.linalg.eigvals(np.linalg.solve(within, total)).real.max() - 1
    assert result["usable"]
    assert result["value"] == pytest.approx(expected, abs=1e-12)
    assert result["statistic"] == "max_generalized_covariance_ratio_minus_one"
    assert not fitting_sampling.ensemble_gelman_rubin(
        chains[:3], ["qiso", "b1"], sampling=sampling
    )["usable"]


def test_linear_report_only_autocorrelation_keeps_opposing_walkers_visible():
    pytest.importorskip("emcee")
    sampling = load_analysis_config()["fitting"]["sampling"]
    draws = np.ones((40, 600, 1))
    draws[20:] = -1
    assert not np.any(draws.mean(axis=0))
    diagnostic = fitting_sampling.ensemble_autocorrelation(draws, ["qiso"], sampling=sampling)
    assert diagnostic["report_only"] is True
    row = diagnostic["parameters"]["qiso"]
    assert row["tau"] is None and row["nominal_autocorrelation_based_ess"] is None
    assert "constant" in row["unavailable_reason"]


def test_linear_native_profiles_roundtrip_keeps_asymmetric_intervals_and_changed_values(tmp_path):
    pytest.importorskip("desilike")
    from desilike.samples import Profiles
    from desilike.samples.profiles import ParameterBestFit
    from desilike.parameter import Samples

    identity = {"measurement_bin": "BGS1", "realization_id": 3, "cap": "NGC+SGC"}
    flags = {"is_valid": True, "is_above_max_edm": False, "has_reached_call_limit": False}
    bestfit = ParameterBestFit(
        [np.array([1.02]), np.array([-4.0]), np.array([-4.5])],
        params=["qiso", "loglikelihood", "logposterior"],
        attrs=flags,
    )
    interval_flags = {name: False for name in fitting_fitting.INTERVAL_CONVERGENCE_FLAGS}
    interval_flags.update(is_valid=True, lower_valid=True, upper_valid=True, at_upper_limit=True)
    profiles = Profiles(
        bestfit=bestfit,
        error=Samples([np.array([0.04])], params=["qiso"]),
        interval=Samples(
            [np.array([-0.03, 0.08])], params=["qiso"], attrs={"qiso": interval_flags}
        ),
        attrs={"identity": identity},
    )
    path = tmp_path / "profiles.npy"
    profiles.save(str(path))
    decoded = fitting_fitting.read_profiles(path, identity=identity)
    row = next((row for row in decoded["parameters"] if row["name"] == "qiso"))
    assert row["value"] == 1.02 and row["parabolic_error"] == 0.04
    np.testing.assert_allclose(row["minos_bounds"], [0.99, 1.1])
    assert row["interval_flags"]["at_upper_limit"]
    assert decoded["chi2"] == 8.0
    profiles.bestfit["qiso"][...] = 0.97
    profiles.save(str(path))
    changed = fitting_fitting.read_profiles(path, identity=identity)
    assert changed["parameters"][0]["value"] == 0.97
    with pytest.raises(ValueError, match="identity"):
        fitting_fitting.read_profiles(path, identity={**identity, "realization_id": 4})


def test_linear_optimizer_selects_a_successful_start_with_its_own_flags():
    config = load_analysis_config()
    count = config["fitting"]["optimization"]["profiler_iterations"]
    starts = np.arange(count)[:, None]

    class Profile:

        def __init__(self, objective, flags):
            self.bestfit = SimpleNamespace(logposterior=np.array([objective]), attrs=flags)
            self.attrs = {}

        def deepcopy(self):
            return copy.deepcopy(self)

    class Profiler:
        profiles = None

        def _get_start(self, *, niterations):
            assert niterations == count
            return starts.copy()

        def maximize(self, *, niterations, start):
            assert niterations == 1
            assert isinstance(start, list) and len(start) == 1
            index = int(start[0][0])
            flags = (
                {}
                if index == 0
                else {
                    "is_valid": index != count - 1,
                    "is_above_max_edm": False,
                    "has_reached_call_limit": False,
                }
            )
            return Profile(float(index), flags)

    selected = fitting_fitting.maximize_usable(Profiler(), config)
    assert selected.attrs["selected_start_index"] == count - 2
    assert len(selected.attrs["optimizer_starts"]) == count
    assert not selected.attrs["optimizer_starts"][0]["converged"]
    assert not selected.attrs["optimizer_starts"][-1]["is_valid"]


@pytest.mark.parametrize(
    "method,only_now", [("profiling", False), ("mcmc", False), ("detection", True)]
)
def test_linear_likelihood_factories_keep_explicit_128_without_study(monkeypatch, method, only_now):

    def forbidden(**kwargs):
        raise AssertionError("production called an offline quadrature study")

    monkeypatch.setattr(diag_fitting, "real_model_quadrature_study", forbidden)
    built = _install_fitting_library_fakes(monkeypatch)
    config = load_analysis_config(Path(__file__).with_name("fixtures") / "analysis_schema14.yaml")
    theory, observable, _, rule = _build_bgs_likelihood(
        config,
        _bgs_geometry_artifact(config),
        pairwise_sigma_mpc_h=LARGEST_PAIRWISE_SIGMA,
        method=method,
        only_now=only_now,
    )
    assert len(built.power_models) == 1
    assert theory.pt.mu.size == rule["mu_quadrature_order"] == 128
    assert theory.kwargs["template"].kwargs["only_now"] is only_now
    assert observable.wmatrix.ellsin == (0, 2, 4)


def test_per_bin_power3_likelihood_uses_the_observed_space_repair(monkeypatch):
    """Schema 16/17 keep power3, so their likelihood must carry the eight alobs coordinates."""
    _install_fitting_library_fakes(monkeypatch)
    config = load_analysis_config()
    assert config["schema_version"] > 15 and config["fitting"]["broadband_model"] == "power3"
    surface = config["fitting"]["fit_surfaces"]["BGS_BRIGHT-21.5"]
    coordinates = config["fitting"]["linear_nuisance"]["allowed_basenames_by_model"]["power3"]
    raw = {f"al{ell}_{power}" for ell in surface["model_multipoles"] for power in (-2, -1, 0)}
    calls = []

    def repaired(factory, kwargs, *, model):
        calls.append(model)
        observable = factory(**kwargs)
        observable.power_broadband_repair = dict(
            full_mu_support=True,
            null_names=["al4_0"],
            raw_names=sorted(raw - {"al4_0"}),
            coordinate_names=list(coordinates),
            effective_rank=len(coordinates),
        )
        return observable

    monkeypatch.setattr(fitting_likelihood, "repaired_observable", repaired)
    _, observable, _, _ = _build_bgs_likelihood(
        config, _bgs_geometry_artifact(config), pairwise_sigma_mpc_h=3.0
    )
    assert calls == ["power3"]
    assert observable.power_broadband_repair["coordinate_names"] == coordinates


def test_linear_likelihood_refuses_a_library_that_changes_the_fixed_rule(monkeypatch):
    _install_fitting_library_fakes(monkeypatch, realized_order=64)
    config = load_analysis_config(Path(__file__).with_name("fixtures") / "analysis_schema14.yaml")
    with pytest.raises(RuntimeError, match="changed the fixed quadrature"):
        _build_bgs_likelihood(config, _bgs_geometry_artifact(config), pairwise_sigma_mpc_h=3.0)


@pytest.mark.parametrize("interrupt_after", ["ensemble", "history"])
def test_linear_native_sampler_resume_preserves_streams_budget_and_common_checks(
    tmp_path, monkeypatch, interrupt_after
):
    pytest.importorskip("desilike")
    pytest.importorskip("emcee")
    config = load_analysis_config()
    config["fitting"]["sampling"].update(
        minimum_iterations=10, check_every=20, absolute_iteration_ceiling=65
    )
    route = {
        "tracer": "BGS_BRIGHT-21.5",
        "measurement_bin": "BGS1",
        "condition": "sigma_0.003",
        "reconstruction_state": "prereconstruction",
    }
    identity = {
        "run_name": "synthetic",
        "code_commit": "b" * 40,
        "analysis_config_sha256": "a" * 64,
    }

    def output(label):
        directory = tmp_path / label
        directory.mkdir(exist_ok=True)
        return {
            "chains": [directory / f"ensemble-{i}.npy" for i in range(4)],
            "summary": directory / "sampling.json",
        }

    monkeypatch.setattr(
        fitting_sampling,
        "ensemble_gelman_rubin",
        lambda *args, **kwargs: {
            "statistic": "max_generalized_covariance_ratio_minus_one",
            "usable": False,
            "value": None,
            "reason": "synthetic unusable check",
        },
    )
    baseline_output, resumed_output = (output("baseline"), output("resumed"))
    arguments = dict(config=config, ap_mode="qiso", route=route, run_identity=identity)
    baseline = fitting_sampling.run_sampling(
        _toy_qiso_likelihood(True), baseline_output, **arguments
    )
    save_checkpoint, save_json = (
        fitting_sampling._save_ensemble_checkpoint,
        fitting_sampling._replace_sampler_json,
    )
    interrupted = []

    def checkpoint(chain, path, sampler, ensemble_identity, stream):
        save_checkpoint(chain, path, sampler, ensemble_identity, stream)
        if interrupt_after == "ensemble" and stream["ensemble_index"] == 1 and (not interrupted):
            interrupted.append(True)
            raise OSError("simulated interruption between ensemble checkpoints")

    def checkpoint_json(path, payload):
        save_json(path, payload)
        if (
            interrupt_after == "history"
            and str(path).endswith("gr_history.json")
            and (not interrupted)
        ):
            interrupted.append(True)
            raise OSError("simulated interruption after the first common check")

    monkeypatch.setattr(fitting_sampling, "_save_ensemble_checkpoint", checkpoint)
    monkeypatch.setattr(fitting_sampling, "_replace_sampler_json", checkpoint_json)
    with pytest.raises(OSError, match="simulated interruption"):
        fitting_sampling.run_sampling(_toy_qiso_likelihood(True), resumed_output, **arguments)
    resumed = fitting_sampling.run_sampling(_toy_qiso_likelihood(True), resumed_output, **arguments)
    assert resumed["stop_reason"] == baseline["stop_reason"] == "budget_exhausted"
    assert [row["step"] for row in resumed["gr_history"]] == [20, 40, 60, 65]
    assert resumed["gr_history"] == baseline["gr_history"]
    for index, (left, right) in enumerate(zip(baseline_output["chains"], resumed_output["chains"])):
        expected = fitting_sampling.read_chain(
            left, identity={**identity, **route, "ensemble_index": index}
        )
        actual = fitting_sampling.read_chain(
            right, identity={**identity, **route, "ensemble_index": index}
        )
        np.testing.assert_array_equal(actual["draws"], expected["draws"])
        assert actual["draws"].shape[:2] == (40, 65)
        assert actual["attrs"]["random_stream"] == expected["attrs"]["random_stream"]
        for a, b in zip(actual["attrs"]["rng_state"], expected["attrs"]["rng_state"]):
            np.testing.assert_array_equal(a, b)
    again = fitting_sampling.run_sampling(_toy_qiso_likelihood(True), resumed_output, **arguments)
    assert again["gr_history"] == resumed["gr_history"]
