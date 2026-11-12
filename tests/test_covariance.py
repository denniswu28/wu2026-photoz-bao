"""Separate-cap covariance values, native normalization, raw selection and numerical retries."""

import numpy as np
from numpy.polynomial.legendre import Legendre, leggauss
import pytest

from pzbao import covariance
from pzbao.config import load_analysis_config


def native_estimator(*, cap="NGC", post=False, radial_edges=None, mu_bins=8, identifier=0):
    """Small real pycorr estimator with unequal cap and DD/DR/RR weighted normalizations."""
    pytest.importorskip("pycorr")
    from pycorr.twopoint_counter import TwoPointCounter
    from pycorr.twopoint_estimator import LandySzalayTwoPointEstimator

    edges = np.asarray(radial_edges if radial_edges is not None else [48.0, 52.0, 56.0])
    mu_edges = np.linspace(-1.0, 1.0, mu_bins + 1)
    s, mu = np.meshgrid(
        (edges[:-1] + edges[1:]) / 2, (mu_edges[:-1] + mu_edges[1:]) / 2, indexing="ij"
    )
    base = (1 + 0.002 * s) * ((1 + 0.4 * mu**2) if cap == "NGC" else (3 - 1.5 * mu**2))
    xi = 0.04 + 0.01 * np.cos(s / 10) + 0.015 * mu**2 + 0.0001 * identifier
    if cap == "SGC":
        xi *= 0.7
    if post:
        xi *= 1.3
    norms = {"DD": 2500.0, "RR": 91000.0, "DR": 15100.0, "SS": 94000.0, "DS": 15200.0}
    if cap == "SGC":
        norms = {"DD": 400.0, "RR": 70200.0, "DR": 3700.0, "SS": 74100.0, "DS": 4300.0}

    def counter(values, norm):
        state = {
            "name": "corrfunc",
            "mode": "smu",
            "edges": [edges.copy(), mu_edges.copy()],
            "seps": [s.copy(), mu.copy()],
            "wcounts": np.asarray(values * norm),
            "ncounts": np.ones_like(values, dtype="i8"),
            "wnorm": norm,
            "size1": 100,
            "size2": 100,
            "autocorr": True,
            "is_reversible": True,
            "compute_sepsavg": [True, False],
            "bin_type": "custom",
            "boxsize": None,
            "los_type": "midpoint",
            "weight_attrs": {},
            "selection_attrs": {},
            "cos_twopoint_weights": None,
            "dtype": np.dtype("f8"),
            "attrs": {},
        }
        return TwoPointCounter.from_state(state)

    counts = {"D1D2": counter(base * (1 + xi), norms["DD"]), "R1R2": counter(base, norms["RR"])}
    if post:
        shifted = base * (1.1 + 0.05 * mu**2)
        cross = (base + shifted) / 2
        counts.update(S1S2=counter(shifted, norms["SS"]), D1S2=counter(cross, norms["DS"]))
    else:
        counts["D1R2"] = counter(base, norms["DR"])
    return LandySzalayTwoPointEstimator(**counts)


@pytest.mark.parametrize("post", [False, True])
def test_linear_native_cap_normalization_preserves_cap_xi_and_uses_unequal_dd(post, tmp_path):
    pytest.importorskip("pycorr")
    from pzbao.clustering import (
        combine_cap_pair_counts,
        bind_pair_count_identity,
        pair_count_identity,
    )
    from pycorr import TwoPointCorrelationFunction

    caps = {cap: native_estimator(cap=cap, post=post) for cap in ("NGC", "SGC")}
    normalized = {cap: estimator.normalize(wnorm="D1D2") for cap, estimator in caps.items()}
    for cap in caps:
        np.testing.assert_allclose(normalized[cap].corr, caps[cap].corr, rtol=1e-13, atol=1e-13)
        for name in caps[cap].count_names:
            assert float(getattr(normalized[cap], name).wnorm) == float(caps[cap].D1D2.wnorm)
    assert normalized["NGC"].D1D2.wnorm != normalized["SGC"].D1D2.wnorm
    expected = sum(value.corr * value.R1R2.wcounts for value in normalized.values()) / sum(
        value.R1R2.wcounts for value in normalized.values()
    )
    combined = combine_cap_pair_counts(caps)
    np.testing.assert_allclose(combined.corr, expected, rtol=1e-12, atol=1e-13)
    assert not np.allclose(combined.corr, (caps["NGC"].corr + caps["SGC"].corr) / 2)
    identity = {"cap": "NGC+SGC", "realization_id": 17, "condition": "sigma_0.003"}
    bind_pair_count_identity(combined, identity)
    path = tmp_path / "counts.npy"
    combined.save(str(path))
    loaded = TwoPointCorrelationFunction.load(str(path))
    assert pair_count_identity(loaded) == identity
    np.testing.assert_array_equal(loaded.R1R2.wcounts, combined.R1R2.wcounts)


def test_linear_cap_operator_matches_independent_bin_integrals_and_keeps_high_multipoles():
    mu_edges = np.linspace(0, 1, 9)
    mu = (mu_edges[:-1] + mu_edges[1:]) / 2
    rr = {
        "NGC": np.array([5 + mu**2, 9 + 2 * mu**2]),
        "SGC": np.array([4 - 3 * mu**2, 7 - 5 * mu**2]),
    }
    operators, weights = covariance.cap_multipole_operators(rr, mu_edges, multipoles=[0, 2, 4])
    nodes, quadrature_weights = leggauss(16)
    integrals = np.array(
        [
            [
                (hi - lo)
                / 2
                * np.dot(quadrature_weights, Legendre.basis(ell)((nodes + 1) * (hi - lo) / 2 + lo))
                for lo, hi in zip(mu_edges[:-1], mu_edges[1:])
            ]
            for ell in (0, 2, 4)
        ]
    )
    independent = {}
    for cap in rr:
        matrix = np.zeros((6, 6))
        for radial in range(2):
            for i, ell in enumerate((0, 2, 4)):
                for j in range(3):
                    matrix[2 * i + radial, 2 * j + radial] = np.sum(
                        (2 * ell + 1)
                        * integrals[i]
                        * rr[cap][radial]
                        / (rr["NGC"][radial] + rr["SGC"][radial])
                        * integrals[j]
                        / np.diff(mu_edges)
                    )
        independent[cap] = matrix
        np.testing.assert_allclose(operators[cap], matrix, atol=2e-14)
    components = {
        cap: {
            key: np.diag(np.arange(1.0, 7.0) * scale)
            for key, scale in (("c2", 1.0), ("c3", 2.0), ("c4", 3.0))
        }
        for cap in rr
    }
    select = np.eye(6)[:2]
    result = covariance.assemble_cap_components(components, operators, selection=select)
    expected = sum(
        select @ independent[cap] @ components[cap]["c4"] @ independent[cap].T @ select.T
        for cap in rr
    )
    np.testing.assert_allclose(result["c4"], expected, atol=1e-13)
    truncated = sum(
        (select @ independent[cap][:, :2])
        @ components[cap]["c4"][:2, :2]
        @ (select @ independent[cap][:, :2]).T
        for cap in rr
    )
    assert not np.allclose(expected, truncated, rtol=1e-7, atol=1e-10)
    np.testing.assert_allclose(weights["NGC"] + weights["SGC"], 1.0)


def test_linear_only_required_raw_blocks_must_be_finite():
    raw = {key: np.eye(6) for key in ("c2", "c3", "c4")}
    raw["c4"][5, 5] = np.nan
    result = covariance.project_raw_components(raw, np.eye(6)[:2])
    np.testing.assert_array_equal(result["c4"], np.eye(2))
    raw["c4"][0, 0] = np.nan
    with pytest.raises(covariance.NonFiniteCovariance, match="not finite"):
        covariance.project_raw_components(raw, np.eye(6)[:2])


def test_linear_scalar_objective_accepts_singular_sample_covariance():
    sample = np.diag([4.0, 0.0, 0.0])
    components = {"c2": np.eye(3), "c3": np.zeros((3, 3)), "c4": np.eye(3)}
    result = covariance.tune_shot_noise_rescaling(
        components, sample, search_limits=(0.2, 2.0), points=101
    )
    grid = np.linspace(0.2, 2.0, 101)
    objectives = 3 * np.log(1 + grid**2) + 4 / (1 + grid**2)
    assert result["shot_noise_rescaling"] == pytest.approx(grid[np.argmin(objectives)])
    assert result["sample_covariance_rank"] == 1
    np.testing.assert_allclose(result["objective_curve"], objectives)


def test_fresh_scalar_changes_with_mock_scatter_and_saves_invalid_candidates():
    components = {"c2": np.eye(2), "c3": np.zeros((2, 2)), "c4": -np.eye(2)}
    values = []
    for variance in (3.0, 8.0):
        result = covariance.tune_shot_noise_rescaling(
            components, variance * np.eye(2), search_limits=(0.5, 4.0), points=8
        )
        values.append(result["shot_noise_rescaling"])
        assert result["objective_curve"][:2] == [None, None]
        assert not result["solution_on_search_boundary"]
    assert values == [2.0, 3.0]


@pytest.mark.parametrize("dimension", [26, 52])
def test_configured_grid_recovers_analytic_minimum_with_rank_24_target(dimension):
    controls = covariance.fresh_shot_noise_controls(load_analysis_config())
    # The singular empirical target has mean variance five; C=(1+alpha**2)I therefore
    # has its continuous objective minimum at alpha=2 in either actual fitted dimension.
    sample = np.diag([5.0 * dimension / 24] * 24 + [0.0] * (dimension - 24))
    result = covariance.tune_shot_noise_rescaling(
        {"c2": np.eye(dimension), "c3": np.zeros_like(sample), "c4": np.eye(dimension)},
        sample,
        search_limits=controls["search_limits"],
        points=controls["search_points"],
    )
    grid = np.linspace(0.1, 10.0, 4096)
    expected = dimension * (np.log(1 + grid**2) + 5 / (1 + grid**2))
    assert result["shot_noise_rescaling"] == grid[np.argmin(expected)]
    assert result["sample_covariance_rank"] == 24
    assert result["evaluated_points"] == 4096
    assert not result["solution_on_search_boundary"]
    np.testing.assert_allclose(result["objective_curve"], expected, rtol=1e-13)


def test_no_positive_definite_scalar_uses_numerical_depth_ladder():
    config = load_analysis_config()
    depths = config["fitting"]["covariance"]["rascalc"]["escalation_ladder_n_loops"]
    attempted = []

    def attempt(depth, retry):
        attempted.append(depth)
        return {
            "c2": np.eye(2),
            "c3": np.zeros((2, 2)),
            "c4": np.eye(2) * (-100 if depth == depths[0] else 1),
        }

    def alpha(parts):
        return covariance.tune_shot_noise_rescaling(
            parts, np.eye(2), search_limits=(0.5, 2.0), points=4
        )["shot_noise_rescaling"]

    _, history = covariance.numerical_ladder(attempt, config=config, alpha=alpha)
    assert attempted == depths[:2]
    assert history == [
        {"n_loops": depths[0], "numerical_failure": "cholesky_failure"},
        {"n_loops": depths[1], "numerical_failure": None},
    ]


@pytest.mark.parametrize(
    "limits,points", [((0.1, np.inf), 5), ((0.1, 2.0), True), ((0.1, 2.0), 2.5), ((0.0, 2.0), 5)]
)
def test_scalar_search_rejects_invalid_numerical_controls(limits, points):
    with pytest.raises(ValueError, match="search interval"):
        covariance.tune_shot_noise_rescaling(
            {key: np.eye(2) for key in ("c2", "c3", "c4")},
            np.eye(2),
            search_limits=limits,
            points=points,
        )


def test_linear_numerical_ladder_does_not_spend_future_infrastructure_requeues():
    config = load_analysis_config()
    calls = []
    failure = TimeoutError("node unavailable")

    def attempt(depth, retry):
        calls.append((depth, retry))
        raise failure

    for _ in range(4):
        with pytest.raises(TimeoutError) as caught:
            covariance.numerical_ladder(attempt, config=config, alpha=1.8)
        assert caught.value is failure
    first = config["fitting"]["covariance"]["rascalc"]["escalation_ladder_n_loops"][0]
    assert calls == [(first, 0), (first, 1)] * 4


def test_linear_numerical_failure_alone_advances_depth():
    config = load_analysis_config()
    depths = config["fitting"]["covariance"]["rascalc"]["escalation_ladder_n_loops"]
    calls = []

    def attempt(depth, retry):
        calls.append(depth)
        return {
            "c2": np.eye(2),
            "c3": np.zeros((2, 2)),
            "c4": np.eye(2) * (-10 if depth == depths[0] else 2),
        }

    components, history = covariance.numerical_ladder(attempt, config=config, alpha=1.5)
    assert calls == depths[:2]
    assert history[0]["numerical_failure"] == "cholesky_failure"
    assert history[-1]["numerical_failure"] is None


@pytest.mark.parametrize(
    "failure,completed,recover",
    [
        (None, True, True),
        ("postprocess", True, True),
        ("postprocess", False, False),
        ("unrelated", True, False),
        ("timeout", True, False),
        ("incomplete", True, False),
    ],
)
def test_linear_fixed_alpha_raw_integration_only_recovers_completed_postprocessing(
    tmp_path, failure, completed, recover
):
    import inspect
    from pathlib import Path

    config = load_analysis_config()
    controls = config["fitting"]["covariance"]["rascalc"]
    estimator = native_estimator(radial_edges=np.arange(0.0, 204.0, 4.0), post=True)
    table = covariance.rascalc_native_table(estimator)
    observed = []

    def postprocess_failure():
        raise ValueError("optional provisional-alpha diagnostic failed")

    # Model the pinned library's actual stack boundary without running compiled integration.
    postprocess_failure.__code__ = postprocess_failure.__code__.replace(
        co_filename="/synthetic/RascalC/post_process/utils.py"
    )

    def run_cov(**kwargs):
        observed.append(kwargs)
        root = Path(kwargs["out_dir"])
        root.mkdir(parents=True)
        if completed:
            (root / "log.txt").write_text("The C++ code finished succesfully\n")
        matrix = np.eye(150)
        matrix[-1, -1] = np.nan  # Outside the fitted window; preserve raw diagnostic failure.
        raw = {}
        for key in ("c2_11", "c3_1,11", "c4_11,11"):
            raw[key + "_full"] = matrix
            raw[key] = np.repeat(
                matrix[None], kwargs["n_loops"] // kwargs["loops_per_sample"], axis=0
            )
        if failure == "incomplete":
            del raw["c3_1,11"]
        np.savez(root / "Raw_Covariance_Matrices_n50_l4.npz", **raw)
        if failure == "postprocess":
            postprocess_failure()
        if failure == "unrelated":
            raise ValueError("unrelated required construction failure")
        if failure == "timeout":
            raise TimeoutError("compiled integration timed out")

    run_cov.__signature__ = inspect.Signature(
        [
            inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY)
            for name in controls["public_arguments"]
        ]
    )
    # Even review references mistakenly present in ordinary defaults cannot switch modes.
    controls["run_arguments"].update(
        xi_11_samples="report-only-reference", xi_sample_cov="report-covariance"
    )
    arguments = dict(
        estimator=estimator,
        xi_table=table,
        random_positions=np.arange(90.0).reshape(30, 3),
        random_weights=np.ones(30),
        data_count=50,
        config=config,
        destination=tmp_path / "integration",
        nthread=1,
        n_loops=controls["escalation_ladder_n_loops"][0],
        alpha=1.8,
        run_cov=run_cov,
    )
    if recover:
        result = covariance.integrate_cap(**arguments)
        assert result["raw_integration_completed"] is True
        assert np.isnan(result["components"]["c4"][-1, -1])
        assert result["shot_noise_rescaling"] == 1.8
        assert bool(result["postprocessing_failure"]) == (failure == "postprocess")
    else:
        with pytest.raises((ValueError, OSError)):
            covariance.integrate_cap(**arguments)
    assert len(observed) == 1
    kwargs = observed[0]
    assert kwargs["xi_11_samples"] is None and kwargs["xi_sample_cov"] is None
    assert kwargs["shot_noise_rescaling1"] == 1.8
    assert kwargs["skip_s_bins"] == (12, 12) and kwargs["skip_l"] == 0
    assert kwargs["seed"] == controls["frozen_integration"]["seed"]
    assert kwargs["loops_per_sample"] == controls["frozen_integration"]["loops_per_sample"]


def test_linear_optional_covariance_diagnostics_do_not_veto_a_usable_required_matrix(monkeypatch):
    pytest.importorskip("RascalC")
    import RascalC.post_process.utils as native

    def undefined(*args, **kwargs):
        raise np.linalg.LinAlgError("undefined inversion-bias diagnostic")

    monkeypatch.setattr(native, "compute_D_precision_matrix", undefined)
    matrix = np.array([[3.0, 0.4], [0.4, 1.0]])
    result = covariance.covariance_diagnostics(matrix, np.repeat(matrix[None], 16, axis=0))
    np.testing.assert_allclose(result["eigenvalues"], np.linalg.eigvalsh(matrix))
    assert result["positive_definite"] is True
    assert result["optional"]["inversion_bias"]["value"] is None
    assert "undefined" in result["optional"]["inversion_bias"]["reason"]


def test_linear_incompatible_dimensions_are_not_numerical_depth_failures():
    config = load_analysis_config()
    calls = []

    def attempt(depth, retry):
        calls.append(depth)
        return {"c2": np.eye(2), "c3": np.eye(3), "c4": np.eye(2)}

    with pytest.raises(ValueError, match="different dimension"):
        covariance.numerical_ladder(attempt, config=config, alpha=1.8)
    assert len(calls) == 1


def test_stronger_only_failure_is_preserved_as_warning_without_deeper_integration():
    # The extrema occur in different modes: the actual sum is diag(2, 5).
    components = {
        "c2": np.diag([1.0, 10.0]),
        "c3": np.zeros((2, 2)),
        "c4": np.diag([1.0, -5.0]),
    }
    before = {key: value.copy() for key, value in components.items()}
    verdict = covariance.projected_structural_verdict(components, shot_noise_rescaling=1.0)
    assert verdict["structural_test_passes"] is True
    assert verdict["structural_test_failures"] == []
    assert verdict["stronger_eigenvalue_test_passes"] is False
    assert verdict["stronger_eigenvalue_margin"] == -4.0
    assert verdict["weaker_eigenvalue_margin"] == pytest.approx(0.5)
    assert len(verdict["structural_test_warnings"]) == 1
    config = load_analysis_config()
    assert covariance.numerical_failure(components, 1.0, config) is None
    calls = []

    def attempt(depth, retry):
        calls.append((depth, retry))
        return components

    recovered, history = covariance.numerical_ladder(attempt, config=config, alpha=1.0)
    assert len(calls) == len(history) == 1
    np.testing.assert_array_equal(covariance.combine_components(recovered, 1.0), np.diag([2, 5]))
    for key in before:
        np.testing.assert_array_equal(components[key], before[key])


@pytest.mark.parametrize(
    "c2,c3,c4,reason",
    [
        ([1.0, 1.0], [2.0, 0.0], [-2.0, 1.0], "rascalc_validity_failure"),
        ([1.0, 1.0], [1.0, 0.0], [-1.0, 1.0], "rascalc_validity_failure"),
        ([0.0, 1.0], [1.0, 0.0], [1.0, 1.0], "rascalc_validity_failure"),
        ([1.0, 1.0], [-3.0, 0.0], [1.0, 1.0], "cholesky_failure"),
        ([1.0, 1.0], [0.0, 0.0], [np.nan, 1.0], "non_finite_covariance"),
    ],
)
def test_required_covariance_failures_remain_blocking(c2, c3, c4, reason):
    components = {"c2": np.diag(c2), "c3": np.diag(c3), "c4": np.diag(c4)}
    assert covariance.numerical_failure(components, 1.0, load_analysis_config()) == reason
