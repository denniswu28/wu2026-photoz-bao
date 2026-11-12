"""Value, identity, multi-tier plotting, and native decomposition regression checks."""

import pzbao.diagnostics.fitting as diag_fitting
import copy
import numpy as np
import pytest


def source_checks(tasks, mocks):
    return [
        {
            "route": {
                "tracer": t["case"]["tracer"],
                "condition": t["case"]["condition"],
                "measurement_bin": t["measurement_bin"],
                "reconstruction_state": t["reconstruction_state"],
            },
            "vectors": len(mocks),
            "profiles": len(mocks) + 1,
            "saved_mean_model_reproduced": True,
        }
        for t in tasks
        if t["inference"] == "current"
    ]


def posterior(covariance):
    covariance = np.asarray(covariance)
    return {
        "parameters": ["qiso", "qap"],
        "covariance": covariance.tolist(),
        "standard_deviations_ddof1": np.sqrt(np.diag(covariance)).tolist(),
        "two_dimensional_fom": float(1 / np.sqrt(np.linalg.det(covariance))),
    }


def test_correlated_joint_fom_and_reciprocal_gain_definitions():
    pre = posterior([[0.04, 0.018], [0.018, 0.09]])
    post = posterior(np.asarray(pre["covariance"]) / 4)
    values = diag_fitting.precision_metrics(pre)
    assert values["precision_1d"]["qiso"] == 5
    assert values["joint_fom"] != pytest.approx(1 / (0.2 * 0.3))
    gain = diag_fitting.gain_metrics(pre, post)
    assert gain["uncertainty_post_over_pre"]["qiso"] == 0.5
    assert gain["precision_post_over_pre"]["qiso"] == 2
    assert gain["joint_fom_post_over_pre"] == pytest.approx(4)
    assert not diag_fitting.gain_metrics(None, post)["available"]
    changed = copy.deepcopy(pre)
    changed["standard_deviations_ddof1"][0] *= 2
    with pytest.raises(AssertionError):
        diag_fitting.precision_metrics(changed)
    changed = copy.deepcopy(pre)
    changed["parameters"].reverse()
    with pytest.raises(ValueError, match="coordinates"):
        diag_fitting.precision_metrics(changed)


def test_frozen_component_sum_and_rank_use_full_covariance():
    template = np.array([0.4, 0.6, 1.0])
    basis = np.array([[1.0, 0.0], [2.0, 1.0], [-1.0, 2.0]])
    coefficients = {"a": 0.3, "b": -0.2}
    covariance = np.array([[2.0, 0.5, 0.0], [0.5, 2.0, 0.3], [0.0, 0.3, 1.0]])

    def evaluate(values):
        return template + basis @ np.array([values["a"], values["b"]])

    result = diag_fitting.frozen_basis(
        evaluate,
        ["a", "b"],
        coefficients,
        covariance,
        evaluate(coefficients),
        rtol=1e-09,
        atol=1e-12,
    )
    np.testing.assert_allclose(result["template"], template)
    np.testing.assert_allclose(result["broadband"], basis @ [0.3, -0.2])
    np.testing.assert_allclose(result["basis"], basis)
    assert result["basis_rank"] == 2 and (not result["rank_deficient"])
    expected = np.linalg.svd(
        np.linalg.solve(np.linalg.cholesky(covariance), basis), compute_uv=False
    )
    np.testing.assert_allclose(result["whitened_basis_singular_values"], expected)


def test_component_check_rejects_refitting_or_nonlinear_nuisance():

    def nonlinear(values):
        return np.array([1 + values["a"] ** 2, 2 + values["a"]])

    with pytest.raises(AssertionError):
        diag_fitting.frozen_basis(
            nonlinear, ["a"], {"a": 0.3}, np.eye(2), nonlinear({"a": 0.3}), rtol=1e-09, atol=1e-12
        )


def test_nearly_null_column_with_large_saved_coefficient_preserves_curve():
    template = np.array([0.02, -0.01, 0.004])
    basis = np.array([[1.0, 1e-18], [-0.3, -2e-18], [0.2, 3e-18]])
    coefficients = {"ordinary": 0.001, "nearly_null": 50000000000000.0}

    def evaluate(values):
        return template + basis @ np.array([values[name] for name in coefficients])

    expected = evaluate(coefficients)
    unit_basis = np.column_stack(
        [
            evaluate({key: float(key == name) for key in coefficients}) - template
            for name in coefficients
        ]
    )
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(
            template + unit_basis @ list(coefficients.values()), expected, rtol=1e-08, atol=1e-10
        )
    result = diag_fitting.frozen_basis(
        evaluate, list(coefficients), coefficients, np.eye(3), expected, rtol=1e-08, atol=1e-10
    )
    np.testing.assert_allclose(result["model"], expected, rtol=1e-08, atol=1e-10)
    np.testing.assert_array_equal(result["coefficients"], [0.001, 50000000000000.0])
    assert result["basis_rank"] == 1 and result["rank_deficient"]
    np.testing.assert_array_equal(result["basis_evaluation_scales"], [1.0, 50000000000000.0])


def test_null_basis_direction_is_reported_without_removal():

    def evaluate(values):
        return np.array([1.0, 2.0]) + values["a"] * np.array([1.0, 3.0])

    result = diag_fitting.frozen_basis(
        evaluate,
        ["a", "b"],
        {"a": 0.2, "b": 3.0},
        np.eye(2),
        evaluate({"a": 0.2, "b": 3.0}),
        rtol=1e-09,
        atol=1e-12,
    )
    assert result["basis_rank"] == 1
    assert result["basis_columns"] == 2 and result["rank_deficient"]
    assert result["basis_condition_number"] is None


def test_full_posterior_report_cannot_publish_a_partial_inventory():
    records = {"ready": {"posterior": {}, "sample_status": "complete"}}
    diag_fitting.require_complete_posteriors(records)
    records["missing"] = {"posterior": None, "sample_status": "missing"}
    with pytest.raises(ValueError, match="all configured posteriors.*missing"):
        diag_fitting.require_complete_posteriors(records)


def test_objective_conventions_are_explicit_and_reject_wrong_treatment():
    from types import SimpleNamespace

    profile = SimpleNamespace(all_params=[SimpleNamespace(solved=True, derived=".auto")])
    marginalized = SimpleNamespace(all_params=[SimpleNamespace(solved=True, derived=".prec")])
    diag_fitting.set_objective_conventions(profile, marginalized)
    assert profile.solved_default == ".best"
    assert marginalized.solved_default == ".marg"
    marginalized.all_params[0].derived = ".auto"
    with pytest.raises(ValueError, match="unexpected solved nuisance convention"):
        diag_fitting.set_objective_conventions(profile, marginalized)
