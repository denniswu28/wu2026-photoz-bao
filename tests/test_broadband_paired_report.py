"""Fixed-point acoustic subtraction, paired inputs, native replay and export checks."""

import pzbao.diagnostics.fitting as diag_fitting
import copy
import fnmatch
import numpy as np
import pytest


class Parameter:

    def __init__(self, name, value, *, fixed=False, derived=False):
        self.name, self.value, self.fixed, self.derived = (name, value, fixed, derived)

    @property
    def solved(self):
        return str(self.derived).startswith(".")

    def update(self, **kwargs):
        self.__dict__.update(kwargs)


class Parameters(list):

    def deepcopy(self):
        return copy.deepcopy(self)

    def select(self, *, basename):
        return [p for p in self if any((fnmatch.fnmatch(p.name, pattern) for pattern in basename))]


class FixedModel:
    """Reject calls if even a single nuisance was left analytically solvable."""

    def __init__(self, smooth=False):
        self.all_params = Parameters(
            [Parameter("qiso", 1.0, fixed=smooth), Parameter("a", 0.0, derived=".auto")]
        )
        self.smooth = smooth
        self.flatdata = np.array([2.0, 4.0, 3.0, 2.0])

    def __call__(self, **values):
        assert all((p.fixed and (not p.solved) for p in self.all_params))
        self.flattheory = np.array([1.0, 1.0, 1.0, 1.0]) + values["a"] * np.array(
            [1.0, 2.0, 3.0, 4.0]
        )
        if not self.smooth:
            self.flattheory += values["qiso"] * np.array([0.0, 0.5, -0.2, 0.0])


def test_fixed_fit_subtraction_preserves_point_and_correlated_residual():
    full, smooth = (FixedModel(), FixedModel(True))
    covariance = np.eye(4) + 0.2 * np.ones((4, 4))
    point = {"qiso": 0.8, "a": 0.3}
    parts = diag_fitting.decompose(full, smooth, point, ["a"], covariance)
    np.testing.assert_allclose(parts["smooth_template"], np.ones(4))
    np.testing.assert_allclose(parts["broadband"], [0.3, 0.6, 0.9, 1.2])
    np.testing.assert_allclose(parts["bao_template"], [0.0, 0.4, -0.16, 0.0])
    np.testing.assert_allclose(
        parts["bao_data"] - parts["bao_template"], full.flatdata - parts["model"]
    )
    residual = full.flatdata - parts["model"]
    white = np.linalg.solve(np.linalg.cholesky(covariance), residual)
    assert white @ white == pytest.approx(residual @ np.linalg.solve(covariance, residual))
    assert point == {"qiso": 0.8, "a": 0.3}


def test_smooth_toggle_cannot_change_the_nuisance_basis():

    class DifferentBasis(FixedModel):

        def __call__(self, **values):
            super().__call__(**values)
            self.flattheory += values["a"]

    with pytest.raises(AssertionError, match="nuisance contribution"):
        diag_fitting.decompose(
            FixedModel(), DifferentBasis(True), {"qiso": 0.8, "a": 0.3}, ["a"], np.eye(4)
        )


@pytest.mark.parametrize("point", [{"qiso": 1.0}, {"qiso": 1.0, "a": np.nan}, {"a": 0.0}])
def test_missing_or_invalid_fitted_values_fail_before_evaluation(point):
    with pytest.raises(ValueError, match="saved"):
        diag_fitting.decompose(FixedModel(), FixedModel(True), point, ["a"], np.eye(4))


def example_row():
    covariance = np.eye(4) + 0.2 * np.ones((4, 4))
    parts = diag_fitting.decompose(
        FixedModel(), FixedModel(True), {"qiso": 0.8, "a": 0.3}, ["a"], covariance
    )
    return {
        "physical_priors": {"qiso": {"limits": [0.8, 1.2]}},
        "metadata": {"damping_prior": {}},
        "fit_edges": np.array([40.0, 80.0, 120.0]),
        "multipoles": [0, 2],
        "separation": np.array([60.0, 100.0]),
        "covariance": covariance,
        "standard_deviation": np.sqrt(np.diag(covariance)),
        "components": parts,
        "residuals": {"data": np.array([2.0, 4.0, 3.0, 2.0]), "chi2": 3.0},
        "parameters": [
            {
                "basename": p,
                "value": 0.8,
                "minos_bounds": [0.79999962, 0.85],
                "interval_flags": {"is_valid": True, "at_lower_limit": True},
                "interval_unavailable_reason": None,
            }
            for p in ("qiso", "qap")
        ],
        "solution": "best verified repair",
        "dof": 2,
        "nuisance_count": 1,
    }


@pytest.mark.parametrize(
    "field", ["covariance", "separation", "fit_edges", "multipoles", "data", "physical_priors"]
)
def test_pair_rejects_incompatible_inputs(field):
    first, second = (example_row(), example_row())
    if field == "data":
        second["residuals"][field][0] += 0.1
    elif field == "physical_priors":
        second[field]["qiso"]["limits"][0] = 0.7
    else:
        second[field][0] += 1
    with pytest.raises((AssertionError, ValueError)):
        diag_fitting.validate_pair(first, second)
