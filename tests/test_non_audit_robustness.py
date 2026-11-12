"""Initialization, paired recovery, and matched numerical-depth contracts."""

import pzbao.diagnostics.fitting as diag_fitting
import pytest


def test_reference_change_keeps_prior_and_places_boundary_start_inside_support():
    pytest.importorskip("desilike")
    from desilike.parameter import ParameterCollection

    class Likelihood:
        all_params = ParameterCollection(
            {
                "qiso": {
                    "value": 1.0,
                    "prior": {"limits": [0.8, 1.2]},
                    "ref": {"limits": [0.99, 1.01]},
                    "fixed": False,
                }
            }
        )

        @property
        def varied_params(self):
            return self.all_params.select(varied=True)

    likelihood = Likelihood()
    before = likelihood.all_params["qiso"].prior.__getstate__()
    refs = diag_fitting.basin_references(likelihood, {"qiso": 0.8}, 0.1)
    assert likelihood.all_params["qiso"].prior.__getstate__() == before
    assert refs["qiso"]["limits"] == pytest.approx([0.8, 0.802])
    with pytest.raises(ValueError, match="outside"):
        diag_fitting.basin_references(likelihood, {"qiso": 0.79}, 0.1)
    with pytest.raises(ValueError, match="definition"):
        diag_fitting.basin_references(likelihood, {"qap": 1.0}, 0.1)
