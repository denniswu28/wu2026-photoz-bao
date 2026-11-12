"""Fixed-prior start coverage and the saved native LRG3 two-basin regression."""

import pzbao.diagnostics.fitting as diag_fitting
import copy
from types import SimpleNamespace
import numpy as np
import pytest
import pzbao.fitting as fitting_fitting
from pzbao.config import load_analysis_config


def fake_profiler():

    class Prior:
        limits = (0.8, 1.2)

        def __getstate__(self):
            return {"limits": self.limits}

    class Profile:

        def __init__(self, value, valid=True):
            self.bestfit = SimpleNamespace(
                logposterior=np.array([value]),
                attrs={
                    "is_valid": valid,
                    "is_above_max_edm": False,
                    "has_reached_call_limit": False,
                },
            )
            self.attrs = {}

        def deepcopy(self):
            return copy.deepcopy(self)

    class Profiler:
        profiles = None
        varied_params = [
            SimpleNamespace(name=n, basename=n, prior=Prior()) for n in ("qiso", "qap")
        ]

        def _get_start(self, *, niterations):
            return np.ones((niterations, 2))

        def maximize(self, *, niterations, start):
            assert niterations == 1 and isinstance(start, list)
            first = float(start[0][0])
            return Profile(100 if first > 1.1 else 3 if first < 0.9 else 0, first <= 1.1)

    return Profiler()


def test_prior_coverage_is_deterministic_and_never_changes_priors_or_config():
    profiler, config = (fake_profiler(), load_analysis_config())
    original = copy.deepcopy(config)
    rows = diag_fitting.fixed_prior_starts(profiler, config, [])
    assert rows == diag_fitting.fixed_prior_starts(profiler, config, [])
    count = config["fitting"]["optimization"]["profiler_iterations"]
    assert len(rows) == 2 * count
    assert all((row["point"] == {"qiso": 1.0, "qap": 1.0} for row in rows[:count]))
    for name in ("qiso", "qap"):
        values = sorted((row["point"][name] for row in rows[count:]))
        np.testing.assert_allclose(values, 0.8 + 0.4 * (np.arange(count) + 0.5) / count)
    assert config == original
    assert all((p.prior.limits == (0.8, 1.2) for p in profiler.varied_params))


def test_best_valid_basin_keeps_all_start_flags_and_actual_native_selection():
    profiler, config = (fake_profiler(), load_analysis_config())
    original = copy.deepcopy(config)
    starts = [
        {"origin": label, "point": {"qiso": q, "qap": 1.0}}
        for label, q in (("interior", 1.0), ("better", 0.81), ("invalid", 1.19))
    ]
    best = diag_fitting.maximize_recorded(profiler, config, starts)
    assert profiler.profiles is best
    assert best.attrs["selected_start_index"] == 1
    assert best.bestfit.logposterior.item() == 3
    assert len(best.attrs["optimizer_starts"]) == 3
    assert not best.attrs["optimizer_starts"][2]["converged"]
    assert best.attrs["optimizer_starts"][1]["origin"] == "better"
    assert not best.attrs["optimizer_search"]["global_optimum_established"]
    assert config == original


@pytest.mark.parametrize(
    "point", [{"qiso": 0.7, "qap": 1.0}, {"qiso": 1.0}, {"qiso": np.nan, "qap": 1.0}]
)
def test_bad_start_fails_before_maximization(point):
    with pytest.raises(ValueError):
        diag_fitting.maximize_recorded(
            fake_profiler(), load_analysis_config(), [{"origin": "bad", "point": point}]
        )


def test_smooth_projection_removes_only_inactive_dilations():
    rows = [{"origin": "bao", "point": {"qiso": 0.8, "qap": 0.8, "b1": 2.0}}]
    result = diag_fitting.smooth_points(rows, ["b1"], ["qiso", "qap"])
    assert result == [{"origin": "bao", "point": {"b1": 2.0}}]
    assert rows[0]["point"]["qiso"] == 0.8
    with pytest.raises(ValueError, match="beyond inactive"):
        diag_fitting.smooth_points(rows, ["b1", "sigmas"], ["qiso", "qap"])
    with pytest.raises(ValueError, match="exclude inactive"):
        diag_fitting.fixed_prior_starts(fake_profiler(), load_analysis_config(), [], smooth=True)


def test_native_writer_selector_can_delegate_without_recursion(monkeypatch):
    profiler, config = (fake_profiler(), load_analysis_config())
    native = fitting_fitting.maximize_usable
    starts = [{"origin": "better", "point": {"qiso": 0.81, "qap": 1.0}}]
    monkeypatch.setattr(
        fitting_fitting, "maximize_usable", lambda *args: pytest.fail("recursive selector")
    )
    best = diag_fitting.maximize_recorded(profiler, config, starts, selector=native)
    assert best.bestfit.logposterior.item() == 3
