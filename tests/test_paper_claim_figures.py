"""Value and serialization contracts for the read-only claim figures."""

import itertools
import json

import numpy as np
import pytest

from pzbao.figures import data as figures
from pzbao.figures import data as comparisons


def parameter(value=1.0, width=0.02):
    return {
        "value": value,
        "minos_bounds": [value - width, value + width],
        "interval_flags": {
            "is_valid": True,
            "lower_valid": True,
            "upper_valid": True,
            "at_lower_limit": False,
            "at_upper_limit": False,
        },
    }


def maximization():
    return {"is_valid": True, "is_above_max_edm": False, "has_reached_call_limit": False}


def row(mock=0, state="prereconstruction"):
    p = parameter(1 + 0.001 * mock, 0.02 if state == figures.STATES[0] else 0.01)
    return {
        "measurement_bin": "LRG2",
        "condition": "sigma_0.003",
        "state": state,
        "mock": mock,
        "parameter": "qiso",
        **p,
        **figures.interval(p, maximization()),
    }


def test_pairing_retains_invalid_diagnostics_and_excludes_invalid_gains():
    rows = [row(m, s) for m, s in itertools.product(range(3), figures.STATES)]
    bad = rows[-1]
    bad["interval_flags"]["at_lower_limit"] = True
    bad.update(figures.interval(bad, maximization()))
    summary, pairs = figures.summarize_recovery(rows, ["sigma_0.003"])
    assert len(pairs) == 3
    assert pairs[0]["profile_width_gain_percent"] == pytest.approx(100)
    assert pairs[-1]["profile_width_gain_percent"] is None
    assert pairs[-1]["post_boundary"] is True
    assert pairs[-1]["post_minus_pre"] == pytest.approx(0)
    post = next(
        r
        for r in summary
        if (r["measurement_bin"], r["parameter"], r["state"]) == ("LRG2", "qiso", figures.STATES[1])
    )
    assert post["n_bestfit"] == 3
    assert post["n_usable_interval"] == 2
    assert post["bias"] == pytest.approx(0.001)
    assert post["bias_sem"] == pytest.approx(0.001 / np.sqrt(3))
    assert next(r for r in summary if r["measurement_bin"] == "BGS1")["bias"] is None


@pytest.mark.parametrize("flag", ["is_valid", "lower_valid", "upper_valid"])
def test_invalid_finite_interval_is_preserved(flag):
    p = parameter()
    p["interval_flags"][flag] = False
    r = figures.interval(p, maximization())
    assert r["half_width"] == pytest.approx(0.02)
    assert r["ordered_bounds"] and not r["usable"]
    assert r["minos_bounds"] == [0.98, 1.02]


def test_unknown_optimizer_is_not_usable():
    assert not figures.interval(parameter(), {})["usable"]


def test_reversed_bounds_and_nonfinite_values():
    p = parameter()
    p["minos_bounds"] = [1.02, 0.98]
    r = figures.interval(p, maximization())
    assert r["half_width"] is None and not r["usable"]
    p["value"] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        figures.interval(p, maximization())


def test_full_robustness_grid_has_no_invented_fit():
    r = {
        "measurement_bin": "LRG2",
        "parameter": "qiso",
        "variant": "baseline",
        "status": "available",
        **figures.interval(parameter(), maximization()),
    }
    grid = figures.robustness_grid([r])
    assert len(grid) == 5 * 2 * 8
    assert sum(x["status"] == "missing" for x in grid) == 79
    assert all("value" not in x for x in grid if x["status"] == "missing")
    assert "r25" in {x["variant"] for x in grid}
    assert "r10" not in {x["variant"] for x in grid}
    with pytest.raises(ValueError, match="duplicate"):
        figures.robustness_grid([r, r])


def test_source_mutation_and_wrong_digest_fail(tmp_path):
    path = tmp_path / "input.json"
    path.write_text('{"value": 2}')
    source = figures.Sources()
    assert source.json(path) == {"value": 2}
    with pytest.raises(ValueError, match="changed"):
        source.read(path, "0" * 64)
    path.write_text('{"value": 3}')
    with pytest.raises(ValueError, match="changed"):
        source.unchanged()


def test_native_identity_requires_commit_and_config_digest():
    with pytest.raises(ValueError, match="identity"):
        figures.require_identity(
            {"run_name": "test", "code_commit": "producer", "analysis_config_sha256": "config"}
        )


def test_selection_normalization_uses_cell_widths():
    counts, edges = [2, 6], [0.1, 0.2, 0.5]
    pdf = figures.normalized_density(counts, edges)
    np.testing.assert_allclose(pdf, [2.5, 2.5])
    assert np.sum(pdf * np.diff(edges)) == pytest.approx(1)
    with pytest.raises(ValueError, match="counts"):
        figures.normalized_density([0, 0], edges)


def test_area_scaling_does_not_invent_relative_reconstruction_gain():
    full = 4 * np.pi * (180 / np.pi) ** 2
    row = figures.area_projection(full / 8, 0.5)
    assert row["area_factor"] == pytest.approx(4)
    assert row["sigma_factor"] == pytest.approx(0.5)
    assert row["gaussian_likelihood_fom_factor"] == pytest.approx(4)
    assert row["relative_reconstruction_gain_factor"] == 1


def test_serialized_flags_and_missing_values_roundtrip(tmp_path):
    p = parameter()
    p["interval_flags"]["upper_valid"] = False
    r = {
        "measurement_bin": "LRG2",
        "parameter": "qiso",
        "variant": "baseline",
        "status": "available",
        **figures.interval(p, maximization()),
    }
    rows = figures.robustness_grid([r])
    figures.write_json(tmp_path / "rows.json", rows)
    figures.write_csv(tmp_path / "rows.csv", rows)
    restored = json.loads((tmp_path / "rows.json").read_text())
    assert restored == rows
    assert '"upper_valid": false' in (tmp_path / "rows.json").read_text()
    assert "missing" in (tmp_path / "rows.csv").read_text()


def test_plot_outputs_keep_all_requested_panels(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    rows = figures.robustness_grid([])
    from tests.notebook_helpers import notebook_namespace

    namespace = notebook_namespace(
        "comparisons", cells={"setup", "comparison-inputs", "robustness"}
    )
    namespace["claim_figures_plot_robustness"](rows)
    assert set(namespace["figures"]) == {"robustness-qiso", "robustness-qap"}
    for fig in namespace["figures"].values():
        assert len(fig.axes) == 5


def test_fom_percent_gain_keeps_missing_and_flagged_pairs():
    rows = []
    for state, value in zip(figures.STATES, (200, 250)):
        rows.append(
            {
                "route": {
                    "measurement_bin": "LRG2",
                    "condition": "sigma_0.003",
                    "reconstruction_state": state,
                },
                "status": "available",
                "fom": value,
                "profile_flagged": state == figures.STATES[1],
                "sampling_limits": ["ESS unavailable"],
                "covariance_validity": {},
            }
        )
    pairs = comparisons.fom_pairs(rows, ["sigma_0.003"])
    pair = next(p for p in pairs if p["measurement_bin"] == "LRG2")
    assert pair["pre"] == 200 and pair["post"] == 250
    assert pair["gain_percent"] == pytest.approx(25)
    assert pair["profile_flagged"] and pair["sampling_limited"]
    assert pairs[0]["gain_percent"] is None
    with pytest.raises(ValueError, match="duplicate"):
        comparisons.fom_pairs(rows + rows, ["sigma_0.003"])
    rows[0]["fom"] = -1
    with pytest.raises(ValueError, match="invalid absolute"):
        comparisons.fom_pairs(rows, ["sigma_0.003"])


