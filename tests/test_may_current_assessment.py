"""Scientific controls for the six-comparison diagnostic, using small actual catalogues."""

import json

import pzbao.diagnostics.fitting as diag_fitting
import pzbao.diagnostics.reconstruction as diag_reconstruction
import pzbao.diagnostics.selection as diag_selection
import numpy as np
import pytest
from pzbao.config import load_analysis_config
from pzbao.selection import thin_randoms_to_smooth_target


def test_empirical_radial_shortage_fails_without_cloning():
    config = load_analysis_config()
    rank = config["selection"]["stable_randomness"]["random_thinning_rank_column"]
    randoms = {
        "RA": np.arange(150, dtype=float),
        "DEC": np.zeros(150),
        "Z": np.repeat([0.45, 0.55], [50, 100]),
        rank: np.arange(150) / 150,
    }
    artifact = {
        "sample_identity": "test",
        "z_edges": [0.4, 0.5, 0.6],
        "target_probability_mass": [2 / 3, 1 / 3],
    }
    with pytest.raises(ValueError, match="support bin"):
        thin_randoms_to_smooth_target(
            randoms,
            artifact,
            data_count=3,
            exact_random_data_ratio=50,
            expected_sample_identity="test",
            config=config,
        )


def test_requested_fkp_changes_only_weights():
    config = load_analysis_config()
    data = {"Z": np.array([0.15, 0.3]), "SOURCE_ID": np.array([9, 4]), "WEIGHT_FKP": np.ones(2)}
    result = diag_selection.requested_fkp(
        data,
        tracer="BGS_BRIGHT-21.5",
        condition="sigma_0.03",
        redshift_column="Z",
        config=config,
        p0=10000,
    )
    np.testing.assert_allclose(result["WEIGHT_FKP"], [1 / 135, 1 / 86.7])
    np.testing.assert_array_equal(result["SOURCE_ID"], data["SOURCE_ID"])
    np.testing.assert_array_equal(data["WEIGHT_FKP"], np.ones(2))


def test_bin_fields_preserve_rows_geometry_and_absolute_floor():
    data = {
        "Z": np.array([0.6, 0.4, 0.8]),
        "Z_OBS": np.array([0.6, 0.4, 0.8]),
        "WEIGHT": np.array([2.0, 1.0, 4.0]),
        "SOURCE_ID": np.array([30, 10, 20]),
    }
    bins = [{"name": "a", "zmin": 0.4, "zmax": 0.6}, {"name": "b", "zmin": 0.6, "zmax": 1.0}]
    seen = []

    def solver(galaxies, randoms, **kwargs):
        seen.append(kwargs)
        assert kwargs["common_geometry"] == {"nmesh": [8, 10, 12]}
        assert kwargs["bias"] == 2 and kwargs["growth_rate"] == 0.77
        assert kwargs["threshold_randoms"] * randoms["WEIGHT"].mean() == pytest.approx(
            0.01 * data["WEIGHT"].mean()
        )
        outputs = []
        for c in (galaxies, randoms):
            outputs.append(
                {
                    **c,
                    **{
                        key: c["SOURCE_ID"] + offset
                        for offset, key in enumerate(("X_SHIFTED", "Y_SHIFTED", "Z_CART_SHIFTED"))
                    },
                }
            )
        return (*outputs, {"count": len(galaxies["Z"])})

    result, randoms, record = diag_reconstruction.reconstruct_bins(
        data,
        data,
        solver=solver,
        kwargs={
            "threshold_randoms": 0.01,
            "common_geometry": {"nmesh": [8, 10, 12]},
            "bias": 2,
            "growth_rate": 0.77,
        },
        bins=bins,
        observed_column="Z_OBS",
        weight=lambda c: c["WEIGHT"],
    )
    np.testing.assert_array_equal(result["SOURCE_ID"], data["SOURCE_ID"])
    np.testing.assert_array_equal(result["X_SHIFTED"], data["SOURCE_ID"])
    assert len(seen) == 2 and record["scope"] == "analysis_bins"
    assert "X_SHIFTED" not in data


def test_shift_reuse_joins_identity_and_rejects_position_changes(tmp_path):
    path = tmp_path / "shifts.npz"
    np.savez(
        path,
        source_ids=[2, 1],
        positions=[[2, 0, 0], [1, 0, 0]],
        shifted_positions=[[3, 0, 0], [2, 0, 0]],
    )
    data = {"ID": np.array([1, 2]), "X": np.array([1, 2]), "Y": np.zeros(2), "Z_CART": np.zeros(2)}
    result = diag_reconstruction.apply_saved_shifts(data, [path], "ID")
    np.testing.assert_array_equal(result["X_SHIFTED"], [2, 3])
    data["X"][0] = 7
    with pytest.raises(AssertionError):
        diag_reconstruction.apply_saved_shifts(data, [path], "ID")


def test_paired_statistic_and_interaction_are_not_unpaired_errors():
    result = diag_fitting.paired_summary([1, 100, 300], [2, 101, 301])
    assert result["mean_difference"] == 1 and result["paired_mean_standard_error"] == 0
    np.testing.assert_array_equal(diag_fitting.factorial([1, 2], [3, 4], [5, 6], [7, 9]), [0, 1])
    with pytest.raises(ValueError, match="nonfinite"):
        diag_fitting.paired_summary([1, 2], [1, np.nan])


def _fit_record(mock, value):
    return {
        "mock": mock,
        "status": "complete",
        "optimizer": {"maximization": {"converged": True}},
        "chi2": 10.0,
        "detection": {"delta_chi2": 4.0},
        "parameters": [
            {
                "basename": name,
                "fixed": False,
                "value": value,
                "interval_flags": {"is_valid": True, "at_lower_limit": True},
                "parabolic_error": 0.02,
                "minos_bounds": [value - 0.02, value + 0.02],
            }
            for name in ("qiso", "qap")
        ],
    }


def test_bias_uncertainty_survives_serialization_and_keeps_boundary_fits(tmp_path):
    records = [_fit_record(mock, value) for mock, value in enumerate([0.98, 1.0, 1.04])]
    result = diag_fitting.fit_metrics(records, mock_ids=[0, 1, 2])
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(result, allow_nan=False))
    saved = json.loads(path.read_text())
    for name in ("qiso", "qap"):
        assert saved[name]["mean_offset_from_truth"] == pytest.approx(0.02 / 3)
        assert saved[name]["mock_scatter_ddof1"] == pytest.approx(np.sqrt(0.0028 / 3))
        assert saved[name]["mean_offset_standard_error"] == pytest.approx(np.sqrt(0.0028) / 3)
        assert saved[name]["boundary_intervals"] == 3
        assert saved[name]["minos_bounds"] == [p["parameters"][0]["minos_bounds"] for p in records]


def test_missing_and_failed_cohort_members_do_not_become_bias_measurements():
    records = [_fit_record(0, 1.1), {"mock": 1, "status": "failed"}]
    result = diag_fitting.fit_metrics(records, mock_ids=[0, 1, 2])
    assert (result["n"], result["required"]) == (1, 3)
    assert result["missing_profiles"] == result["failed_profiles"] == 1
    assert "qiso" not in result
    assert "qap" not in result
    assert len(records) == 2


def test_single_fit_has_no_estimated_ensemble_scatter_or_standard_error():
    result = diag_fitting.fit_metrics([_fit_record(0, 1.1)], mock_ids=[0])
    assert result["qiso"]["mean_offset_from_truth"] == pytest.approx(0.1)
    assert result["qiso"]["mock_scatter_ddof1"] is None
    assert result["qiso"]["mean_offset_standard_error"] is None
    json.dumps(result, allow_nan=False)


def test_cohort_identity_and_nonfinite_values_fail_closed():
    record = _fit_record(0, 1.0)
    with pytest.raises(ValueError, match="duplicate mock identity"):
        diag_fitting.fit_metrics([record, record])
    with pytest.raises(ValueError, match="duplicate requested"):
        diag_fitting.fit_metrics([record], mock_ids=[0, 0])
    with pytest.raises(ValueError, match="unexpected"):
        diag_fitting.fit_metrics([record], mock_ids=[1])
    with pytest.raises(ValueError, match="nonfinite"):
        diag_fitting.fit_metrics([_fit_record(0, np.nan)])
    with pytest.raises(ValueError, match="duplicate requested"):
        diag_fitting.individual_values({"fits": [record]}, "qiso", [0, 0])


def test_paired_values_are_aligned_by_identity_not_record_order():
    first = [_fit_record(0, 1.0), _fit_record(1, 1.04)]
    second = [_fit_record(1, 1.05), _fit_record(0, 1.01)]
    left = diag_fitting.individual_values({"fits": first}, "qiso", [0, 1])
    right = diag_fitting.individual_values({"fits": second}, "qiso", [0, 1])
    result = diag_fitting.paired_summary(left, right)
    assert result["mean_difference"] == pytest.approx(0.01)
    assert result["paired_mean_standard_error"] == pytest.approx(0.0, abs=1e-15)


def test_selection_overlap_checks_noise_and_rank_not_only_counts():
    left = {
        "source_ids": np.array([1, 2]),
        "ranks": np.array([0.2, 0.3]),
        "noise": np.array([-0.1, 0.5]),
        "rsd_redshift": np.array([0.5, 0.6]),
        "exterior": np.array([False, True]),
    }
    right = {key: value[::-1].copy() for key, value in left.items()}
    assert diag_selection.selection_pair(left, right)["common_objects"] == 2
    right["noise"][0] += 0.01
    with pytest.raises(AssertionError, match="noise"):
        diag_selection.selection_pair(left, right)
