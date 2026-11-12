"""Known coordinate shifts cross the saved-file and reporting boundaries."""

import numpy as np
import pytest

from pzbao.diagnostics import reconstruction as plots
from pzbao.diagnostics import fitting, sampling


def save(path, ids=None, before=None, after=None, redshift=None):
    before = (
        np.asarray([[3.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 5.0]])
        if before is None
        else before
    )
    after = before + [1.0, -2.0, 3.0] if after is None else after
    np.savez(
        path,
        source_ids=np.asarray([3, 1, 2] if ids is None else ids),
        redshift=np.asarray([0.15, 0.2, 0.25] if redshift is None else redshift),
        positions=before,
        shifted_positions=after,
    )


def test_signed_cartesian_and_local_los_shifts_from_saved_coordinates(tmp_path):
    path = tmp_path / "shifts.npz"
    save(path)
    values, fingerprint = plots.read_shifts(path, 0.1, 0.3)
    np.testing.assert_array_equal(values[:, :3], [[1, -2, 3]] * 3)
    np.testing.assert_array_equal(values[:, 3], [-2, 3, 1])
    np.testing.assert_allclose(values[:, 4], np.sqrt([10, 5, 13]))
    stats = plots.moments(values)
    assert stats["mean"][:3] == [1, -2, 3]
    assert stats["sd_ddof0"][:3] == [0, 0, 0]
    assert stats["rms"][:3] == [1, 2, 3]
    assert len(fingerprint) == 64
    # Bin edges use the original observed redshift, not reconstructed coordinates.
    selected, _ = plots.read_shifts(path, 0.15, 0.2)
    assert len(selected) == 1


def test_pairing_fingerprint_ignores_row_order_and_shift_but_not_original_position(tmp_path):
    first = tmp_path / "a.npz"
    save(first)
    with np.load(first) as saved:
        values = {k: saved[k][::-1] for k in saved.files}
    second = tmp_path / "b.npz"
    np.savez(second, **values)
    assert plots.read_shifts(first, 0.1, 0.3)[1] == plots.read_shifts(second, 0.1, 0.3)[1]
    values["shifted_positions"] = values["shifted_positions"] + 7
    np.savez(second, **values)
    assert plots.read_shifts(first, 0.1, 0.3)[1] == plots.read_shifts(second, 0.1, 0.3)[1]
    values["positions"] = values["positions"] + 1
    np.savez(second, **values)
    assert plots.read_shifts(first, 0.1, 0.3)[1] != plots.read_shifts(second, 0.1, 0.3)[1]


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_ids", [1, 1, 2]),
        ("positions", np.full((3, 3), np.nan)),
        ("shifted_positions", np.ones((3, 2))),
        ("positions", np.zeros((3, 3))),
    ],
)
def test_invalid_saved_shifts_fail_closed(tmp_path, field, value):
    path = tmp_path / "bad.npz"
    save(path)
    with np.load(path) as saved:
        data = dict(saved)
    data[field] = value
    np.savez(path, **data)
    with pytest.raises(ValueError):
        plots.read_shifts(path, 0.1, 0.3)


def test_histogram_retains_tails_and_last_edge():
    values = np.array([-10.0, -1.0, 0.0, 1.0, 10.0])
    result = plots.histogram(values, [-1.0, 0.0, 1.0])
    assert result["counts"] == [1, 2]
    assert result["underflow"] == result["overflow"] == 1
    assert sum(result["density"]) == pytest.approx(0.6)
    full = plots.histogram(values, [-10.0, 0.0, 10.0])
    assert full["underflow"] == full["overflow"] == 0
    assert np.dot(full["density"], [10.0, 10.0]) == pytest.approx(1.0)


def test_equal_mock_mean_does_not_galaxy_weight_realizations():
    rows = []
    for n, mean, density in ((10, -2.0, [1.0, 0.0]), (1000, 2.0, [0.0, 1.0])):
        rows.append(
            {
                "moments": {"count": n, "mean": [mean], "sd_ddof0": [1.0]},
                "histograms": [{"density": density}],
            }
        )
    density, mean, sd = plots.equal_mock_distribution(rows, 0)
    np.testing.assert_array_equal(density, [0.5, 0.5])
    assert mean == 0 and sd == pytest.approx(np.sqrt(5.0))
    assert plots.gaussian(np.arange(3), 0, 0) is None


def test_joint_fom_retains_correlation_and_rejects_fixed_ap():
    values = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]])
    assert sampling.joint_summary(values)["fom"] == pytest.approx(0.75)
    # Unit determinant shear changes correlation, not the joint area.
    changed = values @ np.array([[1.0, 1.0], [0.0, 1.0]])
    assert sampling.joint_summary(changed)["fom"] == pytest.approx(0.75)
    with pytest.raises(ValueError):
        sampling.joint_summary(np.column_stack((np.arange(4), np.ones(4))))
    with pytest.raises(ValueError):
        sampling.joint_summary(np.ones((4, 1)))


def test_boundary_flags_cannot_disappear_from_plot_classification():
    fits = [
        {
            "mock": None,
            "status": "complete",
            "parameters": [
                {
                    "basename": p,
                    "interval_flags": {
                        "is_valid": True,
                        "at_lower_limit": False,
                        "at_upper_limit": False,
                    },
                }
                for p in ("qiso", "qap")
            ],
        }
    ]
    assert not fitting.interval_flagged(fits)
    fits[0]["parameters"][1]["interval_flags"]["at_upper_limit"] = True
    assert fitting.interval_flagged(fits)
    assert fitting.interval_flagged([])
