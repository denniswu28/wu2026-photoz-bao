"""Numerical plots and executable notebooks for stage inspection."""

import json
from pathlib import Path
import numpy as np
import pytest

from pzbao.diagnostics import covariance, fitting


def test_plots_preserve_signed_values_and_joint_geometry():
    from pzbao.figures.style import axes

    matrix = np.array([[2.0, 0.3], [0.3, 1.0]])
    result = covariance.compare_matrices(matrix, 0.8 * matrix)
    np.testing.assert_allclose(result["generalized_eigenvalues"], [0.8, 0.8])
    np.testing.assert_allclose(result["difference"], -0.2 * matrix)
    ax = axes()
    reused, summary = fitting.plot_residuals([1.0, 2.0], [1.0, -2.0], [0.0, 0.0], matrix, ax=ax)
    assert reused is ax
    np.testing.assert_array_equal(summary["residual"], [1.0, -2.0])
    assert summary["chi2"] == pytest.approx(
        np.array([1.0, -2.0]) @ np.linalg.solve(matrix, [1.0, -2.0])
    )


@pytest.mark.parametrize(
    "stage",
    ["catalog", "selection", "reconstruction", "clustering", "covariance", "fitting", "sampling"],
)
def test_stage_notebook_executes_small_illustrative_cells(stage):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(__file__).resolve().parents[1] / "pzbao" / "diagnostics" / (stage + ".ipynb")
    notebook = json.loads(path.read_text())
    namespace = {"__name__": "__main__"}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            exec(compile("".join(cell["source"]), str(path), "exec"), namespace)
    assert namespace["RUN_ROOT"] is None
    assert "values" in namespace
    plt.close("all")
