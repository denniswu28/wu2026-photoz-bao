"""Preserved numerical reconstruction diagnostics, including direct native solver parity."""

import copy
import os
from types import SimpleNamespace

import numpy as np
import pytest

from pzbao.config import load_analysis_config
from pzbao.diagnostics import reconstruction as mesh


def design():
    scientific = load_analysis_config()["measurement"]["reconstruction"][
        "exploratory_mesh_comparison"
    ]
    return {
        **copy.deepcopy(scientific),
        "resources": {"screen_real_mesh_count": 12, "mesh_memory_screen_gib": 1},
    }


def test_physical_box_and_center_survive_rounding():
    geometry = {"boxsize_mpc_h": [3832, 5920, 3720], "boxcenter_mpc_h": [-925, -127, 923]}
    selector = lambda n: 2 ** np.ceil(np.log2(n)).astype(int)
    for case in design()["cases"]:
        out = mesh.mesh_geometry(geometry, case, selector)
        assert out["boxsize_mpc_h"] == geometry["boxsize_mpc_h"]
        assert out["boxcenter_mpc_h"] == geometry["boxcenter_mpc_h"]
        np.testing.assert_allclose(
            np.asarray(out["nmesh"]) * out["cellsize_mpc_h"], geometry["boxsize_mpc_h"]
        )
        if "nmesh" in case:
            assert out["nmesh"] == [512] * 3
            assert len(set(out["cellsize_mpc_h"])) == 3
        else:
            assert max(out["cellsize_mpc_h"]) <= case["maximum_cell_mpc_h"]


def test_bin_selection_uses_observed_data_and_original_random_redshift():
    calls = []

    def select(catalog, **kwargs):
        calls.append(kwargs["redshift_column"])
        keep = (catalog[kwargs["redshift_column"]] >= 0.6) & (
            catalog[kwargs["redshift_column"]] < 0.8
        )
        return {k: v[keep] for k, v in catalog.items()}

    pipeline = SimpleNamespace(stable_measurement_subset=select)
    config = {"selection": {"stable_randomness": {"observed_redshift_column": "Z_OBS"}}}
    catalog = {
        "Z": np.array([0.59, 0.7, 0.8]),
        "Z_OBS": np.array([0.61, 0.81, 0.79]),
        "SOURCE_ID": np.arange(3),
    }
    assert mesh.subset(catalog, "LRG2", "data", config, pipeline)["SOURCE_ID"].tolist() == [0, 2]
    assert mesh.subset(catalog, "LRG2", "randoms", config, pipeline)["SOURCE_ID"].tolist() == [1]
    assert calls == ["Z_OBS", "Z"]


@pytest.mark.skipif(
    not os.environ.get("SLURM_JOB_ID"), reason="native reconstruction runs only inside a Slurm compute job"
)
def test_native_multigrid_matches_direct_library():
    pytest.importorskip("pyrecon")
    from pyrecon import MultiGridReconstruction
    from pzbao import reconstruction as pipeline

    config = load_analysis_config()
    d = design()
    rng = np.random.default_rng(38)
    center = np.array([1000.0, 200.0, 300.0])
    positions = [center + rng.uniform(-40, 40, size=(n, 3)) for n in (120, 4000)]
    catalogs = [
        {**dict(zip(("X", "Y", "Z_CART"), pos.T)), "WEIGHT": np.ones(len(pos))} for pos in positions
    ]
    settings = {
        "growth_rate": 0.8,
        "bias": 2.0,
        "weight_expression": "WEIGHT",
        "resampler": "tsc",
        "fft_plan": "estimate",
        "mesh_dtype": "f8",
    }
    geometry = {"boxsize_mpc_h": [128.0] * 3, "boxcenter_mpc_h": center.tolist()}
    result = mesh.reconstruct_with_mesh(
        *catalogs,
        design=d,
        case={"nmesh": [32] * 3},
        geometry=geometry,
        settings=settings,
        config=config,
        pipeline=pipeline,
    )
    direct = MultiGridReconstruction(
        f=0.8,
        bias=2.0,
        nmesh=32,
        boxsize=128.0,
        boxcenter=center,
        resampler="tsc",
        fft_plan="estimate",
        dtype="f8",
        los=None,
        position_type="pos",
        threshold_randoms=0.01,
    )
    direct.assign_data(positions[0])
    direct.assign_randoms(positions[1])
    direct.set_density_contrast(smoothing_radius=15.0, check=False)
    direct.run(jacobi_damping_factor=0.4, jacobi_niterations=5, vcycle_niterations=6)
    for catalog, pos in zip(result[:2], positions):
        np.testing.assert_allclose(
            np.column_stack([catalog[k] for k in pipeline.SHIFTED_COLUMNS]),
            direct.read_shifted_positions(pos, field="disp+rsd"),
            rtol=1e-12,
            atol=1e-12,
        )
    assert result[2]["geometry"]["cellsize_mpc_h"] == [4.0] * 3
