"""Deterministic external engines with real eight-stage serialization and inference."""

import json
from pathlib import Path
from types import SimpleNamespace

from click.testing import CliRunner
import pytest

from pzbao import run as driver
from pzbao.config import load_analysis_config


@pytest.mark.parametrize("boundary_experiment", [False, True])
def test_linear_native_pilot_crosses_all_eight_saved_stage_boundaries(
    tmp_path, monkeypatch, boundary_experiment
):
    """Synthetic catalogues/engines, real preparation, native products, fits, sampling and report.

    No catalogue acquisition, FFT reconstruction, pair counting or Monte-Carlo integration runs.
    The external engines return deterministic numerical fixtures in their actual native formats;
    no whole stage, stage publisher, covariance assembler, fitter, sampler or report is replaced.
    """
    import csv
    import numpy as np

    pytest.importorskip("pycorr")
    pypower = pytest.importorskip("pypower")
    pytest.importorskip("desilike")
    pytest.importorskip("iminuit")
    pytest.importorskip("emcee")
    from desilike.base import BaseCalculator
    from desilike.likelihoods import BaseGaussianLikelihood
    from pzbao import (
        catalog,
        selection,
        reconstruction,
        clustering,
        covariance,
        likelihood,
        fitting,
    )
    from tests.test_covariance import native_estimator
    from tests.test_run import synthetic_scalar_grid

    synthetic_scalar_grid(monkeypatch)

    class Cosmology:
        rs_drag = 100.0

        def comoving_radial_distance(self, z):
            return 3000 * np.asarray(z)

        def growth_rate(self, z):
            return 0.5 + 0.3 * np.asarray(z)

        def efunc(self, z):
            return 1 + 0.5 * np.asarray(z)

        def get_fourier(self):
            return self

        def pk_interpolator(self, **kwargs):
            return lambda k, z: 1000 * np.exp(-np.asarray(k)) / (1 + z) ** 2

    monkeypatch.setattr(
        driver, "_code_identity", lambda: {"code_commit": "c" * 40, "repository": str(tmp_path)}
    )
    monkeypatch.setattr(selection, "build_mock_cosmology", lambda config: Cosmology())
    monkeypatch.setattr(likelihood, "build_mock_cosmology", lambda config: Cosmology())
    monkeypatch.setenv("BAO_COVARIANCE_PROCESS", "1")
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    monkeypatch.setattr(
        catalog,
        "register_parent_inputs",
        lambda root, tracer, mock_id, config: [
            {
                "tracer": tracer,
                "role": role,
                "mock_id": mock_id,
                "cap": cap,
                "random_split": 0 if role == "random" else None,
                "sha256": "a" * 64,
                "size_bytes": 1,
                "path": str(Path(root) / f"{tracer}-{mock_id}-{cap}-{role}"),
            }
            for cap in ("NGC", "SGC")
            for role in ("data", "random", "reservoir")
        ],
    )

    def raw_parent(records, *, tracer, mock_id, cap, config, include_randoms=True, parallel=None):
        field = config["tracers"][tracer]["field"]
        width = config["selection"]["radial_randoms"]["support_bin_width_redshift"]
        edges = np.arange(field["zmin"], field["zmax"] + 0.5 * width, width)
        rng = np.random.default_rng(7000 + mock_id + 100 * (cap == "SGC"))

        def catalogue(z, role):
            count = len(z)
            identifiers = np.arange(count, dtype="u8") + 1000000 * (
                1 + mock_id + 100 * (role == "random")
            )
            return {
                "SOURCE_ID": identifiers,
                "TARGETID": identifiers.copy(),
                "Z": z,
                # Survey-sized angular support is required by the actual P(k) geometry.
                "RA": rng.uniform(10, 170, count) + (180 if cap == "SGC" else 0),
                "DEC": rng.uniform(-70, 70, count),
                "WEIGHT": np.ones(count),
            }

        # Independent shell fluctuations make the ensemble radial variance measurable.
        counts = rng.poisson(3 if cap == "NGC" else 2, len(edges) - 1)
        data_z = np.repeat(edges[:-1], counts) + rng.uniform(0.05, 0.95, int(counts.sum())) * width
        result = {"data": catalogue(data_z, "data"), "reservoir": None}
        if boundary_experiment and tracer == "BGS_BRIGHT-21.5":
            from pzbao.selection import attach_stable_randomness

            exterior = catalogue(np.full(100, 0.099), "data")
            exterior["SOURCE_ID"] += 10**12
            exterior["TARGETID"] = exterior["SOURCE_ID"].copy()
            streams = attach_stable_randomness(exterior, role="data", config=config)
            index = np.flatnonzero(streams["PHOTOZ_DEVIATE"] > 1.0)[0]
            result["data"] = {
                key: np.append(value, exterior[key][index]) for key, value in result["data"].items()
            }
            # Even an empty reservoir must pass through the real merge so the
            # clustering exterior row produces the saved edge-support diagnostics.
            result["reservoir"] = {key: value[:0].copy() for key, value in result["data"].items()}
        if include_randoms:
            random_z = np.repeat(edges[:-1], 400) + np.tile(
                (np.arange(400) + 0.5) / 400 * width, len(edges) - 1
            )
            result["random"] = catalogue(random_z, "random")
        return result

    monkeypatch.setattr(catalog, "reservoir_slice_path", lambda *a, **k: Path(__file__))
    monkeypatch.setattr(catalog, "read_registered_catalogue", lambda *a, **k: {})
    monkeypatch.setattr(selection, "_read_parent", raw_parent)
    solves, integration_calls, constructors = [], [], []

    def reconstruct(data, randoms, **kwargs):
        solves.append((kwargs["mask_family"]["cap"], kwargs["growth_rate"]))
        geometry = kwargs["common_geometry"]
        for catalogue in (data, randoms):
            xyz = np.column_stack([catalogue[key] for key in ("X", "Y", "Z_CART")])
            assert np.all(
                np.abs(xyz - geometry["boxcenter_mpc_h"])
                < np.asarray(geometry["boxsize_mpc_h"]) / 2
            )
            for before, after in zip(("X", "Y", "Z_CART"), reconstruction.SHIFTED_COLUMNS):
                catalogue[after] = catalogue[before] + 0.25
        return (
            data,
            randoms,
            {
                "shell_diagnostics": {
                    "shells": [
                        {
                            "shell": 0,
                            "radial_range_mpc_h": [100.0, 3000.0],
                            "retained_supported_cell_fraction": 1.0,
                            "displacement_rms_mpc_h": np.sqrt(3) / 4,
                        }
                    ]
                }
            },
        )

    def counts(data, randoms, **kwargs):
        cap = "NGC" if data["RA"].mean() < 180 else "SGC"
        if kwargs["reconstructed"]:
            assert kwargs["precomputed_random_counts"] is not None
        return native_estimator(
            cap=cap,
            post=kwargs["reconstructed"],
            identifier=int(data["SOURCE_ID"][0] // 1000000) - 1,
            radial_edges=kwargs["s_edges_mpc_h"],
            mu_bins=kwargs["mu_bins"],
        )

    def power(data, randoms, *, reconstructed, geometry_artifact, config):
        edges = np.linspace(0.0, 0.4, 81)
        k = (edges[:-1] + edges[1:]) / 2
        norm = 100.0 if data["RA"].mean() < 180 else 20.0
        values = np.array([(1 + ell) * np.exp(-k) for ell in (0, 2, 4)])
        return pypower.PowerSpectrumMultipoles(
            edges=[edges],
            modes=[k],
            power_nonorm=norm * values * (1.1 if reconstructed else 1.0),
            nmodes=np.full(len(k), 100.0),
            ells=(0, 2, 4),
            wnorm=norm,
        )

    def integrate(**kwargs):
        destination = kwargs["destination"]
        destination.mkdir()
        dimension = 3 * (len(kwargs["xi_table"][3]) - 1)
        components = {
            "c2": np.eye(dimension) * 1.0e-6,
            "c3": np.eye(dimension) * 2.0e-6,
            "c4": np.eye(dimension) * 4.0e-6,
        }
        samples = {
            key: np.array([value * factor for factor in np.linspace(0.98, 1.02, 16)])
            for key, value in components.items()
        }
        archive = destination / "raw.npz"
        np.savez(archive, **components)
        integration_calls.append(kwargs["n_loops"])
        return {
            "components": components,
            "subsamples": samples,
            "archive": archive,
            "raw_integration_completed": True,
            "ordering": "radial_major",
            "multipoles": [0, 2, 4],
            "separation_edges_mpc_h": kwargs["xi_table"][3].tolist(),
            "n_loops": kwargs["n_loops"],
        }

    class Theory(BaseCalculator):
        _params = {
            name: {
                "value": 1.0,
                "prior": {"limits": [0.8, 1.2]},
                "ref": {"limits": [0.99, 1.01]},
                "proposal": 0.001,
            }
            for name in ("qiso", "qap")
        }

        def initialize(self, data, ap_mode, only_now):
            self.data = np.asarray(data)
            self.only_now = only_now
            self.init.params["qap"].update(fixed=ap_mode == "qiso")

        def calculate(self, qiso=1.0, qap=1.0):
            basis = np.linspace(-1, 1, len(self.data))
            self.flatmodel = self.data + (qiso - 1) + (qap - 1) * basis
            if self.only_now:
                self.flatmodel = self.flatmodel + 0.001 * np.cos(np.arange(len(self.data)))

    class Likelihood(BaseGaussianLikelihood):
        def initialize(self, theory, data, covariance):
            self.theory = theory
            super().initialize(data=data, covariance=covariance)

        @property
        def flattheory(self):
            return self.theory.flatmodel

    def build(data, matrix, **kwargs):
        # Actual native profiles/chains sample the just-read saved vector and covariance.
        constructors.append((kwargs["measurement_bin"], kwargs["z_eff"], np.asarray(matrix).copy()))
        assert kwargs["config"]["fitting"]["photoz_response"]["mu_quadrature_order"] == 128
        theory = Theory(data=data, ap_mode=kwargs["ap_mode"], only_now=kwargs["only_now"])
        likelihood = Likelihood(theory=theory, data=data, covariance=matrix)
        observable = SimpleNamespace(power_broadband_repair={"synthetic_theory": True})
        return theory, observable, likelihood, {"order": 128, "synthetic_theory": True}

    monkeypatch.setattr(reconstruction, "run_reconstruction", reconstruct)
    monkeypatch.setattr(clustering, "compute_xi", counts)
    monkeypatch.setattr(clustering, "compute_power_spectrum_secondary", power)
    monkeypatch.setattr(covariance, "integrate_cap", integrate)
    monkeypatch.setattr(likelihood, "build_likelihood", build)
    cli = CliRunner()
    root = tmp_path / "native pilot"
    workload = "boundary" if boundary_experiment else "pilot"
    extra = []
    if boundary_experiment:
        from pzbao.diagnostics.selection import fixed_depth, variant_config

        reference = driver.create_run(
            tmp_path, "unchanged-reference", workload="pilot", catalog_root=tmp_path
        )
        assert not driver.execute_stage(reference, "catalog", cpus=1)["errors"]
        assert not driver.execute_stage(reference, "select", cpus=1)["errors"]
        design = reference["config"]["selection"]["boundary_support"]["experiment_design"]
        target = design["targets"][0]
        derived = fixed_depth(
            selection.read_preparation(reference, target["tracer"]),
            target=target,
            config=reference["config"],
            design=design,
        )
        config = variant_config(
            reference,
            target,
            mode="truncate",
            depth=derived["depth_redshift"],
            previous_depth=derived["reference_depth_redshift"],
            design=design,
        )
        config_path = tmp_path / "boundary.yaml"
        driver._write_yaml(config_path, config)
        # The fixture config includes its small scalar grid, including on explicit reads.
        original_loader = driver.load_analysis_config
        monkeypatch.setattr(
            driver,
            "load_analysis_config",
            lambda path=None: load_analysis_config(path) if path else original_loader(),
        )
        extra = ["--config", str(config_path)]
    arguments = [
        "catalog",
        "--output-root",
        str(root),
        "--run-name",
        "eight-stage",
        "--catalog-root",
        str(tmp_path),
        "--workload",
        workload,
    ] + extra
    root = root / "eight-stage"
    source_root = root
    for stage in driver.STAGES[:-1]:
        if stage == "fit":
            source_identity = driver._source_identity(driver.load_run(source_root))
            source_snapshot = (source_root / "run.yaml").read_bytes()
            failed = source_root / "fit" / "attempts" / "retained" / "failure.txt"
            failed.parent.mkdir(parents=True)
            failed.write_text("original failed fit remains evidence\n")
            monkeypatch.setattr(
                driver,
                "_code_identity",
                lambda: {"code_commit": "d" * 40, "repository": str(tmp_path)},
            )
            root = root.parent / "corrected-inference"
            arguments.extend(["--inference-run", str(root)])
        result = cli.invoke(driver.main, arguments)
        assert result.exit_code == 0, (stage, result.output, result.exception)
        handoff = root / stage / driver.HANDOFFS[stage]
        saved = driver._read_yaml(handoff)
        assert len(saved["units"]) == saved["expected_unit_count"] and not saved["errors"]
        arguments = [driver.STAGES[driver.STAGES.index(stage) + 1], "--input", str(handoff)]
    run = driver.load_run(root)
    source_run = run["source_run"]
    assert run["metadata"]["source_run"] == source_identity
    assert run["metadata"]["code_commit"] == "d" * 40
    assert (source_root / "run.yaml").read_bytes() == source_snapshot
    assert failed.read_text() == "original failed fit remains evidence\n"
    assert not (source_root / "fit" / "units").exists()
    assert len(solves) == (50 if boundary_experiment else 100)
    assert len(integration_calls) == (4 if boundary_experiment else 8)
    assert len(constructors) == (
        2 * 52 + 2 if boundary_experiment else 2 * 104 + 4
    )  # Two profiles per fit and one mean likelihood per posterior.
    selected_route = driver.routes_for(run["config"], workload)[0]
    source = (
        source_root / "covariance/routes" / driver.route_name(selected_route) / "covariance.npz"
    )
    for route in driver.routes_for(run["config"], workload):
        saved = covariance.read_route_covariance(source_run, route)
        np.testing.assert_array_equal(saved["arrays"]["multipoles"], [0, 2])
        assert saved["arrays"]["vectors"].shape == (25, 52)
        np.testing.assert_array_equal(
            saved["arrays"]["mean_covariance"], saved["arrays"]["covariance"] / 25
        )
        preparation = selection.read_preparation(source_run, route["tracer"])
        if boundary_experiment:
            assert "source_run" not in source_run
            for cap in ("NGC", "SGC"):
                group = preparation["conditions"][route["condition"]][cap]
                assert all(
                    record["boundary_experiment"]["removed_clustering_count"] == 1
                    for record in group["boundary_counts_by_mock"]
                )
        fitted = root / "fit" / "units" / (driver.route_name(route) + "--mean")
        metadata = json.loads((fitted / "fit.json").read_text())
        assert metadata["identity"]["code_commit"] == "d" * 40
        assert metadata["source_run"] == source_identity
        assert driver._read_yaml(fitted / "result.yaml")["source_run"] == source_identity
        redshifts = preparation["effective_redshifts"][route["condition"]]
        assert metadata["z_eff"] == redshifts[route["measurement_bin"]]["z_eff"]
        # Per-bin reconstruction (revision 2.13) solves each bin at its own frozen effective
        # redshift, so the recorded reconstruction inputs are the bin's, not the parent's.
        solved = redshifts[route["measurement_bin"]]
        assert metadata["reconstruction_inputs"]["z_eff"] == solved["z_eff"]
        assert metadata["reconstruction_inputs"]["growth_rate"] == solved["growth_rate"]
        assert metadata["reconstruction_inputs"]["z_eff"] != redshifts["parent"]["z_eff"]
        assert metadata["reconstruction_inputs"]["bias"] == (
            run["config"]["tracers"][route["tracer"]]["reconstruction"]["fiducial_bias"]
        )
        assert metadata["fitting_growth_rate"] == pytest.approx(0.5 + 0.3 * metadata["z_eff"])
        if route["reconstruction_state"] == "postreconstruction":
            assert (
                metadata["analytic_transverse_damping"]["attained_density_h3_mpc3"]
                == metadata["attained_density_h3_mpc3"]
            )
        native = fitting.read_profiles(fitted / "profiles.npy")
        qiso = next(row for row in native["parameters"] if row["basename"] == "qiso")
        assert qiso["value"] == pytest.approx(1.0, abs=1.0e-5)
    covariance_metadata = source.parent / "covariance.json"
    original_covariance_metadata = covariance_metadata.read_bytes()
    wrong = json.loads(original_covariance_metadata)
    wrong["code_commit"] = run["metadata"]["code_commit"]
    covariance_metadata.write_text(json.dumps(wrong))
    mean_unit = {**selected_route, "realization_id": None}
    with pytest.raises(ValueError, match="another route or snapshot"):
        likelihood._route_likelihood(run, mean_unit, method="profiling")
    covariance_metadata.write_bytes(original_covariance_metadata)
    notes = root / "review" / "review.md"
    notes.write_text("Synthetic review notes to preserve.\n")
    result = cli.invoke(driver.main, ["review", "--run", str(root)])
    assert result.exit_code == 0, (result.output, result.exception)
    assert notes.read_text() == "Synthetic review notes to preserve.\n"
    with (root / "review" / "coverage.csv").open() as stream:
        coverage = list(csv.DictReader(stream))
    assert all(
        row["coverage"] in {"complete", "stopping_rule_reached", "budget_exhausted"}
        for row in coverage
    )
    with (root / "review" / "posterior.csv").open() as stream:
        posterior = list(csv.DictReader(stream))
    assert len(posterior) == (4 if boundary_experiment else 8)
    assert all(len(json.loads(row["ensemble_contributions"])) == 4 for row in posterior)
    assert list((root / "review").glob("*power*.png"))
    assert list((root / "review").glob("*.pdf"))
