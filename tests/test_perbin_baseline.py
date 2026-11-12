"""The per-bin baseline loader: run roots in, the baseline notebook's mapping out."""

import copy
import json
import zlib

import numpy as np
import pytest
import yaml

from pzbao import fitting, run as driver, sampling
from pzbao.config import UNIFIED_PCS2, UNIFIED_PRIOR, load_analysis_config
from pzbao.figures import data

CODE = {"code_commit": "a" * 40, "repository": "fixture", "dirty": False, "code_state_sha256": "x"}
SAMPLED = ["qiso", "qap", "b1"]


def _profile(value=1.0, width=0.02, valid=True):
    parameter = {
        "value": value,
        "fixed": False,
        "derived": False,
        "parabolic_error": width,
        "minos_offsets": [-width, width],
        "minos_bounds": [value - width, value + width],
        "interval_flags": {"is_valid": valid, "lower_valid": valid, "upper_valid": valid},
    }
    return {
        "parameters": [
            {**parameter, "basename": "qiso", "name": "qiso"},
            {**parameter, "basename": "qap", "name": "qap"},
        ],
        "maximization": {
            "is_valid": True,
            "is_above_max_edm": False,
            "has_reached_call_limit": False,
        },
        "chi2": 40.0,
    }


def _chain(walkers=3, steps=4, seed=0):
    rng = np.random.default_rng(seed)
    draws = 1.0 + 0.01 * rng.standard_normal((walkers, steps, len(SAMPLED)))
    return {
        "draws": draws,
        "parameters": list(SAMPLED),
        "axes": ["walker", "iteration", "parameter"],
    }


def _write_preparation(run, identity):
    """A radial preparation whose cells tile every measurement bin of each tracer."""
    stage = data._preparation_stage(data.Sources(), run["root"])
    for tracer, spec in run["config"]["tracers"].items():
        directory = run["root"] / stage / "units" / tracer
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "result.yaml").write_text(yaml.safe_dump(dict(identity)))
        bins = spec["measurement_bins"]
        edges = sorted({edge for shell in bins for edge in (shell["zmin"], shell["zmax"])})
        edges = [edges[0] - 0.05, *edges, edges[-1] + 0.05]
        counts = [100.0] * (len(edges) - 1)
        caps = {
            cap: {"radial": {"z_edges": edges, "mean_counts": counts}}
            for cap in spec["footprint"]["area_deg2"]
        }
        preparation = {
            "tracer": tracer,
            "realization_ids": list(range(25)),
            "conditions": {tier: copy.deepcopy(caps) for tier in data.TIERS},
        }
        (directory / "preparation.json").write_text(json.dumps(preparation))


def _publish_fit_units(refit, measurement_identity, *, skip=()):
    base = {k: refit["metadata"][k] for k in data.IDENTITY}
    for route in driver.routes_for(refit["config"], refit["metadata"]["workload"]):
        route = {k: route[k] for k in data._ROUTE_KEYS}
        for mock in (None, *range(25)):
            if (route["measurement_bin"], route["condition"], mock) in skip:
                continue
            unit = data._perbin_unit(refit, "fit", route, mock)
            unit.mkdir(parents=True)
            (unit / "profiles.npy").write_bytes(b"fixture")
            (unit / "fit.json").write_text(
                json.dumps(
                    {
                        "identity": {**route, "realization_id": mock, **base},
                        "covariance_identity": measurement_identity,
                        "chi2": 40.0,
                        "degrees_of_freedom": 38,
                        "detection": {"chi2_smooth": 60.0},
                    }
                )
            )


def _publish_samples(refit, *, skip=()):
    base = {k: refit["metadata"][k] for k in data.IDENTITY}
    count = refit["config"]["fitting"]["sampling"]["ensembles"]
    for route in driver.routes_for(refit["config"], refit["metadata"]["workload"]):
        route = {k: route[k] for k in data._ROUTE_KEYS}
        if (route["measurement_bin"], route["condition"]) in skip:
            continue
        unit = data._perbin_unit(refit, "sample", route)
        unit.mkdir(parents=True)
        ensembles = []
        for index in range(count):
            (unit / f"ensemble-{index}.npy").write_bytes(b"chain")
            ensembles.append(
                {
                    "ensemble_index": index,
                    "path": f"ensemble-{index}.npy",
                    "steps": 4,
                    "retained_draws": 6,
                    "diagnostics": {
                        "walkers": 3,
                        "retained_steps": 2,
                        "parameters": {
                            "qiso": {"unavailable_reason": None},
                            "qap": {"unavailable_reason": "short chain" if index == 0 else None},
                        },
                    },
                }
            )
        (unit / "sampling.json").write_text(
            json.dumps(
                {
                    "identity": {**route, **base},
                    "sampled_parameters": SAMPLED,
                    "complete_ensemble_collection": True,
                    "burnin_fraction": 0.5,
                    "ensembles": ensembles,
                }
            )
        )


@pytest.fixture
def perbin(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "_code_identity", lambda **kwargs: dict(CODE))
    current = load_analysis_config()
    monkeypatch.setattr(driver, "load_analysis_config", lambda: copy.deepcopy(current))
    # Per-realization best fits must scatter, as the joint-tier correlation requires.
    monkeypatch.setattr(
        fitting,
        "read_profiles",
        lambda path, *, identity: _profile(
            value=1.0 + (zlib.crc32(str(path).encode()) % 997) / 2e4
        ),
    )
    monkeypatch.setattr(sampling, "read_chain", lambda path, **kwargs: _chain(seed=len(str(path))))
    measurement = driver.create_run(
        tmp_path, "primary", workload="full_grid", catalog_root=tmp_path
    )
    refit = driver.create_run(
        tmp_path,
        "refit",
        workload="full_grid",
        catalog_root=tmp_path,
        source_run=measurement,
        inference_definition=UNIFIED_PRIOR,
    )
    identity = {k: measurement["metadata"][k] for k in data.IDENTITY}
    _write_preparation(measurement, identity)
    return measurement, refit, identity


def test_perbin_baseline_builds_the_notebook_mapping(perbin):
    pytest.importorskip("cosmoprimo")  # the density rows need the configured mock cosmology
    measurement, refit, identity = perbin
    _publish_fit_units(refit, identity)
    _publish_samples(refit)
    baseline = data.load_perbin_baseline(refit["root"], measurement["root"])
    assert set(baseline) == {
        "draws",
        "fits",
        "mean_fits",
        "profiles",
        "joint",
        "correlation",
        "fom",
        "posterior_summary",
        "posterior_records",
        "profile_diagnostics",
        "measurement_identity",
        "density",
        "density_axes",
        "sources",
    }
    assert baseline["measurement_identity"] == identity
    assert len(baseline["posterior_records"]) == 50 and len(baseline["fom"]) == 25
    assert len(baseline["posterior_summary"]) == 100 and len(baseline["density"]) == 25
    assert {len(members) for members in baseline["profiles"].values()} == {25}
    assert len(baseline["profile_diagnostics"]) == 50 * 25 * 2
    key = next(iter(baseline["draws"]))
    draws = baseline["draws"][key]
    assert draws.shape == (4 * 6, 2)
    record = next(
        r
        for r in baseline["posterior_records"]
        if data.historical_key(r["route"], __import__("pzbao.figures.style").figures.style) == key
    )
    np.testing.assert_allclose(
        record["fom"], 1 / np.sqrt(np.linalg.det(np.cov(draws, rowvar=False, ddof=1)))
    )
    assert record["sampling_limits"] == ["ensemble 0 qap: short chain"]
    assert baseline["fits"][key]["qiso_std"] == pytest.approx(draws[:, 0].std(ddof=1))
    assert baseline["mean_fits"][key]["detection"]["chi2_smooth"] == 60.0
    assert all(pair["sampling_limited"] for pair in baseline["fom"])
    assert {row["status"] for row in baseline["joint"]} == {"available"}
    member = baseline["profiles"][key][0]
    assert member["valid"] and member["qiso_minos"] == pytest.approx(0.02)
    assert member["dof"] == 38
    counts = data.baseline_interval_counts(baseline["profile_diagnostics"])
    assert len(counts) == 50 and all(row["usable_pairs"] == 25 for row in counts)


def test_perbin_baseline_fails_closed_on_a_missing_unit(perbin):
    measurement, refit, identity = perbin
    _publish_fit_units(refit, identity, skip={("LRG3", "sigma_0.03", 7)})
    _publish_samples(refit)
    with pytest.raises(ValueError, match="Incomplete individual-realization cohort"):
        data.load_perbin_baseline(refit["root"], measurement["root"])


def test_perbin_baseline_fails_closed_on_a_missing_posterior(perbin):
    measurement, refit, identity = perbin
    _publish_fit_units(refit, identity)
    _publish_samples(refit, skip={("BGS2", "native")})
    with pytest.raises(ValueError, match="Missing per-bin posterior"):
        data.load_perbin_baseline(refit["root"], measurement["root"])


def test_perbin_baseline_requires_the_unified_refit_of_this_measurement(perbin, tmp_path):
    measurement, refit, identity = perbin
    check = driver.create_run(
        tmp_path,
        "pcs2",
        workload="full_grid",
        catalog_root=tmp_path,
        source_run=measurement,
        inference_definition=UNIFIED_PCS2,
    )
    with pytest.raises(ValueError, match="unified-prior refit"):
        data.load_perbin_baseline(check["root"], measurement["root"])
    other = driver.create_run(tmp_path, "other", workload="full_grid", catalog_root=tmp_path)
    with pytest.raises(ValueError, match="does not record this measurement"):
        data.load_perbin_baseline(refit["root"], other["root"])


def test_preparation_stage_follows_the_run_interface(perbin):
    measurement, _, _ = perbin
    assert data._preparation_stage(data.Sources(), measurement["root"]) == "select"
