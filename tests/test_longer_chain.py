"""Verify deterministic continuation and correlation-aware diagnostic summaries."""

import pzbao.diagnostics.sampling as diag_sampling
import copy
import numpy as np
import pytest
import pzbao.sampling as fitting_sampling
from pzbao.config import load_analysis_config


def test_batch_errors_keep_correlated_walkers_together():
    rng = np.random.RandomState(193)
    single = rng.normal(size=(4000, 1))
    replicated = np.repeat(single, 40, axis=1)
    one, forty = (diag_sampling.batch_errors(single, 3), diag_sampling.batch_errors(replicated, 3))
    for a, b in zip(one["estimates"], forty["estimates"]):
        np.testing.assert_allclose(a["mcse"]["mean"], b["mcse"]["mean"], rtol=1e-12)
    assert len(forty["estimates"]) == 2
    assert diag_sampling.batch_errors(single, None)["estimates"] == []
    assert (
        "fewer than eight"
        in diag_sampling.batch_errors(single[:10], 5)["estimates"][0]["unavailable_reason"]
    )


def test_native_continuation_preserves_draws_rng_and_originals(tmp_path, monkeypatch):
    pytest.importorskip("desilike")
    pytest.importorskip("emcee")
    from desilike.likelihoods import BaseLikelihood

    class Gaussian(BaseLikelihood):
        _params = {
            "qiso": dict(
                value=1.0,
                prior=dict(limits=[0.8, 1.2]),
                ref=dict(limits=[0.99, 1.01]),
                proposal=0.01,
            )
        }

        def calculate(self, qiso=1.0):
            self.loglikelihood = -0.5 * ((qiso - 1.0) / 0.025) ** 2

    config = load_analysis_config()
    config["fitting"]["sampling"].update(
        minimum_iterations=10, check_every=20, absolute_iteration_ceiling=65
    )
    sampling = config["fitting"]["sampling"]
    route = {
        "tracer": "BGS_BRIGHT-21.5",
        "measurement_bin": "BGS1",
        "condition": "sigma_0.003",
        "reconstruction_state": "prereconstruction",
    }
    identity = {"run_name": "toy", "code_commit": "a" * 40, "analysis_config_sha256": "b" * 64}
    streams = fitting_sampling.ensemble_random_streams(config, route)

    def paths(name):
        directory = tmp_path / name
        directory.mkdir()
        return {
            "chains": [directory / f"ensemble-{i}.npy" for i in range(4)],
            "summary": directory / "sampling.json",
        }

    arguments = dict(config=config, ap_mode="qiso", route=route, run_identity=identity)
    monkeypatch.setattr(
        fitting_sampling, "ensemble_gelman_rubin", lambda *a, **kw: {"usable": False, "value": None}
    )
    reference_paths = paths("reference")
    fitting_sampling.run_sampling(Gaussian(), reference_paths, **arguments)
    monkeypatch.setattr(
        fitting_sampling, "ensemble_gelman_rubin", lambda *a, **kw: {"usable": True, "value": 0.0}
    )
    parent_paths = paths("parent")
    stopped = fitting_sampling.run_sampling(Gaussian(), parent_paths, **arguments)
    assert stopped["stopping_rule_reached"]
    original_bytes = [path.read_bytes() for path in parent_paths["chains"]]
    decoded = [fitting_sampling.read_chain(path) for path in parent_paths["chains"]]
    full_identity = {**identity, **route}
    assert (
        diag_sampling.validate_checkpoints(
            decoded, streams, identity=full_identity, sampling=sampling, sampled_parameters=["qiso"]
        )
        == 40
    )
    bad = copy.deepcopy(decoded)
    bad[0]["attrs"]["identity"]["code_commit"] = "wrong"
    with pytest.raises(ValueError, match="identity changed"):
        diag_sampling.validate_checkpoints(
            bad, streams, identity=full_identity, sampling=sampling, sampled_parameters=["qiso"]
        )
    bad = copy.deepcopy(decoded)
    del bad[1]["attrs"]["rng_state"]
    with pytest.raises(ValueError, match="lacks RNG"):
        diag_sampling.validate_checkpoints(
            bad, streams, identity=full_identity, sampling=sampling, sampled_parameters=["qiso"]
        )
    destination = tmp_path / "extended"
    destination.mkdir()
    diagnostic_identity = {**full_identity, "run_name": "diagnostic"}
    summaries = list(
        diag_sampling.continue_ensembles(
            Gaussian(),
            decoded,
            streams,
            identity=diagnostic_identity,
            sampling=sampling,
            sampled_parameters=["qiso"],
            endpoints=[60, 65],
            destination=destination,
        )
    )
    assert [step for step, _ in summaries] == [60, 65]
    for index, parent in enumerate(parent_paths["chains"]):
        assert parent.read_bytes() == original_bytes[index]
        expected = fitting_sampling.read_chain(reference_paths["chains"][index])
        actual = fitting_sampling.read_chain(
            destination / f"ensemble-{index}.npy",
            identity={**diagnostic_identity, "ensemble_index": index},
        )
        np.testing.assert_array_equal(actual["draws"], expected["draws"])
        assert actual["chain"].shape == (65, 40)
        for left, right in zip(actual["attrs"]["rng_state"], expected["attrs"]["rng_state"]):
            np.testing.assert_array_equal(left, right)


def test_constant_derived_quantity_is_not_a_failed_sampling_diagnostic(monkeypatch):
    config = load_analysis_config()["fitting"]["sampling"]
    rng = np.random.RandomState(2)
    rows = [
        {
            "draws": np.stack([rng.normal(size=(40, 1000)), np.ones((40, 1000))], axis=-1),
            "parameters": ["qiso", "DH_over_DM"],
            "chain": None,
        }
        for _ in range(4)
    ]
    monkeypatch.setattr(
        fitting_sampling, "ensemble_gelman_rubin", lambda *a, **k: {"usable": True, "value": 0.001}
    )
    summary = diag_sampling.summarize(rows, sampled_parameters=["qiso"], sampling=config)
    for row in summary["ensembles"]:
        fixed = row["parameters"]["DH_over_DM"]
        assert fixed["mean"] == 1.0
        assert "not applicable" in fixed["autocorrelation"]["unavailable_reason"]
        assert fixed["sampling_error"]["estimates"] == []
