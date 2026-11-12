"""Absolute FoM contracts across native chain serialization and plotted values."""

from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from pzbao.figures import data as plot
from pzbao.diagnostics import sampling as stage_sampling


def save(path, data):
    path.write_text(json.dumps(data))


@pytest.fixture
def native(tmp_path):
    pytest.importorskip("desilike")
    from desilike.samples import Chain
    from pzbao.sampling import read_chain

    sampling = dict(ensembles=4, walkers=4, burnin_fraction=0.5)
    config = dict(
        fitting=dict(
            broadband_model="power3",
            sampling=sampling,
            nonlinear_parameters={
                p: dict(prior=dict(limits=[0.8, 1.2])) for p in stage_sampling.PARAMETERS
            },
        ),
        reporting=dict(alpha_rail_tolerance=0.005),
    )
    route = dict(
        tracer="LRG",
        measurement_bin="LRG2",
        condition="sigma_0.001",
        reconstruction_state="postreconstruction",
    )
    identity = dict(
        run_name="fixture", code_commit="science", analysis_config_sha256="definition", **route
    )
    model = dict(
        identity=identity,
        realization_id=None,
        route=route,
        covariance_identity=dict(run_name="covariance"),
        broadband_representation=dict(broadband_model="power3"),
    )
    flags = dict(
        is_valid=True,
        lower_valid=True,
        upper_valid=True,
        at_lower_limit=False,
        at_upper_limit=False,
    )
    expected = dict(
        index=12,
        route=route,
        identity=identity,
        current_definition=dict(analysis=config),
        model={k: v for k, v in model.items() if k != "identity"},
        profile=dict(
            parameters=[
                dict(basename=p, fixed=False, interval_flags=flags)
                for p in stage_sampling.PARAMETERS
            ],
            unavailable_reason=None,
        ),
        covariance_validity=dict(structural_test_passes=True),
    )
    rows, proofs, streams, kept = [], [], [], []
    rng = np.random.default_rng(12)
    for index in range(4):
        iso = 1 + rng.normal(0, 0.02, (40, 4))
        ap = 1 + 0.7 * (iso - 1) + rng.normal(0, 0.03, (40, 4))
        chain = Chain(data=[iso, ap], params=list(stage_sampling.PARAMETERS))
        stream = dict(root_seed=4201, spawn_key=[2, 1, 1, index], ensemble_index=index)
        streams.append(stream)
        chain.attrs.update(
            identity=dict(identity, ensemble_index=index),
            random_stream=stream,
            acceptance_fraction=[0.4] * 4,
        )
        name = f"ensemble-{index}.npy"
        chain.save(str(tmp_path / name))
        diagnostics = {
            p: dict(
                tau=None,
                nominal_autocorrelation_based_ess=None,
                unavailable_reason="fixture chain too short",
            )
            for p in stage_sampling.PARAMETERS
        }
        rows.append(
            dict(
                ensemble_index=index,
                path=name,
                steps=40,
                retained_draws=80,
                acceptance_fraction=[0.4] * 4,
                random_stream=stream,
                diagnostics=dict(parameters=diagnostics),
            )
        )
        proofs.append(dict(ensemble_index=index, path=name, shape=[4, 40, 2]))
        kept.append(np.stack([iso[20:].T, ap[20:].T], axis=-1).reshape(-1, 2))
    summary = dict(
        identity=identity,
        ensembles=rows,
        complete_ensemble_collection=True,
        sampled_parameters=list(stage_sampling.PARAMETERS),
        gr_history=[dict(step=40, value=0.02)],
        burnin_fraction=0.5,
        inference_thinning=1,
        stopping_rule_reached=False,
        budget_exhausted=True,
        stop_reason="budget_exhausted",
    )
    save(tmp_path / "sampling.json", summary)
    save(
        tmp_path / "gr_history.json",
        dict(identity=identity, sampling=sampling, history=summary["gr_history"]),
    )
    save(
        tmp_path / "attempt.json",
        dict(current_definition=expected["current_definition"], sampling=sampling),
    )
    save(tmp_path / "model.json", model)
    save(
        tmp_path / "complete.json",
        dict(
            index=12,
            identity=identity,
            chain_readback=proofs,
            **{k: summary[k] for k in ("stopping_rule_reached", "budget_exhausted", "stop_reason")},
        ),
    )
    fitting = SimpleNamespace(
        read_chain=read_chain, ensemble_random_streams=lambda config, route: streams
    )
    return tmp_path, expected, fitting, np.concatenate(kept)


def test_native_values_and_correlation_survive_serialization(native):
    root, expected, fitting, draws = native
    row = stage_sampling.posterior(root, expected, fitting)
    covariance = np.cov(draws, rowvar=False, ddof=1)
    np.testing.assert_allclose(row["covariance"], covariance, rtol=1e-13)
    assert row["fom"] == pytest.approx(1 / np.sqrt(np.linalg.det(covariance)))
    assert row["fom"] != pytest.approx(1 / np.prod(np.std(draws, axis=0, ddof=1)))
    assert row["retained_draws"] == 320
    assert len(row["sampling_limits"]) == 9
    assert not row["scientific_acceptance"]
    assert not row["profile_flagged"]
    assert len(row["input_sha256"]) == 9


@pytest.mark.parametrize(
    "change",
    [
        "identity",
        "model",
        "individual",
        "burnin",
        "duplicate",
        "missing",
        "axes",
        "retained",
        "stream",
        "definition",
    ],
)
def test_incompatible_native_products_rejected(native, change):
    root, expected, fitting, _ = native
    name = "sampling.json"
    data = stage_sampling.read_json(root / name)
    if change == "identity":
        data["identity"]["condition"] = "native"
    elif change == "model":
        name, data = "model.json", stage_sampling.read_json(root / "model.json")
        data["broadband_representation"]["broadband_model"] = "pcs2"
    elif change == "individual":
        name, data = "model.json", stage_sampling.read_json(root / "model.json")
        data["realization_id"] = 0
    elif change == "burnin":
        data["burnin_fraction"] = 0.25
    elif change == "duplicate":
        data["ensembles"][1] = deepcopy(data["ensembles"][0])
    elif change == "missing":
        data["ensembles"].pop()
    elif change == "axes":
        data["ensembles"][0]["steps"] = 41
    elif change == "retained":
        data["ensembles"][0]["retained_draws"] = 79
    elif change == "stream":
        data["ensembles"][0]["random_stream"]["root_seed"] += 1
    elif change == "definition":
        name, data = "attempt.json", stage_sampling.read_json(root / "attempt.json")
        data["current_definition"]["analysis"]["fitting"]["broadband_model"] = "pcs2"
    save(root / name, data)
    with pytest.raises(ValueError):
        stage_sampling.posterior(root, expected, fitting)


def test_profile_flags_are_retained_without_filtering_posterior(native):
    root, expected, fitting, _ = native
    expected["profile"]["parameters"][0]["interval_flags"]["at_lower_limit"] = True
    row = stage_sampling.posterior(root, expected, fitting)
    assert row["profile_flagged"]
    assert row["profile"]["parameters"][0]["interval_flags"]["at_lower_limit"]
    assert row["fom"] > 0


@pytest.mark.parametrize(
    "values", [np.ones((20, 2)), np.zeros((1, 2)), np.full((20, 2), np.nan), np.ones((20, 3))]
)
def test_undefined_fom_rejected(values):
    with pytest.raises(ValueError):
        stage_sampling.joint_summary(values)


def row(name, tier, state, value):
    return dict(
        route=dict(measurement_bin=name, condition=tier, reconstruction_state=state),
        status="available",
        fom=value,
        profile_flagged=False,
        sampling_limits=[],
    )


def test_absolute_bar_heights_gains_and_missing_slots(tmp_path):
    records = [
        row("LRG2", "native", plot.STATES[0], 200),
        row("LRG2", "native", plot.STATES[1], 300),
        row("LRG2", "sigma_0.001", plot.STATES[1], 80),
    ]
    records[1]["profile_flagged"] = True
    records[1]["sampling_limits"] = ["ESS unavailable"]
    from tests.notebook_helpers import notebook_namespace

    namespace = notebook_namespace("comparisons", cells={"setup", "comparison-inputs", "joint-fom"})
    fig, pairs = namespace["joint_fom_figure"](records)
    axis = fig.axes[3]
    assert sorted(patch.get_height() for patch in axis.patches) == [80, 200, 300]
    assert axis.get_yscale() == "log"
    assert len(fig.axes[0].patches) == 0  # Missing BGS is not a zero FoM.
    assert any(text.get_text() == "NA" for text in fig.axes[0].texts)
    pair = next(p for p in pairs if p["measurement_bin"] == "LRG2" and p["condition"] == "native")
    assert pair["gain"] == 1.5
    missing = next(
        p for p in pairs if p["measurement_bin"] == "LRG2" and p["condition"] == "sigma_0.001"
    )
    assert missing["pre"] is None and missing["gain"] is None
    gain_axis = fig.axes[8]
    assert gain_axis.lines[0].get_ydata()[0] == 1.5
    assert np.isnan(gain_axis.lines[0].get_ydata()[1])
    assert any(text.get_text() == "†" for text in gain_axis.texts)
    assert "partial grid" in fig._suptitle.get_text()
    for extension in ("pdf", "png"):
        path = tmp_path / ("fixture." + extension)
        fig.savefig(path, bbox_inches="tight")
        assert path.stat().st_size > 1000
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_duplicate_routes_and_nonpositive_values_rejected():
    value = row("LRG2", "native", plot.STATES[0], 10)
    with pytest.raises(ValueError, match="duplicate"):
        plot.paired_values([value, value])
    value["fom"] = 0
    with pytest.raises(ValueError, match="invalid absolute"):
        plot.paired_values([value])


def test_input_mutation_rejected(tmp_path):
    path = tmp_path / "input.json"
    path.write_text("{}")
    sources = {str(path): stage_sampling.digest(path)}
    path.write_text('{"changed": true}')
    with pytest.raises(ValueError, match="source bytes"):
        stage_sampling.verify_files(sources)
