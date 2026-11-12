"""Input workers and condition reuse preserve the serial catalogue calculation."""

import copy
import hashlib
import json

from joblib import Parallel, parallel_config
import numpy as np
import pytest

from pzbao import catalog as loader, selection as modifier, run as driver
from pzbao.config import load_analysis_config


@pytest.fixture
def input_pool(monkeypatch):
    # Force several real process chunks, including an incomplete final chunk.
    monkeypatch.setattr(loader, "_ROW_CHUNK_BYTES", 128)
    with parallel_config(backend="loky", inner_max_num_threads=1):
        with Parallel(n_jobs=2, return_as="generator", max_nbytes=None) as pool:
            yield pool


def catalogue(size, low=0.1, high=0.4):
    return {
        "TARGETID": np.arange(size, dtype=np.int64),
        "RA": np.linspace(1, 359, size),
        "DEC": np.linspace(-80, 80, size),
        "Z": np.linspace(low, high, size, endpoint=False),
        "WEIGHT": np.linspace(0.5, 1.5, size),
        "PHOTSYS": np.full(size, "N"),
    }


@pytest.mark.parametrize("tracer", ["LRG", "BGS_BRIGHT-21.5"])
@pytest.mark.parametrize("role", ["data", "reservoir", "random"])
def test_parallel_identity_matches_frozen_token_bytes(input_pool, tracer, role):
    config = load_analysis_config()
    source = {key: value[::-1] for key, value in catalogue(39).items()}
    kwargs = dict(tracer=tracer, mock_id=8, cap="SGC", role=role, config=config)
    if role == "random":
        kwargs["random_split"] = 17
    serial = loader.assign_source_identity(source, **kwargs)
    parallel = loader.assign_source_identity(source, parallel=input_pool, **kwargs)
    for key in serial:
        np.testing.assert_array_equal(parallel[key], serial[key])
    if role != "random":
        expected = [
            int.from_bytes(
                hashlib.blake2b(f"{tracer}|8|SGC|{role}|{target}".encode(), digest_size=8).digest(),
                "little",
            )
            for target in source["TARGETID"]
        ]
        np.testing.assert_array_equal(parallel["SOURCE_ID"], np.asarray(expected, dtype=np.uint64))
    duplicate = {key: value.copy() for key, value in source.items()}
    for value in duplicate.values():
        value[-1] = value[0]
    with pytest.raises(ValueError, match="duplicate"):
        loader.assign_source_identity(duplicate, parallel=input_pool, **kwargs)
    if role == "random":
        duplicate["RA"][-1] = np.nan
        with pytest.raises(ValueError, match="finite"):
            loader.assign_source_identity(duplicate, parallel=input_pool, **kwargs)


def test_parallel_streams_preserve_uint64_extremes_and_namespaces(input_pool):
    config = load_analysis_config()
    source = catalogue(39)
    source["SOURCE_ID"] = np.arange(2**64 - 39, 2**64, dtype=np.uint64)[::-1]
    for role in ("data", "random"):
        serial = modifier.attach_stable_randomness(source, role=role, config=config)
        parallel = modifier.attach_stable_randomness(
            source, role=role, config=config, parallel=input_pool
        )
        for key in serial:
            np.testing.assert_array_equal(serial[key], parallel[key])
    ids = source["SOURCE_ID"]
    expected = np.array(
        [
            (
                int.from_bytes(
                    hashlib.blake2b(f"{int(i)}|42|photoz-u1".encode(), digest_size=8).digest(),
                    "little",
                )
                + 0.5
            )
            / float(2**64)
            for i in ids
        ]
    )
    np.testing.assert_array_equal(
        modifier._stable_uniform(ids, seed=42, namespace="photoz-u1", parallel=input_pool), expected
    )


@pytest.mark.parametrize("tracer", ["LRG", "BGS_BRIGHT-21.5"])
def test_all_conditions_reuse_streams_and_preserve_selected_values(input_pool, monkeypatch, tracer):
    config = load_analysis_config()
    field = loader.reconstruction_field(tracer, config)
    data = loader.assign_source_identity(
        catalogue(100, field["zmin"], field["zmax"]),
        tracer=tracer,
        mock_id=8,
        cap="NGC",
        role="data",
        config=config,
    )
    randoms = loader.assign_source_identity(
        catalogue(1000, field["zmin"], field["zmax"]),
        tracer=tracer,
        mock_id=8,
        cap="NGC",
        role="random",
        random_split=0,
        config=config,
    )
    before = copy.deepcopy((data, randoms))
    bins = loader.measurement_bins(tracer, config)
    monkeypatch.setattr(modifier, "comoving_shell_volume", lambda *a, **k: [1e9] * len(bins))
    cache = modifier._InputPreparationCache(config, parallel=input_pool)
    for tier in config["selection"]["tiers"]:
        identity = f"{tracer}|{tier['label']}|cap=NGC"
        radial = {
            "z_edges": np.linspace(field["zmin"], field["zmax"], 6).tolist(),
            "target_probability_mass": [0.2] * 5,
            "sample_identity": identity,
        }
        kwargs = dict(
            tracer=tracer,
            tier_label=tier["label"],
            cap="NGC",
            cosmology=object(),
            radial_artifact=radial,
            sample_identity=identity,
            frozen_counts_by_bin={b["name"]: 2 for b in bins},
            config=config,
        )
        expected = modifier.prepare_tracer_field(data, randoms, **kwargs)
        actual = modifier.prepare_tracer_field(data, randoms, input_cache=cache, **kwargs)
        # As the matched-density selector prepares a matched tier: its target cleared on a
        # copy, through the pass's cache built from the run configuration.
        cleared = copy.deepcopy(config)
        for entry in cleared["selection"]["tiers"]:
            if entry["label"] == tier["label"]:
                entry["target_nbar_h3_mpc3"] = None
        matched = {**kwargs, "config": cleared}
        uncached = modifier.prepare_tracer_field(data, randoms, **matched)
        reused = modifier.prepare_tracer_field(data, randoms, input_cache=cache, **matched)
        for one, other in ((expected, actual), (uncached, reused)):
            assert one[2] == other[2]
            for left, right in zip(one[:2], other[:2]):
                assert left.keys() == right.keys()
                for key in left:
                    np.testing.assert_array_equal(left[key], right[key])
    assert len(cache.catalogues) == 2
    assert len(cache.orders) == 1
    for raw, snapshot in zip((data, randoms), before):
        for key in raw:
            np.testing.assert_array_equal(raw[key], snapshot[key])
    # A tier's target density sets counts only: the matched-density run clears it on a copy and
    # reuses the very same cached arrays.
    matched = copy.deepcopy(config)
    for tier in matched["selection"]["tiers"]:
        tier["target_nbar_h3_mpc3"] = None
    for role, catalog in (("data", data), ("random", randoms)):
        reused = cache.normalized(catalog, role=role, config=matched)
        assert reused is cache.normalized(catalog, role=role, config=config)
    assert len(cache.catalogues) == 2
    changed = copy.deepcopy(config)
    changed["selection"]["stable_randomness"]["seed"] += 1
    with pytest.raises(ValueError, match="different selection"):
        cache.normalized(data, role="data", config=changed)
    relabelled = copy.deepcopy(matched)
    relabelled["selection"]["tiers"][0]["label"] = "other"
    with pytest.raises(ValueError, match="different selection"):
        cache.normalized(data, role="data", config=relabelled)


@pytest.mark.parametrize("role", ["data", "random"])
def test_cached_normalization_owns_arrays_and_preserves_public_copy_contract(role):
    config = load_analysis_config()
    source = catalogue(20)
    source["SOURCE_ID"] = np.arange(20, dtype=np.uint64)
    source["WEIGHT_COMP"] = np.full(20, 2.0)
    before = copy.deepcopy(source)
    expected = modifier.attach_stable_randomness(
        modifier.normalize_weights(source), role=role, config=config
    )
    cache = modifier._InputPreparationCache(config)
    actual = cache.normalized(source, role=role, config=config)
    assert actual is cache.normalized(source, role=role, config=config)
    assert actual.keys() == expected.keys()
    assert "WEIGHT_COMP" not in actual
    for key in actual:
        assert actual[key].dtype == expected[key].dtype
        np.testing.assert_array_equal(actual[key], expected[key])
        assert not actual[key].flags.writeable
        if key in source:
            assert not np.shares_memory(actual[key], source[key])
    public = modifier.attach_stable_randomness(source, role=role, config=config)
    for key in source:
        np.testing.assert_array_equal(source[key], before[key])
        assert source[key].flags.writeable
        assert public[key].flags.writeable
        assert not np.shares_memory(public[key], source[key])
        public[key][0] = 0
        np.testing.assert_array_equal(source[key], before[key])


def test_cached_bin_order_preserves_rank_ties_edges_and_support_errors(monkeypatch):
    config = load_analysis_config()
    source = catalogue(603)
    source["SOURCE_ID"] = np.arange(603, dtype=np.uint64)
    source["Z"] = np.concatenate((np.tile([0.1, 0.2, 0.3], 200), [0.09, 0.4, 0.41]))
    # Every rank ties: the frozen stable sort must retain original row order.
    monkeypatch.setattr(modifier, "_stable_uniform", lambda ids, **k: np.ones(len(ids)))
    cache = modifier._InputPreparationCache(config)
    randoms = cache.normalized(source, role="random", config=config)
    artifact = dict(
        z_edges=[0.1, 0.2, 0.3, 0.4],
        target_probability_mass=[0.2, 0.3, 0.5],
        sample_identity="fixture",
    )
    kwargs = dict(
        data_count=4, exact_random_data_ratio=50, expected_sample_identity="fixture", config=config
    )
    serial, serial_record = modifier.thin_randoms_to_smooth_target(randoms, artifact, **kwargs)
    cached, cached_record = modifier.thin_randoms_to_smooth_target(
        randoms, artifact, input_cache=cache, **kwargs
    )
    expected_ids = np.sort(
        np.concatenate((np.arange(0, 120, 3), np.arange(1, 181, 3), np.arange(2, 302, 3)))
    )
    np.testing.assert_array_equal(cached["SOURCE_ID"], expected_ids)
    assert serial_record == cached_record
    for key in serial:
        np.testing.assert_array_equal(cached[key], serial[key])
    with pytest.raises(ValueError, match="support bin"):
        modifier.thin_randoms_to_smooth_target(
            randoms, artifact, input_cache=cache, **{**kwargs, "data_count": 10}
        )
    with pytest.raises(ValueError, match="another tracer"):
        modifier.thin_randoms_to_smooth_target(
            randoms, artifact, input_cache=cache, **{**kwargs, "expected_sample_identity": "wrong"}
        )
    with pytest.raises(ValueError, match="immutable catalogue"):
        modifier.thin_randoms_to_smooth_target(dict(randoms), artifact, input_cache=cache, **kwargs)


def test_cached_random_order_preserves_distinct_class_counts_and_radial_predictions():
    config = load_analysis_config()
    source = catalogue(600)
    source["SOURCE_ID"] = np.arange(600, dtype=np.uint64)
    source["Z"] = np.repeat([0.5, 0.7, 0.9], 200)
    original = copy.deepcopy(source)
    cache = modifier._InputPreparationCache(config)
    prepared = cache.normalized(source, role="random", config=config)
    parent_model = {
        "calibration_mock_ids": [0, 1],
        "sample_identity": "exclusive-parent",
        "convergence_evidence": {"converged": True},
        "z_edges": [0.4, 0.6, 0.8, 1.0],
        "mean_counts": [2.0, 3.0, 5.0],
        "target_probability_mass": [0.2, 0.3, 0.5],
    }
    ratio = config["selection"]["radial_randoms"]["exact_random_data_ratio"]
    selections = []
    for sigma0, count in ((0.003, 2), (0.01, 3), (0.03, 4)):
        identity = f"exclusive-class-{sigma0}"
        radial = modifier.forward_radial_model(
            parent_model,
            observed_edges=parent_model["z_edges"],
            sigma0=sigma0,
            sample_identity=identity,
        )
        kwargs = dict(
            data_count=count,
            exact_random_data_ratio=ratio,
            expected_sample_identity=identity,
            config=config,
        )
        serial, expected = modifier.thin_randoms_to_smooth_target(prepared, radial, **kwargs)
        cached, actual = modifier.thin_randoms_to_smooth_target(
            prepared, radial, input_cache=cache, **kwargs
        )
        assert actual == expected
        assert len(cached["SOURCE_ID"]) == count * ratio
        for key in serial:
            assert cached[key].dtype == serial[key].dtype
            np.testing.assert_array_equal(cached[key], serial[key])
        selections.append(actual["target_counts_by_bin"])
    assert len({tuple(counts) for counts in selections}) == 3
    assert len(cache.catalogues) == 1 and len(cache.orders) == 1
    for key in original:
        np.testing.assert_array_equal(source[key], original[key])


def test_input_driver_passes_worker_count_and_limits_nested_threads(monkeypatch, tmp_path):
    import pzbao.selection as pipeline_selection

    monkeypatch.setattr(pipeline_selection, "select_stage", lambda run, unit, path, *, cpus: cpus)
    assert driver._science("select", {}, {}, tmp_path, 4) == 4
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "4")
    # Restore all environment changes after this test.
    for name in (
        "NTHREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        monkeypatch.setenv(name, "1")
    driver._configure_threads("select", 4)
    import os

    assert os.environ["OMP_NUM_THREADS"] == os.environ["OPENBLAS_NUM_THREADS"] == "1"
    driver._configure_threads("cluster", 4)
    assert os.environ["OMP_NUM_THREADS"] == "4"


def test_complete_input_publication_matches_uncached_serial_values(monkeypatch, tmp_path):
    import pzbao.selection as pipeline_selection

    config = load_analysis_config()

    class FixtureCosmology:
        def comoving_radial_distance(self, z):
            return np.asarray(z) * 2500.0

        def growth_rate(self, z):
            return 0.8

    monkeypatch.setattr(
        pipeline_selection, "build_mock_cosmology", lambda config: FixtureCosmology()
    )
    tracer = "BGS_BRIGHT-21.5"
    data = catalogue(40, 0.12, 0.38)
    randoms = catalogue(10000)
    randoms["SOURCE_ID"] = np.arange(10000, dtype=np.uint64)

    def read_parent(records, *, mock_id, cap, include_randoms=True, **kwargs):
        parent = {key: values.copy() for key, values in data.items()}
        parent["SOURCE_ID"] = np.arange(40, dtype=np.uint64) + 100 * mock_id
        parent["Z"] += (mock_id % 3 - 1) * 0.001
        return dict(data=parent, reservoir=None, random=randoms)

    monkeypatch.setattr(pipeline_selection, "_read_parent", read_parent)
    monkeypatch.setattr(
        pipeline_selection,
        "read_catalog_registration",
        lambda run, tracer: {
            "sources": [
                dict(role="reservoir", mock_id=mock_id, cap=cap)
                for mock_id in config["analysis"]["mock_ids"]
                for cap in ("NGC", "SGC")
            ]
        },
    )

    # Isolate orchestration from the spline optimizer; keep actual selection, FKP,
    # Cartesian conversion, geometry and weighted-redshift reductions on all 25 mocks.
    def radial_model(histogram, edges, *, sample_identity, **kwargs):
        mass = np.mean(histogram, axis=0).astype(float) + 1.0
        mass /= mass.sum()
        return dict(
            z_edges=edges.tolist(),
            target_probability_mass=mass.tolist(),
            sample_identity=sample_identity,
            convergence_evidence={"converged": True},
        )

    monkeypatch.setattr(pipeline_selection, "fit_smooth_radial_model", radial_model)
    cache_type = pipeline_selection._InputPreparationCache
    run = dict(
        config=config,
        metadata=dict(
            run_name="fixture",
            code_commit="a",
            analysis_config_sha256="b",
            workload="pilot",
            catalog_root=str(tmp_path),
        ),
    )
    unit = dict(tracer=tracer, realization_ids=config["analysis"]["mock_ids"])
    saved = []
    for name, cpus, cache_factory in (
        ("serial", 1, lambda *a, **k: None),
        ("parallel", 2, cache_type),
    ):
        monkeypatch.setattr(pipeline_selection, "_InputPreparationCache", cache_factory)
        # A moderate boundary forces real random-rank work through several workers.
        monkeypatch.setattr(loader, "_ROW_CHUNK_BYTES", 16000)
        destination = tmp_path / name
        destination.mkdir()
        pipeline_selection.select_stage(run, unit, destination, cpus=cpus)
        saved.append((destination / "preparation.json").read_bytes())
    assert saved[0] == saved[1]
    published = json.loads(saved[1])
    assert len(published["conditions"]) == 5
    assert published["realization_ids"] == config["analysis"]["mock_ids"]
    assert all(
        value["parent"]["denominator"] > 0 for value in published["effective_redshifts"].values()
    )


@pytest.mark.parametrize("cpus", [0, -1, True, 1.5])
def test_input_worker_count_must_be_positive_integer(cpus, tmp_path):
    from pzbao.selection import select_stage

    with pytest.raises(ValueError, match="positive CPU"):
        select_stage({}, {}, tmp_path, cpus=cpus)
