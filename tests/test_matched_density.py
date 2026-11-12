"""V_eff-matched LRG2/LRG3 density tiers: rule, handoff producer, validator, readers.

Density-matching revision 2.15. The factor table is
recomputed from frozen inputs; the handoff is produced through the real selector on a small
per-bin primary fixture and consumed by the density successor's validator and readers.
"""

from copy import deepcopy
import functools
import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml
from click.testing import CliRunner

from pzbao import fitting, run as driver
from pzbao import selection as pipeline_selection
from pzbao.catalog import FULL_SKY_AREA_DEG2
from pzbao.config import analysis_config_sha256, load_analysis_config, validate_analysis_config
from pzbao.config import density
from pzbao.config.density import (
    CAPS,
    CONDITIONS,
    NAME,
    density_preparation,
    density_run_config,
    matched_density_plan,
    plan_counts,
)
from pzbao.figures import data
from tests.test_boundary_support import per_bin_preparation_fields
from tests.test_selection import _catalog

ROOT = Path(__file__).resolve().parents[1]
BINS = ("LRG1", "LRG2", "LRG3")
CODE = {"code_commit": "a" * 40, "repository": "fixture", "dirty": False, "code_state_sha256": "x"}

# Frozen ceilings (minimum attainable count over the 25 realizations) and primary counts of the
# kept LRG selection the per-bin primary reuses and regenerates, read from its published
# select/units/LRG/preparation.json (run full-v2--scan--R15, 2026-10-14). The shell volumes
# come from the repository's density figure data; cap areas, P0 and targets from the config.
KEPT_CEILINGS = {
    "sigma_0.003": {"NGC": (579343, 898010, 841721), "SGC": (327557, 500301, 472375)},
    "sigma_0.001": {"NGC": (579188, 898382, 841918), "SGC": (327606, 499909, 472855)},
}
KEPT_PRIMARY = {
    "sigma_0.003": {"NGC": (542086, 118183, 69471), "SGC": (305394, 66580, 39138)},
    "sigma_0.001": {"NGC": (135521, 29545, 17367), "SGC": (76348, 16645, 9784)},
}


def frozen_lrg_inputs(config):
    """A source preparation holding the kept selection's volumes, ceilings and primary counts."""
    areas = config["tracers"]["LRG"]["footprint"]["area_deg2"]
    totals = {
        row["measurement_bin"]: row["volume_mpc_h3"]
        for row in json.loads((ROOT / "paper/figures/source/baseline/density.json").read_text())
        if row["condition"] == "native" and row["measurement_bin"] in BINS
    }
    assert set(totals) == set(BINS)
    conditions = {}
    for condition in CONDITIONS:
        conditions[condition] = {}
        for cap in CAPS:
            fraction = areas[cap] / sum(areas.values())
            conditions[condition][cap] = {
                "counts": dict(zip(BINS, KEPT_PRIMARY[condition][cap])),
                "volumes_mpc_h3": {name: totals[name] * fraction for name in BINS},
                "attainable_counts_by_mock": [list(KEPT_CEILINGS[condition][cap])] * 25,
            }
    return {"tracer": "LRG", "realization_ids": list(range(25)), "conditions": conditions}


def test_matching_rule_reproduces_the_frozen_factor_table():
    config = load_analysis_config()
    plan = matched_density_plan(frozen_lrg_inputs(config), config)
    inputs = plan["inputs"]
    assert inputs["reference_area_deg2"] == 30000.0
    assert inputs["reference_f_sky"] == pytest.approx(
        30000.0 / FULL_SKY_AREA_DEG2, rel=0, abs=1e-15
    )
    assert inputs["volume_ratio"] == pytest.approx(4.2359, abs=5e-5)
    assert inputs["p0_h3_mpc3"] == config["tracers"]["LRG"]["fkp_p0_h3_mpc3"] == 10000.0
    expected = {
        # tier, bin: n_M, factor over SPHEREx, requested NGC/SGC, selected NGC/SGC, bounded
        ("sigma_0.003", "LRG2"): (5.712e-4, 8.10, (957549, 539452), (898010, 500301), True),
        ("sigma_0.003", "LRG3"): (9.771e-5, 3.09, (214806, 121015), (214806, 121015), False),
        ("sigma_0.001", "LRG2"): (4.459e-5, 2.53, (74749, 42111), (74749, 42111), False),
        ("sigma_0.001", "LRG3"): (1.774e-5, 2.25, (39006, 21974), (39006, 21974), False),
    }
    for (condition, name), (n_matched, factor, requested, selected, bounded) in expected.items():
        row = plan["conditions"][condition][name]
        assert row["in_experiment"] and not row["unattainable"]
        assert row["n_matched_h3_mpc3"] == pytest.approx(n_matched, rel=2e-4)
        assert row["factor_matched_over_spherex"] == pytest.approx(factor, abs=5e-3)
        assert tuple(row["caps"][cap]["requested"] for cap in CAPS) == requested
        assert tuple(row["caps"][cap]["selected"] for cap in CAPS) == selected
        assert row["bounded"] is bounded
        if bounded:
            assert row["veff_fraction"] == pytest.approx(0.9793, abs=5e-4)
            assert row["shortfall"] == {"NGC": 957549 - 898010, "SGC": 539452 - 500301}
            assert row["factor_selected_over_primary"] == pytest.approx(7.57, abs=5e-3)
        else:
            # The per-cap floor drops less than one object, so the fraction sits just below 1.
            assert 1 - 1e-4 < row["veff_fraction"] <= 1.0
            assert row["shortfall"] == {"NGC": 0, "SGC": 0}
    for condition in CONDITIONS:
        lrg1 = plan["conditions"][condition]["LRG1"]
        assert lrg1["unattainable"] and not lrg1["in_experiment"] and not lrg1["bounded"]
        assert lrg1["g_matched"] > 1 and lrg1["n_matched_h3_mpc3"] is None
        assert lrg1["veff_target_mpc_h3"] > sum(c["volume_mpc_h3"] for c in lrg1["caps"].values())
        assert plan_counts(plan, condition, "NGC")["LRG1"] == KEPT_PRIMARY[condition]["NGC"][0]
    # The neighbouring team value (0.75) is a different plan: nothing is defaulted inside.
    neighbour = deepcopy(config)
    block = neighbour["selection"]["veff_matched_density"]
    block["reference_area_deg2"] = 0.75 * FULL_SKY_AREA_DEG2
    block["reference_f_sky"] = 0.75
    other = matched_density_plan(frozen_lrg_inputs(neighbour), neighbour)
    assert other["conditions"]["sigma_0.003"]["LRG2"]["veff_fraction"] == pytest.approx(
        0.9495, abs=5e-4
    )
    assert other["conditions"]["sigma_0.003"]["LRG3"][
        "factor_matched_over_spherex"
    ] == pytest.approx(3.19, abs=5e-3)
    assert other != plan
    lighter = deepcopy(config)
    lighter["tracers"]["LRG"]["fkp_p0_h3_mpc3"] = 8000.0
    assert matched_density_plan(frozen_lrg_inputs(lighter), lighter) != plan
    # Doubling P0 makes LRG2 unattainable; the configured bin list must then say so.
    heavier = deepcopy(config)
    heavier["tracers"]["LRG"]["fkp_p0_h3_mpc3"] = 20000.0
    with pytest.raises(ValueError, match="exactly the attainable bins"):
        matched_density_plan(frozen_lrg_inputs(heavier), heavier)


@pytest.mark.parametrize(
    "defect",
    [
        "unauthorized_bounded",
        "fewer_than_primary",
        "noninteger_matrix",
        "mismatched_evidence",
        "wrong_mock_order",
        "unattainable_in_experiment",
        "attainable_outside_experiment",
        "foreign_authority",
    ],
)
def test_matching_rule_refuses_unauthorized_bounding_and_inconsistent_evidence(defect):
    config = load_analysis_config()
    source = frozen_lrg_inputs(config)
    group = source["conditions"]["sigma_0.003"]["SGC"]
    if defect == "unauthorized_bounded":
        config["selection"]["veff_matched_density"]["bounded_allowed"] = []
    elif defect == "fewer_than_primary":
        group["counts"]["LRG3"] = 121016  # one more than the matched request
    elif defect == "noninteger_matrix":
        group["attainable_counts_by_mock"] = [
            list(map(float, row)) for row in group["attainable_counts_by_mock"]
        ]
    elif defect in ("mismatched_evidence", "wrong_mock_order"):
        group["selection_diagnostics_by_mock"] = [
            {
                "realization_id": mock,
                "selection": {
                    "bins": {
                        name: {
                            "attained_count": group["counts"][name],
                            "maximum_attainable_count": group["attainable_counts_by_mock"][mock][i],
                        }
                        for i, name in enumerate(BINS)
                    }
                },
            }
            for mock in range(25)
        ]
        if defect == "mismatched_evidence":
            group["selection_diagnostics_by_mock"][24]["selection"]["bins"]["LRG3"][
                "attained_count"
            ] += 1
        else:
            group["selection_diagnostics_by_mock"][24]["realization_id"] = 23
    elif defect == "unattainable_in_experiment":
        config["selection"]["veff_matched_density"]["measurement_bins"] = ["LRG1", "LRG2", "LRG3"]
    elif defect == "attainable_outside_experiment":
        config["selection"]["veff_matched_density"]["measurement_bins"] = ["LRG2"]
    else:
        config["selection"]["veff_matched_density"][
            "count_authority"
        ] = "doubled_lrg23_exact_counts"
    with pytest.raises(ValueError):
        matched_density_plan(source, config)
    if defect in ("mismatched_evidence", "wrong_mock_order"):
        # Consistent evidence is accepted and checked, not ignored.
        rows = group["selection_diagnostics_by_mock"]
        rows[24]["realization_id"] = 24
        rows[24]["selection"]["bins"]["LRG3"]["attained_count"] = group["counts"]["LRG3"]
        matched_density_plan(source, config)


# ------------------------------------------------------------------ small per-bin primary fixture

SMALL_VOLUMES = {"NGC": 2.0e5, "SGC": 1.5e5}
SMALL_LAYOUT = {"LRG1": (0.45, 0.55, 75), "LRG2": (0.65, 0.75, 50), "LRG3": (0.85, 0.95, 75)}
SMALL_EDGES = np.linspace(0.4, 1.0, 7)


def small_catalog(mock, cap):
    start = 400 * mock + (20000 if cap == "SGC" else 0)
    redshifts = np.concatenate(
        [np.repeat([low, high], count) for low, high, count in SMALL_LAYOUT.values()]
    )
    return {
        "data": _catalog(range(start, start + len(redshifts)), redshifts),
        "reservoir": None,
    }


def small_source_preparation(config):
    """The per-bin primary's published LRG preparation at toy volumes and ceilings."""
    targets = {tier["label"]: tier["target_nbar_h3_mpc3"] for tier in config["selection"]["tiers"]}
    attainable = [SMALL_LAYOUT[name][2] * 2 for name in BINS]
    conditions = {}
    for tier in config["selection"]["tiers"]:
        label = tier["label"]
        conditions[label] = {}
        for cap in CAPS:
            volume = SMALL_VOLUMES[cap]
            counts = {
                name: (
                    attainable[i]
                    if targets[label] is None
                    else min(attainable[i], int(math.floor(targets[label][name] * volume)))
                )
                for i, name in enumerate(BINS)
            }
            identity = f"LRG|tier={label}|cap={cap}"
            conditions[label][cap] = {
                "counts": counts,
                "volumes_mpc_h3": {name: volume for name in BINS},
                "attainable_counts_by_mock": [list(attainable) for _ in range(25)],
                "radial": {"sample_identity": identity, "z_edges": SMALL_EDGES.tolist()},
                "selection_diagnostics_by_mock": [
                    {
                        "realization_id": mock,
                        "selection": {
                            "bins": {
                                name: {
                                    "attained_count": counts[name],
                                    "maximum_attainable_count": attainable[i],
                                }
                                for i, name in enumerate(BINS)
                            }
                        },
                        "randoms": {
                            "available_counts_by_bin": [5000] * 6,
                            "radial_sample_identity": identity,
                            "exact_random_data_ratio": 50,
                            "retained_rows_unchanged": True,
                        },
                    }
                    for mock in range(25)
                ],
            }
    return {
        "tracer": "LRG",
        "realization_ids": list(range(25)),
        "sources": [
            {"role": "reservoir", "mock_id": mock, "cap": cap} for cap in CAPS for mock in range(25)
        ],
        "conditions": conditions,
        "effective_redshifts": {label: {"parent": {"growth_rate": 0.8}} for label in conditions},
        "power_geometry": {"boxsize": [2000.0] * 3},
        "catalogues": [],
        **per_bin_preparation_fields(config, "LRG"),
    }


def publish_small_primary(tmp_path, monkeypatch, *, name="primary", config=None):
    """An eight-command per-bin primary run with its published LRG selection unit."""
    monkeypatch.setattr(driver, "_code_identity", lambda **kwargs: dict(CODE))
    config = load_analysis_config() if config is None else config
    monkeypatch.setattr(
        driver,
        "load_analysis_config",
        lambda path=None: deepcopy(config) if path is None else load_analysis_config(path),
    )
    run = driver.create_run(tmp_path, name, workload="full_grid", catalog_root=tmp_path)
    unit = run["root"] / "select" / "units" / "LRG"
    unit.mkdir(parents=True)
    result = {
        **{k: run["metadata"][k] for k in ("run_name", "code_commit", "analysis_config_sha256")},
        "stage": "select",
        "unit": {"unit_id": "LRG", "tracer": "LRG", "realization_ids": list(range(25))},
    }
    (unit / "result.yaml").write_text(yaml.safe_dump(result))
    preparation = small_source_preparation(run["config"])
    (unit / "preparation.json").write_text(json.dumps(preparation))
    (run["root"] / "select" / "selection.yaml").write_text(
        yaml.safe_dump(
            {
                "stage": "select",
                "run_name": run["metadata"]["run_name"],
                "analysis_config_sha256": run["metadata"]["analysis_config_sha256"],
                "units": [{"unit": result["unit"], "path": "units/LRG/result.yaml"}],
            }
        )
    )
    return run, preparation


@functools.lru_cache(maxsize=None)
def _produced_handoff():
    """Produce the handoff once through the real selector and radial producer."""
    import tempfile

    from _pytest.monkeypatch import MonkeyPatch

    tmp_path = Path(tempfile.mkdtemp(prefix="matched-density-"))
    patch = MonkeyPatch()
    try:
        run, preparation = publish_small_primary(tmp_path, patch)
        patch.setattr(
            "pzbao.catalog._read_parent",
            lambda records, *, tracer, mock_id, cap, config, include_randoms=True, parallel=None: (
                small_catalog(mock_id, cap)
            ),
        )
        plan, handoff = density.produce_handoff(run, tmp_path / "handoff", "prepare-radial")
    finally:
        patch.undo()
    return run, preparation, plan, handoff, tmp_path / "handoff"


@pytest.fixture
def produced():
    run, preparation, plan, handoff, output = _produced_handoff()
    return run, deepcopy(preparation), deepcopy(plan), deepcopy(handoff), output


def test_producer_selects_nested_matched_tiers_through_the_real_selector(produced):
    run, preparation, plan, handoff, output = produced
    config = run["config"]
    assert (output / "complete.json").exists() and not (output / "failure.json").exists()
    complete = json.loads((output / "complete.json").read_text())
    assert complete["resolved_snapshot_sha256"] == handoff["resolved_snapshot_sha256"]
    for condition in CONDITIONS:
        for cap in CAPS:
            counts = plan_counts(plan, condition, cap)
            baseline = plan_counts(plan, condition, cap, "baseline")
            assert (
                counts["LRG1"]
                == baseline["LRG1"]
                == preparation["conditions"][condition][cap]["counts"]["LRG1"]
            )
            records = handoff["definition"]["selection_records"][condition][cap]
            assert [row["mock"] for row in records] == list(range(25))
            for row in records:
                assert row["pairing"]["bins"] == {
                    name: {"baseline": baseline[name], "selected": counts[name]} for name in BINS
                }
                assert row["pairing"]["common_objects"] == sum(baseline.values())
                assert sum(row["histogram"]) == sum(counts.values())
                native = row["native_selector_record"]["bins"]
                assert all(native[name]["attained_count"] == counts[name] for name in BINS)
    # LRG2 at 0.003 is bounded at its ceiling in the larger cap only; nothing else is bounded.
    lrg2 = plan["conditions"]["sigma_0.003"]["LRG2"]
    assert (
        lrg2["bounded"] and lrg2["caps"]["NGC"]["selected"] == lrg2["caps"]["NGC"]["ceiling"] == 100
    )
    assert lrg2["caps"]["NGC"]["requested"] > 100 and lrg2["caps"]["SGC"]["requested"] < 100
    assert lrg2["shortfall"]["SGC"] == 0 and lrg2["veff_fraction"] < 1
    assert not any(
        row["bounded"]
        for condition in CONDITIONS
        for name, row in plan["conditions"][condition].items()
        if (condition, name) != ("sigma_0.003", "LRG2")
    )
    for condition in CONDITIONS:
        for cap in CAPS:
            radial = handoff["definition"]["cap_products"][condition][cap]["radial"]
            assert radial["sample_identity"] == f"LRG|tier={condition}|density={NAME}|cap={cap}"
            assert radial["convergence_evidence"]["converged"]
            support = handoff["definition"]["cap_products"][condition][cap]["random_support"]
            assert support["randoms_per_mock"] == 50 * sum(
                plan_counts(plan, condition, cap).values()
            )
    assert handoff["scope"]["conditions"] == list(CONDITIONS)
    assert handoff["scientific_acceptance"] is False
    assert len(list(output.glob("sigma_0.00*--*--mock-*.json"))) == 100
    # Selection records come from the real selector on real galaxies: nesting holds per bin.
    raw = small_catalog(3, "NGC")
    histogram, proof = density.select_pair(raw, preparation, config, plan, "sigma_0.003", 3, "NGC")
    assert (
        proof["pairing"]["bins"]["LRG1"]["selected"] == proof["pairing"]["bins"]["LRG1"]["baseline"]
    )
    assert proof["pairing"]["bins"]["LRG2"]["selected"] == 100
    assert histogram.sum() == sum(plan_counts(plan, "sigma_0.003", "NGC").values())


def test_pairing_proof_rejects_changed_noise_lost_members_or_duplicates(produced):
    from pzbao.diagnostics.selection import selected_data

    run, preparation, plan, _, _ = produced
    config = run["config"]
    raw = small_catalog(0, "SGC")
    counts = plan_counts(plan, "sigma_0.001", "SGC")
    candidate_preparation = deepcopy(preparation)
    candidate_preparation["conditions"]["sigma_0.001"]["SGC"]["counts"] = counts
    baseline, _ = selected_data(raw, preparation, config, "LRG", "sigma_0.001", 0, "SGC")
    candidate, _ = selected_data(
        raw, candidate_preparation, density.selection_config(config), "LRG", "sigma_0.001", 0, "SGC"
    )
    expected = {
        name: {
            "baseline": plan_counts(plan, "sigma_0.001", "SGC", "baseline")[name],
            "selected": counts[name],
        }
        for name in BINS
    }
    proof = density.paired_membership(baseline, candidate, config, expected)
    assert proof["object_noise_and_ranks_identical"] and proof["common_objects"] == len(
        baseline["SOURCE_ID"]
    )
    stable = config["selection"]["stable_randomness"]
    for defect in ("noise", "lost_member", "duplicate", "count"):
        damaged = {key: value.copy() for key, value in candidate.items()}
        row = int(np.flatnonzero(damaged["SOURCE_ID"] == baseline["SOURCE_ID"][0])[0])
        if defect == "noise":
            damaged[stable["photoz_deviate_column"]][row] += 0.01
        elif defect == "lost_member":
            damaged["SOURCE_ID"][row] = 999999
        elif defect == "duplicate":
            damaged["SOURCE_ID"][row] = damaged["SOURCE_ID"][(row + 1) % len(damaged["SOURCE_ID"])]
        else:
            damaged = {key: value[1:] for key, value in candidate.items()}
        with pytest.raises(ValueError):
            density.paired_membership(baseline, damaged, config, expected)


@pytest.mark.parametrize("mass", [[0.101, 0.199, 0.7], [1 / 3] * 3, [0.1, 0.1, 0.8]])
def test_random_targets_match_the_native_thinner_exactly(mass):
    config = load_analysis_config()
    rank = config["selection"]["stable_randomness"]["random_thinning_rank_column"]
    randoms = {
        "RA": np.arange(900, dtype=float),
        "DEC": np.zeros(900),
        "Z": np.repeat([0.5, 0.7, 0.9], 300),
        rank: np.tile(np.arange(300) / 300, 3),
    }
    radial = {
        "sample_identity": "matched-test",
        "z_edges": [0.4, 0.6, 0.8, 1.0],
        "target_probability_mass": mass,
    }
    selected, proof = pipeline_selection.thin_randoms_to_smooth_target(
        randoms,
        radial,
        data_count=7,
        exact_random_data_ratio=50,
        expected_sample_identity="matched-test",
        config=config,
    )
    assert density.random_targets(mass, 350).tolist() == proof["target_counts_by_bin"]
    assert len(selected["Z"]) == 350 and proof["retained_rows_unchanged"]


@pytest.mark.parametrize("defect", [None, "shortage", "wrong_identity", "missing_mock"])
def test_saved_random_support_checks_every_mock(defect):
    radial = {
        "sample_identity": "new-density",
        "z_edges": [0.4, 0.6, 0.8, 1.0],
        "target_probability_mass": [0.101, 0.199, 0.7],
        "calibration_mock_ids": list(range(25)),
    }
    rows = [
        {
            "realization_id": mock,
            "randoms": {
                "available_counts_by_bin": [300, 300, 300],
                "radial_sample_identity": "primary",
                "exact_random_data_ratio": 50,
                "retained_rows_unchanged": True,
            },
        }
        for mock in range(25)
    ]
    group = {
        "radial": {**radial, "sample_identity": "primary"},
        "selection_diagnostics_by_mock": rows,
    }
    if defect == "shortage":
        rows[11]["randoms"]["available_counts_by_bin"][2] = 244
    elif defect == "wrong_identity":
        rows[11]["randoms"]["radial_sample_identity"] = "other"
    elif defect == "missing_mock":
        rows.pop()
    if defect is None:
        result = density.random_support(radial, group, 7, 50)
        assert len(result["checks"]) == 25 and result["randoms_per_mock"] == 350
        assert result["target_counts_by_bin"] == [35, 70, 245]
    else:
        with pytest.raises(ValueError):
            density.random_support(radial, group, 7, 50)


def test_handoff_round_trip_preserves_unrelated_tiers_and_refuses_changed_definitions(
    produced, tmp_path
):
    run, preparation, plan, handoff, output = produced
    path = output / "prepared-inputs.json"
    candidate, configuration = density.read_handoff(
        path, run, preparation, handoff["resolved_snapshot_sha256"]
    )
    assert configuration == density.selection_config(run["config"])
    for label in ("native", "sigma_0.01", "sigma_0.03"):
        assert candidate["conditions"][label] == preparation["conditions"][label]
    for key in ("sources", "geometry", "windows", "power_geometry", "effective_redshifts"):
        assert candidate[key] == preparation[key]
    for condition in CONDITIONS:
        for cap in CAPS:
            group = candidate["conditions"][condition][cap]
            assert group["counts"] == plan_counts(plan, condition, cap)
            assert "selection_diagnostics_by_mock" not in group
            assert len(group["density_selection_by_mock"]) == 25
            assert group["actual_random_rows_validated"] is False
    with pytest.raises(ValueError, match="snapshot"):
        density.read_handoff(path, run, preparation, "0" * 64)
    other = {**run, "metadata": {**run["metadata"], "run_name": "another-primary"}}
    with pytest.raises(ValueError, match="changed source"):
        density.read_handoff(path, other, preparation, handoff["resolved_snapshot_sha256"])
    for defect in (
        "geometry",
        "source",
        "configuration",
        "missing_mock",
        "counts",
        "radial_mean",
        "matching",
    ):
        changed = deepcopy(handoff)
        definition = changed["definition"]
        if defect == "geometry":
            definition["preparation"]["geometry"]["NGC"]["LRG1"]["calibration_mock_ids"] = [0]
        elif defect == "source":
            definition["source"]["code_commit"] = "2" * 40
        elif defect == "configuration":
            definition["configuration"]["measurement"]["reconstruction"]["iterations"] += 1
        elif defect == "missing_mock":
            definition["selection_records"]["sigma_0.003"]["NGC"].pop()
        elif defect == "counts":
            definition["preparation"]["conditions"]["sigma_0.001"]["SGC"]["counts"]["LRG3"] += 1
        elif defect == "radial_mean":
            definition["cap_products"]["sigma_0.003"]["NGC"]["radial"]["mean_counts"][0] += 1
        else:
            definition["matching"]["inputs"]["reference_f_sky"] = 0.75
        changed["resolved_snapshot_sha256"] = analysis_config_sha256(definition)
        damaged = tmp_path / f"{defect}.json"
        damaged.write_text(json.dumps(changed))
        with pytest.raises(ValueError):
            density.read_handoff(damaged, run, preparation, changed["resolved_snapshot_sha256"])


def _rehash(handoff):
    result = deepcopy(handoff)
    result["resolved_snapshot_sha256"] = analysis_config_sha256(result["definition"])
    return result, result["resolved_snapshot_sha256"]


def test_density_successor_binds_the_two_tier_handoff_and_recomputes_every_count(
    produced, tmp_path
):
    run, preparation, plan, handoff, output = produced
    config = load_analysis_config()
    digest = handoff["resolved_snapshot_sha256"]
    derived = density_run_config(config, handoff, digest)
    validate_analysis_config(derived)
    workload = derived["analysis"]["workloads"]["density"]
    assert workload["sigma_z_tiers"] == ["sigma_0.001", "sigma_0.003"]
    assert workload["measurement_bins"] == ["LRG2", "LRG3"]
    assert derived["selection"]["density_run"]["handoff_sha256"] == digest
    for key in ("physics", "tracers", "measurement"):
        assert derived[key] == config[key]
    assert len(driver.stage_units("reconstruct", derived, "density")) == 50
    assert len(driver.stage_units("fit", derived, "density")) == 8 * 26
    prepared = density_preparation(derived, "LRG")
    assert prepared["geometry"] == {} and prepared["effective_redshifts"] == {}
    assert prepared["selection_source"]["density_handoff_sha256"] == digest
    assert set(prepared["conditions"]) == {t["label"] for t in config["selection"]["tiers"]}
    for condition in CONDITIONS:
        for cap in CAPS:
            assert prepared["conditions"][condition][cap]["counts"] == plan_counts(
                plan, condition, cap
            )
    with pytest.raises(ValueError, match="supplies only LRG"):
        density_preparation(derived, "BGS_BRIGHT-21.5")
    with pytest.raises(ValueError, match="differs from the identified"):
        density_run_config(config, handoff, "0" * 64)
    from pzbao.config.sensitivity import sensitivity_run_configs
    from pzbao.config.transfer import transfer_run_configs
    from tests.legacy_config import load_schema16_config

    for source in (
        sensitivity_run_configs(config)["scan--R25"],
        transfer_run_configs(config)["self--R25"],
        derived,
        load_schema16_config(),
    ):
        with pytest.raises(ValueError, match="schema-17 per-bin primary configuration"):
            density_run_config(source, handoff, digest)
    changed = deepcopy(config)
    changed["selection"]["radial_randoms"]["stable_thinning_seed"] += 1
    with pytest.raises(ValueError, match="changes inputs, weights, streams or cohort"):
        density_run_config(changed, handoff, digest)
    # A changed count, however consistently rehashed, no longer follows the recorded rule.
    damaged = deepcopy(handoff)
    damaged["definition"]["selection_records"]["sigma_0.003"]["NGC"][3]["pairing"]["bins"]["LRG2"][
        "selected"
    ] += 1
    damaged, damaged_digest = _rehash(damaged)
    with pytest.raises(ValueError, match="exact matched counts"):
        density_run_config(config, damaged, damaged_digest)
    damaged = deepcopy(handoff)
    damaged["definition"]["matching"]["conditions"]["sigma_0.001"]["LRG3"]["caps"]["SGC"][
        "selected"
    ] += 1
    damaged, damaged_digest = _rehash(damaged)
    with pytest.raises(ValueError, match="matching rule"):
        density_run_config(config, damaged, damaged_digest)
    # A different reference sky fraction or P0, carried consistently through the embedded
    # configuration, is recomputed and refused: the plan has no hidden defaults.
    for key, value in (("reference_f_sky", 0.75), ("p0", 9000.0)):
        other = deepcopy(config)
        damaged = deepcopy(handoff)
        if key == "reference_f_sky":
            block = other["selection"]["veff_matched_density"]
            block["reference_f_sky"], block["reference_area_deg2"] = (
                value,
                value * FULL_SKY_AREA_DEG2,
            )
            other["reporting"]["forecast_target_fsky"] = value
            damaged["definition"]["configuration"]["selection"]["veff_matched_density"] = deepcopy(
                block
            )
        else:
            other["tracers"]["LRG"]["fkp_p0_h3_mpc3"] = value
            damaged["definition"]["configuration"]["tracers"]["LRG"]["fkp_p0_h3_mpc3"] = value
        damaged, damaged_digest = _rehash(damaged)
        with pytest.raises(ValueError, match="matching rule"):
            density_run_config(other, damaged, damaged_digest)
    # A one-tier (legacy doubled) handoff and a wrong radial identity are refused.
    legacy = deepcopy(handoff)
    legacy["scope"]["conditions"] = ["sigma_0.001"]
    legacy, legacy_digest = _rehash(legacy)
    with pytest.raises(ValueError, match="differs from the identified"):
        density_run_config(config, legacy, legacy_digest)
    legacy = deepcopy(handoff)
    for section in ("cap_products", "selection_records"):
        legacy["definition"][section].pop("sigma_0.003")
    legacy, legacy_digest = _rehash(legacy)
    with pytest.raises((ValueError, KeyError)):
        density_run_config(config, legacy, legacy_digest)
    damaged = deepcopy(handoff)
    radial = damaged["definition"]["cap_products"]["sigma_0.001"]["NGC"]["radial"]
    radial["sample_identity"] = "LRG|tier=sigma_0.001|density=doubled_lrg23_exact_counts|cap=NGC"
    damaged["definition"]["preparation"]["conditions"]["sigma_0.001"]["NGC"]["radial"] = deepcopy(
        radial
    )
    damaged, damaged_digest = _rehash(damaged)
    with pytest.raises(ValueError, match="radial identity"):
        density_run_config(config, damaged, damaged_digest)
    # The driver writes the definition only with its handoff, into an existing directory.
    path = tmp_path / "prepared-inputs.json"
    path.write_text(json.dumps(handoff))
    destination = tmp_path / "configs"
    result = CliRunner().invoke(
        driver.main, ["per-bin-yamls", "--output-dir", str(destination), "--only", "primary"]
    )
    assert result.exit_code == 0, (result.output, result.exception)
    result = CliRunner().invoke(
        driver.main,
        ["per-bin-yamls", "--output-dir", str(destination), "--only", "density"]
        + ["--density-handoff", str(path), "--density-snapshot", digest],
    )
    assert result.exit_code == 0, (result.output, result.exception)
    assert {p.name for p in destination.iterdir()} == {"primary.yaml", "density.yaml"}
    reloaded = load_analysis_config(destination / "density.yaml")
    assert reloaded == derived
    assert (
        yaml.safe_load((destination / "density.yaml").read_text())["cluster"]["statistics"] == "xi"
    )
    result = CliRunner().invoke(
        driver.main, ["per-bin-yamls", "--output-dir", str(destination), "--only", "primary"]
    )
    assert result.exit_code != 0 and "refusing to overwrite" in result.output
    result = CliRunner().invoke(
        driver.main, ["per-bin-yamls", "--output-dir", str(destination), "--only", "other"]
    )
    assert result.exit_code != 0 and "unknown per-bin definitions" in result.output


def _identified_sample(config, size=12):
    from pzbao import catalog as loader

    field = loader.reconstruction_field("LRG", config)
    sample = {
        "TARGETID": np.arange(size, dtype=np.int64),
        "RA": np.linspace(1, 359, size),
        "DEC": np.linspace(-80, 80, size),
        "Z": np.linspace(field["zmin"], field["zmax"], size, endpoint=False),
        "WEIGHT": np.linspace(0.5, 1.5, size),
        "PHOTSYS": np.full(size, "N"),
    }
    return loader.assign_source_identity(
        sample, tracer="LRG", mock_id=0, cap="NGC", role="data", config=config
    )


def test_exact_count_override_applies_to_both_matched_tiers_only(produced, monkeypatch):
    run, preparation, plan, handoff, _ = produced
    config = load_analysis_config()
    derived = density_run_config(config, handoff, handoff["resolved_snapshot_sha256"])
    seen = {}
    sample = _identified_sample(derived)
    # The pass's real input cache, built from the run configuration as select_stage builds it;
    # every condition must reuse it although the matched tiers clear their target.
    cache = pipeline_selection._InputPreparationCache(derived)

    def fake_prepare(data, randoms, *, tier_label, config, frozen_counts_by_bin, **kwargs):
        seen[tier_label] = (deepcopy(config["selection"]["tiers"]), dict(frozen_counts_by_bin))
        kwargs["input_cache"].normalized(sample, role="data", config=config)
        return {"Z": np.zeros(1)}, {"Z": np.zeros(1)}, {"stub": True}

    monkeypatch.setattr(pipeline_selection, "prepare_tracer_field", fake_prepare)
    monkeypatch.setattr(pipeline_selection, "add_cartesian_positions", lambda data, *a, **k: data)
    prepared = density_preparation(derived, "LRG")
    raw = {"data": None, "random": None, "reservoir": None}
    for condition in ("native", "sigma_0.001", "sigma_0.003", "sigma_0.01", "sigma_0.03"):
        pipeline_selection._prepared_parent(
            raw,
            prepared,
            tracer="LRG",
            condition=condition,
            mock_id=0,
            cap="NGC",
            cosmology=None,
            config=derived,
            input_cache=cache,
        )
    assert len(cache.catalogues) == 1
    targets = {t["label"]: t["target_nbar_h3_mpc3"] for t in config["selection"]["tiers"]}
    assert set(seen) == set(targets)
    for condition, (tiers, counts) in seen.items():
        # Only the matched tier being selected loses its target; every other tier keeps it.
        for tier in tiers:
            withdrawn = condition in CONDITIONS and tier["label"] == condition
            expected = None if withdrawn else targets[tier["label"]]
            assert tier["target_nbar_h3_mpc3"] == expected, (condition, tier["label"])
        if condition in CONDITIONS:
            assert counts == plan_counts(plan, condition, "NGC")
        else:
            assert counts == preparation["conditions"][condition]["NGC"]["counts"]


def _derived_density_config(handoff):
    return density_run_config(load_analysis_config(), handoff, handoff["resolved_snapshot_sha256"])


def _native_selector_bins(counts):
    """What the selector saves for a tier whose target the exact-count handoff withdrew."""
    return {
        name: {
            "attained_count": count,
            "requested_nbar_h3_mpc3": None,
            "requested_count": None,
            "bounded_by_attainable": False,
            "target_basis": "native_attainable",
        }
        for name, count in counts.items()
    }


def test_matched_labels_follow_the_handoff_and_leave_other_records_alone(produced):
    _, _, plan, handoff, _ = produced
    derived = _derived_density_config(handoff)
    label = density.matched_selection_record
    record = {
        "method": "stable_exact_rank",
        "bins": _native_selector_bins(plan_counts(plan, "sigma_0.003", "NGC")),
    }
    before = deepcopy(record)
    labelled = label(record, derived, "LRG", "sigma_0.003", "NGC")
    assert record == before, "a saved record keeps its bytes"
    assert labelled["method"] == "stable_exact_rank"

    row = plan["conditions"]["sigma_0.003"]["LRG2"]
    bin_record = labelled["bins"]["LRG2"]
    assert bin_record["target_basis"] == "veff_matched_handoff"
    assert bin_record["requested_nbar_h3_mpc3"] == row["n_matched_h3_mpc3"] > 0
    assert bin_record["requested_count"] == row["caps"]["NGC"]["requested"]
    assert bin_record["attained_count"] == row["caps"]["NGC"]["selected"]
    # LRG1 cannot reach the matched effective volume and keeps the primary selection.
    assert labelled["bins"]["LRG1"]["target_basis"] == "primary_selection_retained"
    assert labelled["bins"]["LRG1"]["requested_nbar_h3_mpc3"] is None
    assert labelled["bins"]["LRG1"]["bounded_by_attainable"] is False

    # LRG2 at 0.003 is bounded in the larger cap only; the flag is per cap and agrees with the plan.
    flags = {
        cap: label(record, derived, "LRG", "sigma_0.003", cap)["bins"]["LRG2"][
            "bounded_by_attainable"
        ]
        for cap in CAPS
    }
    assert flags == {"NGC": True, "SGC": False}
    for condition in CONDITIONS:
        for name in BINS:
            assert (
                any(
                    label(record, derived, "LRG", condition, cap)["bins"][name][
                        "bounded_by_attainable"
                    ]
                    for cap in CAPS
                )
                == plan["conditions"][condition][name]["bounded"]
            )

    unbound = load_analysis_config()
    assert label(record, unbound, "LRG", "sigma_0.003", "NGC") is record
    assert label(record, derived, "LRG", "sigma_0.01", "NGC") is record
    assert label(record, derived, "BGS_BRIGHT-21.5", "sigma_0.003", "NGC") is record


def test_field_preparation_records_the_handoff_target_through_the_real_selector(monkeypatch):
    config = deepcopy(load_analysis_config())
    for tier in config["selection"]["tiers"]:
        if tier["label"] == "sigma_0.003":
            tier["target_nbar_h3_mpc3"] = None
    config["selection"]["density_run"] = {
        "handoff": {
            "definition": {
                "matching": {
                    "conditions": {
                        "sigma_0.003": {
                            "LRG1": {
                                "in_experiment": False,
                                "n_matched_h3_mpc3": None,
                                "caps": {"NGC": {"requested": None}},
                                "shortfall": {"NGC": 0},
                            },
                            "LRG2": {
                                "in_experiment": True,
                                "n_matched_h3_mpc3": 3e-6,
                                "caps": {"NGC": {"requested": 3}},
                                "shortfall": {"NGC": 1},
                            },
                            "LRG3": {
                                "in_experiment": True,
                                "n_matched_h3_mpc3": 2e-6,
                                "caps": {"NGC": {"requested": 2}},
                                "shortfall": {"NGC": 0},
                            },
                        }
                    }
                }
            }
        }
    }
    identity = "LRG|sigma_0.003|density=matched|cap=NGC"
    radial = pipeline_selection.fit_smooth_radial_model(
        [[2, 2, 2], [2, 2, 2]],
        [0.4, 0.6, 0.8, 1.0],
        calibration_mock_ids=[0, 1],
        smoothing_method="gaussian_kernel",
        bandwidth=1.0,
        sample_identity=identity,
    )
    monkeypatch.setattr(
        pipeline_selection, "comoving_shell_volume", lambda *args, **kwargs: [1e6, 1e6, 1e6]
    )
    _, _, record = pipeline_selection.prepare_tracer_field(
        _catalog(range(1, 7), [0.45, 0.46, 0.65, 0.66, 0.85, 0.86]),
        _catalog(range(100, 400), np.linspace(0.4, 1.0, 300, endpoint=False)),
        tracer="LRG",
        tier_label="sigma_0.003",
        cap="NGC",
        cosmology=object(),
        radial_artifact=radial,
        sample_identity=identity,
        frozen_counts_by_bin={"LRG1": 2, "LRG2": 2, "LRG3": 2},
        config=config,
    )
    bins = record["selection"]["bins"]
    assert bins["LRG2"]["target_basis"] == "veff_matched_handoff"
    assert bins["LRG2"]["requested_nbar_h3_mpc3"] == 3e-6
    assert bins["LRG2"]["requested_count"] == 3
    assert bins["LRG2"]["attained_count"] == 2
    assert bins["LRG2"]["bounded_by_attainable"] is True
    assert bins["LRG3"]["bounded_by_attainable"] is False
    assert bins["LRG1"]["target_basis"] == "primary_selection_retained"


def test_review_reports_the_handoff_target_for_a_run_published_with_native_labels(
    produced, tmp_path
):
    from pzbao.diagnostics import review

    _, _, plan, handoff, _ = produced
    metadata = {
        "run_name": "density-fixture",
        "code_commit": "a" * 40,
        "analysis_config_sha256": "b" * 64,
        "interface_version": 2,
        "workload": "density",
    }
    route = {
        "tracer": "LRG",
        "measurement_bin": "LRG2",
        "condition": "sigma_0.003",
        "reconstruction_state": "postreconstruction",
    }
    directory = tmp_path / "reconstruct" / "units" / "LRG--sigma_0.003--mock-0"
    directory.mkdir(parents=True)
    unit = {
        "unit_id": directory.name,
        "tracer": "LRG",
        "condition": "sigma_0.003",
        "realization_id": 0,
    }
    (directory / "result.yaml").write_text(
        yaml.safe_dump({**metadata, "stage": "reconstruct", "unit": unit})
    )
    (directory / "reconstruction.json").write_text(
        json.dumps(
            {
                "diagnostics": {
                    cap: {
                        "reconstruction": {"shell_diagnostics": {"shells": []}},
                        "selection": {
                            "selection": {
                                "bins": _native_selector_bins(plan_counts(plan, "sigma_0.003", cap))
                            }
                        },
                    }
                    for cap in CAPS
                }
            }
        )
    )
    run = {
        "root": tmp_path,
        "config": _derived_density_config(handoff),
        "metadata": metadata,
    }
    record, _ = review.collect_route(run, route)
    rows = {row["cap"]: row for row in record["density"]}
    row = plan["conditions"]["sigma_0.003"]["LRG2"]
    assert set(rows) == set(CAPS)
    assert rows["NGC"]["target_basis"] == "veff_matched_handoff"
    assert rows["NGC"]["requested_nbar_h3_mpc3"] == row["n_matched_h3_mpc3"]
    assert rows["NGC"]["requested_count"] == row["caps"]["NGC"]["requested"]
    assert rows["NGC"]["bounded_by_attainable"] is True
    assert rows["SGC"]["bounded_by_attainable"] is False


def test_density_handoff_command_writes_the_count_plan_and_refuses_other_runs(
    tmp_path, monkeypatch
):
    run, _ = publish_small_primary(tmp_path, monkeypatch)
    result = CliRunner().invoke(
        driver.main,
        [
            "density-handoff",
            "counts",
            "--source-run",
            str(run["root"]),
            "--output",
            str(tmp_path / "c"),
        ],
    )
    assert result.exit_code == 0, (result.output, result.exception)
    assert "sigma_0.003 LRG2: 100 / 85" in result.output and "bounded" in result.output
    assert "LRG1" in result.output and "unattainable, primary kept" in result.output
    attempt = json.loads((tmp_path / "c" / "attempt.json").read_text())
    assert attempt["matching"]["inputs"]["count_authority"] == NAME
    assert json.loads((tmp_path / "c" / "complete.json").read_text())["input_handoff"] is None
    with pytest.raises(FileExistsError):
        density.produce_handoff(run, tmp_path / "c", "counts")
    # Anything but a published per-bin primary measurement is refused before any work.
    (run["root"] / "select" / "selection.yaml").unlink()
    with pytest.raises(ValueError, match="not published"):
        density.produce_handoff(run, tmp_path / "d", "counts")
    pilot = driver.create_run(tmp_path, "pilot", workload="pilot", catalog_root=tmp_path)
    with pytest.raises(ValueError, match="published per-bin primary"):
        density.produce_handoff(pilot, tmp_path / "e", "counts")
    scratch = driver.create_run(
        tmp_path, "scratch-run", workload="full_grid", catalog_root=tmp_path, scratch=True
    )
    with pytest.raises(ValueError, match="published per-bin primary"):
        density.produce_handoff(scratch, tmp_path / "f", "counts")
    assert (tmp_path / "f" / "failure.json").exists() is False


def test_schema16_per_bin_primary_is_refused_as_source_and_as_comparison_partner(
    produced, tmp_path, monkeypatch
):
    """A campaign prepared before the matched-density block is re-prepared, never derived from.

    The per-bin definitions prepared on 2026-10-16 were written at schema 16. Neither the
    handoff producer nor the comparison reader accepts such a primary: the density successor
    shares its partner's whole scientific definition, so the operator is told to re-prepare at
    the current configuration instead of pairing two schemas.
    """
    from tests.legacy_config import load_schema16_config

    legacy = load_schema16_config()
    assert legacy["schema_version"] == 16 and "veff_matched_density" not in legacy["selection"]
    run, _ = publish_small_primary(tmp_path, monkeypatch, name="legacy", config=legacy)
    assert run["metadata"]["workload"] == "full_grid" and run["metadata"]["mode"] == "production"
    with pytest.raises(ValueError, match="schema-17 per-bin primary.*re-prepared"):
        density.matched_source(run)
    with pytest.raises(ValueError, match="schema-17 per-bin primary"):
        density.produce_handoff(run, tmp_path / "legacy-handoff", "counts")
    assert not (tmp_path / "legacy-handoff" / "attempt.json").exists()
    _, _, _, handoff, _ = produced
    derived = density_run_config(
        load_analysis_config(), handoff, handoff["resolved_snapshot_sha256"]
    )
    definition = tmp_path / "density.yaml"
    definition.write_text(yaml.safe_dump(derived, sort_keys=False))
    density_run = driver.create_run(
        tmp_path, "density", workload="density", catalog_root=tmp_path, config_path=definition
    )
    with pytest.raises(ValueError, match="beyond the density definition"):
        data.matched_density_comparison(density_run["root"], run["root"])


def test_schema17_forecast_sky_fraction_must_equal_the_matched_reference():
    """The reference sky fraction is written twice (matched rule, reporting forecasts): one value.

    Schema 17 refuses a configuration whose forecast sky fraction differs from, or lacks, the
    matched-density reference; schema 16 carries no matched block and keeps its own value.
    """
    from tests.legacy_config import load_schema16_config

    config = load_analysis_config()
    block = config["selection"]["veff_matched_density"]
    assert config["reporting"]["forecast_target_fsky"] == block["reference_f_sky"]
    validate_analysis_config(config)
    for defect in ("differs", "missing"):
        skewed = deepcopy(config)
        if defect == "differs":
            skewed["reporting"]["forecast_target_fsky"] = 0.75
        else:
            skewed["reporting"].pop("forecast_target_fsky")
        with pytest.raises(ValueError, match="forecast sky fraction must equal"):
            validate_analysis_config(skewed)
    moved = deepcopy(config)
    moved["selection"]["veff_matched_density"]["reference_area_deg2"] = 0.75 * FULL_SKY_AREA_DEG2
    moved["selection"]["veff_matched_density"]["reference_f_sky"] = 0.75
    with pytest.raises(ValueError, match="forecast sky fraction must equal"):
        validate_analysis_config(moved)
    moved["reporting"]["forecast_target_fsky"] = 0.75
    validate_analysis_config(moved)
    legacy = load_schema16_config()
    assert legacy["reporting"]["forecast_target_fsky"] == 0.65
    validate_analysis_config(legacy)


# ------------------------------------------------------------------ eight-command readers


def _profile(half_width, value=1.01):
    parameter = {
        "value": value,
        "minos_bounds": [value - half_width, value + half_width],
        "interval_flags": {"is_valid": True, "lower_valid": True, "upper_valid": True},
        "fixed": False,
    }
    return {
        "parameters": [{**parameter, "basename": "qiso"}, {**parameter, "basename": "qap"}],
        "maximization": {
            "is_valid": True,
            "is_above_max_edm": False,
            "has_reached_call_limit": False,
        },
    }


def _publish_fits(run, routes, *, mocks=(None, *range(25))):
    for unit in driver.stage_units("fit", run["config"], run["metadata"]["workload"]):
        key = (
            unit["tracer"],
            unit["measurement_bin"],
            unit["condition"],
            unit["reconstruction_state"],
        )
        if key not in routes or unit["realization_id"] not in mocks:
            continue
        directory = run["root"] / "fit" / "units" / unit["unit_id"]
        directory.mkdir(parents=True)
        (directory / "profiles.npy").write_bytes(b"fixture")


def test_matched_readers_pair_the_density_run_with_the_per_bin_primary(
    produced, tmp_path, monkeypatch
):
    # The comparison partner is the handoff's recorded source run; a run of the same
    # configuration published elsewhere does not share its source objects.
    primary, _, plan, handoff, _ = produced
    twin, _ = publish_small_primary(tmp_path, monkeypatch)
    assert twin["root"] != primary["root"] and twin["metadata"]["run_name"] == "primary"
    derived = density_run_config(
        load_analysis_config(), handoff, handoff["resolved_snapshot_sha256"]
    )
    definition = tmp_path / "density.yaml"
    definition.write_text(yaml.safe_dump(derived, sort_keys=False))
    density_run = driver.create_run(
        tmp_path, "density", workload="density", catalog_root=tmp_path, config_path=definition
    )
    routes = set(data.matched_density_routes(density_run["config"]))
    assert len(routes) == 8 and all(route[0] == "LRG" for route in routes)
    _publish_fits(
        density_run, routes, mocks=(None, *range(24))
    )  # one matched realization unpublished
    _publish_fits(primary, routes)

    def fake_profiles(path, *, identity):
        unit = Path(path).parent.name
        return _profile(0.01 if unit.endswith("--mean") else 0.03)

    monkeypatch.setattr(fitting, "read_profiles", fake_profiles)
    rows, matching = data.matched_density_fits(density_run["root"])
    assert matching == plan
    assert len(rows) == 8 * 26 * 2
    complete = [r for r in rows if r["status"] == "complete"]
    assert len(complete) == 8 * 25 * 2 and all(r["variant"] == "matched" for r in rows)
    bounded = {(r["condition"], r["measurement_bin"]) for r in rows if r["bounded"]}
    assert bounded == {("sigma_0.003", "LRG2")}
    for row in rows:
        context = plan["conditions"][row["condition"]][row["measurement_bin"]]
        assert row["veff_fraction"] == context["veff_fraction"]
        assert row["matched_nbar_h3_mpc3"] == context["n_matched_h3_mpc3"]
        assert row["spherex_nbar_h3_mpc3"] == context["n_spherex_h3_mpc3"]
        assert row["selected_counts"] == {cap: context["caps"][cap]["selected"] for cap in CAPS}
    with pytest.raises(ValueError, match="density successor run"):
        data.matched_density_fits(primary["root"])
    result = data.matched_density_comparison(density_run["root"], primary["root"])
    assert len(result["rows"]) == 2 * 8 * 26 * 2
    assert len(result["summary"]) == 8 * 2 * 2
    matched_summary = [r for r in result["summary"] if r["variant"] == "matched"]
    assert all(
        r["n_saved"] == 24 and r["n_failed"] == 1 and r["mean_fit_usable"] for r in matched_summary
    )
    assert all(
        r["n_saved"] == 25 and r["mean_fit_half_width"] == pytest.approx(0.01)
        for r in result["summary"]
        if r["variant"] == "primary"
    )
    assert len(result["pairs"]) == 8 * 2 * 24
    assert all(p["width_ratio_matched_over_primary"] == pytest.approx(1.0) for p in result["pairs"])
    assert all(p["matched_minus_primary"] == 0.0 for p in result["pairs"])
    single = result["single_vs_mean"]
    assert len(single) == 8 * 2 * 24
    for row in single:
        context = plan["conditions"][row["condition"]][row["measurement_bin"]]
        assert row["width_ratio_single_matched_over_mean_primary"] == pytest.approx(3.0)
        assert row["veff_mean_primary_mpc_h3"] == pytest.approx(25 * context["veff_primary_mpc_h3"])
        assert row["expected_sqrt_veff_ratio"] == pytest.approx(
            math.sqrt(25 * context["veff_primary_mpc_h3"] / context["veff_attained_mpc_h3"])
        )
    with pytest.raises(ValueError, match="per-bin primary measurement"):
        data.matched_density_comparison(density_run["root"], density_run["root"])
    # Same configuration, another root: refused before its (unpublished) fits are read, naming
    # the identity fields that differ from the handoff's recorded source.
    with pytest.raises(ValueError, match=r"recorded source; differing fields: root '"):
        data.matched_density_comparison(density_run["root"], twin["root"])
    renamed, _ = publish_small_primary(tmp_path, monkeypatch, name="renamed")
    with pytest.raises(ValueError, match=r"differing fields: root '.*', run_name 'renamed'"):
        data.matched_density_comparison(density_run["root"], renamed["root"])
    other_config = load_analysis_config()
    other_config["fitting"]["damping"]["gaussian_widths_mpc_h"]["sigma_perp"] = 1.25
    other, _ = publish_small_primary(tmp_path, monkeypatch, name="other", config=other_config)
    with pytest.raises(ValueError, match="beyond the density definition"):
        data.matched_density_comparison(density_run["root"], other["root"])


def test_matched_readers_pair_the_unified_refits_through_the_primary_measurement(
    produced, tmp_path, monkeypatch
):
    # Revision 2.20: the density and primary fits are unified-prior refits; the primary refit's
    # measurement run is the handoff's recorded source, and the families never mix.
    from pzbao.config import UNIFIED_PRIOR

    primary, _, _, handoff, _ = produced
    monkeypatch.setattr(driver, "_code_identity", lambda **kwargs: dict(CODE))
    derived = density_run_config(
        load_analysis_config(), handoff, handoff["resolved_snapshot_sha256"]
    )
    definition = tmp_path / "density.yaml"
    definition.write_text(yaml.safe_dump(derived, sort_keys=False))
    density_run = driver.create_run(
        tmp_path, "density", workload="density", catalog_root=tmp_path, config_path=definition
    )
    refits = {
        name: driver.create_run(
            tmp_path / "unified",
            name,
            workload=source["metadata"]["workload"],
            catalog_root=source["metadata"]["catalog_root"],
            source_run=source,
            inference_definition=UNIFIED_PRIOR,
        )
        for name, source in (("density", density_run), ("primary", primary))
    }
    routes = set(data.matched_density_routes(density_run["config"]))
    for run in refits.values():
        _publish_fits(run, routes)
    monkeypatch.setattr(
        fitting, "read_profiles", lambda path, *, identity: _profile(0.02, value=1.0)
    )
    result = data.matched_density_comparison(refits["density"]["root"], refits["primary"]["root"])
    assert len(result["summary"]) == 8 * 2 * 2
    assert all(r["n_saved"] == 25 and r["mean_fit_usable"] for r in result["summary"])
    assert {r["source"].split("/unified/")[1].split("/")[0] for r in result["rows"]} == {
        "density",
        "primary",
    }
    with pytest.raises(ValueError, match="different prior families"):
        data.matched_density_comparison(refits["density"]["root"], primary["root"])
    with pytest.raises(ValueError, match="different prior families"):
        data.matched_density_comparison(density_run["root"], refits["primary"]["root"])
