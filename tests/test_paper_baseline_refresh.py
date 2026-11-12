"""Evidence-boundary checks for the proposed-paper display adapter."""

from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from pzbao.figures import data as refresh
from pzbao.diagnostics import fitting as notebook_rendering
from tests.notebook_helpers import notebook_namespace


@pytest.mark.parametrize("delta", [-4.0, 0.0, 9.0])
def test_notebook_detection_preserves_signed_difference(delta):
    result = notebook_rendering.detection_statistics(10.0, 10.0 + delta)
    assert result["delta_chi2"] == delta
    assert result["sqrt_delta_chi2_proxy"] == np.sqrt(max(delta, 0))
    with pytest.raises(ValueError, match="finite"):
        notebook_rendering.detection_statistics(10.0, np.nan)


def density_grid():
    from pzbao.figures.style import ZBIN_GRID_PANELS

    rows = []
    for name, (_, lower, upper, _) in zip(refresh.BINS, ZBIN_GRID_PANELS):
        for tier in refresh.TIERS:
            # Include neighboring shells: they must not enter the measurement average.
            edges = [lower - 0.01, lower, (lower + upper) / 2, upper, upper + 0.01]
            counts = [900, 10, 30, 900]
            volumes = [1, 2, 6, 1]
            rows.append(
                dict(
                    bin=name,
                    condition=tier,
                    edges=edges,
                    mean_counts=counts,
                    volume_mpc_h3=volumes,
                    nbar_h3_mpc3=np.asarray(counts) / volumes,
                )
            )
    return rows


def test_density_uses_selected_shell_and_volume_weighting():
    rows = refresh.saved_density_rows(density_grid())
    assert len(rows) == 25
    for row in rows:
        assert row["mean_count"] == 40
        assert row["volume_mpc_h3"] == 8
        assert row["attained_nbar_h3_mpc3"] == 5
        np.testing.assert_array_equal(row["density"], [5, 5])
    bad = density_grid()
    bad[0]["volume_mpc_h3"][1] = 20
    with pytest.raises(AssertionError):
        refresh.saved_density_rows(bad)
    with pytest.raises(ValueError, match="grid"):
        refresh.saved_density_rows(density_grid()[:-1])


def interval_grid():
    rows = []
    for name in refresh.LRG_BINS:
        for tier in refresh.TIERS:
            for state in refresh.STATES:
                for mock in range(25):
                    for parameter in refresh.PARAMETERS:
                        rows.append(
                            dict(
                                measurement_bin=name,
                                condition=tier,
                                state=state,
                                mock=mock,
                                parameter=parameter,
                                minos_bounds=[0.98, 1.02],
                                boundary=False,
                                interval_flags=dict(
                                    is_valid=True, lower_valid=True, upper_valid=True
                                ),
                                optimizer=dict(
                                    maximization=dict(
                                        is_valid=True,
                                        is_above_max_edm=False,
                                        has_reached_call_limit=False,
                                    )
                                ),
                            )
                        )
    return rows


def test_interval_counts_preserve_unavailable_cohorts_and_paired_selection():
    rows = interval_grid()
    rows[0]["boundary"] = True
    counts = refresh.baseline_interval_counts(rows)
    lrg = next(
        r
        for r in counts
        if (r["measurement_bin"], r["condition"], r["parameter"]) == ("LRG1", "native", "qiso")
    )
    assert lrg["available_pairs"] == 25 and lrg["usable_pairs"] == 24
    assert all(r["usable_pairs"] is None for r in counts if r["measurement_bin"].startswith("BGS"))
    with pytest.raises(ValueError, match="Incomplete"):
        refresh.baseline_interval_counts(rows[1:])
    with pytest.raises(ValueError, match="Duplicate"):
        refresh.baseline_interval_counts(rows + rows[:1])


def test_table_retains_joint_correlation_and_flags_without_display_cuts():
    namespace = notebook_namespace(
        "baseline", cells={"setup", "baseline-inputs", "baseline-tables"}
    )
    rows, summary = [], []
    covariance = np.array([[0.01, 0.006], [0.006, 0.01]])
    fom = 1 / np.sqrt(np.linalg.det(covariance))
    for name in refresh.BINS:
        for tier in refresh.TIERS:
            for state in refresh.STATES:
                route = dict(measurement_bin=name, condition=tier, reconstruction_state=state)
                # Native records have covariance but no mean; the reader derives means
                # from retained draws and emits them in its separate posterior summary.
                rows.append(
                    dict(
                        route=route,
                        covariance=covariance.tolist(),
                        fom=fom,
                        profile_flagged=True,
                    )
                )
                for parameter, mean in zip(refresh.PARAMETERS, [0.85, 1.15]):
                    summary.append(dict(route, parameter=parameter, mean=mean))
    render = namespace["posterior_constraints_tex"]
    table = render(list(reversed(rows)), summary)
    assert table.count(r"$0.8500\pm0.1000$") == 50
    assert table.count(r"$1.1500\pm0.1000$") == 50
    assert table.count("125.0") == 50  # Includes correlation; diagonal precision would be 100.
    assert table.count(r"$^{\dagger}$") == 50
    assert table.count(r"\begin{table*}") == 5
    with pytest.raises(ValueError, match="Incomplete or duplicate"):
        render(rows, summary[:-1])
    with pytest.raises(ValueError, match="Incomplete or duplicate"):
        render(rows, summary + summary[:1])


def grid():
    identity = dict(run_name="measurements", code_commit="a" * 40, analysis_config_sha256="b" * 64)
    records = []
    for name in refresh.BINS:
        for tier in refresh.TIERS:
            for state in refresh.STATES:
                route = dict(measurement_bin=name, condition=tier, reconstruction_state=state)
                records.append(
                    dict(
                        route=route,
                        identity=route.copy(),
                        status="available",
                        covariance_identity=identity.copy(),
                    )
                )
    return records, identity


@pytest.mark.parametrize("change", ["duplicate", "missing", "different_measurements"])
def test_incompatible_plot_inputs_fail(change):
    records, identity = grid()
    refresh.validate_grid(records, identity)
    if change == "duplicate":
        records.append(deepcopy(records[0]))
    elif change == "missing":
        records.pop()
    else:
        records[0]["covariance_identity"]["analysis_config_sha256"] = "c" * 64
    with pytest.raises(ValueError):
        refresh.validate_grid(records, identity)


def test_native_draws_burn_once_and_preserve_joint_covariance(tmp_path):
    rng = np.random.default_rng(17)
    arrays, kept, ensembles = [], [], []
    for index in range(4):
        values = rng.normal(size=(2, 6, 3))
        values[:, 3:, 1] += 0.7 * values[:, 3:, 0]
        values[:, :3, :] += 100  # Including burn-in must fail the covariance check.
        arrays.append(values)
        kept.append(values[:, 3:, :2].reshape(-1, 2))
        ensembles.append(
            dict(
                ensemble_index=index,
                path=f"ensemble-{index}.npy",
                steps=6,
                retained_draws=6,
                diagnostics=dict(walkers=2, retained_steps=3),
            )
        )
    expected = np.concatenate(kept)
    covariance = np.cov(expected, rowvar=False, ddof=1)
    record = dict(
        source=str(tmp_path),
        input_sha256={},
        identity={"run_name": "native-test"},
        sampling=dict(
            sampled_parameters=["qiso", "qap", "b1"], ensembles=ensembles, burnin_fraction=0.5
        ),
        covariance=covariance,
        fom=1 / np.sqrt(np.linalg.det(covariance)),
        retained_draws=24,
    )

    def reader(path, *, parameter_names, identity):
        assert parameter_names == ["qiso", "qap", "b1"]
        assert identity["run_name"] == "native-test"
        return dict(
            draws=arrays[identity["ensemble_index"]], axes=["walker", "iteration", "parameter"]
        )

    actual = refresh.retained_draws(record, reader, refresh.Sources())
    np.testing.assert_array_equal(actual, expected)
    ensembles[0]["diagnostics"]["walkers"] = 3
    with pytest.raises(ValueError, match="shape"):
        refresh.retained_draws(record, reader, refresh.Sources())
    ensembles[0]["diagnostics"]["walkers"] = 2
    record["covariance"] = np.diag(np.diag(covariance))
    with pytest.raises(AssertionError):
        refresh.retained_draws(record, reader, refresh.Sources())


def test_covariance_overlay_requires_exact_route_and_common_measurements():
    records, identity = grid()
    overlay = dict(
        identity,
        run_name="bgs-covariance",
        source_run=identity.copy(),
        published_routes=["BGS1--native--prereconstruction"],
    )
    records[0]["covariance_identity"] = {k: overlay[k] for k in refresh.IDENTITY}
    refresh.validate_grid(records, identity, [overlay])
    overlay["source_run"]["code_commit"] = "d" * 40
    with pytest.raises(ValueError, match="different measurements"):
        refresh.validate_grid(records, identity, [overlay])


def test_profile_adapter_keeps_flagged_rows_and_does_not_invent_bgs():
    _, identity = grid()
    style = SimpleNamespace(
        ZBIN_GRID_PANELS=[
            ("BGS", 0.1, 0.2, ""),
            ("BGS", 0.2, 0.4, ""),
            ("LRG", 0.4, 0.6, ""),
            ("LRG", 0.6, 0.8, ""),
            ("LRG", 0.8, 1.0, ""),
        ]
    )
    rows = [
        dict(
            parameter=parameter,
            covariance_identity=identity,
            measurement_bin="LRG1",
            condition="native",
            state="prereconstruction",
            mock=mock,
            value=1.0,
            parabolic_error=0.03,
            optimizer_ok=True,
            boundary=True,
            interval_valid=False,
            identity={"run_name": "primary-power3--LRG1--native--prereconstruction"},
        )
        for mock in range(25)
        for parameter in ("qiso", "qap")
    ]
    profiles = refresh.profile_rows(rows, style, identity)
    assert len(profiles) == 1
    members = next(iter(profiles.values()))
    assert set(members) == set(range(25))
    assert all(row["valid"] and row["qiso_err"] == 0.03 for row in members.values())
    with pytest.raises(ValueError, match="Incomplete"):
        refresh.profile_rows(rows[:-1], style, identity)
    # Minos half-widths survive only for valid intervals without prior-boundary contact.
    for row in rows:
        row.update(
            category="valid" if row["mock"] else "boundary",
            width=0.02,
            chi2=40.0,
            degrees_of_freedom=38,
        )
    members = next(iter(refresh.profile_rows(rows, style, identity).values()))
    assert np.isnan(members[0]["qiso_minos"]) and members[1]["qiso_minos"] == 0.02
    assert members[1]["chi2"] == 40.0 and members[1]["dof"] == 38
    rows[1]["chi2"] = 41.0
    with pytest.raises(ValueError, match="disagree on chi2"):
        refresh.profile_rows(rows, style, identity)


@pytest.mark.parametrize("measurement_bin", ["LRG1", "LRG2"])
def test_detection_uses_registered_profile_identity_and_measurements(measurement_bin):
    _, identity = grid()
    route = dict(
        tracer="LRG",
        measurement_bin=measurement_bin,
        condition="native",
        reconstruction_state="postreconstruction",
    )
    profile = dict(
        identity=dict(identity, run_name="compatible-profile", **route, realization_id=None),
        parameters=[dict(value=1.0)],
    )
    saved = dict(
        profile,
        source_run=dict(
            root="/measurements",
            code_commit=identity["code_commit"],
            analysis_config_sha256=identity["analysis_config_sha256"],
        ),
    )
    record = dict(
        route=route, identity=dict(identity, run_name="posterior", **route), profile=profile
    )
    refresh.validate_detection_fit(saved, record, identity)
    wrong_route = deepcopy(record)
    wrong_route["route"]["measurement_bin"] = "LRG2" if measurement_bin == "LRG1" else "LRG1"
    with pytest.raises(ValueError, match="route differs"):
        refresh.validate_detection_fit(saved, wrong_route, identity)
    saved["source_run"]["analysis_config_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="different measurements"):
        refresh.validate_detection_fit(saved, record, identity)


@pytest.mark.parametrize("lrg1_delta", [-4.0, 826.0])
def test_detection_draws_signed_lrg1_values_in_a_square_panel(lrg1_delta):
    namespace = notebook_namespace(
        "baseline", cells={"setup", "baseline-inputs", "baseline-style", "detection"}
    )
    mean_fits, fits = {}, {}
    for lower, upper, mean, delta in ((0.4, 0.6, 0.99, lrg1_delta), (0.6, 0.8, 1.03, 81.0)):
        for condition in namespace["SIGMA_LABEL_TO_VALUE"]:
            key = ("LRG", lower, upper, condition, 15, "postrecon")
            mean_fits[key] = dict(chi2=10.0, detection=dict(chi2_smooth=10.0 + delta))
            fits[key] = dict(qiso_mean=mean, qiso_std=0.02)
    fig, ax = namespace["plt"].subplots()
    rows = namespace["draw_detection"](ax, mean_fits, fits)
    assert rows is namespace["supporting_tables"]["detection"]
    assert len(rows) == 5
    assert all(row["measurement_bin"] == "LRG1" for row in rows)
    assert all(row["delta_chi2"] == lrg1_delta for row in rows)
    assert all(row["sqrt_delta_chi2_proxy"] == np.sqrt(max(lrg1_delta, 0)) for row in rows)
    np.testing.assert_allclose(
        ax.lines[0].get_ydata(), ((ax.lines[0].get_xdata() - 0.99) / 0.02) ** 2
    )
    np.testing.assert_allclose(ax.lines[1].get_ydata(), lrg1_delta)
    assert not ax.texts
    assert ax.get_ylim()[1] > lrg1_delta
    fig.canvas.draw()
    # The Z proxy axis is retired; the bowl panel is square for the combined figure.
    assert not ax.child_axes
    assert ax.get_box_aspect() == 1
    namespace["plt"].close(fig)


def realization_profiles(namespace):
    """Five LRG1 tiers x two states x 25 mocks with unusable intervals and one failed fit."""
    profiles = {}
    rng = np.random.default_rng(3)
    for label in namespace["SIGMA_LABEL_TO_VALUE"]:
        for state in ("prerecon", "postrecon"):
            members = {}
            for mock in range(25):
                members[mock] = dict(
                    valid=mock != 7,
                    dof=38,
                    chi2=38.0 + mock,
                    qiso_bf=1 + 0.01 * rng.standard_normal(),
                    qap_bf=1 + 0.02 * rng.standard_normal(),
                    qiso_err=0.01,
                    qap_err=0.02,
                    qiso_minos=np.nan if mock in (3, 4) else 0.01 + 0.0001 * mock,
                    qap_minos=np.nan if mock == 5 else 0.02,
                )
            profiles[("LRG", 0.4, 0.6, label, 15, state)] = members
            # A second complete bin, wider by half, exercises the per-bin joint rows.
            profiles[("LRG", 0.6, 0.8, label, 15, state)] = {
                mock: dict(
                    row, qiso_minos=row["qiso_minos"] * 1.5, qap_minos=row["qap_minos"] * 1.5
                )
                for mock, row in members.items()
            }
    return profiles


def test_realizations_figure_summarizes_only_usable_intervals():
    namespace = notebook_namespace(
        "baseline", cells={"setup", "baseline-inputs", "baseline-style", "lrg1-realizations"}
    )
    profiles = realization_profiles(namespace)
    with namespace["plt"].style.context(namespace["PRD_STYLE_FILE"]):
        fig = namespace["lrg1_realizations_figure"](profiles)
    assert len(fig.axes) == 5
    assert [ax.get_yscale() for ax in fig.axes] == ["linear", "linear", "log", "log", "linear"]
    # The shared x axis holds categorical blocks: major ticks only, under the paper style that
    # still gives the continuous y axes their minor ticks.
    assert not fig.axes[-1].xaxis.get_minorticklocs().size
    assert fig.axes[0].yaxis.get_minorticklocs().size
    assert "dof $=38$" in fig.axes[4].get_ylabel()
    summary = namespace["supporting_tables"]["lrg1-realizations"]
    assert len(summary) == 5 * 2 * 5
    members = profiles[("LRG", 0.4, 0.6, "sigma_0.003", 15, "postrecon")]
    row = next(
        r
        for r in summary
        if (r["condition"], r["state"], r["quantity"]) == ("sigma_0.003", "postrecon", "qiso_bf")
    )
    usable = [m["qiso_bf"] for k, m in members.items() if k not in (3, 4, 7)]
    assert row["count"] == 22
    np.testing.assert_allclose(row["mean"], np.mean(usable))
    np.testing.assert_allclose(row["std"], np.std(usable, ddof=1))
    chi2 = next(
        r
        for r in summary
        if (r["condition"], r["state"], r["quantity"]) == ("desi_baseline", "prerecon", "chi2")
    )
    assert chi2["count"] == 24
    namespace["plt"].close(fig)


def test_survey_effective_volume_matches_a_hand_calculation():
    cosmology = SimpleNamespace(comoving_radial_distance=lambda z: 3000.0 * np.asarray(z))
    nbar, p0, f_sky = 2e-4, 1e4, 0.25
    shell = f_sky * 4 * np.pi / 3 * ((3000.0 * 0.6) ** 3 - (3000.0 * 0.4) ** 3)
    expected = shell * (nbar * p0 / (1 + nbar * p0)) ** 2 / 1e9
    value = refresh.survey_effective_volume(
        cosmology, [0.4, 0.6], [nbar], f_sky=f_sky, p0_h3_mpc3=p0
    )
    np.testing.assert_allclose(value, expected)
    edges = np.array([0.4, 0.5, 0.6])
    density = refresh.surface_density_nbar(
        cosmology, edges, arcmin2_density=1.0, dndz=lambda z: float(0.4 <= z <= 0.6)
    )
    per_steradian = (180 * 60 / np.pi) ** 2 * 0.5
    np.testing.assert_allclose(
        density, per_steradian / (np.diff((3000.0 * edges) ** 3) / 3), rtol=1e-6
    )
    with pytest.raises(ValueError, match="geometry"):
        refresh.survey_effective_volume(cosmology, [0.6, 0.4], [nbar], f_sky=f_sky, p0_h3_mpc3=p0)


def synthetic_joint_rows(mean=(1.0, 1.0), sd=(0.01, 0.03)):
    """Joint records in the data-layer shape, with equal weights and uncorrelated tiers."""
    rows = []
    tiers = list(refresh.JOINT_TIERS)
    for name in refresh.BINS:
        for state in refresh.STATES:
            scale = 0.8 if state == "postreconstruction" else 1.0
            covariance = np.diag((np.array(sd) * scale) ** 2)
            weights = {parameter: [1 / len(tiers)] * len(tiers) for parameter in refresh.PARAMETERS}
            rows.append(
                dict(
                    route=dict(measurement_bin=name, condition="joint", reconstruction_state=state),
                    tiers=tiers,
                    status="available",
                    parameters=list(refresh.PARAMETERS),
                    realizations_used=25,
                    realizations_excluded=[],
                    mean=list(mean),
                    covariance=covariance.tolist(),
                    sd=np.sqrt(np.diag(covariance)).tolist(),
                    fom=float(1 / np.sqrt(np.linalg.det(covariance))),
                    weights=weights,
                    removed_tiers={parameter: [] for parameter in refresh.PARAMETERS},
                    realization_combination=dict(weights=weights, tiers=tiers),
                    estimate_correlation=np.eye(2 * len(tiers)).tolist(),
                )
            )
    return rows


def test_literature_forest_uses_lrg1_usable_widths_with_better_to_the_right():
    namespace = notebook_namespace(
        "baseline",
        cells={
            "setup",
            "baseline-inputs",
            "baseline-style",
            "presentation-style",
            "literature-comparison",
        },
    )
    profiles = realization_profiles(namespace)
    namespace["literature_comparison"](profiles, synthetic_joint_rows())
    table = namespace["supporting_tables"]["literature-comparison"]
    row = next(r for r in table["current"] if r["condition"] == "sigma_0.003")
    members = profiles[("LRG", 0.4, 0.6, "sigma_0.003", 15, "postrecon")].values()
    widths = [m["qiso_minos"] for m in members if np.isfinite(m["qiso_minos"])]
    assert row["measurement_bin"] == "LRG1"
    assert row["postrecon"]["usable_intervals"] == 23
    assert row["postrecon"]["median_minos_half_width"] == np.median(widths)
    # The unsmeared control is not a survey row; the joint row combines the four tiers' widths.
    conditions = [r["condition"] for r in table["current"]]
    assert "desi_baseline" not in conditions and conditions[-1] == "joint"
    joint = table["current"][-1]
    medians = np.array(
        [
            next(r for r in table["current"] if r["condition"] == tier)["postrecon"][
                "median_minos_half_width"
            ]
            for tier in refresh.JOINT_TIERS
        ]
    )
    # Uncorrelated tiers combine with inverse-variance weights: the consensus convention.
    np.testing.assert_allclose(
        joint["postrecon"]["median_minos_half_width"], 1 / np.sqrt(np.sum(medians**-2.0))
    )
    np.testing.assert_allclose(sum(joint["postrecon"]["weights"]), 1.0)
    assert joint["postrecon"]["removed_tiers"] == []
    # Every redshift range has a joint row, in redshift order; bins without per-realization
    # profiles are placeholders, and nothing is labelled by tracer name.
    by_bin = table["joint_by_bin"]
    assert [r["measurement_bin"] for r in by_bin] == list(refresh.BINS)
    assert [r["status"] for r in by_bin] == [
        "unavailable",
        "unavailable",
        "available",
        "available",
        "unavailable",
    ]
    second = next(r for r in by_bin if r["measurement_bin"] == "LRG2")
    np.testing.assert_allclose(
        second["postrecon"]["median_minos_half_width"],
        1.5 * joint["postrecon"]["median_minos_half_width"],
    )
    texts = [t.get_text() for t in namespace["figures"]["fig_literature_comparison"].axes[0].texts]
    assert not any("LRG" in t or "BGS" in t for t in texts)
    assert texts.count("pending") == 3
    left, right = namespace["figures"]["fig_literature_comparison"].axes[0].get_xlim()
    assert left > right


def test_literature_forest_half_fills_combinations_that_mix_pipeline_and_forecast(tmp_path):
    namespace = notebook_namespace(
        "baseline",
        cells={
            "setup",
            "baseline-inputs",
            "baseline-style",
            "presentation-style",
            "literature-comparison",
        },
    )
    matched = {"sigma_0.001", "sigma_0.003"}
    rows = [
        dict(
            measurement_bin=name,
            condition=tier,
            reconstruction_state=state,
            prediction=dict(
                kind=(
                    "matched_veff_pipeline"
                    if name == "LRG2" and tier in matched
                    else "conditional_forecast"
                ),
                sigma_qiso=0.02,
            ),
        )
        for name in ("LRG1", "LRG2")
        for tier in refresh.JOINT_TIERS
        for state in refresh.STATES
    ]
    path = tmp_path / "spherex-prediction.json"
    path.write_text(json.dumps(dict(rows=rows)))
    namespace["spherex_forecast_choice"] = str(path)
    namespace["literature_comparison"](realization_profiles(namespace), synthetic_joint_rows())
    ax = namespace["figures"]["fig_literature_comparison"].axes[0]
    # Prediction diamonds are smaller than the DES Y6 reference diamond.
    diamonds = [
        line for line in ax.lines if line.get_marker() == "D" and line.get_markersize() == 4
    ]
    # LRG1: four tiers and the joint row, before and after reconstruction, all forecast (open).
    # LRG2: only its joint row is drawn, mixing two matched tiers with two forecast tiers.
    half = [line for line in diamonds if line.get_fillstyle() == "left"]
    open_ = [line for line in diamonds if line.get_markerfacecolor() == "white"]
    assert len(half) == 2 and len(open_) == 10 and len(diamonds) == 12
    labels = [text.get_text() for text in ax.get_legend().get_texts()]
    assert labels[2:] == [r"SPHEREx, matched $V_{\rm eff}$ and forecast", "SPHEREx, forecast"]
    joint = next(
        r
        for r in namespace["supporting_tables"]["literature-comparison"]["joint_by_bin"]
        if r["measurement_bin"] == "LRG2"
    )
    assert joint["postrecon"]["spherex_prediction"]["kind"] == "mixed"
    namespace["plt"].close("all")


def test_fom_bars_drop_the_unsmeared_control_and_add_the_joint_bar():
    namespace = notebook_namespace(
        "baseline",
        cells={"setup", "baseline-inputs", "baseline-style", "presentation-style", "absolute-fom"},
    )
    pairs = [
        dict(measurement_bin=name, condition=tier, pre=100.0, post=150.0, gain_percent=50.0)
        for name in refresh.BINS
        for tier in refresh.TIERS
    ]
    joint_rows = synthetic_joint_rows()
    for row in joint_rows:  # one bin without per-realization profiles: no joint bar, marked n/a
        if row["route"]["measurement_bin"] == "BGS1":
            row.clear()
            row.update(
                route=dict(measurement_bin="BGS1", condition="joint", reconstruction_state="x"),
                tiers=list(refresh.JOINT_TIERS),
                status="unavailable",
                reason="no per-realization profiles",
            )
    for state, row in zip(refresh.STATES, [r for r in joint_rows if r["status"] != "available"]):
        row["route"]["reconstruction_state"] = state
    namespace["presentation_plot_fom"](pairs, joint_rows)
    panels = [ax for ax in namespace["figures"]["fig_fom_2d"].axes if ax.get_title()]
    assert len(panels) == len(refresh.BINS)
    for ax, name in zip(panels, refresh.BINS):
        assert [t.get_text() for t in ax.get_xticklabels()] == [
            "0.001",
            "0.003",
            "0.01",
            "0.03",
            "joint",
        ]
        columns = 4 if name == "BGS1" else 5
        assert len(ax.patches) == 4 * columns  # filled, hatched and two outlines per column
        assert ax.get_ylim()[0] == 0  # one linear axis per panel, no broken axes
        assert all(p.get_hatch() in (None, "///") for p in ax.patches)
        assert ("n/a" in [t.get_text() for t in ax.texts]) == (name == "BGS1")
        # A dashed line separates the four classes from their joint combination.
        dashed = [line.get_xdata()[0] for line in ax.lines if line.get_linestyle() == "--"]
        assert dashed == [3.5]
    joint = namespace["supporting_tables"]["fom-joint"]
    assert [r["measurement_bin"] for r in joint] == [b for b in refresh.BINS if b != "BGS1"]
    expected = (1 / 0.8) ** 2
    np.testing.assert_allclose([r["post"] / r["pre"] for r in joint], [expected] * 4, rtol=1e-12)


def test_distance_ratio_figure_uses_the_desi_convention_and_joint_records():
    namespace = notebook_namespace(
        "baseline",
        cells={
            "setup",
            "baseline-inputs",
            "baseline-style",
            "presentation-style",
            "distance-ratios",
        },
    )
    iso, ap = 1.02, 0.97
    joint = synthetic_joint_rows(mean=(iso, ap), sd=(0.01, 0.03))
    z_eff_rows = [
        dict(measurement_bin=name, condition=tier, z_eff=0.2 * (i + 1))
        for i, name in enumerate(refresh.BINS)
        for tier in refresh.TIERS
    ]
    namespace["distance_ratio_figure"](joint, z_eff_rows)
    figure = namespace["figures"]["fig_distance_ratios"]
    assert len(figure.axes) == 4
    # One DESI tracer hue per redshift range: filled dark points after reconstruction and open
    # points before it, with no tracer names in the legend.
    from matplotlib.colors import to_rgba
    from pzbao.figures import style

    faces = [
        to_rgba(line.get_markerfacecolor())
        for line in figure.axes[0].lines
        if line.get_marker() == "o"
    ]
    white = to_rgba("white")
    assert faces.count(white) == len(refresh.BINS)
    assert set(faces) - {white} == {style.zbin_pair(index)[1] for index in range(len(refresh.BINS))}
    legend_texts = [t.get_text() for t in figure.legends[0].get_texts()]
    assert not any("LRG" in t or "BGS" in t for t in legend_texts)
    assert {"Pre-reconstruction", "Post-reconstruction"} <= set(legend_texts)
    for ax in figure.axes:
        # Unity at the centre of a symmetric range, and z_eff over 0-1.
        low, high = ax.get_ylim()
        assert low + high == pytest.approx(2.0)
        assert np.isclose(ax.get_yticks(), 1.0).any()
        assert ax.get_xlim() == pytest.approx((0.0, 1.0))
    table = namespace["supporting_tables"]["distance-ratios"]
    assert len(table["rows"]) == len(refresh.BINS) * len(refresh.STATES)
    row = next(
        r
        for r in table["rows"]
        if r["measurement_bin"] == "LRG1" and r["reconstruction_state"] == "postreconstruction"
    )
    np.testing.assert_allclose(row["z_eff"], 0.6)
    np.testing.assert_allclose(row["ratios"]["D_V/r_d"]["value"], iso)
    # desilike fits qap = qpar/qper, so D_M/D_H is its reciprocal.
    np.testing.assert_allclose(row["ratios"]["D_M/D_H"]["value"], 1 / ap)
    np.testing.assert_allclose(row["ratios"]["D_M/r_d"]["value"], iso * ap ** (-1 / 3))
    np.testing.assert_allclose(row["ratios"]["D_H/r_d"]["value"], iso * ap ** (2 / 3))
    np.testing.assert_allclose(row["ratios"]["D_V/r_d"]["error"], 0.01 * 0.8)
    np.testing.assert_allclose(row["ratios"]["D_M/D_H"]["error"], 0.03 * 0.8 / ap**2)
    # Linear propagation of independent (alpha_iso, alpha_AP) errors to alpha_perp.
    expected = np.hypot(ap ** (-1 / 3) * 0.008, iso * ap ** (-4 / 3) / 3 * 0.024)
    np.testing.assert_allclose(row["ratios"]["D_M/r_d"]["error"], expected)
    with pytest.raises(ValueError, match="effective redshifts incomplete"):
        namespace["distance_ratio_figure"](joint, z_eff_rows[:3])


def test_joint_constraint_rows_combine_degraded_tiers_and_exclude_failed_optimizations():
    from pzbao.figures import style

    rng = np.random.default_rng(11)
    widths = {
        "native": 0.004,
        "sigma_0.001": 0.006,
        "sigma_0.003": 0.005,
        "sigma_0.01": 0.02,
        "sigma_0.03": 0.04,
    }
    draws, profiles = {}, {}
    for name in refresh.BINS:
        for state in refresh.STATES:
            shared = rng.standard_normal((25, 2))
            for tier in refresh.TIERS:
                route = dict(measurement_bin=name, condition=tier, reconstruction_state=state)
                key = refresh.historical_key(route, style)
                width = np.array([widths[tier], 3 * widths[tier]])
                draws[key] = 1.0 + rng.standard_normal((2000, 2)) * width
                own = rng.standard_normal((25, 2))
                values = 1.0 + 5 * width * (np.sqrt(0.5) * shared + np.sqrt(0.5) * own)
                profiles[key] = {
                    mock: dict(
                        valid=not (mock == 7 and tier == "sigma_0.01"),
                        qiso_bf=float(values[mock, 0]),
                        qap_bf=float(values[mock, 1]),
                    )
                    for mock in range(25)
                }
    rows = refresh.joint_constraint_rows(draws, profiles, style, seed=3, resamples=20)
    assert len(rows) == len(refresh.BINS) * len(refresh.STATES)
    assert all(row["status"] == "available" for row in rows)
    # Without per-realization profiles for one tier the bin's joint constraint is unavailable.
    partial = {k: v for k, v in profiles.items() if not (k[0] == "LRG" and k[3] == "sigma_0.03")}
    rows_partial = refresh.joint_constraint_rows(draws, partial, style, seed=3, resamples=20)
    assert {r["route"]["measurement_bin"] for r in rows_partial if r["status"] != "available"} == {
        "LRG1",
        "LRG2",
        "LRG3",
    }
    for row in rows:
        assert row["route"]["condition"] == refresh.JOINT_CONDITION
        assert row["tiers"] == list(refresh.JOINT_TIERS) and "native" not in row["tiers"]
        assert row["realizations_used"] == 24 and row["realizations_excluded"] == [7]
        covariance = np.array(row["covariance"])
        np.testing.assert_allclose(row["fom"], 1 / np.sqrt(np.linalg.det(covariance)))
        np.testing.assert_allclose(row["sd"], np.sqrt(np.diag(covariance)))
        route = dict(row["route"], reconstruction_state=row["route"]["reconstruction_state"])
        best = min(
            np.sqrt(
                np.cov(
                    draws[refresh.historical_key(dict(route, condition=tier), style)], rowvar=False
                )[0, 0]
            )
            for tier in refresh.JOINT_TIERS
        )
        assert row["sd"][0] <= best * (1 + 1e-9)
        for parameter in refresh.PARAMETERS:
            weights = np.array(row["weights"][parameter])
            assert np.all(weights >= 0)
            np.testing.assert_allclose(weights.sum(), 1.0)
        assert row["bootstrap"]["seed"] == 3
        assert row["bootstrap"]["resamples_used"] + row["bootstrap"]["resamples_skipped"] == 20
        assert row["realization_combination"]["tiers"] == list(refresh.JOINT_TIERS)
        assert len(row["realization_combination"]["joint_width"]) == 2
        assert set(row["leave_one_out"]) == {
            "scatter",
            "tier_scatter",
            "best_tier",
            "best_tier_scatter",
            "improves",
        }


def test_effective_redshift_rows_cover_every_bin_and_tier_from_the_preparations():
    config = dict(
        tracers={
            "BGS_BRIGHT-21.5": dict(measurement_bins=[dict(name="BGS1"), dict(name="BGS2")]),
            "LRG": dict(measurement_bins=[dict(name="LRG1"), dict(name="LRG2"), dict(name="LRG3")]),
        },
        analysis=dict(workloads=dict(full_grid=dict(sigma_z_tiers=list(refresh.TIERS)))),
    )
    preparations = {
        tracer: dict(
            tracer=tracer,
            effective_redshifts={
                tier: {
                    shell["name"]: dict(z_eff=0.5 + 0.01 * index)
                    for shell in spec["measurement_bins"]
                }
                for index, tier in enumerate(refresh.TIERS)
            },
        )
        for tracer, spec in config["tracers"].items()
    }
    rows = refresh.effective_redshift_rows(preparations, config)
    assert len(rows) == 25
    assert {(r["measurement_bin"], r["condition"]) for r in rows} == set(
        (name, tier) for name in refresh.BINS for tier in refresh.TIERS
    )
    preparations["LRG"]["effective_redshifts"]["native"]["LRG1"]["z_eff"] = -1.0
    with pytest.raises(ValueError, match="Invalid effective redshift"):
        refresh.effective_redshift_rows(preparations, config)


def test_density_table_rows_are_redshift_ranges_with_the_unit_in_the_header():
    namespace = notebook_namespace(
        "baseline",
        cells={
            "setup",
            "baseline-inputs",
            "baseline-style",
            "presentation-style",
            "baseline-tables",
        },
    )
    rows = [
        dict(measurement_bin=name, condition=tier, attained_nbar_h3_mpc3=1e-4 * (index + 1))
        for index, name in enumerate(refresh.BINS)
        for tier in refresh.TIERS
    ]
    table = namespace["density_constraints_tex"](rows)
    assert r"\multicolumn{5}{c}{$\bar n\ [10^{-4}\,h^3\,{\rm Mpc}^{-3}]$}" in table
    assert r"$0.4\leq z<0.6$ & 3.000 & 3.000 & 3.000 & 3.000 & 3.000 \\" in table
    assert "LRG1 &" not in table and r"\sigma_z/(1+z)=0" in table


def test_joint_gaussian_helpers_for_the_posterior_figures():
    namespace = notebook_namespace(
        "baseline", cells={"setup", "baseline-inputs", "baseline-style", "post1d", "post2d"}
    )
    grid = namespace["_QGRID"]
    density = namespace["joint_gaussian_density"](grid, 1.0, 0.0004)
    np.testing.assert_allclose(np.trapz(density, grid), 1.0, rtol=1e-6)
    assert grid[np.argmax(density)] == 1.0
    with pytest.raises(ValueError, match="positive"):
        namespace["joint_gaussian_density"](grid, 1.0, 0.0)
    covariance = np.array([[4e-4, 1e-4], [1e-4, 9e-4]])
    points = namespace["joint_gaussian_ellipse"]([1.0, 1.0], covariance, 2.0)
    offsets = points - np.array([1.0, 1.0])
    mahalanobis = np.einsum("ij,jk,ik->i", offsets, np.linalg.inv(covariance), offsets)
    np.testing.assert_allclose(mahalanobis, 4.0, rtol=1e-9)
    namespace["joint_rows"] = synthetic_joint_rows()
    for row in namespace["joint_rows"]:
        if row["route"]["measurement_bin"] == "BGS1":
            row["status"] = "unavailable"
    assert namespace["joint_record_for"]("BGS1") is None
    assert namespace["joint_record_for"]("LRG1")["route"]["measurement_bin"] == "LRG1"
