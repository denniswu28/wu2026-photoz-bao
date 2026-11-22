"""Notebook behavior and coordinate-level contracts, independent of campaign launchers."""

import json

import numpy as np
import pytest

from pzbao.figures import data
from tests.notebook_helpers import ROOT, notebook_namespace


@pytest.mark.parametrize("name", ["baseline", "comparisons", "forecasts"])
def test_blank_notebooks_do_not_run_science_or_write(name, monkeypatch, tmp_path):
    from pzbao import run

    monkeypatch.delenv("BAO_RUN_ROOT", raising=False)
    monkeypatch.setattr(run, "_science", lambda *a, **k: pytest.fail("Notebook started science"))
    before = {p for p in tmp_path.rglob("*")}
    namespace = notebook_namespace(name)
    assert namespace["figures"] == {}
    assert namespace["OUTPUT_DIR"] is None
    assert {p for p in tmp_path.rglob("*")} == before


def correlation_record():
    artists = []
    for axis in range(10):
        for tier in range(5 if axis % 2 == 0 else 4):
            curve = np.array([[40.0, tier + 1.0], [44.0, tier + 2.0]])
            artists.extend(
                [dict(axis=axis, kind="line", xy=curve), dict(axis=axis, kind="line", xy=curve[:1])]
            )
            artists.append(
                dict(axis=axis, kind="errorbar", xy=[[40.0, tier + 0.5], [40.0, tier + 1.5]])
            )
    return {"artists": artists}


def test_saved_pooled_coordinates_and_paired_scatter_survive_decoding():
    panels = data.saved_correlation_panels(correlation_record())
    assert len(panels) == 10
    assert len(panels[0]) == 5 and len(panels[1]) == 4
    for panel in panels:
        for row in panel.values():
            np.testing.assert_array_equal(row["error"], [0.5])
            assert row["curve"].shape == (2, 2)
    bad = correlation_record()
    bad["artists"][2]["xy"][0][0] = 41.0
    with pytest.raises(AssertionError):
        data.saved_correlation_panels(bad)


def overlay_fixture():
    ids = np.arange(25)
    pooled = {
        state: dict(
            ids=ids,
            pooled_separation=np.array([50.0, 54.0]),
            pooled=np.array([monopole, [1.0, 2.0]]),
            mean=np.full((2, 2), 100.0),
        )
        for state, monopole in zip(data.STATES, ([0.02, 0.03], [0.03, 0.05]))
    }
    mean_covariance = np.diag([0.0001, 0.0004, 9.0, 16.0])
    covariance = dict(
        realization_ids=ids,
        fit_edges=np.array([48.0, 52.0, 56.0]),
        multipoles=np.array([0, 2]),
        mean_covariance=mean_covariance,
        covariance=25 * mean_covariance,
    )
    return pooled, covariance


def test_reconstruction_overlay_preserves_pooled_values_and_mean_errors(tmp_path):
    pooled, covariance = overlay_fixture()
    decoded = data.reconstruction_overlay_values(
        pooled, covariance, measurement_bin="LRG1", condition="sigma_0.003"
    )
    data.write_json(tmp_path / "overlay.json", decoded)
    saved = json.loads((tmp_path / "overlay.json").read_text())
    assert saved["measurement_bin"] == "LRG1"
    assert saved["condition"] == "sigma_0.003"
    for state, expected in zip(data.STATES, ([50.0, 87.48], [75.0, 145.8])):
        np.testing.assert_allclose(np.asarray(saved[state]["curve"])[:, 1], expected)
        np.testing.assert_allclose(saved[state]["error"], [25.0, 58.32])
    # The mean covariance is already C/25; no second mean-error scaling is allowed.
    np.testing.assert_allclose(np.asarray(saved["residual"])[:, 1], [1.0, 1.0])


@pytest.mark.parametrize("failure", ["ids", "scaling", "coordinates"])
def test_reconstruction_overlay_rejects_incompatible_inputs(failure):
    pooled, covariance = overlay_fixture()
    if failure == "ids":
        pooled[data.STATES[0]]["ids"] = np.arange(1, 26)
    elif failure == "scaling":
        covariance["mean_covariance"] *= 25
    else:
        pooled[data.STATES[0]]["pooled_separation"][0] = 60.0
    with pytest.raises((ValueError, AssertionError)):
        data.reconstruction_overlay_values(
            pooled, covariance, measurement_bin="LRG1", condition="sigma_0.003"
        )


def test_lrg1_correlation_figure_pairs_states_per_tier_and_adds_square_bowls():
    namespace = notebook_namespace(
        "baseline",
        cells={"setup", "baseline-inputs", "baseline-style", "detection", "correlation"},
    )
    pooled, covariance = overlay_fixture()
    overlays = {
        condition: data.reconstruction_overlay_values(
            pooled, covariance, measurement_bin="LRG1", condition=condition
        )
        for condition in data.TIERS
    }
    mean_fits, fits = {}, {}
    for condition in namespace["SIGMA_LABEL_TO_VALUE"]:
        key = ("LRG", 0.4, 0.6, condition, 15, "postrecon")
        mean_fits[key] = dict(chi2=10.0, detection=dict(chi2_smooth=30.0))
        fits[key] = dict(qiso_mean=1.0, qiso_std=0.02)
    fig = namespace["lrg1_correlation_figure"](overlays, mean_fits, fits)
    try:
        assert len(fig.axes) == 5 * 2 + 1
        mains, bowl = fig.axes[0:10:2], fig.axes[10]
        for main, sigma in zip(mains, namespace["SIGMA_LABEL_TO_VALUE"].values()):
            pre, post = [line for line in main.lines if line.get_linestyle() != "None"]
            assert (pre.get_linestyle(), post.get_linestyle()) == ("--", "-")
            assert pre.get_color() == namespace["sigma_color"](sigma, "prerecon")
            np.testing.assert_allclose(post.get_ydata(), [75.0, 145.8])
        fig.canvas.draw()
        assert bowl.get_box_aspect() == 1 and not bowl.child_axes
        assert len(fig.legends[0].get_texts()) == 4
    finally:
        namespace["plt"].close(fig)


@pytest.mark.parametrize("wrong_release", [False, True])
def test_native_overlay_selects_the_requested_pair(tmp_path, monkeypatch, wrong_release):
    from pzbao import clustering, covariance
    from pzbao.diagnostics import clustering as diagnostics

    pooled, arrays = overlay_fixture()
    identity = dict(zip(data.IDENTITY, ("fixture", "a" * 40, "b" * 64)))
    run = dict(root=tmp_path, metadata=dict(identity))
    if wrong_release:
        run["metadata"]["run_name"] = "other"
    routes = [
        dict(
            tracer="LRG",
            measurement_bin="LRG1",
            condition="sigma_0.003",
            reconstruction_state=state,
        )
        for state in data.STATES
    ]

    def route_reader(root, selectors):
        assert selectors == dict(measurement_bin="LRG1", condition="sigma_0.003")
        return run, routes

    def covariance_reader(selected_run, route):
        assert selected_run is run and route == routes[1]
        return dict(directory=tmp_path, arrays=arrays)

    def measurement_reader(selected_run, route):
        assert selected_run is run and route in routes
        return [
            dict(
                identity=dict(route, cap="NGC+SGC", realization_id=mid),
                directory=tmp_path,
                counts=f"counts-{mid}.npy",
                estimator=(route["reconstruction_state"], mid),
            )
            for mid in range(25)
        ]

    def summarize(estimators, ids, edges):
        np.testing.assert_array_equal(ids, np.arange(25))
        np.testing.assert_array_equal(edges, arrays["fit_edges"])
        assert sorted(estimators) == ids
        return pooled[estimators[0][0]]

    class Sources:
        def read(self, path):
            return b"fixture"

    monkeypatch.setattr(data, "_run_routes", route_reader)
    monkeypatch.setattr(covariance, "read_route_covariance", covariance_reader)
    monkeypatch.setattr(clustering, "read_measurements", measurement_reader)
    monkeypatch.setattr(diagnostics, "summarize_estimators", summarize)
    options = dict(
        measurement_bin="LRG1",
        condition="sigma_0.003",
        expected_identity=identity,
        sources=Sources(),
    )
    if wrong_release:
        with pytest.raises(ValueError, match="another measurement release"):
            data.load_reconstruction_overlay(tmp_path, **options)
    else:
        decoded = data.load_reconstruction_overlay(tmp_path, **options)
        assert decoded["measurement_identity"] == identity
        assert decoded["measurement_bin"] == "LRG1"
        np.testing.assert_allclose(decoded["residual"][:, 1], [1.0, 1.0])


def test_saved_tables_do_not_require_historical_renderer_sources(tmp_path):
    (tmp_path / "provenance.json").write_text(
        json.dumps({"source_bytes": {"/retired/checkout/renderer.py": "old-source"}})
    )
    (tmp_path / "values.json").write_text('{"flagged": true, "delta_chi2": -2.5}')
    values = data.read_tables(tmp_path, ["values.json"])["values.json"]
    assert values == {"flagged": True, "delta_chi2": -2.5}


def test_save_cell_writes_figures_and_allows_plot_updates(tmp_path):
    namespace = notebook_namespace("baseline", cells={"setup"})
    fig, ax = namespace["plt"].subplots()
    (line,) = ax.plot([1.0, 2.0], [3.0, 4.0])
    namespace["show"](fig, "fixture")
    destination = tmp_path / "figures"
    namespace["destination_choice"] = destination
    notebook = json.loads((ROOT / "pzbao/figures/baseline.ipynb").read_text())
    cell = next(c for c in notebook["cells"] if c["id"] == "export")
    source = "".join(cell["source"]).replace("OUTPUT_DIR = None", "OUTPUT_DIR = destination_choice")
    exec(compile(source, "baseline:export", "exec"), namespace)
    assert {p.name for p in destination.iterdir()} == {"fixture.png", "fixture.pdf"}
    assert (destination / "fixture.pdf").stat().st_size > 0
    before = (destination / "fixture.png").read_bytes()
    line.set_ydata([4.0, 3.0])
    exec(compile(source, "baseline:export", "exec"), namespace)
    assert (destination / "fixture.png").read_bytes() != before
    np.testing.assert_array_equal(line.get_ydata(), [4.0, 3.0])


def test_survey_landscape_takes_the_spherex_sky_fraction_from_configuration():
    from pzbao.config import load_analysis_config

    notebook = json.loads((ROOT / "pzbao/figures/baseline.ipynb").read_text())
    cell = next(c for c in notebook["cells"] if c.get("id") == "survey-landscape")
    source = "".join(cell["source"])
    assert 'SPHEREX_FSKY = float(configuration["reporting"]["forecast_target_fsky"])' in source
    assert "0.75" not in source
    full_sky_deg2 = 4 * np.pi * (180 / np.pi) ** 2
    configured = load_analysis_config()["reporting"]["forecast_target_fsky"]
    assert configured == pytest.approx(30000.0 / full_sky_deg2, rel=1e-12)


def test_exported_spherex_volumes_use_the_configured_sky_fraction():
    from pzbao.config import load_analysis_config

    exported = json.loads((ROOT / "paper/figures/source/baseline/survey-veff.json").read_text())
    configured = load_analysis_config()["reporting"]["forecast_target_fsky"]
    assert {row["f_sky"] for row in exported["spherex"]} == {configured}


def test_distance_ratios_follow_the_fitted_ap_convention():
    """desilike: qpar = qiso * qap**(2/3) and qper = qiso * qap**(-1/3)."""
    notebook = json.loads((ROOT / "pzbao/figures/baseline.ipynb").read_text())
    source = "".join(
        next(c for c in notebook["cells"] if c.get("id") == "distance-ratios")["source"]
    )
    namespace = {"np": np}
    exec(source.split("def distance_ratio_figure")[0], namespace)
    convert = namespace["alpha_to_distance_ratios"]
    iso, ap = 1.02, 1.05
    covariance = np.array([[1.0e-4, 2.0e-5], [2.0e-5, 4.0e-4]])
    ratios = convert([iso, ap], covariance)
    qpar, qper = iso * ap ** (2 / 3), iso * ap ** (-1 / 3)
    assert ratios["D_H/r_d"]["value"] == pytest.approx(qpar)
    assert ratios["D_M/r_d"]["value"] == pytest.approx(qper)
    assert ratios["D_M/D_H"]["value"] == pytest.approx(qper / qpar)
    assert ratios["D_V/r_d"]["value"] == pytest.approx((qper**2 * qpar) ** (1 / 3))
    step = 1e-6
    for key, row in ratios.items():
        gradient = [
            (convert(np.add([iso, ap], delta), covariance)[key]["value"] - row["value"]) / step
            for delta in ((step, 0.0), (0.0, step))
        ]
        expected = np.sqrt(np.array(gradient) @ covariance @ np.array(gradient))
        assert row["error"] == pytest.approx(expected, rel=1e-4)


def test_replot_forms_the_saved_spherex_predictions_from_their_forecast_record():
    """Applying the current forecast record returns the saved rows, and the earlier record
    changes only the SPHEREx predictions."""
    from pzbao.figures.replot_baseline import apply_spherex_forecast

    source = ROOT / "paper/figures/source"
    joint = json.loads((source / "baseline/joint-constraint.json").read_text())
    table = json.loads((source / "baseline/literature-comparison.json").read_text())
    current = json.loads((source / "forecast/spherex-prediction-unreconstructed.json").read_text())
    earlier = json.loads((source / "forecast/spherex-prediction.json").read_text())
    states = ("prerecon", "postrecon")

    def rows(rows_table):
        return rows_table["current"] + rows_table["joint_by_bin"]

    def predictions(rows_table):
        found = [row[state]["spherex_prediction"] for row in rows(rows_table) for state in states]
        return [item["sigma"] for item in found], [item["kind"] for item in found]

    def everything_else(rows_table):
        stripped = json.loads(json.dumps(rows_table))
        for row in rows(stripped):
            for state in states:
                del row[state]["spherex_prediction"]
        return stripped

    # Combined rows pass through a linear solve, so the last digit depends on the platform.
    again = apply_spherex_forecast(table, joint, current)
    assert everything_else(again) == everything_else(table)
    assert predictions(again)[1] == predictions(table)[1]
    assert predictions(again)[0] == pytest.approx(predictions(table)[0], rel=1e-12)
    before = apply_spherex_forecast(table, joint, earlier)
    assert everything_else(before) == everything_else(table)
    lrg1 = next(r for r in before["joint_by_bin"] if r["measurement_bin"] == "LRG1")
    # The value the 2026-10-26 render saved from the earlier record.
    assert lrg1["prerecon"]["spherex_prediction"]["sigma"] == pytest.approx(
        0.007722709247844823, rel=1e-12
    )
    # Only the predictions before reconstruction differ between the two records.
    for old, new in zip(rows(before), rows(table)):
        assert new["postrecon"]["spherex_prediction"]["sigma"] == pytest.approx(
            old["postrecon"]["spherex_prediction"]["sigma"], rel=2e-4
        )


def test_baseline_replot_draws_the_saved_rows_without_run_inputs(monkeypatch):
    """The survey, FoM, distance-ratio, paired-gain, literature, photo-z response, radial
    selection n(z) and gain-against-density figures redraw from their committed rows alone."""
    from types import SimpleNamespace

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import to_rgba

    from pzbao import catalog, run
    from pzbao.figures.replot_baseline import REPLOT_FIGURES, draw_from_saved_rows

    monkeypatch.setattr(run, "_science", lambda *a, **k: pytest.fail("Replot started science"))
    # Fig. 4 takes r_d from the configured cosmology; a stand-in keeps the test free of cosmoprimo.
    drag_scale = 100.0
    monkeypatch.setattr(
        catalog, "build_mock_cosmology", lambda config=None: SimpleNamespace(rs_drag=drag_scale)
    )
    namespace = draw_from_saved_rows(ROOT, ROOT / "paper")
    assert set(namespace["figures"]) == set(REPLOT_FIGURES)
    assert namespace["OUTPUT_DIR"] is None
    rows = json.loads((ROOT / "paper/figures/source/baseline/paired-gains.json").read_text())
    quality = namespace["figures"]["fig_recon_quality"]
    # One error bar per available summary: the four classes and the joint in every panel.
    assert sum(len(ax.containers) for ax in quality.axes) == sum(
        r["summary"] is not None for r in rows
    )
    for ax in quality.axes:
        # Points only, with no line joining the classes, and a dashed line before the joint.
        assert not [line for line in ax.lines if line.get_linestyle() == "-"]
        dashed = [line.get_xdata()[0] for line in ax.lines if line.get_linestyle() == "--"]
        assert dashed == pytest.approx([np.sqrt(0.03 * 0.07)])
    # Open and filled markers of one shape have the same outer diameter: sqrt(area) + edge.
    landscape = {
        c.get_label(): c for c in namespace["figures"]["fig_survey_landscape"].axes[0].collections
    }

    def outer(label):
        return np.sqrt(landscape[label].get_sizes()[0]) + landscape[label].get_linewidths()[0]

    assert outer("SPHEREx galaxy samples") == pytest.approx(outer("This work (mock samples)"))
    assert outer(r"DESI DR1, $0.1<z<0.4$") == pytest.approx(outer(r"DESI DR1, $0.4<z<1.1$"))
    legend = [t.get_text() for t in quality.legends[0].get_texts()]
    assert "Joint of the four classes" in legend
    forest = namespace["figures"]["fig_literature_comparison"]
    names = {t.get_text() for t in forest.axes[0].texts}
    # The 0.4 <= z < 0.6 joint row is named like the other bins' joint rows.
    assert "$0.4\\leq z<0.6$, joint" in names
    assert not any("four classes" in n for n in names)
    assert not any("tier" in n or "condition" in n for n in names | set(legend))
    # Fig. 4 marks the BAO wavenumber 2 pi / r_d of the configured cosmology in both panels.
    k_bao = 2 * np.pi / drag_scale
    for ax in namespace["figures"]["fig_los_damping"].axes:
        vertical = [line for line in ax.lines if len(set(line.get_xdata())) == 1]
        assert len(ax.lines) - len(vertical) == 12  # four classes by three wedges
        assert [line.get_xdata()[0] for line in vertical] == pytest.approx([k_bao])
        assert vertical[0].get_linestyle() == "--"
    # Categorical axes carry major ticks only; their continuous partners keep the minor ticks.
    for name in ("fig_survey_landscape", "fig_fom_2d", "fig_recon_quality"):
        for ax in namespace["figures"][name].axes:
            if ax.axison:  # not the FoM legend panel
                assert not ax.xaxis.get_minorticklocs().size
                assert ax.yaxis.get_minorticklocs().size
    # The n(z) key sits in the first panel, boxed on an opaque white face so it does not blend
    # with the shading, inside that panel and centred across it.
    radial = namespace["figures"]["fig_nz"]
    assert [ax.get_legend() is not None for ax in radial.axes] == [True] + [False] * 4
    key = radial.axes[0].get_legend()
    assert key.get_frame_on()
    assert key.get_frame().get_facecolor() == pytest.approx(to_rgba("white"))
    canvas = FigureCanvasAgg(radial)
    canvas.draw()
    box = key.get_window_extent(canvas.get_renderer())
    panel = radial.axes[0].get_window_extent(canvas.get_renderer())
    assert panel.x0 <= box.x0 and box.x1 <= panel.x1 and panel.y0 <= box.y0 and box.y1 <= panel.y1
    assert box.x0 - panel.x0 == pytest.approx(panel.x1 - box.x1, abs=1.0)
    # The gain against density: every main and matched sample once per panel, at the density and
    # effective volume of its saved rows, with the ratio of its saved posterior widths.
    source = ROOT / "paper/figures/source"
    summary = json.loads((source / "baseline/posterior-summary.json").read_text())
    volumes = json.loads((source / "baseline/precision-values.json").read_text())["rows"]
    matched = json.loads((source / "appendix/matched-density-posterior.json").read_text())["rows"]
    density = json.loads((source / "robustness/density.json").read_text())

    def width(rows, name, tier, state):
        (row,) = [
            r
            for r in rows
            if (r["measurement_bin"], r["condition"], r["reconstruction_state"], r["parameter"])
            == (name, tier, state, "qiso")
        ]
        return row

    expected, contact = [], 0
    for name in data.BINS:
        for tier in data.TIERS:
            (row,) = [r for r in volumes if (r["measurement_bin"], r["condition"]) == (name, tier)]
            pre, post = (width(summary, name, tier, state) for state in data.STATES)
            contact += pre["prior_contact"] or post["prior_contact"]
            expected.append(
                (
                    sum(cap["count"] for cap in row["caps"])
                    / row["volume_mpc_h3"]
                    * row["p0_mpc_h3"],
                    row["veff_mpc_h3"] / 1e9,
                    pre["sd"] / post["sd"],
                )
            )
    routes = sorted({(r["measurement_bin"], r["condition"]) for r in matched})
    assert len(routes) == 4
    for name, tier in routes:
        (row,) = [
            r
            for r in density["summary"]
            if (r["measurement_bin"], r["condition"], r["variant"], r["parameter"])
            == (name, tier, "matched", "qiso")
            and r["reconstruction_state"] == data.STATES[0]
        ]
        pre, post = (width(matched, name, tier, state) for state in data.STATES)
        contact += pre["prior_contact"] or post["prior_contact"]
        expected.append(
            (
                row["attained_nbar_h3_mpc3"] * density["matching"]["inputs"]["p0_h3_mpc3"],
                row["veff_attained_mpc_h3"] / 1e9,
                pre["sd"] / post["sd"],
            )
        )
    gain = namespace["figures"]["fig_gain_density"]
    assert [ax.get_xscale() for ax in gain.axes] == ["log", "log"]
    for column, ax in enumerate(gain.axes):
        markers = [line for line in ax.lines if line.get_gid() in ("main", "matched")]
        assert [line.get_gid() for line in markers].count("matched") == 4
        drawn = sorted((line.get_xdata()[0], line.get_ydata()[0]) for line in markers)
        assert drawn == pytest.approx(sorted((row[column], row[2]) for row in expected))
        # Open markers are the samples whose posterior reaches a prior bound in either state.
        assert sum(to_rgba(line.get_mfc()) == to_rgba("white") for line in markers) == contact
        # One arrow per matched sample, and the dotted line of no gain.
        assert sum(text.arrow_patch is not None for text in ax.texts) == 4
        assert [line.get_ydata()[0] for line in ax.lines if line.get_linestyle() == ":"] == [1.0]
    # The threshold of Sec. IV C: among the samples with sigma_0 <= 0.003, every one below
    # n P0 = 0.4 is widened by reconstruction and every one above is narrowed.
    precise = [row for row in expected[:25] if expected.index(row) % 5 < 3] + expected[25:]
    assert all((row[2] > 1) == (row[0] > 0.4) for row in precise)


def _paper_robustness_namespace(captured):
    namespace = notebook_namespace("comparisons", cells={"setup", "presentation-style"})
    namespace.update(
        run_robustness=None,
        scan_summary=None,
        transfer_intervals=[],
        density_appendix=None,
        display=captured.append,
    )
    notebook = json.loads((ROOT / "pzbao/figures/comparisons.ipynb").read_text())
    cell = next(c for c in notebook["cells"] if c["id"] == "paper-robustness")
    exec(compile("".join(cell["source"]), "comparisons:paper-robustness", "exec"), namespace)
    assert namespace["figures"] == {}  # no inputs, no figures
    return namespace


def _bands(axis):
    from matplotlib.collections import PolyCollection

    return sum(isinstance(collection, PolyCollection) for collection in axis.collections)


def _filled(axis):
    return sum(
        1
        for line in axis.lines
        if line.get_marker() == "o" and line.get_markerfacecolor() != "white"
    )


def test_paper_robustness_figures_draw_each_available_row_and_flag_the_rest():
    captured = []
    namespace = _paper_robustness_namespace(captured)
    means = []
    for name in data.BINS:
        for parameter in data.PARAMETERS:
            for variant in ("baseline", "pre", "flat", "pcs2", "reciso", "r25"):
                if (name, variant) == ("LRG3", "r25"):
                    continue  # stays missing in the grid
                value = 1.0 + (0.01 if variant == "flat" else 0.0)
                means.append(
                    dict(
                        measurement_bin=name,
                        parameter=parameter,
                        variant=variant,
                        status="available",
                        value=value,
                        minos_bounds=[value - 0.02, value + 0.02],
                        ordered_bounds=True,
                        half_width=0.02,
                        usable=(name, variant) != ("BGS1", "pre"),
                    )
                )
    namespace["paper_robustness_choices"](data.robustness_grid(means))
    # One forest figure per dilation, a panel per bin with the reference and five variants: LRG3
    # lacks its R25 row and BGS1's pre-reconstruction interval is drawn open.
    for suffix in ("iso", "ap"):
        choices = namespace["figures"][f"fig_robustness_choices_{suffix}"]
        assert len(choices.axes) == 5
        assert [_filled(axis) for axis in choices.axes] == [5, 6, 6, 6, 5]
        flat = [line.get_xdata()[0] for line in choices.axes[1].lines if line.get_marker() == "o"]
        assert flat[0] == pytest.approx(1.0) and flat[2] == pytest.approx(1.01)  # absolute values
        assert _bands(choices.axes[1]) == 1  # the reference interval band
    # A bin whose reference alpha_AP is prior-bounded is dropped from the alpha_AP figure only.
    bounded = [
        (
            dict(r, usable=False)
            if (r["measurement_bin"], r["parameter"], r["variant"]) == ("BGS2", "qap", "baseline")
            else r
        )
        for r in means
    ]
    namespace["paper_robustness_choices"](data.robustness_grid(bounded))
    assert len(namespace["figures"]["fig_robustness_choices_ap"].axes) == 4
    assert len(namespace["figures"]["fig_robustness_choices_iso"].axes) == 5

    summary = [
        dict(
            smoothing_mpc_h=radius,
            condition=tier,
            measurement_bin=name,
            parameter=parameter,
            reconstruction_state="postreconstruction",
            mean_fit_value=None if (radius, name[:3]) == (35.0, "LRG") else 1.0,
            mean_fit_half_width=0.02,
            mean_fit_usable=True,
            scatter=0.05 * (1 + (radius - 15) / 100),
        )
        for radius in (15.0, 25.0, 35.0)
        for tier in ("sigma_0.003", "sigma_0.01", "sigma_0.03")
        for name in data.BINS
        for parameter in data.PARAMETERS
    ]
    namespace["paper_robustness_smoothing"](summary)
    smoothing = namespace["figures"]["fig_robustness_smoothing"]
    # Two dilations by three conditions; each panel groups the radii of every bin.
    assert len(smoothing.axes) == 6
    assert [_filled(axis) for axis in smoothing.axes] == [3 * 2 + 2 * 3] * 6  # R35 has BGS only
    assert all(_bands(axis) == 5 for axis in smoothing.axes)  # one band per bin group
    assert not any(line.get_marker() == "s" for axis in smoothing.axes for line in axis.lines)
    note = captured[-1].data
    assert "R35 LRG1" in note and "R35 LRG3" in note and "R35 BGS1" not in note

    intervals = [
        dict(
            source_kind=kind,
            measurement_bin=name,
            condition=condition,
            parameter=parameter,
            status="missing" if (kind, name) == ("cross", "LRG3") else "complete",
            value=1.0,
            half_width=0.03,
            usable=True,
        )
        for kind in ("pre", "self", "cross")
        for name in data.BINS
        for condition in ("sigma_0.01", "sigma_0.03")
        for parameter in data.PARAMETERS
    ]
    namespace["paper_robustness_transfer"](intervals)
    transfer = namespace["figures"]["fig_robustness_transfer"]
    # alpha_iso only: a panel per bin for each target condition, LRG3 without its cross entry.
    assert [_filled(axis) for axis in transfer.axes] == [3, 3, 3, 3, 2] * 2
    assert all(_bands(axis) == 1 for axis in transfer.axes)  # the self-recon band

    density = dict(
        summary=[
            dict(
                variant=variant,
                measurement_bin=name,
                condition=condition,
                reconstruction_state=state,
                parameter=parameter,
                mean_fit_value=1.0,
                mean_fit_half_width=0.02,
                mean_fit_usable=True,
            )
            for variant in ("primary", "matched")
            for name in ("LRG2", "LRG3")
            for condition in ("sigma_0.001", "sigma_0.003")
            for state in data.STATES
            for parameter in data.PARAMETERS
        ]
    )
    namespace["paper_robustness_density"](density)
    # Four routes, each at both densities before and after reconstruction: circles at the target
    # density and squares at the matched density, so the two differ in shape as well as hue.
    for axis in namespace["figures"]["fig_robustness_density"].axes:
        assert _filled(axis) == 8
        assert sum(line.get_marker() == "s" for line in axis.lines) == 8
    legend = namespace["figures"]["fig_robustness_density"].legends[0]
    assert [handle.get_marker() for handle in legend.legend_handles] == ["o", "o", "s", "s"]
    # Rows are named by redshift range and precision, never by tracer bin.
    rows = [
        t.get_text()
        for t in namespace["figures"]["fig_robustness_density"].axes[0].get_yticklabels()
    ]
    assert len(rows) == 4 and not any("LRG" in t or "BGS" in t for t in rows)
    assert rows[0] == "$0.6\\leq z<0.8$\n$\\sigma_z/(1+z)=0.001$"
    assert set(namespace["figures"]) == {
        "fig_robustness_choices_iso",
        "fig_robustness_choices_ap",
        "fig_robustness_smoothing",
        "fig_robustness_transfer",
        "fig_robustness_density",
    }
    # The rows are categories: major ticks only, while the dilation axes keep their minor ticks.
    for figure in namespace["figures"].values():
        for ax in figure.axes:
            assert not ax.yaxis.get_minorticklocs().size
            assert ax.xaxis.get_minorticklocs().size


def test_transfer_mean_fits_take_both_states_of_the_self_run_and_the_cross_targets(monkeypatch):
    from pzbao import run as driver

    targets = [
        dict(tracer="LRG", measurement_bin="LRG1", condition=c, reconstruction_state=s)
        for c in ("sigma_0.01", "sigma_0.03")
        for s in ("postreconstruction",)
    ]
    monkeypatch.setattr(driver, "load_run", lambda root: {"root": root})
    monkeypatch.setattr(data, "routes_for_run", lambda run: targets)
    requested = []

    def fit_rows(root, *, variant, routes):
        requested.append((root, variant, sorted(routes)))
        return [
            dict(**dict(zip(data._ROUTE_KEYS, route)), realization_id=mock, status="complete")
            for route in routes
            for mock in (None, 0, 1)
        ]

    monkeypatch.setattr(data, "_fit_rows", fit_rows)
    rows = data.transfer_mean_fits("cross", "self")
    assert [(root, variant) for root, variant, _ in requested] == [
        ("self", "pre"),
        ("self", "self"),
        ("cross", "cross"),
    ]
    assert {route[3] for route in requested[0][2]} == {"prereconstruction"}
    assert {route[2] for _, _, routes in requested for route in routes} == {
        "sigma_0.01",
        "sigma_0.03",
    }
    assert len(rows) == 3 * 2 and all(r["realization_id"] is None for r in rows)
    assert [r["source_kind"] for r in rows] == ["pre"] * 2 + ["self"] * 2 + ["cross"] * 2


def test_consensus_width_is_the_combined_error_of_correlated_estimates():
    # Two independent equal estimates halve the variance; full correlation gains nothing.
    assert data.consensus_width([0.02, 0.02], np.eye(2)) == pytest.approx(0.02 / np.sqrt(2))
    assert data.consensus_width([0.02, 0.04], [[1.0, 0.5], [0.5, 1.0]]) == pytest.approx(0.02)


def test_joint_recon_gain_pairs_each_realization_across_the_four_conditions():
    tiers = ["sigma_0.001", "sigma_0.003", "sigma_0.01", "sigma_0.03"]
    profiles = {}
    for tier in tiers:
        for state, width in (("prerecon", 0.04), ("postrecon", 0.02)):
            profiles[("LRG", 0.4, 0.6, tier, 15, state)] = {
                mock: dict(
                    valid=mock != 3 or tier != "sigma_0.01",
                    qiso_err=width,
                    qap_err=2 * width,
                    qiso_bf=1.0,
                )
                for mock in range(25)
            }
    joint_rows = [
        dict(
            route=dict(measurement_bin="LRG1", reconstruction_state=state),
            status="available",
            tiers=tiers,
            estimate_correlation=np.eye(8).tolist(),
        )
        for state in data.STATES
    ]
    one = data.joint_recon_gain_distribution(profiles, joint_rows, "LRG1", "LRG", 0.4, 0.6, 15)
    two = data.joint_recon_gain_distribution(
        profiles, joint_rows, "LRG1", "LRG", 0.4, 0.6, 15, kind="2d"
    )
    # Equal conditions combine to half the width in both states: the gain is the tiers' ratio.
    assert one["median"] == pytest.approx(2.0) and two["median"] == pytest.approx(4.0)
    assert one["n"] == 24  # mock 3 is invalid in one condition and drops out
    assert (
        data.joint_recon_gain_distribution(profiles, joint_rows[:1], "LRG1", "LRG", 0.4, 0.6, 15)
        is None
    )


def test_spherex_prediction_takes_the_matched_pipeline_only_at_the_target_volume():
    from pzbao.figures.render_spherex_forecast import PIPELINE_VEFF_FRACTION, prediction

    row = dict(status="available", target_area=dict(sigma=dict(qiso=0.01)))
    matched = prediction(row, [0.012, 0.010, 0.011], PIPELINE_VEFF_FRACTION, False)
    assert matched["kind"] == "matched_veff_pipeline" and matched["sigma_qiso"] == pytest.approx(
        0.011
    )
    short = prediction(row, [0.012], PIPELINE_VEFF_FRACTION - 0.01, True)
    assert short == dict(
        kind="conditional_forecast", sigma_qiso=0.01, veff_fraction=PIPELINE_VEFF_FRACTION - 0.01
    )
    assert prediction(dict(status="unavailable"), [], None, None) is None
