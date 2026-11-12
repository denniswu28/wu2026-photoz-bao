"""Native comparison identity and available-only presentation contracts."""

import pytest

from pzbao.figures import data as figures
from pzbao.figures import data as refresh


def test_available_rows_keep_flagged_results_and_omit_missing_and_set_aside():
    rows = [
        dict(
            measurement_bin="LRG2",
            parameter="qiso",
            variant=v,
            status="missing" if v == "r25" else "available",
            usable=False,
            boundary=True,
            value=0.98,
        )
        for v in ("pk", "baseline", "flat", "r25")
    ]
    result = refresh.available_rows(rows, "LRG2", "qiso")
    assert [r["variant"] for r in result] == ["baseline", "flat"]
    assert all(r["boundary"] and not r["usable"] and r["value"] == 0.98 for r in result)
    with pytest.raises(ValueError, match="Duplicate"):
        refresh.available_rows(rows + [rows[1]], "LRG2", "qiso")


@pytest.mark.parametrize("r25_bins", [("LRG2", "LRG3"), tuple(figures.BINS)])
def test_plot_ticks_match_available_rows_and_preserve_aliases(tmp_path, r25_bins):
    import matplotlib

    matplotlib.use("Agg")
    from IPython.display import Markdown

    rows = []
    for name in figures.BINS:
        for parameter in figures.PARAMETERS:
            for variant in refresh.R25_VARIANTS:
                available = variant != "r25" or name in r25_bins
                rows.append(
                    dict(
                        measurement_bin=name,
                        parameter=parameter,
                        variant=variant,
                        status="available" if available else "missing",
                        value=1.01,
                        minos_bounds=[0.80, 1.20],
                        ordered_bounds=True,
                        usable=True,
                    )
                )
    from tests.notebook_helpers import notebook_namespace

    namespace = notebook_namespace(
        "comparisons", cells={"setup", "presentation-style", "comparison-inputs", "design"}
    )
    displayed = []
    namespace["display"] = displayed.append
    namespace["r25_plot_design"](rows)
    assert set(namespace["figures"]) == {"design-qiso", "design-qap"}
    captions = [item.data for item in displayed if isinstance(item, Markdown)]
    assert len(captions) == 2
    assert all("R25 is available for " + ", ".join(r25_bins) in text for text in captions)
    for name, fig in namespace["figures"].items():
        limits = (0.96, 1.04) if name == "design-qiso" else (0.85, 1.15)
        for bin_name, ax in zip(figures.BINS, fig.axes):
            assert len(ax.get_yticklabels()) == 4 + (bin_name in r25_bins)
            assert ax.get_xlim() == pytest.approx(limits)
            for marker, edge in zip(("<", ">"), limits):
                marked = [line for line in ax.lines if line.get_marker() == marker]
                assert marked and all(list(line.get_xdata()) == [edge] for line in marked)


def test_r25_tasks_cover_bgs_and_lrg_without_hardcoded_selectors():
    tasks = [
        dict(
            case=dict(name="R25", condition="sigma_0.003"),
            measurement_bin=name,
            reconstruction_state="postreconstruction",
            inference="power3",
        )
        for name in figures.BINS
    ]
    assert refresh.r25_tasks(dict(inference_tasks=tasks)) == list(enumerate(tasks))
    with pytest.raises(ValueError, match="five bins"):
        refresh.r25_tasks(dict(inference_tasks=tasks[1:]))


def test_new_bgs_rows_can_extend_existing_lrg_refresh_without_changing_saved_fits():
    existing = dict(
        measurement_bin="LRG2",
        parameter="qiso",
        variant="r25",
        status="available",
        value=0.9882070001597152,
    )
    missing = dict(measurement_bin="BGS1", parameter="qiso", variant="r25", status="missing")
    added = dict(missing, status="available", value=1.01)
    assert refresh.merge_r25_rows([existing, missing], [existing, added]) == [existing, added]
    with pytest.raises(ValueError, match="replace an existing result"):
        refresh.merge_r25_rows([existing], [dict(existing, value=1.0)])


