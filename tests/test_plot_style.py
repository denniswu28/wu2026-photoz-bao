"""Tests for the multi-tracer helpers and shared Paired visual encoding."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402  must precede pyplot import

import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "pzbao"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from figures.style import (  # noqa: E402
    REVIEWED_PAPER_FIGURE_NAMES,
    SIGMA_BASELINE,
    SIGMA_TIERS,
    ZBIN_GRID_PANELS,
    categorical_color,
    categorical_pair,
    empty_cap_overlay,
    ensure_paper_figure_output_allowed,
    prd_savefig,
    sigma_color,
    sigma_pair,
    zbin_color,
    zbin_grid_axes,
    zbin_panel_label,
)


def test_paper_figure_output_gate_is_fail_closed():
    ensure_paper_figure_output_allowed("fig_cf")

    with pytest.raises(RuntimeError, match="not in the paper's figure set"):
        ensure_paper_figure_output_allowed("fig_template_robustness")


def test_prd_savefig_rejects_retired_name_before_writing():
    class NoWriteFigure:
        def savefig(self, *_args, **_kwargs):
            raise AssertionError("savefig must not run for a retired output")

    with pytest.raises(RuntimeError, match="figure set"):
        prd_savefig(NoWriteFigure(), "fig_los_smearing")


def test_reviewed_paper_figure_output_count():
    assert len(REVIEWED_PAPER_FIGURE_NAMES) == 19


def test_reviewed_output_names_match_active_pdf_png_pairs():
    figures_root = Path(__file__).resolve().parent.parent / "paper" / "figures"
    pdf_names = {path.stem for path in figures_root.rglob("*.pdf")}
    png_names = {path.stem for path in figures_root.rglob("*.png")}

    assert pdf_names == png_names == REVIEWED_PAPER_FIGURE_NAMES


def test_zbin_grid_axes_returns_six_axes_in_canonical_order():
    fig, axes = zbin_grid_axes()
    try:
        assert len(axes) == 6
        for i in range(5):
            assert axes[i].axison
        assert axes[5].axison is False
    finally:
        plt.close(fig)


def test_zbin_grid_axes_panel_count_matches_canonical_list():
    assert len(ZBIN_GRID_PANELS) == 5


def test_zbin_grid_axes_canonical_order():
    tracers = [panel[0] for panel in ZBIN_GRID_PANELS]
    assert tracers == ["BGS_BRIGHT-21.5", "BGS_BRIGHT-21.5", "LRG", "LRG", "LRG"]
    zmins = [panel[1] for panel in ZBIN_GRID_PANELS]
    assert zmins == sorted(zmins)


def test_zbin_panel_label_accepts_canonical_z_bins():
    fig, axes = zbin_grid_axes()
    try:
        for ax, (tracer, zmin, zmax, _) in zip(axes[:5], ZBIN_GRID_PANELS):
            zbin_panel_label(ax, tracer, zmin, zmax)
            assert any(text.get_text() for text in ax.texts)
    finally:
        plt.close(fig)


def test_zbin_panel_label_rejects_non_canonical_z_bin():
    fig, axes = zbin_grid_axes()
    try:
        with pytest.raises(ValueError, match="not a canonical z-bin panel"):
            zbin_panel_label(axes[0], "LRG", 0.0, 0.2)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("mode", ["full", "sigma_axis"])
def test_empty_cap_overlay_draws_a_patch(mode):
    fig, axes = zbin_grid_axes()
    try:
        ax = axes[0]
        ax.set_xlim(0, 0.04)
        n_before = len(ax.patches)
        empty_cap_overlay(ax, mode=mode)
        assert len(ax.patches) > n_before
        assert any("empty cap" in text.get_text() for text in ax.texts)
    finally:
        plt.close(fig)


def test_empty_cap_overlay_rejects_unknown_mode():
    fig, axes = zbin_grid_axes()
    try:
        with pytest.raises(ValueError, match="mode must be"):
            empty_cap_overlay(axes[0], mode="bogus")
    finally:
        plt.close(fig)


def test_empty_cap_overlay_text_can_be_suppressed():
    fig, axes = zbin_grid_axes()
    try:
        ax = axes[0]
        ax.set_xlim(0, 0.04)
        empty_cap_overlay(ax, text="")
        assert not any("empty cap" in text.get_text() for text in ax.texts)
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    "sigma,light,dark",
    [
        (SIGMA_BASELINE, "#c4c4c4", "#3a3a3a"),
        (0.001, "#a6caf2", "#0584c6"),
        (0.003, "#9bd3ad", "#058d50"),
        (0.01, "#f4bc7a", "#ed9900"),
        (0.03, "#ffb59b", "#ff4501"),
    ],
)
def test_sigma_pairs_are_locked(sigma, light, dark):
    assert sigma_pair(sigma) == (mcolors.to_rgba(light), mcolors.to_rgba(dark))
    assert sigma_color(sigma, "prerecon") == mcolors.to_rgba(light)
    assert sigma_color(sigma, "postrecon") == mcolors.to_rgba(dark)
    assert sigma_color(sigma) == mcolors.to_rgba(dark)


def test_spectroscopic_baseline_has_zero_imposed_smearing():
    assert SIGMA_BASELINE == 0.0


def test_sigma_pair_rejects_negative_uncertainty():
    with pytest.raises(ValueError, match="non-negative"):
        sigma_pair(-0.001)


def test_sigma_color_rejects_unknown_state():
    with pytest.raises(ValueError, match="recon"):
        sigma_color(0.003, "invalid")


@pytest.mark.parametrize(
    "level,light,dark",
    [
        (0, "#a6caf2", "#0584c6"),
        (1, "#9bd3ad", "#058d50"),
        (2, "#f4bc7a", "#ed9900"),
    ],
)
def test_categorical_pairs_are_locked(level, light, dark):
    assert categorical_pair(level) == (mcolors.to_rgba(light), mcolors.to_rgba(dark))
    assert categorical_color(level, "first") == mcolors.to_rgba(light)
    assert categorical_color(level, "second") == mcolors.to_rgba(dark)
    assert categorical_color(level) == mcolors.to_rgba(dark)


def test_paper_palette_is_opaque():
    colors = [
        sigma_color(sigma, state)
        for sigma in (SIGMA_BASELINE, *SIGMA_TIERS)
        for state in (None, "prerecon", "postrecon")
    ]
    colors += [
        categorical_color(level, state) for level in range(6) for state in (None, "first", "second")
    ]
    colors += [zbin_color(index) for index in range(5)]
    assert {color[3] for color in colors} == {1.0}


@pytest.mark.parametrize("level", [-1, 6])
def test_categorical_pair_rejects_out_of_range_level(level):
    with pytest.raises(ValueError, match="categorical level"):
        categorical_pair(level)


def test_categorical_pair_rejects_non_integer_level():
    with pytest.raises(TypeError, match="integer"):
        categorical_pair(0.5)


def test_categorical_color_rejects_unknown_state():
    with pytest.raises(ValueError, match="state"):
        categorical_color(0, "invalid")


def test_zbin_palette_is_qualitative_and_unique():
    colors = [zbin_color(index) for index in range(5)]
    assert len(set(colors)) == 5


@pytest.mark.parametrize(
    "name", ["steelblue", "seagreen", "yellowgreen", "orange", "orangered", "firebrick"]
)
def test_paired_constants_follow_desi_hue_derivation(name):
    from figures import style

    pair = getattr(style, "DESI_" + name.upper())
    assert style.paired_from_hue(style.DESI_TRACER_COLORS[name]) == pair
    contrast = np.linalg.norm(np.subtract(*(style._hex_to_lab(c) for c in pair)))
    assert contrast >= 30
