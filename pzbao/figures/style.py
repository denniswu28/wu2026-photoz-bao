r"""PRD paper plot-style helpers.

Tiny, dependency-free utilities for the BAO paper figures. The module
provides:

* Figure-size helpers (`one_col`, `two_col`, `wide`) returning ``(w, h)``
  tuples matching PRD column widths.
* Color encoders (`sigma_color`, `zbin_color`, `categorical_pair`) on the
  DESI DR1 tracer hues of Paillas et al. (2024, arXiv:2404.03005v3). Each
  photo-z tier or categorical level has an opaque matplotlib-Paired-style
  light/dark pair for two-state comparisons (see `paired_from_hue`).
* Pre/post-recon style dicts (`PRE_STYLE`, `POST_STYLE`) for the two-state
  axis. Dashed grey for pre-recon and solid near-black for post-recon.
* `prd_savefig` writing both PDF (vector, for REVTeX) and PNG (300 dpi
  preview) to `paper/figures/`, after checking the paper's figure-name
  allow list.
* `panel_label` placing `(a)`-style panel labels.

Usage
-----
::

    from pathlib import Path
    import matplotlib.pyplot as plt

    from pzbao.figures.style import (
        one_col, two_col, wide,
        sigma_color, zbin_color,
        PRE_STYLE, POST_STYLE,
        prd_savefig, panel_label,
    )

    plt.style.use(Path(__file__).parents[2] / "paper" / "prd.mplstyle")
    fig, ax = plt.subplots(figsize=one_col())
    ax.plot(s, xi_pre, **PRE_STYLE, label="pre-recon")
    ax.plot(s, xi_post, **POST_STYLE, label="post-recon")
    ax.set_xlabel("s (Mpc/h)")
    ax.set_ylabel(r"s$^2 \xi_0(s)$ (Mpc/h)$^2$")
    panel_label(ax, "a")
    prd_savefig(fig, "fig_cf")
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
#: Project root, resolved relative to this file. Robust to ``cwd``.
_PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

#: Canonical output directory for paper figures (PDF + PNG pairs).
PAPER_FIGURES_DIR: Path = _PROJECT_ROOT / "paper" / "figures"

#: Path to the rcParams style file.
PRD_STYLE_FILE: Path = _PROJECT_ROOT / "paper" / "prd.mplstyle"

#: Figure names the paper notebooks may write. Notebook execution cannot create any other
#: paper figure, so an exploratory cell never overwrites or adds a paper asset by accident.
REVIEWED_PAPER_FIGURE_NAMES = frozenset(
    {
        "fig_cf",
        "fig_distance_ratios",
        "fig_fom_2d",
        "fig_gain_density",
        "fig_literature_comparison",
        "fig_los_damping",
        "fig_lrg1_realizations",
        "fig_nz",
        "fig_pipeline",
        "fig_post1d",
        "fig_post2d",
        "fig_recon_quality",
        "fig_robustness_choices_ap",
        "fig_robustness_choices_iso",
        "fig_robustness_density",
        "fig_robustness_smoothing",
        "fig_robustness_transfer",
        "fig_sigma_r_precision",
        "fig_survey_landscape",
    }
)


# ---------------------------------------------------------------------------
# Figure sizing — PRD column widths
# ---------------------------------------------------------------------------
def one_col(h: float = 2.5) -> Tuple[float, float]:
    """PRD single-column figure size [(w=3.4 in, h=h)]."""
    return (3.4, h)


def two_col(h: float = 4.0) -> Tuple[float, float]:
    """PRD two-column figure size [(w=7.0 in, h=h)] (full text width)."""
    return (7.0, h)


def wide(h: float = 4.0) -> Tuple[float, float]:
    """Alias for ``two_col`` — explicit "spans both columns" intent."""
    return (7.0, h)


# ---------------------------------------------------------------------------
# Color encoders — Paillas et al. (2024) light/dark pairs
# ---------------------------------------------------------------------------
#: Locked σ_z tier ordering for color encoding. The DESI baseline uses
#: a dedicated grey pair distinct from the four qualitative sample pairs.
#: Baseline mocks are unsmeared, so their imposed uncertainty is exactly
#: σ_z = 0. Log-like σ axes must therefore use a zero-supporting scale
#: such as ``symlog`` rather than moving the baseline to a positive proxy.
SIGMA_TIERS: Tuple[float, ...] = (0.001, 0.003, 0.01, 0.03)
SIGMA_BASELINE: float = 0.0

#: DESI DR1 tracer colors, read from the vector figures 10-12 of Paillas et
#: al. (2024, arXiv:2404.03005v3): DR1 multipoles and BAO detection levels.
DESI_TRACER_COLORS = {
    "steelblue": "#4682b4",  # ELG 1.1-1.6
    "seagreen": "#2e8b57",  # QSO 0.8-2.1
    "yellowgreen": "#9acd32",  # BGS 0.1-0.4
    "orange": "#ffa500",  # LRG 0.4-0.6
    "orangered": "#ff4500",  # LRG 0.6-0.8
    "firebrick": "#b22222",  # LRG 0.8-1.1
    "lightskyblue": "#87cefa",  # ELG 0.8-1.1
}
_D65 = (0.95047, 1.0, 1.08883)


def _hex_to_lab(color: str) -> Tuple[float, float, float]:
    """CIELAB (D65) coordinates of an sRGB hex color."""
    rgb = [
        c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
        for c in mpl.colors.to_rgb(color)
    ]
    xyz = (
        np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
        @ rgb
        / _D65
    )
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    return 116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])


def _lch_to_rgb(lightness: float, chroma: float, hue: float) -> np.ndarray:
    """Unclipped sRGB for a CIELCh color; components outside [0, 1] are out of gamut."""
    fy = (lightness + 16) / 116
    f = np.array([fy + chroma * np.cos(hue) / 500, fy, fy - chroma * np.sin(hue) / 200])
    xyz = np.where(f**3 > 216 / 24389, f**3, (116 * f - 16) / (24389 / 27)) * _D65
    linear = (
        np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]])
        @ xyz
    )
    return np.where(
        linear <= 0.0031308, 12.92 * linear, 1.055 * np.abs(linear) ** (1 / 2.4) - 0.055
    )


def _max_chroma(lightness: float, hue: float) -> float:
    """Largest in-gamut chroma at fixed lightness and hue, on a 0.5 grid."""
    chroma = 0.0
    while chroma < 150 and np.all(np.abs(_lch_to_rgb(lightness, chroma + 0.5, hue) - 0.5) <= 0.5):
        chroma += 0.5
    return chroma


def paired_from_hue(
    color: str,
    dark_lightness: float = 70.0,
    light_lightness: float = 80.0,
    light_chroma_ratio: float = 0.55,
) -> Tuple[str, str]:
    """Derive a matplotlib-Paired-style ``(light, dark)`` pair on a tracer hue.

    Paired couples a vivid dark member (L* ~ 48-67) with a pastel light member
    (L* ~ 73-84) carrying about half its chroma. Keeping the DESI hue, the dark
    member takes min(L*, ``dark_lightness``) at maximum in-gamut chroma and the
    light member takes ``light_lightness`` at ``light_chroma_ratio`` of that
    chroma. The paper constants below were produced by this function and are
    then fixed, so rendering never depends on the gamut search.
    """
    lightness, a, b = _hex_to_lab(color)
    hue = np.arctan2(b, a)
    dark_l = min(lightness, dark_lightness)
    dark_c = _max_chroma(dark_l, hue)
    light_c = min(light_chroma_ratio * dark_c, _max_chroma(light_lightness, hue))
    return tuple(
        mpl.colors.to_hex(np.clip(_lch_to_rgb(l, c, hue), 0, 1))
        for l, c in ((light_lightness, light_c), (dark_l, dark_c))
    )


#: Opaque Paired-style ``(light, dark)`` pairs from ``paired_from_hue``.
DESI_STEELBLUE = ("#a6caf2", "#0584c6")
DESI_SEAGREEN = ("#9bd3ad", "#058d50")
DESI_YELLOWGREEN = ("#b7d07c", "#84bc01")
DESI_ORANGE = ("#f4bc7a", "#ed9900")
DESI_ORANGERED = ("#ffb59b", "#ff4501")
DESI_FIREBRICK = ("#ffb4a7", "#bc0119")
#: DES Y6 marker in the survey landscape: a yellow outside the DESI tracer hues.
SURVEY_YELLOW = "#e6b800"
#: The joint tier combination is not a tier: neutral dark grey (pre-reconstruction is hatched
#: or drawn in the light grey "#c4c4c4").
JOINT_COLOR = "#3a3a3a"

#: Light/dark pairs for the baseline and four paper photo-z samples, ordered
#: blue, green, orange, red. The unsmeared baseline keeps a neutral grey pair.
#: Dark is the default for plots that do not encode reconstruction state;
#: stateful plots use light/open/dashed for pre and dark/filled/solid for
#: post. All members are opaque.
SIGMA_COLOR_PAIRS = {
    SIGMA_BASELINE: ("#c4c4c4", "#3a3a3a"),
    0.001: DESI_STEELBLUE,
    0.003: DESI_SEAGREEN,
    0.01: DESI_ORANGE,
    0.03: DESI_ORANGERED,
}
SIGMA_BASELINE_COLOR: str = SIGMA_COLOR_PAIRS[SIGMA_BASELINE][1]

#: Reusable light/dark pairs for non-ordinal categorical levels. Figures
#: should assign these by a stable category order, then use the light member
#: for the first state and the dark member for the second. The first four levels
#: run blue, green, orange, red. The light sky blue tracer color is omitted: its
#: derived pair has DeltaE < 30.
CATEGORICAL_COLOR_PAIRS: Tuple[Tuple[str, str], ...] = (
    DESI_STEELBLUE,
    DESI_SEAGREEN,
    DESI_ORANGE,
    DESI_ORANGERED,
    DESI_YELLOWGREEN,
    DESI_FIREBRICK,
)


def categorical_pair(level: int) -> Tuple[tuple, tuple]:
    """Return the ``(light, dark)`` colors for one categorical level.

    Parameters
    ----------
    level:
        Zero-based position in a stable categorical ordering. Unlike
        :func:`sigma_pair`, categories are never interpolated or matched by
        numeric distance.
    """
    if not isinstance(level, int) or isinstance(level, bool):
        raise TypeError("categorical level must be an integer")
    if not 0 <= level < len(CATEGORICAL_COLOR_PAIRS):
        raise ValueError(f"categorical level must be in [0, {len(CATEGORICAL_COLOR_PAIRS) - 1}]")
    light, dark = CATEGORICAL_COLOR_PAIRS[level]
    return mpl.colors.to_rgba(light), mpl.colors.to_rgba(dark)


def categorical_color(level: int, state: Optional[str] = None) -> tuple:
    """Return one member of a categorical light/dark pair.

    ``state='first'`` selects the light member and ``state='second'`` selects
    the dark member. Omitting *state* returns the dark member for single-state
    categorical curves.
    """
    if state not in {None, "first", "second"}:
        raise ValueError("state must be None, 'first', or 'second'")
    light, dark = categorical_pair(level)
    return light if state == "first" else dark


def sigma_pair(sigma: float) -> Tuple[tuple, tuple]:
    """Return ``(light, dark)`` RGBA colors for a paper photo-z sample.

    Non-canonical values use the nearest canonical non-baseline tier. This
    keeps exploratory smoothing/density figures deterministic without
    inventing a misleading continuous color scale.
    """
    if sigma in SIGMA_COLOR_PAIRS:
        light, dark = SIGMA_COLOR_PAIRS[sigma]
    elif sigma <= 0:
        raise ValueError("sigma must be non-negative")
    else:
        import numpy as np

        nearest = min(SIGMA_TIERS, key=lambda value: abs(np.log10(sigma) - np.log10(value)))
        light, dark = SIGMA_COLOR_PAIRS[nearest]
    return mpl.colors.to_rgba(light), mpl.colors.to_rgba(dark)


def sigma_color(sigma: float, recon: Optional[str] = None, cmap: Optional[str] = None) -> tuple:
    """Return the opaque color for a photo-z sample and reconstruction state.

    ``recon='prerecon'`` returns the light member and ``recon='postrecon'``
    returns the dark member. Omitting *recon* also returns the dark member,
    preserving the historical call pattern for single-state plots. *cmap* is
    accepted for backward compatibility and intentionally ignored.

    Parameters
    ----------
    sigma:
        σ_z value (in (1+z) units).
    recon:
        ``'prerecon'``, ``'postrecon'``, or ``None``.

    Returns
    -------
    tuple
        RGBA tuple suitable for ``ax.plot(..., color=)``.
    """
    if recon not in {None, "prerecon", "postrecon"}:
        raise ValueError("recon must be None, 'prerecon', or 'postrecon'")
    light, dark = sigma_pair(sigma)
    return light if recon == "prerecon" else dark


#: Solid Paillas et al. tracer colors in z-bin order; BGS1 takes seagreen
#: because Paillas et al. have a single BGS bin.
ZBIN_COLORS: Tuple[str, ...] = tuple(
    DESI_TRACER_COLORS[name]
    for name in ("seagreen", "yellowgreen", "orange", "orangered", "firebrick")
)


#: Paired-style ``(light, dark)`` pairs on the same tracer hues, in z-bin order, for figures
#: that show both reconstruction states of each redshift bin.
ZBIN_COLOR_PAIRS: Tuple[Tuple[str, str], ...] = (
    DESI_SEAGREEN,
    DESI_YELLOWGREEN,
    DESI_ORANGE,
    DESI_ORANGERED,
    DESI_FIREBRICK,
)


def zbin_pair(zbin_idx: int) -> Tuple[tuple, tuple]:
    """Return the ``(light, dark)`` RGBA colors of one redshift bin."""
    if not 0 <= zbin_idx < len(ZBIN_COLOR_PAIRS):
        raise ValueError(f"zbin_idx must be in [0, {len(ZBIN_COLOR_PAIRS) - 1}]")
    light, dark = ZBIN_COLOR_PAIRS[zbin_idx]
    return mpl.colors.to_rgba(light), mpl.colors.to_rgba(dark)


def zbin_color(zbin_idx: int, n_bins: int = 5, cmap: Optional[str] = None) -> tuple:
    """Map a z-bin index to a fixed opaque Paillas et al. tracer color.

    Parameters
    ----------
    zbin_idx:
        Index of the z-bin (e.g. 0 for the lowest, 4 for the highest in
        a 5-bin layout).
    n_bins:
        Total number of z-bins in the layout (default 5 for the
        BGS1+BGS2+LRG1+LRG2+LRG3 paper layout).
    *n_bins* and *cmap* remain accepted for backward compatibility.
    """
    if not 0 <= zbin_idx < len(ZBIN_COLORS):
        raise ValueError(f"zbin_idx must be in [0, {len(ZBIN_COLORS) - 1}]")
    return mpl.colors.to_rgba(ZBIN_COLORS[zbin_idx])


# ---------------------------------------------------------------------------
# Pre/post-recon styles
# ---------------------------------------------------------------------------
#: Pre-reconstruction line style — dashed grey, grayscale-safe.
PRE_STYLE: dict = dict(color="#7f7f7f", linestyle="--")

#: Post-reconstruction line style — solid near-black, distinct from the
#: pre-reconstruction state in both tone and linestyle.
POST_STYLE: dict = dict(color="#1f1f1f", linestyle="-")


# ---------------------------------------------------------------------------
# Correlation-function marker / error-bar style
# ---------------------------------------------------------------------------
#: Error-bar style for ξ_ℓ(s) data points: a small dot marker with a capless
#: vertical bar (``capsize=0`` → no top/bottom horizontal whiskers, just the
#: bar). ``ls="none"`` so the connecting line is drawn separately at full
#: resolution. Pass per-tier ``color``/``ecolor``/``mfc``/``mec`` at the call
#: site (e.g. from :func:`sigma_color`).
XI_ERRORBAR_KW: dict = dict(fmt="o", ms=2.5, elinewidth=0.7, capsize=0, ls="none")

#: Subsample stride for ξ_ℓ(s) markers/error bars — place them on every Nth
#: s-bin to reduce overlap when several σ tiers are overlaid in one panel.
XI_MARKER_STRIDE: int = 2


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def ensure_paper_figure_output_allowed(name: str) -> None:
    """Refuse a figure name outside the paper's figure set."""
    if name not in REVIEWED_PAPER_FIGURE_NAMES:
        raise RuntimeError(
            f"Paper figure output {name!r} is not in the paper's figure set. "
            "Add it to REVIEWED_PAPER_FIGURE_NAMES in pzbao/figures/style.py to write it."
        )


def prd_savefig(fig, name: str, subdir: str = "") -> Tuple[Path, Path]:
    """Save *fig* as a PDF + PNG pair under ``paper/figures/[subdir]/``.

    PDF is vector (for REVTeX), PNG is 300 dpi (for preview). The
    ``bbox_inches="tight"`` and ``pad_inches=0.02`` defaults from the
    mplstyle file are respected.

    Parameters
    ----------
    fig:
        Figure to save.
    name:
        Basename without extension. Convention: ``fig_<topic>[_<qualifier>]``.
    subdir:
        Optional subdirectory under ``paper/figures/``.

    Returns
    -------
    (Path, Path)
        Paths to the PDF and PNG outputs.
    """
    ensure_paper_figure_output_allowed(name)
    out_dir = PAPER_FIGURES_DIR / subdir if subdir else PAPER_FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{name}.pdf"
    png_path = out_dir / f"{name}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    return pdf_path, png_path


# ---------------------------------------------------------------------------
# Multi-tracer z-bin grid layout
# ---------------------------------------------------------------------------
#: Canonical 2x3 panel order for multi-tracer paper figures. Each entry is
#: ``(tracer, zmin, zmax, display_label)``; the sixth slot is reserved for a
#: shared legend axis. Display labels are z-ranges (matching
#: ``zbin_panel_label``) — figures identify bins by redshift, not tracer.
ZBIN_GRID_PANELS: Tuple[Tuple[str, float, float, str], ...] = (
    ("BGS_BRIGHT-21.5", 0.1, 0.2, r"$0.1 \leq z < 0.2$"),
    ("BGS_BRIGHT-21.5", 0.2, 0.4, r"$0.2 \leq z < 0.4$"),
    ("LRG", 0.4, 0.6, r"$0.4 \leq z < 0.6$"),
    ("LRG", 0.6, 0.8, r"$0.6 \leq z < 0.8$"),
    ("LRG", 0.8, 1.0, r"$0.8 \leq z < 1.0$"),
)


def zbin_grid_axes(
    figsize_kind: str = "two_col",
    h: float = 4.5,
    *,
    sharex: bool = True,
    sharey: bool = True,
    wspace: float = 0.0,
    hspace: float = 0.0,
    xlabel: str = None,
    ylabel: str = None,
):
    """Build a 2x3 panel grid for multi-tracer paper figures.

    Returns six axes in canonical order::

        (ax_BGS1, ax_BGS2, ax_LRG1, ax_LRG2, ax_LRG3, ax_legend)

    The sixth axis has its frame and ticks turned off and is reserved for
    a shared legend via ``ax_legend.legend(handles, labels, loc='center')``.

    Panels are condensed with zero (or near-zero) inter-panel spacing and a
    single shared x-/y-axis label (``supxlabel``/``supylabel``). Inner tick
    labels are hidden so only the figure-border panels carry tick text. The
    orphan legend cell at flat index ``[5]`` is handled explicitly: panel
    ``[2]`` (LRG1) keeps its x-tick labels because no data panel sits beneath
    it.

    Parameters
    ----------
    figsize_kind:
        ``"one_col"``, ``"two_col"``, or ``"wide"`` — selects the width via
        the standard PRD column helpers.
    h:
        Figure height in inches. Default 4.5 in keeps each panel readable
        at print size while fitting two-column width.
    sharex, sharey:
        Share the corresponding axis across all panels (default ``True``).
        Pass ``False`` for figures that need per-panel limits (e.g. the
        ``fig_post2d`` zoom or the linear-y ``fig_fom`` bars); the shared
        labels are still applied.
    wspace, hspace:
        Inter-panel spacing as a fraction of the average axis size, passed to
        the constrained-layout engine. Default 0.0 removes the gaps.
    xlabel, ylabel:
        Single shared axis labels drawn via :meth:`Figure.supxlabel` /
        :meth:`Figure.supylabel`. ``None`` draws nothing.

    Returns
    -------
    (Figure, tuple of six Axes)
        ``(fig, (ax_BGS1, ax_BGS2, ax_LRG1, ax_LRG2, ax_LRG3, ax_legend))``.
    """
    width_fn = {"one_col": one_col, "two_col": two_col, "wide": wide}.get(figsize_kind, two_col)
    fig, axes_2d = plt.subplots(
        2, 3, figsize=width_fn(h=h), sharex=sharex, sharey=sharey, layout="constrained"
    )
    # Collapse the *inter-panel* gaps via the constrained-layout engine while
    # it still auto-manages the outer padding (for supxlabel/supylabel). Using
    # subplots_adjust here would fight the shared-label spacing.
    fig.get_layout_engine().set(wspace=wspace, hspace=hspace)

    flat = tuple(axes_2d.ravel())
    legend_ax = flat[5]
    legend_ax.axis("off")

    # Orphan-aware tick visibility. Canonical 2x3 flat index -> (row, col):
    #   [0]=(0,0) BGS1  [1]=(0,1) BGS2  [2]=(0,2) LRG1
    #   [3]=(1,0) LRG2  [4]=(1,1) LRG3  [5]=(1,2) legend (off)
    # x-tick labels: only the bottom-most data panel of each column. col2's
    # bottom is the (empty) legend cell, so LRG1 [2] is the bottom of its
    # column and must keep x-ticks. ``ax.label_outer()`` gets this wrong
    # (it keys off is_last_row()), so set visibility explicitly.
    if sharex:
        for i in (0, 1):
            flat[i].tick_params(axis="x", labelbottom=False)
        for i in (2, 3, 4):
            flat[i].tick_params(axis="x", labelbottom=True)
    if sharey:
        for i in (1, 2, 4):
            flat[i].tick_params(axis="y", labelleft=False)
        for i in (0, 3):
            flat[i].tick_params(axis="y", labelleft=True)

    if xlabel is not None:
        fig.supxlabel(xlabel)
    if ylabel is not None:
        fig.supylabel(ylabel)

    return fig, flat


def zbin_grid_axes_with_residual(
    figsize_kind: str = "two_col",
    h: float = 6.5,
    *,
    height_ratios: Tuple[float, float] = (3.0, 1.0),
    wspace: float = 0.0,
    hspace: float = 0.05,
    xlabel: str = None,
    ylabel_main: str = None,
    ylabel_resid: str = None,
):
    """2x3 z-bin grid where each panel is a (main, residual) vertical pair.

    Mirrors :func:`zbin_grid_axes` (canonical :data:`ZBIN_GRID_PANELS` order,
    legend in the sixth slot), but splits every data cell into a tall main
    axis above a compressed residual strip that shares the separation axis —
    the standard "correlation function + residual" stacked layout.

    All main axes share both ``s`` and the main y-scale; all residual axes
    share both ``s`` and the residual y-scale. Inner tick labels are hidden so
    only the figure-border panels carry tick text: y-tick labels appear on the
    left column only (BGS1, LRG2); residual x-tick labels appear on the
    bottom-most data panel of each column (LRG1, LRG2, LRG3 — orphan-aware,
    since LRG1 sits above the empty legend cell); main panels never show
    x-tick labels because a residual strip always sits directly beneath them.

    Returns
    -------
    (Figure, tuple)
        ``(fig, ((m_BGS1, r_BGS1), (m_BGS2, r_BGS2), (m_LRG1, r_LRG1),
        (m_LRG2, r_LRG2), (m_LRG3, r_LRG3), legend_ax))`` — five
        ``(main, residual)`` axis pairs followed by the legend axis.
    """
    width_fn = {"one_col": one_col, "two_col": two_col, "wide": wide}.get(figsize_kind, two_col)
    fig = plt.figure(figsize=width_fn(h=h), layout="constrained")
    fig.get_layout_engine().set(wspace=wspace, hspace=hspace)

    # One flat 5x3 gridspec (NOT nested subgridspecs): rows are
    # [main, resid, spacer, main, resid], so every panel is a direct cell of a
    # single grid. The layout engine then aligns all columns/rows uniformly —
    # nested subgridspecs let constrained-layout resize each cell for its own
    # decorations, which misaligns the panel edges. The empty spacer row
    # (height ``pair_gap``) opens the inter-pair gap while ``hspace`` keeps the
    # main/residual of a pair tight.
    main_h, resid_h = height_ratios
    pair_gap = 0.6
    gs = fig.add_gridspec(5, 3, height_ratios=[main_h, resid_h, pair_gap, main_h, resid_h])

    # Canonical flat order -> (main_row, resid_row, col):
    #   [0]BGS1=(0,1,0) [1]BGS2=(0,1,1) [2]LRG1=(0,1,2)
    #   [3]LRG2=(3,4,0) [4]LRG3=(3,4,1)  legend=rows3-4,col2.
    layout = [(0, 1, 0), (0, 1, 1), (0, 1, 2), (3, 4, 0), (3, 4, 1)]
    anchor_x = None  # shared separation axis across every panel
    main_y = None  # shared y across main panels
    resid_y = None  # shared y across residual panels
    pairs = []
    for mrow, rrow, col in layout:
        ax_main = fig.add_subplot(gs[mrow, col], sharex=anchor_x, sharey=main_y)
        anchor_x = anchor_x or ax_main
        ax_resid = fig.add_subplot(gs[rrow, col], sharex=anchor_x, sharey=resid_y)
        main_y = main_y or ax_main
        resid_y = resid_y or ax_resid
        ax_main.tick_params(axis="x", labelbottom=False)  # residual sits below
        pairs.append((ax_main, ax_resid))

    legend_ax = fig.add_subplot(gs[3:5, 2])
    legend_ax.axis("off")

    # Residual x-tick labels: bottom-most data panel of each column only.
    # Columns: col0 -> LRG2[3], col1 -> LRG3[4], col2 -> LRG1[2] (above the
    # empty legend cell). Hide for the top-row BGS panels [0], [1].
    for i in (0, 1):
        pairs[i][1].tick_params(axis="x", labelbottom=False)
    for i in (2, 3, 4):
        pairs[i][1].tick_params(axis="x", labelbottom=True)
    # y-tick labels: left column only (BGS1[0], LRG2[3]); hide BGS2[1],
    # LRG1[2], LRG3[4] on both the main and residual rows.
    for i in (1, 2, 4):
        pairs[i][0].tick_params(axis="y", labelleft=False)
        pairs[i][1].tick_params(axis="y", labelleft=False)

    if xlabel is not None:
        fig.supxlabel(xlabel)
    if ylabel_main is not None:
        for i in (0, 3):
            pairs[i][0].set_ylabel(ylabel_main)
    if ylabel_resid is not None:
        for i in (0, 3):
            pairs[i][1].set_ylabel(ylabel_resid)

    return fig, tuple(pairs) + (legend_ax,)


def zbin_column_axes(
    figsize_kind: str = "one_col",
    h: float = 7.0,
    *,
    sharex: bool = True,
    sharey: bool = False,
    hspace: float = 0.0,
    xlabel: str = None,
    ylabel: str = None,
):
    """Build a 5x1 vertical-stack layout for per-z-bin trend figures.

    Returns five axes in canonical z-bin order (no orphan legend cell)::

        (ax_BGS1, ax_BGS2, ax_LRG1, ax_LRG2, ax_LRG3)

    The panels share the x-axis (only the bottom panel shows x-tick labels and
    the shared :meth:`Figure.supxlabel`). The y-axis is per-panel by default
    (``sharey=False``) so z-bins with different magnitude ranges each autoscale
    — appropriate for the linear-y threshold/anisotropy summaries.

    There is no spare cell for a legend; place it at the figure level, e.g.::

        fig.legend(handles, labels, loc="upper center",
                   bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False)

    Parameters
    ----------
    figsize_kind:
        Width selector. Default ``"one_col"`` (3.4 in) — a narrow, tall stack.
    h:
        Figure height in inches. Default 7.0 gives ~1.3 in per panel.
    sharex:
        Share the x-axis across panels (default ``True``); only the bottom
        panel keeps x-tick labels.
    sharey:
        Share the y-axis (default ``False`` — per-panel autoscale).
    hspace:
        Vertical inter-panel spacing (constrained-layout fraction). Default 0.
    xlabel, ylabel:
        Single shared labels via ``supxlabel`` / ``supylabel``.

    Returns
    -------
    (Figure, tuple of five Axes)
        ``(fig, (ax_BGS1, ax_BGS2, ax_LRG1, ax_LRG2, ax_LRG3))``.
    """
    width_fn = {"one_col": one_col, "two_col": two_col, "wide": wide}.get(figsize_kind, one_col)
    fig, axes_1d = plt.subplots(
        5, 1, figsize=width_fn(h=h), sharex=sharex, sharey=sharey, layout="constrained"
    )
    fig.get_layout_engine().set(hspace=hspace)

    flat = tuple(axes_1d.ravel())
    # Shared x: hide x-tick labels on the top four panels; bottom keeps them.
    if sharex:
        for ax in flat[:-1]:
            ax.tick_params(axis="x", labelbottom=False)
        flat[-1].tick_params(axis="x", labelbottom=True)

    if xlabel is not None:
        fig.supxlabel(xlabel)
    if ylabel is not None:
        fig.supylabel(ylabel)

    return fig, flat


def zbin_panel_label(
    ax,
    tracer: str,
    zmin: float,
    zmax: float,
    loc: str = "upper left",
    **text_kwargs,
) -> None:
    """Draw the canonical inside-corner z-bin annotation on *ax*.

    Renders only the z-range (e.g. ``0.6 \\leq z < 0.8``); the tracer name is
    intentionally omitted so panels are identified by redshift, not survey.
    The ``(tracer, zmin, zmax)`` triple is still validated against
    :data:`ZBIN_GRID_PANELS` to catch off-grid panels.

    Parameters
    ----------
    ax:
        Matplotlib axes (one of the five data panels from
        :func:`zbin_grid_axes`).
    tracer, zmin, zmax:
        Identify which panel this is; must match one row of
        :data:`ZBIN_GRID_PANELS` exactly.
    loc:
        Corner position; see :func:`panel_label`.
    """
    for t, z0, z1, _label in ZBIN_GRID_PANELS:
        if t == tracer and abs(z0 - zmin) < 1e-9 and abs(z1 - zmax) < 1e-9:
            break
    else:
        raise ValueError(
            f"({tracer}, {zmin}, {zmax}) is not a canonical z-bin panel. "
            f"Allowed: {[(t, z0, z1) for t, z0, z1, _ in ZBIN_GRID_PANELS]}"
        )
    coords = {
        "upper left": (0.04, 0.96, "left", "top"),
        "upper right": (0.96, 0.96, "right", "top"),
        "lower left": (0.04, 0.04, "left", "bottom"),
        "lower right": (0.96, 0.04, "right", "bottom"),
    }
    if loc not in coords:
        raise ValueError(f"Unknown loc: {loc!r}")
    x, y, ha, va = coords[loc]
    text_kwargs.setdefault("transform", ax.transAxes)
    text_kwargs.setdefault("ha", ha)
    text_kwargs.setdefault("va", va)
    text_kwargs.setdefault("fontsize", "small")
    text_kwargs.setdefault(
        "bbox", dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="0.75", linewidth=0.5)
    )
    ax.text(x, y, f"${zmin:.1f} \\leq z < {zmax:.1f}$", **text_kwargs)


def empty_cap_overlay(
    ax,
    *,
    mode: str = "full",
    sigma_threshold: float = 0.0,
    hatch: str = "///",
    alpha: float = 0.25,
    facecolor: str = "lightgrey",
    text: str = r"$\sigma_z > 0$ empty cap" "\n(mock random floor at $z=0.1$)",
) -> None:
    """Draw a hatched translucent band marking BGS1's σ>0 empty-cap region.

    BGS1 spans z=[0.1, 0.2], so σ>0 measurements are possible. This overlay remains
    available for cells where the BAO posterior rails at the prior bounds
    (typically σ_z ≥ 0.01 in the narrow [0.1, 0.2] shell).

    Parameters
    ----------
    ax:
        Axes to overlay.
    mode:
        ``"full"`` covers the whole panel. ``"sigma_axis"`` covers only
        the σ>``sigma_threshold`` region on the x-axis (use for figures
        whose x-axis is σ_z).
    sigma_threshold:
        Left edge of the band for ``mode="sigma_axis"``. Default 0 means
        the band starts immediately above the baseline.
    hatch, alpha, facecolor:
        ``Patch`` styling.
    text:
        Centered label text. Set to ``""`` to suppress.
    """
    if mode == "full":
        ax.axvspan(*ax.get_xlim(), facecolor=facecolor, alpha=alpha, hatch=hatch, zorder=10)
    elif mode == "sigma_axis":
        xlim = ax.get_xlim()
        ax.axvspan(
            max(sigma_threshold, xlim[0]),
            xlim[1],
            facecolor=facecolor,
            alpha=alpha,
            hatch=hatch,
            zorder=10,
        )
    else:
        raise ValueError(f"empty_cap_overlay mode must be 'full' or 'sigma_axis', got {mode!r}")
    if text:
        ax.text(
            0.5,
            0.5,
            text,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize="x-small",
            color="dimgrey",
            zorder=11,
        )


def panel_label(ax, letter: str, loc: str = "upper left", **text_kwargs) -> None:
    """Place a parenthesized lowercase panel label on *ax*.

    Default position is upper-left interior corner. Text uses the figure's
    default font (no boldface — PRD style).

    Parameters
    ----------
    ax:
        Matplotlib axes.
    letter:
        Single letter (e.g. ``"a"``, ``"b"``). Wrapped in parentheses.
    loc:
        ``"upper left"``, ``"upper right"``, ``"lower left"``,
        ``"lower right"``.
    """
    coords = {
        "upper left": (0.04, 0.96, "left", "top"),
        "upper right": (0.96, 0.96, "right", "top"),
        "lower left": (0.04, 0.04, "left", "bottom"),
        "lower right": (0.96, 0.04, "right", "bottom"),
    }
    if loc not in coords:
        raise ValueError(f"Unknown panel-label loc: {loc!r}")
    x, y, ha, va = coords[loc]
    text_kwargs.setdefault("transform", ax.transAxes)
    text_kwargs.setdefault("ha", ha)
    text_kwargs.setdefault("va", va)
    ax.text(x, y, f"({letter})", **text_kwargs)


def axes(ax=None):
    """Respect caller-owned axes and interactive backends."""
    if ax is not None:
        return ax
    import matplotlib.pyplot as plt

    return plt.subplots()[1]


def save_figure(figure, destination, *, formats=("png", "pdf"), dpi=150):
    """Save an explicitly selected figure without closing the notebook's figure."""
    from pathlib import Path

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for extension in formats:
        path = destination.with_suffix("." + extension)
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        paths.append(path)
    return paths


def axes_contract(fig):
    fig.canvas.draw()
    return {
        "size_inches": fig.get_size_inches().tolist(),
        "dpi": fig.dpi,
        "figure_texts": [
            {
                "text": text.get_text(),
                "position": fig.transFigure.inverted().transform(
                    text.get_transform().transform(text.get_position())
                ),
                "bounds": text.get_window_extent().transformed(fig.transFigure.inverted()).bounds,
                "fontsize": text.get_fontsize(),
                "rotation": text.get_rotation(),
                "alignment": (text.get_ha(), text.get_va()),
            }
            for text in fig.texts
        ],
        "figure_legends": [
            {
                "bounds": legend.get_window_extent().transformed(fig.transFigure.inverted()).bounds,
                "labels": [text.get_text() for text in legend.get_texts()],
                "fontsize": [text.get_fontsize() for text in legend.get_texts()],
            }
            for legend in fig.legends
        ],
        "axes": [
            {
                "position": ax.get_position().bounds,
                "xlim": ax.get_xlim(),
                "ylim": ax.get_ylim(),
                "xticks": ax.get_xticks(),
                "yticks": ax.get_yticks(),
                "xminor": ax.get_xticks(minor=True),
                "yminor": ax.get_yticks(minor=True),
                "xlabel": ax.get_xlabel(),
                "ylabel": ax.get_ylabel(),
                "xscale": ax.get_xscale(),
                "yscale": ax.get_yscale(),
                "label_fontsize": (ax.xaxis.label.get_fontsize(), ax.yaxis.label.get_fontsize()),
            }
            for ax in fig.axes
        ],
    }


def lock_axes(fig, original):
    from matplotlib.ticker import FixedLocator

    if len(fig.axes) != len(original["axes"]):
        raise ValueError("Historical panel geometry changed")
    fig.set_size_inches(original["size_inches"])
    fig.set_dpi(original["dpi"])
    fig.set_layout_engine("none")
    for ax, spec in zip(fig.axes, original["axes"]):
        if (ax.get_xlabel(), ax.get_ylabel(), ax.get_xscale(), ax.get_yscale()) != (
            spec["xlabel"],
            spec["ylabel"],
            spec["xscale"],
            spec["yscale"],
        ):
            raise ValueError("Historical axis label or scale changed")
        ax.set_xlim(spec["xlim"])
        ax.set_ylim(spec["ylim"])
        ax.xaxis.set_major_locator(FixedLocator(spec["xticks"]))
        ax.yaxis.set_major_locator(FixedLocator(spec["yticks"]))
        ax.xaxis.set_minor_locator(FixedLocator(spec["xminor"]))
        ax.yaxis.set_minor_locator(FixedLocator(spec["yminor"]))
        ax.set_position(spec["position"])
    if len(fig.legends) != len(original["figure_legends"]):
        raise ValueError("Historical figure legend count changed")
    for legend, spec in zip(fig.legends, original["figure_legends"]):
        if [text.get_text() for text in legend.get_texts()] != spec["labels"]:
            raise ValueError("Historical figure legend labels changed")
        # A default figure legend follows fig.bbox, which savefig temporarily
        # replaces with its crop. Anchor the measured original position in
        # transFigure instead, so PDF/PNG cropping cannot move it off the page.
        legend.set_loc("lower left")
        legend.borderaxespad = 0
        legend.set_bbox_to_anchor(spec["bounds"][:2], transform=fig.transFigure)
    if len(fig.texts) != len(original["figure_texts"]):
        raise ValueError("Historical shared label count changed")
    for text, spec in zip(fig.texts, original["figure_texts"]):
        if text.get_text() != spec["text"] or (text.get_ha(), text.get_va()) != tuple(
            spec["alignment"]
        ):
            raise ValueError("Historical shared label text or alignment changed")
        # Constrained layout positions shared labels during the original draw.
        # The new figure has not drawn yet when its layout engine is disabled.
        text.set_transform(fig.transFigure)
        text.set_position(spec["position"])
    actual = axes_contract(fig)
    for old, new in zip(original["axes"], actual["axes"]):
        for field in ("position", "xlim", "ylim", "xticks", "yticks", "label_fontsize"):
            np.testing.assert_allclose(new[field], old[field], rtol=0, atol=1e-10)
    for old, new in zip(original["figure_legends"], actual["figure_legends"]):
        np.testing.assert_allclose(new["bounds"], old["bounds"], rtol=0, atol=1e-10)
        np.testing.assert_array_equal(new["fontsize"], old["fontsize"])
    for old, new in zip(original["figure_texts"], actual["figure_texts"]):
        for field in ("position", "bounds", "fontsize", "rotation"):
            np.testing.assert_allclose(new[field], old[field], rtol=0, atol=1e-10)
    return actual


def artist_data(fig):
    """Save plotted coordinates and report clipping, including error bars and bands."""
    rows = []
    for index, ax in enumerate(fig.axes):
        xy_sets = []
        for line in ax.lines:
            if line.get_transform() == ax.transData:
                xy_sets.append(("line", np.column_stack(line.get_data())))
        for collection in ax.collections:
            if hasattr(collection, "get_segments"):
                xy_sets.extend(("errorbar", segment) for segment in collection.get_segments())
            elif collection.get_offset_transform() == ax.transData:
                xy_sets.append(("scatter", collection.get_offsets()))
            elif collection.get_transform() == ax.transData:
                xy_sets.extend(
                    ("band_or_contour", path.vertices) for path in collection.get_paths()
                )
        for patch in ax.patches:
            if hasattr(patch, "get_height"):
                xy_sets.append(
                    ("bar", np.array([[patch.get_x(), patch.get_y() + patch.get_height()]]))
                )
        for kind, xy in xy_sets:
            xy = np.asarray(xy, dtype=float)
            if xy.ndim != 2 or xy.shape[1] != 2 or not len(xy):
                continue
            finite = np.all(np.isfinite(xy), axis=1)
            xlo, xhi = sorted(ax.get_xlim())
            ylo, yhi = sorted(ax.get_ylim())
            clipped = finite & (
                (xy[:, 0] < xlo) | (xy[:, 0] > xhi) | (xy[:, 1] < ylo) | (xy[:, 1] > yhi)
            )
            rows.append(
                {
                    "axis": index,
                    "kind": kind,
                    "xy": xy,
                    "clipped_points": int(clipped.sum()),
                    "finite_points": int(finite.sum()),
                    "outside": xy[clipped],
                }
            )
    return rows
