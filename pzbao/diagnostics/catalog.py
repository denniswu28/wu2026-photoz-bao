"""Reusable catalog diagnostics. Inputs and experiment choices are explicit arguments."""

from __future__ import annotations


import numpy as np


def describe(catalogue, *, redshift="Z", identity="SOURCE_ID"):
    """Summarize supplied rows without selecting them or changing any weights."""
    from ..catalog import validate_catalog

    count = validate_catalog(catalogue, required=(redshift, identity))
    z = np.asarray(catalogue[redshift], dtype=float)
    ids = np.asarray(catalogue[identity])
    return {
        "rows": count,
        "columns": list(catalogue),
        "unique_ids": len(np.unique(ids)),
        "nonfinite_redshifts": int(np.count_nonzero(~np.isfinite(z))),
        "redshift_range": [float(np.nanmin(z)), float(np.nanmax(z))] if count else None,
    }


def plot_redshift(catalogue, *, redshift="Z", bins=40, ax=None, label=None):
    """Return the axes and histogram so notebook cells can change the presentation."""
    from ..figures.style import axes

    ax = axes(ax)
    counts, edges = np.histogram(catalogue[redshift], bins=bins)
    ax.stairs(counts, edges, label=label)
    ax.set(xlabel="Redshift", ylabel="Objects")
    return ax, {"counts": counts, "edges": edges}
