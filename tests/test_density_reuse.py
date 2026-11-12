"""Original sample identities and saved-selection checks across reuse boundaries."""

import pzbao.diagnostics.selection as diag_selection
import copy
import numpy as np
import pytest


def selection(ids):
    ids = np.asarray(ids)
    return {
        "source_ids": ids,
        "redshift": np.full(len(ids), 0.5),
        "rsd_redshift": 0.49 + ids * 0.0001,
        "noise": ids * 0.01,
        "ranks": ids * 0.02,
        "exterior": np.zeros(len(ids), dtype=bool),
    }


def test_saved_pairing_retains_noise_ranks_and_nested_counts():
    saved = selection([1, 2, 3])
    result = diag_selection.check_pairing(
        saved, copy.deepcopy(saved), selection([1, 2]), [{"name": "LRG1", "zmin": 0.4, "zmax": 0.6}]
    )
    assert result["common_objects"] == 2
    assert result["nested_bin_counts"] == {"LRG1": {"baseline": 2, "density": 3, "common": 2}}


@pytest.mark.parametrize("failure", ["noise", "rank", "duplicate", "nonnested", "redshift"])
def test_pairing_rejects_changes(failure):
    saved, selected, baseline = (selection([1, 2, 3]), selection([1, 2, 3]), selection([1, 2]))
    if failure == "noise":
        baseline["noise"][0] += 0.1
    elif failure == "rank":
        selected["ranks"][0] += 0.1
    elif failure == "duplicate":
        saved["source_ids"][1] = 1
    elif failure == "nonnested":
        baseline = selection([1, 4])
    else:
        saved["redshift"][0] = np.nan
    with pytest.raises((ValueError, AssertionError)):
        diag_selection.check_pairing(
            saved, selected, baseline, [{"name": "LRG1", "zmin": 0.4, "zmax": 0.6}]
        )
