"""Common reconstruction geometry survives serialization and contains every family member."""

import json

import numpy as np
import pytest

from pzbao.reconstruction import (
    adopt_common_geometry,
    derive_reconstruction_geometry,
    reconstruction_mask_family,
)


def _family(cap="NGC", smoothing=15.0):
    return reconstruction_mask_family(tracer="LRG", cap=cap, smoothing_radius_mpc_h=smoothing)


def _corners(low=0.0, high=10.0):
    return np.array([[low] * 3, [high] * 3])


def _box(nmesh=14):
    return {
        "family": _family(),
        "nmesh": [nmesh] * 3,
        "boxsize_mpc_h": [float(nmesh)] * 3,
        "boxcenter_mpc_h": [5.0] * 3,
    }


def _adopt(geometry, positions, *, family=None, ceiling=329.0):
    return adopt_common_geometry(
        geometry,
        positions,
        positions,
        family=family or _family(),
        cell_size_mpc_h=1.0,
        mesh_dtype="f8",
        source_visible_real_mesh_count=6,
        single_rank_peak_rss_ceiling_gib=ceiling,
        known_position_weight_shift_array_bytes=112,
    )


def test_two_different_position_envelopes_adopt_one_serialized_parent_mesh(tmp_path):
    path = tmp_path / "geometry.json"
    path.write_text(json.dumps(_box()))
    loaded = json.loads(path.read_text())
    first, second = _corners(), _corners(0.5, 10.0)
    derived = [
        derive_reconstruction_geometry(
            array, array, cell_size_mpc_h=1.0, box_padding=1.4, mesh_dtype="f8"
        )
        for array in (first, second)
    ]
    assert derived[0]["box_center_mpc_h"] == [5.0] * 3
    assert derived[1]["box_center_mpc_h"] == [5.25] * 3
    adopted = [_adopt(loaded, array) for array in (first, second)]
    for value in adopted:
        assert value["adopted_nmesh"] == [14] * 3
        assert value["adopted_boxsize_mpc_h"] == [14.0] * 3
        assert value["adopted_boxcenter_mpc_h"] == [5.0] * 3


@pytest.mark.parametrize(
    "changes",
    [
        {"nmesh": []},
        {"nmesh": [14, 14]},
        {"nmesh": [14, 14, 0]},
        {"nmesh": [14, 14, 14.0]},
        {"nmesh": [14, 14, True]},
        {"nmesh": [14, 14, 13]},
        {"boxsize_mpc_h": [14.0, 14.0, 14.1]},
        {"boxsize_mpc_h": [14.0, 14.0, -14.0]},
        {"boxcenter_mpc_h": [5.0, np.nan, 5.0]},
    ],
)
def test_malformed_shared_geometry_is_refused(changes):
    with pytest.raises(ValueError, match="dimensions or cell size"):
        _adopt({**_box(), **changes}, _corners())


@pytest.mark.parametrize(
    "family",
    [
        _family(cap="SGC"),
        _family(smoothing=10.0),
        reconstruction_mask_family(
            tracer="BGS_BRIGHT-21.5", cap="NGC", smoothing_radius_mpc_h=15.0
        ),
    ],
)
def test_shared_geometry_never_crosses_a_parent_cap_or_smoothing_radius(family):
    with pytest.raises(ValueError, match="another parent or cap"):
        _adopt(_box(), _corners(), family=family)


def test_shared_box_contains_actual_positions_and_has_an_exclusive_upper_face():
    for positions in (
        _corners(-2.5, 10.0),
        _corners(0.0, 12.0),
        _corners(0.0, np.nan),
        np.empty((0, 3)),
    ):
        with pytest.raises(ValueError, match="contain finite"):
            _adopt(_box(), positions)
    _adopt(_box(), _corners(-2.0, np.nextafter(12.0, 0.0)))


def test_allocation_floor_uses_the_adopted_mesh_and_both_diagnostic_boolean_arrays():
    ceiling = 5.0e-4
    small = _adopt(_box(14), _corners(), ceiling=ceiling)
    large = _adopt(_box(28), _corners(), ceiling=ceiling)
    for nmesh, result in ((14, small), (28, large)):
        floor = (nmesh**3 * (6 * 8 + 2) + 112) / 1024**3
        assert result["resource_estimate"]["source_visible_peak_floor_gib"] == floor
    assert small["resource_estimate"]["source_visible_floor_within_single_rank_ceiling"]
    assert not large["resource_estimate"]["source_visible_floor_within_single_rank_ceiling"]


def test_mesh_family_has_no_realization_or_measurement_bin_coordinate():
    assert _family() == {
        "tracer": "LRG",
        "cap": "NGC",
        "smoothing_mpc_h": 15.0,
        "identity": "LRG|cap=NGC|smoothing_mpc_h=15",
    }
    # Separate-bin transfer solves name their bin explicitly; no family names a realization.
    assert (
        reconstruction_mask_family(
            tracer="LRG", cap="NGC", smoothing_radius_mpc_h=15.0, measurement_bin="LRG1"
        )["identity"]
        == "LRG|cap=NGC|smoothing_mpc_h=15|bin=LRG1"
    )
    with pytest.raises(TypeError):
        reconstruction_mask_family(
            tracer="LRG", cap="NGC", smoothing_radius_mpc_h=15.0, realization_id=0
        )
