"""Parent reconstruction, numerical geometry, weighting and native estimator semantics."""

import copy
import json
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

import pzbao.reconstruction as pipeline_reconstruction
from tests.legacy_config import load_analysis_config
from pzbao.catalog import build_mock_cosmology, measurement_bins
from pzbao.clustering import combine_cap_power, compute_power_spectrum_secondary, compute_xi
from pzbao.reconstruction import (
    derive_reconstruction_geometry,
    estimate_linear_bias,
    exact_growth_rate,
    weighted_random_zeff_accumulator,
    radial_shell_partition,
    reconstruction_mask_family,
    run_reconstruction,
)
from pzbao.catalog import measurement_weight, reconstruction_weight
from pzbao.selection import stable_measurement_subset

MASK_FAMILY = reconstruction_mask_family(tracer="LRG", cap="NGC", smoothing_radius_mpc_h=15.0)


def _field():
    return {
        "RA": np.array([1.0, 2.0, 3.0]),
        "DEC": np.array([4.0, 5.0, 6.0]),
        "Z": np.array([0.11, 0.21, 0.31]),
        "X": np.array([10.0, 20.0, 30.0]),
        "Y": np.array([11.0, 21.0, 31.0]),
        "Z_CART": np.array([12.0, 22.0, 32.0]),
        "WEIGHT": np.ones(3),
        "WEIGHT_FKP": np.ones(3),
        "SOURCE_ID": np.array([1, 2, 3], dtype=np.uint64),
    }


def _adopted_geometry(nmesh=(8, 8, 8), boxsize=(32.0, 32.0, 32.0), boxcenter=(20.0, 21.0, 22.0)):
    """Stand in for the frozen family geometry that a real box artifact carries."""
    return {
        "mask_family": dict(MASK_FAMILY),
        "adopted_nmesh": [int(value) for value in nmesh],
        "adopted_boxsize_mpc_h": [float(value) for value in boxsize],
        "adopted_boxcenter_mpc_h": [float(value) for value in boxcenter],
        "adopted_cell_size_mpc_h": 4.0,
        "adopted_mesh_cell_count": int(np.prod(nmesh)),
        "minimum_per_side_clearance_mpc_h": 6.0,
        "resource_estimate": {"source_visible_floor_within_single_rank_ceiling": True},
    }


def _patch_adoption(monkeypatch, **geometry):
    """Skip artifact I/O where the test is about something other than box lineage."""
    adopted = _adopted_geometry(**geometry)
    monkeypatch.setattr(
        pipeline_reconstruction,
        "adopt_common_geometry",
        lambda *args, **kwargs: adopted,
    )
    return adopted


# One radial test field and the mesh it is solved on. Objects lie along the x axis with a
# small transverse spread, so each object's comoving radius is essentially its x coordinate
# and the shell a cell or an object belongs to can be read off by hand. The transverse spread
# is not decoration: the geometry preflight refuses a catalogue with zero extent on any axis.
_SHELL_NMESH = (60, 6, 6)
_SHELL_BOXSIZE = (240.0, 24.0, 24.0)
_SHELL_BOXCENTER = (200.0, 0.0, 0.0)


def _radial_field(count=21, first=100.0, step=10.0):
    index = np.arange(count)
    radii = first + step * index
    return {
        "RA": np.zeros(count),
        "DEC": np.zeros(count),
        "Z": np.full(count, 0.5),
        "X": radii.astype(float),
        "Y": np.where(index % 2 == 0, 2.0, -2.0),
        "Z_CART": np.where(index % 3 == 0, 2.0, -2.0),
        "WEIGHT": np.ones(count),
        "WEIGHT_FKP": np.ones(count),
        "SOURCE_ID": index.astype(np.uint64),
    }


def _shell_reconstruction(monkeypatch, *, shift=1.0, zeroed_cells=None):
    """Install a painting PyRecon stub on the radial mesh and return the shift it applies."""

    class Reconstruction(_StagedReconstruction):
        def __init__(self, **kwargs):
            self.calls = []
            self.mpicomm = SimpleNamespace(size=1, rank=0)
            self.nmesh = np.asarray(kwargs["nmesh"], dtype=int)
            self.boxsize = np.asarray(kwargs["boxsize"], dtype=float)
            self.boxcenter = np.asarray(kwargs["boxcenter"], dtype=float)
            self.cellsize = self.boxsize / self.nmesh
            self.zeroed_cells = zeroed_cells

        def read_shifted_positions(self, positions, field):
            moved = np.asarray(positions, dtype=float).copy()
            if callable(shift):
                return shift(moved)
            return moved + shift

    monkeypatch.setitem(
        sys.modules,
        "pyrecon",
        types.SimpleNamespace(IterativeFFTReconstruction=Reconstruction),
    )
    return _patch_adoption(
        monkeypatch,
        nmesh=_SHELL_NMESH,
        boxsize=_SHELL_BOXSIZE,
        boxcenter=_SHELL_BOXCENTER,
    )


def _shell_cells_between(low, high):
    """Boolean mesh selecting the cells whose comoving radius lies in ``[low, high)``."""
    offset = np.asarray(_SHELL_BOXCENTER) - 0.5 * np.asarray(_SHELL_BOXSIZE)
    axes = [offset[axis] + 4.0 * np.arange(_SHELL_NMESH[axis]) for axis in range(3)]
    radii = np.sqrt(
        axes[0][:, None, None] ** 2 + axes[1][None, :, None] ** 2 + axes[2][None, None, :] ** 2
    )
    return (radii >= low) & (radii < high)


def _reconstruction_controls():
    config = load_analysis_config()
    return {
        **pipeline_reconstruction.configured_reconstruction_settings(config),
        "mask_family": MASK_FAMILY,
        "common_geometry": _frozen_family_box(nmesh=(8, 8, 8)),
        "identity_column": "SOURCE_ID",
        "weight_expression": "WEIGHT",
        "pyrecon_kw_weights": None,
        "config": config,
    }


def _power_geometry(config):
    return {
        "cell_size_mpc_h": 4.0,
        "nmesh": [8, 8, 8],
        "boxsize_mpc_h": [32.0] * 3,
        "boxcenter_mpc_h": [20.0, 21.0, 22.0],
        "native_k_edges_h_mpc": [0.0, 0.1, 0.2, 0.3, 0.4],
    }


def _set_mock_geometry(reconstruction, *, nmesh=(8, 8, 8), boxsize=(32.0, 32.0, 32.0)):
    reconstruction.nmesh = np.asarray(nmesh, dtype=int)
    reconstruction.boxsize = np.asarray(boxsize, dtype=float)
    reconstruction.cellsize = reconstruction.boxsize / reconstruction.nmesh
    reconstruction.boxcenter = np.asarray([20.0, 21.0, 22.0])


class _StagedReconstruction:
    """Minimal pinned-PyRecon staged API used by synthetic boundary tests.

    The random mesh is painted rather than filled, because the per-shell diagnostics measure
    the field over the cells that actually hold random weight. A uniformly occupied mesh would
    put supported cells at radii no catalogue object can explain and would exercise the
    diagnostics on a field no reconstruction can produce.
    """

    #: Set by a test to zero part of the density field the way the threshold would.
    zeroed_cells = None

    def _cell(self, position):
        offset = self.boxcenter - 0.5 * self.boxsize
        index = np.rint((np.asarray(position, dtype=float) - offset) / self.cellsize)
        return tuple(int(np.clip(value, 0, size - 1)) for value, size in zip(index, self.nmesh))

    def assign_data(self, positions, weights, position_type):
        self.calls.append(("assign_data", np.asarray(weights).copy(), position_type))

    def assign_randoms(self, positions, weights, position_type):
        self.calls.append(("assign_randoms", np.asarray(weights).copy(), position_type))
        self.mesh_randoms = np.zeros(tuple(self.nmesh), dtype=float)
        for position, weight in zip(np.asarray(positions, dtype=float), np.asarray(weights)):
            self.mesh_randoms[self._cell(position)] += float(weight)

    def set_density_contrast(self, **kwargs):
        self.calls.append(("set_density_contrast", kwargs))
        self.mesh_delta = np.ones(tuple(self.nmesh), dtype=float)
        if self.zeroed_cells is not None:
            self.mesh_delta[self.zeroed_cells] = 0.0
        # The pinned library deletes its random mesh here, so anything the pipeline needs from
        # the painted randoms has to have been taken before this call. The stub deletes it too,
        # which is what makes every test below a check on that ordering.
        del self.mesh_randoms

    def run(self, **kwargs):
        self.calls.append(("run", kwargs))


def test_reconstruction_geometry_matches_even_quantization_and_reports_memory_floor():
    positions = np.asarray([[0.0, 0.0, 0.0], [256.0, 256.0, 256.0]])
    geometry = derive_reconstruction_geometry(
        positions,
        positions,
        cell_size_mpc_h=1.0,
        box_padding=2.0,
        mesh_dtype="f8",
    )
    assert geometry["derived_even_nmesh"] == [512, 512, 512]
    assert geometry["mesh_cell_count"] == 512**3
    assert geometry["resource_estimate"]["gib_per_real_mesh"] == pytest.approx(1.0)
    assert geometry["resource_estimate"]["source_visible_peak_floor_gib"] > 6.0
    assert geometry["resource_estimate"]["source_visible_peak_mesh_equivalents"] == 6
    assert geometry["resource_estimate"]["actual_peak_rss_requires_canary"] is True


def test_exact_growth_uses_supplied_cosmology_at_frozen_redshift():
    class Cosmology:
        def growth_rate(self, redshift):
            assert redshift == pytest.approx(0.73)
            return 0.81

    assert exact_growth_rate(0.73, Cosmology()) == pytest.approx(0.81)


def test_reconstruction_and_measurement_weights_are_distinct_and_fail_closed():
    catalog = _field()
    catalog["WEIGHT"] = np.array([2.0, 3.0, 4.0])
    catalog["WEIGHT_FKP"] = np.array([0.5, 0.25, 0.1])
    np.testing.assert_allclose(reconstruction_weight(catalog, "WEIGHT"), [2.0, 3.0, 4.0])
    np.testing.assert_allclose(measurement_weight(catalog, "WEIGHT*WEIGHT_FKP"), [1.0, 0.75, 0.4])
    with pytest.raises(ValueError, match="reconstruction weight expression"):
        reconstruction_weight(catalog, "WEIGHT*WEIGHT_FKP")
    with pytest.raises(ValueError, match="two-point weight expression"):
        measurement_weight(catalog, "WEIGHT")


def test_power_caps_combine_through_estimator_normalization():
    class Power:
        def __init__(self, value):
            self.value = value
            self.ells = (0, 2, 4)
            self.edges = [np.arange(4.0)]
            self.wnorm = 1.0

        @classmethod
        def average(cls, *values):
            return cls(sum(value.value for value in values))

    combined = combine_cap_power({"NGC": Power(2), "SGC": Power(3)})
    assert combined.value == 5


def test_large_scale_kaiser_estimator_recovers_bias():
    growth, bias = 0.8, 2.0
    amplitude = bias**2 + 2.0 * bias * growth / 3.0 + growth**2 / 5.0
    matter = np.array([0.01, 0.02, 0.03])
    result = estimate_linear_bias(
        [50, 70, 90],
        amplitude * matter,
        matter,
        [1, 1, 1],
        tracer="LRG",
        field_identity="LRG|broad",
        mock_ids=list(range(25)),
        growth_rate=growth,
        effective_redshift=0.7,
        config=load_analysis_config(),
    )
    assert result["bias"] == pytest.approx(bias)


def test_reconstruction_shifts_one_field_with_recsym(monkeypatch):
    calls = []

    class Reconstruction(_StagedReconstruction):
        def __init__(self, **kwargs):
            calls.append(("init", kwargs))
            self.calls = calls
            self.mpicomm = SimpleNamespace(size=1, rank=0)
            _set_mock_geometry(self)

        def read_shifted_positions(self, positions, field):
            calls.append(("shift", field))
            return np.asarray(positions) + 1.0

    monkeypatch.setitem(
        sys.modules,
        "pyrecon",
        types.SimpleNamespace(IterativeFFTReconstruction=Reconstruction),
    )
    _patch_adoption(monkeypatch)
    data, randoms = _field(), _field()
    data["WEIGHT"], randoms["WEIGHT"] = np.array([2.0, 3.0, 4.0]), np.array([5.0, 6.0, 7.0])
    data["WEIGHT_FKP"], randoms["WEIGHT_FKP"] = np.full(3, 0.1), np.full(3, 0.2)
    shifted_data, shifted_randoms, record = run_reconstruction(
        data,
        randoms,
        bias=2.0,
        growth_rate=0.8,
        smoothing_radius_mpc_h=15,
        convention="recsym",
        **_reconstruction_controls(),
    )
    assert "mpicomm" not in calls[0][1]
    assert calls[0][1]["nmesh"].tolist() == [8, 8, 8]
    assert calls[0][1]["boxsize"].tolist() == [32.0, 32.0, 32.0]
    assert calls[0][1]["boxcenter"].tolist() == [20.0, 21.0, 22.0]
    assert "niterations" not in calls[0][1]
    assert calls[0][1]["resampler"] == "tsc"
    assert "kw_weights" not in calls[0][1]
    np.testing.assert_allclose(calls[1][1], [2.0, 3.0, 4.0])
    np.testing.assert_allclose(calls[2][1], [5.0, 6.0, 7.0])
    assert calls[3][0] == "set_density_contrast"
    # The library's own consistency block is disabled because installed pyrecon 1.0.0 sums
    # ragged boolean-selected slabs inside it and raises for every irregular footprint. The
    # flag alone would be a silent flip, so the record must also carry why it is off and what
    # replaced it; the guard that lived inside the block is asserted separately below.
    # The value that actually reaches the library. It is the configured PyRecon-native floor,
    # identical for every field and every photo-z tier; PyRecon applies its own mean-random-weight
    # scaling to it. The superseded prescription handed a field-derived scalar here instead.
    assert calls[3][1] == {
        "threshold_randoms": 0.01,
        "smoothing_radius": 15.0,
        "check": False,
        "kw_weights": None,
    }
    assert record["pyrecon_internal_check"] is False
    assert record["pyrecon_internal_check_reason"] == (
        "pyrecon_1.0.0_ragged_slab_sum_in_density_check"
    )
    assert record["very_few_randoms_guard"] == "reimplemented_from_disabled_library_check"
    assert calls[4] == ("run", {"niterations": 3})
    assert calls[5:] == [("shift", "disp+rsd"), ("shift", "disp+rsd")]
    np.testing.assert_allclose(shifted_data["X_SHIFTED"], data["X"] + 1)
    np.testing.assert_allclose(shifted_randoms["X_SHIFTED"], randoms["X"] + 1)
    assert "X_SHIFTED" not in data
    assert record["realized_nmesh"] == [8, 8, 8]
    assert record["realized_cellsize_mpc_h"] == [4.0, 4.0, 4.0]
    assert record["geometry_preflight"]["padded_target_extent_mpc_h"] == [28.0] * 3
    assert record["geometry_preflight"]["quantization_expansion_mpc_h"] == [4.0] * 3
    assert record["iterations"] == 3
    assert record["weight_expression"] == "WEIGHT"
    assert record["pyrecon_kw_weights"] is None
    assert record["threshold_field_identity"] == MASK_FAMILY["identity"]
    assert record["mask_family"] == dict(MASK_FAMILY)


def test_radial_shell_partition_tiles_the_field_without_a_sliver_shell():
    """Equal shells covering the extent exactly, so no shell is a statistical accident."""
    partition = radial_shell_partition(100.0, 300.0, width_mpc_h=30.0)
    assert partition["count"] == 7
    assert partition["width_mpc_h"] == pytest.approx(200.0 / 7)
    assert partition["edges_mpc_h"][0] == 100.0
    assert partition["edges_mpc_h"][-1] == pytest.approx(300.0)
    # A trimming partition would leave 20 Mpc/h of field in a final shell of its own; this one
    # spreads the remainder across every shell instead.
    assert 30.0 * partition["count"] > 200.0
    single = radial_shell_partition(100.0, 110.0, width_mpc_h=30.0)
    assert single["count"] == 1 and single["width_mpc_h"] == pytest.approx(10.0)
    with pytest.raises(ValueError, match="positive increasing radial extent"):
        radial_shell_partition(300.0, 100.0, width_mpc_h=30.0)


def test_reconstruction_measures_and_gates_every_redshift_shell(monkeypatch):
    """The per-shell table is built, populated, and gated on a healthy field."""
    _shell_reconstruction(monkeypatch)
    data, randoms = _radial_field(), _radial_field()
    _, _, record = run_reconstruction(
        data,
        randoms,
        bias=2.0,
        growth_rate=0.8,
        smoothing_radius_mpc_h=15,
        convention="recsym",
        **_reconstruction_controls(),
    )
    diagnostics = record["shell_diagnostics"]
    # The width follows the smoothing radius rather than an absolute distance, and the
    # partition reaches one painting stencil past the outermost object so every supported
    # cell has a shell.
    assert diagnostics["shell_width_smoothing_multiple"] == 2.0
    partition = diagnostics["partition"]
    assert partition["target_width_mpc_h"] == 30.0
    # The stencil reach is the norm of the three per-axis offsets, 1.5 * 4 * sqrt(3), not one
    # of them: a node can be one and a half cells away on all three axes at once.
    assert partition["minimum_mpc_h"] == pytest.approx(
        np.sqrt(100.0**2 + 8) - 1.5 * 4.0 * np.sqrt(3.0)
    )
    assert partition["count"] == len(diagnostics["shells"]) == 7
    for shell in diagnostics["shells"]:
        assert shell["object_count"] == 3
        assert shell["supported_cell_count"] == 3
        assert shell["solved_zeroed_cell_count"] == 0
        assert shell["retained_supported_cell_fraction"] == 1.0
        assert shell["displacement_rms_ratio"] == pytest.approx(1.0)
        assert shell["moved_object_fraction"] == 1.0
    assert diagnostics["field_supported_cell_count"] == 21
    assert diagnostics["field_displacement_rms_mpc_h"] == pytest.approx(np.sqrt(3.0))
    # Nothing here scores the reconstruction. Every displacement-derived summary is evidence,
    # and the one refusal left is on a non-finite scale, which is field invalidity (SPEC 12.1).
    assert diagnostics["decides_nothing"] is True
    assert diagnostics["gated_quantities"] == ["displacement_rms_is_finite"]
    for name in (
        "retained_supported_cell_fraction",
        "displacement_rms_mpc_h",
        "displacement_rms_ratio_to_median_shell",
        "moved_object_fraction",
    ):
        assert name in diagnostics["evidence_only_quantities"]
    # The retired limits are gone from the record as well as from the check, so a reader cannot
    # recover a threshold to compare against.
    for retired in (
        "minimum_retained_supported_cell_fraction",
        "minimum_displacement_rms_ratio_to_median_shell",
        "maximum_displacement_rms_ratio_to_median_shell",
        "all_shells_within_gates",
    ):
        assert retired not in diagnostics
    # The whole reconstruction record is canonically hashed into every product's provenance,
    # which forbids NumPy scalars and any non-finite value. A shell table that cannot be
    # hashed would abort the cell after the measurement rather than before it.
    assert json.dumps(record, sort_keys=True, allow_nan=False)


def _shell_record(monkeypatch, **kwargs):
    """Reconstruct one stubbed field and hand back its per-shell diagnostic table."""
    _shell_reconstruction(monkeypatch, **kwargs)
    _, _, record = run_reconstruction(
        _radial_field(),
        _radial_field(),
        bias=2.0,
        growth_rate=0.8,
        smoothing_radius_mpc_h=15,
        convention="recsym",
        **_reconstruction_controls(),
    )
    return record["shell_diagnostics"]


def _shell_at(diagnostics, low, high):
    """The one shell whose bounds contain the interval the stub singled out."""
    inside = [
        shell
        for shell in diagnostics["shells"]
        if float(shell["radial_range_mpc_h"][0]) <= low
        and float(shell["radial_range_mpc_h"][1]) >= high
    ]
    assert len(inside) == 1, f"expected one shell spanning [{low}, {high}], got {len(inside)}"
    return inside[0]


def test_a_zeroed_redshift_shell_is_published_rather_than_refused(monkeypatch):
    """The defect the old whole-field check could not see, on a field that is otherwise ordinary.

    Every object still moves and the whole-field displacement RMS is unchanged, because the stub
    shifts every position. Only the per-shell cell accounting can tell that a redshift range lost
    its density field -- which is why the accounting is kept.

    It is kept as *evidence*. The retained fraction once carried a `0.10` limit that stopped the
    cell here; that limit scored a displacement, and a displacement is not what this experiment
    measures (SPEC 12.2). So the run completes and the record says plainly that this shell retained
    nothing, which is what a reader needs to see.
    """
    dark = _shell_cells_between(185.0, 215.0)
    diagnostics = _shell_record(monkeypatch, zeroed_cells=dark)

    assert _shell_at(diagnostics, 185.0, 215.0)["retained_supported_cell_fraction"] == 0.0
    # Every other shell kept its support, so the record localizes the defect rather than just
    # reporting that something is wrong somewhere.
    others = [
        shell
        for shell in diagnostics["shells"]
        if not (
            float(shell["radial_range_mpc_h"][0]) <= 185.0
            and float(shell["radial_range_mpc_h"][1]) >= 215.0
        )
    ]
    assert others and all(s["retained_supported_cell_fraction"] == 1.0 for s in others)
    assert diagnostics["decides_nothing"] is True


def test_a_runaway_shell_displacement_is_published_rather_than_refused(monkeypatch):
    """One shell of seven carrying all the displacement, reported at its true size.

    This field is also why the ratio is referenced to the median shell and not to the field: its
    ratio to the field-wide RMS is capped at 2.65, the square root of the shell count, which is the
    most any shell can reach against an average it dominates. Against the median it reads 401, so
    the published number is informative at the scale the anomaly actually has.
    """

    def runaway(positions):
        moved = positions + 1.0
        radii = np.sqrt((positions**2).sum(axis=1))
        moved[(radii >= 185.0) & (radii < 215.0)] += 400.0
        return moved

    diagnostics = _shell_record(monkeypatch, shift=runaway)
    ratio = _shell_at(diagnostics, 185.0, 215.0)["displacement_rms_ratio_to_median_shell"]
    assert ratio == pytest.approx(401.0, rel=1e-3)
    # Referenced to the whole field the same shell would have read 2.65 at most.
    assert ratio > np.sqrt(len(diagnostics["shells"]))
    assert diagnostics["decides_nothing"] is True


def test_a_collapsed_shell_displacement_is_published_rather_than_refused(monkeypatch):
    """The other end of the same reported range: a shell whose displacement never formed."""

    def collapsed(positions):
        moved = positions + 1.0
        radii = np.sqrt((positions**2).sum(axis=1))
        selected = (radii >= 185.0) & (radii < 215.0)
        moved[selected] = positions[selected] + 1.0e-4
        return moved

    diagnostics = _shell_record(monkeypatch, shift=collapsed)
    ratio = _shell_at(diagnostics, 185.0, 215.0)["displacement_rms_ratio_to_median_shell"]
    # Far below the 0.783-1.110 range two real fields produced, and finite, so it is reported.
    assert 0.0 < ratio < 0.01
    assert diagnostics["decides_nothing"] is True


def test_a_non_finite_displacement_is_still_refused(monkeypatch):
    """Retiring the displacement *limits* did not retire field invalidity (SPEC 12.1).

    A non-finite displacement is not an anomalous measurement to publish -- it means the solve
    produced no result, so there is nothing to report about. It is caught on the shifted positions
    themselves, before any shell is summarized, which is why no shell-level repeat of the check
    exists.
    """

    def broken(positions):
        moved = positions + 1.0
        radii = np.sqrt((positions**2).sum(axis=1))
        moved[(radii >= 185.0) & (radii < 215.0)] = np.nan
        return moved

    _shell_reconstruction(monkeypatch, shift=broken)
    with pytest.raises(ValueError, match="invalid shifted data positions"):
        run_reconstruction(
            _radial_field(),
            _radial_field(),
            bias=2.0,
            growth_rate=0.8,
            smoothing_radius_mpc_h=15,
            convention="recsym",
            **_reconstruction_controls(),
        )


def test_reconstruction_carries_the_library_very_few_randoms_guard(monkeypatch):
    """The guard lived inside the disabled check block, so it is proven to have survived."""
    _shell_reconstruction(monkeypatch)
    # Three random rows inside one cell: the mesh then holds painted weight in a single cell,
    # which is exactly the state the library's disabled block refused.
    randoms = _radial_field(count=3, first=200.0, step=0.5)
    randoms["Y"] = np.full(3, 2.0)
    randoms["Z_CART"] = np.full(3, 2.0)
    with pytest.raises(ValueError, match="very few randoms"):
        run_reconstruction(
            _radial_field(),
            randoms,
            bias=2.0,
            growth_rate=0.8,
            smoothing_radius_mpc_h=15,
            convention="recsym",
            **_reconstruction_controls(),
        )


def _frozen_family_box(*, family=MASK_FAMILY, nmesh=(12, 12, 12), boxcenter=(20.0, 21.0, 22.0)):
    return {
        "family": family,
        "nmesh": list(nmesh),
        "boxsize_mpc_h": [4.0 * value for value in nmesh],
        "boxcenter_mpc_h": list(boxcenter),
    }


def _echoing_pyrecon(monkeypatch, constructed):
    """Install a fake PyRecon that realizes exactly the geometry it was handed."""

    class Reconstruction(_StagedReconstruction):
        def __init__(self, **kwargs):
            constructed.append(kwargs)
            self.calls = []
            self.mpicomm = SimpleNamespace(size=1, rank=0)
            self.nmesh = np.asarray(kwargs["nmesh"], dtype=int)
            self.boxsize = np.asarray(kwargs["boxsize"], dtype=float)
            self.boxcenter = np.asarray(kwargs["boxcenter"], dtype=float)
            self.cellsize = self.boxsize / self.nmesh

        def read_shifted_positions(self, positions, field):
            return np.asarray(positions) + 1.0

    monkeypatch.setitem(
        sys.modules,
        "pyrecon",
        types.SimpleNamespace(IterativeFFTReconstruction=Reconstruction),
    )


def test_reconstruction_adopts_one_frozen_mesh_for_cells_with_different_extrema(
    tmp_path, monkeypatch
):
    """End to end: two members of one family run on one mesh.

    The two cells stand for two photo-z tiers or two mocks of the same field. Their catalogue
    extrema differ slightly, so the box each would derive for itself differs too, and the two
    would reconstruct on meshes of different shape -- measured as 638 against 640 cells on one
    axis between two real tier densities. Here both adopt the geometry the frozen artifact
    carries, which is deliberately not the geometry either cell would have derived, and PyRecon
    is constructed with identical mesh arguments.
    """
    record_artifact = _frozen_family_box()
    constructed = []
    _echoing_pyrecon(monkeypatch, constructed)
    controls = _reconstruction_controls()
    controls["common_geometry"] = record_artifact
    records = []
    for offset in (0.0, 0.5):
        data, randoms = _field(), _field()
        for catalog in (data, randoms):
            for column in ("X", "Y", "Z_CART"):
                catalog[column] = catalog[column] + offset
        _, _, record = run_reconstruction(
            data,
            randoms,
            bias=2.0,
            growth_rate=0.8,
            smoothing_radius_mpc_h=15,
            convention="recsym",
            **controls,
        )
        records.append(record)
    assert len(constructed) == 2
    for kwargs in constructed:
        assert kwargs["nmesh"].tolist() == [12, 12, 12]
        assert kwargs["boxsize"].tolist() == [48.0] * 3
        assert kwargs["boxcenter"].tolist() == [20.0, 21.0, 22.0]
    # Each cell still derives its own box as evidence, and those boxes differ -- which is
    # exactly why deriving the mesh made the shared artifact unusable.
    first, second = (item["geometry_preflight"] for item in records)
    assert first["box_center_mpc_h"] == [20.0, 21.0, 22.0]
    assert second["box_center_mpc_h"] == [20.5, 21.5, 22.5]
    assert first["derived_even_nmesh"] == second["derived_even_nmesh"] == [8, 8, 8]
    adopted = [item["adopted_geometry"] for item in records]
    assert adopted[0]["adopted_nmesh"] == adopted[1]["adopted_nmesh"] == [12, 12, 12]
    assert (
        adopted[0]["adopted_boxcenter_mpc_h"]
        == adopted[1]["adopted_boxcenter_mpc_h"]
        == [20.0, 21.0, 22.0]
    )
    assert records[0]["mask_family"] == dict(MASK_FAMILY)
    assert records[0]["realized_nmesh"] == [12, 12, 12]


def test_reconstruction_refuses_a_box_frozen_for_another_field(monkeypatch):
    """The family key omits mock and tier by design, so the artifact must bind its own field."""
    constructed = []
    _echoing_pyrecon(monkeypatch, constructed)
    controls = _reconstruction_controls()
    controls["common_geometry"] = _frozen_family_box(
        family=reconstruction_mask_family(tracer="LRG", cap="SGC", smoothing_radius_mpc_h=15.0),
    )
    with pytest.raises(ValueError, match="another parent or cap"):
        run_reconstruction(
            _field(),
            _field(),
            bias=2.0,
            growth_rate=0.8,
            smoothing_radius_mpc_h=15,
            convention="recsym",
            **controls,
        )
    assert not constructed


def test_reconstruction_refuses_a_frozen_box_its_catalogue_leaves(monkeypatch):
    """Adoption replaces the derivation that used to imply containment, so it is proven."""
    constructed = []
    _echoing_pyrecon(monkeypatch, constructed)
    controls = _reconstruction_controls()
    controls["common_geometry"] = _frozen_family_box(nmesh=(4, 4, 4), boxcenter=(20.0, 21.0, 22.0))
    with pytest.raises(ValueError, match="does not contain"):
        run_reconstruction(
            _field(),
            _field(),
            bias=2.0,
            growth_rate=0.8,
            smoothing_radius_mpc_h=15,
            convention="recsym",
            **controls,
        )
    assert not constructed


def test_reconstruction_requires_the_family_to_name_its_own_smoothing_radius(monkeypatch):
    _patch_adoption(monkeypatch)
    with pytest.raises(ValueError, match="names another smoothing radius"):
        run_reconstruction(
            _field(),
            _field(),
            bias=2.0,
            growth_rate=0.8,
            smoothing_radius_mpc_h=10,
            convention="recsym",
            **_reconstruction_controls(),
        )


def test_reconstruction_rejects_unregistered_convention_before_import():
    # RecSym is primary; RecIso is admitted only for its robustness run (see core sensitivity).
    with pytest.raises(ValueError, match="RecSym or the RecIso check"):
        run_reconstruction(
            _field(),
            _field(),
            bias=2.0,
            growth_rate=0.8,
            smoothing_radius_mpc_h=15,
            convention="recalt",
            **_reconstruction_controls(),
        )


def test_reconstruction_rejects_nonfinite_scientific_inputs_before_import():
    with pytest.raises(ValueError, match="finite and positive"):
        run_reconstruction(
            _field(),
            _field(),
            bias=float("nan"),
            growth_rate=0.8,
            smoothing_radius_mpc_h=15,
            convention="recsym",
            **_reconstruction_controls(),
        )


def test_reconstruction_rejects_pyrecon_internal_weighting_before_import(monkeypatch):
    import builtins

    original = builtins.__import__

    def without_pyrecon(name, *args, **kwargs):
        if name == "pyrecon" or name.startswith("pyrecon."):
            pytest.fail("invalid weighting must be refused before importing optional PyRecon")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_pyrecon)
    controls = _reconstruction_controls()
    controls["pyrecon_kw_weights"] = "WEIGHT_FKP"
    with pytest.raises(ValueError, match="kw_weights must remain absent"):
        run_reconstruction(
            _field(),
            _field(),
            bias=2.0,
            growth_rate=0.8,
            smoothing_radius_mpc_h=15,
            convention="recsym",
            **controls,
        )


def test_reconstruction_rejects_realized_mesh_drift(monkeypatch):
    class Reconstruction:
        def __init__(self, **kwargs):
            self.mpicomm = SimpleNamespace(size=1, rank=0)
            _set_mock_geometry(self, nmesh=(4, 4, 4), boxsize=(16.0, 16.0, 16.0))

    monkeypatch.setitem(
        sys.modules,
        "pyrecon",
        types.SimpleNamespace(IterativeFFTReconstruction=Reconstruction),
    )
    _patch_adoption(monkeypatch)
    with pytest.raises(ValueError, match="adopted common mesh"):
        run_reconstruction(
            _field(),
            _field(),
            bias=2.0,
            growth_rate=0.8,
            smoothing_radius_mpc_h=15,
            convention="recsym",
            **_reconstruction_controls(),
        )


@pytest.mark.parametrize("failure", ["shape", "nonfinite", "null"])
def test_reconstruction_rejects_invalid_shifted_outputs(monkeypatch, failure):
    class Reconstruction(_StagedReconstruction):
        def __init__(self, **kwargs):
            self.calls = []
            self.mpicomm = SimpleNamespace(size=1, rank=0)
            _set_mock_geometry(self)

        def read_shifted_positions(self, positions, field):
            if failure == "shape":
                return np.asarray(positions)[:, :2]
            result = np.asarray(positions).copy()
            if failure == "nonfinite":
                result[0, 0] = np.nan
            return result

    monkeypatch.setitem(
        sys.modules,
        "pyrecon",
        types.SimpleNamespace(IterativeFFTReconstruction=Reconstruction),
    )
    _patch_adoption(monkeypatch)
    expected = "null shifted" if failure == "null" else "invalid shifted"
    with pytest.raises(ValueError, match=expected):
        run_reconstruction(
            _field(),
            _field(),
            bias=2.0,
            growth_rate=0.8,
            smoothing_radius_mpc_h=15,
            convention="recsym",
            **_reconstruction_controls(),
        )


def test_measurement_bins_are_exact_identity_subsets():
    config = load_analysis_config()
    data = _field()
    data["Z_OBS"] = np.array([0.11, 0.41, 0.31])
    result = stable_measurement_subset(
        data,
        tracer="BGS_BRIGHT-21.5",
        bin_name="BGS1",
        redshift_column="Z_OBS",
        config=config,
    )
    np.testing.assert_array_equal(result["SOURCE_ID"], [1])

    renamed_config = copy.deepcopy(config)
    renamed_config["selection"]["stable_randomness"]["identity_column"] = "OBJECT_ID"
    renamed = _field()
    renamed["Z_OBS"] = np.array([0.11, 0.41, 0.31])
    renamed["OBJECT_ID"] = renamed.pop("SOURCE_ID")
    result = stable_measurement_subset(
        renamed,
        tracer="BGS_BRIGHT-21.5",
        bin_name="BGS1",
        redshift_column="Z_OBS",
        config=renamed_config,
    )
    np.testing.assert_array_equal(result["OBJECT_ID"], [1])

    invalid_identity = _field()
    invalid_identity["Z_OBS"] = np.array([0.11, 0.41, 0.31])
    invalid_identity["SOURCE_ID"] = invalid_identity["SOURCE_ID"].astype(float)
    with pytest.raises(ValueError, match="non-negative integers"):
        stable_measurement_subset(
            invalid_identity,
            tracer="BGS_BRIGHT-21.5",
            bin_name="BGS1",
            redshift_column="Z_OBS",
            config=config,
        )
    with pytest.raises(ValueError, match="omits required Z_OBS"):
        stable_measurement_subset(
            _field(),
            tracer="BGS_BRIGHT-21.5",
            bin_name="BGS1",
            redshift_column="Z_OBS",
            config=config,
        )


def test_reconstruction_rejects_a_multi_rank_library_communicator(monkeypatch):
    class Reconstruction:
        def __init__(self, **kwargs):
            self.mpicomm = SimpleNamespace(size=2, rank=0)
            _set_mock_geometry(self)

    monkeypatch.setitem(
        sys.modules,
        "pyrecon",
        types.SimpleNamespace(IterativeFFTReconstruction=Reconstruction),
    )
    _patch_adoption(monkeypatch)
    with pytest.raises(RuntimeError, match="pyrecon on one rank"):
        run_reconstruction(
            _field(),
            _field(),
            bias=2.0,
            growth_rate=0.8,
            smoothing_radius_mpc_h=15,
            convention="recsym",
            **_reconstruction_controls(),
        )


def test_xi_uses_landy_szalay_pre_and_shifted_random_post(monkeypatch):
    calls = []

    class Correlation:
        def __init__(self, *args, **kwargs):
            calls.append((args, kwargs))

    monkeypatch.setitem(
        sys.modules,
        "pycorr",
        types.SimpleNamespace(TwoPointCorrelationFunction=Correlation),
    )
    data, randoms = _field(), _field()
    data["WEIGHT"], randoms["WEIGHT"] = np.array([2.0, 3.0, 4.0]), np.array([5.0, 6.0, 7.0])
    data["WEIGHT_FKP"], randoms["WEIGHT_FKP"] = np.full(3, 0.1), np.full(3, 0.2)
    for catalog in (data, randoms):
        catalog.update(
            X_SHIFTED=catalog["X"] + 1,
            Y_SHIFTED=catalog["Y"] + 1,
            Z_CART_SHIFTED=catalog["Z_CART"] + 1,
        )
    compute_xi(
        data,
        randoms,
        reconstructed=False,
        s_edges_mpc_h=[0, 1, 2],
        mu_bins=4,
        line_of_sight="midpoint",
        nthreads=1,
        declared_estimator="landy_szalay",
        weight_expression="WEIGHT*WEIGHT_FKP",
    )
    assert "shifted_positions1" not in calls[-1][1]
    assert "mpicomm" not in calls[-1][1]
    assert calls[-1][1]["estimator"] == "landyszalay"
    np.testing.assert_allclose(calls[-1][1]["data_weights1"], [0.2, 0.3, 0.4])
    np.testing.assert_allclose(calls[-1][1]["randoms_weights1"], [1.0, 1.2, 1.4])
    compute_xi(
        data,
        randoms,
        reconstructed=True,
        s_edges_mpc_h=[0, 1, 2],
        mu_bins=4,
        line_of_sight="midpoint",
        nthreads=1,
        declared_estimator="shifted_random",
        weight_expression="WEIGHT*WEIGHT_FKP",
    )
    assert "shifted_positions1" in calls[-1][1]
    assert calls[-1][1]["los"] == "midpoint"
    assert calls[-1][1]["estimator"] == "landyszalay"


def test_xi_rejects_any_non_midpoint_line_of_sight_before_import():
    with pytest.raises(ValueError, match="midpoint"):
        compute_xi(
            _field(),
            _field(),
            reconstructed=False,
            s_edges_mpc_h=[0, 1],
            mu_bins=2,
            line_of_sight="endpoint",
            nthreads=1,
            declared_estimator="landy_szalay",
            weight_expression="WEIGHT*WEIGHT_FKP",
        )


def test_xi_rejects_estimator_label_that_does_not_name_its_implementation():
    with pytest.raises(ValueError, match="configured xi estimator label"):
        compute_xi(
            _field(),
            _field(),
            reconstructed=True,
            s_edges_mpc_h=[0, 1],
            mu_bins=2,
            line_of_sight="midpoint",
            nthreads=1,
            declared_estimator="landy_szalay",
            weight_expression="WEIGHT*WEIGHT_FKP",
        )


def test_secondary_power_rejects_wrong_los_or_multipoles_before_import():
    config = copy.deepcopy(load_analysis_config())
    config["measurement"]["power_spectrum_secondary"]["line_of_sight"] = "endpoint"
    with pytest.raises(ValueError, match="firstpoint"):
        compute_power_spectrum_secondary(
            _field(),
            _field(),
            reconstructed=False,
            geometry_artifact=_power_geometry(config),
            config=config,
        )


def test_secondary_power_uses_measurement_weights(monkeypatch):
    calls = []

    class Power:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self.poles = object()

    monkeypatch.setitem(sys.modules, "pypower", types.SimpleNamespace(CatalogFFTPower=Power))
    data, randoms = _field(), _field()
    data["WEIGHT"], randoms["WEIGHT"] = np.array([2.0, 3.0, 4.0]), np.array([5.0, 6.0, 7.0])
    data["WEIGHT_FKP"], randoms["WEIGHT_FKP"] = np.full(3, 0.1), np.full(3, 0.2)
    config = load_analysis_config()
    compute_power_spectrum_secondary(
        data,
        randoms,
        reconstructed=False,
        geometry_artifact=_power_geometry(config),
        config=config,
    )
    np.testing.assert_allclose(calls[0]["data_weights1"], [0.2, 0.3, 0.4])
    np.testing.assert_allclose(calls[0]["randoms_weights1"], [1.0, 1.2, 1.4])
    assert calls[0]["nmesh"] == (8, 8, 8)
    assert calls[0]["interlacing"] == 3
    config = copy.deepcopy(load_analysis_config())
    config["measurement"]["power_spectrum_secondary"]["multipoles"] = [0, 2]
    with pytest.raises(ValueError, match="multipoles 0,2,4"):
        compute_power_spectrum_secondary(
            _field(),
            _field(),
            reconstructed=False,
            geometry_artifact=_power_geometry(config),
            config=config,
        )


def test_uniform_edges_have_exact_declared_endpoints_and_reject_partial_bins():
    edges = pipeline_reconstruction._uniform_edges(0.0, 0.3, 0.1, label="test")
    np.testing.assert_allclose(edges, [0.0, 0.1, 0.2, 0.3])
    assert edges[-1] == pytest.approx(0.3)
    with pytest.raises(ValueError, match="integer multiple"):
        pipeline_reconstruction._uniform_edges(0.0, 0.31, 0.1, label="test")


def test_reconstruction_algorithm_is_explicit_and_fail_closed():
    with pytest.raises(ValueError, match="iterative_fft"):
        run_reconstruction(
            _field(),
            _field(),
            bias=2.0,
            growth_rate=0.8,
            smoothing_radius_mpc_h=15,
            convention="recsym",
            **{**_reconstruction_controls(), "algorithm": "another_algorithm"},
        )


def test_the_reconstructed_field_is_solved_once_per_parent_tracer():
    """The five measurement bins are cuts of two parent catalogues, not five solves.

    The field identity is what keeps one parent's artifacts from being mistaken for another's,
    so every bin of a parent resolves to that parent and a bin of the other parent is refused.
    The contract is asserted in full in ``tests/test_parent_reconstruction.py``; what is here is
    the piece the rest of this module's fixtures depend on.
    """
    config = load_analysis_config()
    assert config["measurement"]["reconstruction"]["scope"] == "one_field_per_tracer"

    identities = set()
    for tracer in config["tracers"]:
        for spec in measurement_bins(tracer, config):
            identity, bin_name = pipeline_reconstruction.reconstruction_field_identity(
                tracer, spec["name"], scope="one_field_per_tracer", config=config
            )
            assert identity == f"{tracer}|broad" and bin_name is None
            identities.add(identity)
    assert identities == {"LRG|broad", "BGS_BRIGHT-21.5|broad"}

    # A bin belonging to the other tracer is refused, so a bright bin cannot name a red field.
    with pytest.raises(ValueError, match="not a configured measurement bin"):
        pipeline_reconstruction.reconstruction_field_identity(
            "LRG", "BGS1", scope="one_field_per_tracer", config=config
        )
    # The retired per-bin division is not a fallback; it is an unregistered scope like any other.
    for scope in ("one_field_per_measurement_bin", "one_field_per_cap"):
        with pytest.raises(ValueError, match="not a registered field division"):
            pipeline_reconstruction.reconstruction_field_identity(
                "LRG", "LRG2", scope=scope, config=config
            )


def test_the_reconstruction_window_is_the_parent_field_with_no_pad():
    """No padding anywhere: the parent field is solved whole and clipped at its own edges."""
    config = load_analysis_config()
    pytest.importorskip("cosmoprimo")
    cosmology = build_mock_cosmology(config)
    window = pipeline_reconstruction.reconstruction_window_bounds(
        "LRG", cosmology=cosmology, smoothing_radius_mpc_h=15.0, config=config
    )
    field = config["tracers"]["LRG"]["field"]
    assert window["window_redshift_range"] == [
        pytest.approx(float(field["zmin"]), abs=1e-12),
        pytest.approx(float(field["zmax"]), abs=1e-12),
    ]
    assert window["window_comoving_range_mpc_h"] == window["field_comoving_range_mpc_h"]
    assert window["clipped_at_field_minimum"] is True
    assert window["clipped_at_field_maximum"] is True
    assert window["measurement_bin"] is None


def test_a_degenerate_tracer_field_is_refused():
    # The window is the field, so a field that spans nothing is a solve with nothing in it.
    config = copy.deepcopy(load_analysis_config())
    pytest.importorskip("cosmoprimo")
    cosmology = build_mock_cosmology(config)
    config["tracers"]["LRG"]["field"] = {"zmin": 0.6, "zmax": 0.6}
    with pytest.raises(ValueError, match="positive redshift interval"):
        pipeline_reconstruction.reconstruction_window_bounds(
            "LRG", cosmology=cosmology, smoothing_radius_mpc_h=15.0, config=config
        )


def test_weighted_redshift_is_ratio_of_pooled_integrals_not_mean_of_cap_means():
    records = [
        weighted_random_zeff_accumulator(
            z, distance, density, [10.0, 20.0], [1.0, 2.0], positions="selected_random_redshifts"
        )
        for z, distance, density in [
            ([0.4, 0.5], [100.0, 150.0], [0.1, 0.2]),
            ([0.6, 0.8], [200.0, 300.0], [0.4, 0.5]),
        ]
    ]
    base = [
        np.array(distance) ** 2 * np.array(density) ** 2 * [10.0, 20.0] * [1.0, 2.0]
        for distance, density in [([100.0, 150.0], [0.1, 0.2]), ([200.0, 300.0], [0.4, 0.5])]
    ]
    expected = (np.dot([0.4, 0.5], base[0]) + np.dot([0.6, 0.8], base[1])) / (
        base[0].sum() + base[1].sum()
    )
    measured = sum(item["numerator"] for item in records) / sum(
        item["denominator"] for item in records
    )
    assert measured == pytest.approx(expected)
    assert measured != pytest.approx(
        np.mean([item["numerator"] / item["denominator"] for item in records])
    )


def test_reconstruction_diagnostics_consume_supplied_snapshot(monkeypatch):
    constructed = []
    _echoing_pyrecon(monkeypatch, constructed)
    controls = _reconstruction_controls()
    observed = []
    original = pipeline_reconstruction.reconstruction_shell_diagnostics

    def record_snapshot(*args, **kwargs):
        observed.append(kwargs["config"])
        return original(*args, **kwargs)

    monkeypatch.setattr(
        pipeline_reconstruction, "reconstruction_shell_diagnostics", record_snapshot
    )
    run_reconstruction(
        _field(),
        _field(),
        bias=2.0,
        growth_rate=0.8,
        smoothing_radius_mpc_h=15.0,
        convention="recsym",
        **controls,
    )
    assert len(observed) == 1 and observed[0] is controls["config"]
