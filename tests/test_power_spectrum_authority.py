"""Direct P(k) geometry, estimator arguments, cap normalization and containment."""

import json
import sys
import types
import numpy as np
import pytest
from pzbao.config import load_analysis_config
from pzbao.clustering import (
    compute_power_spectrum_secondary,
    derive_power_spectrum_geometry,
    combine_cap_power,
    _direct_lattice_mode_counts,
)

TRACER = "BGS_BRIGHT-21.5"


def _positions():
    # Large enough that the fundamental mode divides the 0.005 comparison width exactly, which
    # is the condition the producer enforces rather than a property of this fixture.
    return [[0.0, 0.0, 0.0], [2000.0, 2000.0, 2000.0]]


def _geometry(config):
    """The real producer, on the real configuration."""
    return derive_power_spectrum_geometry(
        _positions(),
        tracer=TRACER,
        config=config,
    )


def _persist_and_reload(artifact, tmp_path):
    """Cross the serialization boundary the production route actually crosses."""
    path = tmp_path / "power_spectrum_geometry.json"
    path.write_text(json.dumps(artifact, sort_keys=True))
    return json.loads(path.read_text())


def _field():
    return {
        "X": np.array([10.0, 20.0, 30.0]),
        "Y": np.array([11.0, 21.0, 31.0]),
        "Z_CART": np.array([12.0, 22.0, 32.0]),
        "WEIGHT": np.ones(3),
        "WEIGHT_FKP": np.full(3, 0.5),
    }


def _consume(artifact, config, monkeypatch):
    """The real consumer, stopped at the library boundary and nowhere earlier."""
    calls = []

    class Power:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self.poles = object()

    monkeypatch.setitem(sys.modules, "pypower", types.SimpleNamespace(CatalogFFTPower=Power))
    compute_power_spectrum_secondary(
        _field(),
        _field(),
        reconstructed=False,
        geometry_artifact=artifact,
        config=config,
    )
    return calls


def test_a_box_that_does_not_contain_the_sample_is_refused(tmp_path, monkeypatch):
    """No configuration digest can establish that the box contains *this* sample.

    The box is built from a reviewed position envelope that this route never reads, so scoping
    validity to the leaves the route does read leaves a real gap: a geometry frozen for one
    envelope stays valid for a sample that outgrew it. Positions outside the box are wrapped or
    dropped by the FFT and the spectrum is wrong with nothing downstream detecting it, so the
    property is checked directly rather than through a hash that stands in for it.
    """
    config = load_analysis_config()
    artifact = _persist_and_reload(_geometry(config), tmp_path)

    outside = _field()
    half = 0.5 * np.asarray(artifact["boxsize_mpc_h"], dtype=float)
    centre = np.asarray(artifact["boxcenter_mpc_h"], dtype=float)
    outside["X"] = np.array([centre[0] + half[0] + 10.0, 0.0, 0.0])

    calls = []

    class Power:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self.poles = object()

    monkeypatch.setitem(sys.modules, "pypower", types.SimpleNamespace(CatalogFFTPower=Power))
    with pytest.raises(ValueError, match="does not contain the data positions"):
        compute_power_spectrum_secondary(
            outside,
            _field(),
            reconstructed=False,
            geometry_artifact=artifact,
            config=config,
        )
    assert not calls, "the estimator was reached with positions outside its own box"


def test_a_position_exactly_on_the_upper_face_is_refused(tmp_path, monkeypatch):
    """The mesh spans [centre - size/2, centre + size/2), so the upper face wraps to index zero.

    Admitting equality there would admit precisely the position this guard exists to catch, and
    the wrap is silent: the spectrum comes back wrong with nothing downstream detecting it.
    """
    config = load_analysis_config()
    artifact = _persist_and_reload(_geometry(config), tmp_path)
    half = 0.5 * np.asarray(artifact["boxsize_mpc_h"], dtype=float)
    centre = np.asarray(artifact["boxcenter_mpc_h"], dtype=float)

    on_face = _field()
    on_face["X"] = np.array([centre[0] + half[0], 0.0, 0.0])

    calls = []

    class Power:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self.poles = object()

    monkeypatch.setitem(sys.modules, "pypower", types.SimpleNamespace(CatalogFFTPower=Power))
    with pytest.raises(ValueError, match="does not contain the data positions"):
        compute_power_spectrum_secondary(
            on_face, _field(), reconstructed=False, geometry_artifact=artifact, config=config
        )
    assert not calls

    # A position just inside the face is still accepted, so the bound is exclusive and not shrunk.
    inside = _field()
    inside["X"] = np.array([centre[0] + half[0] - 1e-6, 0.0, 0.0])
    compute_power_spectrum_secondary(
        inside, _field(), reconstructed=False, geometry_artifact=artifact, config=config
    )
    assert calls


def test_linear_power_geometry_roundtrip_and_resolved_constructor_controls(tmp_path, monkeypatch):
    config = load_analysis_config()
    geometry = _persist_and_reload(_geometry(config), tmp_path)
    calls = _consume(geometry, config, monkeypatch)
    assert len(calls) == 1
    kwargs = calls[0]
    spec = config["measurement"]["power_spectrum_secondary"]
    assert kwargs["los"] == "firstpoint" and kwargs["ells"] == (0, 2, 4)
    assert kwargs["interlacing"] == spec["interlacing"]
    assert kwargs["resampler"] == spec["resampler"]
    np.testing.assert_array_equal(kwargs["data_weights1"], np.full(3, 0.5))
    np.testing.assert_array_equal(kwargs["randoms_weights1"], np.full(3, 0.5))
    np.testing.assert_array_equal(kwargs["edges"], geometry["native_k_edges_h_mpc"])
    np.testing.assert_allclose(
        np.asarray(kwargs["boxsize"]) / kwargs["nmesh"], spec["cell_size_mpc_h"]
    )
    native = geometry["native_bin_width_h_mpc"]
    np.testing.assert_allclose(
        np.asarray(geometry["comparison_k_edges_h_mpc"]) / native,
        np.rint(np.asarray(geometry["comparison_k_edges_h_mpc"]) / native),
        atol=1e-10,
    )


def test_linear_lattice_mode_counts_match_explicit_three_dimensional_enumeration():
    mesh, size, edges = (
        np.array([8, 10, 12]),
        np.array([300.0, 400.0, 500.0]),
        np.linspace(0, 0.1, 11),
    )
    axes = [2 * np.pi * np.fft.fftfreq(n, d=length / n) for n, length in zip(mesh, size)]
    radii = np.sqrt(sum(value**2 for value in np.meshgrid(*axes, indexing="ij")))
    expected = np.histogram(radii, bins=edges)[0]
    np.testing.assert_array_equal(
        _direct_lattice_mode_counts(mesh, size, edges, chunk_size=7), expected
    )


def test_linear_native_power_cap_average_sums_numerators_and_normalizations(tmp_path):
    pypower = pytest.importorskip("pypower")
    grid = np.array([0.0, 0.1, 0.2, 0.3])
    modes = (grid[:-1] + grid[1:]) / 2
    powers = {}
    for cap, normalization, amplitude in (("NGC", 100.0, 2.0), ("SGC", 20.0, 7.0)):
        powers[cap] = pypower.PowerSpectrumMultipoles(
            edges=[grid],
            modes=[modes],
            power_nonorm=np.arange(1.0, 10.0).reshape(3, 3) * amplitude * normalization,
            nmodes=np.ones(3),
            ells=(0, 2, 4),
            wnorm=normalization,
        )
    combined = combine_cap_power(powers)
    expected = (powers["NGC"].power_nonorm + powers["SGC"].power_nonorm) / 120.0
    np.testing.assert_allclose(
        combined.get_power(remove_shotnoise=False, null_zero_mode=False), expected
    )
    path = tmp_path / "power.npy"
    combined.save(str(path))
    loaded = pypower.PowerSpectrumMultipoles.load(str(path))
    np.testing.assert_array_equal(loaded.power_nonorm, combined.power_nonorm)
    assert loaded.wnorm == combined.wnorm
    powers["SGC"].edges[0] = powers["SGC"].edges[0] + 0.01
    with pytest.raises(ValueError, match="grids differ"):
        combine_cap_power(powers)


@pytest.mark.parametrize("coordinate", ["X", "Y", "Z_CART"])
def test_linear_nonfinite_power_positions_refuse_before_fft(monkeypatch, coordinate):
    config = load_analysis_config()
    geometry = _geometry(config)
    data = _field()
    data[coordinate][0] = np.nan
    monkeypatch.setitem(
        sys.modules,
        "pypower",
        types.SimpleNamespace(
            CatalogFFTPower=lambda **kwargs: pytest.fail("non-finite input reached FFT")
        ),
    )
    with pytest.raises(ValueError, match="finite three-dimensional"):
        compute_power_spectrum_secondary(
            data, _field(), reconstructed=False, geometry_artifact=geometry, config=config
        )
