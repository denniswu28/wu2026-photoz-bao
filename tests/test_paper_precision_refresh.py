from pzbao.config import analysis_config_sha256

"""Values and source boundaries for the advisor-requested fractional precision figure."""

import json

import numpy as np
import pytest
import yaml

from pzbao.figures import data as precision


def caps(volume=2.0e9, density=1.0e-4):
    return [dict(cap=cap, volume_mpc_h3=volume, nbar_h3_mpc3=density) for cap in ("NGC", "SGC")]


def test_fractional_units_and_cosmic_variance_normalization():
    row = precision.precision_row("LRG1", "native", 0.5, caps(), 10000.0, "LRG")
    assert row["veff_mpc_h3"] == pytest.approx(1.0e9)
    assert row["sigma_r_over_r"] == pytest.approx(0.008939136118525084)
    assert row["cosmic_variance_sigma_r_over_r"] == pytest.approx(0.004469568059262542)
    assert row["sigma_r_over_r"] / row["cosmic_variance_sigma_r_over_r"] == pytest.approx(2.0)
    enlarged = precision.precision_row("LRG1", "native", 0.5, caps(volume=8.0e9), 10000.0, "LRG")
    assert enlarged["sigma_r_over_r"] == pytest.approx(row["sigma_r_over_r"] / 2)


def test_caps_are_weighted_separately_and_dr1_uses_the_same_p0():
    components = caps()
    components[1]["nbar_h3_mpc3"] = 3.0e-4
    row = precision.precision_row("BGS1", "native", 0.15, components, 7000.0, "BGS")
    expected = 2.0e9 * ((0.7 / 1.7) ** 2 + (2.1 / 3.1) ** 2)
    assert row["veff_mpc_h3"] == pytest.approx(expected)
    dr1 = precision.precision_row("BGS1", "desi_dr1", 0.15, components, 7000.0, "BGS")
    assert dr1["sigma_r_over_r"] == row["sigma_r_over_r"]
    assert expected != pytest.approx(4.0e9 * (1.4 / 2.4) ** 2)


@pytest.mark.parametrize("volume,density", [(0.0, 1.0e-4), (2.0e9, -1.0e-4), (np.nan, 1.0e-4)])
def test_invalid_forecast_values_rejected(volume, density):
    with pytest.raises(ValueError):
        precision.precision_row("LRG1", "native", 0.5, caps(volume, density), 10000.0, "LRG")


def test_wrong_release_identity_stops_before_output(tmp_path, monkeypatch):
    source, frozen = tmp_path / "source", tmp_path / "frozen"
    source.mkdir()
    frozen.mkdir()
    (source / "complete.json").write_text(
        '{"source_bytes_unchanged":true,"covariance_readback_matches":true}'
    )
    (source / "provenance.json").write_text("{}")
    expected_identity = dict(
        run_name="baseline", analysis_config_sha256="b" * 64, code_commit="a" * 40
    )
    identity = dict(expected_identity, code_commit="0" * 40)
    (frozen / "run.yaml").write_text(yaml.safe_dump(identity))
    with pytest.raises(ValueError, match="Wrong frozen measurement release"):
        precision.read_precision_sources(precision.Sources(), frozen, expected_identity)
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("changed", [None, "analysis", "registry"])
def test_complete_saved_snapshot_identity_precedes_preparation(tmp_path, monkeypatch, changed):
    source, frozen = tmp_path / "source", tmp_path / "frozen"
    source.mkdir()
    frozen.mkdir()
    (source / "complete.json").write_text(
        '{"source_bytes_unchanged":true,"covariance_readback_matches":true}'
    )
    (source / "provenance.json").write_text("{}")
    config = {"tracers": {"LRG": {"fkp_p0_h3_mpc3": 10000.0}}}
    registry = {"routes": [{"alpha_sn": 1.1}]}
    # The producer hashes both serialized source components, as pzbao.run does.
    identity = dict(
        run_name="baseline",
        code_commit="a" * 40,
        analysis_config_sha256=analysis_config_sha256(
            {"analysis": config, "frozen_alpha_sn": registry}
        ),
    )
    (frozen / "run.yaml").write_text(yaml.safe_dump(identity))
    if changed == "analysis":
        config["tracers"]["LRG"]["fkp_p0_h3_mpc3"] = 7000.0
    if changed == "registry":
        registry["routes"][0]["alpha_sn"] = 1.2
    (frozen / "analysis.yaml").write_text(yaml.safe_dump(config))
    (frozen / "frozen_alpha_sn.json").write_text(json.dumps(registry))
    directory = frozen / "inputs/units/LRG"
    directory.mkdir(parents=True)
    (directory / "result.yaml").write_text(yaml.safe_dump(dict(identity, code_commit="0" * 40)))
    expected = (
        "Wrong preparation identity"
        if changed is None
        else "Changed configuration or frozen registry"
    )
    with pytest.raises(ValueError, match=expected):
        precision.read_precision_sources(precision.Sources(), frozen, identity)
    assert not (tmp_path / "output").exists()


def test_frozen_preparation_and_dr1_reference_use_actual_cap_volumes(tmp_path):
    from pzbao.figures.style import ZBIN_GRID_PANELS

    config = dict(
        tracers={},
        analysis=dict(workloads=dict(full_grid=dict(sigma_z_tiers=list(precision.TIERS)))),
    )
    preparations, bins, components = {}, [], []
    for name, (tracer, lower, upper, _) in zip(precision.BINS, ZBIN_GRID_PANELS):
        tracer = "LRG" if tracer == "LRG" else "BGS_BRIGHT-21.5"
        shell = dict(name=name, zmin=lower, zmax=upper)
        config["tracers"].setdefault(tracer, dict(fkp_p0_h3_mpc3=10000.0, measurement_bins=[]))[
            "measurement_bins"
        ].append(shell)
        bins.append(
            dict(
                bin=name, tracer=tracer, zmin=lower, zmax=upper, redshift_center=(lower + upper) / 2
            )
        )
        components.append(
            [
                dict(region=cap, shell_volume_mpc_h3=volume / 4, nbar_h3_mpc3=1e-4)
                for cap, volume in (("NGC", 1e9), ("SGC", 2e9))
            ]
        )
    for tracer, spec in config["tracers"].items():
        prep = dict(
            tracer=tracer, realization_ids=list(range(25)), conditions={}, effective_redshifts={}
        )
        for tier in precision.TIERS:
            prep["conditions"][tier] = {
                cap: dict(
                    counts={s["name"]: int(volume * 1e-4) for s in spec["measurement_bins"]},
                    volumes_mpc_h3={s["name"]: volume for s in spec["measurement_bins"]},
                )
                for cap, volume in (("NGC", 1e9), ("SGC", 2e9))
            }
            prep["effective_redshifts"][tier] = {
                s["name"]: dict(z_eff=(s["zmin"] + s["zmax"]) / 2) for s in spec["measurement_bins"]
            }
        preparations[tracer] = prep
    reference = dict(
        absolute_veff_plot=dict(
            bins=bins, scenarios=[dict(key="desi_dr1", components_by_bin=components)]
        )
    )
    # Cross the saved JSON boundary before consuming either source.
    source = tmp_path / "fixture.json"
    source.write_text(json.dumps(dict(preparations=preparations, reference=reference)))
    saved = precision.Sources().json(source)
    mocks = precision.mock_rows(saved["preparations"], config)
    dr1 = precision.dr1_rows(precision.dr1_reference(saved["reference"]), config)
    assert len(mocks) == 25 and len(dr1) == 5
    for row in dr1:
        mock = next(
            r
            for r in mocks
            if r["measurement_bin"] == row["measurement_bin"] and r["condition"] == "native"
        )
        assert row["p0_mpc_h3"] == mock["p0_mpc_h3"]
        assert row["volume_mpc_h3"] == mock["volume_mpc_h3"] / 4
        assert row["veff_mpc_h3"] == mock["veff_mpc_h3"] / 4
        assert row["sigma_r_over_r"] == pytest.approx(2 * mock["sigma_r_over_r"])


def test_plot_contains_four_tiers_dr1_and_fractional_cv_without_the_control(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    tiers = ("native", "sigma_0.001", "sigma_0.003", "sigma_0.01", "sigma_0.03", "desi_dr1")
    for tier in tiers:
        for name, z in zip(precision.BINS, (0.15, 0.3, 0.5, 0.7, 0.9)):
            rows.append(
                precision.precision_row(
                    name, tier, z, caps(), 10000.0, "BGS" if name.startswith("BGS") else "LRG"
                )
            )
    from tests.notebook_helpers import notebook_namespace

    namespace = notebook_namespace(
        "forecasts", cells={"setup", "presentation-style", "precision_caption", "precision"}
    )
    namespace["precision_plot_precision"](rows)
    captured = {"figure": namespace["figures"]["sigmaR-over-R-vs-z"]}
    try:
        ax = captured["figure"].axes[0]
        # Four photometric tiers, the DR1 forecast and the cosmic-variance reference; the
        # unsmeared control is not drawn, and nothing is labelled by tracer name.
        assert len(ax.lines) == 6
        assert ax.lines[0].get_label() == r"Mock $\sigma_z/(1+z)=0.001$"
        assert "native" not in " ".join(line.get_label() for line in ax.lines)
        assert "DR1" in ax.lines[-2].get_label()
        np.testing.assert_allclose(
            ax.lines[0].get_ydata(), [r["sigma_r_over_r"] for r in rows[5:10]]
        )
        np.testing.assert_allclose(
            ax.lines[-1].get_ydata(), [r["cosmic_variance_sigma_r_over_r"] for r in rows[10:15]]
        )
        assert ax.lines[0].get_ydata().max() < 0.01
        assert ax.get_ylim()[1] > ax.lines[0].get_ydata().max()
    finally:
        plt.close(captured["figure"])
