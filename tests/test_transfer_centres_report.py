"""The review report of transfer damping-centre candidates."""

import pytest
import yaml

from pzbao.config import load_analysis_config
from pzbao.diagnostics import transfer_centres as report


def record(bin_name, tracer, source, target, perp, parallel, *, railed=(), analytic=None):
    diagnostics = {
        "compression_chi2": 1e-5,
        "rms_residual_over_sigma_by_multipole": {"ell0": 1e-3, "ell2": 2e-3},
        "curvature_errors_mpc_h": {"sigma_perp": 4.0, "sigma_parallel": 15.0},
        "fitted_b1": 1.49,
        "damping_above_one": {"fraction_of_template_modes": 0.0},
        "bounds_mpc_h": [0.0, 30.0],
        "railed": list(railed),
    }
    if analytic is not None:
        diagnostics["self_field_analytic_sigma_perp_mpc_h"] = analytic
    return {
        "context": {
            "tracer": tracer,
            "measurement_bin": bin_name,
            "source_class": source,
            "target_class": target,
            "target_redshift": {"z_eff": {"BGS1": 0.16, "BGS2": 0.32}.get(bin_name, 0.7)},
            "bias": 1.5,
        },
        "centers_mpc_h": {"sigma_perp": perp, "sigma_parallel": parallel, "sigma_streaming": 2.0},
        "evidence": {"kind": "source_target_theory", "diagnostics": diagnostics},
    }


def candidate():
    return {
        "units": "Mpc/h",
        "review_reference": None,
        "producer": {"run_name": "full-v3--self--R25", "code_commit": "c" * 40},
        "decision": "2026-10-13 — Transfer damping centres by predicted-multipole compression",
        "records": [
            record("BGS1", "BGS_BRIGHT-21.5", "sigma_0.003", "sigma_0.003", 5.5, 8.9, analytic=5.5),
            record(
                "BGS2",
                "BGS_BRIGHT-21.5",
                "sigma_0.03",
                "sigma_0.03",
                5.8,
                30.0,
                railed=["sigma_parallel"],
            ),
            record("LRG1", "LRG", "sigma_0.03", "sigma_0.03", 5.2, 2.9),
            record("LRG2", "LRG", "sigma_0.03", "sigma_0.03", 4.8, 9.6),
            record("LRG3", "LRG", "sigma_0.03", "sigma_0.03", 4.4, 9.1),
        ],
    }


def test_rows_flag_railing_outliers_and_inverted_damping_but_not_prior_dominance():
    rows = report.record_rows(candidate(), load_analysis_config(), "full-v3--self--R25")
    flags = {row["bin"]: row["flags"] for row in rows}
    assert flags["BGS1"] == [] and flags["LRG2"] == [] and flags["LRG3"] == []
    assert flags["BGS2"][0] == "σ∥ railed" and "outlier" in flags["BGS2"][-1]
    assert "σ∥ < σ⊥" in flags["LRG1"] and any("outlier" in f for f in flags["LRG1"])
    widths = load_analysis_config()["fitting"]["damping"]["gaussian_widths_mpc_h"]
    # The compression curvature error over the configured prior width, shown, never a flag.
    assert rows[0]["sigma_parallel_error_over_width"] == 15.0 / widths["sigma_parallel"]
    assert rows[0]["analytic_sigma_perp"] == 5.5 and rows[1]["analytic_sigma_perp"] is None


def test_report_renders_once_per_candidate_and_keeps_notes(tmp_path):
    path = tmp_path / "self--R25.candidate.yaml"
    path.write_text(yaml.safe_dump(candidate(), allow_unicode=True))
    other = tmp_path / "self--R15.candidate.yaml"
    other.write_text(yaml.safe_dump(candidate(), allow_unicode=True))
    destination = report.report_directory(tmp_path / "reports", "full-v3--self--R25", path)
    assert destination.name == f"centres-full-v3--self--R25--{report.candidate_digest(path)[:8]}"
    config = load_analysis_config()
    page = report.render_centres_report(
        path, config, destination, run_name="full-v3--self--R25", companions={"R15": other}
    )
    text = page.read_text()
    assert "Flagged records (2 of 5)" in text and report.candidate_digest(path) in text
    for name in ("centre-ladders", "constraint-strength", "consistency", "compression-health"):
        assert (destination / f"{name}.png").is_file() and (destination / f"{name}.pdf").is_file()
    assert (destination / "records.csv").read_text().count("\n") == 6
    (destination / "notes.md").write_text("note: BGS2 looks railed\n")
    report.render_centres_report(path, config, destination, run_name="full-v3--self--R25")
    assert (destination / "notes.md").read_text() == "note: BGS2 looks railed\n"
    frozen = tmp_path / "frozen.yaml"
    frozen.write_text(yaml.safe_dump({**candidate(), "review_reference": "x"}, allow_unicode=True))
    with pytest.raises(ValueError, match="already frozen"):
        report.render_centres_report(frozen, config, tmp_path / "f", run_name="x")


def test_native_diagnostics_render_per_record_and_against_the_data(tmp_path):
    import numpy as np

    path = tmp_path / "self--R25.candidate.yaml"
    path.write_text(yaml.safe_dump(candidate(), allow_unicode=True))
    native = tmp_path / "native"
    native.mkdir()
    s = np.linspace(60.0, 150.0, 10)
    shape = np.concatenate(
        [np.exp(-(((s - 100) / 10) ** 2)), 0.3 * np.exp(-(((s - 100) / 10) ** 2))]
    )
    grid = np.linspace(0.0, 30.0, 7)
    surface = np.add.outer((grid - 5.0) ** 2, (grid - 9.0) ** 2) / 10.0
    np.savez(
        native / f"{report.record_key(candidate()['records'][3]['context'])}.npz",
        separations=s,
        multipoles=np.array([0, 2]),
        sigma=np.full(20, 0.01),
        prediction=shape,
        model_prediction=shape * 1.001,
        measured=shape + 0.005,
        model_data_at_centre=shape,
        model_data_free=shape,
        grid=grid,
        dchi2_prediction=surface,
        dchi2_data=surface + 0.5,
        recomputed_centre=np.array([4.8, 9.6]),
        data_free_centre=np.array([5.0, 9.0]),
        data_free_errors=np.array([2.0, 6.0]),
        data_b1=np.array([2.0, 2.0]),
    )
    destination = tmp_path / "report"
    page = report.render_centres_report(
        path, load_analysis_config(), destination, run_name="r", diagnostics_dir=native
    )
    text = page.read_text()
    assert "Native diagnostics: 1 of 5 records" in text and "provenance unrecorded" in text
    assert "consistency check, never a way to choose" in text
    assert (destination / "data-vs-centres.png").is_file()
    assert (destination / "record--LRG2--sigma_0p03--sigma_0p03.png").is_file()
    # Diagnostics stamped with another candidate's digest are refused, not drawn.
    arrays = dict(np.load(native / "LRG2--sigma_0.03--sigma_0.03.npz"))
    stamp = {"code_commit": np.asarray("c"), "candidate_sha256": np.asarray("0" * 64)}
    np.savez(native / "LRG2--sigma_0.03--sigma_0.03.npz", **arrays, **stamp, points=7)
    with pytest.raises(ValueError, match="another candidate"):
        report.render_centres_report(
            path, load_analysis_config(), destination, run_name="r", diagnostics_dir=native
        )


def test_missing_curvature_error_is_flagged_and_still_renders(tmp_path):
    artifact = candidate()
    artifact["records"][0]["evidence"]["diagnostics"]["curvature_errors_mpc_h"]["sigma_perp"] = None
    rows = report.record_rows(artifact, load_analysis_config(), "r")
    assert "σ⊥ no curvature error" in rows[0]["flags"]
    path = tmp_path / "c.candidate.yaml"
    path.write_text(yaml.safe_dump(artifact, allow_unicode=True))
    page = report.render_centres_report(
        path, load_analysis_config(), tmp_path / "out", run_name="r"
    )
    assert "Flagged records (3 of 5)" in page.read_text()


def test_native_job_marks_failed_records_and_refuses_mixed_provenance(tmp_path, monkeypatch):
    import numpy as np

    path = tmp_path / "c.candidate.yaml"
    path.write_text(yaml.safe_dump(candidate(), allow_unicode=True))

    def diagnostics(run, record, points):
        if record["context"]["measurement_bin"] == "BGS2":
            raise RuntimeError("profile did not converge")
        return {"grid": np.linspace(0.0, 30.0, points)}

    monkeypatch.setattr(report, "centre_diagnostics", diagnostics)
    monkeypatch.setattr(report, "code_commit", lambda: "a" * 40)
    failures = report.run_diagnostics({}, path, tmp_path / "native", points=5)
    assert list(failures) == ["BGS2--sigma_0.03--sigma_0.03"]
    marker = yaml.safe_load(
        (tmp_path / "native/BGS2--sigma_0.03--sigma_0.03.failed.txt").read_text()
    )
    assert marker["error"] == "RuntimeError: profile did not converge"
    assert marker["candidate_sha256"] == report.candidate_digest(path)
    finished = sorted(p.name for p in (tmp_path / "native").glob("*.npz"))
    assert len(finished) == 4 and not any("partial" in name for name in finished)
    with np.load(tmp_path / "native/LRG2--sigma_0.03--sigma_0.03.npz") as archive:
        assert str(archive["candidate_sha256"]) == report.candidate_digest(path)
        assert str(archive["code_commit"]) == "a" * 40 and int(archive["points"]) == 5
    # A resume after a code change refuses before computing anything, even the failed record.
    monkeypatch.setattr(report, "code_commit", lambda: "b" * 40)
    monkeypatch.setattr(report, "centre_diagnostics", lambda run, record, points: {"grid": 0})
    with pytest.raises(ValueError, match="another provenance"):
        report.run_diagnostics({}, path, tmp_path / "native", points=5)
    assert not (tmp_path / "native/BGS2--sigma_0.03--sigma_0.03.npz").exists()

    def mismatch(run, record, points):
        raise report.RecordMismatch("the candidate record does not describe this run's route")

    monkeypatch.setattr(report, "centre_diagnostics", mismatch)
    with pytest.raises(report.RecordMismatch):
        report.run_diagnostics({}, path, tmp_path / "other", points=5)


def test_native_diagnostics_refuse_a_record_of_another_route(monkeypatch):
    from pzbao import likelihood

    record_ = candidate()["records"][0]
    other = {**record_["context"], "measurement_bin": "BGS2"}
    monkeypatch.setattr(
        likelihood, "transfer_compression_setup", lambda run, route: {"context": other}
    )
    with pytest.raises(ValueError, match="does not describe this run's route"):
        report.centre_diagnostics({"config": {}}, record_)


def test_native_diagnostics_refuse_a_proxy_covariance_record(monkeypatch):
    from pzbao import likelihood

    record_ = candidate()["records"][0]
    record_["evidence"]["diagnostics"]["covariance_source"] = {"proxy": True}
    setup = {"context": record_["context"], "holder": {"metadata": {}}}
    monkeypatch.setattr(likelihood, "transfer_compression_setup", lambda run, route: setup)
    with pytest.raises(report.RecordMismatch, match="proxy covariance"):
        report.centre_diagnostics({"config": {}}, record_)
    record_["evidence"]["diagnostics"]["covariance_source"] = {"run_name": "other"}
    setup["holder"]["metadata"] = {"run_name": "this"}
    with pytest.raises(report.RecordMismatch, match="differs in run_name"):
        report.centre_diagnostics({"config": {}}, record_)


def test_report_reads_failures_bounded_b1_and_a_short_free_fit(tmp_path, monkeypatch):
    import numpy as np

    path = tmp_path / "c.candidate.yaml"
    path.write_text(yaml.safe_dump(candidate(), allow_unicode=True))
    native = tmp_path / "native"
    native.mkdir()
    stamp = {
        "code_commit": "a" * 40,
        "candidate_sha256": report.candidate_digest(path),
        "points": 5,
    }
    (native / "BGS2--sigma_0.03--sigma_0.03.failed.txt").write_text(
        yaml.safe_dump({"error": "RuntimeError: no covariance\nsecond line", **stamp})
    )
    config = load_analysis_config()
    page = report.render_centres_report(
        path, config, tmp_path / "a", run_name="r", diagnostics_dir=native
    )
    text = page.read_text()
    assert "No record finished" in text and "RuntimeError: no covariance" in text
    assert "Largest" not in text and "provenance unrecorded" not in text
    s = np.linspace(60.0, 150.0, 10)
    grid = np.linspace(0.0, 30.0, 5)
    np.savez(
        native / "LRG2--sigma_0.03--sigma_0.03.npz",
        separations=s,
        multipoles=np.array([0, 2]),
        sigma=np.full(20, 0.01),
        prediction=np.ones(20),
        model_prediction=np.ones(20),
        measured=np.ones(20),
        model_data_at_centre=np.ones(20),
        model_data_free=np.ones(20),
        grid=grid,
        dchi2_prediction=np.zeros((5, 5)),
        dchi2_data=np.zeros((5, 5)),
        dchi2_prediction_points=np.array([0.0, 0.25]),
        dchi2_data_points=np.array([1.5, 0.0]),
        slice_b1=np.array([1.5, 0.2]),
        dchi2_data_profiled=np.array(-0.01),
        recomputed_centre=np.array([4.8, 9.6]),
        data_free_centre=np.array([5.0, 9.0]),
        data_free_errors=np.array([2.0, np.nan]),
        data_b1=np.array([0.2, 0.2]),
        **{k: np.asarray(v) for k, v in stamp.items()},
    )
    text = report.render_centres_report(
        path, config, tmp_path / "b", run_name="r", diagnostics_dir=native
    ).read_text()
    assert "0.200 (at its prior bound)" in text and "free fit short of its optimum" in text
    assert "(no error)" in text and "0.25" in text and "beyond 0.1 Mpc/h are flagged" in text
