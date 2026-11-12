"""Review report for transfer damping-centre candidates, read before they are frozen.

Transfer damping centres come from a predicted-multipole compression. Freezing them is a
separate, reviewed step, because weakly constrained centres must not be frozen as if
measured; every freeze gets this HTML report. Layer 1 reads the candidate artifacts and the
run configuration only. Layer 2 (``centre_diagnostics``, run inside a compute allocation)
recomputes each record's compression and keeps the arrays the
producer discards. Nothing here changes a candidate or freezes a centre.
"""

from __future__ import annotations

import hashlib
import html
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

PARAMETERS = ("sigma_perp", "sigma_parallel")
SYMBOL = {"sigma_perp": "σ⊥", "sigma_parallel": "σ∥"}
# The compression fits a noiseless prediction, so its curvature error is the width the data's
# mean covariance would allow, not an uncertainty of the centre: above one prior width the fit's
# damping follows the prior centre. Drawn for orientation only; it does not decide which
# centres to freeze.
PRIOR_DOMINATED_WIDTHS = 1.0
# The diagnostics recompute each centre with the producer's own setup. Along a flat direction
# the optimizer's near-tied starts can land a few tenths apart (2026-10-19 preview: 32 of 50
# records within 1e-4 Mpc/h; the rest moved mostly in sigma_parallel, whose curvature errors
# there are 15-26 Mpc/h, and by at most 0.015 Mpc/h in sigma_perp).
# Chosen after that preview: beyond a tenth of the smallest prior width (sigma_perp, 1 Mpc/h)
# the centre is not reproducible as frozen.
RECOMPUTED_TOLERANCE_MPC_H = 0.1
PROVENANCE = ("code_commit", "candidate_sha256", "points")


class RecordMismatch(ValueError):
    """The candidate record and the run it is diagnosed against do not belong together."""


def candidate_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def report_directory(root, run_name, candidate_path):
    """`centres-<run>--<candidate sha256[:8]>`, so a re-run candidate gets its own report."""
    return Path(root) / f"centres-{run_name}--{candidate_digest(candidate_path)[:8]}"


def code_commit():
    """HEAD of this checkout; `-dirty-<digest>` when its package differs, `unknown` outside git."""
    repository = Path(__file__).resolve().parents[2]

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(repository), *args], text=True, stderr=subprocess.DEVNULL
        )

    try:
        head = git("rev-parse", "HEAD").strip()
        changes = git("diff", "HEAD", "--", "pzbao") + git(
            "ls-files", "--others", "--exclude-standard", "--", "pzbao"
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    if not changes:
        return head
    return f"{head}-dirty-{hashlib.sha256(changes.encode()).hexdigest()[:8]}"


def _sigma(label):
    return 0.0 if label == "native" else float(label.split("_", 1)[1])


def record_rows(candidate, config, run_name):
    """One row per candidate record with its centres, constraint strength and flags."""
    damping = config["fitting"]["damping"]
    widths = damping["gaussian_widths_mpc_h"]
    rows = []
    for record in candidate["records"]:
        context, centres = record["context"], record["centers_mpc_h"]
        diagnostics = record["evidence"]["diagnostics"]
        baseline = damping["baseline_centers_mpc_h"][context["tracer"]]["postreconstruction"]
        row = {
            "run": run_name,
            "tracer": context["tracer"],
            "bin": context["measurement_bin"],
            "source": context["source_class"],
            "target": context["target_class"],
            "z_eff": float(context["target_redshift"]["z_eff"]),
            "bias": float(context["bias"]),
            "fitted_b1": float(diagnostics["fitted_b1"]),
            "compression_chi2": float(diagnostics["compression_chi2"]),
            "rms_ell0": float(diagnostics["rms_residual_over_sigma_by_multipole"]["ell0"]),
            "rms_ell2": float(diagnostics["rms_residual_over_sigma_by_multipole"]["ell2"]),
            "damping_above_one": float(
                diagnostics["damping_above_one"]["fraction_of_template_modes"]
            ),
            "analytic_sigma_perp": diagnostics.get("self_field_analytic_sigma_perp_mpc_h"),
            "sigma_streaming": float(centres["sigma_streaming"]),
        }
        railed = set(diagnostics.get("railed") or [])
        flags = []
        for name in PARAMETERS:
            value = float(centres[name])
            # The producer writes None where Minuit reports no error.
            raw = diagnostics["curvature_errors_mpc_h"].get(name)
            error = float("nan") if raw is None else float(raw)
            bounds = diagnostics["bounds_mpc_h"]  # one shared [lower, upper] or one per parameter
            lower, upper = (
                float(edge) for edge in (bounds[name] if isinstance(bounds, dict) else bounds)
            )
            at_bound = min(abs(value - lower), abs(upper - value)) < 1e-2
            row.update(
                {
                    name: value,
                    f"{name}_error": error,
                    f"{name}_error_over_width": error / float(widths[name]),
                    f"{name}_baseline": float(baseline[name]),
                    f"{name}_width": float(widths[name]),
                    f"{name}_bounds": [lower, upper],
                }
            )
            if name in railed:
                flags.append(f"{SYMBOL[name]} railed")
            elif at_bound:
                flags.append(f"{SYMBOL[name]} at bound")
            if raw is None:
                flags.append(f"{SYMBOL[name]} no curvature error")
        if row["sigma_parallel"] < row["sigma_perp"]:
            flags.append("σ∥ < σ⊥")
        row["flags"] = flags
        rows.append(row)
    _flag_neighbour_outliers(rows)
    return rows


def _flag_neighbour_outliers(rows):
    """A centre far from the median of the other bins at the same source and target.

    The other bins span both tracers and include railed ones; the median keeps one railed
    neighbour from moving the reference much, and the flag only points the reader at a record.
    """
    for name in PARAMETERS:
        for row in rows:
            others = [
                other[name]
                for other in rows
                if other is not row
                and other["source"] == row["source"]
                and other["target"] == row["target"]
            ]
            if len(others) < 2:
                continue
            median = float(np.median(others))
            if abs(row[name] - median) > max(2.0, 0.5 * median):
                row["flags"].append(f"{SYMBOL[name]} outlier vs other bins ({median:.1f})")


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save(figure, destination):
    from ..figures.style import save_figure

    save_figure(figure, destination)
    _plt().close(figure)
    return destination.with_suffix(".png").name, destination.with_suffix(".pdf").name


def _color(target):
    from ..figures.style import sigma_color

    return sigma_color(_sigma(target))


def _clipped(value, error, lower, upper):
    """Error bar halves clipped to the bounds; a missing error draws no bar."""
    error = np.nan_to_num(np.asarray(error, dtype=float), nan=0.0)
    value = np.asarray(value, dtype=float)
    return np.minimum(error, value - lower), np.minimum(error, upper - value)


def plot_ladders(rows, destination):
    """Centres against effective redshift with curvature errors, prior widths and references."""
    plt = _plt()
    figure, panels = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for axis, name in zip(panels, PARAMETERS):
        lower, upper = rows[0][f"{name}_bounds"]
        axis.axhspan(lower, upper, color="0.95", zorder=0)
        for edge in (lower, upper):
            axis.axhline(edge, color="0.5", lw=0.8, ls=":")
        for tracer in sorted({row["tracer"] for row in rows}):
            members = [row for row in rows if row["tracer"] == tracer]
            span = [
                min(r["z_eff"] for r in members) - 0.03,
                max(r["z_eff"] for r in members) + 0.03,
            ]
            axis.plot(span, [members[0][f"{name}_baseline"]] * 2, color="0.3", ls="--", lw=1)
        targets = sorted({row["target"] for row in rows}, key=_sigma)
        for offset, target in zip(np.linspace(-0.012, 0.012, len(targets)), targets):
            members = [row for row in rows if row["target"] == target]
            z = np.array([row["z_eff"] for row in members]) + offset
            value = np.array([row[name] for row in members])
            error = [row[f"{name}_error"] for row in members]
            label = f"{members[0]['source']} → {target}"
            axis.errorbar(
                z,
                value,
                yerr=_clipped(value, error, lower, upper),
                fmt="o",
                color=_color(target),
                ms=5,
                capsize=2,
                label=label,
            )
            width = members[0][f"{name}_width"]
            axis.errorbar(z, value, yerr=width, fmt="none", color=_color(target), lw=6, alpha=0.25)
            for row, x in zip(members, z):
                if any(flag.startswith(SYMBOL[name]) for flag in row["flags"]):
                    axis.plot(x, row[name], marker="x", color="crimson", ms=11, mew=2)
        if name == "sigma_perp":
            analytic = [row for row in rows if row["analytic_sigma_perp"] is not None]
            if analytic:
                axis.plot(
                    [row["z_eff"] for row in analytic],
                    [row["analytic_sigma_perp"] for row in analytic],
                    "s",
                    mfc="none",
                    color="k",
                    ms=7,
                    label="self-field analytic σ⊥",
                )
        axis.set_xlabel("target z_eff")
        axis.set_ylabel(f"{SYMBOL[name]} centre [Mpc/h]")
        axis.set_ylim(lower - 1, upper + 1)
    panels[0].legend(fontsize=7, loc="upper left")
    figure.suptitle(
        "Candidate centres: thin bars = compression curvature error (clipped to the bounds), "
        "wide bars = ±prior width the fit would impose,\ndashed = baseline post-reconstruction "
        "centre, red × = that panel's flag (railed / at bound / no error / outlier; σ∥ < σ⊥ in "
        "the σ∥ panel), grey band = allowed [0, 30]",
        fontsize=8,
    )
    return _save(figure, destination)


def plot_constraint(rows, destination):
    """Curvature error over prior width: above one, the fit's damping follows the prior centre."""
    plt = _plt()
    figure, axis = plt.subplots(figsize=(11, 3.8))
    labels = [f"{row['bin']}\n{row['source'][6:]}→{row['target'][6:]}" for row in rows]
    x = np.arange(len(rows))
    for shift, name, marker in ((-0.15, "sigma_perp", "o"), (0.15, "sigma_parallel", "D")):
        axis.scatter(
            x + shift,
            [row[f"{name}_error_over_width"] for row in rows],
            c=[_color(row["target"]) for row in rows],
            marker=marker,
            label=SYMBOL[name],
            edgecolor="k",
            lw=0.4,
        )
    axis.axhline(PRIOR_DOMINATED_WIDTHS, color="crimson", lw=0.8, ls="--", label="one prior width")
    axis.set_yscale("log")
    axis.set_xticks(x, labels, fontsize=6)
    axis.set_ylabel("curvature error / prior width")
    axis.legend(fontsize=7)
    axis.set_title(
        "Data width over prior width: above the line the fit's damping follows the prior centre, "
        "so the centre itself, not the data, sets it",
        fontsize=9,
    )
    return _save(figure, destination)


def plot_consistency(rows, companions, destination):
    """The same target bin seen from other runs: self against transfer source, R15 against R25."""
    plt = _plt()
    pairs = []
    for other_name, other_rows in companions.items():
        index = {(r["bin"], r["target"]): r for r in other_rows}
        matched = [
            (row, index[(row["bin"], row["target"])])
            for row in rows
            if (row["bin"], row["target"]) in index
        ]
        if matched:
            pairs.append((other_name, matched))
    if not pairs:
        return None
    figure, panels = plt.subplots(len(pairs), 2, figsize=(9, 3.6 * len(pairs)), squeeze=False)
    run = rows[0]["run"]
    for (other_name, matched), axes_row in zip(pairs, panels):
        for axis, name in zip(axes_row, PARAMETERS):
            # Curvature errors are drawn in the ladders; here only positions, against the 1:1
            # line and its ±1 prior-width band, so a disagreement the fit would feel stands out.
            width = rows[0][f"{name}_width"]
            values = [v for row, other in matched for v in (row[name], other[name])]
            edge = (0.0, max(values) + 2.0)
            line = np.array(edge)
            axis.fill_between(line, line - width, line + width, color="0.9")
            axis.plot(line, line, color="0.5", lw=0.8)
            for row, other in matched:
                axis.plot(other[name], row[name], "o", color=_color(row["target"]), ms=6)
                axis.annotate(row["bin"], (other[name], row[name]), fontsize=7)
            axis.set_xlim(*edge)
            axis.set_ylim(*edge)
            axis.set_xlabel(f"{SYMBOL[name]} in {other_name}")
            axis.set_ylabel(f"{SYMBOL[name]} in {run}")
    figure.suptitle("Same bin and target class in other runs (1:1 line)", fontsize=9)
    return _save(figure, destination)


def plot_health(rows, destination):
    """Compression health: b1 recovery, chi-square, residuals and D > 1."""
    plt = _plt()
    figure, panels = plt.subplots(1, 4, figsize=(13, 3.2))
    x = np.arange(len(rows))
    colors = [_color(row["target"]) for row in rows]
    series = (
        ("fitted b1 / context bias", [row["fitted_b1"] / row["bias"] for row in rows], False),
        ("compression χ²", [row["compression_chi2"] for row in rows], True),
        ("RMS residual / σ (ℓ=0 ●, ℓ=2 ◆)", None, True),
        ("template modes with D > 1", [row["damping_above_one"] for row in rows], False),
    )
    for axis, (title, values, log) in zip(panels, series):
        if values is None:
            axis.scatter(x, [row["rms_ell0"] for row in rows], c=colors, marker="o")
            axis.scatter(x, [row["rms_ell2"] for row in rows], c=colors, marker="D")
        else:
            axis.scatter(x, values, c=colors)
        if log:
            axis.set_yscale("log")
        axis.set_title(title, fontsize=8)
        axis.set_xticks(x, [row["bin"] for row in rows], fontsize=5, rotation=90)
    panels[0].axhline(1, color="0.4", lw=0.8)
    return _save(figure, destination)


# ---------------------------------------------------------------- layer 2: native diagnostics


def _loglikelihood(value):
    # Solved nuisances add a trailing derivatives axis; column 0 is the solution's value.
    return float(np.asarray(value, dtype=float).reshape(1, -1)[0, 0])


def _route(context):
    return {
        "tracer": context["tracer"],
        "measurement_bin": context["measurement_bin"],
        "condition": context["target_class"],
        "reconstruction_state": "postreconstruction",
    }


def centre_diagnostics(run, record, *, points=13):
    """Arrays a centre review needs that the producer discards, for one candidate record.

    Recomputes the compression with the producer's own setup, then adds the template at the
    centre against the predicted and the measured mean-of-25 multipoles, the measured mean's own
    preferred damping (dilations at one, b1 and broadband free), the profiled Delta chi-square of
    the measured mean between the candidate centre and that preference, and conditional Delta
    chi-square slices over (sigma_perp, sigma_parallel): b1 fixed, broadband solved at its best.
    """
    from ..config import same_identity
    from ..fitting import maximize_usable
    from ..likelihood import (
        TRANSFER_CENTRE_BOUNDS_MPC_H,
        _fix_parameters,
        _predict_and_compress,
        build_likelihood,
        transfer_compression_setup,
    )

    setup = transfer_compression_setup(run, _route(record["context"]))
    if not same_identity(setup["context"], record["context"]):
        raise RecordMismatch("the candidate record does not describe this run's route")
    source = record["evidence"]["diagnostics"].get("covariance_source") or {}
    if source.get("proxy"):
        # The setup would read this run's own covariance and window, not the proxy's.
        raise RecordMismatch("the record's centre was compressed with a proxy covariance")
    metadata = setup["holder"]["metadata"]
    differing = [
        key
        for key in ("run_name", "code_commit", "analysis_config_sha256")
        if key in source and source[key] != metadata[key]
    ]
    if differing:
        raise RecordMismatch(f"the record's covariance source differs in {', '.join(differing)}")
    config, common, arrays = setup["config"], setup["common"], setup["arrays"]
    prediction, model, summary = _predict_and_compress(
        common,
        setup["prediction_values"],
        damping=setup["damping"],
        streaming=setup["streaming"],
        config=config,
    )
    fixed = {k: v for k, v in setup["prediction_values"].items() if k not in ("b1", "sigmas")}
    fixed["sigmas"] = setup["streaming"]
    centres = record["centers_mpc_h"]
    candidate = (float(centres["sigma_perp"]), float(centres["sigma_parallel"]))
    recomputed = (summary["fitted"]["sigmaper"], summary["fitted"]["sigmapar"])
    patterns = list(config["fitting"]["linear_nuisance"]["parameter_patterns"])
    seed = int(config["fitting"]["optimization"]["random_seed"])

    def fit(data, pins):
        # Imported here so the identity checks above run without desilike (CI has none).
        from desilike.profilers import MinuitProfiler

        _, observable, likelihood, _ = build_likelihood(data, **common)
        _fix_parameters(likelihood, {**fixed, **pins})
        best = maximize_usable(MinuitProfiler(likelihood, seed=seed), config)
        free = [n for n in ("sigmaper", "sigmapar", "b1") if n not in pins]
        values = {name: float(best.bestfit[name][0]) for name in free}
        errors = {}
        for name in free:
            try:
                errors[name] = float(best.error[name][0])
            except (AttributeError, KeyError, IndexError, TypeError):
                errors[name] = float("nan")
        loglikelihood = _loglikelihood(best.bestfit["loglikelihood"])
        broadband = [
            str(p.name) for p in likelihood.all_params.select(basename=patterns) if not p.fixed
        ]
        _fix_parameters(likelihood, {name: float(best.bestfit[name][0]) for name in broadband})
        likelihood(**values)
        return values, errors, loglikelihood, np.array(observable.flattheory, dtype=float)

    measured = np.asarray(arrays["vectors"], dtype=float).mean(axis=0)
    pinned = {"sigmaper": candidate[0], "sigmapar": candidate[1]}
    data_at_centre, _, data_centre_loglikelihood, data_model = fit(measured, pinned)
    data_free, data_errors, data_free_loglikelihood, data_free_model = fit(measured, {})
    data_best = (data_free["sigmaper"], data_free["sigmapar"])
    lower, upper = TRANSFER_CENTRE_BOUNDS_MPC_H
    grid = np.linspace(lower, upper, points)

    def surface(data, b1, extra):
        """-2 log L on the grid and at ``extra`` points, b1 fixed, from the lowest of them."""
        _, _, likelihood, _ = build_likelihood(data, **common)
        # Profile the solved broadband, as the profiler does for the compression and the fits.
        likelihood.solved_default = ".best"
        _fix_parameters(likelihood, {**fixed, "b1": b1})

        def value(a, b):
            return -2.0 * _loglikelihood(likelihood(sigmaper=a, sigmapar=b))

        values = np.array([[value(a, b) for b in grid] for a in grid])
        points_ = np.array([value(a, b) for a, b in extra])
        floor = min(values.min(), points_.min())
        return values - floor, points_ - floor

    prediction_map, prediction_points = surface(
        prediction, summary["fitted"]["b1"], (recomputed, candidate)
    )
    data_map, data_points = surface(measured, data_at_centre["b1"], (candidate, data_best))
    return {
        "separations": np.asarray(arrays["separations"], dtype=float).mean(axis=0),
        "multipoles": np.asarray(arrays["multipoles"]),
        "sigma": np.sqrt(np.diag(np.asarray(arrays["mean_covariance"], dtype=float))),
        "prediction": prediction,
        "model_prediction": model,
        "measured": measured,
        "model_data_at_centre": data_model,
        "model_data_free": data_free_model,
        "grid": grid,
        "dchi2_prediction": prediction_map,
        "dchi2_data": data_map,
        # Slice values at (recomputed centre, candidate) and at (candidate, data best).
        "dchi2_prediction_points": prediction_points,
        "dchi2_data_points": data_points,
        "slice_b1": np.array([summary["fitted"]["b1"], data_at_centre["b1"]]),
        # Both fits profile b1 and the broadband: the measured mean's cost of the candidate.
        "dchi2_data_profiled": np.array(
            -2.0 * (data_centre_loglikelihood - data_free_loglikelihood)
        ),
        "recomputed_centre": np.array(recomputed),
        "data_free_centre": np.array(data_best),
        "data_free_errors": np.array([data_errors["sigmaper"], data_errors["sigmapar"]]),
        "data_b1": np.array([data_at_centre["b1"], data_free["b1"]]),
    }


def record_key(context):
    return f"{context['measurement_bin']}--{context['source_class']}--{context['target_class']}"


def _provenance(archive):
    return {key: archive[key].item() for key in PROVENANCE if key in archive.files}


def run_diagnostics(run, candidate_path, output_dir, *, points=13):
    """Native job body: one .npz per candidate record, created once; skips finished records.

    A record that fails writes ``<record>.failed.txt`` and the others still run; the failures
    are returned. Finished files of another candidate, code commit or grid are refused before
    anything is computed, so a resume never mixes arrays of different provenance.
    """
    candidate = yaml.safe_load(Path(candidate_path).read_text())
    provenance = {
        "code_commit": code_commit(),
        "candidate_sha256": candidate_digest(candidate_path),
        "points": int(points),
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stale = []
    for record in candidate["records"]:
        path = output_dir / f"{record_key(record['context'])}.npz"
        if path.exists():
            with np.load(path) as archive:
                found = _provenance(archive)
            if found != provenance:
                stale.append(f"{path.name} ({found or 'no provenance'})")
    if stale:
        raise ValueError(
            f"{output_dir} holds diagnostics of another provenance than {provenance}: "
            f"{', '.join(stale)}; render them without --native, or remove the directory by "
            "hand before recomputing"
        )
    failures = {}
    for record in candidate["records"]:
        key = record_key(record["context"])
        path, failed = output_dir / f"{key}.npz", output_dir / f"{key}.failed.txt"
        if path.exists():
            continue
        try:
            arrays = centre_diagnostics(run, record, points=points)
        except RecordMismatch:
            raise
        except Exception as error:  # one failing record must not hide the others
            failures[key] = f"{type(error).__name__}: {error}"
            partial = failed.with_name(failed.name + ".partial")
            partial.write_text(yaml.safe_dump({"error": failures[key], **provenance}))
            partial.replace(failed)
            continue
        partial = path.with_suffix(".partial.npz")
        np.savez(partial, **arrays, **{k: np.asarray(v) for k, v in provenance.items()})
        partial.replace(path)
        failed.unlink(missing_ok=True)
    return failures


def _diagnostics(candidate, digest, directory):
    """Finished arrays, failures and the one provenance of a native directory."""
    if directory is None or not Path(directory).is_dir():
        return {}, {}, None
    found, failures, provenances = {}, {}, set()
    for record in candidate["records"]:
        key = record_key(record["context"])
        path = Path(directory) / f"{key}.npz"
        if path.exists():
            with np.load(path) as archive:
                found[key] = {k: archive[k] for k in archive.files}
                provenances.add(tuple(sorted(_provenance(archive).items())))
        elif (Path(directory) / f"{key}.failed.txt").exists():
            marker = yaml.safe_load((Path(directory) / f"{key}.failed.txt").read_text())
            failures[key] = str(marker["error"])
            provenances.add(tuple(sorted((k, marker[k]) for k in PROVENANCE if k in marker)))
    if len(provenances) > 1:
        raise ValueError(f"{directory} mixes diagnostics of different provenance")
    provenance = dict(provenances.pop()) if provenances else None
    if provenance and provenance.get("candidate_sha256") != digest:
        raise ValueError(f"{directory} holds diagnostics of another candidate")
    return found, failures, provenance


def _within_one_error_of_bound(value, error, lower, upper):
    return not np.isfinite(error) or min(value - lower, upper - value) < error


def plot_data_against_centres(rows, keys, diagnostics, destination):
    """Candidate centre against the measured mean's own preferred damping, per record."""
    plt = _plt()
    figure, panels = plt.subplots(1, 2, figsize=(11, 4.2))
    for axis, (index, name) in zip(panels, enumerate(PARAMETERS)):
        lower, upper = rows[0][f"{name}_bounds"]
        axis.plot([lower, upper], [lower, upper], color="0.5", lw=0.8)
        width = rows[0][f"{name}_width"]
        line = np.array([lower, upper])
        axis.fill_between(line, line - width, line + width, color="0.9")
        for row, key in zip(rows, keys):
            if key not in diagnostics:
                continue
            item = diagnostics[key]
            value = float(item["data_free_centre"][index])
            error = float(item["data_free_errors"][index])
            # Near zero the damping depends on σ², so the curvature error there means little.
            loose = _within_one_error_of_bound(value, error, lower, upper)
            axis.errorbar(
                row[name],
                value,
                yerr=np.reshape(_clipped(value, error, lower, upper), (2, 1)),
                fmt="o",
                color=_color(row["target"]),
                mfc="none" if loose else _color(row["target"]),
                ms=5,
            )
            axis.annotate(row["bin"], (row[name], value), fontsize=6)
        axis.set_xlim(lower - 1, upper + 1)
        axis.set_ylim(lower - 1, upper + 1)
        axis.set_xlabel(f"candidate {SYMBOL[name]} (theory compression)")
        axis.set_ylabel(f"recovery-mock mean preferred {SYMBOL[name]}")
    figure.suptitle(
        "Consistency only, never for choosing a centre: the mean of the 25 recovery mocks, "
        "dilations fixed at 1; grey band = ±prior width;\nbars = curvature error clipped to the "
        "bounds; open = within one error of a bound (not constrained there)",
        fontsize=8,
    )
    return _save(figure, destination)


def plot_record(row, item, destination):
    """One record: multipoles and models, residuals, and the two conditional Δχ² slices."""
    plt = _plt()
    figure, panels = plt.subplots(1, 4, figsize=(17, 3.8))
    s = item["separations"]
    multipoles = [int(ell) for ell in item["multipoles"]]
    blocks = len(multipoles)
    split = lambda vector: np.asarray(vector).reshape(blocks, -1)
    for ell, data, sigma, pred, model, model_data, model_free in zip(
        multipoles,
        split(item["measured"]),
        split(item["sigma"]),
        split(item["prediction"]),
        split(item["model_prediction"]),
        split(item["model_data_at_centre"]),
        split(item["model_data_free"]),
    ):
        axis = panels[0]
        axis.errorbar(s, s**2 * data, yerr=s**2 * sigma, fmt=".", ms=3, label=f"mock mean ℓ={ell}")
        axis.plot(
            s, s**2 * pred, "-", lw=1, label=f"theory prediction (context b1, no broadband) ℓ={ell}"
        )
        axis.plot(s, s**2 * model_data, "--", lw=1, label=f"template at centre, fitted b1 ℓ={ell}")
        axis.plot(s, s**2 * model_free, ":", lw=1, label=f"template at mock-mean best ℓ={ell}")
        panels[1].plot(s, (pred - model) / sigma, "-", label=f"prediction − template ℓ={ell}")
        panels[1].plot(s, (data - model_data) / sigma, ".", label=f"mock mean − template ℓ={ell}")
    panels[0].set_xlabel("s [Mpc/h]")
    panels[0].set_ylabel("s² ξ_ℓ")
    panels[0].legend(fontsize=5)
    panels[1].axhline(0, color="0.5", lw=0.8)
    panels[1].set_ylabel("residual / σ(mean of 25)")
    panels[1].legend(fontsize=5)
    b1 = item.get("slice_b1")
    for axis, key, title in (
        (panels[2], "dchi2_prediction", "prediction"),
        (panels[3], "dchi2_data", "recovery-mock mean"),
    ):
        grid = item["grid"]
        mesh = axis.pcolormesh(grid, grid, np.minimum(item[key].T, 25), shading="auto")
        axis.contour(grid, grid, item[key].T, levels=[2.30, 6.18], colors="w", linewidths=0.8)
        figure.colorbar(mesh, ax=axis)
        axis.plot(row["sigma_perp"], row["sigma_parallel"], "r*", ms=12, label="candidate")
        axis.add_patch(
            plt.Rectangle(
                (
                    row["sigma_perp"] - row["sigma_perp_width"],
                    row["sigma_parallel"] - row["sigma_parallel_width"],
                ),
                2 * row["sigma_perp_width"],
                2 * row["sigma_parallel_width"],
                fill=False,
                ec="r",
                lw=0.8,
            )
        )
        if key == "dchi2_data":
            axis.plot(
                *item["data_free_centre"], "wx", ms=10, mew=2, label="mock-mean best (b1 free)"
            )
        fixed = "" if b1 is None else f" {b1[0 if key == 'dchi2_prediction' else 1]:.3f}"
        axis.set_xlabel("σ⊥ [Mpc/h]")
        axis.set_ylabel("σ∥ [Mpc/h]")
        axis.set_title(f"Δχ² slice, {title}, b1 fixed{fixed}", fontsize=8)
        axis.legend(fontsize=6)
    figure.suptitle(
        f"{row['bin']} {row['source']} → {row['target']}: flags {'; '.join(row['flags']) or 'none'}"
        " — conditional slices (b1 fixed: narrower than profiled regions), white contours at "
        "Δχ² 2.3 and 6.2; red box = ±prior width",
        fontsize=9,
    )
    return _save(figure, destination)


TABLE_COLUMNS = (
    "bin",
    "source",
    "target",
    "z_eff",
    "sigma_perp",
    "sigma_perp_error",
    "sigma_parallel",
    "sigma_parallel_error",
    "analytic_sigma_perp",
    "fitted_b1",
    "compression_chi2",
    "flags",
)
NATIVE_COLUMNS = (
    "bin",
    "source",
    "target",
    "sigma_perp",
    "sigma_parallel",
    "recomputed_difference",
    "prediction_dchi2_at_candidate",
    "mock_mean_sigma_perp",
    "mock_mean_sigma_parallel",
    "mock_mean_b1_at_centre",
    "mock_mean_dchi2_profiled",
    "mock_mean_dchi2_slice_at_candidate",
)


def _table(rows, columns=TABLE_COLUMNS):
    from .fitting import table

    shown = []
    for row in rows:
        shown.append({**row, "flags": "; ".join(row.get("flags", ())) or "—"})
    return table(shown, columns)


def _native_rows(rows, keys, native, b1_limits):
    """Recomputation check and the recovery-mock mean's numbers, one row per finished record."""
    shown = []
    for row, key in zip(rows, keys):
        if key not in native:
            continue
        item = native[key]
        difference = float(
            np.abs(item["recomputed_centre"] - [row["sigma_perp"], row["sigma_parallel"]]).max()
        )
        if difference > RECOMPUTED_TOLERANCE_MPC_H:
            row["flags"].append(f"recomputed centre moves {difference:.2g} Mpc/h")
        best, errors = item["data_free_centre"], item["data_free_errors"]
        mean = {}
        for index, name in enumerate(PARAMETERS):
            lower, upper = row[f"{name}_bounds"]
            text = f"{best[index]:.2f} ± {errors[index]:.2f}"
            if not np.isfinite(errors[index]):
                text += " (no error)"
            elif _within_one_error_of_bound(best[index], errors[index], lower, upper):
                text += " (at a bound)"
            mean[f"mock_mean_{name}"] = text
        # b1 on its prior bound limits both the profiled Delta chi-square and the data slice.
        b1 = float(item["data_b1"][0])
        b1_text = f"{b1:.3f}"
        if min(abs(b1 - b1_limits[0]), abs(b1_limits[1] - b1)) < 1e-3:
            b1_text += " (at its prior bound)"
        profiled = None
        if "dchi2_data_profiled" in item:
            value = float(item["dchi2_data_profiled"])
            # The free fit contains the pinned one, so below zero it stopped short.
            short = " (free fit short of its optimum)" if value < -1e-6 else ""
            profiled = f"{value:.3g}{short}"
        points = item.get("dchi2_data_points")
        prediction_points = item.get("dchi2_prediction_points")
        shown.append(
            {
                **row,
                **mean,
                "recomputed_difference": difference,
                # The prediction slice at the candidate: the chi-square cost of any drift.
                "prediction_dchi2_at_candidate": (
                    None if prediction_points is None else float(prediction_points[1])
                ),
                "mock_mean_b1_at_centre": b1_text,
                "mock_mean_dchi2_profiled": profiled,
                "mock_mean_dchi2_slice_at_candidate": (
                    None if points is None else float(points[0])
                ),
            }
        )
    return shown


def render_centres_report(
    candidate_path, config, destination, *, run_name, companions=None, diagnostics_dir=None
):
    """Write `index.html`, figures and `records.csv` for one candidate artifact.

    ``companions`` maps other run names to their candidate paths for the consistency panels.
    A `notes.md` is created once and never overwritten.
    """
    from ..figures.data import write_csv

    candidate_path = Path(candidate_path)
    digest = candidate_digest(candidate_path)
    candidate = yaml.safe_load(candidate_path.read_text())
    if candidate.get("review_reference") is not None:
        raise ValueError("the report reviews an unfrozen candidate; this one is already frozen")
    rows = record_rows(candidate, config, run_name)
    others = {
        name: record_rows(yaml.safe_load(Path(path).read_text()), config, name)
        for name, path in (companions or {}).items()
    }
    keys = [record_key(record["context"]) for record in candidate["records"]]
    native, failures, provenance = _diagnostics(candidate, digest, diagnostics_dir)
    b1_limits = config["fitting"]["nonlinear_parameters"]["b1"]["prior"]["limits"]
    native_rows = _native_rows(rows, keys, native, [float(edge) for edge in b1_limits])
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    figures = [
        ("Centres by redshift", plot_ladders(rows, destination / "centre-ladders")),
        ("Constraint strength", plot_constraint(rows, destination / "constraint-strength")),
        ("Consistency across runs", plot_consistency(rows, others, destination / "consistency")),
        ("Compression health", plot_health(rows, destination / "compression-health")),
    ]
    if native:
        figures.append(
            (
                "Recovery-mock mean against the theory centres (consistency only)",
                plot_data_against_centres(rows, keys, native, destination / "data-vs-centres"),
            )
        )
        for row, key in zip(rows, keys):
            if key in native:
                figures.append(
                    # File names stay dot-free: with_suffix would cut "sigma_0.03" at its dot.
                    (
                        key,
                        plot_record(
                            row, native[key], destination / f"record--{key}".replace(".", "p")
                        ),
                    )
                )
    if native or failures:
        if provenance:
            origin = (
                f"code <code>{html.escape(str(provenance['code_commit']))}</code>, "
                f"{int(provenance['points'])}-point grid"
            )
        else:
            origin = (
                "<b>provenance unrecorded</b> (written before the diagnostics stored it; "
                "a preview, not a review record)"
            )
        layer2 = f"<p>Native diagnostics: {len(native)} of {len(rows)} records ({origin}). "
        if native_rows:
            worst = max(r["recomputed_difference"] for r in native_rows)
            layer2 += (
                f"Largest |recomputed − candidate| centre: {worst:.2g} Mpc/h; records beyond "
                f"{RECOMPUTED_TOLERANCE_MPC_H} Mpc/h are flagged.</p>"
            )
        else:
            layer2 += "No record finished.</p>"
        if failures:
            layer2 += "<p><b>Failed records</b>: " + "; ".join(
                f"{html.escape(k)}: {html.escape(v)}" for k, v in failures.items()
            )
            layer2 += "</p>"
        layer2 += (
            "<p><b>The recovery-mock panels are a consistency check, never a way to choose or "
            "tune a centre.</b> They use the mean of the 25 recovery mocks the fits will later "
            "run on. Since 2026-10-19 they are kept out of the centres; "
            "recovery-mock dilations were already excluded. "
            "Dilations are fixed at one, so any residual shift in the mocks leaks into their "
            "preferred damping. The profiled Δχ² profiles b1 and the broadband in both fits; "
            "the Δχ² maps are conditional slices with b1 fixed, so their contours are narrower "
            "than profiled regions, and they are measured from the lowest point evaluated.</p>"
            f"{_table(native_rows, NATIVE_COLUMNS)}"
        )
    else:
        layer2 = (
            "<p><b>Native diagnostics pending</b>: profiles, prediction and data overlays "
            "appear once the diagnostics job has run for this candidate.</p>"
        )
    write_csv(destination / "records.csv", [{**r, "flags": "; ".join(r["flags"])} for r in rows])
    notes = destination / "notes.md"
    if not notes.exists():
        with notes.open("x") as stream:
            stream.write(f"# Notes: {run_name} centres\n\n")
    flagged = [row for row in rows if row["flags"]]
    sections = "".join(
        f"<h2>{html.escape(title)}</h2><a href='{pdf}'>PDF</a><br><img src='{png}' "
        "style='max-width:100%'>"
        for title, files in figures
        if files
        for png, pdf in [files]
    )
    observed = datetime.now(timezone.utc).isoformat(timespec="seconds")
    page = (
        "<!doctype html><meta charset='utf-8'>"
        f"<title>Centres review {html.escape(run_name)}</title>"
        "<style>body{font-family:sans-serif;max-width:1200px;margin:2em auto;padding:0 1em}"
        "table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ccc;padding:2px 6px}"
        "</style>"
        f"<h1>Transfer damping centres: {html.escape(run_name)}</h1>"
        f"<p>Observed {observed}. Report code <code>{html.escape(code_commit())}</code>.</p>"
        f"<p>Candidate <code>{html.escape(str(candidate_path))}</code>, sha256 "
        f"<code>{digest}</code>; producer "
        f"<code>{html.escape(str(candidate['producer']['code_commit']))}</code>. "
        f"Decision: {html.escape(str(candidate['decision']))}. Not frozen. "
        f"<a href='notes.md'>Notes</a> · <a href='records.csv'>records.csv</a></p>"
        "<p><b>How to read this.</b> Each centre is the template damping that best reproduces a "
        "noiseless theory prediction of the transferred multipoles. The compression χ² and "
        "residuals show how well the template can represent that prediction. The curvature "
        "error is the width the measured data's mean covariance would allow with the dilations "
        "fixed, a lower bound on the width in the production fits (a single realization is "
        "about five times wider). Where it exceeds the prior width (constraint plot above the "
        "line), the fit's damping follows the prior centre, so the frozen centre itself sets it "
        "and its theory must be right. Warning signs that a centre is not determined: railing "
        "or sitting at the [0, 30] bounds (the template cannot tell values apart along that "
        "direction), a missing curvature error, an outlier against the other bins of the same "
        "target class, σ∥ below σ⊥, and disagreement beyond a prior width with the same bin in "
        "the other runs (grey band in the consistency panels).</p>"
        f"{layer2}<h2>Flagged records ({len(flagged)} of {len(rows)})</h2>{_table(flagged)}"
        f"{sections}<h2>All records</h2>{_table(rows)}"
    )
    partial = destination / "index.partial.html"
    partial.write_text(page)
    partial.replace(destination / "index.html")
    return destination / "index.html"
