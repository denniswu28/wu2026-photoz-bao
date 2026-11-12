"""Predict one survey's dilation precision at SPHEREx's effective volume.

Where the V_eff-matched density run reaches at least 90 per cent of the SPHEREx effective volume
(chosen 2026-10-23), the prediction is that pipeline's measured single-realization precision:
the median Minos half-width over usable intervals. Every other route takes the theoretical
estimate: the conditional density-and-area forecast (pzbao.diagnostics.covariance.forecast_route)
at the condition's target density and the configured sky fraction, normalized to one survey.
Rows are saved with provenance for the literature-comparison figure; the script writes
nothing but the output file.
"""

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
from collections import defaultdict
import subprocess
import sys

import numpy as np

#: Minimum attained fraction of the SPHEREx effective volume for the matched pipeline to stand in
#: for SPHEREx; below it the theoretical estimate is used (chosen 2026-10-23).
PIPELINE_VEFF_FRACTION = 0.9


def _widths(result):
    names = list(result["parameter_names"])
    covariance = np.asarray(result["covariance"], dtype=float)
    # The returned covariance is the two-dilation marginal, ordered as the dilations appear.
    dilations = [name for name in names if name in ("qiso", "qap")]
    sigma = np.sqrt(np.diag(covariance))
    return dict(zip(dilations, sigma.tolist())), float(result["fom"])


def prediction(row, matched_widths, veff_fraction, bounded):
    """The SPHEREx prediction for one route: the matched pipeline where it reaches the volume."""
    if veff_fraction is not None and veff_fraction >= PIPELINE_VEFF_FRACTION and matched_widths:
        return dict(
            kind="matched_veff_pipeline",
            sigma_qiso=float(np.median(matched_widths)),
            veff_fraction=veff_fraction,
            bounded=bounded,
            usable_intervals=len(matched_widths),
        )
    if row.get("status") == "available":
        return dict(
            kind="conditional_forecast",
            sigma_qiso=row["target_area"]["sigma"]["qiso"],
            veff_fraction=veff_fraction,
        )
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-run", type=Path, required=True, help="the unified primary refit")
    parser.add_argument(
        "--density-run", type=Path, required=True, help="the unified V_eff-matched density refit"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; preserve it and choose a new path")
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))
    from pzbao.diagnostics import covariance as forecast
    from pzbao.figures import data as readers
    from pzbao.run import load_run

    run = load_run(args.fit_run.resolve())
    target_fsky = float(forecast.configured_target_fsky())
    started = datetime.now(timezone.utc).isoformat()
    rows = []
    for route in readers.routes_for_run(run):
        if route["condition"] == "native":  # no density target: area-only, not a SPHEREx sample
            continue
        selected = {key: route[key] for key in readers._ROUTE_KEYS}
        row = dict(selected)
        try:
            result = forecast.forecast_route(
                run, selected, target_fsky=target_fsky, n_realizations=1
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            row.update(status="unavailable", reason=str(error))
            rows.append(row)
            continue
        for label in ("current", "target", "target_area"):
            sigma, fom = _widths(result["results"][label])
            row[label] = dict(sigma=sigma, fom=fom)
        row.update(
            status=result["status"],
            attained_nbar_h3_mpc3=result["attained_nbar"],
            target_nbar_h3_mpc3=result["target_nbar"],
            area_factor=result["area_factor"],
            target_fsky=result["target_fsky"],
            current_sigma_perp=result["current_sigma_perp"],
            target_sigma_perp=result["target_sigma_perp"],
            n_realizations_normalization=result["n_mocks_normalization"],
        )
        rows.append(row)
        print("forecast", *selected.values(), row["target_area"]["sigma"], flush=True)
    matched_rows, _ = readers.matched_density_fits(args.density_run.resolve())
    widths, fractions, seen = defaultdict(list), {}, defaultdict(set)
    for fit in matched_rows:
        key = (fit["measurement_bin"], fit["condition"], fit["reconstruction_state"])
        fractions[key] = (fit["veff_fraction"], fit["bounded"])
        if fit["parameter"] != "qiso" or fit["realization_id"] is None:
            continue
        if fit["realization_id"] in seen[key]:
            raise ValueError(f"duplicate matched realization {fit['realization_id']} for {key}")
        seen[key].add(fit["realization_id"])
        if fit.get("status") == "complete" and fit["usable"]:
            widths[key].append(fit["half_width"])
    for row in rows:
        key = (row["measurement_bin"], row["condition"], row["reconstruction_state"])
        row["prediction"] = prediction(row, widths[key], *fractions.get(key, (None, None)))
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    record = dict(
        started_utc=started,
        finished_utc=datetime.now(timezone.utc).isoformat(),
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        fit_run=str(args.fit_run.resolve()),
        density_run=str(args.density_run.resolve()),
        pipeline_veff_fraction=PIPELINE_VEFF_FRACTION,
        run_identity={k: run["metadata"][k] for k in readers.IDENTITY},
        renderer_checkout=str(repo),
        renderer_commit=commit,
        renderer_diff=subprocess.check_output(
            ["git", "diff", "--", "pzbao"], cwd=repo, text=True
        ),
        method=(
            "matched V_eff pipeline (median usable single-realization Minos half-width) where the "
            "density run attains at least the pipeline V_eff fraction; otherwise the conditional "
            "density-and-area forecast (exploratory revisions 2.11-F1/F3) at the condition's "
            "target density and the configured sky fraction, one survey (n_realizations=1)"
        ),
        target_fsky=target_fsky,
        rows=rows,
        scientific_acceptance="Not established by this forecast",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    readers.write_json(args.output, record)
    print("Forecast rows:", args.output, flush=True)


if __name__ == "__main__":
    main()
