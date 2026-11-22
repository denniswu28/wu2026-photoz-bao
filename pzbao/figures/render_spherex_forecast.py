"""Predict one survey's dilation precision at SPHEREx's effective volume.

Where the V_eff-matched density run reaches at least 90 per cent of the SPHEREx effective volume
(chosen 2026-10-23), the prediction is that pipeline's measured single-realization precision:
the median Minos half-width over usable intervals. Every other route takes the theoretical
estimate: the conditional density-and-area forecast (pzbao.diagnostics.covariance.forecast_route)
at the condition's target density and the configured sky fraction, normalized to one survey.
Before reconstruction the forecast is evaluated at the unreconstructed damping, not at the
prior centre shared by both states (decision of 2026-11-21).
Rows are saved with provenance for the literature-comparison figure; the script writes
nothing but the output file.

``--saved-selection`` takes the selection quantities from the saved baseline rows where the
measurement run no longer holds its per-unit selection records (see ``saved_selection``).
``--matched-record`` carries the matched-pipeline predictions over from an earlier record of
this script where the density run can no longer be loaded.
"""

import argparse
from datetime import datetime, timezone
import json
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


def saved_selection(source, config, cosmology):
    """Selection quantities of the forecast, read from the saved baseline rows.

    The forecast takes three things from a tracer's selection record: each sample's effective
    redshift, its mean count and volume in the measurement bin, and its radial density there.
    The baseline render saved all three (effective-redshifts.json and density.json), the radial
    counts summed over the two caps. The stand-in gives both caps that common radial density,
    its only difference from the record; the counts, volumes and effective redshifts are the
    record's own. It serves where the per-unit records are gone and is no selection product.
    """
    density = {
        (row["measurement_bin"], row["condition"]): row
        for row in json.loads((source / "density.json").read_text())
    }
    redshift = {
        (row["measurement_bin"], row["condition"]): row["z_eff"]
        for row in json.loads((source / "effective-redshifts.json").read_text())
    }

    def read(run, tracer):
        spec = config["tracers"][tracer]
        bins = [shell["name"] for shell in spec["measurement_bins"]]
        conditions, effective = {}, {}
        for label in sorted({condition for name, condition in density if name in bins}):
            rows = [density[(name, label)] for name in bins]
            edges = np.asarray(rows[0]["edges"], dtype=float)
            for row in rows[1:]:
                np.testing.assert_allclose(row["edges"][0], edges[-1], rtol=0, atol=1e-10)
                edges = np.r_[edges, np.asarray(row["edges"][1:], dtype=float)]
            radial = np.concatenate([np.asarray(row["density"], dtype=float) for row in rows])
            starts = np.cumsum([0] + [len(row["density"]) for row in rows])
            shell = np.diff(np.asarray(cosmology.comoving_radial_distance(edges)) ** 3)
            groups = {}
            for cap, area in spec["footprint"]["area_deg2"].items():
                volume = shell * area * (np.pi / 180) ** 2 / 3
                counts = radial * volume
                groups[cap] = dict(
                    radial=dict(z_edges=edges.tolist(), mean_counts=counts.tolist()),
                    volumes_mpc_h3={
                        name: float(volume[a:b].sum())
                        for name, a, b in zip(bins, starts[:-1], starts[1:])
                    },
                    counts={
                        name: float(counts[a:b].sum())
                        for name, a, b in zip(bins, starts[:-1], starts[1:])
                    },
                )
            for name, row in zip(bins, rows):
                # The saved totals are the record's: the stand-in must return them unchanged.
                for key, saved in (("volumes_mpc_h3", "volume_mpc_h3"), ("counts", "mean_count")):
                    np.testing.assert_allclose(
                        sum(group[key][name] for group in groups.values()), row[saved], rtol=1e-9
                    )
            conditions[label] = groups
            effective[label] = {name: dict(z_eff=redshift[(name, label)]) for name in bins}
        return dict(tracer=tracer, conditions=conditions, effective_redshifts=effective)

    return read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-run", type=Path, required=True, help="the unified primary refit")
    matched = parser.add_mutually_exclusive_group(required=True)
    matched.add_argument("--density-run", type=Path, help="the unified V_eff-matched density refit")
    matched.add_argument(
        "--matched-record",
        type=Path,
        help="an earlier record of this script: its matched-pipeline predictions and attained "
        "V_eff fractions are carried over unchanged",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--saved-selection",
        type=Path,
        help="saved baseline rows (paper/figures/source/baseline) standing in for the "
        "measurement run's per-unit selection records",
    )
    parser.add_argument(
        "--prior-centre-prereconstruction",
        action="store_true",
        help="evaluate pre-reconstruction routes at the damping-prior centre, as before "
        "2026-11-21 (comparison only)",
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; preserve it and choose a new path")
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))
    from pzbao.diagnostics import covariance as forecast
    from pzbao.figures import data as readers
    from pzbao.run import load_run

    run = load_run(args.fit_run.resolve())
    if args.saved_selection is not None:
        from pzbao import catalog, likelihood, selection

        reader = saved_selection(
            args.saved_selection.resolve(),
            run["config"],
            catalog.build_mock_cosmology(run["config"]),
        )
        # forecast_route and the route likelihood each read the record through their own name.
        selection.read_preparation = likelihood.read_preparation = reader
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
                run,
                selected,
                target_fsky=target_fsky,
                n_realizations=1,
                unreconstructed_prereconstruction=not args.prior_centre_prereconstruction,
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
            target_sigma_parallel=result["target_sigma_parallel"],
            n_realizations_normalization=result["n_mocks_normalization"],
        )
        rows.append(row)
        print("forecast", *selected.values(), row["target_area"]["sigma"], flush=True)
    if args.matched_record is not None:
        earlier = {
            (row["measurement_bin"], row["condition"], row["reconstruction_state"]): row[
                "prediction"
            ]
            for row in json.loads(args.matched_record.read_text())["rows"]
        }
        for row in rows:
            carried = earlier[
                (row["measurement_bin"], row["condition"], row["reconstruction_state"])
            ]
            if carried is not None and carried["kind"] == "matched_veff_pipeline":
                row["prediction"] = carried
            else:
                row["prediction"] = prediction(
                    row, [], None if carried is None else carried["veff_fraction"], None
                )
    else:
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
        density_run=None if args.density_run is None else str(args.density_run.resolve()),
        matched_record=(
            None if args.matched_record is None else str(args.matched_record.resolve())
        ),
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
        prereconstruction_damping=(
            "damping-prior centre"
            if args.prior_centre_prereconstruction
            else "unreconstructed: the transverse formula in the limit S -> 0, "
            "parallel = (1 + f) transverse"
        ),
        selection_source=(
            "the measurement run's selection records"
            if args.saved_selection is None
            else "saved baseline rows (density.json, effective-redshifts.json) under "
            f"{args.saved_selection.resolve()}: record counts, volumes and effective redshifts, "
            "one radial density for both caps"
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
