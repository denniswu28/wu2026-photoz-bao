"""Execute the baseline notebook once and save the baseline paper figures.

Inputs and destination are explicit; the notebook owns all figure choices. Existing builds
are preserved.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, help="the saved baseline bundle (broad field)")
    parser.add_argument(
        "--fit-run",
        type=Path,
        help="the unified-prior refit of the per-bin primary; --selection-run is then its "
        "measurement run and no bundle is read",
    )
    parser.add_argument("--selection-run", type=Path, required=True)
    parser.add_argument(
        "--spherex-forecast",
        type=Path,
        help="SPHEREx prediction rows from render_spherex_forecast.py for the literature figure",
    )
    parser.add_argument("--paper-dir", type=Path)
    parser.add_argument("--build-name", required=True)
    args = parser.parse_args()
    if (args.report is None) == (args.fit_run is None):
        parser.error("Give exactly one of --report (saved bundle) or --fit-run (per-bin refit)")
    repo = Path(__file__).resolve().parents[2]
    paper = (args.paper_dir or repo / "paper").resolve()
    if not (paper / "figures").is_dir():
        parser.error("Paper destination must contain the figures directory")
    if args.spherex_forecast is not None and not args.spherex_forecast.is_file():
        parser.error("SPHEREx prediction rows not found")
    if Path(args.build_name).name != args.build_name or args.build_name in (".", ".."):
        parser.error("Build name must be a single directory name")
    destination = paper / "figures"
    build = paper / "build" / args.build_name
    if build.exists():
        parser.error("Build already exists; preserve it and choose a new build name")

    os.environ.update(
        MPLBACKEND="Agg",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
    )
    build.mkdir(parents=True)
    started = datetime.now(timezone.utc).isoformat()
    try:
        with (build / "validation.log").open("w") as log:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    "tests/test_paper_baseline_refresh.py",
                    "tests/test_figure_notebooks.py",
                    "tests/test_plot_style.py",
                    "tests/test_paper_precision_refresh.py",
                ],
                cwd=repo,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        print("Targeted tests passed", flush=True)
        notebook_path = repo / "pzbao/figures/baseline.ipynb"
        notebook = json.loads(notebook_path.read_text())
        namespace = dict(
            baseline_report_choice=args.report.resolve() if args.report else None,
            fit_run_choice=args.fit_run.resolve() if args.fit_run else None,
            selection_run_choice=args.selection_run.resolve(),
            spherex_forecast_choice=(
                args.spherex_forecast.resolve() if args.spherex_forecast else None
            ),
            figure_output_choice=destination,
            paper_dir_choice=paper,
        )
        os.chdir(repo)
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code" or cell["id"] in ("native-run", "power"):
                continue
            print("Notebook cell:", cell["id"], flush=True)
            exec(
                compile("".join(cell["source"]), str(notebook_path) + ":" + cell["id"], "exec"),
                namespace,
            )
        required = {
            "fig_cf",
            "fig_lrg1_realizations",
            "fig_distance_ratios",
            "fig_post1d",
            "fig_post2d",
            "fig_fom_2d",
            "fig_recon_quality",
            "fig_nz",
            "fig_pipeline",
            "fig_los_damping",
            "fig_literature_comparison",
            "fig_sigma_r_precision",
            "fig_survey_landscape",
        }
        if set(namespace["figures"]) != required:
            raise ValueError("Rendered figure set differs from the paper's baseline figures")
        for name in required:
            for suffix in (".pdf", ".png"):
                if not (destination / (name + suffix)).stat().st_size:
                    raise ValueError("Empty rendered figure: " + name + suffix)
        (build / "complete.json").write_text(
            json.dumps(
                dict(
                    started_utc=started,
                    finished_utc=datetime.now(timezone.utc).isoformat(),
                    slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                    checkout=str(repo),
                    baseline_report=str(args.report.resolve()) if args.report else None,
                    fit_run=str(args.fit_run.resolve()) if args.fit_run else None,
                    selection_run=str(args.selection_run.resolve()),
                    spherex_forecast=(
                        str(args.spherex_forecast.resolve()) if args.spherex_forecast else None
                    ),
                    paper_directory=str(paper),
                    figures=sorted(required),
                    verification="Targeted tests, native readback and render",
                    visual_review="Pending",
                    scientific_acceptance="Not established",
                ),
                indent=2,
            )
            + "\n"
        )
        print("Baseline figures saved for visual review:", destination, flush=True)
    except Exception as error:
        (build / "failed.json").write_text(
            json.dumps(
                dict(
                    started_utc=started,
                    failed_utc=datetime.now(timezone.utc).isoformat(),
                    error=repr(error),
                    scientific_acceptance="Not established",
                ),
                indent=2,
            )
            + "\n"
        )
        raise


if __name__ == "__main__":
    main()
