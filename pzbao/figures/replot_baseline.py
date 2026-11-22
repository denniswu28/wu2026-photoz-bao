"""Redraw baseline paper figures from the rows an earlier baseline render saved.

render_baseline.py reads the per-bin runs and writes the rows each figure draws to
paper/figures/source/baseline/. For the figures whose rows are complete there (the survey
landscape, the joint FoM bars, the distance ratios, the paired reconstruction gains, the
literature comparison, the photo-z response, the radial selection n(z) and the gain against
density), this script runs the baseline notebook with no run inputs and those rows injected, so a
change of figure design needs neither the runs nor their storage. Figures drawn from posterior
draws (post1d, post2d) cannot be redrawn this way: the draws are not saved.

``--spherex-forecast`` first replaces the SPHEREx predictions of the saved literature rows by
those of a record of render_spherex_forecast.py, as the full render would form them.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

# Figure name -> saved tables it is drawn from.
REPLOT_FIGURES = {
    "fig_survey_landscape": ("survey-veff",),
    "fig_fom_2d": ("fom-values", "fom-joint"),
    "fig_distance_ratios": ("distance-ratios",),
    "fig_recon_quality": ("paired-gains",),
    "fig_literature_comparison": ("literature-comparison",),
    "fig_los_damping": ("los-damping",),
    "fig_nz": ("density", "survey-window"),
    # The matched-density tables are saved beside the baseline rows by their own producers.
    "fig_gain_density": (
        "posterior-summary",
        "precision-values",
        "../robustness/density",
        "../appendix/matched-density-posterior",
    ),
}
SKIPPED_CELLS = ("native-run", "power")


def apply_spherex_forecast(table, joint_rows, forecast):
    """The saved literature rows with their SPHEREx predictions taken from a forecast record.

    Class rows take the record's prediction; combined rows combine the four class predictions
    with the measured correlation of the saved joint record, as the literature-comparison cell of
    the baseline notebook does on a full render. Nothing else in the rows changes.
    """
    from pzbao.figures import data as readers

    predictions = {
        (row["measurement_bin"], row["condition"], row["reconstruction_state"]): row["prediction"]
        for row in forecast["rows"]
        if row.get("prediction") is not None
    }
    joint = {
        (row["route"]["measurement_bin"], row["route"]["reconstruction_state"]): row
        for row in joint_rows
    }
    states = dict(prerecon="prereconstruction", postrecon="postreconstruction")

    def width(name, condition, state):
        prediction = predictions.get((name, condition, state))
        if prediction is None:
            return None
        return dict(sigma=float(prediction["sigma_qiso"]), kind=prediction["kind"])

    table = json.loads(json.dumps(table))
    for row in table["current"] + table["joint_by_bin"]:
        if row.get("status", "available") != "available":
            continue
        name = row["measurement_bin"]
        for short, state in states.items():
            if row["condition"] != "joint":
                row[short]["spherex_prediction"] = width(name, row["condition"], state)
                continue
            record = joint[(name, state)]
            classes = record["tiers"]
            iso = [2 * index for index in range(len(classes))]
            correlation = [[record["estimate_correlation"][a][b] for b in iso] for a in iso]
            widths = [width(name, condition, state) for condition in classes]
            kinds = {item["kind"] for item in widths if item is not None}
            row[short]["spherex_prediction"] = (
                None
                if any(item is None for item in widths)
                else dict(
                    sigma=readers.consensus_width([item["sigma"] for item in widths], correlation),
                    kind=kinds.pop() if len(kinds) == 1 else "mixed",
                )
            )
    return table


def draw_from_saved_rows(repo, paper):
    """Execute the baseline notebook with the saved rows injected; return its namespace."""
    source = paper / "figures" / "source" / "baseline"
    rows = {
        name: json.loads((source / f"{name}.json").read_text())
        for tables in REPLOT_FIGURES.values()
        for name in tables
    }
    notebook_path = repo / "pzbao/figures/baseline.ipynb"
    notebook = json.loads(notebook_path.read_text())
    # The same cell sequence as render_baseline.py, so every producer sees the plotting state of
    # a full render; without run inputs only the saved-rows branches draw.
    namespace = dict(
        baseline_report_choice=None,
        fit_run_choice=None,
        selection_run_choice=None,
        spherex_forecast_choice=None,
        figure_output_choice=None,
        paper_dir_choice=paper,
        saved_baseline_rows=rows,
    )
    cwd = os.getcwd()
    os.chdir(repo)
    try:
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code" or cell["id"] in SKIPPED_CELLS:
                continue
            exec(
                compile("".join(cell["source"]), f"{notebook_path}:{cell['id']}", "exec"), namespace
            )
    finally:
        os.chdir(cwd)
    missing = set(REPLOT_FIGURES) - set(namespace["figures"])
    if missing:
        raise ValueError("Baseline figures were not drawn: " + ", ".join(sorted(missing)))
    return namespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-dir", type=Path)
    parser.add_argument("--build-name", required=True)
    parser.add_argument(
        "--spherex-forecast",
        type=Path,
        help="record of render_spherex_forecast.py whose predictions replace those of the saved "
        "literature rows before the redraw",
    )
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    paper = (args.paper_dir or repo / "paper").resolve()
    if not (paper / "figures").is_dir():
        parser.error("Paper destination must contain the figures directory")
    if Path(args.build_name).name != args.build_name or args.build_name in (".", ".."):
        parser.error("Build name must be a single directory name")
    destination = paper / "figures"
    source = destination / "source" / "baseline"
    build = paper / "build" / args.build_name
    if build.exists():
        parser.error("Build already exists; preserve it and choose a new build name")
    sys.path.insert(0, str(repo))
    from pzbao.figures import data as readers
    from pzbao.figures.style import ensure_paper_figure_output_allowed

    os.environ.update(MPLBACKEND="Agg")
    build.mkdir(parents=True)
    started = datetime.now(timezone.utc).isoformat()
    if args.spherex_forecast is not None:
        forecast_path = args.spherex_forecast.resolve()
        table = apply_spherex_forecast(
            json.loads((source / "literature-comparison.json").read_text()),
            json.loads((source / "joint-constraint.json").read_text()),
            json.loads(forecast_path.read_text()),
        )
        table["spherex_forecast"] = str(forecast_path)
        readers.write_json(source / "literature-comparison.json", table)
    namespace = draw_from_saved_rows(repo, paper)
    for name in REPLOT_FIGURES:
        ensure_paper_figure_output_allowed(name)
        # The export cell's arguments, so a redraw of unchanged rows reproduces the render.
        namespace["figures"][name].savefig(
            destination / f"{name}.png", dpi=300, bbox_inches="tight"
        )
        namespace["figures"][name].savefig(destination / f"{name}.pdf", bbox_inches="tight")
    render = json.loads((source / "provenance.json").read_text())
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    record = dict(
        started_utc=started,
        finished_utc=datetime.now(timezone.utc).isoformat(),
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        renderer_checkout=str(repo),
        renderer_commit=commit,
        renderer_diff=subprocess.check_output(
            ["git", "diff", "--", "pzbao/figures"], cwd=repo, text=True
        ),
        method="redrawn from the saved rows of the source render",
        spherex_forecast=(
            None if args.spherex_forecast is None else str(args.spherex_forecast.resolve())
        ),
        source_render={
            key: render.get(key)
            for key in (
                "observed_utc",
                "baseline_report",
                "selection_run",
                "measurement_identity",
                "renderer_commit",
            )
        },
        figures={
            name: [f"{table}.json" for table in tables] for name, tables in REPLOT_FIGURES.items()
        },
        visual_review="Pending",
        scientific_acceptance="Not established by this rendering",
    )
    readers.write_json(source / "replot-provenance.json", record)
    (build / "complete.json").write_text(json.dumps(record, indent=2) + "\n")
    print("Baseline figures redrawn for visual review:", destination, flush=True)


if __name__ == "__main__":
    main()
