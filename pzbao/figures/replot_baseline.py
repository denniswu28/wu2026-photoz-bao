"""Redraw baseline paper figures from the rows an earlier baseline render saved.

render_baseline.py reads the per-bin runs and writes the rows each figure draws to
paper/figures/source/baseline/. For the figures whose rows are complete there (the survey
landscape, the joint FoM bars, the distance ratios, the paired reconstruction gains, the
literature comparison and the photo-z response), this script runs the baseline notebook with no
run inputs and those rows injected, so a change of figure design needs neither the runs nor their
storage. Figures drawn from posterior draws (post1d, post2d) cannot be redrawn this way: the draws
are not saved.
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
}
SKIPPED_CELLS = ("native-run", "power")


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
