"""Redraw the robustness paper figures from the rows an earlier robustness render saved.

render_robustness.py reads the runs named in its provenance and writes the rows each figure draws
to paper/figures/source/robustness/ (choices, smoothing, transfer and density). This script runs
only the notebook producers on those saved rows, so a change of figure design needs neither the
runs nor their storage; the provenance records the render the rows come from.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

PRODUCER_CELLS = ("setup", "presentation-style", "paper-robustness")


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
    source = destination / "source" / "robustness"
    build = paper / "build" / args.build_name
    if build.exists():
        parser.error("Build already exists; preserve it and choose a new build name")
    sys.path.insert(0, str(repo))
    from pzbao.figures import data as readers
    from pzbao.figures.render_robustness import PAPER_FIGURES
    from pzbao.figures.style import ensure_paper_figure_output_allowed

    rows = {
        name: json.loads((source / f"{name}.json").read_text())
        for name in ("choices", "smoothing", "transfer", "density", "provenance")
    }
    build.mkdir(parents=True)
    started = datetime.now(timezone.utc).isoformat()
    notebook_path = repo / "pzbao/figures/comparisons.ipynb"
    cells = {c["id"]: c for c in json.loads(notebook_path.read_text())["cells"]}
    namespace = {}
    os.chdir(repo)
    for name in PRODUCER_CELLS:
        if name == "paper-robustness":
            namespace.update(
                run_robustness=rows["choices"],
                scan_summary=rows["smoothing"],
                transfer_intervals=rows["transfer"]["intervals"],
                density_appendix=rows["density"],
            )
        print("Notebook cell:", name, flush=True)
        exec(compile("".join(cells[name]["source"]), f"{notebook_path}:{name}", "exec"), namespace)
    missing = set(PAPER_FIGURES) - set(namespace["figures"])
    if missing:
        raise ValueError("Robustness figures were not drawn: " + ", ".join(sorted(missing)))
    for name in PAPER_FIGURES:
        ensure_paper_figure_output_allowed(name)
        namespace["figures"][name].savefig(destination / f"{name}.pdf")
        namespace["figures"][name].savefig(destination / f"{name}.png", dpi=300)
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
        source_render=rows["provenance"],
        runs=rows["provenance"]["runs"],
        figures=list(PAPER_FIGURES),
        visual_review="Pending",
        scientific_acceptance="Not established by this rendering",
    )
    readers.write_json(source / "provenance.json", record)
    (build / "complete.json").write_text(json.dumps(record, indent=2) + "\n")
    print("Robustness figures redrawn for visual review:", destination, flush=True)


if __name__ == "__main__":
    main()
