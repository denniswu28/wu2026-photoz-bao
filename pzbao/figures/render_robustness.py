"""Execute the comparisons notebook once and save the five robustness paper figures.

Inputs and destination are explicit; the notebook owns all figure choices. The figures and the rows they draw are written beside the
other paper figures, and the build directory records the runs, commit and figure set.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

PAPER_FIGURES = (
    "fig_robustness_choices_iso",
    "fig_robustness_choices_ap",
    "fig_robustness_smoothing",
    "fig_robustness_transfer",
    "fig_robustness_density",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-run", type=Path, required=True, help="the unified primary refit"
    )
    parser.add_argument("--flat-run", type=Path, required=True)
    parser.add_argument("--pcs2-run", type=Path, required=True, help="serves both tracers")
    parser.add_argument("--reciso-run", type=Path, required=True)
    parser.add_argument("--scan-run", type=Path, action="append", required=True)
    parser.add_argument(
        "--transfer",
        type=Path,
        nargs=2,
        action="append",
        required=True,
        metavar=("CROSS_RUN", "SELF_RUN"),
    )
    parser.add_argument("--density-run", type=Path, required=True)
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
    sys.path.insert(0, str(repo))
    from pzbao.figures import data as readers
    from pzbao.figures.style import ensure_paper_figure_output_allowed

    build.mkdir(parents=True)
    started = datetime.now(timezone.utc).isoformat()
    with (build / "validation.log").open("w") as log:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "tests/test_run_figure_readers.py",
                "tests/test_figure_notebooks.py",
            ],
            cwd=repo,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    print("Targeted tests passed", flush=True)
    resolved = lambda path: str(path.resolve())  # noqa: E731
    choices = dict(
        baseline=resolved(args.baseline_run),
        flat=resolved(args.flat_run),
        pcs2_lrg=resolved(args.pcs2_run),
        pcs2_bgs=resolved(args.pcs2_run),
        reciso=resolved(args.reciso_run),
        scan=[resolved(path) for path in args.scan_run],
        transfer=[(resolved(cross), resolved(self)) for cross, self in args.transfer],
        density=resolved(args.density_run),
        density_primary=resolved(args.baseline_run),
    )
    notebook_path = repo / "pzbao/figures/comparisons.ipynb"
    notebook = json.loads(notebook_path.read_text())
    namespace = dict(robustness_run_choices=choices)
    os.chdir(repo)
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        print("Notebook cell:", cell["id"], flush=True)
        exec(
            compile("".join(cell["source"]), str(notebook_path) + ":" + cell["id"], "exec"),
            namespace,
        )
    missing = set(PAPER_FIGURES) - set(namespace["figures"])
    if missing:
        raise ValueError("Robustness figures were not drawn: " + ", ".join(sorted(missing)))
    for name in PAPER_FIGURES:
        ensure_paper_figure_output_allowed(name)
        figure = namespace["figures"][name]
        figure.savefig(destination / f"{name}.pdf")
        figure.savefig(destination / f"{name}.png", dpi=300)
    source = destination / "source" / "robustness"
    source.mkdir(parents=True, exist_ok=True)
    readers.write_json(source / "choices.json", namespace["run_robustness"])
    readers.write_json(source / "smoothing.json", namespace["scan_summary"])
    readers.write_json(source / "reproducibility.json", namespace["reproducibility"])
    readers.write_json(
        source / "transfer.json",
        dict(
            intervals=namespace["transfer_intervals"],
            paired=[
                dict(cross_run=run, summaries=summaries)
                for run, (_, summaries) in namespace["transfer_results"]
            ],
        ),
    )
    density = namespace["density_appendix"]
    readers.write_json(
        source / "density.json", dict(summary=density["summary"], matching=density["matching"])
    )
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
        runs=choices,
        figures=list(PAPER_FIGURES),
        visual_review="Pending",
        scientific_acceptance="Not established by this rendering",
    )
    readers.write_json(source / "provenance.json", record)
    (build / "complete.json").write_text(json.dumps(record, indent=2) + "\n")
    print("Robustness figures ready for visual review:", destination, flush=True)


if __name__ == "__main__":
    main()
