# wu2026-photoz-bao

[![DOI](https://zenodo.org/badge/1454775766.svg)](https://doi.org/10.5281/zenodo.24433146)

Analysis code for *BAO Reconstruction in the Presence of Photometric Redshift Uncertainty*
(Wu, Paillas & Eifler 2026).

The analysis tests standard density-field reconstruction on the public DESI DR1 AbacusSummit
mocks after adding Gaussian redshift errors and down-sampling toward SPHEREx-like number
densities. It measures correlation-function multipoles before and after reconstruction, builds
RascalC covariances with a calibrated shot-noise rescaling, and fits the BAO dilations by
profile likelihood and MCMC.

This repository is the record of that analysis. For a reusable library of the same methods
(photo-z smearing, random catalogues, reconstruction, measurement and fitting), see
[photo-bao](https://github.com/denniswu28/photo-bao).

## Layout

| Path | Contents |
| --- | --- |
| `pzbao/` | The pipeline. One module per stage: `catalog`, `selection`, `reconstruction`, `clustering`, `covariance`, `fitting`, `sampling`; the likelihood is in `likelihood.py`, the photo-z response and damping in `theory.py` and `models/`. |
| `pzbao/config/analysis.yaml` | Every scientific setting, validated against a schema when it is loaded. |
| `pzbao/run.py` | The command line (`pzbao`): runs, units, resume and Slurm array plans. |
| `pzbao/diagnostics/` | Per-stage diagnostics, the HTML/Markdown run report and inspection notebooks. |
| `pzbao/figures/` | The paper's figure notebooks and the scripts that render or redraw them. |
| `paper/figures/` | The paper's figures and, under `source/`, the rows each one is drawn from. |
| `paper/runs/` | The frozen configurations of the runs behind the paper. |
| `docs/` | How to run the pipeline, and where each result came from. |
| `tests/` | The test suite. |

## Install

```bash
conda env create -f environment.yml
conda activate pzbao
python -m pip install --no-deps --no-build-isolation -r requirements-science.txt
python -m pip install --no-deps -e .
pzbao --help
```

`requirements-science.txt` pins pyrecon, pycorr, pypower, cosmoprimo, desilike and RascalC to the
commits the paper's results were produced with; `--no-deps` stops pip from replacing those pins.

## Data

The input catalogues are the DESI DR1 AbacusSummit "complete" mocks, which are public:
`pzbao catalog --download` fetches the missing files from `data.desi.lbl.gov`, checking each
against the publisher's checksum manifest when it is present. The same-tracer reservoir slices that supply galaxies beyond the redshift edges are
cut from the parent mocks with `pzbao/data/ffa_reservoir_extractor.py`.

The SPHEREx number-density targets in `analysis.yaml` are samples 1–3 of
`galaxy_density_v28_base_cbe.txt` from
[SPHEREx/Public-products](https://github.com/SPHEREx/Public-products); its checksum is recorded
next to them.

Products go under `$BAO_STORAGE_ROOT` (default `./outputs`). The full analysis needs several
terabytes and a Slurm cluster; catalogues, chains and RascalC outputs are not in this repository.

## Run

The pipeline runs in eight stages, each reading the previous stage's saved products:

`catalog → select → reconstruct → cluster → covariance → fit → sample → review`

[docs/reproduce.md](docs/reproduce.md) walks through one run and lists the runs behind the paper.
Every run freezes its configuration and records the code commit, input identities and a
configuration digest next to its outputs; later stages refuse inputs that do not match.

## Figures

The paper figures are drawn by `pzbao/figures/baseline.ipynb` and `comparisons.ipynb` from saved
runs. Eleven of the eighteen can also be redrawn from the rows shipped in
`paper/figures/source/`, without the runs:

```bash
python pzbao/figures/replot_baseline.py --build-name redraw-baseline
python pzbao/figures/replot_robustness.py --build-name redraw-robustness
```

The other seven (`fig_cf`, `fig_lrg1_realizations`, `fig_nz`, `fig_pipeline`,
`fig_sigma_r_precision`, `fig_post1d`, `fig_post2d`) are drawn only by `render_baseline.py` from
the runs, because not all their inputs are saved here; the posterior samples, for example, are not.

## Tests

```bash
python -m pytest
```

Tests that need a Slurm job or the original runs skip.

## Citation and license

Cite the paper, and this repository with [CITATION.cff](CITATION.cff). MIT license; see
[LICENSE](LICENSE). Two functions in `pzbao/models/` are adapted from
[desilike](https://github.com/cosmodesi/desilike) and keep its BSD-3-Clause license; see
[LICENSES/](LICENSES/desilike-BSD-3-Clause.txt).
