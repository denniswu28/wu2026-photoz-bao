# Running the pipeline

## One run, stage by stage

Each stage reads the previous stage's handoff file and writes its own under the run directory.
Paths below are examples.

```bash
export BAO_STORAGE_ROOT=/path/to/storage
RUNS=$BAO_STORAGE_ROOT/runs
CATALOGS=$BAO_STORAGE_ROOT/data/prepared

pzbao catalog --run-name example --workload pilot --catalog-root $CATALOGS --output-root $RUNS --download
pzbao select      --input $RUNS/example/catalog/catalog.yaml
pzbao reconstruct --input $RUNS/example/select/selection.yaml
pzbao cluster     --input $RUNS/example/reconstruct/reconstruction.yaml --statistics xi
pzbao covariance  --input $RUNS/example/cluster/clustering.yaml
pzbao fit         --input $RUNS/example/covariance/covariances.yaml
pzbao sample      --input $RUNS/example/fit/fits.yaml
pzbao review      --run $RUNS/example
```

- `catalog` creates the run. It freezes the configuration (`analysis.yaml`, or the file given
  with `--config`) and the shot-noise registry into the run directory, and records the code
  commit and a digest of the configuration in `run.yaml`. Production runs need a clean checkout;
  `--scratch` records uncommitted changes and writes under `$RUNS/scratch/` instead.
- `--workload` chooses which tracers, redshift-error conditions and realizations the run covers
  (`pilot` is a small subset; the paper's runs used `full_grid`, `smoothing_scan`, `reciso`,
  `density` and `transfer`, defined in `analysis.yaml`).
- Every stage splits into units (a tracer, a realization, a route) that can run separately with
  `--unit`, or as a Slurm array: add `--scheduler slurm --account ACCOUNT` to print the array
  script, and `--submit` to submit it.
- `--resume` reuses completed units with matching options. Changing a scientific setting needs a
  new run.
- A refit of an existing measurement with a different fit definition writes a separate run:

  ```bash
  pzbao fit --input $RUNS/example/covariance/covariances.yaml \
      --inference-run $RUNS/example-refit --inference-definition unified_prior_v1
  ```

`pzbao review` renders a run's products, complete or partial, as HTML, Markdown, CSV and figures,
and keeps any notes already written in the run's `review/review.md`.

## The runs behind the paper

The paper's measurements and fits come from the 17 runs listed in
[provenance.md](provenance.md); the frozen configurations of 16 of them are in `paper/runs/`. To repeat a
run, pass its configuration to `catalog` with `--config paper/runs/<run>.yaml` and the
workload listed for it, and run the remaining stages as above. The fit runs (`unified-fit-v3--*`)
refit a measurement run with `--inference-run` and `--inference-definition`: `unified_prior_v1`
for the main fits, `unified_prior_flat_v1` and `unified_prior_pcs2_v1` for the two checks.

The full set is large: 25 realizations of two tracers, five redshift-error conditions and
several smoothing radii. The paper's runs needed several terabytes of storage and a Slurm
cluster.
