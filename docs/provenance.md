# Where the paper's results came from

The paper's measurements, fits and figures come from the 17 runs below, made in October 2026 on
the DESI DR1 AbacusSummit mocks. They were produced by earlier versions of this code. This
repository is the code as of 10 November 2026.

## Which code version produced what

- **15 October 2026:** the catalogue, selection, reconstruction and clustering products of the two
  cross-σz sample runs at 25 Mpc/h (`full-v3--self--R25`, `full-v3--sigma_0.003--R25`). These
  products came from an earlier run of the same definition, and the later run reused them. Reuse
  requires an identical configuration digest. That earlier run is no longer on our storage, so
  these two runs and their fits can no longer be reloaded from the runs themselves.
- **16 October 2026:** every other measurement stage, including the covariances of the two
  cross-σz runs.
- **19 October 2026:** the V_eff-matched density measurement (`per-bin-extra-v2--density`). That run
  is no longer on our storage; its configuration digest survives in the fit run that read it.
- **21 October 2026:** every fit and posterior sample.

## Changes since those versions

Each run's configuration is frozen with the run, so later code changes cannot alter it; they
change the code that executes it. The changes to the science modules are:

- **After 15 October:**
  - per-bin reconstruction for the core analysis (the cross-σz path already solved one field per
    redshift bin and cap);
  - the V_eff-matched density tiers;
  - reuse of an earlier run's products in a later run of the same definition;
  - the joint constraint from the combination of the redshift-precision samples.
- **After 16 October:**
  - the flat-prior robustness check;
  - the matched-density tiers in the per-cap input cache;
  - per-radius transverse damping centres for the smoothing scan;
  - uniform-prior fits of the cross-σz runs at every smoothing radius;
  - the per-bin damping priors that every fit of 21 October uses.
- **After 21 October:** one change, on 9 November 2026: the matched-density targets are now
  recorded from the density handoff rather than from the native sample labels.

## Checks made for this release

- **Configurations:** each run's frozen configuration loads and passes schema validation with this
  release. That includes the configuration of `unified-fit-v3--density`, which is not shipped: it
  embeds the prepared-catalogue manifest with storage paths.
- **Digests:** recomputed from the frozen files, each digest equals the one recorded with the run.
- **Shot-noise registry:** all 17 runs used the registry shipped as
  `pzbao/config/frozen_alpha_sn.json`.

A digest identifies the configuration, not the code. The campaign has not been rerun with this
release.

To recompute a digest:

```python
import json, yaml
from pzbao.config import analysis_config_sha256

config = yaml.safe_load(open("paper/runs/unified-fit-v3--primary.yaml"))
registry = json.load(open("pzbao/config/frozen_alpha_sn.json"))
print(analysis_config_sha256({"analysis": config, "frozen_alpha_sn": registry}))
```

## The runs

| Run | Workload | Stages run here | Products reused from | Code date | Configuration digest | Shipped |
| --- | --- | --- | --- | --- | --- | --- |
| `per-bin-v2--primary` | full_grid | catalog, select, reconstruct, cluster, covariance, fit, sample | - | 2026-10-16 | `a1d00dc768cf196d` | yes |
| `per-bin-v2--scan--R15` | smoothing_scan | catalog, select, reconstruct, cluster, covariance | - | 2026-10-16 | `54063e9b5c477eca` | yes |
| `per-bin-v2--scan--R25` | smoothing_scan | catalog, select, reconstruct, cluster, covariance | - | 2026-10-16 | `e6fe21bdcd3220b7` | yes |
| `per-bin-v2--scan--R35` | smoothing_scan | catalog, select, reconstruct, cluster, covariance | - | 2026-10-16 | `ccaedd0a90c7a465` | yes |
| `per-bin-v2--reciso--R15` | reciso | catalog, select, reconstruct, cluster, covariance | - | 2026-10-16 | `49df2ecc49295783` | yes |
| `full-v3--self--R25` | transfer | covariance | catalog: full-v2-recovery--self--R25 (code 2026-10-15), select: full-v2-recovery--self--R25 (code 2026-10-15), reconstruct: full-v2-recovery--self--R25 (code 2026-10-15), cluster: full-v2-recovery--self--R25 (code 2026-10-15) | 2026-10-16 | `4c12fe4e0dfc4377` | yes |
| `full-v3--sigma_0.003--R25` | transfer | covariance | catalog: full-v2-recovery--sigma_0.003--R25 (code 2026-10-15), select: full-v2-recovery--sigma_0.003--R25 (code 2026-10-15), reconstruct: full-v2-recovery--sigma_0.003--R25 (code 2026-10-15), cluster: full-v2-recovery--sigma_0.003--R25 (code 2026-10-15) | 2026-10-16 | `bb7c26b37f7c1773` | yes |
| `unified-fit-v3--primary` | full_grid | fit, sample | measurement: per-bin-v2--primary (code 2026-10-16) | 2026-10-21 | `f47cc6ad34a372a1` | yes |
| `unified-fit-v3--primary--flat` | full_grid | fit | measurement: per-bin-v2--primary (code 2026-10-16) | 2026-10-21 | `02f72b06c22509ba` | yes |
| `unified-fit-v3--primary--pcs2` | full_grid | fit | measurement: per-bin-v2--primary (code 2026-10-16) | 2026-10-21 | `e17d7dea5f122648` | yes |
| `unified-fit-v3--reciso--R15` | reciso | fit, sample | measurement: per-bin-v2--reciso--R15 (code 2026-10-16) | 2026-10-21 | `88ce59fa9ca1c06e` | yes |
| `unified-fit-v3--scan--R15` | smoothing_scan | fit, sample | measurement: per-bin-v2--scan--R15 (code 2026-10-16) | 2026-10-21 | `ef273d1466fb2c98` | yes |
| `unified-fit-v3--scan--R25` | smoothing_scan | fit, sample | measurement: per-bin-v2--scan--R25 (code 2026-10-16) | 2026-10-21 | `b5bddbb1cdb4db2e` | yes |
| `unified-fit-v3--scan--R35` | smoothing_scan | fit, sample | measurement: per-bin-v2--scan--R35 (code 2026-10-16) | 2026-10-21 | `ffba4a5058d62b2c` | yes |
| `unified-fit-v3--density` | density | fit, sample | measurement: per-bin-extra-v2--density (code 2026-10-19) | 2026-10-21 | `9f80ccf624713e46` | no (embeds catalogue paths) |
| `unified-fit-v3--self--R25` | transfer | fit, sample | measurement: full-v3--self--R25 (code 2026-10-16) | 2026-10-21 | `c8905acf2466263b` | yes |
| `unified-fit-v3--sigma_0.003--R25` | transfer | fit, sample | measurement: full-v3--sigma_0.003--R25 (code 2026-10-16) | 2026-10-21 | `3e1a114523b07deb` | yes |

The digests are the first 16 hexadecimal characters of the full SHA-256 digest.
