"""Eight independently inspectable commands for immutable BAO runs and direct scientific products.

Help and scheduler planning import no optional science libraries. The scientific modules own
calculations and native formats; this module owns settings, unit execution and publication.
"""

from __future__ import annotations

import datetime
import importlib.metadata
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from contextlib import ExitStack, contextmanager

import click
import yaml

from .config import (
    BGS_ANISOTROPIC,
    CORE_POWER3_INFERENCE,
    CORE_INFERENCE,
    CORE_SCAN_INFERENCE,
    TRANSFER_INFERENCE,
    UNIFIED_PRIOR,
    UNIFIED_PRIOR_INFERENCE,
    DEFAULT_ANALYSIS_CONFIG_PATH,
    analysis_config_sha256,
    load_analysis_config,
    same_identity,
    validate_inference_config,
    derive_inference_config,
)
from .config.schema_validation import (
    PER_BIN_SCHEMA_VERSIONS,
    _UniqueKeyLoader,
    validate_analysis_config,
)
from .artifacts import resolve_saved_path
from .storage import (
    same_location,
    same_run_identity,
    inspect_transfer_storage,
    transfer_storage_writer,
)

from .config.stages import (  # one authority for stage order, handoff names and adoption
    ADOPTION_RECEIPT,
    HANDOFFS,
    REGISTERED_STAGES,
    STAGES,
)


def _read_yaml(path):
    with Path(path).open() as stream:
        value = yaml.load(stream, Loader=_UniqueKeyLoader)
    if not isinstance(value, dict):
        raise ValueError(f"expected a mapping in {path}")
    return value


def _write_yaml(path, value):
    path = Path(path)
    partial = path.with_name(path.name + "." + uuid.uuid4().hex + ".partial")
    partial.write_text(yaml.safe_dump(value, sort_keys=False))
    os.replace(partial, path)


def _safe_name(value):
    if (
        not value
        or value in {".", ".."}
        or any(
            c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
            for c in value
        )
    ):
        raise ValueError(
            "run and unit names may contain only letters, numbers, dots, dashes and underscores"
        )
    return value


def _code_identity(*, allow_dirty=False):
    repository = Path(__file__).resolve().parents[1]
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=repository, text=True
    ).strip()
    if dirty and not allow_dirty:
        raise ValueError(
            "production run creation requires a clean code checkout; use --scratch for experiments"
        )
    result = {"code_commit": commit, "repository": str(repository)}
    if allow_dirty:
        patch = subprocess.check_output(
            [
                "git",
                "diff",
                "--binary",
                "HEAD",
                "--",
                "pzbao",
                "pyproject.toml",
                "environment.yml",
            ],
            cwd=repository,
            text=True,
        )
        untracked = (
            subprocess.check_output(
                ["git", "ls-files", "--others", "--exclude-standard", "-z", "--", "pzbao"],
                cwd=repository,
            )
            .decode()
            .split(chr(0))
        )
        sources = {}
        for relative in filter(None, untracked):
            path = repository / relative
            if path.suffix in {".py", ".yaml", ".yml", ".json", ".toml"}:
                sources[relative] = path.read_text()
        state = {"commit": commit, "patch": patch, "untracked": sources}
        result.update(
            dirty=bool(dirty), code_state_sha256=analysis_config_sha256(state), source_state=state
        )
    return result


def _source_identity(run):
    return {
        "root": str(run["root"]),
        **{
            key: run["metadata"][key]
            for key in ("run_name", "code_commit", "analysis_config_sha256")
        },
    }


def create_run(
    output_root,
    name,
    *,
    workload,
    catalog_root,
    source_run=None,
    config_path=None,
    inference_definition=None,
    scratch=False,
    damping_centres=None,
    covariance_runs=(),
    predecessor=None,
):
    """Create a named run once, materializing the exact scalar registry into its snapshot.

    ``predecessor`` records an earlier run of the same definition whose published units this
    run adopts (``adopt-predecessor``); it must share workload, catalogue root, statistics,
    configuration and scalar registry, and only its identity is recorded here.
    """
    _safe_name(name)
    if predecessor is not None:
        if source_run is not None:
            raise ValueError("an inference run registers nothing; its source run supplies products")
        predecessor = load_run(predecessor, allow_source=False)
        if predecessor["metadata"].get("mode") == "scratch" and not scratch:
            raise ValueError("a production successor cannot adopt scratch products")
        if workload != predecessor["metadata"]["workload"] or not same_location(
            catalog_root, predecessor["metadata"]["catalog_root"]
        ):
            raise ValueError("a successor retains its predecessor's workload and catalog root")
    providers = _covariance_providers(source_run, inference_definition, covariance_runs)
    if source_run is not None and config_path is not None:
        raise ValueError("inference reuse cannot change its source configuration")
    if source_run is not None:
        source_run = load_run(source_run["root"], allow_source=False)
        if source_run["metadata"].get("mode") == "scratch" and not scratch:
            raise ValueError("inference from scratch products must remain in scratch mode")
        if workload != source_run["metadata"]["workload"] or not same_location(
            catalog_root, source_run["metadata"]["catalog_root"]
        ):
            raise ValueError("inference must retain its source workload and catalog root")
    config = (
        (load_analysis_config() if config_path is None else load_analysis_config(config_path))
        if source_run is None
        else source_run["config"]
    )
    if inference_definition is not None:
        if source_run is None:
            raise ValueError("an inference definition requires an original measurement source")
        config = derive_inference_config(source_run["config"], inference_definition)
    if damping_centres is not None:
        if source_run is None or inference_definition not in TRANSFER_INFERENCE:
            raise ValueError("damping adoption requires a new transfer inference run")
        artifact = _read_yaml(damping_centres)
        config["measurement"]["reconstruction"]["transfer_run"]["damping_centres"] = artifact
        validate_inference_config(source_run["config"], config, inference_definition)
        _validate_centre_adoption(source_run, config)
    experiment = config["selection"]["boundary_support"].get("experiment")
    if experiment is not None:
        from .diagnostics.selection import validate_reference_variant

        reference = load_run(experiment["reference"]["root"], allow_source=False)
        if not same_run_identity(experiment["reference"], _source_identity(reference)):
            raise ValueError("boundary reference identity changed")
        validate_reference_variant(reference, config)
        if source_run is None and workload != "boundary":
            raise ValueError("boundary variants must use their affected workload")
        proposed_root = Path(output_root).expanduser().resolve() / name
        if proposed_root.is_relative_to(reference["root"]):
            raise ValueError("boundary output must be outside the reference run")
    if workload not in config["analysis"]["workloads"]:
        raise ValueError("unknown configured workload")
    transfer = config["measurement"]["reconstruction"].get("transfer_run")
    if transfer is not None and workload != "transfer":
        raise ValueError("exclusive transfer snapshots require the transfer workload")
    statistics = "both"
    if config_path is not None:
        document = _read_yaml(config_path)
        statistics = document.get("cluster", {}).get("statistics", "both")
    if transfer is not None:
        statistics = "xi"
    code = _code_identity(allow_dirty=True) if scratch else _code_identity()
    registry_path = (
        DEFAULT_ANALYSIS_CONFIG_PATH.parent
        / config["fitting"]["covariance"]["shot_noise_rescaling"]["frozen_registry"]
    )
    if source_run is not None:
        registry_path = source_run["root"] / "frozen_alpha_sn.json"
    elif experiment is not None:
        registry_path = reference["root"] / "frozen_alpha_sn.json"
    registry_bytes = registry_path.read_bytes()
    registry = json.loads(registry_bytes)
    snapshot = {"analysis": config, "frozen_alpha_sn": registry}
    output_root = Path(output_root).expanduser().resolve()
    if scratch:
        output_root = output_root / "scratch" if "scratch" not in output_root.parts else output_root
    elif "scratch" in output_root.parts:
        raise ValueError("production output cannot use a scratch destination")
    root = output_root / name
    if source_run is not None and root.is_relative_to(source_run["root"]):
        raise ValueError("inference output must be outside the source run")
    if predecessor is not None:
        if (
            not same_identity(config, predecessor["config"])
            or registry_bytes != (predecessor["root"] / "frozen_alpha_sn.json").read_bytes()
            or analysis_config_sha256(snapshot) != predecessor["metadata"]["analysis_config_sha256"]
            or statistics != predecessor["metadata"]["statistics"]
        ):
            raise ValueError("successor definition differs from its predecessor's snapshot")
        if root.is_relative_to(predecessor["root"]) or predecessor["root"].is_relative_to(root):
            raise ValueError("successor output must be outside its predecessor")
    root.parent.mkdir(parents=True, exist_ok=True)
    # Array tasks share one snapshot; serialize only its create-only publication.
    with (root.parent / ("." + name + ".creation.lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if root.exists():
            raise FileExistsError(f"named run already exists: {root}")
        partial = root.with_name(root.name + "." + uuid.uuid4().hex + ".partial")
        partial.mkdir()
        (partial / "analysis.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        (partial / "frozen_alpha_sn.json").write_bytes(registry_bytes)
        if damping_centres is not None:
            (partial / "damping_centres.yaml").write_text(yaml.safe_dump(artifact, sort_keys=False))
        versions = {"joblib": importlib.metadata.version("joblib")}
        for entry in config["analysis"]["execution_environment"]["packages"]:
            name_ = entry["distribution"]
            try:
                versions[name_] = importlib.metadata.version(name_)
            except importlib.metadata.PackageNotFoundError:
                versions[name_] = None
        metadata = {
            "run_name": name,
            "interface_version": 2,
            "statistics": statistics,
            "mode": "scratch" if scratch else "production",
            "workload": workload,
            **code,
            "analysis_config_sha256": analysis_config_sha256(snapshot),
            "catalog_root": str(resolve_saved_path(catalog_root).resolve()),
            "versions": versions,
            "creation_command": list(sys.argv),
            "thread_environment": {
                key: os.environ.get(key)
                for key in config["analysis"]["execution_environment"][
                    "thread_environment_variables"
                ]
            },
        }
        if source_run is not None:
            metadata["source_run"] = _source_identity(source_run)
        if predecessor is not None:
            metadata["predecessor"] = _source_identity(predecessor)
        if inference_definition is not None:
            metadata["inference_definition"] = inference_definition
        if damping_centres is not None:
            metadata["damping_centres_file"] = "damping_centres.yaml"
        if providers:
            metadata["covariance_runs"] = [_source_identity(item) for item in providers]
        _write_yaml(partial / "run.yaml", metadata)
        # Every published directory is nonempty. rename cannot overwrite a competing complete run.
        os.rename(partial, root)
    return load_run(root)


def load_run(root, *, allow_source=True):
    root = resolve_saved_path(root).resolve()
    metadata = _read_yaml(root / "run.yaml")
    config = _read_yaml(root / "analysis.yaml")
    validate_analysis_config(config)
    registry = json.loads((root / "frozen_alpha_sn.json").read_bytes())
    digest = analysis_config_sha256({"analysis": config, "frozen_alpha_sn": registry})
    if digest != metadata.get("analysis_config_sha256"):
        raise ValueError("run snapshot or materialized scalar registry changed")
    if metadata.get("run_name") != root.name:
        raise ValueError("run directory and recorded name disagree")
    run = {"root": root, "metadata": metadata, "config": config, "registry": registry}
    if "damping_centres_file" in metadata:
        if metadata["damping_centres_file"] != "damping_centres.yaml" or not same_identity(
            _read_yaml(root / "damping_centres.yaml"),
            config["measurement"]["reconstruction"]["transfer_run"]["damping_centres"],
        ):
            raise ValueError("adopted damping artifact differs from the inference snapshot")
    if "source_run" in metadata:
        if not allow_source:
            raise ValueError("an inference source must be an original measurement run")
        source = load_run(metadata["source_run"]["root"], allow_source=False)
        if (
            not same_run_identity(metadata["source_run"], _source_identity(source))
            or metadata["workload"] != source["metadata"]["workload"]
            or not same_location(metadata["catalog_root"], source["metadata"]["catalog_root"])
        ):
            raise ValueError("inference source identity or snapshot changed")
        validate_inference_config(source["config"], config, metadata.get("inference_definition"))
        if not same_identity(registry, source["registry"]):
            raise ValueError("inference scalar registry differs from its source")
        run["source_run"] = source
        if "covariance_runs" in metadata:
            providers = _covariance_providers(
                source,
                metadata.get("inference_definition"),
                [entry["root"] for entry in metadata["covariance_runs"]],
            )
            if len(providers) != len(metadata["covariance_runs"]) or any(
                not same_run_identity(entry, _source_identity(item))
                for entry, item in zip(metadata["covariance_runs"], providers)
            ):
                raise ValueError("recorded covariance provider identity changed")
            run["covariance_runs"] = providers
    elif "inference_definition" in metadata:
        raise ValueError("an inference definition requires an original measurement source")
    if "predecessor" in metadata:
        # A successor adopts from an original run of the same definition; every load re-checks
        # that the predecessor still is that run, so changed or moved predecessors fail closed.
        predecessor = load_run(metadata["predecessor"]["root"], allow_source=False)
        if (
            "source_run" in metadata
            or not same_run_identity(metadata["predecessor"], _source_identity(predecessor))
            or predecessor["metadata"]["analysis_config_sha256"]
            != metadata["analysis_config_sha256"]
            or not same_identity(predecessor["config"], config)
            or predecessor["metadata"]["workload"] != metadata["workload"]
            or predecessor["metadata"].get("statistics") != metadata.get("statistics")
            or not same_location(metadata["catalog_root"], predecessor["metadata"]["catalog_root"])
            or (
                predecessor["metadata"].get("mode") == "scratch"
                and metadata.get("mode") != "scratch"
            )
        ):
            raise ValueError("predecessor identity or snapshot changed")
        run["predecessor"] = predecessor
    experiment = config["selection"]["boundary_support"].get("experiment")
    if experiment is not None:
        reference_root = resolve_saved_path(experiment["reference"]["root"]).resolve()
        reference_config = _read_yaml(reference_root / "analysis.yaml")
        if "experiment" in reference_config["selection"]["boundary_support"]:
            raise ValueError("boundary reference must be an original measurement run")
        reference = load_run(reference_root, allow_source=False)
        if not same_run_identity(experiment["reference"], _source_identity(reference)):
            raise ValueError("boundary reference identity changed")
        from .diagnostics.selection import validate_reference_variant

        validate_reference_variant(reference, config)
        run["boundary_reference"] = reference
    return run


def _budgeted_run(config):
    """Transfer, core sensitivity and density successor runs require storage admission.

    The per-bin primary run is admitted under the same budget whenever one is supplied; the
    campaign runner always supplies it. Plain CLI runs of the primary definition stay possible
    without a budget so that tests and bounded local probes need no campaign document.
    """
    recon = config["measurement"]["reconstruction"]
    return (
        "transfer_run" in recon
        or "sensitivity_run" in recon
        or "density_run" in config["selection"]
    )


def _covariance_providers(source, definition, roots):
    """Published BGS anisotropic covariances that a core power3 refit may read, never copy."""
    if not roots:
        return []
    if source is None or definition not in CORE_POWER3_INFERENCE:
        raise ValueError("covariance providers belong only to core power3 inference runs")
    providers = []
    for root in roots:
        provider = load_run(root)
        if provider["metadata"].get("inference_definition") != BGS_ANISOTROPIC or not (
            same_run_identity(provider["metadata"].get("source_run"), _source_identity(source))
        ):
            raise ValueError("a covariance provider must be a BGS anisotropic run of this source")
        providers.append(provider)
    if len({str(item["root"]) for item in providers}) != len(providers):
        raise ValueError("duplicate covariance provider")
    return providers


def routes_for(config, workload):
    controls = config["analysis"]["workloads"][workload]
    tracers = {
        item["name"]: tracer
        for tracer, spec in config["tracers"].items()
        for item in spec["measurement_bins"]
    }
    return [
        {
            "tracer": tracers[name],
            "measurement_bin": name,
            "condition": condition,
            "reconstruction_state": state,
        }
        for name in controls["measurement_bins"]
        for condition in controls["sigma_z_tiers"]
        for state in controls["reconstruction_states"]
    ]


def route_name(route):
    return "--".join(
        str(route[key]) for key in ("measurement_bin", "condition", "reconstruction_state")
    )


def comparison_routes(routes, config):
    """The routes of the paper's fit checks: one tier and state across the five bins."""
    scope = config["fitting"]["paper_comparisons"]
    return [
        route
        for route in routes
        if route["measurement_bin"] in scope["measurement_bins"]
        and route["condition"] == scope["condition"]
        and route["reconstruction_state"] == scope["reconstruction_state"]
    ]


def stage_units(stage, config, workload, *, inference_definition=None):
    """Derive the workload from the snapshot; a sampling unit owns all four ensembles."""
    routes = routes_for(config, workload)
    if inference_definition is not None:
        if inference_definition in TRANSFER_INFERENCE:
            if stage not in {"fit", "sample"}:
                raise ValueError("transfer inference reuses measured vectors and covariances")
            high = config["measurement"]["reconstruction"]["exploratory_transfer"]["source_class"]
            # The flat headline and the Gaussian check fit every route (revision 2.19); the
            # PCS2 check fits the low-precision post-reconstruction routes.
            routes = (
                routes
                if inference_definition in ("transfer_reference", "transfer_flat")
                else [
                    route
                    for route in routes
                    if route["condition"] != high
                    and route["reconstruction_state"] == "postreconstruction"
                ]
            )
        elif inference_definition in UNIFIED_PRIOR_INFERENCE:
            # The unified prior refit fits every route of its source and its two checks the
            # primary's comparison routes (revision 2.20).
            if stage not in {"fit", "sample"}:
                raise ValueError("the unified prior refit reuses measured vectors and covariances")
            if inference_definition != UNIFIED_PRIOR:
                routes = comparison_routes(routes, config)
        elif inference_definition in CORE_INFERENCE:
            if stage not in {"fit", "sample"}:
                raise ValueError("core power3 refits reuse measured vectors and covariances")
            # The primary refit and the scan centre refits (revision 2.18) keep every route.
            if inference_definition not in ("primary_power3_v1", *CORE_SCAN_INFERENCE):
                routes = comparison_routes(routes, config)
        elif inference_definition != BGS_ANISOTROPIC or stage not in {
            "covariance",
            "fit",
            "sample",
        }:
            raise ValueError("BGS inference publishes only combined covariances, fits and samples")
        else:
            routes = [route for route in routes if route["tracer"] == "BGS_BRIGHT-21.5"]
        if stage == "covariance":
            return [{"unit_id": route_name(route), **route} for route in routes]
    identifiers = list(config["analysis"]["workloads"][workload]["mock_ids"])
    if stage in {"catalog", "select"}:
        return [
            {
                "unit_id": tracer,
                "tracer": tracer,
                "realization_ids": list(config["analysis"]["mock_ids"]),
            }
            for tracer in dict.fromkeys(route["tracer"] for route in routes)
        ]
    if stage in {"reconstruct", "cluster"}:
        parents = list(dict.fromkeys((route["tracer"], route["condition"]) for route in routes))
        return [
            {
                "unit_id": f"{tracer}--{condition}--mock-{identifier}",
                "tracer": tracer,
                "condition": condition,
                "realization_id": identifier,
                "measurement_bins": list(
                    dict.fromkeys(
                        route["measurement_bin"]
                        for route in routes
                        if route["tracer"] == tracer and route["condition"] == condition
                    )
                ),
            }
            for tracer, condition in parents
            for identifier in identifiers
        ]
    if stage == "covariance":
        return [
            {"unit_id": route_name(route) + "--" + cap, **route, "cap": cap}
            for route in routes
            for cap in ("NGC", "SGC")
        ]
    if stage == "fit":
        return [
            {
                "unit_id": route_name(route)
                + "--"
                + ("mean" if identifier is None else f"mock-{identifier}"),
                **route,
                "realization_id": identifier,
            }
            for route in routes
            for identifier in [*identifiers, None]
        ]
    if stage == "sample":
        return [
            {
                "unit_id": route_name(route),
                **route,
                "ensemble_count": config["fitting"]["sampling"]["ensembles"],
            }
            for route in routes
        ]
    raise ValueError(f"stage has no execution units: {stage}")


def _selected_units(units, selection):
    if selection is None:
        return units
    # A batched array task names its units comma-separated; unit names never contain commas.
    if "," in selection:
        chosen = [unit for part in selection.split(",") for unit in _selected_units(units, part)]
        if len({unit["unit_id"] for unit in chosen}) != len(chosen):
            raise ValueError("a unit is selected more than once")
        return chosen
    if selection.isdecimal():
        index = int(selection)
        if index >= len(units):
            raise ValueError("unit index is outside the configured workload")
        return [units[index]]
    selected = [unit for unit in units if unit["unit_id"] == selection]
    if not selected:
        raise ValueError(f"unit is absent from the configured workload: {selection}")
    return selected


def _configure_threads(stage, cpus):
    """Limit libraries before CLI numerical imports; ensembles share this allocation."""
    allocated = os.environ.get("SLURM_CPUS_PER_TASK")
    if allocated is not None and cpus > int(allocated):
        raise click.UsageError("--cpus exceeds the current Slurm allocation")
    os.environ["NTHREADS"] = str(cpus)
    os.environ["OMP_NUM_THREADS"] = (
        "1" if stage in {"catalog", "select", "covariance"} else str(cpus)
    )
    for key in (
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[key] = "1"


def slurm_plan(
    stage,
    units,
    *,
    input_path,
    cpus,
    memory,
    walltime,
    account,
    partition=None,
    run=None,
    inference_run=None,
    boundary_experiment=False,
    inference_definition=None,
    config_path=None,
    statistics=None,
    save_catalogs=True,
    source_group=False,
    peer_inputs=(),
    reuse_selection=None,
    max_concurrent=1,
    resume=False,
    damping_centres=None,
    storage_budget=None,
    storage_outputs=(),
    transfer=False,
    after=(),
    covariance_runs=(),
    review=True,
    units_per_task=1,
):
    """Render an array using exactly the same unit handler as local execution; never submit.

    With ``units_per_task`` above one, each array task runs a contiguous slice of the units, in
    order, in one process: one storage admission and one handoff publication per slice.
    """
    if any(not str(job).replace("_", "").isdecimal() for job in after):
        raise ValueError("--after requires numeric Slurm job IDs (optionally JOB_TASK)")
    if not units or cpus < 1 or not memory or not walltime:
        raise ValueError("scheduler planning requires units and a positive allocated CPU count")
    if boundary_experiment and len(units) != 1:
        raise ValueError("boundary jobs require one explicit --unit; do not build a queued backlog")
    if not account:
        raise ValueError("scheduler planning requires an explicit --account")
    resources = [memory, walltime, account] + ([] if partition is None else [partition])
    if any(
        not value or any(character.isspace() for character in str(value)) for value in resources
    ):
        raise ValueError("scheduler resource fields must be single tokens")
    if type(max_concurrent) is not int or max_concurrent < 1:
        raise ValueError("scheduler planning requires a positive concurrency bound")
    if type(units_per_task) is not int or units_per_task < 1:
        raise ValueError("scheduler planning requires a positive units-per-task count")
    if units_per_task > 1 and source_group:
        raise ValueError("source groups cannot share an array task")
    storage_plan = None
    if transfer or source_group or reuse_selection is not None or storage_budget is not None:
        # Budgeted run kinds and selection reuse (a campaign-only path that can write hundreds
        # of gigabytes of regenerated tiers) require admission; any supplied budget bounds the
        # plan as well, which is how a campaign admits the per-bin primary run.
        if storage_budget is None:
            raise ValueError("transfer planning requires --storage-budget")
        storage_plan = inspect_transfer_storage(
            _read_yaml(storage_budget),
            stage=stage,
            outputs=storage_outputs,
            requested_concurrency=max_concurrent,
            unit_count=units_per_task,
        )
        max_concurrent = storage_plan["max_concurrent"]
    args = [sys.executable, "-m", "pzbao.run", stage, "--cpus", str(cpus)]
    if storage_budget is not None:
        args.extend(["--storage-budget", str(Path(storage_budget).resolve())])
    if resume:
        args.append("--resume")
    if stage == "sample" and not review:
        args.append("--no-review")
    for peer in peer_inputs:
        args.extend(["--peer-input", str(Path(peer).resolve())])
    if reuse_selection is not None:
        args.extend(["--reuse-selection", str(Path(reuse_selection).resolve())])
    if damping_centres is not None:
        args.extend(["--damping-centres", str(Path(damping_centres).resolve())])
    for provider in covariance_runs:
        args.extend(["--covariance-run", str(Path(provider).resolve())])
    if stage == "catalog":
        args.extend(
            [
                "--run-name",
                run["run_name"],
                "--output-root",
                str(Path(run["output_root"]).resolve()),
                "--catalog-root",
                str(Path(run["catalog_root"]).resolve()),
                "--workload",
                run["workload"],
            ]
        )
        if run.get("config_path") is not None:
            args.extend(["--config", str(Path(run["config_path"]).resolve())])
        if run.get("predecessor") is not None:
            args.extend(["--predecessor", str(Path(run["predecessor"]).resolve())])
        if run.get("scratch"):
            args.append("--scratch")
        if run.get("download"):
            args.append("--download")
    else:
        args.extend(["--input", str(Path(input_path).resolve())])
        if config_path is not None:
            args.extend(["--config", str(Path(config_path).resolve())])
        if stage == "cluster" and statistics is not None:
            args.extend(["--statistics", statistics])
        if stage == "reconstruct" and not save_catalogs:
            args.append("--no-save-catalogs")
    if inference_run is not None:
        if stage != "fit" and not (
            stage == "covariance" and inference_definition == BGS_ANISOTROPIC
        ):
            raise ValueError("inference creation requires fit or BGS covariance reprojection")
        args.extend(["--inference-run", str(Path(inference_run).resolve())])
    if inference_definition is not None:
        if inference_run is None or not (
            (stage == "covariance" and inference_definition == BGS_ANISOTROPIC)
            or (
                stage == "fit"
                and inference_definition
                in (*TRANSFER_INFERENCE, *CORE_INFERENCE, *UNIFIED_PRIOR_INFERENCE)
            )
        ):
            raise ValueError("inference definition requires its matching stage and destination")
        args.extend(["--inference-definition", inference_definition])
    # Unit names preserve selected subsets; the job itself expands the deferred array index.
    selections = (
        "unit_ids=(" + " ".join(shlex.quote(_safe_name(unit["unit_id"])) for unit in units) + ")"
    )
    selector = "--source-group" if source_group else "--unit"
    task_units = '"${unit_ids[$SLURM_ARRAY_TASK_ID]}"'
    if units_per_task > 1:
        # The task's slice of unit_ids, comma-joined.
        offset = f"$((SLURM_ARRAY_TASK_ID * {units_per_task}))"
        task_units = f'"$(IFS=,; echo "${{unit_ids[*]:{offset}:{units_per_task}}}")"'
    command = shlex.join(args) + " " + selector + " " + task_units
    tasks = -(-len(units) // units_per_task)
    ensemble_count = sum(unit.get("ensemble_count", 0) for unit in units)
    return "\n".join(
        [
            "#!/bin/bash",
            f"#SBATCH --job-name=bao-{stage}",
            f"#SBATCH --array=0-{tasks - 1}%{max_concurrent}",
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks=1",
            f"#SBATCH --cpus-per-task={cpus}",
            f"#SBATCH --mem={memory}",
            f"#SBATCH --time={walltime}",
            f"#SBATCH --account={account}",
            "#SBATCH --mail-type=END,FAIL",
            *([f"#SBATCH --partition={partition}"] if partition else []),
            "#SBATCH --no-requeue",
            *(
                [
                    f"#SBATCH --dependency=afterok:{':'.join(after)}",
                    "#SBATCH --kill-on-invalid-dep=yes",
                ]
                if after
                else []
            ),
            "set -euo pipefail",
            *(
                []
                if storage_plan is None
                else [
                    f"# Storage peak bound: {storage_plan['conservative_peak_bytes']} bytes; "
                    f"limit: {storage_plan['storage_limit_bytes']} bytes; admission rechecked at "
                    "execution."
                ]
            ),
            f"# {len(units)} units{f' in {tasks} tasks' if units_per_task > 1 else ''}; "
            f"{ensemble_count} sampler ensembles in total.",
            "# Ensembles run sequentially inside each route unit; CPUs are not multiplied by four.",
            f"cd {shlex.quote(str(Path(__file__).resolve().parents[1]))}",
            f"export NTHREADS={cpus}",
            f"export OMP_NUM_THREADS={1 if stage in {'catalog', 'select', 'covariance'} else cpus}",
            "export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1",
            selections,
            command,
            "",
        ]
    )


def _validate_products(directory, result):
    products = result.get("products")
    if not isinstance(products, list) or not products:
        raise ValueError("a completed unit must declare readable scientific products")
    for product in products:
        path = (directory / product["path"]).resolve()
        if (
            not path.is_relative_to(directory.resolve())
            or not path.is_file()
            or path.stat().st_size == 0
        ):
            raise ValueError(f"missing or invalid scientific product: {product['path']}")
        format_ = product["format"]
        if format_ == "json":
            json.loads(path.read_text())
        elif format_ == "yaml":
            _read_yaml(path)
        elif format_ == "npz":
            import numpy as np

            with np.load(path, allow_pickle=False) as archive:
                for key in product["arrays"]:
                    values = archive[key]
                    if not values.size or not np.all(np.isfinite(values)):
                        raise ValueError(f"required array {key} is empty or non-finite")
        elif format_ == "catalogue":
            from .catalog import read_catalogue

            read_catalogue(path, identity=product["identity"])
        elif format_ == "pycorr":
            from pycorr import TwoPointCorrelationFunction
            from .clustering import pair_count_identity

            estimator = TwoPointCorrelationFunction.load(str(path))
            if not same_identity(pair_count_identity(estimator), product["identity"]):
                raise ValueError("native pair-count product identity changed")
        elif format_ == "profiles":
            from .fitting import read_profiles

            read_profiles(path, identity=product["identity"])
        elif format_ == "chain":
            from .sampling import read_chain

            read_chain(path, identity=product["identity"])
        elif format_ == "rascalc_components":
            import numpy as np

            with np.load(path, allow_pickle=False) as archive:
                dimension, count = product["dimension"], product["subsamples"]
                for key in ("c2", "c3", "c4"):
                    if archive[key].shape != (dimension, dimension) or archive[
                        key + "_samples"
                    ].shape != (count, dimension, dimension):
                        raise ValueError("raw covariance component archive is incomplete")
        elif format_ not in {"txt", "csv", "png", "pdf"}:
            raise ValueError(f"unrecognized scientific product format {format_!r}")


def _unit_subdirectory(run, stage):
    if stage == "covariance" and run["metadata"].get("inference_definition") == BGS_ANISOTROPIC:
        return "routes"
    return "units"


def _unit_result(run, stage, unit, *, validate_products=True):
    directory = run["root"] / stage / _unit_subdirectory(run, stage) / unit["unit_id"]
    if not directory.exists():
        return None
    record = _read_yaml(directory / "result.yaml")
    if (
        not same_identity(record.get("unit"), unit)
        or not same_identity(record.get("source_run"), run["metadata"].get("source_run"))
        or record.get("stage") != stage
        or any(
            record.get(key) != run["metadata"][key]
            for key in ("run_name", "code_commit", "analysis_config_sha256")
        )
    ):
        raise ValueError("published unit identity or snapshot differs from this request")
    if validate_products:
        _validate_products(directory, record)
    return record


_IDENTITY_KEYS = ("run_name", "code_commit", "analysis_config_sha256")
#: Result fields this driver writes at publication; an adopted unit never copies them.
_DRIVER_RESULT_KEYS = frozenset(
    {
        *_IDENTITY_KEYS,
        "unit",
        "stage",
        "execution_code",
        "stage_options",
        "runtime",
        "source_run",
        "source_group",
        "registered_from",
    }
)


def _predecessor_handoff(run, stage):
    """The recorded predecessor's published handoff of this stage, verified to be its own."""
    predecessor = run.get("predecessor")
    if predecessor is None:
        raise ValueError("adoption requires a run created with --predecessor")
    path = predecessor["root"] / stage / HANDOFFS[stage]
    if not path.is_file():
        raise ValueError(
            f"predecessor {predecessor['metadata']['run_name']} has not published {stage}"
        )
    handoff = _read_yaml(path)
    if (
        handoff.get("stage") != stage
        or (path.parent / handoff["run"]).resolve() != predecessor["root"]
        or any(
            handoff.get(key) != predecessor["metadata"][key]
            for key in ("run_name", "analysis_config_sha256")
        )
    ):
        raise ValueError(f"handoff is not the predecessor's published {stage} handoff: {path}")
    return path, handoff, predecessor


def _reidentified(identity, source, target):
    """Replace the predecessor's own identity triple; any other embedded identity is refused."""
    if not isinstance(identity, dict) or any(
        identity.get(key) != source[key] for key in _IDENTITY_KEYS
    ):
        raise ValueError("adopted product embeds an identity that is not the predecessor's")
    return {**identity, "run_name": target["run_name"], "code_commit": target["code_commit"]}


def _retired_runs(run, source, siblings):
    """Predecessor root -> (predecessor, successor) for this run and every declared sibling.

    A sibling is another successor of the same campaign, loaded with its own recorded
    predecessor. A product reference into any of these predecessors is redirected into the
    successor that adopted it; a reference into any other run must still resolve as recorded.
    """
    retired = {source["root"]: (source, run)}
    for sibling in siblings:
        predecessor = sibling.get("predecessor")
        if predecessor is None:
            raise ValueError(
                f"sibling {sibling['metadata']['run_name']} was not created with --predecessor"
            )
        if sibling["root"] == run["root"] or predecessor["root"] in retired:
            raise ValueError("siblings must be other successors with other predecessors")
        retired[predecessor["root"]] = (predecessor, sibling)
    return retired


def _sidecar(path):
    """The saved catalogue metadata beside a product, or nothing when the product is gone."""
    sidecar = Path(path).with_suffix(".json")
    return json.loads(sidecar.read_text()) if sidecar.is_file() else {}


def _redirected_base(base, retired):
    """A base inside a retired predecessor becomes the adopted product of its successor.

    The adopted sidecar must already carry the successor's identity and record adoption from
    that very predecessor, so a sibling's selection is adopted before the aliases of it are. A
    base outside every retired run (an earlier campaign's immutable catalogue) must still
    exist under exactly the recorded identity.
    """
    path = Path(base["path"])
    for root, (predecessor, successor) in retired.items():
        if not path.is_relative_to(root):
            continue
        target = successor["root"] / path.relative_to(root)
        identity = _reidentified(base["identity"], predecessor["metadata"], successor["metadata"])
        metadata = _sidecar(target)
        if not same_identity(metadata.get("identity"), identity) or metadata.get(
            "registered_from"
        ) != _source_identity(predecessor):
            raise ValueError(
                f"{successor['metadata']['run_name']} has not adopted the base the predecessor "
                f"references: {target}"
            )
        return {"path": str(target), "identity": identity}
    if not same_identity(_sidecar(path).get("identity"), base["identity"]):
        raise ValueError(f"base outside the predecessor is missing or re-identified: {path}")
    return base


def _redirected_run_identity(entry, retired):
    """A run identity record naming a retired predecessor names its successor instead."""
    for root, (predecessor, successor) in retired.items():
        if entry.get("run_name") != predecessor["metadata"]["run_name"]:
            continue
        if (
            not same_run_identity(entry, _source_identity(predecessor))
            or resolve_saved_path(entry["root"]).resolve() != root
        ):
            raise ValueError("adopted document names a retired run under another identity")
        return _source_identity(successor)
    return entry


def _refuse_retired_references(document, retired, label):
    """No adopted document may still name a retired run by path or identity triple.

    ``registered_from`` is the one field that records where a product came from.
    """
    roots = [str(root) for root in retired]
    identities = [
        {key: predecessor["metadata"][key] for key in _IDENTITY_KEYS}
        for predecessor, _ in retired.values()
    ]

    def walk(value):
        if isinstance(value, dict):
            if all(key in value for key in _IDENTITY_KEYS) and any(
                all(value[key] == identity[key] for key in _IDENTITY_KEYS)
                for identity in identities
            ):
                raise ValueError(f"adopted document still carries a retired identity: {label}")
            for key, item in value.items():
                if key != "registered_from":
                    walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
        elif isinstance(value, str) and any(
            value == root or value.startswith(root + "/") for root in roots
        ):
            raise ValueError(f"adopted document still names a retired run: {label}")

    walk(document)


def _redirected_document(document, retired, source_identity, target_identity):
    """Re-identify and redirect the references a stage document carries; True when changed.

    Catalogue listings (preparation.json, reconstruction.json) carry the predecessor's
    identities; a reused selection names the run it aliased (``selection_source``); clustering
    records reference their randoms by absolute path and identity
    (``measurement_random_reference_v1``), which must resolve inside the adopted products.
    """
    changed = False
    if "catalogues" in document:
        document["catalogues"] = [
            {**item, "identity": _reidentified(item["identity"], source_identity, target_identity)}
            for item in document["catalogues"]
        ]
        changed = True
    if "selection_source" in document:
        redirected = _redirected_run_identity(document["selection_source"], retired)
        changed = changed or redirected != document["selection_source"]
        document["selection_source"] = redirected
    for measurement in document.get("measurements", []):
        randoms = measurement.get("randoms")
        if randoms is None or isinstance(randoms, str):
            continue  # no randoms, or an archive inside the unit that moves with it
        if randoms.get("representation") != "measurement_random_reference_v1":
            raise ValueError("adopted measurement randoms use an unsupported reference")
        measurement["randoms"] = {**randoms, **_redirected_base(randoms, retired)}
        changed = True
    return changed


def _adopted_stage_options(run, stage):
    """The options a plain --resume execution of the stage records; adopted units carry them."""
    if stage == "catalog":
        return {"download": False}
    return _resolve_stage_options(run, stage, None, None, True)


def _same_filesystem(source, destination):
    return os.stat(source).st_dev == os.stat(destination).st_dev


def _already_adopted(run, stage, unit, record, source_dir, destination, registered):
    """An existing destination is accepted only as this unit's completed earlier adoption.

    The destination result must carry this run's identity and a ``registered_from`` naming
    exactly this predecessor unit, handoff and result digests, every adopted product must be
    present and every predecessor product gone; anything else at the destination refuses.
    """
    try:
        adopted = _unit_result(run, stage, unit, validate_products=False)
        earlier = adopted.get("registered_from") or {}
        complete = all(earlier.get(key) == value for key, value in registered.items()) and all(
            (destination / product["path"]).is_file() for product in adopted["products"]
        )
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise FileExistsError(
            f"adoption destination already exists: {destination} ({exc})"
        ) from exc
    if not complete or any(
        (source_dir / product["path"]).exists() for product in record["products"]
    ):
        raise FileExistsError(
            f"adoption destination already exists and is not this unit's completed adoption: "
            f"{destination}"
        )


def _adoption(stage, run, unit, handoff_path, handoff, source, handoff_sha256, retired):
    """Plan one unit's adoption without writing anything.

    None means the predecessor never published the unit, so the stage computes it here.
    Otherwise the plan names the product files to move (one filesystem), the catalogue
    sidecars and stage documents to rewrite under this run's identity with their references
    redirected to the adopted products of this run or of a declared sibling, and the result
    to publish. Every embedded identity must be the predecessor's own, and no adopted
    document may still name a retired run. A destination that already holds this unit's
    completed adoption (an earlier invocation stopped before its receipt) is kept as is.
    Nothing of the predecessor is read beyond its published unit and handoff; its attempts
    are never consulted.
    """
    rows = [row for row in handoff["units"] if row["unit"]["unit_id"] == unit["unit_id"]]
    if not rows:
        return None
    if len(rows) != 1 or not same_identity(rows[0]["unit"], unit):
        raise ValueError("predecessor published a different unit under this name")
    record = _unit_result(source, stage, unit, validate_products=False)
    if record is None:
        raise ValueError("predecessor handoff lists a unit it has not published")
    source_dir = (
        source["root"] / stage / _unit_subdirectory(source, stage) / unit["unit_id"]
    ).resolve()
    if (handoff_path.parent / rows[0]["path"]).resolve() != source_dir / "result.yaml":
        raise ValueError("predecessor handoff points away from its published unit")
    destination = run["root"] / stage / _unit_subdirectory(run, stage) / unit["unit_id"]
    registered = {
        **_source_identity(source),
        "stage": stage,
        "unit_path": rows[0]["path"],
        "handoff": str(handoff_path),
        "handoff_sha256": handoff_sha256,
        "result_sha256": hashlib.sha256((source_dir / "result.yaml").read_bytes()).hexdigest(),
    }
    plan = {"unit": unit, "source_dir": source_dir, "destination": destination}
    if destination.exists():
        _already_adopted(run, stage, unit, record, source_dir, destination, registered)
        return {**plan, "moves": [], "documents": {}, "bytes": 0, "result": None, "adopted": True}
    if not _same_filesystem(source_dir, run["root"]):
        raise ValueError("adoption moves products within one filesystem; this run lies on another")
    source_identity, target_identity = source["metadata"], run["metadata"]
    moves, documents, products = [], {}, [None] * len(record["products"])

    def origin_of(product):
        origin = (source_dir / product["path"]).resolve()
        if (
            not origin.is_relative_to(source_dir)
            or not origin.is_file()
            or not origin.stat().st_size
        ):
            raise ValueError(f"missing or invalid predecessor product: {product['path']}")
        return origin

    # Catalogue products first: their re-identified sidecars replace the predecessor's own.
    for index, product in enumerate(record["products"]):
        if product["format"] != "catalogue":
            continue
        origin = origin_of(product)
        sidecar = origin.with_suffix(".json")
        saved = sidecar.read_bytes()
        metadata = json.loads(saved)
        if not same_identity(metadata.get("identity"), product["identity"]):
            raise ValueError("predecessor catalogue sidecar identity differs from its product")
        identity = _reidentified(product["identity"], source_identity, target_identity)
        metadata["identity"] = identity
        if metadata.get("base") is not None:
            metadata["base"] = _redirected_base(metadata["base"], retired)
        metadata["registered_from"] = _source_identity(source)
        _refuse_retired_references(metadata, retired, product["path"])
        relative = sidecar.relative_to(source_dir)
        documents[relative] = (saved, json.dumps(metadata, indent=2, allow_nan=False))
        moves.append(relative)
        if origin != sidecar:
            moves.append(origin.relative_to(source_dir))
        products[index] = {**product, "identity": identity}
    for index, product in enumerate(record["products"]):
        if products[index] is not None:
            continue
        origin = origin_of(product)
        products[index] = dict(product)
        relative = origin.relative_to(source_dir)
        if relative in moves:
            continue  # the sidecar of a displaced shard, re-identified above
        if product["format"] == "json":
            document = json.loads(origin.read_text())
            if isinstance(document, dict) and _redirected_document(
                document, retired, source_identity, target_identity
            ):
                documents[relative] = (
                    origin.read_bytes(),
                    json.dumps(document, indent=2, allow_nan=False),
                )
            _refuse_retired_references(document, retired, product["path"])
        moves.append(relative)
    result = {key: value for key, value in record.items() if key not in _DRIVER_RESULT_KEYS}
    result["products"] = products
    result["registered_from"] = {
        **registered,
        "execution_code": record.get("execution_code"),
        "stage_options": record.get("stage_options"),
        "registered_from": record.get("registered_from"),
        "files": {"moved": len(moves)},
    }
    return {
        **plan,
        "moves": moves,
        "documents": documents,
        "bytes": sum((source_dir / relative).stat().st_size for relative in moves),
        "result": result,
        "adopted": False,
    }


def _adopt_unit(run, stage, plan, code):
    """Move one planned unit into this run; a failure moves every file back before raising."""
    unit, source_dir = plan["unit"], plan["source_dir"]
    stage_root = run["root"] / stage
    attempt = stage_root / "attempts" / (unit["unit_id"] + "." + uuid.uuid4().hex)
    attempt.mkdir()
    moved, rewritten = [], set()
    started = time.monotonic()
    try:
        for relative in plan["moves"]:
            target = attempt / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            os.rename(source_dir / relative, target)
            moved.append(relative)
        for relative, (_, text) in plan["documents"].items():
            rewritten.add(relative)
            (attempt / relative).write_text(text)
        result = dict(plan["result"])
        result["runtime"] = {
            "adopted_at": datetime.datetime.now().astimezone().isoformat(),
            "elapsed_seconds": time.monotonic() - started,
        }
        result["execution_code"] = code
        result["stage_options"] = _adopted_stage_options(run, stage)
        result.update(
            unit=unit,
            stage=stage,
            **{key: run["metadata"][key] for key in _IDENTITY_KEYS},
        )
        _write_yaml(attempt / "result.yaml", result)
        os.rename(attempt, plan["destination"])
    except BaseException:
        for relative in reversed(moved):
            target = attempt / relative
            if relative in rewritten:
                target.write_bytes(plan["documents"][relative][0])
            os.rename(target, source_dir / relative)
        shutil.rmtree(attempt, ignore_errors=True)
        raise


def adopt_predecessor(run, stage, *, dry_run=False, siblings=()):
    """Move the recorded predecessor's published units of one stage into this run.

    Every unit the predecessor's handoff lists is adopted: its product files are moved
    (``os.rename``, one filesystem), catalogue sidecars and the stage documents are rewritten
    under this run's identity with their references redirected to the adopted products of
    this run or of a declared sibling successor, and the unit result is published under this
    run's identity with ``registered_from`` naming the predecessor, its handoff and result
    digests and its execution code. Units the predecessor never published stay absent, so the
    stage computes them on ``--resume``; that execution also validates every adopted product
    natively before it publishes the handoff. Everything is checked before the first move
    under one stage lock, a failed unit is moved back, a unit an interrupted invocation
    already adopted is recognized and kept, and the stage receipt is written once, after the
    last unit.
    """
    if stage not in REGISTERED_STAGES:
        raise ValueError(f"{stage} units are computed, never adopted")
    code = (
        _code_identity(allow_dirty=True)
        if run["metadata"].get("mode") == "scratch"
        else _code_identity()
    )
    if code["code_commit"] != run["metadata"]["code_commit"]:
        raise ValueError("execution checkout differs from this run's recorded code commit")
    handoff_path, handoff, source = _predecessor_handoff(run, stage)
    retired = _retired_runs(run, source, siblings)
    stage_root = run["root"] / stage
    receipt = stage_root / ADOPTION_RECEIPT
    digest = hashlib.sha256(handoff_path.read_bytes()).hexdigest()

    def planned():
        if receipt.exists():
            raise FileExistsError(f"{stage} has already adopted its predecessor's units: {receipt}")
        plans, computed = [], []
        for unit in stage_units(stage, run["config"], run["metadata"]["workload"]):
            plan = _adoption(stage, run, unit, handoff_path, handoff, source, digest, retired)
            if plan is None:
                computed.append(unit["unit_id"])
            else:
                plans.append(plan)
        summary = {
            "stage": stage,
            "predecessor": _source_identity(source),
            "siblings": [_source_identity(sibling) for _, sibling in retired.values()][1:],
            "handoff": str(handoff_path),
            "handoff_sha256": digest,
            "units_adopted": [plan["unit"]["unit_id"] for plan in plans],
            "units_previously_adopted": [p["unit"]["unit_id"] for p in plans if p["adopted"]],
            "units_computed": computed,
            "files_moved": sum(len(plan["moves"]) for plan in plans),
            "bytes_moved": sum(plan["bytes"] for plan in plans),
        }
        return plans, summary

    if dry_run:
        return planned()[1]
    with _exclusive_lock(stage_root / ".locks" / "adopt.lock", f"an adoption of {stage}"):
        plans, summary = planned()
        started = time.monotonic()
        (stage_root / _unit_subdirectory(run, stage)).mkdir(parents=True, exist_ok=True)
        (stage_root / "attempts").mkdir(exist_ok=True)
        for plan in plans:
            if plan["adopted"]:
                continue
            with _unit_lock(run, stage, plan["unit"]):
                _adopt_unit(run, stage, plan, code)
        summary.update(
            execution_code=code,
            adopted_at=datetime.datetime.now().astimezone().isoformat(),
            elapsed_seconds=time.monotonic() - started,
        )
        with receipt.open("x") as handle:
            json.dump(summary, handle, indent=2)
    return summary


def _science(stage, run, unit, attempt, cpus):
    options = run.get("stage_options", {})
    if stage == "catalog":
        from .catalog import catalog_stage

        return catalog_stage(run, unit, attempt, download=options.get("download", False), cpus=cpus)
    if stage == "select":
        from .selection import select_stage, reuse_selection_stage

        if "reuse_selection" in options:
            source = _run_from_handoff(options["reuse_selection"], "reconstruct")
            return reuse_selection_stage(run, unit, attempt, source=source, cpus=cpus)
        return select_stage(run, unit, attempt, cpus=cpus)
    if stage == "reconstruct":
        from .reconstruction import reconstruct_stage

        return reconstruct_stage(
            run, unit, attempt, save_catalogs=options.get("save_catalogs", True)
        )
    if stage == "cluster":
        from .clustering import cluster_stage

        return cluster_stage(
            run,
            unit,
            attempt,
            nthreads=cpus,
            statistics=options.get("statistics", run["metadata"].get("statistics", "both")),
        )
    if stage == "covariance":
        if "source_run" in run:
            raise ValueError("an inference run must not launch cap integrations")
        return _run_cap_attempt(run, unit, attempt, cpus=cpus)
    if stage == "fit":
        from .fitting import fit_unit

        return fit_unit(run, unit, attempt)
    if stage == "sample":
        from .sampling import sample_route

        return sample_route(run, unit, attempt)
    raise ValueError("unknown scientific stage")


@contextmanager
def _exclusive_lock(path, owner):
    """One nonblocking exclusive lock on a file; a second holder fails instead of waiting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another writer owns {owner}") from exc
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _unit_lock(run, stage, unit):
    """One nonblocking writer lock per result identity for ordinary and grouped execution."""
    directory = run["root"] / stage / ".locks"
    return _exclusive_lock(directory / (_safe_name(unit["unit_id"]) + ".lock"), "this target unit")


def source_groups(runs):
    """Group sibling targets by source/radius/mock; array selectors do not claim compatibility."""
    groups, seen = {}, set()
    for run in runs:
        root = str(run["root"].resolve())
        if root in seen:
            raise ValueError("duplicate peer selection input")
        seen.add(root)
        config = run["config"]
        recon = config["measurement"]["reconstruction"]
        if "transfer_run" not in recon or "source_run" in run:
            raise ValueError("source groups require original transfer selection runs")
        radius = recon["smoothing_values_mpc_h"][0]
        for unit in stage_units("reconstruct", config, run["metadata"]["workload"]):
            source = recon["transfer_run"]["source_class"]
            source = unit["condition"] if source == "self" else source
            selector = (
                f"{unit['tracer']}--source-{source}--R{radius:g}--mock-{unit['realization_id']}"
            )
            group = groups.setdefault(selector, {"unit_id": selector, "targets": []})
            group["targets"].append({"run": run, "unit": unit})
    return list(groups.values())


def planned_source_groups(runs, selection, tracer=None):
    """Select groups for a plan; a tracer filter lets each tracer depend on its own selection."""
    groups = source_groups(runs)
    if selection != "all":
        return _selected_units(groups, selection)
    return [group for group in groups if tracer is None or _group_tracer(group) == tracer]


def _group_tracer(group):
    return group["targets"][0]["unit"]["tracer"]


@contextmanager
def _transfer_writer(runs, stage, *, unit_count=1, extra_outputs=()):
    """Require the same operational budget for every sibling; do not alter run snapshots.

    A supplied budget is enforced for any run, so a campaign admits its per-bin primary run
    through the same accounting; only budgeted run kinds refuse to execute without one.
    """
    budgets = [run.get("storage_budget") for run in runs]
    if not any(_budgeted_run(run["config"]) for run in runs) and all(
        budget is None for budget in budgets
    ):
        yield None
        return
    if any(budget is None for budget in budgets):
        raise ValueError("transfer execution requires --storage-budget")
    if any(not same_identity(budget, budgets[0]) for budget in budgets[1:]):
        raise ValueError("peer runs require the same campaign storage budget")
    with transfer_storage_writer(
        budgets[0],
        stage=stage,
        outputs=[run["root"] for run in runs] + list(extra_outputs),
        unit_count=unit_count,
    ) as receipt:
        yield receipt


#: Polling cadence and progress interval of the stage's coverage lock (``_handoff_lock``).
HANDOFF_POLL_SECONDS = 2.0
HANDOFF_PROGRESS_SECONDS = 300.0


@contextmanager
def _handoff_lock(path):
    """Hold the stage's coverage lock, polling for it rather than blocking.

    On the cluster's NFS storage (NFSv3 with ``local_lock=none``) a blocking flock waits for the network lock
    manager's grant: on 2026-10-21 finished fits waited a median 636 s (BGS) for this lock,
    whose holder rebuilds the handoff in about a second. Storage admission polls the same way
    and reports its waits; a wait here is reported on stderr too.
    """
    started = noted = time.monotonic()
    waited = False
    with path.open("a") as stream:
        while True:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                waited = True
                now = time.monotonic()
                if now - noted >= HANDOFF_PROGRESS_SECONDS:
                    click.echo(f"handoff: {path} held; waited {now - started:.0f} s", err=True)
                    noted = now
                time.sleep(HANDOFF_POLL_SECONDS)
        if waited:
            click.echo(
                f"handoff: acquired {path} after {time.monotonic() - started:.0f} s", err=True
            )
        yield


def _publish_handoff(run, stage, units, errors):
    stage_root = run["root"] / stage
    stage_root.mkdir(parents=True, exist_ok=True)
    # Serialize discovery with publication of the shared array-task coverage summary.
    with _handoff_lock(stage_root / ".handoff.lock"):
        # One listing instead of a path check per configured unit.
        directory = stage_root / _unit_subdirectory(run, stage)
        published = (
            {entry.name for entry in os.scandir(directory) if entry.is_dir()}
            if directory.is_dir()
            else set()
        )
        records = []
        for unit in units:
            if unit["unit_id"] not in published:
                continue
            try:
                record = _unit_result(run, stage, unit, validate_products=False)
                if record is not None:
                    records.append(
                        {
                            "unit": unit,
                            "path": f"{_unit_subdirectory(run, stage)}/{unit['unit_id']}/result.yaml",
                        }
                    )
            except Exception as exc:
                errors.append({"unit_id": unit["unit_id"], "error": str(exc)})
        handoff = {
            "stage": stage,
            "run_name": run["metadata"]["run_name"],
            "run": "..",
            "analysis_config_sha256": run["metadata"]["analysis_config_sha256"],
            "units": records,
            "expected_unit_count": len(units),
            "errors": errors,
        }
        if "predecessor" in run["metadata"]:
            handoff["predecessor"] = run["metadata"]["predecessor"]
        _write_yaml(stage_root / HANDOFFS[stage], handoff)
    return handoff


def execute_source_group(runs, selection, *, cpus=1, resume=False):
    """Publish each target independently while holding deterministic ordinary-unit locks."""
    groups = _selected_units(source_groups(runs), selection)
    if len(groups) != 1:
        raise ValueError("local grouped execution requires one exact source-group selector")
    targets = groups[0]["targets"]
    with _transfer_writer(runs, "reconstruct") as storage_receipt:
        return _execute_source_group(
            runs,
            targets,
            selection,
            cpus=cpus,
            resume=resume,
            storage_receipt=storage_receipt,
        )


def _execute_source_group(runs, targets, selection, *, cpus, resume, storage_receipt):
    from .reconstruction import validate_source_group, reconstruct_source_group

    for target in targets:
        run = target["run"]
        code = (
            _code_identity(allow_dirty=True)
            if run["metadata"].get("mode") == "scratch"
            else _code_identity()
        )
        if code["code_commit"] != run["metadata"]["code_commit"]:
            raise ValueError("execution checkout differs from a grouped run's recorded commit")
        target["run"] = {**run, "execution_code": code}
    identities = validate_source_group(targets)
    errors = {str(run["root"]): [] for run in runs}
    started = time.monotonic()
    with ExitStack() as locks:
        for target in sorted(
            targets, key=lambda t: (str(t["run"]["root"].resolve()), t["unit"]["unit_id"])
        ):
            locks.enter_context(_unit_lock(target["run"], "reconstruct", target["unit"]))
        pending = []
        # Validate every existing target before creating any attempts. Published targets
        # are immutable, and an explicit resume never blesses different scratch source.
        for target in targets:
            run, unit = target["run"], target["unit"]
            existing = _unit_result(run, "reconstruct", unit)
            if existing is not None:
                if not resume:
                    raise FileExistsError("stage unit already exists; use --resume")
                if not same_identity(
                    existing.get("stage_options", {}), run.get("stage_options", {})
                ):
                    raise ValueError("resume options differ from the published stage")
                if (
                    run["metadata"].get("mode") == "scratch"
                    and existing.get("execution_code") != run["execution_code"]
                ):
                    raise ValueError("scratch source changed; use a new run name")
                continue
            pending.append(target)
        for target in pending:
            run, unit = target["run"], target["unit"]
            stage_root = run["root"] / "reconstruct"
            (stage_root / "units").mkdir(exist_ok=True)
            (stage_root / "attempts").mkdir(exist_ok=True)
            attempt = stage_root / "attempts" / (unit["unit_id"] + "." + uuid.uuid4().hex)
            attempt.mkdir()
            target["destination"] = attempt
            _write_yaml(
                attempt / "attempt.yaml",
                {
                    "unit": unit,
                    "analysis_config_sha256": run["metadata"]["analysis_config_sha256"],
                    "execution_code": run["execution_code"],
                    "stage_options": run.get("stage_options", {}),
                    "source_group": selection,
                },
            )
        try:
            results = reconstruct_source_group(pending, identities=identities) if pending else []
        except Exception as exc:
            for target in pending:
                (target["destination"] / "failure.txt").write_text(traceback.format_exc())
            results = [{"error": {"error": str(exc), "type": type(exc).__name__}} for _ in pending]
        for target, result in zip(pending, results):
            run, unit, attempt = target["run"], target["unit"], target["destination"]
            try:
                if "error" in result:
                    errors[str(run["root"])].append({"unit_id": unit["unit_id"], **result["error"]})
                    continue
                observed = (
                    _code_identity(allow_dirty=True)
                    if run["metadata"].get("mode") == "scratch"
                    else _code_identity()
                )
                if observed != run["execution_code"]:
                    raise ValueError("source changed during grouped execution")
                result.update(
                    unit=unit,
                    stage="reconstruct",
                    execution_code=observed,
                    stage_options=run.get("stage_options", {}),
                    source_group=selection,
                    runtime={
                        "allocated_cpus": cpus,
                        "group_elapsed_seconds": time.monotonic() - started,
                        "storage": storage_receipt,
                    },
                    **{
                        key: run["metadata"][key]
                        for key in ("run_name", "code_commit", "analysis_config_sha256")
                    },
                )
                _write_yaml(attempt / "result.yaml", result)
                _validate_products(attempt, result)
                os.rename(attempt, run["root"] / "reconstruct" / "units" / unit["unit_id"])
            except Exception as exc:
                errors[str(run["root"])].append(
                    {"unit_id": unit["unit_id"], "error": str(exc), "type": type(exc).__name__}
                )
                if attempt.exists():
                    (attempt / "failure.txt").write_text(traceback.format_exc())
    handoffs = [
        _publish_handoff(
            run,
            "reconstruct",
            stage_units("reconstruct", run["config"], run["metadata"]["workload"]),
            errors[str(run["root"])],
        )
        for run in runs
    ]
    return {
        "handoffs": handoffs,
        "errors": [error for values in errors.values() for error in values],
    }


def execute_stage(run, stage, *, selection=None, cpus=1, resume=False):
    code = (
        _code_identity(allow_dirty=True)
        if run["metadata"].get("mode") == "scratch"
        else _code_identity()
    )
    if code["code_commit"] != run["metadata"]["code_commit"]:
        raise ValueError("execution checkout differs from this run's recorded code commit")
    transfer = run["config"]["measurement"]["reconstruction"].get("transfer_run")
    if transfer and transfer["pilot"] and stage in {"covariance", "fit", "sample"}:
        raise ValueError(
            "three-mock transfer pilot is engineering-only; covariance and recovery require the full ensemble"
        )
    definition = run["metadata"].get("inference_definition")
    reproject = definition == BGS_ANISOTROPIC and stage == "covariance"
    if "source_run" in run and stage not in {"fit", "sample"} and not reproject:
        raise ValueError("an inference run can only publish fits and samples")
    units = stage_units(
        stage, run["config"], run["metadata"]["workload"], inference_definition=definition
    )
    chosen = _selected_units(units, selection)
    with _transfer_writer([run], stage, unit_count=len(chosen)) as storage_receipt:
        return _execute_stage_units(
            run,
            stage,
            units,
            chosen,
            code=code,
            cpus=cpus,
            resume=resume,
            reproject=reproject,
            storage_receipt=storage_receipt,
        )


def _execute_stage_units(
    run, stage, units, chosen, *, code, cpus, resume, reproject, storage_receipt
):
    stage_root = run["root"] / stage
    options = run.get("stage_options", {})
    (stage_root / _unit_subdirectory(run, stage)).mkdir(parents=True, exist_ok=True)
    (stage_root / "attempts").mkdir(exist_ok=True)
    errors = []
    for unit in ([] if reproject else chosen):
        attempt = None
        started = time.monotonic()
        lock = ExitStack()
        try:
            lock.enter_context(_unit_lock(run, stage, unit))
            existing = _unit_result(run, stage, unit)
            if existing is not None:
                if not resume:
                    raise FileExistsError(
                        "stage unit already exists; use --resume to reuse unchanged work"
                    )
                if not same_identity(existing.get("stage_options", {}), options):
                    raise ValueError("resume options differ from the published stage")
                if (
                    run["metadata"].get("mode") == "scratch"
                    and existing.get("execution_code") != code
                ):
                    raise ValueError(
                        "scratch source changed; use a new run name instead of reusing this unit"
                    )
                continue
            previous = sorted(
                (stage_root / "attempts").glob(unit["unit_id"] + ".*"),
                key=lambda path: path.stat().st_mtime_ns,
            )
            if resume and stage == "sample" and previous:
                attempt = previous[-1]
                identity = _read_yaml(attempt / "attempt.yaml")
                if not same_identity(
                    identity,
                    {
                        "unit": unit,
                        "analysis_config_sha256": run["metadata"]["analysis_config_sha256"],
                        "execution_code": code,
                        "stage_options": options,
                    },
                ):
                    raise ValueError("sampler attempt identity differs from the requested unit")
            else:
                attempt = stage_root / "attempts" / (unit["unit_id"] + "." + uuid.uuid4().hex)
                attempt.mkdir()
                _write_yaml(
                    attempt / "attempt.yaml",
                    {
                        "unit": unit,
                        "analysis_config_sha256": run["metadata"]["analysis_config_sha256"],
                        "execution_code": code,
                        "stage_options": options,
                    },
                )
            result = _science(stage, {**run, "execution_code": code}, unit, attempt, cpus)
            observed_code = (
                _code_identity(allow_dirty=True)
                if run["metadata"].get("mode") == "scratch"
                else _code_identity()
            )
            if observed_code != code:
                raise ValueError(
                    "source changed during execution; retained attempt cannot be published"
                )
            runtime = {
                "allocated_cpus": cpus,
                "elapsed_seconds": time.monotonic() - started,
                "thread_environment": {
                    key: os.environ.get(key)
                    for key in run["config"]["analysis"]["execution_environment"][
                        "thread_environment_variables"
                    ]
                },
            }
            if storage_receipt is not None:
                runtime["storage"] = storage_receipt
            if run["config"]["analysis"]["execution_environment"]["record_threadpoolctl"]:
                try:
                    from threadpoolctl import threadpool_info

                    runtime["threadpools"] = threadpool_info()
                except ImportError:
                    runtime["threadpools_unavailable_reason"] = "threadpoolctl is not installed"
            result["runtime"] = runtime
            result["execution_code"] = code
            result["stage_options"] = options
            if "source_run" in run:
                result["source_run"] = run["metadata"]["source_run"]
            result.update(
                unit=unit,
                stage=stage,
                **{
                    key: run["metadata"][key]
                    for key in ("run_name", "code_commit", "analysis_config_sha256")
                },
            )
            _write_yaml(attempt / "result.yaml", result)
            _validate_products(attempt, result)
            os.rename(attempt, stage_root / "units" / unit["unit_id"])
        except Exception as exc:
            errors.append(
                {"unit_id": unit["unit_id"], "error": str(exc), "type": type(exc).__name__}
            )
            if attempt is not None and attempt.exists():
                (attempt / "failure.txt").write_text(traceback.format_exc())
            click.echo(f"{unit['unit_id']}: {exc}", err=True)
        finally:
            lock.close()
    if stage == "covariance":
        errors.extend(_assemble_covariances(run, chosen, cpus=cpus, execution_code=code))
    return _publish_handoff(run, stage, units, errors)


def _run_cap_attempt(run, unit, attempt, *, cpus, n_loops=None):
    from .covariance import covariance_cap_stage, is_infrastructure_failure

    retries = 0 if "transfer_run" in run["config"]["measurement"]["reconstruction"] else 1
    for retry in range(retries + 1):
        destination = attempt / f"cap-attempt-{retry}"
        destination.mkdir()
        try:
            result = covariance_cap_stage(run, unit, destination, nthread=cpus, n_loops=n_loops)
            result["cap_directory"] = destination.name
            for product in result["products"]:
                product["path"] = destination.name + "/" + product["path"]
            return result
        except Exception as exc:
            (destination / "failure.txt").write_text(traceback.format_exc())
            if retry == retries or not is_infrastructure_failure(exc):
                raise


def _cap_directory(unit_directory):
    record = _read_yaml(unit_directory / "result.yaml")
    directory = (unit_directory / record["cap_directory"]).resolve()
    if not directory.is_relative_to(unit_directory.resolve()):
        raise ValueError("cap product path leaves its unit")
    return directory


def _assemble_covariances(run, chosen, *, cpus, execution_code=None):
    """Own route publication and recovery; covariance calculations never own runtime locks."""
    from .covariance import assemble_route, NumericalExhaustion, read_route_covariance

    stage_root = run["root"] / "covariance"
    (stage_root / "routes").mkdir(exist_ok=True)
    (stage_root / "depths").mkdir(exist_ok=True)
    errors = []
    reproject = run["metadata"].get("inference_definition") == BGS_ANISOTROPIC
    source = run["source_run"] if reproject else run
    routes = routes_for(run["config"], run["metadata"]["workload"])
    wanted = {route_name(unit) for unit in chosen}
    scratch = run["metadata"].get("mode") == "scratch"

    def check_code(record=None):
        if scratch and execution_code is not None:
            observed = (
                record.get("execution_code")
                if record is not None
                else _code_identity(allow_dirty=True)
            )
            if observed != execution_code:
                raise ValueError("covariance scratch source changed; use a new run")

    for route in routes:
        name = route_name(route)
        if name not in wanted:
            continue
        target = stage_root / "routes" / name
        cap_units = {
            cap: {"unit_id": name + "--" + cap, **route, "cap": cap} for cap in ("NGC", "SGC")
        }
        with (stage_root / ("." + name + ".lock")).open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            attempt = None
            try:
                if target.exists():
                    result = _read_yaml(target / "result.yaml")
                    if not same_identity(result["route"], route) or any(
                        result.get(key) != run["metadata"][key]
                        for key in ("run_name", "code_commit", "analysis_config_sha256")
                    ):
                        raise ValueError("completed covariance route identity differs")
                    _validate_products(target, result)
                    check_code(result)
                    if reproject:
                        _unit_result(run, "covariance", {"unit_id": name, **route})
                        read_route_covariance(run, route)
                    continue
                if reproject:
                    saved = read_route_covariance(source, route)
                    cap_directories = {
                        cap: (source["root"] / relative).resolve()
                        for cap, relative in saved["metadata"]["cap_integrations"].items()
                    }
                    if any(
                        not path.is_relative_to(source["root"]) for path in cap_directories.values()
                    ):
                        raise ValueError("source cap integration path leaves its measurement run")
                else:
                    if any(
                        _unit_result(run, "covariance", unit) is None for unit in cap_units.values()
                    ):
                        continue
                    cap_directories = {
                        cap: _cap_directory(stage_root / "units" / unit["unit_id"])
                        for cap, unit in cap_units.items()
                    }
                    for unit in cap_units.values():
                        check_code(_unit_result(run, "covariance", unit))
                attempt = stage_root / "attempts" / (name + ".assembly." + uuid.uuid4().hex)
                attempt.mkdir()

                def integrate_next(cap, depth):
                    if reproject:
                        raise ValueError("BGS reprojection cannot launch deeper cap integrations")
                    unit = cap_units[cap]
                    complete = stage_root / "depths" / f"{unit['unit_id']}--{depth}"
                    if complete.exists():
                        record = _read_yaml(complete / "result.yaml")
                        if (
                            not same_identity(record["unit"], unit)
                            or not same_identity(record["n_loops"], depth)
                            or any(
                                record.get(key) != run["metadata"][key]
                                for key in ("run_name", "code_commit", "analysis_config_sha256")
                            )
                        ):
                            raise ValueError("completed deeper cap identity differs")
                        _validate_products(complete, record)
                        check_code(record)
                        return _cap_directory(complete)
                    partial = stage_root / "attempts" / (complete.name + "." + uuid.uuid4().hex)
                    partial.mkdir()
                    record = _run_cap_attempt(run, unit, partial, cpus=cpus, n_loops=depth)
                    record.update(
                        unit=unit,
                        execution_code=execution_code,
                        **{
                            key: run["metadata"][key]
                            for key in ("run_name", "code_commit", "analysis_config_sha256")
                        },
                    )
                    _write_yaml(partial / "result.yaml", record)
                    _validate_products(partial, record)
                    check_code()
                    os.rename(partial, complete)
                    return _cap_directory(complete)

                result = assemble_route(
                    run, route, cap_directories, attempt, integrate_next=integrate_next
                )
                result.update(
                    route=route,
                    execution_code=execution_code,
                    **{
                        key: run["metadata"][key]
                        for key in ("run_name", "code_commit", "analysis_config_sha256")
                    },
                )
                if reproject:
                    result.update(
                        unit={"unit_id": name, **route},
                        stage="covariance",
                        source_run=run["metadata"]["source_run"],
                    )
                _write_yaml(attempt / "result.yaml", result)
                _validate_products(attempt, result)
                check_code()
                os.rename(attempt, target)
            except Exception as exc:
                errors.append({"route": route, "error": str(exc), "type": type(exc).__name__})
                if attempt is not None:
                    (attempt / "failure.txt").write_text(traceback.format_exc())
                    if isinstance(exc, NumericalExhaustion):
                        _write_yaml(attempt / "numerical_exhaustion.yaml", {"history": exc.history})
    return errors


def _run_from_handoff(path, stage, *, pending=False):
    path = resolve_saved_path(path).resolve()
    previous = STAGES[STAGES.index(stage) - 1]
    allowed = {"select", "reconstruct"} if stage == "cluster" else {previous}
    if pending and not path.exists():
        # Planning a dependent array: the handoff is published by an upstream job, so only
        # its location inside an existing run is checked here. Execution still reads it.
        if path.parent.name not in allowed or path.name != HANDOFFS[path.parent.name]:
            raise ValueError(f"pending {stage} input must be an upstream handoff location")
        return load_run(path.parent.parent)
    handoff = _read_yaml(path)
    if handoff.get("stage") not in allowed:
        raise ValueError(f"{stage} requires a handoff from {', '.join(sorted(allowed))}")
    run = load_run(path.parent / handoff["run"])
    if (
        handoff.get("analysis_config_sha256") != run["metadata"]["analysis_config_sha256"]
        or handoff.get("run_name") != run["metadata"]["run_name"]
    ):
        raise ValueError("handoff belongs to another run or snapshot")
    return run


class _SavedPath(click.Path):
    """Resolve recorded locations before Click validates the destination."""

    def convert(self, value, param, ctx):
        try:
            value = resolve_saved_path(value)
        except (OSError, ValueError) as exc:
            self.fail(str(exc), param, ctx)
        return super().convert(value, param, ctx)


@click.group()
def main():
    """BAO analysis: catalog → select → reconstruct → cluster → covariance → fit → sample → review."""


@main.command()
@click.option("--config", "config_path", type=_SavedPath(path_type=Path, exists=True))
@click.option("--pilot", is_flag=True, help="Core engineering subset; not a recovery ensemble.")
def transfer_matrix(config_path, pilot):
    """Print the opt-in transfer dependency inventory as YAML; never submit jobs."""
    from .config.transfer import transfer_matrix as build_matrix

    click.echo(
        yaml.safe_dump(
            build_matrix(load_analysis_config(config_path), pilot=pilot), sort_keys=False
        )
    )


@main.command()
@click.option("--config", "config_path", type=_SavedPath(path_type=Path, exists=True))
@click.option("--output-dir", required=True, type=click.Path(path_type=Path))
def sensitivity_yamls(config_path, output_dir):
    """Write the smoothing-scan and RecIso experiment YAMLs without running any stage."""
    from .config.sensitivity import sensitivity_run_configs
    from .config.stages import experiment_document

    configs = sensitivity_run_configs(load_analysis_config(config_path))
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, config in configs.items():
        document = experiment_document(config)
        document["cluster"]["statistics"] = "xi"
        path = output_dir / (name + ".yaml")
        _write_yaml(path, document)
        click.echo(str(path))


@main.command()
@click.option("--config", "config_path", type=_SavedPath(path_type=Path, exists=True))
@click.option("--output-dir", required=True, type=click.Path(path_type=Path))
@click.option("--density-handoff", type=click.Path(path_type=Path, exists=True))
@click.option("--density-snapshot")
@click.option("--only", "only", multiple=True, help="write only these named definitions")
def per_bin_yamls(config_path, output_dir, density_handoff, density_snapshot, only):
    """Write the per-bin primary, smoothing-scan, RecIso and optional density run YAMLs.

    The primary is the per-bin configuration itself (xi and the secondary P(k) diagnostic
    bins); the sensitivity snapshots derive from it; the density successor binds the identified
    V_eff-matched exact-count handoff (``density-handoff prepare-radial``) and its resolved
    snapshot digest. ``--only`` adds named definitions to an existing directory without
    rewriting any file. Nothing runs or is submitted.
    """
    from .config.sensitivity import sensitivity_run_configs
    from .config.stages import experiment_document

    config = load_analysis_config(config_path)
    recon = config["measurement"]["reconstruction"]
    if (
        config["schema_version"] not in PER_BIN_SCHEMA_VERSIONS
        or recon["scope"] != "separate_measurement_bins"
    ):
        raise click.UsageError("per-bin runs derive from the schema-16/17 per-bin configuration")
    if (density_handoff is None) != (density_snapshot is None):
        raise click.UsageError("--density-handoff and --density-snapshot are given together")
    documents = {"primary": (config, "both")}
    for name, snapshot in sensitivity_run_configs(config).items():
        documents[name] = (snapshot, "xi")
    if density_handoff is not None:
        from .config.density import density_run_config

        handoff = json.loads(Path(density_handoff).read_text())
        documents["density"] = (density_run_config(config, handoff, density_snapshot), "xi")
    if only:
        unknown = set(only) - set(documents)
        if unknown:
            raise click.UsageError(f"unknown per-bin definitions: {sorted(unknown)}")
        documents = {name: documents[name] for name in only}
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
    for name in documents:
        if (output_dir / (name + ".yaml")).exists():
            raise click.UsageError(f"refusing to overwrite {output_dir / (name + '.yaml')}")
    for name, (snapshot, statistics) in documents.items():
        document = experiment_document(snapshot)
        document["cluster"]["statistics"] = statistics
        path = output_dir / (name + ".yaml")
        _write_yaml(path, document)
        click.echo(str(path))


@main.command("density-handoff")
@click.argument("action", type=click.Choice(["counts", "prepare-radial"]))
@click.option(
    "--source-run",
    "source_root",
    required=True,
    type=_SavedPath(path_type=Path, exists=True),
    help="The per-bin primary run whose published LRG selection supplies the matched counts.",
)
@click.option(
    "--output",
    required=True,
    type=click.Path(path_type=Path),
    help="New directory for the count plan or the prepared-inputs.json handoff.",
)
def density_handoff(action, source_root, output):
    """Derive the V_eff-matched LRG2/LRG3 exact-count handoff from a published per-bin primary.

    ``counts`` writes the matched count plan for review (minutes). ``prepare-radial`` also
    selects both matched tiers for every realization and cap through the native selector,
    proves paired membership, fits the matched radial templates and writes
    ``prepared-inputs.json`` with the resolved snapshot digest that ``per-bin-yamls
    --density-handoff ... --density-snapshot ...`` binds. Nothing is published, reconstructed
    or submitted.
    """
    from .config.density import CAPS, CONDITIONS, produce_handoff

    plan, handoff = produce_handoff(load_run(source_root, allow_source=False), output, action)
    for condition in CONDITIONS:
        for name, row in plan["conditions"][condition].items():
            counts = " / ".join(str(row["caps"][cap]["selected"]) for cap in CAPS)
            state = (
                "unattainable, primary kept"
                if row["unattainable"]
                else f"x{row['factor_selected_over_primary']:.2f} over primary"
            )
            click.echo(
                f"{condition} {name}: {counts} ({state}; V_eff fraction "
                f"{row['veff_fraction']:.4f}{'; bounded' if row['bounded'] else ''})"
            )
    if handoff is not None:
        click.echo(f"resolved_snapshot_sha256 {handoff['resolved_snapshot_sha256']}")
    click.echo(str(Path(output) / "complete.json"))


@main.command()
@click.option("--config", "config_path", type=_SavedPath(path_type=Path, exists=True))
@click.option("--pilot", is_flag=True)
@click.option("--output-dir", required=True, type=click.Path(path_type=Path))
@click.option("--damping-centres", type=click.Path(path_type=Path, exists=True))
def transfer_yamls(config_path, pilot, output_dir, damping_centres):
    """Write complete native-stage YAMLs without running or submitting any stage."""
    from .config.transfer import transfer_run_configs
    from .config.stages import experiment_document

    configs = transfer_run_configs(load_analysis_config(config_path), pilot=pilot)
    if damping_centres is not None:
        artifact = _read_yaml(damping_centres)
        for config in configs.values():
            config["measurement"]["reconstruction"]["transfer_run"]["damping_centres"] = artifact
            validate_analysis_config(config)
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, config in configs.items():
        document = experiment_document(config)
        document["cluster"]["statistics"] = "xi"
        path = output_dir / (name + ".yaml")
        _write_yaml(path, document)
        click.echo(str(path))


@main.command()
@click.option("--run", "run_root", required=True, type=_SavedPath(path_type=Path, exists=True))
@click.option("--tracer", "selected_tracer", type=click.Choice(["LRG", "BGS_BRIGHT-21.5"]))
def transfer_centre_contexts(run_root, selected_tracer):
    """Print measured source-target inputs for independent damping review, with no centres."""
    from .selection import read_preparation
    from .theory import transfer_damping_context

    run = load_run(run_root)
    if "transfer_run" not in run["config"]["measurement"]["reconstruction"]:
        raise click.UsageError("damping contexts require an exclusive-transfer run")
    preparations, records = {}, []
    for route in routes_for(run["config"], "transfer"):
        if route["reconstruction_state"] != "postreconstruction":
            continue
        tracer = route["tracer"]
        if selected_tracer is not None and tracer != selected_tracer:
            continue
        if tracer not in preparations:
            preparations[tracer] = read_preparation(run, tracer)
        records.append(
            {
                "context": transfer_damping_context(
                    run["config"],
                    preparations[tracer],
                    tracer=tracer,
                    measurement_bin=route["measurement_bin"],
                    target_class=route["condition"],
                ),
                "centers_mpc_h": None,
                "evidence": None,
            }
        )
    click.echo(
        yaml.safe_dump(
            {"units": "Mpc/h", "review_reference": None, "records": records}, sort_keys=False
        )
    )


@main.command()
@click.option("--run", "run_root", required=True, type=_SavedPath(path_type=Path, exists=True))
@click.option(
    "--covariance-from",
    "covariance_root",
    type=_SavedPath(path_type=Path, exists=True),
    help="Labelled proxy: a run publishing the same routes' covariance and window.",
)
@click.option("--tracer", "selected_tracer", type=click.Choice(["LRG", "BGS_BRIGHT-21.5"]))
@click.option("--output", required=True, type=click.Path(path_type=Path))
def transfer_centres(run_root, covariance_root, selected_tracer, output):
    """Write candidate source-target damping centres; adoption needs a review reference."""
    from .likelihood import transfer_centre_artifact

    run = load_run(run_root)
    if "transfer_run" not in run["config"]["measurement"]["reconstruction"]:
        raise click.UsageError("damping centres require an exclusive-transfer run")
    if output.exists():
        raise click.UsageError(f"centre artifact already exists: {output}")
    covariance_run = None if covariance_root is None else load_run(covariance_root)
    artifact = transfer_centre_artifact(
        run,
        routes_for(run["config"], run["metadata"]["workload"]),
        covariance_run=covariance_run,
        tracer=selected_tracer,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        yaml.safe_dump(artifact, handle, sort_keys=False)
    click.echo(str(output))


@main.command("transfer-centres-report")
@click.option("--run", "run_root", required=True, type=_SavedPath(path_type=Path, exists=True))
@click.option("--candidate", required=True, type=_SavedPath(path_type=Path, exists=True))
@click.option(
    "--companion",
    "companions",
    multiple=True,
    help="NAME=PATH of another run's candidate, drawn in the consistency panels.",
)
@click.option("--output-root", required=True, type=click.Path(path_type=Path))
@click.option(
    "--native",
    is_flag=True,
    help="First compute the native diagnostics (profiles, prediction and data overlays) "
    "into the report's native/ directory; run inside a compute allocation.",
)
def transfer_centres_report(run_root, candidate, companions, output_root, native):
    """Render the review report of an unfrozen centre candidate; it changes nothing it reads."""
    from .diagnostics.transfer_centres import (
        render_centres_report,
        report_directory,
        run_diagnostics,
    )

    if any("=" not in item for item in companions):
        raise click.BadParameter("use NAME=PATH", param_hint="--companion")
    pairs = dict(item.split("=", 1) for item in companions)
    run = load_run(run_root)
    name = run["metadata"]["run_name"]
    artifact = yaml.safe_load(Path(candidate).read_text())
    # Checked before the native diagnostics, which take minutes per record.
    if artifact.get("review_reference") is not None:
        raise click.ClickException("the candidate is already frozen; reports review candidates")
    for key in ("run_name", "analysis_config_sha256"):
        if artifact["producer"].get(key) != run["metadata"].get(key):
            raise click.ClickException(f"the candidate's producer {key} is not this run's")
    destination = report_directory(output_root, name, candidate)
    failures = run_diagnostics(run, candidate, destination / "native") if native else {}
    path = render_centres_report(
        candidate,
        run["config"],
        destination,
        run_name=name,
        companions=pairs,
        diagnostics_dir=destination / "native",
    )
    click.echo(str(path))
    if failures:
        raise click.ClickException(
            f"native diagnostics failed for {len(failures)} records: {', '.join(failures)}"
        )


@main.command()
@click.option("--run-name", required=True)
@click.option("--output-root", required=True, type=click.Path(path_type=Path))
@click.option("--catalog-root", required=True, type=_SavedPath(path_type=Path, exists=True))
@click.option("--workload", default="pilot", show_default=True)
@click.option(
    "--scratch", is_flag=True, help="Record edited source and write under output-root/scratch."
)
@click.option(
    "--download",
    is_flag=True,
    help="Download missing supported sources; validation alone is the default.",
)
@click.option(
    "--config",
    "config_path",
    type=_SavedPath(path_type=Path, exists=True),
    help="Complete experiment definition to freeze for a new independent run.",
)
@click.option(
    "--resume", is_flag=True, help="Resume preparation under this run's unchanged snapshot."
)
@click.option("--unit", help="Configured tracer unit name or zero-based index.")
@click.option("--cpus", type=click.IntRange(min=1), default=1)
@click.option("--scheduler", type=click.Choice(["local", "slurm"]), default="local")
@click.option("--submit", is_flag=True)
@click.option("--memory")
@click.option("--walltime")
@click.option("--account", help="Slurm allocation account, required for a dry plan.")
@click.option("--partition", help="Optional Slurm partition.")
@click.option("--max-concurrent", type=click.IntRange(min=1), default=1)
@click.option("--storage-budget", type=_SavedPath(path_type=Path, exists=True))
@click.option("--after", multiple=True, help="Plan an afterok dependency on this Slurm job.")
@click.option(
    "--create-snapshot",
    is_flag=True,
    help="With a Slurm plan, freeze the run snapshot now so dependent arrays can be planned.",
)
@click.option(
    "--predecessor",
    type=_SavedPath(path_type=Path, exists=True),
    help="Earlier run of this definition whose published units adopt-predecessor moves here.",
)
def catalog(
    run_name,
    output_root,
    catalog_root,
    workload,
    scratch,
    download,
    config_path,
    resume,
    unit,
    cpus,
    scheduler,
    submit,
    memory,
    walltime,
    account,
    partition,
    max_concurrent,
    storage_budget,
    after,
    create_snapshot,
    predecessor,
):
    """Download or validate sources and create the run's catalogue registration."""
    if (after or create_snapshot) and scheduler != "slurm":
        raise click.UsageError("--after and --create-snapshot only apply to Slurm plans")
    if config_path is not None:
        document = _read_yaml(config_path)
        if "shared" in document and "catalog" not in document:
            raise click.UsageError("catalog configuration requires its complete catalog section")
    if scratch and "scratch" not in output_root.parts:
        output_root = output_root / "scratch"
    root = output_root / run_name
    if scheduler == "slurm":
        if create_snapshot:
            # The planning checkout is the frozen producer; array tasks resume this snapshot.
            if not root.exists():
                create_run(
                    output_root,
                    run_name,
                    workload=workload,
                    catalog_root=catalog_root,
                    config_path=config_path,
                    scratch=scratch,
                    predecessor=predecessor,
                )
            resume = True
        if resume and root.exists():
            existing = load_run(root)
            if existing["metadata"]["workload"] != workload or not same_location(
                existing["metadata"]["catalog_root"], catalog_root
            ):
                raise click.UsageError(
                    "resuming inputs requires the original workload and catalog root"
                )
            _check_recorded_predecessor(existing, predecessor)
            config = existing["config"]
            if config_path is not None and not same_identity(
                load_analysis_config(config_path), config
            ):
                raise click.UsageError("--config differs from the existing run snapshot")
        else:
            config = (
                load_analysis_config() if config_path is None else load_analysis_config(config_path)
            )
        units = _selected_units(stage_units("catalog", config, workload), unit)
        script = slurm_plan(
            "catalog",
            units,
            input_path=None,
            cpus=cpus,
            memory=memory,
            walltime=walltime,
            account=account,
            partition=partition,
            boundary_experiment="experiment" in config["selection"]["boundary_support"],
            run={
                "run_name": run_name,
                "output_root": output_root,
                "catalog_root": catalog_root,
                "workload": workload,
                "config_path": config_path,
                "scratch": scratch,
                "download": download,
                "predecessor": predecessor,
            },
            max_concurrent=max_concurrent,
            storage_budget=storage_budget,
            storage_outputs=[root] + ([catalog_root] if download else []),
            transfer=_budgeted_run(config),
            resume=resume,
            after=after,
        )
        click.echo(script)
        if submit:
            click.confirm("Submit this exact displayed array plan with sbatch?", abort=True)
            subprocess.run(["sbatch"], input=script, text=True, check=True)
        return
    if submit:
        raise click.UsageError("--submit requires --scheduler slurm")
    _configure_threads("catalog", cpus)
    root = output_root / run_name
    config = (
        load_run(root)["config"]
        if resume and root.exists()
        else (
            load_analysis_config(config_path) if config_path is not None else load_analysis_config()
        )
    )
    budget = _read_yaml(storage_budget) if storage_budget is not None else None
    admission_run = {"root": root, "config": config, "storage_budget": budget}
    chosen = _selected_units(stage_units("catalog", config, workload), unit)
    with _transfer_writer(
        [admission_run],
        "catalog",
        unit_count=len(chosen),
        extra_outputs=[catalog_root] if download else [],
    ):
        _catalog_local(
            root,
            output_root=output_root,
            run_name=run_name,
            workload=workload,
            catalog_root=catalog_root,
            config_path=config_path,
            scratch=scratch,
            resume=resume,
            download=download,
            unit=unit,
            cpus=cpus,
            budget=budget,
            predecessor=predecessor,
        )


def _check_recorded_predecessor(run, predecessor):
    """A resumed successor keeps the predecessor it recorded at creation."""
    recorded = run["metadata"].get("predecessor")
    if predecessor is not None and (
        recorded is None or not same_location(recorded["root"], predecessor)
    ):
        raise click.UsageError("--predecessor differs from the existing run's recorded predecessor")


def _catalog_local(
    root,
    *,
    output_root,
    run_name,
    workload,
    catalog_root,
    config_path,
    scratch,
    resume,
    download,
    unit,
    cpus,
    budget,
    predecessor=None,
):
    if resume and root.exists():
        run = load_run(root)
    else:
        try:
            run = create_run(
                output_root,
                run_name,
                workload=workload,
                catalog_root=catalog_root,
                config_path=config_path,
                scratch=scratch,
                predecessor=predecessor,
            )
        except FileExistsError:
            if not resume:
                raise
            run = load_run(root)
    _check_recorded_predecessor(run, predecessor)
    if config_path is not None and not same_identity(
        load_analysis_config(config_path), run["config"]
    ):
        raise click.UsageError("--config differs from the existing run snapshot")
    if (
        not same_location(run["metadata"]["catalog_root"], catalog_root)
        or run["metadata"]["workload"] != workload
    ):
        raise click.UsageError(
            "resuming inputs requires the run's original workload and catalog root"
        )
    if (run["metadata"].get("mode") == "scratch") != scratch:
        raise click.UsageError("--scratch must match the existing run mode")
    run["stage_options"] = {"download": download}
    run["storage_budget"] = budget
    result = execute_stage(run, "catalog", selection=unit, cpus=cpus, resume=resume)
    if result["errors"]:
        raise click.ClickException("input preparation has incomplete units; see saved errors")


def _validate_centre_adoption(source, config):
    """Exercise the native resolver for every required post-reconstruction context."""
    from .selection import read_preparation
    from .theory import transfer_damping_context, resolve_transfer_damping_prior

    preparations = {}
    for route in routes_for(source["config"], source["metadata"]["workload"]):
        if route["reconstruction_state"] != "postreconstruction":
            continue
        tracer = route["tracer"]
        if tracer not in preparations:
            preparations[tracer] = read_preparation(source, tracer)
        context = transfer_damping_context(
            config,
            preparations[tracer],
            tracer=tracer,
            measurement_bin=route["measurement_bin"],
            target_class=route["condition"],
        )
        # The artifact validation shares the production resolver, including independent
        # calibration IDs, streaming, evidence and exact context checks.
        from .catalog import build_mock_cosmology

        z_eff = context["target_redshift"]["z_eff"]
        resolve_transfer_damping_prior(
            tracer,
            "postreconstruction",
            context=context,
            sigma_z_over_one_plus_z=context["sigma0"]["target"],
            z_eff=z_eff,
            e_z_eff=float(build_mock_cosmology(config).efunc(z_eff)),
            config=config,
        )


def _inference_destination(
    source, target, *, definition, resume, create, damping_centres=None, covariance_runs=()
):
    """Inspect or create a derived run without mutating the source definition."""
    if "source_run" in source:
        raise ValueError("an inference source must be an original measurement run")
    target = resolve_saved_path(target).resolve()
    scratch = source["metadata"].get("mode") == "scratch"
    if scratch and "scratch" not in target.parent.parts:
        raise ValueError("scratch inference destination must be under a scratch directory")
    if target.is_relative_to(source["root"]):
        raise ValueError("inference output must be outside the source run")
    if target.exists():
        if not resume:
            raise FileExistsError(f"named run already exists: {target}")
        derived = load_run(target)
        if damping_centres is not None and not same_identity(
            _read_yaml(damping_centres),
            derived["config"]["measurement"]["reconstruction"]["transfer_run"]["damping_centres"],
        ):
            raise ValueError("cannot mutate damping centres of an existing inference run")
    elif create:
        try:
            derived = create_run(
                target.parent,
                target.name,
                workload=source["metadata"]["workload"],
                catalog_root=source["metadata"]["catalog_root"],
                source_run=source,
                inference_definition=definition,
                scratch=scratch,
                damping_centres=damping_centres,
                covariance_runs=covariance_runs,
            )
        except FileExistsError:
            if not resume:
                raise
            derived = load_run(target)
    else:
        providers = _covariance_providers(source, definition, covariance_runs)
        config = derive_inference_config(source["config"], definition)
        if damping_centres is not None:
            if definition not in TRANSFER_INFERENCE:
                raise ValueError("damping adoption requires transfer inference")
            config["measurement"]["reconstruction"]["transfer_run"]["damping_centres"] = _read_yaml(
                damping_centres
            )
            _validate_centre_adoption(source, config)
        validate_inference_config(source["config"], config, definition)
        # Planning uses no output identity and writes no snapshot or products.
        derived = {
            "root": target,
            "config": config,
            "metadata": {
                "workload": source["metadata"]["workload"],
                "source_run": _source_identity(source),
                "inference_definition": definition,
            },
        }
        if providers:
            derived["metadata"]["covariance_runs"] = [_source_identity(p) for p in providers]
            derived["covariance_runs"] = providers
    recorded = [entry["root"] for entry in derived["metadata"].get("covariance_runs", [])]
    if [str(resolve_saved_path(root).resolve()) for root in covariance_runs] != [
        str(resolve_saved_path(root).resolve()) for root in recorded
    ]:
        raise ValueError("inference destination records different covariance providers")
    if (
        not same_run_identity(derived["metadata"].get("source_run"), _source_identity(source))
        or derived["metadata"].get("inference_definition") != definition
        or (target.exists() and (derived["metadata"].get("mode") == "scratch") != scratch)
    ):
        raise ValueError("inference destination belongs to another source or definition")
    return derived


def _resolve_stage_options(run, stage, config_path, statistics, save_catalogs):
    """Freeze execution choices beside products; CLI statistics explicitly overrides YAML."""
    from .config.stages import read_stage_config

    options = {}
    if config_path is not None:
        options = read_stage_config(config_path, stage=stage, baseline=run["config"])
    if stage == "cluster":
        options["statistics"] = statistics or options.get(
            "statistics", run["metadata"].get("statistics", "both")
        )
        if (
            "transfer_run" in run["config"]["measurement"]["reconstruction"]
            and options["statistics"] != "xi"
        ):
            raise ValueError("exclusive transfer uses the correlation-function statistic")
    if stage == "reconstruct":
        options["save_catalogs"] = save_catalogs
    return options


def _stage_command(
    stage,
    input_path,
    unit,
    cpus,
    scheduler,
    submit,
    memory,
    walltime,
    resume,
    account,
    partition,
    inference_run=None,
    inference_definition=None,
    config_path=None,
    statistics=None,
    save_catalogs=True,
    source_group=None,
    peer_input=(),
    reuse_selection=None,
    max_concurrent=1,
    damping_centres=None,
    storage_budget=None,
    after=(),
    tracer=None,
    covariance_run=(),
    review=True,
    units_per_task=1,
):
    if covariance_run and (stage != "fit" or inference_definition not in CORE_POWER3_INFERENCE):
        raise click.UsageError("--covariance-run requires fit with a core power3 definition")
    if tracer is not None and (
        scheduler != "slurm" or (source_group is not None and source_group != "all")
    ):
        raise click.UsageError("--tracer filters a Slurm plan or a --source-group all plan")
    if after and scheduler != "slurm":
        raise click.UsageError("--after only plans dependent Slurm arrays")
    if units_per_task != 1 and scheduler != "slurm":
        raise click.UsageError("--units-per-task only plans Slurm arrays; name units with --unit")
    # Upstream handoffs may be pending only while planning an afterok-dependent array.
    for path in (input_path, *peer_input, *([] if reuse_selection is None else [reuse_selection])):
        if not after and not Path(path).exists():
            raise click.UsageError(f"input handoff does not exist: {path}")
    if source_group is not None and (
        stage != "reconstruct" or unit is not None or not save_catalogs
    ):
        raise click.UsageError(
            "--source-group requires reconstruct with catalogue saving and excludes --unit"
        )
    if peer_input and source_group is None:
        raise click.UsageError("--peer-input requires --source-group")
    if reuse_selection is not None and stage != "select":
        raise click.UsageError("--reuse-selection requires select")
    if damping_centres is not None and (
        stage != "fit" or inference_run is None or inference_definition not in TRANSFER_INFERENCE
    ):
        raise click.UsageError(
            "--damping-centres requires fit and a new transfer inference destination"
        )
    run = _run_from_handoff(input_path, stage, pending=bool(after))
    budget = _read_yaml(storage_budget) if storage_budget is not None else None
    if reuse_selection is not None and budget is None:
        raise ValueError("selection reuse requires --storage-budget")
    run["storage_budget"] = budget
    stage_options = _resolve_stage_options(run, stage, config_path, statistics, save_catalogs)
    if reuse_selection is not None:
        stage_options["reuse_selection"] = str(reuse_selection.resolve())
    run["stage_options"] = stage_options
    group_runs = [run]
    for path in peer_input:
        peer = _run_from_handoff(path, "reconstruct", pending=bool(after))
        peer["storage_budget"] = budget
        peer["stage_options"] = dict(stage_options)
        group_runs.append(peer)
    if inference_definition is not None and (
        inference_run is None
        or not (
            (stage == "covariance" and inference_definition == BGS_ANISOTROPIC)
            or (
                stage == "fit"
                and inference_definition
                in (*TRANSFER_INFERENCE, *CORE_INFERENCE, *UNIFIED_PRIOR_INFERENCE)
            )
        )
    ):
        raise click.UsageError("inference definition requires its matching stage and destination")
    if inference_run is not None:
        if stage != "fit" and not (
            stage == "covariance" and inference_definition == BGS_ANISOTROPIC
        ):
            raise click.UsageError("inference creation requires fit or BGS covariance reprojection")
        source = run
        run = _inference_destination(
            source,
            inference_run,
            definition=inference_definition,
            resume=resume,
            create=False,
            damping_centres=damping_centres,
            covariance_runs=covariance_run,
        )
        run["storage_budget"] = budget
    if scheduler == "slurm":
        units = (
            planned_source_groups(group_runs, source_group, tracer)
            if source_group is not None
            else [
                item
                for item in _selected_units(
                    stage_units(
                        stage,
                        run["config"],
                        run["metadata"]["workload"],
                        inference_definition=run["metadata"].get("inference_definition"),
                    ),
                    unit,
                )
                if tracer is None or item.get("tracer") == tracer
            ]
        )
        if not units:
            raise click.UsageError("the plan selects no units")
        script = slurm_plan(
            stage,
            units,
            input_path=input_path,
            cpus=cpus,
            memory=memory,
            walltime=walltime,
            account=account,
            partition=partition,
            inference_run=inference_run,
            boundary_experiment="experiment" in run["config"]["selection"]["boundary_support"],
            inference_definition=inference_definition,
            config_path=config_path,
            statistics=stage_options.get("statistics"),
            save_catalogs=save_catalogs,
            source_group=source_group is not None,
            peer_inputs=peer_input,
            reuse_selection=reuse_selection,
            max_concurrent=max_concurrent,
            resume=resume,
            damping_centres=damping_centres,
            covariance_runs=covariance_run,
            storage_budget=storage_budget,
            storage_outputs=(
                [item["root"] for item in group_runs] if source_group is not None else [run["root"]]
            ),
            transfer=_budgeted_run(run["config"]),
            after=after,
            review=review,
            units_per_task=units_per_task,
        )
        click.echo(script)
        if submit:
            click.confirm("Submit this exact displayed array plan with sbatch?", abort=True)
            subprocess.run(["sbatch"], input=script, text=True, check=True)
        return
    if submit:
        raise click.UsageError("--submit requires --scheduler slurm")
    _configure_threads(stage, cpus)
    if _budgeted_run(run["config"]):
        if budget is None:
            raise ValueError("transfer execution requires --storage-budget")
        local_count = (
            1
            if source_group is not None
            else len(
                _selected_units(
                    stage_units(
                        stage,
                        run["config"],
                        run["metadata"]["workload"],
                        inference_definition=run["metadata"].get("inference_definition"),
                    ),
                    unit,
                )
            )
        )
        inspect_transfer_storage(
            budget,
            stage=stage,
            outputs=(
                [item["root"] for item in group_runs] if source_group is not None else [run["root"]]
            ),
            unit_count=local_count,
        )
    if inference_run is not None:
        with _transfer_writer([run], stage):
            run = _inference_destination(
                source,
                inference_run,
                definition=inference_definition,
                resume=resume,
                create=True,
                damping_centres=damping_centres,
                covariance_runs=covariance_run,
            )
        run["storage_budget"] = budget
    if (
        stage == "covariance"
        and run["metadata"].get("inference_definition") is None
        and os.environ.get("BAO_COVARIANCE_PROCESS") != "1"
    ):
        environment = dict(os.environ, BAO_COVARIANCE_PROCESS="1")
        for key in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        ):
            environment[key] = "1"
        args = [
            sys.executable,
            "-m",
            "pzbao.run",
            stage,
            "--input",
            str(input_path),
            "--cpus",
            str(cpus),
        ]
        if unit is not None:
            args.extend(["--unit", unit])
        if config_path is not None:
            args.extend(["--config", str(config_path.resolve())])
        if resume:
            args.append("--resume")
        if storage_budget is not None:
            args.extend(["--storage-budget", str(storage_budget.resolve())])
        completed = subprocess.run(args, env=environment)
        if completed.returncode:
            raise click.ClickException("covariance worker reported incomplete units")
        return
    run["stage_options"] = stage_options
    result = (
        execute_source_group(group_runs, source_group, cpus=cpus, resume=resume)
        if source_group is not None
        else execute_stage(run, stage, selection=unit, cpus=cpus, resume=resume)
    )
    if stage == "sample" and review:
        from .diagnostics.review import render_review

        with _transfer_writer([run], "review"):
            render_review(run["root"])
    if result["errors"]:
        raise click.ClickException(f"{stage} has incomplete units; see saved errors")


def _add_science_command(stage):
    @main.command(name=stage)
    @click.option("--input", "input_path", required=True, type=_SavedPath(path_type=Path))
    @click.option(
        "--config",
        "config_path",
        type=_SavedPath(path_type=Path, exists=True),
        help="Stage YAML or combined experiment YAML.",
    )
    @click.option(
        "--after",
        multiple=True,
        help="Plan an afterok dependency on this Slurm job; its upstream handoff may be pending.",
    )
    @click.option(
        "--unit", help="Configured unit name or zero-based unit index; comma-separated for several."
    )
    @click.option("--cpus", type=click.IntRange(min=1), default=1, show_default=True)
    @click.option("--scheduler", type=click.Choice(["local", "slurm"]), default="local")
    @click.option("--submit", is_flag=True, help="Request submission of the displayed Slurm plan.")
    @click.option(
        "--memory", help="Slurm memory request, required for a dry plan; use measured requirements."
    )
    @click.option(
        "--walltime", help="Slurm time request, required for a dry plan; use measured requirements."
    )
    @click.option("--account", help="Slurm allocation account, required for a dry plan.")
    @click.option("--partition", help="Optional Slurm partition.")
    @click.option(
        "--storage-budget",
        type=_SavedPath(path_type=Path, exists=True),
        help="Operational campaign byte bounds and current quota evidence; required for transfer.",
    )
    @click.option(
        "--max-concurrent",
        type=click.IntRange(min=1),
        default=1,
        help="Maximum simultaneous array tasks; must fit the reviewed byte budget.",
    )
    @click.option(
        "--resume", is_flag=True, help="Reuse complete units and unchanged sampler checkpoints."
    )
    def command(**kwargs):
        """Consume the preceding stage's saved products under the same immutable snapshot."""
        _stage_command(stage, **kwargs)

    if stage in ("fit", "sample"):
        click.option(
            "--units-per-task",
            type=click.IntRange(min=1),
            default=1,
            show_default=True,
            help="Units each Slurm array task runs in order in one process.",
        )(command)
    if stage == "sample":
        click.option(
            "--review/--no-review",
            default=True,
            help="Render a review after sampling; disable when a separate review job follows.",
        )(command)
    if stage == "cluster":
        click.option(
            "--statistics",
            type=click.Choice(["xi", "pk", "both"]),
            default=None,
            help="Override the YAML statistic selector; record the resolved choice.",
        )(command)
    if stage == "reconstruct":
        click.option(
            "--source-group", help="Exact source-group selector, or all for a Slurm plan."
        )(command)
        click.option(
            "--peer-input",
            multiple=True,
            type=_SavedPath(path_type=Path),
            help="Explicit sibling selection handoff for compatible cross-run solves.",
        )(command)
        click.option(
            "--save-catalogs/--no-save-catalogs",
            default=True,
            help="Save reusable reconstructed catalogues (default).",
        )(command)
    if stage == "select":
        click.option(
            "--reuse-selection",
            type=_SavedPath(path_type=Path),
            help="Reuse a compatible immutable selected cohort with new run provenance.",
        )(command)
    click.option(
        "--tracer",
        type=click.Choice(["LRG", "BGS_BRIGHT-21.5"]),
        help="Plan only this tracer's units or source groups.",
    )(command)
    if stage in {"fit", "covariance"}:
        click.option(
            "--inference-run",
            type=click.Path(path_type=Path),
            help="Create a separate inference run using this checkout and the input's frozen products.",
        )(command)
    if stage == "fit":
        click.option(
            "--damping-centres",
            type=_SavedPath(path_type=Path, exists=True),
            help="Adopt the reviewed artifact in a new inference snapshot.",
        )(command)
    if stage in {"fit", "covariance"}:
        click.option(
            "--inference-definition",
            type=click.Choice(
                [BGS_ANISOTROPIC, *TRANSFER_INFERENCE, *CORE_INFERENCE, *UNIFIED_PRIOR_INFERENCE]
            ),
            help="Named covariance reprojection, transfer check or core power3 refit.",
        )(command)
    if stage == "fit":
        click.option(
            "--covariance-run",
            multiple=True,
            type=_SavedPath(path_type=Path, exists=True),
            help="Published BGS anisotropic covariance run read by a core power3 refit.",
        )(command)
    return command


for _stage in STAGES[1:-1]:
    _add_science_command(_stage)


@main.command(name="adopt-predecessor")
@click.option("--run", "run_root", required=True, type=_SavedPath(path_type=Path, exists=True))
@click.option(
    "--stage",
    required=True,
    type=click.Choice(REGISTERED_STAGES),
    help="Stage whose published predecessor units move here before its array is submitted.",
)
@click.option("--dry-run", is_flag=True, help="Report the stage coverage and move nothing.")
@click.option(
    "--sibling",
    "siblings",
    multiple=True,
    type=_SavedPath(path_type=Path, exists=True),
    help="Another successor of this campaign; references into its predecessor are redirected "
    "to the products it adopted (repeatable).",
)
def adopt_predecessor_command(run_root, stage, dry_run, siblings):
    """Move the recorded predecessor's published units of one stage into this successor run."""
    summary = adopt_predecessor(
        load_run(run_root),
        stage,
        dry_run=dry_run,
        siblings=[load_run(sibling) for sibling in siblings],
    )
    adopted, computed = summary["units_adopted"], summary["units_computed"]
    verb = "would adopt" if dry_run else "adopted"
    click.echo(
        f"{stage}: {verb} {len(adopted)} units of {summary['predecessor']['run_name']} "
        f"({summary['files_moved']} files, {summary['bytes_moved']} bytes); "
        f"{len(computed)} units left for the stage to compute"
    )
    if summary["units_previously_adopted"]:
        click.echo(
            f"  {len(summary['units_previously_adopted'])} units were adopted by an earlier "
            "invocation and kept"
        )
    for unit_id in computed:
        click.echo(f"  compute {unit_id}")
    if not dry_run:
        click.echo(f"receipt {run_root / stage / ADOPTION_RECEIPT}")


@main.command()
@click.option("--run", "run_root", required=True, type=_SavedPath(path_type=Path, exists=True))
@click.option("--config", "config_path", type=_SavedPath(path_type=Path, exists=True))
@click.option("--storage-budget", type=_SavedPath(path_type=Path, exists=True))
@click.option(
    "--reference-run",
    "reference_root",
    type=_SavedPath(path_type=Path, exists=True),
    help="Compare a transfer run with saved self-reconstruction fits and identical targets.",
)
def review(run_root, config_path, reference_root, storage_budget):
    """Render available products and partial coverage without running missing science."""
    from .diagnostics.review import render_review

    run = load_run(run_root)
    run["storage_budget"] = _read_yaml(storage_budget) if storage_budget is not None else None
    if config_path is not None:
        from .config.stages import read_stage_config

        read_stage_config(config_path, stage="review", baseline=run["config"])
    with _transfer_writer([run], "review"):
        render_review(run_root, reference_root=reference_root)


if __name__ == "__main__":
    main()
