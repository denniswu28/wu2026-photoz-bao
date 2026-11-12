"""Stage views of the scientific YAML, with the existing analysis file as the default authority."""

from copy import deepcopy
from pathlib import Path

import yaml

from .schema_validation import _UniqueKeyLoader, same_identity

STAGES = ("catalog", "select", "reconstruct", "cluster", "covariance", "fit", "sample", "review")
#: Handoff document each stage publishes under ``<run>/<stage>/``; review publishes a report.
HANDOFFS = {
    "catalog": "catalog.yaml",
    "select": "selection.yaml",
    "reconstruct": "reconstruction.yaml",
    "cluster": "clustering.yaml",
    "covariance": "covariances.yaml",
    "fit": "fits.yaml",
    "sample": "samples.yaml",
}
#: Stages whose published units a successor run adopts from its predecessor (same definition,
#: new code) by moving them under its own identity; later stages are always computed afresh.
REGISTERED_STAGES = STAGES[:4]
#: Receipt an adoption writes under ``<run>/<stage>/``; its presence means the stage's units
#: the predecessor published now live here and the stage array may run with --resume.
ADOPTION_RECEIPT = "adopted.json"
SHARED = ("schema_version", "configuration_name", "choice_provenance", "physics", "analysis")
#: Optional keys a stage section may carry beyond the canonical baseline; the scientific
#: validator owns their content. ``selection.density_run`` binds a density successor's handoff;
#: ``selection.veff_matched_density`` exists from schema 17 on, so frozen earlier documents
#: expand against the current baseline without it and the validator requires it where due.
OPTIONAL_SECTION_KEYS = {
    ("select", "selection"): frozenset({"density_run", "veff_matched_density"})
}


def handoff_path(run_root, stage):
    """Path of a stage's published handoff inside a run root."""
    if stage not in HANDOFFS:
        raise ValueError(f"stage {stage!r} publishes no handoff document")
    return Path(run_root) / stage / HANDOFFS[stage]


def adoption_receipt(run_root, stage):
    """Path of a stage's adoption receipt inside a run root."""
    if stage not in REGISTERED_STAGES:
        raise ValueError(f"stage {stage!r} units are computed, never adopted")
    return Path(run_root) / stage / ADOPTION_RECEIPT


def stage_document(config, stage):
    """Return a complete editable section and its shared context, without unrelated stages."""
    if stage not in STAGES:
        raise ValueError(f"unknown configuration stage: {stage}")
    measurement, fitting = config["measurement"], config["fitting"]
    parts = {
        "catalog": {"tracers": config["tracers"]},
        "select": {"selection": config["selection"]},
        "reconstruct": {"measurement": {"reconstruction": measurement["reconstruction"]}},
        "cluster": {
            "statistics": "both",
            "measurement": {
                key: measurement[key] for key in ("weights", "xi", "power_spectrum_secondary")
            },
        },
        "covariance": {"fitting": {"covariance": fitting["covariance"]}},
        "fit": {
            "fitting": {
                key: value
                for key, value in fitting.items()
                if key not in {"covariance", "sampling"}
            }
        },
        "sample": {"fitting": {"sampling": fitting["sampling"]}},
        "review": {"reporting": config["reporting"]},
    }
    return deepcopy({"shared": {key: config[key] for key in SHARED}, stage: parts[stage]})


def experiment_document(config):
    """Split one complete scientific definition into the eight editor-facing stage sections."""
    result = stage_document(config, "catalog")
    for stage in STAGES[1:]:
        result[stage] = stage_document(config, stage)[stage]
    result["cluster"]["statistics"] = "both"
    return result


def expand_document(document, baseline, *, stage=None):
    """Expand complete stage sections; omitted future stages retain canonical baseline settings.

    An individual stage file must provide its shared context and complete owned section.
    Upstream run snapshots, never current defaults, provide context for downstream commands.
    """
    allowed = {"shared", *STAGES}
    if not isinstance(document, dict) or set(document) - allowed or "shared" not in document:
        raise ValueError("stage YAML requires shared and named stage sections")
    present = set(document) - {"shared"}
    if not present or (stage is not None and stage not in present):
        raise ValueError(f"stage YAML does not contain the requested {stage} section")
    if not isinstance(document["shared"], dict) or set(document["shared"]) != set(SHARED):
        raise ValueError(f"shared must supply its complete keys: {list(SHARED)}")
    result = deepcopy(baseline)
    result.update(deepcopy(document["shared"]))
    for name in present:
        fragment = deepcopy(document[name])
        if not isinstance(fragment, dict):
            raise ValueError(f"{name} must be a mapping")
        if name == "cluster":
            statistics = fragment.pop("statistics", "both")
            if statistics not in {"xi", "pk", "both"}:
                raise ValueError("cluster.statistics must be xi, pk or both")
        expected = stage_document(baseline, name)[name]
        if name == "cluster":
            expected.pop("statistics")
        # Exact section keys catch omissions; scientific validators own the nested schemas.
        if set(fragment) != set(expected):
            raise ValueError(f"{name} requires its complete section keys: {sorted(expected)}")
        for key in expected:
            optional = OPTIONAL_SECTION_KEYS.get((name, key), frozenset())
            if isinstance(expected[key], dict) and (
                not isinstance(fragment[key], dict)
                or (set(fragment[key]) ^ set(expected[key])) - optional
            ):
                raise ValueError(f"{name}.{key} has missing or unknown settings")
            if key in {"measurement", "fitting"}:
                result[key].update(deepcopy(fragment[key]))
            else:
                result[key] = deepcopy(fragment[key])
    return result


def read_stage_config(path, *, stage, baseline):
    """Read only the requested scientific section and require unchanged upstream definitions."""
    with Path(path).open() as stream:
        document = yaml.load(stream, Loader=_UniqueKeyLoader)
    if not isinstance(document, dict):
        raise ValueError("configuration must be a mapping")
    if "shared" in document:
        expanded = expand_document(document, baseline, stage=stage)
        selector = document.get("cluster", {}).get("statistics", "both")
    else:
        expanded = document
        selector = "both"
    from .schema_validation import validate_analysis_config

    validate_analysis_config(expanded)
    # A handoff fixes scientific values. Change the experiment in a new named run.
    if not same_identity(stage_document(expanded, stage), stage_document(baseline, stage)):
        raise ValueError(
            "stage scientific settings differ from the run snapshot; create a new experiment"
        )
    if any(not same_identity(expanded[key], baseline[key]) for key in SHARED):
        raise ValueError("shared settings differ from the upstream run snapshot")
    if not same_identity(expanded, baseline):
        raise ValueError("configuration differs from the run snapshot; create a new experiment")
    return {"statistics": selector} if stage == "cluster" else {}
