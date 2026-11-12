"""CLI unit dispatch and create-only handoffs, distinct from native scientific integration tests."""

import json
from pathlib import Path
from click.testing import CliRunner
import pytest
from pzbao import run as driver
from pzbao.config import load_analysis_config


@pytest.mark.parametrize(
    "workload,counts",
    [
        (
            "pilot",
            {
                "catalog": 2,
                "select": 2,
                "reconstruct": 50,
                "cluster": 50,
                "covariance": 8,
                "fit": 104,
                "sample": 4,
            },
        ),
        (
            "full_grid",
            {
                "catalog": 2,
                "select": 2,
                "reconstruct": 250,
                "cluster": 250,
                "covariance": 100,
                "fit": 1300,
                "sample": 50,
            },
        ),
    ],
)
def test_cli_workload_dispatch_and_emitted_handoffs_from_empty_root(
    tmp_path, monkeypatch, workload, counts
):
    """Isolate dispatch/publication here; native numerical paths have separate boundary tests."""
    monkeypatch.setattr(
        driver, "_code_identity", lambda: {"code_commit": "a" * 40, "repository": str(tmp_path)}
    )
    monkeypatch.setenv("BAO_COVARIANCE_PROCESS", "1")
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    calls, reviews = ([], [])

    def science(stage, run, unit, destination, cpus):
        calls.append((stage, unit["unit_id"]))
        (destination / "dispatch.json").write_text(json.dumps(unit))
        return {"products": [{"path": "dispatch.json", "format": "json"}]}

    monkeypatch.setattr(driver, "_science", science)
    monkeypatch.setattr(driver, "_assemble_covariances", lambda *args, **kwargs: [])
    from pzbao.diagnostics import review

    monkeypatch.setattr(review, "render_review", lambda root, **kwargs: reviews.append(Path(root)))
    cli = CliRunner()
    root = tmp_path / "runs with spaces" / "workload"
    assert not root.exists()
    arguments = [
        "catalog",
        "--output-root",
        str(root.parent),
        "--run-name",
        root.name,
        "--catalog-root",
        str(tmp_path),
        "--workload",
        workload,
    ]
    for stage in driver.STAGES[:-1]:
        result = cli.invoke(driver.main, arguments)
        assert result.exit_code == 0, (stage, result.output, result.exception)
        handoff_path = root / stage / driver.HANDOFFS[stage]
        handoff = driver._read_yaml(handoff_path)
        assert handoff["expected_unit_count"] == len(handoff["units"]) == counts[stage]
        assert handoff["errors"] == []
        assert sum((observed == stage for observed, _ in calls)) == counts[stage]
        for record in handoff["units"]:
            saved = driver._read_yaml(handoff_path.parent / record["path"])
            assert saved["unit"] == record["unit"]
            product = handoff_path.parent / record["path"]
            assert json.loads((product.parent / "dispatch.json").read_text()) == record["unit"]
        arguments = [driver.STAGES[driver.STAGES.index(stage) + 1], "--input", str(handoff_path)]
    result = cli.invoke(driver.main, ["review", "--run", str(root)])
    assert result.exit_code == 0
    assert reviews == [root, root]
    before = len(calls)
    result = cli.invoke(
        driver.main, ["sample", "--input", str(root / "fit" / "fits.yaml"), "--resume"]
    )
    assert result.exit_code == 0 and len(calls) == before


def test_each_printed_array_uses_current_commands_and_configured_unit_names(tmp_path):
    config = load_analysis_config()
    for workload in ("pilot", "full_grid"):
        for stage in driver.STAGES[:-1]:
            units = driver.stage_units(stage, config, workload)
            script = driver.slurm_plan(
                stage,
                units,
                account="test-account",
                cpus=1,
                memory="4G",
                walltime="00:10:00",
                input_path=tmp_path / "previous.yaml",
                run={
                    "run_name": "new",
                    "output_root": tmp_path,
                    "catalog_root": tmp_path,
                    "workload": workload,
                },
            )
            assert f"-m pzbao.run {stage}" in script
            assert f"#SBATCH --array=0-{len(units) - 1}" in script
            assert units[-1]["unit_id"] in script
            assert "unit_ids[$SLURM_ARRAY_TASK_ID]" in script
