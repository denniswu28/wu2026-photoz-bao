"""Publisher input checks and the sole resolved snapshot digest have different jobs."""

import hashlib

import pytest

from pzbao import run as driver
from pzbao import catalog
from tests.test_catalog import _registered_inputs
from tests import test_run

linear_run = test_run.linear_run


def test_reformatting_saved_configuration_does_not_change_resolved_identity(linear_run):
    path = linear_run["root"] / "analysis.yaml"
    path.write_text("# An ordinary formatting annotation.\n\n" + path.read_text())
    assert driver.load_run(linear_run["root"])["metadata"] == linear_run["metadata"]


def test_mutating_a_saved_scientific_value_requires_a_new_named_run(linear_run):
    path = linear_run["root"] / "analysis.yaml"
    document = path.read_text()
    assert "profiler_iterations: 8" in document
    path.write_text(document.replace("profiler_iterations: 8", "profiler_iterations: 9"))
    with pytest.raises(ValueError, match="snapshot.*changed"):
        driver.load_run(linear_run["root"])


def test_raw_publisher_checksum_failure_prevents_registration(tmp_path):
    from pzbao.config import load_analysis_config
    from pathlib import Path

    config = load_analysis_config()
    records = _registered_inputs(tmp_path, config)
    path = Path(records[0]["path"])
    checksum = path.with_name(path.name + ".sha256")
    checksum.write_text(hashlib.sha256(path.read_bytes()).hexdigest() + " " + path.name)
    catalog.register_parent_inputs(tmp_path, tracer="LRG", mock_id=2, config=config)
    # Same-size changes are still visible to the publisher checksum at registration.
    content = bytearray(path.read_bytes())
    content[-1] ^= 1
    path.write_bytes(content)
    with pytest.raises(ValueError, match="publisher checksum"):
        catalog.register_parent_inputs(tmp_path, tracer="LRG", mock_id=2, config=config)
