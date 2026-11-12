"""May count-rule values, paired membership and actual-volume FKP contracts."""

import pzbao.diagnostics.selection as diag_selection
import pytest


@pytest.mark.parametrize(
    "requested,available,retained,skipped",
    [
        (94, 100, 94, False),
        (95, 100, 100, True),
        (95, 101, 101, True),
        (95, 102, 95, False),
        (110, 100, 100, True),
        (0, 0, 0, True),
    ],
)
def test_skip_boundary_and_no_upsampling(requested, available, retained, skipped):
    assert diag_selection.retained_count(requested, available, 0.05) == (retained, skipped)
