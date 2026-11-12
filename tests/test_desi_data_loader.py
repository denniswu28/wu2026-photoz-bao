"""
Unit tests for pzbao/data/desi_data_loader.py.

Tests exercise URL construction, covariance slicing, and H5 wrapper classes
without requiring network access.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Make the package importable
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = Path(__file__).parent.parent / "pzbao"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from data.desi_data_loader import (  # noqa: E402
    DATA_BASE_URL,
    MockPycorrObject,
    MockPypowerObject,
    _data_folder,
    cut_matrix,
)


@pytest.mark.parametrize(
    "module,import_root",
    [
        ("pzbao.data.desi_data_loader", _SCRIPTS_DIR.parent),
        ("data.desi_data_loader", _SCRIPTS_DIR),
    ],
)
def test_loader_imports_with_configured_storage_root(tmp_path, module, import_root):
    root = tmp_path / "storage"
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import importlib, sys; print(importlib.import_module(sys.argv[1]).CACHE_DIR)",
            module,
        ],
        cwd=tmp_path,
        env=dict(os.environ, PYTHONPATH=str(import_root), BAO_STORAGE_ROOT=str(root)),
        text=True,
        capture_output=True,
        check=True,
    )
    assert Path(completed.stdout.strip()) == root / "data/desi-observations"
    assert not root.exists()


# ---------------------------------------------------------------------------
# _data_folder helper
# ---------------------------------------------------------------------------


class TestDataFolder:
    def test_version_1_5(self):
        assert _data_folder("1.5") == "data"

    def test_version_1(self):
        assert _data_folder("1") == "data"

    def test_version_1_2(self):
        assert _data_folder("1.2") == "data_v1.2"

    def test_version_other(self):
        assert _data_folder("2.0") == "data_v2.0"


# ---------------------------------------------------------------------------
# DATA_BASE_URL
# ---------------------------------------------------------------------------


class TestConstants:
    def test_base_url(self):
        assert "data.desi.lbl.gov" in DATA_BASE_URL
        assert "full-shape-bao-clustering" in DATA_BASE_URL


# ---------------------------------------------------------------------------
# cut_matrix
# ---------------------------------------------------------------------------


class TestCutMatrix:
    def test_identity_cut(self):
        """Cutting with full range returns the original matrix."""
        n = 45  # 15 bins x 3 ells
        cov = np.eye(n)
        xcov = np.arange(15, dtype=float)
        ellscov = (0, 2, 4)
        xlim = {0: (0, 14), 2: (0, 14), 4: (0, 14)}
        result = cut_matrix(cov, xcov, ellscov, xlim)
        np.testing.assert_array_equal(result, cov)

    def test_single_ell_cut(self):
        """Cutting to one ell returns a sub-block."""
        n_bins = 10
        n = n_bins * 3
        cov = np.arange(n * n).reshape(n, n).astype(float)
        xcov = np.arange(n_bins, dtype=float)
        ellscov = (0, 2, 4)
        xlim = {0: (0, 9)}
        result = cut_matrix(cov, xcov, ellscov, xlim)
        assert result.shape == (n_bins, n_bins)

    def test_range_cut(self):
        """Cutting a range within one ell."""
        n_bins = 10
        n = n_bins * 3
        cov = np.eye(n)
        xcov = np.arange(n_bins, dtype=float)
        ellscov = (0, 2, 4)
        # Keep bins 3..7 for ell=0 only
        xlim = {0: (3, 7)}
        result = cut_matrix(cov, xcov, ellscov, xlim)
        assert result.shape == (5, 5)

    def test_assertion_on_size_mismatch(self):
        cov = np.eye(10)
        xcov = np.arange(4, dtype=float)
        with pytest.raises(AssertionError):
            cut_matrix(cov, xcov, (0, 2, 4), {0: (0, 3)})


# ---------------------------------------------------------------------------
# MockPycorrObject
# ---------------------------------------------------------------------------


class TestMockPycorrObject:
    def _make(self, n=50):
        s = np.linspace(2, 198, n)
        poles = {0: np.ones(n), 2: np.zeros(n), 4: np.zeros(n)}
        edges = np.linspace(0, 200, n + 1)
        return MockPycorrObject(s, poles, edges)

    def test_attributes(self):
        obj = self._make()
        assert len(obj.s) == 50
        assert obj.shape == (50, 3)
        assert obj.projs == [0, 2, 4]

    def test_get_corr(self):
        obj = self._make()
        xi0 = obj.get_corr(ell=0)
        np.testing.assert_array_equal(xi0, np.ones(50))

    def test_copy(self):
        obj = self._make()
        obj2 = obj.copy()
        obj2.s[0] = -999
        assert obj.s[0] != -999

    def test_select(self):
        obj = self._make()
        obj.select((50, 150))
        assert obj.s.min() >= 50
        assert obj.s.max() <= 150


# ---------------------------------------------------------------------------
# MockPypowerObject
# ---------------------------------------------------------------------------


class TestMockPypowerObject:
    def _make(self, n=40):
        k = np.linspace(0.01, 0.39, n)
        poles = {0: np.ones(n), 2: np.zeros(n), 4: np.zeros(n)}
        k_edges = np.linspace(0.0, 0.4, n + 1)
        return MockPypowerObject(k, poles, k_edges)

    def test_attributes(self):
        obj = self._make()
        assert len(obj.k) == 40
        assert obj.shape == (40, 3)
        assert obj.projs == [0, 2, 4]

    def test_copy(self):
        obj = self._make()
        obj2 = obj.copy()
        obj2.k[0] = -999
        assert obj.k[0] != -999
