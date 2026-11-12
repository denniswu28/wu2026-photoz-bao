"""DESI DR1 Data Loader — download and parse published data products.

Provides functions to fetch correlation function poles, power spectrum poles,
and covariance matrices from the DESI DR1 public data repository.  Files are
cached under ``BAO_STORAGE_ROOT/data/desi-observations/`` on first download.

The standalone reader remains available for notebook comparisons with DESI DR1.
"""

import logging
from pathlib import Path

import numpy as np
import requests

try:
    from ..storage import storage_root
except ImportError:  # standalone imports with the package directory on sys.path
    from storage import storage_root

logger = logging.getLogger(__name__)

DATA_BASE_URL = "https://data.desi.lbl.gov/public/dr1/vac/dr1/full-shape-bao-clustering"

CACHE_DIR = storage_root() / "data/desi-observations"


# ---------------------------------------------------------------------------
# Download helper
# ---------------------------------------------------------------------------


def download_file(url, local_path):
    """Download *url* to *local_path* if it doesn't already exist."""
    local_path = Path(local_path)
    if not local_path.exists():
        print(f"Downloading {url} to {local_path}...")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Download complete.")
        except Exception as e:
            print(f"Failed to download {url}: {e}")
            if local_path.exists():
                local_path.unlink()
            raise e
    else:
        print(f"File already exists: {local_path}")
    return local_path


# ---------------------------------------------------------------------------
# Version / folder helpers
# ---------------------------------------------------------------------------


def _data_folder(version="1.5"):
    """Map version string to the server-side folder name."""
    v = str(version)
    if v in ("1.5", "1"):
        return "data"
    if v == "1.2":
        return "data_v1.2"
    return f"data_v{version}"


# ---------------------------------------------------------------------------
# HDF5 → pycorr-like wrapper
# ---------------------------------------------------------------------------


class MockPycorrObject:
    """Lightweight pycorr-compatible wrapper around HDF5 correlation data."""

    def __init__(self, s, poles, edges):
        self.s = s
        self.poles = poles
        self.edges = (edges, np.linspace(-1, 1, 200))
        self.projs = list(poles.keys())
        self.shape = (len(s), len(self.projs))

    def __copy__(self):
        return MockPycorrObject(
            self.s.copy(), {k: v.copy() for k, v in self.poles.items()}, self.edges[0].copy()
        )

    def copy(self):
        return self.__copy__()

    def select(self, xlim):
        smin, smax = xlim[0], xlim[1]
        mask = (self.s >= smin) & (self.s <= smax)
        self.s = self.s[mask]
        for k in self.poles:
            self.poles[k] = self.poles[k][mask]
        return self

    def sepavg(self):
        return self.s

    def get_corr(self, ell=0, ignore_nan=False, return_cov=False, return_sep=False):
        return self.poles[ell]

    def __getitem__(self, item):
        return self


def load_h5_as_pycorr(filename):
    """Load an HDF5 correlation-poles file into a :class:`MockPycorrObject`."""
    import h5py

    with h5py.File(filename, "r") as f:
        s = f["0"]["s"][...]
        poles = {}
        for ell in [0, 2, 4]:
            if str(ell) in f:
                poles[ell] = f[str(ell)]["value"][...]

    with h5py.File(filename, "r") as f:
        if "s_edges" in f["0"]:
            s_edges = f["0"]["s_edges"][...]
            if s_edges.ndim == 2 and s_edges.shape[1] == 2:
                s_edges = np.append(s_edges[:, 0], s_edges[-1, 1])
        else:
            ds = np.diff(s).mean()
            s_edges = np.concatenate([s - ds / 2, [s[-1] + ds / 2]])

    return MockPycorrObject(s, poles, s_edges)


# ---------------------------------------------------------------------------
# HDF5 → power-spectrum-like wrapper
# ---------------------------------------------------------------------------


class MockPypowerObject:
    """Lightweight pypower-compatible wrapper around HDF5 spectrum data."""

    def __init__(self, k, poles, k_edges):
        self.k = k
        self.poles = poles
        self.k_edges = k_edges
        self.projs = list(poles.keys())
        self.shape = (len(k), len(self.projs))

    def __copy__(self):
        return MockPypowerObject(
            self.k.copy(), {k: v.copy() for k, v in self.poles.items()}, self.k_edges.copy()
        )

    def copy(self):
        return self.__copy__()


def load_h5_as_pypower(filename):
    """Load an HDF5 spectrum-poles file into a :class:`MockPypowerObject`."""
    import h5py

    with h5py.File(filename, "r") as f:
        # Try 'k' key first, fall back to 's' (some files may use 's')
        grp = f["0"]
        if "k" in grp:
            k = grp["k"][...]
        elif "s" in grp:
            k = grp["s"][...]
        else:
            raise KeyError(f"No 'k' or 's' dataset in {filename}/0. Keys: {list(grp.keys())}")

        poles = {}
        for ell in [0, 2, 4]:
            if str(ell) in f:
                poles[ell] = f[str(ell)]["value"][...]

    with h5py.File(filename, "r") as f:
        grp = f["0"]
        if "k_edges" in grp:
            k_edges = grp["k_edges"][...]
            if k_edges.ndim == 2 and k_edges.shape[1] == 2:
                k_edges = np.append(k_edges[:, 0], k_edges[-1, 1])
        elif "s_edges" in grp:
            k_edges = grp["s_edges"][...]
            if k_edges.ndim == 2 and k_edges.shape[1] == 2:
                k_edges = np.append(k_edges[:, 0], k_edges[-1, 1])
        else:
            dk = np.diff(k).mean()
            k_edges = np.concatenate([k - dk / 2, [k[-1] + dk / 2]])

    return MockPypowerObject(k, poles, k_edges)


# ---------------------------------------------------------------------------
# Correlation function poles (xi)
# ---------------------------------------------------------------------------


def read_xi_poles(
    tracer="LRG",
    region="GCcomb",
    version="1.5",
    zmin=0.4,
    zmax=0.6,
    smin=0,
    smax=200,
    recon_algorithm=None,
    recon_mode="recsym",
    smoothing_radius=15,
    concatenate=False,
    ells=[0, 2, 4],
    cache_dir=None,
):
    """Download and load DESI DR1 correlation function multipoles.

    Parameters
    ----------
    tracer : str
        Galaxy tracer (e.g. ``"LRG"``).
    region : str
        Sky region (``"GCcomb"``, ``"NGC"``, ``"SGC"``).
    version : str
        Data release version (``"1.5"`` or ``"1"`` → ``data/``).
    zmin, zmax : float
        Redshift bin edges.
    recon_algorithm : str or None
        Set to ``"recsym"`` for post-reconstruction.  ``None`` for pre-recon.
    cache_dir : Path or None
        Override default cache directory.

    Returns
    -------
    MockPycorrObject
    """
    from pycorr import TwoPointCorrelationFunction

    data_folder = _data_folder(version)
    z_str = f"{zmin:.1f}-{zmax:.1f}"

    if recon_algorithm:
        filename = f"correlation-recsym-poles_{tracer}_{region}_z{z_str}.h5"
        url_path = f"v1.0/{data_folder}/recsym/correlation/{filename}"
    else:
        filename = f"correlation-poles_{tracer}_{region}_z{z_str}.h5"
        url_path = f"v1.0/{data_folder}/correlation/{filename}"

    full_url = f"{DATA_BASE_URL}/{url_path}"
    _cache = Path(cache_dir) if cache_dir else CACHE_DIR
    local_file = _cache / filename

    download_file(full_url, local_file)

    if str(local_file).endswith(".h5"):
        return load_h5_as_pycorr(local_file)
    else:
        return TwoPointCorrelationFunction.load(local_file)


# ---------------------------------------------------------------------------
# Power spectrum poles (pk)
# ---------------------------------------------------------------------------


def read_pk_poles(
    tracer="LRG",
    region="GCcomb",
    version="1.5",
    zmin=0.4,
    zmax=0.6,
    cache_dir=None,
):
    """Download and load DESI DR1 power spectrum multipoles.

    Parameters
    ----------
    tracer : str
        Galaxy tracer (e.g. ``"LRG"``).
    region : str
        Sky region (``"GCcomb"``, ``"NGC"``, ``"SGC"``).
    version : str
        Data release version.
    zmin, zmax : float
        Redshift bin edges.
    cache_dir : Path or None
        Override default cache directory.

    Returns
    -------
    MockPypowerObject
    """
    data_folder = _data_folder(version)
    z_str = f"{zmin:.1f}-{zmax:.1f}"

    filename = f"spectrum-poles_{tracer}_{region}_z{z_str}.h5"
    url_path = f"v1.0/{data_folder}/spectrum/{filename}"

    full_url = f"{DATA_BASE_URL}/{url_path}"
    _cache = Path(cache_dir) if cache_dir else CACHE_DIR
    local_file = _cache / filename

    download_file(full_url, local_file)
    return load_h5_as_pypower(local_file)


# ---------------------------------------------------------------------------
# Covariance matrices
# ---------------------------------------------------------------------------


def read_xi_cov(
    tracer="LRG",
    region="GCcomb",
    version="1.5",
    zmin=0.4,
    zmax=0.6,
    cov_type="RascalC",
    ells=(0, 2, 4),
    smin=0,
    smax=200,
    recon_algorithm=None,
    recon_mode="recsym",
    smoothing_radius=15,
    reference_data=None,
    cache_dir=None,
):
    """Download and load DESI DR1 xi covariance matrix.

    Parameters
    ----------
    tracer, region, version, zmin, zmax :
        Tracer identification (same as :func:`read_xi_poles`).
    cov_type : str
        ``"RascalC"`` (default) or ``"EZmock"``.
    ells : tuple
        Multipole orders present in the covariance.
    smin, smax : float
        Separation range for cutting the matrix.
    recon_algorithm : str or None
        ``"recsym"`` for post-reconstruction covariance.
    reference_data : MockPycorrObject or None
        If provided, the covariance is cut to match the reference
        separation centres.
    cache_dir : Path or None
        Override default cache directory.

    Returns
    -------
    ndarray or None
    """
    data_folder = _data_folder(version)
    z_str = f"{zmin:.1f}-{zmax:.1f}"

    if cov_type.startswith("RascalC"):
        if recon_algorithm:
            filename = f"covariance_correlation-recsym-poles_{tracer}_{region}_z{z_str}.h5"
        else:
            filename = f"covariance_correlation-poles_{tracer}_{region}_z{z_str}.h5"

        url_path = f"v1.0/{data_folder}/covariance/RascalC/{filename}"
        full_url = f"{DATA_BASE_URL}/{url_path}"
        _cache = Path(cache_dir) if cache_dir else CACHE_DIR
        local_file = _cache / filename

        download_file(full_url, local_file)

        import h5py

        with h5py.File(local_file, "r") as f:
            if "covariance" in f:
                cov = f["covariance"][...]
            elif "value" in f:
                cov = f["value"][...]
            else:
                raise KeyError(
                    f"Could not find 'covariance' or 'value' dataset in "
                    f"{local_file}. Keys: {list(f.keys())}"
                )

        if reference_data is not None:
            n_bins_cov = len(cov) // 3

            # Support both MockPycorrObject (.s) and real pycorr objects (.sepavg())
            if hasattr(reference_data, "s"):
                s_ref = np.asarray(reference_data.s)
            else:
                s_ref = np.asarray(reference_data.sepavg())
            if len(s_ref) == n_bins_cov:
                s_centers = s_ref
            elif len(s_ref) > n_bins_cov:
                s_ref_ge20 = s_ref[s_ref >= 20]
                if len(s_ref_ge20) == n_bins_cov:
                    s_centers = s_ref_ge20
                else:
                    s_centers = s_ref[-n_bins_cov:]
            else:
                if n_bins_cov == 45:
                    s_edges = np.linspace(20, 200, 46)
                    s_centers = (s_edges[1:] + s_edges[:-1]) / 2.0
                else:
                    s_centers = s_ref

            slim = {ell: (smin, smax) for ell in ells}
            cov = cut_matrix(cov, s_centers, (0, 2, 4), slim)

        return cov

    elif cov_type == "EZmock":
        print("Warning: EZmock covariance requires downloading 1000 mock files. Skipping.")
        return None

    return None


def read_pk_cov(
    tracer="LRG",
    region="GCcomb",
    version="1.5",
    zmin=0.4,
    zmax=0.6,
    cache_dir=None,
):
    """Download and load DESI DR1 power spectrum covariance (EZmock).

    Parameters
    ----------
    tracer, region, version, zmin, zmax :
        Tracer identification.
    cache_dir : Path or None
        Override default cache directory.

    Returns
    -------
    ndarray or None
    """
    data_folder = _data_folder(version)
    z_str = f"{zmin:.1f}-{zmax:.1f}"

    filename = f"covariance_spectrum-poles_{tracer}_{region}_z{z_str}.h5"
    url_path = f"v1.0/{data_folder}/covariance/EZmock/{filename}"
    full_url = f"{DATA_BASE_URL}/{url_path}"
    _cache = Path(cache_dir) if cache_dir else CACHE_DIR
    local_file = _cache / filename

    download_file(full_url, local_file)

    import h5py

    with h5py.File(local_file, "r") as f:
        if "covariance" in f:
            cov = f["covariance"][...]
        elif "value" in f:
            cov = f["value"][...]
        else:
            raise KeyError(
                f"Could not find 'covariance' or 'value' dataset in "
                f"{local_file}. Keys: {list(f.keys())}"
            )

    return cov


# ---------------------------------------------------------------------------
# Covariance slicing
# ---------------------------------------------------------------------------


def cut_matrix(cov, xcov, ellscov, xlim):
    """Cut a block-diagonal covariance matrix to selected ell/x ranges.

    Parameters
    ----------
    cov : ndarray
        Full covariance matrix (n_ell * n_x, n_ell * n_x).
    xcov : ndarray
        Separation (or wavenumber) centres for each block.
    ellscov : tuple
        Multipole orders in the covariance (e.g. ``(0, 2, 4)``).
    xlim : dict
        ``{ell: (xmin, xmax)}`` selection per multipole.

    Returns
    -------
    ndarray
        Sliced covariance matrix.
    """
    assert len(cov) == len(xcov) * len(
        ellscov
    ), "Input matrix has size {}, different than {} x {}".format(len(cov), len(xcov), len(ellscov))
    indices = []
    for ell, lim in xlim.items():
        index = ellscov.index(ell) * len(xcov) + np.arange(len(xcov))
        index = index[(xcov >= lim[0]) & (xcov <= lim[1])]
        indices.append(index)
    indices = np.concatenate(indices, axis=0)
    return cov[np.ix_(indices, indices)]
