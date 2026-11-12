"""Explicit historical defaults for broad-field saved-format regression fixtures."""

from pathlib import Path

from pzbao.config import load_analysis_config as _load

_FIXTURES = Path(__file__).with_name("fixtures")


def load_analysis_config(path=None):
    return _load(path or _FIXTURES / "analysis_schema15.yaml")


def load_schema16_config():
    """The per-bin definition before the V_eff-matched density block (schema 16, revision 2.13)."""
    return _load(_FIXTURES / "analysis_schema16.yaml")
