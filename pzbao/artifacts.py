"""Read relocated evidence through an explicitly selected, hash-bound archive map."""

from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path

try:
    from .storage import resolve_path
except ImportError:  # direct execution from the package directory
    from storage import resolve_path


def file_digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _inside(archive, relative):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Archive map path leaves its archive")
    candidate = archive / path
    if not candidate.resolve().is_relative_to(archive):
        raise ValueError("Archive map link leaves its archive")
    return candidate


@lru_cache(maxsize=4)
def _read_map(archive, modified, size, manifest_modified, manifest_size):
    payload = json.loads((archive / "paths.json").read_text())
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported saved-artifact path map")
    if file_digest(archive / "manifest.json") != payload["source_manifest_sha256"]:
        raise ValueError("Recovery manifest differs from the artifact path map")
    manifest = json.loads((archive / "manifest.json").read_text())
    if manifest.get("verified") is not True:
        raise ValueError("Recovery archive has not passed verification")
    sources = {
        row["worktree"]: row
        for row in manifest["source_checkouts"]
        if not row.get("external_archive_unchanged")
    }
    for entry in payload["roots"]:
        if entry["source"] not in sources:
            raise ValueError("Artifact root is absent from the recovery manifest")
        _inside(archive, entry["destination"])
    for original, entry in payload["files"].items():
        matches = [
            root for root in payload["roots"] if Path(original).is_relative_to(root["source"])
        ]
        if len(matches) != 1:
            raise ValueError("Ambiguous archived artifact identity")
        root = matches[0]
        relative = Path(original).relative_to(root["source"])
        expected = sources[root["source"]]["files"].get(str(relative))
        if expected is None or any(entry.get(key) != value for key, value in expected.items()):
            raise ValueError("Artifact identity differs from the recovery manifest")
        if entry["path"] != str(Path(root["destination"]) / relative):
            raise ValueError("Archive map path leaves its archive layout")
    return payload


def _product_archive(path):
    """Locate the repository's earlier, same-checkout products archive."""
    if path.exists():
        return path
    for parent in path.parents:
        if parent.name == "products":
            relative = path.relative_to(parent)
            if relative.parts[:2] != ("archive", "originals"):
                candidate = parent / "archive" / "originals" / relative
                if candidate.exists():
                    return candidate
            break
    return path


def resolve_saved_path(path, *, archive=None):
    """Locate saved bytes without rewriting identities or extracting files.

    Set BAO_RECOVERY_ARCHIVE, or pass archive, to read a materialized recovery
    archive. Its explicit map takes precedence for mapped files, allowing
    readback checks before retirement. Missing product paths can also resolve
    through the existing products/archive/originals layout in the same checkout.
    Consumers still check their recorded content identities.
    """
    path = Path(path)
    selected = archive or os.environ.get("BAO_RECOVERY_ARCHIVE")
    if not selected:
        return _product_archive(resolve_path(path))
    archive = resolve_path(selected).resolve()
    info = (archive / "paths.json").stat()
    manifest_info = (archive / "manifest.json").stat()
    payload = _read_map(
        archive, info.st_mtime_ns, info.st_size, manifest_info.st_mtime_ns, manifest_info.st_size
    )
    original = path.absolute()
    entry = payload["files"].get(str(original))
    if entry is not None:
        candidate = _inside(archive, entry["path"])
        if entry["kind"] == "file":
            if file_digest(candidate) != entry["sha256"]:
                raise ValueError(f"Archived evidence bytes differ: {original}")
            return candidate
        # Directory aliases are retained for browsing; native files have exact entries.
        if entry["kind"] == "symlink" and candidate.is_dir():
            return candidate
        raise ValueError(f"Unsupported archived evidence alias: {original}")
    for root in sorted(payload["roots"], key=lambda row: len(row["source"]), reverse=True):
        if original.is_relative_to(root["source"]):
            relative = Path(root["destination"]) / original.relative_to(root["source"])
            candidate = _inside(archive, relative)
            if candidate.is_dir():
                return candidate
    return _product_archive(resolve_path(path))
