"""Relocation changes file locations, never saved identities or accepted bytes."""

import hashlib
import json

import pytest

from pzbao.artifacts import resolve_saved_path


@pytest.fixture
def archive(tmp_path):
    root = tmp_path / "archive"
    root.mkdir()
    original = tmp_path / "retired" / "products" / "fit.json"
    original.parent.mkdir(parents=True)
    original.write_text("newer source bytes")
    saved = root / "artifacts" / "producer" / "products" / "fit.json"
    saved.parent.mkdir(parents=True)
    saved.write_text('{"identity": "original producer", "value": 1.01}')
    identity = dict(kind="file", sha256=hashlib.sha256(saved.read_bytes()).hexdigest())
    manifest = json.dumps(
        dict(
            verified=True,
            source_checkouts=[
                dict(worktree=str(original.parents[1]), files={"products/fit.json": identity})
            ],
        )
    ).encode()
    (root / "manifest.json").write_bytes(manifest)
    mapping = dict(
        schema_version=1,
        source_manifest_sha256=hashlib.sha256(manifest).hexdigest(),
        roots=[dict(source=str(original.parents[1]), destination="artifacts/producer")],
        files={
            str(original): dict(
                kind="file",
                path=str(saved.relative_to(root)),
                sha256=hashlib.sha256(saved.read_bytes()).hexdigest(),
            )
        },
    )
    (root / "paths.json").write_text(json.dumps(mapping))
    return root, original, saved, mapping


def test_explicit_archive_reads_original_evidence_without_mutating_sources(archive, monkeypatch):
    root, original, saved, _ = archive
    monkeypatch.delenv("BAO_RECOVERY_ARCHIVE", raising=False)
    assert resolve_saved_path(original) == original
    assert resolve_saved_path(original, archive=root) == saved
    assert json.loads(saved.read_text())["identity"] == "original producer"
    assert original.read_text() == "newer source bytes"
    monkeypatch.setenv("BAO_RECOVERY_ARCHIVE", str(root))
    assert resolve_saved_path(original.parent) == saved.parent
    assert resolve_saved_path(original) == saved


def test_archived_byte_change_is_rejected(archive):
    root, original, saved, _ = archive
    saved.write_text("changed")
    with pytest.raises(ValueError, match="evidence bytes differ"):
        resolve_saved_path(original, archive=root)


def test_recovery_archive_can_move_without_rewriting_its_map(archive, monkeypatch, tmp_path):
    root, original, saved, _ = archive
    storage = tmp_path / "storage"
    destination = storage / "archive/recovery/original"
    destination.parent.mkdir(parents=True)
    manifest = (root / "manifest.json").read_bytes()
    mapping = (root / "paths.json").read_bytes()
    relative = saved.relative_to(root)
    root.rename(destination)
    location = storage / "archive/relocation/paths.json"
    location.parent.mkdir()
    location.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "roots": [{"source": str(root), "destination": "archive/recovery/original"}],
            }
        )
    )
    monkeypatch.setenv("BAO_STORAGE_ROOT", str(storage))
    monkeypatch.setenv("BAO_RECOVERY_ARCHIVE", str(root))
    assert resolve_saved_path(original) == destination / relative
    assert (destination / "manifest.json").read_bytes() == manifest
    assert (destination / "paths.json").read_bytes() == mapping


def test_executing_source_identity_does_not_redirect_to_archived_code(archive, monkeypatch):
    from pzbao.figures.data import Sources

    root, original, saved, _ = archive
    monkeypatch.setenv("BAO_RECOVERY_ARCHIVE", str(root))
    original.parent.mkdir(parents=True, exist_ok=True)
    original.write_bytes(b"current source")
    sources = Sources()
    assert sources.read(original) == saved.read_bytes()
    assert sources.read_current(original) == b"current source"
    sources.unchanged()
    original.write_bytes(b"edited source")
    with pytest.raises(ValueError, match="source bytes changed"):
        sources.unchanged()


def test_prior_product_archive_keeps_existing_sources_and_consumer_hash_checks(
    tmp_path, monkeypatch
):
    from pzbao.figures.data import Sources

    monkeypatch.delenv("BAO_RECOVERY_ARCHIVE", raising=False)
    original = tmp_path / "products" / "campaign" / "fit.json"
    saved = tmp_path / "products" / "archive" / "originals" / "campaign" / "fit.json"
    saved.parent.mkdir(parents=True)
    saved.write_bytes(b'{"qiso": 1.01}')
    digest = hashlib.sha256(saved.read_bytes()).hexdigest()
    assert resolve_saved_path(original) == saved
    assert Sources().read(original, digest) == saved.read_bytes()
    with pytest.raises(ValueError, match="source bytes changed"):
        Sources().read(original, "0" * 64)
    original.parent.mkdir(parents=True)
    original.write_bytes(b'{"qiso": 0.99}')
    assert resolve_saved_path(original) == original


def test_changed_manifest_and_escaping_map_are_rejected(archive):
    root, original, _, mapping = archive
    mapping["files"][str(original)]["path"] = "../outside"
    (root / "paths.json").write_text(json.dumps(mapping))
    with pytest.raises(ValueError, match="leaves its archive"):
        resolve_saved_path(original, archive=root)
    (root / "manifest.json").write_text("changed")
    mapping["files"][str(original)]["path"] = "artifacts/producer/products/fit.json"
    (root / "paths.json").write_text(json.dumps(mapping))
    with pytest.raises(ValueError, match="manifest differs"):
        resolve_saved_path(original, archive=root)
