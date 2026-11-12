"""Campaign admission crosses the actual CLI, physical storage and process-lock boundaries."""

import copy
from datetime import datetime, timedelta, timezone
import errno
import fcntl
import json
import os
from pathlib import Path
import re
import select
import subprocess
import sys
import tempfile

from click.testing import CliRunner
import pytest
import yaml

from pzbao import run as runner, storage
from pzbao.config import load_analysis_config
from pzbao.config.transfer import transfer_run_configs
from tests.test_grouped_transfer import (  # noqa: F401 -- shares pytest fixtures
    CODE,
    resource_budget,
    selector,
    siblings,
    native_stub,
)


def test_inventory_restarts_after_a_staging_file_is_renamed(tmp_path, monkeypatch):
    staging = tmp_path / "report.partial.pdf"
    staging.write_bytes(b"x" * 65536)
    published = tmp_path / "report.pdf"
    original = Path.resolve
    renamed = []

    def publish_before_resolve(path, *args, **kwargs):
        if path == staging and not renamed:
            staging.rename(published)
            renamed.append(True)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", publish_before_resolve)
    measured = storage.allocated_campaign_bytes([tmp_path])
    assert renamed and measured["logical_bytes"] == 65536
    assert measured["allocated_bytes"] == sum(
        p.stat().st_blocks * 512 for p in (tmp_path, published)
    )
    assert measured["unique_inodes"] == 2


def test_inventory_churn_has_a_bound_and_missing_inputs_still_fail(tmp_path, monkeypatch):
    with pytest.raises(FileNotFoundError):
        storage.allocated_campaign_bytes([tmp_path / "missing"])
    broken = tmp_path / "broken-link"
    broken.symlink_to(tmp_path / "missing")
    with pytest.raises(FileNotFoundError):
        storage.allocated_campaign_bytes([tmp_path])
    original = Path.resolve
    calls = []

    def disappearing(path, *args, **kwargs):
        if path == broken:
            # A disappearing directory entry, rather than a persistent broken link.
            calls.append(path)
            raise FileNotFoundError(str(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", disappearing)
    monkeypatch.setattr(Path, "is_symlink", lambda path: False)
    with pytest.raises(FileNotFoundError, match="inventory"):
        storage.allocated_campaign_bytes([tmp_path])
    assert len(calls) == 3


def test_reservation_is_not_visible_until_its_json_is_complete(tmp_path, monkeypatch):
    original = storage.json.dump
    observed = []

    def reader_during_write(record, stream, *args, **kwargs):
        # A remote reader must never discover an empty published reservation while
        # its owner retains the lifetime lock and the NFS write descriptor.
        for path in (tmp_path / ".storage-writers").glob("*/reservation.json"):
            observed.append(json.loads(path.read_text()))
        return original(record, stream, *args, **kwargs)

    monkeypatch.setattr(storage.json, "dump", reader_during_write)
    budget = resource_budget(tmp_path)
    with storage.transfer_storage_writer(budget, stage="select", outputs=[tmp_path]) as receipt:
        record = json.loads(Path(receipt["reservation"]).read_text())
        assert record["reserved_bytes"] == budget["writer_bounds"]["select"]
        assert record["outputs"] == [str(tmp_path)]
    assert observed == []


@pytest.mark.parametrize(
    "change,match",
    [
        ("missing_component", "complete campaign components"),
        ("missing_stage", "no bounded writer"),
        ("expired", "expired"),
        ("future", "future-dated"),
        ("naive", "timezone"),
        ("missing_evidence", "evidence"),
        ("exclude_owned", "cannot be excluded"),
        ("missing_root", "complete campaign root"),
        ("negative_bytes", "integer byte bounds"),
    ],
)
def test_invalid_capacity_fails_before_any_write(tmp_path, change, match):
    budget = resource_budget(tmp_path)
    observed = datetime.now(timezone.utc)
    if change == "missing_component":
        budget["remaining_components"].pop("peak_staging")
    elif change == "missing_stage":
        budget["writer_bounds"].pop("reconstruct")
    elif change == "expired":
        budget["valid_until"] = (observed - timedelta(seconds=1)).isoformat()
    elif change == "future":
        budget["observed_at"] = (observed + timedelta(minutes=30)).isoformat()
    elif change == "naive":
        budget["observed_at"] = observed.replace(tzinfo=None).isoformat()
    elif change == "missing_evidence":
        budget["evidence"]["quota"] = ""
    elif change == "exclude_owned":
        owned = tmp_path / "failed-attempt"
        owned.mkdir()
        budget["shared_source_roots"] = [str(owned)]
    elif change == "missing_root":
        budget["accounting_roots"] = []
    elif change == "negative_bytes":
        budget["writer_bounds"]["reconstruct"] = -1
    before = set(tmp_path.iterdir())
    with pytest.raises(ValueError, match=match):
        with storage.transfer_storage_writer(budget, stage="reconstruct", outputs=[tmp_path]):
            pytest.fail("an invalid budget reached the producer")
    assert set(tmp_path.iterdir()) == before


def test_output_escape_and_unknown_budget_fail_closed(tmp_path):
    budget = resource_budget(tmp_path)
    with pytest.raises(ValueError, match="leaves"):
        storage.inspect_transfer_storage(
            budget, stage="select", outputs=[tmp_path.parent / "outside"]
        )
    with pytest.raises(ValueError, match="complete version-1"):
        storage.inspect_transfer_storage(None, stage="select", outputs=[tmp_path])
    assert not (tmp_path / ".storage-writers").exists()


def test_read_only_projection_counts_failures_and_quota_growth(tmp_path, monkeypatch):
    budget = resource_budget(tmp_path)
    budget["max_concurrent"] = 8
    bound = budget["writer_bounds"]["reconstruct"]
    additional = sum(budget["remaining_components"].values())
    budget["observed_allocated_bytes"] = storage.allocated_campaign_bytes([tmp_path])[
        "allocated_bytes"
    ]
    budget["quota_headroom_bytes"] = additional + 2 * bound + 1024**2
    # Failed/staging bytes count before a new producer may enter.
    (tmp_path / "failed.bin").write_bytes(b"x" * (2 * 1024**2))
    before = set(tmp_path.iterdir())
    plan = storage.inspect_transfer_storage(
        budget, stage="reconstruct", outputs=[tmp_path], requested_concurrency=8
    )
    assert plan["max_concurrent"] == 1
    assert plan["quota_headroom_bytes"] < budget["quota_headroom_bytes"] - 1024**2
    assert plan["conservative_peak_bytes"] <= plan["storage_limit_bytes"]
    assert set(tmp_path.iterdir()) == before
    monkeypatch.setattr(storage.shutil, "disk_usage", lambda path: type("Disk", (), {"free": 1})())
    with pytest.raises(ValueError, match="no bounded writer"):
        storage.inspect_transfer_storage(budget, stage="reconstruct", outputs=[tmp_path])


def test_free_space_floor_bounds_admission_with_plentiful_quota(tmp_path, monkeypatch):
    budget = resource_budget(tmp_path)
    bound = budget["writer_bounds"]["reconstruct"]
    budget["quota_headroom_bytes"] = 10 * storage.XDISK_FREE_FLOOR_BYTES
    budget["free_space_floor_bytes"] = storage.XDISK_FREE_FLOOR_BYTES
    additional = sum(budget["remaining_components"].values())

    def free(nbytes):
        disk = type("Disk", (), {"free": nbytes})()
        monkeypatch.setattr(storage.shutil, "disk_usage", lambda path: disk)

    # Past the former 3 TiB cap the floor alone bounds the campaign: one writer's bound of free
    # space above it admits the writer, one byte less refuses it, and none at all leaves no
    # room for the planned volume.
    budget["observed_allocated_bytes"] = 4 * 1024**4
    monkeypatch.setattr(
        storage, "allocated_campaign_bytes", lambda *a, **k: {"allocated_bytes": 4 * 1024**4}
    )
    assert additional < bound
    free(storage.XDISK_FREE_FLOOR_BYTES + bound)
    plan = storage.inspect_transfer_storage(budget, stage="reconstruct", outputs=[tmp_path])
    assert plan["max_concurrent"] == 1
    assert plan["storage_limit_bytes"] == 4 * 1024**4 + bound
    free(storage.XDISK_FREE_FLOOR_BYTES + bound - 1)
    with pytest.raises(ValueError, match="no bounded writer"):
        storage.inspect_transfer_storage(budget, stage="reconstruct", outputs=[tmp_path])
    free(storage.XDISK_FREE_FLOOR_BYTES)
    with pytest.raises(ValueError, match="complete campaign peak"):
        storage.inspect_transfer_storage(budget, stage="reconstruct", outputs=[tmp_path])
    # A budget without the field predates the floor and keeps every free byte available.
    del budget["free_space_floor_bytes"]
    assert storage.inspect_transfer_storage(budget, stage="reconstruct", outputs=[tmp_path])
    with pytest.raises(ValueError, match="nonnegative integer"):
        storage.inspect_transfer_storage(
            {**budget, "free_space_floor_bytes": -1}, stage="reconstruct", outputs=[tmp_path]
        )


def test_temporary_files_remain_counted_and_environment_restores_on_failure(tmp_path):
    budget = resource_budget(tmp_path)
    old = {key: os.environ.get(key) for key in ("TMPDIR", "JOBLIB_TEMP_FOLDER")}
    old_tempdir = tempfile.tempdir
    with pytest.raises(RuntimeError, match="fixture failure"):
        with storage.transfer_storage_writer(budget, stage="select", outputs=[tmp_path]) as receipt:
            temporary = Path(tempfile.gettempdir())
            assert temporary.is_relative_to(tmp_path)
            retained = temporary / "failed-staging.bin"
            retained.write_bytes(b"x" * 65536)
            assert os.environ["JOBLIB_TEMP_FOLDER"] == str(temporary)
            with storage.transfer_storage_writer(
                budget, stage="select", outputs=[tmp_path]
            ) as nested:
                assert nested == receipt
            with pytest.raises(ValueError, match="nested writer"):
                with storage.transfer_storage_writer(
                    budget, stage="select", outputs=[tmp_path], unit_count=2
                ):
                    pytest.fail("nested execution exceeded its reservation")
            raise RuntimeError("fixture failure")
    assert retained.stat().st_size == 65536
    assert tempfile.tempdir == old_tempdir
    assert {key: os.environ.get(key) for key in old} == old
    projected = storage.inspect_transfer_storage(budget, stage="select", outputs=[tmp_path])
    assert projected["active_writers"] == 0 and projected["existing_allocated_bytes"] >= 65536


def test_independent_process_reservations_bound_concurrency_and_survive_abrupt_exit(tmp_path):
    budget = resource_budget(tmp_path)
    child = r"""
import json, os, sys
from pathlib import Path
from pzbao.storage import transfer_storage_writer
budget = json.loads(sys.argv[1])
with transfer_storage_writer(budget, stage="reconstruct", outputs=[budget["campaign_root"]]) as receipt:
    (Path(os.environ["TMPDIR"]) / "retained.bin").write_bytes(b"x" * 65536)
    print(json.dumps(receipt), flush=True)
    sys.stdin.readline()
    os._exit(17)
"""
    process = subprocess.Popen(
        [sys.executable, "-c", child, json.dumps(budget)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert select.select([process.stdout], [], [], 20)[
            0
        ], "fixture worker did not report admission"
        line = process.stdout.readline()
        assert line, "fixture worker failed before admission"
        child_receipt = json.loads(line)
        plan = storage.inspect_transfer_storage(
            budget, stage="reconstruct", outputs=[tmp_path], requested_concurrency=8
        )
        assert plan["active_writers"] == 1 and plan["max_concurrent"] == 1
        assert plan["inflight_reserved_bytes"] == budget["writer_bounds"]["reconstruct"]
        changed = copy.deepcopy(budget)
        changed["max_concurrent"] += 1
        refreshed = storage.inspect_transfer_storage(
            changed, stage="reconstruct", outputs=[tmp_path], requested_concurrency=8
        )
        assert refreshed["max_concurrent"] == 1  # Preserve the old writer's smaller limit.
        assert refreshed["inflight_reserved_bytes"] == plan["inflight_reserved_bytes"]
        with storage.transfer_storage_writer(budget, stage="reconstruct", outputs=[tmp_path]):
            with pytest.raises(ValueError, match="concurrency is exhausted"):
                storage.inspect_transfer_storage(budget, stage="reconstruct", outputs=[tmp_path])
    finally:
        stdout, stderr = process.communicate(input="finish\n", timeout=20)
    assert process.returncode == 17, (stdout, stderr)
    record = Path(child_receipt["reservation"])
    assert record.is_file() and (record.parent / "temporary/retained.bin").stat().st_size == 65536
    assert (
        storage.inspect_transfer_storage(budget, stage="reconstruct", outputs=[tmp_path])[
            "active_writers"
        ]
        == 0
    )


def test_exclusive_storage_locks_are_write_open_and_own_reservation_is_never_reopened(
    tmp_path, monkeypatch
):
    """Emulate the NFS write-open requirement even when fixtures run on local XFS."""
    original_flock = storage.fcntl.flock
    original_open = Path.open
    protected = set()

    def nfs_flock(stream, operation):
        if operation & fcntl.LOCK_EX:
            assert fcntl.fcntl(stream, fcntl.F_GETFL) & os.O_ACCMODE != os.O_RDONLY
        return original_flock(stream, operation)

    def guarded_open(path, *args, **kwargs):
        assert path not in protected, "reopening a locally owned lock could release it"
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(storage.fcntl, "flock", nfs_flock)
    monkeypatch.setattr(Path, "open", guarded_open)
    budget = resource_budget(tmp_path)
    for _ in range(2):
        with storage.transfer_storage_writer(budget, stage="select", outputs=[tmp_path]) as row:
            protected.add(Path(row["reservation"]))
            state = storage.inspect_transfer_storage(budget, stage="select", outputs=[tmp_path])
            assert state["active_writers"] == 1
            assert state["inflight_reserved_bytes"] == budget["writer_bounds"]["select"]
            protected.clear()
        assert (
            storage.inspect_transfer_storage(budget, stage="select", outputs=[tmp_path])[
                "active_writers"
            ]
            == 0
        )


def test_unknown_lock_state_fails_closed_without_starting_a_writer(tmp_path, monkeypatch):
    budget = resource_budget(tmp_path)
    with storage.transfer_storage_writer(budget, stage="select", outputs=[tmp_path]):
        pass
    original = storage.fcntl.flock

    def broken_remote_lock(stream, operation):
        if Path(stream.name).name == "reservation.json":
            raise OSError(errno.ENOLCK, "NFS lock manager unavailable")
        return original(stream, operation)

    monkeypatch.setattr(storage.fcntl, "flock", broken_remote_lock)
    with pytest.raises(ValueError, match="cannot establish"):
        storage.inspect_transfer_storage(budget, stage="select", outputs=[tmp_path])
    assert len(list((tmp_path / ".storage-writers").glob("*/reservation.json"))) == 1


def test_admission_moves_released_records_aside_and_keeps_live_ones(tmp_path):
    """Every admission probes every record under the lock (2026-10-21: 6,013 records)."""
    budget = resource_budget(tmp_path)
    directory = tmp_path / ".storage-writers"
    with storage.transfer_storage_writer(budget, stage="reconstruct", outputs=[tmp_path]) as row:
        record = json.loads(Path(row["reservation"]).read_text())
    released = Path(row["reservation"]).parent
    # A sibling's live writer: its own open file description holds the lifetime lock.
    live = directory / "sibling-writer"
    live.mkdir()
    (live / "reservation.json").write_text(json.dumps(record))
    holder = (live / "reservation.json").open("r+")
    fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        # Planning counts the live writer and moves nothing.
        state = storage.inspect_transfer_storage(budget, stage="reconstruct", outputs=[tmp_path])
        assert state["active_writers"] == 1 and released.is_dir()
        with storage.transfer_storage_writer(
            budget, stage="reconstruct", outputs=[tmp_path]
        ) as admitted:
            assert admitted["active_writers"] == 1
            assert admitted["inflight_reserved_bytes"] == budget["writer_bounds"]["reconstruct"]
            kept = directory / "released" / released.name / "reservation.json"
            assert not released.exists() and json.loads(kept.read_text()) == record
            assert (live / "reservation.json").is_file()
            assert Path(admitted["reservation"]).is_file()
    finally:
        holder.close()
    # The released writer and the admission's own record stay where they are until the
    # next admission; the probe now reaches two records instead of three.
    assert sorted(path.parent.name for path in directory.glob("*/reservation.json")) == sorted(
        [live.name, Path(admitted["reservation"]).parent.name]
    )


def test_a_failed_move_leaves_the_record_and_still_admits(tmp_path, monkeypatch):
    budget = resource_budget(tmp_path)
    with storage.transfer_storage_writer(budget, stage="select", outputs=[tmp_path]) as row:
        pass

    original = storage.os.replace

    def refuse(source, destination):
        if Path(destination).parent.name == "released":
            raise OSError(errno.EACCES, "NFS refused the rename")
        return original(source, destination)

    monkeypatch.setattr(storage.os, "replace", refuse)
    with storage.transfer_storage_writer(budget, stage="select", outputs=[tmp_path]) as admitted:
        assert admitted["active_writers"] == 0
    assert Path(row["reservation"]).is_file()


def hold_admission_lock(tmp_path):
    """Hold the campaign admission lock through a separate open file, as a sibling task would."""
    directory = tmp_path / ".storage-writers"
    directory.mkdir(exist_ok=True)
    holder = (directory / "admission.lock").open("a+")
    fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return holder


def test_sibling_admission_waits_for_a_held_lock_then_admits(tmp_path, monkeypatch):
    import threading
    import time

    monkeypatch.setattr(storage, "ADMISSION_POLL_SECONDS", 0.05)
    budget = resource_budget(tmp_path)
    holder = hold_admission_lock(tmp_path)
    release = threading.Timer(0.5, lambda: (fcntl.flock(holder, fcntl.LOCK_UN), holder.close()))
    release.start()
    started = time.monotonic()
    try:
        with storage.transfer_storage_writer(budget, stage="select", outputs=[tmp_path]) as row:
            assert Path(row["reservation"]).is_file()
    finally:
        release.join()
    assert time.monotonic() - started >= 0.4


def test_admission_lock_held_past_the_former_bound_still_admits_once_released(
    tmp_path, monkeypatch, capsys
):
    """A sibling's admission is waited out however long it takes, and the wait is reported.

    The one-hour bound of 2026-10-14 (ADMISSION_WAIT_SECONDS) failed 82 transfer tasks on
    2026-10-16 that were merely queued behind their siblings. The clock is simulated and
    advances only through the waiter's own sleeps, so nothing here sleeps for real.
    """
    import time as real_time

    former_bound = 3600.0
    held_for = 2 * former_bound + 100.0
    poll = 100.0
    monkeypatch.setattr(storage, "ADMISSION_POLL_SECONDS", poll)
    budget = resource_budget(tmp_path)
    holder = hold_admission_lock(tmp_path)

    class SimulatedClock:
        def __init__(self):
            self.now = 0.0
            self.slept = []

        def monotonic(self):
            return self.now

        def sleep(self, seconds):
            self.slept.append(seconds)
            self.now += seconds
            if self.now >= held_for and not holder.closed:
                fcntl.flock(holder, fcntl.LOCK_UN)
                holder.close()

        def __getattr__(self, name):
            return getattr(real_time, name)

    clock = SimulatedClock()
    monkeypatch.setattr(storage, "time", clock)
    started = real_time.monotonic()
    with storage.transfer_storage_writer(budget, stage="select", outputs=[tmp_path]) as row:
        assert Path(row["reservation"]).is_file()
    assert holder.closed
    assert clock.slept == [poll] * int(held_for / poll)
    assert clock.now == held_for > former_bound
    assert real_time.monotonic() - started < 5.0
    lock = tmp_path / ".storage-writers" / "admission.lock"
    note = re.compile(r"storage admission: .*waited (\d+) s for " + re.escape(str(lock)))
    waited = [int(note.fullmatch(line).group(1)) for line in capsys.readouterr().err.splitlines()]
    interval = storage.ADMISSION_PROGRESS_SECONDS
    assert interval % poll == 0, "notes are expected on poll boundaries"
    assert waited == [int(k * interval) for k in range(1, int(held_for // interval) + 1)]
    assert waited[-1] > former_bound


def test_admission_lock_errors_fail_at_once_without_waiting(tmp_path, monkeypatch):
    import time

    budget = resource_budget(tmp_path)
    original = storage.fcntl.flock

    def broken_remote_lock(stream, operation):
        if Path(stream.name).name == "admission.lock":
            raise OSError(errno.ENOLCK, "NFS lock manager unavailable")
        return original(stream, operation)

    monkeypatch.setattr(storage.fcntl, "flock", broken_remote_lock)
    started = time.monotonic()
    with pytest.raises(ValueError, match="no automatic retry"):
        with storage.transfer_storage_writer(budget, stage="select", outputs=[tmp_path]):
            pass
    assert time.monotonic() - started < storage.ADMISSION_POLL_SECONDS
    assert not list((tmp_path / ".storage-writers").glob("*/reservation.json"))


def test_release_settles_at_most_its_bound_and_all_of_it_when_unmeasurable(tmp_path, monkeypatch):
    budget = resource_budget(tmp_path)
    bound = budget["writer_bounds"]["select"]
    directory = tmp_path / ".storage-writers"
    with storage.transfer_storage_writer(budget, stage="select", outputs=[tmp_path]):
        (tmp_path / "large.bin").write_bytes(b"x" * (bound + 1024**2))
    assert json.loads((directory / "inventory.json").read_text())["settled_bytes"] == bound

    def unreadable(roots, since_ns):
        raise PermissionError("outputs unreadable")

    monkeypatch.setattr(storage, "_bytes_changed_since", unreadable)
    with storage.transfer_storage_writer(budget, stage="select", outputs=[tmp_path]):
        pass
    assert json.loads((directory / "inventory.json").read_text())["settled_bytes"] == 2 * bound


def test_finished_writer_release_settle_stays_bounded_under_an_orphaned_lock(
    tmp_path, monkeypatch, capsys
):
    """A finished writer waits at most INVENTORY_REUSE_SECONDS to settle, then returns.

    Handoff publication runs inside the writer, so a lock orphaned at release must not turn a
    published stage into a walltime failure. The holder never releases and the clock is
    simulated: the writer gives up at the inventory window, leaves the settled bytes
    unchanged, releases its reservation lock and reports the wait while it lasts.
    """
    import time as real_time

    poll = 100.0
    bound = storage.INVENTORY_REUSE_SECONDS
    interval = storage.ADMISSION_PROGRESS_SECONDS
    assert bound % poll == 0 and interval % poll == 0, "events are expected on poll boundaries"
    monkeypatch.setattr(storage, "ADMISSION_POLL_SECONDS", poll)
    budget = resource_budget(tmp_path)
    directory = tmp_path / ".storage-writers"

    class SimulatedClock:
        def __init__(self):
            self.now = 0.0
            self.slept = []

        def monotonic(self):
            return self.now

        def sleep(self, seconds):
            self.slept.append(seconds)
            self.now += seconds
            assert self.now <= bound, "the finished writer is still waiting past the window"

        def __getattr__(self, name):
            return getattr(real_time, name)

    clock = SimulatedClock()
    monkeypatch.setattr(storage, "time", clock)
    holder = None
    started = real_time.monotonic()
    try:
        with storage.transfer_storage_writer(budget, stage="select", outputs=[tmp_path]) as row:
            reservation = Path(row["reservation"])
            settled = json.loads((directory / "inventory.json").read_text())["settled_bytes"]
            # Admission released the lock before yielding; an orphaned sibling now holds it.
            holder = hold_admission_lock(tmp_path)
        assert real_time.monotonic() - started < 5.0
        assert clock.now == bound
        assert clock.slept == [poll] * int(bound / poll)
        # The bound was never settled into the record, and the reservation lock is released.
        assert budget["writer_bounds"]["select"] > 0
        assert json.loads((directory / "inventory.json").read_text())["settled_bytes"] == settled
        assert str(reservation) not in storage._OWN_STORAGE_RESERVATIONS
        with reservation.open("r+") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock = directory / "admission.lock"
        note = re.compile(r"storage admission: .*waited (\d+) s for " + re.escape(str(lock)))
        waited = [
            int(note.fullmatch(line).group(1)) for line in capsys.readouterr().err.splitlines()
        ]
        # The settle gives up at the bound before another note is due, so the notes cover the
        # interval multiples strictly below it.
        due = [int(k * interval) for k in range(1, int(bound // interval) + 1)]
        assert waited == [note_at for note_at in due if note_at < bound]
        assert waited and waited[-1] < bound
    finally:
        if holder is not None:
            fcntl.flock(holder, fcntl.LOCK_UN)
            holder.close()


def test_finished_writer_release_is_bounded_while_a_sibling_thread_waits_for_admission(
    tmp_path, monkeypatch
):
    """The release deadline also covers the process mutex a sibling thread holds while waiting.

    A sibling admission in the same process holds the process-wide mutex for its whole
    unbounded wait on an orphaned campaign lock. A finished writer releasing meanwhile still
    returns within INVENTORY_REUSE_SECONDS (shortened here to real tenths of a second) with its
    reservation lock free, its record forgotten and the settled bytes unchanged, while the
    sibling keeps waiting without a deadline and admits once the orphaned lock is released.
    Before the deadline covered the mutex, this release blocked for as long as the sibling did;
    the timer below then frees the orphaned lock so that case fails instead of hanging.
    """
    import threading
    import time

    poll, bound, safety_net = 0.05, 0.5, 10.0
    monkeypatch.setattr(storage, "ADMISSION_POLL_SECONDS", poll)
    monkeypatch.setattr(storage, "INVENTORY_REUSE_SECONDS", bound)
    budget = resource_budget(tmp_path)
    directory = tmp_path / ".storage-writers"
    sibling = {}

    def admit_sibling():
        try:
            with storage.transfer_storage_writer(budget, stage="select", outputs=[tmp_path]) as row:
                sibling["receipt"] = row
        except Exception as exc:
            sibling["error"] = exc

    holder = None

    def release_holder():
        if holder is not None and not holder.closed:
            fcntl.flock(holder, fcntl.LOCK_UN)
            holder.close()

    worker = threading.Thread(target=admit_sibling)
    unstick = threading.Timer(safety_net, release_holder)
    try:
        with storage.transfer_storage_writer(budget, stage="select", outputs=[tmp_path]) as row:
            reservation = Path(row["reservation"])
            settled = json.loads((directory / "inventory.json").read_text())["settled_bytes"]
            # Admission released the lock before yielding; an orphaned sibling now holds it and
            # a second thread of this process queues behind it, holding the process mutex.
            holder = hold_admission_lock(tmp_path)
            worker.start()
            deadline = time.monotonic() + safety_net
            while storage._STORAGE_ADMISSION_MUTEX.acquire(blocking=False):
                storage._STORAGE_ADMISSION_MUTEX.release()
                assert time.monotonic() < deadline, "the sibling never took the process mutex"
                time.sleep(0.01)
            unstick.start()
            started = time.monotonic()
        elapsed = time.monotonic() - started
        assert bound <= elapsed < safety_net
        assert worker.is_alive() and not holder.closed, "the sibling's admission wait ended early"
        assert json.loads((directory / "inventory.json").read_text())["settled_bytes"] == settled
        assert str(reservation) not in storage._OWN_STORAGE_RESERVATIONS
        with reservation.open("r+") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        unstick.cancel()
        release_holder()
        worker.join(timeout=safety_net)
        assert not worker.is_alive() and "error" not in sibling, sibling.get("error")
        assert Path(sibling["receipt"]["reservation"]).is_file()
        # The sibling's inventory found the released reservation's lock free and skipped it.
        assert sibling["receipt"]["active_writers"] == 0
    finally:
        unstick.cancel()
        release_holder()
        if worker.is_alive():
            worker.join(timeout=safety_net)


def test_refreshed_budget_preserves_live_bounds_and_rejects_changed_scope(tmp_path):
    budget = resource_budget(tmp_path)
    with storage.transfer_storage_writer(budget, stage="select", outputs=[tmp_path]):
        changed = copy.deepcopy(budget)
        changed["observed_allocated_bytes"] = storage.allocated_campaign_bytes([tmp_path])[
            "allocated_bytes"
        ]
        observed = datetime.now(timezone.utc)
        changed["observed_at"] = observed.isoformat()
        changed["valid_until"] = (observed + timedelta(hours=2)).isoformat()
        changed["evidence"]["quota"] = "new synthetic observation"
        # Keep the refreshed bound above the NFS metadata floor.
        changed["writer_bounds"]["select"] -= 1024
        row = storage.inspect_transfer_storage(changed, stage="select", outputs=[tmp_path])
        assert row["inflight_reserved_bytes"] == budget["writer_bounds"]["select"]
        extra = tmp_path / "extra"
        extra.mkdir()
        changed["accounting_roots"].append(str(extra))
        with pytest.raises(ValueError, match="incompatible accounting scope"):
            storage.inspect_transfer_storage(changed, stage="select", outputs=[tmp_path])


def test_budget_deducts_already_counted_growth_once_and_grants_no_deletion_credit():
    quantities = dict(
        existing_allocated_bytes=1000,
        remaining_components=dict.fromkeys(storage.TRANSFER_STORAGE_COMPONENTS, 100),
        bytes_per_writer=100,
        quota_headroom_bytes=1000,
        filesystem_headroom_bytes=2000,
        requested_concurrency=3,
    )
    before = storage.transfer_storage_plan(**quantities)
    after = storage.transfer_storage_plan(
        **{
            **quantities,
            "existing_allocated_bytes": 1300,
            "quota_headroom_bytes": 700,
            "filesystem_headroom_bytes": 1700,
        },
        accounted_growth_bytes=300,
    )
    # Reservations fill planned components: existing + max(remaining, writers).
    assert before["conservative_peak_bytes"] == after["conservative_peak_bytes"] == 1700
    assert after["unallocated_component_bytes"] == 400
    assert after["max_concurrent"] == before["max_concurrent"] == 3
    with pytest.raises(ValueError, match="integer byte bounds"):
        storage.transfer_storage_plan(**quantities, accounted_growth_bytes=-1)


def test_admission_scans_the_inventory_only_once(tmp_path, monkeypatch):
    scans = []
    original = storage.allocated_campaign_bytes

    def measured(*args, **kwargs):
        scans.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(storage, "allocated_campaign_bytes", measured)
    with storage.transfer_storage_writer(
        resource_budget(tmp_path), stage="select", outputs=[tmp_path]
    ):
        pass
    assert len(scans) == 1


def test_mixed_filesystem_budget_cannot_use_one_quota_observation(tmp_path, monkeypatch):
    budget = resource_budget(tmp_path)
    other = tmp_path / "node-local"
    other.mkdir()
    budget["accounting_roots"].append(str(other))
    original = Path.stat

    def different_device(path, *args, **kwargs):
        info = original(path, *args, **kwargs)
        if path == other:
            values = list(info)
            values[2] += 1
            return os.stat_result(values)
        return info

    monkeypatch.setattr(Path, "stat", different_device)
    with pytest.raises(ValueError, match="share the campaign filesystem"):
        storage.inspect_transfer_storage(budget, stage="select", outputs=[tmp_path])


def test_missing_budget_blocks_grouped_and_ordinary_producers(
    siblings, native_stub, monkeypatch  # noqa: F811
):
    for run in siblings:
        run.pop("storage_budget")
    with pytest.raises(ValueError, match="requires --storage-budget"):
        runner.execute_source_group(siblings, selector())
    with pytest.raises(ValueError, match="requires --storage-budget"):
        runner.execute_stage(siblings[0], "reconstruct", selection="LRG--sigma_0.003--mock-0")
    assert not native_stub
    assert all(not (run["root"] / "reconstruct").exists() for run in siblings)


def test_group_publication_records_one_reservation_for_all_siblings(
    siblings, native_stub  # noqa: F811
):
    result = runner.execute_source_group(siblings, selector())
    assert result["errors"] == []
    group = next(item for item in runner.source_groups(siblings) if item["unit_id"] == selector())
    receipts = []
    for target in group["targets"]:
        directory = target["run"]["root"] / "reconstruct/units" / target["unit"]["unit_id"]
        record = yaml.safe_load((directory / "result.yaml").read_text())
        receipts.append(record["runtime"]["storage"])
    assert len(receipts) == 3 and all(value == receipts[0] for value in receipts)
    assert Path(receipts[0]["reservation"]).is_file()
    assert (
        receipts[0]["bytes_per_writer"]
        == siblings[0]["storage_budget"]["writer_bounds"]["reconstruct"]
    )


def test_slurm_plan_clamps_array_and_local_execution_rechecks_budget(
    siblings, native_stub, monkeypatch  # noqa: F811
):
    budget = copy.deepcopy(siblings[0]["storage_budget"])
    budget["max_concurrent"] = 1
    path = siblings[0]["root"].parent / "capacity.yaml"
    path.write_text(yaml.safe_dump(budget))
    plan = runner.slurm_plan(
        "reconstruct",
        runner.source_groups(siblings),
        input_path="select/selection.yaml",
        cpus=2,
        memory="4G",
        walltime="00:05:00",
        account="fixture",
        source_group=True,
        peer_inputs=["peer/select/selection.yaml"],
        max_concurrent=16,
        storage_budget=path,
        storage_outputs=[run["root"] for run in siblings],
    )
    assert "%1\n" in plan and "--storage-budget" in plan and str(path) in plan
    assert "Storage peak bound:" in plan and "--no-requeue" in plan
    # The same path can expire between planning and execution; no producer may then run.
    budget["valid_until"] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    path.write_text(yaml.safe_dump(budget))
    monkeypatch.setattr(runner, "_run_from_handoff", lambda *a, **k: siblings[0])
    result = CliRunner().invoke(
        runner.main,
        [
            "reconstruct",
            "--input",
            __file__,
            "--unit",
            "LRG--sigma_0.003--mock-0",
            "--storage-budget",
            str(path),
        ],
    )
    assert result.exit_code != 0 and "expired" in str(result.exception)
    assert not native_stub and not (siblings[0]["root"] / "reconstruct").exists()


@pytest.mark.parametrize("scheduler", ["local", "slurm"])
def test_catalog_rejects_missing_budget_before_creating_snapshot(tmp_path, scheduler):
    config = transfer_run_configs(load_analysis_config())["self--R25"]
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(config))
    output = tmp_path / "campaign"
    args = [
        "catalog",
        "--run-name",
        "bounded",
        "--output-root",
        str(output),
        "--catalog-root",
        str(tmp_path),
        "--config",
        str(path),
        "--workload",
        "transfer",
        "--scratch",
        "--unit",
        "LRG",
        "--scheduler",
        scheduler,
    ]
    if scheduler == "slurm":
        args += ["--account", "fixture", "--memory", "4G", "--walltime", "00:05:00"]
    result = CliRunner().invoke(runner.main, args)
    assert result.exit_code != 0 and "requires --storage-budget" in str(result.exception)
    assert not output.exists()


def test_catalog_snapshot_and_multiple_units_share_one_bounded_reservation(tmp_path, monkeypatch):
    config = transfer_run_configs(load_analysis_config())["self--R25"]
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(yaml.safe_dump(config))
    budget = resource_budget(tmp_path)
    budget_path = tmp_path / "capacity.yaml"
    budget_path.write_text(yaml.safe_dump(budget))
    monkeypatch.setattr(runner, "_code_identity", lambda **kwargs: dict(CODE))
    observed = []

    def science(stage, run, unit, attempt, cpus):
        assert (run["root"] / "run.yaml").is_file()
        state = storage.inspect_transfer_storage(budget, stage=stage, outputs=[run["root"]])
        assert state["active_writers"] == 1
        observed.append(state["inflight_reserved_bytes"])
        (attempt / "fixture.json").write_text(json.dumps({"tracer": unit["tracer"]}))
        return {"products": [{"path": "fixture.json", "format": "json", "keys": ["tracer"]}]}

    monkeypatch.setattr(runner, "_science", science)
    result = CliRunner().invoke(
        runner.main,
        [
            "catalog",
            "--run-name",
            "bounded",
            "--output-root",
            str(tmp_path),
            "--catalog-root",
            str(tmp_path),
            "--config",
            str(config_path),
            "--workload",
            "transfer",
            "--scratch",
            "--storage-budget",
            str(budget_path),
        ],
    )
    assert result.exit_code == 0, (result.output, result.exception)
    units = runner.stage_units("catalog", config, "transfer")
    assert len(observed) == len(units)
    assert observed == [len(units) * budget["writer_bounds"]["catalog"]] * len(units)
    records = list((tmp_path / ".storage-writers").glob("*/reservation.json"))
    assert len(records) == 1
    root = tmp_path / "scratch/bounded"
    handoff = yaml.safe_load((root / "catalog/catalog.yaml").read_text())
    assert not handoff["errors"] and len(handoff["units"]) == len(units)
    for unit in units:
        record = yaml.safe_load(
            (root / "catalog/units" / unit["unit_id"] / "result.yaml").read_text()
        )
        assert record["runtime"]["storage"]["reservation"] == str(records[0])


def test_covariance_rejects_expired_budget_before_worker_and_forwards_valid_budget(
    siblings, monkeypatch  # noqa: F811
):
    run = siblings[0]
    path = run["root"].parent / "capacity.yaml"
    valid_budget = copy.deepcopy(run["storage_budget"])
    budget = copy.deepcopy(valid_budget)
    budget["valid_until"] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    path.write_text(yaml.safe_dump(budget))
    monkeypatch.setattr(runner, "_run_from_handoff", lambda *args, **kwargs: run)
    monkeypatch.delenv("BAO_COVARIANCE_PROCESS", raising=False)
    calls = []

    def worker(args, *, env):
        calls.append((args, env))
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(runner.subprocess, "run", worker)
    args = ["covariance", "--input", __file__, "--unit", "0", "--storage-budget", str(path)]
    result = CliRunner().invoke(runner.main, args)
    assert result.exit_code != 0 and "expired" in str(result.exception)
    assert not calls and not (run["root"] / "covariance").exists()
    path.write_text(yaml.safe_dump(valid_budget))
    result = CliRunner().invoke(runner.main, args)
    assert result.exit_code == 0, (result.output, result.exception)
    assert len(calls) == 1
    command, environment = calls[0]
    assert command[command.index("--storage-budget") + 1] == str(path)
    assert environment["BAO_COVARIANCE_PROCESS"] == "1"


def test_review_requires_admission_before_rendering(siblings, monkeypatch):  # noqa: F811
    from pzbao.diagnostics import review

    run = siblings[0]
    budget = copy.deepcopy(run["storage_budget"])
    monkeypatch.setattr(runner, "load_run", lambda *args: run)
    calls = []

    def render(root, **kwargs):
        state = storage.inspect_transfer_storage(budget, stage="review", outputs=[root])
        assert state["active_writers"] == 1
        calls.append(root)

    monkeypatch.setattr(review, "render_review", render)
    args = ["review", "--run", str(run["root"])]
    result = CliRunner().invoke(runner.main, args)
    assert result.exit_code != 0 and "requires --storage-budget" in str(result.exception)
    assert not calls
    path = run["root"].parent / "capacity.yaml"
    path.write_text(yaml.safe_dump(budget))
    result = CliRunner().invoke(runner.main, args + ["--storage-budget", str(path)])
    assert result.exit_code == 0, (result.output, result.exception)
    assert calls == [run["root"]]


def test_admission_reuses_a_recent_inventory_and_keeps_released_writes_counted(
    tmp_path, monkeypatch
):
    budget = resource_budget(tmp_path)
    bounds = budget["writer_bounds"]
    calls = []
    original = storage.allocated_campaign_bytes

    def counting(roots, *, shared_source_roots=()):
        calls.append(1)
        return original(roots, shared_source_roots=shared_source_roots)

    monkeypatch.setattr(storage, "allocated_campaign_bytes", counting)
    directory = tmp_path / ".storage-writers"
    with storage.transfer_storage_writer(budget, stage="select", outputs=[tmp_path]) as first:
        assert len(calls) == 1
        record = json.loads((directory / "inventory.json").read_text())
        assert record["settled_bytes"] == 0
        assert record["allocated_bytes"] == first["existing_allocated_bytes"]
    # A released writer settles what changed under its outputs, capped at its bound: one that
    # wrote nothing settles only its own reservation metadata. The next admission
    # inside the window reuses the inventory instead of walking the roots again.
    record = json.loads((directory / "inventory.json").read_text())
    idle = record["settled_bytes"]
    assert 0 <= idle < 1024**2 < bounds["select"]
    with storage.transfer_storage_writer(budget, stage="reconstruct", outputs=[tmp_path]) as second:
        assert len(calls) == 1
        assert second["existing_allocated_bytes"] == record["allocated_bytes"] + idle
        assert second["inflight_reserved_bytes"] == 0
        (tmp_path / "unit.bin").write_bytes(b"x" * 3 * 1024**2)
    record = json.loads((directory / "inventory.json").read_text())
    assert 3 * 1024**2 <= record["settled_bytes"] - idle <= bounds["reconstruct"]
    # Planning reads the fresh record, counts the settled bounds and writes nothing.
    plan = storage.inspect_transfer_storage(budget, stage="select", outputs=[tmp_path])
    assert len(calls) == 1
    assert plan["existing_allocated_bytes"] == record["allocated_bytes"] + record["settled_bytes"]
    aged = {**record, "observed_at": record["observed_at"] - storage.INVENTORY_REUSE_SECONDS - 1}
    (directory / "inventory.json").write_text(json.dumps(aged))
    storage.inspect_transfer_storage(budget, stage="select", outputs=[tmp_path])
    assert len(calls) == 2
    assert json.loads((directory / "inventory.json").read_text()) == aged
    # A writer replaces an expired record and starts the settled bytes afresh.
    with storage.transfer_storage_writer(budget, stage="select", outputs=[tmp_path]):
        assert len(calls) == 3
    fresh = json.loads((directory / "inventory.json").read_text())
    assert fresh["observed_at"] > aged["observed_at"]
    assert fresh["settled_bytes"] < 1024**2
    # A corrupt record or another accounting scope never serves a reused inventory.
    (directory / "inventory.json").write_text("{not json")
    storage.inspect_transfer_storage(budget, stage="select", outputs=[tmp_path])
    assert len(calls) == 4
    with storage.transfer_storage_writer(budget, stage="select", outputs=[tmp_path]):
        assert len(calls) == 5
    other_root = tmp_path / "other"
    other_root.mkdir()
    other = resource_budget(other_root)
    assert storage._campaign_bytes(other, directory, record=False) >= 0
    assert len(calls) == 6
    saved = json.loads((directory / "inventory.json").read_text())
    assert saved["scope"]["accounting_roots"] == [str(tmp_path.resolve())]
