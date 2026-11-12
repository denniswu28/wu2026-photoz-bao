"""Locations of relocated data; scientific identities remain in the original records."""

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import lru_cache
import fcntl
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile
import threading
import time
import uuid


def storage_root():
    """Return the storage root (``$BAO_STORAGE_ROOT``, else ``./outputs``) without creating it."""
    return Path(os.environ.get("BAO_STORAGE_ROOT", "outputs")).expanduser()


# Since 2026-10-17 a campaign may exceed its former 3 TiB cap while at least 1 TiB of the
# shared filesystem stays free. Budgets record the floor they were prepared
# with; budgets without the field predate it and carry the cap in their quota headroom.
XDISK_FREE_FLOOR_BYTES = 1024**4
TRANSFER_STORAGE_COMPONENTS = (
    "selected_catalogues",
    "shifted_shards",
    "clustering",
    "raw_covariance",
    "fits_sampling_reports",
    "retained_pilots_and_failures",
    "peak_staging",
)


def allocated_campaign_bytes(roots, *, shared_source_roots=()):
    """Count physical campaign files once, including failures and staging, following links.

    Shared source archives must be explicitly identified. Directory/file inode sets prevent
    symlink loops, overlapping roots and hardlinks from inflating the physical accounting.
    """
    excluded = tuple(Path(root).resolve(strict=True) for root in shared_source_roots)
    roots = tuple(Path(root).resolve(strict=True) for root in roots)
    # Atomic publication can rename a listed temporary file before it is visited.
    # Restart the whole inventory instead of omitting its bytes. Persistent churn,
    # missing roots, dangling links and all other I/O failures still fail closed.
    for attempt in range(3):
        seen, pending, allocated, logical = set(), list(roots), 0, 0
        try:
            while pending:
                path = pending.pop()
                path = path.resolve(strict=True)
                if any(path == root or path.is_relative_to(root) for root in excluded):
                    continue
                info = path.stat()
                identity = (info.st_dev, info.st_ino)
                if identity in seen:
                    continue
                seen.add(identity)
                allocated += info.st_blocks * 512
                if stat.S_ISDIR(info.st_mode):
                    pending.extend(path.iterdir())
                elif stat.S_ISREG(info.st_mode):
                    logical += info.st_size
                else:
                    raise ValueError("campaign storage contains an unsupported file type")
        except FileNotFoundError as exc:
            if path in roots or path.is_symlink():
                raise
            if attempt == 2:
                raise FileNotFoundError(
                    "campaign changed during all three inventory passes"
                ) from exc
        else:
            return {
                "allocated_bytes": allocated,
                "logical_bytes": logical,
                "unique_inodes": len(seen),
            }


def transfer_storage_plan(
    *,
    existing_allocated_bytes,
    remaining_components,
    bytes_per_writer,
    quota_headroom_bytes,
    filesystem_headroom_bytes,
    requested_concurrency,
    inflight_reserved_bytes=0,
    accounted_growth_bytes=0,
):
    """Fail closed on incomplete volume accounting; bound both campaign and in-flight peaks.

    Remaining components describe retained growth from the budget observation, including
    extra staging. Deduct growth already physically counted exactly once. A writer's bounded
    reservation is part of the planned components it fills, so admission requires
    existing + max(remaining - growth, in-flight + new reservations) within the headroom.
    Every caller must supply measured or conservative upper bounds, never average writes.
    """
    if set(remaining_components) != set(TRANSFER_STORAGE_COMPONENTS):
        raise ValueError("storage plan lacks complete campaign components")
    quantities = [
        existing_allocated_bytes,
        quota_headroom_bytes,
        filesystem_headroom_bytes,
        bytes_per_writer,
        requested_concurrency,
        inflight_reserved_bytes,
        accounted_growth_bytes,
        *remaining_components.values(),
    ]
    if any(type(value) is not int or value < 0 for value in quantities):
        raise ValueError("storage accounting requires nonnegative integer byte bounds")
    if bytes_per_writer == 0 or requested_concurrency == 0:
        raise ValueError("storage planning requires a positive writer bound and concurrency")
    additional = max(0, sum(remaining_components.values()) - accounted_growth_bytes)
    headroom = min(quota_headroom_bytes, filesystem_headroom_bytes)
    if additional > headroom:
        raise ValueError("complete campaign peak exceeds the verified storage headroom")
    concurrency = min(
        requested_concurrency,
        (headroom - inflight_reserved_bytes) // bytes_per_writer,
    )
    if concurrency < 1:
        raise ValueError("no bounded writer fits the remaining campaign capacity")
    return {
        "storage_limit_bytes": existing_allocated_bytes + headroom,
        "existing_allocated_bytes": existing_allocated_bytes,
        "remaining_components": dict(remaining_components),
        "unallocated_component_bytes": additional,
        "accounted_growth_bytes": accounted_growth_bytes,
        "bytes_per_writer": bytes_per_writer,
        "max_concurrent": concurrency,
        "inflight_reserved_bytes": inflight_reserved_bytes,
        "conservative_peak_bytes": existing_allocated_bytes
        + max(additional, inflight_reserved_bytes + concurrency * bytes_per_writer),
        "quota_headroom_bytes": quota_headroom_bytes,
        "filesystem_headroom_bytes": filesystem_headroom_bytes,
    }


def validate_transfer_storage_budget(budget, *, stage, outputs, now=None):
    """Validate operational bounds; evidence pointers do not establish their scientific validity.

    Quota headroom and volume/writer bounds must come from reviewed measurements or
    conservative estimates. Their observation has an explicit expiry, and subsequent
    campaign growth consumes the quoted quota headroom without granting deletion credits.
    """
    keys = {
        "schema_version",
        "campaign_root",
        "accounting_roots",
        "shared_source_roots",
        "observed_allocated_bytes",
        "quota_headroom_bytes",
        "observed_at",
        "valid_until",
        "remaining_components",
        "writer_bounds",
        "max_concurrent",
        "evidence",
    }
    if (
        not isinstance(budget, dict)
        or not keys <= set(budget) <= keys | {"free_space_floor_bytes"}
        or type(budget["schema_version"]) is not int
        or budget["schema_version"] != 1
    ):
        raise ValueError("transfer execution requires a complete version-1 storage budget")
    # Take an owned, JSON-compatible copy before storing it in a reservation.
    budget = json.loads(json.dumps(budget, allow_nan=False))
    budget.setdefault("free_space_floor_bytes", 0)
    for key in (
        "observed_allocated_bytes",
        "quota_headroom_bytes",
        "free_space_floor_bytes",
        "max_concurrent",
    ):
        if type(budget[key]) is not int or budget[key] < (1 if key == "max_concurrent" else 0):
            raise ValueError("storage budget requires nonnegative integer byte bounds")
    for key in ("remaining_components", "writer_bounds"):
        if not isinstance(budget[key], dict) or any(
            type(value) is not int or value < (1 if key == "writer_bounds" else 0)
            for value in budget[key].values()
        ):
            raise ValueError("storage budget requires nonnegative integer byte bounds")
    if set(budget["remaining_components"]) != set(TRANSFER_STORAGE_COMPONENTS):
        raise ValueError("storage plan lacks complete campaign components")
    if stage not in budget["writer_bounds"]:
        raise ValueError(f"storage budget has no bounded writer for {stage}")
    evidence = budget["evidence"]
    if (
        not isinstance(evidence, dict)
        or set(evidence) != {"quota", "volume", "writers"}
        or any(not isinstance(value, str) or not value.strip() for value in evidence.values())
    ):
        raise ValueError("storage budget requires quota, volume and writer evidence references")
    observed, expires = (
        datetime.fromisoformat(budget[key]) for key in ("observed_at", "valid_until")
    )
    now = now or datetime.now(timezone.utc)
    if observed.tzinfo is None or expires.tzinfo is None or not observed <= now < expires:
        raise ValueError("storage budget observation is future-dated, expired or lacks a timezone")

    def absolute(value, *, must_exist=True):
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise ValueError("storage budget paths must be absolute")
        return resolve_path(value).resolve(strict=must_exist)

    root = absolute(budget["campaign_root"])
    if not root.is_dir():
        raise ValueError("campaign storage root must be an existing directory")
    for key in ("accounting_roots", "shared_source_roots"):
        if not isinstance(budget[key], list):
            raise ValueError("storage accounting roots must be explicit lists")
        budget[key] = [str(absolute(value)) for value in budget[key]]
    roots, shared = (
        list(map(Path, budget[key])) for key in ("accounting_roots", "shared_source_roots")
    )
    if root not in roots or any(root == item or root.is_relative_to(item) for item in shared):
        raise ValueError("storage budget must account for the complete campaign root")
    if any(item.is_relative_to(root) for item in shared):
        raise ValueError("campaign-owned paths cannot be excluded as shared source archives")
    if any(item.stat().st_dev != root.stat().st_dev for item in roots):
        raise ValueError("storage accounting roots must share the campaign filesystem")
    # One campaign may keep topic folders side by side; each must be an accounting root.
    if not outputs or any(
        not any(Path(path).resolve().is_relative_to(item) for item in roots) for path in outputs
    ):
        raise ValueError("writer output leaves the accounted campaign roots")
    budget["campaign_root"] = str(root)
    # The reservation record and temporary directory must themselves fit its bound.
    overhead = len(json.dumps(budget).encode()) + 4 * os.statvfs(root).f_frsize
    if budget["writer_bounds"][stage] < overhead:
        raise ValueError("writer bound does not include storage reservation metadata")
    return budget


_STORAGE_ADMISSION_MUTEX = threading.RLock()
_OWN_STORAGE_RESERVATIONS = {}


def _storage_scope(budget):
    """Refreshed observations may coexist, but cannot drop any accounted storage root."""
    return (
        budget["schema_version"],
        budget["campaign_root"],
        tuple(sorted(budget["accounting_roots"])),
        tuple(sorted(budget["shared_source_roots"])),
    )


# Admission holds the lock for one inventory, a probe of each live reservation and a
# reservation write, so sibling array tasks that start together queue for it. A held lock is another writer's admission in
# progress, not an error: contention waits without a deadline, writing one progress line to
# stderr at this interval so a long wait stays visible in the task's error file. Any other
# lock error still fails at once.
ADMISSION_POLL_SECONDS = 2.0
ADMISSION_PROGRESS_SECONDS = 300.0
# One physical inventory of the campaign roots takes tens of seconds (28 s for 57k inodes on
# 2026-10-16). Admissions run serially under the lock, so ninety array tasks starting together
# exhausted the former hour-long wait and 82 of them failed closed before measuring anything.
# An inventory is therefore reused for this long; see ``_campaign_bytes`` for why that stays
# conservative. A finished writer also waits at most this long to settle its bound under the
# lock; see the release path of ``transfer_storage_writer``.
INVENTORY_REUSE_SECONDS = 900.0
# Released reservation records are kept, as history, in this subdirectory of the writer
# directory, where the probe of live records does not reach them. With 6,013 records kept in
# place on 2026-10-21, admissions came every 20 s and fit tasks waited 10 min on average.
RELEASED_RESERVATIONS = "released"


@contextmanager
def _storage_admission_lock(path, *, create=False, wait_seconds=None):
    """Serialize local threads and NFS clients; wait on contention, fail closed on errors.

    A held lock is another writer's admission in progress, so an admission (``wait_seconds``
    None) waits for it without a deadline, polling every ADMISSION_POLL_SECONDS and noting the
    elapsed wait and the lock path on stderr every ADMISSION_PROGRESS_SECONDS. Only a finished
    writer's release settle passes a bound (INVENTORY_REUSE_SECONDS) and raises ValueError once
    the lock has been held that long. The deadline covers the whole wait: it starts before the
    process mutex that serializes local threads, and that mutex is polled rather than blocked
    on, because a sibling thread holds it through its own unbounded wait. A lock-manager error
    fails closed: nothing is retried after a failure and no lock is ever removed.
    """
    started = noted = time.monotonic()

    def wait_one_poll(cause):
        nonlocal noted
        now = time.monotonic()
        if wait_seconds is not None and now - started >= wait_seconds:
            raise ValueError(
                f"storage admission lock still held after {wait_seconds:.0f} s"
            ) from cause
        if now - noted >= ADMISSION_PROGRESS_SECONDS:
            print(
                "storage admission: lock held by another writer; waited "
                f"{now - started:.0f} s for {path}",
                file=sys.stderr,
                flush=True,
            )
            noted = now
        time.sleep(ADMISSION_POLL_SECONDS)

    while not _STORAGE_ADMISSION_MUTEX.acquire(blocking=False):
        wait_one_poll(None)
    try:
        with path.open("a+" if create else "r+") as stream:
            while True:
                try:
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError as exc:
                    wait_one_poll(exc)
                except OSError as exc:
                    raise ValueError(
                        "storage admission lock unavailable; no automatic retry"
                    ) from exc
            yield
    finally:
        _STORAGE_ADMISSION_MUTEX.release()


def _active_storage_writers(directory, budget, *, retire=False):
    """Under the admission lock, count live file locks; never trust a saved PID.

    Never reopen a lock owned by this process: POSIX-style locks may not conflict with
    themselves, and closing another descriptor may release them. Remote lock-manager
    failure is not evidence of death; errors fail closed and stale locks remain reserved.
    An admission (``retire``) moves each released record into ``released/`` before it
    returns, so later admissions probe only live writers; planning never moves one. A move
    that fails leaves the remaining records in place for a later admission.
    """
    active, released = [], []
    for path in sorted(directory.glob("*/reservation.json")):
        owned = _OWN_STORAGE_RESERVATIONS.get(str(path))
        if owned is not None and owned[0] == os.getpid():
            record = owned[1]
        else:
            # NFS exclusive flock requires a descriptor opened for writing.
            with path.open("r+") as stream:
                try:
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    record = json.load(stream)
                except OSError as exc:
                    raise ValueError("cannot establish storage reservation lock state") from exc
                else:
                    fcntl.flock(stream, fcntl.LOCK_UN)
                    released.append(path.parent)
                    continue
        previous = record.get("budget")
        if not isinstance(previous, dict) or _storage_scope(previous) != _storage_scope(budget):
            raise ValueError("another storage budget has an incompatible accounting scope")
        bound = record.get("reserved_bytes")
        if type(bound) is not int or bound <= 0:
            raise ValueError("invalid active storage reservation")
        active.append(record)
    if retire and released:
        # Every admission probes every record under the lock, so released records kept in
        # place made admission slower with each writer ever admitted (2026-10-21).
        try:
            (directory / RELEASED_RESERVATIONS).mkdir(exist_ok=True)
            for parent in released:
                os.replace(parent, directory / RELEASED_RESERVATIONS / parent.name)
        except OSError:
            pass
    return active


def _inventory_scope(budget):
    return {
        "accounting_roots": sorted(
            str(Path(root).resolve()) for root in budget["accounting_roots"]
        ),
        "shared_source_roots": sorted(
            str(Path(root).resolve()) for root in budget["shared_source_roots"]
        ),
    }


def _write_json_atomically(path, record):
    partial = path.with_suffix(".partial")
    with partial.open("w") as stream:
        stream.write(json.dumps(record))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)


def _campaign_bytes(budget, directory, *, record):
    """Physical campaign bytes, reusing an inventory younger than INVENTORY_REUSE_SECONDS.

    Called under the admission lock. A reused inventory stays conservative: every live writer
    adds its full reservation to the measured bytes, and a writer that releases inside the
    window adds what changed under its outputs (at most its reservation) to the record as
    settled bytes until the next fresh inventory replaces it. Only a writer that dies without releasing leaves its partial write uncounted,
    bounded by one reservation, until the record expires. Planning (``record=False``) may read
    a fresh record but never writes one.
    """
    path = None if directory is None else Path(directory) / "inventory.json"
    scope = _inventory_scope(budget)
    now = time.time()
    if path is not None and path.is_file():
        try:
            saved = json.loads(path.read_text())
        except ValueError:
            saved = {}
        observed = saved.get("observed_at") if isinstance(saved, dict) else None
        if (
            isinstance(saved, dict)
            and saved.get("scope") == scope
            and type(observed) in (int, float)
            and 0.0 <= now - observed <= INVENTORY_REUSE_SECONDS
            and type(saved.get("allocated_bytes")) is int
            and type(saved.get("settled_bytes")) is int
            and saved["settled_bytes"] >= 0
        ):
            return saved["allocated_bytes"] + saved["settled_bytes"]
    measured = allocated_campaign_bytes(
        budget["accounting_roots"], shared_source_roots=budget["shared_source_roots"]
    )
    if path is not None and record:
        _write_json_atomically(
            path,
            {
                "scope": scope,
                "observed_at": now,
                "allocated_bytes": measured["allocated_bytes"],
                "settled_bytes": 0,
            },
        )
    return measured["allocated_bytes"]


def _bytes_changed_since(roots, since_ns):
    """Allocated bytes of inodes under ``roots`` whose status changed at or after ``since_ns``.

    A released writer settles this instead of its full bound. It covers everything the writer
    created, rewrote, renamed or linked under its declared outputs, and also whatever sibling
    writers did there meanwhile, which keeps the settle conservative. Timestamps come from the
    file server, like the reference taken at admission. Links are followed once per inode;
    entries removed during the walk count nothing.
    """
    seen, pending, total = set(), [Path(root) for root in roots], 0
    while pending:
        path = pending.pop()
        try:
            info = path.stat()
            identity = (info.st_dev, info.st_ino)
            if identity in seen:
                continue
            seen.add(identity)
            if stat.S_ISDIR(info.st_mode):
                pending.extend(path.iterdir())
        except FileNotFoundError:
            continue
        if info.st_ctime_ns >= since_ns:
            total += info.st_blocks * 512
    return total


def _settle_reservation(directory, reserved_bytes):
    """Under the admission lock, keep a released writer's writes counted until a fresh inventory."""
    path = Path(directory) / "inventory.json"
    if not path.is_file():
        return
    try:
        saved = json.loads(path.read_text())
    except ValueError:
        return
    if isinstance(saved, dict) and type(saved.get("settled_bytes")) is int:
        saved["settled_bytes"] += int(reserved_bytes)
        _write_json_atomically(path, saved)


def _transfer_storage_state(
    budget, stage, active, *, requested_concurrency, unit_count=1, directory=None, record=False
):
    if type(unit_count) is not int or unit_count < 1:
        raise ValueError("storage admission requires at least one execution unit")
    limit = min([budget["max_concurrent"]] + [row["budget"]["max_concurrent"] for row in active])
    available_slots = limit - len(active)
    if available_slots < 1:
        raise ValueError("campaign storage writer concurrency is exhausted")
    current = _campaign_bytes(budget, directory, record=record)
    growth = max(0, current - budget["observed_allocated_bytes"])
    headroom = budget["quota_headroom_bytes"] - growth
    if headroom < 0:
        raise ValueError("campaign growth exhausted the verified quota headroom")
    plan = transfer_storage_plan(
        existing_allocated_bytes=current,
        remaining_components=budget["remaining_components"],
        bytes_per_writer=budget["writer_bounds"][stage] * unit_count,
        quota_headroom_bytes=headroom,
        filesystem_headroom_bytes=max(
            0,
            min(shutil.disk_usage(path).free for path in budget["accounting_roots"])
            - budget["free_space_floor_bytes"],
        ),
        requested_concurrency=min(requested_concurrency, available_slots),
        inflight_reserved_bytes=sum(row["reserved_bytes"] for row in active),
        accounted_growth_bytes=growth,
    )
    return {**plan, "campaign_root": budget["campaign_root"], "active_writers": len(active)}


def inspect_transfer_storage(budget, *, stage, outputs, requested_concurrency=1, unit_count=1):
    """Non-mutating planning; existing lock files need write-open access on NFS."""
    budget = validate_transfer_storage_budget(budget, stage=stage, outputs=outputs)
    directory = Path(budget["campaign_root"]) / ".storage-writers"
    lock = directory / "admission.lock"
    if lock.exists():
        with _storage_admission_lock(lock):
            return _transfer_storage_state(
                budget,
                stage,
                _active_storage_writers(directory, budget),
                requested_concurrency=requested_concurrency,
                unit_count=unit_count,
                directory=directory,
            )
    return _transfer_storage_state(
        budget,
        stage,
        [],
        requested_concurrency=requested_concurrency,
        unit_count=unit_count,
    )


_ACTIVE_TRANSFER_WRITER = ContextVar("bao_transfer_storage_writer", default=None)


@contextmanager
def transfer_storage_writer(budget, *, stage, outputs, unit_count=1):
    """Reserve a bounded invocation across sibling runs and keep temporary files accounted.

    Live reservations are conservatively added to measured bytes even when some writes
    already exist. On release a writer settles only the bytes changed under its outputs since
    admission, capped at its reservation, so a resumed unit that writes nothing settles almost
    nothing. Admission moves released records aside, never deletes one. No
    background monitor, retry, stale-file deletion or quota override is introduced. The
    resource bounds remain a prerequisite to be justified by measurements.
    """
    budget = validate_transfer_storage_budget(budget, stage=stage, outputs=outputs)
    enclosing = _ACTIVE_TRANSFER_WRITER.get()
    if enclosing is not None and enclosing["pid"] == os.getpid():
        if (
            enclosing["budget"] != budget
            or enclosing["stage"] != stage
            or unit_count > enclosing["unit_count"]
            or any(
                not any(
                    Path(path).resolve().is_relative_to(Path(parent).resolve())
                    for parent in enclosing["outputs"]
                )
                for path in outputs
            )
        ):
            raise ValueError("nested writer differs from its storage reservation")
        yield enclosing["receipt"]
        return
    # Cheap baseline rejection precedes metadata creation. The physical inventory runs
    # once under the admission lock, not twice. It remains conservative for live writes
    # because their full reservations are added to the scanned bytes.
    if type(unit_count) is not int or unit_count < 1:
        raise ValueError("storage admission requires at least one execution unit")
    transfer_storage_plan(
        existing_allocated_bytes=budget["observed_allocated_bytes"],
        remaining_components=budget["remaining_components"],
        bytes_per_writer=budget["writer_bounds"][stage] * unit_count,
        quota_headroom_bytes=budget["quota_headroom_bytes"],
        # Live filesystem space is checked with the physical inventory below. Applying
        # today's free space to yesterday's remaining volume here would count growth twice.
        filesystem_headroom_bytes=budget["quota_headroom_bytes"],
        requested_concurrency=1,
    )
    directory = Path(budget["campaign_root"]) / ".storage-writers"
    directory.mkdir(exist_ok=True)
    with _storage_admission_lock(directory / "admission.lock", create=True):
        budget = validate_transfer_storage_budget(budget, stage=stage, outputs=outputs)
        plan = _transfer_storage_state(
            budget,
            stage,
            _active_storage_writers(directory, budget, retire=True),
            requested_concurrency=1,
            unit_count=unit_count,
            directory=directory,
            record=True,
        )
        own = directory / uuid.uuid4().hex
        own.mkdir()
        admitted_ns = own.stat().st_ctime_ns
        record = {
            "budget": budget,
            "stage": stage,
            "outputs": list(map(str, outputs)),
            "reserved_bytes": plan["bytes_per_writer"],
            "admission": plan,
        }
        # Close the writer before publishing its name: remote NFS clients must not
        # cache a zero-length reservation while we keep its lifetime lock open.
        partial = own / "reservation.partial"
        with partial.open("x") as stream:
            json.dump(record, stream)
            stream.flush()
            os.fsync(stream.fileno())
        reservation = partial.open("r+")
        try:
            fcntl.flock(reservation, fcntl.LOCK_EX | fcntl.LOCK_NB)
            os.replace(partial, own / "reservation.json")
            _OWN_STORAGE_RESERVATIONS[str(own / "reservation.json")] = (os.getpid(), record)
        except BaseException:
            reservation.close()
            raise
    receipt = {**plan, "reservation": str(own / "reservation.json")}
    token = _ACTIVE_TRANSFER_WRITER.set(
        {
            "budget": budget,
            "stage": stage,
            "unit_count": unit_count,
            "receipt": receipt,
            "pid": os.getpid(),
            "outputs": list(map(str, outputs)),
        }
    )
    old_tempdir = tempfile.tempdir
    old_environment = {key: os.environ.get(key) for key in ("TMPDIR", "JOBLIB_TEMP_FOLDER")}
    try:
        temporary = own / "temporary"
        temporary.mkdir()
        tempfile.tempdir = str(temporary)
        for key in old_environment:
            os.environ[key] = str(temporary)
        yield receipt
    finally:
        tempfile.tempdir = old_tempdir
        for key, value in old_environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        _ACTIVE_TRANSFER_WRITER.reset(token)
        # Settle before releasing: between the two the bound is counted twice, which is
        # conservative. The stage's products are already published, so the settle must not
        # inherit admission's unbounded wait: it gives up after INVENTORY_REUSE_SECONDS, when
        # every record observed before this release has expired and any newer record already
        # measured the published files, and it never raises over the finished stage. The bound
        # includes the process mutex, which a sibling thread may hold through its own wait.
        # Measured outside the lock; an unreadable output settles the full bound.
        try:
            written = min(
                record["reserved_bytes"],
                _bytes_changed_since([*record["outputs"], own], admitted_ns),
            )
        except OSError:
            written = record["reserved_bytes"]
        try:
            with _storage_admission_lock(
                directory / "admission.lock", wait_seconds=INVENTORY_REUSE_SECONDS
            ):
                _settle_reservation(directory, written)
        except (OSError, ValueError):
            pass
        # The lifetime lock is freed without that mutex. Closing before forgetting keeps a
        # sibling thread's inventory conservative: while the record is still owned it counts
        # this writer without reopening the lock; once forgotten, the lock is already free and
        # the sibling skips the record.
        reservation.close()
        _OWN_STORAGE_RESERVATIONS.pop(str(own / "reservation.json"), None)


@lru_cache(maxsize=8)
def _read_locations(root, modified, changed, size):
    path = root / "archive/relocation/paths.json"
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != 1 or not isinstance(payload.get("roots"), list):
        raise ValueError("Unsupported storage relocation map")
    rows = []
    for row in payload["roots"]:
        source, relative = Path(row["source"]), Path(row["destination"])
        if (
            not source.is_absolute()
            or source == Path("/")
            or ".." in source.parts
            or relative.is_absolute()
            or not relative.parts
            or ".." in relative.parts
        ):
            raise ValueError("Invalid storage relocation root")
        destination = root / relative
        if not destination.resolve().is_relative_to(root):
            raise ValueError("Storage relocation leaves its storage root")
        for previous_source, previous_destination in rows:
            if source.is_relative_to(previous_source) or previous_source.is_relative_to(source):
                raise ValueError("Ambiguous storage relocation sources")
            if destination.is_relative_to(
                previous_destination
            ) or previous_destination.is_relative_to(destination):
                raise ValueError("Ambiguous storage relocation destinations")
        if source.is_relative_to(root) or root.is_relative_to(source):
            raise ValueError("Storage relocation cannot remap its own storage root")
        rows.append((source, destination))
    return rows


def resolve_path(path):
    """Translate only explicitly mapped roots; do not search for similar filenames.

    The migration publishes its map after moving complete bundles. An unmapped
    path is returned unchanged. Missing files inside a mapped bundle remain
    missing, allowing ordinary readers to report partial runs normally.
    """
    path = Path(path).expanduser()
    root = storage_root().resolve()
    mapping = root / "archive/relocation/paths.json"
    if not mapping.is_file():
        return path
    info = mapping.stat()
    rows = _read_locations(root, info.st_mtime_ns, info.st_ctime_ns, info.st_size)
    original = path.absolute()
    for source, destination in rows:
        if not original.is_relative_to(source):
            continue
        if not destination.is_dir():
            raise FileNotFoundError(f"Relocated storage bundle is missing: {destination}")
        candidate = destination / original.relative_to(source)
        if not candidate.resolve().is_relative_to(root):
            raise ValueError("Relocated file link leaves its storage root")
        if path.exists() and path.resolve() != candidate.resolve():
            raise ValueError(f"Both original and relocated storage paths exist: {path}")
        return candidate
    return path


def same_location(left, right):
    """Compare two recorded locations using only the installed relocation map."""
    return resolve_path(left).resolve() == resolve_path(right).resolve()


def same_run_identity(actual, expected):
    """Allow a relocated root while checking every other identity field exactly."""
    from .config import same_identity

    if not isinstance(actual, dict) or not isinstance(expected, dict):
        return False
    if not isinstance(actual.get("root"), str) or not isinstance(expected.get("root"), str):
        return False
    return same_identity(
        {key: value for key, value in actual.items() if key != "root"},
        {key: value for key, value in expected.items() if key != "root"},
    ) and same_location(actual["root"], expected["root"])
