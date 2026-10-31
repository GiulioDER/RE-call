"""Serialize cooperating edits to the client's shared configuration file."""

from __future__ import annotations

import errno
import os
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def configuration_lock(path: Path) -> Iterator[None]:
    """Hold a stable sibling OS lock from the first read through atomic replacement.

    Keep the lock file in place: unlinking it lets another process lock a different
    inode while a waiter still holds the original. The OS releases locks on exit.
    """
    target = path.resolve()
    lock_path = target.with_name(target.name + ".recall-lock")
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lock_path, flags, 0o600)
    acquired = False
    try:
        deadline = time.monotonic() + 30.0
        while not acquired:
            try:
                if sys.platform == "win32":
                    import msvcrt

                    os.lseek(descriptor, 0, os.SEEK_SET)
                    msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except OSError as exc:
                if exc.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                    raise
                if time.monotonic() >= deadline:
                    raise OSError(f"configuration is busy: {target}") from None
                time.sleep(0.01)
        yield
    finally:
        try:
            if acquired:
                if sys.platform == "win32":
                    import msvcrt

                    os.lseek(descriptor, 0, os.SEEK_SET)
                    msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)
