"""Small cross-process locks for launcher single-instance and maintenance."""

from __future__ import annotations

from pathlib import Path
import os
from typing import BinaryIO

from .errors import LauncherError


class InterProcessLock:
    """Hold a one-byte OS lock until the context exits."""

    def __init__(self, path: Path, *, purpose: str) -> None:
        self.path = path.resolve()
        self.purpose = purpose
        self._handle: BinaryIO | None = None

    def acquire(self) -> "InterProcessLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle: BinaryIO | None = None
        try:
            handle = self.path.open("a+b")
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
                os.fsync(handle.fileno())
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, IOError) as error:
            try:
                if handle is not None:
                    handle.close()
            except OSError:
                pass
            raise LauncherError(
                f"无法取得 {self.purpose} 锁；另一 SPImaging 实例或维护任务仍在运行"
            ) from error
        assert handle is not None
        self._handle = handle
        return self

    def release(self) -> None:
        handle = self._handle
        if handle is None:
            return
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
            self._handle = None

    def __enter__(self) -> "InterProcessLock":
        return self.acquire()

    def __exit__(self, *_: object) -> None:
        self.release()
