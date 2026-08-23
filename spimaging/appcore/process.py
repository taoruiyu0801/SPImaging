"""Cancellation primitives and Windows process-tree ownership."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import os
from pathlib import Path
import threading


class CancellationToken:
    """A cooperative token backed by memory and an optional request file."""

    def __init__(self, request_file: str | Path | None = None) -> None:
        self.request_file = Path(request_file) if request_file is not None else None
        self._event = threading.Event()

    def request(self) -> None:
        self._event.set()
        if self.request_file is not None:
            self.request_file.parent.mkdir(parents=True, exist_ok=True)
            self.request_file.write_text("cancel\n", encoding="utf-8")

    @property
    def requested(self) -> bool:
        return self._event.is_set() or bool(
            self.request_file is not None and self.request_file.is_file()
        )

    def clear_file(self) -> None:
        if self.request_file is not None and self.request_file.is_file():
            self.request_file.unlink()


if os.name == "nt":
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.argtypes = (ctypes.c_void_p, wintypes.LPCWSTR)
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.SetInformationJobObject.argtypes = (
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    )
    kernel32.SetInformationJobObject.restype = wintypes.BOOL
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel32.TerminateJobObject.argtypes = (wintypes.HANDLE, wintypes.UINT)
    kernel32.TerminateJobObject.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL

    class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]


class WindowsJob:
    """Kill-on-close job object; a no-op on non-Windows systems."""

    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
    JobObjectExtendedLimitInformation = 9
    PROCESS_TERMINATE = 0x0001
    PROCESS_SET_QUOTA = 0x0100

    def __init__(self) -> None:
        self.handle = None
        if os.name != "nt":
            return
        handle = kernel32.CreateJobObjectW(None, None)
        if not handle:
            raise OSError(ctypes.get_last_error(), "CreateJobObjectW failed")
        information = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        information.BasicLimitInformation.LimitFlags = self.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        success = kernel32.SetInformationJobObject(
            handle,
            self.JobObjectExtendedLimitInformation,
            ctypes.byref(information),
            ctypes.sizeof(information),
        )
        if not success:
            error = ctypes.get_last_error()
            kernel32.CloseHandle(handle)
            raise OSError(error, "SetInformationJobObject failed")
        self.handle = handle

    def assign(self, pid: int) -> bool:
        if self.handle is None or os.name != "nt":
            return False
        process = kernel32.OpenProcess(
            self.PROCESS_TERMINATE | self.PROCESS_SET_QUOTA,
            False,
            int(pid),
        )
        if not process:
            return False
        try:
            return bool(kernel32.AssignProcessToJobObject(self.handle, process))
        finally:
            kernel32.CloseHandle(process)

    def terminate(self, exit_code: int = 1) -> None:
        if self.handle is not None and os.name == "nt":
            kernel32.TerminateJobObject(self.handle, int(exit_code))

    def close(self) -> None:
        if self.handle is not None and os.name == "nt":
            kernel32.CloseHandle(self.handle)
            self.handle = None

    def __enter__(self) -> "WindowsJob":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
