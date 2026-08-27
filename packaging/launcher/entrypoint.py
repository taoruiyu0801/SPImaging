"""Absolute-import wrapper and frozen Tcl/Tk build health check."""

from __future__ import annotations

import sys
from typing import Sequence


TCL_SELF_TEST_ARGUMENT = "--launcher-tcl-self-test"


def tcl_self_test() -> int:
    """Return non-zero when the frozen Tcl DLL and script library do not match."""

    try:
        import tkinter

        interpreter = tkinter.Tcl()
        patchlevel = interpreter.eval("info patchlevel")
        library = interpreter.eval("info library")
    except Exception as error:
        print(f"SPImaging launcher Tcl/Tk self-test failed: {error}", file=sys.stderr)
        return 86
    print(f"Tcl {patchlevel}; library={library}")
    return 0


def entrypoint(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments == [TCL_SELF_TEST_ARGUMENT]:
        return tcl_self_test()

    from launcher.app import main

    return main(arguments)


if __name__ == "__main__":
    raise SystemExit(entrypoint())
