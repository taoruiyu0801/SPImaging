"""Absolute-import wrapper used only by the frozen launcher build."""

from launcher.app import main


if __name__ == "__main__":
    raise SystemExit(main())
