"""SPImaging photon-efficient imaging workbench."""

# ``PRODUCT_VERSION`` is the version used by the software-copyright product
# name. ``__version__`` remains the tested technical build identifier used by
# packaging and update compatibility checks.
PRODUCT_NAME = "SPImaging单光子三维成像仿真与智能重建软件"
PRODUCT_VERSION = "V1.0"
__version__ = "0.2.0-beta.1"


def product_display_name() -> str:
    """Return the consistent software-copyright name shown in the desktop UI."""

    return f"{PRODUCT_NAME} {PRODUCT_VERSION}"


__all__ = ["PRODUCT_NAME", "PRODUCT_VERSION", "__version__", "product_display_name"]
