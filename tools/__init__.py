# # File: star_tracker/__init__.py
"""
DSP-tools - dsp image processing tools
"""
# ---------------------------------------------------------------------------
# Imports
from importlib.metadata import version, PackageNotFoundError
from tools.main import main

# ---------------------------------------------------------------------------
# Public API re-exports
# ---------------------------------------------------------------------------

__all__ = [
    "main",
    "__version__",
]

try:
    __version__ = version("DSP-tools")
except PackageNotFoundError:        # running from a checkout / no wheel yet
    __version__ = "0.0.0.dev0"