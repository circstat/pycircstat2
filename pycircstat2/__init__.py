from importlib import metadata as _metadata

from .base import Axial, Circular
from .utils import load_data
from .visualization import circ_plot

try:  # Prefer installed package metadata
    __version__ = _metadata.version("pycircstat2")
except _metadata.PackageNotFoundError:  # pragma: no cover - during local dev
    __version__ = "0.0.0"

__all__ = ["Axial", "Circular", "circ_plot", "load_data", "__version__"]
