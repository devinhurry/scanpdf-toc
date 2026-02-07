"""Public package API for scanpdf-toc."""

from importlib.metadata import PackageNotFoundError, version

from .generator import PdfTocGenerator, TocEntry

try:
    __version__ = version("scanpdf-toc")
except PackageNotFoundError:  # pragma: no cover - local editable installs
    __version__ = "0.1.0"

__all__ = ["PdfTocGenerator", "TocEntry", "__version__"]
