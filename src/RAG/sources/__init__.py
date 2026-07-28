"""RAG source registry — each module exposes a BaseSource subclass."""

from .local_file_source import LocalFileSource
from .fred_api_source import FredApiSource
from .web_source import WebSource

__all__ = ["LocalFileSource", "FredApiSource", "WebSource"]
