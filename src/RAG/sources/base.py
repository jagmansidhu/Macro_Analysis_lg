"""Abstract base class for all RAG data sources."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document


@dataclass
class IndexingResult:
    """Summary returned by any ingest operation."""

    num_added: int = 0
    num_updated: int = 0
    num_skipped: int = 0
    num_deleted: int = 0
    errors: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"added={self.num_added} updated={self.num_updated} "
            f"skipped={self.num_skipped} deleted={self.num_deleted} "
            f"errors={len(self.errors)}"
        )

    def merge(self, other: "IndexingResult") -> "IndexingResult":
        """Combine two results (useful when ingesting multiple batches)."""
        return IndexingResult(
            num_added=self.num_added + other.num_added,
            num_updated=self.num_updated + other.num_updated,
            num_skipped=self.num_skipped + other.num_skipped,
            num_deleted=self.num_deleted + other.num_deleted,
            errors=self.errors + other.errors,
        )


class BaseSource(ABC):
    """
    Contract every RAG data source must satisfy.

    Subclasses implement:
      - ``fetch``      — on-demand pull triggered by the agent (returns docs, indexes them)
      - ``ingest_all`` — bulk re-index of the entire source (used at startup or on reset)
    """

    # Human-readable name shown in logs and tool descriptions.
    source_name: str = "unnamed_source"

    @abstractmethod
    def fetch(self, query: str, **kwargs: Any) -> list[Document]:
        """
        Pull documents relevant to ``query`` from this source.

        Implementations MUST:
        1. Fetch raw content from the external system.
        2. Wrap it in ``langchain_core.documents.Document`` objects with a
           ``source`` metadata key that uniquely identifies the origin.
        3. Index the documents into PGVector before returning them so that
           follow-up queries are served from the DB (cache-first behaviour).

        Returns the list of freshly fetched (and now indexed) Documents.
        """

    @abstractmethod
    def ingest_all(self) -> IndexingResult:
        """
        Bulk-index the entire source.

        Used:
          - At server startup (incremental mode — skips already-indexed content)
          - When an operator wants a full reset of this source's data
        """
