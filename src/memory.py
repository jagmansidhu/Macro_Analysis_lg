"""Long-term memory: persists Analysis_Profile records to Postgres with vector search."""

from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Session

from config import DB_URL, embeddings

EMBEDDING_DIM = 1536  # matches output_dimensionality in config.py


class Base(DeclarativeBase):
    pass


class AnalysisRecord(Base):
    __tablename__ = "analysis_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=True)
    desc = Column(Text, nullable=True)
    outcome = Column(Text, nullable=True)
    data = Column(JSONB, nullable=False, default=list)
    user_query = Column(Text, nullable=True)
    query_embedding = Column(Vector(EMBEDDING_DIM), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), nullable=True)


engine = create_engine(DB_URL)
Base.metadata.create_all(engine)


def _embed_text(text: str) -> list[float]:
    """Embed a single string using the project's embedding model."""
    return embeddings.embed_query(text)


def save_profile(profile, user_query: str | None = None) -> int:
    """Save an Analysis_Profile to Postgres with an embedded query vector. Returns the row id."""
    query_vec = _embed_text(user_query) if user_query else None

    record = AnalysisRecord(
        name=profile.name,
        desc=profile.desc,
        outcome=profile.outcome,
        data=profile.data,
        user_query=user_query,
        query_embedding=query_vec,
    )
    with Session(engine) as session:
        session.add(record)
        session.commit()
        row_id = record.id

    print(f"[Memory] Saved analysis profile (id={row_id})")
    return row_id


def update_profile(record_id: int, profile) -> int:
    """Update an existing analysis record with new profile data. Returns the record id."""
    with Session(engine) as session:
        record = session.get(AnalysisRecord, record_id)
        if record is None:
            print(f"[Memory] Record id={record_id} not found, saving as new.")
            return save_profile(profile, user_query=None)

        record.name = profile.name
        record.desc = profile.desc
        record.outcome = profile.outcome
        record.data = profile.data
        record.updated_at = datetime.now(timezone.utc)
        session.commit()

    print(f"[Memory] Updated analysis profile (id={record_id})")
    return record_id


def search_past_analyses(query: str, top_k: int = 3, similarity_threshold: float = 0.75) -> list[dict]:
    """Find past analyses whose user_query is semantically similar to the new query.

    Returns a list of dicts with keys: id, user_query, name, desc, outcome, data, similarity, created_at.
    Only returns results above the similarity_threshold (cosine similarity).
    """
    query_vec = _embed_text(query)

    # Use pgvector's cosine distance operator: <=> returns distance (1 - similarity)
    distance = AnalysisRecord.query_embedding.cosine_distance(query_vec).label("distance")

    with Session(engine) as session:
        rows = (
            session.query(AnalysisRecord, distance)
            .filter(AnalysisRecord.query_embedding.isnot(None))
            .order_by(distance)
            .limit(top_k)
            .all()
        )

    results = []
    for record, dist in rows:
        similarity = 1.0 - dist
        if similarity < similarity_threshold:
            continue
        results.append({
            "id": record.id,
            "user_query": record.user_query,
            "name": record.name,
            "desc": record.desc,
            "outcome": record.outcome,
            "data": record.data,
            "similarity": round(similarity, 4),
            "created_at": record.created_at.isoformat() if record.created_at else None,
        })

    return results
