"""Create the Qdrant collections: production, and (Phase 4) staging, used
for admin-console uploads pending review/promotion.

Usage: python -m app.scripts.init_collection
"""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from app.clients.qdrant_client import get_qdrant_client
from app.clients.voyage_client import get_voyage_client
from app.config.settings import get_settings
from app.services.ingestion.embedding import embed_query


def _create_if_missing(qdrant: QdrantClient, collection: str, vector_size: int) -> None:
    if qdrant.collection_exists(collection):
        print(f"Collection '{collection}' already exists.")
        return
    qdrant.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    print(f"Created collection '{collection}' (size={vector_size}, distance=cosine).")


def main() -> None:
    settings = get_settings()
    qdrant = get_qdrant_client()
    voyage = get_voyage_client()
    vector_size = len(embed_query(voyage, "dimension probe"))

    _create_if_missing(qdrant, settings.qdrant_collection_prod, vector_size)
    _create_if_missing(qdrant, settings.qdrant_collection_staging, vector_size)


if __name__ == "__main__":
    main()
