"""Create the Qdrant collection used for Phase 1 (single, unfiltered collection).

Usage: python -m app.scripts.init_collection
"""

from qdrant_client.models import Distance, VectorParams

from app.clients.qdrant_client import get_qdrant_client
from app.clients.voyage_client import get_voyage_client
from app.config.settings import get_settings
from app.services.ingestion.embedding import embed_query


def main() -> None:
    settings = get_settings()
    qdrant = get_qdrant_client()
    voyage = get_voyage_client()

    vector_size = len(embed_query(voyage, "dimension probe"))

    if qdrant.collection_exists(settings.qdrant_collection_prod):
        print(f"Collection '{settings.qdrant_collection_prod}' already exists.")
        return

    qdrant.create_collection(
        collection_name=settings.qdrant_collection_prod,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    print(
        f"Created collection '{settings.qdrant_collection_prod}' "
        f"(size={vector_size}, distance=cosine)."
    )


if __name__ == "__main__":
    main()
