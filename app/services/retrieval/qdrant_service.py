
import logfire
from qdrant_client import QdrantClient
from app.config import settings
from app.services.retrieval.embeddings import embed_query


client = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY,
    timeout=30,
    prefer_grpc=False,
)


def test_qdrant_connection():
    try:
        collections = client.get_collections()
        names = [c.name for c in collections.collections]

        if settings.QDRANT_COLLECTION_NAME not in names:
            raise RuntimeError(
                f"Collection not found: {settings.QDRANT_COLLECTION_NAME}"
            )

        info = client.get_collection(settings.QDRANT_COLLECTION_NAME)

        result = {
            "connected": True,
            "collection": settings.QDRANT_COLLECTION_NAME,
            "points": info.points_count,
            "status": str(info.status),
        }

        logfire.info("Qdrant connection successful", **result)
        return result

    except Exception:
        logfire.exception("Qdrant connection test failed")
        raise


def search_enterprise_knowledge(query: str, limit: int = 8):
    try:
        query_vector = embed_query(query)

        if len(query_vector) != 3072:
            raise ValueError(
                f"Expected 3072 dimensions, got {len(query_vector)}"
            )

        response = client.query_points(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            query=query_vector,
            limit=limit,
            with_payload=True,
        )

        results = []

        for point in response.points:
            payload = point.payload or {}

            results.append({
                "content": payload.get("text", ""),
                "source": payload.get("source", "Unknown"),
                "score": point.score,
            })

        logfire.info(
            "Qdrant search completed",
            result_count=len(results),
        )

        return results

    except Exception:
        logfire.exception("Qdrant search failed")
        return []
