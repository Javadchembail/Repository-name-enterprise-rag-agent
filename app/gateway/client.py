import logfire
from portkey_ai import Portkey, createHeaders, PORTKEY_GATEWAY_URL
from langchain_openai import ChatOpenAI

from app.config import settings


def get_langchain_llm(feature: str = "rag") -> ChatOpenAI:
    """
    Returns a Portkey-backed ChatOpenAI using Portkey Config jd.

    Portkey is responsible for:
      - primary/fallback routing
      - retry handling
      - Groq provider selection

    The OpenAI-compatible client itself does NOT retry.
    This prevents long waits such as:
        429 -> wait 56 seconds -> retry
    """

    return ChatOpenAI(
        api_key=settings.PORTKEY_API_KEY,
        base_url=PORTKEY_GATEWAY_URL,
        model="openai/gpt-oss-120b",
        temperature=0,

        # IMPORTANT:
        # Do not let ChatOpenAI perform its own retries.
        # Portkey Config handles retry + fallback.
        max_retries=0,

        default_headers=createHeaders(
            api_key=settings.PORTKEY_API_KEY,
            config="pc-jd-a20ddb",
            metadata={
                "feature": feature,
                "_user": "rag-system-today",
                "environment": "dev"
            }
        )
    )


# Keep this for responder.py compatibility.
# Do NOT attach GATEWAY_CONFIG here.
portkey_client = Portkey(
    api_key=settings.PORTKEY_API_KEY
)


def extract_cache_status(response) -> str:
    """
    Pull x-portkey-cache-status from the Portkey response headers.
    """

    for attr in ("_raw_response", "_response", "_http_response"):
        raw = getattr(response, attr, None)

        if raw is not None:
            status = getattr(raw, "headers", {}).get(
                "x-portkey-cache-status", ""
            )

            if status:
                return status.upper()

    return "MISS"