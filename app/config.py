import os
from dotenv import load_dotenv


load_dotenv()

class settings:
    # --- GEMINI EMBEDDINGS ---
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

     # --- VECTOR DB (QDRANT) ---
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_COLLECTION_NAME = "enterprise_rag"  #genai


     # --- REASONING ENGINE (GROQ) ---
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL =  "openai/gpt-oss-120b" #"llama-3.3-70b-versatile"
    GROQ_FALLBACK_API_KEY = os.getenv("GROQ_FALLBACK_API_KEY")


    # --- LLM GATEWAY (PORTKEY) ---
    PORTKEY_API_KEY = os.getenv("PORTKEY_API_KEY")
    GROQ_SLUG =  "rag"     # primary: @rag/openai/gpt-oss-120b
    GROQ_SLUG_2 = "rag1"  # fallback: "@rag1/openai/gpt-oss-120b"


settings = settings()