import logfire
from langchain_groq import ChatGroq
from nemoguardrails import RailsConfig, LLMRails

from app.config import settings
from app.guardrails.colang_rules import (
    COLANG_CONTENT,
    YAML_CONTENT,
    RAIL_INDICATORS,
)


_rails: LLMRails | None = None


def initialize_rails() -> None:
    """
    Build the NeMo LLMRails singleton at app startup.
    Uses GPT-OSS-120B for guardrail intent classification.
    """
    global _rails

    guard_llm = ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model="openai/gpt-oss-120b",
        temperature=0
    )

    config = RailsConfig.from_content(
        colang_content=COLANG_CONTENT,
        yaml_content=YAML_CONTENT
    )

    _rails = LLMRails(
        config,
        llm=guard_llm
    )

    logfire.info(
        "🛡️ NeMo Guardrails initialised (GPT-OSS-120B)."
    )


def guard(message: str) -> tuple[bool, str | None]:
    """
    Run a user message through the NeMo rails gate.

    Returns:
        (True, rail_response) — a rail fired; return this response immediately,
                                skip the RAG pipeline entirely.
        (False, None) — message is clean; proceed to LangGraph.
    """

    if _rails is None:
        logfire.warning(
            "⚠️ Guardrails not initialised — skipping gate."
        )
        return False, None

    with logfire.span("🛡️ Guardrails Check"):

        # Normalize punctuation so:
        # "hello", "hello?", "hello!" are treated consistently.
        normalized = (
            message
            .strip()
            .lower()
            .rstrip("?!.,")
        )

        # ---------------------------------------------------------
        # Greeting
        # ---------------------------------------------------------

        greetings = {
            "hello",
            "hi",
            "hey",
            "good morning",
            "good afternoon",
            "what's up",
            "howdy"
        }

        if normalized in greetings:
            return True, (
                "Hello! I'm your Enterprise IT Assistant. "
                "I specialise in Kubernetes, Intel hardware, "
                "and enterprise networking. What can I help "
                "you with today?"
            )

        # ---------------------------------------------------------
        # Farewell
        # ---------------------------------------------------------

        farewells = {
            "bye",
            "goodbye",
            "see you",
            "thanks bye",
            "that is all",
            "i am done",
            "see you later"
        }

        if normalized in farewells:
            return True, (
                "Goodbye! Feel free to return whenever you have "
                "more enterprise IT questions. Have a great day!"
            )

        # ---------------------------------------------------------
        # Capabilities
        # ---------------------------------------------------------

        capabilities = {
            "what can you do",
            "what do you know",
            "help",
            "what are you",
            "who are you",
            "what topics do you cover",
            "what can i ask you",
            "what are your capabilities",
            "what is your purpose",
            "what can you help me with"
        }

        if normalized in capabilities:
            return True, (
                "I'm an Enterprise AI Assistant with deep expertise in: "
                "Kubernetes (deployment, scaling, networking, operators), "
                "Intel Hardware (CPUs, FPGAs, SRIOV, NICs), "
                "Enterprise Networking (SDN, VLANs, BGP, routing). "
                "Ask me anything in these areas!"
            )

        # ---------------------------------------------------------
        # NeMo Input Rail
        # ---------------------------------------------------------

        result = _rails.generate(
            messages=[
                {
                    "role": "user",
                    "content": normalized
                }
            ],
            options={
                "rails": ["input"]
            }
        )

        # ---------------------------------------------------------
        # TEMPORARY DEBUG
        # ---------------------------------------------------------

        print("\n" + "=" * 80)
        print("🛡️ GUARDRAIL DEBUG")
        print("=" * 80)
        print(f"INPUT: {normalized}")
        print("RAW NEMO RESULT:")
        print(result)
        print("=" * 80 + "\n")

        # ---------------------------------------------------------
        # Extract NeMo Response
        # ---------------------------------------------------------

        if isinstance(result, dict):

            content = result.get(
                "content",
                ""
            )

        else:

            response = getattr(
                result,
                "response",
                []
            )

            content = (
                response[0].get(
                    "content",
                    ""
                )
                if response
                else ""
            )

        # ---------------------------------------------------------
        # TEMPORARY DEBUG — EXTRACTED CONTENT
        # ---------------------------------------------------------

        print("\n" + "-" * 80)
        print("🛡️ EXTRACTED GUARDRAIL CONTENT")
        print("-" * 80)
        print(content)
        print("-" * 80 + "\n")

        # ---------------------------------------------------------
        # Remove GPT-OSS reasoning from returned response
        # ---------------------------------------------------------

        if (
            "<think>" in content
            and "</think>" in content
        ):
            content = (
                content
                .split("</think>", 1)[1]
                .strip()
            )

        # ---------------------------------------------------------
        # Detect Guardrail Response
        # ---------------------------------------------------------

        fired = any(
            indicator.lower() in content.lower()
            for indicator in RAIL_INDICATORS
        )

        if "can't respond to that" in content.lower():
            fired = True

        # ---------------------------------------------------------
        # TEMPORARY DEBUG — FINAL DECISION
        # ---------------------------------------------------------

        print("\n" + "-" * 80)
        print("🛡️ GUARDRAIL DECISION")
        print("-" * 80)
        print(f"Input: {normalized}")
        print(f"Content: {content}")
        print(f"RAIL_INDICATORS: {RAIL_INDICATORS}")
        print(f"Fired: {fired}")
        print("-" * 80 + "\n")

        if fired:

            logfire.info(
                f"🛡️ Guardrails fired | query='{message[:80]}'"
            )

            return True, content

        logfire.info(
            "✅ Guardrails passed."
        )

        return False, None