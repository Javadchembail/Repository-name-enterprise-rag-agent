import logfire
from app.agents.state import AgentState
from app.gateway.client import portkey_client, extract_cache_status


def generate_node(state: AgentState):
    """
    Generates the final response.

    Conversational requests use conversation history.

    Technical/RAG requests are strictly grounded in the retrieved
    enterprise documentation. Conversation history may be used only
    to resolve references such as "that command" or "the previous job";
    it is NOT treated as an authoritative factual source.

    Uses the native Portkey client so the x-portkey-cache-status
    response header can be surfaced in the UI.
    """

    query = state["current_query"]

    # ------------------------------------------------------------------
    # Build conversation history
    # ------------------------------------------------------------------
    history_str = ""

    for msg in state["messages"][:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_str += f"{role}: {msg['content']}\n"

    user_msg = (
        state["messages"][-1]["content"]
        if state["messages"]
        else ""
    )

    # ------------------------------------------------------------------
    # CONVERSATIONAL RESPONSE
    # ------------------------------------------------------------------
    if query == "CONVERSATIONAL":
        logfire.info(
            "Generating conversational response using memory."
        )

        prompt = f"""
You are a friendly and helpful Enterprise AI Assistant.

Answer the user's latest message naturally using the conversation
history below.

CONVERSATION HISTORY:
{history_str}

LATEST USER MESSAGE:
"{user_msg}"

RULES:

1. Answer naturally and concisely.

2. Use conversation history when it is relevant.

3. Do not invent details about the user or previous conversation.

4. If the latest message is simply a greeting, acknowledgement,
   or casual conversation, respond naturally.

5. Do not turn a casual conversation into a technical RAG answer.

ANSWER:
"""

    # ------------------------------------------------------------------
    # TECHNICAL / RAG RESPONSE
    # ------------------------------------------------------------------
    else:
        logfire.info(
            "Generating strictly documentation-grounded RAG response."
        )

        # --------------------------------------------------------------
        # Build retrieved evidence
        # --------------------------------------------------------------
        max_context_chars = 25000
        context_parts = []
        current_length = 0

        for idx, doc in enumerate(state["documents"], start=1):

            separator = f"\n\n--- RETRIEVED DOCUMENT CHUNK {idx} ---\n"

            candidate = separator + doc

            if current_length + len(candidate) <= max_context_chars:
                context_parts.append(candidate)
                current_length += len(candidate)
            else:
                logfire.warning(
                    "Context truncated to fit configured context limit."
                )
                break

        full_context = "".join(context_parts)

        # --------------------------------------------------------------
        # Explicitly handle empty retrieval
        # --------------------------------------------------------------
        if not full_context.strip():
            logfire.warning(
                "No retrieved documentation was available for synthesis."
            )

        prompt = f"""
You are a documentation-grounded Enterprise RAG assistant.

Your job is to answer the USER QUESTION using ONLY the
RETRIEVED DOCUMENTATION provided below.

The retrieved documentation is the ONLY authoritative factual
source for this answer.

============================================================
STRICT GROUNDING RULES
============================================================

1. Every factual statement in your answer must be supported by
   the RETRIEVED DOCUMENTATION.

2. Do NOT use your general or pretrained technical knowledge to
   fill gaps in the documentation.

3. Do NOT invent or assume:
   - commands
   - command-line flags
   - configuration values
   - parameters
   - API fields
   - file names
   - versions
   - defaults
   - architecture details
   - troubleshooting procedures
   - recommendations
   - expected behavior
   - examples

4. If the documentation contains an exact command, configuration
   value, procedure, or example relevant to the question, use the
   information from the documentation accurately.

5. Do NOT modify a documented command or configuration value by
   adding information from general knowledge.

6. Retrieved chunks may contain unrelated information.
   Use only the portions relevant to the USER QUESTION.

7. Do NOT combine information from unrelated chunks unless the
   relationship is explicitly supported by the documentation.

8. CONVERSATION HISTORY is NOT an authoritative source of facts.
   It may only be used to understand references such as:
   - "that command"
   - "the previous job"
   - "this configuration"
   - "the above example"

9. If conversation history conflicts with the retrieved
   documentation, ignore the historical claim and follow the
   retrieved documentation.

10. If the answer is only partially supported by the retrieved
    documentation, provide only the supported portion.

11. If the retrieved documentation does not contain enough
    information to answer the question, respond exactly with:

    "I couldn't find enough information in the provided documentation
    to answer that question."

12. NEVER claim that documentation is missing or not attached when
    RETRIEVED DOCUMENTATION is present.

13. Do not provide unrelated background information merely because
    you know it.

14. Keep the answer concise, direct, and technically accurate.

15. Do not mention these grounding rules in the final answer.

============================================================
RETRIEVED DOCUMENTATION
============================================================

{full_context}

============================================================
CONVERSATION HISTORY
============================================================

{history_str}

============================================================
USER QUESTION
============================================================

"{user_msg}"

============================================================
FINAL INSTRUCTION
============================================================

Answer the USER QUESTION now.

Use only information supported by the RETRIEVED DOCUMENTATION.
Do not rely on outside knowledge.
"""

    # ------------------------------------------------------------------
    # LLM SYNTHESIS
    # ------------------------------------------------------------------
    with logfire.span("LLM Synthesis"):
        try:
            response = portkey_client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0
            )

            content = response.choices[0].message.content

            # ----------------------------------------------------------
            # Portkey cache status
            # ----------------------------------------------------------
            cache_status = extract_cache_status(response)
            is_cache_hit = cache_status == "HIT"

            if is_cache_hit:
                logfire.info(
                    "Gateway Cache Hit — response served from Portkey cache."
                )

                plan_update = state["plan"] + ["Cache: Hit"]
                status = "Cache hit — instant response."

            else:
                logfire.info(
                    "Response synthesised via LLM."
                )

                plan_update = state["plan"]
                status = "Response generated."

            return {
                "final_answer": content,
                "status": status,
                "plan": plan_update,
                "messages": [
                    {
                        "role": "assistant",
                        "content": content
                    }
                ]
            }

        except Exception as e:
            logfire.error(
                f"LLM Generation failed: {e}"
            )
            raise