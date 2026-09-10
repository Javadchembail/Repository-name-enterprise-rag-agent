"""
Phase 1 — Live Pipeline.

Calls the running FastAPI /query endpoint for each golden sample.

Captures:
    actual_response
    actual_contexts
    actual_tools_called

The complete LLM response is preserved for evaluation.
"""

import time
import copy
import json
import os
import requests
import logfire


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

API_URL = "http://localhost:8000/query"

# IMPORTANT:
# Keep the full response for RAGAS evaluation.
# 1200 is a safety ceiling, but much larger than the old 300-character limit.
RESPONSE_TRUNCATE = 1200

# Delay between live API calls.
DELAY_BETWEEN_CALLS = 10

# Maximum time allowed for one FastAPI request.
REQUEST_TIMEOUT = 120


# ─────────────────────────────────────────────────────────────────────────────
# Tool Detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_tool(thought_process: list) -> str:
    """
    Maps the thought_process list from /query response to a tool name.

    Planner sets:
        'Intent: Technical' + 'Search Term: ...'
            → retrieve_documents

        'Intent: Conversational/Memory'
            → direct_answer

    main.py sets:
        'Intent: Guardrails Fired'
            → guardrails
    """

    if not thought_process:
        return "unknown"

    joined = " ".join(
        str(item)
        for item in thought_process
    ).lower()

    if "guardrails fired" in joined:
        return "guardrails"

    if (
        "intent: technical" in joined
        or "search term:" in joined
        or "context retrieved" in joined
    ):
        return "retrieve_documents"

    if (
        "conversational" in joined
        or "memory" in joined
    ):
        return "direct_answer"

    return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Live Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    golden_dataset: dict,
    progress_callback=None
) -> dict:
    """
    Enriches each rag_sample in golden_dataset with live API results.

    Returns a deep copy with:

        actual_response
        actual_contexts
        actual_tools_called

    populated.

    progress_callback(
        i,
        total,
        question,
        stage,
        response=""
    )
    is called per step.
    """

    dataset = copy.deepcopy(
        golden_dataset
    )

    samples = dataset["rag_samples"]

    n = len(samples)

    logfire.info(
        "🚀 Starting live pipeline",
        total_samples=n,
        api_url=API_URL,
    )

    with logfire.span(
        "🚀 Eval Phase 1 — Live Pipeline",
        total_samples=n,
    ):

        for i, sample in enumerate(samples):

            question = sample["question"]

            query_number = i + 1

            # ================================================================
            # START QUERY
            # ================================================================

            logfire.info(
                f"🚀 Starting Query "
                f"{query_number}/{n}: {question}"
            )

            # ================================================================
            # CALLBACK — CALLING
            # ================================================================

            if progress_callback:

                logfire.info(
                    f"📤 Sending progress callback — "
                    f"calling — Query {query_number}/{n}"
                )

                try:

                    progress_callback(
                        i,
                        n,
                        question,
                        "calling"
                    )

                    logfire.info(
                        f"➡️ Calling callback finished — "
                        f"Query {query_number}/{n}"
                    )

                except Exception as e:

                    logfire.error(
                        f"⚠️ Calling progress callback failed — "
                        f"Query {query_number}/{n}: {e}"
                    )

            # ================================================================
            # FASTAPI REQUEST
            # ================================================================

            with logfire.span(
                f"📤 Live Query {query_number}/{n}",
                question=question[:80],
                domain=sample.get(
                    "domain",
                    ""
                ),
            ):

                try:

                    logfire.info(
                        f"📡 Calling FastAPI /query "
                        f"for Query {query_number}/{n}"
                    )

                    start_time = time.time()

                    resp = requests.post(
                        API_URL,
                        json={
                            "q": question,
                            "thread_id": f"eval_run_{i}",
                        },
                        timeout=REQUEST_TIMEOUT,
                    )

                    elapsed = (
                        time.time()
                        - start_time
                    )

                    logfire.info(
                        f"📥 FastAPI response received "
                        f"for Query {query_number}/{n}",
                        elapsed_seconds=round(
                            elapsed,
                            2
                        ),
                        status_code=resp.status_code,
                    )

                    resp.raise_for_status()

                    logfire.info(
                        f"🔍 Parsing JSON response "
                        f"for Query {query_number}/{n}"
                    )

                    data = resp.json()

                    # ========================================================
                    # EXTRACT RESPONSE
                    # ========================================================

                    raw_answer = (
                        data.get("answer")
                        or ""
                    )

                    thought_process = (
                        data.get(
                            "thought_process"
                        )
                        or []
                    )

                    sources = (
                        data.get("sources")
                        or []
                    )

                    # ========================================================
                    # STORE RESULTS
                    # ========================================================

                    # IMPORTANT:
                    # Preserve a much larger portion of the answer.
                    #
                    # The previous 300-character limit could cut off
                    # commands, configuration values, explanations,
                    # and important answer details.
                    sample["actual_response"] = (
                        raw_answer[:RESPONSE_TRUNCATE]
                    )

                    # Preserve the top 5 sources returned by the API.
                    sample["actual_contexts"] = (
                        sources[:5]
                    )

                    detected_tool = detect_tool(
                        thought_process
                    )

                    sample["actual_tools_called"] = [
                        detected_tool
                    ]

                    # ========================================================
                    # SUCCESS LOG
                    # ========================================================

                    logfire.info(
                        "✅ Response captured",
                        tool=detected_tool,
                        response_chars=len(
                            raw_answer
                        ),
                        stored_response_chars=len(
                            sample["actual_response"]
                        ),
                        context_chunks=len(
                            sources
                        ),
                        stored_context_chunks=len(
                            sample["actual_contexts"]
                        ),
                        elapsed_seconds=round(
                            elapsed,
                            2
                        ),
                    )

                # ============================================================
                # CONNECTION ERROR
                # ============================================================

                except requests.exceptions.ConnectionError as e:

                    logfire.error(
                        "❌ Cannot reach FastAPI — "
                        "is the app running on :8000?",
                        error=str(e),
                    )

                    sample["actual_response"] = ""

                    sample["actual_contexts"] = (
                        sample.get(
                            "relevant_contexts",
                            []
                        )
                    )

                    sample["actual_tools_called"] = [
                        "unknown"
                    ]

                # ============================================================
                # TIMEOUT
                # ============================================================

                except requests.exceptions.Timeout as e:

                    logfire.error(
                        f"⏰ FastAPI timeout — "
                        f"Query {query_number}/{n}",
                        error=str(e),
                    )

                    sample["actual_response"] = ""

                    sample["actual_contexts"] = (
                        sample.get(
                            "relevant_contexts",
                            []
                        )
                    )

                    sample["actual_tools_called"] = [
                        "unknown"
                    ]

                # ============================================================
                # HTTP ERROR
                # ============================================================

                except requests.exceptions.HTTPError as e:

                    logfire.error(
                        f"❌ HTTP error — "
                        f"Query {query_number}/{n}",
                        error=str(e),
                    )

                    sample["actual_response"] = ""

                    sample["actual_contexts"] = (
                        sample.get(
                            "relevant_contexts",
                            []
                        )
                    )

                    sample["actual_tools_called"] = [
                        "unknown"
                    ]

                # ============================================================
                # JSON ERROR
                # ============================================================

                except json.JSONDecodeError as e:

                    logfire.error(
                        f"❌ Invalid JSON response — "
                        f"Query {query_number}/{n}",
                        error=str(e),
                    )

                    sample["actual_response"] = ""

                    sample["actual_contexts"] = (
                        sample.get(
                            "relevant_contexts",
                            []
                        )
                    )

                    sample["actual_tools_called"] = [
                        "unknown"
                    ]

                # ============================================================
                # OTHER ERROR
                # ============================================================

                except Exception as e:

                    logfire.exception(
                        f"❌ Query failed — "
                        f"Query {query_number}/{n}: {e}"
                    )

                    sample["actual_response"] = ""

                    sample["actual_contexts"] = (
                        sample.get(
                            "relevant_contexts",
                            []
                        )
                    )

                    sample["actual_tools_called"] = [
                        "unknown"
                    ]

            # ================================================================
            # CALLBACK — DONE
            # ================================================================

            logfire.info(
                f"➡️ Query {query_number}/{n} finished — "
                f"before progress callback"
            )

            if progress_callback:

                try:

                    progress_callback(
                        i,
                        n,
                        question,
                        "done",
                        sample["actual_response"]
                    )

                    logfire.info(
                        f"➡️ Query {query_number}/{n} "
                        f"progress callback finished"
                    )

                except Exception as e:

                    logfire.error(
                        f"⚠️ Done progress callback failed — "
                        f"Query {query_number}/{n}: {e}"
                    )

            # ================================================================
            # DELAY BEFORE NEXT QUERY
            # ================================================================

            if i < n - 1:

                logfire.info(
                    f"⏳ Waiting "
                    f"{DELAY_BETWEEN_CALLS}s "
                    f"before Query "
                    f"{query_number + 1}/{n}"
                )

                # ------------------------------------------------------------
                # Tell Streamlit we are waiting
                # ------------------------------------------------------------

                if progress_callback:

                    try:

                        progress_callback(
                            i,
                            n,
                            question,
                            "waiting",
                            sample["actual_response"]
                        )

                    except Exception as e:

                        logfire.error(
                            f"⚠️ Waiting progress callback failed — "
                            f"Query {query_number}/{n}: {e}"
                        )

                # ------------------------------------------------------------
                # Countdown
                # ------------------------------------------------------------

                for remaining in range(
                    DELAY_BETWEEN_CALLS,
                    0,
                    -1
                ):

                    logfire.info(
                        f"⏳ Query "
                        f"{query_number + 1}/{n} "
                        f"starts in {remaining}s"
                    )

                    time.sleep(1)

                logfire.info(
                    f"➡️ Delay finished — "
                    f"starting Query "
                    f"{query_number + 1}/{n}"
                )

    # ─────────────────────────────────────────────────────────────────────────
    # PIPELINE COMPLETE
    # ─────────────────────────────────────────────────────────────────────────

    logfire.info(
        "🎉 Eval Phase 1 — Live Pipeline completed",
        total_samples=n,
    )

    return dataset


# ─────────────────────────────────────────────────────────────────────────────
# Save Results
# ─────────────────────────────────────────────────────────────────────────────

def save_results(
    dataset: dict,
    path: str
) -> None:

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            dataset,
            f,
            indent=2,
            ensure_ascii=False
        )


# ─────────────────────────────────────────────────────────────────────────────
# Load Golden Dataset
# ─────────────────────────────────────────────────────────────────────────────

def load_golden_dataset() -> dict:

    golden_path = os.path.join(
        os.path.dirname(__file__),
        "golden_dataset.json"
    )

    with open(
        golden_path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)