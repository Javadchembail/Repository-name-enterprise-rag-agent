import os
import uuid

import requests
import streamlit as st
import logfire
from dotenv import load_dotenv


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

env_path = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".env",
    )
)

load_dotenv(dotenv_path=env_path)


# ============================================================
# INITIALIZE LOGFIRE
# ============================================================

try:
    token = os.getenv("LOGFIRE_TOKEN")

    if not token:
        print("ERROR: LOGFIRE_TOKEN is empty or None!")

    logfire.configure(token=token)

    # Disabled due to OpenTelemetry bug on Windows:
    # logfire.instrument_requests()

    LOGFIRE_STATUS = "Connected & Tracing"

except Exception as e:
    print(f"Logfire Init Error in UI: {e}")
    LOGFIRE_STATUS = f"Standby (Error: {e})"


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Enterprise Agentic RAG",
    page_icon="🤖",
    layout="wide",
)


# ============================================================
# AVATARS
# ============================================================

AI_AVATAR = "🤖"
USER_AVATAR = "👤"


# ============================================================
# SESSION MANAGEMENT
# ============================================================

if "session_id" not in st.session_state:

    st.session_state.session_id = str(uuid.uuid4())

    logfire.info(
        f"✨ New User Session Created: "
        f"{st.session_state.session_id}"
    )


if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# REQUEST STATE
# ============================================================

if "pending_prompt" not in st.session_state:

    st.session_state.pending_prompt = None


if "processing_request" not in st.session_state:

    st.session_state.processing_request = False


if "last_request_id" not in st.session_state:

    st.session_state.last_request_id = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🧠 Agent OS")

    st.markdown("---")

    st.success(
        f"Logfire: {LOGFIRE_STATUS}"
    )

    st.info(
        f"Memory ID: "
        f"{st.session_state.session_id[:8]}"
    )

    if st.button(
        "🗑️ Clear History & Memory",
        width="stretch",
        type="primary",
    ):

        logfire.warning(
            "🗑️ Memory Wipe Triggered for session: "
            f"{st.session_state.session_id}"
        )

        st.session_state.messages = []

        st.session_state.pending_prompt = None

        st.session_state.processing_request = False

        st.session_state.last_request_id = None

        st.session_state.session_id = str(uuid.uuid4())

        logfire.info(
            "✨ New User Session Created after memory wipe: "
            f"{st.session_state.session_id}"
        )

        st.rerun()


# ============================================================
# MAIN CHAT
# ============================================================

st.title("🤖 Enterprise Agentic Assistant")


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    avatar = (
        AI_AVATAR
        if message["role"] == "assistant"
        else USER_AVATAR
    )

    with st.chat_message(
        message["role"],
        avatar=avatar,
    ):

        st.markdown(
            message["content"]
        )

        sources = message.get(
            "sources",
            [],
        )

        if sources:

            with st.expander(
                "📄 View Retrieved Context (Sources)"
            ):

                for i, source in enumerate(
                    sources
                ):

                    preview = (
                        source[:100]
                        .replace("\n", " ")
                        + "..."
                    )

                    with st.expander(
                        f"Chunk {i + 1}: {preview}"
                    ):

                        st.info(source)


# ============================================================
# CHAT INPUT
# ============================================================

prompt = st.chat_input(
    "Ask about your documentation..."
)


# ============================================================
# ACCEPT NEW MESSAGE
# ============================================================

if (
    prompt
    and not st.session_state.processing_request
    and not st.session_state.pending_prompt
):

    request_id = str(uuid.uuid4())

    st.session_state.last_request_id = request_id

    st.session_state.pending_prompt = prompt

    st.session_state.processing_request = True

    # Save user message exactly once
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
            "request_id": request_id,
        }
    )

    logfire.info(
        "👤 User message stored.",
        request_id=request_id,
    )

    st.rerun()


# ============================================================
# PROCESS PENDING REQUEST
# ============================================================

if (
    st.session_state.pending_prompt
    and st.session_state.processing_request
):

    prompt = st.session_state.pending_prompt

    request_id = st.session_state.last_request_id

    answer = None

    sources = []

    request_error = None


    # ========================================================
    # USER CHAT TRACE
    # ========================================================

    with logfire.span(
        "💬 User Chat Interaction",
        user_query=prompt,
        session_id=st.session_state.session_id,
        request_id=request_id,
    ):

        # ====================================================
        # ASSISTANT CHAT MESSAGE
        # ====================================================

        with st.chat_message(
            "assistant",
            avatar=AI_AVATAR,
        ):

            # =================================================
            # PROCESSING STATUS
            #
            # IMPORTANT:
            # Only processing information goes inside this
            # status container.
            #
            # The final answer is rendered AFTER this block.
            # =================================================

            with st.status(
                "🔍 Agent is thinking...",
                expanded=True,
            ) as status:

                try:

                    # ==========================================
                    # CALL RAG BACKEND
                    # ==========================================

                    with logfire.span(
                        "📡 Calling RAG Backend",
                        request_id=request_id,
                    ):

                        base_url = os.getenv(
                            "BACKEND_URL",
                            "http://localhost:8000",
                        ).rstrip("/")

                        url = f"{base_url}/query"

                        payload = {
                            "q": prompt,
                            "thread_id": (
                                st.session_state.session_id
                            ),
                        }

                        response = requests.post(
                            url,
                            json=payload,
                            timeout=120,
                        )

                        response.raise_for_status()

                        data = response.json()


                    # ==========================================
                    # GET ANSWER
                    # ==========================================

                    answer = data.get(
                        "answer",
                        "",
                    )

                    if not answer:

                        answer = (
                            "I apologize, but I couldn't "
                            "generate a response for this request."
                        )


                    # ==========================================
                    # GET REASONING
                    # ==========================================

                    steps = data.get(
                        "thought_process",
                        [],
                    )


                    # ==========================================
                    # GET SOURCES
                    # ==========================================

                    sources = data.get(
                        "sources",
                        [],
                    )


                    # ==========================================
                    # SHOW REASONING INSIDE STATUS
                    # ==========================================

                    for step in steps:

                        st.write(
                            f"⚙️ {step}"
                        )


                    # ==========================================
                    # UPDATE STATUS
                    # ==========================================

                    status.update(
                        label="✅ Answer Synthesized",
                        state="complete",
                        expanded=False,
                    )


                    logfire.info(
                        "✅ Backend response received.",
                        request_id=request_id,
                    )


                # =================================================
                # CONNECTION ERROR
                # =================================================

                except requests.exceptions.ConnectionError as e:

                    request_error = str(e)

                    logfire.error(
                        f"❌ Backend connection failed: "
                        f"{request_error}"
                    )

                    status.update(
                        label="❌ Backend connection failed",
                        state="error",
                        expanded=True,
                    )

                    answer = (
                        "I couldn't connect to the backend. "
                        "Please make sure the FastAPI backend "
                        "is running on port 8000."
                    )


                # =================================================
                # TIMEOUT ERROR
                # =================================================

                except requests.exceptions.Timeout as e:

                    request_error = str(e)

                    logfire.error(
                        f"⏱️ Backend request timed out: "
                        f"{request_error}"
                    )

                    status.update(
                        label="⏱️ Backend request timed out",
                        state="error",
                        expanded=True,
                    )

                    answer = (
                        "The backend request timed out. "
                        "Please try again in a moment."
                    )


                # =================================================
                # HTTP ERROR
                # =================================================

                except requests.exceptions.HTTPError as e:

                    request_error = str(e)

                    logfire.error(
                        f"❌ Backend HTTP error: "
                        f"{request_error}"
                    )

                    status.update(
                        label="❌ Backend returned an error",
                        state="error",
                        expanded=True,
                    )

                    answer = (
                        "The backend returned an error while "
                        "processing your request."
                    )


                # =================================================
                # INVALID RESPONSE
                # =================================================

                except ValueError as e:

                    request_error = str(e)

                    logfire.error(
                        f"❌ Invalid backend response: "
                        f"{request_error}"
                    )

                    status.update(
                        label="❌ Invalid backend response",
                        state="error",
                        expanded=True,
                    )

                    answer = (
                        "The backend returned an invalid response."
                    )


                # =================================================
                # GENERAL REQUEST ERROR
                # =================================================

                except requests.exceptions.RequestException as e:

                    request_error = str(e)

                    logfire.error(
                        f"❌ Backend request failed: "
                        f"{request_error}"
                    )

                    status.update(
                        label="❌ Backend request failed",
                        state="error",
                        expanded=True,
                    )

                    answer = (
                        "The request to the backend failed. "
                        "Please try again."
                    )


                # =================================================
                # UNEXPECTED ERROR
                # =================================================

                except Exception as e:

                    request_error = str(e)

                    logfire.error(
                        f"❌ Unexpected UI error: "
                        f"{request_error}"
                    )

                    status.update(
                        label="❌ Unexpected error",
                        state="error",
                        expanded=True,
                    )

                    answer = (
                        "An unexpected error occurred while "
                        "processing your request."
                    )


            # =====================================================
            # FINAL ANSWER
            #
            # IMPORTANT:
            # This is OUTSIDE st.status().
            # =====================================================

            if answer:

                st.markdown(answer)


            # =====================================================
            # SHOW SOURCES
            #
            # Sources are also OUTSIDE st.status().
            # =====================================================

            if sources:

                with st.expander(
                    "📄 View Retrieved Context (Sources)"
                ):

                    for i, source in enumerate(
                        sources
                    ):

                        preview = (
                            source[:100]
                            .replace("\n", " ")
                            + "..."
                        )

                        with st.expander(
                            f"Chunk {i + 1}: {preview}"
                        ):

                            st.info(source)


            # =====================================================
            # SAVE ASSISTANT MESSAGE
            # =====================================================

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "request_id": request_id,
                }
            )


            logfire.info(
                "✅ Chat cycle completed successfully.",
                request_id=request_id,
            )


    # =========================================================
    # CLEAR REQUEST STATE
    #
    # This prevents duplicate processing when Streamlit
    # reruns the application.
    # =========================================================

    st.session_state.pending_prompt = None

    st.session_state.processing_request = False

    st.session_state.last_request_id = None