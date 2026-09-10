"""
Phase 2 — RAGAS + Tool Correctness metrics.

Uses JUDGE_GROQ key so production GROQ_API_KEY is never exhausted by eval runs.

All LLM-based metrics run one sample at a time with cooldowns between samples
and experiments — calibrated for Groq's on_demand tier.

RAGAS 0.3.1 compatibility:
- Uses llm_factory() for the Groq/OpenAI-compatible judge.
- Uses LangChain HuggingFaceEmbeddings.
- Wraps LangChain embeddings with LangchainEmbeddingsWrapper so RAGAS
  can use embed_text()/aembed_text().
"""

import os
import asyncio

from dotenv import load_dotenv

import logfire
import pandas as pd

from ragas.dataset_schema import SingleTurnSample
from ragas.llms import llm_factory
from ragas.embeddings import LangchainEmbeddingsWrapper

from langchain_huggingface import HuggingFaceEmbeddings

from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    AnswerCorrectness,
    AnswerSimilarity,
)


# ============================================================
# Configuration
# ============================================================

load_dotenv()

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
JUDGE_MODEL = "openai/gpt-oss-120b"

# Cooldown between experiments
COOLDOWN_STANDARD = 62

# Cooldown between individual samples
COOLDOWN_MINI = 40

# One sample at a time to control Groq TPM
GENERAL_BATCH_SIZE = 1

# Keep enough context for evaluation while controlling token usage
CONTEXT_TRUNCATE = 1200
CONTEXT_LIMIT = 5


# ============================================================
# Build RAGAS Judge
# ============================================================

def _build_judge():
    """
    Build the RAGAS LLM judge and embedding model.

    JUDGE_GROQ is preferred so evaluation traffic is separated
    from the production GROQ_API_KEY.
    """

    api_key = os.getenv("JUDGE_GROQ") or os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "JUDGE_GROQ or GROQ_API_KEY must be set."
        )

    # RAGAS 0.3.1 uses LangChain's ChatOpenAI internally.
    # Groq provides an OpenAI-compatible API.
    os.environ["OPENAI_API_KEY"] = api_key

    llm = llm_factory(
        model=JUDGE_MODEL,
        base_url=GROQ_BASE_URL,
    )

    # LangChain HuggingFace embeddings
    hf_embeddings = HuggingFaceEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
    )

    # IMPORTANT:
    # RAGAS 0.3.1 expects embed_text()/aembed_text().
    # LangChain embeddings expose embed_query()/embed_documents().
    # This wrapper bridges the two interfaces.
    embeddings = LangchainEmbeddingsWrapper(
        hf_embeddings
    )

    return llm, embeddings


# ============================================================
# Cooldown
# ============================================================

async def _cooldown(
    seconds: int,
    label: str,
    status_cb=None,
):
    """
    Wait between requests so the evaluation stays within
    Groq's TPM limits.
    """

    msg = (
        f"⏳ {seconds}s cooldown after "
        f"{label} (Groq TPM buffer)..."
    )

    if status_cb:
        status_cb(msg)

    # Sleep in 10-second chunks so the UI remains responsive.
    remaining = seconds

    while remaining > 0:
        sleep_for = min(10, remaining)
        await asyncio.sleep(sleep_for)
        remaining -= sleep_for

    if status_cb:
        status_cb(
            "✅ Ready — starting next experiment."
        )


# ============================================================
# Prepare Samples
# ============================================================

def _prep_samples(
    golden_dataset: dict,
) -> list:
    """
    Returns only samples with actual_response populated.

    Contexts are truncated and limited so individual RAGAS
    requests remain within the Groq token limit.
    """

    valid = []

    for s in golden_dataset["rag_samples"]:

        response = s.get(
            "actual_response",
            "",
        ).strip()

        # Skip samples for which Phase 1 did not produce an answer.
        if not response:
            continue

        raw_contexts = (
            s.get("actual_contexts")
            or s.get("relevant_contexts")
            or []
        )

        contexts = [
            c[:CONTEXT_TRUNCATE]
            for c in raw_contexts[:CONTEXT_LIMIT]
        ]

        valid.append(
            {
                **s,
                "actual_contexts": contexts,
            }
        )

    return valid


# ============================================================
# Score DataFrame
# ============================================================

def _score_df(
    metric_key: str,
    samples: list,
    scores,
) -> pd.DataFrame:
    """
    Convert RAGAS scores into a DataFrame.

    RAGAS 0.3.1 single_turn_ascore() returns float values.
    """

    return pd.DataFrame(
        [
            {
                "question": s["question"][:65],
                metric_key: round(
                    float(score),
                    3,
                ),
            }
            for s, score in zip(
                samples,
                scores,
            )
        ]
    )


# ============================================================
# RAGAS 0.3.1 Single Sample Scoring
# ============================================================

async def _batched_score(
    metric,
    inputs: list,
    samples: list,
    status_cb=None,
    label: str = "",
) -> list:
    """
    Runs RAGAS 0.3.1 scoring one sample at a time.

    RAGAS 0.3.1 uses single_turn_ascore().

    We intentionally process one sample at a time because the
    Groq on-demand tier has a relatively small TPM limit.
    """

    all_scores = []

    for idx, sample_input in enumerate(inputs):

        # Cooldown before every sample after the first.
        if idx > 0:
            await _cooldown(
                COOLDOWN_MINI,
                f"{label} sample {idx}",
                status_cb,
            )

        sample = SingleTurnSample(
            **sample_input
        )

        score = await metric.single_turn_ascore(
            sample
        )

        all_scores.append(score)

    return all_scores


# ============================================================
# Run All Metrics
# ============================================================

async def run_all_metrics(
    golden_dataset: dict,
    status_cb=None,
    resume_results=None,
    checkpoint_cb=None,
) -> dict:
    """
    Runs all 6 experiments and resumes from completed checkpoints.

    Parameters
    ----------
    golden_dataset:
        Evaluation dataset containing RAG samples.

    status_cb:
        Optional callback used to update the Streamlit UI.

    resume_results:
        Previously completed metric results.
        Example:
            {
                "faithfulness": DataFrame(...),
                "answer_relevancy": DataFrame(...),
            }

    checkpoint_cb:
        Callback called immediately after each experiment completes.

        This allows evals/app.py to persist the result to
        eval_state.json.

    Returns
    -------
    dict
        Metric name -> pandas DataFrame.
    """

    # --------------------------------------------------------
    # Build judge
    # --------------------------------------------------------

    judge_llm, ragas_embeddings = _build_judge()

    # --------------------------------------------------------
    # Prepare samples
    # --------------------------------------------------------

    samples = _prep_samples(
        golden_dataset
    )

    if not samples:
        raise ValueError(
            "No samples with actual_response found. "
            "Run Phase 1 first."
        )

    # --------------------------------------------------------
    # Restore previous checkpoints
    # --------------------------------------------------------

    results = dict(
        resume_results or {}
    )

    # --------------------------------------------------------
    # Checkpoint helper
    # --------------------------------------------------------

    def checkpoint(
        key,
        df,
    ):
        results[key] = df

        if checkpoint_cb:
            checkpoint_cb(
                key,
                df,
            )

    # ========================================================
    # Evaluation span
    # ========================================================

    with logfire.span(
        "🧪 Eval Phase 2 — All Metrics",
        total_samples=len(samples),
    ):

        # ====================================================
        # Exp 1 — Faithfulness
        # ====================================================

        if "faithfulness" in results:

            if status_cb:
                status_cb(
                    "⏭️ Exp 1/6 — Faithfulness "
                    "already saved. Skipping."
                )

        else:

            if status_cb:
                status_cb(
                    f"🧪 Exp 1/6 — Faithfulness "
                    f"({len(samples)} samples)..."
                )

            with logfire.span(
                "🧪 Exp 1 — Faithfulness"
            ):

                inputs = [
                    {
                        "user_input": s["question"],
                        "response": s["actual_response"],
                        "retrieved_contexts": s[
                            "actual_contexts"
                        ],
                    }
                    for s in samples
                ]

                scores = await _batched_score(
                    Faithfulness(
                        llm=judge_llm
                    ),
                    inputs,
                    samples,
                    status_cb,
                    "Faithfulness",
                )

                df = _score_df(
                    "faithfulness",
                    samples,
                    scores,
                )

                checkpoint(
                    "faithfulness",
                    df,
                )

                logfire.info(
                    "🧪 Faithfulness done",
                    avg=round(
                        df["faithfulness"].mean(),
                        3,
                    ),
                )

            await _cooldown(
                COOLDOWN_STANDARD,
                "Faithfulness",
                status_cb,
            )

        # ====================================================
        # Exp 2 — Answer Relevancy
        # ====================================================

        if "answer_relevancy" in results:

            if status_cb:
                status_cb(
                    "⏭️ Exp 2/6 — Answer Relevancy "
                    "already saved. Skipping."
                )

        else:

            if status_cb:
                status_cb(
                    f"🧪 Exp 2/6 — Answer Relevancy "
                    f"({len(samples)} samples)..."
                )

            with logfire.span(
                "🧪 Exp 2 — Answer Relevancy"
            ):

                inputs = [
                    {
                        "user_input": s["question"],
                        "response": s["actual_response"],
                    }
                    for s in samples
                ]

                scores = await _batched_score(
                    AnswerRelevancy(
                        llm=judge_llm,
                        embeddings=ragas_embeddings,
                    ),
                    inputs,
                    samples,
                    status_cb,
                    "Answer Relevancy",
                )

                df = _score_df(
                    "answer_relevancy",
                    samples,
                    scores,
                )

                checkpoint(
                    "answer_relevancy",
                    df,
                )

                logfire.info(
                    "🧪 Answer Relevancy done",
                    avg=round(
                        df["answer_relevancy"].mean(),
                        3,
                    ),
                )

            await _cooldown(
                COOLDOWN_STANDARD,
                "Answer Relevancy",
                status_cb,
            )

        # ====================================================
        # Exp 3 — Context Precision
        # ====================================================

        if "context_precision" in results:

            if status_cb:
                status_cb(
                    "⏭️ Exp 3/6 — Context Precision "
                    "already saved. Skipping."
                )

        else:

            if status_cb:
                status_cb(
                    f"🧪 Exp 3/6 — Context Precision "
                    f"({len(samples)} samples)..."
                )

            with logfire.span(
                "🧪 Exp 3 — Context Precision"
            ):

                inputs = [
                    {
                        "user_input": s["question"],
                        "reference": s["reference"],
                        "retrieved_contexts": s[
                            "actual_contexts"
                        ],
                    }
                    for s in samples
                ]

                scores = await _batched_score(
                    ContextPrecision(
                        llm=judge_llm
                    ),
                    inputs,
                    samples,
                    status_cb,
                    "Context Precision",
                )

                df = _score_df(
                    "context_precision",
                    samples,
                    scores,
                )

                checkpoint(
                    "context_precision",
                    df,
                )

                logfire.info(
                    "🧪 Context Precision done",
                    avg=round(
                        df["context_precision"].mean(),
                        3,
                    ),
                )

            await _cooldown(
                COOLDOWN_STANDARD,
                "Context Precision",
                status_cb,
            )

        # ====================================================
        # Exp 4 — Context Recall
        # ====================================================

        if "context_recall" in results:

            if status_cb:
                status_cb(
                    "⏭️ Exp 4/6 — Context Recall "
                    "already saved. Skipping."
                )

        else:

            if status_cb:
                status_cb(
                    f"🧪 Exp 4/6 — Context Recall "
                    f"({len(samples)} samples)..."
                )

            with logfire.span(
                "🧪 Exp 4 — Context Recall"
            ):

                inputs = [
                    {
                        "user_input": s["question"],
                        "reference": s["reference"],
                        "retrieved_contexts": s[
                            "actual_contexts"
                        ],
                    }
                    for s in samples
                ]

                scores = await _batched_score(
                    ContextRecall(
                        llm=judge_llm
                    ),
                    inputs,
                    samples,
                    status_cb,
                    "Context Recall",
                )

                df = _score_df(
                    "context_recall",
                    samples,
                    scores,
                )

                checkpoint(
                    "context_recall",
                    df,
                )

                logfire.info(
                    "🧪 Context Recall done",
                    avg=round(
                        df["context_recall"].mean(),
                        3,
                    ),
                )

            await _cooldown(
                COOLDOWN_STANDARD,
                "Context Recall",
                status_cb,
            )

        # ====================================================
        # Exp 5 — Answer Correctness
        # ====================================================

        if "answer_correctness" in results:

            if status_cb:
                status_cb(
                    "⏭️ Exp 5/6 — Answer Correctness "
                    "already saved. Skipping."
                )

        else:

            if status_cb:
                status_cb(
                    f"🧪 Exp 5/6 — Answer Correctness "
                    f"({len(samples)} samples)..."
                )

            with logfire.span(
                "🧪 Exp 5 — Answer Correctness"
            ):

                inputs = [
                    {
                        "user_input": s["question"],
                        "response": s["actual_response"],
                        "reference": s["reference"],
                    }
                    for s in samples
                ]

                scores = await _batched_score(
                    AnswerCorrectness(
                        llm=judge_llm,
                        embeddings=ragas_embeddings,
                        answer_similarity=AnswerSimilarity(
                            embeddings=ragas_embeddings
                        ),
                    ),
                    inputs,
                    samples,
                    status_cb,
                    "Answer Correctness",
                )

                df = _score_df(
                    "answer_correctness",
                    samples,
                    scores,
                )

                checkpoint(
                    "answer_correctness",
                    df,
                )

                logfire.info(
                    "🧪 Answer Correctness done",
                    avg=round(
                        df["answer_correctness"].mean(),
                        3,
                    ),
                )

            await _cooldown(
                COOLDOWN_STANDARD,
                "Answer Correctness",
                status_cb,
            )

        # ====================================================
        # Exp 6 — Tool Correctness
        # ====================================================

        if "tool_correctness" in results:

            if status_cb:
                status_cb(
                    "⏭️ Exp 6/6 — Tool Correctness "
                    "already saved. Skipping."
                )

        else:

            if status_cb:
                status_cb(
                    "⚡ Exp 6/6 — Tool Correctness "
                    "(zero LLM calls)..."
                )

            with logfire.span(
                "🧪 Exp 6 — Tool Correctness"
            ):

                tool_rows = []

                for s in samples:

                    called = set(
                        s.get(
                            "actual_tools_called"
                        )
                        or []
                    )

                    expected = set(
                        s.get(
                            "expected_tools"
                        )
                        or []
                    )

                    union = len(
                        called | expected
                    )

                    score = (
                        len(
                            called & expected
                        ) / union
                        if union > 0
                        else 0.0
                    )

                    tool_rows.append(
                        {
                            "question": s[
                                "question"
                            ][:65],
                            "tool_correctness": round(
                                score,
                                3,
                            ),
                        }
                    )

                df = pd.DataFrame(
                    tool_rows
                )

                checkpoint(
                    "tool_correctness",
                    df,
                )

                logfire.info(
                    "🧪 Tool Correctness done",
                    avg=round(
                        df[
                            "tool_correctness"
                        ].mean(),
                        3,
                    ),
                )

        # ====================================================
        # Complete
        # ====================================================

        if status_cb:
            status_cb(
                "✅ All 6 experiments complete!"
            )

    return results