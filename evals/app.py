# ─────────────────────────────────────────────────────────────────────────────
# CRITICAL: logfire must be configured before all other imports
# ─────────────────────────────────────────────────────────────────────────────
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import logfire
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"), service_name="evals")

# ─────────────────────────────────────────────────────────────────────────────
import asyncio
import json
import nest_asyncio
import pandas as pd
import streamlit as st

nest_asyncio.apply()

from evals.pipeline import run_pipeline, load_golden_dataset
from evals.guardrails_eval import run_guardrails_eval, compute_guardrails_metrics
from evals.metrics import run_all_metrics

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Enterprise RAG — Eval Suite",
    page_icon="🧪",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
SCORE_COLORS = {
    "excellent": "#15803d",  # 0.90–1.00
    "good": "#22c55e",       # 0.75–0.89
    "fair": "#f59e0b",       # 0.50–0.74
    "poor": "#ea580c",       # 0.25–0.49
    "very_poor": "#b91c1c",  # 0.00–0.24
}


def _badge(score: float) -> str:
    if score >= 0.75:
        return "🟢"
    elif score >= 0.5:
        return "🟡"
    return "🔴"


def _grade(score: float) -> str:
    if score >= 0.75:
        return "✅ Good"
    elif score >= 0.5:
        return "⚠️ Fair"
    return "❌ Poor"


def _color_score(val):
    try:
        score = float(val)
    except (TypeError, ValueError):
        return ""

    if score >= 0.90:
        bg = SCORE_COLORS["excellent"]
        fg = "#ffffff"
    elif score >= 0.75:
        bg = SCORE_COLORS["good"]
        fg = "#ffffff"
    elif score >= 0.50:
        bg = SCORE_COLORS["fair"]
        fg = "#111827"
    elif score >= 0.25:
        bg = SCORE_COLORS["poor"]
        fg = "#ffffff"
    else:
        bg = SCORE_COLORS["very_poor"]
        fg = "#ffffff"

    return (
        f"background-color: {bg}; "
        f"color: {fg}; "
        "font-weight: 700; "
        "border-radius: 4px;"
    )

def _render_metric_table(
    df: pd.DataFrame,
    metric_col: str,
    title: str,
):
    avg = df[metric_col].mean()

    st.markdown(
        f"**{title}** — AVG: {_badge(avg)} `{avg:.2f}` {_grade(avg)}"
    )

    styled = (
        df.style
        .applymap(_color_score, subset=[metric_col])
        .format({metric_col: "{:.3f}"})
    )

    st.dataframe(
        styled,
        width="stretch",
        hide_index=True,
    )


def _run_async(coro):
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coro)


# ─────────────────────────────────────────────────────────────────────────────
# Persistent evaluation state
# ─────────────────────────────────────────────────────────────────────────────
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_state.json")


def _save_eval_state():
    """Persist pipeline/guardrail/metric state so Streamlit restarts do not lose work."""
    state = {
        "pipeline_done": bool(st.session_state.get("pipeline_done", False)),
        "enriched_dataset": st.session_state.get("enriched_dataset"),
        "guardrails_results": st.session_state.get("guardrails_results"),
        "metric_results": {
            key: df.to_dict(orient="records")
            for key, df in (st.session_state.get("metric_results") or {}).items()
            if isinstance(df, pd.DataFrame)
        },
    }
    tmp_file = STATE_FILE + ".tmp"
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp_file, STATE_FILE)


def _load_eval_state():
    """Restore previously persisted evaluation state, if available."""
    if not os.path.exists(STATE_FILE):
        return

    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            state = json.load(f)

        enriched = state.get("enriched_dataset")
        if enriched:
            st.session_state.enriched_dataset = enriched
            st.session_state.pipeline_done = bool(state.get("pipeline_done", False))

        if state.get("guardrails_results") is not None:
            st.session_state.guardrails_results = state["guardrails_results"]

        metric_results = {}
        for key, rows in (state.get("metric_results") or {}).items():
            metric_results[key] = pd.DataFrame(rows)
        if metric_results:
            st.session_state.metric_results = metric_results

    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        st.warning(f"⚠️ Could not restore saved evaluation state: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Session state init
# ─────────────────────────────────────────────────────────────────────────────
if "golden" not in st.session_state:
    st.session_state.golden = load_golden_dataset()

if "pipeline_done" not in st.session_state:
    st.session_state.pipeline_done = False

if "enriched_dataset" not in st.session_state:
    st.session_state.enriched_dataset = None

if "guardrails_results" not in st.session_state:
    st.session_state.guardrails_results = None

if "metric_results" not in st.session_state:
    st.session_state.metric_results = None

if "pipeline_rows" not in st.session_state:
    st.session_state.pipeline_rows = []

# Restore persisted state after defaults exist.
if not st.session_state.get("_eval_state_loaded", False):
    _load_eval_state()
    st.session_state._eval_state_loaded = True


golden = st.session_state.golden

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.title("🧪 Enterprise RAG — Evaluation Suite")

st.caption(
    "Step 1: Review ground truth → Step 2: Run live pipeline → Step 3: Score with RAGAS"
)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(
    [
        "📋 Step 1 — Ground Truth",
        "🚀 Step 2 — Live Pipeline",
        "📊 Step 3 — Eval Metrics",
    ]
)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — Ground Truth
# ═════════════════════════════════════════════════════════════════════════════
with tab1:

    st.subheader("Ground Truth Dataset")

    st.markdown(
        "These are the **golden Q&A pairs** built by parsing your real enterprise documents. "
        "Each entry has a question, a reference answer (ground truth), and the expected tool the RAG agent should call."
    )

    rag_rows = []

    for s in golden["rag_samples"]:
        rag_rows.append(
            {
                "ID": s["id"],
                "Domain": s["domain"].replace("_", " ").title(),
                "Question": s["question"],
                "Reference Answer": (
                    s["reference"][:120] + "..."
                    if len(s["reference"]) > 120
                    else s["reference"]
                ),
                "Expected Tool": (
                    s["expected_tools"][0]
                    if s["expected_tools"]
                    else "—"
                ),
            }
        )

    df_golden = pd.DataFrame(rag_rows)

    st.dataframe(
        df_golden,
        width="stretch",
        hide_index=True,
    )

    st.caption(
        f"✅ {len(rag_rows)} golden RAG samples from 5 enterprise docs"
    )

    st.divider()

    st.subheader("Guardrails Test Cases")

    st.markdown(
        "These inputs test whether the safety rails correctly **block adversarial inputs** "
        "and **let through legitimate questions**."
    )

    g_rows = []

    for g in golden["guardrails_samples"]:

        expected_label = (
            "🛡️ Block"
            if g["expected_blocked"]
            else "✅ Pass"
        )

        g_rows.append(
            {
                "ID": g["id"],
                "Input": g["input"],
                "Expected": expected_label,
                "Type": g["type"],
                "Description": g["description"],
            }
        )

    st.dataframe(
        pd.DataFrame(g_rows),
        width="stretch",
        hide_index=True,
    )

    st.caption(
        "6 guardrails test cases: 3 adversarial (should block) + 3 legit (should pass)"
    )

    with st.expander("View raw golden_dataset.json"):
        st.json(golden)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — Live Pipeline
# ═════════════════════════════════════════════════════════════════════════════
with tab2:

    st.subheader("Live Pipeline — Collect Real Responses")

    st.markdown(
        "Sends each golden question to your **running FastAPI app** (`localhost:8000/query`). "
        "Captures the actual response, retrieved contexts, and tool called. "
        "Responses are truncated to 300 chars to save tokens for the RAGAS judging step."
    )

    st.info(
        "⚠️ Make sure your FastAPI backend is running first: "
        "`uvicorn app.main:app --reload --port 8000`",
        icon="⚠️",
    )

    col_p1, col_p2, col_p3 = st.columns([1, 1, 2])

    run_pipeline_btn = col_p1.button(
        "▶️ Run Live Pipeline",
        type="primary",
        width="stretch",
        disabled=st.session_state.pipeline_done,
    )

    reset_btn = col_p2.button(
        "🔄 Reset & Re-run",
        width="stretch",
        disabled=not st.session_state.pipeline_done,
    )

    if reset_btn:

        st.session_state.pipeline_done = False
        st.session_state.enriched_dataset = None
        st.session_state.guardrails_results = None
        st.session_state.metric_results = None
        st.session_state.pipeline_rows = []

        if os.path.exists(STATE_FILE):
            try:
                os.remove(STATE_FILE)
            except OSError:
                pass

        st.rerun()

    if run_pipeline_btn:

        st.session_state.pipeline_rows = []

        progress_bar = st.progress(
            0,
            text="Starting pipeline...",
        )

        live_table_slot = st.empty()
        status_slot = st.empty()

        def pipeline_cb(
            i,
            total,
            question,
            stage,
            response="",
        ):

            pct = int((i / total) * 100)

            if stage == "calling":

                progress_bar.progress(
                    pct,
                    text=(
                        f"[{i + 1}/{total}] "
                        f"Calling /query: {question[:60]}..."
                    ),
                )

                return

            # Only the final "done" stage creates a table row.
            # Intermediate stages such as "waiting" are ignored.
            if stage != "done":
                return

            short_q = (
                question[:55] + "..."
                if len(question) > 55
                else question
            )

            short_r = (
                response[:80] + "..."
                if len(response) > 80
                else response
            )

            st.session_state.pipeline_rows.append(
                {
                    "#": i + 1,
                    "Question": short_q,
                    "Live Response (truncated)": (
                        short_r
                        if short_r
                        else "⚠️ No response"
                    ),
                    "Status": (
                        "✅"
                        if short_r
                        else "❌"
                    ),
                }
            )

            live_table_slot.dataframe(
                pd.DataFrame(
                    st.session_state.pipeline_rows
                ),
                width="stretch",
                hide_index=True,
            )

            progress_bar.progress(
                int(((i + 1) / total) * 100),
                text=f"[{i + 1}/{total}] ✅ Done",
            )

        with logfire.span(
            "🚀 Streamlit — Run Pipeline Button"
        ):

            enriched = run_pipeline(
                golden,
                progress_callback=pipeline_cb,
            )

            st.session_state.enriched_dataset = enriched

        progress_bar.progress(
            100,
            text="✅ All responses collected!",
        )

        status_slot.success(
            f"💾 {len(enriched['rag_samples'])} responses stored in session."
        )

        # ── Guardrails tests ──────────────────────────────────────────────────

        st.divider()

        st.subheader("Guardrails Tests")

        g_progress = st.progress(
            0,
            text="Running guardrails tests...",
        )

        g_status_slot = st.empty()

        def g_cb(
            i,
            total,
            input_text,
        ):

            g_progress.progress(
                int((i / total) * 100),
                text=(
                    f"[{i + 1}/{total}] "
                    f"Testing: {input_text[:60]}..."
                ),
            )

        with logfire.span(
            "🛡️ Streamlit — Guardrails Tests"
        ):

            g_results = run_guardrails_eval(
                enriched["guardrails_samples"],
                progress_callback=g_cb,
            )

            g_metrics = compute_guardrails_metrics(
                g_results
            )

            st.session_state.guardrails_results = g_results
            st.session_state.pipeline_done = True
            _save_eval_state()

        g_progress.progress(
            100,
            text="✅ Guardrails tests complete!",
        )

        g_rows_live = []

        for r in g_results:

            result_label = {
                "TP": "🛡️ Blocked ✅",
                "TN": "✅ Passed ✅",
                "FP": "🛡️ Blocked ❌ (False Positive)",
                "FN": "✅ Passed ❌ (Missed)",
            }.get(
                r["result"],
                r["result"],
            )

            g_rows_live.append(
                {
                    "ID": r["id"],
                    "Input": r["input"][:70],
                    "Expected": (
                        "🛡️ Block"
                        if r["expected_blocked"]
                        else "✅ Pass"
                    ),
                    "Actual": (
                        "Blocked"
                        if r["actual_blocked"]
                        else "Passed"
                    ),
                    "Result": result_label,
                }
            )

        st.dataframe(
            pd.DataFrame(g_rows_live),
            width="stretch",
            hide_index=True,
        )

        mc1, mc2, mc3, mc4 = st.columns(4)

        mc1.metric(
            "Correct",
            f"{g_metrics['correct']}/{g_metrics['total']}",
        )

        mc2.metric(
            "Precision",
            f"{g_metrics['precision']:.2f}",
        )

        mc3.metric(
            "Recall",
            f"{g_metrics['recall']:.2f}",
        )

        mc4.metric(
            "Accuracy",
            f"{g_metrics['accuracy']:.2f}",
        )

    elif st.session_state.pipeline_done:

        st.success(
            "✅ Pipeline already run. See results below."
        )

        resp_rows = []

        for s in st.session_state.enriched_dataset["rag_samples"]:

            resp_rows.append(
                {
                    "#": s["id"],
                    "Domain": s["domain"].replace(
                        "_",
                        " "
                    ).title(),
                    "Question": s["question"][:60],
                    "Live Response": (
                        s["actual_response"][:100] + "..."
                        if len(
                            s.get(
                                "actual_response",
                                "",
                            )
                        ) > 100
                        else s.get(
                            "actual_response",
                            "",
                        )
                    ),
                    "Tool Called": (
                        s["actual_tools_called"][0]
                        if s.get("actual_tools_called")
                        else "—"
                    ),
                    "Contexts Retrieved": len(
                        s.get(
                            "actual_contexts",
                            [],
                        )
                    ),
                }
            )

        st.dataframe(
            pd.DataFrame(resp_rows),
            width="stretch",
            hide_index=True,
        )

        if st.session_state.guardrails_results:

            st.divider()

            st.subheader(
                "Guardrails Results (from previous run)"
            )

            g_rows_prev = []

            for r in st.session_state.guardrails_results:

                result_label = {
                    "TP": "🛡️ Blocked ✅",
                    "TN": "✅ Passed ✅",
                    "FP": "Blocked ❌ FP",
                    "FN": "Passed ❌ FN",
                }.get(
                    r["result"],
                    r["result"],
                )

                g_rows_prev.append(
                    {
                        "ID": r["id"],
                        "Input": r["input"][:70],
                        "Result": result_label,
                    }
                )

            st.dataframe(
                pd.DataFrame(g_rows_prev),
                width="stretch",
                hide_index=True,
            )

            gm = compute_guardrails_metrics(
                st.session_state.guardrails_results
            )

            mc1, mc2, mc3, mc4 = st.columns(4)

            mc1.metric(
                "Correct",
                f"{gm['correct']}/{gm['total']}",
            )

            mc2.metric(
                "Precision",
                f"{gm['precision']:.2f}",
            )

            mc3.metric(
                "Recall",
                f"{gm['recall']:.2f}",
            )

            mc4.metric(
                "Accuracy",
                f"{gm['accuracy']:.2f}",
            )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — Eval Metrics
# ═════════════════════════════════════════════════════════════════════════════
with tab3:

    st.subheader("📊 Step 3 — RAGAS Evaluation")

    # Fixed order keeps the UI easy to follow even when the run resumes.
    metric_display_names = {
        "faithfulness": "Faithfulness",
        "answer_relevancy": "Answer Relevancy",
        "context_precision": "Context Precision",
        "context_recall": "Context Recall",
        "answer_correctness": "Answer Correctness",
        "tool_correctness": "Tool Correctness",
    }

    experiment_titles = {
        key: f"Exp {idx} — {name}"
        for idx, (key, name) in enumerate(metric_display_names.items(), start=1)
    }

    metric_results = st.session_state.metric_results or {}
    completed_keys = [key for key in metric_display_names if key in metric_results]
    completed_count = len(completed_keys)
    total_metrics = len(metric_display_names)

    if not st.session_state.pipeline_done:

        st.warning(
            "⚠️ Complete Step 2 (Live Pipeline) first to collect responses."
        )

    else:

        # ── Overview ─────────────────────────────────────────────────────────
        if completed_count == total_metrics:
            st.success(
                "🎉 Evaluation complete — all 6 metric experiments are saved."
            )
        elif completed_count > 0:
            remaining = total_metrics - completed_count
            next_key = next(
                key for key in metric_display_names if key not in metric_results
            )
            st.info(
                f"💾 Saved evaluation found: **{completed_count}/{total_metrics}** metrics complete. "
                f"The next run will resume from **{experiment_titles[next_key]}**. "
                f"Completed metrics will be skipped."
            )
        else:
            st.info(
                "▶️ Ready to evaluate the 15 live responses with 6 metric experiments. "
                "Progress is checkpointed after every completed metric."
            )

        # ── Progress ─────────────────────────────────────────────────────────
        progress_value = completed_count / total_metrics
        st.progress(
            progress_value,
            text=f"Metric progress: {completed_count}/{total_metrics} experiments complete",
        )

        # ── Experiment status cards ──────────────────────────────────────────
        st.markdown("### Experiment Status")

        status_cols = st.columns(3)
        for idx, (key, name) in enumerate(metric_display_names.items()):
            col = status_cols[idx % 3]
            with col:
                with st.container(border=True):
                    if key in metric_results:
                        st.markdown(f"### ✅ Exp {idx + 1}")
                        st.markdown(f"**{name}**")
                        df = metric_results[key]
                        if isinstance(df, pd.DataFrame) and key in df.columns and not df.empty:
                            avg = float(df[key].mean())
                            st.metric("Average", f"{avg:.3f}")
                        st.caption("Saved ✓ — will be skipped on resume")
                    else:
                        st.markdown(f"### ⏳ Exp {idx + 1}")
                        st.markdown(f"**{name}**")
                        st.caption("Waiting")

        st.divider()

        # ── What this step does ──────────────────────────────────────────────
        with st.expander("ℹ️ What happens in Step 3?", expanded=False):
            st.markdown(
                """
**15 live questions** are scored across **6 experiments**:

1. **Faithfulness** — Is the answer supported by the retrieved context?
2. **Answer Relevancy** — Does the answer directly address the question?
3. **Context Precision** — Are the retrieved contexts relevant?
4. **Context Recall** — Did retrieval contain the information needed for the answer?
5. **Answer Correctness** — How well does the answer match the reference answer?
6. **Tool Correctness** — Did the agent use the expected tool? *(local check; no LLM call)*

The first five use the RAGAS judge. Samples are processed one at a time with cooldowns.
"""
            )

        st.warning(
            "⏱️ This is intentionally slow to protect the Groq rate limit. "
            "Do not click the button repeatedly. If the process stops, restart Streamlit and use the saved checkpoint to continue.",
            icon="⏱️",
        )

        with st.expander("⚙️ Evaluation configuration", expanded=False):
            c1, c2, c3 = st.columns(3)
            c1.metric("Questions", "15")
            c2.metric("Experiments", "6")
            c3.metric("LLM metrics", "5")
            st.caption(
                "Judge: JUDGE_GROQ • Model: GPT-OSS-120B • Cooldown: 40s between samples • "
                "Tool Correctness uses no LLM"
            )

        # ── Run / Resume button ──────────────────────────────────────────────
        if completed_count == total_metrics:
            run_metrics_btn = st.button(
                "✅ All Metrics Complete",
                type="primary",
                width="stretch",
                disabled=True,
            )
        else:
            button_label = (
                "▶️ Resume Evaluation"
                if completed_count > 0
                else "▶️ Start Evaluation"
            )
            run_metrics_btn = st.button(
                button_label,
                type="primary",
                width="stretch",
                disabled=False,
            )

        if run_metrics_btn:

            status_slot = st.empty()
            current_slot = st.empty()
            live_progress = st.progress(
                progress_value,
                text=f"Starting — {completed_count}/{total_metrics} experiments already saved",
            )

            results_slots = {}
            for key in metric_display_names:
                results_slots[key] = st.empty()

            def status_cb(msg: str):
                status_slot.info(msg)

                # Make the currently running experiment obvious without
                # changing the underlying evaluation logic.
                current_text = msg
                for key, name in metric_display_names.items():
                    if name.lower() in msg.lower() or key.replace("_", " ").lower() in msg.lower():
                        current_text = f"🔄 **{experiment_titles[key]}**\n\n{msg}"
                        break
                current_slot.markdown(current_text)

            saved_results = st.session_state.metric_results or {}

            def checkpoint_cb(key, df):
                if st.session_state.metric_results is None:
                    st.session_state.metric_results = {}

                st.session_state.metric_results[key] = df
                _save_eval_state()

                done_now = len(
                    [k for k in metric_display_names if k in st.session_state.metric_results]
                )
                live_progress.progress(
                    done_now / total_metrics,
                    text=f"Saved {done_now}/{total_metrics} experiments — {experiment_titles[key]} complete",
                )

                if key in results_slots:
                    with results_slots[key].container():
                        _render_metric_table(
                            df,
                            key,
                            experiment_titles[key],
                        )

            try:
                with logfire.span("📊 Streamlit — Run Metrics Button"):
                    metric_results = _run_async(
                        run_all_metrics(
                            st.session_state.enriched_dataset,
                            status_cb=status_cb,
                            resume_results=saved_results,
                            checkpoint_cb=checkpoint_cb,
                        )
                    )

                st.session_state.metric_results = metric_results
                _save_eval_state()

                live_progress.progress(
                    1.0,
                    text="All 6/6 experiments complete and saved",
                )
                current_slot.empty()
                status_slot.success("🎉 All 6 experiments complete and checkpointed!")

            except Exception as exc:
                # Completed experiments are already checkpointed. The next run
                # will skip them and resume from the failed experiment.
                _save_eval_state()
                status_slot.error(
                    f"❌ Evaluation stopped: {type(exc).__name__}: {exc}"
                )
                st.info(
                    "💾 Your completed metric checkpoints are safe. Restart Streamlit if needed, "
                    "then click **Resume Evaluation**."
                )
                with st.expander("🔎 Technical error details", expanded=False):
                    st.exception(exc)

        # ── Saved results ─────────────────────────────────────────────────────
        saved_results = st.session_state.metric_results or {}

        if saved_results and not run_metrics_btn:
            st.markdown("### 📈 Saved Results")
            st.caption(
                "These results are loaded from the evaluation checkpoint and remain available after restarting Streamlit."
            )

            for key, name in metric_display_names.items():
                if key in saved_results:
                    _render_metric_table(
                        saved_results[key],
                        key,
                        experiment_titles[key],
                    )

        # ── Final Summary ─────────────────────────────────────────────────────
        if saved_results:

            st.divider()
            st.subheader("🏁 Final Summary")

            mr = saved_results

            summary = [
                (
                    "Faithfulness",
                    mr.get("faithfulness", pd.DataFrame()).get(
                        "faithfulness", pd.Series(dtype=float)
                    ).mean(),
                ),
                (
                    "Answer Relevancy",
                    mr.get("answer_relevancy", pd.DataFrame()).get(
                        "answer_relevancy", pd.Series(dtype=float)
                    ).mean(),
                ),
                (
                    "Context Precision",
                    mr.get("context_precision", pd.DataFrame()).get(
                        "context_precision", pd.Series(dtype=float)
                    ).mean(),
                ),
                (
                    "Context Recall",
                    mr.get("context_recall", pd.DataFrame()).get(
                        "context_recall", pd.Series(dtype=float)
                    ).mean(),
                ),
                (
                    "Answer Correctness",
                    mr.get("answer_correctness", pd.DataFrame()).get(
                        "answer_correctness", pd.Series(dtype=float)
                    ).mean(),
                ),
                (
                    "Tool Correctness",
                    mr.get("tool_correctness", pd.DataFrame()).get(
                        "tool_correctness", pd.Series(dtype=float)
                    ).mean(),
                ),
            ]

            summary_cols = st.columns(len(summary))
            for col, (name, score) in zip(summary_cols, summary):
                if pd.notna(score):
                    col.metric(
                        label=name,
                        value=f"{score:.2f}",
                        delta=_grade(score),
                    )
                else:
                    col.metric(label=name, value="—")

            if st.session_state.guardrails_results:
                gm = compute_guardrails_metrics(
                    st.session_state.guardrails_results
                )

                st.markdown("### 🛡️ Guardrails Summary")
                g1, g2, g3, g4 = st.columns(4)
                g1.metric("Correct", f"{gm['correct']}/{gm['total']}")
                g2.metric("Precision", f"{gm['precision']:.2f}")
                g3.metric("Recall", f"{gm['recall']:.2f}")
                g4.metric("Accuracy", f"{gm['accuracy']:.2f}")

            summary_df = pd.DataFrame(
                [
                    {
                        "Metric": name,
                        "Score": f"{score:.3f}" if pd.notna(score) else "—",
                        "Grade": _grade(score) if pd.notna(score) else "—",
                    }
                    for name, score in summary
                ]
            )

            st.dataframe(
                summary_df,
                width="stretch",
                hide_index=True,
            )
