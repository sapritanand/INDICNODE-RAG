"""
Streamlit UI for the Indicnode RAG Assistant.

Usage:
    streamlit run ui/streamlit_app.py

Requires the FastAPI backend to be running at http://localhost:8000.
Set the backend URL with the BACKEND_URL environment variable if different.
"""

from __future__ import annotations

import os
import time

import httpx
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Indicnode RAG Assistant",
    page_icon="🤖",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
st.title("🤖 Indicnode RAG Assistant")
st.caption("Ask questions about AI, LLMs, RAG, embeddings, prompt engineering, and evaluation.")

# ---------------------------------------------------------------------------
# Sidebar – health & metrics
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("System Status")

    if st.button("Refresh Status"):
        st.rerun()

    try:
        health = httpx.get(f"{BACKEND_URL}/health", timeout=5).json()
        status = health.get("status", "unknown")
        colour = "🟢" if status == "ok" else "🟡"
        st.write(f"{colour} **{status.upper()}**")
        st.write(f"Chunks indexed: **{health.get('chunks_indexed', 0)}**")
        st.write(f"Documents: **{health.get('documents_indexed', 0)}**")
        st.write(f"Index ready: **{health.get('index_ready', False)}**")
    except Exception:
        st.error("Backend unreachable. Is the server running?")

    st.divider()
    st.header("Metrics")
    try:
        metrics = httpx.get(f"{BACKEND_URL}/metrics", timeout=5).json()
        st.metric("Total queries", metrics.get("total_queries", 0))
        st.metric("Rejected queries", metrics.get("rejected_queries", 0))
        st.metric("Cache hits", metrics.get("cache_hits", 0))
        st.metric(
            "Avg latency",
            f"{metrics.get('avg_response_time_ms', 0):.0f} ms",
        )
    except Exception:
        st.write("Metrics unavailable.")

    st.divider()
    st.caption("Example questions:")
    examples = [
        "What is retrieval-augmented generation?",
        "How does FAISS work?",
        "Explain chain-of-thought prompting.",
        "What is BM25 and how does it differ from dense retrieval?",
        "What are the limitations of LLMs?",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{ex[:20]}"):
            st.session_state["prefill"] = ex

# ---------------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
prefill = st.session_state.pop("prefill", "")
user_input = st.chat_input("Ask a question …") or prefill

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating …"):
            try:
                t0 = time.time()
                resp = httpx.post(
                    f"{BACKEND_URL}/query",
                    json={"question": user_input, "top_k": 3},
                    timeout=60,
                )
                resp.raise_for_status()
                data = resp.json()
            except httpx.HTTPStatusError as e:
                st.error(f"HTTP {e.response.status_code}: {e.response.text}")
                st.stop()
            except Exception as e:
                st.error(f"Request failed: {e}")
                st.stop()

        # Rejected query
        if data.get("rejected"):
            reason = data.get("reason", "unknown")
            message = data.get("message", "")
            icon = "⚠️" if reason == "out_of_scope" else "🚫"
            colour = "#fff3cd" if reason == "out_of_scope" else "#f8d7da"
            st.markdown(
                f"""<div style="background:{colour};padding:12px;border-radius:6px;">
                {icon} <b>Query rejected</b> ({reason})<br>{message}
                </div>""",
                unsafe_allow_html=True,
            )
            assistant_content = f"{icon} **Rejected ({reason}):** {message}"

        # Successful response
        else:
            answer = data.get("answer", "")
            sources = data.get("sources", [])
            response_time = data.get("response_time_ms", 0)
            cached = data.get("cached", False)
            tokens = data.get("tokens_used")

            st.markdown(answer)

            # Metadata row
            routing_tier = data.get("routing_tier")
            model_used = data.get("model_used")
            used_fallback = data.get("used_fallback", False)

            meta_parts = [f"⏱ {response_time:.0f} ms"]
            if cached:
                meta_parts.append("⚡ cached")
            if tokens:
                meta_parts.append(f"🔢 {tokens} tokens")
            if routing_tier:
                tier_label = "fast tier" if routing_tier == "fast" else "smart tier"
                meta_parts.append(f"🔀 {tier_label}")
            if model_used:
                short_model = model_used.split("/")[-1]
                meta_parts.append(f"🤖 {short_model}")
            if used_fallback:
                meta_parts.append("⚠️ fallback")
            st.caption("  ·  ".join(meta_parts))

            # Sources expander
            if sources:
                with st.expander(f"📚 Sources ({len(sources)})"):
                    for i, src in enumerate(sources, 1):
                        st.markdown(
                            f"**[{i}] {src['source']}** (score: {src['score']:.3f})"
                        )
                        st.markdown(
                            f"> {src['content'][:250]}{'…' if len(src['content']) > 250 else ''}"
                        )

            assistant_content = answer

    st.session_state.messages.append(
        {"role": "assistant", "content": assistant_content}
    )
