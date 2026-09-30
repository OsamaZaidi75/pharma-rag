"""Streamlit demo UI: Ask (single Q&A) · Chat (with memory) · Agent (ReAct).

Two ways to run:
- Standalone (default): the RAG pipeline runs in-process via local_backend.
  This is what Streamlit Community Cloud uses — no API server needed.
- API mode: set API_URL (env var or Streamlit secret) to point at a running
  FastAPI backend, e.g. http://localhost:8000 for local dev.

Run: streamlit run src/ui/app.py
"""
import os

import httpx
import streamlit as st

API_URL = os.environ.get("API_URL", "").strip()
try:
    API_URL = API_URL or st.secrets.get("API_URL", "").strip()
except Exception:
    pass
USE_API = bool(API_URL)

if USE_API:

    def get_health() -> dict:
        return httpx.get(f"{API_URL}/health", timeout=5).json()

    def get_drugs() -> list:
        return httpx.get(f"{API_URL}/drugs", timeout=10).json().get("drugs", [])

    def post_ask(payload: dict, timeout: int) -> dict:
        resp = httpx.post(f"{API_URL}/ask", json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    def post_ask_agent(payload: dict, timeout: int) -> dict:
        resp = httpx.post(f"{API_URL}/ask-agent", json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()

else:
    import local_backend

    def get_health() -> dict:
        return local_backend.get_health()

    def get_drugs() -> list:
        return local_backend.get_drugs()["drugs"]

    def post_ask(payload: dict, timeout: int) -> dict:
        return local_backend.ask(**payload)

    def post_ask_agent(payload: dict, timeout: int) -> dict:
        return local_backend.ask_agent(**payload)


st.set_page_config(page_title="Pharma RAG", page_icon="💊")
st.title("💊 Pharma Label Q&A")
st.caption("Grounded answers from official FDA drug labels (DailyMed).")

mode = st.sidebar.radio(
    "Mode",
    ["Ask", "Chat", "Agent"],
    help=(
        "Ask: single question, fixed RAG pipeline. "
        "Chat: remembers the conversation, resolves follow-ups. "
        "Agent: the model plans its own tool calls (ReAct)."
    ),
)

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Retrieval candidates", 5, 50, 20)
    top_n = st.slider("Sources in answer", 1, 10, 5)
    rerank = st.checkbox("Cross-encoder rerank", value=True)
    max_iter = st.slider("Agent max iterations", 1, 15, 8)
    try:
        health = get_health()
        label = "API" if USE_API else "Standalone"
        st.success(f"{label} ok · {health['indexed_chunks']} chunks indexed")
        drugs = get_drugs()
        if drugs:
            st.write("Indexed drugs:", ", ".join(drugs))
    except Exception as e:
        where = f"API at {API_URL}" if USE_API else "pipeline"
        st.error(f"{where} unreachable: {e}")


def _render_sources(data):
    st.subheader("Sources")
    for src in data["sources"]:
        with st.expander(f"[{src['ref']}] {src['drug_name']} — {src['section_title']}"):
            st.write(src["preview"] + "…")
            st.caption(f"retrieval score: {src['score']}")


# ---------------------------------------------------------------- Ask mode
if mode == "Ask":
    question = st.text_input(
        "Ask about a drug label",
        placeholder="What are the contraindications of atorvastatin?",
    )
    if st.button("Ask") and question.strip():
        with st.spinner("Searching labels and generating answer..."):
            try:
                data = post_ask(
                    {"question": question, "top_k": top_k, "top_n": top_n, "rerank": rerank},
                    timeout=120,
                )
            except Exception as e:
                st.error(f"Request failed: {e}")
                st.stop()

        if data.get("indexed_drug"):
            st.info(f"✨ Drug '{data['indexed_drug']}' was indexed on-demand from DailyMed.")
        st.subheader("Answer")
        st.write(data["answer"])
        st.caption(f"Model: {data['model']}")
        _render_sources(data)


# ---------------------------------------------------------------- Chat mode
elif mode == "Chat":
    st.caption("Chat remembers the conversation — try a follow-up like “what about its dosage?”.")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if st.sidebar.button("Clear chat"):
        st.session_state.chat_history = []
        st.rerun()

    for m in st.session_state.chat_history:
        with st.chat_message(m["role"]):
            st.write(m["content"])

    if q := st.chat_input("Ask about a drug label"):
        st.session_state.chat_history.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.write(q)

        # Send prior turns only (not the just-asked question), bounded.
        history = st.session_state.chat_history[:-1][-10:]
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    data = post_ask(
                        {
                            "question": q,
                            "history": history,
                            "top_k": top_k,
                            "top_n": top_n,
                            "rerank": rerank,
                        },
                        timeout=180,
                    )
                except Exception as e:
                    st.error(f"Request failed: {e}")
                    st.stop()

            if data.get("rewritten_question"):
                st.caption(f"Understood as: “{data['rewritten_question']}”")
            if data.get("indexed_drug"):
                st.info(f"✨ Drug '{data['indexed_drug']}' was indexed on-demand from DailyMed.")
            st.write(data["answer"])
            st.caption(f"Model: {data['model']}")
            with st.expander("Sources"):
                for src in data["sources"]:
                    st.write(f"[{src['ref']}] {src['drug_name']} — {src['section_title']}")

        st.session_state.chat_history.append({"role": "assistant", "content": data["answer"]})


# ---------------------------------------------------------------- Agent mode
else:
    st.caption("The agent plans its own tool calls — watch the trace to see it think.")
    if "agent_history" not in st.session_state:
        st.session_state.agent_history = []
    if st.sidebar.button("Clear agent chat"):
        st.session_state.agent_history = []
        st.rerun()

    for m in st.session_state.agent_history:
        with st.chat_message(m["role"]):
            st.write(m["content"])

    if q := st.chat_input("Ask the agent (e.g. compare two drugs)"):
        st.session_state.agent_history.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.write(q)

        with st.chat_message("assistant"):
            with st.spinner("Agent is working..."):
                try:
                    data = post_ask_agent(
                        {"question": q, "max_iterations": max_iter},
                        timeout=600,
                    )
                except Exception as e:
                    st.error(f"Request failed: {e}")
                    st.stop()

            with st.expander(f"🔍 Agent trace ({data['iterations']} steps)", expanded=True):
                for step in data["trace"]:
                    st.markdown(f"**Step {step['iteration']}**")
                    if step.get("thought"):
                        st.write(f"💭 {step['thought'][:400]}")
                    if step.get("action"):
                        st.code(f"{step['action']}({step['args']})")
                        st.write(f"👁 {step['observation'][:600]}")
                    st.divider()
            if data.get("indexed_drugs"):
                st.caption(f"Drugs used: {', '.join(data['indexed_drugs'])}")
            st.write(data["answer"])
            st.caption(f"Model: {data['model']}")
            with st.expander("Sources"):
                for src in data["sources"]:
                    st.write(f"[{src['ref']}] {src['drug_name']} — {src['section_title']}")

        st.session_state.agent_history.append({"role": "assistant", "content": data["answer"]})
