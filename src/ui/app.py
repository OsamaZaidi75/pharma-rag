"""Streamlit demo UI: Ask (single Q&A) · Chat (with memory) · Agent (ReAct).

Run: streamlit run src/ui/app.py
"""
import os

import httpx
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

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
        health = httpx.get(f"{API_URL}/health", timeout=5).json()
        st.success(f"API ok · {health['indexed_chunks']} chunks indexed")
        drugs = httpx.get(f"{API_URL}/drugs", timeout=10).json().get("drugs", [])
        if drugs:
            st.write("Indexed drugs:", ", ".join(drugs))
    except Exception as e:
        st.error(f"API unreachable at {API_URL}: {e}")


def _render_sources(data):
    st.subheader("Sources")
    for src in data["sources"]:
        with st.expander(f"[{src['ref']}] {src['drug_name']} — {src['section_title']}"):
            st.write(src["preview"] + "…")
            st.caption(f"retrieval score: {src['score']}")


def _post(path, payload, timeout):
    resp = httpx.post(f"{API_URL}{path}", json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------- Ask mode
if mode == "Ask":
    question = st.text_input(
        "Ask about a drug label",
        placeholder="What are the contraindications of atorvastatin?",
    )
    if st.button("Ask") and question.strip():
        with st.spinner("Searching labels and generating answer..."):
            try:
                data = _post(
                    "/ask",
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
                    data = _post(
                        "/ask",
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
                    data = _post(
                        "/ask-agent",
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
