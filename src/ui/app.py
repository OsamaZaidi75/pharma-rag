"""Streamlit demo UI. Run: streamlit run src/ui/app.py"""
import os

import httpx
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Pharma RAG", page_icon="💊")
st.title("💊 Pharma Label Q&A")
st.caption("Grounded answers from official FDA drug labels (DailyMed).")

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Retrieval candidates", 5, 50, 20)
    top_n = st.slider("Sources in answer", 1, 10, 5)
    rerank = st.checkbox("Cross-encoder rerank", value=True)
    try:
        health = httpx.get(f"{API_URL}/health", timeout=5).json()
        st.success(f"API ok · {health['indexed_chunks']} chunks indexed")
        drugs = httpx.get(f"{API_URL}/drugs", timeout=10).json().get("drugs", [])
        if drugs:
            st.write("Indexed drugs:", ", ".join(drugs))
    except Exception as e:
        st.error(f"API unreachable at {API_URL}: {e}")

question = st.text_input(
    "Ask about a drug label",
    placeholder="What are the contraindications of atorvastatin?",
)

if st.button("Ask") and question.strip():
    with st.spinner("Searching labels and generating answer..."):
        try:
            resp = httpx.post(
                f"{API_URL}/ask",
                json={
                    "question": question,
                    "top_k": top_k,
                    "top_n": top_n,
                    "rerank": rerank,
                },
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            st.error(f"Request failed: {e}")
            st.stop()

    if data.get("indexed_drug"):
        st.info(f"✨ Drug '{data['indexed_drug']}' was indexed on-demand from DailyMed.")

    st.subheader("Answer")
    st.write(data["answer"])
    st.caption(f"Model: {data['model']}")

    st.subheader("Sources")
    for src in data["sources"]:
        with st.expander(f"[{src['ref']}] {src['drug_name']} — {src['section_title']}"):
            st.write(src["preview"] + "…")
            st.caption(f"retrieval score: {src['score']}")
