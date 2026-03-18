import streamlit as st
import os
import hashlib
from typing import List

from retrieval.faiss_index import FAISSIndex
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.bm25_index import BM25Index
from processing.embeddings import embed_query, create_embeddings
from processing.chunking import chunk_documents
from langchain_community.document_loaders import PyPDFLoader

from rag.qa_pipeline import QAPipeline
from rag.llm import llm
from retrieval.combined_retriever import CombinedRetriever
from retrieval.reranker import Reranker

st.set_page_config(
    page_title="Hybrid RAG Research Assistant",
    page_icon="🧠",
    layout="centered"
)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

OFFLINE_INDEX_PATH = os.path.join(BASE_DIR, "indexes", "faiss_offline.index")
OFFLINE_CHUNKS_PATH = os.path.join(BASE_DIR, "indexes", "chunks_offline.pkl")

ONLINE_INDEX_PATH = os.path.join(BASE_DIR, "indexes", "faiss_online.index")
ONLINE_CHUNKS_PATH = os.path.join(BASE_DIR, "indexes", "chunks_online.pkl")

UPLOAD_FOLDER = os.path.join(BASE_DIR, "data", "uploaded")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "file_hashes" not in st.session_state:
    st.session_state.file_hashes = {}


@st.cache_resource
def load_pipeline():
    offline_index = FAISSIndex.load(
        OFFLINE_INDEX_PATH,
        OFFLINE_CHUNKS_PATH,
        dimension=384
    )

    online_index = FAISSIndex.load(
        ONLINE_INDEX_PATH,
        ONLINE_CHUNKS_PATH,
        dimension=384
    )

    bm25_offline = BM25Index(offline_index.text_chunks)
    bm25_online = BM25Index(online_index.text_chunks)

    offline_retriever = HybridRetriever(
        offline_index,
        bm25_offline,
        embed_query
    )

    online_retriever = HybridRetriever(
        online_index,
        bm25_online,
        embed_query
    )

    retriever = CombinedRetriever(
        offline_retriever,
        online_retriever
    )

    qa_pipeline = QAPipeline(
        retriever=retriever,
        reranker=Reranker(),
        llm=llm
    )

    return qa_pipeline, online_index


qa_pipeline, online_index = load_pipeline()

st.markdown("""
<style>
[data-testid="stChatInput"] {
    border-radius: 12px !important;
    background-color: #1f2937 !important;
    border: 1px solid #374151 !important;
}
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([8, 1])

with col1:
    st.markdown("## 🧠 RAG Assistant")
    st.caption("Chat with your documents using AI")

with col2:
    if st.button("🧹"):
        st.session_state.messages = []
        st.rerun()

st.subheader("📄 Upload PDFs")

uploaded_files = st.file_uploader(
    "Choose PDFs",
    type="pdf",
    accept_multiple_files=True
)


def rebuild_online_index_with_overwrite(new_chunks: List):
    existing_chunks = getattr(online_index, "text_chunks", []) or []

    new_sources = {
        c.metadata.get("source")
        for c in new_chunks
        if hasattr(c, "metadata")
    }

    filtered_existing = [
        c for c in existing_chunks
        if not (
            hasattr(c, "metadata") and
            c.metadata.get("source") in new_sources
        )
    ]

    combined_chunks = filtered_existing + new_chunks

    if not combined_chunks:
        return

    embeddings = create_embeddings(combined_chunks)
    online_index.build_index(embeddings, combined_chunks)
    online_index.save(ONLINE_INDEX_PATH, ONLINE_CHUNKS_PATH)


if uploaded_files:
    progress = st.progress(0)
    all_new_chunks = []

    for i, uploaded_file in enumerate(uploaded_files):

        file_bytes = uploaded_file.read()
        file_hash = hashlib.md5(file_bytes).hexdigest()

        if uploaded_file.name in st.session_state.file_hashes:
            if st.session_state.file_hashes[uploaded_file.name] == file_hash:
                st.info(f"⚠️ {uploaded_file.name} already processed")
                progress.progress((i + 1) / len(uploaded_files))
                continue

        file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)

        with open(file_path, "wb") as f:
            f.write(file_bytes)

        loader = PyPDFLoader(file_path)
        pages = loader.load()

        for page in pages:
            page.metadata["source"] = uploaded_file.name
            page.metadata["page"] = page.metadata.get("page", 0) + 1

        chunks = chunk_documents(pages)
        all_new_chunks.extend(chunks)

        st.session_state.file_hashes[uploaded_file.name] = file_hash

        st.success(f"✅ {uploaded_file.name} processed")
        progress.progress((i + 1) / len(uploaded_files))

    if all_new_chunks:
        with st.spinner("🔄 Indexing documents..."):
            rebuild_online_index_with_overwrite(all_new_chunks)
            st.success("🚀 Index updated!")

st.subheader("💬 Chat with your documents")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

prompt = st.chat_input("Ask something...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = qa_pipeline.run(prompt)

            answer = result["answer"]
            sources = result.get("source_chunks", []) or []

            st.write(answer)

            if sources:
                st.markdown("### 📚 Sources")

                unique_sources = []
                seen = set()

                for src in sources:
                    if hasattr(src, "metadata"):
                        s = src.metadata.get("source", "Unknown")
                        p = src.metadata.get("page", "?")

                        key = f"{s}-{p}"
                        if key not in seen:
                            seen.add(key)
                            unique_sources.append((s, p))

                for s, p in unique_sources[:3]:
                    st.markdown(f"- 📄 **{s}** (Page {p})")

                if len(unique_sources) > 3:
                    with st.expander("See more sources"):
                        for s, p in unique_sources[3:]:
                            st.markdown(f"- 📄 **{s}** (Page {p})")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })
