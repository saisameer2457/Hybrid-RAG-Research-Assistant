# 🧠 Hybrid RAG Research Assistant

Ask questions over research papers and PDFs using **hybrid retrieval (BM25 + FAISS)** and LLMs.

---

## 📌 Overview

This project is a **Retrieval-Augmented Generation (RAG)** system that enables users to query research papers using natural language.

It combines:

* **Keyword search (BM25)** for exact matches
* **Semantic search (FAISS)** for contextual understanding
* **Cross-encoder reranking** to improve relevance

The system reduces hallucinations by grounding responses in actual document content and provides **source citations with page numbers**.

---

## 🧠 High-Level Architecture

```
Docs → Chunking → Embeddings → Hybrid Retriever → Reranker → LLM
```

**Explanation:**

* Documents are split into chunks and converted into embeddings
* Hybrid retrieval combines keyword + semantic search
* Reranker improves final relevance
* LLM generates grounded answers

---

## 🔬 Detailed Architecture

```
                PDF Documents
                       ↓
                Text Cleaning
                       ↓
                    Chunking
                       ↓
        ┌──────────────┴──────────────┐
        │                             │
 Offline Processing           Online Processing
 (papers folder)              (user uploads)
        │                             │
        ↓                             ↓
 Embeddings (HF)              Embeddings (HF)
        │                             │
 FAISS Index                  FAISS Index
 BM25 Index                   BM25 Index
        │                             │
 Hybrid Retriever            Hybrid Retriever
 (FAISS + BM25)              (FAISS + BM25)
        │                             │
        └──────────┬──────────────────┘
                   ↓
          Combined Retriever
        (merge both sources)
                   ↓
           Cross Encoder
            (reranking)
                   ↓
               LLM Answer
```

**Key idea:**

* Each data source (offline + uploaded) performs **independent hybrid retrieval**
* Results are then **merged and globally reranked**

---

## ✨ Features

* 🔍 Hybrid Retrieval (**BM25 + FAISS**)
* 📚 Multi-source retrieval (**offline + user uploads**)
* ⚡ Cross-encoder reranking for better relevance
* 🧠 LLM-based answer generation
* 📄 Source citation with page numbers
* 🚀 Interactive UI using Streamlit
* 🔄 Incremental indexing for uploaded documents

---

## 📂 Project Structure

```
Hybrid RAG Research Assistant/

├── data/
│   ├── papers/          # Offline research papers
│   └── uploaded/        # User uploaded PDFs
│
├── indexes/
│   ├── faiss_offline.index
│   ├── faiss_online.index
│   ├── chunks_offline.pkl
│   ├── chunks_online.pkl
│   └── bm25_offline.pkl
│
├── ingestion/
│   ├── pdf_loader.py
│   └── text_cleaning.py
│
├── processing/
│   ├── chunking.py
│   └── embeddings.py
│
├── retrieval/
│   ├── faiss_index.py
│   ├── bm25_index.py
│   ├── hybrid_retriever.py
│   ├── combined_retriever.py
│   └── reranker.py
│
├── rag/
│   ├── prompt.py
│   ├── qa_pipeline.py
│   └── llm.py
│
├── build_index.py
└── streamlit_app.py
```

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd rag-research-assistant
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the application

```bash
python -m streamlit run main.py
```

---

## 🧪 Example Usage

**User Query:**

```
What is the attention mechanism in transformers?
```

**Output:**

* Generated answer based on retrieved chunks
* Sources:

  * 📄 paper1.pdf (Page 3)
  * 📄 paper2.pdf (Page 7)

---

## 🧠 Key Design Decisions

* Used **hybrid retrieval** to balance precision (BM25) and semantic understanding (FAISS)
* Separated **offline and online indexes** for scalability and dynamic updates
* Applied **cross-encoder reranking** to improve final answer quality
* Stored FAISS indexes to enable fast query-time retrieval
* Designed a **modular pipeline** for extensibility

---
