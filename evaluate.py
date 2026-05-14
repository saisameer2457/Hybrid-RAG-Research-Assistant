import re
import pandas as pd

from rag.qa_pipeline import QAPipeline

from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import Reranker

from retrieval.faiss_index import FAISSIndex
from retrieval.bm25_index import BM25Index

from processing.embeddings import embed_query

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


def parse_qa_file(filepath):

    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    pattern = r"Question\s+\d+:\s*(.*?)\s*Ground Truth:\s*(.*?)(?=Question\s+\d+:|$)"

    matches = re.findall(pattern, text, re.DOTALL)

    dataset = []

    for q, a in matches:
        dataset.append({"question": q.strip(), "ground_truth": a.strip()})

    return dataset


qa_dataset = parse_qa_file("evaluation_text.txt")

print(f"Loaded {len(qa_dataset)} questions")


faiss_index = FAISSIndex.load(
    "indexes/faiss_offline.index", "indexes/chunks_offline.pkl")

bm25_index = BM25Index.load("indexes/bm25_offline.pkl")


class BM25Retriever:

    def __init__(self, bm25):
        self.bm25 = bm25

    def retrieve(self, query, retrieval_k=10):
        return self.bm25.search(query, retrieval_k)


class FAISSRetriever:

    def __init__(self, faiss):
        self.faiss = faiss

    def retrieve(self, query, retrieval_k=10):
        return self.faiss.search(query, retrieval_k)


bm25_retriever = BM25Retriever(bm25_index)

faiss_retriever = FAISSRetriever(faiss_index)

hybrid_retriever = HybridRetriever(
    faiss_index=faiss_index, bm25_index=bm25_index, embed_query=embed_query)

reranker = Reranker()

experiments = {
    "BM25 Only": {"retriever": bm25_retriever, "reranker": None},

    "FAISS Only": {"retriever": faiss_retriever, "reranker": None},

    "Hybrid": {"retriever": hybrid_retriever, "reranker": None},

    "Hybrid + Reranker": {"retriever": hybrid_retriever, "reranker": reranker}
}


def get_chunk_text(chunk):

    if hasattr(chunk, "page_content"):
        return chunk.page_content.lower()

    return str(chunk).lower()


embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")


def compute_metrics(retrieved_chunks, ground_truth):

    hitrate_at_5 = 0
    hitrate_at_10 = 0

    reciprocal_rank = 0

    gt_embedding = embedding_model.encode(ground_truth)

    similarities = []

    for idx, chunk in enumerate(retrieved_chunks):
        if hasattr(chunk, "page_content"):
            chunk_text = chunk.page_content
        else:
            chunk_text = str(chunk)

        chunk_embedding = embedding_model.encode(chunk_text)

        similarity = cosine_similarity([gt_embedding], [chunk_embedding])[0][0]

        similarities.append((idx, similarity))

    if len(similarities) == 0:
        return {"hitrate@5": 0, "hitrate@10": 0, "mrr": 0}

    threshold = 0.70

    relevant_ranks = []

    for idx, similarity in similarities:

        if similarity >= threshold:
            relevant_ranks.append(idx)

    if len(relevant_ranks) == 0:
        return {"hitrate@5": 0, "hitrate@10": 0, "mrr": 0}

    best_rank = min(relevant_ranks)

    if best_rank < 5:
        hitrate_at_5 = 1

    if best_rank < 10:
        hitrate_at_10 = 1

    reciprocal_rank = 1 / (best_rank + 1)

    return {
        "hitrate@5": hitrate_at_5, "hitrate@10": hitrate_at_10, "mrr": reciprocal_rank
    }


final_scores = []

for experiment_name, config in experiments.items():

    print("\n===================================")
    print(f"RUNNING: {experiment_name}")
    print("===================================\n")

    qa_pipeline = QAPipeline(
        retriever=config["retriever"], reranker=config["reranker"], llm=None)

    metrics_list = []

    for idx, item in enumerate(qa_dataset):

        question = item["question"]
        ground_truth = item["ground_truth"]

        print(f"[{idx+1}/{len(qa_dataset)}] {question}")

        try:
            result = qa_pipeline.run(question, generate_answer=False)

            retrieved_chunks = result["source_chunks"]

            metrics = compute_metrics(retrieved_chunks, ground_truth)

            metrics_list.append(metrics)

        except Exception as e:

            print(f"ERROR: {e}")

    df = pd.DataFrame(metrics_list)

    if df.empty:
        print(f"No successful evaluations for {experiment_name}")
        continue

    final_scores.append({
        "Experiment": experiment_name, "HitRate@5": df["hitrate@5"].mean(),
        "HitRate@10": df["hitrate@10"].mean(), "MRR": df["mrr"].mean()
    })

results_df = pd.DataFrame(final_scores)

print("\n===================================")
print("FINAL RESULTS")
print("===================================\n")

print(results_df)

results_df.to_csv("retrieval_evaluation_results.csv", index=False)

print("\nSaved results to retrieval_evaluation_results.csv")
