from sentence_transformers import CrossEncoder


class Reranker:

    def __init__(self, model_name="BAAI/bge-reranker-base"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query, documents, top_k=10):

        pairs = []

        for doc in documents:
            text = doc.page_content if hasattr(
                doc, "page_content") else str(doc)
            pairs.append((query, text))

        scores = self.model.predict(pairs)

        scored_docs = list(zip(documents, scores))

        ranked = sorted(scored_docs, key=lambda x: x[1], reverse=True)

        return [doc for doc, _ in ranked[:top_k]]
