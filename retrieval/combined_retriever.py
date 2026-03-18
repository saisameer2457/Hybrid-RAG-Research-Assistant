class CombinedRetriever:
    def __init__(self, retriever_a, retriever_b):
        self.retriever_a = retriever_a
        self.retriever_b = retriever_b

    def retrieve(self, query, retrieval_k=40):
        k_half = retrieval_k // 2

        # 🔥 FIX HERE
        results_a = self.retriever_a.retrieve(query, k_half)
        results_b = self.retriever_b.retrieve(query, k_half)

        combined = results_a + results_b

        # remove duplicates
        seen = set()
        unique = []

        for doc in combined:
            key = doc.page_content if hasattr(
                doc, "page_content") else str(doc)

            if key not in seen:
                seen.add(key)
                unique.append(doc)

        return unique[:retrieval_k]
