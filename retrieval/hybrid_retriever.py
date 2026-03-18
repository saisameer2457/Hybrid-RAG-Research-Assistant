class HybridRetriever:

    def __init__(self, faiss_index, bm25_index, embed_query,
                 semantic_weight=2.0, keyword_weight=0.8):

        self.faiss_index = faiss_index
        self.bm25_index = bm25_index
        self.embed_query = embed_query
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight

    def retrieve(self, query, retrieval_k=40):

        query_embedding = self.embed_query(query)

        semantic_results = self.faiss_index.search(
            query_embedding, retrieval_k)

        keyword_results = self.bm25_index.search(query, retrieval_k)

        scores = {}
        documents = {}

        def get_key(doc):
            return id(doc)

        def is_valid(doc):
            return hasattr(doc, "metadata")

        for rank, doc in enumerate(semantic_results):
            if not is_valid(doc):
                continue

            key = get_key(doc)
            scores[key] = scores.get(
                key, 0.0) + self.semantic_weight * (retrieval_k - rank)
            documents[key] = doc

        for rank, doc in enumerate(keyword_results):
            if not is_valid(doc):
                continue

            key = get_key(doc)
            scores[key] = scores.get(key, 0.0) + \
                self.keyword_weight * (retrieval_k - rank)
            documents[key] = doc

        ranked_keys = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        return [documents[k] for k, _ in ranked_keys[:retrieval_k]]
