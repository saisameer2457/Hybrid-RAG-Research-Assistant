import faiss
import numpy as np
import pickle
import os

from processing.embeddings import embed_query


class FAISSIndex:

    def __init__(self, dimension):

        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.text_chunks = []

    def _ensure_index(self):

        if not hasattr(self, "index") or self.index.d != self.dimension:
            self.index = faiss.IndexFlatIP(self.dimension)

    def build_index(self, embeddings, chunks):

        self._ensure_index()

        self.index = faiss.IndexFlatIP(self.dimension)

        vectors = np.asarray(embeddings, dtype="float32")
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension mismatch. expected {self.dimension}, got {vectors.shape[1]}"
            )

        self.index.add(vectors)
        self.text_chunks = list(chunks)

    def add_documents(self, embeddings, chunks):

        self._ensure_index()

        vectors = np.asarray(embeddings, dtype="float32")
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension mismatch. expected {self.dimension}, got {vectors.shape[1]}"
            )

        self.index.add(vectors)
        self.text_chunks.extend(chunks)

    def search(self, query_or_embedding, top_k=5):

        if len(self.text_chunks) == 0 or self.index.ntotal == 0:
            return []

        if isinstance(query_or_embedding, (str, np.str_,)):
            query_embedding = embed_query(str(query_or_embedding))
        else:
            query_embedding = query_or_embedding

        try:
            query_vector = np.asarray(query_embedding, dtype="float32")
        except Exception as e:
            raise ValueError(
                f"Could not convert query embedding to float array: {e}")

        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        if query_vector.shape[1] != self.dimension:
            raise ValueError(
                f"Query embedding dimension mismatch. expected {self.dimension}, got {query_vector.shape[1]}"
            )

        top_k = min(top_k, max(1, self.index.ntotal))

        distances, indices = self.index.search(query_vector, top_k)

        results = []
        seen = set()

        for idx in indices[0]:
            if idx < 0 or idx >= len(self.text_chunks):
                continue

            chunk = self.text_chunks[idx]

            if hasattr(chunk, "page_content"):
                key = chunk.page_content
            else:
                key = str(chunk)

            if key not in seen:
                seen.add(key)
                results.append(chunk)

        return results

    def save(self, index_path, metadata_path):

        faiss.write_index(self.index, index_path)

        with open(metadata_path, "wb") as f:
            pickle.dump(self.text_chunks, f)

    @classmethod
    def load(cls, index_path, metadata_path, dimension=768):

        if os.path.exists(index_path) and os.path.exists(metadata_path):
            index = faiss.read_index(index_path)

            with open(metadata_path, "rb") as f:
                chunks = pickle.load(f)

            obj = cls(index.d)
            obj.index = index
            obj.text_chunks = chunks
        else:
            obj = cls(dimension)

        return obj
