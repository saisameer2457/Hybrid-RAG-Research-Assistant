from rank_bm25 import BM25Okapi
import pickle
import os


class BM25Index:

    def __init__(self, chunks=None):
        self.chunks = chunks if chunks is not None else []
        self.bm25 = None

        if self.chunks:
            self._build_index()

    def _build_index(self):

        texts = []

        for chunk in self.chunks:
            if hasattr(chunk, "page_content"):
                texts.append(chunk.page_content)
            else:

                texts.append(str(chunk))

        tokenized_chunks = [text.split() for text in texts]
        self.bm25 = BM25Okapi(tokenized_chunks)

    def search(self, query, top_k=5):
        if not self.bm25 or len(self.chunks) == 0:
            return []

        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        results = []

        for i in ranked_indices:
            chunk = self.chunks[i]

            if hasattr(chunk, "metadata"):
                results.append(chunk)

        return results

    def add_documents(self, new_chunks):

        valid_chunks = [c for c in new_chunks if hasattr(c, "metadata")]
        self.chunks.extend(valid_chunks)
        self._build_index()

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.chunks, f)

    @classmethod
    def load(cls, path):

        if os.path.exists(path):
            with open(path, "rb") as f:
                chunks = pickle.load(f)

            return cls(chunks)

        else:
            return cls([])
