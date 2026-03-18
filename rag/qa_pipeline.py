from rag.prompt import PROMPT_TEMPLATE
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity


class QAPipeline:

    def __init__(self, retriever, reranker, llm, max_context_chars=4000):
        self.retriever = retriever
        self.reranker = reranker
        self.llm = llm
        self.max_context_chars = max_context_chars

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    def compress_chunks(self, query, chunks, top_k=10):

        if not chunks:
            return []

        query_emb = self.embeddings.embed_query(query)

        texts = [
            chunk.page_content if hasattr(
                chunk, "page_content") else str(chunk)
            for chunk in chunks
        ]

        chunk_embs = self.embeddings.embed_documents(texts)

        scored = []

        for chunk, chunk_emb in zip(chunks, chunk_embs):
            score = cosine_similarity([query_emb], [chunk_emb])[0][0]
            scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)

        return [chunk for _, chunk in scored[:top_k]]

    def build_context(self, chunks):

        context_parts = []
        total_length = 0

        for chunk in chunks:

            text = chunk.page_content if hasattr(
                chunk, "page_content") else str(chunk)

            if hasattr(chunk, "metadata"):
                source = chunk.metadata.get("source", "unknown")
                page = chunk.metadata.get("page", "?")
            else:
                source = "unknown"
                page = "?"

            section = f"[Source: {source} | Page: {page}]\n{text}\n\n"

            if total_length + len(section) > self.max_context_chars:
                break

            context_parts.append(section)
            total_length += len(section)

        return "".join(context_parts)

    def format_sources(self, chunks):

        sources = {}

        for chunk in chunks:

            if hasattr(chunk, "metadata"):
                source = chunk.metadata.get("source")
                page = chunk.metadata.get("page")

                if source:
                    if source not in sources:
                        sources[source] = set()

                    if page:
                        sources[source].add(page)

        formatted_sources = []

        for source, pages in sources.items():

            pages = sorted(pages)

            if len(pages) == 1:
                formatted_sources.append(f"{source} (page {pages[0]})")
            else:
                pages_str = ", ".join(map(str, pages))
                formatted_sources.append(f"{source} (pages {pages_str})")

        return formatted_sources

    def run(self, question):

        chunks = self.retriever.retrieve(question)

        chunks = self.reranker.rerank(question, chunks, top_k=20)

        chunks = self.compress_chunks(question, chunks, top_k=10)

        context = self.build_context(chunks)

        prompt = PROMPT_TEMPLATE.format(
            context=context,
            question=question
        )

        answer = self.llm(prompt)

        formatted_sources = self.format_sources(chunks)

        return {
            "answer": answer,
            "sources": formatted_sources,
            "source_chunks": chunks
        }
