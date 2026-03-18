from ingestion.pdf_loader import load_pdfs
from processing.chunking import chunk_documents
from processing.embeddings import create_embeddings

from retrieval.faiss_index import FAISSIndex
from retrieval.bm25_index import BM25Index


def main():

    print("Loading PDFs...")
    documents = load_pdfs("data/papers")

    print("Chunking documents...")
    chunks = chunk_documents(documents)

    print("Creating embeddings...")
    embeddings = create_embeddings(chunks)

    print("Building FAISS index...")
    dimension = len(embeddings[0])
    faiss_index = FAISSIndex(dimension)
    faiss_index.build_index(embeddings, chunks)
    faiss_index.save("faiss_offline.index", "chunks_offline.pkl")

    print("Building BM25 index...")
    bm25_index = BM25Index(chunks)
    bm25_index.save("bm25_offline.pkl")

    print("Indexes built successfully!")


if __name__ == "__main__":
    main()
