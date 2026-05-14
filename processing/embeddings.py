from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("BAAI/bge-small-en-v1.5")


def create_embeddings(chunks):

    texts = []

    for chunk in chunks:
        if hasattr(chunk, "page_content"):
            texts.append(chunk.page_content)
        else:
            texts.append(str(chunk))

    embeddings = model.encode(texts, normalize_embeddings=True)

    embeddings = np.asarray(embeddings, dtype="float32")
    return embeddings


def embed_query(query):

    query_text = f"query: {query}"

    embedding = model.encode([query_text], normalize_embeddings=True)[0]

    return np.asarray(embedding, dtype="float32")
