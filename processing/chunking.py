from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def chunk_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    cleaned_chunks = []

    for chunk in chunks:
        metadata = getattr(chunk, "metadata", {}) or {}

        source = metadata.get("source", "unknown")
        page = metadata.get("page", "?")

        cleaned_chunks.append(
            Document(
                page_content=chunk.page_content,
                metadata={
                    "source": source,
                    "page": page
                }
            )
        )

    return cleaned_chunks
