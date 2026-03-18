import os
from langchain_community.document_loaders import PyPDFLoader


def load_pdfs(folder_path: str):

    documents = []

    for file in os.listdir(folder_path):

        if file.endswith(".pdf"):

            pdf_path = os.path.join(folder_path, file)

            loader = PyPDFLoader(pdf_path)

            pages = loader.load()

            for page in pages:
                page.metadata["source"] = file
                page.metadata["page"] = page.metadata.get("page", 0) + 1

            documents.extend(pages)

    return documents


if __name__ == "__main__":

    folder = "data/papers"

    docs = load_pdfs(folder)

    print(f"Loaded {len(docs)} pages from PDFs")
