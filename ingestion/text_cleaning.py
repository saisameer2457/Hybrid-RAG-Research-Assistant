import re


def clean_text(text: str) -> str:

    text = re.sub(r"\n+", " ", text)

    text = re.sub(r"\s+", " ", text)

    text = text.replace("\x00", "")

    return text.strip()


def clean_documents(documents):

    for doc in documents:
        doc.page_content = clean_text(doc.page_content)

    return documents
