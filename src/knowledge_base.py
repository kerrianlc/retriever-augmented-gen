from langchain.docstore.document import Document as LangchainDocument
from datasets import load_dataset


def knowledge_base(knowledge_base: str):
    ds = load_dataset(knowledge_base, split="train")
    return [
        LangchainDocument(page_content=doc["Abstracts"], metadata={"Titles": doc["Titles"]})
        for doc in ds
    ]
