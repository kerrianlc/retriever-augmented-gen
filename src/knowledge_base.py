from langchain.docstore.document import Document as LangchainDocument
from datasets import load_dataset


def knowledge_base(knowledge_base: str):
    ds = load_dataset(knowledge_base, split="train")
    return [
        LangchainDocument(page_content=doc["text"], metadata={"source": doc["source"]})
        for doc in ds
    ]
