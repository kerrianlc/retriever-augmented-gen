from langchain.docstore.document import Document as LangchainDocument
from datasets import load_dataset

ds = load_dataset("m-ric/huggingface_doc", split="train")
RAW_KNOWLEDGE_BASE = [
    LangchainDocument(page_content=doc["text"], metadata={"source": doc["source"]}) for doc in ds
]