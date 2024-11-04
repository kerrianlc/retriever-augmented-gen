import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from src.text_splitter import EMBEDDING_MODEL_NAME, split_documents
from src.reader import reader_llm
from src.prompt import rag_prompt_template
from src.knowledge_base import knowledge_base, KNOWLEDGE_BASE
import time

pd.set_option("display.max_colwidth", None)


def main(user_query: str):
    start = time.time()
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        multi_process=True,
        model_kwargs={"device": "cuda"},
        encode_kwargs={
            "normalize_embeddings": True
        },  # Set `True` for cosine similarity
    )
    raw_knowledge_base = knowledge_base(KNOWLEDGE_BASE)
    docs_processed = split_documents(
        512,  # We choose a chunk size adapted to our model
        raw_knowledge_base,
        tokenizer_name=EMBEDDING_MODEL_NAME,
    )
    knowledge_vect_db = FAISS.from_documents(
        docs_processed, embedding_model, distance_strategy=DistanceStrategy.COSINE
    )

    reader_llm, read_tokenizer = reader_llm()
    rag_prompt_template = rag_prompt_template(read_tokenizer)
    retrieved_docs = knowledge_vect_db.similarity_search(query=user_query, k=5)

    retrieved_docs_text = [
        doc.page_content for doc in retrieved_docs
    ]  # We only need the text of the documents
    context = "\nExtracted documents:\n"
    context += "".join(
        [f"Document {str(i)}:::\n" + doc for i, doc in enumerate(retrieved_docs_text)]
    )

    final_prompt = rag_prompt_template.format(question=user_query, context=context)

    # Redact an answer
    answer = reader_llm(final_prompt)[0]["generated_text"]
    print(answer)
    end = time.time()
    print(f"Time elapsed: {end - start} secs")


if __name__ == "__main__":
    main(user_query="what is type I error?")
