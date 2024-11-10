import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from src.splitter import recursive_chunker
from src.reader import reader_llm
from src.prompt.prompt import rag_prompt_template
from src.knowledge_base import knowledge_base
from src.common import DOC_EMBED_MODEL_NAME, KNOWLEDGE_BASE
import time

device = "cuda:0"

def main(user_query: str, neighbors=5):
    start = time.time()
    print("---- RETRIEVAL DOCS ----")
    # Retrieval of documents from dataset
    raw_knowledge_base = knowledge_base(KNOWLEDGE_BASE)
    # Split documents into more comprehensible chunks
    docs_processed = recursive_chunker(
        512,  # We choose a chunk size adapted to our model
        raw_knowledge_base,
        tokenizer_name=DOC_EMBED_MODEL_NAME,
    )
    # Embeds the chunks in a vector space 
    # Store embedded docs in a vectorstore, we will retrieve the adequate docs based on a similarity search
    end = time.time()
    print(f"Time elapsed: {end - start} secs")
    print("---- EMBED CHUNKS ----")
    embedding_model = HuggingFaceEmbeddings(
        model_name=DOC_EMBED_MODEL_NAME,
        multi_process=True,
        model_kwargs={"device": "cuda"},
        encode_kwargs={
            "normalize_embeddings": True
        },  # Set `True` for cosine similarity
    )
    knowledge_vect_db = FAISS.from_documents(
        docs_processed, embedding_model, distance_strategy=DistanceStrategy.COSINE
    )
    # Import the LLM that acts as a reader
    end = time.time()
    print(f"Time elapsed: {end - start} secs")
    print("---- READING DOCS ----")
    reader, read_tokenizer = reader_llm()
    rag_prompt = rag_prompt_template(read_tokenizer)

    # Based on the user query we find the k closest clusters in the vectorstore
    retrieved_docs = knowledge_vect_db.similarity_search(query=user_query, k=neighbors)

    retrieved_docs_text = [
        doc.page_content for doc in retrieved_docs
    ]  # We only need the text of the documents
    context = "\nExtracted documents:\n"
    context += "".join(
        [f"Document {str(i)}:::\n" + doc for i, doc in enumerate(retrieved_docs_text)]
    )

    final_prompt = rag_prompt.format(question=user_query, context=context)
    end = time.time()
    print(f"Time elapsed: {end - start} secs")
    print("---- GENERATION ----")
    # Redact an answer
    answer = reader(final_prompt)[0]["generated_text"]
    print(answer)
    end = time.time()
    print(f"Total time elapsed: {end - start} secs")


if __name__ == "__main__":
    main(user_query="how to backpropagate error in DL setting?")
