import argparse
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from llama_index.core import ServiceContext, VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SentenceTransformerRerank
from get_embedding_function import get_embedding_function
import pandas as pd
from IPython.display import HTML, display

CHROMA_PATH = "chroma"
PROMPT_TEMPLATE = """
Carefully read the following documents:
{documents}
---
Now, respond to the question based solely on the provided documents above: {question}
Instructions for Answering:
- Utilize ONLY the provided documents to formulate your response. Refrain from assumptions or external knowledge.
- If the documents directly support an answer, provide that answer clearly and concisely.
- If the documents do not contain enough information to answer the question, state: 'The answer cannot be determined from the provided documents.'
- If no documents are provided, respond with: 'No documents are available to answer the question.'
Please adhere to these guidelines rigorously.
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()

    # Setting up the ServiceContext
    service_context = ServiceContext(
        llm_predictor=None,
        prompt_helper=None,
        embed_model=None,
        transformations=None,
        llama_logger=None,
        callback_manager=None
    )

    # Initialize Chroma with an embedding function
    chroma = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())

    # Assume Chroma can generate a list of Node objects, or it is wrapped in such a way
    nodes = [chroma.get_node()]  # You may need to implement get_node() in Chroma or use an equivalent

    # Initializing VectorStoreIndex with a list of nodes
    vector_index = VectorStoreIndex(
        nodes=nodes,
        service_context=service_context
    )

    # Other components
    reranker = SentenceTransformerRerank(model="BAAI/bge-reranker-base", top_n=5)
    retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=10, service_context=service_context)

    reranked_results = process_query(args.query_text, vector_index, reranker)
    prompt = generate_prompt(reranked_results, args.query_text)
    response_text = generate_response(prompt)
    print(f"Response: {response_text}")


def process_query(query_text, vector_index, reranker):
    # Search the DB and retrieve documents.
    results = vector_index.retrieve(query_text)
    # Apply reranking to the retrieved results.
    reranked_results = reranker.rerank(results, query_text)
    return reranked_results

def generate_prompt(reranked_results, query_text):
    documents_details = "\n\n---\n\n".join([f"Document: {doc.page_content}\nScore: {score}" for doc, score in reranked_results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    return prompt_template.format(documents=documents_details, question=query_text)

def generate_response(prompt):
    model = Ollama(model="phi3")
    return model.invoke(prompt)

if __name__ == "__main__":
    main()
