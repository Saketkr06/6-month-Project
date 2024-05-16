import streamlit as st
import altair as alt
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms.ollama import Ollama
from langchain.schema.document import Document
import tiktoken
import cohere
from qdrant_client import QdrantClient
import requests
import pandas as pd
from langchain_community.embeddings import OllamaEmbeddings
import os
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize the necessary components
def get_embedding_function():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings

enc = tiktoken.get_encoding("cl100k_base")
cohere_api_key = "5kxf79VQjj9SjtqtgZ2G4Ar1rkSOepvIeG6I1pOD"
qdrant_url = "https://bef959b7-fc25-4d76-8716-28b854a86ad7.us-east4-0.gcp.cloud.qdrant.io:6333"
qdrant_api_key = "K3-Ld3ura-Kvz-3LzmaASYwDSPEYT30MhoLHMZmqRFPuyv_2724xcA"
co = cohere.Client(api_key=cohere_api_key)

class Document:
    def __init__(self, content, metadata=None):
        self.page_content = content
        self.metadata = metadata if metadata is not None else {}

def load_documents(url):
    logging.info("Loading documents from URL")
    response = requests.get(url)
    data = response.json()
    df = pd.json_normalize(data, record_path=['2020-2021'])

    documents = []
    for index, row in df.iterrows():
        content = f"In {row['Month']} {row['State'].strip()}, the energy requirement was {row['energy_requirement']} and the energy availability was {row['energy_availability']}."
        metadata = {"State": row['State'].strip(), "Month": row['Month']}
        documents.append(Document(content, metadata))
    logging.info(f"Loaded {len(documents)} documents")
    return documents, df

def encoded_length(text):
    """Encode text and return its length."""
    return len(enc.encode(text))

def split_documents(documents):
    logging.info("Splitting documents into chunks")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=395,
        chunk_overlap=100,
        length_function=encoded_length,
        separators=["\n", ".", "!", "?", ",", ";", "\n\n", "\n", " ", ""],
        is_separator_regex=False
    )
    processed_docs = []
    for doc in documents:
        chunks = text_splitter.split_documents([doc])
        for index, chunk in enumerate(chunks):
            if index > 0:
                chunk.metadata = {
                    **chunk.metadata,
                    "additional_info": "Continuation from previous month"
                }
            processed_docs.append(chunk)
    logging.info(f"Split into {len(processed_docs)} chunks")
    return processed_docs

llm_local = Ollama(model="wizardlm2")

embeddings = get_embedding_function()
url = "https://cea.nic.in/api/psp_energy.php"
documents, df = load_documents(url)
texts = split_documents(documents)

# Initialize Qdrant and create collection if it doesn't exist
qdrant_client = QdrantClient(
    url=qdrant_url,
    api_key=qdrant_api_key,
)

collection_name = "energy_data"

# Check if the collection exists before creating it
if not qdrant_client.get_collection(collection_name):
    logging.info(f"Creating collection: {collection_name}")
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config={"size": 768, "distance": "Cosine"}
    )

# Check if embeddings have been upserted
embedding_file = "embeddings.json"
if os.path.exists(embedding_file):
    logging.info("Loading embeddings from file")
    # Load embeddings and metadata from file
    with open(embedding_file, "r") as f:
        data = json.load(f)
    r1 = [item["embedding"] for item in data]
    metadata_list = [item["metadata"] for item in data]
else:
    logging.info("Generating embeddings and metadata")
    # Generate embeddings and metadata
    r1 = embeddings.embed_documents(texts)
    metadata_list = [{"text": str(text)} for text in texts]
    # Save embeddings and metadata to file
    with open(embedding_file, "w") as f:
        json.dump([{"embedding": emb, "metadata": meta} for emb, meta in zip(r1, metadata_list)], f)

    # Upsert documents into Qdrant collection in batches
    batch_size = 100
    logging.info("Upserting embeddings into Qdrant")
    for i in range(0, len(r1), batch_size):
        batch_points = []
        for j, (emb, meta) in enumerate(zip(r1[i:i + batch_size], metadata_list[i:i + batch_size])):
            batch_points.append({
                "id": i + j,
                "vector": emb,
                "payload": meta
            })
        qdrant_client.upsert(collection_name=collection_name, points=batch_points)

logging.info("Done upserting")

def get_query_embedding(text):
    embedding = embeddings.embed_query(text)
    return embedding

def query(query_text):
    logging.info(f"Received query: {query_text}")
    question_embedding = get_query_embedding(query_text)

    search_result = qdrant_client.search(
        collection_name=collection_name,
        query_vector=question_embedding,
        limit=20,
        with_payload=True
    )
    similar_texts = []
    # Extract metadata from query result
    docs = {hit.payload["text"]: hit.id for hit in search_result}

    logging.info("Reranking documents")
    # Rerank the documents
    rerank_docs = co.rerank(
        model="rerank-english-v3.0",
        query=query_text,
        documents=[{"text": text} for text in docs.keys()],
        top_n=10,
        return_documents=True
    )

    # Extract reranked documents
    reranked_texts = [doc.document.text for doc in rerank_docs.results]
    context_text = " ".join(reranked_texts)

    PROMPT_TEMPLATE = """
    Carefully read the following context:
    {context}
    ---
    Now, respond to the question based solely on the context provided above: {question}

    Instructions for Answering:
    - Utilize ONLY the provided context to formulate your response. Refrain from assumptions or external knowledge.
    - If the context directly supports an answer, provide that answer clearly and concisely.
    - Do not alter the answers by rounding off, take the numerical values directly from the knowledge.
    - If the context does not contain enough information to answer the question, state: 'The answer cannot be determined from the provided context.'
    - If no context is provided, respond with: 'No context is available to answer the question.'

    Please adhere to these guidelines rigorously.
    """
    prompt = PROMPT_TEMPLATE.format(context=context_text, question=query_text)
    response_text = llm_local.invoke(prompt)
    return response_text, context_text

# Streamlit app
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f0f5;
    }
    .reportview-container {
        background: url("https://www.transparenttextures.com/patterns/brushed-alum.png");
        background-size: cover;
    }
    .title {
        font-size: 36px;
        font-weight: bold;
        text-align: center;
        color: #4B4B4B;
    }
    .input-box {
        text-align: center;
    }
    .submit-button {
        display: flex;
        justify-content: center;
        margin-top: 20px;
    }
    .response-text {
        background-color: #d4edda;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        color: #155724;
    }
    .context-text {
        background-color: #cce5ff;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        color: #004085;
    }
    .input-label {
        color: #FF0000;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="title">EnerSage</div>', unsafe_allow_html=True)

# Add an input box for user queries
st.markdown('<div class="input-box">', unsafe_allow_html=True)
st.markdown('<div class="input-label">Enter your query:</div>', unsafe_allow_html=True)
query_text = st.text_input("Enter your query:", label_visibility='collapsed')
st.markdown('</div>', unsafe_allow_html=True)

if st.button("Submit", key="submit-button"):
    response, context_text = query(query_text)
    st.markdown('<div class="response-text">### Response</div>', unsafe_allow_html=True)
    st.write(f'<div class="response-text">{response}</div>', unsafe_allow_html=True)
    st.markdown('<div class="context-text">### Context</div>', unsafe_allow_html=True)
    st.write(f'<div class="context-text">{context_text}</div>', unsafe_allow_html=True)
