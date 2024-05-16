from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain_groq import ChatGroq
from get_embedding_function import get_embedding_function
from langchain.schema.document import Document
from env import cohere_api_key
import tiktoken
import cohere
from pinecone import Pinecone, ServerlessSpec
enc = tiktoken.get_encoding("cl100k_base")

class Document:
    def __init__(self, content, metadata=None):
        self.page_content = content
        self.metadata = metadata if metadata is not None else {}

import requests
import pandas as pd

url = "https://cea.nic.in/api/psp_energy.php"

def load_documents(url):
    response = requests.get(url)
    data = response.json()
    df = pd.json_normalize(data, record_path=['2019-2020'])

    documents = []
    for index, row in df.iterrows():
        content = f"In {row['Month']} {row['State'].strip()}, the energy requirement was {row['energy_requirement']} and the energy availability was {row['energy_availability']}."
        metadata = {"State": row['State'].strip(), "Month": row['Month']}
        documents.append(Document(content, metadata))
    return documents

def encoded_length(text):
    """Encode text and return its length."""
    return len(enc.encode(text))

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=395,
        chunk_overlap=100,
        length_function=encoded_length,  # Reference to the correct function
        separators=["\n", ".", "!", "?", ",", ";", "\n\n", "\n", " ", ""],
        is_separator_regex=False
    )
    processed_docs = []
    for doc in documents:
        chunks = text_splitter.split_documents([doc])
        for index, chunk in enumerate(chunks):
            if index > 0:
                chunk.metadata = {
                    **chunk.metadata,  # Ensure you are modifying chunk's metadata, not doc's metadata directly
                    "additional_info": "Continuation from previous month"
                }
            processed_docs.append(chunk)

    return processed_docs

llm_local = Ollama(model="wizardlm2")

embeddings = get_embedding_function()
documents = load_documents(url)
texts = split_documents(documents)

r1 = embeddings.embed_documents(texts)
pinecone_api_key="0b430136-866c-419e-b6c2-f073c167026f"
pc = Pinecone(api_key=pinecone_api_key)
pc.create_index(
    name="test1",
    dimension=768,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)

index = pc.Index("test121")

for i in range(len(texts)):
    # Ensure the text value is a string
    text_value = str(texts[i])
    # Create the metadata dictionary with 'text' key having a string value
    metadata = {"text": text_value}
    # Insert into Pinecone
    index.upsert([(str(i), r1[i], metadata)])

print("done upserting...")