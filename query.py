#from env import pinecone_api_key,cohere_api_key
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms.ollama import Ollama
# from langchain_groq import ChatGroq
# from get_embedding_function import get_embedding_function
from langchain.schema.document import Document
import tiktoken
import cohere
from pinecone import Pinecone, ServerlessSpec
import requests
import pandas as pd
from langchain_community.embeddings import OllamaEmbeddings

enc = tiktoken.get_encoding("cl100k_base")


def get_embedding_function():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings


url = "https://cea.nic.in/api/psp_energy.php"


class Document:
    def __init__(self, content, metadata=None):
        self.page_content = content
        self.metadata = metadata if metadata is not None else {}


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

index = "1011"  #needs to be changed
pinecone_api_key = "0b430136-866c-419e-b6c2-f073c167026f"
pc = Pinecone(api_key=pinecone_api_key)
pc.create_index(
    name="index",
    dimension=768,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)

index = pc.Index("index")

for i in range(len(texts)):
    # Ensure the text value is a string
    text_value = str(texts[i])
    # Create the metadata dictionary with 'text' key having a string value
    metadata = {"text": text_value}
    # Insert into Pinecone
    index.upsert([(str(i), r1[i], metadata)])

print("done upsetting...")


def get_query_embedding(text):
    embedding = embeddings.embed_query(text)
    return embedding


embeddings = get_embedding_function()
co = cohere.Client(api_key="5kxf79VQjj9SjtqtgZ2G4Ar1rkSOepvIeG6I1pOD")


def query(query_text):
    question_embedding = get_query_embedding(query_text)

    query_result = index.query(vector=question_embedding, top_k=5, include_metadata=True)
    similar_texts = []
    # Extract metadata from query result
    docs = {x["metadata"]['text']: i for i, x in enumerate(query_result["matches"])}

    # Rerank the documents
    rerank_docs = co.rerank(
        model="rerank-english-v3.0",
        query=query_text,
        documents=list(docs.keys()),
        top_n=5,
        return_documents=True
    )

    # Extract reranked documents
    reranked_texts = [doc.document.text for doc in rerank_docs.results]
    print(query_text)
    print(reranked_texts)
    context_text = " ".join(reranked_texts)

    PROMPT_TEMPLATE = """
    Carefully read the following context:
    {context}
    ---
    Now, respond to the question based solely on the context provided above: {question}

    Instructions for Answering:
    - Utilize ONLY the provided context to formulate your response. Refrain from assumptions or external knowledge.
    - If the context directly supports an answer, provide that answer clearly and concisely.
    - Do not alter the answers by rounding off,take the numerical values directly from the knowledge.
    - If the context does not contain enough information to answer the question, state: 'The answer cannot be determined from the provided context.'
    - If no context is provided, respond with: 'No context is available to answer the question.'

    Please adhere to these guidelines rigorously.
    """
    prompt = PROMPT_TEMPLATE.format(context=context_text, question=query_text)
    model = Ollama(model="wizardlm2")
    response_text = model.invoke(prompt)
    return response_text


print(query("What is the energy requirement in Delhi in Mar-20 ?"))
