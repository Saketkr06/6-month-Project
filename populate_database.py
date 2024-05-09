import argparse
import os
import shutil
import requests
import pandas as pd
from langchain.document_loaders.pdf import PyPDFDirectoryLoader  # Assume this is for future use
from langchain.schema.document import Document
from langchain.vectorstores.chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from get_embedding_function import get_embedding_function
import tiktoken
CHROMA_PATH = "chroma"
enc = tiktoken.get_encoding("cl100k_base")

class Document:
    def __init__(self, content, metadata=None):
        self.page_content = content
        self.metadata = metadata if metadata is not None else {}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()

    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)

def load_documents():
    url = "https://cea.nic.in/api/psp_energy.php"
    response = requests.get(url)
    data = response.json()
    df = pd.json_normalize(data, record_path=['2019-2020'])

    documents = []
    for index, row in df.iterrows():
        content = f"In {row['Month']} {row['State'].strip()}, the energy requirement was {row['energy_requirement']} and the energy availability was {row['energy_availability']}."
        metadata = {"State": row['State'].strip(), "Month": row['Month']}
        documents.append(Document(content, metadata))

    return documents  # This should be outside the for-loop


def length_function(text: str) -> int:
    return len(enc.encode(text))

def encoded_length(text):
    """Encode text and return its length."""
    return len(enc.encode(text))

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=256,
        chunk_overlap=50,
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
def add_to_chroma(chunks):
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())
    # Fetch existing documents' IDs from the database
    existing_documents = db.get()
    print("Existing documents:", existing_documents)  # This should help verify the structure
    # Extract the list of IDs from the 'ids' key in the dictionary returned by db.get()
    existing_ids = set(existing_documents['ids'])  # Adjusting to extract 'ids' from the dictionary
    # Prepare new chunks to add to the database by filtering out those with existing IDs
    new_chunks = [chunk for chunk in chunks if getattr(chunk, 'metadata', {}).get("id", None) not in existing_ids]
    if new_chunks:
        db.add_documents(new_chunks)  # Ensure that add_documents method is compatible with your data structure
        db.persist()
        print(f"👉 Adding new documents: {len(new_chunks)}")
    else:
        print("✅ No new documents to add")

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print("Database cleared.")

if __name__ == "__main__":
    main()