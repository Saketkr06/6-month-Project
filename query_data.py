import argparse
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"
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
def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)


def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)
    print(results)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # print(prompt)

    model = Ollama(model="wizardlm2")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}"
    print(formatted_response)
    return response_text

if __name__ == "__main__":
    main()