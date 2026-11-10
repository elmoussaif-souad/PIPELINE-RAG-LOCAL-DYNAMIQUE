import argparse

from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama

from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Réponds à la question en te basant uniquement sur le contexte suivant :

{context}

---

Question : {question}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="Ta question.")
    args = parser.parse_args()
    query_rag(args.query_text)


def query_rag(query_text: str):
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function(),
    )

    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join(doc.page_content for doc, _score in results)
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(
        context=context_text, question=query_text
    )

    model = Ollama(model="llama3.1:8b")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id") for doc, _score in results]
    print(f"\nRéponse : {response_text}\n")
    print(f"Sources : {sources}")


if __name__ == "__main__":
    main()