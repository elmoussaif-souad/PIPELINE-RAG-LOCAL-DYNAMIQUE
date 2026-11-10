import argparse
import os
import shutil

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_community.vectorstores import Chroma

from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"
DATA_PATH = "data"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Vider la base avant.")
    args = parser.parse_args()

    if args.reset:
        print("Suppression de la base existante...")
        if os.path.exists(CHROMA_PATH):
            shutil.rmtree(CHROMA_PATH)

    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)


def load_documents():
    loader = PyPDFDirectoryLoader(DATA_PATH)
    return loader.load()


def split_documents(documents: list[Document]):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
    )
    return splitter.split_documents(documents)


def add_to_chroma(chunks: list[Document]):
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function(),
    )

    chunks = calculate_chunk_ids(chunks)

    existing = db.get(include=[])
    existing_ids = set(existing["ids"])
    print(f"Documents déjà dans la base : {len(existing_ids)}")

    new_chunks = [c for c in chunks if c.metadata["id"] not in existing_ids]

    if new_chunks:
        print(f"Ajout de {len(new_chunks)} nouveaux morceaux...")
        db.add_documents(new_chunks, ids=[c.metadata["id"] for c in new_chunks])
    else:
        print("Aucun nouveau document à ajouter.")


def calculate_chunk_ids(chunks):
    """Crée un id unique par morceau : fichier:page:index"""
    last_page_id = None
    current_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_index += 1
        else:
            current_index = 0

        chunk.metadata["id"] = f"{current_page_id}:{current_index}"
        last_page_id = current_page_id

    return chunks


if __name__ == "__main__":
    main()