import os
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, CSVLoader, Docx2txtLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.llms.ollama import Ollama
from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"
DATA_PATH = "data"

# Un lecteur par format
LOADERS = {
    ".pdf": lambda p: PyPDFLoader(p),
    ".txt": lambda p: TextLoader(p, encoding="utf-8"),
    ".md": lambda p: TextLoader(p, encoding="utf-8"),
    ".csv": lambda p: CSVLoader(p, encoding="utf-8"),
    ".docx": lambda p: Docx2txtLoader(p),
}
SUPPORTED_TYPES = [ext.lstrip(".") for ext in LOADERS]

PROMPT_TEMPLATE = """Tu es un assistant qui répond en français, uniquement à partir du contexte fourni.
Si la réponse n'est pas dans le contexte, réponds exactement : "Je ne sais pas d'après les documents fournis."

Contexte :
{context}

Historique de la conversation :
{history}

Question : {question}

Réponse :"""


def get_db():
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function(),
    )


def save_upload(filename, content):
    """Enregistre un fichier envoyé depuis l'interface dans data/."""
    os.makedirs(DATA_PATH, exist_ok=True)
    path = os.path.join(DATA_PATH, os.path.basename(filename))
    with open(path, "wb") as f:
        f.write(content)
    return path


def load_file(path):
    ext = os.path.splitext(path)[1].lower()
    if ext not in LOADERS:
        raise ValueError(f"Format non supporté : {ext}")
    return LOADERS[ext](path).load()


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=80, length_function=len
    )
    return splitter.split_documents(documents)


def add_chunks(chunks):
    """Ajoute uniquement les nouveaux morceaux. Retourne leur nombre."""
    last_id, index = None, 0
    for chunk in chunks:
        source = os.path.basename(chunk.metadata.get("source", "inconnu"))
        page = chunk.metadata.get("page", chunk.metadata.get("row", 0))
        current = f"{source}:{page}"
        index = index + 1 if current == last_id else 0
        chunk.metadata["id"] = f"{current}:{index}"
        last_id = current

    db = get_db()
    existing = set(db.get(include=[])["ids"])
    new_chunks = [c for c in chunks if c.metadata["id"] not in existing]
    if new_chunks:
        db.add_documents(new_chunks, ids=[c.metadata["id"] for c in new_chunks])
    return len(new_chunks)


def index_file(path):
    return add_chunks(split_documents(load_file(path)))


def list_indexed_files():
    metas = get_db().get(include=["metadatas"])["metadatas"]
    return sorted({os.path.basename(m.get("source", "")) for m in metas if m})


def reset_database():
    """Vide la base et supprime les fichiers de data/."""
    get_db().delete_collection()
    if os.path.isdir(DATA_PATH):
        for name in os.listdir(DATA_PATH):
            os.remove(os.path.join(DATA_PATH, name))


def retrieve(question, k=5):
    return get_db().similarity_search_with_score(question, k=k)


def format_sources(results):
    sources = []
    for doc, _ in results:
        name = os.path.basename(doc.metadata.get("source", "inconnu"))
        if "page" in doc.metadata:
            label = f"{name} (page {doc.metadata['page'] + 1})"
        else:
            label = name
        if label not in sources:
            sources.append(label)
    return sources


def stream_answer(question, history, results, model):
    """Générateur : renvoie la réponse morceau par morceau."""
    context = "\n\n---\n\n".join(doc.page_content for doc, _ in results)
    hist = "\n".join(
        f"{'Utilisateur' if m['role'] == 'user' else 'Assistant'} : {m['content']}"
        for m in history[-6:]
    ) or "(aucun)"
    prompt = PROMPT_TEMPLATE.format(context=context, history=hist, question=question)
    llm = Ollama(model=model)
    for chunk in llm.stream(prompt):
        yield chunk
        