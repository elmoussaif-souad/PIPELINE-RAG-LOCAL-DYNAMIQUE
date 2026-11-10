import requests
import streamlit as st
import rag_core as rag

st.set_page_config(page_title="RAG local dynamique", page_icon="📚")
st.title("📚 RAG local dynamique")


def get_ollama_models():
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        names = [m["name"] for m in r.json()["models"]]
        return [n for n in names if "embed" not in n]
    except Exception:
        return []


if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------- Barre latérale ----------
with st.sidebar:
    st.header("Documents")
    files = st.file_uploader(
        "Ajouter des documents",
        type=rag.SUPPORTED_TYPES,
        accept_multiple_files=True,
    )
    if st.button("Indexer les fichiers", disabled=not files):
        for f in files:
            with st.spinner(f"Indexation de {f.name}..."):
                path = rag.save_upload(f.name, f.getvalue())
                try:
                    n = rag.index_file(path)
                    st.success(f"{f.name} : {n} morceaux ajoutés")
                except Exception as e:
                    st.error(f"{f.name} : {e}")

    indexed = rag.list_indexed_files()
    st.markdown("**Documents indexés :**")
    if indexed:
        for name in indexed:
            st.write(f"• {name}")
    else:
        st.caption("Aucun document pour l'instant.")

    st.divider()
    models = get_ollama_models()
    if models:
        model = st.selectbox("Modèle de réponse", models)
    else:
        model = st.text_input("Modèle de réponse", value="mistral")
        st.warning("Ollama ne répond pas. Vérifie qu'il est lancé.")

    if st.button("Effacer la conversation"):
        st.session_state.messages = []
        st.rerun()
    if st.button("Réinitialiser la base"):
        rag.reset_database()
        st.session_state.messages = []
        st.rerun()

# ---------- Historique affiché ----------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            st.caption("Sources : " + " · ".join(m["sources"]))

# ---------- Nouvelle question ----------
if question := st.chat_input("Pose ta question sur tes documents"):
    history = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if not rag.list_indexed_files():
            answer = "Ajoute d'abord un document dans la barre latérale."
            sources = []
            st.markdown(answer)
        else:
            results = rag.retrieve(question)
            answer = st.write_stream(rag.stream_answer(question, history, results, model))
            sources = rag.format_sources(results)
            st.caption("Sources : " + " · ".join(sources))

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )