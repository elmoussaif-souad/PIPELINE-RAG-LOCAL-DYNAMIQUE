# Mon RAG local

Assistant documentaire **100 % local** basé sur le RAG (*Retrieval-Augmented Generation*) avec Ollama, LangChain, ChromaDB et Streamlit. Les documents ne quittent jamais votre machine : la vectorisation et la génération de réponses sont exécutées localement.

## Fonctionnalités

- Interface de chat web (Streamlit)
- Téléversement et indexation de documents en direct
- Formats pris en charge : PDF, DOCX, TXT, MD, CSV
- Indexation incrémentale : seuls les nouveaux blocs de texte sont vectorisés
- Historique de conversation (6 derniers messages)
- Réponses en streaming, mot par mot
- Sources citées avec le nom du fichier et le numéro de page
- Choix du modèle Ollama depuis l'interface
- Utilisation possible en terminal

## Architecture

```text
 [ Interface Streamlit (app.py) ]   ou   [ Dossier data/ ]
        |  PDF, DOCX, TXT, MD, CSV
        v
 Lecteurs de documents (rag_core.py)
        |
        v   blocs de 800 caractères, chevauchement de 80
 RecursiveCharacterTextSplitter (LangChain)
        |
        v   ID unique = fichier : page : index
 Comparaison avec les IDs déjà présents dans ChromaDB
        |
        v   seuls les nouveaux blocs sont vectorisés
 Ollama (nomic-embed-text)  ->  vecteurs
        |
        v
 Base vectorielle ChromaDB (dossier chroma/)
        |
 =======|======================================================
        v   recherche des 5 blocs les plus proches
 Question de l'utilisateur
        |
        v   contexte + historique + question
 Prompt Template (LangChain, en français)
        |
        v   inférence locale
 Ollama (mistral ou autre modèle)
        |
        v   streaming
 Réponse + sources (fichier, page)
```

## Structure du projet

| Fichier | Rôle |
|---|---|
| `app.py` | Interface web Streamlit (chat, upload, choix du modèle) |
| `rag_core.py` | Logique commune : lecture, découpage, indexation, recherche, génération |
| `get_embedding_function.py` | Configuration du modèle d'embeddings |
| `populate_database.py` | Construction de la base depuis `data/` (terminal) |
| `query_data.py` | Questions en ligne de commande |
| `requirements.txt` | Dépendances Python |
| `data/` | Documents à indexer (non versionné) |
| `chroma/` | Base vectorielle locale (non versionnée) |

## Installation

1. Installer [Ollama](https://ollama.com), puis télécharger les modèles :
   ```bash
   ollama pull nomic-embed-text
   ollama pull mistral
   ```
2. Créer l'environnement virtuel : `python -m venv venv`
3. L'activer : `venv\Scripts\activate`
4. Installer les dépendances : `pip install -r requirements.txt`

## Utilisation

### Interface web (recommandé)

Lancer Ollama, puis :

```bash
streamlit run app.py
```

L'application s'ouvre sur http://localhost:8501. Déposez vos documents dans la barre latérale, cliquez sur **Indexer les fichiers**, puis posez vos questions.

### Terminal

```bash
python populate_database.py
python query_data.py "ta question"
```

Les documents sont à placer au préalable dans le dossier `data/`.

## Dépannage

| Problème | Solution |
|---|---|
| `ModuleNotFoundError` | Activer le venv et relancer `pip install -r requirements.txt` |
| Erreur de connexion ou réponses vides | Vérifier qu'Ollama est lancé (`ollama list`) |
| `model not found` | Télécharger le modèle (`ollama pull mistral`) |
| Conflit `numpy` / `pandas` | Utiliser `numpy<2` et `pandas<3` (`pip install "pandas<3" "numpy<2"`) |
| Réponses vides sur un PDF | Le PDF est probablement scanné, sans texte sélectionnable |