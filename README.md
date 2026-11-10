# Mon RAG local

Projet RAG (Retrieval-Augmented Generation) en local avec Ollama, LangChain et ChromaDB.
Basé sur le tutoriel de pixegami : https://github.com/pixegami/rag-tutorial-v2

## Installation

1. Installer Ollama et télécharger les modèles utilisés
2. Créer l'environnement virtuel : `python -m venv venv`
3. L'activer : `venv\Scripts\activate`
4. Installer les dépendances : `pip install -r requirements.txt`

## Utilisation

1. Mettre les PDF dans le dossier `data/`
2. Construire la base : `python populate_database.py`
3. Poser une question : `python query_data.py "ta question"`