from pathlib import Path

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings

from .embedder import get_embedder
from .knowledge_base import CONCEPT_DOCS

PERSIST_DIR = Path(__file__).resolve().parent / "chroma_db"


class SentenceTransformerEmbedding(EmbeddingFunction):
    """Wraps the SentenceTransformer model from embedder.py for ChromaDB."""

    def __init__(self):
        self._model = get_embedder()

    def __call__(self, input: Documents) -> Embeddings:
        return self._model.encode(list(input)).tolist()

    @staticmethod
    def name() -> str:
        return "all-MiniLM-L6-v2"

    def get_config(self):
        return {}

    @staticmethod
    def build_from_config(config):
        return SentenceTransformerEmbedding()


def get_vectorstore():
    """
    Creates or loads a persistent ChromaDB collection for financial
    explanation documents, seeding it with the concept documents from
    knowledge_base.py on first use.
    """
    client = chromadb.PersistentClient(path=str(PERSIST_DIR))

    collection = client.get_or_create_collection(
        name="fintech_docs",
        embedding_function=SentenceTransformerEmbedding(),
    )

    if collection.count() == 0:
        collection.add(
            ids=[doc["id"] for doc in CONCEPT_DOCS],
            documents=[doc["text"] for doc in CONCEPT_DOCS],
        )

    return collection


def retrieve_context(collection, query_text: str, k: int = 3) -> list[str]:
    """
    Returns the top-k concept documents most relevant to query_text.
    """
    if not query_text.strip():
        return []

    results = collection.query(query_texts=[query_text], n_results=k)
    return results["documents"][0] if results["documents"] else []
