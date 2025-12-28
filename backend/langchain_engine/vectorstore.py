import chromadb
from chromadb.utils import embedding_functions
from .embedder import get_embedder

# Load embedding model from embedder.py
embedder = get_embedder()

def get_vectorstore():
    """
    Creates or loads a ChromaDB collection for financial explanation documents.
    Embeddings are generated using the SentenceTransformer model.
    """
    client = chromadb.Client()

    # Convert embedding method to a simple lambda wrapper for Chroma
    def embed_fn(texts):
        return embedder.encode(texts).tolist()

    # Create or load a persistent collection
    collection = client.get_or_create_collection(
        name="fintech_docs",
        embedding_function=embed_fn
    )

    return collection
