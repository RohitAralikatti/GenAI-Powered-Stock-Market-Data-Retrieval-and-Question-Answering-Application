from sentence_transformers import SentenceTransformer

# Load a small, fast embedding model for RAG
def get_embedder():
    """
    Returns a SentenceTransformer embedding model.
    This model is used to convert text into vector embeddings.
    """
    return SentenceTransformer("all-MiniLM-L6-v2")
