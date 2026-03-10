"""
Responsible for converting text into vector embeddings using sentence-transformers.
"""

from sentence_transformers import SentenceTransformer

# Load model once at module level to avoid reloading on every call
_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_text(text: str) -> list[float]:
    """
    Converts a text string into a vector embedding.

    Args:
        text: The input text to embed.

    Returns:
        A list of floats representing the embedding vector.
    """
    embedding = _model.encode(text, convert_to_numpy=True)
    return embedding.tolist()
    