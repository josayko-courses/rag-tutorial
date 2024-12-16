from typing import List

from sentence_transformers import SentenceTransformer


# Load the `gte-small` model using the Sentence Transformers library
# https://huggingface.co/thenlper/gte-small#usage
def create_embedding_model():
    return SentenceTransformer("thenlper/gte-small")


def get_embedding(text: str) -> List[float]:
    """
    Generate the embedding for a piece of text.

    Args:
        text (str): Text to embed.

    Returns:
        List[float]: Embedding of the text as a list.
    """
    embedding_model = create_embedding_model()
    embedding = embedding_model.encode(text)
    return embedding.tolist()
