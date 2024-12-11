from typing import Dict, List

from langchain.text_splitter import RecursiveCharacterTextSplitter


# https://python.langchain.com/docs/how_to/split_by_token/
# For text data, you typically want to keep 1-2 paragraphs (~200 tokens) in a single chunk
# Chunk overlap of 15-20% of the chunk size is recommended
def create_text_splitter():
    # Separators to split on
    separators = ["\n\n", "\n", " ", "", "#", "##", "###"]
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-4", separators=separators, chunk_size=200, chunk_overlap=30
    )
    return text_splitter


def get_chunks(doc: Dict, text_field: str) -> List[Dict]:
    """
    Chunk up a document.

    Args:
        doc (Dict): Parent document to generate chunks from.
        text_field (str): Text field to chunk.

    Returns:
        List[Dict]: List of chunked documents.
    """
    # Extract the field to chunk from `doc`
    text = doc[text_field]
    # NOTE: `text` is a string
    text_splitter = create_text_splitter()
    chunks = text_splitter.split_text(text)

    # Iterate through `chunks` and for each chunk:
    # 1. Create a shallow copy of `doc`, call it `temp`
    # 2. Set the `text_field` field in `temp` to the content of the chunk
    # 3. Append `temp` to `chunked_data`
    chunked_data = []
    for chunk in chunks:
        temp = doc.copy()
        temp[text_field] = chunk
        chunked_data.append(temp)

    return chunked_data
