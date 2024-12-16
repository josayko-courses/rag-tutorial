import json
import os

from dotenv import load_dotenv
from pymongo import MongoClient
from tqdm import tqdm

from utils.chunk_data import get_chunks
from utils.generate_embeddings import get_embedding
from utils.load_dataset import load_dataset

load_dotenv()


def main():
    # Initialize a MongoDB Python client
    MONGODB_URI = os.getenv("MONGODB_URI")
    mongodb_client = MongoClient(MONGODB_URI, appname="devrel.workshop.rag")
    # Check the connection to the server
    pong = mongodb_client.admin.command("ping")
    if pong["ok"] == 1:
        print("MongoDB connection successful")

    docs = load_dataset()
    # Check the number of documents in the dataset
    print(f"Number of documents: {len(docs)}")
    # Preview a document
    print(f"preview: {json.dumps(docs[0], indent=2)}")
    split_docs = []
    # Iterate through `docs`, use the `get_chunks` function to chunk up the documents based on the "body" field, and add the list of chunked documents to `split_docs` initialized above.
    for doc in docs:
        chunks = get_chunks(doc, "body")
        split_docs.extend(chunks)

    # Check that the length of the list of chunked documents is greater than the length of `docs`
    print(f"Length of split_docs: {len(split_docs)}")

    # Preview one of the items in split_docs- ensure that it is a Python dictionary
    print(f"preview doc: {json.dumps(split_docs[0], indent=2)}")

    embedded_docs = []
    for doc in tqdm(split_docs):
        doc["embedding"] = get_embedding(doc["body"])
        embedded_docs.append(doc)

    # Check that the length of `embedded_docs` is the same as that of `split_docs`
    print(f"Length of embedded_docs: {len(embedded_docs)}")

    # Ingest data into MongoDB
    DB_NAME = "mongodb_rag_lab"
    COLLECTION_NAME = "knowledge_base"
    ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index"

    collection = mongodb_client[DB_NAME][COLLECTION_NAME]
    collection.delete_many({})
    collection.insert_many(embedded_docs)
    print(
        f"Ingested {collection.count_documents({})} documents into the {COLLECTION_NAME} collection."
    )


if __name__ == "__main__":
    main()
