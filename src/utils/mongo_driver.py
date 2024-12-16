import time

from pymongo import MongoClient

from utils.generate_embeddings import get_embedding


class MongoDriver:
    client: MongoClient
    db_name: str
    vector_search_index_name: str

    def __init__(
        self,
        uri: str,
        appname="devrel.workshop.rag",
        db_name="mongodb_rag_lab",
        vector_search_index_name="vector_index",
    ) -> None:
        self.client = MongoClient(uri, appname=appname)
        self.db_name = db_name
        self.vector_search_index_name = vector_search_index_name

    def ingest_data(self, collection_name: str, embedded_docs: list) -> None:
        collection = self.client[self.db_name][collection_name]
        collection.delete_many({})
        collection.insert_many(embedded_docs)
        print(
            f"Ingested {collection.count_documents({})} documents into the {collection_name} collection."
        )

    def create_vector_search_index(self, collection_name: str, index_name: str) -> None:
        # Create vector index definition specifying:
        # path: Path to the embeddings field
        # numDimensions: Number of embedding dimensions- depends on the embedding model used
        # similarity: Similarity metric. One of cosine, euclidean, dotProduct.
        model = {
            "name": index_name,
            "type": "vectorSearch",
            "definition": {
                "fields": [
                    {
                        "type": "vector",
                        "path": "embedding",
                        "numDimensions": 384,
                        "similarity": "cosine",
                    }
                ]
            },
        }

        self.client[self.db_name][collection_name].create_search_index(model=model)
        self.vector_search_index_name = index_name

    def update_search_index(self, collection_name: str) -> None:
        model = {
            "name": self.vector_search_index_name,
            "type": "vectorSearch",
            "definition": {
                "fields": [
                    {
                        "type": "vector",
                        "path": "embedding",
                        "numDimensions": 384,
                        "similarity": "cosine",
                    },
                    {"type": "filter", "path": "metadata.contentType"},
                ]
            },
        }
        self.client[self.db_name][collection_name].update_search_index(
            name=self.vector_search_index_name, definition=model["definition"]
        )

    def update_search_index_2(self, collection_name: str) -> None:
        model = {
            "name": self.vector_search_index_name,
            "type": "vectorSearch",
            "definition": {
                "fields": [
                    {
                        "type": "vector",
                        "path": "embedding",
                        "numDimensions": 384,
                        "similarity": "cosine",
                    },
                    {"type": "filter", "path": "metadata.contentType"},
                    {"type": "filter", "path": "updated"},
                ]
            },
        }
        self.client[self.db_name][collection_name].update_search_index(
            name=self.vector_search_index_name, definition=model["definition"]
        )
        self.__wait_for_index(collection_name)

    def __wait_for_index(self, collection_name):
        # Poll for readiness
        timeout = 60  # seconds
        start_time = time.time()
        while not self.__is_index_ready(collection_name, self.vector_search_index_name):
            if time.time() - start_time > timeout:
                raise TimeoutError(
                    f"Index {self.vector_search_index_name} did not become ready within {timeout} seconds."
                )
            time.sleep(5)  # Wait for a few seconds before checking again

    # Function to check if the index is ready
    def __is_index_ready(self, collection_name, index_name):
        indexes = self.client[self.db_name][collection_name].list_search_indexes()
        for index in indexes:
            if index["name"] == index_name and index["status"] == "READY":
                return True
        return False

    def vector_search(self, collection_name: str, user_query: str):
        """
        Retrieve relevant documents for a user query using vector search.

        Args:
        user_query (str): The user's query string.

        Returns:
        list: A list of matching documents.
        """

        # Generate embedding for the `user_query` using the `get_embedding` function defined in Step 5
        query_embedding = get_embedding(user_query)

        # Define an aggregation pipeline consisting of a $vectorSearch stage, followed by a $project stage
        # Set the number of candidates to 150 and only return the top 5 documents from the vector search
        # In the $project stage, exclude the `_id` field and include only the `body` field and `vectorSearchScore`
        # NOTE: Use variables defined previously for the `index`, `queryVector` and `path` fields in the $vectorSearch stage
        pipeline = [
            {
                "$vectorSearch": {
                    "index": self.vector_search_index_name,
                    "queryVector": query_embedding,
                    "path": "embedding",
                    "numCandidates": 150,
                    "limit": 5,
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "body": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]

        # Execute the aggregation `pipeline` and store the results in `results`
        results = self.client[self.db_name][collection_name].aggregate(pipeline)
        return list(results)
