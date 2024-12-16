import json
import os

from dotenv import load_dotenv

from utils.generate_embeddings import get_embedding
from utils.mongo_driver import MongoDriver

load_dotenv()


def search():
    MONGODB_URI = os.getenv("MONGODB_URI")
    assert isinstance(MONGODB_URI, str)
    mongodb_driver = MongoDriver(MONGODB_URI)
    mongodb_client = mongodb_driver.client
    # Check the connection to the server
    pong = mongodb_client.admin.command("ping")
    if pong["ok"] == 1:
        print("MongoDB connection successful")

    COLLECTION_NAME = "knowledge_base"

    # Run vector search queries
    user_query_1 = "What is MongoDB Atlas Search?"
    user_query_2 = "What are triggers in MongoDB Atlas?"
    result_1 = mongodb_driver.vector_search(COLLECTION_NAME, user_query_1)
    result_2 = mongodb_driver.vector_search(COLLECTION_NAME, user_query_2)

    print(user_query_1, json.dumps(result_1, indent=2))
    print(user_query_2, json.dumps(result_2, indent=2))

    # Modify the vector search index `model` to include the `metadata.contentType` field as a `filter` field
    # https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-type/#about-the-filter-type
    mongodb_driver.update_search_index(COLLECTION_NAME)
    # Embed the user query
    query_embedding = get_embedding(user_query_1)
    # Modify the $vectorSearch stage of the aggregation pipeline defined previously to include a filter for documents where the `metadata.contentType` field has the value "Video"
    pipeline = [
        {
            "$vectorSearch": {
                "index": mongodb_driver.vector_search_index_name,
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": 150,
                "limit": 5,
                "filter": {"metadata.contentType": "Video"},
            }
        },
        {"$project": {"_id": 0, "body": 1, "score": {"$meta": "vectorSearchScore"}}},
    ]
    # Execute the aggregation pipeline and view the results
    results = mongodb_client[mongodb_driver.db_name][COLLECTION_NAME].aggregate(
        pipeline
    )
    print(user_query_1, json.dumps(list(results), indent=2))

    # Update vector search index
    mongodb_driver.update_search_index_2(COLLECTION_NAME)
    # Embed the user query
    query_embedding = get_embedding(user_query_2)
    # Modify the $vectorSearch stage of the aggregation pipeline defined previously to include a filter for documents where
    # the `metadata.contentType` field has the value "Tutorial"
    # AND
    # the `updated` field is greater than or equal to "2024-05-19"
    pipeline_2 = [
        {
            "$vectorSearch": {
                "index": mongodb_driver.vector_search_index_name,
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": 150,
                "limit": 5,
                "filter": {
                    "$and": [
                        {"metadata.contentType": "Tutorial"},
                        {"updated": {"$gte": "2024-05-19"}},
                    ]
                },
            }
        },
        {
            "$project": {
                "_id": 0,
                "body": 1,
                "updated": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]
    # Execute the aggregation pipeline and view the results
    results = mongodb_client[mongodb_driver.db_name][COLLECTION_NAME].aggregate(
        pipeline_2
    )
    print(user_query_2, json.dumps(list(results), indent=2))


if __name__ == "__main__":
    search()
