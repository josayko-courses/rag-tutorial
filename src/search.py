import json
import os

from dotenv import load_dotenv

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


if __name__ == "__main__":
    search()
