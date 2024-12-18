import os
import time
from typing import Any

from dotenv import load_dotenv
from fireworks.client import Fireworks

from utils.mongo_driver import MongoDriver

load_dotenv()

# Initializing the Fireworks AI client and the model string
fw_client = Fireworks()
model = "accounts/fireworks/models/llama-v3-8b-instruct"

MONGODB_URI = os.getenv("MONGODB_URI")
assert isinstance(MONGODB_URI, str)
mongodb_driver = MongoDriver(MONGODB_URI)
mongodb_client = mongodb_driver.client
# Check the connection to the server
pong = mongodb_client.admin.command("ping")
if pong["ok"] == 1:
    print("MongoDB connection successful")

COLLECTION_NAME = "knowledge_base"


# Define a function to create the user prompt for our RAG application
def create_prompt(user_query: str) -> str:
    """
    Create a chat prompt that includes the user query and retrieved context.

    Args:
        user_query (str): The user's query string.

    Returns:
        str: The chat prompt string.
    """
    # Retrieve the most relevant documents for the `user_query` using the `vector_search` method
    context = mongodb_driver.vector_search(COLLECTION_NAME, user_query)
    # Join the retrieved documents into a single string, where each document is separated by two new lines ("\n\n")
    context = "\n\n".join([doc.get("body") for doc in context])
    # Prompt consisting of the question and relevant context to answer it
    prompt = f"Answer the question based only on the following context. If the context is empty, say I DON'T KNOW\n\nContext:\n{context}\n\nQuestion:{user_query}"
    return prompt


# Define a function to answer user queries using Fireworks' Chat Completion API
def generate_answer(user_query: str) -> None:
    """
    Generate an answer to the user query.

    Args:
        user_query (str): The user's query string.
    """
    # Use the `create_prompt` function above to create a chat prompt
    prompt = create_prompt(user_query)
    # Use the `prompt` created above to populate the `content` field in the chat message
    response: Any = fw_client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    # Run the `generate_answer` function with a user query
    user_query_1 = "What is MongoDB Atlas Search?"
    answer_1 = generate_answer(user_query_1)
    print(user_query_1)
    time.sleep(3)
    print(answer_1)
    user_query_2 = "What did I just ask you ?"
    answer_2 = generate_answer(user_query_1)
    print(user_query_2)
    time.sleep(3)
    print(answer_2)
    user_query_3 = "What time is it ?"
    answer_3 = generate_answer(user_query_3)
    print(user_query_3)
    time.sleep(3)
    print(answer_3)
    user_query_4 = "How Atlas Search maps documents?"
    answer_4 = generate_answer(user_query_4)
    print(user_query_4)
    time.sleep(3)
    print(answer_4)
